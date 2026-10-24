# verifiers/rules_analytical.py
import numpy as np


class CGMVerifierAnalytical:
    """
    Deterministic neuro-symbolic physiological verification engine for CGM forecasting.
    Performs hard kinematic pruning followed by continuous credit assignment J(tau)
    over metabolic flux, temporal continuity, curvature smoothness, and asymmetric clinical risk.
    """

    def __init__(
        self,
        max_roc_mmoll_per_min: float = 0.12,
        max_accel_mmoll_per_min2: float = 0.03,
        hypo_safety_floor: float = 3.9,
        interval_minutes: int = 5,
        default_isf: float = 2.5,
        default_cir: float = 10.0,
        lambda_metabolic: float = 1.5,
        lambda_continuity: float = 0.5,
        lambda_smoothness: float = 0.08,
        lambda_hypo_risk: float = 1.2,
    ):
        self.max_roc = max_roc_mmoll_per_min
        self.max_accel = max_accel_mmoll_per_min2
        self.hypo_floor = hypo_safety_floor
        self.interval_minutes = interval_minutes
        self.default_isf = default_isf
        self.default_cir = default_cir
        self.lambda_metabolic = lambda_metabolic
        self.lambda_continuity = lambda_continuity
        self.lambda_smoothness = lambda_smoothness
        self.lambda_hypo_risk = lambda_hypo_risk

    def score_candidates(
        self,
        candidates: np.ndarray,
        current_glucose: np.ndarray,
        recent_history: np.ndarray,
        active_iob: np.ndarray = None,
        active_cob: np.ndarray = None,
    ):
        """
        Args:
            candidates: (B, N, H) array of sampled trajectories in mmol/L
            current_glucose: (B,) current glucose at t_0 in mmol/L
            recent_history: (B, Lookback) historical CGM trajectory leading to t_0
            active_iob: (B,) active insulin on board in Units
            active_cob: (B,) active carbs on board in grams

        Returns:
            best_trajectories: (B, H) array of selected trajectories
            validity_mask: (B, N) boolean array indicating candidates passing hard kinematic checks
            total_costs: (B, N) total credit assignment scores J(tau)
        """
        B, N, H = candidates.shape
        dt = float(self.interval_minutes)

        if active_iob is None:
            active_iob = np.zeros(B, dtype=np.float32)
        if active_cob is None:
            active_cob = np.zeros(B, dtype=np.float32)

        # -------------------------------------------------------------
        # 1. Hard Deterministic Kinematic Pruning
        # -------------------------------------------------------------
        # Prepend current_glucose to evaluate the transition from t_0 to step 1
        cg_expanded = current_glucose[:, None, None]  # (B, 1, 1)
        full_trajectories = np.concatenate(
            [np.repeat(cg_expanded, N, axis=1), candidates], axis=2
        )  # (B, N, H + 1)

        diffs = np.diff(full_trajectories, axis=2)  # (B, N, H)
        roc = np.abs(diffs) / dt  # mmol/L per min

        accel = np.abs(np.diff(diffs, axis=2)) / (dt ** 2)  # (B, N, H - 1)

        valid_roc = np.all(roc <= self.max_roc, axis=2)  # (B, N)
        if accel.shape[2] > 0:
            valid_accel = np.all(accel <= self.max_accel, axis=2)  # (B, N)
        else:
            valid_accel = np.ones((B, N), dtype=bool)

        validity_mask = valid_roc & valid_accel  # (B, N)

        # -------------------------------------------------------------
        # 2. Metabolic Drift Loss (L_metabolic)
        # -------------------------------------------------------------
        tau_dia = 180.0  # Duration of insulin action (min)
        tau_carb = 120.0  # Duration of carb absorption (min)
        k_carb = self.default_isf / self.default_cir  # mmol/L per gram

        # Expected velocity in mmol/L per min: (B,)
        v_insulin = (active_iob * self.default_isf) / tau_dia
        v_carb = (active_cob * k_carb) / tau_carb
        expected_slope = (v_carb - v_insulin)[:, None, None]  # (B, 1, 1)

        cand_slopes = diffs / dt  # (B, N, H)
        loss_metabolic = np.mean((cand_slopes - expected_slope) ** 2, axis=2)  # (B, N)

        # -------------------------------------------------------------
        # 3. Temporal Continuity Loss (L_continuity)
        # -------------------------------------------------------------
        # Compute incoming velocity leading into t_0 from the last 2 context steps
        if recent_history.shape[1] >= 2:
            incoming_slope = (recent_history[:, -1] - recent_history[:, -2]) / dt  # (B,)
        else:
            incoming_slope = np.zeros(B, dtype=np.float32)

        incoming_slope_exp = incoming_slope[:, None]  # (B, 1)
        first_step_slope = diffs[:, :, 0] / dt  # (B, N)
        loss_continuity = (first_step_slope - incoming_slope_exp) ** 2  # (B, N)

        # -------------------------------------------------------------
        # 4. Curvature Smoothness Loss (L_smoothness)
        # -------------------------------------------------------------
        if full_trajectories.shape[2] >= 3:
            second_diffs = np.diff(full_trajectories, n=2, axis=2)  # (B, N, H - 1)
            loss_smoothness = np.mean(second_diffs ** 2, axis=2)  # (B, N)
        else:
            loss_smoothness = np.zeros((B, N), dtype=np.float32)

        # -------------------------------------------------------------
        # 5. Asymmetric Hypoglycemia Risk Loss (L_hypo_risk)
        # -------------------------------------------------------------
        # Penalizes trajectories that predict safe glucose (> 3.9 mmol/L)
        # when active IOB is elevated and context momentum is downward.
        cand_min_glucose = np.min(candidates, axis=2)  # (B, N)
        hypo_gap = np.maximum(0.0, cand_min_glucose - self.hypo_floor)  # (B, N)

        # Active IOB factor: triggers progressively above 0.5 Units
        iob_factor = np.maximum(0.0, active_iob[:, None] - 0.5)  # (B, 1)

        # Directional trigger: amplify penalty if glucose was already descending
        descending_momentum = np.maximum(0.0, -incoming_slope_exp)  # (B, 1)

        loss_hypo_risk = (
            iob_factor * (1.0 + 5.0 * descending_momentum) * (hypo_gap ** 2) / 10.0
        )  # (B, N)

        # -------------------------------------------------------------
        # 6. Total Cost J(tau) & Selection
        # -------------------------------------------------------------
        total_costs = (
            self.lambda_metabolic * loss_metabolic
            + self.lambda_continuity * loss_continuity
            + self.lambda_smoothness * loss_smoothness
            + self.lambda_hypo_risk * loss_hypo_risk
        )

        # Heavily penalize paths that violated hard kinematic constraints
        penalized_costs = np.where(validity_mask, total_costs, total_costs + 1e5)

        # Fallback handling: if a window has 0 valid candidates, fallback to unpenalized argmin
        has_valid = np.any(validity_mask, axis=1)  # (B,)
        best_indices = np.zeros(B, dtype=np.int64)

        for b in range(B):
            if has_valid[b]:
                best_indices[b] = np.argmin(penalized_costs[b])
            else:
                best_indices[b] = np.argmin(total_costs[b])

        # Gather winning trajectories
        batch_indices = np.arange(B)
        best_trajectories = candidates[batch_indices, best_indices, :]  # (B, H)

        return best_trajectories, validity_mask, total_costs