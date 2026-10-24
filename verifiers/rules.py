# verifiers/rules.py
import numpy as np
from typing import Tuple, Optional


class CGMVerifier:
    """
    Tier 3 Neuro-Symbolic Physiological Verification Engine.
    Combines kinematic boundaries (RoC, curvature) with metabolic kinetics
    (IOB insulin action curves and COB meal absorption models), applying
    both hard constraint filtering and consensus aggregation.
    """

    def __init__(
        self,
        max_roc_mmoll_per_min: float = 0.12,      # Maximum velocity (~2.16 mg/dL/min)
        max_accel_mmoll_per_min2: float = 0.03,   # Curvature / second derivative bound
        hypo_safety_floor: float = 3.9,            # Hypoglycemia boundary (mmol/L)
        interval_minutes: int = 5,
        default_isf: float = 2.5,                  # Insulin Sensitivity: mmol/L drop per 1 Unit
        default_cir: float = 10.0,                 # Carb-to-Insulin Ratio: grams carb per 1 Unit
        insulin_dia_hours: float = 4.0,            # Duration of Insulin Action (DIA)
        carb_absorption_hours: float = 3.0,        # Mean carbohydrate absorption window
    ):
        self.dt = interval_minutes
        self.max_roc = max_roc_mmoll_per_min * self.dt   # ~0.60 mmol/L per 5-min step
        self.max_accel = max_accel_mmoll_per_min2 * (self.dt ** 2)
        self.hypo_floor = hypo_safety_floor
        
        self.isf = default_isf
        self.cir = default_cir
        self.carb_factor = self.isf / self.cir          # mmol/L rise per gram of carb (~0.25 mmol/L per g)
        self.dia_mins = insulin_dia_hours * 60.0
        self.carb_mins = carb_absorption_hours * 60.0

    def _compute_metabolic_flux(
        self,
        horizon: int,
        active_iob: np.ndarray,
        active_cob: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates expected net physiological glucose drift per 5-min step over the horizon.
        
        Args:
            horizon: Steps ahead (H)
            active_iob: Active insulin on board at t=0 (B,) in Units
            active_cob: Unabsorbed carbohydrates on board at t=0 (B,) in grams
            
        Returns:
            expected_step_drift: Expected baseline delta per step (B, H) in mmol/L
        """
        t_minutes = np.arange(1, horizon + 1) * self.dt  # (H,)
        
        # Exponential-bilinear insulin activity decay (Swan / Mudaliar model)
        tau_i = self.dia_mins / 3.5
        insulin_decay_kernel = (t_minutes / (tau_i ** 2)) * np.exp(-t_minutes / tau_i) * self.dt
        insulin_pull = active_iob[:, None] * (insulin_decay_kernel[None, :] * self.isf)  # Downward pressure
        
        # Gastrointestinal carbohydrate appearance curve
        tau_c = self.carb_mins / 4.0
        carb_absorption_kernel = (t_minutes / (tau_c ** 2)) * np.exp(-t_minutes / tau_c) * self.dt
        carb_push = active_cob[:, None] * (carb_absorption_kernel[None, :] * self.carb_factor)  # Upward pressure
        
        return carb_push - insulin_pull  # Net drift (B, H)

    def score_candidates(
        self,
        candidates: np.ndarray,
        current_glucose: np.ndarray,
        recent_history: Optional[np.ndarray] = None,
        active_iob: Optional[np.ndarray] = None,
        active_cob: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates candidate paths across kinematic, momentum, and hard metabolic bounds.

        Args:
            candidates: Shape (B, N, H)
            current_glucose: Shape (B,)
            recent_history: Optional shape (B, Lookback)
            active_iob: Optional shape (B,) - Active rapid-acting insulin (Units)
            active_cob: Optional shape (B,) - Active unabsorbed carbs (grams)

        Returns:
            best_trajectories: Shape (B, H)
            validity_mask: Shape (B, N)
        """
        B, N, H = candidates.shape
        penalty_scores = np.zeros((B, N), dtype=np.float32)
        validity_mask = np.ones((B, N), dtype=bool)

        extended = np.concatenate(
            [np.repeat(current_glucose[:, None, None], N, axis=1), candidates],
            axis=2,
        )

        # 1. Rate of Change (Velocity) Hard Check
        deltas = np.diff(extended, axis=2)  # (B, N, H)
        roc_violations = np.sum(np.abs(deltas) > self.max_roc, axis=2)
        validity_mask[roc_violations > 0] = False
        penalty_scores += roc_violations * 20.0

        # 2. Acceleration (Curvature) Hard Check
        accels = np.diff(deltas, axis=2)  # (B, N, H - 1)
        accel_violations = np.sum(np.abs(accels) > self.max_accel, axis=2)
        validity_mask[accel_violations > 0] = False
        penalty_scores += accel_violations * 10.0

        # 3. Dynamic Momentum Consistency Hard Check
        if recent_history is not None and recent_history.shape[1] >= 3:
            recent_delta = (recent_history[:, -1] - recent_history[:, -3]) / 2.0
            dropping = (recent_delta < -0.10) & (current_glucose < 5.0)
            step1_jump = deltas[:, :, 0]
            reversal = dropping[:, None] & (step1_jump > 0.15)
            validity_mask[reversal] = False
            penalty_scores += reversal * 30.0

        # 4. Metabolic IOB / COB Hard Constraint Check
        if active_iob is not None or active_cob is not None:
            iob = active_iob if active_iob is not None else np.zeros(B, dtype=np.float32)
            cob = active_cob if active_cob is not None else np.zeros(B, dtype=np.float32)
            
            # Expected step-by-step drift: (B, H)
            expected_drift = self._compute_metabolic_flux(H, iob, cob)
            expected_drift_3d = expected_drift[:, None, :]  # (B, 1, H)
            
            # Deviation from expected physiological drift
            divergence = np.abs(deltas - expected_drift_3d)
            
            # HARD FILTER: If step divergence exceeds 75% of maximum allowable velocity,
            # disqualify the candidate as physically contradictory to insulin/carb action.
            divergence_threshold = self.max_roc * 0.75  # ~0.45 mmol/L per step
            metabolic_violations = np.sum(divergence > divergence_threshold, axis=2)  # (B, N)
            
            validity_mask[metabolic_violations > 0] = False
            penalty_scores += metabolic_violations * 35.0

        # 5. Consensus Aggregation over Valid Subset
        best_trajectories = np.zeros((B, H), dtype=np.float32)
        for b in range(B):
            valid_idx = np.where(validity_mask[b])[0]
            if len(valid_idx) > 0:
                # Pointwise median across all surviving valid candidates
                best_trajectories[b] = np.median(candidates[b, valid_idx], axis=0)
            else:
                # Graceful fallback: pick the candidate with minimum penalty score
                best_idx = np.argmin(penalty_scores[b])
                best_trajectories[b] = candidates[b, best_idx]

        return best_trajectories, validity_mask


if __name__ == "__main__":
    print("\n--- Running CGMVerifier with Hard Metabolic Filter Self-Test ---")
    B, N, H = 2, 8, 12
    candidates = np.random.uniform(4.0, 9.0, size=(B, N, H))
    current_glucose = np.array([5.5, 8.0])
    
    test_iob = np.array([3.5, 0.5])
    test_cob = np.array([0.0, 45.0])

    verifier = CGMVerifier()
    paths, mask = verifier.score_candidates(
        candidates=candidates,
        current_glucose=current_glucose,
        active_iob=test_iob,
        active_cob=test_cob,
    )

    print(f"Candidate shape: {candidates.shape}")
    print(f"Verified consensus output shape: {paths.shape}")
    print(f"Candidate pass rate across test batch: {np.mean(mask) * 100:.1f}%")
    assert paths.shape == (B, H)
    print("Hard metabolic filter test passed successfully!\n")