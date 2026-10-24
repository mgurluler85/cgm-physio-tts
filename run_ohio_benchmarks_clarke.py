# run_ohio_benchmarks.py
import os
import numpy as np
import pandas as pd
from verifiers.rules_analytical import CGMVerifierAnalytical
from evaluation.metrics import compute_cgm_metrics


def load_event_features(metadata_path, events_path, num_windows):
    """
    Computes biologically grounded active IOB and COB per window
    directly from historical_events.csv using pharmacokinetic decay modeling.
    """
    iob = np.zeros(num_windows, dtype=np.float32)
    cob = np.zeros(num_windows, dtype=np.float32)

    target_path = events_path if os.path.exists(events_path) else metadata_path

    if os.path.exists(target_path):
        try:
            df_ev = pd.read_csv(target_path)
            tau_dia = 180.0   # Duration of insulin action (min)
            tau_carb = 120.0  # Duration of carbohydrate absorption (min)

            for w_idx in range(num_windows):
                w_events = df_ev[df_ev["window_id"] == w_idx]
                if w_events.empty:
                    continue

                # 1. Active IOB from boluses
                bolus_rows = w_events[w_events["event_type"].str.lower() == "bolus"]
                for _, row in bolus_rows.iterrows():
                    dose = float(row["value"])
                    mins_ago = float(row["minutes_before_forecast"])
                    if 0 <= mins_ago <= tau_dia:
                        fraction_remaining = 1.0 - (mins_ago / tau_dia)
                        iob[w_idx] += dose * fraction_remaining

                # 2. Active COB from meals
                meal_rows = w_events[w_events["event_type"].str.lower() == "meal"]
                for _, row in meal_rows.iterrows():
                    carbs = float(row["value"])
                    mins_ago = float(row["minutes_before_forecast"])
                    if 0 <= mins_ago <= tau_carb:
                        fraction_remaining = 1.0 - (mins_ago / tau_carb)
                        cob[w_idx] += carbs * fraction_remaining

            print(f"[load_event_features] Successfully parsed {os.path.basename(target_path)}:")
            print(f"  -> Active IOB: non-zero in {np.count_nonzero(iob)}/{num_windows} windows (max={iob.max():.2f} U)")
            print(f"  -> Active COB: non-zero in {np.count_nonzero(cob)}/{num_windows} windows (max={cob.max():.1f} g)")

        except Exception as e:
            print(f"[load_event_features] Error parsing events from {target_path}: {e}")

    return iob, cob


def evaluate_candidate_file(exp_path, eval_path, metadata_path, events_path):
    exp_data = np.load(exp_path)
    candidates = exp_data["candidates"]  # (B, N, H)
    contexts = exp_data["contexts"]      # (B, Lookback)

    eval_data = np.load(eval_path)
    y_true = eval_data["futures"]        # (B, H)

    B, N, H = candidates.shape
    current_glucose = contexts[:, -1]
    active_iob, active_cob = load_event_features(metadata_path, events_path, B)

    # Baseline (Candidate #0 - Single greedy pass)
    baseline_trajectories = candidates[:, 0, :]
    baseline_metrics = compute_cgm_metrics(y_true, baseline_trajectories)

    # Verifier with Asymmetric Hypoglycemia Risk Penalty & Active Metabolic Flux
    verifier = CGMVerifierAnalytical(
        max_roc_mmoll_per_min=0.12,
        max_accel_mmoll_per_min2=0.03,
        hypo_safety_floor=3.9,
        interval_minutes=5,
        default_isf=2.5,
        default_cir=10.0,
        lambda_metabolic=1.5,
        lambda_continuity=0.5,
        lambda_smoothness=0.08,
        lambda_hypo_risk=1.2,
    )

    best_trajectories, validity_mask, _ = verifier.score_candidates(
        candidates=candidates,
        current_glucose=current_glucose,
        recent_history=contexts,
        active_iob=active_iob,
        active_cob=active_cob,
    )

    verified_metrics = compute_cgm_metrics(y_true, best_trajectories)
    pass_rate = np.mean(validity_mask) * 100.0

    return H, pass_rate, baseline_metrics, verified_metrics


def run_horizon_sweep():
    base_dir = "sameeya_handoff" if os.path.exists("sameeya_handoff") else "data_ohio/sameeya_handoff"
    meta_path = os.path.join(base_dir, "window_metadata.csv")
    ev_path = os.path.join(base_dir, "historical_events.csv")

    horizon_configs = [
        ("chronos_N16_H12.npz", "evaluation_N16_H12.npz", 60),
        ("chronos_N16_H24.npz", "evaluation_N16_H24.npz", 120),
        ("chronos_N16_H48.npz", "evaluation_N16_H48.npz", 240),
    ]

    results = []
    print("\nRunning Forecast Horizon Sweep on OhioT1DM with Asymmetric Hypo Risk & Clarke EGA (N=16)...")

    for cand_file, eval_file, mins in horizon_configs:
        exp_full = os.path.join(base_dir, "experiments", cand_file)
        eval_full = os.path.join(base_dir, "experiments", eval_file)

        if not os.path.exists(exp_full) or not os.path.exists(eval_full):
            print(f"Skipping {cand_file}: files not found.")
            continue

        h_val, pass_rate, b_m, v_m = evaluate_candidate_file(exp_full, eval_full, meta_path, ev_path)

        # Extract Clarke EGA metrics if present, otherwise default to 0.0
        base_zone_ab = b_m.get("Clarke Zone A+B (%)", 0.0)
        ver_zone_ab = v_m.get("Clarke Zone A+B (%)", 0.0)
        base_zone_a = b_m.get("Clarke Zone A (%)", 0.0)
        ver_zone_a = v_m.get("Clarke Zone A (%)", 0.0)

        results.append({
            "H": h_val,
            "Minutes": mins,
            "Pass Rate (%)": pass_rate,
            "Base RMSE": b_m["RMSE (mmol/L)"],
            "Ver RMSE": v_m["RMSE (mmol/L)"],
            "RMSE Delta": ((v_m["RMSE (mmol/L)"] - b_m["RMSE (mmol/L)"]) / b_m["RMSE (mmol/L)"]) * 100.0,
            "Base MAE": b_m["MAE (mmol/L)"],
            "Ver MAE": v_m["MAE (mmol/L)"],
            "MAE Delta": ((v_m["MAE (mmol/L)"] - b_m["MAE (mmol/L)"]) / b_m["MAE (mmol/L)"]) * 100.0,
            "Base MARD": b_m["MARD (%)"],
            "Ver MARD": v_m["MARD (%)"],
            "MARD Delta": ((v_m["MARD (%)"] - b_m["MARD (%)"]) / b_m["MARD (%)"]) * 100.0,
            "Base FNR": b_m["Hypo FNR (%)"],
            "Ver FNR": v_m["Hypo FNR (%)"],
            "Base Zone A": base_zone_a,
            "Ver Zone A": ver_zone_a,
            "Base Zone A+B": base_zone_ab,
            "Ver Zone A+B": ver_zone_ab,
        })

    # Summary Display
    print("\n" + "=" * 118)
    print(
        f"{'Horizon':<14} | {'Pass%':<6} | {'RMSE (Ver / Base)':<21} | "
        f"{'MARD (Ver / Base)':<19} | {'Hypo FNR (Ver/Base)':<20} | {'Clarke A+B (Ver/Base)'}"
    )
    print("-" * 118)
    for r in results:
        h_label = f"H={r['H']} ({r['Minutes']}m)"
        rmse_str = f"{r['Ver RMSE']:.3f} / {r['Base RMSE']:.3f} ({r['RMSE Delta']:+.1f}%)"
        mard_str = f"{r['Ver MARD']:.2f}% / {r['Base MARD']:.2f}%"
        fnr_str = f"{r['Ver FNR']:.1f}% / {r['Base FNR']:.1f}%"
        ceg_str = f"{r['Ver Zone A+B']:.1f}% / {r['Base Zone A+B']:.1f}%"
        print(f"{h_label:<14} | {r['Pass Rate (%)']:<6.1f} | {rmse_str:<21} | {mard_str:<19} | {fnr_str:<20} | {ceg_str}")
    print("=" * 118 + "\n")


if __name__ == "__main__":
    run_horizon_sweep()