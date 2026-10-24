
from pathlib import Path

import numpy as np
import pandas as pd

from verifiers.rules_analytical import CGMVerifierAnalytical

ROOT = Path("results/official_test")
INPUT = ROOT / "official_test_chronos_candidates.npz"
OUTPUT = ROOT / "verifier"
OUTPUT.mkdir(parents=True, exist_ok=True)

HORIZONS = [12, 24, 48]


def calculate_metrics(predictions, targets):
    errors = predictions - targets

    return {
        "mae_mmoll": float(
            np.mean(np.abs(errors))
        ),
        "rmse_mmoll": float(
            np.sqrt(np.mean(errors ** 2))
        ),
    }


with np.load(INPUT) as data:
    contexts = data["contexts"].astype(np.float32)
    futures = data["futures"].astype(np.float32)
    candidates = data["candidates"].astype(np.float32)
    patient_ids = data["patient_ids"].astype(str)
    forecast_starts = data["forecast_starts"].astype(str)

assert contexts.shape == (494, 48)
assert futures.shape == (494, 48)
assert candidates.shape == (494, 20, 48)

verifier = CGMVerifierAnalytical(
    max_roc_mmoll_per_min=0.12,
    max_accel_mmoll_per_min2=0.03,
    hypo_safety_floor=3.9,
    interval_minutes=5,
    default_isf=2.5,
    default_cir=10.0,
    lambda_metabolic=1.0,
    lambda_continuity=0.5,
    lambda_smoothness=0.2,
)

rows = []
patient_rows = []
window_rows = []

all_predictions = {}
all_validity_masks = {}
all_scores = {}

print("\nEXPERIMENT 49: OFFICIAL TEST VERIFIER")
print("=" * 70)
print("Windows:", len(contexts))
print("Candidates per window:", candidates.shape[1])

for horizon in HORIZONS:
    # Horizon-specific verifier selection.
    # Same candidate pool, truncated to each horizon.
    candidate_subset = candidates[:, :, :horizon]

    selected, validity, scores = verifier.score_candidates(
        candidates=candidate_subset.copy(),
        current_glucose=contexts[:, -1],
        recent_history=contexts,
        active_iob=None,
        active_cob=None,
    )

    mean20 = candidate_subset.mean(axis=1)
    single = candidate_subset[:, 0, :]
    targets = futures[:, :horizon]

    all_predictions[f"H{horizon}"] = selected
    all_validity_masks[f"H{horizon}"] = validity
    all_scores[f"H{horizon}"] = scores

    n_valid = validity.sum(axis=1)
    no_valid = int(np.sum(n_valid == 0))
    any_valid = int(np.sum(n_valid > 0))

    best_indices = np.argmin(scores, axis=1)
    selected_valid = validity[
        np.arange(len(validity)),
        best_indices,
    ]

    print(f"\nHORIZON {horizon}")
    print("-" * 70)
    print("Windows with >=1 valid candidate:", any_valid)
    print("Windows with no valid candidates:", no_valid)
    print(
        "Selected candidate valid:",
        int(selected_valid.sum()),
        "/",
        len(selected_valid),
    )

    methods = {
        "Chronos Single": single,
        "Chronos Mean-20": mean20,
        "Chronos Verifier-20": selected,
    }

    for method, prediction in methods.items():
        result = calculate_metrics(
            prediction, targets
        )

        rows.append({
            "method": method,
            "horizon": horizon,
            **result,
        })

        print(
            f"{method:23s} "
            f"MAE={result['mae_mmoll']:.4f} "
            f"RMSE={result['rmse_mmoll']:.4f}"
        )

        per_window = np.mean(
            np.abs(prediction - targets),
            axis=1,
        )

        for i, value in enumerate(per_window):
            window_rows.append({
                "method": method,
                "horizon": horizon,
                "patient_id": patient_ids[i],
                "forecast_start": forecast_starts[i],
                "window_mae": float(value),
            })

        for patient in np.unique(patient_ids):
            mask = patient_ids == patient

            patient_result = calculate_metrics(
                prediction[mask],
                targets[mask],
            )

            patient_rows.append({
                "method": method,
                "horizon": horizon,
                "patient_id": patient,
                "n_windows": int(mask.sum()),
                **patient_result,
            })

    rows.append({
        "method": "Validity diagnostic",
        "horizon": horizon,
        "mae_mmoll": np.nan,
        "rmse_mmoll": np.nan,
        "windows_no_valid": no_valid,
        "fraction_no_valid": no_valid / len(contexts),
        "fraction_selected_valid": float(
            selected_valid.mean()
        ),
    })


pd.DataFrame(rows).to_csv(
    OUTPUT / "verifier_metrics.csv",
    index=False,
)

pd.DataFrame(patient_rows).to_csv(
    OUTPUT / "verifier_patient_metrics.csv",
    index=False,
)

pd.DataFrame(window_rows).to_csv(
    OUTPUT / "verifier_window_errors.csv",
    index=False,
)

np.savez_compressed(
    OUTPUT / "verifier_test_predictions.npz",
    predictions_h12=all_predictions["H12"],
    predictions_h24=all_predictions["H24"],
    predictions_h48=all_predictions["H48"],
    validity_h12=all_validity_masks["H12"],
    validity_h24=all_validity_masks["H24"],
    validity_h48=all_validity_masks["H48"],
    scores_h12=all_scores["H12"],
    scores_h24=all_scores["H24"],
    scores_h48=all_scores["H48"],
    futures=futures,
    patient_ids=patient_ids,
    forecast_starts=forecast_starts,
)

print("\nSUMMARY: MAE")
print("=" * 70)

results = pd.DataFrame(rows)
results = results[
    results["method"] != "Validity diagnostic"
]

print(
    results.pivot(
        index="method",
        columns="horizon",
        values="mae_mmoll",
    ).round(4).to_string()
)

print("\nSaved to:", OUTPUT)
print("EXPERIMENT 49 COMPLETED")
