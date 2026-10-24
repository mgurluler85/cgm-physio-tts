
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("results/official_test")
CHRONOS_FILE = ROOT / "official_test_chronos_candidates.npz"
LSTM_FILE = (
    ROOT / "lstm/official_test_lstm_predictions.npz"
)

OUT = ROOT / "benchmark"
OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = [12, 24, 48]


def metrics(pred, actual):
    error = pred - actual
    return (
        float(np.mean(np.abs(error))),
        float(np.sqrt(np.mean(error ** 2))),
    )


with np.load(CHRONOS_FILE) as data:
    contexts = data["contexts"]
    futures = data["futures"]
    candidates = data["candidates"]
    patient_ids = data["patient_ids"].astype(str)
    forecast_starts = data["forecast_starts"].astype(str)

with np.load(LSTM_FILE) as data:
    lstm_predictions = data["predictions"]
    lstm_seeds = data["seeds"]
    lstm_patient_ids = data["patient_ids"].astype(str)
    lstm_starts = data["forecast_starts"].astype(str)
    lstm_futures = data["futures"]

assert contexts.shape == (494, 48)
assert futures.shape == (494, 48)
assert candidates.shape == (494, 20, 48)
assert lstm_predictions.shape == (5, 494, 48)

# Ensure both experiments use exactly the same windows.
assert np.array_equal(patient_ids, lstm_patient_ids)
assert np.array_equal(forecast_starts, lstm_starts)
assert np.allclose(futures, lstm_futures)

persistence = np.repeat(contexts[:, -1:], 48, axis=1)
chronos_single = candidates[:, 0, :]
chronos_mean20 = candidates.mean(axis=1)

rows = []
patient_rows = []
window_rows = []


def record(method, horizon, predictions, seed=None):
    target = futures[:, :horizon]
    pred = predictions[:, :horizon]

    mae, rmse = metrics(pred, target)

    rows.append({
        "method": method,
        "seed": seed,
        "horizon": horizon,
        "mae_mmoll": mae,
        "rmse_mmoll": rmse,
    })

    per_window_mae = np.mean(
        np.abs(pred - target), axis=1
    )

    for i, value in enumerate(per_window_mae):
        window_rows.append({
            "method": method,
            "seed": seed,
            "horizon": horizon,
            "patient_id": patient_ids[i],
            "forecast_start": forecast_starts[i],
            "window_mae": float(value),
        })

    for patient in np.unique(patient_ids):
        mask = patient_ids == patient
        p_mae, p_rmse = metrics(
            pred[mask], target[mask]
        )

        patient_rows.append({
            "method": method,
            "seed": seed,
            "horizon": horizon,
            "patient_id": patient,
            "n_windows": int(mask.sum()),
            "mae_mmoll": p_mae,
            "rmse_mmoll": p_rmse,
        })


for horizon in HORIZONS:
    record("Persistence", horizon, persistence)
    record("Chronos Single", horizon, chronos_single)
    record("Chronos Mean-20", horizon, chronos_mean20)

    # Horizon-specific oracle: select the candidate
    # with lowest true MAE for the evaluated horizon.
    # This is not a deployable forecasting method.
    candidate_errors = np.mean(
        np.abs(
            candidates[:, :, :horizon]
            - futures[:, None, :horizon]
        ),
        axis=2,
    )

    best_indices = np.argmin(candidate_errors, axis=1)

    oracle = candidates[
        np.arange(len(candidates)),
        best_indices,
        :
    ]

    record("Oracle-20 (horizon-specific)", horizon, oracle)

    for i, seed in enumerate(lstm_seeds):
        record(
            "LSTM",
            horizon,
            lstm_predictions[i],
            seed=int(seed),
        )


metrics_df = pd.DataFrame(rows)
patient_df = pd.DataFrame(patient_rows)
window_df = pd.DataFrame(window_rows)

metrics_df.to_csv(
    OUT / "official_test_metrics.csv", index=False
)
patient_df.to_csv(
    OUT / "official_test_patient_metrics.csv",
    index=False,
)
window_df.to_csv(
    OUT / "official_test_window_errors.csv",
    index=False,
)

# Show LSTM mean and sample standard deviation
# across five independently trained seeds.
lstm_summary = (
    metrics_df[metrics_df["method"] == "LSTM"]
    .groupby("horizon")
    .agg(
        mean_mae=("mae_mmoll", "mean"),
        std_mae=("mae_mmoll", "std"),
        mean_rmse=("rmse_mmoll", "mean"),
        std_rmse=("rmse_mmoll", "std"),
    )
    .reset_index()
)

other_methods = metrics_df[
    metrics_df["method"] != "LSTM"
]

comparison = other_methods.pivot(
    index="method",
    columns="horizon",
    values="mae_mmoll",
)

comparison.loc["LSTM (5-seed mean)"] = (
    lstm_summary.set_index("horizon")["mean_mae"]
)

# Patient-equal macro MAE:
# every patient contributes equally regardless
# of the number of test windows.
patient_macro = (
    patient_df
    .groupby(["method", "seed", "horizon"],
             dropna=False)["mae_mmoll"]
    .mean()
    .reset_index(name="patient_macro_mae")
)

patient_macro.to_csv(
    OUT / "official_test_patient_macro.csv",
    index=False,
)

print("\nEXPERIMENT 48: OFFICIAL TEST BENCHMARK")
print("=" * 70)
print("Windows:", len(contexts))
print("Patients:", len(np.unique(patient_ids)))

print("\nMAE (mmol/L)")
print("-" * 70)
print(comparison.round(4).to_string())

print("\nLSTM MULTI-SEED SUMMARY")
print("-" * 70)
print(lstm_summary.round(4).to_string(index=False))

print("\nPATIENT-EQUAL MACRO MAE")
print("-" * 70)
macro_display = (
    patient_macro[
        patient_macro["method"] != "LSTM"
    ]
    .pivot(
        index="method",
        columns="horizon",
        values="patient_macro_mae",
    )
)

lstm_macro = (
    patient_macro[
        patient_macro["method"] == "LSTM"
    ]
    .groupby("horizon")["patient_macro_mae"]
    .mean()
)

macro_display.loc["LSTM (5-seed mean)"] = lstm_macro
print(macro_display.round(4).to_string())

print("\nSaved results to:", OUT)
print("EXPERIMENT 48 COMPLETED")
