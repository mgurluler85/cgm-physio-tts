
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

TEST_FILE = Path(
    "results/official_test/official_test_windows.npz"
)
MODEL_DIR = Path("results/lstm/multiseed")
OUT_DIR = Path("results/official_test/lstm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 2026]
HORIZONS = [12, 24, 48]


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_layers,
        dropout,
        prediction_length,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.head = nn.Linear(
            hidden_size,
            prediction_length,
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])


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


with np.load(TEST_FILE) as data:
    contexts = data["contexts"].astype(np.float32)
    futures = data["futures"].astype(np.float32)
    patient_ids = data["patient_ids"].astype(str)
    forecast_starts = data["forecast_starts"].astype(str)

assert contexts.shape[1] == 48
assert futures.shape[1] == 48
assert len(contexts) == 494
assert len(np.unique(patient_ids)) == 12
assert np.isfinite(contexts).all()
assert np.isfinite(futures).all()

persistence = np.repeat(
    contexts[:, -1:],
    48,
    axis=1,
)

all_predictions = []
metric_rows = []
patient_rows = []

print("\nEXPERIMENT 46: OFFICIAL TEST — LSTM")
print("=" * 70)
print("Test windows:", len(contexts))
print("Patients:", len(np.unique(patient_ids)))
print("Seeds:", SEEDS)

for seed in SEEDS:
    checkpoint_path = (
        MODEL_DIR / f"lstm_seed_{seed}.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["seed"] == seed
    assert checkpoint["context_length"] == 48
    assert checkpoint["prediction_length"] == 48

    model = LSTMForecaster(
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
        prediction_length=checkpoint["prediction_length"],
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )
    model.eval()

    mean = checkpoint["mean"]
    std = checkpoint["std"]

    normalized = (
        (contexts - mean) / std
    )

    batches = []

    with torch.no_grad():
        for batch in np.array_split(
            normalized,
            max(1, int(np.ceil(len(normalized) / 64))),
        ):
            x = torch.from_numpy(
                batch.astype(np.float32)
            ).unsqueeze(-1)

            pred = model(x).numpy()
            batches.append(pred)

    predictions = (
        np.concatenate(batches, axis=0)
        * std + mean
    )

    assert predictions.shape == futures.shape
    assert np.isfinite(predictions).all()

    all_predictions.append(predictions)

    print(f"\nSeed {seed}")

    for horizon in HORIZONS:
        metrics = calculate_metrics(
            predictions[:, :horizon],
            futures[:, :horizon],
        )

        metric_rows.append({
            "method": "LSTM",
            "seed": seed,
            "horizon": horizon,
            "minutes": horizon * 5,
            **metrics,
        })

        print(
            f"  H={horizon}: "
            f"MAE={metrics['mae_mmoll']:.4f}, "
            f"RMSE={metrics['rmse_mmoll']:.4f}"
        )

        for patient_id in np.unique(patient_ids):
            mask = patient_ids == patient_id

            patient_metrics = calculate_metrics(
                predictions[mask, :horizon],
                futures[mask, :horizon],
            )

            patient_rows.append({
                "method": "LSTM",
                "seed": seed,
                "horizon": horizon,
                "patient_id": patient_id,
                "n_windows": int(mask.sum()),
                **patient_metrics,
            })


# Persistence baseline: no training required.
print("\nPersistence baseline")

for horizon in HORIZONS:
    metrics = calculate_metrics(
        persistence[:, :horizon],
        futures[:, :horizon],
    )

    metric_rows.append({
        "method": "Persistence",
        "seed": -1,
        "horizon": horizon,
        "minutes": horizon * 5,
        **metrics,
    })

    print(
        f"  H={horizon}: "
        f"MAE={metrics['mae_mmoll']:.4f}, "
        f"RMSE={metrics['rmse_mmoll']:.4f}"
    )

    for patient_id in np.unique(patient_ids):
        mask = patient_ids == patient_id

        patient_metrics = calculate_metrics(
            persistence[mask, :horizon],
            futures[mask, :horizon],
        )

        patient_rows.append({
            "method": "Persistence",
            "seed": -1,
            "horizon": horizon,
            "patient_id": patient_id,
            "n_windows": int(mask.sum()),
            **patient_metrics,
        })


metrics_df = pd.DataFrame(metric_rows)
patient_df = pd.DataFrame(patient_rows)

metrics_df.to_csv(
    OUT_DIR / "official_test_lstm_metrics.csv",
    index=False,
)

patient_df.to_csv(
    OUT_DIR / "official_test_patient_metrics.csv",
    index=False,
)

np.savez_compressed(
    OUT_DIR / "official_test_lstm_predictions.npz",
    predictions=np.stack(all_predictions),
    seeds=np.array(SEEDS),
    contexts=contexts,
    futures=futures,
    persistence=persistence,
    patient_ids=patient_ids,
    forecast_starts=forecast_starts,
)

summary = (
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

summary.to_csv(
    OUT_DIR / "official_test_lstm_summary.csv",
    index=False,
)

print("\nMULTI-SEED TEST RESULTS")
print("=" * 70)

print(
    metrics_df[
        metrics_df["method"] == "LSTM"
    ].pivot(
        index="seed",
        columns="horizon",
        values="mae_mmoll",
    ).round(4)
)

print("\nLSTM SUMMARY")
print(
    summary.round(4).to_string(index=False)
)

print("\nSaved results to:", OUT_DIR)
print("EXPERIMENT 46 COMPLETED")
