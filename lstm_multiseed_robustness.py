
from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path("results/lstm")
OUT = ROOT / "multiseed"
OUT.mkdir(parents=True, exist_ok=True)

VALIDATION_FILE = Path(
    "results/validation/validation_candidates_mmoll.npz"
)

SEEDS = [42, 123, 456, 789, 2026]
EPOCHS = 150
PATIENCE = 15
BATCH_SIZE = 32
LR = 0.001

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.1

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LSTMForecaster(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            batch_first=True,
        )

        self.head = nn.Linear(HIDDEN_SIZE, 48)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])


def load_xy(path):
    with np.load(path) as data:
        x = data["contexts"].astype(np.float32)
        y = data["futures"].astype(np.float32)

    return x, y


x_train, y_train = load_xy(ROOT / "lstm_train.npz")
x_stop, y_stop = load_xy(ROOT / "lstm_early_stop.npz")

with np.load(VALIDATION_FILE) as data:
    x_val = data["contexts"].astype(np.float32)
    y_val = data["futures"].astype(np.float32)
    candidates = data["candidates"].astype(np.float32)
    patient_ids = data["patient_ids"].astype(str)

assert x_val.shape == (240, 48)
assert y_val.shape == (240, 48)
assert candidates.shape == (240, 20, 48)

# Use training data only for normalization.
train_values = np.concatenate(
    [x_train.ravel(), y_train.ravel()]
)

mean = float(train_values.mean())
std = float(train_values.std())

assert std > 0

def norm(x):
    return (x - mean) / std


train_dataset = TensorDataset(
    torch.from_numpy(norm(x_train)).unsqueeze(-1),
    torch.from_numpy(norm(y_train)),
)

stop_dataset = TensorDataset(
    torch.from_numpy(norm(x_stop)).unsqueeze(-1),
    torch.from_numpy(norm(y_stop)),
)

x_val_tensor = torch.from_numpy(
    norm(x_val)
).unsqueeze(-1)

mean20 = candidates.mean(axis=1)

all_metrics = []
all_predictions = []

print("\nEXPERIMENT 44: MULTI-SEED LSTM")
print("=" * 70)
print("Device:", device)
print("Seeds:", SEEDS)
print("Training windows:", len(x_train))
print("Early-stop windows:", len(x_stop))
print("Validation windows:", len(x_val))

for seed in SEEDS:
    print(f"\n{'=' * 70}")
    print(f"TRAINING SEED {seed}")
    print("=" * 70)

    set_seed(seed)

    # Create a new model and shuffled loader for each seed.
    model = LSTMForecaster().to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    stop_loader = DataLoader(
        stop_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    best_epoch = None
    bad_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_total = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)
            loss = criterion(pred, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            optimizer.step()

            train_total += loss.item() * len(xb)

        train_loss = train_total / len(train_dataset)

        model.eval()
        stop_total = 0.0

        with torch.no_grad():
            for xb, yb in stop_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                loss = criterion(model(xb), yb)
                stop_total += loss.item() * len(xb)

        stop_loss = stop_total / len(stop_dataset)

        if stop_loss < best_loss - 1e-6:
            best_loss = stop_loss
            best_state = copy.deepcopy(
                model.state_dict()
            )
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Train {train_loss:.5f} | "
                f"Early-stop {stop_loss:.5f}"
            )

        if bad_epochs >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch}"
            )
            break

    assert best_state is not None

    model.load_state_dict(best_state)
    model.eval()

    predictions = []

    with torch.no_grad():
        for xb in torch.split(x_val_tensor, 64):
            output = model(xb.to(device))
            predictions.append(
                output.cpu().numpy()
            )

    predictions = (
        np.concatenate(predictions, axis=0)
        * std + mean
    )

    all_predictions.append(predictions)

    for horizon in [12, 24, 48]:
        target = y_val[:, :horizon]
        pred = predictions[:, :horizon]

        mae = float(np.mean(np.abs(pred - target)))
        rmse = float(
            np.sqrt(np.mean((pred - target) ** 2))
        )

        all_metrics.append({
            "seed": seed,
            "horizon": horizon,
            "minutes": horizon * 5,
            "mae_mmoll": mae,
            "rmse_mmoll": rmse,
            "best_epoch": best_epoch,
            "early_stop_mse": best_loss,
        })

    print(
        f"Seed {seed} completed | "
        f"Best epoch: {best_epoch}"
    )

    torch.save(
        {
            "model_state": best_state,
            "mean": mean,
            "std": std,
            "seed": seed,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "context_length": 48,
            "prediction_length": 48,
            "best_epoch": best_epoch,
        },
        OUT / f"lstm_seed_{seed}.pt",
    )

metrics = pd.DataFrame(all_metrics)

summary = (
    metrics.groupby("horizon")
    .agg(
        mean_mae=("mae_mmoll", "mean"),
        std_mae=("mae_mmoll", "std"),
        min_mae=("mae_mmoll", "min"),
        max_mae=("mae_mmoll", "max"),
        mean_rmse=("rmse_mmoll", "mean"),
        std_rmse=("rmse_mmoll", "std"),
    )
    .reset_index()
)

for horizon in [12, 24, 48]:
    mean20_mae = float(
        np.mean(
            np.abs(
                mean20[:, :horizon]
                - y_val[:, :horizon]
            )
        )
    )

    summary.loc[
        summary["horizon"] == horizon,
        "chronos_mean20_mae",
    ] = mean20_mae

    summary.loc[
        summary["horizon"] == horizon,
        "lstm_minus_chronos",
    ] = (
        summary.loc[
            summary["horizon"] == horizon,
            "mean_mae",
        ] - mean20_mae
    )

metrics.to_csv(
    OUT / "lstm_seed_metrics.csv",
    index=False,
)

summary.to_csv(
    OUT / "lstm_multiseed_summary.csv",
    index=False,
)

np.savez_compressed(
    OUT / "lstm_multiseed_predictions.npz",
    predictions=np.stack(all_predictions),
    seeds=np.array(SEEDS),
    patient_ids=patient_ids,
    futures=y_val,
)

print("\nMULTI-SEED RESULTS")
print("=" * 70)
print(
    metrics.pivot(
        index="seed",
        columns="horizon",
        values="mae_mmoll",
    ).round(4)
)

print("\nSUMMARY")
print(
    summary.round(4).to_string(index=False)
)

print("\nSaved results to:", OUT)
print("EXPERIMENT 44 COMPLETED")
