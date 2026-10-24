
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

DATA_DIR = Path("data/OhioT1DM")
VALIDATION_FILE = Path(
    "results/validation/validation_candidates_mmoll.npz"
)
OUT_DIR = Path("results/lstm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONTEXT = 48
HORIZON = 48
TOTAL = CONTEXT + HORIZON
CONVERSION_FACTOR = 18.0182

MIN_INTERVAL = 4
MAX_INTERVAL = 6

# Match the chronological 80% boundary in experiment 21.
CHRONOS_TRAIN_FRACTION = 0.80

# Reserve the later 15% of the pre-validation region
# for LSTM early stopping.
LSTM_EARLY_STOP_FRACTION = 0.15

SEED = 42


def load_glucose(xml_path):
    root = ET.parse(xml_path).getroot()
    events = root.find("glucose_level").findall("event")

    records = [
        {
            "timestamp": event.get("ts"),
            "glucose": float(event.get("value"))
            / CONVERSION_FACTOR,
        }
        for event in events
    ]

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%d-%m-%Y %H:%M:%S",
    )

    return df.sort_values("timestamp").reset_index(drop=True)


def get_segments(df):
    time_diff = (
        df["timestamp"].diff().dt.total_seconds() / 60
    )

    segment_ids = (
        ~time_diff.between(MIN_INTERVAL, MAX_INTERVAL)
    ).cumsum()

    return [
        segment.reset_index(drop=True)
        for _, segment in df.groupby(segment_ids)
    ]


def build_windows(df):
    windows = []

    for segment in get_segments(df):
        values = segment["glucose"].to_numpy(
            dtype=np.float32
        )
        timestamps = segment["timestamp"].to_numpy()

        for start in range(
            0,
            len(values) - TOTAL + 1,
            HORIZON,
        ):
            end = start + TOTAL

            windows.append({
                "context": values[start:start + CONTEXT],
                "future": values[
                    start + CONTEXT:end
                ],
                "start": pd.Timestamp(timestamps[start]),
                "end": pd.Timestamp(timestamps[end - 1]),
                "forecast_start": pd.Timestamp(
                    timestamps[start + CONTEXT]
                ),
            })

    return windows


def no_overlap(a, b):
    return a["end"] < b["start"]


def save_windows(path, windows):
    if not windows:
        raise RuntimeError(
            f"No windows available for {path}"
        )

    np.savez_compressed(
        path,
        contexts=np.stack(
            [w["context"] for w in windows]
        ),
        futures=np.stack(
            [w["future"] for w in windows]
        ),
        patient_ids=np.array(
            [w["patient_id"] for w in windows]
        ),
        forecast_starts=np.array(
            [
                str(w["forecast_start"])
                for w in windows
            ]
        ),
    )


def main():
    all_train = []
    all_early_stop = []

    files = sorted(
        DATA_DIR.glob("*/train/*.xml")
    )

    if not files:
        raise RuntimeError(
            "No OhioT1DM training XML files found."
        )

    with np.load(VALIDATION_FILE) as val:
        val_ids = val["patient_ids"].astype(str)
        val_starts = pd.to_datetime(
            val["forecast_starts"]
        )

    print("\nEXPERIMENT 40: LSTM DATA PREPARATION")
    print("=" * 75)

    for xml_path in files:
        patient_id = xml_path.name.split("-")[0]
        windows = build_windows(
            load_glucose(xml_path)
        )

        split = int(
            len(windows) * CHRONOS_TRAIN_FRACTION
        )

        earlier = windows[:split]
        later = windows[split:]

        if not earlier or not later:
            raise RuntimeError(
                f"Insufficient windows for {patient_id}"
            )

        # Preserve the exact split boundary used by
        # experiment 21 and verify existing holdouts.
        selected_val_times = val_starts[
            val_ids == patient_id
        ]

        if len(selected_val_times) == 0:
            raise RuntimeError(
                f"No validation records for {patient_id}"
            )

        allowed_times = {
            w["forecast_start"] for w in later
        }

        for ts in selected_val_times:
            if ts not in allowed_times:
                raise RuntimeError(
                    f"Validation mismatch for {patient_id}: {ts}"
                )

        # Exclude any earlier window whose recorded interval
        # overlaps the full later validation region.
        boundary_start = min(
            w["start"] for w in later
        )

        earlier = [
            w for w in earlier
            if w["end"] < boundary_start
        ]

        # Chronological train / early-stopping split.
        stop_split = int(
            len(earlier)
            * (1 - LSTM_EARLY_STOP_FRACTION)
        )

        train = earlier[:stop_split]
        early_stop = earlier[stop_split:]

        if not train or not early_stop:
            raise RuntimeError(
                f"Insufficient train/early-stop "
                f"windows for {patient_id}"
            )

        # Remove boundary overlap between train and
        # early stopping, including shared context.
        early_start = min(
            w["start"] for w in early_stop
        )

        train = [
            w for w in train
            if w["end"] < early_start
        ]

        if not train:
            raise RuntimeError(
                f"No leakage-free training windows "
                f"for {patient_id}"
            )

        for w in train:
            w["patient_id"] = patient_id

        for w in early_stop:
            w["patient_id"] = patient_id

        # Sanity checks for chronological separation.
        assert max(w["end"] for w in train) < min(
            w["start"] for w in early_stop
        )
        assert max(w["end"] for w in early_stop) < (
            boundary_start
        )

        all_train.extend(train)
        all_early_stop.extend(early_stop)

        print(
            f"Patient {patient_id}: "
            f"train={len(train)}, "
            f"early_stop={len(early_stop)}, "
            f"Chronos_holdout={len(selected_val_times)}"
        )

    train_path = OUT_DIR / "lstm_train.npz"
    stop_path = OUT_DIR / "lstm_early_stop.npz"

    save_windows(train_path, all_train)
    save_windows(stop_path, all_early_stop)

    print("\nDATASET SUMMARY")
    print("-" * 75)
    print("Training windows:", len(all_train))
    print("Early-stopping windows:", len(all_early_stop))
    print("Existing Chronos validation windows:", len(val_ids))
    print("Saved:", train_path)
    print("Saved:", stop_path)
    print("Units: mmol/L")
    print("Experiment 40 completed.")


if __name__ == "__main__":
    main()
