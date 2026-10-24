
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

DATA_DIR = Path("data/OhioT1DM")
OUT_DIR = Path("results/official_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONTEXT_LENGTH = 48
PREDICTION_LENGTH = 48
WINDOW_LENGTH = CONTEXT_LENGTH + PREDICTION_LENGTH

MIN_INTERVAL = 4
MAX_INTERVAL = 6
CONVERSION_FACTOR = 18.0182


def load_glucose(xml_path):
    root = ET.parse(xml_path).getroot()
    glucose_element = root.find("glucose_level")

    if glucose_element is None:
        raise RuntimeError(
            f"Missing glucose_level: {xml_path}"
        )

    records = []

    for event in glucose_element.findall("event"):
        records.append({
            "timestamp": event.get("ts"),
            "glucose": float(event.get("value"))
            / CONVERSION_FACTOR,
        })

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError(
            f"No glucose records: {xml_path}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%d-%m-%Y %H:%M:%S",
    )

    return df.sort_values(
        "timestamp"
    ).reset_index(drop=True)


def extract_windows(df):
    time_diff = (
        df["timestamp"].diff().dt.total_seconds()
        / 60
    )

    segment_ids = (
        ~time_diff.between(
            MIN_INTERVAL,
            MAX_INTERVAL,
        )
    ).cumsum()

    windows = []

    for _, segment in df.groupby(segment_ids):
        values = segment["glucose"].to_numpy(
            dtype=np.float32
        )
        timestamps = segment["timestamp"].to_numpy()

        for start in range(
            0,
            len(values) - WINDOW_LENGTH + 1,
            PREDICTION_LENGTH,
        ):
            end = start + WINDOW_LENGTH

            windows.append({
                "context": values[
                    start:start + CONTEXT_LENGTH
                ],
                "future": values[
                    start + CONTEXT_LENGTH:end
                ],
                "forecast_start": str(
                    timestamps[start + CONTEXT_LENGTH]
                ),
            })

    return windows


def main():
    test_files = sorted(
        DATA_DIR.glob("*/test/*.xml")
    )

    if len(test_files) != 12:
        raise RuntimeError(
            f"Expected 12 test XML files, "
            f"found {len(test_files)}"
        )

    contexts = []
    futures = []
    patient_ids = []
    forecast_starts = []

    print("\nEXPERIMENT 45: OFFICIAL TEST WINDOWS")
    print("=" * 70)

    for xml_path in test_files:
        patient_id = xml_path.name.split("-")[0]

        df = load_glucose(xml_path)
        windows = extract_windows(df)

        print(
            f"Patient {patient_id}: "
            f"{len(windows)} test windows"
        )

        for window in windows:
            contexts.append(window["context"])
            futures.append(window["future"])
            patient_ids.append(patient_id)
            forecast_starts.append(
                window["forecast_start"]
            )

    if not contexts:
        raise RuntimeError(
            "No valid test windows found."
        )

    output = OUT_DIR / "official_test_windows.npz"

    np.savez_compressed(
        output,
        contexts=np.stack(contexts),
        futures=np.stack(futures),
        patient_ids=np.array(patient_ids),
        forecast_starts=np.array(forecast_starts),
    )

    print("\nTEST DATASET SUMMARY")
    print("-" * 70)
    print("Patients:", len(set(patient_ids)))
    print("Total windows:", len(contexts))
    print("Contexts:", np.stack(contexts).shape)
    print("Futures:", np.stack(futures).shape)
    print("Units: mmol/L")
    print("Saved:", output)
    print("EXPERIMENT 45 COMPLETED")


if __name__ == "__main__":
    main()
