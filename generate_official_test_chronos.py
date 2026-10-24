
from pathlib import Path

import numpy as np
import torch
from chronos import ChronosPipeline

INPUT_PATH = Path(
    "results/official_test/official_test_windows.npz"
)

OUTPUT_PATH = Path(
    "results/official_test/official_test_chronos_candidates.npz"
)

MODEL_NAME = "amazon/chronos-t5-small"
NUM_SAMPLES = 20
PREDICTION_LENGTH = 48

CONVERSION_FACTOR = 18.0182

torch.manual_seed(42)
np.random.seed(42)


def main():
    with np.load(INPUT_PATH) as data:
        contexts = data["contexts"].astype(np.float32)
        futures = data["futures"].astype(np.float32)
        patient_ids = data["patient_ids"]
        forecast_starts = data["forecast_starts"]

    assert contexts.shape == (494, 48)
    assert futures.shape == (494, 48)

    print("\nEXPERIMENT 47: CHRONOS OFFICIAL TEST")
    print("=" * 70)
    print("Model:", MODEL_NAME)
    print("Windows:", len(contexts))
    print("Samples per window:", NUM_SAMPLES)
    print("Prediction length:", PREDICTION_LENGTH)
    print("Device: CPU")

    # Original validation experiment used mg/dL.
    contexts_mgdl = contexts * CONVERSION_FACTOR

    pipeline = ChronosPipeline.from_pretrained(
        MODEL_NAME,
        device_map="cpu",
        torch_dtype=torch.float32,
    )

    all_candidates = []

    for i, context in enumerate(contexts_mgdl):
        context_tensor = torch.tensor(
            context,
            dtype=torch.float32,
        )

        with torch.no_grad():
            predictions = pipeline.predict(
                context_tensor,
                prediction_length=PREDICTION_LENGTH,
                num_samples=NUM_SAMPLES,
            )

        candidates_mgdl = (
            predictions.cpu().numpy()[0]
        )

        if candidates_mgdl.shape != (
            NUM_SAMPLES,
            PREDICTION_LENGTH,
        ):
            raise ValueError(
                f"Unexpected shape: "
                f"{candidates_mgdl.shape}"
            )

        # Convert predictions back to mmol/L.
        candidates_mmoll = (
            candidates_mgdl / CONVERSION_FACTOR
        )

        all_candidates.append(candidates_mmoll)

        if (i + 1) % 20 == 0:
            print(
                f"Completed {i + 1}/{len(contexts)} windows"
            )

    candidates_array = np.stack(
        all_candidates
    ).astype(np.float32)

    assert candidates_array.shape == (
        494,
        20,
        48,
    )

    assert np.isfinite(candidates_array).all()

    np.savez_compressed(
        OUTPUT_PATH,
        contexts=contexts,
        futures=futures,
        candidates=candidates_array,
        patient_ids=patient_ids,
        forecast_starts=forecast_starts,
    )

    print("\nCHRONOS TEST DATASET")
    print("-" * 70)
    print("Contexts:", contexts.shape)
    print("Futures:", futures.shape)
    print("Candidates:", candidates_array.shape)
    print("Total candidates:", len(contexts) * NUM_SAMPLES)
    print("Units: mmol/L")
    print("Saved:", OUTPUT_PATH)
    print("\nEXPERIMENT 47 COMPLETED")


if __name__ == "__main__":
    main()
