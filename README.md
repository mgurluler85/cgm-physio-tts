# CGM Forecasting — Chronos, LSTM and Physiology-Informed Verification

Code for the official OhioT1DM test-set comparison of Chronos test-time scaling, LSTM baselines and candidate verification.

## Scripts

- `45_prepare_official_test_windows.py` — prepares official test windows from locally obtained OhioT1DM data.
- `47_generate_official_test_chronos.py` — generates Chronos candidates.
- `40_prepare_lstm_dataset.py` — prepares LSTM training data.
- `44_lstm_multiseed_robustness.py` — trains the five-seed LSTM ensemble.
- `46_evaluate_lstm_official_test.py` — evaluates LSTM predictions on the official test set.
- `48_official_test_benchmark.py` — calculates MAE/RMSE comparison results.
- `49_official_test_verifier.py` and `verifiers/rules_analytical.py` — existing analytical verifier implementation (preserved unchanged).

## Data

This repository does not include raw OhioT1DM patient records, patient-level predictions, or other patient-specific data to protect patient privacy and respect dataset access restrictions.

The scripts require locally obtained OhioT1DM data under data/OhioT1DM/, with intermediate outputs stored under results/. The dataset should be obtained through its authorized access channel.

## Results

`results/official_test/benchmark/official_test_metrics.csv` contains aggregate Chronos/LSTM benchmark metrics. `results/official_test/verifier/verifier_metrics.csv` contains the aggregate results of the included analytical verifier implementation. These verifier values are from that implementation and should not be substituted for the separately reported physiology-informed IOB/COB poster results.

## Physiology-informed verification

The `verifiers/` and `evaluation/` modules and `run_ohio_benchmarks_clarke.py` include the insulin/carbohydrate-aware verification and clinical evaluation code. These files are included without modifications.
