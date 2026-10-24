# evaluation/metrics.py
import numpy as np


def clarke_error_grid(y_true, y_pred):
    """
    Computes Clarke Error Grid zone percentages (A, B, C, D, E).
    Accepts y_true and y_pred in mmol/L, converts to mg/dL for standard thresholds.
    y_true, y_pred: flattened 1D arrays or matching ndarrays.
    Returns: dict with percentage of points in Zone A, Zone B, Zone A+B, etc.
    """
    ref = np.asarray(y_true, dtype=np.float64).flatten() * 18.0182
    pred = np.asarray(y_pred, dtype=np.float64).flatten() * 18.0182
    n = len(ref)
    
    if n == 0:
        return {"Zone A (%)": 0.0, "Zone B (%)": 0.0, "Zone A+B (%)": 0.0, "Zone C/D/E (%)": 0.0}

    zone_a = 0
    zone_b = 0
    zone_c = 0
    zone_d = 0
    zone_e = 0

    for r, p in zip(ref, pred):
        # Zone A
        if (p <= 70 and r <= 70) or (p <= 1.2 * r and p >= 0.8 * r):
            zone_a += 1
        # Zone E
        elif (r <= 70 and p >= 180) or (r >= 180 and p <= 70):
            zone_e += 1
        # Zone D
        elif ((r >= 70 and r <= 290) and (p >= r + 110)) or ((r >= 70 and r <= 180) and (p <= (7/5) * r - 182)):
            zone_d += 1
        # Zone C
        elif (r >= 130 and p <= (7/5) * r - 110) or (r >= 180 and r <= 290 and p >= (7/5) * r + 110):
            zone_c += 1
        # Zone B
        else:
            zone_b += 1

    pct_a = (zone_a / n) * 100.0
    pct_b = (zone_b / n) * 100.0
    pct_ab = ((zone_a + zone_b) / n) * 100.0
    pct_cde = ((zone_c + zone_d + zone_e) / n) * 100.0

    return {
        "Clarke Zone A (%)": pct_a,
        "Clarke Zone B (%)": pct_b,
        "Clarke Zone A+B (%)": pct_ab,
        "Clarke Dangerous C/D/E (%)": pct_cde,
    }

def compute_cgm_metrics(y_true, y_pred, hypo_threshold=3.9):
    """
    Computes RMSE, MAE, MARD, Hypo FNR, and Clarke Error Grid zones.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Standard continuous metrics
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mard = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-6))) * 100.0

    # Hypoglycemia FNR (< hypo_threshold)
    true_hypo = y_true < hypo_threshold
    pred_hypo = y_pred < hypo_threshold

    total_true_hypo = np.sum(true_hypo)
    if total_true_hypo > 0:
        hypo_fnr = (np.sum(true_hypo & ~pred_hypo) / total_true_hypo) * 100.0
    else:
        hypo_fnr = 0.0

    # Clarke Error Grid
    ceg_results = clarke_error_grid(y_true, y_pred)

    metrics = {
        "RMSE (mmol/L)": float(rmse),
        "MAE (mmol/L)": float(mae),
        "MARD (%)": float(mard),
        "Hypo FNR (%)": float(hypo_fnr),
        **ceg_results
    }
    return metrics