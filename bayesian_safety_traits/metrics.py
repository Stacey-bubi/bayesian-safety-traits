from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


def expected_calibration_error(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10) -> float:
    """Bin predictions and average |empirical rate - predicted prob| per bin.

    Args:
        y_true: 0/1 outcomes.
        p_pred: predicted P(refuse), same length as y_true.
        n_bins: number of equal-width probability bins.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(p_pred, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - p_pred[mask].mean())
    return float(ece)


def coverage(true_rate: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Fraction of empirical rates that fall inside their predicted interval.

    Args:
        true_rate: empirical refusal rate per cell (e.g. per prompt x alpha).
        lo, hi: matching lower/upper interval bounds, e.g. from
            RefusalModel.predict_interval().
    """
    return float(((true_rate >= lo) & (true_rate <= hi)).mean())


def score(y_true: np.ndarray, p_pred: np.ndarray) -> dict[str, float]:
    """Compute log loss, Brier score, and ECE for a set of predictions.

    Args:
        y_true: 0/1 outcomes.
        p_pred: predicted P(refuse), same length as y_true.
    """
    p_pred = np.clip(p_pred, 1e-6, 1 - 1e-6)
    return {
        "log_loss": log_loss(y_true, p_pred, labels=[0, 1]),
        "brier": brier_score_loss(y_true, p_pred),
        "ece": expected_calibration_error(np.asarray(y_true), p_pred),
    }
