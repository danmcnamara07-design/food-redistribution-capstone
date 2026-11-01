"""Forecast accuracy metrics."""
import numpy as np


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def wape(y, yhat):
    """Weighted absolute percentage error: total absolute error / total actual."""
    y = np.asarray(y, dtype=float)
    return float(np.sum(np.abs(y - np.asarray(yhat))) / max(np.sum(np.abs(y)), 1e-9))


def pinball(y, q_pred, q):
    """Quantile (pinball) loss for quantile q."""
    diff = np.asarray(y) - np.asarray(q_pred)
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def coverage(y, lo, hi):
    """Share of actuals inside the [lo, hi] band."""
    y = np.asarray(y)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))
