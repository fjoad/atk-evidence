"""Pure score-geometry functions; no data loader, training or experiment CLI."""

import numpy as np


def mse_box_bounds(x):
    """Bounds for every reconstruction in [0,1]^T, not only a fitted model."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not x.shape[0] or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("Need a nonempty finite matrix of daily rows")
    with np.errstate(over="raise", invalid="raise"):
        lower = np.mean(np.square(x - np.clip(x, 0., 1.)), axis=1)
        upper = np.mean(np.maximum(np.square(x), np.square(x - 1.)), axis=1)
    return lower, upper


def interval_decisions(lower, upper, threshold):
    """Conservative forced decisions for score > threshold; not joint achievability."""
    lower, upper = np.asarray(lower), np.asarray(upper)
    if (lower.ndim != 1 or not len(lower) or lower.shape != upper.shape
            or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or np.any(lower < 0) or np.any(lower > upper)
            or not np.isfinite(threshold) or threshold < 0):
        raise ValueError("Invalid score intervals or threshold")
    alarm, benign = lower > threshold, upper <= threshold
    return {"must_alarm": alarm, "must_be_benign": benign,
            "unresolved": ~(alarm | benign)}
