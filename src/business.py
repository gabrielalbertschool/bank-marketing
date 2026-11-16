"""Business layer: expected profit of calling only the clients the model targets."""
from __future__ import annotations

import numpy as np


def profit(y_true, y_prob, threshold, cost, gain) -> dict:
    """Profit = gain * (subscribers called) - cost * (calls made)."""
    y_true = np.asarray(y_true)
    call = np.asarray(y_prob) >= threshold
    n_calls, n_hits = int(call.sum()), int(y_true[call].sum())
    return {"threshold": float(threshold), "calls": n_calls, "subscribers_reached": n_hits,
            "share_called": n_calls / len(y_true),
            "share_subscribers_captured": n_hits / max(int(y_true.sum()), 1),
            "profit": gain * n_hits - cost * n_calls}


def profit_curve(y_true, y_prob, cost, gain, grid=None):
    grid = np.round(np.linspace(0.0, 0.8, 161), 4) if grid is None else grid
    return [profit(y_true, y_prob, t, cost, gain) for t in grid]


def best_threshold(y_true, y_prob, cost, gain):
    curve = profit_curve(y_true, y_prob, cost, gain)
    best = max(curve, key=lambda r: r["profit"])
    return best["threshold"], curve
