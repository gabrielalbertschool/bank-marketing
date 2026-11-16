"""Metrics shared by every model (same metrics for all, as required)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, log_loss, precision_score, recall_score, roc_auc_score)

CV_SCORING = {"average_precision": "average_precision", "roc_auc": "roc_auc",
              "f1": "f1", "recall": "recall", "precision": "precision",
              "neg_brier": "neg_brier_score"}


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """ECE with equal-width bins on [0, 1] (definition of Coding Session 8)."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for m in range(n_bins):
        mask = idx == m
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def reliability_table(y_true, y_prob, n_bins: int = 10):
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for m in range(n_bins):
        mask = idx == m
        if mask.any():
            rows.append({"bin": f"[{bins[m]:.1f},{bins[m+1]:.1f})", "count": int(mask.sum()),
                         "mean_pred": float(y_prob[mask].mean()),
                         "frac_pos": float(y_true[mask].mean())})
    return rows


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {"roc_auc": roc_auc_score(y_true, y_prob),
            "pr_auc": average_precision_score(y_true, y_prob),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
            "brier": brier_score_loss(y_true, y_prob),
            "log_loss": log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6)),
            "ece": expected_calibration_error(y_true, y_prob),
            "threshold": threshold}


def bootstrap_ci(y_true, y_prob, metric: str, n_boot: int = 1000, seed: int = 42, threshold: float = 0.5):
    """95% percentile bootstrap interval of one metric (resampling test clients with replacement)."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y_true), len(y_true))
        if y_true[i].min() == y_true[i].max():
            continue
        vals.append(compute_metrics(y_true[i], y_prob[i], threshold)[metric])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
