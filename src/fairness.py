"""Fairness audit on age groups + post-processing mitigation (group-specific thresholds)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def age_groups(age, bins, labels):
    return pd.cut(age, bins=bins, labels=labels, right=False).astype(str)


def group_metrics(y_true, y_prob, groups, thresholds) -> pd.DataFrame:
    """thresholds: float (same for all) or dict group -> threshold."""
    y_true, y_prob, groups = np.asarray(y_true), np.asarray(y_prob), np.asarray(groups)
    rows = []
    for g in pd.unique(groups):
        m = groups == g
        t = thresholds[g] if isinstance(thresholds, dict) else thresholds
        yt, yp = y_true[m], y_prob[m] >= t
        tp, fp = int((yp & (yt == 1)).sum()), int((yp & (yt == 0)).sum())
        pos, neg = int((yt == 1).sum()), int((yt == 0).sum())
        rows.append({"group": g, "n": int(m.sum()), "positives": pos,
                     "base_rate": yt.mean(), "threshold": t,
                     "selection_rate": yp.mean(),
                     "TPR": tp / pos if pos else np.nan,
                     "FPR": fp / neg if neg else np.nan,
                     "precision": tp / (tp + fp) if (tp + fp) else np.nan,
                     "roc_auc": roc_auc_score(yt, y_prob[m]) if 0 < pos < m.sum() else np.nan,
                     "pr_auc": average_precision_score(yt, y_prob[m]) if pos else np.nan})
    order = {"<25": 0, "25-60": 1, ">60": 2}
    return pd.DataFrame(rows).sort_values("group", key=lambda s: s.map(order)).reset_index(drop=True)


def gaps(df: pd.DataFrame) -> dict:
    """Equal opportunity difference = max-min TPR; also DP and FPR differences."""
    return {"equal_opportunity_diff": float(df["TPR"].max() - df["TPR"].min()),
            "demographic_parity_diff": float(df["selection_rate"].max() - df["selection_rate"].min()),
            "fpr_diff": float(df["FPR"].max() - df["FPR"].min())}


def equal_opportunity_thresholds(y_true, y_prob, groups, target_tpr):
    """For each group, the highest threshold whose TPR on VALIDATION reaches target_tpr."""
    y_true, y_prob, groups = np.asarray(y_true), np.asarray(y_prob), np.asarray(groups)
    out = {}
    for g in pd.unique(groups):
        m = groups == g
        scores = np.sort(y_prob[m & (y_true == 1)])[::-1]
        k = int(np.ceil(target_tpr * len(scores)))      # number of positives to catch
        out[g] = float(scores[max(k - 1, 0)])
    return out
