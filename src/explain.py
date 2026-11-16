"""SHAP explanations (global + local) on the final tuned pipeline."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def shap_values_for(pipe, X):
    """Return (Explanation, transformed frame) for the model step of a fitted Pipeline."""
    Xt = pipe[:-1].transform(X)
    names = list(pipe.named_steps["preprocess"].get_feature_names_out())
    Xt = pd.DataFrame(Xt, columns=names, index=X.index)
    model = pipe.named_steps["model"]
    if hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, Xt)
    else:
        explainer = shap.TreeExplainer(model)
    sv = explainer(Xt)
    if sv.values.ndim == 3:            # (n, features, classes) for RandomForest
        sv = sv[:, :, 1]
    return sv, Xt


def global_plots(sv, out_dir, max_display=15):
    plt.figure()
    shap.plots.bar(sv, max_display=max_display, show=False)
    plt.title("Global importance: mean |SHAP| (log-odds)")
    plt.tight_layout(); plt.savefig(out_dir / "shap_global_bar.png", dpi=150, bbox_inches="tight"); plt.close("all")
    plt.figure()
    shap.plots.beeswarm(sv, max_display=max_display, show=False)
    plt.tight_layout(); plt.savefig(out_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close("all")
    imp = pd.Series(np.abs(sv.values).mean(0), index=sv.feature_names).sort_values(ascending=False)
    return imp


def local_plots(sv, cases: dict, out_dir):
    """cases: label -> positional index in sv. One waterfall per case + top-5 drivers."""
    summary = {}
    for label, i in cases.items():
        plt.figure()
        shap.plots.waterfall(sv[i], max_display=10, show=False)
        plt.title(label)
        plt.tight_layout(); plt.savefig(out_dir / f"shap_local_{label}.png", dpi=150, bbox_inches="tight"); plt.close("all")
        contrib = pd.Series(sv.values[i], index=sv.feature_names)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:5]
        summary[label] = {"base_value": float(sv.base_values[i]),
                          "logit": float(sv.base_values[i] + sv.values[i].sum()),
                          "top_contributions": {k: float(v) for k, v in top.items()}}
    return summary
