"""Report figures (light style, saved as PNG)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
C = ["#2E5EAA", "#E4572E", "#17A398", "#F3A712", "#7A4EAB"]


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def eda(X, y, out):
    df = X.assign(y=y)
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    y.value_counts().sort_index().rename({0: "no", 1: "yes"}).plot.bar(ax=ax[0, 0], color=C[:2])
    ax[0, 0].set_title(f"Target balance (yes = {y.mean():.1%})"); ax[0, 0].set_xlabel("")
    order = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    df.groupby("month")["y"].mean().reindex(order).plot.bar(ax=ax[0, 1], color=C[0])
    ax[0, 1].set_title("Subscription rate by call month"); ax[0, 1].set_xlabel("")
    df.groupby("poutcome")["y"].mean().sort_values().plot.barh(ax=ax[1, 0], color=C[2])
    ax[1, 0].set_title("Subscription rate by previous-campaign outcome"); ax[1, 0].set_ylabel("")
    roll = pd.Series(y.values).rolling(2000).mean()
    ax[1, 1].plot(roll.values, color=C[1])
    ax[1, 1].set_title("Rolling subscription rate vs row order (time)")
    ax[1, 1].set_xlabel("row index (file is chronological)")
    _save(fig, out / "eda_overview.png")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for k, lab in [(0, "no"), (1, "yes")]:
        ax[0].hist(np.clip(X.loc[y == k, "duration"], 0, 1500), bins=60, alpha=.6, density=True,
                   label=lab, color=C[k])
    ax[0].set_title("Call duration (s) by outcome  [LEAKY]"); ax[0].legend()
    ages = pd.cut(X["age"], [0, 25, 60, 200], labels=["<25", "25-60", ">60"], right=False)
    r = df.groupby(ages, observed=True)["y"].agg(["mean", "size"])
    ax[1].bar(r.index.astype(str), r["mean"], color=C[3])
    for i, (m, n) in enumerate(zip(r["mean"], r["size"])):
        ax[1].text(i, m + .005, f"{m:.1%}\nn={n}", ha="center", fontsize=8)
    ax[1].set_title("Subscription rate by age group (sensitive attribute)")
    ax[1].set_ylim(0, r["mean"].max() * 1.35)
    _save(fig, out / "eda_duration_age.png")


def bars(df, metric_cols, title, path, ylabel="score"):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    df[metric_cols].plot.bar(ax=ax, color=C[:len(metric_cols)], rot=0)
    ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xlabel("")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.3f", fontsize=7)
    _save(fig, path)


def pr_curves(curves: dict, y, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for (name, p), c in zip(curves.items(), C):
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, label=name, color=c)
    ax.axhline(np.mean(y), ls="--", color="grey", label=f"prevalence = {np.mean(y):.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall (validation)")
    ax.legend(fontsize=8); _save(fig, path)


def optuna_history(frames: dict, path):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for (name, df), c in zip(frames.items(), C):
        v = df["cv_average_precision_mean"].values
        ax.scatter(range(len(v)), v, s=12, color=c, alpha=.5)
        ax.plot(np.maximum.accumulate(v), color=c, label=f"{name} (best {v.max():.4f})")
    ax.set_xlabel("trial"); ax.set_ylabel("5-fold CV PR-AUC"); ax.set_title("Optuna tuning history")
    ax.legend(fontsize=8); _save(fig, path)


def reliability(tables: dict, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    for (name, t), c in zip(tables.items(), C):
        t = pd.DataFrame(t)
        ax.plot(t["mean_pred"], t["frac_pos"], "o-", color=c, label=name)
        ax.scatter(t["mean_pred"], t["frac_pos"], s=np.sqrt(t["count"]) * 3, color=c, alpha=.3)
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed fraction of subscribers")
    ax.set_title("Reliability curve (test set, 10 equal-width bins)"); ax.legend(fontsize=8)
    _save(fig, path)


def profit_plot(curve, best_t, theo_t, path):
    df = pd.DataFrame(curve)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(df["threshold"], df["profit"], color=C[0])
    ax.axvline(best_t, color=C[1], ls="--", label=f"chosen t = {best_t:.3f}")
    ax.axvline(theo_t, color=C[2], ls=":", label=f"theory C/G = {theo_t:.2f}")
    ax.set_xlabel("decision threshold on calibrated probability"); ax.set_ylabel("profit (EUR)")
    ax.set_title("Campaign profit vs threshold (validation)"); ax.legend(); _save(fig, path)


def fairness_plot(before, after, path):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    x = np.arange(len(before))
    for i, col in enumerate(["TPR", "selection_rate"]):
        ax[i].bar(x - .2, before[col], .4, label="before", color=C[0])
        ax[i].bar(x + .2, after[col], .4, label="after (group thresholds)", color=C[2])
        ax[i].set_xticks(x, before["group"]); ax[i].set_title(f"{col} by age group (test)")
        ax[i].set_ylim(0, 1.1)
        for xx, v in zip(x - .2, before[col]): ax[i].text(xx, v + .02, f"{v:.2f}", ha="center", fontsize=8)
        for xx, v in zip(x + .2, after[col]): ax[i].text(xx, v + .02, f"{v:.2f}", ha="center", fontsize=8)
    ax[0].legend(fontsize=8); _save(fig, path)
