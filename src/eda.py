"""Exploratory checks run before any modelling.

Everything we look at before trusting the data: duplicates, ranges, hidden missing values,
cardinality, outliers, links with the target, time drift. Each step prints a short line,
saves its table in outputs/results/eda/ and returns the key numbers for results.json.

Data-quality checks (duplicates, ranges, consistency) look at the whole file, because a duplicate
across train and test would be a leak. Everything that can guide a modelling choice (distributions,
outliers, links with the target, correlations) is computed on the TRAINING set only.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif

from .data import CATEGORICAL

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
VIOLET, TURQUOISE, MAGENTA, GREY = "#6D28D9", "#0284A8", "#C026D3", "#A9A3C9"


def _save_fig(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def overview(X, y):
    """Size, types and target values."""
    return {"n_rows": len(X), "n_columns": X.shape[1],
            "memory_mb": round(X.memory_usage(deep=True).sum() / 1e6, 1),
            "dtypes": X.dtypes.astype(str).value_counts().to_dict(),
            "target_values": sorted(int(v) for v in y.unique())}


def duplicates(X, y):
    """Exact copies, and identical clients with a different answer."""
    df = X.assign(y=y)
    same_x = X.duplicated(keep=False)
    conflicts = df[same_x].groupby(list(X.columns), observed=True)["y"].nunique().gt(1).sum()
    return {"exact_duplicates": int(df.duplicated().sum()),
            "duplicated_features": int(X.duplicated().sum()),
            "same_features_different_answer": int(conflicts)}


def range_checks(X):
    """Ranges written down before looking at the data."""
    return {
        "age between 18 and 95": bool(X["age"].between(18, 95).all()),
        "day between 1 and 31": bool(X["day"].between(1, 31).all()),
        "month is a real month": bool(X["month"].isin(MONTHS).all()),
        "campaign >= 1": bool((X["campaign"] >= 1).all()),
        "previous >= 0": bool((X["previous"] >= 0).all()),
        "pdays is -1 or positive": bool(((X["pdays"] == -1) | (X["pdays"] > 0)).all()),
        "duration >= 0": bool((X["duration"] >= 0).all()),
    }


def previous_campaign_consistency(X):
    """pdays = -1, previous = 0 and poutcome = unknown should all mean 'never contacted'."""
    never = X["pdays"] == -1
    table = pd.crosstab([never.rename("pdays == -1"), (X["previous"] == 0).rename("previous == 0")], X["poutcome"])
    mismatch = int(((~never) & (X["poutcome"] == "unknown")).sum() + (never & (X["poutcome"] != "unknown")).sum())
    return table, mismatch


def missing_values(X):
    """True NaN and hidden missing values ('unknown', pdays = -1)."""
    unk = pd.DataFrame({"n_unknown": (X[CATEGORICAL] == "unknown").sum(),
                        "share": (X[CATEGORICAL] == "unknown").mean()})
    unk = unk[unk["n_unknown"] > 0].sort_values("share", ascending=False)
    return int(X.isna().sum().sum()), float((X["pdays"] == -1).mean()), unk


def cardinality(X):
    rows = {}
    for c in CATEGORICAL:
        vc = X[c].value_counts(normalize=True)
        rows[c] = {"levels": X[c].nunique(), "most_frequent": vc.index[0], "most_frequent_share": vc.iloc[0],
                   "rarest": vc.index[-1], "rarest_share": vc.iloc[-1]}
    return pd.DataFrame(rows).T.sort_values("levels", ascending=False)


def numeric_profile(X, num):
    """Quantiles and outliers outside the 1.5 x IQR fences."""
    q = X[num].quantile([0, .01, .05, .25, .5, .75, .95, .99, 1]).T
    q1, q3 = X[num].quantile(.25), X[num].quantile(.75)
    iqr = q3 - q1
    out = (X[num] < q1 - 1.5 * iqr) | (X[num] > q3 + 1.5 * iqr)
    outliers = pd.DataFrame({"n_outliers": out.sum(), "share": out.mean()}).sort_values("share", ascending=False)
    return q, outliers


def rates_by_category(X, y):
    df = X.assign(y=y)
    parts = []
    for c in CATEGORICAL:
        r = df.groupby(c)["y"].agg(calls="size", rate="mean").reset_index().rename(columns={c: "level"})
        r.insert(0, "column", c)
        parts.append(r)
    return pd.concat(parts, ignore_index=True)


def _cramers_v(a, b):
    t = pd.crosstab(a, b)
    chi2 = chi2_contingency(t)[0]
    return float(np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1))))


def association(X, y, num, seed):
    """Cramér's V (categorical) or |Pearson r| (numeric), plus mutual information."""
    strength = {c: _cramers_v(X[c], y) for c in CATEGORICAL}
    strength.update({c: float(abs(np.corrcoef(X[c], y)[0, 1])) for c in num})
    enc = X.copy()
    for c in CATEGORICAL:
        enc[c] = enc[c].astype("category").cat.codes
    mi = mutual_info_classif(enc, y, discrete_features=[c in CATEGORICAL for c in X.columns], random_state=seed)
    return pd.DataFrame({"association": strength, "mutual_information": pd.Series(mi, index=X.columns)}) \
        .sort_values("mutual_information", ascending=False)


def strongest_pairs(X, num, k=6):
    sp = X[num].corr(method="spearman")
    pairs = sp.where(np.triu(np.ones(sp.shape, bool), 1)).stack().sort_values(key=abs, ascending=False)
    return pairs.head(k).rename("spearman")


def column_decisions(X):
    """Would we know this column before dialing? What do we do with it?"""
    return pd.DataFrame([
        ("duration", "length of the call, known only AFTER it ends: target leakage", "drop"),
        ("pdays", "-1 is a code for 'never', not a number", "prev_contacted + pdays_clean"),
        ("balance", "very skewed, can be negative", "signed log_balance"),
        ("campaign", "long tail", "log_campaign"),
        ("housing, loan", "two kinds of debt", "any_loan (kept as well)"),
        ("default", f"only {(X['default'] == 'yes').mean():.1%} 'yes', but legitimate", "keep"),
        ("contact, day, month", "planned before the call, but also a proxy for the period", "keep, watch in SHAP"),
    ], columns=["column", "reason", "decision"])


def _figures(X, y, num, assoc, fig_dir):
    fig, axes = plt.subplots(2, 4, figsize=(15, 6)); axes = axes.ravel()
    for a, c in zip(axes, num):
        a.hist(X[c], bins=50, color=VIOLET, alpha=.85); a.set_title(f"{c} (skew {X[c].skew():.1f})")
    axes[-1].axis("off")
    _save_fig(fig, fig_dir / "eda_numeric_distributions.png")

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.3))
    for a, c in zip(axes, ["balance", "campaign", "age", "previous"]):
        a.boxplot([X.loc[y == 0, c], X.loc[y == 1, c]], tick_labels=["no", "yes"],
                  flierprops={"markersize": 2, "alpha": .3})
        a.set_title(c)
    _save_fig(fig, fig_dir / "eda_boxplots_by_answer.png")

    s = assoc.sort_values("mutual_information")
    colors = [MAGENTA if c == "duration" else None for c in s.index]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    ax[0].barh(s.index, s["association"], color=[c or VIOLET for c in colors]); ax[0].set_title("association with the target")
    ax[1].barh(s.index, s["mutual_information"], color=[c or TURQUOISE for c in colors]); ax[1].set_title("mutual information")
    _save_fig(fig, fig_dir / "eda_association.png")


def run_eda(X, y, results_dir, fig_dir, seed=42, X_full=None, y_full=None) -> dict:
    """X, y = training set. X_full, y_full = whole file, used only for the data-quality checks."""
    out = results_dir / "eda"
    out.mkdir(parents=True, exist_ok=True)
    num = [c for c in X.columns if c not in CATEGORICAL]
    Xq, yq = (X_full, y_full) if X_full is not None else (X, y)

    ov = overview(Xq, yq)
    print(f"[EDA] whole file: {ov['n_rows']:,} rows x {ov['n_columns']} columns, {ov['memory_mb']} MB, types {ov['dtypes']}")

    dup = duplicates(Xq, yq)
    print(f"[EDA] whole file, duplicates: {dup}")

    checks = range_checks(Xq)
    bad = [k for k, ok in checks.items() if not ok]
    print(f"[EDA] whole file, range checks: {len(checks) - len(bad)}/{len(checks)} ok" + (f", failing: {bad}" if bad else ""))

    table, mismatch = previous_campaign_consistency(Xq)
    table.to_csv(out / "previous_campaign_consistency.csv")
    print(f"[EDA] whole file, pdays / previous / poutcome disagree on {mismatch} rows")
    print(f"[EDA] from here on: training set only ({len(X):,} rows), so no test row guides a choice")

    n_nan, never_share, unk = missing_values(X)
    unk.to_csv(out / "unknown_values.csv")
    print(f"[EDA] NaN: {n_nan} | pdays == -1: {never_share:.1%} | 'unknown': "
          + ", ".join(f"{c} {r.share:.1%}" for c, r in unk.iterrows()))

    card = cardinality(X)
    card.to_csv(out / "cardinality.csv")
    print(f"[EDA] categorical levels: max {card['levels'].max()}, rarest level "
          f"{card['rarest_share'].astype(float).min():.2%} ({card['rarest_share'].astype(float).idxmin()})")

    quant, outliers = numeric_profile(X, num)
    quant.to_csv(out / "numeric_quantiles.csv"); outliers.to_csv(out / "numeric_outliers_iqr.csv")
    print("[EDA] IQR outliers: " + ", ".join(f"{c} {r.share:.1%}" for c, r in outliers.iterrows()))

    rates = rates_by_category(X, y)
    rates.to_csv(out / "rate_by_category.csv", index=False)
    medians = X.assign(y=y).groupby("y")[num].median().T.rename(columns={0: "median_no", 1: "median_yes"})
    medians.to_csv(out / "numeric_median_by_answer.csv")
    pout = rates[rates.column == "poutcome"].set_index("level")["rate"]
    print(f"[EDA] subscription rate {y.mean():.1%}; after a past success {pout['success']:.1%}")

    assoc = association(X, y, num, seed)
    assoc.to_csv(out / "association_with_target.csv")
    print("[EDA] strongest links with the target: " + ", ".join(assoc.index[:5]))

    pairs = strongest_pairs(X, num)
    pairs.to_csv(out / "spearman_top_pairs.csv")
    a, b = pairs.index[0]
    print(f"[EDA] strongest pair of features: {a} / {b} (Spearman {pairs.iloc[0]:.2f})")

    first, last = float(y[: len(y) // 2].mean()), float(y[int(.9 * len(y)):].mean())
    print(f"[EDA] file in time order: rate {first:.1%} in the first half vs {last:.1%} in the last 10%")

    decisions = column_decisions(X)
    decisions.to_csv(out / "column_decisions.csv", index=False)

    _figures(X, y, num, assoc, fig_dir)

    return {"computed_on": {"quality_checks": "whole file", "everything_else": f"train only ({len(X)} rows)"},
            "overview": ov, "duplicates": dup, "range_checks": checks,
            "previous_campaign_mismatch_rows": mismatch, "nan": n_nan,
            "max_levels": int(card["levels"].max()),
            "outlier_share": outliers["share"].round(4).to_dict(),
            "association_top5": assoc.head(5).round(4).to_dict(orient="index"),
            "strongest_feature_pair": {"pair": f"{a}/{b}", "spearman": round(float(pairs.iloc[0]), 3)},
            "column_decisions": decisions.to_dict(orient="records")}
