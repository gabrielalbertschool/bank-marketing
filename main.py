"""End-to-end entry point: python main.py  (add --trials N for a quicker run).

Stages: data -> split -> EDA (train only) -> leakage check -> imbalance ablation -> baselines -> Optuna tuning
(3 models) -> model selection -> calibration -> business threshold -> test evaluation (once)
-> refit check on train+val -> SHAP -> fairness audit + mitigation -> temporal robustness -> save model for serving.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_validate

from src import plots
from src.business import best_threshold, profit
from src.data import CATEGORICAL, load_data, split_data
from src.eda import run_eda
from src.evaluation import CV_SCORING, bootstrap_ci, compute_metrics, reliability_table
from src.explain import global_plots, local_plots, shap_values_for
from src.fairness import age_groups, equal_opportunity_thresholds, gaps, group_metrics
from src.models import MODEL_NAMES, PRETTY, make_pipeline
from src.tuning import study_to_frame, tune_model
from src.utils import load_config, rel, save_json, set_seed

warnings.filterwarnings("ignore", category=UserWarning)


def cv_scores(pipe, X, y, cv):
    r = cross_validate(pipe, X, y, cv=cv, scoring=CV_SCORING, n_jobs=-1)
    out = {k: float(np.mean(r[f"test_{k}"])) for k in CV_SCORING}
    out["brier"] = -out.pop("neg_brier")
    return out


def main(n_trials=None):
    t_start = time.time()
    cfg = load_config()
    if n_trials:
        cfg["tuning"]["n_trials"] = n_trials
    seed = cfg["seed"]; set_seed(seed)
    FIG, RES, ART, LOG = (cfg["paths"][k] for k in ("figures", "results", "artifacts", "run_log"))
    (ART / "models").mkdir(exist_ok=True)
    R = {"config": {k: v for k, v in cfg.items() if k != "paths"}}
    drop = tuple(cfg["features"]["drop_leaky"])
    cost, gain = cfg["business"]["cost_per_call"], cfg["business"]["gain_per_subscription"]
    cv = StratifiedKFold(cfg["tuning"]["cv_folds"], shuffle=True, random_state=seed)

    # ---------------- 1. data + EDA ----------------
    X, y, source = load_data(cfg)
    print(f"[1] data: {X.shape} from {source}; positive rate {y.mean():.4f}")
    R["data"] = {"source": source, "n_rows": len(X), "n_features": X.shape[1],
                 "positive_rate": float(y.mean()), "n_positive": int(y.sum()),
                 "missing_nan": int(X.isna().sum().sum())}

    # split FIRST: every statistic that can guide a choice is computed on train only
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y, cfg)
    R["split"] = {n: {"n": len(s), "positive_rate": float(s.mean())}
                  for n, s in [("train", y_tr), ("val", y_va), ("test", y_te)]}
    print(f"[1] split train/val/test = {len(X_tr)}/{len(X_va)}/{len(X_te)}")

    # EDA on train only (rows kept in file order, so the drift along the file is still visible)
    tr = X_tr.sort_index(); ytr = y_tr.loc[tr.index]
    unknown = {c: float((tr[c] == "unknown").mean()) for c in CATEGORICAL}
    R["data"].update({"eda_on": "train only",
                      "unknown_share": {k: v for k, v in unknown.items() if v > 0},
                      "pdays_never_contacted_share": float((tr["pdays"] == -1).mean()),
                      "duration_median_by_class": tr.groupby(ytr)["duration"].median().to_dict(),
                      "rate_by_poutcome": tr.assign(y=ytr).groupby("poutcome")["y"].mean().to_dict(),
                      "rate_first_half_rows": float(ytr.iloc[: len(ytr) // 2].mean()),
                      "rate_last_10pct_rows": float(ytr.iloc[int(.9 * len(ytr)):].mean())})
    plots.eda(tr, ytr, FIG)
    R["eda"] = run_eda(tr, ytr, RES, FIG, seed, X_full=X, y_full=y)

    # ---------------- 2. leakage: duration ----------------
    leak = {}
    for name in ["logreg", "lightgbm"]:
        for label, d in [("with_duration", ()), ("without_duration", drop)]:
            leak[f"{name}|{label}"] = cv_scores(make_pipeline(name, X_tr, seed=seed, drop=d), X_tr, y_tr, cv)
    R["leakage"] = leak
    ld = pd.DataFrame(leak).T[["roc_auc", "average_precision"]]
    ld.index = [i.replace("|", "\n").replace("logreg", "LogReg").replace("lightgbm", "LightGBM") for i in ld.index]
    plots.bars(ld, ["roc_auc", "average_precision"], "Leakage check: 5-fold CV on train, with vs without `duration`",
               FIG / "leakage_duration.png")
    print("[2] leakage:\n", pd.DataFrame(leak).T[["roc_auc", "average_precision"]].round(4))

    # ---------------- 3. class imbalance ablation ----------------
    imb = {}
    for name in ["logreg", "lightgbm"]:
        for cw in [None, "balanced"]:
            imb[f"{name}|{cw or 'none'}"] = cv_scores(
                make_pipeline(name, X_tr, seed=seed, drop=drop, class_weight=cw), X_tr, y_tr, cv)
    R["imbalance"] = imb
    print("[3] imbalance:\n", pd.DataFrame(imb).T[["average_precision", "recall", "precision", "f1", "brier"]].round(4))

    # ---------------- 4. baselines (default hyper-params) on validation ----------------
    base = {}
    for name in ["dummy"] + MODEL_NAMES:
        p = make_pipeline(name, X_tr, seed=seed, drop=drop).fit(X_tr, y_tr)
        base[name] = compute_metrics(y_va, p.predict_proba(X_va)[:, 1])
    R["baseline_val"] = base

    # ---------------- 5. Optuna tuning + run log ----------------
    log_path = ART / "run_log.jsonl"; log_path.unlink(missing_ok=True)
    db = ART / "optuna_studies.db"; db.unlink(missing_ok=True)
    tuned, frames, val_probs, R["tuning"] = {}, {}, {}, {}
    for name in MODEL_NAMES:
        t0 = time.time()
        study = tune_model(name, X_tr, y_tr, cfg, log_path, db)
        frames[PRETTY[name]] = study_to_frame(study)
        pipe = make_pipeline(name, X_tr, study.best_params, seed=seed, drop=drop).fit(X_tr, y_tr)
        mpath = ART / "models" / f"{name}_best.joblib"; joblib.dump(pipe, mpath)
        tuned[name] = pipe
        val_probs[name] = pipe.predict_proba(X_va)[:, 1]
        R["tuning"][name] = {"best_params": study.best_params, "best_trial": study.best_trial.number,
                             "best_cv": {k: v for k, v in study.best_trial.user_attrs.items()},
                             "n_trials": len(study.trials), "minutes": round((time.time() - t0) / 60, 2),
                             "model_file": rel(mpath),
                             "val": compute_metrics(y_va, val_probs[name])}
        print(f"[5] {name}: best CV PR-AUC {study.best_value:.4f} | val PR-AUC "
              f"{R['tuning'][name]['val']['pr_auc']:.4f} ({R['tuning'][name]['minutes']} min)")
    # flat CSV run log with artifact column (best trials point to the saved model file)
    log = pd.read_json(log_path, lines=True)
    params = pd.json_normalize(log["params"]).add_prefix("param_")
    log = pd.concat([log.drop(columns="params"), params], axis=1)
    for name in MODEL_NAMES:
        bt = R["tuning"][name]["best_trial"]
        m = (log["model"] == name) & (log["trial"] == bt)
        log.loc[m, "artifact"] = log.loc[m, "artifact"] + f" | {R['tuning'][name]['model_file']}"
        log["is_best"] = log.get("is_best", False) | m
    log["plot"] = rel(FIG / "optuna_history.png")
    log.to_csv(LOG / "run_log.csv", index=False)
    print(f"[5] run log: {len(log)} trials -> {rel(LOG / 'run_log.csv')}")
    plots.optuna_history(frames, FIG / "optuna_history.png")
    plots.pr_curves({PRETTY[n]: val_probs[n] for n in MODEL_NAMES}, y_va, FIG / "pr_curves_val.png")
    comp = pd.DataFrame({PRETTY[n]: R["tuning"][n]["val"] for n in MODEL_NAMES}).T
    plots.bars(comp, ["roc_auc", "pr_auc", "f1", "recall"], "Tuned models on validation (threshold 0.5 for F1/recall)",
               FIG / "model_comparison.png")

    # ---------------- 6. selection + calibration ----------------
    best = max(MODEL_NAMES, key=lambda n: R["tuning"][n]["val"]["pr_auc"])
    print(f"[6] selected model: {best}")
    raw = tuned[best]
    cal = CalibratedClassifierCV(clone(raw), method="isotonic", cv=cv, n_jobs=-1).fit(X_tr, y_tr)
    p_va_raw, p_va_cal = raw.predict_proba(X_va)[:, 1], cal.predict_proba(X_va)[:, 1]
    R["selected_model"] = best
    R["calibration_val"] = {"raw": compute_metrics(y_va, p_va_raw), "isotonic": compute_metrics(y_va, p_va_cal)}

    # ---------------- 7. business threshold (validation) ----------------
    t_star, curve = best_threshold(y_va, p_va_cal, cost, gain)
    plots.profit_plot(curve, t_star, cost / gain, FIG / "profit_curve.png")
    R["business"] = {"cost_per_call": cost, "gain_per_subscription": gain,
                     "theoretical_threshold": cost / gain, "chosen_threshold": t_star,
                     "val_at_chosen": profit(y_va, p_va_cal, t_star, cost, gain)}

    # ---------------- 8. FINAL test evaluation (test used once from here) ----------------
    p_te_raw, p_te = raw.predict_proba(X_te)[:, 1], cal.predict_proba(X_te)[:, 1]
    R["test"] = {"raw": compute_metrics(y_te, p_te_raw), "calibrated": compute_metrics(y_te, p_te),
                 "calibrated_at_business_t": compute_metrics(y_te, p_te, t_star)}
    R["test_all_models"] = {n: compute_metrics(y_te, tuned[n].predict_proba(X_te)[:, 1]) for n in MODEL_NAMES}
    rt_raw, rt_cal = reliability_table(y_te, p_te_raw), reliability_table(y_te, p_te)
    def verdict(table, min_count=30):
        """Bin-by-bin reading of the reliability curve (only bins with at least `min_count` clients)."""
        rows = [r for r in table if r["count"] >= min_count]
        n = sum(r["count"] for r in rows)
        gaps_ = [r["mean_pred"] - r["frac_pos"] for r in rows]
        over, under = sum(g > 0.02 for g in gaps_), sum(g < -0.02 for g in gaps_)
        weighted = sum(r["count"] * abs(g) for r, g in zip(rows, gaps_)) / n
        if over > len(rows) / 2:
            label = "over-confident (probabilities too high)"
        elif under > len(rows) / 2:
            label = "under-confident (probabilities too low)"
        else:
            label = "well calibrated"
        return {"label": label, "bins_used": len(rows), "bins_over": int(over), "bins_under": int(under),
                "weighted_abs_gap": float(weighted), "max_abs_gap": float(max(abs(g) for g in gaps_))}
    R["test_ci95"] = {k: bootstrap_ci(y_te, p_te, k, seed=seed) for k in ("roc_auc", "pr_auc", "brier")}
    v_raw, v_cal = verdict(rt_raw), verdict(rt_cal)
    R["calibration_test"] = {"verdict_raw": v_raw["label"], "verdict_calibrated": v_cal["label"],
                             "bins_raw": v_raw, "bins_calibrated": v_cal,
                             "raw_table": rt_raw, "cal_table": rt_cal,
                             "raw_mean_pred": float(p_te_raw.mean()), "cal_mean_pred": float(p_te.mean()),
                             "observed_rate": float(y_te.mean())}
    plots.reliability({f"{PRETTY[best]} raw (class-weighted)": rt_raw,
                       f"{PRETTY[best]} + isotonic": rt_cal}, FIG / "reliability_test.png")
    R["business"]["test"] = {"model": profit(y_te, p_te, t_star, cost, gain),
                             "call_everyone": profit(y_te, np.ones(len(y_te)), 0.5, cost, gain),
                             "model_t_0.5_raw": profit(y_te, p_te_raw, 0.5, cost, gain)}
    sens = []
    for c_ in [2, 5, 10]:
        for g_ in [30, 50, 100]:
            tt, _ = best_threshold(y_va, p_va_cal, c_, g_)
            m_ = profit(y_te, p_te, tt, c_, g_); a_ = profit(y_te, np.ones(len(y_te)), .5, c_, g_)
            sens.append({"cost": c_, "gain": g_, "threshold": tt, "profit_model": m_["profit"],
                         "profit_call_all": a_["profit"], "share_called": m_["share_called"]})
    R["business"]["sensitivity"] = sens
    print(f"[8] test: {json.dumps({k: round(v, 4) for k, v in R['test']['calibrated'].items()})}")
    print(f"[8] profit test: model {R['business']['test']['model']['profit']:.0f} EUR vs call-all "
          f"{R['business']['test']['call_everyone']['profit']:.0f} EUR")

    # refit check: same hyper-parameters, calibrated model re-fitted on train + validation.
    # Not the headline model: the business and fairness thresholds were chosen on validation
    # predictions, which needs a model that never saw validation. This only measures what we lose.
    X_tv, y_tv = pd.concat([X_tr, X_va]), pd.concat([y_tr, y_va])
    cal_tv = CalibratedClassifierCV(clone(raw), method="isotonic", cv=cv, n_jobs=-1).fit(X_tv, y_tv)
    p_te_tv = cal_tv.predict_proba(X_te)[:, 1]
    R["refit_check"] = {"train_rows": len(X_tv), **compute_metrics(y_te, p_te_tv),
                        "profit_at_t": profit(y_te, p_te_tv, t_star, cost, gain)["profit"]}
    print(f"[8] refit on train+val: test PR-AUC {R['refit_check']['pr_auc']:.4f} "
          f"(headline {R['test']['calibrated']['pr_auc']:.4f})")

    # ---------------- 9. SHAP ----------------
    Xs = X_te.sample(2000, random_state=seed)
    sv, _ = shap_values_for(raw, Xs)
    imp = global_plots(sv, FIG)
    R["n_model_features"] = len(sv.feature_names)
    ps = cal.predict_proba(Xs)[:, 1]; ys = y_te.loc[Xs.index].values; pred = ps >= t_star
    pick = lambda mask, key: int(np.where(mask)[0][np.argmax(key[mask])])
    cases = {"TP_high": pick(pred & (ys == 1), ps), "FP": pick(pred & (ys == 0), ps),
             "FN": pick(~pred & (ys == 1), -ps), "TN_low": pick(~pred & (ys == 0), -ps)}
    loc = local_plots(sv, cases, FIG)
    for k, i in cases.items():
        loc[k].update({"calibrated_proba": float(ps[i]), "true_label": int(ys[i]),
                       "client": {c: (v.item() if hasattr(v, "item") else v)
                                  for c, v in Xs.iloc[i].drop("duration").items()}})
    R["shap"] = {"global_top15": imp.head(15).to_dict(), "local": loc}

    # ---------------- 10. fairness (age) ----------------
    fb, fl = cfg["fairness"]["bins"], cfg["fairness"]["labels"]
    g_va, g_te = age_groups(X_va["age"], fb, fl), age_groups(X_te["age"], fb, fl)
    before = group_metrics(y_te, p_te, g_te, t_star)
    target = R["test"]["calibrated_at_business_t"]["recall"]
    val_overall_tpr = compute_metrics(y_va, p_va_cal, t_star)["recall"]
    thr = equal_opportunity_thresholds(y_va, p_va_cal, g_va, val_overall_tpr)
    after = group_metrics(y_te, p_te, g_te, thr)
    t_vec = g_te.map(thr).values
    dec_after = (p_te >= t_vec)
    prof_after = gain * int((dec_after & (y_te.values == 1)).sum()) - cost * int(dec_after.sum())
    R["fairness"] = {"definition": "equal opportunity (TPR parity across age groups)",
                     "val_group_metrics_before": group_metrics(y_va, p_va_cal, g_va, t_star).to_dict("records"),
                     "group_thresholds": thr, "target_tpr_val": val_overall_tpr,
                     "before": before.to_dict("records"), "after": after.to_dict("records"),
                     "gaps_before": gaps(before), "gaps_after": gaps(after),
                     "profit_before": R["business"]["test"]["model"]["profit"], "profit_after": prof_after,
                     "recall_before": target, "recall_after": float(dec_after[y_te.values == 1].mean()),
                     "precision_before": R["test"]["calibrated_at_business_t"]["precision"],
                     "precision_after": float(y_te.values[dec_after].mean()),
                     "calls_before": int((p_te >= t_star).sum()), "calls_after": int(dec_after.sum())}
    # bootstrap 95% CI of the equal-opportunity gap on test (small groups -> wide intervals)
    rng = np.random.default_rng(seed); yv, gv = y_te.values, g_te.values
    eo_b, eo_a = [], []
    for _ in range(1000):
        i = rng.integers(0, len(yv), len(yv))
        tb = pd.Series((p_te[i] >= t_star)[yv[i] == 1]).groupby(gv[i][yv[i] == 1]).mean()
        ta = pd.Series(dec_after[i][yv[i] == 1]).groupby(gv[i][yv[i] == 1]).mean()
        eo_b.append(tb.max() - tb.min()); eo_a.append(ta.max() - ta.min())
    dec_before = p_te >= t_star
    cost_b = []
    for _ in range(1000):
        i = rng.integers(0, len(yv), len(yv))
        pb = gain * (dec_before[i] & (yv[i] == 1)).sum() - cost * dec_before[i].sum()
        pa = gain * (dec_after[i] & (yv[i] == 1)).sum() - cost * dec_after[i].sum()
        cost_b.append(pb - pa)
    R["fairness"]["profit_cost"] = float(R["fairness"]["profit_before"] - prof_after)
    R["fairness"]["profit_cost_share"] = float(R["fairness"]["profit_cost"] / R["fairness"]["profit_before"])
    R["fairness"]["profit_cost_ci95"] = [float(np.percentile(cost_b, 2.5)), float(np.percentile(cost_b, 97.5))]
    R["fairness"]["test_group_sizes"] = before.set_index("group")[["n", "positives"]].to_dict("index")
    R["fairness"]["eo_gap_ci95_before"] = [float(np.percentile(eo_b, 2.5)), float(np.percentile(eo_b, 97.5))]
    R["fairness"]["eo_gap_ci95_after"] = [float(np.percentile(eo_a, 2.5)), float(np.percentile(eo_a, 97.5))]
    plots.fairness_plot(before, after, FIG / "fairness_age.png")
    print("[10] fairness before:\n", before.round(3), "\n after:\n", after.round(3))

    # ---------------- 11. temporal robustness (extra, not used for any choice) ----------------
    # train on the oldest 80% of the file WITHOUT any row of our test set, evaluate on the newest 20%
    cut = int(.8 * len(X))
    old = X.index[:cut].difference(X_te.index)
    new = X.index[cut:]
    tp = make_pipeline(best, X, R["tuning"][best]["best_params"], seed=seed, drop=drop).fit(X.loc[old], y.loc[old])
    R["temporal_check"] = {"train_rows": len(old), "test_rows": len(new),
                           "train_positive_rate": float(y.loc[old].mean()),
                           "test_positive_rate": float(y.loc[new].mean()),
                           **compute_metrics(y.loc[new], tp.predict_proba(X.loc[new])[:, 1])}
    print(f"[11] temporal check: ROC-AUC {R['temporal_check']['roc_auc']:.3f} on the newest 20% "
          f"(trained on {len(old)} older rows, none from the test set)")

    # ---------------- 12. save for serving ----------------
    joblib.dump(cal, ART / "final_model.joblib")
    save_json({"model": best, "threshold": t_star, "cost_per_call": cost, "gain_per_subscription": gain},
              ART / "decision.json")
    R["runtime_minutes"] = round((time.time() - t_start) / 60, 2)
    save_json(R, RES / "results.json")
    print(f"[done] results -> {rel(RES / 'results.json')} in {R['runtime_minutes']} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=None, help="override Optuna n_trials (quick test)")
    main(ap.parse_args().trials)
