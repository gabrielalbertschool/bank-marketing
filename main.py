"""End-to-end entry point: python main.py  (add --trials N for a quicker run).

Stages: EDA -> split -> leakage check -> imbalance ablation -> baselines -> Optuna tuning
(3 models) -> model selection -> calibration -> business threshold -> test evaluation (once)
-> SHAP -> fairness audit + mitigation -> temporal robustness -> save model for serving.
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
    unknown = {c: float((X[c] == "unknown").mean()) for c in CATEGORICAL}
    R["data"] = {"source": source, "n_rows": len(X), "n_features": X.shape[1],
                 "positive_rate": float(y.mean()), "n_positive": int(y.sum()),
                 "missing_nan": int(X.isna().sum().sum()),
                 "unknown_share": {k: v for k, v in unknown.items() if v > 0},
                 "pdays_never_contacted_share": float((X["pdays"] == -1).mean()),
                 "duration_median_by_class": X.groupby(y)["duration"].median().to_dict(),
                 "rate_by_poutcome": X.assign(y=y).groupby("poutcome")["y"].mean().to_dict(),
                 "rate_first_half_rows": float(y[: len(y) // 2].mean()),
                 "rate_last_10pct_rows": float(y[int(.9 * len(y)):].mean())}
    plots.eda(X, y, FIG)
    R["eda"] = run_eda(X, y, RES, FIG, seed)

    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y, cfg)
    R["split"] = {n: {"n": len(s), "positive_rate": float(s.mean())}
                  for n, s in [("train", y_tr), ("val", y_va), ("test", y_te)]}
    print(f"[1] split train/val/test = {len(X_tr)}/{len(X_va)}/{len(X_te)}")

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
    def verdict(p):  # over-confident = predicts more "yes" than observed in the bins that matter
        gap = float(np.mean(p) - y_te.mean())
        return "over-confident (probabilities too high)" if gap > 0.01 else (
            "under-confident (probabilities too low)" if gap < -0.01 else "well calibrated on average")
    R["test_ci95"] = {k: bootstrap_ci(y_te, p_te, k, seed=seed) for k in ("roc_auc", "pr_auc", "brier")}
    R["calibration_test"] = {"verdict_raw": verdict(p_te_raw), "verdict_calibrated": verdict(p_te),
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
    R["fairness"]["eo_gap_ci95_before"] = [float(np.percentile(eo_b, 2.5)), float(np.percentile(eo_b, 97.5))]
    R["fairness"]["eo_gap_ci95_after"] = [float(np.percentile(eo_a, 2.5)), float(np.percentile(eo_a, 97.5))]
    plots.fairness_plot(before, after, FIG / "fairness_age.png")
    print("[10] fairness before:\n", before.round(3), "\n after:\n", after.round(3))

    # ---------------- 11. temporal robustness (extra, not used for any choice) ----------------
    cut = int(.8 * len(X))
    tp = make_pipeline(best, X, R["tuning"][best]["best_params"], seed=seed, drop=drop).fit(X.iloc[:cut], y.iloc[:cut])
    R["temporal_check"] = {"train_rows": cut, "test_rows": len(X) - cut,
                           "test_positive_rate": float(y.iloc[cut:].mean()),
                           **compute_metrics(y.iloc[cut:], tp.predict_proba(X.iloc[cut:])[:, 1])}

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
