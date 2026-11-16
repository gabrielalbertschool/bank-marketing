"""Optuna tuning with StratifiedKFold CV on the TRAINING set only, plus a full run log."""
from __future__ import annotations

import json
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from .evaluation import CV_SCORING
from .models import make_pipeline, suggest_params
from .utils import rel


def tune_model(name, X_train, y_train, cfg, log_path, study_storage):
    """Run ``n_trials`` Optuna (TPE) trials; each trial = 5-fold CV mean PR-AUC.

    Every trial is appended to ``log_path`` (JSON lines) with params, CV metrics (mean/std),
    duration and the artifact it maps to (Optuna study DB + trial number).
    """
    seed = cfg["seed"]
    cv = StratifiedKFold(n_splits=cfg["tuning"]["cv_folds"], shuffle=True, random_state=seed)
    drop = tuple(cfg["features"]["drop_leaky"])

    def objective(trial):
        params = suggest_params(trial, name)
        pipe = make_pipeline(name, X_train, params, seed=seed, drop=drop)
        t0 = time.time()
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=CV_SCORING, n_jobs=-1)
        metrics = {}
        for k in CV_SCORING:
            v = res[f"test_{k}"]
            if k == "neg_brier":
                k, v = "brier", -v
            metrics[f"cv_{k}_mean"] = float(np.mean(v))
            metrics[f"cv_{k}_std"] = float(np.std(v))
        for k, v in metrics.items():
            trial.set_user_attr(k, v)
        record = {"model": name, "trial": trial.number, "params": params, **metrics,
                  "fit_seconds": round(time.time() - t0, 2),
                  "artifact": f"{rel(study_storage)}::study={name}::trial={trial.number}"}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return metrics["cv_average_precision_mean"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=name, direction="maximize", storage=f"sqlite:///{study_storage}",
        load_if_exists=False, sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=cfg["tuning"]["n_trials"], show_progress_bar=False)
    return study


def study_to_frame(study) -> pd.DataFrame:
    rows = []
    for t in study.trials:
        rows.append({"trial": t.number, **{f"param_{k}": v for k, v in t.params.items()},
                     **t.user_attrs})
    return pd.DataFrame(rows)
