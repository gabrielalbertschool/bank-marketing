"""Model zoo: every model is wrapped in ONE sklearn Pipeline (features -> encoding -> model)."""
from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import make_preprocessor

MODEL_NAMES = ["logreg", "random_forest", "lightgbm"]
PRETTY = {"dummy": "Dummy (prior)", "logreg": "Logistic Regression",
          "random_forest": "Random Forest", "lightgbm": "LightGBM"}


def make_estimator(name: str, params: dict | None, seed: int, class_weight="balanced"):
    params = dict(params or {})
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logreg":
        base = dict(C=1.0, l1_ratio=0.0, solver="saga", max_iter=3000)
        base.update(params)
        return LogisticRegression(class_weight=class_weight, random_state=seed, **base)
    if name == "random_forest":
        base = dict(n_estimators=300, min_samples_leaf=5, max_features="sqrt")
        base.update(params)
        return RandomForestClassifier(class_weight=class_weight, random_state=seed, n_jobs=1, **base)
    if name == "lightgbm":
        base = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample_freq=1)
        base.update(params)
        return LGBMClassifier(class_weight=class_weight, random_state=seed, n_jobs=1,
                              verbose=-1, **base)
    raise ValueError(name)


def make_pipeline(name: str, X_sample, params=None, seed: int = 42,
                  drop=("duration",), class_weight="balanced") -> Pipeline:
    """Single Pipeline = FeatureEngineer -> ColumnTransformer (scale + one-hot) -> model.

    Trees do not need scaling but it is harmless; keeping one preprocessing for every model
    makes the comparison fair and the Pipeline identical across models.
    """
    fe, ct = make_preprocessor(X_sample, drop=drop, scale=True)
    return Pipeline([("features", fe), ("preprocess", ct),
                     ("model", make_estimator(name, params, seed, class_weight))])


def suggest_params(trial, name: str) -> dict:
    """Optuna search spaces (log scale for scale-type hyper-parameters)."""
    if name == "logreg":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True),
                # l1_ratio = 0 -> Ridge (L2), 1 -> Lasso (L1), in between -> Elastic Net
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0)}
    if name == "random_forest":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                "max_depth": trial.suggest_int("max_depth", 4, 24),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.3, 0.5])}
    if name == "lightgbm":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)}
    raise ValueError(name)
