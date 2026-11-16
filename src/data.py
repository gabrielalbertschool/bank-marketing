"""Data loading and honest train / validation / test split."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import ROOT

# OpenML 1461 stores the 16 UCI attributes as V1..V16 (same order as the UCI file)
UCI_NAMES = ["age", "job", "marital", "education", "default", "balance", "housing",
             "loan", "contact", "day", "month", "duration", "campaign", "pdays",
             "previous", "poutcome"]
CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "poutcome"]


def _from_openml(data_id: int) -> tuple[pd.DataFrame, pd.Series]:
    from sklearn.datasets import fetch_openml
    bank = fetch_openml(data_id=data_id, as_frame=True)
    X, y = bank.data.copy(), bank.target
    X.columns = UCI_NAMES
    y = (y.astype(str) == "2").astype(int)          # OpenML codes: 1 = no, 2 = yes
    return X, y


# Public copy of the same UCI "bank-full" file (45,211 rows), used only if OpenML is unreachable
MIRROR_URL = "https://raw.githubusercontent.com/pycaret/datasets/main/data/common/bank.csv"


def _download_mirror(dest) -> None:
    """Save the public copy in the UCI format (';' separator, target column 'y')."""
    df = pd.read_csv(MIRROR_URL).rename(columns={"deposit": "y"})
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, sep=";", index=False)


def _from_local(path) -> tuple[pd.DataFrame, pd.Series]:
    if not (ROOT / path).exists():
        print(f"[data] {path} not found; downloading the public copy of bank-full.csv.")
        _download_mirror(ROOT / path)
    df = pd.read_csv(ROOT / path, sep=";")
    y = (df.pop("y") == "yes").astype(int)
    return df[UCI_NAMES], y


def load_data(cfg: dict) -> tuple[pd.DataFrame, pd.Series, str]:
    """Load Bank Marketing. Tries OpenML first, falls back to the identical UCI csv."""
    try:
        X, y = _from_openml(cfg["data"]["openml_id"])
        source = f"fetch_openml(data_id={cfg['data']['openml_id']})"
    except Exception as exc:  # network blocked / offline
        print(f"[data] OpenML unavailable ({type(exc).__name__}); using local UCI copy.")
        X, y = _from_local(cfg["data"]["local_fallback"])
        source = f"local {cfg['data']['local_fallback']} (UCI bank-full, same 45,211 rows)"
    for c in CATEGORICAL:
        X[c] = X[c].astype(str)
    for c in [c for c in UCI_NAMES if c not in CATEGORICAL]:
        X[c] = pd.to_numeric(X[c])
    y.name = "subscribed"
    return X.reset_index(drop=True), y.reset_index(drop=True), source


def split_data(X, y, cfg: dict):
    """Stratified 60 / 20 / 20 split. The test set is touched once, at the end."""
    seed = cfg["seed"]
    test_size, val_size = cfg["data"]["test_size"], cfg["data"]["val_size"]
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_size / (1 - test_size), stratify=y_tmp, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test
