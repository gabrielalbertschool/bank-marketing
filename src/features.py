"""Feature engineering + preprocessing, all inside one sklearn Pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless, row-wise feature creation (no statistic learnt -> no leakage).

    * ``prev_contacted``  : 1 if the client was reached in a previous campaign
                            (``pdays == -1`` is a sentinel meaning "never").
    * ``pdays_clean``     : days since last contact, sentinel -1 replaced by 999
                            so that the scale is monotone ("very long ago").
    * ``log_balance``     : signed log1p of the yearly balance (very skewed, negatives exist).
    * ``log_campaign``    : log1p of the number of calls in this campaign (heavy tail).
    * ``any_loan``        : 1 if the client has a housing OR a personal loan (debt burden).
    * ``drop``            : leaky columns removed here (``duration``).
    """

    def __init__(self, drop=("duration",)):
        self.drop = drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["prev_contacted"] = (X["pdays"] != -1).astype(int)
        X["pdays_clean"] = X["pdays"].where(X["pdays"] != -1, 999)
        X["log_balance"] = np.sign(X["balance"]) * np.log1p(np.abs(X["balance"]))
        X["log_campaign"] = np.log1p(X["campaign"])
        X["any_loan"] = ((X["housing"] == "yes") | (X["loan"] == "yes")).astype(int)
        X = X.drop(columns=["pdays", "balance", "campaign"])
        X = X.drop(columns=[c for c in (self.drop or ()) if c in X.columns])
        return X

    def get_feature_names_out(self, input_features=None):
        return None


def make_preprocessor(X_sample: pd.DataFrame, drop=("duration",), scale: bool = True):
    """Return (FeatureEngineer, ColumnTransformer) built from the engineered column list."""
    fe = FeatureEngineer(drop=drop)
    cols = fe.transform(X_sample.head(5)).columns
    cat = [c for c in cols if c in CATEGORICAL]
    num = [c for c in cols if c not in CATEGORICAL]
    ct = ColumnTransformer(
        [("num", StandardScaler() if scale else "passthrough", num),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)],
        verbose_feature_names_out=False)
    return fe, ct
