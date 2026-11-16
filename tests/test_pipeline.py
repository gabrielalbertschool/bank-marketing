"""Fast unit tests of the safeguards (no training needed): run `python -m pytest tests`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data, split_data  # noqa: E402
from src.features import FeatureEngineer  # noqa: E402
from src.utils import load_config  # noqa: E402

CFG = load_config()
X, y, _ = load_data(CFG)


def test_split_is_disjoint_and_stratified():
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y, CFG)
    assert not (set(X_tr.index) & set(X_va.index) or set(X_tr.index) & set(X_te.index)
                or set(X_va.index) & set(X_te.index))
    for s in (y_tr, y_va, y_te):
        assert abs(s.mean() - y.mean()) < 0.005


def test_duration_is_removed():
    out = FeatureEngineer(drop=("duration",)).transform(X.head(50))
    assert "duration" not in out.columns
    assert {"prev_contacted", "pdays_clean", "log_balance", "log_campaign", "any_loan"} <= set(out.columns)


if __name__ == "__main__":
    test_split_is_disjoint_and_stratified(); test_duration_is_removed(); print("pipeline tests passed")
