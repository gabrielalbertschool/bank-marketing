"""Smoke tests: run `python -m pytest tests` (or `python tests/test_api.py`) after `python main.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from src.serve import app  # noqa: E402

CLIENT = {"age": 35, "job": "management", "marital": "married", "education": "tertiary",
          "default": "no", "balance": 1500, "housing": "no", "loan": "no", "contact": "cellular",
          "day": 15, "month": "oct", "campaign": 1, "pdays": 90, "previous": 2, "poutcome": "success"}


def test_health():
    r = TestClient(app).get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict():
    r = TestClient(app).post("/predict", json=CLIENT)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability_subscribe"] <= 1.0
    assert isinstance(body["call"], bool)
    print(body)


if __name__ == "__main__":
    test_health(); test_predict(); print("API smoke tests passed")
