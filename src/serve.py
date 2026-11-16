"""Prediction endpoint (FastAPI).  Run:  uvicorn src.serve:app --port 8000

POST /predict with one client record (raw columns, `duration` NOT needed) ->
calibrated subscription probability + call / do-not-call decision at the business threshold.
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from .data import UCI_NAMES
from .utils import ROOT

MODEL = joblib.load(ROOT / "outputs" / "artifacts" / "final_model.joblib")
DECISION = json.loads((ROOT / "outputs" / "artifacts" / "decision.json").read_text())

app = FastAPI(title="Bank Marketing - who should we call?")


class Client(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: float
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str


@app.get("/health")
def health():
    return {"status": "ok", "model": DECISION["model"], "threshold": DECISION["threshold"]}


@app.post("/predict")
def predict(client: Client):
    row = client.model_dump()
    # `duration` (length of the call) is only known AFTER the call, so the client never sends it.
    # The fitted Pipeline still expects the 16 raw columns in the same order, and its first step
    # (FeatureEngineer) deletes `duration` before anything else. So we fill it with 0: the value
    # is never read, and changing it does not change the prediction (checked in tests/test_api.py).
    row["duration"] = 0
    X = pd.DataFrame([row])[UCI_NAMES]
    p = float(MODEL.predict_proba(X)[0, 1])
    t = DECISION["threshold"]
    return {"probability_subscribe": round(p, 4), "threshold": t, "call": p >= t,
            "expected_value_eur": round(p * DECISION["gain_per_subscription"] - DECISION["cost_per_call"], 2)}
