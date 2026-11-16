# Bank Marketing: who should the bank call?

Group project, *Applied ML: Foundations, Engineering & Evaluation* (MSA-DATI07-05).
**Team:** Eyal Toledano, Hugo Durand, Gabriel Bohbot.
**Dataset:** Option 2, Bank Marketing, `fetch_openml(data_id=1461, as_frame=True)`: 45,211 calls, 11.7% "yes".

We predict, **before** a call, the probability that a client subscribes to a term deposit. The bank can then call
only the clients worth calling. One command runs the whole project and produces every number, figure and log used in the report.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py                  # full run: about 12 min on 4 CPU cores (30 Optuna trials x 3 models)
python main.py --trials 3       # quick smoke run: about 4 min
python -m pytest tests          # unit tests + API test (the API test needs main.py to have run first)
uvicorn src.serve:app --port 8000    # prediction endpoint: POST /predict, GET /health
```

The data are downloaded with `fetch_openml(data_id=1461)`. If OpenML is unreachable, the loader uses `data/bank-full.csv`
(the same 45,211 UCI rows, `;`-separated); if that file is missing too, it downloads a public copy into `data/` automatically.

## Repository layout

```
main.py               end-to-end pipeline, stages in order (see the docstring)
config.yaml           seed, split sizes, Optuna budget, business costs, fairness groups, output paths
src/data.py           loading (OpenML 1461) + stratified 60/20/20 train/validation/test split
src/eda.py            checks before modelling: duplicates, ranges, hidden missing values, cardinality, outliers,
                      links with the target, time drift (tables in outputs/results/eda/)
src/features.py       FeatureEngineer: removes the leaky `duration`, adds 5 features; ColumnTransformer
src/models.py         3 models + Dummy baseline, one sklearn Pipeline each, Optuna search spaces
src/tuning.py         Optuna (TPE, seeded) with StratifiedKFold(5) CV on train only; writes the run log
src/evaluation.py     shared metrics (ROC-AUC, PR-AUC, F1, precision, recall, Brier, log-loss, ECE), bootstrap CI
src/business.py       campaign profit and the profit-maximising threshold
src/explain.py        SHAP global (bar, beeswarm) and local (waterfall) explanations
src/fairness.py       age-group metrics, fairness gaps, group-threshold mitigation
src/plots.py          report figures
src/serve.py          FastAPI prediction endpoint
tests/                split / leakage unit tests and API test
run_log/run_log.csv   run log: one row per Optuna trial (params, CV metrics, linked artifact)
outputs/              created by main.py, not versioned: figures/, artifacts/ (models, Optuna DB), results/results.json
```

## Method

1. **Split, then EDA**: stratified 60/20/20 split first. `src/eda.py` runs the data-quality checks on the whole file
   (no duplicates, valid ranges, consistent columns) and everything that can guide a choice on the **training set only**
   (hidden missing values `unknown` / `pdays = -1`, outliers kept and log-transformed, link of each column with the target, drift). Train is used for fitting, 5-fold CV and Optuna. Validation is used for model choice,
   the business threshold and the fairness thresholds. Test is used **once**, at the end.
2. **Leakage**: `duration` is known only after the call, so it is dropped inside the Pipeline. LightGBM CV ROC-AUC falls from 0.930 to 0.785.
3. **Features**: `prev_contacted`, `pdays_clean`, `log_balance`, `log_campaign`, `any_loan`, plus one-hot encoding and scaling.
4. **Imbalance**: `class_weight="balanced"`, measured with PR-AUC (random baseline = 0.117).
5. **Models**: Logistic Regression (Elastic Net), Random Forest and LightGBM, all compared with the same metrics.
6. **Tuning**: Optuna, 30 trials per model, objective = mean 5-fold CV PR-AUC on train. Every trial goes to `run_log/run_log.csv`.
7. **SHAP**, **calibration** (isotonic, verdict read bin by bin on the reliability curve), **cost-based threshold** and
   **fairness** audit on age with a post-processing mitigation (profit cost and bootstrap confidence intervals reported).
8. **Checks reported, never used for a choice**: the same model re-fitted on train + validation (refit check), and a temporal
   check trained on the oldest 80% of the file **without any test row** and evaluated on the newest 20%.

Why the validation set is used for three choices (model, business threshold, fairness thresholds) and why the headline model is
not re-fitted on train + validation: those thresholds must be read on predictions of a model that never saw the validation rows.
The test set stays untouched until the end; the refit check shows what a refit would change.

**API note**: `POST /predict` takes the 15 pre-call columns. `duration` is filled with 0 only because the fitted Pipeline expects
the 16 raw columns; its first step deletes `duration`, so the value is never used (`tests/test_api.py` checks it).

## Key results (untouched test set, 9,043 clients)

| | value |
|---|---|
| Selected model | LightGBM + isotonic calibration |
| ROC-AUC / PR-AUC | 0.806 / 0.461 (random PR-AUC = 0.117) |
| Brier / ECE, raw, then calibrated | 0.155 / 0.256, then 0.080 / 0.006 |
| Profit with threshold 0.11 (5 EUR per call, 50 EUR per sale) | 24,525 EUR vs 7,685 EUR when calling everyone |
| Equal-opportunity gap across age groups | 0.325, then 0.004 after mitigation (profit 23,305 EUR) |

All seeds are fixed (`seed: 42`), so two runs give identical numbers.
