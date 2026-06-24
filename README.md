# CardioSurv

**AI-Powered Cardiovascular Risk Triage & Survival Prediction**

End-to-end machine learning pipeline for cardiovascular risk classification, treatment recommendation, and two-year survival prediction. Built as the final assessment for AIT201 *Applied Machine Learning* at Xiamen University Malaysia (Academic Session 2026/04).

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/api-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Deploy](https://img.shields.io/badge/deploy-Render-46E3B7.svg)](https://render.com/)
[![Tests](https://img.shields.io/badge/tests-106%20passing-brightgreen.svg)](./tests)

---

## Overview

CardioSurv takes routine patient vital signs and produces:

1. **Risk classification** — Low / Medium / High cardiovascular risk via Random Forest (F1-macro 0.9321, AUC-OVR 0.9880 on a held-out 20% test set).
2. **Clinical routing** — rule-based assignment to either a cardiac SBRT pathway (validated by the Biologically Effective Dose formula, BED = 87.5 Gy) or a pharmacological pathway (intensity calibrated by GRACE Risk Score).
3. **Survival prediction** — per-patient two-year survival probabilities with and without the recommended intervention, using Kaplan–Meier estimators and a Cox Proportional Hazards model (C-index 0.85).

The complete system is deployed as a live web application: FastAPI backend, PostgreSQL persistence layer, and HTML/CSS/JS frontend with an interactive Three.js heart model.

---

## Architecture

```
                    ┌────────────────────┐
                    │  Patient Vitals    │
                    │ (frontend form)    │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  FastAPI Backend   │
                    │  /predict, /recom… │
                    └─────────┬──────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼───────┐    ┌────────▼────────┐    ┌──────▼───────┐
│ Part 1:       │    │ Clinical Router │    │ Part 2:      │
│ RF Classifier │───▶│ (BED + GRACE)   │───▶│ XGBoost +    │
│ → Risk class  │    │ → SBRT vs Med   │    │ Cox PH       │
└───────────────┘    └─────────────────┘    └──────┬───────┘
                                                   │
                                          ┌────────▼────────┐
                                          │ 2-year survival │
                                          │ (with vs w/o)   │
                                          └─────────────────┘
```

---

## Repository Layout

```
cardiosurv/
├── src/
│   ├── api/             # FastAPI app, schemas, 5 endpoints
│   ├── clinical/        # BED formula, GRACE score, routing layer
│   ├── data/            # Dataset download, merge, cleaning
│   ├── db/              # SQLAlchemy models, sessions, seed data
│   ├── evaluation/      # Metrics + plotting helpers
│   ├── features/        # Feature engineering (AgeBin, BP_RiskLevel, HRSI)
│   └── models/          # Part 1 classifier, Part 2 recommender + survival
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_part1_modeling.ipynb   # Risk classifier training
│   ├── 03_part2_modeling.ipynb   # Recommender + survival
│   └── CardioSurv_final.ipynb    # Self-contained end-to-end notebook
├── frontend/            # 4-page web UI (index, predict, results, history)
├── tests/               # 106 unit tests (BED, GRACE, feature functions)
├── models/              # Trained .pkl artefacts (gitignored)
├── docs/                # Report, slides, API contract, deployment runbook
├── scripts/             # Smoke test + keep-warm utilities
├── Dockerfile           # Container build
├── docker-compose.yml   # Local stack (api + postgres)
├── render.yaml          # Render.com deployment config
└── requirements.txt     # Pinned Python dependencies
```

---

## Quick Start

### Local development

```bash
# 1. Clone and install
git clone https://github.com/ayoub003184/cardiosurv.git
cd cardiosurv
pip install -r requirements.txt

# 2. Train (or download pre-trained models into ./models/)
python -m src.data.merge          # produces data/processed/merged.csv
python -m src.features.engineering # produces data/processed/features.csv
python -m src.models.part1_classifier
python -m src.models.part2_recommender

# 3. Run the API
uvicorn src.api.main:app --reload --port 8000

# 4. Open the frontend
open frontend/index.html
```

The frontend auto-detects environment — it talks to `localhost:8000` in dev and the Render URL in production.

### Docker

```bash
docker-compose up --build
```

Spins up the FastAPI service on port 8000 and a PostgreSQL container on port 5432.

### Running tests

```bash
pytest tests/ -v
```

106 tests covering BED formula edge cases, GRACE score canonical reference cases, and feature engineering monotonicity.

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness + model version check |
| `POST` | `/predict` | Risk classification from vitals |
| `POST` | `/recommend` | Treatment recommendation + survival |
| `GET` | `/history` | Patient prediction history |
| `GET` | `/patient/{id}` | Full patient record |

See [`docs/api_contract.md`](./docs/api_contract.md) for the full schema and example payloads.

---

## Dataset

The training cohort is the **harmonised merger of two public datasets**:

| Source | Rows | Contribution |
|---|---|---|
| [Kaggle Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) | 918 | Baseline vitals, ECG, exercise angina, ST slope |
| [UCI Cleveland Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) | 303 | Adds `ca` (fluoroscopy vessel count) and `thal` (thalassemia type) |
| **Merged cohort** | **1,421** | 14 clinical features for downstream modelling |

Survival times and event labels are **synthetically simulated** under clinically grounded rules (GRACE thresholds + published arrhythmia prevalence) as a proof-of-concept. Real longitudinal validation would require MIMIC-IV access (CITI credentialing pending). This limitation is documented transparently in the report.

---

## Engineered Features

Three domain-informed features augment the raw measurements:

| Feature | Formula | Clinical rationale |
|---|---|---|
| `AgeBin` | `<40 / 40–49 / 50–59 / 60–69 / 70+` | Captures non-linear age-related risk escalation |
| `BP_RiskLevel` | AHA categories (Normal → Crisis) | Maps continuous RestingBP to actionable clinical bands |
| `HeartRateStressIndex` | `MaxHR / (220 − Age)` | Normalises chronotropic capacity against age-predicted maximum; low values indicate ischaemic blunting |

`HeartRateStressIndex` ranks among the top-5 Random Forest feature importances, validating the feature engineering design.

---

## Results

### Part 1 — Risk Classifier (held-out 20% test set)

| Metric | Random Forest | XGBoost | Winner |
|---|---|---|---|
| Accuracy | **0.9614** | 0.9579 | RF |
| F1-macro | **0.9321** | 0.9251 | RF |
| AUC-OVR | **0.9880** | 0.9853 | RF |

Random Forest selected as the production model. F1-macro is the primary metric because the Medium-risk class is only 8.4% of the cohort, and macro-averaging penalises poor minority recall — the clinically costliest failure mode.

### Part 2 — Recommender + Survival

| Component | Metric | Value |
|---|---|---|
| XGBoost recommender | Accuracy | 1.00 *(deterministic routing — see report §5.3)* |
| Cox Proportional Hazards | C-index | **0.85** *(threshold ≥ 0.65 — passed)* |
| Kaplan–Meier branches | Log-rank p | < 0.05 *(significant separation)* |

---

## Limitations

Documented honestly in the report:

- Single-source cohort (n = 1,421) with no external validation
- Survival labels are synthetic — pending MIMIC-IV access
- The 1.00 recommender accuracy reflects deterministic label construction, not true clinical predictive power
- Medium-risk class recall is the weakest link despite class weighting

---

## Future Work

- Replace synthetic survival labels with MIMIC-IV longitudinal records
- External validation on Framingham / NHANES cohorts
- Bayesian hyperparameter optimisation (Optuna)
- Wearable integration (Apple Health / Fitbit) for continuous vitals
- Prospective pilot against cardiologist assessment

---

## Team

Group 4, AIT201 Applied Machine Learning, Xiamen University Malaysia (2026/04).

| Student ID | Name | Role |
|---|---|---|
| AIT2502197 | Cherfaoui Ayoub | Project lead, documentation, backend/frontend |
| AIT2504054 | Ang Jing Ru | Feature engineering, data pipeline |
| AIT2502196 | Bougacha Mohamed | ML engineer (Part 1 + Part 2 models) |
| AIT2502198 | Chiluba Chifunilo | Backend engineer, deployment |
| AIT2502012 | Law Zi Ying | Frontend developer, API scaffolding |
| AIT2504061 | Tan Guan Han | EDA, evaluation, unit testing |

---

## References

Key methodological references (full APA list in [`docs/CardioSurv_Report (1).pdf`](./docs)):

- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*.
- Cox, D. R. (1972). Regression models and life-tables. *JRSS-B*, 34(2), 187–202.
- Cuculich, P. S., et al. (2017). Noninvasive cardiac radiation for ablation of ventricular tachycardia. *NEJM*, 377(24), 2325–2336.
- Eagle, K. A., et al. (2004). GRACE risk score. *JAMA*, 291(22), 2727–2733.
- Fowler, J. F. (1989). The linear-quadratic formula and progress in fractionated radiotherapy. *Br J Radiol*, 62(740), 679–694.
- Johnson, A. E. W., et al. (2023). MIMIC-IV. *Scientific Data*, 10(1).

---

## Licence

This is an academic project. The code is released for educational reference only and is **not a medical device**. Do not use it for actual clinical decision-making.

---

## Acknowledgements

Supervised by Dr. Mas Ira Syafila Binti Mohd Hilmi Tan, School of Computing and Data Science, Xiamen University Malaysia.
