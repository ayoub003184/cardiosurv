"""
src/api/main.py
---------------
CardioSurv API  —  mock scaffold (Task 3).

All 5 endpoints from docs/api_contract.md are implemented here.
Every response is hardcoded / computed from request flags — no database,
no real model. The `mock: true` field in every response body signals to
consumers that Chiluba's Task 4 has not yet replaced these values.

Run with:
    uvicorn src.api.main:app --reload

Then open http://localhost:8000/docs to exercise every endpoint in Swagger UI.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    HealthResponse,
    ModelVersions,
    PatientVitalsRequest,
    PredictResponse,
    RiskProbabilities,
    RecommendRequest,
    RecommendResponse,
    HistoryItem,
    HistoryResponse,
    PatientFullResponse,
    ErrorResponse,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

_START_TIME = time.time()

app = FastAPI(
    title="CardioSurv API",
    version="v1",
    description=(
        "Cardiovascular risk classification and intervention recommendation API.\n\n"
        "> **Mock scaffold** — all responses are hardcoded. "
        "Real model integration is Task 4 (Chiluba).\n\n"
        "Source of truth: `docs/api_contract.md`"
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",   # common React dev server
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Mock data helpers
# ---------------------------------------------------------------------------

def _mock_history_records(n: int = 10) -> list[HistoryItem]:
    """Generate `n` deterministic hardcoded history records."""
    templates = [
        # (risk, branch, intervention, confidence, survival_with)
        ("High",   "SBRT",       "Cardiac Radioablation (SBRT)",    0.91, 0.81),
        ("Medium", "Medication",  "Medication Intensity Calibration", 0.78, 0.88),
        ("Low",    "Medication",  "Medication Intensity Calibration", 0.85, 0.94),
        ("High",   "SBRT",       "Cardiac Radioablation (SBRT)",    0.93, 0.76),
        ("Medium", "SBRT",       "Cardiac Radioablation (SBRT)",    0.80, 0.83),
        ("Low",    "Medication",  "Medication Intensity Calibration", 0.90, 0.96),
        ("High",   "SBRT",       "Cardiac Radioablation (SBRT)",    0.88, 0.79),
        ("Medium", "Medication",  "Medication Intensity Calibration", 0.74, 0.85),
        ("Low",    "Medication",  "Medication Intensity Calibration", 0.92, 0.95),
        ("High",   "SBRT",       "Cardiac Radioablation (SBRT)",    0.95, 0.82),
    ]
    records = []
    for i in range(min(n, len(templates))):
        risk, branch, itype, conf, surv = templates[i]
        records.append(HistoryItem(
            record_id=f"REC-{i+1:03d}",
            patient_id=f"P{i+1:03d}",
            timestamp=datetime(2025, 11, i + 1, 9, 14, 0, tzinfo=timezone.utc),
            risk_category=risk,
            confidence=conf,
            branch=branch,
            intervention_type=itype,
            survival_with=surv,
        ))
    return records


# Pre-build all 10 so GET /history is instant
_ALL_HISTORY = _mock_history_records(10)

# Hardcoded patient roster for GET /patients/{patient_id}
_PATIENTS: dict[str, PatientFullResponse] = {
    "P001": PatientFullResponse(
        patient_id="P001", age=68, heart_rate=92, systolic_bp=130,
        diastolic_bp=85, creatinine_umol_l=110, killip_class=1,
        has_arrhythmia=True, cardiac_arrest=False, st_deviation=True,
        elevated_enzymes=True, risk_category="High", confidence=0.91,
        branch="SBRT", intervention_type="Cardiac Radioablation (SBRT)",
        bed_gy=87.5, grace_score=None,
        survival_without=0.54, survival_with=0.81,
    ),
    "P002": PatientFullResponse(
        patient_id="P002", age=55, heart_rate=72, systolic_bp=138,
        diastolic_bp=88, creatinine_umol_l=85, killip_class=1,
        has_arrhythmia=False, cardiac_arrest=False, st_deviation=False,
        elevated_enzymes=False, risk_category="Low", confidence=0.87,
        branch="Medication", intervention_type="Medication Intensity Calibration",
        bed_gy=None, grace_score=91,
        survival_without=0.82, survival_with=0.94,
    ),
    "P003": PatientFullResponse(
        patient_id="P003", age=74, heart_rate=105, systolic_bp=110,
        diastolic_bp=70, creatinine_umol_l=155, killip_class=2,
        has_arrhythmia=True, cardiac_arrest=True, st_deviation=True,
        elevated_enzymes=True, risk_category="High", confidence=0.95,
        branch="SBRT", intervention_type="Cardiac Radioablation (SBRT)",
        bed_gy=87.5, grace_score=None,
        survival_without=0.48, survival_with=0.76,
    ),
    "P004": PatientFullResponse(
        patient_id="P004", age=62, heart_rate=88, systolic_bp=145,
        diastolic_bp=92, creatinine_umol_l=100, killip_class=1,
        has_arrhythmia=False, cardiac_arrest=False, st_deviation=True,
        elevated_enzymes=True, risk_category="Medium", confidence=0.78,
        branch="Medication", intervention_type="Medication Intensity Calibration",
        bed_gy=None, grace_score=131,
        survival_without=0.68, survival_with=0.86,
    ),
    "P005": PatientFullResponse(
        patient_id="P005", age=49, heart_rate=65, systolic_bp=150,
        diastolic_bp=95, creatinine_umol_l=78, killip_class=1,
        has_arrhythmia=False, cardiac_arrest=False, st_deviation=False,
        elevated_enzymes=False, risk_category="Low", confidence=0.90,
        branch="Medication", intervention_type="Medication Intensity Calibration",
        bed_gy=None, grace_score=59,
        survival_without=0.88, survival_with=0.96,
    ),
}


def _sbrt_recommend(patient_id: str) -> RecommendResponse:
    return RecommendResponse(
        patient_id=patient_id,
        branch="SBRT",
        intervention_type="Cardiac Radioablation (SBRT)",
        intensity=None,
        bed_gy=87.5,
        bed_valid=False,   # 87.5 Gy < 100 Gy ablative threshold
        grace_score=None,
        grace_risk_category=None,
        survival_without=0.54,
        survival_with=0.81,
        counterfactual=(
            "Without intervention: 46% 2-year mortality risk. "
            "With SBRT at 25 Gy (BED 87.5 Gy): 19% mortality risk."
        ),
        model_version="part2_recommender_v1.0_MOCK",
        mock=True,
    )


def _medication_recommend(patient_id: str) -> RecommendResponse:
    return RecommendResponse(
        patient_id=patient_id,
        branch="Medication",
        intervention_type="Medication Intensity Calibration",
        intensity="Intensified",
        bed_gy=None,
        bed_valid=None,
        grace_score=131,
        grace_risk_category="Intermediate",
        survival_without=0.71,
        survival_with=0.88,
        counterfactual=(
            "Without intervention: 29% 2-year mortality risk. "
            "With Intensified medication therapy: 12% mortality risk."
        ),
        model_version="part2_recommender_v1.0_MOCK",
        mock=True,
    )


# ---------------------------------------------------------------------------
# 1. GET /api/v1/health
# ---------------------------------------------------------------------------

@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Infrastructure"],
)
def health() -> HealthResponse:
    """
    Liveness / readiness probe.

    Returns service status, loaded model versions, and uptime in seconds.
    No authentication required.
    """
    return HealthResponse(
        status="ok",
        version="v1",
        model_versions=ModelVersions(
            part1_classifier="v1.0_MOCK",
            part2_recommender="v1.0_MOCK",
        ),
        uptime_seconds=round(time.time() - _START_TIME, 2),
    )


# ---------------------------------------------------------------------------
# 2. POST /api/v1/predict
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/predict",
    response_model=PredictResponse,
    summary="Predict cardiovascular risk",
    tags=["Prediction"],
)
def predict(body: PatientVitalsRequest) -> PredictResponse:
    """
    Accept patient vital signs and return a cardiovascular risk classification.

    **Mock behaviour:** probabilities and risk category are hardcoded.
    The `mock: true` flag signals that the real Part-1 classifier is not yet wired in.

    Routing hint for /recommend:
    - `risk_category == "High"` → SBRT branch
    - `risk_category == "Medium" AND has_arrhythmia` → SBRT branch (arrhythmia escalation)
    - all other cases → Medication branch
    """
    # Deterministic mock: lean on the vitals to produce plausible-looking output
    if body.has_arrhythmia or body.cardiac_arrest or (body.age > 70 and body.killip_class >= 2):
        risk, low, med, high = "High",   0.04, 0.11, 0.85
    elif body.st_deviation or body.elevated_enzymes or body.age > 60:
        risk, low, med, high = "Medium", 0.06, 0.83, 0.11
    else:
        risk, low, med, high = "Low",    0.82, 0.13, 0.05

    return PredictResponse(
        patient_id=body.patient_id,
        risk_category=risk,
        confidence=round(max(low, med, high), 2),
        probabilities=RiskProbabilities(Low=low, Medium=med, High=high),
        model_version="part1_classifier_v1.0_MOCK",
        mock=True,
    )


# ---------------------------------------------------------------------------
# 3. POST /api/v1/recommend
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/recommend",
    response_model=RecommendResponse,
    summary="Recommend intervention",
    tags=["Recommendation"],
)
def recommend(body: RecommendRequest) -> RecommendResponse:
    """
    Accept a risk classification + patient metadata and return an intervention
    recommendation with 2-year survival probability and counterfactual output.

    **Routing logic (mirrors `src/clinical/routing.py`):**
    - `risk_category == "High"` → **SBRT branch** (Cardiac Radioablation)
    - `risk_category == "Medium" AND has_arrhythmia` → **SBRT branch** (arrhythmia escalation)
    - all other cases → **Medication branch**

    **Mock behaviour:** BED, GRACE score, and survival probabilities are hardcoded.
    """
    go_sbrt = (
        body.risk_category == "High"
        or (body.risk_category == "Medium" and body.has_arrhythmia)
    )
    if go_sbrt:
        return _sbrt_recommend(body.patient_id)
    return _medication_recommend(body.patient_id)


# ---------------------------------------------------------------------------
# 4. GET /api/v1/history
# ---------------------------------------------------------------------------

@app.get(
    "/api/v1/history",
    response_model=HistoryResponse,
    summary="List historical prediction records",
    tags=["History"],
)
def history(
    limit:  Annotated[int, Query(ge=1, le=50, description="Max records to return.")] = 10,
    offset: Annotated[int, Query(ge=0,         description="Records to skip.")]       = 0,
) -> HistoryResponse:
    """
    Return paginated historical patient prediction records.

    **Mock behaviour:** returns up to 10 hardcoded records. `limit` and `offset`
    are respected for slicing, but total is always ≤ 10.
    """
    sliced = _ALL_HISTORY[offset: offset + limit]
    return HistoryResponse(
        total=len(_ALL_HISTORY),
        offset=offset,
        limit=limit,
        records=sliced,
    )


# ---------------------------------------------------------------------------
# 5. GET /api/v1/patients/{patient_id}
# ---------------------------------------------------------------------------

@app.get(
    "/api/v1/patients/{patient_id}",
    response_model=PatientFullResponse,
    responses={404: {"model": ErrorResponse, "description": "Patient not found."}},
    summary="Get full patient record",
    tags=["Patients"],
)
def get_patient(patient_id: str) -> PatientFullResponse:
    """
    Return the full stored record for a single patient.

    **Mock behaviour:** only P001–P005 are hardcoded. Any other `patient_id`
    returns **404**.  Chiluba (Task 4) will replace this with a real DB lookup.
    """
    record = _PATIENTS.get(patient_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Patient {patient_id!r} not found. "
                   f"Available mock IDs: {sorted(_PATIENTS.keys())}",
        )
    return record
