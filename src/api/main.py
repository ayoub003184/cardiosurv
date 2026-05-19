"""
src/api/main.py
---------------
CardioSurv API  —  mock scaffold (Task 3).

Implements all 5 endpoints from docs/api_contract.md (Day-1 locked contract).
All responses are hardcoded mock data. The `mock` scaffold is clearly signalled
so Chiluba (Task 4) knows exactly what to replace with real DB/model calls.

Run:
    uvicorn src.api.main:app --reload
    open http://localhost:8000/docs
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.api.schemas import (
    # Health
    HealthResponse, ModelVersions,
    # Predict
    PatientVitalsRequest, PredictResponse, RiskProbabilities,
    # Recommend
    RecommendRequest, RecommendResponse,
    # History
    HistoryItem, HistoryResponse,
    # Patients
    PatientFullResponse, PatientVitals, PredictionRecord, RecommendationRecord,
    # Errors
    ErrorResponse, ErrorBody,
)

# ---------------------------------------------------------------------------
# App + rate limiter (§7)
# ---------------------------------------------------------------------------

_START_TIME = time.time()

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="CardioSurv API",
    version="v1",
    description=(
        "Cardiovascular risk classification and intervention recommendation API.\n\n"
        "> **Mock scaffold (Task 3)** — all responses are hardcoded. "
        "Chiluba (Task 4) will replace hardcoded values with real DB + model calls.\n\n"
        "Contract: `docs/api_contract.md`"
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — localhost dev origins + production Render URL placeholder (§0)
# Replace RENDER_FRONTEND_URL with the actual Static Site URL once deployed.
RENDER_FRONTEND_URL = "https://cardiosurv-frontend.onrender.com"  # update on deploy

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        RENDER_FRONTEND_URL,
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Custom 404 error handler — enforces uniform error shape (§6)
# ---------------------------------------------------------------------------

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"error": {
            "code": "NOT_FOUND",
            "message": exc.detail,
            "details": [],
        }},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    code_map = {404: "NOT_FOUND", 422: "VALIDATION_ERROR",
                429: "RATE_LIMITED", 500: "INTERNAL_ERROR", 503: "MODEL_UNAVAILABLE"}
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {
            "code": code_map.get(exc.status_code, "INTERNAL_ERROR"),
            "message": exc.detail if isinstance(exc.detail, str) else str(exc.detail),
            "details": [],
        }},
    )


# ---------------------------------------------------------------------------
# Mock data store — in-memory, mimics what the DB will hold
# (Chiluba replaces these dicts with real Postgres queries in Task 4)
# ---------------------------------------------------------------------------

def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# Fixed UUIDs so mock responses are stable across restarts
_PATIENT_IDS = {
    "P1": "9c1b7e64-3a4f-4d3a-a4c8-2e5e7b71ed12",
    "P2": "a2c3d4e5-f6a7-4b8c-9d0e-1f2a3b4c5d6e",
    "P3": "b3c4d5e6-a7b8-4c9d-0e1f-2a3b4c5d6e7f",
    "P4": "c4d5e6f7-b8c9-4d0e-1f2a-3b4c5d6e7f80",
    "P5": "d5e6f780-c9d0-4e1f-2a3b-4c5d6e7f8091",
}
_PRED_IDS = {
    "P1": "f0e9d8c7-b6a5-4948-83a1-72635849affe",
    "P2": "e1f0a9b8-c7d6-4e5f-9283-6174950bfffe",
    "P3": "d2e1b0a9-d8e7-4f60-a394-7285061cffff",
    "P4": "c3d2c1b0-e9f0-4071-b4a5-8396172d0000",
    "P5": "b4e3d2c1-f0a1-4182-c5b6-94a7283e1111",
}
_REC_IDS = {
    "P1": "11111111-2222-3333-4444-555555555555",
    "P2": "22222222-3333-4444-5555-666666666666",
    "P3": "33333333-4444-5555-6666-777777777777",
    "P4": "44444444-5555-6666-7777-888888888888",
    "P5": "55555555-6666-7777-8888-999999999999",
}

# prediction_id → (patient_id, risk_category, age, has_arrhythmia)
# This is what the mock /recommend looks up when given a prediction_id.
_PREDICTIONS: dict[str, dict] = {
    _PRED_IDS["P1"]: {
        "patient_id": _PATIENT_IDS["P1"], "risk_category": "Medium",
        "confidence": 0.83, "age": 56, "has_arrhythmia": False,
        "probabilities": {"Low": 0.06, "Medium": 0.83, "High": 0.11},
        "created_at": datetime(2026, 6, 20, 14, 21, 9, tzinfo=timezone.utc),
    },
    _PRED_IDS["P2"]: {
        "patient_id": _PATIENT_IDS["P2"], "risk_category": "High",
        "confidence": 0.91, "age": 72, "has_arrhythmia": True,
        "probabilities": {"Low": 0.02, "Medium": 0.07, "High": 0.91},
        "created_at": datetime(2026, 6, 20, 15, 0, 0, tzinfo=timezone.utc),
    },
    _PRED_IDS["P3"]: {
        "patient_id": _PATIENT_IDS["P3"], "risk_category": "Low",
        "confidence": 0.87, "age": 45, "has_arrhythmia": False,
        "probabilities": {"Low": 0.87, "Medium": 0.10, "High": 0.03},
        "created_at": datetime(2026, 6, 21, 9, 5, 0, tzinfo=timezone.utc),
    },
    _PRED_IDS["P4"]: {
        "patient_id": _PATIENT_IDS["P4"], "risk_category": "Medium",
        "confidence": 0.78, "age": 62, "has_arrhythmia": True,
        "probabilities": {"Low": 0.09, "Medium": 0.78, "High": 0.13},
        "created_at": datetime(2026, 6, 21, 11, 30, 0, tzinfo=timezone.utc),
    },
    _PRED_IDS["P5"]: {
        "patient_id": _PATIENT_IDS["P5"], "risk_category": "High",
        "confidence": 0.95, "age": 74, "has_arrhythmia": True,
        "probabilities": {"Low": 0.01, "Medium": 0.04, "High": 0.95},
        "created_at": datetime(2026, 6, 21, 14, 0, 0, tzinfo=timezone.utc),
    },
}

# Build the full history list from the mock predictions
_HISTORY_ROWS: list[HistoryItem] = [
    HistoryItem(
        prediction_id=_PRED_IDS["P1"], patient_id=_PATIENT_IDS["P1"],
        created_at=datetime(2026, 6, 20, 14, 21, 9, tzinfo=timezone.utc),
        age=56, risk_category="Medium", confidence=0.83,
        branch="Medication", intervention_type="beta_blocker+moderate_statin+aspirin",
        survival_without=0.81, survival_with=0.92, survival_improvement=0.11,
    ),
    HistoryItem(
        prediction_id=_PRED_IDS["P2"], patient_id=_PATIENT_IDS["P2"],
        created_at=datetime(2026, 6, 20, 15, 0, 0, tzinfo=timezone.utc),
        age=72, risk_category="High", confidence=0.91,
        branch="SBRT", intervention_type="cardiac_sbrt_25Gy_1fx",
        survival_without=0.58, survival_with=0.81, survival_improvement=0.23,
    ),
    HistoryItem(
        prediction_id=_PRED_IDS["P3"], patient_id=_PATIENT_IDS["P3"],
        created_at=datetime(2026, 6, 21, 9, 5, 0, tzinfo=timezone.utc),
        age=45, risk_category="Low", confidence=0.87,
        branch="Medication", intervention_type="low_dose_statin+lifestyle",
        survival_without=0.91, survival_with=0.96, survival_improvement=0.05,
    ),
    HistoryItem(
        prediction_id=_PRED_IDS["P4"], patient_id=_PATIENT_IDS["P4"],
        created_at=datetime(2026, 6, 21, 11, 30, 0, tzinfo=timezone.utc),
        age=62, risk_category="Medium", confidence=0.78,
        branch="SBRT", intervention_type="cardiac_sbrt_25Gy_1fx",
        survival_without=0.65, survival_with=0.83, survival_improvement=0.18,
    ),
    HistoryItem(
        prediction_id=_PRED_IDS["P5"], patient_id=_PATIENT_IDS["P5"],
        created_at=datetime(2026, 6, 21, 14, 0, 0, tzinfo=timezone.utc),
        age=74, risk_category="High", confidence=0.95,
        branch="SBRT", intervention_type="cardiac_sbrt_25Gy_1fx",
        survival_without=0.52, survival_with=0.79, survival_improvement=0.27,
    ),
    HistoryItem(
        prediction_id="e9f0a1b2-c3d4-4e5f-6a7b-8c9d0e1f2a3b",
        patient_id="f0a1b2c3-d4e5-4f60-7182-9a0b1c2d3e4f",
        created_at=datetime(2026, 6, 22, 8, 0, 0, tzinfo=timezone.utc),
        age=50, risk_category="Low", confidence=0.90,
        branch="Medication", intervention_type="low_dose_statin+lifestyle",
        survival_without=0.88, survival_with=0.95, survival_improvement=0.07,
    ),
    HistoryItem(
        prediction_id="a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d",
        patient_id="b2c3d4e5-f6a7-4b8c-9d0e-1f2a3b4c5d6e",
        created_at=datetime(2026, 6, 22, 10, 15, 0, tzinfo=timezone.utc),
        age=68, risk_category="High", confidence=0.88,
        branch="SBRT", intervention_type="cardiac_sbrt_25Gy_1fx",
        survival_without=0.55, survival_with=0.80, survival_improvement=0.25,
    ),
    HistoryItem(
        prediction_id="b2c3d4e5-f6a7-4b8c-9d0e-1f2a3b4c5d6e",
        patient_id="c3d4e5f6-a7b8-4c9d-0e1f-2a3b4c5d6e7f",
        created_at=datetime(2026, 6, 22, 14, 45, 0, tzinfo=timezone.utc),
        age=59, risk_category="Medium", confidence=0.74,
        branch="Medication", intervention_type="beta_blocker+moderate_statin+aspirin",
        survival_without=0.76, survival_with=0.89, survival_improvement=0.13,
    ),
    HistoryItem(
        prediction_id="c3d4e5f6-a7b8-4c9d-0e1f-2a3b4c5d6e7f",
        patient_id="d4e5f6a7-b8c9-4d0e-1f2a-3b4c5d6e7f80",
        created_at=datetime(2026, 6, 23, 9, 30, 0, tzinfo=timezone.utc),
        age=66, risk_category="High", confidence=0.93,
        branch="SBRT", intervention_type="cardiac_sbrt_25Gy_1fx",
        survival_without=0.51, survival_with=0.78, survival_improvement=0.27,
    ),
    HistoryItem(
        prediction_id="d4e5f6a7-b8c9-4d0e-1f2a-3b4c5d6e7f80",
        patient_id="e5f6a7b8-c9d0-4e1f-2a3b-4c5d6e7f8091",
        created_at=datetime(2026, 6, 23, 11, 0, 0, tzinfo=timezone.utc),
        age=42, risk_category="Low", confidence=0.92,
        branch="Medication", intervention_type="low_dose_statin+lifestyle",
        survival_without=0.92, survival_with=0.97, survival_improvement=0.05,
    ),
]

# Full patient records for GET /patients/{patient_id}
_PATIENTS: dict[str, PatientFullResponse] = {
    _PATIENT_IDS["P1"]: PatientFullResponse(
        patient_id=_PATIENT_IDS["P1"],
        created_at=datetime(2026, 6, 20, 14, 20, 55, tzinfo=timezone.utc),
        vitals=PatientVitals(
            age=56, sex="M", chest_pain_type="ATA", resting_bp=138,
            cholesterol=230, fasting_bs=0, resting_ecg="Normal",
            max_hr=150, exercise_angina="N", oldpeak=1.2, st_slope="Up",
        ),
        predictions=[PredictionRecord(
            prediction_id=_PRED_IDS["P1"], risk_category="Medium", confidence=0.83,
            probabilities=RiskProbabilities(Low=0.06, Medium=0.83, High=0.11),
            model_version="part1_classifier_v1.0",
            created_at=datetime(2026, 6, 20, 14, 21, 9, tzinfo=timezone.utc),
        )],
        recommendations=[RecommendationRecord(
            recommendation_id=_REC_IDS["P1"], branch="Medication",
            intervention_type="beta_blocker+moderate_statin+aspirin",
            intensity="Moderate", bed_gy=None, bed_valid=None,
            grace_score=132, grace_risk_category="Intermediate",
            survival_without=0.81, survival_with=0.92,
            model_version="part2_recommender_v1.0",
            created_at=datetime(2026, 6, 20, 14, 21, 11, tzinfo=timezone.utc),
        )],
    ),
    _PATIENT_IDS["P2"]: PatientFullResponse(
        patient_id=_PATIENT_IDS["P2"],
        created_at=datetime(2026, 6, 20, 14, 58, 0, tzinfo=timezone.utc),
        vitals=PatientVitals(
            age=72, sex="M", chest_pain_type="ASY", resting_bp=160,
            cholesterol=280, fasting_bs=1, resting_ecg="LVH",
            max_hr=110, exercise_angina="Y", oldpeak=3.5, st_slope="Flat",
        ),
        predictions=[PredictionRecord(
            prediction_id=_PRED_IDS["P2"], risk_category="High", confidence=0.91,
            probabilities=RiskProbabilities(Low=0.02, Medium=0.07, High=0.91),
            model_version="part1_classifier_v1.0",
            created_at=datetime(2026, 6, 20, 15, 0, 0, tzinfo=timezone.utc),
        )],
        recommendations=[RecommendationRecord(
            recommendation_id=_REC_IDS["P2"], branch="SBRT",
            intervention_type="cardiac_sbrt_25Gy_1fx",
            intensity="High", bed_gy=87.5, bed_valid=True,
            grace_score=None, grace_risk_category=None,
            survival_without=0.58, survival_with=0.81,
            model_version="part2_recommender_v1.0",
            created_at=datetime(2026, 6, 20, 15, 0, 5, tzinfo=timezone.utc),
        )],
    ),
    _PATIENT_IDS["P3"]: PatientFullResponse(
        patient_id=_PATIENT_IDS["P3"],
        created_at=datetime(2026, 6, 21, 9, 3, 0, tzinfo=timezone.utc),
        vitals=PatientVitals(
            age=45, sex="F", chest_pain_type="NAP", resting_bp=120,
            cholesterol=200, fasting_bs=0, resting_ecg="Normal",
            max_hr=170, exercise_angina="N", oldpeak=0.5, st_slope="Up",
        ),
        predictions=[PredictionRecord(
            prediction_id=_PRED_IDS["P3"], risk_category="Low", confidence=0.87,
            probabilities=RiskProbabilities(Low=0.87, Medium=0.10, High=0.03),
            model_version="part1_classifier_v1.0",
            created_at=datetime(2026, 6, 21, 9, 5, 0, tzinfo=timezone.utc),
        )],
        recommendations=[RecommendationRecord(
            recommendation_id=_REC_IDS["P3"], branch="Medication",
            intervention_type="low_dose_statin+lifestyle",
            intensity="Low", bed_gy=None, bed_valid=None,
            grace_score=59, grace_risk_category="Low",
            survival_without=0.91, survival_with=0.96,
            model_version="part2_recommender_v1.0",
            created_at=datetime(2026, 6, 21, 9, 5, 8, tzinfo=timezone.utc),
        )],
    ),
}


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _should_route_to_sbrt(risk_category: str, has_arrhythmia: bool) -> bool:
    """Mirror src/clinical/routing.py logic."""
    return risk_category == "High" or (risk_category == "Medium" and has_arrhythmia)


def _mock_sbrt_response(
    patient_id: str, prediction_id: str, rec_id: str
) -> RecommendResponse:
    return RecommendResponse(
        recommendation_id=rec_id,
        patient_id=patient_id,
        prediction_id=prediction_id,
        branch="SBRT",
        intervention_type="cardiac_sbrt_25Gy_1fx",
        intensity="High",                       # locked enum: Low|Moderate|High
        bed_gy=87.5,
        bed_valid=True,                          # realistic positive example
        grace_score=None,
        grace_risk_category=None,
        survival_without=0.58,
        survival_with=0.81,
        model_version="part2_recommender_v1.0",
        created_at=_now(),
    )


def _mock_medication_response(
    patient_id: str, prediction_id: str, rec_id: str
) -> RecommendResponse:
    return RecommendResponse(
        recommendation_id=rec_id,
        patient_id=patient_id,
        prediction_id=prediction_id,
        branch="Medication",
        intervention_type="beta_blocker+moderate_statin+aspirin",
        intensity="Moderate",                    # locked enum: Low|Moderate|High
        bed_gy=None,
        bed_valid=None,
        grace_score=132,
        grace_risk_category="Intermediate",
        survival_without=0.81,
        survival_with=0.92,
        model_version="part2_recommender_v1.0",
        created_at=_now(),
    )


# ---------------------------------------------------------------------------
# §1  GET /api/v1/health  (no rate limit per §7)
# ---------------------------------------------------------------------------

@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Infrastructure"],
)
def health() -> HealthResponse:
    """Liveness / readiness probe. No auth, no rate limit (§1)."""
    return HealthResponse(
        status="ok",
        version="v1",
        model_versions=ModelVersions(
            part1_classifier="v1.0_MOCK",
            part2_recommender="v1.0_MOCK",
            survival_cox="v1.0_MOCK",
        ),
        uptime_seconds=round(time.time() - _START_TIME, 2),
    )


# ---------------------------------------------------------------------------
# §2  POST /api/v1/predict  (30/min/IP per §7)
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/predict",
    response_model=PredictResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Field validation error."},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded."},
    },
    summary="Predict cardiovascular risk",
    tags=["Prediction"],
)
@limiter.limit("30/minute")
def predict(request: Request, body: PatientVitalsRequest) -> PredictResponse:
    """
    Accept patient vital signs and return a Part-1 risk classification (§2).

    Internally (mock): validates → generates UUIDs → returns hardcoded risk.
    Real implementation (Task 4): inserts patients row → calls part1_predict() → inserts predictions row.

    Risk derivation (from data contract §4):
    - HeartDisease=0 → Low
    - HeartDisease=1, no severe flags → Medium
    - HeartDisease=1, any of [Oldpeak≥2, ExerciseAngina=Y, BP Stage2+] → High
    """
    # Mock risk derivation — plausible logic based on field values
    has_severe = (
        body.oldpeak >= 2.0
        or body.exercise_angina == "Y"
        or body.resting_bp >= 140
    )
    has_disease_signal = (
        body.chest_pain_type == "ASY"
        or body.fasting_bs == 1
        or body.resting_ecg in ("ST", "LVH")
        or body.st_slope in ("Flat", "Down")
    )

    if has_disease_signal and has_severe:
        risk, low, med, high = "High",   0.04, 0.09, 0.87
    elif has_disease_signal:
        risk, low, med, high = "Medium", 0.06, 0.83, 0.11
    else:
        risk, low, med, high = "Low",    0.85, 0.11, 0.04

    patient_id    = _uuid()
    prediction_id = _uuid()

    # Store in mock in-memory store so /recommend can look it up
    _PREDICTIONS[prediction_id] = {
        "patient_id":     patient_id,
        "risk_category":  risk,
        "confidence":     round(max(low, med, high), 2),
        "age":            body.age,
        "has_arrhythmia": False,      # not in predict request; clinician sets in /recommend
        "probabilities":  {"Low": low, "Medium": med, "High": high},
        "created_at":     _now(),
    }

    return PredictResponse(
        patient_id=patient_id,
        prediction_id=prediction_id,
        risk_category=risk,
        confidence=round(max(low, med, high), 2),
        probabilities=RiskProbabilities(Low=low, Medium=med, High=high),
        model_version="part1_classifier_v1.0",
        created_at=_now(),
    )


# ---------------------------------------------------------------------------
# §3  POST /api/v1/recommend  (30/min/IP per §7)
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/recommend",
    response_model=RecommendResponse,
    responses={
        404: {"model": ErrorResponse, "description": "prediction_id not found."},
        422: {"model": ErrorResponse, "description": "Field validation error."},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded."},
    },
    summary="Recommend intervention",
    tags=["Recommendation"],
)
@limiter.limit("30/minute")
def recommend(request: Request, body: RecommendRequest) -> RecommendResponse:
    """
    Accept a `prediction_id` + clinician `has_arrhythmia` flag and return a
    Part-2 treatment recommendation with 2-year survival pair (§3).

    Routing: High OR (Medium AND has_arrhythmia) → SBRT branch; else → Medication branch.

    Real implementation (Task 4): looks up prediction → calls part2_recommend() → inserts recommendations row.
    """
    pred = _PREDICTIONS.get(body.prediction_id)
    if pred is None:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction '{body.prediction_id}' not found.",
        )

    rec_id = _uuid()
    patient_id    = pred["patient_id"]
    risk_category = pred["risk_category"]

    if _should_route_to_sbrt(risk_category, body.has_arrhythmia):
        return _mock_sbrt_response(patient_id, body.prediction_id, rec_id)
    return _mock_medication_response(patient_id, body.prediction_id, rec_id)


# ---------------------------------------------------------------------------
# §4  GET /api/v1/history  (120/min/IP per §7)
# ---------------------------------------------------------------------------

@app.get(
    "/api/v1/history",
    response_model=HistoryResponse,
    responses={429: {"model": ErrorResponse, "description": "Rate limit exceeded."}},
    summary="List historical prediction records",
    tags=["History"],
)
@limiter.limit("120/minute")
def history(
    request: Request,
    page: int = Query(default=1,  ge=1,       description="1-indexed page number."),
    size: int = Query(default=20, ge=1, le=100, description="Records per page."),
) -> HistoryResponse:
    """
    Return paginated history of predictions + recommendations across all patients (§4).
    `survival_improvement` is computed server-side as `survival_with − survival_without`.
    """
    total  = len(_HISTORY_ROWS)
    offset = (page - 1) * size
    items  = _HISTORY_ROWS[offset: offset + size]
    return HistoryResponse(page=page, size=size, total=total, items=items)


# ---------------------------------------------------------------------------
# §5  GET /api/v1/patients/{patient_id}  (120/min/IP per §7)
# ---------------------------------------------------------------------------

@app.get(
    "/api/v1/patients/{patient_id}",
    response_model=PatientFullResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Patient not found."},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded."},
    },
    summary="Get full patient record",
    tags=["Patients"],
)
@limiter.limit("120/minute")
def get_patient(request: Request, patient_id: str) -> PatientFullResponse:
    """
    Return the full record for a patient: vitals + all predictions + all recommendations (§5).

    Mock: only the UUIDs for P1–P3 are hardcoded.
    Real implementation (Task 4): DB join across patients / predictions / recommendations.
    """
    record = _PATIENTS.get(patient_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Patient '{patient_id}' not found.",
        )
    return record
