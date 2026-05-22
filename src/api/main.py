"""
src/api/main.py
---------------
CardioSurv API — Production-ready (Task D Part A)
"""

from __future__ import annotations

import time
import uuid
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from src.api.schemas import (
    HealthResponse, ModelVersions,
    PatientVitalsRequest, PredictResponse, RiskProbabilities,
    RecommendRequest, RecommendResponse,
    HistoryItem, HistoryResponse,
    PatientFullResponse, PatientVitals, PredictionRecord, RecommendationRecord,
    ErrorResponse,
)
from src.db.models import AuditLog, Base, Patient, Prediction, Recommendation
from src.db.session import engine, get_db
from src.models.part1_classifier import load as load_model
from src.models.part1_classifier import predict as classifier_predict
from src.models.part2_recommender import load as load_part2
from src.models.part2_recommender import recommend as part2_recommend
from src.clinical.routing import route_patient as clinical_route
from src.clinical.routing import compute_grace as clinical_compute_grace

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model bundle (populated via lifespan)
# ---------------------------------------------------------------------------
MODEL_BUNDLE = {}

# ---------------------------------------------------------------------------
# Lifespan: Load models ONCE at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup/shutdown events."""
    logger.info("[lifespan] Starting up — loading models...")
    
    try:
        # Load Part-1 classifier
        MODEL_BUNDLE["part1"] = load_model("models/part1_classifier_v1.0.pkl")
        MODEL_BUNDLE["loaded"] = True
        logger.info("[lifespan] ✓ Part-1 classifier loaded")
    except FileNotFoundError as e:
        MODEL_BUNDLE["loaded"] = False
        logger.error(f"[lifespan] ✗ Model file not found: {e}")
    
    # Load Part-2 recommender (Bougacha T-A) — wired by T-E
    try:
        MODEL_BUNDLE["part2"] = load_part2("models/part2_recommender_v1.0.pkl")
        MODEL_BUNDLE["part2_loaded"] = True
        logger.info("[lifespan] ✓ Part-2 recommender loaded")
    except FileNotFoundError as e:
        MODEL_BUNDLE["part2_loaded"] = False
        logger.error(f"[lifespan] ✗ Part-2 model file not found: {e}")
    except Exception as e:
        MODEL_BUNDLE["part2_loaded"] = False
        logger.error(f"[lifespan] ✗ Part-2 model failed to load: {e}")
    
    yield  # Application runs here
    
    # Shutdown cleanup
    logger.info("[lifespan] Shutting down — clearing model cache")
    MODEL_BUNDLE.clear()

# ---------------------------------------------------------------------------
# Create tables on startup (safe — does nothing if tables already exist)
# ---------------------------------------------------------------------------
Base.metadata.create_all(engine)

# ---------------------------------------------------------------------------
# App + rate limiter
# ---------------------------------------------------------------------------
_START_TIME = time.time()
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="CardioSurv API",
    version="v1",
    description=(
        "Cardiovascular risk classification and intervention recommendation API.\n\n"
        "Contract: `docs/api_contract.md`"
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,  # ← Key change: use lifespan for model loading
)

app.state.limiter = limiter

# ---------------------------------------------------------------------------
# CORS Middleware
# ---------------------------------------------------------------------------
RENDER_FRONTEND_URL = "https://cardiosurv-frontend.onrender.com"

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
# § Error Handlers (ORDER MATTERS: specific → general)
# ---------------------------------------------------------------------------

#  Custom 422 Validation Error Handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return structured validation errors showing which field failed and why."""
    errors = []
    for err in exc.errors():
        loc = [str(l) for l in err["loc"] if l != "body"]
        errors.append({
            "field": ".".join(loc) if loc else "unknown",
            "message": err["msg"],
            "type": err["type"],
            "input": err.get("input")
        })
    
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": errors
            }
        }
    )

#  Custom 429 Rate Limit Handler
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Return structured rate limit error matching API contract."""
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "code": "RATE_LIMITED",
                "message": "Rate limit exceeded. Try again later.",
                "details": []
            }
        }
    )

#  Generic HTTPException Handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    code_map = {404: "NOT_FOUND", 422: "VALIDATION_ERROR",
                429: "RATE_LIMITED", 500: "INTERNAL_ERROR", 503: "MODEL_UNAVAILABLE"}
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": code_map.get(exc.status_code, "INTERNAL_ERROR"),
                "message": exc.detail if isinstance(exc.detail, str) else str(exc.detail),
                "details": [],
            }
        },
    )

#  Global 500 Error Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all handler: log internally, return clean JSON to user."""
    logger.error(f"[500] Unhandled exception: {type(exc).__name__}: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again later."
        }
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uuid() -> str:
    return str(uuid.uuid4())

def _now() -> datetime:
    return datetime.now(timezone.utc)

def _log_request(db: Session, route: str, status_code: int,
                 request_ip: str | None, latency_ms: int) -> None:
    """Insert an audit_log row (best-effort — never raises)."""
    try:
        db.add(AuditLog(
            route=route,
            status_code=status_code,
            request_ip=request_ip,
            latency_ms=latency_ms,
        ))
        db.commit()
    except Exception:
        db.rollback()

# ---------------------------------------------------------------------------
# §1  GET /api/v1/health
# ---------------------------------------------------------------------------

@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Infrastructure"],
)
def health() -> HealthResponse:
    """Liveness / readiness probe. No auth, no rate limit."""
    part1_tag = "v1.0" if MODEL_BUNDLE.get("loaded") else "NOT_LOADED"
    part2_tag = "v1.0" if MODEL_BUNDLE.get("part2_loaded") else "NOT_LOADED"
    survival_tag = "v1.0" if MODEL_BUNDLE.get("part2_loaded") else "NOT_LOADED"
    return HealthResponse(
        status="ok",
        version="v1",
        model_versions=ModelVersions(
            part1_classifier=part1_tag,
            part2_recommender=part2_tag,
            survival_cox=survival_tag,
        ),
        uptime_seconds=round(time.time() - _START_TIME, 2),
    )

# ---------------------------------------------------------------------------
# §2  POST /api/v1/predict
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/predict",
    response_model=PredictResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Field validation error."},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded."},
        503: {"model": ErrorResponse, "description": "Model not loaded."},
    },
    summary="Predict cardiovascular risk",
    tags=["Prediction"],
)
@limiter.limit("30/minute")
def predict(
    request: Request,
    body: PatientVitalsRequest,
    db: Session = Depends(get_db),
) -> PredictResponse:
    """
    Accept patient vital signs, persist them, run the Part-1 classifier,
    persist the prediction, and return the risk classification.
    """
    t0 = time.time()

    if not MODEL_BUNDLE.get("loaded") or MODEL_BUNDLE.get("part1") is None:
        raise HTTPException(status_code=503, detail="Model not available.")

    # 1. Insert patient row
    patient = Patient(
        age=body.age,
        sex=body.sex,
        chest_pain_type=body.chest_pain_type,
        resting_bp=body.resting_bp,
        cholesterol=body.cholesterol,
        fasting_bs=body.fasting_bs,
        resting_ecg=body.resting_ecg,
        max_hr=body.max_hr,
        exercise_angina=body.exercise_angina,
        oldpeak=body.oldpeak,
        st_slope=body.st_slope,
    )
    db.add(patient)
    db.flush()  # get patient.id

    # 2. Build feature dict for the classifier
    age_val = body.age
    bp_val  = body.resting_bp

    if bp_val < 120:
        bp_risk = "Normal"
    elif bp_val < 130:
        bp_risk = "Elevated"
    elif bp_val < 140:
        bp_risk = "Stage1"
    elif bp_val < 180:
        bp_risk = "Stage2"
    else:
        bp_risk = "Crisis"

    if age_val < 40:
        age_bin = "<40"
    elif age_val < 50:
        age_bin = "40-49"
    elif age_val < 60:
        age_bin = "50-59"
    elif age_val < 70:
        age_bin = "60-69"
    else:
        age_bin = "70+"

    hr_stress = round(body.max_hr / (220 - body.age), 3) if (220 - body.age) != 0 else 0.0

    features = {
        "Age":                  body.age,
        "Sex":                  body.sex,
        "ChestPainType":        body.chest_pain_type,
        "RestingBP":            body.resting_bp,
        "Cholesterol":          body.cholesterol,
        "FastingBS":            body.fasting_bs,
        "RestingECG":           body.resting_ecg,
        "MaxHR":                body.max_hr,
        "ExerciseAngina":       body.exercise_angina,
        "Oldpeak":              body.oldpeak,
        "ST_Slope":             body.st_slope,
        "AgeBin":               age_bin,
        "BP_RiskLevel":         bp_risk,
        "HeartRateStressIndex": hr_stress,
    }

    # 3. Run classifier
    result = classifier_predict(features, model_bundle=MODEL_BUNDLE["part1"])

    # 4. Insert prediction row
    prediction = Prediction(
        patient_id=patient.id,
        risk_category=result["risk_category"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        model_version=result["model_version"],
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    latency = int((time.time() - t0) * 1000)
    _log_request(db, "/api/v1/predict", 200,
                 request.client.host if request.client else None, latency)

    probs = result["probabilities"]
    return PredictResponse(
        patient_id=patient.id,
        prediction_id=prediction.id,
        risk_category=result["risk_category"],
        confidence=result["confidence"],
        probabilities=RiskProbabilities(
            Low=probs.get("Low", 0.0),
            Medium=probs.get("Medium", 0.0),
            High=probs.get("High", 0.0),
        ),
        model_version=result["model_version"],
        created_at=prediction.created_at,
    )

# ---------------------------------------------------------------------------
# §3  POST /api/v1/recommend
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
def recommend(
    request: Request,
    body: RecommendRequest,
    db: Session = Depends(get_db),
) -> RecommendResponse:
    """
    Run the real Part-2 ML pipeline (XGBoost intervention + Cox survival),
    persist the recommendation, and return the treatment plan.

    Flow (T-E):
      1. Look up prediction + patient rows in DB
      2. Build features_dict from the patient's stored vitals
      3. Call part2_recommend() → ML intervention + survival
      4. Call clinical_route() → bed_valid + grace_risk_category (DB fields)
      5. Map intensity_level (Standard/Intensified/Maximal/N/A) → Low/Moderate/High
      6. Insert recommendations row and return RecommendResponse
    """
    t0 = time.time()

    # ── Step 1: Look up prediction ──────────────────────────────────────────
    prediction = db.query(Prediction).filter(
        Prediction.id == body.prediction_id
    ).first()

    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction '{body.prediction_id}' not found.",
        )

    patient = db.query(Patient).filter(Patient.id == prediction.patient_id).first()

    if patient is None:
        raise HTTPException(
            status_code=404,
            detail=f"Patient '{prediction.patient_id}' not found.",
        )

    # ── Step 2: Verify Part-2 model is loaded ───────────────────────────────
    if not MODEL_BUNDLE.get("part2_loaded") or MODEL_BUNDLE.get("part2") is None:
        raise HTTPException(
            status_code=503,
            detail="Part-2 recommender model is not loaded. Check server logs.",
        )

    # ── Step 3: Build features_dict from stored patient vitals ──────────────
    features_dict = {
        "patient_id":      prediction.patient_id,
        "age":             patient.age,
        "sex":             patient.sex,
        "chest_pain_type": patient.chest_pain_type,
        "resting_bp":      patient.resting_bp,
        "cholesterol":     patient.cholesterol,
        "fasting_bs":      patient.fasting_bs,
        "resting_ecg":     patient.resting_ecg,
        "max_hr":          patient.max_hr,
        "exercise_angina": patient.exercise_angina,
        "oldpeak":         float(patient.oldpeak),
        "st_slope":        patient.st_slope,
        "has_arrhythmia":  bool(body.has_arrhythmia),
    }

    # ── Step 4: Call the real Part-2 ML pipeline ────────────────────────────
    try:
        result = part2_recommend(features_dict, bundle=MODEL_BUNDLE["part2"])
    except Exception as e:
        logger.exception("[recommend] part2_recommend failed")
        raise HTTPException(
            status_code=500,
            detail=f"Part-2 recommendation failed: {e}",
        )

    # ── Step 5: Also run clinical_route to populate bed_valid + grace_cat ──
    # (These two DB columns are not returned by part2_recommend.)
    route_input = {
        "has_arrhythmia": bool(body.has_arrhythmia),
        "age":            patient.age,
        "heart_rate":     patient.max_hr,
        "systolic_bp":    float(patient.resting_bp),
    }
    route_result = clinical_route(
        route_input,
        predicted_risk=prediction.risk_category,
        confidence_score=float(prediction.confidence),
    )

    # ── Step 6: Derive DB-shaped fields from the ML result ──────────────────
    # Branch: SBRT if intervention_type is the radioablation slug, else Medication.
    branch = "SBRT" if result["intervention_type"] == "cardiac_sbrt_25Gy_1fx" else "Medication"

    # Map Part-2 intensity_level → DB intensity (String(8) column).
    #   "Standard"    → "Low"
    #   "Intensified" → "Moderate"
    #   "Maximal"     → "High"
    #   "N/A" (SBRT)  → "High"
    INTENSITY_MAP = {
        "Standard":    "Low",
        "Intensified": "Moderate",
        "Maximal":     "High",
        "N/A":         "High",
    }
    intensity_db = INTENSITY_MAP.get(result["intensity_level"], "Moderate")

    bed_gy_val      = result.get("bed_gy")  # None for Medication
    grace_score_val = result.get("grace_score")  # None for SBRT

    if branch == "SBRT":
        bed_valid_val = route_result.bed_valid
        grace_cat_val = None
    else:
        bed_valid_val = None
        grace_cat_val = (
            route_result.grace_result.risk_category
            if route_result.grace_result else None
        )

    # ── Step 7: Persist the recommendation row ──────────────────────────────
    rec = Recommendation(
        prediction_id=prediction.id,
        branch=branch,
        intervention_type=result["intervention_type"],
        intensity=intensity_db,
        bed_gy=bed_gy_val,
        bed_valid=bed_valid_val,
        grace_score=grace_score_val,
        grace_risk_category=grace_cat_val,
        survival_without=float(result["survival_without"]),
        survival_with=float(result["survival_with"]),
        model_version=result.get("model_version", "part2_recommender_v1.0"),
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)

    latency = int((time.time() - t0) * 1000)
    _log_request(db, "/api/v1/recommend", 200,
                 request.client.host if request.client else None, latency)

    return RecommendResponse(
        recommendation_id=rec.id,
        patient_id=prediction.patient_id,
        prediction_id=prediction.id,
        branch=rec.branch,
        intervention_type=rec.intervention_type,
        intensity=rec.intensity,
        bed_gy=float(rec.bed_gy) if rec.bed_gy is not None else None,
        bed_valid=rec.bed_valid,
        grace_score=rec.grace_score,
        grace_risk_category=rec.grace_risk_category,
        survival_without=float(rec.survival_without),
        survival_with=float(rec.survival_with),
        model_version=rec.model_version,
        created_at=rec.created_at,
    )

# ---------------------------------------------------------------------------
# §4  GET /api/v1/history
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
    page: int = Query(default=1, ge=1, description="1-indexed page number."),
    size: int = Query(default=20, ge=1, le=100, description="Records per page."),
    db: Session = Depends(get_db),
) -> HistoryResponse:
    """Return paginated history of predictions + recommendations from the database."""

    total = db.query(Prediction).count()
    offset = (page - 1) * size

    rows = (
        db.query(Prediction, Recommendation, Patient)
        .outerjoin(Recommendation, Recommendation.prediction_id == Prediction.id)
        .outerjoin(Patient, Patient.id == Prediction.patient_id)
        .order_by(Prediction.created_at.desc())
        .offset(offset)
        .limit(size)
        .all()
    )

    items = []
    for pred, rec, pat in rows:
        survival_without = float(rec.survival_without) if rec else 0.0
        survival_with    = float(rec.survival_with)    if rec else 0.0
        items.append(HistoryItem(
            prediction_id=pred.id,
            patient_id=pred.patient_id,
            created_at=pred.created_at,
            age=pat.age if pat else 0,
            risk_category=pred.risk_category,
            confidence=float(pred.confidence),
            branch=rec.branch if rec else "N/A",
            intervention_type=rec.intervention_type if rec else "N/A",
            survival_without=survival_without,
            survival_with=survival_with,
            survival_improvement=round(survival_with - survival_without, 3),
        ))

    return HistoryResponse(page=page, size=size, total=total, items=items)

# ---------------------------------------------------------------------------
# §5  GET /api/v1/patients/{patient_id}
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
def get_patient(
    request: Request,
    patient_id: str,
    db: Session = Depends(get_db),
) -> PatientFullResponse:
    """Return the full record for a patient: vitals + all predictions + all recommendations."""

    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if patient is None:
        raise HTTPException(
            status_code=404,
            detail=f"Patient '{patient_id}' not found.",
        )

    predictions = (
        db.query(Prediction)
        .filter(Prediction.patient_id == patient_id)
        .order_by(Prediction.created_at.desc())
        .all()
    )

    pred_records = [
        PredictionRecord(
            prediction_id=p.id,
            risk_category=p.risk_category,
            confidence=float(p.confidence),
            probabilities=RiskProbabilities(
                Low=p.probabilities.get("Low", 0.0),
                Medium=p.probabilities.get("Medium", 0.0),
                High=p.probabilities.get("High", 0.0),
            ),
            model_version=p.model_version,
            created_at=p.created_at,
        )
        for p in predictions
    ]

    rec_records = []
    for p in predictions:
        recs = (
            db.query(Recommendation)
            .filter(Recommendation.prediction_id == p.id)
            .all()
        )
        for r in recs:
            rec_records.append(RecommendationRecord(
                recommendation_id=r.id,
                branch=r.branch,
                intervention_type=r.intervention_type,
                intensity=r.intensity,
                bed_gy=float(r.bed_gy) if r.bed_gy is not None else None,
                bed_valid=r.bed_valid,
                grace_score=r.grace_score,
                grace_risk_category=r.grace_risk_category,
                survival_without=float(r.survival_without),
                survival_with=float(r.survival_with),
                model_version=r.model_version,
                created_at=r.created_at,
            ))

    return PatientFullResponse(
        patient_id=patient.id,
        created_at=patient.created_at,
        vitals=PatientVitals(
            age=patient.age,
            sex=patient.sex,
            chest_pain_type=patient.chest_pain_type,
            resting_bp=patient.resting_bp,
            cholesterol=patient.cholesterol,
            fasting_bs=patient.fasting_bs,
            resting_ecg=patient.resting_ecg,
            max_hr=patient.max_hr,
            exercise_angina=patient.exercise_angina,
            oldpeak=float(patient.oldpeak),
            st_slope=patient.st_slope,
        ),
        predictions=pred_records,
        recommendations=rec_records,
    )
