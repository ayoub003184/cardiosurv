"""
src/api/schemas.py
------------------
Pydantic models for the CardioSurv API v1.

Source of truth: docs/api_contract.md (Day-1 locked contract, owner M4).
Every field name, type, and enum matches §1–§6 of that document exactly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# §6  Uniform error shape
# ---------------------------------------------------------------------------

class ErrorDetail(BaseModel):
    field:  Optional[str] = None
    reason: Optional[str] = None
    got:    Optional[object] = None


class ErrorBody(BaseModel):
    code:    str                    = Field(..., examples=["VALIDATION_ERROR"])
    message: str
    details: list[ErrorDetail]      = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """
    All non-2xx responses use this shape (§6).

        { "error": { "code": "…", "message": "…", "details": […] } }

    Standard codes: VALIDATION_ERROR · NOT_FOUND · RATE_LIMITED
                    INTERNAL_ERROR · MODEL_UNAVAILABLE
    """
    error: ErrorBody

    model_config = {"json_schema_extra": {"example": {
        "error": {
            "code": "NOT_FOUND",
            "message": "Prediction f0e9... not found.",
            "details": [],
        }
    }}}


# ---------------------------------------------------------------------------
# §1  GET /api/v1/health
# ---------------------------------------------------------------------------

class ModelVersions(BaseModel):
    part1_classifier:  str = "v1.0"
    part2_recommender: str = "v1.0"
    survival_cox:      str = "v1.0"    # required by contract §1


class HealthResponse(BaseModel):
    status:         str           = Field(..., description="'ok' when healthy.")
    version:        str           = Field(..., description="API version string.")
    model_versions: ModelVersions
    uptime_seconds: float         = Field(..., description="Seconds since process start.")

    model_config = {"json_schema_extra": {"example": {
        "status": "ok",
        "version": "v1",
        "model_versions": {
            "part1_classifier": "v1.0",
            "part2_recommender": "v1.0",
            "survival_cox": "v1.0",
        },
        "uptime_seconds": 12345,
    }}}


# ---------------------------------------------------------------------------
# §2  POST /api/v1/predict
# ---------------------------------------------------------------------------

class PatientVitalsRequest(BaseModel):
    """
    Request body for POST /api/v1/predict (§2).
    Field names and constraints match the Kaggle heart.csv / unified schema (data contract §2).
    """
    age:             int   = Field(..., ge=1,   le=120,  description="Age in years.")
    sex:             str   = Field(...,          description="'M' or 'F'.")
    chest_pain_type: str   = Field(...,          description="TA | ATA | NAP | ASY.")
    resting_bp:      int   = Field(..., ge=50,  le=250,  description="Resting blood pressure (mmHg).")
    cholesterol:     int   = Field(..., ge=50,  le=800,  description="Serum cholesterol (mg/dL).")
    fasting_bs:      int   = Field(..., ge=0,   le=1,    description="Fasting blood sugar ≥120 mg/dL: 0 or 1.")
    resting_ecg:     str   = Field(...,          description="Normal | ST | LVH.")
    max_hr:          int   = Field(..., ge=40,  le=230,  description="Maximum heart rate achieved (bpm).")
    exercise_angina: str   = Field(...,          description="'N' or 'Y'.")
    oldpeak:         float = Field(..., ge=-3.0, le=7.0, description="ST depression induced by exercise.")
    st_slope:        str   = Field(...,          description="Up | Flat | Down.")

    model_config = {"json_schema_extra": {"example": {
        "age": 56, "sex": "M", "chest_pain_type": "ATA",
        "resting_bp": 138, "cholesterol": 230, "fasting_bs": 0,
        "resting_ecg": "Normal", "max_hr": 150, "exercise_angina": "N",
        "oldpeak": 1.2, "st_slope": "Up",
    }}}


class RiskProbabilities(BaseModel):
    Low:    float = Field(..., ge=0, le=1)
    Medium: float = Field(..., ge=0, le=1)
    High:   float = Field(..., ge=0, le=1)


class PredictResponse(BaseModel):
    """Response body for POST /api/v1/predict (§2)."""
    patient_id:    str              = Field(..., description="Server-generated UUID v4.")
    prediction_id: str              = Field(..., description="Server-generated UUID v4.")
    risk_category: str              = Field(..., description="Low | Medium | High.")
    confidence:    float            = Field(..., ge=0, le=1)
    probabilities: RiskProbabilities
    model_version: str
    created_at:    datetime         = Field(..., description="RFC 3339 UTC timestamp.")

    model_config = {"json_schema_extra": {"example": {
        "patient_id":    "9c1b7e64-3a4f-4d3a-a4c8-2e5e7b71ed12",
        "prediction_id": "f0e9d8c7-b6a5-4948-83a1-72635849affe",
        "risk_category": "Medium",
        "confidence":    0.83,
        "probabilities": {"Low": 0.06, "Medium": 0.83, "High": 0.11},
        "model_version": "part1_classifier_v1.0",
        "created_at":    "2026-06-20T14:21:09Z",
    }}}


# ---------------------------------------------------------------------------
# §3  POST /api/v1/recommend
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    """
    Request body for POST /api/v1/recommend (§3).
    Only two fields — the endpoint looks up everything else via prediction_id.
    """
    prediction_id:  str  = Field(..., description="UUID from POST /predict response.")
    has_arrhythmia: bool = Field(False, description="Clinician-entered flag. True → triggers SBRT branch for High-risk patients.")

    model_config = {"json_schema_extra": {"example": {
        "prediction_id": "f0e9d8c7-b6a5-4948-83a1-72635849affe",
        "has_arrhythmia": False,
    }}}


class RecommendResponse(BaseModel):
    """Response body for POST /api/v1/recommend (§3)."""
    recommendation_id:   str            = Field(..., description="Server-generated UUID v4.")
    patient_id:          str            = Field(..., description="UUID of the patient.")
    prediction_id:       str            = Field(..., description="UUID of the linked prediction.")
    branch:              str            = Field(..., description="SBRT | Medication.")
    intervention_type:   str            = Field(..., description="Machine-readable intervention slug.")
    intensity:           str            = Field(..., description="Low | Moderate | High.")
    bed_gy:              Optional[float]= Field(None, description="BED in Gy (SBRT branch only).")
    bed_valid:           Optional[bool] = Field(None, description="BED meets ablative threshold (SBRT only).")
    grace_score:         Optional[int]  = Field(None, description="GRACE score (Medication branch only).")
    grace_risk_category: Optional[str]  = Field(None, description="GRACE category (Medication branch only).")
    survival_without:    float          = Field(..., ge=0, le=1)
    survival_with:       float          = Field(..., ge=0, le=1)
    model_version:       str
    created_at:          datetime       = Field(..., description="RFC 3339 UTC timestamp.")

    model_config = {"json_schema_extra": {"example": {
        "recommendation_id":   "11111111-2222-3333-4444-555555555555",
        "patient_id":          "9c1b7e64-3a4f-4d3a-a4c8-2e5e7b71ed12",
        "prediction_id":       "f0e9d8c7-b6a5-4948-83a1-72635849affe",
        "branch":              "SBRT",
        "intervention_type":   "cardiac_sbrt_25Gy_1fx",
        "intensity":           "High",
        "bed_gy":              87.5,
        "bed_valid":           True,
        "grace_score":         None,
        "grace_risk_category": None,
        "survival_without":    0.58,
        "survival_with":       0.81,
        "model_version":       "part2_recommender_v1.0",
        "created_at":          "2026-06-20T14:21:11Z",
    }}}


# ---------------------------------------------------------------------------
# §4  GET /api/v1/history
# ---------------------------------------------------------------------------

class HistoryItem(BaseModel):
    """One row in the history list (§4)."""
    prediction_id:       str            = Field(..., description="UUID.")
    patient_id:          str            = Field(..., description="UUID.")
    created_at:          datetime       = Field(..., description="RFC 3339 UTC.")
    age:                 int
    risk_category:       str
    confidence:          float          = Field(..., ge=0, le=1)
    branch:              str            = Field(..., description="SBRT | Medication.")
    intervention_type:   str
    survival_without:    float          = Field(..., ge=0, le=1)
    survival_with:       float          = Field(..., ge=0, le=1)
    survival_improvement:float          = Field(..., description="survival_with − survival_without, computed server-side.")

    model_config = {"json_schema_extra": {"example": {
        "prediction_id":       "f0e9d8c7-b6a5-4948-83a1-72635849affe",
        "patient_id":          "9c1b7e64-3a4f-4d3a-a4c8-2e5e7b71ed12",
        "created_at":          "2026-06-20T14:21:09Z",
        "age":                 56,
        "risk_category":       "Medium",
        "confidence":          0.83,
        "branch":              "Medication",
        "intervention_type":   "beta_blocker+moderate_statin+aspirin",
        "survival_without":    0.81,
        "survival_with":       0.92,
        "survival_improvement":0.11,
    }}}


class HistoryResponse(BaseModel):
    """Response body for GET /api/v1/history (§4)."""
    page:  int              = Field(..., ge=1, description="Current 1-indexed page.")
    size:  int              = Field(..., ge=1, description="Page size requested.")
    total: int              = Field(..., description="Total records available.")
    items: list[HistoryItem]

    model_config = {"json_schema_extra": {"example": {
        "page": 1, "size": 20, "total": 47, "items": [],
    }}}


# ---------------------------------------------------------------------------
# §5  GET /api/v1/patients/{patient_id}
# ---------------------------------------------------------------------------

class PatientVitals(BaseModel):
    """Vitals sub-object inside PatientFullResponse (§5)."""
    age:             int
    sex:             str
    chest_pain_type: str
    resting_bp:      int
    cholesterol:     int
    fasting_bs:      int
    resting_ecg:     str
    max_hr:          int
    exercise_angina: str
    oldpeak:         float
    st_slope:        str


class PredictionRecord(BaseModel):
    """One prediction entry inside PatientFullResponse (§5)."""
    prediction_id: str
    risk_category: str
    confidence:    float          = Field(..., ge=0, le=1)
    probabilities: RiskProbabilities
    model_version: str
    created_at:    datetime


class RecommendationRecord(BaseModel):
    """One recommendation entry inside PatientFullResponse (§5)."""
    recommendation_id:   str
    branch:              str
    intervention_type:   str
    intensity:           str
    bed_gy:              Optional[float] = None
    bed_valid:           Optional[bool]  = None
    grace_score:         Optional[int]   = None
    grace_risk_category: Optional[str]   = None
    survival_without:    float
    survival_with:       float
    model_version:       str
    created_at:          datetime


class PatientFullResponse(BaseModel):
    """Response body for GET /api/v1/patients/{patient_id} (§5)."""
    patient_id:      str
    created_at:      datetime
    vitals:          PatientVitals
    predictions:     list[PredictionRecord]
    recommendations: list[RecommendationRecord]

    model_config = {"json_schema_extra": {"example": {
        "patient_id": "9c1b7e64-3a4f-4d3a-a4c8-2e5e7b71ed12",
        "created_at": "2026-06-20T14:20:55Z",
        "vitals": {
            "age": 56, "sex": "M", "chest_pain_type": "ATA",
            "resting_bp": 138, "cholesterol": 230, "fasting_bs": 0,
            "resting_ecg": "Normal", "max_hr": 150, "exercise_angina": "N",
            "oldpeak": 1.2, "st_slope": "Up",
        },
        "predictions": [{
            "prediction_id": "f0e9...", "risk_category": "Medium",
            "confidence": 0.83,
            "probabilities": {"Low": 0.06, "Medium": 0.83, "High": 0.11},
            "model_version": "part1_classifier_v1.0",
            "created_at": "2026-06-20T14:21:09Z",
        }],
        "recommendations": [{
            "recommendation_id": "1111...", "branch": "Medication",
            "intervention_type": "beta_blocker+moderate_statin+aspirin",
            "intensity": "Moderate",
            "bed_gy": None, "bed_valid": None,
            "grace_score": 132, "grace_risk_category": "Intermediate",
            "survival_without": 0.81, "survival_with": 0.92,
            "model_version": "part2_recommender_v1.0",
            "created_at": "2026-06-20T14:21:11Z",
        }],
    }}}
