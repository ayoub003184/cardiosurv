"""
src/api/schemas.py
------------------
Pydantic request/response models for the CardioSurv API (v1).

Every model here corresponds 1-to-1 with a shape in docs/api_contract.md.
Fields that are None in mock responses are typed Optional so FastAPI renders
them in the Swagger schema rather than hiding them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared / primitives
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Returned on 4xx / 5xx errors."""
    detail: str = Field(..., description="Human-readable error message.")
    code: str   = Field(..., description="Machine-readable error code.", examples=["NOT_FOUND"])

    model_config = {"json_schema_extra": {"example": {
        "detail": "Patient P999 not found.",
        "code": "NOT_FOUND",
    }}}


# ---------------------------------------------------------------------------
# 1. Health
# ---------------------------------------------------------------------------

class ModelVersions(BaseModel):
    part1_classifier:  str = Field("v1.0", description="Risk classifier version.")
    part2_recommender: str = Field("v1.0", description="Intervention recommender version.")


class HealthResponse(BaseModel):
    """GET /api/v1/health"""
    status:          str           = Field(..., description="'ok' when the service is healthy.")
    version:         str           = Field(..., description="API version string.")
    model_versions:  ModelVersions
    uptime_seconds:  float         = Field(..., description="Seconds since process start.")

    model_config = {"json_schema_extra": {"example": {
        "status": "ok",
        "version": "v1",
        "model_versions": {"part1_classifier": "v1.0", "part2_recommender": "v1.0"},
        "uptime_seconds": 42.3,
    }}}


# ---------------------------------------------------------------------------
# 2. Predict
# ---------------------------------------------------------------------------

class PatientVitalsRequest(BaseModel):
    """
    POST /api/v1/predict  — request body.
    Vital signs and clinical flags collected at admission.
    """
    patient_id:         str   = Field(...,  description="Unique patient identifier.")
    age:                int   = Field(...,  ge=1,  le=120, description="Age in years.")
    heart_rate:         float = Field(...,  ge=1,  le=300, description="Heart rate (bpm).")
    systolic_bp:        float = Field(...,  ge=1,  le=350, description="Systolic blood pressure (mmHg).")
    diastolic_bp:       float = Field(...,  ge=1,  le=250, description="Diastolic blood pressure (mmHg).")
    creatinine_umol_l:  float = Field(...,  ge=0,          description="Serum creatinine (µmol/L).")
    killip_class:       int   = Field(1,    ge=1,  le=4,  description="Killip class I–IV.")
    has_arrhythmia:     bool  = Field(False,               description="Documented arrhythmia at admission.")
    cardiac_arrest:     bool  = Field(False,               description="Cardiac arrest at admission.")
    st_deviation:       bool  = Field(False,               description="ST-segment deviation on ECG.")
    elevated_enzymes:   bool  = Field(False,               description="Elevated cardiac enzymes/markers.")

    model_config = {"json_schema_extra": {"example": {
        "patient_id": "P001",
        "age": 68,
        "heart_rate": 92,
        "systolic_bp": 130,
        "diastolic_bp": 85,
        "creatinine_umol_l": 110,
        "killip_class": 1,
        "has_arrhythmia": False,
        "cardiac_arrest": False,
        "st_deviation": True,
        "elevated_enzymes": True,
    }}}


class RiskProbabilities(BaseModel):
    Low:    float = Field(..., ge=0, le=1)
    Medium: float = Field(..., ge=0, le=1)
    High:   float = Field(..., ge=0, le=1)


class PredictResponse(BaseModel):
    """POST /api/v1/predict  — response body."""
    patient_id:     str              = Field(..., description="Echoed patient identifier.")
    risk_category:  str              = Field(..., description="Predicted risk class: Low | Medium | High.")
    confidence:     float            = Field(..., ge=0, le=1, description="Confidence in the predicted class.")
    probabilities:  RiskProbabilities
    model_version:  str              = Field(..., description="Classifier model version tag.")
    mock:           bool             = Field(True, description="True while real model is pending.")

    model_config = {"json_schema_extra": {"example": {
        "patient_id": "P001",
        "risk_category": "Medium",
        "confidence": 0.83,
        "probabilities": {"Low": 0.06, "Medium": 0.83, "High": 0.11},
        "model_version": "part1_classifier_v1.0_MOCK",
        "mock": True,
    }}}


# ---------------------------------------------------------------------------
# 3. Recommend
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    """
    POST /api/v1/recommend  — request body.
    Combines the predict output with raw vitals so the recommender
    has both risk label and clinical context.
    """
    patient_id:        str   = Field(...,  description="Unique patient identifier.")
    risk_category:     str   = Field(...,  description="Risk class from /predict: Low | Medium | High.")
    confidence:        float = Field(...,  ge=0, le=1)
    age:               int   = Field(...,  ge=1, le=120)
    heart_rate:        float = Field(...,  ge=1, le=300)
    systolic_bp:       float = Field(...,  ge=1, le=350)
    creatinine_umol_l: float = Field(...,  ge=0)
    killip_class:      int   = Field(1,    ge=1, le=4)
    has_arrhythmia:    bool  = Field(False, description="Routes High/Medium+arrhythmia → SBRT.")
    cardiac_arrest:    bool  = Field(False)
    st_deviation:      bool  = Field(False)
    elevated_enzymes:  bool  = Field(False)

    model_config = {"json_schema_extra": {"example": {
        "patient_id": "P001",
        "risk_category": "High",
        "confidence": 0.91,
        "age": 68,
        "heart_rate": 92,
        "systolic_bp": 130,
        "creatinine_umol_l": 110,
        "killip_class": 1,
        "has_arrhythmia": True,
        "cardiac_arrest": False,
        "st_deviation": True,
        "elevated_enzymes": True,
    }}}


class RecommendResponse(BaseModel):
    """POST /api/v1/recommend  — response body."""
    patient_id:           str            = Field(..., description="Echoed patient identifier.")
    branch:               str            = Field(..., description="Routing branch: SBRT | Medication.")
    intervention_type:    str            = Field(..., description="Human-readable intervention label.")
    intensity:            Optional[str]  = Field(None, description="Medication intensity tier (Medication branch only).")
    bed_gy:               Optional[float]= Field(None, description="Biologically Effective Dose in Gy (SBRT branch only).")
    bed_valid:            Optional[bool] = Field(None, description="Whether BED meets ablative threshold.")
    grace_score:          Optional[int]  = Field(None, description="GRACE risk score (Medication branch only).")
    grace_risk_category:  Optional[str]  = Field(None, description="GRACE risk category (Medication branch only).")
    survival_without:     float          = Field(..., ge=0, le=1, description="2-year survival probability without intervention.")
    survival_with:        float          = Field(..., ge=0, le=1, description="2-year survival probability with intervention.")
    counterfactual:       str            = Field(..., description="Plain-English counterfactual summary.")
    model_version:        str            = Field(..., description="Recommender model version tag.")
    mock:                 bool           = Field(True)

    model_config = {"json_schema_extra": {"example": {
        "patient_id": "P001",
        "branch": "SBRT",
        "intervention_type": "Cardiac Radioablation (SBRT)",
        "intensity": None,
        "bed_gy": 87.5,
        "bed_valid": False,
        "grace_score": None,
        "grace_risk_category": None,
        "survival_without": 0.54,
        "survival_with": 0.81,
        "counterfactual": "Without intervention: 46% mortality risk. With SBRT at 25 Gy: 19% mortality risk.",
        "model_version": "part2_recommender_v1.0_MOCK",
        "mock": True,
    }}}


# ---------------------------------------------------------------------------
# 4. History
# ---------------------------------------------------------------------------

class HistoryItem(BaseModel):
    """A single historical prediction record."""
    record_id:         str      = Field(..., description="Unique record identifier.")
    patient_id:        str
    timestamp:         datetime = Field(..., description="ISO-8601 UTC timestamp of the prediction.")
    risk_category:     str
    confidence:        float    = Field(..., ge=0, le=1)
    branch:            str      = Field(..., description="SBRT | Medication")
    intervention_type: str
    survival_with:     float    = Field(..., ge=0, le=1)

    model_config = {"json_schema_extra": {"example": {
        "record_id": "REC-001",
        "patient_id": "P001",
        "timestamp": "2025-11-01T09:14:00Z",
        "risk_category": "High",
        "confidence": 0.91,
        "branch": "SBRT",
        "intervention_type": "Cardiac Radioablation (SBRT)",
        "survival_with": 0.81,
    }}}


class HistoryResponse(BaseModel):
    """GET /api/v1/history  — response body."""
    total:   int              = Field(..., description="Total records matching the query.")
    offset:  int              = Field(..., ge=0)
    limit:   int              = Field(..., ge=1)
    records: list[HistoryItem]

    model_config = {"json_schema_extra": {"example": {
        "total": 10,
        "offset": 0,
        "limit": 10,
        "records": [],
    }}}


# ---------------------------------------------------------------------------
# 5. Patient full record
# ---------------------------------------------------------------------------

class PatientFullResponse(BaseModel):
    """GET /api/v1/patients/{patient_id}  — response body."""
    patient_id:           str
    age:                  int
    heart_rate:           float
    systolic_bp:          float
    diastolic_bp:         float
    creatinine_umol_l:    float
    killip_class:         int
    has_arrhythmia:       bool
    cardiac_arrest:       bool
    st_deviation:         bool
    elevated_enzymes:     bool
    risk_category:        str
    confidence:           float          = Field(..., ge=0, le=1)
    branch:               str
    intervention_type:    str
    bed_gy:               Optional[float]= None
    grace_score:          Optional[int]  = None
    survival_without:     float          = Field(..., ge=0, le=1)
    survival_with:        float          = Field(..., ge=0, le=1)
    mock:                 bool           = True

    model_config = {"json_schema_extra": {"example": {
        "patient_id": "P001",
        "age": 68,
        "heart_rate": 92,
        "systolic_bp": 130,
        "diastolic_bp": 85,
        "creatinine_umol_l": 110,
        "killip_class": 1,
        "has_arrhythmia": True,
        "cardiac_arrest": False,
        "st_deviation": True,
        "elevated_enzymes": True,
        "risk_category": "High",
        "confidence": 0.91,
        "branch": "SBRT",
        "intervention_type": "Cardiac Radioablation (SBRT)",
        "bed_gy": 87.5,
        "grace_score": None,
        "survival_without": 0.54,
        "survival_with": 0.81,
        "mock": True,
    }}}
