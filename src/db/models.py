"""
src/db/models.py
----------------
SQLAlchemy ORM models for CardioSurv.
4 tables as defined in docs/schemas.md §6:
  - patients
  - predictions
  - recommendations
  - audit_logs

Usage:
    from src.db.session import engine
    from src.db.models import Base
    Base.metadata.create_all(engine)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
)
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Table 1: patients
# ---------------------------------------------------------------------------

class Patient(Base):
    __tablename__ = "patients"

    id               = Column(String,      primary_key=True, default=_uuid)
    created_at       = Column(DateTime,    default=_now, nullable=False)
    age              = Column(Integer,     nullable=False)
    sex              = Column(String(1),   nullable=False)           # M / F
    chest_pain_type  = Column(String(3),   nullable=False)           # TA / ATA / NAP / ASY
    resting_bp       = Column(Integer,     nullable=False)
    cholesterol      = Column(Integer,     nullable=False)
    fasting_bs       = Column(Integer,     nullable=False)           # 0 or 1
    resting_ecg      = Column(String(8),   nullable=False)           # Normal / ST / LVH
    max_hr           = Column(Integer,     nullable=False)
    exercise_angina  = Column(String(1),   nullable=False)           # N / Y
    oldpeak          = Column(Numeric(4, 2), nullable=False)
    st_slope         = Column(String(5),   nullable=False)           # Up / Flat / Down

    # Relationships
    predictions      = relationship("Prediction",    back_populates="patient",
                                    cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Patient id={self.id} age={self.age} sex={self.sex}>"


# ---------------------------------------------------------------------------
# Table 2: predictions
# ---------------------------------------------------------------------------

class Prediction(Base):
    __tablename__ = "predictions"

    id             = Column(String,       primary_key=True, default=_uuid)
    patient_id     = Column(String,       ForeignKey("patients.id"), nullable=False)
    risk_category  = Column(String(8),    nullable=False)            # Low / Medium / High
    confidence     = Column(Numeric(4, 3), nullable=False)
    probabilities  = Column(JSON,         nullable=False)            # {"Low":0.x, "Medium":0.x, "High":0.x}
    model_version  = Column(String(64),   nullable=False)
    created_at     = Column(DateTime,     default=_now, nullable=False)

    # Relationships
    patient         = relationship("Patient",        back_populates="predictions")
    recommendations = relationship("Recommendation", back_populates="prediction",
                                   cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return (
            f"<Prediction id={self.id} patient_id={self.patient_id} "
            f"risk={self.risk_category} conf={self.confidence}>"
        )


# ---------------------------------------------------------------------------
# Table 3: recommendations
# ---------------------------------------------------------------------------

class Recommendation(Base):
    __tablename__ = "recommendations"

    id                   = Column(String,        primary_key=True, default=_uuid)
    prediction_id        = Column(String,        ForeignKey("predictions.id"), nullable=False)
    branch               = Column(String(16),    nullable=False)     # SBRT / Medication
    intervention_type    = Column(String(128),   nullable=False)
    intensity            = Column(String(8),     nullable=False)     # Low / Moderate / High
    bed_gy               = Column(Numeric(6, 2), nullable=True)      # NULL for Medication branch
    bed_valid            = Column(Boolean,       nullable=True)      # NULL for Medication branch
    grace_score          = Column(Integer,       nullable=True)      # NULL for SBRT branch
    grace_risk_category  = Column(String(16),    nullable=True)      # NULL for SBRT branch
    survival_without     = Column(Numeric(4, 3), nullable=False)
    survival_with        = Column(Numeric(4, 3), nullable=False)
    model_version        = Column(String(64),    nullable=False)
    created_at           = Column(DateTime,      default=_now, nullable=False)

    # Relationships
    prediction = relationship("Prediction", back_populates="recommendations")

    def __repr__(self) -> str:
        return (
            f"<Recommendation id={self.id} branch={self.branch} "
            f"intervention={self.intervention_type}>"
        )


# ---------------------------------------------------------------------------
# Table 4: audit_logs
# ---------------------------------------------------------------------------

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id          = Column(BigInteger, primary_key=True, autoincrement=True)
    route       = Column(String(64),  nullable=False)
    status_code = Column(Integer,     nullable=False)
    request_ip  = Column(String(45),  nullable=True)   # IPv4 or IPv6 string
    latency_ms  = Column(Integer,     nullable=True)
    created_at  = Column(DateTime,    default=_now, nullable=False)

    def __repr__(self) -> str:
        return (
            f"<AuditLog id={self.id} route={self.route} "
            f"status={self.status_code} latency={self.latency_ms}ms>"
        )
