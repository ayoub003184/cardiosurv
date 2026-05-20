"""
src/db/seed.py
--------------
Inserts 10 sample patients (and their predictions) into the database for testing.

Run:
    python -m src.db.seed

Expected output:
    [seed] Creating tables if they do not exist...
    [seed] Inserted 10 patients into the database
"""

from __future__ import annotations

from src.db.models import Base, Patient, Prediction
from src.db.session import SessionLocal, engine

# ---------------------------------------------------------------------------
# Sample data (10 patients covering Low / Medium / High risk)
# ---------------------------------------------------------------------------

SAMPLE_PATIENTS = [
    {
        "age": 56, "sex": "M", "chest_pain_type": "ATA", "resting_bp": 138,
        "cholesterol": 230, "fasting_bs": 0, "resting_ecg": "Normal",
        "max_hr": 150, "exercise_angina": "N", "oldpeak": 1.2, "st_slope": "Up",
    },
    {
        "age": 72, "sex": "M", "chest_pain_type": "ASY", "resting_bp": 160,
        "cholesterol": 280, "fasting_bs": 1, "resting_ecg": "LVH",
        "max_hr": 110, "exercise_angina": "Y", "oldpeak": 3.5, "st_slope": "Flat",
    },
    {
        "age": 45, "sex": "F", "chest_pain_type": "NAP", "resting_bp": 120,
        "cholesterol": 200, "fasting_bs": 0, "resting_ecg": "Normal",
        "max_hr": 170, "exercise_angina": "N", "oldpeak": 0.5, "st_slope": "Up",
    },
    {
        "age": 62, "sex": "M", "chest_pain_type": "ASY", "resting_bp": 145,
        "cholesterol": 260, "fasting_bs": 1, "resting_ecg": "ST",
        "max_hr": 120, "exercise_angina": "Y", "oldpeak": 2.8, "st_slope": "Flat",
    },
    {
        "age": 38, "sex": "F", "chest_pain_type": "TA",  "resting_bp": 115,
        "cholesterol": 185, "fasting_bs": 0, "resting_ecg": "Normal",
        "max_hr": 182, "exercise_angina": "N", "oldpeak": 0.0, "st_slope": "Up",
    },
    {
        "age": 67, "sex": "M", "chest_pain_type": "ASY", "resting_bp": 162,
        "cholesterol": 268, "fasting_bs": 1, "resting_ecg": "ST",
        "max_hr": 100, "exercise_angina": "Y", "oldpeak": 2.5, "st_slope": "Flat",
    },
    {
        "age": 50, "sex": "F", "chest_pain_type": "ATA", "resting_bp": 128,
        "cholesterol": 210, "fasting_bs": 0, "resting_ecg": "Normal",
        "max_hr": 160, "exercise_angina": "N", "oldpeak": 0.8, "st_slope": "Up",
    },
    {
        "age": 74, "sex": "M", "chest_pain_type": "ASY", "resting_bp": 175,
        "cholesterol": 295, "fasting_bs": 1, "resting_ecg": "LVH",
        "max_hr": 95,  "exercise_angina": "Y", "oldpeak": 4.0, "st_slope": "Down",
    },
    {
        "age": 42, "sex": "F", "chest_pain_type": "NAP", "resting_bp": 118,
        "cholesterol": 195, "fasting_bs": 0, "resting_ecg": "Normal",
        "max_hr": 175, "exercise_angina": "N", "oldpeak": 0.2, "st_slope": "Up",
    },
    {
        "age": 59, "sex": "M", "chest_pain_type": "ASY", "resting_bp": 142,
        "cholesterol": 245, "fasting_bs": 0, "resting_ecg": "ST",
        "max_hr": 130, "exercise_angina": "Y", "oldpeak": 1.8, "st_slope": "Flat",
    },
]

# Matching mock predictions for each patient
SAMPLE_PREDICTIONS = [
    {"risk_category": "Medium", "confidence": 0.830, "probabilities": {"Low": 0.06, "Medium": 0.83, "High": 0.11}, "model_version": "part1_classifier_v1.0"},
    {"risk_category": "High",   "confidence": 0.910, "probabilities": {"Low": 0.02, "Medium": 0.07, "High": 0.91}, "model_version": "part1_classifier_v1.0"},
    {"risk_category": "Low",    "confidence": 0.870, "probabilities": {"Low": 0.87, "Medium": 0.10, "High": 0.03}, "model_version": "part1_classifier_v1.0"},
    {"risk_category": "High",   "confidence": 0.880, "probabilities": {"Low": 0.03, "Medium": 0.09, "High": 0.88}, "model_version": "part1_classifier_v1.0"},
    {"risk_category": "Low",    "confidence": 0.920, "probabilities": {"Low": 0.92, "Medium": 0.06, "High": 0.02}, "model_version": "part1_classifier_v1.0"},
    {"risk_category": "High",   "confidence": 0.950, "probabilities": {"Low": 0.01, "Medium": 0.04, "High": 0.95}, "model_version": "part1_classifier_v1.0"},
    {"risk_category": "Low",    "confidence": 0.850, "probabilities": {"Low": 0.85, "Medium": 0.11, "High": 0.04}, "model_version": "part1_classifier_v1.0"},
    {"risk_category": "High",   "confidence": 0.960, "probabilities": {"Low": 0.01, "Medium": 0.03, "High": 0.96}, "model_version": "part1_classifier_v1.0"},
    {"risk_category": "Low",    "confidence": 0.900, "probabilities": {"Low": 0.90, "Medium": 0.07, "High": 0.03}, "model_version": "part1_classifier_v1.0"},
    {"risk_category": "Medium", "confidence": 0.740, "probabilities": {"Low": 0.09, "Medium": 0.74, "High": 0.17}, "model_version": "part1_classifier_v1.0"},
]


# ---------------------------------------------------------------------------
# Seed function
# ---------------------------------------------------------------------------

def seed() -> None:
    print("[seed] Creating tables if they do not exist...")
    Base.metadata.create_all(engine)

    db = SessionLocal()
    try:
        # Skip if already seeded
        existing = db.query(Patient).count()
        if existing >= 10:
            print(f"[seed] Database already has {existing} patients — skipping seed.")
            return

        for patient_data, pred_data in zip(SAMPLE_PATIENTS, SAMPLE_PREDICTIONS):
            patient = Patient(**patient_data)
            db.add(patient)
            db.flush()  # get patient.id before committing

            prediction = Prediction(
                patient_id=patient.id,
                **pred_data,
            )
            db.add(prediction)

        db.commit()
        print("[seed] Inserted 10 patients into the database")

    except Exception as exc:
        db.rollback()
        print(f"[seed] ERROR: {exc}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed()
