"""
CardioSurv – Part 1 Risk Classifier
====================================
Trains a Random Forest and an XGBoost classifier on features.csv,
evaluates both, saves the winner, and exposes a predict() function
the API layer can call with a single dict.

Usage
-----
    python -m src.models.part1_classifier

    # or from another module:
    from src.models.part1_classifier import predict, load
    model = load("models/part1_classifier_v1.0.pkl")
    result = predict({"age": 56, "sex": "M", ...})
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import xgboost as xgb

# ---------------------------------------------------------------------------
# Paths (relative to project root; override with env vars if needed)
# ---------------------------------------------------------------------------
DATA_PATH   = Path(os.getenv("FEATURES_CSV",  "data/processed/features.csv"))
MODEL_PATH  = Path(os.getenv("MODEL_PATH",    "models/part1_classifier_v1.0.pkl"))
MODEL_VERSION = "part1_classifier_v1.0"

# ---------------------------------------------------------------------------
# Feature / label configuration
# ---------------------------------------------------------------------------
TARGET = "RiskCategory"
LABEL_ORDER = ["Low", "Medium", "High"]   # fixed class index 0/1/2

NUMERIC_FEATURES = [
    "Age", "RestingBP", "Cholesterol", "FastingBS",
    "MaxHR", "Oldpeak", "HeartRateStressIndex",
]
CATEGORICAL_FEATURES = [
    "Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope",
    "AgeBin", "BP_RiskLevel",
]

# ---------------------------------------------------------------------------
# 1.  Preprocessing pipeline
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Return a ColumnTransformer that:
      • StandardScaler  → numeric columns
      • OneHotEncoder   → categorical columns (drop='first' to avoid dummy trap)
    """
    numeric_transformer = Pipeline([
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer,  NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",   # drop HeartDisease, raw RiskCategory, etc.
    )
    return preprocessor


# ---------------------------------------------------------------------------
# 2.  Model factories
# ---------------------------------------------------------------------------

def train_random_forest(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def train_xgboost(X: np.ndarray, y: np.ndarray) -> xgb.XGBClassifier:
    clf = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    clf.fit(X, y)
    return clf


# ---------------------------------------------------------------------------
# 3.  Evaluation helper
# ---------------------------------------------------------------------------

def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    """
    Returns a dict with:
        accuracy, f1_macro, auc_ovr, confusion_matrix (as nested list)
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc      = round(accuracy_score(y_test, y_pred), 4)
    f1_macro = round(f1_score(y_test, y_pred, average="macro"), 4)
    auc_ovr  = round(
        roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"), 4
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "accuracy":         acc,
        "f1_macro":         f1_macro,
        "auc_ovr":          auc_ovr,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# 4.  Inference  (this is what the API calls)
# ---------------------------------------------------------------------------

def predict(features_dict: dict, model_bundle: dict | None = None) -> dict:
    """
    Accept a flat dict of raw patient features and return the JSON shape
    defined in docs/schemas.md §4:

        {
          "risk_category": "Medium",
          "confidence": 0.83,
          "probabilities": {"Low": 0.06, "Medium": 0.83, "High": 0.11},
          "model_version": "part1_classifier_v1.0"
        }

    If model_bundle is None the model is loaded from MODEL_PATH automatically.
    """
    if model_bundle is None:
        model_bundle = load(MODEL_PATH)

    pipeline: Pipeline    = model_bundle["pipeline"]
    le:       LabelEncoder = model_bundle["label_encoder"]

    # Build a single-row DataFrame so the ColumnTransformer gets named columns
    row = pd.DataFrame([features_dict])

    # Ensure expected columns are present; fill missing with sensible defaults
    for col in NUMERIC_FEATURES:
        if col not in row.columns:
            row[col] = 0.0
    for col in CATEGORICAL_FEATURES:
        if col not in row.columns:
            row[col] = "Unknown"

    proba_array = pipeline.predict_proba(row)[0]          # shape (3,)
    class_idx   = int(np.argmax(proba_array))
    risk_label  = le.inverse_transform([class_idx])[0]

    proba_dict = {
        label: round(float(p), 4)
        for label, p in zip(le.classes_, proba_array)
    }

    return {
        "risk_category": risk_label,
        "confidence":    round(float(proba_array[class_idx]), 4),
        "probabilities": proba_dict,
        "model_version": model_bundle.get("version", MODEL_VERSION),
    }


# ---------------------------------------------------------------------------
# 5.  Persistence
# ---------------------------------------------------------------------------

def save(bundle: dict, path: str | Path = MODEL_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    print(f"[part1] Saved {path}")


def load(path: str | Path = MODEL_PATH) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python -m src.models.part1_classifier` first."
        )
    return joblib.load(path)


# ---------------------------------------------------------------------------
# 6.  main()  –  load → split → train both → compare → save winner
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Load ────────────────────────────────────────────────────────────────
    print(f"[part1] Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"[part1] Dataset shape: {df.shape}")

    # ── Prepare X and y ─────────────────────────────────────────────────────
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_raw = df[feature_cols].copy()

    le = LabelEncoder()
    le.classes_ = np.array(LABEL_ORDER)          # fix class order Low=0, Med=1, High=2
    y = le.transform(df[TARGET])

    # ── Train / test split  (stratified 80 / 20) ────────────────────────────
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[part1] Train: {len(X_train_raw)} rows, Test: {len(X_test_raw)} rows")

    # ── Build preprocessor ──────────────────────────────────────────────────
    preprocessor = build_preprocessing_pipeline()

    # ── Random Forest ───────────────────────────────────────────────────────
    print("[part1] Training Random Forest ...")
    t0 = time.time()
    rf_pipeline = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=None,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
    ])
    rf_pipeline.fit(X_train_raw, y_train)
    rf_metrics = evaluate(rf_pipeline, X_test_raw, y_test)
    print(
        f"[part1] RF  → acc={rf_metrics['accuracy']}  "
        f"f1_macro={rf_metrics['f1_macro']}  auc={rf_metrics['auc_ovr']}  "
        f"({time.time()-t0:.1f}s)"
    )

    # ── XGBoost ─────────────────────────────────────────────────────────────
    print("[part1] Training XGBoost ...")
    t0 = time.time()

    # Fit preprocessor on training data using the RF pipeline's fitted preprocessor
    # (already fitted above — reuse it for XGB to keep transforms identical)
    xgb_preprocessor = build_preprocessing_pipeline()
    xgb_pipeline = Pipeline([
        ("pre", xgb_preprocessor),
        ("clf", xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0,
        )),
    ])
    xgb_pipeline.fit(X_train_raw, y_train)
    xgb_metrics = evaluate(xgb_pipeline, X_test_raw, y_test)
    print(
        f"[part1] XGB → acc={xgb_metrics['accuracy']}  "
        f"f1_macro={xgb_metrics['f1_macro']}  auc={xgb_metrics['auc_ovr']}  "
        f"({time.time()-t0:.1f}s)"
    )

    # ── Pick winner ─────────────────────────────────────────────────────────
    if xgb_metrics["f1_macro"] >= rf_metrics["f1_macro"]:
        winner_name    = "XGBoost"
        winner_pipeline = xgb_pipeline
        winner_metrics  = xgb_metrics
    else:
        winner_name    = "RandomForest"
        winner_pipeline = rf_pipeline
        winner_metrics  = rf_metrics

    print(f"[part1] Winner: {winner_name}")

    # ── Save bundle ─────────────────────────────────────────────────────────
    bundle = {
        "pipeline":      winner_pipeline,
        "label_encoder": le,
        "version":       MODEL_VERSION,
        "winner":        winner_name,
        "metrics": {
            "RandomForest": rf_metrics,
            "XGBoost":      xgb_metrics,
        },
        "feature_cols": feature_cols,
    }
    save(bundle, MODEL_PATH)
    print(f"[part1] Saved models/part1_classifier_v1.0.pkl")

    # ── Confusion matrix summary ─────────────────────────────────────────────
    print("\n[part1] Confusion Matrix (rows=actual, cols=predicted):")
    print(f"        {'Low':>6}  {'Medium':>6}  {'High':>6}")
    for i, row_label in enumerate(LABEL_ORDER):
        row_vals = winner_metrics["confusion_matrix"][i]
        print(f"  {row_label:<7}  {row_vals[0]:>5}   {row_vals[1]:>5}   {row_vals[2]:>5}")

    # ── Quick smoke-test with predict() ─────────────────────────────────────
    print("\n[part1] Smoke-test predict():")
    test_patient = {
        "Age": 67, "Sex": "M", "ChestPainType": "ASY",
        "RestingBP": 162, "Cholesterol": 268, "FastingBS": 1,
        "RestingECG": "ST", "MaxHR": 100, "ExerciseAngina": "Y",
        "Oldpeak": 2.5, "ST_Slope": "Flat",
        "AgeBin": "60-69", "BP_RiskLevel": "Stage2",
        "HeartRateStressIndex": round(100 / (220 - 67), 3),
    }
    result = predict(test_patient, model_bundle=bundle)
    print(f"  Input:  67-y/o male, high BP, exercise angina, ST depression")
    print(f"  Output: {json.dumps(result, indent=4)}")


if __name__ == "__main__":
    main()
