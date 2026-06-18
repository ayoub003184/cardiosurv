"""
CardioSurv – Part 1 Risk Classifier
====================================
Trains a Random Forest and an XGBoost classifier on features.csv,
evaluates both, saves the winner, and exposes a predict() function
the API layer can call with a single dict.

Also generates four diagnostic figures to reports/figures/:
    confusion_matrix_rf.png
    confusion_matrix_xgb.png
    feature_importance_rf.png
    feature_importance_xgb.png

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
import matplotlib
matplotlib.use("Agg")  # headless-safe backend; must be set before pyplot import
import matplotlib.pyplot as plt
import seaborn as sns
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
DATA_PATH     = Path(os.getenv("FEATURES_CSV",  "data/processed/features.csv"))
MODEL_PATH    = Path(os.getenv("MODEL_PATH",    "models/part1_classifier_v1.0.pkl"))
FIGURES_DIR   = Path(os.getenv("FIGURES_DIR",   "reports/figures"))
MODEL_VERSION = "part1_classifier_v1.0"

# Shared colour palette (kept consistent with src/evaluation/plots.py)
PALETTE = {
    "Low": "#2ecc71",
    "Medium": "#f39c12",
    "High": "#e74c3c",
    "primary": "#3498db",
    "secondary": "#9b59b6",
}

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
# 3b.  Figure generation  (confusion matrices + feature importance)
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: list[list[int]],
    labels: list[str],
    title: str,
    save_path: str | Path,
):
    """
    Save an annotated confusion-matrix heatmap (rows=actual, cols=predicted).
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm_arr = np.array(cm)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[part1] Saved {save_path}")


def plot_feature_importance(
    model,
    feature_names: list[str],
    title: str,
    save_path: str | Path,
    top_n: int = 15,
):
    """
    Save a horizontal bar chart of the top_n feature importances.
    Works for any estimator exposing `feature_importances_`
    (RandomForestClassifier and XGBClassifier both do).
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    importances = np.asarray(model.feature_importances_)
    order = np.argsort(importances)[::-1][:top_n]
    top_features    = [feature_names[i] for i in order][::-1]   # reversed for barh top-down
    top_importances = importances[order][::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_features, top_importances, color=PALETTE["primary"])
    ax.set_xlabel("Importance (mean decrease in impurity)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[part1] Saved {save_path}")


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
# 6.  main()  –  load → split → train both → compare → save winner → figures
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

    # ── Random Forest ───────────────────────────────────────────────────────
    print("[part1] Training Random Forest ...")
    t0 = time.time()
    rf_preprocessor = build_preprocessing_pipeline()
    rf_pipeline = Pipeline([
        ("pre", rf_preprocessor),
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
        winner_name     = "XGBoost"
        winner_pipeline = xgb_pipeline
        winner_metrics  = xgb_metrics
    else:
        winner_name     = "RandomForest"
        winner_pipeline = rf_pipeline
        winner_metrics  = rf_metrics

    print(f"[part1] Winner: {winner_name}")

    # ── Save bundle (winner only — used by the API) ─────────────────────────
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

    # ── Confusion matrix summary (console) ───────────────────────────────────
    print("\n[part1] Confusion Matrix (rows=actual, cols=predicted) — winner:")
    print(f"        {'Low':>6}  {'Medium':>6}  {'High':>6}")
    for i, row_label in enumerate(LABEL_ORDER):
        row_vals = winner_metrics["confusion_matrix"][i]
        print(f"  {row_label:<7}  {row_vals[0]:>5}   {row_vals[1]:>5}   {row_vals[2]:>5}")

    # ── Figures: confusion matrix + feature importance, for BOTH models ─────
    print("\n[part1] Generating figures ...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Feature names after one-hot encoding, taken from each fitted preprocessor
    rf_feature_names  = list(rf_pipeline.named_steps["pre"].get_feature_names_out())
    xgb_feature_names = list(xgb_pipeline.named_steps["pre"].get_feature_names_out())

    plot_confusion_matrix(
        rf_metrics["confusion_matrix"], LABEL_ORDER,
        title="Random Forest — Confusion Matrix",
        save_path=FIGURES_DIR / "confusion_matrix_rf.png",
    )
    plot_confusion_matrix(
        xgb_metrics["confusion_matrix"], LABEL_ORDER,
        title="XGBoost — Confusion Matrix",
        save_path=FIGURES_DIR / "confusion_matrix_xgb.png",
    )
    plot_feature_importance(
        rf_pipeline.named_steps["clf"], rf_feature_names,
        title="Random Forest — Top 15 Feature Importances",
        save_path=FIGURES_DIR / "feature_importance_rf.png",
    )
    plot_feature_importance(
        xgb_pipeline.named_steps["clf"], xgb_feature_names,
        title="XGBoost — Top 15 Feature Importances",
        save_path=FIGURES_DIR / "feature_importance_xgb.png",
    )

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
