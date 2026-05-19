"""
Clean the merged dataset and produce a model-ready CSV.

Pipeline:
  1. Load data/processed/merged.csv (output of src/data/merge.py).
  2. Drop the helper `source` column (kept for diagnostics, not for modelling).
  3. Replace clinically impossible sentinels (RestingBP == 0, Cholesterol == 0)
     with NaN. The Kaggle dataset uses 0 to mean "not measured" for these two
     fields, which would otherwise distort distributions and feature scaling.
  4. Drop rows whose key vitals are still missing after sentinel-replacement
     (Age, Sex, RestingBP, Cholesterol, MaxHR).
  5. Impute the remaining missingness:
       - Numeric columns      -> median
       - Categorical columns  -> mode
  6. Coerce final dtypes.
  7. Write data/processed/cleaned.csv.

Run:
    python -m src.data.clean

Output:
    data/processed/cleaned.csv   (12 unified columns, no NaNs, no implausible zeros)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
INPUT_FILE    = PROCESSED_DIR / "merged.csv"
OUTPUT_FILE   = PROCESSED_DIR / "cleaned.csv"

# Columns whose 0 is clinically impossible and should be treated as missing
ZERO_AS_MISSING = ["RestingBP", "Cholesterol"]

# Columns whose missingness is non-recoverable; rows missing any of these are dropped
REQUIRED_VITALS = ["Age", "Sex", "RestingBP", "Cholesterol", "MaxHR"]

NUMERIC_COLS = ["Age", "RestingBP", "Cholesterol", "FastingBS",
                "MaxHR", "Oldpeak"]
CATEGORICAL_COLS = ["Sex", "ChestPainType", "RestingECG",
                    "ExerciseAngina", "ST_Slope"]
TARGET_COL = "HeartDisease"


# ---------------------------------------------------------------------------
# Cleaning steps
# ---------------------------------------------------------------------------

def replace_impossible_zeros(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ZERO_AS_MISSING:
        n_zero = (df[col] == 0).sum()
        if n_zero:
            print(f"[clean] {col}: replacing {n_zero} zeros with NaN")
        df[col] = df[col].replace(0, np.nan)
    return df


def drop_unrecoverable_rows(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=REQUIRED_VITALS).reset_index(drop=True)
    print(f"[clean] Dropped {before - len(df)} rows missing required vitals "
          f"({REQUIRED_VITALS})")
    return df


def impute_remaining(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_COLS:
        if df[col].isna().any():
            median = df[col].median()
            n = df[col].isna().sum()
            df[col] = df[col].fillna(median)
            print(f"[clean] {col}: imputed {n} NaN with median={median}")
    for col in CATEGORICAL_COLS:
        if df[col].isna().any():
            mode = df[col].mode().iloc[0]
            n = df[col].isna().sum()
            df[col] = df[col].fillna(mode)
            print(f"[clean] {col}: imputed {n} NaN with mode='{mode}'")
    return df


def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Age"]          = df["Age"].astype(int)
    df["RestingBP"]    = df["RestingBP"].astype(int)
    df["Cholesterol"]  = df["Cholesterol"].astype(int)
    df["FastingBS"]    = df["FastingBS"].astype(int)
    df["MaxHR"]        = df["MaxHR"].astype(int)
    df["Oldpeak"]      = df["Oldpeak"].astype(float)
    df["HeartDisease"] = df["HeartDisease"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full cleaning pipeline to a merged DataFrame in memory."""
    # Drop diagnostic source column if present
    if "source" in df.columns:
        df = df.drop(columns=["source"])

    df = replace_impossible_zeros(df)
    df = drop_unrecoverable_rows(df)
    df = impute_remaining(df)
    df = coerce_dtypes(df)
    return df


def main() -> pd.DataFrame:
    print(f"[clean] Loading {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"[clean] Input shape:  {df.shape}")

    cleaned = clean_dataframe(df)

    print(f"[clean] Output shape: {cleaned.shape}")
    print(f"[clean] Any remaining NaNs? "
          f"{int(cleaned.isna().sum().sum())} (should be 0)")
    print(f"[clean] HeartDisease balance:\n"
          f"{cleaned[TARGET_COL].value_counts().to_string()}")

    cleaned.to_csv(OUTPUT_FILE, index=False)
    print(f"[clean] Wrote {OUTPUT_FILE}")
    return cleaned


if __name__ == "__main__":
    main()
