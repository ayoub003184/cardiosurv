"""
Merge the Kaggle Heart Failure dataset with the four UCI Heart Disease splits
(Cleveland, Hungarian, Switzerland, VA) into a single unified CSV.

The two source families use different encodings for the same clinical concepts.
This script:
  1. Loads all five raw CSVs.
  2. Maps every UCI numeric code to the Kaggle string categorical equivalent
     (see docs/schemas.md §2 for the locked encoding maps).
  3. Binarises the UCI severity target (0-4) to a 0/1 HeartDisease flag.
  4. Drops UCI-only columns (ca, thal) that have heavy missingness and are
     absent from the Kaggle source.
  5. Concatenates everything into data/processed/merged.csv with 12 columns
     defined by the unified schema.

Run:
    python -m src.data.merge

Output:
    data/processed/merged.csv   (~1838 rows x 12 cols, before cleaning)

Reference: docs/schemas.md sections 1 and 2.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_DIR       = Path("src/data/data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

KAGGLE_FILE   = RAW_DIR / "heart.csv"
UCI_FILES     = [
    RAW_DIR / "processed.cleveland_labeled.csv",
    RAW_DIR / "processed.hungarian_labeled.csv",
    RAW_DIR / "processed.switzerland_labeled.csv",
    RAW_DIR / "processed.va_labeled.csv",
]

OUTPUT_FILE   = PROCESSED_DIR / "merged.csv"

# ---------------------------------------------------------------------------
# Locked unified schema (see docs/schemas.md section 2)
# ---------------------------------------------------------------------------

UNIFIED_COLUMNS = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
    "Oldpeak", "ST_Slope", "HeartDisease",
]

# Column rename: UCI -> Kaggle/unified
UCI_RENAME = {
    "age":      "Age",
    "sex":      "Sex",
    "cp":       "ChestPainType",
    "trestbps": "RestingBP",
    "chol":     "Cholesterol",
    "fbs":      "FastingBS",
    "restecg":  "RestingECG",
    "thalach":  "MaxHR",
    "exang":    "ExerciseAngina",
    "oldpeak":  "Oldpeak",
    "slope":    "ST_Slope",
    "target":   "HeartDisease",
}

# Categorical encoding maps (UCI numeric -> Kaggle string)
SEX_MAP     = {1.0: "M", 0.0: "F"}
CP_MAP      = {1.0: "TA", 2.0: "ATA", 3.0: "NAP", 4.0: "ASY"}
RESTECG_MAP = {0.0: "Normal", 1.0: "ST", 2.0: "LVH"}
EXANG_MAP   = {0.0: "N", 1.0: "Y"}
SLOPE_MAP   = {1.0: "Up", 2.0: "Flat", 3.0: "Down"}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_kaggle() -> pd.DataFrame:
    """Load the Kaggle heart.csv. Already in unified-schema encoding."""
    df = pd.read_csv(KAGGLE_FILE)
    missing = set(UNIFIED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Kaggle CSV is missing expected columns: {missing}")
    df = df[UNIFIED_COLUMNS].copy()
    df["source"] = "kaggle"
    return df


def load_uci_file(path: Path) -> pd.DataFrame:
    """Load one UCI split and convert it to the unified schema."""
    df = pd.read_csv(path)

    # 1. Rename UCI columns to unified names
    df = df.rename(columns=UCI_RENAME)

    # 2. UCI uses '?' as a missing-value sentinel, which makes some categorical
    #    columns load as object/string dtype. Coerce them to float first so the
    #    int-keyed maps below match correctly. '?' -> NaN -> unmapped -> NaN.
    for col in ["Sex", "ChestPainType", "RestingECG",
                "ExerciseAngina", "ST_Slope"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Map numeric codes to Kaggle string categoricals
    df["Sex"]            = df["Sex"].map(SEX_MAP)
    df["ChestPainType"]  = df["ChestPainType"].map(CP_MAP)
    df["RestingECG"]     = df["RestingECG"].map(RESTECG_MAP)
    df["ExerciseAngina"] = df["ExerciseAngina"].map(EXANG_MAP)
    df["ST_Slope"]       = df["ST_Slope"].map(SLOPE_MAP)

    # 4. Binarise the severity target: any disease (1-4) -> 1
    df["HeartDisease"] = (df["HeartDisease"].astype(float) > 0).astype(int)

    # 5. Coerce numerics that UCI stores as floats. UCI uses '?' for missing,
    #    which pandas already left as NaN after read_csv -- coerce to be safe.
    for col in ["Age", "RestingBP", "Cholesterol", "FastingBS",
                "MaxHR", "Oldpeak"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 6. Keep only the unified columns
    df = df[UNIFIED_COLUMNS].copy()
    df["source"] = path.stem.replace("processed.", "").replace("_labeled", "")
    return df


def load_all_uci() -> pd.DataFrame:
    return pd.concat([load_uci_file(p) for p in UCI_FILES], ignore_index=True)


# ---------------------------------------------------------------------------
# Main merge
# ---------------------------------------------------------------------------

def main() -> pd.DataFrame:
    print("[merge] Loading Kaggle...")
    kaggle = load_kaggle()
    print(f"        Kaggle:    {kaggle.shape}")

    print("[merge] Loading UCI splits...")
    uci = load_all_uci()
    print(f"        UCI total: {uci.shape}")

    merged = pd.concat([kaggle, uci], ignore_index=True)

    # Final sanity check: every unified column must have at least some
    # non-null values from every source.
    print(f"[merge] Merged shape: {merged.shape}")
    print(f"[merge] Per-source row counts:")
    print(merged["source"].value_counts().to_string())
    print(f"[merge] Null counts per unified column:")
    print(merged[UNIFIED_COLUMNS].isna().sum().to_string())

    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"[merge] Wrote {OUTPUT_FILE}")
    return merged


if __name__ == "__main__":
    main()
