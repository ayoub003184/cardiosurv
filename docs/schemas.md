# CardioSurv — Data & Output Schemas

> **Status:** Day-1 Contract — locked. Any change must be agreed in the weekly sync and bumped via the version footer.
> **Owners:** M1 (data) · M2 (Part 1 outputs) · M3 (Part 2 outputs) · M4 (API)

This document is the contract that lets all 6 team members work in parallel.
Every column name, type, encoding, and JSON field below is **frozen** once signed off.

---

## 1. Raw datasets

### 1.1 Kaggle Heart Failure Prediction (`heart.csv`)

| Column         | Type        | Values / Range                                  |
|----------------|-------------|-------------------------------------------------|
| Age            | int         | 28–77                                            |
| Sex            | categorical | `M`, `F`                                         |
| ChestPainType  | categorical | `TA`, `ATA`, `NAP`, `ASY`                        |
| RestingBP      | int         | 0–200 mmHg (note: 0 is invalid → drop in clean) |
| Cholesterol    | int         | 0–603 mg/dL (note: 0 is invalid → drop in clean)|
| FastingBS      | int         | `0` (<120 mg/dL), `1` (≥120 mg/dL)               |
| RestingECG     | categorical | `Normal`, `ST`, `LVH`                            |
| MaxHR          | int         | 60–202 bpm                                       |
| ExerciseAngina | categorical | `N`, `Y`                                         |
| Oldpeak        | float       | -2.6 to 6.2 (ST depression)                      |
| ST_Slope       | categorical | `Up`, `Flat`, `Down`                             |
| HeartDisease   | int         | `0` (no disease), `1` (disease)                  |

Rows: **918**.

### 1.2 UCI Cleveland / Hungarian / Switzerland / VA (4 files)

| Column   | Type  | Native encoding                          |
|----------|-------|------------------------------------------|
| age      | float | years                                    |
| sex      | float | `1.0` = M, `0.0` = F                     |
| cp       | float | `1` typical, `2` atypical, `3` non-anginal, `4` asymptomatic |
| trestbps | float | resting BP mmHg                          |
| chol     | float | serum cholesterol mg/dL                  |
| fbs      | float | `0`/`1` fasting BS > 120 mg/dL            |
| restecg  | float | `0` normal, `1` ST-T abnormality, `2` LVH |
| thalach  | float | max HR achieved                          |
| exang    | float | `0`/`1` exercise-induced angina           |
| oldpeak  | float | ST depression                            |
| slope    | float | `1` upsloping, `2` flat, `3` downsloping  |
| ca       | str   | `0`–`3` major vessels; `?` = missing      |
| thal     | str   | `3` normal, `6` fixed defect, `7` reversible; `?` = missing |
| target   | int   | `0` (no disease), `1`–`4` (severity)      |

Rows: Cleveland 303 · Hungarian 294 · Switzerland 123 · VA 200 = **920**.

> **Important:** `ca` and `thal` are dropped from the unified schema. They are absent
> from the Kaggle file (which is the larger source) and their UCI coverage is patchy
> (heavy `?` in Hungarian/Switzerland/VA). They are *not* part of the Part 1 feature set.

---

## 2. Unified schema (merged.csv)

After harmonisation in `src/data/merge.py`, both datasets share these 12 columns:

| Column         | Type        | Encoding (post-merge)                          |
|----------------|-------------|------------------------------------------------|
| Age            | int         | years                                          |
| Sex            | categorical | `M`, `F`                                       |
| ChestPainType  | categorical | `TA`, `ATA`, `NAP`, `ASY`                      |
| RestingBP      | int         | mmHg                                           |
| Cholesterol    | int         | mg/dL                                          |
| FastingBS      | int         | `0`, `1`                                       |
| RestingECG     | categorical | `Normal`, `ST`, `LVH`                          |
| MaxHR          | int         | bpm                                            |
| ExerciseAngina | categorical | `N`, `Y`                                       |
| Oldpeak        | float       | ST depression                                  |
| ST_Slope       | categorical | `Up`, `Flat`, `Down`                           |
| HeartDisease   | int         | `0` no disease · `1` any disease (UCI 1–4 collapsed to 1) |

### Encoding maps (UCI → unified)

```python
SEX_MAP        = {1.0: "M", 0.0: "F"}
CP_MAP         = {1.0: "TA", 2.0: "ATA", 3.0: "NAP", 4.0: "ASY"}
RESTECG_MAP    = {0.0: "Normal", 1.0: "ST", 2.0: "LVH"}
EXANG_MAP      = {0.0: "N", 1.0: "Y"}
SLOPE_MAP      = {1.0: "Up", 2.0: "Flat", 3.0: "Down"}
# target: any value > 0 → 1 (binary HeartDisease)
```

---

## 3. Engineered features (features.csv)

Produced by `src/features/engineering.py` on top of merged.csv.

| Feature                | Type        | Formula / Bins                                                   |
|------------------------|-------------|------------------------------------------------------------------|
| AgeBin                 | categorical | `<40`, `40-49`, `50-59`, `60-69`, `70+`                          |
| BP_RiskLevel           | categorical | `Normal` (<120) · `Elevated` (120–129) · `Stage1` (130–139) · `Stage2` (140–179) · `Crisis` (≥180), per AHA guidelines |
| HeartRateStressIndex   | float       | `MaxHR / (220 - Age)` — fraction of age-predicted max HR reached |

All original 12 unified columns are retained; the 3 engineered columns are appended.

---

## 4. Part 1 classifier output schema (locked)

```json
{
  "patient_id":     "string (UUID v4)",
  "risk_category":  "Low | Medium | High",
  "confidence":     0.0,
  "model_version":  "part1_classifier_v1.0",
  "probabilities":  {"Low": 0.0, "Medium": 0.0, "High": 0.0}
}
```

`confidence` = max(probabilities). M3's `routing.py` already consumes this exact shape.

### Risk-category derivation rule

Part 1 trains a 3-class classifier. The label is derived from clinical features
during preprocessing (not from `HeartDisease` directly), using this rule agreed on Day 1:

```
HeartDisease = 0                                                    ->  Low
HeartDisease = 1  AND  none of [Oldpeak >= 2, ExerciseAngina = Y,
                                BP_RiskLevel in {Stage2, Crisis}]   ->  Medium
HeartDisease = 1  AND  any of the above                             ->  High
```

---

## 5. Part 2 recommender output schema (locked)

```json
{
  "patient_id":           "string",
  "branch":               "SBRT | Medication",
  "intervention_type":    "string (e.g. 'cardiac_sbrt_25Gy_1fx', 'high_intensity_statin+beta_blocker')",
  "intensity":            "Low | Moderate | High",
  "bed_gy":               0.0,
  "bed_valid":            true,
  "grace_score":          0,
  "grace_risk_category":  "Low | Intermediate | High",
  "survival_without":     0.0,
  "survival_with":        0.0,
  "model_version":        "part2_recommender_v1.0"
}
```

Notes:
- `bed_gy` and `bed_valid` are `null` for the Medication branch.
- `grace_score` and `grace_risk_category` are `null` for the SBRT branch.
- `survival_without` / `survival_with` are 2-year probabilities (0–1).

---

## 6. Database tables (Postgres, owned by M1)

### patients
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | server-generated |
| created_at | TIMESTAMPTZ | default now() |
| age | INT | |
| sex | VARCHAR(1) | `M`/`F` |
| chest_pain_type | VARCHAR(3) | |
| resting_bp | INT | |
| cholesterol | INT | |
| fasting_bs | SMALLINT | |
| resting_ecg | VARCHAR(8) | |
| max_hr | INT | |
| exercise_angina | VARCHAR(1) | |
| oldpeak | NUMERIC(4,2) | |
| st_slope | VARCHAR(5) | |

### predictions
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| patient_id | UUID FK -> patients.id | |
| risk_category | VARCHAR(8) | |
| confidence | NUMERIC(4,3) | |
| probabilities | JSONB | |
| model_version | VARCHAR(64) | |
| created_at | TIMESTAMPTZ | |

### recommendations
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| prediction_id | UUID FK -> predictions.id | |
| branch | VARCHAR(16) | |
| intervention_type | VARCHAR(128) | |
| intensity | VARCHAR(8) | |
| bed_gy | NUMERIC(6,2) | NULLable |
| bed_valid | BOOLEAN | NULLable |
| grace_score | INT | NULLable |
| grace_risk_category | VARCHAR(16) | NULLable |
| survival_without | NUMERIC(4,3) | |
| survival_with | NUMERIC(4,3) | |
| model_version | VARCHAR(64) | |
| created_at | TIMESTAMPTZ | |

### audit_logs
| Column | Type | Notes |
|---|---|---|
| id | BIGSERIAL PK | |
| route | VARCHAR(64) | |
| status_code | INT | |
| request_ip | INET | |
| latency_ms | INT | |
| created_at | TIMESTAMPTZ | |

---

## Version

| Version | Date       | Author | Change                |
|---------|------------|--------|-----------------------|
| 1.0     | Week 1     | M1     | Initial Day-1 contract |
