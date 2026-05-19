# CardioSurv — API Contract

> **Status:** Day-1 Contract — locked. Any breaking change must bump the API version.
> **Owner:** M4 · **Consumers:** M5 (frontend), M2/M3 (model integration)

Base URL (dev):  `http://localhost:8000`
Base URL (prod, mock):  `https://cardiosurv-api.onrender.com`
Base URL (prod, real):  `https://cardiosurv-api.onrender.com` (same; flipped on Week-3 deploy)

All endpoints return JSON. All errors follow the standard error shape in §6.

---

## 0. Conventions

- **Content-Type:** `application/json` on every request with a body.
- **CORS:** `Access-Control-Allow-Origin` is set to the frontend Render Static Site URL plus `http://localhost:5500`/`http://127.0.0.1:5500` for local dev.
- **Authentication:** none for the project demo (single-team prototype). Rate limit only.
- **Timestamps:** RFC 3339 UTC (e.g. `2026-06-20T14:21:09Z`).
- **IDs:** UUID v4 strings.
- **API versioning:** path-prefixed (`/api/v1/...`). v1 is the only supported version.

---

## 1. `GET /api/v1/health`

Health probe. No auth, no rate limit.

**Response 200**
```json
{
  "status": "ok",
  "version": "v1",
  "model_versions": {
    "part1_classifier": "v1.0",
    "part2_recommender": "v1.0",
    "survival_cox":     "v1.0"
  },
  "uptime_seconds": 12345
}
```

---

## 2. `POST /api/v1/predict`

Submits a patient's vital signs and returns the Part-1 risk classification.
Internally: validates input → inserts a `patients` row → calls `part1_predict(...)` → inserts a `predictions` row.

**Request body**
```json
{
  "age":             56,
  "sex":             "M",
  "chest_pain_type": "ATA",
  "resting_bp":      138,
  "cholesterol":     230,
  "fasting_bs":      0,
  "resting_ecg":     "Normal",
  "max_hr":          150,
  "exercise_angina": "N",
  "oldpeak":         1.2,
  "st_slope":        "Up"
}
```

**Field validation (Pydantic)**

| Field             | Type    | Constraint                              |
|-------------------|---------|------------------------------------------|
| age               | int     | `1 <= age <= 120`                        |
| sex               | str     | `"M"` or `"F"`                            |
| chest_pain_type   | str     | one of `TA`, `ATA`, `NAP`, `ASY`         |
| resting_bp        | int     | `50 <= resting_bp <= 250`                 |
| cholesterol       | int     | `50 <= cholesterol <= 800`                |
| fasting_bs        | int     | `0` or `1`                                |
| resting_ecg       | str     | one of `Normal`, `ST`, `LVH`              |
| max_hr            | int     | `40 <= max_hr <= 230`                     |
| exercise_angina   | str     | `"N"` or `"Y"`                            |
| oldpeak           | float   | `-3.0 <= oldpeak <= 7.0`                  |
| st_slope          | str     | one of `Up`, `Flat`, `Down`               |

**Response 200**
```json
{
  "patient_id":    "9c1b7e64-3a4f-4d3a-a4c8-2e5e7b71ed12",
  "prediction_id": "f0e9d8c7-b6a5-4948-83a1-72635849affe",
  "risk_category": "Medium",
  "confidence":    0.83,
  "probabilities": {"Low": 0.06, "Medium": 0.83, "High": 0.11},
  "model_version": "part1_classifier_v1.0",
  "created_at":    "2026-06-20T14:21:09Z"
}
```

**Errors:** 422 on field validation, 500 on internal error.

---

## 3. `POST /api/v1/recommend`

Takes a `prediction_id` (from `/predict`) and returns the Part-2 treatment recommendation plus the counterfactual survival pair.
Internally: looks up the prediction → calls `part2_recommend(...)` → inserts a `recommendations` row.

**Request body**
```json
{
  "prediction_id": "f0e9d8c7-b6a5-4948-83a1-72635849affe",
  "has_arrhythmia": false
}
```

`has_arrhythmia` is a clinician-entered flag (boolean). It is the **decision-node** input that, combined with `risk_category=="High"`, triggers the SBRT branch.

**Response 200 — SBRT branch**
```json
{
  "recommendation_id":   "11111111-2222-3333-4444-555555555555",
  "patient_id":          "9c1b7e64-3a4f-4d3a-a4c8-2e5e7b71ed12",
  "prediction_id":       "f0e9d8c7-b6a5-4948-83a1-72635849affe",
  "branch":              "SBRT",
  "intervention_type":   "cardiac_sbrt_25Gy_1fx",
  "intensity":           "High",
  "bed_gy":              87.5,
  "bed_valid":           true,
  "grace_score":         null,
  "grace_risk_category": null,
  "survival_without":    0.58,
  "survival_with":       0.81,
  "model_version":       "part2_recommender_v1.0",
  "created_at":          "2026-06-20T14:21:11Z"
}
```

**Response 200 — Medication branch**
```json
{
  "recommendation_id":   "11111111-2222-3333-4444-555555555555",
  "patient_id":          "9c1b7e64-3a4f-4d3a-a4c8-2e5e7b71ed12",
  "prediction_id":       "f0e9d8c7-b6a5-4948-83a1-72635849affe",
  "branch":              "Medication",
  "intervention_type":   "beta_blocker+moderate_statin+aspirin",
  "intensity":           "Moderate",
  "bed_gy":              null,
  "bed_valid":           null,
  "grace_score":         132,
  "grace_risk_category": "Intermediate",
  "survival_without":    0.81,
  "survival_with":       0.92,
  "model_version":       "part2_recommender_v1.0",
  "created_at":          "2026-06-20T14:21:11Z"
}
```

**Errors:** 404 if `prediction_id` not found, 422 on validation, 500 on internal error.

---

## 4. `GET /api/v1/history`

Returns the most recent predictions+recommendations across all patients. Paginated.

**Query parameters**

| Param  | Type | Default | Notes                          |
|--------|------|---------|---------------------------------|
| page   | int  | 1       | 1-indexed                       |
| size   | int  | 20      | max 100                         |

**Response 200**
```json
{
  "page": 1,
  "size": 20,
  "total": 47,
  "items": [
    {
      "prediction_id":         "f0e9...",
      "patient_id":            "9c1b...",
      "created_at":            "2026-06-20T14:21:09Z",
      "age":                   56,
      "risk_category":         "Medium",
      "confidence":            0.83,
      "branch":                "Medication",
      "intervention_type":     "beta_blocker+moderate_statin+aspirin",
      "survival_without":      0.81,
      "survival_with":         0.92,
      "survival_improvement":  0.11
    }
  ]
}
```

`survival_improvement` is computed server-side as `survival_with - survival_without`.
If a record has no recommendation yet, recommendation-related fields are `null`.

---

## 5. `GET /api/v1/patients/{patient_id}`

Returns the full record for one patient: vitals + every prediction + every recommendation.

**Response 200**
```json
{
  "patient_id": "9c1b7e64-3a4f-4d3a-a4c8-2e5e7b71ed12",
  "created_at": "2026-06-20T14:20:55Z",
  "vitals": {
    "age": 56, "sex": "M", "chest_pain_type": "ATA",
    "resting_bp": 138, "cholesterol": 230, "fasting_bs": 0,
    "resting_ecg": "Normal", "max_hr": 150, "exercise_angina": "N",
    "oldpeak": 1.2, "st_slope": "Up"
  },
  "predictions": [
    {
      "prediction_id": "f0e9...", "risk_category": "Medium",
      "confidence": 0.83,
      "probabilities": {"Low": 0.06, "Medium": 0.83, "High": 0.11},
      "model_version": "part1_classifier_v1.0",
      "created_at": "2026-06-20T14:21:09Z"
    }
  ],
  "recommendations": [
    {
      "recommendation_id": "1111...", "branch": "Medication",
      "intervention_type": "beta_blocker+moderate_statin+aspirin",
      "intensity": "Moderate",
      "bed_gy": null, "bed_valid": null,
      "grace_score": 132, "grace_risk_category": "Intermediate",
      "survival_without": 0.81, "survival_with": 0.92,
      "model_version": "part2_recommender_v1.0",
      "created_at": "2026-06-20T14:21:11Z"
    }
  ]
}
```

**Errors:** 404 if `patient_id` not found.

---

## 6. Error shape (uniform)

Every non-2xx response follows this shape:

```json
{
  "error": {
    "code":    "VALIDATION_ERROR",
    "message": "Field 'age' must be between 1 and 120",
    "details": [
      {"field": "age", "reason": "out of range", "got": -3}
    ]
  }
}
```

Standard codes:

| HTTP | Code               | When                                       |
|------|--------------------|--------------------------------------------|
| 422  | `VALIDATION_ERROR` | Pydantic field validation failed           |
| 404  | `NOT_FOUND`        | Resource by ID does not exist              |
| 429  | `RATE_LIMITED`     | slowapi limit hit on `/predict`            |
| 500  | `INTERNAL_ERROR`   | unhandled exception (logged server-side)   |
| 503  | `MODEL_UNAVAILABLE`| model failed to load at startup            |

---

## 7. Rate limits

| Endpoint          | Limit              |
|-------------------|--------------------|
| `POST /predict`   | 30 / minute / IP   |
| `POST /recommend` | 30 / minute / IP   |
| `GET  /history`   | 120 / minute / IP  |
| `GET  /patients/*`| 120 / minute / IP  |
| `GET  /health`    | unlimited          |

Implemented via `slowapi`.

---

## 8. Auto-generated docs

FastAPI exposes:
- Swagger UI: `/docs`
- ReDoc:      `/redoc`
- OpenAPI JSON: `/openapi.json`

These are kept in sync with the Pydantic schemas in `src/api/schemas.py`. The
hand-written contract in this document is the **source of truth** for any cross-team
discussion; the auto-generated docs are derived from code and may lag by minutes.

---

## Version

| Version | Date    | Author | Change |
|---------|---------|--------|--------|
| 1.0     | Week 1  | M4     | Initial Day-1 contract |
