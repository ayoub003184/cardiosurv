// CardioSurv — API Client
// Auto-detects local vs production environment
const API_BASE = (
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1"
) ? "http://localhost:8000" : "https://cardiosurv-api.onrender.com";

/**
 * Internal helper — fetch + parse JSON, throw on error with status code.
 */
async function _request(method, path, body) {
  const opts = {
    method,
    headers: { "Content-Type": "application/json" },
  };
  if (body !== undefined) opts.body = JSON.stringify(body);

  const res = await fetch(API_BASE + path, opts);
  let data;
  try { data = await res.json(); } catch { data = null; }

  if (!res.ok) {
    const err = new Error(
      data?.error?.message || `HTTP ${res.status}`
    );
    err.status  = res.status;
    err.code    = data?.error?.code || "UNKNOWN";
    err.details = data?.error?.details || [];
    throw err;
  }
  return data;
}

/** GET /api/v1/health */
async function getHealth() {
  return _request("GET", "/api/v1/health");
}

/** POST /api/v1/predict — body matches PredictRequest schema */
async function submitPredict(body) {
  return _request("POST", "/api/v1/predict", body);
}

/** POST /api/v1/recommend — body: { prediction_id, has_arrhythmia } */
async function submitRecommend(body) {
  return _request("POST", "/api/v1/recommend", body);
}

/** GET /api/v1/history?page=N&size=20 */
async function getHistory(page = 1, size = 20) {
  return _request("GET", `/api/v1/history?page=${page}&size=${size}`);
}

/** GET /api/v1/patients/{id} */
async function getPatient(id) {
  return _request("GET", `/api/v1/patients/${id}`);
}

// ── CLIENT-SIDE VALIDATION ──────────────────────────────────────────────────
function validatePredictForm(data) {
  const errors = {};

  const req = (k, label) => {
    if (data[k] === undefined || data[k] === null || data[k] === "") {
      errors[k] = `${label} is required.`;
    }
  };

  req("age",             "Age");
  req("sex",             "Sex");
  req("chest_pain_type", "Chest Pain Type");
  req("resting_bp",      "Resting BP");
  req("cholesterol",     "Cholesterol");
  req("fasting_bs",      "Fasting Blood Sugar");
  req("resting_ecg",     "Resting ECG");
  req("max_hr",          "Max Heart Rate");
  req("exercise_angina", "Exercise Angina");
  req("oldpeak",         "Oldpeak");
  req("st_slope",        "ST Slope");

  if (!errors.age) {
    const v = Number(data.age);
    if (!Number.isInteger(v) || v < 1 || v > 120)
      errors.age = "Age must be an integer between 1 and 120.";
  }
  if (!errors.resting_bp) {
    const v = Number(data.resting_bp);
    if (!Number.isInteger(v) || v < 50 || v > 250)
      errors.resting_bp = "Resting BP must be between 50 and 250.";
  }
  if (!errors.cholesterol) {
    const v = Number(data.cholesterol);
    if (!Number.isInteger(v) || v < 50 || v > 800)
      errors.cholesterol = "Cholesterol must be between 50 and 800.";
  }
  if (!errors.max_hr) {
    const v = Number(data.max_hr);
    if (!Number.isInteger(v) || v < 40 || v > 230)
      errors.max_hr = "Max HR must be between 40 and 230.";
  }
  if (!errors.oldpeak) {
    const v = Number(data.oldpeak);
    if (isNaN(v) || v < -3.0 || v > 7.0)
      errors.oldpeak = "Oldpeak must be between -3.0 and 7.0.";
  }
  if (!errors.fasting_bs) {
    const v = Number(data.fasting_bs);
    if (v !== 0 && v !== 1)
      errors.fasting_bs = "Fasting BS must be 0 or 1.";
  }
  if (!errors.sex && !["M","F"].includes(data.sex))
    errors.sex = "Sex must be M or F.";
  if (!errors.chest_pain_type && !["TA","ATA","NAP","ASY"].includes(data.chest_pain_type))
    errors.chest_pain_type = "Invalid chest pain type.";
  if (!errors.resting_ecg && !["Normal","ST","LVH"].includes(data.resting_ecg))
    errors.resting_ecg = "Invalid ECG value.";
  if (!errors.exercise_angina && !["N","Y"].includes(data.exercise_angina))
    errors.exercise_angina = "Exercise angina must be N or Y.";
  if (!errors.st_slope && !["Up","Flat","Down"].includes(data.st_slope))
    errors.st_slope = "Invalid ST slope.";

  return { valid: Object.keys(errors).length === 0, errors };
}