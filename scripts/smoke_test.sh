#!/usr/bin/env bash
# ============================================================================
# CardioSurv API — Smoke Test
# ============================================================================
# Hits all 5 endpoints against a given base URL and asserts they respond.
# Use this after deploying to Render to confirm the live service works.
#
#   bash scripts/smoke_test.sh                                     # default localhost
#   bash scripts/smoke_test.sh https://cardiosurv-api.onrender.com # production
# ============================================================================

set -e

BASE="${1:-http://localhost:8000}"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo "Smoke-testing CardioSurv API at: ${BASE}"
echo "============================================================"

# Helper — runs curl, checks HTTP status, pretty-prints
hit() {
    local method=$1 path=$2 expected=$3 data=$4 desc=$5
    echo
    echo -e "${YELLOW}[$desc]${NC} $method $path"
    if [ -n "$data" ]; then
        resp=$(curl -sS -o /tmp/cardiosurv_smoke.json -w "%{http_code}" \
                    -X "$method" "${BASE}${path}" \
                    -H "Content-Type: application/json" \
                    -d "$data")
    else
        resp=$(curl -sS -o /tmp/cardiosurv_smoke.json -w "%{http_code}" \
                    -X "$method" "${BASE}${path}")
    fi

    if [ "$resp" = "$expected" ]; then
        echo -e "  ${GREEN}✓ HTTP $resp${NC}"
        head -c 400 /tmp/cardiosurv_smoke.json | python3 -m json.tool 2>/dev/null \
            || head -c 400 /tmp/cardiosurv_smoke.json
        echo
    else
        echo -e "  ${RED}✗ Expected $expected, got $resp${NC}"
        cat /tmp/cardiosurv_smoke.json
        exit 1
    fi
}

# 1. Health
hit GET "/api/v1/health" 200 "" "Health check"

# 2. Predict — Case 1 from Phase 4 (high-risk male, arrhythmia)
PREDICT_BODY='{"age":67,"sex":"M","chest_pain_type":"ASY","resting_bp":162,"cholesterol":268,"fasting_bs":1,"resting_ecg":"ST","max_hr":100,"exercise_angina":"Y","oldpeak":2.5,"st_slope":"Flat"}'
hit POST "/api/v1/predict" 200 "$PREDICT_BODY" "Predict (high-risk patient)"

# Pull the prediction_id out for the recommend call
PRED_ID=$(python3 -c "import json; print(json.load(open('/tmp/cardiosurv_smoke.json'))['prediction_id'])")
echo "   captured prediction_id=${PRED_ID:0:8}..."

# 3. Recommend — uses the prediction_id we just captured
REC_BODY="{\"prediction_id\":\"${PRED_ID}\",\"has_arrhythmia\":true}"
hit POST "/api/v1/recommend" 200 "$REC_BODY" "Recommend (SBRT branch)"

# 4. History
hit GET "/api/v1/history?page=1&limit=5" 200 "" "History list"

# 5. Patient detail
PATIENT_ID=$(python3 -c "import json; print(json.load(open('/tmp/cardiosurv_smoke.json'))['items'][0]['patient_id'])")
hit GET "/api/v1/patients/${PATIENT_ID}" 200 "" "Patient detail"

echo
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}✓ All 5 endpoints responded correctly${NC}"
echo -e "${GREEN}============================================================${NC}"
