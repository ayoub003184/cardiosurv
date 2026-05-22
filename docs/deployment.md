# CardioSurv API — Deployment Runbook

> **Audience:** anyone on the team who needs to deploy, debug, or demo the live API.
> **Last updated:** 22 May 2026 — alongside the T-D Part B work.

---

## 1. What gets deployed

| Resource           | Type                  | Plan        | Region    |
|--------------------|-----------------------|-------------|-----------|
| `cardiosurv-api`   | Render Web Service    | Free (Python 3.11) | Singapore |
| `cardiosurv-db`    | Render PostgreSQL     | Free (90-day) | Singapore |
| `cardiosurv-frontend` | Render Static Site | Free        | Singapore |

The repo contains `render.yaml` (Render Blueprint) — Render reads it on
first apply and provisions everything in one shot. You can also do it
manually via the dashboard if Blueprints are not available on the free plan.

---

## 2. Deploy the API (one-time setup)

### Option A — Blueprint (preferred, single click)

1. Go to **[Render Dashboard](https://dashboard.render.com)** → **New +** → **Blueprint**.
2. Connect the `cardiosurv-main` GitHub repo.
3. Render reads `render.yaml` and shows a preview of the two resources.
4. Click **Apply**. Wait ~5 minutes for the first build.

### Option B — Manual (if Blueprints unavailable)

1. **Web Service** → New + → Web Service → connect repo:
   - **Runtime:** Python 3.11
   - **Region:** Singapore
   - **Branch:** `main`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT --workers 1`
   - **Health Check Path:** `/api/v1/health`
   - **Plan:** Free
2. **Database** → New + → PostgreSQL → name `cardiosurv-db`, region Singapore, plan Free.
3. In the Web Service → **Environment** tab, set:

   | Key | Value |
   |---|---|
   | `PYTHON_VERSION` | `3.11.9` |
   | `ENVIRONMENT` | `production` |
   | `MODEL_DIR` | `./models` |
   | `PART1_MODEL_PATH` | `models/part1_classifier_v1.0.pkl` |
   | `PART2_MODEL_PATH` | `models/part2_recommender_v1.0.pkl` |
   | `SURVIVAL_MODEL_PATH` | `models/survival_cox_v1.0.pkl` |
   | `FEATURES_CSV` | `data/processed/features.csv` |
   | `CORS_ORIGINS` | `*` (tighten once frontend URL is known) |
   | `DATABASE_URL` | *(use the Internal Database URL from cardiosurv-db)* |

4. **Auto-Deploy:** make sure it is **On** for pushes to `main`.

---

## 3. Initial DB setup (run **once** after first deploy)

The schema is not migrated automatically. SSH into the service via the
dashboard **Shell** tab and run:

```bash
python -c "from src.db.session import engine; from src.db.models import Base; Base.metadata.create_all(engine)"
```

You should see no output (success). Confirm with:

```bash
python -c "from src.db.session import engine; print(engine.dialect.name); print(list(engine.dialect.get_table_names(engine.connect())))"
```

Expected: `postgresql` and `['patients', 'predictions', 'recommendations', 'audit_logs']`.

---

## 4. Smoke-test the live service

From your laptop, run:

```bash
bash scripts/smoke_test.sh https://cardiosurv-api.onrender.com
```

The script hits all 5 endpoints in order:

| # | Endpoint | Expected |
|---|----------|----------|
| 1 | `GET  /api/v1/health` | `200` + `model_versions.part1_classifier="v1.0"`, `part2_recommender="v1.0"` |
| 2 | `POST /api/v1/predict` | `200` + `risk_category="High"` |
| 3 | `POST /api/v1/recommend` | `200` + `branch="SBRT"`, `bed_gy=87.5` |
| 4 | `GET  /api/v1/history` | `200` + ≥1 item |
| 5 | `GET  /api/v1/patients/{id}` | `200` + full record |

If any step fails, **stop and check the logs** (next section) before proceeding.

You can also paste a single curl into the terminal:

```bash
curl https://cardiosurv-api.onrender.com/api/v1/health | python -m json.tool
```

---

## 5. Checking logs

### Live tail (Render dashboard)
Web Service → **Logs** tab → "Live" toggle on top right.

Logs are filtered by severity. The startup banner you want to see:

```
INFO:src.api.main:[lifespan] Starting up — loading models...
INFO:src.api.main:[lifespan] ✓ Part-1 classifier loaded
INFO:src.api.main:[lifespan] ✓ Part-2 recommender loaded
INFO:src.api.main:[cors] Allowed origins: ['*']
INFO:     Uvicorn running on http://0.0.0.0:10000
```

If you see `✗ Part-2 model file not found`, the model `.pkl` is missing from
the repo or `PART2_MODEL_PATH` is set wrong.

### Logs from the CLI (Render CLI)

```bash
render logs --service cardiosurv-api --tail
```

---

## 6. Keep the demo warm

Render free tier **sleeps after 15 minutes of inactivity**, with a 30–60 s
cold-start delay on the next request. For the live demo on **25 June 2026**,
the team should ping the health endpoint every ~10 minutes for the hour
before the demo.

### Option 1 — local laptop (simplest)

```bash
python scripts/keep_warm.py --url https://cardiosurv-api.onrender.com/api/v1/health
```

Leave it running in a terminal tab.

### Option 2 — UptimeRobot (free, runs in cloud)

1. Sign up at <https://uptimerobot.com>
2. New Monitor → HTTP(s) → URL: `https://cardiosurv-api.onrender.com/api/v1/health`
3. Interval: 5 minutes (free tier minimum)

### Option 3 — cron-job.org

Free, web-based, no install. Same target URL, interval 10 minutes.

---

## 7. Common issues & fixes

| Symptom | Cause | Fix |
|---|---|---|
| `404 Not Found` on the live URL | Service still building/booting | Wait 2-3 minutes, check Logs tab |
| `502 Bad Gateway` | Worker crashed at startup | Check Logs for the traceback — usually missing `.pkl` or DB URL |
| `health` returns `part2_recommender: "NOT_LOADED"` | Part-2 `.pkl` not in repo, or wrong env var | `git ls-files models/` → confirm `.pkl` is committed. Check `PART2_MODEL_PATH`. |
| `Worker (pid:N) was sent SIGKILL!` | Out-of-memory (Render free tier = 512 MB) | Reduce workers to 1 in `Procfile`/`render.yaml` (already done). |
| Frontend gets CORS error | `CORS_ORIGINS` doesn't include the frontend URL | Update `CORS_ORIGINS` in dashboard → "Manual Deploy" → "Deploy latest commit" |
| DB is gone after 90 days | Free Postgres expires after 90 days | Provision a new DB, repoint `DATABASE_URL`, rerun §3 |

---

## 8. Updating the deployment

Push to `main` — auto-deploy is on. To trigger a redeploy without code change:
dashboard → **Manual Deploy** → **Deploy latest commit**.

To roll back: dashboard → **Events** tab → find the previous successful deploy
→ **Rollback**.

---

## 9. Tear-down

To avoid forgotten cloud bills:

1. Dashboard → cardiosurv-api → **Settings** → **Delete Service**
2. Dashboard → cardiosurv-db → **Settings** → **Delete Database**
3. Dashboard → cardiosurv-frontend (Law Zi Ying's Static Site) → **Delete**

Free plans have no charge, but cleaning up keeps the team's dashboard tidy.
