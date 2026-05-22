# CardioSurv API — Deployment Runbook

##  Deploy to Render (Free Tier)

### Step 1: Create Web Service
1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **New +** → **Web Service**
3. Connect your GitHub repo
4. Configure:
   - **Runtime**: Python 3.11
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
   - **Environment Variables**: Add from `.env.example`

### Step 2: Provision Database
1. In Render dashboard, create **PostgreSQL** (free tier)
2. Copy `Internal Database URL` → set as `DATABASE_URL` env var
3. Run DB init via Render shell:
   ```bash
   python -c "from src.db.session import engine; from src.db.models import Base; Base.metadata.create_all(engine)"
