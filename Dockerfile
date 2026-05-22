# CardioSurv API — production Dockerfile
# Used by docker-compose for local prod-shape testing.
# Render works either with this Dockerfile (Docker runtime) or with the
# Procfile (Python runtime) — the team picked Python runtime in render.yaml.

FROM python:3.11-slim

# System deps (psycopg2 needs libpq, lifelines needs gcc for some pure-python builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache pip layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt psycopg2-binary

# Copy source
COPY src     ./src
COPY models  ./models
COPY data    ./data

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
