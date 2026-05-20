"""
src/db/session.py
-----------------
Database engine, session factory, and FastAPI dependency.

Reads DATABASE_URL from the environment (falls back to SQLite for local dev).
Prod can swap to Postgres by setting:
    DATABASE_URL=postgresql://user:pass@host:5432/cardiosurv

Usage (FastAPI dependency injection):
    from src.db.session import get_db

    @app.post("/api/v1/predict")
    def predict(body: PatientVitalsRequest, db: Session = Depends(get_db)):
        ...

Usage (standalone / scripts):
    from src.db.session import SessionLocal
    db = SessionLocal()
    try:
        ...
    finally:
        db.close()
"""

from __future__ import annotations

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///cardiosurv.db")

# connect_args is only needed for SQLite (multi-thread safety in FastAPI)
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    echo=False,          # set echo=True to log all SQL statements during debugging
)

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)

# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

def get_db() -> Generator[Session, None, None]:
    """
    Yield a database session and guarantee it is closed after the request,
    even if an exception is raised.

    Inject with:  db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
