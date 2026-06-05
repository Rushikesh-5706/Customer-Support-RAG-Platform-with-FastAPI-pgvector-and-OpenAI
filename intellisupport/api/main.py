"""
IntelliSupport — FastAPI application entrypoint.

Run with:
    uvicorn api.main:app --reload
(from the intellisupport/ directory)
"""

import logging
import psycopg2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="IntelliSupport — Customer Support RAG Platform",
    description=(
        "Production-grade Retrieval-Augmented Generation (RAG) platform for B2B SaaS "
        "customer support. Features hybrid BM25+vector retrieval, intent classification, "
        "LLM-as-judge evaluation, and structured feedback collection."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
def run_migrations() -> None:
    logger.info("Running database migrations...")
    try:
        conn = psycopg2.connect(settings.database_url)
        with open("database/migrations/001_initial.sql", "r") as f:
            migration_sql = f.read()
        with conn.cursor() as cur:
            cur.execute(migration_sql)
        conn.commit()
        conn.close()
        logger.info("Database migration completed successfully.")
    except Exception as exc:
        logger.error("Migration failed: %s", exc)
        raise


@app.get("/", tags=["Root"])
def root():
    return {
        "service": "IntelliSupport RAG Platform",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
