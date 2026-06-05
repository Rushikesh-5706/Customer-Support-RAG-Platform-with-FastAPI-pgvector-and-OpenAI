import logging
import psycopg2
from contextlib import asynccontextmanager
from pathlib import Path

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

_MIGRATION_SQL = (
    Path(__file__).parent.parent / "database" / "migrations" / "001_initial.sql"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Running database migration...")
    try:
        conn = psycopg2.connect(settings.database_url)
        with conn.cursor() as cur:
            cur.execute(_MIGRATION_SQL.read_text())
        conn.commit()
        conn.close()
        logger.info("Database migration complete.")
    except Exception as exc:
        logger.error("Migration failed: %s", exc)
        raise
    yield
    logger.info("IntelliSupport shutting down.")


app = FastAPI(
    title="IntelliSupport — Customer Support RAG Platform",
    description=(
        "Production-grade Retrieval-Augmented Generation platform for B2B SaaS "
        "customer support. Hybrid BM25 and vector retrieval, intent classification, "
        "LLM-as-judge evaluation, and structured feedback collection."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["Root"])
def root():
    return {
        "service": "IntelliSupport RAG Platform",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
