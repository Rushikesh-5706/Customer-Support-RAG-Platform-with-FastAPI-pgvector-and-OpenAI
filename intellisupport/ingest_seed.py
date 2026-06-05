"""
Ingestion runner script for IntelliSupport seed documents.

Usage:
    python ingest_seed.py

Prerequisites:
    1. PostgreSQL running and accepting connections (DATABASE_URL in .env).
    2. A valid OPENAI_API_KEY in .env.
    3. pip install -r requirements.txt completed.

This script runs the SQL migration, loads all seed documents, chunks them,
embeds them via OpenAI, and stores everything in PostgreSQL.
"""

import logging
import psycopg2

from config import settings
from ingestion.loader import DocumentLoader
from ingestion.chunker import DocumentChunker
from ingestion.embedder import Embedder
from seed_documents import SEED_DOCUMENTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_migration(conn) -> None:
    from pathlib import Path
    migration_path = Path(__file__).parent / "database" / "migrations" / "001_initial.sql"
    logger.info("Running database migration from %s...", migration_path)
    sql = migration_path.read_text()
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    logger.info("Migration complete.")


def main() -> None:
    logger.info("Connecting to database: %s", settings.database_url)
    conn = psycopg2.connect(settings.database_url)

    run_migration(conn)

    loader = DocumentLoader()
    chunker = DocumentChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    embedder = Embedder(model=settings.embedding_model)

    logger.info("Loading %d seed documents...", len(SEED_DOCUMENTS))
    documents = loader.load_batch(SEED_DOCUMENTS)
    logger.info("Validated %d documents.", len(documents))

    docs_saved = loader.save_to_db(documents, conn)
    logger.info("Saved %d documents to database.", docs_saved)

    chunks = chunker.chunk_batch(documents)
    logger.info("Created %d chunks from %d documents.", len(chunks), len(documents))

    logger.info("Embedding %d chunks via OpenAI (model: %s)...", len(chunks), settings.embedding_model)
    chunks_embedded = embedder.embed_and_store_chunks(chunks, conn)
    logger.info("Embedded and stored %d chunks.", chunks_embedded)

    conn.close()
    logger.info("Seed ingestion complete. %d docs | %d chunks | %d embeddings.",
                docs_saved, len(chunks), chunks_embedded)


if __name__ == "__main__":
    main()
