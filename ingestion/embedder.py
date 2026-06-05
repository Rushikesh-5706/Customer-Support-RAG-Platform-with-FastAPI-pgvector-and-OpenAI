import time
import json
import logging
from openai import OpenAI
from ingestion.chunker import Chunk
from config import settings

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    pass


class Embedder:

    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 100):
        self.model = model
        self.batch_size = batch_size
        self._client = OpenAI(api_key=settings.openai_api_key)

    def embed_text(self, text: str) -> list[float]:
        last_exc: Exception = None
        for attempt in range(3):
            try:
                resp = self._client.embeddings.create(model=self.model, input=text)
                return resp.data[0].embedding
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning(
                    "Embedding call failed (attempt %d/3): %s. Retrying in %ds.",
                    attempt + 1, exc, wait,
                )
                time.sleep(wait)
        raise EmbeddingError(
            f"OpenAI embedding API failed after 3 attempts. Last error: {last_exc}"
        )

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start: start + self.batch_size]
            last_exc: Exception = None
            for attempt in range(3):
                try:
                    resp = self._client.embeddings.create(model=self.model, input=batch)
                    ordered = sorted(resp.data, key=lambda e: e.index)
                    all_embeddings.extend(e.embedding for e in ordered)
                    break
                except Exception as exc:
                    last_exc = exc
                    wait = 2 ** attempt
                    logger.warning(
                        "Batch embedding failed (attempt %d/3): %s. Retrying in %ds.",
                        attempt + 1, exc, wait,
                    )
                    time.sleep(wait)
            else:
                raise EmbeddingError(
                    f"Batch embedding failed after 3 attempts. Last error: {last_exc}"
                )
        return all_embeddings

    def embed_and_store_chunks(self, chunks: list[Chunk], conn) -> int:
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embed_batch(texts)

        insert_sql = """
            INSERT INTO intellisupport.chunks
                (chunk_id, doc_id, content, chunk_index, token_count, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s, %s::vector, %s)
            ON CONFLICT (chunk_id) DO UPDATE
                SET content     = EXCLUDED.content,
                    chunk_index = EXCLUDED.chunk_index,
                    token_count = EXCLUDED.token_count,
                    embedding   = EXCLUDED.embedding,
                    metadata    = EXCLUDED.metadata
        """
        count = 0
        with conn.cursor() as cur:
            for chunk, embedding in zip(chunks, embeddings):
                cur.execute(insert_sql, (
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.content,
                    chunk.chunk_index,
                    chunk.token_count,
                    str(embedding),
                    json.dumps(chunk.metadata),
                ))
                count += 1
        conn.commit()
        return count
