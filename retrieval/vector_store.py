import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    score: float
    retrieval_method: str


class VectorStore:

    def __init__(self, conn):
        self._conn = conn

    def similarity_search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[RetrievedChunk]:
        sql = """
            SELECT
                chunk_id,
                doc_id,
                content,
                1 - (embedding <=> %s::vector) AS sim_score
            FROM intellisupport.chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        vec = str(query_embedding)
        with self._conn.cursor() as cur:
            cur.execute(sql, (vec, vec, top_k))
            rows = cur.fetchall()
        return [
            RetrievedChunk(
                chunk_id=row[0],
                doc_id=row[1],
                content=row[2],
                score=float(row[3]),
                retrieval_method="vector",
            )
            for row in rows
        ]

    def similarity_search_with_threshold(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.75,
    ) -> list[RetrievedChunk]:
        sql = """
            SELECT
                chunk_id,
                doc_id,
                content,
                1 - (embedding <=> %s::vector) AS sim_score
            FROM intellisupport.chunks
            WHERE 1 - (embedding <=> %s::vector) >= %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        vec = str(query_embedding)
        with self._conn.cursor() as cur:
            cur.execute(sql, (vec, vec, threshold, vec, top_k))
            rows = cur.fetchall()
        return [
            RetrievedChunk(
                chunk_id=row[0],
                doc_id=row[1],
                content=row[2],
                score=float(row[3]),
                retrieval_method="vector",
            )
            for row in rows
        ]
