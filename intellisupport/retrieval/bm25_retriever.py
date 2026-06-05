import logging
from rank_bm25 import BM25Okapi
from retrieval.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)


class BM25Retriever:

    def __init__(self, conn):
        self._conn = conn
        self._chunks: list[dict] = []
        self._index: BM25Okapi = None
        self._build_index()

    def _load_chunks(self) -> list[dict]:
        sql = """
            SELECT chunk_id, doc_id, content
            FROM intellisupport.chunks
            ORDER BY doc_id, chunk_index
        """
        with self._conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        return [{"chunk_id": r[0], "doc_id": r[1], "content": r[2]} for r in rows]

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def _build_index(self) -> None:
        self._chunks = self._load_chunks()
        if not self._chunks:
            logger.warning("No chunks in database. BM25 index is empty.")
            self._index = None
            return
        tokenized = [self._tokenize(ch["content"]) for ch in self._chunks]
        self._index = BM25Okapi(tokenized)
        logger.info("BM25 index built with %d chunks.", len(self._chunks))

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if self._index is None or not self._chunks:
            return []

        tokenized_query = self._tokenize(query)
        raw_scores = self._index.get_scores(tokenized_query)
        max_score = float(max(raw_scores))

        if max_score == 0.0:
            return []

        ranked = sorted(enumerate(raw_scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, raw_score in ranked:
            normalized = float(raw_score) / max_score
            ch = self._chunks[idx]
            results.append(RetrievedChunk(
                chunk_id=ch["chunk_id"],
                doc_id=ch["doc_id"],
                content=ch["content"],
                score=normalized,
                retrieval_method="bm25",
            ))
        return results

    def rebuild_index(self, conn) -> None:
        self._conn = conn
        self._build_index()
