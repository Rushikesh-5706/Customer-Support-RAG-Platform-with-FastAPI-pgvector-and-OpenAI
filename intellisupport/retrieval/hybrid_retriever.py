import logging
from retrieval.vector_store import VectorStore, RetrievedChunk
from retrieval.bm25_retriever import BM25Retriever

logger = logging.getLogger(__name__)


class HybridRetriever:

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        alpha: float = 0.7,
    ):
        self._vector_store = vector_store
        self._bm25_retriever = bm25_retriever
        self.alpha = alpha

    def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        candidate_count = top_k * 2

        vector_results = self._vector_store.similarity_search(
            query_embedding, top_k=candidate_count
        )
        bm25_results = self._bm25_retriever.search(query, top_k=candidate_count)

        vector_scores: dict[str, float] = {r.chunk_id: r.score for r in vector_results}
        bm25_scores: dict[str, float] = {r.chunk_id: r.score for r in bm25_results}

        chunk_registry: dict[str, RetrievedChunk] = {}
        for r in vector_results + bm25_results:
            if r.chunk_id not in chunk_registry:
                chunk_registry[r.chunk_id] = r

        merged_chunks: dict[str, RetrievedChunk] = {}
        all_ids = set(vector_scores) | set(bm25_scores)

        for cid in all_ids:
            v = vector_scores.get(cid, 0.0)
            b = bm25_scores.get(cid, 0.0)

            if cid in vector_scores and cid in bm25_scores:
                final_score = self.alpha * v + (1 - self.alpha) * b
                method = "hybrid"
            elif cid in vector_scores:
                final_score = self.alpha * v
                method = "vector"
            else:
                final_score = (1 - self.alpha) * b
                method = "bm25"

            ref = chunk_registry[cid]
            merged_chunks[cid] = RetrievedChunk(
                chunk_id=cid,
                doc_id=ref.doc_id,
                content=ref.content,
                score=final_score,
                retrieval_method=method,
            )

        ranked = sorted(merged_chunks.values(), key=lambda c: c.score, reverse=True)
        return ranked[:top_k]

    def retrieve_with_reranking(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        candidates = self.retrieve(query, query_embedding, top_k=top_k * 3)
        query_tokens = set(query.lower().split())

        reranked: list[RetrievedChunk] = []
        for chunk in candidates:
            chunk_tokens = set(chunk.content.lower().split())
            union = len(query_tokens | chunk_tokens)
            intersection = len(query_tokens & chunk_tokens)
            jaccard = intersection / union if union > 0 else 0.0
            rerank_score = 0.8 * chunk.score + 0.2 * jaccard
            reranked.append(RetrievedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                content=chunk.content,
                score=rerank_score,
                retrieval_method=chunk.retrieval_method,
            ))

        reranked.sort(key=lambda c: c.score, reverse=True)
        return reranked[:top_k]
