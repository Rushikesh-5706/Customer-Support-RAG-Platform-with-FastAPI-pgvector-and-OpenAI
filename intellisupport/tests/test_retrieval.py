"""
Tests for the retrieval module: VectorStore, BM25Retriever, HybridRetriever.
All spec-required test function names are included.
"""

import pytest
from unittest.mock import MagicMock, patch
from retrieval.vector_store import VectorStore, RetrievedChunk
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever


def make_chunk(chunk_id: str, doc_id: str = "doc_001", content: str = "test content",
               score: float = 0.9, method: str = "vector") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content,
        score=score,
        retrieval_method=method,
    )


# ─── VectorStore ──────────────────────────────────────────────────────────────

class TestVectorStore:

    def _mock_conn(self, rows):
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = rows
        conn.cursor.return_value.__enter__.return_value = cursor
        return conn

    # SPEC-REQUIRED: exact name
    def test_vector_similarity_search_returns_k_results(self):
        """similarity_search must return exactly top_k results."""
        rows = [
            ("chunk_doc_001_0", "doc_001", "Billing content", 0.92),
            ("chunk_doc_002_0", "doc_002", "Technical content", 0.85),
        ]
        store = VectorStore(self._mock_conn(rows))
        results = store.similarity_search([0.1] * 5, top_k=2)
        assert len(results) == 2
        assert results[0].chunk_id == "chunk_doc_001_0"
        assert results[0].retrieval_method == "vector"

    # SPEC-REQUIRED: exact name
    def test_vector_similarity_scores_range(self):
        """Vector similarity scores must be in [0.0, 1.0]."""
        rows = [
            ("chunk_doc_001_0", "doc_001", "Content A", 0.92),
            ("chunk_doc_002_0", "doc_002", "Content B", 0.78),
            ("chunk_doc_003_0", "doc_003", "Content C", 0.55),
        ]
        store = VectorStore(self._mock_conn(rows))
        results = store.similarity_search([0.1] * 5, top_k=3)
        for r in results:
            assert 0.0 <= r.score <= 1.0, f"Score {r.score} out of [0,1] range"

    def test_similarity_search_returns_chunks(self):
        rows = [
            ("chunk_doc_001_0", "doc_001", "Billing content", 0.92),
            ("chunk_doc_002_0", "doc_002", "Technical content", 0.85),
        ]
        store = VectorStore(self._mock_conn(rows))
        results = store.similarity_search([0.1] * 5, top_k=2)
        assert len(results) == 2
        assert results[0].score == pytest.approx(0.92)

    def test_similarity_search_with_threshold_returns_filtered(self):
        rows = [("chunk_doc_001_0", "doc_001", "Content", 0.88)]
        store = VectorStore(self._mock_conn(rows))
        results = store.similarity_search_with_threshold([0.1] * 5, top_k=5, threshold=0.75)
        assert len(results) == 1
        assert results[0].score == pytest.approx(0.88)

    def test_similarity_search_empty_db(self):
        store = VectorStore(self._mock_conn([]))
        results = store.similarity_search([0.1] * 5, top_k=5)
        assert results == []


# ─── BM25Retriever ────────────────────────────────────────────────────────────

class TestBM25Retriever:

    def _make_retriever(self, chunks):
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = [
            (c["chunk_id"], c["doc_id"], c["content"]) for c in chunks
        ]
        conn.cursor.return_value.__enter__.return_value = cursor
        return BM25Retriever(conn)

    def _five_chunk_corpus(self):
        return [
            {"chunk_id": "chunk_doc_001_0", "doc_id": "doc_001",
             "content": "billing invoice payment refund subscription plan"},
            {"chunk_id": "chunk_doc_002_0", "doc_id": "doc_002",
             "content": "technical error authentication api key token"},
            {"chunk_id": "chunk_doc_003_0", "doc_id": "doc_003",
             "content": "account management team member invitation role"},
            {"chunk_id": "chunk_doc_004_0", "doc_id": "doc_004",
             "content": "slack integration webhook channel notification"},
            {"chunk_id": "chunk_doc_005_0", "doc_id": "doc_005",
             "content": "data export gdpr csv json backup archive"},
        ]

    # SPEC-REQUIRED: exact name
    def test_bm25_search_returns_results(self):
        """BM25 must return at least 1 result for a query matching corpus terms."""
        retriever = self._make_retriever(self._five_chunk_corpus())
        results = retriever.search("billing invoice payment", top_k=3)
        assert len(results) >= 1
        assert results[0].retrieval_method == "bm25"
        assert results[0].chunk_id == "chunk_doc_001_0"

    # SPEC-REQUIRED: exact name — checks BM25 ranks keyword-matching chunk highest
    def test_bm25_keyword_relevance(self):
        """BM25 must rank the chunk with exact keyword overlap first."""
        retriever = self._make_retriever(self._five_chunk_corpus())
        results = retriever.search("billing invoice payment", top_k=5)
        assert len(results) >= 1
        # The chunk about billing must be ranked first for a billing query
        assert results[0].chunk_id == "chunk_doc_001_0", (
            f"Expected billing chunk at rank 1, got '{results[0].chunk_id}'"
        )

    def test_search_returns_results(self):
        retriever = self._make_retriever(self._five_chunk_corpus())
        results = retriever.search("billing invoice payment", top_k=3)
        assert len(results) >= 1
        assert results[0].retrieval_method == "bm25"

    def test_search_scores_in_range(self):
        chunks = [
            {"chunk_id": "chunk_doc_001_0", "doc_id": "doc_001", "content": "refund payment plan upgrade billing"},
            {"chunk_id": "chunk_doc_002_0", "doc_id": "doc_002", "content": "login reset password mfa authentication"},
            {"chunk_id": "chunk_doc_003_0", "doc_id": "doc_003", "content": "slack github jira integration webhook"},
            {"chunk_id": "chunk_doc_004_0", "doc_id": "doc_004", "content": "export csv json gdpr data backup"},
            {"chunk_id": "chunk_doc_005_0", "doc_id": "doc_005", "content": "feature roadmap request product capability"},
        ]
        retriever = self._make_retriever(chunks)
        results = retriever.search("payment refund billing", top_k=3)
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_empty_corpus_returns_empty(self):
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        conn.cursor.return_value.__enter__.return_value = cursor
        retriever = BM25Retriever(conn)
        results = retriever.search("anything", top_k=5)
        assert results == []

    def test_rebuild_index(self):
        chunks = [{"chunk_id": "c1", "doc_id": "doc_001", "content": "hello world"}]
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = [(c["chunk_id"], c["doc_id"], c["content"]) for c in chunks]
        conn.cursor.return_value.__enter__.return_value = cursor
        retriever = BM25Retriever(conn)
        retriever.rebuild_index(conn)
        assert retriever._index is not None


# ─── HybridRetriever ──────────────────────────────────────────────────────────

class TestHybridRetriever:

    def _make_hybrid(self, vector_results, bm25_results):
        vector_store = MagicMock(spec=VectorStore)
        vector_store.similarity_search.return_value = vector_results

        bm25 = MagicMock(spec=BM25Retriever)
        bm25.search.return_value = bm25_results

        return HybridRetriever(vector_store, bm25, alpha=0.7)

    # SPEC-REQUIRED: exact name — verifies duplicate chunk_ids are merged, not doubled
    def test_hybrid_deduplication(self):
        """When vector and BM25 return the same chunk_id, it must appear only once."""
        shared_chunk = make_chunk("c1", score=1.0, method="vector")
        bm25_chunk = make_chunk("c1", score=0.8, method="bm25")
        hybrid = self._make_hybrid([shared_chunk], [bm25_chunk])
        results = hybrid.retrieve("test query", [0.1] * 5, top_k=5)
        chunk_ids = [r.chunk_id for r in results]
        assert chunk_ids.count("c1") == 1, (
            f"Duplicate chunk_id 'c1' appeared {chunk_ids.count('c1')} times — must be deduplicated"
        )

    # SPEC-REQUIRED: exact name — verifies all scores are in valid range
    def test_hybrid_retriever_score_range(self):
        """All hybrid scores must be in [0.0, 1.0]."""
        v_chunks = [make_chunk("c1", score=0.5, method="vector")]
        b_chunks = [make_chunk("c2", score=0.3, method="bm25")]
        hybrid = self._make_hybrid(v_chunks, b_chunks)
        results = hybrid.retrieve("query", [0.0] * 5, top_k=5)
        for r in results:
            assert 0.0 <= r.score <= 1.0, f"Score {r.score} out of [0,1] range"

    def test_retrieve_merges_results(self):
        v_chunks = [make_chunk("c1", score=0.9, method="vector")]
        b_chunks = [make_chunk("c2", score=0.8, method="bm25")]
        hybrid = self._make_hybrid(v_chunks, b_chunks)
        results = hybrid.retrieve("test query", [0.1] * 5, top_k=5)
        chunk_ids = [r.chunk_id for r in results]
        assert "c1" in chunk_ids
        assert "c2" in chunk_ids

    def test_retrieve_hybrid_score_blend(self):
        v_chunks = [make_chunk("c1", score=1.0, method="vector")]
        b_chunks = [make_chunk("c1", score=1.0, method="bm25")]
        hybrid = self._make_hybrid(v_chunks, b_chunks)
        results = hybrid.retrieve("test", [0.1] * 5, top_k=5)
        assert len(results) == 1
        assert results[0].score == pytest.approx(1.0)
        assert results[0].retrieval_method == "hybrid"

    def test_retrieve_with_reranking_returns_top_k(self):
        chunks = [make_chunk(f"c{i}", score=float(i) / 10) for i in range(10)]
        hybrid = self._make_hybrid(chunks[:5], chunks[5:])
        results = hybrid.retrieve_with_reranking("test", [0.1] * 5, top_k=3)
        assert len(results) == 3

    def test_retrieve_scores_not_negative(self):
        v_chunks = [make_chunk("c1", score=0.5, method="vector")]
        b_chunks = [make_chunk("c2", score=0.3, method="bm25")]
        hybrid = self._make_hybrid(v_chunks, b_chunks)
        results = hybrid.retrieve("query", [0.0] * 5, top_k=5)
        for r in results:
            assert r.score >= 0.0
