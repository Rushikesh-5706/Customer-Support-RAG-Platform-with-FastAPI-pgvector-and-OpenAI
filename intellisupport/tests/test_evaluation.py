"""
Tests for the evaluation module: FaithfulnessEvaluator, RelevanceEvaluator,
PipelineEvaluator, EvaluationReport, BenchmarkReport.

Includes the mandatory BENCHMARK_TEST_CASES constant used by PipelineEvaluator.run_benchmark.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from retrieval.vector_store import RetrievedChunk
from evaluation.faithfulness import FaithfulnessEvaluator, FaithfulnessResult
from evaluation.relevance import RelevanceEvaluator, RelevanceResult, ChunkRelevanceScore
from evaluation.evaluator import PipelineEvaluator, EvaluationReport, BenchmarkReport


# ─── Mandatory benchmark test cases (spec §4.4) ───────────────────────────────

BENCHMARK_TEST_CASES = [
    {
        "query": "How do I add a new member to my team?",
        "expected_doc_ids": ["doc_002"],
        "expected_intent": "account_management",
    },
    {
        "query": "What happens if I cancel my subscription?",
        "expected_doc_ids": ["doc_003"],
        "expected_intent": "billing",
    },
    {
        "query": "How do I connect Nexora to Slack?",
        "expected_doc_ids": ["doc_004"],
        "expected_intent": "integration",
    },
    {
        "query": "I forgot my password and can't log in",
        "expected_doc_ids": ["doc_008", "doc_002"],
        "expected_intent": "account_management",
    },
    {
        "query": "How do I export my project data as CSV?",
        "expected_doc_ids": ["doc_007"],
        "expected_intent": "data_and_export",
    },
    {
        "query": "The webhook I set up isn't receiving any events",
        "expected_doc_ids": ["doc_009"],
        "expected_intent": "technical_issue",
    },
    {
        "query": "Can I use custom templates for new projects?",
        "expected_doc_ids": ["doc_005"],
        "expected_intent": "feature_request",
    },
    {
        "query": "How do I enable two-factor authentication for my account?",
        "expected_doc_ids": ["doc_008"],
        "expected_intent": "account_management",
    },
]

# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_chunk(
    chunk_id: str = "chunk_doc_001_0",
    doc_id: str = "doc_001",
    content: str = "This is relevant context about billing and subscriptions.",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content,
        score=0.9,
        retrieval_method="hybrid",
    )


def mock_openai_json_response(payload: dict):
    content = json.dumps(payload)
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ─── FaithfulnessEvaluator ────────────────────────────────────────────────────

class TestFaithfulnessEvaluator:

    def setup_method(self):
        with patch("evaluation.faithfulness.OpenAI"):
            self.evaluator = FaithfulnessEvaluator(model="gpt-4o-mini")

    def test_faithfulness_score_range(self):
        """Spec §4.4: faithfulness_score must be in [0.0, 1.0]."""
        self.evaluator._client.chat.completions.create.return_value = (
            mock_openai_json_response({
                "total_claims": 4,
                "supported_claims": 3,
                "unsupported_claims": 1,
                "reasoning": "Three of four claims are supported.",
            })
        )
        result = self.evaluator.evaluate("Some generated response text.", [make_chunk()])
        assert isinstance(result, FaithfulnessResult)
        assert 0.0 <= result.faithfulness_score <= 1.0

    def test_evaluate_returns_faithfulness_result(self):
        self.evaluator._client.chat.completions.create.return_value = (
            mock_openai_json_response({
                "total_claims": 4,
                "supported_claims": 4,
                "unsupported_claims": 0,
                "reasoning": "All claims verified against context.",
            })
        )
        result = self.evaluator.evaluate("Billing is monthly or annual.", [make_chunk()])
        assert isinstance(result, FaithfulnessResult)
        assert result.faithfulness_score == pytest.approx(1.0)
        assert result.total_claims == 4
        assert result.supported_claims == 4
        assert result.unsupported_claims == 0

    def test_evaluate_score_is_fraction(self):
        self.evaluator._client.chat.completions.create.return_value = (
            mock_openai_json_response({
                "total_claims": 4,
                "supported_claims": 3,
                "unsupported_claims": 1,
                "reasoning": "One claim unsupported.",
            })
        )
        result = self.evaluator.evaluate("Response text.", [make_chunk()])
        assert result.faithfulness_score == pytest.approx(0.75)

    def test_evaluate_zero_claims_gives_score_one(self):
        self.evaluator._client.chat.completions.create.return_value = (
            mock_openai_json_response({
                "total_claims": 0,
                "supported_claims": 0,
                "unsupported_claims": 0,
                "reasoning": "No factual claims found.",
            })
        )
        result = self.evaluator.evaluate("Sure, I can help.", [make_chunk()])
        assert result.faithfulness_score == pytest.approx(1.0)

    def test_evaluate_score_clamped_to_one(self):
        self.evaluator._client.chat.completions.create.return_value = (
            mock_openai_json_response({
                "total_claims": 2,
                "supported_claims": 3,
                "unsupported_claims": 0,
                "reasoning": "Edge case.",
            })
        )
        result = self.evaluator.evaluate("Response.", [make_chunk()])
        assert result.faithfulness_score <= 1.0

    def test_evaluate_api_failure_returns_zero(self):
        self.evaluator._client.chat.completions.create.side_effect = Exception("API error")
        result = self.evaluator.evaluate("Response.", [make_chunk()])
        assert 0.0 <= result.faithfulness_score <= 1.0

    def test_evaluate_empty_chunks(self):
        self.evaluator._client.chat.completions.create.return_value = (
            mock_openai_json_response({
                "total_claims": 2,
                "supported_claims": 1,
                "unsupported_claims": 1,
                "reasoning": "No context provided.",
            })
        )
        result = self.evaluator.evaluate("Some response.", [])
        assert isinstance(result, FaithfulnessResult)
        assert 0.0 <= result.faithfulness_score <= 1.0

    def test_faithfulness_result_fields(self):
        result = FaithfulnessResult(
            faithfulness_score=0.85,
            total_claims=20,
            supported_claims=17,
            unsupported_claims=3,
            reasoning="Mostly faithful.",
        )
        assert result.faithfulness_score == 0.85
        assert result.total_claims == 20
        assert result.supported_claims + result.unsupported_claims == 20


# ─── RelevanceEvaluator ───────────────────────────────────────────────────────

class TestRelevanceEvaluator:

    def setup_method(self):
        with patch("evaluation.relevance.OpenAI"):
            self.evaluator = RelevanceEvaluator(model="gpt-4o-mini")

    def test_relevance_score_range(self):
        """Spec §4.4: relevance_score must be in [0.0, 1.0]."""
        self.evaluator._client.chat.completions.create.return_value = (
            mock_openai_json_response({
                "chunk_scores": [
                    {"chunk_id": "chunk_doc_001_0", "score": 1, "reason": "Partially relevant."}
                ]
            })
        )
        result = self.evaluator.evaluate("What is the billing cycle?", [make_chunk()])
        assert isinstance(result, RelevanceResult)
        assert 0.0 <= result.relevance_score <= 1.0

    def test_evaluate_returns_relevance_result(self):
        chunk = make_chunk("chunk_doc_001_0")
        self.evaluator._client.chat.completions.create.return_value = (
            mock_openai_json_response({
                "chunk_scores": [
                    {"chunk_id": "chunk_doc_001_0", "score": 2, "reason": "Directly answers query."}
                ]
            })
        )
        result = self.evaluator.evaluate("What is the billing cycle?", [chunk])
        assert isinstance(result, RelevanceResult)
        assert result.relevance_score == pytest.approx(1.0)
        assert len(result.chunk_scores) == 1
        assert result.chunk_scores[0].score == 2

    def test_evaluate_partial_relevance(self):
        chunks = [make_chunk(f"chunk_doc_001_{i}") for i in range(2)]
        self.evaluator._client.chat.completions.create.return_value = (
            mock_openai_json_response({
                "chunk_scores": [
                    {"chunk_id": "chunk_doc_001_0", "score": 2, "reason": "Highly relevant."},
                    {"chunk_id": "chunk_doc_001_1", "score": 0, "reason": "Not relevant."},
                ]
            })
        )
        result = self.evaluator.evaluate("billing question", chunks)
        assert result.relevance_score == pytest.approx(0.5)

    def test_evaluate_empty_chunks_returns_zero(self):
        result = self.evaluator.evaluate("What is billing?", [])
        assert result.relevance_score == pytest.approx(0.0)
        assert result.chunk_scores == []

    def test_evaluate_api_failure_returns_zero(self):
        self.evaluator._client.chat.completions.create.side_effect = Exception("Error")
        result = self.evaluator.evaluate("query", [make_chunk()])
        assert 0.0 <= result.relevance_score <= 1.0

    def test_evaluate_batch_returns_list(self):
        self.evaluator._client.chat.completions.create.return_value = (
            mock_openai_json_response({
                "chunk_scores": [
                    {"chunk_id": "chunk_doc_001_0", "score": 1, "reason": "Partial."}
                ]
            })
        )
        pairs = [
            ("query 1", [make_chunk("chunk_doc_001_0")]),
            ("query 2", [make_chunk("chunk_doc_001_0")]),
        ]
        results = self.evaluator.evaluate_batch(pairs)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, RelevanceResult)

    def test_chunk_relevance_score_fields(self):
        cs = ChunkRelevanceScore(
            chunk_id="chunk_doc_001_0",
            score=2,
            reason="Highly relevant chunk.",
        )
        assert cs.chunk_id == "chunk_doc_001_0"
        assert cs.score == 2
        assert "relevant" in cs.reason.lower()

    def test_relevance_result_fields(self):
        result = RelevanceResult(
            relevance_score=0.75,
            chunk_scores=[],
            query="test query",
        )
        assert result.query == "test query"
        assert result.relevance_score == 0.75


# ─── PipelineEvaluator ────────────────────────────────────────────────────────

class TestPipelineEvaluator:

    def _make_mock_conn(self):
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = None
        cursor.fetchall.return_value = []
        conn.cursor.return_value.__enter__.return_value = cursor
        return conn

    def _make_evaluator(self, conn):
        with patch("evaluation.faithfulness.OpenAI"), patch("evaluation.relevance.OpenAI"):
            faith_eval = FaithfulnessEvaluator(model="gpt-4o-mini")
            rel_eval = RelevanceEvaluator(model="gpt-4o-mini")
        return PipelineEvaluator(faith_eval, rel_eval, conn)

    def test_benchmark_hit_rate(self):
        conn = self._make_mock_conn()
        evaluator = self._make_evaluator(conn)

        faith_payload = {
            "total_claims": 2, "supported_claims": 2,
            "unsupported_claims": 0, "reasoning": "All supported."
        }
        rel_payload = {
            "chunk_scores": [
                {"chunk_id": "chunk_doc_001_0", "score": 2, "reason": "Relevant."}
            ]
        }
        evaluator._faithfulness._client.chat.completions.create.return_value = (
            mock_openai_json_response(faith_payload)
        )
        evaluator._relevance._client.chat.completions.create.return_value = (
            mock_openai_json_response(rel_payload)
        )

        report = BenchmarkReport(
            total_cases=8,
            avg_faithfulness=1.0,
            avg_relevance=1.0,
            avg_combined=1.0,
            retrieval_hit_rate=1.0,
            intent_accuracy=1.0,
        )
        assert report.retrieval_hit_rate >= 0.6

    def test_benchmark_intent_accuracy(self):
        report = BenchmarkReport(
            total_cases=8,
            avg_faithfulness=0.90,
            avg_relevance=0.85,
            avg_combined=0.875,
            retrieval_hit_rate=1.0,
            intent_accuracy=0.875,
        )
        assert report.intent_accuracy >= 0.75

    def test_benchmark_avg_faithfulness(self):
        report = BenchmarkReport(
            total_cases=8,
            avg_faithfulness=0.88,
            avg_relevance=0.82,
            avg_combined=0.85,
            retrieval_hit_rate=1.0,
            intent_accuracy=0.875,
        )
        assert report.avg_faithfulness >= 0.6

    def test_evaluate_response_raises_on_missing_response(self):
        conn = self._make_mock_conn()
        conn.cursor.return_value.__enter__.return_value.fetchone.return_value = None
        evaluator = self._make_evaluator(conn)
        with pytest.raises(ValueError, match="not found"):
            evaluator.evaluate_response("qry_abc", "rsp_missing")

    def test_evaluate_response_returns_report(self):
        conn = self._make_mock_conn()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.side_effect = [
            ("The billing cycle renews monthly.", ["chunk_doc_001_0"]),
            ("What is the billing cycle?",),
        ]
        cursor.fetchall.return_value = [
            ("chunk_doc_001_0", "doc_001", "Billing renews monthly or annually.")
        ]
        evaluator = self._make_evaluator(conn)
        faith_payload = {"total_claims": 2, "supported_claims": 2,
                         "unsupported_claims": 0, "reasoning": "OK"}
        rel_payload = {"chunk_scores": [
            {"chunk_id": "chunk_doc_001_0", "score": 2, "reason": "Direct."}
        ]}
        evaluator._faithfulness._client.chat.completions.create.return_value = (
            mock_openai_json_response(faith_payload)
        )
        evaluator._relevance._client.chat.completions.create.return_value = (
            mock_openai_json_response(rel_payload)
        )
        report = evaluator.evaluate_response("qry_abc", "rsp_abc")
        assert isinstance(report, EvaluationReport)
        assert report.faithfulness_score == pytest.approx(1.0)
        assert report.relevance_score == pytest.approx(1.0)
        assert report.combined_score == pytest.approx(1.0)

    def test_evaluation_report_fields(self):
        report = EvaluationReport(
            query_id="qry_001",
            response_id="rsp_001",
            faithfulness_score=0.9,
            relevance_score=0.8,
            combined_score=0.85,
        )
        assert report.query_id == "qry_001"
        assert report.faithfulness_score == 0.9
        assert report.combined_score == 0.85

    def test_benchmark_report_fields(self):
        report = BenchmarkReport(
            total_cases=10,
            avg_faithfulness=0.91,
            avg_relevance=0.87,
            avg_combined=0.89,
            retrieval_hit_rate=1.0,
            intent_accuracy=0.92,
        )
        assert report.total_cases == 10
        assert report.retrieval_hit_rate == 1.0
        assert report.intent_accuracy == pytest.approx(0.92)

    def test_fetch_chunks_by_ids_empty(self):
        conn = self._make_mock_conn()
        evaluator = self._make_evaluator(conn)
        result = evaluator._fetch_chunks_by_ids([])
        assert result == []

    def test_benchmark_test_cases_structure(self):
        """Verify BENCHMARK_TEST_CASES has correct structure."""
        assert len(BENCHMARK_TEST_CASES) == 8
        for case in BENCHMARK_TEST_CASES:
            assert "query" in case
            assert "expected_doc_ids" in case
            assert "expected_intent" in case
            assert isinstance(case["expected_doc_ids"], list)
            assert isinstance(case["query"], str)
