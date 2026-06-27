"""
Tests for the generation module: IntentClassifier, PromptBuilder, ResponseGenerator.
All spec-required test function names are included.
Note: classify_* tests live here because the spec lists only 4 test files
(test_ingestion, test_retrieval, test_generation, test_evaluation).
"""

import pytest
from unittest.mock import MagicMock, patch
from retrieval.vector_store import RetrievedChunk
from classification.intent_classifier import IntentClassifier, IntentResult, VALID_INTENTS
from generation.prompt_builder import PromptBuilder
from generation.response_generator import ResponseGenerator, GeneratedResponse


def make_chunk(chunk_id: str = "chunk_doc_001_0", doc_id: str = "doc_001",
               content: str = "Context content here.") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content,
        score=0.9,
        retrieval_method="hybrid",
    )


def make_intent(intent: str = "billing", confidence: float = 0.9) -> IntentResult:
    return IntentResult(intent=intent, confidence=confidence)


# ─── IntentClassifier ────────────────────────────────────────────────────────

class TestIntentClassifier:

    def setup_method(self):
        with patch("classification.intent_classifier.OpenAI"):
            self.classifier = IntentClassifier(model="gpt-4o-mini")

    def _mock_response(self, intent: str, confidence: float):
        import json
        choice = MagicMock()
        choice.message.content = json.dumps({"intent": intent, "confidence": confidence})
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    # SPEC-REQUIRED: exact name
    def test_classify_billing_intent(self):
        """Billing queries must be classified as 'billing'."""
        self.classifier._client.chat.completions.create.return_value = (
            self._mock_response("billing", 0.95)
        )
        result = self.classifier.classify("How do I upgrade my subscription plan?")
        assert isinstance(result, IntentResult)
        assert result.intent == "billing"
        assert result.confidence == pytest.approx(0.95)

    # SPEC-REQUIRED: exact name
    def test_classify_technical_intent(self):
        """Technical error queries must be classified as 'technical_issue'."""
        self.classifier._client.chat.completions.create.return_value = (
            self._mock_response("technical_issue", 0.92)
        )
        result = self.classifier.classify("I'm getting a 401 Unauthorized error from the API.")
        assert isinstance(result, IntentResult)
        assert result.intent == "technical_issue"
        assert result.confidence == pytest.approx(0.92)

    # SPEC-REQUIRED: exact name
    def test_classify_confidence_range(self):
        """Confidence must always be in [0.0, 1.0]."""
        self.classifier._client.chat.completions.create.return_value = (
            self._mock_response("billing", 0.87)
        )
        result = self.classifier.classify("What is the refund policy?")
        assert 0.0 <= result.confidence <= 1.0

    def test_classify_billing_query(self):
        self.classifier._client.chat.completions.create.return_value = (
            self._mock_response("billing", 0.95)
        )
        result = self.classifier.classify("How do I upgrade my subscription plan?")
        assert result.intent == "billing"

    def test_classify_all_valid_intents(self):
        valid_intents = [
            "billing", "technical_issue", "feature_request",
            "integration", "account_management", "data_and_export", "general_inquiry",
        ]
        for intent in valid_intents:
            self.classifier._client.chat.completions.create.return_value = (
                self._mock_response(intent, 0.9)
            )
            result = self.classifier.classify(f"query about {intent}")
            assert result.intent == intent

    def test_classify_unknown_intent_falls_back(self):
        self.classifier._client.chat.completions.create.return_value = (
            self._mock_response("completely_unknown_intent", 0.5)
        )
        result = self.classifier.classify("some query")
        assert result.intent in VALID_INTENTS
        assert result.confidence >= 0.0

    def test_classify_api_failure_returns_default(self):
        self.classifier._client.chat.completions.create.side_effect = Exception("API down")
        result = self.classifier.classify("any query")
        assert result.intent in VALID_INTENTS
        assert result.confidence >= 0.0

    def test_confidence_clamped_to_one(self):
        self.classifier._client.chat.completions.create.return_value = (
            self._mock_response("billing", 1.5)
        )
        result = self.classifier.classify("query")
        assert result.confidence <= 1.0


# ─── PromptBuilder ────────────────────────────────────────────────────────────

class TestPromptBuilder:

    def setup_method(self):
        self.builder = PromptBuilder()

    # SPEC-REQUIRED: exact name
    def test_build_rag_prompt_structure(self):
        """RAG prompt must return exactly [system_msg, user_msg]."""
        messages = self.builder.build_rag_prompt(
            query="What is the refund policy?",
            retrieved_chunks=[make_chunk()],
            intent=make_intent("billing"),
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    # SPEC-REQUIRED: exact name
    def test_prompt_contains_chunk_ids(self):
        """User message must contain chunk_ids so the model can cite sources."""
        chunk = make_chunk(chunk_id="chunk_doc_001_0")
        messages = self.builder.build_rag_prompt("query", [chunk], make_intent())
        user_msg = messages[1]["content"]
        assert "chunk_doc_001_0" in user_msg

    # SPEC-REQUIRED: exact name
    def test_generate_response_fields(self):
        """GeneratedResponse must have response_text, model, prompt_tokens,
        completion_tokens, total_tokens."""
        result = GeneratedResponse(
            response_text="The refund policy is 30 days.",
            model="gpt-4o-mini",
            prompt_tokens=100,
            completion_tokens=40,
            total_tokens=140,
        )
        assert result.response_text == "The refund policy is 30 days."
        assert result.model == "gpt-4o-mini"
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 40
        assert result.total_tokens == 140

    def test_build_rag_prompt_returns_two_messages(self):
        messages = self.builder.build_rag_prompt(
            query="What is the refund policy?",
            retrieved_chunks=[make_chunk()],
            intent=make_intent("billing"),
        )
        assert len(messages) == 2

    def test_build_rag_prompt_includes_chunk_content(self):
        chunk = make_chunk(content="Refunds are available within 30 days.")
        messages = self.builder.build_rag_prompt("refund?", [chunk], make_intent())
        user_msg = messages[1]["content"]
        assert "Refunds are available within 30 days." in user_msg

    def test_build_rag_prompt_includes_chunk_ids(self):
        chunk = make_chunk(chunk_id="chunk_doc_001_0")
        messages = self.builder.build_rag_prompt("query", [chunk], make_intent())
        user_msg = messages[1]["content"]
        assert "chunk_doc_001_0" in user_msg

    def test_build_rag_prompt_includes_intent(self):
        messages = self.builder.build_rag_prompt("query", [make_chunk()], make_intent("technical_issue"))
        system_msg = messages[0]["content"]
        assert "technical_issue" in system_msg

    def test_build_rag_prompt_multiple_chunks(self):
        chunks = [make_chunk(f"chunk_doc_001_{i}") for i in range(3)]
        messages = self.builder.build_rag_prompt("query", chunks, make_intent())
        user_msg = messages[1]["content"]
        assert "CHUNK 1" in user_msg
        assert "CHUNK 2" in user_msg
        assert "CHUNK 3" in user_msg

    def test_build_clarification_prompt_returns_two_messages(self):
        messages = self.builder.build_clarification_prompt("obscure query", make_intent())
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_build_clarification_prompt_includes_query(self):
        messages = self.builder.build_clarification_prompt("mysterious problem", make_intent())
        user_msg = messages[1]["content"]
        assert "mysterious problem" in user_msg

    def test_estimate_prompt_tokens_positive(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the refund policy?"},
        ]
        tokens = self.builder.estimate_prompt_tokens(messages)
        assert tokens > 0

    def test_tone_billing_in_system_message(self):
        messages = self.builder.build_rag_prompt("query", [make_chunk()], make_intent("billing"))
        assert "pricing" in messages[0]["content"].lower() or "billing" in messages[0]["content"].lower()


# ─── ResponseGenerator ────────────────────────────────────────────────────────

class TestResponseGenerator:

    def setup_method(self):
        with patch("generation.response_generator.OpenAI"):
            self.generator = ResponseGenerator(model="gpt-4o-mini", max_tokens=256, temperature=0.1)

    def _mock_api_response(self, text: str, prompt_tokens: int = 100,
                            completion_tokens: int = 50, model: str = "gpt-4o-mini"):
        choice = MagicMock()
        choice.message.content = text
        resp = MagicMock()
        resp.choices = [choice]
        resp.model = model
        resp.usage.prompt_tokens = prompt_tokens
        resp.usage.completion_tokens = completion_tokens
        resp.usage.total_tokens = prompt_tokens + completion_tokens
        return resp

    def test_generate_returns_generated_response(self):
        self.generator._client.chat.completions.create.return_value = (
            self._mock_api_response("The refund policy is 30 days.")
        )
        messages = [{"role": "user", "content": "refund policy?"}]
        result = self.generator.generate(messages)
        assert isinstance(result, GeneratedResponse)
        assert result.response_text == "The refund policy is 30 days."
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150

    def test_generate_with_fallback_uses_primary_when_good(self):
        self.generator._client.chat.completions.create.return_value = (
            self._mock_api_response("A comprehensive and detailed answer about billing.")
        )
        messages = [{"role": "user", "content": "q"}]
        result = self.generator.generate_with_fallback(messages, messages)
        assert "comprehensive" in result.response_text

    def test_generate_with_fallback_uses_fallback_when_primary_too_short(self):
        short = self._mock_api_response("No.")
        long_text = "A much longer fallback response explaining the issue in great detail with context."
        long = self._mock_api_response(long_text)
        self.generator._client.chat.completions.create.side_effect = [short, long]
        primary = [{"role": "user", "content": "short"}]
        fallback = [{"role": "user", "content": "fallback"}]
        result = self.generator.generate_with_fallback(primary, fallback)
        assert result.response_text == long_text

    def test_generated_response_fields(self):
        result = GeneratedResponse(
            response_text="Answer.",
            model="gpt-4o-mini",
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
        )
        assert result.total_tokens == 70
