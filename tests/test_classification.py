"""
Tests for the classification module: IntentClassifier, IntentResult.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from classification.intent_classifier import (
    IntentClassifier,
    IntentResult,
    VALID_INTENTS,
)


class TestIntentClassifier:

    def setup_method(self):
        with patch("classification.intent_classifier.OpenAI"):
            self.classifier = IntentClassifier(model="gpt-4o-mini")

    def _mock_response(self, intent: str, confidence: float):
        content = json.dumps({"intent": intent, "confidence": confidence})
        choice = MagicMock()
        choice.message.content = content
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_classify_billing_query(self):
        self.classifier._client.chat.completions.create.return_value = (
            self._mock_response("billing", 0.95)
        )
        result = self.classifier.classify("How do I upgrade my subscription plan?")
        assert isinstance(result, IntentResult)
        assert result.intent == "billing"
        assert result.confidence == pytest.approx(0.95)

    def test_classify_technical_issue(self):
        self.classifier._client.chat.completions.create.return_value = (
            self._mock_response("technical_issue", 0.88)
        )
        result = self.classifier.classify("I'm getting a 401 error on every API call.")
        assert result.intent == "technical_issue"

    def test_classify_all_valid_intents(self):
        for intent in VALID_INTENTS:
            self.classifier._client.chat.completions.create.return_value = (
                self._mock_response(intent, 0.9)
            )
            result = self.classifier.classify("sample query")
            assert result.intent == intent

    def test_classify_unknown_intent_falls_back(self):
        self.classifier._client.chat.completions.create.return_value = (
            self._mock_response("completely_unknown_intent", 0.7)
        )
        result = self.classifier.classify("weird query")
        assert result.intent == "general_inquiry"
        assert result.confidence == 0.0

    def test_classify_api_failure_returns_default(self):
        self.classifier._client.chat.completions.create.side_effect = Exception("Network error")
        result = self.classifier.classify("any query")
        assert result.intent == "general_inquiry"
        assert result.confidence == 0.0

    def test_confidence_clamped_to_one(self):
        self.classifier._client.chat.completions.create.return_value = (
            self._mock_response("billing", 1.5)
        )
        result = self.classifier.classify("query")
        assert result.confidence <= 1.0

    def test_confidence_clamped_to_zero(self):
        self.classifier._client.chat.completions.create.return_value = (
            self._mock_response("billing", -0.5)
        )
        result = self.classifier.classify("query")
        assert result.confidence >= 0.0

    def test_classify_batch(self):
        self.classifier._client.chat.completions.create.return_value = (
            self._mock_response("billing", 0.9)
        )
        queries = ["query 1", "query 2", "query 3"]
        results = self.classifier.classify_batch(queries)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, IntentResult)

    def test_build_messages_contains_all_intents(self):
        messages = self.classifier._build_messages("test query")
        system_msg = messages[0]["content"]
        for intent in VALID_INTENTS:
            assert intent in system_msg

    def test_intent_result_fields(self):
        result = IntentResult(intent="billing", confidence=0.75)
        assert result.intent == "billing"
        assert result.confidence == 0.75
