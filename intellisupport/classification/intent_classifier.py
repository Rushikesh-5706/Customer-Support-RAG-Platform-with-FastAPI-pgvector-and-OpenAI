import json
import logging
from pydantic import BaseModel
from openai import OpenAI
from config import settings

logger = logging.getLogger(__name__)

VALID_INTENTS = frozenset({
    "billing",
    "technical_issue",
    "feature_request",
    "integration",
    "account_management",
    "data_and_export",
    "general_inquiry",
})

_INTENT_DESCRIPTIONS = {
    "billing": (
        "Questions about pricing, subscription plans, invoices, payment methods, "
        "refunds, plan upgrades, downgrades, or cancellation."
    ),
    "technical_issue": (
        "Bug reports, error messages, crashes, unexpected application behavior, "
        "features not working, or requests for technical troubleshooting."
    ),
    "feature_request": (
        "Questions about existing features, platform capabilities, "
        "product roadmap, or how to accomplish a task the user cannot find."
    ),
    "integration": (
        "Questions about connecting Nexora to third-party tools such as Slack, "
        "GitHub, Jira, or any external service."
    ),
    "account_management": (
        "Login issues, password resets, two-factor authentication, team member "
        "invitations, role and permission changes, or account security settings."
    ),
    "data_and_export": (
        "Data exports, backups, CSV or JSON downloads, data deletion, "
        "GDPR data requests, or data portability."
    ),
    "general_inquiry": (
        "Any question that does not clearly fit the above categories."
    ),
}


class IntentResult(BaseModel):
    intent: str
    confidence: float


class IntentClassifier:

    def __init__(self, model: str = "gpt-4o-mini"):
        self._model = model
        self._client = OpenAI(api_key=settings.openai_api_key)

    def _build_messages(self, query: str) -> list[dict]:
        intent_block = "\n".join(
            f'  "{label}": {desc}'
            for label, desc in _INTENT_DESCRIPTIONS.items()
        )
        system_content = (
            "You are an intent classification engine for a B2B SaaS customer support platform.\n"
            "Classify the customer query into exactly one of these intent labels:\n\n"
            f"{intent_block}\n\n"
            "Respond with a JSON object containing exactly two keys:\n"
            '  "intent": one of the label strings above\n'
            '  "confidence": float between 0.0 and 1.0\n\n'
            "No preamble. No explanation. Only the JSON object."
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Customer query: {query}"},
        ]

    def classify(self, query: str) -> IntentResult:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=self._build_messages(query),
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=80,
            )
            parsed = json.loads(resp.choices[0].message.content)
            intent = parsed.get("intent", "general_inquiry")
            confidence = float(parsed.get("confidence", 0.0))

            if intent not in VALID_INTENTS:
                logger.warning("Unknown intent '%s' returned. Defaulting.", intent)
                intent = "general_inquiry"
                confidence = 0.0

            return IntentResult(
                intent=intent,
                confidence=max(0.0, min(1.0, confidence)),
            )
        except Exception as exc:
            logger.error("Intent classification failed: %s", exc)
            return IntentResult(intent="general_inquiry", confidence=0.0)

    def classify_batch(self, queries: list[str]) -> list[IntentResult]:
        return [self.classify(q) for q in queries]
