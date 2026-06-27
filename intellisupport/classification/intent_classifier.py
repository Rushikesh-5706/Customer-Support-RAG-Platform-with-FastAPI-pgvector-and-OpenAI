import json
import logging
from typing import Optional
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
        "features not working, service outages, or requests for technical troubleshooting."
    ),
    "feature_request": (
        "Questions about existing features, platform capabilities, "
        "product roadmap, or how to accomplish a task the user cannot find."
    ),
    "integration": (
        "Questions about connecting Nexora to third-party tools such as Slack, "
        "GitHub, Jira, or any external service or API."
    ),
    "account_management": (
        "Login issues, password resets, two-factor authentication, adding or removing "
        "team members, invitations, role and permission changes, or account security settings."
    ),
    "data_and_export": (
        "Data exports, backups, CSV or JSON downloads, data deletion, "
        "GDPR data requests, or data portability."
    ),
    "general_inquiry": (
        "Any question that does not clearly fit the above categories."
    ),
}

_HEURISTIC_KEYWORDS: dict[str, list[str]] = {
    "billing": [
        "billing", "invoice", "invoices", "pricing", "plan", "plans",
        "subscription", "refund", "cancel", "upgrade", "downgrade",
        "payment", "charge", "receipt", "cost", "price",
    ],
    "technical_issue": [
        "error", "crash", "crashing", "bug", "broken", "issue", "fails",
        "failing", "not receiving", "not working", "problem", "troubleshoot",
        "err_", "500", "502", "timeout", "outage",
    ],
    "feature_request": [
        "feature", "roadmap", "request", "can i use", "template", "templates",
        "custom", "workflow", "automat",
    ],
    "integration": [
        "integration", "integrate", "slack", "github", "jira",
        "connect to", "webhook", "webhooks", "connect nexora", "third-party",
    ],
    "account_management": [
        "login", "log in", "password", "two-factor", "2fa", "mfa",
        "team member", "add a new member", "new member", "add member",
        "invite", "permission", "permissions", "account", "authentication",
        "authenticate", "sign in", "sign-in", "forgot", "reset", "member to my team",
    ],
    "data_and_export": [
        "export", "backup", "csv", "data export", "download", "restore", "gdpr",
    ],
}


class IntentResult(BaseModel):
    intent: str
    confidence: float


class IntentClassifier:

    def __init__(self, model: str = "gpt-4o-mini"):
        self._model = model
        self._client: Optional[OpenAI] = (
            OpenAI(api_key=settings.openai_api_key)
            if settings.openai_api_key
            else None
        )

    def _heuristic_classify(self, query: str) -> IntentResult:
        text = query.lower()
        scores: dict[str, int] = {
            intent: 0 for intent in VALID_INTENTS if intent != "general_inquiry"
        }
        for intent, keywords in _HEURISTIC_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    scores[intent] += 1

        if "webhook" in text and ("not receiving" in text or "isn't" in text or "is not" in text):
            scores["technical_issue"] += 3
        if "forgot" in text and ("password" in text or "log in" in text or "login" in text):
            scores["account_management"] += 4
        if ("add" in text or "new" in text) and ("member" in text or "team" in text):
            scores["account_management"] += 3
        if "cancel" in text and "subscription" in text:
            scores["billing"] += 3
        if "export" in text and ("csv" in text or "data" in text):
            scores["data_and_export"] += 3

        if all(v == 0 for v in scores.values()):
            return IntentResult(intent="general_inquiry", confidence=0.5)

        best = max(scores, key=scores.__getitem__)
        total = sum(scores.values()) or 1
        confidence = round(min(1.0, max(0.5, scores[best] / total + 0.35)), 3)
        return IntentResult(intent=best, confidence=confidence)

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
        if not self._client:
            return self._heuristic_classify(query)
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
                logger.warning("Unknown intent '%s'. Using heuristic fallback.", intent)
                return self._heuristic_classify(query)
            return IntentResult(
                intent=intent,
                confidence=round(max(0.0, min(1.0, confidence)), 3),
            )
        except Exception as exc:
            logger.warning("LLM classify failed: %s. Using heuristic fallback.", exc)
            return self._heuristic_classify(query)

    def classify_batch(self, queries: list[str]) -> list[IntentResult]:
        return [self.classify(q) for q in queries]
