import json
import logging
from pydantic import BaseModel
from openai import OpenAI
from retrieval.vector_store import RetrievedChunk
from config import settings

logger = logging.getLogger(__name__)


class FaithfulnessResult(BaseModel):
    faithfulness_score: float
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    reasoning: str


class FaithfulnessEvaluator:

    def __init__(self, model: str = "gpt-4o-mini"):
        self._model = model
        self._client = OpenAI(api_key=settings.openai_api_key)

    def evaluate(
        self,
        response_text: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> FaithfulnessResult:
        context = "\n\n".join(
            f"[{ch.chunk_id}]: {ch.content}" for ch in retrieved_chunks
        )

        system_content = (
            "You are an impartial AI evaluation judge. Your task is to measure the "
            "factual faithfulness of a generated response against a set of source context documents.\n\n"
            "Definitions:\n"
            "  CLAIM: Any factual assertion in the response. Ignore greetings, meta-commentary, "
            "  and phrases like 'based on the documentation'.\n"
            "  SUPPORTED: The claim is directly and explicitly verifiable from the context. "
            "  Reasonable inference from stated facts counts as supported.\n"
            "  UNSUPPORTED: The claim is absent from the context, contradicts the context, "
            "  or requires outside knowledge not present in the context.\n\n"
            "Steps:\n"
            "  1. Extract every factual claim from the response.\n"
            "  2. For each claim, determine supported or unsupported using only the context.\n"
            "  3. Count total_claims, supported_claims, and unsupported_claims.\n\n"
            "Respond with a JSON object with exactly these keys: "
            "total_claims (int), supported_claims (int), unsupported_claims (int), "
            "reasoning (short string). "
            "No text outside the JSON object."
        )

        user_content = (
            f"CONTEXT DOCUMENTS:\n{context}\n\n"
            f"GENERATED RESPONSE:\n{response_text}"
        )

        try:
            api_resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=512,
            )
            parsed = json.loads(api_resp.choices[0].message.content)
            total = int(parsed.get("total_claims", 0))
            supported = int(parsed.get("supported_claims", 0))
            unsupported = int(parsed.get("unsupported_claims", 0))
            score = (supported / total) if total > 0 else 1.0
            score = max(0.0, min(1.0, score))

            return FaithfulnessResult(
                faithfulness_score=score,
                total_claims=total,
                supported_claims=supported,
                unsupported_claims=unsupported,
                reasoning=parsed.get("reasoning", ""),
            )
        except Exception as exc:
            logger.error("Faithfulness evaluation error: %s", exc)
            return FaithfulnessResult(
                faithfulness_score=0.0,
                total_claims=0,
                supported_claims=0,
                unsupported_claims=0,
                reasoning=f"Evaluation failed: {exc}",
            )
