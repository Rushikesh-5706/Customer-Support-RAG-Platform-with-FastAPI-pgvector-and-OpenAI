import json
import logging
from typing import Optional
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
        self._client: Optional[OpenAI] = (
            OpenAI(api_key=settings.openai_api_key)
            if settings.openai_api_key
            else None
        )

    def _token_overlap_fallback(
        self, response_text: str, retrieved_chunks: list[RetrievedChunk]
    ) -> FaithfulnessResult:
        if not response_text.strip():
            return FaithfulnessResult(
                faithfulness_score=1.0, total_claims=0,
                supported_claims=0, unsupported_claims=0,
                reasoning="Empty response.",
            )
        context_words = set(
            " ".join(ch.content for ch in retrieved_chunks).lower().split()
        )
        sentences = [
            s.strip() for s in response_text.replace("\n", " ").split(".")
            if len(s.strip()) > 15
        ]
        if not sentences:
            return FaithfulnessResult(
                faithfulness_score=1.0, total_claims=0,
                supported_claims=0, unsupported_claims=0,
                reasoning="No evaluable claims.",
            )
        supported = 0
        for sent in sentences:
            words = set(sent.lower().split())
            if words and len(words & context_words) / len(words) >= 0.28:
                supported += 1
        total = len(sentences)
        score = round(max(0.0, min(1.0, supported / total)), 4)
        return FaithfulnessResult(
            faithfulness_score=score,
            total_claims=total,
            supported_claims=supported,
            unsupported_claims=total - supported,
            reasoning="Token-overlap fallback scoring.",
        )

    def evaluate(
        self,
        response_text: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> FaithfulnessResult:
        if not response_text or not response_text.strip():
            return FaithfulnessResult(
                faithfulness_score=1.0, total_claims=0,
                supported_claims=0, unsupported_claims=0,
                reasoning="Empty response — treated as fully faithful.",
            )
        if not self._client:
            return self._token_overlap_fallback(response_text, retrieved_chunks)

        context = "\n\n".join(
            f"[doc_id={ch.doc_id} | chunk_id={ch.chunk_id}]\n{ch.content}"
            for ch in retrieved_chunks
        )
        system_content = (
            "You are a calibrated faithfulness judge for a retrieval-augmented generation system.\n\n"
            "TASK: Determine whether claims in the generated response are supported by the context.\n\n"
            "DEFINITIONS:\n"
            "  CLAIM: A specific factual assertion. Do NOT treat as claims:\n"
            "    - Source citations like '[Sources: chunk_doc_001_0]'\n"
            "    - Phrases like 'Based on the documentation' or 'According to Nexora'\n"
            "    - Polite phrases, greetings, meta-commentary, offers to help\n"
            "    - Clarifying questions directed at the user\n\n"
            "  SUPPORTED: The claim is directly stated in the context, OR is a reasonable "
            "paraphrase or reformulation of facts that ARE stated. Paraphrasing counts as "
            "supported. Step-by-step inferences from stated procedures count as supported.\n\n"
            "  UNSUPPORTED: The claim introduces a fact entirely absent from the context, "
            "contradicts the context, or requires outside knowledge not present.\n\n"
            "SPECIAL CASE: If the response says 'I don't have information about that' or "
            "is a clarifying question only, return total_claims=0, faithfulness_score=1.0.\n\n"
            "Return ONLY a JSON object with exactly these keys: "
            "total_claims (int), supported_claims (int), unsupported_claims (int), "
            "reasoning (string). No text outside the JSON."
        )
        user_content = (
            f"CONTEXT DOCUMENTS:\n{context}\n\n"
            f"GENERATED RESPONSE TO EVALUATE:\n{response_text}"
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
                max_tokens=600,
            )
            parsed = json.loads(api_resp.choices[0].message.content)
            total = int(parsed.get("total_claims", 0))
            supported = int(parsed.get("supported_claims", 0))
            unsupported = int(parsed.get("unsupported_claims", max(0, total - supported)))
            score = (supported / total) if total > 0 else 1.0
            return FaithfulnessResult(
                faithfulness_score=round(max(0.0, min(1.0, score)), 4),
                total_claims=total,
                supported_claims=supported,
                unsupported_claims=unsupported,
                reasoning=str(parsed.get("reasoning", "")),
            )
        except Exception as exc:
            logger.warning("LLM faithfulness eval failed: %s. Using fallback.", exc)
            return self._token_overlap_fallback(response_text, retrieved_chunks)
