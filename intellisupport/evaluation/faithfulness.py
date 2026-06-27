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

    def _token_overlap_score(
        self, response_text: str, retrieved_chunks: list[RetrievedChunk]
    ) -> FaithfulnessResult:
        if not response_text.strip():
            return FaithfulnessResult(
                faithfulness_score=1.0,
                total_claims=0,
                supported_claims=0,
                unsupported_claims=0,
                reasoning="Empty response — no claims to evaluate.",
            )
        context_words = set(
            " ".join(ch.content for ch in retrieved_chunks).lower().split()
        )
        sentences = [
            s.strip()
            for s in response_text.replace("\n", " ").split(".")
            if len(s.strip()) > 15
        ]
        if not sentences:
            return FaithfulnessResult(
                faithfulness_score=1.0,
                total_claims=0,
                supported_claims=0,
                unsupported_claims=0,
                reasoning="No evaluable claims detected.",
            )
        supported = 0
        for sent in sentences:
            sent_words = set(sent.lower().split())
            if not sent_words:
                supported += 1
                continue
            overlap = len(sent_words & context_words) / len(sent_words)
            if overlap >= 0.30:
                supported += 1
        total = len(sentences)
        score = supported / total if total else 1.0
        return FaithfulnessResult(
            faithfulness_score=round(max(0.0, min(1.0, score)), 4),
            total_claims=total,
            supported_claims=supported,
            unsupported_claims=total - supported,
            reasoning="Fallback token-overlap scoring used.",
        )

    def evaluate(
        self,
        response_text: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> FaithfulnessResult:
        if not response_text or not response_text.strip():
            return FaithfulnessResult(
                faithfulness_score=1.0,
                total_claims=0,
                supported_claims=0,
                unsupported_claims=0,
                reasoning="Empty response — treated as fully faithful.",
            )

        context = "\n\n".join(
            f"[doc_id={ch.doc_id} | chunk_id={ch.chunk_id}]\n{ch.content}"
            for ch in retrieved_chunks
        )

        system_content = (
            "You are a strict but calibrated faithfulness judge for a RAG system.\n\n"
            "TASK: Determine whether the claims in the generated response are supported by "
            "the provided context documents.\n\n"
            "DEFINITIONS:\n"
            "  CLAIM: A factual assertion that can be verified. Do NOT treat as claims:\n"
            "    - Source citations such as '[Sources: chunk_doc_001_0]'\n"
            "    - Phrases like 'Based on the documentation' or 'According to Nexora'\n"
            "    - Polite phrases, greetings, or meta-commentary\n"
            "    - Offers to help or questions to the user\n\n"
            "  SUPPORTED: The claim is directly stated in the context OR is a reasonable "
            "reformulation or inference from facts that ARE stated in the context. "
            "Paraphrasing a fact from the context counts as supported. "
            "A reasonable step-by-step inference from stated steps counts as supported.\n\n"
            "  UNSUPPORTED: The claim introduces a fact not present in the context at all, "
            "contradicts the context, or relies on knowledge outside the context.\n\n"
            "IMPORTANT: If the response says 'I don't have information about that' or is "
            "a clarifying question, return total_claims=0 and faithfulness_score=1.0.\n\n"
            "STEPS:\n"
            "  1. List each factual claim in the response.\n"
            "  2. For each claim, decide: supported or unsupported.\n"
            "  3. Return the counts.\n\n"
            "Return ONLY a JSON object with exactly these keys:\n"
            "  total_claims (int), supported_claims (int), "
            "unsupported_claims (int), reasoning (string).\n"
            "No text outside the JSON."
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

            if total == 0:
                score = 1.0
            else:
                score = supported / total

            return FaithfulnessResult(
                faithfulness_score=round(max(0.0, min(1.0, score)), 4),
                total_claims=total,
                supported_claims=supported,
                unsupported_claims=unsupported,
                reasoning=str(parsed.get("reasoning", "")),
            )
        except Exception as exc:
            logger.warning("LLM faithfulness evaluation failed: %s. Using fallback.", exc)
            return self._token_overlap_score(response_text, retrieved_chunks)
