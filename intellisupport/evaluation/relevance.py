import json
import logging
from typing import Optional
from pydantic import BaseModel
from openai import OpenAI
from retrieval.vector_store import RetrievedChunk
from config import settings

logger = logging.getLogger(__name__)


class ChunkRelevanceScore(BaseModel):
    chunk_id: str
    score: int
    reason: str


class RelevanceResult(BaseModel):
    relevance_score: float
    chunk_scores: list[ChunkRelevanceScore]
    query: str


class RelevanceEvaluator:

    def __init__(self, model: str = "gpt-4o-mini"):
        self._model = model
        self._client: Optional[OpenAI] = (
            OpenAI(api_key=settings.openai_api_key)
            if settings.openai_api_key
            else None
        )

    def _lexical_fallback(
        self, query: str, retrieved_chunks: list[RetrievedChunk]
    ) -> RelevanceResult:
        query_tokens = set(query.lower().split())
        chunk_scores: list[ChunkRelevanceScore] = []
        total = 0
        for chunk in retrieved_chunks:
            chunk_tokens = set(chunk.content.lower().split())
            if not query_tokens or not chunk_tokens:
                score = 0
                overlap = 0.0
            else:
                overlap = len(query_tokens & chunk_tokens) / max(1, len(query_tokens))
                score = 2 if overlap >= 0.28 else (1 if overlap >= 0.10 else 0)
            total += score
            chunk_scores.append(ChunkRelevanceScore(
                chunk_id=chunk.chunk_id, score=score,
                reason=f"Lexical overlap fallback: {round(overlap, 3)}",
            ))
        rel_score = (total / (2 * len(retrieved_chunks))) if retrieved_chunks else 0.0
        return RelevanceResult(
            relevance_score=round(max(0.0, min(1.0, rel_score)), 4),
            chunk_scores=chunk_scores,
            query=query,
        )

    def evaluate(
        self, query: str, retrieved_chunks: list[RetrievedChunk]
    ) -> RelevanceResult:
        if not retrieved_chunks:
            return RelevanceResult(relevance_score=0.0, chunk_scores=[], query=query)
        if not self._client:
            return self._lexical_fallback(query, retrieved_chunks)

        system_content = (
            "You are a calibrated retrieval quality judge for a customer support RAG system.\n\n"
            "TASK: For each retrieved chunk, rate relevance to the customer query:\n\n"
            "  2 = HIGHLY RELEVANT: The chunk directly addresses the query topic and contains "
            "information that would help answer it. If the query is about topic X and the chunk "
            "is about topic X with useful details, it scores 2.\n\n"
            "  1 = PARTIALLY RELEVANT: The chunk is about a related area but does not directly "
            "answer the specific question asked.\n\n"
            "  0 = NOT RELEVANT: Completely different topic with no connection to the query.\n\n"
            "CALIBRATION:\n"
            "  - A chunk about two-factor authentication is highly relevant to a 2FA question: score 2.\n"
            "  - A chunk about billing plans is highly relevant to a subscription query: score 2.\n"
            "  - A chunk about Slack integration is highly relevant to a Slack connection query: score 2.\n"
            "  - Err toward 2 when topic match is clear, even if the chunk doesn't give a complete answer.\n"
            "  - Score every chunk. Do not skip any.\n\n"
            "Return ONLY JSON with key 'chunk_scores' containing objects with "
            "'chunk_id' (string), 'score' (0|1|2), 'reason' (string). "
            "No text outside the JSON."
        )
        chunks_payload = [
            {"chunk_id": ch.chunk_id, "doc_id": ch.doc_id, "content": ch.content[:800]}
            for ch in retrieved_chunks
        ]
        user_content = json.dumps({"query": query, "retrieved_chunks": chunks_payload})

        try:
            api_resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=1200,
            )
            parsed = json.loads(api_resp.choices[0].message.content)
            raw = parsed.get("chunk_scores", [])
            chunk_scores = []
            total = 0
            for item in raw:
                score = max(0, min(2, int(item.get("score", 0))))
                total += score
                chunk_scores.append(ChunkRelevanceScore(
                    chunk_id=str(item.get("chunk_id", "")),
                    score=score,
                    reason=str(item.get("reason", "")),
                ))
            rel_score = (total / (2 * len(chunk_scores))) if chunk_scores else 0.0
            return RelevanceResult(
                relevance_score=round(max(0.0, min(1.0, rel_score)), 4),
                chunk_scores=chunk_scores,
                query=query,
            )
        except Exception as exc:
            logger.warning("LLM relevance eval failed: %s. Using fallback.", exc)
            return self._lexical_fallback(query, retrieved_chunks)

    def evaluate_batch(
        self, queries_and_chunks: list[tuple[str, list[RetrievedChunk]]]
    ) -> list[RelevanceResult]:
        return [self.evaluate(q, chunks) for q, chunks in queries_and_chunks]
