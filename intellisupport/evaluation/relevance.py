import json
import logging
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
        self._client = OpenAI(api_key=settings.openai_api_key)

    def evaluate(
        self, query: str, retrieved_chunks: list[RetrievedChunk]
    ) -> RelevanceResult:
        if not retrieved_chunks:
            return RelevanceResult(relevance_score=0.0, chunk_scores=[], query=query)

        chunks_text = "\n\n".join(
            f"chunk_id: {ch.chunk_id}\ncontent: {ch.content}"
            for ch in retrieved_chunks
        )

        system_content = (
            "You are an impartial AI evaluation judge assessing retrieval quality.\n"
            "For each retrieved chunk, rate its relevance to answering the query:\n\n"
            "  0 = Not relevant: chunk does not address the query.\n"
            "  1 = Partially relevant: chunk is topically related but does not directly answer.\n"
            "  2 = Highly relevant: chunk directly and substantially addresses the query.\n\n"
            "Respond with JSON: "
            '{"chunk_scores": [{"chunk_id": "...", "score": 0|1|2, "reason": "..."}, ...]}\n'
            "Score every chunk listed. No text outside the JSON."
        )

        user_content = (
            f"QUERY: {query}\n\nRETRIEVED CHUNKS:\n{chunks_text}"
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
                max_tokens=1024,
            )
            parsed = json.loads(api_resp.choices[0].message.content)
            raw = parsed.get("chunk_scores", [])
            chunk_scores = [
                ChunkRelevanceScore(
                    chunk_id=item["chunk_id"],
                    score=int(item["score"]),
                    reason=item.get("reason", ""),
                )
                for item in raw
            ]
            total_possible = 2 * len(retrieved_chunks)
            total_achieved = sum(cs.score for cs in chunk_scores)
            rel_score = (total_achieved / total_possible) if total_possible > 0 else 0.0
            rel_score = max(0.0, min(1.0, rel_score))

            return RelevanceResult(
                relevance_score=rel_score,
                chunk_scores=chunk_scores,
                query=query,
            )
        except Exception as exc:
            logger.error("Relevance evaluation error: %s", exc)
            return RelevanceResult(relevance_score=0.0, chunk_scores=[], query=query)

    def evaluate_batch(
        self, queries_and_chunks: list[tuple[str, list[RetrievedChunk]]]
    ) -> list[RelevanceResult]:
        return [self.evaluate(q, chunks) for q, chunks in queries_and_chunks]
