from typing import Optional
from pydantic import BaseModel, Field


# ─── Ingestion ───────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    documents: list[dict] = Field(
        ...,
        description="List of raw document dicts. Each must contain doc_id, title, content.",
    )


class IngestResponse(BaseModel):
    documents_saved: int
    chunks_created: int
    chunks_embedded: int


# ─── Query / Answer ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The customer's support question.")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to retrieve.")


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    score: float
    retrieval_method: str


class QueryResponse(BaseModel):
    query_id: str
    response_id: str
    response_text: str
    intent: str
    intent_confidence: float
    retrieved_chunks: list[RetrievedChunkOut]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# ─── Evaluation ──────────────────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    query_id: str
    response_id: str


class EvaluateResponse(BaseModel):
    query_id: str
    response_id: str
    faithfulness_score: float
    relevance_score: float
    combined_score: float


class BenchmarkRequest(BaseModel):
    test_cases: list[dict] = Field(
        ...,
        description=(
            "List of test cases. Each must have: query (str), "
            "expected_doc_ids (list[str]), expected_intent (str)."
        ),
    )


class BenchmarkResponse(BaseModel):
    total_cases: int
    avg_faithfulness: float
    avg_relevance: float
    avg_combined: float
    retrieval_hit_rate: float
    intent_accuracy: float


# ─── Feedback ─────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    response_id: str
    rating: int = Field(..., ge=1, le=5, description="Rating between 1 (worst) and 5 (best).")
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    response_id: str
    rating: int
    comment: Optional[str]


class FeedbackSummaryResponse(BaseModel):
    response_id: str
    avg_rating: float
    total_count: int


# ─── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    database: str
    version: str
