from typing import Optional
from pydantic import BaseModel, Field


# ─── Query / Answer ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The customer's support question.")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to retrieve.")


class QueryResponse(BaseModel):
    query_id: str
    response_id: str
    response_text: str
    intent: str
    intent_confidence: float
    retrieved_chunk_ids: list[str]           # spec: list of chunk_id strings, NOT full objects
    faithfulness_score: Optional[float]      # spec: null at query time
    relevance_score: Optional[float]         # spec: null at query time


# ─── Evaluation ──────────────────────────────────────────────────────────────

class EvaluateResponse(BaseModel):
    response_id: str
    faithfulness_score: float
    relevance_score: float
    combined_score: float


# ─── Feedback ─────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    response_id: str
    rating: int = Field(..., ge=1, le=5, description="Rating between 1 (worst) and 5 (best).")
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    message: str                             # spec: {"feedback_id": "...", "message": "Feedback recorded"}


class FeedbackSummaryResponse(BaseModel):
    response_id: str
    avg_rating: float
    total_count: int


# ─── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    db_connected: bool                       # spec: db_connected (bool), not "database" (str)
    chunks_indexed: int                      # spec: chunks_indexed count
