import logging
from uuid import uuid4

import psycopg2
from fastapi import APIRouter, Depends, HTTPException

from config import settings
from api.schemas import (
    QueryRequest, QueryResponse,
    EvaluateResponse,
    FeedbackRequest, FeedbackResponse, FeedbackSummaryResponse,
    HealthResponse,
)
from ingestion.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from classification.intent_classifier import IntentClassifier
from generation.prompt_builder import PromptBuilder
from generation.response_generator import ResponseGenerator
from evaluation.faithfulness import FaithfulnessEvaluator
from evaluation.relevance import RelevanceEvaluator
from evaluation.evaluator import PipelineEvaluator
from feedback.feedback_store import FeedbackStore

logger = logging.getLogger(__name__)
router = APIRouter()


def get_conn():
    conn = psycopg2.connect(settings.database_url)
    try:
        yield conn
    finally:
        conn.close()


# GET /health
@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check(conn=Depends(get_conn)):
    """
    Returns service status, database connectivity flag, and chunk count.
    Response: { "status": "ok", "db_connected": true, "chunks_indexed": 47 }
    """
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.execute("SELECT COUNT(*) FROM intellisupport.chunks")
            chunks_count = cur.fetchone()[0]
        db_connected = True
    except Exception as exc:
        logger.error("DB health check failed: %s", exc)
        db_connected = False
        chunks_count = 0
    return HealthResponse(status="ok", db_connected=db_connected, chunks_indexed=chunks_count)


# POST /query
@router.post("/query", response_model=QueryResponse, tags=["Query"])
def query(request: QueryRequest, conn=Depends(get_conn)):
    """
    Classify intent, retrieve hybrid chunks, generate response, persist to DB.
    Response includes retrieved_chunk_ids (list[str]) and null evaluation scores.
    """
    top_k = request.top_k if request.top_k is not None else settings.top_k

    classifier = IntentClassifier(model=settings.generation_model)
    embedder = Embedder(model=settings.embedding_model)
    vector_store = VectorStore(conn)
    bm25 = BM25Retriever(conn)
    hybrid = HybridRetriever(vector_store, bm25, alpha=settings.hybrid_alpha)
    prompt_builder = PromptBuilder()
    generator = ResponseGenerator(model=settings.generation_model)

    intent_result = classifier.classify(request.query)
    query_embedding = embedder.embed_text(request.query)
    retrieved = hybrid.retrieve_with_reranking(request.query, query_embedding, top_k=top_k)

    # spec: query_id = f"qry_{uuid4().hex[:8]}"  (8 chars)
    query_id = f"qry_{uuid4().hex[:8]}"
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO intellisupport.queries
                (query_id, raw_query, intent, intent_confidence)
            VALUES (%s, %s, %s, %s)
            """,
            (query_id, request.query, intent_result.intent, intent_result.confidence),
        )
    conn.commit()

    if retrieved:
        messages = prompt_builder.build_rag_prompt(request.query, retrieved, intent_result)
        fallback = prompt_builder.build_clarification_prompt(request.query, intent_result)
        generated = generator.generate_with_fallback(messages, fallback)
    else:
        messages = prompt_builder.build_clarification_prompt(request.query, intent_result)
        generated = generator.generate(messages)

    # spec: response_id = f"rsp_{uuid4().hex[:8]}"  (8 chars)
    response_id = f"rsp_{uuid4().hex[:8]}"
    chunk_ids = [ch.chunk_id for ch in retrieved]
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO intellisupport.responses
                (response_id, query_id, response_text, retrieved_chunk_ids)
            VALUES (%s, %s, %s, %s)
            """,
            (response_id, query_id, generated.response_text, chunk_ids),
        )
    conn.commit()

    return QueryResponse(
        query_id=query_id,
        response_id=response_id,
        response_text=generated.response_text,
        intent=intent_result.intent,
        intent_confidence=intent_result.confidence,
        retrieved_chunk_ids=chunk_ids,         # spec: list of chunk_id strings
        faithfulness_score=None,               # spec: null at query time
        relevance_score=None,                  # spec: null at query time
    )


# POST /evaluate/{response_id}   ← path param, not request body
@router.post("/evaluate/{response_id}", response_model=EvaluateResponse, tags=["Evaluation"])
def evaluate_response(response_id: str, conn=Depends(get_conn)):
    """
    Trigger LLM-as-judge evaluation for an existing response.
    Loads query+response+chunks from DB, computes faithfulness+relevance, persists scores.
    """
    # Fetch query_id for this response
    with conn.cursor() as cur:
        cur.execute(
            "SELECT query_id FROM intellisupport.responses WHERE response_id = %s",
            (response_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Response '{response_id}' not found.")
    query_id = row[0]

    evaluator = PipelineEvaluator(
        faithfulness_evaluator=FaithfulnessEvaluator(model=settings.generation_model),
        relevance_evaluator=RelevanceEvaluator(model=settings.generation_model),
        conn=conn,
    )
    try:
        report = evaluator.evaluate_response(query_id, response_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return EvaluateResponse(
        response_id=report.response_id,
        faithfulness_score=report.faithfulness_score,
        relevance_score=report.relevance_score,
        combined_score=report.combined_score,
    )


# POST /feedback
@router.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
def submit_feedback(request: FeedbackRequest, conn=Depends(get_conn)):
    """
    Submit a 1–5 star rating and optional comment for a response.
    Response: { "feedback_id": "fb_xxxxxxxx", "message": "Feedback recorded" }
    """
    store = FeedbackStore(conn)
    try:
        feedback_id = store.store_feedback(
            response_id=request.response_id,
            rating=request.rating,
            comment=request.comment,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return FeedbackResponse(
        feedback_id=feedback_id,
        message="Feedback recorded",
    )


# GET /feedback/summary/{response_id}
@router.get(
    "/feedback/summary/{response_id}",
    response_model=FeedbackSummaryResponse,
    tags=["Feedback"],
)
def get_feedback_summary(response_id: str, conn=Depends(get_conn)):
    """
    Retrieve aggregated feedback (avg rating, count) for a given response.
    """
    store = FeedbackStore(conn)
    summary = store.get_feedback_summary(response_id)
    return FeedbackSummaryResponse(
        response_id=summary.response_id,
        avg_rating=summary.avg_rating,
        total_count=summary.total_count,
    )
