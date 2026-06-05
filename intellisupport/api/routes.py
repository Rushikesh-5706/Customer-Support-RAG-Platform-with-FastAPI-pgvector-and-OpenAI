import logging
from uuid import uuid4

import psycopg2
from fastapi import APIRouter, Depends, HTTPException

from config import settings
from api.schemas import (
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse, RetrievedChunkOut,
    EvaluateRequest, EvaluateResponse,
    BenchmarkRequest, BenchmarkResponse,
    FeedbackRequest, FeedbackResponse, FeedbackSummaryResponse,
    HealthResponse,
)
from ingestion.loader import DocumentLoader
from ingestion.chunker import DocumentChunker
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


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check(conn=Depends(get_conn)):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        db_status = "connected"
    except Exception as exc:
        logger.error("DB health check failed: %s", exc)
        db_status = "unreachable"
    return HealthResponse(status="ok", database=db_status, version="1.0.0")


@router.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
def ingest_documents(request: IngestRequest, conn=Depends(get_conn)):
    loader = DocumentLoader()
    chunker = DocumentChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    embedder = Embedder(model=settings.embedding_model)

    documents = loader.load_batch(request.documents)
    if not documents:
        raise HTTPException(
            status_code=422,
            detail="No valid documents found in request. Check doc_id format and content.",
        )

    docs_saved = loader.save_to_db(documents, conn)
    chunks = chunker.chunk_batch(documents)
    chunks_embedded = embedder.embed_and_store_chunks(chunks, conn)

    return IngestResponse(
        documents_saved=docs_saved,
        chunks_created=len(chunks),
        chunks_embedded=chunks_embedded,
    )


@router.post("/query", response_model=QueryResponse, tags=["Query"])
def query(request: QueryRequest, conn=Depends(get_conn)):
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

    query_id = f"qry_{uuid4().hex[:16]}"
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

    response_id = f"rsp_{uuid4().hex[:16]}"
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
        retrieved_chunks=[
            RetrievedChunkOut(
                chunk_id=ch.chunk_id,
                doc_id=ch.doc_id,
                content=ch.content,
                score=ch.score,
                retrieval_method=ch.retrieval_method,
            )
            for ch in retrieved
        ],
        prompt_tokens=generated.prompt_tokens,
        completion_tokens=generated.completion_tokens,
        total_tokens=generated.total_tokens,
    )


@router.post("/evaluate", response_model=EvaluateResponse, tags=["Evaluation"])
def evaluate_response(request: EvaluateRequest, conn=Depends(get_conn)):
    evaluator = PipelineEvaluator(
        faithfulness_evaluator=FaithfulnessEvaluator(model=settings.generation_model),
        relevance_evaluator=RelevanceEvaluator(model=settings.generation_model),
        conn=conn,
    )
    try:
        report = evaluator.evaluate_response(request.query_id, request.response_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return EvaluateResponse(
        query_id=report.query_id,
        response_id=report.response_id,
        faithfulness_score=report.faithfulness_score,
        relevance_score=report.relevance_score,
        combined_score=report.combined_score,
    )


@router.post("/benchmark", response_model=BenchmarkResponse, tags=["Evaluation"])
def run_benchmark(request: BenchmarkRequest, conn=Depends(get_conn)):
    if not request.test_cases:
        raise HTTPException(status_code=422, detail="test_cases must be non-empty.")
    evaluator = PipelineEvaluator(
        faithfulness_evaluator=FaithfulnessEvaluator(model=settings.generation_model),
        relevance_evaluator=RelevanceEvaluator(model=settings.generation_model),
        conn=conn,
    )
    report = evaluator.run_benchmark(request.test_cases)
    return BenchmarkResponse(
        total_cases=report.total_cases,
        avg_faithfulness=report.avg_faithfulness,
        avg_relevance=report.avg_relevance,
        avg_combined=report.avg_combined,
        retrieval_hit_rate=report.retrieval_hit_rate,
        intent_accuracy=report.intent_accuracy,
    )


@router.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
def submit_feedback(request: FeedbackRequest, conn=Depends(get_conn)):
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
        response_id=request.response_id,
        rating=request.rating,
        comment=request.comment,
    )


@router.get(
    "/feedback/{response_id}/summary",
    response_model=FeedbackSummaryResponse,
    tags=["Feedback"],
)
def get_feedback_summary(response_id: str, conn=Depends(get_conn)):
    store = FeedbackStore(conn)
    summary = store.get_feedback_summary(response_id)
    return FeedbackSummaryResponse(
        response_id=summary.response_id,
        avg_rating=summary.avg_rating,
        total_count=summary.total_count,
    )
