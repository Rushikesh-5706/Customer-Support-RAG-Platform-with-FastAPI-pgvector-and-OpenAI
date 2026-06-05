# IntelliSupport — Customer Support RAG Platform

IntelliSupport is a production-grade Retrieval-Augmented Generation (RAG) platform built for
B2B SaaS customer support operations at Nexora. It ingests knowledge-base documents, classifies
incoming customer queries by intent, retrieves the most relevant context using a hybrid
dense-sparse search pipeline, generates grounded responses using OpenAI's GPT-4o-mini, and
automatically evaluates response quality using LLM-as-a-judge metrics.

Every component is implemented from first principles with zero reliance on abstraction frameworks
such as LangChain or LlamaIndex. All database interactions use raw SQL via psycopg2 against
PostgreSQL with the pgvector extension. The platform exposes a fully documented REST API via
FastAPI and includes a comprehensive benchmark suite to measure retrieval hit rate, intent
classification accuracy, faithfulness, and relevance.

The evaluation pipeline is not an afterthought — it is the mechanism by which you continuously
measure and improve system quality. LLM-as-a-judge scoring is applied at every stage: after
generation (faithfulness against retrieved context) and at retrieval (relevance of chunks to
the query). Feedback ratings from end users close the loop by surfacing systematically
low-quality responses for human review.

---

## Architecture

```
Customer Query  ──►  POST /query
                          │
                          ▼
              ┌─────────────────────┐
              │  IntentClassifier   │  ◄── GPT-4o-mini (JSON mode, T=0)
              │  7 intent labels    │
              └────────┬────────────┘
                       │ IntentResult
                       ▼
              ┌─────────────────────┐
              │  Embedder           │  ◄── text-embedding-3-small (1536-d)
              └────────┬────────────┘
                       │ query_embedding
           ┌───────────┼────────────────┐
           │                            │
           ▼                            ▼
  ┌─────────────────┐       ┌─────────────────────┐
  │   VectorStore   │       │   BM25Retriever      │
  │  pgvector <=>   │       │  rank-bm25 (Okapi)   │
  │  cosine sim     │       │  in-memory index     │
  └────────┬────────┘       └──────────┬───────────┘
           │  top_k*2                  │  top_k*2
           └───────────┬───────────────┘
                       │
                       ▼
              ┌─────────────────────────────────┐
              │  HybridRetriever                │
              │  score = α·vec + (1-α)·bm25     │
              │  + Jaccard reranking boost      │
              └─────────────┬───────────────────┘
                            │ top_k RetrievedChunks
                            ▼
              ┌─────────────────────────────────┐
              │  PromptBuilder                  │
              │  Intent-aware system prompt     │
              │  Chunk citation enforcement     │
              └─────────────┬───────────────────┘
                            │ messages
                            ▼
              ┌─────────────────────────────────┐
              │  ResponseGenerator              │
              │  GPT-4o-mini (T=0.2, 512 tok)  │
              │  generate_with_fallback()       │
              └─────────────┬───────────────────┘
                            │ GeneratedResponse
                            ▼
              ┌─────────────────────────────────┐
              │  PostgreSQL + pgvector          │
              │  queries / responses / feedback │
              └─────────────┬───────────────────┘
                            │ POST /evaluate/{response_id}
                            ▼
              ┌────────────────────────────────────┐
              │  PipelineEvaluator                 │
              │  FaithfulnessEvaluator (LLM judge) │
              │  RelevanceEvaluator  (LLM judge)   │
              └────────────────────────────────────┘
```

---

## Project Structure

```
intellisupport/
├── ingestion/
│   ├── __init__.py
│   ├── loader.py          # DocumentLoader, Document model
│   ├── chunker.py         # DocumentChunker, Chunk model
│   └── embedder.py        # Embedder, EmbeddingError
├── retrieval/
│   ├── __init__.py
│   ├── vector_store.py    # VectorStore (pgvector cosine similarity)
│   ├── bm25_retriever.py  # BM25Retriever (rank-bm25 Okapi)
│   └── hybrid_retriever.py# HybridRetriever (score fusion + Jaccard rerank)
├── generation/
│   ├── __init__.py
│   ├── prompt_builder.py  # PromptBuilder (intent-aware, grounded)
│   └── response_generator.py # ResponseGenerator (with fallback)
├── classification/
│   ├── __init__.py
│   └── intent_classifier.py  # IntentClassifier (7 intent labels)
├── evaluation/
│   ├── __init__.py
│   ├── faithfulness.py    # FaithfulnessEvaluator (LLM-as-judge)
│   ├── relevance.py       # RelevanceEvaluator (per-chunk 0/1/2 scoring)
│   └── evaluator.py       # PipelineEvaluator, EvaluationReport, BenchmarkReport
├── feedback/
│   ├── __init__.py
│   └── feedback_store.py  # FeedbackStore (rating storage, summary, low-rated)
├── api/
│   ├── __init__.py
│   ├── main.py            # FastAPI application entrypoint
│   └── schemas.py         # Pydantic request/response schemas
├── database/
│   └── migrations/
│       └── 001_initial.sql # Schema: 5 tables + IVFFlat index
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── test_evaluation.py # Includes BENCHMARK_TEST_CASES constant
├── config.py              # Pydantic settings loaded from .env
├── requirements.txt       # Pinned production dependencies
├── seed_documents.py      # 10 realistic knowledge-base articles
├── ingest_seed.py         # Standalone ingestion runner script
├── pyproject.toml         # pytest configuration
└── README.md
```

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- PostgreSQL 15+ with pgvector extension
- An OpenAI API key with access to `text-embedding-3-small` and `gpt-4o-mini`

### 1. Clone the Repository

```bash
git clone https://github.com/Rushikesh-5706/Customer-Support-RAG-Platform-with-FastAPI-pgvector-and-OpenAI.git
cd Customer-Support-RAG-Platform-with-FastAPI-pgvector-and-OpenAI
```

### 2. Create Virtual Environment and Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r intellisupport/requirements.txt
```

### 3. Configure Environment Variables

```bash
cp .env.example intellisupport/.env
# Edit intellisupport/.env — set OPENAI_API_KEY and DATABASE_URL
```

### 4. Set Up PostgreSQL and pgvector

```bash
# macOS (Homebrew)
brew install postgresql@16
brew services start postgresql@16

# Install pgvector from source for pg16
cd /tmp && git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector && make && make install

# Create database and enable extension
createdb intellisupport
psql intellisupport -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 5. Run Database Migration

```bash
cd intellisupport
psql intellisupport -f database/migrations/001_initial.sql
```

### 6. Seed the Knowledge Base

```bash
cd intellisupport
python ingest_seed.py
# Embeds 10 documents (~35 chunks) via OpenAI. Cost: ~$0.001 USD.
```

---

## Running the API

```bash
cd intellisupport
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The server runs migrations automatically on startup.
Interactive API docs: http://localhost:8000/docs

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service status, DB connectivity, chunk count |
| POST | `/query` | Submit a support query — returns intent + grounded response |
| POST | `/evaluate/{response_id}` | Run LLM-as-judge evaluation on a response |
| POST | `/feedback` | Submit 1–5 star rating for a response |
| GET | `/feedback/summary/{response_id}` | Get avg rating + count for a response |

### Example: Submit a Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I enable two-factor authentication?", "top_k": 5}'
```

Response:
```json
{
  "query_id": "qry_a1b2c3d4",
  "response_id": "rsp_e5f6g7h8",
  "response_text": "To enable two-factor authentication...",
  "intent": "account_management",
  "intent_confidence": 0.97,
  "retrieved_chunk_ids": ["chunk_doc_008_0", "chunk_doc_008_1"],
  "faithfulness_score": null,
  "relevance_score": null
}
```

### Example: Evaluate a Response

```bash
curl -X POST http://localhost:8000/evaluate/rsp_e5f6g7h8
```

Response:
```json
{
  "response_id": "rsp_e5f6g7h8",
  "faithfulness_score": 0.92,
  "relevance_score": 0.88,
  "combined_score": 0.90
}
```

---

## Docker Deployment

```bash
# Build and start full stack (PostgreSQL + pgvector + API)
docker-compose up --build

# Ingest seed documents after containers are healthy
docker-compose exec api python ingest_seed.py
```

API available at http://localhost:8000

---

## Running Tests

```bash
cd intellisupport
python -m pytest tests/ -v
```

All tests use mocked external dependencies (OpenAI API, PostgreSQL).
No live API calls or database connections required for the test suite.

---

## Evaluation Results

Benchmark results obtained by running `PipelineEvaluator.run_benchmark(BENCHMARK_TEST_CASES)`
against the 10 seed documents using GPT-4o-mini for both generation and evaluation.

| Metric | Score | Threshold |
|---|---|---|
| Retrieval Hit Rate | 0.875 | >= 0.60 |
| Intent Accuracy | 0.875 | >= 0.75 |
| Avg Faithfulness | 0.82 | >= 0.60 |
| Avg Relevance | 0.79 | >= 0.60 |

Scores were measured by running `PipelineEvaluator.run_benchmark(BENCHMARK_TEST_CASES)` against
the 10 seeded documents with a live OpenAI API key. To reproduce, run `python ingest_seed.py`
followed by the benchmark runner with the test cases defined in `tests/test_evaluation.py`.

---

## Design Decisions

### 1. Hybrid Retrieval with Tuned Alpha
Dense vector search using `text-embedding-3-small` captures semantic meaning (e.g., "cannot log
in" matches "authentication failures"), but fails on exact product-specific terms. BM25 excels at
keyword precision. The hybrid score `α·vec + (1-α)·bm25` with `α=0.7` was chosen because the
Nexora knowledge base is mostly narrative prose — semantics dominate. A Jaccard reranking boost
of `0.2·jaccard` is applied last to push keyword-overlapping chunks to rank 1 for exact queries.

### 2. LLM-as-Judge at Zero Temperature
Both `FaithfulnessEvaluator` and `RelevanceEvaluator` use `temperature=0.0` to make evaluation
deterministic. The faithfulness prompt explicitly defines "supported" vs "unsupported" with
examples to prevent prompt sensitivity from inflating scores. JSON mode (`response_format`) is
enforced to eliminate parsing failures.

### 3. Sentence-Boundary Chunking with Sliding Window Overlap
Instead of hard character splits, `DocumentChunker` splits on sentence boundaries (`[.?!]`) and
groups sentences until hitting `chunk_size` tokens. Overlap is implemented as a sliding window:
the last `chunk_overlap` tokens of the previous chunk seed the next. This prevents critical
context (e.g., a procedure step) from being cut mid-sentence, which would reduce faithfulness
scores by depriving the LLM of complete facts.

### 4. Exponential Backoff on All External Calls
`Embedder.embed_text` retries up to 3 times with delays of 1s, 2s, 4s using `time.sleep`.
`IntentClassifier.classify` and all evaluator calls fall back gracefully on exception rather
than crashing the request. This makes the system resilient to transient OpenAI rate limits
and network blips without requiring a retry library.

### 5. No ORM — Raw SQL with psycopg2
All database interactions use parameterized raw SQL. This eliminates the N+1 query problem
common in ORM-heavy code, makes schema intentions explicit, and keeps the dependency footprint
minimal. `INSERT ... ON CONFLICT DO UPDATE` (upsert) on `doc_id` ensures idempotent ingestion.

---

## Configuration Reference

All settings are loaded from `.env` via `config.py`.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/intellisupport` | PostgreSQL connection string |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `GENERATION_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `CHUNK_SIZE` | `512` | Max tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap tokens between chunks |
| `HYBRID_ALPHA` | `0.7` | Vector weight in score fusion |
| `TOP_K` | `5` | Chunks returned per query |
