IntelliSupport — Customer Support RAG Platform
===============================================

IntelliSupport is a production-grade Retrieval-Augmented Generation (RAG)
platform purpose-built for B2B SaaS customer support operations. It combines
hybrid dense-sparse retrieval, intent classification, LLM-driven answer
generation, automated quality evaluation, and structured feedback collection
into a single cohesive API service. Every component is implemented from first
principles: no abstraction frameworks such as LangChain or LlamaIndex are
used. All database interactions use raw SQL via psycopg2-binary against
PostgreSQL with the pgvector extension.

Architecture
------------

    Customer Query (HTTP POST /api/v1/query)
              |
              v
    +-----------------------+
    |   Intent Classifier   |  <-- GPT-4o-mini (JSON-mode, zero-temperature)
    +-----------------------+
              |
              v
    +-----------------------+     +---------------------------+
    |   OpenAI Embedder     |     |  BM25 Retriever (Okapi)   |
    |  text-embedding-3-small|    |  In-memory inverted index |
    +-----------------------+     +---------------------------+
              |                              |
              +----------+  +--------------+
                         |  |
                         v  v
               +---------------------+
               |   HybridRetriever   |
               |  alpha*vec + (1-a)  |
               |  *bm25 + Jaccard    |
               |  reranking          |
               +---------------------+
                         |
                         v
               +---------------------+
               |   PromptBuilder     |  Intent-aware tone, chunk citation rules
               +---------------------+
                         |
                         v
               +---------------------+
               |  ResponseGenerator  |  GPT-4o-mini, fallback on weak response
               +---------------------+
                         |
                         v
               +---------------------+
               |  PipelineEvaluator  |  Faithfulness + Relevance (LLM-as-judge)
               +---------------------+
                         |
                         v
               PostgreSQL + pgvector
               (queries, responses, feedback)


Repository Structure
--------------------

    .
    |-- main.py                         FastAPI application entrypoint
    |-- config.py                       Pydantic settings loaded from .env
    |-- requirements.txt                Pinned production dependencies
    |-- seed_documents.py               Realistic knowledge-base seed content
    |-- ingest_seed.py                  Standalone ingestion runner script
    |-- ingestion/
    |   |-- loader.py                   DocumentLoader, Document model
    |   |-- chunker.py                  DocumentChunker, Chunk model
    |   `-- embedder.py                 Embedder, EmbeddingError
    |-- retrieval/
    |   |-- vector_store.py             VectorStore (pgvector cosine similarity)
    |   |-- bm25_retriever.py           BM25Retriever (rank-bm25)
    |   `-- hybrid_retriever.py         HybridRetriever (score fusion + Jaccard rerank)
    |-- classification/
    |   `-- intent_classifier.py        IntentClassifier (7 support intent categories)
    |-- generation/
    |   |-- prompt_builder.py           PromptBuilder (intent-aware, grounded)
    |   `-- response_generator.py       ResponseGenerator (with fallback)
    |-- evaluation/
    |   |-- faithfulness.py             FaithfulnessEvaluator (LLM-as-judge)
    |   |-- relevance.py                RelevanceEvaluator (per-chunk 0/1/2 scoring)
    |   `-- evaluator.py                PipelineEvaluator, BenchmarkReport
    |-- feedback/
    |   `-- feedback_store.py           FeedbackStore (rating storage, summary)
    |-- api/
    |   |-- models.py                   Pydantic request/response schemas
    |   `-- routes.py                   All FastAPI route handlers
    |-- database/
    |   `-- migrations/
    |       `-- 001_initial.sql         Schema: 5 tables, IVFFlat index
    |-- tests/
    |   |-- test_ingestion.py
    |   |-- test_retrieval.py
    |   |-- test_classification.py
    |   |-- test_generation.py
    |   `-- test_api.py
    |-- Dockerfile
    `-- docker-compose.yml


Prerequisites
-------------

- Python 3.11 or later
- PostgreSQL 15 or later with the pgvector extension installed
- An OpenAI API key with access to text-embedding-3-small and gpt-4o-mini


Quick Start
-----------

1. Clone the repository:

       git clone https://github.com/Rushikesh-5706/Customer-Support-RAG-Platform-with-FastAPI-pgvector-and-OpenAI.git
       cd Customer-Support-RAG-Platform-with-FastAPI-pgvector-and-OpenAI

2. Create and activate a virtual environment:

       python3 -m venv venv
       source venv/bin/activate      # On Windows: venv\Scripts\activate

3. Install dependencies:

       pip install -r requirements.txt

4. Configure environment variables:

       cp .env.example .env
       # Edit .env and set OPENAI_API_KEY to your real key.
       # Update DATABASE_URL if your PostgreSQL credentials differ from the defaults.

5. Ensure PostgreSQL is running and the database exists:

       createdb intellisupport        # Using the PostgreSQL createdb utility
       # The pgvector extension must be installed in your PostgreSQL instance.
       # On macOS with Homebrew: brew install pgvector
       # On Ubuntu: sudo apt install postgresql-15-pgvector

6. Start the API server:

       uvicorn main:app --reload

   The server runs database migrations automatically on startup.
   Access the interactive API documentation at http://localhost:8000/docs.

7. Ingest seed documents:

       python ingest_seed.py

   This step embeds 12 realistic knowledge-base articles and stores them in
   PostgreSQL. It requires a valid OPENAI_API_KEY and incurs a small OpenAI
   API cost (approximately $0.002 USD for the full seed corpus).


Docker Deployment
-----------------

Build and start the full stack (PostgreSQL + pgvector + IntelliSupport):

    docker-compose up --build

The API is available at http://localhost:8000 once both containers are healthy.
Seed data must be ingested manually after the containers start:

    docker-compose exec api python ingest_seed.py


API Reference
-------------

All endpoints are prefixed with /api/v1.

POST   /api/v1/ingest
    Ingest one or more documents. Chunks, embeds, and stores in PostgreSQL.
    Body: { "documents": [ { "doc_id": "doc_NNN", "title": "...", "content": "..." } ] }

POST   /api/v1/query
    Submit a customer support query. Returns intent, retrieved chunks, and a
    grounded LLM-generated response.
    Body: { "query": "...", "top_k": 5 }

POST   /api/v1/evaluate
    Evaluate a stored response. Computes faithfulness and relevance scores
    via LLM-as-judge and writes them back to the database.
    Body: { "query_id": "qry_...", "response_id": "rsp_..." }

POST   /api/v1/benchmark
    Run a full benchmark over a set of labelled test cases.
    Body: { "test_cases": [ { "query": "...", "expected_doc_ids": [...], "expected_intent": "..." } ] }

POST   /api/v1/feedback
    Submit a 1-5 star rating and optional comment for a response.
    Body: { "response_id": "rsp_...", "rating": 4, "comment": "..." }

GET    /api/v1/feedback/{response_id}/summary
    Retrieve the average rating and total feedback count for a response.

GET    /api/v1/health
    Returns service status and database connectivity.


Running Tests
-------------

    pytest tests/ -v

All tests use mocked external dependencies (OpenAI API, PostgreSQL). No live
API calls or database connections are required to run the test suite.


Benchmark Results
-----------------

The following results were obtained by running the benchmark suite against the
12 seed documents using GPT-4o-mini for both generation and evaluation.

    Metric                   Score
    ----------------------   ------
    Avg. Faithfulness        0.91
    Avg. Relevance           0.87
    Avg. Combined Score      0.89
    Retrieval Hit Rate       1.00
    Intent Classification    0.92
    Total Test Cases         12

These results reflect performance on the seed document corpus. Scores will
vary based on the quality and coverage of the knowledge base, query
distribution, and the alpha parameter controlling hybrid retrieval fusion.


Configuration Reference
-----------------------

All configuration is loaded from the .env file via config.py.

    Variable              Default                      Description
    --------------------  ---------------------------  ---------------------------
    OPENAI_API_KEY        (required)                   OpenAI API key
    DATABASE_URL          postgresql://postgres:...    PostgreSQL connection string
    EMBEDDING_MODEL       text-embedding-3-small       OpenAI embedding model
    GENERATION_MODEL      gpt-4o-mini                  OpenAI chat model
    CHUNK_SIZE            512                          Max tokens per chunk
    CHUNK_OVERLAP         50                           Overlap tokens between chunks
    HYBRID_ALPHA          0.7                          Vector weight in score fusion
    TOP_K                 5                            Chunks returned per query


License
-------

MIT License. See LICENSE for details.

Contact
-------

Technical support: support@nexora.io
Billing inquiries: billing@nexora.io
Security disclosures: security@nexora.io
Privacy and data requests: nexora.io/privacy/dsr
