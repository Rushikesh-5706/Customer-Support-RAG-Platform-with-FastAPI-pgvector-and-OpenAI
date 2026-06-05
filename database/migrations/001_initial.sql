CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS intellisupport;

CREATE TABLE IF NOT EXISTS intellisupport.documents (
    id          SERIAL PRIMARY KEY,
    doc_id      VARCHAR(64) UNIQUE NOT NULL,
    title       TEXT NOT NULL,
    source_url  TEXT,
    content     TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMP DEFAULT NOW(),
    updated_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS intellisupport.chunks (
    id          SERIAL PRIMARY KEY,
    chunk_id    VARCHAR(64) UNIQUE NOT NULL,
    doc_id      VARCHAR(64) REFERENCES intellisupport.documents(doc_id),
    content     TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    token_count INTEGER,
    embedding   VECTOR(1536),
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS intellisupport.queries (
    id                  SERIAL PRIMARY KEY,
    query_id            VARCHAR(64) UNIQUE NOT NULL,
    raw_query           TEXT NOT NULL,
    intent              VARCHAR(64),
    intent_confidence   FLOAT,
    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS intellisupport.responses (
    id                  SERIAL PRIMARY KEY,
    response_id         VARCHAR(64) UNIQUE NOT NULL,
    query_id            VARCHAR(64) REFERENCES intellisupport.queries(query_id),
    response_text       TEXT NOT NULL,
    retrieved_chunk_ids TEXT[],
    faithfulness_score  FLOAT,
    relevance_score     FLOAT,
    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS intellisupport.feedback (
    id          SERIAL PRIMARY KEY,
    feedback_id VARCHAR(64) UNIQUE NOT NULL,
    response_id VARCHAR(64) REFERENCES intellisupport.responses(response_id),
    rating      SMALLINT CHECK (rating BETWEEN 1 AND 5),
    comment     TEXT,
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id
    ON intellisupport.chunks(doc_id);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON intellisupport.chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);
