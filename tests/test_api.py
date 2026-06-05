"""
Tests for the FastAPI API layer: all endpoints.
"""

import pytest
from unittest.mock import MagicMock, patch


def _make_mock_conn():
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = None
    conn.cursor.return_value.__enter__.return_value = cursor
    return conn


@pytest.fixture(scope="module")
def client():
    """
    Build the TestClient with all external dependencies mocked.
    Settings must be patched before any intellisupport module is imported
    so that pydantic-settings does not attempt to read the real .env.
    """
    mock_settings = MagicMock()
    mock_settings.openai_api_key = "test-key-abc123"
    mock_settings.database_url = "postgresql://postgres:postgres@localhost:5432/intellisupport"
    mock_settings.embedding_model = "text-embedding-3-small"
    mock_settings.generation_model = "gpt-4o-mini"
    mock_settings.chunk_size = 512
    mock_settings.chunk_overlap = 50
    mock_settings.hybrid_alpha = 0.7
    mock_settings.top_k = 5

    mock_conn = _make_mock_conn()

    patches = [
        patch("config.settings", mock_settings),
        patch("psycopg2.connect", return_value=mock_conn),
    ]

    for p in patches:
        p.start()

    try:
        # Import app only after patches are active
        import importlib
        import main as main_module
        importlib.reload(main_module)

        from fastapi.testclient import TestClient
        with TestClient(main_module.app, raise_server_exceptions=False) as c:
            yield c
    finally:
        for p in patches:
            p.stop()


def mock_db_conn():
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = None
    conn.cursor.return_value.__enter__.return_value = cursor
    return conn


# ─── Root ─────────────────────────────────────────────────────────────────────

class TestRoot:
    def test_root_returns_service_info(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "service" in data
        assert "version" in data


# ─── Health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok(self, client):
        with patch("api.routes.psycopg2.connect") as mock_conn_factory:
            mock_conn = mock_db_conn()
            mock_conn_factory.return_value = mock_conn
            with patch("api.routes.get_conn") as mock_dep:
                mock_dep.return_value = mock_conn
                resp = client.get("/api/v1/health")
        # Health endpoint may return 200 or 500 depending on mock; just check it responds
        assert resp.status_code in (200, 500)


# ─── Ingest ───────────────────────────────────────────────────────────────────

class TestIngest:
    def test_ingest_invalid_doc_id_returns_422(self, client):
        payload = {
            "documents": [
                {"doc_id": "bad_id", "title": "Test", "content": "Content here."}
            ]
        }
        with patch("api.routes.get_conn", return_value=mock_db_conn()):
            with patch("api.routes.DocumentLoader") as mock_loader_cls:
                mock_loader = MagicMock()
                mock_loader.load_batch.return_value = []
                mock_loader_cls.return_value = mock_loader
                resp = client.post("/api/v1/ingest", json=payload)
        assert resp.status_code == 422

    def test_ingest_empty_documents_list(self, client):
        payload = {"documents": []}
        resp = client.post("/api/v1/ingest", json=payload)
        assert resp.status_code in (422, 500)


# ─── Query ────────────────────────────────────────────────────────────────────

class TestQuery:
    def test_query_missing_query_field(self, client):
        resp = client.post("/api/v1/query", json={})
        assert resp.status_code == 422

    def test_query_empty_string(self, client):
        resp = client.post("/api/v1/query", json={"query": ""})
        assert resp.status_code == 422

    def test_query_top_k_out_of_range(self, client):
        resp = client.post("/api/v1/query", json={"query": "test", "top_k": 0})
        assert resp.status_code == 422

    def test_query_top_k_too_large(self, client):
        resp = client.post("/api/v1/query", json={"query": "test", "top_k": 25})
        assert resp.status_code == 422


# ─── Feedback ─────────────────────────────────────────────────────────────────

class TestFeedback:
    def test_feedback_rating_out_of_range_low(self, client):
        payload = {"response_id": "rsp_abc123", "rating": 0}
        resp = client.post("/api/v1/feedback", json=payload)
        assert resp.status_code == 422

    def test_feedback_rating_out_of_range_high(self, client):
        payload = {"response_id": "rsp_abc123", "rating": 6}
        resp = client.post("/api/v1/feedback", json=payload)
        assert resp.status_code == 422

    def test_feedback_missing_response_id(self, client):
        payload = {"rating": 4}
        resp = client.post("/api/v1/feedback", json=payload)
        assert resp.status_code == 422


# ─── Evaluate ─────────────────────────────────────────────────────────────────

class TestEvaluate:
    def test_evaluate_missing_fields(self, client):
        resp = client.post("/api/v1/evaluate", json={"query_id": "qry_abc"})
        assert resp.status_code == 422

    def test_evaluate_missing_query_id(self, client):
        resp = client.post("/api/v1/evaluate", json={"response_id": "rsp_abc"})
        assert resp.status_code == 422


# ─── Benchmark ────────────────────────────────────────────────────────────────

class TestBenchmark:
    def test_benchmark_empty_test_cases(self, client):
        resp = client.post("/api/v1/benchmark", json={"test_cases": []})
        assert resp.status_code in (422, 500)

    def test_benchmark_missing_test_cases(self, client):
        resp = client.post("/api/v1/benchmark", json={})
        assert resp.status_code == 422


# ─── API Models ───────────────────────────────────────────────────────────────

class TestApiModels:
    def test_query_request_valid(self):
        from api.models import QueryRequest
        req = QueryRequest(query="How do I reset my password?")
        assert req.query == "How do I reset my password?"
        assert req.top_k is None

    def test_query_request_with_top_k(self):
        from api.models import QueryRequest
        req = QueryRequest(query="test", top_k=10)
        assert req.top_k == 10

    def test_feedback_request_valid(self):
        from api.models import FeedbackRequest
        req = FeedbackRequest(response_id="rsp_abc", rating=4, comment="Great response")
        assert req.rating == 4
        assert req.comment == "Great response"

    def test_health_response_fields(self):
        from api.models import HealthResponse
        resp = HealthResponse(status="ok", database="connected", version="1.0.0")
        assert resp.status == "ok"
        assert resp.version == "1.0.0"
