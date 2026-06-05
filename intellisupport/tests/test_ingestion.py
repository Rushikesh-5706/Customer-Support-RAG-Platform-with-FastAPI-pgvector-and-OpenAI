"""
Tests for the ingestion module: DocumentLoader, DocumentChunker, Embedder.
"""

import pytest
from unittest.mock import MagicMock, patch
from ingestion.loader import Document, DocumentLoader
from ingestion.chunker import Chunk, DocumentChunker
from ingestion.embedder import Embedder, EmbeddingError


# ─── DocumentLoader ───────────────────────────────────────────────────────────

class TestDocumentLoader:

    def setup_method(self):
        self.loader = DocumentLoader()

    def test_load_from_dict_valid(self):
        data = {
            "doc_id": "doc_001",
            "title": "Test Document",
            "content": "This is a valid document with sufficient content.",
        }
        doc = self.loader.load_from_dict(data)
        assert isinstance(doc, Document)
        assert doc.doc_id == "doc_001"
        assert doc.title == "Test Document"

    def test_load_from_dict_invalid_doc_id_format(self):
        data = {"doc_id": "document_001", "title": "Bad", "content": "Content."}
        with pytest.raises(ValueError, match="Invalid doc_id"):
            self.loader.load_from_dict(data)

    def test_load_from_dict_doc_id_no_leading_zeros(self):
        data = {"doc_id": "doc_1", "title": "Bad", "content": "Content."}
        with pytest.raises(ValueError, match="Invalid doc_id"):
            self.loader.load_from_dict(data)

    def test_load_from_dict_empty_content(self):
        data = {"doc_id": "doc_002", "title": "Empty", "content": "   "}
        with pytest.raises(ValueError, match="empty or whitespace-only"):
            self.loader.load_from_dict(data)

    def test_load_from_dict_missing_content(self):
        data = {"doc_id": "doc_003", "title": "Missing Content"}
        with pytest.raises(ValueError):
            self.loader.load_from_dict(data)

    def test_load_batch_skips_invalid(self):
        data_list = [
            {"doc_id": "doc_001", "title": "Valid", "content": "Good content here."},
            {"doc_id": "bad_id", "title": "Invalid", "content": "Content."},
            {"doc_id": "doc_002", "title": "Also Valid", "content": "More content."},
        ]
        docs = self.loader.load_batch(data_list)
        assert len(docs) == 2
        assert docs[0].doc_id == "doc_001"
        assert docs[1].doc_id == "doc_002"

    def test_load_from_dict_optional_fields(self):
        data = {
            "doc_id": "doc_099",
            "title": "Optional Fields",
            "content": "Content with optional fields.",
            "source_url": "https://example.com",
            "metadata": {"key": "value"},
        }
        doc = self.loader.load_from_dict(data)
        assert doc.source_url == "https://example.com"
        assert doc.metadata == {"key": "value"}

    def test_save_to_db(self):
        docs = [
            Document(doc_id="doc_001", title="Title 1", content="Content 1."),
            Document(doc_id="doc_002", title="Title 2", content="Content 2."),
        ]
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        count = self.loader.save_to_db(docs, mock_conn)

        assert count == 2
        mock_conn.commit.assert_called_once()
        assert mock_cursor.execute.call_count == 2


# ─── DocumentChunker ──────────────────────────────────────────────────────────

class TestDocumentChunker:

    def setup_method(self):
        self.chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)

    def _make_doc(self, content: str, doc_id: str = "doc_001") -> Document:
        return Document(doc_id=doc_id, title="Test", content=content)

    def test_chunk_document_produces_chunks(self):
        # Build content with explicit sentence boundaries so chunker splits correctly.
        # chunk_size=50 tokens; each sentence is ~12 tokens; 200 tokens total → multiple chunks.
        sentences = [
            "This is the first sentence about billing and subscription plans.",
            "The second sentence covers technical issue troubleshooting steps.",
            "Here is the third sentence about account management and password resets.",
            "The fourth sentence discusses integration with Slack and GitHub tools.",
            "Fifth sentence covers data export formats including JSON and CSV files.",
            "Sixth sentence is about feature requests and platform roadmap items.",
            "The seventh sentence addresses authentication errors and API key formats.",
            "Eighth sentence explains the hybrid retrieval scoring and alpha parameter.",
            "The ninth sentence covers faithfulness and relevance evaluation scores.",
            "Tenth sentence talks about invoice management and payment failure handling.",
            "Eleventh sentence is about SLA guarantees and service credit calculations.",
            "Twelfth sentence describes GDPR data subject access and erasure requests.",
            "The thirteenth sentence covers Jira integration and bidirectional sync.",
            "Fourteenth sentence explains the dunning process and account suspension.",
            "The fifteenth sentence addresses MFA enrollment and recovery procedures.",
        ]
        content = " ".join(sentences)
        doc = self._make_doc(content)
        chunks = self.chunker.chunk_document(doc)
        assert len(chunks) > 1
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.doc_id == "doc_001"

    def test_chunk_ids_are_unique(self):
        content = " ".join(["word"] * 200)
        doc = self._make_doc(content)
        chunks = self.chunker.chunk_document(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_index_is_sequential(self):
        content = " ".join(["word"] * 200)
        doc = self._make_doc(content)
        chunks = self.chunker.chunk_document(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_short_document_produces_single_chunk(self):
        doc = self._make_doc("Short content here.")
        chunks = self.chunker.chunk_document(doc)
        assert len(chunks) == 1

    def test_token_count_reflects_content(self):
        doc = self._make_doc("one two three four five.")
        chunks = self.chunker.chunk_document(doc)
        assert chunks[0].token_count == 5

    def test_chunk_batch_processes_multiple_docs(self):
        docs = [
            self._make_doc(" ".join(["a"] * 100), "doc_001"),
            self._make_doc(" ".join(["b"] * 100), "doc_002"),
        ]
        all_chunks = self.chunker.chunk_batch(docs)
        doc_ids = {c.doc_id for c in all_chunks}
        assert "doc_001" in doc_ids
        assert "doc_002" in doc_ids


# ─── Embedder ─────────────────────────────────────────────────────────────────

class TestEmbedder:

    def setup_method(self):
        with patch("ingestion.embedder.OpenAI"):
            self.embedder = Embedder(model="text-embedding-3-small", batch_size=2)

    def test_embed_text_returns_list_of_floats(self):
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        self.embedder._client.embeddings.create.return_value = mock_resp

        result = self.embedder.embed_text("hello world")
        assert result == [0.1, 0.2, 0.3]

    def test_embed_text_retries_on_failure(self):
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(embedding=[0.5])]
        self.embedder._client.embeddings.create.side_effect = [
            Exception("API error"),
            mock_resp,
        ]

        with patch("ingestion.embedder.time.sleep"):
            result = self.embedder.embed_text("test")
        assert result == [0.5]

    def test_embed_text_raises_after_3_failures(self):
        self.embedder._client.embeddings.create.side_effect = Exception("Always fails")
        with patch("ingestion.embedder.time.sleep"):
            with pytest.raises(EmbeddingError):
                self.embedder.embed_text("fail")

    def test_embed_batch_handles_batching(self):
        def mock_create(model, input):
            resp = MagicMock()
            resp.data = [
                MagicMock(index=i, embedding=[float(i)])
                for i in range(len(input))
            ]
            return resp

        self.embedder._client.embeddings.create.side_effect = mock_create
        texts = ["a", "b", "c", "d"]
        results = self.embedder.embed_batch(texts)
        assert len(results) == 4
