import re
import logging
from pydantic import BaseModel
from ingestion.loader import Document

logger = logging.getLogger(__name__)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    token_count: int
    metadata: dict = {}


class DocumentChunker:

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r'(?<=[.?!])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _token_count(self, text: str) -> int:
        return len(text.split())

    def chunk_document(self, document: Document) -> list[Chunk]:
        sentences = self._split_sentences(document.content)
        chunks: list[Chunk] = []
        chunk_index = 0
        current_sentences: list[str] = []
        current_tokens = 0

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = self._token_count(sentence)

            # If a single sentence exceeds chunk_size, emit it as its own chunk
            if sentence_tokens >= self.chunk_size:
                if current_sentences:
                    body = ' '.join(current_sentences)
                    chunks.append(Chunk(
                        chunk_id=f"chunk_{document.doc_id}_{chunk_index}",
                        doc_id=document.doc_id,
                        content=body,
                        chunk_index=chunk_index,
                        token_count=self._token_count(body),
                        metadata={"source_doc": document.doc_id},
                    ))
                    chunk_index += 1
                    current_sentences = []
                    current_tokens = 0

                chunks.append(Chunk(
                    chunk_id=f"chunk_{document.doc_id}_{chunk_index}",
                    doc_id=document.doc_id,
                    content=sentence,
                    chunk_index=chunk_index,
                    token_count=sentence_tokens,
                    metadata={"source_doc": document.doc_id},
                ))
                chunk_index += 1
                i += 1
                continue

            if current_tokens + sentence_tokens > self.chunk_size and current_sentences:
                body = ' '.join(current_sentences)
                chunks.append(Chunk(
                    chunk_id=f"chunk_{document.doc_id}_{chunk_index}",
                    doc_id=document.doc_id,
                    content=body,
                    chunk_index=chunk_index,
                    token_count=self._token_count(body),
                    metadata={"source_doc": document.doc_id},
                ))
                chunk_index += 1

                # Build overlap window from trailing sentences
                overlap_tokens = 0
                overlap_sentences: list[str] = []
                for sent in reversed(current_sentences):
                    sent_tokens = self._token_count(sent)
                    if overlap_tokens + sent_tokens <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_tokens += sent_tokens
                    else:
                        break

                current_sentences = overlap_sentences
                current_tokens = overlap_tokens
                # Do NOT increment i — reprocess this sentence in the new window
            else:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens
                i += 1

        if current_sentences:
            body = ' '.join(current_sentences)
            chunks.append(Chunk(
                chunk_id=f"chunk_{document.doc_id}_{chunk_index}",
                doc_id=document.doc_id,
                content=body,
                chunk_index=chunk_index,
                token_count=self._token_count(body),
                metadata={"source_doc": document.doc_id},
            ))

        return chunks

    def chunk_batch(self, documents: list[Document]) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        for document in documents:
            all_chunks.extend(self.chunk_document(document))
        return all_chunks
