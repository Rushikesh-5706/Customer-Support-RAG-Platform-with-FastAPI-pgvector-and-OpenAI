import re
import json
import logging
from typing import Optional

import psycopg2
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_DOC_ID_PATTERN = re.compile(r'^doc_\d{3}$')


class Document(BaseModel):
    doc_id: str
    title: str
    content: str
    source_url: Optional[str] = None
    metadata: dict = {}


class DocumentLoader:

    def load_from_dict(self, data: dict) -> Document:
        doc_id = data.get('doc_id', '')
        if not _DOC_ID_PATTERN.match(doc_id):
            raise ValueError(
                f"Invalid doc_id '{doc_id}'. Must match pattern doc_NNN (e.g., doc_001)."
            )
        content = data.get('content', '')
        if not content or not content.strip():
            raise ValueError(
                f"Document '{doc_id}' has empty or whitespace-only content."
            )
        return Document(
            doc_id=doc_id,
            title=data.get('title', ''),
            content=content,
            source_url=data.get('source_url'),
            metadata=data.get('metadata', {}),
        )

    def load_batch(self, data_list: list[dict]) -> list[Document]:
        loaded = []
        for item in data_list:
            try:
                doc = self.load_from_dict(item)
                loaded.append(doc)
            except ValueError as exc:
                logger.warning("Skipping document — validation error: %s", exc)
        return loaded

    def save_to_db(self, documents: list[Document], conn) -> int:
        upsert_sql = """
            INSERT INTO intellisupport.documents
                (doc_id, title, content, source_url, metadata)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (doc_id) DO UPDATE
                SET content    = EXCLUDED.content,
                    title      = EXCLUDED.title,
                    source_url = EXCLUDED.source_url,
                    metadata   = EXCLUDED.metadata,
                    updated_at = NOW()
        """
        count = 0
        with conn.cursor() as cur:
            for doc in documents:
                cur.execute(upsert_sql, (
                    doc.doc_id,
                    doc.title,
                    doc.content,
                    doc.source_url,
                    json.dumps(doc.metadata),
                ))
                count += 1
        conn.commit()
        return count
