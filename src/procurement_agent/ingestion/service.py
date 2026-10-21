"""Upload validation and the parse -> chunk -> embed -> index pipeline."""

from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import PurePath

from ..config import Settings
from ..errors import ValidationFailure
from ..models import DocumentCategory, DocumentRecord
from ..providers.base import EmbeddingProvider
from ..retrieval.base import SearchBackend
from ..security import sanitize_filename
from .chunker import chunk_sections
from .parsers import detect_content_type, parse_document
from .registry import DocumentRegistry

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        registry: DocumentRegistry,
        embeddings: EmbeddingProvider,
        search: SearchBackend,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.embeddings = embeddings
        self.search = search

    def upload(
        self,
        *,
        filename: str,
        data: bytes,
        category: DocumentCategory,
        supplier_id: str | None,
        owner: str,
    ) -> DocumentRecord:
        safe_name = sanitize_filename(filename)
        extension = PurePath(safe_name).suffix.lower()
        if extension not in self.settings.allowed_extension_set:
            raise ValidationFailure(
                f"Unsupported file type '{extension or '(none)'}'. Allowed: "
                + ", ".join(sorted(self.settings.allowed_extension_set))
            )
        if not data:
            raise ValidationFailure("Uploaded file is empty.")
        if len(data) > self.settings.max_upload_bytes:
            raise ValidationFailure(f"File exceeds the {self.settings.max_upload_bytes} byte limit.")
        if category is DocumentCategory.SUPPLIER and not supplier_id:
            raise ValidationFailure("supplier_id is required for supplier documents.")

        record = DocumentRecord(
            document_id=uuid.uuid4().hex,
            filename=safe_name,
            category=category,
            supplier_id=supplier_id if category is DocumentCategory.SUPPLIER else None,
            owner=owner,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            content_type=detect_content_type(extension, data),
        )
        self.registry.add(record, data)
        logger.info("document.uploaded", extra={"document_id": record.document_id, "category": category})
        return record

    async def index(self, document_ids: list[str], owner: str) -> list[DocumentRecord]:
        records = [self.registry.get(doc_id, owner) for doc_id in document_ids]
        await self.search.ensure_index()
        indexed = []
        for record in records:
            extension = PurePath(record.filename).suffix.lower()
            sections = parse_document(extension, self.registry.read_bytes(record))
            chunks = chunk_sections(
                record, sections, self.settings.chunk_max_chars, self.settings.chunk_overlap_chars
            )
            if not chunks:
                raise ValidationFailure(f"No extractable text in '{record.filename}'.")
            vectors, tokens = await self.embeddings.embed([c.content for c in chunks])
            await self.search.delete_document(record.document_id, owner)
            await self.search.upsert(chunks, vectors)
            record = record.model_copy(update={"indexed": True, "chunk_count": len(chunks)})
            self.registry.update(record)
            indexed.append(record)
            logger.info(
                "document.indexed",
                extra={
                    "document_id": record.document_id,
                    "chunks": len(chunks),
                    "suspicious_chunks": sum(c.suspicious for c in chunks),
                    "embedding_tokens": tokens,
                    "search_backend": self.search.name,
                },
            )
        return indexed
