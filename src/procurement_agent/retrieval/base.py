"""Search backend interface. Every query is scoped to an owner so documents never leak between
principals, regardless of which backend is in use."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..config import RetrievalMode
from ..models import Chunk, DocumentCategory


@dataclass(frozen=True)
class SearchFilter:
    owner: str
    category: DocumentCategory | None = None
    supplier_id: str | None = None
    document_ids: tuple[str, ...] | None = None

    def matches(self, chunk: Chunk) -> bool:
        if chunk.owner != self.owner:
            return False
        if self.category is not None and chunk.category != self.category:
            return False
        if self.supplier_id is not None and chunk.supplier_id != self.supplier_id:
            return False
        if self.document_ids is not None and chunk.document_id not in self.document_ids:
            return False
        return True


class SearchBackend(Protocol):
    name: str

    async def ensure_index(self) -> None: ...

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    async def delete_document(self, document_id: str, owner: str) -> None: ...

    async def search(
        self,
        *,
        query_text: str,
        query_vector: list[float] | None,
        search_filter: SearchFilter,
        top: int,
        mode: RetrievalMode,
    ) -> list[Chunk]: ...
