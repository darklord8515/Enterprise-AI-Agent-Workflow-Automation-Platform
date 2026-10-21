"""In-memory search backend for mock mode and tests.

Mirrors the Azure AI Search query semantics closely enough to test the workflow: BM25 keyword
ranking, cosine vector ranking, and Reciprocal Rank Fusion for hybrid (the fusion method Azure AI
Search documents for hybrid queries).
"""

from __future__ import annotations

import math
from collections import Counter

from ..config import RetrievalMode
from ..models import Chunk
from ..providers.mock import _STOPWORDS, tokenize
from .base import SearchFilter

_RRF_K = 60


class InMemorySearchBackend:
    name = "in_memory"

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}

    async def ensure_index(self) -> None:
        return None

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = vector

    async def delete_document(self, document_id: str, owner: str) -> None:
        for cid in [
            c.chunk_id for c in self._chunks.values() if c.document_id == document_id and c.owner == owner
        ]:
            self._chunks.pop(cid)
            self._vectors.pop(cid)

    def _bm25(self, query: str, candidates: list[Chunk]) -> list[tuple[str, float]]:
        terms = [t for t in tokenize(query) if t not in _STOPWORDS]
        docs = {c.chunk_id: Counter(tokenize(c.content)) for c in candidates}
        if not docs or not terms:
            return []
        avg_len = sum(sum(d.values()) for d in docs.values()) / len(docs)
        n = len(docs)
        scores: dict[str, float] = {}
        for term in set(terms):
            df = sum(1 for d in docs.values() if term in d)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for cid, counts in docs.items():
                tf = counts.get(term, 0)
                if tf:
                    length = sum(counts.values())
                    scores[cid] = scores.get(cid, 0.0) + idf * tf * 2.2 / (
                        tf + 1.2 * (0.25 + 0.75 * length / avg_len)
                    )
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    def _cosine(self, vector: list[float], candidates: list[Chunk]) -> list[tuple[str, float]]:
        scored = []
        for chunk in candidates:
            other = self._vectors[chunk.chunk_id]
            scored.append((chunk.chunk_id, sum(a * b for a, b in zip(vector, other, strict=True))))
        return sorted(scored, key=lambda kv: kv[1], reverse=True)

    async def search(
        self,
        *,
        query_text: str,
        query_vector: list[float] | None,
        search_filter: SearchFilter,
        top: int,
        mode: RetrievalMode,
    ) -> list[Chunk]:
        candidates = [c for c in self._chunks.values() if search_filter.matches(c)]
        keyword = self._bm25(query_text, candidates) if mode is not RetrievalMode.VECTOR else []
        vector = (
            self._cosine(query_vector, candidates)[: max(top * 4, 20)]
            if mode is not RetrievalMode.KEYWORD and query_vector is not None
            else []
        )
        if mode is RetrievalMode.KEYWORD:
            ranked = keyword
        elif mode is RetrievalMode.VECTOR:
            ranked = vector
        else:
            fused: dict[str, float] = {}
            for ranking in (keyword, vector):
                for rank, (cid, _) in enumerate(ranking, start=1):
                    fused[cid] = fused.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
            ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        return [self._chunks[cid].model_copy(update={"score": score}) for cid, score in ranked[:top]]
