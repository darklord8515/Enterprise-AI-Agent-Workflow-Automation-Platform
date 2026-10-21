"""Sentence-aware chunking with overlap and citation metadata.

Approach adapted from the ``SentenceTextSplitter`` in Azure-Samples/azure-search-openai-demo (MIT):
split on sentence boundaries, pack sentences up to a size limit, and carry a ~10% tail overlap into the
next chunk. Simplified here to character limits and to never cross page/section boundaries, so every
chunk maps to exactly one citable page or section.
"""

from __future__ import annotations

import re

from ..models import Chunk, DocumentRecord
from ..security import looks_like_prompt_injection
from .parsers import Section

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n{2,}|\n(?=\s*[-*#]|\s*\d+[.)])")


def _sentences(text: str) -> list[str]:
    return [s for s in (p.strip() for p in _SENTENCE_END.split(text)) if s]


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    return [sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars)]


def chunk_sections(
    record: DocumentRecord, sections: list[Section], max_chars: int, overlap_chars: int
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        pieces: list[str] = []
        for sentence in _sentences(section.text):
            pieces.extend(_hard_split(sentence, max_chars) if len(sentence) > max_chars else [sentence])

        current: list[str] = []
        length = 0
        for piece in pieces:
            if current and length + len(piece) + 1 > max_chars:
                chunks.append(_make_chunk(record, section, current, len(chunks)))
                tail: list[str] = []
                tail_len = 0
                for prev in reversed(current):
                    if tail_len + len(prev) > overlap_chars:
                        break
                    tail.insert(0, prev)
                    tail_len += len(prev) + 1
                current, length = tail, tail_len
            current.append(piece)
            length += len(piece) + 1
        if current:
            chunks.append(_make_chunk(record, section, current, len(chunks)))
    return chunks


def _make_chunk(record: DocumentRecord, section: Section, sentences: list[str], index: int) -> Chunk:
    content = "\n".join(sentences)
    return Chunk(
        chunk_id=f"{record.document_id}-{index:04d}",
        document_id=record.document_id,
        source_file=record.filename,
        category=record.category,
        supplier_id=record.supplier_id,
        owner=record.owner,
        content=content,
        page=section.page,
        section=section.section,
        chunk_index=index,
        suspicious=looks_like_prompt_injection(content),
    )
