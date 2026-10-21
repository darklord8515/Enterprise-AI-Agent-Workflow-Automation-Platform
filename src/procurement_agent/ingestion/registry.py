"""Document metadata registry and raw file storage on local disk.

Single-process only (a lock guards a JSON file). A multi-instance deployment would replace this with a
database and Blob Storage; see README "Limitations".
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from ..errors import NotFound
from ..models import DocumentRecord


class DocumentRegistry:
    def __init__(self, data_dir: Path) -> None:
        self.files_dir = data_dir / "uploads"
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self._path = data_dir / "documents.json"
        self._lock = threading.Lock()
        self._records: dict[str, DocumentRecord] = {}
        if self._path.exists():
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._records = {k: DocumentRecord.model_validate(v) for k, v in raw.items()}

    def _persist(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({k: v.model_dump(mode="json") for k, v in self._records.items()}, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self._path)

    def add(self, record: DocumentRecord, data: bytes) -> None:
        with self._lock:
            (self.files_dir / record.document_id).write_bytes(data)
            self._records[record.document_id] = record
            self._persist()

    def update(self, record: DocumentRecord) -> None:
        with self._lock:
            self._records[record.document_id] = record
            self._persist()

    def get(self, document_id: str, owner: str) -> DocumentRecord:
        """Return a document only if it belongs to ``owner`` (no existence leak across owners)."""
        record = self._records.get(document_id)
        if record is None or record.owner != owner:
            raise NotFound(f"Document '{document_id}' not found")
        return record

    def read_bytes(self, record: DocumentRecord) -> bytes:
        return (self.files_dir / record.document_id).read_bytes()

    def list(self, owner: str) -> list[DocumentRecord]:
        return [r for r in self._records.values() if r.owner == owner]
