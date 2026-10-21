"""Pydantic schemas shared by the API, workflow and providers."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


def utcnow() -> datetime:
    return datetime.now(UTC)


class DocumentCategory(StrEnum):
    SUPPLIER = "supplier"
    POLICY = "policy"


class FindingStatus(StrEnum):
    MET = "met"
    GAP = "gap_identified"
    INSUFFICIENT = "insufficient_evidence"


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# --- Documents --------------------------------------------------------------------------------


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    category: DocumentCategory
    supplier_id: str | None = None
    owner: str
    size_bytes: int
    sha256: str
    content_type: str
    uploaded_at: datetime = Field(default_factory=utcnow)
    indexed: bool = False
    chunk_count: int = 0


class Chunk(BaseModel):
    """A retrievable unit of text with the metadata needed for citations and access control."""

    chunk_id: str
    document_id: str
    source_file: str
    category: DocumentCategory
    supplier_id: str | None = None
    owner: str
    content: str
    page: int | None = None
    section: str | None = None
    chunk_index: int = 0
    suspicious: bool = False
    score: float | None = None


class IndexRequest(BaseModel):
    document_ids: list[str] = Field(min_length=1, max_length=50)


class IndexResponse(BaseModel):
    indexed: list[DocumentRecord]
    provider_mode: str


# --- Workflow structured outputs --------------------------------------------------------------


class PlannedCheck(BaseModel):
    requirement_id: str = Field(description="Stable identifier, e.g. REQ-001")
    requirement: str = Field(description="The requirement the supplier must satisfy")
    search_query: str = Field(description="Query used to find supplier evidence for this requirement")


class Plan(BaseModel):
    checks: list[PlannedCheck]


class CitedEvidence(BaseModel):
    chunk_id: str = Field(description="ID of the retrieved chunk this quote comes from")
    quote: str = Field(description="Verbatim quote copied from that chunk")


class RequirementAssessment(BaseModel):
    requirement_id: str
    status: FindingStatus
    evidence: list[CitedEvidence]
    explanation: str
    missing_information: list[str]
    recommended_follow_up: list[str]


class Assessment(BaseModel):
    assessments: list[RequirementAssessment]


# --- Report -----------------------------------------------------------------------------------


class EvidenceItem(BaseModel):
    chunk_id: str
    document_id: str
    source_file: str
    category: DocumentCategory
    page: int | None = None
    section: str | None = None
    quote: str


class Finding(BaseModel):
    requirement_id: str
    requirement: str
    status: FindingStatus
    supporting_evidence: list[EvidenceItem]
    explanation: str
    missing_information: list[str]
    recommended_follow_up: list[str]
    review_notes: list[str] = Field(default_factory=list)


class TraceEvent(BaseModel):
    node: str
    started_at: datetime
    duration_ms: float
    detail: str = ""


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    llm_calls: int = 0


class Report(BaseModel):
    supplier_id: str | None
    task: str
    summary: dict[str, int]
    findings: list[Finding]
    warnings: list[str]
    disclaimer: str
    provider_mode: str
    revisions: int
    latency_ms: float
    usage: Usage
    trace: list[TraceEvent]


# --- Tasks ------------------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    task: str = Field(min_length=5, description="Natural-language description of the evaluation")
    supplier_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")
    requirements: list[str] | None = Field(
        default=None,
        description="Optional explicit requirements. If omitted, they are extracted from policy documents.",
    )
    document_ids: list[str] | None = Field(
        default=None, description="Optional restriction to specific indexed documents."
    )

    @field_validator("requirements")
    @classmethod
    def _strip_requirements(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        cleaned = [r.strip() for r in v if r and r.strip()]
        if any(len(r) > 500 for r in cleaned):
            raise ValueError("each requirement must be at most 500 characters")
        return cleaned or None


class TaskRecord(BaseModel):
    task_id: str
    owner: str
    status: TaskStatus
    request: AnalyzeRequest
    created_at: datetime = Field(default_factory=utcnow)
    completed_at: datetime | None = None
    report: Report | None = None
    error: dict[str, Any] | None = None


class TaskAccepted(BaseModel):
    task_id: str
    status: TaskStatus


class HealthResponse(BaseModel):
    status: str
    version: str
    provider_mode: str
    retrieval_mode: str
    auth_enabled: bool
