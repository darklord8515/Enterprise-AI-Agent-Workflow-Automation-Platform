"""Typed LangGraph state for the supplier-evaluation workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from ..models import Chunk, Finding, PlannedCheck, Report, RequirementAssessment, TraceEvent, Usage

Route = Literal["report", "revise", "insufficient"]


def add_usage(left: Usage | None, right: Usage | None) -> Usage:
    """Reducer: nodes return usage *deltas* which are summed into the running total."""
    left = left or Usage()
    if right is None:
        return left
    return Usage(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
        embedding_tokens=left.embedding_tokens + right.embedding_tokens,
        llm_calls=left.llm_calls + right.llm_calls,
    )


def merge_dict(left: dict | None, right: dict | None) -> dict:
    """Reducer: later values for the same key replace earlier ones (used across revisions)."""
    return {**(left or {}), **(right or {})}


class WorkflowState(TypedDict, total=False):
    # Inputs
    task: str
    supplier_id: str
    owner: str
    explicit_requirements: list[str] | None
    document_ids: list[str] | None

    # Planner
    checks: list[PlannedCheck]

    # Retriever / analyst / reviewer, keyed by requirement_id
    evidence: Annotated[dict[str, list[Chunk]], merge_dict]
    assessments: Annotated[dict[str, RequirementAssessment], merge_dict]
    findings: Annotated[dict[str, Finding], merge_dict]
    review_issues: dict[str, list[str]]  # blocking issues from the latest review only

    # Routing
    targets: list[str]  # requirement_ids processed in the current pass
    revision_count: int
    route: Route

    # Observability
    warnings: Annotated[list[str], operator.add]
    trace: Annotated[list[TraceEvent], operator.add]
    usage: Annotated[Usage, add_usage]

    # Output
    report: Report
