"""Workflow nodes. Each node is a single responsibility step in one LangGraph graph; they are not
independent autonomous agents.

planner -> retriever -> analyst -> evidence_reviewer -> router
router  -> report_generator | retriever (bounded revision) | insufficient_evidence -> report_generator
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..config import Settings
from ..errors import AppError, WorkflowError
from ..models import (
    Assessment,
    Chunk,
    DocumentCategory,
    EvidenceItem,
    Finding,
    FindingStatus,
    Plan,
    PlannedCheck,
    Report,
    RequirementAssessment,
    TraceEvent,
    Usage,
)
from ..providers.base import ChatProvider, EmbeddingProvider
from ..retrieval.base import SearchBackend, SearchFilter
from . import prompts
from .state import WorkflowState

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This report is an evidence-assistance output generated from the documents provided. It is not a "
    "legal or regulatory compliance determination. A qualified person must review the findings and "
    "cited sources before any procurement decision."
)


@dataclass
class WorkflowDeps:
    settings: Settings
    chat: ChatProvider
    embeddings: EmbeddingProvider
    search: SearchBackend


NodeFn = Callable[[WorkflowState], Awaitable[dict[str, Any]]]


class NodeExecutionError(WorkflowError):
    """Raised when a node fails; carries the node name and the underlying typed error."""

    def __init__(self, node: str, cause: Exception) -> None:
        inner = (
            cause.to_dict()
            if isinstance(cause, AppError)
            else {
                "code": "internal_error",
                "message": type(cause).__name__,
                "retryable": False,
            }
        )
        super().__init__(f"Node '{node}' failed: {inner['message']}", retryable=inner["retryable"])
        self.node = node
        self.inner = inner

    def to_dict(self) -> dict:
        return {**self.inner, "failed_node": self.node}


def traced(name: str, fn: NodeFn) -> NodeFn:
    """Wrap a node with timing, a trace event and structured logging. Nodes may return a ``_detail``
    string that is recorded in the trace."""

    async def wrapper(state: WorkflowState) -> dict[str, Any]:
        started_at = datetime.now(UTC)
        t0 = time.perf_counter()
        try:
            update = await fn(state)
        except Exception as exc:
            logger.exception("workflow.node_failed", extra={"node": name})
            raise NodeExecutionError(name, exc) from exc
        duration = round((time.perf_counter() - t0) * 1000, 2)
        detail = update.pop("_detail", "")
        logger.info(
            "workflow.node_completed", extra={"node": name, "duration_ms": duration, "detail": detail}
        )
        update["trace"] = [TraceEvent(node=name, started_at=started_at, duration_ms=duration, detail=detail)]
        return update

    return wrapper


def _normalize(text: str) -> str:
    text = text.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[*_`]", "", text)
    return re.sub(r"\s+", " ", text).strip(" .\"'")


def _suspicious_warnings(chunks: list[Chunk]) -> list[str]:
    return [
        f"Chunk {c.chunk_id} ({c.source_file}) contains text resembling instructions to an AI system. "
        "It was treated as data only; review the source document manually."
        for c in chunks
        if c.suspicious
    ]


def build_nodes(deps: WorkflowDeps) -> dict[str, NodeFn]:
    settings = deps.settings

    def supplier_filter(state: WorkflowState) -> SearchFilter:
        doc_ids = tuple(state["document_ids"]) if state.get("document_ids") else None
        return SearchFilter(
            owner=state["owner"],
            category=DocumentCategory.SUPPLIER,
            supplier_id=state["supplier_id"],
            document_ids=doc_ids,
        )

    def policy_filter(state: WorkflowState) -> SearchFilter:
        doc_ids = tuple(state["document_ids"]) if state.get("document_ids") else None
        return SearchFilter(owner=state["owner"], category=DocumentCategory.POLICY, document_ids=doc_ids)

    # -- 1. Planner -------------------------------------------------------------------------------
    async def planner(state: WorkflowState) -> dict[str, Any]:
        explicit = state.get("explicit_requirements")
        policy_chunks: list[Chunk] = []
        usage = Usage()
        if not explicit:
            query = f"{state['task']} supplier requirements must"
            vectors, tokens = await deps.embeddings.embed([query])
            usage.embedding_tokens += tokens
            policy_chunks = await deps.search.search(
                query_text=query,
                query_vector=vectors[0],
                search_filter=policy_filter(state),
                top=20,
                mode=settings.retrieval_mode,
            )
            if not policy_chunks:
                raise WorkflowError(
                    "No requirements available: pass 'requirements' explicitly or upload and index "
                    "company policy documents."
                )
        result = await deps.chat.generate_structured(
            task_name="plan",
            system_prompt=prompts.PLANNER_SYSTEM,
            user_prompt=prompts.planner_user_prompt(state["task"], explicit, policy_chunks),
            response_model=Plan,
            context={
                "task": state["task"],
                "explicit_requirements": explicit,
                "policy_chunks": [c.model_dump(mode="json") for c in policy_chunks],
            },
        )
        usage.prompt_tokens += result.usage.prompt_tokens
        usage.completion_tokens += result.usage.completion_tokens
        usage.llm_calls += 1

        checks: list[PlannedCheck] = []
        seen: set[str] = set()
        for check in result.value.checks:
            rid = check.requirement_id.strip() or f"REQ-{len(checks) + 1:03d}"
            if rid in seen or not check.requirement.strip():
                continue
            seen.add(rid)
            checks.append(check.model_copy(update={"requirement_id": rid}))
        if not checks:
            raise WorkflowError("The planner produced no checkable requirements.")
        warnings = _suspicious_warnings(policy_chunks)
        if len(checks) > settings.max_requirements:
            warnings.append(
                f"Plan truncated from {len(checks)} to {settings.max_requirements} requirements (MAX_REQUIREMENTS)."
            )
            checks = checks[: settings.max_requirements]
        return {
            "checks": checks,
            "targets": [c.requirement_id for c in checks],
            "revision_count": 0,
            "usage": usage,
            "warnings": warnings,
            "_detail": f"{len(checks)} checks ({'explicit' if explicit else 'extracted from policy'})",
        }

    # -- 2. Retriever -----------------------------------------------------------------------------
    async def retriever(state: WorkflowState) -> dict[str, Any]:
        revision = state.get("revision_count", 0)
        checks = [c for c in state["checks"] if c.requirement_id in state["targets"]]
        # On a revision pass, broaden the query and the result count.
        queries = [c.search_query if revision == 0 else f"{c.requirement} {c.search_query}" for c in checks]
        top = settings.retrieval_top_k * (2 if revision else 1)
        vectors, tokens = await deps.embeddings.embed(queries)
        evidence: dict[str, list[Chunk]] = {}
        all_chunks: list[Chunk] = []
        for check, query, vector in zip(checks, queries, vectors, strict=True):
            supplier = await deps.search.search(
                query_text=query,
                query_vector=vector,
                search_filter=supplier_filter(state),
                top=top,
                mode=settings.retrieval_mode,
            )
            policy = await deps.search.search(
                query_text=check.requirement,
                query_vector=vector,
                search_filter=policy_filter(state),
                top=2,
                mode=settings.retrieval_mode,
            )
            evidence[check.requirement_id] = supplier + policy
            all_chunks.extend(supplier + policy)
        seen: set[str] = set()
        unique = [c for c in all_chunks if not (c.chunk_id in seen or seen.add(c.chunk_id))]
        empty = sum(
            1 for cid in evidence if not any(c.category is DocumentCategory.SUPPLIER for c in evidence[cid])
        )
        return {
            "evidence": evidence,
            "usage": Usage(embedding_tokens=tokens),
            "warnings": _suspicious_warnings(unique),
            "_detail": f"pass {revision}: {len(unique)} chunks for {len(checks)} checks; "
            f"{empty} with no supplier evidence; mode={settings.retrieval_mode.value}",
        }

    # -- 3. Analyst -------------------------------------------------------------------------------
    async def analyst(state: WorkflowState) -> dict[str, Any]:
        checks = [c for c in state["checks"] if c.requirement_id in state["targets"]]
        items = [
            {
                "requirement_id": c.requirement_id,
                "requirement": c.requirement,
                "chunks": state["evidence"].get(c.requirement_id, []),
            }
            for c in checks
        ]
        feedback = state.get("review_issues") or {}
        result = await deps.chat.generate_structured(
            task_name="analyze",
            system_prompt=prompts.ANALYST_SYSTEM,
            user_prompt=prompts.analyst_user_prompt(items, feedback),
            response_model=Assessment,
            context={
                "requirements": [
                    {
                        "requirement_id": i["requirement_id"],
                        "requirement": i["requirement"],
                        "evidence": [c.model_dump(mode="json") for c in i["chunks"]],
                    }
                    for i in items
                ],
                "feedback": feedback,
            },
        )
        by_id = {a.requirement_id: a for a in result.value.assessments}
        assessments: dict[str, RequirementAssessment] = {}
        missing = 0
        for check in checks:
            if check.requirement_id in by_id:
                assessments[check.requirement_id] = by_id[check.requirement_id]
            else:
                missing += 1
                assessments[check.requirement_id] = RequirementAssessment(
                    requirement_id=check.requirement_id,
                    status=FindingStatus.INSUFFICIENT,
                    evidence=[],
                    explanation="The analysis step returned no assessment for this requirement.",
                    missing_information=["A completed assessment for this requirement."],
                    recommended_follow_up=["Re-run the evaluation or review manually."],
                )
        return {
            "assessments": assessments,
            "usage": Usage(
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                llm_calls=1,
            ),
            "_detail": f"{len(assessments)} assessed, {missing} missing from model output",
        }

    # -- 4. Evidence reviewer (deterministic) -----------------------------------------------------
    async def evidence_reviewer(state: WorkflowState) -> dict[str, Any]:
        checks = {c.requirement_id: c for c in state["checks"]}
        findings: dict[str, Finding] = {}
        blocking: dict[str, list[str]] = {}
        for rid in state["targets"]:
            assessment = state["assessments"][rid]
            retrieved = {c.chunk_id: c for c in state["evidence"].get(rid, [])}
            notes: list[str] = []
            valid: list[EvidenceItem] = []
            for cited in assessment.evidence:
                chunk = retrieved.get(cited.chunk_id)
                if chunk is None:
                    notes.append(f"Citation {cited.chunk_id} does not refer to retrieved evidence; removed.")
                    continue
                if not cited.quote.strip() or _normalize(cited.quote) not in _normalize(chunk.content):
                    notes.append(
                        f"Quote attributed to {cited.chunk_id} was not found verbatim in that source; removed."
                    )
                    continue
                if (
                    chunk.category is DocumentCategory.POLICY
                    and assessment.status is not FindingStatus.INSUFFICIENT
                ):
                    notes.append(
                        f"{cited.chunk_id} is company policy text and cannot evidence supplier compliance; removed."
                    )
                    continue
                if chunk.suspicious:
                    notes.append(
                        f"Cited source {cited.chunk_id} was flagged for possible prompt injection; verify manually."
                    )
                valid.append(
                    EvidenceItem(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        source_file=chunk.source_file,
                        category=chunk.category,
                        page=chunk.page,
                        section=chunk.section,
                        quote=cited.quote.strip(),
                    )
                )
            if assessment.status is not FindingStatus.INSUFFICIENT and not any(
                e.category is DocumentCategory.SUPPLIER for e in valid
            ):
                blocking[rid] = notes + [
                    f"Status '{assessment.status.value}' is not supported by any verifiable supplier citation."
                ]
            findings[rid] = Finding(
                requirement_id=rid,
                requirement=checks[rid].requirement,
                status=assessment.status,
                supporting_evidence=valid,
                explanation=assessment.explanation,
                missing_information=assessment.missing_information,
                recommended_follow_up=assessment.recommended_follow_up,
                review_notes=notes,
            )
        return {
            "findings": findings,
            "review_issues": blocking,
            "_detail": f"{len(findings)} reviewed, {len(blocking)} unsupported",
        }

    # -- 5. Router --------------------------------------------------------------------------------
    async def router(state: WorkflowState) -> dict[str, Any]:
        failing = list(state.get("review_issues") or {})
        revision = state.get("revision_count", 0)
        if not failing:
            return {"route": "report", "_detail": "all findings supported -> report"}
        if revision < settings.max_revisions:
            return {
                "route": "revise",
                "targets": failing,
                "revision_count": revision + 1,
                "_detail": f"{len(failing)} unsupported -> revision {revision + 1}/{settings.max_revisions}",
            }
        return {
            "route": "insufficient",
            "targets": failing,
            "_detail": f"{len(failing)} still unsupported after {revision} revision(s) -> insufficient evidence",
        }

    # -- 5b. Insufficient-evidence result ---------------------------------------------------------
    async def insufficient_evidence(state: WorkflowState) -> dict[str, Any]:
        downgraded: dict[str, Finding] = {}
        for rid in state["targets"]:
            finding = state["findings"][rid]
            downgraded[rid] = finding.model_copy(
                update={
                    "status": FindingStatus.INSUFFICIENT,
                    "explanation": (
                        f"The analysis concluded '{finding.status.value}', but that conclusion could not be "
                        "verified against the retrieved sources, so it is reported as insufficient evidence. "
                        f"Original explanation: {finding.explanation}"
                    ),
                    "missing_information": finding.missing_information
                    or ["Verifiable supplier documentation addressing this requirement."],
                    "recommended_follow_up": [
                        "Request explicit documentation from the supplier.",
                        "Have a reviewer check the source documents manually.",
                    ],
                    "review_notes": state["review_issues"].get(rid, finding.review_notes),
                }
            )
        return {"findings": downgraded, "_detail": f"{len(downgraded)} downgraded to insufficient_evidence"}

    # -- 6. Report generator ----------------------------------------------------------------------
    async def report_generator(state: WorkflowState) -> dict[str, Any]:
        findings = [state["findings"][c.requirement_id] for c in state["checks"]]
        summary = {s.value: sum(1 for f in findings if f.status is s) for s in FindingStatus}
        trace = state.get("trace", [])
        latency = (datetime.now(UTC) - trace[0].started_at).total_seconds() * 1000 if trace else 0.0
        report = Report(
            supplier_id=state["supplier_id"],
            task=state["task"],
            summary=summary,
            findings=findings,
            warnings=list(dict.fromkeys(state.get("warnings", []))),
            disclaimer=DISCLAIMER,
            provider_mode=f"{deps.chat.name}/{deps.embeddings.name}/{deps.search.name}",
            revisions=state.get("revision_count", 0),
            latency_ms=round(latency, 1),
            usage=state.get("usage") or Usage(),
            trace=trace,
        )
        return {"report": report, "_detail": ", ".join(f"{k}={v}" for k, v in summary.items())}

    return {
        "planner": planner,
        "retriever": retriever,
        "analyst": analyst,
        "evidence_reviewer": evidence_reviewer,
        "router": router,
        "insufficient_evidence": insufficient_evidence,
        "report_generator": report_generator,
    }


__all__ = ["DISCLAIMER", "NodeExecutionError", "WorkflowDeps", "build_nodes", "traced"]
