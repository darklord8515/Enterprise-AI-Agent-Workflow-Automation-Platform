"""Evaluation harness: ingest the dataset, run every case through the workflow and score it.

Metrics (all computed from actual runs; nothing is estimated):
* retrieval_recall: share of requirements with an expected evidence file where a chunk from that file
  was retrieved for the requirement.
* classification_accuracy: predicted status == expected status.
* citation_precision: share of reported citations whose quote is found verbatim in the cited chunk.
* expected_evidence_hit: share of met/gap expectations where a citation comes from the expected file
  and contains the expected text.
* unsupported_claims: met/gap findings in the final report without a supplier citation (should be 0).
* citations_removed_by_reviewer: invalid citations the evidence reviewer caught before reporting.
* insufficient_evidence_recall: expected insufficient_evidence predicted as such.
* forbidden_evidence_leaks: citations from another supplier's documents (access/filter check).
* injection_warning_ok: prompt-injection warning raised exactly when expected.
* workflow_completion, latency and token usage per case.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .container import Container
from .models import DocumentCategory, FindingStatus
from .workflow.graph import run_workflow


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


@dataclass
class CaseResult:
    case_id: str
    completed: bool
    error: str | None = None
    latency_ms: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)
    injection_warning_ok: bool | None = None
    forbidden_evidence_leaks: int = 0
    citations_total: int = 0
    citations_verbatim: int = 0
    citations_removed_by_reviewer: int = 0
    unsupported_claims: int = 0
    revisions: int = 0


async def ingest_dataset(container: Container, dataset_dir: Path, owner: str) -> dict[str, str]:
    """Upload and index all dataset documents; returns filename -> document_id."""
    spec = json.loads((dataset_dir / "cases.json").read_text(encoding="utf-8"))
    ids: dict[str, str] = {}
    for doc in spec["documents"]:
        record = container.ingestion.upload(
            filename=doc["file"],
            data=(dataset_dir / "documents" / doc["file"]).read_bytes(),
            category=DocumentCategory(doc["category"]),
            supplier_id=doc.get("supplier_id"),
            owner=owner,
        )
        ids[doc["file"]] = record.document_id
    await container.ingestion.index(list(ids.values()), owner)
    return ids


async def run_case(container: Container, case: dict[str, Any], owner: str) -> CaseResult:
    result = CaseResult(case_id=case["case_id"], completed=False)
    started = time.perf_counter()
    try:
        state = await run_workflow(
            container.graph,
            task=case["task"],
            supplier_id=case["supplier_id"],
            owner=owner,
            requirements=case.get("requirements"),
        )
    except Exception as exc:  # noqa: BLE001 - recorded as a failed case
        result.error = f"{type(exc).__name__}: {exc}"
        result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return result

    report = state["report"]
    result.completed = True
    result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
    result.usage = report.usage.model_dump()
    result.revisions = report.revisions
    findings = {f.requirement_id: f for f in report.findings}
    chunk_text = {c.chunk_id: c.content for chunks in state["evidence"].values() for c in chunks}

    for finding in report.findings:
        result.citations_removed_by_reviewer += sum("removed" in n for n in finding.review_notes)
        if finding.status is not FindingStatus.INSUFFICIENT and not any(
            e.category is DocumentCategory.SUPPLIER for e in finding.supporting_evidence
        ):
            result.unsupported_claims += 1
        for ev in finding.supporting_evidence:
            result.citations_total += 1
            if _norm(ev.quote) in _norm(chunk_text.get(ev.chunk_id, "")):
                result.citations_verbatim += 1
            if ev.source_file in case.get("forbidden_evidence_files", []):
                result.forbidden_evidence_leaks += 1

    for exp in case["expected"]:
        rid = exp["requirement_id"]
        finding = findings.get(rid)
        retrieved_files = {c.source_file for c in state["evidence"].get(rid, [])}
        row: dict[str, Any] = {
            "requirement_id": rid,
            "expected": exp["status"],
            "predicted": finding.status.value if finding else None,
            "correct": bool(finding and finding.status.value == exp["status"]),
            "retrieved_expected_file": (exp["evidence_file"] in retrieved_files)
            if exp["evidence_file"]
            else None,
            "evidence_hit": None,
        }
        if exp["evidence_file"] and exp["status"] != FindingStatus.INSUFFICIENT.value and finding:
            row["evidence_hit"] = any(
                ev.source_file == exp["evidence_file"] and _norm(exp["evidence_contains"]) in _norm(ev.quote)
                for ev in finding.supporting_evidence
            )
        result.rows.append(row)

    has_warning = any("instructions to an AI system" in w for w in report.warnings)
    result.injection_warning_ok = has_warning == bool(case.get("expect_injection_warning"))
    return result


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 3) if den else None


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    rows = [r for c in results for r in c.rows]
    retrieval = [r["retrieved_expected_file"] for r in rows if r["retrieved_expected_file"] is not None]
    hits = [r["evidence_hit"] for r in rows if r["evidence_hit"] is not None]
    insufficient = [r for r in rows if r["expected"] == FindingStatus.INSUFFICIENT.value]
    return {
        "cases": len(results),
        "workflow_completion": _ratio(sum(c.completed for c in results), len(results)),
        "requirements_evaluated": len(rows),
        "classification_accuracy": _ratio(sum(r["correct"] for r in rows), len(rows)),
        "retrieval_recall": _ratio(sum(retrieval), len(retrieval)),
        "citation_precision": _ratio(
            sum(c.citations_verbatim for c in results), sum(c.citations_total for c in results)
        ),
        "expected_evidence_hit": _ratio(sum(hits), len(hits)),
        "insufficient_evidence_recall": _ratio(
            sum(r["predicted"] == FindingStatus.INSUFFICIENT.value for r in insufficient), len(insufficient)
        ),
        "unsupported_claims": sum(c.unsupported_claims for c in results),
        "citations_removed_by_reviewer": sum(c.citations_removed_by_reviewer for c in results),
        "forbidden_evidence_leaks": sum(c.forbidden_evidence_leaks for c in results),
        "injection_warning_ok": all(
            c.injection_warning_ok for c in results if c.injection_warning_ok is not None
        ),
        "mean_latency_ms": round(sum(c.latency_ms for c in results) / len(results), 1) if results else None,
        "total_prompt_tokens": sum(c.usage.get("prompt_tokens", 0) for c in results),
        "total_completion_tokens": sum(c.usage.get("completion_tokens", 0) for c in results),
        "total_embedding_tokens": sum(c.usage.get("embedding_tokens", 0) for c in results),
    }


async def run_evaluation(container: Container, dataset_dir: Path) -> dict[str, Any]:
    # A fresh owner per run isolates eval documents from anything else in the index.
    owner = f"eval-{uuid.uuid4().hex[:8]}"
    spec = json.loads((dataset_dir / "cases.json").read_text(encoding="utf-8"))
    ingest_started = time.perf_counter()
    await ingest_dataset(container, dataset_dir, owner)
    ingest_ms = round((time.perf_counter() - ingest_started) * 1000, 1)
    results = [await run_case(container, case, owner) for case in spec["cases"]]
    return {
        "provider_mode": container.settings.provider_mode.value,
        "providers": {
            "chat": container.chat.name,
            "embeddings": container.embeddings.name,
            "search": container.search.name,
        },
        "retrieval_mode": container.settings.retrieval_mode.value,
        "ingest_ms": ingest_ms,
        "summary": summarize(results),
        "cases": [r.__dict__ for r in results],
    }
