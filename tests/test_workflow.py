from __future__ import annotations

import pytest

from procurement_agent.container import build_container
from procurement_agent.errors import ProviderRateLimitError
from procurement_agent.models import FindingStatus
from procurement_agent.workflow.graph import run_workflow
from procurement_agent.workflow.nodes import NodeExecutionError

from .conftest import ScriptedChatProvider, ingest_files, make_settings

TASK = "Evaluate Halvorsen against the procurement policy."
ISO_REQ = "The supplier must hold a current ISO 9001 quality management certification."


def _first_supplier_chunk(context):
    item = context["requirements"][0]
    return next(c for c in item["evidence"] if c["category"] == "supplier")


async def test_happy_path_extracts_requirements_and_reports(container):
    await ingest_files(container)
    state = await run_workflow(container.graph, task=TASK, supplier_id="halvorsen", owner="tester")
    report = state["report"]

    assert [f.requirement_id for f in report.findings] == [f"REQ-{i:03d}" for i in range(1, 9)]
    status = {f.requirement_id: f.status for f in report.findings}
    assert status["REQ-001"] is FindingStatus.MET
    assert status["REQ-003"] is FindingStatus.GAP  # £2m cover vs £5m required
    assert status["REQ-004"] is FindingStatus.INSUFFICIENT  # never "met" by absence
    for finding in report.findings:
        if finding.status is not FindingStatus.INSUFFICIENT:
            assert finding.supporting_evidence, finding.requirement_id
            assert all(e.category.value == "supplier" for e in finding.supporting_evidence)
    assert [t.node for t in report.trace] == [
        "planner",
        "retriever",
        "analyst",
        "evidence_reviewer",
        "router",
        "report_generator",
    ]
    assert report.disclaimer and "not a legal" in report.disclaimer


async def test_fabricated_citation_triggers_bounded_revision_then_succeeds(tmp_path):
    def analyze(context, call_no):
        item = context["requirements"][0]
        if call_no == 1:  # hallucinated chunk id and quote
            evidence = [{"chunk_id": "made-up-0001", "quote": "Certified to ISO 9001."}]
        else:
            chunk = _first_supplier_chunk(context)
            evidence = [{"chunk_id": chunk["chunk_id"], "quote": chunk["content"].split("\n")[0]}]
        return {
            "assessments": [
                {
                    "requirement_id": item["requirement_id"],
                    "status": "met",
                    "evidence": evidence,
                    "explanation": "x",
                    "missing_information": [],
                    "recommended_follow_up": [],
                }
            ]
        }

    chat = ScriptedChatProvider(analyze)
    container = build_container(make_settings(tmp_path, max_revisions=1), chat=chat)
    await ingest_files(container)
    state = await run_workflow(
        container.graph, task=TASK, supplier_id="halvorsen", owner="tester", requirements=[ISO_REQ]
    )
    report = state["report"]
    assert len(chat.analyze_calls) == 2
    assert chat.analyze_calls[1]["feedback"]["REQ-001"]  # reviewer feedback passed to the revision
    assert report.revisions == 1
    assert report.findings[0].status is FindingStatus.MET
    assert [t.node for t in report.trace].count("retriever") == 2


async def test_persistently_unsupported_claim_becomes_insufficient_evidence(tmp_path):
    def analyze(context, call_no):
        item = context["requirements"][0]
        chunk = _first_supplier_chunk(context)
        return {
            "assessments": [
                {
                    "requirement_id": item["requirement_id"],
                    "status": "met",
                    "evidence": [
                        {"chunk_id": chunk["chunk_id"], "quote": "This sentence is not in the document."}
                    ],
                    "explanation": "Claimed met.",
                    "missing_information": [],
                    "recommended_follow_up": [],
                }
            ]
        }

    chat = ScriptedChatProvider(analyze)
    container = build_container(make_settings(tmp_path, max_revisions=2), chat=chat)
    await ingest_files(container)
    state = await run_workflow(
        container.graph, task=TASK, supplier_id="halvorsen", owner="tester", requirements=[ISO_REQ]
    )
    finding = state["report"].findings[0]
    assert len(chat.analyze_calls) == 3  # initial + 2 bounded revisions, then stop
    assert finding.status is FindingStatus.INSUFFICIENT
    assert finding.supporting_evidence == []
    assert any("not found verbatim" in n for n in finding.review_notes)
    assert [t.node for t in state["trace"]][-2:] == ["insufficient_evidence", "report_generator"]


async def test_policy_text_cannot_prove_supplier_compliance(tmp_path):
    def analyze(context, call_no):
        item = context["requirements"][0]
        policy = next(c for c in item["evidence"] if c["category"] == "policy")
        return {
            "assessments": [
                {
                    "requirement_id": item["requirement_id"],
                    "status": "met",
                    "evidence": [{"chunk_id": policy["chunk_id"], "quote": policy["content"].split("\n")[0]}],
                    "explanation": "x",
                    "missing_information": [],
                    "recommended_follow_up": [],
                }
            ]
        }

    container = build_container(make_settings(tmp_path, max_revisions=0), chat=ScriptedChatProvider(analyze))
    await ingest_files(container)
    state = await run_workflow(
        container.graph, task=TASK, supplier_id="halvorsen", owner="tester", requirements=[ISO_REQ]
    )
    finding = state["report"].findings[0]
    assert finding.status is FindingStatus.INSUFFICIENT
    assert any("company policy text" in n for n in finding.review_notes)


async def test_missing_assessment_is_reported_as_insufficient(tmp_path):
    container = build_container(
        make_settings(tmp_path), chat=ScriptedChatProvider(lambda ctx, n: {"assessments": []})
    )
    await ingest_files(container)
    state = await run_workflow(
        container.graph, task=TASK, supplier_id="halvorsen", owner="tester", requirements=[ISO_REQ]
    )
    assert state["report"].findings[0].status is FindingStatus.INSUFFICIENT


async def test_no_requirements_and_no_policy_fails_clearly(container):
    await ingest_files(container, files=[("halvorsen_certifications.md", "supplier", "halvorsen")])
    with pytest.raises(NodeExecutionError) as info:
        await run_workflow(container.graph, task=TASK, supplier_id="halvorsen", owner="tester")
    assert info.value.node == "planner"
    assert "No requirements available" in info.value.to_dict()["message"]


async def test_supplier_with_no_documents_yields_insufficient_not_met(container):
    await ingest_files(container)
    state = await run_workflow(
        container.graph, task=TASK, supplier_id="unknown-supplier", owner="tester", requirements=[ISO_REQ]
    )
    assert state["report"].findings[0].status is FindingStatus.INSUFFICIENT


async def test_other_owners_documents_are_never_retrieved(container):
    await ingest_files(container, owner="alice")
    state = await run_workflow(
        container.graph, task=TASK, supplier_id="halvorsen", owner="mallory", requirements=[ISO_REQ]
    )
    assert state["evidence"]["REQ-001"] == []
    assert state["report"].findings[0].status is FindingStatus.INSUFFICIENT


async def test_provider_error_propagates_without_mock_fallback(tmp_path):
    class FailingChat:
        name = "failing"

        async def generate_structured(self, **_):
            raise ProviderRateLimitError("rate limited", retryable=True)

    container = build_container(make_settings(tmp_path), chat=FailingChat())
    await ingest_files(container)
    with pytest.raises(NodeExecutionError) as info:
        await run_workflow(
            container.graph, task=TASK, supplier_id="halvorsen", owner="tester", requirements=[ISO_REQ]
        )
    assert info.value.to_dict() == {
        "code": "provider_rate_limited",
        "message": "rate limited",
        "retryable": True,
        "failed_node": "planner",
    }


async def test_prompt_injection_document_is_flagged_and_ignored(container):
    await ingest_files(
        container,
        files=[
            ("northgate_procurement_policy.md", "policy", None),
            ("halvorsen_submission_notes.txt", "supplier", "halvorsen"),
        ],
    )
    state = await run_workflow(container.graph, task=TASK, supplier_id="halvorsen", owner="tester")
    report = state["report"]
    assert any("instructions to an AI system" in w for w in report.warnings)
    assert report.summary["met"] == 0
