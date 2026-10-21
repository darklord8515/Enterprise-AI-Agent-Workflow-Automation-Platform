"""Assemble the explicit LangGraph StateGraph.

```
START -> planner -> retriever -> analyst -> evidence_reviewer -> router
router --report-------> report_generator -> END
router --revise-------> retriever            (bounded by MAX_REVISIONS)
router --insufficient-> insufficient_evidence -> report_generator
```
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from .nodes import WorkflowDeps, build_nodes, traced
from .state import WorkflowState

# Hard ceiling on graph steps, independent of MAX_REVISIONS, as a last-resort loop guard.
RECURSION_LIMIT = 40


def _route(state: WorkflowState) -> str:
    return state["route"]


def build_graph(deps: WorkflowDeps) -> Any:
    nodes = build_nodes(deps)
    graph = StateGraph(WorkflowState)
    for name, fn in nodes.items():
        graph.add_node(name, traced(name, fn))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "analyst")
    graph.add_edge("analyst", "evidence_reviewer")
    graph.add_edge("evidence_reviewer", "router")
    graph.add_conditional_edges(
        "router",
        _route,
        {"report": "report_generator", "revise": "retriever", "insufficient": "insufficient_evidence"},
    )
    graph.add_edge("insufficient_evidence", "report_generator")
    graph.add_edge("report_generator", END)
    return graph.compile()


async def run_workflow(
    graph: Any,
    *,
    task: str,
    supplier_id: str,
    owner: str,
    requirements: list[str] | None = None,
    document_ids: list[str] | None = None,
) -> WorkflowState:
    initial: WorkflowState = {
        "task": task,
        "supplier_id": supplier_id,
        "owner": owner,
        "explicit_requirements": requirements,
        "document_ids": document_ids,
        "warnings": [],
        "trace": [],
    }
    final: WorkflowState = await graph.ainvoke(initial, config={"recursion_limit": RECURSION_LIMIT})
    # The report was built before the report_generator's own trace event was appended.
    final["report"] = final["report"].model_copy(update={"trace": final["trace"]})
    return final
