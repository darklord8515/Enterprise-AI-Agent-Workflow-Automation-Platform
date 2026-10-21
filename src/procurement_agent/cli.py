"""Command-line entry points.

procurement-agent demo   # ingest the sample dataset and print a report for one case
procurement-agent eval   # run the evaluation dataset and write eval/results/<mode>-<timestamp>.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from .config import get_settings
from .container import build_container
from .evaluation import ingest_dataset, run_evaluation
from .logging_config import configure_logging
from .workflow.graph import run_workflow

DEFAULT_DATASET = Path("eval/dataset")


async def _demo(dataset: Path, case_id: str | None) -> int:
    settings = get_settings()
    if settings.provider_mode.value == "mock":
        settings = settings.model_copy(
            update={"data_dir": Path(tempfile.mkdtemp(prefix="procurement-demo-"))}
        )
    container = build_container(settings)
    try:
        await ingest_dataset(container, dataset, owner="demo")
        cases = json.loads((dataset / "cases.json").read_text(encoding="utf-8"))["cases"]
        case = next((c for c in cases if c["case_id"] == case_id), cases[0])
        state = await run_workflow(
            container.graph,
            task=case["task"],
            supplier_id=case["supplier_id"],
            owner="demo",
            requirements=case.get("requirements"),
        )
        print(state["report"].model_dump_json(indent=2))
    finally:
        await container.close()
    return 0


async def _eval(dataset: Path, output_dir: Path) -> int:
    settings = get_settings()
    settings = settings.model_copy(update={"data_dir": Path(tempfile.mkdtemp(prefix="procurement-eval-"))})
    container = build_container(settings)
    try:
        results = await run_evaluation(container, dataset)
    finally:
        await container.close()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"{settings.provider_mode.value}-{stamp}.json"
    path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(json.dumps(results["summary"], indent=2))
    print(f"\nFull results written to {path}", file=sys.stderr)
    return 0 if results["summary"]["workflow_completion"] == 1.0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="procurement-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run one dataset case end to end and print the report")
    demo.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    demo.add_argument("--case", default=None)
    ev = sub.add_parser("eval", help="Run the evaluation dataset")
    ev.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ev.add_argument("--output", type=Path, default=Path("eval/results"))
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging("WARNING", settings.provider_mode.value)
    print(f"[procurement-agent] provider_mode={settings.provider_mode.value}", file=sys.stderr)
    if args.command == "demo":
        return asyncio.run(_demo(args.dataset, args.case))
    return asyncio.run(_eval(args.dataset, args.output))


if __name__ == "__main__":
    raise SystemExit(main())
