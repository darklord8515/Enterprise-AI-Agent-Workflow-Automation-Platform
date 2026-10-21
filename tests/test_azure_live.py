"""Opt-in tests against real Azure resources.

Skipped unless RUN_AZURE_TESTS=1 and PROVIDER_MODE=azure settings are present in the environment.
They create documents under a unique owner in the configured index.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from procurement_agent.config import Settings
from procurement_agent.container import build_container
from procurement_agent.evaluation import run_evaluation

from .conftest import DATASET

pytestmark = [
    pytest.mark.azure,
    pytest.mark.skipif(os.getenv("RUN_AZURE_TESTS") != "1", reason="set RUN_AZURE_TESTS=1 to run"),
]


async def test_live_azure_evaluation_completes():
    settings = Settings(provider_mode="azure", data_dir=Path(tempfile.mkdtemp()))
    container = build_container(settings)
    try:
        results = await run_evaluation(container, DATASET)
    finally:
        await container.close()
    summary = results["summary"]
    print(summary)
    assert summary["workflow_completion"] == 1.0
    assert summary["unsupported_claims"] == 0
    assert summary["forbidden_evidence_leaks"] == 0
