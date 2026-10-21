from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from procurement_agent.api.app import create_app
from procurement_agent.config import Settings
from procurement_agent.container import build_container
from procurement_agent.models import Assessment, DocumentCategory
from procurement_agent.providers.base import LLMUsage, StructuredResult
from procurement_agent.providers.mock import MockChatProvider

DATASET = Path(__file__).resolve().parents[1] / "eval" / "dataset"


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    base = {"provider_mode": "mock", "data_dir": tmp_path, "api_keys": "", "_env_file": None}
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def container(settings: Settings):
    return build_container(settings)


async def ingest_files(
    container, owner: str = "tester", files: list[tuple[str, str, str | None]] | None = None
):
    """Upload + index dataset files. ``files`` items are (filename, category, supplier_id)."""
    files = files or [
        ("northgate_procurement_policy.md", "policy", None),
        ("halvorsen_certifications.md", "supplier", "halvorsen"),
        ("halvorsen_insurance.md", "supplier", "halvorsen"),
        ("halvorsen_company_profile.md", "supplier", "halvorsen"),
    ]
    ids = []
    for name, category, supplier in files:
        record = container.ingestion.upload(
            filename=name,
            data=(DATASET / "documents" / name).read_bytes(),
            category=DocumentCategory(category),
            supplier_id=supplier,
            owner=owner,
        )
        ids.append(record.document_id)
    await container.ingestion.index(ids, owner)
    return ids


class ScriptedChatProvider:
    """Uses the mock planner, but lets a test control analyst output per call."""

    name = "scripted"

    def __init__(self, analyze_fn) -> None:
        self._mock = MockChatProvider()
        self._analyze_fn = analyze_fn
        self.analyze_calls: list[dict] = []

    async def generate_structured(self, *, task_name, system_prompt, user_prompt, response_model, context):
        if task_name == "plan":
            return await self._mock.generate_structured(
                task_name=task_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=response_model,
                context=context,
            )
        self.analyze_calls.append(context)
        value = self._analyze_fn(context, len(self.analyze_calls))
        return StructuredResult(value=Assessment.model_validate(value), usage=LLMUsage(10, 5))


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_client(tmp_path: Path):
    app = create_app(make_settings(tmp_path, api_keys="alice-key:alice,bob-key:bob"))
    with TestClient(app) as c:
        yield c
