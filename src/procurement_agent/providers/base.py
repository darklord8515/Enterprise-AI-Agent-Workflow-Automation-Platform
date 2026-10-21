"""Provider interfaces used by ingestion and the workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class StructuredResult(Generic[T]):  # noqa: UP046 - keep 3.11 compatible
    value: T
    usage: LLMUsage


class ChatProvider(Protocol):
    name: str

    async def generate_structured(
        self,
        *,
        task_name: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        context: dict[str, Any],
    ) -> StructuredResult[T]:
        """Return a validated ``response_model`` instance.

        ``system_prompt``/``user_prompt`` are what a real model sees. ``context`` is the structured
        input the prompts were rendered from; real providers ignore it, the deterministic mock uses
        it so it does not have to parse prompt text.
        """
        ...


class EmbeddingProvider(Protocol):
    name: str
    dimensions: int

    async def embed(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """Return (vectors, tokens_used)."""
        ...
