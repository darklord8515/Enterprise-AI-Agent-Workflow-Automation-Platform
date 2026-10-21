"""Builds the provider set for the configured PROVIDER_MODE. There is no automatic fallback: in azure
mode every component is a real Azure client, and failures propagate as errors."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .config import ProviderMode, Settings
from .ingestion.registry import DocumentRegistry
from .ingestion.service import IngestionService
from .providers.base import ChatProvider, EmbeddingProvider
from .retrieval.base import SearchBackend
from .services.tasks import TaskService
from .workflow.graph import build_graph
from .workflow.nodes import WorkflowDeps

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    chat: ChatProvider
    embeddings: EmbeddingProvider
    search: SearchBackend
    registry: DocumentRegistry
    ingestion: IngestionService
    tasks: TaskService
    graph: Any

    async def close(self) -> None:
        close = getattr(self.search, "close", None)
        if close:
            await close()


def build_providers(settings: Settings) -> tuple[ChatProvider, EmbeddingProvider, SearchBackend]:
    if settings.provider_mode is ProviderMode.AZURE:
        from .providers.azure_openai import (
            AzureOpenAIChatProvider,
            AzureOpenAIEmbeddingProvider,
            build_openai_client,
        )
        from .retrieval.azure_search import AzureSearchBackend

        client = build_openai_client(settings)
        return (
            AzureOpenAIChatProvider(settings, client),
            AzureOpenAIEmbeddingProvider(settings, client),
            AzureSearchBackend(settings),
        )

    from .providers.mock import MockChatProvider, MockEmbeddingProvider
    from .retrieval.memory import InMemorySearchBackend

    return (
        MockChatProvider(),
        MockEmbeddingProvider(settings.mock_embedding_dimensions),
        InMemorySearchBackend(),
    )


def build_container(
    settings: Settings,
    *,
    chat: ChatProvider | None = None,
    embeddings: EmbeddingProvider | None = None,
    search: SearchBackend | None = None,
) -> Container:
    if chat is None or embeddings is None or search is None:
        default_chat, default_embeddings, default_search = build_providers(settings)
        chat = chat or default_chat
        embeddings = embeddings or default_embeddings
        search = search or default_search
    registry = DocumentRegistry(settings.data_dir)
    graph = build_graph(WorkflowDeps(settings=settings, chat=chat, embeddings=embeddings, search=search))
    logger.info(
        "container.ready",
        extra={"chat": chat.name, "embeddings": embeddings.name, "search": search.name},
    )
    return Container(
        settings=settings,
        chat=chat,
        embeddings=embeddings,
        search=search,
        registry=registry,
        ingestion=IngestionService(settings, registry, embeddings, search),
        tasks=TaskService(graph),
        graph=graph,
    )
