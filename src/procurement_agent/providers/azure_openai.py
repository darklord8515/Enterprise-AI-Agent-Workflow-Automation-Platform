"""Azure OpenAI adapter for structured chat completions and embeddings.

Auth: API key if ``AZURE_OPENAI_API_KEY`` is set, otherwise Entra ID via ``DefaultAzureCredential``
(the keyless pattern used by Azure-Samples/langchain-agent-python and azure-search-openai-demo).

Endpoint: if ``AZURE_OPENAI_API_VERSION`` is set, the classic ``AsyncAzureOpenAI`` client is used with
that version; otherwise the GA v1 endpoint ``<endpoint>/openai/v1/`` is used, which needs no version.

Retries/timeouts are delegated to the OpenAI SDK (exponential backoff on 408/429/5xx/connection
errors), bounded by ``AZURE_OPENAI_MAX_RETRIES`` and ``AZURE_OPENAI_TIMEOUT_SECONDS``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import openai
from openai import AsyncAzureOpenAI, AsyncOpenAI

from ..config import Settings
from ..errors import (
    DeploymentNotFoundError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderUnavailableError,
)
from .base import LLMUsage, StructuredResult, T

logger = logging.getLogger(__name__)

_COGNITIVE_SCOPE = "https://cognitiveservices.azure.com/.default"
_EMBED_BATCH = 16


def build_openai_client(settings: Settings) -> AsyncOpenAI:
    endpoint = settings.azure_openai_endpoint.rstrip("/")
    api_key = settings.azure_openai_api_key.get_secret_value()
    token_provider = None
    if not api_key:
        from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider

        token_provider = get_bearer_token_provider(DefaultAzureCredential(), _COGNITIVE_SCOPE)

    common: dict[str, Any] = {
        "timeout": settings.azure_openai_timeout_seconds,
        "max_retries": settings.azure_openai_max_retries,
    }
    if settings.azure_openai_api_version:
        return AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_version=settings.azure_openai_api_version,
            api_key=api_key or None,
            azure_ad_token_provider=token_provider,
            **common,
        )
    if endpoint.endswith("/openai/v1"):
        endpoint = endpoint[: -len("/openai/v1")]
    return AsyncOpenAI(base_url=f"{endpoint}/openai/v1/", api_key=api_key or token_provider, **common)


def map_openai_error(exc: Exception, deployment: str) -> ProviderError:
    """Translate SDK exceptions into actionable, typed application errors."""
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return ProviderAuthError(
            "Azure OpenAI rejected the credentials. Check AZURE_OPENAI_API_KEY or the RBAC role "
            "'Cognitive Services OpenAI User' for the Entra identity."
        )
    if isinstance(exc, openai.RateLimitError):
        return ProviderRateLimitError(
            f"Azure OpenAI rate limit hit on deployment '{deployment}' after bounded retries.",
            retryable=True,
        )
    if isinstance(exc, openai.NotFoundError):
        return DeploymentNotFoundError(
            f"Azure OpenAI deployment '{deployment}' was not found. Check the deployment name and endpoint."
        )
    if isinstance(exc, openai.APITimeoutError):
        return ProviderUnavailableError("Azure OpenAI request timed out.", retryable=True)
    if isinstance(exc, openai.APIConnectionError):
        return ProviderUnavailableError("Could not connect to Azure OpenAI endpoint.", retryable=True)
    if isinstance(exc, openai.InternalServerError):
        return ProviderUnavailableError("Azure OpenAI returned a server error.", retryable=True)
    if isinstance(exc, openai.LengthFinishReasonError):
        return ProviderResponseError("Model output was truncated (max tokens reached).")
    if isinstance(exc, openai.ContentFilterFinishReasonError):
        return ProviderResponseError("Model output was blocked by the content filter.")
    if isinstance(exc, openai.BadRequestError):
        return ProviderResponseError(f"Azure OpenAI rejected the request: {exc.message}")
    return ProviderError(f"Unexpected Azure OpenAI error: {type(exc).__name__}")


class AzureOpenAIChatProvider:
    name = "azure_openai"

    def __init__(self, settings: Settings, client: AsyncOpenAI | None = None) -> None:
        self.settings = settings
        self.deployment = settings.azure_openai_chat_deployment
        self.client = client or build_openai_client(settings)

    async def generate_structured(
        self,
        *,
        task_name: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        context: dict[str, Any],
    ) -> StructuredResult[T]:
        kwargs: dict[str, Any] = {}
        if self.settings.azure_openai_temperature is not None:
            kwargs["temperature"] = self.settings.azure_openai_temperature
        started = time.perf_counter()
        try:
            completion = await self.client.chat.completions.parse(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=response_model,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - mapped to typed errors
            mapped = map_openai_error(exc, self.deployment)
            logger.error("azure_openai.chat_failed", extra={"task": task_name, "error": mapped.code})
            raise mapped from exc

        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            raise ProviderResponseError(f"Model refused the '{task_name}' request: {message.refusal}")
        if message.parsed is None:
            raise ProviderResponseError(f"Model returned no structured output for '{task_name}'.")
        usage = LLMUsage(
            prompt_tokens=getattr(completion.usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(completion.usage, "completion_tokens", 0) or 0,
        )
        logger.info(
            "azure_openai.chat_ok",
            extra={
                "task": task_name,
                "deployment": self.deployment,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            },
        )
        return StructuredResult(value=message.parsed, usage=usage)


class AzureOpenAIEmbeddingProvider:
    name = "azure_openai"

    def __init__(self, settings: Settings, client: AsyncOpenAI | None = None) -> None:
        self.deployment = settings.azure_openai_embedding_deployment
        self.dimensions = settings.azure_openai_embedding_dimensions
        self.client = client or build_openai_client(settings)

    async def embed(self, texts: list[str]) -> tuple[list[list[float]], int]:
        vectors: list[list[float]] = []
        tokens = 0
        for start in range(0, len(texts), _EMBED_BATCH):
            batch = texts[start : start + _EMBED_BATCH]
            try:
                response = await self.client.embeddings.create(
                    model=self.deployment, input=batch, dimensions=self.dimensions
                )
            except Exception as exc:  # noqa: BLE001
                raise map_openai_error(exc, self.deployment) from exc
            for item in sorted(response.data, key=lambda d: d.index):
                if len(item.embedding) != self.dimensions:
                    raise ProviderResponseError(
                        f"Embedding has {len(item.embedding)} dimensions, expected {self.dimensions}. "
                        "Check AZURE_OPENAI_EMBEDDING_DIMENSIONS matches the model."
                    )
                vectors.append(item.embedding)
            tokens += getattr(response.usage, "total_tokens", 0) or 0
        return vectors, tokens
