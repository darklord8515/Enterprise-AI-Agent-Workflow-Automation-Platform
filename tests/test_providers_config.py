"""Azure provider adapter tests using fake SDK clients (no network)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest
from pydantic import ValidationError

from procurement_agent.config import Settings
from procurement_agent.errors import (
    DeploymentNotFoundError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderUnavailableError,
)
from procurement_agent.models import Plan
from procurement_agent.providers.azure_openai import (
    AzureOpenAIChatProvider,
    AzureOpenAIEmbeddingProvider,
    build_openai_client,
    map_openai_error,
)

ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"
_REQ = httpx.Request("POST", "https://example.openai.azure.com/openai/v1/chat/completions")


def _status_error(cls, code):
    return cls("boom", response=httpx.Response(code, request=_REQ), body=None)


def azure_settings(**kw) -> Settings:
    base = dict(
        provider_mode="azure",
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_openai_api_key="test-key",
        azure_openai_chat_deployment="chat",
        azure_openai_embedding_deployment="embed",
        azure_openai_embedding_dimensions=4,
        azure_search_endpoint="https://example.search.windows.net",
        _env_file=None,
    )
    base.update(kw)
    return Settings(**base)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (_status_error(openai.AuthenticationError, 401), ProviderAuthError),
        (_status_error(openai.PermissionDeniedError, 403), ProviderAuthError),
        (_status_error(openai.RateLimitError, 429), ProviderRateLimitError),
        (_status_error(openai.NotFoundError, 404), DeploymentNotFoundError),
        (_status_error(openai.InternalServerError, 500), ProviderUnavailableError),
        (openai.APITimeoutError(request=_REQ), ProviderUnavailableError),
        (openai.APIConnectionError(request=_REQ), ProviderUnavailableError),
        (_status_error(openai.BadRequestError, 400), ProviderResponseError),
    ],
)
def test_openai_errors_map_to_typed_errors(exc, expected):
    mapped = map_openai_error(exc, "chat")
    assert isinstance(mapped, expected)
    assert "test-key" not in mapped.message


def test_rate_limit_is_marked_retryable():
    assert map_openai_error(_status_error(openai.RateLimitError, 429), "chat").retryable


class _FakeCompletions:
    def __init__(self, result=None, exc=None):
        self.result, self.exc, self.kwargs = result, exc, None

    async def parse(self, **kwargs):
        self.kwargs = kwargs
        if self.exc:
            raise self.exc
        return self.result


def _fake_client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def _completion(parsed, refusal=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, refusal=refusal))],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
    )


async def _call(provider):
    return await provider.generate_structured(
        task_name="plan", system_prompt="s", user_prompt="u", response_model=Plan, context={}
    )


async def test_chat_provider_returns_structured_output_and_usage():
    plan = Plan(checks=[])
    completions = _FakeCompletions(result=_completion(plan))
    provider = AzureOpenAIChatProvider(azure_settings(), client=_fake_client(completions))
    result = await _call(provider)
    assert result.value is plan
    assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (11, 7)
    assert completions.kwargs["model"] == "chat"
    assert completions.kwargs["response_format"] is Plan
    assert "temperature" not in completions.kwargs  # omitted unless configured


async def test_chat_provider_handles_refusal_and_errors():
    refused = AzureOpenAIChatProvider(
        azure_settings(), client=_fake_client(_FakeCompletions(_completion(None, "no")))
    )
    with pytest.raises(ProviderResponseError):
        await _call(refused)
    failing = AzureOpenAIChatProvider(
        azure_settings(), client=_fake_client(_FakeCompletions(exc=_status_error(openai.NotFoundError, 404)))
    )
    with pytest.raises(DeploymentNotFoundError):
        await _call(failing)


async def test_embedding_provider_checks_dimensions():
    class FakeEmbeddings:
        def __init__(self, dims):
            self.dims = dims

        async def create(self, model, input, dimensions):
            data = [SimpleNamespace(index=i, embedding=[0.1] * self.dims) for i in range(len(input))]
            return SimpleNamespace(data=data, usage=SimpleNamespace(total_tokens=3))

    ok = AzureOpenAIEmbeddingProvider(azure_settings(), client=SimpleNamespace(embeddings=FakeEmbeddings(4)))
    vectors, tokens = await ok.embed(["a", "b"])
    assert len(vectors) == 2 and tokens == 3
    bad = AzureOpenAIEmbeddingProvider(azure_settings(), client=SimpleNamespace(embeddings=FakeEmbeddings(3)))
    with pytest.raises(ProviderResponseError):
        await bad.embed(["a"])


def test_client_uses_v1_endpoint_without_api_version():
    client = build_openai_client(azure_settings())
    assert str(client.base_url) == "https://example.openai.azure.com/openai/v1/"
    assert client.max_retries == 3


def test_client_uses_versioned_azure_client_when_configured():
    client = build_openai_client(azure_settings(azure_openai_api_version="2024-10-21"))
    assert isinstance(client, openai.AsyncAzureOpenAI)


def test_azure_mode_requires_settings():
    with pytest.raises(ValidationError) as info:
        Settings(provider_mode="azure", _env_file=None)
    assert "AZURE_OPENAI_ENDPOINT" in str(info.value)


def test_env_example_loads_and_placeholders_are_rejected_in_azure_mode():
    assert Settings(_env_file=ENV_EXAMPLE).provider_mode.value == "mock"
    with pytest.raises(ValidationError) as info:
        Settings(_env_file=ENV_EXAMPLE, provider_mode="azure")
    assert "placeholders" in str(info.value)


def test_api_key_parsing():
    s = Settings(api_keys="k1:alice, k2:bob ,", _env_file=None)
    assert s.api_key_map == {"k1": "alice", "k2": "bob"}
