"""Application configuration, loaded exclusively from environment variables / .env.

No endpoints, deployment names, API versions or credentials are hardcoded. In ``azure`` mode the
required Azure settings are validated at startup so a misconfiguration fails loudly instead of
silently degrading to mock behaviour.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderMode(StrEnum):
    MOCK = "mock"
    AZURE = "azure"


class RetrievalMode(StrEnum):
    HYBRID = "hybrid"
    VECTOR = "vector"
    KEYWORD = "keyword"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_ignore_empty=True
    )

    # --- Runtime mode -------------------------------------------------------------------------
    provider_mode: ProviderMode = ProviderMode.MOCK
    log_level: str = "INFO"
    data_dir: Path = Path("./data")
    # Web UI at "/" and the read-only sample dataset at "/samples" (fictional documents only).
    serve_ui: bool = True
    samples_dir: Path = Path("./eval/dataset")

    # --- API security -------------------------------------------------------------------------
    # Comma-separated "key:principal" pairs, e.g. "k1:alice,k2:bob". Documents and tasks are
    # scoped to the principal that created them.
    api_keys: SecretStr = SecretStr("")
    max_upload_bytes: int = 5 * 1024 * 1024
    allowed_extensions: str = ".txt,.md,.pdf"
    max_task_chars: int = 2000
    max_requirements: int = 20

    # --- Workflow -----------------------------------------------------------------------------
    max_revisions: int = Field(default=1, ge=0, le=3)
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 120

    # --- Azure OpenAI -------------------------------------------------------------------------
    azure_openai_endpoint: str = ""
    # Optional: if empty, Entra ID (DefaultAzureCredential) is used.
    azure_openai_api_key: SecretStr = SecretStr("")
    # Optional: if empty the GA v1 endpoint (<endpoint>/openai/v1/) is used, which needs no version.
    azure_openai_api_version: str = ""
    azure_openai_chat_deployment: str = ""
    azure_openai_embedding_deployment: str = ""
    azure_openai_embedding_dimensions: int = 1536
    # Some reasoning models reject temperature; leave empty to omit it.
    azure_openai_temperature: float | None = None
    azure_openai_timeout_seconds: float = 60.0
    azure_openai_max_retries: int = Field(default=3, ge=0, le=6)

    # --- Azure AI Search ----------------------------------------------------------------------
    azure_search_endpoint: str = ""
    azure_search_api_key: SecretStr = SecretStr("")
    azure_search_index: str = "procurement-evidence"
    azure_search_timeout_seconds: float = 30.0

    # --- Mock mode ----------------------------------------------------------------------------
    mock_embedding_dimensions: int = 256

    @model_validator(mode="after")
    def _validate_azure(self) -> Settings:
        if self.provider_mode is ProviderMode.AZURE:
            missing = [
                name
                for name in (
                    "azure_openai_endpoint",
                    "azure_openai_chat_deployment",
                    "azure_openai_embedding_deployment",
                    "azure_search_endpoint",
                    "azure_search_index",
                )
                if not getattr(self, name)
            ]
            if missing:
                raise ValueError(
                    "PROVIDER_MODE=azure but required settings are missing: "
                    + ", ".join(m.upper() for m in missing)
                )
            placeholders = [
                name
                for name in (
                    "azure_openai_endpoint",
                    "azure_openai_chat_deployment",
                    "azure_openai_embedding_deployment",
                    "azure_search_endpoint",
                )
                if "<" in getattr(self, name)
            ]
            if placeholders:
                raise ValueError(
                    "Replace the .env.example placeholders for: " + ", ".join(p.upper() for p in placeholders)
                )
        return self

    @property
    def allowed_extension_set(self) -> set[str]:
        return {e.strip().lower() for e in self.allowed_extensions.split(",") if e.strip()}

    @property
    def api_key_map(self) -> dict[str, str]:
        """Map of API key -> principal. Empty means auth is disabled (local dev only)."""
        mapping: dict[str, str] = {}
        for pair in self.api_keys.get_secret_value().split(","):
            pair = pair.strip()
            if not pair:
                continue
            key, _, principal = pair.partition(":")
            mapping[key.strip()] = principal.strip() or "default"
        return mapping

    @property
    def embedding_dimensions(self) -> int:
        if self.provider_mode is ProviderMode.AZURE:
            return self.azure_openai_embedding_dimensions
        return self.mock_embedding_dimensions


@lru_cache
def get_settings() -> Settings:
    return Settings()
