"""Typed errors. Provider failures are surfaced, never masked by mock fallbacks."""

from __future__ import annotations


class AppError(Exception):
    code = "internal_error"
    http_status = 500

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "retryable": self.retryable}


class ProviderError(AppError):
    code = "provider_error"
    http_status = 502


class ProviderAuthError(ProviderError):
    code = "provider_auth_failed"


class ProviderRateLimitError(ProviderError):
    code = "provider_rate_limited"
    http_status = 429


class DeploymentNotFoundError(ProviderError):
    code = "deployment_not_found"


class ProviderUnavailableError(ProviderError):
    code = "provider_unavailable"
    http_status = 503


class ProviderResponseError(ProviderError):
    code = "provider_invalid_response"


class SearchError(AppError):
    code = "search_error"
    http_status = 502


class IndexSchemaMismatchError(SearchError):
    code = "index_schema_mismatch"


class ValidationFailure(AppError):
    code = "validation_error"
    http_status = 422


class NotFound(AppError):
    code = "not_found"
    http_status = 404


class WorkflowError(AppError):
    code = "workflow_error"
