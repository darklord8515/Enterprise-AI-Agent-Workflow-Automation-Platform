"""Security helpers: API-key auth, filename sanitisation, OData escaping and prompt-injection
heuristics for retrieved content."""

from __future__ import annotations

import hmac
import re
from pathlib import PurePath

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from .config import Settings, get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

LOCAL_PRINCIPAL = "local-dev"


def require_principal(
    api_key: str | None = Depends(_api_key_header),
    settings: Settings = Depends(get_settings),
) -> str:
    """Resolve the caller's principal from the X-API-Key header.

    If no API keys are configured, auth is disabled and every caller is ``local-dev``. That is only
    acceptable for local development; the startup log warns about it.
    """
    keys = settings.api_key_map
    if not keys:
        return LOCAL_PRINCIPAL
    if api_key:
        for candidate, principal in keys.items():
            if hmac.compare_digest(candidate.encode(), api_key.encode()):
                return principal
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._\- ]+")


def sanitize_filename(name: str) -> str:
    """Drop any path components and unusual characters from a client-supplied filename."""
    base = PurePath(name.replace("\\", "/")).name
    cleaned = _SAFE_NAME.sub("_", base).strip(" .")
    return cleaned[:120] or "upload"


def odata_quote(value: str) -> str:
    """Escape a value for use inside an OData string literal (single quotes doubled)."""
    return "'" + value.replace("'", "''") + "'"


# Phrases typical of instructions aimed at the model rather than a human reader. Matching chunks are
# flagged (and surfaced as report warnings); they are still treated purely as quoted data.
_INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above|earlier) (instructions|prompts|directions)",
    r"disregard (all |any )?(previous|prior|above|the) (instructions|rules|system)",
    r"you are now\b",
    r"\bsystem prompt\b",
    r"(mark|classify|report|rate) (all|every|each) (requirements?|checks?|items?) as (met|compliant|passed)",
    r"\bact as\b.{0,40}\b(assistant|model|ai)\b",
    r"</?(system|assistant|instructions?)>",
    r"\bdo not (report|mention|flag)\b",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def looks_like_prompt_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text))


def neutralize_delimiters(text: str) -> str:
    """Prevent retrieved text from closing the <evidence> wrapper used in prompts."""
    return re.sub(r"</?\s*evidence[^>]*>", "[removed-tag]", text, flags=re.IGNORECASE)
