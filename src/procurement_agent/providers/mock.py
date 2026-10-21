"""Deterministic mock providers for local development and tests.

These are NOT language models. The mock chat provider is a transparent keyword/negation heuristic
that produces outputs in the same schema as the real model so the full workflow (including the
evidence reviewer) can be exercised without Azure credentials. Results obtained in mock mode measure
the pipeline and the heuristic, not Azure OpenAI quality.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from ..models import (
    Assessment,
    CitedEvidence,
    FindingStatus,
    Plan,
    PlannedCheck,
    RequirementAssessment,
)
from .base import LLMUsage, StructuredResult, T

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
    "within",
    "must",
    "shall",
    "should",
    "supplier",
    "suppliers",
    "provide",
    "provides",
    "hold",
    "holds",
    "maintain",
    "maintains",
    "current",
    "valid",
    "least",
    "all",
    "any",
    "their",
    "they",
    "we",
    "our",
    "will",
    "can",
    "per",
    "each",
    "evidence",
    "company",
    "required",
    "requirement",
    "demonstrate",
    "documented",
    "date",
    "years",
    "year",
    "months",
}
_NEGATIONS = re.compile(
    r"\b(not|no|never|none|without|expired|lapsed|pending|planned|plan to|intend|intends|"
    r"working towards|in progress|under way|underway|does not|do not|has not|have not|cannot)\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_REQ_LINE = re.compile(r"^\s*[-*]?\s*\**(REQ-\d{1,4})\**\s*[:.\-–]\s*(.+?)\s*$", re.MULTILINE)
_MONEY = re.compile(r"[£$€]\s?(\d+(?:[.,]\d+)*)\s*(million|m|k|thousand)?\b", re.IGNORECASE)


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in _WORD.findall(text.lower())]


def key_terms(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1}


def _parse_amounts(text: str) -> list[float]:
    amounts = []
    for number, unit in _MONEY.findall(text):
        value = float(number.replace(",", ""))
        unit = (unit or "").lower()
        if unit in {"million", "m"}:
            value *= 1_000_000
        elif unit in {"k", "thousand"}:
            value *= 1_000
        amounts.append(value)
    return amounts


class MockEmbeddingProvider:
    """Feature-hashed bag-of-words vectors: deterministic and lexically meaningful."""

    name = "mock"

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        for token in tokenize(text):
            if token in _STOPWORDS:
                continue
            digest = hashlib.sha256(token.encode()).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimensions
            vec[idx] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed(self, texts: list[str]) -> tuple[list[list[float]], int]:
        return [self._vector(t) for t in texts], 0


class MockChatProvider:
    """Rule-based stand-in for the planner and analyst LLM calls."""

    name = "mock"

    async def generate_structured(
        self,
        *,
        task_name: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        context: dict[str, Any],
    ) -> StructuredResult[T]:
        if task_name == "plan":
            value: Any = self._plan(context)
        elif task_name == "analyze":
            value = self._analyze(context)
        else:
            raise ValueError(f"MockChatProvider does not support task '{task_name}'")
        return StructuredResult(value=response_model.model_validate(value.model_dump()), usage=LLMUsage())

    # -- planner -------------------------------------------------------------------------------

    def _plan(self, context: dict[str, Any]) -> Plan:
        explicit = context.get("explicit_requirements") or []
        if explicit:
            checks = [
                PlannedCheck(requirement_id=f"REQ-{i:03d}", requirement=r, search_query=r)
                for i, r in enumerate(explicit, start=1)
            ]
            return Plan(checks=checks)
        seen: dict[str, PlannedCheck] = {}
        for chunk in context.get("policy_chunks", []):
            for req_id, text in _REQ_LINE.findall(chunk["content"]):
                if req_id not in seen:
                    seen[req_id] = PlannedCheck(requirement_id=req_id, requirement=text, search_query=text)
        return Plan(checks=sorted(seen.values(), key=lambda c: c.requirement_id))

    # -- analyst -------------------------------------------------------------------------------

    def _analyze(self, context: dict[str, Any]) -> Assessment:
        return Assessment(assessments=[self._assess(item) for item in context["requirements"]])

    def _assess(self, item: dict[str, Any]) -> RequirementAssessment:
        terms = key_terms(item["requirement"])
        best: tuple[float, str, str] | None = None  # (score, chunk_id, sentence)
        for chunk in item["evidence"]:
            if chunk["category"] != "supplier":
                continue  # policy text can never prove supplier compliance
            for raw in _SENTENCE_SPLIT.split(chunk["content"]):
                sentence = raw.strip().lstrip("-*# ").strip()
                if len(sentence) < 15:
                    continue
                overlap = len(terms & set(tokenize(sentence)))
                score = overlap / max(len(terms), 1)
                if overlap >= 2 and score >= 0.34 and (best is None or score > best[0]):
                    best = (score, chunk["chunk_id"], sentence)

        rid = item["requirement_id"]
        if best is None:
            return RequirementAssessment(
                requirement_id=rid,
                status=FindingStatus.INSUFFICIENT,
                evidence=[],
                explanation="No retrieved supplier document addresses this requirement.",
                missing_information=[f"Supplier evidence for: {item['requirement']}"],
                recommended_follow_up=["Request documentation from the supplier covering this requirement."],
            )

        _, chunk_id, sentence = best
        status = FindingStatus.MET
        reason = "The supplier document states this requirement is satisfied."
        required = _parse_amounts(item["requirement"])
        offered = _parse_amounts(sentence)
        if required and offered and max(offered) < max(required):
            status = FindingStatus.GAP
            reason = f"The stated amount ({max(offered):,.0f}) is below the required {max(required):,.0f}."
        elif _NEGATIONS.search(sentence):
            status = FindingStatus.GAP
            reason = "The supplier document indicates the requirement is not (yet) satisfied."

        return RequirementAssessment(
            requirement_id=rid,
            status=status,
            evidence=[CitedEvidence(chunk_id=chunk_id, quote=sentence)],
            explanation=f"[mock heuristic] {reason}",
            missing_information=[]
            if status is FindingStatus.MET
            else ["Confirmation of a remediation plan or updated documentation."],
            recommended_follow_up=[]
            if status is FindingStatus.MET
            else ["Raise the gap with the supplier and request a remediation timeline."],
        )
