"""AzureSearchBackend tests through a fake azure-core HTTP transport.

These exercise the real SDK serialisation (request URLs and JSON bodies) without network access. They
verify what we send to Azure AI Search, not how the live service responds.
"""

from __future__ import annotations

import json

import pytest
from azure.core.pipeline.transport import AsyncHttpTransport
from azure.core.rest._http_response_impl_async import AsyncHttpResponseImpl
from azure.core.utils import CaseInsensitiveDict

from procurement_agent.config import RetrievalMode, Settings
from procurement_agent.errors import IndexSchemaMismatchError, SearchError
from procurement_agent.models import Chunk, DocumentCategory
from procurement_agent.retrieval.azure_search import AzureSearchBackend
from procurement_agent.retrieval.base import SearchFilter

ENDPOINT = "https://svc.search.windows.net"
HIT = {
    "@search.score": 0.9,
    "id": "d-0000",
    "content": "Cyber cover is £2 million.",
    "document_id": "d",
    "source_file": "a.md",
    "category": "supplier",
    "supplier_id": "s1",
    "owner": "o",
    "page": None,
    "section": "Insurance",
    "chunk_index": 0,
    "suspicious": False,
}


class FakeTransport(AsyncHttpTransport):
    def __init__(self, handler):
        self.handler = handler
        self.requests: list[tuple[str, str, dict | None]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def open(self):
        return None

    async def close(self):
        return None

    async def send(self, request, **kwargs):
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url, body))
        status, payload = self.handler(request.method, request.url, body)
        response = AsyncHttpResponseImpl(
            request=request,
            internal_response=None,
            status_code=status,
            reason="",
            content_type="application/json",
            headers=CaseInsensitiveDict({"Content-Type": "application/json; charset=utf-8"}),
            stream_download_generator=None,
        )
        response._content = json.dumps(payload).encode() if payload is not None else b""
        response._is_stream_consumed = True
        return response


def _settings(**kw) -> Settings:
    base = dict(
        provider_mode="azure",
        azure_openai_endpoint="https://aoai",
        azure_openai_chat_deployment="c",
        azure_openai_embedding_deployment="e",
        azure_openai_embedding_dimensions=3,
        azure_search_endpoint=ENDPOINT,
        azure_search_api_key="k",
        azure_search_index="idx",
        _env_file=None,
    )
    base.update(kw)
    return Settings(**base)


def _backend(handler) -> tuple[AzureSearchBackend, FakeTransport]:
    transport = FakeTransport(handler)
    return AzureSearchBackend(_settings(), transport=transport), transport


def _search_handler(method, url, body):
    if "docs/search" in url:
        return 200, {"value": [HIT]}
    return 404, {"error": {"code": "NotFound", "message": "nope"}}


@pytest.mark.parametrize(
    ("mode", "has_text", "has_vector"),
    [
        (RetrievalMode.HYBRID, True, True),
        (RetrievalMode.VECTOR, False, True),
        (RetrievalMode.KEYWORD, True, False),
    ],
)
async def test_search_request_shape_per_mode(mode, has_text, has_vector):
    backend, transport = _backend(_search_handler)
    results = await backend.search(
        query_text="cyber insurance",
        query_vector=[0.1, 0.2, 0.3],
        top=4,
        mode=mode,
        search_filter=SearchFilter(owner="o", category=DocumentCategory.SUPPLIER, supplier_id="s1"),
    )
    _, url, body = transport.requests[-1]
    assert "/indexes('idx')/docs/search" in url
    assert body["filter"] == "owner eq 'o' and category eq 'supplier' and supplier_id eq 's1'"
    assert body["top"] == 4
    assert ("search" in body and body["search"] == "cyber insurance") is has_text
    assert ("vectorQueries" in body) is has_vector
    if has_vector:
        assert body["vectorQueries"][0]["fields"] == "embedding"
        assert body["vectorQueries"][0]["vector"] == [0.1, 0.2, 0.3]
    assert results[0].chunk_id == "d-0000" and results[0].section == "Insurance" and results[0].score == 0.9


async def test_ensure_index_creates_index_with_configured_dimensions():
    def handler(method, url, body):
        if method == "GET" and "/indexes" in url:
            return 200, {"value": []}
        if method == "POST" and url.split("?")[0].endswith("/indexes"):
            return 201, body
        return 404, None

    backend, transport = _backend(handler)
    await backend.ensure_index()
    created = next(b for m, u, b in transport.requests if m == "POST")
    embedding = next(f for f in created["fields"] if f["name"] == "embedding")
    assert embedding["dimensions"] == 3
    assert created["name"] == "idx"


async def test_ensure_index_detects_dimension_mismatch():
    # REST representation of an existing index whose vector field was built for 1536 dimensions.
    existing = {
        "name": "idx",
        "fields": [
            {"name": "id", "type": "Edm.String", "key": True},
            {
                "name": "embedding",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "dimensions": 1536,
                "vectorSearchProfile": "embedding-profile",
            },
        ],
    }

    def handler(method, url, body):
        if method == "GET" and "indexes('idx')" in url:
            return 200, existing
        if method == "GET" and "/indexes" in url:
            return 200, {"value": [{"name": "idx"}]}
        return 404, None

    backend, _ = _backend(handler)
    with pytest.raises(IndexSchemaMismatchError):
        await backend.ensure_index()


async def test_upsert_sends_ids_metadata_and_vectors():
    def handler(method, url, body):
        if "docs/search.index" in url:
            return 200, {
                "value": [{"key": d["id"], "status": True, "statusCode": 200} for d in body["value"]]
            }
        return 404, None

    backend, transport = _backend(handler)
    chunk = Chunk(
        chunk_id="d-0000",
        document_id="d",
        source_file="a.md",
        category=DocumentCategory.SUPPLIER,
        supplier_id="s1",
        owner="o",
        content="text",
        section="S",
    )
    await backend.upsert([chunk], [[0.1, 0.2, 0.3]])
    doc = transport.requests[-1][2]["value"][0]
    assert doc["@search.action"] == "mergeOrUpload"
    assert doc["id"] == "d-0000" and doc["owner"] == "o" and doc["embedding"] == [0.1, 0.2, 0.3]
    assert "chunk_id" not in doc and "score" not in doc


async def test_auth_failure_is_mapped_to_search_error():
    backend, _ = _backend(lambda m, u, b: (403, {"error": {"code": "Forbidden", "message": "denied"}}))
    with pytest.raises(SearchError) as info:
        await backend.search(
            query_text="x",
            query_vector=None,
            top=1,
            mode=RetrievalMode.KEYWORD,
            search_filter=SearchFilter(owner="o"),
        )
    assert "HTTP 403" in info.value.message or "authentication" in info.value.message
