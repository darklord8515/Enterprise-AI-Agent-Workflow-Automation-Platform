"""Azure AI Search backend: index management, chunk upload with embeddings, and keyword / vector /
hybrid retrieval with OData metadata filters.

Index layout and query construction follow the patterns in Azure-Samples/azure-search-openai-demo
(MIT): a searchable ``content`` field, filterable source metadata, an HNSW (cosine) vector field whose
dimensions must equal the embedding model's, and hybrid queries that send ``search_text`` together
with a ``VectorizedQuery``. Re-implemented here for this project's schema (owner / supplier / page /
section fields used for access control and citations).
"""

from __future__ import annotations

import logging
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
    ServiceRequestError,
)
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from ..config import RetrievalMode, Settings
from ..errors import IndexSchemaMismatchError, SearchError
from ..models import Chunk
from ..security import odata_quote
from .base import SearchFilter

logger = logging.getLogger(__name__)

EMBEDDING_FIELD = "embedding"
_SELECT = [
    "id",
    "content",
    "document_id",
    "source_file",
    "category",
    "supplier_id",
    "owner",
    "page",
    "section",
    "chunk_index",
    "suspicious",
]
_UPLOAD_BATCH = 100


def build_filter(f: SearchFilter) -> str:
    clauses = [f"owner eq {odata_quote(f.owner)}"]
    if f.category is not None:
        clauses.append(f"category eq {odata_quote(f.category.value)}")
    if f.supplier_id is not None:
        clauses.append(f"supplier_id eq {odata_quote(f.supplier_id)}")
    if f.document_ids:
        # search.in takes a delimited list; document ids are server-generated hex so ',' is safe.
        joined = ",".join(d.replace("'", "''") for d in f.document_ids)
        clauses.append(f"search.in(document_id, '{joined}', ',')")
    return " and ".join(clauses)


def build_index(name: str, dimensions: int) -> SearchIndex:
    string_filter = {"type": SearchFieldDataType.String, "filterable": True}
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SimpleField(name="document_id", **string_filter),
        SimpleField(name="source_file", **string_filter, facetable=True),
        SimpleField(name="category", **string_filter, facetable=True),
        SimpleField(name="supplier_id", **string_filter, facetable=True),
        SimpleField(name="owner", **string_filter),
        SimpleField(name="section", type=SearchFieldDataType.String),
        SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32),
        SimpleField(name="suspicious", type=SearchFieldDataType.Boolean, filterable=True),
        SearchField(
            name=EMBEDDING_FIELD,
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=dimensions,
            vector_search_profile_name="embedding-profile",
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw", parameters=HnswParameters(metric="cosine"))],
        profiles=[VectorSearchProfile(name="embedding-profile", algorithm_configuration_name="hnsw")],
    )
    return SearchIndex(name=name, fields=fields, vector_search=vector_search)


def _map_error(exc: Exception, action: str) -> SearchError:
    if isinstance(exc, ClientAuthenticationError):
        return SearchError(
            f"Azure AI Search authentication failed during {action}. Check AZURE_SEARCH_API_KEY or the "
            "'Search Index Data Contributor' / 'Search Service Contributor' roles."
        )
    if isinstance(exc, ResourceNotFoundError):
        return SearchError(f"Azure AI Search resource not found during {action}. Check the index name.")
    if isinstance(exc, ServiceRequestError):
        return SearchError(f"Could not reach Azure AI Search during {action}.", retryable=True)
    if isinstance(exc, HttpResponseError):
        retryable = exc.status_code in (429, 503)
        return SearchError(
            f"Azure AI Search error during {action}: HTTP {exc.status_code}", retryable=retryable
        )
    return SearchError(f"Unexpected Azure AI Search error during {action}: {type(exc).__name__}")


class AzureSearchBackend:
    name = "azure_ai_search"

    def __init__(self, settings: Settings, transport: Any = None) -> None:
        self.index_name = settings.azure_search_index
        self.dimensions = settings.azure_openai_embedding_dimensions
        key = settings.azure_search_api_key.get_secret_value()
        if key:
            credential: Any = AzureKeyCredential(key)
        else:
            from azure.identity.aio import DefaultAzureCredential

            credential = DefaultAzureCredential()
        client_kwargs: dict[str, Any] = {
            "connection_timeout": settings.azure_search_timeout_seconds,
            "read_timeout": settings.azure_search_timeout_seconds,
            "retry_total": 3,
        }
        if transport is not None:  # test seam: inject a fake HTTP transport
            client_kwargs["transport"] = transport
        self.index_client = SearchIndexClient(settings.azure_search_endpoint, credential, **client_kwargs)
        self.search_client = SearchClient(
            settings.azure_search_endpoint, self.index_name, credential, **client_kwargs
        )
        self._index_checked = False

    async def ensure_index(self) -> None:
        """Create the index if absent; if present, verify the vector dimensions match the embeddings."""
        if self._index_checked:
            return
        try:
            names = [n async for n in self.index_client.list_index_names()]
            if self.index_name not in names:
                await self.index_client.create_index(build_index(self.index_name, self.dimensions))
                logger.info("azure_search.index_created", extra={"index": self.index_name})
            else:
                existing = await self.index_client.get_index(self.index_name)
                field = next((f for f in existing.fields if f.name == EMBEDDING_FIELD), None)
                if field is None or field.vector_search_dimensions != self.dimensions:
                    found = getattr(field, "vector_search_dimensions", None)
                    raise IndexSchemaMismatchError(
                        f"Index '{self.index_name}' vector field has {found} dimensions but the embedding "
                        f"model is configured for {self.dimensions}. Recreate the index or fix "
                        "AZURE_OPENAI_EMBEDDING_DIMENSIONS."
                    )
        except IndexSchemaMismatchError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _map_error(exc, "index setup") from exc
        self._index_checked = True

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        docs = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            doc = chunk.model_dump(mode="json", exclude={"score", "chunk_id"})
            doc["id"] = chunk.chunk_id
            doc[EMBEDDING_FIELD] = vector
            docs.append(doc)
        try:
            for start in range(0, len(docs), _UPLOAD_BATCH):
                results = await self.search_client.merge_or_upload_documents(
                    docs[start : start + _UPLOAD_BATCH]
                )
                failed = [r.key for r in results if not r.succeeded]
                if failed:
                    raise SearchError(
                        f"Failed to index {len(failed)} chunks (e.g. {failed[0]}).", retryable=True
                    )
        except SearchError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _map_error(exc, "upload") from exc

    async def delete_document(self, document_id: str, owner: str) -> None:
        flt = f"document_id eq {odata_quote(document_id)} and owner eq {odata_quote(owner)}"
        try:
            seen: set[str] = set()
            for _ in range(10):  # bounded: 10 x 1000 chunks covers the 5 MB upload limit
                results = await self.search_client.search(
                    search_text="*", filter=flt, select=["id"], top=1000
                )
                keys = [r["id"] async for r in results if r["id"] not in seen]
                if not keys:
                    break
                seen.update(keys)
                await self.search_client.delete_documents([{"id": k} for k in keys])
        except Exception as exc:  # noqa: BLE001
            raise _map_error(exc, "delete") from exc

    async def search(
        self,
        *,
        query_text: str,
        query_vector: list[float] | None,
        search_filter: SearchFilter,
        top: int,
        mode: RetrievalMode,
    ) -> list[Chunk]:
        vector_queries = None
        if mode is not RetrievalMode.KEYWORD and query_vector is not None:
            vector_queries = [
                VectorizedQuery(
                    vector=query_vector, k_nearest_neighbors=min(max(top * 4, 20), 50), fields=EMBEDDING_FIELD
                )
            ]
        search_text = None if mode is RetrievalMode.VECTOR else query_text
        try:
            results = await self.search_client.search(
                search_text=search_text,
                vector_queries=vector_queries,
                filter=build_filter(search_filter),
                select=_SELECT,
                top=top,
            )
            chunks = []
            async for doc in results:
                chunks.append(
                    Chunk(
                        chunk_id=doc["id"],
                        document_id=doc["document_id"],
                        source_file=doc["source_file"],
                        category=doc["category"],
                        supplier_id=doc.get("supplier_id"),
                        owner=doc["owner"],
                        content=doc["content"],
                        page=doc.get("page"),
                        section=doc.get("section"),
                        chunk_index=doc.get("chunk_index") or 0,
                        suspicious=bool(doc.get("suspicious")),
                        score=doc.get("@search.score"),
                    )
                )
            return chunks
        except Exception as exc:  # noqa: BLE001
            raise _map_error(exc, "search") from exc

    async def close(self) -> None:
        await self.search_client.close()
        await self.index_client.close()
