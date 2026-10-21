# Third-party notices and reuse record

This project was designed after inspecting two Microsoft sample repositories. **No source files were
copied verbatim.** Specific design patterns were studied and re-implemented for this project's schema
and workflow. The table below records exactly what was reused, adapted, or newly written, so that no
upstream work is presented as independently developed.

Upstream snapshots inspected (shallow clones, October 2026):

| Repository | Commit | License |
|---|---|---|
| [Azure-Samples/azure-search-openai-demo](https://github.com/Azure-Samples/azure-search-openai-demo) | `558ffd397eddcfe4da5c85ebc285796b367eb9b2` | MIT |
| [Azure-Samples/langchain-agent-python](https://github.com/Azure-Samples/langchain-agent-python) | `ec823a4f1e04e4da39edb36c9904aa00b24ec79e` | Apache-2.0 |

## Reuse record

| Component in this repo | Status | Upstream source and what was taken |
|---|---|---|
| `ingestion/chunker.py` | **Adapted** | Repo A `prepdocslib/textsplitter.py` (`SentenceTextSplitter`): sentence-boundary packing with ~10% tail overlap. Re-implemented with character limits; chunks never cross page/section boundaries. |
| `retrieval/azure_search.py` index schema | **Adapted** | Repo A `prepdocslib/searchmanager.py`: searchable `content`, filterable source metadata fields, HNSW (cosine) vector field with explicit `vector_search_dimensions`. Field set changed (owner, supplier_id, page, section, suspicious). No vectorizers, compression, ACL index projection or semantic config. |
| `retrieval/azure_search.py` query | **Adapted** | Repo A `approaches/approach.py` (`search`, `build_filter`): hybrid query = `search_text` + `VectorizedQuery`; OData single-quote escaping. |
| `providers/azure_openai.py` auth | **Adapted** | Both repos: API key **or** Entra ID via `DefaultAzureCredential` + `get_bearer_token_provider` with the Cognitive Services scope; Repo B's normalisation of the `/openai/v1` endpoint. |
| Bounded retries on rate limits | **Pattern** | Repo A `prepdocslib/embeddings.py` retries on `RateLimitError` with tenacity. Here the OpenAI SDK's built-in bounded backoff (`max_retries`) is used instead. |
| `workflow/state.py` reducers | **Pattern** | Repo B `agent/app/state.py`: typed state with `Annotated[..., reducer]` fields. Different state and reducers. |
| `workflow/nodes.py` evidence reviewer | **Pattern** | Repo B `middleware/validate.py`: "cited doc ids must be among the retrieved doc ids" groundedness check. Extended here with verbatim-quote verification, supplier-vs-policy source rules, bounded revision and downgrade to insufficient evidence. |
| Tool restriction principle | **Pattern** | Repo B `mcp/app.py` restricts SQL to read-only SELECT. This project goes further: the LLM has no tools at all; all retrieval is done by deterministic code. |
| Everything else (FastAPI API, LangGraph graph and routing, planner/analyst prompts, mock providers, in-memory hybrid search, upload validation, auth and ownership scoping, prompt-injection heuristics, evaluation harness and dataset, tests) | **New** | Written for this project. |

Not used from upstream: Quart app, React frontend, Blob Storage / Document Intelligence ingestion,
Cosmos DB chat history, `azd`/Bicep infrastructure (Repo A); LangChain `create_agent` + middleware,
MCP server, PostgreSQL/pgvector, Starlette app (Repo B).

---

## Azure-Samples/azure-search-openai-demo: MIT License

```
MIT License

Copyright (c) 2023 Azure Samples

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Azure-Samples/langchain-agent-python: Apache License 2.0

The upstream repository is licensed under the Apache License, Version 2.0. It ships no NOTICE file.
Full license text: <https://www.apache.org/licenses/LICENSE-2.0>.

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Runtime dependencies

Installed from PyPI under their own licenses (not vendored): FastAPI (MIT), Uvicorn (BSD-3),
Pydantic / pydantic-settings (MIT), LangGraph (MIT), openai (Apache-2.0), azure-search-documents,
azure-identity, azure-core (MIT), aiohttp (Apache-2.0), pypdf (BSD-3), python-multipart (Apache-2.0).
