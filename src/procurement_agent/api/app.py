"""FastAPI application."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Query, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..config import Settings, get_settings
from ..container import Container, build_container
from ..errors import AppError, ValidationFailure
from ..logging_config import configure_logging
from ..models import (
    AnalyzeRequest,
    DocumentCategory,
    DocumentRecord,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    TaskAccepted,
    TaskRecord,
    TaskStatus,
)
from ..security import require_principal

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def get_container(request: Request) -> Container:
    return request.app.state.container


# Module-level so FastAPI can resolve them under ``from __future__ import annotations``.
Principal = Annotated[str, Depends(require_principal)]
Deps = Annotated[Container, Depends(get_container)]


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level, settings.provider_mode.value)
        app.state.container = container or build_container(settings)
        logger.info(
            "app.started",
            extra={"version": __version__, "retrieval_mode": settings.retrieval_mode.value},
        )
        if not settings.api_key_map:
            logger.warning(
                "app.auth_disabled: API_KEYS is empty; all callers act as 'local-dev'. Do not expose."
            )
        yield
        await app.state.container.close()

    app = FastAPI(
        title="Enterprise AI Agent Workflow Automation Platform",
        version=__version__,
        description="Evidence-grounded supplier evaluation with LangGraph, Azure OpenAI and Azure AI Search.",
        lifespan=lifespan,
    )
    app.dependency_overrides[get_settings] = lambda: settings

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content={"error": exc.to_dict()})

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            provider_mode=settings.provider_mode.value,
            retrieval_mode=settings.retrieval_mode.value,
            auth_enabled=bool(settings.api_key_map),
        )

    @app.post("/documents/upload", response_model=DocumentRecord, status_code=status.HTTP_201_CREATED)
    async def upload_document(
        principal: Principal,
        deps: Deps,
        file: Annotated[UploadFile, File(description="A .txt, .md or .pdf document")],
        category: Annotated[DocumentCategory, Form()],
        supplier_id: Annotated[str | None, Form(pattern=r"^[A-Za-z0-9_\-]{1,64}$")] = None,
    ) -> DocumentRecord:
        # Read at most limit+1 bytes so oversized uploads are rejected without buffering them fully.
        data = await file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise ValidationFailure(f"File exceeds the {settings.max_upload_bytes} byte limit.")
        return deps.ingestion.upload(
            filename=file.filename or "upload",
            data=data,
            category=category,
            supplier_id=supplier_id,
            owner=principal,
        )

    @app.get("/documents", response_model=list[DocumentRecord])
    async def list_documents(principal: Principal, deps: Deps) -> list[DocumentRecord]:
        return deps.registry.list(principal)

    @app.post("/documents/index", response_model=IndexResponse)
    async def index_documents(body: IndexRequest, principal: Principal, deps: Deps) -> IndexResponse:
        indexed = await deps.ingestion.index(body.document_ids, principal)
        return IndexResponse(indexed=indexed, provider_mode=settings.provider_mode.value)

    @app.post(
        "/tasks/analyze",
        response_model=TaskRecord | TaskAccepted,
        status_code=status.HTTP_202_ACCEPTED,
        responses={200: {"model": TaskRecord, "description": "Completed task when wait=true"}},
    )
    async def analyze(
        body: AnalyzeRequest,
        principal: Principal,
        deps: Deps,
        wait: Annotated[bool, Query(description="Run synchronously and return the finished task")] = False,
    ):
        if len(body.task) > settings.max_task_chars:
            raise ValidationFailure(f"task exceeds {settings.max_task_chars} characters")
        if body.requirements and len(body.requirements) > settings.max_requirements:
            raise ValidationFailure(f"at most {settings.max_requirements} requirements are allowed")
        if body.document_ids:
            for doc_id in body.document_ids:  # ownership check before the workflow runs
                deps.registry.get(doc_id, principal)
        record = deps.tasks.create(body, principal)
        if wait:
            record = await deps.tasks.run(record.task_id)
            return JSONResponse(status_code=status.HTTP_200_OK, content=record.model_dump(mode="json"))
        deps.tasks.start(record)
        return TaskAccepted(task_id=record.task_id, status=TaskStatus.PENDING)

    @app.get("/tasks/{task_id}", response_model=TaskRecord)
    async def get_task(task_id: str, principal: Principal, deps: Deps) -> TaskRecord:
        return deps.tasks.get(task_id, principal)

    if settings.serve_ui:
        # Static UI. It only talks to the JSON API above, so auth and ownership rules still apply.
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/", include_in_schema=False)
        async def ui() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})

        samples = settings.samples_dir
        if (samples / "cases.json").is_file() and (samples / "documents").is_dir():
            app.mount("/samples", StaticFiles(directory=samples), name="samples")

    return app
