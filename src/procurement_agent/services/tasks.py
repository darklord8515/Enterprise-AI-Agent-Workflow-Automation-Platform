"""Task lifecycle: create, run the workflow, record the report or a typed error.

Tasks are kept in process memory and run as background asyncio tasks. This is adequate for a
single-instance demo; a production deployment would use a durable queue and store (see README).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from ..errors import AppError, NotFound
from ..models import AnalyzeRequest, TaskRecord, TaskStatus, utcnow
from ..workflow.graph import run_workflow

logger = logging.getLogger(__name__)


class TaskService:
    def __init__(self, graph: Any) -> None:
        self.graph = graph
        self._tasks: dict[str, TaskRecord] = {}
        self._running: set[asyncio.Task] = set()

    def create(self, request: AnalyzeRequest, owner: str) -> TaskRecord:
        record = TaskRecord(task_id=uuid.uuid4().hex, owner=owner, status=TaskStatus.PENDING, request=request)
        self._tasks[record.task_id] = record
        return record

    def get(self, task_id: str, owner: str) -> TaskRecord:
        record = self._tasks.get(task_id)
        if record is None or record.owner != owner:
            raise NotFound(f"Task '{task_id}' not found")
        return record

    def start(self, record: TaskRecord) -> None:
        task = asyncio.create_task(self.run(record.task_id))
        self._running.add(task)
        task.add_done_callback(self._running.discard)

    async def run(self, task_id: str) -> TaskRecord:
        record = self._tasks[task_id]
        record.status = TaskStatus.RUNNING
        logger.info("task.started", extra={"task_id": task_id})
        try:
            final = await run_workflow(
                self.graph,
                task=record.request.task,
                supplier_id=record.request.supplier_id,
                owner=record.owner,
                requirements=record.request.requirements,
                document_ids=record.request.document_ids,
            )
            record.report = final["report"]
            record.status = TaskStatus.COMPLETED
        except AppError as exc:
            record.status = TaskStatus.FAILED
            record.error = exc.to_dict()
            logger.warning("task.failed", extra={"task_id": task_id, "error": record.error})
        except Exception as exc:  # noqa: BLE001 - never leak internals to the client
            record.status = TaskStatus.FAILED
            record.error = {
                "code": "internal_error",
                "message": "Unexpected workflow failure",
                "retryable": False,
            }
            logger.exception("task.crashed", extra={"task_id": task_id, "exc_type": type(exc).__name__})
        finally:
            record.completed_at = utcnow()
        logger.info("task.finished", extra={"task_id": task_id, "status": record.status})
        return record
