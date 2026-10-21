from __future__ import annotations

import time

from .conftest import DATASET


def _upload(client, name, category, supplier_id=None, headers=None, data=None, filename=None):
    form = {"category": category}
    if supplier_id:
        form["supplier_id"] = supplier_id
    content = data if data is not None else (DATASET / "documents" / name).read_bytes()
    return client.post(
        "/documents/upload", files={"file": (filename or name, content)}, data=form, headers=headers or {}
    )


def _ingest(client, headers=None):
    ids = [
        _upload(client, "northgate_procurement_policy.md", "policy", headers=headers).json()["document_id"],
        _upload(client, "halvorsen_certifications.md", "supplier", "halvorsen", headers).json()[
            "document_id"
        ],
        _upload(client, "halvorsen_insurance.md", "supplier", "halvorsen", headers).json()["document_id"],
    ]
    resp = client.post("/documents/index", json={"document_ids": ids}, headers=headers or {})
    assert resp.status_code == 200, resp.text
    return ids


def test_health_reports_provider_mode(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["provider_mode"] == "mock"
    assert body["auth_enabled"] is False


def test_end_to_end_upload_index_analyze_sync(client):
    _ingest(client)
    resp = client.post(
        "/tasks/analyze?wait=true",
        json={"task": "Evaluate Halvorsen against policy", "supplier_id": "halvorsen"},
    )
    assert resp.status_code == 200, resp.text
    task = resp.json()
    assert task["status"] == "completed"
    report = task["report"]
    assert report["provider_mode"] == "mock/mock/in_memory"
    assert len(report["findings"]) == 8
    assert client.get(f"/tasks/{task['task_id']}").json()["status"] == "completed"


def test_async_task_can_be_polled(client):
    _ingest(client)
    resp = client.post(
        "/tasks/analyze",
        json={
            "task": "Evaluate Halvorsen",
            "supplier_id": "halvorsen",
            "requirements": ["The supplier must hold ISO 9001 quality management certification."],
        },
    )
    assert resp.status_code == 202
    task_id = resp.json()["task_id"]
    for _ in range(50):
        status = client.get(f"/tasks/{task_id}").json()["status"]
        if status in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert status == "completed"


def test_failed_task_returns_typed_error(client):
    supplier_only = _upload(client, "halvorsen_certifications.md", "supplier", "halvorsen").json()
    client.post("/documents/index", json={"document_ids": [supplier_only["document_id"]]})
    task = client.post(
        "/tasks/analyze?wait=true", json={"task": "Evaluate Halvorsen", "supplier_id": "halvorsen"}
    ).json()
    assert task["status"] == "failed"
    assert task["error"]["failed_node"] == "planner"


def test_auth_required_when_keys_configured(auth_client):
    assert auth_client.get("/health").status_code == 200  # health stays public
    assert _upload(auth_client, "halvorsen_insurance.md", "supplier", "halvorsen").status_code == 401
    bad = {"X-API-Key": "wrong"}
    assert _upload(auth_client, "halvorsen_insurance.md", "supplier", "halvorsen", bad).status_code == 401
    good = {"X-API-Key": "alice-key"}
    assert _upload(auth_client, "halvorsen_insurance.md", "supplier", "halvorsen", good).status_code == 201


def test_documents_and_tasks_are_isolated_between_principals(auth_client):
    alice, bob = {"X-API-Key": "alice-key"}, {"X-API-Key": "bob-key"}
    ids = _ingest(auth_client, alice)
    # Bob cannot index, reference or list Alice's documents.
    assert auth_client.post("/documents/index", json={"document_ids": ids}, headers=bob).status_code == 404
    assert auth_client.get("/documents", headers=bob).json() == []
    resp = auth_client.post(
        "/tasks/analyze?wait=true",
        json={"task": "Evaluate Halvorsen", "supplier_id": "halvorsen", "document_ids": ids},
        headers=bob,
    )
    assert resp.status_code == 404
    # Bob's own evaluation sees no evidence from Alice's documents.
    task = auth_client.post(
        "/tasks/analyze?wait=true",
        json={
            "task": "Evaluate Halvorsen",
            "supplier_id": "halvorsen",
            "requirements": ["The supplier must hold ISO 9001 quality management certification."],
        },
        headers=bob,
    ).json()
    assert task["report"]["findings"][0]["status"] == "insufficient_evidence"
    # Alice's task is invisible to Bob.
    alice_task = auth_client.post(
        "/tasks/analyze?wait=true",
        json={"task": "Evaluate Halvorsen", "supplier_id": "halvorsen"},
        headers=alice,
    ).json()
    assert auth_client.get(f"/tasks/{alice_task['task_id']}", headers=bob).status_code == 404


def test_upload_rejects_disallowed_extension(client):
    resp = _upload(client, None, "policy", data=b"#!/bin/sh\necho hi", filename="run.sh")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_upload_rejects_fake_pdf_and_binary_text(client):
    assert _upload(client, None, "policy", data=b"not a pdf", filename="x.pdf").status_code == 422
    assert _upload(client, None, "policy", data=b"abc\x00def", filename="x.txt").status_code == 422
    assert _upload(client, None, "policy", data=b"\xff\xfe\xfa", filename="x.md").status_code == 422


def test_upload_rejects_oversized_and_empty_files(client, tmp_path):
    big = b"a" * (5 * 1024 * 1024 + 1)
    assert _upload(client, None, "policy", data=big, filename="big.txt").status_code == 422
    assert _upload(client, None, "policy", data=b"", filename="empty.txt").status_code == 422


def test_upload_requires_supplier_id_for_supplier_docs(client):
    resp = _upload(client, "halvorsen_insurance.md", "supplier")
    assert resp.status_code == 422


def test_upload_sanitizes_path_traversal_filename(client):
    resp = _upload(client, None, "policy", data=b"hello world policy", filename="../../etc/passwd.txt")
    assert resp.status_code == 201
    assert resp.json()["filename"] == "passwd.txt"


def test_analyze_validates_input(client):
    assert client.post("/tasks/analyze", json={"task": "x", "supplier_id": "s"}).status_code == 422
    assert (
        client.post("/tasks/analyze", json={"task": "Evaluate it", "supplier_id": "bad id; drop"}).status_code
        == 422
    )
    too_many = {"task": "Evaluate it", "supplier_id": "s", "requirements": [f"r{i}" for i in range(30)]}
    assert client.post("/tasks/analyze", json=too_many).status_code == 422


def test_unknown_task_is_404(client):
    assert client.get("/tasks/does-not-exist").status_code == 404


def test_web_ui_and_samples_are_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "Evidence Review" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/app.css").status_code == 200
    cases = client.get("/samples/cases.json")
    assert cases.status_code == 200 and cases.json()["documents"]
    assert client.get("/samples/documents/halvorsen_insurance.md").status_code == 200


def test_samples_mount_does_not_expose_other_files(client):
    assert client.get("/samples/../.env").status_code == 404
    assert client.get("/static/../config.py").status_code == 404


def test_ui_can_be_disabled(tmp_path):
    from fastapi.testclient import TestClient

    from procurement_agent.api.app import create_app

    from .conftest import make_settings

    with TestClient(create_app(make_settings(tmp_path, serve_ui=False))) as c:
        assert c.get("/").status_code == 404
        assert c.get("/health").status_code == 200
