from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import src.chat.router as router


class _HealthyClient:
    def __init__(self, ok: bool):
        self._ok = ok
        self.model = "llama"

    async def health(self):
        return self._ok


class _Orchestrator:
    def __init__(self):
        self.raise_start = None
        self.raise_message = None
        self.raise_summary = None

    async def start_session(self, **kwargs):
        if self.raise_start:
            raise self.raise_start
        return "s-1", "welcome"

    async def quick_actions(self, **kwargs):
        return [{"label": "a", "message": "b"}]

    async def send_message(self, **kwargs):
        if self.raise_message:
            raise self.raise_message
        return "reply"

    async def summary(self, **kwargs):
        if self.raise_summary:
            raise self.raise_summary
        return {
            "session_id": "s-1",
            "risk_score": 0.7,
            "risk_label": "high",
            "message_count": 2,
            "created_at": 1.0,
            "last_active": 2.0,
        }


class _KnowledgeDb:
    def __init__(self):
        self.items = []
        self.calls = 0

    def create_chunk(self, **kwargs):
        self.calls += 1
        self.items.append(
            {
                "id": self.calls,
                "chunk_id": kwargs["chunk_id"],
                "category": kwargs["category"],
                "is_active": True,
                "has_embedding": True,
            }
        )
        return {"id": self.calls, "chunk_id": kwargs["chunk_id"]}

    def list_chunks(self, include_inactive=False):
        if include_inactive:
            return list(self.items)
        return [item for item in self.items if item.get("is_active")]


def _req(headers=None):
    return SimpleNamespace(headers=headers or {})


def test_chat_health_ok_and_degraded(monkeypatch):
    monkeypatch.setattr(router, "get_ollama_client", lambda: _HealthyClient(True))
    r1 = asyncio.run(router.chat_health())
    assert r1["status"] == "ok"
    assert r1["model"] == "llama"

    monkeypatch.setattr(router, "get_ollama_client", lambda: _HealthyClient(False))
    r2 = asyncio.run(router.chat_health())
    assert r2["status"] == "degraded"


def test_start_session_success_and_error(monkeypatch):
    orch = _Orchestrator()
    monkeypatch.setattr(router, "get_orchestrator", lambda: orch)
    body = router.StartSessionRequest(
        customer_data={"lead_time": 10},
        risk_score=0.6,
        risk_label="mid",
    )

    out = asyncio.run(router.start_session(body))
    assert out.session_id == "s-1"
    assert out.quick_actions[0]["label"] == "a"

    orch.raise_start = RuntimeError("boom")
    with pytest.raises(HTTPException) as ex:
        asyncio.run(router.start_session(body))
    assert ex.value.status_code == 500


def test_message_success_404_and_500(monkeypatch):
    orch = _Orchestrator()
    monkeypatch.setattr(router, "get_orchestrator", lambda: orch)
    body = router.ChatMessageRequest(session_id="s-1", message="hello")

    out = asyncio.run(router.message(body))
    assert out.bot_message == "reply"

    orch.raise_message = ValueError("not found")
    with pytest.raises(HTTPException) as ex1:
        asyncio.run(router.message(body))
    assert ex1.value.status_code == 404

    orch.raise_message = RuntimeError("crash")
    with pytest.raises(HTTPException) as ex2:
        asyncio.run(router.message(body))
    assert ex2.value.status_code == 500


def test_summary_success_and_404(monkeypatch):
    orch = _Orchestrator()
    monkeypatch.setattr(router, "get_orchestrator", lambda: orch)

    out = asyncio.run(router.summary("s-1"))
    assert out.session_id == "s-1"

    orch.raise_summary = ValueError("missing")
    with pytest.raises(HTTPException) as ex:
        asyncio.run(router.summary("s-1"))
    assert ex.value.status_code == 404


def test_knowledge_admin_key_guard(monkeypatch):
    monkeypatch.setenv("DS_ADMIN_KEY", "admin-secret")
    with pytest.raises(HTTPException) as ex:
        asyncio.run(router.list_knowledge_chunks(_req(), include_inactive=True))
    assert ex.value.status_code == 403


def test_knowledge_ingest_and_stats(monkeypatch):
    db = _KnowledgeDb()
    monkeypatch.setenv("DS_ADMIN_KEY", "admin-secret")
    monkeypatch.setattr(router, "_require_db_store", lambda: db)

    body = router.KnowledgeIngestRequest(
        source_name="policy-update",
        source_type="text",
        content=" ".join(["retention"] * 260),
        category="playbook",
        tags=["retention"],
        priority=4,
        chunk_size=300,
        chunk_overlap=50,
    )
    req = _req({"x-admin-key": "admin-secret"})
    out = asyncio.run(router.ingest_knowledge(body, req))
    assert out["chunks_created"] >= 1
    assert out["chunks_failed"] == 0

    stats = asyncio.run(router.knowledge_stats(req))
    assert stats["total_chunks"] == out["chunks_created"]
    assert stats["embedded_chunks"] == out["chunks_created"]
    assert stats["categories"]["playbook"] == out["chunks_created"]
