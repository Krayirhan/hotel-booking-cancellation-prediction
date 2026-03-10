"""test_knowledge_api.py — Tests for knowledge management API endpoints.

Tests:
  POST /chat/knowledge         — ingest_knowledge endpoint
  GET  /chat/knowledge/stats   — knowledge_stats endpoint
  GET  /chat/admin/knowledge   — list knowledge chunks (admin)

Uses FastAPI TestClient mocking the KnowledgeDbStore to avoid pgvector dependency.
"""

from __future__ import annotations

import os
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

os.environ.setdefault("DS_API_KEY", "test-key")
os.environ.setdefault("DS_ENV", "development")

import src.chat.router as router  # noqa: E402

API_KEY = "test-key"


# ── Knowledge DB mock ──────────────────────────────────────────────────────────

class _MockKnowledgeDb:
    """Minimal mock of KnowledgeDbStore for API unit tests."""

    def __init__(self):
        self._chunks: list[dict] = []
        self._next_id = 1

    def create_chunk(self, *, chunk_id, category, tags, title, content, priority=5) -> dict:
        row = {
            "id": self._next_id,
            "chunk_id": chunk_id,
            "category": category,
            "tags": tags,
            "title": title,
            "content": content,
            "priority": priority,
            "is_active": True,
            "has_embedding": False,
        }
        self._chunks.append(row)
        self._next_id += 1
        return row

    def list_chunks(self, *, include_inactive: bool = False) -> list[dict]:
        if include_inactive:
            return list(self._chunks)
        return [c for c in self._chunks if c.get("is_active", True)]

    def count_active(self) -> int:
        return sum(1 for c in self._chunks if c.get("is_active", True))

    def count_embedded(self) -> int:
        return sum(1 for c in self._chunks if c.get("has_embedding", False))

    def get_category_stats(self) -> list[dict]:
        cats: dict[str, int] = {}
        for c in self._chunks:
            if c.get("is_active", True):
                cats[c["category"]] = cats.get(c["category"], 0) + 1
        return [{"category": k, "count": v} for k, v in cats.items()]

    def update_chunk(self, chunk_id: str, updates: dict) -> dict | None:
        for c in self._chunks:
            if c["chunk_id"] == chunk_id:
                c.update(updates)
                return c
        return None

    def delete_chunk(self, chunk_id: str) -> bool:
        before = len(self._chunks)
        self._chunks = [c for c in self._chunks if c["chunk_id"] != chunk_id]
        return len(self._chunks) < before

    def rebuild_embeddings(self) -> int:
        return 0


# ── Helper ─────────────────────────────────────────────────────────────────────

def _ingest_payload(**overrides) -> router.KnowledgeIngestRequest:
    base = {
        "chunk_id": "test_001",
        "category": "test",
        "tags": ["test", "unit"],
        "title": "Test Chunk",
        "content": "Bu bir test bilgi parçasıdır.",
        "priority": 5,
    }
    base.update(overrides)
    return router.KnowledgeIngestRequest(**base)


# ── ingest_knowledge tests ─────────────────────────────────────────────────────

def test_ingest_knowledge_no_db_raises_503(monkeypatch):
    monkeypatch.setattr(router, "get_knowledge_db_store", lambda: None)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(router.ingest_knowledge(_ingest_payload()))
    assert exc.value.status_code == 503


def test_ingest_knowledge_success(monkeypatch):
    db = _MockKnowledgeDb()
    monkeypatch.setattr(router, "get_knowledge_db_store", lambda: db)

    payload = _ingest_payload(chunk_id="pol_new", category="iptal", title="Yeni politika")
    result = asyncio.run(router.ingest_knowledge(payload))

    assert result["chunk_id"] == "pol_new"
    assert result["category"] == "iptal"


def test_ingest_knowledge_creates_chunk_in_db(monkeypatch):
    db = _MockKnowledgeDb()
    monkeypatch.setattr(router, "get_knowledge_db_store", lambda: db)

    payload = _ingest_payload(chunk_id="c1", category="upsell")
    asyncio.run(router.ingest_knowledge(payload))

    chunks = db.list_chunks()
    assert any(c["chunk_id"] == "c1" for c in chunks)


def test_ingest_knowledge_exception_returns_500(monkeypatch):
    bad_db = MagicMock()
    bad_db.create_chunk.side_effect = RuntimeError("DB exploded")
    monkeypatch.setattr(router, "get_knowledge_db_store", lambda: bad_db)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(router.ingest_knowledge(_ingest_payload()))
    assert exc.value.status_code == 500


# ── knowledge_stats tests ──────────────────────────────────────────────────────

def test_knowledge_stats_no_db_returns_503(monkeypatch):
    monkeypatch.setattr(router, "get_knowledge_db_store", lambda: None)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(router.knowledge_stats())
    assert exc.value.status_code == 503


def test_knowledge_stats_empty_db(monkeypatch):
    db = _MockKnowledgeDb()
    monkeypatch.setattr(router, "get_knowledge_db_store", lambda: db)

    result = asyncio.run(router.knowledge_stats())
    assert result["total"] == 0
    assert result["active"] == 0
    assert isinstance(result.get("by_category", []), list)


def test_knowledge_stats_reflects_ingested_chunks(monkeypatch):
    db = _MockKnowledgeDb()
    monkeypatch.setattr(router, "get_knowledge_db_store", lambda: db)

    # Ingest two chunks in different categories
    for i in range(3):
        asyncio.run(router.ingest_knowledge(_ingest_payload(chunk_id=f"c{i}", category="iptal")))
    asyncio.run(router.ingest_knowledge(_ingest_payload(chunk_id="c99", category="upsell")))

    result = asyncio.run(router.knowledge_stats())
    assert result["active"] == 4

    cats = {row["category"]: row["count"] for row in result.get("by_category", [])}
    assert cats.get("iptal") == 3
    assert cats.get("upsell") == 1


# ── admin list_knowledge tests ─────────────────────────────────────────────────

def test_list_knowledge_no_db_returns_503(monkeypatch):
    monkeypatch.setattr(router, "get_knowledge_db_store", lambda: None)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(router.list_knowledge())
    assert exc.value.status_code == 503


def test_list_knowledge_returns_all_active(monkeypatch):
    db = _MockKnowledgeDb()
    monkeypatch.setattr(router, "get_knowledge_db_store", lambda: db)

    for i in range(3):
        asyncio.run(router.ingest_knowledge(_ingest_payload(chunk_id=f"k{i}", category="test")))

    result = asyncio.run(router.list_knowledge(include_inactive=False))
    assert isinstance(result, list)
    assert len(result) == 3
