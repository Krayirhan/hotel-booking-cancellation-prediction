from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..utils import get_logger

logger = get_logger("chat.memory")


@dataclass
class Message:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ChatSession:
    session_id: str
    customer_data: dict[str, Any]
    risk_score: float
    risk_label: str
    messages: list[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def add_message(self, *, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))
        self.last_active = time.time()

    def to_ollama_messages(self, *, system_prompt: str) -> list[dict[str, str]]:
        payload: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for msg in self.messages:
            payload.append({"role": msg.role, "content": msg.content})
        return payload

    def is_expired(self, *, ttl_seconds: int) -> bool:
        return (time.time() - self.last_active) > ttl_seconds


# ── In-memory session store (default / fallback) ───────────────────────────────


class SessionStore:
    """Thread-safe in-memory session store with TTL-based expiry.

    Cleanup is driven by:
    1. Lazy eviction on ``get_session`` for the accessed session.
    2. Bulk ``_cleanup_expired()`` called by the periodic background task (#29).
    """

    def __init__(self, *, ttl_seconds: int = 3600, max_history: int = 20) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_history = max_history
        self._sessions: dict[str, ChatSession] = {}
        self._lock = threading.Lock()  # guards all _sessions dict access

    def create_session(
        self,
        *,
        customer_data: dict[str, Any],
        risk_score: float,
        risk_label: str,
    ) -> ChatSession:
        session = ChatSession(
            session_id=str(uuid.uuid4()),
            customer_data=customer_data,
            risk_score=risk_score,
            risk_label=risk_label,
        )
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get_session(self, *, session_id: str) -> ChatSession | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.is_expired(ttl_seconds=self.ttl_seconds):
                del self._sessions[session_id]
                return None
        return session

    def save_session(self, session: ChatSession) -> None:
        """No-op for in-memory store — object reference is already live."""
        pass

    def trim_history(self, *, session: ChatSession) -> None:
        if len(session.messages) <= self.max_history:
            return
        session.messages = (
            session.messages[:2] + session.messages[-(self.max_history - 2) :]
        )

    def _cleanup_expired(self) -> None:
        """Remove all sessions that have exceeded their idle TTL (#29)."""
        with self._lock:
            expired = [
                sid
                for sid, session in self._sessions.items()
                if session.is_expired(ttl_seconds=self.ttl_seconds)
            ]
            for sid in expired:
                self._sessions.pop(sid, None)


# ── Redis-backed session store (#25) ──────────────────────────────────────────


class RedisSessionStore:
    """Redis-backed session store that survives API restarts (#25).

    Sessions are stored as JSON blobs with TTL matching the session idle timeout.
    After any mutation (add_message, trim_history) callers must call
    ``save_session()`` to persist the updated state.
    """

    def __init__(
        self, redis_client: Any, *, ttl_seconds: int = 3600, max_history: int = 20
    ) -> None:
        self._r = redis_client
        self.ttl_seconds = ttl_seconds
        self.max_history = max_history
        self._prefix = "ds:chat:sess:"

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    def _serialize(self, session: ChatSession) -> str:
        return json.dumps(
            {
                "session_id": session.session_id,
                "customer_data": session.customer_data,
                "risk_score": session.risk_score,
                "risk_label": session.risk_label,
                "created_at": session.created_at,
                "last_active": session.last_active,
                "messages": [
                    {"role": m.role, "content": m.content, "timestamp": m.timestamp}
                    for m in session.messages
                ],
            }
        )

    def _deserialize(self, raw: str) -> ChatSession:
        d = json.loads(raw)
        session = ChatSession(
            session_id=d["session_id"],
            customer_data=d["customer_data"],
            risk_score=d["risk_score"],
            risk_label=d["risk_label"],
        )
        session.created_at = d["created_at"]
        session.last_active = d["last_active"]
        session.messages = [
            Message(role=m["role"], content=m["content"], timestamp=m["timestamp"])
            for m in d.get("messages", [])
        ]
        return session

    def create_session(
        self,
        *,
        customer_data: dict[str, Any],
        risk_score: float,
        risk_label: str,
    ) -> ChatSession:
        session = ChatSession(
            session_id=str(uuid.uuid4()),
            customer_data=customer_data,
            risk_score=risk_score,
            risk_label=risk_label,
        )
        self._r.setex(
            self._key(session.session_id), self.ttl_seconds, self._serialize(session)
        )
        return session

    def get_session(self, *, session_id: str) -> ChatSession | None:
        raw = self._r.get(self._key(session_id))
        if raw is None:
            return None
        try:
            session = self._deserialize(raw)
        except Exception:
            return None
        # Slide the TTL window on access
        self._r.expire(self._key(session_id), self.ttl_seconds)
        return session

    def save_session(self, session: ChatSession) -> None:
        """Persist the (mutated) session back to Redis."""
        remaining = self._r.ttl(self._key(session.session_id))
        ttl = max(60, remaining if remaining > 0 else self.ttl_seconds)
        self._r.setex(self._key(session.session_id), ttl, self._serialize(session))

    def trim_history(self, *, session: ChatSession) -> None:
        if len(session.messages) <= self.max_history:
            return
        session.messages = (
            session.messages[:2] + session.messages[-(self.max_history - 2) :]
        )

    def _cleanup_expired(self) -> None:
        """Redis handles expiry via TTL — this is a no-op for the Redis backend."""
        pass


# ── Factory ────────────────────────────────────────────────────────────────────

_store: SessionStore | RedisSessionStore | None = None


def get_session_store() -> SessionStore | RedisSessionStore:
    global _store
    if _store is not None:
        return _store
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            import redis as _redis  # type: ignore[import]

            client = _redis.Redis.from_url(
                redis_url, decode_responses=True, socket_timeout=2
            )
            client.ping()
            ttl = int(os.getenv("CHAT_SESSION_TTL_SECONDS", "3600"))
            _store = RedisSessionStore(client, ttl_seconds=ttl)
            logger.info("Chat session store: Redis backend active")
            return _store
        except Exception as exc:
            logger.warning(
                "Chat session store: Redis unavailable, using in-memory. reason=%s", exc
            )
    _store = SessionStore()
    return _store
