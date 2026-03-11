"""api_lifespan.py — FastAPI lifespan context manager and startup helpers.

This module is intentionally separate from ``api.py`` so that:
  1. Startup/teardown logic can be unit-tested without importing the full
     FastAPI app (avoids circular imports and heavy fixture setup).
  2. ``api.py`` stays focused on route definitions and HTTP concerns.

Entry-point design note:
  - This module has **no** imports from ``api.py``; it is a strict leaf in the
    import graph (api.py → api_lifespan.py, never the reverse).
  - ``api.py`` imports only the ``lifespan`` symbol from here.

Usage (inside api.py):
    from .api_lifespan import lifespan
    app = FastAPI(..., lifespan=lifespan)
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from .config import ExperimentConfig
from .dashboard import init_dashboard_store
from .db_bootstrap import ensure_required_tables, run_migrations
from .guest_store import init_guest_store
from .rate_limit import BaseRateLimiter, build_rate_limiter
from .tracing import init_tracing, instrument_fastapi
from .user_store import init_user_store, seed_admin
from .utils import get_logger

logger = get_logger("api.lifespan")


# ─── Environment helpers ───────────────────────────────────────────────────────


def _is_non_prod_like_env() -> bool:
    """Return True if the current env is dev / local / test."""
    env = os.getenv("DS_ENV", "development").strip().lower()
    return env in {"dev", "development", "local", "test", "testing"}


def _expected_admin_key() -> str | None:
    """DS_ADMIN_KEY tanımlıysa /reload endpoint'i için ayrı koruma sağlar."""
    return os.getenv("DS_ADMIN_KEY")


def _validate_admin_key_startup() -> None:
    """In non-dev environments, crash-fast if DS_ADMIN_KEY is absent."""
    if _is_non_prod_like_env():
        return
    if not _expected_admin_key():
        env = os.getenv("DS_ENV", "production")
        raise RuntimeError(
            f"DS_ADMIN_KEY must be set when DS_ENV={env!r} to protect reload endpoints."
        )


# ─── Startup sub-steps ────────────────────────────────────────────────────────


def _build_runtime_rate_limiter() -> BaseRateLimiter:
    """Construct the rate limiter from config + environment overrides."""
    cfg = ExperimentConfig().api
    return build_rate_limiter(
        backend=os.getenv("RATE_LIMIT_BACKEND") or cfg.rate_limit_backend,
        redis_url=os.getenv("REDIS_URL") or cfg.redis_url,
        key_prefix=os.getenv("RATE_LIMIT_REDIS_KEY_PREFIX") or cfg.redis_key_prefix,
    )


def _warn_multiworker_rate_limit() -> None:
    """Emit a warning when memory rate limiting is used with multiple workers."""
    cfg = ExperimentConfig().api
    workers = int(os.getenv("WEB_CONCURRENCY", os.getenv("UVICORN_WORKERS", "1")))
    rate_backend = (
        (os.getenv("RATE_LIMIT_BACKEND") or cfg.rate_limit_backend).strip().lower()
    )
    if workers > 1 and rate_backend == "memory":
        logger.warning(
            "RATE LIMIT UYARISI: %d worker algılandı ancak rate_limit_backend='memory'. "
            "Her worker kendi bellek bucket'ını tutar; gerçek sınır %d/dk katına çıkabilir. "
            "Dağıtık doğru limitler için RATE_LIMIT_BACKEND=redis kullanın.",
            workers,
            cfg.rate_limit_per_minute * workers,
        )


# ─── Background tasks ──────────────────────────────────────────────────────────


async def _periodic_chat_cleanup(interval_seconds: int = 300) -> None:
    """Background task: evict idle chat sessions every 5 minutes (#29)."""
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            from .chat.memory import get_session_store  # lazy import

            store = get_session_store()
            if hasattr(store, "_cleanup_expired"):
                store._cleanup_expired()
                logger.debug("Periodic chat session cleanup ran")
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning("Periodic chat cleanup error: %s", exc)


# ─── Lifespan context ──────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan: initialise all services on startup, close on shutdown.

    Startup sequence
    ----------------
    1. OpenTelemetry initialisation
    2. DS_ADMIN_KEY enforcement (non-dev environments)
    3. Database migrations + store initialisation
    4. Model serving state + rate limiter
    5. Multi-worker rate limit warning
    6. Background chat cleanup task

    Shutdown sequence
    -----------------
    1. Cancel background cleanup task
    2. Close Ollama async connection pool
    3. Set ``app.state.shutting_down = True``
    """
    init_tracing(service_name="hotel-booking-cancellation-prediction-api")
    instrument_fastapi(app)
    _validate_admin_key_startup()

    # ── Database & stores ──────────────────────────────────────────────────────
    database_url = os.getenv("DATABASE_URL", "sqlite:///./reports/dashboard.db")
    run_migrations(database_url)
    ensure_required_tables(database_url)

    init_user_store(database_url)
    seed_admin()

    from sqlalchemy import create_engine as _create_engine

    _engine = _create_engine(database_url, pool_pre_ping=True, future=True)
    init_guest_store(_engine)

    try:
        from .chat.knowledge.db_store import init_knowledge_db_store

        init_knowledge_db_store(_engine)
    except Exception as exc:
        logger.warning(
            "Could not initialize knowledge DB store (pgvector): %s — using TF-IDF fallback",
            exc,
        )

    init_dashboard_store()

    # ── Serving state ──────────────────────────────────────────────────────────
    from .api_shared import load_serving_state

    try:
        app.state.serving = load_serving_state()
    except Exception as exc:
        logger.warning("Could not load serving state: %s — API starts degraded", exc)
        app.state.serving = None

    app.state.rate_limiter = _build_runtime_rate_limiter()
    app.state.shutting_down = False
    app.state._reload_lock = asyncio.Lock()

    _warn_multiworker_rate_limit()

    _cleanup_task = asyncio.create_task(_periodic_chat_cleanup())

    yield

    # ── Graceful shutdown ──────────────────────────────────────────────────────
    _cleanup_task.cancel()
    try:
        await _cleanup_task
    except asyncio.CancelledError:
        pass

    try:
        from .chat.ollama_client import get_ollama_client

        await get_ollama_client().aclose()
    except Exception:  # nosec B110 — best-effort teardown
        pass

    app.state.shutting_down = True
