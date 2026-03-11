"""
apps/backend/settings.py

Runtime settings for the hotel-booking-cancellation-prediction API server.
All configuration is read from environment variables with sensible defaults.
No extra dependencies — uses stdlib os.getenv only.

Docker Compose wires every field via its ``environment:`` block;
.env files are loaded by python-dotenv or the container runtime.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List


# ── Helpers ──────────────────────────────────────────────────────────────────


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).strip().lower() in ("true", "1", "yes")


# ── Settings dataclass ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ServerSettings:
    """
    All runtime configuration, evaluated once at module import.

    Resolves from environment variables (or .env loaded upstream by
    Docker Compose, python-dotenv, or the OS).  Every field maps
    1-to-1 to an environment variable of the same name (upper-case).
    """

    # ── Server ───────────────────────────────────────────────────────────────
    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8000))
    workers: int = field(
        default_factory=lambda: _env_int(
            "WEB_CONCURRENCY", _env_int("UVICORN_WORKERS", 1)
        )
    )
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "info"))
    log_format: str = field(default_factory=lambda: _env("LOG_FORMAT", "json"))
    timeout_graceful_shutdown: int = field(
        default_factory=lambda: _env_int("GRACEFUL_SHUTDOWN_SECONDS", 10)
    )
    reload: bool = field(default_factory=lambda: _env_bool("UVICORN_RELOAD", False))

    # ── Auth & API security ───────────────────────────────────────────────────
    require_api_key: bool = field(
        default_factory=lambda: _env_bool("REQUIRE_API_KEY", True)
    )
    ds_api_key: str = field(default_factory=lambda: _env("DS_API_KEY", ""))
    ds_admin_key: str = field(default_factory=lambda: _env("DS_ADMIN_KEY", ""))

    dashboard_auth_enabled: bool = field(
        default_factory=lambda: _env_bool("DASHBOARD_AUTH_ENABLED", True)
    )
    dashboard_admin_username: str = field(
        default_factory=lambda: _env("DASHBOARD_ADMIN_USERNAME", "ds_readonly")
    )
    dashboard_admin_password: str = field(
        default_factory=lambda: _env("DASHBOARD_ADMIN_PASSWORD", "")
    )
    dashboard_admin_password_admin: str = field(
        default_factory=lambda: _env("DASHBOARD_ADMIN_PASSWORD_ADMIN", "")
    )
    # JSON dict of extra users: {"alice": "pass1", "bob": "pass2"}
    dashboard_extra_users: str = field(
        default_factory=lambda: _env("DASHBOARD_EXTRA_USERS", "{}")
    )
    dashboard_token_ttl_minutes: int = field(
        default_factory=lambda: _env_int("DASHBOARD_TOKEN_TTL_MINUTES", 480)
    )

    # ── Persistence ──────────────────────────────────────────────────────────
    database_url: str = field(
        default_factory=lambda: _env("DATABASE_URL", "sqlite:///./reports/dashboard.db")
    )
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", ""))
    rate_limit_backend: str = field(
        default_factory=lambda: _env("RATE_LIMIT_BACKEND", "memory")
    )
    rate_limit_per_minute: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_PER_MINUTE", 120)
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    cors_allow_origins: str = field(
        default_factory=lambda: _env(
            "CORS_ALLOW_ORIGINS",
            "http://localhost:5173,http://localhost:3000",
        )
    )

    # ── Frontend redirect ─────────────────────────────────────────────────────
    dashboard_url: str = field(
        default_factory=lambda: _env("DASHBOARD_URL", "http://localhost:5173")
    )

    # ── Chat / LLM ────────────────────────────────────────────────────────────
    ollama_base_url: str = field(
        default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: _env("OLLAMA_MODEL", "llama3.2:3b")
    )
    chat_session_ttl_seconds: int = field(
        default_factory=lambda: _env_int("CHAT_SESSION_TTL_SECONDS", 3600)
    )

    # ── OpenTelemetry ─────────────────────────────────────────────────────────
    otel_enabled: bool = field(default_factory=lambda: _env_bool("OTEL_ENABLED", False))
    otel_service_name: str = field(
        default_factory=lambda: _env("OTEL_SERVICE_NAME", "hotel-booking-cancellation-prediction-api")
    )
    otel_exporter_otlp_endpoint: str = field(
        default_factory=lambda: _env(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
        )
    )
    otel_deployment_env: str = field(
        default_factory=lambda: _env("OTEL_DEPLOYMENT_ENV", "production")
    )

    # ── Derived helpers ───────────────────────────────────────────────────────

    @property
    def cors_origins_list(self) -> List[str]:
        """Split CORS_ALLOW_ORIGINS into a Python list."""
        return [x.strip() for x in self.cors_allow_origins.split(",") if x.strip()]

    @property
    def extra_users_dict(self) -> Dict[str, str]:
        """Parse DASHBOARD_EXTRA_USERS JSON into a dict."""
        try:
            return json.loads(self.dashboard_extra_users) or {}
        except Exception:
            return {}


# Singleton — evaluated once at first import
settings = ServerSettings()
