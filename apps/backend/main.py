"""
apps/backend/main.py

Production entrypoint for the hotel-booking-cancellation-prediction API server.

Usage
-----
# Start with environment-variable defaults
    python -m apps.backend.main

# Override specific settings
    python -m apps.backend.main --port 9000 --workers 4

# Development hot-reload (forces workers=1)
    python -m apps.backend.main --reload --log-level debug

Docker / Kubernetes
-------------------
    CMD ["python", "-m", "apps.backend.main"]

The actual FastAPI application lives in src/api.py.
This file is purely responsible for process startup, CLI argument
parsing, log configuration, and graceful shutdown.
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger("backend.startup")


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    # Import settings here so environment variables are already resolved
    from apps.backend.settings import settings  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="hotel-booking-cancellation-prediction API server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default=settings.host,
        help="Bind address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.port,
        help="Bind port",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=settings.workers,
        help="Uvicorn worker processes (forced to 1 when --reload is set)",
    )
    parser.add_argument(
        "--log-level",
        default=settings.log_level,
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Log verbosity",
    )
    parser.add_argument(
        "--timeout-graceful-shutdown",
        type=int,
        default=settings.timeout_graceful_shutdown,
        dest="timeout_graceful_shutdown",
        help="Seconds to wait for in-flight requests on SIGTERM",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=settings.reload,
        help="Enable hot-reload (development only, forces workers=1)",
    )
    return parser.parse_args()


# ── Log config ───────────────────────────────────────────────────────────────


def _build_log_config(log_format: str, log_level: str) -> dict:
    """Return a uvicorn-compatible log-config dictionary."""
    level = log_level.upper()
    if log_format == "json":
        fmt = (
            '{"time":"%(asctime)s","level":"%(levelname)s",'
            '"name":"%(name)s","msg":%(message)r}'
        )
    else:
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": fmt, "datefmt": "%Y-%m-%dT%H:%M:%S"},
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stdout",
            },
            "access": {
                "class": "logging.StreamHandler",
                "formatter": "access",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": level, "propagate": False},
            "uvicorn.error": {
                "handlers": ["default"],
                "level": level,
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["access"],
                "level": level,
                "propagate": False,
            },
        },
        "root": {"handlers": ["default"], "level": level},
    }


# ── Server startup ────────────────────────────────────────────────────────────


def run(
    *,
    host: str,
    port: int,
    workers: int,
    log_level: str,
    timeout_graceful_shutdown: int,
    reload: bool,
    log_format: str,
) -> None:
    """Start uvicorn with the given parameters."""
    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn is not installed. Run: pip install uvicorn")
        sys.exit(1)

    if reload and workers > 1:
        logger.warning("--reload is set: forcing workers=1")
        workers = 1

    logger.info(
        "Starting hotel-booking-cancellation-prediction API | host=%s port=%d workers=%d "
        "log_level=%s graceful_shutdown=%ds reload=%s",
        host,
        port,
        workers,
        log_level,
        timeout_graceful_shutdown,
        reload,
    )

    uvicorn_kwargs: dict = dict(
        app="src.api:app",
        host=host,
        port=port,
        log_level=log_level,
        log_config=_build_log_config(log_format, log_level),
        timeout_graceful_shutdown=timeout_graceful_shutdown,
        reload=reload,
        access_log=True,
    )
    # --workers is mutually exclusive with --reload in uvicorn
    if not reload:
        uvicorn_kwargs["workers"] = workers

    uvicorn.run(**uvicorn_kwargs)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()
    from apps.backend.settings import settings  # noqa: PLC0415

    run(
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        timeout_graceful_shutdown=args.timeout_graceful_shutdown,
        reload=args.reload,
        log_format=settings.log_format,
    )


if __name__ == "__main__":
    main()
