"""
tracing.py — OpenTelemetry distributed tracing instrumentation.

Provides:
- Trace context propagation (W3C TraceContext)
- Automatic FastAPI span creation
- Custom span helpers for ML inference
- Jaeger/Tempo export via OTLP

Configuration via environment variables:
  OTEL_ENABLED=true
  OTEL_SERVICE_NAME=hotel-booking-cancellation-prediction-api
  OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
  OTEL_TRACES_SAMPLER=parentbased_traceidalgo
  OTEL_TRACES_SAMPLER_ARG=0.1        (sample 10% in production)
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Any, Generator, Optional

from .utils import get_logger

logger = get_logger("tracing")

_tracer = None
_initialized = False
_init_lock = threading.Lock()  # guards one-time initialization


def _otel_enabled() -> bool:
    return os.getenv("OTEL_ENABLED", "false").lower() in ("true", "1", "yes")


def _otlp_insecure() -> bool:
    """Use insecure gRPC only when explicitly requested (default: False for prod safety)."""
    return os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "false").lower() in (
        "true",
        "1",
        "yes",
    )


def init_tracing(service_name: str = "hotel-booking-cancellation-prediction-api") -> None:
    """Initialize OpenTelemetry tracing if OTEL_ENABLED=true."""
    global _tracer, _initialized

    with _init_lock:
        if _initialized:
            return
        _init_tracing_locked(service_name)


def _init_tracing_locked(service_name: str) -> None:
    """Inner init — must be called while holding _init_lock."""
    global _tracer, _initialized

    if not _otel_enabled():
        logger.info("OpenTelemetry tracing disabled (OTEL_ENABLED != true)")
        _initialized = True
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        resource = Resource.create(
            {
                SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME", service_name),
                "service.version": os.getenv("OTEL_SERVICE_VERSION", "1.1.0"),
                "deployment.environment": os.getenv(
                    "OTEL_DEPLOYMENT_ENV", "production"
                ),
            }
        )

        provider = TracerProvider(resource=resource)

        otlp_endpoint = os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
        )
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=_otlp_insecure())
        provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(__name__)
        _initialized = True

        logger.info(
            f"OpenTelemetry tracing initialized: service={service_name} "
            f"endpoint={otlp_endpoint}"
        )

    except ImportError as e:
        logger.warning(f"OpenTelemetry packages not installed, tracing disabled: {e}")
        _initialized = True
    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry: {e}")
        _initialized = True


def instrument_fastapi(app: Any) -> None:
    """Instrument a FastAPI app with OpenTelemetry middleware."""
    if not _otel_enabled():
        return

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls="health,ready,metrics",
        )
        logger.info("FastAPI OpenTelemetry instrumentation applied")
    except ImportError:
        logger.warning("opentelemetry-instrumentation-fastapi not installed")
    except Exception as e:
        logger.error(f"Failed to instrument FastAPI: {e}")


def get_tracer():
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None:
        try:
            from opentelemetry import trace

            _tracer = trace.get_tracer(__name__)
        except ImportError:
            return None
    return _tracer


@contextmanager
def trace_span(
    name: str,
    attributes: Optional[dict] = None,
) -> Generator:
    """Create a traced span. Falls back to no-op if tracing is disabled."""
    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    try:
        with tracer.start_as_current_span(name) as span:
            if attributes:
                for k, v in attributes.items():
                    span.set_attribute(k, v)
            yield span
    except Exception as exc:
        logger.debug("trace_span fallback (span creation failed): %s", exc)
        yield None


@contextmanager
def trace_inference(
    endpoint: str,
    n_rows: int,
    model_name: str = "",
) -> Generator:
    """Convenience span for ML inference operations."""
    attrs = {
        "ml.endpoint": endpoint,
        "ml.n_rows": n_rows,
        "ml.model_name": model_name,
    }
    with trace_span(f"inference.{endpoint}", attributes=attrs) as span:
        yield span


def add_span_event(name: str, attributes: Optional[dict] = None) -> None:
    """Add an event to the current active span."""
    try:
        from opentelemetry import trace as trace_api

        span = trace_api.get_current_span()
        if span and span.is_recording():
            span.add_event(name, attributes=attributes or {})
    except (ImportError, Exception):
        pass


def set_span_attribute(key: str, value: Any) -> None:
    """Set an attribute on the current active span."""
    try:
        from opentelemetry import trace as trace_api

        span = trace_api.get_current_span()
        if span and span.is_recording():
            span.set_attribute(key, value)
    except (ImportError, Exception):
        pass
