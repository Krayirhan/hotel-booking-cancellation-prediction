# ADR-005: Distributed Tracing with OpenTelemetry

## Status
Accepted — 2026-02-18

## Context
Prometheus metrics provide aggregate observability (request rates, latencies, error rates),
but they cannot answer:

- "Why was *this specific* request slow?"
- "Which model loading step took the longest during reload?"
- "How does inference latency break down: feature preparation vs model.predict_proba?"

We need per-request trace context propagation for debugging production issues.

## Decision

### OpenTelemetry (OTel) Instrumentation

**Module**: `src/tracing.py`

**Architecture**:
```
FastAPI Request
  → OTel middleware (auto-span)
    → trace_inference("decide", n_rows=N)
      → validate_and_prepare_features (child span)
      → model.predict_proba (child span)
    → response
  → OTLP gRPC export → Jaeger/Tempo
```

**Configuration** (all via environment variables):
| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_ENABLED` | `false` | Master switch |
| `OTEL_SERVICE_NAME` | `hotel-booking-cancellation-prediction-api` | Service identifier |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | Collector endpoint |
| `OTEL_DEPLOYMENT_ENV` | `production` | Environment tag |

**Key design decisions**:
1. **Opt-in**: Tracing disabled by default (`OTEL_ENABLED=false`) — zero overhead in production until enabled
2. **Graceful degradation**: If OTel packages not installed, tracing silently disabled
3. **ML-specific attributes**: Custom `ml.*` namespace (`ml.endpoint`, `ml.n_rows`, `ml.model_name`, `ml.threshold_used`)
4. **Health endpoints excluded**: `/health`, `/ready`, `/metrics` not traced (noise reduction)

### Backend: Jaeger (development) / Tempo (production)
- Local dev: `jaegertracing/all-in-one` in docker-compose
- Production: Grafana Tempo (long-term storage, Grafana integration)

## Consequences

### Positive
- Per-request latency breakdown for debugging
- Trace context propagated via W3C TraceContext headers
- ML-specific span attributes enable inference-focused queries
- Zero overhead when disabled

### Negative
- OTel SDK adds ~5 dependencies
- OTLP export adds ~1-2ms per request (batched)
- Sampling needed in production (10% default) to control costs

### Dependencies Added
- `opentelemetry-api`
- `opentelemetry-sdk`
- `opentelemetry-exporter-otlp-proto-grpc`
- `opentelemetry-instrumentation-fastapi`
