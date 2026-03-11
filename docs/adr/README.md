# Architecture Decision Records (ADR)

This directory contains Architecture Decision Records for the Hotel Booking Cancellation Prediction.

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-001](001-ml-pipeline-architecture.md) | ML Pipeline Architecture | Accepted | 2026-02-16 |
| [ADR-002](002-cost-sensitive-decisioning.md) | Cost-Sensitive Decision Framework | Accepted | 2026-02-16 |
| [ADR-003](003-api-serving-design.md) | API Serving Design | Accepted | 2026-02-17 |
| [ADR-004](004-helm-gitops-deployment.md) | Helm + GitOps Deployment | Accepted | 2026-02-18 |
| [ADR-005](005-distributed-tracing.md) | Distributed Tracing with OpenTelemetry | Accepted | 2026-02-18 |
| [ADR-006](006-data-validation-framework.md) | Data Validation with Pandera | Accepted | 2026-02-18 |
| [ADR-007](007-api-versioning.md) | API Versioning Strategy | Accepted | 2026-02-18 |
| [ADR-008](008-lua-rate-limiting.md) | Lua-Based Distributed Rate Limiting | Accepted | 2026-02-19 |
| [ADR-009](009-model-health-metrics.md) | Prometheus Gauges for Model Health Monitoring | Accepted | 2026-02-19 |
| [ADR-010](010-versioned-api-v1-v2.md) | Versioned API (v1 / v2) Strategy | Accepted | 2026-02-19 |
| [ADR-011](011-ollama-self-hosted.md) | Self-Hosted LLM with Ollama | Accepted | 2026-02-19 |
| [ADR-012](012-postgres-plus-redis.md) | Dual Storage: PostgreSQL + Redis | Accepted | 2026-02-19 |
| [ADR-013](013-calibration-sigmoid-isotonic.md) | Two-Stage Model Calibration | Accepted | 2026-02-19 |

## ADR Format

We follow the [Michael Nygard ADR template](https://github.com/joelparkerhenderson/architecture-decision-record/blob/main/templates/decision-record-template-by-michael-nygard/index.md):

```
# Title

## Status
Proposed | Accepted | Deprecated | Superseded

## Context
What is the issue that we're seeing that is motivating this decision?

## Decision
What is the change that we're proposing and/or doing?

## Consequences
What becomes easier or more difficult to do because of this change?
```
