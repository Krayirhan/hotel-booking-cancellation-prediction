"""metrics.py — Prometheus metric definitions for the Hotel Booking Cancellation Prediction API.

All Prometheus counters, gauges, and histograms are defined here as module-level
singletons.  Importing this module registers the collectors with the default
registry.  The ``render_metrics()`` helper serialises the registry into the
Prometheus text exposition format for the ``/metrics`` endpoint.

Metric categories:
    - Request-level: total count, latency histogram
    - Inference: row count, error count per endpoint/model
    - Drift & quality: AUC gauge, PSI gauge, action rate, label drift
    - Knowledge retrieval: call count, empty results, hit count, similarity
"""

from __future__ import annotations

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

REQUEST_COUNT = Counter(
    "ds_api_requests_total",
    "Total API requests",
    ["path", "method", "status"],
)

# Buckets cover fast health-checks (1 ms) through slow ML batch inference (30 s).
# Standard Prometheus defaults are not suitable for ML workloads.
_LATENCY_BUCKETS = (
    0.001,
    0.005,
    0.010,
    0.025,
    0.050,
    0.100,  # sub-100 ms
    0.250,
    0.500,
    1.0,
    2.5,
    5.0,  # 100 ms – 5 s  (typical inference)
    10.0,
    30.0,  # slow / batch
)

REQUEST_LATENCY = Histogram(
    "ds_api_request_latency_seconds",
    "API request latency in seconds",
    ["path", "method"],
    buckets=_LATENCY_BUCKETS,
)

INFERENCE_ROWS = Counter(
    "ds_api_inference_rows_total",
    "Total number of records processed by inference endpoints",
    ["endpoint", "model"],
)

INFERENCE_ERRORS = Counter(
    "ds_api_inference_errors_total",
    "Total number of inference errors",
    ["endpoint", "model"],
)

GUEST_RISK_FALLBACK_TOTAL = Counter(
    "ds_guest_risk_fallback_total",
    "Number of guest risk fallback decisions grouped by reason",
    ["reason"],  # model_not_loaded | inference_error
)

# ── Drift & quality gauges (set by monitoring CLI / scheduled job) ──────────

MODEL_AUC = Gauge(
    "ds_model_roc_auc",
    "Latest model ROC-AUC score from evaluation run",
    ["model", "run_id"],
)

PSI_SCORE = Gauge(
    "ds_feature_psi",
    "Population Stability Index for each feature vs reference distribution",
    ["feature"],
)

ACTION_RATE = Gauge(
    "ds_model_action_rate",
    "Fraction of records where the model recommends action=1",
    ["model", "run_id"],
)

LABEL_DRIFT = Gauge(
    "ds_label_drift_rate",
    "Observed positive label rate delta vs reference (abs value)",
    ["run_id"],
)


# ── Knowledge retrieval quality gauges ─────────────────────────────────────

KNOWLEDGE_RETRIEVAL_TOTAL = Counter(
    "ds_knowledge_retrieval_total",
    "Total number of knowledge retrieval calls",
    ["method"],  # 'vector' | 'fallback'
)

KNOWLEDGE_RETRIEVAL_EMPTY = Counter(
    "ds_knowledge_retrieval_empty_total",
    "Knowledge retrievals that returned zero chunks (empty result)",
    ["method"],
)

KNOWLEDGE_RETRIEVAL_HIT_COUNT = Histogram(
    "ds_knowledge_retrieval_hits",
    "Number of chunks returned per retrieval call",
    ["method"],
    buckets=(0, 1, 2, 3, 5, 10),
)

KNOWLEDGE_RETRIEVAL_HIT_RATIO = Gauge(
    "ds_knowledge_retrieval_hit_ratio",
    "Rolling hit ratio of retrievals in the current process window",
    ["method"],
)

KNOWLEDGE_RETRIEVAL_QUALITY_TOTAL = Counter(
    "ds_knowledge_retrieval_quality_total",
    "Retrieval quality buckets by top-1 similarity",
    ["method", "bucket"],  # low | medium | high | no_similarity
)

KNOWLEDGE_SIMILARITY_SCORE = Histogram(
    "ds_knowledge_similarity_score",
    "Cosine similarity score distribution for retrieved chunks (pgvector)",
    buckets=(0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
