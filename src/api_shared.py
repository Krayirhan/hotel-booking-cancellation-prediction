"""
api_shared.py — Shared models, state accessors, and utilities for versioned API routers.

Extracted to avoid circular imports between api.py ↔ api_v1.py / api_v2.py.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Literal

from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict

from .config import ExperimentConfig, Paths
from .policy import load_decision_policy
from .predict import load_feature_spec
from .utils import get_logger, sha256_file

import joblib

logger = get_logger("api_shared")


# ─── Serving State ─────────────────────────────────────────────────────
@dataclass
class ServingState:
    model: Any
    policy_path: Path
    feature_spec: Dict[str, Any]
    policy: Any


# ─── Request / Response Models ─────────────────────────────────────────
class RecordsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: List[Dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str


class ReadyResponse(BaseModel):
    status: Literal["ok"]
    service: Literal["ready"]
    model: str
    policy_path: str


class SchemaReportResponse(BaseModel):
    missing_columns: List[str] = Field(default_factory=list)
    extra_columns: List[str] = Field(default_factory=list)
    feature_count_expected: int
    feature_count_input: int
    feature_count_used: int


class PredictProbaResponse(BaseModel):
    n: int
    proba: List[float]
    schema_report: SchemaReportResponse


class DecideResultItem(BaseModel):
    proba: float
    action: int
    threshold_used: float
    max_action_rate_used: float | None = None
    model_used: str


class DecideReportResponse(BaseModel):
    missing_columns: List[str] = Field(default_factory=list)
    extra_columns: List[str] = Field(default_factory=list)
    feature_count_expected: int
    feature_count_input: int
    feature_count_used: int
    n_rows: int
    predicted_action_rate: float
    threshold_used: float
    max_action_rate_used: float | None = None
    model_used: str
    ranking_mode: str


class DecideResponse(BaseModel):
    n: int
    results: List[DecideResultItem]
    report: DecideReportResponse


class ReloadResponse(BaseModel):
    status: Literal["ok"]
    message: str
    model: str
    policy_path: str


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    request_id: str | None = None


# ─── State Loading ─────────────────────────────────────────────────────
def load_serving_state() -> ServingState:
    cfg = ExperimentConfig()
    paths = Paths()
    active_slot_path = paths.reports_metrics / "active_slot.json"
    if active_slot_path.exists():
        slot_payload = json.loads(active_slot_path.read_text(encoding="utf-8"))
        slot = str(slot_payload.get("active_slot", "default"))
        if slot in {"blue", "green"}:
            policy_path = paths.reports_metrics / f"decision_policy.{slot}.json"
        else:
            policy_path = paths.reports_metrics / "decision_policy.json"
    else:
        policy_path = paths.reports_metrics / "decision_policy.json"
    policy = load_decision_policy(policy_path)
    if policy.raw.get("policy_version") != cfg.contract.policy_version:
        raise RuntimeError("Policy contract version mismatch")

    model_artifact = policy.selected_model_artifact
    if not model_artifact:
        raise RuntimeError("Policy does not contain selected_model_artifact")

    model_path = paths.project_root / model_artifact
    if not model_path.exists():
        raise RuntimeError(f"Model artifact not found: {model_path}")

    expected_sha = policy.raw.get("selected_model_sha256")
    if expected_sha:
        actual_sha = sha256_file(str(model_path))
        if actual_sha != expected_sha:
            raise RuntimeError("Model checksum mismatch")

    model = joblib.load(model_path)

    run_id = str(policy.raw.get("run_id", ""))
    run_feature_spec = paths.reports_metrics / run_id / "feature_spec.json"
    global_feature_spec = paths.reports / "feature_spec.json"
    feature_spec_path = (
        run_feature_spec if run_feature_spec.exists() else global_feature_spec
    )
    feature_spec = load_feature_spec(feature_spec_path)
    if feature_spec.get("schema_version") != cfg.contract.feature_schema_version:
        raise RuntimeError("Feature schema contract version mismatch")

    # Optional schema contract artifact check (versioned schema metadata)
    run_schema_contract = paths.reports_metrics / run_id / "schema_contract.json"
    global_schema_contract = paths.reports_metrics / "schema_contract.json"
    schema_contract_path = (
        run_schema_contract if run_schema_contract.exists() else global_schema_contract
    )
    if schema_contract_path.exists():
        schema_contract = json.loads(schema_contract_path.read_text(encoding="utf-8"))
        if schema_contract.get("schema_version") != cfg.contract.feature_schema_version:
            raise RuntimeError("Schema contract artifact version mismatch")
        feature_spec["_schema_contract"] = schema_contract

    # Inject reference stats & categories for inference-time validation
    ref_stats_path = paths.reports_metrics / "reference_stats.json"
    if ref_stats_path.exists():
        feature_spec["_reference_stats"] = json.loads(
            ref_stats_path.read_text(encoding="utf-8")
        )
    ref_cats_path = paths.reports_metrics / "reference_categories.json"
    if ref_cats_path.exists():
        feature_spec["_reference_categories"] = json.loads(
            ref_cats_path.read_text(encoding="utf-8")
        )
    # Reference expected volume for per-request anomaly checks
    lineage_path = paths.reports_metrics / "data_lineage_preprocess.json"
    if lineage_path.exists():
        try:
            lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
            expected_rows = int(lineage.get("processed_rows", 0) or 0)
            if expected_rows > 0:
                feature_spec["_reference_volume_rows"] = expected_rows
        except Exception:  # nosec B110 — optional enrichment; lineage file may be malformed
            pass

    return ServingState(
        model=model,
        policy_path=policy_path,
        feature_spec=feature_spec,
        policy=policy,
    )


# ─── Shared Inference Helpers (#18 — eliminate copy-paste) ────────────────────
def exec_predict_proba(
    payload: "RecordsPayload",
    serving: "ServingState",
    endpoint: str,
) -> "tuple[list[float], Any, str]":
    """Core predict_proba logic shared by all versioned routers.

    Returns ``(proba_list, schema_report, model_name)``.
    Raises ``ValueError`` for bad client input, ``Exception`` for server errors.
    Callers are responsible for mapping these to appropriate HTTP status codes
    and incrementing INFERENCE_ERRORS.
    """
    import pandas as pd
    from .predict import validate_and_prepare_features
    from .metrics import INFERENCE_ROWS
    from .tracing import trace_inference, set_span_attribute

    max_rows = ExperimentConfig().api.max_payload_records
    if len(payload.records) > max_rows:
        raise ValueError(f"Payload too large. Max records={max_rows}")
    df = pd.DataFrame(payload.records)
    model_name = str(getattr(serving.policy, "selected_model", ""))
    with trace_inference(endpoint, n_rows=len(df), model_name=model_name):
        X, schema_report = validate_and_prepare_features(
            df, serving.feature_spec, fail_on_missing=True
        )
        proba = serving.model.predict_proba(X)[:, 1]
        set_span_attribute("ml.result_count", int(len(proba)))
    INFERENCE_ROWS.labels(endpoint=endpoint, model=model_name).inc(len(proba))
    return [float(x) for x in proba], schema_report, model_name


def exec_decide(
    payload: "RecordsPayload",
    serving: "ServingState",
    endpoint: str,
) -> "tuple[Any, Any, str]":
    """Core decide logic shared by all versioned routers.

    Returns ``(actions_df, pred_report, model_name)``.
    Raises ``ValueError`` for bad client input, ``Exception`` for server errors.
    """
    import pandas as pd
    from .predict import predict_with_policy
    from .metrics import INFERENCE_ROWS
    from .tracing import trace_inference, set_span_attribute

    max_rows = ExperimentConfig().api.max_payload_records
    if len(payload.records) > max_rows:
        raise ValueError(f"Payload too large. Max records={max_rows}")
    df = pd.DataFrame(payload.records)
    model_name = str(serving.policy.selected_model_artifact)
    with trace_inference(endpoint, n_rows=len(df), model_name=model_name):
        actions_df, pred_report = predict_with_policy(
            model=serving.model,
            policy=serving.policy,
            df_input=df,
            feature_spec_payload=serving.feature_spec,
            model_used=model_name,
        )
        set_span_attribute("ml.result_count", int(len(actions_df)))
    INFERENCE_ROWS.labels(endpoint=endpoint, model=model_name).inc(len(actions_df))
    return actions_df, pred_report, model_name


def error_response(
    *,
    status_code: int,
    error_code: str,
    message: str,
    request_id: str | None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error_code": error_code,
            "message": message,
            "request_id": request_id,
        },
    )


# ─── Shared app-ref (eliminates _app_ref duplication in api_v1/api_v2) ──────
_shared_app_ref = None


def set_shared_app_ref(app) -> None:
    """Call once at startup (from api.py) to register the FastAPI app instance."""
    global _shared_app_ref
    _shared_app_ref = app


def get_shared_app_ref():
    """Return the registered FastAPI app instance (may be None before startup)."""
    return _shared_app_ref


def get_serving_state_for_router() -> ServingState:
    """Shared helper used by api_v1 and api_v2 routers.

    Reads serving state from app.state if available, otherwise loads from disk
    and caches it on app.state for subsequent calls.
    """
    if _shared_app_ref is not None:
        serving = getattr(_shared_app_ref.state, "serving", None)
        if serving is not None:
            return serving
    serving = load_serving_state()
    if _shared_app_ref is not None:
        _shared_app_ref.state.serving = serving
    return serving
