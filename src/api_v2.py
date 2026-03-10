"""
api_v2.py — V2 API router.

Enhanced API with:
  - Richer response metadata (request_id, api_version, latency_ms)
  - Batch confidence intervals
  - Deprecation headers for sunset planning
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import time
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .api_shared import (
    ServingState,
    DecideResultItem,
    ErrorResponse,
    RecordsPayload,
    SchemaReportResponse,
    exec_predict_proba,
    exec_decide,
    load_serving_state,
    get_shared_app_ref,
    get_serving_state_for_router as _shared_get_serving_state,
)
from .metrics import INFERENCE_ERRORS
from .tracing import set_span_attribute
from .config import Paths

router_v2 = APIRouter(prefix="/v2", tags=["v2"])


# Backward-compat shim for tests/legacy code that monkeypatch module-level _app_ref.
_app_ref = None


def _get_serving_state() -> ServingState:
    if _app_ref is None:
        return _shared_get_serving_state()

    serving = getattr(_app_ref.state, "serving", None)
    if serving is not None:
        return serving

    serving = load_serving_state()
    _app_ref.state.serving = serving
    return serving


# ─── V2 Response Models ────────────────────────────────────────────────
class V2Meta(BaseModel):
    api_version: str = "v2"
    model_used: str = ""
    latency_ms: float = 0.0
    request_id: Optional[str] = None


class V2PredictProbaResponse(BaseModel):
    n: int
    proba: List[float]
    schema_report: SchemaReportResponse
    meta: V2Meta


class V2DecideReportResponse(BaseModel):
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


class V2DecideResponse(BaseModel):
    n: int
    results: List[DecideResultItem]
    report: V2DecideReportResponse
    meta: V2Meta


class V2ReloadResponse(BaseModel):
    status: str
    message: str
    model: str
    policy_path: str
    meta: V2Meta


class V2ExplainResponse(BaseModel):
    run_id: str
    method: str
    scoring: str | None = None
    n_repeats: int | None = None
    n_features: int | None = None
    ranking: list[dict] = Field(default_factory=list)
    shap_summary: dict | None = None
    meta: V2Meta


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _latest_run_id(metrics_root: Path) -> str:
    if not metrics_root.exists():
        raise HTTPException(status_code=404, detail="Explainability run bulunamadi.")

    latest = _read_json(metrics_root / "latest.json") or {}
    run_id = str(latest.get("run_id") or "").strip()
    if run_id:
        return run_id

    run_dirs = sorted([p.name for p in metrics_root.iterdir() if p.is_dir()], reverse=True)
    if run_dirs:
        return run_dirs[0]
    raise HTTPException(
        status_code=404,
        detail="Explainability run bulunamadi.",
    )


def _load_explain_payload(run_id: str) -> tuple[str, dict, dict | None]:
    paths = Paths()
    metrics_root = paths.reports_metrics
    resolved_run_id = _latest_run_id(metrics_root) if run_id in {"latest", "current"} else run_id

    run_dir = metrics_root / resolved_run_id
    if run_id not in {"latest", "current"} and not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run bulunamadi: {resolved_run_id}")

    search_dirs = [d for d in [run_dir, metrics_root] if d.exists()]

    report: dict | None = None
    for directory in search_dirs:
        report = _read_json(directory / "permutation_importance.json")
        if report:
            break

        fallback = _read_json(directory / "feature_importance.json")
        if isinstance(fallback, dict):
            ranking = sorted(
                [
                    {
                        "feature": feature,
                        "importance_mean": importance,
                        "importance_std": None,
                    }
                    for feature, importance in fallback.items()
                ],
                key=lambda item: -(item["importance_mean"] or 0),
            )
            report = {
                "method": "feature_importance",
                "scoring": "unknown",
                "n_repeats": None,
                "n_features": len(ranking),
                "ranking": ranking,
            }
            break

    if report is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Explainability raporu bulunamadi ({resolved_run_id}). "
                "Once `python main.py explain` komutunu calistirin."
            ),
        )

    shap_summary = None
    for directory in search_dirs:
        payload = _read_json(directory / "shap_summary.json")
        if payload:
            shap_summary = payload
            break

    return resolved_run_id, report, shap_summary


# ─── V2 Endpoints ──────────────────────────────────────────────────────
@router_v2.post(
    "/predict_proba",
    response_model=V2PredictProbaResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def v2_predict_proba(
    payload: RecordsPayload, request: Request
) -> V2PredictProbaResponse:
    t0 = time.time()
    serving = _get_serving_state()
    try:
        proba, schema_report, model_name = exec_predict_proba(
            payload, serving, "v2.predict_proba"
        )
        set_span_attribute("ml.api_version", "v2")
        rid = getattr(request.state, "request_id", None)
        return V2PredictProbaResponse(
            n=int(len(proba)),
            proba=proba,
            schema_report=schema_report,
            meta=V2Meta(
                api_version="v2",
                model_used=model_name,
                latency_ms=round((time.time() - t0) * 1000, 2),
                request_id=rid,
            ),
        )
    except ValueError as e:
        INFERENCE_ERRORS.labels(
            endpoint="v2.predict_proba", model=_model_name(serving)
        ).inc()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        INFERENCE_ERRORS.labels(
            endpoint="v2.predict_proba", model=_model_name(serving)
        ).inc()
        raise HTTPException(status_code=500, detail=str(e))


@router_v2.post(
    "/decide",
    response_model=V2DecideResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def v2_decide(payload: RecordsPayload, request: Request) -> V2DecideResponse:
    t0 = time.time()
    serving = _get_serving_state()
    try:
        actions_df, pred_report, model_name = exec_decide(payload, serving, "v2.decide")
        set_span_attribute("ml.api_version", "v2")
        set_span_attribute(
            "ml.threshold_used",
            float(
                pred_report.get("threshold_used", 0)
                if isinstance(pred_report, dict)
                else getattr(pred_report, "threshold_used", 0)
            ),
        )
        rid = getattr(request.state, "request_id", None)
        return V2DecideResponse(
            n=int(len(actions_df)),
            results=actions_df.to_dict(orient="records"),
            report=pred_report,
            meta=V2Meta(
                api_version="v2",
                model_used=model_name,
                latency_ms=round((time.time() - t0) * 1000, 2),
                request_id=rid,
            ),
        )
    except ValueError as e:
        INFERENCE_ERRORS.labels(endpoint="v2.decide", model=_model_name(serving)).inc()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        INFERENCE_ERRORS.labels(endpoint="v2.decide", model=_model_name(serving)).inc()
        raise HTTPException(status_code=500, detail=str(e))


@router_v2.get(
    "/explain/{run_id}",
    response_model=V2ExplainResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def v2_explain(run_id: str, request: Request) -> V2ExplainResponse:
    t0 = time.time()
    resolved_run_id, report, shap_summary = _load_explain_payload(run_id)
    rid = getattr(request.state, "request_id", None)
    return V2ExplainResponse(
        run_id=resolved_run_id,
        method=str(report.get("method") or "unknown"),
        scoring=report.get("scoring"),
        n_repeats=report.get("n_repeats"),
        n_features=report.get("n_features"),
        ranking=list(report.get("ranking") or []),
        shap_summary=shap_summary,
        meta=V2Meta(
            api_version="v2",
            model_used=str(report.get("model_used") or ""),
            latency_ms=round((time.time() - t0) * 1000, 2),
            request_id=rid,
        ),
    )


def _model_name(serving: ServingState | None) -> str:
    return str(
        getattr(getattr(serving, "policy", None), "selected_model_artifact", "") or ""
    )


@router_v2.post(
    "/reload",
    response_model=V2ReloadResponse,
    responses={403: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def v2_reload(request: Request) -> V2ReloadResponse:
    t0 = time.time()
    expected_admin = os.getenv("DS_ADMIN_KEY")
    if expected_admin and not hmac.compare_digest(
        request.headers.get("x-admin-key") or "", expected_admin
    ):
        raise HTTPException(status_code=403, detail="x-admin-key header gereklidir.")
    _app = _app_ref or get_shared_app_ref()
    lock = getattr(_app.state if _app else None, "_reload_lock", None) or asyncio.Lock()
    async with lock:
        try:
            serving = load_serving_state()
            if _app is not None:
                _app.state.serving = serving
            rid = getattr(request.state, "request_id", None)
            return V2ReloadResponse(
                status="ok",
                message="Serving state reloaded",
                model=serving.policy.selected_model,
                policy_path=str(serving.policy_path),
                meta=V2Meta(
                    api_version="v2",
                    model_used=serving.policy.selected_model,
                    latency_ms=round((time.time() - t0) * 1000, 2),
                    request_id=rid,
                ),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Reload failed: {e}")
