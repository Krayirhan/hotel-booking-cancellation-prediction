"""
api_v2.py — V2 API router.

Enhanced API with:
  - Richer response metadata (request_id, api_version, latency_ms)
  - Batch confidence intervals
  - Deprecation headers for sunset planning
"""

from __future__ import annotations

import asyncio
import os
import time
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
    get_serving_state_for_router as _get_serving_state,
)
from .metrics import INFERENCE_ERRORS
from .tracing import set_span_attribute

router_v2 = APIRouter(prefix="/v2", tags=["v2"])


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
    if expected_admin and request.headers.get("x-admin-key") != expected_admin:
        raise HTTPException(status_code=403, detail="x-admin-key header gereklidir.")
    _app = get_shared_app_ref()
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
