"""
api_v1.py — V1 API router.

Original API endpoints mounted under /v1 prefix.
Maintains backward compatibility.
"""

from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter, HTTPException, Request

from .api_shared import (
    ServingState,
    DecideResponse,
    ErrorResponse,
    PredictProbaResponse,
    RecordsPayload,
    ReloadResponse,
    exec_predict_proba,
    exec_decide,
    load_serving_state,
    set_shared_app_ref as _set_app_ref,
    get_shared_app_ref,
    get_serving_state_for_router as _get_serving_state,
)
from .metrics import INFERENCE_ERRORS

router_v1 = APIRouter(prefix="/v1", tags=["v1"])


@router_v1.post(
    "/predict_proba",
    response_model=PredictProbaResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def v1_predict_proba(payload: RecordsPayload) -> PredictProbaResponse:
    serving = _get_serving_state()
    try:
        proba, schema_report, model_name = exec_predict_proba(
            payload, serving, "v1.predict_proba"
        )
        return PredictProbaResponse(
            n=int(len(proba)),
            proba=proba,
            schema_report=schema_report,
        )
    except ValueError as e:
        INFERENCE_ERRORS.labels(
            endpoint="v1.predict_proba", model=_model_name(serving)
        ).inc()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # server-side failure → 500, not 400 (#19)
        INFERENCE_ERRORS.labels(
            endpoint="v1.predict_proba", model=_model_name(serving)
        ).inc()
        raise HTTPException(status_code=500, detail=str(e))


@router_v1.post(
    "/decide",
    response_model=DecideResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def v1_decide(payload: RecordsPayload) -> DecideResponse:
    serving = _get_serving_state()
    try:
        actions_df, pred_report, model_name = exec_decide(payload, serving, "v1.decide")
        return DecideResponse(
            n=int(len(actions_df)),
            results=actions_df.to_dict(orient="records"),
            report=pred_report,
        )
    except ValueError as e:
        INFERENCE_ERRORS.labels(endpoint="v1.decide", model=_model_name(serving)).inc()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # server-side failure → 500 (#19)
        INFERENCE_ERRORS.labels(endpoint="v1.decide", model=_model_name(serving)).inc()
        raise HTTPException(status_code=500, detail=str(e))


def _model_name(serving: ServingState | None) -> str:
    """Extract model name from serving state safely."""
    return str(
        getattr(getattr(serving, "policy", None), "selected_model_artifact", "") or ""
    )


@router_v1.post(
    "/reload",
    response_model=ReloadResponse,
    responses={403: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def v1_reload(request: Request) -> ReloadResponse:
    expected_admin = os.getenv("DS_ADMIN_KEY")
    if expected_admin and request.headers.get("x-admin-key") != expected_admin:
        raise HTTPException(status_code=403, detail="x-admin-key header gereklidir.")
    _app = get_shared_app_ref()
    lock = (
        getattr(_app.state if _app else None, "_reload_lock", None)
        or asyncio.Lock()
    )
    async with lock:
        try:
            serving = load_serving_state()
            if _app is not None:
                _app.state.serving = serving
            return {
                "status": "ok",
                "message": "Serving state reloaded",
                "model": serving.policy.selected_model,
                "policy_path": str(serving.policy_path),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Reload failed: {e}")
