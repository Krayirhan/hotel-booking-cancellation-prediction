from __future__ import annotations

import hmac
import os
import time
import uuid

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse

from .api_shared import (
    ServingState,
    RecordsPayload,
    HealthResponse,
    PredictProbaResponse,
    DecideResponse,
    ReloadResponse,
    ErrorResponse,
    exec_predict_proba,
    exec_decide,
    load_serving_state,
    require_admin_key,
    reload_serving_state_for_app,
    error_response,
)
from .api_lifespan import lifespan, _build_runtime_rate_limiter
from .config import ExperimentConfig
from .dashboard import router_dashboard
from .dashboard_auth import router_dashboard_auth, check_login_attempt_allowed
from .guests import router_guests
from .chat import router_chat
from .metrics import (
    INFERENCE_ERRORS,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    render_metrics,
)
from .utils import get_logger

logger = get_logger("api")


def _api_key_required() -> bool:
    return ExperimentConfig().api.require_api_key


def _expected_api_key() -> str | None:
    cfg = ExperimentConfig().api
    return os.getenv(cfg.api_key_env_var)


def _split_csv_env(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _public_exact_paths() -> set[str]:
    cfg = ExperimentConfig().api
    base = set(cfg.public_paths_exact)
    base.update(_split_csv_env(os.getenv("API_PUBLIC_PATHS_EXACT")))
    return base


def _public_prefix_paths() -> tuple[str, ...]:
    cfg = ExperimentConfig().api
    base = list(cfg.public_paths_prefixes)
    base.extend(_split_csv_env(os.getenv("API_PUBLIC_PATHS_PREFIXES")))
    return tuple(dict.fromkeys(base))


def _is_public_path(path: str) -> bool:
    if path in _public_exact_paths():
        return True
    return any(path.startswith(prefix) for prefix in _public_prefix_paths())


def _load_serving_state() -> ServingState:
    """Wrapper for testability (monkeypatching)."""
    return load_serving_state()


app = FastAPI(
    title="Hotel Booking Cancellation Prediction Serving API",
    version="1.2.0",
    description=(
        "Production inference API for hotel cancellation decisioning. "
        "Supports health checks, readiness checks, probability scoring, policy-based actioning, "
        "runtime metrics, and safe model/policy reload. "
        "Versioned endpoints available under /v1 and /v2 prefixes."
    ),
    lifespan=lifespan,
)


def _cors_allow_origins() -> list[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:5173,http://localhost:3000")
    return [x.strip() for x in raw.split(",") if x.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "x-api-key",
        "x-admin-key",
        "x-request-id",
    ],
)

# ── Register versioned routers ────────────────────────────────────────
from .api_v1 import router_v1  # noqa: E402
from .api_v2 import router_v2  # noqa: E402
from .api_shared import set_shared_app_ref  # noqa: E402

app.include_router(router_v1)
app.include_router(router_v2)
app.include_router(router_dashboard)
app.include_router(router_dashboard_auth)
app.include_router(router_chat)
app.include_router(router_guests)
set_shared_app_ref(app)  # single call — v1 & v2 both use api_shared._shared_app_ref


def _error_response(
    *,
    status_code: int,
    error_code: str,
    message: str,
    request_id: str | None,
):
    return error_response(
        status_code=status_code,
        error_code=error_code,
        message=message,
        request_id=request_id,
    )


def _get_serving_state() -> ServingState:
    serving = getattr(app.state, "serving", None)
    if serving is None:
        try:
            serving = _load_serving_state()
            app.state.serving = serving
        except Exception:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded — service unavailable",
            )
    return serving


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    rid = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.request_id = rid
    started = time.time()
    request_path = request.url.path
    if request.method.upper() == "OPTIONS":
        return await call_next(request)

    # Security gate
    if _api_key_required() and not _is_public_path(request_path):
        expected = _expected_api_key()
        if not expected:
            return _error_response(
                status_code=503,
                error_code="api_key_not_configured",
                message="API key is not configured",
                request_id=rid,
            )
        got = request.headers.get("x-api-key") or ""
        if not hmac.compare_digest(got, expected):
            return _error_response(
                status_code=401,
                error_code="unauthorized",
                message="Unauthorized",
                request_id=rid,
            )

    # Rate limit gate — API key varsa onu kullan (key bazlı limit), yoksa IP'ye düş
    api_key_for_limit = request.headers.get("x-api-key")
    client = api_key_for_limit or (request.client.host if request.client else "unknown")
    limiter = getattr(app.state, "rate_limiter", None)
    if limiter is None:
        limiter = _build_runtime_rate_limiter()
        app.state.rate_limiter = limiter

    # ── Brute-force koruması: /auth/login için sıkı limit (10 istek/dk/IP) ──
    if request_path == "/auth/login" and request.method.upper() == "POST":
        login_ip = request.client.host if request.client else "unknown"
        allowed, retry_after, reason = check_login_attempt_allowed(
            username=None,
            client_ip=login_ip,
        )
        if not allowed:
            return _error_response(
                status_code=429,
                error_code="login_backoff",
                message=(
                    f"Giris denemesi gecici olarak bloklandi ({reason}). "
                    f"{retry_after} saniye sonra tekrar deneyin."
                ),
                request_id=rid,
            )
        if not limiter.allow(f"login:{login_ip}", 10):
            return _error_response(
                status_code=429,
                error_code="login_rate_limit",
                message="Cok fazla giris denemesi. Lutfen bir dakika bekleyin.",
                request_id=rid,
            )

    if not limiter.allow(client, ExperimentConfig().api.rate_limit_per_minute):
        return _error_response(
            status_code=429,
            error_code="rate_limit_exceeded",
            message="Rate limit exceeded",
            request_id=rid,
        )

    # Request size guard (8MB)
    max_bytes = 8 * 1024 * 1024
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > max_bytes:
                return _error_response(
                    status_code=413,
                    error_code="payload_too_large",
                    message="Payload too large",
                    request_id=rid,
                )
        except ValueError:
            return _error_response(
                status_code=400,
                error_code="invalid_content_length",
                message="Invalid content-length",
                request_id=rid,
            )

    if bool(getattr(app.state, "shutting_down", False)):
        return _error_response(
            status_code=503,
            error_code="service_shutting_down",
            message="Service is shutting down",
            request_id=rid,
        )

    response = await call_next(request)
    response.headers["x-request-id"] = rid
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = (
        "default-src 'none'; frame-ancestors 'none'"
    )
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    response.headers["Strict-Transport-Security"] = (
        "max-age=63072000; includeSubDomains"
    )

    latency_ms = round((time.time() - started) * 1000.0, 2)
    logger.info(
        "request completed",
        extra={
            "request_id": rid,
            "path": request_path,
            "method": request.method,
            "status": response.status_code,
            "latency_ms": latency_ms,
        },
    )
    REQUEST_COUNT.labels(
        path=request_path, method=request.method, status=str(response.status_code)
    ).inc()
    REQUEST_LATENCY.labels(path=request_path, method=request.method).observe(
        (time.time() - started)
    )
    return response


@app.get("/health")
def health() -> HealthResponse:
    return {
        "status": "ok",
        "service": "alive",
    }


@app.get("/ready")
def ready():
    serving = getattr(app.state, "serving", None)
    if serving is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "service": "not_ready",
                "model": None,
                "policy_path": None,
            },
        )
    return {
        "status": "ok",
        "service": "ready",
        "model": serving.policy.selected_model,
        "policy_path": str(serving.policy_path),
    }


@app.get("/metrics")
def metrics() -> Response:
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)


@app.get("/dashboard", include_in_schema=False)
def dashboard_redirect() -> RedirectResponse:
    url = os.getenv("DASHBOARD_URL", "http://localhost:5173")
    return RedirectResponse(url=url, status_code=307)


@app.post(
    "/predict_proba",
    response_model=PredictProbaResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def predict_proba(payload: RecordsPayload) -> PredictProbaResponse:
    serving = _get_serving_state()
    try:
        proba, schema_report, _ = exec_predict_proba(payload, serving, "predict_proba")
        return PredictProbaResponse(
            n=int(len(proba)), proba=proba, schema_report=schema_report
        )
    except ValueError as e:
        _serving = getattr(app.state, "serving", None)
        _model = str(
            getattr(getattr(_serving, "policy", None), "selected_model", "") or ""
        )
        INFERENCE_ERRORS.labels(endpoint="predict_proba", model=_model).inc()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _serving = getattr(app.state, "serving", None)
        _model = str(
            getattr(getattr(_serving, "policy", None), "selected_model", "") or ""
        )
        INFERENCE_ERRORS.labels(endpoint="predict_proba", model=_model).inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/decide",
    response_model=DecideResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def decide(payload: RecordsPayload) -> DecideResponse:
    serving = _get_serving_state()
    try:
        actions_df, pred_report, _ = exec_decide(payload, serving, "decide")
        return DecideResponse(
            n=int(len(actions_df)),
            results=actions_df.to_dict(orient="records"),
            report=pred_report,
        )
    except ValueError as e:
        _serving = getattr(app.state, "serving", None)
        _model = str(
            getattr(getattr(_serving, "policy", None), "selected_model_artifact", "")
            or ""
        )
        INFERENCE_ERRORS.labels(endpoint="decide", model=_model).inc()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _serving = getattr(app.state, "serving", None)
        _model = str(
            getattr(getattr(_serving, "policy", None), "selected_model_artifact", "")
            or ""
        )
        INFERENCE_ERRORS.labels(endpoint="decide", model=_model).inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/reload",
    response_model=ReloadResponse,
    responses={403: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def reload_serving_state(request: Request) -> ReloadResponse:
    require_admin_key(
        request,
        detail="Bu endpoint icin x-admin-key header'i gereklidir.",
    )
    try:
        serving = await reload_serving_state_for_app(app, loader=_load_serving_state)
        return {
            "status": "ok",
            "message": "Serving state reloaded",
            "model": serving.policy.selected_model,
            "policy_path": str(serving.policy_path),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    rid = getattr(request.state, "request_id", None)
    detail = str(exc.detail)
    return _error_response(
        status_code=exc.status_code,
        error_code="http_error",
        message=detail,
        request_id=rid,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", None)
    logger.exception(f"Unhandled exception: {exc}", extra={"request_id": rid})
    return _error_response(
        status_code=500,
        error_code="internal_error",
        message="Internal server error",
        request_id=rid,
    )
