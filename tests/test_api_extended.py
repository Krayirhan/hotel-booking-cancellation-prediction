from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Response
from starlette.requests import Request

import src.api as api


def _make_request(
    *,
    path: str = "/predict_proba",
    method: str = "POST",
    headers: dict[str, str] | None = None,
    client: tuple[str, int] | None = ("127.0.0.1", 12345),
) -> Request:
    hdrs = [
        (k.lower().encode("utf-8"), str(v).encode("utf-8"))
        for k, v in (headers or {}).items()
    ]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "headers": hdrs,
        "client": client,
        "server": ("testserver", 80),
    }

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, _receive)


def _json_body(resp: Response) -> dict:
    return json.loads(resp.body.decode("utf-8"))


class _Limiter:
    def __init__(self, decisions: list[bool] | None = None):
        self.decisions = list(decisions or [])
        self.calls: list[tuple[str, int]] = []

    def allow(self, client: str, limit: int) -> bool:
        self.calls.append((client, limit))
        if self.decisions:
            return self.decisions.pop(0)
        return True


async def _ok_next(_request: Request):
    return Response(content="ok", media_type="text/plain")


def _dummy_serving():
    return SimpleNamespace(
        policy=SimpleNamespace(
            selected_model="xgb",
            selected_model_artifact="models/xgb.joblib",
        ),
        policy_path=Path("reports/decision_policy.json"),
    )


def test_get_serving_state_load_and_503(monkeypatch):
    api.app.state.serving = None
    loaded = _dummy_serving()
    monkeypatch.setattr(api, "_load_serving_state", lambda: loaded)
    assert api._get_serving_state() is loaded
    assert api.app.state.serving is loaded

    api.app.state.serving = None
    monkeypatch.setattr(
        api,
        "_load_serving_state",
        lambda: (_ for _ in ()).throw(RuntimeError("no model")),
    )
    with pytest.raises(HTTPException) as ex:
        api._get_serving_state()
    assert ex.value.status_code == 503


def test_middleware_options_short_circuit(monkeypatch):
    req = _make_request(path="/predict_proba", method="OPTIONS")
    api.app.state.rate_limiter = _Limiter([True])
    api.app.state.shutting_down = False
    monkeypatch.setattr(api, "_api_key_required", lambda: True)
    resp = asyncio.run(api.request_context_middleware(req, _ok_next))
    assert resp.status_code == 200


def test_middleware_api_key_not_configured(monkeypatch):
    req = _make_request(path="/predict_proba", headers={})
    api.app.state.rate_limiter = _Limiter([True])
    api.app.state.shutting_down = False
    monkeypatch.setattr(api, "_api_key_required", lambda: True)
    monkeypatch.setattr(api, "_expected_api_key", lambda: None)
    resp = asyncio.run(api.request_context_middleware(req, _ok_next))
    assert resp.status_code == 503
    assert _json_body(resp)["error_code"] == "api_key_not_configured"


def test_middleware_unauthorized_api_key(monkeypatch):
    req = _make_request(path="/predict_proba", headers={"x-api-key": "wrong"})
    api.app.state.rate_limiter = _Limiter([True])
    api.app.state.shutting_down = False
    monkeypatch.setattr(api, "_api_key_required", lambda: True)
    monkeypatch.setattr(api, "_expected_api_key", lambda: "correct")
    resp = asyncio.run(api.request_context_middleware(req, _ok_next))
    assert resp.status_code == 401
    assert _json_body(resp)["error_code"] == "unauthorized"


def test_middleware_builds_limiter_when_missing_and_rate_limits(monkeypatch):
    limiter = _Limiter([False])
    req = _make_request(path="/predict_proba")
    api.app.state.rate_limiter = None
    api.app.state.shutting_down = False
    monkeypatch.setattr(api, "_api_key_required", lambda: False)
    monkeypatch.setattr(api, "_build_runtime_rate_limiter", lambda: limiter)
    resp = asyncio.run(api.request_context_middleware(req, _ok_next))
    assert resp.status_code == 429
    assert _json_body(resp)["error_code"] == "rate_limit_exceeded"
    assert api.app.state.rate_limiter is limiter


def test_middleware_login_bruteforce_limit(monkeypatch):
    req = _make_request(path="/auth/login", method="POST")
    api.app.state.rate_limiter = _Limiter([False])
    api.app.state.shutting_down = False
    monkeypatch.setattr(api, "_api_key_required", lambda: False)
    resp = asyncio.run(api.request_context_middleware(req, _ok_next))
    assert resp.status_code == 429
    assert _json_body(resp)["error_code"] == "login_rate_limit"


def test_middleware_content_length_guards(monkeypatch):
    api.app.state.rate_limiter = _Limiter([True, True, True])
    api.app.state.shutting_down = False
    monkeypatch.setattr(api, "_api_key_required", lambda: False)

    too_large = _make_request(
        path="/predict_proba",
        headers={"content-length": str(9 * 1024 * 1024)},
    )
    resp1 = asyncio.run(api.request_context_middleware(too_large, _ok_next))
    assert resp1.status_code == 413
    assert _json_body(resp1)["error_code"] == "payload_too_large"

    invalid = _make_request(path="/predict_proba", headers={"content-length": "abc"})
    resp2 = asyncio.run(api.request_context_middleware(invalid, _ok_next))
    assert resp2.status_code == 400
    assert _json_body(resp2)["error_code"] == "invalid_content_length"


def test_middleware_shutting_down_and_success_headers(monkeypatch):
    monkeypatch.setattr(api, "_api_key_required", lambda: False)

    api.app.state.rate_limiter = _Limiter([True])
    api.app.state.shutting_down = True
    req1 = _make_request(path="/predict_proba")
    resp1 = asyncio.run(api.request_context_middleware(req1, _ok_next))
    assert resp1.status_code == 503
    assert _json_body(resp1)["error_code"] == "service_shutting_down"

    api.app.state.rate_limiter = _Limiter([True])
    api.app.state.shutting_down = False
    req2 = _make_request(path="/predict_proba")
    resp2 = asyncio.run(api.request_context_middleware(req2, _ok_next))
    assert resp2.status_code == 200
    assert "x-request-id" in resp2.headers
    assert resp2.headers["X-Content-Type-Options"] == "nosniff"
    assert resp2.headers["X-Frame-Options"] == "DENY"
    assert "Strict-Transport-Security" in resp2.headers


def test_ready_degraded_and_dashboard_redirect(monkeypatch):
    api.app.state.serving = None
    degraded = api.ready()
    assert degraded.status_code == 503

    monkeypatch.setenv("DASHBOARD_URL", "https://dashboard.example.com")
    redirect = api.dashboard_redirect()
    assert redirect.status_code == 307
    assert redirect.headers["location"] == "https://dashboard.example.com"


def test_predict_proba_and_decide_error_paths(monkeypatch):
    payload = api.RecordsPayload(records=[{"lead_time": 10}])
    monkeypatch.setattr(api, "_get_serving_state", lambda: _dummy_serving())

    monkeypatch.setattr(
        api,
        "exec_predict_proba",
        lambda *_: (_ for _ in ()).throw(ValueError("bad request")),
    )
    with pytest.raises(HTTPException) as ex1:
        api.predict_proba(payload)
    assert ex1.value.status_code == 400

    monkeypatch.setattr(
        api,
        "exec_predict_proba",
        lambda *_: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(HTTPException) as ex2:
        api.predict_proba(payload)
    assert ex2.value.status_code == 500

    monkeypatch.setattr(
        api,
        "exec_decide",
        lambda *_: (_ for _ in ()).throw(ValueError("bad request")),
    )
    with pytest.raises(HTTPException) as ex3:
        api.decide(payload)
    assert ex3.value.status_code == 400

    monkeypatch.setattr(
        api,
        "exec_decide",
        lambda *_: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(HTTPException) as ex4:
        api.decide(payload)
    assert ex4.value.status_code == 500


def test_reload_serving_state_admin_and_failure_paths(monkeypatch):
    api.app.state._reload_lock = asyncio.Lock()
    monkeypatch.setenv("DS_ADMIN_KEY", "admin-secret")

    req1 = _make_request(path="/reload", headers={})
    with pytest.raises(HTTPException) as ex1:
        asyncio.run(api.reload_serving_state(req1))
    assert ex1.value.status_code == 403

    req2 = _make_request(path="/reload", headers={"x-admin-key": "admin-secret"})
    monkeypatch.setattr(api, "_load_serving_state", lambda: _dummy_serving())
    out = asyncio.run(api.reload_serving_state(req2))
    assert out["status"] == "ok"
    assert out["model"] == "xgb"

    monkeypatch.setattr(
        api,
        "_load_serving_state",
        lambda: (_ for _ in ()).throw(RuntimeError("load failed")),
    )
    with pytest.raises(HTTPException) as ex2:
        asyncio.run(api.reload_serving_state(req2))
    assert ex2.value.status_code == 500
    assert "Reload failed" in str(ex2.value.detail)


def test_validate_admin_key_startup_enforces_non_dev(monkeypatch):
    monkeypatch.setenv("DS_ENV", "production")
    monkeypatch.delenv("DS_ADMIN_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DS_ADMIN_KEY must be set"):
        api._validate_admin_key_startup()


def test_validate_admin_key_startup_allows_dev_without_key(monkeypatch):
    monkeypatch.setenv("DS_ENV", "development")
    monkeypatch.delenv("DS_ADMIN_KEY", raising=False)
    api._validate_admin_key_startup()


def test_exception_handlers_return_structured_error():
    req = _make_request(path="/x")
    req.state.request_id = "rid-123"

    resp1 = asyncio.run(
        api.http_exception_handler(req, HTTPException(status_code=418, detail="teapot"))
    )
    assert resp1.status_code == 418
    body1 = _json_body(resp1)
    assert body1["error_code"] == "http_error"
    assert body1["request_id"] == "rid-123"

    resp2 = asyncio.run(api.unhandled_exception_handler(req, RuntimeError("crash")))
    assert resp2.status_code == 500
    body2 = _json_body(resp2)
    assert body2["error_code"] == "internal_error"
    assert body2["request_id"] == "rid-123"


def test_periodic_chat_cleanup_runs_and_handles_errors(monkeypatch):
    class _Store:
        def __init__(self):
            self.cleaned = 0

        def _cleanup_expired(self):
            self.cleaned += 1

    store = _Store()
    import src.chat.memory as chat_memory

    calls = {"n": 0}

    async def _sleep_once_then_cancel(_seconds):
        calls["n"] += 1
        if calls["n"] == 1:
            return None
        raise asyncio.CancelledError()

    monkeypatch.setattr(api.asyncio, "sleep", _sleep_once_then_cancel)
    monkeypatch.setattr(chat_memory, "get_session_store", lambda: store)
    asyncio.run(api._periodic_chat_cleanup(interval_seconds=0))
    assert store.cleaned == 1

    calls2 = {"n": 0}

    async def _sleep_for_error_case(_seconds):
        calls2["n"] += 1
        if calls2["n"] == 1:
            return None
        raise asyncio.CancelledError()

    monkeypatch.setattr(api.asyncio, "sleep", _sleep_for_error_case)
    monkeypatch.setattr(
        chat_memory,
        "get_session_store",
        lambda: (_ for _ in ()).throw(RuntimeError("store down")),
    )
    asyncio.run(api._periodic_chat_cleanup(interval_seconds=0))


def test_lifespan_degraded_and_shutdown_paths(monkeypatch):
    fake_app = SimpleNamespace(state=SimpleNamespace())

    monkeypatch.setenv("WEB_CONCURRENCY", "2")
    monkeypatch.setenv("RATE_LIMIT_BACKEND", "memory")
    monkeypatch.setattr(api, "init_tracing", lambda **kwargs: None)
    monkeypatch.setattr(api, "instrument_fastapi", lambda *_: None)
    monkeypatch.setattr(api, "run_migrations", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(api, "ensure_required_tables", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(api, "init_user_store", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(api, "seed_admin", lambda: None)
    monkeypatch.setattr(api, "init_guest_store", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(api, "init_dashboard_store", lambda: None)
    monkeypatch.setattr(
        api,
        "_load_serving_state",
        lambda: (_ for _ in ()).throw(RuntimeError("model load fail")),
    )
    monkeypatch.setattr(api, "_build_runtime_rate_limiter", lambda: _Limiter([True]))

    async def _cleanup_forever(*args, **kwargs):
        await asyncio.sleep(3600)

    monkeypatch.setattr(api, "_periodic_chat_cleanup", _cleanup_forever)

    import src.chat.ollama_client as ollama_client

    class _BadClient:
        async def aclose(self):
            raise RuntimeError("close fail")

    monkeypatch.setattr(ollama_client, "get_ollama_client", lambda: _BadClient())

    async def _run():
        async with api.lifespan(fake_app):
            assert fake_app.state.serving is None
            assert fake_app.state.shutting_down is False
            assert isinstance(fake_app.state._reload_lock, asyncio.Lock)

    asyncio.run(_run())
    assert fake_app.state.shutting_down is True
