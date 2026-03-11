"""dashboard_auth.py — PostgreSQL-backed authentication (single admin user).

Flow:
  POST /auth/login   → verify against `users` table → issue bearer token (Redis / in-memory)
  GET  /auth/me      → validate token, return username
  POST /auth/logout  → revoke token
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import bcrypt
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

from .user_store import get_user_store
from .utils import get_logger

logger = get_logger("dashboard_auth")

router_dashboard_auth = APIRouter(prefix="/auth", tags=["dashboard-auth"])

# ── In-memory token store (fallback when Redis unavailable) ──────────────────
_token_lock = threading.Lock()
_token_store: Dict[str, Dict[str, Any]] = {}
_MAX_TOKENS_PER_USER = 5
_MAX_TOTAL_TOKENS = 10_000

# ── Redis token backend ───────────────────────────────────────────────────────
_REDIS_TOKEN_PREFIX = "ds:auth:tok:"  # nosec B105
_REDIS_USER_PREFIX = "ds:auth:usr:"
_REDIS_DEVICE_PREFIX = "ds:auth:dev:"
_redis_client: Any = None

# ── Brute-force guard (progressive backoff + lockout) ───────────────────────
_login_guard_lock = threading.Lock()
_login_failures_by_user: Dict[str, list[float]] = {}
_login_failures_by_ip: Dict[str, list[float]] = {}
_login_backoff_until_user: Dict[str, float] = {}
_login_backoff_until_ip: Dict[str, float] = {}
_login_lockout_until_user: Dict[str, float] = {}
_login_lockout_until_ip: Dict[str, float] = {}


def _get_redis_client() -> Any | None:
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        return None
    try:
        import redis as _redis  # type: ignore[import]

        client = _redis.Redis.from_url(
            redis_url, decode_responses=True, socket_timeout=2
        )
        client.ping()
        _redis_client = client
        logger.info("Auth token store: Redis backend active (%s)", redis_url)
        return _redis_client
    except Exception as exc:
        logger.warning(
            "Auth token store: Redis unavailable, using in-memory fallback. reason=%s",
            exc,
        )
        return None


def _redis_device_key(username: str, device_id: str) -> str:
    safe_device = hashlib.sha1(device_id.encode("utf-8")).hexdigest()
    return f"{_REDIS_DEVICE_PREFIX}{username}:{safe_device}"


def _redis_add_token(
    r: Any,
    token: str,
    username: str,
    expires: datetime,
    *,
    issued_at: datetime | None = None,
    device_id: str = "unknown-device",
    session_id: str | None = None,
    rotated_from: str | None = None,
) -> None:
    if issued_at is None:
        issued_at = datetime.now(timezone.utc)
    if not session_id:
        session_id = secrets.token_urlsafe(12)

    ttl = max(1, int((expires - datetime.now(timezone.utc)).total_seconds()))
    payload = {
        "username": username,
        "expires_at": _iso(expires),
        "issued_at": _iso(issued_at),
        "device_id": device_id,
        "session_id": session_id,
    }
    if rotated_from:
        payload["rotated_from"] = rotated_from

    r.setex(
        f"{_REDIS_TOKEN_PREFIX}{token}",
        ttl,
        json.dumps(payload),
    )
    user_key = f"{_REDIS_USER_PREFIX}{username}"
    r.zadd(user_key, {token: expires.timestamp()})
    r.zadd(_redis_device_key(username, device_id), {token: expires.timestamp()})
    r.expire(user_key, ttl + 60)
    r.expire(_redis_device_key(username, device_id), ttl + 60)


def _redis_get_token(r: Any, token: str) -> Dict[str, Any] | None:
    raw = r.get(f"{_REDIS_TOKEN_PREFIX}{token}")
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _redis_remove_token(
    r: Any,
    token: str,
    username: str | None = None,
    device_id: str | None = None,
) -> None:
    r.delete(f"{_REDIS_TOKEN_PREFIX}{token}")
    if username:
        r.zrem(f"{_REDIS_USER_PREFIX}{username}", token)
    if username and device_id:
        r.zrem(_redis_device_key(username, device_id), token)


def _redis_enforce_user_limit(r: Any, username: str) -> None:
    user_key = f"{_REDIS_USER_PREFIX}{username}"
    r.zremrangebyscore(user_key, 0, datetime.now(timezone.utc).timestamp())
    count = r.zcard(user_key)
    if count >= _MAX_TOKENS_PER_USER:
        to_remove = r.zrange(user_key, 0, count - _MAX_TOKENS_PER_USER)
        for old_tok in to_remove:
            data = _redis_get_token(r, old_tok) or {}
            _redis_remove_token(
                r,
                old_tok,
                username=username,
                device_id=data.get("device_id"),
            )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _auth_enabled() -> bool:
    return os.getenv("DASHBOARD_AUTH_ENABLED", "true").strip().lower() not in {
        "false",
        "0",
        "no",
    }


def _token_ttl_minutes() -> int:
    try:
        return max(int(os.getenv("DASHBOARD_TOKEN_TTL_MINUTES", "480")), 5)
    except Exception:
        return 480


def _parse_int_env(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        return max(int(os.getenv(name, str(default))), minimum)
    except Exception:
        return default


def _login_window_seconds() -> int:
    return _parse_int_env("DASHBOARD_LOGIN_WINDOW_SECONDS", 900, minimum=60)


def _login_backoff_start_after() -> int:
    return _parse_int_env("DASHBOARD_LOGIN_BACKOFF_START_AFTER", 3, minimum=1)


def _login_base_backoff_seconds() -> int:
    return _parse_int_env("DASHBOARD_LOGIN_BASE_BACKOFF_SECONDS", 2, minimum=1)


def _login_max_backoff_seconds() -> int:
    return _parse_int_env("DASHBOARD_LOGIN_MAX_BACKOFF_SECONDS", 120, minimum=1)


def _login_lockout_after_failures() -> int:
    return _parse_int_env("DASHBOARD_LOGIN_LOCKOUT_AFTER", 8, minimum=2)


def _login_lockout_seconds() -> int:
    return _parse_int_env("DASHBOARD_LOGIN_LOCKOUT_SECONDS", 900, minimum=30)


def _is_non_prod_env() -> bool:
    env = os.getenv("DS_ENV", "development").strip().lower()
    return env in {"dev", "development", "local", "test", "testing"}


def _allow_insecure_dev_login() -> bool:
    return os.getenv("DASHBOARD_ALLOW_INSECURE_DEV_LOGIN", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _get_users() -> Dict[str, str]:
    """Legacy env-based users for compatibility and unit-test stability."""
    users: Dict[str, str] = {}

    admin_password = os.getenv("DASHBOARD_ADMIN_PASSWORD_ADMIN", "").strip()
    if not admin_password:
        if _is_non_prod_env() and _allow_insecure_dev_login():
            admin_password = "admin123"
            logger.warning(
                "Using insecure development admin password because "
                "DASHBOARD_ALLOW_INSECURE_DEV_LOGIN=true."
            )
        else:
            raise RuntimeError(
                "DASHBOARD_ADMIN_PASSWORD_ADMIN must be set "
                "(or enable DASHBOARD_ALLOW_INSECURE_DEV_LOGIN=true in development)."
            )

    if admin_password == "replace-me":
        if _is_non_prod_env():
            logger.warning(
                "DASHBOARD_ADMIN_PASSWORD_ADMIN uses placeholder value 'replace-me'."
            )
        else:
            raise RuntimeError(
                "DASHBOARD_ADMIN_PASSWORD_ADMIN cannot be 'replace-me' in non-dev environments."
            )

    users["admin"] = admin_password

    legacy_user = os.getenv("DASHBOARD_ADMIN_USERNAME", "").strip()
    legacy_pass = os.getenv("DASHBOARD_ADMIN_PASSWORD", "").strip()
    if legacy_user and legacy_pass:
        users[legacy_user] = legacy_pass

    raw_extra = os.getenv("DASHBOARD_EXTRA_USERS", "").strip()
    if raw_extra:
        try:
            parsed = json.loads(raw_extra)
            if isinstance(parsed, dict):
                for username, passwd in parsed.items():
                    if isinstance(username, str) and isinstance(passwd, str):
                        users[username] = passwd
        except Exception:
            logger.warning("Invalid DASHBOARD_EXTRA_USERS JSON ignored.")

    return users


def _verify_credentials(username: str, password: str) -> bool:
    try:
        store = get_user_store()
    except RuntimeError:
        store = None
    if store is not None:
        try:
            if store.verify_password(username, password):
                return True
        except Exception:
            if not _is_non_prod_env():
                return False
        # In non-prod/test contexts allow env-based fallback for compatibility.
        if not _is_non_prod_env():
            return False

    # Compatibility fallback for isolated/unit tests where DB store is not initialized.
    try:
        users = _get_users()
    except RuntimeError:
        return False
    expected = users.get(username)
    if not expected:
        return False
    if expected.startswith("$2"):
        try:
            return bcrypt.checkpw(password.encode("utf-8"), expected.encode("utf-8"))
        except Exception:
            return False
    return secrets.compare_digest(password, expected)


def _cleanup_expired_tokens() -> None:
    now = datetime.now(timezone.utc)
    with _token_lock:
        expired = [k for k, v in _token_store.items() if v.get("expires_at") <= now]
        for key in expired:
            _token_store.pop(key, None)


def _cleanup_expired_tokens_locked() -> None:
    now = datetime.now(timezone.utc)
    expired = [k for k, v in _token_store.items() if v.get("expires_at") <= now]
    for key in expired:
        _token_store.pop(key, None)


def _parse_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        return None
    token = authorization[len(prefix) :].strip()
    return token or None


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed
        except Exception:
            return None
    return None


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _audit_event(event: str, **fields: Any) -> None:
    payload = {"event": event, "at": datetime.now(timezone.utc).isoformat(), **fields}
    logger.info("auth_audit %s", json.dumps(payload, ensure_ascii=True, sort_keys=True))


def _client_ip(request: Request | None) -> str:
    if request is None:
        return "unknown"
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _derive_device_id(request: Request | None) -> str:
    if request is None:
        return "unknown-device"
    raw = (request.headers.get("x-device-id") or "").strip()
    if raw:
        return raw[:120]
    ua = request.headers.get("user-agent") or "unknown-agent"
    return "ua-" + hashlib.sha1(ua.encode("utf-8")).hexdigest()[:16]


def _now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def _prune_failure_window(records: list[float], now_ts: float) -> list[float]:
    min_ts = now_ts - float(_login_window_seconds())
    return [t for t in records if t >= min_ts]


def _remaining_seconds(until_ts: float, now_ts: float) -> int:
    if until_ts <= now_ts:
        return 0
    return int(until_ts - now_ts + 0.999)


def _scope_penalty_remaining(
    key: str,
    now_ts: float,
    *,
    backoff_map: Dict[str, float],
    lockout_map: Dict[str, float],
) -> tuple[int, str]:
    lock_until = lockout_map.get(key, 0.0)
    lock_left = _remaining_seconds(lock_until, now_ts)
    if lock_left > 0:
        return lock_left, "lockout"

    backoff_until = backoff_map.get(key, 0.0)
    backoff_left = _remaining_seconds(backoff_until, now_ts)
    if backoff_left > 0:
        return backoff_left, "backoff"

    lockout_map.pop(key, None)
    backoff_map.pop(key, None)
    return 0, "none"


def check_login_attempt_allowed(
    *,
    username: str | None,
    client_ip: str,
) -> tuple[bool, int, str]:
    """Return whether a login attempt is allowed right now."""
    now_ts = _now_ts()
    user_key = (username or "").strip().lower()

    with _login_guard_lock:
        ip_retry, ip_reason = _scope_penalty_remaining(
            client_ip,
            now_ts,
            backoff_map=_login_backoff_until_ip,
            lockout_map=_login_lockout_until_ip,
        )
        user_retry, user_reason = (0, "none")
        if user_key:
            user_retry, user_reason = _scope_penalty_remaining(
                user_key,
                now_ts,
                backoff_map=_login_backoff_until_user,
                lockout_map=_login_lockout_until_user,
            )

    retry_after = max(ip_retry, user_retry)
    if retry_after <= 0:
        return True, 0, "ok"

    reason = "lockout" if "lockout" in {ip_reason, user_reason} else "backoff"
    return False, retry_after, reason


def _apply_scope_failure(
    key: str,
    now_ts: float,
    failures_map: Dict[str, list[float]],
    backoff_map: Dict[str, float],
    lockout_map: Dict[str, float],
) -> tuple[int, int, str]:
    records = _prune_failure_window(failures_map.get(key, []), now_ts)
    records.append(now_ts)
    failures_map[key] = records
    fail_count = len(records)

    if fail_count >= _login_lockout_after_failures():
        until = now_ts + float(_login_lockout_seconds())
        lockout_map[key] = until
        backoff_map.pop(key, None)
        return fail_count, _remaining_seconds(until, now_ts), "lockout"

    if fail_count >= _login_backoff_start_after():
        exponent = fail_count - _login_backoff_start_after()
        delay = min(
            _login_max_backoff_seconds(),
            _login_base_backoff_seconds() * (2**exponent),
        )
        until = now_ts + float(delay)
        backoff_map[key] = max(backoff_map.get(key, 0.0), until)
        return fail_count, int(delay), "backoff"

    return fail_count, 0, "none"


def record_login_attempt(
    *,
    username: str,
    client_ip: str,
    success: bool,
    reason: str | None = None,
) -> None:
    """Update progressive backoff/lockout state for login attempts."""
    user_key = username.strip().lower()
    now_ts = _now_ts()

    with _login_guard_lock:
        if success:
            _login_failures_by_user.pop(user_key, None)
            _login_backoff_until_user.pop(user_key, None)
            _login_lockout_until_user.pop(user_key, None)
            _audit_event(
                "login_success",
                username=username,
                client_ip=client_ip,
                reason=reason or "ok",
            )
            return

        user_count, user_wait, user_state = _apply_scope_failure(
            user_key,
            now_ts,
            _login_failures_by_user,
            _login_backoff_until_user,
            _login_lockout_until_user,
        )
        ip_count, ip_wait, ip_state = _apply_scope_failure(
            client_ip,
            now_ts,
            _login_failures_by_ip,
            _login_backoff_until_ip,
            _login_lockout_until_ip,
        )
        retry_after = max(user_wait, ip_wait)
        state = (
            "lockout"
            if "lockout" in {user_state, ip_state}
            else ("backoff" if "backoff" in {user_state, ip_state} else "none")
        )

    _audit_event(
        "login_failure",
        username=username,
        client_ip=client_ip,
        reason=reason or "invalid_credentials",
        user_failures=user_count,
        ip_failures=ip_count,
        penalty_state=state,
        retry_after_seconds=retry_after,
    )


def _clear_ip_failure_if_needed(client_ip: str) -> None:
    with _login_guard_lock:
        failures = _prune_failure_window(
            _login_failures_by_ip.get(client_ip, []), _now_ts()
        )
        if not failures:
            _login_failures_by_ip.pop(client_ip, None)
            _login_backoff_until_ip.pop(client_ip, None)
            _login_lockout_until_ip.pop(client_ip, None)


def _revoke_inmemory_tokens(
    *,
    username: str,
    device_id: str | None = None,
    exclude_token: str | None = None,
) -> int:
    removed = 0
    with _token_lock:
        for tok, data in list(_token_store.items()):
            if data.get("username") != username:
                continue
            if exclude_token and tok == exclude_token:
                continue
            if device_id and data.get("device_id") != device_id:
                continue
            _token_store.pop(tok, None)
            removed += 1
    return removed


def _redis_user_tokens(r: Any, username: str) -> list[str]:
    return list(r.zrange(f"{_REDIS_USER_PREFIX}{username}", 0, -1))


def _revoke_redis_tokens(
    *,
    r: Any,
    username: str,
    device_id: str | None = None,
    exclude_token: str | None = None,
) -> int:
    removed = 0
    for tok in _redis_user_tokens(r, username):
        if exclude_token and tok == exclude_token:
            continue
        data = _redis_get_token(r, tok) or {}
        if device_id and data.get("device_id") != device_id:
            continue
        _redis_remove_token(r, tok, username=username, device_id=data.get("device_id"))
        removed += 1
    return removed


def revoke_user_tokens(
    *,
    username: str,
    device_id: str | None = None,
    exclude_token: str | None = None,
) -> int:
    r = _get_redis_client()
    if r is not None:
        return _revoke_redis_tokens(
            r=r,
            username=username,
            device_id=device_id,
            exclude_token=exclude_token,
        )
    return _revoke_inmemory_tokens(
        username=username,
        device_id=device_id,
        exclude_token=exclude_token,
    )


# ── Pydantic schemas ──────────────────────────────────────────────────────────


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: str
    username: str


class RevokeDeviceRequest(BaseModel):
    device_id: str


class ForceLogoutRequest(BaseModel):
    username: str
    device_id: str | None = None


# ── Dependency ────────────────────────────────────────────────────────────────


def _token_data(token: str) -> Dict[str, Any] | None:
    r = _get_redis_client()
    if r is not None:
        return _redis_get_token(r, token)
    _cleanup_expired_tokens()
    with _token_lock:
        return _token_store.get(token)


def _admin_key_valid(value: str | None) -> bool:
    expected = (os.getenv("DS_ADMIN_KEY") or "").strip()
    if not expected:
        return False
    candidate = value or ""
    return hmac.compare_digest(candidate, expected)


def require_dashboard_user(
    authorization: str | None = Header(default=None),
) -> Dict[str, Any]:
    """Validate bearer token and return user dict."""
    if not _auth_enabled():
        return {"username": "anonymous", "auth_enabled": False}

    token = _parse_bearer_token(authorization)
    if token is None:
        raise HTTPException(status_code=401, detail="Oturum gerekli")

    data = _token_data(token)

    if data is None:
        raise HTTPException(
            status_code=401, detail="Geçersiz veya süresi dolmuş oturum"
        )

    expires_at = _parse_datetime(data.get("expires_at"))
    if expires_at is not None and expires_at <= datetime.now(timezone.utc):
        username = str(data.get("username") or "")
        if username:
            r = _get_redis_client()
            if r is not None:
                _redis_remove_token(
                    r,
                    token,
                    username=username,
                    device_id=data.get("device_id"),
                )
            else:
                with _token_lock:
                    _token_store.pop(token, None)
        raise HTTPException(
            status_code=401, detail="Geçersiz veya süresi dolmuş oturum"
        )

    return {
        "username": data.get("username", "admin"),
        "auth_enabled": True,
        "token": token,
        "device_id": data.get("device_id", "unknown-device"),
        "session_id": data.get("session_id"),
        "issued_at": data.get("issued_at"),
        "expires_at": data.get("expires_at"),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router_dashboard_auth.post("/login", response_model=LoginResponse)
def dashboard_login(
    payload: LoginRequest,
    request: Request,
) -> LoginResponse:
    client_ip = _client_ip(request)
    device_id = _derive_device_id(request)

    if not _auth_enabled():
        expires = datetime.now(timezone.utc) + timedelta(minutes=_token_ttl_minutes())
        return LoginResponse(
            access_token="auth-disabled",  # nosec B106
            expires_at=expires.isoformat(),
            username="anonymous",
        )

    allowed, retry_after, penalty = check_login_attempt_allowed(
        username=payload.username,
        client_ip=client_ip,
    )
    if not allowed:
        _audit_event(
            "login_blocked",
            username=payload.username,
            client_ip=client_ip,
            device_id=device_id,
            penalty=penalty,
            retry_after_seconds=retry_after,
        )
        raise HTTPException(
            status_code=429,
            detail=(
                "Çok fazla başarısız giriş denemesi. "
                f"Lütfen {retry_after} saniye sonra tekrar deneyin."
            ),
        )

    if not _verify_credentials(payload.username, payload.password):
        record_login_attempt(
            username=payload.username,
            client_ip=client_ip,
            success=False,
            reason="invalid_credentials",
        )
        raise HTTPException(status_code=401, detail="Geçersiz kullanıcı adı veya şifre")

    record_login_attempt(
        username=payload.username,
        client_ip=client_ip,
        success=True,
        reason="credentials_verified",
    )
    _clear_ip_failure_if_needed(client_ip)

    token = secrets.token_urlsafe(32)
    session_id = secrets.token_urlsafe(12)
    issued_at = datetime.now(timezone.utc)
    expires = datetime.now(timezone.utc) + timedelta(minutes=_token_ttl_minutes())
    token_data: Dict[str, Any] = {
        "username": payload.username,
        "expires_at": expires,
        "issued_at": issued_at,
        "device_id": device_id,
        "session_id": session_id,
    }

    r = _get_redis_client()
    if r is not None:
        _redis_enforce_user_limit(r, payload.username)
        _redis_add_token(
            r,
            token,
            username=payload.username,
            expires=expires,
            issued_at=issued_at,
            device_id=device_id,
            session_id=session_id,
        )
    else:
        with _token_lock:
            if len(_token_store) >= _MAX_TOTAL_TOKENS:
                _cleanup_expired_tokens_locked()
                if len(_token_store) >= _MAX_TOTAL_TOKENS:
                    raise HTTPException(
                        status_code=503, detail="Token store kapasitesi dolu."
                    )
            user_tokens = [
                k
                for k, v in _token_store.items()
                if v.get("username") == payload.username
            ]
            if len(user_tokens) >= _MAX_TOKENS_PER_USER:
                oldest = min(user_tokens, key=lambda k: _token_store[k]["expires_at"])
                _token_store.pop(oldest, None)
            _token_store[token] = token_data

    _audit_event(
        "token_issued",
        username=payload.username,
        client_ip=client_ip,
        device_id=device_id,
        session_id=session_id,
    )
    return LoginResponse(
        access_token=token,
        expires_at=expires.isoformat(),
        username=payload.username,
    )


@router_dashboard_auth.get("/me")
def dashboard_me(user: Dict[str, Any] = Depends(require_dashboard_user)):
    return {
        "status": "ok",
        "username": user.get("username"),
        "auth_enabled": user.get("auth_enabled", True),
    }


@router_dashboard_auth.post("/logout")
def dashboard_logout(
    authorization: str | None = Header(default=None),
    _user: Dict[str, Any] = Depends(require_dashboard_user),
):
    token = _parse_bearer_token(authorization)
    if token is not None:
        device_id = _user.get("device_id")
        r = _get_redis_client()
        if r is not None:
            _redis_remove_token(
                r,
                token,
                _user.get("username"),
                device_id=device_id,
            )
        else:
            with _token_lock:
                _token_store.pop(token, None)
        _audit_event(
            "logout",
            username=_user.get("username"),
            device_id=device_id,
            session_id=_user.get("session_id"),
        )
    return {"status": "ok", "message": "Oturum kapatıldı"}


@router_dashboard_auth.post("/refresh", response_model=LoginResponse)
def dashboard_refresh(
    request: Request,
    authorization: str | None = Header(default=None),
    user: Dict[str, Any] = Depends(require_dashboard_user),
) -> LoginResponse:
    token = _parse_bearer_token(authorization)
    if token is None:
        raise HTTPException(status_code=401, detail="Oturum gerekli")

    current = _token_data(token)
    if current is None:
        raise HTTPException(
            status_code=401, detail="Geçersiz veya süresi dolmuş oturum"
        )

    username = str(user.get("username") or current.get("username") or "admin")
    old_device = str(
        current.get("device_id") or user.get("device_id") or "unknown-device"
    )
    device_id = _derive_device_id(request) if request is not None else old_device
    if not device_id or device_id == "unknown-device":
        device_id = old_device

    new_token = secrets.token_urlsafe(32)
    session_id = secrets.token_urlsafe(12)
    issued_at = datetime.now(timezone.utc)
    expires = issued_at + timedelta(minutes=_token_ttl_minutes())

    r = _get_redis_client()
    if r is not None:
        _redis_enforce_user_limit(r, username)
        _redis_add_token(
            r,
            new_token,
            username=username,
            expires=expires,
            issued_at=issued_at,
            device_id=device_id,
            session_id=session_id,
            rotated_from=token,
        )
        _redis_remove_token(
            r,
            token,
            username=username,
            device_id=current.get("device_id"),
        )
    else:
        token_data: Dict[str, Any] = {
            "username": username,
            "expires_at": expires,
            "issued_at": issued_at,
            "device_id": device_id,
            "session_id": session_id,
            "rotated_from": token,
        }
        with _token_lock:
            _token_store.pop(token, None)
            _token_store[new_token] = token_data

    _audit_event(
        "token_rotated",
        username=username,
        old_token_suffix=token[-6:],
        new_token_suffix=new_token[-6:],
        device_id=device_id,
        session_id=session_id,
    )
    return LoginResponse(
        access_token=new_token,
        expires_at=expires.isoformat(),
        username=username,
    )


@router_dashboard_auth.post("/logout-all")
def dashboard_logout_all(user: Dict[str, Any] = Depends(require_dashboard_user)):
    username = str(user.get("username") or "admin")
    current_token = user.get("token")
    revoked = revoke_user_tokens(username=username, exclude_token=current_token)
    if current_token:
        r = _get_redis_client()
        if r is not None:
            _redis_remove_token(
                r,
                str(current_token),
                username=username,
                device_id=user.get("device_id"),
            )
        else:
            with _token_lock:
                _token_store.pop(str(current_token), None)
        revoked += 1
    _audit_event("logout_all", username=username, revoked=revoked)
    return {"status": "ok", "revoked": revoked}


@router_dashboard_auth.post("/revoke-device")
def dashboard_revoke_device(
    payload: RevokeDeviceRequest,
    user: Dict[str, Any] = Depends(require_dashboard_user),
):
    username = str(user.get("username") or "admin")
    revoked = revoke_user_tokens(username=username, device_id=payload.device_id)
    _audit_event(
        "revoke_device",
        username=username,
        device_id=payload.device_id,
        revoked=revoked,
    )
    return {
        "status": "ok",
        "username": username,
        "device_id": payload.device_id,
        "revoked": revoked,
    }


@router_dashboard_auth.post("/force-logout")
def dashboard_force_logout(
    payload: ForceLogoutRequest,
    x_admin_key: str | None = Header(default=None, alias="x-admin-key"),
):
    if not _admin_key_valid(x_admin_key):
        raise HTTPException(status_code=403, detail="x-admin-key header gereklidir.")
    revoked = revoke_user_tokens(
        username=payload.username,
        device_id=payload.device_id,
    )
    _audit_event(
        "force_logout",
        username=payload.username,
        device_id=payload.device_id,
        revoked=revoked,
    )
    return {
        "status": "ok",
        "username": payload.username,
        "device_id": payload.device_id,
        "revoked": revoked,
    }
