from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query

from .config import Paths
from .dashboard_auth import require_dashboard_user
from .dashboard_store import DashboardStore, SQLALCHEMY_AVAILABLE
from .utils import get_logger

logger = get_logger("dashboard")

# ── Redis cache (optional — graceful degradation when Redis is unavailable) ───

_CACHE_TTL_SECONDS = int(os.getenv("DASHBOARD_CACHE_TTL_SECONDS", "45"))
_CACHE_PREFIX = "ds:dash:overview:"
_dashboard_redis: Any = None


def _get_dashboard_redis() -> Any | None:
    """Lazily connect to Redis for dashboard caching. Returns None on failure."""
    global _dashboard_redis
    if _dashboard_redis is not None:
        return _dashboard_redis
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        return None
    try:
        import redis as _redis  # type: ignore[import]

        client = _redis.Redis.from_url(redis_url, decode_responses=True, socket_timeout=2)
        client.ping()
        _dashboard_redis = client
        logger.info("Dashboard cache: Redis active (TTL=%ds)", _CACHE_TTL_SECONDS)
        return _dashboard_redis
    except Exception as exc:
        logger.debug("Dashboard cache: Redis unavailable, no caching. reason=%s", exc)
        return None


def _cache_get(key: str) -> Dict[str, Any] | None:
    r = _get_dashboard_redis()
    if r is None:
        return None
    try:
        raw = r.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None


def _cache_set(key: str, value: Dict[str, Any]) -> None:
    r = _get_dashboard_redis()
    if r is None:
        return
    try:
        r.setex(key, _CACHE_TTL_SECONDS, json.dumps(value, default=str))
    except Exception as exc:
        logger.debug("Dashboard cache write failed: %s", exc)


def _cache_invalidate(pattern: str = f"{_CACHE_PREFIX}*") -> None:
    """Invalidate all dashboard overview cache keys (used on reload)."""
    r = _get_dashboard_redis()
    if r is None:
        return
    try:
        keys = r.keys(pattern)
        if keys:
            r.delete(*keys)
    except Exception as exc:
        logger.debug("Dashboard cache invalidation failed: %s", exc)

router_dashboard = APIRouter(prefix="/dashboard/api", tags=["dashboard"])

_store = None


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read JSON file %s: %s", path, exc)
        return default


def _run_dirs(metrics_root: Path) -> List[str]:
    if not metrics_root.exists():
        return []
    dirs = [p.name for p in metrics_root.iterdir() if p.is_dir()]
    dirs.sort(reverse=True)
    return dirs


def _detect_latest_run_id(metrics_root: Path) -> str:
    latest_json = _read_json(metrics_root / "latest.json", {})
    latest_run = latest_json.get("run_id") if isinstance(latest_json, dict) else None
    if latest_run:
        return str(latest_run)

    dirs = _run_dirs(metrics_root)
    if not dirs:
        raise HTTPException(
            status_code=404, detail="No experiment run found under reports/metrics"
        )
    return dirs[0]


def _load_snapshot(paths: Paths, run_id: str | None = None) -> Dict[str, Any]:
    metrics_root = paths.reports_metrics
    selected_run = run_id or _detect_latest_run_id(metrics_root)
    run_dir = metrics_root / selected_run
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {selected_run}")

    cv_summary = _read_json(run_dir / "cv_summary.json", {})
    decision_policy = _read_json(run_dir / "decision_policy.json", {})

    test_metrics: Dict[str, Dict[str, Any]] = {}
    for path in sorted(run_dir.glob("*_metrics.json")):
        if path.name == "calibration_metrics.json":
            continue
        payload = _read_json(path, {})
        if not isinstance(payload, dict):
            continue
        required = {"roc_auc", "f1", "precision", "recall"}
        if required.issubset(payload.keys()):
            model_name = path.name.replace("_metrics.json", "")
            test_metrics[model_name] = payload

    model_names = sorted(set(cv_summary.keys()) | set(test_metrics.keys()))
    models: List[Dict[str, Any]] = []

    for name in model_names:
        cv = cv_summary.get(name, {}) if isinstance(cv_summary, dict) else {}
        tm = test_metrics.get(name, {})
        models.append(
            {
                "model_name": name,
                "train_cv_roc_auc_mean": cv.get("roc_auc_mean"),
                "train_cv_roc_auc_std": cv.get("roc_auc_std"),
                "cv_folds": cv.get("cv_folds"),
                "test_roc_auc": tm.get("roc_auc"),
                "test_f1": tm.get("f1"),
                "test_precision": tm.get("precision"),
                "test_recall": tm.get("recall"),
                "test_threshold": tm.get("threshold"),
                "n_test": tm.get("n_test"),
                "positive_rate_test": tm.get("positive_rate_test"),
            }
        )

    models.sort(
        key=lambda x: (x.get("test_roc_auc") is None, -(x.get("test_roc_auc") or 0.0))
    )

    return {
        "run_id": selected_run,
        "available_runs": _run_dirs(metrics_root),
        "source_path": str(run_dir),
        "champion": {
            "selected_model": decision_policy.get("selected_model"),
            "threshold": decision_policy.get("threshold"),
            "expected_net_profit": decision_policy.get("expected_net_profit"),
            "max_action_rate": decision_policy.get("max_action_rate"),
            "ranking_mode": decision_policy.get("ranking_mode"),
        },
        "models": models,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def init_dashboard_store() -> None:
    global _store

    if not SQLALCHEMY_AVAILABLE:
        logger.warning("SQLAlchemy is not installed; dashboard persistence disabled")
        _store = None
        return

    database_url = os.getenv("DATABASE_URL", "sqlite:///./reports/dashboard.db")

    try:
        _store = DashboardStore(database_url=database_url)
        logger.info("Dashboard database initialized: %s", database_url)
    except Exception as exc:
        logger.warning(
            "Could not initialize dashboard database (%s): %s", database_url, exc
        )
        _store = None


def _persist_snapshot(snapshot: Dict[str, Any]) -> None:
    if _store is None:
        return
    try:
        _store.upsert_snapshot(snapshot)
    except Exception as exc:
        logger.warning("Dashboard snapshot could not be persisted: %s", exc)


@router_dashboard.get("/overview")
def dashboard_overview(
    run_id: str | None = Query(default=None),
    _user: Dict[str, Any] = Depends(require_dashboard_user),
):
    """Return overview snapshot for the given run (or latest).

    Results are Redis-cached for ``DASHBOARD_CACHE_TTL_SECONDS`` (default 45 s)
    to avoid repeated filesystem scans on refresh-heavy dashboards.
    The cache is keyed on run_id so different runs are cached independently.
    """
    cache_key = f"{_CACHE_PREFIX}{run_id or 'latest'}"
    cached = _cache_get(cache_key)
    if cached is not None:
        cached["cache_hit"] = True
        return cached

    paths = Paths()
    snapshot = _load_snapshot(paths=paths, run_id=run_id)
    _persist_snapshot(snapshot)
    snapshot["db_enabled"] = _store is not None
    snapshot["cache_hit"] = False
    _cache_set(cache_key, snapshot)
    return snapshot


@router_dashboard.get("/runs")
def dashboard_runs(
    limit: int = Query(default=20, ge=1, le=100),
    _user: Dict[str, Any] = Depends(require_dashboard_user),
):
    paths = Paths()
    filesystem_runs = _run_dirs(paths.reports_metrics)

    db_runs: List[Dict[str, Any]] = []
    if _store is not None:
        try:
            db_runs = _store.list_runs(limit=limit)
        except Exception as exc:
            logger.warning("Dashboard runs could not be read from database: %s", exc)

    return {
        "runs": filesystem_runs[:limit],
        "db_runs": db_runs,
        "db_enabled": _store is not None,
    }


def _mask_database_url(url: str) -> str:
    return re.sub(r"://([^:/?#]+):([^@]+)@", r"://\1:***@", url)


@router_dashboard.get("/db-status")
def dashboard_db_status(_user: Dict[str, Any] = Depends(require_dashboard_user)):
    database_url = os.getenv("DATABASE_URL", "sqlite:///./reports/dashboard.db")
    masked_url = _mask_database_url(database_url)
    backend = "postgresql" if database_url.startswith("postgresql") else "sqlite"

    connected = False
    reason = "Dashboard store not initialized"
    if _store is not None:
        try:
            with _store.engine.connect() as conn:
                conn.exec_driver_sql("SELECT 1")
            connected = True
            reason = "ok"
        except Exception as exc:
            reason = str(exc)

    return {
        "db_enabled": _store is not None,
        "database_backend": backend,
        "database_url": masked_url,
        "connected": connected,
        "reason": reason,
    }


@router_dashboard.get("/monitoring")
def dashboard_monitoring(_user: Dict[str, Any] = Depends(require_dashboard_user)):
    """
    Return the latest drift / monitoring report.

    Reads ``reports/monitoring/latest_monitoring_report.json`` first;
    falls back to the newest dated subdirectory's ``monitoring_report.json``.
    Produces a 404 if no report exists (run ``python main.py monitor`` first).
    """
    paths = Paths()
    monitoring_dir = paths.reports_monitoring

    # 1. Try the pre-computed latest file
    report = _read_json(monitoring_dir / "latest_monitoring_report.json", None)

    # 2. Scan dated sub-directories (newest first)
    if report is None and monitoring_dir.exists():
        dated_dirs = sorted(
            [d for d in monitoring_dir.iterdir() if d.is_dir()],
            reverse=True,
        )
        for d in dated_dirs:
            report = _read_json(d / "monitoring_report.json", None)
            if report is not None:
                break

    if report is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Monitoring raporu bulunamadı. "
                "Önce `python main.py monitor` komutunu çalıştırın."
            ),
        )

    return report


@router_dashboard.get("/explain")
def dashboard_explain(
    run_id: str | None = Query(default=None),
    _user: Dict[str, Any] = Depends(require_dashboard_user),
):
    """
    Return permutation feature importance for the selected (or latest) run.

    Prefers the richer ``permutation_importance.json`` (ranked list with std);
    falls back to ``feature_importance.json`` (flat dict) and wraps it into
    the same shape so consumers always see a consistent ``ranking`` array.
    """
    paths = Paths()
    metrics_root = paths.reports_metrics
    selected_run = run_id or _detect_latest_run_id(metrics_root)
    run_dir = metrics_root / selected_run

    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Koşu bulunamadı: {selected_run}")

    # Prefer the richer permutation importance report
    report = _read_json(run_dir / "permutation_importance.json", None)

    if report is None:
        # Fall back to flat dict and normalise it
        raw = _read_json(run_dir / "feature_importance.json", None)
        if isinstance(raw, dict):
            ranking = sorted(
                [
                    {"feature": k, "importance_mean": v, "importance_std": None}
                    for k, v in raw.items()
                ],
                key=lambda x: -(x["importance_mean"] or 0),
            )
            report = {
                "method": "feature_importance",
                "scoring": "unknown",
                "n_repeats": None,
                "n_features": len(ranking),
                "ranking": ranking,
            }

    if report is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Önem raporu bulunamadı ({selected_run}). "
                "Önce `python main.py explain` komutunu çalıştırın."
            ),
        )

    return {"run_id": selected_run, **report}


@router_dashboard.get("/system")
def dashboard_system(_user: Dict[str, Any] = Depends(require_dashboard_user)):
    """
    Aggregate health check for all backend dependencies.

    Probes: Database, Redis, Ollama (LLM), and the serving model artefact.
    Returns per-service status objects plus an ``overall`` summary field.
    """
    import urllib.request as _urllib_request
    from urllib.parse import urlparse as _urlparse

    paths = Paths()
    services: Dict[str, Any] = {}

    # ── 1. Database ───────────────────────────────────────────────────────────
    database_url = os.getenv("DATABASE_URL", "sqlite:///./reports/dashboard.db")
    db_backend = "postgresql" if database_url.startswith("postgresql") else "sqlite"
    db_connected = False
    db_reason = "store not initialized"
    if _store is not None:
        try:
            with _store.engine.connect() as conn:
                conn.exec_driver_sql("SELECT 1")
            db_connected = True
            db_reason = "ok"
        except Exception as exc:
            db_reason = str(exc)[:300]

    services["database"] = {
        "name": "PostgreSQL" if db_backend == "postgresql" else "SQLite",
        "status": "ok" if db_connected else "error",
        "connected": db_connected,
        "backend": db_backend,
        "url": _mask_database_url(database_url),
        "reason": db_reason,
    }

    # ── 2. Redis ──────────────────────────────────────────────────────────────
    redis_url = os.getenv("REDIS_URL", "")
    redis_ok = False
    redis_reason = "REDIS_URL yapılandırılmamış (isteğe bağlı)"
    if redis_url:
        try:
            import redis as _redis  # type: ignore[import]

            r = _redis.Redis.from_url(
                redis_url, socket_timeout=2, decode_responses=True
            )
            r.ping()
            redis_ok = True
            redis_reason = "ok"
        except Exception as exc:
            redis_reason = str(exc)[:300]

    masked_redis: str | None = None
    if redis_url:
        masked_redis = redis_url.split("@")[-1] if "@" in redis_url else redis_url

    services["redis"] = {
        "name": "Redis",
        "status": "ok" if redis_ok else ("unconfigured" if not redis_url else "error"),
        "connected": redis_ok,
        "url": masked_redis,
        "reason": redis_reason,
    }

    # ── 3. Ollama (LLM) ───────────────────────────────────────────────────────
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    ollama_ok = False
    ollama_model_found = False
    ollama_reason = "erişilemiyor"
    try:
        parsed = _urlparse(ollama_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                "Invalid OLLAMA_BASE_URL; only absolute http(s) URLs are allowed"
            )
        req = _urllib_request.urlopen(f"{ollama_url}/api/tags", timeout=3)  # nosec B310
        body = json.loads(req.read())
        available_models = [m.get("name", "") for m in body.get("models", [])]
        ollama_ok = True
        ollama_model_found = any(ollama_model in m for m in available_models)
        ollama_reason = (
            "ok"
            if ollama_model_found
            else f"'{ollama_model}' modeli Ollama'da bulunamadı"
        )
    except Exception as exc:
        ollama_reason = str(exc)[:300]

    services["ollama"] = {
        "name": "Ollama (LLM)",
        "status": (
            "ok"
            if (ollama_ok and ollama_model_found)
            else ("warning" if ollama_ok else "error")
        ),
        "connected": ollama_ok,
        "model_found": ollama_model_found,
        "url": ollama_url,
        "model": ollama_model,
        "reason": ollama_reason,
    }

    # ── 4. Serving model artefact ─────────────────────────────────────────────
    model_ok = False
    model_name: str | None = None
    model_reason = "yüklü değil"
    try:
        # decision_policy.json'ı kullan — load_serving_state ile aynı mantık
        active_slot_path = paths.reports_metrics / "active_slot.json"
        if active_slot_path.exists():
            slot_payload = _read_json(active_slot_path, {})
            slot = str((slot_payload or {}).get("active_slot", "default"))
            policy_path = (
                paths.reports_metrics / f"decision_policy.{slot}.json"
                if slot in {"blue", "green"}
                else paths.reports_metrics / "decision_policy.json"
            )
        else:
            policy_path = paths.reports_metrics / "decision_policy.json"

        policy = _read_json(policy_path, None)
        if policy and isinstance(policy, dict):
            artifact_rel = policy.get("selected_model_artifact")
            model_name = policy.get("selected_model") or policy.get("run_id")
            if artifact_rel:
                artifact_path = paths.project_root / artifact_rel
                model_ok = artifact_path.exists()
                model_reason = (
                    "ok" if model_ok else f"artefact bulunamadı: {artifact_path}"
                )
            else:
                model_reason = "decision_policy'de selected_model_artifact yok"
        else:
            # Fallback: model_registry.json (string değer desteğiyle)
            latest_json = _read_json(paths.models / "latest.json", None)
            run_id_str = (latest_json or {}).get("run_id", "")
            latest_registry_path = (latest_json or {}).get("model_registry")
            registry_path = None
            if isinstance(latest_registry_path, str) and latest_registry_path:
                p = Path(latest_registry_path)
                registry_path = p if p.is_absolute() else (paths.project_root / p)
            elif run_id_str:
                registry_path = (
                    paths.reports_metrics / run_id_str / "model_registry.json"
                )

            registry = (
                _read_json(registry_path, None) if registry_path is not None else None
            )
            if registry and isinstance(registry, dict):
                for v in registry.values():
                    artifact_rel = (
                        v
                        if isinstance(v, str)
                        else (v.get("path") if isinstance(v, dict) else None)
                    )
                    if artifact_rel and (paths.project_root / artifact_rel).exists():
                        model_ok = True
                        model_name = run_id_str
                        model_reason = "ok"
                        break
            if not model_ok:
                model_reason = f"run_id={run_id_str} (artefact yolu doğrulanamadı)"
    except Exception as exc:
        model_reason = str(exc)[:300]

    services["model"] = {
        "name": "Servis Modeli",
        "status": "ok" if model_ok else "warning",
        "loaded": model_ok,
        "model_name": model_name,
        "reason": model_reason,
    }

    # ── Overall summary ───────────────────────────────────────────────────────
    statuses = [s["status"] for s in services.values()]
    if all(s == "ok" for s in statuses):
        overall = "ok"
    elif any(s == "error" for s in statuses):
        overall = "degraded"
    else:
        overall = "partial"

    return {
        "overall": overall,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "services": services,
    }
