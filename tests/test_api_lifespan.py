"""Tests for src.api_lifespan — startup/shutdown lifecycle helpers."""

from __future__ import annotations

import asyncio

import pytest


# ── Environment helpers ────────────────────────────────────────────────────────


class TestIsNonProdLikeEnv:
    """_is_non_prod_like_env returns True only for dev/local/test."""

    def test_dev(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DS_ENV", "development")
        from src.api_lifespan import _is_non_prod_like_env

        assert _is_non_prod_like_env() is True

    def test_test(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DS_ENV", "test")
        from src.api_lifespan import _is_non_prod_like_env

        assert _is_non_prod_like_env() is True

    def test_production(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DS_ENV", "production")
        from src.api_lifespan import _is_non_prod_like_env

        assert _is_non_prod_like_env() is False

    def test_staging(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DS_ENV", "staging")
        from src.api_lifespan import _is_non_prod_like_env

        assert _is_non_prod_like_env() is False


# ── Admin key validation ───────────────────────────────────────────────────────


class TestValidateAdminKeyStartup:
    """_validate_admin_key_startup must raise in production without DS_ADMIN_KEY."""

    def test_dev_env_no_key_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DS_ENV", "development")
        monkeypatch.delenv("DS_ADMIN_KEY", raising=False)
        from src.api_lifespan import _validate_admin_key_startup

        _validate_admin_key_startup()  # should not raise

    def test_prod_env_no_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DS_ENV", "production")
        monkeypatch.delenv("DS_ADMIN_KEY", raising=False)
        from src.api_lifespan import _validate_admin_key_startup

        with pytest.raises(RuntimeError, match="DS_ADMIN_KEY must be set"):
            _validate_admin_key_startup()

    def test_prod_env_with_key_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DS_ENV", "production")
        monkeypatch.setenv("DS_ADMIN_KEY", "secret-123")
        from src.api_lifespan import _validate_admin_key_startup

        _validate_admin_key_startup()  # should not raise


# ── Rate limiter builder ───────────────────────────────────────────────────────


class TestBuildRuntimeRateLimiter:
    """_build_runtime_rate_limiter returns a BaseRateLimiter instance."""

    def test_returns_limiter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RATE_LIMIT_BACKEND", raising=False)
        monkeypatch.delenv("REDIS_URL", raising=False)
        from src.api_lifespan import _build_runtime_rate_limiter

        limiter = _build_runtime_rate_limiter()
        assert hasattr(limiter, "allow")

    def test_env_override_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RATE_LIMIT_BACKEND", "memory")
        monkeypatch.delenv("REDIS_URL", raising=False)
        from src.api_lifespan import _build_runtime_rate_limiter

        limiter = _build_runtime_rate_limiter()
        assert limiter.allow("test-client", 100) is True


# ── Multi-worker warning ───────────────────────────────────────────────────────


class TestWarnMultiworkerRateLimit:
    """_warn_multiworker_rate_limit emits warning for memory + multi-worker."""

    def test_single_worker_no_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("WEB_CONCURRENCY", "1")
        monkeypatch.delenv("RATE_LIMIT_BACKEND", raising=False)
        from src.api_lifespan import _warn_multiworker_rate_limit

        with caplog.at_level("WARNING"):
            _warn_multiworker_rate_limit()
        assert "RATE LIMIT UYARISI" not in caplog.text

    def test_multi_worker_memory_warns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WEB_CONCURRENCY", "4")
        monkeypatch.setenv("RATE_LIMIT_BACKEND", "memory")
        import src.api_lifespan as mod

        calls: list[tuple] = []

        def _fake_warning(*args, **kwargs):
            calls.append((args, kwargs))

        monkeypatch.setattr(mod.logger, "warning", _fake_warning)
        mod._warn_multiworker_rate_limit()
        assert calls, "Expected warning call for multi-worker memory rate limit"


# ── Periodic chat cleanup ──────────────────────────────────────────────────────


class TestPeriodicChatCleanup:
    """_periodic_chat_cleanup runs until cancelled."""

    def test_cancellation(self) -> None:
        from src.api_lifespan import _periodic_chat_cleanup

        async def _run() -> None:
            task = asyncio.create_task(_periodic_chat_cleanup(interval_seconds=0))
            await asyncio.sleep(0.05)
            task.cancel()
            await task
            assert task.done()

        asyncio.run(_run())
