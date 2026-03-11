"""Tests for src.db_bootstrap — Alembic migrations + table verification."""

from __future__ import annotations

from pathlib import Path
import pytest

from src.db_bootstrap import (
    CORE_REQUIRED_TABLES,
    OPTIONAL_TABLE_FLAGS,
    REQUIRED_TABLES,
    ensure_required_tables,
    resolve_required_tables_from_env,
    run_migrations,
)


class TestRunMigrations:
    """run_migrations applies Alembic migrations to the given DB URL."""

    def test_applies_migrations_sqlite(self, tmp_path: Path) -> None:
        db_url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
        # run_migrations should call alembic upgrade head without error
        run_migrations(db_url)

    def test_missing_alembic_ini_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_migrations raises RuntimeError when alembic.ini is absent."""
        import src.db_bootstrap as mod

        db_url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
        # Point db_bootstrap's __file__ to a fake path under tmp_path so that
        # Path(__file__).resolve().parent.parent resolves to tmp_path, where
        # no alembic.ini exists.
        fake_file = tmp_path / "src" / "db_bootstrap.py"
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        fake_file.touch()
        monkeypatch.setattr(mod, "__file__", str(fake_file))

        with pytest.raises(RuntimeError, match="Alembic config missing"):
            run_migrations(db_url)


class TestEnsureRequiredTables:
    """ensure_required_tables raises RuntimeError when tables are missing."""

    def test_raises_when_tables_missing(self, tmp_path: Path) -> None:
        db_url = f"sqlite:///{(tmp_path / 'empty.db').as_posix()}"
        with pytest.raises(RuntimeError, match="Missing required tables"):
            ensure_required_tables(db_url)

    def test_passes_with_custom_empty_list(self, tmp_path: Path) -> None:
        db_url = f"sqlite:///{(tmp_path / 'empty.db').as_posix()}"
        # No required tables → should pass
        ensure_required_tables(db_url, required_tables=[])

    def test_passes_after_migrations(self, tmp_path: Path) -> None:
        db_url = f"sqlite:///{(tmp_path / 'migrated.db').as_posix()}"
        run_migrations(db_url)
        ensure_required_tables(db_url)


class TestRequiredTablesConstant:
    """REQUIRED_TABLES contains all expected table names."""

    def test_contains_core_tables(self) -> None:
        assert "experiment_runs" in REQUIRED_TABLES
        assert "model_metrics" in REQUIRED_TABLES
        assert "users" in REQUIRED_TABLES
        assert "hotel_guests" in REQUIRED_TABLES
        assert "knowledge_chunks" in REQUIRED_TABLES

    def test_is_tuple(self) -> None:
        assert isinstance(REQUIRED_TABLES, tuple)


class TestRequiredTablesFlags:
    """Optional module table requirements can be toggled with env flags."""

    def test_default_includes_optional_tables(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DB_REQUIRE_GUESTS_TABLE", raising=False)
        monkeypatch.delenv("DB_REQUIRE_KNOWLEDGE_TABLE", raising=False)
        resolved = resolve_required_tables_from_env()
        assert set(resolved) == set(REQUIRED_TABLES)

    def test_flags_can_disable_optional_tables(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DB_REQUIRE_GUESTS_TABLE", "false")
        monkeypatch.setenv("DB_REQUIRE_KNOWLEDGE_TABLE", "0")
        resolved = resolve_required_tables_from_env()
        assert set(resolved) == set(CORE_REQUIRED_TABLES)

    def test_optional_flag_map_contains_known_tables(self) -> None:
        assert OPTIONAL_TABLE_FLAGS["hotel_guests"] == "DB_REQUIRE_GUESTS_TABLE"
        assert OPTIONAL_TABLE_FLAGS["knowledge_chunks"] == "DB_REQUIRE_KNOWLEDGE_TABLE"
