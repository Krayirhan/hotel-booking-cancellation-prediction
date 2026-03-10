"""
alembic/env.py

Alembic migration environment for ds-project DashboardStore schema.

DATABASE_URL is read from the environment variable at runtime — never
hardcoded here.  Falls back to a local SQLite file for development
convenience when the env var is not set.

Usage:
  # Generate a new revision after changing DashboardStore tables:
  alembic revision --autogenerate -m "add column X to model_metrics"

  # Apply pending migrations:
  alembic upgrade head

  # Rollback one step:
  alembic downgrade -1
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# ── Alembic config ───────────────────────────────────────────────────────────
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ── Resolve DATABASE_URL from environment ─────────────────────────────────────
_db_url = os.getenv(
    "DATABASE_URL",
    config.get_main_option("sqlalchemy.url", "sqlite:///./reports/dashboard.db"),
)
# Override whatever is in alembic.ini with the runtime env var
config.set_main_option("sqlalchemy.url", _db_url)

# ── Target metadata (autogenerate support) ────────────────────────────────────
# Tüm store'ların tablo şemaları burada birleştirilir:
#   - DashboardStore   → experiment_runs, model_metrics
#   - UserStore        → users
#   - GuestStore       → hotel_guests
#   - KnowledgeDbStore → knowledge_chunks
# Bu sayede `alembic revision --autogenerate` gerçek schema diffini hesaplayabilir.
try:
    from sqlalchemy import (
        Boolean,
        Column,
        Date,
        DateTime,
        Float,
        Integer,
        JSON,
        MetaData as _Meta,
        String,
        Table,
        Text,
    )

    _meta = _Meta()

    # ── DashboardStore ────────────────────────────────────────────────────────
    Table(
        "experiment_runs",
        _meta,
        Column("run_id", String(64), primary_key=True),
        Column("selected_model", String(256), nullable=True),
        Column("threshold", Float, nullable=True),
        Column("expected_net_profit", Float, nullable=True),
        Column("max_action_rate", Float, nullable=True),
        Column("source_path", String(1024), nullable=True),
        Column("updated_at", DateTime(timezone=True), nullable=False),
    )
    Table(
        "model_metrics",
        _meta,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("run_id", String(64), nullable=False),
        Column("model_name", String(256), nullable=False),
        Column("train_cv_roc_auc_mean", Float, nullable=True),
        Column("train_cv_roc_auc_std", Float, nullable=True),
        Column("test_roc_auc", Float, nullable=True),
        Column("test_f1", Float, nullable=True),
        Column("test_precision", Float, nullable=True),
        Column("test_recall", Float, nullable=True),
        Column("test_threshold", Float, nullable=True),
        Column("n_test", Integer, nullable=True),
        Column("positive_rate_test", Float, nullable=True),
        Column("updated_at", DateTime(timezone=True), nullable=False),
    )

    # ── UserStore ─────────────────────────────────────────────────────────────
    Table(
        "users",
        _meta,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("username", String(64), nullable=False, unique=True),
        Column("password_hash", String(256), nullable=False),
        Column("role", String(32), nullable=False, server_default="viewer"),
        Column("is_active", Boolean, nullable=False, server_default="true"),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("updated_at", DateTime(timezone=True), nullable=False),
    )

    # ── GuestStore ────────────────────────────────────────────────────────────
    Table(
        "hotel_guests",
        _meta,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("first_name", String(100), nullable=False),
        Column("last_name", String(100), nullable=False),
        Column("email", String(200), nullable=True),
        Column("phone", String(30), nullable=True),
        Column("nationality", String(10), nullable=True),
        Column("identity_no", String(50), nullable=True),
        Column("birth_date", Date, nullable=True),
        Column("gender", String(10), nullable=True),
        Column("vip_status", Boolean, nullable=False, server_default="false"),
        Column("notes", Text, nullable=True),
        Column("hotel", String(50), nullable=True),
        Column("lead_time", Integer, nullable=True),
        Column("deposit_type", String(50), nullable=True),
        Column("market_segment", String(50), nullable=True),
        Column("adults", Integer, nullable=True),
        Column("children", Float, nullable=True),
        Column("babies", Integer, nullable=True),
        Column("stays_in_week_nights", Integer, nullable=True),
        Column("stays_in_weekend_nights", Integer, nullable=True),
        Column("is_repeated_guest", Integer, nullable=True),
        Column("previous_cancellations", Integer, nullable=True),
        Column("adr", Float, nullable=True),
        Column("risk_score", Float, nullable=True),
        Column("risk_label", String(20), nullable=True),
        Column("risk_scored_at", DateTime(timezone=True), nullable=True),
        Column("is_active", Boolean, nullable=False, server_default="true"),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("updated_at", DateTime(timezone=True), nullable=False),
    )

    # ── KnowledgeDbStore (pgvector) ───────────────────────────────────────────
    Table(
        "knowledge_chunks",
        _meta,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("chunk_id", String(80), nullable=False, unique=True),
        Column("category", String(50), nullable=False, server_default="general"),
        Column("tags", JSON, nullable=True),
        Column("title", String(200), nullable=False),
        Column("content", Text, nullable=False),
        Column("priority", Integer, nullable=False, server_default="5"),
        Column("is_active", Boolean, nullable=False, server_default="true"),
        Column("has_embedding", Boolean, nullable=False, server_default="false"),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("updated_at", DateTime(timezone=True), nullable=False),
    )

    target_metadata = _meta
except Exception:
    # Fallback: autogenerate will not diff against schema but migrations still run
    target_metadata = None


def run_migrations_offline() -> None:
    """Offline mode: emit SQL to stdout without connecting to the DB."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Online mode: connect to the DB and apply migrations."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
