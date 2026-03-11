"""
alembic/env.py

Alembic migration environment for hotel-booking-cancellation-prediction schema.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

_db_url = os.getenv(
    "DATABASE_URL",
    config.get_main_option("sqlalchemy.url", "sqlite:///./reports/dashboard.db"),
)
config.set_main_option("sqlalchemy.url", _db_url)

# Target metadata for autogenerate/check.
# Keep this aligned with alembic revision history.
try:
    from sqlalchemy import (
        Boolean,
        Column,
        Date,
        DateTime,
        Float,
        ForeignKeyConstraint,
        Index,
        Integer,
        JSON,
        MetaData as _Meta,
        String,
        Table,
        Text,
        UniqueConstraint,
    )

    _meta = _Meta()

    try:
        from pgvector.sqlalchemy import Vector

        _embedding_type = Vector(768).with_variant(Text(), "sqlite")
    except Exception:
        _embedding_type = Text()

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

    model_metrics = Table(
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
        ForeignKeyConstraint(
            ["run_id"], ["experiment_runs.run_id"], ondelete="CASCADE"
        ),
    )
    Index("ix_model_metrics_run_id", model_metrics.c.run_id)

    users = Table(
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
    Index("ix_users_username", users.c.username, unique=True)

    hotel_guests = Table(
        "hotel_guests",
        _meta,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("first_name", String(100), nullable=False),
        Column("last_name", String(100), nullable=False),
        Column("email", String(200), nullable=True),
        Column("phone", String(30), nullable=True),
        Column("nationality", String(3), nullable=True),
        Column("identity_no", String(50), nullable=True),
        Column("birth_date", Date, nullable=True),
        Column("gender", String(10), nullable=True),
        Column("vip_status", Boolean, nullable=False, server_default="false"),
        Column("notes", Text, nullable=True),
        Column("hotel", String(50), nullable=False, server_default="City Hotel"),
        Column("lead_time", Integer, nullable=False, server_default="0"),
        Column("deposit_type", String(30), nullable=False, server_default="No Deposit"),
        Column(
            "market_segment", String(30), nullable=False, server_default="Online TA"
        ),
        Column("adults", Integer, nullable=False, server_default="2"),
        Column("children", Integer, nullable=False, server_default="0"),
        Column("babies", Integer, nullable=False, server_default="0"),
        Column("stays_in_week_nights", Integer, nullable=False, server_default="0"),
        Column("stays_in_weekend_nights", Integer, nullable=False, server_default="1"),
        Column("is_repeated_guest", Integer, nullable=False, server_default="0"),
        Column("previous_cancellations", Integer, nullable=False, server_default="0"),
        Column("adr", Float, nullable=True),
        Column("risk_score", Float, nullable=True),
        Column("risk_label", String(10), nullable=True),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("updated_at", DateTime(timezone=True), nullable=False),
    )
    Index("ix_hotel_guests_last_name", hotel_guests.c.last_name)
    Index("ix_hotel_guests_email", hotel_guests.c.email)

    Table(
        "knowledge_chunks",
        _meta,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("chunk_id", String(50), nullable=False),
        Column("category", String(50), nullable=False, server_default="general"),
        Column("tags", JSON, nullable=False, server_default="[]"),
        Column("title", String(200), nullable=False),
        Column("content", Text, nullable=False),
        Column("priority", Integer, nullable=False, server_default="5"),
        Column("is_active", Boolean, nullable=False, server_default="true"),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("updated_at", DateTime(timezone=True), nullable=False),
        Column("embedding", _embedding_type, nullable=True),
        UniqueConstraint("chunk_id", name="uq_knowledge_chunks_chunk_id"),
    )

    target_metadata = _meta
except Exception:
    # Fallback: migrations still run but autogenerate/check is disabled.
    target_metadata = None


def run_migrations_offline() -> None:
    """Offline mode: emit SQL without connecting to the DB."""
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
