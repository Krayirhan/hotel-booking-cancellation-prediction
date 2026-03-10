"""dashboard_store.py — SQLAlchemy persistence layer for dashboard experiment data.

DashboardStore writes experiment run snapshots (champion selection, model metrics)
to PostgreSQL or SQLite and exposes a ``list_runs`` query for the dashboard API.

This module is intentionally separate from the FastAPI router (``dashboard.py``)
to make unit testing and potential future migration easier.

Usage:
    from src.dashboard_store import DashboardStore

    store = DashboardStore(database_url="postgresql+psycopg://...")
    store.upsert_snapshot(snapshot_dict)
    runs = store.list_runs(limit=20)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from .utils import get_logger

logger = get_logger("dashboard_store")

try:
    from sqlalchemy import (
        Column,
        DateTime,
        Float,
        Integer,
        MetaData,
        String,
        Table,
        create_engine,
        delete,
        insert,
        select,
    )

    SQLALCHEMY_AVAILABLE = True
except Exception:
    SQLALCHEMY_AVAILABLE = False


class DashboardStore:
    """PostgreSQL/SQLite persistence for experiment run snapshots.

    Tables managed:
        experiment_runs  — one row per run (champion selection + metadata)
        model_metrics    — one row per model per run (AUC, F1, precision, recall, …)
    """

    def __init__(self, database_url: str) -> None:
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy is not available")

        self.database_url = database_url
        self.engine = create_engine(database_url, pool_pre_ping=True, future=True)
        self.metadata = MetaData()

        self.runs = Table(
            "experiment_runs",
            self.metadata,
            Column("run_id", String(64), primary_key=True),
            Column("selected_model", String(256), nullable=True),
            Column("threshold", Float, nullable=True),
            Column("expected_net_profit", Float, nullable=True),
            Column("max_action_rate", Float, nullable=True),
            Column("source_path", String(1024), nullable=True),
            Column("updated_at", DateTime(timezone=True), nullable=False),
        )

        self.model_metrics = Table(
            "model_metrics",
            self.metadata,
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

    def create_schema(self) -> None:
        """Create tables if they do not already exist."""
        self.metadata.create_all(self.engine)

    def upsert_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Replace all metrics for a run_id with the given snapshot.

        Fully replaces `model_metrics` rows for the run (delete + insert)
        to avoid partial-update race conditions.
        """
        run_id = snapshot["run_id"]
        champion = snapshot.get("champion") or {}
        now = datetime.now(timezone.utc)

        with self.engine.begin() as conn:
            conn.execute(
                delete(self.model_metrics).where(self.model_metrics.c.run_id == run_id)
            )
            conn.execute(delete(self.runs).where(self.runs.c.run_id == run_id))

            conn.execute(
                insert(self.runs).values(
                    run_id=run_id,
                    selected_model=champion.get("selected_model"),
                    threshold=champion.get("threshold"),
                    expected_net_profit=champion.get("expected_net_profit"),
                    max_action_rate=champion.get("max_action_rate"),
                    source_path=snapshot.get("source_path"),
                    updated_at=now,
                )
            )

            for row in snapshot.get("models", []):
                conn.execute(
                    insert(self.model_metrics).values(
                        run_id=run_id,
                        model_name=row.get("model_name"),
                        train_cv_roc_auc_mean=row.get("train_cv_roc_auc_mean"),
                        train_cv_roc_auc_std=row.get("train_cv_roc_auc_std"),
                        test_roc_auc=row.get("test_roc_auc"),
                        test_f1=row.get("test_f1"),
                        test_precision=row.get("test_precision"),
                        test_recall=row.get("test_recall"),
                        test_threshold=row.get("test_threshold"),
                        n_test=row.get("n_test"),
                        positive_rate_test=row.get("positive_rate_test"),
                        updated_at=now,
                    )
                )

    def list_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent experiment runs ordered by updated_at desc."""
        with self.engine.begin() as conn:
            stmt = (
                select(self.runs).order_by(self.runs.c.updated_at.desc()).limit(limit)
            )
            rows = conn.execute(stmt).mappings().all()

        out: List[Dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "run_id": row["run_id"],
                    "selected_model": row["selected_model"],
                    "threshold": row["threshold"],
                    "expected_net_profit": row["expected_net_profit"],
                    "max_action_rate": row["max_action_rate"],
                    "updated_at": (
                        row["updated_at"].isoformat() if row["updated_at"] else None
                    ),
                }
            )
        return out
