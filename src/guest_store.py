"""guest_store.py — PostgreSQL-backed hotel guest management.

Provides a SQLAlchemy-based GuestStore that:
  - stores guests in the `hotel_guests` table (Alembic migration 854e7dedec10)
  - separates personal info (DB only) from model features (used for prediction)
  - exposes CRUD helpers used by guests.py router

Usage:
    from src.guest_store import init_guest_store, get_guest_store

    # At startup (api.py lifespan):
    init_guest_store(engine)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    func,
    insert,
    or_,
    select,
    update,
    desc,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from .utils import get_logger

logger = get_logger("guest_store")

_guest_store: Optional["GuestStore"] = None


class GuestStoreError(RuntimeError):
    """Base class for GuestStore failures."""


class GuestStoreNotInitializedError(GuestStoreError):
    """Raised when module-level GuestStore singleton is not initialised."""


class GuestStoreUnavailableError(GuestStoreError):
    """Raised for operational DB failures (connection, SQL errors, etc.)."""


class GuestStore:
    """SQLAlchemy-backed hotel guest repository."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.metadata = MetaData()
        self.guests = Table(
            "hotel_guests",
            self.metadata,
            # ── Identity ──────────────────────────────────────────────────────
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("first_name", String(100), nullable=False),
            Column("last_name", String(100), nullable=False),
            Column("email", String(200), nullable=True),
            Column("phone", String(30), nullable=True),
            Column("nationality", String(3), nullable=True),  # ISO-3166 alpha-3
            Column("identity_no", String(50), nullable=True),  # TC / Pasaport
            Column("birth_date", Date(), nullable=True),
            Column("gender", String(10), nullable=True),  # M / F / other
            Column("vip_status", Boolean(), nullable=False, server_default="false"),
            Column("notes", Text(), nullable=True),
            # ── Model features (used for risk prediction) ─────────────────────
            Column("hotel", String(50), nullable=False, server_default="City Hotel"),
            Column("lead_time", Integer(), nullable=False, server_default="0"),
            Column(
                "deposit_type", String(30), nullable=False, server_default="No Deposit"
            ),
            Column(
                "market_segment", String(30), nullable=False, server_default="Online TA"
            ),
            Column("adults", Integer(), nullable=False, server_default="2"),
            Column("children", Integer(), nullable=False, server_default="0"),
            Column("babies", Integer(), nullable=False, server_default="0"),
            Column(
                "stays_in_week_nights", Integer(), nullable=False, server_default="0"
            ),
            Column(
                "stays_in_weekend_nights", Integer(), nullable=False, server_default="1"
            ),
            Column("is_repeated_guest", Integer(), nullable=False, server_default="0"),
            Column(
                "previous_cancellations", Integer(), nullable=False, server_default="0"
            ),
            Column("adr", Float(), nullable=True),
            # ── Prediction result ─────────────────────────────────────────────
            Column("risk_score", Float(), nullable=True),
            Column("risk_label", String(10), nullable=True),
            # ── Meta ──────────────────────────────────────────────────────────
            Column("created_at", DateTime(timezone=True), nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
        )
        Index("ix_hotel_guests_last_name", self.guests.c.last_name)
        Index("ix_hotel_guests_email", self.guests.c.email)

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def create_guest(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Insert a new guest row and return the complete row dict."""
        now = datetime.now(timezone.utc)
        row_data = {**data, "created_at": now, "updated_at": now}
        try:
            with self.engine.begin() as conn:
                stmt = insert(self.guests).values(**row_data)
                if getattr(conn.dialect, "insert_returning", False):
                    result = conn.execute(stmt.returning(self.guests))
                    new_row = result.mappings().first()
                else:
                    result = conn.execute(stmt)
                    inserted_id = (
                        result.inserted_primary_key[0]
                        if result.inserted_primary_key
                        else None
                    )
                    if inserted_id is None:
                        raise GuestStoreUnavailableError(
                            "Guest create failed: inserted id is unavailable"
                        )
                    new_row = (
                        conn.execute(
                            select(self.guests).where(self.guests.c.id == inserted_id)
                        )
                        .mappings()
                        .first()
                    )
        except SQLAlchemyError as exc:
            raise GuestStoreUnavailableError("Guest create failed") from exc
        if new_row is None:
            raise GuestStoreUnavailableError("Guest create failed: empty insert result")
        return dict(new_row)

    def list_guests(
        self,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Return guests ordered by newest first, with optional name/email search."""
        stmt = select(self.guests).order_by(desc(self.guests.c.created_at))
        if search:
            pattern = f"%{search}%"
            stmt = stmt.where(
                or_(
                    self.guests.c.first_name.ilike(pattern),
                    self.guests.c.last_name.ilike(pattern),
                    self.guests.c.email.ilike(pattern),
                )
            )
        stmt = stmt.limit(limit).offset(offset)
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(stmt).mappings().all()
        except SQLAlchemyError as exc:
            raise GuestStoreUnavailableError("Guest list failed") from exc
        return [dict(r) for r in rows]

    def count_guests(self, search: str | None = None) -> int:
        """Return total guest count (for pagination)."""
        stmt = select(func.count()).select_from(self.guests)
        if search:
            pattern = f"%{search}%"
            stmt = stmt.where(
                or_(
                    self.guests.c.first_name.ilike(pattern),
                    self.guests.c.last_name.ilike(pattern),
                    self.guests.c.email.ilike(pattern),
                )
            )
        try:
            with self.engine.connect() as conn:
                return conn.execute(stmt).scalar() or 0
        except SQLAlchemyError as exc:
            raise GuestStoreUnavailableError("Guest count failed") from exc

    def get_guest(self, guest_id: int) -> Dict[str, Any] | None:
        """Return a single guest by id, or None."""
        try:
            with self.engine.connect() as conn:
                row = (
                    conn.execute(
                        select(self.guests).where(self.guests.c.id == guest_id)
                    )
                    .mappings()
                    .first()
                )
        except SQLAlchemyError as exc:
            raise GuestStoreUnavailableError("Guest get failed") from exc
        return dict(row) if row else None

    def update_guest(
        self, guest_id: int, data: Dict[str, Any]
    ) -> Dict[str, Any] | None:
        """Partial update. Returns updated row dict, or None if not found."""
        now = datetime.now(timezone.utc)
        update_data = {**data, "updated_at": now}
        try:
            with self.engine.begin() as conn:
                stmt = (
                    update(self.guests)
                    .where(self.guests.c.id == guest_id)
                    .values(**update_data)
                )
                if getattr(conn.dialect, "update_returning", False):
                    result = conn.execute(stmt.returning(self.guests))
                    row = result.mappings().first()
                else:
                    result = conn.execute(stmt)
                    if result.rowcount == 0:
                        return None
                    row = (
                        conn.execute(
                            select(self.guests).where(self.guests.c.id == guest_id)
                        )
                        .mappings()
                        .first()
                    )
        except SQLAlchemyError as exc:
            raise GuestStoreUnavailableError("Guest update failed") from exc
        return dict(row) if row else None

    def delete_guest(self, guest_id: int) -> bool:
        """Delete a guest by id. Returns True if deleted, False if not found."""
        from sqlalchemy import delete as sa_delete

        try:
            with self.engine.begin() as conn:
                result = conn.execute(
                    sa_delete(self.guests).where(self.guests.c.id == guest_id)
                )
        except SQLAlchemyError as exc:
            raise GuestStoreUnavailableError("Guest delete failed") from exc
        return result.rowcount > 0


# ── Module-level singleton ────────────────────────────────────────────────────


def init_guest_store(engine: Engine) -> None:
    """Initialize the global GuestStore with an existing SQLAlchemy engine."""
    global _guest_store
    _guest_store = GuestStore(engine)
    logger.info("GuestStore initialized")


def get_guest_store() -> GuestStore:
    if _guest_store is None:
        raise GuestStoreNotInitializedError(
            "GuestStore is not initialized. Call init_guest_store() first."
        )
    return _guest_store
