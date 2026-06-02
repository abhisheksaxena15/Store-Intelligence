"""
app/db/orm_models.py
====================
SQLAlchemy ORM table definitions.

Schema design:
  - events: raw event ledger; append-only; indexed for common queries
  - sessions: one row per visitor session (aggregated from events)
  - anomalies: active anomaly records
  - store_metrics_cache: rolling hourly aggregate cache (optional optimisation)

Indexes cover:
  - (store_id, timestamp): primary time-range query pattern
  - (store_id, visitor_id): session deduplication
  - event_id: idempotency lookup
  - (store_id, event_type): anomaly and funnel queries
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class EventORM(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    store_id: Mapped[str] = mapped_column(String(64), nullable=False)
    camera_id: Mapped[str] = mapped_column(String(64), nullable=False)
    visitor_id: Mapped[str] = mapped_column(String(64), nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    zone_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dwell_ms: Mapped[int] = mapped_column(Integer, default=0)
    is_staff: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    # metadata stored as JSON string to avoid schema migration overhead
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("event_id", name="uq_event_id"),
        Index("ix_events_store_ts", "store_id", "timestamp"),
        Index("ix_events_store_visitor", "store_id", "visitor_id"),
        Index("ix_events_store_type", "store_id", "event_type"),
        Index("ix_events_store_zone", "store_id", "zone_id"),
    )


class SessionORM(Base):
    """
    Aggregated visitor session.

    One row per (store_id, visitor_id) pair.
    Updated incrementally as events arrive.

    Re-entries: a visitor who exits and re-enters gets a REENTRY event
    but their session row is updated (not duplicated). This prevents
    double-counting in /funnel.
    """
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    store_id: Mapped[str] = mapped_column(String(64), nullable=False)
    visitor_id: Mapped[str] = mapped_column(String(64), nullable=False)
    first_entry_ts: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_exit_ts: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    visited_billing: Mapped[bool] = mapped_column(Boolean, default=False)
    converted: Mapped[bool] = mapped_column(Boolean, default=False)
    total_dwell_ms: Mapped[int] = mapped_column(Integer, default=0)
    zone_count: Mapped[int] = mapped_column(Integer, default=0)
    reentry_count: Mapped[int] = mapped_column(Integer, default=0)
    is_staff: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("store_id", "visitor_id", name="uq_session"),
        Index("ix_sessions_store", "store_id"),
    )


class AnomalyORM(Base):
    __tablename__ = "anomalies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    anomaly_id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True)
    store_id: Mapped[str] = mapped_column(String(64), nullable=False)
    anomaly_type: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    suggested_action: Mapped[str] = mapped_column(Text, nullable=False)
    detected_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    zone_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    threshold_value: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (
        Index("ix_anomalies_store_active", "store_id", "resolved_at"),
    )