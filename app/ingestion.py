"""
app/ingestion.py
================
Event ingest service.

Responsibilities:
  - Validate each event (Pydantic handles schema)
  - Deduplicate by event_id (idempotent — same payload twice is safe)
  - Persist to events table
  - Update session aggregate table
  - Trigger anomaly detection after each batch
  - Return partial success response (accepted + rejected + errors)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.orm_models import EventORM, SessionORM
from app.models import EventError, EventType, IngestResponse, StoreEvent

logger = logging.getLogger(__name__)


async def ingest_events(
    events: list[StoreEvent],
    db: AsyncSession,
) -> IngestResponse:
    """
    Idempotently ingest a batch of events.

    Processing order:
      1. Bulk-check which event_ids already exist
      2. Insert new events
      3. Upsert session aggregates for affected visitors
    """
    if not events:
        return IngestResponse(accepted=0, rejected=0)

    # ── Step 1: Identify existing event_ids (for dedup reporting) ─────────────
    incoming_ids = [e.event_id for e in events]
    result = await db.execute(
        select(EventORM.event_id).where(EventORM.event_id.in_(incoming_ids))
    )
    existing_ids: set[str] = {row[0] for row in result.fetchall()}
    duplicate_ids = list(existing_ids)

    # ── Step 2: Insert new events ─────────────────────────────────────────────
    new_events = [e for e in events if e.event_id not in existing_ids]
    errors: list[EventError] = []
    accepted = 0

    rows_to_insert = []
    for idx, evt in enumerate(events):
        if evt.event_id in existing_ids:
            continue   # silently skip duplicates — idempotent
        try:
            rows_to_insert.append({
                "event_id": evt.event_id,
                "store_id": evt.store_id,
                "camera_id": evt.camera_id,
                "visitor_id": evt.visitor_id,
                "event_type": evt.event_type.value,
                "timestamp": evt.timestamp.replace(tzinfo=None),
                "zone_id": evt.zone_id,
                "dwell_ms": evt.dwell_ms,
                "is_staff": evt.is_staff,
                "confidence": evt.confidence,
                "metadata_json": json.dumps(evt.metadata.model_dump()),
                "ingested_at": datetime.utcnow(),
            })
            accepted += 1
        except Exception as e:
            errors.append(EventError(event_id=evt.event_id, index=idx, error=str(e)))

    if rows_to_insert:
        try:
            # Use INSERT OR IGNORE for SQLite (handles race conditions)
            stmt = sqlite_insert(EventORM).prefix_with("OR IGNORE").values(rows_to_insert)
            await db.execute(stmt)
        except Exception as e:
            logger.error(f"Bulk insert error: {e}")
            # Fall back to one-by-one
            for row in rows_to_insert:
                try:
                    db.add(EventORM(**row))
                    await db.flush()
                except IntegrityError:
                    await db.rollback()

    # ── Step 3: Upsert session aggregates ─────────────────────────────────────
    await _update_sessions(new_events, db)

    await db.commit()

    logger.info(
        "Ingest complete",
        extra={
            "accepted": accepted,
            "rejected": len(errors),
            "duplicates": len(duplicate_ids),
            "total_received": len(events),
        },
    )

    return IngestResponse(
        accepted=accepted,
        rejected=len(errors),
        errors=errors,
        duplicate_ids=duplicate_ids,
    )


async def _update_sessions(events: list[StoreEvent], db: AsyncSession) -> None:
    """
    Upsert session rows based on incoming events.

    Session logic:
      - ENTRY creates or updates session first_entry_ts
      - EXIT sets last_exit_ts
      - ZONE_ENTER on a billing zone sets visited_billing=True
      - BILLING_QUEUE_ABANDON does NOT set converted
      - REENTRY increments reentry_count (does NOT create a duplicate session)

    Conversion is NOT set here — it requires POS correlation which happens
    in the metrics computation layer (time-window based).
    """
    # Group by (store_id, visitor_id)
    sessions: dict[tuple[str, str], dict] = {}

    for evt in events:
        if evt.is_staff:
            continue   # staff sessions not tracked in customer metrics

        key = (evt.store_id, evt.visitor_id)
        if key not in sessions:
            sessions[key] = {
                "store_id": evt.store_id,
                "visitor_id": evt.visitor_id,
                "first_entry_ts": None,
                "last_exit_ts": None,
                "visited_billing": False,
                "reentry_delta": 0,
                "dwell_delta": 0,
                "zone_delta": 0,
                "is_staff": False,
            }

        s = sessions[key]
        ts = evt.timestamp.replace(tzinfo=None)

        if evt.event_type == EventType.ENTRY:
            if s["first_entry_ts"] is None or ts < s["first_entry_ts"]:
                s["first_entry_ts"] = ts

        elif evt.event_type == EventType.EXIT:
            if s["last_exit_ts"] is None or ts > s["last_exit_ts"]:
                s["last_exit_ts"] = ts

        elif evt.event_type in (EventType.ZONE_ENTER, EventType.ZONE_EXIT):
            if evt.zone_id and "BILLING" in evt.zone_id.upper():
                s["visited_billing"] = True
            s["zone_delta"] += 1

        elif evt.event_type == EventType.ZONE_DWELL:
            s["dwell_delta"] += evt.dwell_ms

        elif evt.event_type == EventType.REENTRY:
            s["reentry_delta"] += 1

    # Upsert each session
    for key, s in sessions.items():
        store_id, visitor_id = key

        result = await db.execute(
            select(SessionORM).where(
                SessionORM.store_id == store_id,
                SessionORM.visitor_id == visitor_id,
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            if s["first_entry_ts"] and (
                existing.first_entry_ts is None
                or s["first_entry_ts"] < existing.first_entry_ts
            ):
                existing.first_entry_ts = s["first_entry_ts"]
            if s["last_exit_ts"] and (
                existing.last_exit_ts is None
                or s["last_exit_ts"] > existing.last_exit_ts
            ):
                existing.last_exit_ts = s["last_exit_ts"]
            if s["visited_billing"]:
                existing.visited_billing = True
            existing.reentry_count += s["reentry_delta"]
            existing.total_dwell_ms += s["dwell_delta"]
            existing.zone_count += s["zone_delta"]
            existing.updated_at = datetime.utcnow()
        else:
            db.add(SessionORM(
                store_id=store_id,
                visitor_id=visitor_id,
                first_entry_ts=s["first_entry_ts"],
                last_exit_ts=s["last_exit_ts"],
                visited_billing=s["visited_billing"],
                converted=False,
                total_dwell_ms=s["dwell_delta"],
                zone_count=s["zone_delta"],
                reentry_count=s["reentry_delta"],
                is_staff=False,
                updated_at=datetime.utcnow(),
            ))