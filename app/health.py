"""
app/health.py
=============
Health check service.

Returns:
  - DB connectivity
  - Per-store last event timestamp + lag
  - STALE_FEED warning if any store has >10 min lag
  - API uptime

This is the "on-call engineer first check" endpoint — it must be accurate
and fast (no expensive queries).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.orm_models import EventORM
from app.models import HealthResponse, StoreFeedStatus

# API startup time (set at import time)
_START_TIME = time.time()

STALE_FEED_THRESHOLD_SECONDS = 600   # 10 minutes


async def check_health(db: AsyncSession) -> HealthResponse:
    now = datetime.utcnow()
    uptime = time.time() - _START_TIME

    # ── DB connectivity ─────────────────────────────────────────────────────
    db_ok = True
    try:
        await db.execute(select(func.count()).select_from(EventORM).limit(1))
    except SQLAlchemyError:
        db_ok = False

    # ── Per-store last event timestamps ──────────────────────────────────────
    store_statuses: list[StoreFeedStatus] = []
    overall_healthy = db_ok

    if db_ok:
        result = await db.execute(
            select(EventORM.store_id, func.max(EventORM.timestamp).label("last_ts"))
            .group_by(EventORM.store_id)
        )
        rows = result.fetchall()

        for row in rows:
            last_ts: datetime = row.last_ts
            lag_seconds = (now - last_ts).total_seconds() if last_ts else None
            status = "OK"
            if lag_seconds is None:
                status = "NO_DATA"
            elif lag_seconds > STALE_FEED_THRESHOLD_SECONDS:
                status = "STALE_FEED"
                overall_healthy = False

            store_statuses.append(StoreFeedStatus(
                store_id=row.store_id,
                last_event_ts=last_ts,
                lag_seconds=round(lag_seconds, 1) if lag_seconds is not None else None,
                status=status,
            ))

    return HealthResponse(
        status="healthy" if overall_healthy else "degraded",
        db_status="ok" if db_ok else "unavailable",
        as_of=now,
        stores=store_statuses,
        uptime_seconds=round(uptime, 1),
    )