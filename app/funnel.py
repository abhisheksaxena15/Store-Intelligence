"""
app/funnel.py
=============
Funnel computation helpers — kept separate from metrics.py to allow
independent testing and future caching.

The funnel is session-based, not event-based:
  - Unit of measurement: a unique (store_id, visitor_id) session
  - Re-entry increments the session's reentry_count but does NOT create a new session
  - Staff sessions (is_staff=True) are excluded at every stage

Funnel stages in order:
  1. ENTRY         — visitor crossed entry threshold (has a session row)
  2. ZONE_VISIT    — session has zone_count > 0
  3. BILLING_QUEUE — session visited_billing == True
  4. PURCHASE      — session converted == True
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.orm_models import SessionORM
from app.models import FunnelResponse, FunnelStage


async def get_funnel(store_id: str, db: AsyncSession) -> FunnelResponse:
    """
    Returns the conversion funnel for the given store.
    Sessions are the unit — re-entries do not double-count.
    """
    now = datetime.utcnow()

    async def count_where(**kwargs) -> int:
        conditions = [
            SessionORM.store_id == store_id,
            SessionORM.is_staff == False,
        ]
        for attr, value in kwargs.items():
            conditions.append(getattr(SessionORM, attr) == value)
        result = await db.execute(select(func.count()).select_from(SessionORM).where(*conditions))
        return result.scalar() or 0

    total = await count_where()
    zone_visitors = await db.execute(
        select(func.count())
        .select_from(SessionORM)
        .where(
            SessionORM.store_id == store_id,
            SessionORM.is_staff == False,
            SessionORM.zone_count > 0,
        )
    )
    zone_count: int = zone_visitors.scalar() or 0

    billing_result = await count_where(visited_billing=True)
    converted_result = await count_where(converted=True)

    def pct(current: int, previous: int) -> float:
        if previous == 0:
            return 0.0
        return round(100.0 * (1.0 - current / previous), 2)

    stages = [
        FunnelStage(stage="ENTRY", count=total, drop_off_pct=0.0),
        FunnelStage(stage="ZONE_VISIT", count=zone_count, drop_off_pct=pct(zone_count, total)),
        FunnelStage(stage="BILLING_QUEUE", count=billing_result, drop_off_pct=pct(billing_result, zone_count)),
        FunnelStage(stage="PURCHASE", count=converted_result, drop_off_pct=pct(converted_result, billing_result)),
    ]

    return FunnelResponse(
        store_id=store_id,
        as_of=now,
        stages=stages,
        total_sessions=total,
    )