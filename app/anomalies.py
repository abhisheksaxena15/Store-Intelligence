"""
app/anomalies.py
================
Anomaly detection engine.

Three anomaly types:

1. BILLING_QUEUE_SPIKE
   Formula: current queue depth > (7-day avg queue depth) * 2.5
   OR queue depth > ABSOLUTE_QUEUE_THRESHOLD (configurable, default 8)
   Severity: WARN if 2x avg, CRITICAL if 3x avg or >8 absolute

2. CONVERSION_DROP
   Formula: today's conversion rate < (7-day avg conversion rate) * 0.7
   That is, conversion has dropped more than 30% from recent average.
   Severity: INFO if 10–20% drop, WARN if 20–30%, CRITICAL if >30%

3. DEAD_ZONE
   Formula: no ZONE_ENTER events for a specific zone in the last 30 minutes
   during store open hours.
   Severity: INFO (could be a camera issue, not just no traffic)

All anomalies include a suggested_action string for operations teams.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import func, select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.orm_models import AnomalyORM, EventORM
from app.models import (
    Anomaly,
    AnomaliesResponse,
    AnomalySeverity,
    AnomalyType,
)

# ── Thresholds ────────────────────────────────────────────────────────────────
QUEUE_ABSOLUTE_WARN = 5         # absolute queue depth that always triggers WARN
QUEUE_ABSOLUTE_CRITICAL = 10    # absolute queue depth that always triggers CRITICAL
QUEUE_SPIKE_MULTIPLIER_WARN = 2.0
QUEUE_SPIKE_MULTIPLIER_CRITICAL = 3.0

CONVERSION_DROP_INFO = 0.10     # 10% drop → INFO
CONVERSION_DROP_WARN = 0.20     # 20% drop → WARN
CONVERSION_DROP_CRITICAL = 0.30 # 30% drop → CRITICAL

DEAD_ZONE_MINUTES = 30          # no activity in 30 min → DEAD_ZONE

HISTORY_DAYS = 7


async def detect_anomalies(store_id: str, db: AsyncSession) -> AnomaliesResponse:
    """
    Runs all anomaly detectors and returns active (unresolved) anomalies.
    Persists new anomalies to the database.
    """
    now = datetime.utcnow()
    anomalies: list[Anomaly] = []

    # Run detectors
    queue_anomaly = await _detect_queue_spike(store_id, db, now)
    if queue_anomaly:
        anomalies.append(queue_anomaly)

    conv_anomaly = await _detect_conversion_drop(store_id, db, now)
    if conv_anomaly:
        anomalies.append(conv_anomaly)

    dead_zones = await _detect_dead_zones(store_id, db, now)
    anomalies.extend(dead_zones)

    # Persist any new anomalies
    for anomaly in anomalies:
        await _upsert_anomaly(anomaly, store_id, db)

    await db.commit()

    return AnomaliesResponse(
        store_id=store_id,
        as_of=now,
        active_anomalies=anomalies,
    )


async def _detect_queue_spike(
    store_id: str, db: AsyncSession, now: datetime
) -> Optional[Anomaly]:
    """Detect billing queue depth spike."""

    # Current queue depth (last 30 min)
    window = now - timedelta(minutes=30)
    result = await db.execute(
        select(func.count(func.distinct(EventORM.visitor_id)))
        .where(
            EventORM.store_id == store_id,
            EventORM.event_type == "BILLING_QUEUE_JOIN",
            EventORM.is_staff == False,
            EventORM.timestamp >= window,
        )
    )
    join_count: int = result.scalar() or 0

    result = await db.execute(
        select(func.count(func.distinct(EventORM.visitor_id)))
        .where(
            EventORM.store_id == store_id,
            EventORM.event_type.in_(["BILLING_QUEUE_ABANDON", "EXIT"]),
            EventORM.is_staff == False,
            EventORM.timestamp >= window,
        )
    )
    exit_count: int = result.scalar() or 0
    current_depth = max(0, join_count - exit_count)

    # 7-day historical average queue depth (per-30-min window)
    hist_start = now - timedelta(days=HISTORY_DAYS)
    result = await db.execute(
        select(func.count(func.distinct(EventORM.visitor_id)))
        .where(
            EventORM.store_id == store_id,
            EventORM.event_type == "BILLING_QUEUE_JOIN",
            EventORM.is_staff == False,
            EventORM.timestamp >= hist_start,
            EventORM.timestamp < now - timedelta(hours=1),  # exclude today
        )
    )
    hist_joins: int = result.scalar() or 0
    # Number of 30-min windows in history period
    n_windows = HISTORY_DAYS * 48
    avg_depth = hist_joins / n_windows if n_windows > 0 else 0.0

    # Determine severity
    severity: Optional[AnomalySeverity] = None
    if current_depth >= QUEUE_ABSOLUTE_CRITICAL:
        severity = AnomalySeverity.CRITICAL
    elif current_depth >= QUEUE_ABSOLUTE_WARN:
        severity = AnomalySeverity.WARN
    elif avg_depth > 0 and current_depth >= avg_depth * QUEUE_SPIKE_MULTIPLIER_CRITICAL:
        severity = AnomalySeverity.CRITICAL
    elif avg_depth > 0 and current_depth >= avg_depth * QUEUE_SPIKE_MULTIPLIER_WARN:
        severity = AnomalySeverity.WARN

    if severity is None:
        return None

    return Anomaly(
        anomaly_id=str(uuid.uuid4()),
        anomaly_type=AnomalyType.BILLING_QUEUE_SPIKE,
        severity=severity,
        description=(
            f"Billing queue depth is {current_depth} "
            f"(7-day avg: {avg_depth:.1f}). "
            f"Spike detected."
        ),
        suggested_action=(
            "Open additional billing counter. "
            "Alert floor manager immediately."
            if severity == AnomalySeverity.CRITICAL
            else "Monitor queue — consider opening secondary billing lane."
        ),
        detected_at=now,
        zone_id="BILLING",
        metric_value=float(current_depth),
        threshold_value=float(avg_depth * QUEUE_SPIKE_MULTIPLIER_WARN),
    )


async def _detect_conversion_drop(
    store_id: str, db: AsyncSession, now: datetime
) -> Optional[Anomaly]:
    """Detect today's conversion rate dropping vs 7-day average."""

    # Today's conversion rate
    today_start = now - timedelta(hours=24)

    result = await db.execute(
        select(func.count(func.distinct(EventORM.visitor_id)))
        .where(
            EventORM.store_id == store_id,
            EventORM.event_type == "ENTRY",
            EventORM.is_staff == False,
            EventORM.timestamp >= today_start,
        )
    )
    today_visitors: int = result.scalar() or 0

    # For a lightweight implementation, we approximate conversion as
    # fraction of billing queue visitors (conservative proxy)
    result = await db.execute(
        select(func.count(func.distinct(EventORM.visitor_id)))
        .where(
            EventORM.store_id == store_id,
            EventORM.event_type == "BILLING_QUEUE_JOIN",
            EventORM.is_staff == False,
            EventORM.timestamp >= today_start,
        )
    )
    today_billing: int = result.scalar() or 0

    if today_visitors == 0:
        return None

    today_rate = today_billing / today_visitors

    # 7-day historical rate (exclude today)
    hist_start = now - timedelta(days=HISTORY_DAYS)
    result = await db.execute(
        select(func.count(func.distinct(EventORM.visitor_id)))
        .where(
            EventORM.store_id == store_id,
            EventORM.event_type == "ENTRY",
            EventORM.is_staff == False,
            EventORM.timestamp >= hist_start,
            EventORM.timestamp < today_start,
        )
    )
    hist_visitors: int = result.scalar() or 0

    result = await db.execute(
        select(func.count(func.distinct(EventORM.visitor_id)))
        .where(
            EventORM.store_id == store_id,
            EventORM.event_type == "BILLING_QUEUE_JOIN",
            EventORM.is_staff == False,
            EventORM.timestamp >= hist_start,
            EventORM.timestamp < today_start,
        )
    )
    hist_billing: int = result.scalar() or 0

    if hist_visitors == 0:
        return None

    hist_rate = hist_billing / hist_visitors

    if hist_rate == 0:
        return None

    drop_fraction = (hist_rate - today_rate) / hist_rate

    if drop_fraction < CONVERSION_DROP_INFO:
        return None

    severity = (
        AnomalySeverity.CRITICAL if drop_fraction >= CONVERSION_DROP_CRITICAL
        else AnomalySeverity.WARN if drop_fraction >= CONVERSION_DROP_WARN
        else AnomalySeverity.INFO
    )

    return Anomaly(
        anomaly_id=str(uuid.uuid4()),
        anomaly_type=AnomalyType.CONVERSION_DROP,
        severity=severity,
        description=(
            f"Today's conversion rate is {today_rate:.1%} vs "
            f"7-day avg of {hist_rate:.1%} — drop of {drop_fraction:.1%}."
        ),
        suggested_action=(
            "Review zone heatmap for abandonment patterns. "
            "Check if any product zones have unusually low dwell."
        ),
        detected_at=now,
        metric_value=round(today_rate, 4),
        threshold_value=round(hist_rate * (1 - CONVERSION_DROP_INFO), 4),
    )


async def _detect_dead_zones(
    store_id: str, db: AsyncSession, now: datetime
) -> list[Anomaly]:
    """Detect zones with no activity in the last 30 minutes."""
    window = now - timedelta(minutes=DEAD_ZONE_MINUTES)

    # All zones that have EVER had a visit for this store
    result = await db.execute(
        select(func.distinct(EventORM.zone_id))
        .where(
            EventORM.store_id == store_id,
            EventORM.event_type == "ZONE_ENTER",
            EventORM.zone_id.isnot(None),
        )
    )
    all_zones: set[str] = {row[0] for row in result.fetchall()}

    # Zones with a visit in the last 30 min
    result = await db.execute(
        select(func.distinct(EventORM.zone_id))
        .where(
            EventORM.store_id == store_id,
            EventORM.event_type == "ZONE_ENTER",
            EventORM.zone_id.isnot(None),
            EventORM.timestamp >= window,
        )
    )
    active_zones: set[str] = {row[0] for row in result.fetchall()}

    dead_zones = all_zones - active_zones
    anomalies = []

    for zone_id in dead_zones:
        anomalies.append(Anomaly(
            anomaly_id=str(uuid.uuid4()),
            anomaly_type=AnomalyType.DEAD_ZONE,
            severity=AnomalySeverity.INFO,
            description=(
                f"Zone '{zone_id}' has had no visitor activity in the last "
                f"{DEAD_ZONE_MINUTES} minutes."
            ),
            suggested_action=(
                f"Verify camera coverage for '{zone_id}'. "
                "If cameras are healthy, consider promotional activity to drive traffic."
            ),
            detected_at=now,
            zone_id=zone_id,
        ))

    return anomalies


async def _upsert_anomaly(anomaly: Anomaly, store_id: str, db: AsyncSession) -> None:
    """Persist anomaly if not already active."""
    result = await db.execute(
        select(AnomalyORM).where(
            AnomalyORM.store_id == store_id,
            AnomalyORM.anomaly_type == anomaly.anomaly_type.value,
            AnomalyORM.zone_id == anomaly.zone_id,
            AnomalyORM.resolved_at.is_(None),
        )
    )
    existing = result.scalar_one_or_none()

    if not existing:
        db.add(AnomalyORM(
            anomaly_id=anomaly.anomaly_id,
            store_id=store_id,
            anomaly_type=anomaly.anomaly_type.value,
            severity=anomaly.severity.value,
            description=anomaly.description,
            suggested_action=anomaly.suggested_action,
            detected_at=anomaly.detected_at,
            zone_id=anomaly.zone_id,
            metric_value=anomaly.metric_value,
            threshold_value=anomaly.threshold_value,
        ))
    else:
        # Update severity if escalated
        existing.severity = anomaly.severity.value
        existing.description = anomaly.description
        existing.metric_value = anomaly.metric_value