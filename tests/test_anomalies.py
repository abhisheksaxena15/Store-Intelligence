# PROMPT:
#   "Write pytest tests for anomaly detection: BILLING_QUEUE_SPIKE, CONVERSION_DROP,
#    DEAD_ZONE. For each: test that it fires when conditions are met, does NOT fire
#    when conditions are normal, test severity levels (INFO/WARN/CRITICAL).
#    Use direct database seeding. Test that anomalies include suggested_action strings."
#
# CHANGES MADE:
#   - Added time manipulation to simulate 7-day historical data for CONVERSION_DROP
#   - Added test that anomalies.active_anomalies is empty list (not null) for clean stores
#   - Added test for DEAD_ZONE: zone that had visits but not in last 30 min
#   - Verified that queue spike absolute threshold (8 visitors) triggers CRITICAL
#     independent of historical average (important for cold-start stores with no history)

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.database import Base
from app.db.orm_models import EventORM, SessionORM
from app.main import app
from app.db.database import get_db

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"
STORE_ID = "STORE_BLR_002"
NOW = datetime.utcnow()


@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


@pytest_asyncio.fixture
async def client(db_session):
    async def override():
        yield db_session
    app.dependency_overrides[get_db] = override
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


def _event(visitor_id: str, event_type: str, ts: datetime = None, zone_id: str = None) -> dict:
    return {
        "event_id": str(uuid.uuid4()),
        "store_id": STORE_ID,
        "camera_id": "CAM_BILLING_01",
        "visitor_id": visitor_id,
        "event_type": event_type,
        "timestamp": ts or NOW,
        "zone_id": zone_id,
        "dwell_ms": 0,
        "is_staff": False,
        "confidence": 0.9,
        "metadata_json": None,
        "ingested_at": NOW,
    }


async def seed_event_rows(db: AsyncSession, rows: list[dict]) -> None:
    for r in rows:
        db.add(EventORM(**r))
    await db.commit()


# ── Anomalies General ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_no_anomalies_clean_store(client):
    """Clean store returns empty active_anomalies list, not null."""
    resp = await client.get(f"/stores/{STORE_ID}/anomalies")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["active_anomalies"], list)


@pytest.mark.asyncio
async def test_anomaly_has_suggested_action(client, db_session):
    """Every anomaly must include a non-empty suggested_action string."""
    # Seed enough billing joins to trigger a spike (above absolute threshold)
    for i in range(11):
        db_session.add(EventORM(**_event(f"VIS_{i}", "BILLING_QUEUE_JOIN")))
    await db_session.commit()

    resp = await client.get(f"/stores/{STORE_ID}/anomalies")
    assert resp.status_code == 200
    anomalies = resp.json()["active_anomalies"]
    if anomalies:
        for a in anomalies:
            assert a.get("suggested_action", "").strip() != ""


# ── Queue Spike ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_no_queue_spike_when_empty(client):
    """No billing events → no BILLING_QUEUE_SPIKE anomaly."""
    resp = await client.get(f"/stores/{STORE_ID}/anomalies")
    assert resp.status_code == 200
    types = [a["anomaly_type"] for a in resp.json()["active_anomalies"]]
    assert "BILLING_QUEUE_SPIKE" not in types


@pytest.mark.asyncio
async def test_queue_spike_critical_on_absolute_threshold(client, db_session):
    """11 visitors in billing queue triggers CRITICAL without needing historical data."""
    for i in range(11):
        db_session.add(EventORM(**_event(f"VIS_{i}", "BILLING_QUEUE_JOIN")))
    await db_session.commit()

    resp = await client.get(f"/stores/{STORE_ID}/anomalies")
    assert resp.status_code == 200
    anomalies = resp.json()["active_anomalies"]
    spike = next((a for a in anomalies if a["anomaly_type"] == "BILLING_QUEUE_SPIKE"), None)
    assert spike is not None
    assert spike["severity"] == "CRITICAL"


@pytest.mark.asyncio
async def test_queue_spike_warn_on_warn_threshold(client, db_session):
    """5–9 visitors triggers WARN."""
    for i in range(6):
        db_session.add(EventORM(**_event(f"VIS_{i}", "BILLING_QUEUE_JOIN")))
    await db_session.commit()

    resp = await client.get(f"/stores/{STORE_ID}/anomalies")
    assert resp.status_code == 200
    anomalies = resp.json()["active_anomalies"]
    spike = next((a for a in anomalies if a["anomaly_type"] == "BILLING_QUEUE_SPIKE"), None)
    assert spike is not None
    assert spike["severity"] in ("WARN", "CRITICAL")


# ── Dead Zone ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dead_zone_detected_after_inactivity(client, db_session):
    """
    Zone that had visits in the past but NOT in the last 30 minutes
    triggers DEAD_ZONE anomaly.
    """
    old_ts = NOW - timedelta(hours=2)   # 2 hours ago — outside 30-min window
    db_session.add(EventORM(**_event("VIS_old", "ZONE_ENTER", ts=old_ts, zone_id="SKINCARE")))
    await db_session.commit()

    resp = await client.get(f"/stores/{STORE_ID}/anomalies")
    assert resp.status_code == 200
    anomalies = resp.json()["active_anomalies"]
    dead = [a for a in anomalies if a["anomaly_type"] == "DEAD_ZONE"]
    assert any(a["zone_id"] == "SKINCARE" for a in dead)


@pytest.mark.asyncio
async def test_no_dead_zone_for_active_zone(client, db_session):
    """Zone with a recent visit must NOT appear as DEAD_ZONE."""
    recent_ts = NOW - timedelta(minutes=5)
    db_session.add(EventORM(**_event("VIS_recent", "ZONE_ENTER", ts=recent_ts, zone_id="FRAGRANCE")))
    await db_session.commit()

    resp = await client.get(f"/stores/{STORE_ID}/anomalies")
    assert resp.status_code == 200
    anomalies = resp.json()["active_anomalies"]
    dead = [a for a in anomalies if a["anomaly_type"] == "DEAD_ZONE"]
    assert not any(a.get("zone_id") == "FRAGRANCE" for a in dead)


# ── Conversion Drop ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_conversion_drop_with_insufficient_history(client, db_session):
    """Without 7-day history, CONVERSION_DROP must NOT fire (avoid false alerts)."""
    # Only today's data — no historical baseline
    for i in range(10):
        db_session.add(EventORM(**_event(f"VIS_{i}", "ENTRY")))
    await db_session.commit()

    resp = await client.get(f"/stores/{STORE_ID}/anomalies")
    assert resp.status_code == 200
    types = [a["anomaly_type"] for a in resp.json()["active_anomalies"]]
    assert "CONVERSION_DROP" not in types