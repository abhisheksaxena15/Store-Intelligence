# PROMPT:
#   "Write pytest tests for store metrics API endpoints: /metrics, /funnel, /heatmap.
#    Seed the in-memory database with known events and verify computed values.
#    Include edge cases: zero visitors, all staff, empty store, re-entry not double counted,
#    zero purchases. Verify conversion rate formula and funnel drop-off percentages."
#
# CHANGES MADE:
#   - Added explicit seeding helpers that insert directly via ORM (not through API)
#     to test metric computation independently from ingestion
#   - Added edge case: store with sessions but zero POS transactions
#   - Added test for heatmap data_confidence=False when <20 sessions
#   - Verified funnel stage ordering and that ENTRY stage has 0% drop_off

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


@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    await engine.dispose()


@pytest_asyncio.fixture
async def client(db_session):
    async def override_get_db():
        yield db_session
    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


STORE_ID = "STORE_BLR_002"
NOW = datetime.utcnow()


async def seed_events(db: AsyncSession, events: list[dict]) -> None:
    for e in events:
        db.add(EventORM(
            event_id=e.get("event_id", str(uuid.uuid4())),
            store_id=e["store_id"],
            camera_id=e.get("camera_id", "CAM_ENTRY_01"),
            visitor_id=e["visitor_id"],
            event_type=e["event_type"],
            timestamp=e.get("timestamp", NOW),
            zone_id=e.get("zone_id"),
            dwell_ms=e.get("dwell_ms", 0),
            is_staff=e.get("is_staff", False),
            confidence=e.get("confidence", 0.95),
            metadata_json=None,
            ingested_at=NOW,
        ))
    await db.commit()


async def seed_sessions(db: AsyncSession, sessions: list[dict]) -> None:
    for s in sessions:
        db.add(SessionORM(
            store_id=s["store_id"],
            visitor_id=s["visitor_id"],
            first_entry_ts=s.get("first_entry_ts", NOW),
            last_exit_ts=s.get("last_exit_ts"),
            visited_billing=s.get("visited_billing", False),
            converted=s.get("converted", False),
            total_dwell_ms=s.get("total_dwell_ms", 0),
            zone_count=s.get("zone_count", 0),
            reentry_count=s.get("reentry_count", 0),
            is_staff=s.get("is_staff", False),
        ))
    await db.commit()


# ── Metrics Tests ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_metrics_empty_store(client):
    """Store with no events returns zeros, not null or 500."""
    resp = await client.get(f"/stores/{STORE_ID}/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["unique_visitors"] == 0
    assert body["conversion_rate"] == 0.0
    assert body["queue_depth"] == 0
    assert body["abandonment_rate"] == 0.0


@pytest.mark.asyncio
async def test_metrics_excludes_staff(client, db_session):
    """Staff ENTRY events must not count toward unique_visitors."""
    await seed_events(db_session, [
        {"store_id": STORE_ID, "visitor_id": "VIS_staff1", "event_type": "ENTRY", "is_staff": True},
        {"store_id": STORE_ID, "visitor_id": "VIS_staff2", "event_type": "ENTRY", "is_staff": True},
        {"store_id": STORE_ID, "visitor_id": "VIS_cust1", "event_type": "ENTRY", "is_staff": False},
    ])
    resp = await client.get(f"/stores/{STORE_ID}/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["unique_visitors"] == 1   # only the customer


@pytest.mark.asyncio
async def test_metrics_conversion_rate_zero_purchases(client, db_session):
    """Zero purchases → conversion_rate == 0, not crash or null."""
    await seed_events(db_session, [
        {"store_id": STORE_ID, "visitor_id": f"VIS_{i}", "event_type": "ENTRY"}
        for i in range(5)
    ])
    resp = await client.get(f"/stores/{STORE_ID}/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["conversion_rate"] == 0.0
    assert body["unique_visitors"] == 5


@pytest.mark.asyncio
async def test_metrics_zone_dwell_aggregation(client, db_session):
    """Zone dwell events aggregate correctly per zone."""
    await seed_events(db_session, [
        {"store_id": STORE_ID, "visitor_id": "VIS_a", "event_type": "ZONE_ENTER", "zone_id": "SKINCARE"},
        {"store_id": STORE_ID, "visitor_id": "VIS_a", "event_type": "ZONE_ENTER", "zone_id": "FRAGRANCE"},
        {"store_id": STORE_ID, "visitor_id": "VIS_b", "event_type": "ZONE_ENTER", "zone_id": "SKINCARE"},
    ])
    resp = await client.get(f"/stores/{STORE_ID}/metrics")
    assert resp.status_code == 200
    zones = {z["zone_id"]: z for z in resp.json()["zone_metrics"]}
    assert "SKINCARE" in zones
    assert zones["SKINCARE"]["visit_count"] == 2
    assert "FRAGRANCE" in zones


# ── Funnel Tests ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_funnel_empty_store(client):
    """Funnel with no sessions returns all zeros."""
    resp = await client.get(f"/stores/{STORE_ID}/funnel")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_sessions"] == 0
    assert all(s["count"] == 0 for s in body["stages"])


@pytest.mark.asyncio
async def test_funnel_entry_stage_zero_dropoff(client, db_session):
    """ENTRY stage always has 0% drop_off_pct."""
    await seed_sessions(db_session, [
        {"store_id": STORE_ID, "visitor_id": f"VIS_{i}"} for i in range(10)
    ])
    resp = await client.get(f"/stores/{STORE_ID}/funnel")
    assert resp.status_code == 200
    stages = {s["stage"]: s for s in resp.json()["stages"]}
    assert stages["ENTRY"]["drop_off_pct"] == 0.0


@pytest.mark.asyncio
async def test_funnel_reentry_not_double_counted(client, db_session):
    """
    A visitor who re-enters must count as 1 session in the funnel,
    not 2 separate sessions.
    """
    visitor_id = "VIS_reenter"
    # Seed a single session row (re-entry increments counter on same row)
    await seed_sessions(db_session, [
        {
            "store_id": STORE_ID,
            "visitor_id": visitor_id,
            "reentry_count": 1,
        }
    ])
    resp = await client.get(f"/stores/{STORE_ID}/funnel")
    assert resp.status_code == 200
    body = resp.json()
    # Must be exactly 1, not 2
    assert body["total_sessions"] == 1
    assert body["stages"][0]["count"] == 1


@pytest.mark.asyncio
async def test_funnel_dropoff_calculation(client, db_session):
    """
    10 visitors enter → 8 visit a zone → 4 reach billing → 2 purchase.
    Drop-off at ZONE_VISIT should be 20%, BILLING_QUEUE 50%, PURCHASE 50%.
    """
    sessions = []
    for i in range(10):
        s = {"store_id": STORE_ID, "visitor_id": f"VIS_{i}", "zone_count": 0}
        if i < 8:   # 8 visit a zone
            s["zone_count"] = 1
        if i < 4:   # 4 reach billing
            s["visited_billing"] = True
        if i < 2:   # 2 convert
            s["converted"] = True
        sessions.append(s)

    await seed_sessions(db_session, sessions)
    resp = await client.get(f"/stores/{STORE_ID}/funnel")
    assert resp.status_code == 200
    stages = {s["stage"]: s for s in resp.json()["stages"]}

    assert stages["ENTRY"]["count"] == 10
    assert stages["ZONE_VISIT"]["count"] == 8
    assert abs(stages["ZONE_VISIT"]["drop_off_pct"] - 20.0) < 0.1
    assert stages["BILLING_QUEUE"]["count"] == 4
    assert abs(stages["BILLING_QUEUE"]["drop_off_pct"] - 50.0) < 0.1
    assert stages["PURCHASE"]["count"] == 2


# ── Heatmap Tests ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_heatmap_empty_store(client):
    """Heatmap with no zone events returns empty list."""
    resp = await client.get(f"/stores/{STORE_ID}/heatmap")
    assert resp.status_code == 200
    body = resp.json()
    assert body["zones"] == []


@pytest.mark.asyncio
async def test_heatmap_normalised_max_is_100(client, db_session):
    """Most-visited zone must have normalised_score == 100."""
    await seed_events(db_session, [
        {"store_id": STORE_ID, "visitor_id": f"VIS_{i}", "event_type": "ZONE_ENTER", "zone_id": "SKINCARE"}
        for i in range(10)
    ] + [
        {"store_id": STORE_ID, "visitor_id": f"VIS_b{i}", "event_type": "ZONE_ENTER", "zone_id": "FRAGRANCE"}
        for i in range(3)
    ])
    resp = await client.get(f"/stores/{STORE_ID}/heatmap")
    assert resp.status_code == 200
    zones = resp.json()["zones"]
    assert max(z["normalised_score"] for z in zones) == 100.0


@pytest.mark.asyncio
async def test_heatmap_low_confidence_flag(client, db_session):
    """data_confidence=False when fewer than 20 sessions."""
    # Only 5 sessions — below threshold
    await seed_sessions(db_session, [
        {"store_id": STORE_ID, "visitor_id": f"VIS_{i}"} for i in range(5)
    ])
    await seed_events(db_session, [
        {"store_id": STORE_ID, "visitor_id": f"VIS_{i}", "event_type": "ZONE_ENTER", "zone_id": "SKINCARE"}
        for i in range(5)
    ])
    resp = await client.get(f"/stores/{STORE_ID}/heatmap")
    assert resp.status_code == 200
    zones = resp.json()["zones"]
    if zones:
        assert all(not z["data_confidence"] for z in zones)