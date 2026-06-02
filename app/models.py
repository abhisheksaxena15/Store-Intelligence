"""
app/models.py
=============
Pydantic models for request/response validation and internal types.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


# ─── Event Schema ─────────────────────────────────────────────────────────────

class EventType(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    ZONE_DWELL = "ZONE_DWELL"
    BILLING_QUEUE_JOIN = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON = "BILLING_QUEUE_ABANDON"
    REENTRY = "REENTRY"


class EventMetadata(BaseModel):
    queue_depth: Optional[int] = None
    sku_zone: Optional[str] = None
    session_seq: int = 0

    model_config = {"extra": "allow"}


class StoreEvent(BaseModel):
    event_id: str = Field(..., description="UUID v4 — must be globally unique")
    store_id: str = Field(..., min_length=1)
    camera_id: str = Field(..., min_length=1)
    visitor_id: str = Field(..., min_length=1)
    event_type: EventType
    timestamp: datetime
    zone_id: Optional[str] = None
    dwell_ms: int = Field(default=0, ge=0)
    is_staff: bool = False
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: EventMetadata = Field(default_factory=EventMetadata)

    @field_validator("event_id")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        try:
            UUID(v)
        except ValueError:
            raise ValueError(f"event_id must be a valid UUID v4: {v!r}")
        return v

    @model_validator(mode="after")
    def validate_zone_for_type(self) -> "StoreEvent":
        zone_required = {EventType.ZONE_ENTER, EventType.ZONE_EXIT, EventType.ZONE_DWELL}
        if self.event_type in zone_required and not self.zone_id:
            raise ValueError(f"zone_id is required for event_type={self.event_type}")
        return self


class IngestRequest(BaseModel):
    events: list[StoreEvent] = Field(..., max_length=500)


class EventError(BaseModel):
    event_id: Optional[str] = None
    index: int
    error: str


class IngestResponse(BaseModel):
    accepted: int
    rejected: int
    errors: list[EventError] = []
    duplicate_ids: list[str] = []


# ─── Metrics Response ─────────────────────────────────────────────────────────

class ZoneDwellMetric(BaseModel):
    zone_id: str
    avg_dwell_seconds: float
    visit_count: int


class MetricsResponse(BaseModel):
    store_id: str
    as_of: datetime
    unique_visitors: int
    conversion_rate: float = Field(..., ge=0.0, le=1.0)
    avg_dwell_seconds: float
    zone_metrics: list[ZoneDwellMetric]
    queue_depth: int
    abandonment_rate: float = Field(..., ge=0.0, le=1.0)


# ─── Funnel Response ──────────────────────────────────────────────────────────

class FunnelStage(BaseModel):
    stage: str
    count: int
    drop_off_pct: float = Field(..., ge=0.0, le=100.0)


class FunnelResponse(BaseModel):
    store_id: str
    as_of: datetime
    stages: list[FunnelStage]
    total_sessions: int


# ─── Heatmap Response ─────────────────────────────────────────────────────────

class HeatmapZone(BaseModel):
    zone_id: str
    visit_count: int
    avg_dwell_seconds: float
    normalised_score: float = Field(..., ge=0.0, le=100.0)
    data_confidence: bool = Field(
        True,
        description="False if fewer than 20 sessions — data may be statistically unreliable"
    )


class HeatmapResponse(BaseModel):
    store_id: str
    as_of: datetime
    zones: list[HeatmapZone]


# ─── Anomaly Response ─────────────────────────────────────────────────────────

class AnomalySeverity(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


class AnomalyType(str, Enum):
    BILLING_QUEUE_SPIKE = "BILLING_QUEUE_SPIKE"
    CONVERSION_DROP = "CONVERSION_DROP"
    DEAD_ZONE = "DEAD_ZONE"
    STALE_FEED = "STALE_FEED"


class Anomaly(BaseModel):
    anomaly_id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    description: str
    suggested_action: str
    detected_at: datetime
    zone_id: Optional[str] = None
    metric_value: Optional[float] = None
    threshold_value: Optional[float] = None


class AnomaliesResponse(BaseModel):
    store_id: str
    as_of: datetime
    active_anomalies: list[Anomaly]


# ─── Health Response ──────────────────────────────────────────────────────────

class StoreFeedStatus(BaseModel):
    store_id: str
    last_event_ts: Optional[datetime]
    lag_seconds: Optional[float]
    status: str   # "OK" | "STALE_FEED" | "NO_DATA"


class HealthResponse(BaseModel):
    status: str   # "healthy" | "degraded"
    db_status: str
    as_of: datetime
    stores: list[StoreFeedStatus]
    uptime_seconds: float