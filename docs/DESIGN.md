# DESIGN.md — Store Intelligence System

## Overview

Store Intelligence converts raw CCTV footage from Apex Retail stores into a live analytics API. The system answers a single north-star question: **what is the offline conversion rate, and why?**

The architecture follows a linear pipeline with clean stage boundaries:

```
Raw CCTV Clips
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Detection Layer  (pipeline/)                           │
│                                                         │
│  YOLOv8 (per frame)                                     │
│      │                                                  │
│      ▼                                                  │
│  ByteTrack (cross-frame association)                    │
│      │                                                  │
│      ▼                                                  │
│  ReIDManager (OSNet cross-camera / re-entry matching)   │
│      │                                                  │
│      ▼                                                  │
│  StaffClassifier (HSV hue + permanence heuristic)       │
│      │                                                  │
│      ▼                                                  │
│  ZoneClassifier (polygon hit-test from store_layout)    │
│      │                                                  │
│      ▼                                                  │
│  DirectionDetector (entry/exit threshold crossing)      │
│      │                                                  │
│      ▼                                                  │
│  EventGenerator (business-logic event emission)         │
│      │                                                  │
│      ▼                                                  │
│  EventEmitter (schema validation + JSONL write)         │
└─────────────────────────────────────────────────────────┘
    │
    │  JSONL event files
    ▼
┌─────────────────────────────────────────────────────────┐
│  Intelligence API  (app/)                               │
│                                                         │
│  POST /events/ingest                                    │
│      │                                                  │
│      ├─► Pydantic validation + deduplication            │
│      ├─► events table (append-only ledger)              │
│      └─► sessions table (upserted aggregate)            │
│                                                         │
│  GET /stores/{id}/metrics   ──► EventORM + SessionORM   │
│  GET /stores/{id}/funnel    ──► SessionORM              │
│  GET /stores/{id}/heatmap   ──► EventORM                │
│  GET /stores/{id}/anomalies ──► EventORM + AnomalyORM   │
│  GET /health                ──► EventORM                │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Live Dashboard  (dashboard/)                           │
│  Streamlit — polls API every 10s                        │
│  Shows: KPIs, funnel, heatmap, anomalies                │
└─────────────────────────────────────────────────────────┘
```

---

## Component Details

### Detection Layer

**YOLOv8n + ByteTrack**
- YOLOv8 runs on every Nth frame (configurable stride, default=2 for speed)
- ByteTrack is invoked via `model.track(persist=True)` — this keeps track IDs stable across frames within a single camera clip
- Class filter: only COCO class 0 (person)

**ReIDManager**
- Primary strategy: OSNet x0_25 appearance embeddings compared via cosine similarity (threshold 0.72)
- Fallback: IoU-based bounding box matching when torchreid is unavailable (CPU/memory constrained deployments)
- Gallery TTL: 5 minutes — visitors who exit and re-enter within this window are matched as re-entries rather than new visitors

**StaffClassifier**
- HSV hue cluster analysis of the upper-body crop (top 40% of bounding box)
- Staff hue ranges are configurable per deployment (default: navy 100–130, burgundy 160–175)
- Permanence fallback: tracks present in >60% of clip frames are classified as staff regardless of colour (catches staff with unusual uniforms)

**ZoneClassifier**
- Reads zone polygon definitions from store_layout.json
- Uses cv2.pointPolygonTest for robust point-in-polygon checking
- Polygons are defined as normalised [0,1] fractions and scaled to pixel coords at startup

**EventGenerator**
- Stateful per-track event logic
- ZONE_DWELL emitted every 30 seconds of continuous zone occupancy
- BILLING_QUEUE_ABANDON determined via POS CSV correlation: if no transaction follows within 5 minutes of billing zone exit, it's an abandonment
- flush_all() called at clip end to close open sessions

---

### Intelligence API

**FastAPI + SQLAlchemy async (aiosqlite)**

All DB operations are non-blocking via async SQLAlchemy. The SQLite WAL mode enables concurrent reads during writes.

**Ingestion idempotency**
- Every event has a UUID v4 `event_id`
- On ingest, existing IDs are bulk-checked; duplicates are acknowledged but not re-inserted
- The response returns `duplicate_ids` so callers can distinguish "accepted" from "already exists"

**Session model**
- `sessions` table stores one row per (store_id, visitor_id)
- Re-entry events increment `reentry_count` on the existing row — they do NOT create a new session
- This is the deduplication mechanism that prevents re-entry inflation in funnel/metrics

**Structured logging**
- Every request logs: `trace_id`, `store_id`, `endpoint`, `latency_ms`, `event_count` (ingest only), `status_code`
- Uses structlog with JSON renderer — suitable for ingestion by any log aggregator (Datadog, CloudWatch, Loki)

---

### Database Schema

```
events (append-only ledger)
  id, event_id (UNIQUE), store_id, camera_id, visitor_id,
  event_type, timestamp, zone_id, dwell_ms, is_staff,
  confidence, metadata_json, ingested_at

  Indexes:
    ix_events_store_ts     (store_id, timestamp)   ← range queries
    ix_events_store_visitor (store_id, visitor_id) ← session lookup
    ix_events_store_type   (store_id, event_type)  ← funnel/anomaly queries
    ix_events_store_zone   (store_id, zone_id)      ← heatmap

sessions (one row per visitor visit)
  id, store_id, visitor_id (UNIQUE per store),
  first_entry_ts, last_exit_ts, visited_billing, converted,
  total_dwell_ms, zone_count, reentry_count, is_staff

anomalies (active operational alerts)
  id, anomaly_id (UNIQUE), store_id, anomaly_type, severity,
  description, suggested_action, detected_at, resolved_at,
  zone_id, metric_value, threshold_value
```

---

## Data Flow

```
1. pipeline/run.sh processes each clip sequentially
2. detect.py: frame → YOLOv8 → ByteTrack → ReID → Zone → EventGenerator → JSONL
3. ingest_events.py: JSONL → batches of 500 → POST /events/ingest
4. API ingestion: Pydantic validation → dedup → DB insert → session upsert
5. Dashboard: polls /metrics /funnel /heatmap /anomalies every 10s
```

---

## AI-Assisted Decisions

### 1. Re-ID Strategy: OSNet vs Simple IoU

I used Claude to evaluate the trade-off between appearance-based Re-ID (torchreid OSNet) and trajectory/IoU-based matching.

**What AI suggested:** Use a full market-1501 pretrained OSNet model with a gallery of 512-d embeddings and cosine similarity matching. It also suggested using a Kalman filter for motion prediction to handle partial occlusions.

**What I actually did:** I implemented the OSNet approach as primary but added a graceful fallback to IoU-based matching for CPU-only environments. I skipped the Kalman filter — the ByteTrack tracker already handles short-term occlusion, and Kalman on top would add latency without clear benefit at 15fps.

**Why I deviated:** The challenge dataset uses a fixed 20-minute clip — not a live stream. Kalman filter state diverges in pre-recorded footage with non-causal frame access. IoU fallback keeps the system deployable on machines without GPU.

### 2. Staff Detection: VLM vs Heuristic

I asked Claude Vision (via API in a test prompt) to classify a staff member from a sample retail frame. It correctly identified the uniform colour and context ("dark polo shirt typical of retail staff") but required a 2-3 second API call per crop.

**What AI suggested:** Use a VLM (GPT-4V or Claude Vision) for staff detection — more accurate on unusual uniform colours.

**What I chose:** HSV hue-cluster heuristic + permanence fallback. At 15fps with frame stride=2, there are ~450 person crops per minute per camera. VLM API calls would cost ~$0.01/crop × 450 = $4.50/minute. For a 20-minute clip that's $90 per camera — impractical.

**For production:** I would use the VLM offline (batch post-processing) to label a training set and fine-tune a lightweight binary classifier. The VLM is useful as a labelling tool, not a production inference engine.

### 3. Anomaly Thresholds: Statistical vs Fixed

I prompted an LLM to derive statistically justified thresholds for queue spike detection using a Z-score approach (flag when current > mean + 2σ).

**What AI suggested:** Use Z-score normalisation: alert when queue depth exceeds μ + 2σ of the 7-day rolling distribution.

**Why I overrode it:** The Z-score approach requires sufficient historical data to compute a stable standard deviation. For new stores (day 1) or stores with very low traffic (σ ≈ 0), the Z-score is either undefined or infinitely sensitive. I added absolute thresholds (WARN at 5, CRITICAL at 10) as floor conditions that fire even without history.

---

## Scaling Considerations

**Current bottleneck at 40 stores real-time:**
The /funnel and /metrics endpoints run multi-join queries on the events table. At 40 stores × ~500 events/hour × 8 hours = ~160K events/day, SQLite handles this without issue.

At production scale (40 stores, real-time, 15fps):
- The events table grows at ~2M rows/day. SQLite WAL mode handles reads fine but writes contend at ~100 events/second.
- Solution: Replace SQLite with TimescaleDB (PostgreSQL + time-series extension). The `events` table becomes a hypertable partitioned by timestamp.
- The session upsert pattern (current: row-by-row) would be replaced by a materialized view refreshed every 30 seconds.
- The anomaly detection would move to a background task (APScheduler or Celery beat) rather than running on every API call.

**Cross-camera deduplication:**
Currently, ReIDManager is per-camera. For cross-camera deduplication (entry camera + floor camera seeing the same person), a shared visitor gallery keyed by store_id is needed. This is architecturally supported — the gallery already groups by store_id — but the cosine threshold may need tuning per camera-pair since appearance changes across camera angles.

---

## Trade-offs

| Decision | Chosen | Alternative | Reason |
|----------|--------|-------------|--------|
| DB | SQLite + WAL | PostgreSQL | No extra container; trivially switched |
| Tracking | ByteTrack (built-in) | StrongSORT | Less complexity; ByteTrack excellent at 15fps |
| Re-ID | OSNet x0_25 + IoU fallback | Full OSNet-x1_0 | CPU deployable; x0_25 sufficient for body-only |
| Staff detection | HSV heuristic | VLM API | Cost and latency prohibitive for per-frame use |
| Dashboard | Streamlit | React | 10x faster to ship; adequate for demo |
| Event storage | JSONL → ingest | Kafka topic | No Kafka infra needed; JSONL is inspectable |