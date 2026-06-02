# CHOICES.md — Three Architectural Decisions

---

## Decision 1: Detection Model — YOLOv8n + ByteTrack

### The Problem
I needed a model that could:
- Run on CPU (challenge environment, no guaranteed GPU)
- Detect individuals within groups (not just blobs)
- Operate at useful speed on 1080p 15fps footage
- Handle partial occlusion gracefully (not fail silently)

### Options Considered

| Model | Pro | Con |
|-------|-----|-----|
| **YOLOv8n** | Fast on CPU, built-in ByteTrack integration, well-documented | Smaller than v8m — lower accuracy on partially occluded people |
| YOLOv8m | Higher mAP on COCO person class | ~3× slower on CPU; makes 48h time budget tight |
| YOLOv9 | Improved architecture | Smaller community; less integrated tooling at challenge time |
| RT-DETR | Transformer-based; excellent crowded-scene detection | Requires GPU for real-time; too slow on CPU for this clip length |
| MediaPipe Pose | Fast, no GPU | Designed for single-person tracking; breaks on group entry |

### What AI Suggested
I asked Claude to compare YOLOv8 vs RT-DETR for a retail detection scenario with groups and partial occlusion. It recommended RT-DETR for accuracy reasons, noting its superior performance on CrowdHuman benchmarks.

I disagreed with this for the challenge context. RT-DETR's performance advantage appears at GPU inference speeds (>30fps). On CPU at 15fps with frame stride=2, RT-DETR would process ~3-4fps — too slow for the 20-minute clips in the 48-hour window. The trade-off favours YOLOv8n: slightly lower accuracy but actually runs.

### What I Chose and Why
**YOLOv8n + ByteTrack (via `model.track(persist=True, tracker="bytetrack.yaml")`)**

The key insight: the tracking is more important than the detection accuracy for this use case. A slightly-missed detection in a frame is corrected by ByteTrack's Kalman filter prediction and track interpolation. A single missed ENTRY event is worse than a few missed detections.

For the billing queue clip specifically, I would upgrade to YOLOv8m if the hardware allows — crowd detection quality matters more there. This is documented as a known limitation.

**For partial occlusion:** YOLOv8's confidence score is preserved and passed through to every emitted event. Low-confidence detections are NOT dropped — they appear in the output with their real confidence value. The challenge explicitly requires this.

### Trade-offs
- **Accuracy:** YOLOv8n mAP50 on COCO person ≈ 53% vs YOLOv8m ≈ 63%. In retail CCTV (top-down angle, good lighting), practical accuracy is higher.
- **Speed:** n model processes 1080p at ~8fps on i7 CPU; with stride=2 this gives effective 4fps analysis, sufficient for detecting entry events (people walk, not teleport).
- **Group handling:** Individual bounding boxes per person within a group are usually separated by ByteTrack track IDs — three people entering produce three track IDs and three ENTRY events, provided separation > ~20px.

---

## Decision 2: Event Schema Design

### The Problem
The schema had to support:
1. Analytics queries: conversion rate, dwell, funnel
2. Anomaly detection: queue depth, dead zones
3. Deduplication: idempotent ingest via event_id
4. Session reconstruction: who is a "new visitor" vs re-entry
5. Staff exclusion at query time (not at ingest)

### Options Considered

**Option A: Flat schema, one table**
All analytics derived from a single `events` table. Simple to ingest. Complex to query.

**Option B: Events + Sessions (chosen)**
Dual-write on ingest: events table is the append-only ledger; sessions table is the mutable aggregate. Analytics queries hit sessions for funnel/conversion, events for heatmap/anomalies.

**Option C: Event sourcing with read-model projection**
Kafka-style: events table is source of truth; read models rebuilt on demand. Correct at scale; massive overkill for this challenge.

### What AI Suggested
I asked an LLM to design the event schema given the challenge's event catalogue. It suggested a fully normalised schema with separate tables for `visitors`, `zones`, `sessions`, `zone_visits`, and `events` linked by foreign keys.

I rejected this because:
1. The foreign key structure requires a visitor to be "registered" before emitting events — but visitors are discovered via detection, not pre-registered. Race conditions at ingest.
2. Five-table joins for the funnel endpoint would be slow.
3. The challenge says schema compliance is evaluated — keeping the event schema close to the given JSON format is safer.

### What I Chose and Why
**Flat event JSON → `events` ORM + `sessions` aggregate ORM**

The `events` table mirrors the event schema exactly — `event_id`, `store_id`, `visitor_id`, `event_type`, etc. Every field is first-class.

The `sessions` table is a denormalised aggregate upserted on ingest. It stores:
- `first_entry_ts`, `last_exit_ts` (for session duration)
- `visited_billing` (Boolean flag — avoids a JOIN for funnel)
- `converted` (Boolean — set when POS correlation matches)
- `reentry_count` (incremented but session NOT duplicated)
- `is_staff` (for filtering)

**Key design decision: staff not excluded at ingest.** Staff events are stored with `is_staff=True`. All analytics queries apply a `WHERE is_staff = FALSE` filter. This means:
- Staff events are auditable
- You can retrospectively re-classify a staff/customer boundary
- The pipeline doesn't need to get staff classification perfect to avoid breaking the API

**session_seq in metadata:** Each event carries its ordinal position within the visitor's session. This makes session reconstruction from the JSONL file possible without requiring a database query.

### Trade-offs
- **Write amplification:** Every ingest triggers an upsert on `sessions` in addition to the `events` insert. At high volume, this creates write contention. Mitigated with batch upserts and WAL mode.
- **Conversion accuracy:** The `converted` flag is set by POS correlation at query time (sliding 5-minute window), not at ingest. This means conversion numbers are always fresh but requires reading POS data on every /metrics call. At 40 stores this would be cached.

---

## Decision 3: API Architecture — FastAPI + SQLite vs Alternatives

### The Problem
The API needs to:
- Ingest 500 events per batch reliably
- Return real-time metrics (no yesterday's cache)
- Handle DB unavailability gracefully (503, not 500)
- Produce structured logs readable by on-call engineers
- Start with `docker compose up` — no manual steps

### Options Considered

**FastAPI + SQLite (chosen)**
- Async I/O throughout
- Single binary — no separate DB container
- Adequate performance for challenge data volumes
- Simple to containerise

**FastAPI + PostgreSQL**
- Better concurrent write performance
- TimescaleDB extension for time-series queries
- Requires second container, environment variable coordination
- More setup time in the 48-hour window

**Flask + SQLite**
- Familiar
- Synchronous — blocks on DB calls
- No automatic OpenAPI docs generation
- Rejected: async is needed for idiomatic FastAPI Pydantic validation

**Django + PostgreSQL**
- ORM is excellent
- Way too much boilerplate for a single-service API
- Rejected immediately

### What AI Suggested
I asked Claude which storage engine to use for a real-time events API. It recommended PostgreSQL with a JSONB column for event metadata, citing better index support on JSON fields and native UUID handling.

I agreed with the PostgreSQL reasoning for production, but for the challenge I overrode it. The evaluation criteria explicitly states "SQLite is fine." Adding PostgreSQL adds a second Docker container, connection pooling configuration, and initialization scripts — all of which introduce failure modes that could break the acceptance gate. SQLite with WAL mode eliminates the external DB dependency entirely.

**I noted in the schema where the PostgreSQL migration would be:** change `DATABASE_URL` and swap `sqlite_insert` for `postgresql_insert` (one line change). The schema is otherwise identical.

### Idempotency Implementation
The challenge requires POST /events/ingest to be "safe to call twice with the same payload."

Two approaches:
1. **Check-then-insert:** SELECT existing IDs, then INSERT only new ones. Two queries, but the bulk SELECT is fast with the `uq_event_id` unique constraint.
2. **INSERT OR IGNORE (SQLite) / ON CONFLICT DO NOTHING (PostgreSQL):** Single query, database enforces uniqueness.

I used approach 1 for the ability to return `duplicate_ids` in the response (approach 2 silently ignores). The response distinguishing "accepted" from "duplicate" is important for pipeline debugging.

### What I Would Change for Production
- Replace SQLite with TimescaleDB (PostgreSQL + time-series)
- Add a Redis cache layer for /metrics and /heatmap (30s TTL)
- Replace synchronous anomaly detection on every API call with a scheduled background task (every 60s)
- Add authentication middleware (API key or JWT)
- Replace SQLite WAL with a write-ahead log replicated to object storage for durability

### Trade-offs
| Aspect | SQLite | PostgreSQL |
|--------|--------|------------|
| Setup complexity | 1 container | 2 containers |
| Concurrent writes | Limited (WAL helps) | Excellent |
| Query performance | Good to ~10M rows | Excellent at any scale |
| Operational risk | Low | Medium (connection pooling, migrations) |
| Challenge fit | ✅ Perfect | Overkill |