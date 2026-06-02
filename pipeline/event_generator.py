"""
pipeline/event_generator.py
============================
Event Generator — the semantic brain of the pipeline.

Converts raw detection observations (track_id, zone, direction, timestamps)
into structured behavioural events matching the challenge schema.

Event types:
  ENTRY, EXIT, ZONE_ENTER, ZONE_EXIT, ZONE_DWELL,
  BILLING_QUEUE_JOIN, BILLING_QUEUE_ABANDON, REENTRY

Design decisions:
  - Each emitted event is a self-contained dict ready for JSON serialisation
  - POS correlation for BILLING_QUEUE_ABANDON uses a sliding 5-minute window
  - ZONE_DWELL is emitted every 30 seconds of continuous zone occupancy
  - Staff are not excluded here — is_staff flag is set; the API layer filters them
  - Confidence is passed through untouched — the challenge explicitly says
    "do not suppress low-confidence events"; they must appear with their real value
"""

from __future__ import annotations

import datetime
import logging
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Threshold for emitting ZONE_DWELL (seconds)
DWELL_EMIT_INTERVAL = 30.0

# Billing zone identifier suffix — matches store_layout.json
BILLING_ZONE_SUFFIX = "BILLING"

# Window (seconds) to look forward in POS data when checking if a billing visitor converted
POS_CORRELATION_WINDOW = 300.0  # 5 minutes


def _iso_ts(unix_ts: float) -> str:
    """Convert Unix timestamp to ISO-8601 UTC string."""
    return datetime.datetime.utcfromtimestamp(unix_ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_event(
    store_id: str,
    camera_id: str,
    visitor_id: str,
    event_type: str,
    timestamp: float,
    zone_id: Optional[str],
    dwell_ms: int,
    is_staff: bool,
    confidence: float,
    session_seq: int,
    metadata: Optional[dict] = None,
) -> dict:
    return {
        "event_id": str(uuid.uuid4()),
        "store_id": store_id,
        "camera_id": camera_id,
        "visitor_id": visitor_id,
        "event_type": event_type,
        "timestamp": _iso_ts(timestamp),
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": is_staff,
        "confidence": round(confidence, 4),
        "metadata": metadata or {
            "queue_depth": None,
            "sku_zone": zone_id,
            "session_seq": session_seq,
        },
    }


class POSCorrelator:
    """
    Loads pos_transactions.csv and provides fast lookup:
    "was there a transaction at this store within T seconds after ts?"
    """

    def __init__(self, pos_csv_path: Optional[Path], store_id: str) -> None:
        self._transactions: list[float] = []   # sorted Unix timestamps for this store

        if pos_csv_path is None or not pos_csv_path.exists():
            logger.info("No POS CSV provided — BILLING_QUEUE_ABANDON detection disabled")
            return

        import csv
        import datetime

        with open(pos_csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("store_id", "").strip() != store_id:
                    continue
                ts_str = row.get("timestamp", "").strip()
                try:
                    ts = datetime.datetime.fromisoformat(
                        ts_str.replace("Z", "+00:00")
                    ).timestamp()
                    self._transactions.append(ts)
                except ValueError:
                    pass

        self._transactions.sort()
        logger.info(f"POS correlator loaded {len(self._transactions)} transactions for {store_id}")

    def transaction_follows(self, after_ts: float, window: float = POS_CORRELATION_WINDOW) -> bool:
        """True if at least one transaction falls in (after_ts, after_ts + window]."""
        import bisect
        lo = bisect.bisect_right(self._transactions, after_ts)
        hi = bisect.bisect_right(self._transactions, after_ts + window)
        return hi > lo


class _TrackEventState:
    """Per-track mutable state tracked by EventGenerator."""
    __slots__ = (
        "visitor_id",
        "is_staff",
        "session_seq",
        "current_zone",
        "zone_enter_ts",
        "last_dwell_emit_ts",
        "has_entered",
        "billing_enter_ts",
        "last_frame_ts",
    )

    def __init__(self, visitor_id: str, is_staff: bool) -> None:
        self.visitor_id = visitor_id
        self.is_staff = is_staff
        self.session_seq = 0
        self.current_zone: Optional[str] = None
        self.zone_enter_ts: Optional[float] = None
        self.last_dwell_emit_ts: Optional[float] = None
        self.has_entered = False
        self.billing_enter_ts: Optional[float] = None
        self.last_frame_ts: float = 0.0

    def next_seq(self) -> int:
        self.session_seq += 1
        return self.session_seq


class EventGenerator:
    """
    Converts per-frame detection observations into schema-compliant events.

    One EventGenerator instance per clip (store + camera combination).
    """

    def __init__(
        self,
        store_id: str,
        camera_id: str,
        clip_start_ts: float,
        fps: float,
        pos_csv_path: Optional[Path] = None,
    ) -> None:
        self.store_id = store_id
        self.camera_id = camera_id
        self.clip_start_ts = clip_start_ts
        self.fps = fps
        self.pos = POSCorrelator(pos_csv_path, store_id)

        self._states: dict[int, _TrackEventState] = {}  # track_id → state
        self._queue_depth: int = 0  # running billing zone queue counter

    def _get_or_create_state(self, track_id: int, visitor_id: str, is_staff: bool) -> _TrackEventState:
        if track_id not in self._states:
            self._states[track_id] = _TrackEventState(visitor_id, is_staff)
        return self._states[track_id]

    def _abs_ts(self, t_sec: float) -> float:
        return self.clip_start_ts + t_sec

    def _is_billing_zone(self, zone_id: Optional[str]) -> bool:
        if zone_id is None:
            return False
        return BILLING_ZONE_SUFFIX in zone_id.upper()

    def process(
        self,
        track_id: int,
        visitor_id: str,
        frame_idx: int,
        t_sec: float,
        bbox: tuple[int, int, int, int],
        zone_id: Optional[str],
        direction: Optional[str],   # "ENTRY", "EXIT", or None
        is_staff: bool,
        is_reentry: bool,
        confidence: float,
    ) -> list[dict]:
        """
        Called once per detection per frame.
        Returns list of new events to emit (may be empty).
        """
        events: list[dict] = []
        state = self._get_or_create_state(track_id, visitor_id, is_staff)
        state.last_frame_ts = t_sec
        abs_ts = self._abs_ts(t_sec)

        def make(event_type: str, zone: Optional[str] = zone_id, dwell_ms: int = 0,
                 extra_meta: Optional[dict] = None) -> dict:
            meta = {
                "queue_depth": None,
                "sku_zone": zone,
                "session_seq": state.session_seq,
            }
            if extra_meta:
                meta.update(extra_meta)
            return _make_event(
                store_id=self.store_id,
                camera_id=self.camera_id,
                visitor_id=state.visitor_id,
                event_type=event_type,
                timestamp=abs_ts,
                zone_id=zone,
                dwell_ms=dwell_ms,
                is_staff=state.is_staff,
                confidence=confidence,
                session_seq=state.next_seq(),
                metadata=meta,
            )

        # ── REENTRY ──────────────────────────────────────────────────────────
        if is_reentry and not state.has_entered:
            state.has_entered = True
            events.append(make("REENTRY", zone=None, dwell_ms=0))

        # ── ENTRY / EXIT ─────────────────────────────────────────────────────
        if direction == "ENTRY" and not state.has_entered:
            state.has_entered = True
            events.append(make("ENTRY", zone=None, dwell_ms=0))

        elif direction == "EXIT" and state.has_entered:
            # Flush current zone before EXIT
            if state.current_zone:
                dwell = int((t_sec - (state.zone_enter_ts or t_sec)) * 1000)
                events.append(make("ZONE_EXIT", zone=state.current_zone, dwell_ms=dwell))
                # Check BILLING_QUEUE_ABANDON
                if self._is_billing_zone(state.current_zone):
                    if not self.pos.transaction_follows(abs_ts):
                        events.append(make(
                            "BILLING_QUEUE_ABANDON",
                            zone=state.current_zone,
                            dwell_ms=dwell,
                            extra_meta={"queue_depth": self._queue_depth},
                        ))
                    if self._queue_depth > 0:
                        self._queue_depth -= 1
                state.current_zone = None
                state.zone_enter_ts = None

            events.append(make("EXIT", zone=None, dwell_ms=0))
            state.has_entered = False

        # ── ZONE_ENTER / ZONE_EXIT ───────────────────────────────────────────
        if zone_id != state.current_zone:
            # Exiting previous zone
            if state.current_zone is not None:
                dwell = int((t_sec - (state.zone_enter_ts or t_sec)) * 1000)
                events.append(make("ZONE_EXIT", zone=state.current_zone, dwell_ms=dwell))

                if self._is_billing_zone(state.current_zone):
                    # Left billing zone — check if converted
                    if not self.pos.transaction_follows(abs_ts):
                        events.append(make(
                            "BILLING_QUEUE_ABANDON",
                            zone=state.current_zone,
                            dwell_ms=dwell,
                            extra_meta={"queue_depth": self._queue_depth},
                        ))
                    if self._queue_depth > 0:
                        self._queue_depth -= 1
                state.last_dwell_emit_ts = None

            # Entering new zone
            if zone_id is not None:
                state.current_zone = zone_id
                state.zone_enter_ts = t_sec
                events.append(make("ZONE_ENTER", zone=zone_id, dwell_ms=0))

                if self._is_billing_zone(zone_id):
                    state.billing_enter_ts = abs_ts
                    if self._queue_depth > 0:
                        self._queue_depth += 1
                        events.append(make(
                            "BILLING_QUEUE_JOIN",
                            zone=zone_id,
                            dwell_ms=0,
                            extra_meta={"queue_depth": self._queue_depth},
                        ))
            else:
                state.current_zone = None
                state.zone_enter_ts = None

        # ── ZONE_DWELL ───────────────────────────────────────────────────────
        if state.current_zone and state.zone_enter_ts is not None:
            time_in_zone = t_sec - state.zone_enter_ts
            last_dwell = state.last_dwell_emit_ts or state.zone_enter_ts
            if (t_sec - last_dwell) >= DWELL_EMIT_INTERVAL:
                dwell_ms = int(time_in_zone * 1000)
                events.append(make("ZONE_DWELL", zone=state.current_zone, dwell_ms=dwell_ms))
                state.last_dwell_emit_ts = t_sec

        return events

    def flush_all(self, final_frame_idx: int, final_t_sec: float) -> list[dict]:
        """
        Called at end of clip to emit terminal events for all still-active tracks.
        Emits ZONE_EXIT + EXIT for everyone still inside at clip end.
        """
        events: list[dict] = []
        abs_ts = self._abs_ts(final_t_sec)

        for track_id, state in self._states.items():
            if not state.has_entered:
                continue

            def make_final(event_type: str, zone: Optional[str] = None, dwell_ms: int = 0) -> dict:
                return _make_event(
                    store_id=self.store_id,
                    camera_id=self.camera_id,
                    visitor_id=state.visitor_id,
                    event_type=event_type,
                    timestamp=abs_ts,
                    zone_id=zone,
                    dwell_ms=dwell_ms,
                    is_staff=state.is_staff,
                    confidence=1.0,  # synthetic terminal event
                    session_seq=state.next_seq(),
                )

            if state.current_zone:
                dwell = int((final_t_sec - (state.zone_enter_ts or final_t_sec)) * 1000)
                events.append(make_final("ZONE_EXIT", zone=state.current_zone, dwell_ms=dwell))

            events.append(make_final("EXIT", zone=None, dwell_ms=0))

        return events