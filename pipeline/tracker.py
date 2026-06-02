"""
pipeline/tracker.py
===================
Track state management for the detection pipeline.

Maintains the lifecycle of each bytetrack local track ID:
  - First seen timestamp
  - Zone history
  - Session sequence counter
  - Bounding box history (for trajectory-based re-ID)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrackState:
    """Live state for a single bytetrack tracklet."""
    track_id: int
    visitor_id: str               # global Re-ID token
    first_seen_ts: float          # Unix timestamp
    last_seen_ts: float
    is_staff: bool = False
    current_zone: Optional[str] = None
    zone_enter_ts: Optional[float] = None  # when current zone was entered
    session_seq: int = 0          # ordinal counter for events in this session
    bbox_history: list[tuple[int,int,int,int]] = field(default_factory=list)
    zone_history: list[str] = field(default_factory=list)
    last_dwell_emit_ts: Optional[float] = None   # last ZONE_DWELL emit time
    is_active: bool = True

    def centroid(self) -> tuple[float, float]:
        if not self.bbox_history:
            return (0.0, 0.0)
        x1, y1, x2, y2 = self.bbox_history[-1]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def trajectory_vector(self) -> tuple[float, float]:
        """Average movement direction over last N frames."""
        if len(self.bbox_history) < 2:
            return (0.0, 0.0)
        prev = self.bbox_history[-2]
        curr = self.bbox_history[-1]
        dx = ((curr[0] + curr[2]) / 2.0) - ((prev[0] + prev[2]) / 2.0)
        dy = ((curr[1] + curr[3]) / 2.0) - ((prev[1] + prev[3]) / 2.0)
        return (dx, dy)

    def next_seq(self) -> int:
        self.session_seq += 1
        return self.session_seq


class TrackStore:
    """
    In-memory store of all active and recently-closed TrackState objects.

    Provides fast lookup by track_id.
    """

    def __init__(self) -> None:
        self._active: dict[int, TrackState] = {}
        self._closed: list[TrackState] = []

    def get_or_create(
        self,
        track_id: int,
        visitor_id: str,
        timestamp: float,
    ) -> TrackState:
        if track_id not in self._active:
            self._active[track_id] = TrackState(
                track_id=track_id,
                visitor_id=visitor_id,
                first_seen_ts=timestamp,
                last_seen_ts=timestamp,
            )
        return self._active[track_id]

    def get(self, track_id: int) -> Optional[TrackState]:
        return self._active.get(track_id)

    def close(self, track_id: int, timestamp: float) -> Optional[TrackState]:
        state = self._active.pop(track_id, None)
        if state:
            state.last_seen_ts = timestamp
            state.is_active = False
            self._closed.append(state)
        return state

    def all_active(self) -> list[TrackState]:
        return list(self._active.values())

    def update_bbox(self, track_id: int, bbox: tuple[int,int,int,int], ts: float) -> None:
        state = self._active.get(track_id)
        if state:
            state.bbox_history.append(bbox)
            if len(state.bbox_history) > 30:   # keep last 30 positions
                state.bbox_history.pop(0)
            state.last_seen_ts = ts