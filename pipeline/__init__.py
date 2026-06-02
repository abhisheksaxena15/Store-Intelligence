"""Package entrypoint for the Store Intelligence detection pipeline."""

from .detect import (
    DetectionConfig,
    StaffClassifier,
    ZoneClassifier,
    DirectionDetector,
)
from .emit import EventEmitter
from .event_generator import EventGenerator
from .ingest_events import ingest as ingest_events
from .reid import ReIDManager
from .tracker import TrackState, TrackStore

__all__ = [
    "DetectionConfig",
    "StaffClassifier",
    "ZoneClassifier",
    "DirectionDetector",
    "EventGenerator",
    "EventEmitter",
    "ingest_events",
    "ReIDManager",
    "TrackState",
    "TrackStore",
]
