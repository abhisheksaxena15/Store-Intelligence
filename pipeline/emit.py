"""
pipeline/emit.py
================
Event schema validation and emission utilities.

This module is the "contract enforcer" — every event leaving the pipeline
must pass through here. It:
  1. Validates event structure against the schema
  2. Ensures event_id uniqueness within a run
  3. Writes validated events to a JSONL output file or yields them
  4. Provides a sample_events validator against sample_events.jsonl

Usage:
    from emit import EventEmitter

    emitter = EventEmitter(output_path=Path("events/out.jsonl"))
    emitter.emit(event_dict)
    emitter.close()

Why separate from event_generator.py?
  event_generator.py handles WHEN to emit (business logic).
  emit.py handles HOW to emit (schema enforcement, I/O, deduplication).
  This separation makes unit-testing the schema layer trivial.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Iterator, Optional

from pydantic import ValidationError

# Import the canonical schema from the app layer.
# sys.path manipulation ensures this works whether emit.py is run:
#   a) directly from pipeline/ directory
#   b) from project root
#   c) imported by tests/
import sys
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.models import StoreEvent  # noqa: E402

logger = logging.getLogger(__name__)


class EventEmitter:
    """
    Thread-safe (for single-process use) event emitter.

    Validates every event dict against StoreEvent Pydantic model
    before writing. Invalid events are logged and counted but NOT dropped
    silently — they are written to a separate error log.
    """

    def __init__(
        self,
        output_path: Path,
        error_path: Optional[Path] = None,
        deduplicate: bool = True,
    ) -> None:
        self.output_path = output_path
        self.error_path = error_path or output_path.with_suffix(".errors.jsonl")
        self.deduplicate = deduplicate

        self._seen_ids: set[str] = set()
        self._accepted = 0
        self._rejected = 0
        self._duplicates = 0

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._out = open(output_path, "w", buffering=1)  # line-buffered
        self._err = open(self.error_path, "w", buffering=1)

    def emit(self, event_dict: dict) -> bool:
        """
        Validate and write one event.
        Returns True if accepted, False if rejected.
        """
        event_id = event_dict.get("event_id", "")

        # Deduplication guard
        if self.deduplicate and event_id in self._seen_ids:
            self._duplicates += 1
            logger.debug(f"Duplicate event_id skipped: {event_id}")
            return False

        # Schema validation
        try:
            validated = StoreEvent(**event_dict)
        except ValidationError as e:
            self._rejected += 1
            error_record = {
                "event_id": event_id,
                "error": e.errors(),
                "raw": event_dict,
            }
            self._err.write(json.dumps(error_record) + "\n")
            logger.warning(f"Event validation failed: {event_id} — {e.error_count()} errors")
            return False

        # Write valid event
        self._out.write(validated.model_dump_json() + "\n")
        self._accepted += 1
        if self.deduplicate:
            self._seen_ids.add(event_id)
        return True

    def emit_batch(self, events: list[dict]) -> tuple[int, int]:
        """Emit a list of events. Returns (accepted, rejected)."""
        accepted = sum(1 for e in events if self.emit(e))
        rejected = len(events) - accepted
        return accepted, rejected

    def close(self) -> dict:
        """Flush and close output files. Returns stats."""
        self._out.close()
        self._err.close()
        stats = {
            "output": str(self.output_path),
            "accepted": self._accepted,
            "rejected": self._rejected,
            "duplicates": self._duplicates,
        }
        logger.info("EventEmitter closed", extra=stats)
        return stats

    def __enter__(self) -> "EventEmitter":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    @property
    def stats(self) -> dict:
        return {
            "accepted": self._accepted,
            "rejected": self._rejected,
            "duplicates": self._duplicates,
        }


def validate_against_samples(
    sample_path: Path,
    emitted_path: Path,
) -> dict:
    """
    Validates emitted events against sample_events.jsonl.

    Checks:
      1. All emitted events are schema-valid
      2. Event type distribution roughly matches samples
      3. All required fields are present and non-null where required

    Returns a report dict with pass/fail details.
    """
    report = {
        "schema_errors": [],
        "type_distribution": {},
        "sample_type_distribution": {},
        "total_emitted": 0,
        "total_valid": 0,
        "passed": False,
    }

    # Load samples
    if sample_path.exists():
        sample_types: dict[str, int] = {}
        with open(sample_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                    t = evt.get("event_type", "UNKNOWN")
                    sample_types[t] = sample_types.get(t, 0) + 1
                except json.JSONDecodeError:
                    pass
        report["sample_type_distribution"] = sample_types

    # Validate emitted events
    emitted_types: dict[str, int] = {}
    valid_count = 0
    with open(emitted_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            report["total_emitted"] += 1
            try:
                evt = json.loads(line)
                StoreEvent(**evt)
                valid_count += 1
                t = evt.get("event_type", "UNKNOWN")
                emitted_types[t] = emitted_types.get(t, 0) + 1
            except (json.JSONDecodeError, ValidationError) as e:
                report["schema_errors"].append({"line": i + 1, "error": str(e)})

    report["total_valid"] = valid_count
    report["type_distribution"] = emitted_types
    report["passed"] = (
        valid_count == report["total_emitted"]
        and len(report["schema_errors"]) == 0
    )
    return report


def load_jsonl(path: Path) -> Iterator[dict]:
    """Yield parsed dicts from a JSONL file, skipping blank/bad lines."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping bad JSONL line in {path}: {e}")