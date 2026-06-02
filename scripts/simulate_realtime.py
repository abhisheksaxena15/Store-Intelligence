"""
scripts/simulate_realtime.py
=============================
Replays a JSONL event file into the API in simulated real-time.

This enables the Live Dashboard (Part E) to show metrics updating live
without needing the actual detection pipeline running simultaneously.

Events are replayed at their original relative timestamps but compressed
by a speed factor (default 60x — 1 minute of footage plays in 1 second).

Usage:
    python scripts/simulate_realtime.py \
        --events data/events/STORE_BLR_002_entry.jsonl \
        --api-url http://localhost:8000 \
        --speed 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests


def iso_to_ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def replay(
    events_path: Path,
    api_url: str,
    speed_factor: float = 60.0,
    batch_window_seconds: float = 1.0,
) -> None:
    """
    Replay events in chronological order at compressed speed.

    Events within a `batch_window_seconds` real-time window are batched
    together into a single ingest call.
    """
    endpoint = f"{api_url.rstrip('/')}/events/ingest"

    # Load all events and sort by timestamp
    events = []
    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not events:
        print("No events to replay.", file=sys.stderr)
        return

    events.sort(key=lambda e: e.get("timestamp", ""))
    first_event_ts = iso_to_ts(events[0]["timestamp"])
    real_start = time.time()

    print(f"Replaying {len(events)} events at {speed_factor}x speed")
    print(f"First event: {events[0]['timestamp']}")
    print(f"Last event:  {events[-1]['timestamp']}")

    batch = []
    total_sent = 0
    i = 0

    while i < len(events):
        evt = events[i]
        event_ts = iso_to_ts(evt["timestamp"])

        # Real time when this event should be sent
        real_target = real_start + (event_ts - first_event_ts) / speed_factor
        now = time.time()

        if now < real_target:
            # Flush current batch before sleeping
            if batch:
                _send_batch(endpoint, batch)
                total_sent += len(batch)
                batch = []
            sleep_time = real_target - now
            print(f"\r  Sent {total_sent}/{len(events)} events... (waiting {sleep_time:.1f}s)", end="")
            time.sleep(min(sleep_time, 0.5))
            continue

        batch.append(evt)
        i += 1

        # Send batch when it reaches 500 events or crosses a real-time boundary
        if len(batch) >= 500:
            _send_batch(endpoint, batch)
            total_sent += len(batch)
            batch = []

    # Final flush
    if batch:
        _send_batch(endpoint, batch)
        total_sent += len(batch)

    print(f"\nReplay complete: {total_sent} events sent")


def _send_batch(endpoint: str, batch: list[dict]) -> None:
    try:
        resp = requests.post(endpoint, json={"events": batch}, timeout=10)
        if resp.status_code not in (200, 207):
            print(f"\n  Warning: HTTP {resp.status_code}", file=sys.stderr)
    except Exception as e:
        print(f"\n  Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--events", required=True)
    p.add_argument("--api-url", default="http://localhost:8000")
    p.add_argument("--speed", type=float, default=60.0, help="Replay speed multiplier")
    args = p.parse_args()
    replay(Path(args.events), args.api_url, args.speed)