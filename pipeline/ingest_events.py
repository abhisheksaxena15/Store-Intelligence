"""
pipeline/ingest_events.py
=========================
Reads a JSONL event file and POSTs batches to the API ingest endpoint.
Handles partial failures (422 validation errors) gracefully.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests


def ingest(events_path: Path, api_url: str, batch_size: int = 500) -> None:
    endpoint = f"{api_url.rstrip('/')}/events/ingest"
    batch: list[dict] = []
    total_sent = 0
    total_accepted = 0
    total_rejected = 0

    def flush(b: list[dict]) -> None:
        nonlocal total_accepted, total_rejected
        if not b:
            return
        try:
            resp = requests.post(endpoint, json={"events": b}, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                total_accepted += data.get("accepted", len(b))
                total_rejected += data.get("rejected", 0)
            elif resp.status_code == 207:
                # Partial success
                data = resp.json()
                total_accepted += data.get("accepted", 0)
                total_rejected += data.get("rejected", 0)
                print(f"  Partial: {data.get('rejected', 0)} events rejected", file=sys.stderr)
            else:
                print(f"  HTTP {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
                total_rejected += len(b)
        except requests.RequestException as e:
            print(f"  Request failed: {e}", file=sys.stderr)
            total_rejected += len(b)

    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
                batch.append(evt)
                total_sent += 1
                if len(batch) >= batch_size:
                    flush(batch)
                    batch = []
                    time.sleep(0.05)  # small backoff to avoid overwhelming API
            except json.JSONDecodeError as e:
                print(f"  Bad JSON line: {e}", file=sys.stderr)

    flush(batch)

    print(
        f"Ingest complete: {total_sent} sent, "
        f"{total_accepted} accepted, "
        f"{total_rejected} rejected"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--events", required=True)
    p.add_argument("--api-url", default="http://localhost:8000")
    p.add_argument("--batch-size", type=int, default=500)
    args = p.parse_args()
    ingest(Path(args.events), args.api_url, args.batch_size)