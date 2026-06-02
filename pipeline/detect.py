"""
pipeline/detect.py
==================
Main detection + tracking script for the Store Intelligence pipeline.

Processes a single CCTV clip:
  1. Runs YOLOv8 person detection on each frame
  2. Associates detections across frames via ByteTrack (built into ultralytics)
  3. Classifies staff vs customer using uniform-colour heuristic (+ optional VLM)
  4. Passes tracklets to reid.py for cross-camera / re-entry linking
  5. Passes tracks to event_generator.py which emits structured events
  6. Writes events to a JSONL file

Usage:
    python detect.py \
        --video clips/STORE_BLR_002/entry.mp4 \
        --store-id STORE_BLR_002 \
        --camera-id CAM_ENTRY_01 \
        --layout data/store_layout.json \
        --output events/STORE_BLR_002_entry.jsonl \
        [--pos-csv data/pos_transactions.csv]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from event_generator import EventGenerator
from reid import ReIDManager
from tracker import TrackState, TrackStore

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

YOLO_MODEL = "yolov8n.pt"       # use yolov8m.pt for better accuracy if time allows
PERSON_CLASS_ID = 0             # COCO class 0 = person
CONF_THRESHOLD = 0.35           # minimum detection confidence to pass downstream
IOU_THRESHOLD = 0.45            # NMS IOU threshold
FRAME_STRIDE = 2                # process every Nth frame to trade speed vs accuracy
ENTRY_ZONE_FRACTION = 0.15      # bottom-N fraction of frame height = entry threshold line


@dataclass
class DetectionConfig:
    video_path: Path
    store_id: str
    camera_id: str
    layout_path: Path
    output_path: Path
    pos_csv_path: Optional[Path] = None
    model_name: str = YOLO_MODEL
    conf_threshold: float = CONF_THRESHOLD
    iou_threshold: float = IOU_THRESHOLD
    frame_stride: int = FRAME_STRIDE
    device: str = "cpu"          # set "cuda" or "mps" if GPU available


# ─── Staff Detection ──────────────────────────────────────────────────────────

class StaffClassifier:
    """
    Heuristic staff detector.

    Strategy: retail staff commonly wear a uniform of consistent, saturated colour
    (e.g. dark polo, branded apron). We compare the dominant HSV hue cluster of
    the upper-body crop against a learned per-store colour histogram. If no
    histogram has been seeded (cold start), we fall back to motion-based
    permanence: a track that remains in-store for the entire clip duration with
    >3 zone transitions is likely staff.

    A VLM alternative (GPT-4V / Claude Vision) is more accurate but incurs latency
    and API cost. Recommended for offline post-processing, not real-time.
    """

    def __init__(self, staff_hue_ranges: list[tuple[int, int]] | None = None):
        # HSV hue ranges for known staff uniform colours (tunable per store)
        # Default: dark navy (100–130) and burgundy (160–175)
        self.staff_hue_ranges: list[tuple[int, int]] = staff_hue_ranges or [
            (100, 130),
            (160, 175),
        ]
        self._track_frame_counts: dict[int, int] = {}   # track_id → frame appearances
        self._total_frames: int = 0

    def update_frame_count(self, track_id: int) -> None:
        self._track_frame_counts[track_id] = (
            self._track_frame_counts.get(track_id, 0) + 1
        )

    def set_total_frames(self, n: int) -> None:
        self._total_frames = n

    def classify(self, frame: np.ndarray, bbox: tuple[int, int, int, int], track_id: int) -> bool:
        """
        Returns True if the detection is likely a staff member.

        bbox: (x1, y1, x2, y2) in pixel coordinates
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Crop upper body (top 40% of bounding box — avoids leg region)
        crop_y2 = y1 + int((y2 - y1) * 0.4)
        crop_y2 = min(crop_y2, h)
        crop = frame[y1:crop_y2, x1:x2]

        if crop.size == 0:
            return False

        # Convert to HSV and extract hue channel
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]  # 0–179 in OpenCV

        # Compute dominant hue via histogram peak
        hist = cv2.calcHist([hue], [0], None, [180], [0, 180])
        dominant_hue = int(np.argmax(hist))

        for lo, hi in self.staff_hue_ranges:
            if lo <= dominant_hue <= hi:
                return True

        # Fallback: permanence heuristic
        if self._total_frames > 0:
            presence_ratio = self._track_frame_counts.get(track_id, 0) / self._total_frames
            if presence_ratio > 0.6:   # present in >60% of frames → likely staff
                return True

        return False


# ─── Zone Classifier ─────────────────────────────────────────────────────────

class ZoneClassifier:
    """
    Maps a bounding box centroid to a named zone using polygon definitions
    from store_layout.json.

    Each zone is defined as a list of [x_fraction, y_fraction] vertices
    (normalised 0–1 relative to frame dimensions). We use cv2.pointPolygonTest
    for robust hit testing.
    """

    def __init__(self, layout: dict, camera_id: str, frame_width: int, frame_height: int):
        self.zones: dict[str, np.ndarray] = {}   # zone_id → polygon in pixels
        self.camera_id = camera_id

        # Load zones relevant to this camera
        camera_zones = layout.get("cameras", {}).get(camera_id, {}).get("zones", [])
        for zone in camera_zones:
            zone_id = zone["zone_id"]
            poly_norm = np.array(zone["polygon"], dtype=np.float32)
            # Scale from normalised fractions to pixel coords
            poly_px = poly_norm.copy()
            poly_px[:, 0] *= frame_width
            poly_px[:, 1] *= frame_height
            self.zones[zone_id] = poly_px.astype(np.int32)

    def classify(self, cx: float, cy: float) -> Optional[str]:
        """Returns the zone_id the centroid falls inside, or None."""
        pt = (float(cx), float(cy))
        for zone_id, poly in self.zones.items():
            if cv2.pointPolygonTest(poly, pt, False) >= 0:
                return zone_id
        return None


# ─── Direction Detector ───────────────────────────────────────────────────────

class DirectionDetector:
    """
    Determines ENTRY vs EXIT by tracking centroid movement across the
    entry-threshold line.

    For the entry camera, the door threshold is treated as a horizontal
    band near the bottom (or top) of the frame. Movement from outside
    (below threshold) to inside (above threshold) = ENTRY, and vice versa.

    The exact axis depends on camera orientation, which is encoded in
    store_layout.json under cameras[camera_id].entry_direction.
    """

    def __init__(self, layout: dict, camera_id: str, frame_height: int, frame_width: int):
        cam_cfg = layout.get("cameras", {}).get(camera_id, {})
        self.entry_direction = cam_cfg.get("entry_direction", "bottom_to_top")
        self.threshold_fraction = cam_cfg.get("threshold_fraction", ENTRY_ZONE_FRACTION)
        self.frame_height = frame_height
        self.frame_width = frame_width

        # Pixel position of threshold line
        if "vertical" in self.entry_direction:
            self.threshold_px = int(self.threshold_fraction * frame_width)
        else:
            self.threshold_px = int((1.0 - self.threshold_fraction) * frame_height)

        # track_id → last side (True = inside store, False = outside)
        self._last_side: dict[int, Optional[bool]] = {}

    def _get_side(self, cx: float, cy: float) -> bool:
        """True = inside store side of threshold."""
        if "vertical" in self.entry_direction:
            return cx < self.threshold_px
        else:
            # bottom_to_top: inside is above the threshold line (lower y value)
            return cy < self.threshold_px

    def update(self, track_id: int, cx: float, cy: float) -> Optional[str]:
        """
        Returns "ENTRY", "EXIT", or None based on centroid crossing the threshold.
        """
        current_side = self._get_side(cx, cy)
        last_side = self._last_side.get(track_id)
        self._last_side[track_id] = current_side

        if last_side is None:
            return None  # first observation, no direction yet

        if not last_side and current_side:
            return "ENTRY"
        if last_side and not current_side:
            return "EXIT"
        return None


# ─── Main Detection Loop ──────────────────────────────────────────────────────

def load_layout(layout_path: Path) -> dict:
    with open(layout_path) as f:
        return json.load(f)


def get_clip_start_timestamp(layout: dict, store_id: str, camera_id: str) -> float:
    """
    Returns the Unix timestamp corresponding to frame 0 of this clip.
    Encoded in store_layout.json as stores[store_id].cameras[camera_id].clip_start_utc
    Fallback: epoch 0 (events will have relative timestamps).
    """
    stores = layout.get("stores", {})
    store = stores.get(store_id, {})
    cameras = store.get("cameras", {})
    cam = cameras.get(camera_id, {})
    start = cam.get("clip_start_utc")
    if start:
        import datetime
        return datetime.datetime.fromisoformat(start.replace("Z", "+00:00")).timestamp()
    return 0.0


def run_detection(config: DetectionConfig) -> list[dict]:
    """
    Main detection loop.

    Returns list of emitted event dicts (also written to config.output_path).
    """
    logger.info(
        "Starting detection",
        extra={
            "store_id": config.store_id,
            "camera_id": config.camera_id,
            "video": str(config.video_path),
        },
    )

    # ── Load models and layout ────────────────────────────────────────────────
    model = YOLO(config.model_name)
    model.to(config.device)

    layout = load_layout(config.layout_path)
    clip_start_ts = get_clip_start_timestamp(layout, config.store_id, config.camera_id)

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(config.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {config.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"Video: {total_frames} frames @ {fps:.1f}fps — {frame_width}x{frame_height}")

    # ── Initialise sub-components ─────────────────────────────────────────────
    staff_clf = StaffClassifier()
    staff_clf.set_total_frames(total_frames // config.frame_stride)

    zone_clf = ZoneClassifier(layout, config.camera_id, frame_width, frame_height)
    direction_det = DirectionDetector(layout, config.camera_id, frame_height, frame_width)

    reid_manager = ReIDManager(
        store_id=config.store_id,
        camera_id=config.camera_id,
        reentry_window_seconds=300,   # 5-min re-entry window
    )

    event_gen = EventGenerator(
        store_id=config.store_id,
        camera_id=config.camera_id,
        clip_start_ts=clip_start_ts,
        fps=fps,
        pos_csv_path=config.pos_csv_path,
    )

    # ── Per-frame state ───────────────────────────────────────────────────────
    frame_idx = 0
    all_events: list[dict] = []

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(config.output_path, "w")

    # ── Detection + tracking loop ─────────────────────────────────────────────
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % config.frame_stride != 0:
                continue

            t_sec = frame_idx / fps  # seconds from clip start

            # Run YOLOv8 with ByteTrack
            results = model.track(
                frame,
                classes=[PERSON_CLASS_ID],
                conf=config.conf_threshold,
                iou=config.iou_threshold,
                persist=True,            # maintain track IDs across calls
                tracker="bytetrack.yaml",
                verbose=False,
            )

            if results[0].boxes is None:
                continue

            boxes = results[0].boxes
            if boxes.id is None:
                continue   # no active tracks this frame

            track_ids = boxes.id.int().cpu().tolist()
            bboxes = boxes.xyxy.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            for track_id, bbox, conf in zip(track_ids, bboxes, confs):
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # Staff classification
                is_staff = staff_clf.classify(frame, (x1, y1, x2, y2), track_id)
                staff_clf.update_frame_count(track_id)

                # Zone classification
                zone_id = zone_clf.classify(cx, cy)

                # Direction (entry/exit) — only meaningful on entry-camera
                direction = direction_det.update(track_id, cx, cy)

                # Re-ID: map bytetrack local ID to global visitor_id
                visitor_id, is_reentry = reid_manager.resolve(
                    track_id=track_id,
                    frame=frame,
                    bbox=(x1, y1, x2, y2),
                    timestamp=clip_start_ts + t_sec,
                )

                # Generate events for this detection
                new_events = event_gen.process(
                    track_id=track_id,
                    visitor_id=visitor_id,
                    frame_idx=frame_idx,
                    t_sec=t_sec,
                    bbox=(x1, y1, x2, y2),
                    zone_id=zone_id,
                    direction=direction,
                    is_staff=is_staff,
                    is_reentry=is_reentry,
                    confidence=float(conf),
                )

                for evt in new_events:
                    out_f.write(json.dumps(evt) + "\n")
                    all_events.append(evt)

    finally:
        cap.release()
        # Flush any pending ZONE_DWELL or EXIT events for tracks still active at clip end
        terminal_events = event_gen.flush_all(frame_idx, frame_idx / fps)
        for evt in terminal_events:
            out_f.write(json.dumps(evt) + "\n")
            all_events.append(evt)
        out_f.close()

    logger.info(
        f"Detection complete: {len(all_events)} events emitted → {config.output_path}"
    )
    return all_events


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Store Intelligence — detection pipeline")
    p.add_argument("--video", required=True, help="Path to input CCTV clip")
    p.add_argument("--store-id", required=True)
    p.add_argument("--camera-id", required=True)
    p.add_argument("--layout", required=True, help="Path to store_layout.json")
    p.add_argument("--output", required=True, help="Output JSONL file path")
    p.add_argument("--pos-csv", default=None, help="Path to pos_transactions.csv")
    p.add_argument("--model", default=YOLO_MODEL)
    p.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    p.add_argument("--stride", type=int, default=FRAME_STRIDE)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    cfg = DetectionConfig(
        video_path=Path(args.video),
        store_id=args.store_id,
        camera_id=args.camera_id,
        layout_path=Path(args.layout),
        output_path=Path(args.output),
        pos_csv_path=Path(args.pos_csv) if args.pos_csv else None,
        model_name=args.model,
        conf_threshold=args.conf,
        frame_stride=args.stride,
        device=args.device,
    )

    events = run_detection(cfg)
    print(f"Emitted {len(events)} events to {cfg.output_path}")