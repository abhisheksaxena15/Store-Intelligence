"""
pipeline/reid.py
================
Re-ID Manager: maps ByteTrack local track IDs → global visitor_id tokens.

Two complementary strategies are used:

1. Appearance-based Re-ID (primary)
   Uses torchreid OSNet to extract a 512-d feature vector from each person crop.
   On every new track assignment, we compare against a gallery of recently-exited
   visitors. If cosine similarity > REID_THRESHOLD, the visitor is "re-entering"
   and gets their original visitor_id back with a REENTRY flag.

2. Trajectory / position heuristic (fallback)
   When torchreid is unavailable (CPU-only / memory limited), we use IoU between
   the new track's bounding box and the last-known bbox of recently-exited tracks.
   Lower accuracy, but works without GPU.

Why OSNet?
  - Small model (~2MB), fast on CPU
  - Designed for short-duration occluded re-ID (exactly our use case)
  - Available via torchreid without custom training

Limitations:
  - Face blur makes appearance-based re-ID harder; we rely on body shape + clothing
  - Two people in identical uniforms (staff) can be confused → mitigated by staff
    being excluded from customer metrics anyway
  - Re-entry window is configurable; default 5 minutes covers typical "stepped out"
    scenarios without false-positive matching unrelated visitors hours apart
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Cosine similarity threshold above which we declare same-person match
REID_THRESHOLD = 0.72

# How long (seconds) to keep gallery entries for re-entry matching
GALLERY_TTL_SECONDS = 300   # 5 minutes


def _make_visitor_id(store_id: str, camera_id: str, track_id: int, ts: float) -> str:
    """Generate a short deterministic visitor token."""
    raw = f"{store_id}:{camera_id}:{track_id}:{ts:.3f}"
    return "VIS_" + hashlib.sha256(raw.encode()).hexdigest()[:6]


@dataclass
class GalleryEntry:
    visitor_id: str
    feature_vec: Optional[np.ndarray]   # 512-d OSNet embedding; None if unavailable
    last_bbox: tuple[int, int, int, int]
    exit_ts: float
    store_id: str
    camera_id: str


class ReIDManager:
    """
    Resolves ByteTrack local track IDs → stable global visitor_id tokens.

    Maintains:
      - track_to_visitor: mapping from active track IDs to visitor_ids
      - gallery: recently-exited visitors available for re-entry matching
    """

    def __init__(
        self,
        store_id: str,
        camera_id: str,
        reentry_window_seconds: float = GALLERY_TTL_SECONDS,
    ) -> None:
        self.store_id = store_id
        self.camera_id = camera_id
        self.reentry_window = reentry_window_seconds

        self._track_to_visitor: dict[int, str] = {}
        self._gallery: list[GalleryEntry] = []
        self._extractor = self._load_extractor()

    def _load_extractor(self):
        """Try to load torchreid OSNet feature extractor. Return None on failure."""
        try:
            import torchreid
            extractor = torchreid.utils.FeatureExtractor(
                model_name="osnet_x0_25",
                model_path=None,   # downloads weights automatically
                device="cpu",
            )
            logger.info("OSNet Re-ID extractor loaded")
            return extractor
        except Exception as e:
            logger.warning(f"torchreid unavailable ({e}), falling back to IoU-based Re-ID")
            return None

    def _extract_features(self, frame: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
        """Extract appearance feature vector from a person crop."""
        if self._extractor is None:
            return None
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        try:
            import torch
            import torchvision.transforms as T
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            img_tensor = transform(crop).unsqueeze(0)
            with torch.no_grad():
                features = self._extractor(img_tensor)
            return features.cpu().numpy().flatten()
        except Exception:
            return None

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _iou_bbox(self, a: tuple, b: tuple) -> float:
        """Intersection-over-Union between two bounding boxes."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 < ix1 or iy2 < iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter + 1e-6)

    def _prune_gallery(self, current_ts: float) -> None:
        """Remove expired gallery entries."""
        self._gallery = [
            g for g in self._gallery
            if (current_ts - g.exit_ts) < self.reentry_window
        ]

    def resolve(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        timestamp: float,
    ) -> Tuple[str, bool]:
        """
        Map a bytetrack track_id to a stable visitor_id.

        Returns (visitor_id, is_reentry).
        is_reentry=True means this visitor was seen before (matched from gallery).
        """
        # Already mapped
        if track_id in self._track_to_visitor:
            return self._track_to_visitor[track_id], False

        # New track — attempt to match against gallery
        self._prune_gallery(timestamp)
        features = self._extract_features(frame, bbox)

        best_match: Optional[GalleryEntry] = None
        best_score = 0.0

        for entry in self._gallery:
            if entry.store_id != self.store_id:
                continue   # cross-store matching not allowed

            if features is not None and entry.feature_vec is not None:
                score = self._cosine_sim(features, entry.feature_vec)
            else:
                # IoU-based fallback
                score = self._iou_bbox(bbox, entry.last_bbox) * 0.5  # lower confidence

            if score > best_score:
                best_score = score
                best_match = entry

        is_reentry = False
        if best_match and best_score >= REID_THRESHOLD:
            visitor_id = best_match.visitor_id
            is_reentry = True
            # Remove from gallery (consumed by re-entry)
            self._gallery = [g for g in self._gallery if g.visitor_id != visitor_id]
            logger.debug(f"Re-entry: track {track_id} → {visitor_id} (score={best_score:.3f})")
        else:
            visitor_id = _make_visitor_id(self.store_id, self.camera_id, track_id, timestamp)

        self._track_to_visitor[track_id] = visitor_id
        return visitor_id, is_reentry

    def on_exit(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        timestamp: float,
    ) -> None:
        """
        Called when a track is confirmed exited.
        Moves visitor to gallery for potential re-entry matching.
        """
        visitor_id = self._track_to_visitor.pop(track_id, None)
        if visitor_id is None:
            return

        features = self._extract_features(frame, bbox)
        self._gallery.append(
            GalleryEntry(
                visitor_id=visitor_id,
                feature_vec=features,
                last_bbox=bbox,
                exit_ts=timestamp,
                store_id=self.store_id,
                camera_id=self.camera_id,
            )
        )
        logger.debug(f"Track {track_id} ({visitor_id}) moved to gallery at t={timestamp:.1f}")