"""Per-track best-reading dedup (FR-1.5, FR-2.2 — one event per track/pass, best-confidence frame).

Note on scope: this is in-memory, scoped to a single video run. Phase 2's
short-TTL dedup cache (in-memory or Redis, per docs/ARCHITECTURE.md section
2.1) covers dedup across gaps longer than a tracker can bridge on its own
(e.g. a vehicle briefly leaving and re-entering frame) — that's out of scope
for this standalone phase. Dedup across a track's ID *changing mid-pass*
(ByteTrack losing/reacquiring a track, or our own IoU fallback kicking in
before ByteTrack confirms a new track) is handled here via `migrate()`,
called from PipelineRunner when ByteTrackTracker reports a merge — see
anpr/tracking/byte_tracker.py's `_reconcile_id`.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np

from ..interfaces import BBox

logger = logging.getLogger(__name__)


@dataclass
class BestReading:
    track_id: int
    vehicle_class: str
    plate_text: str
    ocr_confidence: float
    vehicle_confidence: float
    frame_idx: int
    timestamp_sec: float
    vehicle_bbox: BBox
    plate_bbox: BBox | None
    evidence_frame_path: str | None = None
    evidence_plate_crop_path: str | None = None


class BestReadingAggregator:
    """Keeps the single highest-OCR-confidence reading seen so far per track_id.

    Evidence images are written to disk immediately whenever a new best
    reading appears (overwriting the previous best for that track_id)
    rather than held as arrays in memory, so this stays cheap on long videos.
    """

    def __init__(self, evidence_dir: str):
        self.evidence_dir = evidence_dir
        os.makedirs(evidence_dir, exist_ok=True)
        self._best: dict[int, BestReading] = {}

    def offer(
        self,
        track_id: int,
        vehicle_class: str,
        plate_text: str,
        ocr_confidence: float,
        vehicle_confidence: float,
        frame_idx: int,
        timestamp_sec: float,
        vehicle_bbox: BBox,
        plate_bbox: BBox | None,
        frame_image: np.ndarray,
        plate_crop_image: np.ndarray | None,
    ) -> None:
        current = self._best.get(track_id)
        if current is not None and current.ocr_confidence >= ocr_confidence:
            return

        frame_path = os.path.join(self.evidence_dir, f"track_{track_id}_frame.jpg")
        plate_path: str | None = os.path.join(self.evidence_dir, f"track_{track_id}_plate.jpg")
        cv2.imwrite(frame_path, frame_image)
        if plate_crop_image is not None and plate_crop_image.size > 0:
            cv2.imwrite(plate_path, plate_crop_image)
        else:
            plate_path = None

        self._best[track_id] = BestReading(
            track_id=track_id,
            vehicle_class=vehicle_class,
            plate_text=plate_text,
            ocr_confidence=ocr_confidence,
            vehicle_confidence=vehicle_confidence,
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            vehicle_bbox=vehicle_bbox,
            plate_bbox=plate_bbox,
            evidence_frame_path=frame_path,
            evidence_plate_crop_path=plate_path,
        )

    def pop(self, track_id: int) -> BestReading | None:
        return self._best.pop(track_id, None)

    def migrate(self, old_id: int, new_id: int) -> None:
        """Re-key a stored best-reading from old_id to new_id (a tracker-level
        ID merge). If new_id already has a reading, keep whichever is
        higher-confidence rather than blindly overwriting."""
        old_reading = self._best.pop(old_id, None)
        if old_reading is None:
            return
        current = self._best.get(new_id)
        if current is not None and current.ocr_confidence >= old_reading.ocr_confidence:
            return
        old_reading.track_id = new_id
        self._best[new_id] = old_reading
