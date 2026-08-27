"""Detection event schema — the record emitted once per finalized track.

Fields mirror docs/DATABASE_SCHEMA.md's detection_events table where a Phase-1
equivalent exists (plate_number_raw, gate/timestamp fields, ocr_confidence,
track_id, evidence paths). Fields that don't exist yet at this phase
(direction, direction_confidence, gate_id, vehicle DB match) are added in
Phase 2/3.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class DetectionEvent:
    track_id: int
    vehicle_class: str
    plate_text: str
    ocr_confidence: float
    vehicle_confidence: float
    frame_idx: int
    timestamp_sec: float
    vehicle_bbox: tuple[int, int, int, int]
    plate_bbox: tuple[int, int, int, int] | None
    evidence_frame_path: str | None
    evidence_plate_crop_path: str | None
    # Basename of the video this event came from, e.g. "dataset_clear_01.mp4".
    # Added after a confirmed scoring bug: scripts/score_accuracy.py used to
    # infer the source video from the events-JSON *filename*
    # ("review2_events.json" -> "review2.mp4") and match ground truth on it,
    # so naming an output run anything other than the dataset filename scored
    # every ground-truth row as "Missed" and every real event as a false
    # positive. The source video is a property of the run, so the run records
    # it rather than leaving the scorer to guess from an unrelated filename.
    source_video: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)
