"""Pipeline configuration defaults. Overridden by run_pipeline.py CLI flags.

Phase 2 will replace/extend this with the per-gate YAML/JSON config
(gate_id, camera source URI, inbound_reference_vector, camera_angle_deg)
described in docs/ARCHITECTURE.md section 7 and the Phase 2 kickoff prompt —
this phase only needs pipeline-level tuning knobs, not gate identity.
"""
from __future__ import annotations

from dataclasses import dataclass

# COCO class ids used by the pretrained Ultralytics COCO weights.
COCO_CAR = 2
COCO_MOTORCYCLE = 3


@dataclass
class PipelineConfig:
    vehicle_weights: str = "yolov8n.pt"
    plate_weights: str = "models/plate_detector.pt"
    device: str = "cpu"

    vehicle_conf: float = 0.35
    vehicle_iou: float = 0.5
    vehicle_classes: tuple[int, ...] = (COCO_CAR, COCO_MOTORCYCLE)
    tracker_cfg: str = "bytetrack.yaml"  # Ultralytics' built-in tracker config, resolved by name

    plate_conf: float = 0.25
    plate_iou: float = 0.45
    plate_margin_ratio: float = 0.15  # crop margin around the vehicle box, as a fraction of box size

    # Confirmed on real footage: 0.95 cleanly suppressed a garbage OCR read
    # from a split-track fragment (a distant/low-quality crop of a car whose
    # plate was correctly read on a later, better fragment) while every
    # genuine plate read in testing still cleared it comfortably (0.96-0.9997
    # on the winning frame). Trade-off: a genuine plate that never gets a
    # close/clear enough frame to cross 0.95 is now silently dropped rather
    # than logged as a low-confidence flag for review (FR-5.4 wants the
    # latter) — worth watching on harder footage (motorbikes, low light) and
    # dialing back per-gate if it starts suppressing real reads.
    ocr_min_confidence: float = 0.95
    ocr_lang: str = "en"

    track_max_age_frames: int = 30
    track_iou_fallback_threshold: float = 0.3

    frame_skip: int = 1  # process every Nth frame (1 = every frame)
