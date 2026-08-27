"""Pipeline configuration defaults — the single source of truth for them.

run_pipeline.py's CLI flags read their `default=` values from a module-level
`DEFAULTS = PipelineConfig()` instance instead of hardcoding numbers a
second time. External review caught this drifting out of sync before: this
file said vehicle_conf=0.35/car+motorcycle-only/no imgsz while the CLI had
already moved to 0.10/+bus+truck/imgsz=1280 — anyone importing PipelineConfig
directly (or using it as a config file base in Phase 2) would silently get
the stale, disproven defaults. Wiring it up as the actual source removes the
duplication that let that happen rather than just re-syncing the numbers
once.

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
COCO_BUS = 5
COCO_TRUCK = 7


@dataclass
class PipelineConfig:
    vehicle_weights: str = "yolov8n.pt"
    plate_weights: str = "models/plate_detector.pt"
    device: str = "cpu"

    # Was 0.35. VERIFIED against ultralytics' shipped bytetrack.yaml:
    # ByteTrack's low-confidence recovery matches down to
    # track_low_thresh=0.1, but conf=0.35 discarded everything below that
    # before ByteTrack ever saw it, defeating its own recovery feature.
    vehicle_conf: float = 0.10
    vehicle_iou: float = 0.5
    # Bus/truck included defensively: a car-like vehicle can be
    # misclassified by the generic COCO-pretrained detector — confirmed by
    # testing — and without these classes it's silently dropped rather than
    # logged under the wrong class.
    vehicle_classes: tuple[int, ...] = (COCO_CAR, COCO_MOTORCYCLE, COCO_BUS, COCO_TRUCK)
    # Was implicitly 640 (ultralytics' default when unset). Source frames at
    # 2560x1440 shrink a ~150px motorcycle to ~37px at 640.
    imgsz: int = 1280
    tracker_cfg: str = "bytetrack.yaml"  # Ultralytics' built-in tracker config, resolved by name

    plate_conf: float = 0.25
    plate_iou: float = 0.45
    plate_margin_ratio: float = 0.15  # crop margin around the vehicle box, as a fraction of box size

    # Geometric plate-box filters. These were hardcoded in
    # YoloPlateDetector.__init__ and unreachable from the CLI, so the one
    # thing a "no plates detected" investigation most needs to sweep
    # couldn't be swept without editing code. See the detector's docstrings
    # for what each measured value they were tuned against was.
    #
    # min_plate_aspect_ratio was 1.0, lowered to 0.8: local motorcycle
    # plates are stacked two-line and close to square, so a genuine one
    # viewed at a gate-camera angle can measure just under 1.0 and was
    # being discarded before OCR ever saw it. Cars are unaffected (observed
    # 1.5-2.0). The upper bound stays at 3.0, which still excludes the
    # "Entrance" sign false positive measured at 4.47.
    min_plate_width_px: int = 24
    min_plate_height_px: int = 10
    min_plate_aspect_ratio: float = 0.8
    max_plate_aspect_ratio: float = 3.0
    # A plate box touching the camera frame edge is likely physically cut
    # off. Now also backstopped by the plate-format pattern, which rejects
    # a truncated read ("545") on its own shape — so this can be dialed
    # down if it turns out to be costing recall at a gate where vehicles
    # pass close to the frame edge.
    plate_frame_edge_margin_px: int = 2

    # WAS 0.95. Lowered after a 33-vehicle evaluation came back with almost
    # every plate missed. 0.95 was tuned against 3 lucky reads on a single
    # clip and did not survive contact with a real dataset — see the long
    # comment in PipelineRunner.__init__ and the confirmed OCR numbers in
    # PaddleOCRPlateReader.read_candidates. In short: it was being applied
    # to a confidence corrupted by province-band/noise text lines, and even
    # a clean correct read of a real plate here scored 0.9307. With
    # candidate selection fixed, the plate-format pattern is the primary
    # filter and this floor only has to reject plate-shaped garbage
    # (observed 0.17-0.84, vs 0.93-0.9998 for genuine reads).
    ocr_min_confidence: float = 0.50
    ocr_lang: str = "en"

    track_max_age_frames: int = 30
    track_iou_fallback_threshold: float = 0.3

    frame_skip: int = 1  # process every Nth frame (1 = every frame)
