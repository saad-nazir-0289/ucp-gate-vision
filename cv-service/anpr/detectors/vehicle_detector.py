"""Ultralytics YOLOv8 vehicle detector, paired with ByteTrack via .track()."""
from __future__ import annotations

import logging

import numpy as np
from ultralytics import YOLO

from ..config import COCO_CAR, COCO_MOTORCYCLE
from ..interfaces import BBox, Detection, VehicleDetector

logger = logging.getLogger(__name__)


class YoloVehicleDetector(VehicleDetector):
    """Vehicle detector + persistent track-ID assignment.

    Design note — why this class calls .track() instead of .predict():
    Ultralytics only exposes ByteTrack through the high-level `YOLO.track()`
    API, which fuses detection and tracking into one call (that's how the
    library's internals are built — the tracker consumes the raw per-frame
    inference output directly, it isn't a decoupled second pass over
    external boxes). Per the spec's instruction to use "ByteTrack via
    Ultralytics' built-in .track() method", this class calls .track()
    internally and attaches the resulting track ID onto each Detection via
    the optional `track_id` field, since it comes for free from that call.

    This does NOT make the Tracker abstraction fake: `ByteTrackTracker`
    (anpr/tracking/byte_tracker.py) still does real, swappable work — it
    owns per-track history/state management (needed for Phase 2 direction
    inference) and falls back to its own IoU-based ID assignment for any
    Detection that arrives without a track_id (e.g. if a future
    VehicleDetector implementation only wraps .predict()). This design
    avoids running the YOLO forward pass twice per frame (once for
    detection, once for tracking) while keeping both interfaces genuinely
    independently replaceable.
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        device: str = "cpu",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        tracker_cfg: str = "bytetrack.yaml",
        classes: tuple[int, ...] = (COCO_CAR, COCO_MOTORCYCLE),
    ):
        logger.info("Loading vehicle detector weights=%s device=%s", weights, device)
        self.model = YOLO(weights)
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tracker_cfg = tracker_cfg
        self.classes = list(classes)
        names = getattr(self.model, "names", {}) or {}
        self._class_names = {cid: names.get(cid, str(cid)) for cid in self.classes}

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.track(
            frame,
            persist=True,
            classes=self.classes,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            tracker=self.tracker_cfg,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        if not results:
            return detections
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        raw_ids = r.boxes.id
        track_ids = raw_ids.cpu().numpy().astype(int) if raw_ids is not None else [None] * len(boxes_xyxy)

        for box, conf, cid, tid in zip(boxes_xyxy, confs, cls_ids, track_ids):
            x1, y1, x2, y2 = box.tolist()
            detections.append(
                Detection(
                    bbox=BBox(x1, y1, x2, y2),
                    confidence=float(conf),
                    class_id=int(cid),
                    class_name=self._class_names.get(int(cid), str(int(cid))),
                    track_id=int(tid) if tid is not None else None,
                )
            )
        return detections
