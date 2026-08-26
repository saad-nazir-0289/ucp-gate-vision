"""Pretrained YOLO license-plate detector, run on a cropped region around each vehicle box.

See cv-service/README.md "Plate detector model" for which pretrained model
this targets and its license.
"""
from __future__ import annotations

import logging

import numpy as np
from ultralytics import YOLO

from ..interfaces import BBox, Detection, PlateDetector

logger = logging.getLogger(__name__)


class YoloPlateDetector(PlateDetector):
    def __init__(
        self,
        weights_path: str,
        device: str = "cpu",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        margin_ratio: float = 0.15,
        default_class_name: str = "license_plate",
        frame_edge_margin_px: int = 2,
        min_plate_width_px: int = 24,
        min_plate_height_px: int = 10,
        min_aspect_ratio: float = 1.0,
        max_aspect_ratio: float = 3.0,
    ):
        logger.info("Loading plate detector weights=%s device=%s", weights_path, device)
        self.model = YOLO(weights_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.margin_ratio = margin_ratio
        self.default_class_name = default_class_name
        # A plate box touching the camera frame's own edge (not the vehicle
        # crop's edge — those are different boundaries) means the plate is
        # very likely physically cut off, not just tightly cropped. Confirmed
        # on real footage: a plate crop clipped by the bottom frame edge
        # scored a *higher* OCR confidence (0.9999) on the truncated text
        # "988" than the correct, complete "AAL988" read from an earlier
        # frame (0.94) — the aggregator's highest-confidence-wins logic then
        # kept the wrong, incomplete one. Rejecting edge-touching plate boxes
        # here stops that at the source rather than patching it in dedup.
        self.frame_edge_margin_px = frame_edge_margin_px
        # A box too small to physically contain a legible plate. Confirmed on
        # real footage: a 20x12px box (background noise on a distant/small
        # vehicle) scored a plausible-looking OCR read ("1", conf 0.80) even
        # though no plate could possibly be legible at that size — the
        # smallest genuine plate box observed in that same footage was
        # ~51x33px (a distant motorcycle plate), so these defaults leave
        # headroom below real plates while rejecting obvious noise. Tune down
        # if your camera setup legitimately produces smaller legible plates.
        self.min_plate_width_px = min_plate_width_px
        self.min_plate_height_px = min_plate_height_px
        # Real plates are a fairly consistent width:height ratio; non-plate
        # rectangular text (signage, banners) often isn't. Confirmed on real
        # footage: every genuine plate box measured so far has ratio ~1.4-1.9
        # (e.g. AAL988 1.54, BUV711 1.86, a motorcycle plate 1.72-1.55), while
        # a false positive on an "Entrance" sign — correctly boxed and
        # correctly OCR'd as text, just not a plate — measured 4.47 (259x58px).
        # Defaults leave generous headroom above/below the observed real
        # range rather than tightly fitting it, since this is a small sample.
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

    def detect(self, frame: np.ndarray, vehicle_box: BBox) -> list[Detection]:
        frame_h, frame_w = frame.shape[:2]
        crop, offset_x, offset_y = self._crop_with_margin(frame, vehicle_box)
        if crop is None or crop.size == 0:
            return []

        results = self.model.predict(
            crop,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
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
        names = getattr(r, "names", None) or {}
        m = self.frame_edge_margin_px

        for box, conf, cid in zip(boxes_xyxy, confs, cls_ids):
            x1, y1, x2, y2 = box.tolist()
            abs_x1, abs_y1, abs_x2, abs_y2 = x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y

            if abs_x1 <= m or abs_y1 <= m or abs_x2 >= frame_w - m or abs_y2 >= frame_h - m:
                logger.debug(
                    "Skipping plate detection touching frame edge (likely truncated): "
                    "bbox=(%.0f,%.0f,%.0f,%.0f) frame=%dx%d",
                    abs_x1, abs_y1, abs_x2, abs_y2, frame_w, frame_h,
                )
                continue

            box_w, box_h = abs_x2 - abs_x1, abs_y2 - abs_y1
            if box_w < self.min_plate_width_px or box_h < self.min_plate_height_px:
                logger.debug(
                    "Skipping plate detection too small to be legible: %.0fx%.0fpx (min %dx%dpx)",
                    box_w, box_h, self.min_plate_width_px, self.min_plate_height_px,
                )
                continue

            aspect_ratio = box_w / box_h if box_h > 0 else float("inf")
            if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                logger.debug(
                    "Skipping plate detection with non-plate aspect ratio: %.0fx%.0fpx ratio=%.2f "
                    "(allowed %.1f-%.1f)",
                    box_w, box_h, aspect_ratio, self.min_aspect_ratio, self.max_aspect_ratio,
                )
                continue

            class_name = names.get(int(cid), self.default_class_name) if isinstance(names, dict) else self.default_class_name
            detections.append(
                Detection(
                    bbox=BBox(abs_x1, abs_y1, abs_x2, abs_y2),
                    confidence=float(conf),
                    class_id=int(cid),
                    class_name=class_name,
                )
            )

        # Sort so callers taking detections[0] get the right vehicle's own
        # plate, not just the highest-confidence plate anywhere in the
        # margin-expanded crop. FIX (per PR #10 external review): in a
        # crowded scene, the margin region around vehicle A's box can
        # contain vehicle B's plate — if B's plate happens to score higher
        # confidence than A's own (partially occluded/angled) plate, the old
        # "highest confidence wins" sort would hand A's track vehicle B's
        # plate text. A plate whose center actually falls inside the
        # vehicle's own (unexpanded) box is preferred regardless of
        # confidence; only if none do do we fall back to the margin region,
        # highest-confidence first.
        def _sort_key(d: Detection) -> tuple[bool, float]:
            cx, cy = (d.bbox.x1 + d.bbox.x2) / 2, (d.bbox.y1 + d.bbox.y2) / 2
            inside_vehicle_box = vehicle_box.x1 <= cx <= vehicle_box.x2 and vehicle_box.y1 <= cy <= vehicle_box.y2
            return (not inside_vehicle_box, -d.confidence)  # False (inside) sorts before True; then by confidence desc

        detections.sort(key=_sort_key)
        if detections:
            logger.debug(
                "Plate stage: %d candidate(s) for vehicle_box=%s, picked %s (conf=%.2f)",
                len(detections), vehicle_box.to_int_tuple(), detections[0].bbox.to_int_tuple(), detections[0].confidence,
            )
        return detections

    def _crop_with_margin(self, frame: np.ndarray, vehicle_box: BBox):
        h, w = frame.shape[:2]
        mx = vehicle_box.width * self.margin_ratio
        my = vehicle_box.height * self.margin_ratio
        x1 = max(0, int(vehicle_box.x1 - mx))
        y1 = max(0, int(vehicle_box.y1 - my))
        x2 = min(w, int(vehicle_box.x2 + mx))
        y2 = min(h, int(vehicle_box.y2 + my))
        if x2 <= x1 or y2 <= y1:
            return None, 0, 0
        return frame[y1:y2, x1:x2], x1, y1
