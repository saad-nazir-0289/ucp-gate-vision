"""Frame-annotation helpers for the output video (vehicle boxes + plate text overlay)."""
from __future__ import annotations

import cv2
import numpy as np

from ..interfaces import BBox

VEHICLE_COLOR = (60, 200, 60)  # BGR
PLATE_COLOR = (60, 140, 255)


def draw_vehicle(frame: np.ndarray, bbox: BBox, track_id: int, class_name: str, confidence: float) -> None:
    x1, y1, x2, y2 = bbox.to_int_tuple()
    cv2.rectangle(frame, (x1, y1), (x2, y2), VEHICLE_COLOR, 2)
    _draw_label(frame, f"#{track_id} {class_name} {confidence:.2f}", x1, y1, VEHICLE_COLOR, above=True)


def draw_plate(frame: np.ndarray, bbox: BBox, text: str, confidence: float) -> None:
    x1, y1, x2, y2 = bbox.to_int_tuple()
    cv2.rectangle(frame, (x1, y1), (x2, y2), PLATE_COLOR, 2)
    _draw_label(frame, f"{text or '?'} {confidence:.2f}", x1, y2, PLATE_COLOR, above=False)


def _draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int], above: bool) -> None:
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ty = (y - 6) if above else (y + th + 6)
    ty = max(th + 2, ty)
    cv2.rectangle(frame, (x, ty - th - 4), (x + tw + 4, ty + baseline), color, -1)
    cv2.putText(frame, text, (x + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
