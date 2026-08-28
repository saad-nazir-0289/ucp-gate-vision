"""
Abstract pipeline interfaces — docs/ARCHITECTURE.md section 5 (Modularity
Requirement).

Each stage (VehicleDetector, PlateDetector, PlateOCR, Tracker) is defined
here as an ABC. Concrete implementations live under anpr/detectors/,
anpr/tracking/, anpr/ocr/ and are swapped via config (see anpr/config.py
and run_pipeline.py CLI flags), never by editing pipeline code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BBox:
    """Axis-aligned bounding box in full-frame pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_int_tuple(self) -> tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))


def iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two boxes, 0.0 if they don't overlap."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Detection:
    """A single per-frame detection, pre-tracking."""

    bbox: BBox
    confidence: float
    class_id: int
    class_name: str
    # Populated only when the detector's concrete implementation performs
    # joint detection+tracking internally (see YoloVehicleDetector's
    # docstring for why that pairing exists for the Ultralytics/ByteTrack
    # combo specifically). Left as None for detectors that only detect.
    track_id: int | None = None


@dataclass
class TrackFrame:
    """One track's state at one processed frame — the unit Tracker history is built from."""

    frame_idx: int
    timestamp_sec: float
    bbox: BBox
    confidence: float


@dataclass
class Track:
    """A vehicle's identity persisted across frames, with its position history.

    The history buffer is exactly what Phase 2's direction inference
    (docs/ARCHITECTURE.md section 7 — scale-trend + displacement-vector)
    will consume; that's why Tracker owns temporal state rather than being a
    stateless per-frame ID slapper.
    """

    track_id: int
    class_name: str
    history: list[TrackFrame] = field(default_factory=list)
    last_seen_frame: int = -1
    active: bool = True

    @property
    def latest(self) -> TrackFrame | None:
        return self.history[-1] if self.history else None


class VehicleDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect vehicles (cars/motorcycles) in a single BGR frame."""
        raise NotImplementedError


class PlateDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray, vehicle_box: BBox) -> list[Detection]:
        """Detect license-plate region(s) within/near a given vehicle box."""
        raise NotImplementedError


@dataclass
class PlateReading:
    """One candidate interpretation of a plate crop's text.

    A plate crop routinely contains more than just the plate number: local
    plates carry a province/city band ("PUNJAB"), and OCR also picks up
    stray noise fragments. Each of those comes back as its own text line,
    so "what does this crop say" has several possible answers, not one.
    `source_lines` records how many OCR text lines were concatenated to
    form this candidate (1 for a single line, 2+ for a genuinely two-line
    plate whose letters and digits are stacked).
    """

    text: str
    confidence: float
    source_lines: int = 1


class PlateOCR(ABC):
    @abstractmethod
    def read(self, plate_crop: np.ndarray) -> tuple[str, float]:
        """Read the plate string from a cropped plate image. Returns (text, confidence)."""
        raise NotImplementedError

    def read_candidates(self, plate_crop: np.ndarray) -> list[PlateReading]:
        """Every plausible reading of this crop, best-confidence first.

        Exists because a single (text, confidence) pair can't express a
        multi-line crop faithfully — see PlateReading. The caller
        (PipelineRunner) picks between candidates using its own
        plate-format policy, keeping the accept/reject decision out of the
        OCR layer exactly as `read()`'s contract already does.

        Default implementation wraps `read()`, so a PlateOCR that only
        knows how to return one answer keeps working unchanged.
        """
        text, confidence = self.read(plate_crop)
        return [PlateReading(text=text, confidence=confidence)] if text else []


class Tracker(ABC):
    @abstractmethod
    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        """Assign/maintain persistent track IDs for this frame's detections."""
        raise NotImplementedError
