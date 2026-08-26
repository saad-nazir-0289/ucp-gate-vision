"""Tests for PipelineRunner's OCR orchestration (external PR #10 review).

Uses fakes for VehicleDetector/PlateDetector/PlateOCR (no real YOLO/PaddleOCR
model loading) and the real ByteTrackTracker (lightweight, no model loading
in its constructor either) so this runs fast with no ML dependencies.
"""
import tempfile
import unittest

import numpy as np

from anpr.interfaces import BBox, Detection, PlateDetector, PlateOCR, VehicleDetector
from anpr.pipeline.runner import PipelineRunner
from anpr.tracking.byte_tracker import ByteTrackTracker


class FakeVehicleDetector(VehicleDetector):
    """Always reports one confirmed track for a single vehicle."""

    def __init__(self, class_name: str = "car"):
        self.class_name = class_name

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return [Detection(bbox=BBox(10, 10, 60, 60), confidence=0.9, class_id=2, class_name=self.class_name, track_id=1)]


class FakePlateDetector(PlateDetector):
    def detect(self, frame: np.ndarray, vehicle_box: BBox) -> list[Detection]:
        return [Detection(bbox=BBox(20, 40, 50, 55), confidence=0.9, class_id=0, class_name="license_plate")]


class FakePlateOCR(PlateOCR):
    """Records how many times read() is called — regression guard against
    the old design's "call, catch TypeError, call again" retry pattern."""

    def __init__(self, text: str, confidence: float):
        self.text = text
        self.confidence = confidence
        self.call_count = 0

    def read(self, plate_crop: np.ndarray) -> tuple[str, float]:
        self.call_count += 1
        return self.text, self.confidence


def _make_frame() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


class TestOCRCalledOnce(unittest.TestCase):
    def _build_runner(self, ocr, evidence_dir, **kwargs):
        return PipelineRunner(
            vehicle_detector=FakeVehicleDetector(kwargs.pop("vehicle_class", "car")),
            plate_detector=FakePlateDetector(),
            plate_ocr=ocr,
            tracker=ByteTrackTracker(fps=20.0),
            evidence_dir=evidence_dir,
            **kwargs,
        )

    def test_ocr_called_exactly_once_per_plate_crop(self):
        """The old design called read() with a kwarg, caught TypeError, and
        called again as a fallback. The current design (OCR returns raw
        text/confidence unconditionally, runner decides acceptance) should
        never need a second call."""
        with tempfile.TemporaryDirectory() as evidence_dir:
            ocr = FakePlateOCR(text="AAL988", confidence=0.99)
            runner = self._build_runner(ocr, evidence_dir)

            runner._process_frame(_make_frame(), frame_idx=0)

            self.assertEqual(ocr.call_count, 1)

    def test_motorcycle_override_accepts_lower_confidence_without_extra_ocr_call(self):
        """The actual diagnostic this override exists for (PR #10's
        motorcycle-accuracy finding): a 0.80-confidence read must be
        accepted for a motorcycle when overridden to 0.75, rejected by the
        car-tuned 0.95 default — and OCR is still only called once."""
        with tempfile.TemporaryDirectory() as evidence_dir:
            ocr = FakePlateOCR(text="ARK2363", confidence=0.80)
            runner = self._build_runner(
                ocr, evidence_dir,
                vehicle_class="motorcycle",
                ocr_min_confidence=0.95,
                ocr_min_confidence_by_class={"motorcycle": 0.75},
            )

            runner._process_frame(_make_frame(), frame_idx=0)
            for track in runner.tracker.finalize_all():
                runner._finalize_track(track)

            self.assertEqual(ocr.call_count, 1)
            self.assertEqual(len(runner._events), 1)
            self.assertEqual(runner._events[0].plate_text, "ARK2363")

    def test_same_confidence_rejected_without_class_override(self):
        """Same 0.80-confidence read, same vehicle class, but no per-class
        override configured — must be rejected against the 0.95 default."""
        with tempfile.TemporaryDirectory() as evidence_dir:
            ocr = FakePlateOCR(text="ARK2363", confidence=0.80)
            runner = self._build_runner(ocr, evidence_dir, vehicle_class="motorcycle", ocr_min_confidence=0.95)

            runner._process_frame(_make_frame(), frame_idx=0)
            for track in runner.tracker.finalize_all():
                runner._finalize_track(track)

            self.assertEqual(ocr.call_count, 1)
            self.assertEqual(len(runner._events), 0, "0.80 must not clear the unmodified 0.95 default")

    def test_non_plate_format_text_rejected(self):
        """DEFAULT_PLATE_PATTERN rejection now happens in the runner, not
        PlateOCR — confirm it still works after moving."""
        with tempfile.TemporaryDirectory() as evidence_dir:
            ocr = FakePlateOCR(text="ENTRANCE", confidence=0.99)
            runner = self._build_runner(ocr, evidence_dir, ocr_min_confidence=0.30)

            runner._process_frame(_make_frame(), frame_idx=0)
            for track in runner.tracker.finalize_all():
                runner._finalize_track(track)

            self.assertEqual(len(runner._events), 0, "pure-letter text must fail the letters-then-digits format check")


if __name__ == "__main__":
    unittest.main()
