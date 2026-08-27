"""Tests for PipelineRunner's OCR orchestration (external PR #10 review).

Uses fakes for VehicleDetector/PlateDetector/PlateOCR (no real YOLO/PaddleOCR
model loading) and the real ByteTrackTracker (lightweight, no model loading
in its constructor either) so this runs fast with no ML dependencies.
"""
import tempfile
import unittest

import numpy as np

from anpr.interfaces import BBox, Detection, PlateDetector, PlateOCR, PlateReading, VehicleDetector
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


class FakeMultiCandidateOCR(PlateOCR):
    """A PlateOCR that reports several candidate readings for one crop, the
    way a real multi-line plate crop does (province band + number, or a
    stacked two-line motorbike plate)."""

    def __init__(self, candidates: list[PlateReading]):
        self.candidates = candidates

    def read(self, plate_crop: np.ndarray) -> tuple[str, float]:
        best = self.candidates[0]
        return best.text, best.confidence

    def read_candidates(self, plate_crop: np.ndarray) -> list[PlateReading]:
        return sorted(self.candidates, key=lambda c: (-c.confidence, c.source_lines))


class TestPlateCandidateSelection(unittest.TestCase):
    """Regression tests for the bug that made a 33-vehicle evaluation report
    almost every plate as missed.

    Every candidate set here is taken from real PaddleOCR output on this
    project's own evidence crops — see
    PaddleOCRPlateReader.read_candidates for the raw numbers.
    """

    def _build_runner(self, ocr, evidence_dir, **kwargs):
        return PipelineRunner(
            vehicle_detector=FakeVehicleDetector(kwargs.pop("vehicle_class", "car")),
            plate_detector=FakePlateDetector(),
            plate_ocr=ocr,
            tracker=ByteTrackTracker(fps=20.0),
            evidence_dir=evidence_dir,
            **kwargs,
        )

    def _run(self, ocr, **kwargs):
        with tempfile.TemporaryDirectory() as evidence_dir:
            runner = self._build_runner(ocr, evidence_dir, **kwargs)
            runner._process_frame(_make_frame(), frame_idx=0)
            for track in runner.tracker.finalize_all():
                runner._finalize_track(track)
            return runner._events

    def test_province_band_does_not_destroy_a_perfect_read(self):
        """THE bug. Real crop: rec_texts ['BUV 711', 'JUNJAU'], rec_scores
        [0.9988, 0.6661] — 'JUNJAU' is the PUNJAB band misread. The old
        code joined them into 'BUV711JUNJAU' at confidence 0.6661, which
        failed the format regex AND the confidence floor, discarding a
        perfect 0.9988 read of a real plate."""
        events = self._run(
            FakeMultiCandidateOCR([
                PlateReading("BUV711", 0.9988, 1),
                PlateReading("JUNJAU", 0.6661, 1),
                PlateReading("BUV711JUNJAU", 0.6661, 2),
            ])
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].plate_text, "BUV711")
        self.assertAlmostEqual(events[0].ocr_confidence, 0.9988)

    def test_low_confidence_noise_line_does_not_veto_the_plate(self):
        """Real crop: ['ARK2363', 'n'] scores [0.9909, 0.1747]. Under
        min(scores) a 0.17 speck of noise dragged a 0.99 plate read below
        every threshold."""
        events = self._run(
            FakeMultiCandidateOCR([
                PlateReading("ARK2363", 0.9909, 1),
                PlateReading("N", 0.1747, 1),
                PlateReading("ARK2363N", 0.1747, 2),
            ])
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].plate_text, "ARK2363")

    def test_genuinely_two_line_plate_is_still_joined(self):
        """The other half of the requirement, and why this can't just be
        "take the highest-confidence line". Real crop: ['GAA', '545'] —
        a stacked plate where NEITHER line is a valid plate on its own and
        the join is the correct answer."""
        events = self._run(
            FakeMultiCandidateOCR([
                PlateReading("GAA", 0.9945, 1),
                PlateReading("545", 0.9825, 1),
                PlateReading("GAA545", 0.9825, 2),
            ])
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].plate_text, "GAA545")

    def test_truncated_read_is_rejected_on_format(self):
        """'545' alone (a plate clipped by the frame edge) must not be
        logged as a plate — the format pattern is what backstops the
        detector's edge filter."""
        events = self._run(FakeMultiCandidateOCR([PlateReading("545", 0.9953, 1)]))
        self.assertEqual(len(events), 0)

    def test_correct_read_below_old_095_floor_is_now_accepted(self):
        """A genuine, correct read of a real plate in this project's own
        footage scored 0.9307. The old 0.95 default rejected it."""
        events = self._run(FakeMultiCandidateOCR([PlateReading("BUV711", 0.9307, 1)]))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].plate_text, "BUV711")

    def test_rejection_reasons_are_counted_for_diagnosis(self):
        """The run summary must distinguish "never read" from "read then
        thrown away on a threshold" — the whole reason the original
        failure was so hard to diagnose from the output."""
        with tempfile.TemporaryDirectory() as evidence_dir:
            ocr = FakeMultiCandidateOCR([PlateReading("ENTRANCE", 0.99, 1)])
            runner = self._build_runner(ocr, evidence_dir)
            runner._process_frame(_make_frame(), frame_idx=0)
            self.assertEqual(runner._reject_reasons["format_mismatch"], 1)

    def test_source_video_recorded_on_events(self):
        """score_accuracy.py matches ground truth on this, instead of
        guessing the video from the events-JSON filename."""
        events = self._run(
            FakeMultiCandidateOCR([PlateReading("BUV711", 0.99, 1)]),
            source_video="dataset_clear_01.mp4",
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].source_video, "dataset_clear_01.mp4")
        self.assertEqual(events[0].to_dict()["source_video"], "dataset_clear_01.mp4")


if __name__ == "__main__":
    unittest.main()
