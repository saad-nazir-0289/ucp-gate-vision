"""Tests for the crowded-scene plate-selection fix (external PR #10 review).

Uses sort_plate_candidates() directly rather than YoloPlateDetector.detect()
so no real YOLO model needs loading — see the function's own docstring in
anpr/detectors/plate_detector.py for why it was pulled out standalone.
"""
import unittest

from anpr.detectors.plate_detector import sort_plate_candidates
from anpr.interfaces import BBox, Detection


class TestSortPlateCandidates(unittest.TestCase):
    def test_lower_confidence_inside_plate_beats_higher_confidence_neighbor(self):
        """The exact scenario from the review: vehicle A's margin-expanded
        crop contains vehicle B's plate too. B's plate scores higher
        confidence, but it's centered outside A's own box — A's own
        (lower-confidence) plate must still be picked first."""
        vehicle_box = BBox(x1=100, y1=100, x2=200, y2=200)

        own_plate_low_conf = Detection(
            bbox=BBox(x1=140, y1=180, x2=170, y2=195),  # center (155, 187.5) — inside vehicle_box
            confidence=0.30,
            class_id=0,
            class_name="license_plate",
        )
        neighbor_plate_high_conf = Detection(
            bbox=BBox(x1=210, y1=180, x2=240, y2=195),  # center (225, 187.5) — outside vehicle_box
            confidence=0.95,
            class_id=0,
            class_name="license_plate",
        )

        result = sort_plate_candidates([neighbor_plate_high_conf, own_plate_low_conf], vehicle_box)

        self.assertIs(result[0], own_plate_low_conf, "own vehicle's plate must win regardless of confidence")
        self.assertIs(result[1], neighbor_plate_high_conf)

    def test_falls_back_to_highest_confidence_when_none_inside(self):
        """If no candidate's center falls inside the vehicle's own box
        (e.g. the plate is only visible in the margin region), fall back to
        highest confidence — better than returning nothing."""
        vehicle_box = BBox(x1=100, y1=100, x2=200, y2=200)
        low = Detection(bbox=BBox(x1=210, y1=180, x2=240, y2=195), confidence=0.40, class_id=0, class_name="license_plate")
        high = Detection(bbox=BBox(x1=210, y1=210, x2=240, y2=225), confidence=0.80, class_id=0, class_name="license_plate")

        result = sort_plate_candidates([low, high], vehicle_box)

        self.assertIs(result[0], high)

    def test_single_candidate_inside_box(self):
        vehicle_box = BBox(x1=0, y1=0, x2=100, y2=100)
        only = Detection(bbox=BBox(x1=40, y1=40, x2=60, y2=60), confidence=0.5, class_id=0, class_name="license_plate")

        result = sort_plate_candidates([only], vehicle_box)

        self.assertEqual(result, [only])

    def test_empty_input(self):
        vehicle_box = BBox(x1=0, y1=0, x2=100, y2=100)
        self.assertEqual(sort_plate_candidates([], vehicle_box), [])


if __name__ == "__main__":
    unittest.main()
