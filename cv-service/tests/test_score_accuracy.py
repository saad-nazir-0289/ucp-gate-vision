"""Tests for scripts/score_accuracy.py's normalization and matching helpers.

scripts/ isn't a package (no __init__.py, matching run_pipeline.py's own
sibling-script convention) — imported via sys.path manipulation.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from score_accuracy import (  # noqa: E402
    GroundTruthRow,
    PipelineEvent,
    character_error_rate,
    edit_distance,
    match_events_to_ground_truth,
    normalize_plate,
    normalize_vehicle_type,
)


class TestNormalizeVehicleType(unittest.TestCase):
    def test_bike_variants_fold_to_bike(self):
        for value in ("bike", "BIKE", "motorcycle", "Motorcycle", "motorbike"):
            self.assertEqual(normalize_vehicle_type(value), "bike")

    def test_car_truck_bus_fold_to_car(self):
        """The fix confirmed necessary by testing: a real car classified as
        'truck' by the vehicle detector must still match a ground-truth row
        labeled 'car', or a correct read scores as a false 'Missed'."""
        for value in ("car", "CAR", "truck", "Truck", "bus", "van", "suv"):
            self.assertEqual(normalize_vehicle_type(value), "car")

    def test_unknown_type_passes_through(self):
        self.assertEqual(normalize_vehicle_type("rickshaw"), "rickshaw")

    def test_blank_or_none(self):
        self.assertEqual(normalize_vehicle_type(""), "")
        self.assertEqual(normalize_vehicle_type(None), "")


class TestNormalizePlate(unittest.TestCase):
    def test_strips_spaces_and_uppercases(self):
        self.assertEqual(normalize_plate("aal 988"), "AAL988")
        self.assertEqual(normalize_plate(" AZF  441 "), "AZF441")

    def test_blank_or_none(self):
        self.assertEqual(normalize_plate(""), "")
        self.assertEqual(normalize_plate(None), "")


class TestEditDistanceAndCER(unittest.TestCase):
    def test_identical_strings(self):
        self.assertEqual(edit_distance("AAL988", "AAL988"), 0)
        self.assertEqual(character_error_rate("AAL988", "AAL988"), 0.0)

    def test_real_example_from_evaluation(self):
        # LEM2025 -> LEH2024: two substitutions (M->H, 5->4).
        self.assertEqual(edit_distance("LEM2025", "LEH2024"), 2)
        self.assertAlmostEqual(character_error_rate("LEM2025", "LEH2024"), 2 / 7, places=6)

    def test_cer_is_none_for_blank_ground_truth(self):
        self.assertIsNone(character_error_rate("", "ANYTHING"))


class TestMatchEventsToGroundTruth(unittest.TestCase):
    def test_ambiguous_flag_only_fires_on_genuine_contention(self):
        """Regression test for a real bug caught during development: an
        earlier version flagged ambiguity from raw GT-to-GT timestamp gaps
        (2x the window), which fired on any two same-type vehicles passing
        within ~10s of each other - i.e. almost all normal traffic. This
        checks the fixed, contention-based version doesn't do that."""
        gt_rows = [
            GroundTruthRow(index=0, filename="c.mp4", timestamp_sec=10, vehicle_type="car", plate="AAA111", condition="clear", notes=""),
            GroundTruthRow(index=1, filename="c.mp4", timestamp_sec=18, vehicle_type="car", plate="BBB222", condition="clear", notes=""),
        ]
        events = [
            PipelineEvent(index=0, filename="c.mp4", timestamp_sec=10.1, vehicle_type="car", plate="AAA111", track_id=1),
            PipelineEvent(index=1, filename="c.mp4", timestamp_sec=18.1, vehicle_type="car", plate="BBB222", track_id=2),
        ]

        match_events_to_ground_truth(gt_rows, events, time_window=5.0)

        self.assertFalse(gt_rows[0].ambiguous, "8s apart with one clean match each is normal traffic, not ambiguous")
        self.assertFalse(gt_rows[1].ambiguous)

    def test_ambiguous_flag_fires_on_real_contention(self):
        gt_rows = [
            GroundTruthRow(index=0, filename="c.mp4", timestamp_sec=50, vehicle_type="bike", plate="AAA111", condition="day", notes=""),
        ]
        events = [
            PipelineEvent(index=0, filename="c.mp4", timestamp_sec=50.2, vehicle_type="bike", plate="AAA111", track_id=1),
            PipelineEvent(index=1, filename="c.mp4", timestamp_sec=51.0, vehicle_type="bike", plate="CCC333", track_id=2),
        ]

        assignments = match_events_to_ground_truth(gt_rows, events, time_window=5.0)

        self.assertTrue(gt_rows[0].ambiguous, "two candidates competed for one GT row — this is genuine contention")
        self.assertEqual(events[assignments[0]].plate, "AAA111", "closest candidate should still win")


if __name__ == "__main__":
    unittest.main()
