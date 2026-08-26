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
    find_duplicate_events,
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

    def test_greedy_would_fail_this_case_exact_matching_must_not(self):
        """Concrete counterexample (FIX #7): GT_A's only candidate is also
        GT_B's closest (but not only) candidate. Global closest-distance-
        first greedy grabs (B, E1) first since it's the single smallest
        distance overall, leaving A with nothing — 1 match total, a false
        "Missed" for A. The correct assignment gives A its only option and
        B its second-best, matching both.

        Numbers: window=0.9. dist(A,E1)=0.25 (A's ONLY candidate — E2 is
        1.0 away, outside the window). dist(B,E1)=0.05 (globally smallest —
        what greedy grabs first). dist(B,E2)=0.7 (B's fallback, still
        within window)."""
        gt_rows = [
            GroundTruthRow(index=0, filename="c.mp4", timestamp_sec=10.0, vehicle_type="car", plate="AAA111", condition="clear", notes=""),
            GroundTruthRow(index=1, filename="c.mp4", timestamp_sec=10.3, vehicle_type="car", plate="BBB222", condition="clear", notes=""),
        ]
        events = [
            PipelineEvent(index=0, filename="c.mp4", timestamp_sec=10.25, vehicle_type="car", plate="AAA111", track_id=1),  # E1
            PipelineEvent(index=1, filename="c.mp4", timestamp_sec=11.0, vehicle_type="car", plate="BBB222", track_id=2),  # E2
        ]

        assignments = match_events_to_ground_truth(gt_rows, events, time_window=0.9)

        self.assertEqual(len(assignments), 2, "both GT rows must be matched — greedy would leave GT_A ('Missed') with only 1 match total")
        self.assertEqual(events[assignments[0]].track_id, 1, "GT_A must get its only candidate, E1")
        self.assertEqual(events[assignments[1]].track_id, 2, "GT_B must get its remaining option, E2")


class TestFindDuplicateEvents(unittest.TestCase):
    def _ev(self, index, plate, t, track_id):
        return PipelineEvent(index=index, filename="c.mp4", timestamp_sec=t, vehicle_type="car", plate=plate, track_id=track_id)

    def test_same_plate_within_window_is_flagged_exact(self):
        events = [self._ev(0, "AAA111", 10.0, 1), self._ev(1, "AAA111", 12.0, 2)]

        exact, near = find_duplicate_events(events, window_sec=5.0)

        self.assertEqual(len(exact), 1)
        self.assertEqual(near, [])

    def test_legitimate_quick_return_outside_default_window_not_flagged(self):
        """FIX #8: the original 15s window would have flagged a vehicle
        legitimately reappearing (e.g. entering, realizing it's the wrong
        gate, immediately driving back out) 12s later as a duplicate. The
        new 5s default doesn't reach that far."""
        events = [self._ev(0, "AAA111", 10.0, 1), self._ev(1, "AAA111", 22.0, 2)]

        exact, near = find_duplicate_events(events, window_sec=5.0)

        self.assertEqual(exact, [], "12s apart is outside the 5s default window — not flagged")
        self.assertEqual(near, [])

    def test_near_miss_plate_reported_separately_from_exact(self):
        """A 1-character-different plate could be OCR noise on the same
        pass, or two different, coincidentally similar plates — reported as
        'near', never conflated with confirmed-same-text 'exact' matches."""
        events = [self._ev(0, "AAA111", 10.0, 1), self._ev(1, "AAA112", 11.0, 2)]

        exact, near = find_duplicate_events(events, window_sec=5.0, max_near_edit_distance=1)

        self.assertEqual(exact, [])
        self.assertEqual(len(near), 1)

    def test_different_plates_beyond_edit_distance_not_flagged_at_all(self):
        events = [self._ev(0, "AAA111", 10.0, 1), self._ev(1, "ZZZ999", 11.0, 2)]

        exact, near = find_duplicate_events(events, window_sec=5.0, max_near_edit_distance=1)

        self.assertEqual(exact, [])
        self.assertEqual(near, [])


if __name__ == "__main__":
    unittest.main()
