"""Tests for scripts/score_accuracy.py's normalization and matching helpers.

scripts/ isn't a package (no __init__.py, matching run_pipeline.py's own
sibling-script convention) — imported via sys.path manipulation.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from score_accuracy import (  # noqa: E402
    GroundTruthRow,
    PipelineEvent,
    character_error_rate,
    edit_distance,
    find_duplicate_events,
    load_pipeline_events,
    match_events_to_ground_truth,
    normalize_plate,
    normalize_vehicle_type,
    score,
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

    def test_ambiguous_flag_fires_on_one_event_contested_by_two_gt_rows(self):
        """FIX #9: the mirror case the one-sided check missed. GT_A and
        GT_B are each other's ONLY candidate event (both see exactly 1
        option, so the old GT-side-only count would never flag either) —
        but that one event is a candidate for BOTH of them, so whichever
        wins it is still a guess."""
        gt_rows = [
            GroundTruthRow(index=0, filename="c.mp4", timestamp_sec=50.0, vehicle_type="car", plate="AAA111", condition="day", notes=""),
            GroundTruthRow(index=1, filename="c.mp4", timestamp_sec=50.3, vehicle_type="car", plate="BBB222", condition="day", notes=""),
        ]
        events = [
            PipelineEvent(index=0, filename="c.mp4", timestamp_sec=50.1, vehicle_type="car", plate="AAA111", track_id=1),
        ]

        match_events_to_ground_truth(gt_rows, events, time_window=5.0)

        self.assertTrue(gt_rows[0].ambiguous, "only 1 candidate from A's own view, but that event was contested by B too")
        self.assertTrue(gt_rows[1].ambiguous)

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

    def test_large_cluster_still_achieves_maximum_cardinality(self):
        """The specific bug reported: the old backtracking implementation
        fell back to broken greedy above 8 GT rows in one cluster. This
        tiles the same steal-pattern counterexample 10x (20 GT rows total,
        all in one contested cluster) — every pair must still fully match;
        the old size-limited fallback would have dropped some to "Missed"."""
        gt_rows = []
        events = []
        for k in range(10):
            base = k * 100.0  # spaced far apart so pairs don't cross-contest each other
            gt_rows.append(GroundTruthRow(index=2 * k, filename="c.mp4", timestamp_sec=base, vehicle_type="car", plate=f"A{k}", condition="clear", notes=""))
            gt_rows.append(GroundTruthRow(index=2 * k + 1, filename="c.mp4", timestamp_sec=base + 0.3, vehicle_type="car", plate=f"B{k}", condition="clear", notes=""))
            events.append(PipelineEvent(index=2 * k, filename="c.mp4", timestamp_sec=base + 0.25, vehicle_type="car", plate=f"A{k}", track_id=2 * k))
            events.append(PipelineEvent(index=2 * k + 1, filename="c.mp4", timestamp_sec=base + 1.0, vehicle_type="car", plate=f"B{k}", track_id=2 * k + 1))

        assignments = match_events_to_ground_truth(gt_rows, events, time_window=0.9)

        self.assertEqual(len(assignments), 20, "all 20 ground-truth rows must match — none dropped to a false 'Missed'")


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


class TestSourceVideoResolution(unittest.TestCase):
    """Regression tests for the bug that scored a real, correctly-read car
    as 'Missed' (fix #10 in the module docstring).

    Matching groups by (filename, vehicle_type), so getting the source
    video wrong doesn't degrade the score — it zeroes it, silently.
    """

    @staticmethod
    def _write_events(output_dir: Path, json_name: str, events: list[dict]) -> None:
        (output_dir / json_name).write_text(json.dumps(events), encoding="utf-8")

    def test_source_video_field_wins_over_the_json_filename(self):
        """The exact failing case: a run written to `review2_events.json`
        against `dataset_clear_01.mp4`. The old filename heuristic called
        this `review2.mp4` and matched nothing."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write_events(out, "review2_events.json", [{
                "source_video": "dataset_clear_01.mp4",
                "timestamp_sec": 20.0, "vehicle_class": "car",
                "plate_text": "AWJ431", "track_id": 3,
            }])

            events = load_pipeline_events(out)

            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].filename, "dataset_clear_01.mp4")

    def test_correctly_read_car_scores_correct_not_missed(self):
        """End-to-end over the real failure: ground truth for
        dataset_clear_01.mp4, events written to an unrelated JSON name."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write_events(out, "review2_events.json", [{
                "source_video": "dataset_clear_01.mp4",
                "timestamp_sec": 20.4, "vehicle_class": "car",
                "plate_text": "AWJ431", "track_id": 3,
            }])
            gt = [GroundTruthRow(
                index=0, filename="dataset_clear_01.mp4", timestamp_sec=20.0,
                vehicle_type="car", plate="AWJ431", condition="clear", notes="",
            )]

            events = load_pipeline_events(out)
            rows = score(gt, events, match_events_to_ground_truth(gt, events, 5.0))

            self.assertEqual(rows[0].result, "Correct")

    def test_full_path_in_source_video_is_reduced_to_a_basename(self):
        """ground_truth.csv holds bare filenames; a run invoked with
        `--video sample_data/dataset_clear_01.mp4` must still match."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write_events(out, "run_events.json", [{
                "source_video": "sample_data/dataset_clear_01.mp4",
                "timestamp_sec": 20.0, "vehicle_class": "car",
                "plate_text": "AWJ431", "track_id": 3,
            }])

            events = load_pipeline_events(out)

            self.assertEqual(events[0].filename, "dataset_clear_01.mp4")

    def test_falls_back_to_json_filename_for_pre_fix_event_files(self):
        """Events written before `source_video` existed must still load,
        using the old heuristic rather than dropping out entirely."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write_events(out, "dataset_clear_01_events.json", [{
                "timestamp_sec": 20.0, "vehicle_class": "car",
                "plate_text": "AWJ431", "track_id": 3,
            }])

            events = load_pipeline_events(out)

            self.assertEqual(events[0].filename, "dataset_clear_01.mp4")


if __name__ == "__main__":
    unittest.main()
