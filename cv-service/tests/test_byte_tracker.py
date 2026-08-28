"""Tests for ByteTrackTracker, particularly the fake-fallback-track bug
(external PR #10 review, confirmed on real logs before fixing) and the
track-merge reconciliation.

No real video/model needed — ByteTrackTracker only manages Track/Detection
dataclasses, no ML model loading in its constructor.
"""
import unittest

from anpr.interfaces import BBox, Detection
from anpr.tracking.byte_tracker import ByteTrackTracker


def _det(track_id, x1=100.0, confidence=0.5, class_name="car"):
    return Detection(
        bbox=BBox(x1=x1, y1=100.0, x2=x1 + 50, y2=150.0),
        confidence=confidence,
        class_id=2,
        class_name=class_name,
        track_id=track_id,
    )


class TestDiscardUntrackedDetections(unittest.TestCase):
    """The critical fix: track_id=None from a self-tracking VehicleDetector
    means ByteTrack itself refused to start a track (below its own
    new_track_thresh) — it must not become a fallback track."""

    def test_id_less_detection_is_dropped_by_default(self):
        tracker = ByteTrackTracker(fps=20.0)  # discard_untracked_detections=True by default

        tracks = tracker.update([_det(track_id=None, confidence=0.15)], frame=None)

        self.assertEqual(tracks, [], "an ID-less detection must not become a track when discarding is on")
        self.assertEqual(tracker.get_stale_tracks(), [])

    def test_real_track_id_still_works_alongside_discarding(self):
        tracker = ByteTrackTracker(fps=20.0)

        tracks = tracker.update([_det(track_id=5, confidence=0.9)], frame=None)

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, 5)

    def test_fallback_matching_still_available_when_explicitly_enabled(self):
        """The IoU fallback isn't deleted — it's opt-in for a hypothetical
        VehicleDetector that genuinely never assigns track_ids itself."""
        tracker = ByteTrackTracker(fps=20.0, discard_untracked_detections=False)

        tracks = tracker.update([_det(track_id=None, confidence=0.9)], frame=None)

        self.assertEqual(len(tracks), 1)
        self.assertGreaterEqual(tracks[0].track_id, 1_000_000, "should get a fallback ID, not be dropped")


class TestTrackMerge(unittest.TestCase):
    def test_fallback_then_real_id_merges_into_real_id(self):
        """Reproduces the AAL988 case: a track starts under a fallback ID
        (ByteTrack hadn't confirmed it yet), then a real ByteTrack ID
        appears for the same physical box — they must merge into one track,
        preferring the real ID as canonical.

        Uses discard_untracked_detections=False specifically to construct
        the fallback-track precondition and test _merge()/_reconcile_id()
        in isolation — not a recommendation to disable discarding in real
        use, which the constructor now defaults to True for."""
        tracker = ByteTrackTracker(fps=20.0, discard_untracked_detections=False, merge_iou_threshold=0.5)

        # Frame 1: unconfirmed, gets a fallback ID via IoU matching path.
        tracker.update([_det(track_id=None, x1=100.0)], frame=None)
        # Frame 2: same box, now a real ByteTrack ID appears.
        tracks = tracker.update([_det(track_id=4, x1=101.0)], frame=None)

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, 4, "real ByteTrack ID should be canonical")
        self.assertEqual(len(tracks[0].history), 2, "history from both frames should be merged, not lost")

        aliases = tracker.pop_new_aliases()
        self.assertEqual(len(aliases), 1)
        old_id, new_id = aliases[0]
        self.assertEqual(new_id, 4)
        self.assertGreaterEqual(old_id, 1_000_000)

    def test_two_different_vehicles_do_not_merge(self):
        tracker = ByteTrackTracker(fps=20.0)

        tracker.update([_det(track_id=1, x1=100.0)], frame=None)
        tracks = tracker.update([_det(track_id=2, x1=500.0)], frame=None)  # far away, no overlap

        self.assertEqual({t.track_id for t in tracks}, {2})
        self.assertEqual(tracker.pop_new_aliases(), [])


if __name__ == "__main__":
    unittest.main()
