"""Tracker implementation pairing with YoloVehicleDetector's ByteTrack-via-.track() IDs.

See YoloVehicleDetector's docstring for why ID assignment happens inside the
detector call. This class's job is temporal state: building/maintaining
per-track history (consumed by Phase 2's direction inference), expiring
stale tracks, discarding detections ByteTrack itself declined to track
(see `discard_untracked_detections`), and reconciling a track whose ID
changed mid-pass (see `_reconcile_id`).

Known architectural compromise (external review, acknowledged not fixed):
the "textbook correct" design would have VehicleDetector call `.predict()`
only and this Tracker own ByteTrack entirely via Ultralytics' lower-level
`BYTETracker` class, so a `None` track_id unambiguously means "needs
assignment." The current pairing — VehicleDetector does its own tracking
via the high-level `.track()` API — means this Tracker has to interpret
what a missing track_id means, which is fragile (see the fix in
`discard_untracked_detections`). Not restructured now: `.track()` is the
stable, documented Ultralytics API; `BYTETracker` is lower-level/semi-
private and was deliberately avoided earlier for that reason. Worth
revisiting in Phase 2 if this class of bug recurs.
"""
from __future__ import annotations

import logging

import numpy as np

from ..interfaces import BBox, Detection, Track, TrackFrame, Tracker, iou

logger = logging.getLogger(__name__)

FALLBACK_ID_START = 1_000_000  # kept far from real ByteTrack IDs to avoid collisions


class ByteTrackTracker(Tracker):
    def __init__(
        self,
        fps: float = 30.0,
        max_age_frames: int = 30,
        iou_fallback_threshold: float = 0.3,
        merge_iou_threshold: float = 0.5,
        history_maxlen: int = 600,
        discard_untracked_detections: bool = True,
    ):
        self.fps = fps or 30.0
        self.max_age_frames = max_age_frames
        self.iou_fallback_threshold = iou_fallback_threshold
        self.merge_iou_threshold = merge_iou_threshold
        self.history_maxlen = history_maxlen
        # CRITICAL FIX (external review, verified against real logs before
        # merging): when paired with YoloVehicleDetector (which does its own
        # tracking via .track()), a detection with track_id=None doesn't
        # mean "this detector never assigns IDs" — it means ByteTrack looked
        # at this exact detection and refused to start a track for it
        # (below its own new_track_thresh, default 0.25). The IoU fallback
        # below was originally meant for a hypothetical future
        # VehicleDetector that never tracks at all, where None really would
        # mean "needs ID assignment." Routing ByteTrack's *rejected*
        # detections through that same fallback silently un-rejects them —
        # confirmed on real footage: lowering vehicle_conf to 0.10 (to let
        # ByteTrack's own low-confidence recovery work) let genuine noise
        # through the detector, ByteTrack correctly refused to track it
        # (returned track_id=None), and this fallback tracked it anyway,
        # producing fallback IDs 1000000+ in the logs and 11 tracks for 3
        # real vehicles in one test run. Default True: discard detections
        # with no track_id instead of fallback-tracking them. Only set
        # False if paired with a VehicleDetector that genuinely never
        # assigns track_ids itself (the fallback's original purpose).
        self.discard_untracked_detections = discard_untracked_detections

        self._tracks: dict[int, Track] = {}
        self._frame_idx = -1
        self._next_fallback_id = FALLBACK_ID_START
        self._pending_history_merge: dict[int, list[TrackFrame]] = {}
        self._new_aliases: list[tuple[int, int]] = []

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        self._frame_idx += 1
        timestamp = self._frame_idx / self.fps

        pre_assigned: list[tuple[Detection, int]] = []
        unresolved: list[Detection] = []
        for det in detections:
            if det.track_id is not None:
                pre_assigned.append((det, det.track_id))
            elif not self.discard_untracked_detections:
                unresolved.append(det)
            # else: dropped — see discard_untracked_detections above.

        assigned = pre_assigned + self._fallback_match(unresolved)

        updated: list[Track] = []
        ids_used_this_frame: set[int] = set()
        for det, proposed_id in assigned:
            track_id = self._reconcile_id(proposed_id, det, ids_used_this_frame)
            ids_used_this_frame.add(track_id)

            track = self._tracks.get(track_id)
            if track is None:
                track = Track(track_id=track_id, class_name=det.class_name)
                absorbed_history = self._pending_history_merge.pop(track_id, None)
                if absorbed_history:
                    track.history.extend(absorbed_history)
                self._tracks[track_id] = track
            track.history.append(
                TrackFrame(
                    frame_idx=self._frame_idx,
                    timestamp_sec=timestamp,
                    bbox=det.bbox,
                    confidence=det.confidence,
                )
            )
            if len(track.history) > self.history_maxlen:
                track.history = track.history[-self.history_maxlen :]
            track.last_seen_frame = self._frame_idx
            track.active = True
            updated.append(track)
        return updated

    def _reconcile_id(self, proposed_id: int, det: Detection, ids_used_this_frame: set[int]) -> int:
        """Detect a track whose ID just changed mid-pass and merge it back.

        Confirmed on real footage (see cv-service/README.md "Real test run"):
        a physical vehicle can get a fallback ID for a few frames before
        ByteTrack's own track is "confirmed" and starts reporting a real ID,
        or ByteTrack can drop and later re-acquire a track under a new real
        ID after a brief miss. Either way the pipeline was logging the same
        physical pass as two separate detection events (FR-2.2 violation).

        If `proposed_id` has no existing track yet, but a *different*,
        currently-active track's last-seen box overlaps it above
        `merge_iou_threshold`, treat them as the same object: merge the old
        track's history into whichever ID should be canonical (a real
        ByteTrack ID wins over a fallback one, since real IDs persist for
        the rest of the object's life) and return the canonical ID.

        This is a same-frame-IoU heuristic, not true re-identification — it
        can't bridge a gap longer than `max_age_frames`, and a coincidental
        high-IoU overlap between two *different* vehicles at the merge
        instant would wrongly fuse them. Good enough to close the common
        case observed in testing; a proper appearance-based Re-ID is out of
        scope for this phase.
        """
        if proposed_id in self._tracks:
            return proposed_id  # already an established track — nothing to reconcile

        best_match_id, best_score = None, 0.0
        for tid, t in self._tracks.items():
            if tid in ids_used_this_frame or not t.active or t.latest is None:
                continue
            if (self._frame_idx - t.last_seen_frame) > self.max_age_frames:
                continue
            score = iou(det.bbox, t.latest.bbox)
            if score > best_score:
                best_score, best_match_id = score, tid
        if best_match_id is None or best_score < self.merge_iou_threshold:
            return proposed_id

        return self._merge(old_id=best_match_id, new_id=proposed_id)

    def _merge(self, old_id: int, new_id: int) -> int:
        old_track = self._tracks.pop(old_id)
        # A real ByteTrack ID wins over a fallback ID (it keeps being
        # reported for the rest of this object's life); otherwise keep the
        # track that already existed.
        if new_id < FALLBACK_ID_START and old_id >= FALLBACK_ID_START:
            canonical_id, absorbed_id = new_id, old_id
        else:
            canonical_id, absorbed_id = old_id, new_id
        self._pending_history_merge[canonical_id] = old_track.history
        self._new_aliases.append((absorbed_id, canonical_id))
        logger.info("Merged track %d into %d (re-identified mid-pass by IoU)", absorbed_id, canonical_id)
        return canonical_id

    def pop_new_aliases(self) -> list[tuple[int, int]]:
        """(absorbed_id, canonical_id) pairs created since the last call.

        Not part of the Tracker ABC — callers that want cross-track dedup
        state (e.g. PipelineRunner's aggregator) should call this
        defensively via getattr(), since other Tracker implementations
        aren't required to support merging.
        """
        aliases, self._new_aliases = self._new_aliases, []
        return aliases

    def _fallback_match(self, detections: list[Detection]) -> list[tuple[Detection, int]]:
        if not detections:
            return []
        active = [
            (tid, t)
            for tid, t in self._tracks.items()
            if t.active and t.latest is not None and (self._frame_idx - t.last_seen_frame) <= self.max_age_frames
        ]
        used_ids: set[int] = set()
        results: list[tuple[Detection, int]] = []
        for det in detections:
            best_id, best_iou = None, 0.0
            for tid, t in active:
                if tid in used_ids:
                    continue
                score = iou(det.bbox, t.latest.bbox)  # type: ignore[union-attr]
                if score > best_iou:
                    best_iou, best_id = score, tid
            if best_id is not None and best_iou >= self.iou_fallback_threshold:
                used_ids.add(best_id)
                results.append((det, best_id))
            else:
                new_id = self._next_fallback_id
                self._next_fallback_id += 1
                results.append((det, new_id))
        return results

    def get_stale_tracks(self) -> list[Track]:
        """Tracks not updated within max_age_frames — pipeline calls this each
        frame to know when to finalize/emit a detection event."""
        stale: list[Track] = []
        for tid in list(self._tracks.keys()):
            t = self._tracks[tid]
            if t.active and (self._frame_idx - t.last_seen_frame) > self.max_age_frames:
                t.active = False
                stale.append(t)
                del self._tracks[tid]
        return stale

    def finalize_all(self) -> list[Track]:
        """Flush every remaining track — called once at end-of-video."""
        remaining = list(self._tracks.values())
        for t in remaining:
            t.active = False
        self._tracks.clear()
        return remaining
