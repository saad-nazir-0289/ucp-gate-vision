"""Standalone pipeline runner: local video file in, annotated video + JSON events out.

No Redis/Postgres integration — see cv-service/README.md. Phase 2 replaces
this runner's video-file loop with a live RTSP source and adds direction
inference + Redis publishing on top of the same per-track finalize point
used here.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field

import cv2

from ..interfaces import PlateDetector, PlateOCR, PlateReading, Track, Tracker, VehicleDetector
from ..ocr.paddle_ocr import DEFAULT_PLATE_PATTERN
from .aggregator import BestReadingAggregator
from .annotate import draw_plate, draw_vehicle
from .events import DetectionEvent

logger = logging.getLogger(__name__)


@dataclass
class RunSummary:
    frames_total: int
    frames_processed: int
    unique_tracks_seen: int
    detection_events_emitted: int
    elapsed_sec: float
    fps_processing: float
    # Why plate/OCR attempts were discarded, counted per reason. Without
    # this, "the pipeline found nothing" is indistinguishable from "the
    # pipeline read every plate correctly and then threw them all away on
    # a threshold" — which is exactly what a 0.95 confidence floor plus a
    # province-band-contaminated text string was doing (see
    # PaddleOCRPlateReader.read_candidates). Printed at end of run.
    reject_reasons: Counter = field(default_factory=Counter)
    tracks_with_no_reading: int = 0


class PipelineRunner:
    def __init__(
        self,
        vehicle_detector: VehicleDetector,
        plate_detector: PlateDetector,
        plate_ocr: PlateOCR,
        tracker: Tracker,
        evidence_dir: str = "output/evidence",
        frame_skip: int = 1,
        min_plate_conf_to_ocr: float = 0.25,
        ocr_min_confidence: float = 0.50,
        ocr_min_confidence_by_class: dict[str, float] | None = None,
        plate_pattern: re.Pattern | None = DEFAULT_PLATE_PATTERN,
        min_plate_text_length: int = 6,
        source_video: str | None = None,
    ):
        self.vehicle_detector = vehicle_detector
        self.plate_detector = plate_detector
        self.plate_ocr = plate_ocr
        self.tracker = tracker
        self.aggregator = BestReadingAggregator(evidence_dir)
        self.frame_skip = max(1, frame_skip)
        self.min_plate_conf_to_ocr = min_plate_conf_to_ocr

        # OCR acceptance (confidence + format) lives here, not inside
        # PlateOCR (external review — fixed): PlateOCR.read() always
        # returns raw text/confidence; deciding whether that's "good
        # enough" is contextual and belongs with the orchestrator, which
        # also makes offline threshold sweeping trivial (change these
        # values, no need to reconstruct the OCR object).
        #
        # ocr_min_confidence: WAS 0.95, lowered to 0.50 after a 33-vehicle
        # evaluation came back with almost every plate missed. 0.95 was
        # tuned against 3 lucky reads on one clip and did not survive
        # contact with a real dataset:
        #   - It was being applied to a confidence that had already been
        #     corrupted by `min(scores)` over province-band/noise lines
        #     (see PaddleOCRPlateReader.read_candidates for the confirmed
        #     numbers). A perfect 0.9988 read arrived here as 0.6661.
        #   - Even on a clean single-line read it is too tight: 'BUV 711'
        #     scored 0.9307 on a genuine, correct read of a real plate in
        #     this project's own footage.
        # With candidate selection fixed, the plate-format pattern below is
        # now the primary filter (it rejects province bands, 'ENTRANCE',
        # and truncated reads like '545' on shape alone), so this floor
        # only has to catch low-confidence garbage that happens to look
        # plate-shaped. Observed noise sat at 0.17-0.84, genuine reads at
        # 0.93-0.9998, so 0.50 separates them with room on both sides.
        # Sweep it with --ocr-min-conf if your footage differs.
        #
        # ocr_min_confidence_by_class: diagnostic for PR #10's finding
        # (motorcycle accuracy 28.57% vs car 68.42%) — e.g.
        # {"motorcycle": 0.40} tests whether the shared floor is still
        # dropping genuine-but-lower-confidence motorcycle reads.
        self.ocr_min_confidence = ocr_min_confidence
        self.ocr_min_confidence_by_class = ocr_min_confidence_by_class or {}
        self.plate_pattern = plate_pattern
        # Reject a reading shorter than a real plate can be. Every one of the
        # 32 plates in sample_data/ground_truth.csv is 6 or 7 characters, so a
        # 5-character reading is a truncation, not a short plate.
        #
        # This is the fix for the one failure mode the confidence floor could
        # not touch. Measured on the V2 benchmark's own events: four of its
        # eight wrong reads were truncations at HIGH confidence -- LEE8980 read
        # as "LEE18" at 0.9997, LEN9009 as "LEN08", LEN910 as "LEN18", AZE335
        # as "AE335". No --ocr-min-conf setting removes those; a sweep from
        # 0.50 to 0.99 left the correct count flat at 20/33. Requiring 6
        # characters removes all four, taking wrong car reads to zero.
        #
        # Deliberately separate from plate_pattern rather than folded into the
        # regex as a lookahead: it is independently meaningful, independently
        # tunable, and gets its own rejection reason so a run's diagnostics can
        # tell "wrong shape" from "truncated".
        self.min_plate_text_length = min_plate_text_length
        self.source_video = source_video

        self._events: list[DetectionEvent] = []
        self._seen_track_ids: set[int] = set()
        self._reject_reasons: Counter = Counter()
        self._tracks_with_no_reading = 0

    def run(self, video_path: str, output_video_path: str, events_json_path: str) -> RunSummary:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(events_json_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps / self.frame_skip, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open VideoWriter for: {output_video_path}")

        start = time.time()
        frame_idx = -1
        frames_processed = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                if frame_idx % self.frame_skip != 0:
                    continue
                frames_processed += 1

                self._process_frame(frame, frame_idx)
                writer.write(frame)

                if frames_processed % 50 == 0:
                    logger.info("Processed %d frames (video frame_idx=%d)", frames_processed, frame_idx)

            # Video ended — finalize any vehicles still on-screen so their
            # best reading isn't silently dropped.
            for track in self.tracker.finalize_all():
                self._finalize_track(track)
        finally:
            cap.release()
            writer.release()

        elapsed = time.time() - start
        with open(events_json_path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self._events], f, indent=2)

        summary = RunSummary(
            frames_total=frames_total,
            frames_processed=frames_processed,
            unique_tracks_seen=len(self._seen_track_ids),
            detection_events_emitted=len(self._events),
            elapsed_sec=elapsed,
            fps_processing=(frames_processed / elapsed) if elapsed > 0 else 0.0,
            reject_reasons=self._reject_reasons.copy(),
            tracks_with_no_reading=self._tracks_with_no_reading,
        )
        logger.info("Run complete: %s", summary)
        return summary

    def _process_frame(self, frame, frame_idx: int) -> None:
        # Detection/OCR always reads from `clean_frame`, a copy untouched by
        # our own overlay drawing. `frame` accumulates annotations for the
        # output video / evidence image. This split exists because of a
        # confirmed bug: drawing a track's label onto `frame` and then
        # cropping *that same array* for plate detection/OCR let the plate
        # detector find our own overlay text and OCR read it back as if it
        # were a real plate — observed on a car mostly off-frame, whose
        # "#11 car 0.39" label was detected as a "plate" and read back as
        # "11CAR039" (strip non-alnum + uppercase of the label text, exactly).
        # Don't remove this copy without re-verifying that can't happen.
        clean_frame = frame.copy()

        vehicle_detections = self.vehicle_detector.detect(clean_frame)
        tracks = self.tracker.update(vehicle_detections, clean_frame)

        # If the tracker merged a track whose ID changed mid-pass (see
        # ByteTrackTracker._reconcile_id), carry over any accumulated best
        # reading under the old ID so it isn't silently orphaned/lost.
        # Not part of the Tracker ABC — only ByteTrackTracker supports it.
        pop_new_aliases = getattr(self.tracker, "pop_new_aliases", None)
        if pop_new_aliases is not None:
            for old_id, new_id in pop_new_aliases():
                self.aggregator.migrate(old_id, new_id)

        for track in tracks:
            self._seen_track_ids.add(track.track_id)
            latest = track.latest
            if latest is None:
                continue
            draw_vehicle(frame, latest.bbox, track.track_id, track.class_name, latest.confidence)

            plate_detections = self.plate_detector.detect(clean_frame, latest.bbox)
            if not plate_detections:
                # Distinguish "the detector saw nothing" from "the detector
                # found a box and a geometric filter discarded it" — those
                # need opposite fixes (a better/retrained model vs. a
                # threshold), and both used to surface here identically.
                # Not part of the PlateDetector ABC, so read defensively.
                filtered = list(getattr(self.plate_detector, "last_reject_reasons", ()) or ())
                reason = filtered[0] if filtered else "no_plate_detected"
                self._trace(frame_idx, latest.timestamp_sec, track.track_id, "plate", "none", reason)
                self._reject_reasons[reason] += 1
                continue
            # Sorted "own vehicle's plate first, then by confidence" — see
            # the sort key in YoloPlateDetector.detect(), not pure confidence.
            best_plate = plate_detections[0]
            if best_plate.confidence < self.min_plate_conf_to_ocr:
                self._trace(
                    frame_idx, latest.timestamp_sec, track.track_id, "plate", "rejected",
                    f"plate_conf_{best_plate.confidence:.2f}_below_{self.min_plate_conf_to_ocr:.2f}",
                )
                self._reject_reasons["plate_conf_below_threshold"] += 1
                continue
            self._trace(frame_idx, latest.timestamp_sec, track.track_id, "plate", "ok", "")

            x1, y1, x2, y2 = best_plate.bbox.to_int_tuple()
            plate_crop = clean_frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
            candidates = self.plate_ocr.read_candidates(plate_crop)

            min_confidence = self.ocr_min_confidence_by_class.get(track.class_name, self.ocr_min_confidence)
            reading, reject_reason = self._select_reading(candidates, min_confidence)

            # Annotate with whatever was chosen, else the best raw candidate,
            # so the output video shows what OCR actually saw even when the
            # reading was rejected — that's the frame a human needs in order
            # to tell a bad threshold from a bad read.
            shown = reading or (candidates[0] if candidates else None)
            draw_plate(frame, best_plate.bbox, shown.text if shown else "", shown.confidence if shown else 0.0)

            self._trace(
                frame_idx, latest.timestamp_sec, track.track_id, "ocr",
                "ok" if reading else "rejected", "" if reading else reject_reason,
            )
            if reading is None:
                self._reject_reasons[reject_reason] += 1
                continue

            self.aggregator.offer(
                track_id=track.track_id,
                vehicle_class=track.class_name,
                plate_text=reading.text,
                ocr_confidence=reading.confidence,
                vehicle_confidence=latest.confidence,
                frame_idx=frame_idx,
                timestamp_sec=latest.timestamp_sec,
                vehicle_bbox=latest.bbox,
                plate_bbox=best_plate.bbox,
                frame_image=frame,  # annotated so far this frame — useful for human review
                plate_crop_image=plate_crop,  # clean, never polluted by our own overlay
            )

        for stale_track in self.tracker.get_stale_tracks():
            self._finalize_track(stale_track)

    def _select_reading(
        self, candidates: list[PlateReading], min_confidence: float
    ) -> tuple[PlateReading | None, str]:
        """Pick the plate reading from a crop's candidate interpretations.

        The acceptance decision PlateOCR itself doesn't make (external
        review — fixed: OCR reports what it saw, this is the caller's
        judgment call). Returns (reading, reason_if_none).

        Selection, not just filtering (the fix for the 33-vehicle
        evaluation's near-total miss rate). A plate crop routinely yields
        several candidate texts — the number alone, the province band, the
        two joined — and the previous code had already collapsed them into
        one string before this method saw it, so a contaminated join was
        all there was to accept or reject. See
        PaddleOCRPlateReader.read_candidates for the confirmed evidence.

        The plate-format pattern is what discriminates: it is the only
        signal that separates 'BUV711' from 'BUV711JUNJAU', and it is
        applied BEFORE the confidence floor so a correct read is never
        rejected on a confidence figure that belongs to a different line.
        Among candidates that match the format, the highest-confidence one
        wins (candidates arrive pre-sorted).
        """
        if not candidates:
            return None, "no_text_recognized"

        if self.min_plate_text_length > 0:
            long_enough = [c for c in candidates if len(c.text) >= self.min_plate_text_length]
            if not long_enough:
                return None, "text_too_short"
            candidates = long_enough

        if self.plate_pattern is not None:
            well_formed = [c for c in candidates if self.plate_pattern.match(c.text)]
            if not well_formed:
                return None, "format_mismatch"
        else:
            well_formed = candidates

        for candidate in well_formed:
            if candidate.confidence >= min_confidence:
                return candidate, ""
        return None, "confidence_below_threshold"

    def _trace(self, frame_idx: int, timestamp_sec: float, track_id: int, stage: str, outcome: str, reason: str) -> None:
        """Structured per-track-per-frame record (external review's
        suggestion): frame, timestamp, track_id, stage, outcome, reason.
        `grep "track=<id>"` on a DEBUG-level log reconstructs one vehicle's
        full journey through the pipeline without guessing where it failed.
        """
        logger.debug(
            "TRACE frame=%d t=%.2f track=%d stage=%s outcome=%s reason=%s",
            frame_idx, timestamp_sec, track_id, stage, outcome, reason,
        )

    def _finalize_track(self, track: Track) -> None:
        reading = self.aggregator.pop(track.track_id)
        if reading is None:
            self._tracks_with_no_reading += 1
            logger.debug("Track %d ended with no usable plate reading — skipped, no event emitted.", track.track_id)
            return

        event = DetectionEvent(
            source_video=self.source_video,
            track_id=reading.track_id,
            vehicle_class=reading.vehicle_class,
            plate_text=reading.plate_text,
            ocr_confidence=reading.ocr_confidence,
            vehicle_confidence=reading.vehicle_confidence,
            frame_idx=reading.frame_idx,
            timestamp_sec=reading.timestamp_sec,
            vehicle_bbox=reading.vehicle_bbox.to_int_tuple(),
            plate_bbox=reading.plate_bbox.to_int_tuple() if reading.plate_bbox else None,
            evidence_frame_path=reading.evidence_frame_path,
            evidence_plate_crop_path=reading.evidence_plate_crop_path,
        )
        self._events.append(event)
        logger.info(
            "Detection event: track=%d plate=%r ocr_conf=%.2f frame=%d t=%.2fs",
            event.track_id,
            event.plate_text,
            event.ocr_confidence,
            event.frame_idx,
            event.timestamp_sec,
        )
        print(json.dumps(event.to_dict()), flush=True)  # flush: stdout is fully buffered when redirected to a file/pipe
