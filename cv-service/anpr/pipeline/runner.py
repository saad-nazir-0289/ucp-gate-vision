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
from dataclasses import dataclass

import cv2

from ..interfaces import PlateDetector, PlateOCR, Track, Tracker, VehicleDetector
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
        ocr_min_confidence: float = 0.95,
        ocr_min_confidence_by_class: dict[str, float] | None = None,
        plate_pattern: re.Pattern | None = DEFAULT_PLATE_PATTERN,
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
        # ocr_min_confidence: confirmed on real footage, 0.95 cleanly drops
        # garbage reads from low-quality track fragments while every
        # genuine plate read still clears it (0.96-0.9997 on the winning
        # frame). Trade-off: a genuine plate that never gets a clear enough
        # frame is silently dropped instead of flagged low-confidence —
        # watch for this on harder footage (motorbikes, low light).
        #
        # ocr_min_confidence_by_class: diagnostic for PR #10's finding
        # (motorcycle accuracy 28.57% vs car 68.42%) — e.g.
        # {"motorcycle": 0.75} tests whether the car-tuned 0.95 floor is
        # silently dropping genuine-but-lower-confidence motorcycle reads.
        self.ocr_min_confidence = ocr_min_confidence
        self.ocr_min_confidence_by_class = ocr_min_confidence_by_class or {}
        self.plate_pattern = plate_pattern

        self._events: list[DetectionEvent] = []
        self._seen_track_ids: set[int] = set()

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
                self._trace(frame_idx, latest.timestamp_sec, track.track_id, "plate", "none", "no_plate_detected")
                continue
            # Sorted "own vehicle's plate first, then by confidence" — see
            # the sort key in YoloPlateDetector.detect(), not pure confidence.
            best_plate = plate_detections[0]
            if best_plate.confidence < self.min_plate_conf_to_ocr:
                self._trace(
                    frame_idx, latest.timestamp_sec, track.track_id, "plate", "rejected",
                    f"plate_conf_{best_plate.confidence:.2f}_below_{self.min_plate_conf_to_ocr:.2f}",
                )
                continue
            self._trace(frame_idx, latest.timestamp_sec, track.track_id, "plate", "ok", "")

            x1, y1, x2, y2 = best_plate.bbox.to_int_tuple()
            plate_crop = clean_frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
            text, ocr_conf = self.plate_ocr.read(plate_crop)
            draw_plate(frame, best_plate.bbox, text, ocr_conf)

            min_confidence = self.ocr_min_confidence_by_class.get(track.class_name, self.ocr_min_confidence)
            accepted, reject_reason = self._accept_ocr_reading(text, ocr_conf, min_confidence)
            self._trace(
                frame_idx, latest.timestamp_sec, track.track_id, "ocr",
                "ok" if accepted else "rejected", "" if accepted else reject_reason,
            )
            if not accepted:
                continue

            self.aggregator.offer(
                track_id=track.track_id,
                vehicle_class=track.class_name,
                plate_text=text,
                ocr_confidence=ocr_conf,
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

    def _accept_ocr_reading(self, text: str, confidence: float, min_confidence: float) -> tuple[bool, str]:
        """The acceptance decision PlateOCR itself no longer makes (external
        review — fixed: OCR returns raw text/confidence, this is the caller's
        judgment call). Returns (accepted, reason_if_rejected)."""
        if not text:
            return False, "no_text_recognized"
        if confidence < min_confidence:
            return False, f"confidence_{confidence:.2f}_below_{min_confidence:.2f}"
        if self.plate_pattern is not None and not self.plate_pattern.match(text):
            return False, f"format_mismatch_pattern_{self.plate_pattern.pattern}"
        return True, ""

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
            logger.debug("Track %d ended with no usable plate reading — skipped, no event emitted.", track.track_id)
            return

        event = DetectionEvent(
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
