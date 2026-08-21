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
import time
from dataclasses import dataclass

import cv2

from ..interfaces import PlateDetector, PlateOCR, Track, Tracker, VehicleDetector
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
    ):
        self.vehicle_detector = vehicle_detector
        self.plate_detector = plate_detector
        self.plate_ocr = plate_ocr
        self.tracker = tracker
        self.aggregator = BestReadingAggregator(evidence_dir)
        self.frame_skip = max(1, frame_skip)
        self.min_plate_conf_to_ocr = min_plate_conf_to_ocr

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
                continue
            best_plate = plate_detections[0]  # PlateDetector returns confidence-sorted
            if best_plate.confidence < self.min_plate_conf_to_ocr:
                continue

            x1, y1, x2, y2 = best_plate.bbox.to_int_tuple()
            plate_crop = clean_frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
            text, ocr_conf = self.plate_ocr.read(plate_crop)
            draw_plate(frame, best_plate.bbox, text, ocr_conf)

            if not text:
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
