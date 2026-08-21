#!/usr/bin/env python3
"""
CLI entry point for the standalone cv-service pipeline (Phase 1).

Runs vehicle detection -> tracking -> plate detection -> OCR -> per-track
dedup against a single local video file, draws annotated output, and logs
one JSON detection event per finalized track. No Redis/Postgres
integration and no direction inference yet — see cv-service/README.md.

Usage:
    python run_pipeline.py --video sample_data/gate1_sample.mp4
"""
from __future__ import annotations

import argparse
import logging
import sys

import cv2

# Import order matters here (confirmed on Windows): anpr.detectors.* imports
# `ultralytics` -> `torch` eagerly at module scope, while anpr.ocr.paddle_ocr
# imports `paddleocr` -> `paddle` lazily, inside PaddleOCRPlateReader.__init__.
# That means torch always finishes loading before paddle does, which avoids
# a real DLL load-order conflict: importing paddle/paddleocr *before* torch
# in the same process makes a later `import torch` fail with
# `OSError: [WinError 127] ... loading torch\lib\shm.dll`. Don't reorder
# these imports (or move the paddleocr import to module scope) without
# re-testing — see cv-service/README.md "Known risks".
from anpr.detectors.plate_detector import YoloPlateDetector
from anpr.detectors.vehicle_detector import YoloVehicleDetector
from anpr.ocr.paddle_ocr import PaddleOCRPlateReader
from anpr.pipeline.runner import PipelineRunner
from anpr.tracking.byte_tracker import ByteTrackTracker


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Campus ANPR — standalone CV pipeline (Phase 1)")
    p.add_argument("--video", required=True, help="Path to a local input video file")
    p.add_argument("--output-video", default="output/annotated.mp4")
    p.add_argument("--events-json", default="output/events.json")
    p.add_argument("--evidence-dir", default="output/evidence")
    p.add_argument(
        "--vehicle-weights",
        default="yolov8n.pt",
        help="Ultralytics COCO-pretrained weights (auto-downloaded by ultralytics if not found locally)",
    )
    p.add_argument(
        "--plate-weights",
        default="models/plate_detector.pt",
        help="Local path to plate-detector weights — see scripts/download_plate_model.py",
    )
    p.add_argument("--device", default="cpu", help="'cpu', 'cuda:0', etc.")
    p.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Ultralytics tracker config name, e.g. 'bytetrack.yaml' (default, fast) or "
        "'botsort.yaml' (adds appearance-based ReID — more robust across occlusion/gaps, more compute)",
    )
    p.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame (1 = every frame)")
    p.add_argument("--vehicle-conf", type=float, default=0.35)
    p.add_argument("--plate-conf", type=float, default=0.25)
    p.add_argument(
        "--ocr-min-conf",
        type=float,
        default=0.95,
        help="Confirmed on real footage: 0.95 cleanly drops garbage reads from low-quality "
        "track fragments while every genuine plate read still clears it (0.96-0.9997 on the "
        "winning frame). Trade-off: a genuine plate that never gets a clear enough frame is "
        "silently dropped instead of flagged low-confidence — watch for this on harder footage.",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("run_pipeline")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        log.error("Could not open video file: %s", args.video)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    log.info("Loading vehicle detector (%s) on %s ...", args.vehicle_weights, args.device)
    vehicle_detector = YoloVehicleDetector(
        weights=args.vehicle_weights,
        device=args.device,
        conf_threshold=args.vehicle_conf,
        tracker_cfg=args.tracker,
    )

    log.info("Loading plate detector (%s) ...", args.plate_weights)
    plate_detector = YoloPlateDetector(
        weights_path=args.plate_weights,
        device=args.device,
        conf_threshold=args.plate_conf,
    )

    log.info("Loading PaddleOCR ...")
    plate_ocr = PaddleOCRPlateReader(min_confidence=args.ocr_min_conf)

    tracker = ByteTrackTracker(fps=fps / max(1, args.frame_skip))

    runner = PipelineRunner(
        vehicle_detector=vehicle_detector,
        plate_detector=plate_detector,
        plate_ocr=plate_ocr,
        tracker=tracker,
        evidence_dir=args.evidence_dir,
        frame_skip=args.frame_skip,
        min_plate_conf_to_ocr=args.plate_conf,
    )

    log.info("Running pipeline on %s ...", args.video)
    summary = runner.run(args.video, args.output_video, args.events_json)

    log.info(
        "Done. frames=%d/%d unique_tracks=%d events=%d elapsed=%.1fs (%.1f fps)",
        summary.frames_processed,
        summary.frames_total,
        summary.unique_tracks_seen,
        summary.detection_events_emitted,
        summary.elapsed_sec,
        summary.fps_processing,
    )
    print(f"\nAnnotated video: {args.output_video}")
    print(f"Detection events (JSON): {args.events_json}")
    print(f"Evidence images: {args.evidence_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
