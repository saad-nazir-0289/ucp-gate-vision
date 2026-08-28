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
import os
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
from anpr.config import PipelineConfig
from anpr.detectors.plate_detector import YoloPlateDetector
from anpr.detectors.vehicle_detector import YoloVehicleDetector
from anpr.ocr.paddle_ocr import PaddleOCRPlateReader
from anpr.pipeline.runner import PipelineRunner
from anpr.tracking.byte_tracker import ByteTrackTracker

# Single source of truth for defaults (external review — fixed: this file
# and anpr/config.py had drifted out of sync before). CLI flags below read
# their `default=` from this instance instead of hardcoding numbers again.
DEFAULTS = PipelineConfig()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Campus ANPR — standalone CV pipeline (Phase 1)")
    p.add_argument("--video", required=True, help="Path to a local input video file")
    p.add_argument("--output-video", default="output/annotated.mp4")
    p.add_argument("--events-json", default="output/events.json")
    p.add_argument("--evidence-dir", default="output/evidence")
    p.add_argument(
        "--vehicle-weights",
        default=DEFAULTS.vehicle_weights,
        help="Ultralytics COCO-pretrained weights (auto-downloaded by ultralytics if not found locally)",
    )
    p.add_argument(
        "--plate-weights",
        default=DEFAULTS.plate_weights,
        help="Local path to plate-detector weights — see scripts/download_plate_model.py",
    )
    p.add_argument("--device", default=DEFAULTS.device, help="'cpu', 'cuda:0', etc.")
    p.add_argument(
        "--tracker",
        default=DEFAULTS.tracker_cfg,
        help="Ultralytics tracker config name, e.g. 'bytetrack.yaml' (default) or 'botsort.yaml' "
        "(adds global motion compensation over ByteTrack — NOT appearance ReID by default; "
        "botsort.yaml ships with with_reid: False, verified against ultralytics' shipped config. "
        "ReID needs a separate config with with_reid: true, e.g. trackers/botsort_reid.yaml here — "
        "tested, made no difference for the specific failure it was tried against).",
    )
    p.add_argument("--frame-skip", type=int, default=DEFAULTS.frame_skip, help="Process every Nth frame (1 = every frame)")
    p.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULTS.imgsz,
        help="Ultralytics inference resolution for the vehicle detector. Was implicitly 640 (the "
        "library default) — verified via the external PR #10 review that at 2560x1440 source "
        "frames, 640 shrinks a motorcycle to ~37px before the model sees it. Sweep 640/960/1280.",
    )
    p.add_argument(
        "--vehicle-conf",
        type=float,
        default=DEFAULTS.vehicle_conf,
        help="Was 0.35. VERIFIED against ultralytics' bytetrack.yaml: ByteTrack's low-confidence "
        "recovery matches down to track_low_thresh=0.1, but conf=0.35 discarded everything below "
        "that before ByteTrack ever saw it, defeating its signature recovery feature. 0.10 lets "
        "the tracker's own thresholds govern association instead of a blunt upstream filter.",
    )
    p.add_argument("--plate-conf", type=float, default=DEFAULTS.plate_conf)
    p.add_argument(
        "--ocr-min-conf",
        type=float,
        default=DEFAULTS.ocr_min_confidence,
        help="WAS 0.95, now 0.50. 0.95 was tuned against 3 lucky reads on one clip and cost "
        "nearly every plate in a later 33-vehicle evaluation: it was applied to a confidence "
        "corrupted by province-band/noise OCR lines, and even a clean correct read of a real "
        "plate here scored 0.9307. The plate-format pattern is now the primary filter; this "
        "floor only rejects plate-shaped garbage (observed 0.17-0.84 vs 0.93+ for real reads).",
    )
    p.add_argument(
        "--ocr-min-conf-motorcycle",
        type=float,
        default=None,
        help="Overrides --ocr-min-conf for motorcycle tracks only (PR #10 measured motorcycle "
        "accuracy 28.57%% vs car 68.42%%). Motorbike plates are smaller and more angled, so their "
        "genuine reads score lower; try e.g. 0.40 and compare motorcycle miss counts.",
    )
    plate_geom = p.add_argument_group(
        "plate-box geometry filters",
        "Reject implausible plate boxes before OCR. Previously hardcoded and unreachable from "
        "the CLI — which made them impossible to sweep during a 'no plates detected' "
        "investigation without editing code. Check a run's rejection summary before tuning: if "
        "it reports mostly `no_plate_detected`, these are the knobs that matter.",
    )
    plate_geom.add_argument("--plate-min-width", type=int, default=DEFAULTS.min_plate_width_px)
    plate_geom.add_argument("--plate-min-height", type=int, default=DEFAULTS.min_plate_height_px)
    plate_geom.add_argument(
        "--plate-min-aspect",
        type=float,
        default=DEFAULTS.min_plate_aspect_ratio,
        help="Was 1.0, now 0.8: local motorcycle plates are stacked two-line and near-square, so "
        "genuine ones at a gate-camera angle measured just under 1.0 and were discarded.",
    )
    plate_geom.add_argument("--plate-max-aspect", type=float, default=DEFAULTS.max_plate_aspect_ratio)
    plate_geom.add_argument(
        "--plate-edge-margin",
        type=int,
        default=DEFAULTS.plate_frame_edge_margin_px,
        help="Reject plate boxes within this many px of the camera frame edge (likely physically "
        "truncated). Now backstopped by the plate-format pattern, which rejects a truncated read "
        "like '545' on its own shape — lower it if this is costing recall at your gate.",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # See the same note in scripts/score_accuracy.py: on a Windows console
    # (cp1252) the em-dashes in this script's diagnostic output render as
    # replacement characters.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass  # not a reconfigurable text stream (piped/captured) — output is fine as-is

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
        imgsz=args.imgsz,
        tracker_cfg=args.tracker,
        classes=DEFAULTS.vehicle_classes,
    )

    log.info("Loading plate detector (%s) ...", args.plate_weights)
    plate_detector = YoloPlateDetector(
        weights_path=args.plate_weights,
        device=args.device,
        conf_threshold=args.plate_conf,
        frame_edge_margin_px=args.plate_edge_margin,
        min_plate_width_px=args.plate_min_width,
        min_plate_height_px=args.plate_min_height,
        min_aspect_ratio=args.plate_min_aspect,
        max_aspect_ratio=args.plate_max_aspect,
    )

    log.info("Loading PaddleOCR ...")
    # No confidence/format params here anymore (external review — fixed):
    # PlateOCR just reads text; PipelineRunner decides what's acceptable.
    plate_ocr = PaddleOCRPlateReader()

    tracker = ByteTrackTracker(fps=fps / max(1, args.frame_skip))

    ocr_min_confidence_by_class = {}
    if args.ocr_min_conf_motorcycle is not None:
        ocr_min_confidence_by_class["motorcycle"] = args.ocr_min_conf_motorcycle

    runner = PipelineRunner(
        vehicle_detector=vehicle_detector,
        plate_detector=plate_detector,
        plate_ocr=plate_ocr,
        tracker=tracker,
        evidence_dir=args.evidence_dir,
        frame_skip=args.frame_skip,
        min_plate_conf_to_ocr=args.plate_conf,
        ocr_min_confidence=args.ocr_min_conf,
        ocr_min_confidence_by_class=ocr_min_confidence_by_class,
        # Recorded into every event so scripts/score_accuracy.py matches
        # ground truth on the actual source video rather than inferring it
        # from the events-JSON filename — that inference silently scored a
        # whole evaluation as 100% missed when the output was named
        # anything other than the dataset file.
        source_video=os.path.basename(args.video),
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
    _print_rejection_summary(summary)

    print(f"\nAnnotated video: {args.output_video}")
    print(f"Detection events (JSON): {args.events_json}")
    print(f"Evidence images: {args.evidence_dir}")
    return 0


def _print_rejection_summary(summary) -> None:
    """Say WHERE vehicles were lost, not just how many events came out.

    Added after a 33-vehicle evaluation reported almost everything missed
    with no way to tell, from the output alone, whether the plates were
    never detected, never read, or read correctly and then thrown away on
    a threshold. (It was the last one — twice over, see
    PaddleOCRPlateReader.read_candidates.) The counts are per plate/OCR
    attempt, not per vehicle: one track contributes one attempt per frame
    it appears in, so read these as proportions, not vehicle counts.
    """
    total = sum(summary.reject_reasons.values())
    print("\n" + "=" * 60)
    print("WHERE PLATE READS WERE LOST (per-frame attempts, not per-vehicle)")
    print("=" * 60)
    print(f"Vehicle tracks seen:                    {summary.unique_tracks_seen}")
    print(f"Tracks that ended with no plate read:   {summary.tracks_with_no_reading}")
    print(f"Detection events emitted:               {summary.detection_events_emitted}")
    if not total:
        print("No rejected plate/OCR attempts.")
        return
    print(f"\nRejected attempts: {total}")
    for reason, count in summary.reject_reasons.most_common():
        print(f"  {reason:<32} {count:>6}  ({count / total * 100:5.1f}%)")
    hints = {
        "no_plate_detected": "plate detector found nothing in the vehicle crop — sweep "
                             "--plate-conf down, or the geometry filters (--plate-min-*, --plate-*-aspect)",
        "plate_conf_below_threshold": "boxes found but below --plate-conf",
        "format_mismatch": "text read, but no candidate matched the plate-format pattern "
                           "(DEFAULT_PLATE_PATTERN in anpr/ocr/paddle_ocr.py) — check it fits your local plates",
        "confidence_below_threshold": "well-formed plate text read but below --ocr-min-conf — lower it",
        "no_text_recognized": "plate crop found but OCR read no text at all — likely too small/blurry",
    }
    print("\nWhat the top reason means:")
    top = summary.reject_reasons.most_common(1)[0][0]
    print(f"  {top}: {hints.get(top, 'see anpr/pipeline/runner.py')}")


if __name__ == "__main__":
    sys.exit(main())
