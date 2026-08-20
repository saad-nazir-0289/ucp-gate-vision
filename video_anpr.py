import argparse
from pathlib import Path

import cv2
from fast_alpr import ALPR


def normalize_confidence(confidence):
    """
    Convert fast-alpr OCR confidence to a single float.

    fast-alpr may return:
        float
        list of floats
    """
    if confidence is None:
        return 0.0

    if isinstance(confidence, (list, tuple)):
        if not confidence:
            return 0.0

        return sum(float(x) for x in confidence) / len(confidence)

    return float(confidence)


def process_video(
    video_path,
    alpr,
    skip_frames,
    min_confidence,
    repeat
):
    """Process one video and return detected plates."""

    print("\n")
    print("=" * 70)
    print(f"VIDEO: {video_path.name}")
    print("=" * 70)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        return set()

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"FPS      : {fps:.2f}")
    print(f"Frames   : {total_frames}")

    if total_frames > 0:
        print(f"Duration : {total_frames / fps:.2f} seconds")

    print("-" * 70)

    frame_idx = 0

    # Plate -> recent confidence values
    plate_history = {}

    # Plates reported for this video
    reported_plates = set()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Skip frames if requested
        if frame_idx % skip_frames != 0:
            frame_idx += 1
            continue

        # Run ALPR
        results = alpr.predict(frame)

        timestamp = frame_idx / fps

        for result in results:

            # No OCR result
            if result.ocr is None:
                continue

            plate_text = result.ocr.text

            if not plate_text:
                continue

            plate_text = plate_text.strip()

            if not plate_text:
                continue

            # Convert confidence to float
            confidence = normalize_confidence(
                result.ocr.confidence
            )

            # Ignore low confidence
            if confidence < min_confidence:
                continue

            # Normalize plate text
            plate_text = (
                plate_text
                .upper()
                .replace(" ", "")
                .replace("-", "")
            )

            if not plate_text:
                continue

            # Add confidence to history
            if plate_text not in plate_history:
                plate_history[plate_text] = []

            plate_history[plate_text].append(confidence)

            # Keep only recent detections
            if len(plate_history[plate_text]) > repeat:
                plate_history[plate_text].pop(0)

            history = plate_history[plate_text]

            # Report after repeated detections
            if (
                len(history) >= repeat
                and plate_text not in reported_plates
            ):

                avg_confidence = sum(history) / len(history)

                minutes = int(timestamp // 60)
                seconds = timestamp % 60

                print(
                    f"[{minutes:02d}:{seconds:05.2f}] "
                    f"PLATE: {plate_text:<15} "
                    f"CONFIDENCE: {avg_confidence:.2f}"
                )

                reported_plates.add(plate_text)

        frame_idx += 1

    cap.release()

    print("-" * 70)
    print(f"Finished: {video_path.name}")
    print(f"Plates detected: {len(reported_plates)}")

    return reported_plates


def main():

    parser = argparse.ArgumentParser(
        description="Process multiple videos with fast-alpr"
    )

    parser.add_argument(
        "--input",
        default="input",
        help="Folder containing input videos"
    )

    parser.add_argument(
        "--detector",
        default="yolo-v9-s-608-license-plate-end2end",
        help="License plate detector model"
    )

    parser.add_argument(
        "--ocr",
        default="cct-s-v2-global-model",
        help="OCR model"
    )

    parser.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        help="Run ANPR every N frames"
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.70,
        help="Minimum OCR confidence"
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Required repeated detections"
    )

    args = parser.parse_args()

    if args.skip_frames < 1:
        parser.error("--skip-frames must be at least 1")

    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    # Input folder
    input_folder = Path(args.input)

    if not input_folder.exists():
        raise RuntimeError(
            f"Input folder does not exist: {input_folder}"
        )

    # Find videos
    video_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm"
    }

    videos = sorted(
        [
            file
            for file in input_folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in video_extensions
        ]
    )

    if not videos:
        raise RuntimeError(
            f"No video files found in: {input_folder}"
        )

    print("=" * 70)
    print("FAST-ALPR MULTI-VIDEO CONSOLE ANPR")
    print("=" * 70)

    print(f"\nInput folder : {input_folder}")
    print(f"Videos found : {len(videos)}")

    print("\nVideos:")

    for i, video in enumerate(videos, 1):
        print(f"  {i}. {video.name}")

    print("\nLoading ALPR models...")
    print(f"Detector : {args.detector}")
    print(f"OCR      : {args.ocr}")

    alpr = ALPR(
        detector_model=args.detector,
        ocr_model=args.ocr
    )

    print("Models loaded.")

    # Overall results
    all_results = {}

    # Process videos
    for video in videos:

        plates = process_video(
            video_path=video,
            alpr=alpr,
            skip_frames=args.skip_frames,
            min_confidence=args.min_confidence,
            repeat=args.repeat
        )

        all_results[video.name] = plates

    # Final summary
    print("\n\n")
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    total_unique_plates = set()

    for video_name, plates in all_results.items():

        print(f"\n{video_name}")

        if plates:
            for plate in sorted(plates):
                print(f"    {plate}")

            total_unique_plates.update(plates)

        else:
            print("    No plates detected")

    print("\n" + "-" * 70)

    print(
        f"Videos processed       : {len(videos)}"
    )

    print(
        f"Unique plates overall  : {len(total_unique_plates)}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()