#!/usr/bin/env python3
"""
Downloader for the pretrained plate-detection model
(Koushim/yolov8-license-plate-detection on HuggingFace Hub — see
cv-service/README.md "Plate detector model" for why this model + its
license).

History (kept for context — see cv-service/README.md "Known risks"):
the originally-picked repo, keremberke/yolov8n-license-plate-detection,
turned out to no longer exist (confirmed via a live 404 from the HF Hub API
while actually running this — not just a wrong filename) once this was
tested end-to-end. Koushim/yolov8-license-plate-detection was found via a
live HfApi search as a replacement: YOLOv8n, single `license_plate` class,
trained via the Ultralytics framework, and explicitly MIT-licensed by its
author (a meaningful improvement — the original pick was AGPL-3.0).

An even earlier version of this script also tried `ultralyticsplus` (a
convenience loader that resolves an HF repo id without needing to know the
exact filename) as a first strategy. That was dropped after actually
running it: ultralyticsplus==0.1.0 hard-pins `ultralytics<8.1.0`, which
conflicts with the `ultralytics==8.4.53` the core pipeline needs — the two
can't be installed in the same environment. huggingface_hub is the only
strategy now.

If it fails, download the weights manually from the model's "Files" tab at
https://huggingface.co/Koushim/yolov8-license-plate-detection
and pass the local path via `--plate-weights` to run_pipeline.py.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

DEFAULT_REPO_ID = "Koushim/yolov8-license-plate-detection"
DEFAULT_OUTPUT = "models/plate_detector.pt"
CANDIDATE_FILENAMES = ["best.pt", "model.pt", "weights/best.pt"]


def try_hf_hub_download(repo_id: str, output_path: str, candidates: list[str]) -> bool:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[download_plate_model] huggingface_hub not installed — skipping this strategy.")
        return False

    for filename in candidates:
        try:
            local_path = hf_hub_download(repo_id=repo_id, filename=filename)
        except Exception:
            continue
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        shutil.copyfile(local_path, output_path)
        print(f"[download_plate_model] Downloaded {filename} -> {output_path}")
        return True

    print(
        f"[download_plate_model] None of the candidate filenames {candidates} were found in "
        f"{repo_id}. Check the repo's Files tab for the actual weight filename and retry with "
        f"--filename <name>, or download manually."
    )
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--filename", default=None, help="Exact filename in the HF repo, if known")
    args = parser.parse_args(argv)

    candidates = [args.filename] if args.filename else CANDIDATE_FILENAMES

    if try_hf_hub_download(args.repo_id, args.output, candidates):
        return 0

    print(
        f"\n[download_plate_model] Automatic download failed. Manually download the weights from "
        f"https://huggingface.co/{args.repo_id} (Files tab), save as {args.output}, then re-run "
        f"run_pipeline.py with --plate-weights {args.output} (or wherever you saved it)."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
