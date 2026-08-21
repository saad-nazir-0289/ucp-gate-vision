# models/

Gitignored local weight cache. Place the plate-detector weights here as
`plate_detector.pt` — see [`../scripts/download_plate_model.py`](../scripts/download_plate_model.py)
and the "Plate detector model" section of [`../README.md`](../README.md).

`yolov8n.pt` (vehicle detector) does **not** need to go here — Ultralytics
auto-downloads it into its own cache (`~/.cache` or similar) the first time
`YOLO("yolov8n.pt")` runs.
