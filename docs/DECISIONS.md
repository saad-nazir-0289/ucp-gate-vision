# Architecture Decisions — review of 2026-08-28

Supplements the decisions table in [`ARCHITECTURE.md` §3](ARCHITECTURE.md).
That table records what was chosen at design time, on paper. This file
records what was **re-examined after Phase 1 met real footage**, and why
several answers changed.

Trigger: the 33-vehicle benchmark
([`ACCURACY_REPORT_MSNUPDATED_33.md`](ACCURACY_REPORT_MSNUPDATED_33.md))
measured 66.67% overall — cars 89.47%, bikes 35.71% — and a rerun on a
second machine still showed vehicles detected but plates not.

## The measurement that drove most of these

A sweep of the plate detector over real frames, varying vehicle size,
confidence, and inference resolution:

| Vehicle in frame | Plate box | Outcome |
|---|---|---|
| 472×486 | 93×46 @0.79 | detected, OCR works |
| 313×285 | 55×29 @0.91 | detected |
| 208×212 | 40×21 @0.33 | detected |
| 150×153 | 23×9 @0.26 | discarded (min-size 24×10) |
| 172×109 | 19×9 @0.69 | discarded (min-size) |
| 90×54 | none at any setting | — |
| 65×114 *(bike-shaped)* | none at any setting | — |

**Plate width is consistently 18–20% of vehicle width.** The detector needs
a vehicle ≈300px wide before it yields an OCR-able plate. Tall/narrow
motorcycle-shaped crops fail hardest — consistent with the bike gap.

## Decisions

| # | Question | Decision | Notes |
|---|---|---|---|
| 1 | Plate detector | **Fine-tune on a Pakistani plate dataset** | The current model is a random HF user's YOLOv8n trained on non-local plates, never validated here. Highest-leverage change available, and permitted by `PROJECT_SPEC.md` §3. |
| 2 | Training data | **Roboflow Pakistani plate sets** | [pakistan-license-plate-detection](https://universe.roboflow.com/license-plate-yf8bv/pakistan-license-plate-detection) (~6,161 images); [Pakistani-Number-plates](https://universe.roboflow.com/malik-kashif-saeed-aswwf/pakistani-number-plates) (336) as a supplement. |
| 3 | Training compute | **Free Colab / Kaggle GPU** | A single-class YOLO fine-tune trains in well under an hour on a T4. Unblocks work without waiting for the RTX 4090/5060. |
| 4 | Vehicle detector | **Upgrade to YOLO26** | Already supported by the pinned `ultralytics==8.4.53`. Noted at review time that vehicle detection is *not* the current bottleneck — vehicles are detected fine — so this is not expected to fix plates. Chosen anyway. |
| 5 | OCR | **No fine-tuning for now** | Reversed during review once it emerged that Roboflow *detection* datasets carry bounding boxes only, no plate text — they cannot train OCR. PaddleOCR already reads clear plates at 0.99+; the failures are undetected/too-small plates, not misread clear ones. Revisit only if wrong reads persist after the detector improves. |
| 6 | Distant vehicles | **Accept; read them when close** | A 20×10px plate is a physics limit, not a model failure. The aggregator already keeps the best reading per track, so a vehicle needs only one good close frame. **Follow-up: verify tracks survive from far to near** — that is the actual requirement. |
| 7 | Plate format regex | **Widened and made configurable** | Now `^[A-Z]{1,4}[0-9]{1,5}$`, overridable via `--plate-pattern`. It is the pipeline's primary OCR filter, so a too-tight pattern silently discards correct reads. All 32 labeled plates still match; every known false positive still rejected. **Done.** |
| 8 | Tracker duplicates | **Defer until after plate fixes** | Changing detection and tracking together makes the next benchmark uninterpretable. |
| 9 | Direction inference | **Fix tracking first, then build** | `ARCHITECTURE.md` §7's fused method reads track history, and tracks currently break mid-pass (10 exact-plate duplicates = one pass split across 2–3 tracks). Direction from a trajectory fragment is noise. |
| 10 | Ground truth quality | **Revisit later** | Known weaknesses recorded below rather than fixed now. |
| 11 | Backend stack | **Keep as specced** | FastAPI + Redis Streams + Postgres. Neither Redis nor Postgres is load-driven at 3 gates, but both are already in `docker-compose.yml` and Redis genuinely decouples CV from backend crashes. Not redesigning before hitting a wall. |
| 12 | Ultralytics AGPL-3.0 | **Defer — academic use** | Accepted for a university project. Must be resolved before any commercial deployment; YOLO26 does not change it. |

## Sequencing that follows

1. Fine-tune the plate detector (decisions 1–3) — **the current bottleneck**
2. Re-run the 33-vehicle benchmark and re-measure
3. Fix tracker duplicates (decision 8)
4. Then direction inference (decision 9)
5. Backend as specced (decision 11)

## Known-weak, deliberately deferred

**Ground truth** is the measurement instrument for every decision above, and
is currently the weakest artifact in the repo:

- Timestamps are coarse integers; every multi-vehicle row is 6s or 9s
- 21 of 33 rows flag ambiguous; two correct reads scored as both missed and
  false-positive because of it
- No direction labels (one vehicle only) — Phase 2 direction inference
  cannot be scored at all as things stand
- No vehicle distance/size metadata, which the sweep above shows is *the*
  determining factor for plate success
- 33 vehicles is small: ±1 vehicle moves accuracy by ±3%

**Not yet questioned in this review:** performance against NFR-1
(1.3–1.4 s/frame on CPU vs a 2–3 s end-to-end alert budget), evidence
storage, RTSP ingestion, auth/JWT, and the frontend.
