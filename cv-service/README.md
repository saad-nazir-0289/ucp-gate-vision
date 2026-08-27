# cv-service

Python computer-vision / ANPR (Automatic Number Plate Recognition) pipeline for the Campus ANPR System.

**Phase 1 status:** standalone pipeline, tested against a single local video file. No Redis/Postgres integration, no live RTSP, no direction inference (entry/exit) yet — those land in Phase 2/3 (see [`../docs/CLAUDE_CODE_KICKOFF.md`](../docs/CLAUDE_CODE_KICKOFF.md)). This phase exists to validate detection + OCR accuracy in isolation before wiring in the rest of the stack.

## What it does

For each frame of an input video:

1. **`VehicleDetector`** — detects cars/motorcycles (Ultralytics YOLOv8, COCO-pretrained, classes filtered to `car`/`motorcycle`)
2. **`Tracker`** — assigns/maintains a persistent track ID per vehicle and keeps its position history (ByteTrack, via Ultralytics' `.track()`)
3. **`PlateDetector`** — detects the license-plate region within a crop around each vehicle's box (pretrained YOLO plate-detection model)
4. **`PlateOCR`** — reads the plate string + confidence from the plate crop (PaddleOCR)
5. Dedup: keeps only the **highest-OCR-confidence reading per track** (not one event per frame) and emits one `DetectionEvent` when that track ends

Output per run: an annotated video, a JSON log of detection events, and evidence images (full frame + plate crop) per event.

## Design principle — modularity

`VehicleDetector`, `PlateDetector`, `PlateOCR`, and `Tracker` are ABCs in [`anpr/interfaces.py`](anpr/interfaces.py), matching `docs/ARCHITECTURE.md` section 5 exactly. Concrete implementations are swapped via constructor args / CLI flags, not code edits — see [`run_pipeline.py`](run_pipeline.py).

**Design note on the Tracker/VehicleDetector pairing:** Ultralytics only exposes ByteTrack through the high-level `YOLO.track()` call, which fuses detection and tracking into a single pass (that's how the library is built internally). `YoloVehicleDetector.detect()` calls `.track()` and attaches the resulting track ID onto each `Detection` (free, since `.track()` already computed it) rather than running the YOLO forward pass twice. `ByteTrackTracker` still does real, independently-swappable work: it owns per-track history/state management (exactly what Phase 2's direction inference needs) and falls back to its own IoU-based ID assignment for detections that arrive without a pre-assigned `track_id` — so pairing a different `VehicleDetector` (one that only wraps `.predict()`) with `ByteTrackTracker` still works. Full details in the docstring of [`anpr/detectors/vehicle_detector.py`](anpr/detectors/vehicle_detector.py).

## Plate detector model — pick and license (read before deploying)

**Picked:** [`Koushim/yolov8-license-plate-detection`](https://huggingface.co/Koushim/yolov8-license-plate-detection) — a YOLOv8n checkpoint (single `license_plate` class), trained via the Ultralytics framework, loads through the same `ultralytics.YOLO` class already used for the vehicle detector.

**This was not the original pick — updated after actually running this end-to-end:**
- The original choice, `keremberke/yolov8n-license-plate-detection`, turned out to **no longer exist** — confirmed via a live 404 `RepositoryNotFoundError` from the HuggingFace Hub API while testing the download script, not just a wrong filename guess. HF listings (and search-engine caches of them) can go stale; the fix was to actually query the live Hub API (`HfApi().list_models(search=...)`) rather than trust cached search results.
- `Koushim/yolov8-license-plate-detection` was found that way and is a genuine improvement: it's **explicitly MIT-licensed by its author** (per its model card `license: mit`), not AGPL-3.0. MIT doesn't carry the copyleft/Enterprise-License question the original pick did.

**License — read this before any commercial/production deployment anyway:**
- The plate-detector checkpoint itself is MIT. However, `docs/ARCHITECTURE.md` still commits to Ultralytics YOLOv8 (COCO weights) + ByteTrack for the **vehicle detector and tracker**, and Ultralytics' stated policy is that YOLO-trained weights fall under **AGPL-3.0** by default unless you hold an **Ultralytics Enterprise License**. ([Ultralytics license page](https://www.ultralytics.com/license)) Picking an MIT plate-detector removes that concern for one stage of the pipeline, not the whole CV stack.
- **Action needed:** confirm with your institution whether an Enterprise License is obtainable, or whether open-sourcing the system is acceptable, before treating this as "production-ready" per `docs/PROJECT_SPEC.md`'s commercially-permissible requirement. Not resolved in code — it's a licensing/procurement decision.

## Setup

```bash
cd cv-service
python -m venv .venv
# Windows: .venv\Scripts\activate | macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Fetch plate-detector weights (see "Plate detector model" above)
python scripts/download_plate_model.py
```

`yolov8n.pt` (vehicle detector) doesn't need a manual download — Ultralytics fetches it automatically on first use.

## Running against a video file

```bash
python run_pipeline.py --video sample_data/gate1_sample.mp4
```

Outputs land in `output/` by default:
- `output/annotated.mp4` — input video with vehicle/plate boxes + OCR text drawn on every processed frame
- `output/events.json` — one JSON object per finalized track: `plate_text`, `ocr_confidence`, `track_id`, `frame_idx`/`timestamp_sec`, boxes, evidence paths
- `output/evidence/track_<id>_frame.jpg`, `track_<id>_plate.jpg` — best-reading evidence images per track

Each event is also printed to stdout as JSON as it's finalized (i.e. as soon as a vehicle leaves frame or the video ends), so you can watch detections happen live in the console.

Useful flags (`python run_pipeline.py --help` for the full list): `--device cuda:0` (GPU), `--frame-skip 2` (process every 2nd frame — see NFR-4/CPU fallback in `docs/ARCHITECTURE.md` section 8), `--vehicle-conf`, `--plate-conf`, `--ocr-min-conf` (confidence thresholds), and the `--plate-min-*`/`--plate-*-aspect`/`--plate-edge-margin` geometry filters.

Each run ends with a **rejection breakdown** — where plate reads were lost
(`no_plate_detected`, `format_mismatch`, `confidence_below_threshold`, …) as a
share of all attempts. Read it before tuning: it distinguishes "the plate was
never found" from "the plate was read correctly and then thrown away on a
threshold," which are the same from the events file alone and were the reason
a 33-vehicle evaluation's near-total miss rate went misdiagnosed (see below).

## The 33-vehicle evaluation: what was actually wrong (found and fixed)

The hand-labeled 33-vehicle evaluation (`sample_data/ground_truth.csv`,
three clips; results in
[`docs/ACCURACY_REPORT_MSNUPDATED_33.md`](../docs/ACCURACY_REPORT_MSNUPDATED_33.md))
measured **66.67% correct overall — cars 89.47%, bikes 35.71%**. The
headline problem is the car/bike gap, not a pipeline that reads nothing:

| Type | Total | Correct | Incorrect | Missed | Accuracy |
|---|---:|---:|---:|---:|---:|
| car | 19 | 17 | 0 | 2 | 89.47% |
| bike | 14 | 5 | 1 | 8 | 35.71% |

Of the 8 missed bikes, the report attributes 4 to "plate was not detected"
and 3 to "plate detected, but OCR failed". The fixes below target that
second group and are the reason to expect the bike number to move; they do
**not** address plate *detection* on fast/small bikes, which is still open.

Three bugs, each confirmed against real output from this project's own
footage rather than reasoned about:

**1. Every OCR text line in a plate crop was glued into one string.**
`PaddleOCRPlateReader.read()` did `"".join(rec_texts)`. Local plates carry a
province band ("PUNJAB") above the number, and OCR also picks up stray
noise, so a crop routinely produces two text lines. Real captured output:

```
rec_texts  = ['BUV 711', 'JUNJAU']     # 'JUNJAU' is the PUNJAB band, misread
rec_scores = [0.9988,    0.6661]
```

The plate number was read **perfectly at 0.9988**. Joining produced
`'BUV711JUNJAU'`, which fails the plate-format regex — so a correct read was
discarded as a format mismatch.

**2. Confidence was `min()` over all those lines.** The same join gave that
perfect read a confidence of 0.6661. Worse with a noise speck:
`['ARK2363', 'n']` scoring `[0.9909, 0.1747]` handed a 0.99 plate read to the
threshold check as **0.17**. So each read was being killed twice over, by two
independent mechanisms, and the run output couldn't tell you either had
happened.

Fixed by replacing "join everything, take the worst score" with *candidate
selection*: `PlateOCR.read_candidates()` enumerates each individual line plus
each contiguous run of lines joined in **visual** order (top-to-bottom —
PaddleOCR's returned order does not track layout, confirmed on two crops of
the same plate where the band came back first in one and second in the
other), and `PipelineRunner._select_reading` picks the best candidate that
matches the plate format. This is why it can't simply be "take the
highest-confidence line": a genuinely stacked two-line plate
(`['GAA', '545']`) must still join, while `['BUV 711', 'JUNJAU']` must not.
The plate-format pattern is the only signal that separates those two cases,
so it is applied *before* the confidence floor.

**3. `--ocr-min-conf` defaulted to 0.95, which was never justified.** It was
tuned against 3 lucky reads on one clip. Besides being applied to the
corrupted confidence above, it's too tight even on a clean read: `'BUV 711'`
scored **0.9307** on a genuine, correct read of a real plate here. Lowered to
**0.50**. With candidate selection fixed the format pattern does the real
filtering (it rejects province bands, `ENTRANCE`, and truncated reads like
`545` on shape alone), leaving this floor to catch only plate-shaped garbage
— observed noise sits at 0.17–0.84, genuine reads at 0.93–0.9998.

**Measured effect** — re-running OCR over the archived evidence crops, every
read the old code accepted is still accepted with identical text (no
regressions), and five previously-discarded reads now come back correct:
`JUNJAUBUV711@0.67 → BUV711@0.9988`, `7UNJASBUV711@0.77 → BUV711@0.9991`,
`PUNJAJBUV711@0.84 → BUV711@0.9977`, `ARK2363N@0.17 → ARK2363@0.9909`, and
`BUV711@0.93` (rejected purely by the 0.95 floor) now accepted.

**4. The scorer inferred the source video from the events-JSON filename.**
`scripts/score_accuracy.py` mapped `output/review2_events.json` →
`review2.mp4`, then matched ground truth grouped by
`(filename, vehicle_type)`. Ground truth says `dataset_clear_01.mp4`. When
the filename sets don't intersect, **every** ground-truth row scores
"Missed" and **every** correct detection is reported as a false positive —
with no hint in the output that filenames were the cause. Reproduced on
this repo's checked-in `output/*_events.json`: 33 vehicles, 100% missed, 9
false positives.

**This did not cause the 33-vehicle result** — that run's outputs were
named after the dataset clips, so matching worked. It's a live trap for
anyone who names an output run after the branch or the experiment instead
of the video, not the explanation for any number in the report above.

Fixed at the source: `run_pipeline.py` now records `source_video` on each
event (it knows what it was given; the scorer was only ever guessing), the
old heuristic remains as a fallback for pre-fix event files, and
`warn_on_filename_mismatch` now states a non-overlapping filename set
outright instead of printing a plausible-looking 0%.

### Still open after these fixes

Three problems the report surfaces that the changes above do **not** touch:

- **Ground-truth timestamps are off by more than the matching window.**
  The *double-penalty* half of this is fixed: a correct read landing
  outside the window used to be counted as "Missed" **and** listed as a
  false positive, describing one correct read as two different failures.
  `find_timestamp_discrepancies` now reports these in their own section
  and keeps them out of the false-positive count. Replaying the
  benchmark's own reported events through the fixed scorer finds **two**
  such cases, not one: `ABA196` (labeled 6.0s, read correctly at 14.9s,
  gap +8.9s) and `AUT094` (labeled 43.0s, read correctly at 37.35s, gap
  −5.6s — barely outside the 5s window, and reported as a detection
  failure caused by the multi-vehicle scene). Both were also counted among
  the 10 false positives.

  The underlying data problem is *not* fixed and is not a code question:
  21 of 33 rows are flagged ambiguous, and the multi-vehicle clips'
  timestamps are all exactly 6s or 9s. Either the labels need tightening
  or `--time-window` needs to reflect how approximate they are. Note the
  accuracy percentages are deliberately unchanged by this fix — matching
  on plate text and then scoring that match as correct would be grading
  OCR with its own output as the answer key.
- **Duplicate events inflate the false-positive count.** With the two
  mistimed reads above accounted for, the remaining 8 reported false
  positives are all second/third detections of a plate that was *also*
  read correctly (`AWJ431` ×3, `BPG516` ×3, `ASJ765`, `BEW278`, `AED186`,
  `AAR356`), matching the 10 exact-plate duplicate pairs. In other words
  the benchmark's false-positive count is **not** measuring wrong reads at
  all — it's measuring tracker dedup failures plus label timing. The
  tracker's `_reconcile_id` IoU merge isn't bridging these; see the
  ByteTrack ID-stability note under "Known risks".
- **Bike plate *detection*.** 4 of the 8 missed bikes never produced a
  plate box at all. Nothing in this change affects that.

**Diagnosing the next run.** The reason the OCR bugs took a full evaluation
to spot is that "no plate found" and "plate read correctly then thrown away
on a threshold" looked identical from the outside. `run_pipeline.py` now ends with
a rejection breakdown — how many attempts died at `no_plate_detected`,
`format_mismatch`, `confidence_below_threshold`, etc., and what the top
reason means. Check it before tuning anything. The plate-box geometry filters
(`--plate-min-width/height`, `--plate-min-aspect`, `--plate-max-aspect`,
`--plate-edge-margin`) are also CLI flags now rather than hardcoded
constructor defaults, so a `no_plate_detected`-dominated run can be swept
without editing code.

**Still unverified** — the dataset clips aren't in this repo (see
`sample_data/README.md`), so the fixes above are confirmed against archived
evidence crops and unit tests, **not** against a full re-run of the 33-vehicle
set. The 66.67% / 89.47% / 35.71% figures above are the numbers **before**
these fixes; nobody has measured after. Re-run and re-score before quoting
any accuracy number:

```bash
python run_pipeline.py --video sample_data/dataset_clear_01.mp4 \
  --events-json output/dataset_clear_01_events.json \
  --evidence-dir output/dataset_clear_01_evidence
python scripts/score_accuracy.py --report docs/ACCURACY_REPORT.md
```

One hypothesis in this area is *not* confirmed: `--plate-min-aspect` was
lowered from 1.0 to 0.8 on the reasoning that local motorbike plates are
stacked two-line and near-square, so genuine ones at a gate-camera angle
could measure just under 1.0 and be discarded before OCR ever saw them. The
motorcycle plates measurable here ran 1.55–1.72, which doesn't demonstrate
the problem. Treat it as a widened guard rail, and check the rejection
summary to see whether plate *detection* is actually a bottleneck.

## Known risks / design trade-offs

- **Python version:** requires **3.10 or 3.11** — `paddlepaddle==3.2.0` has no published wheel for 3.12+ (confirmed: 3.14 failed with "no matching distribution" during setup). Check installed versions with `py -0p` (Windows) and target one explicitly: `py -3.10 -m venv .venv`.
- **PaddleOCR API volatility:** PaddleOCR's Python API changed materially between the 2.x line (`.ocr(img, cls=True)`) and the 3.x line pinned here (`.predict(img)` with a different result schema). [`anpr/ocr/paddle_ocr.py`](anpr/ocr/paddle_ocr.py) extracts results defensively across a couple of known shapes and logs a clear error (rather than silently returning garbage) if a future point release changes the schema again.
- **`ultralyticsplus` dropped:** originally used as a convenience loader for `keremberke/*` HF models in `scripts/download_plate_model.py`, but confirmed (by actually running `pip install`) to hard-pin `ultralytics<8.1.0`, which conflicts with the `ultralytics==8.4.53` the core pipeline needs. Removed from `requirements.txt`; the download script now uses `huggingface_hub` directly instead.
- **torch/paddle DLL load-order conflict on Windows (confirmed):** if `paddle`/`paddleocr` is imported *before* `torch`/`ultralytics` in the same process, `import torch` later fails with `OSError: [WinError 127] ... loading "torch\lib\shm.dll"`. Importing in the reverse order (torch first) works fine. `run_pipeline.py` is written to instantiate `YoloVehicleDetector`/`YoloPlateDetector` before `PaddleOCRPlateReader` specifically to keep this order — don't reorder that without re-testing. If you use these components outside `run_pipeline.py`, import/instantiate anything `ultralytics`-based before anything `paddleocr`-based in the same process.
- **AGPL-3.0 exposure:** see "Plate detector model" above — applies to the whole Ultralytics-based CV stack, not just this model.
- **Own-overlay-as-plate contamination (found and fixed):** an earlier version of `anpr/pipeline/runner.py` drew each track's vehicle-box label directly onto the frame *before* running plate detection/OCR on that same frame array. On a real test clip, a vehicle mostly off-frame had its own `"#11 car 0.39"` label detected as a "plate" and OCR read it back — cleaned of punctuation/spaces, `"#11 car 0.39"` becomes exactly `"11CAR039"`, which is what got logged. Fixed by running all detection/OCR against an unannotated `frame.copy()` and only drawing onto the display/output frame afterward — see the comment in `PipelineRunner._process_frame`. If you ever restructure that method, keep detection/OCR reading from a clean copy.
- **Best-OCR-confidence frame isn't always the most complete read (found and fixed):** observed on the same test clip — a track's plate was fully legible ("AAL 988") in an earlier frame (OCR confidence 0.94), but a later frame where the plate crop was clipped by the bottom image edge scored *higher* OCR confidence (0.9999) on the truncated text "988" alone. Since `BestReadingAggregator` picks strictly by highest `ocr_confidence`, it kept the truncated read over the complete one for that finalized event. Fixed at the source: `YoloPlateDetector.detect()` now rejects any plate box that touches the camera frame's own edge (not the vehicle crop's edge — see the `frame_edge_margin_px` check), since that box is very likely physically cut off. Re-tested: the truncated "988" read no longer occurs; the same track now finalizes on the complete "AAL988" reading. This is a same-frame geometric heuristic, not a content-aware "is this plate actually complete" check, so a plate that's genuinely fully visible but happens to sit right at the frame edge would also be rejected — an acceptable trade-off for now.
- **Same vehicle logged as two separate detection events (found and fixed):** confirmed on the same test clip — the same physical car and the same physical motorcycle each produced two separate detection events under two different track IDs, because ByteTrack's own track either wasn't "confirmed" yet (reporting no ID for the first few frames, so our IoU fallback assigned a temporary one) or got briefly lost and re-acquired under a new real ID mid-pass. Fixed with a same-video, in-memory reconciliation heuristic in `ByteTrackTracker._reconcile_id`: whenever a brand-new track ID appears with no existing track entry, check currently-active tracks for a high-IoU (≥0.5) match against its box; if found, merge histories and prefer a real ByteTrack ID as canonical over a fallback one. `BestReadingAggregator.migrate()` carries over the better-scoring accumulated reading so nothing is lost across the merge. Re-tested: both vehicles now produce exactly one event each. This is an IoU-proximity heuristic, not true appearance-based re-identification — it can't bridge a gap longer than `max_age_frames` (30 frames / ~1.5s by default), and two *different* vehicles that happen to overlap heavily at the merge instant could theoretically be wrongly fused. Real cross-track dedup spanning longer gaps (e.g. a vehicle leaving and re-entering frame) is still Phase 2 scope, per `docs/ARCHITECTURE.md` section 2.1's short-TTL dedup cache.
- **Minor cosmetic loose end from the merge fix:** when a fallback/absorbed track ID is merged away, its evidence images already written to disk under the old ID (e.g. `track_1000000_frame.jpg`) aren't deleted — they just become orphaned files not referenced by the final `events.json`. Harmless, not cleaned up.
- **Plate detector false-positives on non-plate rectangular text (found and fixed):** on a second test clip, the detector boxed an "Entrance" sign in the scene and OCR correctly read it as `"ENTRANCE"` (conf 0.9999) — a real detection and a real OCR read, just not a plate. Fixed with two complementary checks, since neither alone is sufficient: (1) an aspect-ratio filter in `YoloPlateDetector` (`min_aspect_ratio`/`max_aspect_ratio`, now 0.8–3.0, see the 33-vehicle section above for why the lower bound moved off 1.0) — every genuine plate box measured so far is ~1.4–1.9 width:height, the "Entrance" sign's box was 4.47; (2) a plate-format regex in `PaddleOCRPlateReader` (`DEFAULT_PLATE_PATTERN = ^[A-Z]{2,4}[0-9]{2,4}$`), applied by `PipelineRunner` — every genuine plate read normalizes to letters-then-digits, `"ENTRANCE"` is pure letters. That regex turned out to carry far more weight than intended: it is also the signal that separates a plate number from the province band glued onto it, and is now the pipeline's primary OCR filter. Re-tested: the false positive is gone; the pipeline now finds and correctly reads the real plate (`GAA545`) on the actual vehicle in that frame instead. Both are heuristics tuned to the plate formats/framing observed so far, not content-aware "is this actually a plate" understanding — a different plate format (e.g. different letter/digit counts) needs `DEFAULT_PLATE_PATTERN` adjusted, and `min_aspect_ratio`/`max_aspect_ratio` may need retuning for other camera angles.
- **ByteTrack ID stability under occlusion/longer gaps — not fully solved:** the merge fix above only bridges gaps up to `max_age_frames` (~1.5s default). For more robust re-identification across longer gaps, Ultralytics' BoT-SORT tracker (adds appearance-based ReID on top of Kalman+IoU) is a one-line config swap (`tracker_cfg="botsort.yaml"` instead of `"bytetrack.yaml"`) since the tracker is already selected via config — not yet tried. Trade-off: extra compute per frame on a pipeline that's already too slow for NFR-1 on CPU (see performance note below).

## Real test runs (input footage: campus gate camera, 2560×1440 @ 20fps)

**Clip 1** — 239-frame/~12s, one car (plate `AAL 988`) and one motorcycle (plate `BAG 9976`) passing an underground gate entrance. After the dedup/truncation fixes above, produces exactly **3 detection events** (down from 6 pre-fix):
- **Car** — one event, `AAL988` at 0.9998 confidence. Correct, complete, single event for the whole pass (previously split into 2 events, one of which was a truncated "988").
- **Motorcycle** — one event, `BAG9976` at 0.9957 confidence. Correct, complete, single event for the whole pass (previously split into 2 events).
- **One spurious low-confidence read ("1") — found and fixed**: came from a 20×12px plate-detector false positive, too small to physically contain a legible plate. Fixed with a minimum-box-size filter (`min_plate_width_px`/`min_plate_height_px`, defaults 24×10px).

**Clip 2** — 634-frame/~32s, multiple cars and a motorcycle. Produces **4 detection events**, 3 confirmed correct against evidence crops (`BUV711`, `ARK2363`, and — after the aspect-ratio/regex fix above — `GAA545` in place of the `"ENTRANCE"` false positive). One (`FN211`, conf 0.80, tiny 37×21px box) is unverified — plausibly a genuine but very small/distant read, plausibly noise; worth a closer manual look, and a candidate for tightening `min_plate_width_px`/`min_plate_height_px` further if it turns out to be spurious.

**Performance (CPU-only, no GPU):** ~1.3–1.4s/processed-frame at 2560×1440 including PaddleOCR calls (measured from timestamped logs: 200 frames in ~285s once models were warm), ~5.5-6 minutes wall-clock per ~12s clip at this frame rate. PaddleOCR also downloads ~140MB of its own OCR models on first use, separate from `requirements.txt` — a one-time cost, not counted above. Even at this rate, one frame alone takes roughly half of NFR-1's entire 2-3s detection-to-alert budget — nowhere close once you add the backend's own consume/match/write/push latency on top. A GPU (RTX 4090 or 5060 planned, not yet available) should close most of this gap; until then, `--frame-skip` and/or a lower processing resolution are the available levers.

## Testing / evaluating accuracy

**Sample video to test with:** a short (1-3 minute) daytime clip of a real or simulated gate lane, ideally shot at roughly the 30-45° off-head-on angle `docs/ARCHITECTURE.md` section 7.4 recommends for Phase 2's direction inference (doesn't matter for this phase, but worth using footage you'll reuse there). Options, cheapest first:
1. **A phone video you take yourself** of a driveway, parking lot entrance, or street with passing traffic — most representative of your actual gate cameras' likely mounting height/angle.
2. **A public traffic/ANPR dataset clip** — search "license plate detection dataset video" or "traffic camera footage" on Roboflow Universe or Kaggle; several include short raw video alongside frame-by-frame annotations, e.g. **UFPR-ALPR** or **CCPD** derivatives (check each dataset's license before using in anything beyond local testing — most academic ANPR datasets are research-only, not commercially licensed).
3. **Synthetic/staged clip** — park a car with a legible plate and record it driving past at gate speed (~5-15 km/h); simplest way to get a clean baseline before testing harder conditions.

Get at least one clip with **cars only**, then a **separate** clip with **motorcycles** — `docs/PROJECT_SPEC.md` section 5 and `docs/FRD.md` section 5.3 both call out that motorbike accuracy needs to be measured and reported separately (smaller, often angled plates), not blended into one number.

**How to evaluate:**
1. Run `python run_pipeline.py --video <your_clip>` and watch `output/annotated.mp4` — confirm vehicle boxes track correctly (no id switching across an uninterrupted pass) and plate boxes land on the actual plate, not background text/other shapes.
2. Manually count ground truth: how many distinct vehicle passes are in the clip, and what their actual plate numbers are.
3. Compare against `output/events.json`:
   - **Detection recall** — did every vehicle pass produce exactly one event (no missed vehicles, no duplicate events for one pass)?
   - **OCR accuracy** — for each event, does `plate_text` exactly match (or come close — note common confusions like `0`/`O`, `1`/`I`) the real plate? Track this **separately for cars vs. motorbikes** per the placeholder targets in `docs/PROJECT_SPEC.md` section 5 (cars >90%, motorbikes >75%) — treat those as targets to validate against, not assumed truths.
   - **Confidence calibration** — do low `ocr_confidence` events correlate with actually-wrong reads? If high-confidence reads are frequently wrong, the confidence threshold (`--ocr-min-conf`) needs raising. Raise it on evidence of *wrong reads being accepted*, not on a hunch that higher is safer: the 0.95 default that preceded the current 0.50 was set that way and silently discarded correct reads scoring 0.93.
4. If plate *detection* (not OCR) is the weak link — boxes missing plates entirely, or landing on the wrong region — that's the moment to swap `--plate-weights` for a different pretrained model rather than proceeding to Phase 2 with a broken foundation (per the kickoff doc's Phase 1 tip: cheaper to swap now than after Phases 2-5 are built around it).
