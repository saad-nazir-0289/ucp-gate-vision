#!/usr/bin/env python3
"""
Score the ANPR pipeline's accuracy against a hand-labeled ground-truth CSV.

This is an independent rewrite, not a patch of the version in PR #10
(`hifza/accuracy-eval`). That version's core matching idea was sound
(verified by review: closest-timestamp-first global assignment, so one
pipeline event can't double-count against two ground-truth rows) and is
kept here. Six concrete issues found reviewing it and testing the pipeline
are fixed:

1. **Undeclared dependency.** It used `pandas`, which isn't in
   requirements.txt and only worked because paddlex happens to pull it in
   transitively. This script otherwise uses only the standard library, plus
   one deliberate, explicitly-declared dependency added later (`scipy`,
   for exact bipartite matching — see fix #7) rather than another
   accidental transitive one.

2. **Vehicle-type normalization didn't fold truck/bus into car.** Confirmed
   by testing (see cv-service/README.md "Known risks"): the generic
   COCO-pretrained vehicle detector can classify a real car as "truck" at
   higher inference resolution or a lower confidence threshold. Without
   this mapping, a correctly-read plate on a misclassified vehicle scores
   as a false "Missed" — the exact bug that would have silently deflated
   the very fixes meant to improve accuracy.

3. **Ambiguous ground truth was invisible.** When two ground-truth rows of
   the same vehicle_type in the same file sit within the matching window of
   each other (common in the multi-vehicle clips — the exact category with
   the worst measured accuracy), timestamp-only matching can attribute a
   correct read to the wrong plate without any signal that happened. This
   script flags those rows explicitly instead of silently guessing.

4. **Binary correct/incorrect lost information.** "LEM2025" -> "LEH2024"
   (one character off) and "LEM2025" -> completely different garbage were
   both just "Incorrect" before. This computes Character Error Rate (CER)
   for every non-exact match, so a near-miss is visibly different from a
   total failure.

5. **False-positive events were never counted.** The original scorer only
   walked from ground truth outward, so a spurious pipeline event matching
   no real vehicle was invisible in the headline numbers. This reports
   unmatched pipeline events explicitly.

6. **Duplicate events were only flagged as "candidates," never actually
   identified.** This checks same-file events for near-identical plate text
   within a short time window and reports them as likely duplicate
   detections of one physical pass (a tracking/dedup issue, scored
   separately from OCR accuracy so it doesn't get conflated with it).

Two more issues found in a follow-up review of this file specifically
(not PR #10's version — these are ones this rewrite introduced):

7. **Greedy timestamp matching could produce a worse result than
   necessary.** `match_events_to_ground_truth` used to sort all
   (ground-truth, event) candidate pairs by distance and greedily assign
   the globally-closest pair first. This isn't guaranteed optimal — a
   concrete failure mode: if ground-truth row A's *only* candidate event is
   also row B's *closest* (but not only) candidate, greedy grabs that event
   for B first, leaving A unmatched ("Missed") even though assigning A its
   sole option and B its second-best would have matched both. See
   `tests/test_score_accuracy.py::test_greedy_would_fail_this_case...` for
   the exact scenario.

   First fix used exact backtracking search per connected component of the
   candidate graph, correct but exponential in the worst case — it fell
   back to the old non-optimal greedy heuristic above 8 ground-truth rows
   in one cluster, meaning a large enough multi-vehicle scene would
   silently lose the guarantee this was built for. Fixed again: now solved
   as a minimum-cost bipartite assignment via
   `scipy.optimize.linear_sum_assignment` (genuinely polynomial, O(n^3),
   for any input size — no fallback needed at all). Maximize-cardinality-
   then-minimize-distance is achieved with a single min-cost run by padding
   the cost matrix to square with zero-cost dummy rows/columns
   ("leave unmatched") and giving out-of-window pairs a cost large enough
   that using more real, in-window edges always wins. `scipy` is added as
   an explicit dependency for this — a deliberate, declared use of a
   well-tested library for a genuinely nontrivial algorithm, not the same
   mistake as the undeclared transitive `pandas` dependency fixed in #1
   (that one was doing a job pure Python did just as well).

9. **Ambiguity detection was one-sided.** Only flagged a ground-truth row
   when 2+ *events* competed for it — missed the mirror case, one event
   that's a candidate for 2+ *ground-truth rows*. From that GT row's own
   perspective it only ever saw one candidate event, so the one-sided check
   never flagged it, even though the match was still a guess (some other
   GT row wanted that same event too). Fixed: ambiguity is now checked in
   both directions.

8. **Duplicate detection could falsely flag a legitimate plate.** The
   original version (15s window, edit-distance <=1 all called "duplicate")
   would flag a vehicle that genuinely reappears within 15 seconds — e.g.
   entering, realizing it's the wrong gate, and immediately driving back
   out — as a tracking bug, when it's actually two correct, separate
   events. Fixed by: shrinking the default window to 5s (closer to the
   tracker's own reconciliation window, so a duplicate surviving past that
   is more informative either way); splitting exact-plate matches from
   near-plate (edit-distance 1) matches, since the latter could just as
   easily be two different, coincidentally similar plates; and rewording
   every report to "candidate duplicate" with the legitimate-reappearance
   explanation stated explicitly, instead of asserting it's a bug.

One more found by actually evaluating a 33-vehicle dataset with it — the
worst of the lot, because it silently reported a working pipeline as a
broken one:

10. **The source video was inferred from the events-JSON filename.**
    `output/review2_events.json` was assumed to be `review2.mp4`. Matching
    groups by (filename, vehicle_type), so naming an output run anything
    other than the dataset video meant no event could match any
    ground-truth row: every row scored "Missed", every correct detection
    was reported as a false positive, and nothing in the output hinted
    that filenames were the reason. Fixed at the source — run_pipeline.py
    now records `source_video` on each event (it knows what it was given;
    this script was only ever guessing) — plus `warn_on_filename_mismatch`
    so a non-overlapping filename set is stated outright instead of
    printing a plausible-looking 0%.

And one more, from the 33-vehicle benchmark itself:

11. **A correct read could be penalized twice.** `ABA196` was read
    correctly at 14.9s with ~1.00 confidence; ground truth labels it at
    6.0s. Outside the 5s window, so it scored as a "Missed" vehicle AND
    was listed as a false-positive event — one correct read reported as
    two different failures. `find_timestamp_discrepancies` now identifies
    these and reports them in their own section, and they no longer
    inflate the false-positive count. Deliberately a reporting fix, not a
    matching one: the accuracy percentages are unchanged, because matching
    on plate text and then scoring that match as correct would be grading
    OCR with its own output as the answer key.

Also handles a real data quirk explicitly: sample_data/README.md notes that
a blank `plate` value in ground_truth.csv means the plate "wasn't readable
from human eye." Those rows can't fairly judge OCR *correctness* (there's
nothing to compare against) — they're scored only for detection recall
(was a vehicle logged at all near that timestamp), kept out of the
correct/incorrect/accuracy-percentage numbers entirely. Silently comparing
a real OCR read against an empty ground-truth string would score every
such case as "Incorrect," which is wrong on its face.

Usage:
    python scripts/score_accuracy.py
    python scripts/score_accuracy.py --time-window 3.0 --report docs/ACCURACY_REPORT.md
    python scripts/score_accuracy.py --ground-truth sample_data/ground_truth.csv --output-dir output
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from scipy.optimize import linear_sum_assignment

logging.basicConfig(format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_GROUND_TRUTH = BASE_DIR / "sample_data" / "ground_truth.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"
DEFAULT_TIME_WINDOW = 5.0  # seconds — same as PR #10's version, kept for comparability
DEFAULT_DUPLICATE_WINDOW = 5.0  # seconds — see find_duplicate_events docstring for why this shrunk from 15s


# ============================================================================
# Normalization — kept close to PR #10's version where it was already right,
# fixed where testing showed it wasn't.
# ============================================================================


def normalize_plate(value: str | None) -> str:
    if not value:
        return ""
    return "".join(str(value).upper().split())


def normalize_vehicle_type(value: str | None) -> str:
    v = (value or "").strip().lower()
    if v in {"bike", "motorcycle", "motorbike"}:
        return "bike"
    # FIX #2: a car-like vehicle can be misclassified as bus/truck by the
    # generic COCO-pretrained detector — confirmed by testing, not
    # theoretical. Treat them as the same ground-truth category so a
    # correct plate read doesn't get penalized for an unrelated
    # classification quirk.
    if v in {"car", "truck", "bus", "van", "suv"}:
        return "car"
    return v


def normalize_condition(value: str | None) -> str:
    v = (value or "").strip().lower()
    for prefix in ("clear", "day", "dusk", "night", "rain"):
        if v.startswith(prefix):
            return prefix
    return v or "unknown"


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance. Plates are short (<12 chars) — plain O(n*m) is plenty."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def character_error_rate(ground_truth: str, predicted: str) -> float | None:
    """FIX #4. None when there's nothing to compare against (empty ground truth)."""
    if not ground_truth:
        return None
    return edit_distance(ground_truth, predicted) / len(ground_truth)


# ============================================================================
# Data loading
# ============================================================================


@dataclass
class GroundTruthRow:
    index: int
    filename: str
    timestamp_sec: float
    vehicle_type: str
    plate: str  # "" means illegible to the human labeler — see module docstring
    condition: str
    notes: str
    ambiguous: bool = False  # FIX #3, set inside match_events_to_ground_truth()


@dataclass
class PipelineEvent:
    index: int
    filename: str
    timestamp_sec: float
    vehicle_type: str
    plate: str
    track_id: int
    # The confidence the pipeline accepted this reading at. Surfaced in the
    # report because without it an "Incorrect" row cannot be acted on: a wrong
    # read at 0.55 is a threshold that is too loose, a wrong read at 0.99 is a
    # model problem, and those need opposite responses. The V2 benchmark
    # (60.61%, down from 66.67% after --ocr-min-conf was lowered 0.95 -> 0.50)
    # could not distinguish them.
    ocr_confidence: float = 0.0
    raw: dict = field(default_factory=dict)


def load_ground_truth(path: Path) -> list[GroundTruthRow]:
    rows: list[GroundTruthRow] = []
    with open(path, newline="", encoding="utf-8") as f:
        for i, raw_row in enumerate(csv.DictReader(f)):
            filename = (raw_row.get("filename") or "").strip()
            if not filename:
                continue  # blank separator lines in the CSV
            rows.append(
                GroundTruthRow(
                    index=i,
                    filename=filename,
                    timestamp_sec=float(raw_row.get("timestamp_sec") or 0),
                    vehicle_type=normalize_vehicle_type(raw_row.get("vehicle_type")),
                    plate=normalize_plate(raw_row.get("plate")),
                    condition=normalize_condition(raw_row.get("condition")),
                    notes=(raw_row.get("notes") or "").strip(),
                )
            )
    return rows


def load_pipeline_events(output_dir: Path) -> list[PipelineEvent]:
    """Load events, taking each one's source video from the event itself.

    CRITICAL FIX (this is what made a real, correctly-read car score as
    "Missed"). This used to derive the source video purely from the
    events-JSON filename: `review2_events.json` -> `review2.mp4`. Matching
    then groups by (filename, vehicle_type), so unless a run's output was
    named exactly after the dataset video, NO event could ever match ANY
    ground-truth row — every row scored "Missed" and every correct
    detection was reported as a false positive. Nothing in the output said
    so; the numbers just looked like a catastrophically bad model.

    `run_pipeline.py` now records `source_video` in each event (the actual
    `--video` basename), which is where that fact belongs. The filename
    heuristic is kept only as a fallback for events.json files written
    before that field existed, and `warn_on_filename_mismatch` below turns
    the silent-total-miss failure into a loud, explicit message.
    """
    events: list[PipelineEvent] = []
    for events_path in sorted(output_dir.glob("*_events.json")):
        derived = events_path.name[: -len("_events.json")] + ".mp4"
        with open(events_path, encoding="utf-8") as f:
            data = json.load(f)
        used_fallback = False
        for raw_event in data:
            source_video = (raw_event.get("source_video") or "").strip()
            if not source_video:
                source_video = derived
                used_fallback = True
            events.append(
                PipelineEvent(
                    index=len(events),
                    filename=Path(source_video).name,
                    timestamp_sec=float(raw_event.get("timestamp_sec", 0)),
                    vehicle_type=normalize_vehicle_type(raw_event.get("vehicle_class")),
                    plate=normalize_plate(raw_event.get("plate_text")),
                    track_id=raw_event.get("track_id", -1),
                    ocr_confidence=float(raw_event.get("ocr_confidence", 0.0) or 0.0),
                    raw=raw_event,
                )
            )
        if used_fallback and data:
            logger.warning(
                "%s has events with no 'source_video' field — guessing the source video is %r from "
                "the JSON filename. If that is not the actual video, every ground-truth row for it "
                "will score as Missed. Re-run run_pipeline.py to record it properly.",
                events_path.name, derived,
            )
    return events


def warn_on_filename_mismatch(gt_rows: list[GroundTruthRow], events: list[PipelineEvent]) -> None:
    """Fail loudly when ground truth and events describe different videos.

    A silent 0% is the single most misleading output this script can
    produce — it is indistinguishable from a genuinely broken model, and
    it is exactly what the `source_video` bug above caused. Matching can
    only ever pair rows within the same filename, so if the two sets of
    filenames don't intersect, say so instead of printing 100% Missed.
    """
    gt_files = {gt.filename for gt in gt_rows}
    event_files = {ev.filename for ev in events}
    if gt_files & event_files:
        return
    print("=" * 60, file=sys.stderr)
    print("WARNING: no filename is common to ground truth and pipeline events.", file=sys.stderr)
    print("Every ground-truth row will score as 'Missed' and every event as a", file=sys.stderr)
    print("false positive — because matching only ever pairs within one video,", file=sys.stderr)
    print("not because the pipeline failed. Fix the filenames, then re-score.", file=sys.stderr)
    print(f"  ground_truth.csv filenames: {sorted(gt_files)}", file=sys.stderr)
    print(f"  pipeline event filenames:   {sorted(event_files)}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


# ============================================================================
# Matching
# ============================================================================


def _flag_ambiguous_ground_truth(gt_rows: list[GroundTruthRow], candidates: list[tuple[float, int, int]]) -> None:
    """FIX #9 (external review): ambiguity must be checked in BOTH
    directions. The previous version only flagged a GT row when 2+ EVENTS
    competed for it. It missed the mirror case: one event that's a
    candidate for 2+ GT rows. From that GT row's own perspective it only
    ever saw a single candidate event, so the one-sided check never fired —
    even though the match is still a guess, since some other GT row wanted
    that same event too.

    Earlier still (now removed), ambiguity was based on raw GT-to-GT
    timestamp gaps (2x the window), which fired on almost all normal
    traffic — caught by testing against synthetic fixtures before trusting
    it. Candidate-contention (in either direction) is precise: it doesn't
    fire on routinely-spaced traffic, only where the matcher had a real
    choice to make.
    """
    candidate_count_by_gt: dict[int, int] = {}
    candidate_count_by_event: dict[int, int] = {}
    for _, gt_index, ev_index in candidates:
        candidate_count_by_gt[gt_index] = candidate_count_by_gt.get(gt_index, 0) + 1
        candidate_count_by_event[ev_index] = candidate_count_by_event.get(ev_index, 0) + 1

    gt_by_index = {gt.index: gt for gt in gt_rows}
    for _, gt_index, ev_index in candidates:
        if candidate_count_by_gt.get(gt_index, 0) > 1 or candidate_count_by_event.get(ev_index, 0) > 1:
            gt_by_index[gt_index].ambiguous = True


def match_events_to_ground_truth(
    gt_rows: list[GroundTruthRow], events: list[PipelineEvent], time_window: float
) -> dict[int, int]:
    """Exact minimum-cost maximum-cardinality bipartite matching (FIX #7,
    revised) via `scipy.optimize.linear_sum_assignment` — one event per GT
    row, one GT row per event, maximizing match count first, then
    minimizing total distance among assignments tied on count.

    Solved per (filename, vehicle_type) group (a GT row and event in
    different files/types are never candidates for each other regardless
    of timestamp, same as before). Within a group, maximize-cardinality-
    then-minimize-distance is achieved with a SINGLE min-cost run: the cost
    matrix is padded to square with zero-cost dummy rows/columns
    (representing "leave this one unmatched"), and any out-of-window real
    pair gets a cost large enough that the solver always prefers using more
    real, in-window edges over fewer — so it can never be cheaper to leave
    a matchable row unmatched just to give another row a slightly shorter
    distance.

    This replaced an earlier version using exact backtracking per connected
    component: correct but exponential in the worst case, so it fell back
    to a non-optimal greedy heuristic above 8 GT rows in one cluster — a
    large enough multi-vehicle scene would silently lose the very guarantee
    it was built for. linear_sum_assignment is O(n^3) for any input size;
    no fallback is needed at all.
    """
    if not gt_rows or not events:
        return {}

    groups_gt: dict[tuple[str, str], list[GroundTruthRow]] = {}
    for gt in gt_rows:
        groups_gt.setdefault((gt.filename, gt.vehicle_type), []).append(gt)
    groups_ev: dict[tuple[str, str], list[PipelineEvent]] = {}
    for ev in events:
        groups_ev.setdefault((ev.filename, ev.vehicle_type), []).append(ev)

    candidates: list[tuple[float, int, int]] = []  # for ambiguity flagging, across all groups
    assignments: dict[int, int] = {}

    for key, gts in groups_gt.items():
        evs = groups_ev.get(key, [])
        if not evs:
            continue

        n, m = len(gts), len(evs)
        size = max(n, m)
        # Any real edge costs at most time_window, and a matching uses at
        # most `size` edges — this comfortably dominates any achievable
        # real-edge total, so the solver never prefers an out-of-window
        # pair over a zero-cost dummy (i.e. leaving something unmatched).
        large_penalty = (time_window + 1.0) * (size + 1) * 10.0

        cost = [[large_penalty] * size for _ in range(size)]
        for i, gt in enumerate(gts):
            for j, ev in enumerate(evs):
                distance = abs(ev.timestamp_sec - gt.timestamp_sec)
                if distance <= time_window:
                    cost[i][j] = distance
                    candidates.append((distance, gt.index, ev.index))
        for i in range(size):
            for j in range(size):
                if i >= n or j >= m:
                    cost[i][j] = 0.0  # dummy row/column — costs nothing to "match"

        row_idx, col_idx = linear_sum_assignment(cost)
        for i, j in zip(row_idx, col_idx):
            if i < n and j < m and cost[i][j] <= time_window:
                assignments[gts[i].index] = evs[j].index

    _flag_ambiguous_ground_truth(gt_rows, candidates)
    return assignments


def find_timestamp_discrepancies(
    gt_rows: list[GroundTruthRow], events: list[PipelineEvent], assignments: dict[int, int]
) -> list[tuple[GroundTruthRow, PipelineEvent]]:
    """Unmatched ground-truth rows whose exact plate text WAS read, just
    outside the matching window. Returns (gt_row, event) pairs.

    FIX #11, from the 33-vehicle benchmark. `ABA196` in
    dataset_multiple_vehicles_02.mp4 was read correctly at 14.9s with OCR
    confidence ~1.00; ground truth labels that vehicle at 6.0s. 8.9s apart,
    so with a 5s window it didn't match — and the report then counted it
    BOTH as a "Missed" vehicle AND as a false-positive event. One correct
    read, penalized twice, described as two different kinds of failure.

    Deliberately a *reporting* fix, not a matching one: these pairs do NOT
    become "Correct" and the accuracy percentages are unchanged. Matching
    on plate text and then scoring that match as a correct read would be
    circular — grading OCR with OCR's own output as the answer key, which
    would credit any event whose text happened to equal some plate in the
    file regardless of when it happened. What is plainly wrong, and is
    fixed, is calling such an event a *false positive*: it read a real
    plate belonging to a real labeled vehicle in that clip. Whether the
    ground-truth timestamp or the window is at fault is a question for
    whoever labeled the data, so this reports the discrepancy and the size
    of the gap rather than silently resolving it either way.
    """
    matched_event_indices = set(assignments.values())
    unmatched_events = [ev for ev in events if ev.index not in matched_event_indices and ev.plate]
    if not unmatched_events:
        return []

    by_file_plate: dict[tuple[str, str], list[PipelineEvent]] = {}
    for ev in unmatched_events:
        by_file_plate.setdefault((ev.filename, ev.plate), []).append(ev)

    claimed: set[int] = set()
    pairs: list[tuple[GroundTruthRow, PipelineEvent]] = []
    for gt in gt_rows:
        if gt.index in assignments or not gt.plate:
            continue
        available = [ev for ev in by_file_plate.get((gt.filename, gt.plate), []) if ev.index not in claimed]
        if not available:
            continue
        # Nearest in time, so a plate legitimately appearing twice in one
        # clip pairs each ground-truth row with its own closest event.
        best = min(available, key=lambda ev: abs(ev.timestamp_sec - gt.timestamp_sec))
        claimed.add(best.index)
        pairs.append((gt, best))
    return pairs


def sweep_ocr_confidence(
    gt_rows: list[GroundTruthRow],
    events: list[PipelineEvent],
    time_window: float,
    thresholds: list[float],
) -> list[dict]:
    """Re-score at several --ocr-min-conf floors, offline, from one run's events.

    Why this exists. --ocr-min-conf was set to 0.95 on three lucky reads from
    one clip, then lowered to 0.50 on three archived evidence crops. The second
    change halved misses (10 -> 5) but multiplied wrong reads (1 -> 8) and took
    overall accuracy DOWN, 66.67% -> 60.61%. Both numbers were picked from a
    handful of examples because measuring the alternative meant a full
    multi-hour pipeline re-run per candidate value.

    It does not. Raising the floor can only ever REMOVE already-accepted
    readings, so every threshold at or above the one a run used can be
    simulated by filtering that run's own events and re-running the match. One
    pipeline run, the whole curve.

    The floor is a recall/precision dial: raising it turns wrong reads back
    into misses. Which is better is a policy question -- for a banned-vehicle
    alert a miss is far worse than a flagged uncertain read -- so this reports
    the trade-off rather than declaring a winner.

    Only valid at or above the run's own floor: lower thresholds would need
    readings the pipeline never recorded, so those rows are marked accordingly.
    """
    results: list[dict] = []
    for threshold in sorted(thresholds):
        kept = [ev for ev in events if ev.ocr_confidence >= threshold]
        # Re-index so the matcher's assignment dict keys stay consistent.
        reindexed = [
            PipelineEvent(
                index=i, filename=ev.filename, timestamp_sec=ev.timestamp_sec,
                vehicle_type=ev.vehicle_type, plate=ev.plate, track_id=ev.track_id,
                ocr_confidence=ev.ocr_confidence, raw=ev.raw,
            )
            for i, ev in enumerate(kept)
        ]
        for gt in gt_rows:
            gt.ambiguous = False  # recomputed per threshold by the matcher
        assignments = match_events_to_ground_truth(gt_rows, reindexed, time_window)
        rows = score(gt_rows, reindexed, assignments)
        scorable = [r for r in rows if r.scorable]
        by_type: dict[str, dict[str, int]] = {}
        for r in scorable:
            b = by_type.setdefault(r.gt.vehicle_type, {"Correct": 0, "Incorrect": 0, "Missed": 0})
            b[r.result] += 1
        matched = set(assignments.values())
        results.append({
            "threshold": threshold,
            "events_kept": len(kept),
            "correct": sum(1 for r in scorable if r.result == "Correct"),
            "incorrect": sum(1 for r in scorable if r.result == "Incorrect"),
            "missed": sum(1 for r in scorable if r.result == "Missed"),
            "total": len(scorable),
            "false_positives": sum(1 for ev in reindexed if ev.index not in matched and ev.plate),
            "by_type": by_type,
        })
    return results


def print_sweep(results: list[dict], run_floor: float | None) -> None:
    print("=" * 78)
    print("OCR CONFIDENCE SWEEP")
    print("=" * 78)
    print(
        "Simulated by filtering THIS run's events, so it is only valid at or above the\n"
        "floor the run actually used (raising a floor can only remove readings; lowering\n"
        "one needs readings that were never recorded). Re-run the pipeline at the lowest\n"
        "floor you want to consider, then sweep upward from there.\n"
    )
    header = f"{'floor':>6} {'events':>7} {'correct':>8} {'wrong':>6} {'missed':>7} {'acc%':>7} {'FP':>4}"
    per_type = sorted({t for r in results for t in r["by_type"]})
    for t in per_type:
        header += f" {t[:8] + ' acc%':>13}"
    print(header)
    print("-" * len(header))
    for r in results:
        flag = " " if run_floor is None or r["threshold"] >= run_floor else "!"
        line = (
            f"{r['threshold']:>6.2f}{flag}{r['events_kept']:>6} {r['correct']:>8} {r['incorrect']:>6} "
            f"{r['missed']:>7} {_pct(r['correct'], r['total']):>6.2f}% {r['false_positives']:>4}"
        )
        for t in per_type:
            b = r["by_type"].get(t)
            tot = sum(b.values()) if b else 0
            line += f" {(_pct(b['Correct'], tot) if b else 0.0):>12.2f}%"
        print(line)
    if run_floor is not None:
        print(f"\n! = below this run's own --ocr-min-conf ({run_floor}); not simulable, ignore.")
    best = max(results, key=lambda r: r["correct"])
    print(
        f"\nMost correct reads at floor {best['threshold']:.2f}: {best['correct']}/{best['total']} "
        f"({_pct(best['correct'], best['total']):.2f}%)."
    )
    print(
        "Pick on policy, not just this column: raising the floor converts wrong reads into\n"
        "misses. For banned-vehicle alerting a miss is worse than a flagged uncertain read."
    )
    print()


def find_duplicate_events(
    events: list[PipelineEvent], window_sec: float = 5.0, max_near_edit_distance: int = 1
) -> tuple[list[tuple[PipelineEvent, PipelineEvent]], list[tuple[PipelineEvent, PipelineEvent]]]:
    """FIX #6, revised (external review — the original version could
    falsely flag a legitimate plate): returns (exact_duplicates,
    near_duplicates), same-file event pairs within `window_sec` of each
    other, split by how sure we can actually be:

    - exact_duplicates: identical plate text. Still not *proof* of a
      tracking bug — a vehicle can legitimately reappear within the window
      (e.g. entering, immediately realizing it's the wrong gate, and
      driving straight back out) — but identical text at least means it's
      the same plate, not a coincidence of two different vehicles.
    - near_duplicates: plate text differs by up to `max_near_edit_distance`
      characters. This is weaker evidence twice over: it could be the same
      pass read slightly differently by OCR on two track fragments (the
      original intent), OR it could be two genuinely different vehicles
      with coincidentally similar plates (e.g. "AAA111" vs "AAA112" from
      the same series) that aren't related at all. Report, don't assert.

    Defaults changed from the original 15s/edit-distance-1-as-one-bucket:
    the window shrunk to 5s (closer to the tracker's own reconciliation
    window — anything the tracker's IoU-merge should have already caught
    resolves well within a couple seconds; a duplicate surviving past that
    is more likely either a genuine second pass or a merge-logic gap worth
    knowing about on its own, not something a wide 15s net should paper
    over by lumping in unrelated re-passes). Both the window and the
    near-duplicate threshold are CLI-configurable — there's no universally
    correct value, only a default that errs toward under- rather than
    over-claiming.
    """
    exact: list[tuple[PipelineEvent, PipelineEvent]] = []
    near: list[tuple[PipelineEvent, PipelineEvent]] = []
    by_file: dict[str, list[PipelineEvent]] = {}
    for ev in events:
        if ev.plate:
            by_file.setdefault(ev.filename, []).append(ev)
    for evs in by_file.values():
        evs.sort(key=lambda e: e.timestamp_sec)
        for i, a in enumerate(evs):
            for b in evs[i + 1 :]:
                if b.timestamp_sec - a.timestamp_sec > window_sec:
                    break
                dist = edit_distance(a.plate, b.plate)
                if dist == 0:
                    exact.append((a, b))
                elif dist <= max_near_edit_distance:
                    near.append((a, b))
    return exact, near


# ============================================================================
# Scoring
# ============================================================================


@dataclass
class ScoredRow:
    gt: GroundTruthRow
    event: PipelineEvent | None
    result: str  # "Correct" | "Incorrect" | "Missed" | "Detected (GT unreadable)"
    cer: float | None
    scorable: bool  # False for blank-ground-truth rows — excluded from accuracy%


def score(gt_rows: list[GroundTruthRow], events: list[PipelineEvent], assignments: dict[int, int]) -> list[ScoredRow]:
    scored: list[ScoredRow] = []
    for gt in gt_rows:
        event = events[assignments[gt.index]] if gt.index in assignments else None

        if not gt.plate:
            # FIX #5's mirror case: nothing to judge OCR correctness against.
            result = "Detected (GT unreadable)" if event else "Missed"
            scored.append(ScoredRow(gt=gt, event=event, result=result, cer=None, scorable=False))
            continue

        if event is None:
            scored.append(ScoredRow(gt=gt, event=None, result="Missed", cer=None, scorable=True))
            continue

        if event.plate == gt.plate:
            scored.append(ScoredRow(gt=gt, event=event, result="Correct", cer=0.0, scorable=True))
        else:
            cer = character_error_rate(gt.plate, event.plate)
            scored.append(ScoredRow(gt=gt, event=event, result="Incorrect", cer=cer, scorable=True))
    return scored


# ============================================================================
# Reporting
# ============================================================================


def _pct(n: int, d: int) -> float:
    return (n / d * 100) if d else 0.0


def _split_false_positives(
    events: list[PipelineEvent],
    assignments: dict[int, int],
    discrepancies: list[tuple[GroundTruthRow, PipelineEvent]],
) -> tuple[list[PipelineEvent], list[PipelineEvent]]:
    """Separate genuine false positives from correct reads that just landed
    outside the matching window (FIX #11 — see find_timestamp_discrepancies).

    A false positive should mean "the pipeline logged a plate that does not
    correspond to any real vehicle". An event reading `ABA196` in a clip
    where `ABA196` is a labeled vehicle is not that, whatever its
    timestamp, and reporting it as one overstates the error rate while
    hiding a ground-truth/labeling problem.
    """
    matched_event_indices = set(assignments.values())
    out_of_window_indices = {ev.index for _, ev in discrepancies}
    genuine = [
        ev for ev in events
        if ev.index not in matched_event_indices and ev.index not in out_of_window_indices and ev.plate
    ]
    out_of_window = [ev for _, ev in discrepancies]
    return genuine, out_of_window


def summarize(rows: list[ScoredRow], key) -> dict[str, dict[str, int]]:
    buckets: dict[str, dict[str, int]] = {}
    for row in rows:
        if not row.scorable:
            continue
        bucket = buckets.setdefault(key(row), {"Correct": 0, "Incorrect": 0, "Missed": 0})
        bucket[row.result] = bucket.get(row.result, 0) + 1
    return buckets


def print_console_report(
    rows: list[ScoredRow],
    events: list[PipelineEvent],
    assignments: dict[int, int],
    duplicates: tuple[list[tuple[PipelineEvent, PipelineEvent]], list[tuple[PipelineEvent, PipelineEvent]]],
    time_window: float,
    duplicate_window: float,
) -> None:
    exact_dupes, near_dupes = duplicates
    scorable = [r for r in rows if r.scorable]
    correct = sum(1 for r in scorable if r.result == "Correct")
    incorrect = sum(1 for r in scorable if r.result == "Incorrect")
    missed = sum(1 for r in scorable if r.result == "Missed")
    total = len(scorable)
    unreadable = sum(1 for r in rows if not r.scorable)
    ambiguous = sum(1 for r in rows if r.gt.ambiguous)

    discrepancies = find_timestamp_discrepancies([r.gt for r in rows], events, assignments)
    false_positives, out_of_window = _split_false_positives(events, assignments, discrepancies)

    print("=" * 60)
    print("OVERALL (excludes ground-truth rows with no human-readable plate)")
    print("=" * 60)
    print(f"Scorable ground-truth vehicles: {total}")
    print(f"  Correct:   {correct} ({_pct(correct, total):.2f}%)")
    print(f"  Incorrect: {incorrect} ({_pct(incorrect, total):.2f}%)")
    print(f"  Missed:    {missed} ({_pct(missed, total):.2f}%)")
    print(f"Ground-truth rows with illegible plate (scored for detection only): {unreadable}")
    print(f"Ground-truth rows flagged ambiguous (contested match, either direction): {ambiguous}")
    print(f"Pipeline false-positive events (plate matches no ground-truth row): {len(false_positives)}")
    if out_of_window:
        print(
            f"Correct plate text read OUTSIDE the {time_window}s window: {len(out_of_window)} "
            f"(counted above as 'Missed', but NOT as false positives — see 'DETECTED OUTSIDE "
            f"THE MATCHING WINDOW' below)"
        )
    print(
        f"Candidate duplicate event pairs within {duplicate_window}s: {len(exact_dupes)} exact-plate, "
        f"{len(near_dupes)} near-plate (NOT proof of a bug either way — see 'CANDIDATE DUPLICATE EVENTS' below)"
    )
    print(f"Time window used for matching: {time_window}s")
    print()

    print("=" * 60)
    print("BY VEHICLE TYPE")
    print("=" * 60)
    for vtype, b in summarize(rows, lambda r: r.gt.vehicle_type).items():
        t = b["Correct"] + b["Incorrect"] + b["Missed"]
        print(f"{vtype}: total={t} correct={b['Correct']} incorrect={b['Incorrect']} missed={b['Missed']} accuracy={_pct(b['Correct'], t):.2f}%")
    print()

    print("=" * 60)
    print("BY CONDITION")
    print("=" * 60)
    for cond, b in summarize(rows, lambda r: r.gt.condition).items():
        t = b["Correct"] + b["Incorrect"] + b["Missed"]
        print(f"{cond}: total={t} correct={b['Correct']} incorrect={b['Incorrect']} missed={b['Missed']} accuracy={_pct(b['Correct'], t):.2f}%")
    print()

    incorrect_rows = [r for r in scorable if r.result == "Incorrect"]
    if incorrect_rows:
        print("=" * 60)
        print("INCORRECT READS (with Character Error Rate)")
        print("=" * 60)
        for r in sorted(incorrect_rows, key=lambda r: r.cer or 0):
            print(
                f"  {r.gt.filename} t={r.gt.timestamp_sec}s  GT={r.gt.plate!r}  got={r.event.plate!r}  "
                f"CER={r.cer:.2f}  ocr_conf={r.event.ocr_confidence:.3f}"
            )
        print()

    if discrepancies:
        print("=" * 60)
        print("DETECTED OUTSIDE THE MATCHING WINDOW")
        print("=" * 60)
        print(
            "The pipeline read these plates EXACTLY right, but too far from the ground-truth\n"
            "timestamp to match. They are still counted as 'Missed' above (matching on plate\n"
            "text would mean grading OCR with its own output), but they are NOT false\n"
            "positives. Either the label's timestamp or --time-window is wrong — check the\n"
            "gap size below, then fix the ground truth or widen the window and re-score."
        )
        for gt, ev in sorted(discrepancies, key=lambda p: -abs(p[1].timestamp_sec - p[0].timestamp_sec)):
            gap = ev.timestamp_sec - gt.timestamp_sec
            print(
                f"  {gt.filename} plate={gt.plate!r} labeled t={gt.timestamp_sec}s, "
                f"detected t={ev.timestamp_sec}s (gap {gap:+.1f}s, track {ev.track_id})"
            )
        print()

    if false_positives:
        print("=" * 60)
        print("FALSE-POSITIVE EVENTS (plate matches no ground-truth row in that file)")
        print("=" * 60)
        for ev in false_positives:
            print(
                f"  {ev.filename} t={ev.timestamp_sec}s track={ev.track_id} class={ev.vehicle_type} "
                f"plate={ev.plate!r} ocr_conf={ev.ocr_confidence:.3f}"
            )
        print()

    if exact_dupes or near_dupes:
        print("=" * 60)
        print("CANDIDATE DUPLICATE EVENTS")
        print("=" * 60)
        print(
            "Same plate (or near-identical text) logged twice, close in time. This is a SIGNAL, "
            "not proof: it could be a tracking/dedup bug (same physical pass logged twice), or a "
            "vehicle legitimately reappearing quickly (e.g. entering, realizing it's the wrong "
            "gate, and immediately leaving). Check evidence images/track history before treating "
            "any of these as confirmed bugs."
        )
        if exact_dupes:
            print(f"\nExact-plate matches ({len(exact_dupes)}):")
            for a, b in exact_dupes:
                print(f"  {a.filename}: track {a.track_id} ({a.plate!r}@{a.timestamp_sec}s) vs track {b.track_id} ({b.plate!r}@{b.timestamp_sec}s)")
        if near_dupes:
            print(f"\nNear-plate matches ({len(near_dupes)}) — could also be two different, coincidentally similar plates:")
            for a, b in near_dupes:
                print(f"  {a.filename}: track {a.track_id} ({a.plate!r}@{a.timestamp_sec}s) vs track {b.track_id} ({b.plate!r}@{b.timestamp_sec}s)")
        print()


def write_results_csv(
    rows: list[ScoredRow],
    path: Path,
    discrepancies: list[tuple[GroundTruthRow, PipelineEvent]] | None = None,
) -> None:
    # gt_index -> seconds between the label and the correct-but-unmatched
    # read, so a "Missed" row that was actually read right is visible in the
    # per-row CSV, not only in the report prose (FIX #11).
    gaps = {gt.index: ev.timestamp_sec - gt.timestamp_sec for gt, ev in (discrepancies or [])}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["filename", "timestamp_sec", "vehicle_type", "condition", "ground_truth_plate",
             "detected_plate", "result", "cer", "ambiguous_ground_truth",
             "detected_outside_window_gap_sec"]
        )
        for r in rows:
            writer.writerow(
                [
                    r.gt.filename, r.gt.timestamp_sec, r.gt.vehicle_type, r.gt.condition,
                    r.gt.plate, r.event.plate if r.event else "", r.result,
                    f"{r.cer:.3f}" if r.cer is not None else "", r.gt.ambiguous,
                    f"{gaps[r.gt.index]:+.2f}" if r.gt.index in gaps else "",
                ]
            )


def write_markdown_report(
    rows: list[ScoredRow],
    events: list[PipelineEvent],
    assignments: dict[int, int],
    duplicates: tuple[list[tuple[PipelineEvent, PipelineEvent]], list[tuple[PipelineEvent, PipelineEvent]]],
    time_window: float,
    duplicate_window: float,
    path: Path,
) -> None:
    exact_dupes, near_dupes = duplicates
    scorable = [r for r in rows if r.scorable]
    correct = sum(1 for r in scorable if r.result == "Correct")
    incorrect = sum(1 for r in scorable if r.result == "Incorrect")
    missed = sum(1 for r in scorable if r.result == "Missed")
    total = len(scorable)
    discrepancies = find_timestamp_discrepancies([r.gt for r in rows], events, assignments)
    false_positives, out_of_window = _split_false_positives(events, assignments, discrepancies)

    lines = [
        "# ANPR Accuracy Evaluation Report",
        "",
        "_Generated by `scripts/score_accuracy.py` — re-run the script to regenerate this file "
        "from current `output/*_events.json` + `sample_data/ground_truth.csv` rather than editing by hand._",
        "",
        "## Overall",
        "",
        "| Metric | Count | Percentage |",
        "|---|---:|---:|",
        f"| Scorable ground-truth vehicles | {total} | 100% |",
        f"| Correct | {correct} | {_pct(correct, total):.2f}% |",
        f"| Incorrect | {incorrect} | {_pct(incorrect, total):.2f}% |",
        f"| Missed | {missed} | {_pct(missed, total):.2f}% |",
        "",
        f"- Ground-truth rows with an illegible plate (excluded from accuracy %, scored for detection only): "
        f"{sum(1 for r in rows if not r.scorable)}",
        f"- Ground-truth rows flagged **ambiguous** (a contested match within {time_window}s — either "
        f"2+ events competed for this row, or its matched event was also wanted by another row): "
        f"{sum(1 for r in rows if r.gt.ambiguous)}",
        f"- False-positive pipeline events (plate matches no ground-truth row in that file): {len(false_positives)}",
        f"- Correct plate text read **outside** the {time_window}s matching window: {len(out_of_window)} "
        f"— counted as \"Missed\" above, but not as false positives; see below",
        f"- Candidate duplicate event pairs within {duplicate_window}s: {len(exact_dupes)} exact-plate, "
        f"{len(near_dupes)} near-plate — a signal, not proof of a bug; see below",
        f"- Matching time window: {time_window}s",
        "",
        "## By Vehicle Type",
        "",
        "| Type | Total | Correct | Incorrect | Missed | Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for vtype, b in summarize(rows, lambda r: r.gt.vehicle_type).items():
        t = b["Correct"] + b["Incorrect"] + b["Missed"]
        lines.append(f"| {vtype} | {t} | {b['Correct']} | {b['Incorrect']} | {b['Missed']} | {_pct(b['Correct'], t):.2f}% |")

    lines += ["", "## By Condition", "", "| Condition | Total | Correct | Incorrect | Missed | Accuracy |", "|---|---:|---:|---:|---:|---:|"]
    for cond, b in summarize(rows, lambda r: r.gt.condition).items():
        t = b["Correct"] + b["Incorrect"] + b["Missed"]
        lines.append(f"| {cond} | {t} | {b['Correct']} | {b['Incorrect']} | {b['Missed']} | {_pct(b['Correct'], t):.2f}% |")

    incorrect_rows = [r for r in scorable if r.result == "Incorrect"]
    if incorrect_rows:
        lines += [
            "", "## Incorrect Reads (with Character Error Rate)", "",
            "`OCR conf` is the confidence the pipeline accepted the wrong reading at. It decides "
            "what to do about the row: a wrong read at low confidence means `--ocr-min-conf` is too "
            "loose, a wrong read at high confidence means the threshold is irrelevant and the model "
            "or the crop is at fault. Use `--sweep-ocr-conf` to pick the floor from this data.",
            "",
            "| File | t | Ground Truth | Detected | CER | OCR conf |", "|---|---:|---|---|---:|---:|",
        ]
        for r in sorted(incorrect_rows, key=lambda r: r.cer or 0):
            lines.append(
                f"| {r.gt.filename} | {r.gt.timestamp_sec}s | `{r.gt.plate}` | `{r.event.plate}` | "
                f"{r.cer:.2f} | {r.event.ocr_confidence:.3f} |"
            )

    missed_rows = [r for r in scorable if r.result == "Missed"]
    if missed_rows:
        lines += ["", "## Missed Vehicles", "", "| File | t | Type | Ground Truth |", "|---|---:|---|---|"]
        for r in missed_rows:
            lines.append(f"| {r.gt.filename} | {r.gt.timestamp_sec}s | {r.gt.vehicle_type} | `{r.gt.plate}` |")

    if discrepancies:
        lines += [
            "",
            "## Detected Outside the Matching Window",
            "",
            "The pipeline read these plates **exactly right**, but too far from the ground-truth "
            "timestamp to match. They are still counted as \"Missed\" above — matching on plate text "
            "would mean grading OCR with its own output as the answer key — but they are **not** "
            "false positives. Either the label's timestamp or `--time-window` is wrong; check the "
            "gap, then fix the ground truth or widen the window and re-score.",
            "",
            "| File | Plate | Labeled at | Detected at | Gap | Track |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for gt, ev in sorted(discrepancies, key=lambda p: -abs(p[1].timestamp_sec - p[0].timestamp_sec)):
            gap = ev.timestamp_sec - gt.timestamp_sec
            lines.append(
                f"| {gt.filename} | `{gt.plate}` | {gt.timestamp_sec}s | {ev.timestamp_sec}s | "
                f"{gap:+.1f}s | {ev.track_id} |"
            )

    if false_positives:
        lines += ["", "## False-Positive Events (plate matches no ground-truth row in that file)", "", "| File | t | Track | Class | Plate | OCR conf |", "|---|---:|---:|---|---|---:|"]
        for ev in false_positives:
            lines.append(
                f"| {ev.filename} | {ev.timestamp_sec}s | {ev.track_id} | {ev.vehicle_type} | "
                f"`{ev.plate}` | {ev.ocr_confidence:.3f} |"
            )

    if exact_dupes or near_dupes:
        lines += [
            "",
            "## Candidate Duplicate Events",
            "",
            "A signal, not proof of a bug: could be one physical pass logged twice (tracking/dedup "
            "issue), or a vehicle legitimately reappearing quickly (e.g. entering, realizing it's "
            "the wrong gate, and immediately leaving). Check evidence images/track history before "
            "treating any of these as confirmed bugs.",
        ]
        if exact_dupes:
            lines += ["", "### Exact plate match", "", "| File | Track A | Plate A | t A | Track B | Plate B | t B |", "|---|---:|---|---:|---:|---|---:|"]
            for a, b in exact_dupes:
                lines.append(f"| {a.filename} | {a.track_id} | `{a.plate}` | {a.timestamp_sec}s | {b.track_id} | `{b.plate}` | {b.timestamp_sec}s |")
        if near_dupes:
            lines += [
                "",
                "### Near plate match (could be two different, coincidentally similar plates)",
                "",
                "| File | Track A | Plate A | t A | Track B | Plate B | t B |",
                "|---|---:|---|---:|---:|---|---:|",
            ]
            for a, b in near_dupes:
                lines.append(f"| {a.filename} | {a.track_id} | `{a.plate}` | {a.timestamp_sec}s | {b.track_id} | `{b.plate}` | {b.timestamp_sec}s |")

    ambiguous_rows = [r.gt for r in rows if r.gt.ambiguous]
    if ambiguous_rows:
        lines += [
            "",
            "## Ambiguous Ground-Truth Rows",
            "",
            f"2 or more pipeline events fell within the {time_window}s matching window of these "
            "ground-truth vehicles, so the matcher had a real choice to make — not just two "
            "vehicles that happened to pass in the same clip. Treat results on these specific "
            "rows with extra caution; verify against evidence images.",
            "",
            "| File | t | Type | Ground Truth |",
            "|---|---:|---|---|",
        ]
        for gt in sorted(set((r.filename, r.timestamp_sec, r.vehicle_type, r.plate) for r in ambiguous_rows)):
            lines.append(f"| {gt[0]} | {gt[1]}s | {gt[2]} | `{gt[3]}` |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================================
# CLI
# ============================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--time-window", type=float, default=DEFAULT_TIME_WINDOW)
    parser.add_argument(
        "--duplicate-window", type=float, default=DEFAULT_DUPLICATE_WINDOW,
        help="Seconds within which two same-plate events in one file are flagged as a candidate "
        "duplicate. Default is intentionally short (see find_duplicate_events docstring) — widen "
        "it if you specifically want to catch slower re-passes, at the cost of more false alarms "
        "on legitimate quick reappearances.",
    )
    parser.add_argument(
        "--duplicate-max-edit-distance", type=int, default=1,
        help="Max character difference still counted as a 'near' duplicate (0 = exact-plate only).",
    )
    parser.add_argument(
        "--sweep-ocr-conf", nargs="*", type=float, default=None, metavar="FLOOR",
        help="Re-score at several --ocr-min-conf floors offline, from this run's events, and print "
        "the accuracy/miss trade-off curve. No pipeline re-run needed. Pass explicit values, or no "
        "values for a default sweep. Only valid at or above the floor the run itself used.",
    )
    parser.add_argument(
        "--run-ocr-min-conf", type=float, default=None,
        help="The --ocr-min-conf the run used, so the sweep can mark thresholds it cannot simulate.",
    )
    parser.add_argument("--results-csv", type=Path, default=None, help="Default: <output-dir>/accuracy_results.csv")
    parser.add_argument("--report", type=Path, default=None, help="Optional path to write a markdown report, e.g. docs/ACCURACY_REPORT.md")
    args = parser.parse_args(argv)

    # This report is full of em-dashes and its whole job is to be read. On a
    # Windows console (cp1252 by default) every one of them renders as a
    # replacement character, which makes the diagnostic output this script
    # exists to produce noticeably harder to read.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass  # not a reconfigurable text stream (piped/captured) — output is fine as-is

    if not args.ground_truth.exists():
        print(f"Ground truth file not found: {args.ground_truth}", file=sys.stderr)
        return 1

    gt_rows = load_ground_truth(args.ground_truth)
    events = load_pipeline_events(args.output_dir)
    if not events:
        print(f"No *_events.json files found in {args.output_dir} — run run_pipeline.py first.", file=sys.stderr)
        return 1

    warn_on_filename_mismatch(gt_rows, events)

    assignments = match_events_to_ground_truth(gt_rows, events, args.time_window)
    duplicates = find_duplicate_events(events, window_sec=args.duplicate_window, max_near_edit_distance=args.duplicate_max_edit_distance)
    rows = score(gt_rows, events, assignments)

    print(f"Ground-truth vehicles: {len(gt_rows)}")
    print(f"Pipeline events:       {len(events)}")
    print()
    print_console_report(rows, events, assignments, duplicates, args.time_window, args.duplicate_window)

    if args.sweep_ocr_conf is not None:
        thresholds = args.sweep_ocr_conf or [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99]
        print_sweep(
            sweep_ocr_confidence(gt_rows, events, args.time_window, thresholds),
            args.run_ocr_min_conf,
        )
        # Matching mutated the ambiguity flags during the sweep; recompute for the
        # main report so its numbers match a plain, unswept run exactly.
        for gt in gt_rows:
            gt.ambiguous = False
        assignments = match_events_to_ground_truth(gt_rows, events, args.time_window)
        rows = score(gt_rows, events, assignments)

    results_csv = args.results_csv or (args.output_dir / "accuracy_results.csv")
    write_results_csv(rows, results_csv, find_timestamp_discrepancies(gt_rows, events, assignments))
    print(f"Detailed results written to: {results_csv}")

    if args.report:
        write_markdown_report(rows, events, assignments, duplicates, args.time_window, args.duplicate_window, args.report)
        print(f"Markdown report written to: {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
