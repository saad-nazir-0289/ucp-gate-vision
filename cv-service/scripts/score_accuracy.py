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
    events: list[PipelineEvent] = []
    for events_path in sorted(output_dir.glob("*_events.json")):
        filename = events_path.name[: -len("_events.json")] + ".mp4"
        with open(events_path, encoding="utf-8") as f:
            data = json.load(f)
        for raw_event in data:
            events.append(
                PipelineEvent(
                    index=len(events),
                    filename=filename,
                    timestamp_sec=float(raw_event.get("timestamp_sec", 0)),
                    vehicle_type=normalize_vehicle_type(raw_event.get("vehicle_class")),
                    plate=normalize_plate(raw_event.get("plate_text")),
                    track_id=raw_event.get("track_id", -1),
                    raw=raw_event,
                )
            )
    return events


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

    matched_event_indices = set(assignments.values())
    false_positives = [ev for ev in events if ev.index not in matched_event_indices and ev.plate]

    print("=" * 60)
    print("OVERALL (excludes ground-truth rows with no human-readable plate)")
    print("=" * 60)
    print(f"Scorable ground-truth vehicles: {total}")
    print(f"  Correct:   {correct} ({_pct(correct, total):.2f}%)")
    print(f"  Incorrect: {incorrect} ({_pct(incorrect, total):.2f}%)")
    print(f"  Missed:    {missed} ({_pct(missed, total):.2f}%)")
    print(f"Ground-truth rows with illegible plate (scored for detection only): {unreadable}")
    print(f"Ground-truth rows flagged ambiguous (contested match, either direction): {ambiguous}")
    print(f"Pipeline false-positive events (matched no ground truth): {len(false_positives)}")
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
            print(f"  {r.gt.filename} t={r.gt.timestamp_sec}s  GT={r.gt.plate!r}  got={r.event.plate!r}  CER={r.cer:.2f}")
        print()

    if false_positives:
        print("=" * 60)
        print("FALSE-POSITIVE EVENTS (no matching ground truth)")
        print("=" * 60)
        for ev in false_positives:
            print(f"  {ev.filename} t={ev.timestamp_sec}s track={ev.track_id} class={ev.vehicle_type} plate={ev.plate!r}")
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


def write_results_csv(rows: list[ScoredRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["filename", "timestamp_sec", "vehicle_type", "condition", "ground_truth_plate",
             "detected_plate", "result", "cer", "ambiguous_ground_truth"]
        )
        for r in rows:
            writer.writerow(
                [
                    r.gt.filename, r.gt.timestamp_sec, r.gt.vehicle_type, r.gt.condition,
                    r.gt.plate, r.event.plate if r.event else "", r.result,
                    f"{r.cer:.3f}" if r.cer is not None else "", r.gt.ambiguous,
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
    matched_event_indices = set(assignments.values())
    false_positives = [ev for ev in events if ev.index not in matched_event_indices and ev.plate]

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
        f"- False-positive pipeline events (matched no ground truth): {len(false_positives)}",
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
        lines += ["", "## Incorrect Reads (with Character Error Rate)", "", "| File | t | Ground Truth | Detected | CER |", "|---|---:|---|---|---:|"]
        for r in sorted(incorrect_rows, key=lambda r: r.cer or 0):
            lines.append(f"| {r.gt.filename} | {r.gt.timestamp_sec}s | `{r.gt.plate}` | `{r.event.plate}` | {r.cer:.2f} |")

    missed_rows = [r for r in scorable if r.result == "Missed"]
    if missed_rows:
        lines += ["", "## Missed Vehicles", "", "| File | t | Type | Ground Truth |", "|---|---:|---|---|"]
        for r in missed_rows:
            lines.append(f"| {r.gt.filename} | {r.gt.timestamp_sec}s | {r.gt.vehicle_type} | `{r.gt.plate}` |")

    if false_positives:
        lines += ["", "## False-Positive Events (no matching ground truth)", "", "| File | t | Track | Class | Plate |", "|---|---:|---:|---|---|"]
        for ev in false_positives:
            lines.append(f"| {ev.filename} | {ev.timestamp_sec}s | {ev.track_id} | {ev.vehicle_type} | `{ev.plate}` |")

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
    parser.add_argument("--results-csv", type=Path, default=None, help="Default: <output-dir>/accuracy_results.csv")
    parser.add_argument("--report", type=Path, default=None, help="Optional path to write a markdown report, e.g. docs/ACCURACY_REPORT.md")
    args = parser.parse_args(argv)

    if not args.ground_truth.exists():
        print(f"Ground truth file not found: {args.ground_truth}", file=sys.stderr)
        return 1

    gt_rows = load_ground_truth(args.ground_truth)
    events = load_pipeline_events(args.output_dir)
    if not events:
        print(f"No *_events.json files found in {args.output_dir} — run run_pipeline.py first.", file=sys.stderr)
        return 1

    assignments = match_events_to_ground_truth(gt_rows, events, args.time_window)
    duplicates = find_duplicate_events(events, window_sec=args.duplicate_window, max_near_edit_distance=args.duplicate_max_edit_distance)
    rows = score(gt_rows, events, assignments)

    print(f"Ground-truth vehicles: {len(gt_rows)}")
    print(f"Pipeline events:       {len(events)}")
    print()
    print_console_report(rows, events, assignments, duplicates, args.time_window, args.duplicate_window)

    results_csv = args.results_csv or (args.output_dir / "accuracy_results.csv")
    write_results_csv(rows, results_csv)
    print(f"Detailed results written to: {results_csv}")

    if args.report:
        write_markdown_report(rows, events, assignments, duplicates, args.time_window, args.duplicate_window, args.report)
        print(f"Markdown report written to: {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
