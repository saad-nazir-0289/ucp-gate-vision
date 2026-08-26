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
   transitively. This script uses only the standard library.

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
import sys
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_GROUND_TRUTH = BASE_DIR / "sample_data" / "ground_truth.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"
DEFAULT_TIME_WINDOW = 5.0  # seconds — same as PR #10's version, kept for comparability


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


def match_events_to_ground_truth(
    gt_rows: list[GroundTruthRow], events: list[PipelineEvent], time_window: float
) -> dict[int, int]:
    """Global closest-timestamp-first assignment — one event per GT row, one
    GT row per event. This part of PR #10's approach was sound; kept as-is.

    Also sets `.ambiguous` on any GT row that had more than one candidate
    event within the time window (FIX #3). Earlier draft of this flag
    compared raw GT-to-GT timestamp gaps against 2x the window, which fires
    on any two same-type vehicles passing within ~10s of each other — i.e.
    almost all normal traffic, not a real signal (caught by testing this
    script against synthetic fixtures before trusting it on real data).
    Flagging actual candidate contention instead — a GT row where 2+ events
    were genuinely competing to match it — is precise: it doesn't fire on
    routinely-spaced traffic, only where the matcher had a real choice to
    make (including, usefully, when that contention comes from a duplicate
    event rather than a second real vehicle).
    """
    candidates: list[tuple[float, int, int]] = []
    for gt in gt_rows:
        for ev in events:
            if ev.filename != gt.filename or ev.vehicle_type != gt.vehicle_type:
                continue
            distance = abs(ev.timestamp_sec - gt.timestamp_sec)
            if distance <= time_window:
                candidates.append((distance, gt.index, ev.index))
    candidates.sort(key=lambda c: c[0])

    candidate_count_by_gt: dict[int, int] = {}
    for _, gt_index, _ in candidates:
        candidate_count_by_gt[gt_index] = candidate_count_by_gt.get(gt_index, 0) + 1

    gt_by_index = {gt.index: gt for gt in gt_rows}
    for gt_index, count in candidate_count_by_gt.items():
        if count > 1:
            gt_by_index[gt_index].ambiguous = True

    assigned_gt: set[int] = set()
    assigned_ev: set[int] = set()
    assignments: dict[int, int] = {}
    for _, gt_index, ev_index in candidates:
        if gt_index in assigned_gt or ev_index in assigned_ev:
            continue
        assignments[gt_index] = ev_index
        assigned_gt.add(gt_index)
        assigned_ev.add(ev_index)
    return assignments


def find_duplicate_events(
    events: list[PipelineEvent], window_sec: float = 15.0, max_edit_distance: int = 1
) -> list[tuple[PipelineEvent, PipelineEvent]]:
    """FIX #6: same-file events reading the same (or near-identical) plate
    within `window_sec` of each other — almost certainly one physical pass
    logged twice, a tracking/dedup issue rather than an OCR accuracy one."""
    duplicates: list[tuple[PipelineEvent, PipelineEvent]] = []
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
                if edit_distance(a.plate, b.plate) <= max_edit_distance:
                    duplicates.append((a, b))
    return duplicates


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
    duplicates: list[tuple[PipelineEvent, PipelineEvent]],
    time_window: float,
) -> None:
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
    print(f"Ground-truth rows flagged ambiguous (2+ candidate events competed for it): {ambiguous}")
    print(f"Pipeline false-positive events (matched no ground truth): {len(false_positives)}")
    print(f"Likely duplicate event pairs (same plate, same file, within 15s): {len(duplicates)}")
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

    if duplicates:
        print("=" * 60)
        print("LIKELY DUPLICATE EVENTS")
        print("=" * 60)
        for a, b in duplicates:
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
    duplicates: list[tuple[PipelineEvent, PipelineEvent]],
    time_window: float,
    path: Path,
) -> None:
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
        f"- Ground-truth rows flagged **ambiguous** (2+ pipeline events within {time_window}s competed "
        f"for this one vehicle — matching can't be fully trusted): {sum(1 for r in rows if r.gt.ambiguous)}",
        f"- False-positive pipeline events (matched no ground truth): {len(false_positives)}",
        f"- Likely duplicate event pairs: {len(duplicates)}",
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

    if duplicates:
        lines += ["", "## Likely Duplicate Events", "", "| File | Track A | Plate A | t A | Track B | Plate B | t B |", "|---|---:|---|---:|---:|---|---:|"]
        for a, b in duplicates:
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
    duplicates = find_duplicate_events(events)
    rows = score(gt_rows, events, assignments)

    print(f"Ground-truth vehicles: {len(gt_rows)}")
    print(f"Pipeline events:       {len(events)}")
    print()
    print_console_report(rows, events, assignments, duplicates, args.time_window)

    results_csv = args.results_csv or (args.output_dir / "accuracy_results.csv")
    write_results_csv(rows, results_csv)
    print(f"Detailed results written to: {results_csv}")

    if args.report:
        write_markdown_report(rows, events, assignments, duplicates, args.time_window, args.report)
        print(f"Markdown report written to: {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
