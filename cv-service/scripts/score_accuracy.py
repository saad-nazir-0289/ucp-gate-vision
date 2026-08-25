import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
GROUND_TRUTH = BASE_DIR / "sample_data" / "ground_truth.csv"
OUTPUT_DIR = BASE_DIR / "output"

TIME_WINDOW = 5.0


def normalize_plate(value):
    if pd.isna(value):
        return ""
    return "".join(str(value).upper().split())


def normalize_vehicle_type(value):
    value = str(value).strip().lower()

    if value in {"bike", "motorcycle", "motorbike"}:
        return "bike"

    if value == "car":
        return "car"

    return value


def normalize_condition(value):
    value = str(value).strip().lower()

    # "day Multiple vehicles" -> "day"
    if value.startswith("day"):
        return "day"

    if value.startswith("dusk"):
        return "dusk"

    if value.startswith("night"):
        return "night"

    if value.startswith("rain"):
        return "rain"

    if value.startswith("clear"):
        return "clear"

    return value


def load_events():
    events = []

    for path in OUTPUT_DIR.glob("*_events.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        filename = path.name.replace("_events.json", ".mp4")

        for event in data:
            event["_filename"] = filename
            event["_vehicle_type"] = normalize_vehicle_type(
                event.get("vehicle_class", "")
            )
            event["_plate"] = normalize_plate(
                event.get("plate_text", "")
            )
            event["_timestamp"] = float(
                event.get("timestamp_sec", 0)
            )

        events.extend(data)

    return events


def main():
    print("Loading ground truth...")

    gt = pd.read_csv(GROUND_TRUTH)

    gt["_vehicle_type"] = gt["vehicle_type"].apply(
        normalize_vehicle_type
    )

    gt["_condition"] = gt["condition"].apply(
        normalize_condition
    )

    gt["_plate"] = gt["plate"].apply(
        normalize_plate
    )

    events = load_events()

    print(f"Ground-truth vehicles: {len(gt)}")
    print(f"Pipeline events:        {len(events)}")
    print()

    results = []

    # Track which pipeline events have already been assigned.
    used_events = set()

    # Store candidate events for duplicate analysis.
    candidate_map = {}

    # ---------------------------------------------------------
    # STEP 1:
    # Find possible events for every ground-truth vehicle.
    # ---------------------------------------------------------

    for gt_index, row in gt.iterrows():

        candidates = []

        filename = str(row["filename"])
        timestamp = float(row["timestamp_sec"])
        vehicle_type = row["_vehicle_type"]

        for event_index, event in enumerate(events):

            if event_index in used_events:
                continue

            if event["_filename"] != filename:
                continue

            if event["_vehicle_type"] != vehicle_type:
                continue

            distance = abs(event["_timestamp"] - timestamp)

            if distance <= TIME_WINDOW:
                candidates.append(
                    (
                        distance,
                        event_index,
                    )
                )

        candidates.sort(key=lambda x: x[0])

        candidate_map[gt_index] = candidates

    # ---------------------------------------------------------
    # STEP 2:
    # Assign events globally from closest match to farthest.
    #
    # This guarantees:
    # ONE pipeline event -> ONE ground-truth vehicle.
    # ---------------------------------------------------------

    all_pairs = []

    for gt_index, candidates in candidate_map.items():
        for distance, event_index in candidates:
            all_pairs.append(
                (
                    distance,
                    gt_index,
                    event_index,
                )
            )

    all_pairs.sort(key=lambda x: x[0])

    assignments = {}

    for distance, gt_index, event_index in all_pairs:

        if gt_index in assignments:
            continue

        if event_index in used_events:
            continue

        assignments[gt_index] = event_index
        used_events.add(event_index)

    # ---------------------------------------------------------
    # STEP 3:
    # Build results.
    # ---------------------------------------------------------

    duplicate_count = 0

    for gt_index, row in gt.iterrows():

        gt_plate = row["_plate"]

        candidates = candidate_map.get(gt_index, [])

        if len(candidates) > 1:
            duplicate_count += 1

        if gt_index not in assignments:

            result = "Missed"
            matched_plate = ""

        else:

            event_index = assignments[gt_index]
            event = events[event_index]

            matched_plate = event["_plate"]

            if matched_plate == gt_plate:
                result = "Correct"
            else:
                result = "Incorrect"

        results.append(
            {
                "filename": row["filename"],
                "timestamp_sec": row["timestamp_sec"],
                "vehicle_type": row["_vehicle_type"],
                "condition": row["_condition"],
                "ground_truth_plate": gt_plate,
                "detected_plate": matched_plate,
                "result": result,
            }
        )

    results_df = pd.DataFrame(results)

    # ---------------------------------------------------------
    # OVERALL
    # ---------------------------------------------------------

    print("========================================")
    print("OVERALL")
    print("========================================")

    summary = results_df["result"].value_counts()

    total = len(results_df)
    correct = summary.get("Correct", 0)
    incorrect = summary.get("Incorrect", 0)
    missed = summary.get("Missed", 0)

    accuracy = correct / total * 100 if total else 0

    print(f"Total:       {total}")
    print(f"Correct:     {correct}")
    print(f"Incorrect:   {incorrect}")
    print(f"Missed:      {missed}")
    print(f"Duplicates:  {duplicate_count}")
    print(f"Accuracy:    {accuracy:.2f}%")

    # ---------------------------------------------------------
    # VEHICLE TYPE
    # ---------------------------------------------------------

    print()
    print("========================================")
    print("BY VEHICLE TYPE")
    print("========================================")

    vehicle_summary = (
        results_df.groupby("vehicle_type")["result"]
        .value_counts()
        .unstack(fill_value=0)
    )

    for vehicle_type, row in vehicle_summary.iterrows():

        total_type = row.sum()
        correct_type = row.get("Correct", 0)

        accuracy_type = (
            correct_type / total_type * 100
            if total_type
            else 0
        )

        print(
            f"{vehicle_type}: "
            f"Total={total_type}, "
            f"Correct={correct_type}, "
            f"Incorrect={row.get('Incorrect', 0)}, "
            f"Missed={row.get('Missed', 0)}, "
            f"Accuracy={accuracy_type:.2f}%"
        )

    # ---------------------------------------------------------
    # CONDITION
    # ---------------------------------------------------------

    print()
    print("========================================")
    print("BY CONDITION")
    print("========================================")

    condition_summary = (
        results_df.groupby("condition")["result"]
        .value_counts()
        .unstack(fill_value=0)
    )

    for condition, row in condition_summary.iterrows():

        total_condition = row.sum()
        correct_condition = row.get("Correct", 0)

        accuracy_condition = (
            correct_condition / total_condition * 100
            if total_condition
            else 0
        )

        print(
            f"{condition}: "
            f"Total={total_condition}, "
            f"Correct={correct_condition}, "
            f"Incorrect={row.get('Incorrect', 0)}, "
            f"Missed={row.get('Missed', 0)}, "
            f"Accuracy={accuracy_condition:.2f}%"
        )

    # ---------------------------------------------------------
    # INDIVIDUAL RESULTS
    # ---------------------------------------------------------

    print()
    print("========================================")
    print("INDIVIDUAL RESULTS")
    print("========================================")

    print(results_df.to_string(index=False))

    # ---------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------

    results_path = OUTPUT_DIR / "accuracy_results.csv"

    results_df.to_csv(
        results_path,
        index=False
    )

    print()
    print(f"Detailed results saved to: {results_path}")


if __name__ == "__main__":
    main()