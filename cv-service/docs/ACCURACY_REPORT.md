
# ANPR Accuracy Evaluation Report

## 1. Executive Summary

This report evaluates the Campus ANPR pipeline against the supplied ground-truth dataset.

The evaluation contains **33 ground-truth vehicle observations** across clear and multiple-vehicle daytime footage.

### Key Results

| Metric | Result |
|---|---:|
| Total vehicles | 33 |
| Correct | 17 |
| Incorrect | 2 |
| Missed | 14 |
| Overall Accuracy | **51.52%** |

The system currently performs significantly better on **cars (68.42%)** than on **motorcycles (28.57%)**.

The largest observed weaknesses are:

1. Motorcycle detection and plate recognition.
2. Multiple-vehicle scenes.
3. Missed vehicle events.
4. Occasional OCR character errors.

These results establish the baseline for subsequent accuracy improvements and the night/rain investigation.

---

## 2. Evaluation Dataset

The evaluation uses the following ground-truth dataset:

```text
cv-service/sample_data/ground_truth.csv
````

### Dataset Composition

| Category    | Vehicles |
| ----------- | -------: |
| Cars        |       19 |
| Motorcycles |       14 |
| **Total**   |   **33** |

### Conditions

| Condition               | Vehicles |
| ----------------------- | -------: |
| Clear                   |       26 |
| Day / Multiple Vehicles |        7 |
| **Total**               |   **33** |

The current dataset does not contain separate ground-truth categories for dusk, night, or rain. Those conditions therefore cannot be evaluated from this dataset yet.

---

## 3. Evaluation Method

The evaluation was performed by running the ANPR pipeline against every available evaluation clip.

Pipeline outputs include:

* Detection events
* Annotated videos
* Evidence frames
* Plate crops

The scoring process then:

1. Loads the ground-truth CSV.
2. Loads the generated `*_events.json` files.
3. Normalizes plate text by:

   * Converting to uppercase.
   * Removing spaces.
4. Normalizes vehicle types:

   * `motorcycle`, `motorbike`, and `bike` → `bike`
   * `car` → `car`
5. Matches pipeline events to ground-truth vehicles using:

   * Video filename
   * Vehicle type
   * Timestamp proximity
6. Ensures a pipeline event is assigned to only one ground-truth vehicle.
7. Classifies each ground-truth vehicle as:

   * **Correct** — detected plate exactly matches ground truth.
   * **Incorrect** — vehicle/event detected but plate text is incorrect.
   * **Missed** — no matching pipeline event.
8. Generates aggregated results by vehicle type and condition.

Detailed results are stored in:

```text
output/accuracy_results.csv
```

---

# 4. Overall Results

| Metric    | Count | Percentage |
| --------- | ----: | ---------: |
| Total     |    33 |       100% |
| Correct   |    17 |     51.52% |
| Incorrect |     2 |      6.06% |
| Missed    |    14 |     42.42% |

### Accuracy

```text
Accuracy = Correct / Total × 100

Accuracy = 17 / 33 × 100

Accuracy = 51.52%
```

The main limitation is **missed vehicles**, which account for 14 of the 33 ground-truth observations.

---

# 5. Results by Vehicle Type

| Vehicle Type |  Total | Correct | Incorrect | Missed |   Accuracy |
| ------------ | -----: | ------: | --------: | -----: | ---------: |
| Car          |     19 |      13 |         1 |      5 | **68.42%** |
| Bike         |     14 |       4 |         1 |      9 | **28.57%** |
| **Total**    | **33** |  **17** |     **2** | **14** | **51.52%** |

### Cars

The system correctly recognized **13 of 19 cars**, achieving an accuracy of **68.42%**.

Only one car produced an incorrect plate read, while five car observations were missed.

### Motorcycles

The system correctly recognized only **4 of 14 motorcycles**, achieving an accuracy of **28.57%**.

Motorcycles therefore represent the largest vehicle-type-specific failure category.

---

# 6. Results by Condition

| Condition |  Total | Correct | Incorrect | Missed |   Accuracy |
| --------- | -----: | ------: | --------: | -----: | ---------: |
| Clear     |     26 |      16 |         2 |      8 | **61.54%** |
| Day       |      7 |       1 |         0 |      6 | **14.29%** |
| **Total** | **33** |  **17** |     **2** | **14** | **51.52%** |

### Clear Conditions

The clear subset achieved **61.54% accuracy**.

```text
16 / 26 correctly recognized
```

### Day / Multiple-Vehicle Conditions

The multiple-vehicle daytime subset achieved only **14.29% accuracy**.

```text
1 / 7 correctly recognized
```

This indicates a significant performance degradation when multiple vehicles are present in the scene.

> Note: The current ground-truth dataset does not provide separate dusk, night, or rain categories. These conditions should be evaluated in a subsequent dataset/evaluation cycle.

---

# 7. Evaluation by Clip

| Clip                               | Ground Truth | Correct | Incorrect | Missed |   Accuracy |
| ---------------------------------- | -----------: | ------: | --------: | -----: | ---------: |
| `dataset_clear_01.mp4`             |           26 |      16 |         2 |      8 | **61.54%** |
| `dataset_multiple_vehicles_01.mp4` |            2 |       0 |         0 |      2 |  **0.00%** |
| `dataset_multiple_vehicles_02.mp4` |            5 |       1 |         0 |      4 | **20.00%** |
| **Total**                          |       **33** |  **17** |     **2** | **14** | **51.52%** |

The complete failure on `dataset_multiple_vehicles_01.mp4` is particularly significant and should be investigated.

---

# 8. Major Failure Patterns

## 8.1 Motorcycle Vehicles Frequently Missed

Motorcycles have the lowest accuracy:

```text
4 / 14 = 28.57%
```

Examples of missed motorcycle plates include:

| Timestamp | Ground Truth |
| --------: | ------------ |
|       14s | `LEQ3288`    |
|       30s | `AYJ2527`    |
|       35s | `LEN9009`    |
|      139s | `LEN910`     |
|      147s | `LRK4645`    |
|        6s | `APM7367`    |
|        6s | `LEX127`     |
|        9s | `LER7662`    |

This suggests that motorcycle detection and/or motorcycle plate localization requires further investigation.

---

## 8.2 Multiple-Vehicle Scenes Have Poor Performance

### `dataset_multiple_vehicles_01.mp4`

```text
Ground truth: 2
Correct:      0
Incorrect:    0
Missed:       2
Accuracy:     0%
```

Both vehicles were missed:

* Car: `AZF441`
* Bike: `LEM5133`

### `dataset_multiple_vehicles_02.mp4`

```text
Ground truth: 5
Correct:      1
Incorrect:    0
Missed:       4
Accuracy:     20%
```

Only:

```text
BKV306 → BKV306
```

was correctly recognized.

The following vehicles were missed:

* `APM7367`
* `LEX127`
* `LER7662`
* `ABA196`

This indicates that simultaneous/multiple-vehicle scenes are currently a significant failure mode.

---

# 9. OCR Errors

Two incorrect plate readings were observed.

| Timestamp | Ground Truth | Detected  | Type      |
| --------: | ------------ | --------- | --------- |
|       11s | `LEM2025`    | `LEH2024` | OCR error |
|       46s | `LEE369`     | `LEEB369` | OCR error |

These cases indicate that the vehicle was associated with an event, but the OCR output did not exactly match the ground-truth plate.

---

# 10. Missed Vehicle Events

The following ground-truth observations had no matching pipeline event.

### Clear Dataset

| Timestamp | Vehicle | Plate     |
| --------: | ------- | --------- |
|       14s | Bike    | `LEQ3288` |
|       20s | Car     | `AWJ431`  |
|       25s | Car     | `LEC3799` |
|       30s | Bike    | `AYJ2527` |
|       35s | Bike    | `LEN9009` |
|       43s | Car     | `AUT094`  |
|      139s | Bike    | `LEN910`  |
|      147s | Bike    | `LRK4645` |

### Multiple-Vehicle Dataset

| Clip                               | Timestamp | Vehicle | Plate     |
| ---------------------------------- | --------: | ------- | --------- |
| `dataset_multiple_vehicles_01.mp4` |        6s | Car     | `AZF441`  |
| `dataset_multiple_vehicles_01.mp4` |        6s | Bike    | `LEM5133` |
| `dataset_multiple_vehicles_02.mp4` |        6s | Bike    | `APM7367` |
| `dataset_multiple_vehicles_02.mp4` |        6s | Bike    | `LEX127`  |
| `dataset_multiple_vehicles_02.mp4` |        9s | Bike    | `LER7662` |
| `dataset_multiple_vehicles_02.mp4` |        6s | Car     | `ABA196`  |

These cases should be investigated using the corresponding annotated videos and evidence images.

---

# 11. Duplicate / Multiple Candidate Events

The scoring process identified **multiple candidate pipeline events within the configured timestamp matching window for several ground-truth observations**.

These are currently treated as **candidate duplicate/multiple-event cases**.

They should not automatically be interpreted as confirmed physical duplicate detections because:

* A vehicle may legitimately generate multiple OCR events.
* Tracking can produce multiple event candidates.
* Multiple vehicles can be present near the same timestamp.

A definitive duplicate-event analysis requires inspection of:

* Track IDs
* Event timestamps
* Vehicle bounding boxes
* Evidence frames

Therefore, no definitive physical duplicate count is reported in the overall accuracy metrics.

---

# 12. Evidence and Failure Investigation

The pipeline generated evidence for detected events under:

```text
output/<clip>_evidence/
```

These evidence images should be reviewed for representative failure cases.

Priority cases for inspection:

1. Missed motorcycles.
2. Missed vehicles in multiple-vehicle scenes.
3. OCR error `LEM2025 → LEH2024`.
4. OCR error `LEE369 → LEEB369`.
5. Complete failure of `dataset_multiple_vehicles_01.mp4`.

The purpose of this inspection is to determine whether failures originate from:

* Vehicle detection
* Vehicle tracking
* Plate detection
* OCR
* Confidence thresholds
* Multiple-object handling

---

# 13. Known vs. Potentially New Failure Modes

The observed failures should be cross-referenced against the existing:

```text
cv-service/README.md
```

Potential failure modes identified during this evaluation include:

* Low motorcycle detection/recognition performance.
* Poor performance in multiple-vehicle scenes.
* Complete event failure on `dataset_multiple_vehicles_01.mp4`.
* High number of missed vehicles.
* OCR character substitutions.

Any genuinely new failure mode should be documented through a separate issue rather than fixed as part of the accuracy-evaluation work.

---

# 14. Evaluation Artifacts

The evaluation generated the following artifacts:

### Ground Truth

```text
sample_data/ground_truth.csv
```

### Pipeline Events

```text
output/*_events.json
```

### Annotated Videos

```text
output/*_annotated.mp4
```

### Evidence Images

```text
output/*_evidence/
```

### Detailed Scoring Results

```text
output/accuracy_results.csv
```

### Scoring Script

```text
scripts/score_accuracy.py
```

---

# 15. Limitations

The current evaluation has several limitations:

1. The dataset contains only **33 ground-truth vehicle observations**.
2. The dataset currently covers only **clear and daytime multiple-vehicle conditions**.
3. There are no dedicated ground-truth categories for **dusk, night, or rain**.
4. The dataset is therefore not sufficient to make conclusions about night/rain performance.
5. Timestamp-based matching can be ambiguous when multiple vehicles appear close together.
6. Duplicate-event analysis requires additional track-level inspection.

The results should therefore be treated as a **baseline evaluation**, not a final production accuracy measurement.

---

# 16. Conclusion

The current ANPR pipeline achieved:

> **51.52% overall accuracy (17/33 vehicles)**

Performance varies substantially by vehicle type:

* **Cars:** 68.42%
* **Motorcycles:** 28.57%

The most significant observed failure modes are:

1. Missed motorcycle vehicles.
2. Poor performance in multiple-vehicle scenes.
3. Missed vehicle events.
4. Occasional OCR character errors.

The results provide a baseline for subsequent pipeline improvements and further evaluation.

In particular, the motorcycle and multiple-vehicle failures should be investigated before drawing conclusions about overall production readiness.

The next evaluation cycle should also include dedicated **day, dusk, night, and rain** ground-truth samples so that environmental-condition performance can be measured independently.

```

This structure is much better for a **GitHub PR** because it separates **what was tested, how it was tested, the actual numbers, failures, evidence, limitations, and conclusion**.
```
