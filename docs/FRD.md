# Functional Requirements Document (FRD)
## Campus ANPR (Automatic Number Plate Recognition) System

---

## 1. Introduction

### 1.1 Purpose
This document specifies the functional requirements for a campus-wide Automatic Number Plate Recognition (ANPR) system. It is intended for the development team, project supervisors, and evaluators, and serves as the reference baseline for what the system must do.

### 1.2 Scope
The system will monitor vehicle entry and exit at 3 campus gates, each serving a distinct parking area. It will detect vehicles (cars and motorbikes), recognize license plates using pretrained computer-vision models, cross-reference a vehicle database, log all detection events, raise real-time alerts for banned vehicles, and present a live operations dashboard to administrators and gate staff.

### 1.3 Intended Audience
- Development team (implementers)
- Project supervisor(s) / evaluators
- Campus security/admin staff (end users of the dashboard)

### 1.4 Definitions & Abbreviations
| Term | Meaning |
|---|---|
| ANPR | Automatic Number Plate Recognition |
| OCR | Optical Character Recognition |
| Gate | A physical entry/exit point with an associated camera |
| Track / Track ID | A vehicle's persistent identity across consecutive video frames |
| Detection Event | One logged record of a vehicle passing a gate |
| RBAC | Role-Based Access Control |

### 1.5 References
- `docs/PROJECT_SPEC.md` — engineering requirements brief
- `docs/ARCHITECTURE.md` — system architecture
- `docs/DATABASE_SCHEMA.md` — database design

---

## 2. Overall Description

### 2.1 Product Perspective
The system is a new, standalone application composed of four cooperating services: a computer-vision/ANPR processing service, a backend API, a database, and a web-based dashboard frontend. It integrates with existing CCTV/IP cameras at 3 gates but does not depend on any other campus IT system.

### 2.2 Product Functions (Summary)
1. Detect vehicles and read license plates in real time at 3 gates
2. Track vehicles and infer entry/exit direction from a single camera per gate
3. Prevent duplicate logging of the same physical vehicle pass
4. Match detected plates against a vehicle database (verified/unverified/banned)
5. Raise real-time alerts for banned vehicles
6. Maintain a full historical log of all detection events with evidence images
7. Provide a live dashboard: occupancy, recent activity, statistics
8. Allow authenticated admins to manage the vehicle database and gate configuration
9. Enforce role-based access to system functions

### 2.3 User Classes and Characteristics
| Role | Description | Typical Access |
|---|---|---|
| Super Admin | Full system control | All functions, incl. user management |
| Gate Operator | Campus security staff monitoring gates | View dashboard, view alerts, verify/flag vehicles |
| Viewer | Read-only stakeholder (e.g. faculty in charge) | View dashboard and history only, no edits |

### 2.4 Operating Environment
- On-premise deployment on campus hardware (server with optional GPU)
- 3 IP/CCTV cameras streaming over campus LAN (RTSP or equivalent)
- Web dashboard accessed via browser on campus network

### 2.5 Assumptions and Dependencies
- Cameras are fixed-position, provide a reasonably unobstructed view of the gate lane, and support RTSP (or a compatible) streaming protocol
- Campus network bandwidth is sufficient to carry 3 concurrent video streams to the processing server
- License plates conform to a recognizable standard format (local plate format); highly damaged, obscured, or non-standard plates are out of scope for guaranteed accuracy
- One camera per gate handles both entry and exit traffic (no physical lane separator at Gates 2/3); camera mounting angle at each gate is adjustable and should target roughly 30-45° off head-on to keep both direction-inference signals (scale trend and displacement vector) reasonably strong

---

## 3. Functional Requirements

Each requirement has a unique ID for traceability (referenced during testing/acceptance).

### 3.1 Vehicle & Plate Detection

| ID | Requirement |
|---|---|
| FR-1.1 | The system shall detect vehicles (cars and motorbikes) in real time from each gate's camera feed. |
| FR-1.2 | The system shall detect and localize the license plate region for each detected vehicle. |
| FR-1.3 | The system shall extract the plate's alphanumeric string via OCR, together with a confidence score. |
| FR-1.4 | The system shall assign a persistent track ID to each vehicle for as long as it remains visible in frame. |
| FR-1.5 | The system shall select the highest-confidence OCR read per track (rather than logging every frame) before emitting a detection event. |
| FR-1.6 | The detection pipeline's individual stages (vehicle detection, plate detection, OCR, tracking) shall each be implemented as swappable, independently replaceable modules. |

### 3.2 Direction & Duplicate Handling

| ID | Requirement |
|---|---|
| FR-2.1 | The system shall determine whether a tracked vehicle is entering or exiting, using a single camera per gate, independently per vehicle track (never as a single decision for the whole gate). |
| FR-2.2 | The system shall log at most one detection event per physical vehicle pass, even if the vehicle is tracked across multiple frames or briefly re-detected after occlusion. |
| FR-2.3 | The system shall infer direction using two fused signals per track: (a) bounding-box scale trend (growing/shrinking over the track's lifetime) and (b) displacement-vector direction (centroid movement compared against a calibrated per-gate reference vector). When the two signals disagree, the system shall use the higher-confidence signal and record the event as low-confidence direction for later review. |
| FR-2.4 | The system shall support configuring, per gate, a one-time calibrated reference vector representing the known inbound direction for that camera's mounting angle. |
| FR-2.5 | The system shall delay committing a direction decision for a track when both direction signals are weak/ambiguous (e.g. a vehicle stopped or idling), rather than committing on early, noisy frames. |

### 3.3 Detection Event Logging

| ID | Requirement |
|---|---|
| FR-3.1 | The system shall record, for every detection event: plate number (as read), gate/parking area, timestamp, entry/exit direction, and OCR confidence. |
| FR-3.2 | The system shall capture and store an evidence image (vehicle frame and/or plate crop) for every detection event. |
| FR-3.3 | The system shall attempt to match each detected plate against the vehicle database and record the match result (matched vehicle, match confidence, or "unmatched"). |

### 3.4 Vehicle Database

| ID | Requirement |
|---|---|
| FR-4.1 | The system shall maintain a vehicle database with owner details, vehicle details, verification status, parking assignment, and banned status. |
| FR-4.2 | The system shall classify each vehicle as Verified, Unverified, or Banned. |
| FR-4.3 | Authorized admins shall be able to create, update, verify, and ban/unban vehicle records. |
| FR-4.4 | The system shall record a reason and timestamp whenever a vehicle is banned. |

### 3.5 Alerts

| ID | Requirement |
|---|---|
| FR-5.1 | The system shall generate a real-time alert when a detected plate matches a banned vehicle. |
| FR-5.2 | Banned-vehicle alerts shall be pushed to the dashboard immediately (not on a polling delay). |
| FR-5.3 | The system shall visually and/or audibly distinguish banned-vehicle alerts from routine detections on the dashboard. |
| FR-5.4 | The system shall flag unmatched/unrecognized plates for review, at lower urgency than banned-vehicle alerts. |

### 3.6 Dashboard

| ID | Requirement |
|---|---|
| FR-6.1 | The dashboard shall display the count and list of vehicles currently inside campus, derived from unmatched entry/exit pairs. |
| FR-6.2 | The dashboard shall display per-gate/per-parking-area occupancy. |
| FR-6.3 | The dashboard shall display a live, continuously updating feed of recent detections with evidence thumbnails. |
| FR-6.4 | The dashboard shall display summary statistics (e.g. entries/exits per period, occupancy trends). |
| FR-6.5 | The dashboard shall update in real time via a push mechanism (e.g. WebSocket), not manual refresh. |

### 3.7 History & Search

| ID | Requirement |
|---|---|
| FR-7.1 | The system shall allow authorized users to search and filter the complete detection history by plate number, date range, gate, direction, and vehicle status. |
| FR-7.2 | Search results shall be paginated and shall allow drill-down to a detection event's evidence image. |

### 3.8 Authentication & Access Control

| ID | Requirement |
|---|---|
| FR-8.1 | The system shall require authentication for all administrative and dashboard access. |
| FR-8.2 | The system shall enforce role-based access control (Super Admin, Gate Operator, Viewer) as defined in section 2.3. |
| FR-8.3 | Only Super Admin and (where applicable) Gate Operator roles shall be able to modify vehicle records or gate configuration; Viewer role shall be read-only. |

### 3.9 Gate/Camera Configuration

| ID | Requirement |
|---|---|
| FR-9.1 | Authorized admins shall be able to configure each gate's camera source, associated parking area, and the calibrated inbound reference vector used for direction inference. |
| FR-9.2 | The system shall continue operating on unaffected gates if one gate's camera or processing worker fails. |

---

## 4. Use Cases

### UC-1: Vehicle Entry Detection
**Actor:** System (automated)
**Trigger:** Vehicle enters camera view at a gate
**Flow:** Vehicle detected → plate detected → OCR read → track direction determined as "entry" → dedup check passes → matched against vehicle DB → detection event logged → dashboard updated → (if banned) alert raised
**Postcondition:** Detection event exists in history; occupancy count updated

### UC-2: Banned Vehicle Alert
**Actor:** Gate Operator (recipient), System (trigger)
**Trigger:** Detected plate matches a vehicle with status = Banned
**Flow:** Detection event processed → banned match found → alert pushed to dashboard in real time → operator sees prominent alert with evidence image and vehicle details
**Postcondition:** Alert visible on dashboard; event flagged in history as `alert_triggered = true`

### UC-3: Admin Bans a Vehicle
**Actor:** Super Admin
**Trigger:** Admin identifies a vehicle that should be banned (e.g. reported incident)
**Flow:** Admin searches/selects vehicle in management UI → sets status to Banned → enters reason → confirms
**Postcondition:** Vehicle status updated; future detections of this plate trigger alerts

### UC-4: Search Detection History
**Actor:** Any authenticated role
**Trigger:** User needs to look up past activity (e.g. "when did plate X last enter?")
**Flow:** User opens history page → applies filters (plate/date/gate/status) → views paginated results → optionally opens evidence image
**Postcondition:** None (read-only)

### UC-5: Gate Camera Failure
**Actor:** System (automated)
**Trigger:** One gate's camera feed drops or CV worker crashes
**Flow:** Affected gate stops producing detection events → other 2 gates continue operating normally → (recommended, see NFR) system logs/flags the outage
**Postcondition:** Two gates fully operational; failed gate flagged for attention

---

## 5. Constraints, Assumptions & Edge Cases

### 5.1 Constraints
- No models may be trained from scratch; only pretrained, open-source or commercially-permissible models may be used for detection/OCR (light fine-tuning of OCR on local plate formats is permitted).
- Must be deployable on campus on-prem hardware without a hard dependency on external cloud inference APIs.

### 5.2 Assumptions
- Plate formats are reasonably standard and legible under normal daytime conditions.
- Each gate has a single camera covering both entry and exit lanes.

### 5.3 Known Edge Cases (explicitly acknowledged, not treated as defects)
- **Low light / night conditions:** OCR accuracy will degrade without IR-capable cameras; treat as a documented limitation unless IR hardware is confirmed available.
- **Motorbike plates:** smaller, often angled — expect measurably lower OCR accuracy than cars; should be evaluated and reported separately from car accuracy.
- **Simultaneous mixed-direction traffic (multiple vehicles entering and exiting at once):** not actually an edge case for the fused direction method — direction is computed per-track independently, so concurrent opposite-direction vehicles are handled the same as isolated ones. The genuine risk here is tracker identity swaps during occlusion (e.g. one vehicle briefly hidden behind another), which is a tracking-quality issue rather than a direction-logic issue.
- **Idling/stopped vehicles (e.g. waiting at a barrier):** both direction signals can be weak while a vehicle is stationary; the system delays commitment until signals are unambiguous (FR-2.5) rather than risk a wrong call on a stopped vehicle.
- **Curving vehicle paths after the gate** (e.g. turning into a parking row): can distort the displacement-vector signal if computed over the full track life; mitigated by weighting recent frames near the gate more heavily than the full trajectory.
- **OCR misreads (character confusion, e.g. 0/O, 1/I):** handled via fuzzy matching against the vehicle database rather than requiring exact string match.

---

## 6. Non-Functional Requirements (Summary)

| ID | Requirement |
|---|---|
| NFR-1 | Detection-to-alert latency for banned vehicles shall not exceed ~2-3 seconds under normal load. |
| NFR-2 | The failure of one gate's camera or processing worker shall not affect the other gates. |
| NFR-3 | Each pipeline stage (detection, tracking, OCR) shall be replaceable without modifying unrelated components. |
| NFR-4 | The system shall run on on-premise campus hardware; GPU is preferred but not mandatory (CPU fallback with reduced frame rate is acceptable). |
| NFR-5 | Evidence images and detection history shall be retained for a configurable retention period (define specific duration with project stakeholders). |

---

## 7. Acceptance Criteria (high level)
- All 3 gates simultaneously detect and log vehicle entries/exits from live or recorded test footage.
- A banned test vehicle triggers a dashboard alert within the latency target (NFR-1).
- No duplicate detection events are logged for a single physical vehicle pass in test footage.
- Dashboard occupancy counts match manually verified counts from test footage.
- History search returns correct results for plate, date range, gate, and status filters.
- Role-based access is enforced: a Viewer-role account cannot modify vehicle records or gate config.

---

## 8. Open Items for Stakeholder Sign-off
- `PROJECT_SPEC.md` section 5 now sets placeholder targets split by vehicle type (cars >90%, motorbikes >75%), addressing the "don't use a single blanket number" concern. Still needs stakeholder confirmation and replacement with actual measured accuracy once tested — the specific thresholds above are placeholders, not validated numbers.
- Confirm evidence image/data retention period (NFR-5).
- Confirm whether IR/low-light-capable cameras will be available, which directly affects night-time accuracy expectations.
- Confirm final list of gate parking areas and their names for use in DB seed data.
