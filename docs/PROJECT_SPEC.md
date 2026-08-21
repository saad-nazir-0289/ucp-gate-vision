# Campus ANPR System — Project Specification

## 1. Overview
A production-ready Automatic Number Plate Recognition (ANPR) system for a university campus with 3 physical gates, each serving a distinct parking area. The system detects vehicles (cars and motorbikes), reads license plates using pretrained/open-source CV models, tracks entry/exit, cross-references a vehicle database (verified / unverified / banned), and provides a real-time operations dashboard.

**Core constraint:** No models are trained from scratch. All detection/OCR stages use prebuilt, open-source or commercially-permissible pretrained models, integrated into a modular pipeline.

**Tech stack:** Python across both the CV/ANPR service and the backend API (FastAPI) — kept in one language deliberately so detection-event and gate-config schemas can be shared directly (e.g. via Pydantic models) between the two services instead of maintained twice. Frontend is React regardless. See docs/ARCHITECTURE.md section 3 for the full reasoning.

## 2. Goals
- Real-time (near-live) detection and logging at 3 gates simultaneously
- Accurate plate reads across cars and motorbikes, varied lighting/weather
- Zero/low duplicate log entries per physical vehicle pass
- Instant alerting when a banned vehicle is detected
- Full historical audit trail, searchable
- Runs reliably on on-prem campus hardware (not cloud-dependent)

## 3. Non-Goals (v1)
- Not building a custom-trained plate detector/OCR from scratch (only light fine-tuning of OCR permitted if needed for local plate formats)
- Not handling non-standard/foreign plates as a priority
- Not doing automated payment/billing (parking assignment tracking only)

## 4. Functional Requirements

### 4.1 Camera / Gate Ingestion
- 3 independent camera streams (RTSP/IP or USB/CCTV feed), one per gate
- Each gate camera handles BOTH entry and exit traffic (single camera, no physical separator at Gates 2/3; direction inferred per-vehicle-track using a fused scale-trend + displacement-vector method — see docs/ARCHITECTURE.md section 7)
- Configurable per-gate: gate name, associated parking area, camera source URI, calibrated inbound reference vector (used by the displacement-vector direction signal), camera mounting angle (informational, off head-on)

### 4.2 Detection Pipeline
- Vehicle detection: cars, motorbikes (extendable to bus/truck)
- License plate detection (localized bounding box within/near vehicle)
- Plate OCR: alphanumeric string extraction with confidence score
- Multi-object tracking across frames (persistent track ID per vehicle while in frame)
- Direction inference (entering vs exiting) per track
- Duplicate-suppression: one logged event per physical pass, not per frame

### 4.3 Vehicle Database
Fields per vehicle record:
- Plate number (normalized string)
- Owner name, contact, owner type (student/faculty/staff/visitor)
- Vehicle make/model/color
- Verification status: `verified` / `unverified` / `banned`
- Assigned parking area/gate (if applicable)
- Ban reason + ban date (if banned)
- Created/updated timestamps

### 4.4 Detection Event Log
Each detection event records:
- Plate number (as read), OCR confidence
- Gate ID, timestamp
- Direction: entry / exit
- Vehicle DB match (if any) + match confidence
- Evidence image (cropped plate + full frame snapshot), stored as file reference
- Track ID (internal)
- Alert flag (if matched to banned vehicle)

### 4.5 Real-Time Alerts
- Banned vehicle detected → immediate push (WebSocket) to dashboard/guard UI, audible/visual alert
- Unverified/unrecognized plate → flagged for review (lower urgency)

### 4.6 Dashboard
- Vehicles currently inside campus (live count + list, derived from unmatched entries)
- Per-gate/per-parking-area occupancy
- Live feed of recent detections (rolling, with thumbnails)
- Entry/exit stats (hourly/daily aggregates)
- Search/filter full historical log: by plate, date range, gate, status, direction
- Banned-vehicle alert panel

### 4.7 Admin & Auth
- Role-based access: Super Admin, Gate Operator, Viewer (read-only)
- Admin can manage vehicle DB (add/edit/verify/ban)
- Admin can configure gates/cameras
- Auth: JWT-based session, password hashing (bcrypt/argon2)

## 5. Non-Functional Requirements
- **Latency:** detection-to-dashboard-alert under ~2-3 seconds for banned vehicles
- **Accuracy target:** measured and tracked separately by vehicle type, since motorbike plates (smaller, often angled) are expected to read measurably worse than car plates (see section 6) — a single blanket number would mask that gap:
  - Cars: >90% correct plate read on clear daytime captures
  - Motorbikes: >75% correct plate read on clear daytime captures
  - Both are placeholder targets for pretrained OCR without local fine-tuning; confirm with stakeholders and replace with actual measured accuracy once tested (see `FRD.md` section 8, open item)
- **Resilience:** one gate/camera failure must not affect others; CV service crash must not take down backend/dashboard
- **Modularity:** each pipeline stage (vehicle detection, plate detection, OCR, tracking) swappable independently
- **Deployment:** must run on on-prem hardware (specify GPU/CPU target once available); no hard dependency on external cloud APIs for core detection

## 6. Out-of-Scope Risks to Flag Early
- Night/low-light accuracy will likely need IR-capable cameras or a lower accuracy target after dark
- Motorbike plates are smaller and often mounted at odd angles — expect lower accuracy than cars; plan for this explicitly rather than treating it as a bug
- Single-camera direction inference is per-track (independent per vehicle), so simultaneous mixed-direction traffic is handled naturally; the real risk is tracker identity swaps during occlusion, and weak/ambiguous signals (e.g. idling vehicles), both of which are mitigated but not eliminated by the fused-signal approach and confidence-based delay logic
