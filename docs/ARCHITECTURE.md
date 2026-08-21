# Campus ANPR System — Architecture

## 1. Service Boundaries

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  CV/ANPR Service │────▶│  Message Broker   │────▶│  Backend API     │
│  (Python)        │     │  (Redis Streams)  │     │  (FastAPI, Python)│
│  1 worker/camera │     └──────────────────┘     └────────┬────────┘
└──────────────────┘                                        │
                                                              ▼
┌──────────────────┐                              ┌─────────────────┐
│  Evidence Storage │◀─────────────────────────────│   PostgreSQL     │
│  (local disk/MinIO)│                              │   (relational DB)│
└──────────────────┘                              └────────┬────────┘
                                                              │
                                                    WebSocket ▼
                                                     ┌─────────────────┐
                                                     │  Frontend        │
                                                     │  (React dashboard)│
                                                     └─────────────────┘
```

## 2. Component Responsibilities

### 2.1 CV/ANPR Service (Python)
- One process (or async task) per camera/gate
- Pipeline stages, each behind a swappable interface:
  1. `VehicleDetector` — YOLOv8/YOLOv11 (Ultralytics), COCO-pretrained (`car`, `motorcycle` classes)
  2. `PlateDetector` — pretrained YOLO plate-detection model (Roboflow Universe / HuggingFace)
  3. `Tracker` — ByteTrack (via Ultralytics `.track()`) for persistent track IDs
  4. `PlateOCR` — PaddleOCR (primary) / EasyOCR (fallback)
- Direction inference: fused scale-trend + displacement-vector method (see section 7) — single camera per gate, no physical separator or dual-zone geometry required
- Local dedup cache (short TTL, in-memory or Redis) to avoid multi-frame duplicate events per pass
- Publishes structured detection events to Redis Stream (one stream per gate or one shared stream tagged by gate_id)
- Does NOT talk to Postgres directly — stays decoupled from backend/business logic

**Why this separation matters:** if the backend or DB is down, the CV service keeps processing and queuing events (Redis persists them) rather than dropping frames/data.

### 2.2 Message Broker (Redis Streams)
- Decouples CV service from backend
- Each event: `{plate, confidence, gate_id, direction, track_id, timestamp, evidence_image_path}`
- Backend consumes via consumer group (allows horizontal scaling of backend workers later)

### 2.3 Backend API (FastAPI, Python — see decision note in section 3)
- Consumes queue events
- Business logic:
  - Normalize/clean plate string
  - Match against vehicle DB (exact + fuzzy match for OCR error tolerance)
  - Determine verified/unverified/banned status
  - Update occupancy state (entry adds to "currently inside", exit removes)
  - Write detection_event row to Postgres
  - If banned → push alert via WebSocket immediately
- REST endpoints: vehicle CRUD, gate config, search/filter history, auth, reports
- WebSocket channel: live detections, occupancy updates, banned alerts
- Auth: JWT + role-based middleware

### 2.4 Database (PostgreSQL)
- Relational integrity across vehicles, gates, detection_events, parking_assignments, users
- See `DATABASE_SCHEMA.md` for full schema

### 2.5 Evidence Storage
- Local disk (served via static path) for v1; MinIO (S3-compatible) if you want future cloud portability
- DB stores only the file path/URL, never binary blobs in Postgres

### 2.6 Frontend (React)
- Dashboard: live occupancy, recent detections feed, alert panel, stats
- Vehicle management UI (admin)
- History search/filter UI
- WebSocket client for real-time updates (Socket.IO or native WS)

## 3. Key Architectural Decisions (and why)

| Decision | Choice | Reasoning |
|---|---|---|
| CV service language | Python | All pretrained model ecosystems (Ultralytics, PaddleOCR) are Python-first |
| Backend framework | FastAPI (Python) | Same language as CV service — detection-event/gate-config schemas shared directly via Pydantic models instead of duplicated across two languages; async support fits the WebSocket + Redis-consumer + REST workload well. Node/Express is a valid alternative only if the team is materially stronger in JS and prefers one language across frontend+backend — not chosen here given the schema-sharing benefit. |
| Broker | Redis Streams | Lightweight, easy to self-host on campus hardware, good enough throughput for 3 gates; avoids RabbitMQ/Kafka operational overhead |
| DB | PostgreSQL | Needs relational integrity + concurrent writes from 3 gates + decent query performance for history search |
| Real-time updates | WebSocket (not polling) | Banned-vehicle alerts need near-instant delivery |
| Tracking | ByteTrack | Integrates directly with Ultralytics, no separate training, good accuracy/speed tradeoff |
| Deployment target | On-prem, GPU-optional | Campus hardware may not have GPU per gate; YOLOv8n is CPU-viable at reduced FPS |

## 4. Data Flow (single detection event, end to end)
1. Camera frame → CV service reads frame
2. YOLO vehicle detector finds vehicle bounding box
3. Plate detector finds plate bounding box within/near vehicle
4. Tracker assigns/maintains track ID across frames
5. OCR reads plate string + confidence once per track (best-frame selection, not every frame)
6. Direction inferred from the track's fused scale-trend + displacement-vector signal (see section 7); low-confidence cases are held until more frames accumulate rather than committed early
7. Dedup check (has this track already been logged?)
8. Event published to Redis Stream
9. Backend consumes event, matches against vehicle DB, writes to Postgres
10. If banned → WebSocket push to dashboard; else → normal live-feed update
11. Frontend dashboard updates in real time

## 5. Modularity Requirement (how to satisfy it in code)
Define abstract interfaces in the CV service, e.g.:
```python
class VehicleDetector(ABC):
    def detect(self, frame) -> list[Detection]: ...

class PlateDetector(ABC):
    def detect(self, frame, vehicle_box) -> list[Detection]: ...

class PlateOCR(ABC):
    def read(self, plate_crop) -> tuple[str, float]: ...

class Tracker(ABC):
    def update(self, detections, frame) -> list[Track]: ...
```
Concrete implementations (`YoloVehicleDetector`, `PaddleOCRPlateReader`, etc.) are swapped via config, not code changes. This is the mechanism that satisfies "swappable pretrained models" from the spec.

## 7. Direction Inference (single camera per gate, no separator)

Gates 2 and 3 have no physical lane separator between entering and exiting traffic (Gate 1 does, and can use simpler dedicated-lane logic if desired). Direction is inferred per-track using a **fused two-signal method**, computed independently for every tracked vehicle — never as a single decision for "the gate."

### 7.1 Signal 1 — Scale trend
Bounding-box area is sampled across each track's lifetime (trimming the first/last few frames, which tend to be noisy — partial vehicle visible, motion blur). The trend (e.g. slope via simple linear regression, or early-third vs late-third average comparison) indicates:
- Growing area → vehicle approaching the camera
- Shrinking area → vehicle receding from the camera

Most reliable near head-on camera angles; weakens as the camera angle moves toward side-on.

### 7.2 Signal 2 — Displacement vector
The track's centroid displacement `(Δx, Δy)` is computed from early to late frames (using the last N frames near the gate is preferred over full track life, since a vehicle's path can curve after passing the gate, e.g. turning into a parking row). This raw vector is compared via dot product against a **calibrated per-gate reference vector** — a one-time setup value recorded per gate representing "known inbound direction" for that camera's specific mounting angle:
- Positive dot product → matches inbound
- Negative dot product → matches outbound

This signal holds up at any camera angle, including angled (non-head-on) mounts, and is the primary signal at wider angles.

### 7.3 Fusion
1. Convert each raw signal into a `{direction, confidence}` vote (confidence reflects how unambiguous the underlying slope/vector magnitude was — a steep scale slope or long clean displacement = high confidence; near-flat/short/jittery = low confidence).
2. Combine as a weighted vote, not a raw numeric average (the two signals aren't on comparable scales):
   - Both agree → commit with high confidence
   - Disagree → take the higher-confidence vote, but flag the event internally as `direction_confidence: low` for later QA (useful for spotting a gate whose camera angle needs adjustment)
   - Both weak/ambiguous (e.g. vehicle stopped at a barrier, idling) → delay the decision, accumulate more frames rather than commit early

### 7.4 Camera angle recommendation
Since camera angle at each gate is adjustable, aim for roughly **30-45° off head-on**. This range keeps both scale-change and lateral displacement signals reasonably strong at the same time. Avoid the extremes: near-0° (head-on) weakens the displacement-vector signal; near-90° (side-on) weakens the scale-trend signal.

### 7.5 Detection event schema addition
`detection_events` (see `DATABASE_SCHEMA.md`) gains a `direction_confidence` field (`high` / `low`) so low-confidence direction calls are queryable/reviewable rather than silently mixed in with confident ones.

## 8. Deployment Topology (campus hardware)
- Single server (or small cluster) hosting: CV service (3 workers), Redis, Postgres, backend, frontend build served statically
- If no GPU available: run YOLOv8n (nano) variants, reduce inference resolution, consider frame-skipping (process every 2nd-3rd frame) — gate traffic speed doesn't require 30fps processing
- If GPU available: run larger YOLO variants for better accuracy, process more frames/sec
- Cameras connect via RTSP over campus LAN; CV service pulls streams directly (OpenCV/FFmpeg)
