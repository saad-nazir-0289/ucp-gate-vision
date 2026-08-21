# Claude Code Kickoff — Phased Prompts

Use these in order, in a project directory that already contains `docs/PROJECT_SPEC.md`, `docs/ARCHITECTURE.md`, and `docs/DATABASE_SCHEMA.md`. Start a fresh Claude Code session per phase (or continue in one session, whichever you prefer) — but always point it at the docs first so it has full context, rather than re-explaining everything in the prompt.

---

## Phase 0 — Repo scaffolding (do this first, once)

```
Read docs/PROJECT_SPEC.md, docs/ARCHITECTURE.md, and docs/DATABASE_SCHEMA.md in full.

Set up the top-level repo structure for this project as a monorepo with these
directories, each with its own README.md stub explaining its purpose:

- cv-service/       (Python — ANPR pipeline)
- backend/          (API — language per ARCHITECTURE.md decision)
- frontend/         (React dashboard)
- docs/             (already has the 3 spec files)
- infra/            (docker-compose, deployment configs)

Create a root README.md that links to each service's README and summarizes
the architecture at a glance (link to docs/ARCHITECTURE.md for detail).

Create a docker-compose.yml in infra/ with placeholder services for:
postgres, redis, backend, cv-service, frontend — don't implement the services
yet, just get the compose skeleton and .env.example files in place so later
phases can slot in.

Do not implement any business logic yet. This phase is scaffolding only.
```

---

## Phase 1 — CV/ANPR service, standalone, single video file input

```
Read docs/PROJECT_SPEC.md and docs/ARCHITECTURE.md, section 2.1 and section 5
(modularity requirement) specifically.

Build the CV/ANPR service in cv-service/ as a standalone pipeline that can be
tested against a single local video file before we wire in live RTSP streams
or the backend/DB. Do not integrate Redis or Postgres yet — this phase proves
the detection pipeline works in isolation.

Requirements:
1. Implement the abstract interfaces from ARCHITECTURE.md section 5:
   VehicleDetector, PlateDetector, PlateOCR, Tracker.
2. Concrete implementations:
   - VehicleDetector: Ultralytics YOLOv8 (pretrained COCO weights), filtered
     to 'car' and 'motorcycle' classes.
   - Tracker: ByteTrack via Ultralytics' built-in .track() method.
   - PlateDetector: research and recommend 1-2 specific pretrained
     license-plate-detection YOLO models (Roboflow Universe or HuggingFace)
     that are open-source/commercially permissible. Tell me which one you
     picked and why before implementing, and flag the license explicitly.
   - PlateOCR: PaddleOCR as primary implementation.
3. A pipeline runner script that:
   - Reads a local video file
   - Runs the full pipeline frame by frame
   - Draws bounding boxes + plate text on output frames, saves as an
     annotated output video
   - Prints each detection event (plate string, confidence, track ID,
     frame timestamp) to console/JSON log
4. Basic dedup: only emit one detection event per track ID (best-confidence
   frame for that track), not one per frame.
5. requirements.txt with pinned versions.
6. A short cv-service/README.md explaining how to run it against a sample
   video.

Do NOT implement direction inference (entry/exit) yet — that requires scale or vector based calculations which comes in Phase 2. Focus on getting accurate
detection + OCR working first.

After implementing, tell me what sample video I should test with and how
to evaluate accuracy.
```

---

## Phase 2 — Direction inference, live RTSP, dedup, Redis publishing

```
Read docs/ARCHITECTURE.md sections 2.1, 2.2, and 4 (data flow).

Extend the cv-service built in Phase 1:

1. Add live RTSP/webcam stream support (not just local video files) via
   OpenCV/FFmpeg, configurable per gate.
2. Implement direction inference per docs/ARCHITECTURE.md section 7 — the
   fused two-signal method, computed independently per track (never as a
   single decision for the whole gate):
   a. Scale-trend signal: bounding-box area slope over the track's
      lifetime (trim first/last few frames before computing).
   b. Displacement-vector signal: centroid displacement over recent frames
      near the gate, compared via dot product against a calibrated
      per-gate inbound reference vector (gates.inbound_reference_vector).
   c. Convert each raw signal to a {direction, confidence} vote; combine
      as a weighted vote (not a raw numeric average). If both signals
      agree, commit with high confidence. If they disagree, use the
      higher-confidence one and mark the event direction_confidence=low.
      If both are weak/ambiguous (e.g. vehicle idling), delay the
      decision and accumulate more frames rather than commit early.
3. Add a per-gate config file (YAML or JSON) specifying: gate_id, camera
   source URI, inbound_reference_vector, and camera_angle_deg
   (informational).
4. Provide a small calibration helper/script: given a short clip or live
   feed of one known-inbound vehicle, compute and output the reference
   vector to paste into that gate's config.
5. Implement Redis Streams publishing: on each finalized detection event
   (post-dedup, direction + direction_confidence determined), publish a
   JSON message matching the event schema in docs/ARCHITECTURE.md section
   4, step 8 (include direction_confidence in the payload).
6. Add an in-memory or Redis-backed short-TTL dedup cache so the same
   physical vehicle pass doesn't get logged twice even across brief tracking
   interruptions (e.g. vehicle briefly occluded).
7. Structured logging so we can debug false positives/negatives per gate,
   including how often direction_confidence=low occurs per gate (a high
   rate signals the camera angle may need adjusting).

Update cv-service/README.md with instructions for running against a live
RTSP source and verifying events land in Redis.
```

---

## Phase 3 — Database + Backend API

```
Read docs/DATABASE_SCHEMA.md and docs/ARCHITECTURE.md section 2.3 in full.

Build the backend service in backend/:

1. Postgres schema/migrations matching docs/DATABASE_SCHEMA.md exactly
   (use whatever migration tool is idiomatic for the chosen framework).
2. Redis Streams consumer that:
   - Reads detection events published by cv-service
   - Normalizes the plate string
   - Matches against the vehicles table (exact match first, then fuzzy
     match with a reasonable edit-distance threshold to tolerate OCR noise)
   - Writes a detection_events row
   - If matched vehicle status is 'banned', mark alert_triggered=true and
     push via WebSocket immediately
3. REST API:
   - Vehicle CRUD (create/update/verify/ban) — admin/super_admin only
   - Gate config CRUD
   - Detection history search/filter (by plate, date range, gate, direction,
     status) with pagination
   - Current occupancy endpoint (vehicles currently inside, per gate)
   - Stats endpoint (entries/exits per period, occupancy over time)
4. WebSocket channel broadcasting: new detection events (live feed),
   occupancy changes, banned-vehicle alerts.
5. JWT-based auth with roles: super_admin, gate_operator, viewer. Middleware
   enforcing role checks on write endpoints.
6. backend/README.md with setup + API endpoint list.

Write this to be resilient to the CV service being temporarily unavailable
(Redis consumer should reconnect/retry, not crash the backend).
```

---

## Phase 4 — Frontend dashboard

```
Read docs/PROJECT_SPEC.md section 4.6 (dashboard requirements) and
docs/ARCHITECTURE.md section 2.6.

Build the frontend in frontend/ (React):

1. Login screen (JWT auth against backend).
2. Main dashboard:
   - Currently-inside vehicle count + list (live, WebSocket-driven)
   - Per-gate/per-parking-area occupancy widgets
   - Recent detections feed (rolling list with evidence thumbnails)
   - Banned-vehicle alert panel (prominent, real-time, visually distinct)
   - Entry/exit stats charts (basic — hourly/daily)
3. Vehicle management page: list/search/add/edit vehicles, change
   verification status, ban/unban with reason.
4. History page: full detection_events search/filter UI (plate, date range,
   gate, direction, status), paginated table, click-through to evidence
   images.
5. Role-aware UI: hide admin-only actions from viewer/gate_operator roles.
6. WebSocket client wired to backend's real-time channel for live updates
   without polling.

Keep styling clean and functional — this is an operations dashboard, not a
marketing site. Use a component library if it speeds this up meaningfully,
otherwise plain CSS/Tailwind is fine.
```

---

## Phase 5 — Integration, docker-compose, deployment hardening

```
Wire everything from infra/docker-compose.yml (scaffolded in Phase 0) into a
working full-stack local deployment: postgres, redis, cv-service (3 gate
configs), backend, frontend.

1. Fill in real service definitions in docker-compose.yml.
2. Environment variable wiring across all services (.env.example complete
   and accurate).
3. Health checks per service.
4. Document, in infra/README.md, what's needed for actual campus deployment:
   GPU vs CPU inference tradeoffs, expected resource usage per gate, and
   what changes are needed to point cv-service at real campus RTSP cameras
   instead of test streams.
5. Run through the full pipeline end-to-end once and report any integration
   issues found.
```

---

## Tips for using these prompts
- Don't skip Phase 1's "standalone, video-file-only" step even though it feels slower — validating detection/OCR accuracy in isolation before wiring in Redis/DB/frontend will save you from debugging three systems at once when accuracy is actually the problem.
- After Phase 1, actually look at the annotated output video and OCR accuracy before proceeding — if plate detection accuracy is poor, it's much cheaper to swap the plate-detector model now than after Phases 2-5 are built around it.
- Ask Claude Code to explicitly state model licenses when it picks a pretrained plate-detector (Phase 1) — some Roboflow Universe models are non-commercial only, which matters for "production-ready."
- Keep each phase in a clean context where possible (new session), always re-pointing at the relevant docs section — this keeps Claude Code focused on the current phase's scope instead of drifting into re-litigating earlier decisions.
