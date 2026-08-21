# Campus ANPR System

Automatic Number Plate Recognition system monitoring vehicle entry/exit at 3 campus gates: real-time detection and OCR, entry/exit direction inference, vehicle-database matching, banned-vehicle alerts, and a live operations dashboard.

Full spec: [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md). Detailed requirements: [`docs/FRD.md`](docs/FRD.md). Full architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Architecture at a glance

```
CV/ANPR Service (Python) ──▶ Redis Streams ──▶ Backend API ──▶ PostgreSQL
   1 worker/camera                                  │
                                                      ▼
                                              WebSocket ──▶ Frontend (React)
```

- **[cv-service/](cv-service/README.md)** (Python) — one worker per gate camera; detects vehicles, detects/reads plates, tracks vehicles, infers entry/exit direction, dedups, and publishes detection events to Redis. Never talks to Postgres directly, so it keeps queuing events if the backend/DB is down.
- **[backend/](backend/README.md)** (API — FastAPI recommended, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#3-key-architectural-decisions-and-why)) — consumes Redis events, matches plates against the vehicle DB, writes history to Postgres, pushes real-time WebSocket alerts, and serves the REST/WebSocket API with JWT auth and role-based access.
- **[frontend/](frontend/README.md)** (React) — live dashboard (occupancy, recent detections, banned-vehicle alerts, stats), vehicle management, and history search, all driven by the backend's REST/WebSocket API.
- **[docs/](docs/README.md)** — requirements, architecture, and database schema references.
- **[infra/](infra/README.md)** — `docker-compose.yml` and `.env.example` for running the full stack together.

Each pipeline stage in `cv-service` (vehicle detection, plate detection, tracking, OCR) is a swappable interface with a pretrained-model implementation, selected via config rather than code changes — see `docs/ARCHITECTURE.md` section 5.

## Repo layout

```
.
├── backend/       # API service (business logic, DB writes, auth, REST + WebSocket)
├── cv-service/    # Python ANPR pipeline (detection, tracking, OCR, direction inference)
├── docs/          # FRD, architecture, database schema, build-phase prompts
├── frontend/      # React operations dashboard
└── infra/         # docker-compose.yml, .env.example
```

## Status

- **cv-service** — Phase 1 done: standalone detection/tracking/OCR pipeline, testable against a single local video file. No Redis/Postgres integration, no live RTSP, no direction inference yet. See [`cv-service/README.md`](cv-service/README.md).
- **backend, frontend** — not yet implemented (scaffolding only).
- **infra** — skeleton only; `docker-compose.yml`'s `backend`/`cv-service`/`frontend` build contexts have no Dockerfiles yet.

See [`docs/CLAUDE_CODE_KICKOFF.md`](docs/CLAUDE_CODE_KICKOFF.md) for the phased build plan (Phase 1: standalone CV pipeline ✅ → Phase 2: direction inference + live RTSP + Redis → Phase 3: DB + backend API → Phase 4: frontend → Phase 5: full integration via `infra/docker-compose.yml`).

## Getting started (once services are implemented)

```bash
cp infra/.env.example infra/.env
# edit infra/.env with real values
cd infra
docker compose up --build
```
