# backend

API and business-logic service for the Campus ANPR System.

## Purpose

Sits between the CV service, the database, and the frontend dashboard. Responsibilities:

- Consume detection events from the Redis Streams broker (one consumer group, so backend workers can scale horizontally later)
- Normalize/clean the raw OCR'd plate string
- Match it against the vehicle database (exact match, then fuzzy match to tolerate OCR noise)
- Determine verified/unverified/banned status and update occupancy state
- Write `detection_events` rows to PostgreSQL
- Push an immediate WebSocket alert when a detected plate matches a banned vehicle
- Serve the REST API: vehicle CRUD, gate configuration, detection-history search/filter, auth, stats/reports
- Serve a WebSocket channel: live detections, occupancy updates, banned-vehicle alerts
- Enforce JWT-based authentication and role-based access control (`super_admin`, `gate_operator`, `viewer`)

## Language/framework decision

**FastAPI (Python)** — decided in `docs/PROJECT_SPEC.md` section 1 and `docs/ARCHITECTURE.md` section 3: same language as `cv-service`, so detection-event and gate-config schemas (Pydantic models) are shared directly instead of duplicated across two languages. Node/Express is noted in `ARCHITECTURE.md` only as a fallback if the team is materially stronger in JS — not the default.

## Status

Not yet implemented — scaffolding only. Postgres migrations, the Redis consumer, REST API, WebSocket channel, and auth land in Phase 3 of the build plan (see `docs/CLAUDE_CODE_KICKOFF.md`).

## Reference docs

- [`docs/PROJECT_SPEC.md`](../docs/PROJECT_SPEC.md) — sections 4.3–4.5, 4.7
- [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) — section 2.3
- [`docs/DATABASE_SCHEMA.md`](../docs/DATABASE_SCHEMA.md) — full schema
- [`docs/FRD.md`](../docs/FRD.md) — sections 3.3–3.5, 3.7, 3.8, 3.9
