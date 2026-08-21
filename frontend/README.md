# frontend

React operations dashboard for the Campus ANPR System.

## Purpose

The web UI used by admins, gate operators, and viewers to monitor and manage the system:

- Login screen (JWT auth against `backend`)
- Live dashboard: currently-inside vehicle count/list, per-gate/per-parking-area occupancy, a rolling recent-detections feed with evidence thumbnails, a prominent banned-vehicle alert panel, and basic entry/exit stats charts
- Vehicle management page (admin): create/update/verify/ban vehicle records
- History page: search/filter the full detection-event log by plate, date range, gate, direction, and status, with drill-down to evidence images
- Role-aware UI that hides admin-only actions from `gate_operator`/`viewer` roles
- WebSocket client subscribed to the backend's real-time channel for live updates without polling

## Status

Not yet implemented — scaffolding only. The dashboard build lands in Phase 4 of the build plan (see `docs/CLAUDE_CODE_KICKOFF.md`).

## Reference docs

- [`docs/PROJECT_SPEC.md`](../docs/PROJECT_SPEC.md) — section 4.6
- [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) — section 2.6
- [`docs/FRD.md`](../docs/FRD.md) — sections 3.6, 3.7, 3.8
