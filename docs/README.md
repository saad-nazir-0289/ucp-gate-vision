# docs

Reference documents for the Campus ANPR System. Read these before working on any service — the phased build prompts in `CLAUDE_CODE_KICKOFF.md` assume they've been read in full.

- [`PROJECT_SPEC.md`](PROJECT_SPEC.md) — top-level engineering spec: overview, goals/non-goals, functional requirements by area, NFRs, and out-of-scope risks to flag early.
- [`FRD.md`](FRD.md) — Functional Requirements Document: the detailed elaboration of the spec — traceable requirements (FR-x.x), use cases, constraints/edge cases, and acceptance criteria.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — system architecture: service boundaries, component responsibilities, key architectural decisions and rationale, end-to-end data flow, the modularity requirement, and the direction-inference algorithm.
- [`DATABASE_SCHEMA.md`](DATABASE_SCHEMA.md) — PostgreSQL schema: `vehicles`, `gates`, `detection_events`, `occupancy_state`, `users`, and indexing notes.
- [`CLAUDE_CODE_KICKOFF.md`](CLAUDE_CODE_KICKOFF.md) — the phased prompt sequence used to build this project (Phase 0 scaffolding → Phase 5 integration).

## Cross-doc consistency (last checked 2026-08-20)

All previously-tracked inconsistencies are resolved as of this check:
- Direction inference: `PROJECT_SPEC.md` §4.1/§6, `ARCHITECTURE.md` §7, and `FRD.md` §5.3 agree on the fused scale-trend + displacement-vector method (no line geometry; per-track, so congestion isn't a real risk).
- Backend language: `PROJECT_SPEC.md` §1 and `ARCHITECTURE.md` §1/§2.3/§3 all now consistently name FastAPI (Python) as the decided choice, Node/Express as fallback-only.
- Per-gate config fields: `PROJECT_SPEC.md` §4.1, `ARCHITECTURE.md` §7.2, and the `gates` table in `DATABASE_SCHEMA.md` all use `inbound_reference_vector` + `camera_angle_deg` (no leftover line-coordinate references).
- Accuracy target: `PROJECT_SPEC.md` §5 now sets separate placeholder targets for cars (>90%) and motorbikes (>75%) instead of one blanket number; `FRD.md` §8's open item is updated to point at these as placeholders still pending stakeholder confirmation/real measurement.

Note: `ARCHITECTURE.md`'s section numbering skips from 5 to 7 (no section 6) — appears to be a pre-existing numbering gap rather than a content inconsistency; left as-is since renumbering would require updating cross-references in `PROJECT_SPEC.md` and every service README that cites "ARCHITECTURE.md section 7".

## Status

Reference material only — no action needed here beyond keeping these in sync if requirements change.
