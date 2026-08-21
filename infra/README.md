# infra

Deployment and orchestration configs for the Campus ANPR System.

## Purpose

Holds everything needed to run the full stack (`postgres`, `redis`, `backend`, `cv-service`, `frontend`) together, plus environment/config templates:

- `docker-compose.yml` — local/on-prem multi-service orchestration
- `.env.example` — template for the environment variables each service needs; copy to `.env` and fill in real values (never commit a real `.env`)

## Status

Skeleton only. `docker-compose.yml` currently defines service shells (images, ports, env wiring, dependency ordering, healthchecks) but points at `backend/`, `cv-service/`, and `frontend/` build contexts that don't have Dockerfiles yet — those are added as each service is implemented in later phases. Full end-to-end wiring and deployment hardening (health checks, resource sizing, GPU vs. CPU notes for campus hardware) land in Phase 5 of the build plan (see `docs/CLAUDE_CODE_KICKOFF.md`).

## Usage (once services are implemented)

```bash
cp .env.example .env
# edit .env with real values
docker compose up --build
```

## Reference docs

- [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) — section 8 (deployment topology)
