# Contributing to UCP Gate Vision

This repository uses a simple branch-and-PR workflow so each contributor can work independently without destabilizing shared work.

## Branches

- `main` — stable shared branch. Branch from this and open PRs back into this branch.
- `msn` — Saad's personal working branch. Its history may be force-pushed/re-written, so **do not branch from `msn`** and do not use it as a shared integration branch.

## Standard workflow

1. Clone the repository.
2. Switch to `main` and pull the latest changes:

   ```bash
   git checkout main
   git pull
   ```

3. Create a clearly named branch from `main`, for example:

   ```bash
   git checkout -b afzaal/dataset-collection
   ```

4. Work independently on your branch and commit your own changes.
5. Do **not** push directly to `main` or `msn`.
6. When your task is ready, open a pull request from your branch **into `main`** and tag Saad for review.
7. Keep your branch current by merging/rebasing from `main` periodically.

## If you need something from `msn`

Only use work from `msn` when your task genuinely depends on a change that has not landed on `main` yet.

Do **not** pull or branch directly from `msn`.

Instead:

1. Ask Saad to merge the required change from `msn` into `main`.
2. Pull the updated `main` into your branch.

This keeps everyone building on stable history.

## CV service setup

Before starting a CV-related issue, follow the **Setup** section in `cv-service/README.md`.

Expected environment:

- Python 3.10 or 3.11
- Virtual environment
- `pip install -r requirements.txt`
- Plate-detector weights downloaded using `scripts/download_plate_model.py`

## Pull request expectations

A PR should include:

- A short summary of what changed
- The issue it addresses, e.g. `Closes #4`
- Test/evaluation evidence where relevant
- Any known limitations or follow-up issues

For evaluation work, include generated reports/results but avoid committing unnecessarily large raw video files unless the project has agreed on Git LFS or another storage method.

## Phase 1 ownership

Current Phase 1 split:

- **Afzaal** — dataset collection and camera-angle calibration
- **Hifza** — accuracy evaluation and night/rain investigation

See the corresponding GitHub issues for detailed acceptance criteria and dependencies.
