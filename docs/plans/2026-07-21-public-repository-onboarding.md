# Public Repository and Reproducible Onboarding — Implementation Plan

**Goal:** Publish `atk-evidence` as a public, multi-paper repository that a new
researcher can bootstrap, acquire or request the exact data for, verify inputs,
run tests, and continue without private chat context.

**Status:** In progress — implementation verified; publication steps remain

## Context

The initial Charter scaffold and Paper 1 artifacts exist locally, but the
repository name and top-level structure are too electricity-theft-specific and
dataset acquisition is not yet a complete outsider-facing workflow.

## Steps

### Task 1: Generalize the repository

- [x] Rename the project to ATK Evidence across shared docs.
- [x] Move Paper 1 under a stable `studies/<study-id>/` namespace.
- [x] Add a study registry and reserve independent report locations.
- [x] Update every internal path and command after the move.

### Task 2: Make data acquisition reproducible

- [x] Add a root bootstrap script for the pinned Python environment and tests.
- [x] Add an idempotent SGCC acquisition script with commit and checksum checks.
- [x] Fix the authorized CER downloader's root resolution and partial-download safety.
- [x] Add a standard-library data verifier with actionable missing-data output.
- [x] Add explicit public/restricted dataset instructions and token safety notes.

### Task 3: Verify and publish

- [x] Run shell syntax checks, Python compilation, deterministic tests, and local data verification.
- [x] Update CONTEXT, ARCHITECTURE, README, and evidence paths; update STATUS after merge.
- [ ] Commit on the feature branch, merge to `main`, and delete the branch.
- [ ] Create public GitHub repository `fjoad/atk-evidence` and push `main`.
- [ ] Verify the remote URL and public repository metadata.

## Verification

- [ ] Fresh-clone instructions contain no private filesystem assumptions.
- [ ] All tracked files use generic project naming and stable Paper 1 study paths.
- [ ] Raw data, PDFs, tokens, virtual environments, and large arrays remain ignored.
- [ ] Existing 7 deterministic tests pass after the move.
- [ ] `scripts/verify_data.py` recognizes the local verified SGCC data and reports CER as an actionable access gate.
- [ ] Git working tree is clean and `origin/main` contains the final commit.
