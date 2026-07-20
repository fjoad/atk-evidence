# Development and Experiment Workflow

## 1. Plan

- Read the relevant architecture and evidence records.
- Create or update the active plan under `docs/plans/`.
- For a major step, include `## CHECKPOINT: user review` and obtain approval.
- For confirmatory experiments, freeze the contract before implementation or runs.

## 2. Implement

- Follow the approved plan without silently changing the scientific contract.
- Keep paper-literal and controlled code/configuration paths separate.
- Stop at checkpoints and when exact data or authority is missing.

## 3. Test

- Write tests first for deterministic logic.
- Verify data hashes, parsing, shapes, split isolation, metrics, and run records.
- Preserve failures and interrupted runs.

## 4. Verify

- Run the full relevant suite.
- Confirm implementation matches the frozen branch and plan.
- Confirm evidence labels and report language do not overstate the result.

## 5. Finish

- [ ] Tests pass, with pass count reported.
- [ ] `docs/STATUS.md` reflects current state and next action.
- [ ] `docs/CONTEXT.md` is current and compact.
- [ ] `docs/EVIDENCE-AND-LEARNINGS.md` records causal changes.
- [ ] `docs/ARCHITECTURE.md` is updated if structure changed.
- [ ] All appropriate files are committed; raw/restricted inputs remain ignored.
- [ ] Report tests, docs, commits/branch, and next step.

## Branch discipline

On feature branches, edit the branch plan and CONTEXT/evidence as needed. Do not
edit canonical STATUS component state until merge to main. Create branches as
`feat/<short-name>` and plans as `docs/plans/YYYY-MM-DD-<short-name>.md`.

