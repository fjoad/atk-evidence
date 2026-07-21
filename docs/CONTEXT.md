# ATK Evidence — Working Memory

**Last updated:** 2026-07-21

## Environment quirks

- Host is Apple M1 Max/macOS; public setup uses root `.venv` while the pre-publication workspace still has a legacy `replication/.venv`.
- Official CER/ISET consumption archives are restricted by ISSDA and require an authorized account; no authorized local copy has been found.
- Built-in macOS `unzip` failed on the multipart SGCC archive; 7-Zip 26.02 verified and extracted it successfully.
- In zsh, lowercase `path` is tied to `PATH`; never use it as a loop variable because system commands disappear for that shell.

## Working patterns

- Run deterministic tests with `bash scripts/test.sh`; it supports the root environment and the legacy local environment.
- Preserve raw files in place and identify them by checksum; study artifacts belong under `studies/<study-id>/results/`.

## Don't repeat

- Do not substitute 48-day SGCC windows for the paper's 48 half-hour CER profiles when assessing the primary reproduction hypothesis.
- Do not let literature provenance work displace the exact-data, paper-literal reproduction task.
- Do not silently correct the paper in the primary track; corrections belong in a separately labeled controlled analysis.
- Do not report a best/lucky seed as a reproduced result.

## Open questions

- Exact membership and order of the three-paper core corpus after Paper 1.
- Reproduction tolerances, finite hyperparameter envelope, seed count, split policy, and computational stopping rule must be frozen before confirmatory runs.
- Authorized acquisition path for the six CER/ISET consumption archives.
- GitHub repository is explicitly authorized as public under `fjoad/atk-evidence`.

## User emphases

- The honest hypothesis is that the reported numbers will not be reproducible from the papers as written, but the project must be genuinely open to being wrong.
- Exact paper-described algorithms and procedures are the highest priority; add nothing extra to the primary track.
- Ambiguities may use reasonable assumptions only with complete documentation.
- Evidence must come from rigorous experiments and statistical assessment, aiming for the strongest defensible conclusion.
- Deliver one LaTeX-style rebuttal/reproduction report per paper and a combined report.
- Claims should be supported by independently rerunnable proof-quality evidence; data acquisition and setup must be explicit from a fresh public clone.

## When to update this file

Update inline when a non-obvious environment fact, failed path, user emphasis,
or active decision appears. Keep entries terse and prune beyond roughly 200
lines. Promote durable causal corrections to `EVIDENCE-AND-LEARNINGS.md`.
