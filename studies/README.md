# Studies

Each target paper lives in an independent directory identified by a stable,
human-readable study ID. A study owns its specification, code, dependency lock,
machine-readable results, and eventual report linkage. Assumptions, code, or
verdicts must not leak between studies without explicit documentation.

The canonical registry is [`registry.toml`](registry.toml).

## Adding a study

1. Add one `[[studies]]` entry to `registry.toml` with a new stable ID.
2. Create `studies/<study-id>/README.md`, `DATA_SOURCES.md`, and an initial
   reproduction-contract plan before implementing experiments.
3. Create `reports/<study-id>/` for the standalone LaTeX report.
4. Register exact claims, datasets, and ambiguity branches independently.
5. Do not import another study's verdict or implementation assumptions.

