# Decision: Public Domain-Neutral Multi-Study Repository

**Date:** 2026-07-21

## Decision

Publish the project as the public GitHub repository `fjoad/atk-evidence`, with
each paper isolated under `studies/<study-id>/` and a canonical TOML registry.

## What was considered

1. **Electricity-theft-specific repository:** rejected because later papers may
   cover other topics and should not inherit a misleading top-level scope.
2. **One repository per paper:** rejected because shared evidence discipline,
   onboarding, and synthesis would fragment, while assumptions can still be
   isolated by study directory.
3. **Chosen — ATK Evidence multi-study repository:** compact public identity,
   stable per-paper boundaries, and room for other domains and future corpora.

## Why this choice

The evidence must be independently inspectable and rerunnable. A public
repository provides persistent code and documentation, while per-study
isolation prevents assumptions and verdicts from leaking between papers.

## Impact

Every new paper must be registered before implementation and receive its own
source specification, data manifest, code, results, and report path. Public
onboarding scripts may fetch open data or authorized restricted files but never
embed credentials or redistribute protected inputs.

