# A readable account of the reproduction

Date: 2026-08-31

Status: local rewrite complete and verified; public deployment awaits approval

## Request

Rewrite the README and public site in the voice of a researcher explaining an
experiment. Lead with the question, the paper's instructions, the implementation,
the result, and the remaining explanations. Preserve the two existing paper
pages and add a separate report for the audited FC-SAE reproduction.

This work does not approve another experiment or change the completed run.
Checkpoint 2 in the [research plan](2026-08-23-clean-reader-reproduction-rebase.md)
still applies.

## Content and scope

- The homepage and README introduce the research in ordinary language and show
  the audited Table III result, not an internal workflow or a verdict ladder.
- The new report follows abstract, question, methods, results, checks,
  discussion, limitations, next questions, and references. It links source
  locations to the exact code revision, includes a model diagram, and reports
  all seven metrics and the choices needed to execute the paper.
- Completed toy experiments, the real-data run, and later artifact checks are
  distinguished. Proposed experiments have no invented results.
- The report states non-reproduction for one declared completion and seed.
  Fixed-score threshold limits are not presented as a bound on every model.
  No claim of fabrication, statistical impossibility, or zero useful work is
  added without evidence.
- The existing electricity method page becomes a clearly dated earlier account
  with a prominent link to the new report. The water-study page keeps its
  evidence and corrections, with simpler, appropriately qualified wording.
- Remove obsolete charter commands and duplicated agent rituals. Keep the
  scientific safeguards and the durable reasoning, historical plans, results,
  source specifications, and correction record. Reduce redundant overview
  documents instead of deleting research history.
- Retain the static GitHub Pages site. Do not introduce a framework, move the
  hosting, or start training. Public deployment awaits the user's answer;
  otherwise provide a local preview and commit the work.

## Steps

1. [x] Inspect public pages, shared instructions, and the source/result records.
2. [x] Verify the source passages and code references used in the report.
3. [x] Rewrite the homepage and README; show a first local preview.
4. [x] Add the report, diagram, and evidence links; update the two earlier pages.
5. [x] Simplify charter-related instructions and documentation navigation.
6. [x] Verify metrics against saved JSON, check links and document structure,
       run deterministic tests, update status, and commit.
7. [x] Hand off the local preview. Do not publish without the user's answer.

## Verification

Use the immutable result and audit records in
`studies/atk-2022-deep-autoencoder/results/clean_reader_anchor_20260831/`.
Code links use scientific revision `a88d17477ad96b01ffa44a50d8ce051dd8d2b5ca`.
No raw data, original paper PDF, credentials, or machine-specific access details
are added to the website. The existing report PDFs remain available and are
clearly described as earlier reports.

Final checks cover all site-relative links, fragment IDs, tables, model widths,
the complete seven-metric comparison, and the explicit limits of the finding.
Use the existing static development flow and `bash scripts/test.sh`; no browser
automation or experimental jobs are needed for this editorial task.

## Completed checks

- Visually rechecked the cited source passages on printed pages 4109, 4115,
  and 4116; checked code links against the experiment's exact revision.
- `bash scripts/test.sh`: all 187 tests passed (140 study tests and 47 root
  tests). Eight new static tests check report numbers, saved diagnostics,
  diagram widths, historical scope, metadata, and site/source links.
- All 77 local Markdown links in the 17 changed/new Markdown files resolved.
- All four HTML pages, shared stylesheet, model diagram, and homepage social
  preview returned HTTP 200 from the local server. No browser automation was
  used.
- `git diff --check` passed. Scientific implementation files, raw inputs,
  results, and existing PDF reports are unchanged.
- CI now fetches Git history so the source-link tests can inspect the exact
  historical revisions cited by the report.

The local preview is served at `http://127.0.0.1:8765/`; the new report is under
`papers/atk-2022-deep-autoencoder/reproduction/`. Public GitHub Pages is not
updated by a local commit. Review or publication is the next editorial action;
the next scientific action still requires Checkpoint 2 approval.
