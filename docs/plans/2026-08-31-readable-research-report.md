# A readable account of the reproduction

Date: 2026-08-31

Status: complete; published to GitHub and verified on GitHub Pages

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
  hosting, or start training. The user explicitly approved public deployment
  on 2026-08-31 after reviewing the local preview.

## Steps

1. [x] Inspect public pages, shared instructions, and the source/result records.
2. [x] Verify the source passages and code references used in the report.
3. [x] Rewrite the homepage and README; show a first local preview.
4. [x] Add the report, diagram, and evidence links; update the two earlier pages.
5. [x] Simplify charter-related instructions and documentation navigation.
6. [x] Verify metrics against saved JSON, check links and document structure,
       run deterministic tests, update status, and commit.
7. [x] Hand off the local preview, then publish after the user's explicit
       approval and verify the public pages.

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

## Diagnostic follow-up publication — 2026-08-31

The user subsequently approved the separate diagnostic sequence in the active
research plan. It is complete. The public report now adds the conditional
Softmax/MSE performance bound, paired useful-work comparisons, small adaptive
geometry control, two scientific figures, and remaining assumptions. The
original numerical table, scientific implementation, and older pages are
unchanged. README, current status, active plan, and explanation register agree
with the new finding; the original interpretation remains in the history.

- `bash scripts/test.sh`: 206 tests pass (140 study and 66 root), including
  14 static report tests.
- Report tests compare upper limits using upward rounding, gains/intervals,
  adaptive-control values, figure bytes, and source/result/contract hashes.
- All 64 local Markdown links in the 10 changed Markdown files checked at
  this stage resolve. The updated homepage, report, and both new figures
  return local HTTP 200. `git diff --check` passes.
- Scientific figure exports retain their original bytes; narrow
  `.gitattributes` entries exempt only those Matplotlib SVGs from Git's
  trailing-whitespace check. No code or measured record was reformatted.
- Existing GitHub Pages remains the publication destination. Report commit
  `97c92363a8264b5ab04d62dc2fc8c5b4c3d8cfb0` is pushed. The
  [Pages deployment](https://github.com/fjoad/atk-evidence/actions/runs/33407618030)
  completed successfully. All nine checked public files (four HTML pages,
  stylesheet, social preview, model diagram, and two diagnostic figures)
  were fetched and match the reviewed files byte for byte.
- Verified public report SHA-256:
  `da92d2bf9db2a2f6065d40b1cc3fc28f0fb47c9fb251dae0335f62b382771eba`.
  Bound figure: `f5704ba617c83f690cb2bfcc642b1ba76bd400cd1c1f44ee8f734a65569d1eb9`.
  Useful-work figure: `5d40a31d1cd28f79cd0f0e4d72d41dc121f1961f1dfc78eb26b0516bd2765da1`.
  No further scientific execution is started by publication.

## Initial publication

On 2026-08-31 the user explicitly requested publication to the existing GitHub
repository and GitHub Pages site. Commit `0f411ec`, including the supporting
evidence commits it cites, was pushed to `main`. The
[Pages deployment](https://github.com/fjoad/atk-evidence/actions/runs/33400529269)
completed successfully. All four public HTML pages, stylesheet, model diagram,
and social-preview image returned HTTP 200 and were byte-identical to the
reviewed local files.

- [Public homepage](https://fjoad.github.io/atk-evidence/)
- [Current reproduction report](https://fjoad.github.io/atk-evidence/papers/atk-2022-deep-autoencoder/reproduction/)
- [GitHub repository and README](https://github.com/fjoad/atk-evidence)

The README now links to the rendered public reports rather than their HTML
source files. Publication does not approve another experiment. The proposed
next questions are recorded under Checkpoint 2 in the active research plan;
experimental execution still requires that approval.
