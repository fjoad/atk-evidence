# Website: show how the investigation develops

**Date:** 2026-09-20

**Audience:** an undergraduate taking an introductory statistics course.

## Current navigation: an open-ended index

The homepage is a growing paper index, not a collection fixed at three items.
Paper pages have a simple “All papers” link back to that index. Do not add a
cross-paper tab bar, “Paper N of 3” badges, or collection-size wording in page
metadata. The index itself is unnumbered. Registry sequence numbers remain
internal identifiers, not a public limit. This supersedes the original
three-tab layout described in the earlier brief below.

## Current refinement: a continuous account, not a dated diary

The user does not want calendar dates organizing the visible reading flow.
Use descriptive headings and natural transitions through the investigation.
Do not show update-date banners, dated section prefixes or editorial assembly
dates on the main pages. Publication years still identify the three papers;
dates remain in internal research logs and linked technical records. The
renderer keeps the already-published section IDs so old links continue to work.
This refines the earlier notebook brief below; the scientific sequence and
corrections remain intact.

## September 23: agreed three-paper notebook structure

This supersedes the staged first implementation below. The homepage is now a
quiet index with three full paper titles and years, not a featured-result card
or a project-wide thesis. Electricity theft/deep autoencoder are one registered
paper; data poisoning and water networks are the other two.

Each canonical paper URL contains a notebook with the paper's claims, starting
hypothesis, chronological checks/results/corrections, supporting records, and
a current conclusion at the end. Three ordinary navigation links styled as
tabs switch papers. A collapsible contents list and scrollable tables support
long entries and small screens without a JavaScript application.

Each study's `RESEARCH_LOG.md` is editable source; run
`.venv/bin/python scripts/render_journals.py` to generate all three pages, or
add `--check` to detect stale output. The older renderer command remains a
compatibility entry point. The notebooks use `site/notebook.css`; detailed
legacy reports retain their original layout and remain linked as archives.

The user explicitly plans to rethink and eventually redo the water and
deep-autoencoder studies under the newer approach, one paper at a time. Their
current results are preserved, not revalidated by this editorial work. Their
new pages say they await reassessment and label reconstructed history. The
water notebook retains early-run provenance caveats, retractions, co-authorship
disclosure and the later limits on its arithmetic/experimental claims.

The poisoning notebook includes the Equation (1) error and tested repair,
F1 and Table V arithmetic, the essential Table IV customer-averaging correction,
the successful baselines and failed SVM, source ambiguities, and the untested
central ensemble. Never treat its old 58-of-68 conditional check as an
unconditional count of erroneous rows.

The user authorized publication to the existing GitHub Pages site after checks.
No experiment or reinterpretation of frozen result bytes is included.

The user wants a research journal: what we first thought, what we checked, what
we observed, and what that observation led us to do next. The technical record
must remain available, but the reader should not need to work through a
paper-style abstract and every diagnostic table to follow the investigation.

## Live-site walkthrough

The following routes were visited through actual browser navigation on
September 20:

| Route | What was checked | Observation |
|---|---|---|
| [Homepage](https://fjoad.github.io/atk-evidence/) | Opening view, study navigation, links | Readable typography, but a long Paper 1 result account precedes the study list. Paper 3 is absent. The page says updated September 2. |
| [Current electricity report](https://fjoad.github.io/atk-evidence/papers/atk-2022-deep-autoencoder/reproduction/) | Clicked from homepage; read contents and body | A paper-like abstract, eleven main sections, four added 7a-7d sections, and many tables. The chronological decisions are embedded in a long formal report. |
| [Performance bound](https://fjoad.github.io/atk-evidence/papers/atk-2022-deep-autoencoder/reproduction/#bound) | Clicked contents link and expanded “The calculation and its assumptions” | Anchor and disclosure work. A useful pattern: explanation first, derivation available underneath. |
| [Earlier electricity notes](https://fjoad.github.io/atk-evidence/papers/atk-2022-deep-autoencoder/) | Clicked report navigation; inspected warning and reading path | Earlier results are visibly labeled, but multiple versions increase the reader's burden. “ELI5” headings and analogies are an uneven fit for the requested tone. |
| [Water study](https://fjoad.github.io/atk-evidence/papers/tlstgt-2025-water/) | Clicked from homepage; read corrections and limitations | Explicit corrections and changed expectations are useful. The opening immediately presents a technical arithmetic search before explaining the investigation's route. |
| Water PDF and GitHub evidence directory | Clicked both links from the water page | PDF URL opened in the browser; GitHub evidence directory rendered. This checks navigation, not every statement or page of the PDF. |

This was a reading/navigation review, not a complete accessibility, mobile,
external-link, or scientific audit of the existing site.

## Editorial direction

- Write in connected, direct prose. Use the model's job and the experiment's
  consequence to introduce technical terms.
- Lead each dated entry with the question or observation that prompted it.
  Explain the current expectation, what was actually done, what happened, and
  why the next step follows. Omit fields that have no real content yet.
- Distinguish observed source problems, interpretations, hypotheses, planned
  tests, completed experiments, and corrections with ordinary words.
- Do not invent experiments, personal reactions, timestamps, or a tidy causal
  history retrospectively. Label reconstructed historical entries as such.
- Keep unrun checks visibly pending. A hypothesis never becomes a result by
  being repeated in multiple pages.
- Append dated corrections and link to the affected entry. Preserve failed
  attempts and favorable results. Do not silently rewrite an earlier expectation
  to make it look prescient.
- Define detection and false alarms in terms of their different denominators.
  Distinguish percentages from percentage points. Explain confidence intervals
  with their sampling unit and scope; avoid implying they cover every possible
  model or measure the probability of fabrication.
- Use a figure or table only when it clarifies a comparison. Limit decorative
  analogy, acronyms, internal evidence labels, hashes, and job identifiers in
  the main narrative; link the detailed records.
- Preserve exact source locators and runnable artifacts underneath the readable
  account. Ordinary words do not require weaker evidence.

## First implementation and future structure

Start with Paper 3. Its Markdown `RESEARCH_LOG.md` is the single editable
journal; a short renderer produces its static website page. Give entries stable
anchors, a compact contents list, dates, and a visible current stage. Add the
study to the index. Keep the wider homepage and older reports available until
their separate rewrite is undertaken.

A later homepage should show the studies and current questions promptly.
Each study should have a readable journal, a compact current evidence summary,
and linked technical records. Old reports and PDFs remain dated archives.
Do not recast old measurements as new experiments during the editorial work.

## Ongoing writing rule

After each material step, update the study journal in the same change as the
code/result and update current status. Record the reason for an experiment
before running it; append its actual outcome afterward. A later change of mind
gets a dated entry explaining which evidence caused it.

The first journal must retain all four fresh Paper 3 questions: poisoning
definition; Equation (1)'s cancellation and plausible BCE repair; the missing
reconstruction objective in joint training; and how ADASYN plus changing
false-alarm rates affects the comparison.
