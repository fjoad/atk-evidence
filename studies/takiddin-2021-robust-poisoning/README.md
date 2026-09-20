# Study 3: robust electricity-theft detection under label poisoning

This study audits:

> A. Takiddin, M. Ismail, U. Zafar, and E. Serpedin, “Robust Electricity
> Theft Detection Against Data Poisoning Attacks in Smart Grids,” *IEEE
> Transactions on Smart Grid*, 12(3), 2675–2684, 2021.

DOI: `10.1109/TSG.2020.3047864`

## Current state

The [research journal](RESEARCH_LOG.md) now records the route through the
investigation in chronological entries: observations, current explanations,
planned checks, outcomes, and corrections. It is also the editable source for
the draft website journal. The [code search](CODE_AVAILABILITY.md) found no
matching public implementation in the checked sources on September 20;
publisher supplement access remains unverified.

The user requested a fresh all-model reproduction on September 20. The
[current plan](../../docs/plans/2026-09-20-robust-all-model-reproduction.md)
covers all seven baselines, both ensembles, all poison levels, and generalized
and customer-specific results. Correct data preparation and ordinary library
implementations with the reported settings are the first priority. The new
phase has reached verified source acquisition and preparation implementation.
The bounded preparation check completed on Panther as job 397206 at frozen
commit 30ce6c4: eight cases passed in 2:16, and all 224 saved arrays passed the
artifact audit. See the [result](results/preparation_20260920/README.md).
The first 100-tree random-forest pair has now completed too (job 397217):
92.31% detection/3.02% false alarms without poisoning and 61.18%/0.44% at
nominal 30% poisoning. Ranking remains strong, and cutoff diagnostics recover
much of the detection decline. This is a small pilot with the documented
preparation dependence, not full Table III reproduction. See the
[first-baseline record](results/rf_pilot_20260920/README.md).

The matched preparation control also completed (job 398164, four new fits).
On identical original test rows, training-only ADASYN lowers AUC by 6.47/8.99
points at 0%/30% poisoning. Whole-source-day grouping produces no further
collapse: AUC remains 93.18/82.97. This supports a bounded preparation-policy
effect while preserving evidence that the baseline works. See the
[matched-control record](results/split_control_20260920/README.md).

The first AdaBoost pair is also complete and audited (job 398348, frozen
d47a6de). The historical SAMME.R completion gives DR/FA/AUC 81.09/15.17/90.71
at p00 and 46.43/5.06/83.74 at p30 on the original pilot. At the paper's
29.9% FA cap, the poisoned saved scores can reach 80.45% detection, versus
70.1% printed. This is a cutoff diagnostic, not full-data reproduction or
validated calibration. See the [AdaBoost record](results/adaboost_pilot_20260920/README.md).

The initial sigmoid SVM pair is complete too (September 21, job 398709,
frozen e698173). AUC is 65.64/63.38 at p00/p30. At the paper's respective
false-alarm caps, no cutoff or reversal reaches its detection targets:
best DR 19.37/33.48 versus 89.2/73.7. Both fits report solver success and
pass artifact checks, but their training accuracy is weak too. The next
question is a bounded read-only kernel/solution diagnostic, not extra seeds.
See the [SVM record](results/svm_pilot_20260921/README.md).

The historical source-only audit completed on September 2. The paper was
identified, fingerprinted, and visually inspected in full; the method, causal
claims, and Tables II–V are frozen. A preregistered audit found that three
sequential-ensemble rows cannot reconcile DR, FA, PR, and ACC at any class
prevalence within one-decimal rounding. This is a source-level internal
inconsistency, not a trained non-reproduction or an inference about intent.
No dataset or model execution occurred in that historical phase.

The study is independent of the earlier electricity-theft audit. Shared
authors, data, attacks, or terminology may motivate checks but cannot transfer
a result or verdict.

## Read next

- [`RESEARCH_LOG.md`](RESEARCH_LOG.md): the readable, dated investigation.
- [`CODE_AVAILABILITY.md`](CODE_AVAILABILITY.md): reproducible code search and access limits.
- [`PREPARATION.md`](PREPARATION.md): the initial executable data choices and
  bounded cluster check.
- [`DATA_SOURCES.md`](DATA_SOURCES.md): fresh source identity and allocation checks.
- [`FIRST_BASELINE.md`](FIRST_BASELINE.md): frozen forest, metric, and diagnostic choices.
- [`SPLIT_RESAMPLING_CHECK.md`](SPLIT_RESAMPLING_CHECK.md): frozen matched controls and stopping rule.
- [`ADABOOST_PILOT.md`](ADABOOST_PILOT.md): historical algorithm completion, fixed pair, and verification.
- [`SVM_PILOT.md`](SVM_PILOT.md): printed sigmoid kernel, omitted-setting completion, and raw scores.
- [`METHOD.md`](METHOD.md): paper-derived experiment and causal specification.
- [`SOURCE_AUDIT_CONTRACT.md`](SOURCE_AUDIT_CONTRACT.md): checks frozen before
  the printed values are analyzed.
- [`SOURCE_AUDIT_FINDING.md`](SOURCE_AUDIT_FINDING.md): source-only arithmetic
  result, limitations, and compute boundary.
- [`EXPLANATION_REGISTER.md`](EXPLANATION_REGISTER.md): live competing
  explanations and discriminating evidence.
- [Current all-model plan](../../docs/plans/2026-09-20-robust-all-model-reproduction.md): revised scope, source issues, and execution sequence.
- [Historical source-audit plan](../../docs/plans/2026-09-02-robust-poisoning-source-audit.md): completed source-only authorization and provenance.

## Experimental boundary

The paper states 50 epochs, batch size 100, an NVIDIA GeForce RTX 2070, and
approximate training times of one hour for shallow detectors, 1.5–3 hours for
deep detectors, three hours for ensemble averaging, and four hours for the
sequential ensemble. A later numerical attempt must treat those statements as
part of the target rather than escaping a mismatch with newer or additional
hardware.

The September 20 plan replaces the earlier source-only stopping point as the
research direction. It prioritizes explicit setup choices, measured costs,
and finite execution budgets before training.

## Keep the journal current

Write each material question or experiment's rationale before running it and
append its outcome afterward. Record a change of mind as a dated correction
that links to the affected entry. Distinguish reconstructed historical entries
from contemporaneous notes. Preserve successful matches and failed attempts.

The journal targets an undergraduate taking introductory statistics. Use
ordinary definitions and concrete consequences; link detailed configurations,
data identities, code, and uncertainty calculations. See the
[website brief](../../docs/WEBSITE-JOURNAL-BRIEF.md).

The static page is generated from the journal with the Markdown parser already
pinned in the repository environment:

```bash
.venv/bin/python scripts/render_robust_journal.py
.venv/bin/python scripts/render_robust_journal.py --check
```

Edit the Markdown source, including its `Updated:` date, rather than the
generated HTML. The page is a local draft until a site deployment is requested.

## Preparation commands

Verify existing inputs against official metadata and archive integrity:

```bash
.venv/bin/python studies/takiddin-2021-robust-poisoning/reproduction/download_data.py --online --check-crc
```

Run hand-checkable software fixtures locally:

```bash
.venv/bin/python -m unittest tests.test_robust_preparation -v
```

Real-data preparation refuses to run outside a Slurm allocation. The short
`reproduction/run_preparation_check.sbatch` wrapper takes the checkout,
revision, raw-data directory, Python environment, and new output directory as
explicit environment inputs. It prepares 20 customers and 28 complete days
each at 0% and 30% poisoning on both paths, including one customer-specific
example. Prepared arrays stay in ignored data directories. Existing output
directories are never overwritten.
