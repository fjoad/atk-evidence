# Robust electricity-theft detection under label poisoning

This study audits:

> A. Takiddin, M. Ismail, U. Zafar, and E. Serpedin, “Robust Electricity
> Theft Detection Against Data Poisoning Attacks in Smart Grids,” *IEEE
> Transactions on Smart Grid*, 12(3), 2675–2684, 2021.

DOI: `10.1109/TSG.2020.3047864`

## Current state

The latest completed step verifies basic learning by an equation-led tanh
alternative to the sequential ensemble. Both full-width, 48-step constructed
cases finish 300 updates at 100% held-out accuracy with clipped binary
cross-entropy about 1e-7. The matched ReLU normal-label reference becomes
nonfinite at update 245; its reversed-label case remains unrun under the
predeclared stop rule. The raw comparison remains failed/incomplete, and
fresh-process GPU reload verifies both completed tanh models exactly. See the
[complete result and preserved failures](results/sequential_cells_20261002/README.md).

The synthetic task is also solved by the sign of each profile's mean. Passing
establishes basic learning, not a temporal capability, theft detection, or
poisoning robustness. Algorithm 1's explicit tanh formulas conflict with
Section IV-C's ReLU selection; the alternative preserves that uncertainty
and other declared native-cell completions. No tanh CER research pair has
run, and the existing research CLI still selects ReLU by default.

Research execution is paused while the public account is brought up to date.
The next scientific decision is to specify and wire a fresh-weight
p00/p30 tanh CER pilot and require a new full-batch runtime gate before any
launch. Synthetic trained weights must not be reused. See the
[current plan](../../docs/plans/2026-09-20-robust-all-model-reproduction.md)
and [research status](../../docs/STATUS.md).

## Completed research pairs

Six detector pairs have completed on the documented 20-customer pilot.
Here p00 and p30 denote nominal 0% and 30% training-label poisoning. Results
use the declared preparation, including pre-split synthesis and dependent
customer-day rows; they are not full-population or full-paper reproductions.
All values below are percentages, with p00 followed by p30.

| Detector | Bounded result | Record |
|---|---|---|
| Random forest | Useful ranking survives poisoning; cutoff diagnostics recover much of the default detection decline. | [Original pair](results/rf_pilot_20260920/README.md) |
| AdaBoost | AUC 90.71 / 83.74; the poisoned scores reach 80.45 detection within the paper's 29.9 false-alarm cap. | [Pair](results/adaboost_pilot_20260920/README.md) |
| Sigmoid SVM | AUC 65.64 / 63.38; no cutoff or score reversal recovers the corresponding printed detection/false-alarm corners. | [Pair](results/svm_pilot_20260921/README.md) |
| Feed-forward | AUC 96.35 / 90.89; within the respective paper false-alarm caps, best detection meets the p00 target to rounding and exceeds the p30 target. | [Pair](results/feed_forward_pilot_20260922/README.md) |
| GRU | Both 50-epoch fits complete; AUC 89.07 / 79.89. The poisoned AUC is close to the reported 79.4, but the joint detection/false-alarm targets are missed. | [Completed pair and preserved partial attempt](results/gru_completion_20260924/README.md) |
| Sequential AEA–GRU–feed-forward | Both 50-epoch fits complete, but their constant scores give AUC 50 / 50; no cutoff or reversal rescues these fitted models. This uses the scalar-Sigmoid bridge and ReLU recurrent cells. | [Pair](results/sequential_pilot_20261001/README.md) |

Cutoff checks describe what is possible with the saved scores; they are not
independently validated calibration. The useful baselines and failed
sequential result both remain part of the evidence. No result establishes a
failure of every implementation or an undocumented author procedure.

## What the follow-up checks changed

- [Matched preparation controls](results/split_control_20260920/README.md)
  show a bounded resampling-policy effect while useful ranking survives.
  The [poisoning-order control](results/poison_balance_20260923/README.md)
  does not rescue the result by poisoning before training-only ADASYN.
- [Independent SVM replay](results/svm_replay_20260921/README.md) reproduces
  native scores and finds large negative kernel directions. It does not
  establish the cause of the performance gap or rule out other settings.
- [AEA geometry](results/aea_geometry_20260924/README.md) excludes the
  printed false-alarm points under the fixed prepared inputs, bounded output,
  MSE score, and 0.51 cutoff. The [repair envelope](results/aea_repair_20260925/README.md)
  leaves some alternatives unexcluded with a free cutoff; this is not a
  successful model. The [recheck](results/aea_recheck_20260925/README.md)
  withdraws the unsupported MAE promotion and corrects earlier reporting.
- [Saved-state observations](results/sequential_saved_state_20261002/README.md)
  and [gate arithmetic](results/sequential_gates_20261002/README.md) explain
  attenuation and amplification in the saved sequential models without any
  fitting. They concern fixed-weight forward behavior, not training onset.
  The subsequent [tanh learning check](results/sequential_cells_20261002/README.md)
  validates the synthetic alternative without claiming a research-data repair.

The [source-only audit](SOURCE_AUDIT_FINDING.md) remains separate: three
printed sequential rows cannot reconcile all their metrics within the
rounding allowance at any class prevalence under a single-confusion-matrix
interpretation. Separately averaged metrics require different reasoning.
That does not exclude achieving
the detection/false-alarm pair alone and does not establish author intent.

ARIMA, trained standalone AEA, ensemble averaging, broader poisoning levels,
customer-specific coverage, full-population reproduction, and the claimed
architectural mechanism remain unfinished. The study is independent of the
older autoencoder and water-network work; no finding transfers between papers.

## Read next

- [Latest full-length learning result](results/sequential_cells_20261002/README.md): completed tanh cases, incomplete ReLU reference, and audit.
- [`GRU_COMPLETION.md`](GRU_COMPLETION.md): same-seed completion budget and preservation requirements.
- [`AEA_SPECIFICATION.md`](AEA_SPECIFICATION.md): source reconstruction and range/loss ambiguities.
- [`SEQUENTIAL_ENSEMBLE_PILOT.md`](SEQUENTIAL_ENSEMBLE_PILOT.md), [`SEQUENTIAL_INTERFACE.md`](SEQUENTIAL_INTERFACE.md), and [`SEQUENTIAL_RECURRENT_CONTROL.md`](SEQUENTIAL_RECURRENT_CONTROL.md): the distinct sequential interpretations and controls.
- [`RESEARCH_LOG.md`](RESEARCH_LOG.md): the continuing investigation through the latest constructed learning check, including corrections and remaining questions.
- [`CODE_AVAILABILITY.md`](CODE_AVAILABILITY.md): reproducible code search and access limits.
- [`PREPARATION.md`](PREPARATION.md): the initial executable data choices and
  bounded cluster check.
- [`DATA_SOURCES.md`](DATA_SOURCES.md): fresh source identity and allocation checks.
- [`FIRST_BASELINE.md`](FIRST_BASELINE.md): frozen forest, metric, and diagnostic choices.
- [`SPLIT_RESAMPLING_CHECK.md`](SPLIT_RESAMPLING_CHECK.md): frozen matched controls and stopping rule.
- [`ADABOOST_PILOT.md`](ADABOOST_PILOT.md): historical algorithm completion, fixed pair, and verification.
- [`SVM_PILOT.md`](SVM_PILOT.md): printed sigmoid kernel, omitted-setting completion, and raw scores.
- [`SVM_REPLAY.md`](SVM_REPLAY.md): frozen independent replay and constrained kernel checks.
- [`FEED_FORWARD_PILOT.md`](FEED_FORWARD_PILOT.md): neural architecture, loss repair, defaults and finite budget.
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

## Website journal

The [website journal](https://fjoad.github.io/atk-evidence/papers/takiddin-2021-robust-poisoning/)
is generated from [RESEARCH_LOG.md](RESEARCH_LOG.md). It follows the completed
research pairs, AEA bounds and corrections, sequential diagnostics and latest
constructed learning check. The linked result records preserve full technical
details. This continuing account is not a final scientific report or a new
validation of the existing experiments.

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
