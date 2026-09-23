# Electricity theft with deep autoencoders

Updated: 2026-09-23

Notes on *Deep Autoencoder-Based Anomaly Detection of Electricity Theft
Cyberattacks in Smart Grids*, Takiddin, Ismail, Zafar and Serpedin,
IEEE Systems Journal (2022). [Publication record](https://doi.org/10.1109/JSYST.2021.3136683).

This account follows the earlier work preserved in the linked findings.
It reorganizes existing evidence rather than describing new experiments.
The full technical report and older accounts remain available below.

**Earlier investigation, awaiting reassessment.** These results were obtained
under an earlier workflow. We are preserving them, not certifying them again
under the current approach. A fresh reassessment is separate future work;
the present active investigation is the data-poisoning paper.

## What the paper claims

The paper trains autoencoders on normal electricity use. An autoencoder tries
to reproduce its input. The proposed detection rule treats a sufficiently large
reconstruction error as evidence of theft. The study compares fully connected,
recurrent and attention-based variants on the named SGCC and Irish ISET datasets.

For the simplest fully connected model, FC-SAE, the ISET table reports 81%
detection at 15% false alarms and 83% balanced accuracy. The recurrent LSTM-SAE
row reports 85% detection at 13% false alarms. These are specific targets,
not interchangeable descriptions of a generally successful model.

## Starting hypothesis

Our working expectation was that the reported performance might not follow
from reasonable implementations of the written method. We needed to test
that expectation while keeping source omissions, repairs and controls visible.
A failed fit would not establish that every implementation must fail.

Three questions gradually separated: do the numbers reproduce, does the
architecture supply the claimed useful behavior, and can this fixed setup
reach the reported target at all? They need different evidence.

## 11 August — An earlier attempt left a preparation question open

The earlier seed-11 experiment detected 26.18% of attacks with 58.22% false
alarms. However, it omitted the paper's test-set ADASYN operation. It was
therefore not a complete reconstruction of that evaluation.

Adding generated benign test examples cannot change an already-fitted model's
detection on unchanged attack examples. That observation limited what the
omitted step could repair for those scores, but it did not exclude a different
trained model. We retained the result and returned to a more explicit source
reconstruction rather than treating the first run as the final verdict.

The [earlier method notes](../../site/papers/atk-2022-deep-autoencoder/earlier-notes.html)
preserve that route, including assumptions, unsuccessful attempts and its
conditional calculations. Do not combine those numbers with later runs as if
they were repetitions of one fixed experiment.

## 30–31 August — Completing a declared FC-SAE reproduction

The clean-reader experiment completed with the named Irish consumption data,
declared source interpretations, one seed, and the printed-position test-set
resampling. It evaluated 8,884,989 rows, including generated benign examples.
The exact population and preprocessing choices are in the
[frozen specification](CLEAN_READER_SPECIFICATION.md).

| Measure | Paper | This run |
|---|---:|---:|
| Detection | 81.00% | 25.48% |
| False alarms | 15.00% | 45.13% |
| Specificity | 85.00% | 54.87% |
| Precision | 81.00% | 36.73% |
| Balanced accuracy | 83.00% | 40.18% |
| F1 | 81.00% | 30.09% |
| Ranking AUC | 81.00% | 39.40% |

This was a substantial mismatch, not a close reproduction. Inputs, saved
scores, confusion counts and all seven metrics passed the independent audit.
But the result still concerned one declared completion and seed, not every
model in the paper. The [initial finding](CLEAN_READER_FINDING.md) records both
the gap and that boundary.

The next useful question was not simply whether another seed would help.
The output layer and score impose restrictions that can be examined without
training again.

## 31 August — A conditional limit stronger than a failed seed

Softmax forces each reconstruction to have nonnegative entries that sum to one.
The prepared consumption values do not generally have that form. We asked:
even if every example could receive its most favorable allowed reconstruction,
could a shared reconstruction-error cutoff reach the published target?

The deliberately optimistic calculation used the true answers to choose the
best allowed outputs. On the fixed prepared evaluation, its upper limits,
rounded upward, are **50.93% balanced accuracy** and **9.25% detection at no
more than 15% false alarms**, versus 83% and 81% reported.

This is an all-weights restriction under the fixed prepared inputs, Softmax
output and MSE score. Changing weights, width or seeds cannot escape that
particular bound. Changing the preparation, output domain or score is outside
it. The evaluation of the analytic bound uses padded floating-point arithmetic,
not certified interval arithmetic. See the [follow-up finding](POST_ANCHOR_FINDING.md).

There was counterevidence to a simpler accusation: the trained score was not
doing nothing. It improved original-row balanced accuracy over zero
reconstruction by about 0.89 percentage points, and retained useful ranking
differences within similar-input-magnitude groups. A small useful contribution
can coexist with a large structural shortfall. The experiment did not establish
the claimed recurrent or attention mechanism.

## 31 August — Testing what the bound depends on

Two additional source-supported scaling readings still capped detection at
the printed cutoff at 29.81% and 33.96%. They did not exhaust every reasonable
normalization. Replacing Softmax with a Sigmoid output removed the all-cutoff
exclusion on the complete evaluation: that alternative remained mathematically
open, rather than becoming an achieved reproduction.

We then trained a small matched Softmax/Sigmoid pair for ten epochs. Sigmoid's
best detection at FA<=15% was **9.75%**, or **25.39%** after favorable score
reversal, against the 81% target. Every cutoff failed for those fitted models
and sampled rows. Its calibration loss was **still improving**; this is not a
universal Sigmoid impossibility proof or an established long-run plateau.

The distinction matters: a permissive bound only means an outcome has not
been excluded; a failed small fit only excludes those particular fitted scores.
The [scaling finding](SOURCE_ASSUMPTION_FINDING.md),
[complete Sigmoid range check](SIGMOID_SANITY_FINDING.md), and
[paired-fit finding](SIGMOID_FIT_FINDING.md) preserve those separate steps.

## 2 September — Giving LSTM-SAE the paper's reported training time

The paper reports 183 minutes for full-ISET LSTM-SAE training without stating
its hardware or epoch count. A separate declared implementation received that
fitting budget on one V100-16GB. This paper-time LSTM-SAE run scored all
8,884,989 prepared evaluation examples and passed its artifact audit.

| Measure | Paper | Run inside 183 minutes |
|---|---:|---:|
| Detection | 85.00% | 16.62% |
| False alarms | 13.00% | 31.91% |
| Balanced accuracy | 86.00% | 42.35% |
| Ranking AUC | 82.00% | 40.30% |

At the reported 13% false-alarm limit, the best saved-score detection was
7.00% in the printed direction and **23.02%** after reversal. No cutoff
rescued that fitted model. It completed about 1.268 epochs; measured throughput
projects ten epochs to **23.98 hours**, but the paper does not claim ten epochs.
Loss was still decreasing. This is a bounded failure inside the reported time,
not an unlimited-time impossibility proof.

The [paper-time finding](PAPER_TIME_BUDGET_FINDING.md) includes all seven metrics,
the clock boundary, hardware, assumptions, saved-score checks and limitations.

## Records and unresolved coverage

- [Current technical reproduction report, figures and all comparisons](../../site/papers/atk-2022-deep-autoencoder/reproduction/index.html).
- [Earlier FC-SAE scientific report (PDF)](../../site/reports/atk-2022-deep-autoencoder.pdf); the web report also contains the later LSTM-SAE result.
- [Earlier account and method notes](../../site/papers/atk-2022-deep-autoencoder/earlier-notes.html).
- [All study results and execution records](results/).

The remaining architectures, source interpretations, SGCC model experiments,
and repeated-run uncertainty are not completed by these checks. Restricted
datasets and source PDFs are not redistributed here. No new experiment ran
for this notebook reorganization.

## Current conclusion

The declared FC-SAE and time-bounded LSTM-SAE implementations did not reproduce
their targeted ISET results. The FC-SAE follow-up establishes a stronger
conditional limit for fixed inputs, Softmax outputs and MSE scoring. The
Sigmoid alternative remains open beyond the tested small pair, and unlimited
LSTM training is not excluded by a finite-time run.

These results do not establish that every autoencoder must fail, that training
learned nothing, or how the authors produced their numbers. Further work must
name a remaining source or mechanism question; the completed runs are not an
invitation to repeat seeds until a preferred answer appears.
