# First GRU pilot

Recorded 2026-09-23 before GRU research-data preparation, fitting or scoring.
The user approved the research continuation. This is an interpreted numerical
pilot (`I/N`) plus constructed runtime measurement, not full reproduction or
proof that recurrence causes an advantage.

## Question and source

Does a conventional Keras completion of the selected GRU learn on the same
pilot as the previous baselines, and how does poisoning change ranking and
the detection/false-alarm tradeoff? Stop source/setup drift by reusing the
original prepared inputs. Do not select a new subset by performance.

The target was read completely during the source phase; Section III-B.2(e)
and Eq.(1), p.2679, Table II and training settings p.2680, and Table III p.2681
were visually rechecked for this implementation. Reference [26] was acquired
from [arXiv:1809.01774](https://arxiv.org/abs/1809.01774), *Deep Recurrent
Electricity Theft Detection in AMI Networks with Random Tuning of Hyper-parameters*,
Nabil et al., ICPR 2018, DOI 10.1109/ICPR.2018.8545748. All six pages were read
and visually inspected. Local file `papers/references/nabil-2018-deep-recurrent-random-tuning.pdf`,
SHA256 `253568ec334bc653c98a15e942958772ca6d9f93aeea18a6abd3a3afdd749fd4`.
The reference informs missing details; it does not replace the target's selected
architecture, epochs, batch, split, poisoning or dataset population.

| Decision | Source / status | Initial executable reading |
|---|---|---|
| Eight hidden layers, 300 units each | Target III-B.2(e), Table II; explicit | Eight GRU layers, not six hidden plus input/output counted as eight |
| One example is one daily profile | Target II-B and reference III/IV-A/V | 48 time steps, one standardized reading per step; no recurrence across days/customers |
| Return sequences except last layer | Target III-B.2(e); reference IV-A | First seven layers return all 48 states; eighth returns final state |
| ReLU hidden activation | Target Table II; generic formula instead prints tanh | Use ReLU candidate activation and conventional sigmoid gates; prioritize the selected table over the generic recurrence |
| Reset-gate order | Target recurrence and reference Algorithm 1 line 11 | `reset_after=False`, applying reset before recurrent multiplication |
| Softmax output | Target Table II; reference IV-A explicitly says two neurons | Dense(2,softmax), in repository order [benign,attack]; coordinate order is a convention, not imported reversed one-hot labels |
| Intermediate projected Softmax | Generic target/reference recurrence prints V*s and softmax per layer; dimensions not supplied | Ordinary native GRU state outputs between layers, no extra inter-layer Dense/Softmax projection; this is an interpretation, not a literal implementation of every printed equation |
| Dropout .2 | Target Table II; placement absent | GRU input dropout .2 in every layer, recurrent_dropout=0; no additional external dropout |
| Weight constraint 5 | Target Table II; type/coverage absent | MaxNorm(5,axis=0) on every input/recurrent kernel and output Dense kernel; biases unconstrained |
| Optimizer Adam | Target Table II; other settings absent | Adam lr .001, beta1=.9, beta2=.999, epsilon 1e-7; no clipping/decay/schedule |
| Initialization | Target omitted; reference V states GlorotUniform without matrix-level detail | GlorotUniform input/output kernels, ordinary Keras Orthogonal recurrent kernels, zero biases and zero initial states |
| Training | Target III-D.2 pp.2679-2680 | 50 epochs, batch 100, final epoch; no validation/test-selected checkpoint |

The initial route is explicitly `I-GRU-native-table`, with 4,058,702 parameters.
No masking, bidirectionality, normalization layers, residual connections,
class/sample weights, state carryover, augmentation or mixed precision.

### Loss repair and probabilities

Target Eq.(1) repeats log(p), eliminating the label under a binary reading.
Reference Eq.(1), PDF p.4, has the missing log(1-p), supporting the standard repair.
Use standard categorical cross-entropy on two-class one-hot labels. For a
two-unit Softmax, this equals mean-over-the-two-outputs binary cross-entropy
in exact arithmetic; summing both binary terms without that channel mean
would instead double the loss. The averaging convention is explicit here.
This is not fitting the target's defective literal loss.

Primary decisions use argmax in [benign,attack] order, exact ties benign.
Retain raw float32 two-output network probabilities. For existing float64
metric auditing, divide each row by its float64 sum (roundoff normalization,
not a learned calibration) and retain the normalized pair too. Require raw
rows within 2e-6 of unit sum, preserve argmax exactly, and report this convention.
Ranking uses the normalized attack probability; raw predictions remain auditable.

### Open alternatives, not silent fixes

Tanh candidate activation from the generic formula; an explicit inter-layer
V/Softmax projection; `reset_after=True`; external rather than GRU-input
dropout; recurrent dropout; other constraint coverage/initializers/optimizer
defaults remain untested alternatives. Do not mix these into the first pair
or claim that it exhausts the ambiguous text.

The papers do not give an explicit Keras tensor-shape call. A one-step,
48-feature layout remains a separate possible implementation reading; it does
not provide within-day recurrent time steps. It must not silently replace the
48-step, one-reading layout merely because it might run faster.

Reference Table II/V reports failed ReLU configurations in its own search.
That is a reason to inspect gradients and learning, not evidence that the
different target architecture must fail. No result is transferred between papers.

## Software, inputs and pairing

Reuse the verified TensorFlow 2.16.2 / Keras 3.4.1 / h5py 3.11.0 environment in
requirements-feed-forward.txt, including NumPy 1.26.4, SciPy 1.13.1,
sklearn 1.5.2, joblib 1.4.2, threadpoolctl 3.5.0. Local package source and
[official GRU documentation](https://keras.io/api/layers/recurrent_layers/gru/)
confirm the reset convention and that ReLU does not qualify for fast cuDNN
GRU. Explicitly use the backend-native implementation on GPU, not a tanh
substitution to obtain different throughput. No XLA, TF32 or CPU fallback.

The original p00/p30 inputs remain 4,464 training and 2,232 test rows from
20 customers and 28 complete days each. Preserve pre-split ADASYN, feature
scale, attack values, row IDs and the same 675 original attack-label flips
from six customers. Input hashes/metadata must match the existing loader.
Only reshape each row from (48,) to (48,1); do not alter values/order.

Use seed 20260920 before each fresh model; per-layer dropout seeds are
20260920+layer_index. Save initial weights and require their hashes to match
between conditions. Dataset shuffling uses the same seeded TensorFlow rule
as the feed-forward pair: entire training-size buffer, new shuffle each epoch,
batch 100 without dropped remainder, deterministic options, one input thread.
All 50 epochs imply 2,250 updates. Fit on observed, never true, labels.

## Constructed checks and runtime gate

Local work is software-only. Verify full architecture/parameters/activations,
input reshape, deterministic paired weights, hand-calculated reset-before
recurrence, loss/label orientation and simple learning with both true and
reversed constructed labels. Small-width/depth overrides are fixture-only and
cannot enter the research runner. Test finite updates, constraints, partial
history persistence and save/reload. Research CLI admits the full model only.

First run one **10-minute** GPU allocation: one V100-16GB, four CPUs, 16 GiB.
Use NumPy default_rng seed 20260923 for a constructed normal input batch
(100,48,1), alternating binary
labels, the full architecture and 12 training updates: two warmups and ten
timed updates. Time one inference batch too; record initialization, every
duration, finite losses, parameter count, weights/device, versions and GPU
memory. No CER inputs, research fits, parameter alternatives or model selection.
Use one `model.fit` call over a repeated constructed dataset, matching the
research runner's API. Repeated `train_on_batch` calls can add retracing costs
and are not the timing instrument for this gate.

The outcome-independent fit gate is: preflight/fixtures pass, all values finite,
and `2250 * max(last ten update times) <= 720 seconds`. This intentionally
uses the slowest warmed step, not the best, as a conservative pilot-cost guide.
It is not a full-data time estimate or an exact runtime guarantee. If it fails,
stop before research data and report; do not silently change ReLU or reduce depth.

If it passes, the same frozen code may run exactly the original p00/p30 pair
inside one **40-minute** allocation on the same GPU type, four CPUs/16 GiB.
Each fit has a 15-minute batch-boundary guard, with partial weights/history
preserved and never called a 50-epoch completion. Remaining allocation covers
startup, fixture checks, scoring, persistence and audits. Outer Slurm limit is
the hard stop. Combined maximum allocation is 50 GPU-minutes for timing+pair.
No data outcome may alter the gate, seed, configuration or these budgets.

## Metrics, records and decision

Table III p00/p30 context (%): DR 92.4/78.5, FA 6.8/20.6, SP 93.2/79.4,
PR 92.3/79.0, ACC 92.8/78.9, F1 92.3/78.7, AUC 92.1/79.4.
These are full-population context, not matched-sample reproduction criteria.

Save all seven metrics/counts, per-attack and original/synthetic breakdowns,
constant-prior and daily-mean controls, all-cutoff/reversal diagnostics at
FA caps 6.8/20.6% and 17.6/33.3%. Test-chosen cutoffs are diagnostic bounds on
saved scores, not validated calibration. No confidence interval from this
single dependent customer-day pilot or claim of temporal-mechanism identification.

Save each epoch's loss/accuracy/time/update count, partial status if applicable,
model/optimizer config, initial/final weights, kernel norms, actual GPU/runtime,
fit/scoring/reload times, raw/normalized probabilities and hashes. Reload must
give exactly the same raw outputs on the same device/batches. All old source
hashes remain bound to recorded immutable Git revisions as direct files grow.

Competing outcomes: useful learning with a default-cutoff gap versus weak
ranking even at favorable cutoffs; operational/optimization failure versus a
completed negative result. Check artifacts/labels/updates/loss first. A large
mismatch does not trigger more seeds. After the pair, audit and record the next
named question before proceeding to AEA/ARIMA or ensemble work. Do not edit,
regenerate or publish the website in this research session.
