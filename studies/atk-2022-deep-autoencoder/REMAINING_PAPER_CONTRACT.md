# Remaining-paper clean-reader contract

**Drafted:** 2026-09-01

**Status:** exact scientific checkpoint; no experiment in this document has
run. Freeze code and execute only after the choices and budgets in Section 12
are accepted without outcome-dependent changes.

**Evidence questions:** numerical reproduction (`N`), mechanism (`M`), and
attainability (`A`) remain separate.

## 1. Why these experiments are promoted

The eligible clean-reader path currently covers one full Table-III row:
ISET FC-SAE. It does not cover LSTM-SAE, FC-VAE, LSTM-VAE, or LSTM-AEA. Those
four rows are necessary to test the paper's reported ordering and therefore
are promoted to numerical depth. A matched FC/LSTM experiment is separately
promoted because the paper attributes its LSTM advantage to temporal
correlation. Tables IV and V are promoted only after eligible full-data models
exist. Table II retains its literal 1,034-to-48 failure and receives a finite
repair family last.

The primary source is the 12-page paper with SHA-256
`f3098e0c27ee19b27bea026aedc3d10e5dbb0c46f5cd01ed5bd5c05b7dcf850f`.
The consequential locations are:

- LSTM-SAE: PDF pp. 5-6, printed pp. 4110-4111, Algorithm 2;
- VAE: PDF pp. 5-7, printed pp. 4110-4112, Equations (8)-(10), Algorithms 3-4;
- LSTM-AEA: PDF pp. 7-9, printed pp. 4112-4114, Figures 5-6,
  Equations (11)-(13), and Algorithm 5;
- common training, scoring, and search: PDF pp. 9-10, printed pp. 4114-4115,
  Algorithm 6, Figure 7, and Table I; and
- numerical targets and explanations: PDF pp. 10-11, printed pp. 4115-4116,
  Tables II-V and their surrounding text.

## 2. Shared ISET data and evaluation identity

Do not prepare a new primary dataset. Reuse the audited cache whose metadata
and arrays produced `CR-ISET-FCSAE-01`:

- source branch `sciencedb-csv-semantic-equivalence-v1`;
- all 4,225 residential meters and strict 48-slot customer-days;
- root seed `20260824`;
- duration-first, in-day Attack 3;
- joint feature-wise `B+M` scaling before the split;
- meter-disjoint 2:1 `B1/B2` split;
- attacks derived only from held-out `B2` meters; and
- printed-position, five-neighbor, exact ADASYN on the test set.

The preserved counts are 1,500,523 benign training profiles and 8,884,989
post-ADASYN test profiles: 4,380,387 benign and 4,504,602 malicious. Every
Table-III model receives identical training and test row identities. Every
eligible score retains meter, day, attack, original/synthetic, and ADASYN
source identity. A cache checksum mismatch stops the run.

The earlier seed-11 all-model breadth map is not eligible because its data,
seed, split, Attack-3, test-resampling, batch, and stopping choices differ.
It may estimate operational cost only; it cannot supply another seed or select
a completion.

## 3. Shared fitting and result contract

The paper says `while not converged` and omits batch, epoch cap, convergence,
seed, and learning rate. Preserve the already approved ordinary clean-reader
completion for every new primary anchor:

- seed `20260824`, with separate deterministic streams for initialization,
  shuffling, latent draws, and scoring draws;
- batch 32, seeded reshuffling each epoch;
- maximum 100 epochs, minimum 10;
- stop after five consecutive epochs without training-objective improvement
  of at least `1e-6`;
- restore the lowest training-objective weights;
- no hidden validation set; and
- Keras optimizer defaults: Adam `1e-3` and SGD `1e-2`.

Each anchor is one immutable attempt. Save every epoch objective, best/final
epoch, weights, configuration, layer inventory, optimizer state where
serializable, warnings, failures, wall time, device, package versions, peak
memory, row-aligned scores, predictions, complete metric vector, and artifact
hashes. Eligible scores must be regenerated after loading the saved weights in
a fresh process. A timeout, OOM, nonfinite objective, serialization failure, or
unsupported deterministic operation is an outcome, not permission for an
unrecorded retry.

At the printed cutoff and at every distinct saved-score boundary, calculate
DR, FA, SP, PR, balanced ACC, F1, and AUC in both score directions. The printed
direction is the numerical reproduction. Reversal is a labeled diagnostic.
Compute the smallest complete seven-metric gap to the reported row. Uncertainty
over test rows uses held-out meters as clusters; ADASYN descendants never count
as independent customers. One seed remains descriptive, not a seed-population
confidence statement.

## 4. Primary Table-III model completions

### 4.1 LSTM-SAE: `CR-ISET-LSTMSAE-01`

Source-fixed fields are encoder widths `(500,300)`, decoder widths `(300,500)`,
hidden Sigmoid, output Sigmoid, input dropout 0.2, Adam, MSE, threshold 0.61,
and high reconstruction error as anomaly.

Algorithm 2 supplies the terminal encoder hidden/cell state to the decoder and
says the encoder output is the decoder input at the initial time step, but it
does not state later decoder inputs. The primary completion therefore uses:

- input layout 48 time steps by one feature;
- the existing 300-wide terminal encoder state as the bottleneck, with no
  extra latent projection;
- sampled input only at decoder step one and zeros for steps 2-48;
- the top encoder hidden/cell state initializes only the first decoder layer;
  and
- no teacher forcing or access to the target during reconstruction.

This is the smallest direct reading of Algorithm 2. Repeat-latent,
autoregressive-feedback, mirrored-all-layer state transfer, and a distinct
latent projection remain named alternatives; none may replace this anchor
after its outcome.

### 4.2 FC-VAE: `CR-ISET-FCVAE-01`

Source-fixed fields are encoder widths `(500,400,300,100)`, the full mirror
`(100,300,400,500)`, ReLU hidden activation, Softmax reconstructed mean,
dropout 0.4 after every hidden layer, Adam, threshold 0.43, and a Gaussian
latent distribution.

The primary completion uses 100-dimensional linear `z_mean` and `z_log_var`
heads after the full printed encoder, one reparameterized latent draw during
each training pass, and the per-example Equation-(10) objective

`sum((x - reconstruction)^2 over 48) + sum(analytic KL over latent units)`,

averaged over the minibatch. `exp(0.5*z_log_var)` supplies a positive standard
deviation. This is a standard executable completion of the omitted positivity
operation; it is not claimed to be printed code.

### 4.3 LSTM-VAE: `CR-ISET-LSTMVAE-01`

Source-fixed fields are encoder widths `(400,300)`, decoder widths `(300,400)`,
Tanh hidden activation, Sigmoid reconstructed mean, no dropout, SGD, threshold
0.47, and a Gaussian latent distribution. Use the same Equation-(10) reduction
and log-variance parameterization as FC-VAE, with a 300-dimensional latent.
Use the LSTM-SAE primary decoder completion: 48-by-1 input, sampled latent at
step one followed by zeros, and top-state-only initialization.

### 4.4 LSTM-AEA: `CR-ISET-LSTMAEA-01`

Source-fixed fields are encoder widths `(500,300,200)`, decoder widths
`(200,300,500)`, hidden and output Sigmoid, no dropout, SGD, MSE, threshold
0.51, and high error as anomaly.

Algorithm 5 refers to a reconstruction before the decoder has produced one
and prints a sum where the prose explicitly says concatenation. The primary
completion is decoder-conditioned additive attention over all encoder time
steps, using the previous decoder hidden state as query; the attention context
is concatenated with the previous scalar reconstruction; the decoder is
autoregressive for 48 steps; and only the top encoder hidden/cell state
initializes the first decoder layer. The first previous reconstruction is a
learned projection of the terminal encoder state. No distinct latent layer is
added because Algorithm 5 uses the attention context directly.

Repeat-latent with concatenation and the dimension-repaired literal-sum branch
remain separate alternatives. The primary is selected because it is the only
one that simultaneously supplies the printed previous-decoder query and the
prose's explicit concatenation without using future targets.

## 5. VAE probability failure and finite score family

The literal VAE score is not executable. Section III-B says low reconstruction
probability is anomalous and averages a number of decoded samples. Section
III-C says a score greater than the threshold is anomalous. The decoder is said
to recover distribution parameters, but Table I provides one reconstruction
output activation and Algorithms 3-4 provide no decoder variance head,
variance floor, likelihood reduction, or sample count.

Do not let one score silently stand for all of those claims. From each primary
VAE fit, preserve these predeclared scores:

1. `kernel_mean_mc10`: average ten values of
   `exp(-0.5 * mean_squared_error)` over deterministic scoring streams;
2. `kernel_sum_mc10`: average ten values of
   `exp(-0.5 * sum_squared_error)`;
3. deterministic latent-mean reconstruction MSE; and
4. deterministic latent-mean reconstruction MSE plus analytic KL.

The first score is the primary threshold-compatible completion and is tested
with low probability as anomaly at 0.43/0.47. Both probability scores are also
tested in the contradictory high-probability direction. MSE and MSE+KL are
all-cutoff source-supported diagnostics and do not inherit the probability
cutoff.

On a predeclared 12,119-row held-out sample, also compute 1-, 10-, and 100-draw
versions. This checks Monte Carlo stability; it does not choose the version
nearest the paper. The full result remains ten draws regardless. A properly
normalized 48-dimensional fixed-unit Gaussian joint density has maximum
`(2*pi)^(-24)`, far below 0.43 and 0.47; record that static incompatibility
without spending a full scoring pass.

The prose-consistent learned-decoder-variance model is a materially different
fit, not a score switch. Before any full version, run one capped 2,048-fit-row,
1,024-calibration-row, ten-epoch check for each VAE architecture with a linear
decoder log-variance head and Gaussian NLL plus KL. It is promoted only if all
losses and variances are finite, both reconstruction and KL terms update, and
the prescribed cutoff is nondegenerate on calibration. Promotion is based on
instrument validity, never proximity to the reported target.

## 6. Cheap feasibility wave before full anchors

No full anchor starts directly. Each model first gets one cluster-only pilot on
the exact cache and code, using deterministic nested row subsets. The pilot
checks shapes, parameter count, finite forward/backward updates, saved-weight
reload, score repeatability, peak memory, and throughput. It is operational
`X`, not numerical evidence.

- one GPU, 16 CPUs, at most 96 GiB RAM;
- at most two hours per model on `gpu-short`;
- 32,768 training rows, 12,119 held-out scoring rows, two epochs;
- batch 32 for fitting;
- inference batches tried in descending order from `{256,128,64,32}` only to
  find the largest memory-safe partition; and
- scores from two safe inference batches must agree within `1e-6` absolute.

Use measured fit steps and score rows per second to project both the minimum
ten-epoch run and the worst-case 100-epoch run. A full anchor is eligible only
if the conservative projection fits within 72 hours on one accessible Panther
GPU and the pilot uses at most 75% of allocated memory. Otherwise preserve an
operational feasibility outcome and return to discussion; do not switch batch,
architecture, decoder, optimizer, data size, or GPU count to force completion.

Historical lower-fidelity timings are context, not projections: FC-VAE 5.3
minutes, LSTM-SAE 6.38 hours, LSTM-VAE about 4.5 hours including recovered
scoring, and LSTM-AEA 43.66 hours, all at batch 512 and without the current
prepared/evaluation contract. The current FC-SAE batch-32 fit took 8:02:46.
Therefore no honest remaining-model ETA exists until the pilots. Pilot exposure
is at most eight GPU-hours; the absolute four-anchor ceiling is 288 GPU-hours,
and the promotion gate should make the realized cost substantially smaller.

## 7. Table III order and seed promotion

After all feasible primary anchors are audited, compare the complete rows and
the published order FC-SAE, LSTM-SAE, FC-VAE, LSTM-VAE, LSTM-AEA. A family
ranking is not a mechanism finding because Table I changes architecture,
activation, dropout, output domain, optimizer, objective, and score together.

Do not automatically add seeds. Softmax-output targets already closed by a
fixed-input global bound need no seed sweep inside that same system. For a
Sigmoid/recurrent model, promote seeds `20260825` and `20260826` only if the
primary fit is instrument-valid, finishes within budget, and its learning or
score envelope leaves a credible ordinary route to the target or material
seed uncertainty. The promotion rule is recorded before reading either seed.
Three seeds permit descriptive dispersion; they do not justify a Gaussian
tail or years-of-search claim by themselves.

## 8. Matched temporal mechanism program

The source claim is:

`LSTM B outperforms dense A because recurrence Z exploits temporal structure S.`

The source-configured Table-III rows cannot identify this claim. After the
anchors, run one bounded matched program with identical data identities,
Sigmoid reconstruction head, MSE, Adam `1e-3`, no dropout, stopping rule, latent
width, score, threshold-selection rule, update budget, and three fixed seeds.
Choose FC widths before outcomes so its trainable parameter count is within 5%
of the LSTM encoder-decoder. Use at most 131,072 benign fitting rows and a fixed
held-out meter sample.

For each paired seed, fit both architectures under:

1. original chronological coordinates;
2. one fixed global coordinate permutation applied to train and test; and
3. independent within-row permutations that preserve every row's multiset but
   remove order.

Measure aggregate and per-attack AUC, constrained DR, fitting loss, paired
score/ranking changes, and especially Attack 6 reversal. Verify the ablations
changed lag dependence while retaining marginal distributions. Report the
difference-in-differences between LSTM and FC under order destruction with a
meter-cluster paired interval. A smallest material advantage is 2 balanced-ACC
points or 0.02 AUC, frozen before execution. Also report zero-parameter energy,
variance, range, zero-count, and order-sensitive lag/reversal rules through the
same held-out identities.

This program tests whether `S` exists and matters, whether FC lacks the useful
behavior, whether recurrence supplies and uses it, and whether that use causes
a paired advantage. It does not alter the numerical reproduction.

## 9. Tables IV and V

### Table IV

The paper calls full `|X_TR|` 60 million while the audited cache contains
1,500,523 training profiles or 72,025,104 scalar readings. Preserve this
disagreement. Interpret Table IV's columns as nested half, three-quarter, and
full fractions of the fixed training-profile order. Report actual profiles and
scalar readings beside every result.

Reuse each full Table-III model's fit time and ACC for the full column. After
all feasible full anchors, Table IV requires ten new fits: half and
three-quarter for each of five proposed models. Each uses the same seed and
model contract. A partial fit has the same pilot and 72-hour gate; an
ineligible or operationally incomplete full model does not produce eligible
partial rows. Our hardware time is reported as our hardware time, not a direct
reproduction of the paper's unspecified hardware.

### Table V

The primary reading uses each full Table-III model, its printed threshold, the
same original held-out benign rows, and one original attack family at a time,
without ADASYN. It requires no retraining and normally no rescoring: select the
original rows from the preserved Table-III score vector using provenance. With
a fixed model, threshold, and benign rows, FA must be identical across attacks;
show this identity before comparing the paper's varying FA cells.

Two alternatives remain visible but do not replace the primary result:

- independently sampled balanced benign evaluation rows per attack, using a
  frozen seed schedule and the same model; and
- independently trained model seeds per attack, the most generous reading of
  “multiple experiments.”

The first is a cheap saved-score analysis. The second would require 30 full
fits and is not promoted by wording alone. It is reconsidered only if a
source artifact identifies per-attack training or if the common-model result
leaves a numerical question that cannot be answered from the structural FA
identity. This is a cost-and-identifiability exclusion, not a claim that the
authors did not retrain.

## 10. Table II finite repaired family

The literal outcome remains first: SGCC supplies 1,034 daily values per
customer while every printed architecture consumes 48 half-hour values, and
the paper states no mapping or missing-value rule. Literal Table II is not
executable.

The primary finite architecture-preserving mappings are:

- earliest 48 days;
- latest 48 days; and
- 48 contiguous chronological means spanning all 1,034 days.

Use exact SGCC customer labels, drop only five completely empty customers, and
make missing handling visible. The primary missing branch linearly
interpolates interior customer gaps and fills remaining edges using medians
computed from benign `B1` only. Zero fill and per-customer observed-mean fill
receive a cheap data/simple-model screen. Promote their neural runs only if
they materially change row values or simple-model rankings under a
predeclared standardized-distance/AUC criterion, not if they happen to move
toward the paper.

Windowed 48-day interpretations change one customer into many dependent
examples and make the paper's row/customer split language non-identifiable.
They remain excluded from the first finite family, with that limitation
stated. For each promoted representation, run the five proposed models and the
feed-forward positive control first. Add the remaining five benchmarks only
once for the primary latest-48 branch, because they do not discriminate the
proposed architecture claims across representation branches. Use root seed
`20260824`; preserve all rows and meter-cluster uncertainty.

## 11. Findings and stopping rules

The numerical finding reports the literal failures, every eligible primary
anchor, every promoted finite interpretation, all seven metrics, thresholds,
directions, seeds, failures, and exact tested space. It does not call an
undocumented completion impossible.

The mechanism finding reports each link in `B > A because Z exploits S` as
supported, contradicted, unidentified, or untested. A source-configured metric
ordering is not mechanism evidence.

The attainability finding combines genuine global bounds with a finite
empirical envelope. It records learning curves, data-size curves, fixed-score
threshold envelopes, capacity/matched-model behavior, cumulative GPU-hours,
and failures. It may say “highly implausible within the declared envelope” only
if the target remains far outside a stable/saturating envelope. It may say
“structurally impossible” only for a separately proved bound under its exact
assumptions. Publication dates do not bound unreported research time, and no
Gaussian tail or “years of search” estimate is inferred from three seeds.

Stop and discuss after each primary anchor audit, after the four-model wave,
after the mechanism wave, and before any Table-IV depth or secondary learned-
variance/full Table-II branch. Do not publish a new scientific result before
that discussion. Preserve favorable, unfavorable, failed, and interrupted
outcomes identically.

## 12. Scientific checkpoint

The proposed first executable wave is exactly four two-hour feasibility jobs,
one per remaining Table-III model. If their gates pass, the proposed second
wave is one immutable full anchor per model with a 72-hour ceiling each. No
Table-IV fit, extra seed, mechanism fit, Table-II fit, or per-attack retraining
is part of those two waves.

The checkpoint decisions are:

1. accept first-step-latent/zeros plus top-state-only as the primary LSTM-SAE
   and LSTM-VAE decoder completion;
2. accept Equation-(10) sum-squared-plus-KL training and ten-draw unnormalized
   Gaussian kernels as the primary VAE completion, with all listed score
   directions preserved;
3. accept autoregressive, previous-reconstruction, concatenated additive
   attention as the primary AEA completion;
4. retain batch 32 for comparability with the audited FC-SAE and stop rather
   than silently enlarge it if the 72-hour gate fails; and
5. accept the staged order: feasibility, four Table-III anchors, discussion,
   matched mechanism, Tables IV/V, then finite Table II.

Any changed decision creates a new dated contract before code is frozen. It
does not edit this document after an outcome.
