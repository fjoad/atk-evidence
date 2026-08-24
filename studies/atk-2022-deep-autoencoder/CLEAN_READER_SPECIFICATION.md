# Clean-reader source freeze

**Date:** 2026-08-24

**Status:** Frozen candidate for Checkpoint 1 review; no formal implementation
or named-data execution is authorized by this document

**Primary evidence question:** numerical reproduction (`N`)

**Implementation semantics:** paper-literal `P` where executable; minimal
reasonable-reader interpretation `I` only where the source cannot execute or
omits a necessary value

## 1. Authority and boundary

The scientific authority is the 12-page publication with SHA-256
`f3098e0c27ee19b27bea026aedc3d10e5dbb0c46f5cd01ed5bd5c05b7dcf850f`:

> Abdulrahman Takiddin, Muhammad Ismail, Usman Zafar, and Erchin Serpedin,
> “Deep Autoencoder-Based Anomaly Detection of Electricity Theft Cyberattacks
> in Smart Grids,” *IEEE Systems Journal* 16(3), 4106–4117 (2022),
> DOI `10.1109/JSYST.2021.3136683`.

All 12 pages were visually re-inspected after the Phase-2 sandbox. The source
locations that determine this freeze are:

- data and attacks: PDF pp. 2–4, printed pp. 4107–4109, Sec. II and (1)–(6);
- FC-SAE method: PDF pp. 4–5, printed pp. 4109–4110, Fig. 3, (7), Algorithm 1;
- scoring and metrics: PDF p. 9, printed p. 4114, Sec. III-C/D;
- optimization and threshold claims: PDF pp. 9–10, printed pp. 4114–4115,
  Algorithm 6, Figs. 7–8, Table I, Sec. IV-B/C; and
- result targets: PDF p. 11, printed p. 4116, Tables II–V and adjacent text.

The provisional source map is
[`CLEAN_READER_ORIENTATION.md`](CLEAN_READER_ORIENTATION.md). The disposable
sandbox is [`DISCOVERY_SANDBOX.md`](DISCOVERY_SANDBOX.md). Neither historical
implementation nor historical outcomes selected the completions below. They
remain closed until Phase 4.

The sandbox affected the *questions* in this freeze, not their answers:

- preserve a simple-rule floor as a later `C/M` control;
- preserve reversal as the clearest temporal witness;
- make output domains explicit before interpreting reconstruction scores; and
- branch the contradictory VAE score directions rather than silently choosing
  one.

None of those controls is part of the primary numerical anchor.

## 2. Frozen numerical target

The paper-level target is the complete pattern transcribed source-only in
[`CLEAN_READER_ORIENTATION.md`](CLEAN_READER_ORIENTATION.md#complete-reported-target-pattern):
all rows and metrics in Tables II–III, all model/data-size cells in Table IV,
and every model/attack cell in Table V. Matching one headline score is not a
paper-level reproduction.

The first exact-data anchor is deliberately one cell family:

> **`CR-ISET-FCSAE-01`: Table III, ISET, FC-SAE, one frozen seed.**

Its complete printed row is DR 81%, FA 15%, SP 85%, PR 81%, ACC 83%, F1 81%,
and AUC 81%. This anchor asks whether the simplest proposed model can traverse
the clean-reader paper-to-data-to-score path once. It cannot establish the
complete paper finding by itself.

## 3. Literal `P` route

An ordinary literal pass gives this order:

1. Select residential half-hour ISET readings for approximately 1.5 years.
2. Treat one day as one 48-value profile.
3. Apply attacks (1)–(6) to every customer’s benign consumption matrix, giving
   six malicious matrices per customer.
4. Merge customers into benign `B` and malicious `M` classes.
5. Normalize both classes to zero mean and unit variance.
6. Split benign customers 2:1 into disjoint `B1` and `B2`; train the anomaly
   detector on `B1`; test on `B2` concatenated with `M`.
7. Apply ADASYN to the minority class in the test set.
8. Train the Table-I FC-SAE on benign `B1` using reconstruction MSE.
9. Threshold per-profile reconstruction error, with larger error meaning
   anomaly, and compute the printed metrics.

The literal route stops before execution for two primary-anchor reasons:

- **`P-FAIL-01`, Attack 3:** (3) gives positive duration `t_l` but prints
  `t_f=t_i-t_l`. The stated interval `[t_i,t_f]` is therefore reversed and
  cannot implement the described forward 4–24 hour bypass (PDF p. 3, printed
  p. 4108).
- **`P-FAIL-02`, validation/threshold:** anomaly-detector `X_TR` is defined as
  benign-only `B1`, yet the paper constructs an ROC curve and maximizes
  validation DR by cross-validation over `X_TR`. Those operations require
  malicious labels. “Median of the interquartile range of the ROC curve” also
  does not uniquely select a score threshold (PDF pp. 4 and 9–10, printed
  pp. 4109 and 4114–4115).

The primary executable route below preserves these failures and labels every
repair `I`; it does not relabel the literal method as executable.

## 4. Exact named-data identity

Use Version 1 of the official ISSDA deposit **CER Smart Metering Project -
Electricity Customer Behaviour Trial, 2009–2010**, persistent identifier
`doi:10.7929/ISSDA/BX59EU`, dataset identifier `ISSDA:0012-00`.

The eligible raw inputs are exactly:

| File | Bytes | Official MD5 |
|---|---:|---|
| `File1.txt.zip` | 101,978,611 | `00203f66f3f5e5201b20ed160b787684` |
| `File2.txt.zip` | 102,197,028 | `5e3af1474d3c8976e2e1e0f8c1969507` |
| `File3.txt.zip` | 101,624,145 | `b537785f8b37cb3e89103600d39da8ff` |
| `File4.txt.zip` | 102,401,577 | `53ec9e70c1610b74ae72417cc010a0c3` |
| `File5.txt.zip` | 102,257,883 | `6f8c7c9dfba3bbfbff0e5f1703e122fc` |
| `File6.txt.zip` | 147,826,765 | `c0a435d0359974f23ce434b5e838e251` |
| `SME and Residential allocations.tab` | 196,316 | `124c10711ab1e7c52cb7317c8f69e42e` |

These identities come from the official dataset metadata, not a project
cache. The anchor must stop at the data gate if any exact input is absent or
has a different checksum. A semantically similar allocation serialization is
not eligible for this first clean-reader anchor without a new visible `I`
branch and Checkpoint-1 review.

## 5. Primary reasonable-reader completion `I`

Every choice here is frozen before the eligible score exists.

### 5.1 Parsing and daily profiles

- Select every meter labeled residential in the exact allocation table. The
  source says “around 3000” rather than naming IDs; using every officially
  labeled residential meter avoids result-guided subsampling.
- Use the complete consumption interval present in the six named archives.
- Interpret the dataset’s day/slot field using the accompanying official data
  documentation.
- A model example is one customer-day with exactly the 48 ordinary half-hour
  slots, in chronological order. Drop the whole customer-day if a slot is
  absent, duplicated, nonfinite, or if the day contains an extra daylight-
  saving slot. Do not impute, truncate, or modify a raw reading.
- Preserve meter ID, day ID, and slot provenance through preparation and
  scoring. A duplicate or malformed record is a recorded preparation failure,
  not a silent overwrite.

This completes the source’s 48-input daily-profile requirement. The paper’s
later “60 million” and “individual readings” statements are interpreted as
scalar-reading counts and online narrative, not as a change in the model’s
statistical unit. Material alternatives not selected are a deterministic
3,000-meter subset, a narrower date window, retaining the first 48 slots of a
long day, and padding or interpolating short days.

### 5.2 Synthetic attacks

Generate attacks in the raw consumption domain before normalization. Use root
seed `20260824` and independent deterministic random streams by attack,
customer, and day.

- **Attack 1:** draw one continuous `alpha ~ Uniform(0.1,0.8)` per customer’s
  attack-1 matrix and multiply every retained reading by it. The absence of
  `d,t` subscripts and the phrase “fixed random fraction” support one draw per
  matrix.
- **Attack 2:** draw independent continuous
  `beta(d,t) ~ Uniform(0.1,0.8)` for each reading and multiply pointwise.
- **Attack 3 (`I-ATTACK3`):** draw integer duration `l ~ Uniform{4,…,24}` hours;
  draw integer start `t_i ~ Uniform{0,…,24-l}`; set `t_f=t_i+l`; and set both
  half-hour readings in that contiguous within-day interval to zero. This is
  the smallest completion that realizes the stated forward 4–24 hour bypass
  without leaving the day. Preserve the literal reversed-interval failure
  alongside this repaired branch.
- **Attack 4:** replace all 48 readings by that customer-day’s arithmetic mean.
- **Attack 5:** multiply the day’s arithmetic mean by independent continuous
  `beta(d,t) ~ Uniform(0.1,0.8)` at each reading.
- **Attack 6:** reverse the 48 readings within each day exactly.

Material alternatives not selected for the first anchor are: one Attack-1 draw
per day or reading; Attack-3 sign repair followed by clipping, wrapping, or an
independent start draw; and continuous-time interval endpoints. They remain
listed `I` branches and cannot be selected after seeing the anchor.

### 5.3 Population identity, normalization, and split

- Construct one benign row and six attack rows per retained customer-day.
- Fit one feature-wise affine standardizer jointly on the complete `B` and `M`
  populations before any split, using population variance (`ddof=0`). Apply
  that same transform to every row. This preserves the source’s stated order,
  including its information leakage; a train-fitted scaler is a later `C`
  control, not reproduction.
- Sort residential meter IDs, deterministically shuffle them with the root
  seed, and assign the first two thirds to `B1` and the remainder to `B2`.
  All days from one meter remain in one side.
- `X_TR` is every standardized benign row from `B1` meters.
- `X_TST` is every standardized benign row from `B2` meters plus all six
  standardized attack rows derived **only** from those same `B2` meters. This
  completion is required by the explicit claim that train and test customers
  are disjoint.
- Apply ADASYN only to `X_TST,Y_TST`, after the customer split and scaling.
  Oversample the minority class to equality using Euclidean distance, five
  neighbors, and the root seed. On ISET the benign class is expected to be the
  minority because each retained benign row has six attack counterparts.
  Preserve original-versus-synthetic row identity. Do not use any resampled
  test row for training.

Material alternatives retained but not selected are global-scalar,
per-customer, and train-fitted normalization; row-random splitting; inclusion
of attacks from training meters in `M`; separate attack RNG regeneration; and
no, pre-split, or training-set ADASYN.

### 5.4 FC-SAE architecture

Use the Table-I and Sec. III-E final configuration directly; do not repeat the
undefined sequential search for the first anchor.

- Input and reconstruction width: 48.
- Eight hidden/latent layers, excluding input and output:
  `400, 300, 200, 100, 100, 200, 300, 400`.
- Sigmoid activation after every hidden/latent dense layer.
- Dropout rate 0.4 after every hidden/latent activation during training only.
- Softmax output over the 48 reconstructed coordinates.
- Dense kernels: Glorot-uniform initialization; biases: zero.
- Objective: mean squared reconstruction error, averaged over 48 coordinates
  and then over the minibatch.
- Optimizer: Adam with learning rate `1e-3` and otherwise frozen Keras defaults.

This reads Table I’s `L*=8` as four encoder-side and four decoder-side
hidden/latent layers, consistent with the widths printed below the table.
Alternatives include counting only encoder layers, omitting the mirrored
central width, and applying dropout only on one side; none is eligible for the
first anchor.

The Softmax output is preserved even though standardized targets generally lie
outside the probability simplex. The exact projection floor is a diagnostic
reported beside the anchor, never a silent reason to replace Softmax with a
linear head.

### 5.5 Training completion

The paper names Keras Sequential API and says “while not converged” but gives
no batch size, learning rate, epoch cap, stopping rule, initialization, seed,
or repeat count. Freeze the following ordinary execution completion:

- one run with root seed `20260824` applied to Python, NumPy, and the neural
  backend;
- deterministic operations where the backend supports them;
- batch size 32, the ordinary Keras `fit` default;
- seeded reshuffling of `X_TR` each epoch;
- maximum 100 epochs;
- after a minimum 10 epochs, stop when training MSE has not improved by at
  least `1e-6` for five consecutive epochs;
- restore the weights with the lowest observed training MSE; and
- use all `X_TR` rows for fitting; do not carve out a hidden validation set.

The anchor does not claim to reproduce the paper’s hyperparameter search or
cross-validation, which are non-executable from benign-only `X_TR`. Printed
optimal settings and the printed FC-SAE threshold are treated as supplied
outputs of that undocumented selection process.

Alternative batch sizes, epoch caps, validation carve-outs, convergence rules,
initializers, seeds, and repeated runs remain materially possible but are not
eligible until promoted after the anchor.

### 5.6 Score, threshold, and metrics

- Score every row by mean squared error over its 48 coordinates.
- High reconstruction error means anomaly for FC-SAE.
- Use the printed FC-SAE threshold `0.58` unchanged. Do not derive a new
  threshold from the test labels and do not attempt to recreate the undefined
  IQR/ROC procedure.
- Predict malicious iff `score > 0.58`; equality is benign.
- Compute TP, TN, FP, and FN at the row level.
- Compute `DR=TP/(TP+FN)`, `FA=FP/(FP+TN)`, `SP=1-FA`,
  `PR=TP/(TP+FP)`, `ACC=(DR+SP)/2`, and F1 as the harmonic mean of DR and PR.
  Use the printed formula for precision rather than its conflicting prose.
- Compute AUC from the continuous high-is-anomaly MSE scores.
- Report rates as percentages without using printed rounding during
  computation.

The literal threshold-construction failure remains part of the result. The
fixed printed threshold merely allows the final published detector to be
tested once.

## 6. Deferred non-anchor literal failures

These source defects do not affect FC-SAE execution but remain frozen before
any later model-family work:

- **VAE objective:** (9) compares distributions over incompatible variables
  and states a bound against a different marginal. Literal evaluation fails;
  a standard ELBO is an `I` repair.
- **VAE variance:** Algorithms 3–4 do not supply a positive variance transform,
  likelihood variance, stabilization constant, KL scaling, or Monte Carlo
  count. No VAE row is executable until those are predeclared.
- **VAE direction:** Sec. III-B says probability below threshold is anomalous;
  Sec. III-C says reconstruction probability greater than threshold is
  anomalous. Both literal readings must remain visible. Low probability is the
  anomaly-consistent `I` orientation; it cannot be chosen from its outcome.
- **LSTM decoder:** tensor orientation, decoder input schedule/teacher forcing,
  masking, and several state connections are insufficiently specified.
- **Attention:** the alignment network, vector dimensions, decoder query, and
  the concatenation/summation around `c_v,t` and reconstructed output are not
  fully executable from Fig. 5 and Algorithm 5.
- **Search:** `L={2,3,4,5}` conflicts with Table-I optima 6 and 8 unless one
  silently changes the layer-count convention. The final printed architecture
  may be used directly, but the reported search cannot be independently
  replayed as written.

These are not evidence that any reported model fails. They define where later
`I` branches would begin.

## 7. Causal-claim map and separated controls

The five source explanations remain:

| ID | `B > A` | Claimed added capability `Z` | Required task structure `S` |
|---|---|---|---|
| M1 | deep autoencoders > shallow/static detectors | learned hierarchical/nonlinear representation | nontrivial structure not captured by simple rules |
| M2 | LSTM > FC counterpart | recurrent temporal modeling | consequential order dependence |
| M3 | VAE > SAE counterpart | latent variance and reconstruction probability | variance information beyond deterministic reconstruction |
| M4 | LSTM-AEA > LSTM-SAE/VAE | attention over important time steps | localized or long-range information plus a context bottleneck |
| M5 | benign-only anomaly detector detects unseen attacks | deviation from learned benign structure | attack departures that generalize beyond trained examples |

The first anchor tests none of these causal links. If Checkpoint 2 later
promotes `M`, eligible controls must separately test simple-rule floors,
presence and importance of `S`, capability of `A` and `B`, isolation of `Z`,
trained use of `Z`, and a fair paired effect. Reversal is the first candidate
for M2 because it preserves the day’s multiset. Linear-output controls,
train-fitted scaling, matched-capacity ablations, variance witnesses,
attention-weight inspection, and structure destruction remain `C/M`, never
primary `N` reproduction.

## 8. First-anchor execution and stopping contract

Before any eligible run, Phase 4 must prove that one transparent five-file
route implements every item above and no historical outcome influenced it.
Then Phase 5 records:

- anchor ID `CR-ISET-FCSAE-01`;
- `N` question and `P/I` classification;
- source-PDF fingerprint and exact raw-data hashes;
- code commit and environment lock;
- seed `20260824` and derived RNG streams;
- architecture, preparation counts, score direction, threshold, and metrics;
- one accelerator at most, a declared wall-time budget, and no adaptive retry;
- competing predictions: complete-row close match, implementation/source
  defect, materially distinct completion required, or non-match inside this
  finite contract; and
- the Phase-6 numerical finding it feeds.

The attempt finishes only when it preserves preparation provenance, every
epoch loss, final/best weights, per-row identities, raw scores, predictions,
confusion counts, all seven metrics, timings, warnings/failures, environment,
configuration, and hashes. A literal operational failure is also a complete
anchor outcome if faithfully preserved.

Stop after that one attempt. Do not change a completion, add a seed, run
another model, tune a threshold, or launch a control before full artifact
inspection and Checkpoint 2. Phase 6 compares the complete seven-value row and
measuring chain, not merely the closest metric.

## 9. Checkpoint-1 questions

User approval is required before Phase 4. Review specifically:

1. Is `CR-ISET-FCSAE-01` the right minimal first anchor?
2. Is the Attack-3 completion—duration first, then an in-day start—the fairest
   ordinary reading of the intended 4–24 hour bypass?
3. Should the first anchor require the exact official allocation `.tab`, as
   frozen here, or admit a separately proven semantic serialization as an `I`
   branch?
4. Is joint pre-split feature-wise scaling the right literal preservation?
5. Are the layer count, dropout placement, and visible training completion
   acceptable outcome-independent readings?
6. Is using the printed threshold 0.58, while preserving its failed derivation,
   preferable to inventing a threshold algorithm?

Disagreement loops to Phase 1 or Phase 2. Approval authorizes only the Phase-4
fidelity assessment; it does not authorize named-data execution, ambiguity
sweeps, repeated seeds, or formal mechanism/attainability work.
