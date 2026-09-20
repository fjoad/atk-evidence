# Paper-derived method specification

**Study:** Takiddin et al., “Robust Electricity Theft Detection Against Data
Poisoning Attacks in Smart Grids,” *IEEE Transactions on Smart Grid* 12(3),
2675–2684, May 2021.

**DOI:** `10.1109/TSG.2020.3047864`

**PDF SHA-256:**
`03a372fb5ee129799d43a50e6bec8f86fb108e9f8bdc85350e3ffe3e13d4a4ff`

**Source state:** all ten pages were rendered and visually inspected before
formal experimental implementation. This file records the paper, not an
inferred author implementation.

## 1. Questions and headline claims

The abstract, Section I-B, and Section V ask and answer three questions:

1. How much do false benign labels in training damage electricity-theft
   detectors? The paper reports detection-rate deterioration of up to 17%.
2. Are generalized or customer-specific detectors more robust? The paper
   reports a generalized-detector advantage of roughly 2–4%.
3. Can a detector remain accurate under poisoning without a separate filtering
   stage? The proposed sequential AEA–GRU–feed-forward ensemble reportedly
   retains DR 92.2% and FA 5.8% at 30% poisoning, versus DR 95.2% and FA 2.9%
   without poisoning.

The abstract additionally reports a deep-over-shallow improvement of 12%, a
sequential-over-ensemble-averaging improvement of up to 10%, and only 1–3%
degradation for the proposed detector under strong poisoning. Section IV-C
describes a 10–25% improvement over benchmark and ensemble-average detectors at
30% poisoning.

## 2. Data and sample unit

### Printed statements

- **Source:** Irish Smart Energy Trial, cited as reference [19] (Section II-A,
  page 2676).
- **Population:** 3,000 residential smart meters.
- **Sampling:** one reading every 30 minutes over 18 months, described as about
  25,000 reports per customer.
- **Notation:** `E_c(d,t)` is customer `c`'s true consumption for day `d` and
  within-day period `t`; `R_c(d,t)` is the meter report.
- **Model sample:** each row of a customer consumption-profile matrix is one
  daily profile (Section II-B and Table I, page 2677). Thirty-minute sampling
  implies 48 ordinary within-day coordinates, but the paper does not explicitly
  state the model input width.

### Material omissions

- The identities of the 3,000 customers are not given.
- Exact dates, daylight-saving treatment, missing readings, duplicates,
  filtering, and day-admission rules are not given.
- The paper calls the restricted Irish data “publicly available” but does not
  state the access snapshot or file checksums.
- No random seed, split identities, or repeated-run count is reported.

These omissions prevent exact population reconstruction from this paper alone.
Existing files from another study may be considered only after independent
identity and checksum verification.

## 3. Theft attacks

Table I (page 2677) applies six functions to every customer matrix:

1. `f1(E_c(d,t)) = alpha E_c(d,t)` — constant partial reduction.
2. `f2(E_c(d,t)) = beta(d,t) E_c(d,t)` — dynamic partial reduction.
3. `f3(E_c(d,t)) = 0` for `t` in `[t_i(d), t_f(d)]`, otherwise the original
   reading — selective bypass.
4. `f4(E_c(d,t)) = E[E_c(d)]` — constant daily-mean report.
5. `f5(E_c(d,t)) = beta(d,t) E[E_c(d)]` — dynamic mean-based report.
6. `f6(E_c(d,t)) = E_c(d,T-t+1)` — within-day reversal.

Each benign matrix yields six malicious matrices. Every daily row is labeled
benign `0` or malicious `1`.

The distributions or fixed values for `alpha`, `beta(d,t)`, `t_i(d)`, and
`t_f(d)` are absent; the paper says the attacks are adopted from reference [1].
The definition of `T` and the mapping from time labels to integer indices are
also omitted. These are material `I` branches, not details to select after
observing performance.

## 4. Data preparation

### 4.1 Generalized novelty detectors

Section III-A.1 (page 2677) gives this order:

1. concatenate benign daily profiles from all customers;
2. split benign profiles into disjoint training and test sets in a 2:1 ratio;
3. concatenate all malicious profiles with the benign test set;
4. apply ADASYN to the final test set, explicitly oversampling the minority
   benign class to balance benign and malicious evaluation samples;
5. fit zero-mean/unit-variance feature scaling on the benign training set; and
6. apply the same scale to the test set.

The split unit is called a “sample”; customer-disjointness is not stated.
ADASYN version, neighbor count, seed, and handling of synthetic test identities
are omitted.

### 4.2 Customer-specific novelty detectors

Repeat the same procedure separately for each customer `c`, producing
`X_TR,c`, `X_TST,c`, and `Y_TST,c`. Table IV reports performance averaged over
customers, but the averaging rule, customer eligibility, failures, and
dispersion are not reported.

### 4.3 Generalized two-class detectors

Section III-A.2 gives this order:

1. concatenate benign and malicious samples across customers;
2. apply ADASYN to balance the complete labeled population;
3. split the balanced data into disjoint training and test sets in a 2:1 ratio;
4. fit feature scaling on the training split; and
5. apply that scale to the test split.

The procedure permits synthetic siblings or source-customer rows to cross the
split unless identity-aware grouping occurs, but no grouping rule is stated.

### 4.4 Customer-specific two-class detectors

Repeat the two-class procedure independently within each customer.

### 4.5 Poisoning

Section III-A.3 (page 2678) defines poisoning as malicious training samples
falsely labeled benign.

- Generalized detector: 0%, 10%, 20%, or 30% of customers “present poisoned
  data.”
- Customer-specific detector: 0%, 10%, 20%, or 30% of customer `c`'s samples
  are poisoned.

The paper does not specify which customers, days, or attack types are selected;
whether all six attacks for a chosen source day are flipped; whether percentages
are exact; how novelty training acquires malicious samples; whether the same
nested identities are used across levels; or whether poisoning precedes or
follows ADASYN and splitting. These omissions can change both the data and the
scientific meaning of generalized-versus-customer-specific robustness.

## 5. Benchmark detectors

Section III-B and Table II specify seven models.

### Novelty detectors

- **ARIMA:** predicts future consumption, uses prediction MSE, and flags scores
  above a threshold. The order, fitting unit, forecast horizon, and score
  aggregation are absent. Printed threshold: `0.58`.
- **AEA:** LSTM encoder/decoder with attention, trained on benign profiles and
  scored by reconstruction error. Three encoder layers `(500, 300, 200)` and
  three decoder layers `(200, 300, 500)`; SGD; dropout `0`; weight constraint
  `1`; Sigmoid hidden and output activations. Printed threshold: `0.51`.

The AEA prose and Figure 2 describe attention over encoder and decoder states
and concatenation of the context with a prior reconstructed output. Algorithm 1
does not fully define decoder inputs, attention dimensions, the initial
reconstructed value, output projection, or reconstruction-loss expression.

### Two-class detectors

- **Random forest:** 100 estimators; other tree and sampling settings omitted.
- **AdaBoost:** 50 estimators and decision-tree weak learners (III-B.2(b),
  p.2678); tree depth, learning rate, and algorithm variant omitted.
  September 20 correction: the earlier note said the base estimator was
  omitted, overlooking the explicit decision-tree family. The initial
  executable completion is separately frozen in ADABOOST_PILOT.md.
- **SVM:** `C=1.0`, Sigmoid kernel; gamma, coefficient, and probability/decision
  score handling omitted.
- **Feed-forward:** six hidden layers, 500 neurons per layer, Adamax, dropout
  `0`, weight constraint `3`, ReLU hidden activation, Sigmoid output.
- **GRU:** eight hidden GRU layers, 300 units each, Adam, dropout `0.2`, weight
  constraint `5`, ReLU hidden activation, Softmax output.

The binary output shape, exact cross-entropy form, optimizer learning rates,
initialization, class weights, and threshold for the classifiers are not stated.

## 6. Hyperparameter selection

Section III-C (page 2679) reports a sequential grid search. One parameter at a
time is chosen by the “best reported DR” before the next parameter is searched.
The finite sets are:

- layers: `{2, 4, 6, 8}`;
- neurons: `{100, 200, 300, 500, 1000}`;
- optimizer: `{SGD, Adam, Adadelta, Adamax}`;
- dropout: `{0, 0.2, 0.4, 0.5}`;
- weight constraint: `{0, 1, 3, 5}`;
- hidden activation: `{ReLU, Sigmoid, Linear, tanh}`; and
- output activation: `{Softmax, Sigmoid}`.

Cross-validation is said to occur over `X_TR` or `X_TR,c`. The number and
identity of folds, poison level used for tuning, threshold used to calculate
DR, parameter order, tie rule, refit rule, and whether results are averaged or
selected are absent. A novelty training set contains only benign labels, so it
cannot yield DR without a labeled validation population that the paper does not
identify.

## 7. Training, hardware, and time

Section III-D.2 and page 2680 state:

- Keras Sequential API;
- 50 epochs for deep detectors;
- batch size `K=100`;
- NVIDIA GeForce RTX 2070 accelerator;
- approximately one hour to train shallow detectors;
- approximately 1.5–3 hours to train deep detectors; and
- approximately two seconds to report one online decision.

Section IV-C (page 2682) states:

- ensemble AEA widths `(500,300,200|200,300,500)`;
- eight recurrent layers with 300 GRUs each;
- one 500-neuron fully connected layer;
- Adam, dropout `0`, weight constraint `1`, ReLU hidden activation, and
  Sigmoid output;
- approximately three hours to train ensemble averaging;
- approximately four hours to train the sequential ensemble; and
- approximately two seconds per online decision.

The source does not say whether times include data preparation, ADASYN,
hyperparameter search, validation, checkpointing, or evaluation; whether each
customer-specific model receives that time; the CPU/RAM/software versions; or
whether a timing is one run, a mean, or a selected run.

### Required later runtime contract

The printed RTX 2070 is the primary device, so no hardware inference is needed
unless that device is unavailable. A future numerical run must freeze one
device, 50 epochs, batch 100, exact fit boundary, seed, data identity, model,
and the applicable one-hour/1.5–3-hour/three-hour/four-hour budget before
execution. A substitute device requires a favorable pre-result throughput
calibration against the RTX 2070. No newer GPU, extra GPU, retry, or longer fit
may rescue the bounded paper-time question.

## 8. Thresholds and metrics

Section III-D.1 defines:

- `DR = TP/(TP+FN)`;
- `FA = FP/(FP+TN)`;
- `SP = 100-FA`;
- `PR = TP/(TP+FP)`;
- ordinary `ACC = (TP+TN)/(TP+TN+FP+FN)`;
- F1 as the harmonic mean of PR and DR; and
- AUC from the ROC curve.

The sentence explaining precision incorrectly describes the fraction of
malicious samples detected, which is DR, while its printed equation is the
standard precision formula.

For novelty models, the paper says the threshold is the “median of the
interquartile range (IQR) of the ROC curve,” obtained by dividing each ROC curve
into three quartiles and taking the median of the IQR. This is not a standard or
uniquely executable operation. Constructing an ROC also requires labeled
examples, but the labeled calibration population and leakage boundary are not
identified.

Tables III–V report all metrics to one decimal percentage point. The source says
ADASYN balances the relevant populations; the static audit therefore checks
both the explicitly printed identities and the additional balanced-population
identities with outward one-decimal rounding allowance.

## 9. Numerical targets

- **Table II:** selected hyperparameters for generalized AEA, feed-forward, and
  GRU models.
- **Table III:** seven generalized detectors × seven metrics × four poisoning
  levels = 196 cells.
- **Table IV:** seven customer-specific detectors × seven metrics × four
  poisoning levels = 196 cells.
- **Table V:** AEA, ensemble averaging, and sequential ensemble × seven metrics
  × four poisoning levels = 84 cells.

Together Tables III–V contain 476 metric cells. Every value is frozen under
`reported/` before the arithmetic audit executes.

## 10. Causal-claim map

| ID | `B > A` | Added capability `Z` | Required structure `S` |
|---|---|---|---|
| M1 | generalized > customer-specific detectors | cross-customer aggregation dilutes mislabeled data and learns shared honest features | transferable structure across customers, with poison sparse enough to dilute |
| M2 | deep > shallow detectors | nonlinear/hierarchical representation | signal unavailable to the shallow comparison under matched data and selection |
| M3 | GRU/AEA > feed-forward | recurrent temporal modeling | consequential within-day order dependence not already captured by fixed coordinates |
| M4 | AEA > GRU/feed-forward | reconstruction plus attention over important times | localized temporal information and a useful reconstruction bottleneck |
| M5 | sequential ensemble > ensemble averaging | serial AEA reconstruction, GRU feature extraction, then classification | complementary information preserved and made more usable by composition than averaging |
| M6 | sequential ensemble remains robust under poisoning | component diversity and staged feature extraction stabilize the boundary | components fail differently and retain label-relevant information as poison increases |

For each claim, later mechanism evidence must separately test whether `S`
exists, whether it is useful, whether `A` lacks the capability, whether `Z`
gives it to `B`, whether trained `B` uses it, and whether that use causes a
matched advantage. Tables III–V alone confound architecture, data volume,
training objective, hyperparameter selection, and thresholding.

## 11. Straight-through candidate and cheap checks

No experimental branch is authorized yet. Subject to the source-only
checkpoint, the least expensive candidate anchor is one generalized two-class
baseline at 0% and 30% poisoning on one frozen nested split. Random forest is
computationally cheapest but leaves default tree semantics; feed-forward is
closer to the central deep-versus-shallow comparison and has a paper-reported
1.5–3-hour ceiling.

Before choosing, the source-only audit must answer:

- whether each printed metric row is arithmetically realizable;
- whether prose degradation and ordering claims match the tables;
- whether the 3,000-customer population can be identified without target-guided
  selection;
- whether novelty and two-class poisoning can be made executable without
  inventing the ordering of poisoning, ADASYN, and splitting; and
- whether generalized/customer-specific comparisons can separate aggregation
  from the much larger generalized training population.

## 12. Current boundary

This is a source specification, not a reproduction result. Shared ingredients
with another paper justify efficient verification but transfer no numerical,
mechanism, attainability, or intent conclusion. Stop after the preregistered
static finding and discuss the first executable branch and its RTX-2070 budget.
