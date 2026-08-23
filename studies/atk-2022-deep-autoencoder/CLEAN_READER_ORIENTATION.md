# Clean-reader rebase — Phase 1 source-only orientation

**Date:** 2026-08-23

**Status:** Phase 1 complete; provisional source reading, not the executable
source freeze

**Evidence class:** source extraction and static audit only; no experimental
`N`, `M`, or `A` result

**Governing plan:**
[`../../docs/plans/2026-08-23-clean-reader-reproduction-rebase.md`](../../docs/plans/2026-08-23-clean-reader-reproduction-rebase.md)

## Scope and isolation

This record captures what a competent reader can provisionally extract from
the publication before consulting the project's earlier method
reconstructions, implementations, or outcomes. The complete 12-page PDF was
visually inspected and its text was independently extracted page by page.

The reader and agents have prior exposure to this project, so this is not a
claim of cognitive blindness. It is a **procedurally source-isolated** pass:
only the fingerprinted PDF and its rendered pages were consulted while the
record below was formed. Earlier project artifacts remain neither confirmed
nor contradicted until a later explicit comparison.

## Source identity

- **VERIFIED:** Abdulrahman Takiddin, Muhammad Ismail, Usman Zafar, and Erchin
  Serpedin, “Deep Autoencoder-Based Anomaly Detection of Electricity Theft
  Cyberattacks in Smart Grids,” *IEEE Systems Journal*, vol. 16, no. 3,
  September 2022, printed pages 4106–4117.
- DOI: `10.1109/JSYST.2021.3136683`.
- Local PDF: 12 pages; 2,994,509 bytes.
- SHA-256:
  `f3098e0c27ee19b27bea026aedc3d10e5dbb0c46f5cd01ed5bd5c05b7dcf850f`.
- **OBSERVED:** the publication contains no author-code or repository link.

Locators below use both PDF page number and printed journal page number.

## Provisional straight-through reading

An ordinary source-only reading gives the following data-to-result path.

1. **Choose one of two datasets.** SGCC contains approximately 40,000
   customers with labeled benign and malicious daily readings over three
   years. ISET contains approximately 3,000 residential meters sampled every
   30 minutes for 1.5 years, or approximately 25,000 readings per customer.
   (PDF pp. 2–3; print pp. 4107–4108; Sec. II-A/B.)
2. **Construct ISET attacks.** Apply six printed transformations to every
   customer's benign consumption matrix: fixed reduction, pointwise random
   reduction, interval bypass, daily mean, randomly scaled daily mean, and
   reversal of time order. Each customer is said to produce six malicious
   matrices. (PDF p. 3; print p. 4108; (1)–(6).)
3. **Form examples and normalize.** A row is described as one daily energy
   profile. The paper merges customers, forms benign `B` and malicious `M`,
   and normalizes both classes to zero mean and unit variance before the
   reported split. (PDF pp. 3–4; print pp. 4108–4109; Sec. II-C.)
4. **Split anomaly-detector data.** Split normalized benign `B` 2:1 into
   `B1` and `B2`; train on `X_TR = B1`; concatenate `B2` with `M` for testing;
   then use ADASYN on the minority class in the test set. Customers in train
   and test are stated to be disjoint. (PDF pp. 3–4; print pp. 4108–4109;
   Sec. II-C.)
5. **Split supervised-detector data separately.** Concatenate benign and
   malicious classes, apply ADASYN, split 2:1 into train and test sets, and
   apply the same stated normalized feature scaling. (PDF p. 4; print
   p. 4109; Sec. II-C.)
6. **Fit five proposed anomaly detectors.** These are FC-SAE, LSTM-SAE,
   FC-VAE, LSTM-VAE, and LSTM-AEA. SAE and AEA use reconstruction MSE; VAE is
   described as using reconstruction probability. Training uses benign data.
   (PDF pp. 4–9; print pp. 4109–4114; Sec. III-A–C and Algorithms 1–5.)
7. **Tune sequentially.** A sequential grid search over listed layer, width,
   optimizer, dropout, and activation choices is said to use cross-validation
   over ISET `X_TR`, choosing settings that improve validation DR. (PDF
   pp. 9–10; print pp. 4114–4115; Sec. III-E, Algorithm 6, Table I.)
8. **Choose anomaly thresholds.** ROC curves are built using the same stated
   cross-validation over ISET `X_TR`; a threshold is described as the “median
   of the interquartile range” of the ROC curve. Seven numerical thresholds
   are then printed. (PDF pp. 9–10; print pp. 4114–4115; Sec. III-D and
   IV-B, Figs. 7–8.)
9. **Evaluate held-out data.** Threshold scores, form confusion matrices, and
   report DR, FA, SP, PR, ACC, F1, and AUC. The paper calls the test data
   completely unseen and distinct from validation. (PDF pp. 9–11; print
   pp. 4114–4116; Sec. III-D and IV-D, Tables II–V.)

This sequence is provisional. Several links are internally contradictory or
underdetermined and therefore cannot yet serve as the formal `P/I` freeze.

## Complete reported target pattern

The primary numerical target is not a single best score. It is the full
ordering and metric pattern printed for both datasets, plus the separate-attack
and data-size results.

### Table II — SGCC

| Detector | DR | FA | SP | PR | ACC | F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| FC-SAE | 83 | 14 | 86 | 83 | 84.5 | 83 | 83 |
| LSTM-SAE | 86 | 12 | 88 | 87 | 87 | 86.5 | 85 |
| FC-VAE | 90 | 9 | 91 | 91 | 90.5 | 90.5 | 88 |
| LSTM-VAE | 93 | 6 | 94 | 93 | 93.5 | 93 | 90 |
| LSTM-AEA | 96 | 4 | 96 | 95 | 96 | 95.5 | 93 |
| Naive Bayes | 75 | 16 | 84 | 75 | 79.5 | 77 | 73 |
| ARIMA | 88 | 10 | 90 | 87 | 89 | 87 | 88 |
| Single-class SVM | 91 | 8.5 | 91.5 | 90 | 91 | 90 | 89 |
| Feed forward | 91 | 9.5 | 90.5 | 90 | 91 | 90.5 | 89 |
| LSTM | 91.5 | 9 | 91 | 90.5 | 91 | 91 | 90 |
| Multiclass SVM | 92 | 7.5 | 92.5 | 91 | 92 | 91.5 | 90 |

Source: PDF p. 11; print p. 4116; Table II.

### Table III — ISET

| Detector | DR | FA | SP | PR | ACC | F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| FC-SAE | 81 | 15 | 85 | 81 | 83 | 81 | 81 |
| LSTM-SAE | 85 | 13 | 87 | 85 | 86 | 85 | 82 |
| FC-VAE | 88 | 11 | 89 | 89 | 88.5 | 88.5 | 85 |
| LSTM-VAE | 91 | 7 | 93 | 91 | 92 | 91 | 86 |
| LSTM-AEA | 94 | 5 | 95 | 93 | 94.5 | 93.5 | 90 |
| Naive Bayes | 73 | 18 | 82 | 73 | 77.5 | 73 | 70 |
| ARIMA | 86 | 12 | 88 | 86 | 87 | 86 | 87 |
| Single-class SVM | 90 | 9 | 91 | 89 | 90.5 | 89.5 | 87 |
| Feed forward | 90 | 11 | 89 | 89 | 89.5 | 89.5 | 88 |
| LSTM | 90.5 | 10 | 90 | 89.5 | 90 | 90 | 89 |
| Multiclass SVM | 91 | 8 | 92 | 90 | 91.5 | 90.5 | 89 |

Source: PDF p. 11; print p. 4116; Table III.

### Table IV — ISET data-size curve

| Model | Metric | `0.5 × N` | `0.75 × N` | `N` |
|---|---|---:|---:|---:|
| FC-SAE | Time (min) | 72 | 97 | 137 |
| FC-SAE | ACC | 70 | 78.5 | 83 |
| LSTM-SAE | Time (min) | 90 | 127 | 183 |
| LSTM-SAE | ACC | 75 | 83 | 86 |
| FC-VAE | Time (min) | 81 | 103 | 141 |
| FC-VAE | ACC | 79.5 | 86 | 88.5 |
| LSTM-VAE | Time (min) | 97 | 132 | 188 |
| LSTM-VAE | ACC | 83 | 90 | 92 |
| LSTM-AEA | Time (min) | 102 | 142 | 193 |
| LSTM-AEA | ACC | 86 | 93 | 94.5 |

Here `N = |X_TR|`; the paper states that full `N` is 60 million. Source: PDF
p. 11; print p. 4116; Table IV and adjacent text.

### Table V — separate ISET attacks

| Model / metric | (1) | (2) | (3) | (4) | (5) | (6) | AVG |
|---|---:|---:|---:|---:|---:|---:|---:|
| FC-SAE DR | 82.5 | 81 | 83 | 80 | 80 | 80 | 81 |
| FC-SAE FA | 15 | 16 | 10 | 17 | 17 | 19 | 15.5 |
| LSTM-SAE DR | 84.5 | 83 | 90 | 82 | 84 | 83 | 84.5 |
| LSTM-SAE FA | 13 | 15 | 9 | 14 | 14 | 14 | 13 |
| FC-VAE DR | 86 | 85 | 93 | 88 | 88 | 87 | 88 |
| FC-VAE FA | 11 | 12 | 8 | 10 | 11 | 12 | 10.5 |
| LSTM-VAE DR | 88.5 | 88 | 95 | 91 | 91 | 90 | 90.5 |
| LSTM-VAE FA | 7.5 | 8 | 4.5 | 8 | 8.5 | 8.5 | 7.5 |
| LSTM-AEA DR | 94 | 93 | 97 | 94 | 94 | 93 | 94 |
| LSTM-AEA FA | 3.5 | 4 | 2.5 | 6.5 | 5.5 | 6.5 | 5 |

Source: PDF p. 11; print p. 4116; Table V.

### Headline claims

- LSTM-AEA is reported as best: ISET DR 94%, FA 5%; SGCC DR 96%, FA 4%.
- Compared with benchmark detectors, AEA is said to improve DR by 3–21% or
  4–21% and FA by approximately 3–13% or 4–13%, depending on the passage and
  dataset.
- LSTM variants are said to improve their fully connected counterparts by
  approximately 3–4 DR points and 2–4 FA points.
- VAE variants are said to improve over SAE by approximately 3–7 DR points and
  2–5 FA points.
- AEA is said to improve over VAE by approximately 3–6 DR points and 2–6 FA
  points.

Sources: PDF pp. 1–2 and 10–12; print pp. 4106–4107 and 4115–4117; Abstract,
Sec. I-B, Sec. IV-D, and Conclusion.

## Provisional causal-claim map

The paper's central explanations can be written as `B > A because Z exploits
S`. This notation distinguishes a reported comparison from the mechanism said
to explain it.

| ID | `B > A` | Added capability `Z` | task structure `S` | Source locator | What the paper reports |
|---|---|---|---|---|---|
| M1 | deep autoencoders > shallow/static anomaly detectors | depth learns representative, sophisticated patterns | hierarchical or nonlinear structure in consumption profiles | PDF pp. 1–2, 11–12; print pp. 4106–4107, 4116–4117; Abstract, Sec. I-A/B, Conclusion | aggregate detector metrics |
| M2 | LSTM-SAE > FC-SAE and LSTM-VAE > FC-VAE | recurrent seq2seq modeling exploits temporal correlation | order-dependent structure across 48 half-hour readings | PDF pp. 1–2, 5, 10–12; print pp. 4106–4107, 4110, 4115–4117; Sec. I-B, III-A.2, IV-D | consistent DR/FA improvements on both datasets |
| M3 | VAE > SAE | latent distribution and reconstruction probability capture variance | normal and anomalous profiles may share a mean but differ in variability | PDF pp. 2, 5–7, 11; print pp. 4107, 4110–4112, 4116; Sec. I-B, III-B, IV-D | consistent DR/FA improvements for FC and LSTM variants |
| M4 | LSTM-AEA > LSTM-SAE/VAE | attention relieves a fixed-context bottleneck and weights important time steps | long-range or localized sequence information | PDF pp. 2, 7–9, 11–12; print pp. 4107, 4112–4114, 4116–4117; Sec. I-B, III-C, IV-D | best aggregate and per-attack metrics |
| M5 | benign-only anomaly detection detects unseen attacks | deviation from learned benign patterns, without malicious training examples | attack-induced departures shared across unseen attack types | PDF pp. 1–4, 11–12; print pp. 4106–4109, 4116–4117; Sec. I-A/B, II, Conclusion | performance on SGCC labels and six synthetic ISET attacks |

### What the printed experiments do and do not identify

For a causal explanation to be identified, later work would need to establish
six links separately:

1. `S` is present in the evaluated task;
2. `S` materially affects the target;
3. baseline `A` lacks the relevant capability;
4. component `Z` supplies that capability;
5. trained model `B` actually uses it; and
6. that use causes a fair paired advantage.

The source reports end-to-end performance comparisons, but it does not show
structure-destruction tests, component ablations under matched capacity,
learned-behavior inspection, or paired uncertainty. Moreover, the compared
systems use different depths, widths, activations, optimizers, dropout, loss or
score types, and thresholds. The printed metrics therefore show an association
between complete configurations and performance; by themselves they do not
identify the explanatory component.

This is a source-design observation, not a finding that any mechanism is
absent.

## Source-only red flags and feasibility questions

### 1. The result pattern is a total ordering

**OBSERVED:** In Tables II and III, every proposed model appears in the same
order—FC-SAE, LSTM-SAE, FC-VAE, LSTM-VAE, LSTM-AEA—for every displayed metric,
with FA reversed. Table IV preserves that order at every data size, and every
model improves monotonically as data size grows. Table V preserves the order
for both DR and FA on each of six attacks. No variance, seed count, sample
count, confidence interval, or failed run is reported.

This is unusual enough to motivate audit, but it is not evidence of selection
or fabrication. Several reported metrics are also algebraically dependent:
the paper defines `SP = 100 - FA`, `ACC = (DR + SP)/2`, and F1 as the harmonic
mean of DR and PR. Their agreement is therefore not independent corroboration.

### 2. The synthetic task may admit trivial shortcuts

**INFERRED FROM THE PRINTED ATTACKS:** Attacks (1) and (2) directly reduce
amplitude; attack (3) inserts long zero intervals; attack (4) makes an entire
day constant; and attack (5) changes both level and within-day form. These may
be separable using total energy, zero count, range, variance, or other
zero-parameter/simple statistics. Attack (6), reversal, preserves the day's
multiset and total and is the clearest printed probe of order sensitivity.

Consequently, high aggregate accuracy does not by itself show that depth,
recurrence, a variational latent space, or attention was needed. A triviality
floor and structure-preserving/destruction controls are needed before making
that attribution.

### 3. Output domains conflict with standardized targets

**STATIC FEASIBILITY OBSERVATION:** The paper first standardizes features to
zero mean and unit variance, then reports Softmax output for the fully
connected autoencoders and sigmoid output for the LSTM autoencoders (PDF
pp. 4, 9–10; print pp. 4109, 4114–4115; Sec. II-C, Algorithm 6, Table I).

For 48-dimensional output, Softmax constrains every reconstruction to the
simplex

`Delta = {x_hat : x_hat_i >= 0 and sum(x_hat_i) = 1}`,

while a sigmoid constrains it to the unit box `(0,1)^48`. A generally
standardized profile contains negative values and need not sum to one. Thus,
for any such input `x`, the best possible per-coordinate MSE is bounded below
by its squared distance to the relevant output set:

`MSE(x, x_hat) >= dist(x, Delta)^2 / 48`

for Softmax, and analogously by `dist(x, [0,1]^48)^2 / 48` for sigmoid.

This is the geometric “ceiling” intuition in a defensible form: the declared
output family cannot exactly reconstruct general standardized inputs,
regardless of compute or optimizer. It does **not** prove that the printed
detection rates are impossible; constrained reconstruction errors could still
rank benign and malicious samples. The next question is whether this domain
mismatch dominates the score and whether all architectures are merely
exploiting the same simple separability.

### 4. The temporal and attention claims are not stressed cleanly

The sequence is fixed at 48 time steps, yet attention is motivated as a remedy
for fixed-context limitations and long sequences. The source gives no sequence
length comparison, order shuffle, time-localized capability witness, attention
weight inspection, or matched attention ablation. The task may contain useful
temporal structure, especially attack (6), but the aggregate tables cannot say
whether the trained models used it.

### 5. Full configurations, not isolated components, are compared

Table I changes depth, optimizer, dropout, hidden activation, and output
activation across model families. SAE and VAE also use different anomaly
scores, and AEA is not a variational model. Therefore:

- LSTM versus FC is not recurrence alone;
- VAE versus SAE is not latent variance alone; and
- AEA versus VAE is not attention alone.

A model can genuinely outperform another while the explanation attached to
the difference remains unidentified.

## Literal contradictions and execution-blocking ambiguities

These observations require resolution in Phase 3. No primary completion is
selected here.

1. **Attack (3) interval is internally inconsistent.** The source defines a
   positive length `t_l = rand(4,24)` but prints `t_f = t_i - t_l`, making the
   final time earlier than the initial time under ordinary interpretation.
   The piecewise interval `[t_i,t_f]` therefore cannot represent the described
   4–24 hour bypass without a repair. (PDF p. 3; print p. 4108; (3).)
2. **VAE score direction contradicts the shared decision rule.** The VAE
   section says low reconstruction probability indicates anomaly. The shared
   rule later says a score greater than the threshold is malicious and below
   it is benign, explicitly grouping reconstruction probability with MSE.
   (PDF pp. 6 and 9; print pp. 4111 and 4114; Sec. III-B and III-C/D.)
3. **The VAE objective contains variable/distribution mismatches.** Equation
   (9) prints `DKL(q(k|x) || p(x))`, although the two terms are over different
   variables, and then states a bound against `log p(k)` rather than the
   preceding `log p(x)`. A standard-form correction may be obvious to an
   expert, but it is still a repair rather than the printed expression. (PDF
   p. 6; print p. 4111; (8)–(9).)
4. **A valid variance parameterization is omitted.** Algorithms 3–4 describe
   producing `sigma_x^2` through a generic activation, while the listed
   activations do not ensure a strictly positive variance in every case. No
   variance-head transform, clipping rule, numerical constant, or decoder
   likelihood parameterization is specified. (PDF pp. 6–7 and 10; print
   pp. 4111–4112 and 4115; Algorithms 3–4, Table I.)
5. **Threshold construction is not executable as written.** “Median of the
   interquartile range of the ROC curve” does not uniquely map ROC points or
   score thresholds to one scalar. The paper lists final thresholds but does
   not define the algorithm that produces them. (PDF pp. 9–10; print
   pp. 4114–4115; Sec. III-D and IV-B.)
6. **Validation data are unclear.** Hyperparameters and thresholds are said to
   use cross-validation over ISET `X_TR`, but anomaly-detector `X_TR` was
   previously defined as benign-only `B1`, while selection maximizes DR and
   draws ROC curves, both requiring malicious labels. The validation
   population and its separation from test attacks are not specified. (PDF
   pp. 4 and 9–10; print pp. 4109 and 4114–4115.)
7. **The customer-disjoint split is not fully bound to malicious examples.**
   The paper says all customers receive all six attacks and writes the test set
   as `B2` concatenated with `M`; it does not explicitly restrict `M` to the
   customers represented by `B2`. The intended customer identity mapping is
   therefore material and underdetermined. (PDF pp. 3–4; print pp. 4108–4109;
   Sec. II-B/C.)
8. **Sample unit and cardinality are inconsistent or incomplete.** A row is a
   48-value daily profile, Table IV says full `|X_TR|` is 60 million, and the
   text later says online decisions operate on “individual readings.” The
   exact statistical unit counted by 60 million and scored by the detector is
   not stated consistently. (PDF pp. 3–5 and 11; print pp. 4108–4110 and 4116.)
9. **Layer-count search and reported optima use different apparent domains.**
   The search range is printed as `L={2,3,4,5}`, but Table I reports 8 layers
   for both FC models and 6 for AEA. This might count encoder and decoder
   together in one place and separately in another, but that convention is not
   stated clearly enough to execute without completion. (PDF pp. 9–10; print
   pp. 4114–4115; Algorithm 6, Table I.)
10. **Preprocessing order uses information outside training.** The literal
    prose normalizes benign and malicious classes before the split and applies
    ADASYN to the test set. This is executable, but the fitting
    population, ADASYN neighbor population, and relationship to the claim of
    completely unseen test data require explicit preservation rather than a
    silent conventional repair. (PDF pp. 3–4 and 10; print pp. 4108–4109 and
    4115; Sec. II-C and IV-D.)

## Other material omissions

The publication does not fully specify:

- the exact SGCC file/version, ISET trial groups, customer IDs, date range,
  missing-value policy, daylight-saving handling, and exclusion criteria;
- whether normalization is global, per feature, per customer, per dataset, or
  fit separately for each model;
- whether `rand(a,b)` is continuous or discrete and whether each draw is per
  dataset, customer, day, or reading;
- batch size, learning rate in the executed configuration, epoch limit,
  convergence criterion, initialization, random seeds, and number of repeats;
- exact tensor orientation, latent widths, decoder inputs or teacher forcing,
  masking, attention dimensions, and alignment-network structure;
- KL/reconstruction scaling, Monte Carlo sample count, probability density
  aggregation, and numerical stabilization for VAE scoring;
- the exact sequential-search tie rule and whether each selected value is held
  fixed for later search stages;
- the exact ROC threshold algorithm and whether ISET-derived thresholds are
  reused unchanged on SGCC;
- test-set counts after ADASYN, the unit of bootstrap/statistical variation,
  or any uncertainty estimate; and
- enough hardware/software detail to interpret the training-time claim beyond
  the use of the Keras Sequential API.

## Breadth-first sandbox questions

If separately authorized, Phase 2 should ask cheap, capability-discriminating
questions before touching full data or production code. The candidate breadth
is:

1. **Domain sanity:** Can each printed output activation produce values in the
   target domain? Compute the analytic projection lower bound and check one
   finite update on hand-sized standardized profiles.
2. **Triviality floor:** On toy profiles shaped like attacks (1)–(6), compare
   total, mean, variance, range, zero count, and an order-insensitive rule with
   the smallest recognizable reconstruction model.
3. **Temporal witness:** Construct paired sequences with identical marginals
   but different order. An order-insensitive baseline should tie; a functioning
   sequence model should be able to separate or reconstruct the relevant
   order-dependent structure.
4. **Variance witness:** Construct normal/anomalous samples with matched means
   and controlled variance differences. Test whether the claimed VAE scoring
   path has a capability unavailable to a matched deterministic score.
5. **Attention witness:** Use a short sequence with one movable, localized
   informative event; compare a matched recurrent model with and without
   attention and inspect the learned weights.
6. **Score direction:** On a few explicit Gaussian examples, verify whether
   reconstruction probability must be thresholded low or high for anomaly.
7. **Threshold semantics:** Feed a hand-sized score/label vector through every
   materially reasonable reading of the printed IQR/ROC wording and determine
   whether any reading uniquely produces the published kind of scalar.
8. **One-example execution:** Verify shapes, parameter domains, loss finiteness,
   and overfit capability for the smallest version of each model family.

These are discovery `X` probes only. Their purpose is to refine the source
questions and catch misunderstandings. They cannot be promoted retroactively
to reproduction, mechanism confirmation, or attainability evidence.

## What is not concluded

- No reported number has yet been reproduced or contradicted.
- No architecture has been shown to lack or possess the claimed learned
  mechanism.
- The geometric reconstruction bound does not bound DR, FA, AUC, or the
  reported target pattern.
- The tidy ordering does not establish selection, fabrication, intent, or an
  undocumented implementation.
- No prior project implementation or outcome has been admitted as evidence.
- No primary repair for Attack (3), VAE direction/objective, validation,
  thresholding, sample identity, or layer count has been selected.

## Phase 1 disposition

- **VERIFIED:** source identity and complete-page inspection.
- **OBSERVED:** the complete reported pattern, causal explanations, source
  contradictions, omissions, and algebraic/output-domain constraints recorded
  above.
- **INFERRED:** much of the synthetic task may be separable through simple
  statistics, but this has not been tested.
- **OPEN:** whether an ordinary source-supported completion can execute and
  reproduce the complete target pattern.
- **OPEN:** whether the data contain and the trained systems use the claimed
  temporal, variance, depth, and attention mechanisms.
- **OPEN:** whether the target lies inside a declared empirical performance
  envelope.

Phase 1 is complete. Phase 2 remains a disposable discovery sandbox and needs
separate authorization under the governing plan's current checkpoint scope.
