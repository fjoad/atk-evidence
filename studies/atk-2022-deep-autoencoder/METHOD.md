# Paper 1 method freeze

**Paper:** Abdulrahman Takiddin, Muhammad Ismail, Usman Zafar, and
Erchin Serpedin, “Deep Autoencoder-Based Anomaly Detection of Electricity
Theft Cyberattacks in Smart Grids,” *IEEE Systems Journal*, 16(3), 4106–4117,
2022. DOI: 10.1109/JSYST.2021.3136683.

**Local source:** ignored file
`papers/his/Deep_Autoencoder-Based_Anomaly_Detection_of_Electricity_Theft_Cyberattacks_in_Smart_Grids.pdf`

**Source identity:** 12 pages; SHA-256
`f3098e0c27ee19b27bea026aedc3d10e5dbb0c46f5cd01ed5bd5c05b7dcf850f`.

**Status:** Source-derived specification. A second independent audit on
2026-08-11 fingerprinted the PDF, extracted its text, and visually inspected
all 12 rendered pages before this file was opened. That pass confirmed the
overall reconstruction, corrected the benchmark count, and added several
previously omitted mathematical and reported-metric contradictions. Historical
code, contracts, and results were not treated as source authorities.

## What the paper actually tests

The paper compares supervised classifiers and anomaly detectors on two
electricity-consumption datasets:

1. SGCC, which already contains benign and malicious customer records.
2. ISET/CER, which contains benign half-hourly readings. The paper generates
   six malicious variants of every customer’s readings.

The proposed anomaly detectors are trained only on benign profiles. A daily
profile is reconstructed, its anomaly score is thresholded, and the resulting
labels are evaluated against benign and malicious test profiles. The proposed
models are FC-SAE, LSTM-SAE, FC-VAE, LSTM-VAE, and LSTM-AEA.

The first executable reproduction lane is ISET with FC-SAE. ISET directly
supports the stated 48-value daily input. SGCC has roughly three years of daily
values per customer while the paper still specifies 48 input neurons and gives
no conversion rule; Table II therefore requires separate interpretations.

## Paper flow at a glance

```text
SGCC: labeled customer histories -----------------------------+
                                                               |
ISET: benign half-hour readings -> 48-value days -> attacks 1-6|
                                                               v
              merge customers -> normalize before splitting
                                  |
              +-------------------+--------------------+
              |                                        |
      anomaly detectors                         supervised detectors
      train on benign B1                        B+M -> ADASYN -> split
      test on B2+all M -> ADASYN                 train and test on labels
              |                                        |
              +-------------------+--------------------+
                                  v
             ISET cross-validation -> Table-I model/settings
                                  |
                       train -> score -> threshold
                                  |
        Table II (SGCC) | Table III (ISET) | Table IV (size/time)
                                  |
                    Table V (one attack at a time)
```

The arrows above are the paper's stated ordering, not an endorsement of it.
The detailed source status of every consequential arrow is below. A compact
human-readable companion is [`PAPER_WORKFLOW.md`](PAPER_WORKFLOW.md).

## Numerical targets

The source transcriptions are stored in:

- [`reported/table_1.csv`](reported/table_1.csv): optimized architecture and
  training choices;
- [`reported/table_2.csv`](reported/table_2.csv): SGCC performance;
- [`reported/table_3.csv`](reported/table_3.csv): ISET performance;
- [`reported/table_4.csv`](reported/table_4.csv): ISET training time and ACC at
  half, three-quarter, and full training size; and
- [`reported/table_5.csv`](reported/table_5.csv): ISET DR and FA for each
  attack separately.

The first anchor is the Table III FC-SAE row:

| DR | FA | SP | PR | ACC | F1 | AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 81 | 15 | 85 | 81 | 83 | 81 | 81 |

Here `ACC` is not ordinary accuracy. The paper defines it as the arithmetic
mean of DR and specificity, i.e. balanced accuracy.

The complete proposed-model targets are:

| Dataset/table | Model | DR | FA | SP | PR | ACC | F1 | AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SGCC / II | FC-SAE | 83 | 14 | 86 | 83 | 84.5 | 83 | 83 |
| SGCC / II | LSTM-SAE | 86 | 12 | 88 | 87 | 87 | 86.5 | 85 |
| SGCC / II | FC-VAE | 90 | 9 | 91 | 91 | 90.5 | 90.5 | 88 |
| SGCC / II | LSTM-VAE | 93 | 6 | 94 | 93 | 93.5 | 93 | 90 |
| SGCC / II | LSTM-AEA | 96 | 4 | 96 | 95 | 96 | 95.5 | 93 |
| ISET / III | FC-SAE | 81 | 15 | 85 | 81 | 83 | 81 | 81 |
| ISET / III | LSTM-SAE | 85 | 13 | 87 | 85 | 86 | 85 | 82 |
| ISET / III | FC-VAE | 88 | 11 | 89 | 89 | 88.5 | 88.5 | 85 |
| ISET / III | LSTM-VAE | 91 | 7 | 93 | 91 | 92 | 91 | 86 |
| ISET / III | LSTM-AEA | 94 | 5 | 95 | 93 | 94.5 | 93.5 | 90 |

Tables II and III also contain six benchmark rows apiece. Their complete
values remain in the checked source transcriptions above.

Table IV reports the following `(training minutes, ACC)` pairs:

| Model | 0.5 × training data | 0.75 × training data | full training data |
|---|---:|---:|---:|
| FC-SAE | (72, 70) | (97, 78.5) | (137, 83) |
| LSTM-SAE | (90, 75) | (127, 83) | (183, 86) |
| FC-VAE | (81, 79.5) | (103, 86) | (141, 88.5) |
| LSTM-VAE | (97, 83) | (132, 90) | (188, 92) |
| LSTM-AEA | (102, 86) | (142, 93) | (193, 94.5) |

The prose calls full `|X_TR|` “60 million.” That is consistent with counting
scalar half-hour readings, not the daily profile rows that the model consumes.

Table V reports attack-specific DR and FA:

| Model/metric | A1 | A2 | A3 | A4 | A5 | A6 | AVG |
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

### Static consistency of the reported metrics

This is a source-level arithmetic audit, not a reproduction result. The
paper's own definitions imply two identities:

```text
F1 = 2 * DR * PR / (DR + PR)
p  = PR * FA / (DR * (1 - PR) + PR * FA)
```

Here `p` is the malicious-class prevalence implied by a row's reported DR, FA,
and PR. If every row in a table uses the paper's single table-specific test
label vector, all rows must imply the same `p`.

- Table II reports Naive Bayes `DR=75`, `PR=75`, and `F1=77`. The first two
  values force `F1=75`, not 77.
- Table II rows imply prevalences from 39.02% to 50.28%; Table III rows imply
  40.00% to 50.28%.
- Even allowing each displayed DR, FA, and PR to differ from its underlying
  value by a generous ±0.5 percentage point, there is no common prevalence.
  In Table II, FC-VAE requires at least 47.22% while Naive Bayes permits at
  most 40.56%. In Table III, the corresponding bounds are 47.73% and 41.44%.

Therefore the complete DR/FA/PR patterns in Tables II and III cannot all arise
from one common test-label population using the formulas printed in the paper
and ordinary rounding. This establishes an internal numerical inconsistency;
it does not establish its cause or imply intent. The row-level calculations
are preserved in
[`results/reported_metrics_audit.csv`](results/reported_metrics_audit.csv) and
[`results/reported_metrics_audit.json`](results/reported_metrics_audit.json).

## ISET data and sample construction

### Source statements

Section II-B says ISET contains readings from smart meters at “around 3000
residential units,” sampled every 30 minutes for one and a half years, yielding
around 25,000 reports per customer.

Section III-A states that the autoencoder input has 48 neurons. Section II-C
says one row is an energy-consumption profile “along the day.” The resulting
model sample is therefore a 48-value daily profile:

```text
customer × calendar day → [slot 1, slot 2, …, slot 48]
```

The paper gives no missing-value, duplicate-slot, daylight-saving-time, or
incomplete-day policy. It gives no rule for selecting exactly 3,000 meters from
the named data.

### Six generated attacks

For customer `c`, day `d`, time `t`, and true consumption `E_c(d,t)`, Section
II-B defines:

1. **Fixed partial reduction, Eq. (1):**
   `f1(E) = alpha E`, with `alpha = rand(0.1, 0.8)` fixed “for all samples.”
2. **Dynamic partial reduction, Eq. (2):**
   `f2(E) = beta(d,t) E`, with a new `beta = rand(0.1, 0.8)` for each reading.
3. **Selective bypass, Eq. (3):** set readings to zero over
   `[t_i(d), t_f(d)]`, where `t_i = rand(0,19)`, `t_l = rand(4,24)`, and the
   printed endpoint is `t_f = t_i - t_l`.
4. **Daily mean, Eq. (4):** `f4(E_c(d,t)) = mean(E_c(d))`.
5. **Randomized daily mean, Eq. (5):**
   `f5(E_c(d,t)) = beta(d,t) mean(E_c(d))`.
6. **Reversal, Eq. (6):** `f6(E_c(d,t)) = E_c(d,T-t+1)`.

The paper applies all six functions to every customer’s benign matrix, so each
customer has six malicious matrices.

Eq. (3) is not executable as an interval with the stated meaning. Because
`t_l > 0`, the printed `t_f = t_i - t_l` is before `t_i`. The variables are
also defined in hours while the source and model use 30-minute values.
The paper does not define whether `rand(a,b)` is continuous or integer-valued,
whether endpoints are inclusive, or how a closed hourly interval maps to the
48 half-hour slots. It reports no random seed or attack-regeneration policy.

## Printed anomaly-detector preparation order

Section II-C prints this sequence:

1. Merge all customers’ benign daily profiles into class `B_raw`.
2. Concatenate all six malicious matrices for all customers into `M_raw`.
3. Normalize **both classes before splitting** to zero mean and unit variance,
   producing `B` and `M`.
4. Split `B` in a 2:1 ratio into disjoint `B1` and `B2`.
5. Set training data `X_TR = B1`.
6. Concatenate `M` with `B2` and label benign `0`, malicious `1`.
7. Run ADASYN **on that test set** to oversample its minority class: malicious
   in SGCC and benign in ISET.
8. The balanced result is `X_TST, Y_TST`.

This is the printed `P` path even though fitting a transform before the split
and synthesizing test examples are statistically improper.

Two internal conflicts must remain visible:

- The opening of Section II-C says the split is over customers and that train
  customers do not appear in test. The detailed paragraph then says “split B
  over the rows (customers).” A row was just defined as one daily profile, so
  “row” and “customer” are not interchangeable.
- The detailed test construction uses all of `M`, which Section II-B defined
  from all customers. It therefore puts malicious derivatives of training
  customers in test, contradicting the claim that all test customers are
  unseen.

For supervised detectors, the paper instead prints:

1. concatenate benign and malicious classes for all customers;
2. apply ADASYN before splitting;
3. split the balanced rows 2:1 into train and test; and
4. apply “the same normalized feature scaling.”

The last sentence conflicts with the preceding anomaly-detector paragraph,
which places scaling before splitting, and does not establish whether the
supervised split is customer-disjoint.

## Models

### FC-SAE anchor

Sections III-A and IV-C, Fig. 3, Algorithm 1, and Table I jointly state:

- input: 48 values;
- encoder widths: 400, 300, 200, 100;
- decoder widths: 100, 200, 300, 400;
- optimized total layer count printed in Table I: `L* = 8`;
- hidden activation: sigmoid;
- output activation: Softmax;
- dropout: 0.4;
- optimizer: Adam;
- loss and anomaly score: mean squared reconstruction error;
- anomaly threshold: 0.58; and
- anomaly direction: score greater than threshold.

The prose and Fig. 3 show a latent layer in addition to the encoder and decoder
hidden layers, but no latent width is ever specified. The only completely
specified executable layout treats the final 100-wide encoder output as the
latent representation and instantiates all eight listed hidden transforms:

```text
48 → 400 → 300 → 200 → 100 → 100 → 200 → 300 → 400 → 48
                   latent representation
```

The Softmax output is preserved in the printed track. It constrains the 48
outputs to nonnegative values summing to one even though the paper has already
standardized the reconstruction targets to zero mean and unit variance. That
domain mismatch is a property of the printed configuration, not something to
silently repair.

Algorithm 1 says “SGD” with a learning rate `eta` and shows a full-dataset
mean-gradient update. Table I later selects Adam. Batch size, learning rate,
epochs, initialization, shuffle policy, and an operational definition of “not
converged” are absent.

### Other proposed models

| Model | Encoder widths | Mirrored decoder | `L*` | Optimizer | Dropout | Hidden | Output | Score |
|---|---|---|---:|---|---:|---|---|---|
| LSTM-SAE | 500, 300 | 300, 500 | 4 | Adam | 0.2 | Sigmoid | Sigmoid | MSE |
| FC-VAE | 500, 400, 300, 100 | 100, 300, 400, 500 | 8 | Adam | 0.4 | ReLU | Softmax | reconstruction probability |
| LSTM-VAE | 400, 300 | 300, 400 | 4 | SGD | 0 | Tanh | Sigmoid | reconstruction probability |
| LSTM-AEA | 500, 300, 200 | 200, 300, 500 | 6 | SGD | 0 | Sigmoid | Sigmoid | MSE |

The VAE section defines a low reconstruction probability as anomalous. The
generic detection paragraph on page 4114 instead says a reconstruction
probability greater than the threshold is malicious. The former is consistent
with the VAE explanation; the latter is a direction contradiction.

The VAE score is not fully executable: the paper does not give the decoder
variance parameterization, number of Monte Carlo samples, probability
aggregation details, numerical scale, or how thresholds 0.43/0.47 correspond
to that probability.

The VAE derivation contains two additional source-level problems:

- Eq. (9) places `q(k|x)` and `p(x)` inside a KL divergence even though they
  are distributions over different variables, and the following sentence
  compares the lower bound with `log p(k)` rather than `log p(x)`. This is not
  the standard ELBO stated by Eq. (8).
- Algorithms 3 and 4 call their second encoder output a variance `sigma_x^2`
  but apply only a generic activation plus an unconstrained bias. No
  exponential, Softplus, clipping, or other positivity rule is stated, so the
  printed computation does not guarantee a valid variance.

Eq. (10) still supplies a recognizable squared-reconstruction-plus-KL training
objective, but it does not repair the missing executable probability score.

## Hyperparameter selection and validation

Algorithm 6 prints a sequential, not Cartesian, search:

1. initialize optimizer=SGD, dropout=0, hidden=ReLU, output=Softmax;
2. search layer count and width;
3. hold those winners fixed and search optimizer;
4. hold those fixed and search dropout;
5. hold those fixed and jointly search hidden/output activations.

Section IV-C gives:

- layers `{2,3,4,5}`;
- widths `{100,200,300,400,500}`;
- optimizers `{SGD, Adam, Adamax, RMSprop}`;
- dropout `{0,0.2,0.4,0.5}`;
- hidden activations `{ReLU,Sigmoid,Linear,Tanh}`; and
- output activations `{Softmax,Sigmoid}`.

The search is not reproducible as written:

- Table I reports FC `L*=8` and AEA `L*=6`, outside `{2,3,4,5}`. The likely
  explanation is that the search range counts layers per side while Table I
  counts encoder plus decoder, but the notation and Algorithm 6 do not say so.
- The selection rule merely says to record DR and FA and choose the improved
  DR; it gives no tie-breaker or scalar tradeoff with FA.
- The search uses cross-validation “over `X_TR` of the ISET dataset.” For
  anomaly detectors `X_TR` is explicitly benign-only, so DR and FA cannot both
  be calculated without an unstated malicious validation population.
- The number and identity of folds, refitting protocol, and random seeds are
  absent.

The first anchor therefore uses the final printed Table I FC-SAE setting
directly. It does not attempt to reverse-engineer the unreported search.

## Thresholds and evaluation

The paper reports anomaly thresholds:

| Detector | ARIMA | One-class SVM | FC-SAE | LSTM-SAE | FC-VAE | LSTM-VAE | LSTM-AEA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Threshold | 0.58 | 0.45 | 0.58 | 0.61 | 0.43 | 0.47 | 0.51 |

It says these arise by dividing the ROC curve into three quartiles and “taking
the median of IQR.” This does not define a unique threshold: an ROC curve
contains score thresholds and pairs of rates, “median of IQR” is not a standard
ROC selection operation, and the paper supplies neither validation identities
nor the underlying scores. The printed constants are executable; their
derivation is not.

For malicious as the positive class, the exact reported formulas are:

```text
DR = TP / (TP + FN)
FA = FP / (FP + TN)
SP = 1 - FA
PR = TP / (TP + FP)
ACC = (DR + SP) / 2
F1 = harmonic_mean(DR, PR)
AUC = area under the ROC curve
```

The paper reports percentages. We calculate fractions internally and multiply
by 100 only for display.

The prose defining precision is itself contradictory: it calls PR the fraction
of correctly detected malicious readings among *all malicious readings*, which
describes DR/recall, while the immediately printed formula is
`TP/(FP+TP)`, the standard precision. The explicit formula governs the audit.

## Table IV and timing

Table IV trains each proposed detector with nested 0.5, 0.75, and full
fractions of ISET `X_TR` and reports training minutes plus ACC. The prose says
full `|X_TR|` is 60 million and online testing takes 1–2 seconds.

No CPU/GPU model, GPU count, memory, software version, precision, batch size,
epoch count, stopping epoch, timing boundary, warm-up, repetitions, or
dispersion is reported. The 1–2 second claim also says detection is on an
“individual reading,” while the model input is a 48-reading day. We can measure
our own fit and scoring times but cannot directly reproduce hardware-dependent
wall time or identify its unit from the paper alone.

## Frozen straight-through executable completion: `P0-ISET-FCSAE`

This is the first declared, pre-outcome executable reading. It preserves every
*executable* printed step and fills only settings without which execution is
impossible. It is therefore a paper-primary `P+I` completion, not a claim that
the source contains a fully executable literal `P`: the unmodified Attack 3
endpoint remains preserved as a `NON-EXECUTABLE` source outcome.

### Data

1. Use every meter labeled residential in the named ISET/CER source. The paper
   gives no reproducible 3,000-meter selection rule. A seeded 3,000-meter
   interpretation remains branch `I-DATA-3000`.
2. Form one row per meter/calendar-day from ordered half-hour slots 1–48.
   Exclude a day unless those 48 slots occur exactly once. No imputation or DST
   repair is introduced in `P0`.
3. Generate attacks from raw, unnormalized profiles for every customer.
   Use seed 11 for all omitted random draws.
4. Resolve attack randomness as follows:
   - Attack 1: one `Uniform[0.1, 0.8]` alpha per customer’s complete malicious
     matrix.
   - Attacks 2 and 5: one independent `Uniform[0.1, 0.8]` beta per half-hour
     reading.
   - Attack 3: draw integer start hour 0--19 and integer duration 4--24,
     inclusive; minimally repair subtraction to addition; map each hour to two
     half-hour slots; zero the half-open slot interval; and clip an endpoint
     past slot 48.
   - Attacks 4 and 6: apply directly by daily row.
5. Concatenate the six attack matrices from all customers into `M_raw`.
6. Fit one feature-wise standardizer to the concatenation of `B_raw` and
   `M_raw`, before splitting, and transform both classes. “Feature-wise” means
   one mean and standard deviation for each of the 48 time positions.
7. Randomly split unique customer IDs 2:1 with seed 11. Put all profiles of a
   customer into its side. Set `X_TR=B1`.
8. Construct the printed test population as `B2 + M` where `M` still contains
   attacks from all customers. Preserve source identities so this contradiction
   is measurable.
9. Apply ADASYN with its library-default neighborhood and seed 11 to that test
   population, oversampling benign rows. Retain both original and synthetic-row
   flags.

### FC-SAE

1. Build the exact specified layout
   `48-400-300-200-100-100-200-300-400-48`.
2. Treat the final encoder width 100 as the latent representation; do not add
   an unspecified extra latent layer.
3. Apply sigmoid to every hidden dense layer, Softmax to the 48-wide output,
   and dropout 0.4 after every hidden activation during training.
4. Optimize profile-level mean squared reconstruction error with Adam,
   learning rate 0.001, float32, and Keras-compatible Glorot/bias defaults.
5. Use shuffled mini-batches of 32. Train for at most 100 epochs and define
   “converged” as no training-loss improvement of at least `1e-4` for five
   consecutive epochs. Keep the lowest-training-loss weights. These are
   explicit completion choices, not paper statements.
6. For every test profile, compute mean squared error over its 48 positions.
   Predict malicious iff MSE is strictly greater than 0.58.
7. Save all scores, predictions, labels, original/synthetic flags, customer/day
   identities, loss history, actual layer inventory, hashes, and load/fit/score/
   total wall times.

### Table III, IV, and V identity

- Table III uses the `P0` test population and metrics above.
- Table IV trains fresh instances on nested 0.5, 0.75, and full fractions of
  `B1`, reports both daily-profile and scalar-reading counts, and evaluates the
  same frozen test population.
- The most literal common-model Table V reading applies the already trained
  model separately to all benign rows and each corresponding one-attack
  malicious matrix, with no ADASYN because those two populations are naturally
  equal in size. It follows that the benign rows, scores, threshold, and FA are
  identical for attacks 1–6. The attack-varying FA values printed in Table V
  cannot arise from this common-model/common-benign experiment.

Seeds 22 and 33 repeat the same method only after seed 11 passes the runnable
and eligibility gates. No setting may be changed because of seed-11 metrics.

## Material interpretation branches

These branches are recorded now, before the new implementation produces
outcomes. The first anchor remains `P0`; later branches vary one material
choice at a time.

| ID | Material question | Paper-consistent readings to test |
|---|---|---|
| I-DATA-POP | “around 3000” | all residential meters; deterministic seeded 3,000 |
| I-DAY | 48 slots / unreported DST | strict 1–48 only; discard extras; average duplicate slots; interpolate to a 48-grid |
| I-A1 | fixed alpha “for all samples” | one per dataset; customer matrix; customer-day |
| I-A2 | dynamic beta | independent half-hour; one beta shared by each hourly pair |
| I-A3-END | impossible endpoint | printed subtraction as a documented failure; addition+clip; addition+wrap; redraw a valid end |
| I-A3-SLOTS | hour variables and closed interval on half-hours | convert hours to two slots with half-open duration; closed endpoint; apply printed integer range directly to 48 positions |
| I-RANDOM | undefined `rand(a,b)` and endpoints | continuous uniform for factors; integer uniform for time values; alternative endpoint conventions |
| I-ATTACK-SEED | random attack reuse | fixed once per prepared data seed; regenerate per model seed; regenerate per experiment |
| I-SPLIT | “rows (customers)” | customer-disjoint; daily-row-random |
| I-MPOP | all `M` versus unseen customers | attacks from all customers; attacks from held-out customers only |
| I-NORM-POP | normalize both classes | joint `B+M`; independently fit `B` and `M`; fit benign only and apply to malicious |
| I-NORM-AXIS | “all feature values” | per time-position; one global scalar; per customer; per daily row |
| I-ADASYN | printed test balancing | test ADASYN as printed; no test resampling; training-only resampling where labels exist |
| I-ADASYN-ORDER | supervised paragraph | normalize then ADASYN; ADASYN then normalize |
| I-LATENT | Fig. 3 versus specified widths | final encoder is latent; add a distinct latent layer with declared finite widths |
| I-LAYERS | search range versus Table I | range counts total layers; range counts layers per encoder/decoder side |
| I-DROPOUT | rate but no placement | after every hidden layer; encoder only; recurrent/inter-layer placement |
| I-TRAIN | missing fit controls | batches 32/512; epochs 10/30/100; fixed versus predeclared convergence/refit policies; optimizer-default learning-rate neighborhood |
| I-VALID | benign-only cross-validation | benign carve-out plus generated attacks; held-out customer validation; bypass search and use Table I |
| I-SEARCH-WIDTH | Algorithm 6 uses one `N_l` but Table I widths differ | uniform-width staged search; per-layer coordinate search; direct Table I replay |
| I-THRESH | non-executable ROC/IQR phrase | printed constant; finite deterministic formalizations frozen before execution |
| I-THRESH-SCOPE | derivation only described on ISET | transfer ISET constants to SGCC; derive dataset-specific constants |
| I-TV | varying FA | common model/common benign; retrain per attack; resample benign per attack; both |
| I-TV-POP | “already balanced” | all benign versus all one-attack rows; held-out benign versus matching held-out attacks |
| I-TV-SIZE | evaluation size absent | full balanced population; deterministic balanced 3,000-per-class subsets |
| I-TIME | `|X_TR|=60m` | scalar-reading cardinality; profile-row cardinality |
| I-SGCC | 1,034 days versus 48 inputs | separately freeze finite crop/window/resample/aggregate readings before Table II |
| I-SGCC-MISSING | missing daily values | drop incomplete customers; zero fill; train-derived interpolation/median; customer mean |
| I-LSTM-INPUT | “48 neurons” in a sequence model | 48 time steps × 1 feature; 1 time step × 48 features |
| I-LSTM-DECODER | ongoing decoder inputs/states absent | repeat latent; latent then zero; autoregressive reconstruction; top-only versus mirrored state transfer |
| I-ATTENTION | prose/Algorithm 5 topology | concatenate versus literal sum of context and reconstruction; finite decoder-conditioned alignment implementations |
| I-VAE-LATENT | latent distribution width absent | predeclared finite widths 2/8/16/32/48/100 |
| I-VAE-LOSS | Eq. (10) reduction | sum-squared error + KL; mean-MSE + KL |
| I-VAE-SCORE | reconstruction probability omissions | fixed/learned decoder variance; 1/10/100 draws; probability aggregation; lower-is-anomalous direction |
| I-SUPERVISED | classifier output absent | binary Softmax/categorical; binary sigmoid/BCE; seven attack-type classes where source labels permit |
| I-ARIMA | only `d=1,q=0` stated | finite `p` values; pooled/per-profile fit; residual-MSE/Gaussian-likelihood scores |
| I-SVM | “kernel=scale, Gamma=sigmoid” is API-invalid | minimal swap to kernel=sigmoid/gamma=scale; preserve printed pair as a documented failure |

Scientifically corrected controls (`C`) will be run separately after the
paper-consistent anchors. At minimum they include train-only scaling,
customer-disjoint malicious test identities, no synthetic test rows, validation
chosen without using final test labels, an output activation compatible with
standardized targets, and uncertainty over repeated seeds.

## Source instruction table

`EXACT` means directly executable as stated. `AMBIGUOUS` means multiple material
readings remain. `CONTRADICTORY` means two source statements cannot both hold.
`NON-EXECUTABLE` means a required operation is not defined well enough to run.

| Instruction | Locator | Status | Consequence |
|---|---|---|---|
| Use SGCC and ISET separately | §II, p. 4107 | EXACT | Two independent dataset lanes |
| ISET is half-hourly for ~1.5 years and ~3,000 residential units | §II-B1, p. 4108 | AMBIGUOUS | No exact meter selection |
| One row is a daily profile | §II-C, p. 4109 | EXACT | Daily model samples |
| Input has 48 neurons | §III-A1, p. 4109 | EXACT | 48 half-hour values |
| SGCC has ~3 years of daily values but models take 48 inputs | §II-A/§III-A, pp. 4107–4110 | NON-EXECUTABLE | Table-II sample construction is absent |
| Handle incomplete/DST days | not stated | NON-EXECUTABLE | Requires declared branch |
| Generate six attacks for every customer | §II-B2, Eqs. 1–6, p. 4108 | EXACT | Six malicious matrices/customer |
| Attack 1 alpha scope | Eq. 1 prose, p. 4108 | AMBIGUOUS | “Fixed … for all samples” has several scopes |
| Attack 2/5 beta granularity | Eqs. 2/5, p. 4108 | AMBIGUOUS | Hour versus half-hour source mismatch |
| Attack 3 endpoint is `t_i-t_l` | Eq. 3, p. 4108 | CONTRADICTORY | End precedes start |
| Attack 3 uses hours on a 48-step input | Eq. 3 and §III-A, pp. 4108–4110 | AMBIGUOUS | Requires hour-to-slot rule |
| `rand(a,b)` distribution and endpoint semantics | not stated | NON-EXECUTABLE | Factors and Attack-3 times require completion choices |
| Attack random draws/reuse across runs | not stated | NON-EXECUTABLE | Seed and regeneration schedule required |
| Normalize B and M before split | §II-C, p. 4109 | EXACT | Preserve pre-split scaling in `P` |
| Normalization population/axis | §II-C, p. 4109 | AMBIGUOUS | Scaler fit and axes absent |
| Split B 2:1 over customers | §II-C, pp. 4108–4109 | EXACT | Primary grouped split |
| Split B over “rows (customers)” | §II-C, p. 4109 | CONTRADICTORY | Daily rows are not customers |
| Train anomaly detector on B1 only | §II-C, p. 4109 | EXACT | Benign-only training |
| Test on B2 plus all M | §II-C, p. 4109 | EXACT | Includes attacks derived from train customers |
| All test customers are unseen | §II-C, p. 4108 | CONTRADICTORY | Cannot hold with all M |
| Apply ADASYN to anomaly test set | §II-C, p. 4109 | EXACT | Printed test synthesis |
| Supervised ADASYN/split/scaling order | §II-C, p. 4109 | AMBIGUOUS | Last scaling sentence conflicts with order |
| FC-SAE eight listed hidden widths | §IV-C/Table I, p. 4115 | EXACT | Four encoder + four decoder transforms |
| A separate latent layer exists | Fig. 3/§III-A1, p. 4109 | NON-EXECUTABLE | Latent width absent |
| FC-SAE sigmoid/Softmax/dropout .4/Adam | Table I, p. 4115 | EXACT | Printed model settings |
| Dropout placement | not stated | NON-EXECUTABLE | Requires completion choice |
| SAE uses MSE and `score > threshold` | Eq. 7/§III-A/§III-C, pp. 4109, 4114 | EXACT | Higher MSE is anomalous |
| FC-SAE threshold is 0.58 | §IV-B, p. 4115 | EXACT | Direct anchor constant |
| Derive threshold by “median of IQR of ROC” | §III-D/§IV-B, pp. 4114–4115 | NON-EXECUTABLE | No unique operation or validation scores |
| VAE low reconstruction probability is anomalous | §III-B, p. 4111 | EXACT | Lower-is-anomalous |
| Generic paragraph says probability above threshold is anomalous | §III-C, p. 4114 | CONTRADICTORY | Opposite VAE direction |
| VAE reconstruction probability implementation | §III-B, p. 4111 | NON-EXECUTABLE | Variance/draw/aggregation details absent |
| VAE ELBO uses `KL(q(k|x)||p(x))` and compares with `log p(k)` | Eq. 9, p. 4111 | CONTRADICTORY | Distributions/variables do not match the stated ELBO |
| VAE variance output has no positivity parameterization | Algorithms 3–4, p. 4112 | NON-EXECUTABLE | A Gaussian variance is not guaranteed valid |
| Algorithm 1 uses SGD; Table I selects Adam | Algorithm 1/Table I, pp. 4110, 4115 | AMBIGUOUS | Final table governs anchor optimizer |
| Search layers from `{2,3,4,5}` | §IV-C, p. 4115 | CONTRADICTORY | Table I reports 6 and 8 |
| Algorithm 6 searches one `N_l`; Table I gives unequal widths | Algorithm 6/§IV-C, pp. 4114–4115 | CONTRADICTORY | Search cannot yield printed layouts as written |
| Sequential search records DR/FA on XTR CV | Algorithm 6/§III-E, p. 4114 | NON-EXECUTABLE | XTR is benign-only; selection rule/folds absent |
| Batch, epochs, convergence, seeds, LR | not stated | NON-EXECUTABLE | Predeclared completion choices required |
| Keras Sequential API | §IV, p. 4114 | AMBIGUOUS | Keras/backend/version absent |
| LSTM input layout and decoder schedule/state transfer | Algorithms 2/4/5, pp. 4111–4113 | AMBIGUOUS | Several materially different seq2seq models |
| AEA concatenates context/output; Algorithm 5 prints a sum | §III-C/Algorithm 5, pp. 4113–4114 | CONTRADICTORY | Merge and dimensional projection absent |
| Algorithm 5 defines reconstructed input using itself before decoding | Algorithm 5, lines 21–23, p. 4113 | NON-EXECUTABLE | Initial/ongoing decoder input is undefined |
| SVM “kernel and Gamma are scale and sigmoid” | §IV-C, p. 4115 | NON-EXECUTABLE | Values are reversed relative to the standard API domains |
| ARIMA has differencing 1 and moving average 0 | §IV-C, p. 4115 | NON-EXECUTABLE | Autoregressive order, fit unit, and score absent |
| Supervised output/loss/label cardinality | not stated | NON-EXECUTABLE | Benchmark classifier identity incomplete |
| DR/FA/SP/ACC/F1 formulas | §III-D, p. 4114 | EXACT | ACC is balanced accuracy |
| Precision prose versus `TP/(FP+TP)` formula | §III-D, p. 4114 | CONTRADICTORY | Formula is used; prose describes recall |
| Table-II Naive Bayes DR/PR/F1 | Table II, p. 4116 | CONTRADICTORY | DR=PR=75 forces F1=75, not 77 |
| Common test prevalence implied by DR/FA/PR | Tables II–III, p. 4116 | CONTRADICTORY | Rows have no common prevalence even with generous rounding |
| Table III reports unseen test results | §IV-D, p. 4115 | AMBIGUOUS | Test construction conflicts with unseen identity |
| Table IV full size is 60m | Table IV/prose, p. 4116 | AMBIGUOUS | Likely scalar readings, not profile rows |
| Table IV training time | Table IV, p. 4116 | NON-EXECUTABLE | Hardware and fit protocol absent |
| Table V has no ADASYN because balanced | Table V prose, p. 4116 | AMBIGUOUS | Test identity not stated |
| Table-V model/test identity with same settings/thresholds | Table V prose, p. 4116 | AMBIGUOUS | Fixed model/common benign implies invariant FA; varying FA requires unstated retraining or resampling |
| Online decision takes 1–2 seconds per individual reading | §IV-D, p. 4116 | CONTRADICTORY | Model consumes a 48-reading profile |

## Source-freeze checkpoint

The PDF supports the overall flow and the printed `P` operations above, but it
does **not** uniquely specify one executable experiment. `P0-ISET-FCSAE` is a
declared straight-through executable completion, not a claim that its `I`
choices are the authors’ hidden implementation.

The independent 2026-08-11 source re-audit is complete. Before existing model
code is trusted or experimental execution resumes, the checkpoint is:

1. agree that `P0` is the first executable anchor while the unmodified printed
   Attack 3 remains a documented non-executable result;
2. confirm that the material branches above cover the reasonable readings that
   could change a reproduction conclusion; and
3. keep later corrected controls separate from paper-consistent results; and
4. audit the five-file implementation against this source table before treating
   any prior or future run as eligible.
