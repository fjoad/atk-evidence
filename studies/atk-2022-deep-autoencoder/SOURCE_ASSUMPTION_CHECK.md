# Source assumptions behind the performance bound

Date: 2026-08-31

Status: source review complete; cheap checks frozen before execution

The user approved the source review and a few discriminating no-training checks.
They explicitly require discussion before updating the public account. Keep the
website, README, and public report unchanged. Save records locally and stop for
discussion after this round. This is not approval for training or publication.

## Question

Does the fixed-pipeline limit depend on an interpretation we supplied, or on
operations that the paper explicitly requires? This is an `A` diagnostic, not a
new numerical reproduction or a recurrence/attention mechanism test.

The complete paper was re-read; printed pp. 4109, 4114, 4115, and 4116 were
visually re-inspected. PDF SHA-256:
`f3098e0c27ee19b27bea026aedc3d10e5dbb0c46f5cd01ed5bd5c05b7dcf850f`.

## Source-to-code map

| Assumption | Paper location and instruction | Implemented operation | Assessment |
|---|---|---|---|
| Unit of input | p. 4109, II-C and III-A: one daily profile, 48 inputs | `prepare_data.py:424-461`; `models.py:155` | Day/48-coordinate reading is explicit; missing/DST filtering is a completion |
| Normalize before splitting | p. 4109, II-C: normalize both classes to zero mean/unit variance, then split B | `prepare_data.py:698-722`: one joint feature-wise standardizer over all B and six attack matrices | Standardization and order explicit; fitted scope and exact formula omitted |
| FC-SAE output | p. 4115, Table I, FC/SAE AO row, reinforced by IV-C | `models.py:170`: Dense(48, activation=softmax) | Softmax is explicit, not an invented head; scalar normalization over 48 outputs follows the ordinary vector operation |
| Anomaly score | p. 4109, III-A/(7), and p. 4114, III-C: MSE between input and reconstruction | `run_experiment.py:270-287`: mean squared coordinate error | MSE is explicit; code uses the natural per-profile reduction |
| Score direction | pp. 4109 and 4114: larger error indicates attack | `run_experiment.py:241`: score > threshold | Explicit for SAE; the separate VAE direction conflict does not change this row |
| Threshold | p. 4115, IV-B: FC-SAE 0.58 | unchanged 0.58 | Explicit final value; its selection procedure remains unclear |
| Accuracy | p. 4114, III-D: mean of DR and specificity | `run_experiment.py:253`: balanced accuracy | Explicit; ordinary row accuracy would be a different metric |
| Population and test set | pp. 4108-4109: about 3,000 residential meters; unseen customers; test ADASYN | all 4,225 labeled residential meters, fixed customer split, held-out attacks, test ADASYN | Several completions remain; no claim to cover every population or resampling |

Code locators refer to the unchanged scientific revision
`a88d17477ad96b01ffa44a50d8ce051dd8d2b5ca`, under this study's `reproduction/`.

## Small finite set to check

1. **Reference:** current joint feature-wise standardization and Softmax.
2. **Joint scalar standardization (`I` input interpretation):** a single mean
   and standard deviation across all readings in B and M, before the split.
   This is a weaker reading of a common scale than feature-wise scaling, but
   the text gives no axis or fitted statistics. Keep the same raw population,
   generated attacks, and held-out customers.
3. **Separate-class feature standardization (`I`, weaker interpretation):**
   normalize B and the pooled six-attack M separately, feature by feature,
   before splitting. The phrase "normalize both classes" leaves this possible,
   although a common fitted transform is more natural and deployment would
   require knowing the class. Do not call it good practice or silently promote
   it as the intended method. Do not normalize each attack separately.
4. **Sigmoid output range (`C/A` search-candidate control):** keep the current
   prepared input but replace the simplex relaxation by the cube [0,1]^48.
   Sigmoid is listed in the paper's searched output choices on p. 4115, but
   Table I selects Softmax for FC-SAE. This is not that final reported model
   and not a test of the LSTM architecture.

Not selected now: per-customer/per-day normalization; train-fitted scaling;
different meter subsets, dates, attack completions, or splits; min-max scaling;
linear output; class-specific thresholds; alternative learned scoring. These
remain distinguished from the finite checks above. Train-fitted scaling
changes the printed order; min-max and linear output are not the stated final
normalization/head. An unavailable author implementation remains unknown.

## Static checks that need no dataset search

SSE = 48*MSE and RMSE = sqrt(MSE) are strictly increasing transforms on
nonnegative errors. With correspondingly transformed cutoffs, rankings, AUC,
and the all-cutoff ROC region are identical. Hence these unit changes cannot
rescue the existing full-evaluation bound. Keeping 0.58 numerically unchanged
would pick a different operating point already covered by that region.
Fixtures check this and the output-domain extrema; no source reinterpretation
is inferred merely from a successful fixture.

## Frozen no-training execution

- Inputs: the unchanged anchor result (SHA-256
  `ae07b42ef6c84242ca9b39db8b8828694d6d4df6859abdee090fc0a613a69154`),
  its metadata, `benign_raw.npy`, `x_test.npy`, `y_test.npy`, and
  `test_source_row.npy`. Verify every consumed file against the saved hashes.
  Also verify the previous full diagnostic JSON against SHA-256
  `0e337db5a7f424dafc46cc3aac0643b5fac77c61799b0558b2434143bf5cd372`;
  reproduce its original-row reference bounds as an end-to-end check.
- Fit the alternative statistics on the full original pre-split population,
  not on the test probe. For scalar scaling, derive the pooled second moment
  from the recorded joint feature means/variances. For separate B/M scaling,
  stream B's raw moments and derive M's moments from the joint moments using
  the exact 1:6 B/M construction. Record the fitted vectors and verify positive
  finite variances. Reconstruct raw values from saved standardized inputs;
  validate benign sample reconstruction against `benign_raw.npy`.
- Primary diagnostic population: the 750,767 held-out source days and six
  attack siblings, totaling 5,255,369 original rows. Synthetic benign rows are
  deliberately excluded. Different scaling changes ADASYN neighbors, so
  transforming old synthetic rows would not execute the alternative method.
- Therefore original-row all-cutoff/AUC limits do NOT automatically bound the
  paper's resampled evaluation. However the maximum DR at fixed threshold
  0.58 applies to the unchanged attack population regardless of any added
  synthetic benign rows. A maximum DR below 80.5% excludes the rounded target
  at that printed cutoff for the specified interpretation/population. A
  feasible label-aware relaxation never proves a trainable model can match it.
- For each branch, compute outward-padded float64 score extrema, maximum
  balanced accuracy/AUC, DR at FA <=15% and <=15.5%, and maximum DR/minimum FA
  at 0.58. Report both directions separately. Softmax uses the previously
  tested simplex helper. For the cube, L=mean((x-clip(x,0,1))^2), and
  U=mean(max(x^2,(x-1)^2)). Padding is 1e-5*(1+U), lower clipped to zero.
- Competing predictions: ordinary normalization completions either retain the
  low ceiling or move it materially. The larger cube may exclude the target
  or leave it feasible. Preserve every result; no branch is selected by target
  proximity and no training is promoted automatically.
- One CPU allocation: four cores, <=8 GiB, <=10 minutes. No GPU, training,
  ADASYN regeneration, heartbeat, repeated seeds, or automatic retry.
- First run 64 evenly spaced source days and all six siblings (448 rows).
  Inspect hashes, identities, round-trip tolerance <=2e-5*(1+abs(raw)), finite
  moments, geometric containment, and timing. Promote one full original-row
  pass only if the pilot passes and takes <=45 seconds with geometry throughput
  consistent with the remaining allocation. All branches execute together.
- One direct analysis script and hand-checkable fixtures; reuse the unchanged
  geometric/ROC helper. Save code/contract hashes, actual runtime, failures,
  and all summaries under a new directory. Original artifacts are read-only.
- Local work is source inspection, fixtures, and record transfer only.
  Scientific data scoring runs on a cluster compute node. A local commit
  freezes code; transfer it directly without pushing or publishing.

## Stop and discussion gate

After the bounded checks, save a separate discussion finding and update the
internal state. Do not change README, website, report, or previous results.
Do not push this round before the user discusses its interpretation. Any
further experiment needs its own recorded question and approval.
