# Next experiment: the jointly trained sequential detector

Recorded 2026-09-26 after the AEA recheck. **Source/design checkpoint only:
implementation, constructed learning checks and GPU timing remain pending.
No research pair is running or authorized by this document alone.**

## Decision and question

Move toward the paper's main supervised ensemble rather than choose a
standalone AEA repair by whether its optimistic bound passes. A standalone
AEA fit is not a prerequisite for the jointly trained model in Section IV-B.
Keep standalone AEA, ARIMA, ensemble averaging, customer-specific coverage
and full-population reproduction on the remaining-work list.

The first numerical question is: does one declared completion of the
sequential classifier retain the reported detection/false-alarm tradeoff
under the existing customer-label poisoning process? This is an interpreted
numerical pilot (`I/N`), not the broad statistical attainability study or a
test establishing that reconstruction causes robustness.

## Source review and four separate choices

All ten pages of the target were read and visually checked in this session.
Its SHA256 remains
`03a372fb5ee129799d43a50e6bec8f86fb108e9f8bdc85350e3ffe3e13d4a4ff`.
Locators below use printed pages 2675-2684.

| Choice | Standalone novelty AEA | Sequential classifier proposed here |
|---|---|---|
| Input scaling | III-A.1, p. 2677: training zero mean/unit variance, reused on test; axis omitted. Featurewise z-score remains an explicit completion, global scaling an open interpretation. Min-max is a separate control. | III-A.2, p. 2677: two-class preparation. Reuse the frozen featurewise-standardized classifier arrays; do not substitute novelty arrays or select another scaler from bound results. |
| Training objective | III-B.1(b), p. 2678 describes reconstruction, but supplies no loss formula. MSE and MAE are separately defensible completions; neither is selected by its score bound. No standalone research fit is proposed now. | IV-B, p. 2682 explicitly optimizes classification cross-entropy through all components. Use standard binary cross-entropy with the repeated logarithm in Eq.(1), p. 2679 explicitly repaired. No reconstruction term, pretraining or frozen encoder. |
| Detection score | Reconstruction error; norm, reduction and units omitted. MSE, MAE and raw-unit variants remain distinct hypotheses. | Scalar final Sigmoid attack probability, not intermediate reconstruction error. |
| Decision cutoff | III-D.2(b), p. 2680 prints 0.51; its ROC/IQR derivation is incomplete. MAE at a free cutoff is not a repair preserving 0.51. | Use probability >0.5, ties benign, as an explicit omitted classification-rule completion. Also report every saved-score cutoff diagnostically; do not import 0.51 or choose a test-optimal deployment threshold. |

The standalone feature-z/global-z MSE exclusions remain valid within their
frozen novelty-pilot scope. They do not constrain a downstream classifier's
probability score. MAE promotion remains withdrawn. There is no new source
evidence selecting MAE, min-max scaling, or a reconstruction training loss.

The accessible Zhao et al. reference [23] was consulted at pp. 425-426
(standard soft attention, Eqs.1-5), p. 428 (decoder, Eqs.14-15), and p. 429
(scaling and training). These complete pages were visually checked. It
supports a previous-decoder-query/encoder-memory attention completion, but
its hierarchical bidirectional topology, [0,1] speech features, RMSProp,
100 epochs and pretrain/freeze procedure are not the target's method.
Reference [22] remains unavailable in full; it supplies no verified missing
loss, activation or optimizer choice here. No new code-availability claim
is made by this source review.

## One explicit executable interpretation

Proposed identifier: `I-SEQ-native-IVC`. Prefer Section IV-C's selected
ensemble hyperparameters over standalone Table II settings and conflicting
generic equations. This is not a literal implementation of every printed
equation. Every completion below must be visible in the model configuration.

| Component or choice | Proposed execution | Source / reason |
|---|---|---|
| Input | `(batch,48,1)`; reset state for each daily profile | II-B, p. 2677: one row is one day's 48 readings |
| Encoder | Native Keras LSTM layers 500,300,200, returning sequence memory and final hidden/cell states | IV-C, p. 2682 gives widths; native cell is an explicit completion of the equation/table conflict |
| Decoder | LSTM cells 200,300,500; initialize both states from encoder layers 3,2,1; zero first scalar feedback | Fig.2, p. 2678 and Algorithm1, p. 2683 imply encoder-state initialization but omit executable mapping and first output |
| Recurrent activations | ReLU candidate/state activation, sigmoid gates; no peepholes | IV-C selects ReLU hidden activations; Algorithm1 prints tanh and peephole terms. Preserve that alternative rather than claim equivalence |
| Attention | All 48 top-encoder states; additive tanh alignment of width 200 using previous final-decoder hidden state; softmax over time; weighted context sum | III-B.1(b)/Fig.2, p. 2678; reference[23] Eqs.1-5. Width 200 and affine parameterization are completions |
| Decode order | Complete encoder memory, then 48 decoder steps; concatenate context and previous scalar output into first decoder; forward-order output | Repairs Algorithm1's use of decoder states before decoder execution; causal timing is supported by reference[23] |
| Intermediate projection | Dense(1, ReLU) per decoder step, yielding `(batch,48,1)` for the GRU and next feedback step | Projection and intermediate activation omitted; scalar preserves the stated reconstructed daily shape, ReLU applies IV-C's hidden-activation choice to this internal layer |
| GRU stack | Eight native GRU layers of 300 units, first seven returning sequences; ReLU candidates, sigmoid gates, `reset_after=False`; no inter-layer Softmax projections | IV-C, p. 2682; III-B.2(e), p. 2679. Ordinary state outputs complete the generic projected-output ambiguity |
| Classifier | Last GRU state -> Dense(500,ReLU) -> Dense(1,Sigmoid) | IV-B/C, p. 2682: one 500-neuron fully connected layer plus output; binary scalar is a completion |
| Optimizer | Adam learning rate .001, beta1 .9, beta2 .999, epsilon 1e-7, no weight decay/clipping/schedule | Adam is explicit in IV-C; remaining values are declared defaults, not claimed author settings |
| Regularization | No dropout; MaxNorm(1,axis=0) on all input/recurrent/attention/projection/Dense kernels; biases unconstrained | IV-C: dropout 0 and constraint 1; type/coverage omitted |
| Initializers | Seed 20260920; GlorotUniform input/attention/projection/Dense kernels, Orthogonal recurrent kernels, zero biases except native LSTM unit-forget bias 1 | Omitted; preserve ordinary declared Keras choices and save initial weights |
| Training | Joint repaired BCE only, all layers trainable; 50 epochs, batch 100 including remainder; final weights, no early accuracy stopping or best-checkpoint selection | IV-B/C refers to III-D.2, pp. 2679-2680; Algorithm1's convergence loop is resolved by the stated finite epoch schedule |

The attention's tanh alignment and sigmoid gates have distinct roles from
the selected ReLU hidden activation. No teacher forcing, extra residuals,
normalization layers, auxiliary labels, class weights or augmentation.
Use the pinned TensorFlow 2.16.2/Keras 3.4.1 environment; float32,
deterministic operations, TF32/XLA/mixed precision off. Do not replace ReLU
with tanh just to obtain the faster cuDNN path.

Expected parameter inventory under these scalar-projection choices:
AEA block 5,031,701 + GRU stack 4,058,100 + Dense head 151,001 = **9,240,802**.
This is an arithmetic specification, not a runtime-verified model count.

Material alternatives remain visible: tanh/peephole equation cells; Sigmoid
or linear intermediate projection; interpreting ReLU as applying only to
later layers; different state bridges; explicit per-layer GRU output
projections; teacher forcing; another standardization axis; constraint
coverage; and ordinary optimizer defaults. Sigmoid intermediate output is
plausible from the standalone component description, while a linear output
permits signed standardized values. Neither is selected by observed
performance. Do not run a Cartesian sweep or change this primary after a
failure without a recorded, separately bounded question.

## Frozen data and comparison scope

Use the unchanged `setup-20260920-attempt1/generalized-two-class-p00` and
`generalized-two-class-p30` preparations, not the 373-row novelty training set.
Metadata SHA256:

- p00: `1c2e5ebfee9584c160fee209851e7f685d8c8d7971922d62a63960bc6308c3de`
- p30: `9ac0faeea220e62ed5af13e39deb174a190cc69749786c38ea58240e5031f21a`

Each has 4,464 training and 2,232 test rows, 20 customers/28 complete days,
original pre-split ADASYN, the existing scaling and dependent observations.
P30 flips 675 malicious training labels from six customers, 15.12097% of all
training rows; it does not corrupt 30% of every training row. Fit observed
labels only. Preserve true labels for evaluation and all original/synthetic
identities. Validate every array against the existing loader's frozen hashes.

Use one seed 20260920 at each condition and require identical initial-weight
hashes. Shuffle with the existing neural runner's seeded deterministic
full-buffer rule, one input thread, no dropped remainder: 45 updates/epoch,
2,250 for a complete fit. Save all epoch histories and all attempts.

Existing forest/FF/GRU scores are contextual comparators on matched test IDs.
The saved GRU has different dropout, constraint and output-head settings;
it is **not** a matched removal-of-AEA ablation. Any causal claim that AEA
adds robustness requires a later matched bypass experiment with the same
downstream settings and separately specified uncertainty. A successful
numerical ensemble pilot alone cannot establish that mechanism.

## Constructed gates before research compute

Implementation is pending. The old six AEA fixtures do not validate this
ensemble: that prototype uses standalone Sigmoid/SGD settings.

Required software checks:

1. Verify runtime layers, parameter inventory, intermediate/final shapes,
   activations, optimizer, constraints and paired initialization.
2. Verify BCE labels and signs independently; demonstrate the printed loss
   is label-independent. All intended layer groups must receive finite
   gradients from classification alone and change on a constructed update.
   Report zero or tiny gradients rather than interpreting mere connectivity
   as useful learning.
3. Hold memory fixed while varying the decoder query; disable the query
   connection as a negative control. Intervene on previous-output feedback
   while holding the other decoder inputs fixed; the subsequent state must
   respond, and disabling that input connection must remove the effect.
4. Use the same layer order/depth with encoder 8/6/4, mirrored decoder,
   eight 8-unit GRUs, Dense 16 and 8 time steps. NumPy seed 20260926 generates
   32 training and 32 fresh test rows, each split equally between labels 0/1;
   each entry is `.75*(2*y-1) + .05*Normal(0,1)`. Keep the specified model
   seed/Adam settings; 300 full-batch updates, 5-minute guard per fit. Repeat
   once from identical initial weights with both train/test labels reversed.
   The daily-mean rule establishes the easy positive-control floor and the
   balanced constant prior has 50% accuracy/BCE log(2). Require each learned
   model to achieve at least 90% fresh-test accuracy and BCE below log(2)/2.
   These are instrument gates on a trivial task, not evidence of temporal
   capability. Preserve both outcomes; a failed learning check stops promotion
   without seed retries or research-test tuning.
5. Fresh-process save/reload must preserve weights, probabilities, internal
   outputs and optimizer counts. Partial/nonfinite paths must preserve the
   failed status and exclude the result from complete-fit comparisons.

Then propose one **20-minute** constructed GPU allocation: one V100-16GB,
4 CPUs/16 GiB host RAM, no CER loading. Full 9,240,802-parameter model;
synthetic normal inputs from NumPy seed 20260926 and balanced alternating
labels; 4,464 rows; three epochs using the same variable-batch `tf.data` and
`model.fit` path as the planned research runner (44 batches of 100 plus 64).
Record initialization/compile time, every batch/epoch duration, final-batch
time, finite loss/gradients, placement, memory and fresh reload/inference.
After the first warmup epoch, let E be the slower complete epochs 2/3 time.

Launch gate: all software checks pass, allocation stays within memory, and
`50 * E <= 3,600 seconds`. This leaves headroom inside a 70-minute per-fit
guard. Failure is an instrument/runtime planning result, not evidence
against the paper. Do not infer timing from the old fixed-batch GRU preflight.

## Proposed research pair, conditional on those gates

One fresh p00/p30 pair, no extra seed/settings or copied pretrained weights.
Proposed ceiling: one **180-minute** job on one V100-16GB,4 CPUs/16 GiB,
70-minute batch-boundary guard per fit, with 40 minutes reserved for startup,
scoring, saving/reload and audits. Including preflight, maximum requested
allocation is 200 GPU-minutes. These are prospective operational ceilings,
not a measured runtime, paper-time equivalence or authorization to submit.
Freeze source, implementation and launch contract and obtain the experiment
checkpoint before submission. The user's present request covers source
resolution and definition of the bounded next step.

Stop on failed input hash, nonfinite values, GPU fallback, failed reload or
partial p00; do not proceed automatically to p30. Preserve partial weights,
optimizer, histories, elapsed time and failure reason. Never extend the
guard, train beyond 50 epochs or rerun a different seed to improve a result.
All research training/scoring remains on a compute node.

## Targets, measurements and decisions

Table V, p. 2683, sequential ensemble context (percentages):

| Poisoning | DR | FA | SP | PR | ACC | F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| p00 | 95.2 | 2.9 | 97.1 | 95.6 | 96.1 | 95.4 | 97.4 |
| p30 | 92.2 | 5.8 | 94.2 | 92.7 | 93.2 | 92.4 | 92.0 |

Report the whole row with raw probabilities/counts at 0.5, original/synthetic
and attack-type breakdowns, constant-prior and daily-mean baselines, and
all-cutoff/reversed-score DR at both 2.9% and 5.8% FA caps in both conditions.
Also show favorable one-decimal allowances (DR target minus .05, FA cap
plus .05). Use a common FA cap when describing poisoning degradation.
The printed 3-point DR drop changes its FA allowance and is not itself a
fixed-FA robustness contrast. Saved-score cutoffs are diagnostics using test
labels, not independent calibration.

Save intermediate outputs before/after fitting and per-row MSE/MAE to the
input, alongside zero and training-mean reconstruction comparisons. These
are descriptions of whether the intermediate representation resembles the
input; they neither add reconstruction loss nor identify causation. Both
scores are recorded regardless of which appears favorable. High classification
with poor reconstruction would separate numerical success from the claimed
reconstruction explanation; it would not prove reconstruction never helps.

Possible outcomes and next decisions:

- A favorable pilot supports attainable performance for this completion;
  next consider independent populations and matched mechanism controls.
- Good ranking with a default-rule miss directs attention to calibration;
  it does not authorize test-selected deployment thresholds.
- Completed weak ranking directs diagnosis to inputs, training and specific
  source alternatives. It does not trigger more seeds automatically.
- Failed constructed learning, nonfinite training or partial completion
  requires instrument/runtime diagnosis before scientific interpretation.

This first pair reports descriptive pilot results only: no independent-row
confidence intervals, family-wide ceiling, full-population reproduction or
author-intent conclusion. The future mountain/attainability investigation
still needs independent units, a finite justified configuration family,
selection-aware uncertainty and frozen stopping rules. Preserve the earlier
single-confusion-matrix inconsistencies; undocumented averaging remains an
open explanation and no internally incompatible tuple becomes a mandatory
success criterion for this pilot.
