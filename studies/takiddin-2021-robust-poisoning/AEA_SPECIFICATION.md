# Standalone AEA: source map and proposed executable interpretation

Recorded September24 before any AEA research fit or new data scoring.
This is source specification, not evidence that an AEA succeeds or fails.
Do not inherit an implementation or result from another paper in this project.

## Sources and access

The complete ten-page target was re-read and all pages visually inspected:
*Robust Electricity Theft Detection Against Data Poisoning Attacks in Smart
Grids*, DOI10.1109/TSG.2020.3047864, pp.2675–2684. Local PDF remains unchanged,
SHA256 `03a372fb5ee129799d43a50e6bec8f86fb108e9f8bdc85350e3ffe3e13d4a4ff`.

Reference [23], Zhao et al., DOI10.1109/JSTSP.2019.2955012, was retrieved from
[the University of Augsburg repository](https://opus.bibliothek.uni-augsburg.de/opus4/frontdoor/deliver/index/docId/67303/file/67303.pdf).
The relevant complete pages425–430 were read and visually inspected, including
Fig.1 and Eqs.1–5,11–15. Its standard attention description uses a previous
decoder state, previous output, learned tanh scoring and a weighted sum of
encoder states. Its own hierarchy, bidirectionality, [0,1] input scaling,
RMSProp/100 epochs and pretrain/freeze sequence differ from the target and
must not silently replace it. The reference supports a conventional attention
completion, not the missing target implementation as a whole.
Local ignored PDF: `papers/references/zhao-2020-hierarchical-attention-autoencoders.pdf`;
SHA256 `eacc7bc72db20c13b7a4f66de0837900ba5204e479f09c3a842ea73b9928b525`
(repository cover plus12 article pages).

Reference [22], Sun and Wu, DOI10.1109/AICAS.2019.8771593, is identified by
the target and [NTU's publication record](https://scholars.lib.ntu.edu.tw/entities/publication/9f9a6216-ca8e-4c84-b4df-73ec7fc34d38).
Full text was not located in the bounded title/author/DOI searches; IEEE
access failed, and the opened NTU record exposed no paper download. Do not
claim to have read it or infer its cells/loss from its title. Reference[24]
was not used to fill missing implementation details. The earlier bounded
[code search](CODE_AVAILABILITY.md) still located no target implementation;
publisher supplementary material remains unverified.

## Three objects that must stay separate

1. **Standalone novelty AEA:** learns to reconstruct observed-benign training
   profiles; an error score and threshold produce an anomaly decision.
2. **AEA-shaped block in the sequential ensemble:** trained through the
   classification loss described in IV-B. No explicit reconstruction term or
   benign pretraining is specified there. A label-trained representation is
   not automatically a reconstruction.
3. **Controls:** explicit benign pretraining, a combined reconstruction/
   classification objective, or different output scaling are separately
   labeled experiments, not unreported repairs to the first two objects.

Equation1's repeated-log(p) error belongs to the classification route.
Standalone reconstruction should not be trained to constant benign class
labels using that equation. A reconstruction objective needs input targets.

## Source map

| Consequential item | Locator | Status and consequence |
|---|---|---|
| Novelty training versus testing | III-A.1, p2677; III-B.1, p2678 | Benign-only training before contamination; benign plus malicious evaluation. Not the classifier preparation. |
| Input scaling | III-A.1, p2677 | Explicit zero-mean/unit-variance training scaling, reused for test data. |
| Test ADASYN | III-A.1, p2677 | Explicitly oversamples benign test examples; preserve and expose it, with original-only views. |
| Poisoning | III-A.3, p2678 | Percentages of customers for generalized models; exact replacement/injection procedure omitted. |
| Recurrent encoder/decoder and attention | III-B.1(b), Fig.2, p2678 | Attention uses encoder states and previous decoder state; context plus reconstructed output enters decoder. Dimensions, first output and execution order incomplete. |
| Cell equations | Algorithm1 lines9–13,28–32, p2683 | Cell-to-gate V terms and tanh candidate/state output; not ordinary native Keras LSTM with Sigmoid activation. |
| Selected widths | TableII/III-D.2(a), p2680 | Encoder500,300,200; decoder200,300,500; six recurrent layers. |
| Selected optimizer/activations | TableII, p2680 | SGD, no dropout, constraint1, Sigmoid hidden/output. Learning rate and constraint type/coverage omitted. |
| Schedule | III-D.2, pp2679–2680 | 50 epochs, batch100; no best-epoch selection specified. |
| Reconstruction error | III-B.1(b), III-D.2(b) | No standalone loss formula, norm/reduction or output projection. MSE is a declared completion, not an explicit AEA equation. |
| Threshold | III-D.2(b), p2680 | Printed0.51. The ROC/IQR description does not uniquely identify a scalar threshold algorithm; do not substitute an optimal ROC rule silently. |
| Ensemble loss | IV-B, p2682; Algorithm1 | Classification cost only is stated; reconstruction/pretraining is not specified. |

### Literal obstacles and smallest repairs

Algorithm1 places attention inside the encoder-time loop but requests
decoder states for that time before running the decoder. Its initialization
line does not explain how a complete sequence of decoder states is available
for that encoder pass. The usual causal reading is therefore incomplete;
this is not a proof that no other completion can execute. The proposed repair first
computes encoder memory, then computes attention within each decoder step.

Encoder states are saved under encoder-layer indices, but decoder-layer
indices differ. Widths reverse too. Direct same-index copying is undefined;
the proposed mirror map is encoder3→decoder1 (200),2→2 (300),1→3 (500), for
both hidden and cell states. No learned projection is needed under this map.

Peephole terms V*c cannot silently become native Keras LSTM. The relationship
between the printed tanh operations and the selected Sigmoid hidden activation
is not explained: applying Sigmoid to Keras's activation argument changes
both candidate and state-output nonlinearities. Preserve the equation-cell
and selected-table interpretations rather than saying the initial completion
implements every printed equation.

## Proposed first interpretation: I-AEA-native-table

This is specified for future implementation, not a launched or completed fit.
It prioritizes TableII's selected settings and the named Keras environment,
while making the following completions explicit:

- Input `(batch,48,1)` from each standardized daily row; no cross-day state.
- Three native Keras LSTMs with500,300,200 units; each returns all states and
  final hidden/cell state. Activation=Sigmoid, recurrent activation=Sigmoid;
  no peepholes. Native candidate/state activations replace Algorithm1 tanh
  under this interpretation, not as a literal transcription.
- Decoder cells200,300,500, with matching native cell settings. Initialize
  from mirrored final encoder states; encoder initial states are zero.
- Attention at decoder time t uses all48 top-encoder states (width200) and
  the previous final decoder hidden state (width500). Additive alignment:
  `e[t,j]=v^T tanh(W_e h_E[j]+W_d h_D[t-1]+b)`, alignment width200;
  `a[t,:]=softmax(e[t,:])` across the48 encoder times;
  `c[t]=sum_j a[t,j] h_E[j]`. Attention width/parameterization are explicit
  omissions completed using the conventional pattern, not reported values.
- Feed `concat(c[t],xhat[t-1])` into decoder1 (201 inputs), then decoder2/3.
  `xhat[0]=0`; `xhat[t]=sigmoid(w_out^T h_D3[t]+b_out)` is one scalar.
  Reconstruct in forward time order. No teacher forcing in training or test.
- Reconstruction target is the same standardized input, not the class label.
  Loss and score are mean squared error across48 readings, with a mean across
  training examples for optimization. Higher score means more anomalous.
- SGD learning rate.01, momentum0, no Nesterov/decay/clipping/schedule;
  float32, no XLA/mixed precision/TF32. Use the existing pinned neural runtime.
  Native LSTM with Sigmoid candidate does not use the tanh cuDNN fast path.
- GlorotUniform input/output/attention kernels, Orthogonal recurrent kernels,
  zero biases except native unit-forget bias1; seed20260920. MaxNorm1,axis0
  on input/recurrent/output/attention kernels; biases unconstrained. These
  initializer and constraint choices are completions, not paper statements.
- No dropout, state carryover, extra layers, sparsity penalty or regularizer.
  The model is not made into the sparse/hierarchical model of a reference.
- Proposed future schedule is exactly50 epochs/batch100/final weights.
  No training launch budget is inferred from the GRU: an eventual constructed
  test must use this model and the actual variable-batch dataset structure.

The dimension plan implies5,031,701 parameters under these choices; verify
against runtime before any fit. Native SGD/LSTM defaults were checked in the
local pinned Keras3.4.1 source; that does not establish the authors' version.

## Branch choice after the zero-fit envelope

Feature-z MAE is the least invasive surviving branch: it preserves the
paper's explicit featurewise standardization, Sigmoid output and threshold
context while changing only the unspecified reconstruction-error norm.
Raw-unit MSE and min-max remain alternatives; min-max is explicitly a control
and raw-unit MSE changes the input/score convention more materially.

`checks/aea_model.py` implements only a constructed fixture for this MAE
branch. It is not part of the five paper-facing files and has not loaded
research data. Four pinned-environment tests pass: 5,031,701 parameters and
shapes, bounded output/attention normalization/query dependence, finite small
fixture updates, and separate MAE/MSE objective identities. This does not
establish a data result or authorize a fit. Feature-z MAE remains an
interpreted `I/A` candidate, not an authenticated author implementation.

### Open alternatives, not an automatic sweep

Equation-style peepholes/tanh; different peephole matrix/diagonal convention;
teacher forcing; zero or learned decoder-state bridges; attention width and
placement; constraint coverage; alternative error reductions remain open.
RMSE and sum-of-squares change the units of0.51 even when their rankings
match MSE. A linear reconstruction head or [0,1] rescaling changes explicit
parts of the initial reading and belongs to a separate correction/control.
Do not silently choose one because it helps reproduce a number.

## Existing novelty pilot and its limitations

The unmodified `setup-20260920-attempt1/generalized-novelty-p00` and p30
preparations already exist. Metadata hashes:

- p00: `dfb6df1a5e5223569682f0066e8d1a2cb4a0a33aa3c13e7974becd463e2f4367`
- p30: `18c4527896cf2aaba5591b9dbf2b9a869f04788e955316e869f68c3d56c6e449`

Metadata reports373 training rows and6704 test rows at both levels. P00 has
373 true benign training rows; p30 has265 benign/108 malicious, all observed
as benign, from six selected customers. The interpretation replaces each
selected training profile with one of its six attacks, not all six at once.
The test contains3344 benign (187 original/3157 synthetic) and3360 malicious
rows. All attacks enter test, including attacks derived from training source
days. At p30,108 exact training row identities also occur in test. Preserve
and report this interpretation; it is not a clean held-out experiment.

Training-contamination changes the fitted scaler, so the same raw test
identities need not have the same standardized inputs across levels. A
novelty result cannot be paired with the classifier's2232-row test as though
only architecture changed. No new arrays were scored to write this note;
only existing metadata and preparation source were inspected.

## Cheap checks before a neural fit

If the reconstructed row r lies in[0,1]^48 and score is MSE, every row x obeys

`L(x)=mean((x-clip(x,0,1))^2) <= MSE(x,r) <= U(x)=mean(max(x^2,(x-1)^2))`.

The lower bound is coordinatewise projection onto the closed output box;
the upper bound chooses the farther endpoint for each coordinate. Using the
closed box is conservative for an ideal open-range Sigmoid and also covers
floating-point saturation. These bounds apply to every set of weights under
the stated output/score interpretation, not to linear outputs or other norms.

Standardized nonconstant features contain negative values. Thus exact
reconstruction is impossible for those entries with a Sigmoid reconstruction
head. This does **not** imply poor anomaly detection: reconstruction errors
can still separate the classes. At fixed threshold tau, L>tau guarantees an
alarm and U<=tau guarantees no alarm. The remaining rows are unresolved,
not necessarily jointly achievable by one network. Treat0.51 rounding and
alternative score units explicitly before interpreting a bound.

First run the separately specified zero-fit check in AEA_GEOMETRY_CHECK.md
after approval. It should validate novelty inputs/overlap and compare these
bounds with zero/half/clipped-input/daily-mean controls. No AEA implementation,
neural fit, new real-data scoring or cluster job has occurred in this step.

Later constructed model fixtures must check shapes/state transfer, attention
normalization and decoder-query dependence, previous-output feedback,
gradient reachability, bounded output, finite updates, reconstruction on a
learnable [0,1] example, range limits on negative targets, and save/reload.
Only after those and the data/score checks should a bounded pilot be proposed.

## Targets and interpretation boundary

TableIII generalized AEA p00/p30 targets: DR94.1/80.8, FA5.2/18.4,
SP94.8/81.6, PR94.5/80.6, ACC94.4/81.1, F194.3/80.7, AUC94.0/80.3.
The literal0.51 operating point and all-cutoff diagnostics must both be
reported; a chosen test cutoff is not validated calibration. Preserve
original/synthetic strata and dependence. No confidence interval from
independent-row assumptions or full-population claim is licensed by the pilot.

A classification loss alone can tolerate an intermediate representation that
differs from the input while a downstream map compensates. A constructed
counterexample demonstrates that logical distinction; it does not prove the
published ensemble cannot learn useful reconstruction incidentally. Tests of
pretraining, joint reconstruction and classification-only training must remain
separate mechanism questions, not silently inherited from standalone AEA.
