# TL-STGT — full method transcription (implement-from-scratch plan)

> **Read [`README.md`](README.md) first** — it carries the audit framing this document assumes.

Every step of the paper's method, in my own words, transcribed page by page so
nothing is skimmed. Tags: **[STATED]** explicit in paper · **[AMBIG]** paper
silent/underspecified · **[CONTRA]** paper internally contradicts itself. After
each part: what our code does + whether it matches.

Pipeline at a glance:
```
epanetCPA benign readings ─▶ inject attacks (replay/DoS/manip, eqs 2-4) ─▶ 50/50 data
      │                                                                        │
      ▼                                                                        ▼
C-Town graph ─(exact node sets from Fig 1)▶ {10,20,31}-node graphs  80/10/10 split (bal.)
      │                                                                        │
      ▼                                                                        ▼
per graph size:  GCN(eq5) ▶ GRU(eqs6-9) ▶ MaxPool ▶ Dense ▶ Transformer(eqs10-11) ▶ Dense ▶ next-step Ŷ
      │                        (train MSE, eq12, Adam)                          │
      ▼  TL: 10─▶20─▶31 warm-start + freeze earlier layers (Alg.1)             ▼
detection: E=Y−Ŷ ▶ squared Mahalanobis ▶ mean over batch of S ▶ threshold(val) ▶ attack?
      ▼
metrics: F1, ACC, DR   (Table I, 9 models × 3 sizes)
```

## PART A — DATA (Section II)

- **A1 [STATED]** Network = C-Town WDS from BATADAL [15]. Model as graph
  G=(V,E): nodes = WDS components (tanks, pumps, valves, junctions), edges =
  connection pipes. Adjacency A: A_ij=1 iff i,j directly connected, else 0.
- **A2 [STATED]** Three graph sizes 10/20/31. The paper NAMES betweenness
  centrality (eq1) `C_B(v)=Σ_{i≠v≠j} η_ij(v)/η_ij` as the reduction criterion,
  BUT **Figure 1 prints the exact node sets and connections of all three graphs**
  (panels a/b/c). We transcribe the figure verbatim rather than recompute eq1 —
  and must, because eq1 disagrees with it: a plain betweenness-top-10 keeps J422
  and drops PU8, yet Fig 1(a) keeps PU8. The figure is the authority.
  - **[RESOLVED] nesting:** 10 ⊂ 20 ⊂ 31, verified from the figure node sets.
  - **[RESOLVED] edges:** induced subgraph of the real 31-node adjacency on the
    kept nodes. Verified against the drawing: Fig 1(a) matches all 12 edges
    exactly; Fig 1(b) matches all 24 (its dense bottom row is the two *crossing*
    edges J317–V2 and PU8–J302, not a chain — checked against the real
    adjacency, which has PU8–J302 & J317–V2 but no PU8–J317/PU8–V2). Fig 1(c) is
    the real graph itself. Connected: 12/24/44 edges.
  - Node sets (dataset column order): 10 = {J289,PU8,T1,PU1,J302,J300,J14,V2,
    J317,J269}; 20 = 10 + {J415,PU4,PU5,PU6,PU7,T7,J256,J306,J307,J422}; 31 = all.
- **A3 [STATED]** Temporal data via **epanetCPA**, **1,400 hours**, **evenly
  split between normal and attack operations**, incl. replay/DoS/manipulation,
  **for each of the three WDSs**. Matrix X∈R^{|T|×|V|}, X_{t,i}=reading of node
  i at time t (tank level / flow / pressure), benign X^b or malicious X^m.
- **A4 [STATED]** Benign X^b = normal-operation readings.
- **A5 [STATED]** Malicious X^m = **modify benign samples**:
  - Replay (eq2): `X^m_{t,i}=X^b_{t-Δt,i}`. Δt = current−replayed step diff.
  - DoS (eq3): `X^m_{t,i}=X^b_{t-1,i}`; "last valid reading before disruption
    freezes during the attack."
  - Manipulation (eq4): `X^m_{t,i}=X^b_{t,i}+δ_{t,i}`, `-5≤δ≤5`, 0.2 steps,
    "selected based on experimental tuning."
  - **[AMBIG]** Δt value; which/how-many sensors an attack hits; how long an
    attack "operation" lasts; δ per-node random vs tuned constant.
- **A6 [STATED]** Split 80/10/10 train/val/test, **equal samples per class in
  each set** (50/50).
  - **[CONTRA]** 50/50 in the *training* set conflicts with training a
    forecaster/reconstructor on *normal* data (standard anomaly detection).
- **OUR CODE:** benign = DeepH2O clean C-Town (real epanetCPA normal ops); real
  31-node adjacency; attacks eqs 2-4 in contiguous "operation" blocks; 50/50;
  80/10/10; **exact Figure-1 node sets hard-coded** (`data.figure_nodes`), edges
  = induced subgraph (no recomputation). Deviation from A3: we use one benign
  series + column-subset per size, not 3 separate 1,400-h runs (physics
  identical, so equivalent). The A6 contradiction is **swept as Ambiguity Axis 1**
  (`--train balanced|benign`); the literal 50/50 reading is the default. A5's δ
  units are **Axis 2** — note δ=5 is 788σ on J280 and 0.12σ on PU2, so "meaningful
  attack size" is sensor-dependent (EVIDENCE.md C2).

## PART B — STGT model (Section III, Fig. 2). Order in Fig. 2:
`Input Graph → Recurrent Graph Convolution → Graph Max Pooling → Dense →
Transformer[MHA→Add&Norm→MHA→Add&Norm] → Dense → Output`

- **B1 [STATED]** Graph conv (eq5): `h_i^{l+1}=tanh(Σ_{j∈N(i)} A_ij/√(d_i d_j) ·
  W^{lg} h_j^l)`. tanh; neighbour sum; symmetric-degree norm; **no self-loop
  term shown**.
- **B2 [STATED]** GRU (eqs 6-9), standard, on the graph-conv output. 64 hidden
  (F3). "We add **a** GRU layer" (singular).
- **B3 [STATED]** Transformer: GRU hidden states → **global max pooling** +
  **dense** → "fixed-sized input to the transformer" → MHA (eq10
  `softmax(QPᵀ/√d_p)F`), MultiHead (eq11), **two** MHA+Add&Norm blocks (Fig 2)
  → **final dense** = "reconstructed vector of the next time step" (dim |V|).
  - **[CONTRA]** global max-pool BEFORE the transformer collapses the time
    dimension, yet the text says the transformer captures "longer-range
    **temporal** dependencies." Pool-then-transformer ⇒ transformer can only be
    spatial (over nodes). The two readings (transformer-over-nodes vs
    -over-time) are both defensible; the paper is inconsistent.
    **SWEPT as Ambiguity Axis 5** (`--transformer nodes|time`); both readings are
    run and reported. NOTE: an earlier entry here recorded "both give STGT-31
    F1 ≈ 30" — those numbers came from the since-fixed threshold bug
    (EVIDENCE.md C3) and are **RETRACTED**. Do not cite them.
- **B4 [STATED]** Loss (eq12) MSE `(1/Z)Σ_z‖Y[z]−Ŷ[z]‖²`; Adam; end-to-end;
  mini-batches; several epochs.
- **B5 [STATED]** Input X∈R^{Z×S×|V|}, S=batch size, |V|=nodes; output =
  prediction of all node features for a given timestamp (one-step-ahead).
  - **[AMBIG]** no explicit sliding-window length; temporal context = the batch
    of S consecutive readings. We use a window W=10 as its stand-in.
- **OUR CODE:** GCN(eq5, tanh, no self-loop) → GRU(64) → temporal max-pool →
  Dense(128,ReLU) → Transformer(8 heads, 2 blocks) → per-node head. Follows the
  Fig-2 order; transformer-over-nodes (B3 CONTRA) — the -over-time variant is
  being tested separately. **MATCH** on B1/B2/B4/F3; B3/B5 documented AMBIG.

## PART C — Transfer learning (Section IV, Algorithm 1, Fig. 3)

- **C1 [STATED]** Alg. 1: for graphs G_1..G_k: init W=W_0 (first) else previous
  W; compute Ŷ; loss (eq12); Adam update; if not last: transfer
  `W_{G_{g+1}}←W_{G_g}` **and freeze the weights in earlier layers**
  (`∂W^{frozen}/∂t=0`).
- **C2 [STATED]** Progressive 10→20→31; save weights, initialize next model.
  - **[AMBIG]** which layers count as "earlier" (we freeze GCN+GRU, fine-tune
    Dense+Transformer+head).
- **OUR CODE:** warm-start via load_state_dict (weights node-count independent),
  freeze GCN+GRU, fine-tune the rest, 10→20→31. **MATCH**.

## PART D — Detection (Section V-A)

- **D1 [STATED]** For deep & proposed models: compare inferences Ŷ with ground
  truth Y; error matrix `E=Y−Ŷ` (per-sample error vectors).
  - **[AMBIG]** "ground truth labels Y" wording, but the covariance-of-E maths
    means Y = true readings (regression targets), not 0/1 labels.
- **D2 [STATED]** Mahalanobis `Ξ=√((e−μ)ᵀ φ⁻¹ (e−μ))`, μ=global mean error,
  φ=covariance of E.
  - **[AMBIG]** whether μ,φ come from all E or normal-only E (we use normal).
- **D3 [STATED]** Mean **squared** Mahalanobis over **consecutive batches**:
  `meanΞ²=(1/S)Σξ_i²`, S=batch size; flag the batch if it exceeds a threshold.
  - **[AMBIG]** per-batch decision vs per-sample; requires attacks to be
    contiguous "operations" for the batch mean to carry signal.
- **D4 [STATED]** Threshold "determined based on the model's performance using
  the validation set."
  - **[AMBIG]** we use a fixed ~5% false-alarm operating point on normal val
    (max-F1-on-val degenerates to flag-all on a 50/50 set).
- **[AMBIG]** shallow models (SVM/RF/LGBM) detection method is NOT stated; we
  use supervised classification on the observed reading.
- **OUR CODE:** MATCH on D1-D3 math; D2/D4 documented AMBIG.

## PART E — Metrics (Section V-A)

- **E1 [STATED]** `F1=2TP/(2TP+FP+FN)`, `ACC=(TP+TN)/N`, `DR=TP/(TP+FN)`.
  Table I reports F1/ACC/DR only (no FA). — **MATCH exactly.**
- **E2 [STATED]** Also report training time (seconds) as model complexity.

## PART F — Hyperparameters (Section V-B; "sequential grid search")

- **F1 [STATED]** SVM γ0.07/C10; RF 10 trees, entropy; LGBM lr0.5/depth2/100. — MATCH
- **F2 [STATED]** FFNN 5 layers/500/tanh; LSTM 3/100/tanh; AE 4 layers/ReLU;
  all Adam. — MATCH
- **F3 [STATED]** TGCN 3 layers/64/lr0.001/Adam; proposed 4 layers/64 GRU/8
  heads/128 dense/ReLU/lr0.001/Adam. — MATCH (STGT "4 layers" read as
  GCN+GRU+2 transformer blocks; documented).

## PART G — Table I (target)
F1/ACC/DR for {SVM,RF,LGBM,FFNN,LSTM,AE,TGCN,STGT,TL-STGT} × {10,20,31}.

## Cross-check summary — did I skip anything?
All 12 equations transcribed (1 betweenness; 2-4 attacks; 5 GCN; 6-9 GRU;
10-11 attention; 12 loss). Detection formulas (Ξ, meanΞ²) and metric formulas
are text, all captured. Figs 1 (graphs), 2 (architecture), 3 (TL), 4 (timing)
accounted for. No numbered step or equation is unaddressed. The material
ambiguities (A2 edges, A6 train-balance, B3 transformer placement, B5 window,
D2 error-fit, D4 threshold, shallow-detection) are each flagged and either
matched or tested as a robustness variant.
