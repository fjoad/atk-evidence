# TL-STGT paper — exact method spec & fidelity audit

> **Read [`README.md`](README.md) first** — it carries the audit framing this document assumes.

Line-by-line reading of Ahasan, Joad, Atat, Thompson, Serpedin, Takiddin,
"Graph Transfer Learning-Based Attack Detection in Cyber-Physical Water
Distribution Systems," EUSIPCO 2025 (pp. 1912–1916), recording exactly what the
paper says and how faithfully our reproduction matches. Status flags:
**MATCH** (implemented as stated) · **FIX** (must change — currently wrong) ·
**AMBIG** (paper under-specified; documented assumption).

Re-read 2026-07-23 after the user (2nd author) insisted we implement the paper
literally rather than approximate. This file is the source of truth for the repro.

---

## 1. Data (Section II)

| # | Paper says (quote / eq) | Our repro | Status |
|---|---|---|---|
| D1 | "adopt the C-Town WDS benchmark dataset [14] from BATADAL [15]" | C-Town via recovered `.inp` / DeepH2O | MATCH |
| D2 | "We generate such readings using **EpanetCPA** … 1,400 hours of data **evenly split between normal and attack**" | benign readings from DeepH2O `processed_clean_scada_dataset.csv` (real epanetCPA C-Town normal ops); attacks synthesized (D4) to 50/50 | **FIX** (v1/v2 used BATADAL's own attack labels, not this) |
| D3 | Benign sample `X^b_{t,i}` = reading of node i at time t | node feature series | MATCH |
| D4 | Attacks **modify benign samples** (eqs 2–4): **Replay** `X^m_{t,i}=X^b_{t-Δt,i}`; **DoS** `X^m_{t,i}=X^b_{t-1,i}`; **Manipulation** `X^m_{t,i}=X^b_{t,i}+δ`, `-5≤δ≤5` in 0.2 steps | implement the three equations on the raw benign readings | **FIX** (this is NOT the `.cpa` closed loop; simpler) |
| D5 | Δt in replay = "time difference between current and replayed time step" (value unspecified) | random past offset (documented range) | AMBIG |
| D6 | "80%–10%–10% ratio with **equal number of samples per class in each set**" | balanced 50/50 in train, val, AND test | **FIX** (v2 balanced test only) |
| D7 | "three graphs, with 10, 20, and 31 nodes"; eq 1 (betweenness) named as criterion, but **Figure 1 prints the exact node sets/edges** of all three | **exact Fig-1 node sets hard-coded** (`data.figure_nodes`), induced-subgraph edges — verified to match the drawing (10-node 12/12 exact; 20-node 24/24, bottom row = crossing J317–V2 & PU8–J302). NB eq 1 alone would pick J422 over PU8, so the figure overrides it. | MATCH (figure-exact) |
| D8 | 31-node adjacency = the C-Town SCADA graph | real `processed_scada_adj_matrix.csv` (31×31; raw is directed w/ self-loops → symmetrized, no self-loop = **44 undirected edges**; 10/20-node induced = 12/24) | **FIX** (v1/v2 used a contracted approximation) |
| D9 | `X ∈ R^{Z×S×|V|}`, S = batch size, |V| = nodes; "prediction of all node features for a given timestamp" (one-step-ahead) | window→next-step forecast | AMBIG (no explicit sliding window; temporal context = a batch of S consecutive readings; we use window W as its stand-in) |

**Key correction (D2/D4):** the paper's attacks are three trivial per-reading
equations applied to benign data — epanetCPA only produces the *benign* series.
So the data is fully reproducible with no MATLAB and no `.cpa` closed-loop.

## 2. Proposed STGT architecture (Section III, Fig. 2)

Figure-2 order: Input Graph → **Recurrent Graph Convolution** → **Graph Max
Pooling** → **Dense** → **Transformer** (Multi-Head Attention → Add&Norm →
Multi-Head Attention → Add&Norm) → **Dense** → Output.

| # | Paper says | Our repro | Status |
|---|---|---|---|
| M1 | Graph conv eq 5: `h_i^{l+1}=tanh(Σ_{j∈N(i)} A_ij/√(d_i d_j) · W h_j)` — tanh, neighbour sum, **no explicit self-loop** | tanh GCN, `D^-1/2 A D^-1/2` with **no self-loop** (`normalized_adj`) | MATCH (self-loop removed 2026-07-23) |
| M2 | GRU: "We add **a** GRU layer", eqs 6–9 standard GRU, 64 hidden | 1 GRU layer, 64 hidden | MATCH |
| M3 | "hidden states from GRU passed through a **global max pooling** layer and a **dense** layer to create a fixed-sized input to the transformer" | temporal max-pool over the window, then Dense(→128) | AMBIG (paper's "global max pool" is shape-inconsistent with a per-node output; we pool over time and keep nodes as the transformer sequence) |
| M4 | Transformer: 2 MHA+Add&Norm blocks (Fig 2); attention eq 10 `softmax(QPᵀ/√d_p)F`; 8 heads | `TransformerEncoder(num_layers=2, nhead=8, d_model=128)` | MATCH |
| M5 | final Dense → "reconstructed vector of the next time step" (dim |V|) | per-node linear head → |V| | MATCH |
| M6 | Loss eq 12: MSE; Adam; end-to-end | MSELoss, Adam | MATCH |
| M7 | "proposed models use **4 layers**, 64 GRU hidden units, 8 transformer heads, **128 dense** neurons, **ReLU**, lr 0.001, Adam" | 64 GRU / 8 heads / 128 dense-ReLU / lr1e-3 Adam; "4 layers" read as GCN+GRU+2 transformer blocks | MATCH (layer-count reading documented) |

## 3. Transfer learning (Section IV, Algorithm 1)

Algorithm 1 verbatim: for each graph `G_1..G_k`: init `W_{G_g}=W_0` (g=1) else
`W_{G_{g-1}}`; train (Adam); if `g<k`: **transfer** `W_{G_{g+1}}←W_{G_g}` and
**"Freeze the weights in earlier layers"** `∂W^{frozen}/∂t=0`.

| # | Paper says | Our repro | Status |
|---|---|---|---|
| T1 | progressive 10→20→31, warm-start next from previous | load_state_dict across sizes (weights are node-count independent) | MATCH |
| T2 | freeze **earlier layers** after transfer | freeze GCN+GRU, fine-tune Dense+Transformer+head | MATCH ("earlier layers" = GCN+GRU is our documented reading) |

## 4. Detection (Section V-A)

Paper: `E = Y − Ŷ`; `μ` = global mean error, `φ` = cov of E;
`Ξ = √((e−μ)ᵀ φ⁻¹ (e−μ))`; then **mean squared** Ξ over **consecutive batches**
`meanΞ² = (1/S)Σξ_i²` (S = batch size); flag a batch if it exceeds a threshold
**tuned on the validation set**.

| # | Paper says | Our repro | Status |
|---|---|---|---|
| E1 | Mahalanobis on residual E=Y−Ŷ | same | MATCH |
| E2 | μ,φ from "E" (all errors?) | we fit on **benign** residuals | AMBIG (benign-fit is the standard anomaly-detection reading; paper ambiguous) |
| E3 | **squared** Ξ, averaged over consecutive batches of S; one decision per batch | squared Ξ, trailing mean window S=32, per-sample decision | AMBIG (per-sample vs per-batch; minor) |
| E4 | "threshold determined ... using the validation set" | **swept as Ambiguity Axis 4**: `--thresh fa5` (fixed ~5% FA on wholly-normal validation batches) and `--thresh maxf1` (maximise validation F1). | AMBIG — both readings are run and reported; neither is assumed. NB an earlier note here claimed max-F1 necessarily degenerates to flag-everything; that was observed under the since-fixed threshold bug (EVIDENCE.md C3) and is **not** treated as settled. Attacks are contiguous "operations" so the batch-averaged detector carries signal. |
| E5 | shallow models' detection method | (paper silent) supervised classifier | AMBIG |

## 5. Metrics (Section V-A) & Table I

`F1 = 2TP/(2TP+FP+FN)`, `ACC = (TP+TN)/N`, `DR = TP/(TP+FN)`. Table I reports
**F1, ACC, DR only** (no FA) at 10/20/31 nodes. — **MATCH** (our metrics exact).

## 6. FIX items — status

1. **Data (D2/D4/D6/D8)** — DONE. Benign DeepH2O readings + attacks via eqs 2–4,
   50/50, balanced 80/10/10, real adjacency, exact Figure-1 node sets.
2. **GCN self-loop (M1)** — DONE. `normalized_adj` is `D^-1/2 A D^-1/2` with no
   self-loop, matching eq 5's neighbour-only sum.
3. Remaining items are MATCH or documented AMBIG. Every AMBIG that materially
   changes results is a swept axis in `AMBIGUITY_REGISTER.md`, not a silent choice.

## 7. Honest residual limitations
- Benign series is DeepH2O canonical C-Town (real epanetCPA normal ops), not the
  paper's *own* 1,400-h run — the exact benign readings differ, the method is
  identical.
- `Δt`, attack-block length, layer-count reading, and the shallow models'
  detection rule remain documented assumptions (Axes 8–10, some still PENDING).
- **Seed sensitivity is material**: TGCN spreads 23.7 F1 points and STGT 11.8
  across 5 seeds on identical hardware/software/data. No single-seed number from
  this repo is quotable; report distributions.
