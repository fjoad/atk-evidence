"""Paper-literal data for the TL-STGT paper (Section II, eqs 1-4).

The paper generates benign readings with epanetCPA, then creates an equal number
of malicious samples by MODIFYING benign samples with three equations:
  Replay (eq2):        X^m_t = X^b_{t-Dt}
  DoS (eq3):           X^m_t = X^b_{t-1}
  Manipulation (eq4):  X^m_t = X^b_t + d,  d in [-5,5] step 0.2
Data is "evenly split between normal and attack" (50/50), split 80/10/10 with
equal samples per class in each set, over graphs of 10/20/31 nodes. The paper
NAMES betweenness centrality (eq1) as the reduction criterion, but Figure 1
prints the exact node sets and connections of all three graphs, so we transcribe
those verbatim instead of recomputing a selection (FIGURE1_NODES below).

Benign readings = the recovered DeepH2O clean C-Town series (real epanetCPA normal
operation). Real 31-node adjacency = processed_scada_adj_matrix.csv. Attacks are
applied to raw readings; standardization (StandardScaler) is fit on benign
training readings only. See PAPER_SPEC.md for the full fidelity audit.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

# Data location. Override with WATER_DATA so the same code runs locally and on a
# cluster without edits (cluster jobs export WATER_DATA=<data dir>).
DEEPH2O = os.environ.get(
    "WATER_DATA", os.path.expanduser("~/data/BATADAL/DeepH2O"))
WINDOW = 10
DELTAS = np.round(np.arange(-5.0, 5.0001, 0.2), 2)   # eq4 grid

# eq-4 delta scaling. The paper writes X^m = X^b + delta with delta in [-5,5] and
# gives no units, so "raw" is the literal reading. But C-Town sensor scales span
# 6600x (benign sd 0.0063 on J280 up to 41.7 on PU2), so one absolute delta is a
# 788-sigma shove on J280 and a 0.12-sigma nudge on PU2. That makes the residual
# equal to the injected perturbation, which is why every architecture returns
# identical metrics. "sigma" is the documented alternative branch, NOT a repair of
# the literal track: it keeps the same [-5,5] grid but reads it in per-sensor
# benign standard deviations. Both are run and reported.
DELTA_SCALES = ("raw", "sigma")


def _benign_and_adj():
    Xb = pd.read_csv(os.path.join(DEEPH2O, "processed_clean_scada_dataset.csv")).to_numpy(float)
    A = pd.read_csv(os.path.join(DEEPH2O, "processed_scada_adj_matrix.csv"), header=None).to_numpy(float)
    A = ((A + A.T) > 0).astype(float)
    np.fill_diagonal(A, 0.0)                          # eq5 sums over neighbours (no self-loop)
    return Xb, A


# Dataset column order (== processed_clean_scada_dataset.csv header, prefixes
# stripped). Node index i in the 31-node adjacency corresponds to FEATURE_LABELS[i].
FEATURE_LABELS = ["T1","T2","T3","T4","T5","T6","T7","PU1","PU2","PU3","PU4",
                  "PU5","PU6","PU7","PU8","PU9","PU10","PU11","V2","J280","J269",
                  "J300","J256","J289","J415","J302","J306","J307","J317","J14","J422"]
_LABEL_IX = {l: i for i, l in enumerate(FEATURE_LABELS)}

# EXACT node configurations of the reduced graphs, READ DIRECTLY FROM FIGURE 1
# (panels a/b/c) -- the printed graphs are the authority, not a recomputation.
# The 10-node set contains PU8, which a plain betweenness-top-10 would drop in
# favour of J422; the figure wins. Sets are nested (10 < 20 < 31); edges are the
# induced subgraph of the real C-Town adjacency (all three stay connected: 12/24/44 edges).
FIGURE1_NODES = {
    10: ["J289","PU8","T1","PU1","J302","J300","J14","V2","J317","J269"],
    20: ["J415","PU6","PU7","J300","PU4","PU5","J256","T7","J14","J307","J422",
         "PU1","J289","J306","J317","PU8","T1","V2","J302","J269"],
    31: FEATURE_LABELS,
}


def figure_nodes(target: int):
    """Column indices for the exact 10/20/31-node graph drawn in Figure 1 of the paper."""
    if target not in FIGURE1_NODES:
        raise ValueError(f"no Figure-1 graph with {target} nodes (paper prints 10/20/31)")
    return sorted(_LABEL_IX[l] for l in FIGURE1_NODES[target])


def _build_observed(Xb, rng, block, attack_frac=1.0, delta_scale="raw"):
    """Alternating normal/attack OPERATIONS (contiguous blocks), ~50/50 overall.

    Within an attack block the observed readings are corrupted per eqs 2-4 (so both
    the history and the target inside an attack period are affected, as physically
    they would be). ``attack_frac`` < 1 corrupts only a random subset of the sensors
    per block (robustness knob; the frozen paper-literal run uses 1.0). Returns the
    observed series and a per-timestep 0/1 label.

    ``delta_scale`` selects the eq-4 branch (see DELTA_SCALES):
      "raw"   — paper-literal: delta is an absolute offset on the reading.
      "sigma" — documented branch: delta is measured in that sensor's benign
                standard deviations, so one delta means the same relative
                disturbance everywhere.
    """
    T, N = Xb.shape
    Xobs = Xb.copy()
    lab = np.zeros(T, dtype=int)
    sd = Xb.std(0); sd[sd == 0] = 1.0          # benign per-sensor scale (branch only)
    kinds = ["replay", "dos", "manipulation"]
    start, toggle, ki = WINDOW, 1, 0        # toggle 1 => attack block first (after initial window)
    while start < T:
        end = min(start + block, T)
        if toggle == 1:
            kind = kinds[ki % 3]; ki += 1
            dt = int(rng.integers(1, WINDOW + 1))
            cols = (np.arange(N) if attack_frac >= 1.0
                    else rng.choice(N, size=max(1, int(round(attack_frac * N))), replace=False))
            for t in range(start, end):
                if kind == "replay":
                    new = Xb[max(0, t - dt)]                # eq2: a past reading
                elif kind == "dos":
                    new = Xb[start - 1]                     # eq3 sustained: frozen last valid reading
                else:
                    d = rng.choice(DELTAS, size=N)          # eq4: +delta
                    new = Xb[t] + (d * sd if delta_scale == "sigma" else d)
                Xobs[t, cols] = new[cols]
                lab[t] = 1
        start, toggle = end, toggle ^ 1
    return Xobs, lab


def make_datasets(node_idx, seed=0, block=60, attack_frac=1.0, delta_scale="raw",
                  window=None, ablate="none"):
    """Build the paper's 50/50 (normal/attack operations) dataset for one graph size.

    ``ablate`` destroys one kind of structure to test whether it was ever being
    used. The paper argues a graph+transformer is needed because spatial and
    temporal features are too hard for an FFNN; that premise implies destroying
    them should hurt.
      space: shift each sensor's series independently, destroying CROSS-SENSOR
             correlation while leaving each sensor's own distribution and
             temporal behaviour intact.
      time:  randomly permute the time steps inside each input window,
             destroying temporal ORDER while leaving the multiset of readings.
    """
    W = WINDOW if window is None else int(window)
    Xb_all, _ = _benign_and_adj()
    Xb = Xb_all[:, node_idx].astype(np.float64)          # (T, N) benign, RAW
    T, N = Xb.shape
    rng = np.random.default_rng(seed)
    if ablate == "space":                                # decorrelate sensors
        shift_rng = np.random.default_rng(50_000 + seed)
        Xb = np.column_stack([np.roll(Xb[:, j], int(shift_rng.integers(1, T))) for j in range(N)])
    Xobs, lab_t = _build_observed(Xb, rng, block, attack_frac, delta_scale)

    # Attack placement stays anchored at the module WINDOW so the schedule is
    # identical across window lengths and runs remain comparable.
    pos = np.arange(max(W, WINDOW), T)                   # classified timesteps
    hist = np.stack([Xobs[t - W:t] for t in pos])        # (M, W, N) OBSERVED history
    if ablate == "time":                                 # destroy temporal ORDER only
        t_rng = np.random.default_rng(60_000 + seed)
        hist = np.stack([h[t_rng.permutation(W)] for h in hist])
    target_true = np.stack([Xb[t] for t in pos])         # true (clean) reading
    target_obs = np.stack([Xobs[t] for t in pos])        # observed reading (corrupted in attack blocks)
    labels = lab_t[pos]
    win_clean = np.array([lab_t[t - W:t + 1].sum() == 0 for t in pos])   # fully-normal window

    M = len(pos)
    i80, i90 = int(0.8 * M), int(0.9 * M)
    tr, va, te = np.arange(i80), np.arange(i80, i90), np.arange(i90, M)

    normal_readings = Xb[pos[np.array([lab_t[t] == 0 for t in pos])]]
    mu = normal_readings.mean(0); sd = normal_readings.std(0); sd[sd == 0] = 1.0
    def z(a): return ((a - mu) / sd).astype(np.float32)
    hist, target_true, target_obs = z(hist), z(target_true), z(target_obs)

    return dict(
        hist=hist, target_true=target_true, target_obs=target_obs, labels=labels,
        train=tr, val=va, test=te,
        train_benign=tr[win_clean[tr]],                  # forecasters train on fully-normal windows
        win_clean=win_clean,                             # history window free of attack readings
        n_nodes=len(node_idx),
    )


def graph():
    """Return the real 31-node adjacency (symmetric, no self-loop)."""
    _, A = _benign_and_adj()
    return A
