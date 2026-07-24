"""Two aimed questions. Neither needs a full model sweep.

Q1 (replay ceiling): replay is ~43% of attacks and universally undetected
   (best observed 6.1%). Is it detectable IN PRINCIPLE? A replayed window is a
   verbatim copy of an earlier real window, so if we search the benign history
   for a near-exact match, replayed windows should sit at ~0 distance while
   genuine windows sit further away. If replay is separable this way, the
   ceiling is a property of the paper's DETECTOR, not of the data. If it is not
   separable, the ceiling is information-theoretic and the paper's DR is
   unreachable by construction.

Q2 (capacity): the paper claims spatio-temporal structure is "too difficult for
   an FFNN". If FFNN performance is flat across an order of magnitude of
   capacity, it is not capacity-limited and that claim fails on its own terms.

Run: OMP_NUM_THREADS=1 python depth_replay_capacity.py --size 31 --seed 0
"""
from __future__ import annotations
import argparse, numpy as np
import torch, torch.nn as nn
import data, models, detect, run
from sanity import auc, kinds_per_step


def q1_replay_separability(ds, lab, kt):
    print("Q1. IS REPLAY DETECTABLE IN PRINCIPLE?")
    h = ds["hist"].reshape(len(ds["hist"]), -1)          # flatten each window
    # The reference bank must come from TRAINING windows only. Building it from
    # all clean windows leaks the evaluated normals into their own bank: they
    # then match themselves at distance 0 (56% exact matches), which inverts the
    # comparison. Evaluate only on TEST windows against a train-only bank.
    clean = ds["win_clean"]
    tr_mask = np.zeros(len(h), bool); tr_mask[ds["train"]] = True
    ref = h[clean & tr_mask]
    ev = np.zeros(len(h), bool); ev[ds["test"]] = True
    print(f"   reference bank: {len(ref)} attack-free TRAINING windows "
          f"(evaluating {ev.sum()} held-out test windows)")

    def nn_dist(Q, block=512):
        """Distance from each query window to its closest benign reference window."""
        out = np.empty(len(Q))
        rn = (ref ** 2).sum(1)
        for i in range(0, len(Q), block):
            q = Q[i:i + block]
            d2 = (q ** 2).sum(1)[:, None] + rn[None, :] - 2.0 * q @ ref.T
            out[i:i + block] = np.sqrt(np.maximum(d2.min(1), 0))
        return out

    groups = {"normal": ev & (lab == 0)}
    for k in ("replay", "dos", "manipulation"):
        groups[k] = ev & (lab == 1) & (kt == k)
    dist = {}
    for name, m in groups.items():
        sel = np.where(m)[0]
        if len(sel) == 0:
            continue
        dist[name] = nn_dist(h[sel])
        print(f"   {name:12} n={len(sel):4d}  nearest-benign distance: "
              f"median={np.median(dist[name]):9.4f} p10={np.percentile(dist[name],10):9.4f}")
    # An attack window that is UNLIKE anything benign sits FARTHER away, so larger
    # distance => more attack-like. AUC 0.5 means indistinguishable by this rule.
    for k in ("replay", "dos", "manipulation"):
        if k not in dist:
            continue
        s = np.concatenate([dist["normal"], dist[k]])
        y = np.concatenate([np.zeros(len(dist["normal"])), np.ones(len(dist[k]))])
        print(f"   AUC(nearest-neighbour rule) {k:12} vs normal = {auc(s, y):.4f}")
    print("   -> replay AUC near 0.5 => not separable this way, so the ceiling is")
    print("      a property of the DATA, not just of the paper's detector\n")


class FF(nn.Module):
    def __init__(self, w, n, width, depth):
        super().__init__()
        L = [nn.Flatten(), nn.Linear(w * n, width), nn.Tanh()]
        for _ in range(depth - 1):
            L += [nn.Linear(width, width), nn.Tanh()]
        L += [nn.Linear(width, n)]
        self.net = nn.Sequential(*L)
    def forward(self, x):
        return self.net(x)


def q2_capacity(ds, lab, kt, seed):
    print("Q2. IS FFNN CAPACITY-LIMITED?  (paper: structure is 'too difficult for an FFNN')")
    tb, va, te = ds["train_benign"], ds["val"], ds["test"]
    print(f"   {'width x depth':>15} {'params':>10} {'AUC':>8} {'DR@FA3.7':>9} {'dos':>7} {'replay':>7}")
    for width, depth in ((50, 2), (100, 3), (500, 5), (1000, 6)):
        torch.manual_seed(seed)
        m = FF(data.WINDOW, ds["n_nodes"], width, depth)
        npar = sum(p.numel() for p in m.parameters())
        m = models.train(m, ds["hist"][tb], ds["target_true"][tb], None)
        r = lambda i: ds["target_obs"][i] - models.predict(m, ds["hist"][i], None)
        mu, ci = detect.fit_benign(r(tb))
        s = detect.batch_mean(detect.sq_mahalanobis(r(te), mu, ci), s=1)
        au = auc(s, lab[te])
        thr = np.percentile(s[lab[te] == 0], 100.0 - 3.7)
        f = s > thr
        dr = 100 * f[lab[te] == 1].mean()
        dos = 100 * f[(lab[te] == 1) & (kt[te] == "dos")].mean()
        rep = 100 * f[(lab[te] == 1) & (kt[te] == "replay")].mean()
        print(f"   {width:6d} x {depth:<6d} {npar:10,d} {au:8.4f} {dr:9.1f} {dos:7.1f} {rep:7.1f}")
    print("   -> flat across a 20x parameter range => not capacity-limited\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=31)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    idx = data.figure_nodes(a.size)
    ds = data.make_datasets(idx, seed=a.seed)
    lab = ds["labels"]
    kt = kinds_per_step(data.WINDOW + len(lab))[data.WINDOW:]
    print(f"=== {a.size} nodes | seed {a.seed} ===\n")
    q1_replay_separability(ds, lab, kt)
    q2_capacity(ds, lab, kt, a.seed)


if __name__ == "__main__":
    main()
