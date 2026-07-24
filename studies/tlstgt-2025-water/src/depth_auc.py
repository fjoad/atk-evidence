"""Depth pass aimed at H1/H2 only. Deliberately small: 2 models, not 9.

H1 (confound): the trained models' apparent advantage over a zero-parameter rule
   may be an operating-point artifact -- they run at FA 12-15% while the trivial
   detector runs at 3.7%. AUC is threshold-free and settles it.
H2 (mechanism): if their gain is real, it must come from replay/DoS, because
   manipulation is already saturated at 100% by the trivial rule. Measured at a
   MATCHED false-alarm rate so the comparison is like-for-like.

Run: OMP_NUM_THREADS=1 python depth_auc.py --size 31 --seed 0
"""
from __future__ import annotations
import argparse, numpy as np
import data, models, detect, run
from sanity import auc, kinds_per_step

MATCH_FA = 3.7          # the trivial detector's operating point at 31 nodes


def dr_at_fa(scores, labels, kinds, fa_target):
    """Set the threshold to hit fa_target on normal samples, then report DR overall
    and per attack type. This is the like-for-like comparison."""
    normal = labels == 0
    thr = np.percentile(scores[normal], 100.0 - fa_target)
    flag = scores > thr
    out = {"FA": 100 * flag[normal].mean(), "DR": 100 * flag[labels == 1].mean()}
    for k in ("replay", "dos", "manipulation"):
        m = (labels == 1) & (kinds == k)
        out[k] = 100 * flag[m].mean() if m.any() else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=31)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    models.torch.manual_seed(a.seed)

    idx = data.figure_nodes(a.size)
    A = data.graph()
    adj = run.adjacency(A, idx, "real", a.seed)
    ds = data.make_datasets(idx, seed=a.seed)
    lab = ds["labels"]
    kt = kinds_per_step(data.WINDOW + len(lab))[data.WINDOW:]
    te, tb, va = ds["test"], ds["train_benign"], ds["val"]
    cfg = dict(train="benign", thresh="fa5", transformer="nodes", errfit="normal",
               adj="real", window=data.WINDOW, ablate="none")

    print(f"=== depth: H1/H2 | {a.size} nodes | seed {a.seed} | matched FA={MATCH_FA}% ===\n")
    rows = {}

    # zero-parameter references
    for name, fn in (("ZSCORE", lambda i: ds["target_obs"][i]),
                     ("PERSIST", lambda i: ds["target_obs"][i] - ds["hist"][i][:, -1, :])):
        mu, ci = detect.fit_benign(fn(tb))
        s = detect.batch_mean(detect.sq_mahalanobis(fn(te), mu, ci), s=1)
        rows[name] = (auc(s, lab[te]), dr_at_fa(s, lab[te], kt[te], MATCH_FA))

    # two trained models only: the paper's simplest deep baseline and its proposal
    for name in ("ffnn", "stgt"):
        an = adj if name in run.GRAPH else None
        tr, tgt = run._train_view(ds, cfg)
        m = models.train(models.build(name, data.WINDOW, ds["n_nodes"]),
                         ds["hist"][tr], tgt[tr], an)
        r = lambda i: run.resid_forecaster(m, ds, i, an)
        mu, ci = detect.fit_benign(r(tb))
        s = detect.batch_mean(detect.sq_mahalanobis(r(te), mu, ci), s=1)
        rows[name.upper()] = (auc(s, lab[te]), dr_at_fa(s, lab[te], kt[te], MATCH_FA))

    print(f"{'model':9} {'AUC':>7} | at matched FA:  {'FA':>5} {'DR':>6} "
          f"{'replay':>7} {'dos':>6} {'manip':>7}")
    for k, (au, d) in rows.items():
        print(f"{k:9} {au:7.4f} |                {d['FA']:5.1f} {d['DR']:6.1f} "
              f"{d['replay']:7.1f} {d['dos']:6.1f} {d['manipulation']:7.1f}")
    print("\nH1: if AUC(trained) ~ AUC(trivial), the advantage was the operating point.")
    print("H2: any real gain must appear in the replay/dos columns.")
    print("Paper's STGT for reference: DR=74.7 (no FA reported).")


if __name__ == "__main__":
    main()
