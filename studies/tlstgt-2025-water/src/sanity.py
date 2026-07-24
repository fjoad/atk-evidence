"""Breadth-first sanity checks. Many small questions, cheaply, before any depth.

Almost nothing here trains a model. The point is to learn where the answer
already lives in the DATA, so that expensive runs are aimed rather than sprayed.
Each check prints a one-line verdict and the number behind it.

Run:  OMP_NUM_THREADS=1 python sanity.py [--size 31]
"""
from __future__ import annotations
import argparse, numpy as np
import data, detect


def auc(scores, labels):
    """Rank-based AUC; no sklearn needed, no fitting."""
    s = np.asarray(scores, float); y = np.asarray(labels).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def kinds_per_step(T, block=60):
    kinds = ["replay", "dos", "manipulation"]
    out = np.array([""] * T, dtype=object)
    start, toggle, ki = data.WINDOW, 1, 0
    while start < T:
        end = min(start + block, T)
        if toggle == 1:
            out[start:end] = kinds[ki % 3]; ki += 1
        start, toggle = end, toggle ^ 1
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--size", type=int, default=31)
    a = ap.parse_args()
    idx = data.figure_nodes(a.size)
    ds = data.make_datasets(idx, seed=0)
    lab = ds["labels"]
    kt = kinds_per_step(data.WINDOW + len(lab))[data.WINDOW:]
    obs, true = ds["target_obs"], ds["target_true"]
    te = ds["test"]
    print(f"=== breadth-first sanity | {a.size} nodes | {len(lab)} samples "
          f"({lab.mean():.0%} attack) ===\n")

    # ---- 1. Do the attacks change the reading at all? -----------------------
    print("1. ATTACK VISIBILITY -- how far does each attack move the reading?")
    print("   (|observed - true benign| in standardized units; 0 => invisible by value)")
    for k in ("replay", "dos", "manipulation"):
        m = (lab == 1) & (kt == k)
        if not m.any():
            continue
        d = np.abs(obs[m] - true[m])
        print(f"   {k:12} mean={d.mean():7.3f}  median={np.median(d):7.3f}  "
              f"max={d.max():9.3f}  frac<0.1sd={np.mean(d < 0.1):.1%}")
    dn = np.abs(obs[lab == 0] - true[lab == 0])
    print(f"   {'normal':12} mean={dn.mean():7.3f}  (sanity: must be 0)\n")

    # ---- 2. Is a single sensor enough? --------------------------------------
    print("2. SINGLE-SENSOR SEPARABILITY -- best AUC using ONE sensor, no model")
    aucs = [auc(np.abs(obs[:, j]), lab) for j in range(obs.shape[1])]
    best = int(np.nanargmax(aucs))
    print(f"   best sensor = {data.FEATURE_LABELS[idx[best]]:6} AUC={aucs[best]:.4f}   "
          f"median sensor AUC={np.nanmedian(aucs):.4f}")
    print(f"   -> AUC ~0.5 means that sensor alone carries no signal\n")

    # ---- 3. Zero-parameter detectors, through the real pipeline -------------
    print("3. ZERO-PARAMETER DETECTORS (identical detect path as every model)")
    tb, va = ds["train_benign"], ds["val"]
    for name, fn in (("zscore ", lambda i: obs[i]),
                     ("persist", lambda i: obs[i] - ds["hist"][i][:, -1, :])):
        for S in (1, 32):
            m = detect.evaluate_forecaster(fn(tb), fn(va), lab[va], fn(te), lab[te],
                                           val_clean=ds["win_clean"][va], batch_s=S)
            print(f"   {name} S={S:2}: F1={m['F1']:5.1f} ACC={m['ACC']:5.1f} "
                  f"DR={m['DR']:5.1f} FA={m['FA']:5.1f}")
    print("   -> compare against the paper's STGT F1=76.3 / ACC=84.8 / DR=74.7\n")

    # ---- 4. Per-attack-type ceiling for a zero-parameter detector -----------
    print("4. WHICH ATTACKS ARE DETECTABLE AT ALL (zscore detector, S=1)")
    r = lambda i: obs[i]
    mu, ci = detect.fit_benign(r(tb))
    s_te = detect.batch_mean(detect.sq_mahalanobis(r(te), mu, ci), s=1)
    thr = np.percentile(detect.batch_mean(detect.sq_mahalanobis(r(va), mu, ci), s=1)[
        detect.clean_batch_mask(lab[va], ds["win_clean"][va], s=1)], 95.0)
    flag = s_te > thr
    for k in ("replay", "dos", "manipulation"):
        m = (lab[te] == 1) & (kt[te] == k)
        if m.any():
            print(f"   {k:12} DR={100*flag[m].mean():5.1f}%")
    print(f"   {'(false alarm)':12} FA={100*flag[lab[te]==0].mean():5.1f}%")
    print("   -> a structural ceiling here binds every model equally\n")

    # ---- 4b. Is DoS catchable by the RIGHT trivial test? --------------------
    # H3: the paper detects via one-step forecast residual. Fed frozen history a
    # forecaster predicts the frozen value, so residual ~ 0 and DoS is invisible.
    # But "the reading stopped changing" is trivially testable directly.
    print("4b. STUCK-SENSOR TEST (2-line rule the paper's method cannot express)")
    h = ds["hist"]
    nochange = np.mean(np.abs(np.diff(h, axis=1)) < 1e-9, axis=(1, 2))  # frac frozen in window
    for k in ("replay", "dos", "manipulation"):
        m = (lab == 1) & (kt == k)
        if m.any():
            print(f"   {k:12} mean frozen-fraction of window = {nochange[m].mean():.3f}")
    print(f"   {'normal':12} mean frozen-fraction of window = {nochange[lab == 0].mean():.3f}")
    print(f"   AUC of the stuck rule for DoS-vs-normal = "
          f"{auc(nochange[(lab == 0) | (kt == 'dos')], (kt == 'dos')[(lab == 0) | (kt == 'dos')]):.4f}")

    # ---- 4c. Composite of two trivial rules --------------------------------
    # H4: z-score catches manipulation, stuck catches DoS. Together, with zero
    # parameters, how much of the attack set is covered?
    print("\n4c. COMPOSITE TRIVIAL DETECTOR (z-score OR stuck), thresholds from validation")
    zs = np.max(np.abs(obs), axis=1)                       # worst-sensor deviation
    nrm_v = (lab[va] == 0)
    tz = np.percentile(zs[va][nrm_v], 95.0)
    tn = np.percentile(nochange[va][nrm_v], 99.0)
    flag_c = (zs > tz) | (nochange > tn)
    m = detect._metrics(lab[te], flag_c[te].astype(int))
    print(f"   composite : F1={m['F1']:5.1f} ACC={m['ACC']:5.1f} DR={m['DR']:5.1f} FA={m['FA']:5.1f}")
    for k in ("replay", "dos", "manipulation"):
        sel = (lab[te] == 1) & (kt[te] == k)
        if sel.any():
            print(f"     {k:12} DR={100*flag_c[te][sel].mean():5.1f}%")
    print("   -> compare with the paper's STGT: F1=76.3 ACC=84.8 DR=74.7\n")

    # ---- 5. How much structure is there to learn? ---------------------------
    # MUST be measured on attack-free windows. Using observed history here reads
    # ~0 correlation at 31 nodes purely because the 788-sigma manipulation spikes
    # on J280 dominate the estimate -- an artifact of the attack, not a property
    # of the benign process.
    print("5. IS THERE STRUCTURE TO LEARN?  (attack-free windows only)")
    h = ds["hist"]
    clean = ds["win_clean"]
    print(f"   clean windows: {clean.sum()} / {len(clean)}")
    C = np.corrcoef(true[clean], rowvar=False); np.fill_diagonal(C, 0)
    print(f"   spatial : mean |cross-sensor corr| = {np.nanmean(np.abs(C)):.4f}")
    hc = h[clean]
    lag1 = np.corrcoef(hc[:, :-1, :].ravel(), hc[:, 1:, :].ravel())[0, 1]
    print(f"   temporal: lag-1 corr within window = {lag1:.4f}")
    nxt = np.corrcoef(true[clean].ravel(), hc[:, -1, :].ravel())[0, 1]
    print(f"   temporal: corr(next reading, last observed) = {nxt:.4f}")
    print("   -> near 0 would mean there is nothing for a spatio-temporal model to use\n")

    # ---- 6. How well can the next reading be predicted without learning? ----
    print("6. FORECAST DIFFICULTY (attack-free windows only, no model)")
    ben = te[clean[te]]
    var = float(np.mean(true[ben] ** 2))
    per = float(np.mean((true[ben] - h[ben][:, -1, :]) ** 2))
    print(f"   MSE predict-zero (variance) = {var:.4f}")
    print(f"   MSE persistence             = {per:.4f}   "
          f"({'persistence helps' if per < var else 'persistence is WORSE'})")
    print(f"   -> if persistence >> variance, the series is NOT smooth at this "
          f"sampling rate\n")


if __name__ == "__main__":
    main()
