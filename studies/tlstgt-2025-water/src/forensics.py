"""Statistical forensics on the PRINTED numbers of Table I.

Scope and limits, stated up front:
  * These tests examine internal properties of the reported numbers. They can
    show that numbers are mutually inconsistent, unattainable from any integer
    confusion matrix, or improbably patterned.
  * They CANNOT distinguish fabrication from transcription error, a different
    unreported experimental setup, results pasted from another experiment, or
    metrics computed by non-standard definitions.
  * "Impossible" is a statement about the numbers, not about intent.

Tests
  A. Attainability  — can each (F1, ACC, DR) triple arise from ANY integer
     confusion matrix on a balanced test set of any plausible size?
  B. Internal consistency — the accuracy ceiling and the joint DR+F1 -> ACC check.
  C. Monotonicity — how surprising is the observed ordering pattern?
  D. Terminal-digit uniformity — a standard number-forensics screen.

Run:  python forensics.py
"""
from __future__ import annotations
import numpy as np
from itertools import product

# Table I, exactly as printed: model -> (F1[10,20,31], ACC[...], DR[...])
T = {
    "SVM":     ([6.1, 13.2, 26.2], [80.7, 78.4, 77.7], [3.2, 9.3, 20.1]),
    "RF":      ([17.5, 22.7, 28.9], [82.4, 80.1, 76.1], [9.5, 12.4, 30.1]),
    "LGBM":    ([19.8, 20.1, 22.3], [82.5, 81.3, 81.9], [11.1, 11.3, 13.3]),
    "FFNN":    ([20.9, 24.1, 36.1], [78.6, 80.6, 83.3], [14.5, 22.7, 24.1]),
    "LSTM":    ([26.0, 27.5, 34.5], [77.0, 70.3, 64.7], [20.6, 25.9, 47.4]),
    "AE":      ([21.5, 31.4, 38.1], [67.2, 80.1, 83.4], [23.1, 24.3, 26.1]),
    "TGCN":    ([49.6, 59.3, 59.8], [79.3, 79.6, 80.4], [51.8, 53.4, 54.8]),
    "STGT":    ([52.3, 59.4, 76.3], [81.5, 83.7, 84.8], [54.2, 55.6, 74.7]),
    "TL-STGT": ([54.8, 65.1, 79.7], [82.6, 86.1, 87.1], [53.6, 59.6, 76.8]),
}
SIZES = [10, 20, 31]


def cells():
    for m, (f1, acc, dr) in T.items():
        for i, s in enumerate(SIZES):
            yield m, s, f1[i], acc[i], dr[i]


def attainable(f1_r, acc_r, dr_r, A):
    """Is (F1, ACC, DR) reproducible from integer TP,TN on a balanced set of A per class?

    Balanced test set: A attack, A benign, N = 2A.
      DR  = TP / A
      ACC = (TP + TN) / (2A)
      F1  = 2TP / (2TP + FP + FN),  FP = A - TN,  FN = A - TP
    All reported to 1 decimal, so we accept any integer pair that rounds to the
    printed triple.
    """
    tp = np.arange(A + 1)
    dr = np.round(100.0 * tp / A, 1)
    cand_tp = tp[dr == dr_r]
    if len(cand_tp) == 0:
        return False
    tn = np.arange(A + 1)
    for t in cand_tp:
        acc = np.round(100.0 * (t + tn) / (2.0 * A), 1)
        ok = tn[acc == acc_r]
        if len(ok) == 0:
            continue
        fp = A - ok
        fn = A - t
        denom = 2.0 * t + fp + fn
        f1 = np.round(np.where(denom > 0, 200.0 * t / np.maximum(denom, 1e-9), 0.0), 1)
        if np.any(f1 == f1_r):
            return True
    return False


def test_A():
    print("=" * 78)
    print("TEST A — ATTAINABILITY from an integer confusion matrix (balanced set)")
    print("=" * 78)
    print("The paper states 1,400 hours, evenly split, 80/10/10 -> a 10% test set.")
    print("That implies ~140 test samples, i.e. A = 70 per class. We also scan a")
    print("wide range of A in case the split is per-window or otherwise larger.\n")
    A_grid = [70, 100, 140, 175, 200, 250, 300, 350, 400, 500, 700]
    print(f"{'A per class':>12} | {'cells attainable':>17} | note")
    for A in A_grid:
        n_ok = sum(attainable(f1, acc, dr, A) for _, _, f1, acc, dr in cells())
        note = ""
        if A == 70:
            note = "<- the size the paper's own description implies"
        print(f"{A:>12} | {n_ok:>10}/27      | {note}")
    print("\nA cell that is unattainable for EVERY A cannot come from any confusion")
    print("matrix on a balanced set, regardless of test-set size.")
    never = []
    for m, s, f1, acc, dr in cells():
        if not any(attainable(f1, acc, dr, A) for A in A_grid):
            never.append((m, s, f1, acc, dr))
    print(f"\ncells unattainable for every A tested: {len(never)}/27")
    for m, s, f1, acc, dr in never[:12]:
        print(f"   {m:8} n={s:<3} F1={f1:5.1f} ACC={acc:5.1f} DR={dr:5.1f}")


def test_B():
    print("\n" + "=" * 78)
    print("TEST B — INTERNAL CONSISTENCY (deterministic, no assumptions beyond 50/50)")
    print("=" * 78)
    viol_ceiling, viol_joint = [], []
    for m, s, f1, acc, dr in cells():
        if acc > (dr + 100.0) / 2.0 + 1e-9:               # ACC = (DR+SP)/2, SP <= 100
            viol_ceiling.append((m, s, acc, (dr + 100) / 2))
        # joint: DR and F1 fix precision -> FP/A -> SP -> implied ACC
        p = dr / 100.0
        if f1 > 0 and p > 0:
            prec = (f1 / 100.0) * p / (2 * p - (f1 / 100.0) * p) if (2 * p - (f1/100)*p) else 0
            # from F1 = 2*prec*rec/(prec+rec)  =>  prec = F1*rec / (2*rec - F1)
            prec = (f1 / 100.0) * p / (2 * p - f1 / 100.0) if (2 * p - f1 / 100.0) > 0 else np.nan
            if np.isfinite(prec) and 0 < prec <= 1:
                fp_over_a = p * (1 - prec) / prec
                sp = 1 - fp_over_a
                implied = 100.0 * (p + sp) / 2.0
                if abs(implied - acc) > 1.0:
                    viol_joint.append((m, s, acc, implied))
    print(f"cells above the accuracy ceiling ACC <= (DR+100)/2 : {len(viol_ceiling)}/27")
    print(f"cells failing the joint DR+F1 -> ACC check (>1 pt)  : {len(viol_joint)}/27")
    print("\nworst ceiling violations (reported ACC vs maximum possible):")
    for m, s, acc, ceil in sorted(viol_ceiling, key=lambda r: r[2] - r[3], reverse=True)[:5]:
        print(f"   {m:8} n={s:<3} reported ACC={acc:5.1f}  ceiling={ceil:5.1f}  "
              f"excess=+{acc-ceil:4.1f}")


def test_C():
    print("\n" + "=" * 78)
    print("TEST C — MONOTONICITY across graph size")
    print("=" * 78)
    mono = {"F1": 0, "ACC": 0, "DR": 0}
    for m, (f1, acc, dr) in T.items():
        for key, v in (("F1", f1), ("ACC", acc), ("DR", dr)):
            if v[0] < v[1] < v[2]:
                mono[key] += 1
    n = len(T)
    print(f"strictly increasing sequences across 10->20->31 nodes:")
    for k, c in mono.items():
        print(f"   {k:4}: {c}/{n} models")
    print("\nUnder the null that size has no systematic effect, each model's three")
    print("values are equally likely in any of 3! = 6 orders, so P(increasing) = 1/6.")
    for k, c in mono.items():
        if c == n:
            print(f"   P(all {n} models increasing in {k} | no effect) = (1/6)^{n} "
                  f"= {(1/6)**n:.2e}")
    print("\nThis is NOT evidence of fabrication on its own: a genuine, strong size")
    print("effect also produces monotone columns. It becomes notable only alongside")
    print("the measured seed noise, which is large relative to several of the")
    print("reported increments (e.g. LGBM F1 19.8 -> 20.1 is +0.3).")
    small = []
    for m, (f1, acc, dr) in T.items():
        for a, b, lo, hi in ((f1[0], f1[1], 10, 20), (f1[1], f1[2], 20, 31)):
            if 0 < b - a < 1.0:
                small.append((m, lo, hi, b - a))
    print(f"\nreported F1 increments smaller than 1.0 point: {len(small)}")
    for m, lo, hi, d in small:
        print(f"   {m:8} {lo}->{hi}: +{d:.1f}")


def test_D():
    print("\n" + "=" * 78)
    print("TEST D — TERMINAL-DIGIT UNIFORMITY")
    print("=" * 78)
    digits = []
    for _, _, f1, acc, dr in cells():
        for v in (f1, acc, dr):
            digits.append(int(round(v * 10)) % 10)
    digits = np.array(digits)
    obs = np.bincount(digits, minlength=10)
    exp = len(digits) / 10.0
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    print(f"n = {len(digits)} reported values (27 cells x 3 metrics)")
    print("last digit :  " + " ".join(f"{d:>3}" for d in range(10)))
    print("count      :  " + " ".join(f"{c:>3}" for c in obs))
    print(f"\nchi-square (9 df) = {chi2:.2f}; critical value at p=0.05 is 16.92")
    print("verdict:", "NOT uniform (p<0.05)" if chi2 > 16.92 else
          "consistent with uniform — no digit anomaly detected")
    print("\nNote: with only 81 values this test has low power. A negative result")
    print("is weak evidence of nothing in particular.")


if __name__ == "__main__":
    test_A(); test_B(); test_C(); test_D()
    print("\n" + "=" * 78)
    print("WHAT THIS DOES AND DOES NOT SUPPORT")
    print("=" * 78)
    print("""Established: the printed triples are not jointly realisable under the
paper's own stated balanced 50/50 protocol and standard metric definitions.
That is deterministic arithmetic, independent of any implementation.

NOT established: how the numbers came to be. Unattainable values are equally
consistent with a transcription/rounding error, metrics computed on a
non-balanced or differently-sized set than described, values copied from a
different experiment, or non-standard metric definitions. Distinguishing those
possibilities requires information not in the paper.

The defensible claim is therefore about the ARTEFACT, not the authors: as
printed, Table I cannot be reproduced by any correct implementation, because no
confusion matrix on the described test set yields those numbers.""")
