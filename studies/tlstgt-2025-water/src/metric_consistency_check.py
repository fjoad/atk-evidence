#!/usr/bin/env python3
"""Internal-consistency audit of Table I in:

  M. R. Ahasan, F. Joad, R. Atat, C. Thompson, E. Serpedin, A. Takiddin,
  "Graph Transfer Learning-Based Attack Detection in Cyber-Physical Water
  Distribution Systems," EUSIPCO 2025, pp. 1912-1916.

The paper states the test set has "equal number of samples per class"
(balanced, 50/50) and defines, standardly:
    DR (recall)   = TP / (TP + FN)
    ACC           = (TP + TN) / (TP + TN + FP + FN)
    F1            = 2*TP / (2*TP + FP + FN)

On a balanced test set (P = N), accuracy reduces to the mean of recall and
specificity:   ACC = (DR + SP)/2,   SP = TN/N <= 1.
Therefore the reported accuracy has a HARD CEILING that depends only on DR:
    ACC <= (DR + 100) / 2.
Any row with ACC above that ceiling cannot come from the stated experiment,
for ANY classifier and ANY threshold.

A tighter check uses DR and F1 together. On a balanced set, (DR, F1) determine
precision, hence the false-alarm rate FA, hence ACC. If that implied ACC differs
from the reported ACC (or the implied FA falls outside [0,100]%), the triple is
mutually inconsistent.

Finally, we drop the balance assumption and solve for the attack prevalence pi
that WOULD reconcile each (DR, F1, ACC) triple, to see whether a single hidden
test-set composition could explain the numbers. It cannot: the rows imply
different prevalences.

Run:  python3 metric_consistency_check.py
Requires only numpy.
"""
import numpy as np

# Table I, DETECTION PERFORMANCE ON CYBERATTACKS (%). Columns per graph size 10/20/31.
TABLE = {
    #                 F1 (10,20,31)          ACC (10,20,31)          DR (10,20,31)
    "SVM":      ([6.1, 13.2, 26.2], [80.7, 78.4, 77.7], [3.2, 9.3, 20.1]),
    "RF":       ([17.5, 22.7, 28.9], [82.4, 80.1, 76.1], [9.5, 12.4, 30.1]),
    "LGBM":     ([19.8, 20.1, 22.3], [82.5, 81.3, 81.9], [11.1, 11.3, 13.3]),
    "FFNN":     ([20.9, 24.1, 36.1], [78.6, 80.6, 83.3], [14.5, 22.7, 24.1]),
    "LSTM":     ([26.0, 27.5, 34.5], [77.0, 70.3, 64.7], [20.6, 25.9, 47.4]),
    "AE":       ([21.5, 31.4, 38.1], [67.2, 80.1, 83.4], [23.1, 24.3, 26.1]),
    "TGCN":     ([49.6, 59.3, 59.8], [79.3, 79.6, 80.4], [51.8, 53.4, 54.8]),
    "STGT":     ([52.3, 59.4, 76.3], [81.5, 83.7, 84.8], [54.2, 55.6, 74.7]),
    "TL-STGT":  ([54.8, 65.1, 79.7], [82.6, 86.1, 87.1], [53.6, 59.6, 76.8]),
}
SIZES = [10, 20, 31]


def implied_fa_balanced(dr, f1):
    """FA (%) implied by DR and F1 on a balanced test set, or None if degenerate."""
    dr, f1 = dr / 100.0, f1 / 100.0
    denom = 2 * dr - f1
    if denom <= 0:
        return None
    pr = f1 * dr / denom            # precision
    if pr <= 0:
        return None
    fa = dr * (1 - pr) / pr         # since PR = DR/(DR+FA) on a balanced set
    return fa * 100.0


def solve_prevalence(dr, f1, acc):
    """Prevalence pi and FA that reconcile (DR,F1,ACC) with NO balance assumption.

    F1  = 2*DR*pi / ((1+DR)*pi + FA*(1-pi))
    ACC = DR*pi + (1-FA)*(1-pi)
    Returns (pi_pct, fa_pct) or None if no feasible root in (0,1)x[0,1].
    """
    dr, f1, acc = dr / 100.0, f1 / 100.0, acc / 100.0

    def fa_of_pi(pi):
        num = pi * (2 * dr - f1 * (1 + dr))
        den = f1 * (1 - pi)
        return num / den if den != 0 else np.inf

    def g(pi):
        fa = fa_of_pi(pi)
        return dr * pi + (1 - fa) * (1 - pi) - acc

    grid = np.linspace(1e-4, 1 - 1e-4, 20000)
    vals = np.array([g(pi) for pi in grid])
    roots = []
    for i in range(len(grid) - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0:
            a, b = grid[i], grid[i + 1]
            for _ in range(60):
                m = (a + b) / 2
                a, b = (m, b) if g(a) * g(m) > 0 else (a, m)
            pi = (a + b) / 2
            fa = fa_of_pi(pi)
            if 0 <= fa <= 1:
                roots.append((pi * 100, fa * 100))
    return roots[0] if roots else None


def main():
    n_cells = 0
    n_ceiling_fail = 0
    n_triple_fail = 0
    print(f"{'model':9} {'size':>4} {'DR':>5} {'ACC':>5} {'F1':>5} "
          f"{'ceil=(DR+100)/2':>15} {'ACC>ceil?':>10} "
          f"{'FA_impl%':>9} {'ACC_impl':>9} {'prevalence%':>11}")
    for model, (f1s, accs, drs) in TABLE.items():
        for k, size in enumerate(SIZES):
            n_cells += 1
            f1, acc, dr = f1s[k], accs[k], drs[k]
            ceil = (dr + 100) / 2
            ceiling_fail = acc > ceil + 1e-9
            n_ceiling_fail += ceiling_fail
            fa_impl = implied_fa_balanced(dr, f1)
            acc_impl = None if fa_impl is None else (dr + 100 - fa_impl) / 2
            # triple inconsistent if implied FA infeasible, or implied ACC far from reported
            triple_fail = (fa_impl is None or fa_impl < 0 or fa_impl > 100
                           or (acc_impl is not None and abs(acc_impl - acc) > 2.0))
            n_triple_fail += triple_fail
            prev = solve_prevalence(dr, f1, acc)
            prev_s = f"{prev[0]:.1f}" if prev else "none"
            fa_s = "n/a" if fa_impl is None else f"{fa_impl:.1f}"
            acc_impl_s = "n/a" if acc_impl is None else f"{acc_impl:.1f}"
            print(f"{model:9} {size:>4} {dr:>5} {acc:>5} {f1:>5} "
                  f"{ceil:>15.1f} {'YES' if ceiling_fail else 'no':>10} "
                  f"{fa_s:>9} {acc_impl_s:>9} {prev_s:>11}")
    print()
    print(f"cells total                              : {n_cells}")
    print(f"cells violating ACC<=(DR+100)/2 ceiling  : {n_ceiling_fail}/{n_cells}")
    print(f"cells inconsistent under DR+F1+ACC check : {n_triple_fail}/{n_cells}")
    print()
    print("Interpretation: on the paper's own balanced test set and standard metric")
    print("definitions, an accuracy above (DR+100)/2 is arithmetically impossible.")
    print("The DR+F1+ACC check is stricter (uses all three reported numbers).")
    print("The prevalence column shows the SINGLE test-set attack fraction that would")
    print("reconcile each row if the set were NOT balanced -- these disagree row to")
    print("row and contradict the stated 50/50 split.")


if __name__ == "__main__":
    main()
