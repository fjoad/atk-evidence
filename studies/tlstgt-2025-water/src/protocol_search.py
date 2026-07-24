"""Exhaustive search for ANY measurement protocol that reproduces the reported table.

WHY THIS EXISTS
---------------
Showing that reported numbers are unattainable under the protocol a paper
*describes* is escapable: a defender can say the description was wrong. Showing
they are unattainable under EVERY plausible protocol is not escapable.

Formally, with
    H0 : the reported triple arose from a real confusion matrix under some
         protocol (any test-set size, any class prevalence, any of the common
         metric definitions),
this script evaluates P(observed table | H0). If no combination in the searched
space reproduces a cell, that likelihood is exactly 0 for the cell -- a
categorical exclusion rather than a small p-value.

The space searched is declared in full below so that any reader can extend it.
A negative result is only as strong as the breadth of this space, and the space
is pre-registered here rather than tuned after seeing outcomes.

DECISIVE CONSTRAINT
-------------------
A paper gets one protocol, not twenty-seven. Even if individual cells are
attainable under exotic settings, the table is only defensible if a SINGLE
(size, prevalence, metric-definition) setting explains every cell. Both results
are reported: per-cell attainability, and joint attainability.

WHAT A NEGATIVE RESULT DOES AND DOES NOT SHOW
---------------------------------------------
It shows the values were not produced by any measurement process in the searched
space. It does NOT distinguish deliberate fabrication from a catastrophic
process failure (for example placeholder values never replaced, or a table
carried over from a draft). Both yield numbers that were never measured.
Separating them requires evidence outside the artefact.

Run:  python protocol_search.py            # summary
      python protocol_search.py --full     # per-cell detail
"""
from __future__ import annotations
import argparse
import numpy as np

# Table I exactly as printed: model -> (F1[10,20,31], ACC[...], DR[...])
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
TOL = 0.05        # reported to 1 d.p., so anything rounding to the printed value


def cells():
    for m, (f1, acc, dr) in T.items():
        for i, s in enumerate(SIZES):
            yield m, s, f1[i], acc[i], dr[i]


# ---------------------------------------------------------------- definitions
# Each variant maps a confusion matrix (tp, fp, tn, fn) to a (F1, ACC, DR)
# triple in percent. These cover the paper's stated definitions plus the
# implementation slips that most commonly produce inflated accuracy.
def _f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return np.where(d > 0, 200.0 * tp / np.maximum(d, 1e-12), 0.0)


DEFS = {
    # as the paper defines them
    "standard": lambda tp, fp, tn, fn: (
        _f1(tp, fp, fn),
        100.0 * (tp + tn) / np.maximum(tp + fp + tn + fn, 1e-12),
        100.0 * tp / np.maximum(tp + fn, 1e-12)),
    # accuracy reported as BALANCED accuracy while DR/F1 stay standard
    "balanced_acc": lambda tp, fp, tn, fn: (
        _f1(tp, fp, fn),
        100.0 * 0.5 * (tp / np.maximum(tp + fn, 1e-12) + tn / np.maximum(tn + fp, 1e-12)),
        100.0 * tp / np.maximum(tp + fn, 1e-12)),
    # "DR" column actually holding precision
    "dr_is_precision": lambda tp, fp, tn, fn: (
        _f1(tp, fp, fn),
        100.0 * (tp + tn) / np.maximum(tp + fp + tn + fn, 1e-12),
        100.0 * tp / np.maximum(tp + fp, 1e-12)),
    # F1 computed over the NEGATIVE class (a common macro/label slip)
    "f1_negative": lambda tp, fp, tn, fn: (
        _f1(tn, fn, fp),
        100.0 * (tp + tn) / np.maximum(tp + fp + tn + fn, 1e-12),
        100.0 * tp / np.maximum(tp + fn, 1e-12)),
    # macro-averaged F1 across both classes
    "f1_macro": lambda tp, fp, tn, fn: (
        0.5 * (_f1(tp, fp, fn) + _f1(tn, fn, fp)),
        100.0 * (tp + tn) / np.maximum(tp + fp + tn + fn, 1e-12),
        100.0 * tp / np.maximum(tp + fn, 1e-12)),
    # specificity reported in the DR column
    "dr_is_specificity": lambda tp, fp, tn, fn: (
        _f1(tp, fp, fn),
        100.0 * (tp + tn) / np.maximum(tp + fp + tn + fn, 1e-12),
        100.0 * tn / np.maximum(tn + fp, 1e-12)),
}


def solutions(f1_r, acc_r, dr_r, n_pos, n_neg, defname):
    """Confusion matrices on (n_pos, n_neg) whose metrics round to the printed triple."""
    fn_def = DEFS[defname]
    tp = np.arange(n_pos + 1)[:, None]
    tn = np.arange(n_neg + 1)[None, :]
    fp = n_neg - tn
    fn = n_pos - tp
    f1, acc, dr = fn_def(tp, fp, tn, fn)
    ok = (np.abs(f1 - f1_r) <= TOL) & (np.abs(acc - acc_r) <= TOL) & (np.abs(dr - dr_r) <= TOL)
    return ok.any(), int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--max-n", type=int, default=1200)
    args = ap.parse_args()

    # Pre-registered search space.
    totals = sorted({*range(20, 301, 2), *range(304, 1001, 8),
                     *[n for n in range(1000, args.max_n + 1, 25)]})
    prevalences = np.round(np.arange(0.02, 0.99, 0.02), 2)
    print("=" * 78)
    print("EXHAUSTIVE PROTOCOL SEARCH")
    print("=" * 78)
    print(f"test-set sizes N     : {len(totals)} values from {min(totals)} to {max(totals)}")
    print(f"class prevalences    : {len(prevalences)} values from {prevalences[0]:.2f} to {prevalences[-1]:.2f}")
    print(f"metric definitions   : {len(DEFS)} ({', '.join(DEFS)})")
    print(f"rounding tolerance   : +/-{TOL}")
    total_combos = len(totals) * len(prevalences) * len(DEFS)
    print(f"protocols per cell   : {total_combos:,}\n")

    # Evaluate each protocol ONCE and test all 27 cells against the achievable
    # metric surface. Looping cells on the outside would rebuild the same grid
    # 27 times over.
    all_cells = list(cells())
    per_cell = {(m, s): [] for m, s, *_ in all_cells}
    protocol_hits = {}
    seen = set()
    for defname, fn_def in DEFS.items():
        for N in totals:
            for pi in prevalences:
                n_pos = int(round(N * pi))
                n_neg = N - n_pos
                if n_pos < 1 or n_neg < 1 or (defname, n_pos, n_neg) in seen:
                    continue
                seen.add((defname, n_pos, n_neg))
                tp = np.arange(n_pos + 1)[:, None]
                tn = np.arange(n_neg + 1)[None, :]
                f1, acc, dr = fn_def(tp, n_neg - tn, tn, n_pos - tp)
                key = (defname, N, float(pi))
                for m, s, f1_r, acc_r, dr_r in all_cells:
                    if (np.abs(f1 - f1_r) <= TOL).any():
                        ok = ((np.abs(f1 - f1_r) <= TOL) & (np.abs(acc - acc_r) <= TOL)
                              & (np.abs(dr - dr_r) <= TOL))
                        if ok.any():
                            per_cell[(m, s)].append(key)
                            protocol_hits.setdefault(key, set()).add((m, s))
    if args.full:
        for (m, s), found in per_cell.items():
            ex = f"  e.g. {found[0]}" if found else ""
            print(f"  {m:8} n={s:<3} protocols that reproduce it: {len(found):>6}{ex}")

    n_cells = len(per_cell)
    unattainable = [k for k, v in per_cell.items() if not v]
    print(f"\n--- PER-CELL ---")
    print(f"cells reproducible by NO protocol in the space : {len(unattainable)}/{n_cells}")
    if unattainable and len(unattainable) < n_cells:
        for m, s in unattainable[:8]:
            print(f"    {m} n={s}")

    print(f"\n--- JOINT (one protocol must explain the whole table) ---")
    if protocol_hits:
        best = max(protocol_hits.items(), key=lambda kv: len(kv[1]))
        print(f"best single protocol explains {len(best[1])}/{n_cells} cells: {best[0]}")
        full = [k for k, v in protocol_hits.items() if len(v) == n_cells]
        print(f"protocols explaining ALL {n_cells} cells: {len(full)}")
    else:
        print(f"no protocol in the space reproduces even one cell")
        print(f"best single protocol explains 0/{n_cells} cells")

    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    if len(unattainable) == n_cells:
        print("""P(observed table | the numbers were measured) = 0 for every protocol in the
searched space. This is categorical exclusion, not a small p-value: no test-set
size, no class prevalence, and none of the common metric definitions yields the
printed values.

The values were therefore not produced by any measurement process in this space.
That does NOT separate deliberate fabrication from catastrophic process failure
(placeholder values never replaced, a table carried from a draft); both produce
numbers that were never measured. Separating them needs evidence outside the
paper.""")
    else:
        print(f"""{n_cells - len(unattainable)} of {n_cells} cells are reproducible under SOME protocol, so
per-cell impossibility does not hold across the board. The decisive question is
then the joint one: whether a single protocol explains the entire table. A paper
gets one protocol, not one per row.""")


if __name__ == "__main__":
    main()
