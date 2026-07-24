"""Deterministic regression tests for detection threshold selection.

Run:  OMP_NUM_THREADS=1 python test_detect.py

Root cause these guard against (found 2026-07-24 by boundary-divergence probing):
the validation threshold was selected over samples labelled normal by their
CURRENT timestep only. But (a) a sample's 10-step history window can still
contain attack readings, and (b) the paper's batch-mean smears the trailing S
samples together, so a normal sample just after an attack block inherits the
attack's score. Both inflate the threshold, so the realized false-alarm rate
came out at 0.4% when FA_TARGET was 5.0.
"""
import numpy as np
import detect

DIM, S = 5, detect.BATCH_S


def _blocks(m, block=60):
    """Alternating normal/attack blocks, attack first (mirrors data._build_observed)."""
    lab = np.zeros(m, dtype=int)
    start, toggle = 0, 1
    while start < m:
        end = min(start + block, m)
        if toggle == 1:
            lab[start:end] = 1
        start, toggle = end, toggle ^ 1
    return lab


def _resid(lab, rng, spike=100.0):
    """Benign residuals ~N(0,I); attack residuals get a huge offset (as a raw-delta
    manipulation does to a low-variance sensor)."""
    r = rng.standard_normal((len(lab), DIM))
    r[lab == 1] += spike
    return r


def test_threshold_not_inflated_by_attack_contamination():
    rng = np.random.default_rng(0)
    train = rng.standard_normal((3000, DIM))
    lab_v = _blocks(600)
    val = _resid(lab_v, rng)

    mu, ci = detect.fit_benign(train)
    s_val = detect.batch_mean(detect.sq_mahalanobis(val, mu, ci))

    naive = np.percentile(s_val[lab_v == 0], 95.0)              # old behaviour
    clean_mask = detect.clean_batch_mask(lab_v)
    fixed = np.percentile(s_val[clean_mask], 95.0)              # new behaviour

    # A clean chi2(5) batch-mean sits near 5; the naive threshold is inflated by
    # post-attack samples whose trailing batch still contains the spike.
    assert fixed < 50, f"clean threshold should stay near chi2({DIM}); got {fixed:.4g}"
    assert naive > 10 * fixed, (
        f"expected naive threshold to be inflated; naive={naive:.4g} fixed={fixed:.4g}")
    print(f"  ok: naive={naive:.4g} inflated vs clean={fixed:.4g} "
          f"({naive/fixed:.1f}x)")


def test_realized_false_alarm_matches_target():
    """The whole point of a fixed-FA operating point: it must actually hit it.

    Threshold is learned from a validation series that CONTAINS attacks (so the
    clean-batch mask has to do its job), then applied to a purely normal series.
    Note `batch_mean` makes scores heavily autocorrelated, so the series must be
    long enough to hold many independent clean stretches for the quantile to be
    estimable at all.
    """
    rng = np.random.default_rng(1)
    train = rng.standard_normal((5000, DIM))
    lab_v = _blocks(12000)                      # ~100 attack/normal blocks
    val = _resid(lab_v, rng)
    lab_t = np.zeros(20000, dtype=int)          # purely normal reference series
    test = _resid(lab_t, rng)

    mu, ci = detect.fit_benign(train)
    s_v = detect.batch_mean(detect.sq_mahalanobis(val, mu, ci))
    s_t = detect.batch_mean(detect.sq_mahalanobis(test, mu, ci))
    for target in (1.0, 5.0, 10.0):
        thr = np.percentile(s_v[detect.clean_batch_mask(lab_v)], 100.0 - target)
        realized = 100.0 * (s_t > thr).mean()
        assert abs(realized - target) <= 3.0, (
            f"target FA {target}% but realized {realized:.2f}% on normal data")
        print(f"  ok: target FA {target:4.1f}% -> realized {realized:5.2f}% on normal data")

    # and the naive selection must MISS the target badly (the bug being fixed)
    thr_naive = np.percentile(s_v[lab_v == 0], 95.0)
    naive_realized = 100.0 * (s_t > thr_naive).mean()
    assert naive_realized < 1.0, (
        f"expected naive threshold to suppress detection; realized {naive_realized:.2f}%")
    print(f"  ok: naive selection realizes {naive_realized:.2f}% FA against a 5% target")


def test_clean_batch_mask_semantics():
    lab = np.zeros(80, dtype=int)
    lab[40:50] = 1
    m = detect.clean_batch_mask(lab, s=8)
    assert not m[40:50].any(), "attack samples are never clean"
    assert not m[50:57].any(), "samples whose trailing batch overlaps the attack are not clean"
    assert m[58:].all(), "once the batch clears the attack, samples are clean again"
    assert m[8:40].all(), "pre-attack samples with a full normal batch are clean"
    win = np.ones(80, bool); win[20] = False
    m2 = detect.clean_batch_mask(lab, win_clean=win, s=8)
    assert not m2[20:28].any(), "a contaminated history window disqualifies its batch"
    print("  ok: mask excludes attacks, trailing overlap, and dirty windows")


if __name__ == "__main__":
    n = 0
    for fn in (test_clean_batch_mask_semantics,
               test_threshold_not_inflated_by_attack_contamination,
               test_realized_false_alarm_matches_target):
        print(f"{fn.__name__}:")
        fn(); n += 1
    print(f"\n{n} test functions passed")
