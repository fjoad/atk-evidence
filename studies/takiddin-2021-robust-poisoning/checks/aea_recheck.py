#!/usr/bin/env python3
"""Independent arithmetic audit of the two preserved AEA diagnostics only."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HASHES = {"geometry": "d6e8b877ce31bd6ccaccbfa9ed478fad9587665782a9192590243f58834a42e5",
          "repair": "fd14a718560989e5575e7e1a8f4f5f153a17fc6fd023ccfdde3ee35ade9c7f69"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scalar_bounds(target, low, high, absolute=False):
    """Coordinatewise endpoint cases, independent of the original clip formula."""
    x = target.astype(np.float64)
    nearest_error = np.where(x < low, low - x, np.where(x > high, x - high, 0.))
    farthest_error = np.where(x <= (low + high) / 2, high - x, x - low)
    if not absolute:
        nearest_error, farthest_error = nearest_error ** 2, farthest_error ** 2
    return nearest_error.mean(axis=1), farthest_error.mean(axis=1)


def pair_auc(benign, malicious):
    """Direct pair counting, not sklearn ROC integration."""
    wins, ties = 0, 0
    for start in range(0, len(malicious), 256):
        m = malicious[start:start + 256, None]
        wins += int(np.count_nonzero(m > benign[None, :]))
        ties += int(np.count_nonzero(m == benign[None, :]))
    return 100 * (wins + .5 * ties) / (len(benign) * len(malicious))


def corner(benign_lower, malicious_upper, cap):
    allowed = math.floor(cap * len(benign_lower) / 100 + 1e-12)
    index = len(benign_lower) - allowed - 1
    threshold = float(np.partition(benign_lower, index)[index])
    return 100 * float(np.mean(malicious_upper > threshold)), threshold


def case_summary(analysis):
    """Keep condition-specific FA caps separate from the printed cutoff."""
    group, target = analysis["groups"]["all"], analysis["paper_context"]
    cap, dr_target = target["FA"] + .05, target["DR"] - .05
    min_fa = group["fixed_cutoff"]["0.515"]["1e-06"]["minimum_FA"]
    max_dr = group["fixed_cutoff"]["0.505"]["1e-06"]["maximum_DR"]
    return {
        "favorable_FA_cap": cap, "favorable_DR_target": dr_target,
        "all_cutoff_DR_upper_bound": group["favorable_envelope"]["at_FA_caps"][str(cap)]["DR_upper_bound"],
        "fixed_cutoff_favorable_minimum_FA": min_fa,
        "fixed_cutoff_favorable_maximum_DR": max_dr,
        "fixed_cutoff_excluded_by_FA": min_fa > cap + 1e-10,
        "fixed_cutoff_excluded_by_DR": max_dr + 1e-10 < dr_target,
        "free_cutoff_interval": group["favorable_threshold_interval"]}


def recheck(geometry, repair):
    for name, directory in (("geometry", geometry), ("repair", repair)):
        if digest(directory / "result.json") != HASHES[name]:
            raise ValueError(f"Unexpected original {name} result")
    old = json.loads((geometry / "result.json").read_text())
    result = json.loads((repair / "result.json").read_text())
    comparisons = {}
    for level, branches in result["cases"].items():
        old_path = geometry / f"{level}_scores.npz"
        assert digest(old_path) == old["cases"][level]["scores_sha256"]
        with np.load(old_path, allow_pickle=False) as f:
            old_scores = dict(f)
        comparisons[level] = {}
        for branch, record in branches.items():
            path = repair / f"{level}_{branch}.npz"
            assert digest(path) == record["archive_sha256"]
            with np.load(path, allow_pickle=False) as f:
                s = dict(f)
            lower, upper = scalar_bounds(s["test_target"], s["box_low"], s["box_high"], branch == "feature_z_mae")
            np.testing.assert_allclose(lower, s["lower"], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(upper, s["upper"], rtol=1e-12, atol=1e-12)
            if branch == "feature_z_mse":
                np.testing.assert_array_equal(lower, old_scores["test_lower_mse"])
                np.testing.assert_array_equal(upper, old_scores["test_upper_mse"])
            a = record["analysis"]
            for group, mask in (("all", np.ones(len(lower), bool)), ("original", ~s["synthetic"]), ("synthetic", s["synthetic"])):
                y, lo, hi = s["labels"][mask], lower[mask], upper[mask]
                g = a["groups"][group]
                for threshold in (.505, .51, .515):
                    for guard in (0., 1e-6):
                        b = g["fixed_cutoff"][str(threshold)][str(guard)]
                        fa = 100 * float(np.mean(np.maximum(lo[y == 0] - guard, 0) > threshold))
                        assert abs(fa - b["minimum_FA"]) < 1e-10
                        if np.any(y == 1):
                            dr = 100 * float(np.mean(hi[y == 1] + guard > threshold))
                            assert abs(dr - b["maximum_DR"]) < 1e-10
                if set(np.unique(y)) != {0, 1}:
                    assert g["favorable_envelope"] is None
                    continue
                for guard, key in ((0., "nominal_envelope"), (1e-6, "favorable_envelope")):
                    l, u = np.maximum(lo - guard, 0), hi + guard
                    e = g[key]
                    assert abs(pair_auc(l[y == 0], u[y == 1]) - e["AUC_upper_bound"]) < 1e-10
                    for cap, bound in e["at_FA_caps"].items():
                        bound_dr, _ = corner(l[y == 0], u[y == 1], float(cap))
                        assert abs(bound_dr - bound["DR_upper_bound"]) < 1e-10
                for fa, dr, guard, key in ((a["paper_context"]["FA"], a["paper_context"]["DR"], 0., "nominal_threshold_interval"),
                        (a["paper_context"]["FA"] + .05, a["paper_context"]["DR"] - .05, 1e-6, "favorable_threshold_interval")):
                    l, u = np.maximum(lo - guard, 0), hi + guard
                    _, left = corner(l[y == 0], u[y == 1], fa)
                    required = math.ceil(dr * int((y == 1).sum()) / 100 - 1e-12)
                    right = float(np.partition(u[y == 1], int((y == 1).sum()) - required)[-required])
                    np.testing.assert_allclose([left, right], [g[key]["minimum_inclusive"], g[key]["maximum_exclusive"]], rtol=1e-12, atol=1e-12)
                if branch == "feature_z_mse":
                    l, u = np.maximum(lo - 1e-6, 0), hi + 1e-6
                    e = g["reverse_favorable_envelope"]
                    assert abs(pair_auc(-u[y == 0], -l[y == 1]) - e["AUC_upper_bound"]) < 1e-10
                    for cap, bound in e["at_FA_caps"].items():
                        dr_upper, _ = corner(-u[y == 0], -l[y == 1], float(cap))
                        assert abs(dr_upper - bound["DR_upper_bound"]) < 1e-10
            comparisons[level][branch] = case_summary(a)
    return {"status": "independent_arithmetic_verified", "original_result_hashes": HASHES,
            "comparisons": comparisons, "new_research_experiments": 0,
            "method": "piecewise endpoint errors; direct AUC pair counts; benign order-statistic ROC bounds",
            "audit_source_sha256": digest(Path(__file__))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(recheck(args.geometry, args.repair), indent=2, sort_keys=True))
