#!/usr/bin/env python3
"""Finite zero-fit repair envelope; label-informed bounds are not detectors."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np
from sklearn.metrics import roc_auc_score

import aea_diagnostic as base

STUDY = base.STUDY
PREVIOUS_HASH = "d6e8b877ce31bd6ccaccbfa9ed478fad9587665782a9192590243f58834a42e5"
BRANCHES = ("feature_z_mse", "feature_z_mae", "raw_unit_mse", "global_z_mse", "minmax_mse")
CAPS = (5.2, 5.25, 18.4, 18.45)
EPS = 1e-10


def error_intervals(x, low=0., high=1., metric="mse"):
    x, low, high = map(lambda v: np.asarray(v, dtype=np.float64), (x, low, high))
    if (x.ndim != 2 or not all(x.shape) or not all(np.isfinite(v).all() for v in (x, low, high))
            or np.any(low > high) or metric not in ("mse", "mae")):
        raise ValueError("Invalid target/output box/error convention")
    nearest = np.abs(x - np.clip(x, low, high))
    farthest = np.maximum(np.abs(x - low), np.abs(x - high))
    power = 2 if metric == "mse" else 1
    with np.errstate(over="raise", invalid="raise"):
        return np.mean(nearest ** power, axis=1), np.mean(farthest ** power, axis=1)


def check_intervals(lower, upper, labels):
    if (lower.ndim != 1 or lower.shape != upper.shape or lower.shape != labels.shape
            or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or np.any(lower < 0) or np.any(lower > upper) or set(np.unique(labels)) != {0, 1}):
        raise ValueError("Need finite intervals and both classes")


def optimistic_envelope(lower, upper, labels, reverse=False):
    check_intervals(lower, upper, labels)
    oracle = np.where(labels == 1, -lower if reverse else upper, -upper if reverse else lower)
    benign, malicious = np.sort(oracle[labels == 0]), np.sort(oracle[labels == 1])
    thresholds = np.r_[-np.inf, np.unique(oracle)]
    fa = 100 * (len(benign) - np.searchsorted(benign, thresholds, side="right")) / len(benign)
    dr = 100 * (len(malicious) - np.searchsorted(malicious, thresholds, side="right")) / len(malicious)
    corners = {}
    for cap in CAPS:
        allowed = np.flatnonzero(fa <= cap + 1e-12)
        i = int(allowed[np.argmax(dr[allowed])])
        corners[str(cap)] = {"DR_upper_bound": float(dr[i]), "oracle_FA": float(fa[i]),
                             "oracle_threshold": float(thresholds[i]) if np.isfinite(thresholds[i]) else None}
    return oracle, {"AUC_upper_bound": 100 * float(roc_auc_score(labels, oracle)), "at_FA_caps": corners,
                    "threshold_rule": "score > threshold; ties move together; null is below all scores",
                    "uses_true_test_labels": True, "is_a_trained_or_deployable_model": False}


def necessary_threshold_interval(lower, upper, labels, fa_cap, dr_target):
    check_intervals(lower, upper, labels)
    if not 0 <= fa_cap < 100 or not 0 < dr_target <= 100:
        raise ValueError("Need FA<100 and positive DR target")
    benign, malicious = np.sort(lower[labels == 0])[::-1], np.sort(upper[labels == 1])[::-1]
    allowed_fp = math.floor(fa_cap * len(benign) / 100 + 1e-12)
    required_tp = math.ceil(dr_target * len(malicious) / 100 - 1e-12)
    left, right = float(benign[allowed_fp]), float(malicious[required_tp - 1])
    return {"minimum_inclusive": left, "maximum_exclusive": right, "empty": left >= right,
            "allowed_false_positives": allowed_fp, "required_true_positives": required_tp}


def scale_interval(interval):
    left, right = interval["minimum_inclusive"], interval["maximum_exclusive"]
    if interval["empty"] or right <= 0:
        return {"empty": True}
    return {"empty": False, "positive_multiplier_minimum_exclusive": .51 / right,
            "positive_multiplier_maximum_inclusive": .51 / left if left else None,
            "maximum_unbounded": left == 0, "formula": "scaled_score = multiplier * original_MSE"}


def view(a, branch):
    raw_train, raw_test = a["train_raw_x"].astype(np.float64), a["test_raw_x"].astype(np.float64)
    low, high = np.asarray(0.), np.asarray(1.)
    location, scale = a["scaler_mean"], a["scaler_scale"]
    zero_scale = np.zeros_like(scale, dtype=bool)
    if branch in ("feature_z_mse", "feature_z_mae"):
        train, test = a["train_x"], a["test_x"]
    elif branch == "raw_unit_mse":
        train, test = a["train_raw_x"], a["test_raw_x"]
        low, high = location, location + scale
    elif branch in ("global_z_mse", "minmax_mse"):
        if branch == "global_z_mse":
            location, original_scale = np.asarray(raw_train.mean()), np.asarray(raw_train.std())
        else:
            location = raw_train.min(axis=0)
            original_scale = raw_train.max(axis=0) - location
        zero_scale = original_scale == 0
        scale = np.where(zero_scale, 1., original_scale)
        train, test = [((x - location) / scale).astype(np.float32) for x in (raw_train, raw_test)]
    else:
        raise ValueError("Undeclared repair branch")
    lower, upper = error_intervals(test, low, high, "mae" if branch == "feature_z_mae" else "mse")
    y = a["test_true_y"]
    saved = {"train_target": train, "test_target": test, "box_low": low, "box_high": high,
            "location": location, "scale": scale, "zero_scale": zero_scale,
            "lower": lower, "upper": upper, "labels": y,
            "synthetic": a["test_synthetic"], "uid": a["test_uid"]}
    saved["label_oracle_high"] = np.where(y == 1, upper, lower)
    saved["label_oracle_high_guarded"] = np.where(y == 1, upper + 1e-6, np.maximum(lower - 1e-6, 0))
    if branch == "feature_z_mse":
        saved["label_oracle_reverse"] = np.where(y == 1, -lower, -upper)
        saved["label_oracle_reverse_guarded"] = np.where(y == 1, -np.maximum(lower - 1e-6, 0), -upper - 1e-6)
    return saved


def reverse_fixed(lower, upper, y, tau, guard):
    lo, hi = np.maximum(lower - guard, 0), upper + guard
    return {"threshold": tau, "guard": guard, "rule": "error < threshold",
            "minimum_FA": 100 * float(np.mean(hi[y == 0] < tau)),
            "maximum_DR": 100 * float(np.mean(lo[y == 1] < tau))}


def analyze(saved, branch, poison_rate):
    target = base.reported_row(poison_rate, "aea")
    y, synth = saved["labels"], saved["synthetic"]
    output = {"branch": branch, "track": "C/A" if branch == "minmax_mse" else "I/A",
              "paper_context": target, "is_reproduction": False, "groups": {},
              "test_coordinates_outside_box": int(((saved["test_target"] < saved["box_low"]) |
                                                   (saved["test_target"] > saved["box_high"])).sum()),
              "zero_scale_count": int(saved["zero_scale"].sum())}
    for group, chosen in (("all", np.ones(len(y), bool)), ("original", ~synth), ("synthetic", synth)):
        if not chosen.any():
            continue
        labels, lower, upper = y[chosen], saved["lower"][chosen], saved["upper"][chosen]
        block = {"n": len(labels), "fixed_cutoff": {str(t): {str(g): base.bound_counts(lower, upper, labels, t, g)
                    for g in (0., 1e-6)} for t in (.505, .51, .515)}}
        if set(np.unique(labels)) == {0, 1}:
            _, block["nominal_envelope"] = optimistic_envelope(lower, upper, labels)
            lo, hi = np.maximum(lower - 1e-6, 0), upper + 1e-6
            _, block["favorable_envelope"] = optimistic_envelope(lo, hi, labels)
            block["nominal_threshold_interval"] = necessary_threshold_interval(lower, upper, labels, target["FA"], target["DR"])
            block["favorable_threshold_interval"] = necessary_threshold_interval(lo, hi, labels, target["FA"] + .05, target["DR"] - .05)
            if branch == "feature_z_mse":
                _, block["reverse_favorable_envelope"] = optimistic_envelope(lo, hi, labels, reverse=True)
                block["reverse_fixed_cutoff"] = [reverse_fixed(lower, upper, labels, t, 1e-6) for t in (.505, .51, .515)]
            if group == "all":
                best = block["favorable_envelope"]["at_FA_caps"][str(target["FA"] + .05)]["DR_upper_bound"]
                block["corner_excluded_even_after_rounding_guard"] = best + EPS < target["DR"] - .05
                block["AUC_excluded_even_after_rounding_guard"] = block["favorable_envelope"]["AUC_upper_bound"] + EPS < target["AUC"] - .05
                if branch == "feature_z_mse":
                    block["positive_score_multiplier_necessary_interval"] = scale_interval(block["favorable_threshold_interval"])
        else:
            block["nominal_envelope"] = block["favorable_envelope"] = None
        output["groups"][group] = block
    return output


def common_intervals(cases):
    result = {}
    for branch in BRANCHES:
        intervals = [cases[level][branch]["analysis"]["groups"]["all"]["favorable_threshold_interval"] for level in ("p00", "p30")]
        left, right = max(i["minimum_inclusive"] for i in intervals), min(i["maximum_exclusive"] for i in intervals)
        item = {"minimum_inclusive": left, "maximum_exclusive": right, "empty": left >= right,
                "relaxation_only_not_a_selected_threshold": True}
        if branch == "feature_z_mse":
            item["positive_score_multiplier_necessary_interval"] = scale_interval(item)
        result[branch] = item
    return result


def verify_prior(preparation, previous):
    if base.digest(previous / "result.json") != PREVIOUS_HASH:
        raise ValueError("Prior geometry result identity changed")
    return base.audit(preparation, previous)


def audit(preparation, previous, output):
    r = json.loads((output / "result.json").read_text())
    if r["status"] != "complete":
        raise ValueError("Only completed repair diagnostics can be audited")
    assert r["model_fits"] == r["neural_inference_calls"] == 0
    assert set(r["cases"]) == {"p00", "p30"}
    assert all(set(c) == set(BRANCHES) for c in r["cases"].values())
    assert r["linear_output_analytic_control"] == {"minimum_error": 0., "finite_error_upper_bound": False,
            "range_only_argument_inconclusive": True, "does_not_establish_neural_attainability": True}
    for name, expected in r["source_sha256"].items():
        base.verify_source(name, expected, r["code_commit"])
    assert verify_prior(preparation, previous) == r["previous_audit"]
    cases, inputs = base.load_pair(preparation)
    assert inputs == r["inputs"]
    for level, a in cases.items():
        for branch in BRANCHES:
            path = output / f"{level}_{branch}.npz"
            assert base.digest(path) == r["cases"][level][branch]["archive_sha256"]
            with np.load(path, allow_pickle=False) as f:
                saved = dict(f)
            expected = view(a, branch)
            assert set(saved) == set(expected)
            for key in expected:
                assert saved[key].dtype == expected[key].dtype
                if saved[key].dtype.kind == "f":
                    np.testing.assert_allclose(saved[key], expected[key], rtol=1e-12, atol=1e-12)
                else:
                    np.testing.assert_array_equal(saved[key], expected[key])
            assert analyze(saved, branch, 0. if level == "p00" else .3) == r["cases"][level][branch]["analysis"]
    assert common_intervals(r["cases"]) == r["common_cutoff_necessary_intervals"]
    return {"status": "verified", "result_sha256": base.digest(output / "result.json"),
            "input_arrays_checked": 56, "repair_archives_checked": 10, "previous_artifacts_unchanged": True,
            "views_intervals_oracle_bounds_and_metrics_verified": True, "model_fits": 0,
            "uses_test_labels_for_relaxation_not_model_training": True}


def run(preparation, previous, output):
    base.require_compute()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    r = {"status": "started", "code_commit": os.environ["EXPECTED_COMMIT"], "slurm_job_id": os.environ["SLURM_JOB_ID"],
         "created_utc": datetime.now(timezone.utc).isoformat(), "host": platform.node(), "python": platform.python_version(),
         "model_fits": 0, "neural_inference_calls": 0,
         "versions": {name: module.__version__ for name, module in (("numpy", np), ("scipy", base.scipy),
                      ("sklearn", base.sklearn), ("joblib", base.joblib), ("threadpoolctl", base.threadpoolctl))},
         "source_sha256": {name: base.digest(STUDY / name) for name in ("AEA_REPAIR_ENVELOPE.md",
             "checks/aea_repair.py", "checks/aea_diagnostic.py", "checks/aea_geometry.py", "requirements-svm.txt",
             "reproduction/analyze_results.py", "reproduction/run_experiment.py", "reported/table_3.csv")}, "cases": {}}
    def save():
        (output / "result.json").write_text(json.dumps(r, indent=2, sort_keys=True, allow_nan=False) + "\n")
    save()
    try:
        if tuple(r["versions"].values()) != ("2.5.1", "1.18.0", "1.9.0", "1.5.3", "3.6.0"):
            raise RuntimeError("Use the pinned CPU environment")
        r["previous_audit"] = verify_prior(preparation, previous)
        cases, r["inputs"] = base.load_pair(preparation)
        for level, a in cases.items():
            r["cases"][level] = {}
            for branch in BRANCHES:
                saved = view(a, branch)
                if branch == "feature_z_mse":
                    with np.load(previous / f"{level}_scores.npz", allow_pickle=False) as old:
                        np.testing.assert_array_equal(saved["lower"], old["test_lower_mse"])
                        np.testing.assert_array_equal(saved["upper"], old["test_upper_mse"])
                path = output / f"{level}_{branch}.npz"
                np.savez_compressed(path, **saved)
                r["cases"][level][branch] = {"archive_sha256": base.digest(path),
                    "analysis": analyze(saved, branch, 0. if level == "p00" else .3)}
                save()
        r["common_cutoff_necessary_intervals"] = common_intervals(r["cases"])
        r["linear_output_analytic_control"] = {"minimum_error": 0., "finite_error_upper_bound": False,
            "range_only_argument_inconclusive": True, "does_not_establish_neural_attainability": True}
        r["status"] = "complete"
    except Exception as exc:
        r.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        r["elapsed_seconds_excluding_audit"] = time.perf_counter() - started
        r["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save()
    try:
        checked = audit(preparation, previous, output)
    except Exception as exc:
        r.update(status="failed", failure_stage="artifact_audit", error_type=type(exc).__name__, error=str(exc))
        save()
        raise
    (output / "artifact_audit.json").write_text(json.dumps(checked, indent=2, sort_keys=True) + "\n")
    return checked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if bool(args.output) == bool(args.audit):
        parser.error("Choose output or audit")
    result = audit(args.preparation, args.previous, args.audit) if args.audit else run(args.preparation, args.previous, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
