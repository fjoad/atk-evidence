#!/usr/bin/env python3
"""Frozen zero-fit novelty-input and output-box check; not an AEA model."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.metrics import roc_auc_score
import threadpoolctl

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "reproduction"))
from analyze_results import digest, metrics, reported_row, threshold_summary, verify_source
from run_experiment import require_compute
from aea_geometry import mse_box_bounds, interval_decisions

METADATA = {"p00": "dfb6df1a5e5223569682f0066e8d1a2cb4a0a33aa3c13e7974becd463e2f4367",
            "p30": "18c4527896cf2aaba5591b9dbf2b9a869f04788e955316e869f68c3d56c6e449"}
CAPS, THRESHOLDS, GUARDS = (5.2, 18.4), (.505, .51, .515), (0., 1e-6)


def validate_case(a, m):
    if m["mode"] != "novelty" or m["scope"] != "generalized":
        raise ValueError("Need the generalized novelty preparation")
    np.testing.assert_array_equal(a["train_observed_y"], 0)
    np.testing.assert_array_equal(a["test_true_y"], a["test_observed_y"])
    assert not a["train_synthetic"].any()
    for part, count in (("train", m["training_rows"]), ("test", m["test_rows"])):
        assert a[f"{part}_x"].shape == (count, 48)
        assert a[f"{part}_x"].dtype == np.float32
        assert len(np.unique(a[f"{part}_uid"])) == count
        assert np.bincount(a[f"{part}_true_y"], minlength=2).tolist() == m[f"{part}_true_counts"]
        np.testing.assert_array_equal(a[f"{part}_true_y"], a[f"{part}_attack"] > 0)
        original = ~a[f"{part}_synthetic"]
        np.testing.assert_array_equal(a[f"{part}_source_id"][original],
                                     a[f"{part}_meter"][original].astype(np.int64) * 1000 + a[f"{part}_day"][original])
        np.testing.assert_array_equal(a[f"{part}_uid"][original],
                                     7 * a[f"{part}_source_id"][original] + a[f"{part}_attack"][original])
        assert np.all(a["scaler_scale"] > 0)
        reconstructed = (a[f"{part}_raw_x"].astype(np.float64) - a["scaler_mean"]) / a["scaler_scale"]
        np.testing.assert_allclose(reconstructed, a[f"{part}_x"], rtol=5e-7, atol=5e-7)
    assert np.bincount(a["train_observed_y"], minlength=2).tolist() == m["train_observed_counts"]
    contaminated = a["train_true_y"] == 1
    assert int(contaminated.sum()) == m["poisoning"]["changed_rows"]
    assert len(np.unique(a["train_meter"][contaminated])) == m["poisoning"]["selected_customer_count"]
    overlap, ti, vi = np.intersect1d(a["train_uid"], a["test_uid"], return_indices=True)
    assert len(overlap) == m["shared_original_row_identities"]
    np.testing.assert_array_equal(a["train_raw_x"][ti], a["test_raw_x"][vi])
    np.testing.assert_array_equal(a["train_true_y"][ti], a["test_true_y"][vi])
    shared = np.intersect1d(a["train_source_id"], a["test_source_id"][~a["test_synthetic"]])
    assert len(shared) == m["shared_source_days"]
    synthetic = np.flatnonzero(a["test_synthetic"])
    assert len(synthetic) == m["resampling"]["generated"]
    np.testing.assert_array_equal(a["test_true_y"][synthetic], 0)
    originals = {int(a["test_uid"][i]): i for i in np.flatnonzero(~a["test_synthetic"])}
    left = np.array([originals[int(uid)] for uid in a["test_parent_a"][synthetic]], dtype=int)
    right = np.array([originals[int(uid)] for uid in a["test_parent_b"][synthetic]], dtype=int)
    np.testing.assert_array_equal(a["test_true_y"][left], 0)
    np.testing.assert_array_equal(a["test_true_y"][right], 0)
    mix = a["test_mix"][synthetic, None]
    assert np.all((mix >= 0) & (mix <= 1))
    expected = (a["test_raw_x"][left].astype(np.float64) + mix *
                (a["test_raw_x"][right] - a["test_raw_x"][left])).astype(np.float32)
    np.testing.assert_array_equal(expected, a["test_raw_x"][synthetic])
    return {"training_rows": len(a["train_x"]), "test_rows": len(a["test_x"]),
            "training_true_counts": m["train_true_counts"], "test_true_counts": m["test_true_counts"],
            "selected_customer_count": len(np.unique(a["train_meter"][contaminated])),
            "contaminated_training_rows": int(contaminated.sum()), "exact_train_test_overlaps": len(overlap),
            "shared_source_days": len(shared), "synthetic_test_rows": len(synthetic),
            "scaler_transform_verified_without_refit": True, "synthetic_ancestry_verified": True}


def load_pair(root):
    cases, records = {}, {}
    for level, expected in METADATA.items():
        directory = root / f"generalized-novelty-{level}"
        if digest(directory / "metadata.json") != expected:
            raise ValueError("Novelty metadata differs from the frozen preparation")
        m = json.loads((directory / "metadata.json").read_text())
        arrays, hashes = {}, {}
        for name, specification in m["files"].items():
            if Path(name).name != name or not name.endswith(".npy"):
                raise ValueError("Invalid input array path")
            path = directory / name
            hashes[name] = digest(path)
            if hashes[name] != specification["sha256"]:
                raise ValueError(f"Input hash changed: {level}/{name}")
            x = np.load(path, allow_pickle=False)
            if list(x.shape) != specification["shape"] or str(x.dtype) != specification["dtype"] or not np.isfinite(x).all():
                raise ValueError(f"Invalid array: {level}/{name}")
            arrays[name[:-4]] = x
        records[level] = {"metadata_sha256": expected, "array_sha256": hashes,
                          "provenance": validate_case(arrays, m)}
        cases[level] = arrays
    for key in cases["p00"]:
        if key.startswith("test_") and key != "test_x":
            np.testing.assert_array_equal(cases["p00"][key], cases["p30"][key])
    for key in ("train_source_id", "train_meter", "train_day"):
        np.testing.assert_array_equal(cases["p00"][key], cases["p30"][key])
    unchanged = cases["p30"]["train_true_y"] == 0
    np.testing.assert_array_equal(cases["p00"]["train_raw_x"][unchanged], cases["p30"]["train_raw_x"][unchanged])
    return cases, records


def quantiles(x):
    return dict(zip(("min", "q25", "median", "q75", "max"), np.quantile(x, [0, .25, .5, .75, 1]).tolist()))


def make_scores(a):
    saved = {key: a[key].copy() for key in a if key.startswith(("train_", "test_"))
             and key.endswith(("_uid", "_true_y", "_synthetic", "_attack", "_source_id", "_meter"))}
    domains = {}
    for part in ("train", "test"):
        x = a[f"{part}_x"].astype(np.float64)
        lower, upper = mse_box_bounds(x)
        saved[f"{part}_lower_mse"], saved[f"{part}_upper_mse"] = lower, upper
        domains[part] = {"coordinates": int(x.size), "negative_coordinates": int((x < 0).sum()),
                         "above_one_coordinates": int((x > 1).sum()),
                         "rows_with_negative_coordinates": int((x < 0).any(axis=1).sum()),
                         "rows_with_above_one_coordinates": int((x > 1).any(axis=1).sum()),
                         "lower_mse": quantiles(lower), "upper_mse": quantiles(upper)}
    x = a["test_x"].astype(np.float64)
    saved.update(test_zero_mse=np.mean(x ** 2, axis=1), test_half_mse=np.mean((x - .5) ** 2, axis=1),
                 test_negative_daily_mean=-a["test_raw_x"].mean(axis=1, dtype=np.float64))
    return saved, domains


def rate(count, total):
    return 100 * count / total if total else None


def bound_counts(lower, upper, labels, threshold, guard):
    decisions = interval_decisions(np.maximum(lower - guard, 0), upper + guard, threshold)
    by_class = {}
    for label, name in ((0, "benign"), (1, "malicious")):
        selected = labels == label
        counts = {key: int(mask[selected].sum()) for key, mask in decisions.items()}
        counts["n"] = int(selected.sum())
        by_class[name] = counts
    b, m = by_class["benign"], by_class["malicious"]
    return {"threshold": threshold, "numerical_guard": guard, "counts": by_class,
            "minimum_FA": rate(b["must_alarm"], b["n"]),
            "maximum_FA": rate(b["n"] - b["must_be_benign"], b["n"]),
            "minimum_DR": rate(m["must_alarm"], m["n"]),
            "maximum_DR": rate(m["n"] - m["must_be_benign"], m["n"])}


def analyze(saved, poison_rate):
    labels, synthetic = saved["test_true_y"], saved["test_synthetic"]
    target = reported_row(poison_rate, "aea")
    groups = {"all": np.ones(len(labels), bool), "original": ~synthetic, "synthetic": synthetic}
    bounds, baselines, exclusions = {}, {}, {}
    for group, select in groups.items():
        if not select.any():
            continue
        y = labels[select]
        bounds[group], baselines[group] = {}, {}
        for unit, transform in (("mse", lambda x: x), ("rmse", np.sqrt), ("sse", lambda x: 48 * x)):
            lower, upper = [transform(saved[f"test_{side}_mse"][select]) for side in ("lower", "upper")]
            measurements = {str(t): {str(g): bound_counts(lower, upper, y, t, g) for g in GUARDS} for t in THRESHOLDS}
            bounds[group][unit] = measurements
            if group == "all":
                exact = measurements["0.51"]["0.0"]
                favorable_fa = measurements["0.515"]["1e-06"]["minimum_FA"]
                favorable_dr = measurements["0.505"]["1e-06"]["maximum_DR"]
                exclusions[unit] = {"literal_FA_cap_excluded": exact["minimum_FA"] > target["FA"],
                    "literal_DR_target_excluded": exact["maximum_DR"] < target["DR"],
                    "rounding_favorable_minimum_FA": favorable_fa,
                    "rounding_favorable_maximum_DR": favorable_dr,
                    "rounding_favorable_FA_cap": target["FA"] + .05,
                    "rounding_favorable_DR_target": target["DR"] - .05,
                    "rounding_favorable_FA_excluded": favorable_fa > target["FA"] + .05,
                    "rounding_favorable_DR_excluded": favorable_dr < target["DR"] - .05,
                    "endpoints_are_separate_optimistic_necessary_conditions": True}
        for name, field in (("zero_mse", "test_zero_mse"), ("half_mse", "test_half_mse"),
                            ("clipped_input_mse", "test_lower_mse"), ("negative_daily_mean", "test_negative_daily_mean")):
            scores = saved[field][select]
            both = len(np.unique(y)) == 2
            baselines[group][name] = {"n": len(y), "AUC": 100 * float(roc_auc_score(y, scores)) if both else None,
                "metrics_at_0.51": metrics(y, scores > .51, scores) if name != "negative_daily_mean" else None,
                "cutoffs": threshold_summary(y, scores, CAPS) if both else None,
                "diagnostic_reversal": threshold_summary(y, -scores, CAPS) if both else None}
    return {"reported_full_population_context": target, "bounds": bounds,
            "necessary_condition_checks": exclusions, "untrained_baselines": baselines,
            "is_neural_reproduction": False, "bounds_are_not_AUC_limits": True}


def audit(preparation, output):
    r = json.loads((output / "result.json").read_text())
    if r["status"] != "complete":
        raise ValueError("Cannot certify an incomplete diagnostic")
    for name, expected in r["source_sha256"].items():
        verify_source(name, expected, r["code_commit"])
    cases, inputs = load_pair(preparation)
    if inputs != r["inputs"]:
        raise ValueError("Input evidence changed")
    for level, a in cases.items():
        path = output / f"{level}_scores.npz"
        if digest(path) != r["cases"][level]["scores_sha256"]:
            raise ValueError("Saved score artifact changed")
        with np.load(path, allow_pickle=False) as f:
            saved = dict(f)
        expected, domain = make_scores(a)
        if set(saved) != set(expected):
            raise ValueError("Score inventory differs")
        for key in saved:
            np.testing.assert_array_equal(saved[key], expected[key])
        if domain != r["cases"][level]["domain"] or analyze(saved, 0. if level == "p00" else .3) != r["cases"][level]["analysis"]:
            raise ValueError("Saved analysis differs from recomputation")
    return {"status": "verified", "result_sha256": digest(output / "result.json"),
            "input_arrays_checked": sum(len(i["array_sha256"]) for i in inputs.values()),
            "score_artifacts_checked": 2, "provenance_bounds_scores_metrics_verified": True,
            "original_inputs_unchanged": True, "model_fits": 0, "neural_inference_calls": 0}


def run(preparation, output):
    require_compute()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    r = {"status": "started", "created_utc": datetime.now(timezone.utc).isoformat(),
         "code_commit": os.environ["EXPECTED_COMMIT"], "slurm_job_id": os.environ["SLURM_JOB_ID"],
         "host": platform.node(), "scope": "zero-fit fixed novelty-pilot geometry; not AEA performance",
         "python": platform.python_version(), "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
         "slurm_gpu_allocation": os.environ.get("SLURM_JOB_GPUS"),
         "model_fits": 0, "neural_inference_calls": 0,
         "versions": {name: module.__version__ for name, module in
                      (("numpy", np), ("scipy", scipy), ("sklearn", sklearn), ("joblib", joblib), ("threadpoolctl", threadpoolctl))},
         "source_sha256": {name: digest(STUDY / name) for name in (
             "AEA_GEOMETRY_CHECK.md", "AEA_SPECIFICATION.md", "requirements-svm.txt", "checks/aea_geometry.py",
             "checks/aea_diagnostic.py", "reproduction/analyze_results.py", "reproduction/run_experiment.py", "reported/table_3.csv")},
         "cases": {}}
    def save():
        (output / "result.json").write_text(json.dumps(r, indent=2, sort_keys=True, allow_nan=False) + "\n")
    save()
    try:
        if tuple(r["versions"].values()) != ("2.5.1", "1.18.0", "1.9.0", "1.5.3", "3.6.0"):
            raise RuntimeError("Use the preserved CPU numerical environment")
        cases, r["inputs"] = load_pair(preparation)
        for level, a in cases.items():
            scores, domain = make_scores(a)
            path = output / f"{level}_scores.npz"
            np.savez_compressed(path, **scores)
            r["cases"][level] = {"scores_sha256": digest(path), "domain": domain,
                                  "analysis": analyze(scores, 0. if level == "p00" else .3)}
            save()
        r["status"] = "complete"
    except Exception as exc:
        r.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        r["elapsed_seconds"] = time.perf_counter() - start
        r["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save()
    try:
        checked = audit(preparation, output)
    except Exception as exc:
        r.update(status="failed", failure_stage="artifact_audit", error_type=type(exc).__name__, error=str(exc))
        save()
        raise
    (output / "artifact_audit.json").write_text(json.dumps(checked, indent=2, sort_keys=True) + "\n")
    return checked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if bool(args.output) == bool(args.audit):
        parser.error("Choose either --output or --audit")
    result = audit(args.preparation, args.audit) if args.audit else run(args.preparation, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
