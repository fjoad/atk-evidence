#!/usr/bin/env python3
"""One training-only poison-before-ADASYN control; preserve prior results."""

import argparse
from dataclasses import fields, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import sklearn
import scipy
import imblearn
import joblib
import threadpoolctl
from sklearn.preprocessing import StandardScaler

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "reproduction"))
from prepare_data import Rows, balance, poison, library_seed, save_arrays
from run_experiment import fit_one, require_compute
from analyze_results import audit_result, digest, METRICS
from split_resampling import load_rows, read_scores

SEED = 20260920
REFERENCE_HASHES = {
    "B-p00": "d4e2c8904255435cabd909a8dda814ad5c7358c5779af440b1c64bfa5dbf5522",
    "B-p30": "735dd1e71e2ed0b3ea07d8c0344f9e089090af88b2090ea57807180341897aae",
}
EXPECTED_VERSIONS = {"numpy": "2.5.1", "scipy": "1.18.0", "sklearn": "1.9.0",
                     "imblearn": "0.14.2", "joblib": "1.5.3", "threadpoolctl": "3.6.0"}


def versions():
    modules = {"numpy": np, "scipy": scipy, "sklearn": sklearn,
               "imblearn": imblearn, "joblib": joblib, "threadpoolctl": threadpoolctl}
    return {name: module.__version__ for name, module in modules.items()}


def observed_balance(originals, seed):
    """ADASYN sees corrupted labels; retain original truth and explicit unknowns."""
    if originals.synthetic.any():
        raise ValueError("Only original rows may enter the observed-label sampler")
    # This is a SAMPLING-LABEL VIEW, not a change to original ground truth.
    sampled, info = balance(replace(originals, true_y=originals.observed_y), seed=seed)
    sampled.true_y[:len(originals)] = originals.true_y
    lookup = {int(uid): int(label) for uid, label in zip(originals.uid, originals.true_y)}
    synthetic = sampled.synthetic
    pair_attacks = np.array([lookup[int(a)] + lookup[int(b)]
                            for a, b in zip(sampled.parent_a[synthetic], sampled.parent_b[synthetic])])
    unknown = np.zeros(len(sampled), bool)
    unknown[synthetic] = pair_attacks > 0
    sampled.true_y[unknown] = -1
    sampled.attack[unknown] = -1
    info["sampling_labels"] = "observed labels, including declared corruption"
    info["synthetic_parent_attack_counts_0_1_2"] = np.bincount(pair_attacks, minlength=3).tolist()
    info["unknown_synthetic_truth"] = int(unknown.sum())
    return sampled, info


def prepare_control(original_train, original_test, population, rate):
    if original_train.synthetic.any() or original_test.synthetic.any():
        raise ValueError("Control begins with original train and test rows only")
    corrupted, poisoning = poison(original_train, {}, mode="two-class", scope="generalized",
                                  rate=rate, population=population, seed=SEED)
    train, sampling = observed_balance(corrupted, library_seed(SEED, 202))
    scaler = StandardScaler().fit(train.x.astype(np.float64))
    arrays = {"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_}
    for prefix, rows in (("train", train), ("test", original_test)):
        for field in fields(Rows):
            arrays[prefix + "_" + ("raw_x" if field.name == "x" else field.name)] = getattr(rows, field.name)
        arrays[prefix + "_x"] = scaler.transform(rows.x).astype(np.float32)
    metadata = {"scope": "training-only order control, not reproduction", "rate": rate,
                "training_rows": len(train), "test_rows": len(original_test),
                "poisoning": poisoning, "resampling": sampling,
                "train_observed_counts": np.bincount(train.observed_y, minlength=2).tolist(),
                "train_truth_counts_unknown_benign_attack": np.bincount(train.true_y + 1, minlength=3).tolist()}
    return arrays, metadata


def read_inputs(directory, expected_metadata):
    if digest(directory / "metadata.json") != expected_metadata:
        raise ValueError("Input metadata changed")
    metadata = json.loads((directory / "metadata.json").read_text())
    arrays = {}
    for name, info in metadata["files"].items():
        path = directory / name
        if digest(path) != info["sha256"]:
            raise ValueError("Input array changed")
        value = np.load(path, allow_pickle=False)
        if list(value.shape) != info["shape"] or str(value.dtype) != info["dtype"] or not np.isfinite(value).all():
            raise ValueError("Invalid input array")
        arrays[name.removesuffix(".npy")] = value
    return arrays, metadata


def references(root):
    records, inputs = {}, {}
    for case, expected in REFERENCE_HASHES.items():
        path = root / case
        if digest(path / "result.json") != expected:
            raise ValueError("Reference result changed")
        audit_result(path)
        records[case] = json.loads((path / "result.json").read_text())
        inputs[case], _ = read_inputs(path / "inputs", records[case]["input_metadata_sha256"])
    return records, inputs


def verify_inputs(arrays, reference):
    """Read-only checks; no regeneration, model inference, or neighbor search."""
    original = ~arrays["train_synthetic"]
    ref_original = ~reference["train_synthetic"]
    for field in fields(Rows):
        suffix = "raw_x" if field.name == "x" else field.name
        np.testing.assert_array_equal(arrays["train_" + suffix][original],
                                      reference["train_" + suffix][ref_original])
        np.testing.assert_array_equal(arrays["test_" + suffix], reference["test_" + suffix])
    if arrays["test_synthetic"].any() or not np.array_equal(arrays["test_true_y"], arrays["test_observed_y"]):
        raise ValueError("Evaluation must have original rows and clean labels")
    uids = arrays["train_uid"]
    if len(np.unique(uids)) != len(uids) or np.intersect1d(uids[original], arrays["test_uid"]).size:
        raise ValueError("Duplicate training identity or original evaluation overlap")
    lookup = {int(uid): i for i, uid in enumerate(uids) if original[i]}
    try:
        a = np.array([lookup[int(uid)] for uid in arrays["train_parent_a"][~original]], dtype=int)
        b = np.array([lookup[int(uid)] for uid in arrays["train_parent_b"][~original]], dtype=int)
    except KeyError as exc:
        raise ValueError("Synthetic parent is not a training original") from exc
    if np.any(arrays["train_observed_y"][np.concatenate([a, b])]) or np.any(arrays["train_observed_y"][~original]):
        raise ValueError("Synthesis must use observed-benign parents and labels")
    mix = arrays["train_mix"][~original, None]
    if not np.all((mix >= 0) & (mix <= 1)):
        raise ValueError("Invalid interpolation coefficient")
    raw = arrays["train_raw_x"]
    expected = (raw[a] + mix * (raw[b] - raw[a])).astype(raw.dtype)
    np.testing.assert_array_equal(raw[~original], expected)
    pair_attacks = arrays["train_true_y"][a] + arrays["train_true_y"][b]
    truth = np.where(pair_attacks == 0, 0, -1)
    np.testing.assert_array_equal(arrays["train_true_y"][~original], truth)
    np.testing.assert_array_equal(arrays["train_attack"][~original], truth)
    scaler = StandardScaler().fit(raw.astype(np.float64))
    np.testing.assert_array_equal(arrays["scaler_mean"], scaler.mean_)
    np.testing.assert_array_equal(arrays["scaler_scale"], scaler.scale_)
    for part in ("train", "test"):
        np.testing.assert_array_equal(arrays[part + "_x"], scaler.transform(arrays[part + "_raw_x"]).astype(np.float32))
    return {"input_arrays_checked": len(arrays), "original_training_rows": int(original.sum()),
            "evaluation_rows": len(arrays["test_uid"]), "evaluation_unchanged": True,
            "synthetic_parent_attack_counts_0_1_2": np.bincount(pair_attacks, minlength=3).tolist(),
            "unknown_synthetic_truth": int(np.count_nonzero(truth == -1)),
            "train_observed_counts": np.bincount(arrays["train_observed_y"], minlength=2).tolist()}


def fit_control(arrays, output):
    report = fit_one(arrays, output, seed=SEED, workers=4)
    if np.any(arrays["train_true_y"] == -1):
        # fit_one's generic all-row truth diagnostic is undefined here.
        report["training"]["true_label_accuracy"] = None
        report["training"]["true_label_accuracy_note"] = "Unknown synthetic truth; all-row truth accuracy not defined"
    return report


def comparison(record, reference_records):
    old, new = reference_records["B-p30"]["analysis"], record["analysis"]
    thresholds = lambda analysis: analysis["thresholds"]["higher_probability"]
    return {"B_p00": reference_records["B-p00"]["analysis"], "B_p30": old, "D_p30": new,
            "D_minus_B_metric_points": {name: None if new["primary"][name] is None or old["primary"][name] is None
                                         else new["primary"][name] - old["primary"][name] for name in METRICS},
            "D_minus_B_common_cap_DR_points": {cap: thresholds(new)["at_false_alarm_caps"][cap]["DR"]
                                                - thresholds(old)["at_false_alarm_caps"][cap]["DR"]
                                                for cap in ("17.6", "33.3")}}


def audit(output, baseline):
    records, inputs = references(baseline)
    audit_result(output)
    record = json.loads((output / "result.json").read_text())
    arrays, metadata = read_inputs(output / "inputs", record["input_metadata_sha256"])
    checked = verify_inputs(arrays, inputs["B-p30"])
    if checked != record["input_audit"]:
        raise ValueError("Input audit differs from recorded facts")
    for key in ("synthetic_parent_attack_counts_0_1_2", "unknown_synthetic_truth"):
        if metadata["resampling"][key] != checked[key]:
            raise ValueError("Sampling provenance summary changed")
    if (metadata != {**record["preparation"], "files": metadata["files"]}
            or metadata["train_observed_counts"] != checked["train_observed_counts"]
            or metadata["train_truth_counts_unknown_benign_attack"]
            != np.bincount(arrays["train_true_y"] + 1, minlength=3).tolist()
            or metadata["resampling"]["generated"] != int(arrays["train_synthetic"].sum())
            or record["training_prior"] != float(arrays["train_observed_y"].mean())):
        raise ValueError("Training counts, prior or preparation record differ")
    if record["fitting_parameters"] != records["B-p30"]["fitting_parameters"]:
        raise ValueError("Model parameters differ from B")
    if record["zero_poison_guard"] != {"all_arrays_identical": True, "arrays": len(arrays), "new_fits": 0}:
        raise ValueError("Zero-poison guard is not recorded as passed")
    if checked["unknown_synthetic_truth"] and record["training"]["true_label_accuracy"] is not None:
        raise ValueError("Cannot report all-row truth accuracy with unknown labels")
    scores = read_scores(output)
    for score_key, input_key in (("uid", "test_uid"), ("labels", "test_true_y"),
                                 ("attack", "test_attack"), ("synthetic", "test_synthetic")):
        np.testing.assert_array_equal(scores[score_key], arrays[input_key])
    np.testing.assert_array_equal(scores["negative_daily_mean"], -arrays["test_raw_x"].mean(axis=1, dtype=np.float64))
    if comparison(record, records) != record["comparison"]:
        raise ValueError("Comparison differs from saved metrics")
    return {"status": "verified", "result_sha256": digest(output / "result.json"),
            "new_fits": 1, "references_retrained": False, **checked}


def run(preparation, baseline, output):
    require_compute()
    if versions() != EXPECTED_VERSIONS:
        raise RuntimeError("Use the frozen main CPU environment")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=STUDY, text=True).strip()
    if commit != os.environ.get("EXPECTED_COMMIT"):
        raise RuntimeError("Frozen code revision mismatch")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source_paths = [Path(__file__), STUDY / "POISON_BALANCE_CHECK.md", STUDY / "checks/split_resampling.py",
                    *[STUDY / "reproduction" / name for name in (
                        "download_data.py", "prepare_data.py", "models.py", "run_experiment.py", "analyze_results.py")],
                    STUDY / "reported/table_3.csv"]
    record = {"status": "started", "case": "D-p30", "model": "random_forest", "seed": SEED,
              "poisoning_rate": .3, "scope": "controlled training-only order comparison, not reproduction",
              "created_utc": datetime.now(timezone.utc).isoformat(), "code_commit": commit,
              "versions": versions(), "python": platform.python_version(), "host": platform.node(),
              "slurm": {key: os.environ.get(key) for key in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE")},
              "source_sha256": {str(path.relative_to(STUDY)): digest(path) for path in source_paths}}
    result_path = output / "result.json"
    result_path.write_text(json.dumps(record, indent=2) + "\n")
    try:
        records, reference_inputs = references(baseline)
        train, test = load_rows(preparation)
        train, test = train.take(~train.synthetic), test.take(~test.synthetic)
        population = np.unique(np.concatenate([train.meter, test.meter]))
        if (len(train), len(test), len(population)) != (2632, 1288, 20):
            raise ValueError("Wrong original pilot cardinalities")
        tick = time.perf_counter()
        zero, _ = prepare_control(train, test, population, 0.)
        if zero.keys() != reference_inputs["B-p00"].keys():
            raise ValueError("Zero-poison array inventory differs")
        for key in zero:
            np.testing.assert_array_equal(zero[key], reference_inputs["B-p00"][key], err_msg=key)
        record["zero_poison_guard"] = {"all_arrays_identical": True, "arrays": len(zero), "new_fits": 0}
        arrays, metadata = prepare_control(train, test, population, .3)
        if metadata["poisoning"]["changed_rows"] != 675:
            raise ValueError("Wrong poisoning count")
        record["input_audit"] = verify_inputs(arrays, reference_inputs["B-p30"])
        save_arrays(output / "inputs", arrays, metadata)
        record["preparation_seconds"] = time.perf_counter() - tick
        record["input_metadata_sha256"] = digest(output / "inputs/metadata.json")
        record["preparation"] = metadata
        record.update(fit_control(arrays, output))
        if record["fitting_parameters"] != records["B-p30"]["fitting_parameters"]:
            raise ValueError("Model parameters differ from B")
        record["comparison"] = comparison(record, records)
        # Recheck immutable references after the new fit.
        references(baseline)
        record["status"] = "complete"
    except Exception as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        record["total_seconds"] = time.perf_counter() - started
        record["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result_path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if not args.audit_only:
        if args.preparation is None:
            parser.error("Execution needs --preparation")
        run(args.preparation, args.baseline, args.output)
    report = audit(args.output, args.baseline)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if not args.audit_only:
        with (args.output / "artifact_audit.json").open("x") as stream:
            stream.write(encoded)
    print(encoded, end="")
