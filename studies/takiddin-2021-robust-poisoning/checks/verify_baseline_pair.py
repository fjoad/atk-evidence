#!/usr/bin/env python3
"""Read-only artifact audit: connect saved predictions to both frozen inputs."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "reproduction"))
from run_experiment import load_preparation
from analyze_results import audit_result, digest


def verify(preparation, attempt):
    checked, loaded, rows = {}, {}, []
    for level in ("p00", "p30"):
        directory = attempt / level
        record = json.loads((directory / "result.json").read_text())
        arrays, metadata, identities = load_preparation(preparation / f"generalized-two-class-{level}")
        if identities != record["input_sha256"]:
            raise ValueError("Consumed inputs differ from preserved preparations")
        if digest(preparation / f"generalized-two-class-{level}/metadata.json") != record["preparation_metadata_sha256"]:
            raise ValueError("Preparation metadata changed")
        if record["actual_poisoning"] != metadata["poisoning"]:
            raise ValueError("Poisoning record differs from preparation")
        with np.load(directory / "predictions.npz", allow_pickle=False) as scores:
            for saved, original in (("labels", "test_true_y"), ("uid", "test_uid"),
                                    ("attack", "test_attack"), ("synthetic", "test_synthetic")):
                np.testing.assert_array_equal(scores[saved], arrays[original])
            np.testing.assert_array_equal(scores["negative_daily_mean"],
                -arrays["test_raw_x"].mean(axis=1, dtype=np.float64))
        checked[level] = audit_result(directory)
        loaded[level] = arrays
        rows.append(record)
    for name in ("train_x", "train_true_y", "test_x", "test_raw_x", "test_uid",
                 "test_true_y", "test_observed_y", "test_synthetic", "test_attack"):
        np.testing.assert_array_equal(loaded["p00"][name], loaded["p30"][name])
    if rows[0]["model"] != rows[1]["model"] or rows[0]["fitting_parameters"] != rows[1]["fitting_parameters"]:
        raise ValueError("Model/settings changed within the pair")
    if rows[0]["code_commit"] != rows[1]["code_commit"] or rows[0]["versions"] != rows[1]["versions"]:
        raise ValueError("Code/runtime changed within the pair")
    if rows[0]["model"] == "feed_forward":
        from run_experiment import weight_hash
        for level, record in zip(("p00", "p30"), rows):
            with np.load(attempt / level / "initial_weights.npz", allow_pickle=False) as initial:
                weights = [initial[f"weight_{i}"] for i in range(len(initial.files))]
                if weight_hash(weights) != record["neural"]["initial_weights_sha256"]:
                    raise ValueError("Initial weight identity differs")
            history = json.loads((attempt / level / "history.json").read_text())
            if (len(history) != 50 or [r["epoch"] for r in history] != list(range(1, 51))
                    or [r["optimizer_updates"] for r in history] != [45 * e for e in range(1, 51)]
                    or not record["neural"]["training_complete"]):
                raise ValueError("Neural training schedule differs from contract")
            if not all(np.isfinite(r["loss"]) and np.isfinite(r["binary_accuracy"]) for r in history):
                raise ValueError("Invalid neural training history")
        if rows[0]["neural"]["initial_weights_sha256"] != rows[1]["neural"]["initial_weights_sha256"]:
            raise ValueError("Neural initializations differ across the poison pair")
    changed = int(np.count_nonzero(loaded["p00"]["train_observed_y"] != loaded["p30"]["train_observed_y"]))
    if changed != 675:
        raise ValueError("Unexpected label change count")
    return {"status": "verified", "model": rows[0]["model"], "code_commit": rows[0]["code_commit"],
            "input_arrays_checked": 20, "training_rows": 4464, "test_rows": 2232,
            "changed_training_labels": changed, "features_and_test_identities_match": True,
            "predictions_bound_to_prepared_labels": True,
            "attempts": checked}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.preparation, args.attempt), indent=2, sort_keys=True, allow_nan=False))
