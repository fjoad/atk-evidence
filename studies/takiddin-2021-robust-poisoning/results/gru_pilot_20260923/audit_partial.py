#!/usr/bin/env python3
"""Read-only audit of job 402378's partial p00 artifacts; no model inference."""

import argparse
import io
import json
from pathlib import Path
import sys
import zipfile

import h5py
import numpy as np

STUDY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STUDY / "reproduction"))
from analyze_results import analyze_scores, audit_result, digest, verify_source
from run_experiment import load_preparation, normalized_softmax, weight_hash


def without_object_ids(value):
    """Keras serialization may add process-local IDs, not model settings."""
    if isinstance(value, dict):
        return {k: without_object_ids(v) for k, v in value.items() if k != "shared_object_id"}
    if isinstance(value, list):
        return [without_object_ids(v) for v in value]
    return value


def verify(attempt, preparation):
    directory = attempt / "p00"
    result_path = directory / "result.json"
    before = digest(result_path)
    assert before == "fa19293a1bac847fd7fb3ab8fe37b25f3ea92f52d88ad8e53cee129bbbdd4602"
    r = json.loads(result_path.read_text())
    assert r["status"] == "partial" and not (attempt / "p30").exists()
    assert r["stop_reason"] == "fifteen-minute batch-boundary guard reached"
    assert r["model"] == "gru" and r["code_commit"] == "46966c7e021e964ff582e1eb56422b75cc6a4ec7"
    for relative, expected in r["source_sha256"].items():
        verify_source(relative, expected, r["code_commit"])
    for name, expected in r["output_sha256"].items():
        assert digest(directory / name) == expected, name
    arrays, metadata, identities = load_preparation(preparation / r["case"])
    assert identities == r["input_sha256"]
    assert digest(preparation / r["case"] / "metadata.json") == r["preparation_metadata_sha256"]
    assert r["actual_poisoning"] == metadata["poisoning"] and r["poisoning_rate"] == 0
    assert float(arrays["train_observed_y"].mean()) == r["training_prior"]
    with np.load(directory / "predictions.npz", allow_pickle=False) as f:
        scores = dict(f)
    for saved, original in (("labels", "test_true_y"), ("uid", "test_uid"),
                            ("attack", "test_attack"), ("synthetic", "test_synthetic")):
        np.testing.assert_array_equal(scores[saved], arrays[original])
    np.testing.assert_array_equal(scores["negative_daily_mean"], -arrays["test_raw_x"].mean(axis=1, dtype=np.float64))
    np.testing.assert_array_equal(scores["probabilities"], normalized_softmax(scores["raw_probabilities"]))
    np.testing.assert_array_equal(scores["predictions"], scores["raw_probabilities"].argmax(axis=1))
    assert analyze_scores(scores, r["training_prior"], r["false_alarm_caps"]) == r["analysis"]
    with np.load(directory / "initial_weights.npz", allow_pickle=False) as f:
        initial = [f[f"weight_{i}"] for i in range(len(f.files))]
    n = r["neural"]
    assert weight_hash(initial) == n["initial_weights_sha256"]
    history = json.loads((directory / "history.json").read_text())
    assert len(history) == 33 and [h["epoch"] for h in history] == list(range(1, 34))
    assert [h["optimizer_updates"] for h in history] == [45 * e for e in range(1, 33)] + [1447]
    assert [h["epoch_complete"] for h in history] == [True] * 32 + [False]
    assert all(np.isfinite(h["loss"]) and 0 <= h["categorical_accuracy"] <= 1 and h["seconds"] > 0 for h in history)
    assert n["epochs_completed"] == 32 and n["optimizer_updates"] == 1447
    assert n["batch_guard_reached"] and not n["training_complete"]
    assert 900 <= r["timing_seconds"]["fit"] < 901
    assert all(r["reload"].values())  # Recorded on-GPU reload, not replayed locally.
    with zipfile.ZipFile(directory / "model.keras") as archive:
        config = json.loads(archive.read("config.json"))
        assert without_object_ids(config) == without_object_ids(n["model_json"])
        with h5py.File(io.BytesIO(archive.read("model.weights.h5")), "r") as f:
            final = []
            for i in range(8):
                layer = "gru" if i == 0 else f"gru_{i}"
                final.extend(f[f"layers/{layer}/cell/vars/{j}"][()] for j in range(3))
            final.extend(f[f"layers/dense/vars/{j}"][()] for j in range(2))
            assert int(f["optimizer/vars/0"][()]) == 1447
    assert sum(v.size for v in final) == n["parameters"] == 4058702
    assert weight_hash(final) == n["final_weights_sha256"] != n["initial_weights_sha256"]
    norms = {}
    for i in range(8):
        for j, label in enumerate(("kernel", "recurrent_kernel")):
            norms[f"gru_{i + 1}/{label}"] = float(np.linalg.norm(final[i * 3 + j].astype(np.float64), axis=0).max())
    norms["output/kernel"] = float(np.linalg.norm(final[-2].astype(np.float64), axis=0).max())
    assert norms == n["maximum_kernel_column_norms"] and max(norms.values()) <= 5.00001
    try:
        audit_result(directory)
    except ValueError as exc:
        assert str(exc) == "Only a completed fit can be audited"
    else:
        raise AssertionError("Partial result passed the completed-fit auditor")
    assert digest(result_path) == before
    return {"status": "verified_partial_artifacts_not_completed_experiment",
            "result_sha256": before, "input_arrays_checked": len(identities),
            "output_hashes_checked": r["output_sha256"], "epochs_completed": 32,
            "partial_epoch_batches": 7, "optimizer_updates": 1447,
            "initial_and_final_weight_hashes_verified": True,
            "serialized_optimizer_iterations_verified": True,
            "kernel_norms_and_model_config_verified": True,
            "saved_scores_labels_metrics_verified": True, "original_result_unchanged": True,
            "completed_result_auditor_correctly_refuses_partial": True,
            "p30_started": False, "local_model_inference_performed": False,
            "new_fits": 0, "audit_script_sha256": digest(Path(__file__))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.attempt, args.preparation), indent=2, sort_keys=True))
