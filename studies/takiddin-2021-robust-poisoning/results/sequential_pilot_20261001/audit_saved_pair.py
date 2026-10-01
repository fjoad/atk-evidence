"""Read saved bytes/scores only; no TensorFlow import, inference or fitting."""

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
from analyze_results import audit_result, digest, metrics, threshold_summary
from run_experiment import weight_hash

CAPS = (2.9, 2.95, 5.8, 5.85, 9.3, 24.4)


def without_object_ids(value):
    """Keras archive-only object identity metadata is not a model setting."""
    if isinstance(value, dict):
        return {k: without_object_ids(v) for k, v in value.items() if k != "shared_object_id"}
    if isinstance(value, list):
        return [without_object_ids(v) for v in value]
    return value


def serialized_weights(directory, record):
    """Rebuild get_weights order from the frozen model's named HDF5 groups."""
    front = "layers/sequential_attention_decoder"
    paths = [f"{front}/vars/{i}" for i in range(4)]
    for i in range(3):
        suffix = f"_{i}" if i else ""
        paths.extend(f"{front}/encoder/lstm{suffix}/cell/vars/{j}" for j in range(3))
    for i in range(3):
        suffix = f"_{i}" if i else ""
        paths.extend(f"{front}/decoder/lstm_cell{suffix}/vars/{j}" for j in range(3))
    paths.extend(f"{front}/projection/vars/{i}" for i in range(2))
    for i in range(8):
        suffix = f"_{i}" if i else ""
        paths.extend(f"layers/gru{suffix}/cell/vars/{j}" for j in range(3))
    paths.extend(f"layers/{layer}/vars/{j}" for layer in ("dense", "dense_1") for j in range(2))
    with zipfile.ZipFile(directory / "model.keras") as archive:
        config = json.loads(archive.read("config.json"))
        assert without_object_ids(config) == without_object_ids(record["fitting_parameters"]["model_json"])
        with h5py.File(io.BytesIO(archive.read("model.weights.h5")), "r") as hdf:
            datasets = {}
            hdf.visititems(lambda n, v: datasets.update({n: v[()]}) if isinstance(v, h5py.Dataset) else None)
    expected = set(paths) | {f"optimizer/vars/{i}" for i in range(2 + 2*len(paths))}
    assert set(datasets) == expected
    assert all(np.isfinite(v).all() for v in datasets.values())
    weights = [datasets[p] for p in paths]
    assert sum(w.size for w in weights) == 9240802
    assert weight_hash(weights) == record["neural"]["final_weights_sha256"]
    assert int(datasets["optimizer/vars/0"]) == 2250
    assert float(datasets["optimizer/vars/1"]) == record["fitting_parameters"]["optimizer_config"]["learning_rate"]
    for i, value in enumerate(weights):
        assert datasets[f"optimizer/vars/{2+2*i}"].shape == value.shape
        assert datasets[f"optimizer/vars/{3+2*i}"].shape == value.shape
    # All matrices plus the attention vector carry MaxNorm(1, axis=0).
    norms = {p: float(np.max(np.linalg.norm(v.astype(np.float64), axis=0)))
             for p, v in zip(paths, weights) if v.ndim == 2 or p == front+"/vars/2"}
    assert max(norms.values()) <= 1.00001
    np.testing.assert_allclose(sorted(norms.values()),
        sorted(record["neural"]["maximum_kernel_column_norms"].values()), atol=0, rtol=0)
    with np.load(directory / "initial_weights.npz", allow_pickle=False) as initial:
        original = [initial[f"weight_{i}"] for i in range(len(paths))]
    assert weight_hash(original) == record["neural"]["initial_weights_sha256"]
    changed = {p: int(np.count_nonzero(a != b)) for p, a, b in zip(paths, original, weights)}
    return {"status": "verified", "variables": len(weights), "parameters": 9240802,
            "optimizer_updates": 2250, "config_matches_except_archive_object_ids": True,
            "final_weights_sha256": weight_hash(weights), "all_finite": True,
            "maximum_constrained_norm": max(norms.values()),
            "changed_parameter_counts": changed}


def analyze(attempt):
    output = {"scope": "read-only saved-artifact audit and descriptive score comparison",
              "model_inferences": 0, "model_fits": 0, "cases": {}}
    comparators = {"random_forest": "rf-pilot-20260920-attempt1",
                   "feed_forward": "feed-forward-pilot-20260922-attempt1",
                   "gru": "gru-completion-20260924-attempt1"}
    for level in ("p00", "p30"):
        directory = attempt / level
        record = json.loads((directory / "result.json").read_text())
        result = {"result": audit_result(directory), "serialized": serialized_weights(directory, record)}
        prior = record["training_prior"]
        history = json.loads((directory / "history.json").read_text())
        result["loss_context"] = {"training_attack_label_fraction": prior,
            "optimal_constant_training_bce": float(-prior*np.log(prior)-(1-prior)*np.log1p(-prior)),
            "final_epoch_training_bce": history[-1]["loss"]}
        with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
            saved = {k: archive[k] for k in archive.files}
        scores = saved["probabilities"][:, 1]
        result["probability"] = {"minimum": float(scores.min()), "maximum": float(scores.max()),
                                 "unique_values": int(len(np.unique(scores)))}
        with np.load(directory / "representations.npz", allow_pickle=False) as rep:
            result["representation"] = {key: {"mean": float(rep[key].mean()),
                "median": float(np.median(rep[key]))} for key in
                ("initial_mse", "final_mse", "initial_mae", "final_mae", "zero_mse", "training_mean_mse")}
            initial = rep["initial_probabilities"].astype(np.float64)
            result["initial_classification"] = metrics(saved["labels"], initial>.5, initial)
            result["initial_probability"] = {"minimum": float(initial.min()), "maximum": float(initial.max()),
                "unique_values": int(len(np.unique(initial)))}
            result["representation_range"] = {
                stage: [float(rep[stage].min()), float(rep[stage].max())] for stage in ("initial", "final")}
            result["representation_across_profiles"] = {stage: {
                "unique_profiles": int(len(np.unique(rep[stage], axis=0))),
                "maximum_range_at_a_fixed_time_step": float(np.ptp(rep[stage].astype(np.float64), axis=0).max())}
                for stage in ("initial", "final")}
        result["matched_context"] = {}
        for name, relative in comparators.items():
            other = attempt.parent / relative / level
            prior_audit = audit_result(other)
            with np.load(other / "predictions.npz", allow_pickle=False) as prior:
                for key in ("labels", "uid", "attack", "synthetic", "negative_daily_mean"):
                    np.testing.assert_array_equal(saved[key], prior[key])
                result["matched_context"][name] = {
                    "saved_predictions_sha256": digest(other / "predictions.npz"),
                    "audit_status": prior_audit["status"],
                    "primary": prior_audit["pilot_primary"],
                    "thresholds": threshold_summary(prior["labels"], prior["probabilities"][:, 1], CAPS)}
        output["cases"][level] = result
    output["status"] = "verified"
    output["limits"] = ["dependent twenty-customer pilot; no row-independent uncertainty",
        "saved-score cutoff selection is diagnostic, not independent calibration",
        "previous GRU differs in settings and is not a matched AEA-removal control",
        "representation errors describe outputs; they do not identify a causal mechanism"]
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.attempt), indent=2, sort_keys=True, allow_nan=False))
