"""Read-only Phase-6 checks supplementing the frozen numerical-anchor audit.

No training, threshold selection, source repair, or new model attempt occurs.
The only write is a new JSON check record; existing artifacts are never changed.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Full artifact inspection must run on a compute node")
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    repo = Path.cwd()
    result = json.loads(args.result.read_text())
    run = args.result.parent
    data = repo / result["data"]["path"]
    metadata = json.loads((data / "metadata.json").read_text())
    checks = {}
    record = {
        "source_result": str(args.result.resolve()),
        "source_result_sha256": hashlib.sha256(args.result.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "classification": "P+I/N artifact verification; no new scientific attempt",
        "checks": checks,
    }

    def check(name, condition):
        checks[name] = bool(condition)

    def load(name):
        return np.load(data / name, mmap_mode="r")

    check("eligible_code_commit", subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
        == result["git_commit"] == "a88d17477ad96b01ffa44a50d8ce051dd8d2b5ca")
    check("metadata_complete", metadata["status"] == "complete")
    check("source_file_count", len(metadata["source"]["files"]) == 7)
    with (repo / "studies/atk-2022-deep-autoencoder/reported/table_3.csv").open() as handle:
        row = next(r for r in csv.DictReader(handle) if r["model"] == "FC-SAE")
    check("target_matches_source_csv", all(
        float(row[k]) == v for k, v in result["reported_table_3"].items()))

    summaries = {}
    for path in sorted(data.glob("*.npy")) + sorted(run.glob("*.npy")):
        a = np.load(path, mmap_mode="r")
        total = finite = 0
        lo, hi = np.inf, -np.inf
        for start in range(0, len(a), 65536):
            block = np.asarray(a[start:start + 65536])
            total += block.size
            finite += int(np.count_nonzero(np.isfinite(block)))
            lo, hi = min(lo, float(np.min(block))), max(hi, float(np.max(block)))
        key = ("data/" if path.parent == data else "run/") + path.name
        summaries[key] = {"shape": list(a.shape), "dtype": str(a.dtype),
                          "values": total, "nonfinite": total - finite,
                          "min": lo, "max": hi}
        check(key + "_finite", total == finite)
    record["full_array_scan"] = summaries
    print("Full array finite scan complete", flush=True)

    meters, days = load("meter_ids.npy"), load("day_numbers.npy")
    train, b2 = load("train_index.npy"), load("b2_index.npy")
    order = load("table_iv_order.npy")
    check("profile_partition", np.array_equal(
        np.sort(np.concatenate((train, b2))), np.arange(len(meters))))
    train_meters, test_meters = np.unique(meters[train]), np.unique(meters[b2])
    check("customer_disjoint", np.intersect1d(train_meters, test_meters).size == 0)
    check("training_permutation", np.array_equal(np.sort(order), np.arange(len(train))))
    check("training_meter_ids", np.array_equal(load("train_meter_ids.npy"), meters[train]))
    check("training_day_ids", np.array_equal(load("train_day_numbers.npy"), days[train]))
    check("positive_scaler", np.all(load("scaler_scale.npy") > 0))
    record["customer_counts"] = {"training": len(train_meters), "test": len(test_meters)}
    original, test_x = load("test_original_x.npy"), load("x_test.npy")
    labels, attacks = load("y_test.npy"), load("test_attack_id.npy")
    source, synthetic = load("test_source_row.npy"), load("test_is_synthetic.npy")
    n = len(original)
    check("original_source_identity", np.array_equal(
        load("test_original_source_row.npy"), np.tile(b2, 7)))
    check("original_attack_identity", np.array_equal(
        load("test_original_attack_id.npy"), np.repeat(np.arange(7), len(b2))))
    check("original_label_identity", np.array_equal(
        load("test_original_y.npy"), np.repeat([0, 1, 1, 1, 1, 1, 1], len(b2))))
    check("preserved_original_labels", np.array_equal(labels[:n], load("test_original_y.npy")))
    check("preserved_original_provenance", np.array_equal(source[:n], load("test_original_source_row.npy"))
          and np.array_equal(attacks[:n], load("test_original_attack_id.npy")))
    check("synthetic_benign_only", np.all(labels[n:] == 0))
    check("synthetic_provenance", np.all(source[n:] == -1) and np.all(attacks[n:] == -1)
          and not np.any(synthetic[:n]) and np.all(synthetic[n:]))
    check("original_features_preserved", all(np.array_equal(
        test_x[i:i + 65536], original[i:i + 65536]) for i in range(0, n, 65536)))
    for attack in range(7):
        values = load("benign.npy" if attack == 0 else f"attack_{attack}.npy")
        check(f"original_feature_block_{attack}", all(np.array_equal(
            original[attack * len(b2) + i:attack * len(b2) + min(i + 65536, len(b2))],
            values[b2[i:i + 65536]]) for i in range(0, len(b2), 65536)))
    benign, x_train = load("benign.npy"), load("x_train.npy")
    check("training_features_match", all(np.array_equal(
        x_train[i:i + 65536], benign[train[i:i + 65536]]) for i in range(0, len(train), 65536)))

    losses = np.asarray(result["history"]["loss"], dtype=float)
    best, best_epoch, wait, stopped = np.inf, None, 0, None
    for epoch, loss in enumerate(losses, 1):
        if loss < best - 1e-6:
            best, best_epoch, wait = loss, epoch, 0
        else:
            wait += 1
        if epoch >= 10 and wait >= 5:
            stopped = epoch
            break
    stop = result["training_stop"]
    check("finite_loss_history", np.isfinite(losses).all())
    check("stopping_replay", best_epoch == stop["best_epoch"] and best == stop["best_loss"]
          and stopped == stop["stopped_epoch"] == len(losses))

    sys.path.insert(0, str(repo / "studies/atk-2022-deep-autoencoder/reproduction"))
    import torch
    from models import build_fc_sae, validate_fc_sae
    import keras
    from analyze_results import metric_vector, METRICS
    torch.set_num_threads(4)
    model = build_fc_sae(seed=20260824)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.load_weights(run / "model.weights.h5")
    record["weight_reload_warnings"] = [str(w.message) for w in caught]
    validate_fc_sae(model)
    check("finite_weights", all(np.isfinite(w).all() for w in model.get_weights()))
    indices = np.concatenate([np.linspace(i * len(b2), (i + 1) * len(b2) - 1, 32, dtype=int)
                              for i in range(7)] + [np.linspace(n, len(test_x) - 1, 32, dtype=int)])
    sample = np.asarray(test_x[indices])
    with torch.no_grad():
        reconstructed = keras.ops.convert_to_numpy(model(sample, training=False))
    fresh_scores = np.mean((sample - reconstructed) ** 2, axis=1)
    scores = np.load(run / "scores.npy", mmap_mode="r")
    record["weight_reload_sample"] = {"rows": len(indices),
        "maximum_absolute_score_difference": float(np.max(np.abs(fresh_scores - scores[indices]))),
        "rtol": 1e-5, "atol": 1e-6, "device": "cpu"}
    check("fresh_weight_score_agreement", np.allclose(fresh_scores, scores[indices], rtol=1e-5, atol=1e-6))
    check("fresh_weight_prediction_agreement", np.array_equal(fresh_scores > 0.58, scores[indices] > 0.58))
    check("softmax_output_domain", np.isfinite(reconstructed).all() and np.all(reconstructed >= 0)
          and np.allclose(reconstructed.sum(axis=1), 1, atol=1e-6))
    for name, key in [("zero_reconstruction_scores.npy", "zero_reconstruction_full_test"),
                      ("softmax_projection_floor_scores.npy", "softmax_projection_floor_full_test")]:
        values = np.load(run / name, mmap_mode="r")
        metrics, _ = metric_vector(labels, values, threshold=0.58, direction="higher")
        check(key + "_metrics", all(abs(metrics[k] - result["baselines"][key][k]) < 1e-10 for k in METRICS))
    record["elapsed_seconds"] = time.perf_counter() - started
    record["status"] = "passed" if all(checks.values()) else "failed"
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
