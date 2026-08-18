#!/usr/bin/env python3
"""Train one Paper 1 model/seed and preserve an immutable result attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
import sklearn
import torch
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import OneClassSVM, SVC

from models import SPECS, build_model, layer_inventory


REPO = Path(__file__).resolve().parents[3]
DEFAULT_DATA = (
    REPO / "data/derived/atk-2022-deep-autoencoder/reproduction/p0-full-none"
)
DEFAULT_RESULTS = (
    REPO
    / "data/derived/atk-2022-deep-autoencoder/reproduction/results/runs"
)
REPORTED = {
    "fc_sae": {"DR": 81, "FA": 15, "SP": 85, "PR": 81, "ACC": 83, "F1": 81, "AUC": 81},
    "lstm_sae": {"DR": 85, "FA": 13, "SP": 87, "PR": 85, "ACC": 86, "F1": 85, "AUC": 82},
    "fc_vae": {"DR": 88, "FA": 11, "SP": 89, "PR": 89, "ACC": 88.5, "F1": 88.5, "AUC": 85},
    "lstm_vae": {"DR": 91, "FA": 7, "SP": 93, "PR": 91, "ACC": 92, "F1": 91, "AUC": 86},
    "lstm_aea": {"DR": 94, "FA": 5, "SP": 95, "PR": 93, "ACC": 94.5, "F1": 93.5, "AUC": 90},
    "naive_bayes": {"DR": 73, "FA": 18, "SP": 82, "PR": 73, "ACC": 77.5, "F1": 73, "AUC": 70},
    "arima": {"DR": 86, "FA": 12, "SP": 88, "PR": 86, "ACC": 87, "F1": 86, "AUC": 87},
    "one_class_svm": {"DR": 90, "FA": 9, "SP": 91, "PR": 89, "ACC": 90.5, "F1": 89.5, "AUC": 87},
    "supervised_feed_forward": {"DR": 90, "FA": 11, "SP": 89, "PR": 89, "ACC": 89.5, "F1": 89.5, "AUC": 88},
    "supervised_lstm": {"DR": 90.5, "FA": 10, "SP": 90, "PR": 89.5, "ACC": 90, "F1": 90, "AUC": 89},
    "multiclass_svm": {"DR": 91, "FA": 8, "SP": 92, "PR": 90, "ACC": 91.5, "F1": 90.5, "AUC": 89},
}
REPORTED_TABLE_2 = {
    "fc_sae": {"DR": 83, "FA": 14, "SP": 86, "PR": 83, "ACC": 84.5, "F1": 83, "AUC": 83},
    "lstm_sae": {"DR": 86, "FA": 12, "SP": 88, "PR": 87, "ACC": 87, "F1": 86.5, "AUC": 85},
    "fc_vae": {"DR": 90, "FA": 9, "SP": 91, "PR": 91, "ACC": 90.5, "F1": 90.5, "AUC": 88},
    "lstm_vae": {"DR": 93, "FA": 6, "SP": 94, "PR": 93, "ACC": 93.5, "F1": 93, "AUC": 90},
    "lstm_aea": {"DR": 96, "FA": 4, "SP": 96, "PR": 95, "ACC": 96, "F1": 95.5, "AUC": 93},
    "naive_bayes": {"DR": 75, "FA": 16, "SP": 84, "PR": 75, "ACC": 79.5, "F1": 77, "AUC": 73},
    "arima": {"DR": 88, "FA": 10, "SP": 90, "PR": 87, "ACC": 89, "F1": 87, "AUC": 88},
    "one_class_svm": {"DR": 91, "FA": 8.5, "SP": 91.5, "PR": 90, "ACC": 91, "F1": 90, "AUC": 89},
    "supervised_feed_forward": {"DR": 91, "FA": 9.5, "SP": 90.5, "PR": 90, "ACC": 91, "F1": 90.5, "AUC": 89},
    "supervised_lstm": {"DR": 91.5, "FA": 9, "SP": 91, "PR": 90.5, "ACC": 91, "F1": 91, "AUC": 90},
    "multiclass_svm": {"DR": 92, "FA": 7.5, "SP": 92.5, "PR": 91, "ACC": 92, "F1": 91.5, "AUC": 90},
}
REPORTED_TABLE_4 = {
    "fc_sae": {
        "half": {"training_minutes": 72, "ACC": 70},
        "three_quarter": {"training_minutes": 97, "ACC": 78.5},
        "full": {"training_minutes": 137, "ACC": 83},
    },
    "lstm_sae": {
        "half": {"training_minutes": 90, "ACC": 75},
        "three_quarter": {"training_minutes": 127, "ACC": 83},
        "full": {"training_minutes": 183, "ACC": 86},
    },
}
REPORTED_TABLE_5_FC_SAE = {
    1: {"DR": 82.5, "FA": 15},
    2: {"DR": 81, "FA": 16},
    3: {"DR": 83, "FA": 10},
    4: {"DR": 80, "FA": 17},
    5: {"DR": 80, "FA": 17},
    6: {"DR": 80, "FA": 19},
}
CLASSICAL_BENCHMARKS = ("naive_bayes", "arima", "one_class_svm", "multiclass_svm")
NEURAL_BENCHMARKS = ("supervised_feed_forward", "supervised_lstm")
BENCHMARKS = (*CLASSICAL_BENCHMARKS, *NEURAL_BENCHMARKS)


def table_context(metadata: dict[str, object]) -> tuple[int, str, dict[str, dict[str, float]]]:
    """Return the one table/dataset target selected by prepared data."""

    if str(metadata.get("dataset", "ISET")).upper() == "SGCC":
        return 2, "SGCC", REPORTED_TABLE_2
    return 3, "ISET", REPORTED


def prepared_method(metadata: dict[str, object]) -> str:
    """Read the cache method, retaining compatibility with tiny test fixtures."""

    return str(metadata.get("method", "I-ADASYN-NONE-ISET"))


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def stable_id(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "<unavailable>"


def confusion_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    direction: str = "higher",
) -> dict[str, float | int]:
    predictions = scores > threshold if direction == "higher" else scores < threshold
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=bool)
    tp = int(np.count_nonzero(predictions & (labels == 1)))
    tn = int(np.count_nonzero(~predictions & (labels == 0)))
    fp = int(np.count_nonzero(predictions & (labels == 0)))
    fn = int(np.count_nonzero(~predictions & (labels == 1)))
    dr = tp / (tp + fn) if tp + fn else float("nan")
    fa = fp / (fp + tn) if fp + tn else float("nan")
    sp = 1 - fa
    precision = tp / (tp + fp) if tp + fp else 0.0
    accuracy = (dr + sp) / 2
    f1 = 2 * dr * precision / (dr + precision) if dr + precision else 0.0
    oriented = scores if direction == "higher" else -scores
    auc = float(roc_auc_score(labels, oriented))
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "DR": 100 * dr, "FA": 100 * fa, "SP": 100 * sp,
        "PR": 100 * precision, "ACC": 100 * accuracy,
        "F1": 100 * f1, "AUC": 100 * auc,
    }


def score_mse(
    model: keras.Model,
    values: np.ndarray,
    target: Path,
    *,
    batch_size: int,
    score_kind: str = "mse",
) -> tuple[np.memmap, float]:
    started = time.perf_counter()
    scores = np.lib.format.open_memmap(
        target, mode="w+", dtype="float32", shape=(values.shape[0],)
    )
    for start in range(0, values.shape[0], batch_size):
        stop = min(start + batch_size, values.shape[0])
        batch = np.asarray(values[start:stop], dtype=np.float32)
        with torch.no_grad():
            reconstruction = keras.ops.convert_to_numpy(model(batch, training=False))
        mse = np.mean(np.square(batch - reconstruction), axis=1)
        if score_kind == "mse":
            scores[start:stop] = mse
        elif score_kind == "reconstruction_probability":
            # The paper omits the probability aggregation and output variance.
            # This is the frozen fixed-unit-variance geometric-mean completion.
            scores[start:stop] = np.exp(-0.5 * mse)
        else:
            raise ValueError(f"unsupported anomaly score {score_kind}")
    scores.flush()
    return scores, time.perf_counter() - started


def score_zero(
    values: np.ndarray,
    target: Path,
    *,
    batch_size: int,
    score_kind: str = "mse",
) -> np.memmap:
    scores = np.lib.format.open_memmap(
        target, mode="w+", dtype="float32", shape=(values.shape[0],)
    )
    for start in range(0, values.shape[0], batch_size):
        stop = min(start + batch_size, values.shape[0])
        batch = np.asarray(values[start:stop], dtype=np.float32)
        mse = np.mean(np.square(batch), axis=1)
        scores[start:stop] = (
            mse if score_kind == "mse" else np.exp(-0.5 * mse)
        )
    scores.flush()
    return scores


def recover_failed_scoring(
    run: Path, data: Path, *, score_batch_override: int | None = None
) -> int:
    """Score preserved weights in a fresh process after a post-fit failure."""

    run = run.resolve()
    failure_path = run / "failure.json"
    weights_path = run / "model.weights.h5"
    output_path = run / "score_recovery.json"
    if output_path.is_file():
        print(output_path.read_text())
        return 0
    if not failure_path.is_file() or not weights_path.is_file():
        raise ValueError("recovery requires preserved failure.json and weights")

    failure = json.loads(failure_path.read_text())
    config = failure["configuration"]
    model_name = str(config["model"])
    if model_name not in SPECS:
        raise ValueError("score recovery is only for proposed anomaly models")
    metadata_path = data / "metadata.json"
    if sha256(metadata_path) != config["data_metadata_sha256"]:
        raise ValueError("recovery data do not match the failed training attempt")

    started = time.perf_counter()
    model = build_model(
        model_name,
        seed=int(config["seed"]),
        learning_rate=float(config["learning_rate"]),
    )
    model.load_weights(weights_path)
    test_view = str(config["test_view"])
    x_name = "x_test.npy" if test_view == "adasyn" else "test_original_x.npy"
    y_name = "y_test.npy" if test_view == "adasyn" else "test_original_y.npy"
    x_test = np.load(data / x_name, mmap_mode="r")
    y_test = np.load(data / y_name, mmap_mode="r")
    score_batch = (
        int(config["score_batch"])
        if score_batch_override is None
        else score_batch_override
    )
    if score_batch < 1:
        raise ValueError("recovery score batch must be positive")
    scores, score_seconds = score_mse(
        model,
        x_test,
        run / "scores.npy",
        batch_size=score_batch,
        score_kind=SPECS[model_name].anomaly_score,
    )
    predictions = (
        scores > SPECS[model_name].threshold
        if SPECS[model_name].anomaly_direction == "higher"
        else scores < SPECS[model_name].threshold
    )
    np.save(run / "predictions.npy", np.asarray(predictions, dtype=np.int8))
    metrics = confusion_metrics(
        y_test,
        scores,
        threshold=SPECS[model_name].threshold,
        direction=SPECS[model_name].anomaly_direction,
    )
    zero_scores = score_zero(
        x_test,
        run / "zero_reconstruction_scores.npy",
        batch_size=score_batch,
        score_kind=SPECS[model_name].anomaly_score,
    )
    recovery = {
        "status": "success",
        "kind": "operational_score_recovery",
        "eligibility": (
            "exploratory_paper_primary_P0"
            if test_view == "adasyn"
            else "exploratory_interpretation_I-ADASYN-NONE"
        ),
        "configuration": config,
        "git_commit": git_commit(),
        "training_git_commit": failure.get("git_commit"),
        "source_failure": str(failure_path),
        "recovery_score_batch": score_batch,
        "data": {
            "path": str(data.resolve()),
            "metadata_sha256": sha256(metadata_path),
            "counts": {
                "test_profiles": int(x_test.shape[0]),
                "test_benign": int(np.count_nonzero(y_test == 0)),
                "test_malicious": int(np.count_nonzero(y_test == 1)),
            },
        },
        "model": {
            "inventory": layer_inventory(model),
            "parameters": int(model.count_params()),
        },
        "metrics": metrics,
        "zero_reconstruction_metrics": confusion_metrics(
            y_test,
            zero_scores,
            threshold=SPECS[model_name].threshold,
            direction=SPECS[model_name].anomaly_direction,
        ),
        "reported_table_3": REPORTED[model_name],
        "timing_seconds": {
            "failed_training_attempt_through_scoring_failure": float(
                failure["elapsed_seconds"]
            ),
            "recovered_score_table_3": score_seconds,
            "recovery_total": time.perf_counter() - started,
        },
        "recovery_note": (
            "Weights were loaded in a fresh process and scored with the original "
            "recorded inference batch. No fitting occurred; the failed record is "
            "preserved. VAE inference uses the latent mean when training=False."
        ),
        "artifacts": {
            "scores": "scores.npy",
            "predictions": "predictions.npy",
            "weights": "model.weights.h5",
            "zero_reconstruction_scores": "zero_reconstruction_scores.npy",
        },
    }
    save_json(output_path, recovery)
    print(json.dumps(metrics, indent=2))
    print(f"saved score recovery: {output_path}")
    return 0


def score_untrained_sample(
    model: keras.Model,
    values: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float,
    batch_size: int,
    score_kind: str = "mse",
    direction: str = "higher",
) -> dict[str, object]:
    benign = np.flatnonzero(labels == 0)[:5_000]
    malicious = np.flatnonzero(labels == 1)[:5_000]
    index = np.concatenate([benign, malicious])
    scores = np.empty(index.size, dtype=np.float32)
    for start in range(0, index.size, batch_size):
        stop = min(start + batch_size, index.size)
        batch = np.asarray(values[index[start:stop]], dtype=np.float32)
        reconstruction = keras.ops.convert_to_numpy(model(batch, training=False))
        mse = np.mean(np.square(batch - reconstruction), axis=1)
        scores[start:stop] = (
            mse if score_kind == "mse" else np.exp(-0.5 * mse)
        )
    return {
        "rows": int(index.size),
        "metrics": confusion_metrics(
            labels[index], scores, threshold=threshold, direction=direction
        ),
        "score": {
            "minimum": float(scores.min()),
            "median": float(np.median(scores)),
            "maximum": float(scores.max()),
        },
    }


def table_v(
    model: keras.Model,
    data: Path,
    run: Path,
    *,
    threshold: float,
    score_batch: int,
    score_kind: str,
    direction: str,
) -> tuple[list[dict[str, object]], float]:
    started = time.perf_counter()
    benign = np.load(data / "benign.npy", mmap_mode="r")
    benign_scores, _ = score_mse(
        model,
        benign,
        run / "table_v_benign_scores.npy",
        batch_size=score_batch,
        score_kind=score_kind,
    )
    rows: list[dict[str, object]] = []
    for attack_id in range(1, 7):
        attacked = np.load(data / f"attack_{attack_id}.npy", mmap_mode="r")
        attack_scores, _ = score_mse(
            model,
            attacked,
            run / f"table_v_attack_{attack_id}_scores.npy",
            batch_size=score_batch,
            score_kind=score_kind,
        )
        labels = np.concatenate(
            [np.zeros(benign.shape[0], dtype=np.int8), np.ones(attacked.shape[0], dtype=np.int8)]
        )
        scores = np.concatenate([benign_scores, attack_scores])
        rows.append(
            {
                "attack": attack_id,
                "metrics": confusion_metrics(
                    labels, scores, threshold=threshold, direction=direction
                ),
            }
        )
    return rows, time.perf_counter() - started


def supervised_source_blocks(data: Path) -> list[tuple[Path, int]]:
    """Return the paper's complete all-customer B+M population."""

    blocks = [(data / "benign.npy", 0)] + [
        (data / f"attack_{attack_id}.npy", 1) for attack_id in range(1, 7)
    ]
    for path, _ in blocks:
        if not path.is_file():
            raise ValueError(f"supervised source block is missing: {path}")
        values = np.load(path, mmap_mode="r")
        if values.ndim != 2 or values.shape[1] != 48:
            raise ValueError(f"invalid supervised source shape {path}: {values.shape}")
    return blocks


def exact_random_train_mask(total_rows: int, *, seed: int) -> np.ndarray:
    """Select exactly floor(2N/3) rows using one seeded random permutation."""

    if total_rows < 2:
        raise ValueError("supervised split requires at least two rows")
    order = np.random.default_rng(seed).permutation(total_rows)
    mask = np.zeros(total_rows, dtype=bool)
    mask[order[: (2 * total_rows) // 3]] = True
    return mask


def run_naive_bayes(
    args: argparse.Namespace,
    *,
    metadata: dict[str, object],
    metadata_path: Path,
) -> int:
    """Execute the smallest documented completion of the paper's NB row."""

    features, labels = supervised_population(args.data)
    table_number, dataset_name, reported_rows = table_context(metadata)
    reported = reported_rows["naive_bayes"]
    split_started = time.perf_counter()
    train_mask = exact_random_train_mask(features.shape[0], seed=args.seed)
    train_index = np.flatnonzero(train_mask)
    test_index = np.flatnonzero(~train_mask)
    split_seconds = time.perf_counter() - split_started
    configuration = {
        "method": f"{prepared_method(metadata)}-NAIVE-BAYES",
        "paper_tables": [str(table_number)],
        "scientific_question": f"Does the Gaussian-NB completion reproduce Table {table_number}?",
        "task": "supervised",
        "model": "naive_bayes",
        "seed": args.seed,
        "train_fraction": "full",
        "test_view": "supervised_original",
        "table_v": False,
        "threshold": 0.5,
        "anomaly_direction": "higher",
        "supervised_adasyn": metadata.get("configuration", {}).get("supervised_adasyn", "none"),
        "split": "seeded_exact_row_random_2_to_1",
        "data_metadata_sha256": sha256(metadata_path),
    }
    configuration_id = stable_id({**configuration, "seed": "<seed>"})
    attempt_id = stable_id(configuration)
    configuration["configuration_id"] = configuration_id
    configuration["attempt_id"] = attempt_id
    run = args.output / f"table_{table_number}" / "naive_bayes" / f"seed_{args.seed}_{attempt_id}"
    run.mkdir(parents=True, exist_ok=True)
    result_path = run / "result.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text())
        if existing.get("status") == "success":
            print(json.dumps(existing["metrics"], indent=2))
            print(f"immutable attempt already complete: {run}")
            return 0
    failure_path = run / "failure.json"
    if failure_path.is_file():
        raise RuntimeError(
            f"immutable attempt already failed: {failure_path}; preserve it and "
            "change an execution setting to create a new attempt"
        )

    save_json(run / "config.json", configuration)
    total_started = time.perf_counter()
    try:
        model = GaussianNB(var_smoothing=1e-9)
        fit_started = time.perf_counter()
        model.fit(features[train_index], labels[train_index])
        fit_seconds = time.perf_counter() - fit_started
        positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
        score_started = time.perf_counter()
        scores = model.predict_proba(features[test_index])[:, positive_column]
        score_seconds = time.perf_counter() - score_started
        predictions = (scores > 0.5).astype(np.int8)
        np.save(run / "scores.npy", scores.astype(np.float32))
        np.save(run / "labels.npy", labels[test_index])
        np.save(run / "predictions.npy", predictions)
        np.save(run / "test_global_row.npy", test_index.astype(np.int64))
        metrics = confusion_metrics(labels[test_index], scores, threshold=0.5)
        model_payload = {
            "classes": model.classes_.tolist(),
            "class_count": model.class_count_.tolist(),
            "class_prior": model.class_prior_.tolist(),
            "theta": model.theta_.tolist(),
            "var": model.var_.tolist(),
            "var_smoothing": 1e-9,
        }
        save_json(run / "model.json", model_payload)
        timing = {
            "split": split_seconds,
            "fit": fit_seconds,
            "score_table_3": score_seconds,
            "total": time.perf_counter() - total_started,
        }
        source_file_records = metadata.get("files", {})
        result = {
            "status": "success",
            "eligibility": f"exploratory_interpretation_{prepared_method(metadata)}",
            "configuration": configuration,
            "git_commit": git_commit(),
            "data": {
                "path": str(args.data),
                "metadata_sha256": sha256(metadata_path),
                "dataset": dataset_name,
                "population": "prepared supervised labeled population",
                "counts": {
                    "total": int(features.shape[0]),
                    "train": int(train_index.size),
                    "test": int(test_index.size),
                    "train_by_class": np.bincount(labels[train_index], minlength=2).tolist(),
                    "test_by_class": np.bincount(labels[test_index], minlength=2).tolist(),
                },
                "files": source_file_records,
                "source_nodes": metadata.get("source_nodes", {}),
            },
            "model": {
                "name": "Gaussian Naive Bayes",
                "paper_detail": "the paper names Naive Bayes but gives no variant or hyperparameters",
                "completion": model_payload,
            },
            "metrics": metrics,
            f"reported_table_{table_number}": reported,
            "reported_table_4": None,
            "difference_reproduced_minus_reported": {
                key: float(metrics[key]) - float(value)
                for key, value in reported.items()
            },
            "table_v": None,
            "timing_seconds": timing,
            "runtime": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "sklearn": sklearn.__version__,
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            },
            "artifacts": {
                "scores": "scores.npy",
                "labels": "labels.npy",
                "predictions": "predictions.npy",
                "test_global_row": "test_global_row.npy",
                "model": "model.json",
            },
        }
        save_json(result_path, result)
    except Exception as exc:
        save_json(
            failure_path,
            {
                "status": "failed",
                "configuration": configuration,
                "git_commit": git_commit(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.perf_counter() - total_started,
            },
        )
        raise
    print(json.dumps(result["metrics"], indent=2))
    print(json.dumps(result["timing_seconds"], indent=2))
    print(f"saved immutable attempt: {run}")
    return 0


def supervised_population(
    data: Path, *, multiclass: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize the paper's all-customer B+M supervised population."""

    prepared_x = data / "supervised_x.npy"
    prepared_y = data / "supervised_y.npy"
    if prepared_x.is_file() and prepared_y.is_file():
        features = np.load(prepared_x, mmap_mode="r")
        labels = np.asarray(np.load(prepared_y, mmap_mode="r"), dtype=np.int8)
        if features.ndim != 2 or features.shape[0] != labels.size:
            raise ValueError("invalid prepared supervised SGCC population")
        return features, labels

    blocks = supervised_source_blocks(data)
    arrays = [np.load(path, mmap_mode="r") for path, _ in blocks]
    features = np.concatenate(arrays).astype(np.float32, copy=False)
    labels = np.concatenate(
        [
            np.full(
                values.shape[0],
                index if multiclass else int(index > 0),
                dtype=np.int8,
            )
            for index, values in enumerate(arrays)
        ]
    )
    return features, labels


def capped_positions(total: int, cap: int | None, *, seed: int) -> np.ndarray:
    """Deterministically cap an already-defined population without replacement."""

    if cap is None or cap >= total:
        return np.arange(total, dtype=np.int64)
    if cap < 1:
        raise ValueError("sample cap must be positive")
    return np.sort(
        np.random.default_rng(seed).choice(total, size=cap, replace=False)
    ).astype(np.int64)


def svm_attack_margin(margins: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Orient binary or multiclass SVM margins toward malicious classes."""

    margins = np.asarray(margins, dtype=np.float64)
    classes = np.asarray(classes)
    if margins.ndim == 1:
        return margins if int(classes[1]) == 1 else -margins
    benign_index = int(np.flatnonzero(classes == 0)[0])
    return np.max(np.delete(margins, benign_index, axis=1), axis=1) - margins[
        :, benign_index
    ]


def score_classifier(
    model: keras.Model,
    features: np.ndarray,
    indices: np.ndarray,
    target: Path,
    *,
    batch_size: int,
    two_class_softmax: bool,
) -> tuple[np.memmap, float]:
    started = time.perf_counter()
    scores = np.lib.format.open_memmap(
        target, mode="w+", dtype="float32", shape=(indices.size,)
    )
    for start in range(0, indices.size, batch_size):
        stop = min(start + batch_size, indices.size)
        batch = np.asarray(features[indices[start:stop]], dtype=np.float32)
        probability = keras.ops.convert_to_numpy(model(batch, training=False))
        scores[start:stop] = (
            probability[:, 1] if two_class_softmax else probability.reshape(-1)
        )
    scores.flush()
    return scores, time.perf_counter() - started


def run_classical_benchmark(
    args: argparse.Namespace,
    *,
    metadata: dict[str, object],
    metadata_path: Path,
) -> int:
    """Run ARIMA or either SVM through one explicit bounded completion."""

    model_name = args.model
    table_number, dataset_name, reported_rows = table_context(metadata)
    reported = reported_rows[model_name]
    task = "supervised" if model_name == "multiclass_svm" else "anomaly"
    test_view = getattr(args, "test_view", "original")
    threshold = {"arima": 0.58, "one_class_svm": 0.45, "multiclass_svm": 0.0}[
        model_name
    ]
    configuration: dict[str, object] = {
        "method": f"{prepared_method(metadata)}-{model_name.upper()}",
        "paper_tables": [str(table_number)],
        "scientific_question": f"Does the frozen {model_name} completion reproduce Table {table_number}?",
        "task": task,
        "model": model_name,
        "seed": args.seed,
        "test_view": test_view if task == "anomaly" else "supervised",
        "train_fraction": "full",
        "table_v": False,
        "threshold": threshold,
        "anomaly_direction": "higher",
        "supervised_adasyn": (
            metadata.get("configuration", {}).get("supervised_adasyn", "none")
            if task == "supervised" else None
        ),
        "split": (
            "seeded_exact_row_random_2_to_1" if task == "supervised" else "B1_vs_B2_plus_M"
        ),
        "data_metadata_sha256": sha256(metadata_path),
    }
    if model_name == "arima":
        configuration["completion"] = {
            "order": [1, 1, 0],
            "fit_unit": "pooled_within_profile_transitions",
            "score": "residual_mse",
        }
    else:
        configuration["completion"] = {
            "kernel": "sigmoid",
            "gamma": "scale",
            "train_cap": (
                args.one_class_svm_train_cap
                if model_name == "one_class_svm"
                else args.multiclass_svm_train_cap
            ),
            "test_cap": args.svm_test_cap,
            "score": (
                "binary_positive_class_margin"
                if model_name == "multiclass_svm"
                else "negative_one_class_decision_function"
            ),
        }
    configuration_id = stable_id({**configuration, "seed": "<seed>"})
    configuration["configuration_id"] = configuration_id
    configuration["attempt_id"] = stable_id(configuration)
    run = (
        args.output
        / f"table_{table_number}"
        / model_name
        / f"seed_{args.seed}_{configuration['attempt_id']}"
    )
    run.mkdir(parents=True, exist_ok=True)
    if (run / "result.json").is_file():
        print(f"immutable attempt already complete: {run}")
        return 0
    save_json(run / "config.json", configuration)
    total_started = time.perf_counter()
    try:
        load_started = time.perf_counter()
        if model_name == "multiclass_svm":
            features, multiclass_labels = supervised_population(
                args.data, multiclass=True
            )
            binary_labels = (multiclass_labels > 0).astype(np.int8)
            train_mask = exact_random_train_mask(features.shape[0], seed=args.seed)
            train_available = np.flatnonzero(train_mask)
            test_available = np.flatnonzero(~train_mask)
            train_position = capped_positions(
                train_available.size,
                args.multiclass_svm_train_cap,
                seed=args.seed,
            )
            test_position = capped_positions(
                test_available.size, args.svm_test_cap, seed=args.seed + 1
            )
            train_index = train_available[train_position]
            test_index = test_available[test_position]
            test_labels = binary_labels[test_index]
        else:
            features = np.load(args.data / "x_train.npy", mmap_mode="r")
            test_x_name = (
                "x_test.npy" if test_view == "adasyn" else "test_original_x.npy"
            )
            test_y_name = (
                "y_test.npy" if test_view == "adasyn" else "test_original_y.npy"
            )
            test_features = np.load(
                args.data / test_x_name, mmap_mode="r"
            )
            all_test_labels = np.load(
                args.data / test_y_name, mmap_mode="r"
            )
            if model_name == "one_class_svm":
                train_index = capped_positions(
                    features.shape[0], args.one_class_svm_train_cap, seed=args.seed
                )
                test_index = capped_positions(
                    test_features.shape[0], args.svm_test_cap, seed=args.seed + 1
                )
            else:
                train_index = np.arange(features.shape[0], dtype=np.int64)
                test_index = np.arange(test_features.shape[0], dtype=np.int64)
            test_labels = np.asarray(all_test_labels[test_index], dtype=np.int8)
        load_seconds = time.perf_counter() - load_started

        fit_started = time.perf_counter()
        if model_name == "arima":
            differences = np.diff(np.asarray(features, dtype=np.float32), axis=1)
            lagged = differences[:, :-1].reshape(-1).astype(np.float64)
            targets = differences[:, 1:].reshape(-1).astype(np.float64)
            lag_mean = float(lagged.mean())
            target_mean = float(targets.mean())
            centered = lagged - lag_mean
            denominator = float(np.dot(centered, centered))
            phi = (
                float(np.dot(centered, targets - target_mean) / denominator)
                if denominator > np.finfo(np.float64).eps
                else 0.0
            )
            intercept = target_mean - phi * lag_mean
            estimator_detail = {"intercept": intercept, "phi": phi}
        elif model_name == "one_class_svm":
            estimator = OneClassSVM(kernel="sigmoid", gamma="scale", nu=0.5)
            estimator.fit(np.asarray(features[train_index], dtype=np.float32))
            estimator_detail = {
                "support_vectors": int(estimator.support_.size), "nu": 0.5
            }
        else:
            estimator = SVC(
                C=1.0,
                kernel="sigmoid",
                gamma="scale",
                decision_function_shape="ovr",
            )
            estimator.fit(features[train_index], multiclass_labels[train_index])
            estimator_detail = {
                "classes": estimator.classes_.tolist(),
                "support_vectors": int(estimator.support_.size),
                "C": 1.0,
            }
        fit_seconds = time.perf_counter() - fit_started

        score_started = time.perf_counter()
        if model_name == "arima":
            scores = np.lib.format.open_memmap(
                run / "scores.npy",
                mode="w+",
                dtype="float32",
                shape=(test_index.size,),
            )
            for start in range(0, test_index.size, args.score_batch):
                stop = min(start + args.score_batch, test_index.size)
                batch = np.asarray(test_features[test_index[start:stop]], dtype=np.float32)
                delta = np.diff(batch, axis=1)
                residual = delta[:, 1:] - (intercept + phi * delta[:, :-1])
                scores[start:stop] = np.mean(np.square(residual), axis=1)
            scores.flush()
        elif model_name == "one_class_svm":
            scores = -estimator.decision_function(
                np.asarray(test_features[test_index], dtype=np.float32)
            ).reshape(-1)
            np.save(run / "scores.npy", scores.astype(np.float32))
        else:
            margins = np.asarray(
                estimator.decision_function(features[test_index]), dtype=np.float64
            )
            scores = svm_attack_margin(margins, estimator.classes_)
            np.save(run / "scores.npy", scores.astype(np.float32))
        score_seconds = time.perf_counter() - score_started
        metrics = confusion_metrics(test_labels, scores, threshold=threshold)
        np.save(run / "labels.npy", test_labels)
        np.save(run / "test_global_row.npy", test_index.astype(np.int64))
        np.save(run / "predictions.npy", (scores > threshold).astype(np.int8))
        save_json(run / "model.json", estimator_detail)
        timing = {
            "load": load_seconds,
            "fit": fit_seconds,
            "score_table_3": score_seconds,
            "total": time.perf_counter() - total_started,
        }
        result = {
            "status": "success",
            "eligibility": f"exploratory_interpretation_{prepared_method(metadata)}",
            "configuration": configuration,
            "git_commit": git_commit(),
            "data": {
                "path": str(args.data),
                "dataset": dataset_name,
                "available_train_rows": int(features.shape[0]),
                "train_rows_used": int(train_index.size),
                "available_test_rows": int(
                    test_features.shape[0]
                    if model_name != "multiclass_svm"
                    else test_available.size
                ),
                "test_rows_used": int(test_index.size),
                "source_nodes": metadata.get("source_nodes", {}),
            },
            "model": estimator_detail,
            "metrics": metrics,
            f"reported_table_{table_number}": reported,
            "reported_table_4": None,
            "difference_reproduced_minus_reported": {
                key: float(metrics[key]) - float(value)
                for key, value in reported.items()
            },
            "table_v": None,
            "timing_seconds": timing,
            "artifacts": {
                "scores": "scores.npy",
                "labels": "labels.npy",
                "predictions": "predictions.npy",
                "test_global_row": "test_global_row.npy",
                "model": "model.json",
            },
        }
        save_json(run / "result.json", result)
    except Exception as exc:
        save_json(
            run / "failure.json",
            {
                "status": "failed",
                "configuration": configuration,
                "git_commit": git_commit(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.perf_counter() - total_started,
            },
        )
        raise
    print(json.dumps(result["metrics"], indent=2))
    print(json.dumps(result["timing_seconds"], indent=2))
    print(f"saved immutable attempt: {run}")
    return 0


def run_supervised_neural(
    args: argparse.Namespace,
    *,
    metadata: dict[str, object],
    metadata_path: Path,
) -> int:
    """Run one paper-sized supervised deep benchmark."""

    features, labels = supervised_population(args.data)
    table_number, dataset_name, reported_rows = table_context(metadata)
    reported = reported_rows[args.model]
    split_started = time.perf_counter()
    train_mask = exact_random_train_mask(features.shape[0], seed=args.seed)
    train_index = np.flatnonzero(train_mask)
    test_index = np.flatnonzero(~train_mask)
    split_seconds = time.perf_counter() - split_started
    learning_rate = 0.001 if args.learning_rate is None else args.learning_rate
    configuration = {
        "method": f"{prepared_method(metadata)}-{args.model.upper()}",
        "paper_tables": [str(table_number)],
        "scientific_question": f"Does the frozen {args.model} completion reproduce Table {table_number}?",
        "task": "supervised",
        "model": args.model,
        "seed": args.seed,
        "epochs_max": args.epochs,
        "batch_size": args.batch_size,
        "score_batch": args.score_batch,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "learning_rate": learning_rate,
        "train_fraction": "full",
        "test_view": "supervised_original",
        "table_v": False,
        "threshold": 0.5,
        "anomaly_direction": "higher",
        "supervised_adasyn": metadata.get("configuration", {}).get("supervised_adasyn", "none"),
        "split": "seeded_exact_row_random_2_to_1",
        "head_completion": (
            "softmax2_sparse_categorical"
            if args.model == "supervised_feed_forward"
            else "sigmoid1_binary"
        ),
        "data_metadata_sha256": sha256(metadata_path),
    }
    configuration_id = stable_id({**configuration, "seed": "<seed>"})
    configuration["configuration_id"] = configuration_id
    configuration["attempt_id"] = stable_id(configuration)
    run = (
        args.output
        / f"table_{table_number}"
        / args.model
        / f"seed_{args.seed}_{configuration['attempt_id']}"
    )
    run.mkdir(parents=True, exist_ok=True)
    if (run / "result.json").is_file():
        print(f"immutable attempt already complete: {run}")
        return 0
    save_json(run / "config.json", configuration)
    total_started = time.perf_counter()
    try:
        model = build_model(
            args.model, seed=args.seed, learning_rate=learning_rate
        )
        inventory = layer_inventory(model)
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="loss",
                min_delta=args.min_delta,
                patience=args.patience,
                restore_best_weights=True,
                verbose=1,
            )
        ]
        fit_started = time.perf_counter()
        history = model.fit(
            np.asarray(features[train_index], dtype=np.float32),
            labels[train_index],
            epochs=args.epochs,
            batch_size=args.batch_size,
            shuffle=True,
            callbacks=callbacks,
            verbose=2,
        )
        fit_seconds = time.perf_counter() - fit_started
        model.save_weights(run / "model.weights.h5")
        scores, score_seconds = score_classifier(
            model,
            features,
            test_index,
            run / "scores.npy",
            batch_size=args.score_batch,
            two_class_softmax=args.model == "supervised_feed_forward",
        )
        test_labels = labels[test_index]
        metrics = confusion_metrics(test_labels, scores, threshold=0.5)
        np.save(run / "labels.npy", test_labels)
        np.save(run / "test_global_row.npy", test_index.astype(np.int64))
        np.save(run / "predictions.npy", (scores > 0.5).astype(np.int8))
        history_payload = {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        }
        save_json(run / "history.json", history_payload)
        timing = {
            "split": split_seconds,
            "fit": fit_seconds,
            "score_table_3": score_seconds,
            "total": time.perf_counter() - total_started,
        }
        result = {
            "status": "success",
            "eligibility": f"exploratory_interpretation_{prepared_method(metadata)}",
            "configuration": configuration,
            "git_commit": git_commit(),
            "data": {
                "path": str(args.data),
                "dataset": dataset_name,
                "counts": {
                    "total": int(features.shape[0]),
                    "train": int(train_index.size),
                    "test": int(test_index.size),
                },
                "source_nodes": metadata.get("source_nodes", {}),
            },
            "model": {
                "inventory": inventory,
                "parameters": int(model.count_params()),
            },
            "metrics": metrics,
            f"reported_table_{table_number}": reported,
            "reported_table_4": None,
            "difference_reproduced_minus_reported": {
                key: float(metrics[key]) - float(value)
                for key, value in reported.items()
            },
            "table_v": None,
            "history": history_payload,
            "timing_seconds": timing,
            "artifacts": {
                "scores": "scores.npy",
                "labels": "labels.npy",
                "predictions": "predictions.npy",
                "test_global_row": "test_global_row.npy",
                "weights": "model.weights.h5",
                "history": "history.json",
            },
        }
        save_json(run / "result.json", result)
    except Exception as exc:
        save_json(
            run / "failure.json",
            {
                "status": "failed",
                "configuration": configuration,
                "git_commit": git_commit(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.perf_counter() - total_started,
            },
        )
        raise
    print(json.dumps(result["metrics"], indent=2))
    print(json.dumps(result["timing_seconds"], indent=2))
    print(f"saved immutable attempt: {run}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=(*tuple(SPECS), *BENCHMARKS), default="fc_sae"
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--score-batch", type=int, default=8_192)
    parser.add_argument("--one-class-svm-train-cap", type=int, default=12_000)
    parser.add_argument("--multiclass-svm-train-cap", type=int, default=30_000)
    parser.add_argument("--svm-test-cap", type=int, default=30_000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--output-activation",
        choices=("paper", "linear"),
        default="paper",
        help="paper preserves Table I; linear is the named one-factor control",
    )
    parser.add_argument("--train-fraction", choices=("half", "three_quarter", "full"), default="full")
    parser.add_argument(
        "--test-view",
        choices=("adasyn", "original"),
        default="original",
        help=(
            "adasyn uses the paper-printed resampled test cache; original uses "
            "the exact B2+M population before ADASYN (I-ADASYN/no-resampling)"
        ),
    )
    parser.add_argument("--table-v", action="store_true")
    parser.add_argument(
        "--recover-scoring",
        type=Path,
        help="score preserved weights from a failed run without retraining",
    )
    parser.add_argument("--recovery-score-batch", type=int)
    args = parser.parse_args()

    if min(args.epochs, args.batch_size, args.score_batch) < 1:
        parser.error("epochs and batch sizes must be positive")
    if args.table_v and args.train_fraction != "full":
        parser.error("Table V is evaluated only from the full-training model")
    if args.output_activation != "paper" and args.model != "fc_sae":
        parser.error("output-activation controls are currently FC-SAE only")
    if args.model in BENCHMARKS and args.table_v:
        parser.error("Table V contains only the five proposed models")
    if args.model in BENCHMARKS and args.train_fraction != "full":
        parser.error("Table IV contains only the five proposed models")
    if args.model in ("naive_bayes", "multiclass_svm", *NEURAL_BENCHMARKS) and args.test_view != "original":
        parser.error(
            "supervised benchmarks use the prepared supervised population"
        )

    metadata_path = args.data / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"prepared-data metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") != "complete":
        raise ValueError("prepared data are not complete")
    table_number, dataset_name, reported_rows = table_context(metadata)
    if table_number == 2 and (args.table_v or args.train_fraction != "full"):
        parser.error("Tables IV and V are ISET-only in the paper")
    if metadata.get("configuration", {}).get("mode") == "full" and not os.environ.get(
        "SLURM_JOB_ID"
    ):
        raise RuntimeError("full preparation, training, and scoring must run in Slurm")
    if args.recover_scoring is not None:
        return recover_failed_scoring(
            args.recover_scoring,
            args.data,
            score_batch_override=args.recovery_score_batch,
        )
    if args.test_view == "adasyn" and metadata.get("configuration", {}).get(
        "test_adasyn"
    ) != "printed":
        raise ValueError("the selected cache does not contain printed ADASYN rows")

    if args.model == "naive_bayes":
        return run_naive_bayes(
            args,
            metadata=metadata,
            metadata_path=metadata_path,
        )
    if args.model in ("arima", "one_class_svm", "multiclass_svm"):
        return run_classical_benchmark(
            args,
            metadata=metadata,
            metadata_path=metadata_path,
        )
    if args.model in NEURAL_BENCHMARKS:
        return run_supervised_neural(
            args,
            metadata=metadata,
            metadata_path=metadata_path,
        )

    resolved_learning_rate = (
        0.01
        if args.learning_rate is None and SPECS[args.model].optimizer == "SGD"
        else (0.001 if args.learning_rate is None else args.learning_rate)
    )
    resolved_output_activation = (
        SPECS[args.model].output_activation
        if args.output_activation == "paper"
        else args.output_activation
    )
    reported = reported_rows[args.model]
    if args.output_activation != "paper":
        method = f"C-OUTPUT-{resolved_output_activation.upper()}-{dataset_name}-{SPECS[args.model].name}"
    else:
        method = f"{prepared_method(metadata)}-{SPECS[args.model].name}"
        if args.test_view == "original":
            method += "-NO-TEST-ADASYN"
    paper_tables = (
        [str(table_number)]
        if table_number == 2
        else (["IV"] if args.train_fraction != "full" else ["III", "IV"])
    )
    if args.table_v:
        paper_tables.append("V")
    configuration = {
        "method": method,
        "paper_tables": paper_tables,
        "scientific_question": (
            f"Does the frozen {SPECS[args.model].name} completion reproduce "
            f"the reported Table {table_number} {dataset_name} row?"
        ),
        "model": args.model,
        "seed": args.seed,
        "epochs_max": args.epochs,
        "batch_size": args.batch_size,
        "score_batch": args.score_batch,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "learning_rate": resolved_learning_rate,
        "train_fraction": args.train_fraction,
        "test_view": args.test_view,
        "table_v": args.table_v,
        "threshold": SPECS[args.model].threshold,
        "anomaly_direction": SPECS[args.model].anomaly_direction,
        "data_metadata_sha256": sha256(metadata_path),
    }
    if args.model in {"lstm_sae", "lstm_vae"}:
        configuration["decoder_completion"] = (
            "repeat_latent_with_mirrored_Algorithm_2_or_4_state_transfer"
        )
    if args.model == "lstm_aea":
        configuration["attention_completion"] = (
            "additive_attention_previous_decoder_queries_concat_context_"
            "repeat_latent_mirrored_state_transfer"
        )
    if SPECS[args.model].anomaly_score == "reconstruction_probability":
        configuration["vae_completion"] = {
            "latent_width": SPECS[args.model].encoder[-1],
            "loss": "mean_reconstruction_mse_plus_mean_analytic_kl",
            "score": "exp(-0.5*profile_mse)",
            "variance": "fixed_unit",
            "samples": 1,
        }
    if args.output_activation != "paper":
        configuration["output_activation"] = resolved_output_activation
    configuration_id = stable_id({**configuration, "seed": "<seed>"})
    attempt_id = stable_id(configuration)
    attempt = f"seed_{args.seed}_{attempt_id}"
    if args.table_v:
        attempt += "_table_v"
    run = (
        args.output
        / (
            "table_2"
            if table_number == 2
            else ("table_4" if args.train_fraction != "full" else "table_3")
        )
        / args.model
        / attempt
    )
    run.mkdir(parents=True, exist_ok=True)
    result_path = run / "result.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text())
        if existing.get("status") == "success":
            print(json.dumps(existing["metrics"], indent=2))
            print(f"immutable attempt already complete: {run}")
            return 0
    failure_path = run / "failure.json"
    if failure_path.is_file():
        raise RuntimeError(
            f"immutable attempt already failed: {failure_path}; preserve it and "
            "change an execution setting to create a new attempt"
        )

    total_started = time.perf_counter()
    configuration["configuration_id"] = configuration_id
    configuration["attempt_id"] = attempt_id
    save_json(run / "config.json", configuration)
    try:
        load_started = time.perf_counter()
        x_train_all = np.load(args.data / "x_train.npy", mmap_mode="r")
        order = np.load(args.data / "table_iv_order.npy", mmap_mode="r")
        fractions = {"half": 0.5, "three_quarter": 0.75, "full": 1.0}
        count = int(np.floor(order.size * fractions[args.train_fraction]))
        if args.train_fraction == "full":
            count = order.size
        train_index = np.asarray(order[:count], dtype=np.int64)
        x_train = np.asarray(x_train_all[train_index], dtype=np.float32)
        test_x_name = (
            "x_test.npy" if args.test_view == "adasyn" else "test_original_x.npy"
        )
        test_y_name = (
            "y_test.npy" if args.test_view == "adasyn" else "test_original_y.npy"
        )
        test_provenance_names = (
            ("test_source_row.npy", "test_attack_id.npy", "test_is_synthetic.npy")
            if args.test_view == "adasyn"
            else ("test_original_source_row.npy", "test_original_attack_id.npy")
        )
        x_test = np.load(args.data / test_x_name, mmap_mode="r")
        y_test = np.load(args.data / test_y_name, mmap_mode="r")
        load_seconds = time.perf_counter() - load_started

        build_started = time.perf_counter()
        model = build_model(
            args.model,
            seed=args.seed,
            learning_rate=resolved_learning_rate,
            output_activation=(
                resolved_output_activation
                if args.output_activation != "paper"
                else None
            ),
        )
        inventory = layer_inventory(model)
        build_seconds = time.perf_counter() - build_started
        untrained = score_untrained_sample(
            model,
            x_test,
            y_test,
            threshold=SPECS[args.model].threshold,
            batch_size=args.score_batch,
            score_kind=SPECS[args.model].anomaly_score,
            direction=SPECS[args.model].anomaly_direction,
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="loss",
                min_delta=args.min_delta,
                patience=args.patience,
                restore_best_weights=True,
                verbose=1,
            )
        ]
        fit_started = time.perf_counter()
        history = model.fit(
            x_train,
            x_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            shuffle=True,
            callbacks=callbacks,
            verbose=2,
        )
        fit_seconds = time.perf_counter() - fit_started
        model.save_weights(run / "model.weights.h5")

        scores, score_seconds = score_mse(
            model,
            x_test,
            run / "scores.npy",
            batch_size=args.score_batch,
            score_kind=SPECS[args.model].anomaly_score,
        )
        predictions = (
            scores > SPECS[args.model].threshold
            if SPECS[args.model].anomaly_direction == "higher"
            else scores < SPECS[args.model].threshold
        )
        np.save(run / "predictions.npy", np.asarray(predictions, dtype=np.int8))
        metrics = confusion_metrics(
            y_test,
            scores,
            threshold=SPECS[args.model].threshold,
            direction=SPECS[args.model].anomaly_direction,
        )
        zero_scores = score_zero(
            x_test,
            run / "zero_reconstruction_scores.npy",
            batch_size=args.score_batch,
            score_kind=SPECS[args.model].anomaly_score,
        )
        zero_metrics = confusion_metrics(
            y_test,
            zero_scores,
            threshold=SPECS[args.model].threshold,
            direction=SPECS[args.model].anomaly_direction,
        )
        table_v_rows: list[dict[str, object]] | None = None
        table_v_seconds = 0.0
        if args.table_v:
            table_v_rows, table_v_seconds = table_v(
                model,
                args.data,
                run,
                threshold=SPECS[args.model].threshold,
                score_batch=args.score_batch,
                score_kind=SPECS[args.model].anomaly_score,
                direction=SPECS[args.model].anomaly_direction,
            )
            if args.model == "fc_sae":
                for row in table_v_rows:
                    reported = REPORTED_TABLE_5_FC_SAE[int(row["attack"])]
                    row["reported"] = reported
                    row["difference_reproduced_minus_reported"] = {
                        key: float(row["metrics"][key]) - float(value)
                        for key, value in reported.items()
                    }

        history_payload = {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        }
        save_json(run / "history.json", history_payload)
        timing = {
            "load": load_seconds,
            "model_build": build_seconds,
            "fit": fit_seconds,
            "score_table_3": score_seconds,
            "score_table_5": table_v_seconds,
            "total": time.perf_counter() - total_started,
        }
        result: dict[str, object] = {
            "status": "success",
            "eligibility": (
                "exploratory_control_C-OUTPUT-LINEAR"
                if args.output_activation != "paper"
                else f"exploratory_interpretation_{prepared_method(metadata)}"
            ),
            "configuration": configuration,
            "git_commit": git_commit(),
            "data": {
                "path": str(args.data),
                "dataset": dataset_name,
                "metadata_sha256": (
                    sha256(metadata_path) if metadata_path.is_file() else None
                ),
                "method": prepared_method(metadata),
                "source_nodes": metadata.get("source_nodes", {}),
                "counts": {
                    "B1_profiles": int(x_train_all.shape[0]),
                    "test_profiles": int(x_test.shape[0]),
                    "test_benign": int(np.count_nonzero(y_test == 0)),
                    "test_malicious": int(np.count_nonzero(y_test == 1)),
                },
                "files": {
                    name: sha256(args.data / name)
                    for name in (
                        "x_train.npy",
                        "table_iv_order.npy",
                        "train_meter_ids.npy",
                        "train_day_numbers.npy",
                        test_x_name,
                        test_y_name,
                        *test_provenance_names,
                    )
                },
                "training_rows_used": int(x_train.shape[0]),
            },
            "model": {
                "specification": {
                    **SPECS[args.model].__dict__,
                    "output_activation": resolved_output_activation,
                },
                "inventory": inventory,
                "parameters": int(model.count_params()),
            },
            "metrics": metrics,
            f"reported_table_{table_number}": reported,
            "reported_table_4": (
                REPORTED_TABLE_4.get(args.model, {}).get(args.train_fraction)
                if table_number == 3
                else None
            ),
            "difference_reproduced_minus_reported": (
                {
                    key: float(metrics[key]) - float(reported[key])
                    for key in reported
                }
                if args.train_fraction == "full"
                else {
                    "ACC": float(metrics["ACC"])
                    - float(REPORTED_TABLE_4[args.model][args.train_fraction]["ACC"])
                }
            ),
            "baselines": {
                "untrained_stratified_sample": untrained,
                "zero_reconstruction_full_test": zero_metrics,
            },
            "table_v": table_v_rows,
            "history": history_payload,
            "timing_seconds": timing,
            "runtime": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "keras": keras.__version__,
                "backend": keras.backend.backend(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
                "sklearn": sklearn.__version__,
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            },
            "artifacts": {
                "scores": "scores.npy",
                "predictions": "predictions.npy",
                "weights": "model.weights.h5",
                "history": "history.json",
            },
        }
        save_json(result_path, result)
    except Exception as exc:
        save_json(
            failure_path,
            {
                "status": "failed",
                "configuration": configuration,
                "git_commit": git_commit(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.perf_counter() - total_started,
            },
        )
        raise
    print(json.dumps(result["metrics"], indent=2))
    print(json.dumps(result["timing_seconds"], indent=2))
    print(f"saved immutable attempt: {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
