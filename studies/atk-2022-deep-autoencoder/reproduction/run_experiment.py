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
BENCHMARKS = ("naive_bayes",)


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
) -> tuple[np.memmap, float]:
    started = time.perf_counter()
    scores = np.lib.format.open_memmap(
        target, mode="w+", dtype="float32", shape=(values.shape[0],)
    )
    for start in range(0, values.shape[0], batch_size):
        stop = min(start + batch_size, values.shape[0])
        batch = np.asarray(values[start:stop], dtype=np.float32)
        reconstruction = keras.ops.convert_to_numpy(model(batch, training=False))
        scores[start:stop] = np.mean(np.square(batch - reconstruction), axis=1)
    scores.flush()
    return scores, time.perf_counter() - started


def score_zero(
    values: np.ndarray,
    target: Path,
    *,
    batch_size: int,
) -> np.memmap:
    scores = np.lib.format.open_memmap(
        target, mode="w+", dtype="float32", shape=(values.shape[0],)
    )
    for start in range(0, values.shape[0], batch_size):
        stop = min(start + batch_size, values.shape[0])
        batch = np.asarray(values[start:stop], dtype=np.float32)
        scores[start:stop] = np.mean(np.square(batch), axis=1)
    scores.flush()
    return scores


def score_untrained_sample(
    model: keras.Model,
    values: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float,
) -> dict[str, object]:
    benign = np.flatnonzero(labels == 0)[:5_000]
    malicious = np.flatnonzero(labels == 1)[:5_000]
    index = np.concatenate([benign, malicious])
    batch = np.asarray(values[index], dtype=np.float32)
    reconstruction = keras.ops.convert_to_numpy(model(batch, training=False))
    scores = np.mean(np.square(batch - reconstruction), axis=1)
    return {
        "rows": int(index.size),
        "metrics": confusion_metrics(labels[index], scores, threshold=threshold),
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
) -> tuple[list[dict[str, object]], float]:
    started = time.perf_counter()
    benign = np.load(data / "benign.npy", mmap_mode="r")
    benign_scores, _ = score_mse(
        model, benign, run / "table_v_benign_scores.npy", batch_size=score_batch
    )
    rows: list[dict[str, object]] = []
    for attack_id in range(1, 7):
        attacked = np.load(data / f"attack_{attack_id}.npy", mmap_mode="r")
        attack_scores, _ = score_mse(
            model,
            attacked,
            run / f"table_v_attack_{attack_id}_scores.npy",
            batch_size=score_batch,
        )
        labels = np.concatenate(
            [np.zeros(benign.shape[0], dtype=np.int8), np.ones(attacked.shape[0], dtype=np.int8)]
        )
        scores = np.concatenate([benign_scores, attack_scores])
        rows.append(
            {
                "attack": attack_id,
                "metrics": confusion_metrics(labels, scores, threshold=threshold),
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

    blocks = supervised_source_blocks(args.data)
    arrays = [np.load(path, mmap_mode="r") for path, _ in blocks]
    features = np.concatenate(arrays).astype(np.float32, copy=False)
    labels = np.concatenate(
        [np.full(values.shape[0], label, dtype=np.int8)
         for values, (_, label) in zip(arrays, blocks, strict=True)]
    )
    split_started = time.perf_counter()
    train_mask = exact_random_train_mask(features.shape[0], seed=args.seed)
    train_index = np.flatnonzero(train_mask)
    test_index = np.flatnonzero(~train_mask)
    split_seconds = time.perf_counter() - split_started
    configuration = {
        "method": "I-SUPERVISED-ADASYN-NONE-ISET-NAIVE-BAYES",
        "paper_tables": ["III"],
        "scientific_question": "Does the Gaussian-NB completion reproduce Table III without the printed full-scale ADASYN step?",
        "task": "supervised",
        "model": "naive_bayes",
        "seed": args.seed,
        "train_fraction": "full",
        "test_view": "supervised_original",
        "table_v": False,
        "threshold": 0.5,
        "anomaly_direction": "higher",
        "supervised_adasyn": "none",
        "split": "seeded_exact_row_random_2_to_1",
        "data_metadata_sha256": sha256(metadata_path),
    }
    configuration_id = stable_id({**configuration, "seed": "<seed>"})
    attempt_id = stable_id(configuration)
    configuration["configuration_id"] = configuration_id
    configuration["attempt_id"] = attempt_id
    run = args.output / "table_3" / "naive_bayes" / f"seed_{args.seed}_{attempt_id}"
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
            "eligibility": "exploratory_interpretation_I-SUPERVISED-ADASYN-NONE",
            "configuration": configuration,
            "git_commit": git_commit(),
            "data": {
                "path": str(args.data),
                "metadata_sha256": sha256(metadata_path),
                "population": "all benign plus all six attacks for all customers",
                "counts": {
                    "total": int(features.shape[0]),
                    "train": int(train_index.size),
                    "test": int(test_index.size),
                    "train_by_class": np.bincount(labels[train_index], minlength=2).tolist(),
                    "test_by_class": np.bincount(labels[test_index], minlength=2).tolist(),
                },
                "files": {
                    path.name: source_file_records.get(path.name)
                    for path, _ in blocks
                },
                "source_nodes": {
                    "supervised_adasyn": {
                        "paper_claim": "apply ADASYN to B+M before the 2:1 split",
                        "literal_status": "not_executed_in_this_interpretation",
                        "assumption": "split and evaluate the preserved original B+M rows",
                    }
                },
            },
            "model": {
                "name": "Gaussian Naive Bayes",
                "paper_detail": "the paper names Naive Bayes but gives no variant or hyperparameters",
                "completion": model_payload,
            },
            "metrics": metrics,
            "reported_table_3": REPORTED["naive_bayes"],
            "reported_table_4": None,
            "difference_reproduced_minus_reported": {
                key: float(metrics[key]) - float(value)
                for key, value in REPORTED["naive_bayes"].items()
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
    if args.model in BENCHMARKS and args.test_view != "original":
        parser.error(
            "the first benchmark breadth row is the explicit no-ADASYN continuation"
        )

    metadata_path = args.data / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"prepared-data metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") != "complete":
        raise ValueError("prepared data are not complete")
    if metadata.get("configuration", {}).get("mode") == "full" and not os.environ.get(
        "SLURM_JOB_ID"
    ):
        raise RuntimeError("full preparation, training, and scoring must run in Slurm")
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

    resolved_learning_rate = 0.001 if args.learning_rate is None else args.learning_rate
    resolved_output_activation = (
        SPECS[args.model].output_activation
        if args.output_activation == "paper"
        else args.output_activation
    )
    if args.output_activation != "paper":
        method = f"C-OUTPUT-{resolved_output_activation.upper()}-ISET-{SPECS[args.model].name}"
    else:
        method = (
            f"P0-ISET-{SPECS[args.model].name}"
            if args.test_view == "adasyn"
            else f"I-ADASYN-NONE-ISET-{SPECS[args.model].name}"
        )
    paper_tables = ["IV"] if args.train_fraction != "full" else ["III", "IV"]
    if args.table_v:
        paper_tables.append("V")
    configuration = {
        "method": method,
        "paper_tables": paper_tables,
        "scientific_question": (
            f"Does the frozen {SPECS[args.model].name} completion reproduce "
            "the reported ISET row?"
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
    if args.output_activation != "paper":
        configuration["output_activation"] = resolved_output_activation
    configuration_id = stable_id({**configuration, "seed": "<seed>"})
    attempt_id = stable_id(configuration)
    attempt = f"seed_{args.seed}_{attempt_id}"
    if args.table_v:
        attempt += "_table_v"
    run = (
        args.output
        / ("table_4" if args.train_fraction != "full" else "table_3")
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
            model, x_test, y_test, threshold=SPECS[args.model].threshold
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
            model, x_test, run / "scores.npy", batch_size=args.score_batch
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
            x_test, run / "zero_reconstruction_scores.npy", batch_size=args.score_batch
        )
        zero_metrics = confusion_metrics(
            y_test, zero_scores, threshold=SPECS[args.model].threshold
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
                else (
                    "exploratory_paper_primary_P0"
                    if args.test_view == "adasyn"
                    else "exploratory_interpretation_I-ADASYN-NONE"
                )
            ),
            "configuration": configuration,
            "git_commit": git_commit(),
            "data": {
                "path": str(args.data),
                "metadata_sha256": (
                    sha256(metadata_path) if metadata_path.is_file() else None
                ),
                "method": metadata["method"],
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
            "reported_table_3": REPORTED[args.model],
            "reported_table_4": (
                REPORTED_TABLE_4.get(args.model, {}).get(args.train_fraction)
            ),
            "difference_reproduced_minus_reported": (
                {
                    key: float(metrics[key]) - float(REPORTED[args.model][key])
                    for key in REPORTED[args.model]
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
