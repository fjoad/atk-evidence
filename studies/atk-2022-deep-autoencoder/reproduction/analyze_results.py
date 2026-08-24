#!/usr/bin/env python3
"""Aggregate every compact-route attempt without selecting a favorite seed."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


REPO = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = (
    REPO
    / "data/derived/atk-2022-deep-autoencoder/reproduction/results/runs"
)
DEFAULT_OUTPUT = (
    REPO
    / "data/derived/atk-2022-deep-autoencoder/reproduction/results/aggregate"
)
METRICS = ("DR", "FA", "SP", "PR", "ACC", "F1", "AUC")
QUANTILES = (0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
TABLE_2_NAMES = {
    "fc_sae": "FC-SAE",
    "lstm_sae": "LSTM-SAE",
    "fc_vae": "FC-VAE",
    "lstm_vae": "LSTM-VAE",
    "lstm_aea": "LSTM-AEA",
    "naive_bayes": "Naive Bayes",
    "arima": "ARIMA",
    "one_class_svm": "Single-class SVM",
    "supervised_feed_forward": "Feed forward",
    "supervised_lstm": "LSTM",
    "multiclass_svm": "Multi-class SVM",
}


def canonical_reported(attempt: dict[str, object]) -> dict[str, float] | None:
    """Use the source-transcribed CSV rather than mutable runner constants."""

    if attempt.get("reported_table_2") is None:
        return attempt.get("reported_table_3")
    target_name = TABLE_2_NAMES[str(attempt["configuration"]["model"])]
    path = REPO / "studies/atk-2022-deep-autoencoder/reported/table_2.csv"
    with path.open(newline="") as handle:
        row = next(item for item in csv.DictReader(handle) if item["model"] == target_name)
    return {metric: float(row[metric]) for metric in METRICS}


def strict_runtime_threshold(
    oriented_threshold: float, scores: np.ndarray, *, direction: str
) -> float:
    """Translate sklearn's inclusive ROC boundary to this runner's strict rule."""

    dtype = np.asarray(scores).dtype
    if direction == "higher":
        value = np.asarray(oriented_threshold, dtype=dtype)
        return float(np.nextafter(value, np.asarray(-np.inf, dtype=dtype)))
    value = np.asarray(-oriented_threshold, dtype=dtype)
    return float(np.nextafter(value, np.asarray(np.inf, dtype=dtype)))


def effective_eligibility(attempt: dict[str, object]) -> str:
    config = attempt["configuration"]
    if config.get("contract") == "clean-reader-v1":
        return "eligible_clean_reader_P+I_N"
    if "I-SGCC-" in str(config.get("method", "")):
        recorded = str(attempt.get("eligibility", ""))
        return (
            recorded
            if recorded.startswith("exploratory_interpretation_I-SGCC-")
            else "exploratory_interpretation_I-SGCC"
        )
    if config.get("output_activation") == "linear":
        return "exploratory_control_C-OUTPUT-LINEAR"
    if config.get("task") == "supervised":
        return (
            "exploratory_paper_primary_supervised"
            if str(config.get("supervised_adasyn", "")).startswith("printed")
            else "exploratory_interpretation_I-SUPERVISED-ADASYN-NONE"
        )
    return (
        "exploratory_paper_primary_P0"
        if config["test_view"] == "adasyn"
        else "exploratory_interpretation_I-ADASYN-NONE"
    )


def analysis_group(config: dict[str, object]) -> str:
    if config.get("configuration_id"):
        return str(config["configuration_id"])
    fields = {
        key: config.get(key)
        for key in (
            "method",
            "task",
            "model",
            "epochs_max",
            "batch_size",
            "patience",
            "min_delta",
            "learning_rate",
            "train_fraction",
            "test_view",
            "table_v",
            "threshold",
            "supervised_adasyn",
            "split",
        )
    }
    return json.dumps(fields, sort_keys=True, separators=(",", ":"))


def array_summary(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values)
    quantiles = np.quantile(values, QUANTILES)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values, dtype=np.float64)),
        "standard_deviation": float(np.std(values, dtype=np.float64)),
        "quantiles": {
            str(quantile): float(value)
            for quantile, value in zip(QUANTILES, quantiles, strict=True)
        },
    }


def chunked_pair_summary(
    left: np.ndarray,
    right: np.ndarray,
    *,
    chunk_size: int = 1_000_000,
) -> dict[str, float]:
    if left.shape != right.shape:
        raise ValueError(f"score shape mismatch: {left.shape} != {right.shape}")
    count = 0
    sum_left = sum_right = sum_left_sq = sum_right_sq = sum_cross = 0.0
    sum_abs_delta = sum_delta_sq = 0.0
    for start in range(0, left.size, chunk_size):
        stop = min(start + chunk_size, left.size)
        x = np.asarray(left[start:stop], dtype=np.float64)
        y = np.asarray(right[start:stop], dtype=np.float64)
        delta = x - y
        count += x.size
        sum_left += float(np.sum(x))
        sum_right += float(np.sum(y))
        sum_left_sq += float(np.dot(x, x))
        sum_right_sq += float(np.dot(y, y))
        sum_cross += float(np.dot(x, y))
        sum_abs_delta += float(np.sum(np.abs(delta)))
        sum_delta_sq += float(np.dot(delta, delta))
    covariance = sum_cross - sum_left * sum_right / count
    left_ss = sum_left_sq - sum_left * sum_left / count
    right_ss = sum_right_sq - sum_right * sum_right / count
    denominator = math.sqrt(max(left_ss, 0) * max(right_ss, 0))
    correlation = covariance / denominator if denominator else float("nan")
    return {
        "pearson_correlation": correlation,
        "mean_absolute_difference": sum_abs_delta / count,
        "root_mean_squared_difference": math.sqrt(sum_delta_sq / count),
    }


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def metric_vector(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    direction: str,
) -> tuple[dict[str, float | int], np.ndarray]:
    """Independently regenerate the runner's printed metric vector."""

    predictions = scores > threshold if direction == "higher" else scores < threshold
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=bool)
    tp = int(np.count_nonzero(predictions & (labels == 1)))
    tn = int(np.count_nonzero(~predictions & (labels == 0)))
    fp = int(np.count_nonzero(predictions & (labels == 0)))
    fn = int(np.count_nonzero(~predictions & (labels == 1)))
    dr = tp / (tp + fn)
    fa = fp / (fp + tn)
    sp = 1 - fa
    precision = tp / (tp + fp) if tp + fp else 0.0
    accuracy = (dr + sp) / 2
    f1 = 2 * dr * precision / (dr + precision) if dr + precision else 0.0
    oriented = scores if direction == "higher" else -scores
    metrics: dict[str, float | int] = {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "DR": 100 * dr,
        "FA": 100 * fa,
        "SP": 100 * sp,
        "PR": 100 * precision,
        "ACC": 100 * accuracy,
        "F1": 100 * f1,
        "AUC": 100 * float(roc_auc_score(labels, oriented)),
    }
    return metrics, predictions.astype(np.int8)


def best_balanced_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    direction: str,
) -> dict[str, float]:
    oriented = scores if direction == "higher" else -scores
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels, oriented, drop_intermediate=True
    )
    index = int(np.argmax(true_positive_rate - false_positive_rate))
    return {
        "direction": direction,
        "threshold": strict_runtime_threshold(
            float(thresholds[index]), scores, direction=direction
        ),
        "comparison": ">" if direction == "higher" else "<",
        "DR": 100 * float(true_positive_rate[index]),
        "FA": 100 * float(false_positive_rate[index]),
        "ACC": 50
        * float(true_positive_rate[index] + 1 - false_positive_rate[index]),
        "AUC": 100 * float(roc_auc_score(labels, oriented)),
    }


def closest_reported_operating_point(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    direction: str,
    reported_dr: float,
    reported_fa: float,
) -> dict[str, float]:
    """Return the threshold whose ROC point is closest to printed DR and FA."""

    oriented = scores if direction == "higher" else -scores
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels, oriented, drop_intermediate=True
    )
    dr = 100 * true_positive_rate
    fa = 100 * false_positive_rate
    distance = np.maximum(np.abs(dr - reported_dr), np.abs(fa - reported_fa))
    index = int(np.argmin(distance))
    return {
        "direction": direction,
        "threshold": strict_runtime_threshold(
            float(thresholds[index]), scores, direction=direction
        ),
        "comparison": ">" if direction == "higher" else "<",
        "DR": float(dr[index]),
        "FA": float(fa[index]),
        "ACC": 50 * float(true_positive_rate[index] + 1 - false_positive_rate[index]),
        "maximum_absolute_DR_FA_gap": float(distance[index]),
    }


def closest_reported_metric_vector(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    direction: str,
    reported: dict[str, float],
) -> dict[str, object]:
    """Exactly minimize the seven-metric gap over every score threshold.

    A deterministic threshold can only change predictions when it crosses a
    distinct saved score. ``roc_curve`` enumerates all such confusion matrices;
    AUC is fixed by the ranking and is included in every candidate's gap.
    """

    oriented = scores if direction == "higher" else -scores
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels, oriented, drop_intermediate=False
    )
    positives = int(np.count_nonzero(labels))
    negatives = int(labels.size - positives)
    tp = true_positive_rate * positives
    fp = false_positive_rate * negatives
    precision = np.divide(
        tp,
        tp + fp,
        out=np.full_like(tp, np.nan, dtype=np.float64),
        where=(tp + fp) > 0,
    )
    f1 = np.divide(
        2 * precision * true_positive_rate,
        precision + true_positive_rate,
        out=np.full_like(tp, np.nan, dtype=np.float64),
        where=(precision + true_positive_rate) > 0,
    )
    auc = 100 * float(roc_auc_score(labels, oriented))
    candidates = {
        "DR": 100 * true_positive_rate,
        "FA": 100 * false_positive_rate,
        "SP": 100 * (1 - false_positive_rate),
        "PR": 100 * precision,
        "ACC": 50 * (true_positive_rate + 1 - false_positive_rate),
        "F1": 100 * f1,
        "AUC": np.full_like(true_positive_rate, auc, dtype=np.float64),
    }
    gaps = np.vstack(
        [np.abs(candidates[metric] - float(reported[metric])) for metric in METRICS]
    )
    maximum_gap = np.max(gaps, axis=0)
    maximum_gap[~np.isfinite(maximum_gap)] = np.inf
    index = int(np.argmin(maximum_gap))
    metrics = {metric: float(values[index]) for metric, values in candidates.items()}
    return {
        "scope": "all deterministic thresholds over this saved score vector",
        "direction": direction,
        "threshold": strict_runtime_threshold(
            float(thresholds[index]), scores, direction=direction
        ),
        "comparison": ">" if direction == "higher" else "<",
        "metrics": metrics,
        "absolute_gap_by_metric": {
            metric: abs(metrics[metric] - float(reported[metric]))
            for metric in METRICS
        },
        "minimum_maximum_absolute_gap": float(maximum_gap[index]),
        "threshold_candidates": int(thresholds.size),
    }


def audit_attempt_arrays(
    run: Path, data: Path, config: dict[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Load aligned labels/attacks/scores for anomaly and supervised attempts."""

    scores = np.load(run / "scores.npy", mmap_mode="r")
    zero_path = run / "zero_reconstruction_scores.npy"
    zero_scores = np.load(zero_path, mmap_mode="r") if zero_path.is_file() else None
    saved_labels = run / "labels.npy"
    if saved_labels.is_file():
        labels = np.load(saved_labels, mmap_mode="r")
        indices = np.load(run / "test_global_row.npy", mmap_mode="r")
        if config.get("task") == "supervised":
            benign_path = data / "benign.npy"
            if benign_path.is_file():
                block_size = int(np.load(benign_path, mmap_mode="r").shape[0])
                attack_ids = np.asarray(indices // block_size, dtype=np.uint8)
            else:
                # SGCC has binary source labels, not the six synthetic ISET attacks.
                attack_ids = np.zeros(indices.size, dtype=np.uint8)
        else:
            attack_name = (
                "test_attack_id.npy"
                if config.get("test_view") == "adasyn"
                else "test_original_attack_id.npy"
            )
            full_attacks = np.load(
                data / attack_name, mmap_mode="r"
            )
            attack_ids = np.asarray(full_attacks[indices], dtype=np.uint8)
    else:
        test_view = str(config["test_view"])
        labels_name = "y_test.npy" if test_view == "adasyn" else "test_original_y.npy"
        attack_name = (
            "test_attack_id.npy"
            if test_view == "adasyn"
            else "test_original_attack_id.npy"
        )
        labels = np.load(data / labels_name, mmap_mode="r")
        attack_ids = np.load(data / attack_name, mmap_mode="r")
    expected = (labels.shape, attack_ids.shape, scores.shape)
    if len(set(expected)) != 1 or (
        zero_scores is not None and zero_scores.shape != scores.shape
    ):
        raise ValueError(f"audit arrays must align, observed shapes: {expected}")
    return labels, attack_ids, scores, zero_scores


def audit_scores(attempt_path: Path) -> dict[str, object]:
    attempt_path = attempt_path.resolve()
    attempt = json.loads(attempt_path.read_text())
    run = attempt_path.parent
    config = attempt["configuration"]
    data = Path(attempt["data"]["path"])
    if not data.is_absolute():
        data = REPO / data
    test_view = str(config["test_view"])
    labels, attack_ids, scores, zero_scores = audit_attempt_arrays(
        run, data, config
    )

    benign = np.asarray(scores[labels == 0])
    malicious = np.asarray(scores[labels == 1])
    threshold = float(config["threshold"])
    direction = str(config.get("anomaly_direction", "higher"))
    predicted_benign = benign > threshold if direction == "higher" else benign < threshold
    false_alarm = 100 * float(np.mean(predicted_benign))
    by_attack: list[dict[str, object]] = []
    attack_range = range(1, 7) if np.any(attack_ids > 0) else ()
    for attack_id in attack_range:
        attack_scores = np.asarray(scores[attack_ids == attack_id])
        predicted_attack = (
            attack_scores > threshold
            if direction == "higher"
            else attack_scores < threshold
        )
        by_attack.append(
            {
                "attack": attack_id,
                "profiles": int(attack_scores.size),
                "DR": 100 * float(np.mean(predicted_attack)),
                "FA_on_common_B2": false_alarm,
                "score": array_summary(attack_scores),
            }
        )

    result = {
        "status": "success",
        "source_result": str(attempt_path),
        "method": config["method"],
        "effective_eligibility": effective_eligibility(attempt),
        "recorded_eligibility": attempt["eligibility"],
        "test_view": test_view,
        "threshold": threshold,
        "anomaly_direction": direction,
        "trained_score": {
            "all": array_summary(scores),
            "benign": array_summary(benign),
            "malicious": array_summary(malicious),
        },
        "trained_vs_zero_reconstruction": (
            chunked_pair_summary(scores, zero_scores)
            if zero_scores is not None
            else None
        ),
        "reported_direction": best_balanced_threshold(
            labels, scores, direction=direction
        ),
        "reversed_direction_control": best_balanced_threshold(
            labels,
            scores,
            direction="lower" if direction == "higher" else "higher",
        ),
        "table_v_heldout_benign_interpretation": by_attack,
    }
    reported = canonical_reported(attempt)
    if reported:
        result["stored_reported_target_matches_canonical"] = (
            attempt.get("reported_table_2") in (None, reported)
        )
        result["closest_reported_operating_point"] = closest_reported_operating_point(
            labels,
            scores,
            direction=direction,
            reported_dr=float(reported["DR"]),
            reported_fa=float(reported["FA"]),
        )
        result["closest_reported_complete_metric_vector"] = (
            closest_reported_metric_vector(
                labels,
                scores,
                direction=direction,
                reported={metric: float(reported[metric]) for metric in METRICS},
            )
        )
    return result


def audit_clean_reader_anchor(attempt_path: Path) -> dict[str, object]:
    """Fail closed unless one result is a complete `CR-ISET-FCSAE-01` artifact."""

    attempt_path = attempt_path.resolve()
    attempt = json.loads(attempt_path.read_text())
    run = attempt_path.parent
    config = attempt.get("configuration", {})
    errors: list[str] = []
    expected_config = {
        "contract": "clean-reader-v1",
        "anchor_id": "CR-ISET-FCSAE-01",
        "method": "CR-ISET-FCSAE-01",
        "model": "fc_sae",
        "seed": 20260824,
        "epochs_max": 100,
        "minimum_epochs": 10,
        "batch_size": 32,
        "patience": 5,
        "min_delta": 1e-6,
        "learning_rate": 0.001,
        "train_fraction": "full",
        "test_view": "adasyn",
        "table_v": False,
        "threshold": 0.58,
        "anomaly_direction": "higher",
        "attempt_rule": "one_attempt_then_stop_for_checkpoint_2",
    }
    for name, expected in expected_config.items():
        if config.get(name) != expected:
            errors.append(
                f"configuration {name}: expected {expected!r}, observed {config.get(name)!r}"
            )
    if attempt.get("eligibility") != "eligible_clean_reader_P+I_N":
        errors.append("result eligibility is not eligible_clean_reader_P+I_N")

    data_path = Path(attempt.get("data", {}).get("path", ""))
    if not data_path.is_absolute():
        data_path = REPO / data_path
    metadata_path = data_path / "metadata.json"
    if not metadata_path.is_file():
        errors.append(f"prepared metadata is missing: {metadata_path}")
        metadata: dict[str, object] = {}
    else:
        metadata = json.loads(metadata_path.read_text())
        if sha256(metadata_path) != config.get("data_metadata_sha256"):
            errors.append("prepared metadata SHA-256 differs from the frozen configuration")
    expected_data = {
        "contract": "clean-reader-v1",
        "mode": "full",
        "seed": 20260824,
        "test_adasyn": "printed",
        "adasyn_neighbors": 5,
        "source_branch": "official-tab-v1",
        "attack_3_completion": "duration_first_in_day",
        "malicious_test_population": "b2",
        "expensive_adasyn_acknowledged": True,
    }
    data_config = metadata.get("configuration", {})
    for name, expected in expected_data.items():
        if data_config.get(name) != expected:
            errors.append(
                f"prepared data {name}: expected {expected!r}, observed {data_config.get(name)!r}"
            )
    source = metadata.get("source", {})
    if source.get("branch") != "official-tab-v1" or not source.get("ready"):
        errors.append("official-tab-v1 source gate is not recorded ready")
    for name, record in source.get("files", {}).items():
        if record.get("status") != "verified":
            errors.append(f"source file is not verified: {name}")

    for name, expected in attempt.get("data", {}).get("files", {}).items():
        path = data_path / name
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"prepared artifact hash mismatch: {name}")
    for name, expected in attempt.get("artifact_sha256", {}).items():
        path = run / name
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"run artifact hash mismatch: {name}")

    required_run_files = {
        "scores.npy",
        "predictions.npy",
        "model.weights.h5",
        "history.json",
        "zero_reconstruction_scores.npy",
        "softmax_projection_floor_scores.npy",
    }
    for name in sorted(required_run_files):
        if not (run / name).is_file():
            errors.append(f"required run artifact is missing: {name}")

    recomputed: dict[str, float | int] | None = None
    maximum_metric_gap: float | None = None
    floor_violation: float | None = None
    if (run / "scores.npy").is_file() and (data_path / "y_test.npy").is_file():
        scores = np.load(run / "scores.npy", mmap_mode="r")
        labels = np.load(data_path / "y_test.npy", mmap_mode="r")
        recomputed, predictions = metric_vector(
            labels,
            scores,
            threshold=float(config.get("threshold", 0.58)),
            direction=str(config.get("anomaly_direction", "higher")),
        )
        recorded = attempt.get("metrics", {})
        maximum_metric_gap = max(
            abs(float(recomputed[name]) - float(recorded.get(name, float("nan"))))
            for name in (*METRICS, "TP", "TN", "FP", "FN")
        )
        if not np.isfinite(maximum_metric_gap) or maximum_metric_gap > 1e-10:
            errors.append(
                f"independently regenerated metrics differ by {maximum_metric_gap}"
            )
        saved_predictions = np.load(run / "predictions.npy", mmap_mode="r")
        if not np.array_equal(predictions, saved_predictions):
            errors.append("saved predictions differ from score-threshold regeneration")
        floor_path = run / "softmax_projection_floor_scores.npy"
        if floor_path.is_file():
            floor = np.load(floor_path, mmap_mode="r")
            if floor.shape != scores.shape:
                errors.append("Softmax projection-floor scores do not align")
            else:
                floor_violation = float(np.max(np.asarray(floor) - np.asarray(scores)))
                if floor_violation > 1e-5:
                    errors.append(
                        "trained reconstruction score falls below the exact "
                        f"Softmax-domain floor by {floor_violation}"
                    )

    history_path = run / "history.json"
    if history_path.is_file():
        history = json.loads(history_path.read_text())
        completed = len(history.get("loss", []))
        stop = attempt.get("training_stop", {})
        if completed != stop.get("epochs_completed"):
            errors.append("history length differs from the recorded stopping state")
        if completed < 10 or completed > 100:
            errors.append(f"completed epoch count is outside 10..100: {completed}")

    return {
        "status": "passed" if not errors else "failed",
        "anchor_id": "CR-ISET-FCSAE-01",
        "source_result": str(attempt_path),
        "errors": errors,
        "independently_recomputed_metrics": recomputed,
        "maximum_recorded_metric_gap": maximum_metric_gap,
        "maximum_floor_minus_trained_score": floor_violation,
        "score_audit": audit_scores(attempt_path) if not errors else None,
    }


def load_attempts(root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    successes: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for path in sorted(root.rglob("result.json")):
        payload = json.loads(path.read_text())
        payload["_path"] = str(path)
        if payload.get("status") == "success":
            successes.append(payload)
    for path in sorted(root.rglob("failure.json")):
        payload = json.loads(path.read_text())
        payload["_path"] = str(path)
        failures.append(payload)
    return successes, failures


def mean_sd(values: list[float]) -> tuple[float, float]:
    return (
        statistics.fmean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(successes: list[dict[str, object]]) -> dict[str, object]:
    individual: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for attempt in successes:
        config = attempt["configuration"]
        eligibility = effective_eligibility(attempt)
        row = {
            "configuration_id": analysis_group(config),
            "method": config["method"],
            "eligibility": eligibility,
            "recorded_eligibility": attempt["eligibility"],
            "eligibility_corrected": eligibility != attempt["eligibility"],
            "model": config["model"],
            "seed": config["seed"],
            "train_fraction": config["train_fraction"],
            "test_view": config["test_view"],
            "table_v": config["table_v"],
            **{metric: attempt["metrics"][metric] for metric in METRICS},
            "fit_seconds": attempt["timing_seconds"]["fit"],
            "score_seconds": attempt["timing_seconds"]["score_table_3"],
            "total_seconds": attempt["timing_seconds"]["total"],
            "path": attempt["_path"],
        }
        individual.append(row)
        grouped[analysis_group(config)].append(attempt)

    table_2_summary: list[dict[str, object]] = []
    table_3_summary: list[dict[str, object]] = []
    table_4_summary: list[dict[str, object]] = []
    for group_id, attempts in sorted(grouped.items()):
        first = attempts[0]
        config = first["configuration"]
        base: dict[str, object] = {
            "configuration_id": group_id,
            "method": config["method"],
            "eligibility": effective_eligibility(attempts[0]),
            "model": config["model"],
            "train_fraction": config["train_fraction"],
            "test_view": config["test_view"],
            "batch_size": config.get("batch_size"),
            "epochs_max": config.get("epochs_max"),
            "successful_seeds": len(attempts),
            "seeds": ";".join(
                str(item["configuration"]["seed"])
                for item in sorted(
                    attempts, key=lambda item: int(item["configuration"]["seed"])
                )
            ),
        }
        if config["train_fraction"] == "full":
            reported_2 = first.get("reported_table_2")
            reported = canonical_reported(first)
            assert reported is not None
            row = dict(base)
            for metric in METRICS:
                values = [float(item["metrics"][metric]) for item in attempts]
                mean, sd = mean_sd(values)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_sd"] = sd
                row[f"{metric}_reported"] = reported[metric]
                row[f"{metric}_difference"] = mean - float(reported[metric])
            (table_2_summary if reported_2 else table_3_summary).append(row)

        reported_4 = first.get("reported_table_4")
        if reported_4 and first.get("reported_table_2") is None:
            row = dict(base)
            acc_mean, acc_sd = mean_sd(
                [float(item["metrics"]["ACC"]) for item in attempts]
            )
            fit_mean, fit_sd = mean_sd(
                [float(item["timing_seconds"]["fit"]) / 60 for item in attempts]
            )
            row.update(
                {
                    "ACC_mean": acc_mean,
                    "ACC_sd": acc_sd,
                    "ACC_reported": reported_4["ACC"],
                    "ACC_difference": acc_mean - float(reported_4["ACC"]),
                    "fit_minutes_mean": fit_mean,
                    "fit_minutes_sd": fit_sd,
                    "training_minutes_reported": reported_4["training_minutes"],
                    "training_minutes_difference": fit_mean
                    - float(reported_4["training_minutes"]),
                }
            )
            table_4_summary.append(row)

    table_v_rows: list[dict[str, object]] = []
    for attempt in successes:
        if attempt.get("table_v") is None:
            continue
        config = attempt["configuration"]
        for entry in attempt["table_v"]:
            row = {
                "configuration_id": analysis_group(config),
                "method": config["method"],
                "eligibility": effective_eligibility(attempt),
                "recorded_eligibility": attempt["eligibility"],
                "model": config["model"],
                "seed": config["seed"],
                "attack": entry["attack"],
                "DR": entry["metrics"]["DR"],
                "FA": entry["metrics"]["FA"],
                "ACC": entry["metrics"]["ACC"],
                "AUC": entry["metrics"]["AUC"],
            }
            for metric, value in entry.get("reported", {}).items():
                row[f"{metric}_reported"] = value
                row[f"{metric}_difference"] = float(entry["metrics"][metric]) - float(value)
            table_v_rows.append(row)
    return {
        "individual": individual,
        "table_2_summary": table_2_summary,
        "table_3_summary": table_3_summary,
        "table_4_summary": table_4_summary,
        "table_5_individual": table_v_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--audit-attempt",
        type=Path,
        help="audit one result.json and its full score arrays",
    )
    parser.add_argument(
        "--audit-clean-reader-anchor",
        type=Path,
        help="fail-closed audit of one CR-ISET-FCSAE-01 result.json",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        help="score-audit JSON path (defaults beside the audited result)",
    )
    args = parser.parse_args()
    if args.audit_attempt is not None and args.audit_clean_reader_anchor is not None:
        parser.error("select only one audit mode")
    if args.audit_clean_reader_anchor is not None:
        payload = audit_clean_reader_anchor(args.audit_clean_reader_anchor)
        output = (
            args.audit_output
            or args.audit_clean_reader_anchor.parent / "clean_reader_anchor_audit.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "passed" else 1
    if args.audit_attempt is not None:
        payload = audit_scores(args.audit_attempt)
        output = args.audit_output or args.audit_attempt.parent / "score_audit.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    successes, failures = load_attempts(args.results)
    tables = aggregate(successes)
    write_csv(args.output / "attempts_individual.csv", tables["individual"])
    write_csv(args.output / "table_2_summary.csv", tables["table_2_summary"])
    write_csv(args.output / "table_3_summary.csv", tables["table_3_summary"])
    write_csv(args.output / "table_4_summary.csv", tables["table_4_summary"])
    write_csv(args.output / "table_5_individual.csv", tables["table_5_individual"])
    payload = {
        "successful_attempts": len(successes),
        "failed_attempts": len(failures),
        "failures": failures,
        **tables,
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"aggregated {len(successes)} successes and {len(failures)} failures "
        f"into {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
