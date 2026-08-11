#!/usr/bin/env python3
"""Aggregate every compact-route attempt without selecting a favorite seed."""

from __future__ import annotations

import argparse
import csv
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


def effective_eligibility(attempt: dict[str, object]) -> str:
    config = attempt["configuration"]
    if config.get("output_activation") == "linear":
        return "exploratory_control_C-OUTPUT-LINEAR"
    if config.get("task") == "supervised":
        return (
            "exploratory_paper_primary_supervised"
            if config.get("supervised_adasyn") == "printed"
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
    oriented_threshold = float(thresholds[index])
    return {
        "direction": direction,
        "threshold": (
            oriented_threshold if direction == "higher" else -oriented_threshold
        ),
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
    oriented_threshold = float(thresholds[index])
    return {
        "direction": direction,
        "threshold": (
            oriented_threshold if direction == "higher" else -oriented_threshold
        ),
        "DR": float(dr[index]),
        "FA": float(fa[index]),
        "ACC": 50 * float(true_positive_rate[index] + 1 - false_positive_rate[index]),
        "maximum_absolute_DR_FA_gap": float(distance[index]),
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
            block_size = int(np.load(data / "benign.npy", mmap_mode="r").shape[0])
            attack_ids = np.asarray(indices // block_size, dtype=np.uint8)
        else:
            full_attacks = np.load(
                data / "test_original_attack_id.npy", mmap_mode="r"
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
    for attack_id in range(1, 7):
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
    reported = attempt.get("reported_table_3")
    if reported:
        result["closest_reported_operating_point"] = closest_reported_operating_point(
            labels,
            scores,
            direction=direction,
            reported_dr=float(reported["DR"]),
            reported_fa=float(reported["FA"]),
        )
    return result


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
            reported = first["reported_table_3"]
            row = dict(base)
            for metric in METRICS:
                values = [float(item["metrics"][metric]) for item in attempts]
                mean, sd = mean_sd(values)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_sd"] = sd
                row[f"{metric}_reported"] = reported[metric]
                row[f"{metric}_difference"] = mean - float(reported[metric])
            table_3_summary.append(row)

        reported_4 = first.get("reported_table_4")
        if reported_4:
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
        "--audit-output",
        type=Path,
        help="score-audit JSON path (defaults beside the audited result)",
    )
    args = parser.parse_args()
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
