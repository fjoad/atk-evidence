#!/usr/bin/env python3
"""Verify completed run artifacts and diagnose their score separation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, roc_curve


STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[1]
DEFAULT_RUNS = REPO / "data/derived/atk-2022-deep-autoencoder/runs/table_2/sgcc"
DEFAULT_CACHE = REPO / "data/derived/atk-2022-deep-autoencoder/sgcc-paper-literal.npz"
ISET_RUNS = REPO / "data/derived/atk-2022-deep-autoencoder/runs/table_3/iset"
ISET_CACHE = REPO / "data/derived/atk-2022-deep-autoencoder/iset-paper-literal.npz"
DEFAULT_MODELS = (
    "fc_sae",
    "lstm_sae",
    "lstm_vae",
    "supervised_feed_forward",
)
ANOMALY_MODELS = {"fc_sae", "lstm_sae", "fc_vae", "lstm_vae", "lstm_aea"}
SUPERVISED_MODELS = {
    "naive_bayes",
    "supervised_feed_forward",
    "supervised_lstm",
    "multiclass_svm",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_latest_attempts(root: Path, models: tuple[str, ...]) -> list[Path]:
    selected: list[Path] = []
    for model in models:
        model_root = root / model
        if not model_root.is_dir():
            continue
        for seed_root in sorted(model_root.glob("seed_*")):
            valid: list[tuple[bool, Path]] = []
            for attempt in sorted((seed_root / "attempts").glob("*")):
                manifest_path = attempt / "manifest.json"
                if not manifest_path.is_file():
                    continue
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") != "complete":
                    continue
                artifacts = manifest.get("artifacts", {})
                if "arrays.npz" not in artifacts:
                    continue
                if all(
                    (attempt / name).is_file()
                    and _sha256(attempt / name) == expected
                    for name, expected in artifacts.items()
                ):
                    metadata = json.loads(
                        (attempt / "metadata.json").read_text(encoding="utf-8")
                    )
                    is_panther_ddp = (
                        "distributed_execution" in metadata.get("execution", {})
                    )
                    valid.append((is_panther_ddp, attempt))
            if valid:
                # The frozen aggregate accepts the the cluster DDP branch for
                # neural evidence. Prefer it over a later ordinary local run;
                # within one branch, use the newest immutable attempt.
                selected.append(
                    sorted(valid, key=lambda item: (item[0], item[1]))[-1][1]
                )
    return selected


def _effect_size(negative: np.ndarray, positive: np.ndarray) -> float:
    pooled_variance = (
        (negative.size - 1) * negative.var(ddof=1)
        + (positive.size - 1) * positive.var(ddof=1)
    ) / (negative.size + positive.size - 2)
    if pooled_variance == 0:
        return 0.0 if positive.mean() == negative.mean() else math.copysign(
            math.inf, positive.mean() - negative.mean()
        )
    return float((positive.mean() - negative.mean()) / math.sqrt(pooled_variance))


def _score_diagnostic(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    negative = scores[labels == 0]
    positive = scores[labels == 1]
    auc = float(roc_auc_score(labels, scores))
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    best = int(np.argmax(true_positive_rate - false_positive_rate))
    reversed_fpr, reversed_tpr, reversed_thresholds = roc_curve(labels, -scores)
    reversed_best = int(np.argmax(reversed_tpr - reversed_fpr))
    return {
        "n": int(labels.size),
        "negative_n": int(negative.size),
        "positive_n": int(positive.size),
        "auc": auc,
        "reversed_auc": 1.0 - auc,
        "oracle_test_threshold": float(thresholds[best]),
        "oracle_test_balanced_accuracy": float(
            (true_positive_rate[best] + 1.0 - false_positive_rate[best]) / 2.0
        ),
        "reversed_oracle_test_threshold_on_negated_score": float(
            reversed_thresholds[reversed_best]
        ),
        "reversed_oracle_test_balanced_accuracy": float(
            (
                reversed_tpr[reversed_best]
                + 1.0
                - reversed_fpr[reversed_best]
            )
            / 2.0
        ),
        "ks_statistic": float(ks_2samp(negative, positive).statistic),
        "cohen_d": _effect_size(negative, positive),
        "negative_mean": float(negative.mean()),
        "positive_mean": float(positive.mean()),
        "negative_quantiles_10_50_90": np.quantile(
            negative, (0.1, 0.5, 0.9)
        ).tolist(),
        "positive_quantiles_10_50_90": np.quantile(
            positive, (0.1, 0.5, 0.9)
        ).tolist(),
    }


def _mean_square_rows(values: np.ndarray, *, chunk_rows: int = 200_000) -> np.ndarray:
    """Compute input energy without materializing a second full test matrix."""

    output = np.empty(values.shape[0], dtype=np.float64)
    for start in range(0, values.shape[0], chunk_rows):
        stop = min(start + chunk_rows, values.shape[0])
        chunk = np.asarray(values[start:stop], dtype=np.float64)
        output[start:stop] = np.mean(np.square(chunk), axis=1)
    return output


def analyze(
    runs_root: Path,
    cache_path: Path,
    models: tuple[str, ...],
    *,
    dataset: str = "sgcc",
) -> dict[str, Any]:
    attempts = _verified_latest_attempts(runs_root, models)
    if not attempts:
        raise FileNotFoundError(f"no verified completed attempts under {runs_root}")

    with np.load(cache_path, allow_pickle=False) as cache:
        anomaly_values = np.asarray(cache["anomaly_test_values"], dtype=np.float32)
        anomaly_labels = np.asarray(cache["anomaly_test_labels"], dtype=np.int8)
        anomaly_ids = np.asarray(cache["anomaly_test_sample_ids"]).astype(str)
        anomaly_synthetic = np.asarray(
            cache["anomaly_test_is_synthetic"], dtype=bool
        )
        if dataset == "iset":
            supervised_labels = np.asarray(
                cache["supervised_test_labels"], dtype=np.int8
            )
            supervised_ids = np.asarray(
                cache["supervised_test_sample_ids"]
            ).astype(str)
            supervised_synthetic = np.asarray(
                cache["supervised_test_is_synthetic"], dtype=bool
            )
        else:
            supervised_labels = supervised_ids = supervised_synthetic = None
    input_energy = _mean_square_rows(anomaly_values)
    input_summary = {
        "minimum": float(anomaly_values.min()),
        "maximum": float(anomaly_values.max()),
        "negative_fraction": float(np.mean(anomaly_values < 0)),
    }

    records: list[dict[str, Any]] = []
    score_vectors: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]] = {}
    for attempt in attempts:
        result = json.loads((attempt / "result.json").read_text(encoding="utf-8"))
        metadata = json.loads((attempt / "metadata.json").read_text(encoding="utf-8"))
        model = str(result["model"])
        seed = int(result["seed"])
        with np.load(attempt / "arrays.npz", allow_pickle=False) as arrays:
            if {"labels", "sample_ids", "is_synthetic"}.issubset(arrays.files):
                labels = np.asarray(arrays["labels"], dtype=np.int8)
                sample_ids = np.asarray(arrays["sample_ids"]).astype(str)
                synthetic = np.asarray(arrays["is_synthetic"], dtype=bool)
            elif dataset == "iset" and model in ANOMALY_MODELS | {"arima", "one_class_svm"}:
                labels = anomaly_labels
                sample_ids = anomaly_ids
                synthetic = anomaly_synthetic
            elif dataset == "iset" and model in SUPERVISED_MODELS:
                assert (
                    supervised_labels is not None
                    and supervised_ids is not None
                    and supervised_synthetic is not None
                )
                labels = supervised_labels
                sample_ids = supervised_ids
                synthetic = supervised_synthetic
            else:
                raise ValueError(
                    f"{attempt} omits row provenance and has no cache binding"
                )
            for key in sorted(name for name in arrays.files if name.startswith("score__")):
                score_name = key.removeprefix("score__")
                scores = np.asarray(arrays[key], dtype=np.float64)
                subsets: dict[str, Any] = {}
                for name, mask in (
                    ("paper_primary_all_rows", np.ones(labels.size, dtype=bool)),
                    ("original_rows_only_diagnostic", ~synthetic),
                ):
                    subsets[name] = _score_diagnostic(labels[mask], scores[mask])
                record: dict[str, Any] = {
                    "model": model,
                    "seed": seed,
                    "score": score_name,
                    "attempt_id": attempt.name,
                    "execution_branch": (
                        "panther_four_v100_ddp"
                        if "distributed_execution"
                        in metadata.get("execution", {})
                        else (
                            "panther_single_gpu"
                            if metadata.get("environment", {}).get("slurm_job_id")
                            else "single_process"
                        )
                    ),
                    "fixed_threshold_metrics": result["metrics"][score_name],
                    "subsets": subsets,
                }
                if (
                    score_name == "reconstruction_mse"
                    and labels.shape == anomaly_labels.shape
                    and np.array_equal(labels, anomaly_labels)
                    and np.array_equal(sample_ids, anomaly_ids)
                ):
                    record["input_energy_diagnostic"] = {
                        "definition": "mean(square(standardized_input))",
                        "correlation": float(np.corrcoef(scores, input_energy)[0, 1]),
                        "mean_absolute_difference": float(
                            np.mean(np.abs(scores - input_energy))
                        ),
                    }
                records.append(record)
                score_vectors[(model, seed, score_name)] = (sample_ids, scores)

    similarities: list[dict[str, Any]] = []
    keys = sorted(score_vectors)
    for left_index, left_key in enumerate(keys):
        left_ids, left_scores = score_vectors[left_key]
        for right_key in keys[left_index + 1 :]:
            right_ids, right_scores = score_vectors[right_key]
            if left_key[2] != right_key[2] or not np.array_equal(left_ids, right_ids):
                continue
            similarities.append(
                {
                    "left": {"model": left_key[0], "seed": left_key[1]},
                    "right": {"model": right_key[0], "seed": right_key[1]},
                    "score": left_key[2],
                    "correlation": float(np.corrcoef(left_scores, right_scores)[0, 1]),
                    "mean_absolute_difference": float(
                        np.mean(np.abs(left_scores - right_scores))
                    ),
                    "maximum_absolute_difference": float(
                        np.max(np.abs(left_scores - right_scores))
                    ),
                }
            )

    return {
        "schema_version": 1,
        "study": "atk-2022-deep-autoencoder",
        "track": "exploratory_paper_literal_score_sanity",
        "dataset": dataset,
        "interpretation_limits": [
            "The oracle threshold uses test labels and is diagnostic only.",
            "Original-only rows are diagnostic; the paper-primary branch includes test-set ADASYN rows.",
            "These are exploratory results, not a confirmatory paper-level verdict.",
        ],
        "standardized_anomaly_test_input": input_summary,
        "selected_attempt_count": len(attempts),
        "score_records": records,
        "aligned_score_similarities": similarities,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("sgcc", "iset"), default="sgcc")
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs_root = args.runs_root or (ISET_RUNS if args.dataset == "iset" else DEFAULT_RUNS)
    cache = args.cache or (ISET_CACHE if args.dataset == "iset" else DEFAULT_CACHE)
    models = tuple(args.models or DEFAULT_MODELS)
    document = analyze(
        runs_root.expanduser().resolve(),
        cache.expanduser().resolve(),
        models,
        dataset=args.dataset,
    )
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
