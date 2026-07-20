"""Fast SGCC separability audit using fixed customer-level splits.

This is intentionally a sanity experiment, not an exact paper reproduction:
the paper never explains how 1,034 SGCC daily readings become a 48-value model
input. Results test whether simple customer-profile statistics already separate
the public labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_record(
    name: str,
    validation_scores: np.ndarray,
    test_scores: np.ndarray,
    y_validation: np.ndarray,
    y_test: np.ndarray,
    target_fa: float,
) -> dict[str, float | str]:
    orientation = 1.0
    if roc_auc_score(y_validation, validation_scores) < 0.5:
        orientation = -1.0
    validation_scores = orientation * validation_scores
    test_scores = orientation * test_scores
    threshold = float(np.quantile(validation_scores[y_validation == 0], 1.0 - target_fa))
    predictions = (test_scores > threshold).astype(np.int8)
    false_alarm = float(np.mean(predictions[y_test == 0] == 1))
    detection_rate = float(np.mean(predictions[y_test == 1] == 1))
    return {
        "model": name,
        "orientation": orientation,
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y_test, test_scores)),
        "detection_rate": detection_rate,
        "false_alarm_rate": false_alarm,
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }


def profile_features(values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    observed = np.sum(~np.isnan(values), axis=1)
    observed_safe = np.maximum(observed, 1)
    zeros = np.nansum(values == 0.0, axis=1)
    positive = np.where(values > 0.0, values, np.nan)
    features = np.column_stack(
        [
            np.nanmean(values, axis=1),
            np.nanmedian(values, axis=1),
            np.nanstd(values, axis=1),
            np.nanmax(values, axis=1),
            np.nanquantile(values, 0.25, axis=1),
            np.nanquantile(values, 0.75, axis=1),
            zeros / observed_safe,
            1.0 - observed / values.shape[1],
            np.sum(~np.isnan(positive), axis=1) / observed_safe,
        ]
    ).astype(np.float32)
    names = [
        "mean",
        "median",
        "std",
        "maximum",
        "q25",
        "q75",
        "zero_fraction",
        "missing_fraction",
        "positive_fraction",
    ]
    return features, names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/sgcc-verified/data.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("replication/results/sgcc_sanity_seed_20260720.json"),
    )
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--target-fa", type=float, default=0.05)
    args = parser.parse_args()
    start = time.perf_counter()

    frame = pd.read_csv(args.input, low_memory=False)
    date_columns = list(frame.columns[2:])
    parsed_dates = pd.to_datetime(date_columns, format="%Y/%m/%d")
    order = np.argsort(parsed_dates.to_numpy())
    ordered_columns = [date_columns[index] for index in order]
    values = frame[ordered_columns].to_numpy(dtype=np.float32)
    labels = frame["FLAG"].to_numpy(dtype=np.int8)
    fully_missing = np.all(np.isnan(values), axis=1)
    dropped_fully_missing = int(np.sum(fully_missing))
    values = values[~fully_missing]
    labels = labels[~fully_missing]
    features, feature_names = profile_features(values)

    all_indices = np.arange(labels.size)
    train_indices, remaining_indices = train_test_split(
        all_indices,
        test_size=0.4,
        stratify=labels,
        random_state=args.seed,
    )
    validation_indices, test_indices = train_test_split(
        remaining_indices,
        test_size=0.5,
        stratify=labels[remaining_indices],
        random_state=args.seed + 1,
    )

    y_train = labels[train_indices]
    y_validation = labels[validation_indices]
    y_test = labels[test_indices]
    results: list[dict[str, float | str]] = []

    for column_index, feature_name in enumerate(feature_names):
        results.append(
            metric_record(
                f"univariate:{feature_name}",
                features[validation_indices, column_index],
                features[test_indices, column_index],
                y_validation,
                y_test,
                args.target_fa,
            )
        )

    feature_logistic = make_pipeline(
        SimpleImputer(strategy="median"),
        RobustScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=args.seed),
    )
    feature_logistic.fit(features[train_indices], y_train)
    results.append(
        metric_record(
            "supervised:logistic_profile_features",
            feature_logistic.predict_proba(features[validation_indices])[:, 1],
            feature_logistic.predict_proba(features[test_indices])[:, 1],
            y_validation,
            y_test,
            args.target_fa,
        )
    )

    raw_linear = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            max_iter=2000,
            tol=1e-4,
            random_state=args.seed,
        ),
    )
    raw_linear.fit(values[train_indices], y_train)
    results.append(
        metric_record(
            "supervised:linear_1034_days",
            raw_linear.predict_proba(values[validation_indices])[:, 1],
            raw_linear.predict_proba(values[test_indices])[:, 1],
            y_validation,
            y_test,
            args.target_fa,
        )
    )

    benign_train_indices = train_indices[y_train == 0]
    isolation = make_pipeline(
        SimpleImputer(strategy="median"),
        RobustScaler(),
        IsolationForest(
            n_estimators=300,
            max_samples=min(4096, benign_train_indices.size),
            random_state=args.seed,
            n_jobs=-1,
        ),
    )
    isolation.fit(features[benign_train_indices])
    results.append(
        metric_record(
            "anomaly:isolation_forest_profile_features",
            -isolation.decision_function(features[validation_indices]),
            -isolation.decision_function(features[test_indices]),
            y_validation,
            y_test,
            args.target_fa,
        )
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    pca_columns = ~np.all(np.isnan(values[benign_train_indices]), axis=0)
    benign_train = imputer.fit_transform(values[benign_train_indices][:, pca_columns])
    benign_train = scaler.fit_transform(benign_train).astype(np.float32)
    pca = PCA(n_components=32, svd_solver="randomized", random_state=args.seed)
    pca.fit(benign_train)

    def pca_scores(indices: np.ndarray) -> np.ndarray:
        transformed = scaler.transform(
            imputer.transform(values[indices][:, pca_columns])
        ).astype(np.float32)
        reconstruction = pca.inverse_transform(pca.transform(transformed))
        return np.mean((transformed - reconstruction) ** 2, axis=1)

    results.append(
        metric_record(
            "anomaly:pca_32_reconstruction",
            pca_scores(validation_indices),
            pca_scores(test_indices),
            y_validation,
            y_test,
            args.target_fa,
        )
    )

    payload = {
        "experiment": "SGCC customer-profile sanity audit; not exact paper reproduction",
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "seed": args.seed,
        "target_validation_false_alarm_rate": args.target_fa,
        "chronologically_sorted": True,
        "dropped_fully_missing_customers": dropped_fully_missing,
        "pca_input_dates_after_train_only_filter": int(np.sum(pca_columns)),
        "split": {
            "train_customers": int(train_indices.size),
            "validation_customers": int(validation_indices.size),
            "test_customers": int(test_indices.size),
            "test_positives": int(np.sum(y_test == 1)),
            "test_negatives": int(np.sum(y_test == 0)),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "duration_seconds": time.perf_counter() - start,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
