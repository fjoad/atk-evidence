"""Classical Paper 1 benchmark implementations.

These implementations complete only details omitted by Takiddin et al. (2022)
and frozen in ``PAPER_LITERAL_CONTRACT.md`` / ``AMBIGUITY_REGISTER.md``:

* Gaussian Naive Bayes uses scikit-learn's ``var_smoothing=1e-9`` default.
* One-class SVM uses ``kernel='sigmoid'``, ``gamma='scale'``, and ``nu=0.5``.
* Supervised SVM uses ``kernel='sigmoid'``, ``gamma='scale'``, and ``C=1``.
* ARIMA crosses the frozen autoregressive orders, pooled/per-profile fits, and
  residual-MSE/Gaussian-likelihood score completions while keeping the paper's
  stated differencing degree 1 and moving-average order 0.

The SVM branches either use deterministic, recorded caps or the full declared
training population. The pooled ARIMA-like scorer is vectorized; per-profile
branches fit each profile independently and are explicitly assumption-bound.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
import re
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import OneClassSVM, SVC

from paper_literal_metrics import ScoreOrientation, threshold_predictions


DEFAULT_DATA_SEED = 20260721
DEFAULT_ONE_CLASS_CAP = 12_000
DEFAULT_SUPERVISED_SVM_CAP = 30_000


def _feature_matrix(values: ArrayLike, name: str) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _label_vector(values: ArrayLike, expected_rows: int, name: str) -> NDArray:
    labels = np.asarray(values)
    if labels.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if labels.size != expected_rows:
        raise ValueError(f"{name} must have one label per feature row")
    return labels


def deterministic_cap_indices(
    n_samples: int,
    max_samples: int | None,
    *,
    seed: int = DEFAULT_DATA_SEED,
    labels: ArrayLike | None = None,
) -> NDArray[np.int64]:
    """Select a deterministic (stratified when labeled) capped subset.

    Returned indices are sorted into source order after random membership is
    chosen.  Every observed class is retained when the cap permits it.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if max_samples is None or max_samples >= n_samples:
        return np.arange(n_samples, dtype=np.int64)
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")

    rng = np.random.default_rng(seed)
    if labels is None:
        return np.sort(rng.choice(n_samples, size=max_samples, replace=False)).astype(
            np.int64
        )

    label_array = _label_vector(labels, n_samples, "labels")
    classes, inverse, counts = np.unique(label_array, return_inverse=True, return_counts=True)
    if max_samples < classes.size:
        raise ValueError("max_samples must be at least the number of observed classes")

    ideal = counts.astype(np.float64) * (max_samples / n_samples)
    allocation = np.floor(ideal).astype(np.int64)
    allocation = np.maximum(allocation, 1)
    allocation = np.minimum(allocation, counts)

    # Largest-remainder allocation, respecting each class's capacity.
    while int(allocation.sum()) < max_samples:
        candidates = np.flatnonzero(allocation < counts)
        priorities = ideal[candidates] - allocation[candidates]
        chosen = int(candidates[np.argmax(priorities)])
        allocation[chosen] += 1
    while int(allocation.sum()) > max_samples:
        candidates = np.flatnonzero(allocation > 1)
        priorities = allocation[candidates] - ideal[candidates]
        chosen = int(candidates[np.argmax(priorities)])
        allocation[chosen] -= 1

    selected: list[NDArray[np.int64]] = []
    for class_index, count in enumerate(allocation):
        members = np.flatnonzero(inverse == class_index)
        selected.append(rng.choice(members, size=int(count), replace=False))
    return np.sort(np.concatenate(selected)).astype(np.int64)


@dataclass(frozen=True)
class BenchmarkResult:
    """Scores, predictions, timing, and fit metadata for one benchmark run."""

    name: str
    scores: NDArray[np.float64]
    predictions: NDArray[np.int8]
    threshold: float
    positive_if: ScoreOrientation
    fit_seconds: float
    score_seconds: float
    train_samples_available: int
    train_samples_used: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def total_seconds(self) -> float:
        return self.fit_seconds + self.score_seconds


def gaussian_nb_benchmark(
    x_train: ArrayLike,
    y_train: ArrayLike,
    x_test: ArrayLike,
    *,
    positive_label: object = 1,
    threshold: float = 0.5,
) -> BenchmarkResult:
    """Fit GaussianNB and return positive-class probability scores."""

    train = _feature_matrix(x_train, "x_train")
    test = _feature_matrix(x_test, "x_test")
    if train.shape[1] != test.shape[1]:
        raise ValueError("x_train and x_test must have the same feature count")
    labels = _label_vector(y_train, train.shape[0], "y_train")

    model = GaussianNB(var_smoothing=1e-9)
    started = time.perf_counter()
    model.fit(train, labels)
    fit_seconds = time.perf_counter() - started
    matches = np.flatnonzero(np.equal(model.classes_, positive_label))
    if matches.size != 1:
        raise ValueError("positive_label must occur in y_train exactly once as a class")

    started = time.perf_counter()
    scores = model.predict_proba(test)[:, int(matches[0])].astype(np.float64)
    predictions = threshold_predictions(scores, threshold, positive_if="higher")
    score_seconds = time.perf_counter() - started
    return BenchmarkResult(
        name="Naive Bayes",
        scores=scores,
        predictions=predictions,
        threshold=float(threshold),
        positive_if="higher",
        fit_seconds=fit_seconds,
        score_seconds=score_seconds,
        train_samples_available=train.shape[0],
        train_samples_used=train.shape[0],
        metadata={"var_smoothing": 1e-9},
    )


def one_class_svm_benchmark(
    x_train_benign: ArrayLike,
    x_test: ArrayLike,
    *,
    max_samples: int | None = DEFAULT_ONE_CLASS_CAP,
    seed: int = DEFAULT_DATA_SEED,
    threshold: float = 0.45,
) -> BenchmarkResult:
    """Fit the frozen one-class SVM branch on benign training samples only.

    scikit-learn's decision function is positive for inliers.  Negating it
    creates an explicit high-is-anomaly score; no test-dependent normalization
    is introduced to make the paper's fixed threshold look better.
    """

    available = _feature_matrix(x_train_benign, "x_train_benign")
    test = _feature_matrix(x_test, "x_test")
    if available.shape[1] != test.shape[1]:
        raise ValueError("x_train_benign and x_test must have the same feature count")
    indices = deterministic_cap_indices(
        available.shape[0], max_samples, seed=seed
    )
    train = available[indices]

    model = OneClassSVM(kernel="sigmoid", gamma="scale", nu=0.5)
    started = time.perf_counter()
    model.fit(train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    scores = -model.decision_function(test).reshape(-1).astype(np.float64)
    predictions = threshold_predictions(scores, threshold, positive_if="higher")
    score_seconds = time.perf_counter() - started
    return BenchmarkResult(
        name="Single-class SVM",
        scores=scores,
        predictions=predictions,
        threshold=float(threshold),
        positive_if="higher",
        fit_seconds=fit_seconds,
        score_seconds=score_seconds,
        train_samples_available=available.shape[0],
        train_samples_used=train.shape[0],
        metadata={
            "kernel": "sigmoid",
            "gamma": "scale",
            "nu": 0.5,
            "cap_seed": seed,
            "max_samples": max_samples,
            "score_definition": "negative decision_function",
        },
    )


def multiclass_svm_benchmark(
    x_train: ArrayLike,
    y_train: ArrayLike,
    x_test: ArrayLike,
    *,
    benign_label: object = 0,
    max_samples: int | None = DEFAULT_SUPERVISED_SVM_CAP,
    seed: int = DEFAULT_DATA_SEED,
    threshold: float = 0.0,
) -> BenchmarkResult:
    """Fit the paper's supervised sigmoid SVM branch.

    For more than two classes, the anomaly score is the maximum non-benign OVR
    margin minus the benign margin.  It is positive exactly when a non-benign
    class outranks the benign class.  For two classes, the signed binary margin
    is oriented so higher always means non-benign.
    """

    available = _feature_matrix(x_train, "x_train")
    test = _feature_matrix(x_test, "x_test")
    if available.shape[1] != test.shape[1]:
        raise ValueError("x_train and x_test must have the same feature count")
    labels = _label_vector(y_train, available.shape[0], "y_train")
    indices = deterministic_cap_indices(
        available.shape[0], max_samples, seed=seed, labels=labels
    )
    train = available[indices]
    train_labels = labels[indices]

    model = SVC(
        C=1.0,
        kernel="sigmoid",
        gamma="scale",
        decision_function_shape="ovr",
    )
    started = time.perf_counter()
    model.fit(train, train_labels)
    fit_seconds = time.perf_counter() - started
    benign_matches = np.flatnonzero(np.equal(model.classes_, benign_label))
    if benign_matches.size != 1:
        raise ValueError("benign_label must occur in y_train exactly once as a class")
    benign_index = int(benign_matches[0])

    started = time.perf_counter()
    margins = np.asarray(model.decision_function(test), dtype=np.float64)
    if model.classes_.size == 2:
        # Binary SVC's one-dimensional margin is positive for classes_[1].
        scores = margins.reshape(-1)
        if benign_index == 1:
            scores = -scores
    else:
        if margins.ndim != 2 or margins.shape[1] != model.classes_.size:
            raise RuntimeError("unexpected multiclass SVC decision-function shape")
        non_benign = np.delete(margins, benign_index, axis=1)
        scores = np.max(non_benign, axis=1) - margins[:, benign_index]
    scores = scores.astype(np.float64, copy=False)
    predictions = threshold_predictions(scores, threshold, positive_if="higher")
    score_seconds = time.perf_counter() - started
    return BenchmarkResult(
        name="Multi-class SVM",
        scores=scores,
        predictions=predictions,
        threshold=float(threshold),
        positive_if="higher",
        fit_seconds=fit_seconds,
        score_seconds=score_seconds,
        train_samples_available=available.shape[0],
        train_samples_used=train.shape[0],
        metadata={
            "kernel": "sigmoid",
            "gamma": "scale",
            "C": 1.0,
            "decision_function_shape": "ovr",
            "cap_seed": seed,
            "max_samples": max_samples,
            "classes": model.classes_.tolist(),
        },
    )


@dataclass(frozen=True)
class ARIMA110ResidualModel:
    """Vectorized pooled ARIMA(1,1,0)-style profile residual model."""

    intercept: float
    phi: float

    @classmethod
    def fit(cls, x_train_benign: ArrayLike) -> "ARIMA110ResidualModel":
        train = _feature_matrix(x_train_benign, "x_train_benign")
        if train.shape[1] < 3:
            raise ValueError("ARIMA(1,1,0) profiles require at least three readings")
        differences = np.diff(train, axis=1)
        lagged = differences[:, :-1].reshape(-1)
        targets = differences[:, 1:].reshape(-1)
        lag_mean = float(np.mean(lagged))
        target_mean = float(np.mean(targets))
        centered = lagged - lag_mean
        denominator = float(np.dot(centered, centered))
        if denominator <= np.finfo(np.float64).eps:
            phi = 0.0
        else:
            phi = float(np.dot(centered, targets - target_mean) / denominator)
        intercept = target_mean - phi * lag_mean
        return cls(intercept=intercept, phi=phi)

    def score_samples(self, values: ArrayLike) -> NDArray[np.float64]:
        matrix = _feature_matrix(values, "values")
        if matrix.shape[1] < 3:
            raise ValueError("ARIMA(1,1,0) profiles require at least three readings")
        differences = np.diff(matrix, axis=1)
        residuals = differences[:, 1:] - (
            self.intercept + self.phi * differences[:, :-1]
        )
        return np.mean(np.square(residuals), axis=1, dtype=np.float64)


def arima_110_benchmark(
    x_train_benign: ArrayLike,
    x_test: ArrayLike,
    *,
    threshold: float = 0.58,
) -> BenchmarkResult:
    """Fit and score the frozen vectorized ARIMA(1,1,0)-style branch."""

    train = _feature_matrix(x_train_benign, "x_train_benign")
    test = _feature_matrix(x_test, "x_test")
    if train.shape[1] != test.shape[1]:
        raise ValueError("x_train_benign and x_test must have the same feature count")

    started = time.perf_counter()
    model = ARIMA110ResidualModel.fit(train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    scores = model.score_samples(test)
    predictions = threshold_predictions(scores, threshold, positive_if="higher")
    score_seconds = time.perf_counter() - started
    return BenchmarkResult(
        name="ARIMA",
        scores=scores,
        predictions=predictions,
        threshold=float(threshold),
        positive_if="higher",
        fit_seconds=fit_seconds,
        score_seconds=score_seconds,
        train_samples_available=train.shape[0],
        train_samples_used=train.shape[0],
        metadata={
            "order": [1, 1, 0],
            "fit": "pooled OLS on within-profile first-difference transitions",
            "score": "within-profile residual MSE",
            "intercept": model.intercept,
            "phi": model.phi,
        },
    )


_ARIMA_COMPLETION = re.compile(
    r"^p(?P<p>0|1|2|5)_(?P<fit>pooled|profile)_(?P<score>mse|likelihood)$"
)


def _ar_design(
    differences: NDArray[np.float64],
    order: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if differences.ndim != 2:
        raise ValueError("differences must be a two-dimensional matrix")
    if differences.shape[1] <= order:
        raise ValueError(
            f"AR({order}) after first differencing needs more than "
            f"{order + 1} original readings"
        )
    targets = differences[:, order:].reshape(-1)
    if order == 0:
        design = np.ones((targets.size, 1), dtype=np.float64)
    else:
        lagged = [
            differences[:, order - lag - 1 : differences.shape[1] - lag - 1]
            .reshape(-1)
            for lag in range(order)
        ]
        design = np.column_stack(
            [np.ones(targets.size, dtype=np.float64), *lagged]
        )
    return design, targets


def _fit_ar_coefficients(
    differences: NDArray[np.float64],
    order: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    design, targets = _ar_design(differences, order)
    coefficients, *_ = np.linalg.lstsq(design, targets, rcond=None)
    residuals = targets - design @ coefficients
    return coefficients, residuals


def _ar_residuals(
    differences: NDArray[np.float64],
    order: int,
    coefficients: NDArray[np.float64],
) -> NDArray[np.float64]:
    design, targets = _ar_design(differences, order)
    return (targets - design @ coefficients).reshape(differences.shape[0], -1)


def arima_completion_benchmark(
    x_train_benign: ArrayLike,
    x_test: ArrayLike,
    *,
    completion: str,
    threshold: float = 0.58,
) -> BenchmarkResult:
    """Execute one frozen ARIMA completion from ambiguity A21.

    Every branch first differences each profile (``d=1``) and has no moving
    average term (``q=0``), exactly as the paper states. ``pooled`` estimates
    one AR model from all benign training transitions. ``profile`` estimates
    one model independently within each scored profile because the paper never
    identifies a cross-profile training unit. The latter is intentionally
    retained as an assumption-bound paper interpretation, not endorsed as the
    corrected detector.
    """

    match = _ARIMA_COMPLETION.fullmatch(completion)
    if match is None:
        raise ValueError(f"unsupported ARIMA completion {completion!r}")
    order = int(match.group("p"))
    fit_unit = match.group("fit")
    score_kind = match.group("score")
    train = _feature_matrix(x_train_benign, "x_train_benign")
    test = _feature_matrix(x_test, "x_test")
    if train.shape[1] != test.shape[1]:
        raise ValueError("x_train_benign and x_test must have the same feature count")
    train_differences = np.diff(train, axis=1)
    test_differences = np.diff(test, axis=1)
    variance_floor = np.finfo(np.float64).eps

    fit_started = time.perf_counter()
    if fit_unit == "pooled":
        coefficients, train_residuals = _fit_ar_coefficients(
            train_differences,
            order,
        )
        residual_variance = max(
            float(np.mean(np.square(train_residuals))),
            variance_floor,
        )
        fit_metadata: Mapping[str, Any] = {
            "coefficients": coefficients.tolist(),
            "residual_variance": residual_variance,
        }
    else:
        coefficients = np.empty(0, dtype=np.float64)
        residual_variance = float("nan")
        fit_metadata = {
            "coefficients": "estimated_independently_within_each_scored_profile",
            "residual_variance": "estimated_independently_within_each_scored_profile",
        }
    fit_seconds = time.perf_counter() - fit_started

    score_started = time.perf_counter()
    if fit_unit == "pooled":
        residual_matrix = _ar_residuals(
            test_differences,
            order,
            coefficients,
        )
        mse = np.mean(np.square(residual_matrix), axis=1)
        if score_kind == "mse":
            scores = mse
        else:
            scores = 0.5 * np.mean(
                np.log(2.0 * np.pi * residual_variance)
                + np.square(residual_matrix) / residual_variance,
                axis=1,
            )
    else:
        scores_list: list[float] = []
        for row in test_differences:
            _, residuals = _fit_ar_coefficients(row.reshape(1, -1), order)
            mse = max(float(np.mean(np.square(residuals))), variance_floor)
            scores_list.append(
                mse
                if score_kind == "mse"
                else 0.5 * (np.log(2.0 * np.pi * mse) + 1.0)
            )
        scores = np.asarray(scores_list, dtype=np.float64)
    predictions = threshold_predictions(scores, threshold, positive_if="higher")
    score_seconds = time.perf_counter() - score_started
    return BenchmarkResult(
        name="ARIMA",
        scores=np.asarray(scores, dtype=np.float64),
        predictions=predictions,
        threshold=float(threshold),
        positive_if="higher",
        fit_seconds=fit_seconds,
        score_seconds=score_seconds,
        train_samples_available=train.shape[0],
        train_samples_used=train.shape[0],
        metadata={
            "completion": completion,
            "order": [order, 1, 0],
            "fit_unit": fit_unit,
            "score": (
                "residual_mse"
                if score_kind == "mse"
                else "mean_gaussian_negative_log_likelihood"
            ),
            **dict(fit_metadata),
        },
    )


# Short names for runner code while preserving descriptive public APIs.
fit_gaussian_nb = gaussian_nb_benchmark
fit_one_class_svm = one_class_svm_benchmark
fit_multiclass_svm = multiclass_svm_benchmark
fit_arima_110 = arima_110_benchmark
fit_arima_completion = arima_completion_benchmark
