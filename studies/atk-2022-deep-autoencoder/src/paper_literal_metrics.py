"""Paper-literal binary metrics for Takiddin et al. (2022).

The paper reports detection rate (DR), false-alarm rate (FA), specificity
(SP), precision (PR), balanced accuracy under the label ``ACC``, F1, and
ROC-AUC.  This module deliberately computes every value from one common set of
labels, continuous anomaly scores, and threshold-derived predictions.  It also
retains the four confusion counts so that the reported percentages remain
auditable.

All values returned by :class:`PaperMetrics` are fractions in ``[0, 1]`` (or
``NaN`` when a denominator/class is absent), not percentages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import roc_auc_score, roc_curve


ScoreOrientation = Literal["higher", "lower"]
ThresholdRule = Literal[
    "printed_constant",
    "roc_central_threshold_median",
    "threshold_iqr_midpoint",
    "threshold_iqr_median",
    "validation_youden_j",
]


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    """Return a finite ratio or NaN when the metric is undefined."""

    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _one_dimensional(values: ArrayLike, name: str) -> NDArray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def threshold_predictions(
    scores: ArrayLike,
    threshold: float,
    *,
    positive_if: ScoreOrientation = "higher",
    inclusive: bool = True,
) -> NDArray[np.int8]:
    """Convert continuous anomaly scores into fixed-threshold predictions.

    ``positive_if`` makes score direction explicit.  No min-max scaling or
    sign inference from the test labels is performed: either would make the
    paper's fixed thresholds data dependent.  Equality is treated as positive
    by default; it is immaterial for continuous scores but deterministic for
    degenerate tests.
    """

    score_array = _one_dimensional(scores, "scores").astype(np.float64, copy=False)
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores must contain only finite values")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if positive_if == "higher":
        predicted = score_array >= threshold if inclusive else score_array > threshold
    elif positive_if == "lower":
        predicted = score_array <= threshold if inclusive else score_array < threshold
    else:
        raise ValueError("positive_if must be 'higher' or 'lower'")
    return predicted.astype(np.int8)


@dataclass(frozen=True)
class PaperMetrics:
    """One auditable realization of the paper's Table II/III metrics."""

    tp: int
    fp: int
    tn: int
    fn: int
    dr: float
    fa: float
    sp: float
    precision: float
    balanced_accuracy: float
    f1: float
    auc: float
    threshold: float
    positive_if: ScoreOrientation

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def positives(self) -> int:
        return self.tp + self.fn

    @property
    def negatives(self) -> int:
        return self.tn + self.fp

    def as_dict(self) -> dict[str, int | float | str]:
        """Return stable machine-readable field names and confusion counts."""

        result: dict[str, int | float | str] = asdict(self)
        result.update(
            {
                "n": self.n,
                "positives": self.positives,
                "negatives": self.negatives,
            }
        )
        return result

    def as_paper_dict(self, *, percentages: bool = True) -> dict[str, int | float]:
        """Return the paper's column labels plus the underlying counts."""

        factor = 100.0 if percentages else 1.0
        return {
            "DR": self.dr * factor,
            "FA": self.fa * factor,
            "SP": self.sp * factor,
            "PR": self.precision * factor,
            "ACC": self.balanced_accuracy * factor,
            "F1": self.f1 * factor,
            "AUC": self.auc * factor,
            "TP": self.tp,
            "FP": self.fp,
            "TN": self.tn,
            "FN": self.fn,
        }


@dataclass(frozen=True)
class ThresholdSelection:
    """One auditable resolution of the paper's undefined ROC/IQR phrase."""

    rule: ThresholdRule
    threshold: float
    positive_if: ScoreOrientation
    sample_count: int
    benign_count: int
    malicious_count: int
    finite_roc_thresholds: int
    details: Mapping[str, float | int | str]

    def as_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "details": dict(self.details),
        }


def select_threshold(
    y_validation: ArrayLike,
    scores: ArrayLike,
    *,
    rule: ThresholdRule,
    positive_if: ScoreOrientation = "higher",
    supplied_threshold: float | None = None,
) -> ThresholdSelection:
    """Execute one predeclared interpretation of “median of IQR of ROC.”

    The paper does not define a unique operation. These branches deliberately
    remain separate:

    - ``roc_central_threshold_median`` takes the median threshold from the
      central half of ROC points in their returned curve order.
    - ``threshold_iqr_midpoint`` takes the midpoint of Q1 and Q3 of all finite
      ROC thresholds.
    - ``threshold_iqr_median`` takes the median of thresholds lying within
      their own Q1--Q3 interval.
    - ``validation_youden_j`` is a corrected-control rule: maximize
      sensitivity + specificity - 1 on the frozen validation population,
      breaking exact ties by the first threshold in scikit-learn's ROC order.

    ``printed_constant`` requires an explicit supplied value and never inspects
    validation labels. Dataset-specific versus ISET-transferred derivation is
    an execution scope, not a threshold formula.
    """

    labels = _one_dimensional(y_validation, "y_validation")
    score_array = _one_dimensional(scores, "scores").astype(np.float64, copy=False)
    if labels.size != score_array.size:
        raise ValueError("y_validation and scores must have the same length")
    if not np.isfinite(score_array).all():
        raise ValueError("scores must contain only finite values")
    if positive_if not in {"higher", "lower"}:
        raise ValueError("positive_if must be 'higher' or 'lower'")
    actual = np.equal(labels, 1).astype(np.int8)
    benign_count = int(np.count_nonzero(actual == 0))
    malicious_count = int(np.count_nonzero(actual == 1))

    if rule == "printed_constant":
        if supplied_threshold is None or not np.isfinite(supplied_threshold):
            raise ValueError(f"{rule} requires a finite supplied_threshold")
        return ThresholdSelection(
            rule=rule,
            threshold=float(supplied_threshold),
            positive_if=positive_if,
            sample_count=int(labels.size),
            benign_count=benign_count,
            malicious_count=malicious_count,
            finite_roc_thresholds=0,
            details={"source": "supplied_without_validation_derivation"},
        )

    if benign_count == 0 or malicious_count == 0:
        raise ValueError("ROC-derived threshold selection requires both classes")
    oriented = score_array if positive_if == "higher" else -score_array
    fpr, tpr, thresholds = roc_curve(actual, oriented)
    finite_mask = np.isfinite(thresholds)
    finite = thresholds[finite_mask]
    if finite.size == 0:
        raise ValueError("ROC produced no finite threshold candidates")
    finite_fpr = fpr[finite_mask]
    finite_tpr = tpr[finite_mask]
    if positive_if == "lower":
        finite = -finite

    if rule == "roc_central_threshold_median":
        start = int(np.floor(0.25 * finite.size))
        stop = int(np.ceil(0.75 * finite.size))
        central = finite[start:stop]
        threshold = float(np.median(central))
        details: dict[str, float | int | str] = {
            "central_start_index": start,
            "central_stop_index_exclusive": stop,
            "central_count": int(central.size),
        }
    elif rule in {"threshold_iqr_midpoint", "threshold_iqr_median"}:
        q1, q3 = np.quantile(finite, [0.25, 0.75])
        if rule == "threshold_iqr_midpoint":
            threshold = float((q1 + q3) / 2.0)
            retained = finite
        else:
            retained = finite[(finite >= q1) & (finite <= q3)]
            if retained.size:
                threshold = float(np.median(retained))
                empty_fallback = "not_required"
            else:
                # With two finite ROC thresholds, interpolated Q1/Q3 can lie
                # strictly between both observations. Preserve executability
                # by taking the median of all finite candidates and record the
                # degenerate fallback rather than returning NaN.
                threshold = float(np.median(finite))
                empty_fallback = "median_all_finite"
        details = {
            "q1": float(q1),
            "q3": float(q3),
            "iqr_retained_count": int(retained.size),
            **(
                {"empty_iqr_fallback": empty_fallback}
                if rule == "threshold_iqr_median"
                else {}
            ),
        }
    elif rule == "validation_youden_j":
        youden = finite_tpr - finite_fpr
        best = float(np.max(youden))
        tied = np.flatnonzero(np.isclose(youden, best, rtol=0.0, atol=1e-12))
        selected_index = int(tied[0])
        threshold = float(finite[selected_index])
        details = {
            "youden_j": best,
            "selected_fpr": float(finite_fpr[selected_index]),
            "selected_tpr": float(finite_tpr[selected_index]),
            "tie_count": int(tied.size),
            "tie_break": "first_finite_threshold_in_roc_order",
        }
    else:
        raise ValueError(f"unsupported threshold rule {rule!r}")

    return ThresholdSelection(
        rule=rule,
        threshold=threshold,
        positive_if=positive_if,
        sample_count=int(labels.size),
        benign_count=benign_count,
        malicious_count=malicious_count,
        finite_roc_thresholds=int(finite.size),
        details=details,
    )


def evaluate_binary_scores(
    y_true: ArrayLike,
    scores: ArrayLike,
    *,
    threshold: float,
    positive_label: object = 1,
    positive_if: ScoreOrientation = "higher",
    inclusive: bool = True,
) -> PaperMetrics:
    """Compute all paper metrics from one score/prediction realization.

    ROC-AUC uses the same scores and the same declared orientation as the fixed
    threshold.  Single-class inputs retain their confusion counts and return
    ``NaN`` for ROC-AUC instead of aborting an experiment.
    """

    labels = _one_dimensional(y_true, "y_true")
    score_array = _one_dimensional(scores, "scores").astype(np.float64, copy=False)
    if labels.size != score_array.size:
        raise ValueError("y_true and scores must have the same length")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores must contain only finite values")

    actual = np.equal(labels, positive_label)
    predicted = threshold_predictions(
        score_array,
        threshold,
        positive_if=positive_if,
        inclusive=inclusive,
    ).astype(bool)
    tp = int(np.count_nonzero(actual & predicted))
    fp = int(np.count_nonzero(~actual & predicted))
    tn = int(np.count_nonzero(~actual & ~predicted))
    fn = int(np.count_nonzero(actual & ~predicted))

    dr = _safe_ratio(tp, tp + fn)
    fa = _safe_ratio(fp, fp + tn)
    sp = _safe_ratio(tn, tn + fp)
    precision = _safe_ratio(tp, tp + fp)
    balanced_accuracy = float(np.nanmean([dr, sp]))
    f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)

    if np.unique(actual).size < 2:
        auc = float("nan")
    else:
        oriented_scores = score_array if positive_if == "higher" else -score_array
        auc = float(roc_auc_score(actual.astype(np.int8), oriented_scores))

    return PaperMetrics(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        dr=dr,
        fa=fa,
        sp=sp,
        precision=precision,
        balanced_accuracy=balanced_accuracy,
        f1=f1,
        auc=auc,
        threshold=float(threshold),
        positive_if=positive_if,
    )


def evaluate_attack_columns(
    benign_scores: ArrayLike,
    attack_scores: Mapping[int | str, ArrayLike],
    *,
    threshold: float,
    positive_if: ScoreOrientation = "higher",
    inclusive: bool = True,
) -> dict[int | str, PaperMetrics]:
    """Evaluate Table V attack columns against one fixed benign score set.

    Reusing the exact same benign scores, threshold, and score orientation is a
    structural requirement: under this experiment definition every attack
    column must have identical FP, TN, FA, and SP values.  Attack-dependent FA
    values therefore require a different benign set, model, or threshold and
    cannot arise from one fixed evaluation.
    """

    benign = _one_dimensional(benign_scores, "benign_scores").astype(
        np.float64, copy=False
    )
    if not attack_scores:
        raise ValueError("attack_scores must not be empty")

    results: dict[int | str, PaperMetrics] = {}
    for attack, values in attack_scores.items():
        malicious = _one_dimensional(values, f"attack_scores[{attack!r}]").astype(
            np.float64, copy=False
        )
        labels = np.concatenate(
            [np.zeros(benign.size, dtype=np.int8), np.ones(malicious.size, dtype=np.int8)]
        )
        scores = np.concatenate([benign, malicious])
        results[attack] = evaluate_binary_scores(
            labels,
            scores,
            threshold=threshold,
            positive_if=positive_if,
            inclusive=inclusive,
        )
    return results


_AGGREGATE_FIELDS = (
    "dr",
    "fa",
    "sp",
    "precision",
    "balanced_accuracy",
    "f1",
    "auc",
)


def aggregate_seed_metrics(
    metrics: Sequence[PaperMetrics] | Iterable[PaperMetrics],
) -> dict[str, object]:
    """Preserve individual seeds and summarize their metric distributions.

    Standard deviation is the sample standard deviation (``ddof=1``) when two
    or more finite values exist, and zero for one value.  This avoids reporting
    a best seed while retaining all values and per-seed confusion counts.
    """

    rows = list(metrics)
    if not rows:
        raise ValueError("at least one seed metric is required")
    if not all(isinstance(row, PaperMetrics) for row in rows):
        raise TypeError("metrics must contain only PaperMetrics instances")

    summaries: dict[str, dict[str, object]] = {}
    for field in _AGGREGATE_FIELDS:
        values = np.asarray([getattr(row, field) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summaries[field] = {
            "values": values.tolist(),
            "n_finite": int(finite.size),
            "mean": float(np.mean(finite)) if finite.size else float("nan"),
            "sample_std": (
                float(np.std(finite, ddof=1)) if finite.size > 1 else (0.0 if finite.size else float("nan"))
            ),
            "min": float(np.min(finite)) if finite.size else float("nan"),
            "max": float(np.max(finite)) if finite.size else float("nan"),
        }

    count_fields = ("tp", "fp", "tn", "fn")
    per_seed_counts = [
        {field: int(getattr(row, field)) for field in count_fields} for row in rows
    ]
    total_counts = {
        field: int(sum(getattr(row, field) for row in rows)) for field in count_fields
    }
    return {
        "n_seeds": len(rows),
        "metric_units": "fraction",
        "metrics": summaries,
        "confusion_counts": {
            "per_seed": per_seed_counts,
            "total": total_counts,
        },
    }


# Descriptive alias used by some callers/report code.
summarize_seed_metrics = aggregate_seed_metrics
