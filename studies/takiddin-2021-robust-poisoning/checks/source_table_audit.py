#!/usr/bin/env python3
"""Audit the paper's printed metric tables under the frozen source contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


STUDY_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_DIR.parents[1]
REPORTED_DIR = STUDY_DIR / "reported"
LEVELS = ("p0", "p10", "p20", "p30")
METRICS = ("DR", "FA", "SP", "PR", "ACC", "F1", "AUC")
EXPECTED_MODELS = {
    "table_3": (
        "random_forest",
        "adaboost",
        "arima",
        "svm",
        "feed_forward",
        "gru",
        "aea",
    ),
    "table_4": (
        "random_forest",
        "adaboost",
        "arima",
        "svm",
        "feed_forward",
        "gru",
        "aea",
    ),
    "table_5": ("aea", "ensemble_averaging", "sequential_ensemble"),
}
ROUNDING_HALF_WIDTH = 0.05
PDF_SHA256 = "03a372fb5ee129799d43a50e6bec8f86fb108e9f8bdc85350e3ffe3e13d4a4ff"


@dataclass(frozen=True)
class Interval:
    low: float
    high: float

    def intersects(self, other: "Interval") -> bool:
        return max(self.low, other.low) <= min(self.high, other.high)

    def intersection(self, other: "Interval") -> "Interval | None":
        low = max(self.low, other.low)
        high = min(self.high, other.high)
        return Interval(low, high) if low <= high else None

    def as_list(self) -> list[float]:
        return [self.low, self.high]


def outward_interval(value: float) -> Interval:
    low = math.nextafter(max(0.0, value - ROUNDING_HALF_WIDTH), -math.inf)
    high = math.nextafter(min(100.0, value + ROUNDING_HALF_WIDTH), math.inf)
    return Interval(max(0.0, low), min(100.0, high))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_table(table_id: str) -> dict[str, dict[str, dict[str, float]]]:
    path = REPORTED_DIR / f"{table_id}.csv"
    values: dict[str, dict[str, dict[str, float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["model", "metric", *LEVELS]:
            raise ValueError(f"{path}: unexpected columns {reader.fieldnames}")
        for row in reader:
            model = row["model"]
            metric = row["metric"]
            if model not in EXPECTED_MODELS[table_id]:
                raise ValueError(f"{path}: unexpected model {model}")
            if metric not in METRICS:
                raise ValueError(f"{path}: unexpected metric {metric}")
            if metric in values.setdefault(model, {}):
                raise ValueError(f"{path}: duplicate {model}/{metric}")
            values[model][metric] = {level: float(row[level]) for level in LEVELS}

    if tuple(values) != EXPECTED_MODELS[table_id]:
        raise ValueError(f"{path}: model order or coverage differs from contract")
    for model, metric_values in values.items():
        if tuple(metric_values) != METRICS:
            raise ValueError(f"{path}: metric order or coverage differs for {model}")
    return values


def harmonic_range(left: Interval, right: Interval) -> Interval:
    def harmonic(a: float, b: float) -> float:
        return 0.0 if a + b == 0.0 else 2.0 * a * b / (a + b)

    return Interval(
        harmonic(left.low, right.low),
        harmonic(left.high, right.high),
    )


def balanced_precision_range(dr: Interval, fa: Interval) -> Interval:
    def precision(dr_value: float, fa_value: float) -> float:
        denominator = dr_value + fa_value
        return 0.0 if denominator == 0.0 else 100.0 * dr_value / denominator

    return Interval(
        precision(dr.low, fa.high),
        precision(dr.high, fa.low),
    )


def constrain_linear_leq(current: Interval, a: float, b: float, c: float) -> Interval | None:
    """Intersect p in current with a + b*p <= c."""

    if abs(b) < 1e-15:
        return current if a <= c else None
    boundary = (c - a) / b
    allowed = Interval(-math.inf, boundary) if b > 0 else Interval(boundary, math.inf)
    return current.intersection(allowed)


def accuracy_prevalence_range(dr: Interval, fa: Interval, acc: Interval) -> Interval | None:
    """Prevalences not excluded by DR, FA, and ACC rounding intervals."""

    specificity = Interval(100.0 - fa.high, 100.0 - fa.low)
    current: Interval | None = Interval(0.0, 1.0)

    # Minimum possible accuracy at p must not exceed the reported maximum.
    current = constrain_linear_leq(
        current,
        specificity.low,
        dr.low - specificity.low,
        acc.high,
    )
    if current is None:
        return None

    # Maximum possible accuracy at p must not fall below the reported minimum.
    return constrain_linear_leq(
        current,
        -specificity.high,
        -(dr.high - specificity.high),
        -acc.low,
    )


def precision_prevalence_range(dr: Interval, fa: Interval, pr: Interval) -> Interval:
    """Conservative hull of prevalences compatible with DR, FA, and PR."""

    def inverse_precision(dr_value: float, fa_value: float, pr_value: float) -> float:
        numerator = pr_value * fa_value
        denominator = dr_value * (100.0 - pr_value) + numerator
        if denominator == 0.0:
            return 0.0
        return numerator / denominator

    candidates = [
        inverse_precision(dr_value, fa_value, pr_value)
        for dr_value in (dr.low, dr.high)
        for fa_value in (fa.low, fa.high)
        for pr_value in (pr.low, pr.high)
    ]
    return Interval(max(0.0, min(candidates)), min(1.0, max(candidates)))


def audit_row(metrics: dict[str, float]) -> dict[str, object]:
    observed = {name: outward_interval(metrics[name]) for name in METRICS}

    specificity_derived = Interval(
        100.0 - observed["FA"].high,
        100.0 - observed["FA"].low,
    )
    f1_derived = harmonic_range(observed["DR"], observed["PR"])
    balanced_acc_derived = Interval(
        (observed["DR"].low + observed["SP"].low) / 2.0,
        (observed["DR"].high + observed["SP"].high) / 2.0,
    )
    balanced_pr_derived = balanced_precision_range(observed["DR"], observed["FA"])
    precision_prevalence = precision_prevalence_range(
        observed["DR"], observed["FA"], observed["PR"]
    )
    accuracy_prevalence = accuracy_prevalence_range(
        observed["DR"], observed["FA"], observed["ACC"]
    )
    prevalence_overlap = (
        accuracy_prevalence is not None
        and precision_prevalence.intersects(accuracy_prevalence)
    )

    return {
        "printed": metrics,
        "specificity": {
            "pass": observed["SP"].intersects(specificity_derived),
            "derived_range": specificity_derived.as_list(),
        },
        "f1": {
            "pass": observed["F1"].intersects(f1_derived),
            "derived_range": f1_derived.as_list(),
        },
        "balanced_accuracy": {
            "pass": observed["ACC"].intersects(balanced_acc_derived),
            "derived_range": balanced_acc_derived.as_list(),
        },
        "balanced_precision": {
            "pass": observed["PR"].intersects(balanced_pr_derived),
            "derived_range": balanced_pr_derived.as_list(),
        },
        "any_prevalence_screen": {
            "pass": prevalence_overlap,
            "precision_implied_range": precision_prevalence.as_list(),
            "accuracy_implied_range": (
                None if accuracy_prevalence is None else accuracy_prevalence.as_list()
            ),
        },
    }


def summarize_checks(rows: Iterable[dict[str, object]]) -> dict[str, dict[str, int]]:
    names = (
        "specificity",
        "f1",
        "balanced_accuracy",
        "balanced_precision",
        "any_prevalence_screen",
    )
    rows = list(rows)
    return {
        name: {
            "pass": sum(bool(row[name]["pass"]) for row in rows),  # type: ignore[index]
            "fail": sum(not bool(row[name]["pass"]) for row in rows),  # type: ignore[index]
            "total": len(rows),
        }
        for name in names
    }


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def prose_calculations(
    tables: dict[str, dict[str, dict[str, dict[str, float]]]],
) -> dict[str, object]:
    generalized = tables["table_3"]
    customer = tables["table_4"]
    proposed = tables["table_5"]
    all_benchmarks = EXPECTED_MODELS["table_3"]
    shallow = ("random_forest", "adaboost", "arima", "svm")
    deep = ("feed_forward", "gru", "aea")
    transitions = (("p0", "p10"), ("p10", "p20"), ("p20", "p30"))

    def dr(table: dict[str, dict[str, dict[str, float]]], model: str, level: str) -> float:
        return table[model]["DR"][level]

    generalized_drop = {
        level: mean(dr(generalized, model, "p0") - dr(generalized, model, level) for model in all_benchmarks)
        for level in LEVELS[1:]
    }
    customer_drop = {
        level: mean(dr(customer, model, "p0") - dr(customer, model, level) for model in all_benchmarks)
        for level in LEVELS[1:]
    }
    generalized_advantage = {
        level: mean(dr(generalized, model, level) - dr(customer, model, level) for model in all_benchmarks)
        for level in LEVELS
    }
    deep_minus_shallow = {
        level: mean(dr(generalized, model, level) for model in deep)
        - mean(dr(generalized, model, level) for model in shallow)
        for level in LEVELS
    }
    aea_gru_feed_forward_order = {
        level: dr(generalized, "aea", level)
        > dr(generalized, "gru", level)
        > dr(generalized, "feed_forward", level)
        for level in LEVELS
    }
    sequential_drop_pp = {
        level: dr(proposed, "sequential_ensemble", "p0")
        - dr(proposed, "sequential_ensemble", level)
        for level in LEVELS[1:]
    }
    sequential_drop_relative_percent = {
        level: 100.0
        * (dr(proposed, "sequential_ensemble", "p0") - dr(proposed, "sequential_ensemble", level))
        / dr(proposed, "sequential_ensemble", "p0")
        for level in LEVELS[1:]
    }

    per_model_dr_drop = {
        table_id: {
            model: {
                level: dr(table, model, "p0") - dr(table, model, level)
                for level in LEVELS[1:]
            }
            for model in table
        }
        for table_id, table in tables.items()
    }
    mean_stepwise_dr_drop = {
        "table_3": {
            f"{start}_to_{end}": mean(
                dr(generalized, model, start) - dr(generalized, model, end)
                for model in all_benchmarks
            )
            for start, end in transitions
        },
        "table_4": {
            f"{start}_to_{end}": mean(
                dr(customer, model, start) - dr(customer, model, end)
                for model in all_benchmarks
            )
            for start, end in transitions
        },
    }
    ensemble_averaging_drop_pp = {
        level: dr(proposed, "ensemble_averaging", "p0")
        - dr(proposed, "ensemble_averaging", level)
        for level in LEVELS[1:]
    }
    ensemble_averaging_drop_relative = {
        level: 100.0
        * (
            dr(proposed, "ensemble_averaging", "p0")
            - dr(proposed, "ensemble_averaging", level)
        )
        / dr(proposed, "ensemble_averaging", "p0")
        for level in LEVELS[1:]
    }
    p30_sequential_differences = {
        baseline: {
            metric: proposed["sequential_ensemble"][metric]["p30"]
            - proposed[baseline][metric]["p30"]
            for metric in METRICS
        }
        for baseline in ("aea", "ensemble_averaging")
    }

    return {
        "per_model_dr_drop_from_p0_percentage_points": per_model_dr_drop,
        "mean_generalized_dr_drop_from_p0_percentage_points": generalized_drop,
        "mean_customer_specific_dr_drop_from_p0_percentage_points": customer_drop,
        "mean_stepwise_dr_drop_percentage_points": mean_stepwise_dr_drop,
        "mean_generalized_minus_customer_specific_dr_percentage_points": generalized_advantage,
        "mean_deep_minus_shallow_generalized_dr_percentage_points": deep_minus_shallow,
        "aea_gt_gru_gt_feed_forward_generalized_dr": aea_gru_feed_forward_order,
        "sequential_dr_drop_from_p0_percentage_points": sequential_drop_pp,
        "sequential_dr_drop_from_p0_relative_percent": sequential_drop_relative_percent,
        "ensemble_averaging_dr_drop_from_p0_percentage_points": ensemble_averaging_drop_pp,
        "ensemble_averaging_dr_drop_from_p0_relative_percent": ensemble_averaging_drop_relative,
        "p30_sequential_minus_baseline_percentage_points": p30_sequential_differences,
    }


def build_audit() -> dict[str, object]:
    tables = {table_id: load_table(table_id) for table_id in EXPECTED_MODELS}
    audited_rows: list[dict[str, object]] = []
    for table_id, table in tables.items():
        for model, metric_rows in table.items():
            for level in LEVELS:
                printed = {metric: metric_rows[metric][level] for metric in METRICS}
                audited_rows.append(
                    {
                        "table": table_id,
                        "model": model,
                        "poisoning": level,
                        **audit_row(printed),
                    }
                )

    input_paths = [
        STUDY_DIR / "METHOD.md",
        STUDY_DIR / "SOURCE_AUDIT_CONTRACT.md",
        Path(__file__).resolve(),
        *(REPORTED_DIR / f"{table_id}.csv" for table_id in ("table_2", *EXPECTED_MODELS)),
    ]
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_id": STUDY_DIR.name,
        "evidence_class": "C/N source-only",
        "pdf_sha256": PDF_SHA256,
        "rounding_half_width_percentage_points": ROUNDING_HALF_WIDTH,
        "input_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256(path) for path in input_paths
        },
        "coverage": {
            "rows": len(audited_rows),
            "metric_cells": len(audited_rows) * len(METRICS),
            "expected_rows": 68,
            "expected_metric_cells": 476,
        },
        "summary": summarize_checks(audited_rows),
        "rows": audited_rows,
        "prose_calculations": prose_calculations(tables),
        "limitations": [
            "A failed balanced identity is conditional on the paper's stated balancing procedure.",
            "The any-prevalence screen is one-sided: disjoint implied ranges prove incompatibility, but overlap does not prove an integer confusion matrix exists.",
            "AUC is not determined by one operating point and was not checked algebraically.",
            "This source-only calculation does not identify how any number was produced or establish author intent.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = build_audit()
    encoded = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
