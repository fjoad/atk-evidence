"""Check whether the paper's tabulated metrics can share one test set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TABLES = {
    "SGCC": [
        ("FC-SAE", 83.0, 14.0, 83.0),
        ("LSTM-SAE", 86.0, 12.0, 87.0),
        ("FC-VAE", 90.0, 9.0, 91.0),
        ("LSTM-VAE", 93.0, 6.0, 93.0),
        ("LSTM-AEA", 96.0, 4.0, 95.0),
        ("Naive Bayes", 75.0, 16.0, 75.0),
        ("ARIMA", 88.0, 10.0, 87.0),
        ("Single-class SVM", 91.0, 8.5, 90.0),
        ("Feed forward", 91.0, 9.5, 90.0),
        ("LSTM", 91.5, 9.0, 90.5),
        ("Multi-class SVM", 92.0, 7.5, 91.0),
    ],
    "ISET": [
        ("FC-SAE", 81.0, 15.0, 81.0),
        ("LSTM-SAE", 85.0, 13.0, 85.0),
        ("FC-VAE", 88.0, 11.0, 89.0),
        ("LSTM-VAE", 91.0, 7.0, 91.0),
        ("LSTM-AEA", 94.0, 5.0, 93.0),
        ("Naive Bayes", 73.0, 18.0, 73.0),
        ("ARIMA", 86.0, 12.0, 86.0),
        ("Single-class SVM", 90.0, 9.0, 89.0),
        ("Feed forward", 90.0, 11.0, 89.0),
        ("LSTM", 90.5, 10.0, 89.5),
        ("Multi-class SVM", 91.0, 8.0, 90.0),
    ],
}


def expected_precision(tpr: float, fpr: float, prevalence: float) -> float:
    numerator = tpr * prevalence
    denominator = numerator + fpr * (1.0 - prevalence)
    return numerator / denominator if denominator else float("nan")


def implied_prevalence(tpr: float, fpr: float, precision: float) -> float:
    numerator = precision * fpr
    denominator = tpr * (1.0 - precision) + precision * fpr
    return numerator / denominator if denominator else float("nan")


def audit_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for dataset, metrics in TABLES.items():
        for model, dr_pct, fa_pct, precision_pct in metrics:
            dr = dr_pct / 100.0
            fa = fa_pct / 100.0
            precision = precision_pct / 100.0
            balanced_precision = expected_precision(dr, fa, 0.5)
            prevalence = implied_prevalence(dr, fa, precision)
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "reported_dr_pct": dr_pct,
                    "reported_fa_pct": fa_pct,
                    "reported_precision_pct": precision_pct,
                    "balanced_expected_precision_pct": 100.0 * balanced_precision,
                    "precision_gap_pp": precision_pct - 100.0 * balanced_precision,
                    "implied_positive_prevalence_pct": 100.0 * prevalence,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    rows = audit_rows()

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    for row in rows:
        print(
            f"{row['dataset']:4s} {row['model']:18s} "
            f"reported_PR={row['reported_precision_pct']:5.1f}% "
            f"balanced_expected={row['balanced_expected_precision_pct']:5.1f}% "
            f"gap={row['precision_gap_pp']:+5.1f}pp "
            f"implied_prevalence={row['implied_positive_prevalence_pct']:5.1f}%"
        )


if __name__ == "__main__":
    main()

