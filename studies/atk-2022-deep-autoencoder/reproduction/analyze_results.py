#!/usr/bin/env python3
"""Aggregate every compact-route attempt without selecting a favorite seed."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


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


def effective_eligibility(attempt: dict[str, object]) -> str:
    config = attempt["configuration"]
    return (
        "exploratory_paper_literal_P0"
        if config["test_view"] == "adasyn"
        else "exploratory_interpretation_I-ADASYN-NONE"
    )


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
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(successes: list[dict[str, object]]) -> dict[str, object]:
    individual: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for attempt in successes:
        config = attempt["configuration"]
        eligibility = effective_eligibility(attempt)
        row = {
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
        if not config["table_v"]:
            grouped[
                (
                    str(config["method"]),
                    str(config["model"]),
                    str(config["train_fraction"]),
                )
            ].append(attempt)

    summary: list[dict[str, object]] = []
    for (method, model, fraction), attempts in sorted(grouped.items()):
        reported = attempts[0]["reported_table_3"]
        row: dict[str, object] = {
            "method": method,
            "eligibility": effective_eligibility(attempts[0]),
            "model": model,
            "train_fraction": fraction,
            "test_view": attempts[0]["configuration"]["test_view"],
            "successful_seeds": len(attempts),
            "seeds": ";".join(
                str(item["configuration"]["seed"])
                for item in sorted(
                    attempts, key=lambda item: int(item["configuration"]["seed"])
                )
            ),
        }
        for metric in METRICS:
            values = [float(item["metrics"][metric]) for item in attempts]
            mean, sd = mean_sd(values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_reported"] = reported[metric]
            row[f"{metric}_difference"] = mean - float(reported[metric])
        summary.append(row)

    table_v_rows: list[dict[str, object]] = []
    for attempt in successes:
        if attempt.get("table_v") is None:
            continue
        config = attempt["configuration"]
        for entry in attempt["table_v"]:
            table_v_rows.append(
                {
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
            )
    return {
        "individual": individual,
        "table_3_summary": summary,
        "table_5_individual": table_v_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    successes, failures = load_attempts(args.results)
    tables = aggregate(successes)
    write_csv(args.output / "table_3_individual.csv", tables["individual"])
    write_csv(args.output / "table_3_summary.csv", tables["table_3_summary"])
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
