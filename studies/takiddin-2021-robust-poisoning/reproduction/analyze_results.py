#!/usr/bin/env python3
"""Calculate this paper's metrics and verify preserved first-baseline results."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


STUDY = Path(__file__).resolve().parents[1]
METRICS = ("DR", "FA", "SP", "PR", "ACC", "F1", "AUC")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def metrics(labels, predictions, scores):
    labels, predictions, scores = map(np.asarray, (labels, predictions, scores))
    if labels.ndim != 1 or not (labels.shape == predictions.shape == scores.shape) or not len(labels):
        raise ValueError("Metrics require equally sized nonempty vectors")
    if not np.isin(labels, [0, 1]).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError("Labels and predictions must be binary")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite")
    tp = int(np.count_nonzero((labels == 1) & (predictions == 1)))
    fn = int(np.count_nonzero((labels == 1) & (predictions == 0)))
    fp = int(np.count_nonzero((labels == 0) & (predictions == 1)))
    tn = int(np.count_nonzero((labels == 0) & (predictions == 0)))
    ratio = lambda a, b: 100.0 * a / b if b else None
    dr, fa, sp = ratio(tp, tp + fn), ratio(fp, fp + tn), ratio(tn, fp + tn)
    return {
        "n": len(labels), "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "DR": dr, "FA": fa, "SP": sp, "PR": ratio(tp, tp + fp),
        "ACC": ratio(tp + tn, len(labels)),
        "F1": ratio(2 * tp, 2 * tp + fp + fn),
        "AUC": 100.0 * float(roc_auc_score(labels, scores)) if tp + fn and tn + fp else None,
        "balanced_accuracy": (dr + sp) / 2 if dr is not None and sp is not None else None,
    }


def threshold_summary(labels, scores, caps=(17.6, 33.3)):
    """Every observed-score boundary; >= threshold; ties move together."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if set(np.unique(labels)) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("ROC diagnostics require both classes and finite scores")
    false_alarm, detection, cutoffs = roc_curve(labels, scores, drop_intermediate=False)
    output = {
        "rule": "score >= threshold; null threshold denotes no alarms",
        "boundary_count": len(cutoffs),
        "best_balanced_accuracy": float(100 * np.max((detection + 1 - false_alarm) / 2)),
        "at_false_alarm_caps": {},
    }
    for cap in caps:
        allowed = np.flatnonzero(false_alarm <= cap / 100 + 1e-12)
        index = int(allowed[np.argmax(detection[allowed])])
        output["at_false_alarm_caps"][str(cap)] = {
            "DR": float(100 * detection[index]), "FA": float(100 * false_alarm[index]),
            "threshold": float(cutoffs[index]) if np.isfinite(cutoffs[index]) else None,
        }
    return output


def reported_row(rate):
    column = f"p{round(100 * rate)}"
    with (STUDY / "reported/table_3.csv").open() as stream:
        result = {r["metric"]: float(r[column]) for r in csv.DictReader(stream)
                  if r["model"] == "random_forest"}
    if set(result) != set(METRICS):
        raise ValueError("Incomplete paper target row")
    return result


def analyze_scores(arrays, training_prior):
    labels = arrays["labels"]
    predictions = arrays["predictions"]
    probability = arrays["probabilities"]
    if probability.shape != (len(labels), 2) or not np.isfinite(probability).all():
        raise ValueError("Expected finite class-0/class-1 probabilities")
    if not np.all((probability >= 0) & (probability <= 1)):
        raise ValueError("Probabilities outside [0,1]")
    np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-12, rtol=0)
    if not np.array_equal(predictions, np.argmax(probability, axis=1)):
        raise ValueError("Saved predictions disagree with library class argmax")
    scores = probability[:, 1]
    original = ~arrays["synthetic"]
    primary = metrics(labels, predictions, scores)
    prior_scores = np.full(len(labels), training_prior)
    prior_predictions = np.full(len(labels), int(training_prior > .5))
    per_attack = {}
    for attack in range(1, 7):
        selected = (labels == 1) & (arrays["attack"] == attack)
        count = int(selected.sum())
        detected = int(predictions[selected].sum())
        per_attack[str(attack)] = {"n": count, "detected": detected,
                                   "DR": 100 * detected / count if count else None}
    benign = {}
    for name, select in (("original", original), ("synthetic", ~original)):
        select = select & (labels == 0)
        count = int(select.sum())
        false_alarms = int(predictions[select].sum())
        benign[name] = {"n": count, "false_alarms": false_alarms,
                        "FA": 100 * false_alarms / count if count else None}
    return {
        "primary": primary,
        "original_rows": metrics(labels[original], predictions[original], scores[original]),
        "per_attack": per_attack, "benign_strata": benign,
        "thresholds": {
            "higher_probability": threshold_summary(labels, scores),
            "diagnostic_reversal": threshold_summary(labels, -scores),
        },
        "controls": {
            "constant_training_prior": metrics(labels, prior_predictions, prior_scores),
            "negative_daily_mean": {
                "definition": "negative mean raw daily kWh; no learned parameter",
                "AUC": 100 * float(roc_auc_score(labels, arrays["negative_daily_mean"])),
                "thresholds": threshold_summary(labels, arrays["negative_daily_mean"]),
            },
        },
    }


def audit_result(directory):
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    if result["status"] != "complete":
        raise ValueError("Only a completed fit can be audited")
    for relative, expected in result.get("source_sha256", {}).items():
        path = (STUDY / relative).resolve()
        if not path.is_relative_to(STUDY) or digest(path) != expected:
            raise ValueError(f"Analysis/source revision mismatch: {relative}")
    for name, expected in result["output_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Output hash mismatch: {name}")
    with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    recomputed = analyze_scores(arrays, result["training_prior"])
    if recomputed != result["analysis"]:
        raise ValueError("Saved metrics/diagnostics differ from saved predictions")
    target = reported_row(result["poisoning_rate"])
    return {
        "status": "verified", "case": result["case"], "code_commit": result["code_commit"],
        "scope": result["scope"], "reported_full_data_context": target,
        "pilot_primary": recomputed["primary"],
        "pilot_minus_reported_percentage_points": {
            name: recomputed["primary"][name] - target[name]
            if recomputed["primary"][name] is not None else None for name in METRICS
        },
        "result_sha256": digest(directory / "result.json"),
        "comparison_is_reproduction": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {"scope": "first-baseline pilot, not full Table III reproduction",
               "attempts": [audit_result(path) for path in args.attempt]}
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(encoded)
    print(encoded, end="")
