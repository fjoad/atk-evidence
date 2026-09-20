#!/usr/bin/env python3
"""Fit one declared shallow detector on one preserved preparation, on Slurm only."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import time
import warnings

import joblib
import numpy as np
import sklearn
import scipy
import threadpoolctl

from models import adaboost, random_forest, svm
from analyze_results import analyze_scores, digest


STUDY = Path(__file__).resolve().parents[1]
EXPECTED_METADATA = {
    "generalized-two-class-p00": "1c2e5ebfee9584c160fee209851e7f685d8c8d7971922d62a63960bc6308c3de",
    "generalized-two-class-p30": "9ac0faeea220e62ed5af13e39deb174a190cc69749786c38ea58240e5031f21a",
}
INPUTS = ("train_x", "train_true_y", "train_observed_y", "test_x", "test_raw_x",
          "test_true_y", "test_observed_y", "test_uid", "test_synthetic", "test_attack")


def load_preparation(directory):
    directory = Path(directory)
    if directory.name not in EXPECTED_METADATA:
        raise ValueError("This initial runner accepts only the two frozen pilot cases")
    meta_path = directory / "metadata.json"
    if digest(meta_path) != EXPECTED_METADATA[directory.name]:
        raise ValueError("Prepared metadata differs from the frozen pilot")
    metadata = json.loads(meta_path.read_text())
    if metadata["mode"] != "two-class" or metadata["scope"] != "generalized":
        raise ValueError("This pilot requires the generalized two-class preparation")
    arrays, identities = {}, {}
    for name in INPUTS:
        path = directory / (name + ".npy")
        expected = metadata["files"][path.name]
        actual_hash = digest(path)
        if actual_hash != expected["sha256"]:
            raise ValueError(f"Input hash mismatch: {name}")
        value = np.load(path, allow_pickle=False)
        if list(value.shape) != expected["shape"] or str(value.dtype) != expected["dtype"] or not np.isfinite(value).all():
            raise ValueError(f"Invalid prepared array: {name}")
        arrays[name], identities[name] = value, actual_hash
    if arrays["train_x"].shape != (4464, 48) or arrays["test_x"].shape != (2232, 48):
        raise ValueError("Frozen pilot cardinality mismatch")
    if not np.array_equal(arrays["test_true_y"], arrays["test_observed_y"]):
        raise ValueError("Test labels were corrupted")
    if set(np.unique(arrays["train_observed_y"])) != {0, 1}:
        raise ValueError("Fitting requires both observed classes")
    return arrays, metadata, identities


def fit_svm(arrays, output, *, seed):
    """Native labels plus raw margins; no extra probability-calibration fits."""
    output = Path(output)
    model = svm(seed)
    tick = time.perf_counter()
    model.fit(arrays["train_x"], arrays["train_observed_y"])
    fit_seconds = time.perf_counter() - tick
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("Unexpected SVM class order")
    tick = time.perf_counter()
    margins = model.decision_function(arrays["test_x"])
    predictions = model.predict(arrays["test_x"])
    train_margins = model.decision_function(arrays["train_x"])
    train_predictions = model.predict(arrays["train_x"])
    score_seconds = time.perf_counter() - tick
    model_path = output / "model.joblib"
    joblib.dump(model, model_path)
    tick = time.perf_counter()
    restored = joblib.load(model_path)  # Only our just-written artifact.
    np.testing.assert_array_equal(margins, restored.decision_function(arrays["test_x"]))
    np.testing.assert_array_equal(predictions, restored.predict(arrays["test_x"]))
    reload_seconds = time.perf_counter() - tick
    saved = {"decision_scores": margins, "predictions": predictions,
             "labels": arrays["test_true_y"], "uid": arrays["test_uid"],
             "synthetic": arrays["test_synthetic"], "attack": arrays["test_attack"],
             "negative_daily_mean": -arrays["test_raw_x"].mean(axis=1, dtype=np.float64)}
    np.savez_compressed(output / "predictions.npz", **saved)
    prior = float(arrays["train_observed_y"].mean())
    caps = (10.2, 17.6, 25.7, 33.3)
    report = {
        "training_prior": prior, "fitting_parameters": model.get_params(),
        "scoring_workers": 1, "false_alarm_caps": list(caps),
        "score_definition": "native SVC.decision_function; class 1 if margin >= 0; not a probability",
        "timing_seconds": {"fit": fit_seconds, "score": score_seconds, "reload": reload_seconds},
        "reload": {"identical_decision_scores": True, "identical_predictions": True},
        "training": {
            "observed_label_accuracy": 100 * float(np.mean(train_predictions == arrays["train_observed_y"])),
            "true_label_accuracy": 100 * float(np.mean(train_predictions == arrays["train_true_y"])),
            "distinct_decision_scores": len(np.unique(train_margins)),
        },
        "svm": {"fit_status": int(model.fit_status_), "iterations": model.n_iter_.tolist(),
                "support_vectors_per_class": model.n_support_.tolist(),
                "gamma_numeric": float(model._gamma), "gamma_auto_context": 1. / arrays["train_x"].shape[1],
                "training_variance_float64": float(arrays["train_x"].astype(np.float64).var()),
                "intercept": model.intercept_.tolist(), "probability_calibration": False,
                "test_margin_min": float(margins.min()), "test_margin_max": float(margins.max()),
                "zero_margin_test_rows": int(np.count_nonzero(margins == 0)),
                "distinct_test_margins": len(np.unique(margins))},
        "analysis": analyze_scores(saved, prior, caps),
        "output_sha256": {name: digest(output / name) for name in ("model.joblib", "predictions.npz")},
    }
    return report


def fit_one(arrays, output, *, seed, workers, model_name="random_forest"):
    """Also used on constructed fixtures; real input is gated by main()."""
    output = Path(output)
    if model_name == "svm":
        return fit_svm(arrays, output, seed=seed)
    if model_name not in ("random_forest", "adaboost"):
        raise ValueError("Unknown model")
    model = random_forest(seed, workers) if model_name == "random_forest" else adaboost(seed)
    fitting_parameters = model.get_params()
    clock = time.perf_counter()
    # True labels must never be supplied to fit() under poisoned training.
    model.fit(arrays["train_x"], arrays["train_observed_y"])
    fit_seconds = time.perf_counter() - clock
    valid_trees = (len(model.estimators_) == 100 if model_name == "random_forest"
                   else 1 <= len(model.estimators_) <= 50)
    if not np.array_equal(model.classes_, [0, 1]) or not valid_trees:
        raise ValueError("Unexpected fitted class or tree inventory")
    if model_name == "random_forest":
        model.set_params(n_jobs=1)
    clock = time.perf_counter()
    probabilities = model.predict_proba(arrays["test_x"])
    predictions = model.predict(arrays["test_x"])
    train_probabilities = model.predict_proba(arrays["train_x"])
    train_predictions = model.predict(arrays["train_x"])
    score_seconds = time.perf_counter() - clock
    model_path = output / "model.joblib"
    joblib.dump(model, model_path)
    clock = time.perf_counter()
    restored = joblib.load(model_path)  # Only our just-written model, never third-party pickle.
    restored_probabilities = restored.predict_proba(arrays["test_x"])
    restored_predictions = restored.predict(arrays["test_x"])
    np.testing.assert_array_equal(probabilities, restored_probabilities)
    np.testing.assert_array_equal(predictions, restored_predictions)
    reload_seconds = time.perf_counter() - clock
    saved = {
        "probabilities": probabilities, "predictions": predictions,
        "labels": arrays["test_true_y"], "uid": arrays["test_uid"],
        "synthetic": arrays["test_synthetic"], "attack": arrays["test_attack"],
        "negative_daily_mean": -arrays["test_raw_x"].mean(axis=1, dtype=np.float64),
    }
    np.savez_compressed(output / "predictions.npz", **saved)
    prior = float(arrays["train_observed_y"].mean())
    caps = (14.1, 17.6, 29.9, 33.3) if model_name == "adaboost" else (17.6, 33.3)
    report = {
        "training_prior": prior, "fitting_parameters": fitting_parameters,
        "scoring_workers": 1,
        "timing_seconds": {"fit": fit_seconds, "score": score_seconds, "reload": reload_seconds},
        "reload": {"identical_probabilities": True, "identical_predictions": True},
        "training": {
            "observed_label_accuracy": 100 * float(np.mean(train_predictions == arrays["train_observed_y"])),
            "true_label_accuracy": 100 * float(np.mean(train_predictions == arrays["train_true_y"])),
            "distinct_positive_probabilities": len(np.unique(train_probabilities[:, 1])),
        },
        "forest": {"trees": len(model.estimators_),
                   "depth_min": min(tree.tree_.max_depth for tree in model.estimators_),
                   "depth_max": max(tree.tree_.max_depth for tree in model.estimators_),
                   "total_nodes": sum(tree.tree_.node_count for tree in model.estimators_)},
        "analysis": analyze_scores(saved, prior, caps),
        "output_sha256": {name: digest(output / name) for name in ("model.joblib", "predictions.npz")},
    }
    if model_name == "adaboost":
        report["false_alarm_caps"] = list(caps)
        report["boosting"] = report.pop("forest")
        count = len(model.estimators_)
        errors, weights = model.estimator_errors_[:count], model.estimator_weights_[:count]
        if (not np.isfinite(errors).all() or not np.isfinite(weights).all()
                or report["boosting"]["depth_max"] > 1):
            raise ValueError("Invalid fitted AdaBoost weak learners")
        report["boosting"].update(algorithm=model.algorithm,
            base_estimator_parameters=model.estimator_.get_params(),
            estimator_errors=errors.tolist(), estimator_weights=weights.tolist(),
            stopped_before_maximum=count < 50)
    return report


def require_compute():
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_JOB_NODELIST"):
        raise RuntimeError("Real-data fitting and scoring require a Slurm compute allocation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("random_forest", "adaboost", "svm"), default="random_forest")
    args = parser.parse_args()
    require_compute()
    required_version = "1.5.2" if args.model == "adaboost" else "1.9.0"
    if sklearn.__version__ != required_version:
        raise RuntimeError(f"{args.model} is frozen to scikit-learn {required_version}")
    if args.model == "adaboost" and (np.__version__, scipy.__version__, joblib.__version__, threadpoolctl.__version__) != ("1.26.4", "1.13.1", "1.4.2", "3.5.0"):
        raise RuntimeError("AdaBoost numerical dependencies differ from the frozen environment")
    if args.model == "svm" and (np.__version__, scipy.__version__, joblib.__version__, threadpoolctl.__version__) != ("2.5.1", "1.18.0", "1.5.3", "3.6.0"):
        raise RuntimeError("SVM numerical dependencies differ from the frozen environment")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result_path = args.output / "result.json"
    record = {
        "status": "started", "model": args.model, "case": args.data.name,
        "scope": "20-customer 28-day exploratory pilot; not full Table III reproduction",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": os.environ.get("EXPECTED_COMMIT"), "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "seed": 20260920, "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                       "sklearn": sklearn.__version__, "joblib": joblib.__version__,
                                       "scipy": scipy.__version__, "threadpoolctl": threadpoolctl.__version__},
        "source_sha256": {str(path.relative_to(STUDY)): digest(path) for path in (
            Path(__file__), Path(__file__).with_name("models.py"),
            Path(__file__).with_name("analyze_results.py"),
            STUDY / {"random_forest": "FIRST_BASELINE.md", "adaboost": "ADABOOST_PILOT.md", "svm": "SVM_PILOT.md"}[args.model],
            STUDY / "reported/table_3.csv",
        )},
    }
    if args.model == "adaboost":
        record["source_sha256"]["requirements-adaboost.txt"] = digest(STUDY / "requirements-adaboost.txt")
    if args.model == "svm":
        record["source_sha256"]["requirements-svm.txt"] = digest(STUDY / "requirements-svm.txt")
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    captured = []
    try:
        tick = time.perf_counter()
        arrays, metadata, identities = load_preparation(args.data)
        record["input_load_seconds"] = time.perf_counter() - tick
        record.update(input_sha256=identities, preparation_metadata_sha256=digest(args.data / "metadata.json"),
                      poisoning_rate=metadata["rate"], actual_poisoning=metadata["poisoning"],
                      preparation_overlap=metadata["synthetic_parent_crossings"],
                      training_rows=len(arrays["train_x"]), test_rows=len(arrays["test_x"]))
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            record.update(fit_one(arrays, args.output, seed=20260920, workers=4, model_name=args.model))
        record["warnings"] = [{"category": w.category.__name__, "message": str(w.message)} for w in captured]
        if args.model == "svm" and record["svm"]["fit_status"] != 0:
            raise RuntimeError("SVM did not converge; preserve artifacts but do not label the fit complete")
        record["status"] = "complete"
    except Exception as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        record["warnings"] = [{"category": w.category.__name__, "message": str(w.message)} for w in captured]
        record["elapsed_seconds"] = time.perf_counter() - started
        record["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result_path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"case": record["case"], "status": record["status"],
                      "metrics": record["analysis"]["primary"], "seconds": record["timing_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
