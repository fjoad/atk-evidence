#!/usr/bin/env python3
"""Fit one declared detector on one preserved preparation, on Slurm only."""

import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time
import warnings

import joblib
import numpy as np
import sklearn
import scipy
import threadpoolctl

from models import adaboost, random_forest, svm, feed_forward
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
    if model_name == "feed_forward":
        return fit_feed_forward(arrays, output, seed=seed)
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


def weight_hash(weights):
    value = hashlib.sha256()
    for weight in weights:
        array = np.asarray(weight)
        value.update(str(array.shape).encode())
        value.update(str(array.dtype).encode())
        value.update(array.tobytes(order="C"))
    return value.hexdigest()


def configure_tensorflow(require_gpu=False):
    import tensorflow as tf
    import keras
    import h5py
    if tf.__version__ != "2.16.2" or keras.__version__ != "3.4.1" or h5py.__version__ != "3.11.0" or keras.backend.backend() != "tensorflow":
        raise RuntimeError("Use the pinned TensorFlow feed-forward environment")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.experimental.enable_op_determinism()
    devices = tf.config.list_physical_devices("GPU")
    if require_gpu and len(devices) != 1:
        raise RuntimeError("Feed-forward research fitting requires exactly one allocated GPU")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    details = [tf.config.experimental.get_device_details(device) for device in devices]
    if require_gpu and "V100" not in details[0].get("device_name", ""):
        raise RuntimeError("Frozen feed-forward pilot requires one V100")
    tf.config.set_soft_device_placement(False)
    selected = "/GPU:0" if require_gpu else "/CPU:0"
    with tf.device(selected):
        probe = tf.Variable(tf.ones((2, 2)))
        with tf.GradientTape() as tape:
            probe_loss = tf.reduce_sum(tf.matmul(probe, tf.ones((2, 2))))
        probe.assign_sub(.01 * tape.gradient(probe_loss, probe))
        if require_gpu and "GPU:0" not in probe.device:
            raise RuntimeError("Constructed GPU update was placed on CPU")
        np.testing.assert_allclose(probe.numpy(), .98, atol=1e-7)
    info = {"tensorflow": tf.__version__, "keras": keras.__version__, "h5py": h5py.__version__, "backend": keras.backend.backend(),
            "gpu_details": details, "build_info": tf.sysconfig.get_build_info(),
            "constructed_update_device": probe.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "deterministic_ops": True, "tf32_enabled": False, "mixed_precision": False}
    if require_gpu:
        check = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                "--format=csv,noheader"], capture_output=True, text=True, check=True)
        info["node_gpu_inventory"] = check.stdout.strip().splitlines()
    return info


def fit_feed_forward(arrays, output, *, seed, epochs=50, batch_size=100, fit_limit_seconds=420):
    """Fixed final-epoch fit. Overrides are for constructed software fixtures only."""
    import tensorflow as tf
    import keras
    output = Path(output)
    gpu = bool(tf.config.list_physical_devices("GPU"))
    device = "/GPU:0" if gpu else "/CPU:0"
    with tf.device(device):
        model = feed_forward(seed)
    weight_devices = sorted({variable.value.device for variable in model.trainable_variables})
    if gpu and any("GPU:0" not in placement for placement in weight_devices):
        raise RuntimeError("Neural weights are not on the allocated GPU")
    if model.count_params() != 1277501:
        raise ValueError("Feed-forward layer inventory differs from the frozen source completion")
    initial = model.get_weights()
    initial_hash = weight_hash(initial)
    np.savez_compressed(output / "initial_weights.npz", **{f"weight_{i}": v for i, v in enumerate(initial)})
    dataset = tf.data.Dataset.from_tensor_slices((arrays["train_x"], arrays["train_observed_y"].astype(np.float32)[:, None]))
    dataset = dataset.shuffle(len(arrays["train_x"]), seed=seed, reshuffle_each_iteration=True).batch(batch_size)
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    dataset = dataset.with_options(options).prefetch(1)
    history, start = [], time.perf_counter()

    class Recorder(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_started = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            row = {"epoch": epoch + 1, "seconds": time.perf_counter() - self.epoch_started,
                   "loss": float(logs["loss"]), "binary_accuracy": float(logs["binary_accuracy"]),
                   "optimizer_updates": int(self.model.optimizer.iterations.numpy())}
            if not all(np.isfinite(row[key]) for key in ("loss", "binary_accuracy")):
                raise ValueError("Nonfinite neural training history")
            history.append(row)
            (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(json.dumps({"training_epoch": row}), flush=True)
            if time.perf_counter() - start >= fit_limit_seconds:
                self.model.stop_training = True

    if gpu:
        tf.config.experimental.reset_memory_stats("GPU:0")
    model.fit(dataset, epochs=epochs, verbose=0, callbacks=[Recorder()])
    fit_seconds = time.perf_counter() - start
    updates = int(model.optimizer.iterations.numpy())
    expected_updates = len(history) * ((len(arrays["train_x"]) + batch_size - 1) // batch_size)
    if updates != expected_updates:
        raise ValueError("Unexpected number of training updates")
    def infer(network, x):
        return np.concatenate([network(tf.convert_to_tensor(x[i:i + batch_size]), training=False).numpy().ravel()
                               for i in range(0, len(x), batch_size)])
    tick = time.perf_counter()
    probability = infer(model, arrays["test_x"])
    train_probability = infer(model, arrays["train_x"])
    predictions, train_predictions = (probability > .5).astype(np.int64), (train_probability > .5).astype(np.int64)
    score_seconds = time.perf_counter() - tick
    with tf.device(device):
        inference_device = model(tf.convert_to_tensor(arrays["test_x"][:1]), training=False).device
    model.save(output / "model.keras")
    tick = time.perf_counter()
    with tf.device(device):
        restored = keras.models.load_model(output / "model.keras", compile=False)
    np.testing.assert_array_equal(probability, infer(restored, arrays["test_x"]))
    reload_seconds = time.perf_counter() - tick
    final = model.get_weights()
    final_hash = weight_hash(final)
    if final_hash == initial_hash:
        raise ValueError("No model weights changed")
    norm_max = [float(np.linalg.norm(layer.kernel.numpy().astype(np.float64), axis=0).max()) for layer in model.layers]
    if max(norm_max) > 3.00001:
        raise ValueError("MaxNorm constraint not respected")
    p64 = probability.astype(np.float64)
    saved = {"probabilities": np.column_stack((1 - p64, p64)), "predictions": predictions,
             "labels": arrays["test_true_y"], "uid": arrays["test_uid"], "attack": arrays["test_attack"],
             "synthetic": arrays["test_synthetic"],
             "negative_daily_mean": -arrays["test_raw_x"].mean(axis=1, dtype=np.float64)}
    np.savez_compressed(output / "predictions.npz", **saved)
    caps = (9.3, 17.6, 24.4, 33.3)
    prior = float(arrays["train_observed_y"].mean())
    return {"training_prior": prior, "false_alarm_caps": list(caps),
        "fitting_parameters": {"hidden_layers": 6, "hidden_width": 500, "hidden_activation": "relu",
            "output_units": 1, "output_activation": "sigmoid", "loss": "standard_binary_crossentropy_repair",
            "optimizer": "Adamax", "learning_rate": .002, "beta_1": .9, "beta_2": .999, "epsilon": 1e-7,
            "kernel_maxnorm": 3., "constraint_axis": 0, "dropout": 0., "epochs": epochs,
            "batch_size": batch_size, "shuffle": True, "seed": seed, "dtype": "float32"},
        "neural": {"parameters": model.count_params(), "epochs_completed": len(history),
            "optimizer_updates": updates, "training_complete": len(history) == epochs,
            "initial_weights_sha256": initial_hash, "final_weights_sha256": final_hash,
            "maximum_incoming_norm_by_layer": norm_max, "model_json": json.loads(model.to_json()),
            "weight_devices": weight_devices,
            "optimizer_config": model.optimizer.get_config(), "inference_device": inference_device,
            "gpu_allocator_bytes": tf.config.experimental.get_memory_info("GPU:0") if gpu else None},
        "timing_seconds": {"fit": fit_seconds, "score": score_seconds, "reload": reload_seconds},
        "reload": {"identical_probabilities": True, "identical_predictions": True},
        "training": {"observed_label_accuracy": 100 * float(np.mean(train_predictions == arrays["train_observed_y"])),
            "true_label_accuracy": 100 * float(np.mean(train_predictions == arrays["train_true_y"])),
            "distinct_positive_probabilities": len(np.unique(train_probability))},
        "analysis": analyze_scores(saved, prior, caps),
        "output_sha256": {name: digest(output / name) for name in ("model.keras", "initial_weights.npz", "history.json", "predictions.npz")}}


def require_compute():
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_JOB_NODELIST"):
        raise RuntimeError("Real-data fitting and scoring require a Slurm compute allocation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("random_forest", "adaboost", "svm", "feed_forward"), default="random_forest")
    args = parser.parse_args()
    require_compute()
    required_version = "1.5.2" if args.model in ("adaboost", "feed_forward") else "1.9.0"
    if sklearn.__version__ != required_version:
        raise RuntimeError(f"{args.model} is frozen to scikit-learn {required_version}")
    if args.model in ("adaboost", "feed_forward") and (np.__version__, scipy.__version__, joblib.__version__, threadpoolctl.__version__) != ("1.26.4", "1.13.1", "1.4.2", "3.5.0"):
        raise RuntimeError("AdaBoost/neural numerical dependencies differ from the frozen environment")
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
            STUDY / {"random_forest": "FIRST_BASELINE.md", "adaboost": "ADABOOST_PILOT.md", "svm": "SVM_PILOT.md", "feed_forward": "FEED_FORWARD_PILOT.md"}[args.model],
            STUDY / "reported/table_3.csv",
        )},
    }
    if args.model == "adaboost":
        record["source_sha256"]["requirements-adaboost.txt"] = digest(STUDY / "requirements-adaboost.txt")
    if args.model == "svm":
        record["source_sha256"]["requirements-svm.txt"] = digest(STUDY / "requirements-svm.txt")
    if args.model == "feed_forward":
        record["source_sha256"]["requirements-feed-forward.txt"] = digest(STUDY / "requirements-feed-forward.txt")
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    captured = []
    try:
        if args.model == "feed_forward":
            record["hardware"] = configure_tensorflow(require_gpu=True)
            record["versions"].update({key: record["hardware"][key] for key in ("tensorflow", "keras", "h5py", "backend")})
            result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
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
        if args.model == "feed_forward" and not record["neural"]["training_complete"]:
            record.update(status="partial", stop_reason="seven-minute training guard reached")
        else:
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
    return 0 if record["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
