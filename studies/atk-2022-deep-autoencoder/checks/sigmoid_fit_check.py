"""One capped paired fit, not a universal impossibility test; see SIGMOID_FIT_CHECK.md."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
import numpy as np
import torch

from post_anchor_diagnostics import RESULT_SHA, digest
from sigmoid_sanity import control_summary, verify_identities


SEED = 20260831
BATCH = 32
MODEL_SHA = "3515415082b26bb91cb5367effbd1eba4324bf250ec47e799f7fccb3e6df83f0"
MODEL_PATH = Path(__file__).resolve().parent.parent / "reproduction/models.py"
spec = importlib.util.spec_from_file_location("sigmoid_fit_source_models", MODEL_PATH)
paper_models = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = paper_models
spec.loader.exec_module(paper_models)


def weight_digest(model):
    checksum = hashlib.sha256()
    for weight in model.get_weights():
        checksum.update(str((weight.shape, str(weight.dtype))).encode())
        checksum.update(weight.tobytes())
    return checksum.hexdigest()


def build_model(activation):
    if activation not in ("softmax", "sigmoid"):
        raise ValueError(activation)
    template = paper_models.build_fc_sae(seed=SEED)
    config = copy.deepcopy(template.get_config())
    for index, layer in enumerate(config["layers"]):
        if layer["class_name"] == "Dropout":
            layer["config"]["seed"] = SEED + index
        if layer["config"]["name"] == "reconstruction":
            layer["config"]["activation"] = activation
    paper_models.set_seed(SEED)
    model = keras.Model.from_config(config)
    model.set_weights(template.get_weights())
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss="mean_squared_error", jit_compile=False)
    paper_models.validate_fc_sae(model, output_activation=activation)
    return model


def select_indices(train_count, base, total, stage):
    fit_count, cal_count, days_count = ((128, 64, 8) if stage == "pilot" else (2048, 1024, 1024))
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(train_count, fit_count + cal_count, replace=False)
    days = np.sort(rng.choice(base, days_count, replace=False))
    synthetic_count = round(days_count * (total - 7 * base) / base)
    synthetic = np.sort(rng.choice(total - 7 * base, synthetic_count, replace=False)) + 7 * base
    return {"fit": chosen[:fit_count], "calibration": chosen[fit_count:],
            "test": np.concatenate([days + group * base for group in range(7)] + [synthetic])}, 7 * days_count


def calibration_cutoff(errors, reverse=False):
    return float(np.quantile(errors, .15 if reverse else .85, method="lower" if reverse else "higher"))


def cutoff_metrics(y, errors, threshold, reverse, auc):
    flagged, positive = (errors < threshold if reverse else errors > threshold), y == 1
    tp, fp = int(np.sum(flagged & positive)), int(np.sum(flagged & ~positive))
    fn, tn = int(np.sum(~flagged & positive)), int(np.sum(~flagged & ~positive))
    dr, fa = tp / (tp + fn), fp / (fp + tn)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "DR": 100 * dr,
            "FA": 100 * fa, "SP": 100 * (1 - fa), "ACC": 50 * (dr + 1 - fa),
            "PR": 100 * tp / (tp + fp) if tp + fp else 0.,
            "F1": 200 * tp / (2 * tp + fp + fn), "AUC": auc}


def summarize(y, errors, calibration, original):
    result = {}
    for view, rows in (("sampled_prepared", slice(None)), ("original", slice(0, original))):
        result[view] = {}
        for direction, reverse in (("printed", False), ("reversed_control", True)):
            details = control_summary(y[rows], errors[rows], reverse=reverse)
            threshold = calibration_cutoff(calibration, reverse)
            details["calibrated_cutoff"] = threshold
            details["calibrated_metrics"] = cutoff_metrics(
                y[rows], errors[rows], threshold, reverse, details["fixed_cutoff"]["AUC"])
            result[view][direction] = details
    return result


def score(model, values, activation):
    scores = np.empty(len(values), dtype=np.float64)
    min_output, max_output, max_sum_error = 1., 0., 0.
    with torch.no_grad():
        for start in range(0, len(values), 1024):
            inputs = values[start:start + 1024]
            output = keras.ops.convert_to_numpy(model(inputs, training=False))
            assert output.shape == inputs.shape and np.isfinite(output).all()
            assert np.all((output >= 0) & (output <= 1))
            min_output, max_output = min(min_output, float(output.min())), max(max_output, float(output.max()))
            if activation == "softmax":
                max_sum_error = max(max_sum_error, float(np.max(np.abs(output.sum(axis=1) - 1))))
                assert max_sum_error < 1e-5
            scores[start:start + len(inputs)] = np.mean((inputs.astype(np.float64) - output.astype(np.float64)) ** 2, axis=1)
    assert np.isfinite(scores).all()
    return scores, {"min": min_output, "max": max_output, "softmax_sum_error": max_sum_error}


class FitTrace(keras.callbacks.Callback):
    def __init__(self, cap_seconds=90):
        super().__init__()
        self.cap_seconds, self.epochs, self.batch_seconds = cap_seconds, [], []
        self.best_loss, self.best_weights, self.best_epoch = float("inf"), None, None
        self.budget_stopped = False

    def on_train_begin(self, logs=None):
        self.started = time.perf_counter()

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_started = time.perf_counter()

    def on_train_batch_end(self, batch, logs=None):
        self.batch_seconds.append(time.perf_counter() - self.batch_started)
        if not np.isfinite(float(logs["loss"])):
            raise FloatingPointError("Nonfinite fit loss")
        if time.perf_counter() - self.started >= self.cap_seconds:
            self.budget_stopped = True
            self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None):
        entry = {key: float(value) for key, value in logs.items()}
        if not all(np.isfinite(value) for value in entry.values()):
            raise FloatingPointError("Nonfinite calibration/fit loss")
        self.epochs.append({"epoch": epoch + 1, **entry})
        if entry["val_loss"] < self.best_loss:
            self.best_loss, self.best_epoch = entry["val_loss"], epoch + 1
            self.best_weights = self.model.get_weights()


def run(args, record):
    started = time.perf_counter()
    assert digest(MODEL_PATH) == MODEL_SHA
    assert digest(args.result) == RESULT_SHA
    result = json.loads(args.result.read_text())
    data = args.root / result["data"]["path"]
    assert digest(data / "metadata.json") == result["data"]["metadata_sha256"]
    record["metadata_sha256"] = result["data"]["metadata_sha256"]
    metadata = json.loads((data / "metadata.json").read_text())
    expected = {key: value["sha256"] for key, value in metadata["files"].items()}
    expected.update(result["data"]["files"])
    names = ("x_train.npy", "x_test.npy", "y_test.npy", "test_attack_id.npy", "test_source_row.npy")
    record["input_sha256"] = {name: digest(data / name) for name in names}
    assert all(value == expected[name] for name, value in record["input_sha256"].items())
    arrays = {name: np.load(data / name, mmap_mode="r") for name in names}
    assert arrays["x_train.npy"].shape == (1500523, 48)
    assert arrays["x_test.npy"].shape == (8884989, 48)
    indices, original = select_indices(1500523, 750767, 8884989, args.stage)
    fit, calibration = [np.asarray(arrays["x_train.npy"][indices[key]], dtype=np.float32) for key in ("fit", "calibration")]
    test = np.asarray(arrays["x_test.npy"][indices["test"]], dtype=np.float32)
    labels, attacks, sources = [np.asarray(arrays[key][indices["test"]]) for key in names[2:]]
    verify_identities(labels, attacks, sources, original)
    assert not np.intersect1d(indices["fit"], indices["calibration"]).size
    assert all(np.isfinite(values).all() for values in (fit, calibration, test))
    np.savez(args.output / "selection.npz", **indices)
    record["population"] = {"fit": len(fit), "calibration": len(calibration), "test": len(test), "original_test": original}
    record["verification_seconds"] = time.perf_counter() - started
    record["models"] = {}
    epochs = 1 if args.stage == "pilot" else 10
    initial_digests = []
    for activation in ("softmax", "sigmoid"):
        model = build_model(activation)
        initial_digests.append(weight_digest(model))
        assert initial_digests[-1] == initial_digests[0]
        record["models"][activation] = details = {"initial_weight_sha256": initial_digests[-1], "parameters": model.count_params()}
        model.save_weights(args.output / f"{activation}.initial.weights.h5")
        scored_at = time.perf_counter()
        initial, initial_range = score(model, test, activation)
        initial_cal, _ = score(model, calibration, activation)
        details["initial_scoring_seconds"] = time.perf_counter() - scored_at
        details["initial"] = summarize(labels, initial, initial_cal, original)
        trace = FitTrace()
        fit_started = time.perf_counter()
        try:
            model.fit(fit, fit, validation_data=(calibration, calibration), epochs=epochs,
                      batch_size=BATCH, shuffle=False, callbacks=[trace], verbose=0)
        finally:
            details["fit_seconds"] = time.perf_counter() - fit_started
            details.update(epochs=trace.epochs, batch_seconds=trace.batch_seconds,
                           updates=len(trace.batch_seconds), budget_stopped=trace.budget_stopped)
            model.save_weights(args.output / f"{activation}.last.weights.h5")
        assert trace.best_weights is not None
        model.set_weights(trace.best_weights)
        details.update(selected_epoch=trace.best_epoch, selected_weight_sha256=weight_digest(model))
        assert details["selected_weight_sha256"] != details["initial_weight_sha256"]
        details["completed_requested_updates"] = len(trace.batch_seconds) == epochs * len(fit) // BATCH
        model.save_weights(args.output / f"{activation}.selected.weights.h5")
        fitted, fitted_range = score(model, test, activation)
        fitted_cal, _ = score(model, calibration, activation)
        details["selected"] = summarize(labels, fitted, fitted_cal, original)
        details["output_ranges"] = {"initial": initial_range, "selected": fitted_range}
        np.savez(args.output / f"{activation}.scores.npz", initial=initial, selected=fitted,
                 initial_calibration=initial_cal, selected_calibration=fitted_cal, labels=labels)
        print(activation, "updates", details["updates"], "fit_seconds", details["fit_seconds"],
              "selected", details["selected"]["sampled_prepared"], flush=True)
    record["artifacts"] = {path.name: digest(path) for path in args.output.iterdir() if path.is_file()}
    record["identical_initial_weights"] = True
    record["status"] = "passed"


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "result", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "small"), required=True)
    parser.add_argument("--analysis-commit", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Experimental training/scoring requires a compute node")
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    record = {"status": "started", "classification": "X/A small paired fit, not a universal bound",
              "analysis_commit": args.analysis_commit, "job_id": os.environ["SLURM_JOB_ID"], "stage": args.stage,
              "script_sha256": digest(script), "contract_sha256": digest(script.parent.parent / "SIGMOID_FIT_CHECK.md"),
              "model_source_sha256": MODEL_SHA, "source_result_sha256": RESULT_SHA, "seed": SEED,
              "helpers": {name: digest(script.parent / name) for name in ("sigmoid_sanity.py", "source_assumption_check.py", "post_anchor_diagnostics.py")},
              "versions": {"numpy": np.__version__, "keras": keras.__version__, "torch": torch.__version__}}
    started = time.perf_counter()
    def alarm(signum, frame):
        raise TimeoutError("300-second process analysis cap")
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(300)
    try:
        run(args, record)
    except Exception:
        record["status"], record["error"] = "failed", traceback.format_exc()
        raise
    finally:
        signal.alarm(0)
        record["elapsed_seconds"] = time.perf_counter() - started
        with (args.output / "result.json").open("x") as handle:
            json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        print(json.dumps({key: record[key] for key in ("status", "stage", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
