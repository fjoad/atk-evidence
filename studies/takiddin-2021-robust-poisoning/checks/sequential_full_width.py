"""Fixed width-only software learning pair, not a research-data experiment."""

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import sequential_constructed as G

EXPECTED_MODEL = "dcefb40c59c3ffa0082c20a29e9cb413e46ffffa8c7ec3af1df712aef1f82299"


def run_case(output, arrays, reverse, *, bridge_activation="relu", fit_guard_seconds=600):
    import tensorflow as tf
    import keras
    output.mkdir()
    model = G.M.sequential_ensemble(G.MODEL_SEED, timesteps=8, bridge_activation=bridge_activation)
    if model.count_params() != 9240802:
        raise ValueError("Full-width parameter count differs")
    x, test_x = arrays["train_x"], arrays["test_x"]
    y = 1-arrays["train_y"] if reverse else arrays["train_y"]
    test_y = 1-arrays["test_y"] if reverse else arrays["test_y"]
    initial = model.get_weights()
    initial_trainable = {v.path: v.numpy().copy() for v in model.trainable_weights}
    np.savez_compressed(output / "initial_weights.npz", **{f"w{i}": w for i,w in enumerate(initial)})
    record = {"label_orientation": "reversed" if reverse else "normal", "status": "started",
              "model_seed": G.MODEL_SEED, "parameter_count": model.count_params(),
              "timesteps": 8, "fit_guard_seconds": fit_guard_seconds,
              "bridge_activation": bridge_activation,
              "initial_weights_sha256": G.R.weight_hash(initial),
              "initial_test": G.measurements(test_y, model(test_x).numpy()),
              "initial_gradient": G.gradient_report(model, x, y)}
    (output / "result.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    history, stopped = [], []
    started = time.monotonic()

    class Recorder(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            loss = float(logs["loss"])
            history.append({"epoch": epoch+1, "loss": loss if np.isfinite(loss) else None,
                            "elapsed_seconds": time.monotonic()-started,
                            "updates": int(self.model.optimizer.iterations.numpy())})
            (output / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False)+"\n")
            if epoch == 0 or (epoch+1) % 25 == 0:
                print(record["label_orientation"], json.dumps(history[-1]), flush=True)
            if not np.isfinite(loss):
                stopped.append("nonfinite_loss")
                self.model.stop_training = True
            elif time.monotonic()-started >= fit_guard_seconds and epoch+1 < 300:
                stopped.append("time_guard")
                self.model.stop_training = True

    options = tf.data.Options()
    options.deterministic = True
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(32).with_options(options)
    try:
        model.fit(dataset, epochs=300, shuffle=False, verbose=0, callbacks=[Recorder()])
        record["status"] = stopped[0] if stopped else "complete"
    except Exception as exc:
        record.update(status="error", error=f"{type(exc).__name__}: {exc}")
    record.update(updates=int(model.optimizer.iterations.numpy()), elapsed_seconds=time.monotonic()-started)
    model.save(output / "final.keras")
    # Reuse only the existing fixed metric/persistence operations.
    record = G.finish_record(output, model, record, initial_trainable, x, y, test_x, test_y)
    restored = keras.models.load_model(output / "final.keras")
    np.testing.assert_array_equal(restored(test_x).numpy(), model(test_x).numpy())
    if G.R.weight_hash(restored.get_weights()) != record["final_weights_sha256"]:
        raise ValueError("Reloaded weights differ")
    if int(restored.optimizer.iterations.numpy()) != record["updates"]:
        raise ValueError("Reloaded optimizer count differs")
    record["reload_exact"] = True
    (output / "result.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"case": record["label_orientation"], "status": record["status"],
                      "test": record["test"], "learning_gate_passed": record["learning_gate_passed"]}), flush=True)
    return record


def run(output):
    if G.digest(G.STUDY / "reproduction/models.py") != EXPECTED_MODEL:
        raise ValueError("Declared scientific model changed")
    output.mkdir(parents=True, exist_ok=False)
    runtime = G.R.configure_tensorflow(False)
    arrays = G.fixture_data()
    previous = G.ROOT / "data/derived/takiddin-2021-robust-poisoning/sequential-constructed-20260927-attempt1-recovery"
    with np.load(previous / "constructed_data.npz") as old:
        for k,v in arrays.items():
            np.testing.assert_array_equal(v, old[k])
    sources = [Path(__file__).resolve(), Path(G.__file__).resolve(), G.STUDY / "reproduction/models.py",
               G.ROOT / "docs/plans/2026-09-28-sequential-execution.md"]
    result = {"scope": "full-width eight-step constructed software learning pair only",
              "status": "started", "runtime": runtime,
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=G.ROOT, text=True).strip(),
              "source_sha256": {str(p.relative_to(G.ROOT)): G.digest(p) for p in sources},
              "same_constructed_arrays_as_previous": True,
              "previous_data_sha256": G.digest(previous / "constructed_data.npz"),
              "model_seed": G.MODEL_SEED, "data_seed": G.DATA_SEED,
              "constant_prior": G.measurements(arrays["test_y"], np.full((32,1), .5)),
              "daily_mean_accuracy": float(np.mean((arrays["test_x"].mean(axis=(1,2))>0)==arrays["test_y"].ravel())),
              "cases": {}}
    save = lambda: (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    save()
    import tensorflow as tf
    with tf.device("/CPU:0"):
        for reverse in (False, True):
            name = "reversed" if reverse else "normal"
            result["cases"][name] = run_case(output / name, arrays, reverse)
            save()
    cases = list(result["cases"].values())
    result["paired_initial_weights"] = len({c["initial_weights_sha256"] for c in cases}) == 1
    result["gate_passed"] = result["paired_initial_weights"] and all(
        c["learning_gate_passed"] and c["gradient_group_gate_passed"]
        and c["weights_finite"] and c["reload_exact"]
        and c["maximum_constrained_norm"] <= 1.00001 for c in cases)
    result["status"] = "passed" if result["gate_passed"] else "failed_gate"
    save()
    print(json.dumps({"status": result["status"], "paired_initial_weights": result["paired_initial_weights"]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r = run(args.output)
    raise SystemExit(0 if r["gate_passed"] else 2)
