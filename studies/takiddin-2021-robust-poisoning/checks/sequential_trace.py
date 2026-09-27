"""One fixed 50-update prefix replay of the failed synthetic learning case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np

import sequential_constructed as G

# Exact frozen file hashes, rather than accepting another dataset with the same shape.
EXPECTED_DATA = "9906021e2cc5a5f8fd3d7b96f134732d98a2229b1392222aa2c5d5608fe58e9f"
EXPECTED_HISTORY = "1b10197abb43ad7235ccbbf3bb959d1ba9d611eea47acc8e56fa479a20fedd26"
EXPECTED_MODEL = "dcefb40c59c3ffa0082c20a29e9cb413e46ffffa8c7ec3af1df712aef1f82299"
STAGES = ["intermediate", *[f"gru_{i}" for i in range(1, 9)], "classifier_hidden", "probability"]


def stats(values):
    """Compact columns: minimum, maximum, standard deviation, positive fraction."""
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite observation")
    return [float(values.min()), float(values.max()), float(values.std()),
            float(np.mean(values > 0))]


def zero_events(rows):
    result = {}
    for name in STAGES[:-1]:
        zero = [row["train"]["layers"][name][3] == 0 for row in rows]
        transitions = [rows[i]["step"] for i in range(1, len(rows)) if zero[i] and not zero[i-1]]
        result[name] = {"initially_zero": bool(zero[0]), "new_zero_steps": transitions,
                        "zero_through_observed_end_from_first_transition":
                            bool(all(zero[next(i for i,r in enumerate(rows) if r["step"] == transitions[0]):]))
                            if transitions else None}
    return result


def affine_comparison(x0, w0, b0, x1, w1, b1):
    """Saved-state forward counterfactuals, with an exact additive identity."""
    x0, w0, b0, x1, w1, b1 = [np.asarray(x, dtype=np.float64) for x in (x0,w0,b0,x1,w1,b1)]
    cases = {
        "before_inputs_before_parameters": x0 @ w0 + b0,
        "before_inputs_after_parameters": x0 @ w1 + b1,
        "after_inputs_before_parameters": x1 @ w0 + b0,
        "after_inputs_after_parameters": x1 @ w1 + b1,
        "after_inputs_after_weights_before_bias": x1 @ w1 + b0,
        "after_inputs_before_weights_after_bias": x1 @ w0 + b1,
    }
    changes = {"input": (x1-x0) @ w0, "weights": x1 @ (w1-w0),
               "bias": np.broadcast_to(b1-b0, cases["after_inputs_after_parameters"].shape)}
    actual = cases["after_inputs_after_parameters"]-cases["before_inputs_before_parameters"]
    return {"cases": {k: {"preactivation": stats(v), "relu_output": stats(np.maximum(v, 0))}
                       for k,v in cases.items()},
            "decomposition": {k: stats(v) for k,v in changes.items()},
            "additive_identity_max_error": float(np.max(np.abs(actual-sum(changes.values()))))}


def run(previous, output):
    import tensorflow as tf
    import keras
    if G.digest(previous / "constructed_data.npz") != EXPECTED_DATA:
        raise ValueError("Unexpected prior constructed input")
    if G.digest(previous / "reversed/history.json") != EXPECTED_HISTORY:
        raise ValueError("Unexpected prior reversed history")
    if G.digest(G.STUDY / "reproduction/models.py") != EXPECTED_MODEL:
        raise ValueError("Frozen model changed")
    original = json.loads((previous / "result.json").read_text())
    history = json.loads((previous / "reversed/history.json").read_text())
    before_files = {str(p.relative_to(previous)): G.digest(p) for p in previous.rglob("*") if p.is_file()}
    with np.load(previous / "constructed_data.npz") as f:
        arrays = {k: f[k] for k in f.files}
    for key, expected in G.fixture_data().items():
        np.testing.assert_array_equal(arrays[key], expected)
    output.mkdir(parents=True, exist_ok=False)
    runtime = G.R.configure_tensorflow(False)
    sources = [Path(__file__).resolve(), G.STUDY / "checks/sequential_constructed.py",
               G.STUDY / "reproduction/models.py",
               G.ROOT / "docs/plans/2026-09-27-sequential-collapse-trace.md"]
    result = {"scope": "constructed prefix trace, not research evidence", "status": "started",
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=G.ROOT, text=True).strip(),
              "source_sha256": {str(p.relative_to(G.ROOT)): G.digest(p) for p in sources},
              "original_file_sha256": before_files, "runtime": runtime,
              "statistics_columns": ["min", "max", "std", "positive_fraction"], "history_prefix_exact": True,
              "observer_state_unchanged": True, "updates_requested": 50}
    save = lambda: (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    save()
    started = time.monotonic()
    rows = []
    model = G.M.sequential_ensemble(G.MODEL_SEED, **G.SMALL)
    expected_initial = original["cases"]["reversed"]["initial_weights_sha256"]
    result["initial_weights_sha256"] = G.R.weight_hash(model.get_weights())
    if result["initial_weights_sha256"] != expected_initial:
        raise ValueError("Different initialization")
    tensors = [model.get_layer("attention_decoder").output[0]]
    tensors += [model.get_layer(f"sequence_gru_{i}").output for i in range(1, 9)]
    tensors += [model.get_layer("classifier_hidden").output, model.outputs[0]]
    probe = keras.Model(model.input, tensors)
    paths = [v.path for v in model.trainable_weights]
    head = model.get_layer("classifier_hidden")
    result["weight_paths"] = [v.path for v in model.weights]
    result["head_weight_indices"] = [result["weight_paths"].index(v.path) for v in (head.kernel, head.bias)]

    @tf.function(jit_compile=False)
    def observed(x, y):
        with tf.GradientTape() as tape:
            values = probe(x, training=False)
            loss = model.loss(y, values[-1])
        return values, tape.gradient(loss, model.trainable_weights)

    def state_hash():
        return G.R.weight_hash(model.get_weights() + [v.numpy() for v in model.optimizer.variables])

    def observe(step, fit_loss=None):
        state_before = state_hash()
        record = {"step": int(step), "fit_loss": fit_loss}
        archive = {f"w{i}": w for i,w in enumerate(model.get_weights())}
        archive.update({f"opt{i}": v.numpy() for i,v in enumerate(model.optimizer.variables)})
        for split in ("train", "test"):
            x, y = arrays[f"{split}_x"], 1-arrays[f"{split}_y"]
            values, gradients = observed(x, y)
            values = [v.numpy() for v in values]
            record[split] = {"layers": {n: stats(v) for n,v in zip(STAGES,values)},
                             "metrics": G.measurements(y, values[-1])}
            archive.update({f"{split}_{n}": v for n,v in zip(STAGES,values)})
            if split == "train":
                norm2 = {g: 0. for g in G.GROUPS}
                per_layer = {}
                for path, gradient in zip(paths, gradients):
                    if gradient is None or not np.isfinite(gradient.numpy()).all():
                        raise ValueError("Missing or nonfinite observed gradient")
                    value = float(np.linalg.norm(gradient.numpy().astype(np.float64)))
                    norm2[G.group(path)] += value**2
                    per_layer[path] = value
                record["gradient_group_l2"] = {g: float(np.sqrt(v)) for g,v in norm2.items()}
                record["gradient_variable_l2"] = per_layer
        raw = archive["train_gru_8"] @ head.kernel.numpy() + head.bias.numpy()
        np.testing.assert_allclose(np.maximum(raw, 0), archive["train_classifier_hidden"], rtol=1e-5, atol=1e-8)
        record["head_preactivation"] = stats(raw)
        np.savez_compressed(output / f"step-{step:03d}.npz", **archive)
        if state_hash() != state_before:
            result["observer_state_unchanged"] = False
            raise ValueError("Observation changed model/optimizer state")
        rows.append(record)
        with (output / "trace.jsonl").open("a") as f:
            f.write(json.dumps(record, allow_nan=False)+"\n")

    class Recorder(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            loss = float(logs["loss"])
            if loss != history[epoch]["loss"]:
                result["history_prefix_exact"] = False
                self.model.stop_training = True
            observe(epoch+1, loss)
            result["updates_completed"] = int(self.model.optimizer.iterations.numpy())
            result["elapsed_seconds"] = time.monotonic()-started
            if result["elapsed_seconds"] >= 300 and epoch+1 < 50:
                result["time_guard"] = True
                self.model.stop_training = True
            if epoch == 0 or (epoch+1) % 10 == 0:
                print(json.dumps({"step": epoch+1, "loss": loss,
                                  "exact_old_loss": result["history_prefix_exact"],
                                  "zero_stages": [n for n in STAGES[:-1] if rows[-1]["train"]["layers"][n][3] == 0]}), flush=True)
            save()

    try:
        with tf.device("/CPU:0"):
            observe(0)
            options = tf.data.Options()
            options.deterministic = True
            options.threading.private_threadpool_size = 1
            options.threading.max_intra_op_parallelism = 1
            dataset = tf.data.Dataset.from_tensor_slices((arrays["train_x"], 1-arrays["train_y"]))
            dataset = dataset.batch(32).with_options(options)
            model.fit(dataset, epochs=50, shuffle=False, verbose=0, callbacks=[Recorder()])
        result["events"] = zero_events(rows)
        head_events = result["events"]["classifier_hidden"]["new_zero_steps"]
        if head_events and result["history_prefix_exact"]:
            step = head_events[0]
            iw, ib = result["head_weight_indices"]
            with np.load(output / f"step-{step-1:03d}.npz") as a, np.load(output / f"step-{step:03d}.npz") as b:
                result["head_transition"] = {"step": step, "comparison": affine_comparison(
                    a["train_gru_8"], a[f"w{iw}"], a[f"w{ib}"],
                    b["train_gru_8"], b[f"w{iw}"], b[f"w{ib}"])}
        result["status"] = ("complete" if result.get("updates_completed") == 50
                            and result["history_prefix_exact"] and result["observer_state_unchanged"]
                            else "incomplete_or_mismatched")
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic()-started
        result["originals_unchanged"] = all(G.digest(previous / k) == v for k,v in before_files.items())
        result["artifact_sha256"] = {p.name: G.digest(p) for p in output.iterdir() if p.name != "result.json"}
        save()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r = run(args.previous, args.output)
    print(json.dumps({"status": r["status"], "events": r.get("events")}, indent=2), flush=True)
    raise SystemExit(0 if r["status"] == "complete" else 2)
