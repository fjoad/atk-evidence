"""Fixed synthetic learning gate; no research-data loader or branch search.

SEQUENTIAL_ENSEMBLE_PILOT.md freezes two label orientations, one model seed,
300 updates each and 300-second guards. Exit 2 means a failed learning gate,
not a reproduction result. All outcomes and weights are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
sys.path.insert(0, str(STUDY / "reproduction"))
import models as M
import run_experiment as R

MODEL_SEED, DATA_SEED = 20260920, 20260926
SMALL = dict(encoder_units=(8, 6, 4), gru_units=8, dense_units=16, timesteps=8)
GROUPS = ("encoder", "attention", "decoder", "intermediate", "gru", "classifier")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture_data():
    generator = np.random.default_rng(DATA_SEED)
    labels = np.repeat(np.array([0., 1.], dtype=np.float32), 16)[:, None]
    arrays = {}
    for split in ("train", "test"):
        arrays[f"{split}_x"] = (.75 * (2*labels[:, :, None]-1)
            + .05 * generator.normal(size=(32, 8, 1))).astype(np.float32)
        arrays[f"{split}_y"] = labels.copy()
    return arrays


def measurements(labels, probability):
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        return {"finite": False, "accuracy": None, "bce": None}
    p = np.clip(probability, 1e-7, 1-1e-7)
    return {"finite": True, "accuracy": float(np.mean((probability > .5) == labels)),
            "bce": float(-np.mean(labels*np.log(p)+(1-labels)*np.log1p(-p))),
            "probability_min": float(probability.min()), "probability_max": float(probability.max())}


def passes_learning(record):
    return bool(record["status"] == "complete" and record["updates"] == 300
            and record["test"]["finite"] and record["test"]["accuracy"] >= .9
            and record["test"]["bce"] < np.log(2)/2)


def group(path):
    if "encoder_lstm" in path:
        return "encoder"
    if "decoder_lstm" in path:
        return "decoder"
    if "intermediate_projection" in path:
        return "intermediate"
    if "sequence_gru" in path:
        return "gru"
    if "attention_decoder/attention_" in path:
        return "attention"
    if "classifier_hidden" in path or "attack_probability" in path:
        return "classifier"
    raise ValueError(f"Unclassified parameter {path}")


def gradient_report(model, x, y):
    import tensorflow as tf
    with tf.GradientTape() as tape:
        loss = model.loss(y, model(x, training=True))
    gradients = tape.gradient(loss, model.trainable_weights)
    squared = {name: 0. for name in GROUPS}
    detail = []
    all_finite = True
    for variable, gradient in zip(model.trainable_weights, gradients):
        name = group(variable.path)
        finite = gradient is not None and bool(np.isfinite(gradient.numpy()).all())
        all_finite &= finite
        norm = float(np.linalg.norm(gradient.numpy().astype(np.float64))) if finite else None
        if norm is not None:
            squared[name] += norm**2
        detail.append({"name": variable.path, "group": name, "finite": finite, "l2": norm})
    return {"finite": all_finite, "group_l2": {k: float(np.sqrt(v)) for k,v in squared.items()},
            "variables": detail}


def run_one(output, arrays, reverse):
    import tensorflow as tf
    import keras
    output.mkdir()
    model = M.sequential_ensemble(MODEL_SEED, **SMALL)
    x, test_x = arrays["train_x"], arrays["test_x"]
    y = 1-arrays["train_y"] if reverse else arrays["train_y"]
    test_y = 1-arrays["test_y"] if reverse else arrays["test_y"]
    initial = model.get_weights()
    np.savez(output / "initial_weights.npz", **{f"w{i}": w for i,w in enumerate(initial)})
    initial_trainable = {v.path: v.numpy().copy() for v in model.trainable_weights}
    record = {"label_orientation": "reversed" if reverse else "normal", "status": "started",
              "model_seed": MODEL_SEED, "parameter_count": model.count_params(),
              "initial_weights_sha256": R.weight_hash(initial),
              "initial_test": measurements(test_y, model(test_x).numpy()),
              "initial_gradient": gradient_report(model, x, y)}
    (output / "result.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    history, stop_reason = [], []
    started = time.monotonic()

    class Recorder(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            finite = bool(np.isfinite(logs.get("loss", np.nan)))
            history.append({"epoch": int(epoch+1), "loss": float(logs["loss"]) if finite else None,
                            "elapsed_seconds": time.monotonic()-started,
                            "updates": int(self.model.optimizer.iterations.numpy())})
            (output / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False)+"\n")
            if epoch == 0 or (epoch+1) % 50 == 0:
                print(record["label_orientation"], history[-1], flush=True)
            if not finite:
                stop_reason.append("nonfinite_loss")
                self.model.stop_training = True
            elif time.monotonic()-started >= 300 and epoch+1 < 300:
                stop_reason.append("time_guard")
                self.model.stop_training = True

    options = tf.data.Options()
    options.deterministic = True
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(32).with_options(options)
    try:
        model.fit(dataset, epochs=300, shuffle=False, verbose=0, callbacks=[Recorder()])
        record["status"] = stop_reason[0] if stop_reason else "complete"
    except Exception as exc:
        record.update(status="error", error=f"{type(exc).__name__}: {exc}")
    record.update(elapsed_seconds=time.monotonic()-started,
                  updates=int(model.optimizer.iterations.numpy()))
    # Save before any final diagnostic that could fail, including partial runs.
    model.save(output / "final.keras")
    return finish_record(output, model, record, initial_trainable, x, y, test_x, test_y)


def finish_record(output, model, record, initial_trainable, x, y, test_x, test_y):
    """Evaluate/persist an already fitted constructed model, without updates."""
    import keras
    probability = model(test_x).numpy()
    front = keras.Model(model.input, model.get_layer("attention_decoder").output)
    sequence, attention = front(test_x)
    np.savez(output / "outputs.npz", probability=probability, sequence=sequence.numpy(),
             attention=attention.numpy(), observed_test_y=test_y)
    record.update(test=measurements(test_y, probability),
                  train=measurements(y, model(x).numpy()),
                  final_gradient=gradient_report(model, x, y))
    changed = {name: [] for name in GROUPS}
    finite_weights, norms = True, []
    for variable in model.trainable_weights:
        finite_weights &= bool(np.isfinite(variable.numpy()).all())
        changed[group(variable.path)].append(not np.array_equal(variable.numpy(), initial_trainable[variable.path]))
        if variable.constraint is not None:
            value = float(np.linalg.norm(variable.numpy().astype(np.float64), axis=0).max())
            norms.append(value if np.isfinite(value) else None)
    record.update(weights_finite=finite_weights,
                  changed_groups={k: any(v) for k,v in changed.items()},
                  maximum_constrained_norm=max(norms) if all(v is not None for v in norms) else None,
                  final_weights_sha256=R.weight_hash(model.get_weights()))
    record["learning_gate_passed"] = passes_learning(record)
    record["gradient_group_gate_passed"] = (
        record["initial_gradient"]["finite"] and record["final_gradient"]["finite"]
        and all(max(record["initial_gradient"]["group_l2"][g],
                    record["final_gradient"]["group_l2"][g]) > 0
                and record["changed_groups"][g] for g in GROUPS))
    record["artifact_sha256"] = {p.name: digest(p) for p in output.iterdir() if p.name != "result.json"}
    (output / "result.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    info = R.configure_tensorflow(False)
    sources = [Path(__file__).resolve(), STUDY / "reproduction/models.py",
               STUDY / "SEQUENTIAL_ENSEMBLE_PILOT.md",
               ROOT / "docs/plans/2026-09-27-sequential-implementation.md"]
    record = {"scope": "constructed software learning gate only; no CER observations",
              "status": "started", "data_seed": DATA_SEED, "model_seed": MODEL_SEED,
              "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
              "runtime": info, "cases": {}}
    arrays = fixture_data()
    np.savez(args.output / "constructed_data.npz", **arrays)
    record["data_sha256"] = digest(args.output / "constructed_data.npz")
    record["constant_prior"] = measurements(arrays["test_y"], np.full((32, 1), .5))
    record["daily_mean_accuracy"] = float(np.mean(
        (arrays["test_x"].mean(axis=(1,2)) > 0) == arrays["test_y"].ravel()))
    output = args.output / "result.json"
    output.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    import tensorflow as tf
    with tf.device("/CPU:0"):
        for reverse in (False, True):
            name = "reversed" if reverse else "normal"
            record["cases"][name] = run_one(args.output / name, arrays, reverse)
            output.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    cases = list(record["cases"].values())
    record["paired_initial_weights"] = len({c["initial_weights_sha256"] for c in cases}) == 1
    record["gate_passed"] = record["paired_initial_weights"] and all(
        c["learning_gate_passed"] and c["gradient_group_gate_passed"] and c["weights_finite"] for c in cases)
    record["status"] = "passed" if record["gate_passed"] else "failed_gate"
    output.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": record["status"], "cases": {
        k: {q: v[q] for q in ("status", "updates", "test", "learning_gate_passed", "gradient_group_gate_passed")}
        for k,v in record["cases"].items()}}, indent=2), flush=True)
    return 0 if record["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
