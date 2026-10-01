"""Full48-step Sigmoid-bridge GPU timing, gated by preserved learning checks."""

import argparse
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import sequential_constructed as G
from sequential_interface import branch_passes


def timing_gate(epochs):
    if len(epochs) != 3 or any(e["updates"] != 45 or not np.isfinite(e["seconds"])
                                or e["seconds"] <= 0 for e in epochs):
        raise ValueError("Need three complete45-update epochs with finite times")
    projected = 50 * max(e["seconds"] for e in epochs[1:])
    return {"projected_fit_seconds": projected, "ceiling_seconds": 3600,
            "passed": bool(projected <= 3600)}


def verify_learning(path, expected_hash):
    if G.digest(path) != expected_hash:
        raise ValueError("Learning result hash differs")
    r = json.loads(path.read_text())
    selected = r["branches"]["sigmoid"]
    if (r["status"] != "complete" or selected["interpretation"] != "I-SEQ-scalar-sigmoid"
            or not branch_passes(selected["cases"])
            or not all(G.passes_learning(c) for c in selected["cases"].values())):
        raise ValueError("The predeclared Sigmoid learning gate did not pass")
    for name in ("reproduction/models.py", "SEQUENTIAL_INTERFACE.md"):
        relative = str((G.STUDY / name).relative_to(G.ROOT))
        if G.digest(G.STUDY / name) != r["source_sha256"][relative]:
            raise ValueError("Scientific model/interface differs from the passing check")
    for c in selected["cases"].values():
        if c["bridge_activation"] != "sigmoid" or c["parameter_count"] != 9240802:
            raise ValueError("Learning result describes another model")
    return {"learning_sha256": expected_hash, "interpretation": selected["interpretation"]}


def run(output, learning, learning_hash):
    G.R.require_compute()
    approval = verify_learning(learning, learning_hash)
    output.mkdir(parents=True, exist_ok=False)
    r = {"status": "started", "scope": "constructed GPU timing; no CER data",
         "research_inputs_loaded": False, "research_fits": 0, "approval": approval,
         "code_commit": os.environ["EXPECTED_COMMIT"], "slurm_job_id": os.environ["SLURM_JOB_ID"],
         "host": platform.node(), "batches": [], "epochs": [],
         "source_sha256": {str(p.relative_to(G.ROOT)): G.digest(p) for p in (
             Path(__file__).resolve(), G.STUDY / "reproduction/models.py",
             G.STUDY / "reproduction/run_experiment.py", G.STUDY / "SEQUENTIAL_INTERFACE.md",
             G.STUDY / "SEQUENTIAL_ENSEMBLE_PILOT.md")}}
    save = lambda: (output / "preflight.json").write_text(json.dumps(r, indent=2, allow_nan=False)+"\n")
    save()
    started = time.perf_counter()
    try:
        r["hardware"] = G.R.configure_tensorflow(True)
        import tensorflow as tf
        import keras
        generator = np.random.default_rng(20260926)
        x = generator.normal(size=(4464, 48, 1)).astype(np.float32)
        y = (np.arange(4464) % 2).astype(np.float32)[:, None]
        r["constructed_array_sha256"] = G.R.weight_hash([x, y])
        tick = time.perf_counter()
        with tf.device("/GPU:0"):
            model = G.M.sequential_ensemble(G.MODEL_SEED, bridge_activation="sigmoid")
        r["build_seconds"] = time.perf_counter()-tick
        r["parameters"] = model.count_params()
        if r["parameters"] != 9240802:
            raise ValueError("Unexpected model inventory")
        r["weight_devices"] = sorted({v.value.device for v in model.trainable_weights})
        if any("GPU:0" not in d for d in r["weight_devices"]):
            raise ValueError("Weights are not on the allocated GPU")
        r["initial_weights_sha256"] = G.R.weight_hash(model.get_weights())
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(4464, seed=G.MODEL_SEED,
            reshuffle_each_iteration=True).batch(100)
        options = tf.data.Options()
        options.deterministic = True
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = 1
        dataset = dataset.with_options(options).prefetch(1)
        tf.config.experimental.reset_memory_stats("GPU:0")

        class Recorder(keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch, self.epoch_started = epoch+1, time.perf_counter()
                self.first_update = int(model.optimizer.iterations.numpy())

            def on_train_batch_begin(self, batch, logs=None):
                self.batch_started = time.perf_counter()

            def on_train_batch_end(self, batch, logs=None):
                loss = float(logs["loss"])
                if not np.isfinite(loss):
                    raise ValueError("Nonfinite preflight loss")
                r["batches"].append({"epoch": self.epoch, "batch": batch+1,
                    "size": 64 if batch == 44 else 100, "loss": loss,
                    "seconds": time.perf_counter()-self.batch_started})
                r["optimizer_updates"] = int(model.optimizer.iterations.numpy())
                save()

            def on_epoch_end(self, epoch, logs=None):
                row = {"epoch": epoch+1, "seconds": time.perf_counter()-self.epoch_started,
                       "updates": int(model.optimizer.iterations.numpy())-self.first_update,
                       "loss": float(logs["loss"])}
                r["epochs"].append(row)
                save()
                print(json.dumps(row), flush=True)

        model.fit(dataset, epochs=3, verbose=0, callbacks=[Recorder()])
        r["gate"] = timing_gate(r["epochs"])
        r["gpu_allocator_bytes"] = tf.config.experimental.get_memory_info("GPU:0")
        r["final_weights_sha256"] = G.R.weight_hash(model.get_weights())
        if not all(np.isfinite(v.numpy()).all() for v in model.trainable_weights):
            raise ValueError("Nonfinite preflight weights")
        r["maximum_constrained_norm"] = max(float(np.linalg.norm(v.numpy().astype(np.float64), axis=0).max())
            for v in model.trainable_weights if v.constraint is not None)
        if r["maximum_constrained_norm"] > 1.00001:
            raise ValueError("Weight constraint violated")
        if r["final_weights_sha256"] == r["initial_weights_sha256"]:
            raise ValueError("No update changed model weights")
        model.save(output / "constructed.keras")
        with tf.device("/GPU:0"):
            before = model(x[:64]).numpy()
            if not np.isfinite(before).all() or np.any((before < 0) | (before > 1)):
                raise ValueError("Invalid preflight probabilities")
            restored = keras.models.load_model(output / "constructed.keras")
            np.testing.assert_array_equal(before, restored(x[:64]).numpy())
        if int(restored.optimizer.iterations.numpy()) != 135:
            raise ValueError("Reloaded optimizer count differs")
        r["reload_exact"] = True
        r["model_sha256"] = G.digest(output / "constructed.keras")
        r["status"] = "complete"
    except Exception as exc:
        r.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        r["elapsed_seconds"] = time.perf_counter()-started
        save()
    return r


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learning", type=Path, required=True)
    parser.add_argument("--learning-sha", required=True)
    args = parser.parse_args()
    result = run(args.output, args.learning, args.learning_sha)
    print(json.dumps({"status": result["status"], "gate": result["gate"]}), flush=True)
    raise SystemExit(0 if result["gate"]["passed"] else 2)
