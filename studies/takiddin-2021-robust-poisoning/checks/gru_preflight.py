#!/usr/bin/env python3
"""Time the frozen full GRU on constructed inputs only; no research data."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
APPROVED_PREFLIGHT_COMMIT = "2d706b103ee03cc705cc3ef3f07718bc3ed7792a"
APPROVED_PREFLIGHT_SHA256 = "238384ed7e52f1438a9bf66202e850b1afa3b18215019dc8e284021633a3a040"
sys.path.insert(0, str(STUDY / "reproduction"))
from models import gru
from run_experiment import configure_tensorflow, require_compute, normalized_softmax, gru_kernel_norms, weight_hash
from analyze_results import digest, verify_source


def gate(update_seconds):
    if len(update_seconds) != 12 or not all(np.isfinite(x) and x > 0 for x in update_seconds):
        raise ValueError("Need two warmup and ten positive finite update times")
    projected = 2250 * max(update_seconds[2:])
    return {"projected_fit_seconds_from_slowest_warm_step": projected,
            "ceiling_seconds": 720, "passes": projected <= 720}


def verify(path, expected_hash, expected_commit, *, authorized_runtime_extension=False):
    if digest(path) != expected_hash:
        raise ValueError("Constructed preflight file changed")
    record = json.loads(path.read_text())
    if record["status"] != "complete" or record["code_commit"] != expected_commit:
        raise ValueError("Need completed preflight from the same frozen code")
    for relative, expected in record["source_sha256"].items():
        verify_source(relative, expected, expected_commit)
    if (record["research_inputs_loaded"] or record["parameters"] != 4058702
            or record["optimizer_updates"] != 12 or record["gate"] != gate(record["update_seconds"])):
        raise ValueError("Preflight scope/schedule/gate differs")
    if authorized_runtime_extension:
        if (expected_commit != APPROVED_PREFLIGHT_COMMIT
                or expected_hash != APPROVED_PREFLIGHT_SHA256):
            raise ValueError("Runtime exception applies only to the approved preflight")
        # Historical source verification above is not enough: scientific files
        # in this launch checkout must also remain byte-identical to the freeze.
        for relative in ("GRU_PILOT.md", "requirements-feed-forward.txt",
                         "reproduction/models.py", "reproduction/run_experiment.py",
                         "reproduction/analyze_results.py"):
            if digest(STUDY / relative) != record["source_sha256"][relative]:
                raise ValueError("Scientific source changed after approved preflight")
        projected = record["gate"]["projected_fit_seconds_from_slowest_warm_step"]
        if projected > 900:
            raise ValueError("Projection exceeds the authorized 900-second ceiling")
        approval = STUDY / "GRU_RUNTIME_EXCEPTION.md"
        return {"status": "verified_with_authorized_exception", "sha256": expected_hash,
                "preflight_commit": expected_commit, "original_gate": record["gate"],
                "approved_ceiling_seconds": 900, "scientific_sources_unchanged": True,
                "approval_document": approval.name, "approval_sha256": digest(approval)}
    if not record["gate"]["passes"]:
        raise ValueError("Constructed runtime gate did not pass; do not load research data")
    return {"status": "verified", "sha256": expected_hash, "gate": record["gate"]}


def run(output):
    require_compute()
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "started", "scope": "constructed inputs only; not paper evidence",
              "research_inputs_loaded": False, "research_fits": 0,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "code_commit": os.environ["EXPECTED_COMMIT"], "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "host": platform.node(), "python": platform.python_version(),
              "source_sha256": {str(p.relative_to(STUDY)): digest(p) for p in (
                  Path(__file__), STUDY / "GRU_PILOT.md", STUDY / "requirements-feed-forward.txt",
                  STUDY / "reproduction/models.py", STUDY / "reproduction/run_experiment.py",
                  STUDY / "reproduction/analyze_results.py")}}
    result_path = output / "preflight.json"
    started = time.perf_counter()
    def save():
        result_path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    save()
    try:
        record["hardware"] = configure_tensorflow(require_gpu=True)
        import tensorflow as tf
        import keras
        generator = np.random.default_rng(20260923)
        x = generator.normal(size=(100, 48, 1)).astype(np.float32)
        y = np.eye(2, dtype=np.float32)[np.arange(100) % 2]
        record["constructed_seed"] = 20260923
        record["constructed_arrays_sha256"] = weight_hash([x, y])
        tick = time.perf_counter()
        with tf.device("/GPU:0"):
            model = gru(20260920)
        record["model_build_seconds"] = time.perf_counter() - tick
        record["parameters"] = model.count_params()
        if record["parameters"] != 4058702:
            raise ValueError("Unexpected GRU parameter inventory")
        record["weight_devices"] = sorted({v.value.device for v in model.trainable_variables})
        if any("GPU:0" not in device for device in record["weight_devices"]):
            raise RuntimeError("GRU weights not on GPU")
        initial = weight_hash(model.get_weights())
        record["update_seconds"], record["losses"] = [], []
        tf.config.experimental.reset_memory_stats("GPU:0")
        save()
        class Recorder(keras.callbacks.Callback):
            def on_train_batch_begin(self, batch, logs=None):
                self.started = time.perf_counter()

            def on_train_batch_end(self, batch, logs=None):
                seconds, loss = time.perf_counter() - self.started, float(logs["loss"])
                if not np.isfinite(loss):
                    raise ValueError("Nonfinite constructed update")
                record["update_seconds"].append(seconds)
                record["losses"].append(loss)
                record["optimizer_updates"] = int(self.model.optimizer.iterations.numpy())
                save()
                print(json.dumps({"constructed_update": batch + 1, "seconds": seconds, "loss": loss}), flush=True)

        dataset = tf.data.Dataset.from_tensors((x, y)).repeat(12)
        options = tf.data.Options()
        options.experimental_deterministic = True
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = 1
        model.fit(dataset.with_options(options).prefetch(1), epochs=1, steps_per_epoch=12,
                  verbose=0, callbacks=[Recorder()])
        tick = time.perf_counter()
        raw_tensor = model(tf.convert_to_tensor(x), training=False)
        raw = raw_tensor.numpy()
        record["inference_seconds"] = time.perf_counter() - tick
        record["inference_device"] = raw_tensor.device
        if "GPU:0" not in raw_tensor.device:
            raise RuntimeError("GRU inference output not on GPU")
        normalized_softmax(raw)
        record["kernel_norms"] = gru_kernel_norms(model)
        record["initial_weights_sha256"] = initial
        record["final_weights_sha256"] = weight_hash(model.get_weights())
        if initial == record["final_weights_sha256"]:
            raise ValueError("Constructed weights did not change")
        record["gpu_allocator_bytes"] = tf.config.experimental.get_memory_info("GPU:0")
        record["gate"] = gate(record["update_seconds"])
        record["versions"] = {"numpy": np.__version__, "tensorflow": tf.__version__, "keras": keras.__version__}
        record["status"] = "complete"
    except Exception as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
        record["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save()
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--commit")
    parser.add_argument("--authorized-runtime-extension", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.verify, args.sha256, args.commit,
                               authorized_runtime_extension=args.authorized_runtime_extension), sort_keys=True))
    elif args.output:
        result = run(args.output)
        print(json.dumps({"status": result["status"], "gate": result["gate"]}), flush=True)
    else:
        parser.error("Need --output or --verify")
