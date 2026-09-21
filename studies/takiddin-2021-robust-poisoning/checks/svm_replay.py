#!/usr/bin/env python3
"""Read-only reconstruction of saved SVM margins and a fixed kernel spectrum."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
from unittest.mock import patch

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.svm import SVC
import threadpoolctl

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "reproduction"))
from run_experiment import load_preparation, require_compute
from analyze_results import audit_result, digest, verify_source

RESULT_HASHES = {
    "p00": "920beced622b9ee93a51cef34857dbd2f266df5e3b41b18c5abe2c84988d4746",
    "p30": "3ea84776a1c63f31b3ac95d075cb3b4af220d501bb0218fdc5db5e65bd1b8ac2",
}
ATOL, RTOL, SPECTRAL_REL = 1e-8, 1e-10, 1e-10


def subset_indices(uid, count=512):
    uid = np.asarray(uid)
    if uid.ndim != 1 or len(np.unique(uid)) != len(uid) or len(uid) < count:
        raise ValueError("Subset needs enough unique training identities")
    generator = np.random.default_rng(np.random.SeedSequence([20260921, 601]))
    selected = generator.permutation(np.argsort(uid, kind="stable"))[:count]
    return selected[np.argsort(uid[selected], kind="stable")]


def kernel(x, z, gamma, coef0):
    return np.tanh(float(gamma) * (np.asarray(x, dtype=np.float64)
                                  @ np.asarray(z, dtype=np.float64).T) + float(coef0))


def manual_scores(x, support, dual, intercept, gamma, coef0, chunk=128):
    if chunk < 1:
        raise ValueError("Chunk size must be positive")
    return np.concatenate([kernel(x[start:start + chunk], support, gamma, coef0)
                           @ np.asarray(dual).ravel() + float(intercept)
                           for start in range(0, len(x), chunk)])


def replay_summary(manual, native, predictions):
    if manual.shape != native.shape or not np.isfinite(manual).all() or not np.isfinite(native).all():
        raise ValueError("Invalid replay score vectors")
    np.testing.assert_array_equal(predictions, native >= 0)
    tolerance = ATOL + RTOL * np.abs(native)
    difference = np.abs(manual - native)
    near_zero = np.abs(native) <= tolerance
    disagreement = (manual >= 0) != predictions
    if np.any(difference > tolerance) or np.any(disagreement & ~near_zero):
        raise ValueError("Independent score replay fails frozen tolerance")
    return {"rows": len(native), "max_absolute_error": float(difference.max()),
            "max_error_tolerance_ratio": float(np.max(difference / tolerance)),
            "near_zero_native_margins": int(near_zero.sum()),
            "prediction_disagreements": int(disagreement.sum()),
            "disagreements_outside_near_zero": int((disagreement & ~near_zero).sum())}


def centered(matrix):
    mean = matrix.mean(axis=1)
    result = matrix - mean[:, None] - mean[None, :] + matrix.mean()
    return (result + result.T) / 2


def kernel_summary(matrix):
    off_diagonal = matrix[~np.eye(len(matrix), dtype=bool)]
    return {"rows": len(matrix), "asymmetry_max": float(np.max(np.abs(matrix - matrix.T))),
        "diagonal_min": float(np.diag(matrix).min()), "diagonal_max": float(np.diag(matrix).max()),
        "off_diagonal_abs_ge_0_99_fraction": float(np.mean(np.abs(off_diagonal) >= .99))}


def selection_summary(uid, synthetic, labels):
    return {"rows": len(uid), "seed": 20260921, "rng_role": 601,
        "uid_sha256": hashlib.sha256(np.asarray(uid, dtype="<i8").tobytes()).hexdigest(),
        "original_rows": int((~synthetic).sum()), "synthetic_rows": int(synthetic.sum()),
        "observed_class_counts": {level: np.bincount(y, minlength=2).tolist() for level, y in labels.items()}}


def spectrum_summary(matrix, eigenvalues, witness):
    scale = max(1., float(np.max(np.abs(eigenvalues))))
    tolerance = SPECTRAL_REL * scale
    np.testing.assert_allclose(np.linalg.norm(witness), 1., atol=1e-12, rtol=0)
    residual = float(np.linalg.norm(matrix @ witness - eigenvalues[0] * witness) / scale)
    rayleigh = float(witness @ matrix @ witness)
    if residual > 1e-10 or abs(rayleigh - eigenvalues[0]) > tolerance:
        raise ValueError("Invalid minimum-eigenvalue witness")
    np.testing.assert_allclose(eigenvalues.sum(), np.trace(matrix), atol=tolerance * len(matrix), rtol=0)
    resolved = np.abs(eigenvalues)[np.abs(eigenvalues) > tolerance]
    negative = eigenvalues < -tolerance
    return {"minimum": float(eigenvalues[0]), "maximum": float(eigenvalues[-1]),
            "tolerance": tolerance, "negative_count": int(negative.sum()),
            "near_zero_count": int(np.count_nonzero(np.abs(eigenvalues) <= tolerance)),
            "positive_count": int(np.count_nonzero(eigenvalues > tolerance)),
            "negative_spectral_mass": float(-eigenvalues[negative].sum()),
            "thresholded_abs_spectrum_ratio": float(resolved.max() / resolved.min()) if len(resolved) else None,
            "minimum_witness_rayleigh": rayleigh, "normalized_witness_residual": residual}


def spectral_checks(matrix):
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or not np.isfinite(matrix).all() or np.max(np.abs(matrix - matrix.T)) > 1e-12):
        raise ValueError("Kernel must be finite and symmetric")
    symmetric = (matrix + matrix.T) / 2
    arrays, report = {}, {}
    for name, value in (("raw", symmetric), ("centered", centered(symmetric))):
        eigenvalues, vectors = np.linalg.eigh(value)
        witness = vectors[:, 0]
        arrays[name + "_eigenvalues"] = eigenvalues
        arrays[name + "_witness"] = witness
        report[name] = spectrum_summary(value, eigenvalues, witness)
    report["kernel"] = kernel_summary(matrix)
    return report, arrays


def equality_witness(matrix, eigenvalues, witness, labels):
    tolerance = SPECTRAL_REL * max(1., float(np.max(np.abs(eigenvalues))))
    if eigenvalues[0] >= -tolerance:
        return {"negative_centered_witness": False}
    if set(np.unique(labels)) != {0, 1} or abs(witness.sum()) > 1e-8:
        raise ValueError("Negative centered witness must have zero sum and both label classes")
    signs = 2 * labels.astype(float) - 1
    direction = signs * witness
    q = signs[:, None] * matrix * signs[None, :]
    value = float(direction @ q @ direction)
    np.testing.assert_allclose(value, eigenvalues[0], atol=tolerance, rtol=0)
    return {"negative_centered_witness": True, "zero_sum_residual": float(abs(witness.sum())),
            "dual_equality_residual": float(abs(signs @ direction)),
            "dual_quadratic_form": value,
            "box_feasibility_at_fitted_solution_tested": False}


def validate_model(model, arrays):
    if not isinstance(model, SVC) or model.kernel != "sigmoid" or model.C != 1 or model.coef0 != 0:
        raise ValueError("Wrong saved model")
    np.testing.assert_array_equal(model.classes_, [0, 1])
    np.testing.assert_array_equal(model.support_vectors_, arrays["train_x"][model.support_].astype(np.float64))
    coefficients = model.dual_coef_.ravel()
    np.testing.assert_array_equal(np.sign(coefficients), 2 * arrays["train_observed_y"][model.support_].astype(float) - 1)
    if np.max(np.abs(coefficients)) > model.C + 1e-10 or abs(coefficients.sum()) > 1e-8:
        raise ValueError("Saved dual coefficients violate declared bounds/equality")
    return {"support_rows_verified": len(model.support_),
            "maximum_alpha": float(np.max(np.abs(coefficients))),
            "dual_coefficient_sum_absolute": float(abs(coefficients.sum()))}


def run(preparation, baseline, output):
    require_compute()
    versions = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                "sklearn": sklearn.__version__, "joblib": joblib.__version__, "threadpoolctl": threadpoolctl.__version__}
    if tuple(versions[k] for k in ("numpy", "scipy", "sklearn", "joblib", "threadpoolctl")) != ("2.5.1", "1.18.0", "1.9.0", "1.5.3", "3.6.0"):
        raise ValueError("Use the original SVM environment")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = {"status": "started", "created_utc": datetime.now(timezone.utc).isoformat(),
              "code_commit": os.environ.get("EXPECTED_COMMIT"), "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "versions": versions, "experimental_fits": 0, "replays": {},
              "source_sha256": {str(p.relative_to(STUDY)): digest(p) for p in (
                  Path(__file__), STUDY / "SVM_REPLAY.md", STUDY / "requirements-svm.txt",
                  STUDY / "reproduction/run_experiment.py", STUDY / "reproduction/analyze_results.py")}}
    path = output / "result.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    loaded, gammas, coefficients, input_hashes, snapshot = {}, [], [], {}, {}
    try:
        # Forbid even an accidental call to sklearn's SVC fitting entry point.
        with patch.object(SVC, "fit", side_effect=RuntimeError("Refitting is forbidden in this diagnostic")):
            for level, expected in RESULT_HASHES.items():
                base = baseline / level
                if digest(base / "result.json") != expected:
                    raise ValueError("Original SVM result changed")
                audit_result(base)
                original = json.loads((base / "result.json").read_text())
                arrays, metadata, identities = load_preparation(preparation / f"generalized-two-class-{level}")
                meta_path = preparation / f"generalized-two-class-{level}/metadata.json"
                snapshot[meta_path] = digest(meta_path)
                if identities != original["input_sha256"]:
                    raise ValueError("Prepared inputs do not match fitted model record")
                for name, value in identities.items():
                    snapshot[preparation / f"generalized-two-class-{level}" / (name + ".npy")] = value
                for name in ("result.json", "model.joblib", "predictions.npz"):
                    snapshot[base / name] = digest(base / name)
                input_hashes[level] = {"result_sha256": expected, "input_sha256": identities,
                                      "model_sha256": digest(base / "model.joblib")}
                model = joblib.load(base / "model.joblib")  # Trusted only after frozen hash validation.
                if model.get_params() != original["fitting_parameters"] or float(model._gamma) != original["svm"]["gamma_numeric"]:
                    raise ValueError("Saved model parameters differ from its frozen result")
                checks = validate_model(model, arrays)
                gammas.append(float(model._gamma))
                coefficients.append(float(model.coef0))
                replay_arrays, reports = {}, {"model_binding": checks}
                with np.load(base / "predictions.npz", allow_pickle=False) as saved:
                    for part in ("train", "test"):
                        x = arrays[part + "_x"]
                        native, predictions = model.decision_function(x), model.predict(x)
                        if part == "test":
                            np.testing.assert_array_equal(native, saved["decision_scores"])
                            np.testing.assert_array_equal(predictions, saved["predictions"])
                        manual = manual_scores(x, model.support_vectors_, model.dual_coef_,
                                               model.intercept_[0], model._gamma, model.coef0)
                        reports[part] = replay_summary(manual, native, predictions)
                        replay_arrays.update({part + "_manual": manual, part + "_native": native,
                                              part + "_predictions": predictions})
                reports["training_accuracy"] = {
                    "observed_label_accuracy": 100 * float(np.mean(replay_arrays["train_predictions"] == arrays["train_observed_y"])),
                    "true_label_accuracy": 100 * float(np.mean(replay_arrays["train_predictions"] == arrays["train_true_y"]))}
                for key, value in reports["training_accuracy"].items():
                    if value != original["training"][key]:
                        raise ValueError("Replayed training accuracy differs from original")
                replay_arrays.update(train_observed_labels=arrays["train_observed_y"], train_true_labels=arrays["train_true_y"])
                np.savez_compressed(output / f"{level}_replay.npz", **replay_arrays)
                result["replays"][level] = reports
                loaded[level] = arrays
                print(json.dumps({"case": level, "replay": reports}), flush=True)
            np.testing.assert_array_equal(loaded["p00"]["train_x"], loaded["p30"]["train_x"])
            if len(set(gammas)) != 1 or len(set(coefficients)) != 1:
                raise ValueError("This shared-kernel check requires identical fitted kernel parameters")
            extra = {}
            for level in ("p00", "p30"):
                data_dir = preparation / f"generalized-two-class-{level}"
                metadata = json.loads((data_dir / "metadata.json").read_text())
                for name in ("train_uid", "train_synthetic"):
                    extra_path = data_dir / (name + ".npy")
                    expected = metadata["files"][extra_path.name]
                    if digest(extra_path) != expected["sha256"]:
                        raise ValueError("Training identity file changed")
                    value = np.load(extra_path, allow_pickle=False)
                    if list(value.shape) != expected["shape"] or str(value.dtype) != expected["dtype"]:
                        raise ValueError("Training identity metadata disagrees")
                    snapshot[extra_path] = expected["sha256"]
                    if name in extra:
                        np.testing.assert_array_equal(value, extra[name])
                    extra[name] = value
            index = subset_indices(extra["train_uid"])
            labels = {level: loaded[level]["train_observed_y"][index] for level in ("p00", "p30")}
            if any(set(np.unique(y)) != {0, 1} for y in labels.values()):
                raise ValueError("Fixed subset lacks a class; do not draw another")
            matrix = kernel(loaded["p00"]["train_x"][index], loaded["p00"]["train_x"][index], gammas[0], coefficients[0])
            spectral, spectral_arrays = spectral_checks(matrix)
            result["spectral"] = spectral
            result["equality_directions"] = {level: equality_witness(matrix,
                spectral_arrays["centered_eigenvalues"], spectral_arrays["centered_witness"], y)
                for level, y in labels.items()}
            result["selection"] = selection_summary(extra["train_uid"][index], extra["train_synthetic"][index], labels)
            np.savez_compressed(output / "kernel.npz", matrix=matrix, indices=index,
                uid=extra["train_uid"][index], synthetic=extra["train_synthetic"][index],
                population_uid=extra["train_uid"], population_synthetic=extra["train_synthetic"],
                labels_p00=labels["p00"], labels_p30=labels["p30"], **spectral_arrays)
            result["kernel_parameters"] = {"gamma": gammas[0], "coef0": coefficients[0]}
            for original_path, expected in snapshot.items():
                if digest(original_path) != expected:
                    raise ValueError("An original artifact changed during read-only work")
            result["original_artifacts_unchanged"] = True
            result["original_hashes"] = input_hashes
            result["output_sha256"] = {p.name: digest(p) for p in output.glob("*.npz")}
            result["status"] = "complete"
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        result["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def audit(output):
    result = json.loads((output / "result.json").read_text())
    if result["status"] != "complete" or result["experimental_fits"] != 0:
        raise ValueError("Diagnostic did not complete as read-only")
    for name, expected in result["source_sha256"].items():
        verify_source(name, expected, result["code_commit"])
    for name, expected in result["output_sha256"].items():
        if digest(output / name) != expected:
            raise ValueError("Diagnostic array changed")
    checked_rows, observed_labels = 0, {}
    for level in ("p00", "p30"):
        with np.load(output / f"{level}_replay.npz", allow_pickle=False) as values:
            for part in ("train", "test"):
                actual = replay_summary(values[part + "_manual"], values[part + "_native"], values[part + "_predictions"])
                if actual != result["replays"][level][part]:
                    raise ValueError("Replay summary disagrees with saved vectors")
                checked_rows += actual["rows"]
            observed_labels[level] = values["train_observed_labels"]
            for name, key in (("observed", "observed_label_accuracy"), ("true", "true_label_accuracy")):
                actual = 100 * float(np.mean(values["train_predictions"] == values[f"train_{name}_labels"]))
                if actual != result["replays"][level]["training_accuracy"][key]:
                    raise ValueError("Training accuracy differs from saved labels")
    with np.load(output / "kernel.npz", allow_pickle=False) as values:
        index = subset_indices(values["population_uid"])
        np.testing.assert_array_equal(index, values["indices"])
        np.testing.assert_array_equal(values["uid"], values["population_uid"][index])
        np.testing.assert_array_equal(values["synthetic"], values["population_synthetic"][index])
        for level in ("p00", "p30"):
            np.testing.assert_array_equal(values["labels_" + level], observed_labels[level][index])
        selected = selection_summary(values["uid"], values["synthetic"],
                                      {level: values["labels_" + level] for level in ("p00", "p30")})
        if selected != result["selection"] or kernel_summary(values["matrix"]) != result["spectral"]["kernel"]:
            raise ValueError("Selection or kernel summary changed")
        for name, matrix in (("raw", values["matrix"]), ("centered", centered(values["matrix"]))):
            actual = spectrum_summary(matrix, values[name + "_eigenvalues"], values[name + "_witness"])
            # BLAS reductions may round differently across verification hosts.
            for key, expected in result["spectral"][name].items():
                if expected is None:
                    assert actual[key] is None
                else:
                    np.testing.assert_allclose(actual[key], expected, atol=1e-10, rtol=1e-10)
        for level in ("p00", "p30"):
            actual = equality_witness(values["matrix"], values["centered_eigenvalues"],
                                      values["centered_witness"], values["labels_" + level])
            for key, expected in result["equality_directions"][level].items():
                np.testing.assert_allclose(actual[key], expected, atol=1e-10, rtol=1e-10)
    return {"status": "verified", "result_sha256": digest(output / "result.json"),
            "diagnostic_arrays": len(result["output_sha256"]), "experimental_fits": 0,
            "replay_rows_checked": checked_rows, "kernel_rows": result["selection"]["rows"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if not args.audit_only:
        if args.preparation is None or args.baseline is None:
            parser.error("execution requires --preparation and --baseline")
        run(args.preparation, args.baseline, args.output)
    checked = audit(args.output)
    if not args.audit_only:
        (args.output / "artifact_audit.json").write_text(json.dumps(checked, indent=2, sort_keys=True) + "\n")
    print(json.dumps(checked, indent=2, sort_keys=True))
