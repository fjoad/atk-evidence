"""Hand-checkable score and spectrum witnesses, never research observations."""

import ast
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from contextlib import redirect_stdout
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_robust_baseline import A, R, M
from tests.test_robust_svm import fixture

PATH = Path(__file__).resolve().parents[1] / "studies/takiddin-2021-robust-poisoning/checks/svm_replay.py"
SPEC = importlib.util.spec_from_file_location("robust_svm_replay", PATH)
D = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = D
with patch.dict(sys.modules, {"run_experiment": R, "analyze_results": A}):
    SPEC.loader.exec_module(D)


class ReplayTests(unittest.TestCase):
    def test_hand_calculation_and_chunks(self):
        x = np.array([[1., 0.], [0., 2.], [1., -1.]])
        support = np.array([[2., 0.], [1., 1.]])
        expected = np.array([.5 * np.tanh(.2 * (v @ support[0]) + .1)
                             - .5 * np.tanh(.2 * (v @ support[1]) + .1) + .3 for v in x])
        for chunk in (1, 2, 128):
            np.testing.assert_allclose(D.manual_scores(x, support, [.5, -.5], .3, .2, .1, chunk), expected, atol=1e-15)

    def test_native_model_binding_replay_and_wrong_intercept(self):
        arrays = fixture()
        model = M.svm(42).fit(arrays["train_x"], arrays["train_observed_y"])
        self.assertEqual(D.validate_model(model, arrays)["support_rows_verified"], len(model.support_))
        x = arrays["test_x"]
        native, predictions = model.decision_function(x), model.predict(x)
        manual = D.manual_scores(x, model.support_vectors_, model.dual_coef_, model.intercept_[0], model._gamma, model.coef0)
        self.assertEqual(D.replay_summary(manual, native, predictions)["prediction_disagreements"], 0)
        with self.assertRaisesRegex(ValueError, "tolerance"):
            D.replay_summary(manual + 1, native, predictions)

    def test_zero_margin_roundoff_is_explicit(self):
        result = D.replay_summary(np.array([-1e-10, 1.]), np.array([1e-10, 1.]), np.array([1, 1]))
        self.assertEqual(result["near_zero_native_margins"], 1)
        self.assertEqual(result["prediction_disagreements"], 1)
        self.assertEqual(result["disagreements_outside_near_zero"], 0)

    def test_psd_matrix_has_no_resolved_negative_eigenvalues(self):
        x = np.array([[1., 2.], [0., 1.], [-1., 0.], [3., -1.]])
        report, _ = D.spectral_checks(x @ x.T)
        self.assertEqual(report["raw"]["negative_count"], 0)
        self.assertEqual(report["centered"]["negative_count"], 0)

    def test_negative_zero_sum_direction_survives_dual_mapping(self):
        matrix = np.array([[1., 2.], [2., 1.]])
        report, arrays = D.spectral_checks(matrix)
        self.assertAlmostEqual(report["raw"]["minimum"], -1.)
        self.assertEqual(report["centered"]["negative_count"], 1)
        for labels in (np.array([0, 1]), np.array([1, 0])):
            result = D.equality_witness(matrix, arrays["centered_eigenvalues"], arrays["centered_witness"], labels)
            self.assertTrue(result["negative_centered_witness"])
            self.assertAlmostEqual(result["dual_quadratic_form"], -1.)
            self.assertLess(result["dual_equality_residual"], 1e-12)

    def test_negative_constant_direction_does_not_imply_dual_curvature(self):
        matrix = np.array([[0., -1.], [-1., 0.]])
        report, arrays = D.spectral_checks(matrix)
        self.assertEqual(report["raw"]["negative_count"], 1)
        self.assertEqual(report["centered"]["negative_count"], 0)
        result = D.equality_witness(matrix, arrays["centered_eigenvalues"], arrays["centered_witness"], np.array([0, 1]))
        self.assertFalse(result["negative_centered_witness"])

    def test_roundoff_and_invalid_matrices(self):
        small, _ = D.spectral_checks(np.diag([-1e-12, 1.]))
        large, _ = D.spectral_checks(np.diag([-1e-4, 1.]))
        self.assertEqual(small["raw"]["negative_count"], 0)
        self.assertEqual(large["raw"]["negative_count"], 1)
        for bad in (np.array([[1., 2.], [1., 1.]]), np.full((2, 2), np.nan)):
            with self.assertRaisesRegex(ValueError, "symmetric"):
                D.spectral_checks(bad)

    def test_selection_is_identity_based_not_input_order(self):
        uid = np.arange(600, dtype=np.int64) - 200
        shuffled = uid[np.random.default_rng(42).permutation(len(uid))]
        first, second = D.subset_indices(uid), D.subset_indices(shuffled)
        np.testing.assert_array_equal(uid[first], shuffled[second])
        self.assertEqual(len(np.unique(first)), 512)
        self.assertTrue(np.all(np.diff(uid[first]) > 0))
        with self.assertRaisesRegex(ValueError, "unique"):
            D.subset_indices(np.zeros(600))

    def test_compute_guard_and_no_fit_call(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            target = Path(directory) / "out"
            with self.assertRaisesRegex(RuntimeError, "Slurm"):
                D.run(Path(directory), Path(directory), target)
            self.assertFalse(target.exists())
        tree = ast.parse(inspect.getsource(D))
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                             and node.func.attr == "fit" for node in ast.walk(tree)))
        self.assertIn('"Refitting is forbidden in this diagnostic"', inspect.getsource(D.run))

    def test_incomplete_artifact_is_not_a_verified_result(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "result.json").write_text('{"status":"failed","experimental_fits":0}')
            with self.assertRaisesRegex(ValueError, "read-only"):
                D.audit(output)

    def test_complete_read_only_pipeline_and_audit_on_constructed_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preparation, baseline, output = root / "preparation", root / "baseline", root / "diagnostic"
            loaded, hashes = {}, {}
            for level in ("p00", "p30"):
                arrays = fixture()
                for name in ("train_x", "train_true_y", "train_observed_y"):
                    arrays[name] = np.concatenate([arrays[name]] * 5)
                if level == "p30":
                    arrays["train_observed_y"][1:121:2] = 0
                data = preparation / f"generalized-two-class-{level}"
                target = baseline / level
                data.mkdir(parents=True)
                target.mkdir(parents=True)
                identities = {}
                for name in R.INPUTS:
                    np.save(data / (name + ".npy"), arrays[name], allow_pickle=False)
                    identities[name] = A.digest(data / (name + ".npy"))
                files = {}
                for name, value in (("train_uid", np.arange(600)), ("train_synthetic", np.zeros(600, dtype=bool))):
                    path = data / (name + ".npy")
                    np.save(path, value, allow_pickle=False)
                    files[path.name] = {"sha256": A.digest(path), "shape": list(value.shape), "dtype": str(value.dtype)}
                metadata = {"files": files}
                (data / "metadata.json").write_text(json.dumps(metadata))
                result = R.fit_svm(arrays, target, seed=42)
                result.update(status="complete", model="svm", case=level, code_commit="fixture",
                    scope="constructed fixture", poisoning_rate=0. if level == "p00" else .3,
                    input_sha256=identities)
                (target / "result.json").write_text(json.dumps(result))
                hashes[level] = A.digest(target / "result.json")
                loaded[level] = arrays, metadata, identities
            with patch.dict(os.environ, {"SLURM_JOB_ID": "fixture", "SLURM_JOB_NODELIST": "fixture"}), \
                    patch.object(D, "RESULT_HASHES", hashes), \
                    patch.object(D, "load_preparation", side_effect=lambda p: loaded[p.name[-3:]]), \
                    redirect_stdout(io.StringIO()):
                result = D.run(preparation, baseline, output)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["experimental_fits"], 0)
            self.assertTrue(result["original_artifacts_unchanged"])
            self.assertEqual(D.audit(output)["replay_rows_checked"], 1320)
            with (output / "kernel.npz").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "array changed"):
                D.audit(output)


if __name__ == "__main__":
    unittest.main()
