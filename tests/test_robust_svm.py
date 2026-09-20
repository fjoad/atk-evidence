"""Constructed-input SVM checks; no consumption observations are fitted here."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

from tests.test_robust_baseline import M, A, R, constructed_input


def fixture(reverse=False):
    arrays = constructed_input(reverse)
    # Separate centered classes, appropriate for a scaled sigmoid-kernel witness.
    arrays["train_x"] -= 2
    arrays["test_x"] -= 2
    return arrays


class SVMTests(unittest.TestCase):
    def test_explicit_parameters_match_stock_defaults_with_printed_kernel(self):
        self.assertEqual(M.svm(42).get_params(), SVC(C=1., kernel="sigmoid",
            probability=False, random_state=42).get_params())

    def test_positive_control_matches_stock_and_persists_raw_margins(self):
        arrays = fixture()
        with tempfile.TemporaryDirectory() as directory:
            result = R.fit_one(arrays, directory, seed=42, workers=1, model_name="svm")
            stock = SVC(C=1., kernel="sigmoid", probability=False, random_state=42)
            stock.fit(arrays["train_x"], arrays["train_observed_y"])
            with np.load(Path(directory) / "predictions.npz") as scores:
                self.assertNotIn("probabilities", scores.files)
                np.testing.assert_array_equal(stock.decision_function(arrays["test_x"]), scores["decision_scores"])
                np.testing.assert_array_equal(stock.predict(arrays["test_x"]), scores["predictions"])
            self.assertEqual(result["analysis"]["primary"]["ACC"], 100.)
            self.assertEqual(result["svm"]["fit_status"], 0)
            self.assertTrue(result["reload"]["identical_decision_scores"])
            self.assertFalse(result["svm"]["probability_calibration"])
            self.assertAlmostEqual(result["svm"]["gamma_numeric"],
                1 / (48 * arrays["train_x"].astype(np.float64).var()), places=14)

    def test_fit_uses_corrupted_not_true_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            result = R.fit_one(fixture(True), directory, seed=42, workers=1, model_name="svm")
            self.assertEqual(result["training"]["observed_label_accuracy"], 100.)
            self.assertEqual(result["training"]["true_label_accuracy"], 0.)
            self.assertEqual(result["analysis"]["primary"]["AUC"], 0.)

    def test_binary_zero_margin_tie_matches_native_class_one(self):
        model = M.svm().fit([[-1.], [1.]], [0, 1])
        self.assertEqual(float(model.decision_function([[0.]])[0]), 0.)
        self.assertEqual(int(model.predict([[0.]])[0]), 1)

    def test_raw_margin_metrics_do_not_clip_or_invent_probabilities(self):
        scores = np.array([-1e6, 1e6, 0., -3., 2., .1])
        labels = np.array([0, 1, 0, 0, 1, 1])
        arrays = dict(decision_scores=scores, predictions=(scores >= 0).astype(int),
            labels=labels, synthetic=np.zeros(6, dtype=bool), attack=labels,
            negative_daily_mean=np.arange(6, dtype=float))
        result = A.analyze_scores(arrays, .5, (10.2, 25.7))
        self.assertEqual(result["primary"]["AUC"], 100 * roc_auc_score(labels, scores))
        self.assertEqual(result["primary"]["FP"], 1)
        self.assertIn("higher_decision_score", result["thresholds"])
        arrays["predictions"][2] = 0
        with self.assertRaisesRegex(ValueError, "margin"):
            A.analyze_scores(arrays, .5)
        arrays["predictions"][2] = 1
        arrays["probabilities"] = np.full((6, 2), .5)
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            A.analyze_scores(arrays, .5)

    def test_audit_uses_svm_targets_and_detects_changed_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            result = R.fit_one(fixture(), directory, seed=42, workers=1, model_name="svm")
            result.update(status="complete", case="fixture", code_commit="fixture",
                          scope="software fixture", poisoning_rate=0., model="svm")
            (Path(directory) / "result.json").write_text(json.dumps(result))
            audited = A.audit_result(directory)
            self.assertEqual(audited["reported_full_data_context"]["DR"], 89.2)
            self.assertEqual(A.reported_row(.3, "svm")["FA"], 25.7)
            with (Path(directory) / "predictions.npz").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "hash"):
                A.audit_result(directory)

    def test_nonconvergence_is_exposed_not_silently_hidden(self):
        limited = M.svm(42).set_params(max_iter=1)
        with tempfile.TemporaryDirectory() as directory, patch.object(R, "svm", return_value=limited):
            result = R.fit_svm(fixture(), directory, seed=42)
            self.assertEqual(result["svm"]["fit_status"], 1)
            self.assertEqual(result["svm"]["iterations"], [1])


if __name__ == "__main__":
    unittest.main()
