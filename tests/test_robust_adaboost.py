"""Constructed AdaBoost fixtures; run full class in requirements-adaboost env."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification

from tests.test_robust_baseline import M, A, R, constructed_input


class AdaBoostContractTests(unittest.TestCase):
    def test_wrong_version_is_rejected(self):
        with patch.object(M.sklearn, "__version__", "1.9.0"):
            with self.assertRaisesRegex(RuntimeError, "1.5.2"):
                M.adaboost()

    def test_target_row_is_adaboost_not_forest(self):
        self.assertEqual(A.reported_row(0., "adaboost")["DR"], 85.7)
        self.assertEqual(A.reported_row(.3, "adaboost")["FA"], 29.9)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            A.reported_row(0., "unknown")

    def test_source_checks_accept_historical_revision_but_not_wrong_bytes(self):
        commit = "48e979a2b10dd7343c9b8b4ed94bd7c76ae78674"
        path = "reproduction/models.py"
        expected = "e91370ea27d4ca38c8e2fd54bbfebce816c214431292713e929fec97a2bc8399"
        A.verify_source(path, expected, commit)
        with self.assertRaisesRegex(ValueError, "revision mismatch"):
            A.verify_source(path, "0" * 64, commit)
        with self.assertRaisesRegex(ValueError, "Invalid source path"):
            A.verify_source("../../README.md", expected, commit)
        with self.assertRaisesRegex(ValueError, "revision mismatch"):
            A.verify_source(path, expected, "not-a-commit")


@unittest.skipUnless(sklearn.__version__ == "1.5.2", "requires isolated AdaBoost 1.5.2 environment")
class AdaBoostFitTests(unittest.TestCase):
    def test_stock_historical_parameters(self):
        model = M.adaboost(42)
        self.assertEqual(model.get_params(), AdaBoostClassifier(
            n_estimators=50, algorithm="SAMME.R", learning_rate=1., random_state=42).get_params())

    def test_positive_control_can_legitimately_stop_before_fifty(self):
        with tempfile.TemporaryDirectory() as directory:
            result = R.fit_one(constructed_input(), directory, seed=42, workers=1, model_name="adaboost")
            self.assertEqual(result["analysis"]["primary"]["ACC"], 100.)
            self.assertEqual(result["boosting"]["trees"], 1)
            self.assertEqual(result["boosting"]["depth_max"], 1)
            self.assertTrue(result["boosting"]["stopped_before_maximum"])
            self.assertTrue(result["reload"]["identical_probabilities"])
            self.assertEqual(result["false_alarm_caps"], [14.1, 17.6, 29.9, 33.3])

    def test_corrupted_labels_are_actually_fitted(self):
        with tempfile.TemporaryDirectory() as directory:
            result = R.fit_one(constructed_input(True), directory, seed=42, workers=1, model_name="adaboost")
            self.assertEqual(result["training"]["observed_label_accuracy"], 100.)
            self.assertEqual(result["training"]["true_label_accuracy"], 0.)
            self.assertEqual(result["analysis"]["primary"]["ACC"], 0.)

    def test_multiple_rounds_match_stock_library_and_survive_audit(self):
        values, labels = make_classification(n_samples=180, n_features=48,
            n_informative=5, n_redundant=2, flip_y=.1, random_state=7)
        values = values.astype(np.float32)
        arrays = constructed_input()
        arrays.update(train_x=values[:120], train_true_y=labels[:120],
            train_observed_y=labels[:120], test_x=values[120:], test_raw_x=values[120:],
            test_true_y=labels[120:], test_observed_y=labels[120:])
        with tempfile.TemporaryDirectory() as directory:
            result = R.fit_one(arrays, directory, seed=42, workers=1, model_name="adaboost")
            self.assertEqual(result["boosting"]["trees"], 50)
            self.assertTrue(np.isfinite(result["boosting"]["estimator_errors"]).all())
            stock = AdaBoostClassifier(n_estimators=50, algorithm="SAMME.R", random_state=42)
            stock.fit(values[:120], labels[:120])
            with np.load(Path(directory) / "predictions.npz") as scores:
                np.testing.assert_array_equal(stock.predict_proba(values[120:]), scores["probabilities"])
            result.update(status="complete", case="fixture", code_commit="fixture",
                scope="constructed fixture", poisoning_rate=0., model="adaboost")
            (Path(directory) / "result.json").write_text(json.dumps(result))
            audited = A.audit_result(directory)
            self.assertEqual(audited["reported_full_data_context"]["DR"], 85.7)
            self.assertFalse(audited["comparison_is_reproduction"])


if __name__ == "__main__":
    unittest.main()
