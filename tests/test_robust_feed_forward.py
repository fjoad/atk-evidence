"""Constructed neural fixtures; isolated pinned TensorFlow runtime required."""

import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.test_robust_baseline import M, A, R, constructed_input

HAS_TF = (importlib.util.find_spec("tensorflow") is not None
          and version("tensorflow") == "2.16.2" and version("keras") == "3.4.1")


class FeedForwardContractTests(unittest.TestCase):
    def test_paper_target_is_not_another_baseline(self):
        self.assertEqual(A.reported_row(0., "feed_forward")["DR"], 90.8)
        self.assertEqual(A.reported_row(.3, "feed_forward")["FA"], 24.4)

    def test_printed_binary_algebra_cancels_labels(self):
        p = np.array([.2, .4, .8])
        for y in (np.zeros(3), np.ones(3), np.array([0, 1, 0])):
            np.testing.assert_allclose(-(y * np.log(p) + (1-y) * np.log(p)), -np.log(p))


@unittest.skipUnless(HAS_TF, "requires isolated TensorFlow environment")
class FeedForwardTensorFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        import keras
        cls.tf, cls.keras = tf, keras
        R.configure_tensorflow(require_gpu=False)

    def test_source_architecture_optimizer_and_parameter_count(self):
        model = M.feed_forward(42)
        self.assertEqual(model.count_params(), 1277501)
        self.assertEqual([layer.units for layer in model.layers], [500] * 6 + [1])
        self.assertEqual([layer.activation.__name__ for layer in model.layers], ["relu"] * 6 + ["sigmoid"])
        self.assertEqual(model.output_shape, (None, 1))
        self.assertEqual(type(model.optimizer).__name__, "Adamax")
        self.assertAlmostEqual(float(model.optimizer.learning_rate.numpy()), .002, places=8)
        self.assertEqual(model.loss.name, "binary_crossentropy")
        for layer in model.layers:
            self.assertEqual(layer.kernel_constraint.max_value, 3.)
            self.assertEqual(layer.kernel_constraint.axis, 0)
            self.assertIsNone(layer.bias_constraint)

    def test_literal_and_repaired_losses_have_different_label_gradients(self):
        tf = self.tf
        p = tf.Variable([[.2], [.4]], dtype=tf.float32)
        literal, repaired = [], []
        for y in (tf.zeros((2, 1)), tf.ones((2, 1))):
            with tf.GradientTape() as tape:
                loss = M.printed_classification_loss(y, p)
            literal.append(tape.gradient(loss, p).numpy())
            with tf.GradientTape() as tape:
                loss = self.keras.losses.BinaryCrossentropy()(y, p)
            repaired.append(tape.gradient(loss, p).numpy())
        np.testing.assert_array_equal(*literal)
        self.assertTrue(np.all(repaired[0] > 0) and np.all(repaired[1] < 0))

    def test_initialization_is_paired_but_layers_are_not_identical(self):
        first = M.feed_forward(42).get_weights()
        second = M.feed_forward(42).get_weights()
        self.assertEqual(R.weight_hash(first), R.weight_hash(second))
        self.assertFalse(np.array_equal(first[2], first[4]))

    def test_maxnorm_projects_incoming_vectors(self):
        model = M.feed_forward(42)
        for layer in (model.layers[0], model.layers[-1]):
            huge = self.tf.ones_like(layer.kernel) * 20
            projected = layer.kernel_constraint(huge).numpy()
            np.testing.assert_allclose(np.linalg.norm(projected, axis=0), 3., atol=1e-5)
            layer.kernel.assign(huge)
        model.train_on_batch(np.full((4, 48), .01, dtype=np.float32),
                             np.array([[0.], [1.], [0.], [1.]], dtype=np.float32))
        for layer in model.layers:
            self.assertLessEqual(float(np.linalg.norm(layer.kernel.numpy(), axis=0).max()), 3.00001)
        self.assertEqual(int(model.optimizer.iterations.numpy()), 1)

    def test_correct_and_corrupted_labels_learn_and_reload(self):
        initial_hashes = []
        for reverse in (False, True):
            arrays = constructed_input(reverse)
            arrays["train_x"] -= 2
            arrays["test_x"] -= 2
            with tempfile.TemporaryDirectory() as directory:
                result = R.fit_feed_forward(arrays, directory, seed=42, epochs=4, batch_size=16)
                self.assertEqual(result["neural"]["epochs_completed"], 4)
                self.assertEqual(result["neural"]["optimizer_updates"], 32)
                self.assertTrue(result["reload"]["identical_probabilities"])
                self.assertGreater(result["training"]["observed_label_accuracy"], 95)
                if reverse:
                    self.assertLess(result["training"]["true_label_accuracy"], 5)
                else:
                    self.assertGreater(result["analysis"]["primary"]["ACC"], 95)
                initial_hashes.append(result["neural"]["initial_weights_sha256"])
                result.update(status="complete", case="fixture", code_commit="fixture",
                    scope="constructed software fixture", poisoning_rate=0., model="feed_forward")
                (Path(directory) / "result.json").write_text(json.dumps(result))
                self.assertEqual(A.audit_result(directory)["reported_full_data_context"]["DR"], 90.8)
        self.assertEqual(*initial_hashes)

    def test_time_guard_preserves_a_partial_epoch_history(self):
        with tempfile.TemporaryDirectory() as directory:
            result = R.fit_feed_forward(constructed_input(), directory, seed=42, epochs=3,
                                       batch_size=100, fit_limit_seconds=0.)
            self.assertFalse(result["neural"]["training_complete"])
            self.assertEqual(result["neural"]["epochs_completed"], 1)
            self.assertTrue((Path(directory) / "history.json").exists())


if __name__ == "__main__":
    unittest.main()
