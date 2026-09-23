"""Constructed GRU fixtures only; no CER preparation or inference locally."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_robust_baseline import A, M, R, constructed_input, load
from tests.test_robust_feed_forward import HAS_TF

PATH = Path(__file__).resolve().parents[1] / "studies/takiddin-2021-robust-poisoning/checks/gru_preflight.py"
with patch.dict("sys.modules", {"models": M, "run_experiment": R, "analyze_results": A}):
    G = load("robust_gru_preflight", PATH)


class GRUContractTests(unittest.TestCase):
    def test_target_is_the_gru_row(self):
        self.assertEqual(A.reported_row(0., "gru")["DR"], 92.4)
        self.assertEqual(A.reported_row(.3, "gru")["FA"], 20.6)

    def test_sequence_reshape_preserves_every_value_and_rejects_wrong_axes(self):
        x = np.arange(96, dtype=np.float32).reshape(2, 48)
        shaped = R.sequence_inputs(x)
        self.assertEqual(shaped.shape, (2, 48, 1))
        np.testing.assert_array_equal(shaped[:, :, 0], x)
        for wrong in (x.astype(np.float64), x.T, shaped):
            with self.assertRaises(ValueError):
                R.sequence_inputs(wrong)

    def test_roundoff_normalization_retains_raw_argmax_and_ties(self):
        raw = np.array([[.49999997, .5], [.5, .5], [.3, .7]], dtype=np.float32)
        normalized = R.normalized_softmax(raw)
        np.testing.assert_allclose(normalized.sum(axis=1), 1., atol=1e-15)
        np.testing.assert_array_equal(normalized.argmax(axis=1), [1, 0, 1])
        for wrong in ([[.1, .1]], [[np.nan, .5]], [[-.1, 1.1]], [[.5, .4, .1]]):
            with self.assertRaises(ValueError):
                R.normalized_softmax(wrong)

    def test_runtime_gate_uses_slowest_warm_step_and_has_no_parameter_search(self):
        times = [100., 40.] + [.1] * 9 + [.3]
        self.assertEqual(G.gate(times)["projected_fit_seconds_from_slowest_warm_step"], 675.)
        self.assertTrue(G.gate(times)["passes"])
        self.assertFalse(G.gate([1.] * 12)["passes"])
        for wrong in ([1.] * 11, [0.] * 12, [np.nan] * 12):
            with self.assertRaises(ValueError):
                G.gate(wrong)

    def test_constructed_cluster_preflight_requires_allocation(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / "attempt"
            with self.assertRaisesRegex(RuntimeError, "Slurm"):
                G.run(path)
            self.assertFalse(path.exists())

    def test_saved_preflight_gate_and_identity_are_rechecked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preflight.json"
            record = {"status": "complete", "code_commit": "fixture", "source_sha256": {},
                      "research_inputs_loaded": False, "parameters": 4058702, "optimizer_updates": 12,
                      "update_seconds": [.1] * 12, "gate": G.gate([.1] * 12)}
            path.write_text(json.dumps(record))
            self.assertEqual(G.verify(path, A.digest(path), "fixture")["status"], "verified")
            record["gate"]["passes"] = False
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                G.verify(path, A.digest(path), "fixture")

    def test_runtime_exception_preserves_original_failed_gate(self):
        path = G.STUDY / "results/gru_preflight_20260923/preflight.json"
        with self.assertRaisesRegex(ValueError, "gate did not pass"):
            G.verify(path, G.APPROVED_PREFLIGHT_SHA256, G.APPROVED_PREFLIGHT_COMMIT)
        result = G.verify(path, G.APPROVED_PREFLIGHT_SHA256, G.APPROVED_PREFLIGHT_COMMIT,
                          authorized_runtime_extension=True)
        self.assertEqual(result["status"], "verified_with_authorized_exception")
        self.assertFalse(result["original_gate"]["passes"])
        self.assertEqual(result["original_gate"]["ceiling_seconds"], 720)
        self.assertEqual(result["approved_ceiling_seconds"], 900)
        self.assertTrue(result["scientific_sources_unchanged"])

    def test_runtime_exception_rejects_wrong_identity_and_scientific_drift(self):
        path = G.STUDY / "results/gru_preflight_20260923/preflight.json"
        with self.assertRaisesRegex(ValueError, "file changed"):
            G.verify(path, "wrong", G.APPROVED_PREFLIGHT_COMMIT, authorized_runtime_extension=True)
        with self.assertRaisesRegex(ValueError, "same frozen code"):
            G.verify(path, G.APPROVED_PREFLIGHT_SHA256, "wrong", authorized_runtime_extension=True)
        real_digest = G.digest
        def drifted_digest(p):
            return "changed" if Path(p).name == "models.py" else real_digest(p)
        with patch.object(G, "digest", side_effect=drifted_digest):
            with self.assertRaisesRegex(ValueError, "Scientific source changed"):
                G.verify(path, G.APPROVED_PREFLIGHT_SHA256, G.APPROVED_PREFLIGHT_COMMIT,
                         authorized_runtime_extension=True)

    def test_runtime_exception_has_fixed_identity_and_ceiling(self):
        original = G.STUDY / "results/gru_preflight_20260923/preflight.json"
        record = json.loads(original.read_text())
        record["update_seconds"] = [1.] * 12
        record["gate"] = G.gate(record["update_seconds"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "constructed.json"
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "only to the approved preflight"):
                G.verify(path, A.digest(path), G.APPROVED_PREFLIGHT_COMMIT,
                         authorized_runtime_extension=True)
            with patch.object(G, "APPROVED_PREFLIGHT_SHA256", A.digest(path)):
                with self.assertRaisesRegex(ValueError, "900-second ceiling"):
                    G.verify(path, A.digest(path), G.APPROVED_PREFLIGHT_COMMIT,
                             authorized_runtime_extension=True)


@unittest.skipUnless(HAS_TF, "requires isolated TensorFlow environment")
class GRUTensorFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        import keras
        cls.tf, cls.keras = tf, keras
        R.configure_tensorflow(require_gpu=False)

    def test_full_source_architecture_and_constraints(self):
        model = M.gru(42)
        self.assertEqual(model.count_params(), 4058702)
        self.assertEqual(model.input_shape, (None, 48, 1))
        self.assertEqual(model.output_shape, (None, 2))
        self.assertEqual([layer.units for layer in model.layers], [300] * 8 + [2])
        for i, layer in enumerate(model.layers[:-1]):
            self.assertEqual(layer.activation.__name__, "relu")
            self.assertEqual(layer.recurrent_activation.__name__, "sigmoid")
            self.assertEqual(layer.return_sequences, i < 7)
            self.assertFalse(layer.reset_after)
            self.assertFalse(layer.use_cudnn)
            self.assertFalse(layer.stateful)
            self.assertEqual(layer.dropout, .2)
            self.assertEqual(layer.recurrent_dropout, 0.)
            for constraint in (layer.kernel_constraint, layer.recurrent_constraint):
                self.assertEqual(constraint.max_value, 5.)
                self.assertEqual(constraint.axis, 0)
        self.assertEqual(model.layers[-1].activation.__name__, "softmax")
        self.assertAlmostEqual(float(model.optimizer.learning_rate.numpy()), .001, places=8)
        self.assertEqual(model.loss.name, "categorical_crossentropy")
        self.assertEqual(len(R.gru_kernel_norms(model)), 17)

    def test_native_cell_matches_reset_before_hand_calculation(self):
        cell = self.keras.layers.GRUCell(2, activation="relu", recurrent_activation="sigmoid", reset_after=False)
        cell.build((None, 2))
        w = np.arange(12, dtype=np.float32).reshape(2, 6) / 20 - .2
        u = np.arange(12, dtype=np.float32).reshape(2, 6) / 15 - .15
        bias = np.arange(6, dtype=np.float32) / 30
        cell.set_weights([w, u, bias])
        x, h = np.array([.3, -.2], np.float32), np.array([.4, -.1], np.float32)
        sig = lambda v: 1 / (1 + np.exp(-v))
        projected = x @ w + bias
        z = sig(projected[:2] + h @ u[:, :2])
        r = sig(projected[2:4] + h @ u[:, 2:4])
        candidate = np.maximum(projected[4:] + (r * h) @ u[:, 4:], 0)
        expected = z * h + (1 - z) * candidate
        actual, state = cell(self.tf.constant(x[None]), [self.tf.constant(h[None])], training=False)
        np.testing.assert_allclose(actual.numpy()[0], expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(actual.numpy(), state[0].numpy())

    def test_two_softmax_loss_repair_matches_binary_channel_mean(self):
        tf, keras = self.tf, self.keras
        logits = tf.Variable([[.1, .4], [-.7, .2]], dtype=tf.float32)
        y = tf.constant([[1., 0.], [0., 1.]])
        gradients, losses = [], []
        for loss in (keras.losses.CategoricalCrossentropy(), keras.losses.BinaryCrossentropy()):
            with tf.GradientTape() as tape:
                value = loss(y, tf.nn.softmax(logits))
            losses.append(float(value))
            gradients.append(tape.gradient(value, logits).numpy())
        np.testing.assert_allclose(losses[0], losses[1], atol=2e-7)
        np.testing.assert_allclose(*gradients, atol=2e-7)

    def test_seed_pairing_and_correct_or_corrupted_constructed_labels(self):
        y = np.arange(32) % 2
        x = np.repeat((2 * y - 1).astype(np.float32)[:, None, None], 48, axis=1)
        hashes = []
        for reverse in (False, True):
            model = M.gru(42, units=8, hidden_layers=2)
            hashes.append(R.weight_hash(model.get_weights()))
            observed = 1 - y if reverse else y
            onehot = np.eye(2, dtype=np.float32)[observed]
            dataset = self.tf.data.Dataset.from_tensors((x, onehot)).repeat(80)
            options = self.tf.data.Options()
            options.threading.private_threadpool_size = 1
            values = model.fit(dataset.with_options(options), epochs=1, steps_per_epoch=80, verbose=0)
            self.assertTrue(np.isfinite(values.history["loss"][0]))
            predictions = model(x, training=False).numpy().argmax(axis=1)
            self.assertGreater(np.mean(predictions == observed), .95)
            if reverse:
                self.assertLess(np.mean(predictions == y), .05)
        self.assertEqual(*hashes)

    def test_constraint_projection_after_a_real_update(self):
        model = M.gru(42, units=4, hidden_layers=1)
        cell = model.layers[0].cell
        cell.kernel.assign(self.tf.ones_like(cell.kernel) * 10)
        cell.recurrent_kernel.assign(self.tf.ones_like(cell.recurrent_kernel) * 10)
        model.layers[-1].kernel.assign(self.tf.ones_like(model.layers[-1].kernel) * 10)
        model.train_on_batch(np.full((2, 48, 1), .001, np.float32), np.eye(2, dtype=np.float32))
        self.assertLessEqual(max(R.gru_kernel_norms(model).values()), 5.00001)

    def test_full_architecture_learns_a_two_sequence_fixture(self):
        y = np.arange(8) % 2
        x = np.repeat((2 * y - 1).astype(np.float32)[:, None, None], 48, axis=1)
        dataset = self.tf.data.Dataset.from_tensors((x, np.eye(2, dtype=np.float32)[y])).repeat(60)
        options = self.tf.data.Options()
        options.threading.private_threadpool_size = 1
        model = M.gru(42)
        history = model.fit(dataset.with_options(options), epochs=1, steps_per_epoch=60, verbose=0)
        self.assertTrue(np.isfinite(history.history["loss"][0]))
        predictions = model(x, training=False).numpy().argmax(axis=1)
        self.assertGreater(np.mean(predictions == y), .95)

    def test_full_fit_path_preserves_batch_guard_partial_history_and_raw_outputs(self):
        arrays = {name: value[:6 if name.startswith("train_") else 4].copy()
                  for name, value in constructed_input().items()}
        with tempfile.TemporaryDirectory() as directory:
            result = R.fit_gru(arrays, directory, seed=42, epochs=2, batch_size=2, fit_limit_seconds=0.)
            self.assertFalse(result["neural"]["training_complete"])
            self.assertEqual(result["neural"]["epochs_completed"], 0)
            self.assertEqual(result["neural"]["optimizer_updates"], 1)
            self.assertTrue(result["reload"]["identical_raw_probabilities"])
            with np.load(Path(directory) / "predictions.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["probabilities"], R.normalized_softmax(saved["raw_probabilities"]))
            history = json.loads((Path(directory) / "history.json").read_text())
            self.assertEqual(len(history), 1)
            self.assertFalse(history[0]["epoch_complete"])
            result.update(status="partial", model="gru", case="fixture", code_commit="fixture",
                          scope="constructed only", poisoning_rate=0.)
            (Path(directory) / "result.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "completed fit"):
                A.audit_result(directory)


if __name__ == "__main__":
    unittest.main()
