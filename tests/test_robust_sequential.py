"""Constructed checks of the declared sequential implementation; no CER."""

from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_robust_feed_forward import HAS_TF, M, R
from tests.test_robust_baseline import load

ROOT = Path(__file__).resolve().parents[1]
SMALL = dict(encoder_units=(8, 6, 4), gru_units=8, dense_units=16, timesteps=8)
G = load("sequential_constructed_gate", ROOT /
         "studies/takiddin-2021-robust-poisoning/checks/sequential_constructed.py")


class SequentialGateTests(unittest.TestCase):
    def test_constant_partial_and_nonfinite_results_cannot_pass(self):
        labels = np.array([0., 1.])
        constant = G.measurements(labels, [.5, .5])
        self.assertAlmostEqual(constant["bce"], np.log(2))
        self.assertEqual(constant["accuracy"], .5)
        result = {"status": "complete", "updates": 300, "test": constant}
        self.assertFalse(G.passes_learning(result))
        result["test"] = G.measurements(labels, [.01, .99])
        self.assertTrue(G.passes_learning(result))
        self.assertIs(type(G.passes_learning(result)), bool)
        self.assertEqual(json.dumps({"passed": G.passes_learning(result)}), '{"passed": true}')
        result["status"] = "time_guard"
        self.assertFalse(G.passes_learning(result))
        result.update(status="complete", updates=299)
        self.assertFalse(G.passes_learning(result))
        result.update(updates=300, test=G.measurements(labels, [np.nan, .9]))
        self.assertFalse(G.passes_learning(result))

    def test_fixed_data_are_balanced_disjoint_and_easily_separable(self):
        first, second = G.fixture_data(), G.fixture_data()
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertFalse(np.array_equal(first["train_x"], first["test_x"]))
        for split in ("train", "test"):
            self.assertEqual(first[f"{split}_x"].shape, (32, 8, 1))
            self.assertEqual(float(first[f"{split}_y"].sum()), 16.)
            prediction = first[f"{split}_x"].mean(axis=(1, 2)) > 0
            np.testing.assert_array_equal(prediction, first[f"{split}_y"].ravel())


@unittest.skipUnless(HAS_TF, "requires isolated TensorFlow environment")
class SequentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        import keras
        cls.tf, cls.keras = tf, keras
        R.configure_tensorflow(False)

    def test_full_source_inventory_and_configuration(self):
        model = M.sequential_ensemble()
        self.assertEqual(model.count_params(), 9240802)
        self.assertEqual(model.input_shape, (None, 48, 1))
        self.assertEqual(model.output_shape, (None, 1))
        front = model.get_layer("attention_decoder")
        self.assertEqual([x.units for x in front.encoder], [500, 300, 200])
        self.assertEqual([x.units for x in front.decoder], [200, 300, 500])
        grus = [x for x in model.layers if isinstance(x, self.keras.layers.GRU)]
        self.assertEqual([x.units for x in grus], [300] * 8)
        self.assertEqual([x.return_sequences for x in grus], [True] * 7 + [False])
        for layer in [*front.encoder, *front.decoder, *grus]:
            self.assertEqual(layer.activation.__name__, "relu")
            self.assertEqual(layer.recurrent_activation.__name__, "sigmoid")
            self.assertEqual(layer.dropout, 0.)
            self.assertEqual(layer.recurrent_dropout, 0.)
            self.assertEqual(layer.kernel_constraint.max_value, 1.)
            self.assertEqual(layer.recurrent_constraint.max_value, 1.)
        self.assertTrue(all(not x.reset_after for x in grus))
        self.assertEqual(front.projection.activation.__name__, "relu")
        self.assertEqual(model.get_layer("classifier_hidden").units, 500)
        self.assertEqual(model.get_layer("attack_probability").activation.__name__, "sigmoid")
        self.assertEqual(model.loss.name, "binary_crossentropy")
        self.assertEqual(type(model.optimizer).__name__, "Adam")
        self.assertAlmostEqual(float(model.optimizer.learning_rate.numpy()), .001, places=8)
        probe = self.keras.Model(model.input, [model.output, *front.output])
        probability, sequence, attention = probe(
            np.linspace(-1., 1., 96, dtype=np.float32).reshape(2, 48, 1))
        self.assertEqual(tuple(sequence.shape), (2, 48, 1))
        self.assertEqual(tuple(attention.shape), (2, 48, 48))
        self.assertTrue(np.isfinite(probability.numpy()).all())
        self.assertTrue(np.isfinite(sequence.numpy()).all())
        np.testing.assert_allclose(attention.numpy().sum(axis=-1), 1., atol=1e-6)

    def test_paired_initialization_and_shape_validation(self):
        first = M.sequential_ensemble(**SMALL).get_weights()
        second = M.sequential_ensemble(**SMALL).get_weights()
        self.assertEqual(R.weight_hash(first), R.weight_hash(second))
        with self.assertRaises(ValueError):
            M.sequential_ensemble(encoder_units=(8, 4), timesteps=8)
        model = M.sequential_ensemble(**SMALL)
        with self.assertRaises(ValueError):
            model(np.zeros((2, 7, 1), dtype=np.float32))

    def test_query_intervention_and_disabled_connection(self):
        front = M.sequential_ensemble(**SMALL).get_layer("attention_decoder")
        tf = self.tf
        # Deliberately varied memory, unchanged in every intervention.
        memory = tf.reshape(tf.linspace(-1., 1., 64), (2, 8, 4))
        projected = tf.einsum("btd,dk->btk", memory, front.attention_encoder_kernel)
        before = tf.zeros((2, 8))
        after = tf.tile(tf.linspace(-3., 3., 8)[None, :], (2, 1))
        _, a = front.attend(memory, projected, before)
        _, b = front.attend(memory, projected, after)
        self.assertGreater(float(tf.reduce_max(tf.abs(a-b))), 1e-7)
        np.testing.assert_allclose(a.numpy().sum(axis=1), 1., atol=1e-6)
        front.attention_decoder_kernel.assign(tf.zeros_like(front.attention_decoder_kernel))
        np.testing.assert_array_equal(front.attend(memory, projected, before)[1].numpy(),
                                      front.attend(memory, projected, after)[1].numpy())

    def test_feedback_changes_state_and_disabled_input_removes_effect(self):
        front = M.sequential_ensemble(**SMALL).get_layer("attention_decoder")
        tf = self.tf
        memory = tf.ones((2, 8, 4)) * .2
        projected = tf.einsum("btd,dk->btk", memory, front.attention_encoder_kernel)
        states = [[tf.ones((2, u))*.1, tf.ones((2, u))*.1] for u in front.decoder_units]
        # Controlled weights keep the witness out of a dead ReLU region.
        cell = front.decoder[0]
        cell.kernel.assign(tf.ones_like(cell.kernel)*.1)
        def state(previous):
            return front.decode_step(previous, states, memory, projected)[1][0][0].numpy()
        self.assertGreater(float(np.max(np.abs(state(tf.zeros((2, 1))) - state(tf.ones((2, 1)))))), 1e-5)
        kernel = cell.kernel.numpy()
        kernel[-1, :] = 0.
        cell.kernel.assign(kernel)
        np.testing.assert_array_equal(state(tf.zeros((2, 1))), state(tf.ones((2, 1))))

    def test_actual_decoder_loop_uses_previous_output_and_mirrored_states(self):
        front = M.sequential_ensemble(**SMALL).get_layer("attention_decoder")
        x = self.tf.ones((2, 8, 1))*.2
        memory, _, states = front.encode(x)
        value, encoder_states = x, []
        for layer in front.encoder:
            value, h, c = layer(value)
            encoder_states.append([h, c])
        np.testing.assert_array_equal(memory.numpy(), value.numpy())
        for actual, expected in zip(states, reversed(encoder_states)):
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a.numpy(), b.numpy())
        front.projection.kernel.assign(self.tf.zeros_like(front.projection.kernel))
        front.projection.bias.assign(self.tf.ones_like(front.projection.bias))
        seen = []
        original = front.decode_step
        def observe(previous, *args, **kwargs):
            seen.append(previous.numpy().copy())
            return original(previous, *args, **kwargs)
        with patch.object(front, "decode_step", observe):
            sequence, attention = front(x)
        self.assertEqual(len(seen), 8)
        np.testing.assert_array_equal(seen[0], np.zeros((2, 1)))
        for previous in seen[1:]:
            np.testing.assert_array_equal(previous, np.ones((2, 1)))
        self.assertEqual(tuple(sequence.shape), (2, 8, 1))
        self.assertEqual(tuple(attention.shape), (2, 8, 8))

    def test_finite_gradient_connectivity_update_and_constraints(self):
        model = M.sequential_ensemble(**SMALL)
        x = np.random.default_rng(20260926).normal(size=(4, 8, 1)).astype(np.float32)
        y = self.tf.constant([[0.], [1.], [0.], [1.]])
        with self.tf.GradientTape() as tape:
            loss = model.loss(y, model(x))
        gradients = tape.gradient(loss, model.trainable_weights)
        self.assertTrue(all(g is not None and np.isfinite(g.numpy()).all() for g in gradients))
        self.assertTrue(np.isfinite(model.train_on_batch(x, y)))
        self.assertEqual(int(model.optimizer.iterations.numpy()), 1)
        for variable in model.trainable_weights:
            self.assertTrue(np.isfinite(variable.numpy()).all())
            if variable.constraint is not None:
                self.assertLessEqual(float(np.linalg.norm(variable.numpy(), axis=0).max()), 1.00001)
        # Zero gradients can satisfy this software check. The separate learning
        # runner measures norms/weight changes and decides promotion.

    def test_fresh_process_reload_preserves_internal_outputs_and_optimizer(self):
        model = M.sequential_ensemble(**SMALL)
        x = np.random.default_rng(4).normal(size=(2, 8, 1)).astype(np.float32)
        model.train_on_batch(x, np.array([[0.], [1.]], dtype=np.float32))
        front = model.get_layer("attention_decoder")
        internal = self.keras.Model(model.input, front.output)
        sequence, attention = internal(x)
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            model.save(d / "model.keras")
            np.save(d / "input.npy", x)
            child = '''
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path("studies/takiddin-2021-robust-poisoning/reproduction").resolve()))
import models
from run_experiment import configure_tensorflow
configure_tensorflow(False)
import keras
models.register_sequential_layer()
d=Path(sys.argv[1]); model=keras.models.load_model(d/"model.keras")
x=np.load(d/"input.npy")
internal=keras.Model(model.input, model.get_layer("attention_decoder").output)
sequence,attention=internal(x)
np.savez(d/"reload.npz", probability=model(x).numpy(), sequence=sequence.numpy(),
 attention=attention.numpy(), iterations=model.optimizer.iterations.numpy(),
 **{f"w{i}": w for i,w in enumerate(model.get_weights())})
'''
            done = subprocess.run([sys.executable, "-c", child, str(d)], cwd=ROOT,
                                  capture_output=True, text=True, timeout=60)
            self.assertEqual(done.returncode, 0, done.stderr)
            with np.load(d / "reload.npz") as saved:
                np.testing.assert_array_equal(saved["probability"], model(x).numpy())
                np.testing.assert_array_equal(saved["sequence"], sequence.numpy())
                np.testing.assert_array_equal(saved["attention"], attention.numpy())
                self.assertEqual(int(saved["iterations"]), 1)
                for i, weight in enumerate(model.get_weights()):
                    np.testing.assert_array_equal(saved[f"w{i}"], weight)


if __name__ == "__main__":
    unittest.main()
