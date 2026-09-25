"""Constructed AEA architecture/attention fixtures; no CER data."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_robust_feed_forward import HAS_TF, R

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAS_TF, "requires isolated TensorFlow environment")
class AEAModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        R.configure_tensorflow(require_gpu=False)
        spec = importlib.util.spec_from_file_location(
            "aea_model_fixture", ROOT / "studies/takiddin-2021-robust-poisoning/checks/aea_model.py")
        cls.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.m)
        cls.tf = __import__("tensorflow")

    def test_parameter_inventory_and_shapes(self):
        model = self.m.build_aea(42, reconstruction_loss="mse")
        self.assertEqual(model.count_params(), 5031701)
        self.assertEqual(model.input_shape, (None, 48, 1))
        self.assertEqual(model.output_shape, (None, 48, 1))
        self.assertEqual(self.m.parameter_formula()["total"], 5031701)
        self.assertEqual(model.loss.name, "mean_squared_error")

    def test_output_range_and_attention_normalization(self):
        model = self.m.build_aea(42, reconstruction_loss="mae", encoder_units=(8, 6, 4), decoder_units=(4, 6, 8))
        x = np.linspace(0., 1., 96, dtype=np.float32).reshape(2, 48, 1)
        output, attention = self.m.attention_weights(model, x)
        self.assertEqual(tuple(output.shape), (2, 48, 1))
        self.assertEqual(tuple(attention.shape), (2, 48, 48))
        self.assertTrue(np.isfinite(output.numpy()).all())
        self.assertTrue(np.all((output.numpy() >= 0.) & (output.numpy() <= 1.)))
        np.testing.assert_allclose(attention.numpy().sum(axis=-1), 1., atol=1e-6)

    def test_query_intervention_holds_encoder_memory_fixed(self):
        model = self.m.build_aea(42, reconstruction_loss="mae", encoder_units=(8, 6, 4), decoder_units=(4, 6, 8))
        layer = model.get_layer("aea_recurrent")
        x = np.linspace(0., 1., 96, dtype=np.float32).reshape(2, 48, 1)

        def first_attention(intervene):
            original = layer.encoder[0].call
            calls = []

            def modified(inputs, states, training=False):
                output, state = original(inputs, states, training=training)
                calls.append(1)
                # Only the final returned state changes. Encoder outputs/memory
                # are unchanged; the altered state is the first decoder query.
                if intervene and len(calls) == layer.steps:
                    state = [state[0] + self.tf.linspace(-3., 3., 8)[None, :], state[1]]
                return output, state

            with patch.object(layer.encoder[0], "call", modified):
                _, weights = self.m.attention_weights(model, x)
            self.assertEqual(len(calls), layer.steps)
            return weights.numpy()[:, 0, :]

        before, after = first_attention(False), first_attention(True)
        self.assertGreater(float(np.max(np.abs(before - after))), 1e-7)
        layer.attention_decoder_kernel.assign(self.tf.zeros_like(layer.attention_decoder_kernel))
        np.testing.assert_array_equal(first_attention(False), first_attention(True))

    def test_invalid_shape_and_finite_update(self):
        model = self.m.build_aea(42, reconstruction_loss="mae", encoder_units=(8, 6, 4), decoder_units=(4, 6, 8))
        x = np.linspace(0., 1., 96, dtype=np.float32).reshape(2, 48, 1)
        initial = model.get_weights()
        history = model.fit(x, x, epochs=2, batch_size=2, shuffle=False, verbose=0)
        self.assertTrue(np.isfinite(history.history["loss"]).all())
        self.assertEqual(int(model.optimizer.iterations.numpy()), 2)
        self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(initial, model.get_weights())))
        for variable in model.trainable_variables:
            self.assertTrue(np.isfinite(variable.numpy()).all())
            if variable.constraint is not None:
                self.assertLessEqual(float(np.linalg.norm(variable.numpy(), axis=0).max()), 1.00001)
        with self.assertRaises(Exception):
            model(np.zeros((2, 47, 1), dtype=np.float32))

    def test_mae_and_mse_are_separate_declared_objectives(self):
        mae = self.m.build_aea(42, encoder_units=(4, 3, 2), decoder_units=(2, 3, 4), reconstruction_loss="mae")
        mse = self.m.build_aea(42, encoder_units=(4, 3, 2), decoder_units=(2, 3, 4), reconstruction_loss="mse")
        self.assertEqual(mae.loss.name, "mean_absolute_error")
        self.assertEqual(mse.loss.name, "mean_squared_error")
        with self.assertRaises(TypeError):
            self.m.build_aea(42)

    def test_serialization_in_fresh_process_preserves_predictions_attention_and_optimizer(self):
        model = self.m.build_aea(42, reconstruction_loss="mae", encoder_units=(4, 3, 2), decoder_units=(2, 3, 4), timesteps=4)
        x = np.linspace(0., 1., 8, dtype=np.float32).reshape(2, 4, 1)
        model.train_on_batch(x, x)
        output, attention = self.m.attention_weights(model, x)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            model.save(directory / "model.keras")
            np.savez(directory / "input.npz", x=x)
            child = '''
import sys
from pathlib import Path
import numpy as np
from tests.test_robust_baseline import R
R.configure_tensorflow(require_gpu=False)
sys.path.insert(0, str(Path("studies/takiddin-2021-robust-poisoning/checks").resolve()))
import aea_model
import keras
d = Path(sys.argv[1])
model = keras.models.load_model(d / "model.keras")
with np.load(d / "input.npz") as f:
    output, attention = aea_model.attention_weights(model, f["x"])
assert model.loss.name == "mean_absolute_error"
np.savez(d / "reload.npz", output=output.numpy(), attention=attention.numpy(),
         iterations=model.optimizer.iterations.numpy(), **{f"w{i}": w for i, w in enumerate(model.get_weights())})
'''
            completed = subprocess.run([sys.executable, "-c", child, str(directory)],
                                       cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with np.load(directory / "reload.npz") as saved:
                np.testing.assert_array_equal(saved["output"], output.numpy())
                np.testing.assert_array_equal(saved["attention"], attention.numpy())
                self.assertEqual(int(saved["iterations"]), 1)
                for i, weight in enumerate(model.get_weights()):
                    np.testing.assert_array_equal(saved[f"w{i}"], weight)


if __name__ == "__main__":
    unittest.main()
