"""Constructed AEA architecture/attention fixtures; no CER data."""

from pathlib import Path
import unittest

import numpy as np

from tests.test_robust_feed_forward import HAS_TF

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAS_TF, "requires isolated TensorFlow environment")
class AEAModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "aea_model_fixture", ROOT / "studies/takiddin-2021-robust-poisoning/checks/aea_model.py")
        cls.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.m)
        cls.tf = __import__("tensorflow")

    def test_parameter_inventory_and_shapes(self):
        model = self.m.build_aea(42)
        self.assertEqual(model.count_params(), 5031701)
        self.assertEqual(model.input_shape, (None, 48, 1))
        self.assertEqual(model.output_shape, (None, 48, 1))
        self.assertEqual(self.m.parameter_formula()["total"], 5031701)
        self.assertEqual(model.loss.name, "mean_absolute_error")

    def test_output_range_attention_normalization_and_query_dependence(self):
        model = self.m.build_aea(42, encoder_units=(8, 6, 4), decoder_units=(4, 6, 8))
        x = np.linspace(0., 1., 96, dtype=np.float32).reshape(2, 48, 1)
        output, attention = self.m.attention_weights(model, x)
        self.assertEqual(tuple(output.shape), (2, 48, 1))
        self.assertEqual(tuple(attention.shape), (2, 48, 48))
        self.assertTrue(np.isfinite(output.numpy()).all())
        self.assertTrue(np.all((output.numpy() >= 0.) & (output.numpy() <= 1.)))
        np.testing.assert_allclose(attention.numpy().sum(axis=-1), 1., atol=1e-6)
        changed = x.copy(); changed[0, 0, 0] = 1 - changed[0, 0, 0]
        _, attention_changed = self.m.attention_weights(model, changed)
        self.assertGreater(float(np.max(np.abs(attention.numpy() - attention_changed.numpy()))), 1e-7)

    def test_invalid_shape_and_finite_update(self):
        model = self.m.build_aea(42, encoder_units=(8, 6, 4), decoder_units=(4, 6, 8))
        x = np.linspace(0., 1., 96, dtype=np.float32).reshape(2, 48, 1)
        history = model.fit(x, x, epochs=2, batch_size=2, shuffle=False, verbose=0)
        self.assertTrue(np.isfinite(history.history["loss"]).all())
        with self.assertRaises(Exception):
            model(np.zeros((2, 47, 1), dtype=np.float32))

    def test_mae_and_mse_are_separate_declared_objectives(self):
        mae = self.m.build_aea(42, encoder_units=(4, 3, 2), decoder_units=(2, 3, 4), reconstruction_loss="mae")
        mse = self.m.build_aea(42, encoder_units=(4, 3, 2), decoder_units=(2, 3, 4), reconstruction_loss="mse")
        self.assertEqual(mae.loss.name, "mean_absolute_error")
        self.assertEqual(mse.loss.name, "mean_squared_error")


if __name__ == "__main__":
    unittest.main()
