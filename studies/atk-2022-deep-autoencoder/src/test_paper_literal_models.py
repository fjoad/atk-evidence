"""Focused, tiny tests for the paper-literal Keras model interfaces."""

from __future__ import annotations

import os

# Keras selects its backend during import.  The project lock uses Torch so the
# tests remain independent of TensorFlow availability.
os.environ.setdefault("KERAS_BACKEND", "torch")

import unittest

import keras
import numpy as np

from paper_literal_models import TemporalAdditiveAttention, build_model


class PaperLiteralModelTests(unittest.TestCase):
    INPUT_LENGTH = 6

    @classmethod
    def setUpClass(cls) -> None:
        if keras.backend.backend() != "torch":
            raise RuntimeError("model tests require KERAS_BACKEND=torch")
        cls.rng = np.random.default_rng(20260721)
        cls.x = cls.rng.normal(size=(4, cls.INPUT_LENGTH)).astype(np.float32)

    def _build(self, name: str):
        width_overrides = {
            "fc_sae": (8, 6, 4, 2),
            "lstm_sae": (4, 2),
            "fc_vae": (8, 6, 4, 2),
            "lstm_vae": (4, 2),
            "lstm_aea": (5, 3, 2),
            "supervised_feed_forward": (8, 7, 6, 5, 4),
            # Four very narrow ReLU-LSTM layers can initialize to an entirely
            # inactive path on some valid Torch versions, making a one-step
            # update test falsely report that training is broken. Keep this
            # tiny, but wide enough to exercise a representative live path.
            "supervised_lstm": (16, 16, 16, 16),
        }
        return build_model(
            name,
            self.INPUT_LENGTH,
            {"encoder_widths": width_overrides[name]},
            seed=11,
        )

    def assertFiniteTrainResult(self, result) -> None:
        if isinstance(result, dict):
            values = list(result.values())
        elif isinstance(result, (tuple, list)):
            values = list(result)
        else:
            values = [result]
        self.assertTrue(np.isfinite(np.asarray(values, dtype=float)).all())

    def snapshotWeights(self, model) -> list[np.ndarray]:
        return [
            np.array(keras.ops.convert_to_numpy(weight), copy=True)
            for weight in model.trainable_weights
        ]

    def assertWeightsChanged(
        self, model, before: list[np.ndarray], message: str
    ) -> None:
        after = [
            np.asarray(keras.ops.convert_to_numpy(weight))
            for weight in model.trainable_weights
        ]
        self.assertEqual(len(before), len(after))
        self.assertTrue(
            any(not np.array_equal(old, new) for old, new in zip(before, after)),
            message,
        )

    def test_reconstruction_shapes_ranges_and_registered_scores(self) -> None:
        for name in ("fc_sae", "lstm_sae", "fc_vae", "lstm_vae", "lstm_aea"):
            with self.subTest(model=name):
                bundle = self._build(name)
                prediction = np.asarray(bundle.model.predict(self.x, verbose=0))
                self.assertEqual(prediction.shape, self.x.shape)
                self.assertTrue(np.isfinite(prediction).all())
                self.assertGreaterEqual(float(prediction.min()), 0.0)
                self.assertLessEqual(float(prediction.max()), 1.0)
                scores = bundle.anomaly_scores(self.x, batch_size=2)
                expected = (
                    {"reconstruction_mse", "mse_plus_kl_surrogate"}
                    if name.endswith("vae")
                    else {"reconstruction_mse"}
                )
                self.assertEqual(set(scores), expected)
                for values in scores.values():
                    self.assertEqual(values.shape, (len(self.x),))
                    self.assertTrue(np.isfinite(values).all())

    def test_table_softmax_outputs_sum_to_one(self) -> None:
        for name in ("fc_sae", "fc_vae"):
            with self.subTest(model=name):
                output = np.asarray(self._build(name).model.predict(self.x, verbose=0))
                np.testing.assert_allclose(output.sum(axis=1), 1.0, atol=1e-5)

        classifier = self._build("supervised_feed_forward")
        probabilities = np.asarray(classifier.model.predict(self.x, verbose=0))
        self.assertEqual(probabilities.shape, (len(self.x), 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)

    def test_paper_optimizers_activations_dropout_and_mirroring(self) -> None:
        fc_sae = self._build("fc_sae")
        self.assertIsInstance(fc_sae.model.optimizer, keras.optimizers.Adam)
        self.assertEqual(fc_sae.model.get_layer("reconstruction").activation.__name__, "softmax")
        self.assertEqual(
            [fc_sae.model.get_layer(f"encoder_dense_{i}").units for i in range(1, 5)],
            [8, 6, 4, 2],
        )
        self.assertEqual(
            [fc_sae.model.get_layer(f"decoder_dense_{i}").units for i in range(1, 4)],
            [4, 6, 8],
        )
        self.assertTrue(
            all(
                layer.rate == 0.4
                for layer in fc_sae.model.layers
                if isinstance(layer, keras.layers.Dropout)
            )
        )

        lstm_sae = self._build("lstm_sae")
        encoder_lstm = lstm_sae.model.get_layer("encoder_lstm_1")
        self.assertIsInstance(lstm_sae.model.optimizer, keras.optimizers.Adam)
        self.assertEqual(encoder_lstm.activation.__name__, "sigmoid")
        self.assertEqual(encoder_lstm.recurrent_activation.__name__, "sigmoid")
        self.assertEqual(encoder_lstm.dropout, 0.2)
        self.assertEqual(encoder_lstm.recurrent_dropout, 0.0)
        self.assertEqual(
            lstm_sae.model.get_layer("decoder_output").layer.activation.__name__,
            "sigmoid",
        )

        fc_vae = self._build("fc_vae")
        self.assertIsInstance(fc_vae.model.optimizer, keras.optimizers.Adam)
        self.assertEqual(fc_vae.model.get_layer("encoder_dense_1").activation.__name__, "relu")
        self.assertEqual(
            fc_vae.decoder.get_layer("decoder_reconstruction").activation.__name__,
            "softmax",
        )

        lstm_vae = self._build("lstm_vae")
        self.assertIsInstance(lstm_vae.model.optimizer, keras.optimizers.SGD)
        self.assertEqual(lstm_vae.model.get_layer("encoder_lstm_1").activation.__name__, "tanh")
        self.assertEqual(
            lstm_vae.model.get_layer("decoder_output").layer.activation.__name__,
            "sigmoid",
        )

        aea = self._build("lstm_aea")
        self.assertIsInstance(aea.model.optimizer, keras.optimizers.SGD)
        self.assertEqual(aea.model.get_layer("encoder_lstm_1").activation.__name__, "sigmoid")
        self.assertEqual(aea.model.get_layer("encoder_lstm_1").dropout, 0.0)

        feed_forward = self._build("supervised_feed_forward")
        self.assertIsInstance(feed_forward.model.optimizer, keras.optimizers.Adamax)
        self.assertEqual(
            feed_forward.model.get_layer("hidden_dense_1").activation.__name__, "relu"
        )
        recurrent = self._build("supervised_lstm")
        self.assertIsInstance(recurrent.model.optimizer, keras.optimizers.Adam)
        self.assertEqual(recurrent.model.get_layer("hidden_lstm_1").activation.__name__, "relu")
        self.assertEqual(recurrent.model.get_layer("hidden_lstm_1").dropout, 0.2)

    def test_vae_mse_plus_kl_surrogate_includes_nonnegative_analytic_kl(self) -> None:
        for name in ("fc_vae", "lstm_vae"):
            with self.subTest(model=name):
                scores = self._build(name).anomaly_scores(self.x)
                kl = (
                    scores["mse_plus_kl_surrogate"]
                    - scores["reconstruction_mse"]
                )
                self.assertTrue(np.isfinite(kl).all())
                self.assertTrue((kl >= -1e-6).all())

    def test_attention_is_trainable_and_normalized_for_every_decoder_step(self) -> None:
        bundle = self._build("lstm_aea")
        self.assertIsNotNone(bundle.attention_model)
        weights = np.asarray(bundle.attention_model.predict(self.x, verbose=0))
        self.assertEqual(
            weights.shape,
            (len(self.x), self.INPUT_LENGTH, self.INPUT_LENGTH),
        )
        self.assertTrue((weights >= 0.0).all())
        np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-5)
        attention_layer = bundle.model.get_layer("temporal_attention")
        self.assertGreater(len(attention_layer.trainable_weights), 0)

    def test_attention_weights_depend_on_decoder_query(self) -> None:
        attention = TemporalAdditiveAttention(attention_units=2)
        encoder_sequence = np.asarray(
            [[[0.0, 1.0], [1.0, 0.0], [2.0, -1.0]]], dtype=np.float32
        )
        decoder_queries = np.asarray(
            [[[0.0, 0.0], [1.0, 0.0]]], dtype=np.float32
        )
        # Build the layer, then make its projections deterministic so this test
        # discriminates query-conditioned attention from a static context.
        attention((encoder_sequence, decoder_queries))
        attention.sequence_projection.set_weights(
            [np.eye(2, dtype=np.float32), np.zeros(2, dtype=np.float32)]
        )
        attention.query_projection.set_weights([np.eye(2, dtype=np.float32)])
        attention.alignment_projection.set_weights(
            [np.ones((2, 1), dtype=np.float32)]
        )
        _, weights = attention((encoder_sequence, decoder_queries))
        weights = np.asarray(keras.ops.convert_to_numpy(weights))
        self.assertEqual(weights.shape, (1, 2, 3))
        self.assertFalse(np.allclose(weights[:, 0, :], weights[:, 1, :]))

    def test_supervised_output_shapes_and_ranges(self) -> None:
        feed_forward = self._build("supervised_feed_forward")
        recurrent = self._build("supervised_lstm")
        ff_output = np.asarray(feed_forward.model.predict(self.x, verbose=0))
        lstm_output = np.asarray(recurrent.model.predict(self.x, verbose=0))
        self.assertEqual(ff_output.shape, (len(self.x), 2))
        self.assertEqual(lstm_output.shape, (len(self.x), 1))
        self.assertTrue(((ff_output >= 0.0) & (ff_output <= 1.0)).all())
        self.assertTrue(((lstm_output >= 0.0) & (lstm_output <= 1.0)).all())
        with self.assertRaisesRegex(ValueError, "supervised model"):
            recurrent.anomaly_scores(self.x)

    def test_default_encoder_widths_match_the_frozen_table(self) -> None:
        cases = {
            "fc_sae": ("encoder_dense", [400, 300, 200, 100]),
            "lstm_sae": ("encoder_lstm", [500, 300]),
            "lstm_vae": ("encoder_lstm", [400, 300]),
            "lstm_aea": ("encoder_lstm", [500, 300, 200]),
            "supervised_feed_forward": (
                "hidden_dense",
                [500, 500, 500, 500, 500],
            ),
            "supervised_lstm": ("hidden_lstm", [300, 300, 300, 300]),
        }
        for name, (prefix, expected) in cases.items():
            with self.subTest(model=name):
                keras.backend.clear_session()
                model = build_model(name, self.INPUT_LENGTH, seed=11).model
                observed = [
                    model.get_layer(f"{prefix}_{index}").units
                    for index in range(1, len(expected) + 1)
                ]
                self.assertEqual(observed, expected)

        keras.backend.clear_session()
        fc_vae = build_model("fc_vae", self.INPUT_LENGTH, seed=11)
        observed_fc_vae = [
            fc_vae.model.get_layer(f"encoder_dense_{index}").units
            for index in range(1, 4)
        ] + [fc_vae.model.get_layer("z_mean").units]
        self.assertEqual(observed_fc_vae, [500, 400, 300, 100])

    def test_finite_one_step_training_for_all_families(self) -> None:
        labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
        for name in ("fc_sae", "lstm_sae", "lstm_aea"):
            with self.subTest(model=name):
                model = self._build(name).model
                before = self.snapshotWeights(model)
                result = model.train_on_batch(self.x, self.x)
                self.assertFiniteTrainResult(result)
                self.assertWeightsChanged(model, before, f"{name} weights did not change")
        for name in ("fc_vae", "lstm_vae"):
            with self.subTest(model=name):
                # VAE reconstruction and KL losses are attached by VAELoss;
                # no target is needed or silently added a second time.
                model = self._build(name).model
                before = self.snapshotWeights(model)
                result = model.train_on_batch(self.x)
                self.assertFiniteTrainResult(result)
                self.assertWeightsChanged(model, before, f"{name} weights did not change")
        model = self._build("supervised_feed_forward").model
        before = self.snapshotWeights(model)
        result = model.train_on_batch(self.x, labels)
        self.assertFiniteTrainResult(result)
        self.assertWeightsChanged(
            model, before, "supervised_feed_forward weights did not change"
        )
        model = self._build("supervised_lstm").model
        before = self.snapshotWeights(model)
        result = model.train_on_batch(
            self.x, labels.astype(np.float32).reshape(-1, 1)
        )
        self.assertFiniteTrainResult(result)
        self.assertWeightsChanged(model, before, "supervised_lstm weights did not change")

    def test_invalid_requests_fail_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown model"):
            build_model("imaginary", self.INPUT_LENGTH)
        with self.assertRaisesRegex(ValueError, "input_length"):
            build_model("fc_sae", 0)
        with self.assertRaisesRegex(ValueError, "dropout"):
            build_model("fc_sae", self.INPUT_LENGTH, {"dropout": 1.0})


if __name__ == "__main__":
    unittest.main()
