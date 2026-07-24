from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")

from keras import ops

from paper_literal_models import (
    GaussianVAELoss,
    VAELoss,
    build_model,
    gaussian_reconstruction_probability,
    parse_vae_score_branch,
)


SOURCE = {"architecture_contract": "paper_source_v2", "latent_width": 16}


def _units(model, prefix: str) -> list[int]:
    return [
        int(layer.units)
        for layer in model.layers
        if layer.name.startswith(prefix) and hasattr(layer, "units")
    ]


class PaperSourceArchitectureTests(unittest.TestCase):
    def test_corrected_contract_is_separate_but_uses_source_layer_topology(
        self,
    ) -> None:
        bundle = build_model(
            "fc_sae",
            8,
            {
                "architecture_contract": "corrected_control_v1",
                "encoder_widths": (6, 4),
                "latent_width": 2,
                "latent_placement": "distinct_projection",
            },
            seed=7,
        )
        self.assertEqual(_units(bundle.model, "encoder_dense_"), [6, 4])
        self.assertEqual(_units(bundle.model, "decoder_dense_"), [4, 6])

    def test_both_supervised_heads_execute_for_both_deep_classifiers(
        self,
    ) -> None:
        values = np.asarray(
            [[0.0] * 8, [0.1] * 8, [0.9] * 8, [1.0] * 8],
            dtype=np.float32,
        )
        for model_name in (
            "supervised_feed_forward",
            "supervised_lstm",
        ):
            for head in ("softmax2_categorical", "sigmoid1_binary"):
                with self.subTest(model=model_name, head=head):
                    bundle = build_model(
                        model_name,
                        8,
                        {
                            "architecture_contract": "paper_source_v2",
                            "encoder_widths": (4, 3),
                            "dropout": 0.0,
                            "supervised_head": head,
                            "lstm_input": "1_step_48_features",
                        },
                        seed=13,
                    )
                    output = ops.convert_to_numpy(
                        bundle.model(values, training=False)
                    )
                    self.assertEqual(
                        output.shape,
                        (4, 1 if head == "sigmoid1_binary" else 2),
                    )
    def test_fc_sae_has_full_mirror_and_both_latent_readings(self) -> None:
        distinct = build_model(
            "fc_sae",
            48,
            {**SOURCE, "latent_placement": "distinct_projection"},
            seed=11,
        )
        self.assertEqual(
            _units(distinct.model, "encoder_dense_"), [400, 300, 200, 100]
        )
        self.assertEqual(
            _units(distinct.model, "decoder_dense_"), [100, 200, 300, 400]
        )
        self.assertEqual(distinct.model.get_layer("latent_projection").units, 16)

        algorithm = build_model(
            "fc_sae",
            48,
            {
                **SOURCE,
                "latent_placement": "existing_bottleneck_representation",
            },
            seed=11,
        )
        with self.assertRaises(ValueError):
            algorithm.model.get_layer("latent_projection")
        self.assertEqual(
            _units(algorithm.model, "decoder_dense_"), [100, 200, 300, 400]
        )

    def test_fc_vae_uses_all_hidden_widths_around_latent_distribution(self) -> None:
        bundle = build_model("fc_vae", 48, SOURCE, seed=11)
        self.assertEqual(
            _units(bundle.model, "encoder_dense_"), [500, 400, 300, 100]
        )
        self.assertEqual(bundle.model.get_layer("z_mean").units, 16)
        self.assertEqual(bundle.model.get_layer("z_log_var").units, 16)
        self.assertIsNotNone(bundle.decoder)
        self.assertEqual(
            _units(bundle.decoder, "decoder_dense_"), [100, 300, 400, 500]
        )

    def test_recurrent_source_branches_preserve_printed_hidden_mirrors(self) -> None:
        sae = build_model(
            "lstm_sae",
            48,
            {**SOURCE, "latent_placement": "distinct_projection"},
            seed=11,
        )
        self.assertEqual(_units(sae.model, "encoder_lstm_"), [500, 300])
        self.assertEqual(_units(sae.model, "decoder_lstm_"), [300, 500])
        self.assertEqual(sae.model.get_layer("latent_projection").units, 16)

        vae = build_model("lstm_vae", 48, SOURCE, seed=11)
        self.assertEqual(_units(vae.model, "encoder_lstm_"), [400, 300])
        self.assertEqual(_units(vae.model, "decoder_lstm_"), [300, 400])
        self.assertEqual(vae.model.get_layer("z_mean").units, 16)

        aea = build_model(
            "lstm_aea",
            48,
            {**SOURCE, "latent_placement": "distinct_projection"},
            seed=11,
        )
        self.assertEqual(_units(aea.model, "encoder_lstm_"), [500, 300, 200])
        self.assertEqual(_units(aea.model, "decoder_lstm_"), [200, 300, 500])
        self.assertEqual(
            aea.model.get_layer("latent_projection").layer.units, 16
        )

    def test_dropout_placement_branches_change_only_declared_arguments(self) -> None:
        dense = build_model(
            "fc_sae",
            8,
            {
                **SOURCE,
                "encoder_widths": (12, 10, 8, 6),
                "latent_placement": "distinct_projection",
                "dense_dropout_scope": "encoder_hidden_layers",
            },
            seed=11,
        )
        dropout_names = {
            layer.name
            for layer in dense.model.layers
            if layer.__class__.__name__ == "Dropout"
        }
        self.assertEqual(
            dropout_names,
            {
                "encoder_dense_1_dropout",
                "encoder_dense_2_dropout",
                "encoder_dense_3_dropout",
                "encoder_dense_4_dropout",
            },
        )

        recurrent = build_model(
            "lstm_sae",
            8,
            {
                **SOURCE,
                "encoder_widths": (6, 4),
                "lstm_dropout_placement": "recurrent_only",
            },
            seed=11,
        )
        lstm_layers = [
            layer
            for layer in recurrent.model.layers
            if layer.__class__.__name__ == "LSTM"
        ]
        self.assertTrue(lstm_layers)
        self.assertTrue(all(layer.dropout == 0.0 for layer in lstm_layers))
        self.assertTrue(all(layer.recurrent_dropout == 0.2 for layer in lstm_layers))

    def test_every_registered_vae_score_has_explicit_direction_and_budget(self) -> None:
        for variance in ("learned", "fixed"):
            for samples in (1, 10, 100):
                name = f"prob_{variance}_var_mc{samples}"
                spec = parse_vae_score_branch(name)
                self.assertEqual(spec.kind, "reconstruction_probability")
                self.assertEqual(spec.positive_if, "lower")
                self.assertEqual(spec.variance, variance)
                self.assertEqual(spec.monte_carlo_samples, samples)
        self.assertEqual(
            parse_vae_score_branch("reconstruction_mse_high").positive_if,
            "higher",
        )
        self.assertEqual(
            parse_vae_score_branch("mse_plus_kl_high").positive_if,
            "higher",
        )

    def test_gaussian_reconstruction_probability_supports_both_variance_readings(
        self,
    ) -> None:
        target = np.array([[0.0], [2.0]], dtype=np.float64)
        means = np.array(
            [
                [[0.0], [0.0]],
                [[1.0], [2.0]],
            ],
            dtype=np.float64,
        )
        fixed = gaussian_reconstruction_probability(
            target,
            means,
            fixed_variance=1.0,
        )
        learned = gaussian_reconstruction_probability(
            target,
            means,
            decoded_log_variances=np.zeros_like(means),
        )
        for key in (
            "reconstruction_probability",
            "log_reconstruction_probability",
            "negative_log_reconstruction_probability",
        ):
            np.testing.assert_allclose(fixed[key], learned[key], atol=1e-12)
            self.assertEqual(fixed[key].shape, (2,))
            self.assertTrue(np.isfinite(fixed[key]).all())
        self.assertGreater(
            fixed["reconstruction_probability"][0],
            fixed["reconstruction_probability"][1],
        )

    def test_gaussian_probability_rejects_an_unidentified_variance(self) -> None:
        target = np.zeros((2, 3))
        means = np.zeros((1, 2, 3))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            gaussian_reconstruction_probability(target, means)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            gaussian_reconstruction_probability(
                target,
                means,
                decoded_log_variances=np.zeros_like(means),
                fixed_variance=1.0,
            )

    def test_fixed_variance_vae_probability_executes_monte_carlo_draws(self) -> None:
        config = {
            **SOURCE,
            "encoder_widths": (8, 6, 4, 2),
        }
        values = np.asarray(
            [[0.0, 0.2, 0.3, 0.5], [0.5, 0.3, 0.2, 0.0]],
            dtype=np.float32,
        )
        first = build_model("fc_vae", 4, config, seed=11)
        second = build_model("fc_vae", 4, config, seed=11)
        first_scores = first.vae_reconstruction_probability(
            values,
            monte_carlo_samples=10,
            variance="fixed",
            fixed_variance=1.0,
            batch_size=2,
        )
        second_scores = second.vae_reconstruction_probability(
            values,
            monte_carlo_samples=10,
            variance="fixed",
            fixed_variance=1.0,
            batch_size=2,
        )
        for key, score in first_scores.items():
            self.assertEqual(score.shape, (2,))
            self.assertTrue(np.isfinite(score).all())
            np.testing.assert_allclose(score, second_scores[key], atol=1e-12)

    def test_learned_decoder_variance_branch_trains_and_scores(self) -> None:
        values = np.asarray(
            [[0.0, 0.2, 0.3, 0.5], [0.5, 0.3, 0.2, 0.0]],
            dtype=np.float32,
        )
        for name, widths in (
            ("fc_vae", (8, 6, 4, 2)),
            ("lstm_vae", (6, 4)),
        ):
            with self.subTest(model=name):
                bundle = build_model(
                    name,
                    4,
                    {
                        **SOURCE,
                        "encoder_widths": widths,
                        "vae_score": "prob_learned_var_mc10",
                    },
                    seed=11,
                )
                result = bundle.model.train_on_batch(values, None)
                self.assertTrue(np.isfinite(np.asarray(result)).all())
                scores = bundle.vae_reconstruction_probability(
                    values,
                    monte_carlo_samples=3,
                    variance="learned",
                    batch_size=2,
                )
                self.assertEqual(
                    set(scores),
                    {
                        "reconstruction_probability",
                        "log_reconstruction_probability",
                        "negative_log_reconstruction_probability",
                    },
                )
                self.assertTrue(
                    all(
                        score.shape == (2,) and np.isfinite(score).all()
                        for score in scores.values()
                    )
                )

    def test_eq10_sum_squared_and_mean_mse_reductions_are_distinct(self) -> None:
        target = np.zeros((2, 4), dtype=np.float32)
        reconstruction = np.ones_like(target)
        z_mean = np.zeros((2, 3), dtype=np.float32)
        z_log_var = np.zeros_like(z_mean)

        mean_layer = VAELoss(reduction="mean_mse_plus_kl")
        mean_layer((target, reconstruction, z_mean, z_log_var))
        summed_layer = VAELoss(reduction="sum_squared_plus_kl")
        summed_layer((target, reconstruction, z_mean, z_log_var))

        mean_loss = float(ops.convert_to_numpy(mean_layer.losses[0]))
        summed_loss = float(ops.convert_to_numpy(summed_layer.losses[0]))
        self.assertAlmostEqual(mean_loss, 1.0, places=6)
        self.assertAlmostEqual(summed_loss, 4.0, places=6)

    def test_learned_variance_loss_uses_the_matching_feature_reduction(self) -> None:
        target = np.zeros((2, 4), dtype=np.float32)
        reconstruction = np.ones_like(target)
        log_variance = np.zeros_like(target)
        z_mean = np.zeros((2, 3), dtype=np.float32)
        z_log_var = np.zeros_like(z_mean)

        mean_layer = GaussianVAELoss(reduction="mean_mse_plus_kl")
        mean_layer(
            (target, reconstruction, log_variance, z_mean, z_log_var)
        )
        summed_layer = GaussianVAELoss(reduction="sum_squared_plus_kl")
        summed_layer(
            (target, reconstruction, log_variance, z_mean, z_log_var)
        )

        mean_loss = float(ops.convert_to_numpy(mean_layer.losses[0]))
        summed_loss = float(ops.convert_to_numpy(summed_layer.losses[0]))
        self.assertAlmostEqual(summed_loss, 4.0 * mean_loss, places=5)

    def test_every_vae_builder_records_and_executes_both_loss_reductions(
        self,
    ) -> None:
        values = np.asarray(
            [[0.0, 0.2, 0.3, 0.5], [0.5, 0.3, 0.2, 0.0]],
            dtype=np.float32,
        )
        for name, widths in (
            ("fc_vae", (8, 6, 4, 2)),
            ("lstm_vae", (6, 4)),
        ):
            for score, layer_name in (
                ("prob_fixed_var_mc1", "vae_loss"),
                ("prob_learned_var_mc1", "gaussian_vae_loss"),
            ):
                for reduction in (
                    "sum_squared_plus_kl",
                    "mean_mse_plus_kl",
                ):
                    with self.subTest(
                        model=name,
                        score=score,
                        reduction=reduction,
                    ):
                        bundle = build_model(
                            name,
                            4,
                            {
                                **SOURCE,
                                "encoder_widths": widths,
                                "vae_score": score,
                                "vae_loss_reduction": reduction,
                            },
                            seed=11,
                        )
                        loss_layer = bundle.model.get_layer(layer_name)
                        self.assertEqual(loss_layer.reduction, reduction)
                        result = bundle.model.train_on_batch(values, None)
                        self.assertTrue(np.isfinite(np.asarray(result)).all())

        with self.assertRaisesRegex(ValueError, "vae_loss_reduction"):
            build_model(
                "fc_vae",
                4,
                {
                    **SOURCE,
                    "encoder_widths": (8, 6, 4, 2),
                    "vae_loss_reduction": "silent_repair",
                },
                seed=11,
            )

    def test_decoder_schedule_state_and_lstm_input_branches_are_structural(
        self,
    ) -> None:
        base = {
            **SOURCE,
            "encoder_widths": (6, 4),
            "latent_placement": "existing_bottleneck_representation",
        }
        repeated = build_model(
            "lstm_sae",
            8,
            {
                **base,
                "decoder_schedule": "repeat_latent",
                "decoder_state": "mirrored_layer_states",
            },
            seed=11,
        )
        first_only = build_model(
            "lstm_sae",
            8,
            {
                **base,
                "decoder_schedule": "first_latent_then_zero",
                "decoder_state": "top_state_only",
            },
            seed=11,
        )
        self.assertEqual(repeated.model.get_layer("decoder_repeat").output.shape[1], 8)
        self.assertEqual(
            first_only.model.get_layer("decoder_first_then_zero").output.shape[1],
            8,
        )
        autoregressive = build_model(
            "lstm_sae",
            8,
            {
                **base,
                "decoder_schedule": "autoregressive_reconstruction",
                "decoder_state": "mirrored_layer_states",
            },
            seed=11,
        )
        values = np.zeros((2, 8), dtype=np.float32)
        reconstruction = np.asarray(
            autoregressive.model.predict(values, verbose=0)
        )
        self.assertEqual(reconstruction.shape, values.shape)
        self.assertTrue(np.isfinite(reconstruction).all())

        autoregressive_vae = build_model(
            "lstm_vae",
            8,
            {
                **SOURCE,
                "encoder_widths": (6, 4),
                "decoder_schedule": "autoregressive_reconstruction",
                "decoder_state": "top_state_only",
            },
            seed=11,
        )
        vae_reconstruction = np.asarray(
            autoregressive_vae.model.predict(values, verbose=0)
        )
        self.assertEqual(vae_reconstruction.shape, values.shape)
        self.assertTrue(np.isfinite(vae_reconstruction).all())

        conventional = build_model(
            "lstm_sae",
            8,
            {**base, "lstm_input": "48_steps_1_feature"},
            seed=11,
        )
        one_step = build_model(
            "lstm_sae",
            8,
            {**base, "lstm_input": "1_step_48_features"},
            seed=11,
        )
        self.assertEqual(
            tuple(conventional.model.get_layer("encoder_sequence").output.shape[1:]),
            (8, 1),
        )
        self.assertEqual(
            tuple(one_step.model.get_layer("encoder_sequence").output.shape[1:]),
            (1, 8),
        )

    def test_attention_merge_runs_both_textual_readings(self) -> None:
        base = {
            **SOURCE,
            "encoder_widths": (6, 4, 2),
            "latent_placement": "existing_bottleneck_representation",
        }
        concatenated = build_model(
            "lstm_aea",
            8,
            {**base, "attention_merge": "additive_concat"},
            seed=11,
        )
        summed = build_model(
            "lstm_aea",
            8,
            {**base, "attention_merge": "literal_sum"},
            seed=11,
        )
        self.assertEqual(
            concatenated.model.get_layer("attention_context_and_decoder").output.shape[-1],
            4,
        )
        self.assertEqual(
            summed.model.get_layer("attention_context_plus_decoder").output.shape[-1],
            2,
        )

    def test_autoregressive_attention_executes_textual_and_state_branches(
        self,
    ) -> None:
        values = np.asarray(
            [
                [0.0, 0.2, 0.3, 0.5, 0.7, 0.4, 0.2, 0.1],
                [0.5, 0.3, 0.2, 0.0, 0.1, 0.4, 0.6, 0.8],
            ],
            dtype=np.float32,
        )
        base = {
            **SOURCE,
            "encoder_widths": (6, 4, 2),
            "decoder_schedule": "autoregressive_reconstruction",
            "latent_placement": "existing_bottleneck_representation",
        }
        for merge in ("additive_concat", "literal_sum"):
            for state in ("mirrored_layer_states", "top_state_only"):
                with self.subTest(merge=merge, state=state):
                    bundle = build_model(
                        "lstm_aea",
                        8,
                        {
                            **base,
                            "attention_merge": merge,
                            "decoder_state": state,
                        },
                        seed=11,
                    )
                    reconstruction = np.asarray(
                        bundle.model.predict(values, verbose=0)
                    )
                    weights = np.asarray(
                        bundle.attention_model.predict(values, verbose=0)
                    )
                    self.assertEqual(reconstruction.shape, values.shape)
                    self.assertTrue(np.isfinite(reconstruction).all())
                    self.assertEqual(weights.shape, (2, 8, 8))
                    np.testing.assert_allclose(
                        weights.sum(axis=-1),
                        1.0,
                        atol=1e-5,
                    )
                    decoder = bundle.model.get_layer(
                        "autoregressive_attention_decoder"
                    )
                    self.assertEqual(decoder.attention_merge, merge)
                    self.assertEqual(decoder.state_policy, state)

        distinct = build_model(
            "lstm_aea",
            8,
            {
                **base,
                "latent_placement": "distinct_projection",
                "latent_width": 3,
            },
            seed=11,
        )
        distinct_decoder = distinct.model.get_layer(
            "autoregressive_attention_decoder"
        )
        self.assertEqual(distinct_decoder.latent_projection.units, 3)
        train_result = distinct.model.train_on_batch(values, values)
        self.assertTrue(np.isfinite(np.asarray(train_result)).all())

        one_step = build_model(
            "lstm_aea",
            8,
            {
                **base,
                "lstm_input": "1_step_48_features",
            },
            seed=11,
        )
        one_step_weights = np.asarray(
            one_step.attention_model.predict(values, verbose=0)
        )
        self.assertEqual(one_step_weights.shape, (2, 8, 1))


if __name__ == "__main__":
    unittest.main()
