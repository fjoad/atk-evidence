from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import keras
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPRODUCTION = ROOT / "studies/atk-2022-deep-autoencoder/reproduction"
sys.path.insert(0, str(REPRODUCTION))

import models  # noqa: E402
import remaining_models  # noqa: E402
import run_experiment  # noqa: E402


class RemainingPaperPilotTests(unittest.TestCase):
    def test_checksum_fallback_is_bound_to_anchor_and_metadata_hashes(self) -> None:
        metadata_sha = "a" * 64
        metadata = {"files": {"x.npy": {"sha256": "b" * 64}}}
        with tempfile.TemporaryDirectory() as temporary:
            anchor = Path(temporary) / "result.json"
            anchor.write_text(
                json.dumps(
                    {
                        "data": {
                            "metadata_sha256": metadata_sha,
                            "files": {"order.npy": "c" * 64},
                        }
                    }
                )
            )
            anchor_sha = run_experiment.sha256(anchor)
            expected, sources, identity = (
                run_experiment.remaining_pilot_checksum_manifest(
                    metadata,
                    metadata_sha,
                    ("x.npy", "order.npy"),
                    anchor_result_path=anchor,
                    anchor_result_sha256=anchor_sha,
                )
            )
            self.assertEqual(expected, {"x.npy": "b" * 64, "order.npy": "c" * 64})
            self.assertEqual(
                sources,
                {
                    "x.npy": "prepared_metadata",
                    "order.npy": "eligible_anchor_result",
                },
            )
            self.assertEqual(identity["sha256"], anchor_sha)
            with self.assertRaisesRegex(ValueError, "different prepared metadata"):
                run_experiment.remaining_pilot_checksum_manifest(
                    metadata,
                    "d" * 64,
                    ("order.npy",),
                    anchor_result_path=anchor,
                    anchor_result_sha256=anchor_sha,
                )
            with self.assertRaisesRegex(ValueError, "record drifted"):
                run_experiment.remaining_pilot_checksum_manifest(
                    metadata,
                    metadata_sha,
                    ("order.npy",),
                    anchor_result_path=anchor,
                    anchor_result_sha256="e" * 64,
                )

    def test_committed_checksum_anchor_is_exact(self) -> None:
        self.assertEqual(
            run_experiment.sha256(run_experiment.PILOT_CHECKSUM_ANCHOR),
            run_experiment.PILOT_CHECKSUM_ANCHOR_SHA256,
        )
        anchor = json.loads(run_experiment.PILOT_CHECKSUM_ANCHOR.read_text())
        self.assertEqual(
            anchor["data"]["files"]["table_iv_order.npy"],
            "f5acf853f2efdcc5e237cdf137cd17590fc0712f8151798de65f5b034f351643",
        )
        metadata_path = (
            ROOT
            / "studies/atk-2022-deep-autoencoder/results/"
            "clean_reader_anchor_20260831/metadata.json"
        )
        metadata = json.loads(metadata_path.read_text())
        expected, sources, identity = (
            run_experiment.remaining_pilot_checksum_manifest(
                metadata,
                run_experiment.sha256(metadata_path),
                ("x_train.npy", "table_iv_order.npy"),
            )
        )
        self.assertEqual(sources["x_train.npy"], "prepared_metadata")
        self.assertEqual(
            sources["table_iv_order.npy"], "eligible_anchor_result"
        )
        self.assertEqual(
            expected["table_iv_order.npy"],
            anchor["data"]["files"]["table_iv_order.npy"],
        )
        self.assertEqual(
            identity["metadata_sha256"], run_experiment.sha256(metadata_path)
        )

    def test_slurm_wrapper_cannot_submit_a_full_anchor(self) -> None:
        wrapper = (REPRODUCTION / "run_remaining_model_pilot.sbatch").read_text()
        for required in (
            "#SBATCH -p gpu-short",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --cpus-per-task=16",
            "#SBATCH --mem=96G",
            "#SBATCH --time=02:00:00",
            '${EXPECTED_COMMIT:',
            '${MODEL:',
            "--contract remaining-paper-feasibility-v1",
            "--epochs 2",
            "--batch-size 32",
        ):
            self.assertIn(required, wrapper)
        self.assertNotIn("remaining-paper-anchor", wrapper)
        self.assertNotIn("--epochs 100", wrapper)

    def test_stochastic_roles_have_distinct_reproducible_streams(self) -> None:
        first = run_experiment.remaining_pilot_seed_streams(20260824)
        repeated = run_experiment.remaining_pilot_seed_streams(20260824)
        self.assertEqual(first, repeated)
        self.assertEqual(len(first), 4)
        self.assertEqual(len(set(first.values())), 4)

    def test_approved_pilot_contract_is_exact(self) -> None:
        args = argparse.Namespace(
            model="lstm_vae",
            seed=20260824,
            epochs=2,
            minimum_epochs=2,
            batch_size=32,
            score_batch=256,
            patience=5,
            min_delta=1e-6,
            learning_rate=None,
            output_activation="paper",
            train_fraction="full",
            test_view="adasyn",
            table_v=False,
        )
        metadata = {
            "method": "CR-ISET-FCSAE-01-DATA",
            "configuration": {
                "contract": "clean-reader-v1",
                "mode": "full",
                "seed": 20260824,
                "test_adasyn": "printed",
                "adasyn_neighbors": 5,
                "source_branch": "sciencedb-csv-semantic-equivalence-v1",
                "attack_3_completion": "duration_first_in_day",
                "malicious_test_population": "b2",
                "expensive_adasyn_acknowledged": True,
            },
        }
        self.assertEqual(
            run_experiment.remaining_pilot_contract_errors(args, metadata), []
        )
        args.batch_size = 512
        self.assertIn(
            "batch_size",
            "\n".join(
                run_experiment.remaining_pilot_contract_errors(args, metadata)
            ),
        )

    def test_pilot_selection_keeps_every_attack_sibling(self) -> None:
        group = run_experiment.PILOT_SOURCE_DAYS
        test_rows = 7 * group + run_experiment.PILOT_SYNTHETIC_ROWS
        order = np.arange(run_experiment.PILOT_TRAIN_ROWS, dtype=np.int64)
        selected = run_experiment.remaining_pilot_selection(
            order,
            test_rows=test_rows,
            original_group_rows=group,
            seed=20260824,
        )
        labels = np.concatenate(
            (
                np.zeros(group, dtype=np.int8),
                np.ones(6 * group, dtype=np.int8),
                np.zeros(run_experiment.PILOT_SYNTHETIC_ROWS, dtype=np.int8),
            )
        )
        attacks = np.concatenate(
            (
                np.repeat(np.arange(7), group),
                np.full(run_experiment.PILOT_SYNTHETIC_ROWS, -1, dtype=np.int8),
            )
        )
        sources = np.concatenate(
            (
                np.tile(np.arange(group, dtype=np.int64), 7),
                np.full(
                    run_experiment.PILOT_SYNTHETIC_ROWS, -1, dtype=np.int64
                ),
            )
        )
        run_experiment.verify_remaining_pilot_identities(
            selected,
            labels=labels,
            attacks=attacks,
            sources=sources,
            original_group_rows=group,
        )
        self.assertEqual(selected["fit"].size, 32_768)
        self.assertEqual(selected["score"].size, 12_119)

    def test_remaining_models_have_frozen_topology_and_finite_forward_pass(self) -> None:
        values = np.zeros((1, 48), dtype=np.float32)
        for name in remaining_models.REMAINING_MODELS:
            with self.subTest(model=name):
                bundle = remaining_models.build_remaining_paper_model(name, seed=11)
                self.assertEqual(
                    bundle.model.count_params(),
                    remaining_models.REMAINING_PARAMETER_COUNTS[name],
                )
                output = keras.ops.convert_to_numpy(
                    bundle.model(values, training=False)
                )
                self.assertEqual(output.shape, values.shape)
                self.assertTrue(np.isfinite(output).all())

    def test_vae_score_draws_are_invariant_to_safe_batch_partition(self) -> None:
        values = np.random.default_rng(11).normal(size=(5, 48)).astype(np.float32)
        bundle = remaining_models.build_remaining_paper_model("fc_vae", seed=11)
        first, _ = run_experiment.score_remaining_bundle(
            bundle,
            values,
            batch_size=5,
            monte_carlo_samples=3,
            scoring_seed=20260824,
        )
        second, _ = run_experiment.score_remaining_bundle(
            bundle,
            values,
            batch_size=2,
            monte_carlo_samples=3,
            scoring_seed=20260824,
        )
        self.assertEqual(set(first), set(second))
        for name in first:
            np.testing.assert_allclose(first[name], second[name], atol=1e-6)

    def test_vae_training_objective_is_exactly_one_summed_eq10_term(self) -> None:
        values = np.random.default_rng(4).normal(size=(3, 48)).astype(np.float32)
        bundle = remaining_models.build_remaining_paper_model("fc_vae", seed=7)
        reconstruction = keras.ops.convert_to_numpy(
            bundle.model(values, training=False)
        )
        mean, log_variance = [
            keras.ops.convert_to_numpy(value)
            for value in bundle.encoder(values, training=False)
        ]
        reconstruction_term = np.sum(
            np.square(values - reconstruction), axis=1
        )
        kl = -0.5 * np.sum(
            1
            + log_variance
            - np.square(mean)
            - np.exp(log_variance),
            axis=1,
        )
        self.assertEqual(len(bundle.model.losses), 1)
        observed = float(
            keras.ops.convert_to_numpy(bundle.model.losses[0])
        )
        self.assertAlmostEqual(observed, float(np.mean(reconstruction_term + kl)), places=5)

    def test_every_remaining_model_takes_one_finite_weight_update(self) -> None:
        values = np.random.default_rng(19).normal(size=(2, 48)).astype(np.float32)
        for name in remaining_models.REMAINING_MODELS:
            with self.subTest(model=name):
                bundle = remaining_models.build_remaining_paper_model(name, seed=19)
                before = run_experiment.weight_digest(bundle.model)
                loss = (
                    bundle.model.train_on_batch(values)
                    if name.endswith("_vae")
                    else bundle.model.train_on_batch(values, values)
                )
                self.assertTrue(np.isfinite(float(loss)))
                self.assertNotEqual(before, run_experiment.weight_digest(bundle.model))


if __name__ == "__main__":
    unittest.main()
