from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from branch_lattice import (
    covered_pairs,
    enumerate_lattice,
    load_lattice,
    pairwise_cases,
    required_pairs,
    resolve_branch,
    stable_branch_id,
    storage_summary,
    write_summary,
)
from branch_runtime import (
    assert_branch_scope,
    load_runtime_branch,
    runtime_from_resolved_branch,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "branch_lattice.toml"
AMBIGUITY_REGISTER = ROOT / "AMBIGUITY_REGISTER.md"


class PairwiseCoverageTests(unittest.TestCase):
    def test_all_options_and_pairs_are_covered(self) -> None:
        options = {
            "a": ["a0", "a1", "a2"],
            "b": ["b0", "b1"],
            "c": ["c0", "c1", "c2", "c3"],
        }
        names = list(options)
        rows = pairwise_cases(names, options)
        observed = set().union(*(covered_pairs(row) for row in rows))
        self.assertEqual(observed, required_pairs(names, options))
        for name in names:
            self.assertEqual({row[name] for row in rows}, set(options[name]))

    def test_branch_ids_are_stable_and_choice_sensitive(self) -> None:
        first = stable_branch_id("family", {"a": "1", "b": "2"})
        reordered = stable_branch_id("family", {"b": "2", "a": "1"})
        changed = stable_branch_id("family", {"a": "2", "b": "2"})
        self.assertEqual(first, reordered)
        self.assertNotEqual(first, changed)

    def test_incompatible_pairs_are_excluded_without_losing_allowed_coverage(
        self,
    ) -> None:
        options = {
            "rule": ["printed", "derived"],
            "labels": ["none", "generated"],
            "scope": ["transferred", "per_dataset"],
        }
        incompatible = {
            ("labels", "none", "rule", "derived"),
            ("rule", "printed", "scope", "per_dataset"),
        }
        rows = pairwise_cases(list(options), options, incompatible)
        self.assertTrue(
            all(not (covered_pairs(row) & incompatible) for row in rows)
        )
        observed = set().union(*(covered_pairs(row) for row in rows))
        self.assertEqual(
            observed,
            required_pairs(list(options), options, incompatible),
        )


class FrozenLatticeTests(unittest.TestCase):
    def test_manifest_validates_and_every_family_has_coverage(self) -> None:
        summary = enumerate_lattice(load_lattice(CONFIG))
        self.assertTrue(summary["coverage"]["verified"])
        self.assertGreater(summary["budget"]["point_estimate"]["semantic_cases"], 0)
        self.assertEqual(
            len({branch["branch_id"] for branch in summary["branches"]}),
            len(summary["branches"]),
        )
        self.assertTrue(
            all(family["pairwise_coverage_verified"] for family in summary["families"])
        )
        self.assertEqual(
            len(summary["printed_anchor_branch_ids"]), len(summary["families"])
        )
        self.assertEqual(
            sum(branch["track"] == "P_anchor" for branch in summary["branches"]),
            len(summary["families"]),
        )
        self.assertEqual(
            set(summary["promotion"]["targets"]),
            {family["family"] for family in summary["families"]},
        )
        self.assertGreaterEqual(len(summary["non_executable"]), 3)
        self.assertGreaterEqual(len(summary["exclusions"]), 4)
        self.assertEqual(
            set(summary["coverage"]["required_ambiguity_ids"]),
            set(summary["coverage"]["items"]),
        )
        registered_ids = set(
            re.findall(
                r"^\| (A\d{2}) \|",
                AMBIGUITY_REGISTER.read_text(encoding="utf-8"),
                flags=re.MULTILINE,
            )
        )
        self.assertEqual(set(summary["coverage"]["items"]), registered_ids)
        self.assertEqual(
            len(summary["corrected_controls"]["branches"]),
            len(summary["families"]),
        )
        self.assertTrue(
            all(
                branch["track"] == "C"
                for branch in summary["corrected_controls"]["branches"]
            )
        )

    def test_summary_round_trip_is_plain_json(self) -> None:
        summary = enumerate_lattice(load_lattice(CONFIG))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.json"
            write_summary(summary, output)
            self.assertTrue(output.read_text(encoding="utf-8").endswith("\n"))
            stored = storage_summary(summary)
            self.assertNotIn("branches", stored)
            self.assertNotIn("branches", stored["corrected_controls"])
            self.assertEqual(
                stored["branch_inventory"]["paper_consistent"]["count"],
                len(summary["branches"]),
            )
            self.assertEqual(
                stored["branch_inventory"]["corrected"]["count"],
                len(summary["corrected_controls"]["branches"]),
            )

    def test_stable_ids_resolve_to_complete_runtime_records(self) -> None:
        summary = enumerate_lattice(load_lattice(CONFIG))
        selected = [
            summary["branches"][0],
            summary["branches"][-1],
            summary["corrected_controls"]["branches"][0],
        ]
        for branch in selected:
            resolved = resolve_branch(CONFIG, branch["branch_id"])
            self.assertEqual(resolved["branch_id"], branch["branch_id"])
            self.assertIn(resolved["dataset"], {"sgcc", "iset"})
            self.assertTrue(resolved["model"])
            self.assertTrue(resolved["choices"])
        with self.assertRaisesRegex(KeyError, "unknown"):
            resolve_branch(CONFIG, "p1-does-not-exist")

    def test_runtime_mapping_preserves_data_model_and_execution_choices(self) -> None:
        summary = enumerate_lattice(load_lattice(CONFIG))
        selected = next(
            branch
            for branch in summary["branches"]
            if branch["family"] == "iset_fc_vae"
            and branch["choices"]["threshold_scope"] == "dataset_specific"
        )
        runtime = load_runtime_branch(
            selected["branch_id"],
            manifest=CONFIG,
        )
        self.assertEqual(runtime["dataset"], "iset")
        self.assertEqual(runtime["model"], "fc_vae")
        self.assertEqual(
            runtime["preparation"]["attack_population"],
            selected["choices"]["attack_population"],
        )
        self.assertEqual(
            runtime["model_overrides"]["vae_score"],
            selected["choices"]["vae_score"],
        )
        self.assertEqual(
            runtime["execution"]["threshold_scope"],
            "dataset_specific",
        )
        assert_branch_scope(runtime, dataset="iset", model="fc_vae", table=3)
        with self.assertRaisesRegex(ValueError, "not sgcc"):
            assert_branch_scope(runtime, dataset="sgcc")

    def test_every_frozen_branch_maps_to_a_complete_runtime_contract(self) -> None:
        summary = enumerate_lattice(load_lattice(CONFIG))
        for branch in [
            *summary["branches"],
            *summary["corrected_controls"]["branches"],
        ]:
            runtime = runtime_from_resolved_branch(branch)
            self.assertEqual(runtime["branch_id"], branch["branch_id"])
            self.assertTrue(runtime["preparation_id"].startswith("prep-"))
            self.assertIsInstance(runtime["preparation"], dict)
            self.assertIsInstance(runtime["model_overrides"], dict)
            self.assertIsInstance(runtime["execution"], dict)
            self.assertFalse(
                any(
                    value is None
                    for value in runtime["preparation"].values()
                )
            )
        sgcc_family = next(
            family
            for family in summary["families"]
            if family["family"] == "sgcc_multiclass_svm"
        )
        self.assertNotIn("multiclass_labels", sgcc_family["dimensions"])

    def test_every_paper_dimension_is_consumed_by_an_execution_surface(self) -> None:
        summary = enumerate_lattice(load_lattice(CONFIG))
        preparation_aliases = {
            "scaling": "scaling",
            "anomaly_adasyn": "anomaly_adasyn",
            "supervised_adasyn": "supervised_adasyn",
            "adasyn_neighbors": "adasyn_neighbors",
            "split_unit": "split_unit",
            "sgcc_representation": "representation",
            "sgcc_missing": "missing",
            "iset_day": "iset_day",
            "iset_meter_population": "meter_population",
            "attack_population": "attack_population",
            "attack1_scope": "attack1_scope",
            "attack2_granularity": "attack2_granularity",
            "attack3_interval": "attack3_interval",
            "attack_hour_mapping": "attack_hour_mapping",
            "attack_regeneration": "attack_regeneration",
        }
        model_dimensions = {
            "latent_width",
            "latent_placement",
            "dense_dropout_scope",
            "lstm_input",
            "decoder_schedule",
            "decoder_state",
            "attention_merge",
            "vae_loss_reduction",
            "vae_score",
            "lstm_dropout_placement",
            "supervised_head",
        }
        execution_dimensions = {
            "validation_policy",
            "threshold_rule",
            "threshold_scope",
            "validation_labels",
            "table_v_identity",
            "table_v_size",
            "arima_completion",
            "svm_training",
            "multiclass_labels",
        }
        regeneration_values = {
            "fixed_per_data_seed": "fixed_per_data_seed",
            "per_model_seed": "regenerate_per_model_seed",
            "per_experiment": "regenerate_per_experiment",
        }

        for branch in summary["branches"]:
            runtime = runtime_from_resolved_branch(branch)
            choices = branch["choices"]
            for dimension in branch["dimensions"]:
                with self.subTest(
                    branch_id=branch["branch_id"],
                    dimension=dimension,
                ):
                    if dimension in preparation_aliases:
                        actual = runtime["preparation"][
                            preparation_aliases[dimension]
                        ]
                        expected = choices[dimension]
                        if dimension == "adasyn_neighbors":
                            expected = int(expected)
                        elif dimension == "attack_regeneration":
                            expected = regeneration_values[expected]
                        self.assertEqual(actual, expected)
                    elif dimension in model_dimensions:
                        actual = runtime["model_overrides"][dimension]
                        expected = choices[dimension]
                        if dimension == "latent_width":
                            expected = int(expected)
                        self.assertEqual(actual, expected)
                    elif dimension in execution_dimensions:
                        self.assertEqual(
                            runtime["execution"][dimension],
                            choices[dimension],
                        )
                    else:
                        self.fail(
                            f"unconsumed branch dimension {dimension!r}"
                        )

    def test_corrected_controls_name_only_scores_the_runner_executes(self) -> None:
        summary = enumerate_lattice(load_lattice(CONFIG))
        expected_scores = {
            "fc_sae": "reconstruction_mse",
            "lstm_sae": "reconstruction_mse",
            "fc_vae": (
                "monte_carlo_gaussian_reconstruction_probability_"
                "learned_variance_mc100"
            ),
            "lstm_vae": (
                "monte_carlo_gaussian_reconstruction_probability_"
                "learned_variance_mc100"
            ),
            "lstm_aea": "reconstruction_mse",
            "arima": "pooled_p1_gaussian_residual_negative_log_likelihood",
            "one_class_svm": "negative_decision_function",
            "naive_bayes": "positive_class_probability",
            "multiclass_svm": "non_benign_decision_margin",
            "supervised_feed_forward": "binary_network_probability",
            "supervised_lstm": "binary_network_probability",
        }
        for branch in summary["corrected_controls"]["branches"]:
            runtime = runtime_from_resolved_branch(branch)
            model = branch["model"]
            with self.subTest(branch_id=branch["branch_id"], model=model):
                self.assertEqual(
                    branch["choices"]["score"],
                    expected_scores[model],
                )
                self.assertEqual(
                    branch["choices"]["model_selection"],
                    "validation_selected_epochs_no_test_access",
                )
                self.assertEqual(
                    runtime["execution"]["threshold_rule"],
                    "validation_youden_j",
                )
                self.assertEqual(
                    runtime["execution"]["validation_policy"],
                    "holdout_no_refit",
                )
                self.assertEqual(
                    runtime["preparation"]["scaling"],
                    "train_benign_only",
                )
                if model.endswith("_vae"):
                    self.assertEqual(
                        runtime["model_overrides"]["vae_score"],
                        "prob_learned_var_mc100",
                    )
                if model == "arima":
                    self.assertEqual(
                        runtime["execution"]["arima_completion"],
                        "p1_pooled_likelihood",
                    )


if __name__ == "__main__":
    unittest.main()
