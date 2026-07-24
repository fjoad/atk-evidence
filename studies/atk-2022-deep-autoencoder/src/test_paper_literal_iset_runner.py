"""Focused tests for the exact-ISET Tables III--V execution adapter."""

from __future__ import annotations

import contextlib
import dataclasses
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from paper_literal_iset_runner import (
    _table_iv_view,
    compact_iset_arrays,
    execute_table_v_identity,
    run_iset,
    table_v_results,
    table_v_results_from_executions,
)
from paper_literal_iset import save_prepared_iset
from branch_lattice import enumerate_lattice, load_lattice
from branch_runtime import load_runtime_branch
from paper_literal_runner import ExecutionResult
from test_paper_literal_iset import prepare_fixture


class IsetRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prepared = prepare_fixture()

    def _result(self) -> ExecutionResult:
        test = self.prepared.anomaly_test
        scores = np.full(test.labels.size, 0.1, dtype=np.float64)
        original_count = self.prepared.metadata["counts"]["anomaly_test_original"]
        scores[:original_count][test.labels[:original_count] == 1] = 0.9
        predictions = (scores >= 0.58).astype(np.int8)
        return ExecutionResult(
            scores={"reconstruction_mse": scores},
            predictions={"reconstruction_mse": predictions},
            labels=test.labels,
            sample_ids=test.sample_ids,
            is_synthetic=test.is_synthetic,
            history={"epochs_completed": 1},
            metrics={"reconstruction_mse": {"threshold": 0.58}},
            fit_seconds=1.0,
            score_seconds=0.5,
            metadata={"model_name": "fc_sae"},
        )

    def test_table_v_is_indexed_from_same_table_iii_scores(self) -> None:
        derived = table_v_results(
            self._result(), self.prepared, threshold=0.58
        )
        primary = derived["score_branches"]["reconstruction_mse"]
        self.assertEqual(derived["samples_per_class"], 7)
        self.assertTrue(primary["false_alarm_invariant"])
        for attack in primary["attacks"].values():
            self.assertEqual(attack["dr"], 1.0)
            self.assertEqual(attack["fa"], 0.0)

    def test_table_v_honors_low_probability_anomaly_orientation(self) -> None:
        result = self._result()
        scores = 1.0 - result.scores["reconstruction_mse"]
        low_result = ExecutionResult(
            **{
                **result.__dict__,
                "scores": {"reconstruction_probability": scores},
                "predictions": {
                    "reconstruction_probability": (scores <= 0.42).astype(
                        np.int8
                    )
                },
                "metadata": {
                    "model_name": "fc_vae",
                    "positive_if": {
                        "reconstruction_probability": "lower"
                    },
                },
            }
        )
        derived = table_v_results(
            low_result,
            self.prepared,
            threshold=0.42,
        )
        primary = derived["score_branches"]["reconstruction_probability"]
        self.assertEqual(primary["positive_if"], "lower")
        for attack in primary["attacks"].values():
            self.assertEqual(attack["dr"], 1.0)
            self.assertEqual(attack["fa"], 0.0)

    def test_all_table_v_identity_branches_define_model_and_row_reuse(
        self,
    ) -> None:
        contract = SimpleNamespace(thresholds={"fc_sae": 0.58})

        def fake_execute(model_name, prepared, frozen, seed, **kwargs):
            del model_name, prepared, frozen, kwargs
            result = self._result()
            return ExecutionResult(
                **{
                    **result.__dict__,
                    "history": {"seed": seed},
                }
            )

        expected_fits = {
            "common_model_common_benign": 1,
            "retrain_per_attack": 6,
            "resplit_per_attack": 1,
            "retrain_and_resplit": 6,
        }
        for identity, fit_count in expected_fits.items():
            with self.subTest(identity=identity), patch(
                "paper_literal_iset_runner.execute_selected_model",
                side_effect=fake_execute,
            ) as execute:
                result = execute_table_v_identity(
                    "fc_sae",
                    self.prepared,
                    contract,
                    11,
                    identity=identity,
                    size="full_heldout",
                )
                self.assertEqual(execute.call_count, fit_count)
                self.assertEqual(
                    result.metadata["logical_model_fits"], fit_count
                )
                self.assertEqual(
                    result.supplemental_results["table_5"][
                        "identity_branch"
                    ],
                    identity,
                )
                benign_count = self.prepared.metadata["counts"][
                    "anomaly_b2_benign"
                ]
                self.assertEqual(result.labels.size, 12 * benign_count)
                self.assertEqual(
                    np.unique(result.sample_ids).size,
                    result.sample_ids.size,
                )

    def test_branch_preflight_requires_and_accepts_matching_cache_identity(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        lattice = root / "config/branch_lattice.toml"
        summary = enumerate_lattice(load_lattice(lattice))
        branch = next(
            item
            for item in summary["branches"]
            if item["family"] == "iset_naive_bayes"
            and item["track"] == "P_anchor"
        )
        runtime = load_runtime_branch(
            branch["branch_id"],
            manifest=lattice,
        )
        prepared = dataclasses.replace(
            self.prepared,
            metadata={
                **dict(self.prepared.metadata),
                "branch_runtime": {
                    "dataset": runtime["dataset"],
                    "preparation_id": runtime["preparation_id"],
                    "preparation": runtime["preparation"],
                    "identity_policy": (
                        "content_addressed_preparation_only; model/evaluation "
                        "branch identity belongs to each run fingerprint"
                    ),
                },
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            cache = temporary / "cache"
            save_prepared_iset(prepared, cache)
            with contextlib.redirect_stdout(io.StringIO()):
                outcomes = run_iset(
                    cache_prefix=cache,
                    output=temporary / "runs",
                    config=root / "config/exploratory_iset.toml",
                    table=3,
                    models=["all"],
                    seeds=[11],
                    branch_id=runtime["branch_id"],
                    branch_manifest=lattice,
                    preflight_only=True,
                )
            self.assertEqual(outcomes, [])

    def test_full_set_resplit_is_recorded_as_a_degenerate_identity_case(
        self,
    ) -> None:
        shared = self._result()
        table, selections = table_v_results_from_executions(
            {attack_id: shared for attack_id in range(1, 7)},
            self.prepared,
            threshold=0.58,
            identity="resplit_per_attack",
            size="full_heldout",
            seed=11,
            model_seeds={attack_id: 11 for attack_id in range(1, 7)},
        )
        self.assertTrue(table["full_set_resplit_degeneracy"])
        for selected in selections.values():
            np.testing.assert_array_equal(
                selected,
                np.arange(
                    self.prepared.metadata["counts"]["anomaly_b2_benign"]
                ),
            )

    def test_compact_arrays_leave_shared_provenance_in_cache(self) -> None:
        arrays = compact_iset_arrays(self._result())
        self.assertEqual(
            set(arrays),
            {"score__reconstruction_mse", "prediction__reconstruction_mse"},
        )

    def test_table_iv_views_use_nested_declared_training_subsets(self) -> None:
        half = _table_iv_view(self.prepared, "half")
        three_quarter = _table_iv_view(self.prepared, "three_quarter")
        self.assertEqual(
            half.anomaly_train.labels.size,
            self.prepared.metadata["counts"]["table_iv_half"],
        )
        np.testing.assert_array_equal(
            half.anomaly_train.sample_ids,
            three_quarter.anomaly_train.sample_ids[
                : half.anomaly_train.sample_ids.size
            ],
        )
        self.assertIs(half.anomaly_test, self.prepared.anomaly_test)


if __name__ == "__main__":
    unittest.main()
