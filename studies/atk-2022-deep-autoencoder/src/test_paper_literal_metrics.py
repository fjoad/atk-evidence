from __future__ import annotations

import math
import unittest

import numpy as np

from paper_literal_metrics import (
    aggregate_seed_metrics,
    evaluate_attack_columns,
    evaluate_binary_scores,
    select_threshold,
)


class PaperLiteralMetricTests(unittest.TestCase):
    def test_balanced_precision_identity_and_paper_accuracy(self) -> None:
        # Balanced classes: DR=3/4, FA=1/4, therefore PR=DR/(DR+FA)=3/4.
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([0.9, 0.2, 0.1, 0.0, 0.9, 0.8, 0.7, 0.1])
        result = evaluate_binary_scores(labels, scores, threshold=0.5)
        self.assertEqual((result.tp, result.fp, result.tn, result.fn), (3, 1, 3, 1))
        self.assertAlmostEqual(result.precision, result.dr / (result.dr + result.fa))
        self.assertAlmostEqual(result.balanced_accuracy, (result.dr + result.sp) / 2)
        self.assertAlmostEqual(result.f1, 0.75)

    def test_low_score_orientation_applies_to_auc_and_predictions(self) -> None:
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        result = evaluate_binary_scores(
            labels, scores, threshold=0.5, positive_if="lower"
        )
        self.assertEqual((result.tp, result.fp, result.tn, result.fn), (2, 0, 2, 0))
        self.assertEqual(result.auc, 1.0)

    def test_table_v_false_alarm_is_invariant_for_fixed_benign_scores(self) -> None:
        benign = np.array([0.1, 0.2, 0.8, 0.9])
        attacks = {
            1: np.array([0.7, 0.8]),
            2: np.array([0.1, 0.2, 0.3]),
            3: np.array([0.95]),
            4: np.array([0.6, 0.65, 0.7, 0.75]),
            5: np.array([0.49, 0.51]),
            6: np.array([1.0, 0.0]),
        }
        columns = evaluate_attack_columns(benign, attacks, threshold=0.5)
        self.assertEqual({metric.fp for metric in columns.values()}, {2})
        self.assertEqual({metric.tn for metric in columns.values()}, {2})
        self.assertEqual({metric.fa for metric in columns.values()}, {0.5})
        self.assertGreater(len({metric.dr for metric in columns.values()}), 1)

    def test_single_class_auc_is_nan_but_counts_survive(self) -> None:
        result = evaluate_binary_scores([0, 0], [0.1, 0.9], threshold=0.5)
        self.assertTrue(math.isnan(result.auc))
        self.assertEqual((result.fp, result.tn), (1, 1))

    def test_seed_summary_retains_values_and_counts(self) -> None:
        first = evaluate_binary_scores([0, 1], [0.1, 0.9], threshold=0.5)
        second = evaluate_binary_scores([0, 1], [0.9, 0.1], threshold=0.5)
        summary = aggregate_seed_metrics([first, second])
        self.assertEqual(summary["n_seeds"], 2)
        self.assertEqual(summary["metrics"]["dr"]["values"], [1.0, 0.0])
        self.assertEqual(summary["confusion_counts"]["total"], {"tp": 1, "fp": 1, "tn": 1, "fn": 1})

    def test_all_roc_iqr_threshold_interpretations_are_deterministic(self) -> None:
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([0.05, 0.20, 0.45, 0.60, 0.35, 0.70, 0.80, 0.95])
        rules = (
            "roc_central_threshold_median",
            "threshold_iqr_midpoint",
            "threshold_iqr_median",
            "validation_youden_j",
        )
        selected = {
            rule: select_threshold(labels, scores, rule=rule) for rule in rules
        }
        repeated = {
            rule: select_threshold(labels, scores, rule=rule) for rule in rules
        }
        self.assertEqual(
            {rule: value.as_dict() for rule, value in selected.items()},
            {rule: value.as_dict() for rule, value in repeated.items()},
        )
        self.assertTrue(
            all(np.isfinite(value.threshold) for value in selected.values())
        )
        self.assertTrue(
            all(value.finite_roc_thresholds > 0 for value in selected.values())
        )

    def test_threshold_selection_preserves_low_score_orientation(self) -> None:
        labels = np.array([0, 0, 1, 1])
        low_is_attack = np.array([0.9, 0.8, 0.2, 0.1])
        selected = select_threshold(
            labels,
            low_is_attack,
            rule="threshold_iqr_median",
            positive_if="lower",
        )
        result = evaluate_binary_scores(
            labels,
            low_is_attack,
            threshold=selected.threshold,
            positive_if=selected.positive_if,
        )
        self.assertGreaterEqual(result.auc, 0.99)

    def test_youden_j_honors_both_score_orientations(self) -> None:
        labels = np.array([0, 0, 1, 1])
        higher = select_threshold(
            labels,
            np.array([0.1, 0.2, 0.8, 0.9]),
            rule="validation_youden_j",
            positive_if="higher",
        )
        lower = select_threshold(
            labels,
            np.array([0.9, 0.8, 0.2, 0.1]),
            rule="validation_youden_j",
            positive_if="lower",
        )

        self.assertEqual(higher.details["youden_j"], 1.0)
        self.assertEqual(lower.details["youden_j"], 1.0)
        self.assertEqual(higher.threshold, 0.8)
        self.assertEqual(lower.threshold, 0.2)

    def test_supplied_threshold_rules_do_not_require_two_classes(self) -> None:
        selected = select_threshold(
            [0, 0],
            [0.1, 0.2],
            rule="printed_constant",
            supplied_threshold=0.58,
        )
        self.assertEqual(selected.threshold, 0.58)
        self.assertEqual(selected.finite_roc_thresholds, 0)


if __name__ == "__main__":
    unittest.main()
