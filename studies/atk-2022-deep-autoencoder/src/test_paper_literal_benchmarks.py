from __future__ import annotations

import unittest

import numpy as np

from paper_literal_benchmarks import (
    ARIMA110ResidualModel,
    arima_completion_benchmark,
    deterministic_cap_indices,
    gaussian_nb_benchmark,
    multiclass_svm_benchmark,
    one_class_svm_benchmark,
)


class PaperLiteralBenchmarkTests(unittest.TestCase):
    def test_deterministic_stratified_cap_keeps_classes(self) -> None:
        labels = np.array([0] * 90 + [1] * 9 + [2])
        first = deterministic_cap_indices(100, 20, seed=7, labels=labels)
        second = deterministic_cap_indices(100, 20, seed=7, labels=labels)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.size, 20)
        self.assertEqual(set(labels[first]), {0, 1, 2})

    def test_gaussian_nb_returns_positive_probability_and_fixed_predictions(self) -> None:
        x_train = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y_train = np.array([0, 0, 1, 1])
        result = gaussian_nb_benchmark(x_train, y_train, np.array([[-1.5], [1.5]]))
        self.assertEqual(result.metadata["var_smoothing"], 1e-9)
        self.assertLess(result.scores[0], 0.5)
        self.assertGreater(result.scores[1], 0.5)
        np.testing.assert_array_equal(result.predictions, [0, 1])

    def test_one_class_svm_is_capped_and_high_score_means_anomaly(self) -> None:
        rng = np.random.default_rng(4)
        train = rng.normal(0.0, 0.1, size=(80, 3))
        test = np.vstack([np.zeros((2, 3)), np.full((2, 3), 8.0)])
        result = one_class_svm_benchmark(
            train, test, max_samples=20, seed=3, threshold=0.0
        )
        self.assertEqual(result.train_samples_used, 20)
        # The sigmoid kernel is not a distance metric, so far-away points need
        # not receive geometrically larger scores.  The robust invariant is
        # explicit orientation: negated decision_function >= 0 is anomalous.
        np.testing.assert_array_equal(result.predictions, result.scores >= 0.0)
        self.assertEqual(result.metadata["nu"], 0.5)

    def test_multiclass_svm_score_separates_benign_from_attack_classes(self) -> None:
        rng = np.random.default_rng(10)
        benign = rng.normal((-3.0, -3.0), 0.2, size=(20, 2))
        attack_1 = rng.normal((0.0, 3.0), 0.2, size=(20, 2))
        attack_2 = rng.normal((3.0, 0.0), 0.2, size=(20, 2))
        train = np.vstack([benign, attack_1, attack_2])
        labels = np.repeat([0, 1, 2], 20)
        test = np.vstack([benign[:3], attack_1[:3], attack_2[:3]])
        result = multiclass_svm_benchmark(
            train, labels, test, max_samples=45, seed=8
        )
        self.assertEqual(result.train_samples_used, 45)
        self.assertEqual(result.metadata["kernel"], "sigmoid")
        self.assertEqual(result.predictions.shape, (9,))
        self.assertGreater(float(np.mean(result.scores[3:])), float(np.mean(result.scores[:3])))

    def test_vectorized_arima_scores_unmodeled_jump_more_highly(self) -> None:
        # First differences follow d_t = 0.5*d_(t-1) + 1 in training.
        rows = []
        for initial in (0.0, 2.0, 4.0, 6.0):
            differences = [initial]
            for _ in range(7):
                differences.append(1.0 + 0.5 * differences[-1])
            rows.append(np.concatenate([[0.0], np.cumsum(differences)]))
        train = np.asarray(rows)
        model = ARIMA110ResidualModel.fit(train)
        clean = train[:1]
        attacked = clean.copy()
        attacked[0, 5] += 20.0
        scores = model.score_samples(np.vstack([clean, attacked]))
        self.assertAlmostEqual(model.phi, 0.5, places=10)
        self.assertAlmostEqual(model.intercept, 1.0, places=10)
        self.assertGreater(scores[1], scores[0] + 1.0)

    def test_all_frozen_arima_completions_execute_and_identify_their_score(
        self,
    ) -> None:
        rng = np.random.default_rng(19)
        train = np.cumsum(rng.normal(size=(12, 12)), axis=1)
        test = np.cumsum(rng.normal(size=(4, 12)), axis=1)
        for order in (0, 1, 2, 5):
            for fit_unit in ("pooled", "profile"):
                for score in ("mse", "likelihood"):
                    completion = f"p{order}_{fit_unit}_{score}"
                    with self.subTest(completion=completion):
                        result = arima_completion_benchmark(
                            train,
                            test,
                            completion=completion,
                        )
                        self.assertEqual(
                            result.metadata["completion"],
                            completion,
                        )
                        self.assertEqual(result.scores.shape, (4,))
                        self.assertTrue(np.isfinite(result.scores).all())

    def test_uncapped_svm_branches_retain_every_training_row(self) -> None:
        rng = np.random.default_rng(23)
        one_class = one_class_svm_benchmark(
            rng.normal(size=(30, 4)),
            rng.normal(size=(3, 4)),
            max_samples=None,
        )
        self.assertEqual(one_class.train_samples_used, 30)
        labels = np.repeat([0, 1, 2], 12)
        multiclass = multiclass_svm_benchmark(
            rng.normal(size=(36, 4)),
            labels,
            rng.normal(size=(3, 4)),
            max_samples=None,
        )
        self.assertEqual(multiclass.train_samples_used, 36)


if __name__ == "__main__":
    unittest.main()
