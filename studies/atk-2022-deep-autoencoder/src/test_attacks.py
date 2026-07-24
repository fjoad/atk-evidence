from __future__ import annotations

import unittest

import numpy as np

from attacks import generate_all_attacks, generate_attack


class AttackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = np.arange(1.0, 49.0)
        self.attacks = generate_all_attacks(self.profile, seed=20260720)

    def test_all_attacks_preserve_shape(self) -> None:
        self.assertEqual(set(self.attacks), set(range(1, 7)))
        for attacked in self.attacks.values():
            self.assertEqual(attacked.shape, self.profile.shape)

    def test_scaling_attacks_reduce_values(self) -> None:
        for attack in (1, 2):
            self.assertTrue(np.all(self.attacks[attack] >= 0.1 * self.profile))
            self.assertTrue(np.all(self.attacks[attack] <= 0.8 * self.profile))

    def test_selective_bypass_is_between_four_and_twenty_four_hours(self) -> None:
        zero_count = int(np.count_nonzero(self.attacks[3] == 0.0))
        self.assertGreaterEqual(zero_count, 8)
        self.assertLessEqual(zero_count, 48)

    def test_mean_and_reverse_attacks(self) -> None:
        self.assertTrue(np.all(self.attacks[4] == np.mean(self.profile)))
        np.testing.assert_array_equal(self.attacks[6], self.profile[::-1])

    def test_attack_one_accepts_frozen_scope_factor(self) -> None:
        first = generate_attack(
            self.profile,
            1,
            np.random.default_rng(1),
            attack1_factor=0.25,
        )
        second = generate_attack(
            self.profile * 2,
            1,
            np.random.default_rng(999),
            attack1_factor=0.25,
        )
        np.testing.assert_allclose(first, self.profile * 0.25)
        np.testing.assert_allclose(second, self.profile * 0.50)

    def test_attack_two_hour_pair_reuses_each_draw(self) -> None:
        attacked = generate_attack(
            self.profile,
            2,
            np.random.default_rng(7),
            attack2_granularity="per_hour_pair",
        )
        factors = attacked / self.profile
        np.testing.assert_allclose(factors[0::2], factors[1::2])
        self.assertGreater(np.unique(factors).size, 1)

    def test_attack_three_repairs_and_hour_mappings_are_bounded(self) -> None:
        for interval in (
            "valid_fit_addition",
            "printed_start_truncate",
            "printed_start_wrap",
        ):
            for mapping, lower, upper in (
                ("two_slots_per_hour", 8, 48),
                ("direct_48_index", 4, 24),
            ):
                with self.subTest(interval=interval, mapping=mapping):
                    attacked = generate_attack(
                        self.profile,
                        3,
                        np.random.default_rng(13),
                        attack3_interval=interval,
                        attack_hour_mapping=mapping,
                    )
                    zero_count = int(np.count_nonzero(attacked == 0.0))
                    self.assertGreaterEqual(zero_count, lower)
                    self.assertLessEqual(zero_count, upper)


if __name__ == "__main__":
    unittest.main()
