from __future__ import annotations

import unittest

import numpy as np

from attacks import generate_all_attacks


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


if __name__ == "__main__":
    unittest.main()

