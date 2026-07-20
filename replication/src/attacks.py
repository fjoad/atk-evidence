"""Synthetic Irish CER attack functions described in Takiddin et al. (2022).

The paper indexes some functions as if a day had 24 hourly values, while its
dataset and model input have 48 half-hour values. The controlled implementation
therefore makes the number of samples per hour explicit.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def generate_attack(
    profile: ArrayLike,
    attack: int,
    rng: np.random.Generator,
    *,
    samples_per_hour: int = 2,
) -> NDArray[np.float64]:
    """Return one attacked copy of a one-day consumption profile.

    Args:
        profile: One-dimensional consumption sequence.
        attack: Integer in 1..6 corresponding to equations (1)..(6).
        rng: Explicit NumPy random generator.
        samples_per_hour: Two for the official half-hour CER readings.
    """

    values = np.asarray(profile, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("profile must be one-dimensional")
    if values.size == 0:
        raise ValueError("profile must not be empty")
    if samples_per_hour < 1:
        raise ValueError("samples_per_hour must be positive")
    if attack not in range(1, 7):
        raise ValueError("attack must be an integer from 1 through 6")

    attacked = values.copy()
    if attack == 1:
        attacked *= rng.uniform(0.1, 0.8)
    elif attack == 2:
        attacked *= rng.uniform(0.1, 0.8, size=values.shape)
    elif attack == 3:
        # The printed paper states tf = ti - tl, which would put the final
        # index before the initial index. The controlled interpretation uses
        # addition and enforces the stated 4-24 hour duration.
        available_hours = max(1, values.size // samples_per_hour)
        duration_hours = int(rng.integers(4, min(24, available_hours) + 1))
        latest_start = max(0, available_hours - duration_hours)
        start_hour = int(rng.integers(0, latest_start + 1))
        start = start_hour * samples_per_hour
        stop = min(values.size, start + duration_hours * samples_per_hour)
        attacked[start:stop] = 0.0
    elif attack == 4:
        attacked.fill(float(np.mean(values)))
    elif attack == 5:
        attacked = rng.uniform(0.1, 0.8, size=values.shape) * float(np.mean(values))
    elif attack == 6:
        attacked = values[::-1].copy()
    return attacked


def generate_all_attacks(
    profile: ArrayLike,
    seed: int,
    *,
    samples_per_hour: int = 2,
) -> dict[int, NDArray[np.float64]]:
    """Generate all six attacks with independent deterministic streams."""

    seed_sequence = np.random.SeedSequence(seed)
    child_sequences = seed_sequence.spawn(6)
    return {
        attack: generate_attack(
            profile,
            attack,
            np.random.default_rng(child_sequences[attack - 1]),
            samples_per_hour=samples_per_hour,
        )
        for attack in range(1, 7)
    }

