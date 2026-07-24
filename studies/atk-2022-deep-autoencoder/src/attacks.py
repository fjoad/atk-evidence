"""Synthetic Irish CER attack functions described in Takiddin et al. (2022).

The paper indexes some functions as if a day had 24 hourly values, while its
dataset and model input have 48 half-hour values. The implementation therefore
keeps both registered hour-to-slot readings and every minimal repair of the
non-executable Attack-3 interval explicit.
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
    attack1_factor: float | None = None,
    attack2_granularity: str = "per_half_hour",
    attack3_interval: str = "valid_fit_addition",
    attack_hour_mapping: str = "two_slots_per_hour",
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
    if attack2_granularity not in {"per_half_hour", "per_hour_pair"}:
        raise ValueError("unsupported attack2_granularity")
    if attack3_interval not in {
        "valid_fit_addition",
        "printed_start_truncate",
        "printed_start_wrap",
    }:
        raise ValueError("unsupported attack3_interval")
    if attack_hour_mapping not in {"two_slots_per_hour", "direct_48_index"}:
        raise ValueError("unsupported attack_hour_mapping")
    mapped_samples_per_hour = (
        samples_per_hour if attack_hour_mapping == "two_slots_per_hour" else 1
    )

    attacked = values.copy()
    if attack == 1:
        factor = (
            float(attack1_factor)
            if attack1_factor is not None
            else float(rng.uniform(0.1, 0.8))
        )
        if not 0.1 <= factor <= 0.8:
            raise ValueError("attack1_factor must lie in [0.1, 0.8]")
        attacked *= factor
    elif attack == 2:
        if attack2_granularity == "per_half_hour":
            factors = rng.uniform(0.1, 0.8, size=values.shape)
        else:
            units = int(np.ceil(values.size / mapped_samples_per_hour))
            factors = np.repeat(
                rng.uniform(0.1, 0.8, size=units),
                mapped_samples_per_hour,
            )[: values.size]
        attacked *= factors
    elif attack == 3:
        # The printed tf=ti-tl expression is preserved as non-executable in the
        # manifest. These are its three predeclared minimal repairs.
        available_hours = max(1, values.size // mapped_samples_per_hour)
        duration_hours = int(rng.integers(4, min(24, available_hours) + 1))
        if attack3_interval == "valid_fit_addition":
            latest_start = max(0, available_hours - duration_hours)
            start_hour = int(rng.integers(0, latest_start + 1))
        else:
            start_hour = int(rng.integers(0, min(19, available_hours - 1) + 1))
        start = start_hour * mapped_samples_per_hour
        length = duration_hours * mapped_samples_per_hour
        if attack3_interval == "printed_start_wrap":
            indices = (start + np.arange(length)) % values.size
            attacked[indices] = 0.0
        else:
            stop = min(values.size, start + length)
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
    attack2_granularity: str = "per_half_hour",
    attack3_interval: str = "valid_fit_addition",
    attack_hour_mapping: str = "two_slots_per_hour",
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
            attack2_granularity=attack2_granularity,
            attack3_interval=attack3_interval,
            attack_hour_mapping=attack_hour_mapping,
        )
        for attack in range(1, 7)
    }
