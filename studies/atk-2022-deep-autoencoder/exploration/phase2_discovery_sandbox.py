#!/usr/bin/env python3
"""Disposable Phase-2 synthetic sandbox.

Classification: exploratory X only. This file intentionally imports no ATK
Evidence implementation or result artifact. See ../DISCOVERY_SANDBOX.md.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 20260824
WIDTH = 48
N_TRAIN = 512
N_TEST = 256


def _auc(benign: np.ndarray, anomalous: np.ndarray) -> float:
    """Exact pairwise AUC for a score whose high direction means anomaly."""
    greater = anomalous[:, None] > benign[None, :]
    equal = anomalous[:, None] == benign[None, :]
    return float(greater.mean() + 0.5 * equal.mean())


def _oracle_balanced_accuracy(
    benign: np.ndarray, anomalous: np.ndarray
) -> dict[str, float | None]:
    values = np.unique(np.concatenate([benign, anomalous]))
    if values.size == 1:
        thresholds = np.array([-np.inf, np.inf])
    else:
        thresholds = np.concatenate(
            [
                np.array([-np.inf]),
                (values[:-1] + values[1:]) / 2.0,
                np.array([np.inf]),
            ]
        )
    best = (-1.0, math.nan)
    for threshold in thresholds:
        tpr = float(np.mean(anomalous >= threshold))
        tnr = float(np.mean(benign < threshold))
        balanced = 0.5 * (tpr + tnr)
        if balanced > best[0]:
            best = (balanced, float(threshold))
    threshold = best[1] if math.isfinite(best[1]) else None
    return {"balanced_accuracy": best[0], "threshold": threshold}


def _score_summary(benign: np.ndarray, anomalous: np.ndarray) -> dict:
    return {
        "auc": _auc(benign, anomalous),
        "oracle": _oracle_balanced_accuracy(benign, anomalous),
        "benign_mean": float(np.mean(benign)),
        "anomalous_mean": float(np.mean(anomalous)),
        "benign_std": float(np.std(benign)),
        "anomalous_std": float(np.std(anomalous)),
    }


def _base_profiles(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0.0, 1.0, WIDTH, endpoint=False)
    template = (
        1.3
        + 0.42 * np.sin(2.0 * np.pi * (t - 0.17))
        + 0.18 * np.sin(4.0 * np.pi * (t + 0.09))
        + 0.48 * np.exp(-0.5 * ((t - 0.76) / 0.08) ** 2)
    )
    rows = np.empty((n, WIDTH), dtype=np.float64)
    for index in range(n):
        shift = int(rng.integers(-3, 4))
        amplitude = float(rng.uniform(0.78, 1.22))
        offset = float(rng.uniform(0.15, 0.55))
        noise = rng.normal(0.0, 0.025, size=WIDTH)
        rows[index] = np.clip(offset + amplitude * np.roll(template, shift) + noise, 0.01, None)
    return rows


def _toy_attacks(
    benign: np.ndarray, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    n = benign.shape[0]
    attack_1 = benign * rng.uniform(0.1, 0.8, size=(n, 1))
    attack_2 = benign * rng.uniform(0.1, 0.8, size=benign.shape)

    # This implements the described forward 4--24 hour bypass solely as a toy
    # witness. It is not a repair admitted to the paper-literal track.
    attack_3 = benign.copy()
    for row in attack_3:
        length = int(rng.integers(8, WIDTH + 1))
        start = int(rng.integers(0, WIDTH - length + 1))
        row[start : start + length] = 0.0

    daily_mean = benign.mean(axis=1, keepdims=True)
    attack_4 = np.repeat(daily_mean, WIDTH, axis=1)
    attack_5 = daily_mean * rng.uniform(0.1, 0.8, size=benign.shape)
    attack_6 = benign[:, ::-1].copy()
    return {
        "attack_1_fixed_reduction": attack_1,
        "attack_2_dynamic_reduction": attack_2,
        "attack_3_forward_bypass_toy": attack_3,
        "attack_4_daily_mean": attack_4,
        "attack_5_randomized_mean": attack_5,
        "attack_6_reversal": attack_6,
    }


FEATURES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "energy": lambda x: x.sum(axis=1),
    "mean": lambda x: x.mean(axis=1),
    "variance": lambda x: x.var(axis=1),
    "range": lambda x: np.ptp(x, axis=1),
    "zero_count": lambda x: np.sum(x <= 1e-12, axis=1).astype(np.float64),
    "roughness": lambda x: np.mean(np.diff(x, axis=1) ** 2, axis=1),
    "linear_trend": lambda x: (
        (x - x.mean(axis=1, keepdims=True))
        @ (np.arange(WIDTH, dtype=np.float64) - (WIDTH - 1) / 2.0)
        / np.sum((np.arange(WIDTH, dtype=np.float64) - (WIDTH - 1) / 2.0) ** 2)
    ),
}


def _robust_deviation(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    center = float(np.median(reference))
    mad = float(np.median(np.abs(reference - center)))
    scale = max(mad, 1e-12)
    return np.abs(values - center) / scale


def _triviality_floor(
    train: np.ndarray, benign: np.ndarray, attacks: dict[str, np.ndarray]
) -> dict:
    output: dict[str, dict] = {}
    for feature_name, feature in FEATURES.items():
        reference = feature(train)
        benign_score = _robust_deviation(reference, feature(benign))
        output[feature_name] = {
            attack_name: _score_summary(
                benign_score, _robust_deviation(reference, feature(attacked))
            )
            for attack_name, attacked in attacks.items()
        }

    invariant_names = ["energy", "mean", "variance", "range", "zero_count"]
    output["reversal_pairwise_invariance"] = {
        name: float(np.max(np.abs(FEATURES[name](benign) - FEATURES[name](attacks["attack_6_reversal"]))))
        for name in invariant_names
    }
    return output


def _project_simplex(row: np.ndarray) -> np.ndarray:
    ordered = np.sort(row)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, row.size + 1)
    valid = ordered - cumulative / indices > 0.0
    rho = int(np.nonzero(valid)[0][-1])
    theta = cumulative[rho] / float(rho + 1)
    return np.maximum(row - theta, 0.0)


def _domain_summary(rows: np.ndarray) -> dict:
    simplex_projection = np.vstack([_project_simplex(row) for row in rows])
    box_projection = np.clip(rows, 0.0, 1.0)
    simplex_bound = np.mean((rows - simplex_projection) ** 2, axis=1)
    box_bound = np.mean((rows - box_projection) ** 2, axis=1)
    return {
        "negative_coordinate_fraction": float(np.mean(rows < 0.0)),
        "unit_box_violation_fraction": float(np.mean((rows < 0.0) | (rows > 1.0))),
        "row_sum_mean": float(np.mean(rows.sum(axis=1))),
        "softmax_simplex_mse_lower_bound_mean": float(np.mean(simplex_bound)),
        "softmax_simplex_mse_lower_bound_min": float(np.min(simplex_bound)),
        "sigmoid_unit_box_mse_lower_bound_mean": float(np.mean(box_bound)),
        "sigmoid_unit_box_mse_lower_bound_min": float(np.min(box_bound)),
    }


def _output_domain_probe(
    train: np.ndarray, benign: np.ndarray, attacks: dict[str, np.ndarray]
) -> dict:
    complete = np.vstack([train, benign, *attacks.values()])
    mean = complete.mean(axis=0)
    std = complete.std(axis=0)
    std[std == 0.0] = 1.0
    groups = {"benign": benign, **attacks}
    output = {
        name: _domain_summary((rows - mean) / std) for name, rows in groups.items()
    }
    output["standardizer"] = {
        "fit_population_rows": int(complete.shape[0]),
        "feature_mean_min": float(mean.min()),
        "feature_mean_max": float(mean.max()),
        "feature_std_min": float(std.min()),
        "feature_std_max": float(std.max()),
    }
    return output


def _temporal_profiles(n: int, rng: np.random.Generator) -> np.ndarray:
    base = np.linspace(0.2, 1.8, WIDTH, endpoint=True)
    base += 0.12 * np.sin(np.linspace(0.0, 4.0 * np.pi, WIDTH, endpoint=False))
    rows = np.empty((n, WIDTH), dtype=np.float64)
    for index in range(n):
        shift = int(rng.integers(0, WIDTH))
        amplitude = float(rng.uniform(0.85, 1.15))
        offset = float(rng.uniform(0.1, 0.4))
        rows[index] = offset + amplitude * np.roll(base, shift) + rng.normal(0.0, 0.01, WIDTH)
    return rows


def _temporal_anomalies(
    benign: np.ndarray, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    amplitude = 0.5 * benign
    block_disruption = np.empty_like(benign)
    block_size = 8
    block_count = WIDTH // block_size
    for index, row in enumerate(benign):
        blocks = row.reshape(block_count, block_size)
        order = rng.permutation(block_count)
        block_disruption[index] = blocks[order].reshape(WIDTH)
    reversal = benign[:, ::-1].copy()
    return {
        "amplitude_reduction_order_irrelevant": amplitude,
        "block_disruption_order_useful": block_disruption,
        "reversal_order_necessary": reversal,
    }


class DenseAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(WIDTH, 24),
            nn.Tanh(),
            nn.Linear(24, 8),
            nn.Tanh(),
            nn.Linear(8, 24),
            nn.Tanh(),
            nn.Linear(24, WIDTH),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class SequenceAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hidden = 14
        latent = 8
        self.encoder = nn.LSTM(1, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.to_hidden = nn.Linear(latent, hidden)
        self.to_cell = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(latent, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(values.unsqueeze(-1))
        latent = torch.tanh(self.to_latent(hidden[-1]))
        initial_hidden = torch.tanh(self.to_hidden(latent)).unsqueeze(0)
        initial_cell = torch.tanh(self.to_cell(latent)).unsqueeze(0)
        repeated = latent.unsqueeze(1).repeat(1, values.shape[1], 1)
        decoded, _ = self.decoder(repeated, (initial_hidden, initial_cell))
        return self.output(decoded).squeeze(-1)


def _parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _safe_pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


def _train_model(
    model: nn.Module,
    train_rows: np.ndarray,
    device: torch.device,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    model.to(device)
    data = torch.tensor(train_rows, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(data), batch_size=64, shuffle=True, generator=generator
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.MSELoss()
    losses: list[float] = []
    first_gradient_finite = False
    for epoch in range(25):
        model.train()
        total = 0.0
        count = 0
        for batch_index, (batch,) in enumerate(loader):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstructed = model(batch)
            loss = loss_function(reconstructed, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at epoch {epoch}")
            loss.backward()
            if epoch == 0 and batch_index == 0:
                first_gradient_finite = all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                    for parameter in model.parameters()
                )
            optimizer.step()
            total += float(loss.item()) * batch.shape[0]
            count += int(batch.shape[0])
        losses.append(total / count)
    return {
        "epoch_losses": losses,
        "first_gradient_finite": bool(first_gradient_finite),
        "loss_decreased": bool(losses[-1] < losses[0]),
        "parameter_count": _parameter_count(model),
    }


@torch.no_grad()
def _reconstruction_scores(
    model: nn.Module, rows: np.ndarray, device: torch.device
) -> np.ndarray:
    model.eval()
    tensor = torch.tensor(rows, dtype=torch.float32)
    scores: list[np.ndarray] = []
    for start in range(0, tensor.shape[0], 256):
        batch = tensor[start : start + 256].to(device)
        reconstructed = model(batch)
        mse = torch.mean((batch - reconstructed) ** 2, dim=1)
        scores.append(mse.cpu().numpy())
    return np.concatenate(scores).astype(np.float64)


def _temporal_witness(rng: np.random.Generator) -> dict:
    train_raw = _temporal_profiles(N_TRAIN, rng)
    benign_raw = _temporal_profiles(N_TEST, rng)
    anomalies_raw = _temporal_anomalies(benign_raw, rng)
    mean = train_raw.mean(axis=0)
    std = train_raw.std(axis=0)
    std[std == 0.0] = 1.0
    train = (train_raw - mean) / std
    benign = (benign_raw - mean) / std
    anomalies = {name: (rows - mean) / std for name, rows in anomalies_raw.items()}

    assert np.allclose(
        np.sort(benign_raw, axis=1),
        np.sort(anomalies_raw["reversal_order_necessary"], axis=1),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {
        "dense_linear_ae": DenseAutoencoder(),
        "seq2seq_lstm_linear_ae": SequenceAutoencoder(),
    }
    result: dict[str, dict] = {}
    saved_scores: dict[str, dict[str, np.ndarray]] = {}
    for offset, (name, model) in enumerate(models.items()):
        training = _train_model(model, train, device, SEED + offset)
        benign_score = _reconstruction_scores(model, benign, device)
        anomaly_scores = {
            anomaly_name: _reconstruction_scores(model, rows, device)
            for anomaly_name, rows in anomalies.items()
        }
        saved_scores[name] = {"benign": benign_score, **anomaly_scores}
        result[name] = {
            "training": training,
            "anomalies": {
                anomaly_name: _score_summary(benign_score, scores)
                for anomaly_name, scores in anomaly_scores.items()
            },
        }

    correlations: dict[str, float | None] = {}
    for anomaly_name in anomalies:
        dense = np.concatenate(
            [saved_scores["dense_linear_ae"]["benign"], saved_scores["dense_linear_ae"][anomaly_name]]
        )
        recurrent = np.concatenate(
            [saved_scores["seq2seq_lstm_linear_ae"]["benign"], saved_scores["seq2seq_lstm_linear_ae"][anomaly_name]]
        )
        correlations[anomaly_name] = _safe_pearson(dense, recurrent)

    return {
        "device": str(device),
        "models": result,
        "dense_lstm_score_pearson": correlations,
        "reversal_multiset_max_abs_difference": float(
            np.max(
                np.abs(
                    np.sort(benign_raw, axis=1)
                    - np.sort(anomalies_raw["reversal_order_necessary"], axis=1)
                )
            )
        ),
    }


def _vae_direction_probe() -> dict:
    reconstruction_errors = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    probability = np.exp(-0.5 * reconstruction_errors**2)
    monotone = bool(np.all(np.diff(probability) < 0.0))
    assert monotone
    return {
        "reconstruction_error": reconstruction_errors.tolist(),
        "fixed_unit_gaussian_relative_probability": probability.tolist(),
        "strictly_decreases_with_error": monotone,
        "anomaly_consistent_direction": "low_probability",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    rng = np.random.default_rng(SEED)
    train = _base_profiles(N_TRAIN, rng)
    benign = _base_profiles(N_TEST, rng)
    attacks = _toy_attacks(benign, rng)

    result = {
        "classification": "X exploratory only",
        "seed": SEED,
        "config": {
            "width": WIDTH,
            "train_rows": N_TRAIN,
            "test_rows": N_TEST,
            "epochs": 25,
            "batch_size": 64,
            "optimizer": "Adam(lr=1e-3)",
            "neural_output": "linear",
            "named_data_used": False,
            "historical_project_imports_used": False,
        },
        "x1_triviality_floor": _triviality_floor(train, benign, attacks),
        "x2_temporal_witness": _temporal_witness(rng),
        "x3_output_domain": _output_domain_probe(train, benign, attacks),
        "x4_vae_score_direction": _vae_direction_probe(),
    }
    result["runtime_seconds"] = float(time.perf_counter() - started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "output": str(args.output),
        "runtime_seconds": result["runtime_seconds"],
        "device": result["x2_temporal_witness"]["device"],
    }, indent=2))


if __name__ == "__main__":
    main()
