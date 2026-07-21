"""Matched-model pilot for the paper's six attacks on SGCC-derived windows.

This is a mechanism check, not a reproduction of the Irish CER experiment.
Each 48-value input is a chronologically ordered 48-day window from a distinct
benign SGCC customer. The six paper attacks are then applied to held-out raw
windows before train-only standardization. Customer splits are disjoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from attacks import generate_attack


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_windows(
    values: np.ndarray,
    customer_indices: np.ndarray,
    count: int,
    rng: np.random.Generator,
    *,
    width: int = 48,
    minimum_observed: float = 0.70,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose at most one sufficiently observed random window per customer."""

    windows: list[np.ndarray] = []
    selected_customers: list[int] = []
    starts: list[int] = []
    candidates = rng.permutation(customer_indices)
    for customer_index in candidates:
        valid_start = values.shape[1] - width
        for start in rng.integers(0, valid_start + 1, size=20):
            window = values[customer_index, start : start + width].astype(np.float64)
            if np.mean(np.isfinite(window)) >= minimum_observed:
                windows.append(window)
                selected_customers.append(int(customer_index))
                starts.append(int(start))
                break
        if len(windows) == count:
            break
    if len(windows) < count:
        raise RuntimeError(f"only found {len(windows)} of {count} requested windows")
    return (
        np.stack(windows),
        np.asarray(selected_customers, dtype=np.int64),
        np.asarray(starts, dtype=np.int64),
    )


def impute_within_window(raw: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    """Linearly interpolate within rows, then use a supplied train-only fallback."""

    frame = pd.DataFrame(raw)
    imputed = frame.interpolate(axis=1, limit_direction="both").to_numpy(dtype=np.float64)
    if fallback is None:
        fallback = np.nanmedian(imputed, axis=0)
        global_fallback = float(np.nanmedian(imputed))
        fallback = np.where(np.isfinite(fallback), fallback, global_fallback)
    missing = ~np.isfinite(imputed)
    if missing.any():
        imputed[missing] = np.broadcast_to(fallback, imputed.shape)[missing]
    return imputed


def make_attacks(raw: np.ndarray, seed: int) -> dict[int, np.ndarray]:
    sequences = np.random.SeedSequence(seed).spawn(raw.shape[0] * 6)
    attacked: dict[int, list[np.ndarray]] = {attack: [] for attack in range(1, 7)}
    stream = 0
    for row in raw:
        for attack in range(1, 7):
            attacked[attack].append(
                generate_attack(
                    row,
                    attack,
                    np.random.default_rng(sequences[stream]),
                    samples_per_hour=2,
                )
            )
            stream += 1
    return {key: np.stack(items) for key, items in attacked.items()}


class FullyConnectedAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(48, 64),
            nn.Tanh(),
            nn.Linear(64, 24),
            nn.Tanh(),
            nn.Linear(24, 64),
            nn.Tanh(),
            nn.Linear(64, 48),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class LSTMAE(nn.Module):
    def __init__(self, *, attention: bool, hidden_size: int = 27) -> None:
        super().__init__()
        self.attention = attention
        self.encoder = nn.LSTM(1, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size * (2 if attention else 1), 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        sequence = values.unsqueeze(-1)
        encoded, state = self.encoder(sequence)
        repeated_context = state[0][-1].unsqueeze(1).expand(-1, values.shape[1], -1)
        decoded, _ = self.decoder(repeated_context, state)
        if self.attention:
            weights = torch.softmax(
                torch.bmm(decoded, encoded.transpose(1, 2)) / (encoded.shape[-1] ** 0.5),
                dim=-1,
            )
            context = torch.bmm(weights, encoded)
            decoded = torch.cat([decoded, context], dim=-1)
        return self.output(decoded).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


@dataclass
class FitResult:
    best_epoch: int
    train_loss: list[float]
    validation_loss: list[float]
    duration_seconds: float


def fit_model(
    model: nn.Module,
    train_values: np.ndarray,
    validation_values: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    patience: int = 5,
) -> FitResult:
    set_seed(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_values.astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_tensor = torch.from_numpy(validation_values.astype(np.float32))
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = -1
    stale = 0
    train_curve: list[float] = []
    validation_curve: list[float] = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch), batch)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * batch.shape[0]
            seen += batch.shape[0]
        train_curve.append(running_loss / seen)

        model.eval()
        validation_sum = 0.0
        validation_seen = 0
        with torch.no_grad():
            for batch in validation_tensor.split(batch_size):
                batch = batch.to(device)
                value = loss_function(model(batch), batch)
                validation_sum += float(value.cpu()) * batch.shape[0]
                validation_seen += batch.shape[0]
        validation_loss = validation_sum / validation_seen
        validation_curve.append(validation_loss)
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a model state")
    model.load_state_dict(best_state)
    model.to(device)
    return FitResult(best_epoch, train_curve, validation_curve, time.perf_counter() - started)


def reconstruction_scores(
    model: nn.Module,
    values: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(values.astype(np.float32))
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tensor.split(batch_size):
            batch = batch.to(device)
            reconstruction = model(batch)
            scores.append(torch.mean((reconstruction - batch) ** 2, dim=1).cpu().numpy())
    return np.concatenate(scores).astype(np.float64)


def evaluate_scores(
    validation_benign: np.ndarray,
    test_benign: np.ndarray,
    test_attacks: dict[int, np.ndarray],
    *,
    target_fa: float,
) -> tuple[dict[str, object], np.ndarray]:
    threshold = float(np.quantile(validation_benign, 1.0 - target_fa))
    false_alarm = float(np.mean(test_benign > threshold))
    attack_metrics: dict[str, dict[str, float]] = {}
    pooled = []
    for attack, scores in test_attacks.items():
        labels = np.concatenate([np.zeros(test_benign.size), np.ones(scores.size)])
        combined = np.concatenate([test_benign, scores])
        attack_metrics[str(attack)] = {
            "roc_auc": float(roc_auc_score(labels, combined)),
            "detection_rate": float(np.mean(scores > threshold)),
        }
        pooled.append(scores)
    pooled_scores = np.concatenate(pooled)
    pooled_labels = np.concatenate(
        [np.zeros(test_benign.size), np.ones(pooled_scores.size)]
    )
    all_scores = np.concatenate([test_benign, pooled_scores])
    return (
        {
            "threshold": threshold,
            "false_alarm_rate": false_alarm,
            "pooled_roc_auc": float(roc_auc_score(pooled_labels, all_scores)),
            "pooled_detection_rate": float(np.mean(pooled_scores > threshold)),
            "by_attack": attack_metrics,
        },
        all_scores,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/sgcc-verified/data.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("studies/atk-2022-deep-autoencoder/results/sgcc_attack_pilot.json"),
    )
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=Path("studies/atk-2022-deep-autoencoder/results/sgcc_attack_pilot_scores.npz"),
    )
    parser.add_argument("--data-seed", type=int, default=20260720)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--train-windows", type=int, default=12000)
    parser.add_argument("--validation-windows", type=int, default=3000)
    parser.add_argument("--test-windows", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--target-fa", type=float, default=0.05)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = parser.parse_args()
    experiment_start = time.perf_counter()

    frame = pd.read_csv(args.input, low_memory=False)
    date_columns = list(frame.columns[2:])
    parsed_dates = pd.to_datetime(date_columns, format="%Y/%m/%d")
    ordered_columns = [date_columns[index] for index in np.argsort(parsed_dates.to_numpy())]
    all_values = frame[ordered_columns].to_numpy(dtype=np.float32)
    benign_indices = np.flatnonzero(frame["FLAG"].to_numpy(dtype=np.int8) == 0)
    train_customers, remainder = train_test_split(
        benign_indices, test_size=0.4, random_state=args.data_seed
    )
    validation_customers, test_customers = train_test_split(
        remainder, test_size=0.5, random_state=args.data_seed + 1
    )
    streams = np.random.SeedSequence(args.data_seed).spawn(3)
    train_raw, train_used, train_starts = choose_windows(
        all_values, train_customers, args.train_windows, np.random.default_rng(streams[0])
    )
    validation_raw, validation_used, validation_starts = choose_windows(
        all_values, validation_customers, args.validation_windows, np.random.default_rng(streams[1])
    )
    test_raw, test_used, test_starts = choose_windows(
        all_values, test_customers, args.test_windows, np.random.default_rng(streams[2])
    )

    train_imputed = impute_within_window(train_raw)
    fallback = np.nanmedian(train_imputed, axis=0)
    validation_imputed = impute_within_window(validation_raw, fallback)
    test_imputed = impute_within_window(test_raw, fallback)
    validation_attacked_raw = make_attacks(validation_imputed, args.data_seed + 101)
    test_attacked_raw = make_attacks(test_imputed, args.data_seed + 202)

    mean = train_imputed.mean(axis=0)
    scale = train_imputed.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardize = lambda array: ((array - mean) / scale).astype(np.float32)
    train = standardize(train_imputed)
    validation = standardize(validation_imputed)
    test = standardize(test_imputed)
    validation_attacks = {key: standardize(value) for key, value in validation_attacked_raw.items()}
    test_attacks = {key: standardize(value) for key, value in test_attacked_raw.items()}

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    raw_scores: dict[str, np.ndarray] = {
        "test_customer_row": test_used,
        "test_window_start": test_starts,
    }
    results: list[dict[str, object]] = []

    centroid = np.mean(train, axis=0)
    centroid_score = lambda values: np.mean((values - centroid) ** 2, axis=1)
    metrics, stored = evaluate_scores(
        centroid_score(validation),
        centroid_score(test),
        {key: centroid_score(value) for key, value in test_attacks.items()},
        target_fa=args.target_fa,
    )
    results.append({"model": "centroid_mse", "seed": None, "parameters": 0, **metrics})
    raw_scores["centroid_mse"] = stored

    pca = PCA(n_components=24, random_state=args.data_seed).fit(train)
    pca_score = lambda values: np.mean((values - pca.inverse_transform(pca.transform(values))) ** 2, axis=1)
    metrics, stored = evaluate_scores(
        pca_score(validation),
        pca_score(test),
        {key: pca_score(value) for key, value in test_attacks.items()},
        target_fa=args.target_fa,
    )
    results.append({"model": "pca_24", "seed": None, "parameters": int(pca.components_.size), **metrics})
    raw_scores["pca_24"] = stored

    model_factories = {
        "fc_ae": FullyConnectedAE,
        "lstm_ae": lambda: LSTMAE(attention=False),
        "lstm_attention_ae": lambda: LSTMAE(attention=True),
    }
    for seed in args.seeds:
        for model_name, factory in model_factories.items():
            set_seed(seed)
            model = factory()
            fit = fit_model(
                model,
                train,
                validation,
                seed=seed,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            validation_scores = reconstruction_scores(
                model, validation, device=device, batch_size=args.batch_size
            )
            test_benign_scores = reconstruction_scores(
                model, test, device=device, batch_size=args.batch_size
            )
            attack_scores = {
                key: reconstruction_scores(model, value, device=device, batch_size=args.batch_size)
                for key, value in test_attacks.items()
            }
            metrics, stored = evaluate_scores(
                validation_scores,
                test_benign_scores,
                attack_scores,
                target_fa=args.target_fa,
            )
            results.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "parameters": parameter_count(model),
                    "best_epoch_zero_based": fit.best_epoch,
                    "training_seconds": fit.duration_seconds,
                    "train_loss": fit.train_loss,
                    "validation_loss": fit.validation_loss,
                    **metrics,
                }
            )
            raw_scores[f"{model_name}_seed_{seed}"] = stored
            print(
                model_name,
                seed,
                f"AUC={metrics['pooled_roc_auc']:.4f}",
                f"DR={metrics['pooled_detection_rate']:.4f}",
                f"FA={metrics['false_alarm_rate']:.4f}",
                flush=True,
            )

    payload = {
        "experiment": "SGCC-derived 48-reading synthetic-attack mechanism pilot; not CER reproduction",
        "limitations": [
            "A 48-value input represents 48 consecutive SGCC days, not 48 CER half-hours.",
            "Only one sampled window per benign customer is used.",
            "The paper does not specify its SGCC 48-value construction, so no claim about its SGCC result follows.",
            (
                f"This run uses {len(args.seeds)} training seeds; the controlled target is at least ten."
                if len(args.seeds) < 10
                else f"This run meets the controlled target with {len(args.seeds)} training seeds."
            ),
        ],
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "data_seed": args.data_seed,
        "training_seeds": args.seeds,
        "chronologically_sorted": True,
        "customer_disjoint": True,
        "one_window_per_customer": True,
        "attack_applied_before_train_only_standardization": True,
        "threshold_uses_only_benign_validation_scores": True,
        "target_false_alarm_rate": args.target_fa,
        "windows": {
            "train": int(train.shape[0]),
            "validation": int(validation.shape[0]),
            "test": int(test.shape[0]),
        },
        "customer_split_sizes": {
            "train": int(train_customers.size),
            "validation": int(validation_customers.size),
            "test": int(test_customers.size),
        },
        "window_metadata_sha256": {
            "train_customers": hashlib.sha256(train_used.tobytes()).hexdigest(),
            "train_starts": hashlib.sha256(train_starts.tobytes()).hexdigest(),
            "validation_customers": hashlib.sha256(validation_used.tobytes()).hexdigest(),
            "validation_starts": hashlib.sha256(validation_starts.tobytes()).hexdigest(),
            "test_customers": hashlib.sha256(test_used.tobytes()).hexdigest(),
            "test_starts": hashlib.sha256(test_starts.tobytes()).hexdigest(),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "device": str(device),
        },
        "duration_seconds": time.perf_counter() - experiment_start,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(args.scores_output, **raw_scores)
    print(json.dumps({"output": str(args.output), "duration_seconds": payload["duration_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
