"""Small exact-ISET execution adapter for Paper 1 Tables III--V.

Table III trains and evaluates every paper model on the prepared CER/ISET
partitions.  For anomaly models, the same Table III score vector is indexed at
the preregistered 3,000 benign and attack rows to produce Table V without
retraining or changing the threshold.  Table IV retrains anomaly models on the
nested half, three-quarter, and full benign-training subsets.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from paper_literal_iset import IsetPaperLiteralData, load_prepared_iset
from paper_literal_metrics import evaluate_attack_columns, threshold_predictions
from paper_literal_runner import (
    ANOMALY_NEURAL_MODELS,
    ExecutionResult,
    RunOutcome,
    RunScope,
    _jsonable,
    _sha256_path,
    _write_preflight,
    execute_selected_model,
    load_contract,
    resolve_models,
    resolve_seeds,
    run_one,
)
from branch_runtime import (
    DEFAULT_LATTICE,
    assert_branch_scope,
    load_runtime_branch,
)


PRIMARY_SCORE = {
    "fc_sae": "reconstruction_mse",
    "lstm_sae": "reconstruction_mse",
    "fc_vae": "reconstruction_mse",
    "lstm_vae": "reconstruction_mse",
    "lstm_aea": "reconstruction_mse",
}
TABLE_IV_SIZES = ("half", "three_quarter", "full")
TABLE_V_IDENTITIES = (
    "common_model_common_benign",
    "retrain_per_attack",
    "resplit_per_attack",
    "retrain_and_resplit",
)
TABLE_V_SIZES = ("full_heldout", "seeded_3000")


def load_verified_cache(
    prefix: str | Path,
) -> tuple[IsetPaperLiteralData, Mapping[str, Any], float]:
    """Checksum-verify and load the prepared exact-ISET cache."""

    started = time.perf_counter()
    cache_prefix = Path(prefix).expanduser().resolve()
    npz_path = cache_prefix.with_suffix(".npz")
    manifest_path = cache_prefix.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prepared = load_prepared_iset(cache_prefix)
    elapsed = time.perf_counter() - started
    actual_sha256 = _sha256_path(npz_path)
    return prepared, {
        "source_path": str(npz_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_path(manifest_path),
        "expected_sha256": manifest["npz_sha256"],
        "actual_sha256": actual_sha256,
        "verified": actual_sha256 == manifest["npz_sha256"],
        "cache_schema_version": manifest["cache_schema_version"],
    }, elapsed


def _table_v_positions(
    result: ExecutionResult, prepared: IsetPaperLiteralData
) -> np.ndarray:
    """Locate the fixed Table V benign rows in Table III's ordered test set."""

    benign_count = int(prepared.metadata["counts"]["anomaly_b2_benign"])
    test_benign_ids = result.sample_ids[:benign_count]
    selected = np.flatnonzero(
        np.isin(test_benign_ids, prepared.table_v_benign.sample_ids)
    )
    expected = prepared.table_v_benign.sample_ids.size
    if selected.size != expected:
        raise RuntimeError(
            f"Table V index recovery found {selected.size} benign rows, expected {expected}"
        )
    if set(test_benign_ids[selected]) != set(prepared.table_v_benign.sample_ids):
        raise RuntimeError("Table V benign identities do not match Table III")
    for attack_id, attack in enumerate(prepared.table_v_attacks, start=1):
        positions = benign_count + (attack_id - 1) * benign_count + selected
        if set(result.sample_ids[positions]) != set(attack.sample_ids):
            raise RuntimeError(
                f"Table V attack {attack_id} identities do not match Table III"
            )
    return selected


def _score_orientation(result: ExecutionResult, branch: str) -> str:
    positive_if = result.metadata.get("positive_if", "higher")
    orientation = (
        positive_if.get(branch, "higher")
        if isinstance(positive_if, Mapping)
        else positive_if
    )
    if orientation not in {"higher", "lower"}:
        raise ValueError(
            f"invalid score orientation {orientation!r} for branch {branch!r}"
        )
    return str(orientation)


def _execution_threshold(
    result: ExecutionResult,
    model_name: str,
    contract: Any,
) -> float:
    thresholds = result.metadata.get("score_thresholds")
    if isinstance(thresholds, Mapping) and thresholds:
        primary = PRIMARY_SCORE.get(model_name)
        if primary in thresholds:
            return float(thresholds[primary])
        first_score = next(iter(result.scores))
        if first_score in thresholds:
            return float(thresholds[first_score])
    return float(contract.thresholds[model_name])


def _identity_digest(sample_ids: np.ndarray) -> str:
    digest = hashlib.sha256()
    for sample_id in np.asarray(sample_ids).astype(str):
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _table_v_selection(
    result: ExecutionResult,
    prepared: IsetPaperLiteralData,
    *,
    attack_id: int,
    size: str,
    resplit: bool,
    seed: int,
) -> np.ndarray:
    """Select one Table-V benign experiment without inspecting its scores."""

    if size not in TABLE_V_SIZES:
        raise ValueError(f"unknown Table V size branch {size!r}")
    benign_count = int(prepared.metadata["counts"]["anomaly_b2_benign"])
    if size == "full_heldout":
        return np.arange(benign_count, dtype=np.int64)
    if not resplit:
        return _table_v_positions(result, prepared)
    count = min(3_000, benign_count)
    selection_seed = int(
        np.random.SeedSequence([int(seed), int(attack_id), 0x7AB1E5])
        .generate_state(1, dtype=np.uint32)[0]
    )
    return np.sort(
        np.random.default_rng(selection_seed).choice(
            benign_count,
            size=count,
            replace=False,
        )
    ).astype(np.int64)


def table_v_results_from_executions(
    executions: Mapping[int, ExecutionResult],
    prepared: IsetPaperLiteralData,
    *,
    threshold: float,
    identity: str,
    size: str,
    seed: int,
    model_seeds: Mapping[int, int],
) -> tuple[Mapping[str, Any], Mapping[int, np.ndarray]]:
    """Evaluate all Table-V columns under one declared experiment identity."""

    if identity not in TABLE_V_IDENTITIES:
        raise ValueError(f"unknown Table V identity branch {identity!r}")
    if set(executions) != set(range(1, 7)):
        raise ValueError("Table V requires one execution mapping for attacks 1 through 6")
    resplit = identity in {"resplit_per_attack", "retrain_and_resplit"}
    first = executions[1]
    score_names = tuple(first.scores)
    if not score_names:
        raise ValueError("Table V execution has no score branches")
    selections = {
        attack_id: _table_v_selection(
            executions[attack_id],
            prepared,
            attack_id=attack_id,
            size=size,
            resplit=resplit,
            seed=seed,
        )
        for attack_id in range(1, 7)
    }
    benign_count = int(prepared.metadata["counts"]["anomaly_b2_benign"])
    score_branches: dict[str, Any] = {}
    for branch in score_names:
        evaluated: dict[int, Any] = {}
        identity_records: dict[str, Any] = {}
        orientation = _score_orientation(first, branch)
        for attack_id in range(1, 7):
            result = executions[attack_id]
            if tuple(result.scores) != score_names:
                raise ValueError("Table V retraining changed the score-branch set")
            if _score_orientation(result, branch) != orientation:
                raise ValueError("Table V retraining changed score orientation")
            selected = selections[attack_id]
            attack_positions = (
                benign_count + (attack_id - 1) * benign_count + selected
            )
            one = evaluate_attack_columns(
                result.scores[branch][selected],
                {attack_id: result.scores[branch][attack_positions]},
                threshold=threshold,
                positive_if=orientation,
            )[attack_id]
            evaluated[attack_id] = one
            identity_records[str(attack_id)] = {
                "model_seed": int(model_seeds[attack_id]),
                "benign_sample_id_sha256": _identity_digest(
                    result.sample_ids[selected]
                ),
                "attack_sample_id_sha256": _identity_digest(
                    result.sample_ids[attack_positions]
                ),
                "samples_per_class": int(selected.size),
            }
        attacks = {
            str(attack_id): metrics.as_dict()
            for attack_id, metrics in evaluated.items()
        }
        score_branches[branch] = {
            "positive_if": orientation,
            "attacks": attacks,
            "average": {
                metric: float(
                    np.mean([getattr(item, metric) for item in evaluated.values()])
                )
                for metric in (
                    "dr",
                    "fa",
                    "sp",
                    "precision",
                    "balanced_accuracy",
                    "f1",
                    "auc",
                )
            },
            "false_alarm_invariant": len(
                {
                    (item.fp, item.tn, item.fa, item.sp)
                    for item in evaluated.values()
                }
            )
            == 1,
            "experiment_identities": identity_records,
        }
    primary = PRIMARY_SCORE.get(str(first.metadata["model_name"]))
    if primary not in score_branches:
        primary = score_names[0]
    return {
        "table": 5,
        "identity_branch": identity,
        "size_branch": size,
        "samples_per_class": {
            str(attack_id): int(selections[attack_id].size)
            for attack_id in range(1, 7)
        },
        "primary_score_branch": primary,
        "threshold": float(threshold),
        "score_branches": score_branches,
        "full_set_resplit_degeneracy": bool(
            resplit and size == "full_heldout"
        ),
        "derivation": (
            "six explicit attack-column experiments; model and benign identity "
            "reuse are determined only by identity_branch"
        ),
    }, selections


def table_v_results(
    result: ExecutionResult,
    prepared: IsetPaperLiteralData,
    *,
    threshold: float,
) -> Mapping[str, Any]:
    """Derive every Table V column from one Table III model realization."""

    selected = _table_v_positions(result, prepared)
    benign_count = int(prepared.metadata["counts"]["anomaly_b2_benign"])
    score_branches: dict[str, Any] = {}
    for branch, values in result.scores.items():
        benign_scores = values[selected]
        attack_scores = {
            attack_id: values[
                benign_count + (attack_id - 1) * benign_count + selected
            ]
            for attack_id in range(1, 7)
        }
        orientation = _score_orientation(result, branch)
        evaluated = evaluate_attack_columns(
            benign_scores,
            attack_scores,
            threshold=threshold,
            positive_if=orientation,
        )
        attacks = {
            str(attack_id): metrics.as_dict()
            for attack_id, metrics in evaluated.items()
        }
        score_branches[branch] = {
            "positive_if": orientation,
            "attacks": attacks,
            "average": {
                metric: float(
                    np.mean([getattr(item, metric) for item in evaluated.values()])
                )
                for metric in (
                    "dr",
                    "fa",
                    "sp",
                    "precision",
                    "balanced_accuracy",
                    "f1",
                    "auc",
                )
            },
            "false_alarm_invariant": len(
                {
                    (item.fp, item.tn, item.fa, item.sp)
                    for item in evaluated.values()
                }
            )
            == 1,
        }
    primary = PRIMARY_SCORE.get(str(result.metadata["model_name"]))
    if primary not in score_branches:
        primary = next(iter(score_branches))
    return {
        "table": 5,
        "samples_per_class": int(selected.size),
        "primary_score_branch": primary,
        "threshold": float(threshold),
        "score_branches": score_branches,
        "derivation": (
            "fixed Table V row identities indexed from the same Table III "
            "score vector; no retraining, threshold change, or resampling"
        ),
    }


def _derived_table_v_model_seed(seed: int, attack_id: int) -> int:
    return int(
        np.random.SeedSequence([int(seed), int(attack_id), 0x5EED5])
        .generate_state(1, dtype=np.uint32)[0]
    )


def execute_table_v_identity(
    model_name: str,
    prepared: IsetPaperLiteralData,
    contract: Any,
    seed: int,
    *,
    identity: str,
    size: str,
    model_overrides: Mapping[str, Any] | None = None,
    validation_policy: str = "holdout_no_refit",
    threshold_rule: str = "printed_constant",
    threshold_scope: str = "iset_transferred",
    validation_labels: str = "printed_threshold_no_derivation",
    transferred_thresholds: Mapping[str, float] | None = None,
) -> ExecutionResult:
    """Train/score the six explicit Table-V experiments and retain every row."""

    if model_name not in ANOMALY_NEURAL_MODELS:
        raise ValueError("Table V contains only anomaly neural models")
    if identity not in TABLE_V_IDENTITIES:
        raise ValueError(f"unknown Table V identity branch {identity!r}")
    retrain = identity in {"retrain_per_attack", "retrain_and_resplit"}
    model_seeds = {
        attack_id: (
            _derived_table_v_model_seed(seed, attack_id) if retrain else int(seed)
        )
        for attack_id in range(1, 7)
    }
    if retrain:
        executions = {
            attack_id: execute_selected_model(
                model_name,
                prepared,
                contract,
                model_seeds[attack_id],
                model_overrides=model_overrides,
                validation_policy=validation_policy,
                threshold_rule=threshold_rule,
                threshold_scope=threshold_scope,
                validation_labels=validation_labels,
                transferred_thresholds=transferred_thresholds,
            )
            for attack_id in range(1, 7)
        }
    else:
        shared = execute_selected_model(
            model_name,
            prepared,
            contract,
            int(seed),
            model_overrides=model_overrides,
            validation_policy=validation_policy,
            threshold_rule=threshold_rule,
            threshold_scope=threshold_scope,
            validation_labels=validation_labels,
            transferred_thresholds=transferred_thresholds,
        )
        executions = {attack_id: shared for attack_id in range(1, 7)}
    executions = {
        attack_id: dataclasses.replace(
            result,
            metadata={**dict(result.metadata), "model_name": model_name},
        )
        for attack_id, result in executions.items()
    }
    for result in executions.values():
        if result.metadata.get("model_name") not in {None, model_name}:
            raise ValueError("Table V execution returned the wrong model")
    table_v, selections = table_v_results_from_executions(
        executions,
        prepared,
        threshold=_execution_threshold(
            executions[1],
            model_name,
            contract,
        ),
        identity=identity,
        size=size,
        seed=seed,
        model_seeds=model_seeds,
    )
    benign_count = int(prepared.metadata["counts"]["anomaly_b2_benign"])
    labels_parts: list[np.ndarray] = []
    sample_id_parts: list[np.ndarray] = []
    synthetic_parts: list[np.ndarray] = []
    score_parts: dict[str, list[np.ndarray]] = {
        branch: [] for branch in executions[1].scores
    }
    for attack_id in range(1, 7):
        result = executions[attack_id]
        selected = selections[attack_id]
        attack_positions = benign_count + (attack_id - 1) * benign_count + selected
        positions = np.concatenate([selected, attack_positions])
        labels_parts.append(
            np.concatenate(
                [
                    np.zeros(selected.size, dtype=np.int8),
                    np.ones(selected.size, dtype=np.int8),
                ]
            )
        )
        sample_id_parts.append(
            np.asarray(
                [
                    f"table5_attack_{attack_id}:{sample_id}"
                    for sample_id in result.sample_ids[positions]
                ],
                dtype=str,
            )
        )
        synthetic_parts.append(result.is_synthetic[positions])
        for branch in score_parts:
            score_parts[branch].append(result.scores[branch][positions])
    scores = {
        branch: np.concatenate(parts).astype(np.float64, copy=False)
        for branch, parts in score_parts.items()
    }
    labels = np.concatenate(labels_parts)
    predictions = {
        branch: threshold_predictions(
            values,
            _execution_threshold(executions[1], model_name, contract),
            positive_if=_score_orientation(executions[1], branch),
        )
        for branch, values in scores.items()
    }
    unique_results = (
        list(executions.values()) if retrain else [executions[1]]
    )
    return ExecutionResult(
        scores=scores,
        predictions=predictions,
        labels=labels,
        sample_ids=np.concatenate(sample_id_parts),
        is_synthetic=np.concatenate(synthetic_parts),
        history={
            "identity_branch": identity,
            "model_seeds": {str(key): value for key, value in model_seeds.items()},
            "per_attack": {
                str(attack_id): executions[attack_id].history
                for attack_id in range(1, 7)
            },
        },
        metrics={
            branch: table_v["score_branches"][branch]["average"]
            for branch in scores
        },
        fit_seconds=float(sum(result.fit_seconds for result in unique_results)),
        score_seconds=float(sum(result.score_seconds for result in unique_results)),
        metadata={
            "model_name": model_name,
            "dataset_adapter": "exact_iset_cache_v1",
            "table_v_identity": identity,
            "table_v_size": size,
            "logical_model_fits": len(unique_results),
            "model_seeds": {str(key): value for key, value in model_seeds.items()},
            "positive_if": executions[1].metadata.get("positive_if", "higher"),
        },
        supplemental_results={"table_5": table_v},
    )


def execute_table_iii(
    model_name: str,
    prepared: IsetPaperLiteralData,
    contract: Any,
    seed: int,
    *,
    model_overrides: Mapping[str, Any] | None = None,
    validation_policy: str = "holdout_no_refit",
    threshold_rule: str = "printed_constant",
    threshold_scope: str = "iset_transferred",
    validation_labels: str = "printed_threshold_no_derivation",
    transferred_thresholds: Mapping[str, float] | None = None,
    classical_options: Mapping[str, Any] | None = None,
    derive_historical_table_v: bool = True,
) -> ExecutionResult:
    result = execute_selected_model(
        model_name,
        prepared,
        contract,
        seed,
        model_overrides=model_overrides,
        validation_policy=validation_policy,
        threshold_rule=threshold_rule,
        threshold_scope=threshold_scope,
        validation_labels=validation_labels,
        transferred_thresholds=transferred_thresholds,
        classical_options=classical_options,
    )
    execution_metadata = {
        **dict(result.metadata),
        "model_name": model_name,
        "dataset_adapter": "exact_iset_cache_v1",
        "persisted_array_alignment": (
            "score and prediction row i align to the checksum-verified prepared "
            "test partition row i; repeated labels and string IDs remain in cache"
        ),
    }
    supplemental: Mapping[str, Any] = {}
    if model_name in ANOMALY_NEURAL_MODELS and derive_historical_table_v:
        staged = dataclasses.replace(result, metadata=execution_metadata)
        supplemental = {
            "table_5": table_v_results(
                staged,
                prepared,
                threshold=_execution_threshold(result, model_name, contract),
            )
        }
    return dataclasses.replace(
        result,
        metadata=execution_metadata,
        supplemental_results=supplemental,
    )


def compact_iset_arrays(result: ExecutionResult) -> Mapping[str, np.ndarray]:
    """Persist run-specific outputs; shared row provenance stays in the cache."""

    arrays: dict[str, np.ndarray] = {}
    for name, values in result.scores.items():
        arrays[f"score__{name}"] = np.asarray(values, dtype=np.float64)
    for name, values in result.predictions.items():
        arrays[f"prediction__{name}"] = np.asarray(values, dtype=np.int8)
    expected = result.labels.size
    if not arrays or any(array.ndim != 1 or array.size != expected for array in arrays.values()):
        raise ValueError("ISET score and prediction arrays must align to the test cache")
    return arrays


def _table_scope(
    table: int,
    *,
    size: str | None = None,
    table_v_identity: str | None = None,
    table_v_size: str | None = None,
    runtime_branch: Mapping[str, Any] | None = None,
) -> RunScope:
    extra: dict[str, Any] = {
        "iset_cache_adapter": "exact_iset_cache_v1",
        "row_provenance_storage": "checksum_verified_cache_by_row_index",
    }
    path = (
        ("branches", str(runtime_branch["branch_id"]), f"table_{table}", "iset")
        if runtime_branch is not None
        else (f"table_{table}", "iset")
    )
    if runtime_branch is not None:
        extra["branch_runtime"] = dict(runtime_branch)
    if size is not None:
        extra["table_iv_training_size"] = size
        path = (*path, size)
    if table == 3:
        extra["coupled_table_v_derivation"] = runtime_branch is None
    if table == 5:
        if table_v_identity is None or table_v_size is None:
            raise ValueError("Table V scope requires identity and size branches")
        extra["table_v_identity"] = table_v_identity
        extra["table_v_size"] = table_v_size
        extra["row_provenance_storage"] = "persisted_in_attempt_arrays"
        path = (*path, table_v_identity, table_v_size)
    return RunScope(
        table,
        "CER/ISET",
        path,
        source_code_files=("paper_literal_iset.py", "paper_literal_iset_runner.py"),
        fingerprint_extra=extra,
    )


def _table_iv_view(
    prepared: IsetPaperLiteralData, size: str
) -> Any:
    return type(
        "IsetTableIVView",
        (),
        {
            "anomaly_train": prepared.table_iv_subset(size),
            "anomaly_validation": prepared.anomaly_validation,
            "anomaly_test": prepared.anomaly_test,
            "supervised_train": prepared.supervised_train,
            "supervised_test": prepared.supervised_test,
            "metadata": prepared.metadata,
        },
    )()


def run_iset(
    *,
    cache_prefix: str | Path,
    output: str | Path,
    config: str | Path,
    table: int,
    models: Sequence[str],
    seeds: Sequence[int] | None,
    sizes: Sequence[str] = TABLE_IV_SIZES,
    table_v_identity: str = "common_model_common_benign",
    table_v_size: str = "seeded_3000",
    branch_id: str | None = None,
    branch_manifest: str | Path = DEFAULT_LATTICE,
    transferred_thresholds_path: str | Path | None = None,
    preflight_only: bool = False,
    force: bool = False,
) -> list[RunOutcome]:
    """Load the exact cache once, then execute the requested declared cells."""

    if table not in {3, 4, 5}:
        raise ValueError("ISET execution accepts Table 3, Table 4, or Table 5")
    runtime_branch: Mapping[str, Any] | None = None
    if branch_id is not None:
        runtime_branch = load_runtime_branch(
            branch_id,
            manifest=branch_manifest,
        )
        assert_branch_scope(
            runtime_branch,
            dataset="iset",
            table=table,
        )
        selected_models = [str(runtime_branch["model"])]
        if "table_v_identity" in runtime_branch["execution"]:
            table_v_identity = str(
                runtime_branch["execution"]["table_v_identity"]
            )
        if "table_v_size" in runtime_branch["execution"]:
            table_v_size = str(runtime_branch["execution"]["table_v_size"])
    else:
        selected_models = resolve_models(models)
    contract = load_contract(config)
    selected_seeds = resolve_seeds(contract, seeds)
    if table == 4:
        unsupported = sorted(set(selected_models).difference(ANOMALY_NEURAL_MODELS))
        if unsupported:
            raise ValueError(f"Table IV contains only anomaly models: {unsupported}")
        invalid_sizes = sorted(set(sizes).difference(TABLE_IV_SIZES))
        if invalid_sizes:
            raise ValueError(f"invalid Table IV sizes: {invalid_sizes}")
    if table == 5:
        unsupported = sorted(set(selected_models).difference(ANOMALY_NEURAL_MODELS))
        if unsupported:
            raise ValueError(f"Table V contains only anomaly models: {unsupported}")
        if table_v_identity not in TABLE_V_IDENTITIES:
            raise ValueError(f"invalid Table V identity: {table_v_identity}")
        if table_v_size not in TABLE_V_SIZES:
            raise ValueError(f"invalid Table V size: {table_v_size}")

    prepared, verification, load_seconds = load_verified_cache(cache_prefix)
    if runtime_branch is not None:
        cache_branch = prepared.metadata.get("branch_runtime")
        if not isinstance(cache_branch, Mapping):
            raise ValueError(
                "branch execution requires a cache prepared with --branch-id"
            )
        if cache_branch.get("preparation_id") != runtime_branch["preparation_id"]:
            raise ValueError(
                "prepared cache branch mismatch: expected "
                f"{runtime_branch['preparation_id']}, received "
                f"{cache_branch.get('preparation_id')}"
            )

    transferred_thresholds: Mapping[str, float] | None = None
    if transferred_thresholds_path is not None:
        raw_thresholds = json.loads(
            Path(transferred_thresholds_path).read_text(encoding="utf-8")
        )
        if not isinstance(raw_thresholds, dict):
            raise ValueError(
                "transferred-threshold artifact must be a JSON object"
            )
        transferred_thresholds = {
            str(key): float(value) for key, value in raw_thresholds.items()
        }
    execution_options = (
        dict(runtime_branch["execution"])
        if runtime_branch is not None
        else {}
    )
    model_overrides = (
        dict(runtime_branch["model_overrides"])
        if runtime_branch is not None
        else {}
    )
    classical_options = {
        key: value
        for key, value in execution_options.items()
        if key in {"arima_completion", "svm_training", "multiclass_labels"}
    }
    validation_policy = str(
        execution_options.get("validation_policy", "holdout_no_refit")
    )
    threshold_rule = str(
        execution_options.get("threshold_rule", "printed_constant")
    )
    threshold_scope = str(
        execution_options.get("threshold_scope", "iset_transferred")
    )
    validation_labels = str(
        execution_options.get(
            "validation_labels",
            "printed_threshold_no_derivation",
        )
    )
    preflight = {
        "schema_version": 1,
        "status": "ready",
        "scope": {
            "study": "atk-2022-deep-autoencoder",
            "table": table,
            "dataset": "CER/ISET",
            "table_3_includes_table_5": table == 3
            and runtime_branch is None,
            "table_v_identity": table_v_identity if table == 5 else None,
            "table_v_size": table_v_size if table == 5 else None,
        },
        "contract": {"path": str(contract.path), "sha256": contract.sha256},
        "data_verification": verification,
        "cache_load_seconds": load_seconds,
        "counts": prepared.metadata["counts"],
        "features": int(prepared.anomaly_train.values.shape[1]),
        "models": selected_models,
        "seeds": selected_seeds,
        "sizes": list(sizes) if table == 4 else [],
        "logical_model_fits_per_run": (
            6
            if table == 5
            and table_v_identity in {"retrain_per_attack", "retrain_and_resplit"}
            else 1
        ),
        "planned_runs": len(selected_models)
        * len(selected_seeds)
        * (len(sizes) if table == 4 else 1),
        "branch_runtime": runtime_branch,
    }
    preflight_path = _write_preflight(Path(output), preflight)
    print(json.dumps({**preflight, "artifact": str(preflight_path)}, sort_keys=True))
    if preflight_only:
        return []

    outcomes: list[RunOutcome] = []
    for model_name in selected_models:
        for seed in selected_seeds:
            selected_sizes: Sequence[str | None] = sizes if table == 4 else (None,)
            for size in selected_sizes:
                run_prepared = (
                    _table_iv_view(prepared, str(size))
                    if size is not None
                    else prepared
                )
                if table == 4:
                    executor = lambda name, data, frozen, run_seed: (
                        execute_selected_model(
                            name,
                            data,
                            frozen,
                            run_seed,
                            model_overrides=model_overrides,
                            validation_policy=validation_policy,
                            threshold_rule=threshold_rule,
                            threshold_scope=threshold_scope,
                            validation_labels=validation_labels,
                            transferred_thresholds=transferred_thresholds,
                            classical_options=classical_options,
                            derive_historical_table_v=runtime_branch is None,
                        )
                    )
                elif table == 3:
                    executor = lambda name, data, frozen, run_seed: (
                        execute_table_iii(
                            name,
                            data,
                            frozen,
                            run_seed,
                            model_overrides=model_overrides,
                            validation_policy=validation_policy,
                            threshold_rule=threshold_rule,
                            threshold_scope=threshold_scope,
                            validation_labels=validation_labels,
                            transferred_thresholds=transferred_thresholds,
                            classical_options=classical_options,
                        )
                    )
                else:
                    executor = lambda name, data, frozen, run_seed: (
                        execute_table_v_identity(
                            name,
                            data,
                            frozen,
                            run_seed,
                            identity=table_v_identity,
                            size=table_v_size,
                            model_overrides=model_overrides,
                            validation_policy=validation_policy,
                            threshold_rule=threshold_rule,
                            threshold_scope=threshold_scope,
                            validation_labels=validation_labels,
                            transferred_thresholds=transferred_thresholds,
                        )
                    )
                outcome = run_one(
                    output=output,
                    model_name=model_name,
                    seed=seed,
                    prepared=run_prepared,
                    contract=contract,
                    verification=verification,
                    data_prep_seconds=load_seconds,
                    force=force,
                    executor=executor,
                    scope=_table_scope(
                        table,
                        size=None if size is None else str(size),
                        table_v_identity=(
                            table_v_identity if table == 5 else None
                        ),
                        table_v_size=table_v_size if table == 5 else None,
                        runtime_branch=runtime_branch,
                    ),
                    array_builder=(
                        None if table == 5 else compact_iset_arrays
                    ),
                )
                outcomes.append(outcome)
                print(json.dumps(_jsonable(outcome), sort_keys=True))
                if outcome.status == "interrupted":
                    return outcomes
            if model_name not in {
                "naive_bayes",
                "arima",
                "one_class_svm",
                "multiclass_svm",
            }:
                try:
                    import keras

                    keras.backend.clear_session()
                except Exception:
                    pass
    return outcomes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--cache-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--table", type=int, choices=(3, 4, 5), required=True)
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--sizes", nargs="+", choices=TABLE_IV_SIZES, default=list(TABLE_IV_SIZES))
    parser.add_argument(
        "--table-v-identity",
        choices=TABLE_V_IDENTITIES,
        default="common_model_common_benign",
    )
    parser.add_argument(
        "--table-v-size",
        choices=TABLE_V_SIZES,
        default="seeded_3000",
    )
    parser.add_argument("--branch-id")
    parser.add_argument(
        "--branch-manifest",
        type=Path,
        default=DEFAULT_LATTICE,
    )
    parser.add_argument("--transferred-thresholds", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outcomes = run_iset(
            cache_prefix=args.cache_prefix,
            output=args.output,
            config=args.config,
            table=args.table,
            models=args.models,
            seeds=args.seeds,
            sizes=args.sizes,
            table_v_identity=args.table_v_identity,
            table_v_size=args.table_v_size,
            branch_id=args.branch_id,
            branch_manifest=args.branch_manifest,
            transferred_thresholds_path=args.transferred_thresholds,
            preflight_only=args.command == "preflight",
            force=args.force,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "preflight_failed",
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                sort_keys=True,
            )
        )
        return 2
    return int(
        any(outcome.status in {"failed", "interrupted"} for outcome in outcomes)
    )


if __name__ == "__main__":
    raise SystemExit(main())
