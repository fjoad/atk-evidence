"""Resolve one frozen Paper 1 branch into preparation and model arguments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from branch_lattice import resolve_branch


STUDY = Path(__file__).resolve().parents[1]
DEFAULT_LATTICE = STUDY / "config" / "branch_lattice.toml"

NEURAL_MODELS = {
    "fc_sae",
    "lstm_sae",
    "fc_vae",
    "lstm_vae",
    "lstm_aea",
    "supervised_feed_forward",
    "supervised_lstm",
}

MODEL_CHOICE_KEYS = {
    "latent_placement",
    "dense_dropout_scope",
    "lstm_input",
    "decoder_schedule",
    "decoder_state",
    "attention_merge",
    "vae_loss_reduction",
    "vae_score",
    "lstm_dropout_placement",
    "supervised_head",
}


def load_runtime_branch(
    branch_id: str,
    *,
    manifest: str | Path = DEFAULT_LATTICE,
) -> dict[str, Any]:
    """Return a stable branch plus normalized executable arguments."""

    branch = resolve_branch(manifest, branch_id)
    return runtime_from_resolved_branch(branch)


def runtime_from_resolved_branch(
    branch: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize a branch record already produced by the lattice enumerator."""

    choices = {str(key): str(value) for key, value in branch["choices"].items()}
    if branch["track"] == "C":
        corrected_execution = {
            "validation_policy": "holdout_no_refit",
            "threshold_rule": "validation_youden_j",
            "threshold_scope": "dataset_specific",
            "validation_labels": "b1_generated_attacks",
        }
        if branch["model"] == "arima":
            corrected_execution["arima_completion"] = "p1_pooled_likelihood"
        if branch["model"] in {"one_class_svm", "multiclass_svm"}:
            corrected_execution["svm_training"] = "full_data"
        if (
            branch["dataset"] == "iset"
            and branch["model"] == "multiclass_svm"
        ):
            corrected_execution["multiclass_labels"] = (
                "benign_plus_six_attacks"
            )
        runtime = {
            **branch,
            "preparation": _corrected_preparation(branch),
            "model_overrides": _corrected_model_overrides(branch),
            "execution": corrected_execution,
        }
        return {
            **runtime,
            "preparation_id": preparation_id(runtime),
        }

    preparation: dict[str, Any] = {
        "scaling": choices.get("scaling"),
        "anomaly_adasyn": choices.get("anomaly_adasyn"),
        "supervised_adasyn": choices.get("supervised_adasyn"),
        "adasyn_neighbors": (
            int(choices["adasyn_neighbors"])
            if "adasyn_neighbors" in choices
            else None
        ),
        "split_unit": choices.get("split_unit"),
    }
    if branch["dataset"] == "sgcc":
        preparation.update(
            {
                "representation": choices.get("sgcc_representation"),
                "missing": choices.get("sgcc_missing"),
            }
        )
    else:
        preparation.update(
            {
                "iset_day": choices.get("iset_day"),
                "meter_population": choices.get("iset_meter_population"),
                "attack_population": choices.get("attack_population"),
                "attack1_scope": choices.get("attack1_scope"),
                "attack2_granularity": choices.get("attack2_granularity"),
                "attack3_interval": choices.get("attack3_interval"),
                "attack_hour_mapping": choices.get("attack_hour_mapping"),
                "attack_regeneration": {
                    "fixed_per_data_seed": "fixed_per_data_seed",
                    "per_model_seed": "regenerate_per_model_seed",
                    "per_experiment": "regenerate_per_experiment",
                }.get(choices.get("attack_regeneration", "")),
            }
        )
    preparation = {
        key: value for key, value in preparation.items() if value is not None
    }

    model_overrides: dict[str, Any] = {}
    if branch["model"] in NEURAL_MODELS:
        model_overrides["architecture_contract"] = "paper_source_v2"
        for key in MODEL_CHOICE_KEYS:
            if key in choices:
                model_overrides[key] = choices[key]
        if "latent_width" in choices:
            model_overrides["latent_width"] = int(choices["latent_width"])

    execution = {
        key: choices[key]
        for key in (
            "validation_policy",
            "threshold_rule",
            "threshold_scope",
            "validation_labels",
            "table_v_identity",
            "table_v_size",
            "arima_completion",
            "svm_training",
            "multiclass_labels",
        )
        if key in choices
    }
    runtime = {
        **branch,
        "preparation": preparation,
        "model_overrides": model_overrides,
        "execution": execution,
    }
    return {
        **runtime,
        "preparation_id": preparation_id(runtime),
    }


def preparation_id(branch: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {
            "dataset": branch["dataset"],
            "preparation": branch["preparation"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"prep-{hashlib.sha256(payload).hexdigest()[:16]}"


def _corrected_preparation(branch: Mapping[str, Any]) -> dict[str, Any]:
    dataset = str(branch["dataset"])
    model = str(branch["model"])
    base: dict[str, Any] = {
        "scaling": "train_benign_only",
        "anomaly_adasyn": "none",
        "supervised_adasyn": "customer_split_then_train_only",
        "split_unit": "customer_disjoint",
    }
    if dataset == "sgcc":
        base.update(
            {
                "representation": "windows_48_rolling",
                "missing": "interpolate_edge_median",
            }
        )
    else:
        base.update(
            {
                "iset_day": "interpolate_grid",
                "meter_population": "all_4225",
                "attack_population": (
                    "heldout_b2_m"
                    if model
                    in {
                        "fc_sae",
                        "lstm_sae",
                        "fc_vae",
                        "lstm_vae",
                        "lstm_aea",
                        "arima",
                        "one_class_svm",
                    }
                    else "all_customer_m"
                ),
                "attack1_scope": "per_profile",
                "attack2_granularity": "per_half_hour",
                "attack3_interval": "valid_fit_addition",
                "attack_hour_mapping": "two_slots_per_hour",
                "attack_regeneration": "fixed_per_data_seed",
            }
        )
    return base


def _corrected_model_overrides(branch: Mapping[str, Any]) -> dict[str, Any]:
    model = str(branch["model"])
    if model not in NEURAL_MODELS:
        return {}
    # Corrected controls remain separately fingerprinted while reusing the
    # source-derived layer topology. Their correction is in isolation,
    # selection, calibration, and (for VAEs) likelihood scoring.
    overrides = {
        "architecture_contract": "corrected_control_v1",
    }
    if model.endswith("_vae"):
        overrides.update(
            {
                "vae_score": "prob_learned_var_mc100",
                "vae_loss_reduction": "mean_mse_plus_kl",
            }
        )
    if model in {"supervised_feed_forward", "supervised_lstm"}:
        overrides["supervised_head"] = "sigmoid1_binary"
    return overrides


def assert_branch_scope(
    branch: Mapping[str, Any],
    *,
    dataset: str,
    model: str | None = None,
    table: int | None = None,
) -> None:
    if str(branch["dataset"]) != dataset:
        raise ValueError(
            f"branch {branch['branch_id']} targets {branch['dataset']}, not {dataset}"
        )
    if model is not None and str(branch["model"]) != model:
        raise ValueError(
            f"branch {branch['branch_id']} targets {branch['model']}, not {model}"
        )
    if table is not None:
        roman = {2: "II", 3: "III", 4: "IV", 5: "V"}[table]
        if roman not in branch["tables"]:
            raise ValueError(
                f"branch {branch['branch_id']} does not cover Table {roman}"
            )
