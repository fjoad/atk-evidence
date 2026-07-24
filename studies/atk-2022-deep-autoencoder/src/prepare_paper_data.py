#!/usr/bin/env python3
"""Prepare the paper-described SGCC or CER/ISET data in one explicit step."""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

from paper_literal_data import (
    ANOMALY_ADASYN_BRANCHES,
    SCALING_BRANCHES,
    SGCC_MISSING_BRANCHES,
    SGCC_REPRESENTATION_BRANCHES,
    SPLIT_UNIT_BRANCHES,
    SUPERVISED_ADASYN_BRANCHES,
    save_prepared_sgcc,
)
from paper_literal_iset import (
    ATTACK1_SCOPE_BRANCHES,
    ATTACK2_GRANULARITY_BRANCHES,
    ATTACK3_INTERVAL_BRANCHES,
    ATTACK_HOUR_MAPPING_BRANCHES,
    ATTACK_POPULATION_BRANCHES,
    ATTACK_REGENERATION_BRANCHES,
    METER_POPULATION_BRANCHES,
    SPLIT_UNIT_BRANCHES as ISET_SPLIT_UNIT_BRANCHES,
    allocation_filename,
    prepare_iset_paper_literal,
    save_prepared_iset,
)
from cer_parser import ISET_DAY_BRANCHES
from paper_literal_runner import load_contract, verify_and_prepare_sgcc
from branch_runtime import (
    DEFAULT_LATTICE,
    assert_branch_scope,
    load_runtime_branch,
)


STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
DEFAULT_CONFIG = STUDY / "config/exploratory_reproduction.toml"
DEFAULT_ISET_CONFIG = STUDY / "config/exploratory_iset.toml"
DEFAULT_CACHE_ROOT = REPO / "data/derived/atk-2022-deep-autoencoder"


def _attach_branch_metadata(prepared: object, args: argparse.Namespace) -> object:
    runtime_branch = getattr(args, "runtime_branch", None)
    if runtime_branch is None:
        return prepared
    metadata = {
        **dict(getattr(prepared, "metadata")),
        "branch_runtime": {
            "dataset": runtime_branch["dataset"],
            "preparation_id": runtime_branch["preparation_id"],
            "preparation": runtime_branch["preparation"],
            "identity_policy": (
                "content_addressed_preparation_only; model/evaluation branch "
                "identity belongs to each run fingerprint"
            ),
        },
    }
    return dataclasses.replace(prepared, metadata=metadata)


def prepare_sgcc(args: argparse.Namespace) -> dict[str, object]:
    contract = load_contract(args.config or DEFAULT_CONFIG)
    prepared, verification, seconds = verify_and_prepare_sgcc(
        args.data,
        contract,
        scaling=args.scaling,
        anomaly_adasyn=args.anomaly_adasyn,
        supervised_adasyn=args.supervised_adasyn,
        adasyn_neighbors=args.adasyn_neighbors,
        representation=args.representation,
        missing=args.missing,
        split_unit=args.split_unit,
    )
    prepared = _attach_branch_metadata(prepared, args)
    npz_path, manifest_path = save_prepared_sgcc(prepared, args.output)
    return {
        "dataset": "SGCC",
        "source": verification,
        "preparation_seconds": seconds,
        "counts": prepared.metadata["counts"],
        "cache": str(npz_path),
        "manifest": str(manifest_path),
    }


def prepare_iset(args: argparse.Namespace) -> dict[str, object]:
    contract = load_contract(args.config or DEFAULT_ISET_CONFIG)
    branch = str(contract.data["iset_allocation_branch"])
    archives = [args.data_dir / f"File{index}.txt.zip" for index in range(1, 7)]
    allocation = args.data_dir / allocation_filename(branch)
    prepared = prepare_iset_paper_literal(
        archive_paths=archives,
        allocation_source=allocation,
        allocation_branch=branch,
        data_seed=int(contract.run["data_seed"]),
        validation_fraction=float(contract.run["validation_fraction_within_train"]),
        adasyn_neighbors=(
            int(contract.data["adasyn_neighbors"])
            if args.adasyn_neighbors is None
            else int(args.adasyn_neighbors)
        ),
        table_v_samples=int(contract.data["table_v_samples_per_class"]),
        attack_population=args.attack_population,
        scaling=args.scaling,
        anomaly_adasyn=args.anomaly_adasyn,
        supervised_adasyn=args.supervised_adasyn,
        attack_seed=args.attack_seed,
        attack1_scope=args.attack1_scope,
        attack2_granularity=args.attack2_granularity,
        attack3_interval=args.attack3_interval,
        attack_hour_mapping=args.attack_hour_mapping,
        attack_regeneration=args.attack_regeneration,
        model_seed=args.model_seed,
        experiment_index=args.experiment_index,
        meter_population=args.meter_population,
        iset_day=args.iset_day,
        split_unit=args.split_unit,
    )
    prepared = _attach_branch_metadata(prepared, args)
    npz_path, manifest_path = save_prepared_iset(prepared, args.output)
    return {
        "dataset": "CER/ISET",
        "allocation_branch": branch,
        "source": prepared.metadata["source"],
        "counts": prepared.metadata["counts"],
        "attack_generation": prepared.metadata["preprocessing"],
        "cache": str(npz_path),
        "manifest": str(manifest_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "frozen contract; defaults to the preserved SGCC contract for sgcc "
            "and the separate ISET-phase contract for iset"
        ),
    )
    parser.add_argument(
        "--branch-id",
        help="stable Paper 1 branch ID; overrides matching preparation options",
    )
    parser.add_argument(
        "--branch-manifest",
        type=Path,
        default=DEFAULT_LATTICE,
    )
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    sgcc = subparsers.add_parser("sgcc", help="prepare labeled SGCC customers")
    sgcc.add_argument(
        "--data", type=Path, default=REPO / "data/raw/sgcc-verified/data.csv"
    )
    sgcc.add_argument(
        "--output",
        type=Path,
        default=REPO / "data/derived/atk-2022-deep-autoencoder/sgcc-paper-literal",
    )
    sgcc.add_argument(
        "--scaling",
        choices=sorted(SCALING_BRANCHES),
        default="joint_featurewise",
    )
    sgcc.add_argument(
        "--anomaly-adasyn",
        choices=sorted(ANOMALY_ADASYN_BRANCHES),
        default="test_set_as_printed",
    )
    sgcc.add_argument(
        "--supervised-adasyn",
        choices=sorted(SUPERVISED_ADASYN_BRANCHES),
        default="before_row_split",
    )
    sgcc.add_argument("--adasyn-neighbors", type=int)
    sgcc.add_argument(
        "--representation",
        choices=sorted(SGCC_REPRESENTATION_BRANCHES),
        default="full_1034",
    )
    sgcc.add_argument(
        "--missing",
        choices=sorted(SGCC_MISSING_BRANCHES),
        default="interpolate_edge_median",
    )
    sgcc.add_argument(
        "--split-unit",
        choices=sorted(SPLIT_UNIT_BRANCHES),
        default="customer_disjoint",
    )

    iset = subparsers.add_parser(
        "iset", help="verify CER files and generate the six paper attacks"
    )
    iset.add_argument(
        "--data-dir", type=Path, default=REPO / "data/raw/cer-sciencedb"
    )
    iset.add_argument(
        "--output",
        type=Path,
        default=REPO / "data/derived/atk-2022-deep-autoencoder/iset-paper-literal",
    )
    iset.add_argument(
        "--attack-population",
        choices=sorted(ATTACK_POPULATION_BRANCHES),
        default="heldout_b2_m",
    )
    iset.add_argument(
        "--scaling",
        choices=sorted(SCALING_BRANCHES),
        default="joint_featurewise",
    )
    iset.add_argument(
        "--anomaly-adasyn",
        choices=sorted(ANOMALY_ADASYN_BRANCHES),
        default="test_set_as_printed",
    )
    iset.add_argument(
        "--supervised-adasyn",
        choices=sorted(SUPERVISED_ADASYN_BRANCHES),
        default="before_row_split",
    )
    iset.add_argument("--adasyn-neighbors", type=int)
    iset.add_argument("--attack-seed", type=int)
    iset.add_argument(
        "--attack-regeneration",
        choices=sorted(ATTACK_REGENERATION_BRANCHES),
        default="fixed_per_data_seed",
    )
    iset.add_argument("--model-seed", type=int)
    iset.add_argument("--experiment-index", type=int)
    iset.add_argument(
        "--attack1-scope",
        choices=sorted(ATTACK1_SCOPE_BRANCHES),
        default="per_profile",
    )
    iset.add_argument(
        "--attack2-granularity",
        choices=sorted(ATTACK2_GRANULARITY_BRANCHES),
        default="per_half_hour",
    )
    iset.add_argument(
        "--attack3-interval",
        choices=sorted(ATTACK3_INTERVAL_BRANCHES),
        default="valid_fit_addition",
    )
    iset.add_argument(
        "--attack-hour-mapping",
        choices=sorted(ATTACK_HOUR_MAPPING_BRANCHES),
        default="two_slots_per_hour",
    )
    iset.add_argument(
        "--meter-population",
        choices=sorted(METER_POPULATION_BRANCHES),
        default="all_4225",
    )
    iset.add_argument(
        "--iset-day",
        choices=sorted(ISET_DAY_BRANCHES),
        default="complete_1_48",
    )
    iset.add_argument(
        "--split-unit",
        choices=sorted(ISET_SPLIT_UNIT_BRANCHES),
        default="customer_disjoint",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    runtime_branch = None
    if args.branch_id:
        runtime_branch = load_runtime_branch(
            args.branch_id,
            manifest=args.branch_manifest,
        )
        assert_branch_scope(runtime_branch, dataset=args.dataset)
        for key, value in runtime_branch["preparation"].items():
            setattr(args, key, value)
        default_output = DEFAULT_CACHE_ROOT / f"{args.dataset}-paper-literal"
        if args.output == default_output:
            args.output = (
                DEFAULT_CACHE_ROOT
                / "branch-caches"
                / runtime_branch["preparation_id"]
            )
    args.runtime_branch = runtime_branch
    started = time.perf_counter()
    result = prepare_sgcc(args) if args.dataset == "sgcc" else prepare_iset(args)
    if runtime_branch is not None:
        result["branch"] = {
            "branch_id": runtime_branch["branch_id"],
            "track": runtime_branch["track"],
            "family": runtime_branch["family"],
            "preparation_id": runtime_branch["preparation_id"],
            "preparation": runtime_branch["preparation"],
        }
    result["total_seconds"] = time.perf_counter() - started
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
