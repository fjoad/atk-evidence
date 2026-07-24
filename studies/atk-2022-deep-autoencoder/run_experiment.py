#!/usr/bin/env python3
"""Run Paper 1 paper-literal SGCC or exact-ISET model/seed cells."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[1]
sys.path.insert(0, str(STUDY / "src"))

from paper_literal_runner import main as run_table_2  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models",
        nargs="*",
        default=["all"],
        help="model names or 'all'; ignored when --branch-id is supplied",
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dataset", choices=("sgcc", "iset"), default="sgcc")
    parser.add_argument("--table", type=int)
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=("half", "three_quarter", "full"),
        default=["half", "three_quarter", "full"],
        help="Table IV training subsets",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="verified SGCC data.csv (SGCC only)",
    )
    parser.add_argument(
        "--cache-prefix",
        type=Path,
        help="prepared ISET cache prefix without .npz/.json suffix",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "data/derived/atk-2022-deep-autoencoder/runs",
    )
    parser.add_argument(
        "--config",
        type=Path,
    )
    parser.add_argument("--branch-id")
    parser.add_argument(
        "--branch-manifest",
        type=Path,
        default=STUDY / "config/branch_lattice.toml",
    )
    parser.add_argument("--transferred-thresholds", type=Path)
    parser.add_argument(
        "--representation",
        choices=(
            "full_1034",
            "windows_48_nonoverlap",
            "windows_48_rolling",
            "first_48",
            "last_48",
            "binned_mean_48",
        ),
        default="full_1034",
    )
    parser.add_argument(
        "--missing",
        choices=(
            "drop_incomplete",
            "zero_fill",
            "interpolate_edge_median",
            "customer_mean",
        ),
        default="interpolate_edge_median",
    )
    parser.add_argument(
        "--split-unit",
        choices=("customer_disjoint", "row_random"),
        default="customer_disjoint",
    )
    parser.add_argument(
        "--table-v-identity",
        choices=(
            "common_model_common_benign",
            "retrain_per_attack",
            "resplit_per_attack",
            "retrain_and_resplit",
        ),
        default="common_model_common_benign",
    )
    parser.add_argument(
        "--table-v-size",
        choices=("full_heldout", "seeded_3000"),
        default="seeded_3000",
    )
    args = parser.parse_args()
    command = "preflight" if args.preflight else "run"
    if args.dataset == "iset":
        from paper_literal_iset_runner import main as run_iset

        table = 3 if args.table is None else args.table
        if args.cache_prefix is not None:
            cache_prefix = args.cache_prefix
        elif args.branch_id is not None:
            from branch_runtime import load_runtime_branch

            runtime_branch = load_runtime_branch(
                args.branch_id,
                manifest=args.branch_manifest,
            )
            cache_prefix = (
                REPO
                / "data/derived/atk-2022-deep-autoencoder/branch-caches"
                / runtime_branch["preparation_id"]
            )
        else:
            cache_prefix = (
                REPO
                / "data/derived/atk-2022-deep-autoencoder/iset-paper-literal"
            )
        config = args.config or STUDY / "config/exploratory_iset.toml"
        forwarded = [
            command,
            "--cache-prefix",
            str(cache_prefix),
            "--output",
            str(args.output),
            "--config",
            str(config),
            "--table",
            str(table),
            "--models",
            *args.models,
            "--sizes",
            *args.sizes,
            "--table-v-identity",
            args.table_v_identity,
            "--table-v-size",
            args.table_v_size,
        ]
        if args.seeds:
            forwarded.extend(["--seeds", *(str(seed) for seed in args.seeds)])
        if args.branch_id:
            forwarded.extend(
                [
                    "--branch-id",
                    args.branch_id,
                    "--branch-manifest",
                    str(args.branch_manifest),
                ]
            )
        if args.transferred_thresholds:
            forwarded.extend(
                [
                    "--transferred-thresholds",
                    str(args.transferred_thresholds),
                ]
            )
        if args.force and not args.preflight:
            forwarded.append("--force")
        return run_iset(forwarded)

    table = 2 if args.table is None else args.table
    data = args.data or REPO / "data/raw/sgcc-verified/data.csv"
    config = args.config or STUDY / "config/exploratory_reproduction.toml"
    forwarded = [
        command,
        "--data",
        str(data),
        "--output",
        str(args.output),
        "--config",
        str(config),
        "--models",
        *args.models,
        "--table",
        str(table),
        "--dataset",
        "sgcc",
        "--representation",
        args.representation,
        "--missing",
        args.missing,
        "--split-unit",
        args.split_unit,
    ]
    if args.seeds:
        forwarded.extend(["--seeds", *(str(seed) for seed in args.seeds)])
    if args.branch_id:
        forwarded.extend(
            [
                "--branch-id",
                args.branch_id,
                "--branch-manifest",
                str(args.branch_manifest),
            ]
        )
    if args.transferred_thresholds:
        forwarded.extend(
            [
                "--transferred-thresholds",
                str(args.transferred_thresholds),
            ]
        )
    if args.force and not args.preflight:
        forwarded.append("--force")
    return run_table_2(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
