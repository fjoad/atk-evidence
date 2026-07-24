"""Enumerate and budget the frozen Paper 1 interpretation lattice.

This is deliberately a small planning utility. It does not prepare data,
train models, submit jobs, or inspect outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import tomllib
from pathlib import Path
from typing import Any, Iterable


IncompatiblePair = tuple[str, str, str, str]


def _pair_key(
    left_name: str,
    left_value: str,
    right_name: str,
    right_value: str,
) -> tuple[str, str, str, str]:
    if left_name < right_name:
        return left_name, left_value, right_name, right_value
    return right_name, right_value, left_name, left_value


def required_pairs(
    names: list[str],
    options: dict[str, list[str]],
    incompatible: set[IncompatiblePair] | None = None,
) -> set[tuple[str, str, str, str]]:
    forbidden = incompatible or set()
    pairs: set[tuple[str, str, str, str]] = set()
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            for left_value, right_value in itertools.product(
                options[left_name], options[right_name]
            ):
                pair = _pair_key(
                    left_name,
                    left_value,
                    right_name,
                    right_value,
                )
                if pair not in forbidden and _partial_can_complete(
                    {
                        pair[0]: pair[1],
                        pair[2]: pair[3],
                    },
                    names,
                    options,
                    forbidden,
                ):
                    pairs.add(pair)
    return pairs


def covered_pairs(row: dict[str, str]) -> set[tuple[str, str, str, str]]:
    pairs: set[tuple[str, str, str, str]] = set()
    names = sorted(row)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            pairs.add(
                _pair_key(
                    left_name,
                    row[left_name],
                    right_name,
                    row[right_name],
                )
            )
    return pairs


def _row_is_allowed(
    row: dict[str, str],
    incompatible: set[IncompatiblePair],
) -> bool:
    return not bool(covered_pairs(row) & incompatible)


def _choice_is_allowed(
    row: dict[str, str],
    name: str,
    value: str,
    incompatible: set[IncompatiblePair],
) -> bool:
    return all(
        _pair_key(old_name, old_value, name, value) not in incompatible
        for old_name, old_value in row.items()
        if old_name != name
    )


def _partial_can_complete(
    partial: dict[str, str],
    names: list[str],
    options: dict[str, list[str]],
    incompatible: set[IncompatiblePair],
) -> bool:
    constrained_names = {
        name
        for pair in incompatible
        for name in (pair[0], pair[2])
        if name in names
    }
    remaining = [
        name
        for name in names
        if name in constrained_names and name not in partial
    ]

    def complete(index: int, row: dict[str, str]) -> bool:
        if index == len(remaining):
            return True
        name = remaining[index]
        for value in options[name]:
            if _choice_is_allowed(row, name, value, incompatible):
                if complete(index + 1, {**row, name: value}):
                    return True
        return False

    return _row_is_allowed(partial, incompatible) and complete(0, partial)


def pairwise_cases(
    names: list[str],
    options: dict[str, list[str]],
    incompatible: set[IncompatiblePair] | None = None,
) -> list[dict[str, str]]:
    """Return deterministic all-options/all-pairs coverage configurations."""

    forbidden = incompatible or set()
    if not names:
        return [{}]
    if len(names) == 1:
        return [{names[0]: value} for value in options[names[0]]]

    required = required_pairs(names, options, forbidden)
    rows = [
        {names[0]: left, names[1]: right}
        for left, right in itertools.product(options[names[0]], options[names[1]])
        if _partial_can_complete(
            {names[0]: left, names[1]: right},
            names,
            options,
            forbidden,
        )
    ]
    if not rows:
        raise ValueError("the first two dimensions have no compatible choices")
    for new_name in names[2:]:
        previous = names[: names.index(new_name)]
        uncovered = {
            pair
            for pair in required
            if new_name in {pair[0], pair[2]}
            and ({pair[0], pair[2]} - {new_name}).issubset(previous)
        }
        for row in rows:
            candidates: list[tuple[int, str]] = []
            for new_value in options[new_name]:
                if not _choice_is_allowed(
                    row,
                    new_name,
                    new_value,
                    forbidden,
                ) or not _partial_can_complete(
                    {**row, new_name: new_value},
                    names,
                    options,
                    forbidden,
                ):
                    continue
                gained = {
                    _pair_key(old_name, row[old_name], new_name, new_value)
                    for old_name in previous
                } & uncovered
                candidates.append((len(gained), new_value))
            if not candidates:
                raise ValueError(
                    f"no compatible {new_name} choice extends row {row}"
                )
            _, chosen = max(candidates, key=lambda item: (item[0], item[1]))
            row[new_name] = chosen
            uncovered -= {
                _pair_key(old_name, row[old_name], new_name, chosen)
                for old_name in previous
            }

        while uncovered:
            uncovered_before = len(uncovered)
            seed_pair = min(uncovered)
            if seed_pair[0] == new_name:
                new_value = seed_pair[1]
            elif seed_pair[2] == new_name:
                new_value = seed_pair[3]
            else:
                raise RuntimeError("uncovered pair does not contain the new dimension")
            row = {new_name: new_value}
            for old_name in previous:
                candidates = []
                for old_value in options[old_name]:
                    if not _choice_is_allowed(
                        row,
                        old_name,
                        old_value,
                        forbidden,
                    ):
                        continue
                    pair = _pair_key(old_name, old_value, new_name, new_value)
                    candidates.append((int(pair in uncovered), old_value))
                if not candidates:
                    raise ValueError(
                        f"cannot complete a compatible row for {seed_pair}"
                    )
                _, chosen = max(candidates, key=lambda item: (item[0], item[1]))
                row[old_name] = chosen
            rows.append(row)
            uncovered -= {
                _pair_key(old_name, row[old_name], new_name, new_value)
                for old_name in previous
            }
            if len(uncovered) >= uncovered_before:
                raise RuntimeError(
                    f"pairwise completion made no progress from {seed_pair}"
                )

    observed = set().union(*(covered_pairs(row) for row in rows))
    missing = required - observed
    if missing:
        raise RuntimeError(f"pairwise generator missed {len(missing)} pairs")
    invalid_rows = [row for row in rows if not _row_is_allowed(row, forbidden)]
    if invalid_rows:
        raise RuntimeError(
            f"pairwise generator produced {len(invalid_rows)} invalid rows"
        )
    for name in names:
        seen = {row[name] for row in rows}
        if seen != set(options[name]):
            raise RuntimeError(f"option coverage failed for {name}")
    return rows


def stable_branch_id(family_id: str, choices: dict[str, str]) -> str:
    payload = json.dumps(
        {"family": family_id, "choices": choices},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{family_id}-{hashlib.sha256(payload).hexdigest()[:12]}"


def load_lattice(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported branch-lattice schema")
    dimensions = raw.get("dimensions", {})
    if not dimensions:
        raise ValueError("branch lattice contains no dimensions")
    for name, record in dimensions.items():
        values = record.get("options", [])
        if not values or len(values) != len(set(values)):
            raise ValueError(f"dimension {name} needs unique nonempty options")
    for record in raw.get("incompatible_pairs", []):
        left_name = record["left_dimension"]
        right_name = record["right_dimension"]
        if left_name == right_name:
            raise ValueError("incompatible pair dimensions must differ")
        for name, value in (
            (left_name, record["left_value"]),
            (right_name, record["right_value"]),
        ):
            if name not in dimensions:
                raise ValueError(
                    f"incompatible pair uses unknown dimension {name}"
                )
            if value not in dimensions[name]["options"]:
                raise ValueError(
                    f"incompatible pair uses unknown {name} value {value}"
                )
    family_ids: set[str] = set()
    for family in raw.get("families", []):
        family_id = family["id"]
        if family_id in family_ids:
            raise ValueError(f"duplicate family ID {family_id}")
        family_ids.add(family_id)
        unknown = set(family["dimensions"]) - set(dimensions)
        if unknown:
            raise ValueError(f"family {family_id} uses unknown dimensions {unknown}")
    required_ambiguities = set(raw.get("coverage", {}).get("required_ambiguity_ids", []))
    coverage_items = raw.get("coverage_items", {})
    if set(coverage_items) != required_ambiguities:
        missing = sorted(required_ambiguities - set(coverage_items))
        extra = sorted(set(coverage_items) - required_ambiguities)
        raise ValueError(
            f"coverage-item mismatch: missing={missing}, extra={extra}"
        )
    non_executable_ids = {
        record["id"] for record in raw.get("non_executable", [])
    }
    for ambiguity_id, references in coverage_items.items():
        if not references:
            raise ValueError(f"coverage item {ambiguity_id} has no references")
        for reference in references:
            kind, separator, target = reference.partition(":")
            if not separator or not target:
                raise ValueError(
                    f"coverage item {ambiguity_id} has invalid reference {reference}"
                )
            if kind == "dimension" and target not in dimensions:
                raise ValueError(
                    f"coverage item {ambiguity_id} uses unknown dimension {target}"
                )
            if kind == "non_executable" and target not in non_executable_ids:
                raise ValueError(
                    f"coverage item {ambiguity_id} uses unknown non-executable {target}"
                )
            if kind not in {
                "dimension",
                "fixed",
                "global",
                "locked",
                "non_executable",
                "recording",
            }:
                raise ValueError(
                    f"coverage item {ambiguity_id} uses unknown reference kind {kind}"
                )
    used_dimensions = {
        dimension
        for family in raw.get("families", [])
        for dimension in family["dimensions"]
    }
    uncovered_dimensions = set(dimensions) - used_dimensions
    if uncovered_dimensions:
        raise ValueError(
            f"dimensions are not assigned to any family: {sorted(uncovered_dimensions)}"
        )
    if not raw.get("corrected_defaults"):
        raise ValueError("corrected-control defaults are missing")
    corrected_dataset_defaults = raw.get("corrected_dataset_defaults", {})
    corrected_model_defaults = raw.get("corrected_model_defaults", {})
    for family in raw.get("families", []):
        if family["dataset"] not in corrected_dataset_defaults:
            raise ValueError(
                f"corrected dataset defaults missing for {family['dataset']}"
            )
        if family["model"] not in corrected_model_defaults:
            raise ValueError(
                f"corrected model defaults missing for {family['model']}"
            )
    target_ids = set(raw.get("reported_roc_targets", {}))
    if target_ids != family_ids:
        missing = sorted(family_ids - target_ids)
        extra = sorted(target_ids - family_ids)
        raise ValueError(
            f"reported ROC target mismatch: missing={missing}, extra={extra}"
        )
    return raw


def _hours(attempts: int, minutes: float, workers: int) -> float:
    return attempts * minutes * max(1, workers) / 60.0


def enumerate_lattice(raw: dict[str, Any]) -> dict[str, Any]:
    dimensions = {
        name: [str(value) for value in record["options"]]
        for name, record in raw["dimensions"].items()
    }
    screen_seeds = int(raw["screen_seed_count"])
    search_evaluations = int(raw["paper_sequential_search_evaluations"])
    search_seeds = int(raw["search_seed_count"])
    confirm_seeds = int(raw["confirm_seed_count"])
    uncertainty = float(raw["budget_uncertainty_multiplier"])
    anchor_defaults = {
        name: str(value) for name, value in raw["anchor_defaults"].items()
    }
    anchor_overrides = raw.get("anchor_overrides", {})
    incompatible = {
        _pair_key(
            record["left_dimension"],
            str(record["left_value"]),
            record["right_dimension"],
            str(record["right_value"]),
        )
        for record in raw.get("incompatible_pairs", [])
    }

    family_summaries: list[dict[str, Any]] = []
    all_branches: list[dict[str, Any]] = []
    corrected_branches: list[dict[str, Any]] = []
    anchor_branch_ids: list[str] = []
    totals = {
        "semantic_cases": 0,
        "screen_attempts": 0,
        "screen_gpu_hours": 0.0,
        "screen_gpu_job_hours": 0.0,
        "screen_cpu_hours": 0.0,
        "all_promote_attempts": 0,
        "all_promote_gpu_hours": 0.0,
        "all_promote_gpu_job_hours": 0.0,
        "all_promote_cpu_hours": 0.0,
        "corrected_attempts": 0,
        "corrected_gpu_hours": 0.0,
        "corrected_gpu_job_hours": 0.0,
        "corrected_cpu_hours": 0.0,
    }
    for family in raw["families"]:
        names = list(family["dimensions"])
        family_incompatible = {
            pair
            for pair in incompatible
            if pair[0] in names and pair[2] in names
        }
        cases = pairwise_cases(
            names,
            dimensions,
            family_incompatible,
        )
        family_overrides = {
            name: str(value)
            for name, value in anchor_overrides.get(family["id"], {}).items()
        }
        anchor = {
            name: family_overrides.get(name, anchor_defaults[name]) for name in names
        }
        for name, value in anchor.items():
            if value not in dimensions[name]:
                raise ValueError(
                    f"anchor {family['id']} selects invalid {name}={value}"
                )
        if not _row_is_allowed(anchor, family_incompatible):
            raise ValueError(
                f"anchor {family['id']} violates an incompatible pair"
            )
        if anchor not in cases:
            cases.append(anchor)
        anchor_id = stable_branch_id(family["id"], anchor)
        anchor_branch_ids.append(anchor_id)
        branches = [
            {
                "branch_id": stable_branch_id(family["id"], choices),
                "family": family["id"],
                "dataset": str(family["dataset"]),
                "model": str(family["model"]),
                "tables": list(family["tables"]),
                "dimensions": list(family["dimensions"]),
                "track": (
                    "P_anchor"
                    if stable_branch_id(family["id"], choices) == anchor_id
                    else "I"
                ),
                "choices": choices,
            }
            for choices in cases
        ]
        all_branches.extend(branches)
        corrected_choices = {
            **{
                name: str(value)
                for name, value in raw["corrected_defaults"].items()
            },
            **{
                name: str(value)
                for name, value in raw["corrected_dataset_defaults"][
                    family["dataset"]
                ].items()
            },
            **{
                name: str(value)
                for name, value in raw["corrected_model_defaults"][
                    family["model"]
                ].items()
            },
        }
        corrected_branches.append(
            {
                "branch_id": stable_branch_id(
                    f"corrected_{family['id']}", corrected_choices
                ),
                "family": family["id"],
                "dataset": str(family["dataset"]),
                "model": str(family["model"]),
                "tables": list(family["tables"]),
                "dimensions": list(family["dimensions"]),
                "track": "C",
                "choices": corrected_choices,
            }
        )
        semantic_cases = len(branches)
        screen_attempts = semantic_cases * screen_seeds
        full_attempts_per_case = (
            search_evaluations * search_seeds + confirm_seeds
        ) * int(family["full_table_multiplier"])
        all_promote_attempts = semantic_cases * full_attempts_per_case
        workers = int(family["gpus"])
        screen_hours = _hours(
            screen_attempts, float(family["screen_minutes"]), workers
        )
        screen_job_hours = (
            screen_attempts * float(family["screen_minutes"]) / 60.0
        )
        full_hours = _hours(
            all_promote_attempts, float(family["full_minutes"]), workers
        )
        full_job_hours = (
            all_promote_attempts * float(family["full_minutes"]) / 60.0
        )
        corrected_multiplier = int(family.get("corrected_table_multiplier", 1))
        corrected_family_attempts = (
            int(raw["corrected_nested_search_evaluations"])
            * int(raw["search_seed_count"])
            + int(raw["confirm_seed_count"])
        ) * corrected_multiplier
        corrected_hours = _hours(
            corrected_family_attempts,
            float(family["full_minutes"]),
            workers,
        )
        corrected_job_hours = (
            corrected_family_attempts * float(family["full_minutes"]) / 60.0
        )
        resource = "gpu" if workers else "cpu"
        family_summary = {
            "family": family["id"],
            "dataset": family["dataset"],
            "model": family["model"],
            "tables": family["tables"],
            "dimensions": names,
            "dimension_count": len(names),
            "raw_cartesian_cases": int(
                __import__("math").prod(len(dimensions[name]) for name in names)
            ),
            "pairwise_semantic_cases": semantic_cases,
            "pairwise_coverage_verified": True,
            "incompatible_pair_count": len(family_incompatible),
            "printed_anchor_branch_id": anchor_id,
            "screen_attempts": screen_attempts,
            "all_promote_attempts": all_promote_attempts,
            "corrected_attempts": corrected_family_attempts,
            "resource": resource,
            "workers_per_attempt": workers if workers else 1,
            "screen_resource_hours": screen_hours,
            "all_promote_resource_hours": full_hours,
            "corrected_resource_hours": corrected_hours,
        }
        family_summaries.append(family_summary)
        totals["semantic_cases"] += semantic_cases
        totals["screen_attempts"] += screen_attempts
        totals["all_promote_attempts"] += all_promote_attempts
        totals["corrected_attempts"] += corrected_family_attempts
        totals[f"screen_{resource}_hours"] += screen_hours
        totals[f"all_promote_{resource}_hours"] += full_hours
        totals[f"corrected_{resource}_hours"] += corrected_hours
        if resource == "gpu":
            totals["screen_gpu_job_hours"] += screen_job_hours
            totals["all_promote_gpu_job_hours"] += full_job_hours
            totals["corrected_gpu_job_hours"] += corrected_job_hours

    corrected_cases = len(raw["families"])
    totals = {
        key: round(value, 3) if isinstance(value, float) else value
        for key, value in totals.items()
    }
    promotion_targets = {}
    promotion_margin = float(raw["promotion"]["reported_auc_margin"])
    borderline_margin = float(raw["promotion"]["borderline_margin"])
    for name, target in raw["reported_roc_targets"].items():
        dr = float(target["dr"])
        fa = float(target["fa"])
        point_lower_bound = (dr + (1.0 - fa)) / 2.0
        promotion_targets[name] = {
            "reported_dr": dr,
            "reported_fa": fa,
            "reported_point_auc_lower_bound": round(point_lower_bound, 6),
            "screen_promotion_floor": round(
                max(0.5, point_lower_bound - promotion_margin), 6
            ),
            "borderline_rerun_floor": round(
                max(
                    0.5,
                    point_lower_bound - promotion_margin - borderline_margin,
                ),
                6,
            ),
        }
    return {
        "schema_version": 1,
        "paper_sha256": raw["paper_sha256"],
        "coverage": {
            "criterion": "every option and every allowed option pair per family",
            "higher_order_policy": (
                "coupled blocks are represented in family dimensions; arbitrary "
                "higher-order Cartesian products are excluded with rationale"
            ),
            "verified": True,
            "required_ambiguity_ids": raw["coverage"]["required_ambiguity_ids"],
            "items": raw["coverage_items"],
            "incompatible_pairs": raw.get("incompatible_pairs", []),
        },
        "screen": {
            "data_fraction": raw["screen_data_fraction"],
            "epochs": raw["screen_epochs"],
            "seed_count": screen_seeds,
        },
        "full_treatment": {
            "sequential_search_evaluations": search_evaluations,
            "search_seed_count": search_seeds,
            "confirm_seed_count": confirm_seeds,
            "budget_is_all_branches_promote_upper_bound": True,
        },
        "promotion": {
            **raw["promotion"],
            "targets": promotion_targets,
        },
        "printed_anchor_branch_ids": anchor_branch_ids,
        "corrected_controls": {
            "semantic_cases": corrected_cases,
            "search_plus_confirmation_attempts": totals["corrected_attempts"],
            "reported_separately": True,
            "branches": corrected_branches,
        },
        "budget": {
            "point_estimate": totals,
            "uncertainty_multiplier": uncertainty,
            "conservative_upper": {
                key: round(value * uncertainty, 3)
                if isinstance(value, float) and key.endswith("_hours")
                else value
                for key, value in totals.items()
            },
            "basis": (
                "existing Paper 1 probes/results where available and conservative "
                "per-family estimates otherwise; update estimates, never outcomes, "
                "after bounded runtime calibration"
            ),
            "scheduler_projection": {
                "max_simultaneous_jobs": 3,
                "screen_gpu_job_hours_serial": totals["screen_gpu_job_hours"],
                "screen_gpu_job_hours_ideal_three_way": round(
                    totals["screen_gpu_job_hours"] / 3.0, 3
                ),
                "queue_time_included": False,
            },
        },
        "families": family_summaries,
        "branches": all_branches,
        "non_executable": raw.get("non_executable", []),
        "exclusions": raw.get("exclusions", []),
        "hyperparameter_envelope": raw["hyperparameter_envelope"],
    }


def _branch_inventory(branches: list[dict[str, Any]]) -> dict[str, Any]:
    payload = json.dumps(
        branches,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "count": len(branches),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def resolve_branch(
    manifest: str | Path,
    branch_id: str,
) -> dict[str, Any]:
    """Resolve one stable branch ID to its complete execution choices."""

    raw = load_lattice(Path(manifest))
    expanded = enumerate_lattice(raw)
    family_by_id = {
        str(family["id"]): family for family in raw["families"]
    }
    candidates = [
        *expanded["branches"],
        *expanded["corrected_controls"]["branches"],
    ]
    matches = [
        branch for branch in candidates if branch["branch_id"] == branch_id
    ]
    if not matches:
        raise KeyError(f"unknown Paper 1 branch ID {branch_id!r}")
    if len(matches) != 1:
        raise RuntimeError(f"branch ID collision for {branch_id!r}")
    branch = dict(matches[0])
    family_id = str(branch["family"])
    family = family_by_id[family_id]
    return {
        **branch,
        "dataset": str(family["dataset"]),
        "model": str(family["model"]),
        "tables": list(family["tables"]),
        "dimensions": list(family["dimensions"]),
        "manifest": str(Path(manifest).resolve()),
    }


def storage_summary(
    summary: dict[str, Any], *, include_branches: bool = False
) -> dict[str, Any]:
    stored = dict(summary)
    paper_branches = stored.pop("branches")
    corrected_controls = dict(stored["corrected_controls"])
    corrected_branches = corrected_controls.pop("branches")
    stored["corrected_controls"] = corrected_controls
    stored["branch_inventory"] = {
        "paper_consistent": _branch_inventory(paper_branches),
        "corrected": _branch_inventory(corrected_branches),
    }
    if include_branches:
        stored["branches"] = paper_branches
        stored["corrected_controls"]["branches"] = corrected_branches
    return stored


def write_summary(
    summary: dict[str, Any],
    output: Path,
    *,
    include_branches: bool = False,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            storage_summary(summary, include_branches=include_branches),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-branches",
        action="store_true",
        help="include all expanded branch records instead of only count/hash inventories",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = enumerate_lattice(load_lattice(args.config))
    write_summary(
        summary,
        args.output,
        include_branches=args.include_branches,
    )
    compact = {
        "semantic_cases": summary["budget"]["point_estimate"]["semantic_cases"],
        "screen_attempts": summary["budget"]["point_estimate"]["screen_attempts"],
        "screen_gpu_hours": summary["budget"]["point_estimate"]["screen_gpu_hours"],
        "screen_cpu_hours": summary["budget"]["point_estimate"]["screen_cpu_hours"],
        "all_promote_gpu_hours": summary["budget"]["point_estimate"]["all_promote_gpu_hours"],
        "all_promote_cpu_hours": summary["budget"]["point_estimate"]["all_promote_cpu_hours"],
        "corrected_attempts": summary["budget"]["point_estimate"]["corrected_attempts"],
        "corrected_gpu_hours": summary["budget"]["point_estimate"]["corrected_gpu_hours"],
        "corrected_cpu_hours": summary["budget"]["point_estimate"]["corrected_cpu_hours"],
        "output": str(args.output),
    }
    print(json.dumps(compact, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
