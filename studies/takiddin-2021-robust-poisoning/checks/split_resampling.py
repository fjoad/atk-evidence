#!/usr/bin/env python3
"""Four matched RF fits controlling resampling placement and source-day overlap."""

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import sklearn
import imblearn
from sklearn.preprocessing import StandardScaler

STUDY = Path(__file__).resolve().parents[1]
REPRO = STUDY / "reproduction"
sys.path.insert(0, str(REPRO))
from prepare_data import Rows, join, balance, poison, rng, library_seed, save_arrays
from run_experiment import fit_one, require_compute, EXPECTED_METADATA
from analyze_results import analyze_scores, audit_result, digest, METRICS

SEED = 20260920
BASELINE_HASHES = {
    "p00": "a47c4f85e0e66891b6d2ad2103bf3f5498ba28b34a99eec94c686e288fec125a",
    "p30": "71708e92acc5c1cdc00681dd93107a436f7c55718b23b363b52de6af0d438803",
}


def ids_hash(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def load_rows(directory):
    expected = EXPECTED_METADATA["generalized-two-class-p00"]
    if digest(directory / "metadata.json") != expected:
        raise ValueError("Original preparation metadata changed")
    metadata = json.loads((directory / "metadata.json").read_text())
    arrays = {}
    for name, info in metadata["files"].items():
        path = directory / name
        if digest(path) != info["sha256"]:
            raise ValueError(f"Preparation array hash mismatch: {name}")
        value = np.load(path, allow_pickle=False)
        if list(value.shape) != info["shape"] or str(value.dtype) != info["dtype"] or not np.isfinite(value).all():
            raise ValueError(f"Invalid preparation array: {name}")
        arrays[name.removesuffix(".npy")] = value
    result = []
    for prefix in ("train", "test"):
        rows = Rows(**{f.name: arrays[prefix + "_" + ("raw_x" if f.name == "x" else f.name)]
                       for f in fields(Rows)})
        if not np.array_equal(rows.true_y, rows.observed_y):
            raise ValueError("Original p00 labels must be clean")
        result.append(rows)
    return tuple(result)


def design(original_train, original_test, seed=SEED):
    """Outcome-blind grouping; recover original profiles without regenerating attacks."""
    train = original_train.take(~original_train.synthetic)
    test = original_test.take(~original_test.synthetic)
    pool = join(train, test)
    pool = pool.take(np.argsort(pool.uid, kind="stable"))
    sources, counts = np.unique(pool.source_id, return_counts=True)
    if (counts != 7).any() or len(np.unique(pool.uid)) != len(pool):
        raise ValueError("Need seven unique original versions of every source day")
    if not np.array_equal(pool.uid, pool.source_id * 7 + pool.attack):
        raise ValueError("Original row identity encoding mismatch")
    if not np.array_equal(pool.attack.reshape(-1, 7), np.tile(np.arange(7), (len(sources), 1))):
        raise ValueError("Incomplete original attack family")
    if not np.array_equal(pool.true_y, pool.attack > 0):
        raise ValueError("Original labels disagree with attack identities")
    shuffled = rng(seed, 501).permutation(sources)
    train_days = shuffled[:2 * len(sources) // 3]
    heldout_days = shuffled[2 * len(sources) // 3:]
    group_train = pool.take(np.isin(pool.source_id, train_days))
    common_mask = np.isin(test.source_id, heldout_days)
    common = test.take(common_mask)
    if set(np.unique(common.true_y)) != {0, 1}:
        raise ValueError("Common evaluation must contain both classes; do not choose another split")
    if np.intersect1d(group_train.source_id, common.source_id).size:
        raise AssertionError("Held-out source day appears in grouped training")
    info = {
        "root_seed": seed, "grouping_rng_role": 501,
        "source_days": len(sources), "source_customers": len(np.unique(pool.meter)),
        "original_pool_rows": len(pool), "row_training_originals": len(train),
        "day_training_originals": len(group_train), "day_training_groups": len(train_days),
        "heldout_day_groups": len(heldout_days), "original_evaluation_rows": len(test),
        "common_evaluation_rows": len(common),
        "common_evaluation_class_counts": np.bincount(common.true_y, minlength=2).tolist(),
        "common_evaluation_source_days": len(np.unique(common.source_id)),
        "common_evaluation_customers": len(np.unique(common.meter)),
        "common_uid_sha256": ids_hash(common.uid),
        "original_test_uid_sha256": ids_hash(test.uid),
        "day_training_uid_sha256": ids_hash(group_train.uid),
    }
    return {"B": train, "C": group_train}, test, common, np.unique(pool.meter), info


def verify_lineage(train, evaluation, *, disjoint_days):
    original = ~train.synthetic
    parents = np.concatenate([train.parent_a[train.synthetic], train.parent_b[train.synthetic]])
    if not np.isin(parents, train.uid[original & (train.true_y == 0)]).all():
        raise ValueError("Synthetic parent is not an original benign training row")
    if np.intersect1d(train.uid[original], evaluation.uid).size or np.isin(parents, evaluation.uid).any():
        raise ValueError("An evaluation row enters training or synthetic ancestry")
    shared_days = np.intersect1d(train.source_id[original], evaluation.source_id)
    if disjoint_days and len(shared_days):
        raise ValueError("Source-day overlap in the grouped control")
    return {"original_row_overlap": 0, "synthetic_parent_crossings": 0,
            "shared_source_days": len(shared_days),
            "all_synthetic_parents_in_training": True}


def inputs_for(train, evaluation, population, rate, *, grouped=False):
    trained, poisoning = poison(train, {}, mode="two-class", scope="generalized",
                                rate=rate, population=population, seed=SEED)
    lineage = verify_lineage(trained, evaluation, disjoint_days=grouped)
    scaler = StandardScaler().fit(trained.x.astype(np.float64))
    arrays = {"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_}
    for prefix, rows in (("train", trained), ("test", evaluation)):
        for f in fields(Rows):
            arrays[prefix + "_" + ("raw_x" if f.name == "x" else f.name)] = getattr(rows, f.name)
        arrays[prefix + "_x"] = scaler.transform(rows.x).astype(np.float32)
    record = {"scope": "controlled pilot", "rate": rate, "training_rows": len(trained),
              "test_rows": len(evaluation), "poisoning": poisoning, "lineage": lineage,
              "train_true_counts": np.bincount(trained.true_y, minlength=2).tolist(),
              "train_observed_counts": np.bincount(trained.observed_y, minlength=2).tolist(),
              "test_true_counts": np.bincount(evaluation.true_y, minlength=2).tolist(),
              "test_uid_sha256": ids_hash(evaluation.uid)}
    return arrays, record


def read_scores(path):
    with np.load(path / "predictions.npz", allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def select_scores(arrays, requested_uids):
    lookup = {int(uid): i for i, uid in enumerate(arrays["uid"])}
    if len(lookup) != len(arrays["uid"]):
        raise ValueError("Saved scores contain duplicate row identities")
    index = np.array([lookup[int(uid)] for uid in requested_uids], dtype=np.int64)
    return {name: values[index] for name, values in arrays.items()}


def difference(new, old):
    return {key: new["primary"][key] - old["primary"][key]
            if new["primary"][key] is not None and old["primary"][key] is not None else None
            for key in METRICS}


def comparisons(output, baseline, common_uids, original_test_uids):
    common_report, original_report, deltas = {}, {}, {}
    for level in ("p00", "p30"):
        ref = baseline / level
        if digest(ref / "result.json") != BASELINE_HASHES[level]:
            raise ValueError("Baseline result changed")
        audit_result(ref)
        reference = json.loads((ref / "result.json").read_text())
        all_scores = {"A": read_scores(ref)}
        priors = {"A": reference["training_prior"]}
        for arm in ("B", "C"):
            path = output / f"{arm}-{level}"
            audit_result(path)
            result = json.loads((path / "result.json").read_text())
            all_scores[arm] = read_scores(path)
            priors[arm] = result["training_prior"]
        selected = {arm: select_scores(scores, common_uids) for arm, scores in all_scores.items()}
        for arm in ("B", "C"):
            for key in ("uid", "labels", "attack", "synthetic", "negative_daily_mean"):
                np.testing.assert_array_equal(selected["A"][key], selected[arm][key])
        common_report[level] = {arm: analyze_scores(scores, priors[arm]) for arm, scores in selected.items()}
        original_report[level] = {
            arm: analyze_scores(select_scores(all_scores[arm], original_test_uids), priors[arm])
            for arm in ("A", "B")
        }
        deltas[level] = {
            "common_B_minus_A_points": difference(common_report[level]["B"], common_report[level]["A"]),
            "common_C_minus_B_points": difference(common_report[level]["C"], common_report[level]["B"]),
            "all_original_B_minus_A_points": difference(original_report[level]["B"], original_report[level]["A"]),
        }
    return {"common_evaluation": common_report, "all_original_evaluation": original_report, "deltas": deltas}


def run(preparation, baseline, output):
    require_compute()
    if sklearn.__version__ != "1.9.0":
        raise RuntimeError("This control is frozen to scikit-learn 1.9.0")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    summary = {
        "status": "started", "scope": "controlled pilot; not full paper reproduction",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": os.environ.get("EXPECTED_COMMIT"), "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "runs": [], "baseline_retrained": False,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "sklearn": sklearn.__version__, "imblearn": imblearn.__version__},
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    source_hashes = {str(path.relative_to(STUDY)): digest(path) for path in (
        Path(__file__), STUDY / "SPLIT_RESAMPLING_CHECK.md", REPRO / "prepare_data.py",
        REPRO / "models.py", REPRO / "run_experiment.py", REPRO / "analyze_results.py",
        STUDY / "reported/table_3.csv",
    )}
    try:
        for level, expected in BASELINE_HASHES.items():
            if digest(baseline / level / "result.json") != expected:
                raise ValueError("Reference baseline result changed")
            audit_result(baseline / level)
        saved_train, saved_test = load_rows(preparation)
        trains, original_test, common, population, info = design(saved_train, saved_test)
        if info["source_days"] != 560 or info["source_customers"] != 20:
            raise ValueError("Control is restricted to the frozen 20-customer pilot")
        summary["design"] = info
        np.save(output / "common_uids.npy", common.uid, allow_pickle=False)
        np.save(output / "original_test_uids.npy", original_test.uid, allow_pickle=False)
        for arm, originals in trains.items():
            evaluation = original_test if arm == "B" else common
            balanced, resampling = balance(originals, seed=library_seed(SEED, 202), neighbors=5)
            for level, rate in (("p00", 0.), ("p30", .30)):
                case = f"{arm}-{level}"
                target = output / case
                target.mkdir()
                result_path = target / "result.json"
                record = {"status": "started", "case": case, "model": "random_forest",
                          "scope": f"controlled pilot arm {arm}; not Table III reproduction",
                          "code_commit": summary["code_commit"], "poisoning_rate": rate, "seed": SEED,
                          "source_sha256": source_hashes, "resampling": resampling}
                result_path.write_text(json.dumps(record, indent=2) + "\n")
                try:
                    arrays, metadata = inputs_for(balanced, evaluation, population, rate, grouped=arm == "C")
                    record["preparation"] = metadata
                    save_arrays(target / "inputs", arrays, metadata)
                    record["input_metadata_sha256"] = digest(target / "inputs/metadata.json")
                    record.update(fit_one(arrays, target, seed=SEED, workers=4))
                    record["status"] = "complete"
                except Exception as exc:
                    record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                    raise
                finally:
                    result_path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
                summary["runs"].append({"case": case, "preparation": metadata,
                                        "fit_seconds": record["timing_seconds"]["fit"]})
                print(json.dumps({"case": case, "training_rows": len(arrays["train_x"]),
                                  "test_rows": len(arrays["test_x"]), "primary": record["analysis"]["primary"]}), flush=True)
        summary.update(comparisons(output, baseline, common.uid, original_test.uid))
        summary["identity_sha256"] = {name: digest(output / name)
                                      for name in ("common_uids.npy", "original_test_uids.npy")}
        summary["status"] = "complete"
    except Exception as exc:
        summary.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - started
        summary["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return summary


def audit(output, baseline):
    summary = json.loads((output / "summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("Control did not finish")
    for name, expected in summary["identity_sha256"].items():
        if digest(output / name) != expected:
            raise ValueError("Evaluation identity file changed")
    common = np.load(output / "common_uids.npy", allow_pickle=False)
    original_test = np.load(output / "original_test_uids.npy", allow_pickle=False)
    checked_files = 0
    matched_inputs = []
    for run_record in summary["runs"]:
        path = output / run_record["case"]
        result = json.loads((path / "result.json").read_text())
        if digest(path / "inputs/metadata.json") != result["input_metadata_sha256"]:
            raise ValueError("Controlled input metadata changed")
        metadata = json.loads((path / "inputs/metadata.json").read_text())
        arrays = {}
        for name, expected in metadata["files"].items():
            value = np.load(path / "inputs" / name, allow_pickle=False)
            if (digest(path / "inputs" / name) != expected["sha256"]
                    or list(value.shape) != expected["shape"] or str(value.dtype) != expected["dtype"]
                    or not np.isfinite(value).all()):
                raise ValueError("Controlled input array changed")
            arrays[name.removesuffix(".npy")] = value
            checked_files += 1
        rows = [Rows(**{f.name: arrays[prefix + "_" + ("raw_x" if f.name == "x" else f.name)]
                       for f in fields(Rows)}) for prefix in ("train", "test")]
        observed = verify_lineage(*rows, disjoint_days=run_record["case"].startswith("C"))
        if observed != metadata["lineage"]:
            raise ValueError("Lineage report differs from saved identities")
        np.testing.assert_array_equal(arrays["test_true_y"], arrays["test_observed_y"])
        np.testing.assert_allclose(arrays["train_raw_x"].mean(axis=0, dtype=np.float64),
                                   arrays["scaler_mean"], rtol=1e-12, atol=1e-12)
        for part in ("train", "test"):
            reconstructed = (arrays[part + "_raw_x"].astype(np.float64)
                             - arrays["scaler_mean"]) / arrays["scaler_scale"]
            np.testing.assert_allclose(reconstructed, arrays[part + "_x"], rtol=5e-7, atol=5e-7)
        matched_inputs.append(select_scores({
            "uid": arrays["test_uid"], "raw_x": arrays["test_raw_x"], "labels": arrays["test_true_y"],
        }, common))
    for matched in matched_inputs[1:]:
        for key in matched:
            np.testing.assert_array_equal(matched_inputs[0][key], matched[key])
    recomputed = comparisons(output, baseline, common, original_test)
    for name, actual in recomputed.items():
        if actual != summary[name]:
            raise ValueError(f"Control summary does not match scores: {name}")
    return {"status": "verified", "slurm_job_id": summary["slurm_job_id"],
            "summary_sha256": digest(output / "summary.json"),
            "new_fits": len(summary["runs"]), "input_arrays_checked": checked_files,
            "common_evaluation_rows": len(common), "all_arms_have_identical_evaluation": True,
            "synthetic_ancestry_confined_to_training": True, "C_source_days_disjoint": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(audit(args.output, args.baseline), indent=2, sort_keys=True))
    else:
        if args.preparation is None:
            parser.error("--preparation is required for execution")
        run(args.preparation, args.baseline, args.output)
        report = audit(args.output, args.baseline)
        (args.output / "artifact_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, sort_keys=True))
