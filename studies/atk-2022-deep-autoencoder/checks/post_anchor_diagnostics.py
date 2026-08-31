"""Frozen no-training output-domain and useful-information checks.

See ../POST_ANCHOR_DIAGNOSTICS.md. Scientific execution requires Slurm.
Inputs are read-only; each invocation creates a new output directory.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


RESULT_SHA = "ae07b42ef6c84242ca9b39db8b8828694d6d4df6859abdee090fc0a613a69154"
SCIENCE_COMMIT = "a88d17477ad96b01ffa44a50d8ce051dd8d2b5ca"
THRESHOLD = 0.58
ANALYSIS_SEED = 20260831
BOOTSTRAPS = 2000


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def simplex_bounds(x):
    """Exact real-arithmetic extrema, evaluated in float64; no padding here."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError("Expected a finite row matrix")
    d = x.shape[1]
    ordered = np.sort(x, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1
    ranks = np.arange(1, d + 1)
    rho = np.sum(ordered > cumulative / ranks, axis=1) - 1
    theta = cumulative[np.arange(len(x)), rho] / (rho + 1)
    projection = np.maximum(x - theta[:, None], 0)
    lower = np.mean((x - projection) ** 2, axis=1)
    energy = np.mean(x * x, axis=1)
    upper = energy + (1 - 2 * x.min(axis=1)) / d
    uniform = np.mean((x - 1 / d) ** 2, axis=1)
    return lower, upper, uniform, energy


def metrics(y, scores):
    y, scores = np.asarray(y), np.asarray(scores)
    positive = y == 1
    flagged = scores > THRESHOLD
    dr = float(flagged[positive].mean())
    fa = float(flagged[~positive].mean())
    return {"DR": 100 * dr, "FA": 100 * fa,
            "ACC": 50 * (dr + 1 - fa),
            "AUC": 100 * float(roc_auc_score(y, scores))}


def oracle_envelope(y, lower, upper, reverse=False):
    """Label-aware outer bound, including EVERY distinct threshold boundary."""
    y = np.asarray(y)
    favoured = np.where(y == 1, -lower if reverse else upper,
                        -upper if reverse else lower)
    fpr, tpr, thresholds = roc_curve(y, favoured, drop_intermediate=False)
    accuracy = 50 * (1 + tpr - fpr)
    best = int(np.argmax(accuracy))
    at_fa = {}
    for limit in (15.0, 15.5):
        idx = int(np.flatnonzero(fpr <= limit / 100)[-1])
        at_fa[str(limit)] = {"max_DR": float(tpr[idx] * 100),
                            "FA": float(fpr[idx] * 100),
                            # sklearn's >= boundary becomes our strict > cutoff.
                            "threshold": float(np.nextafter(thresholds[idx], -np.inf)) if np.isfinite(thresholds[idx]) else None}
    sample = np.unique(np.concatenate((np.linspace(0, len(fpr) - 1, min(2001, len(fpr)), dtype=int), [best])))
    result = {
        "direction": "lower_control" if reverse else "higher_printed",
        "threshold_candidates": len(thresholds),
        "max_ACC": float(accuracy[best]),
        "max_AUC": float(np.trapezoid(tpr, fpr) * 100),
        "at_FA_cap": at_fa,
        "target_pair_not_excluded": at_fa["15.0"]["max_DR"] >= 81,
        "rounded_target_pair_not_excluded": at_fa["15.5"]["max_DR"] >= 80.5,
        "curve": {"FA": (fpr[sample] * 100).tolist(), "DR": (tpr[sample] * 100).tolist()},
    }
    if not reverse:
        result["at_printed_threshold"] = {
            "max_DR": float(np.mean(upper[y == 1] > THRESHOLD) * 100),
            "min_FA": float(np.mean(lower[y == 0] > THRESHOLD) * 100),
        }
    return result


def interval(values):
    return [float(v) for v in np.quantile(values, [0.025, 0.975])]


def paired_customer_statistics(score_sets, customer_ids, bootstraps=BOOTSTRAPS, seed=ANALYSIS_SEED):
    """All seven original blocks share the same ordered source days."""
    names = list(score_sets)
    n = len(customer_ids)
    customers, inverse = np.unique(customer_ids, return_inverse=True)
    c = len(customers)
    if c < 2:
        raise ValueError("At least two customer clusters are required")
    counts = np.bincount(inverse, minlength=c)
    flagged = np.empty((len(names), 7, c))
    wins = np.empty((len(names), 6, c))
    pair_details = {}
    for m, name in enumerate(names):
        scores = np.asarray(score_sets[name][:7 * n]).reshape(7, n)
        for group in range(7):
            flagged[m, group] = np.bincount(inverse, weights=scores[group] > THRESHOLD, minlength=c)
        pair_details[name] = []
        for attack in range(1, 7):
            difference = scores[attack] - scores[0]
            credit = (difference > 0).astype(float) + 0.5 * (difference == 0)
            wins[m, attack - 1] = np.bincount(inverse, weights=credit, minlength=c)
            pair_details[name].append({"attack": attack,
                "attack_above_parent_percent": float(100 * np.mean(difference > 0)),
                "ties_percent": float(100 * np.mean(difference == 0)),
                "win_rate_half_ties": float(100 * credit.mean())})
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(c, np.full(c, 1 / c), size=bootstraps)
    denominator = weights @ counts
    observed_rates = flagged.sum(axis=2) / n
    bootstrap_rates = np.stack([weights @ flagged[m].T / denominator[:, None] for m in range(len(names))])
    observed_ba = 50 * (1 - observed_rates[:, :1] + observed_rates[:, 1:])
    bootstrap_ba = 50 * (1 - bootstrap_rates[:, :, :1] + bootstrap_rates[:, :, 1:])
    by_control = {}
    for m, name in enumerate(names[1:], 1):
        delta = observed_ba[0] - observed_ba[m]
        delta_boot = bootstrap_ba[0] - bootstrap_ba[m]
        paired_delta = 100 * (wins[0] - wins[m])
        pair_boot = weights @ paired_delta.T / denominator[:, None]
        ci = interval(delta_boot.mean(axis=1))
        by_control[name] = {
            "original_ACC_gain_pp": float(delta.mean()),
            "original_ACC_gain_95CI_pp": ci,
            "within_predeclared_plus_minus_1pp": ci[0] > -1 and ci[1] < 1,
            "per_attack": [{"attack": a + 1, "ACC_gain_pp": float(delta[a]),
                            "ACC_gain_95CI_pp": interval(delta_boot[:, a]),
                            "source_pair_win_gain_pp": float(paired_delta[a].sum() / n),
                            "source_pair_win_gain_95CI_pp": interval(pair_boot[:, a])}
                           for a in range(6)],
        }
    return {"customers": c, "source_days": n, "resamples": bootstraps, "seed": seed,
            "conditional_on": "fixed fitted model, split, preprocessing, and generated attacks; original rows only",
            "comparisons": by_control, "source_pair_rates": pair_details}


def energy_band_rankings(y, energy, trained, bins=100):
    edges = np.unique(np.quantile(energy, np.linspace(0, 1, bins + 1)))
    membership = np.searchsorted(edges[1:-1], energy, side="right")
    rows, total_pairs = [], 0
    sums = {"energy": 0.0, "trained": 0.0, "trained_minus_energy": 0.0}
    corrections = trained.astype(np.float64) - energy
    for b in range(max(1, len(edges) - 1)):
        selected = membership == b
        labels = y[selected]
        positives, negatives = int(labels.sum()), int(len(labels) - labels.sum())
        pairs = positives * negatives
        if not pairs:
            continue
        values = {"energy": energy[selected], "trained": trained[selected],
                  "trained_minus_energy": corrections[selected]}
        aucs = {name: float(100 * roc_auc_score(labels, value)) for name, value in values.items()}
        for name in sums:
            sums[name] += aucs[name] * pairs
        total_pairs += pairs
        rows.append({"bin": b, "rows": len(labels), "positive_negative_pairs": pairs, "AUC": aucs})
    return {"requested_bins": bins, "usable_bins": len(rows), "pair_count": total_pairs,
            "pair_weighted_within_bin_AUC": {k: v / total_pairs if total_pairs else None for k, v in sums.items()},
            "bins": rows, "interpretation": "Ranking within label-blind energy bands, not exact conditional independence"}


def changed_decisions(y, trained, control):
    a, b = trained > THRESHOLD, control > THRESHOLD
    changed = a != b
    good = a == y
    return {"rows": len(y), "changed": int(changed.sum()),
            "beneficial": int(np.sum(changed & good)),
            "harmful": int(np.sum(changed & ~good))}


def make_figures(record, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for view, label, color in [("full", "Post-ADASYN evaluation", "#225c65"),
                                ("original", "Original rows only", "#805b28")]:
        curve = record["bounds"][view]["printed"]["curve"]
        ax.plot(curve["FA"], curve["DR"], label=label + " — optimistic limit", color=color)
    ax.scatter([15], [81], marker="*", s=160, color="#923f32", label="Paper target", zorder=3)
    measured = record["metrics"]["full"]["trained"]
    ax.scatter([measured["FA"]], [measured["DR"]], s=45, color="#222823", label="Saved FC-SAE", zorder=3)
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="False alarms (%)", ylabel="Attacks detected (%)",
           title="Even label-aware outputs remain subject to the score range")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output / "output-domain-envelope.svg")
    plt.close(fig)
    comparison = record["customer_statistics"]["comparisons"]["zero"]
    effects = [comparison["original_ACC_gain_pp"]] + [r["ACC_gain_pp"] for r in comparison["per_attack"]]
    cis = [comparison["original_ACC_gain_95CI_pp"]] + [r["ACC_gain_95CI_pp"] for r in comparison["per_attack"]]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    positions = np.arange(7)
    ax.axvspan(-1, 1, color="#eef2ee", label="Predeclared ±1-point region")
    ax.axvline(0, color="#60675f", linewidth=0.8)
    for i, (effect, ci) in enumerate(zip(effects, cis)):
        ax.plot(ci, [i, i], color="#225c65", linewidth=1.5)
        ax.scatter([effect], [i], color="#225c65", s=25)
    ax.set_yticks(positions, ["All six attacks"] + [f"Attack {i}" for i in range(1, 7)])
    ax.invert_yaxis()
    ax.set(xlabel="Trained minus zero-reconstruction balanced accuracy (points)",
           title="Training's contribution varies by attack\nOriginal rows; 95% paired customer-bootstrap intervals")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "useful-work-by-attack.svg")
    plt.close(fig)


def run_analysis(args, record):
    started = time.perf_counter()
    run = args.result.parent
    result = json.loads(args.result.read_text())
    if digest(args.result) != RESULT_SHA or result["git_commit"] != SCIENCE_COMMIT:
        raise ValueError("The frozen source attempt does not match")
    data = args.root / result["data"]["path"]
    if digest(data / "metadata.json") != result["data"]["metadata_sha256"]:
        raise ValueError("Data metadata mismatch")
    metadata = json.loads((data / "metadata.json").read_text())
    data_names = ("x_test.npy", "y_test.npy", "test_attack_id.npy", "test_source_row.npy", "meter_ids.npy")
    run_names = ("scores.npy", "zero_reconstruction_scores.npy", "softmax_projection_floor_scores.npy")
    verified = {}
    for folder, names, expected in [(data, data_names, {**{k: v["sha256"] for k, v in metadata["files"].items()}, **result["data"]["files"]}),
                                     (run, run_names, result["artifact_sha256"])]:
        for name in names:
            actual = digest(folder / name)
            if actual != expected[name]:
                raise ValueError(f"Input checksum mismatch: {name}")
            verified[name] = actual
    record["input_sha256"] = verified
    record["verification_seconds"] = time.perf_counter() - started
    print("Consumed artifact hashes verified", flush=True)
    arrays = {name: np.load(data / name, mmap_mode="r") for name in data_names}
    saved = {name: np.load(run / name, mmap_mode="r") for name in run_names}
    total, base = len(arrays["y_test.npy"]), metadata["counts"]["B2_profiles"]
    full = args.stage == "full"
    base_indices = np.arange(base) if full else np.linspace(0, base - 1, 64, dtype=int)
    indices = (None if full else np.concatenate([base_indices + g * base for g in range(7)] +
              [np.linspace(7 * base, total - 1, 64, dtype=int)]))
    select = lambda a: np.asarray(a) if full else np.asarray(a[indices])
    labels, attack_ids, sources = [select(arrays[k]) for k in ("y_test.npy", "test_attack_id.npy", "test_source_row.npy")]
    n, original = len(labels), 7 * len(base_indices)
    expected_sources = np.tile(sources[:len(base_indices)], 7)
    if not (np.array_equal(sources[:original], expected_sources)
            and np.array_equal(attack_ids[:original], np.repeat(np.arange(7), len(base_indices)))
            and np.array_equal(labels[:original], np.repeat([0, 1, 1, 1, 1, 1, 1], len(base_indices)))
            and np.all(labels[original:] == 0) and np.all(sources[original:] == -1)
            and np.all(attack_ids[original:] == -1)):
        raise ValueError("Original-block or synthetic identity mismatch")
    customer_ids = np.asarray(arrays["meter_ids.npy"][sources[:len(base_indices)]])
    scores = {"trained": select(saved["scores.npy"]), "zero": select(saved["zero_reconstruction_scores.npy"]),
              "projection_floor": select(saved["softmax_projection_floor_scores.npy"])}
    lower, upper, uniform = (np.empty(n, dtype=np.float64) for _ in range(3))
    geometry_started = time.perf_counter()
    max_score_violation, max_saved_floor_difference, max_zero_difference = 0.0, 0.0, 0.0
    for start in range(0, n, 32768):
        stop = min(start + 32768, n)
        x = arrays["x_test.npy"][start:stop] if full else arrays["x_test.npy"][indices[start:stop]]
        lo, hi, constant, energy = simplex_bounds(x)
        padding = 1e-5 * (1 + hi)
        lower[start:stop], upper[start:stop], uniform[start:stop] = np.maximum(0, lo - padding), hi + padding, constant
        trained = scores["trained"][start:stop]
        violation = max(float(np.max(lower[start:stop] - trained)), float(np.max(trained - upper[start:stop])))
        max_score_violation = max(max_score_violation, violation)
        floor_error = np.abs(lo - scores["projection_floor"][start:stop])
        zero_error = np.abs(energy - scores["zero"][start:stop])
        max_saved_floor_difference = max(max_saved_floor_difference, float(floor_error.max()))
        max_zero_difference = max(max_zero_difference, float(zero_error.max()))
        if not (np.isfinite(trained).all() and np.all(lo <= hi + padding)
                and np.all(floor_error <= padding) and np.all(zero_error <= padding)):
            raise ValueError("Non-finite score or saved baseline mismatch")
    if max_score_violation > 0:
        raise ValueError("Saved trained score lies outside the padded range")
    scores["uniform"] = uniform
    record["geometry"] = {"rows": n, "features": 48, "seconds": time.perf_counter() - geometry_started,
        "outward_padding": "1e-5 * (1 + upper_endpoint)", "max_trained_range_violation": max_score_violation,
        "max_saved_floor_difference": max_saved_floor_difference, "max_saved_zero_difference": max_zero_difference}
    print("Geometry and saved-score containment passed", record["geometry"], flush=True)
    record["bounds"], record["metrics"] = {}, {}
    for view, rows in [("full", slice(None)), ("original", slice(0, original))]:
        y, lo, hi = labels[rows], lower[rows], upper[rows]
        record["bounds"][view] = {"printed": oracle_envelope(y, lo, hi), "reversed_control": oracle_envelope(y, lo, hi, reverse=True)}
        record["metrics"][view] = {name: metrics(y, values[rows]) for name, values in scores.items()}
        record.setdefault("decision_changes", {})[view] = {name: changed_decisions(y, scores["trained"][rows], values[rows])
            for name, values in scores.items() if name != "trained"}
        print(view, "optimistic bound", {k: v for k, v in record["bounds"][view]["printed"].items() if k != "curve"}, flush=True)
    record["per_attack"] = []
    for attack in range(1, 7):
        rows = np.r_[0:len(base_indices), attack * len(base_indices):(attack + 1) * len(base_indices)]
        record["per_attack"].append({"attack": attack, "metrics": {name: metrics(labels[rows], values[rows]) for name, values in scores.items()}})
    record["synthetic_benign"] = {name: {"rows": n - original, "FA": float(100 * np.mean(values[original:] > THRESHOLD))}
                                     for name, values in scores.items()}
    record["customer_statistics"] = paired_customer_statistics(scores, customer_ids)
    record["energy_band_rankings"] = energy_band_rankings(labels[:original], scores["zero"][:original], scores["trained"][:original])
    record["elapsed_seconds"] = time.perf_counter() - started
    record["status"] = "passed"
    print("Useful-information summaries complete", flush=True)
    make_figures(record, args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Scientific scoring must run on a cluster compute node")
    args.output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    record = {"stage": args.stage, "status": "running", "classification": "C/A and C/M diagnostics on frozen P+I inputs",
              "source_result_sha256": RESULT_SHA, "scientific_commit": SCIENCE_COMMIT,
              "script_sha256": digest(script), "contract_sha256": digest(script.parent.parent / "POST_ANCHOR_DIAGNOSTICS.md"),
              "analysis_commit": subprocess.check_output(["git", "-C", str(script.parent), "rev-parse", "HEAD"], text=True).strip(),
              "job_id": os.environ["SLURM_JOB_ID"], "analysis_seed": ANALYSIS_SEED,
              "versions": {"numpy": np.__version__}, "threshold": THRESHOLD,
              "numerical_scope": "Closed-simplex analytic relaxation, conservative float64 evaluation; not certified interval arithmetic"}
    try:
        run_analysis(args, record)
    except Exception:
        record["status"], record["error"] = "failed", traceback.format_exc()
        raise
    finally:
        with (args.output / "diagnostics.json").open("x") as handle:
            json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    print("Saved", args.output, "elapsed", record["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    main()
