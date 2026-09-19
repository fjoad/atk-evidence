#!/usr/bin/env python3
"""Prepare Paper 3 inputs in the order declared in PREPARATION.md.

Actual CER preparation is allowed only in a Slurm compute job. The fixture
route generates software-test inputs and cannot be reported as a reproduction.
No model construction, training, or scoring is performed here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time
import zipfile

import imblearn
from imblearn.over_sampling import ADASYN
import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from download_data import ALLOCATION_CSV, ALLOCATION_TAB, DEFAULT_RAW, FILES, hashes, read_allocation, verify


STUDY = Path(__file__).resolve().parents[1]
SEED = 20260920
FEATURES = 48


def rng(seed: int, role: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, role]))


def library_seed(seed: int, role: int) -> int:
    return int(np.random.SeedSequence([seed, role]).generate_state(1)[0])


@dataclass
class Rows:
    x: np.ndarray
    true_y: np.ndarray
    observed_y: np.ndarray
    meter: np.ndarray
    day: np.ndarray
    attack: np.ndarray
    source_id: np.ndarray
    uid: np.ndarray
    synthetic: np.ndarray
    parent_a: np.ndarray
    parent_b: np.ndarray
    mix: np.ndarray

    def take(self, index: np.ndarray) -> "Rows":
        return Rows(**{f.name: getattr(self, f.name)[index].copy() for f in fields(self)})

    def __len__(self) -> int:
        return len(self.true_y)


def join(*parts: Rows) -> Rows:
    return Rows(**{f.name: np.concatenate([getattr(p, f.name) for p in parts]) for f in fields(Rows)})


def original_rows(x: np.ndarray, meter: np.ndarray, day: np.ndarray, attack: int) -> Rows:
    count = len(x)
    source_id = meter.astype(np.int64) * 1000 + day
    uid = source_id * 7 + attack
    label = np.full(count, int(attack > 0), np.int8)
    return Rows(np.asarray(x, np.float32), label, label.copy(), meter.copy(), day.copy(),
                np.full(count, attack, np.int8), source_id, uid, np.zeros(count, bool),
                uid.copy(), np.full(count, -1, np.int64), np.zeros(count))


def select_customers(mapping: dict[int, int], count: int, seed: int) -> np.ndarray:
    population = np.array(sorted(k for k, v in mapping.items() if v == 1), dtype=np.int32)
    if not 1 <= count <= len(population):
        raise ValueError("Requested customer count exceeds the residential population")
    # Prefix of a fixed permutation makes a pilot a subset of the full selection.
    return np.sort(rng(seed, 1).permutation(population)[:count])


def profiles(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Require exactly one of each half-hour slot; never fill malformed days."""
    if frame.empty:
        raise ValueError("No readings for the declared selection")
    frame = frame.copy()
    if not np.isfinite(frame.kwh).all() or (frame.kwh < 0).any():
        raise ValueError("Readings must be finite and nonnegative")
    frame["day"] = frame.day_time // 100
    frame["slot"] = frame.day_time % 100
    if (frame.day < 1).any() or (frame.day > 999).any():
        raise ValueError("Day code lies outside documented three-digit encoding")
    grouped = frame.groupby(["meter_id", "day"]).slot.agg(["size", "nunique", "min", "max"])
    good = (
        (grouped["size"] == 48) & (grouped["nunique"] == 48)
        & (grouped["min"] == 1) & (grouped["max"] == 48)
    )
    admitted = grouped.index[good]
    indexed = frame.set_index(["meter_id", "day"])
    kept = indexed.loc[indexed.index.isin(admitted)].reset_index()
    if kept.empty:
        raise ValueError("No complete 48-slot days")
    matrix = kept.pivot(index=["meter_id", "day"], columns="slot", values="kwh").sort_index()
    summary = {
        "input_rows": len(frame), "candidate_days": len(grouped),
        "admitted_days": int(good.sum()), "excluded_days": int((~good).sum()),
        "days_with_duplicate_slots": int((grouped["size"] != grouped["nunique"]).sum()),
        "days_with_extra_slots": int((grouped["max"] > 48).sum()),
    }
    return (matrix.to_numpy(dtype=np.float32),
            matrix.index.get_level_values(0).to_numpy(dtype=np.int32),
            matrix.index.get_level_values(1).to_numpy(dtype=np.int32), summary)


def read_profiles(raw: Path, selected: np.ndarray, day_start: int, day_stop: int,
                  chunk_rows: int = 1_000_000) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    blocks = []
    scanned = 0
    for filename in FILES:
        with zipfile.ZipFile(raw / filename) as archive:
            members = [x.filename for x in archive.infolist() if not x.is_dir()]
            if len(members) != 1:
                raise ValueError("Expected one consumption text file per archive")
            with archive.open(members[0]) as stream:
                for chunk in pd.read_csv(stream, sep=r"\s+", header=None,
                                         names=["meter_id", "day_time", "kwh"],
                                         dtype={"meter_id": "int32", "day_time": "int32", "kwh": "float32"},
                                         chunksize=chunk_rows):
                    scanned += len(chunk)
                    day = chunk.day_time // 100
                    mask = chunk.meter_id.isin(selected) & day.between(day_start, day_stop)
                    if mask.any():
                        blocks.append(chunk.loc[mask].copy())
        print(f"read {filename}; rows scanned={scanned}", flush=True)
    if not blocks:
        raise ValueError("No selected readings")
    x, meters, days, record = profiles(pd.concat(blocks, ignore_index=True))
    record["scanned_rows"] = scanned
    absent = np.setdiff1d(selected, np.unique(meters))
    if len(absent):
        raise ValueError(f"{len(absent)} selected customers have no complete days; no silent replacement")
    return x, meters, days, record


def attack_arrays(x: np.ndarray, meter: np.ndarray, *, seed: int,
                  alpha_scope: str = "customer", bypass: str = "duration-first") -> dict[int, np.ndarray]:
    """Target Table I order, not the different h4/h5 order in Jokar's thesis."""
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or x.shape[1] != FEATURES or len(meter) != len(x):
        raise ValueError("Attacks require one customer ID per 48-coordinate day")
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Attack inputs must be nonnegative finite consumption")
    unique, inverse = np.unique(meter, return_inverse=True)
    if alpha_scope == "customer":
        alpha = rng(seed, 101).uniform(.1, .8, len(unique))[inverse, None]
    elif alpha_scope == "global":
        alpha = rng(seed, 101).uniform(.1, .8)
    else:
        raise ValueError("Unknown alpha scope")
    attack3_rng = rng(seed, 103)
    duration = attack3_rng.integers(8, 49, len(x))
    if bypass == "duration-first":
        start = attack3_rng.integers(0, 49 - duration)
    elif bypass == "start-first-clip":
        # Reference [26]'s [0,42] start; boundary convention remains a choice.
        attack3_rng = rng(seed, 103)
        start = attack3_rng.integers(0, 43, len(x))
        duration = attack3_rng.integers(8, 49, len(x))
    else:
        raise ValueError("Unknown bypass completion")
    stop = np.minimum(start + duration, FEATURES)
    bypassed = x.copy()
    positions = np.arange(FEATURES)[None, :]
    bypassed[(positions >= start[:, None]) & (positions < stop[:, None])] = 0
    daily_mean = x.mean(axis=1, keepdims=True)
    return {
        1: (x * alpha).astype(np.float32),
        2: (x * rng(seed, 102).uniform(.1, .8, x.shape)).astype(np.float32),
        3: bypassed,
        4: np.repeat(daily_mean, FEATURES, axis=1),
        5: (daily_mean * rng(seed, 105).uniform(.1, .8, x.shape)).astype(np.float32),
        6: x[:, ::-1].copy(),
    }


class RecordingNeighbors(NearestNeighbors):
    """Record the two neighbor-index queries made by unmodified ADASYN."""

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        answer = super().kneighbors(X, n_neighbors, return_distance)
        indices = answer[1] if return_distance else answer
        if not hasattr(self, "queries_"):
            self.queries_ = []
        self.queries_.append(indices.copy())
        return answer


def balance(rows: Rows, *, seed: int, neighbors: int = 5) -> tuple[Rows, dict]:
    """Use stock ADASYN values; replay its draws only to recover ancestry."""
    if imblearn.__version__ != "0.14.2":
        raise RuntimeError("Provenance replay is frozen to imbalanced-learn 0.14.2")
    if not np.array_equal(rows.true_y, rows.observed_y) or rows.synthetic.any():
        raise ValueError("Initial ADASYN requires clean original labels")
    benign = np.flatnonzero(rows.true_y == 0)
    malicious = np.count_nonzero(rows.true_y == 1)
    if len(benign) >= malicious:
        raise ValueError("Expected the benign class to be the strict minority")
    estimator = RecordingNeighbors(n_neighbors=neighbors + 1, n_jobs=1)
    sampler = ADASYN(sampling_strategy={0: int(malicious)}, random_state=seed, n_neighbors=estimator)
    x_resampled, labels = sampler.fit_resample(rows.x, rows.true_y)
    global_neighbors, benign_neighbors = sampler.nn_.queries_
    difficulty = np.sum(rows.true_y[global_neighbors[:, 1:]] != 0, axis=1) / neighbors
    allocation = np.rint(difficulty / difficulty.sum() * (malicious - len(benign))).astype(int)
    anchors = np.repeat(np.arange(len(benign)), allocation)
    random_state = np.random.RandomState(seed)
    columns = random_state.choice(neighbors, size=len(anchors))
    mix = random_state.uniform(size=(len(anchors), 1))
    parent_a = benign[anchors]
    parent_b = benign[benign_neighbors[:, 1:][anchors, columns]]
    generated = len(labels) - len(rows)
    if generated != len(anchors) or not np.array_equal(labels[:len(rows)], rows.true_y):
        raise AssertionError("ADASYN output no longer matches provenance replay")
    # Check every interpolation in bounded blocks without another neighbor search.
    for start in range(0, generated, 50_000):
        sl = slice(start, start + 50_000)
        expected = (rows.x[parent_a[sl]] + mix[sl]
                    * (rows.x[parent_b[sl]] - rows.x[parent_a[sl]])).astype(rows.x.dtype)
        if not np.array_equal(expected, x_resampled[len(rows) + start:len(rows) + start + len(expected)]):
            raise AssertionError("Synthetic interpolation provenance mismatch")
    new = Rows(
        x_resampled[len(rows):], labels[len(rows):], labels[len(rows):].copy(),
        np.full(generated, -1, np.int32), np.full(generated, -1, np.int32),
        np.zeros(generated, np.int8), np.full(generated, -1, np.int64),
        -np.arange(1, generated + 1, dtype=np.int64), np.ones(generated, bool),
        rows.uid[parent_a], rows.uid[parent_b], mix[:, 0],
    )
    return join(rows, new), {
        "implementation": "unmodified imbalanced-learn ADASYN with recorded neighbor queries",
        "neighbors": neighbors, "seed": seed, "generated": generated,
        "class_counts": np.bincount(labels, minlength=2).tolist(),
        "all_interpolations_verified": True,
    }


def split_indices(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if count < 3:
        raise ValueError("Need at least three samples for the 2:1 split")
    order = rng(seed, 201).permutation(count)
    return order[:2 * count // 3], order[2 * count // 3:]


def poison(train: Rows, attacks: dict[int, Rows], *, mode: str, scope: str,
           rate: float, population: np.ndarray, seed: int,
           denominator: str = "all-training") -> tuple[Rows, dict]:
    if not np.isfinite(rate) or not 0 <= rate <= .30:
        raise ValueError("Poisoning rate must be in the studied interval [0,0.30]")
    result = train.take(np.arange(len(train)))
    eligible = np.flatnonzero(train.true_y == 1) if mode == "two-class" else np.arange(len(train))
    selected_customers = np.array([], dtype=np.int32)
    if scope == "generalized":
        selected_customers = rng(seed, 301).permutation(np.sort(population))[:int(np.floor(rate * len(population)))]
        changed = eligible[np.isin(train.meter[eligible], selected_customers)]
    elif scope == "customer-specific":
        base_count = len(eligible) if denominator == "malicious-training" and mode == "two-class" else len(train)
        count = int(np.floor(rate * base_count))
        if count > len(eligible):
            raise ValueError("Requested poison count exceeds malicious training examples")
        changed = rng(seed, 302).permutation(eligible)[:count]
    else:
        raise ValueError("Unknown scope")
    if mode == "two-class":
        result.observed_y[changed] = 0
    elif mode == "novelty":
        choices = rng(seed, 303).integers(1, 7, len(train))
        lookup = {int(uid): i for i, uid in enumerate(attacks[1].source_id)}
        for attack in range(1, 7):
            target = changed[choices[changed] == attack]
            source = np.array([lookup[int(s)] for s in train.source_id[target]], dtype=np.int64)
            for f in fields(Rows):
                getattr(result, f.name)[target] = getattr(attacks[attack], f.name)[source]
        result.observed_y[:] = 0
    else:
        raise ValueError("Unknown detector path")
    return result, {
        "rate": rate, "scope": scope, "operation": "label-flip" if mode == "two-class" else "replace-with-attack",
        "selection_denominator": "selected-customers" if scope == "generalized" else denominator,
        "selected_customer_count": len(selected_customers), "changed_rows": len(changed),
        "training_rows": len(train), "changed_fraction_of_training": len(changed) / len(train),
        "eligible_rows": len(eligible),
        "changed_fraction_of_eligible": len(changed) / len(eligible) if len(eligible) else 0,
    }


def prepare(x: np.ndarray, meter: np.ndarray, day: np.ndarray, *, mode: str,
            scope: str = "generalized", customer: int | None = None,
            rate: float = 0, seed: int = SEED, neighbors: int = 5,
            alpha_scope: str = "customer", bypass: str = "duration-first",
            denominator: str = "all-training") -> tuple[dict[str, np.ndarray], dict]:
    if mode not in ("novelty", "two-class") or denominator not in ("all-training", "malicious-training"):
        raise ValueError("Invalid preparation mode or poisoning denominator")
    if scope == "customer-specific":
        if customer is None:
            raise ValueError("Customer-specific preparation requires an explicit customer")
        select = meter == customer
        x, meter, day = x[select], meter[select], day[select]
    if not len(x) or len(np.unique(meter.astype(np.int64) * 1000 + day)) != len(x):
        raise ValueError("Input profiles need unique customer/day identities")
    benign = original_rows(x, meter, day, 0)
    attacks = {a: original_rows(v, meter, day, a) for a, v in
               attack_arrays(x, meter, seed=seed, alpha_scope=alpha_scope, bypass=bypass).items()}
    if mode == "novelty":
        train_i, test_i = split_indices(len(benign), seed)
        train = benign.take(train_i)
        evaluation = join(benign.take(test_i), *attacks.values())
        test, resampling = balance(evaluation, seed=library_seed(seed, 202), neighbors=neighbors)
        order = ["benign-split", "all-attacks-to-test", "test-ADASYN", "training-poison", "train-fit-scaler"]
    else:
        pool, resampling = balance(join(benign, *attacks.values()), seed=library_seed(seed, 202), neighbors=neighbors)
        train_i, test_i = split_indices(len(pool), seed)
        train, test = pool.take(train_i), pool.take(test_i)
        order = ["benign-and-attacks", "pool-ADASYN", "row-split", "training-poison", "train-fit-scaler"]
    train, poisoning = poison(train, attacks, mode=mode, scope=scope, rate=rate,
                              population=np.unique(meter), seed=seed, denominator=denominator)
    scaler = StandardScaler().fit(train.x.astype(np.float64))
    arrays = {"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_}
    for prefix, rows in (("train", train), ("test", test)):
        for f in fields(Rows):
            arrays[prefix + "_" + ("raw_x" if f.name == "x" else f.name)] = getattr(rows, f.name)
        arrays[prefix + "_x"] = scaler.transform(rows.x).astype(np.float32)
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Prepared arrays contain nonfinite values")
    test_original = test.uid[~test.synthetic]
    shared_source = np.intersect1d(train.source_id[~train.synthetic], test.source_id[~test.synthetic])
    overlap = np.intersect1d(train.uid[~train.synthetic], test_original)
    synthetic_crossing = {}
    for name, left, right in (("train_synthetic_test_parent", train, test),
                              ("test_synthetic_train_parent", test, train)):
        parents = np.concatenate([left.parent_a[left.synthetic], left.parent_b[left.synthetic]])
        synthetic_crossing[name] = int(np.isin(parents, right.uid[~right.synthetic]).sum())
    record = {
        "mode": mode, "scope": scope, "rate": rate, "seed": seed,
        "alpha_scope": alpha_scope, "bypass": bypass, "order": order,
        "profiles": len(x), "customer_count": len(np.unique(meter)),
        "training_rows": len(train), "test_rows": len(test),
        "train_true_counts": np.bincount(train.true_y, minlength=2).tolist(),
        "train_observed_counts": np.bincount(train.observed_y, minlength=2).tolist(),
        "test_true_counts": np.bincount(test.true_y, minlength=2).tolist(),
        "shared_original_row_identities": len(overlap), "shared_source_days": len(shared_source),
        "synthetic_parent_crossings": synthetic_crossing,
        "resampling": resampling, "poisoning": poisoning,
        "interpretation": "explicit completion of PREPARATION.md; not exact author sample identity",
    }
    return arrays, record


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constructed consumption-shaped arrays; software validation only."""
    meter = np.repeat(np.arange(1, 21, dtype=np.int32), 28)
    day = np.tile(np.arange(195, 223, dtype=np.int32), 20)
    x = rng(17, 1).lognormal(0, .8, (len(meter), 48)).astype(np.float32)
    return x, meter, day


def save_arrays(output: Path, arrays: dict[str, np.ndarray], record: dict) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, values in arrays.items():
        filename = output / (name + ".npy")
        np.save(filename, values, allow_pickle=False)
        files[filename.name] = {"shape": list(values.shape), "dtype": str(values.dtype),
                                "sha256": hashes(filename)["sha256"]}
    complete = {**record, "files": files}
    (output / "metadata.json").write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n")
    return complete


def require_compute() -> None:
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_JOB_NODELIST"):
        raise RuntimeError("Real-data preparation requires a Slurm compute allocation; use --fixture for software tests")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--customers", type=int, default=3000)
    parser.add_argument("--day-start", type=int, default=1)
    parser.add_argument("--day-stop", type=int, default=999)
    parser.add_argument("--days-per-customer", type=int)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--rates", type=float, nargs="+", default=[0., .30])
    parser.add_argument("--include-customer-specific", action="store_true")
    parser.add_argument("--allow-expensive-adasyn", action="store_true")
    args = parser.parse_args()
    if not args.fixture:
        require_compute()
    if args.day_start > args.day_stop or len(set(args.rates)) != len(args.rates):
        parser.error("Invalid day interval or repeated poison rate")
    if not all(np.isfinite(p) and 0 <= p <= .30 for p in args.rates):
        parser.error("Rates must be in [0,0.30]")
    if args.days_per_customer is not None and args.days_per_customer < 1:
        parser.error("days-per-customer must be positive")
    bounded_days = min(args.days_per_customer or 999, args.day_stop - args.day_start + 1)
    if not args.fixture and not args.allow_expensive_adasyn and args.customers * bounded_days > 2000:
        parser.error("The requested range exceeds the small preparation budget; cost it before --allow-expensive-adasyn")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    status = {"status": "started", "created_utc": datetime.now(timezone.utc).isoformat(),
              "fixture": args.fixture, "job_id": os.environ.get("SLURM_JOB_ID"),
              "code_commit": os.environ.get("EXPECTED_COMMIT"),
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "pandas": pd.__version__, "sklearn": sklearn.__version__, "imblearn": imblearn.__version__},
              "source_hashes": {p.name: hashes(p)["sha256"] for p in
                               (Path(__file__), Path(__file__).with_name("download_data.py"), STUDY / "PREPARATION.md")},
              "runs": []}
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    try:
        if args.fixture:
            x, meter, day = fixture()
            status["source"] = "constructed software fixture; ineligible as empirical paper evidence"
        else:
            status["source"] = verify(args.raw)
            if not status["source"]["ready"]:
                raise ValueError("Raw source identity failed")
            allocation = args.raw / (ALLOCATION_TAB if (args.raw / ALLOCATION_TAB).exists() else ALLOCATION_CSV)
            selected = select_customers(read_allocation(allocation), args.customers, args.seed)
            x, meter, day, status["parser"] = read_profiles(args.raw, selected, args.day_start, args.day_stop)
            if args.days_per_customer is not None:
                kept = np.concatenate([np.flatnonzero(meter == customer)[:args.days_per_customer]
                                       for customer in selected])
                x, meter, day = x[kept], meter[kept], day[kept]
            np.save(args.output / "selected_customers.npy", selected, allow_pickle=False)
        status["selection"] = {"customers": len(np.unique(meter)), "profiles": len(x),
                               "day_start": int(day.min()), "day_stop": int(day.max()), "seed": args.seed,
                               "requested_day_range": [args.day_start, args.day_stop],
                               "first_complete_days_per_customer": args.days_per_customer}
        if len(x) > 2000 and not args.allow_expensive_adasyn:
            raise RuntimeError("More than 2000 source days requires an explicitly costed ADASYN allocation")
        scopes = ["generalized", "customer-specific"] if args.include_customer_specific else ["generalized"]
        for scope in scopes:
            for mode in ("two-class", "novelty"):
                for rate in args.rates:
                    name = f"{scope}-{mode}-p{round(rate * 100):02d}"
                    tick = time.perf_counter()
                    arrays, record = prepare(x, meter, day, mode=mode, scope=scope,
                                             customer=int(meter.min()), rate=rate, seed=args.seed)
                    record["elapsed_seconds"] = time.perf_counter() - tick
                    save_arrays(args.output / name, arrays, record)
                    status["runs"].append({"name": name, **record})
                    print(json.dumps({"name": name, "train": record["training_rows"],
                                      "test": record["test_rows"], "seconds": record["elapsed_seconds"]}), flush=True)
        status["status"] = "complete"
    except Exception as exc:
        status.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        status["elapsed_seconds"] = time.perf_counter() - started
        summary_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
