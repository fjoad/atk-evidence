"""The two predeclared interface readings; fixed constructed inputs only."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import sequential_constructed as G
from sequential_full_width import run_case


def branch_passes(cases):
    return bool(len(cases) == 2
        and len({c["initial_weights_sha256"] for c in cases.values()}) == 1
        and all(c["learning_gate_passed"] and c["gradient_group_gate_passed"]
                and c["weights_finite"] and c["reload_exact"]
                and c["maximum_constrained_norm"] <= 1.00001 for c in cases.values()))


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    runtime = G.R.configure_tensorflow(False)
    arrays = G.fixture_data()
    previous = G.ROOT / "data/derived/takiddin-2021-robust-poisoning/sequential-constructed-20260927-attempt1-recovery"
    with np.load(previous / "constructed_data.npz") as old:
        for name, values in arrays.items():
            np.testing.assert_array_equal(values, old[name])
    old_wide = json.loads((G.STUDY / "results/sequential_full_width_20260928/result.json").read_text())
    initial_hash = old_wide["cases"]["normal"]["initial_weights_sha256"]
    sources = [Path(__file__).resolve(), Path(G.__file__).resolve(),
               G.STUDY / "checks/sequential_full_width.py", G.STUDY / "reproduction/models.py",
               G.STUDY / "SEQUENTIAL_INTERFACE.md", G.ROOT / "docs/plans/2026-10-01-sequential-interface.md"]
    result = {"scope": "fixed full-width eight-step constructed interface comparison",
              "status": "started", "runtime": runtime,
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=G.ROOT, text=True).strip(),
              "source_sha256": {str(p.relative_to(G.ROOT)): G.digest(p) for p in sources},
              "prior_data_sha256": G.digest(previous / "constructed_data.npz"),
              "historical_relu_initial_weights_sha256": initial_hash,
              "promotion_candidate_predeclared": "sigmoid", "branches": {}}
    save = lambda: (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    save()
    import tensorflow as tf
    with tf.device("/CPU:0"):
        for activation, track in (("sigmoid", "I-SEQ-scalar-sigmoid"), ("linear", "C-SEQ-scalar-linear")):
            directory = output / activation
            directory.mkdir()
            cases = {}
            result["branches"][activation] = {"interpretation": track, "cases": cases}
            for reverse in (False, True):
                name = "reversed" if reverse else "normal"
                cases[name] = run_case(directory / name, arrays, reverse,
                                       bridge_activation=activation, fit_guard_seconds=300)
                if cases[name]["initial_weights_sha256"] != initial_hash:
                    raise ValueError("Interface comparison changed initial parameter values")
                save()
            result["branches"][activation]["gate_passed"] = branch_passes(cases)
            save()
    result["sigmoid_eligible_for_timing"] = result["branches"]["sigmoid"]["gate_passed"]
    result["status"] = "complete"
    save()
    print(json.dumps({"status": result["status"], "sigmoid_eligible_for_timing": result["sigmoid_eligible_for_timing"],
                      "branch_passes": {k: v["gate_passed"] for k,v in result["branches"].items()}}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r = run(args.output)
    raise SystemExit(0 if r["sigmoid_eligible_for_timing"] else 2)
