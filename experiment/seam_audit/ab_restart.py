"""
ab_restart.py — A/B equivalence test of switch sets that must not change the arithmetic (opt release check).

From the same saved K=1 state (single_step snapshots, N = 2000), K = 2 restarts with the (1,2) seam
configuration and the production transport (count-aware worker, split transfer queues, the dimension's
production pool factor): three identical base runs (upload order 0), one base run with a SHUFFLED upload
order (seed 1), and three test runs = the same plus the switch set under test (upload order 0). Unlike the
single-step gate (k = 1, particles away from any crossing) this follows the runs through the first
migrations: 2-D 1M crosses from k = 5, 3-D 1M from k = 1 (13 crossings by k = 50).

Why repeats: two identical K = 2 runs are not bitwise equal (the GPU's atomic orders differ: in 3-D ~20
particles flip float32 rho by 1 ULP at k = 1), and such rare discrete events dominate any statistic of a
small particle group (one flip next to the migrant moves the 90th percentile of the group's acceleration
difference by three orders of magnitude). A single floor pair therefore says little. Here the floor is the
set of 6 base pairs (3 identical pairs, 3 vs the shuffled order) and the test is the set of 9 test-vs-base
pairs; per step, particle group (all dumped fluid particles; the seam columns |x - cut| < 2 h; fluid
particles within 2 h of a particle that changed slab up to k) and field, the statistic is the rms (all
particles) or the 90th percentile of |difference| (seam / near-migrant groups). The floor is the larger of
the median over the identical pairs and the median over the shuffled-order pairs: a switch set that changes
the particle memory order (resized pools move the pid layout and the bootstrap defrag order) is compared
with a reordered (1,2) run, as the single-step gate does. PASS: the median over the 9 test pairs <=
LIMIT x that floor (plus an absolute floor of 1e-7 x the field's rms), for every step, group and field. A
systematic error on the migrant path moves every test pair; a random event moves one pair of nine.

Usage:
    .venv/Scripts/python.exe -m experiment.seam_audit.ab_restart --out logs/seam_audit/opt/ab_release
    .venv/Scripts/python.exe -m experiment.seam_audit.ab_restart --out ... --env KEY=VALUE   (extra test switches)
    .venv/Scripts/python.exe -m experiment.seam_audit.ab_restart --out ... --compare-only
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SEAM_L2 = {"V6_KEEP_DEPARTED": "1", "V6_GHOST_LAYERS": "2"}
PRODUCTION = {"V6_WORKER_COUNT_AWARE": "1", "V6_SPLIT_TRANSFER_QUEUES": "1",
              "V6_CASCADE_FORCE": "1", "V6_BAND_VOXEL_DISPATCH": "1"}
RELEASE = {
    2: {"V6_LEAN_TRANSPORT": "1", "V6_GHOST_POOL_FACTOR": "0.29", "V6_MIGRANT_POOL_FACTOR": "0.05",
        "V6_DEPARTED_FACE_FRACTION": "0.8", "V6_COMPACT_GHOST_LISTS": "1", "V6_PACKED_REPLICAS": "1",
        "V6_BAND_SLOT_LANES": "64", "V6_INIT_SEAM_CLAMP": "1"},
    3: {"V6_LEAN_TRANSPORT": "1", "V6_GHOST_POOL_FACTOR": "0.5", "V6_MIGRANT_POOL_FACTOR": "0.02",
        "V6_DEPARTED_FACE_FRACTION": "0.64", "V6_COMPACT_GHOST_LISTS": "1", "V6_PACKED_REPLICAS": "1",
        "V6_BAND_SLOT_LANES": "64", "V6_INIT_SEAM_CLAMP": "1"},
}
PRODUCTION_FACTOR = {2: "0.25", 3: "1.0"}
# name: (case yaml, dimension, steps, step-1 window, window, smoothing length)
CASES = {
    "cavity2d_1m": ("cases/lid_driven_cavity_2d_gen/case.yaml", 2, "1,2,5,10,20,50,100", 12, 12, 0.005),
    "cavity3d_1m": ("cases/cavity3d_1m/case.yaml", 3, "1,2,5,10,20,50", 8, 5, 0.02),
}
FIELDS = ("position", "velocity", "acceleration", "shift", "density", "pressure", "kernel_sum")
BASE_RUNS = ("base_1", "base_2", "base_3")
TEST_RUNS = ("test_1", "test_2", "test_3")
LIMIT = 2.0


def run_restarts(args, out_dir: pathlib.Path, case_name: str, extra: dict) -> dict:
    case_path, dimension, steps, step1_window, window, _h = CASES[case_name]
    base = {key: value for key, value in os.environ.items() if not key.startswith("V6_")}
    base.update({"VK_LOADER_LAYERS_DISABLE": "VK_LAYER_KHRONOS_validation", "PYTHONIOENCODING": "utf-8"})
    base.update(PRODUCTION)
    base["V6_GHOST_POOL_FACTOR"] = PRODUCTION_FACTOR[dimension]
    base.update(SEAM_L2)
    test_environment = dict(base)
    test_environment.update(RELEASE[dimension])
    test_environment.update(extra)
    runs = [(name, base, 0) for name in BASE_RUNS] + [("base_s", base, 1)]
    runs += [(name, test_environment, 0) for name in TEST_RUNS]
    directory = out_dir / case_name
    directory.mkdir(parents=True, exist_ok=True)
    codes = {}
    for run_name, environment, seed in runs:
        if (directory / f"{run_name}_N2000.json").exists() and not args.rerun:
            codes[run_name] = 0
            continue
        command = [sys.executable, "-m", "experiment.seam_audit.single_step", "restart", "--case", case_path,
                   "--out-dir", str(directory), "--snapshot-dir", f"logs/seam_audit/single_step/{case_name}",
                   "--run-name", run_name, "--slabs", "2", "--device-map", "0,1", "--snapshot-step", "2000",
                   "--shuffle-seed", str(seed), "--steps", steps, "--step1-window", str(step1_window),
                   "--window", str(window)]
        started = time.time()
        with open(directory / f"{run_name}.log", "w", encoding="utf-8") as log:
            completed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       cwd=_REPO_ROOT, timeout=args.timeout)
        codes[run_name] = completed.returncode
        print(f"[ab_restart] {case_name}/{run_name}: exit {completed.returncode} ({time.time() - started:.0f} s)",
              flush=True)
        (directory / f"{run_name}.environment.json").write_text(
            json.dumps({key: value for key, value in environment.items() if key.startswith("V6_")}, indent=1),
            encoding="utf-8")
    return codes


def _difference_statistic(first, second, statistic):
    difference = np.abs(second - first).reshape(first.shape[0], -1)
    if statistic == "rms":
        return float(np.sqrt(np.mean(difference ** 2)))
    return float(np.quantile(difference.max(axis=1), 0.9))


def compare(out_dir: pathlib.Path, case_name: str) -> dict:
    from scipy.spatial import cKDTree

    _case_path, _dimension, steps, _step1_window, _window, smoothing_length = CASES[case_name]
    directory = out_dir / case_name
    names = BASE_RUNS + ("base_s",) + TEST_RUNS
    documents = {name: json.loads((directory / f"{name}_N2000.json").read_text(encoding="utf-8")) for name in names}
    floor_pairs = list(itertools.combinations(BASE_RUNS, 2)) + [(name, "base_s") for name in BASE_RUNS]
    test_pairs = [(base, test) for base in BASE_RUNS for test in TEST_RUNS]
    result = {"case": case_name, "steps": {}, "valid": {name: document.get("valid") for name, document in documents.items()},
              "migrations": {name: len(document.get("migrations", [])) for name, document in documents.items()},
              "floor_pairs": floor_pairs, "test_pairs": test_pairs}
    failures, worst = [], 0.0
    for step in [int(value) for value in steps.split(",")]:
        dumps = {name: np.load(directory / f"{name}_N2000_k{step}.npz") for name in names}
        order = {name: {int(identifier): index for index, identifier in enumerate(dump["id"])}
                 for name, dump in dumps.items()}
        common = sorted(set.intersection(*[set(value) for value in order.values()]))
        index = {name: np.array([order[name][identifier] for identifier in common]) for name in names}
        reference = dumps["base_1"]
        positions = reference["position"][index["base_1"]].astype(np.float64)
        slab_zero = reference["slab"][index["base_1"]] == 0
        cut_x = positions[slab_zero, 0].max() if slab_zero.any() else 0.0
        groups = {"all": np.ones(len(common), dtype=bool),
                  "seam columns": np.abs(positions[:, 0] - cut_x) < 2 * smoothing_length}
        migrated = sorted({migration[1] for migration in documents["base_1"].get("migrations", [])
                           if migration[0] <= step})
        migrant_positions = [reference["position"][order["base_1"][identifier]].astype(np.float64)
                             for identifier in migrated if identifier in order["base_1"]]
        if migrant_positions:
            distance, _ = cKDTree(np.asarray(migrant_positions)).query(positions, k=1)
            groups["near migrants"] = distance < 2 * smoothing_length
        step_result = {"common": len(common), "migrated_so_far": len(migrated), "groups": {}}
        for group_name, mask in groups.items():
            if not mask.any():
                continue
            statistic = "rms" if group_name == "all" else "p90"
            group_result = {"n": int(mask.sum()), "statistic": statistic, "fields": {}}
            values = {name: {field: dumps[name][field][index[name]][mask].astype(np.float64) for field in FIELDS}
                      for name in names}
            for field in FIELDS:
                floor_values = [_difference_statistic(values[a][field], values[b][field], statistic)
                                for a, b in floor_pairs]
                identical_median = statistics.median(floor_values[:3])
                shuffled_median = statistics.median(floor_values[3:])
                test_values = [_difference_statistic(values[a][field], values[b][field], statistic)
                               for a, b in test_pairs]
                scale = float(np.sqrt(np.mean(values["base_1"][field] ** 2)))
                floor_median = max(identical_median, shuffled_median)
                test_median = statistics.median(test_values)
                ratio = test_median / max(floor_median, 1e-7 * max(scale, 1e-30))
                worst = max(worst, ratio)
                group_result["fields"][field] = {
                    "identical_median": identical_median, "shuffled_median": shuffled_median,
                    "floor_median": floor_median, "floor_max": max(floor_values),
                    "test_median": test_median, "test_max": max(test_values), "ratio": ratio}
                if ratio > LIMIT:
                    failures.append(f"k={step} {group_name} {field} ({statistic}): test median {test_median:.3e} "
                                    f"vs floor median {floor_median:.3e} (floor max {max(floor_values):.3e})")
            density_flips = {f"{a}-{b}": int(np.count_nonzero(values[a]["density"] != values[b]["density"]))
                             for a, b in floor_pairs + test_pairs}
            group_result["density_flips_floor_median"] = statistics.median(
                [density_flips[f"{a}-{b}"] for a, b in floor_pairs])
            group_result["density_flips_test_median"] = statistics.median(
                [density_flips[f"{a}-{b}"] for a, b in test_pairs])
            step_result["groups"][group_name] = group_result
        result["steps"][str(step)] = step_result
    result["worst_ratio"] = worst
    result["failures"] = failures
    result["pass"] = not failures and all(result["valid"].values())
    return result


def render(results: list) -> str:
    lines = ["| case | k | group | n | statistic | field | floor median (max) | test median (max) | test / floor | "
             "ρ flips floor / test (median) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for result in results:
        for step, step_result in result["steps"].items():
            for group_name, group_result in step_result["groups"].items():
                for field in ("acceleration", "density", "pressure", "shift"):
                    entry = group_result["fields"][field]
                    lines.append(f"| {result['case']} | {step} | {group_name} | {group_result['n']} | "
                                 f"{group_result['statistic']} | {field} | {entry['floor_median']:.2e} "
                                 f"({entry['floor_max']:.2e}) | {entry['test_median']:.2e} ({entry['test_max']:.2e}) | "
                                 f"{entry['ratio']:.2f} | {group_result['density_flips_floor_median']:g} / "
                                 f"{group_result['density_flips_test_median']:g} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B equivalence test (release switch set vs (1,2))")
    parser.add_argument("--out", default="logs/seam_audit/opt/ab_release")
    parser.add_argument("--cases", default="cavity2d_1m,cavity3d_1m")
    parser.add_argument("--env", action="append", default=[], help="extra KEY=VALUE for the test runs")
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--rerun", action="store_true", help="re-run restarts whose result already exists")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = dict(item.split("=", 1) for item in args.env)
    results = []
    for case_name in args.cases.split(","):
        if not args.compare_only:
            codes = run_restarts(args, out_dir, case_name, extra)
            if any(codes.values()):
                print(f"[ab_restart] {case_name}: restart failures {codes}", flush=True)
        results.append(compare(out_dir, case_name))
    (out_dir / "result.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    (out_dir / "table.md").write_text(render(results) + "\n", encoding="utf-8")
    for result in results:
        print(f"[ab_restart] {result['case']}: pass={result['pass']} worst test/floor median ratio "
              f"{result['worst_ratio']:.2f}; migrations {result['migrations']} valid {all(result['valid'].values())}",
              flush=True)
        for failure in result["failures"][:8]:
            print("   ", failure)
    return 0 if all(result["pass"] for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
