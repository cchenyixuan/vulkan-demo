"""
decomposition_audit.py - E33 (paper section 6.2): is a K > 1 run as close to a K = 1 run as two K = 1 runs are to
each other, at every column distance from the seam?

E32 part 1 (docs/perf_model/E32.md) showed that ONE pair of identical runs is no noise floor in 2-D at long
horizons: the runs leave the round-off regime at random times between 300 and 2000 steps, so two K = 1 runs can end
30 times further apart than two others. This audit uses every pair instead (the ensemble of nowait_audit).

Runs: 2-D 1M and 3-D 1M restarted from the N = 2000 snapshots (dump_state --restart-snapshot) at depth 2 with the
code defaults (release set, phase-A no-wait) and V6_DELTA_DENSITY=1 (as E14 / E32), dumping every particle 300 and
2000 steps after the restart. Arms, 6 runs each, the arm order rotating from trial to trial:
  2-D 1M  k1   K = 1 on GPU 0 in trials 1 / 3 / 5, on GPU 1 in trials 2 / 4 / 6
          k2   K = 2, equal weights (cut 103, the middle of the cavity)
          k2s  K = 2, weights 0.9,1.1 (cut 93): does anything depend on where the seam is?
          k4   K = 4, device map 0,1,0,1 (cuts 53, 103, 153)
  3-D 1M  k1, k2 (cut 15), k2s (cut 14)
  both    k1s  K = 1 with the rows uploaded in one fixed shuffled order (dump_state --shuffle-seed 1 in every
               run): a different particle layout and summation order with no decomposition - the control for a
               systematic K = 1 / K = 2 difference at round-off level (added after the main campaign, binned at
               the k2 cuts)
Before every run nvidia-smi must show no other compute process (step_trace_campaign.wait_for_idle_gpus, logged).

analyze: for every K > 1 arm X the 12 runs of k1 and X give 66 pairs: within K = 1 (15), within X (15) and across
(36). Each pair's id-matched rms difference (every run holds every particle, dumps sorted by global id) is taken per
bin of column distance to the nearest of X's cuts - the K = 1 runs binned at the same virtual seams, from one
reference run's positions, so a particle sits in the same bin for every pair - for velocity, acceleration, shift,
density, pressure and kernel sum (fluid particles). Per (case, horizon, X, field, bin): the three class medians,
across / within-K=1, and the exact permutation tests of nowait_audit over the 924 relabellings (cross: across vs
within pairs, a systematic difference; one: pairs with an X run vs within-K=1, any extra difference). Card swap:
the within-K=1 pairs split into GPU 0 x GPU 0, GPU 1 x GPU 1 and GPU 0 x GPU 1 (cross test over the 20 relabellings
of the six K = 1 runs; with 3 + 3 runs its smallest possible p is 0.1). Sensitivity: the velocity tests at d = 0
again after adding a synthetic seam defect to the X runs (the same in every X run, or different in each), of rms
f times the within-K=1 median. Writes OUT/ensemble.json, and with --docs the tables (e33_tables.md), the summary
(e33_summary.json) and the figure (fig_decomposition.png) there.

Usage (GPU, then CPU):
  .venv/Scripts/python.exe -m experiment.seam_audit.decomposition_audit run --out logs/e33/decomposition
  .venv/Scripts/python.exe -m experiment.seam_audit.decomposition_audit analyze --out logs/e33/decomposition \\
      --docs docs/seam_audit/e33
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from experiment.seam_audit.dump_state import dump_file_paths  # noqa: E402
from experiment.seam_audit.nowait_audit import (  # noqa: E402
    RUN_TIMEOUT_SECONDS,
    STALL_MARKERS,
    bin_masks,
    log_line,
    permutation_tests,
    result_line,
    seam_distance,
)

SNAPSHOT_2D = "logs/seam_audit/single_step/cavity2d_1m/snapshot_N2000.npz"
SNAPSHOT_3D = "logs/seam_audit/single_step/cavity3d_1m/snapshot_N2000.npz"
CASES = {
    "cavity2d_1m": {"case": "cases/lid_driven_cavity_2d_gen/case.yaml", "snapshot": SNAPSHOT_2D,
                    "arms": ("k1", "k2", "k2s", "k4", "k1s"), "label": "2-D 1M"},
    "cavity3d_1m": {"case": "cases/cavity3d_1m/case.yaml", "snapshot": SNAPSHOT_3D,
                    "arms": ("k1", "k2", "k2s", "k1s"), "label": "3-D 1M"},
}
SHUFFLE_SEED = 1                      # the k1s arm: one fixed alternative layout, as K = 2 is
VIRTUAL_CUT_ARM = {"k1s": "k2"}       # an arm without cuts is binned at this arm's cuts
# arm: (slabs, weights, device map; K = 1 alternates GPU 0 / GPU 1 by trial)
ARMS = {"k1": (1, None, None), "k2": (2, None, "0,1"), "k2s": (2, "0.9,1.1", "0,1"), "k4": (4, None, "0,1,0,1"),
        "k1s": (1, None, None)}
ARM_LABELS = {"k2": "K = 2", "k2s": "K = 2, cut shifted", "k4": "K = 4", "k1s": "K = 1, shuffled order"}
TRIALS = 6
HORIZONS = (300, 2000)
FIELDS = ("velocity", "acceleration", "shift", "density", "pressure", "kernel_sum")
TABLE_BINS = ("d0", "d1_3", "d4_7", "far", "all")
TABLE_BIN_LABELS = {"d0": "0", "d1_3": "1–3", "d4_7": "4–7", "far": "≥ 8", "all": "全部"}
FIGURE_BINS = (("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4–7", 4, 7), ("8–15", 8, 15), ("≥16", 16, None))
LID_VELOCITY = 1.0           # m/s, the U of the cavity
IDENTICAL_FIELDS = ("velocity", "density")   # share of particles bit-identical in both runs of a pair
IDENTICAL_BINS = ("d0", "all")
SENSITIVITY_FACTORS = (0.5, 1.0, 2.0)


def run_name(arm: str, trial: int) -> str:
    return f"k1g{(trial - 1) % 2}_t{trial}" if arm == "k1" else f"{arm}_t{trial}"


def run_order(arms) -> list:
    """(trial, arm) in execution order: every trial runs every arm, the order rotating by one per trial."""
    order = []
    for trial in range(1, TRIALS + 1):
        shift = (trial - 1) % len(arms)
        order += [(trial, arm) for arm in arms[shift:] + arms[:shift]]
    return order


def run_environment() -> dict:
    environment = {key: value for key, value in os.environ.items() if not key.startswith(("V5_", "V6_"))}
    environment.update({"VK_LOADER_LAYERS_DISABLE": "VK_LAYER_KHRONOS_validation", "PYTHONIOENCODING": "utf-8",
                        "PYTHONUNBUFFERED": "1", "V6_DELTA_DENSITY": "1"})
    return environment


def run_one(case_name: str, arm: str, trial: int, out_directory: pathlib.Path, campaign_log: pathlib.Path) -> dict:
    from experiment.v6.analysis.step_trace_campaign import wait_for_idle_gpus
    case = CASES[case_name]
    slabs, weights, device_map = ARMS[arm]
    if arm in ("k1", "k1s"):
        device_map = str((trial - 1) % 2)
    name = run_name(arm, trial)
    dump_directory = out_directory / "dumps" / case_name
    dump_directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "experiment.seam_audit.dump_state", "--version", "v6", "--case", case["case"],
               "--slabs", str(slabs), "--device-map", device_map, "--restart-snapshot", case["snapshot"],
               "--horizons", ",".join(str(horizon) for horizon in HORIZONS), "--depth", "2",
               "--out-dir", str(dump_directory), "--run-name", name]
    if weights:
        command += ["--weights", weights]
    if arm == "k1s":
        command += ["--shuffle-seed", str(SHUFFLE_SEED)]
    attempts = []
    summary = {}
    for attempt in (1, 2):
        gpu = wait_for_idle_gpus(lambda message: log_line(campaign_log, message))
        log_line(campaign_log, f"{case_name} {name} (slabs {slabs}, weights {weights or 'equal'}, device map "
                               f"{device_map}) attempt {attempt}: nvidia-smi idle after {gpu['waited_s']} s, "
                               f"gpus {gpu['gpus']}")
        log_path = out_directory / "logs" / f"{case_name}__{name}__a{attempt}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with open(log_path, "w", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=_REPOSITORY_ROOT, env=run_environment(), stdout=stream,
                                       stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=RUN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = -9
        summary = result_line(log_path.read_text(encoding="utf-8", errors="replace"))
        error = summary.get("error") or ""
        attempts.append({"attempt": attempt, "rc": code, "seconds": round(time.time() - started, 1),
                         "error": error, "log": str(log_path), "gpu_check": gpu})
        log_line(campaign_log, f"  -> exit {code} in {attempts[-1]['seconds']} s" + (f", error {error}" if error else ""))
        if not (code == 1 and any(marker in error for marker in STALL_MARKERS)):
            break
    record = {"case": case_name, "arm": arm, "trial": trial, "run": name, "device_map": device_map,
              "rc": attempts[-1]["rc"], "attempts": attempts, "result": summary,
              "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(out_directory / "results.jsonl", "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    return record


def run_campaign(out_directory: pathlib.Path, case_names) -> int:
    out_directory.mkdir(parents=True, exist_ok=True)
    campaign_log = out_directory / "campaign.log"
    done = {}
    for (case_name, name), record in last_records_by_run(out_directory / "results.jsonl").items():
        done[(case_name, name)] = record
    failures = 0
    for case_name, case in CASES.items():
        if case_names and case_name not in case_names:
            continue
        for trial, arm in run_order(case["arms"]):
            name = run_name(arm, trial)
            previous = done.get((case_name, name))
            dumps = [dump_file_paths(out_directory / "dumps" / case_name, name, horizon)["npz"] for horizon in HORIZONS]
            if previous and previous["rc"] == 0 and all(path.exists() for path in dumps):
                log_line(campaign_log, f"{case_name} {name}: done, skipped")
                continue
            failures += run_one(case_name, arm, trial, out_directory, campaign_log)["rc"] != 0
    log_line(campaign_log, f"campaign finished, {failures} failed run(s)")
    return 1 if failures else 0


def last_records_by_run(results_path: pathlib.Path) -> dict:
    records = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[(record["case"], record["run"])] = record
    return records


# ----------------------------------------------------------------------------- analysis

def figure_masks(distance: np.ndarray) -> dict:
    return {label: (distance >= low) & (distance <= high if high is not None else True)
            for label, low, high in FIGURE_BINS}


def runs_present(dump_directory: pathlib.Path, arm: str, horizon: int) -> list:
    names = [run_name(arm, trial) for trial in range(1, TRIALS + 1)]
    return [name for name in names if dump_file_paths(dump_directory, name, horizon)["npz"].exists()]


def class_of(first: str, second: str, test_runs: set) -> str:
    first_test, second_test = first in test_runs, second in test_runs
    if first_test and second_test:
        return "within_x"
    if first_test or second_test:
        return "cross"
    return "within_k1"


def card_of(name: str) -> str:
    return name[2:4] if name.startswith("k1g") else ""


def pair_rms(values: dict, first: str, second: str, masks: dict) -> dict:
    difference = values[first] - values[second]
    magnitude_squared = (np.einsum("ij,ij->i", difference, difference) if difference.ndim == 2
                         else difference * difference)
    return {label: float(np.sqrt(np.mean(magnitude_squared[mask]))) if mask.any() else None
            for label, mask in masks.items()}


def pair_identical(values: dict, first: str, second: str, masks: dict) -> dict:
    """Share of the bin's particles whose value (every component) is bit-identical in the two runs: how far a
    pair still is from leaving the round-off regime (two runs that never diverged share every bit)."""
    equal = values[first] == values[second]
    if equal.ndim == 2:
        equal = equal.all(axis=1)
    return {label: float(np.mean(equal[masks[label]])) if masks[label].any() else None for label in IDENTICAL_BINS}


def class_medians(pairs: dict, classify) -> dict:
    groups = {}
    for (first, second), value in pairs.items():
        if value is not None:
            groups.setdefault(classify(first, second), []).append(value)
    return {label: {"median": float(np.median(items)), "min": float(min(items)), "max": float(max(items)),
                    "pairs": len(items)} for label, items in groups.items()}


def summarize(run_names, test_runs, distances: dict) -> dict:
    """Class medians, ratios and the permutation tests of one (field, bin)."""
    groups = {"within_k1": [], "within_x": [], "cross": []}
    for (first, second), value in distances.items():
        groups[class_of(first, second, test_runs)].append(value)
    median = {label: float(np.median(items)) for label, items in groups.items() if items}
    tests = permutation_tests(run_names, distances, test_runs=test_runs)
    return {"median": median, "range": {label: [float(min(items)), float(max(items))] for label, items in groups.items() if items},
            "cross_over_within_k1": median["cross"] / median["within_k1"] if median.get("within_k1") else None,
            "within_x_over_within_k1": median["within_x"] / median["within_k1"] if median.get("within_k1") else None,
            "cross_p": tests["cross_p"], "one_p": tests["one_p"], "cross_statistic": tests["cross_statistic"],
            "one_statistic": tests["one_statistic"], "relabellings": tests["relabellings"]}


def sensitivity(values: dict, k1_runs, x_runs, mask: np.ndarray, within_k1_median: float, seed: int) -> list:
    """Velocity at d = 0 with a synthetic defect of per-particle rms f * within-K=1 median added to every X run:
    'systematic' (the same vectors in every X run) or 'random' (independent in each)."""
    rows = []
    run_names = list(k1_runs) + list(x_runs)
    selected = {name: values[name][mask] for name in run_names}
    dimension = next(iter(selected.values())).shape[1]
    for kind in ("systematic", "random"):
        for factor in SENSITIVITY_FACTORS:
            generator = np.random.default_rng(seed)
            sigma = factor * within_k1_median / np.sqrt(dimension)
            shared = generator.normal(0.0, sigma, size=next(iter(selected.values())).shape)
            perturbed = dict(selected)
            for name in x_runs:
                defect = shared if kind == "systematic" else generator.normal(0.0, sigma, size=shared.shape)
                perturbed[name] = selected[name] + defect
            distances = {}
            for first, second in itertools.combinations(run_names, 2):
                difference = perturbed[first] - perturbed[second]
                distances[(first, second)] = float(np.sqrt(np.mean(np.einsum("ij,ij->i", difference, difference))))
            entry = summarize(run_names, set(x_runs), distances)
            rows.append({"kind": kind, "factor": factor, "cross_over_within_k1": entry["cross_over_within_k1"],
                         "within_x_over_within_k1": entry["within_x_over_within_k1"], "cross_p": entry["cross_p"],
                         "one_p": entry["one_p"]})
    return rows


def analyze_case(case_name: str, out_directory: pathlib.Path) -> dict:
    dump_directory = out_directory / "dumps" / case_name
    arms = [arm for arm in CASES[case_name]["arms"] if arm != "k1"]
    result = {}
    for horizon in HORIZONS:
        k1_runs = runs_present(dump_directory, "k1", horizon)
        x_runs = {arm: runs_present(dump_directory, arm, horizon) for arm in arms}
        short = {arm: len(names) for arm, names in {"k1": k1_runs, **x_runs}.items() if len(names) < 2}
        if short:
            raise ValueError(f"{case_name} N={horizon}: arms with fewer than 2 dumps {short}")
        reference = k1_runs[0]
        sidecar = json.loads(dump_file_paths(dump_directory, reference, horizon)["json"].read_text(encoding="utf-8"))
        fluid_groups = [index for index, kind in enumerate(sidecar["material_kinds"]) if int(kind) == 0]
        with np.load(dump_file_paths(dump_directory, reference, horizon)["npz"]) as archive:
            identifiers = archive["id"]
            fluid = np.isin(archive["material"], fluid_groups)
            reference_x = archive["position"][:, 0].astype(np.float64)
        masks, fine, cuts = {}, {}, {}
        for arm in arms:
            cut_arm = VIRTUAL_CUT_ARM.get(arm, arm)
            arm_sidecars = [json.loads(dump_file_paths(dump_directory, name, horizon)["json"].read_text(encoding="utf-8"))
                            for name in x_runs[cut_arm]]
            arm_cuts = {tuple(item["cuts"]) for item in arm_sidecars}
            if len(arm_cuts) != 1:
                raise ValueError(f"{case_name} {arm} N={horizon}: runs disagree on the cuts {arm_cuts}")
            cuts[arm] = list(arm_cuts.pop())
            distance = seam_distance(reference_x, sidecar["origin_x"], sidecar["smoothing_length"], cuts[arm])[fluid]
            masks[arm] = bin_masks(distance)
            fine[arm] = figure_masks(distance)
        entry = {"k1_runs": k1_runs, "x_runs": x_runs, "cuts": cuts, "fluid": int(fluid.sum()),
                 "bin_counts": {arm: {label: int(mask.sum()) for label, mask in masks[arm].items()} for arm in arms},
                 "fields": {}, "figure": {}, "cards": {}, "sensitivity": {}, "invariants": {}, "identical": {},
                 "cards_identical": {}}
        for name in k1_runs + [name for arm in arms for name in x_runs[arm]]:
            invariants = json.loads(dump_file_paths(dump_directory, name, horizon)["json"].read_text(encoding="utf-8"))["invariants"]
            entry["invariants"][name] = {"valid": invariants["valid"], "far_migration_count": invariants.get("far_migration_count", 0),
                                         "drift": invariants["drift"], "overflow_total": invariants["overflow_total"],
                                         "stamp_errors_gpu": invariants["stamp_errors_gpu"],
                                         "stamp_errors_host": invariants["stamp_errors_host"]}
        invalid = [name for name, item in entry["invariants"].items() if not item["valid"] or item["far_migration_count"]]
        if invalid:
            raise ValueError(f"{case_name} N={horizon}: invalid runs (an invariant or far migration not 0): {invalid}")
        for field in FIELDS:
            values = {}
            for name in k1_runs + [name for arm in arms for name in x_runs[arm]]:
                with np.load(dump_file_paths(dump_directory, name, horizon)["npz"]) as archive:
                    if not np.array_equal(archive["id"], identifiers):
                        raise ValueError(f"{case_name} {name} N={horizon}: ids differ from {reference}")
                    values[name] = archive[field][fluid].astype(np.float64)
            # every pair once: a within-K=1 pair serves every arm (binned at that arm's seams)
            table = {arm: {} for arm in arms}
            figure = {arm: {} for arm in arms}
            same = {arm: {} for arm in arms}
            for first, second in itertools.combinations(k1_runs, 2):
                for arm in arms:
                    table[arm][(first, second)] = pair_rms(values, first, second, masks[arm])
                    if field == "velocity":
                        figure[arm][(first, second)] = pair_rms(values, first, second, fine[arm])
                    if field in IDENTICAL_FIELDS:
                        same[arm][(first, second)] = pair_identical(values, first, second, masks[arm])
            for arm in arms:
                for first, second in itertools.combinations(k1_runs + x_runs[arm], 2):
                    if first in k1_runs and second in k1_runs:
                        continue
                    table[arm][(first, second)] = pair_rms(values, first, second, masks[arm])
                    if field == "velocity":
                        figure[arm][(first, second)] = pair_rms(values, first, second, fine[arm])
                    if field in IDENTICAL_FIELDS:
                        same[arm][(first, second)] = pair_identical(values, first, second, masks[arm])
            if field in IDENTICAL_FIELDS:
                entry["identical"][field] = {
                    arm: {label: class_medians({pair: shares[label] for pair, shares in same[arm].items()},
                                               lambda first, second, arm=arm: class_of(first, second, set(x_runs[arm])))
                          for label in IDENTICAL_BINS} for arm in arms}
                entry["cards_identical"][field] = {
                    label: class_medians({pair: shares[label] for pair, shares in same["k2"].items()
                                          if pair[0] in k1_runs and pair[1] in k1_runs},
                                         lambda first, second: "x".join(sorted((card_of(first), card_of(second)))))
                    for label in IDENTICAL_BINS}
            entry["fields"][field] = {}
            for arm in arms:
                run_names = k1_runs + x_runs[arm]
                per_bin = {}
                for label in TABLE_BINS:
                    distances = {pair: bins[label] for pair, bins in table[arm].items() if bins[label]}
                    per_bin[label] = summarize(run_names, set(x_runs[arm]), distances)
                    per_bin[label]["pairs"] = {f"{first}-{second}": value for (first, second), value in distances.items()}
                entry["fields"][field][arm] = per_bin
                if field == "velocity":
                    figure_bins = {}
                    for label, _, _ in FIGURE_BINS:
                        distances = {pair: bins[label] for pair, bins in figure[arm].items() if bins[label]}
                        if len(distances) == len(list(itertools.combinations(run_names, 2))):
                            figure_bins[label] = summarize(run_names, set(x_runs[arm]), distances)
                            figure_bins[label]["pairs"] = {f"{first}-{second}": value
                                                           for (first, second), value in distances.items()}
                    entry["figure"][arm] = figure_bins
                    seed = 1000 * HORIZONS.index(horizon) + 10 * arms.index(arm) + (0 if case_name.startswith("cavity2d") else 5)
                    entry["sensitivity"][arm] = sensitivity(values, k1_runs, x_runs[arm], masks[arm]["d0"],
                                                            per_bin["d0"]["median"]["within_k1"], seed)
            # card swap: the within-K=1 pairs at the K = 2 (equal) arm's virtual seams
            entry["cards"][field] = {}
            for label in TABLE_BINS:
                distances = {pair: bins[label] for pair, bins in table["k2"].items()
                             if pair[0] in k1_runs and pair[1] in k1_runs and bins[label]}
                groups = {}
                for (first, second), value in distances.items():
                    key = "x".join(sorted((card_of(first), card_of(second))))
                    groups.setdefault(key, []).append(value)
                tests = permutation_tests(k1_runs, distances, test_runs=[name for name in k1_runs if card_of(name) == "g1"])
                median = {key: float(np.median(items)) for key, items in groups.items()}
                same = float(np.median(groups.get("g0xg0", []) + groups.get("g1xg1", [])))
                entry["cards"][field][label] = {"median": median, "pairs": {key: len(items) for key, items in groups.items()},
                                                "across_over_same": median.get("g0xg1", float("nan")) / same if same else None,
                                                "cross_p": tests["cross_p"], "relabellings": tests["relabellings"]}
            del values
        result[str(horizon)] = entry
        print(f"[decomposition_audit] {case_name} N={horizon}: {len(k1_runs)} K = 1 runs, "
              + ", ".join(f"{arm} {len(x_runs[arm])} (cuts {cuts[arm]})" for arm in arms), flush=True)
    return result


# ----------------------------------------------------------------------------- tables and figure

def markdown_tables(results: dict) -> str:
    lines = ["## velocity: class medians per bin", "",
             "| 算例 | N | 臂 | 箱 | K=1 内 | K>1 内 | 跨 K | 跨 K / K=1 内 | K>1 内 / K=1 内 | p 交叉 | p 含 K>1 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for case_name, horizons in results.items():
        for horizon, entry in horizons.items():
            for arm, bins in entry["fields"]["velocity"].items():
                for label in TABLE_BINS:
                    item = bins[label]
                    median = item["median"]
                    lines.append(f"| {CASES[case_name]['label']} | {horizon} | {ARM_LABELS[arm]} | {TABLE_BIN_LABELS[label]} | "
                                 f"{median['within_k1']:.3g} | {median['within_x']:.3g} | {median['cross']:.3g} | "
                                 f"{item['cross_over_within_k1']:.2f} | {item['within_x_over_within_k1']:.2f} | "
                                 f"{item['cross_p']:.3f} | {item['one_p']:.3f} |")
    lines += ["", "## every field: across / within-K=1 median ratio per bin (d = 0 / 1–3 / 4–7 / ≥ 8 / all)", "",
              "| 算例 | N | 臂 | 速度 | 加速度 | shift | ρ | p | kernel sum |", "|---|---|---|---|---|---|---|---|---|"]
    for case_name, horizons in results.items():
        for horizon, entry in horizons.items():
            for arm in entry["fields"]["velocity"]:
                cells = []
                for field in FIELDS:
                    bins = entry["fields"][field][arm]
                    cells.append(" / ".join(f"{bins[label]['cross_over_within_k1']:.2f}" for label in TABLE_BINS))
                lines.append(f"| {CASES[case_name]['label']} | {horizon} | {ARM_LABELS[arm]} | " + " | ".join(cells) + " |")
    lines += ["", "## permutation tests: p < 0.05 counts (6 fields x 5 bins = 30 tests per row and statistic)", "",
              "| 算例 | N | 臂 | 交叉 p < 0.05 | 含 K>1 p < 0.05 | 最小 p 交叉 | 最小 p 含 K>1 |", "|---|---|---|---|---|---|---|"]
    total_tests = total_cross = total_one = 0
    for case_name, horizons in results.items():
        for horizon, entry in horizons.items():
            for arm in entry["fields"]["velocity"]:
                items = [entry["fields"][field][arm][label] for field in FIELDS for label in TABLE_BINS]
                cross = sum(item["cross_p"] < 0.05 for item in items)
                one = sum(item["one_p"] < 0.05 for item in items)
                total_tests += len(items)
                total_cross += cross
                total_one += one
                lines.append(f"| {CASES[case_name]['label']} | {horizon} | {ARM_LABELS[arm]} | {cross} | {one} | "
                             f"{min(item['cross_p'] for item in items):.3f} | {min(item['one_p'] for item in items):.3f} |")
    lines.append(f"| 合计 | | | {total_cross} / {total_tests} | {total_one} / {total_tests} | 机会期望 "
                 f"{0.05 * total_tests:.0f} | |")
    lines += ["", "## card swap: within-K=1 velocity medians (pairs at the K = 2 virtual seams)", "",
              "| 算例 | N | 箱 | GPU 0 x GPU 0 | GPU 1 x GPU 1 | GPU 0 x GPU 1 | 跨卡 / 同卡 | p 交叉(20 种重标,最小 0.1) |",
              "|---|---|---|---|---|---|---|---|"]
    for case_name, horizons in results.items():
        for horizon, entry in horizons.items():
            for label in TABLE_BINS:
                item = entry["cards"]["velocity"][label]
                median = item["median"]
                lines.append(f"| {CASES[case_name]['label']} | {horizon} | {TABLE_BIN_LABELS[label]} | "
                             f"{median.get('g0xg0', float('nan')):.3g} | {median.get('g1xg1', float('nan')):.3g} | "
                             f"{median.get('g0xg1', float('nan')):.3g} | {item['across_over_same']:.2f} | {item['cross_p']:.2f} |")
    lines += ["", "## bit-identical share of fluid particles per pair (median of the class; velocity / density)", "",
              "| 算例 | N | 臂 | 箱 | K=1 内 | K>1 内 | 跨 K |", "|---|---|---|---|---|---|---|"]
    for case_name, horizons in results.items():
        for horizon, entry in horizons.items():
            for arm in entry["identical"]["velocity"]:
                for label in IDENTICAL_BINS:
                    cells = []
                    for kind in ("within_k1", "within_x", "cross"):
                        cells.append(" / ".join(f"{entry['identical'][field][arm][label][kind]['median']:.3f}"
                                                for field in IDENTICAL_FIELDS))
                    lines.append(f"| {CASES[case_name]['label']} | {horizon} | {ARM_LABELS[arm]} | {TABLE_BIN_LABELS[label]} | "
                                 + " | ".join(cells) + " |")
    lines += ["", "## card swap: bit-identical share within K = 1 (velocity / density, all fluid particles)", "",
              "| 算例 | N | GPU 0 x GPU 0 | GPU 1 x GPU 1 | GPU 0 x GPU 1 |", "|---|---|---|---|---|"]
    for case_name, horizons in results.items():
        for horizon, entry in horizons.items():
            cells = []
            for kind in ("g0xg0", "g1xg1", "g0xg1"):
                cells.append(" / ".join(f"{entry['cards_identical'][field]['all'][kind]['median']:.3f}"
                                        for field in IDENTICAL_FIELDS))
            lines.append(f"| {CASES[case_name]['label']} | {horizon} | " + " | ".join(cells) + " |")
    lines += ["", "## d = 0 velocity difference relative to the lid speed U = 1 m/s (medians)", "",
              "| 算例 | N | 臂 | K=1 内 / U | 跨 K / U | 全部粒子:跨 K / U |", "|---|---|---|---|---|---|"]
    for case_name, horizons in results.items():
        for horizon, entry in horizons.items():
            for arm, bins in entry["fields"]["velocity"].items():
                lines.append(f"| {CASES[case_name]['label']} | {horizon} | {ARM_LABELS[arm]} | "
                             f"{bins['d0']['median']['within_k1'] / LID_VELOCITY:.2g} | "
                             f"{bins['d0']['median']['cross'] / LID_VELOCITY:.2g} | {bins['all']['median']['cross'] / LID_VELOCITY:.2g} |")
    lines += ["", "## sensitivity: velocity at d = 0 with a synthetic seam defect in the K > 1 runs", "",
              "| 算例 | N | 臂 | 缺陷 | f | 跨 K / K=1 内 | K>1 内 / K=1 内 | p 交叉 | p 含 K>1 |", "|---|---|---|---|---|---|---|---|---|"]
    for case_name, horizons in results.items():
        for horizon, entry in horizons.items():
            for arm, rows in entry["sensitivity"].items():
                for row in rows:
                    lines.append(f"| {CASES[case_name]['label']} | {horizon} | {ARM_LABELS[arm]} | {row['kind']} | {row['factor']:g} | "
                                 f"{row['cross_over_within_k1']:.2f} | {row['within_x_over_within_k1']:.2f} | "
                                 f"{row['cross_p']:.3f} | {row['one_p']:.3f} |")
    lines += ["", "## every field and bin (class medians: within K=1 / within K>1 / across; p cross / p one)", "",
              "| 算例 | N | 臂 | 场 | 箱 | K=1 内 | K>1 内 | 跨 K | 跨 / K=1 内 | p 交叉 | p 含 K>1 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for case_name, horizons in results.items():
        for horizon, entry in horizons.items():
            for field in FIELDS:
                for arm, bins in entry["fields"][field].items():
                    for label in TABLE_BINS:
                        item = bins[label]
                        median = item["median"]
                        lines.append(f"| {CASES[case_name]['label']} | {horizon} | {ARM_LABELS[arm]} | {field} | "
                                     f"{TABLE_BIN_LABELS[label]} | {median['within_k1']:.3g} | {median['within_x']:.3g} | "
                                     f"{median['cross']:.3g} | {item['cross_over_within_k1']:.2f} | {item['cross_p']:.3f} | "
                                     f"{item['one_p']:.3f} |")
    return "\n".join(lines) + "\n"


def figure(results: dict, path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    columns = [(case_name, arm) for case_name, case in CASES.items() if case_name in results
               for arm in case["arms"] if arm != "k1"]
    if not columns:
        return
    figure_handle, axes = plt.subplots(len(HORIZONS), len(columns), figsize=(3.3 * len(columns), 3.4 * len(HORIZONS)),
                                       sharey="row", squeeze=False)
    styles = {"within_k1": ("o", "#1f77b4", "K = 1 vs K = 1"), "within_x": ("^", "#d62728", "K > 1 vs K > 1"),
              "cross": ("x", "#555555", "K = 1 vs K > 1")}
    offsets = {"within_k1": -0.22, "cross": 0.0, "within_x": 0.22}
    jitter = np.random.default_rng(33)
    for row, horizon in enumerate(HORIZONS):
        for column, (case_name, arm) in enumerate(columns):
            axis = axes[row][column]
            entry = results[case_name][str(horizon)]
            bins = entry["figure"][arm]
            test_runs = set(entry["x_runs"][arm])
            labels = [label for label, _, _ in FIGURE_BINS if label in bins]
            for position, label in enumerate(labels):
                for pair, value in bins[label]["pairs"].items():
                    first, second = pair.split("-")
                    kind = class_of(first, second, test_runs)
                    marker, color, _ = styles[kind]
                    axis.scatter(position + offsets[kind] + jitter.uniform(-0.06, 0.06), value, marker=marker, s=11,
                                 color=color, alpha=0.8, linewidths=0.8)
            axis.set_yscale("log")
            for position, label in enumerate(labels):    # x in data, y in axes coordinates: above the points
                axis.text(position, 0.99, f"×{bins[label]['cross_p']:.2f}\n+{bins[label]['one_p']:.2f}", ha="center",
                          va="top", fontsize=5.6, color="#333333", transform=axis.get_xaxis_transform())
            axis.set_xticks(range(len(labels)))
            axis.set_xticklabels(labels, fontsize=7)
            if row == len(HORIZONS) - 1:
                axis.set_xlabel("column distance to the nearest seam", fontsize=7)
            if column == 0:
                axis.set_ylabel(f"N = {horizon}: rms |Δv| between two runs (m/s)", fontsize=7)
            if row == 0:
                axis.set_title(f"{CASES[case_name]['label']}, {ARM_LABELS[arm]}\ncuts {entry['cuts'][arm]}", fontsize=8)
            axis.grid(alpha=0.3, which="both")
            axis.tick_params(labelsize=7)
        low, high = axes[row][0].get_ylim()          # the row shares y: headroom for the p labels
        axes[row][0].set_ylim(low, high * 12)
    handles = [plt.Line2D([], [], marker=marker, color=color, linestyle="", label=label)
               for marker, color, label in styles.values()]
    figure_handle.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    figure_handle.suptitle("E33: id-matched velocity difference of every pair of runs (6 K = 1 + 6 K > 1 per column); "
                           "p of the exact permutation tests at the top of each bin (× cross, + pairs with a K > 1 run)",
                           fontsize=8.5)
    figure_handle.tight_layout(rect=(0, 0.04, 1, 0.97))
    figure_handle.savefig(path, dpi=160)
    plt.close(figure_handle)


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def analyze(out_directory: pathlib.Path, docs_directory, case_names) -> int:
    results = {}
    for case_name in CASES:
        if case_names and case_name not in case_names:
            continue
        results[case_name] = analyze_case(case_name, out_directory)
    (out_directory / "ensemble.json").write_text(json.dumps(json_ready(results), indent=1), encoding="utf-8")
    tables = markdown_tables(results)
    (out_directory / "ensemble.md").write_text(tables, encoding="utf-8")
    if docs_directory:
        docs = pathlib.Path(docs_directory)
        docs.mkdir(parents=True, exist_ok=True)
        (docs / "e33_tables.md").write_text(tables, encoding="utf-8")
        summary = {case_name: {horizon: {key: value for key, value in entry.items() if key not in ("fields", "figure")}
                               | {"fields": {field: {arm: {label: {key: value for key, value in item.items() if key != "pairs"}
                                                          for label, item in bins.items()}
                                                    for arm, bins in arms.items()}
                                             for field, arms in entry["fields"].items()}}
                               for horizon, entry in horizons.items()}
                   for case_name, horizons in results.items()}
        (docs / "e33_summary.json").write_text(json.dumps(json_ready(summary), indent=1), encoding="utf-8")
        figure(results, docs / "fig_decomposition.png")
        print(f"[decomposition_audit] wrote {docs / 'e33_tables.md'}, e33_summary.json, fig_decomposition.png", flush=True)
    print(f"[decomposition_audit] wrote {out_directory / 'ensemble.json'} and ensemble.md", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=("run", "analyze"))
    parser.add_argument("--out", default="logs/e33/decomposition")
    parser.add_argument("--docs", default=None, help="analyze: also write tables, summary and figure here")
    parser.add_argument("--cases", default="", help="comma list of case names (default: both)")
    arguments = parser.parse_args()
    names = [name for name in arguments.cases.split(",") if name]
    out_directory = pathlib.Path(arguments.out)
    if arguments.mode == "run":
        return run_campaign(out_directory, names)
    return analyze(out_directory, arguments.docs, names)


if __name__ == "__main__":
    sys.exit(main())
