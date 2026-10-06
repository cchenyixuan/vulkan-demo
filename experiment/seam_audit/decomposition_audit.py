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
of the six K = 1 runs; with 3 + 3 runs its smallest possible p is 0.1), and the same split of the control arm,
which alternates the GPUs the same way. Sensitivity: the velocity tests at d = 0 again after adding a synthetic
seam defect to the X runs (the same in every X run, or different in each), of rms f times the within-K=1 median.
Because the 30 tests of a configuration (6 fields x 5 bins) are strongly dependent, their p < 0.05 count is judged
against the joint permutation null (every relabelling applied to all 30 at once), also with the d = 0 velocity test
replaced by each synthetic defect (how the configuration-level verdict reacts to a seam defect); each ratio also
gets its chance range (2.5-97.5 % over the relabellings), and each test the between-group term D^2 / s_1^2 of the
squared distances (group offset over K = 1 run-to-run scatter, unmoved by a tighter or wider arm) with its
permutation p. Diagnostics: the 10 trio splits of every arm's six runs, the within-arm medians in the two columns
at each cut, and the audit gate's one-pair floor among the K = 1 runs. Writes
OUT/ensemble.json, and with --docs the tables (e33_tables.md), the summary (e33_summary.json) and the figure
(fig_decomposition.png) there.

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
    AUDIT_LIMIT,
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
TRIO_BINS = ("d0", "all")         # the 3 + 3 splits of each arm's six runs
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
    """The GPU of a K = 1 run: k1 and k1s alternate GPU 0 / GPU 1 from trial to trial."""
    if name.startswith("k1g"):
        return name[2:4]
    if name.startswith("k1s_t"):
        return f"g{(int(name.split('_t')[1]) - 1) % 2}"
    return ""


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


def relabelling_statistics(run_names, test_runs, distances: dict) -> dict:
    """The statistics of nowait_audit.permutation_tests under every relabelling (each choice of len(test_runs) of
    the runs as the tested arm, in itertools.combinations order), vectorised: cross, one, the across / within-K=1
    and within-X / within-K=1 median ratios (within-K=1 = the pairs of the runs NOT chosen), and the between-group
    term of the squared distances, between = (mean across d^2 - (mean within-X d^2 + mean within-K=1 d^2) / 2) /
    (mean within-K=1 d^2 / 2): with run-to-run scatter s_X, s_1 and a group offset D, the means are s_X^2 + s_1^2 + D^2,
    2 s_X^2 and 2 s_1^2, so between = D^2 / s_1^2, an offset estimate that a tighter or wider X arm alone does not
    move (the cross statistic, a difference of mean logs, also rises a little when only the scatter differs).
    'observed' indexes the actual labelling."""
    names = list(run_names)
    index = {name: position for position, name in enumerate(names)}
    pairs = [pair for pair, value in distances.items() if value > 0]
    values = np.array([distances[pair] for pair in pairs])
    logs = np.log(values)
    choices = list(itertools.combinations(range(len(names)), len(test_runs)))
    member = np.zeros((len(choices), len(names)), dtype=bool)
    for row, choice in enumerate(choices):
        member[row, list(choice)] = True
    first = member[:, [index[pair[0]] for pair in pairs]]
    second = member[:, [index[pair[1]] for pair in pairs]]
    cross = first != second
    with_one = first | second

    within_x = first & second
    within_k1 = ~with_one

    def mean_log(mask):
        return (mask * logs).sum(axis=1) / mask.sum(axis=1)

    def mean_square(mask):
        return (mask * values * values).sum(axis=1) / mask.sum(axis=1)
    ratio = np.array([np.median(values[cross[row]]) / np.median(values[within_k1[row]]) for row in range(len(choices))])
    within_ratio = np.array([np.median(values[within_x[row]]) / np.median(values[within_k1[row]])
                             for row in range(len(choices))])
    square_k1 = mean_square(within_k1)
    between = (mean_square(cross) - (mean_square(within_x) + square_k1) / 2) / (square_k1 / 2)
    return {"cross": mean_log(cross) - mean_log(~cross), "one": mean_log(with_one) - mean_log(within_k1), "ratio": ratio,
            "within_ratio": within_ratio, "between": between,
            "observed": choices.index(tuple(sorted(index[name] for name in test_runs)))}


def relabelled_p(statistic: np.ndarray) -> np.ndarray:
    """The one-sided p of every relabelling against all of them (as permutation_tests, tolerance 1e-12)."""
    ordered = np.sort(statistic)
    return (statistic.size - np.searchsorted(ordered, statistic - 1e-12, side="left")) / statistic.size


def summarize(run_names, test_runs, distances: dict) -> dict:
    """Class medians, ratios and the permutation tests of one (field, bin); ratio_null_95 = the 2.5-97.5 % range of
    the across / within-K=1 median ratio over the relabellings (what chance alone gives with these runs)."""
    groups = {"within_k1": [], "within_x": [], "cross": []}
    for (first, second), value in distances.items():
        groups[class_of(first, second, test_runs)].append(value)
    median = {label: float(np.median(items)) for label, items in groups.items() if items}
    tests = permutation_tests(run_names, distances, test_runs=test_runs)
    relabelled = relabelling_statistics(run_names, test_runs, distances)
    for kind in ("cross", "one"):      # the vectorised statistics must reproduce the reference test exactly
        if relabelled_p(relabelled[kind])[relabelled["observed"]] != tests[f"{kind}_p"]:
            raise ValueError(f"relabelling_statistics disagrees with permutation_tests ({kind})")
    return {"median": median, "range": {label: [float(min(items)), float(max(items))] for label, items in groups.items() if items},
            "cross_over_within_k1": median["cross"] / median["within_k1"] if median.get("within_k1") else None,
            "within_x_over_within_k1": median["within_x"] / median["within_k1"] if median.get("within_k1") else None,
            "ratio_null_95": [float(value) for value in np.percentile(relabelled["ratio"], (2.5, 97.5))],
            "within_ratio_null_95": [float(value) for value in np.percentile(relabelled["within_ratio"], (2.5, 97.5))],
            "between": float(relabelled["between"][relabelled["observed"]]),
            "between_p": float(relabelled_p(relabelled["between"])[relabelled["observed"]]),
            "cross_p": tests["cross_p"], "one_p": tests["one_p"], "cross_statistic": tests["cross_statistic"],
            "one_statistic": tests["one_statistic"], "relabellings": tests["relabellings"]}


def joint_null(run_names, test_runs, tests: dict, scenarios: dict = None) -> dict:
    """How many of a configuration's tests (every field x bin) reach p < 0.05, against the joint permutation null:
    each relabelling is applied to all tests at once, which keeps their dependence (density and pressure are almost
    the same test, in 2-D the far bin is almost every particle). tests: {(field, bin): distances}.
    P = share of relabellings with at least the observed count; P(0) = share with none; p_none_both = share with
    none of the tests significant in either statistic. scenarios: {name: {(field, bin): distances}} recompute the
    verdict with those tests replaced (the synthetic seam defects of sensitivity)."""
    statistics = {key: relabelling_statistics(run_names, test_runs, distances) for key, distances in tests.items()}
    out = joint_counts(statistics)
    if scenarios:
        out["scenarios"] = {}
        for name, replacement in scenarios.items():
            changed = dict(statistics)
            for key, distances in replacement.items():
                changed[key] = relabelling_statistics(run_names, test_runs, distances)
            out["scenarios"][name] = joint_counts(changed)
    return out


def joint_counts(statistics: dict) -> dict:
    out = {}
    observed = next(iter(statistics.values()))["observed"]
    total = 0
    for kind in ("cross", "one"):
        counts = np.sum([relabelled_p(item[kind]) < 0.05 for item in statistics.values()], axis=0)
        total = total + counts
        out[kind] = {"count": int(counts[observed]), "tests": len(statistics),
                     "p_at_least": float(np.mean(counts >= counts[observed])), "p_none": float(np.mean(counts == 0)),
                     "null_mean_count": float(np.mean(counts))}
    out["p_none_both"] = float(np.mean(total == 0))
    return out


def trio_splits(runs, distances: dict) -> dict:
    """The 10 ways to split 6 runs into two trios: the median of each trio's 3 within pairs and their ratio
    (larger / smaller). 'parity' = trials 1, 3, 5 against 2, 4, 6, for K = 1 the GPU 0 / GPU 1 split."""
    def distance(first, second):
        return distances.get((first, second), distances.get((second, first)))
    odd = {name for name in runs if int(name.split("_t")[1]) % 2 == 1}
    rows, parity = [], None
    for trio in itertools.combinations(runs[1:], 2):
        first_trio = [runs[0], *trio]
        second_trio = [name for name in runs if name not in first_trio]
        medians = [float(np.median([distance(*pair) for pair in itertools.combinations(group, 2)]))
                   for group in (first_trio, second_trio)]
        rows.append({"trios": [first_trio, second_trio], "medians": medians, "ratio": max(medians) / min(medians)})
        if odd in (set(first_trio), set(second_trio)):
            odd_median, even_median = medians if set(first_trio) == odd else medians[::-1]
            parity = {"odd": odd_median, "even": even_median, "ratio": rows[-1]["ratio"]}
            rows[-1]["parity"] = True
    return {"splits": sorted(rows, key=lambda row: row["ratio"]), "parity": parity,
            "other_ratios": sorted(row["ratio"] for row in rows if not row.get("parity"))}


def sensitivity(values: dict, k1_runs, x_runs, mask: np.ndarray, within_k1_median: float, seed: int) -> tuple:
    """Velocity at d = 0 with a synthetic defect of per-particle rms f * within-K=1 median added to every X run:
    'systematic' (the same vectors in every X run) or 'random' (independent in each). One draw per kind (the
    generator is re-seeded), f scales it. Returns the rows and {scenario name: perturbed distances}, so that the
    configuration-level verdict can be recomputed with this one test replaced (joint_null scenarios)."""
    rows, perturbed_distances = [], {}
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
                         "one_p": entry["one_p"], "between": entry["between"], "between_p": entry["between_p"]})
            perturbed_distances[scenario_name(kind, factor)] = distances
    return rows, perturbed_distances


def scenario_name(kind: str, factor: float) -> str:
    return f"{kind}_{factor:g}"


def gate_null(runs, distances: dict, limit: float = AUDIT_LIMIT) -> dict:
    """The audit gate's primary ratio d(B1, A1) / d(A1, A2) over every ordered triple of distinct K = 1 runs
    (6 x 5 x 4 = 120): how often identical settings fail a one-pair floor (E32 part 1 did this for K > 1 runs)."""
    def distance(first, second):
        return distances.get((first, second), distances.get((second, first)))
    ratios = np.array([distance(test, first) / distance(first, second)
                       for first, second, test in itertools.permutations(runs, 3)])
    return {"triples": int(ratios.size), "fail_fraction": float(np.mean(ratios > limit)),
            "median_ratio": float(np.median(ratios))}


def card_split(runs, distances: dict) -> dict:
    """Pairs of six K = 1 runs by GPU (g0xg0, g1xg1, g0xg1): medians, across / same, and the cross test of the
    GPU 1 runs against the GPU 0 runs over the 20 relabellings (with 3 + 3 runs the smallest possible p is 0.1)."""
    groups = {}
    for (first, second), value in distances.items():
        groups.setdefault("x".join(sorted((card_of(first), card_of(second)))), []).append(value)
    tests = permutation_tests(runs, distances, test_runs=[name for name in runs if card_of(name) == "g1"])
    median = {key: float(np.median(items)) for key, items in groups.items()}
    same = float(np.median(groups.get("g0xg0", []) + groups.get("g1xg1", [])))
    return {"median": median, "pairs": {key: len(items) for key, items in groups.items()},
            "across_over_same": median.get("g0xg1", float("nan")) / same if same else None,
            "cross_p": tests["cross_p"], "relabellings": tests["relabellings"]}


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
        # the two columns on either side of every cut of any arm (d = 0 of that cut alone): is a column pair
        # noisier for every arm, seam or not?
        column_masks = {cut: seam_distance(reference_x, sidecar["origin_x"], sidecar["smoothing_length"], [cut])[fluid] == 0
                        for cut in sorted({cut for arm in arms for cut in cuts[arm]})}
        entry = {"k1_runs": k1_runs, "x_runs": x_runs, "cuts": cuts, "fluid": int(fluid.sum()),
                 "bin_counts": {arm: {label: int(mask.sum()) for label, mask in masks[arm].items()} for arm in arms},
                 "fields": {}, "figure": {}, "cards": {}, "sensitivity": {}, "invariants": {}, "identical": {},
                 "cards_identical": {}, "cards_control": {}, "joint": {}, "trios": {}, "seam_columns": {},
                 "gate_null": {}}
        joint_inputs = {arm: {} for arm in arms}
        defect_inputs = {arm: {} for arm in arms}
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
                    joint_inputs[arm][(field, label)] = distances
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
                    entry["sensitivity"][arm], defect_inputs[arm] = sensitivity(values, k1_runs, x_runs[arm], masks[arm]["d0"],
                                                            per_bin["d0"]["median"]["within_k1"], seed)
            # card swap: the within-K=1 pairs at the K = 2 (equal) arm's virtual seams; the control arm alternates the
            # GPUs the same way (binned at the same cuts)
            entry["cards"][field] = {label: card_split(k1_runs, {pair: bins[label] for pair, bins in table["k2"].items()
                                                                 if pair[0] in k1_runs and pair[1] in k1_runs and bins[label]})
                                     for label in TABLE_BINS}
            entry["gate_null"][field] = {label: gate_null(k1_runs, {pair: bins[label] for pair, bins in table["k2"].items()
                                                                    if pair[0] in k1_runs and pair[1] in k1_runs})
                                         for label in ("d0", "all")}
            if "k1s" in arms:
                control = x_runs["k1s"]
                entry["cards_control"][field] = {label: card_split(control, {pair: bins[label] for pair, bins in table["k1s"].items()
                                                                             if pair[0] in control and pair[1] in control and bins[label]})
                                                 for label in TABLE_BINS}
            if field == "velocity":
                for arm, runs, source in [("k1", k1_runs, "k2")] + [(arm, x_runs[arm], arm) for arm in arms]:
                    if len(runs) == TRIALS:
                        entry["trios"][arm] = {label: trio_splits(runs, {pair: bins[label] for pair, bins in table[source].items()
                                                                         if pair[0] in runs and pair[1] in runs})
                                               for label in TRIO_BINS}
                for cut, mask in column_masks.items():
                    within = {}
                    for arm, runs in [("k1", k1_runs)] + [(arm, x_runs[arm]) for arm in arms]:
                        distances = {f"{first}-{second}": pair_rms(values, first, second, {"band": mask})["band"]
                                     for first, second in itertools.combinations(runs, 2)}
                        within[arm] = {"median": float(np.median(list(distances.values()))),
                                       "range": [min(distances.values()), max(distances.values())], "pairs": distances}
                    entry["seam_columns"][str(cut)] = {"columns": [cut - 1, cut], "fluid": int(mask.sum()), "within": within}
            del values
        for arm in arms:
            entry["joint"][arm] = joint_null(k1_runs + x_runs[arm], x_runs[arm], joint_inputs[arm],
                                             {name: {("velocity", "d0"): distances}
                                              for name, distances in defect_inputs[arm].items()})
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
              "P = the joint permutation null: each of the 924 relabellings applied to all 30 tests at once (keeps their "
              "dependence: ρ and p are almost the same test, in 2-D the ≥ 8 bin is almost every particle); P(≥) = share of "
              "relabellings with at least the observed count, P(0) = share with none, null mean = expected count; "
              "P(0, both) = share with none of the 60 tests of both statistics significant.", "",
              "| 算例 | N | 臂 | 交叉 p < 0.05 | 含 K>1 p < 0.05 | 最小 p 交叉 | 最小 p 含 K>1 | 交叉 P(≥) / P(0) / null mean | "
              "含 K>1 P(≥) / P(0) / null mean | P(0, both) |", "|---|---|---|---|---|---|---|---|---|---|"]
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
                joint = entry["joint"][arm]
                lines.append(f"| {CASES[case_name]['label']} | {horizon} | {ARM_LABELS[arm]} | {cross} | {one} | "
                             f"{min(item['cross_p'] for item in items):.3f} | {min(item['one_p'] for item in items):.3f} | "
                             + " | ".join(f"{joint[kind]['p_at_least']:.3f} / {joint[kind]['p_none']:.2f} / "
                                          f"{joint[kind]['null_mean_count']:.1f}" for kind in ("cross", "one"))
                             + f" | {joint['p_none_both']:.2f} |")
    lines.append(f"| 合计 | | | {total_cross} / {total_tests} | {total_one} / {total_tests} | 机会期望 "
                 f"{0.05 * total_tests:.0f}(检验独立时) | | | | |")
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
    for title, key in (("card swap, K = 1, every field (pairs at the K = 2 virtual seams)", "cards"),
                       ("card swap, control: the six shuffled K = 1 runs (GPU 0 in trials 1 / 3 / 5, GPU 1 in 2 / 4 / 6, "
                        "binned at the same K = 2 cuts)", "cards_control")):
        lines += ["", f"## {title}", "",
                  "| 算例 | N | 场 | 箱 | GPU 0 x GPU 0 | GPU 1 x GPU 1 | GPU 0 x GPU 1 | 跨卡 / 同卡 | p 交叉 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for case_name, horizons in results.items():
            for horizon, entry in horizons.items():
                for field in FIELDS:
                    for label in TABLE_BINS:
                        item = entry[key].get(field, {}).get(label)
                        if not item:
                            continue
                        median = item["median"]
                        lines.append(f"| {CASES[case_name]['label']} | {horizon} | {field} | {TABLE_BIN_LABELS[label]} | "
                                     f"{median.get('g0xg0', float('nan')):.3g} | {median.get('g1xg1', float('nan')):.3g} | "
                                     f"{median.get('g0xg1', float('nan')):.3g} | {item['across_over_same']:.2f} | "
                                     f"{item['cross_p']:.2f} |")
    lines += ["", "## trios: the 10 splits of each arm's six runs into 3 + 3 (velocity; median of each trio's 3 pairs)", "",
              "奇 / 偶 = trials 1, 3, 5 / 2, 4, 6 (K = 1 and the control: GPU 0 / GPU 1); the other 9 splits are arbitrary.", "",
              "| 算例 | N | 臂 | 箱 | 奇三次 | 偶三次 | 奇偶比 | 另外 9 种分法的比 | 10 种里 ≥ 奇偶比的个数 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for case_name, horizons in results.items():
        for horizon, entry in horizons.items():
            for arm, bins in entry["trios"].items():
                for label in TRIO_BINS:
                    item = bins[label]
                    parity, others = item["parity"], item["other_ratios"]
                    lines.append(f"| {CASES[case_name]['label']} | {horizon} | {ARM_LABELS.get(arm, 'K = 1')} | "
                                 f"{TABLE_BIN_LABELS[label]} | {parity['odd']:.3g} | {parity['even']:.3g} | {parity['ratio']:.2f} | "
                                 f"{others[0]:.2f}–{others[-1]:.2f} | "
                                 f"{1 + sum(value >= parity['ratio'] for value in others)} |")
    seam_arms = ("k1", "k2", "k2s", "k4", "k1s")
    lines += ["", "## seam columns: within-arm velocity medians in the two columns on either side of each cut", "",
              "Each arm's own 15 pairs in the d = 0 band of that one cut, seam or not (median, range): is a column pair "
              "noisier for every arm?", "",
              "| 算例 | N | 切点 | 列 | 流体粒子 | " + " | ".join(ARM_LABELS.get(arm, "K = 1") for arm in seam_arms) + " |",
              "|---|---|---|---|---|" + "---|" * len(seam_arms)]
    for case_name, horizons in results.items():
        for horizon, entry in horizons.items():
            for cut, item in entry["seam_columns"].items():
                cells = []
                for arm in seam_arms:
                    within = item["within"].get(arm)
                    cells.append(f"{within['median']:.3g} ({within['range'][0]:.2g}–{within['range'][1]:.2g})" if within else "—")
                lines.append(f"| {CASES[case_name]['label']} | {horizon} | {cut} | {item['columns'][0]}, {item['columns'][1]} | "
                             f"{item['fluid']} | " + " | ".join(cells) + " |")
    lines += ["", f"## the audit gate's one-pair floor among the six K = 1 runs (ratio > {AUDIT_LIMIT:g}, 120 ordered triples, "
              "pairs at the K = 2 virtual seams)", "",
              "| 算例 | N | 箱 | " + " | ".join(FIELDS) + " |", "|---|---|---|" + "---|" * len(FIELDS)]
    for case_name, horizons in results.items():
        for horizon, entry in horizons.items():
            for label in ("d0", "all"):
                lines.append(f"| {CASES[case_name]['label']} | {horizon} | {TABLE_BIN_LABELS[label]} | "
                             + " | ".join(f"{entry['gate_null'][field][label]['fail_fraction']:.0%}" for field in FIELDS) + " |")
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
              "联合 = the configuration's 30 tests with the d = 0 velocity test replaced by the perturbed one: p < 0.05 count "
              "and P(≥) under the joint permutation null (cross / one).", "",
              "| 算例 | N | 臂 | 缺陷 | f | 跨 K / K=1 内 | K>1 内 / K=1 内 | 组间项 (p) | p 交叉 | p 含 K>1 | 联合 交叉 个数 P(≥) | "
              "联合 含 K>1 个数 P(≥) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for case_name, horizons in results.items():
        for horizon, entry in horizons.items():
            for arm, rows in entry["sensitivity"].items():
                for row in rows:
                    joint = entry["joint"][arm]["scenarios"][scenario_name(row["kind"], row["factor"])]
                    lines.append(f"| {CASES[case_name]['label']} | {horizon} | {ARM_LABELS[arm]} | {row['kind']} | {row['factor']:g} | "
                                 f"{row['cross_over_within_k1']:.2f} | {row['within_x_over_within_k1']:.2f} | "
                                 f"{row['between']:+.3f} ({row['between_p']:.3f}) | {row['cross_p']:.3f} | {row['one_p']:.3f} | "
                                 f"{joint['cross']['count']} {joint['cross']['p_at_least']:.3f} | "
                                 f"{joint['one']['count']} {joint['one']['p_at_least']:.3f} |")
    lines += ["", "## every field and bin (class medians: within K=1 / within K>1 / across; p cross / p one)", "",
              "机会 95 % = the 2.5–97.5 % range of the ratio over the 924 relabellings of the same 12 runs. 组间项 = "
              "(mean across d² − (mean within-K>1 d² + mean within-K=1 d²) / 2) / (mean within-K=1 d² / 2) = D² / s₁², the "
              "group offset in units of the K = 1 run-to-run scatter (unmoved by a tighter or wider K > 1 arm), with its "
              "one-sided permutation p.", "",
              "| 算例 | N | 臂 | 场 | 箱 | K=1 内 | K>1 内 | 跨 K | 跨 / K=1 内 | 机会 95 % | K>1 内 / K=1 内 | 机会 95 % | "
              "组间项 (p) | p 交叉 | p 含 K>1 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for case_name, horizons in results.items():
        for horizon, entry in horizons.items():
            for field in FIELDS:
                for arm, bins in entry["fields"][field].items():
                    for label in TABLE_BINS:
                        item = bins[label]
                        median = item["median"]
                        lines.append(f"| {CASES[case_name]['label']} | {horizon} | {ARM_LABELS[arm]} | {field} | "
                                     f"{TABLE_BIN_LABELS[label]} | {median['within_k1']:.3g} | {median['within_x']:.3g} | "
                                     f"{median['cross']:.3g} | {item['cross_over_within_k1']:.2f} | "
                                     f"{item['ratio_null_95'][0]:.2f}–{item['ratio_null_95'][1]:.2f} | "
                                     f"{item['within_x_over_within_k1']:.2f} | "
                                     f"{item['within_ratio_null_95'][0]:.2f}–{item['within_ratio_null_95'][1]:.2f} | "
                                     f"{item['between']:+.3f} ({item['between_p']:.3f}) | {item['cross_p']:.3f} | "
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
            joint = entry["joint"][arm]
            counts = (f"p < 0.05: × {joint['cross']['count']}/{joint['cross']['tests']} (P {joint['cross']['p_at_least']:.3f}), "
                      f"+ {joint['one']['count']}/{joint['one']['tests']} (P {joint['one']['p_at_least']:.3f})")
            if row == 0:
                axis.set_title(f"{CASES[case_name]['label']}, {ARM_LABELS[arm]}\ncuts {entry['cuts'][arm]}\n{counts}", fontsize=7)
            else:
                axis.set_title(counts, fontsize=7)
            axis.grid(alpha=0.3, which="both")
            axis.tick_params(labelsize=7)
        low, high = axes[row][0].get_ylim()          # the row shares y: headroom for the p labels
        axes[row][0].set_ylim(low, high * 12)
    handles = [plt.Line2D([], [], marker=marker, color=color, linestyle="", label=label)
               for marker, color, label in styles.values()]
    figure_handle.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    figure_handle.suptitle("E33: id-matched velocity difference of every pair of runs (6 K = 1 + 6 K > 1 per column); "
                           "p of the exact permutation tests at the top of each bin (× cross, + pairs with a K > 1 run);\n"
                           "panel titles: tests with p < 0.05 of the 30 (6 fields × 5 bins) and P of that count under the "
                           "joint permutation null", fontsize=8.5)
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
