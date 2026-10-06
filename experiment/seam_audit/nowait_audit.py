"""
nowait_audit.py - E32 part 1: depth-2 multi-step audit of V6_PHASE_A_NO_WAIT.

E31's two gates (single_step, ab_restart) restart the chain at depth 1: phase A
of frame n is submitted only after frame n - 1 has completed, so the
frame_done(n - 1) wait that V6_PHASE_A_NO_WAIT=1 drops is always satisfied
before the GPU reaches it. Here every run restarts from the SAME saved state
(single_step's snapshot_N2000.npz) at depth 2 through dump_state
--restart-snapshot, which dumps every particle 300 and 2000 steps later:

  w0_t1, w1_t1, w0_t2, w1_t2, ..., w0_t6, w1_t6   (w0 = V6_PHASE_A_NO_WAIT=0,
                                       w1 = =1; alternating, one process each)

analyze (the gate): analyze.py (KD match within 0.05 h, bins by column distance
to the nearest cut) compares the test pair B1, B2 = w1_t1, w1_t2 with the noise
floor of the two =0 runs A1, A2 = w0_t1, w0_t2. Gate = opt_validate's audit gate
without the crossing window: every d = 0 rms ratio of acceleration, shift,
velocity, density, pressure and kernel_sum (B1 vs A1 over A1 vs A2) and the
second pair's d = 0 ratios of acceleration, shift and density must be
<= AUDIT_LIMIT (2.0; None / inf / NaN fail), at both horizons. w0_t3 against the
same floor is the null comparison (both sides =0). Every run must also keep
drift, missing / duplicate ids, every overflow counter, GPU and host frame-stamp
errors and far migrations at 0.

ensemble: one =0 pair is a poor floor once the runs leave the round-off regime
at random times (section comment below); this mode uses all 12 runs: the
id-matched rms difference of every pair per column-distance bin, within-=0 /
within-=1 / cross medians, an exact permutation test over the arm labels and
the share of =0 triples (A1, A2, B1 = =0 or =1) that fail the gate's ratio.

Environment of every run: the caller's environment without any V5_* / V6_*
variable (the v6 code defaults = the release set) plus validation off,
V6_DELTA_DENSITY=1 (exact rho in the dumps, as every audit since E14) and the
V6_PHASE_A_NO_WAIT value of the arm. dump_state adds V6_TRANSPORT_EXTENSION=1.
Before every run nvidia-smi must show no other compute process
(step_trace_campaign.wait_for_idle_gpus); the check is logged. K = 4
(device map 0,1,0,1: two sims per card) runs last; a run that stalls (exit 1
with "stalled" / "died" in the error) is retried once, an invalid run never.

Usage (GPU, then CPU):
  .venv/Scripts/python.exe -m experiment.seam_audit.nowait_audit run --out logs/e32/nowait_audit
  .venv/Scripts/python.exe -m experiment.seam_audit.nowait_audit analyze --out logs/e32/nowait_audit
  .venv/Scripts/python.exe -m experiment.seam_audit.nowait_audit ensemble --out logs/e32/nowait_audit
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from experiment.seam_audit.dump_state import RESULT_PREFIX, dump_file_paths  # noqa: E402

AUDIT_LIMIT = 2.0                      # opt_validate.AUDIT_LIMIT
AUDIT_FIELDS = ("acceleration", "shift", "velocity", "density", "pressure", "kernel_sum")
SECOND_PAIR_FIELDS = ("acceleration", "shift", "density")
HORIZONS = (300, 2000)
SNAPSHOT_2D = "logs/seam_audit/single_step/cavity2d_1m/snapshot_N2000.npz"
SNAPSHOT_3D = "logs/seam_audit/single_step/cavity3d_1m/snapshot_N2000.npz"
CONFIGURATIONS = (
    {"name": "cavity2d_1m_k2", "case": "cases/lid_driven_cavity_2d_gen/case.yaml", "slabs": 2,
     "device_map": "0,1", "snapshot": SNAPSHOT_2D},
    {"name": "cavity3d_1m_k2", "case": "cases/cavity3d_1m/case.yaml", "slabs": 2,
     "device_map": "0,1", "snapshot": SNAPSHOT_3D},
    {"name": "cavity2d_1m_k4", "case": "cases/lid_driven_cavity_2d_gen/case.yaml", "slabs": 4,
     "device_map": "0,1,0,1", "snapshot": SNAPSHOT_2D},
)
# (run name, V6_PHASE_A_NO_WAIT), in execution order: =0 and =1 alternating, TRIALS of each
TRIALS = 6
RUN_ORDER = tuple((f"w{setting}_t{trial}", setting) for trial in range(1, TRIALS + 1) for setting in ("0", "1"))
REFERENCE_RUNS = ("w0_t1", "w0_t2")
TEST_RUNS = ("w1_t1", "w1_t2")
NULL_RUNS = ("w0_t3",)
RUN_TIMEOUT_SECONDS = 1800
STALL_MARKERS = ("stalled", "died")


def log_line(log_path: pathlib.Path, message: str) -> None:
    stamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(stamped, flush=True)
    with open(log_path, "a", encoding="utf-8") as stream:
        stream.write(stamped + "\n")


def run_environment(no_wait: str) -> dict:
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("V5_", "V6_"))}
    environment.update({"VK_LOADER_LAYERS_DISABLE": "VK_LAYER_KHRONOS_validation",
                        "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                        "V6_DELTA_DENSITY": "1", "V6_PHASE_A_NO_WAIT": no_wait})
    return environment


def result_line(text: str) -> dict:
    for line in reversed(text.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])
    return {}


def last_records(results_path: pathlib.Path) -> dict:
    """(configuration, run name) -> the last record written for it."""
    records = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[(record["configuration"], record["run"])] = record
    return records


def run_one(configuration: dict, run_name: str, no_wait: str, out_directory: pathlib.Path,
            campaign_log: pathlib.Path) -> dict:
    from experiment.v6.analysis.step_trace_campaign import wait_for_idle_gpus
    dump_directory = out_directory / "dumps" / configuration["name"]
    dump_directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "experiment.seam_audit.dump_state", "--version", "v6",
               "--case", configuration["case"], "--slabs", str(configuration["slabs"]),
               "--device-map", configuration["device_map"],
               "--restart-snapshot", configuration["snapshot"],
               "--horizons", ",".join(str(horizon) for horizon in HORIZONS), "--depth", "2",
               "--out-dir", str(dump_directory), "--run-name", run_name]
    attempts = []
    for attempt in (1, 2):
        gpu = wait_for_idle_gpus(lambda message: log_line(campaign_log, message))
        log_line(campaign_log, f"{configuration['name']} {run_name} (V6_PHASE_A_NO_WAIT={no_wait}) "
                               f"attempt {attempt}: nvidia-smi idle after {gpu['waited_s']} s, "
                               f"gpus {gpu['gpus']}")
        log_path = out_directory / "logs" / f"{configuration['name']}__{run_name}__a{attempt}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with open(log_path, "w", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=_REPOSITORY_ROOT, env=run_environment(no_wait),
                                       stdout=stream, stderr=subprocess.STDOUT)
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
        log_line(campaign_log, f"  -> exit {code} in {attempts[-1]['seconds']} s"
                               + (f", error {error}" if error else ""))
        stalled = code == 1 and any(marker in error for marker in STALL_MARKERS)
        if not stalled:
            break
    record = {"configuration": configuration["name"], "run": run_name, "no_wait": no_wait,
              "rc": attempts[-1]["rc"], "attempts": attempts, "result": summary,
              "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(out_directory / "results.jsonl", "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    return record


def run_campaign(out_directory: pathlib.Path, configuration_names) -> int:
    out_directory.mkdir(parents=True, exist_ok=True)
    campaign_log = out_directory / "campaign.log"
    done = last_records(out_directory / "results.jsonl")
    failures = 0
    for configuration in CONFIGURATIONS:
        if configuration_names and configuration["name"] not in configuration_names:
            continue
        for run_name, no_wait in RUN_ORDER:
            previous = done.get((configuration["name"], run_name))
            dumps = [dump_file_paths(out_directory / "dumps" / configuration["name"], run_name, horizon)
                     for horizon in HORIZONS]
            if previous and previous["rc"] == 0 and all(paths["npz"].exists() for paths in dumps):
                log_line(campaign_log, f"{configuration['name']} {run_name}: done, skipped")
                continue
            record = run_one(configuration, run_name, no_wait, out_directory, campaign_log)
            failures += record["rc"] != 0
    log_line(campaign_log, f"campaign finished, {failures} failed run(s)")
    return 1 if failures else 0


# ----------------------------------------------------------------------------- analysis
def finite_or_infinite(value) -> float:
    if value is None:
        return float("inf")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return number if number == number else float("inf")


def sidecar_problems(dump_directory: pathlib.Path, run_names, horizon: int) -> tuple[list, dict]:
    problems, invariants_by_run = [], {}
    for run_name in run_names:
        sidecar = dump_file_paths(dump_directory, run_name, horizon)["json"]
        if not sidecar.exists():
            problems.append(f"{run_name}: no N={horizon} dump")
            continue
        invariants = json.loads(sidecar.read_text(encoding="utf-8")).get("invariants", {})
        invariants_by_run[run_name] = invariants
        if not invariants.get("valid") or invariants.get("far_migration_count", 0) != 0:
            problems.append(f"{run_name} N={horizon}: invariants {invariants}")
    return problems, invariants_by_run


def bin_ratio(report: dict, comparison: str, field: str, bin_label: str):
    entry = report["comparisons"].get(comparison)
    if not entry:
        return None
    statistics = entry["fields"][field]["bins"].get(bin_label)
    return statistics["ratio"]["rms"] if statistics else None


def analyze_configuration(configuration: dict, out_directory: pathlib.Path, make_figures: bool) -> list:
    from experiment.seam_audit.analyze import analyze
    dump_directory = out_directory / "dumps" / configuration["name"]
    rows = []
    for horizon in HORIZONS:
        problems, invariants = sidecar_problems(dump_directory, REFERENCE_RUNS + TEST_RUNS + NULL_RUNS,
                                                horizon)
        reference = [dump_file_paths(dump_directory, name, horizon)["npz"] for name in REFERENCE_RUNS]
        row = {"configuration": configuration["name"], "horizon": horizon, "problems": problems,
               "invariants": invariants}
        for label, test_names in (("test", TEST_RUNS), ("null", NULL_RUNS)):
            test = [dump_file_paths(dump_directory, name, horizon)["npz"] for name in test_names]
            if not all(path.exists() for path in reference + test):
                row[label] = None
                if label == "test":
                    problems.append("dumps missing for the test analysis")
                continue
            analysis_directory = out_directory / "analysis" / f"{configuration['name']}_N{horizon}_{label}"
            report = analyze([str(path) for path in reference], [str(path) for path in test],
                             out_dir=analysis_directory, make_figures=make_figures)
            summary = report["summary"]
            column0 = {field: report["verdict"]["column0_rms_ratio"][field].get("all")
                       for field in AUDIT_FIELDS}
            row[label] = {
                "analysis": str(analysis_directory),
                "triplets": summary["triplets"],
                "column0_count": summary["column0_counts"]["all"],
                "column0_rms_ratio": column0,
                "second_pair_d0_rms_ratio": summary["second_pair_d0_rms_ratio"],
                "test_noise_d0_rms_ratio": summary["test_noise_d0_rms_ratio"],
                "far_bin_worst_rms_ratio": {field: value["value"] for field, value
                                            in report["verdict"]["far_bin_worst_rms_ratio"].items()},
                "interior_rms_ratio": {field: bin_ratio(report, "primary", field, "interior")
                                       for field in AUDIT_FIELDS},
                "noise_rms_d0": {field: report["comparisons"]["primary"]["fields"][field]["bins"]["0"]
                                 ["noise"]["rms"] for field in SECOND_PAIR_FIELDS},
                "unmatched": summary["unmatched"],
                "id_agreement_rate": summary["id_agreement_rate"],
                "all_runs_valid": summary["all_runs_valid"],
            }
        test = row.get("test")
        values = []
        if test:
            values += [finite_or_infinite(value) for value in test["column0_rms_ratio"].values()]
            second = test["second_pair_d0_rms_ratio"] or {}
            if not second:
                problems.append("second pair missing")
            values += [finite_or_infinite(second.get(field)) for field in SECOND_PAIR_FIELDS]
        row["worst"] = max(values) if values else None
        row["pass"] = bool(test) and not problems and bool(test["all_runs_valid"]) and \
            row["worst"] <= AUDIT_LIMIT
        rows.append(row)
        print(f"[nowait_audit] {configuration['name']} N={horizon}: worst {row['worst']} "
              f"-> {'PASS' if row['pass'] else 'FAIL'} {problems if problems else ''}", flush=True)
    return rows


def format_ratio(value) -> str:
    if value is None:
        return "—"
    number = finite_or_infinite(value)
    return "inf" if number == float("inf") else f"{number:.2f}"


def write_summary(rows: list, records: dict, out_directory: pathlib.Path) -> None:
    lines = ["# E32 part 1: depth-2 V6_PHASE_A_NO_WAIT audit", "",
             f"Gate: every d = 0 rms ratio (B1 vs A1 over A1 vs A2, fields {', '.join(AUDIT_FIELDS)}) "
             f"and the second pair's d = 0 ratios ({', '.join(SECOND_PAIR_FIELDS)}) <= {AUDIT_LIMIT}; "
             "every run valid with far migrations 0. Null = w0_t3 (=0) against the same floor.", "",
             "| configuration | N | test d0 acc / shift / vel / rho / p / ksum | 2nd pair acc / shift / rho | "
             "far worst acc / rho | interior acc / rho | null d0 acc / shift / rho | worst | gate |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        test, null = row.get("test") or {}, row.get("null") or {}
        column0 = test.get("column0_rms_ratio", {})
        second = test.get("second_pair_d0_rms_ratio") or {}
        far = test.get("far_bin_worst_rms_ratio", {})
        interior = test.get("interior_rms_ratio", {})
        null_column0 = null.get("column0_rms_ratio", {})
        lines.append(
            f"| {row['configuration']} | {row['horizon']} | "
            + " / ".join(format_ratio(column0.get(field)) for field in AUDIT_FIELDS) + " | "
            + " / ".join(format_ratio(second.get(field)) for field in SECOND_PAIR_FIELDS) + " | "
            + " / ".join(format_ratio(far.get(field)) for field in ("acceleration", "density")) + " | "
            + " / ".join(format_ratio(interior.get(field)) for field in ("acceleration", "density")) + " | "
            + " / ".join(format_ratio(null_column0.get(field)) for field in SECOND_PAIR_FIELDS) + " | "
            + f"{format_ratio(row['worst'])} | {'PASS' if row['pass'] else 'FAIL'} |")
    lines += ["", "## Runs", "", "| configuration | run | NO_WAIT | attempts | exit | seconds |",
              "|---|---|---|---|---|---|"]
    for (configuration, run_name), record in sorted(records.items()):
        lines.append(f"| {configuration} | {run_name} | {record['no_wait']} | {len(record['attempts'])} | "
                     f"{record['rc']} | {sum(attempt['seconds'] for attempt in record['attempts']):.0f} |")
    problems = [f"{row['configuration']} N={row['horizon']}: {problem}" for row in rows
                for problem in row["problems"]]
    if problems:
        lines += ["", "## Problems", ""] + [f"- {problem}" for problem in problems]
    (out_directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_directory / "summary.json").write_text(json.dumps({"rows": rows, "limit": AUDIT_LIMIT,
                                                           "pass": all(row["pass"] for row in rows)},
                                                          indent=1, default=str), encoding="utf-8")


# ----------------------------------------------------------------------------- ensemble
# The gate above takes ONE =0 pair as the noise floor. When the runs leave the round-off regime at random
# times (2-D 1M between N = 300 and 2000: identical =0 runs end 1.6e-5 to 5e-4 apart in velocity rms), one
# pair is a poor floor and even =0 against =0 fails it. The ensemble view uses every run: the id-matched
# rms difference of every pair (every run holds every particle, dumps sorted by id), per column-distance
# bin, split into within-=0, within-=1 and cross pairs, and an exact permutation test over the arm labels.
ENSEMBLE_FIELDS = ("velocity", "acceleration", "shift", "density", "pressure", "kernel_sum")
ENSEMBLE_BINS = ("d0", "d1_3", "d4_7", "far", "all")


def seam_distance(x: np.ndarray, origin_x: float, smoothing_length: float, cuts) -> np.ndarray:
    """Column distance to the nearest cut, as analyze.classify_columns: d = c - s right of cut s, s - 1 - c
    left of it (so the two columns touching a cut are d = 0)."""
    columns = np.floor((x - origin_x) / smoothing_length).astype(np.int64)
    distance = np.full(columns.shape, np.iinfo(np.int64).max, dtype=np.int64)
    for cut in cuts:
        distance = np.minimum(distance, np.where(columns >= cut, columns - cut, cut - 1 - columns))
    return distance


def bin_masks(distance: np.ndarray) -> dict:
    return {"d0": distance == 0, "d1_3": (distance >= 1) & (distance <= 3), "d4_7": (distance >= 4) & (distance <= 7),
            "far": distance >= 8, "all": np.ones(distance.shape, dtype=bool)}


def pair_label(first: str, second: str) -> str:
    return "within_0" if first[1] == second[1] == "0" else ("within_1" if first[1] == second[1] == "1" else "cross")


def permutation_tests(run_names, distances: dict, test_runs=None) -> dict:
    """Exact over every relabelling that keeps the arm sizes. distances: {(first, second): d > 0}.
    cross: mean log d of the cross pairs minus mean log d of the within pairs (a systematic =1 difference);
    one: mean log d of every pair with an =1 run minus mean log d of the within-=0 pairs (any extra
    difference, systematic or random). One-sided p = fraction of relabellings with a statistic >= observed.
    test_runs: the runs of the tested arm (default: the =1 runs, name[1] == "1"; E33 passes its K > 1 arm)."""
    import itertools
    names = list(run_names)
    if test_runs is None:
        ones = [name for name in names if name[1] == "1"]
    else:
        selected = set(test_runs)
        ones = [name for name in names if name in selected]
    logs = {pair: math.log(value) for pair, value in distances.items() if value > 0}

    def statistics(arm_one) -> tuple:
        cross, within, with_one, within_zero = [], [], [], []
        for (first, second), value in logs.items():
            first_one, second_one = first in arm_one, second in arm_one
            (cross if first_one != second_one else within).append(value)
            (with_one if (first_one or second_one) else within_zero).append(value)
        return (np.mean(cross) - np.mean(within), np.mean(with_one) - np.mean(within_zero))

    observed = statistics(set(ones))
    relabelled = [statistics(set(choice)) for choice in itertools.combinations(names, len(ones))]
    return {"cross_statistic": float(observed[0]), "one_statistic": float(observed[1]),
            "cross_p": float(np.mean([value[0] >= observed[0] - 1e-12 for value in relabelled])),
            "one_p": float(np.mean([value[1] >= observed[1] - 1e-12 for value in relabelled])),
            "relabellings": len(relabelled)}


def gate_rates(distances: dict, run_names, limit: float = AUDIT_LIMIT) -> dict:
    """Id-matched form of the gate's primary ratio d(B1, A1) / d(A1, A2) over every ordered triple of
    distinct runs: B1 from =0 (the null: how often identical settings fail) and B1 from =1."""
    zeros = [name for name in run_names if name[1] == "0"]
    ones = [name for name in run_names if name[1] == "1"]

    def distance(first, second):
        return distances.get((first, second), distances.get((second, first)))
    out = {}
    for label, candidates in (("null", zeros), ("test", ones)):
        ratios = []
        for first_reference in zeros:
            for second_reference in zeros:
                if second_reference == first_reference:
                    continue
                floor = distance(first_reference, second_reference)
                for test in candidates:
                    if test in (first_reference, second_reference) or not floor:
                        continue
                    ratios.append(distance(test, first_reference) / floor)
        ratios = np.array(ratios)
        out[label] = {"triples": int(ratios.size), "fail_fraction": float(np.mean(ratios > limit)) if ratios.size else None,
                      "median_ratio": float(np.median(ratios)) if ratios.size else None,
                      "p90_ratio": float(np.percentile(ratios, 90)) if ratios.size else None}
    return out


def ensemble_configuration(configuration: dict, out_directory: pathlib.Path) -> dict:
    import itertools
    dump_directory = out_directory / "dumps" / configuration["name"]
    result = {}
    for horizon in HORIZONS:
        names = [name for name, _ in RUN_ORDER if dump_file_paths(dump_directory, name, horizon)["npz"].exists()]
        sidecar = json.loads(dump_file_paths(dump_directory, names[0], horizon)["json"].read_text(encoding="utf-8"))
        fluid_groups = [index for index, kind in enumerate(sidecar["material_kinds"]) if int(kind) == 0]
        with np.load(dump_file_paths(dump_directory, names[0], horizon)["npz"]) as archive:
            identifiers = archive["id"]
            fluid = np.isin(archive["material"], fluid_groups)
            distance = seam_distance(archive["position"][:, 0].astype(np.float64), sidecar["origin_x"],
                                     sidecar["smoothing_length"], sidecar["cuts"])
        masks = {label: mask[fluid] for label, mask in bin_masks(distance).items()}
        entry = {"runs": names, "bin_counts": {label: int(mask.sum()) for label, mask in masks.items()}, "fields": {}}
        for field in ENSEMBLE_FIELDS:
            values = {}
            for name in names:
                with np.load(dump_file_paths(dump_directory, name, horizon)["npz"]) as archive:
                    if not np.array_equal(archive["id"], identifiers):
                        raise ValueError(f"{name} N={horizon}: ids differ from {names[0]}")
                    values[name] = archive[field][fluid].astype(np.float64)
            per_bin = {}
            for bin_label, mask in masks.items():
                if not mask.any():
                    continue
                distances = {}
                for first, second in itertools.combinations(names, 2):
                    difference = values[first][mask] - values[second][mask]
                    magnitude = (np.linalg.norm(difference, axis=1) if difference.ndim == 2
                                 else np.abs(difference))
                    distances[(first, second)] = float(np.sqrt(np.mean(magnitude * magnitude)))
                groups = {}
                for (first, second), value in distances.items():
                    groups.setdefault(pair_label(first, second), []).append(value)
                per_bin[bin_label] = {
                    "median": {label: float(np.median(items)) for label, items in groups.items()},
                    "range": {label: [float(min(items)), float(max(items))] for label, items in groups.items()},
                    "pairs": {f"{first}-{second}": value for (first, second), value in distances.items()},
                    "permutation": permutation_tests(names, distances),
                    "gate_rates": gate_rates(distances, names)}
            entry["fields"][field] = per_bin
            del values
        result[str(horizon)] = entry
        print(f"[nowait_audit] ensemble {configuration['name']} N={horizon}: {len(names)} runs", flush=True)
    return result


def write_ensemble(results: dict, out_directory: pathlib.Path) -> None:
    lines = ["# E32 part 1: ensemble view of the depth-2 V6_PHASE_A_NO_WAIT audit", "",
             "Id-matched rms difference of every run pair (fluid), median per pair class; p = exact permutation "
             "test over the arm labels (cross: cross pairs vs within pairs; one: pairs with an =1 run vs "
             "within-=0 pairs). Gate rates: share of ordered triples (A1, A2 from =0, B1) whose id-matched "
             f"d(B1, A1) / d(A1, A2) exceeds {AUDIT_LIMIT}, B1 from =0 (null) and from =1 (test).", "",
             "| configuration | N | field | bin | within =0 | within =1 | cross | cross / within =0 | p cross | "
             "p one | gate fail null | gate fail test |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for configuration, horizons in results.items():
        for horizon, entry in horizons.items():
            for field in ENSEMBLE_FIELDS:
                for bin_label in ("d0", "far", "all"):
                    item = entry["fields"][field].get(bin_label)
                    if not item:
                        continue
                    median = item["median"]
                    ratio = (median["cross"] / median["within_0"]) if median.get("within_0") else float("inf")
                    rates = item["gate_rates"]
                    lines.append(
                        f"| {configuration} | {horizon} | {field} | {bin_label} | {median.get('within_0', 0):.3g} | "
                        f"{median.get('within_1', 0):.3g} | {median.get('cross', 0):.3g} | {ratio:.2f} | "
                        f"{item['permutation']['cross_p']:.3f} | {item['permutation']['one_p']:.3f} | "
                        f"{rates['null']['fail_fraction']:.2f} | {rates['test']['fail_fraction']:.2f} |")
    (out_directory / "ensemble.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_directory / "ensemble.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


def ensemble_campaign(out_directory: pathlib.Path, configuration_names) -> int:
    results = {}
    for configuration in CONFIGURATIONS:
        if configuration_names and configuration["name"] not in configuration_names:
            continue
        results[configuration["name"]] = ensemble_configuration(configuration, out_directory)
    write_ensemble(results, out_directory)
    print(f"[nowait_audit] ensemble -> {out_directory / 'ensemble.md'}", flush=True)
    return 0


def analyze_campaign(out_directory: pathlib.Path, configuration_names, make_figures: bool) -> int:
    rows = []
    for configuration in CONFIGURATIONS:
        if configuration_names and configuration["name"] not in configuration_names:
            continue
        rows += analyze_configuration(configuration, out_directory, make_figures)
    write_summary(rows, last_records(out_directory / "results.jsonl"), out_directory)
    passed = bool(rows) and all(row["pass"] for row in rows)
    print(f"[nowait_audit] {'PASS' if passed else 'FAIL'} ({len(rows)} rows) -> "
          f"{out_directory / 'summary.md'}", flush=True)
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=("run", "analyze", "ensemble"))
    parser.add_argument("--out", default="logs/e32/nowait_audit")
    parser.add_argument("--configurations", default="",
                        help="comma list of configuration names (default: all, K = 4 last)")
    parser.add_argument("--no-figures", action="store_true")
    arguments = parser.parse_args()
    names = [name for name in arguments.configurations.split(",") if name]
    out_directory = pathlib.Path(arguments.out)
    if arguments.mode == "run":
        return run_campaign(out_directory, names)
    if arguments.mode == "ensemble":
        return ensemble_campaign(out_directory, names)
    return analyze_campaign(out_directory, names, not arguments.no_figures)


if __name__ == "__main__":
    sys.exit(main())
