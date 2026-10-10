"""
e7_full_summary.py — E7 full campaign: the batch report's tables (markdown) and numbers (JSON) from the log
directories of one or more batch jobs (the e7_lib.sh layout, mirrored from ~/run/logs/e7_b<batch><line>_<id>).

Per job directory: results.jsonl (parse_run_v6.py rows; from the rc2 harness on with solver_tag, solver_tree and
repository_root), index.tsv (e7_lib.sh: label, role, family, case, K, trial, reference kind, reference case,
point, line, host and, from the rc2 harness on, the solver tag; 11-column files read as solver ''),
provenance.json (from the rc2 harness on also solver_tag, expected_label, root and, when the job also runs the rc1
deployment, rc1_root / rc1_tree / rc1_expected_tree / rc1_tree_match / rc1_commit / rc1_manifest / rc1_scripts /
rc1_harness_commit / rc1_match), provenance_end.json (from the rc2 harness on: the same checks at the end of the job,
or in the cleanup of a job that stopped early; unchanged / changes / checked_in, or incomplete),
shader_cache.jsonl, prechecks.jsonl, memory_windows.jsonl, weights/<point>.json,
traces/<point>_<arm>/ and, optional, sacct.txt (sacct -P -X with JobID, JobName, Start, End, Elapsed, NodeList,
State); --plan takes the JSON of e7_plan.py --json for the node-hour comparison. Every run's index solver (column
12) is compared with its results row's solver_tag (a run without a tag reads as ''): a difference is listed in
tag_mismatches and printed in bold next to the collisions.

Definitions (user, 2026-10-10):
  fps of a run = the bench's steady window (steady_steps / steady_seconds or the printed steady fps, whichever
  print is finer for the run, see precise_fps);
  arms: E (equal weights) and C (calibrated); a point whose K runs carry two or more solver tags (the rc1 vs rc2
  pair points ab_<...>) has the arms per solver (C_rc1, C_rc2);
  per trial and arm, against the reference set of the same trial: strong eta = fps_K / (K mean fps_1), weak (F3,
  F4: the reference is the per-card case) eta = fps_K / mean fps_1, pairs (n11320, or a K = 1 reference the
  pre-check found too large) eta = fps_K / ((K / 2) mean fps_pair), flagged; eta_min the same with the slowest
  card of the set (min instead of mean);
  over the trials: mean +- std (sample), median, min / max, and two modes when the largest gap G between the
  sorted values exceeds both an absolute threshold A (2 % of |median| for fps, eta and eta_min; 0.02 = 2
  percentage points for the arm difference) and 3 x the pooled within-cluster standard deviation of the two
  clusters split at G (n >= 3; the count of each mode; see describe);
  arm difference = (fps_C - fps_E) / fps_E per trial (the two runs of one trial are adjacent);
  solver pairs: per trial, both solvers' runs against the same reference set, delta_fps = (fps_rc2 - fps_rc1) /
  fps_rc1 and delta_eta = eta_rc2 - eta_rc1 (delta_eta_min the same), over the trials without the mode test, with
  the standard error SD / sqrt(n) and the 95 % half-width t(0.975, n - 1) x SE (n >= 2), and each solver's
  trial-to-trial eta SD; the position term = mean of the K run in the first slot after the reference set minus mean
  of the one in the second slot (fps, eta, eta_min; the slot order is index.tsv's), with the solver that led each
  trial; with unequal leads it enters the mean delta with weight (leads of rc2 - leads of rc1) / n, so the
  lead-balanced estimates (each order weighted equally: delta = (mean | rc2 led + mean | rc1 led) / 2, position =
  half their difference) are kept too when both orders occur;
  dest guards (E7 B2, rc2's transport; parse_run_v6 dest_guard): per point and arm (per solver on a pair point) the
  sums over the arm's runs (timed and traced) that print them: precheck, relay, fallback, blocking, relay slept,
  released; flagged (bold) when fallback or blocking > 0 in a run with V7_DEST_GUARD=relay;
  traces (experiment/v7/analysis/step_trace_model.run_summary), per step on the worse link, p50 / p95: the main
  ratio r_chain = t_chain / T_B (t_chain = t_tr minus the receiver's two waits: the transport chain itself), and
  r = t_tr / T_B for reference (it contains the receiver's two waits, which the late wake-ups contaminate);
  t_chain and t_tr per link (median of the links' p50, largest p95); T_B per sim (median and largest p50); the
  traced run's fps against the mean of the same arm's timing runs;
  eta-r figure (--figure): x = r_chain p50 of an arm's traced run with a line to its p95, y = eta mean +- std over
  the trials of the SAME arm, r p50 as an open marker; its numbers in a CSV next to it;
  barrier: per simultaneous group the spread (max - min) over its members of the release time, of the loop start
  and of the steady-window start (loop start + loop seconds - steady seconds);
  clocks: per run the median SM clock and power of its GPUs in the loop window (telemetry_loop); per point the
  mean over the K runs' GPUs against the mean over the reference members' own GPUs;
  construction: read-in (case_loaded, bench imports included), partition, contexts + simulators (bootstrap_start -
  partition_done), bootstrap; warm runs (timing arms; reference sets after the first trial) against read-in +
  1 s x K.

Usage:
    python docs/cluster_v6/scripts/e7_full_summary.py JOB_DIR [JOB_DIR ...] [--plan PLAN.json] [--out REPORT.md]
        [--json OUT.json] [--figure ETA_R.png]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
import statistics
import sys

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
INDEX_COLUMNS = ("label", "role", "family", "case", "slabs", "trial", "reference_kind", "reference_case", "point",
                 "line", "host", "solver")
ARMS = ("E", "C")
ARM_OF_ROLE = {"equal": "E", "calibrated": "C", "trace_equal": "E", "trace_calibrated": "C"}
MODE_RELATIVE_GAP = 0.02            # two modes: the largest gap exceeds 2 % of |median| (fps, eta, eta_min) ...
MODE_SPREAD_FACTOR = 3.0            # ... and 3 x the pooled within-cluster standard deviation
ARM_DIFFERENCE_MODE_GAP = 0.02      # the arm difference is a fraction already: 2 percentage points
BASELINE_SOLVER = "rc1"             # solver pairs: every other solver tag of a point against this one
# git rev-parse <tag>:experiment/v7 -> tag; names the trees in the provenance lines (the checks compare hashes)
TREE_NAMES = {"82dd6a740fa2a97a44e6a31d75392f75e486163b": "v7-rc1"}
FIGURE_ARM_COLORS = {"E": "#2a78d6", "C": "#eb6834"}        # categorical slots 1 and 2 of the dataviz palette
FIGURE_REFERENCE_MARKERS = {"strong": "o", "weak": "s", "pairs": "D"}
FIGURE_ZOOM_STANDARD_DEVIATION = 0.01   # the eta-r figure's lower panel: the rows whose eta std over the trials is at
                                        # most this
ETA_R_COLUMNS = ("job", "point", "arm", "solver", "K", "reference", "trials", "eta_mean", "eta_std", "eta_trial_min",
                 "eta_trial_max", "eta_min_mean", "r_chain_p50", "r_chain_p95", "r_p50", "r_p95", "trace_status")
# two-sided 95 % quantiles of Student's t by degrees of freedom (above 30: 1.96; between the listed ones: the next
# smaller listed, i.e. the wider interval)
T_QUANTILE_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
                  10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
                  19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042}
DEST_GUARD_COUNTS = ("precheck", "relay", "fallback", "blocking", "relay_slept", "released")


# ----------------------------------------------------------------------------- loading

def read_json_lines(path: pathlib.Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json_file(path: pathlib.Path):
    """One JSON object of the harness (provenance.json, provenance_end.json): None when the file is absent;
    strict=False (a stray control character, a \\r from a Windows nvidia-smi, must not stop the report);
    {"unreadable": <its first 500 characters>} when it does not parse."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"), strict=False)
    except json.JSONDecodeError:
        return {"unreadable": path.read_text(encoding="utf-8", errors="replace")[:500]}


def load_job(directory: pathlib.Path) -> dict:
    rows = {}
    for row in read_json_lines(directory / "results.jsonl"):
        rows[row["label"]] = row
    index = []
    if (directory / "index.tsv").exists():
        for line in (directory / "index.tsv").read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) >= len(INDEX_COLUMNS) - 1:       # files before the solver column have 11 columns
                entry = dict(zip(INDEX_COLUMNS, fields))
                entry["slabs"] = int(entry["slabs"]) if entry["slabs"].isdigit() else entry["slabs"]
                entry["trial"] = int(entry["trial"]) if entry["trial"].isdigit() else 0
                entry["solver"] = entry.get("solver", "").strip()
                index.append(entry)
    weights = {}
    for path in sorted((directory / "weights").glob("*.json")) if (directory / "weights").exists() else []:
        try:
            weights[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            weights[path.stem] = None
    provenance = read_json_file(directory / "provenance.json") or {}
    return {"directory": directory, "name": directory.name, "rows": rows, "index": index, "weights": weights,
            "provenance": provenance, "provenance_end": read_json_file(directory / "provenance_end.json"),
            "shader_cache": read_json_lines(directory / "shader_cache.jsonl"),
            "prechecks": read_json_lines(directory / "prechecks.jsonl"),
            "memory_windows": {entry["label"]: entry for entry in read_json_lines(directory / "memory_windows.jsonl")},
            "sacct": read_sacct(directory / "sacct.txt")}


def read_sacct(path: pathlib.Path) -> list:
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return []
    header = lines[0].split("|")
    return [dict(zip(header, line.split("|"))) for line in lines[1:]]


# ----------------------------------------------------------------------------- numbers

def precise_fps(row: dict):
    """The steady window's fps from whichever of the bench's two prints is finer for this run: the seconds carry
    2 decimals (relative error <= 0.005 s / seconds, coarse for fast runs: 0.25 % at 1000 fps over 2000 steps),
    the fps 1 decimal (<= 0.05 / fps, coarse for slow runs: 0.3 % at 15 fps); the finer one is <= 0.04 % here."""
    if not row:
        return None
    steps, seconds, printed = row.get("steady_steps"), row.get("steady_seconds"), row.get("steady_fps")
    candidates = []
    if steps and seconds:
        candidates.append((0.005 / seconds, steps / seconds))
    if printed:
        candidates.append((0.05 / printed, printed))
    return min(candidates)[1] if candidates else None


def precise_steady_seconds(row: dict):
    fps = precise_fps(row)
    steps = (row or {}).get("steady_steps")
    return steps / fps if steps and fps else (row or {}).get("steady_seconds")


def describe(values: list, absolute_gap=None, mode_rule: bool = True) -> dict:
    """n, mean, std (sample), median, min / max and the values (None and non-finite dropped). With mode_rule and
    n >= 3 the two-modes test: the largest gap G between the sorted values (the upper one on a tie) splits
    them into a low cluster L and a high cluster H; two modes when G exceeds the absolute threshold A
    (absolute_gap, else MODE_RELATIVE_GAP x |median|) AND MODE_SPREAD_FACTOR x the pooled within-cluster
    standard deviation s_w = sqrt((SS_L + SS_H) / (n - 2)) (SS = sum of squares about the cluster's mean; a
    one-value cluster adds 0, so s_w comes from the other one; equal values give 0 and A decides). mode_test
    keeps gap G, threshold A and spread s_w; modes the count and median of each cluster."""
    values = [value for value in values if value is not None and math.isfinite(value)]
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    out = {"n": len(values), "mean": statistics.fmean(values),
           "std": statistics.stdev(values) if len(values) > 1 else 0.0,
           "median": statistics.median(values), "min": ordered[0], "max": ordered[-1], "values": values}
    if mode_rule and len(values) >= 3:
        gaps = [(ordered[index + 1] - ordered[index], index) for index in range(len(ordered) - 1)]
        gap, position = max(gaps)
        low, high = ordered[:position + 1], ordered[position + 1:]
        squares = sum((value - statistics.fmean(cluster)) ** 2 for cluster in (low, high) for value in cluster)
        spread = math.sqrt(squares / (len(values) - 2))
        threshold = absolute_gap if absolute_gap is not None else MODE_RELATIVE_GAP * abs(out["median"])
        out["mode_test"] = {"gap": gap, "threshold": threshold, "spread": spread}
        if gap > threshold and gap > MODE_SPREAD_FACTOR * spread:
            out["modes"] = [{"count": len(low), "center": statistics.median(low)},
                            {"count": len(high), "center": statistics.median(high)}]
    return out


def text_of(stats: dict, digits: int = 1, scale: float = 1.0) -> str:
    if not stats or stats.get("n", 0) == 0:
        return "-"
    form = f"{{:.{digits}f}}"
    text = (f"{form.format(stats['mean'] * scale)} ± {form.format(stats['std'] * scale)} "
            f"(med {form.format(stats['median'] * scale)}; {form.format(stats['min'] * scale)}–"
            f"{form.format(stats['max'] * scale)}; n={stats['n']})")
    if "modes" in stats:
        text += " **two modes: " + " / ".join(f"{mode['count']}× ~{form.format(mode['center'] * scale)}"
                                              for mode in stats["modes"]) + "**"
    return text


def stage(row: dict, name: str, key: str = "seconds"):
    return ((row or {}).get("stages") or {}).get(name, {}).get(key)


def steady_start_epoch(row: dict):
    start = stage(row, "loop_start", "epoch")
    steady_seconds = precise_steady_seconds(row)
    if start is None or row.get("loop_seconds") is None or steady_seconds is None:
        return None
    return start + row["loop_seconds"] - steady_seconds


def gpu_means(row: dict, gpus) -> dict:
    loop = (row or {}).get("telemetry_loop") or {}
    clocks = [loop[str(gpu)]["sm_clock_median"] for gpu in gpus
              if str(gpu) in loop and loop[str(gpu)].get("sm_clock_median") is not None]
    powers = [loop[str(gpu)]["power_median"] for gpu in gpus
              if str(gpu) in loop and loop[str(gpu)].get("power_median") is not None]
    return {"sm_clock": statistics.fmean(clocks) if clocks else None,
            "power": statistics.fmean(powers) if powers else None}


def own_vram_mib(row: dict):
    peaks = (row or {}).get("telemetry_peaks") or {}
    values = [peaks[str(gpu)]["memory_used_peak_mib"] for gpu in (row or {}).get("device_map") or []
              if str(gpu) in peaks]
    return max(values) if values else None


# ----------------------------------------------------------------------------- points

def group_label(member_label: str) -> str:
    return member_label.rsplit("_", 1)[0]


def arm_key(arm: str, solver: str, by_solver: bool) -> str:
    """E / C, or C_rc1 when the point's arms are kept per solver (a run without a tag keeps the bare arm)."""
    return f"{arm}_{solver}" if by_solver and solver else arm


def split_arm(key: str) -> tuple:
    """C_rc1 -> (C, rc1); C -> (C, '')."""
    arm, _, solver = key.partition("_")
    return arm, solver


def collect_points(job: dict) -> dict:
    """point -> {family, case, slabs, line, host, solvers, arms, trials {t: {<arm>, references {group: {...}}}},
    traces {<arm>}, calibration, collisions, tag_mismatches}. The arms are E and C; when the point's K runs (timed
    and traced) carry two or more solver tags (index column 12: the rc1 vs rc2 pair points) they are E / C plus
    _<solver> (C_rc1, C_rc2), so the two solvers' runs of one trial do not overwrite each other. A second K run for
    one trial and arm still replaces the first one; collisions lists it. A K run keeps its position in index.tsv
    ("order": the run order of the job). tag_mismatches lists every run whose results row's solver_tag (none = '')
    differs from its index solver."""
    solvers: dict = {}
    for entry in job["index"]:
        if entry["role"] in ARM_OF_ROLE:
            solvers.setdefault(entry["point"], set()).add(entry["solver"])
    points: dict = {}
    for position, entry in enumerate(job["index"]):
        role = entry["role"]
        if role not in ("calibration", "equal", "calibrated", "reference", "reference_skipped", "trace_equal",
                        "trace_calibrated"):
            continue
        point = points.setdefault(entry["point"], {"point": entry["point"], "family": None, "case": None,
                                                   "slabs": None, "line": entry["line"], "host": entry["host"],
                                                   "solvers": sorted(solvers.get(entry["point"], ())),
                                                   "trials": {}, "traces": {}, "job": job["name"],
                                                   "collisions": [], "tag_mismatches": []})
        if role != "reference":
            point["family"], point["case"], point["slabs"] = entry["family"], entry["case"], entry["slabs"]
        if role == "calibration":
            continue
        row = job["rows"].get(entry["label"])
        if row is not None and (row.get("solver_tag") or "") != entry["solver"]:
            point["tag_mismatches"].append({"label": entry["label"], "index_solver": entry["solver"],
                                            "row_solver_tag": row.get("solver_tag")})
        arm = arm_key(ARM_OF_ROLE[role], entry["solver"], len(point["solvers"]) > 1) if role in ARM_OF_ROLE else None
        if role in ("trace_equal", "trace_calibrated"):
            if arm in point["traces"]:
                point["collisions"].append({"trial": "trace", "arm": arm, "replaced": point["traces"][arm]["label"],
                                            "by": entry["label"]})
            point["traces"][arm] = {"label": entry["label"], "row": row}
            continue
        trial = point["trials"].setdefault(entry["trial"], {"references": {}})
        if role in ("equal", "calibrated"):
            if arm in trial:
                point["collisions"].append({"trial": entry["trial"], "arm": arm, "replaced": trial[arm]["label"],
                                            "by": entry["label"]})
            trial[arm] = {"label": entry["label"], "row": row, "order": position}
        elif role == "reference_skipped":
            trial["references"][entry["label"]] = {"kind": entry["reference_kind"], "skipped": True,
                                                   "case": entry["reference_case"]}
        else:
            group = trial["references"].setdefault(group_label(entry["label"]), {
                "kind": entry["reference_kind"], "case": entry["reference_case"], "members": [],
                "pairs": entry["label"].rsplit("_", 1)[1].startswith("p")})
            group["members"].append({"label": entry["label"], "row": row})
    for point in points.values():
        point["calibration"] = job["weights"].get(point["point"])
        point["arms"] = list(ARMS)
        if len(point["solvers"]) > 1:
            keys = {key for trial in point["trials"].values() for key in trial if key != "references"}
            point["arms"] = sorted(keys | set(point["traces"]), key=lambda key: (ARMS.index(split_arm(key)[0]), key))
    return points


def reference_type(group: dict, point_case: str) -> str:
    if group.get("pairs"):
        return "pairs"
    return "strong" if group["case"] == point_case else "weak"


def efficiency(fps_k, slabs: int, group: dict, point_case: str):
    """(eta, eta_min, type) of one K run against one reference group, or None."""
    fps_values = [precise_fps(member["row"]) for member in group.get("members", [])
                  if member["row"] and member["row"].get("status") in ("pass", "threshold_only")]
    fps_values = [value for value in fps_values if value]
    if fps_k is None or not fps_values or len(fps_values) != len(group.get("members", [])):
        return None
    kind = reference_type(group, point_case)
    factor = {"strong": slabs, "weak": 1, "pairs": slabs / 2}[kind]
    return (fps_k / (factor * statistics.fmean(fps_values)), fps_k / (factor * min(fps_values)), kind)


def solver_pairs(point: dict) -> list:
    """The arms to compare on a point with two or more solver tags: per arm (E, C) every other solver against the
    baseline (BASELINE_SOLVER when the point has it, else the first tag), when both have runs."""
    if len(point["solvers"]) < 2:
        return []
    baseline = BASELINE_SOLVER if BASELINE_SOLVER in point["solvers"] else point["solvers"][0]
    pairs = []
    for arm in ARMS:
        for solver in point["solvers"]:
            baseline_arm, solver_arm = arm_key(arm, baseline, True), arm_key(arm, solver, True)
            if solver != baseline and baseline_arm in point["arms"] and solver_arm in point["arms"]:
                pairs.append({"arm": arm, "baseline": baseline, "solver": solver, "baseline_arm": baseline_arm,
                              "solver_arm": solver_arm})
    return pairs


def t_quantile_975(degrees_of_freedom: int) -> float:
    """Student's t, two-sided 95 % (T_QUANTILE_975; above 30 degrees of freedom 1.96)."""
    if degrees_of_freedom > max(T_QUANTILE_975):
        return 1.96
    return T_QUANTILE_975[max(key for key in T_QUANTILE_975 if key <= degrees_of_freedom)]


def with_resolution(stats: dict) -> dict:
    """A describe() result of per-trial differences plus standard_error = std / sqrt(n) and half_width_95 =
    t(0.975, n - 1) x standard_error (n >= 2): the size of a mean difference the trials can resolve."""
    if stats.get("n", 0) >= 2:
        stats["standard_error"] = stats["std"] / math.sqrt(stats["n"])
        stats["half_width_95"] = t_quantile_975(stats["n"] - 1) * stats["standard_error"]
    return stats


def position_summary(series: dict, relative: bool = False) -> dict:
    """The position term of a solver pair over the trials where both runs count: the mean of the K runs in the
    first slot after the reference set minus the mean of those in the second slot ("difference"), with both means
    and the per-trial first - second values; relative: also the difference over the mean of all those runs."""
    first, second = series["first"], series["second"]
    if not first:
        return {"n": 0}
    out = {"n": len(first), "first_mean": statistics.fmean(first), "second_mean": statistics.fmean(second),
           "values": [early - late for early, late in zip(first, second)], "trials": series["trials"]}
    out["difference"] = out["first_mean"] - out["second_mean"]
    if relative:
        out["relative"] = out["difference"] / statistics.fmean(first + second)
    return out


def lead_balanced(values: list, signs: list):
    """Per-trial differences delta_t = d + p x s_t (s_t = +1 when the compared solver ran in the first K slot after
    the reference set, -1 when the baseline did; p = the position effect, first minus second slot, in the units of
    delta): d = (mean over s = +1 + mean over s = -1) / 2, p = (mean over s = +1 - mean over s = -1) / 2, so each
    lead order weighs equally (the plain mean of delta carries p x (leads of the solver - leads of the baseline) /
    n). None unless both orders occur."""
    led = [value for value, sign in zip(values, signs) if sign == 1]
    trailed = [value for value, sign in zip(values, signs) if sign == -1]
    if not led or not trailed:
        return None
    return {"delta": (statistics.fmean(led) + statistics.fmean(trailed)) / 2,
            "position": (statistics.fmean(led) - statistics.fmean(trailed)) / 2,
            "solver_led": len(led), "baseline_led": len(trailed)}


def dest_guard_sums(rows: list) -> dict:
    """The dest guard outcomes (parse_run_v6 dest_guard, every worker) summed over the rows that carry them: runs
    (rows with the lines) of runs_total, relay_runs (a worker with V7_DEST_GUARD=relay in its line, else the bench's
    switch line), the DEST_GUARD_COUNTS sums, and flagged: fallback or blocking > 0 in a relay-mode run. {} when no
    row carries them (rc1, batch 1)."""
    carrying = [row for row in rows if row.get("dest_guard")]
    if not carrying:
        return {}
    out = {"runs": len(carrying), "runs_total": len(rows), "relay_runs": 0, **{name: 0 for name in DEST_GUARD_COUNTS},
           "flagged": False}
    for row in carrying:
        workers = list(row["dest_guard"].values())
        relay = (any((worker.get("switches") or {}).get("V7_DEST_GUARD") == "relay" for worker in workers)
                 or str((row.get("solver_switches") or {}).get("V7_DEST_GUARD", "")).strip().lower() == "relay")
        for worker in workers:
            for name in DEST_GUARD_COUNTS:
                out[name] += int(worker.get(name) or 0)
        if relay:
            out["relay_runs"] += 1
            if any(int(worker.get("fallback") or 0) > 0 or int(worker.get("blocking") or 0) > 0 for worker in workers):
                out["flagged"] = True
    return out


def analyze_point(point: dict) -> dict:
    out = {key: point[key] for key in ("point", "family", "case", "slabs", "line", "host", "job", "solvers", "arms")}
    arms = {arm: [] for arm in point["arms"]}
    etas: dict = {}
    differences: dict = {}      # solver of the arms ('' unless the arms are per solver) -> C - E per trial
    pairs = solver_pairs(point)
    deltas = [{"fps": [], "fps_trials": [], "fps_signs": [], "eta": {}, "eta_min": {}, "leads": [],
               "position": {"fps": {"first": [], "second": [], "trials": []}, "eta": {}, "eta_min": {}}}
              for _ in pairs]
    for trial_number, trial in sorted(point["trials"].items()):
        fps = {}
        for arm in point["arms"]:
            run = trial.get(arm)
            value = precise_fps(run["row"]) if run and run["row"] and run["row"].get("status") == "pass" else None
            fps[arm] = value
            arms[arm].append(value)
        for arm in point["arms"]:
            name, solver = split_arm(arm)
            calibrated = arm_key("C", solver, True)
            if name == "E" and fps.get(arm) and fps.get(calibrated):
                differences.setdefault(solver, []).append((fps[calibrated] - fps[arm]) / fps[arm])
        signs = []          # per pair: +1 the compared solver ran first after the reference set, -1 the baseline did
        for pair, delta in zip(pairs, deltas):
            baseline_run, solver_run = trial.get(pair["baseline_arm"]), trial.get(pair["solver_arm"])
            sign = None
            if baseline_run and solver_run and baseline_run.get("order") is not None \
                    and solver_run.get("order") is not None:
                sign = 1 if solver_run["order"] < baseline_run["order"] else -1
                delta["leads"].append({"trial": trial_number, "first": pair["solver"] if sign == 1 else pair["baseline"]})
            signs.append(sign)
            baseline, other = fps.get(pair["baseline_arm"]), fps.get(pair["solver_arm"])
            if baseline and other:
                delta["fps"].append((other - baseline) / baseline)
                delta["fps_trials"].append(trial_number)
                delta["fps_signs"].append(sign)
                if sign is not None:
                    position = delta["position"]["fps"]
                    position["first"].append(other if sign == 1 else baseline)
                    position["second"].append(baseline if sign == 1 else other)
                    position["trials"].append(trial_number)
        for group_name, group in trial["references"].items():
            if group.get("skipped"):
                continue
            results = {}
            for arm in point["arms"]:
                result = efficiency(fps.get(arm), point["slabs"], group, point["case"])
                if result is None:
                    continue
                eta, eta_min, kind = result
                key = f"{kind}:{group['case']}"
                bucket = etas.setdefault(key, {name: {"eta": [], "eta_min": []} for name in point["arms"]})
                bucket[arm]["eta"].append(eta)
                bucket[arm]["eta_min"].append(eta_min)
                results[arm] = (key, eta, eta_min)
            for pair, delta, sign in zip(pairs, deltas, signs):     # both solvers against this one reference set
                if pair["baseline_arm"] in results and pair["solver_arm"] in results:
                    key, eta, eta_min = results[pair["baseline_arm"]]
                    _, other_eta, other_eta_min = results[pair["solver_arm"]]
                    for name, base_value, other_value in (("eta", eta, other_eta), ("eta_min", eta_min, other_eta_min)):
                        series = delta[name].setdefault(key, {"values": [], "trials": [], "signs": []})
                        series["values"].append(other_value - base_value)
                        series["trials"].append(trial_number)
                        series["signs"].append(sign)
                        if sign is not None:
                            position = delta["position"][name].setdefault(key, {"first": [], "second": [], "trials": []})
                            position["first"].append(other_value if sign == 1 else base_value)
                            position["second"].append(base_value if sign == 1 else other_value)
                            position["trials"].append(trial_number)
    out["fps"] = {arm: describe(values) for arm, values in arms.items()}
    out["eta"] = {key: {arm: {name: describe(values) for name, values in measures.items()}
                        for arm, measures in bucket.items()} for key, bucket in etas.items()}
    out["arm_difference"] = describe(differences.get("", []), ARM_DIFFERENCE_MODE_GAP)
    out["arm_difference_by_solver"] = {solver: describe(values, ARM_DIFFERENCE_MODE_GAP)
                                       for solver, values in sorted(differences.items()) if solver}
    out["solver_pairs"] = []
    for pair, delta in zip(pairs, deltas):
        entry = dict(pair)
        entry["leads"] = delta["leads"]
        entry["delta_fps"] = dict(with_resolution(describe(delta["fps"], mode_rule=False)), trials=delta["fps_trials"],
                                  lead_balanced=lead_balanced(delta["fps"], delta["fps_signs"]))
        entry["position_fps"] = position_summary(delta["position"]["fps"], relative=True)
        for name in ("eta", "eta_min"):
            entry["delta_" + name] = {key: dict(with_resolution(describe(series["values"], mode_rule=False)),
                                                trials=series["trials"],
                                                lead_balanced=lead_balanced(series["values"], series["signs"]))
                                      for key, series in delta[name].items()}
            entry["position_" + name] = {key: position_summary(series)
                                         for key, series in delta["position"][name].items()}
        # each solver's trial-to-trial eta SD (the noise the pair's mean difference sits in)
        entry["eta_trial_standard_deviation"] = {
            key: {solver: (bucket[arm]["eta"].get("std") if bucket.get(arm, {}).get("eta", {}).get("n", 0) >= 2
                           else None)
                  for solver, arm in ((pair["baseline"], pair["baseline_arm"]), (pair["solver"], pair["solver_arm"]))}
            for key, bucket in out["eta"].items()}
        out["solver_pairs"].append(entry)
    # dest guard outcomes per arm (rc2's transport): the arm's timed runs and its traced run
    out["dest_guard"] = {}
    for arm in point["arms"]:
        rows = [trial[arm]["row"] for _, trial in sorted(point["trials"].items()) if trial.get(arm) and trial[arm]["row"]]
        if (point["traces"].get(arm) or {}).get("row"):
            rows.append(point["traces"][arm]["row"])
        sums = dest_guard_sums(rows)
        if sums:
            out["dest_guard"][arm] = sums
    calibration = point.get("calibration") or {}
    out["calibration"] = {"weights": calibration.get("weights"), "cuts": calibration.get("cuts"),
                          "rounds": [{"round": entry.get("round"), "cuts": entry.get("cuts"),
                                      "next_cuts": entry.get("next_cuts"), "changed": entry.get("changed"),
                                      "busy_us": entry.get("busy_us"), "pilot_fps": entry.get("pilot_steady_fps")}
                                     for entry in calibration.get("rounds", [])]}
    out["traces"] = {}
    for arm, trace in point["traces"].items():
        row = trace["row"]
        traced = precise_fps(row) if row else None
        untraced = out["fps"].get(arm, {}).get("mean")
        out["traces"][arm] = {"label": trace["label"], "status": row.get("status") if row else "missing",
                              "fps": traced,
                              "fps_vs_untraced": (traced - untraced) / untraced if traced and untraced else None}
    out["collisions"] = point["collisions"]
    out["tag_mismatches"] = point["tag_mismatches"]
    return out


def trace_metrics(directory: pathlib.Path) -> dict:
    """step_trace_model.run_summary of one trace directory, reduced to the report's numbers: r_chain (the main
    ratio) and r on the worse link per step, t_chain and t_tr per link, T_B per sim."""
    if not (directory / "run_meta.json").exists():
        return {"missing": True}
    if str(_REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPOSITORY_ROOT))
    from experiment.v7.analysis import step_trace_model
    summary = step_trace_model.run_summary(step_trace_model.load_run(directory))
    links = summary.get("links", {})
    sims = summary.get("sims", [])

    def finite(values):
        return [value for value in values if value is not None and math.isfinite(value)]
    t_chain_p50 = finite([entry.get("t_chain_p50") for entry in links.values()])
    t_chain_p95 = finite([entry.get("t_chain_p95") for entry in links.values()])
    t_tr_p50 = finite([entry.get("t_tr_p50") for entry in links.values()])
    t_tr_p95 = finite([entry.get("t_tr_p95") for entry in links.values()])
    t_b = finite([entry.get("T_B_p50_us") for entry in sims])
    return {"r_chain_p50": summary.get("r_chain_worst_p50"), "r_chain_p95": summary.get("r_chain_worst_p95"),
            "r_p50": summary.get("r_worst_p50"), "r_p95": summary.get("r_worst_p95"),
            "t_chain_p50_median_link": statistics.median(t_chain_p50) if t_chain_p50 else None,
            "t_chain_p95_max_link": max(t_chain_p95) if t_chain_p95 else None,
            "t_tr_p50_median_link": statistics.median(t_tr_p50) if t_tr_p50 else None,
            "t_tr_p95_max_link": max(t_tr_p95) if t_tr_p95 else None,
            "T_B_p50_median_sim": statistics.median(t_b) if t_b else None,
            "T_B_p50_max_sim": max(t_b) if t_b else None,
            "exposed_steps_fraction": summary.get("exposed_steps_fraction"),
            "coverage": summary.get("coverage"), "links": len(links), "steady_fps": summary.get("steady_fps")}


# ----------------------------------------------------------------------------- other tables

def barrier_groups(job: dict) -> list:
    groups: dict = {}
    for entry in job["index"]:
        if entry["role"] in ("reference", "precheck"):
            groups.setdefault(group_label(entry["label"]), []).append(job["rows"].get(entry["label"]))
    out = []
    for name, rows in groups.items():
        rows = [row for row in rows if row]
        released = [row["barrier"]["released_epoch"] for row in rows if row.get("barrier")]
        loop_starts = [stage(row, "loop_start", "epoch") for row in rows]
        loop_starts = [value for value in loop_starts if value is not None]
        steady_starts = [value for value in (steady_start_epoch(row) for row in rows) if value is not None]
        waits = [row["barrier"]["waited_seconds"] for row in rows if row.get("barrier")]
        statuses = sorted({row["barrier"]["status"] for row in rows if row.get("barrier")})
        out.append({"group": name, "members": len(rows),
                    "release_spread_ms": (max(released) - min(released)) * 1e3 if released else None,
                    "loop_start_spread_ms": (max(loop_starts) - min(loop_starts)) * 1e3 if loop_starts else None,
                    "steady_start_spread_ms": (max(steady_starts) - min(steady_starts)) * 1e3 if steady_starts else None,
                    "longest_wait_s": max(waits) if waits else None, "statuses": statuses})
    return out


def clocks_per_point(job: dict, points: dict) -> list:
    out = []
    for name, point in points.items():
        k_runs = [trial[arm]["row"] for trial in point["trials"].values() for arm in point["arms"]
                  if trial.get(arm) and trial[arm]["row"]]
        references = [member["row"] for trial in point["trials"].values()
                      for group in trial["references"].values() for member in group.get("members", [])
                      if member["row"]]
        k_values = [gpu_means(row, row.get("device_map") or []) for row in k_runs]
        reference_values = [gpu_means(row, row.get("device_map") or []) for row in references]

        def mean_of(values, key):
            items = [value[key] for value in values if value[key] is not None]
            return statistics.fmean(items) if items else None
        entry = {"point": name, "k_sm_clock": mean_of(k_values, "sm_clock"), "k_power": mean_of(k_values, "power"),
                 "reference_sm_clock": mean_of(reference_values, "sm_clock"),
                 "reference_power": mean_of(reference_values, "power")}
        if entry["k_sm_clock"] is not None and entry["reference_sm_clock"] is not None:
            entry["sm_clock_difference"] = entry["k_sm_clock"] - entry["reference_sm_clock"]
        if entry["k_power"] is not None and entry["reference_power"] is not None:
            entry["power_difference"] = entry["k_power"] - entry["reference_power"]
        out.append(entry)
    return out


def construction_rows(job: dict) -> list:
    out = []
    for entry in job["index"]:
        row = job["rows"].get(entry["label"])
        if not row or entry["role"] not in ("equal", "calibrated", "reference", "trace_equal", "trace_calibrated",
                                            "precheck", "soak", "anatomy", "selftest"):
            continue
        read_in = stage(row, "case_loaded")
        partition = stage(row, "partition_done")
        simulators = stage(row, "bootstrap_start")
        bootstrap_end = stage(row, "bootstrap_end")
        slabs = row.get("slab_count") or 1
        warm = entry["role"] in ("equal", "calibrated") or (entry["role"] == "reference" and entry["trial"] >= 2)
        out.append({"label": entry["label"], "role": entry["role"], "point": entry["point"], "case": entry["case"],
                    "K": slabs, "warm": warm,
                    "read_in_s": read_in,
                    "partition_s": partition - read_in if partition is not None and read_in is not None else None,
                    "simulators_s": simulators - partition if simulators is not None and partition is not None else None,
                    "construction_s": simulators,
                    "predicted_warm_s": read_in + 1.0 * slabs if read_in is not None else None,
                    "bootstrap_s": bootstrap_end - simulators if bootstrap_end is not None and simulators is not None
                    else None,
                    "build_s": row.get("build_seconds"), "obj_cache_hits": row.get("obj_cache_hits")})
    return out


def precheck_table(job: dict) -> list:
    out = []
    members: dict = {}
    for entry in job["index"]:
        if entry["role"] == "precheck":
            members.setdefault(group_label(entry["label"]), []).append(job["rows"].get(entry["label"]))
    for check in job["prechecks"]:
        label = f"pre_{check['case'].replace('cavity2d_', '2d_').replace('cavity3d_', '3d_')}_{check['kind']}_x{check['parties']}"
        rows = [row for row in members.get(label, []) if row]
        window = job["memory_windows"].get(label, {})
        vram = [own_vram_mib(row) for row in rows]
        vram = [value for value in vram if value is not None]
        out.append({"case": check["case"], "kind": check["kind"], "parties": check["parties"],
                    "verdict": check["verdict"], "members": len(rows),
                    "statuses": sorted({row["status"] for row in rows}),
                    "vram_peak_mib_max": max(vram) if vram else None, "vram_peak_mib_min": min(vram) if vram else None,
                    "host_process_peak_gib": (window.get("window_process_hwm_max_bytes") or 0) / 2 ** 30 or None,
                    "host_process_peak_sum_gib": (window.get("window_process_hwm_sum_bytes") or 0) / 2 ** 30 or None,
                    "node_in_use_peak_gib": (window.get("node_used_peak_bytes") or 0) / 2 ** 30 or None,
                    "build_s_max": max((row.get("build_seconds") or 0) for row in rows) if rows else None})
    return out


def soak_table(job: dict) -> list:
    """The extras (soak, selftest, anatomy, fulltrace): invariants, pool series, per-defrag watermarks, the host
    loop's [loop] intervals (V7_LOOP_TRACE=1), the loop rows the wrapper wrote (run_chain_v6.py --loop-trace:
    parse_run_v6 loop_trace = directory, rows, segments, error), the sampled anatomy frames (mean per simulator) and
    the defrag times."""
    out = []
    for entry in job["index"]:
        if entry["role"] not in ("soak", "selftest", "anatomy", "fulltrace"):
            continue
        row = job["rows"].get(entry["label"])
        if not row:
            out.append({"label": entry["label"], "status": "missing"})
            continue
        regions: dict = {}
        for series in row.get("pool_series", []):
            region = regions.setdefault(series["region"], {"capacity": series["capacity"], "windows": []})
            for index, peak in enumerate(series["peaks"]):
                if index >= len(region["windows"]):
                    region["windows"].append(0)
                region["windows"][index] = max(region["windows"][index], peak)
            region["capacity"] = min(value for value in (region["capacity"], series["capacity"]) if value) \
                if series["capacity"] else region["capacity"]
        defrags: dict = {}
        for report in row.get("defrag_reports", []):
            frame = defrags.setdefault(report["frame"], {"used_fraction_max": 0.0, "interval_migration_max": 0,
                                                         "overflow_install_tail": 0})
            frame["used_fraction_max"] = max(frame["used_fraction_max"], float(report.get("used_fraction", 0) or 0))
            frame["interval_migration_max"] = max(frame["interval_migration_max"],
                                                  int(report.get("interval_migration", 0) or 0))
            frame["overflow_install_tail"] += int(report.get("overflow_install_tail", 0) or 0)
        anatomy: dict = {}          # sim -> item -> values over the sampled frames (GPU µs)
        for frame_entry in row.get("anatomy_frames", []):
            items = anatomy.setdefault(frame_entry.get("sim", 0), {})
            for key, value in frame_entry.items():
                if key not in ("frame", "sim") and isinstance(value, (int, float)):
                    items.setdefault(key, []).append(float(value))
        defrag_time: dict = {}      # sim -> wall ms / GPU µs of every submit_defrag_and_wait (bootstrap included)
        for timing in row.get("defrag_times", []):
            bucket = defrag_time.setdefault(timing.get("sim", 0), {"wall_ms": [], "gpu_us": []})
            bucket["wall_ms"].append(float(timing["wall_ms"]))
            if timing.get("gpu_us") is not None:
                bucket["gpu_us"].append(float(timing["gpu_us"]))
        out.append({"label": entry["label"], "role": entry["role"], "status": row["status"],
                    "reasons": row.get("reasons"), "steps": row.get("total_steps"), "fps": precise_fps(row),
                    "drift": row.get("drift"), "overflow_total": row.get("overflow_total"),
                    "far_migration_total": row.get("far_migration_total"),
                    "stamp_errors": (row.get("stamp_errors_gpu"), row.get("stamp_errors_host")),
                    "seam_checks_ok": all(seam["ok"] for seam in row.get("seam_checks", [])) if row.get("seam_checks")
                    else None,
                    "regions": regions, "defrags": dict(sorted(defrags.items())),
                    "pool_health": [entry_sim.get("pool_health") for entry_sim in row.get("simulators", [])],
                    "anatomy_frames": len(row.get("anatomy_frames", [])), "loop_intervals": len(row.get("loop_intervals", [])),
                    "defrag_times": len(row.get("defrag_times", [])),
                    "loop": row.get("loop_intervals", []), "loop_trace": row.get("loop_trace"),
                    "anatomy": {sim: {key: {"mean": statistics.fmean(values), "n": len(values)}
                                      for key, values in items.items()} for sim, items in sorted(anatomy.items())},
                    "defrag_time": {sim: {name: {"mean": statistics.fmean(values), "max": max(values), "n": len(values)}
                                          for name, values in bucket.items() if values}
                                    for sim, bucket in sorted(defrag_time.items())}})
    return out


def validity(job: dict) -> dict:
    counts: dict = {}
    failures = []
    roles = {entry["label"]: entry["role"] for entry in job["index"]}
    for label, row in job["rows"].items():
        role = roles.get(label, "other")
        bucket = counts.setdefault(role, {})
        bucket[row["status"]] = bucket.get(row["status"], 0) + 1
        if row["status"] != "pass":
            failures.append({"label": label, "role": role, "status": row["status"], "reasons": row.get("reasons"),
                             "rc": row.get("rc")})
    missing = [entry["label"] for entry in job["index"]
               if entry["label"] not in job["rows"] and entry["role"] not in ("calibration", "reference_skipped",
                                                                                "cross_skipped")]
    return {"counts": counts, "failures": failures, "missing_rows": missing}


def same_tree(first, second) -> bool:
    """Two experiment/v7 tree hashes agree (either may be abbreviated)."""
    return bool(first) and bool(second) and (first.startswith(second) or second.startswith(first))


def solver_checks(job: dict) -> list:
    """Per solver tag of the result rows (parse_run_v6 solver_tag / solver_tree / repository_root; rows without
    them are left out): the runs, their experiment/v7 trees and repository roots, and the tree expected for the
    tag (expected_from names the provenance.json key): rc1_expected_tree (else rc1_tree) for rc1 when the job
    carries the rc1 deployment, else the job's expected_tree (else experiment_v7_tree). shares_tree_with names
    other tags that ran the same tree (then a pair compares a solver with itself)."""
    provenance = job["provenance"]
    tags: dict = {}
    for row in job["rows"].values():
        if row.get("solver_tag") is None and row.get("solver_tree") is None:
            continue
        entry = tags.setdefault(row.get("solver_tag") or "", {"runs": 0, "trees": set(), "roots": set()})
        entry["runs"] += 1
        entry["trees"].add(row.get("solver_tree") or "")
        entry["roots"].add(row.get("repository_root") or "")
    out = []
    for tag, entry in sorted(tags.items()):
        if tag == "rc1" and (provenance.get("rc1_expected_tree") or provenance.get("rc1_tree")):
            source = "rc1_expected_tree" if provenance.get("rc1_expected_tree") else "rc1_tree"
        else:
            source = "expected_tree" if provenance.get("expected_tree") else "experiment_v7_tree"
        expected = provenance.get(source) or ""
        out.append({"solver": tag, "runs": entry["runs"], "trees": sorted(entry["trees"]),
                    "roots": sorted(entry["roots"]), "expected_tree": expected, "expected_from": source,
                    "tree_match": all(same_tree(tree, expected) for tree in entry["trees"])})
    for entry in out:
        entry["shares_tree_with"] = [other["solver"] for other in out if other is not entry
                                     and any(same_tree(tree, other_tree) for tree in entry["trees"]
                                             for other_tree in other["trees"])]
    return out


def node_hours(jobs: list, plan: dict) -> list:
    out = []
    plan_hours = {}
    for batch, lines in (plan or {}).get("batches_hours", {}).items():
        for line, hours in lines.items():
            plan_hours[f"b{batch}{line}"] = hours
    for job in jobs:
        name = job["provenance"].get("job") or job["name"]
        for record in job["sacct"]:
            if "." in record.get("JobID", ""):
                continue
            elapsed = record.get("Elapsed", "")
            match = re.match(r"(?:(\d+)-)?(\d+):(\d+):(\d+)", elapsed)
            hours = None
            if match:
                days, hh, mm, ss = (int(value) if value else 0 for value in match.groups())
                hours = days * 24 + hh + mm / 60 + ss / 3600
            out.append({"job": name, "slurm_job": record.get("JobID"), "node": record.get("NodeList"),
                        "start": record.get("Start"), "end": record.get("End"), "state": record.get("State"),
                        "node_hours": hours, "planned_hours": plan_hours.get(name)})
    return out


# ----------------------------------------------------------------------------- report

def fmt(value, digits: int = 1, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{value:.{digits}f}{suffix}"


def tree_text(tree, name=None) -> str:
    """82dd6a740fa2 (v7-rc1): the hash's first 12 characters and its name (given, else TREE_NAMES')."""
    if not tree:
        return "unknown"
    name = name or next((label for full, label in TREE_NAMES.items() if same_tree(full, tree)), None)
    return tree[:12] + (f" ({name})" if name else "")


def provenance_end_line(provenance: dict, provenance_end):
    """The end-of-job provenance line from provenance_end.json (None = the file is absent): unchanged, CHANGED
    (bold, with the changes), not checked (an incomplete check) or unreadable, with where the check ran when it was
    the cleanup of a job that stopped early; a job with a solver_tag (the rc2 harness, which writes the file) but no
    file says so; a job without either (batch 1) gets no line (None)."""
    if provenance_end is None:
        return "- provenance at job end: **no provenance_end.json**" if provenance.get("solver_tag") else None
    if "unreadable" in provenance_end:
        return "- provenance at job end: **provenance_end.json unreadable**"
    where = provenance_end.get("checked_in")
    early = where and where != "job_end"
    suffix = f" (checked in the {where} cleanup: the job stopped before its end)" if early else ""
    if provenance_end.get("unchanged") is True:
        return "- provenance at job end: unchanged" + suffix
    if provenance_end.get("unchanged") is False:
        return ("- provenance at job end: **CHANGED DURING THE JOB:** "
                f"{(provenance_end.get('changes') or '').strip() or '(no changes listed)'}" + suffix)
    return (f"- provenance at job end: **not checked{f' ({where} cleanup)' if early else ''}: "
            f"{provenance_end.get('incomplete') or 'no verdict in provenance_end.json'}**")


def provenance_lines(provenance: dict, solvers: list, provenance_end=False) -> list:
    """One line per solver deployment: the job's own (solver_tag, root, experiment_v7_tree against expected_tree,
    named expected_label when they match) and, when the job also runs rc1, the rc1 deployment (rc1_root,
    rc1_tree against rc1_expected_tree, rc1_commit, rc1_harness_commit, its manifests); then the end-of-job line
    (provenance_end_line: provenance_end = the job's provenance_end.json, None when it has none; False, the default,
    leaves the line out); then one line per solver tag of the result rows: runs, trees and roots against the tree
    expected for the tag (solver_checks), marked as start-of-job hashes when the end check found a change (the rows
    carry the hashes taken at the start)."""
    match = provenance.get("tree_match")
    verdict = "= expected" if match else f"**MISMATCH**, expected {(provenance.get('expected_tree') or '')[:12]}"
    tag = provenance.get("solver_tag")
    tree = tree_text(provenance.get("experiment_v7_tree"), provenance.get("expected_label") if match else None)
    lines = [f"- job deployment{f' (solver {tag})' if tag else ''}: "
             + (f"root {provenance['root']}, " if provenance.get("root") else "")
             + f"experiment/v7 tree {tree} {verdict}, commit {(provenance.get('commit') or '')[:12] or '-'}"]
    if any(key.startswith("rc1_") for key in provenance):
        rc1_verdict = ""
        if "rc1_tree_match" in provenance:
            rc1_verdict = (" = expected" if provenance["rc1_tree_match"]
                           else f" **MISMATCH**, expected {(provenance.get('rc1_expected_tree') or '')[:12]}")
        extras = [f"{text} {provenance[key]}"
                  for key, text in (("rc1_manifest", "manifest"), ("rc1_scripts", "scripts")) if key in provenance]
        if provenance.get("rc1_match") is False:
            extras.append("**rc1 checks failed**")
        lines.append(f"- rc1 deployment (solver rc1): root {provenance.get('rc1_root')}, experiment/v7 tree "
                     f"{tree_text(provenance.get('rc1_tree'))}{rc1_verdict}, commit "
                     f"{(provenance.get('rc1_commit') or '')[:12] or '-'}, harness "
                     f"{(provenance.get('rc1_harness_commit') or '-')[:12]}"
                     + "".join(f", {extra}" for extra in extras))
    changed = False
    if provenance_end is not False:
        end_line = provenance_end_line(provenance, provenance_end)
        if end_line:
            lines.append(end_line)
        changed = bool(provenance_end) and provenance_end.get("unchanged") is False
    hashes = " (start-of-job hashes)" if changed else ""
    for entry in solvers:
        if not entry["expected_tree"]:
            check = "no expected tree in provenance.json"
        elif entry["trees"] == [""]:
            check = "no tree in the rows"
        elif entry["tree_match"]:
            check = f"= {entry['expected_from']}"
        else:
            check = f"**MISMATCH**, expected {entry['expected_tree'][:12]} ({entry['expected_from']})"
        name = (provenance.get("expected_label") if entry["tree_match"] and not entry["expected_from"].startswith("rc1")
                else None)
        lines.append(f"- runs tagged {entry['solver'] or '(no tag)'}{hashes}: {entry['runs']}, experiment/v7 tree "
                     f"{', '.join(tree_text(tree, name) for tree in entry['trees'])} {check}; root "
                     f"{', '.join(root or '-' for root in entry['roots'])}"
                     + (f"; **same tree as {', '.join(entry['shares_tree_with'])}**" if entry["shares_tree_with"]
                        else ""))
    return lines


def trace_text(trace: dict) -> str:
    """One traced run: r_chain first (the main ratio), then r, T_B and the fps against the untraced runs."""
    if trace.get("error") or trace.get("missing"):
        return f"{trace.get('status')} ({trace.get('error') or 'no trace files'})"
    versus = trace.get("fps_vs_untraced")
    return (f"{trace.get('status')}: r_chain {fmt(trace.get('r_chain_p50'), 3)}/{fmt(trace.get('r_chain_p95'), 3)}, "
            f"t_chain {fmt(trace.get('t_chain_p50_median_link'), 0)}/{fmt(trace.get('t_chain_p95_max_link'), 0)}; "
            f"r {fmt(trace.get('r_p50'), 3)}/{fmt(trace.get('r_p95'), 3)}, "
            f"t_tr {fmt(trace.get('t_tr_p50_median_link'), 0)}/{fmt(trace.get('t_tr_p95_max_link'), 0)}; "
            f"T_B {fmt(trace.get('T_B_p50_median_sim'), 0)} (max {fmt(trace.get('T_B_p50_max_sim'), 0)}), "
            f"fps {fmt(versus * 100 if versus is not None else None, 2, ' %')}")


def trial_values(stats: dict, digits: int, scale: float = 1.0, suffix: str = "") -> str:
    """t1 +0.12, t2 -0.05: the per-trial values of a describe() result that carries its trials."""
    text = ", ".join(f"t{trial} {value * scale:+.{digits}f}"
                     for trial, value in zip(stats.get("trials", []), stats.get("values", [])))
    return text + suffix if text else "-"


def standard_error_text(stats: dict, digits: int, scale: float = 1.0, suffix: str = "") -> str:
    """'; SE 0.0012' after a difference's describe() text (with_resolution; nothing when n < 2)."""
    if stats.get("standard_error") is None:
        return ""
    return f"; SE {stats['standard_error'] * scale:.{digits}f}{suffix}"


def position_text(position: dict, digits: int, unit: str = "") -> str:
    """The position term (position_summary): first - second slot, with the relative value when there is one."""
    if not position.get("n"):
        return "-"
    text = f"{position['difference']:+.{digits}f}{unit}"
    if position.get("relative") is not None:
        text += f" ({position['relative'] * 100:+.2f} %)"
    return text


def resolution_line(label: str, delta: dict, digits: int, scale: float, unit: str, pair: dict, position: dict,
                    eta_key=None) -> str:
    """One pair quantity's resolution: mean delta +- SE (n; 95 % half-width, t), for eta each solver's trial-to-trial
    eta SD, the position term, the leads per trial and the lead-balanced delta / position."""
    if not delta.get("n"):
        return f"- {label}: no trial with both runs"
    text = f"- {label}: Δ {delta['mean'] * scale:+.{digits}f}{unit}"
    if delta.get("standard_error") is not None:
        text += (f" ± {delta['standard_error'] * scale:.{digits}f}{unit} SE (n {delta['n']}; 95 % ± "
                 f"{delta['half_width_95'] * scale:.{digits}f}{unit}, t {t_quantile_975(delta['n'] - 1):.2f})")
    else:
        text += f" (n {delta['n']}, no SE)"
    if eta_key is not None:
        deviations = (pair.get("eta_trial_standard_deviation") or {}).get(eta_key) or {}
        text += "; eta SD over the trials " + ", ".join(f"{solver} {fmt(deviations.get(solver), 4)}"
                                                         for solver in (pair["baseline"], pair["solver"]))
    if position.get("n"):
        text += "; first − second K slot " + (position_text(position, digits) if eta_key is not None
                                              else position_text(position, 2, " fps"))
    leads = pair.get("leads") or []
    text += "; leads " + (", ".join(f"t{lead['trial']} {lead['first']}" for lead in leads) or "-")
    balanced = delta.get("lead_balanced")
    if balanced:
        text += (f"; lead-balanced Δ {balanced['delta'] * scale:+.{digits}f}{unit}, position "
                 f"{balanced['position'] * scale:+.{digits}f}{unit}")
    return text


def report(jobs: list, plan: dict, trace_root_override=None) -> tuple:
    lines, data = [], {"jobs": []}
    for job in jobs:
        points = collect_points(job)
        analyzed = [analyze_point(point) for point in points.values()]
        for item in analyzed:
            for arm, trace in item["traces"].items():          # traces/<point>_<arm>: E, C, C_rc1, ...
                directory = job["directory"] / "traces" / f"{item['point']}_{arm}"
                try:
                    trace.update(trace_metrics(directory))
                except Exception as error:                         # noqa: BLE001
                    trace["error"] = f"{type(error).__name__}: {error}"
        job_data = {"name": job["name"], "provenance": job["provenance"],
                    "provenance_end": job.get("provenance_end"), "solver_checks": solver_checks(job),
                    "shader_cache": job["shader_cache"],
                    "validity": validity(job), "prechecks": precheck_table(job), "points": analyzed,
                    "barrier": barrier_groups(job), "clocks": clocks_per_point(job, points),
                    "construction": construction_rows(job), "soak": soak_table(job)}
        data["jobs"].append(job_data)

        provenance = job["provenance"]
        lines += [f"## {job['name']}", "",
                  f"- host {provenance.get('host')}, slurm job {provenance.get('slurm_job')}, harness "
                  f"{(provenance.get('harness_commit') or '')[:12]}, driver {provenance.get('driver')}"]
        lines += provenance_lines(provenance, job_data["solver_checks"], job.get("provenance_end"))
        for check in job["shader_cache"]:
            lines.append(f"- shader cache ({check['stage']}): {check['verdict']}, {check['files']} files, "
                         f"{check['mib']} MiB in {check['directory']}; new files in ~/.cache/nvidia: "
                         f"{check['default_location_new_files']}")
        valid = job_data["validity"]
        lines.append("- runs: " + "; ".join(f"{role} " + ", ".join(f"{status} {count}" for status, count in
                                                                     sorted(statuses.items()))
                                            for role, statuses in sorted(valid["counts"].items())))
        for failure in valid["failures"]:
            lines.append(f"  - **{failure['label']}** ({failure['role']}): {failure['status']} rc={failure['rc']} "
                         f"{'; '.join(failure['reasons'] or [])}")
        if valid["missing_rows"]:
            lines.append(f"  - no result row: {', '.join(valid['missing_rows'])}")
        for item in analyzed:
            for collision in item["collisions"]:
                lines.append(f"  - **{item['point']}: two K runs for trial {collision['trial']}, arm "
                             f"{collision['arm']}: {collision['by']} replaced {collision['replaced']}** (solver tags?)")
            for mismatch in item["tag_mismatches"]:
                lines.append(f"  - **{item['point']}: solver tag of {mismatch['label']}: index "
                             f"{mismatch['index_solver'] or '(none)'}, results row "
                             f"{mismatch['row_solver_tag'] or '(none)'}**")
        lines.append("")

        if job_data["prechecks"]:
            lines += ["### Pre-checks", "",
                      "| case | kind | processes | verdict | statuses | VRAM peak per card (MiB, min–max) | "
                      "host: largest process / sum (GiB) | node in use peak (GiB) | build max (s) |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for check in job_data["prechecks"]:
                lines.append(f"| {check['case']} | {check['kind']} | {check['parties']} | {check['verdict']} | "
                             f"{','.join(check['statuses'])} | {fmt(check['vram_peak_mib_min'], 0)}–"
                             f"{fmt(check['vram_peak_mib_max'], 0)} | {fmt(check['host_process_peak_gib'])} / "
                             f"{fmt(check['host_process_peak_sum_gib'])} | {fmt(check['node_in_use_peak_gib'])} | "
                             f"{fmt(check['build_s_max'])} |")
            lines.append("")

        if analyzed:
            lines += ["### Points: fps and efficiency", "",
                      "| point | family | K | arm | fps | eta (type: reference) | eta_min |", "|---|---|---|---|---|---|---|"]
            for item in analyzed:
                for arm in item["arms"]:
                    eta_texts = [f"{key}: {text_of(values[arm]['eta'], 3)}" for key, values in item["eta"].items()]
                    eta_min_texts = [f"{key}: {text_of(values[arm]['eta_min'], 3)}" for key, values in item["eta"].items()]
                    lines.append(f"| {item['point']} | {item['family']} | {item['slabs']} | {arm} | "
                                 f"{text_of(item['fps'][arm], 2)} | {'<br>'.join(eta_texts) or '-'} | "
                                 f"{'<br>'.join(eta_min_texts) or '-'} |")
            lines += ["", "### Points: calibration, arm difference, traces", "",
                      "r_chain = t_chain / T_B is the main ratio (t_chain: the transport chain without the receiver's "
                      "two waits); r = t_tr / T_B contains those waits and with them the late wake-ups. Both per step "
                      "on the worse link; t_chain, t_tr: median of the links' p50 / largest p95.", "",
                      "| point | weights | cuts | rounds (cuts → next, changed) | C − E per trial | trace E: r_chain "
                      "p50/p95, t_chain p50/p95 µs; r p50/p95, t_tr p50/p95 µs; T_B p50 µs, fps vs untraced | "
                      "trace C: same |", "|---|---|---|---|---|---|---|"]
            for item in analyzed:
                calibration = item["calibration"]
                rounds = "; ".join(f"{entry['cuts']}→{entry['next_cuts']} {'changed' if entry['changed'] else 'kept'}"
                                   for entry in calibration["rounds"]) or "-"
                trace_texts = []
                for name in ARMS:           # one column per arm; a pair point's traces per solver in one cell
                    texts = [(f"{split_arm(arm)[1]}: " if split_arm(arm)[1] else "") + trace_text(trace)
                             for arm, trace in item["traces"].items() if split_arm(arm)[0] == name]
                    trace_texts.append("<br>".join(texts) or "-")
                differences = ([f"{text_of(item['arm_difference'], 2, 100.0)} %"] if item["arm_difference"]["n"]
                               else [])
                differences += [f"{solver}: {text_of(stats, 2, 100.0)} %"
                                for solver, stats in item["arm_difference_by_solver"].items()]
                weights = ", ".join(f"{value:.4f}" for value in calibration["weights"] or [])
                lines.append(f"| {item['point']} | {weights or '-'} | {calibration['cuts'] or '-'} | {rounds} | "
                             f"{'<br>'.join(differences) or '-'} | {trace_texts[0]} | {trace_texts[1]} |")
            lines.append("")
            pairs = [(item, pair) for item in analyzed for pair in item["solver_pairs"]]
            for baseline, solver in sorted({(pair["baseline"], pair["solver"]) for _, pair in pairs}):
                wrapper = ""
                if baseline == "rc1" and any(key.startswith("rc1_") for key in provenance):
                    wrapper = (f" The {baseline} runs ran the rc1 deployment's own wrapper (its run_chain_v6.py, harness "
                               f"{(provenance.get('rc1_harness_commit') or '-')[:12]}), the {solver} runs the job's "
                               f"(harness {(provenance.get('harness_commit') or '-')[:12]}): a wrapper-side difference "
                               f"is part of Δ.")
                lines += [f"### {baseline} vs {solver} pairs", "",
                          f"Per trial, both runs against the same reference set: Δfps = (fps_{solver} − "
                          f"fps_{baseline}) / fps_{baseline}, Δeta = eta_{solver} − eta_{baseline}; over the trials "
                          f"mean ± std (med; min–max; n) and the standard error SE = std / √n, no mode test on the "
                          f"differences. First − second K slot: the mean of the K run right after the reference set "
                          f"minus the mean of the one after it (the position term).{wrapper}", "",
                          f"| point | K | arm | quantity | {baseline} | {solver} | {solver} − {baseline} | per trial | "
                          f"first − second K slot |", "|---|---|---|---|---|---|---|---|---|"]
                resolution = []
                for item, pair in pairs:
                    if (pair["baseline"], pair["solver"]) != (baseline, solver):
                        continue
                    first = f"| {item['point']} | {item['slabs']} | {pair['arm']} |"
                    delta = pair["delta_fps"]
                    position = pair.get("position_fps") or {"n": 0}
                    lines.append(f"{first} fps | {text_of(item['fps'][pair['baseline_arm']], 2)} | "
                                 f"{text_of(item['fps'][pair['solver_arm']], 2)} | "
                                 f"{text_of(delta, 3, 100.0) + ' %' + standard_error_text(delta, 3, 100.0, ' %') if delta['n'] else '-'} | "
                                 f"{trial_values(delta, 3, 100.0, ' %')} | "
                                 f"{position_text(position, 2, ' fps')} |")
                    resolution.append(resolution_line(f"{item['point']} {pair['arm']} fps", delta, 3, 100.0, " %",
                                                      pair, position))
                    for key, values in item["eta"].items():
                        for name in ("eta", "eta_min"):
                            delta = pair["delta_" + name].get(key, {"n": 0})
                            position = (pair.get("position_" + name) or {}).get(key, {"n": 0})
                            lines.append(f"{first} {name} {key} | {text_of(values[pair['baseline_arm']][name], 4)} | "
                                         f"{text_of(values[pair['solver_arm']][name], 4)} | {text_of(delta, 4)}"
                                         f"{standard_error_text(delta, 4)} | {trial_values(delta, 4)} | "
                                         f"{position_text(position, 4)} |")
                            if name == "eta":
                                resolution.append(resolution_line(f"{item['point']} {pair['arm']} eta {key}", delta, 4,
                                                                  1.0, "", pair, position, key))
                lines += ["", f"Resolution over the trials (95 % = t(n − 1) × SE). With unequal leads the position "
                              f"term enters the mean Δ with weight (leads of {solver} − leads of {baseline}) / n; the "
                              f"lead-balanced Δ weighs both orders equally (Δ = mean when {solver} led + mean when "
                              f"{baseline} led, halved; position = half their difference).", ""]
                lines += resolution
                lines.append("")
            guarded = [(item, arm, sums) for item in analyzed for arm, sums in (item.get("dest_guard") or {}).items()]
            if guarded:
                lines += ["### Dest guards (E7 B2, rc2's transport)", "",
                          "Summed over the arm's runs (timed and traced) that print the worker lines (parse_run_v6 "
                          "dest_guard); bold: fallback or blocking > 0 in a run with V7_DEST_GUARD=relay, whose dest "
                          "guards should not sleep in the driver.", ""]
                for item, arm, sums in guarded:
                    text = (f"{item['point']} {arm}: {sums['runs']} of {sums['runs_total']} runs ({sums['relay_runs']} "
                            f"with V7_DEST_GUARD=relay): precheck {sums['precheck']}, relay {sums['relay']}, fallback "
                            f"{sums['fallback']}, blocking {sums['blocking']} (relay slept {sums['relay_slept']}, "
                            f"released {sums['released']})")
                    lines.append(f"- **{text}**" if sums["flagged"] else f"- {text}")
                lines.append("")

        barrier = [entry for entry in job_data["barrier"]]
        if barrier:
            lines += ["### Start barrier (simultaneous groups)", "",
                      "| group | members | statuses | longest wait (s) | release spread (ms) | loop start spread (ms) | "
                      "steady start spread (ms) |", "|---|---|---|---|---|---|---|"]
            for entry in barrier:
                lines.append(f"| {entry['group']} | {entry['members']} | {','.join(entry['statuses'])} | "
                             f"{fmt(entry['longest_wait_s'])} | {fmt(entry['release_spread_ms'], 0)} | "
                             f"{fmt(entry['loop_start_spread_ms'], 0)} | {fmt(entry['steady_start_spread_ms'], 0)} |")
            lines.append("")

        if job_data["clocks"]:
            lines += ["### Clocks and power (loop window medians, mean over GPUs)", "",
                      "| point | K runs: SM MHz / W | references: SM MHz / W | difference K − reference |",
                      "|---|---|---|---|"]
            for entry in job_data["clocks"]:
                lines.append(f"| {entry['point']} | {fmt(entry['k_sm_clock'], 0)} / {fmt(entry['k_power'], 0)} | "
                             f"{fmt(entry['reference_sm_clock'], 0)} / {fmt(entry['reference_power'], 0)} | "
                             f"{fmt(entry.get('sm_clock_difference'), 0)} MHz / {fmt(entry.get('power_difference'), 0)} W |")
            lines.append("")

        warm = [entry for entry in job_data["construction"] if entry["warm"]]
        if warm:
            groups: dict = {}
            for entry in warm:
                kind = "K runs" if entry["role"] in ("equal", "calibrated") else "references"
                groups.setdefault((entry["point"], kind, entry["K"]), []).append(entry)
            lines += ["### Construction of warm runs (mean over the runs; s since the bench started)", "",
                      "| point | runs | K | n | read-in | partition | contexts + simulators | to bootstrap start | "
                      "read-in + 1 s × K | bootstrap |", "|---|---|---|---|---|---|---|---|---|---|"]

            def mean_of(entries, key):
                values = [entry[key] for entry in entries if entry[key] is not None]
                return statistics.fmean(values) if values else None
            for (point_name, kind, slabs), entries in groups.items():
                lines.append(f"| {point_name} | {kind} | {slabs} | {len(entries)} | {fmt(mean_of(entries, 'read_in_s'), 2)} | "
                             f"{fmt(mean_of(entries, 'partition_s'), 2)} | {fmt(mean_of(entries, 'simulators_s'), 2)} | "
                             f"{fmt(mean_of(entries, 'construction_s'), 2)} | {fmt(mean_of(entries, 'predicted_warm_s'), 2)} | "
                             f"{fmt(mean_of(entries, 'bootstrap_s'), 2)} |")
            lines.append("")

        for soak in job_data["soak"]:
            lines += [f"### {soak['role']} {soak['label']}", "",
                      f"- status {soak['status']} {soak.get('reasons') or ''}; steps {soak.get('steps')}, fps "
                      f"{fmt(soak.get('fps'), 1)}; drift {soak.get('drift')}, overflow {soak.get('overflow_total')}, "
                      f"far migration {soak.get('far_migration_total')}, stamps {soak.get('stamp_errors')}, seam checks "
                      f"{soak.get('seam_checks_ok')}; anatomy frames {soak.get('anatomy_frames')}, [loop] intervals "
                      f"{soak.get('loop_intervals')}, defrag times {soak.get('defrag_times')}"]
            loop_trace = soak.get("loop_trace")
            if loop_trace:              # run_chain_v6.py --loop-trace (item 11): bold when it wrote no rows or failed
                text = (f"loop rows {loop_trace.get('rows')} in {loop_trace.get('segments')} segments "
                        f"({loop_trace.get('directory')})"
                        + (f", error {loop_trace['error']}" if loop_trace.get("error") else ""))
                lines.append(f"- **{text}**" if not loop_trace.get("rows") or loop_trace.get("error") else f"- {text}")
            for region, values in (soak.get("regions") or {}).items():
                capacity = values["capacity"]
                peaks = values["windows"]
                lines.append(f"- pool {region} (capacity {capacity}): peak per 1000 frames, max over links: "
                             + ", ".join(str(peak) for peak in peaks)
                             + (f" (max {max(peaks) / capacity:.4f} of capacity)" if capacity and peaks else ""))
            if soak.get("defrags"):
                lines.append("- own pool watermark per defrag (max over slabs: used fraction / interval migration): "
                             + ", ".join(f"f{frame}: {entry['used_fraction_max']:.4f}/{entry['interval_migration_max']}"
                                         for frame, entry in soak["defrags"].items()))
            for loop in soak.get("loop") or []:
                lines.append(f"- host loop f{loop.get('frame')} ({loop.get('frames')} frames, V7_LOOP_TRACE): period p50 "
                             f"{loop.get('period_ms_p50')} ms (max {loop.get('period_ms_max')}); in _submit_frame p50 "
                             f"{loop.get('submit_ms_p50')} / mean {loop.get('submit_ms_mean')} / max {loop.get('submit_ms_max')} "
                             f"ms; blocked in frame waits p50 {loop.get('wait_ms_p50')} / mean {loop.get('wait_ms_mean')} ms; "
                             f"cpu share {loop.get('cpu_share')}")
            for sim, timings in (soak.get("defrag_time") or {}).items():
                parts = [f"{name} mean {fmt(values['mean'], 1)} (max {fmt(values['max'], 1)}, n={values['n']})"
                         for name, values in timings.items()]
                lines.append(f"- defrag sim{sim} (every call, bootstrap included): " + "; ".join(parts))
            anatomy = soak.get("anatomy") or {}
            if anatomy:
                simulators = list(anatomy)
                items = []
                for values in anatomy.values():
                    items += [key for key in values if key not in items]
                lines += ["", f"GPU µs per frame, mean over the sampled frames (n = "
                          f"{max(value['n'] for values in anatomy.values() for value in values.values())} per simulator):", "",
                          "| item | " + " | ".join(f"sim{sim}" for sim in simulators) + " |",
                          "|---|" + "---|" * len(simulators)]
                for key in items:
                    lines.append(f"| {key} | " + " | ".join(fmt(anatomy[sim][key]["mean"], 1) if key in anatomy[sim] else "-"
                                                           for sim in simulators) + " |")
            lines.append("")
    hours = node_hours(jobs, plan)
    data["node_hours"] = hours
    if hours:
        lines += ["## Node-hours", "", "| job | slurm job | node | start | end | state | used (h) | planned (h) |",
                  "|---|---|---|---|---|---|---|---|"]
        for entry in hours:
            lines.append(f"| {entry['job']} | {entry['slurm_job']} | {entry['node']} | {entry['start']} | {entry['end']} | "
                         f"{entry['state']} | {fmt(entry['node_hours'], 2)} | {fmt(entry['planned_hours'], 2)} |")
        lines.append("")
    return "\n".join(lines), data


# ----------------------------------------------------------------------------- eta-r figure

def eta_r_rows(data: dict) -> list:
    """One row per job, point, arm and reference: eta over the trials of the arm's timing runs and the ratios of
    the same arm's traced run (empty without one). Reads the --json content; a point without 'arms' / 'solvers'
    (the batch-1 JSON) has the arms E and C of one untagged solver."""
    rows = []
    for job in data.get("jobs", []):
        for item in job.get("points", []):
            solvers = item.get("solvers") or [""]
            for arm in item.get("arms") or list(ARMS):
                name, solver = split_arm(arm)
                trace = (item.get("traces") or {}).get(arm) or {}
                for reference, values in (item.get("eta") or {}).items():
                    eta = (values.get(arm) or {}).get("eta") or {}
                    eta_min = (values.get(arm) or {}).get("eta_min") or {}
                    if not eta.get("n"):
                        continue
                    rows.append({"job": job.get("name"), "point": item["point"], "arm": name,
                                 "solver": solver or (solvers[0] if len(solvers) == 1 else ""),
                                 "K": item.get("slabs"), "reference": reference, "trials": eta["n"],
                                 "eta_mean": eta.get("mean"), "eta_std": eta.get("std"),
                                 "eta_trial_min": eta.get("min"), "eta_trial_max": eta.get("max"),
                                 "eta_min_mean": eta_min.get("mean"),
                                 "r_chain_p50": trace.get("r_chain_p50"), "r_chain_p95": trace.get("r_chain_p95"),
                                 "r_p50": trace.get("r_p50"), "r_p95": trace.get("r_p95"),
                                 "trace_status": trace.get("status") or ""})
    return rows


def eta_r_figure(data: dict, path: pathlib.Path) -> list:
    """The eta-r figure (PNG at path) and its numbers (eta_r_rows, CSV next to it with the same name). Per point,
    arm and reference with a traced run that passed: x = r_chain p50 of the trace (filled marker) with a line to
    its p95, y = eta mean +- std over the trials of the SAME arm's timing runs; r p50 as an open marker at the same
    eta. Color = arm, marker shape = reference kind, the number = the point (key in the legend; a pair point's
    solver after it); log x. Upper panel all rows, lower panel the eta band of the rows with eta std <=
    FIGURE_ZOOM_STANDARD_DEVIATION (the reproducible points), when that is a narrower band. Returns the two paths."""
    rows = eta_r_rows(data)
    csv_path = path.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ETA_R_COLUMNS)
        writer.writeheader()
        writer.writerows({key: "" if row[key] is None else row[key] for key in ETA_R_COLUMNS} for row in rows)

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter, NullFormatter

    def finite(value):
        return isinstance(value, (int, float)) and math.isfinite(value)
    plotted = [row for row in rows if finite(row["r_chain_p50"]) and row["r_chain_p50"] > 0
               and finite(row["eta_mean"]) and row["trace_status"] in ("pass", "threshold_only")]
    numbers, solvers, jobs = {}, {}, {}
    for row in plotted:
        numbers.setdefault((row["job"], row["point"]), len(numbers) + 1)
        solvers.setdefault((row["job"], row["point"]), set()).add(row["solver"])
        jobs.setdefault(row["point"], set()).add(row["job"])
    reproducible = [row["eta_mean"] for row in plotted
                    if finite(row["eta_std"]) and row["eta_std"] <= FIGURE_ZOOM_STANDARD_DEVIATION]
    everything = [value for row in plotted for value in (row["eta_mean"] - (row["eta_std"] or 0.0),
                                                         row["eta_mean"] + (row["eta_std"] or 0.0))]
    zoom = None
    if len(reproducible) >= 2 and everything:
        zoom = (min(reproducible) - 0.01, max(reproducible) + 0.01)
        if zoom[1] - zoom[0] > 0.5 * (max(everything) - min(everything)):
            zoom = None
    ink, secondary, muted, hairline = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    figure, axes = pyplot.subplots(2 if zoom else 1, 1, figsize=(10, 9.5 if zoom else 6), sharex=True, squeeze=False)
    axes = [axis for (axis,) in axes]
    for panel, axis in enumerate(axes):
        for row in plotted:
            color = FIGURE_ARM_COLORS.get(row["arm"], muted)
            marker = FIGURE_REFERENCE_MARKERS.get(row["reference"].split(":")[0], "o")
            eta = row["eta_mean"]
            if finite(row["r_chain_p95"]):
                axis.plot([row["r_chain_p50"], row["r_chain_p95"]], [eta, eta], "-", color=color, lw=1.5, alpha=0.5,
                          solid_capstyle="round", zorder=2)
            axis.errorbar(row["r_chain_p50"], eta, yerr=row["eta_std"] or 0.0, fmt="none", ecolor=color,
                          elinewidth=1.2, capsize=2.5, zorder=3)
            axis.plot(row["r_chain_p50"], eta, marker, color=color, ms=8, mec="white", mew=1.5, zorder=4)
            if finite(row["r_p50"]) and row["r_p50"] > 0:
                axis.plot(row["r_p50"], eta, marker, mfc="white", mec=color, mew=1.4, ms=7, zorder=4)
            key = (row["job"], row["point"])
            label = str(numbers[key]) + (f" {row['solver']}" if len(solvers[key]) > 1 and row["solver"] else "")
            axis.annotate(label, (row["r_chain_p50"], eta), xytext=(4, 3), textcoords="offset points",
                          fontsize=7.5, color=secondary, zorder=5, annotation_clip=True)
        if plotted:
            axis.set_xscale("log")
            axis.xaxis.set_major_formatter(FormatStrFormatter("%g"))
            axis.axvline(1.0, color=muted, lw=0.8, zorder=1)
        if panel == 1:
            axis.set_ylim(*zoom)
            axis.set_title(f"zoom: the eta band of the rows whose eta std over the trials is at most "
                           f"{FIGURE_ZOOM_STANDARD_DEVIATION}", loc="left", fontsize=9, color=secondary)
        axis.set_ylabel("eta: mean ± std over the trials\nof the traced run's arm", color=ink)
        axis.grid(True, which="major", color=hairline, lw=0.8)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_color("#c3c2b7")
        axis.tick_params(which="both", colors=secondary, labelsize=8, labelbottom=True)
    if not plotted:
        axes[0].text(0.5, 0.5, "no traced run with r_chain", transform=axes[0].transAxes, ha="center",
                     color=secondary)
    else:
        axes[0].annotate("ratio 1: as long as phase B", (1.0, 0.0), xycoords=("data", "axes fraction"),
                         xytext=(4, 5), textcoords="offset points", fontsize=7, color=muted)
        lower, upper = axes[0].get_xlim()       # under one decade the minor ticks carry the labels, in plain numbers
        for axis in axes:
            axis.xaxis.set_minor_formatter(FormatStrFormatter("%g") if upper < 10 * lower else NullFormatter())
    axes[-1].set_xlabel("transport ratio per step on the worse link: r_chain = t_chain / T_B, p50 (filled) with a "
                        "line to p95;\nopen marker: r = t_tr / T_B p50 (contains the receiver's two waits)", color=ink)
    axes[0].set_title("E7: efficiency against the transport ratio (eta and r of the same arm)", loc="left",
                      fontsize=11, color=ink)

    def header(text):
        return Line2D([], [], ls="", marker="", label=text)
    handles = [header("color: arm")]
    handles += [Line2D([], [], ls="", marker="o", ms=8, color=FIGURE_ARM_COLORS[arm], label=text)
                for arm, text in (("E", "E, equal weights"), ("C", "C, calibrated"))]
    handles += [header("shape: eta against the reference")]
    handles += [Line2D([], [], ls="", marker=FIGURE_REFERENCE_MARKERS.get(kind, "o"), ms=7, color=muted, label=kind)
                for kind in sorted({row["reference"].split(":")[0] for row in plotted})]
    handles += [header("fill: ratio"),
                Line2D([], [], ls="-", lw=1.5, marker="o", ms=7, color=muted, label="r_chain p50, line to p95"),
                Line2D([], [], ls="", marker="o", ms=7, mfc="white", mec=muted, mew=1.4, label="r p50"),
                header("points")]
    handles += [header(f"{number}  {point}" + (f" ({job})" if len(jobs[point]) > 1 else ""))
                for (job, point), number in numbers.items()]
    axes[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8,
                   handlelength=1.6)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    pyplot.close(figure)
    return [path, csv_path]


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return jsonable(value.item())
    if isinstance(value, pathlib.Path):
        return str(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="E7 full campaign batch report")
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--plan", default=None, help="e7_plan.py --json output (planned hours per job)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--figure", default=None,
                        help="eta-r figure (PNG); its numbers go to a CSV of the same name next to it")
    arguments = parser.parse_args()
    plan = json.loads(pathlib.Path(arguments.plan).read_text(encoding="utf-8")) if arguments.plan else {}
    jobs = [load_job(pathlib.Path(directory)) for directory in arguments.directories]
    text, data = report(jobs, plan)
    if arguments.json:
        pathlib.Path(arguments.json).write_text(json.dumps(jsonable(data), indent=1), encoding="utf-8")
    # the report first: a figure that fails (no matplotlib, no such directory) must not cost the markdown
    if arguments.out:
        pathlib.Path(arguments.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.buffer.write((text + "\n").encode("utf-8"))
        sys.stdout.flush()
    if arguments.figure:
        try:
            for path in eta_r_figure(jsonable(data), pathlib.Path(arguments.figure)):
                print(f"wrote {path}", file=sys.stderr)
        except Exception as error:                                     # noqa: BLE001
            print(f"eta-r figure {arguments.figure} failed: {type(error).__name__}: {error} (the report is written)",
                  file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
