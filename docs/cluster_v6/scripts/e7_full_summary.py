"""
e7_full_summary.py — E7 full campaign: the batch report's tables (markdown) and numbers (JSON) from the log
directories of one or more batch jobs (the e7_lib.sh layout, mirrored from ~/run/logs/e7_b<batch><line>_<id>).

Per job directory: results.jsonl (parse_run_v6.py rows), index.tsv (e7_lib.sh: label, role, family, case, K,
trial, reference kind, reference case, point, line, host), provenance.json, shader_cache.jsonl, prechecks.jsonl,
memory_windows.jsonl, weights/<point>.json, traces/<point>_{E,C}/ and, optional, sacct.txt (sacct -P -X with
JobID, JobName, Start, End, Elapsed, NodeList, State); --plan takes the JSON of e7_plan.py --json for the
node-hour comparison.

Definitions (user, 2026-10-10):
  fps of a run = the bench's steady window: steady_steps / steady_seconds;
  per trial and arm, against the reference set of the same trial: strong eta = fps_K / (K mean fps_1), weak (F3,
  F4: the reference is the per-card case) eta = fps_K / mean fps_1, pairs (n11320, or a K = 1 reference the
  pre-check found too large) eta = fps_K / ((K / 2) mean fps_pair), flagged; eta_min the same with the slowest
  card of the set (min instead of mean);
  over the trials: mean +- std (sample), median, min / max, and the modes when the largest gap between the sorted
  values exceeds 10 % of the median (the count of each mode);
  arm difference = (fps_C - fps_E) / fps_E per trial (the two runs of one trial are adjacent);
  traces (experiment/v7/analysis/step_trace_model.run_summary): r = t_tr / T_B on the worse link of each step,
  p50 / p95; t_tr per link (median of the links' p50, largest p95); T_B per sim (median and largest p50); the
  traced run's fps against the mean of the same arm's timing runs;
  barrier: per simultaneous group the spread (max - min) over its members of the release time, of the loop start
  and of the steady-window start (loop start + loop seconds - steady seconds);
  clocks: per run the median SM clock and power of its GPUs in the loop window (telemetry_loop); per point the
  mean over the K runs' GPUs against the mean over the reference members' own GPUs;
  construction: read-in (case_loaded, bench imports included), partition, contexts + simulators (bootstrap_start -
  partition_done), bootstrap; warm runs (timing arms; reference sets after the first trial) against read-in +
  1 s x K.

Usage:
    python docs/cluster_v6/scripts/e7_full_summary.py JOB_DIR [JOB_DIR ...] [--plan PLAN.json] [--out REPORT.md]
        [--json OUT.json]
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
                 "line", "host")
MODE_GAP = 0.10


# ----------------------------------------------------------------------------- loading

def read_json_lines(path: pathlib.Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_job(directory: pathlib.Path) -> dict:
    rows = {}
    for row in read_json_lines(directory / "results.jsonl"):
        rows[row["label"]] = row
    index = []
    if (directory / "index.tsv").exists():
        for line in (directory / "index.tsv").read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) >= len(INDEX_COLUMNS):
                entry = dict(zip(INDEX_COLUMNS, fields))
                entry["slabs"] = int(entry["slabs"]) if entry["slabs"].isdigit() else entry["slabs"]
                entry["trial"] = int(entry["trial"]) if entry["trial"].isdigit() else 0
                index.append(entry)
    weights = {}
    for path in sorted((directory / "weights").glob("*.json")) if (directory / "weights").exists() else []:
        try:
            weights[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            weights[path.stem] = None
    provenance_path = directory / "provenance.json"
    return {"directory": directory, "name": directory.name, "rows": rows, "index": index, "weights": weights,
            "provenance": json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {},
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
    if not row:
        return None
    steps, seconds = row.get("steady_steps"), row.get("steady_seconds")
    if steps and seconds:
        return steps / seconds
    return row.get("steady_fps")


def describe(values: list) -> dict:
    values = [value for value in values if value is not None and math.isfinite(value)]
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    out = {"n": len(values), "mean": statistics.fmean(values),
           "std": statistics.stdev(values) if len(values) > 1 else 0.0,
           "median": statistics.median(values), "min": ordered[0], "max": ordered[-1], "values": values}
    if len(values) >= 3:
        gaps = [(ordered[index + 1] - ordered[index], index) for index in range(len(ordered) - 1)]
        gap, position = max(gaps)
        if gap > MODE_GAP * out["median"]:
            low, high = ordered[:position + 1], ordered[position + 1:]
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
    if start is None or row.get("loop_seconds") is None or row.get("steady_seconds") is None:
        return None
    return start + row["loop_seconds"] - row["steady_seconds"]


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


def collect_points(job: dict) -> dict:
    """point -> {family, case, slabs, line, host, trials {t: {E, C, references {group: {...}}}}, traces, calibration}."""
    points: dict = {}
    for entry in job["index"]:
        role = entry["role"]
        if role not in ("calibration", "equal", "calibrated", "reference", "reference_skipped", "trace_equal",
                        "trace_calibrated"):
            continue
        point = points.setdefault(entry["point"], {"point": entry["point"], "family": None, "case": None,
                                                   "slabs": None, "line": entry["line"], "host": entry["host"],
                                                   "trials": {}, "traces": {}, "job": job["name"]})
        if role != "reference":
            point["family"], point["case"], point["slabs"] = entry["family"], entry["case"], entry["slabs"]
        if role == "calibration":
            continue
        row = job["rows"].get(entry["label"])
        if role in ("trace_equal", "trace_calibrated"):
            point["traces"]["E" if role == "trace_equal" else "C"] = {"label": entry["label"], "row": row}
            continue
        trial = point["trials"].setdefault(entry["trial"], {"references": {}})
        if role in ("equal", "calibrated"):
            trial["E" if role == "equal" else "C"] = {"label": entry["label"], "row": row}
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


def analyze_point(point: dict) -> dict:
    out = {key: point[key] for key in ("point", "family", "case", "slabs", "line", "host", "job")}
    arms = {"E": [], "C": []}
    etas: dict = {}
    differences = []
    for trial_number, trial in sorted(point["trials"].items()):
        fps = {}
        for arm in ("E", "C"):
            run = trial.get(arm)
            value = precise_fps(run["row"]) if run and run["row"] and run["row"].get("status") == "pass" else None
            fps[arm] = value
            arms[arm].append(value)
        if fps.get("E") and fps.get("C"):
            differences.append((fps["C"] - fps["E"]) / fps["E"])
        for group_name, group in trial["references"].items():
            if group.get("skipped"):
                continue
            for arm in ("E", "C"):
                result = efficiency(fps.get(arm), point["slabs"], group, point["case"])
                if result is None:
                    continue
                eta, eta_min, kind = result
                key = f"{kind}:{group['case']}"
                bucket = etas.setdefault(key, {"E": {"eta": [], "eta_min": []}, "C": {"eta": [], "eta_min": []}})
                bucket[arm]["eta"].append(eta)
                bucket[arm]["eta_min"].append(eta_min)
    out["fps"] = {arm: describe(values) for arm, values in arms.items()}
    out["eta"] = {key: {arm: {name: describe(values) for name, values in measures.items()}
                        for arm, measures in bucket.items()} for key, bucket in etas.items()}
    out["arm_difference"] = describe(differences)
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
        untraced = out["fps"][arm].get("mean")
        out["traces"][arm] = {"label": trace["label"], "status": row.get("status") if row else "missing",
                              "fps": traced,
                              "fps_vs_untraced": (traced - untraced) / untraced if traced and untraced else None}
    return out


def trace_metrics(directory: pathlib.Path) -> dict:
    """step_trace_model.run_summary of one trace directory, reduced to the report's numbers."""
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
    t_tr_p50 = finite([entry.get("t_tr_p50") for entry in links.values()])
    t_tr_p95 = finite([entry.get("t_tr_p95") for entry in links.values()])
    t_b = finite([entry.get("T_B_p50_us") for entry in sims])
    return {"r_p50": summary.get("r_worst_p50"), "r_p95": summary.get("r_worst_p95"),
            "r_chain_p50": summary.get("r_chain_worst_p50"), "r_chain_p95": summary.get("r_chain_worst_p95"),
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
        k_runs = [trial[arm]["row"] for trial in point["trials"].values() for arm in ("E", "C")
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
    out = []
    for entry in job["index"]:
        if entry["role"] not in ("soak", "selftest"):
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
                    "defrag_times": len(row.get("defrag_times", []))})
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


def report(jobs: list, plan: dict, trace_root_override=None) -> tuple:
    lines, data = [], {"jobs": []}
    for job in jobs:
        points = collect_points(job)
        analyzed = [analyze_point(point) for point in points.values()]
        for item in analyzed:
            for arm in ("E", "C"):
                directory = job["directory"] / "traces" / f"{item['point']}_{arm}"
                if arm in item["traces"]:
                    try:
                        item["traces"][arm].update(trace_metrics(directory))
                    except Exception as error:                     # noqa: BLE001
                        item["traces"][arm]["error"] = f"{type(error).__name__}: {error}"
        job_data = {"name": job["name"], "provenance": job["provenance"], "shader_cache": job["shader_cache"],
                    "validity": validity(job), "prechecks": precheck_table(job), "points": analyzed,
                    "barrier": barrier_groups(job), "clocks": clocks_per_point(job, points),
                    "construction": construction_rows(job), "soak": soak_table(job)}
        data["jobs"].append(job_data)

        provenance = job["provenance"]
        lines += [f"## {job['name']}", "",
                  f"- host {provenance.get('host')}, slurm job {provenance.get('slurm_job')}, harness "
                  f"{(provenance.get('harness_commit') or '')[:12]}, experiment/v7 tree "
                  f"{(provenance.get('experiment_v7_tree') or '')[:12]} "
                  f"({'= v7-rc1' if provenance.get('tree_match') else 'MISMATCH'}), driver {provenance.get('driver')}"]
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
                for arm in ("E", "C"):
                    eta_texts = [f"{key}: {text_of(values[arm]['eta'], 3)}" for key, values in item["eta"].items()]
                    eta_min_texts = [f"{key}: {text_of(values[arm]['eta_min'], 3)}" for key, values in item["eta"].items()]
                    lines.append(f"| {item['point']} | {item['family']} | {item['slabs']} | {arm} | "
                                 f"{text_of(item['fps'][arm], 2)} | {'<br>'.join(eta_texts) or '-'} | "
                                 f"{'<br>'.join(eta_min_texts) or '-'} |")
            lines += ["", "### Points: calibration, arm difference, traces", "",
                      "| point | weights | cuts | rounds (cuts → next, changed) | C − E per trial | trace E: r p50/p95, "
                      "t_tr p50/p95 µs, T_B p50 µs, fps vs untraced | trace C: same |", "|---|---|---|---|---|---|---|"]
            for item in analyzed:
                calibration = item["calibration"]
                rounds = "; ".join(f"{entry['cuts']}→{entry['next_cuts']} {'changed' if entry['changed'] else 'kept'}"
                                   for entry in calibration["rounds"]) or "-"
                trace_texts = []
                for arm in ("E", "C"):
                    trace = item["traces"].get(arm)
                    if not trace:
                        trace_texts.append("-")
                        continue
                    if trace.get("error") or trace.get("missing"):
                        trace_texts.append(f"{trace.get('status')} ({trace.get('error') or 'no trace files'})")
                        continue
                    trace_texts.append(f"{trace.get('status')}: r {fmt(trace.get('r_p50'), 3)}/{fmt(trace.get('r_p95'), 3)}, "
                                       f"t_tr {fmt(trace.get('t_tr_p50_median_link'), 0)}/{fmt(trace.get('t_tr_p95_max_link'), 0)}, "
                                       f"T_B {fmt(trace.get('T_B_p50_median_sim'), 0)} (max {fmt(trace.get('T_B_p50_max_sim'), 0)}), "
                                       f"fps {fmt((trace.get('fps_vs_untraced') or 0) * 100 if trace.get('fps_vs_untraced') is not None else None, 2, ' %')}")
                weights = ", ".join(f"{value:.4f}" for value in calibration["weights"] or [])
                lines.append(f"| {item['point']} | {weights or '-'} | {calibration['cuts'] or '-'} | {rounds} | "
                             f"{text_of(item['arm_difference'], 2, 100.0)} % | {trace_texts[0]} | {trace_texts[1]} |")
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
    arguments = parser.parse_args()
    plan = json.loads(pathlib.Path(arguments.plan).read_text(encoding="utf-8")) if arguments.plan else {}
    jobs = [load_job(pathlib.Path(directory)) for directory in arguments.directories]
    text, data = report(jobs, plan)
    if arguments.json:
        pathlib.Path(arguments.json).write_text(json.dumps(jsonable(data), indent=1), encoding="utf-8")
    if arguments.out:
        pathlib.Path(arguments.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.buffer.write((text + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
