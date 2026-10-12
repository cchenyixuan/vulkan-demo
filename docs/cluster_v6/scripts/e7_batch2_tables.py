"""E7 batch 2 (revision 2 of 2026-10-12, 2-D only) report tables, data only.

Reads job directories in the log-mirror layout (index.tsv, results.jsonl, traces/<point>_<arm>/, weights/) and
writes the tables the revision asks for, per point and arm:

  1. fps (with the defrag events, as the bench prints it), eta against the simultaneous K = 1 references of the
     same trial without the defrag events (mean, min and max over the trials; eta with the events next to it for
     runs without the event lines), eta_min, r_own p50 / p95 of the arm's traced run, the SM clock ratio K run /
     references, ns per particle of the K runs and of the K = 1 references (step time x K / particles).
  2. Same-load-normalised eta of the strong-scaling points: eta_same = fps_K / mean fps_1 of the K = 1 case whose
     particle count is nearest to the K run's per-card load (every passing K = 1 run of that case in the given
     jobs; the weak-scaling definition), and its ratio to the strong eta of table 1.
  3. ns per particle of every run (CSV next to --out, markdown appendix).
  4. Every defrag event (case, K, run, frame, wall / drain / idle ms) and a per-case digest K = 1 next to each K.
  5. Anatomy runs: per simulator the sum of the kernel durations and the phase sums of the sampled frames against
     the whole-step wall time (1000 / steady fps and the [loop] period).
  6. Traced runs (--trace DIR): per simulator medians of T_A, A->B, T_B, B->C, T_C, C->A, period and busy
     (e7_item0.slab_timings).
  7. --old DIR: the same (case, K) points of older jobs (rc1) next to the new ones (fps, eta, both with the events).

Usage:
  python docs/cluster_v6/scripts/e7_batch2_tables.py --job DIR [--job DIR ...] [--old DIR ...] [--trace DIR ...]
         [--solver rc2] [--tolerance 0.06] --out tables.md [--json tables.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import e7_full_summary as summary  # noqa: E402
import e7_item0 as item0  # noqa: E402

KERNELS = ("predict", "update_voxel", "ghost_send_trailing", "correction_density_interior", "force_deep_interior",
           "expand_lists", "install_trailing", "append_departed", "correction_density_boundary", "density_copy",
           "force")
GAPS = ("a_to_b_gap", "b_to_c_gap", "readback_trailing_sched_gap", "upload_trailing_to_c_gap")


def fmt(value, digits=1, suffix=""):
    if value is None or (isinstance(value, float) and value != value):
        return "-"
    return f"{value:.{digits}f}{suffix}"


def rng(stats: dict, digits=1, scale=1.0, suffix=""):
    """mean (min–max) of a describe() dict."""
    if not stats or not stats.get("n"):
        return "-"
    if stats["n"] == 1:
        return fmt(stats["mean"] * scale, digits, suffix)
    return f"{fmt(stats['mean'] * scale, digits)} ({fmt(stats['min'] * scale, digits)}–{fmt(stats['max'] * scale, digits)}{suffix})"


def desc(values: list) -> dict:
    values = [value for value in values if value is not None]
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": statistics.fmean(values), "min": min(values), "max": max(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0, "values": values}


def passing(row) -> bool:
    return bool(row) and row.get("status") in ("pass", "threshold_only")


def ns_per_particle(row) -> float | None:
    fps = summary.precise_fps(row)
    particles = (row or {}).get("loaded_particles")
    slabs = (row or {}).get("slab_count") or 1
    if not fps or not particles:
        return None
    return 1e9 * slabs / (fps * particles)


def ns_per_particle_nodefrag(row) -> float | None:
    fps = summary.nodefrag_fps(row)
    particles = (row or {}).get("loaded_particles")
    slabs = (row or {}).get("slab_count") or 1
    if not fps or not particles:
        return None
    return 1e9 * slabs / (fps * particles)


def window_of(row) -> str:
    if not row:
        return "-"
    return f"{row.get('total_steps')}/{(row.get('total_steps') or 0) - (row.get('steady_steps') or 0)}"


# ----------------------------------------------------------------------------- table 1: points

def point_rows(job: dict, points: dict, solver_filter: str | None) -> list:
    out = []
    for name, point in points.items():
        for arm in point["arms"]:
            arm_name, arm_solver = summary.split_arm(arm)
            if solver_filter and len(point["solvers"]) > 1 and arm_solver != solver_filter:
                continue
            trials = []
            for trial_number, trial in sorted(point["trials"].items()):
                run = trial.get(arm)
                if not run or not passing(run["row"]):
                    continue
                row = run["row"]
                fps = summary.precise_fps(row)
                fps_nodefrag = summary.nodefrag_fps(row)
                clock_k = summary.gpu_means(row, row.get("device_map") or [])
                entry = {"trial": trial_number, "label": run["label"], "fps": fps, "fps_nodefrag": fps_nodefrag,
                         "ns": ns_per_particle(row), "ns_nodefrag": ns_per_particle_nodefrag(row),
                         "sm_clock": clock_k["sm_clock"], "power": clock_k["power"], "node": row.get("node"),
                         "window": window_of(row), "particles": row.get("loaded_particles"),
                         "references": {}}
                for group_name, group in trial["references"].items():
                    if group.get("skipped"):
                        continue
                    members = [member["row"] for member in group.get("members", []) if passing(member["row"])]
                    if len(members) != len(group.get("members", [])) or not members:
                        continue
                    kind = summary.reference_type(group, point["case"])
                    factor = {"strong": point["slabs"], "weak": 1, "pairs": point["slabs"] / 2}[kind]
                    fps_1 = [summary.precise_fps(member) for member in members]
                    fps_1_nodefrag = [summary.nodefrag_fps(member) for member in members]
                    clocks_1 = [summary.gpu_means(member, member.get("device_map") or [])["sm_clock"] for member in members]
                    clocks_1 = [value for value in clocks_1 if value is not None]
                    reference = {"kind": kind, "case": group["case"], "members": len(members),
                                 "fps_1": fps_1, "ns_1": [ns_per_particle(member) for member in members],
                                 "ns_1_nodefrag": [ns_per_particle_nodefrag(member) for member in members],
                                 "window": window_of(members[0]),
                                 "eta": fps / (factor * statistics.fmean(fps_1)),
                                 "eta_min": fps / (factor * min(fps_1)),
                                 "sm_clock_1": statistics.fmean(clocks_1) if clocks_1 else None}
                    if fps_nodefrag and all(fps_1_nodefrag):
                        reference["eta_nodefrag"] = fps_nodefrag / (factor * statistics.fmean(fps_1_nodefrag))
                        reference["eta_min_nodefrag"] = fps_nodefrag / (factor * min(fps_1_nodefrag))
                    if reference["sm_clock_1"] and clock_k["sm_clock"]:
                        reference["clock_ratio"] = clock_k["sm_clock"] / reference["sm_clock_1"]
                    entry["references"][f"{kind}:{group['case']}"] = reference
                trials.append(entry)
            trace = point["traces"].get(arm)
            trace_metrics = {}
            if trace:
                directory = job["directory"] / "traces" / f"{name}_{arm}"
                try:
                    trace_metrics = summary.trace_metrics(directory)
                except Exception as error:  # noqa: BLE001
                    trace_metrics = {"error": f"{type(error).__name__}: {error}"}
                trace_metrics["label"] = trace["label"]
                trace_metrics["status"] = (trace["row"] or {}).get("status", "missing")
                trace_metrics["fps"] = summary.precise_fps(trace["row"]) if trace["row"] else None
            reference_keys = sorted({key for entry in trials for key in entry["references"]})
            references = {}
            for key in reference_keys:
                items = [entry["references"][key] for entry in trials if key in entry["references"]]
                references[key] = {
                    "kind": items[0]["kind"], "case": items[0]["case"], "members_per_trial": items[0]["members"],
                    "window": items[0]["window"], "trials": len(items),
                    "eta": desc([item["eta"] for item in items]),
                    "eta_min": desc([item["eta_min"] for item in items]),
                    "eta_nodefrag": desc([item.get("eta_nodefrag") for item in items]),
                    "eta_min_nodefrag": desc([item.get("eta_min_nodefrag") for item in items]),
                    "fps_1": desc([value for item in items for value in item["fps_1"]]),
                    "ns_1": desc([value for item in items for value in item["ns_1"]]),
                    "ns_1_nodefrag": desc([value for item in items for value in item["ns_1_nodefrag"]]),
                    "clock_ratio": desc([item.get("clock_ratio") for item in items]),
                    "sm_clock_1": desc([item.get("sm_clock_1") for item in items])}
            out.append({"job": job["name"], "point": name, "arm": arm, "arm_name": arm_name,
                        "solver": arm_solver or (point["solvers"][0] if len(point["solvers"]) == 1 else "")
                        or next((((trial.get(arm) or {}).get("row") or {}).get("solver_tag") or ""
                                 for _, trial in sorted(point["trials"].items()) if trial.get(arm)), "") or "untagged",
                        "family": point["family"], "case": point["case"], "K": point["slabs"], "line": point["line"],
                        "host": point["host"], "nodes": sorted({entry["node"] for entry in trials if entry["node"]}),
                        "trials": trials, "windows": sorted({entry["window"] for entry in trials}),
                        "particles": trials[0]["particles"] if trials else None,
                        "fps": desc([entry["fps"] for entry in trials]),
                        "fps_nodefrag": desc([entry["fps_nodefrag"] for entry in trials]),
                        "ns": desc([entry["ns"] for entry in trials]),
                        "ns_nodefrag": desc([entry["ns_nodefrag"] for entry in trials]),
                        "sm_clock": desc([entry["sm_clock"] for entry in trials]),
                        "references": references, "trace": trace_metrics})
    return out


def case_size(case: str) -> int:
    digits = "".join(character for character in case.split("_n", 1)[1] if character.isdigit()) if "_n" in case else "0"
    return int(digits or 0)


def sort_key(row: dict):
    return (row["case"].startswith("cavity3d"), "_k" in row["case"], case_size(row["case"]), row["K"], row["arm"], row["job"])


def table_points(rows: list) -> list:
    lines = ["| point | job | node | case | K | arm | trials | window | fps | η (no defrag) | η_min (no defrag) | η (with defrag) | η_min (with defrag) | r_own p50 / p95 | SM clock K / ref | ns/particle K | ns/particle K=1 ref |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in sorted(rows, key=sort_key):
        trace = row["trace"] or {}
        r_own = "-"
        if trace.get("r_own_p50") is not None:
            r_own = f"{trace['r_own_p50']:.3f} / {trace['r_own_p95']:.3f}"
        elif trace.get("error"):
            r_own = "trace: " + trace["error"][:40]
        if not row["references"]:
            lines.append(f"| {row['point']} | {row['job']} | {','.join(row['nodes'])} | {row['case']} | {row['K']} | {row['arm']} | "
                         f"{row['fps']['n']} | {','.join(row['windows'])} | {rng(row['fps'])} | - | - | - | - | {r_own} | - | "
                         f"{rng(row['ns'], 3)} | - |")
            continue
        for key, reference in row["references"].items():
            lines.append(
                f"| {row['point']} | {row['job']} | {','.join(row['nodes'])} | {row['case']} | {row['K']} | {row['arm']} | "
                f"{row['fps']['n']} | {','.join(row['windows'])} | {rng(row['fps'])} | "
                f"{rng(reference['eta_nodefrag'], 3)} | {rng(reference['eta_min_nodefrag'], 3)} | "
                f"{rng(reference['eta'], 3)} | {rng(reference['eta_min'], 3)} | {r_own} | "
                f"{rng(reference['clock_ratio'], 3)} | {rng(row['ns'], 3)} | {rng(reference['ns_1'], 3)} ({reference['kind']}: {reference['case']}, {reference['window']}) |")
    return lines


# ----------------------------------------------------------------------------- table 2: same-load eta

def k1_runs(jobs: list) -> dict:
    """case -> every passing K = 1 reference run (rows) of the jobs."""
    out: dict = {}
    for job in jobs:
        for entry in job["index"]:
            if entry["role"] != "reference":
                continue
            row = job["rows"].get(entry["label"])
            if passing(row) and (row.get("slab_count") or 1) == 1:
                out.setdefault(entry["reference_case"], []).append({"row": row, "job": job["name"], "label": entry["label"]})
    return out


def same_load_rows(rows: list, k1: dict, tolerance: float) -> list:
    sizes = {case: statistics.fmean(run["row"]["loaded_particles"] for run in runs) for case, runs in k1.items()}
    out = []
    for row in sorted(rows, key=sort_key):
        strong = next((reference for reference in row["references"].values() if reference["kind"] == "strong"), None)
        if strong is None or not row["particles"] or row["K"] == 1:
            continue
        per_card = row["particles"] / row["K"]
        best = min(sizes.items(), key=lambda item: abs(item[1] - per_card) / per_card, default=None)
        entry = {"point": row["point"], "job": row["job"], "arm": row["arm"], "case": row["case"], "K": row["K"],
                 "per_card": per_card, "eta_strong": strong["eta"], "eta_strong_nodefrag": strong["eta_nodefrag"]}
        if best is None or abs(best[1] - per_card) / per_card > tolerance:
            entry["same_case"] = None
            out.append(entry)
            continue
        case, size = best
        runs = k1[case]
        fps_1 = [summary.precise_fps(run["row"]) for run in runs]
        fps_1_nodefrag = [summary.nodefrag_fps(run["row"]) for run in runs]
        mean_1 = statistics.fmean(fps_1)
        entry.update({"same_case": case, "same_particles": size, "mismatch": (size - per_card) / per_card,
                      "same_runs": len(runs), "same_jobs": sorted({run["job"] for run in runs}),
                      "same_windows": sorted({window_of(run["row"]) for run in runs}),
                      "fps_1": desc(fps_1),
                      "eta_same": desc([entry_t["fps"] / mean_1 for entry_t in row["trials"]])})
        entry["ratio"] = entry["eta_same"]["mean"] / strong["eta"]["mean"] if strong["eta"].get("n") else None
        if all(fps_1_nodefrag) and strong["eta_nodefrag"].get("n"):
            mean_1_nodefrag = statistics.fmean(fps_1_nodefrag)
            entry["fps_1_nodefrag"] = desc(fps_1_nodefrag)
            entry["eta_same_nodefrag"] = desc([entry_t["fps_nodefrag"] / mean_1_nodefrag
                                               for entry_t in row["trials"] if entry_t["fps_nodefrag"]])
            entry["ratio_nodefrag"] = entry["eta_same_nodefrag"]["mean"] / strong["eta_nodefrag"]["mean"]
        out.append(entry)
    return out


def table_same_load(entries: list) -> list:
    lines = ["| point | job | arm | K | per-card particles | same-load K=1 case (particles, mismatch) | K=1 runs (jobs; windows) | K=1 fps | η_same (with defrag) | η_strong (with defrag) | ratio | η_same (no defrag) | η_strong (no defrag) | ratio (no defrag) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for entry in entries:
        if entry.get("same_case") is None:
            lines.append(f"| {entry['point']} | {entry['job']} | {entry['arm']} | {entry['K']} | {entry['per_card'] / 1e6:.2f} M | "
                         f"no K=1 case within tolerance | - | - | - | {rng(entry['eta_strong'], 3)} | - | - | "
                         f"{rng(entry['eta_strong_nodefrag'], 3)} | - |")
            continue
        lines.append(
            f"| {entry['point']} | {entry['job']} | {entry['arm']} | {entry['K']} | {entry['per_card'] / 1e6:.2f} M | "
            f"{entry['same_case']} ({entry['same_particles'] / 1e6:.2f} M, {entry['mismatch'] * 100:+.1f} %) | "
            f"{entry['same_runs']} ({', '.join(entry['same_jobs'])}; {', '.join(entry['same_windows'])}) | "
            f"{rng(entry['fps_1'])} | {rng(entry['eta_same'], 3)} | {rng(entry['eta_strong'], 3)} | {fmt(entry['ratio'], 3)} | "
            f"{rng(entry.get('eta_same_nodefrag'), 3)} | {rng(entry['eta_strong_nodefrag'], 3)} | {fmt(entry.get('ratio_nodefrag'), 3)} |")
    return lines


# ----------------------------------------------------------------------------- table 3: ns per particle per run

def run_rows(jobs: list) -> list:
    out = []
    for job in jobs:
        for position, entry in enumerate(job["index"]):
            row = job["rows"].get(entry["label"])
            if not row or entry["role"] in ("calibration", "reference_skipped", "cross_skipped"):
                continue
            fps = summary.precise_fps(row)
            out.append({"job": job["name"], "order": position, "label": entry["label"], "role": entry["role"],
                        "point": entry["point"], "case": row.get("case") or entry["case"], "K": row.get("slab_count"),
                        "trial": entry["trial"], "node": row.get("node"), "status": row.get("status"),
                        "particles": row.get("loaded_particles"), "steady_steps": row.get("steady_steps"),
                        "steady_seconds": summary.precise_steady_seconds(row), "fps": fps,
                        "step_ms": 1e3 / fps if fps else None, "ns_per_particle": ns_per_particle(row),
                        "fps_nodefrag": summary.nodefrag_fps(row), "ns_per_particle_nodefrag": ns_per_particle_nodefrag(row),
                        "defrag_in_window_ms": (row.get("defrag_events_summary") or {}).get("in_window_ms"),
                        "sm_clock": summary.gpu_means(row, row.get("device_map") or [])["sm_clock"]})
    return out


def table_runs(rows: list) -> list:
    lines = ["| job | run | role | K | particles | steady steps / s | fps | step ms | ns/particle | ns/particle (no defrag) | SM MHz |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['job']} | {row['label']} | {row['role']} | {row['K']} | {row['particles']} | "
                     f"{row['steady_steps']} / {fmt(row['steady_seconds'], 2)} | {fmt(row['fps'])} | {fmt(row['step_ms'], 3)} | "
                     f"{fmt(row['ns_per_particle'], 3)} | {fmt(row['ns_per_particle_nodefrag'], 3)} | {fmt(row['sm_clock'], 0)} |")
    return lines


# ----------------------------------------------------------------------------- table 4: defrag events

def defrag_rows(jobs: list) -> list:
    out = []
    for job in jobs:
        for entry in job["index"]:
            row = job["rows"].get(entry["label"])
            if not row or not row.get("defrag_events"):
                continue
            warmup = (row.get("total_steps") or 0) - (row.get("steady_steps") or 0)
            for event in row["defrag_events"]:
                out.append({"job": job["name"], "label": entry["label"], "role": entry["role"], "point": entry["point"],
                            "case": entry["case"] if entry["role"] != "reference" else entry["reference_case"],
                            "K": row.get("slab_count"), "frame": event["frame"], "wall_ms": event["wall_ms"],
                            "drain_ms": event["drain_ms"], "idle_ms": event["idle_ms"], "final": event["final"],
                            "in_window": event["frame"] >= warmup and not event["final"], "at_s": event["at_s"]})
    return out


def table_defrag_events(rows: list) -> list:
    if not rows:
        return ["(no defrag event lines in these jobs)"]
    lines = ["| job | run | case | K | frame | wall ms | drain ms | idle ms | in window |", "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['job']} | {row['label']} | {row['case']} | {row['K']} | {row['frame']} | {row['wall_ms']:.1f} | "
                     f"{row['drain_ms']:.1f} | {row['idle_ms']:.1f} | {'yes' if row['in_window'] else 'no'} |")
    return lines


def table_defrag_digest(rows: list) -> list:
    """Per case: wall ms per event (mean, min–max, events) for K = 1 and every K, in-window events only."""
    groups: dict = {}
    for row in rows:
        if row["in_window"]:
            groups.setdefault(row["case"], {}).setdefault(row["K"], []).append(row["wall_ms"])
    ks = sorted({k for per_case in groups.values() for k in per_case})
    if not ks:
        return ["(no defrag event lines in these jobs)"]
    lines = ["| case | " + " | ".join(f"K={k} wall ms (events)" for k in ks) + " |", "|---|" + "---|" * len(ks)]
    for case in sorted(groups, key=lambda name: (name.startswith("cavity3d"), "_k" in name, case_size(name))):
        cells = []
        for k in ks:
            values = groups[case].get(k)
            cells.append(f"{rng(desc(values))} ({len(values)})" if values else "-")
        lines.append(f"| {case} | " + " | ".join(cells) + " |")
    return lines


# ----------------------------------------------------------------------------- table 5: anatomy

def anatomy_rows(jobs: list) -> list:
    out = []
    for job in jobs:
        for entry in job["index"]:
            row = job["rows"].get(entry["label"])
            if not row or not row.get("anatomy_frames"):
                continue
            fps = summary.precise_fps(row)
            step_ms = 1e3 / fps if fps else None
            periods = [interval["period_ms_p50"] for interval in row.get("loop_intervals", []) if interval.get("period_ms_p50")]
            submits = [interval["submit_ms_mean"] for interval in row.get("loop_intervals", []) if interval.get("submit_ms_mean")]
            waits = [interval["wait_ms_mean"] for interval in row.get("loop_intervals", []) if interval.get("wait_ms_mean")]
            per_sim: dict = {}
            for frame in row["anatomy_frames"]:
                per_sim.setdefault(frame.get("sim", 0), []).append(frame)
            simulators = []
            for sim, frames in sorted(per_sim.items()):
                present = [key for key in KERNELS if any(key in frame for frame in frames)]
                kernel_sums = [sum(frame.get(key, 0.0) for key in present) for frame in frames]
                phase_sums = [sum(frame.get(key, 0.0) for key in ("phase_a", "phase_b", "phase_c")) for frame in frames]
                simulators.append({
                    "sim": sim, "frames": len(frames), "kernels": present,
                    "kernel_sum_ms": statistics.fmean(kernel_sums) / 1e3,
                    "kernel_sum_min_ms": min(kernel_sums) / 1e3, "kernel_sum_max_ms": max(kernel_sums) / 1e3,
                    "phase_sum_ms": statistics.fmean(phase_sums) / 1e3,
                    "phases_ms": {key: statistics.fmean(frame.get(key, 0.0) for frame in frames) / 1e3
                                  for key in ("phase_a", "phase_b", "phase_c")},
                    "gaps_ms": {key: statistics.fmean(frame.get(key, 0.0) for frame in frames) / 1e3
                                for key in GAPS if any(key in frame for frame in frames)},
                    "kernels_ms": {key: statistics.fmean(frame.get(key, 0.0) for frame in frames) / 1e3 for key in present},
                    "kernel_share_of_step": (statistics.fmean(kernel_sums) / 1e3 / step_ms) if step_ms else None})
            out.append({"job": job["name"], "label": entry["label"], "case": entry["case"], "K": row.get("slab_count"),
                        "status": row.get("status"), "steps": row.get("total_steps"), "fps": fps, "step_ms": step_ms,
                        "loop_period_ms_p50": statistics.fmean(periods) if periods else None,
                        "loop_submit_ms_mean": statistics.fmean(submits) if submits else None,
                        "loop_wait_ms_mean": statistics.fmean(waits) if waits else None,
                        "simulators": simulators,
                        "kernel_sum_max_ms": max(sim["kernel_sum_ms"] for sim in simulators),
                        "particles": row.get("loaded_particles"), "own_particles": row.get("own_particles")})
    return out


def table_anatomy(rows: list) -> list:
    lines = []
    for row in rows:
        lines += [f"**{row['label']}** ({row['job']}, {row['case']}, K = {row['K']}, status {row['status']}, {row['steps']} steps): "
                  f"steady fps {fmt(row['fps'])}, whole-step wall from fps {fmt(row['step_ms'], 3)} ms, [loop] period p50 "
                  f"{fmt(row['loop_period_ms_p50'], 3)} ms, [loop] submit mean {fmt(row['loop_submit_ms_mean'], 3)} ms, "
                  f"wait mean {fmt(row['loop_wait_ms_mean'], 3)} ms; max over simulators of the kernel sum {fmt(row['kernel_sum_max_ms'], 3)} ms.",
                  "",
                  "| sim | own particles | frames | kernel sum ms (min–max) | kernel sum / step wall | phase A | phase B | phase C | A+B+C | "
                  + " | ".join(key for key in GAPS) + " |",
                  "|---|---|---|---|---|---|---|---|---|" + "---|" * len(GAPS)]
        for sim in row["simulators"]:
            own = (row.get("own_particles") or [None] * (sim["sim"] + 1))[sim["sim"]] if row.get("own_particles") else None
            lines.append(f"| {sim['sim']} | {own} | {sim['frames']} | {sim['kernel_sum_ms']:.3f} ({sim['kernel_sum_min_ms']:.3f}–{sim['kernel_sum_max_ms']:.3f}) | "
                         f"{fmt(sim['kernel_share_of_step'], 3)} | {sim['phases_ms']['phase_a']:.3f} | {sim['phases_ms']['phase_b']:.3f} | "
                         f"{sim['phases_ms']['phase_c']:.3f} | {sim['phase_sum_ms']:.3f} | "
                         + " | ".join(fmt(sim["gaps_ms"].get(key), 3) for key in GAPS) + " |")
        lines.append("")
        kernels = row["simulators"][0]["kernels"]
        lines += ["| sim | " + " | ".join(kernels) + " |", "|---|" + "---|" * len(kernels)]
        for sim in row["simulators"]:
            lines.append(f"| {sim['sim']} | " + " | ".join(f"{sim['kernels_ms'].get(key, 0.0):.3f}" for key in kernels) + " |")
        lines.append("")
    return lines


# ----------------------------------------------------------------------------- table 6: traces

def trace_rows(traces: list) -> list:
    if str(item0._REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(item0._REPOSITORY_ROOT))
    from experiment.v7.analysis import step_trace_model
    out = []
    for trace in traces:
        run = step_trace_model.load_run(trace)
        out.append({"trace": str(trace), "case": pathlib.Path(run["meta"]["case"]).parent.name,
                    "warmup": run["meta"]["warmup"], "slabs": item0.slab_timings(run)})
    return out


def table_traces(rows: list) -> list:
    lines = []
    for row in rows:
        lines += [f"**{pathlib.Path(row['trace']).name}** ({row['case']}, warmup {row['warmup']}): medians over the steady steps, ms", "",
                  "| sim | steps | T_A | A→B | T_B | B→C | T_C | C→A | period | busy |", "|---|---|---|---|---|---|---|---|---|---|"]
        for slab in row["slabs"]:         # slab_timings gives µs
            lines.append(f"| {slab['sim']} | {slab['steps']} | " + " | ".join(
                f"{slab[key] / 1e3:.3f}" for key in ("T_A", "A_to_B", "T_B", "B_to_C", "T_C", "C_to_A", "period", "busy")) + " |")
        lines.append("")
    return lines


# ----------------------------------------------------------------------------- table 7: old vs new

def table_old_new(new_rows: list, old_rows: list) -> list:
    lines = ["| case | K | arm | old job | old fps | old η (with defrag) | new job | new fps | new η (with defrag) | new η (no defrag) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for new in sorted(new_rows, key=sort_key):
        matches = [old for old in old_rows if old["case"] == new["case"] and old["K"] == new["K"] and old["arm_name"] == new["arm_name"]]
        if not matches:
            continue
        new_reference = next(iter(new["references"].values()), None)
        for old in matches:
            old_reference = next(iter(old["references"].values()), None)
            lines.append(f"| {new['case']} | {new['K']} | {new['arm_name']} | {old['job']} ({old['solver']}) | {rng(old['fps'])} | "
                         f"{rng(old_reference['eta'], 3) if old_reference else '-'} | {new['job']} ({new['solver']}) | {rng(new['fps'])} | "
                         f"{rng(new_reference['eta'], 3) if new_reference else '-'} | "
                         f"{rng(new_reference['eta_nodefrag'], 3) if new_reference else '-'} |")
    return lines


# ----------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--job", action="append", default=[], required=True)
    parser.add_argument("--old", action="append", default=[], help="older jobs (rc1) for the old / new table")
    parser.add_argument("--trace", action="append", default=[])
    parser.add_argument("--solver", default=None, help="on points with several solver tags keep only this one's arms")
    parser.add_argument("--tolerance", type=float, default=0.06, help="same-load reference: max relative particle mismatch")
    parser.add_argument("--out", default=None)
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()

    jobs = [summary.load_job(pathlib.Path(directory)) for directory in arguments.job]
    olds = [summary.load_job(pathlib.Path(directory)) for directory in arguments.old]
    rows = [row for job in jobs for row in point_rows(job, summary.collect_points(job), arguments.solver)]
    old_rows = [row for job in olds for row in point_rows(job, summary.collect_points(job), None)]
    same = same_load_rows(rows, k1_runs(jobs), arguments.tolerance)
    runs = run_rows(jobs)
    defrag = defrag_rows(jobs)
    anatomy = anatomy_rows(jobs)
    traces = trace_rows([pathlib.Path(directory) for directory in arguments.trace])

    lines = ["# E7 batch 2 tables (data only)", "",
             "Jobs: " + ", ".join(job["name"] for job in jobs) + (("; old: " + ", ".join(job["name"] for job in olds)) if olds else ""), "",
             "η = fps_K / (K · mean fps of the K=1 references of the same trial) for strong points, fps_K / mean fps_1 for "
             "weak points (per-card case); η_min uses the slowest reference. 'no defrag' = both sides from the steady "
             "window minus the wall time of the in-window defrag events (harness b8e73dd and later; '-' = runs without "
             "the event lines). fps always includes the events. ns/particle = step time × K / loaded particles. "
             "r_own = own chain time / T_B on the worst link per step (p50 / p95) of the arm's traced run.", "",
             "## 1. Points", ""] + table_points(rows) + ["", "## 2. Same-load-normalised η (strong-scaling points)", "",
             "Reference = every passing K=1 run of the case whose particle count is nearest to the K run's per-card load "
             f"(tolerance {arguments.tolerance * 100:.0f} %); η_same = fps_K / mean fps_1 (the weak-scaling definition, per "
             "trial); ratio = η_same / η_strong (means).", ""] \
        + table_same_load(same) + ["", "## 4. Defrag events (in-window) per case, wall ms per event", ""] + table_defrag_digest(defrag) \
        + ["", "## 5. Anatomy: kernel sums against the whole-step wall time (ms, mean over the sampled frames)", ""] \
        + (table_anatomy(anatomy) or ["(no anatomy runs)"]) + ["", "## 6. Traced runs: per-simulator phase medians", ""] \
        + (table_traces(traces) or ["(no --trace)"]) + ["", "## 7. Old (rc1) next to new (rc2) on the same case and K", ""] \
        + (table_old_new(rows, old_rows) if olds else ["(no --old)"]) \
        + ["", "## Appendix A. Every defrag event", ""] + table_defrag_events(defrag) \
        + ["", "## Appendix B. ns per particle of every run", ""] + table_runs(runs)
    text = "\n".join(lines) + "\n"
    if arguments.out:
        out = pathlib.Path(arguments.out)
        out.write_text(text, encoding="utf-8")
        with out.with_suffix(".runs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(runs[0].keys()) if runs else ["job"])
            writer.writeheader()
            writer.writerows(runs)
        with out.with_suffix(".defrag.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(defrag[0].keys()) if defrag else ["job"])
            writer.writeheader()
            writer.writerows(defrag)
    else:
        sys.stdout.write(text)
    if arguments.json:
        pathlib.Path(arguments.json).write_text(json.dumps(summary.jsonable(
            {"points": rows, "same_load": same, "runs": runs, "defrag_events": defrag, "anatomy": anatomy, "traces": traces,
             "old_points": old_rows}), indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
