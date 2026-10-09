"""
parse_run_v6.py — one E30 run log (run_chain_v6.py + the v6 chain bench) -> one
JSON row: completion, fps, conservation, every overflow counter, far migration,
stamp errors, alive at start and end, init clamp count, NaN / field check, seam
checks, host-setting evidence (transfer queues, worker pins, switch interval),
partition columns, pool capacities and departed peaks, per-link copy times and
bytes, anatomy lines, errors, wall time and (with --telemetry) the per-GPU VRAM
peak inside the run window.

Classification (the E30 stop rule):
  pass                rc 0 and the final line present
  threshold_only      rc 1, the final line present, every invariant 0, seam
                      overshoot / duplicate checks OK, no NaN, only the field
                      band (rho 1000 +-5 %, vmax <= 1.001) failed -> recorded,
                      not a hard failure
  hard_failure        anything else (drift, overflow, far migration, stamp
                      errors, seam FAIL, NaN, crash, timeout, missing final line,
                      a log of another solver than --solver)

Solver (E39): the runner tags carry the solver ([chain_v6] / [chain_v7],
[partition_*], [case_loader_*], [SimV6] / [SimV7]). --solver names the one the
run was started with (e30_lib.sh passes it for E30_SOLVER=v7); without it the
solver is read from the log (run_chain_v6.py's "[e30] solver=" line or the
first runner tag; v6 when there is none). Every row records "solver" and
"solver_in_log", and v7 rows also the bench's own switch line
("[chain_v7] v7 switches: ...") as "solver_switches".

E7: run_chain_v6.py's build stages with host memory ("stages", "build_seconds"
= time to the first timed frame), each simulator's raw pool-health watermarks
("simulators[i].pool_health"), the ghost-pool region peaks of V*_POOL_PEAKS=1
runs ("pool_regions") and the counts of Khronos validation messages
("validation_messages", "validation_unavailable") are recorded too; none of
them enters the classification.

E7 full campaign (recorded, not part of the verdict either): the start barrier
("barrier": parties, arrival / release epochs, wait, status), the stage epochs,
the --defrag-log lines ("defrag_reports" per frame and slab, "defrag_times"),
the full anatomy durations paired with the bench's [anatomy] lines
("anatomy_frames": frame, slab, every duration, install_sum), the host loop
lines of V7_LOOP_TRACE=1 anatomy runs ("loop_intervals"), the pool demand per
1000 frames ("pool_series"), and with --telemetry the loop window's clocks per
GPU ("telemetry_loop": SM clock median / min, power median, temperature max,
utilization median; window = the stage epochs loop_start .. loop_end). The row
is appended to --results with a single write (concurrent reference processes
share the file).

Usage:
    python parse_run_v6.py --log RUN.log --label LABEL --rc RC --start EPOCH --end EPOCH
                           [--telemetry telemetry.csv] [--results results.jsonl] [--solver v7]
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import pathlib
import re
import sys

NUMBER = r"[-+]?\d[\d,]*"
SOLVERS = ("v6", "v7")
# the first solver-tagged line of a run log: run_chain_v6.py's "[e30] solver=v7" (non-default solvers),
# else the first runner tag ([chain_v6] switchinterval_s=..., or a loader / partition line)
SOLVER_PATTERN = re.compile(r"\[(?:e30\] solver=|chain_|case_loader_|partition_)(v\d+)\b")


def to_int(text: str) -> int:
    return int(text.replace(",", ""))


def log_solver(text: str):
    """The solver a run log was written by (None when it carries no solver tag)."""
    match = SOLVER_PATTERN.search(text)
    return match.group(1) if match else None


def parse_log(text: str, solver: str = "v6") -> dict:
    row: dict = {"solver": solver}
    lines = text.splitlines()
    chain = rf"\[chain_{solver}\]"
    partition = rf"\[partition_{solver}\]"
    case_loader = rf"\[case_loader_{solver}\]"
    simulator = rf"\[Sim{solver.upper()}\]"

    match = re.search(r"\[e30\] switchinterval_s=([\d.e-]+)", text)
    row["switch_interval_s"] = float(match.group(1)) if match else None
    row["config_lines"] = [line for line in lines if line.startswith("[e30-config]")]
    match = re.search(case_loader + r" total loaded: (" + NUMBER + r") particles", text)
    row["loaded_particles"] = to_int(match.group(1)) if match else None
    match = re.search(partition + r" chain N=(\d+): columns (.*?) of (\d+); particles (.*)", text)
    if match:
        row["slab_count"] = int(match.group(1))
        row["grid_columns"] = int(match.group(3))
        row["own_columns"] = [int(end) - int(start) for start, end in
                              re.findall(r"\[(\d+),(\d+)\)", match.group(2))]
        row["own_particles"] = [to_int(value) for value in re.findall(NUMBER, match.group(4))]
    match = re.search(chain + r" K=(\d+) weights=\[([^\]]*)\] device_map=\[([^\]]*)\]", text)
    if match:
        row["weights"] = [float(value) for value in match.group(2).split(",") if value.strip()]
        row["device_map"] = [int(value) for value in match.group(3).split(",") if value.strip()]
    match = re.search(chain + r" weights source=(\S+) cuts=\[([^\]]*)\]", text)   # E32 runner line
    if match:
        row["weights_source"] = match.group(1)
        row["cuts"] = [int(value) for value in match.group(2).split(",") if value.strip()]
    row["obj_cache_hits"] = len(re.findall(r"\[e30\] obj cache hit", text))
    match = re.search(partition + r" seam: (.*)", text)
    row["seam_layout"] = match.group(1).strip() if match else None
    row["partition_warnings"] = len(re.findall(partition + r" WARN", text))
    row["replica_region_warning"] = "WARNING: replica region" in text
    row["transfer_queues_split"] = len(re.findall(r"transfer queues: 2 \(readback \+ upload split\)", text))
    row["transfer_queues_shared"] = len(re.findall(r"transfer queues: 1", text))
    # cpu list = digits, commas, dashes only: two worker threads can print onto one line
    # ("... cpus 0-31[worker s1_to_s0] pinned ...", A100 job 1550154 k4_3d8m)
    row["worker_pins"] = re.findall(r"\[worker (\S+)\] pinned to dest device (\d+) cpus ([0-9,\-]+)", text)
    row["worker_pin_skipped"] = len(re.findall(r"affinity pin skipped", text))
    row["device_local_mb"] = [float(value) for value in
                              re.findall(simulator + r" device-local buffers: \d+, ([\d.]+) MB", text)]
    row["defrag_scratch_mb"] = [float(value) for value in
                                re.findall(simulator + r" defrag scratch buffers: \d+, ([\d.]+) MB", text)]
    bootstrap_alive = [to_int(value) for value in re.findall(simulator + r" bootstrap done: alive=(" + NUMBER + ")", text)]
    row["alive_start"] = sum(bootstrap_alive) if bootstrap_alive else None

    match = re.search(chain + r" TOTAL: (\d+) steps in ([\d.]+)s = ([\d.]+) fps", text)
    row["total_steps"], row["loop_seconds"], row["total_fps"] = (
        (int(match.group(1)), float(match.group(2)), float(match.group(3))) if match else (None, None, None))
    match = re.search(chain + r" STEADY \(post-warmup (\d+)\): (\d+) steps in ([\d.]+)s = ([\d.]+) fps", text)
    row["steady_fps"] = float(match.group(4)) if match else None
    row["steady_steps"] = int(match.group(2)) if match else None
    row["steady_seconds"] = float(match.group(3)) if match else None

    simulators = []
    for match in re.finditer(chain + r" sim(\d+) seam: ghost_layers=(\d+) departed peak/frame=(\d+) "
                             r"capacity=(\d+) far_migration=(\d+)(.*)", text):
        counters = {name: int(value) for name, value in re.findall(r"(overflow_\w+)=(\d+)", match.group(6))}
        simulators.append({"sim": int(match.group(1)), "ghost_layers": int(match.group(2)),
                           "departed_peak": int(match.group(3)), "departed_capacity": int(match.group(4)),
                           "far_migration": int(match.group(5)), "overflow": counters})
    for match in re.finditer(chain + r" sim(\d+) \(dev(\d+)\): alive=(" + NUMBER + r") pool_used=([\d.]+)% "
                             r"peak_migration=(\d+) drops=(\d+) stamp_err=(\d+)", text):
        index = int(match.group(1))
        target = next((entry for entry in simulators if entry["sim"] == index), None)
        if target is None:
            target = {"sim": index}
            simulators.append(target)
        target.update({"device": int(match.group(2)), "alive": to_int(match.group(3)),
                       "pool_used_percent": float(match.group(4)), "peak_migration": int(match.group(5)),
                       "drops": int(match.group(6)), "stamp_errors": int(match.group(7))})
    for match in re.finditer(r"\[e30\] sim(\d+) \(dev(-?\d+)\): initialization_seam_clamp_count=(\S+) "
                             r"overflow_initialization_outside=(\S+)", text):
        index = int(match.group(1))
        target = next((entry for entry in simulators if entry["sim"] == index), None)
        if target is None:
            target = {"sim": index}
            simulators.append(target)
        target["init_clamp_count"] = None if match.group(3) == "None" else int(match.group(3))
    for match in re.finditer(r"\[e30\] sim(\d+) \(dev(-?\d+)\) pool_health: (.*)", text):     # E7 (E15)
        index = int(match.group(1))
        target = next((entry for entry in simulators if entry["sim"] == index), None)
        if target is None:
            target = {"sim": index}
            simulators.append(target)
        target["pool_health"] = {name: (None if value == "None" else int(value))
                                 for name, value in re.findall(r"(\w+)=(-?\d+|None)", match.group(3))}
    row["simulators"] = sorted(simulators, key=lambda entry: entry["sim"])
    clamp_counts = [entry.get("init_clamp_count") for entry in row["simulators"]]
    row["init_clamp_total"] = (sum(value for value in clamp_counts if value is not None)
                               if any(value is not None for value in clamp_counts) else None)
    overflow_sums: dict = {}
    for entry in row["simulators"]:
        for name, value in entry.get("overflow", {}).items():
            overflow_sums[name] = overflow_sums.get(name, 0) + value
    row["overflow_by_counter"] = overflow_sums

    match = re.search(chain + r" final: total=(" + NUMBER + r") \(expected (" + NUMBER + r")\) drift=(-?\d+) "
                      r"stamp_errors gpu=(\d+) host=(\d+) overflow_total=(\d+) far_migration_total=(\d+)", text)
    row["final_line"] = bool(match)
    if match:
        row.update({"alive_end": to_int(match.group(1)), "alive_expected": to_int(match.group(2)),
                    "drift": int(match.group(3)), "stamp_errors_gpu": int(match.group(4)),
                    "stamp_errors_host": int(match.group(5)), "overflow_total": int(match.group(6)),
                    "far_migration_total": int(match.group(7))})
    seam_lines = re.findall(chain + r" seam (\d+) \(col (\d+)\): L_overshoot=(\S+)dx R_overshoot=(\S+)dx "
                            r"dup=(\d+) (OK|\*\*\* FAIL \*\*\*)", text)
    row["seam_checks"] = [{"seam": int(seam), "column": int(column), "left_overshoot_dx": float(left),
                           "right_overshoot_dx": float(right), "duplicates": int(duplicates),
                           "ok": verdict == "OK"} for seam, column, left, right, duplicates, verdict in seam_lines]
    match = re.search(chain + r" fields: rho\[(\S+),(\S+)\] vmax=(\S+) (OK|\*\*\* FAIL \*\*\*)", text)
    if match:
        values = [match.group(1), match.group(2), match.group(3)]
        row["fields"] = {"rho_min": values[0], "rho_max": values[1], "vmax": values[2],
                         "ok": match.group(4) == "OK"}
        row["nan_seen"] = any(value.lower() in ("nan", "-nan", "inf", "-inf") for value in values)
    else:
        row["fields"] = None
        row["nan_seen"] = None                       # no field check ran (e.g. --no-seam-check)
    row["validation_failed_line"] = "*** VALIDATION FAILED ***" in text
    if solver != "v6":                   # the v7 bench prints its own switch registry (E39)
        match = re.search(chain + rf" {solver} switches: (.*)", text)
        row["solver_switches"] = dict(re.findall(r"(\S+?)=(\S+)", match.group(1))) if match else None

    worker_copy = {}
    for match in re.finditer(r"\[worker (\S+)\] us p50/p90/max: (.*)", text):
        segments = dict(re.findall(r"(\w+)=([\d/]+)", match.group(2)))
        worker_copy[match.group(1)] = segments
    row["worker_segments_us"] = worker_copy
    link_bytes = {}
    for match in re.finditer(r"\[link (\S+)\] bytes/frame: host_copy=([\d.]+) KiB dma=([\d.]+) KiB over (\d+) frames",
                             text):
        link_bytes[match.group(1)] = {"host_copy_kib": float(match.group(2)), "dma_kib": float(match.group(3)),
                                      "frames": int(match.group(4))}
    row["link_bytes"] = link_bytes
    row["anatomy_lines"] = [line for line in lines if line.startswith("[anatomy]")]
    row["migration_drop_lines"] = [line for line in lines if line.startswith("[migration]")]

    error_patterns = ("Traceback (most recent call last)", "OutOfDeviceMemory", "VkError", "STALL AUTOPSY",
                      "DIED", "STALE READBACK", "Segmentation fault", "out of range", "MemoryError")
    row["errors"] = sorted({pattern for pattern in error_patterns if pattern in text})

    # E7 lines of run_chain_v6.py (recorded, not part of the verdict): the build stages with this process's host
    # memory (solvers other than v6; the last line of a stage wins, i.e. the timed chain's after any pilots),
    # the ghost-pool region peaks with V*_POOL_PEAKS=1 (E15), and the Khronos validation layer's messages
    # (VulkanContext messenger on stderr: "[Vulkan <ESC>[91mERROR<ESC>[0m] ...", WARNING alike).
    stages = {}
    for match in re.finditer(r"\[e30\] stage (\w+): t=([\d.]+)s VmRSS=(\S+) VmHWM=(\S+)(?: epoch=([\d.]+))?", text):
        stages[match.group(1)] = {"seconds": float(match.group(2)),
                                  "vm_rss": match.group(3), "vm_hwm": match.group(4),
                                  "epoch": float(match.group(5)) if match.group(5) else None}
    row["stages"] = stages
    row["build_seconds"] = stages.get("loop_start", {}).get("seconds")
    row["pool_regions"] = [
        {"link": link, "region": region, "capacity": None if capacity == "None" else int(capacity),
         "peak": int(peak), "frame_of_peak": int(frame), "p999": int(p999), "mean": float(mean),
         "last": int(last), "frames": int(frames), "occupancy": occupancy}
        for link, region, capacity, peak, frame, p999, mean, last, frames, occupancy in re.findall(
            r"\[e30\] pool (\S+) (\w+): capacity=(\d+|None) peak=(\d+) frame_of_peak=(\d+) p999=(\d+) "
            r"mean=([\d.]+) last=(\d+) frames=(\d+) occupancy=(\S+)", text)]
    row["validation_messages"] = {
        "error": len(re.findall(r"\[Vulkan (?:\x1b\[\d+m)?ERROR", text)),
        "warning": len(re.findall(r"\[Vulkan (?:\x1b\[\d+m)?WARNING", text))}
    row["validation_unavailable"] = "validation layer requested but not available" in text

    # E7 full campaign lines (recorded only)
    match = re.search(r"\[e30\] barrier dir=(\S+) parties=(\d+) arrived_epoch=([\d.]+) released_epoch=([\d.]+) "
                      r"waited=([\d.]+)s seen=(\d+) status=(\w+)", text)
    row["barrier"] = ({"directory": match.group(1), "parties": int(match.group(2)),
                       "arrived_epoch": float(match.group(3)), "released_epoch": float(match.group(4)),
                       "waited_seconds": float(match.group(5)), "seen": int(match.group(6)),
                       "status": match.group(7)} if match else None)
    row["defrag_reports"] = [
        {"frame": int(frame), "sim": int(sim), **{name: number_or_text(value)
                                                  for name, value in re.findall(r"(\w+)=(\S+)", fields)}}
        for frame, sim, fields in re.findall(r"\[e30\] defrag f(\d+) sim(\d+): (.*)", text)]
    row["defrag_times"] = [
        {"sim": int(sim), "wall_ms": float(wall), "gpu_us": float(gpu) if gpu else None}
        for sim, wall, gpu in re.findall(r"\[e30\] defrag_time sim(\d+): wall_ms=([\d.]+)(?: gpu_us=([\d.]+))?",
                                         text)]
    anatomy_full = [dict((name, number_or_text(value)) for name, value in re.findall(r"(\w+)=(\S+)", fields))
                    for fields in re.findall(r"\[e30\] anatomy_all call=\d+: (.*)", text)]
    anatomy_heads = re.findall(r"^\[anatomy\] f(\d+) s(\d+):", text, flags=re.MULTILINE)
    row["anatomy_frames"] = [{"frame": int(frame), "sim": int(sim), **durations}
                             for (frame, sim), durations in zip(anatomy_heads, anatomy_full)]
    row["anatomy_unpaired"] = len(anatomy_full) - len(anatomy_heads)
    row["loop_intervals"] = [
        {"frame": int(frame), **{name: number_or_text(value) for name, value in re.findall(r"(\w+)=(\S+)", fields)}}
        for frame, fields in re.findall(r"^\[loop\] f(\d+): (.*)", text, flags=re.MULTILINE)]
    row["pool_series"] = [
        {"link": link, "region": region, "capacity": None if capacity == "None" else int(capacity),
         "window": int(window), "peaks": [int(value) for value in peaks.split(",") if value]}
        for link, region, capacity, window, peaks in re.findall(
            r"\[e30\] pool_series (\S+) (\w+): capacity=(\d+|None) window=(\d+) peaks=([\d,]*)", text)]
    return row


def number_or_text(value: str):
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() and "." not in value and "e" not in value.lower() else number


def classify(row: dict, return_code: int) -> tuple[str, list[str]]:
    reasons = []
    if row.get("solver_in_log") and row["solver_in_log"] != row["solver"]:
        reasons.append(f"solver mismatch: run as {row['solver']}, log of {row['solver_in_log']}")
    if return_code in (124, 137):
        reasons.append(f"timeout/killed rc={return_code}")
    if return_code not in (0, 1, 124, 137):
        reasons.append(f"rc={return_code}")
    if row["errors"]:
        reasons.append("errors: " + ",".join(row["errors"]))
    if not row["final_line"]:
        reasons.append("no final line")
    else:
        if row["drift"] != 0:
            reasons.append(f"drift={row['drift']}")
        if row["overflow_total"] != 0:
            reasons.append(f"overflow_total={row['overflow_total']}")
        if row["far_migration_total"] != 0:
            reasons.append(f"far_migration_total={row['far_migration_total']}")
        if row["stamp_errors_gpu"] or row["stamp_errors_host"]:
            reasons.append(f"stamp_errors gpu={row['stamp_errors_gpu']} host={row['stamp_errors_host']}")
    if any(not seam["ok"] for seam in row["seam_checks"]):
        reasons.append("seam check FAIL")
    if row["nan_seen"]:
        reasons.append("NaN/inf in fields")
    if reasons:
        return "hard_failure", reasons
    if return_code == 1:
        if row["fields"] is not None and not row["fields"]["ok"]:
            return "threshold_only", ["field band only (rho 1000+-5% / vmax<=1.001)"]
        return "hard_failure", ["rc=1 without an identified cause"]
    return "pass", []


def telemetry_peaks(telemetry_path: str, start_epoch: float, end_epoch: float) -> dict:
    """Per-GPU peak memory.used [MiB] (and power, util) between start and end (local time stamps)."""
    peaks: dict = {}
    path = pathlib.Path(telemetry_path)
    if not path.exists():
        return peaks
    with path.open(encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        header = None
        for record in reader:
            if not record:
                continue
            if header is None or record[0].strip().startswith("timestamp"):
                header = [name.strip() for name in record]
                continue
            values = dict(zip(header, (value.strip() for value in record)))
            try:
                stamp = datetime.datetime.strptime(values["timestamp"], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            except (KeyError, ValueError):
                continue
            if not (start_epoch <= stamp <= end_epoch):
                continue
            index = values.get("index")
            memory_key = next((key for key in values if key.startswith("memory.used")), None)
            utilization_key = next((key for key in values if key.startswith("utilization.gpu")), None)
            if index is None or memory_key is None:
                continue
            try:
                memory_used = float(values[memory_key].split()[0])
            except ValueError:
                continue
            entry = peaks.setdefault(index, {"memory_used_peak_mib": 0.0, "utilization_peak": 0.0, "samples": 0})
            entry["memory_used_peak_mib"] = max(entry["memory_used_peak_mib"], memory_used)
            entry["samples"] += 1
            if utilization_key:
                try:
                    entry["utilization_peak"] = max(entry["utilization_peak"],
                                                    float(values[utilization_key].split()[0]))
                except ValueError:
                    pass
    return peaks


def telemetry_window(telemetry_path: str, start_epoch: float, end_epoch: float) -> dict:
    """Per-GPU SM clock median / min [MHz], power median [W], temperature max [C] and utilization median [%]
    between start and end (local time stamps): the clocks covariate of the efficiency rule."""
    samples: dict = {}
    path = pathlib.Path(telemetry_path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        header = None
        for record in csv.reader(handle):
            if not record:
                continue
            if header is None or record[0].strip().startswith("timestamp"):
                header = [name.strip() for name in record]
                continue
            values = dict(zip(header, (value.strip() for value in record)))
            try:
                stamp = datetime.datetime.strptime(values["timestamp"], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            except (KeyError, ValueError):
                continue
            if not (start_epoch <= stamp <= end_epoch) or values.get("index") is None:
                continue
            entry = samples.setdefault(values["index"], {"clock": [], "power": [], "temperature": [], "utilization": []})
            for key_prefix, name in (("clocks.current.sm", "clock"), ("power.draw", "power"),
                                     ("temperature.gpu", "temperature"), ("utilization.gpu", "utilization")):
                key = next((key for key in values if key.startswith(key_prefix)), None)
                if key is None:
                    continue
                try:
                    entry[name].append(float(values[key].split()[0]))
                except (ValueError, IndexError):
                    pass

    def median(values):
        ordered = sorted(values)
        return ordered[len(ordered) // 2] if ordered else None

    return {index: {"sm_clock_median": median(entry["clock"]), "sm_clock_min": min(entry["clock"], default=None),
                    "power_median": median(entry["power"]), "temperature_max": max(entry["temperature"], default=None),
                    "utilization_median": median(entry["utilization"]), "samples": len(entry["clock"])}
            for index, entry in samples.items()}


def append_row(results_path: str, row: dict) -> None:
    """One os.write of the whole line on an O_APPEND descriptor: rows of processes that finish together (a
    reference set) never interleave, whatever their length."""
    import os
    data = (json.dumps(row) + "\n").encode("utf-8")
    descriptor = os.open(results_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        written = 0
        while written < len(data):
            written += os.write(descriptor, data[written:])
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description="E30 run-log parser")
    parser.add_argument("--log", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--rc", type=int, required=True)
    parser.add_argument("--start", type=float, default=None, help="epoch seconds, run start")
    parser.add_argument("--end", type=float, default=None, help="epoch seconds, run end")
    parser.add_argument("--telemetry", default=None)
    parser.add_argument("--results", default=None, help="append the JSON row here")
    parser.add_argument("--node", default=None)
    parser.add_argument("--solver", choices=SOLVERS, default=None,
                        help="the solver the run was started with (run_chain_v6.py --solver); default: read from "
                             "the log (v6 when it names none). A log of another solver is a hard failure")
    arguments = parser.parse_args()

    text = pathlib.Path(arguments.log).read_text(encoding="utf-8", errors="replace")
    row = {"label": arguments.label, "rc": arguments.rc, "node": arguments.node}
    solver_in_log = log_solver(text)
    row.update(parse_log(text, arguments.solver or solver_in_log or "v6"))
    row["solver_in_log"] = solver_in_log
    if arguments.start is not None and arguments.end is not None:
        row["wall_seconds"] = round(arguments.end - arguments.start, 1)
        if arguments.telemetry:
            row["telemetry_peaks"] = telemetry_peaks(arguments.telemetry, arguments.start, arguments.end)
            loop_start = (row.get("stages") or {}).get("loop_start", {}).get("epoch")
            loop_end = (row.get("stages") or {}).get("loop_end", {}).get("epoch")
            if loop_start and loop_end and loop_end > loop_start:
                row["telemetry_loop"] = telemetry_window(arguments.telemetry, loop_start, loop_end)
    row["status"], row["reasons"] = classify(row, arguments.rc)
    if arguments.results:
        append_row(arguments.results, row)
    vram = ""
    if row.get("telemetry_peaks"):
        vram = " vram_peak_mib=" + "/".join(f"{index}:{entry['memory_used_peak_mib']:.0f}"
                                            for index, entry in sorted(row["telemetry_peaks"].items()))
    steady = row["steady_fps"] if row["steady_fps"] is not None else "-"
    print(f"RESULT label={row['label']} status={row['status']} rc={row['rc']} steady_fps={steady} "
          f"total_fps={row['total_fps']} drift={row.get('drift')} overflow_total={row.get('overflow_total')} "
          f"far_migration={row.get('far_migration_total')} stamps={row.get('stamp_errors_gpu')}+"
          f"{row.get('stamp_errors_host')} alive={row.get('alive_start')}->{row.get('alive_end')} "
          f"clamp={row.get('init_clamp_total')} nan={row.get('nan_seen')} "
          f"fields={'-' if row['fields'] is None else ('OK' if row['fields']['ok'] else 'FAIL')} "
          f"columns={row.get('own_columns')} tq2={row['transfer_queues_split']} "
          f"pins={len(row['worker_pins'])}/skip{row['worker_pin_skipped']} "
          f"wall={row.get('wall_seconds')}s{vram}"
          + (f" solver={row['solver']}" if row["solver"] != "v6" else "")
          + (f" build={row['build_seconds']}s cache_hits={row['obj_cache_hits']}"
             if row.get("build_seconds") is not None else "")
          + (f" vk_messages={row['validation_messages']['error']}E/{row['validation_messages']['warning']}W"
             if any(row["validation_messages"].values()) else "")
          + (" vk_layer=unavailable" if row["validation_unavailable"] else "")
          + (f" barrier={row['barrier']['status']}/{row['barrier']['waited_seconds']:.1f}s"
             if row.get("barrier") else "")
          + (f" reasons={';'.join(row['reasons'])}" if row["reasons"] else ""), flush=True)
    return 0 if row["status"] in ("pass", "threshold_only") else 3


if __name__ == "__main__":
    sys.exit(main())
