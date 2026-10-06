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
                      errors, seam FAIL, NaN, crash, timeout, missing final line)

Usage:
    python parse_run_v6.py --log RUN.log --label LABEL --rc RC --start EPOCH --end EPOCH
                           [--telemetry telemetry.csv] [--results results.jsonl]
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


def to_int(text: str) -> int:
    return int(text.replace(",", ""))


def parse_log(text: str) -> dict:
    row: dict = {}
    lines = text.splitlines()

    match = re.search(r"\[e30\] switchinterval_s=([\d.e-]+)", text)
    row["switch_interval_s"] = float(match.group(1)) if match else None
    row["config_lines"] = [line for line in lines if line.startswith("[e30-config]")]
    match = re.search(r"\[case_loader_v6\] total loaded: (" + NUMBER + r") particles", text)
    row["loaded_particles"] = to_int(match.group(1)) if match else None
    match = re.search(r"\[partition_v6\] chain N=(\d+): columns (.*?) of (\d+); particles (.*)", text)
    if match:
        row["slab_count"] = int(match.group(1))
        row["grid_columns"] = int(match.group(3))
        row["own_columns"] = [int(end) - int(start) for start, end in
                              re.findall(r"\[(\d+),(\d+)\)", match.group(2))]
        row["own_particles"] = [to_int(value) for value in re.findall(NUMBER, match.group(4))]
    match = re.search(r"\[chain_v6\] K=(\d+) weights=\[([^\]]*)\] device_map=\[([^\]]*)\]", text)
    if match:
        row["weights"] = [float(value) for value in match.group(2).split(",") if value.strip()]
        row["device_map"] = [int(value) for value in match.group(3).split(",") if value.strip()]
    match = re.search(r"\[chain_v6\] weights source=(\S+) cuts=\[([^\]]*)\]", text)   # E32 runner line
    if match:
        row["weights_source"] = match.group(1)
        row["cuts"] = [int(value) for value in match.group(2).split(",") if value.strip()]
    row["obj_cache_hits"] = len(re.findall(r"\[e30\] obj cache hit", text))
    match = re.search(r"\[partition_v6\] seam: (.*)", text)
    row["seam_layout"] = match.group(1).strip() if match else None
    row["partition_warnings"] = len(re.findall(r"\[partition_v6\] WARN", text))
    row["replica_region_warning"] = "WARNING: replica region" in text
    row["transfer_queues_split"] = len(re.findall(r"transfer queues: 2 \(readback \+ upload split\)", text))
    row["transfer_queues_shared"] = len(re.findall(r"transfer queues: 1", text))
    # cpu list = digits, commas, dashes only: two worker threads can print onto one line
    # ("... cpus 0-31[worker s1_to_s0] pinned ...", A100 job 1550154 k4_3d8m)
    row["worker_pins"] = re.findall(r"\[worker (\S+)\] pinned to dest device (\d+) cpus ([0-9,\-]+)", text)
    row["worker_pin_skipped"] = len(re.findall(r"affinity pin skipped", text))
    row["device_local_mb"] = [float(value) for value in
                              re.findall(r"\[SimV6\] device-local buffers: \d+, ([\d.]+) MB", text)]
    row["defrag_scratch_mb"] = [float(value) for value in
                                re.findall(r"\[SimV6\] defrag scratch buffers: \d+, ([\d.]+) MB", text)]
    bootstrap_alive = [to_int(value) for value in re.findall(r"\[SimV6\] bootstrap done: alive=(" + NUMBER + ")", text)]
    row["alive_start"] = sum(bootstrap_alive) if bootstrap_alive else None

    match = re.search(r"\[chain_v6\] TOTAL: (\d+) steps in ([\d.]+)s = ([\d.]+) fps", text)
    row["total_steps"], row["loop_seconds"], row["total_fps"] = (
        (int(match.group(1)), float(match.group(2)), float(match.group(3))) if match else (None, None, None))
    match = re.search(r"\[chain_v6\] STEADY \(post-warmup (\d+)\): (\d+) steps in ([\d.]+)s = ([\d.]+) fps", text)
    row["steady_fps"] = float(match.group(4)) if match else None
    row["steady_steps"] = int(match.group(2)) if match else None

    simulators = []
    for match in re.finditer(r"\[chain_v6\] sim(\d+) seam: ghost_layers=(\d+) departed peak/frame=(\d+) "
                             r"capacity=(\d+) far_migration=(\d+)(.*)", text):
        counters = {name: int(value) for name, value in re.findall(r"(overflow_\w+)=(\d+)", match.group(6))}
        simulators.append({"sim": int(match.group(1)), "ghost_layers": int(match.group(2)),
                           "departed_peak": int(match.group(3)), "departed_capacity": int(match.group(4)),
                           "far_migration": int(match.group(5)), "overflow": counters})
    for match in re.finditer(r"\[chain_v6\] sim(\d+) \(dev(\d+)\): alive=(" + NUMBER + r") pool_used=([\d.]+)% "
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
    row["simulators"] = sorted(simulators, key=lambda entry: entry["sim"])
    clamp_counts = [entry.get("init_clamp_count") for entry in row["simulators"]]
    row["init_clamp_total"] = (sum(value for value in clamp_counts if value is not None)
                               if any(value is not None for value in clamp_counts) else None)
    overflow_sums: dict = {}
    for entry in row["simulators"]:
        for name, value in entry.get("overflow", {}).items():
            overflow_sums[name] = overflow_sums.get(name, 0) + value
    row["overflow_by_counter"] = overflow_sums

    match = re.search(r"\[chain_v6\] final: total=(" + NUMBER + r") \(expected (" + NUMBER + r")\) drift=(-?\d+) "
                      r"stamp_errors gpu=(\d+) host=(\d+) overflow_total=(\d+) far_migration_total=(\d+)", text)
    row["final_line"] = bool(match)
    if match:
        row.update({"alive_end": to_int(match.group(1)), "alive_expected": to_int(match.group(2)),
                    "drift": int(match.group(3)), "stamp_errors_gpu": int(match.group(4)),
                    "stamp_errors_host": int(match.group(5)), "overflow_total": int(match.group(6)),
                    "far_migration_total": int(match.group(7))})
    seam_lines = re.findall(r"\[chain_v6\] seam (\d+) \(col (\d+)\): L_overshoot=(\S+)dx R_overshoot=(\S+)dx "
                            r"dup=(\d+) (OK|\*\*\* FAIL \*\*\*)", text)
    row["seam_checks"] = [{"seam": int(seam), "column": int(column), "left_overshoot_dx": float(left),
                           "right_overshoot_dx": float(right), "duplicates": int(duplicates),
                           "ok": verdict == "OK"} for seam, column, left, right, duplicates, verdict in seam_lines]
    match = re.search(r"\[chain_v6\] fields: rho\[(\S+),(\S+)\] vmax=(\S+) (OK|\*\*\* FAIL \*\*\*)", text)
    if match:
        values = [match.group(1), match.group(2), match.group(3)]
        row["fields"] = {"rho_min": values[0], "rho_max": values[1], "vmax": values[2],
                         "ok": match.group(4) == "OK"}
        row["nan_seen"] = any(value.lower() in ("nan", "-nan", "inf", "-inf") for value in values)
    else:
        row["fields"] = None
        row["nan_seen"] = None                       # no field check ran (e.g. --no-seam-check)
    row["validation_failed_line"] = "*** VALIDATION FAILED ***" in text

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
    return row


def classify(row: dict, return_code: int) -> tuple[str, list[str]]:
    reasons = []
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
    arguments = parser.parse_args()

    text = pathlib.Path(arguments.log).read_text(encoding="utf-8", errors="replace")
    row = {"label": arguments.label, "rc": arguments.rc, "node": arguments.node}
    row.update(parse_log(text))
    if arguments.start is not None and arguments.end is not None:
        row["wall_seconds"] = round(arguments.end - arguments.start, 1)
        if arguments.telemetry:
            row["telemetry_peaks"] = telemetry_peaks(arguments.telemetry, arguments.start, arguments.end)
    row["status"], row["reasons"] = classify(row, arguments.rc)
    if arguments.results:
        with open(arguments.results, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
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
          + (f" reasons={';'.join(row['reasons'])}" if row["reasons"] else ""), flush=True)
    return 0 if row["status"] in ("pass", "threshold_only") else 3


if __name__ == "__main__":
    sys.exit(main())
