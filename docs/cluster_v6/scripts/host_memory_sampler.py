"""
host_memory_sampler.py — E7: sample the node's host memory while job steps run (Linux, no root).

Sampling mode writes one CSV row per interval:

  - wall time (epoch seconds),
  - the job cgroup's memory use (cgroup v2 memory.current, or v1 memory.usage_in_bytes; page cache and the
    tmpfs pages this job wrote, e.g. /dev/shm obj caches, are charged there too), empty when unreadable,
  - node memory in use = MemTotal - MemAvailable (/proc/meminfo, the whole node, other tenants included),
  - /dev/shm used bytes,
  - the summed resident set (VmRSS) of this user's python processes whose command line matches --match
    (python only: the `timeout` / `taskset` parents carry the same command line), their count, and the
    largest high-water mark (VmHWM) among them.

Every matched process's VmHWM is also kept per pid (the last value read before it exits) with the times it was
first and last seen, in <out>.hwm.csv, rewritten after every sample, so a short-lived process's peak is not
lost between samples and windows cut while the sampler runs can use it. SIGTERM / SIGINT stop the sampler
cleanly.

Summary mode reads a CSV (and its .hwm.csv) and prints the peaks inside a time window:

    python host_memory_sampler.py --out FILE.csv [--interval 1.0] [--match REGEX] &
    python host_memory_sampler.py --summary FILE.csv [--since EPOCH] [--until EPOCH] [--label LABEL] [--json OUT]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import sys
import time

DEFAULT_MATCH = r"run_chain_v6\.py|fused_single_step|bringup_check_v6\.py|obj_cache_build\.py"
FIELDS = ["time", "cgroup_bytes", "node_used_bytes", "shm_used_bytes", "matched_rss_bytes",
          "matched_processes", "matched_hwm_max_bytes"]


def read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return None


def cgroup_memory_file() -> str | None:
    """The memory-usage file of this process's cgroup (v2 unified, else v1 memory controller)."""
    text = read_text("/proc/self/cgroup") or ""
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy_id, controllers, path = parts
        if hierarchy_id == "0" and controllers == "":
            candidate = f"/sys/fs/cgroup{path}/memory.current"
            if os.path.exists(candidate):
                return candidate
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and "memory" in parts[1].split(","):
            candidate = f"/sys/fs/cgroup/memory{parts[2]}/memory.usage_in_bytes"
            if os.path.exists(candidate):
                return candidate
    return None


def node_used_bytes() -> int | None:
    values = {}
    for line in (read_text("/proc/meminfo") or "").splitlines():
        name, _, rest = line.partition(":")
        fields = rest.split()
        if fields:
            values[name] = int(fields[0]) * 1024
    if "MemTotal" in values and "MemAvailable" in values:
        return values["MemTotal"] - values["MemAvailable"]
    return None


def shm_used_bytes() -> int | None:
    try:
        status = os.statvfs("/dev/shm")
    except OSError:
        return None
    return (status.f_blocks - status.f_bfree) * status.f_frsize


def matched_processes(pattern: re.Pattern, own_pid: int) -> dict[int, tuple[int, int]]:
    """{pid: (VmRSS bytes, VmHWM bytes)} of this user's processes whose command line matches."""
    user_id = os.getuid()
    result = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == own_pid:
            continue
        status = read_text(f"/proc/{entry}/status")
        command = read_text(f"/proc/{entry}/cmdline")
        if not status or not command:
            continue
        values = {}
        for line in status.splitlines():
            name, _, rest = line.partition(":")
            values[name] = rest.split()
        if not values.get("Uid") or int(values["Uid"][0]) != user_id:
            continue
        if not (values.get("Name") or [""])[0].startswith("python"):
            continue
        if not pattern.search(command.replace("\0", " ")):
            continue
        resident = int(values.get("VmRSS", ["0"])[0]) * 1024
        high_water = int(values.get("VmHWM", ["0"])[0]) * 1024
        result[int(entry)] = (resident, high_water)
    return result


def sample(arguments: argparse.Namespace) -> int:
    if not os.path.isdir("/proc") or not hasattr(os, "getuid"):
        print("[memory] no /proc on this host: sampler not started", flush=True)
        return 0
    pattern = re.compile(arguments.match)
    cgroup_file = cgroup_memory_file()
    stop = {"requested": False}

    def request_stop(signal_number, frame):          # noqa: ARG001
        stop["requested"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    # pid -> [VmHWM bytes, command, first seen, last seen]
    high_water_by_process: dict[int, list] = {}
    with open(arguments.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        handle.write(f"# cgroup_file={cgroup_file or 'none'} match={arguments.match}\n")
        handle.flush()
        while not stop["requested"]:
            started = time.time()
            processes = matched_processes(pattern, os.getpid())
            for process_id, (_, high_water) in processes.items():
                record = high_water_by_process.get(process_id)
                if record is None:
                    command = (read_text(f"/proc/{process_id}/cmdline") or "").replace("\0", " ").strip()[:300]
                    record = high_water_by_process[process_id] = [0, command, started, started]
                record[0] = max(record[0], high_water)
                record[3] = started
            cgroup_text = read_text(cgroup_file) if cgroup_file else None
            writer.writerow([f"{started:.3f}",
                             cgroup_text.strip() if cgroup_text else "",
                             node_used_bytes() or "",
                             shm_used_bytes() or "",
                             sum(resident for resident, _ in processes.values()),
                             len(processes),
                             max((high_water for _, high_water in processes.values()), default=0)])
            handle.flush()
            write_high_water_file(arguments.out + ".hwm.csv", high_water_by_process)
            time.sleep(max(0.0, arguments.interval - (time.time() - started)))
    write_high_water_file(arguments.out + ".hwm.csv", high_water_by_process)
    return 0


def write_high_water_file(path: str, high_water_by_process: dict) -> None:
    """<out>.hwm.csv, rewritten after every sample (temporary file + rename) so that windows cut while the
    sampler runs can read every process seen so far: pid, VmHWM, first and last sample time, command."""
    temporary = path + ".tmp"
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pid", "hwm_bytes", "first_seen", "last_seen", "command"])
        for process_id, (high_water, command, first_seen, last_seen) in sorted(high_water_by_process.items()):
            writer.writerow([process_id, high_water, f"{first_seen:.3f}", f"{last_seen:.3f}", command])
    os.replace(temporary, path)


def summarize(arguments: argparse.Namespace) -> int:
    if not os.path.exists(arguments.summary):
        print(f"[memory] {arguments.label}: no samples ({arguments.summary} absent: the sampler did not run)",
              flush=True)
        return 0
    rows = []
    with open(arguments.summary, encoding="utf-8") as handle:
        for row in csv.DictReader(line for line in handle if not line.startswith("#")):
            moment = float(row["time"])
            if arguments.since is not None and moment < arguments.since:
                continue
            if arguments.until is not None and moment > arguments.until:
                continue
            rows.append(row)

    def peak(name: str) -> int | None:
        values = [int(row[name]) for row in rows if row.get(name, "") not in ("", None)]
        return max(values) if values else None

    gibibyte = 1024.0 ** 3
    result = {"label": arguments.label, "samples": len(rows),
              "since": arguments.since, "until": arguments.until,
              "cgroup_peak_bytes": peak("cgroup_bytes"),
              "node_used_peak_bytes": peak("node_used_bytes"),
              "shm_used_peak_bytes": peak("shm_used_bytes"),
              "matched_rss_sum_peak_bytes": peak("matched_rss_bytes"),
              "matched_processes_peak": peak("matched_processes"),
              "matched_hwm_max_bytes": peak("matched_hwm_max_bytes")}
    processes = []
    try:
        with open(arguments.summary + ".hwm.csv", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if arguments.since is not None and float(row["last_seen"]) < arguments.since:
                    continue
                if arguments.until is not None and float(row["first_seen"]) > arguments.until:
                    continue
                processes.append(int(row["hwm_bytes"]))
    except OSError:
        pass
    result["window_processes"] = len(processes)
    result["window_process_hwm_max_bytes"] = max(processes) if processes else None
    result["window_process_hwm_sum_bytes"] = sum(processes) if processes else None
    first = rows[0] if rows else None
    if first:
        result["cgroup_start_bytes"] = int(first["cgroup_bytes"]) if first["cgroup_bytes"] else None
        result["node_used_start_bytes"] = int(first["node_used_bytes"]) if first["node_used_bytes"] else None
    text = " ".join(f"{name}={value / gibibyte:.2f}GiB" if isinstance(value, int) and name.endswith("bytes")
                    else f"{name}={value}" for name, value in result.items())
    print(f"[memory] {text}", flush=True)
    if arguments.json:
        with open(arguments.json, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="E7 host memory sampler")
    parser.add_argument("--out", help="sampling mode: CSV to write")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--match", default=DEFAULT_MATCH, help="regex on the command line of processes to sum")
    parser.add_argument("--summary", help="summary mode: CSV to read")
    parser.add_argument("--since", type=float, default=None)
    parser.add_argument("--until", type=float, default=None)
    parser.add_argument("--label", default="")
    parser.add_argument("--json", default=None, help="append the summary as one JSON line")
    arguments = parser.parse_args()
    if arguments.summary:
        return summarize(arguments)
    if not arguments.out:
        parser.error("--out or --summary is required")
    return sample(arguments)


if __name__ == "__main__":
    sys.exit(main())
