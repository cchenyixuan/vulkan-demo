"""
report_tables_v6.py — E30 run table (markdown) from the results.jsonl rows that
parse_run_v6.py appends during a job.

One row per run with the S2-S4 record fields: completion, steady fps, drift,
all overflow counters (sum; the non-zero ones are named), far_migration, frame
stamp errors (GPU + host), alive start -> end, initialization seam clamp count,
NaN, wall clock, per-card VRAM peak (nvidia-smi samples inside the run window).
A run of another solver than v6 (E39: parse_run_v6.py's "solver", e.g. v7)
carries it after the case name, "case · v7"; v6 rows are printed as before.

Usage:
    python docs/cluster_v6/scripts/report_tables_v6.py JOB_LABEL=RESULTS.jsonl [...]
    python docs/cluster_v6/scripts/report_tables_v6.py --caps JOB_LABEL=CAPS.json [...]   (capability table)
"""

from __future__ import annotations

import json
import re
import sys


def case_name(row: dict) -> str:
    for line in row.get("config_lines", []):
        match = re.search(r"case cases/(\S+?)/case\.yaml", line)        # E7: also cases/aligned/<name>
        if match:
            return match.group(1)
    return "?"


def case_text(row: dict) -> str:
    """The case, with the solver after it when the run was not v6 (rows from before E39 carry no solver)."""
    solver = row.get("solver") or "v6"
    return case_name(row) + (f" · {solver}" if solver != "v6" else "")


def vram_text(row: dict) -> str:
    peaks = row.get("telemetry_peaks") or {}
    used = [(int(index), values.get("memory_used_peak_mib")) for index, values in peaks.items()]
    # cards the run touched: the device map when the log names it, else every card above 100 MiB
    device_map = row.get("device_map")
    if device_map:
        touched = sorted(set(int(index) for index in device_map))
        used = [(index, value) for index, value in used if index in touched]
    else:
        used = [(index, value) for index, value in used if value and value > 100]
    return " / ".join(f"{index}:{value / 1024:.1f}" for index, value in sorted(used) if value is not None) or "-"


def overflow_text(row: dict) -> str:
    total = row.get("overflow_total")
    nonzero = [name for name, value in (row.get("overflow_by_counter") or {}).items() if value]
    return f"{total}" + (f" ({', '.join(nonzero)})" if nonzero else "")


def table(groups: list[tuple[str, list[dict]]]) -> str:
    header = ("| job | run | case | K | steps / warmup | steady fps | drift | overflow | far_migration | "
              "stamp errors GPU+host | alive start → end | init clamp | NaN | wall s | VRAM peak GiB (card:value) | status |")
    lines = [header, "|" + "---|" * 16]
    for job_label, rows in groups:
        for row in rows:
            steps = row.get("total_steps")
            steady = row.get("steady_steps")
            warmup = steps - steady if isinstance(steps, int) and isinstance(steady, int) else "?"
            alive = f"{row.get('alive_start'):,} → {row.get('alive_end'):,}" if row.get("alive_start") else "-"
            lines.append(
                f"| {job_label} | {row.get('label')} | {case_text(row)} | {row.get('slab_count')} | "
                f"{steps} / {warmup} | {row.get('steady_fps')} | {row.get('drift')} | {overflow_text(row)} | "
                f"{row.get('far_migration_total')} | "
                f"{row.get('stamp_errors_gpu')}+{row.get('stamp_errors_host')} | {alive} | "
                f"{row.get('init_clamp_total')} | {'yes' if row.get('nan_seen') else 'no'} | "
                f"{row.get('wall_seconds')} | {vram_text(row)} | {row.get('status')} |")
    return "\n".join(lines)


def capability_table(groups: list[tuple[str, dict]]) -> str:
    """One row per NVIDIA card of each caps.json (caps_v6.py), then one row per host."""
    lines = ["| job | card (v6 / smi) | PCI | name | driver / API | VRAM GiB | queue families (index: flags × count, timestamp bits) "
             "| transfer-only family × queues | timestampPeriod ns | calibrated domains KHR / EXT | NUMA node: cpulist | PCIe link |",
             "|" + "---|" * 12]
    host_lines = ["| job | host | clocksource | job cpus | python-vulkan / loader |", "|---|---|---|---|---|"]
    for job_label, caps in groups:
        for entry in caps.get("devices", []):
            if not entry.get("is_nvidia_discrete"):
                continue
            families = "; ".join(f"{family['index']}: {family['flags']} × {family['count']}, {family['timestamp_valid_bits']}"
                                 for family in entry.get("queue_families", []))
            transfer_only = [family for family in entry.get("queue_families", []) if family.get("transfer_only")]
            transfer_text = ", ".join(f"family {family['index']} × {family['count']}" for family in transfer_only) or "none"
            domains = entry.get("calibrateable_domains", {})

            def domain_text(value):
                return ", ".join(value) if isinstance(value, list) else str(value)
            numa = entry.get("numa", {})
            heap = next((heap["gib"] for heap in entry.get("heaps", []) if heap.get("device_local")), "?")
            lines.append(
                f"| {job_label} | {entry.get('v6_index')} / {entry.get('nvidia_smi_index')} | {entry.get('pci_bus_id')} | "
                f"{entry.get('name')} | {entry.get('driver_info')} / {entry.get('api_version')} | {heap} | {families} | "
                f"{transfer_text} | {entry.get('timestamp_period_ns')} | KHR: {domain_text(domains.get('KHR'))}; "
                f"EXT: {domain_text(domains.get('EXT'))} | {numa.get('numa_node')}: {numa.get('local_cpulist')} | "
                f"{numa.get('current_link_speed')} x{numa.get('current_link_width')} |")
        host = caps.get("host", {})
        host_lines.append(f"| {job_label} | {host.get('hostname')} | {host.get('clocksource')} | {host.get('job_cpus')} | "
                          f"{host.get('python_vulkan')} / {host.get('vulkan_loader')} |")
    return "\n".join(lines) + "\n\n" + "\n".join(host_lines)


def main() -> int:
    arguments = sys.argv[1:]
    caps_mode = bool(arguments) and arguments[0] == "--caps"
    groups = []
    for argument in arguments[1:] if caps_mode else arguments:
        job_label, path = argument.split("=", 1)
        with open(path, encoding="utf-8") as handle:
            if caps_mode:
                groups.append((job_label, json.load(handle)))
            else:
                groups.append((job_label, [json.loads(line) for line in handle if line.strip()]))
    print(capability_table(groups) if caps_mode else table(groups))
    return 0


if __name__ == "__main__":
    sys.exit(main())
