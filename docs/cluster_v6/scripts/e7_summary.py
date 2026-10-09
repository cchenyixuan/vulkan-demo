"""
e7_summary.py — E7 smoke tables (markdown) from one job's log directory.

Reads what the job wrote next to its runs:
  - results.jsonl (parse_run_v6.py, one row per chain run): status, K, case, fps, invariants, chain build
    time and this process's host memory peak (run_chain_v6.py stage lines), VRAM peak, wall time;
  - fss/*.json (fused_single_step.py): verdict, rows compared, deep-wall rows left out, whether L / kernel
    sum / grad rho are bit-identical, and the rho / P differences (primary fields, after the copy);
  - memory_windows.jsonl (host_memory_sampler.py --summary): host memory peaks inside a run window;
  - for runs with pool-region lines (V7_POOL_PEAKS=1, E15): per link and region the capacity, peak and
    p99.9, and per slab the departed peak / capacity and the install-tail depth.

Usage: python docs/cluster_v6/scripts/e7_summary.py LOG_DIR [--out FILE.md]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

FLUID_CATEGORIES = ("interior_fluid", "band_fluid", "ghost_self_fluid")
CORRECTION_FIELDS = ("correction_inverse", "kernel_sum", "density_gradient")


def read_json_lines(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def case_name(row: dict) -> str:
    for line in row.get("config_lines", []):
        match = re.search(r"case cases/(\S+?)/case\.yaml", line)
        if match:
            return match.group(1).replace("aligned/", "")
    return "?"


def gibibytes(text) -> str:
    return text.replace("GiB", "") if isinstance(text, str) else "-"


def run_table(rows: list[dict]) -> str:
    lines = ["| run | case | K | status | steady fps | total fps | build (s) | VmHWM at exit (GiB) | drift | overflow | "
             "far migration | stamps | alive start → end | VRAM peak (MiB, max card) | wall (s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        stages = row.get("stages") or {}
        exit_stage = stages.get("exit") or stages.get("loop_end") or {}
        vram = [entry.get("memory_used_peak_mib") for entry in (row.get("telemetry_peaks") or {}).values()
                if entry.get("memory_used_peak_mib")]
        vram_text = f"{max(vram):.0f}" if vram else "-"
        alive = (f"{row['alive_start']:,} → {row['alive_end']:,}"
                 if isinstance(row.get("alive_start"), int) and isinstance(row.get("alive_end"), int) else "-")
        lines.append(
            f"| {row['label']} | {case_name(row)} | {row.get('slab_count', 1)} | {row['status']} | "
            f"{row.get('steady_fps') or '-'} | {row.get('total_fps') or '-'} | {row.get('build_seconds') or '-'} | "
            f"{gibibytes(exit_stage.get('vm_hwm'))} | {row.get('drift', '-')} | {row.get('overflow_total', '-')} | "
            f"{row.get('far_migration_total', '-')} | {row.get('stamp_errors_gpu', '-')}+{row.get('stamp_errors_host', '-')} | "
            f"{alive} | {vram_text} | {row.get('wall_seconds', '-')} |")
        if row.get("reasons"):
            lines.append(f"| ↳ reasons | {'; '.join(row['reasons'])} |" + " |" * 13)
    return "\n".join(lines)


def fss_table(directory: pathlib.Path) -> str:
    lines = ["| configuration | K | verdict | rows compared | deep-wall rows left out | L / kernel sum / ∇ρ "
             "bit-identical | ρ rows differing (fluid) | of them 1 ULP | max ULP ρ | max \\|Δρ\\| | max \\|ΔP\\| (Pa) | wall (s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for path in sorted(directory.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        statistics = document.get("statistics", {})
        rows = sum(fields["correction_inverse"].get("rows", 0) for fields in statistics.values()
                   if "correction_inverse" in fields)
        left_out = sum((document.get("rows_left_out_of_correction_fields") or {}).values())
        differing = one_ulp = 0
        max_ulp = 0
        max_density = max_pressure = 0.0
        for category, fields in statistics.items():
            density = fields.get("primary_density")
            pressure = fields.get("primary_pressure")
            if density and category in FLUID_CATEGORIES:     # a category without rows carries {"rows": 0}
                differing += density.get("rows", 0) - density.get("bit_identical", density.get("rows", 0))
                one_ulp += density.get("ulp_1", 0)
                max_ulp = max(max_ulp, density.get("max_ulp") or 0)
                max_density = max(max_density, density.get("max_abs") or 0.0)
            if pressure:
                max_pressure = max(max_pressure, pressure.get("max_abs") or 0.0)
        lines.append(f"| {path.stem} | {len(document.get('device_map', []))} | {document.get('verdict')} | {rows:,} | "
                     f"{left_out:,} | {document.get('bit_identical_correction_fields')} | {differing} | {one_ulp} | "
                     f"{max_ulp} | {max_density:.3e} | {max_pressure:.3g} | {document.get('wall_s', 0):.1f} |")
    return "\n".join(lines)


def memory_table(windows: list[dict]) -> str:
    gibibyte = 1024.0 ** 3

    def text(value):
        return f"{value / gibibyte:.1f}" if isinstance(value, (int, float)) else "-"

    lines = ["| window | samples | job cgroup peak (GiB) | node in use peak (GiB) | run processes RSS sum peak (GiB) | "
             "largest process peak (VmHWM, GiB) | sum of process peaks (GiB) | processes |",
             "|---|---|---|---|---|---|---|---|"]
    for window in windows:
        lines.append(f"| {window['label']} | {window['samples']} | {text(window.get('cgroup_peak_bytes'))} | "
                     f"{text(window.get('node_used_peak_bytes'))} | {text(window.get('matched_rss_sum_peak_bytes'))} | "
                     f"{text(window.get('window_process_hwm_max_bytes'))} | {text(window.get('window_process_hwm_sum_bytes'))} | "
                     f"{window.get('window_processes', '-')} |")
    return "\n".join(lines)


def pool_tables(rows: list[dict]) -> str:
    parts = []
    for row in rows:
        regions = row.get("pool_regions") or []
        if not regions:
            continue
        by_region: dict = {}
        for entry in regions:
            by_region.setdefault(entry["region"], []).append(entry)
        lines = [f"**{row['label']}** ({case_name(row)}, {row.get('total_steps')} steps)", "",
                 "| region | links | capacity (slots) | peak (max over links) | link of the peak | p99.9 (max) | "
                 "mean (max) | peak / capacity |", "|---|---|---|---|---|---|---|---|"]
        for region, entries in sorted(by_region.items()):
            worst = max(entries, key=lambda entry: entry["peak"] / entry["capacity"] if entry["capacity"] else 0)
            capacities = sorted({entry["capacity"] for entry in entries})
            lines.append(f"| {region} | {len(entries)} | {'/'.join(f'{value:,}' for value in capacities)} | "
                         f"{worst['peak']:,} | {worst['link']} | {max(entry['p999'] for entry in entries):,} | "
                         f"{max(entry['mean'] for entry in entries):,.1f} | {worst['occupancy']} |")
        lines += ["", "| slab | departed peak / capacity | install tail: peak migration | pool used |",
                  "|---|---|---|---|"]
        for simulator in row.get("simulators", []):
            lines.append(f"| {simulator['sim']} | {simulator.get('departed_peak', '-')} / "
                         f"{simulator.get('departed_capacity', '-')} | {simulator.get('peak_migration', '-')} | "
                         f"{simulator.get('pool_used_percent', '-')} % |")
        parts.append("\n".join(lines))
    return "\n\n".join(parts) if parts else "(no pool-region lines)"


def main() -> int:
    parser = argparse.ArgumentParser(description="E7 smoke tables from a job log directory")
    parser.add_argument("log_directory")
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()
    directory = pathlib.Path(arguments.log_directory)
    rows = read_json_lines(directory / "results.jsonl")
    sections = ["## Runs", run_table(rows),
                "## fused_single_step (B1)", fss_table(directory / "fss"),
                "## Host memory windows", memory_table(read_json_lines(directory / "memory_windows.jsonl")),
                "## Pool peaks (V7_POOL_PEAKS=1)", pool_tables(rows)]
    text = "\n\n".join(sections) + "\n"
    if arguments.out:
        pathlib.Path(arguments.out).write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
