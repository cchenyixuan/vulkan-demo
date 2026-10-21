"""
e8_p1_summary.py — E8 phase 1: the tables of the smoke job from its pulled log directory (the e30_lib layout:
results.jsonl, index.tsv, <label>.log, telemetry.csv, host_memory.csv, provenance.json, caps.json, traces/8m_K2,
compare/*.json, dumps/*.json) plus the job's SLURM .out.

Prints (markdown):
  1  job: node, cards, driver, power caps, start / end, provenance
  2  runs: every RESULT row (status, fps, steps, wall, build, alive, invariants, seam / field checks, own columns and
     particles, device-local + scratch MiB, VRAM peak per card, SM clock / power / temperature, ns per particle-step)
  3  VRAM: measured peak per card against e8_vram's model (per run with slabs on a card)
  4  pools: the [e30] pool lines (V7_POOL_PEAKS) and the pool_health lines of every run
  5  links: host copy / DMA bytes per link and step (link_bytes)
  6  trace 8m_K2: per-sim phases (e7_trace_phases) and the hop split (e7_hop_split) with the trace's steady fps
  7  field comparisons: compare/*.json (max / rms |dv|/U, |drho|/rho0 by seam distance)
  8  bring-up
and a JSON of the numbers (--json).

Usage:
    python docs/cluster_v6/scripts/e8_p1_summary.py --job DIR --out-md OUT.md [--json OUT.json] [--slurm-out FILE]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[3]))
import e8_vram  # noqa: E402

RUN_CASE = {"8m": "cavity3d_n192", "s48": "cavity3d_n48", "weak2": "cavity3d_n192_k2", "64m": "cavity3d_n392"}


def load_rows(directory: pathlib.Path) -> list[dict]:
    path = directory / "results.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fmt(value, digits=1):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def grep(path: pathlib.Path, pattern: str) -> list[str]:
    if not path.exists():
        return []
    expression = re.compile(pattern)
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if expression.search(line)]


def run_tool(script: str, arguments: list[str]) -> str:
    command = [sys.executable, str(HERE.parent / script)] + arguments
    try:
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                   cwd=str(HERE.parents[3]), timeout=600)
        return (completed.stdout + ("\n[stderr] " + completed.stderr.strip() if completed.returncode else "")).strip()
    except Exception as error:                                      # noqa: BLE001
        return f"({script} failed: {error})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, help="the pulled job directory (e7_p1_<id>)")
    parser.add_argument("--slurm-out", default=None, help="the SLURM .out file (default: <job>/*.out)")
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()
    job = pathlib.Path(arguments.job)
    out_file = pathlib.Path(arguments.slurm_out) if arguments.slurm_out else next(iter(job.glob("*.out")), None)
    rows = load_rows(job)
    report = {"runs": {}, "compare": {}, "job": {}}
    lines = []

    # ---- 1 job
    lines += ["## 1. 作业", ""]
    if out_file and out_file.exists():
        head = grep(out_file, r"^=== E30 |^PROVENANCE |^GPU health|^power caps|NVIDIA GeForce|^job cpus|^=== E30 .* done|barrier=|^=== E7 job")
        for line in head[:14]:
            lines.append(f"    {line[:230]}")
        report["job"]["out_lines"] = head[:14]
    provenance = job / "provenance.json"
    if provenance.exists():
        report["job"]["provenance"] = json.loads(provenance.read_text(encoding="utf-8"))
    caps = job / "caps.json"
    if caps.exists():
        report["job"]["caps"] = json.loads(caps.read_text(encoding="utf-8"))
    lines.append("")

    # ---- 2 runs
    lines += ["## 2. 运行（RESULT 行）", "",
              "| label | status | K | steps | steady fps | wall s | build s | alive start→end | drift / overflow / far / stamps / clamp | "
              "seam / NaN | own columns | own particles | device-local + scratch MiB | VRAM peak MiB per card | SM MHz (median / min) | "
              "W median | °C max | ns / particle-step |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        label = row["label"]
        slabs = row.get("slab_count")
        particles = row.get("alive_end") or row.get("loaded_particles")
        fps = row.get("steady_fps")
        ns = (1e9 / (fps * particles) * slabs) if fps and particles and slabs else None
        peaks = row.get("telemetry_peaks") or {}
        loop = row.get("telemetry_loop") or {}
        vram = "/".join(f"{int(k)}:{v.get('memory_used_peak_mib', 0):,.0f}" for k, v in sorted(peaks.items(), key=lambda item: int(item[0]))
                        if v.get("memory_used_peak_mib", 0) > 100)
        clocks = "/".join(f"{v.get('sm_clock_median', 0):.0f}" for k, v in sorted(loop.items(), key=lambda item: int(item[0])))
        clock_min = "/".join(f"{v.get('sm_clock_min', 0):.0f}" for k, v in sorted(loop.items(), key=lambda item: int(item[0])))
        power = "/".join(f"{v.get('power_median', 0):.0f}" for k, v in sorted(loop.items(), key=lambda item: int(item[0])))
        temperature = "/".join(f"{v.get('temperature_max', 0):.0f}" for k, v in sorted(loop.items(), key=lambda item: int(item[0])))
        device_local = row.get("device_local_mb") or []
        scratch = row.get("defrag_scratch_mb") or []
        memory = "/".join(f"{a + b:,.0f}" for a, b in zip(device_local, scratch)) if device_local else "-"
        seam = row.get("seam_checks")
        seam_text = ("-" if not seam else "/".join(str(check.get("status", check)) if isinstance(check, dict) else str(check) for check in seam))
        lines.append(f"| {label} | {row.get('status')} (rc {row.get('rc')}) | {slabs} | {row.get('total_steps')} | {fmt(fps)} | "
                     f"{fmt(row.get('wall_seconds'))} | {fmt(row.get('build_seconds'), 2)} | {fmt(row.get('alive_start'))}→{fmt(row.get('alive_end'))} | "
                     f"{row.get('drift')} / {row.get('overflow_total')} / {row.get('far_migration_total')} / {row.get('stamp_errors_gpu')}+{row.get('stamp_errors_host')} / {row.get('init_clamp_total')} | "
                     f"{seam_text} / {row.get('nan_seen')} | {row.get('own_columns')} | {row.get('own_particles')} | {memory} | {vram} | "
                     f"{clocks} / {clock_min} | {power} | {temperature} | {fmt(ns, 2)} |")
        report["runs"][label] = {"status": row.get("status"), "rc": row.get("rc"), "slabs": slabs, "steps": row.get("total_steps"),
                                 "steady_fps": fps, "wall_seconds": row.get("wall_seconds"), "build_seconds": row.get("build_seconds"),
                                 "alive": [row.get("alive_start"), row.get("alive_end")], "drift": row.get("drift"),
                                 "overflow_total": row.get("overflow_total"), "far_migration": row.get("far_migration_total"),
                                 "stamps": [row.get("stamp_errors_gpu"), row.get("stamp_errors_host")], "init_clamp": row.get("init_clamp_total"),
                                 "seam_checks": seam, "nan_seen": row.get("nan_seen"), "own_columns": row.get("own_columns"),
                                 "own_particles": row.get("own_particles"), "device_local_mb": device_local, "scratch_mb": scratch,
                                 "vram_peaks": peaks, "telemetry_loop": loop, "ns_per_particle_step": ns, "link_bytes": row.get("link_bytes"),
                                 "pool_regions": row.get("pool_regions"), "cuts": row.get("cuts"), "stages": row.get("stages"),
                                 "reasons": row.get("reasons"), "errors": row.get("errors")}
    lines.append("")

    # ---- 3 VRAM vs model
    lines += ["## 3. 显存：实测峰值对模型", "", "| run | card | slab own columns | model device-local + scratch MiB | measured | model VRAM | measured peak | diff |",
              "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        short = row["label"].split("_")[0]
        case = RUN_CASE.get(short)
        slabs = row.get("slab_count")
        if not case or not slabs or row.get("status") != "pass":
            continue
        n, n_x = e8_vram.CASES[case]
        grid = e8_vram.lattice(n, n_x)
        columns = row.get("own_columns") or []
        own = row.get("own_particles") or []
        peaks = row.get("telemetry_peaks") or {}
        device_map = row.get("device_map") or list(range(slabs))
        for index in range(slabs):
            if index >= len(columns):
                break
            entry = e8_vram.slab_memory(grid["face"], grid["total"], columns[index], own[index], index > 0, index < slabs - 1, slabs)
            card = str(device_map[index]) if index < len(device_map) else str(index)
            measured_static = ((row.get("device_local_mb") or [None])[index] or 0) + ((row.get("defrag_scratch_mb") or [None])[index] or 0)
            peak = (peaks.get(card) or {}).get("memory_used_peak_mib")
            lines.append(f"| {row['label']} | {card} | {columns[index]} | {entry['total_mib']:,.0f} | {measured_static:,.0f} | {entry['vram_mib']:,.0f} | "
                         f"{fmt(peak, 0)} | {((peak / entry['vram_mib'] - 1) * 100) if peak else 0:+.1f} % |")
    lines.append("")

    # ---- 4 pools
    lines += ["## 4. 池：区的用量（V7_POOL_PEAKS）与 pool_health", ""]
    for row in rows:
        log = job / f"{row['label']}.log"
        pool_lines = grep(log, r"\[e30\] pool |pool_health:|\[partition_v7\] seam:|\[partition_v7\] chain N=")
        if pool_lines:
            lines.append(f"**{row['label']}**")
            lines.append("")
            for line in pool_lines[:40]:
                lines.append(f"    {line[:230]}")
            lines.append("")
    # ---- 5 links
    lines += ["## 5. 链路字节（每条链路每步）", "", "| run | link | host copy KiB | DMA KiB | frames |", "|---|---|---|---|---|"]
    for row in rows:
        for name, entry in sorted((row.get("link_bytes") or {}).items()):
            lines.append(f"| {row['label']} | {name} | {entry.get('host_copy_kib')} | {entry.get('dma_kib')} | {entry.get('frames')} |")
    lines.append("")
    # ---- 6 trace
    trace = job / "traces" / "8m_K2"
    lines += ["## 6. trace（8m_K2）", ""]
    if (trace / "run_meta.json").exists():
        lines += ["```", run_tool("e7_trace_phases.py", [f"8m_K2={trace}"]), "```", "",
                  "```", run_tool("e7_hop_split.py", [f"8m_K2={trace}", "--per-link"]), "```", ""]
        report["trace"] = {"phases": run_tool("e7_trace_phases.py", [f"8m_K2={trace}", "--json", str(job / "trace_phases_8m_K2.json")]),
                           "hops": run_tool("e7_hop_split.py", [f"8m_K2={trace}", "--per-link", "--json", str(job / "hop_split_8m_K2.json")])}
    else:
        lines.append("(no trace directory)")
    lines.append("")
    # ---- 7 compare
    lines += ["## 7. 场比较（第 300 步，K = 2 对 K = 1，流体粒子）", ""]
    for path in sorted((job / "compare").glob("*.json")) if (job / "compare").exists() else []:
        summary = json.loads(path.read_text(encoding="utf-8"))
        report["compare"][path.stem] = summary
        lines.append(f"**{path.stem}**: cuts {summary.get('cuts')}, {summary.get('particles_compared'):,} particles, missing "
                     f"{summary.get('missing_in_test')} / {summary.get('missing_in_reference')}; max |dv|/U {summary.get('dv_max_over_U'):.3e}, "
                     f"rms {summary.get('dv_rms_over_U'):.3e}; max |drho|/rho0 {summary.get('drho_max_over_rho0'):.3e}, rms {summary.get('drho_rms_over_rho0'):.3e}")
        lines.append("")
        md = path.with_suffix(".md")
        if md.exists():
            lines += [line for line in md.read_text(encoding="utf-8").splitlines() if line.startswith("|")]
            lines.append("")
    for path in sorted((job / "dumps").glob("*.json")) if (job / "dumps").exists() else []:
        document = json.loads(path.read_text(encoding="utf-8"))
        invariants = document.get("invariants", {})
        lines.append(f"dump {path.stem}: valid {invariants.get('valid')}, drift {invariants.get('drift')}, overflow {invariants.get('overflow_total')}, "
                     f"missing {invariants.get('missing_ids')}, duplicates {invariants.get('duplicate_ids')}, stamps {invariants.get('stamp_errors_gpu')}/"
                     f"{invariants.get('stamp_errors_host')}, far {invariants.get('far_migration_count')}, alive {document.get('bookkeeping', {}).get('alive_total')}, "
                     f"frames {document.get('frames_time_s', 0):.1f} s")
    lines.append("")
    # ---- 8 bring-up
    lines += ["## 8. bring-up 与步骤", ""]
    for label in ("s0_bringup", "s0_caps", "s0_obj_cache"):
        log = job / f"{label}.log"
        tail = grep(log, r"PASS|FAIL|fps|devices|staged|hit|missing|OK")[-6:]
        lines.append(f"**{label}**")
        for line in tail:
            lines.append(f"    {line[:200]}")
        lines.append("")
    text = "\n".join(lines)
    pathlib.Path(arguments.out_md).write_text(text + "\n", encoding="utf-8")
    print(text[:6000])
    if arguments.json:
        pathlib.Path(arguments.json).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
