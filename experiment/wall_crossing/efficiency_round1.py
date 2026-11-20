"""
efficiency_round1.py - report section 26: the timing tables of the first efficiency round.

    .venv/Scripts/python.exe -m experiment.wall_crossing.efficiency_round1

Reads logs/wall_crossing/section26/timing/<size>_step/summary.json (kernel_timing.py, V7_BENCH_TICK_DETAIL=step:
the step's GPU time), <size>_kernel/summary.json (a tick after every dispatch), c3d_k2_steady / c3d_k2_anatomy
(dual_timing.py), logs/wall_crossing/near_wall_census/summary.json (the particles each build processes) and
logs/wall_crossing/section26/statistics/*.json (register counts), and writes claude/wall_crossing/tables/
efficiency_round1_{single,kernels,dual_steady,dual_phases}.md.
Labels: simple; base_6beb49c (the commit before this round, run from its own checkout); rule0_material (this tree,
marker rule 0 = sections 22-25 and the near-wall build reading material[j] pair by pair: neither change); rule1_material
(rule 1, the distance rule, only); rule1_position (both: this tree's defaults).
"""
from __future__ import annotations

import json
import pathlib
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.run_log import LOG_ROOT     # noqa: E402

TIMING = LOG_ROOT / "section26" / "timing"
SIZES = (("n250", "2-D 250²"), ("n500", "2-D 500²"), ("n1000", "2-D 1000²"), ("c3d", "3-D 1M"))
# Blackwell GB202 (sm_120): 64K 32-bit registers and at most 48 warps per SM; registers go per warp in units of 256;
# the workgroup is 128 threads = 4 warps
REGISTERS_PER_SM, MAXIMUM_WARPS, ALLOCATION_UNIT, WARPS_PER_GROUP = 65536, 48, 256, 4


def theoretical_occupancy(registers: int) -> float:
    per_warp = -(-registers * 32 // ALLOCATION_UNIT) * ALLOCATION_UNIT
    warps = min(MAXIMUM_WARPS, REGISTERS_PER_SM // per_warp)
    warps -= warps % WARPS_PER_GROUP
    return warps / MAXIMUM_WARPS


def median(summary: dict, label: str, key: str) -> float:
    return summary["labels"][label]["kernels"][key]["median_us"]


def spread(summary: dict, label: str, key: str) -> str:
    entry = summary["labels"][label]["kernels"][key]
    return f"{entry['minimum_us']:.1f}–{entry['maximum_us']:.1f}"


def main() -> int:
    from experiment.wall_crossing.report_tables import markdown_table, write
    rows = []
    for size, text in SIZES:
        summary = json.loads((TIMING / f"{size}_step" / "summary.json").read_text(encoding="utf-8"))
        simple = median(summary, "simple", "step_total_us")
        values = {label: median(summary, label, "step_total_us") for label in summary["labels"]}
        cells = [text, f"{simple:.1f}（{spread(summary, 'simple', 'step_total_us')}）"]
        for label in ("base_6beb49c", "rule0_material", "rule1_material", "rule1_position"):
            cells.append(f"{values[label] / simple:.3f}")
        first = values["rule1_material"] / values["rule0_material"] - 1.0
        second = values["rule1_position"] / values["rule1_material"] - 1.0
        cells += [f"{100 * first:+.1f} %", f"{100 * second:+.1f} %",
                  f"{100 * (values['rule1_position'] / values['base_6beb49c'] - 1.0):+.1f} %"]
        cells.append(f"{100 * (values['rule1_position_no_constraint'] / values['rule1_position'] - 1.0):+.2f} %"
                     if "rule1_position_no_constraint" in values else "—")
        rows.append(cells)
    write("efficiency_round1_single.md", markdown_table(
        ["算例", "simple 每步 µs（三轮范围）", "6beb49c / simple", "两项都不做 / simple", "只做 ① / simple",
         "①② / simple（本轮代码）", "① 的变化", "② 的变化", "本轮代码对 6beb49c", "关掉总体积修正"], rows))

    census = {entry["case"]: entry for entry in
              json.loads((LOG_ROOT / "near_wall_census" / "summary.json").read_text(encoding="utf-8"))}
    statistics_root = LOG_ROOT / "section26" / "statistics"
    kernel_rows = []
    for size, text, case, simple_stats, crossing_stats in (
            ("n1000", "2-D 1000²", "lid_driven_cavity_2d_n1000_xi0p001_eps0p0025_crossing", "n1000_simple", "n1000_crossing"),
            ("c3d", "3-D 1M", "cavity3d_1m_crossing", "c3d_simple", "c3d_crossing")):
        summary = json.loads((TIMING / f"{size}_kernel" / "summary.json").read_text(encoding="utf-8"))
        counts = census[case]
        simple_registers = json.loads((statistics_root / f"{simple_stats}.json").read_text(encoding="utf-8"))["pipelines"]
        crossing_registers = json.loads((statistics_root / f"{crossing_stats}.json").read_text(encoding="utf-8"))["pipelines"]

        def registers(table: dict, name: str) -> int:
            entry = table[name]
            entry = entry[0] if isinstance(entry, list) else entry
            return int(entry.get("statistics", entry)["Register Count"])

        for pass_name, pass_text, simple_key, pipeline in (("correction_density", "fused", "correction_density_us",
                                                            "correction_density_all"),
                                                           ("force", "受力", "force_us", "force_all")):
            simple_note = ("全部流体（另有全部壁面粒子：simple 的 fused 也为它们算密度）" if pass_name == "correction_density"
                           else "全部流体（壁面粒子读材料后返回）")
            lines = (("simple", "simple", simple_key, counts["fluid"], simple_note,
                      registers(simple_registers, pipeline)),
                     ("rule1_position", "crossing 内部版本", f"{pass_name}_split_interior_us",
                      counts["distance"]["warps"]["interior"]["busy_threads"], "内部体素的流体",
                      registers(crossing_registers, f"{pipeline}_split_interior")),
                     ("rule1_position", "crossing 近壁版本", f"{pass_name}_split_near_wall_us",
                      counts["distance"]["fluid_near_wall"], "近壁体素的流体（另有同体素的全部壁面粒子很快返回）",
                      registers(crossing_registers, f"{pipeline}_split_near_wall")),
                     ("rule1_material", "crossing 近壁版本，逐对查材料（诊断）", f"{pass_name}_split_near_wall_us",
                      counts["distance"]["fluid_near_wall"], "同上", registers(crossing_registers,
                                                                              f"{pipeline}_split_near_wall")))
            for label, build, key, particles, particle_text, register_count in lines:
                time = median(summary, label, key)
                kernel_rows.append([text, pass_text, build, f"{time:.1f}", f"{particles:,}（{particle_text}）",
                                    f"{1000.0 * time / particles:.2f}", str(register_count),
                                    f"{100 * theoretical_occupancy(register_count):.1f} %"])
    write("efficiency_round1_kernels.md", markdown_table(
        ["算例", "遍", "版本", "耗时 µs（每次调度前后打时间戳）", "处理的粒子", "每粒子 ns", "每线程寄存器",
         "理论占用率（寄存器所限）"], kernel_rows))

    if not (TIMING / "c3d_k2_anatomy" / "summary.json").exists():
        return 0
    steady = json.loads((TIMING / "c3d_k2_steady" / "summary.json").read_text(encoding="utf-8"))
    simple_fps = steady["labels"]["simple"]["steady_fps"]["median"]
    steady_rows = []
    for label, entry in steady["labels"].items():
        fps = entry["steady_fps"]
        steady_rows.append([label, f"{fps['median']:.1f}（{fps['minimum']:.1f}–{fps['maximum']:.1f}）",
                            f"{simple_fps / fps['median']:.3f}", str(entry.get("failures", 0))])
    write("efficiency_round1_dual_steady.md", markdown_table(
        ["运行", "稳态步速 fps（三轮范围）", "每步时间 / simple", "失败"], steady_rows))
    anatomy = json.loads((TIMING / "c3d_k2_anatomy" / "summary.json").read_text(encoding="utf-8"))
    keys = ("phase_a", "phase_b", "phase_c", "b_to_c_gap", "c_to_a_gap", "a_to_b_gap",
            "correction_density_interior_split_interior", "correction_density_interior_split_near_wall",
            "force_deep_interior_split_interior", "force_deep_interior_split_near_wall",
            "correction_density_interior", "force_deep_interior", "correction_density_boundary", "force",
            "readback_leading_dma", "readback_trailing_dma", "upload_leading_dma", "upload_trailing_dma",
            "upload_leading_to_c_gap", "upload_trailing_to_c_gap")
    phase_rows = []
    for label, entry in anatomy["labels"].items():
        for slab, values in sorted(entry["slabs"].items()):
            phase_rows.append([label, slab] + [f"{values[key]['median_us']:.0f}" if key in values else "—"
                                               for key in keys])
    write("efficiency_round1_dual_phases.md", markdown_table(["运行", "卡"] + list(keys), phase_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
