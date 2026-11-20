"""
gpu_trace_summary.py - report section 26, 2.2: hardware metrics from Nsight Graphics GPU Trace exports
(ngfx --activity "GPU Trace Profiler" --auto-export, metric set "Top-Level Triage" on Blackwell GB20x, real-time
shader profiler) of three steps of the single bench, one export per configuration.

    .venv/Scripts/python.exe -m experiment.wall_crossing.gpu_trace_summary --dir logs/wall_crossing/section26/gpu_trace

The export (<label>/BASE/GPUTRACE_FRAME.xls, tab-separated name / value) holds the traced range as a whole: its
"elapsed" averages include the idle time between the bench's steps, so every rate is divided by the fraction of
elapsed cycles the compute engine was active (gr__compute_cycles_active_queue_sync):
  - achieved occupancy = warps active per SM per active cycle / 48 (GB202: 48 warps per SM);
  - active threads per executed instruction (of 32; smsp__average_thread_inst_executed_per_inst_executed);
  - warp states from the PC sampler (tpc__warps_issue_stalled_*): each as a share of the active warp-cycles
    ("selected" = issuing an instruction);
  - DRAM traffic per step = dram__sectors.sum x 32 B / 3, DRAM bandwidth while active (with --time-every-action
    the trace serialises every dispatch and the counts look as if the caches were flushed between dispatches: a 250^2
    step "reads" ~210 MB for a ~16 MB working set and writes nothing; compare these rows with each other only);
  - SM throughput while active (% of peak).
GPU Trace locks the clocks to base by default, so its durations are not the timing tables'. The per-build rows come
from runs with V7_WALL_CROSSING_SPLIT_TIMING_SKIP (one build's dispatches per split pass; the other class keeps a
stale state; the step's predict / update_voxel / density copy are in every row).
Writes claude/wall_crossing/tables/gpu_trace_summary.md and <dir>/summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

WARPS_PER_SM = 48
STEPS = 3
STATES = (("selected", "发指令"), ("long_scoreboard_pipe_l1tex", "等显存 / L1（long scoreboard）"),
          ("wait", "固定延迟依赖（wait）"), ("short_scoreboard", "short scoreboard"),
          ("branch_resolving", "分支"), ("not_selected", "就绪未选中"), ("dispatch_stall", "dispatch"),
          ("no_instruction", "取指"), ("math_pipe_throttle", "算术管线满"), ("lg_throttle", "LSU 队列满"),
          ("mio_throttle_pipe_mio", "MIO 队列满"), ("drain", "drain"), ("barrier", "barrier"), ("misc", "其他"))
LABELS = (("c3d_simple", "3-D 1M", "simple（整步）"), ("c3d_crossing", "3-D 1M", "crossing（整步）"),
          ("c3d_near_wall_only", "3-D 1M", "crossing 只调度近壁版本"),
          ("c3d_interior_only", "3-D 1M", "crossing 只调度内部版本"),
          ("n1000_simple", "2-D 1000²", "simple（整步）"), ("n1000_crossing", "2-D 1000²", "crossing（整步）"),
          ("n1000_near_wall_only", "2-D 1000²", "crossing 只调度近壁版本"),
          ("n1000_interior_only", "2-D 1000²", "crossing 只调度内部版本"))
# report section 27: the same plus the near-wall-only runs with the register trims off and with the pairwise wall
# pressure (this round's code; --section 27)
LABELS_SECTION27 = (LABELS[:3] + (("c3d_near_wall_only_trim_none", "3-D 1M", "只调度近壁版本，寄存器精简关（= 2f71864）"),
                                  ("c3d_near_wall_only_pairwise", "3-D 1M", "只调度近壁版本，壁面压力逐对计算")) + LABELS[3:4]
                    + LABELS[4:7] + (("n1000_near_wall_only_trim_none", "2-D 1000²", "只调度近壁版本，寄存器精简关（= 2f71864）"),
                                     ("n1000_near_wall_only_pairwise", "2-D 1000²", "只调度近壁版本，壁面压力逐对计算"))
                    + LABELS[7:8])


def read_frame(path: pathlib.Path) -> dict:
    values = {}
    with open(path, encoding="utf-8", errors="replace") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) >= 2:
                try:
                    values[row[0]] = float(row[1])
                except ValueError:
                    pass
    return values


def summarise(values: dict) -> dict:
    active = values["gr__compute_cycles_active_queue_sync.avg.pct_of_peak_sustained_elapsed"] / 100.0
    warps = values["GPUTrace.PCSampler.tpc__warps_active_shader_compute.avg.per_cycle_elapsed"]
    states = {}
    for key, _ in STATES:
        name = f"GPUTrace.PCSampler.tpc__warps_issue_stalled_{key}.avg.per_cycle_elapsed"
        if name in values and warps > 0:
            states[key] = values[name] / warps
    return {"compute_active_fraction": active,
            "achieved_occupancy": warps / active / WARPS_PER_SM if active > 0 else float("nan"),
            "threads_per_instruction": values["smsp__average_thread_inst_executed_per_inst_executed.ratio"],
            "states": states,
            "dram_bytes_per_step": values["dram__sectors.sum"] * 32.0 / STEPS,
            # the export's ".per_second" is per nanosecond: sectors / ns x 32 B = GB/s
            "dram_bandwidth_active_gb_s": values["dram__sectors.sum.per_second"] * 32.0 / active
            if active > 0 else float("nan"),
            "trace_milliseconds": values.get("gpc__cycles_elapsed.avg", float("nan"))
            / (values.get("gpc__cycles_elapsed.avg.per_second", float("nan")) * 1e3),
            "sm_throughput_active_pct": values["GPUTrace.sm__throughput.avg.pct_of_peak_sustained_elapsed"] / active,
            "registers_allocated_active_pct":
                values["tpc__sm_rf_registers_allocated_shader_cs_queue_sync_realtime.avg.pct_of_peak_sustained_elapsed"]
                / active}


def main() -> int:
    from experiment.wall_crossing.report_tables import markdown_table, write
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--section", type=int, default=26, help="26: section 26's runs; 27: with the extra rows")
    arguments = parser.parse_args()
    directory = pathlib.Path(arguments.dir)
    labels = LABELS_SECTION27 if arguments.section == 27 else LABELS
    suffix = "" if arguments.section == 26 else f"_section{arguments.section}"
    summary, rows, state_rows = {}, [], []
    for label, case, text in labels:
        path = directory / label / "BASE" / "GPUTRACE_FRAME.xls"
        if not path.exists():
            rows.append([case, text, "没有导出"] + [""] * 5)
            continue
        result = summarise(read_frame(path))
        summary[label] = result
        rows.append([case, text, f"{100 * result['achieved_occupancy']:.1f} %",
                     f"{result['threads_per_instruction']:.1f}（{100 * result['threads_per_instruction'] / 32:.0f} %）",
                     f"{result['dram_bytes_per_step'] / 1e6:.1f}", f"{result['dram_bandwidth_active_gb_s']:.0f}",
                     f"{result['sm_throughput_active_pct']:.1f} %", f"{100 * result['compute_active_fraction']:.1f} %"])
        state_rows.append([case, text] + [f"{100 * result['states'].get(key, 0.0):.1f}" for key, _ in STATES[:9]])
    (directory / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    write(f"gpu_trace_summary{suffix}.md", markdown_table(
        ["算例", "运行", "实际占用率（计算忙时）", "每条指令的活动线程（/32）", "显存流量 MB / 步",
         "计算忙时的显存带宽 GB/s", "计算忙时 SM 吞吐（峰值的 %）", "trace 期间计算忙的比例"], rows))
    write(f"gpu_trace_states{suffix}.md", markdown_table(
        ["算例", "运行"] + [f"{text} %" for _, text in STATES[:9]], state_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
