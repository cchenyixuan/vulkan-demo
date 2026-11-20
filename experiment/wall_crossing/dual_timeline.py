"""
dual_timeline.py - report section 26, 2.3: the pipelined timeline of a two-slab chain run from an Nsight Systems
Vulkan trace (command-buffer level; no GPU performance counters needed).

    .venv/Scripts/python.exe -m experiment.wall_crossing.dual_timeline run --label simple --case cases/cavity3d_1m/case.yaml
    .venv/Scripts/python.exe -m experiment.wall_crossing.dual_timeline tables --label simple --label crossing

run: `nsys profile --trace=vulkan --vulkan-gpu-workload=individual` around experiment/v7/_run_v7_chain_bench.py
(equal weights, device map 0,1, depth 2, no anatomy timestamps), exported to sqlite under --dir (outside the
repository; the .nsys-rep is deleted after the export, the sqlite after `tables` unless --keep).
tables: per GPU and Vulkan context (queue) of the steps in the measured window (the last --fraction of the run's
workloads): workloads per step, GPU busy time per step (the union of the workloads' intervals), and on each GPU's
compute queue the step period and the idle time per step, split by the position of the gap in the step's cycle of
command buffers (the queue's workloads repeat in a fixed order every step; the longest is phase B). Writes
claude/wall_crossing/tables/dual_timeline.md and logs/wall_crossing/section26/dual_timeline/summary.json.
On 2026-11-20 (two 5090s, Nsight Systems 2025.5.2) the second GPU's workload times in the export were unusable
(single workloads of about 1 s on its compute queue); only the first GPU's rows are read in the report.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sqlite3
import statistics
import subprocess
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.run_log import LOG_ROOT, LoggedRun     # noqa: E402

NSYS = r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.5.2\target-windows-x64\nsys.exe"


def run(arguments) -> int:
    directory = pathlib.Path(arguments.dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / arguments.label
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", VK_LOADER_LAYERS_DISABLE="VK_LAYER_KHRONOS_validation")
    for setting in arguments.env:
        name, _, value = setting.partition("=")
        environment[name] = value
    command = [NSYS, "profile", "--trace=vulkan", "--vulkan-gpu-workload=individual", "--sample=none",
               "--cpuctxsw=none", "-o", str(stem), "--force-overwrite=true",
               sys.executable, "experiment/v7/_run_v7_chain_bench.py", "--case", arguments.case, "--weights", "1,1",
               "--device-map", "0,1", "--max-steps", str(arguments.max_steps), "--warmup", str(arguments.warmup)]
    with LoggedRun(kind="dual_timeline", case=arguments.case, gpu_uuids=arguments.uuid,
                   command=" ".join(command[command.index(sys.executable):])) as logged:
        logged.extra["label"] = arguments.label
        logged.extra["environment"] = arguments.env
        with open(str(stem) + ".log", "w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise RuntimeError(f"nsys profile exit {result.returncode}, see {stem}.log")
        subprocess.run([NSYS, "export", "--type", "sqlite", "--force-overwrite", "true", "-o", str(stem) + ".sqlite",
                        str(stem) + ".nsys-rep"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pathlib.Path(str(stem) + ".nsys-rep").unlink()
    return 0


def union_length(intervals: np.ndarray) -> float:
    total, current_start, current_end = 0.0, None, None
    for start, end in intervals[np.argsort(intervals[:, 0])]:
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        total += current_end - current_start
    return total


def analyse(path: pathlib.Path, fraction: float) -> dict:
    connection = sqlite3.connect(str(path))
    rows = np.array(connection.execute("select gpu, contextId, start, end from VULKAN_WORKLOAD order by start").fetchall(),
                    dtype=np.float64)
    connection.close()
    result = {}
    cut = rows[int(len(rows) * (1.0 - fraction)), 2]
    window = rows[rows[:, 2] >= cut]
    span_end = window[:, 3].max()
    for gpu in sorted({int(value) for value in window[:, 0]}):
        on_gpu = window[window[:, 0] == gpu]
        contexts = {}
        for context in sorted({int(value) for value in on_gpu[:, 1]}):
            workloads = on_gpu[on_gpu[:, 1] == context]
            durations = (workloads[:, 3] - workloads[:, 2]) / 1000.0
            contexts[context] = {"workloads": int(len(workloads)), "median_us": float(np.median(durations)),
                                 "busy_us": union_length(workloads[:, 2:4]) / 1000.0}
        # the compute queue: the context with the most busy time
        compute = max(contexts, key=lambda key: contexts[key]["busy_us"])
        workloads = on_gpu[on_gpu[:, 1] == compute]
        durations = (workloads[:, 3] - workloads[:, 2]) / 1000.0
        # the step's cycle: from one long workload (phase B: above half the 90th percentile; a defrag's extra
        # command buffer is too rare to move the percentile) to the next; only cycles of the modal length count
        longest = np.flatnonzero(durations > 0.5 * np.percentile(durations, 90))
        cycle = int(statistics.mode(np.diff(longest))) if len(longest) > 2 else 1
        starts_list, ends_list, next_starts = [], [], []
        for first, second in zip(longest[:-1], longest[1:]):
            if second - first != cycle:
                continue
            starts_list.append(workloads[first:second, 2])
            ends_list.append(workloads[first:second, 3])
            next_starts.append(workloads[second, 2])
        starts, ends, next_starts = np.array(starts_list), np.array(ends_list), np.array(next_starts)
        steps = len(starts)
        period = float(np.median(next_starts - starts[:, 0])) / 1000.0
        busy = (ends - starts) / 1000.0
        # idle gaps before workload k of the cycle (k = 0 is phase B: the gap between the cycle's last workload
        # and the next cycle's phase B)
        gaps = np.zeros((steps, cycle))
        gaps[:, 0] = (next_starts - ends[:, cycle - 1]) / 1000.0
        for position in range(1, cycle):
            gaps[:, position] = (starts[:, position] - ends[:, position - 1]) / 1000.0
        result[f"gpu{gpu}"] = {"contexts": {str(key): value for key, value in contexts.items()},
                               "compute_context": int(compute), "cycle": cycle, "steps": int(steps),
                               "period_us": period,
                               "per_step": {str(context): value["busy_us"] / steps for context, value in contexts.items()},
                               "compute_workload_median_us": [float(np.median(busy[:, k])) for k in range(cycle)],
                               "compute_gap_before_median_us": [float(np.median(np.maximum(gaps[:, k], 0.0)))
                                                                for k in range(cycle)]}
    return result


def tables(arguments) -> int:
    from experiment.wall_crossing.report_tables import markdown_table, write
    directory = pathlib.Path(arguments.dir)
    summary, rows = {}, []
    for label in arguments.label:
        result = analyse(directory / f"{label}.sqlite", arguments.fraction)
        summary[label] = result
        for gpu, entry in result.items():
            cycle = entry["cycle"]
            rows.append([label, gpu, f"{entry['period_us']:.1f}",
                         " / ".join(f"{value:.1f}" for value in entry["compute_workload_median_us"]),
                         " / ".join(f"{value:.1f}" for value in entry["compute_gap_before_median_us"]),
                         f"{sum(entry['compute_workload_median_us']):.1f}",
                         "；".join(f"队列 {context}：{value['workloads'] // max(entry['steps'], 1)} 个/步，"
                                  f"中位 {value['median_us']:.1f} µs"
                                  for context, value in entry["contexts"].items()
                                  if int(context) != entry["compute_context"])])
        if not arguments.keep:
            (directory / f"{label}.sqlite").unlink()
    out = LOG_ROOT / "section26" / "dual_timeline"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=int), encoding="utf-8")
    write("dual_timeline.md", markdown_table(
        ["运行", "卡", "每步周期 µs", "计算队列每步各命令缓冲的 GPU 时长（中位，µs；从 phase B 起）",
         "各命令缓冲之前的空闲（中位，µs）", "计算队列每步忙的时间合计", "其他队列"], rows))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--label", required=True)
    runner.add_argument("--case", required=True)
    runner.add_argument("--env", action="append", default=[])
    runner.add_argument("--max-steps", type=int, default=3000)
    runner.add_argument("--warmup", type=int, default=1500)
    runner.add_argument("--uuid", action="append",
                        default=["fb83f0dafd9e1d38dc05e7ae643e22a8", "ae137c668a40f5acaab90a83f7cda175"])
    runner.add_argument("--dir", required=True)
    tabler = sub.add_parser("tables")
    tabler.add_argument("--label", action="append", required=True)
    tabler.add_argument("--dir", required=True)
    tabler.add_argument("--fraction", type=float, default=0.4)
    tabler.add_argument("--keep", action="store_true")
    arguments = parser.parse_args()
    return run(arguments) if arguments.command == "run" else tables(arguments)


if __name__ == "__main__":
    sys.exit(main())
