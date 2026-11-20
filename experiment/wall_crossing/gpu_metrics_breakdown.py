"""
NOTE (report section 26, 2026-11-20): on the second machine this gave no usable per-dispatch data: the trace has no
workload rows (the single bench records its step command buffer before the collection starts) and in the sampled
window GR Active sat at 100 % with Compute Warps in Flight at 0, which is not understood. Kept for a later attempt.

gpu_metrics_breakdown.py - report section 26, 2.2: per-dispatch hardware metrics of the single-GPU step from Nsight
Systems' GPU Metrics sampling (needs a process with access to the GPU performance counters: on this project's
machines the Administrator session of the second machine) and the step's own per-dispatch GPU timestamps.

    python -m experiment.wall_crossing.gpu_metrics_breakdown run --label c3d_crossing \
        --case cases/cavity3d_1m_crossing/case.yaml --device 0 --dir D:/.../gpu_metrics
    python -m experiment.wall_crossing.gpu_metrics_breakdown analyse --label c3d_crossing --dir D:/.../gpu_metrics

run: `nsys profile --trace=vulkan --vulkan-gpu-workload=individual --gpu-metrics-devices=<gpu>
--gpu-metrics-frequency=<Hz>` around experiment/v7/_run_v7_single_bench.py with a tick after every dispatch (also
between the two dispatches of a split pass, so each dispatch runs alone between barriers) and its per-frame CSV;
exported to sqlite, the .nsys-rep deleted.
analyse: every step command buffer of the measured window (a workload within 20 % of the median step's duration)
is laid out with the CSV's median per-dispatch durations in recording order (predict, update_voxel, the fused pass's
dispatches, density copy, force's dispatches; a split pass near-wall first, the default order); the metric samples
inside the middle 80 % of each dispatch's window are averaged per metric over all steps. The tick overhead sits in
the CSV durations, so the windows are approximate at the edges (hence the 80 %). Writes <dir>/<label>_metrics.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sqlite3
import statistics
import subprocess
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

NSYS_CANDIDATES = (r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.2\target-windows-x64\nsys.exe",
                   r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.5.2\target-windows-x64\nsys.exe")
ORDER_SPLIT = ("predict_us", "update_voxel_us", "correction_density_split_near_wall_us",
               "correction_density_split_interior_us", "density_copy_us", "force_split_near_wall_us",
               "force_split_interior_us")
ORDER_PLAIN = ("predict_us", "update_voxel_us", "correction_density_us", "density_copy_us", "force_us")


def nsys() -> str:
    for candidate in NSYS_CANDIDATES:
        if pathlib.Path(candidate).exists():
            return candidate
    raise FileNotFoundError("nsys.exe not found")


def run(arguments) -> int:
    directory = pathlib.Path(arguments.dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / arguments.label
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", VK_LOADER_LAYERS_DISABLE="VK_LAYER_KHRONOS_validation",
                       V7_BENCH_TICK_DETAIL="kernel")
    for setting in arguments.env:
        name, _, value = setting.partition("=")
        environment[name] = value
    command = [nsys(), "profile", "--trace=vulkan", "--vulkan-gpu-workload=individual", "--sample=none",
               "--cpuctxsw=none", f"--gpu-metrics-devices={arguments.gpu}",
               f"--gpu-metrics-frequency={arguments.frequency}", "--delay", str(arguments.delay), "--duration",
               str(arguments.duration), "-o", str(stem), "--force-overwrite=true",
               sys.executable, "experiment/v7/_run_v7_single_bench.py", "--case", arguments.case, "--device",
               str(arguments.device), "--max-steps", str(arguments.max_steps), "--warmup", str(arguments.warmup),
               "--bench-csv", str(stem) + "_frames.csv"]
    with open(str(stem) + ".log", "w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        print(f"nsys exit {result.returncode}; see {stem}.log")
    subprocess.run([nsys(), "export", "--type", "sqlite", "--force-overwrite", "true", "-o", str(stem) + ".sqlite",
                    str(stem) + ".nsys-rep"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pathlib.Path(str(stem) + ".nsys-rep").unlink()
    return 0


def analyse(arguments) -> int:
    """Steps from the idle gaps of 'GR Active' (the single bench waits for each frame, so the GPU idles between
    steps; the step command buffer is recorded before the collection starts, so the trace has no workload rows),
    dispatch windows from the CSV's median durations scaled to each step's active span."""
    import numpy as np
    directory = pathlib.Path(arguments.dir)
    stem = directory / arguments.label
    with open(str(stem) + "_frames.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    def has_values(key: str) -> bool:
        return any(row.get(key) not in (None, "") for row in rows)
    order = ORDER_SPLIT if has_values("correction_density_split_interior_us") else ORDER_PLAIN
    durations = {key: statistics.median(float(row[key]) for row in rows if row.get(key) not in (None, ""))
                 for key in order}
    connection = sqlite3.connect(str(stem) + ".sqlite")
    names = dict(connection.execute("select metricId, metricName from TARGET_INFO_GPU_METRICS").fetchall())
    samples = np.array(connection.execute("select timestamp, metricId, value from GPU_METRICS").fetchall(),
                       dtype=np.float64)
    connection.close()
    samples = samples[np.argsort(samples[:, 0], kind="stable")]
    active_id = next(key for key, name in names.items() if name.startswith("GR Active"))
    active = samples[samples[:, 1] == active_id]
    busy = active[:, 2] > 5.0
    # runs of busy samples = steps (with their in-step ticks the dispatches are back to back on the GPU)
    edges = np.flatnonzero(np.diff(busy.astype(np.int8)))
    starts = edges[busy[edges + 1]] + 1
    ends = edges[~busy[edges + 1]] + 1
    if ends.size and starts.size and ends[0] < starts[0]:
        ends = ends[1:]
    count = min(starts.size, ends.size)
    spans = np.stack([active[starts[:count], 0], active[ends[:count] - 1, 0]], axis=1)
    lengths = spans[:, 1] - spans[:, 0]
    total = sum(durations.values())
    step = np.median(lengths[np.abs(lengths / 1000.0 - total) < 0.3 * total])
    steps = spans[np.abs(lengths - step) < 0.15 * step]
    offsets, position = {}, 0.0
    for key in order:
        offsets[key] = (position, position + durations[key])
        position += durations[key]
    scale = (step / 1000.0) / position        # stretch the CSV layout to the sampled step length
    result = {"label": arguments.label, "steps": int(len(steps)), "step_us_sampled": float(step / 1000.0),
              "step_us_ticks": float(position), "dispatch_us": durations, "metrics": {}}
    times = samples[:, 0]
    for key in order:
        first, last = offsets[key]
        width = (last - first) * scale * 1000.0
        if width < 3 * (1e9 / arguments.frequency):
            continue
        parts = [slice(np.searchsorted(times, start + first * scale * 1000.0 + 0.1 * width),
                       np.searchsorted(times, start + first * scale * 1000.0 + 0.9 * width)) for start, _ in steps]
        values = {}
        for metric_id, name in names.items():
            chunks = [samples[part][samples[part][:, 1] == metric_id, 2] for part in parts]
            chunks = [chunk for chunk in chunks if chunk.size]
            if chunks:
                values[name] = float(np.mean(np.concatenate(chunks)))
        result["metrics"][key] = values
    (directory / f"{arguments.label}_metrics.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({"steps": result["steps"], "step_us_sampled": result["step_us_sampled"],
                      "step_us_ticks": result["step_us_ticks"]}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--label", required=True)
    runner.add_argument("--case", required=True)
    runner.add_argument("--device", type=int, default=0)
    runner.add_argument("--gpu", type=int, default=0, help="the GPU index nsys uses for --gpu-metrics-devices")
    runner.add_argument("--env", action="append", default=[])
    runner.add_argument("--max-steps", type=int, default=3000)
    runner.add_argument("--warmup", type=int, default=1000)
    runner.add_argument("--frequency", type=int, default=50000)
    runner.add_argument("--delay", type=float, default=20.0, help="seconds before collection starts")
    runner.add_argument("--duration", type=float, default=6.0, help="seconds of collection")
    runner.add_argument("--dir", required=True)
    analyser = sub.add_parser("analyse")
    analyser.add_argument("--label", required=True)
    analyser.add_argument("--dir", required=True)
    analyser.add_argument("--frequency", type=int, default=50000)
    arguments = parser.parse_args()
    return run(arguments) if arguments.command == "run" else analyse(arguments)


if __name__ == "__main__":
    sys.exit(main())
