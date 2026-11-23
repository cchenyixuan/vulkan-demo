"""
nsys_metrics.py - claude/wall_store_report.md section 10.4: Nsight Systems' GPU Metrics (gb20x set, 200 kHz, the
headless 5090) on the plain builds' steady steps, set against the in-shader dispatch timeline (dispatch_timeline.py).

GPU Metrics sample hardware counters (SMs active, compute warps in flight, ...) every 5 us without touching the
command stream, so nothing is serialised. They do not say which dispatch a warp belongs to; this script cuts the
sample series into steps at the idle gaps between them (the single-GPU loop waits for the fence after every step),
aligns the steps at their first active sample and averages them. The in-shader records give, for the same build,
the warps resident per dispatch over time (64 invocations per workgroup = 2 warps); their sum is what GPU Metrics'
"Compute Warps in Flight" counts.

    .venv/Scripts/python.exe experiment/wall_crossing/nsys_metrics.py --reports <dir with *.nsys-rep> \
        --labels n500_default,n500_ae,... [--figure n500]

Writes logs/wall_store/round4/nsys/<label>.json (per step: active span; averaged profile of SMs active and warps in
flight) and, with --figure <prefix>, claude/wall_store/figures/round4_nsys_<prefix>.png.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
import subprocess
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = REPOSITORY_ROOT / "logs" / "wall_store" / "round4" / "nsys"
TIMELINE = REPOSITORY_ROOT / "logs" / "wall_store" / "round4" / "timeline"
FIGURES = REPOSITORY_ROOT / "claude" / "wall_store" / "figures"
NSYS = pathlib.Path("C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.5.2/target-windows-x64/nsys.exe")
METRICS = {"sms_active": "SMs Active [Throughput %]", "warps": "Compute Warps in Flight [Avg Warps per Cycle]",
           "warps_percent": "Compute Warps in Flight [Throughput %]", "issue": "SM Issue [Throughput %]",
           "clock": "GPC Clock Frequency [MHz]", "dram_read": "DRAM Read Bandwidth [Throughput %]"}
STREAMING_MULTIPROCESSORS = 170
WARPS_PER_WORKGROUP = 2


def series(report: pathlib.Path) -> dict:
    database = report.with_suffix(".sqlite")
    if not database.exists():
        subprocess.run([str(NSYS), "export", "--type", "sqlite", "--force-overwrite", "true", "-o", str(database),
                        str(report)], check=True, capture_output=True)
    connection = sqlite3.connect(database)
    names = {name: metric_id for metric_id, name in
             connection.execute("select metricId, metricName from TARGET_INFO_GPU_METRICS")}
    result = {}
    for key, name in METRICS.items():
        rows = np.array(connection.execute("select timestamp, value from GPU_METRICS where metricId = ? "
                                           "order by timestamp", (names[name],)).fetchall(), dtype=np.float64)
        result[key] = rows
    connection.close()
    return result


def steps(data: dict, idle_percent: float = 2.0, minimum_samples: int = 20) -> list:
    """(first index, last index) of every active stretch of the SMs-active series (one step each)."""
    active = data["sms_active"][:, 1] > idle_percent
    edges = np.flatnonzero(np.diff(active.astype(np.int8)))
    stretches, start = [], None
    for index in edges:
        if not active[index] and active[index + 1]:
            start = index + 1
        elif active[index] and not active[index + 1] and start is not None:
            if index - start + 1 >= minimum_samples:
                stretches.append((start, index))
            start = None
    return stretches


def profile(data: dict, stretches: list) -> dict:
    """The steps' samples aligned at the first active sample; median per sample index over the steps whose length is
    within 2 samples of the median length."""
    lengths = np.array([last - first + 1 for first, last in stretches])
    typical = int(np.median(lengths))
    chosen = [(first, last) for first, last in stretches if abs((last - first + 1) - typical) <= 2]
    period = float(np.median(np.diff(data["sms_active"][:, 0]))) / 1e3
    result = {"steps": len(stretches), "steps_used": len(chosen), "sample_us": period,
              "step_active_us_median": float(np.median(lengths)) * period,
              "step_active_us_quartiles": [float(np.percentile(lengths, 25)) * period,
                                           float(np.percentile(lengths, 75)) * period]}
    for key in ("sms_active", "warps", "issue", "clock", "dram_read"):
        values = data[key][:, 1]
        stacked = np.array([values[first:first + typical] for first, last in chosen
                            if first + typical <= len(values)])
        result[key] = np.median(stacked, axis=0).tolist()
    return result


def clock_profile(label: str, step_us: float = 1.0) -> dict:
    """Warps resident per dispatch from the in-shader records (first sample), per pass, on a 1 us grid."""
    data = np.load(TIMELINE / f"{label}.npz")
    groups = int(data["groups"])
    start, end = data["start"][0].astype(np.int64), data["end"][0].astype(np.int64)
    result = {}
    for kernel, interior_slot, near_slot in (("fused", 0, 1), ("force", 2, 3)):
        spans = []
        for slot, limit in ((interior_slot, groups), (near_slot, start.shape[1])):
            begin, finish = start[slot][:limit], end[slot][:limit]
            written = (begin > 0) & (finish >= begin)
            spans.append((begin[written], finish[written]))
        origin = min(spans[0][0].min(), spans[1][0].min())
        last = max(spans[0][1].max(), spans[1][1].max())
        grid = np.arange(origin, last, step_us * 1e3)
        counts = []
        for begin, finish in spans:
            counts.append(((begin[None, :] <= grid[:, None]) & (finish[None, :] > grid[:, None])).sum(axis=1)
                          * WARPS_PER_WORKGROUP / STREAMING_MULTIPROCESSORS)
        result[kernel] = {"time_us": ((grid - origin) / 1e3).tolist(), "interior": counts[0].tolist(),
                          "near_wall": counts[1].tolist()}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--figure", default="")
    arguments = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for label in arguments.labels.split(","):
        data = series(pathlib.Path(arguments.reports) / f"{label}.nsys-rep")
        stretches = steps(data)
        summary = profile(data, stretches)
        summary["capture_s"] = float(data["sms_active"][-1, 0] - data["sms_active"][0, 0]) / 1e9
        (OUT / f"{label}.json").write_text(json.dumps(summary), encoding="utf-8")
        summaries[label] = summary
        print(f"{label}: {summary['steps']} steps in {summary['capture_s']:.2f} s, active span median "
              f"{summary['step_active_us_median']:.1f} us (quartiles {summary['step_active_us_quartiles'][0]:.1f}-"
              f"{summary['step_active_us_quartiles'][1]:.1f}), clock {np.median(summary['clock']):.0f} MHz")
    if arguments.figure:
        draw(arguments.figure, summaries)
    return 0


def draw(prefix: str, summaries: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(REPOSITORY_ROOT))
    from experiment.wall_crossing.report_figures import save, setup_style
    setup_style()
    labels = [label for label in summaries if label.startswith(prefix)]
    figure, axes = plt.subplots(len(labels), 1, figsize=(12.0, 2.3 * len(labels) + 0.8), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, label in zip(axes, labels):
        summary = summaries[label]
        time = np.arange(len(summary["warps"])) * summary["sample_us"]
        axis.plot(time, np.array(summary["warps"]) / STREAMING_MULTIPROCESSORS, color="#2a78d6", linewidth=1.6,
                  label="GPU Metrics：每个 SM 的计算线程束（中位数）")
        axis.set_ylabel("线程束 / SM")
        axis.set_title(f"{label}：一步的 {summary['steps_used']} 次（共 {summary['steps']} 次）对齐后的中位数，"
                       f"活跃 {summary['step_active_us_median']:.0f} µs", fontsize=10)
        axis.set_ylim(0, 50)
        axis.grid(alpha=0.3)
    axes[-1].set_xlabel("从一步的第一个活跃采样开始的时间 µs（5 µs 一个采样）")
    axes[0].legend(loc="upper right", frameon=False)
    FIGURES.mkdir(parents=True, exist_ok=True)
    save(figure, FIGURES / f"round4_nsys_{prefix}.png")


if __name__ == "__main__":
    sys.exit(main())
