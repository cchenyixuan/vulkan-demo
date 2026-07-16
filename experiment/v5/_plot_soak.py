"""
_plot_soak.py — render the soak run's time-series report figure.

Reads intervals.jsonl + telemetry.csv from a _run_v5_soak.py output directory
and produces soak_report.png: 8 small-multiple panels on a shared elapsed-hours
axis (interval fps, migration, worker memcpy, SM clock, temperature, power,
working set, conservation drift). One y-axis per panel — never dual-axis.

Usage:
    .venv/Scripts/python.exe experiment/v5/_plot_soak.py logs/soak_8m_perdir_20260715
"""

from __future__ import annotations

import csv
import datetime
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reference categorical palette slots 1 & 2 (light mode) — entity mapping is
# fixed across panels: GPU0 / sim_a / worker a→b = blue, GPU1 / sim_b /
# worker b→a = green.
BLUE = "#2a78d6"
GREEN = "#008300"
TEXT_PRIMARY = "#333333"
TEXT_MUTED = "#767676"
GRID = "#e6e6e6"


def load_intervals(out_dir: pathlib.Path) -> list[dict]:
    return [json.loads(line)
            for line in open(out_dir / "intervals.jsonl", encoding="utf-8")]


def load_telemetry(out_dir: pathlib.Path) -> dict[str, dict[str, list]]:
    per_gpu: dict[str, dict[str, list]] = {}
    rows = list(csv.reader(open(out_dir / "telemetry.csv", encoding="utf-8")))
    t0 = None
    for row in rows[1:]:
        if len(row) < 7:
            continue
        try:
            t = datetime.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
        except ValueError:
            continue
        if t0 is None:
            t0 = t
        gpu = row[1].strip()
        if gpu not in ("0", "1"):
            continue
        series = per_gpu.setdefault(gpu, {"h": [], "util": [], "power": [],
                                          "temp": [], "clock": []})
        series["h"].append((t - t0).total_seconds() / 3600.0)
        series["util"].append(float(row[2].replace("%", "").strip()))
        series["power"].append(float(row[3].replace("W", "").strip()))
        series["temp"].append(float(row[4].strip()))
        series["clock"].append(float(row[5].replace("MHz", "").strip()))
    return per_gpu


def rolling_mean(values: list[float], window: int) -> list[float]:
    out, acc = [], 0.0
    for index, value in enumerate(values):
        acc += value
        if index >= window:
            acc -= values[index - window]
        out.append(acc / min(index + 1, window))
    return out


def style_axis(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, loc="left", pad=8)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_MUTED)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    for spine_name, spine in ax.spines.items():
        spine.set_visible(spine_name in ("left", "bottom"))
        spine.set_color(GRID)


def main() -> int:
    out_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                           else "logs/soak_8m_perdir_20260715")
    intervals = load_intervals(out_dir)
    telemetry = load_telemetry(out_dir)
    summary = json.loads((out_dir / "summary.json").read_text())
    meta = json.loads((out_dir / "meta.json").read_text())

    hours = [r["elapsed_s"] / 3600.0 for r in intervals]
    fps = [r["interval_fps"] for r in intervals]
    fps_smooth = rolling_mean(fps, 36)  # ~5 min at 8.3 s/interval

    fig, axes = plt.subplots(4, 2, figsize=(13, 15), dpi=150)
    fig.suptitle(
        f"12 h soak — cavity 8M, 2× RTX 5090, depth-2, {meta['sync_scheme']} scheme\n"
        f"{summary['frames_total']:,} frames · mean {summary['mean_interval_fps']} fps · "
        f"drift {summary['final_drift']} · {meta['started_iso']}",
        fontsize=12, color=TEXT_PRIMARY, y=0.995)

    # 1. interval fps
    ax = axes[0][0]
    ax.plot(hours, fps, color=BLUE, linewidth=0.6, alpha=0.30)
    ax.plot(hours, fps_smooth, color=BLUE, linewidth=1.8)
    ax.annotate(f"peak {max(fps):.1f}", xy=(hours[fps.index(max(fps))], max(fps)),
                fontsize=8, color=TEXT_MUTED, xytext=(6, 4), textcoords="offset points")
    ax.annotate(f"final {fps_smooth[-1]:.1f}", xy=(hours[-1], fps_smooth[-1]),
                fontsize=8, color=TEXT_MUTED, xytext=(-48, -12), textcoords="offset points")
    style_axis(ax, "Interval fps (raw + 5-min mean)", "fps")

    # 2. migration per interval
    ax = axes[0][1]
    ax.plot(hours, rolling_mean([r["a_interval_migration"] for r in intervals], 36),
            color=BLUE, linewidth=1.8, label="sim A")
    ax.plot(hours, rolling_mean([r["b_interval_migration"] for r in intervals], 36),
            color=GREEN, linewidth=1.8, label="sim B")
    ax.legend(fontsize=8, frameon=False, labelcolor=TEXT_PRIMARY)
    style_axis(ax, "Cross-slab migration per 1000 frames (5-min mean)", "particles")

    # 3. worker memcpy p50
    ax = axes[1][0]
    ax.plot(hours, rolling_mean(
        [r["workers"].get("a_to_b", {}).get("copy_us_p50", float("nan"))
         for r in intervals], 36), color=BLUE, linewidth=1.8, label="worker a→b")
    ax.plot(hours, rolling_mean(
        [r["workers"].get("b_to_a", {}).get("copy_us_p50", float("nan"))
         for r in intervals], 36), color=GREEN, linewidth=1.8, label="worker b→a")
    ax.legend(fontsize=8, frameon=False, labelcolor=TEXT_PRIMARY)
    style_axis(ax, "Ghost-worker memcpy p50 (5-min mean)", "µs")

    # 4-6. telemetry: SM clock / temperature / power
    for ax, key, title, unit in (
            (axes[1][1], "clock", "SM clock", "MHz"),
            (axes[2][0], "temp", "GPU temperature", "°C"),
            (axes[2][1], "power", "Board power", "W")):
        for gpu, color in (("0", BLUE), ("1", GREEN)):
            series = telemetry.get(gpu)
            if series:
                ax.plot(series["h"], series[key], color=color, linewidth=1.4,
                        label=f"GPU {gpu}")
        ax.legend(fontsize=8, frameon=False, labelcolor=TEXT_PRIMARY)
        style_axis(ax, title, unit)

    # 7. working set
    ax = axes[3][0]
    ax.plot(hours, [r["working_set_mb"] for r in intervals],
            color=BLUE, linewidth=1.8)
    ax.set_ylim(0, max(r["working_set_mb"] for r in intervals) * 1.3)
    style_axis(ax, "Host process working set (leak check)", "MB")

    # 8. conservation drift
    ax = axes[3][1]
    drifts = [r["drift"] for r in intervals]
    ax.plot(hours, drifts, color=BLUE, linewidth=1.8)
    ax.set_ylim(-5, 5)
    nonzero = sum(1 for d in drifts if d != 0)
    ax.annotate(f"{len(drifts):,} intervals, {nonzero} non-zero",
                xy=(0.5, 0.75), xycoords="axes fraction", ha="center",
                fontsize=9, color=TEXT_MUTED)
    style_axis(ax, "Conservation drift (alive − expected)", "particles")

    for row in axes:
        for ax in row:
            ax.set_xlim(0, max(hours))
    for ax in axes[3]:
        ax.set_xlabel("elapsed hours", fontsize=9, color=TEXT_MUTED)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = out_dir / "soak_report.png"
    fig.savefig(out_path, facecolor="white")
    print(f"[plot_soak] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
