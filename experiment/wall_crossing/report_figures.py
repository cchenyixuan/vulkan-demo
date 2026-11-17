"""
report_figures.py - the figures of claude/wall_crossing_report.md, drawn from the result files of this work.

    .venv/Scripts/python.exe -m experiment.wall_crossing.report_figures [--out claude/wall_crossing/figures]

Every figure reads files that the run tools wrote (nothing is typed in here) and is skipped when its data is
not there yet:
  hydrostatic_2d.png / hydrostatic_3d.png   logs/wall_crossing/hydrostatic/<name>/series.csv + summary.json
  cavity_profile_errors.png                 claude/wall_crossing/cavity/tables.json (cavity_tables.py)
  cavity_convergence.png                    the same file
  cavity_density_level.png                  the same file (the runs' sample histories)
  kernel_timing.png                         logs/wall_crossing/timing/<set>/summary.json (kernel_timing.py)
  cavity3d.png                              logs/wall_crossing/box_run/cavity3d_1m_<mode>/series.csv + profiles.json

One colour per wall mode in every figure (the mode keeps its colour whichever modes a panel shows); a legend in
every figure with two or more modes; text in the ink colours, never in a series colour.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.run_log import LOG_ROOT     # noqa: E402

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e0"
# categorical slots 1-4 of the validated palette (blue, orange, aqua, yellow), fixed per wall mode
MODES = ("crossing", "crossing_wall_velocity", "simple", "adami")
MODE_COLOUR = {"crossing": "#2a78d6", "crossing_wall_velocity": "#eb6834", "simple": "#1baf7a", "adami": "#eda100"}
MODE_TEXT = {"crossing": "crossing（extrapolate）", "crossing_wall_velocity": "crossing（wall_velocity）",
             "simple": "simple", "adami": "adami"}
MODE_MARKER = {"crossing": "o", "crossing_wall_velocity": "s", "simple": "^", "adami": "D"}


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": ["Microsoft YaHei", "DejaVu Sans"], "font.size": 9.5, "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK_PRIMARY,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "lines.linewidth": 1.8, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.fontsize": 8.5, "text.color": INK_PRIMARY, "figure.dpi": 150,
        "savefig.dpi": 170, "savefig.bbox": "tight"})


MODE_SHORT = {"crossing": "crossing", "crossing_wall_velocity": "wall_velocity", "simple": "simple", "adami": "adami"}
LEGEND_INCHES = 0.40        # top of the legend row, from the top edge
PANELS_INCHES = 0.60        # top of the panels' area (their titles included), from the top edge


def finish(figure, title: str, handles: dict, note: str | None = None) -> None:
    """Title, one legend for the whole figure in the fixed mode order, an optional note at the right of the
    legend row, then the panels below: the three rows are placed in inches so they sit the same in every figure."""
    height = figure.get_figheight()
    figure.suptitle(title, x=0.01, y=1.0 - 0.05 / height, ha="left", va="top", fontsize=12, fontweight="bold",
                    color=INK_PRIMARY)
    ordered = [mode for mode in MODES if mode in handles]
    figure.legend([handles[mode] for mode in ordered], [MODE_TEXT[mode] for mode in ordered], loc="upper left",
                  bbox_to_anchor=(0.01, 1.0 - LEGEND_INCHES / height), ncol=len(ordered), handlelength=2.2,
                  columnspacing=1.6, labelcolor=INK_PRIMARY, borderaxespad=0.0)
    if note:
        figure.text(0.99, 1.0 - (LEGEND_INCHES + 0.03) / height, note, ha="right", va="top", fontsize=8.5,
                    color=INK_SECONDARY)
    figure.tight_layout(rect=(0, 0, 1, 1.0 - PANELS_INCHES / height))


def save(figure, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    print(f"[report_figures] wrote {path.relative_to(REPOSITORY_ROOT)}")


def read_csv(path: pathlib.Path) -> dict:
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(row[key]) if row[key] not in ("", None) else np.nan for row in rows])
            for key in rows[0]} if rows else {}


# =============================================================================
# Hydrostatic boxes
# =============================================================================

def hydrostatic_figure(dimension: int, prefix: str, directions: dict, out: pathlib.Path) -> None:
    root = LOG_ROOT / "hydrostatic"
    runs = {}
    for direction in directions:
        for mode in MODES:
            directory = root / f"{prefix}_{direction}_{mode}"
            if (directory / "summary.json").exists() and (directory / "series.csv").exists():
                runs[(direction, mode)] = (json.loads((directory / "summary.json").read_text(encoding="utf-8")),
                                           read_csv(directory / "series.csv"))
    if not runs:
        return
    figure, axes = plt.subplots(2, len(directions), figsize=(5.4 * len(directions), 6.6), sharex=True, squeeze=False)
    handles = {}
    for column, (direction, direction_text) in enumerate(directions.items()):
        for row, (quantity, quantity_text) in enumerate((
                ("speed_maximum", "最大速度 / √(gH)"),
                ("speed_root_mean_square", "动能 / (½ M g H) = (速度均方根 / √(gH))²"))):
            axis = axes[row, column]
            for mode in MODES:
                if (direction, mode) not in runs:
                    continue
                summary, series = runs[(direction, mode)]
                scale = summary["scales"]["velocity_sqrt_g_extent"]
                values = series[quantity] / scale
                if quantity == "speed_root_mean_square":
                    values = values ** 2
                values = np.maximum(values, 1e-12)
                line, = axis.plot(series["time"], values, color=MODE_COLOUR[mode], zorder=3)
                handles[mode] = line
                if summary.get("failed") or len(series["time"]) == 1:
                    # a run that lost particles ends at that sample: mark where it stopped
                    axis.plot(series["time"][-1], values[-1], marker="X", markersize=9, color=MODE_COLOUR[mode],
                              markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)
                    axis.annotate(f"{MODE_SHORT[mode]}：t = {series['time'][-1]:.2f} 丢失粒子，停止",
                                  (series["time"][-1], values[-1]), xytext=(9, 7), textcoords="offset points",
                                  fontsize=8, color=INK_SECONDARY, va="bottom")
            axis.set_yscale("log")
            axis.set_ylim(top=axis.get_ylim()[1] * 4.0)      # room for the notes above the highest line
            axis.set_title(f"{direction_text}：{quantity_text}" if row == 0 else quantity_text)
            if row == 1:
                axis.set_xlabel("时间（模拟单位）")
    finish(figure, f"{dimension}-D 静水盒：残余运动随时间（越低越好）", handles)
    save(figure, out / f"hydrostatic_{dimension}d.png")


# =============================================================================
# 2-D cavity, Re = 1000
# =============================================================================

def cavity_records(tables_path: pathlib.Path) -> dict:
    if not tables_path.exists():
        return {}
    return json.loads(tables_path.read_text(encoding="utf-8"))["runs"]


def cavity_profile_figure(records: dict, out: pathlib.Path) -> None:
    resolutions = sorted({record["resolution"] for record in records.values()
                          if not record["failed"] and "profiles" in record})
    if not resolutions:
        return
    figure, axes = plt.subplots(len(resolutions), 2, figsize=(10.8, 3.1 * len(resolutions) + 0.9), squeeze=False,
                                sharex=True)
    handles = {}
    for row, resolution in enumerate(resolutions):
        for column, (line_name, title) in enumerate((("u", "u(y) 沿竖直中线"), ("v", "v(x) 沿水平中线"))):
            axis = axes[row, column]
            axis.axhline(0.0, color=INK_SECONDARY, linewidth=0.7, zorder=2)
            for mode in MODES:
                record = records.get(f"n{resolution}_{mode}")
                if record is None or record["failed"] or "profiles" not in record:
                    continue
                profile = record["profiles"][line_name]
                line, = axis.plot(profile["position"], profile["error_wall"], color=MODE_COLOUR[mode],
                                  marker=MODE_MARKER[mode], markersize=4.2, markeredgecolor=SURFACE,
                                  markeredgewidth=0.6, zorder=3)
                handles[mode] = line
                if mode == "crossing":
                    axis.plot(profile["position"], profile["error_mid"], color=MODE_COLOUR[mode], linestyle=(0, (4, 2)),
                              linewidth=1.3, zorder=3)
            shown = [records[f"n{resolution}_{mode}"] for mode in MODES
                     if f"n{resolution}_{mode}" in records and not records[f"n{resolution}_{mode}"]["failed"]]
            state = "" if all(record.get("complete") for record in shown) else "（含未跑完的运行）"
            axis.set_title(f"{resolution}² {title}{state}")
            if row == len(resolutions) - 1:
                axis.set_xlabel("单位方腔内的位置")
            if column == 0:
                axis.set_ylabel("计算值 − Marchi 2021（盖速为 1）")
    finish(figure, "Re = 1000 方腔：中线速度相对 Marchi 2021 的逐点误差（28 点；越靠近 0 越好）", handles,
           note="实线 = wall 帧（各模式同一取法）；蓝色虚线 = crossing 的 mid 帧（盒面）")
    save(figure, out / "cavity_profile_errors.png")


def cavity_convergence_figure(records: dict, out: pathlib.Path) -> None:
    panels = (("ψ_min 偏差（wall 帧）", lambda record: abs(record["extrema"]["psi_min"]["deviation_wall_percent"])
               if record["extrema"].get("psi_min") else None, "%"),
              ("u_min 偏差", lambda record: abs(record["extrema"]["u_min"]["deviation_percent"]), "%"),
              ("v_max 偏差", lambda record: abs(record["extrema"]["v_max"]["deviation_percent"]), "%"),
              ("v_min 偏差", lambda record: abs(record["extrema"]["v_min"]["deviation_percent"]), "%"),
              ("u 剖面 L2 误差（wall 帧）", lambda record: record["errors"]["wall_u"]["L2"], ""),
              ("v 剖面 L2 误差（wall 帧）", lambda record: record["errors"]["wall_v"]["L2"], ""))
    usable = [record for record in records.values() if not record["failed"] and record.get("complete")]
    if len({record["resolution"] for record in usable}) < 2:
        return
    figure, axes = plt.subplots(2, 3, figsize=(11.4, 6.6), sharex=True)
    handles = {}
    for axis, (title, value_of, unit) in zip(axes.flat, panels):
        for mode in MODES:
            series = sorted((record for record in usable if record["name"] == f"n{record['resolution']}_{mode}"),
                            key=lambda record: record["resolution"])
            points = [(record["resolution"], value_of(record)) for record in series]
            points = [(resolution, value) for resolution, value in points if value is not None]
            if not points:
                continue
            line, = axis.plot([point[0] for point in points], [point[1] for point in points],
                              color=MODE_COLOUR[mode], marker=MODE_MARKER[mode], markersize=6.5,
                              markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
            handles[mode] = line
            last_resolution, last_value = points[-1]
            axis.annotate(f"{last_value:.2f} {unit}".strip() if unit else f"{last_value:.4f}",
                          (last_resolution, last_value), xytext=(7, 0), textcoords="offset points", fontsize=8,
                          color=INK_SECONDARY, va="center")
        resolutions = sorted({record["resolution"] for record in usable})
        reference = np.array(resolutions, dtype=float)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(resolutions)
        axis.set_xticklabels([f"{resolution}²" for resolution in resolutions])
        axis.minorticks_off()
        axis.set_xlim(reference[0] / 1.15, reference[-1] * 1.55)
        axis.set_title(title + ("（%）" if unit else ""))
    for axis in axes[1]:
        axis.set_xlabel("分辨率")
    finish(figure, "Re = 1000 方腔：误差随分辨率（|偏差|，双对数；越低越好）", handles)
    save(figure, out / "cavity_convergence.png")


def cavity_density_figure(records: dict, out: pathlib.Path) -> None:
    resolutions = sorted({record["resolution"] for record in records.values() if "history" in record})
    if not resolutions:
        return
    figure, axes = plt.subplots(1, len(resolutions), figsize=(max(4.6 * len(resolutions), 7.6), 4.5), squeeze=False)
    handles = {}
    for axis, resolution in zip(axes[0], resolutions):
        lowest, highest = 0.0, 0.0
        clipped = []
        for mode in MODES:
            record = records.get(f"n{resolution}_{mode}")
            if record is None or "history" not in record:
                continue
            history = record["history"]
            time = np.array(history["time"])
            level = 0.5 * (np.array(history["density_min"]) + np.array(history["density_max"])) - 1000.0
            line, = axis.plot(time, level, color=MODE_COLOUR[mode], zorder=3)
            handles[mode] = line
            if mode == "crossing_wall_velocity":
                # its level falls far below the others' range: the panel keeps their range and says so
                in_first_half = time <= 0.5 * time[-1]
                rate = float(np.polyfit(time[in_first_half], level[in_first_half], 1)[0]) \
                    if in_first_half.sum() >= 3 else float("nan")
                clipped.append((time[-1], float(level.min()), rate, record["failed"]))
            else:
                lowest = min(lowest, float(level.min()))
                highest = max(highest, float(level.max()))
        span = highest - lowest
        axis.set_ylim(lowest - 0.22 * span - 0.02, highest + 0.06 * span + 0.02)
        for time_last, level_lowest, rate, failed in clipped:
            if level_lowest < axis.get_ylim()[0]:
                axis.annotate(f"wall_velocity 超出本图范围：约 {rate:+.2f} kg/m³ 每时间单位"
                              + (f"，t = {time_last:.1f} 丢失粒子" if failed else f"（到 t = {time_last:.1f}）"),
                              (0.03, 0.035), xycoords="axes fraction", fontsize=8, color=INK_SECONDARY)
        axis.set_title(f"{resolution}²")
        axis.set_xlabel("时间（模拟单位）")
    axes[0][0].set_ylabel("(ρ_min + ρ_max) / 2 − ρ₀（kg/m³）")
    finish(figure, "Re = 1000 方腔：流体密度水平随时间（0 = 静止密度）", handles)
    save(figure, out / "cavity_density_level.png")


# =============================================================================
# Per-kernel timing
# =============================================================================

def timing_figure(out: pathlib.Path) -> None:
    sets = []
    for name, title in (("n1000", "2-D 方腔 1000²"), ("cavity3d_1m", "3-D 方腔 1M")):
        path = LOG_ROOT / "timing" / name / "summary.json"
        if path.exists():
            sets.append((title, json.loads(path.read_text(encoding="utf-8"))["labels"]))
    if not sets:
        return
    kernels = (("correction_density_us", "第一遍（修正 + 密度）"), ("force_us", "第二遍（力）"),
               ("wall_extrapolate_us", "壁面遍（仅 adami）"), ("step_total_us", "整步"))
    figure, axes = plt.subplots(len(sets), len(kernels), figsize=(11.6, 2.9 * len(sets) + 1.0), squeeze=False,
                                sharey="row")
    handles = {}
    for row, (title, labels) in enumerate(sets):
        for column, (kernel, kernel_text) in enumerate(kernels):
            axis = axes[row, column]
            positions, values, colours = [], [], []
            for index, mode in enumerate(MODES):
                label = f"branch_{mode}"
                if label not in labels or kernel not in labels[label]["kernels"]:
                    continue
                positions.append(index)
                values.append(labels[label]["kernels"][kernel]["median_us"])
                colours.append(MODE_COLOUR[mode])
                handles[mode] = plt.Rectangle((0, 0), 1, 1, color=MODE_COLOUR[mode])
            bars = axis.bar(positions, values, width=0.44, color=colours, zorder=3)
            for bar, value in zip(bars, values):
                axis.annotate(f"{value:.0f}", (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 3),
                              textcoords="offset points", ha="center", fontsize=8.5, color=INK_PRIMARY)
            axis.set_xticks(range(len(MODES)))
            axis.set_xticklabels([MODE_SHORT[mode].replace("_", "_\n") for mode in MODES], fontsize=7.5)
            axis.tick_params(axis="x", length=0)
            axis.set_xlim(-0.6, 3.6)
            axis.grid(axis="x", visible=False)
            axis.set_title(f"{title}：{kernel_text}" if column == 0 else kernel_text)
            if column == 0:
                axis.set_ylabel("µs / 步")
    for row in range(len(sets)):
        axes[row, 0].set_ylim(0, axes[row, 0].get_ylim()[1] * 1.06)
    finish(figure, "各核 GPU 时间（µs / 步，三轮交替测量的中位数；每行同一纵轴；越低越好）", handles)
    save(figure, out / "kernel_timing.png")


# =============================================================================
# 3-D cavity
# =============================================================================

def cavity3d_figure(out: pathlib.Path) -> None:
    root = LOG_ROOT / "box_run"
    runs = {}
    for mode in MODES:
        directory = root / f"cavity3d_1m_{mode}"
        if (directory / "series.csv").exists():
            profiles = (json.loads((directory / "profiles.json").read_text(encoding="utf-8"))
                        if (directory / "profiles.json").exists() else [])
            runs[mode] = (read_csv(directory / "series.csv"), profiles)
    if not runs:
        return
    common_time = min(profiles[-1]["time"] for _, profiles in runs.values() if profiles) \
        if any(profiles for _, profiles in runs.values()) else None
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    handles = {}
    for mode, (series, profiles) in runs.items():
        line, = axes[0].plot(series["time"], series["kinetic_energy"], color=MODE_COLOUR[mode], zorder=3)
        handles[mode] = line
        if common_time is None or not profiles:
            continue
        chosen = min(profiles, key=lambda profile: abs(profile["time"] - common_time))
        axes[1].plot(chosen["u_along_y"]["velocity"], chosen["u_along_y"]["coordinate"], color=MODE_COLOUR[mode],
                     zorder=3)
        axes[2].plot(chosen["v_along_x"]["coordinate"], chosen["v_along_x"]["velocity"], color=MODE_COLOUR[mode],
                     zorder=3)
    axes[0].set_title("动能随时间")
    axes[0].set_xlabel("时间（模拟单位）")
    axes[0].set_ylabel("流体动能（求解器单位）")
    time_text = "" if common_time is None else f"（t = {common_time:.1f}）"
    axes[1].set_title(f"u 沿过盒心的 y 向中线{time_text}")
    axes[1].set_xlabel("u / 盖速")
    axes[1].set_ylabel("y（盒内归一位置）")
    axes[2].set_title(f"v 沿过盒心的 x 向中线{time_text}")
    axes[2].set_xlabel("x（盒内归一位置）")
    axes[2].set_ylabel("v / 盖速")
    finish(figure, "3-D 顶盖方腔（1M 粒子）：各壁面模式到同一时刻", handles)
    save(figure, out / "cavity3d.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="claude/wall_crossing/figures")
    parser.add_argument("--tables", default="claude/wall_crossing/cavity/tables.json")
    arguments = parser.parse_args()
    out = (REPOSITORY_ROOT / arguments.out).resolve()
    setup_style()
    hydrostatic_figure(2, "2d_n250", {"gy": "重力沿 −y", "g30": "重力偏离 −y 30°"}, out)
    hydrostatic_figure(3, "3d_n48", {"gy": "重力沿 −y", "gtilt": "重力偏离两个轴"}, out)
    records = cavity_records((REPOSITORY_ROOT / arguments.tables).resolve())
    if records:
        cavity_profile_figure(records, out)
        cavity_convergence_figure(records, out)
        cavity_density_figure(records, out)
    timing_figure(out)
    cavity3d_figure(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
