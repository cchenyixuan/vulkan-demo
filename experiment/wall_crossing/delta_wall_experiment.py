"""
delta_wall_experiment.py - tables and figure of the mechanism test "does the mean density of wall_boundary
crossing drift because the walls' share of the density diffusion (delta term) has no rho0 reference?".

    .venv/Scripts/python.exe -m experiment.wall_crossing.delta_wall_experiment

The runs (box_run.py, 250^2 cavity, from rest, a sample at every defrag boundary = every 1000 steps = 0.03 time
units, the fluid's mean density and mean pressure as float64 means over the alive fluid particles):
    dw_n250_crossing_extension      crossing as specified (numerics.wall_crossing.delta_wall extension)
    dw_n250_crossing_rest_density   the diagnostic: the wall neighbours' diffusion takes rho0 - rho_i pair by pair
    dw_n250_simple, dw_n250_adami   the two existing wall modes over the same window
    dw_n250_crossing_wall_velocity_rest_density   the diagnostic with continuity wall_velocity, to t = 15
This script reads their series.csv and writes
    claude/wall_crossing/tables/delta_wall.md            drift rate (linear fit of the mean density over
                                                         1 <= t <= 5), mean pressure and share of fluid particles
                                                         with negative pressure at the last sample (t = 5.01)
    claude/wall_crossing/tables/delta_wall_side_run.md   the wall_velocity run
    claude/wall_crossing/tables/stopped_runs.md          the runs stopped at the user's request (runs.jsonl)
    claude/wall_crossing/delta_wall/summary.json
    claude/wall_crossing/figures/delta_wall.png          mean density, mean pressure, kinetic energy against time
"""

from __future__ import annotations

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

from experiment.wall_crossing.report_figures import GRID, INK_PRIMARY, INK_SECONDARY, SURFACE, setup_style   # noqa: E402
from experiment.wall_crossing.report_tables import markdown_table, run_records                               # noqa: E402
from experiment.wall_crossing.run_log import LOG_ROOT                                                        # noqa: E402

REPORT_ROOT = REPOSITORY_ROOT / "claude" / "wall_crossing"
REST_DENSITY = 1000.0           # stick_water of materials/standard.yaml
DYNAMIC_PRESSURE = 1000.0       # rho0 U^2 with the lid speed U = 1
FIT_WINDOW = (1.0, 5.011)
# the diagnostic variant first: red, blue, aqua, yellow is the order of this set that passes every palette check
RUNS = (("dw_n250_crossing_rest_density", "crossing，delta_wall = rest_density（诊断）", "#e34948", (0, (5, 2))),
        ("dw_n250_crossing_extension", "crossing，delta_wall = extension（现状）", "#2a78d6", "solid"),
        ("dw_n250_simple", "simple", "#1baf7a", "solid"),
        ("dw_n250_adami", "adami", "#eda100", "solid"))
TABLE_ORDER = ("dw_n250_crossing_extension", "dw_n250_crossing_rest_density", "dw_n250_simple", "dw_n250_adami")
SIDE_RUN = "dw_n250_crossing_wall_velocity_rest_density"


def load_series(name: str) -> dict:
    with open(LOG_ROOT / "box_run" / name / "series.csv", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def measure(series: dict, window: tuple) -> dict:
    time = series["time"]
    inside = (time >= window[0]) & (time <= window[1])
    slope, intercept = np.polyfit(time[inside], series["density_mean"][inside], 1)
    residual = series["density_mean"][inside] - (slope * time[inside] + intercept)
    last = int(np.argmin(np.abs(time - window[1])))
    return {
        "samples": int(len(time)), "samples_in_fit": int(inside.sum()), "fit_window": [float(time[inside][0]),
                                                                                       float(time[inside][-1])],
        "drift_rate": float(slope), "fit_residual_root_mean_square": float(residual.std()),
        "time_at_reading": float(time[last]),
        "density_mean_minus_rest": float(series["density_mean"][last] - REST_DENSITY),
        "pressure_mean": float(series["pressure_mean"][last]),
        "pressure_mean_over_dynamic": float(series["pressure_mean"][last] / DYNAMIC_PRESSURE),
        "pressure_negative_fraction": float(series["pressure_negative_fraction"][last]),
        "kinetic_energy": float(series["kinetic_energy"][last]),
        "density_deviation_maximum": float(series["density_deviation_maximum"].max()),
        "speed_maximum": float(series["speed_maximum"].max()),
        "wall_distance_minimum_over_spacing": float(series["wall_distance_minimum_over_spacing"].min()),
        "outside_box_count_maximum": int(series["outside_box_count"].max()),
        "gradient_fallbacks": int(series["gradient_fallbacks"][-1]),
        "alive_last": int(series["alive"][-1]), "time_last": float(time[-1])}


def main() -> int:
    records = {}
    for record in run_records():
        if record.get("kind") == "box_run" and record.get("name"):
            records[record["name"]] = record
    measured = {name: measure(load_series(name), FIT_WINDOW) for name in TABLE_ORDER}
    text = {name: label for name, label, _, _ in RUNS}
    notes = {
        "dw_n250_crossing_extension": "基准；任务文件第 5 节的写法",
        "dw_n250_crossing_rest_density": "壁邻居的 δ 项逐对取 ρ0 − ρ_i，去掉循环后的 G_i·(L_i × 壁向量) 一项；其余不变",
        "dw_n250_simple": "对照", "dw_n250_adami": "对照"}
    rows = []
    for name in TABLE_ORDER:
        entry = measured[name]
        rows.append([
            text[name], f"{entry['drift_rate']:+.5f}",
            f"{entry['pressure_mean']:+.1f} Pa（{entry['pressure_mean_over_dynamic']:+.3f} ρ0U²）",
            f"{100.0 * entry['pressure_negative_fraction']:.2f} %",
            f"{entry['density_mean_minus_rest']:+.5f}", f"{entry['kinetic_energy']:.3f}", notes[name],
            f"{records[name]['run_id']}（{records[name]['commit']}）"])
    table = ("250² 方腔，从静止开始到 t = 5.01（167 000 步）；每个 defrag 边界采样一次（每 1000 步 = 0.03 个时间单位，"
             "共 168 个样本，含 t = 0）。平均密度和平均压力是对存活的流体粒子（不含壁）在主机端用 float64 求的平均；"
             "密度按存储值加存储偏移换算成真实密度（本配置下偏移为 0，存的就是 ρ）。漂移率 = 平均密度对 "
             f"t ∈ [{measured[TABLE_ORDER[0]]['fit_window'][0]:.2f}, {measured[TABLE_ORDER[0]]['fit_window'][1]:.2f}] "
             f"的线性拟合斜率（{measured[TABLE_ORDER[0]]['samples_in_fit']} 个样本）。压力、负压比例、动能取最后一个样本"
             f"（t = {measured[TABLE_ORDER[0]]['time_at_reading']:.2f}）。\n\n"
             + markdown_table(["运行", "漂移率（kg/m³ 每时间单位）", "t = 5 的平均压力", "负压粒子比例",
                               "t = 5 的平均密度 − ρ0（kg/m³）", "t = 5 的动能", "备注", "记录（提交）"], rows))
    (REPORT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "tables" / "delta_wall.md").write_text(table, encoding="utf-8", newline="\n")

    side_series = load_series(SIDE_RUN)
    side = measure(side_series, FIT_WINDOW)
    side_long = measure(side_series, (1.0, float(side_series["time"][-1]) + 1e-9))
    side_rows = [
        ["到达的时刻", f"t = {side['time_last']:.2f}（{int(side_series['step'][-1]):,} 步），粒子数 {side['alive_last']:,} / "
                  f"{int(side_series['alive'][0]):,}"],
        ["最大密度偏差（整个运行）", f"{side_long['density_deviation_maximum']:.3f} kg/m³"],
        ["越出盒的流体粒子（各样本的最大值）", str(side_long["outside_box_count_maximum"])],
        ["φ 的最小值", f"{side_long['wall_distance_minimum_over_spacing']:.3f} dx"],
        ["A_f 回退（累计）", str(side_long["gradient_fallbacks"])],
        ["最大速度（整个运行，盖速 = 1）", f"{side_long['speed_maximum']:.3f}"],
        ["漂移率", f"t ∈ [1, 5]：{side['drift_rate']:+.5f}；t ∈ [1, 15]：{side_long['drift_rate']:+.5f} kg/m³ 每时间单位"],
        ["平均压力", f"t = 5：{side['pressure_mean']:+.1f} Pa；t = 15：{side_long['pressure_mean']:+.1f} Pa"],
        ["负压粒子比例", f"t = 5：{100.0 * side['pressure_negative_fraction']:.2f} %；t = 15："
                   f"{100.0 * side_long['pressure_negative_fraction']:.2f} %"],
        ["记录（提交）", f"{records[SIDE_RUN]['run_id']}（{records[SIDE_RUN]['commit']}），状态 {records[SIDE_RUN]['status']}"]]
    (REPORT_ROOT / "tables" / "delta_wall_side_run.md").write_text(
        markdown_table(["量", "crossing，continuity = wall_velocity，delta_wall = rest_density，250²"], side_rows),
        encoding="utf-8", newline="\n")

    stopped = [record for record in run_records()
               if str(record.get("status", "")).startswith(("stopped:", "not started:"))]
    stop_rows = []
    for record in stopped:
        progress = record.get("progress", {})
        kept = []
        if progress.get("samples_kept"):
            kept.append("样本、快照、日志")
        if "last_checkpoint" in progress:
            kept.append(f"最后的检查点 {progress['last_checkpoint']}" if progress["last_checkpoint"]
                        else "还没有检查点（每 10 个时间单位一个）")
        if progress.get("kept"):
            kept.append(str(progress["kept"]))
        stop_rows.append([
            record.get("name", "?"), record["machine"],
            "运行中被停止" if str(record["status"]).startswith("stopped:") else "排队中，未启动",
            "—" if not progress.get("step") else
            f"t = {progress['time']:.2f} / {progress['target_time']:g}（{int(progress['step']):,} 步）",
            "；".join(kept) or "—", record["run_id"]])
    if stop_rows:
        (REPORT_ROOT / "tables" / "stopped_runs.md").write_text(
            markdown_table(["运行", "机器", "状态", "停止时的进度", "保留的数据", "记录"], stop_rows),
            encoding="utf-8", newline="\n")

    summary = {"fit_window": list(FIT_WINDOW), "runs": measured, "side_run": {"to_t5": side, "whole_run": side_long},
               "run_ids": {name: records[name]["run_id"] for name in (*TABLE_ORDER, SIDE_RUN)},
               "stopped_runs": [record["run_id"] for record in stopped]}
    (REPORT_ROOT / "delta_wall").mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "delta_wall" / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8",
                                                             newline="\n")

    # ---- figure: three quantities, one axis each
    setup_style()
    figure, axes = plt.subplots(1, 3, figsize=(13.6, 4.7))
    panels = (("density_mean", lambda values: values - REST_DENSITY, "流体平均密度 − ρ0（kg/m³）", "{:+.4f}"),
              ("pressure_mean", lambda values: values / DYNAMIC_PRESSURE, "流体平均压力 / (ρ0 U²)", "{:+.3f}"),
              ("kinetic_energy", lambda values: values, "流体动能（求解器单位）", "{:.2f}"))
    handles, labels = [], []
    for axis, (key, transform, title, value_format) in zip(axes, panels):
        ends = []
        for name, label, colour, style in RUNS:
            series = load_series(name)
            values = transform(series[key])
            line, = axis.plot(series["time"], values, color=colour, linestyle=style, zorder=3)
            ends.append((float(values[-1]), value_format.format(values[-1]), colour))
            if axis is axes[0]:
                handles.append(line)
                labels.append(label)
        # value at the end of every line, spread so that the labels of lines that end together do not overlap
        low, high = axis.get_ylim()
        gap = 0.055 * (high - low)
        placed = []
        for value, text_value, colour in sorted(ends):
            position = value if not placed or value - placed[-1] >= gap else placed[-1] + gap
            placed.append(position)
            axis.annotate(text_value, (5.01, value), xytext=(5.12, position), textcoords="data", fontsize=8,
                          color=INK_SECONDARY, va="center", annotation_clip=False)
            axis.plot([5.03, 5.1], [value, position], color=colour, linewidth=0.8, clip_on=False, zorder=2)
        axis.axhline(0.0, color=INK_SECONDARY, linewidth=0.7, zorder=2) if key != "kinetic_energy" else None
        axis.set_xlim(0, 5.01)
        axis.set_title(title)
        axis.set_xlabel("时间（模拟单位）")
    height = figure.get_figheight()
    figure.suptitle("250² 方腔 t = 0–5：壁邻居的 δ 项有没有 ρ0 参考，对密度水平的影响", x=0.01, y=1.0 - 0.05 / height,
                    ha="left", va="top", fontsize=12, fontweight="bold", color=INK_PRIMARY)
    figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0 - 0.40 / height), ncol=4,
                  handlelength=2.6, columnspacing=1.6, labelcolor=INK_PRIMARY, borderaxespad=0.0)
    figure.tight_layout(rect=(0, 0, 0.985, 1.0 - 0.60 / height), w_pad=3.2)
    (REPORT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
    figure.savefig(REPORT_ROOT / "figures" / "delta_wall.png")
    plt.close(figure)
    print(table)
    print((REPORT_ROOT / "tables" / "delta_wall_side_run.md").read_text(encoding="utf-8"))
    print((REPORT_ROOT / "tables" / "stopped_runs.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
