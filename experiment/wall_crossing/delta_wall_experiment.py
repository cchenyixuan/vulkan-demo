"""
delta_wall_experiment.py - tables and figure of the mechanism test "does the mean density of wall_boundary
crossing drift because the walls' share of the density diffusion (delta term) has no rho0 reference?".

    .venv/Scripts/python.exe -m experiment.wall_crossing.delta_wall_experiment

The runs (box_run.py, 250^2 cavity, from rest, a sample at every defrag boundary = every 1000 steps = 0.03 time
units, the fluid's mean density and mean pressure as float64 means over the alive fluid particles):
    dw_n250_crossing_extension      crossing as specified (numerics.wall_crossing.delta_wall extension)
    dw_n250_crossing_rest_density   the diagnostic: the wall neighbours' diffusion takes rho0 - rho_i pair by pair
    dw_n250_crossing_body_force     the second diagnostic: the pair term rho_i g . (r_j - r_i) / c^2, which is
                                    zero in the cavity (no gravity): the walls contribute nothing to the diffusion
    dw_n250_simple, dw_n250_adami   the two existing wall modes over the same window
    dw_n250_crossing_wall_velocity_rest_density   the diagnostic with continuity wall_velocity, to t = 15
and the 2-D hydrostatic boxes (hydrostatic.py, gravity along -y and 30 degrees off) with delta_wall extension
(the runs of the main series) and body_force.
This script reads their series.csv / summary.json and writes
    claude/wall_crossing/tables/delta_wall.md            drift rate (linear fit of the mean density over
                                                         1 <= t <= 5), mean pressure and share of fluid particles
                                                         with negative pressure at the last sample (t = 5.01)
    claude/wall_crossing/tables/delta_wall_side_run.md   the wall_velocity run
    claude/wall_crossing/tables/delta_wall_hydrostatic.md   extension against body_force in the hydrostatic boxes
    claude/wall_crossing/tables/stopped_runs.md          the runs stopped at the user's request (runs.jsonl)
    claude/wall_crossing/delta_wall/summary.json
    claude/wall_crossing/figures/delta_wall.png          mean density, mean pressure, kinetic energy against time
    claude/wall_crossing/figures/delta_wall_hydrostatic.png   residual speed against time, the two forms
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
# red, blue, green, yellow, aqua is the order of this set that passes every palette check (validate_palette of the
# dataviz skill, light surface); the two diagnostic variants are dashed, so a line they cover still shows
RUNS = (("dw_n250_crossing_rest_density", "crossing，delta_wall = rest_density（诊断）", "#e34948", (0, (5, 2))),
        ("dw_n250_crossing_extension", "crossing，delta_wall = extension（现状）", "#2a78d6", "solid"),
        ("dw_n250_crossing_body_force", "crossing，delta_wall = body_force（诊断）", "#008300", (0, (7, 2, 1.5, 2))),
        ("dw_n250_adami", "adami", "#eda100", "solid"),
        ("dw_n250_simple", "simple", "#1baf7a", "solid"))
TABLE_ORDER = ("dw_n250_crossing_extension", "dw_n250_crossing_rest_density", "dw_n250_crossing_body_force",
               "dw_n250_simple", "dw_n250_adami")
SIDE_RUN = "dw_n250_crossing_wall_velocity_rest_density"
# the 2-D hydrostatic boxes: direction -> (its text, the extension run, the body_force run)
HYDROSTATIC_RUNS = {
    "gy": ("重力沿 −y", "2d_n250_gy_crossing", "2d_n250_gy_crossing_delta_wall_body_force"),
    "g30": ("重力偏离 −y 30°", "2d_n250_g30_crossing", "2d_n250_g30_crossing_delta_wall_body_force")}
HYDROSTATIC_FORMS = (("extension（现状）", "#2a78d6", "solid"), ("body_force（诊断）", "#008300", (0, (7, 2, 1.5, 2))))


def load_hydrostatic(name: str) -> tuple:
    """(summary.json, series.csv as arrays) of one hydrostatic run."""
    directory = LOG_ROOT / "hydrostatic" / name
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with open(directory / "series.csv", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return summary, {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def hydrostatic_comparison(records: dict) -> dict:
    """The table and the figure of extension against body_force in the two 2-D hydrostatic boxes. Both runs of a
    direction are read at the same time (the last sample both reached). records: runs.jsonl by run id."""
    columns = ["量"]
    table_rows: dict = {}
    summary_out = {}
    order = []

    def put(label: str, value: str) -> None:
        if label not in table_rows:
            table_rows[label] = []
            order.append(label)
        table_rows[label].append(value)

    loaded = {}
    for direction, (direction_text, extension_name, body_force_name) in HYDROSTATIC_RUNS.items():
        pair = [load_hydrostatic(extension_name), load_hydrostatic(body_force_name)]
        loaded[direction] = pair
        common_time = min(float(series["time"][-1]) for _, series in pair)
        summary_out[direction] = {}
        for (form_text, _, _), name, (summary, series) in zip(HYDROSTATIC_FORMS, (extension_name, body_force_name), pair):
            columns.append(f"{direction_text}，{form_text}")
            final = summary["final"]
            scale = summary["scales"]["velocity_sqrt_g_extent"]
            pressure_scale = summary["scales"]["pressure_rho_g_extent"]
            layers = {tuple(layer["wall_distance_over_spacing"]): layer for layer in final["layers"]}
            at_common = int(np.argmin(np.abs(series["time"] - common_time)))
            tail = series["time"] >= 0.8 * common_time
            tail &= series["time"] <= common_time + 1e-9
            rest_density = REST_DENSITY
            put("到达的时刻", f"t = {summary['time']:.2f}（{summary['steps']:,} 步）"
                + ("，失败" if summary.get("failed") else "，粒子数守恒"))
            put("比较的时刻", f"t = {series['time'][at_common]:.2f}")
            put("速度 rms / √(gH)，该时刻", f"{series['speed_root_mean_square'][at_common] / scale:.2e}")
            put("速度 rms / √(gH)，末 1/5 段平均", f"{series['speed_root_mean_square'][tail].mean() / scale:.2e}")
            put("速度最大 / √(gH)，该时刻", f"{series['speed_maximum'][at_common] / scale:.2e}")
            put("速度最大 / √(gH)，末 1/5 段的最大", f"{series['speed_maximum'][tail].max() / scale:.2e}")
            put("动能，该时刻（求解器单位）", f"{series['kinetic_energy'][at_common]:.3e}")
            put("压力 rms 误差（去电平）/ ρ0 g H，末态", f"{100 * final['pressure_error_root_mean_square_over_scale']:.4f} %")
            put("压力最大误差（去电平）/ ρ0 g H，末态", f"{100 * final['pressure_error_maximum_over_scale']:.3f} %")
            put("压力 rms 误差（不去电平）/ ρ0 g H，末态",
                f"{100 * final['pressure_raw_error_root_mean_square_over_scale']:.4f} %")
            put("盒中心的压力电平 / ρ0 g H，末态", f"{100 * final['pressure_level_over_scale']:+.4f} %")
            put("拟合压力梯度的误差 / ρ0 g，末态", f"{100 * final['fitted_gradient_error_over_rho_g']:.4f} %")
            for key, layer_text in (((0.0, 1.0), "近壁第 1 层（φ < dx）"), ((1.0, 2.0), "第 2 层"),
                                    ((2.0, 3.0), "第 3 层"), ((5.0, None), "内部（φ ≥ 5 dx）")):
                layer = layers.get(key)
                put(f"速度 rms / √(gH)，{layer_text}，末态",
                    "—" if layer is None else f"{layer['speed_root_mean_square'] / scale:.2e}")
                put(f"压力 rms 误差 / ρ0 g H，{layer_text}，末态",
                    "—" if layer is None else f"{100 * layer['pressure_error_root_mean_square'] / pressure_scale:.4f} %")
            put("密度范围 − ρ0，该时刻（kg/m³）",
                f"{series['density_minimum'][at_common] - rest_density:+.3f} … "
                f"{series['density_maximum'][at_common] - rest_density:+.3f}")
            put("φ 的最小值（整个运行）", f"{summary['whole_run']['wall_distance_minimum_over_spacing']:.3f} dx")
            put("A_f 回退（累计）", str(summary["whole_run"]["gradient_fallbacks"]))
            put("步 / 秒；GPU", f"{summary['steps_per_second'] or 0:.0f}；{(summary.get('device_uuid') or '?')[:8]}")
            record = records.get(summary["run_id"], {})
            put("记录（提交）", f"{summary['run_id']}（{record.get('commit', '?')}）")
            summary_out[direction][name] = {
                "run_id": summary["run_id"], "time": summary["time"], "time_compared": float(series["time"][at_common]),
                "speed_root_mean_square_over_scale": float(series["speed_root_mean_square"][at_common] / scale),
                "speed_root_mean_square_over_scale_last_fifth": float(series["speed_root_mean_square"][tail].mean() / scale),
                "speed_maximum_over_scale": float(series["speed_maximum"][at_common] / scale),
                "speed_maximum_over_scale_last_fifth": float(series["speed_maximum"][tail].max() / scale),
                "kinetic_energy": float(series["kinetic_energy"][at_common]),
                "pressure_error_root_mean_square_over_scale": final["pressure_error_root_mean_square_over_scale"],
                "pressure_error_maximum_over_scale": final["pressure_error_maximum_over_scale"],
                "pressure_raw_error_root_mean_square_over_scale": final["pressure_raw_error_root_mean_square_over_scale"],
                "pressure_level_over_scale": final["pressure_level_over_scale"],
                "fitted_gradient_error_over_rho_g": final["fitted_gradient_error_over_rho_g"],
                "density_minimum_minus_rest": float(series["density_minimum"][at_common] - rest_density),
                "density_maximum_minus_rest": float(series["density_maximum"][at_common] - rest_density)}
    text = ("2-D 静水盒（250² 方腔的几何，盖静止，|g| = 10，c0 = 100，从均匀密度、零速度开始）。速度以 √(|g| H) 为单位，"
            "压力以 ρ0 |g| H 为单位（H = 盒沿重力方向的尺寸）。“去电平” = 减去全场平均偏差后的误差；"
            "“该时刻” = 两次运行都到达的最后一个样本。\n\n"
            + markdown_table(columns, [[label] + table_rows[label] for label in order]))
    (REPORT_ROOT / "tables" / "delta_wall_hydrostatic.md").write_text(text, encoding="utf-8", newline="\n")

    # ---- figure: rms and largest speed against time, one panel per direction and quantity (one axis each)
    setup_style()
    figure, axes = plt.subplots(2, 2, figsize=(11.6, 7.4), sharex=True)
    handles = []
    for column, (direction, (direction_text, _, _)) in enumerate(HYDROSTATIC_RUNS.items()):
        for row, (key, quantity_text) in enumerate((("speed_root_mean_square", "速度 rms / √(gH)"),
                                                    ("speed_maximum", "速度最大 / √(gH)"))):
            axis = axes[row][column]
            for (form_text, colour, style), (summary, series) in zip(HYDROSTATIC_FORMS, loaded[direction]):
                scale = summary["scales"]["velocity_sqrt_g_extent"]
                positive = series["time"] > 0.0
                line, = axis.plot(series["time"][positive], series[key][positive] / scale, color=colour,
                                  linestyle=style, zorder=3)
                if row == 0 and column == 0:
                    handles.append((line, form_text))
            axis.set_yscale("log")
            axis.set_title(f"{direction_text}：{quantity_text}")
            if row == 1:
                axis.set_xlabel("时间（模拟单位）")
    height = figure.get_figheight()
    figure.suptitle("2-D 静水盒：壁邻居 δ 项的两种写法，残余速度随时间", x=0.01, y=1.0 - 0.05 / height, ha="left",
                    va="top", fontsize=12, fontweight="bold", color=INK_PRIMARY)
    figure.legend([line for line, _ in handles], [text_form for _, text_form in handles], loc="upper left",
                  bbox_to_anchor=(0.01, 1.0 - 0.40 / height), ncol=2, handlelength=2.8, columnspacing=1.6,
                  labelcolor=INK_PRIMARY, borderaxespad=0.0)
    figure.tight_layout(rect=(0, 0, 1.0, 1.0 - 0.60 / height), w_pad=2.4)
    figure.savefig(REPORT_ROOT / "figures" / "delta_wall_hydrostatic.png")
    plt.close(figure)
    print(text)
    return summary_out


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
    records_by_run_id = {}
    for record in run_records():
        records_by_run_id[record["run_id"]] = record
        if record.get("kind") == "box_run" and record.get("name"):
            records[record["name"]] = record
    measured = {name: measure(load_series(name), FIT_WINDOW) for name in TABLE_ORDER}
    text = {name: label for name, label, _, _ in RUNS}
    notes = {
        "dw_n250_crossing_extension": "基准；任务文件第 5 节的写法",
        "dw_n250_crossing_rest_density": "壁邻居的 δ 项逐对取 ρ0 − ρ_i，去掉循环后的 G_i·(L_i × 壁向量) 一项；其余不变",
        "dw_n250_crossing_body_force": "壁邻居的 δ 项逐对取 ρ_i (g − a_wall)·(r_j − r_i) / c²，去掉循环后的一项；"
                                       "方腔没有重力，所以壁邻居对 δ 项的贡献为零；其余不变",
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

    hydrostatic = hydrostatic_comparison(records_by_run_id)
    summary = {"fit_window": list(FIT_WINDOW), "runs": measured, "side_run": {"to_t5": side, "whole_run": side_long},
               "run_ids": {name: records[name]["run_id"] for name in (*TABLE_ORDER, SIDE_RUN)},
               "hydrostatic_2d": hydrostatic,
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
    figure.suptitle("250² 方腔 t = 0–5：壁邻居 δ 项的三种写法对密度水平的影响", x=0.01, y=1.0 - 0.05 / height,
                    ha="left", va="top", fontsize=12, fontweight="bold", color=INK_PRIMARY)
    figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0 - 0.40 / height), ncol=5,
                  handlelength=3.0, columnspacing=1.4, labelcolor=INK_PRIMARY, borderaxespad=0.0)
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
