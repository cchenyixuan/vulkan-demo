"""
delta_wall_experiment.py - tables and figure of the mechanism test "does the mean density of wall_boundary
crossing drift because the walls' share of the density diffusion (delta term) has no rho0 reference?".

    .venv/Scripts/python.exe -m experiment.wall_crossing.delta_wall_experiment

The runs (box_run.py, 250^2 cavity, from rest, a sample at every defrag boundary = every 1000 steps = 0.03 time
units, the fluid's mean density and mean pressure as float64 means over the alive fluid particles):
    dw_n250_crossing_extension      crossing as specified (numerics.wall_crossing.delta_wall extension)
    nd_n250_crossing_r1, _r2        crossing with the default of 2026-11-18 (continuity_wall_component
                                    normal), two runs; every other crossing run of this list was made
                                    with full, the default until then
    dw_n250_crossing_rest_density   the diagnostic: the wall neighbours' diffusion takes rho0 - rho_i pair by pair
    dw_n250_crossing_body_force     the second diagnostic: the pair term rho_i g . (r_j - r_i) / c^2, which is
                                    zero in the cavity (no gravity): the walls contribute nothing to the diffusion
    dw_n250_crossing_continuity_normal   a control: the wall term of the continuity equation with the normal
                                    component of the velocity difference only (continuity_wall_component)
    dw_n250_crossing_no_pst         a control: no particle shifting (fails before the fit window; the table
                                    says what it reached)
    dw_n250_simple, dw_n250_adami   the two existing wall modes over the same window
    dw_n250_crossing_wall_velocity_rest_density   the diagnostic with continuity wall_velocity, to t = 15
and the 2-D hydrostatic boxes (hydrostatic.py, gravity along -y and 30 degrees off) with delta_wall extension
(the runs of the main series) and body_force.
This script reads their series.csv / summary.json and writes
    claude/wall_crossing/tables/delta_wall.md            drift rate (linear fit of the mean density over
                                                         1 <= t <= 5), mean pressure and share of fluid particles
                                                         with negative pressure at the last sample (t = 5.01)
    claude/wall_crossing/tables/delta_wall_side_run.md   the wall_velocity run
    claude/wall_crossing/tables/delta_wall_parts.md      the drift rate against the parts of the delta term that act
                                                         (with the stage-1 run without any density diffusion)
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
        ("dw_n250_crossing_extension", "crossing，delta_wall = extension（连续性壁项 full，旧缺省）", "#2a78d6", "solid"),
        ("dw_n250_crossing_body_force", "crossing，delta_wall = body_force（诊断）", "#008300", (0, (7, 2, 1.5, 2))),
        ("dw_n250_adami", "adami", "#eda100", "solid"),
        ("dw_n250_simple", "simple", "#1baf7a", "solid"))
# the runs of the new default (continuity_wall_component normal, 2026-11-18): rows of the table when they exist
NEW_DEFAULT_RUNS = ("nd_n250_crossing_r1", "nd_n250_crossing_r2")
TABLE_ORDER = ("dw_n250_crossing_extension", *NEW_DEFAULT_RUNS, "dw_n250_crossing_rest_density",
               "dw_n250_crossing_body_force", "dw_n250_crossing_continuity_normal", "dw_n250_simple", "dw_n250_adami")
# runs of the table that are not lines of the figure: name -> text
TABLE_ONLY_TEXT = {"dw_n250_crossing_continuity_normal": "crossing，连续性的壁项只取法向分量（改缺省之前用开关跑的）",
                   "nd_n250_crossing_r1": "**crossing，新缺省（连续性壁项 normal），第 1 次**",
                   "nd_n250_crossing_r2": "**crossing，新缺省（连续性壁项 normal），第 2 次**"}
# a run that failed before the fit window: reported with what it reached (after the row of INSERT_AFTER)
FAILED_RUN = "dw_n250_crossing_no_pst"
FAILED_RUN_TEXT = "crossing，关掉粒子位移修正（`use_pst: false`）"
FAILED_RUN_AFTER = "dw_n250_crossing_continuity_normal"
SIDE_RUN = "dw_n250_crossing_wall_velocity_rest_density"
CONTEXT_RUN = "diag_n250_crossing_no_diffusion"     # stage 1: crossing without any density diffusion, to t = 6
# the 2-D hydrostatic boxes: direction -> (its text, the runs: (name, column text, form index))
# form 0 = extension, 1 = body_force. The "_history" runs carry the mean density and the pressure level at every
# sample (hydrostatic.py as of 21b700a); the other two are the main series' extension run and the first
# body_force run, so each form has two runs and their difference shows what one run is worth.
HYDROSTATIC_RUNS = {
    "gy": ("重力沿 −y", (
        ("2d_n250_gy_crossing", "extension，此前的运行", 0),
        ("2d_n250_gy_crossing_history", "extension，重复（带历史）", 0),
        ("2d_n250_gy_crossing_delta_wall_body_force", "body_force，第 1 次", 1),
        ("2d_n250_gy_crossing_delta_wall_body_force_history", "body_force，第 2 次（带历史）", 1))),
    "g30": ("重力偏离 −y 30°", (
        ("2d_n250_g30_crossing", "extension，此前的运行", 0),
        ("2d_n250_g30_crossing_history", "extension，重复（带历史）", 0),
        ("2d_n250_g30_crossing_delta_wall_body_force", "body_force，第 1 次", 1),
        ("2d_n250_g30_crossing_delta_wall_body_force_history", "body_force，第 2 次（带历史）", 1)))}
HYDROSTATIC_FORMS = (("extension（连续性壁项 full）", "#2a78d6", "solid"), ("body_force（诊断）", "#008300", (0, (7, 2, 1.5, 2))))
HYDROSTATIC_INTRODUCTION = (
    "2-D 静水盒（250² 方腔的几何，盖静止，|g| = 10，c0 = 100，从均匀密度、零速度开始，到 t = 10.02）。速度以 "
    "√(|g| H) 为单位，压力以 ρ0 |g| H 为单位（H = 盒沿重力方向的尺寸）。“去电平” = 减去全场平均偏差后的误差；"
    "“电平” = 压力减去 ρ0 g·(r − r_中心) 之后对全部流体粒子的平均。每种写法各有两次运行：两次之间的差别"
    "就是单次运行的可重复程度。带历史的两次在每个样本都记了平均密度和电平（工具在 21b700a 加的列）。"
    "这张表的四次运行都是连续性壁项 full（2026-11-18 改缺省之前）。")
HYDROSTATIC_LEVEL_TIMES = (2.5, 5.0, 7.5, 10.0)
HYDROSTATIC_DRIFT_WINDOW = (3.0, 10.03)     # after the settling (the level-free pressure error is flat from t = 2)


def load_hydrostatic(name: str) -> tuple:
    """(summary.json, series.csv as arrays) of one hydrostatic run."""
    directory = LOG_ROOT / "hydrostatic" / name
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with open(directory / "series.csv", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return summary, {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def hydrostatic_comparison(records: dict, runs_by_direction: dict = None, forms: tuple = None,
                           introduction: str = None, table_name: str = "delta_wall_hydrostatic.md",
                           figure_name: str = "delta_wall_hydrostatic.png",
                           figure_title: str = "2-D 静水盒：壁邻居 δ 项的两种写法（压力以 ρ0 g H 为单位）") -> dict:
    """The tables (one per gravity direction) and the figure of two forms of the mode in the two 2-D
    hydrostatic boxes, two runs of each form (default: extension against body_force). records: runs.jsonl by
    run id. runs_by_direction: direction -> (its text, ((run name, column text, form index), ...)); forms:
    ((text, colour, line style), ...)."""
    runs_by_direction = HYDROSTATIC_RUNS if runs_by_direction is None else runs_by_direction
    forms = HYDROSTATIC_FORMS if forms is None else forms
    summary_out = {}
    loaded = {}
    texts = [HYDROSTATIC_INTRODUCTION if introduction is None else introduction]
    for direction, (direction_text, runs) in runs_by_direction.items():
        columns = ["量"]
        order = []
        table_rows: dict = {}

        def put(label: str, value: str) -> None:
            if label not in table_rows:
                table_rows[label] = []
                order.append(label)
            table_rows[label].append(value)

        loaded[direction] = []
        summary_out[direction] = {}
        for name, column_text, form in runs:
            summary, series = load_hydrostatic(name)
            loaded[direction].append((name, form, summary, series))
            columns.append(column_text)
            final = summary["final"]
            scale = summary["scales"]["velocity_sqrt_g_extent"]
            pressure_scale = summary["scales"]["pressure_rho_g_extent"]
            layers = {tuple(layer["wall_distance_over_spacing"]): layer for layer in final["layers"]}
            time = series["time"]
            tail = time >= 0.8 * time[-1]
            has_history = "pressure_level_at_centre" in series
            put("到达的时刻", f"t = {summary['time']:.2f}" + ("，失败" if summary.get("failed") else "，粒子数守恒"))
            put("速度 rms / √(gH)，末 1/5 段平均", f"{series['speed_root_mean_square'][tail].mean() / scale:.2e}")
            put("速度最大 / √(gH)，末 1/5 段的最大", f"{series['speed_maximum'][tail].max() / scale:.2e}")
            for key, layer_text in (((0.0, 1.0), "近壁第 1 层（φ < dx）"), ((1.0, 2.0), "第 2 层"),
                                    ((2.0, 3.0), "第 3 层"), ((5.0, None), "内部（φ ≥ 5 dx）")):
                layer = layers.get(key)
                put(f"速度 rms / √(gH)，{layer_text}，末态",
                    "—" if layer is None else f"{layer['speed_root_mean_square'] / scale:.2e}")
            put("压力 rms 误差（去电平），末态", f"{100 * final['pressure_error_root_mean_square_over_scale']:.4f} %")
            put("压力最大误差（去电平），末态", f"{100 * final['pressure_error_maximum_over_scale']:.3f} %")
            put("拟合压力梯度的误差 / ρ0 g，末态", f"{100 * final['fitted_gradient_error_over_rho_g']:.4f} %")
            put("盒中心的压力电平，末态", f"{100 * final['pressure_level_over_scale']:+.3f} %")
            put("压力 rms 误差（不去电平），末态", f"{100 * final['pressure_raw_error_root_mean_square_over_scale']:.3f} %")
            entry = {
                "run_id": summary["run_id"], "time": summary["time"],
                "speed_root_mean_square_over_scale_last_fifth": float(series["speed_root_mean_square"][tail].mean() / scale),
                "speed_maximum_over_scale_last_fifth": float(series["speed_maximum"][tail].max() / scale),
                "pressure_error_root_mean_square_over_scale": final["pressure_error_root_mean_square_over_scale"],
                "pressure_error_maximum_over_scale": final["pressure_error_maximum_over_scale"],
                "pressure_raw_error_root_mean_square_over_scale": final["pressure_raw_error_root_mean_square_over_scale"],
                "pressure_level_over_scale": final["pressure_level_over_scale"],
                "fitted_gradient_error_over_rho_g": final["fitted_gradient_error_over_rho_g"],
                "density_deviation_maximum": summary["whole_run"]["density_deviation_maximum"]}
            if has_history:
                level = series["pressure_level_at_centre"] / pressure_scale
                at_times = [float(level[int(np.argmin(np.abs(time - instant)))]) for instant in HYDROSTATIC_LEVEL_TIMES]
                inside = (time >= HYDROSTATIC_DRIFT_WINDOW[0]) & (time <= HYDROSTATIC_DRIFT_WINDOW[1])
                density_slope = float(np.polyfit(time[inside], series["density_mean"][inside], 1)[0])
                level_slope = float(np.polyfit(time[inside], level[inside], 1)[0])
                put("电平在 t = " + "、".join(f"{instant:g}" for instant in HYDROSTATIC_LEVEL_TIMES),
                    " / ".join(f"{100 * value:+.3f} %" for value in at_times))
                put(f"电平的变化率，t ∈ [{HYDROSTATIC_DRIFT_WINDOW[0]:g}, 10]（每时间单位）", f"{100 * level_slope:+.4f} %")
                put(f"平均密度的漂移率，t ∈ [{HYDROSTATIC_DRIFT_WINDOW[0]:g}, 10]（kg/m³ 每时间单位）", f"{density_slope:+.5f}")
                put("平均密度 − ρ0，末态（kg/m³）", f"{series['density_mean'][-1] - REST_DENSITY:+.5f}")
                entry.update({"pressure_level_over_scale_at_times": dict(zip(map(str, HYDROSTATIC_LEVEL_TIMES), at_times)),
                              "pressure_level_rate_over_scale": level_slope, "density_mean_drift_rate": density_slope,
                              "density_mean_minus_rest_last": float(series["density_mean"][-1] - REST_DENSITY)})
            else:
                for label in ("电平在 t = " + "、".join(f"{instant:g}" for instant in HYDROSTATIC_LEVEL_TIMES),
                              f"电平的变化率，t ∈ [{HYDROSTATIC_DRIFT_WINDOW[0]:g}, 10]（每时间单位）",
                              f"平均密度的漂移率，t ∈ [{HYDROSTATIC_DRIFT_WINDOW[0]:g}, 10]（kg/m³ 每时间单位）",
                              "平均密度 − ρ0，末态（kg/m³）"):
                    put(label, "—（没有记）")
            # every run has the density range: the midpoint of the range, as a pressure through c0^2, is a rough
            # stand-in for the level (the extremes are single particles; it sits up to ~0.3 points off the level)
            midpoint_level = ((0.5 * (series["density_minimum"] + series["density_maximum"]) - REST_DENSITY)
                              * summary["scales"]["speed_of_sound"] ** 2 / pressure_scale)
            put("由密度范围的中点估计的电平（粗略），t = " + "、".join(f"{instant:g}" for instant in HYDROSTATIC_LEVEL_TIMES),
                " / ".join(f"{100 * midpoint_level[int(np.argmin(np.abs(time - instant)))]:+.2f} %"
                           for instant in HYDROSTATIC_LEVEL_TIMES))
            put("密度范围 − ρ0，末态（kg/m³）",
                f"{final['density_minimum'] - REST_DENSITY:+.3f} … {final['density_maximum'] - REST_DENSITY:+.3f}")
            put("φ 的最小值（整个运行）", f"{summary['whole_run']['wall_distance_minimum_over_spacing']:.3f} dx")
            put("A_f 回退（累计）", str(summary["whole_run"]["gradient_fallbacks"]))
            record = records.get(summary["run_id"], {})
            put("记录（提交）", f"{summary['run_id']}（{record.get('commit', '?')}）")
            summary_out[direction][name] = entry
        texts.append(f"**{direction_text}**\n\n"
                     + markdown_table(columns, [[label] + table_rows[label] for label in order]))
    text = "\n\n".join(texts)
    (REPORT_ROOT / "tables" / table_name).write_text(text + "\n", encoding="utf-8", newline="\n")

    # ---- figure: one row per direction; the pressure level, the rms speed and the level-free pressure error
    # against time (one axis each). The level and the pressure error exist for the history runs only; the rms
    # speed is drawn for both runs of each form (the other run thinner), and the other run's final level is a
    # marker on the level panel.
    from matplotlib.lines import Line2D

    setup_style()
    figure, axes = plt.subplots(2, 3, figsize=(13.6, 7.8), sharex=True)
    handles = {}
    for row, (direction, (direction_text, _)) in enumerate(runs_by_direction.items()):
        level_ends = {}         # form -> [(value, x)] of the line's end and of the other run's marker
        for name, form, summary, series in loaded[direction]:
            form_text, colour, style = forms[form]
            scale = summary["scales"]["velocity_sqrt_g_extent"]
            pressure_scale = summary["scales"]["pressure_rho_g_extent"]
            positive = series["time"] > 0.0
            has_history = "pressure_level_at_centre" in series
            line, = axes[row][1].plot(series["time"][positive], series["speed_root_mean_square"][positive] / scale,
                                      color=colour, linestyle=style, linewidth=2.0 if has_history else 1.0,
                                      alpha=1.0 if has_history else 0.55, zorder=3 if has_history else 2)
            if has_history:
                handles[form_text] = line
                axes[row][0].plot(series["time"], 100.0 * series["pressure_level_at_centre"] / pressure_scale,
                                  color=colour, linestyle=style, zorder=3)
                axes[row][2].plot(series["time"][positive],
                                  100.0 * series["pressure_error_root_mean_square"][positive] / pressure_scale,
                                  color=colour, linestyle=style, zorder=3)
                level_ends.setdefault(form, []).append(
                    (100.0 * float(series["pressure_level_at_centre"][-1]) / pressure_scale, float(series["time"][-1])))
            else:
                # the other run of this form has no level column: its final level (summary.json) as a marker
                final_level = 100.0 * summary["final"]["pressure_level_over_scale"]
                axes[row][0].plot([summary["time"]], [final_level], marker="o", markersize=7, color=colour,
                                  markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none", zorder=4,
                                  clip_on=False)
                level_ends.setdefault(form, []).append((final_level, float(summary["time"])))
        # the values at the right edge, spread so that two that end together do not overlap
        low, high = axes[row][0].get_ylim()
        gap = 0.06 * (high - low)
        placed = []
        for value, instant in sorted(entry for ends in level_ends.values() for entry in ends):
            position = value if not placed or value - placed[-1] >= gap else placed[-1] + gap
            placed.append(position)
            axes[row][0].annotate(f"{value:+.2f}", (instant, value), xytext=(instant + 0.35, position),
                                  textcoords="data", fontsize=8, color=INK_SECONDARY, va="center",
                                  annotation_clip=False)
        axes[row][0].axhline(0.0, color=INK_SECONDARY, linewidth=0.7, zorder=2)
        axes[row][0].set_title(f"{direction_text}：盒中心的压力电平（%）")
        axes[row][1].set_title(f"{direction_text}：速度 rms / √(gH)")
        axes[row][2].set_title(f"{direction_text}：压力 rms 误差，去电平（%）")
        axes[row][1].set_yscale("log")
        axes[row][2].set_yscale("log")
    for axis in axes[1]:
        axis.set_xlabel("时间（模拟单位）")
    height = figure.get_figheight()
    figure.suptitle(figure_title, x=0.01, y=1.0 - 0.05 / height,
                    ha="left", va="top", fontsize=12, fontweight="bold", color=INK_PRIMARY)
    legend_handles = list(handles.values()) + [
        Line2D([], [], color=INK_SECONDARY, linewidth=1.0, alpha=0.55),
        Line2D([], [], color=INK_SECONDARY, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
               linestyle="none")]
    legend_labels = list(handles.keys()) + ["细线：同一写法的另一次运行（只有速度）", "圆点：另一次运行的末态电平"]
    figure.legend(legend_handles, legend_labels, loc="upper left", bbox_to_anchor=(0.01, 1.0 - 0.40 / height),
                  ncol=4, handlelength=3.0, columnspacing=1.6, labelcolor=INK_PRIMARY, borderaxespad=0.0)
    figure.tight_layout(rect=(0, 0, 0.985, 1.0 - 0.66 / height), w_pad=3.6)
    figure.savefig(REPORT_ROOT / "figures" / figure_name)
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
    table_order = [name for name in TABLE_ORDER
                   if name not in NEW_DEFAULT_RUNS or (LOG_ROOT / "box_run" / name / "series.csv").exists()]
    measured = {name: measure(load_series(name), FIT_WINDOW) for name in table_order}
    text = {**{name: label for name, label, _, _ in RUNS}, **TABLE_ONLY_TEXT}
    notes = {
        "dw_n250_crossing_extension": "任务文件第 5 节的写法；连续性壁项取完整的速度差（full），2026-11-18 之前的缺省",
        "nd_n250_crossing_r1": "2026-11-18 起的缺省：连续性壁项只取法向分量，其余同上一行",
        "nd_n250_crossing_r2": "同上，重复",
        "dw_n250_crossing_rest_density": "壁邻居的 δ 项逐对取 ρ0 − ρ_i，去掉循环后的 G_i·(L_i × 壁向量) 一项；其余不变",
        "dw_n250_crossing_body_force": "壁邻居的 δ 项逐对取 ρ_i (g − a_wall)·(r_j − r_i) / c²，去掉循环后的一项；"
                                       "方腔没有重力，所以壁邻居对 δ 项的贡献为零；其余不变",
        "dw_n250_crossing_continuity_normal": "壁邻居的速度差先投影到壁面法向（壁粒子到盒面最近点的方向），再乘 "
                                              "(1 + d_j/φ_i)，只用在连续性方程里；其余不变。与新缺省是同一种设置",
        "dw_n250_simple": "对照", "dw_n250_adami": "对照"}
    rows = []
    for name in table_order:
        entry = measured[name]
        rows.append([
            text[name], f"{entry['drift_rate']:+.5f}",
            f"{entry['pressure_mean']:+.1f} Pa（{entry['pressure_mean_over_dynamic']:+.3f} ρ0U²）",
            f"{100.0 * entry['pressure_negative_fraction']:.2f} %",
            f"{entry['density_mean_minus_rest']:+.5f}", f"{entry['kinetic_energy']:.3f}", notes[name],
            f"{records[name]['run_id']}（{records[name]['commit']}）"])
        if name == FAILED_RUN_AFTER and (LOG_ROOT / "box_run" / FAILED_RUN / "series.csv").exists():
            failed = load_series(FAILED_RUN)
            expected = int(failed["alive"][0])
            intact = failed["alive"] == expected
            last_intact = int(np.flatnonzero(intact)[-1])
            first_after_start = 1 if len(failed["time"]) > 1 else 0
            failed_note = (
                f"不稳定，没有跑到拟合窗口。粒子数守恒的最后一个样本是 t = {failed['time'][last_intact]:.2f}"
                f"（{int(failed['step'][last_intact]):,} 步）：平均密度 − ρ0 = "
                f"{failed['density_mean'][last_intact] - REST_DENSITY:+.4f}，平均压力 "
                f"{failed['pressure_mean'][last_intact]:+.1f} Pa，A_f 回退累计 "
                f"{int(failed['gradient_fallbacks'][last_intact]):,} 次，越出盒的流体粒子 "
                f"{int(failed['outside_box_count'][last_intact])} 个。t = {failed['time'][first_after_start]:.2f} 时"
                f"最小密度已经是 {failed['density_minimum'][first_after_start]:.0f} kg/m³、最大速度 "
                f"{failed['speed_maximum'][first_after_start]:.2f}。下一个样本 t = {failed['time'][-1]:.2f} 时存活粒子 "
                f"{int(failed['alive'][-1]):,} / {expected:,}（壁粒子 {expected - 63001:,} 个都在，流体只剩 "
                f"{int(failed['alive'][-1]) - (expected - 63001):,} 个）")
            rows.append([FAILED_RUN_TEXT, "—", "—", "—", "—", "—", failed_note,
                         f"{records[FAILED_RUN]['run_id']}（{records[FAILED_RUN]['commit']}），状态 "
                         f"{records[FAILED_RUN]['status'].split(':')[0]}"])
    table = ("250² 方腔，从静止开始到 t = 5.01（167 000 步）；每个 defrag 边界采样一次（每 1000 步 = 0.03 个时间单位，"
             "共 168 个样本，含 t = 0）。平均密度和平均压力是对存活的流体粒子（不含壁）在主机端用 float64 求的平均；"
             "密度按存储值加存储偏移换算成真实密度（本配置下偏移为 0，存的就是 ρ）。漂移率 = 平均密度对 "
             f"t ∈ [{measured[TABLE_ORDER[0]]['fit_window'][0]:.2f}, {measured[TABLE_ORDER[0]]['fit_window'][1]:.2f}] "
             f"的线性拟合斜率（{measured[TABLE_ORDER[0]]['samples_in_fit']} 个样本）。压力、负压比例、动能取最后一个样本"
             f"（t = {measured[TABLE_ORDER[0]]['time_at_reading']:.2f}）。除标明“新缺省”的行以外，crossing 各行的连续性"
             "壁项都是 full（2026-11-18 改缺省之前的缺省值）。\n\n"
             + markdown_table(["运行", "漂移率（kg/m³ 每时间单位）", "t = 5 的平均压力", "负压粒子比例",
                               "t = 5 的平均密度 − ρ0（kg/m³）", "t = 5 的动能", "备注", "记录（提交）"], rows))
    (REPORT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "tables" / "delta_wall.md").write_text(table, encoding="utf-8", newline="\n")

    # ---- which parts of the delta term act, against the drift rate. The run without any density diffusion is a
    # diagnostic run of stage 1 (existing data: sampled every 0.21 time units to t = 6, no pressure columns); its
    # drift rate is fitted over the same window here.
    context_series = load_series(CONTEXT_RUN)
    context_inside = (context_series["time"] >= FIT_WINDOW[0]) & (context_series["time"] <= FIT_WINDOW[1])
    context_rate = float(np.polyfit(context_series["time"][context_inside],
                                    context_series["density_mean"][context_inside], 1)[0])
    context_rows = [
        ["无（`use_density_diffusion: false`）", "无", f"{context_rate:+.5f}",
         f"阶段 1 的诊断运行，已有数据；窗口内 {int(context_inside.sum())} 个样本（每 0.21 个时间单位一个）",
         f"{records[CONTEXT_RUN]['run_id']}（{records[CONTEXT_RUN]['commit']}）"],
        ["有", "无（`body_force`，方腔没有重力）", f"{measured['dw_n250_crossing_body_force']['drift_rate']:+.5f}", "本节",
         f"{records['dw_n250_crossing_body_force']['run_id']}（{records['dw_n250_crossing_body_force']['commit']}）"],
        ["有", "延拓 G_i·(r_j − r_i)（`extension`）", f"{measured['dw_n250_crossing_extension']['drift_rate']:+.5f}",
         "第 6 节", f"{records['dw_n250_crossing_extension']['run_id']}（{records['dw_n250_crossing_extension']['commit']}）"],
        ["有", "ρ0 − ρ_i（`rest_density`）", f"{measured['dw_n250_crossing_rest_density']['drift_rate']:+.5f}", "第 6 节",
         f"{records['dw_n250_crossing_rest_density']['run_id']}（{records['dw_n250_crossing_rest_density']['commit']}）"]]
    (REPORT_ROOT / "tables" / "delta_wall_parts.md").write_text(
        markdown_table(["流体邻居的 δ 项", "壁邻居的 δ 项", "漂移率，t ∈ [1, 5]（kg/m³ 每时间单位）", "来源", "记录（提交）"],
                       context_rows), encoding="utf-8", newline="\n")

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
               "run_ids": {name: records[name]["run_id"] for name in (*table_order, SIDE_RUN)},
               "hydrostatic_2d": hydrostatic,
               "without_density_diffusion": {"run": CONTEXT_RUN, "run_id": records[CONTEXT_RUN]["run_id"],
                                             "drift_rate": context_rate, "samples_in_fit": int(context_inside.sum())},
               "stopped_runs": [record["run_id"] for record in stopped]}
    (REPORT_ROOT / "delta_wall").mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "delta_wall" / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8",
                                                             newline="\n")

    # ---- figure: three quantities, one axis each
    setup_style()
    figure, axes = plt.subplots(1, 3, figsize=(13.6, 5.0))
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
    figure.suptitle("250² 方腔 t = 0–5：壁邻居 δ 项的三种写法对密度水平的影响（连续性壁项 full）", x=0.01, y=1.0 - 0.05 / height,
                    ha="left", va="top", fontsize=12, fontweight="bold", color=INK_PRIMARY)
    figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0 - 0.40 / height), ncol=3,
                  handlelength=3.0, columnspacing=1.4, labelcolor=INK_PRIMARY, borderaxespad=0.0)
    figure.tight_layout(rect=(0, 0, 0.985, 1.0 - 0.80 / height), w_pad=3.2)
    (REPORT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
    figure.savefig(REPORT_ROOT / "figures" / "delta_wall.png")
    plt.close(figure)
    print(table)
    print((REPORT_ROOT / "tables" / "delta_wall_side_run.md").read_text(encoding="utf-8"))
    print((REPORT_ROOT / "tables" / "stopped_runs.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
