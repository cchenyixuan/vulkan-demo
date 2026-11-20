"""
cavity3d_resolution.py - report section 25 B / C: the 3-D cavity (Re = 1000) at 50, 70 and 100 spacings per side
against the benchmark of Albensoeder & Kuhlmann (2005), per wall method (crossing, simple, adami).

    .venv/Scripts/python.exe -m experiment.wall_crossing.cavity3d_resolution [--window 35 45]

Per method and resolution (RUNS; a run that does not exist or has no profile in the window is left out): the
window-mean centre-line profiles, the three extrema's deviations from the benchmark (of the magnitude, + = stronger),
the profile errors L2 / Linf over the 15 inner benchmark points (cavity3d_reference.errors, mid frame), the kinetic
energy's window slope, the stop (a run that stopped before the window has no profile row) and the steps per second.
The convergence order from the three resolutions, error ~ dx^p with dx = 1 / spacings: the two pairwise orders
p = ln(e1 / e2) / ln(dx1 / dx2) and the least-squares slope of ln e on ln dx, only for a quantity whose error keeps
its sign and decreases monotonically with dx (otherwise the table says so and gives no order).
Writes claude/wall_crossing/tables/cavity3d_resolution_*.md, claude/wall_crossing/figures/cavity3d_resolution_*.png
and logs/wall_crossing/cavity3d_resolution/summary.json.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.cavity3d_reference import (REFERENCE_U, REFERENCE_V, errors, read_reference,  # noqa: E402
                                                         read_series, window_mean_profiles)
from experiment.wall_crossing.run_log import LOG_ROOT     # noqa: E402

REPORT_ROOT = REPOSITORY_ROOT / "claude" / "wall_crossing"
REST_DENSITY = 1000.0
SPACINGS = (50, 70, 100)
# method -> {spacings: run name}; the 100 runs are those of sections 19 / 24 and the remote adami of section 25
RUNS = {"crossing": {50: "s25_cavity3d_n50_crossing", 70: "s25_cavity3d_n70_crossing",
                     100: "s24_cavity3d_full_integral"},
        "simple": {50: "s25_cavity3d_n50_simple", 70: "s25_cavity3d_n70_simple", 100: "cavity3d_1m_simple_long"},
        "adami": {50: "s25_cavity3d_n50_adami", 70: "s25_cavity3d_n70_adami", 100: "s25_cavity3d_1m_adami_remote"}}
METHOD_TEXT = {"crossing": "crossing（新缺省 + 总体积修正含积分项）", "simple": "simple", "adami": "adami"}
RESOLUTION_COLOUR = {50: "#2a78d6", 70: "#eb6834", 100: "#1baf7a"}
QUANTITIES = ("u_min", "v_max", "v_min", "l2_u", "l2_v", "linf_u", "linf_v")
QUANTITY_TEXT = {"u_min": "u_min 偏差", "v_max": "v_max 偏差", "v_min": "v_min 偏差", "l2_u": "u(y) L2",
                 "l2_v": "v(x) L2", "linf_u": "u(y) L∞", "linf_v": "v(x) L∞"}


def order(values: dict) -> dict:
    """Pairwise and least-squares orders of |error| ~ dx^p over the available resolutions."""
    points = [(1.0 / spacings, values[spacings]) for spacings in SPACINGS if spacings in values]
    if len(points) < 2:
        return {"status": "不到两个分辨率，不给阶"}
    signs = {math.copysign(1.0, value) for _, value in points if value != 0.0}
    if len(signs) > 1:
        return {"status": "误差变号（穿过零），不给阶"}
    magnitudes = [(dx, abs(value)) for dx, value in points]
    # finer = smaller dx = later in SPACINGS order: the error must fall from coarse to fine
    decreasing = all(later < earlier for (_, earlier), (_, later) in zip(magnitudes, magnitudes[1:]))
    pairwise = [math.log(first / second) / math.log(dx_first / dx_second)
                for (dx_first, first), (dx_second, second) in zip(magnitudes, magnitudes[1:])
                if first > 0 and second > 0]
    if not decreasing:
        return {"status": "误差随 dx 不单调，不给阶", "pairwise": pairwise}
    slope = float(np.polyfit(np.log([dx for dx, _ in magnitudes]), np.log([value for _, value in magnitudes]), 1)[0])
    return {"status": "ok", "pairwise": pairwise, "least_squares": slope}


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiment.wall_crossing.report_figures import INK_PRIMARY, INK_SECONDARY, save, setup_style
    from experiment.wall_crossing.report_tables import markdown_table, write
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", nargs=2, type=float, default=(35.0, 45.0))
    arguments = parser.parse_args()
    window = tuple(arguments.window)
    reference_y, reference_u = read_reference(REFERENCE_U)
    reference_x, reference_v = read_reference(REFERENCE_V)
    reference_extrema = {"u_min": float(reference_u.min()), "v_max": float(reference_v.max()),
                         "v_min": float(reference_v.min())}
    summary = {"window": window, "methods": {}}
    table_rows, run_rows, order_rows = [], [], []
    loaded: dict = {}
    for method, runs in RUNS.items():
        summary["methods"][method] = {}
        for spacings in SPACINGS:
            name = runs[spacings]
            directory = LOG_ROOT / "box_run" / name
            if not (directory / "series.csv").exists():
                run_rows.append([METHOD_TEXT[method], str(spacings), f"`{name}`", "没有这个运行", "", "", "", "", ""])
                continue
            series = read_series(directory / "series.csv")
            last_time = float(series["time"][-1])
            stop = None
            if (directory / "watch_stop.json").exists():
                stop = json.loads((directory / "watch_stop.json").read_text(encoding="utf-8"))
            profiles = (json.loads((directory / "profiles.json").read_text(encoding="utf-8"))
                        if (directory / "profiles.json").exists() else [])
            mean = window_mean_profiles(profiles, window) if profiles else None
            rows = (series["time"] >= window[0] - 1e-9) & (series["time"] <= window[1] + 1e-9)
            slope = (float(np.polyfit(series["time"][rows], series["kinetic_energy"][rows], 1)[0])
                     if rows.sum() > 2 else float("nan"))
            energy = float(np.mean(series["kinetic_energy"][rows])) if rows.any() else float("nan")
            speed = float(np.median(series["steps_per_second"][series["time"] >= min(10.0, last_time / 2)]))
            maximum_speed = float(np.nanmax(series["speed_maximum"]))
            run_rows.append([METHOD_TEXT[method], str(spacings), f"`{name}`", f"{last_time:.2f}",
                             "—" if stop is None else stop.get("reason", "stopped"), f"{maximum_speed:.3f}",
                             f"{energy:.3f}" if np.isfinite(energy) else "—",
                             f"{slope:+.4f}（{slope / energy * 100:+.3f} %）" if np.isfinite(slope) else "—",
                             f"{1000.0 / speed:.2f}（{speed:.1f}）"])
            record = {"run": name, "last_time": last_time, "stop": stop, "kinetic_energy_mean": energy,
                      "kinetic_energy_slope": slope, "steps_per_second": speed, "maximum_speed": maximum_speed}
            if mean is not None and last_time >= window[1] - 1e-6:
                u, v = mean["u_along_y"], mean["v_along_x"]
                extrema = {"u_min": float(np.nanmin(u["velocity"])), "v_max": float(np.nanmax(v["velocity"])),
                           "v_min": float(np.nanmin(v["velocity"]))}
                error_u = errors(u["coordinate"], u["velocity"], reference_y, reference_u)
                error_v = errors(v["coordinate"], v["velocity"], reference_x, reference_v)
                values = {key: (abs(extrema[key]) - abs(reference_extrema[key])) / abs(reference_extrema[key])
                          for key in ("u_min", "v_max", "v_min")}
                values.update({"l2_u": error_u["l2"], "l2_v": error_v["l2"], "linf_u": error_u["linf"],
                               "linf_v": error_v["linf"]})
                record.update({"extrema": extrema, "values": values, "profiles_in_window": mean["count"]})
                loaded.setdefault(method, {})[spacings] = mean
                table_rows.append([METHOD_TEXT[method], str(spacings),
                                   *[f"{values[key] * 100:+.1f} %" for key in ("u_min", "v_max", "v_min")],
                                   *[f"{values[key]:.4f}" for key in ("l2_u", "l2_v", "linf_u", "linf_v")]])
            summary["methods"][method][spacings] = record
        for key in QUANTITIES:
            values = {spacings: record["values"][key] for spacings, record in summary["methods"][method].items()
                      if "values" in record}
            result = order(values)
            summary["methods"][method].setdefault("orders", {})[key] = result
            if result["status"] == "ok":
                text = (f"p = {result['least_squares']:.2f}（两两的阶：" +
                        "、".join(f"{value:.2f}" for value in result["pairwise"]) + "）")
            elif "pairwise" in result:
                text = f"{result['status']}（两两的阶：" + "、".join(f"{value:.2f}" for value in result["pairwise"]) + "）"
            else:
                text = result["status"]
            order_rows.append([METHOD_TEXT[method], QUANTITY_TEXT[key], text])

    write("cavity3d_resolution_errors.md", markdown_table(
        ["方法", "每边间距数", "u_min 偏差", "v_max 偏差", "v_min 偏差", "u(y) L2", "v(x) L2", "u(y) L∞", "v(x) L∞"],
        table_rows))
    write("cavity3d_resolution_runs.md", markdown_table(
        ["方法", "每边间距数", "运行", "跑到 t", "停止", "全程最大速度", "窗口内动能平均", "窗口内动能斜率",
         "每步 ms（步 / 秒）"], run_rows))
    write("cavity3d_resolution_orders.md", markdown_table(["方法", "量", "收敛阶（误差 ∝ dx^p）"], order_rows))
    # runs that stopped on the speed / outside watch: the stop and the fastest particle at every sample before it
    stop_rows = []
    for method, runs in RUNS.items():
        for spacings in SPACINGS:
            directory = LOG_ROOT / "box_run" / runs[spacings]
            if not (directory / "watch_stop.json").exists():
                continue
            stop = json.loads((directory / "watch_stop.json").read_text(encoding="utf-8"))
            series = read_series(directory / "series.csv")
            history = "；".join(
                f"t {time:.2f}：{speed:.3f}（{x:+.3f}, {y:+.3f}, {z:+.3f}）"
                for time, speed, x, y, z in zip(series["time"], series["speed_maximum"], series["fastest_x"],
                                                 series["fastest_y"], series["fastest_z"]) if time > 0)
            stop_rows.append([METHOD_TEXT[method], str(spacings), f"`{runs[spacings]}`",
                              f"第 {stop['step']:,} 步，t = {stop['time']:.3f}", stop["reason"].split(": ", 1)[-1],
                              history])
    write("cavity3d_resolution_stops.md", markdown_table(
        ["方法", "每边间距数", "运行", "停下的时刻", "原因", "此前每次采样的最大速度（最快粒子的位置 x, y, z）"],
        stop_rows if stop_rows else [["—", "—", "—", "没有运行停下", "—", "—"]]))
    out = LOG_ROOT / "cavity3d_resolution"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")

    setup_style()
    for method, means in loaded.items():
        figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
        for spacings in SPACINGS:
            if spacings not in means:
                continue
            mean = means[spacings]
            colour = RESOLUTION_COLOUR[spacings]
            axes[0].plot(mean["u_along_y"]["velocity"], mean["u_along_y"]["coordinate"], color=colour, linewidth=1.6,
                         label=f"每边 {spacings} 个间距")
            axes[1].plot(mean["v_along_x"]["coordinate"], mean["v_along_x"]["velocity"], color=colour, linewidth=1.6)
        axes[0].plot(reference_u, reference_y, "o", color=INK_PRIMARY, markersize=4.5, label="Albensoeder & Kuhlmann 2005")
        axes[1].plot(reference_x, reference_v, "o", color=INK_PRIMARY, markersize=4.5)
        axes[0].set_xlabel("u / U")
        axes[0].set_ylabel("y / L")
        axes[0].set_title("竖线 x = z = 0.5")
        axes[1].set_xlabel("x / L")
        axes[1].set_ylabel("v / U")
        axes[1].set_title("横线 y = z = 0.5")
        handles, labels = axes[0].get_legend_handles_labels()
        height = figure.get_figheight()
        resolution_text = ("三个分辨率" if len(means) == len(SPACINGS) else
                           "只有每边 " + "、".join(str(value) for value in SPACINGS if value in means) +
                           " 个间距（其余的运行没有到统计窗口）")
        figure.suptitle(f"3-D 方腔 Re = 1000，{METHOD_TEXT[method]}：t = {window[0]:g}–{window[1]:g} 的平均剖面，{resolution_text}",
                        x=0.01, y=1.0 - 0.05 / height, ha="left", va="top", fontsize=12, fontweight="bold",
                        color=INK_PRIMARY)
        figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0 - 0.40 / height), ncol=4,
                      handlelength=3.0, labelcolor=INK_PRIMARY, borderaxespad=0.0)
        figure.tight_layout(rect=(0, 0, 1.0, 1.0 - 0.70 / height), w_pad=2.0)
        save(figure, REPORT_ROOT / "figures" / f"cavity3d_resolution_{method}.png")

    # the three methods at 100 spacings per side (1M) together (section 25 C)
    finest = {method: means[SPACINGS[-1]] for method, means in loaded.items() if SPACINGS[-1] in means}
    if finest:
        method_colour = {"crossing": "#2a78d6", "simple": "#eb6834", "adami": "#1baf7a"}
        figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
        for method, mean in finest.items():
            axes[0].plot(mean["u_along_y"]["velocity"], mean["u_along_y"]["coordinate"], color=method_colour[method],
                         linewidth=1.6, label=METHOD_TEXT[method])
            axes[1].plot(mean["v_along_x"]["coordinate"], mean["v_along_x"]["velocity"], color=method_colour[method],
                         linewidth=1.6)
        axes[0].plot(reference_u, reference_y, "o", color=INK_PRIMARY, markersize=4.5, label="Albensoeder & Kuhlmann 2005")
        axes[1].plot(reference_x, reference_v, "o", color=INK_PRIMARY, markersize=4.5)
        axes[0].set_xlabel("u / U")
        axes[0].set_ylabel("y / L")
        axes[0].set_title("竖线 x = z = 0.5")
        axes[1].set_xlabel("x / L")
        axes[1].set_ylabel("v / U")
        axes[1].set_title("横线 y = z = 0.5")
        handles, labels = axes[0].get_legend_handles_labels()
        height = figure.get_figheight()
        figure.suptitle(f"3-D 方腔 Re = 1000，每边 {SPACINGS[-1]} 个间距（1M）：三种壁面方法，t = {window[0]:g}–{window[1]:g} 的平均剖面",
                        x=0.01, y=1.0 - 0.05 / height, ha="left", va="top", fontsize=12, fontweight="bold",
                        color=INK_PRIMARY)
        figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0 - 0.40 / height), ncol=4,
                      handlelength=3.0, labelcolor=INK_PRIMARY, borderaxespad=0.0)
        figure.tight_layout(rect=(0, 0, 1.0, 1.0 - 0.70 / height), w_pad=2.0)
        save(figure, REPORT_ROOT / "figures" / "cavity3d_resolution_1m_methods.png")

    # histories: kinetic energy of every run; crossing's mean density / pressure / theta / I; simple / adami density
    figure, axes = plt.subplots(2, 3, figsize=(15.0, 7.4))
    panels = {"crossing": 0, "simple": 1, "adami": 2}
    for method, runs in RUNS.items():
        for spacings in SPACINGS:
            directory = LOG_ROOT / "box_run" / runs[spacings]
            if not (directory / "series.csv").exists():
                continue
            series = read_series(directory / "series.csv")
            colour = RESOLUTION_COLOUR[spacings]
            axes[0, panels[method]].plot(series["time"], series["kinetic_energy"], color=colour, linewidth=1.2,
                                         label=f"每边 {spacings} 个间距")
            axes[1, panels[method]].plot(series["time"], series["density_mean"] - REST_DENSITY, color=colour,
                                         linewidth=1.2)
    for method, column in panels.items():
        axes[0, column].set_title(f"{method}：流体动能")
        axes[1, column].set_title(f"{method}：平均密度 − ρ0（kg/m³）")
        for row in range(2):
            axes[row, column].set_xlabel("时间（模拟单位）")
            axes[row, column].axvspan(*window, color=INK_SECONDARY, alpha=0.08, linewidth=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    height = figure.get_figheight()
    figure.suptitle("3-D 方腔：三个分辨率的动能与平均密度（灰带 = 统计窗口）", x=0.01, y=1.0 - 0.05 / height, ha="left",
                    va="top", fontsize=12, fontweight="bold", color=INK_PRIMARY)
    figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0 - 0.40 / height), ncol=3,
                  handlelength=3.0, labelcolor=INK_PRIMARY, borderaxespad=0.0)
    figure.tight_layout(rect=(0, 0, 1.0, 1.0 - 0.70 / height), w_pad=2.0, h_pad=1.5)
    save(figure, REPORT_ROOT / "figures" / "cavity3d_resolution_history.png")

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.0))
    for spacings in SPACINGS:
        directory = LOG_ROOT / "box_run" / RUNS["crossing"][spacings]
        if not (directory / "series.csv").exists():
            continue
        series = read_series(directory / "series.csv")
        colour = RESOLUTION_COLOUR[spacings]
        axes[0].plot(series["time"], series["pressure_mean"], color=colour, linewidth=1.2, label=f"每边 {spacings} 个间距")
        constraint = [json.loads(line) for line in (directory / "volume_constraint.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip() and "theta" in line] \
            if (directory / "volume_constraint.jsonl").exists() else []
        if constraint:
            axes[1].plot([record["time"] for record in constraint], [record["theta"] for record in constraint],
                         color=colour, linewidth=1.2)
            axes[2].plot([record["time"] for record in constraint], [record.get("integral", np.nan) for record in constraint],
                         color=colour, linewidth=1.2)
    axes[0].set_title("crossing：平均压力（Pa）")
    axes[1].set_title("crossing：θ")
    axes[2].set_title("crossing：积分项 I")
    for axis in axes:
        axis.set_xlabel("时间（模拟单位）")
        axis.axvspan(*window, color=INK_SECONDARY, alpha=0.08, linewidth=0)
    handles, labels = axes[0].get_legend_handles_labels()
    height = figure.get_figheight()
    figure.suptitle("3-D 方腔 crossing：平均压力、θ 与 I（灰带 = 统计窗口）", x=0.01, y=1.0 - 0.05 / height, ha="left",
                    va="top", fontsize=12, fontweight="bold", color=INK_PRIMARY)
    figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0 - 0.40 / height), ncol=3,
                  handlelength=3.0, labelcolor=INK_PRIMARY, borderaxespad=0.0)
    figure.tight_layout(rect=(0, 0, 1.0, 1.0 - 0.70 / height), w_pad=2.0)
    save(figure, REPORT_ROOT / "figures" / "cavity3d_resolution_crossing_constraint.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
