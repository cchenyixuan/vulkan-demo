"""
cavity3d_reference.py - report section 24 B: the 3-D cavity (1M, Re = 1000) of box_run runs against the benchmark of
Albensoeder & Kuhlmann (2005), Tables 5 and 6 (experiment/validation/reference/cavity3d_re1000_albensoeder_kuhlmann_
2005_*.csv; source and conventions in their headers).

    .venv/Scripts/python.exe -m experiment.wall_crossing.cavity3d_reference [--window 35 45]

Per run of RUNS that exists under logs/wall_crossing/box_run/:
  profiles   the centre-line profiles box_run recorded (Shepard average of the fluid particles, 101 points; the
             vertical line u(y) and the horizontal line v(x) through the box centre, in the plane z = centre; the
             coordinates normalised by the fluid box widened by one particle radius, i.e. the walls half-way between
             the last fluid layer and the first wall layer: the "mid frame" of section 16), averaged over the window.
             Mapping: our lid is the upper y wall and moves in +x, as in the benchmark; both coordinates run from 0 at
             the low wall to 1 at the high wall, so the benchmark's (x, y) are ours without a change.
  errors     our window-mean profile interpolated linearly to the benchmark's points; L2 = rms and Linf = largest
             |ours - benchmark| (in units of the lid speed) over the benchmark points strictly inside the cavity, and
             again without the points closer to a wall than the support radius h (there the Shepard average sees fluid
             on one side only, section 19.5);
  extrema    u_min on the vertical line, v_max and v_min on the horizontal line: ours = the extremum of the 101-point
             mean profile (the coordinate of its point), the benchmark's = the extremum of its table points (the
             tables list values at Ghia's points, not the exact extrema); the deviation is that of the
             magnitude, (|ours| - |benchmark|) / |benchmark|, + = stronger;
  history    kinetic energy (the window's mean and slope), mean density, mean pressure, theta and I of the volume
             constraint (the run's volume_constraint.jsonl), steps per second.
Writes claude/wall_crossing/tables/cavity3d_reference_*.md, claude/wall_crossing/figures/cavity3d_reference_*.png and
logs/wall_crossing/cavity3d_reference/summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.run_log import LOG_ROOT     # noqa: E402

REFERENCE_ROOT = REPOSITORY_ROOT / "experiment" / "validation" / "reference"
REFERENCE_U = REFERENCE_ROOT / "cavity3d_re1000_albensoeder_kuhlmann_2005_u_vertical.csv"
REFERENCE_V = REFERENCE_ROOT / "cavity3d_re1000_albensoeder_kuhlmann_2005_v_horizontal.csv"
REPORT_ROOT = REPOSITORY_ROOT / "claude" / "wall_crossing"
REST_DENSITY = 1000.0
# (run name, text, colour, line width, dash): categorical slots of the report's palette in a fixed order
RUNS = (("cavity3d_1m_crossing_long", "crossing 完整速度差，比例修正（第 19 节）", "#2a78d6", 1.6, "solid"),
        ("s24_cavity3d_full_integral", "crossing 完整速度差，比例 + 积分修正（本节）", "#eb6834", 1.6, "solid"),
        ("s24_cavity3d_normal_integral", "crossing 只取法向，比例 + 积分修正（本节）", "#eda100", 1.6, (0, (5, 2))),
        ("cavity3d_1m_simple_long", "simple（第 19 节）", "#1baf7a", 1.6, "solid"))
# the support radius divided by the cavity's edge in the mid frame (h = 0.04, edge 1.01)
SUPPORT_OVER_EDGE = 0.04 / 1.01


def read_reference(path: pathlib.Path) -> tuple:
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    data = np.array([[float(value) for value in row.split(",")] for row in rows[1:]])
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1]


def read_series(path: pathlib.Path) -> dict:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(row[key]) if row[key] not in ("", None) else np.nan for row in rows])
            for key in rows[0]}


def window_mean_profiles(profiles: list, window: tuple) -> dict | None:
    chosen = [entry for entry in profiles if window[0] - 1e-9 <= entry["time"] <= window[1] + 1e-9]
    if not chosen:
        return None
    result = {"count": len(chosen), "times": [entry["time"] for entry in chosen]}
    for line in ("u_along_y", "v_along_x"):
        result[line] = {"coordinate": np.asarray(chosen[0][line]["coordinate"]),
                        "velocity": np.nanmean([entry[line]["velocity"] for entry in chosen], axis=0)}
    return result


def errors(coordinate, velocity, reference_coordinate, reference_velocity) -> dict:
    inside = (reference_coordinate > 0.0) & (reference_coordinate < 1.0)
    away = inside & (np.minimum(reference_coordinate, 1.0 - reference_coordinate) >= SUPPORT_OVER_EDGE)
    finite = np.isfinite(velocity)
    ours = np.interp(reference_coordinate, coordinate[finite], velocity[finite])
    difference = ours - reference_velocity
    return {"points": int(inside.sum()), "l2": float(np.sqrt(np.mean(difference[inside] ** 2))),
            "linf": float(np.max(np.abs(difference[inside]))),
            "points_away": int(away.sum()), "l2_away": float(np.sqrt(np.mean(difference[away] ** 2))),
            "linf_away": float(np.max(np.abs(difference[away]))),
            "at_points": [[float(c), float(r), float(o)] for c, r, o in
                          zip(reference_coordinate[inside], reference_velocity[inside], ours[inside])]}


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
    reference_extrema = {"u_min": (float(reference_u.min()), float(reference_y[np.argmin(reference_u)])),
                         "v_max": (float(reference_v.max()), float(reference_x[np.argmax(reference_v)])),
                         "v_min": (float(reference_v.min()), float(reference_x[np.argmin(reference_v)]))}
    summary = {"window": window, "reference_extrema": reference_extrema, "runs": {}}
    extrema_rows, error_rows, history_rows, steady_rows = [], [], [], []
    plotted = []
    for name, text, colour, width, dash in RUNS:
        directory = LOG_ROOT / "box_run" / name
        if not (directory / "profiles.json").exists():
            continue
        profiles = json.loads((directory / "profiles.json").read_text(encoding="utf-8"))
        mean = window_mean_profiles(profiles, window)
        series = read_series(directory / "series.csv")
        last_time = float(series["time"][-1])
        if mean is None:
            print(f"[cavity3d_reference] {name}: no profile in the window yet (last t = {last_time:.2f})")
            continue
        u, v = mean["u_along_y"], mean["v_along_x"]
        extrema = {"u_min": (float(np.nanmin(u["velocity"])), float(u["coordinate"][np.nanargmin(u["velocity"])])),
                   "v_max": (float(np.nanmax(v["velocity"])), float(v["coordinate"][np.nanargmax(v["velocity"])])),
                   "v_min": (float(np.nanmin(v["velocity"])), float(v["coordinate"][np.nanargmin(v["velocity"])]))}
        error_u = errors(u["coordinate"], u["velocity"], reference_y, reference_u)
        error_v = errors(v["coordinate"], v["velocity"], reference_x, reference_v)
        cells = [text, f"{mean['times'][0]:.2f}–{mean['times'][-1]:.2f}（{mean['count']} 条）"]
        for key in ("u_min", "v_max", "v_min"):
            value, where = extrema[key]
            reference_value = reference_extrema[key][0]
            # the deviation of the magnitude: + = stronger than the benchmark
            cells.append(f"{value:+.4f}（{where:.2f}），{(abs(value) - abs(reference_value)) / abs(reference_value) * 100:+.1f} %")
        extrema_rows.append(cells)
        error_rows.append([text, f"{error_u['l2']:.4f}", f"{error_u['linf']:.4f}", f"{error_u['l2_away']:.4f}",
                           f"{error_u['linf_away']:.4f}", f"{error_v['l2']:.4f}", f"{error_v['linf']:.4f}",
                           f"{error_v['l2_away']:.4f}", f"{error_v['linf_away']:.4f}"])
        # history: kinetic energy in the window (mean, slope), mean density / pressure, theta / I at the end
        rows = (series["time"] >= window[0] - 1e-9) & (series["time"] <= window[1] + 1e-9)
        slope = float(np.polyfit(series["time"][rows], series["kinetic_energy"][rows], 1)[0]) if rows.sum() > 2 else np.nan
        energy = float(np.mean(series["kinetic_energy"][rows]))
        constraint = []
        if (directory / "volume_constraint.jsonl").exists():
            constraint = [json.loads(line) for line in (directory / "volume_constraint.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip() and "theta" in line]
        constraint_window = [record for record in constraint if window[0] - 1e-9 <= record["time"] <= window[1] + 1e-9]
        theta = float(np.mean([record["theta"] for record in constraint_window])) if constraint_window else np.nan
        integral = (float(np.mean([record["integral"] for record in constraint_window]))
                    if constraint_window and "integral" in constraint_window[0] else np.nan)
        speed = float(np.median(series["steps_per_second"][series["time"] >= 10.0])) \
            if (series["time"] >= 10.0).any() else float(series["steps_per_second"][-1])
        history_rows.append([text, f"{last_time:.2f}", f"{energy:.3f}", f"{slope:+.4f}（{slope / energy * 100:+.3f} %）",
                             f"{np.mean(series['density_mean'][rows]) - REST_DENSITY:+.4f}",
                             f"{np.mean(series['pressure_mean'][rows]):+.1f}",
                             f"{theta:+.2e}" if np.isfinite(theta) else "—",
                             f"{integral:+.2e}" if np.isfinite(integral) else "—",
                             f"{1000.0 / speed:.2f}（{speed:.1f}）"])
        # how far from steady: the extrema of an earlier window of the same length, and an exponential fit of the
        # kinetic energy after its maximum, K(t) = K_inf + A exp(-t / T) for t >= 20
        earlier = window_mean_profiles(profiles, (window[0] - (window[1] - window[0]), window[0]))
        drift = None
        if earlier is not None:
            earlier_extrema = (float(np.nanmin(earlier["u_along_y"]["velocity"])),
                               float(np.nanmax(earlier["v_along_x"]["velocity"])),
                               float(np.nanmin(earlier["v_along_x"]["velocity"])))
            drift = [extrema["u_min"][0] - earlier_extrema[0], extrema["v_max"][0] - earlier_extrema[1],
                     extrema["v_min"][0] - earlier_extrema[2]]
        fit = None
        late = series["time"] >= 20.0
        if late.sum() > 20 and last_time >= window[1] - 1e-6:
            from scipy.optimize import curve_fit
            time_late, energy_late = series["time"][late], series["kinetic_energy"][late]
            try:
                parameters, _ = curve_fit(lambda t, limit, amplitude, scale: limit + amplitude * np.exp(-(t - 20.0) / scale),
                                          time_late, energy_late, p0=(energy_late[-1], energy_late[0] - energy_late[-1], 10.0),
                                          maxfev=20000)
                fit = {"limit": float(parameters[0]), "amplitude": float(parameters[1]), "time_scale": float(parameters[2]),
                       "remaining_at_end": float(energy_late[-1] - parameters[0])}
            except RuntimeError:
                fit = None
        steady_rows.append([text,
                            ("—" if drift is None else
                             f"{drift[0]:+.4f} / {drift[1]:+.4f} / {drift[2]:+.4f}"),
                            ("—" if fit is None else
                             f"K∞ = {fit['limit']:.3f}，T = {fit['time_scale']:.1f}，t = {last_time:.0f} 时还差 "
                             f"{fit['remaining_at_end']:+.3f}（{fit['remaining_at_end'] / fit['limit'] * 100:+.2f} %）")])
        summary["runs"][name] = {"text": text, "profiles_in_window": mean["count"], "last_time": last_time,
                                 "extrema": extrema, "errors_u": error_u, "errors_v": error_v,
                                 "kinetic_energy_mean": energy, "kinetic_energy_slope": slope,
                                 "theta_mean": theta, "integral_mean": integral, "steps_per_second": speed,
                                 "extrema_drift_from_earlier_window": drift, "kinetic_energy_fit": fit}
        plotted.append((name, text, colour, width, dash, mean, series, constraint))

    write("cavity3d_reference_extrema.md", markdown_table(
        ["运行", "窗口内的剖面", "竖线 u_min（位置 y），大小相对基准", "横线 v_max（位置 x），大小相对基准",
         "横线 v_min（位置 x），大小相对基准"],
        extrema_rows + [["基准（表 5、6 的格点）", "—",
                         f"{reference_extrema['u_min'][0]:+.4f}（{reference_extrema['u_min'][1]:.4f}）",
                         f"{reference_extrema['v_max'][0]:+.4f}（{reference_extrema['v_max'][1]:.4f}）",
                         f"{reference_extrema['v_min'][0]:+.4f}（{reference_extrema['v_min'][1]:.4f}）"]]))
    write("cavity3d_reference_errors.md", markdown_table(
        ["运行", "u(y) L2", "u(y) L∞", "u(y) L2，离壁 ≥ h", "u(y) L∞，离壁 ≥ h", "v(x) L2", "v(x) L∞",
         "v(x) L2，离壁 ≥ h", "v(x) L∞，离壁 ≥ h"], error_rows))
    write("cavity3d_reference_history.md", markdown_table(
        ["运行", "跑到 t", "窗口内动能平均", "窗口内动能斜率（每单位时间）", "窗口内平均密度 − ρ0",
         "窗口内平均压力（Pa）", "窗口内 θ 平均", "窗口内 I 平均", "每步 ms（步 / 秒，t ≥ 10）"], history_rows))
    write("cavity3d_reference_steady.md", markdown_table(
        ["运行", f"极值相对前一个窗口（t = {2 * window[0] - window[1]:g}–{window[0]:g}）的变化：u_min / v_max / v_min",
         "动能 t ≥ 20 的指数拟合 K∞ + A exp(−(t − 20) / T)"], steady_rows))
    # per-point table of the two lines
    point_rows = []
    for line, reference_coordinate, reference_velocity, key in (("u(y)", reference_y, reference_u, "errors_u"),
                                                                ("v(x)", reference_x, reference_v, "errors_v")):
        inside = (reference_coordinate > 0) & (reference_coordinate < 1)
        for index, (coordinate, value) in enumerate(zip(reference_coordinate[inside], reference_velocity[inside])):
            cells = [line, f"{coordinate:.4f}", f"{value:+.4f}"]
            for name, *_ in plotted:
                cells.append(f"{summary['runs'][name][key]['at_points'][index][2]:+.4f}")
            point_rows.append(cells)
    write("cavity3d_reference_points.md", markdown_table(
        ["线", "坐标", "基准", *[summary["runs"][name]["text"] for name, *_ in plotted]], point_rows))
    out = LOG_ROOT / "cavity3d_reference"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    setup_style()
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for name, text, colour, width, dash, mean, series, constraint in plotted:
        axes[0].plot(mean["u_along_y"]["velocity"], mean["u_along_y"]["coordinate"], color=colour, linewidth=width,
                     linestyle=dash, label=text)
        axes[1].plot(mean["v_along_x"]["coordinate"], mean["v_along_x"]["velocity"], color=colour, linewidth=width,
                     linestyle=dash)
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
    figure.suptitle(f"3-D 方腔 1M，Re = 1000：t = {window[0]:g}–{window[1]:g} 的平均中心线剖面与基准", x=0.01,
                    y=1.0 - 0.05 / height, ha="left", va="top", fontsize=12, fontweight="bold", color=INK_PRIMARY)
    figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0 - 0.40 / height), ncol=3,
                  handlelength=3.0, labelcolor=INK_PRIMARY, borderaxespad=0.0)
    figure.tight_layout(rect=(0, 0, 1.0, 1.0 - 0.95 / height), w_pad=2.0)
    save(figure, REPORT_ROOT / "figures" / "cavity3d_reference_profiles.png")

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 7.0))
    for name, text, colour, width, dash, mean, series, constraint in plotted:
        axes[0, 0].plot(series["time"], series["kinetic_energy"], color=colour, linewidth=1.2, linestyle=dash,
                        label=text)
        axes[0, 1].plot(series["time"], series["density_mean"] - REST_DENSITY, color=colour, linewidth=1.2,
                        linestyle=dash)
        if constraint:
            times = [record["time"] for record in constraint]
            axes[1, 0].plot(times, [record["theta"] for record in constraint], color=colour, linewidth=1.2,
                            linestyle=dash)
            if "integral" in constraint[-1]:
                axes[1, 1].plot(times, [record.get("integral", np.nan) for record in constraint], color=colour,
                                linewidth=1.2, linestyle=dash)
    axes[0, 0].set_title("流体动能")
    axes[0, 1].set_title("流体平均密度 − ρ0（kg/m³）")
    axes[1, 0].set_title("总体积修正的 θ（每段的时间从 0 计；这些运行都只有一段）")
    axes[1, 1].set_title("积分项 I")
    for axis in axes.flat:
        axis.set_xlabel("时间（模拟单位）")
        axis.axvspan(*window, color=INK_SECONDARY, alpha=0.08, linewidth=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    height = figure.get_figheight()
    figure.suptitle("3-D 方腔 1M：动能、平均密度、θ 与 I 的历史（灰带 = 统计窗口）", x=0.01, y=1.0 - 0.05 / height,
                    ha="left", va="top", fontsize=12, fontweight="bold", color=INK_PRIMARY)
    figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0 - 0.40 / height), ncol=2,
                  handlelength=3.0, labelcolor=INK_PRIMARY, borderaxespad=0.0)
    figure.tight_layout(rect=(0, 0, 1.0, 1.0 - 0.95 / height), w_pad=2.0, h_pad=1.5)
    save(figure, REPORT_ROOT / "figures" / "cavity3d_reference_history.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
