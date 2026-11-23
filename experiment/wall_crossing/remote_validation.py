"""
remote_validation.py - A1 and A2 of claude/wall_store_report.md section 9: the long runs of the wall store (21877e3)
on the remote 5090, against the particle-pool version (tag crossing-pool's line: report wall_crossing_report.md
sections 15 and 24) and the benchmarks. Reads only the small files pulled back to logs/wall_store/round3/remote/.

    .venv/Scripts/python.exe -m experiment.wall_crossing.remote_validation a1 [--round5]
    .venv/Scripts/python.exe -m experiment.wall_crossing.remote_validation a2 [--round5]

--round5 (report section 11.5): the round-5 run of the final default (commit 0b92959, the same remote 5090, the same
job arguments; small files in logs/wall_store/round5/remote/) as a third row / line; the tables and the figure get
the prefix round5_ instead of round3_, the round-3 files stay as they are.

a1  3-D 1M cavity (box_run r3_cavity3d_1m_store, to t = 45) against Albensoeder & Kuhlmann (2005) with section 24's
    post-processing (cavity3d_reference.py: the 101-point centre-line profiles of box_run, averaged over t = 35-45, in
    the mid frame; extrema deviations of the magnitude, L2 / Linf over the benchmark's interior points), next to the
    particle-pool run of section 24 (s24_cavity3d_full_integral, the same case settings, logs/wall_crossing/box_run);
    the history: kinetic energy (window mean, slope, an exponential fit for the distance to steady), mean density,
    mean pressure, theta and I of the volume constraint. Tables round3_a1_*.md, figure round3_a1_history.png.
a2  2-D 500^2 cavity (cavity_run r3_n500_store, t = 80-100; cavity_window_analysis.py ran on the remote box where the
    snapshots are): psi_min in the wall and the mid frame, the three extrema, the L2 against Marchi 2021, next to the
    particle-pool run of section 15 (n500_reflection_full_volume; its psi_min from that report's table: the
    snapshots were not kept, its samples are, so its extrema and L2 are recomputed here with the same analysis).
    Table round3_a2.md; figure round3_a2_history.png (kinetic energy, mean density, theta of both runs).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

REMOTE = REPOSITORY_ROOT / "logs" / "wall_store" / "round3" / "remote"
ROUND5_REMOTE = REPOSITORY_ROOT / "logs" / "wall_store" / "round5" / "remote"
POOL_BOX = REPOSITORY_ROOT / "logs" / "wall_crossing" / "box_run" / "s24_cavity3d_full_integral"
POOL_CAVITY_ROOT = REPOSITORY_ROOT / "logs" / "wall_crossing" / "cavity"
TABLES = REPOSITORY_ROOT / "claude" / "wall_store" / "tables"
FIGURES = REPOSITORY_ROOT / "claude" / "wall_store" / "figures"
REST_DENSITY = 1000.0
# section 15's table (wall_crossing_report.md): the particle-pool 500^2 run's psi_min, wall and mid frame
POOL_N500_PSI = {"wall": -0.11803, "mid": -0.11827}


def a1(round5: bool = False) -> int:
    from experiment.wall_crossing.cavity3d_reference import (REFERENCE_U, REFERENCE_V, errors, read_reference,
                                                             read_series, window_mean_profiles)
    window = (35.0, 45.0)
    reference_y, reference_u = read_reference(REFERENCE_U)
    reference_x, reference_v = read_reference(REFERENCE_V)
    reference = {"u_min": float(reference_u.min()), "v_max": float(reference_v.max()), "v_min": float(reference_v.min())}
    runs = {"store": (REMOTE / "store" / "box_run" / "r3_cavity3d_1m_store", "壁面独立存储（21877e3，远端）"),
            "pool": (POOL_BOX, "粒子池版本（第 24 节 s24_cavity3d_full_integral，本机）")}
    if round5:
        runs = {"round5": (ROUND5_REMOTE / "box_run" / "r5_cavity3d_1m", "最终缺省（0b92959，远端，第 11 节）"), **runs}
    prefix = "round5" if round5 else "round3"
    summary, extrema_rows, history_rows = {}, [], []
    loaded = {}
    for key, (directory, text) in runs.items():
        profiles = json.loads((directory / "profiles.json").read_text(encoding="utf-8"))
        series = read_series(directory / "series.csv")
        mean = window_mean_profiles(profiles, window)
        u, v = mean["u_along_y"], mean["v_along_x"]
        extrema = {"u_min": float(np.nanmin(u["velocity"])), "v_max": float(np.nanmax(v["velocity"])),
                   "v_min": float(np.nanmin(v["velocity"]))}
        deviation = {name: (abs(value) - abs(reference[name])) / abs(reference[name]) for name, value in extrema.items()}
        error_u = errors(u["coordinate"], u["velocity"], reference_y, reference_u)
        error_v = errors(v["coordinate"], v["velocity"], reference_x, reference_v)
        rows = (series["time"] >= window[0] - 1e-9) & (series["time"] <= window[1] + 1e-9)
        energy = float(np.mean(series["kinetic_energy"][rows]))
        slope = float(np.polyfit(series["time"][rows], series["kinetic_energy"][rows], 1)[0])
        constraint = [json.loads(line) for line in (directory / "volume_constraint.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip() and "theta" in line]
        chosen = [record for record in constraint if window[0] - 1e-9 <= record["time"] <= window[1] + 1e-9]
        fit = None
        late = series["time"] >= 20.0
        if late.sum() > 20:
            from scipy.optimize import curve_fit
            try:
                parameters, _ = curve_fit(lambda t, limit, amplitude, scale: limit + amplitude * np.exp(-(t - 20.0) / scale),
                                          series["time"][late], series["kinetic_energy"][late],
                                          p0=(series["kinetic_energy"][late][-1], 0.5, 10.0), maxfev=20000)
                fit = {"limit": float(parameters[0]), "time_scale": float(parameters[2]),
                       "remaining": float(series["kinetic_energy"][late][-1] - parameters[0])}
            except RuntimeError:
                fit = None
        speed = float(np.median(series["steps_per_second"][series["time"] >= 10.0]))
        entry = {"text": text, "last_time": float(series["time"][-1]), "profiles": mean["count"],
                 "extrema": extrema, "deviation": deviation, "errors_u": error_u, "errors_v": error_v,
                 "kinetic_energy_mean": energy, "kinetic_energy_slope": slope, "kinetic_energy_fit": fit,
                 "density_mean": float(np.mean(series["density_mean"][rows])),
                 "pressure_mean": float(np.mean(series["pressure_mean"][rows])),
                 "theta_mean": float(np.mean([record["theta"] for record in chosen])),
                 "integral_mean": float(np.mean([record["integral"] for record in chosen])),
                 "speed_maximum": float(np.max(series["speed_maximum"])),
                 "alive": [int(series["fluid_alive"][0]), int(series["fluid_alive"][-1])],
                 "outside_box_maximum": int(np.max(series["outside_box_count"])),
                 "steps_per_second": speed}
        summary[key] = entry
        loaded[key] = (series, constraint, mean)
        extrema_rows.append(f"| {text} | {entry['last_time']:.2f} | {entry['profiles']} | "
                            + " | ".join(f"{extrema[name]:+.4f}（{100 * deviation[name]:+.1f} %）"
                                         for name in ("u_min", "v_max", "v_min"))
                            + f" | {error_u['l2']:.4f} / {error_v['l2']:.4f} | {error_u['linf']:.4f} / {error_v['linf']:.4f} |")
        history_rows.append(
            f"| {text} | {energy:.3f} | {slope:+.4f}（{100 * slope / energy:+.3f} % / 单位时间） | "
            + ("—" if fit is None else f"K∞ = {fit['limit']:.3f}，T = {fit['time_scale']:.1f}，末态还差 "
               f"{fit['remaining']:+.3f}（{100 * fit['remaining'] / fit['limit']:+.2f} %）")
            + f" | {entry['density_mean'] - REST_DENSITY:+.4f} | {entry['pressure_mean']:+.1f} | "
            f"{entry['theta_mean']:+.2e} | {entry['integral_mean']:+.2e} | {entry['speed_maximum']:.3f} | "
            f"{entry['alive'][0]:,} → {entry['alive'][-1]:,}；越出盒 {entry['outside_box_maximum']} | "
            f"{1000.0 / speed:.2f}（{speed:.1f}） |")
    extrema_rows.append(f"| 基准（Albensoeder & Kuhlmann 2005，表 5、6 的格点） | — | — | {reference['u_min']:+.4f} | "
                        f"{reference['v_max']:+.4f} | {reference['v_min']:+.4f} | — | — |")
    def compare(first: str, second: str) -> tuple:
        extrema = {name: summary[first]["extrema"][name] - summary[second]["extrema"][name]
                   for name in ("u_min", "v_max", "v_min")}
        profiles = {}
        for line in ("u_along_y", "v_along_x"):
            delta = np.asarray(loaded[first][2][line]["velocity"]) - np.asarray(loaded[second][2][line]["velocity"])
            profiles[line] = {"l2": float(np.sqrt(np.nanmean(delta ** 2))), "linf": float(np.nanmax(np.abs(delta)))}
        return extrema, profiles
    difference, profile_difference = compare("store", "pool")
    summary["difference"] = {"extrema": difference, "profiles": profile_difference}
    pair_lines = []
    if round5:
        for first, second, text in (("round5", "store", "最终缺省 − 21877e3"), ("round5", "pool", "最终缺省 − 粒子池")):
            extrema, profiles = compare(first, second)
            summary[f"difference_{first}_{second}"] = {"extrema": extrema, "profiles": profiles}
            pair_lines.append(f"{text}：剖面之差 u(y) L2 {profiles['u_along_y']['l2']:.4f}、L∞ {profiles['u_along_y']['linf']:.4f}；"
                              f"v(x) L2 {profiles['v_along_x']['l2']:.4f}、L∞ {profiles['v_along_x']['linf']:.4f}；极值之差 "
                              f"u_min {extrema['u_min']:+.4f}，v_max {extrema['v_max']:+.4f}，v_min {extrema['v_min']:+.4f}。")
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / f"{prefix}_a1_extrema.md").write_text(
        "| 3-D 1M 方腔，t = 35–45 平均 | 跑到 t | 窗口内剖面数 | u_min（大小相对基准） | v_max | v_min | 剖面 L2：u(y) / v(x) | L∞ |\n"
        "|---|---|---|---|---|---|---|---|\n" + "\n".join(extrema_rows) + "\n\n"
        f"两个版本 101 点平均剖面之差：u(y) L2 {profile_difference['u_along_y']['l2']:.4f}、L∞ "
        f"{profile_difference['u_along_y']['linf']:.4f}；v(x) L2 {profile_difference['v_along_x']['l2']:.4f}、L∞ "
        f"{profile_difference['v_along_x']['linf']:.4f}（盖速为单位）。极值之差（新 − 池）：u_min {difference['u_min']:+.4f}，"
        f"v_max {difference['v_max']:+.4f}，v_min {difference['v_min']:+.4f}。\n"
        + "".join("\n" + line + "\n" for line in pair_lines), encoding="utf-8", newline="\n")
    (TABLES / f"{prefix}_a1_history.md").write_text(
        "| 运行 | 窗口内动能平均 | 窗口内动能斜率 | 动能 t ≥ 20 的指数拟合 | 窗口内平均密度 − ρ0 | 窗口内平均压力 Pa | θ 平均 | I 平均 | 全程最大速度 | 粒子数 | 每步 ms（步 / 秒） |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(history_rows) + "\n", encoding="utf-8", newline="\n")
    ((ROUND5_REMOTE if round5 else REMOTE) / "a1_summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiment.wall_crossing.report_figures import save, setup_style
    setup_style()
    colours = {"store": "#2a78d6", "pool": "#eb6834", "round5": "#1baf7a"}
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 7.6))
    for key, (series, constraint, _) in loaded.items():
        label = runs[key][1]
        axes[0, 0].plot(series["time"], series["kinetic_energy"], color=colours[key], linewidth=1.4, label=label)
        axes[0, 1].plot(series["time"], series["density_mean"] - REST_DENSITY, color=colours[key], linewidth=1.2)
        axes[1, 0].plot(series["time"], series["pressure_mean"], color=colours[key], linewidth=1.2)
        times = [record["time"] for record in constraint]
        axes[1, 1].plot(times, [record["theta"] for record in constraint], color=colours[key], linewidth=1.2,
                        label=f"θ，{label}")
        axes[1, 1].plot(times, [record["integral"] for record in constraint], color=colours[key], linewidth=1.2,
                        linestyle=(0, (4, 2)), label=f"I，{label}")
    axes[0, 0].set_ylabel("动能（J）")
    axes[0, 1].set_ylabel("平均密度 − ρ0（kg/m³）")
    axes[1, 0].set_ylabel("平均压力（Pa）")
    axes[1, 1].set_ylabel("θ、I")
    for axis in axes.ravel():
        axis.set_xlabel("t")
        axis.axvspan(*window, color="#e9e8e4", zorder=0)
    axes[0, 0].legend(loc="lower right", fontsize=8)
    axes[1, 1].legend(loc="center right", fontsize=7)
    figure.suptitle("3-D 1M 方腔：动能、平均密度、平均压力、总体积修正的 θ 与 I（灰底：统计窗口 t = 35–45）")
    FIGURES.mkdir(parents=True, exist_ok=True)
    save(figure, FIGURES / f"{prefix}_a1_history.png")
    print(json.dumps({key: {"deviation": value["deviation"], "l2": [value["errors_u"]["l2"], value["errors_v"]["l2"]]}
                      for key, value in summary.items() if key in runs}, indent=1))
    return 0


def a2(round5: bool = False) -> int:
    from experiment.validation import cavity_reference
    from experiment.validation.cavity_analysis import analyse_run, load_run
    store = json.loads((REMOTE / "store" / "r3_n500_store_analysis.json").read_text(encoding="utf-8"))
    reference = cavity_reference.marchi2021()["extrema"]
    psi_reference = float(reference["psi_min"][0])
    # the particle-pool run: its samples are kept, its snapshots not (psi_min from section 15's table)
    run = load_run("n500_reflection_full_volume", root=POOL_CAVITY_ROOT)
    keep = run["times"] <= 100.0 + 1e-9
    run["times"], run["kinetic_energy"] = run["times"][keep], run["kinetic_energy"][keep]
    run["arrays"] = {name: values[keep] for name, values in run["arrays"].items()}
    pool_analysis = analyse_run(run, average_span=20.0, tolerance=1.0e-3, window=10.0, with_stream_function=False)
    pool = {"window": pool_analysis["window"], "samples_in_window": pool_analysis["samples_in_window"],
            "psi_min": {"wall": POOL_N500_PSI["wall"], "mid": POOL_N500_PSI["mid"]},
            "extrema": {name: {"value": entry["value"]} for name, entry in pool_analysis["extrema"].items()},
            "errors": {f"{frame}_{method}_{line}": {"L2": entry["L2"], "Linf": entry["Linf"]}
                       for (frame, method, line), entry in pool_analysis["errors"].items()}}

    def deviation(value, reference_value):
        return 100.0 * (abs(value) - abs(reference_value)) / abs(reference_value)

    rows = []
    entries = [("壁面独立存储（21877e3，远端）", store), ("粒子池版本（第 15 节 n500_reflection_full_volume）", pool)]
    if round5:
        latest = json.loads((ROUND5_REMOTE / "r5_n500_analysis.json").read_text(encoding="utf-8"))
        entries.insert(0, ("最终缺省（0b92959，远端，第 11 节）", latest))
    for text, entry in entries:
        psi = entry["psi_min"]
        cells = [text, f"{entry['window'][0]:.1f}–{entry['window'][1]:.1f}（{entry['samples_in_window']}）",
                 f"{psi['mid']:.5f}（{deviation(psi['mid'], psi_reference):+.2f} %）",
                 f"{psi['wall']:.5f}（{deviation(psi['wall'], psi_reference):+.2f} %）"]
        for name in ("u_min", "v_max", "v_min"):
            value = entry["extrema"][name]["value"]
            cells.append(f"{value:+.5f}（{deviation(value, float(reference[name][0])):+.2f} %）")
        errors = entry["errors"]
        cells.append(f"{errors['wall_mls_u']['L2']:.4f} / {errors['wall_mls_v']['L2']:.4f}")
        cells.append(f"{errors['mid_dense_u']['L2']:.4f} / {errors['mid_dense_v']['L2']:.4f}")
        rows.append("| " + " | ".join(cells) + " |")
    rows.append(f"| Marchi et al. 2021 | — | {psi_reference:.5f} | {psi_reference:.5f} | "
                + " | ".join(f"{float(reference[name][0]):+.5f}" for name in ("u_min", "v_max", "v_min")) + " | — | — |")
    (TABLES / ("round5_a2.md" if round5 else "round3_a2.md")).write_text(
        "| 2-D 500² 方腔，t = 80–100 平均 | 窗口（样本数） | ψ_min，mid 帧（壁面在盒面） | ψ_min，wall 帧 | u_min | v_max | v_min | "
        "剖面 L2 对 Marchi，wall 帧 MLS：u / v | mid 帧：u / v |\n|---|---|---|---|---|---|---|---|---|\n"
        + "\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    ((ROUND5_REMOTE if round5 else REMOTE) / "a2_summary.json").write_text(
        json.dumps({"store": store, "pool": pool, **({"round5": entries[0][1]} if round5 else {})}, indent=1,
                   default=float), encoding="utf-8")
    print("\n".join(rows))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("a1", "a2"))
    parser.add_argument("--round5", action="store_true")
    arguments = parser.parse_args()
    return a1(arguments.round5) if arguments.action == "a1" else a2(arguments.round5)


if __name__ == "__main__":
    sys.exit(main())
