"""
pressure_pairwise_check.py - report section 27, 3.3: numerics.wall_crossing.pressure_wall_evaluation pairwise against
summed (the default) on the 2-D hydrostatic boxes and the 2-D 250^2 cavity at t = 10.

    .venv/Scripts/python.exe -m experiment.wall_crossing.pressure_pairwise_check

Hydrostatic (hydrostatic.py run, t = 10, gravity -y and tilted 30 degrees): the final largest / rms speed (over
sqrt(|g| H)), the speed in the layer next to the wall, the level-free rms pressure error and the pressure level at the
box centre (over rho0 |g| H and in Pa), the mean density of the last sample (series.csv).
Cavity (cavity_run.py --t-end 10 --t-max 10 --average-span 2: from rest to t = 10, the average over 8-10;
experiment/validation/cavity_analysis.analyse_run): psi_min and the three velocity extrema (time-averaged dense
centre lines), the profile errors L2 against Marchi et al. (2021) in the wall frame (MLS), and the two runs'
dense-line profiles against each other (L2 and Linf of the difference). At t = 10 the flow is not steady yet: both
runs are compared at the same time. Writes claude/wall_crossing/tables/pressure_pairwise_*.md and
logs/wall_crossing/section27/pressure_pairwise.json.
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.run_log import LOG_ROOT     # noqa: E402

MODES = (("summed", "累加后合成（现有）"), ("pairwise", "逐对直接计算"))


def last_series_row(path: pathlib.Path) -> dict:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {key: float(value) for key, value in rows[-1].items() if value not in ("", None)}


def main() -> int:
    from experiment.validation.cavity_analysis import analyse_run, load_run
    from experiment.wall_crossing.report_tables import markdown_table, write
    result = {"hydrostatic": {}, "cavity": {}}
    hydro_rows = []
    for gravity, text in (("gy", "重力沿 −y"), ("g30", "重力倾斜 30°")):
        for mode, mode_text in MODES:
            directory = LOG_ROOT / "hydrostatic" / f"s27_2d_n250_{gravity}_{mode}"
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            final = summary["final"]
            series = last_series_row(directory / "series.csv")
            first_layer = final["layers"][0]
            entry = {"speed_maximum_over_scale": final["speed_maximum_over_scale"],
                     "speed_root_mean_square_over_scale": final["speed_root_mean_square_over_scale"],
                     "first_layer_speed_maximum": first_layer["speed_maximum"],
                     "pressure_error_root_mean_square_over_scale": final["pressure_error_root_mean_square_over_scale"],
                     "pressure_level_over_scale": final["pressure_level_over_scale"],
                     "pressure_level_at_centre": final["pressure_level_at_centre"],
                     "density_mean": series.get("density_mean", float("nan")),
                     "fitted_gradient_error_over_rho_g": final["fitted_gradient_error_over_rho_g"],
                     "time": summary["time"], "run_id": summary["run_id"]}
            result["hydrostatic"][f"{gravity}_{mode}"] = entry
            hydro_rows.append([text, mode_text, f"{entry['speed_maximum_over_scale']:.4f}",
                               f"{entry['speed_root_mean_square_over_scale']:.5f}",
                               f"{entry['first_layer_speed_maximum']:.4f}",
                               f"{entry['pressure_error_root_mean_square_over_scale']:.5f}",
                               f"{entry['pressure_level_over_scale']:+.5f}",
                               f"{entry['fitted_gradient_error_over_rho_g']:.4f}",
                               f"{entry['pressure_level_at_centre']:+.1f}", f"{entry['density_mean'] - 1000.0:+.5f}"])
    write("pressure_pairwise_hydrostatic.md", markdown_table(
        ["静水盒（2-D 250²，t = 10）", "壁面压力", "最大速度 / √(gH)", "均方根速度 / √(gH)",
         "离壁第一层的最大速度（m/s）", "压力误差均方根 / ρ0gH（去掉水平）", "盒中心的压力水平 / ρ0gH",
         "拟合的压力梯度误差 / ρ0g", "盒中心的压力水平（Pa）", "最后一个样本的平均密度 − ρ0（kg/m³）"], hydro_rows))

    analyses = {}
    for mode, _ in MODES:
        run = load_run(f"s27_n250_{mode}", root=LOG_ROOT / "cavity")
        analyses[mode] = analyse_run(run, average_span=2.0, tolerance=1.0e-3, window=10.0, with_stream_function=True)
    cavity_rows = []
    previous_path = LOG_ROOT / "section27" / "pressure_pairwise.json"
    previous = (json.loads(previous_path.read_text(encoding="utf-8")).get("cavity", {})
                if previous_path.exists() else {})
    for mode, mode_text in MODES:
        analysis = analyses[mode]
        stream = analysis["stream"] or {}
        if not stream and previous.get(mode, {}).get("psi_min_wall") is not None:
            # the run's snapshots were deleted after the first analysis (C: space): psi_min from that analysis
            stream = {"psi_min_wall": previous[mode]["psi_min_wall"], "psi_min_fluid": previous[mode]["psi_min_fluid"]}
            mode_text += "（ψ_min 取自第一次分析，帧已删）"
        extrema = analysis["extrema"]
        errors = analysis["errors"]
        entry = {"psi_min_wall": stream.get("psi_min_wall"), "psi_min_fluid": stream.get("psi_min_fluid"),
                 "u_min": extrema["u_min"]["value"], "v_max": extrema["v_max"]["value"],
                 "v_min": extrema["v_min"]["value"],
                 "l2_u": errors[("wall", "mls", "u")]["L2"], "l2_v": errors[("wall", "mls", "v")]["L2"],
                 "window": analysis["window"], "samples": analysis["samples_in_window"]}
        result["cavity"][mode] = entry
        cavity_rows.append([mode_text, f"{entry['psi_min_wall']:.6f}", f"{entry['u_min']:.5f}", f"{entry['v_max']:.5f}",
                            f"{entry['v_min']:.5f}", f"{entry['l2_u']:.5f}", f"{entry['l2_v']:.5f}",
                            f"{entry['window'][0]:.1f}–{entry['window'][1]:.1f}（{entry['samples']} 个样本）"])
    summed, pairwise = result["cavity"]["summed"], result["cavity"]["pairwise"]
    cavity_rows.append(["差（逐对 − 现有）", f"{pairwise['psi_min_wall'] - summed['psi_min_wall']:+.6f}"
                        f"（{100 * (pairwise['psi_min_wall'] / summed['psi_min_wall'] - 1):+.2f} %）",
                        f"{pairwise['u_min'] - summed['u_min']:+.5f}", f"{pairwise['v_max'] - summed['v_max']:+.5f}",
                        f"{pairwise['v_min'] - summed['v_min']:+.5f}", f"{pairwise['l2_u'] - summed['l2_u']:+.5f}",
                        f"{pairwise['l2_v'] - summed['l2_v']:+.5f}", ""])
    difference = {}
    for line in ("u", "v"):
        first = analyses["summed"]["statistics"][f"{line}_dense_mls"]["mean"]
        second = analyses["pairwise"]["statistics"][f"{line}_dense_mls"]["mean"]
        delta = second - first
        difference[line] = {"l2": float(np.sqrt(np.mean(delta ** 2))), "linf": float(np.abs(delta).max())}
    result["cavity"]["profile_difference"] = difference
    write("pressure_pairwise_cavity.md", markdown_table(
        ["2-D 250² 方腔，t = 10（8–10 平均）", "ψ_min（wall 帧）", "u_min", "v_max", "v_min",
         "u(y) L2 对 Marchi", "v(x) L2 对 Marchi", "平均窗口"], cavity_rows))
    out = LOG_ROOT / "section27"
    out.mkdir(parents=True, exist_ok=True)
    (out / "pressure_pairwise.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"profile_difference": difference}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
