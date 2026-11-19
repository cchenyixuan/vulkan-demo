"""
volume_integral_check.py - report section 24 A: the volume constraint with and without the integral term
(numerics.volume_constraint.integral), from box_run runs of the same case.

    .venv/Scripts/python.exe -m experiment.wall_crossing.volume_integral_check --name n250 \
        --run s24_n250_proportional "比例" --run s24_n250_integral "比例 + 积分" [--window 5 10]

Per run: the fluid's mean density (series.csv: float64 mean of the true density), theta, rate and I (the
constraint's log); in the window: the mean density offset (mean - rho0), theta's mean and standard deviation, I at the
end, and the number of sign changes of theta (an oscillation of the controller would show as long runs of one sign
alternating with the other at a period ~ 2 pi sqrt(tau T_I); noise shows as frequent sign changes). Writes
claude/wall_crossing/tables/volume_integral_<name>.md and claude/wall_crossing/figures/volume_integral_<name>.png.
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

REST_DENSITY = 1000.0
COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiment.wall_crossing.report_figures import INK_PRIMARY, INK_SECONDARY, save, setup_style
    from experiment.wall_crossing.report_tables import markdown_table, write
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--run", nargs=2, action="append", required=True, metavar=("RUN", "TEXT"))
    parser.add_argument("--window", nargs=2, type=float, default=(5.0, 10.0))
    arguments = parser.parse_args()
    window = tuple(arguments.window)
    rows, plotted = [], []
    for name, text in arguments.run:
        directory = LOG_ROOT / "box_run" / name
        with open(directory / "series.csv", newline="", encoding="utf-8") as handle:
            series = list(csv.DictReader(handle))
        time = np.array([float(row["time"]) for row in series])
        density = np.array([float(row["density_mean"]) for row in series])
        records = [json.loads(line) for line in (directory / "volume_constraint.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip() and "theta" in line]
        record_time = np.array([record["time"] for record in records])
        theta = np.array([record["theta"] for record in records])
        integral = np.array([record.get("integral", np.nan) for record in records])
        inside = (time >= window[0] - 1e-9) & (time <= window[1] + 1e-9)
        inside_records = (record_time >= window[0] - 1e-9) & (record_time <= window[1] + 1e-9)
        signs = np.sign(theta[inside_records])
        changes = int(np.sum(signs[1:] != signs[:-1]))
        rows.append([text, f"{np.mean(density[inside]) - REST_DENSITY:+.5f}",
                     f"{np.mean(theta[inside_records]):+.2e}", f"{np.std(theta[inside_records]):.2e}",
                     f"{changes} / {int(inside_records.sum()) - 1}",
                     f"{integral[-1]:+.2e}" if np.isfinite(integral[-1]) else "—",
                     f"{density[-1] - REST_DENSITY:+.5f}"])
        plotted.append((text, time, density, record_time, theta, integral))
    write(f"volume_integral_{arguments.name}.md", markdown_table(
        ["修正", f"t = {window[0]:g}–{window[1]:g} 平均密度 − ρ0（kg/m³）", "θ 平均", "θ 标准差",
         "θ 变号次数 / 相邻更新对数", "最后的 I", "最后的平均密度 − ρ0"], rows))
    setup_style()
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    for index, (text, time, density, record_time, theta, integral) in enumerate(plotted):
        colour = COLOURS[index]
        axes[0].plot(time, density - REST_DENSITY, color=colour, linewidth=1.2, label=text)
        axes[1].plot(record_time, theta, color=colour, linewidth=1.2, marker="o", markersize=2.5)
        if np.isfinite(integral).any():
            axes[2].plot(record_time, integral, color=colour, linewidth=1.2, marker="o", markersize=2.5)
    axes[0].set_title("平均密度 − ρ0（kg/m³）")
    axes[1].set_title("θ（每次更新）")
    axes[2].set_title("积分项 I")
    for axis in axes:
        axis.set_xlabel("时间（模拟单位）")
        axis.axvspan(*window, color=INK_SECONDARY, alpha=0.08, linewidth=0)
    handles, labels = axes[0].get_legend_handles_labels()
    height = figure.get_figheight()
    figure.suptitle(f"总体积修正：比例与比例 + 积分（{arguments.name}，灰带 = 统计窗口）", x=0.01, y=1.0 - 0.05 / height,
                    ha="left", va="top", fontsize=12, fontweight="bold", color=INK_PRIMARY)
    figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0 - 0.40 / height), ncol=len(labels),
                  handlelength=3.0, labelcolor=INK_PRIMARY, borderaxespad=0.0)
    figure.tight_layout(rect=(0, 0, 1.0, 1.0 - 0.55 / height), w_pad=2.0)
    save(figure, REPOSITORY_ROOT / "claude" / "wall_crossing" / "figures" / f"volume_integral_{arguments.name}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
