"""
e8_scaling_figures.py — E8 phase 2, deliverable 2: the scaling figures from e8_p2_tables' tables.json, vector PDF (+ PNG)
in the style of e7_scaling_figures.py (Times 8 pt, ticks in, no top / right spines, elsarticle text width 384 pt).

  strong_scaling.pdf  (a) eta against particles per device (log axis) for the K = 8 strong points (4M .. 64M);
                      (b) eta against K for the 64M cube at K = 2 / 4 / 8.
  weak_scaling.pdf    eta against K for the weak boxes (8M per device, K = 2 / 4 / 8), drawn 0.6 text width wide.

eta = the tables' eta without the defrag events (mean over the trials; the bar is min to max; --eta with uses the
with-defrag values). Both figures shade eta = 1 +- 0.02. The plotted values go to OUT/scaling_figures.csv.

Usage:
    python docs/cluster_v6/scripts/e8_scaling_figures.py --tables OUT/tables.json --out DIR [--eta nodefrag|with]
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e7_scaling_figures as style  # noqa: E402

COLOR_MAIN, COLOR_RED, TEXT_WIDTH, FONT_SIZE = style.COLOR_MAIN, style.COLOR_RED, style.TEXT_WIDTH, style.FONT_SIZE


def rows_of(tables: dict, eta_key: str) -> list:
    rows = []
    for point in tables["points"]:
        stats = point["stats"].get(eta_key) or {}
        if not stats.get("n"):
            continue
        rows.append({"point": point["point"], "case": point["case"], "K": point["K"], "kind": point["kind"], "job": point["job"],
                     "per_gpu": point["per_card"], "particles": point["particles"], "trials": stats["n"], "eta": stats["mean"],
                     "eta_min": stats["min"], "eta_max": stats["max"]})
    return rows


def strong_figure(rows: list, out: pathlib.Path) -> list:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullFormatter
    k8 = sorted((r for r in rows if r["kind"] == "strong" and r["K"] == 8), key=lambda r: r["per_gpu"])
    sweep = sorted((r for r in rows if r["kind"] == "strong" and r["case"] == "cavity3d_n392"), key=lambda r: r["K"])
    width, panel, height_panel = TEXT_WIDTH, 2.05, 1.85
    left, gap, bottom, top = 0.42, 0.62, 0.40, 0.44
    figure = plt.figure(figsize=(width, bottom + height_panel + top))
    axis_n = style.axes_inches(figure, left, bottom, panel, height_panel)
    axis_k = style.axes_inches(figure, left + panel + gap, bottom, panel, height_panel)
    main = dict(color=COLOR_MAIN, marker="o", ms=3.4, mfc=COLOR_MAIN, mec=COLOR_MAIN, lw=0.9)
    for axis in (axis_n, axis_k):
        style.band(axis)
        style.frame(axis)
    for row in k8:
        row["per_gpu_M"] = row["per_gpu"] / 1e6
    if k8:
        style.errorbar(axis_n, k8, "per_gpu_M", label="_nolegend_", **main)
        axis_n.set_xscale("log")
        axis_n.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
        axis_n.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis_n.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=12))
        axis_n.xaxis.set_minor_formatter(NullFormatter())
        xs = [row["per_gpu_M"] for row in k8]
        axis_n.set_xlim(min(xs) / 1.6, max(xs) * 1.6)
        total_axis = axis_n.secondary_xaxis("top", functions=(lambda load: load * 8.0, lambda total: total / 8.0))
        total_axis.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
        total_axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        total_axis.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=12))
        total_axis.xaxis.set_minor_formatter(NullFormatter())
        total_axis.set_xlabel("total particles (M)", labelpad=2.5)
        total_axis.tick_params(which="both", direction="in", labelsize=FONT_SIZE - 0.5)
    low = min([row["eta_min"] for row in k8 + sweep] + [0.98]) - 0.03
    axis_n.set_ylim(max(0.0, low), 1.06)
    axis_n.yaxis.set_major_locator(MultipleLocator(0.1 if low < 0.8 else 0.02))
    axis_n.set_xlabel("particles per device (M)", labelpad=1.5)
    axis_n.set_ylabel(r"$\eta$", labelpad=1.5)
    if sweep:
        style.errorbar(axis_k, sweep, "K", label="_nolegend_", **main)
    axis_k.set_xlim(1.6, 8.4)
    axis_k.xaxis.set_major_locator(MultipleLocator(1))
    values = [row[key] for row in sweep for key in ("eta_min", "eta_max")] + [0.98, 1.02]
    axis_k.set_ylim(min(values) - 0.012, max(values) + 0.012)
    axis_k.yaxis.set_major_locator(MultipleLocator(0.02))
    axis_k.set_xlabel(r"number of devices $K$", labelpad=1.5)
    axis_k.set_ylabel(r"$\eta$", labelpad=1.5)
    style.panel_label(axis_n, "(a)")
    style.panel_label(axis_k, "(b)")
    style.save(figure, out / "strong_scaling.pdf")
    plt.close(figure)
    return [dict(row, figure="strong_scaling", panel="a") for row in k8] + [dict(row, figure="strong_scaling", panel="b") for row in sweep]


def weak_figure(rows: list, out: pathlib.Path) -> list:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    weak = sorted((r for r in rows if r["kind"] == "weak"), key=lambda r: r["K"])
    height_panel, left, right, bottom, top = 1.85, 0.42, 0.10, 0.40, 0.12
    panel = 0.6 * TEXT_WIDTH - left - right
    figure = plt.figure(figsize=(left + panel + right, bottom + height_panel + top))
    axis = style.axes_inches(figure, left, bottom, panel, height_panel)
    style.band(axis)
    style.frame(axis)
    if weak:
        style.errorbar(axis, weak, "K", color=COLOR_MAIN, marker="o", ms=3.2, mfc=COLOR_MAIN, mec=COLOR_MAIN, lw=0.9, label="8M per device")
    values = [row[key] for row in weak for key in ("eta_min", "eta_max")] + [0.98, 1.02]
    axis.set_xlim(1.6, 8.4)
    axis.xaxis.set_major_locator(MultipleLocator(1))
    low, high = min(values) - 0.012, max(style.WEAK_Y_TOP, max(values) + 0.012)
    axis.set_ylim(low, high)
    axis.yaxis.set_major_locator(MultipleLocator(0.02))
    axis.set_xlabel(r"number of devices $K$", labelpad=1.5)
    axis.set_ylabel(r"$\eta$", labelpad=1.5)
    axis.legend(loc="lower left", frameon=False, handlelength=1.4, borderaxespad=0.2)
    style.save(figure, out / "weak_scaling.pdf")
    plt.close(figure)
    return [dict(row, figure="weak_scaling", panel="") for row in weak]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--eta", choices=("nodefrag", "with"), default="nodefrag")
    arguments = parser.parse_args()
    style.setup_style()
    tables = json.loads(pathlib.Path(arguments.tables).read_text(encoding="utf-8"))
    rows = rows_of(tables, "eta" if arguments.eta == "nodefrag" else "eta_with")
    out = pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    plotted = strong_figure(rows, out) + weak_figure(rows, out)
    with (out / "scaling_figures.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["figure", "panel", "point", "case", "K", "kind", "job", "per_gpu", "particles", "trials", "eta", "eta_min", "eta_max"])
        for row in plotted:
            writer.writerow([row["figure"], row["panel"], row["point"], row["case"], row["K"], row["kind"], row["job"], row["per_gpu"],
                             row["particles"], row["trials"], row["eta"], row["eta_min"], row["eta_max"]])
    print(f"[e8_scaling_figures] {len(plotted)} plotted rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
