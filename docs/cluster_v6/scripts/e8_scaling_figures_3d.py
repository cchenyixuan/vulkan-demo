"""
e8_scaling_figures_3d.py — the paper's 3-D scaling figures (user 2026-10-24) from e8_p2_tables' tables.json, drawn with
the 2-D figures' style module (e7_scaling_figures.py: Times 8 pt, ticks in, no top / right spines, panel labels above the
top-left corner, elsarticle text width 384 pt, eta = 1 +- 0.02 band), laid out like the 2-D strong_scaling.pdf /
weak_scaling.pdf so that the two pairs can face each other in the manuscript.

  strong_scaling_3d.pdf  (a) eta against K for the 64M cube (cavity3d_n392) at K = 2 / 4 / 8 (mean over the trials, bar =
                         min..max), the K axis of the 2-D figure (a) (1.6..8.4, ticks every 1), y axis = data +- 0.012 as
                         in 2-D (a);
                         (b) eta against the particles per device (log axis, walls included; the total on a secondary top
                         axis = 8 x the load) for the seven cubes at K = 8 (4M / 8M / 16M / 32M / 64M / 76M / 128M; 4M and 8M
                         five trials, the rest three); the 76M cube (the one cut into 13 whole columns per device) with an
                         open marker, the rest filled; y axis 0..1.06 with ticks every 0.2 as in 2-D (b).
  weak_scaling_3d.pdf    eta against K (2 / 4 / 8) for the weak boxes (8M cube per device), 0.6 text width wide, the colour /
                         marker of the 2-D figure's "8M per device" series; the eta axis is the 2-D weak figure's (derived
                         from its scaling_figures.csv with the same rule, --weak-ylim-from, or given as --weak-ylim LOW,HIGH).

eta = the tables' eta without the defrag events (--eta with: with them). The plotted values go to OUT/strong_scaling_3d.csv
and OUT/weak_scaling_3d.csv (one row per plotted point, the columns of e8_scaling_figures' scaling_figures.csv + marker).

Usage:
    python docs/cluster_v6/scripts/e8_scaling_figures_3d.py --tables docs/cluster_v6/data/e8/phase2/tables.json
        --out ../vulkan-demo/manuscripts/fig --weak-ylim-from ../vulkan-demo-e7/docs/cluster_v6/data/e7/batch2/figures/scaling_figures.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e7_scaling_figures as style  # noqa: E402
from e8_scaling_figures import rows_of  # noqa: E402

COLOR_MAIN, TEXT_WIDTH, FONT_SIZE = style.COLOR_MAIN, style.TEXT_WIDTH, style.FONT_SIZE
SWEEP_CASE = "cavity3d_n392"                 # (a): the 64M cube
OPEN_CASES = ("cavity3d_n416",)              # (b): open marker (76M, 13 whole columns per device)
WEAK_SERIES = next(entry for entry in style.LOADS if entry[1].startswith("8M"))   # the 2-D "8M per device" colour / marker
COLUMNS = ("figure", "panel", "point", "case", "K", "kind", "reference_kind", "job", "per_gpu", "particles", "trials",
           "eta", "eta_min", "eta_max", "marker")


def strong_figure(rows: list, out: pathlib.Path) -> list:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullFormatter
    sweep = sorted((r for r in rows if r["kind"] == "strong" and r["case"] == SWEEP_CASE and 2 <= r["K"] <= 8), key=lambda r: r["K"])
    k8 = sorted((r for r in rows if r["kind"] == "strong" and r["K"] == 8), key=lambda r: r["per_gpu"])
    if not sweep or not k8:
        raise SystemExit(f"strong_scaling_3d: no data (sweep {len(sweep)}, K = 8 points {len(k8)})")
    # the geometry of the 2-D strong_scaling.pdf: two 2.05 x 1.85 in panels, (a) = K sweep, (b) = load sweep
    width, panel, height_panel = TEXT_WIDTH, 2.05, 1.85
    left, gap, bottom, top = 0.42, 0.62, 0.40, 0.44
    figure = plt.figure(figsize=(width, bottom + height_panel + top))
    axis_k = style.axes_inches(figure, left, bottom, panel, height_panel)
    axis_n = style.axes_inches(figure, left + panel + gap, bottom, panel, height_panel)
    main = dict(color=COLOR_MAIN, marker="o", ms=3.4, mfc=COLOR_MAIN, mec=COLOR_MAIN, lw=0.9)
    for axis in (axis_k, axis_n):
        style.band(axis)
        style.frame(axis)
    # (a)
    style.errorbar(axis_k, sweep, "K", label="_nolegend_", **main)
    axis_k.set_xlim(1.6, 8.4)
    axis_k.xaxis.set_major_locator(MultipleLocator(1))
    values = [row[key] for row in sweep for key in ("eta_min", "eta_max")] + [0.98, 1.02]
    axis_k.set_ylim(min(values) - 0.012, max(values) + 0.012)
    axis_k.yaxis.set_major_locator(MultipleLocator(0.02))
    axis_k.set_xlabel(r"number of devices $K$", labelpad=1.5)
    axis_k.set_ylabel(r"$\eta$", labelpad=1.5)
    # (b): the series through all seven cubes (filled), then the open-marker cubes drawn on top
    for row in k8:
        row["per_gpu_M"] = row["per_gpu"] / 1e6
        row["marker"] = "open" if row["case"] in OPEN_CASES else "filled"
    style.errorbar(axis_n, k8, "per_gpu_M", label="_nolegend_", **main)
    open_rows = [row for row in k8 if row["marker"] == "open"]
    if open_rows:
        labels = ", ".join(f"{round(row['particles'] / 1e6):d}M" for row in open_rows)
        axis_n.plot([row["per_gpu_M"] for row in open_rows], [row["eta"] for row in open_rows], linestyle="none", marker="o",
                    ms=3.4, mfc="white", mec=COLOR_MAIN, mew=0.8, label=f"{labels}, 13 columns per device", zorder=4)
        axis_n.legend(loc="lower right", frameon=False, handlelength=1.2, borderaxespad=0.2)
    axis_n.set_xscale("log")
    axis_n.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    axis_n.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis_n.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=12))
    axis_n.xaxis.set_minor_formatter(NullFormatter())
    xs = [row["per_gpu_M"] for row in k8]
    axis_n.set_xlim(min(xs) / 1.6, max(xs) * 1.6)
    axis_n.set_ylim(0.0, 1.06)
    axis_n.yaxis.set_major_locator(MultipleLocator(0.2))
    axis_n.set_xlabel("particles per device (M)", labelpad=1.5)
    axis_n.set_ylabel(r"$\eta$", labelpad=1.5)
    total_axis = axis_n.secondary_xaxis("top", functions=(lambda load: load * 8.0, lambda total: total / 8.0))
    total_axis.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    total_axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    total_axis.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=12))
    total_axis.xaxis.set_minor_formatter(NullFormatter())
    total_axis.set_xlabel("total particles (M)", labelpad=2.5)
    total_axis.tick_params(which="both", direction="in", labelsize=FONT_SIZE - 0.5)
    style.panel_label(axis_k, "(a)")
    style.panel_label(axis_n, "(b)")
    style.save(figure, out / "strong_scaling_3d.pdf")
    plt.close(figure)
    return [dict(row, figure="strong_scaling_3d", panel="a", marker="filled") for row in sweep] + \
           [dict(row, figure="strong_scaling_3d", panel="b") for row in k8]


def weak_axis_range(csv_path: pathlib.Path | None, explicit: str | None) -> tuple:
    """(low, high, legend_eta): the 2-D weak figure's eta axis by e7_scaling_figures' rule (0.012 below the lowest bar,
    top = max(WEAK_Y_TOP, 0.012 above the highest), legend anchored at 0.42 of the data-driven range)."""
    if explicit:
        low, high = (float(part) for part in explicit.split(","))
        return low, high, low + 0.42 * (high - low)
    if csv_path is None:
        return None
    values = [0.98, 1.02]
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("figure") == "weak_scaling":
                values += [float(row["eta_min"]), float(row["eta_max"])]
    if len(values) == 2:
        raise SystemExit(f"{csv_path}: no weak_scaling rows")
    low, high_data = min(values) - 0.012, max(values) + 0.012
    high = max(style.WEAK_Y_TOP, high_data)
    return low, high, low + 0.42 * (high_data - low)


def weak_figure(rows: list, out: pathlib.Path, axis_range: tuple | None) -> list:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    weak = sorted((r for r in rows if r["kind"] == "weak" and 2 <= r["K"] <= 8), key=lambda r: r["K"])
    if not weak:
        raise SystemExit("weak_scaling_3d: no weak points")
    _, label, color, marker = WEAK_SERIES
    height_panel, left, right, bottom, top = 1.85, 0.42, 0.10, 0.40, 0.12
    panel = 0.6 * TEXT_WIDTH - left - right
    figure = plt.figure(figsize=(left + panel + right, bottom + height_panel + top))
    axis = style.axes_inches(figure, left, bottom, panel, height_panel)
    style.band(axis)
    style.frame(axis)
    style.errorbar(axis, weak, "K", color=color, marker=marker, ms=3.2, mfc=color, mec=color, lw=0.9, label=label)
    axis.set_xlim(1.6, 8.4)
    axis.xaxis.set_major_locator(MultipleLocator(1))
    values = [row[key] for row in weak for key in ("eta_min", "eta_max")] + [0.98, 1.02]
    if axis_range is None:
        low, high = min(values) - 0.012, max(style.WEAK_Y_TOP, max(values) + 0.012)
        legend_eta = low + 0.42 * (max(values) + 0.012 - low)
    else:
        low, high, legend_eta = axis_range
        if min(values) < low or max(values) > high:
            raise SystemExit(f"weak_scaling_3d: the data ({min(values):.4f}..{max(values):.4f}) leave the given eta axis ({low:.4f}..{high:.4f})")
    axis.set_ylim(low, high)
    axis.yaxis.set_major_locator(MultipleLocator(0.02))
    axis.set_xlabel(r"number of devices $K$", labelpad=1.5)
    axis.set_ylabel(r"$\eta$", labelpad=1.5)
    axis.legend(loc="center", bbox_to_anchor=(0.5, (legend_eta - low) / (high - low)), frameon=False, handlelength=1.4,
                borderaxespad=0.2)
    style.save(figure, out / "weak_scaling_3d.pdf")
    plt.close(figure)
    print(f"[weak_scaling_3d] eta axis {low:.5f}..{high:.5f}, legend at eta {legend_eta:.4f}")
    return [dict(row, figure="weak_scaling_3d", panel="", marker="filled") for row in weak]


def write_csv(path: pathlib.Path, rows: list) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[csv] {len(rows)} rows -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="E8 3-D strong / weak scaling figures in the 2-D figures' style")
    parser.add_argument("--tables", required=True, help="tables.json of e8_p2_tables.py")
    parser.add_argument("--out", required=True, help="output directory (the manuscript's fig/)")
    parser.add_argument("--eta", choices=("nodefrag", "with"), default="nodefrag")
    parser.add_argument("--weak-ylim-from", default=None, help="the 2-D figures' scaling_figures.csv: weak_scaling_3d takes its eta axis")
    parser.add_argument("--weak-ylim", default=None, help="LOW,HIGH: the eta axis of weak_scaling_3d (instead of --weak-ylim-from)")
    arguments = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    style.setup_style()
    tables = json.loads(pathlib.Path(arguments.tables).read_text(encoding="utf-8"))
    rows = rows_of(tables, "eta" if arguments.eta == "nodefrag" else "eta_with")
    out = pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    axis_range = weak_axis_range(pathlib.Path(arguments.weak_ylim_from) if arguments.weak_ylim_from else None, arguments.weak_ylim)
    strong = strong_figure(rows, out)
    weak = weak_figure(rows, out, axis_range)
    write_csv(out / "strong_scaling_3d.csv", strong)
    write_csv(out / "weak_scaling_3d.csv", weak)
    for row in strong + weak:
        print(f"  {row['figure']} {row['panel'] or '-'} {row['point']:16s} {row['job']:20s} K {row['K']} per device {row['per_gpu'] / 1e6:6.2f}M "
              f"trials {row['trials']} eta {row['eta']:.4f} ({row['eta_min']:.4f}-{row['eta_max']:.4f}) {row['marker']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
