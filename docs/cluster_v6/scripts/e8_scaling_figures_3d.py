"""
e8_scaling_figures_3d.py — the manuscript's 3-D scaling figure (user 2026-10-24, second version: one figure, the 3-D
series beside the 2-D ones) from e8_p2_tables' tables.json and the 2-D figure's scaling_figures.csv, drawn with the 2-D
figures' style module (e7_scaling_figures.py: Times 8 pt, ticks in, no top / right spines, panel labels above the
top-left corner, elsarticle text width 384 pt, eta = 1 +- 0.02 band) in the geometry of the 2-D strong_scaling.pdf.

  strong_scaling_3d.pdf  (a) eta against K: the 64M cube at K = 2 / 4 / 8 (filled circles, the main colour), the weak
                         boxes of 8M per device at K = 2 / 4 / 8 (filled squares, the 2-D weak figure's "8M per device"
                         colour) and the 2-D 64M cavity at K = 2..8 (grey thin line, small dots; the 2-D figure's panel
                         (a) data); eta axis 0.80..1.04, legend lower right.
                         (b) eta against the particles per device (log axis, walls included; the total on the top axis
                         = 8 x the load) for the seven cubes at K = 8 (filled circles; 76M open) and the nine 2-D squares
                         at K = 8 (grey thin line, small dots; the 2-D panel (b) data, the 128M pair-reference point
                         included); the x axis reaches down to the 2-D 0.04M point; legend "3D", "2D", "3D, 76M"
                         (third version, user 2026-10-24: no column labels beside the points; the (a) legend reads
                         "3D strong, 64M", "3D weak, 8M per device", "2D strong, 64M").
  The marker is the mean over the trials, the bar the minimum to the maximum (3-D: 4M and 8M five trials, the rest three).

eta = the tables' eta without the defrag events (--eta with: with them). The plotted values go to OUT/strong_scaling_3d.csv
(one row per plotted point of both panels and all series: the columns of e8_scaling_figures' scaling_figures.csv +
series, marker, fluid_columns_per_device, columns_label).

Usage:
    python docs/cluster_v6/scripts/e8_scaling_figures_3d.py --tables docs/cluster_v6/data/e8/phase2/tables.json
        --two-d-csv ../vulkan-demo-e7/docs/cluster_v6/data/e7/batch2/figures/scaling_figures.csv --out ../vulkan-demo/manuscripts/fig
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
COLOR_WEAK = next(entry for entry in style.LOADS if entry[1].startswith("8M"))[2]     # the 2-D weak figure's 8M series
COLOR_2D = "#8c8c8c"
SWEEP_CASE = "cavity3d_n392"                 # (a): the 64M cube
SWEEP_CASE_2D = "cavity2d_n8000"             # (a): the 2-D 64M cavity
OPEN_CASES = ("cavity3d_n416",)              # (b): open marker (76M, 13 whole columns per device)
# fluid columns of the cubes (the cases' generate.txt: 38 / 48 / 61 / 78 / 98 / 104 / 124 fluid columns at h/dx = 4) and their rounded
# per-device count (5 / 6 / 8 / 10 / 12 / 13 / 16): CSV columns only (the second version wrote them beside the points)
FLUID_COLUMNS = {"cavity3d_n152": 38, "cavity3d_n192": 48, "cavity3d_n244": 61, "cavity3d_n312": 78,
                 "cavity3d_n392": 98, "cavity3d_n416": 104, "cavity3d_n496": 124}
COLUMN_LABELS = {"cavity3d_n152": "5", "cavity3d_n192": "6", "cavity3d_n244": "8", "cavity3d_n312": "10",
                 "cavity3d_n392": "12", "cavity3d_n416": "13", "cavity3d_n496": "16"}
COLUMNS = ("figure", "panel", "series", "point", "case", "K", "kind", "reference_kind", "job", "per_gpu", "particles",
           "trials", "eta", "eta_min", "eta_max", "marker", "fluid_columns_per_device", "columns_label")


def two_d_rows(csv_path: pathlib.Path) -> list:
    """The 2-D figure's plotted points (its scaling_figures.csv) in the row layout of rows_of."""
    rows = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["figure"] != "strong_scaling":
                continue
            reference = row["reference"]
            rows.append({"point": row["point"], "case": row["case"], "K": int(row["K"]), "kind": "strong",
                         "reference_kind": "pairs" if reference.startswith("pairs:") else "k1", "job": row["job"],
                         "per_gpu": float(row["per_gpu"]), "particles": int(row["particles"]), "trials": int(row["trials"]),
                         "eta": float(row["eta"]), "eta_min": float(row["eta_min"]), "eta_max": float(row["eta_max"]),
                         "panel_2d": row["panel"]})
    if not rows:
        raise SystemExit(f"{csv_path}: no strong_scaling rows")
    return rows


def grey_style() -> dict:
    return dict(color=COLOR_2D, marker="o", ms=2.0, mfc=COLOR_2D, mec=COLOR_2D, lw=0.6)


def strong_figure(rows: list, rows_2d: list, out: pathlib.Path) -> list:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullFormatter
    sweep = sorted((r for r in rows if r["kind"] == "strong" and r["case"] == SWEEP_CASE and 2 <= r["K"] <= 8), key=lambda r: r["K"])
    weak = sorted((r for r in rows if r["kind"] == "weak" and 2 <= r["K"] <= 8), key=lambda r: r["K"])
    k8 = sorted((r for r in rows if r["kind"] == "strong" and r["K"] == 8), key=lambda r: r["per_gpu"])
    sweep_2d = sorted((r for r in rows_2d if r["panel_2d"] == "a" and r["case"] == SWEEP_CASE_2D and 2 <= r["K"] <= 8), key=lambda r: r["K"])
    k8_2d = sorted((r for r in rows_2d if r["panel_2d"] == "b" and r["K"] == 8), key=lambda r: r["per_gpu"])
    for name, series in (("3-D sweep", sweep), ("3-D weak", weak), ("3-D K = 8", k8), ("2-D sweep", sweep_2d), ("2-D K = 8", k8_2d)):
        if not series:
            raise SystemExit(f"strong_scaling_3d: no data for the {name} series")
    # the geometry of the 2-D strong_scaling.pdf: two 2.05 x 1.85 in panels, (a) = K sweep, (b) = load sweep
    width, panel, height_panel = TEXT_WIDTH, 2.05, 1.85
    left, gap, bottom, top = 0.42, 0.62, 0.40, 0.44
    figure = plt.figure(figsize=(width, bottom + height_panel + top))
    axis_k = style.axes_inches(figure, left, bottom, panel, height_panel)
    axis_n = style.axes_inches(figure, left + panel + gap, bottom, panel, height_panel)
    main = dict(color=COLOR_MAIN, marker="o", ms=3.4, mfc=COLOR_MAIN, mec=COLOR_MAIN, lw=0.9)
    weak_style = dict(color=COLOR_WEAK, marker="s", ms=3.2, mfc=COLOR_WEAK, mec=COLOR_WEAK, lw=0.9)
    for axis in (axis_k, axis_n):
        style.band(axis)
        style.frame(axis)
    # (a): the 2-D sweep first (behind), then the 3-D weak boxes and the 3-D 64M cube
    style.errorbar(axis_k, sweep_2d, "K", label="2D strong, 64M", zorder=2, **grey_style())
    style.errorbar(axis_k, weak, "K", label="3D weak, 8M per device", zorder=3, **weak_style)
    style.errorbar(axis_k, sweep, "K", label="3D strong, 64M", zorder=4, **main)
    axis_k.set_xlim(1.6, 8.4)
    axis_k.xaxis.set_major_locator(MultipleLocator(1))
    axis_k.set_ylim(0.80, 1.04)
    axis_k.yaxis.set_major_locator(MultipleLocator(0.02))
    axis_k.set_xlabel(r"number of devices $K$", labelpad=1.5)
    axis_k.set_ylabel(r"$\eta$", labelpad=1.5)
    handles, labels = axis_k.get_legend_handles_labels()
    order = [labels.index(name) for name in ("3D strong, 64M", "3D weak, 8M per device", "2D strong, 64M")]
    axis_k.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower right", frameon=False, ncol=2,
                  handlelength=1.4, columnspacing=0.8, labelspacing=0.25, borderpad=0.2, borderaxespad=0.2)
    # (b): the 2-D squares (grey, behind), the seven cubes (filled; the open-marker cubes drawn on top)
    for row in k8 + k8_2d:
        row["per_gpu_M"] = row["per_gpu"] / 1e6
    for row in k8:
        row["marker"] = "open" if row["case"] in OPEN_CASES else "filled"
    style.errorbar(axis_n, k8_2d, "per_gpu_M", label="2D", zorder=2, **grey_style())
    style.errorbar(axis_n, k8, "per_gpu_M", label="3D", zorder=3, **main)
    open_rows = [row for row in k8 if row["marker"] == "open"]
    if open_rows:
        axis_n.plot([row["per_gpu_M"] for row in open_rows], [row["eta"] for row in open_rows], linestyle="none", marker="o",
                    ms=3.4, mfc="white", mec=COLOR_MAIN, mew=0.8,
                    label="3D, " + ", ".join(f"{round(row['particles'] / 1e6):d}M" for row in open_rows), zorder=4)
    axis_n.set_xscale("log")
    axis_n.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    axis_n.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis_n.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=12))
    axis_n.xaxis.set_minor_formatter(NullFormatter())
    xs = [row["per_gpu_M"] for row in k8 + k8_2d]
    axis_n.set_xlim(min(xs) / 1.6, max(xs) * 1.6)
    axis_n.set_ylim(0.0, 1.06)
    axis_n.yaxis.set_major_locator(MultipleLocator(0.2))
    axis_n.set_xlabel("particles per device (M)", labelpad=1.5)
    axis_n.set_ylabel(r"$\eta$", labelpad=1.5)
    handles, labels = axis_n.get_legend_handles_labels()
    order = [labels.index(name) for name in ("3D", "2D") if name in labels] + \
            [i for i, name in enumerate(labels) if name not in ("3D", "2D")]
    axis_n.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower right", frameon=False, handlelength=1.4,
                  borderaxespad=0.2)
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

    def tagged(series: list, panel_name: str, series_name: str, marker: str = "filled") -> list:
        result = []
        for row in series:
            columns = FLUID_COLUMNS.get(row["case"])
            result.append(dict(row, figure="strong_scaling_3d", panel=panel_name, series=series_name, marker=row.get("marker", marker),
                               fluid_columns_per_device=(f"{columns / 8.0:.2f}" if columns and row["K"] == 8 else ""),
                               columns_label=COLUMN_LABELS.get(row["case"], "") if row["K"] == 8 and panel_name == "b" else ""))
        return result
    return (tagged(sweep, "a", "3D cube 64M") + tagged(weak, "a", "3D weak 8M per device") + tagged(sweep_2d, "a", "2D square 64M", "dot") +
            tagged(k8, "b", "3D cubes") + tagged(k8_2d, "b", "2D squares", "dot"))


def main() -> int:
    parser = argparse.ArgumentParser(description="E8 3-D scaling figure beside the 2-D series, in the 2-D figures' style")
    parser.add_argument("--tables", required=True, help="tables.json of e8_p2_tables.py")
    parser.add_argument("--two-d-csv", required=True, help="the 2-D figures' scaling_figures.csv (e7_scaling_figures.py)")
    parser.add_argument("--out", required=True, help="output directory (the manuscript's fig/)")
    parser.add_argument("--eta", choices=("nodefrag", "with"), default="nodefrag")
    arguments = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    style.setup_style()
    tables = json.loads(pathlib.Path(arguments.tables).read_text(encoding="utf-8"))
    rows = rows_of(tables, "eta" if arguments.eta == "nodefrag" else "eta_with")
    rows_2d = two_d_rows(pathlib.Path(arguments.two_d_csv))
    out = pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    plotted = strong_figure(rows, rows_2d, out)
    with (out / "strong_scaling_3d.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(plotted)
    print(f"[csv] {len(plotted)} rows -> {out / 'strong_scaling_3d.csv'}")
    for row in plotted:
        print(f"  {row['panel']} {row['series']:24s} {row['point']:16s} K {row['K']} per device {row['per_gpu'] / 1e6:6.2f}M trials {row['trials']} "
              f"eta {row['eta']:.4f} ({row['eta_min']:.4f}-{row['eta_max']:.4f}) {row['marker']} {row['columns_label']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
