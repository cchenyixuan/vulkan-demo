"""
e7_scaling_figures.py — E7 batch 2 (fourth supplement, user 2026-10-15): the paper's scaling figures from the tables
JSON of e7_batch2_tables.py, vector PDF (+ PNG), in the serif style of the cavity_re1000 figure
(experiment/validation/cavity_re1000_figure.py on the v7-rc3 branch: Times 8 pt, ticks in, no top / right spines,
panel labels above the top-left corner, elsarticle text width 384 pt).

  strong_scaling.pdf  (a) eta against K: 2d_n8000 at K = 2..8, and 2d_n11320 at K = 8 (another marker; its reference
                      is K = 2 pairs of the case); (b) eta against particles per GPU (log axis): every strong K = 8 point
                      (2d_n11320 with the same marker as in (a)).
  weak_scaling.pdf    eta against K (2..8) for the three per-GPU loads (the k members of n1440 / n2840 / n5640).

eta = the tables' eta without the defrag events (--eta nodefrag, default) or with them (--eta with); the marker is
the mean over the trials, the bar the minimum to the maximum. Both figures shade eta = 1 +- 0.02. Points of every
node share one marker; the node is in the report only. Candidates: arm C, the --window (default 1500/500), jobs not in
--exclude-job; of several candidates of one (case, K) the one with the most trials wins, then the one listed later
in the tables (the later job). The plotted values go to <out>/scaling_figures.csv.

Usage:
    python docs/cluster_v6/scripts/e7_scaling_figures.py --tables tables_b2.json --out DIR
        [--exclude-job e7_b2T_1695731 ...] [--eta nodefrag|with] [--window 1500/500]
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys

FONT_SIZE = 8.0
TEXT_WIDTH = 384.0 / 72.0               # elsarticle preprint / 1p text width (as cavity_re1000)
COLOR_MAIN = "#08519c"
COLOR_LIGHT = "#6baed6"
COLOR_RED = "#b2182b"
BAND_COLOR = "#c6dbef"
LOADS = (("cavity2d_n1440", r"2.1M per device", COLOR_MAIN, "o"),
         ("cavity2d_n2840", r"8.1M per device", COLOR_LIGHT, "s"),
         ("cavity2d_n5640", r"32.0M per device", COLOR_RED, "^"))


def setup_style() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE - 1.5, "xtick.labelsize": FONT_SIZE - 0.5, "ytick.labelsize": FONT_SIZE - 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.edgecolor": "#333333", "axes.linewidth": 0.6,
        "xtick.color": "#333333", "ytick.color": "#333333", "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.minor.size": 1.6, "ytick.minor.size": 1.6, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "lines.linewidth": 1.0, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.bbox": None,
    })


def axes_inches(figure, left: float, bottom: float, width: float, height: float):
    figure_width, figure_height = figure.get_size_inches()
    return figure.add_axes([left / figure_width, bottom / figure_height, width / figure_width, height / figure_height])


def panel_label(axis, text: str) -> None:
    axis.annotate(text, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(-2.0, 5.0), textcoords="offset points",
                  ha="right", va="bottom", fontsize=FONT_SIZE + 0.5)


def frame(axis) -> None:
    axis.tick_params(which="both", top=False, right=False)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def band(axis) -> None:
    axis.axhspan(0.98, 1.02, color=BAND_COLOR, alpha=0.55, lw=0, zorder=0)
    axis.axhline(1.0, color="#777777", lw=0.4, zorder=0.5)


def select(points: list, exclude: set, window: str, eta_key: str) -> dict:
    """(case, K) -> the chosen point's numbers (mean, min, max of eta, trials, job, host, per-GPU particles, reference)."""
    chosen = {}
    for order, point in enumerate(points):
        if point["arm"] != "C" or point["job"] in exclude or window not in point.get("windows", []):
            continue
        for name, reference in point["references"].items():
            stats = reference.get(eta_key) or {}
            if not stats.get("n"):
                continue
            entry = {"case": point["case"], "K": point["K"], "point": point["point"], "job": point["job"],
                     "host": point["host"], "kind": reference["kind"], "reference": name,
                     "per_gpu": point["particles"] / point["K"], "particles": point["particles"],
                     "trials": stats["n"], "eta": stats["mean"], "eta_min": stats["min"], "eta_max": stats["max"],
                     "order": order}
            key = (point["case"], point["K"])
            if key not in chosen or (entry["trials"], entry["order"]) > (chosen[key]["trials"], chosen[key]["order"]):
                chosen[key] = entry
    return chosen


def errorbar(axis, rows: list, x_key: str, **style) -> None:
    xs = [row[x_key] for row in rows]
    ys = [row["eta"] for row in rows]
    low = [row["eta"] - row["eta_min"] for row in rows]
    high = [row["eta_max"] - row["eta"] for row in rows]
    axis.errorbar(xs, ys, yerr=[low, high], capsize=1.6, elinewidth=0.6, capthick=0.6, **style)


def strong_figure(chosen: dict, out: pathlib.Path) -> list:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullFormatter
    sweep = sorted((row for (case, slabs), row in chosen.items()
                    if case == "cavity2d_n8000" and 2 <= slabs <= 8 and row["kind"] == "strong"), key=lambda r: r["K"])
    largest = [row for (case, slabs), row in chosen.items() if case == "cavity2d_n11320" and slabs == 8]
    k8 = sorted((row for (case, slabs), row in chosen.items()
                 if slabs == 8 and row["kind"] == "strong" and not re.search(r"_k\d+$", case)), key=lambda r: r["per_gpu"])
    width, panel, height_panel = TEXT_WIDTH, 2.05, 1.85
    left, gap, bottom, top = 0.42, 0.62, 0.40, 0.20
    figure = plt.figure(figsize=(width, bottom + height_panel + top))
    axis_k = axes_inches(figure, left, bottom, panel, height_panel)
    axis_n = axes_inches(figure, left + panel + gap, bottom, panel, height_panel)
    main = dict(color=COLOR_MAIN, marker="o", ms=3.4, mfc=COLOR_MAIN, mec=COLOR_MAIN, lw=0.9)
    other = dict(color=COLOR_RED, marker="s", ms=3.4, mfc="none", mec=COLOR_RED, mew=0.8, ls="none")
    for axis in (axis_k, axis_n):
        band(axis)
        frame(axis)
    # labels (user 2026-10-20): "64M" / "128M, pair reference" in (a); (b) keeps only the pair-reference legend entry and
    # reads particles per device in M with plain 0.1 / 1 / 10 ticks (as loss_sources.pdf (a)); "GPU" -> "device"
    errorbar(axis_k, sweep, "K", label=f"{int(sweep[0]['particles'] // 1e6)}M" if sweep else None, **main)
    if largest:
        errorbar(axis_k, largest, "K", label=f"{int(largest[0]['particles'] // 1e6)}M, pair reference", **other)
    axis_k.set_xlim(1.6, 8.4)
    axis_k.xaxis.set_major_locator(MultipleLocator(1))
    values = [row[key] for row in sweep + largest for key in ("eta_min", "eta_max")] + [0.98, 1.02]
    axis_k.set_ylim(min(values) - 0.012, max(values) + 0.012)
    axis_k.yaxis.set_major_locator(MultipleLocator(0.02))
    axis_k.set_xlabel(r"number of devices $K$", labelpad=1.5)
    axis_k.set_ylabel(r"$\eta$", labelpad=1.5)
    axis_k.legend(loc="lower left", frameon=False, handlelength=1.4, borderaxespad=0.2)
    for row in k8 + largest:
        row["per_gpu_M"] = row["per_gpu"] / 1e6
    errorbar(axis_n, k8, "per_gpu_M", label="_nolegend_", **main)
    if largest:
        errorbar(axis_n, largest, "per_gpu_M", label=f"{int(largest[0]['particles'] // 1e6)}M, pair reference", **other)
    axis_n.set_xscale("log")
    axis_n.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    axis_n.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis_n.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=12))
    axis_n.xaxis.set_minor_formatter(NullFormatter())
    xs = [row["per_gpu_M"] for row in k8 + largest]
    axis_n.set_xlim(min(xs) / 1.6, max(xs) * 1.6)
    axis_n.set_ylim(0.0, 1.06)
    axis_n.yaxis.set_major_locator(MultipleLocator(0.2))
    axis_n.set_xlabel("particles per device (M)", labelpad=1.5)
    axis_n.set_ylabel(r"$\eta$", labelpad=1.5)
    axis_n.legend(loc="lower right", frameon=False, handlelength=1.4, borderaxespad=0.2)
    panel_label(axis_k, "(a)")
    panel_label(axis_n, "(b)")
    save(figure, out / "strong_scaling.pdf")
    plt.close(figure)
    return [dict(row, figure="strong_scaling", panel="a") for row in sweep + largest] + \
           [dict(row, figure="strong_scaling", panel="b") for row in k8 + largest]


def weak_figure(chosen: dict, out: pathlib.Path) -> list:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    panel, height_panel, left, right, bottom, top = 2.05, 1.85, 0.42, 0.10, 0.40, 0.12
    figure = plt.figure(figsize=(left + panel + right, bottom + height_panel + top))
    axis = axes_inches(figure, left, bottom, panel, height_panel)
    band(axis)
    frame(axis)
    rows_all, values = [], [0.98, 1.02]
    for reference_case, label, color, marker in LOADS:
        rows = sorted((row for (case, slabs), row in chosen.items()
                       if row["kind"] == "weak" and row["reference"] == f"weak:{reference_case}" and 2 <= slabs <= 8),
                      key=lambda r: r["K"])
        if not rows:
            continue
        errorbar(axis, rows, "K", color=color, marker=marker, ms=3.2, mfc=color, mec=color, lw=0.9, label=label)
        rows_all += [dict(row, figure="weak_scaling", panel="") for row in rows]
        values += [row[key] for row in rows for key in ("eta_min", "eta_max")]
    axis.set_xlim(1.6, 8.4)
    axis.xaxis.set_major_locator(MultipleLocator(1))
    axis.set_ylim(min(values) - 0.012, max(values) + 0.012)
    axis.yaxis.set_major_locator(MultipleLocator(0.02))
    axis.set_xlabel(r"number of devices $K$", labelpad=1.5)
    axis.set_ylabel(r"$\eta$", labelpad=1.5)
    # in the empty strip between the 2.1M series (eta ~0.91-0.94) and the band (lower left covered the 2.1M points)
    axis.legend(loc="center", bbox_to_anchor=(0.5, 0.42), frameon=False, handlelength=1.4, borderaxespad=0.2)
    save(figure, out / "weak_scaling.pdf")
    plt.close(figure)
    return rows_all


def save(figure, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    renderer = figure.canvas.get_renderer()
    box = figure.get_tightbbox(renderer)
    width, height = figure.get_size_inches()
    if box.x0 < -1e-3 or box.y0 < -1e-3 or box.x1 > width + 1e-3 or box.y1 > height + 1e-3:
        raise RuntimeError(f"{path.name}: artists outside the page: {box} vs {width:.3f} x {height:.3f} in")
    figure.savefig(path, dpi=600)
    figure.savefig(path.with_suffix(".png"), dpi=300)
    print(f"[figure] wrote {path} (+ .png), {width:.3f} x {height:.3f} in", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="E7 strong / weak scaling figures (vector PDF)")
    parser.add_argument("--tables", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--exclude-job", action="append", default=[])
    parser.add_argument("--eta", choices=("nodefrag", "with"), default="nodefrag")
    parser.add_argument("--window", default="1500/500")
    arguments = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    setup_style()
    tables = json.loads(pathlib.Path(arguments.tables).read_text(encoding="utf-8"))
    chosen = select(tables["points"], set(arguments.exclude_job), arguments.window,
                    "eta_nodefrag" if arguments.eta == "nodefrag" else "eta")
    out = pathlib.Path(arguments.out)
    rows = strong_figure(chosen, out) + weak_figure(chosen, out)
    columns = ("figure", "panel", "case", "K", "point", "job", "host", "reference", "particles", "per_gpu", "trials",
               "eta", "eta_min", "eta_max")
    with open(out / "scaling_figures.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"eta: {arguments.eta}; {len(rows)} plotted rows -> {out / 'scaling_figures.csv'}")
    for row in rows:
        print(f"  {row['figure']} {row['panel'] or '-'} {row['point']:18s} {row['job']:18s} {row['host']} "
              f"trials {row['trials']} eta {row['eta']:.3f} ({row['eta_min']:.3f}-{row['eta_max']:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
