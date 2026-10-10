"""
_plot_single_baseline_paper.py — paper figures from the single-GPU baseline
campaign (data: logs/single_baseline_20260923_v5fix/, nothing is re-run).

Figure A (Sec. 3.5, V0 only)   manuscripts/fig/single_gpu_v0_throughput.{pdf,png}
    top    particle throughput, million particle-steps per second
    bottom fps
    two lines: 1 and 2 frames in flight; log x with ticks 1M ... 32M; 1M-orig excluded.

Figure B (end of Sec. 4)       manuscripts/fig/v5_single_vs_v0.{pdf,png}
    V5 single-GPU mode / V0 fps ratio at the same in-flight depth (two lines),
    error bars = std of the trial-wise ratio over the 3 interleaved trials.

Serif typography (Times / STIX fallback), vector PDF with embedded TrueType
(fonttype 42), 300 dpi PNG alongside.

Usage:
    .venv/Scripts/python.exe _plot_single_baseline_paper.py
        --dir logs/single_baseline_20260923_v5fix --out-dir manuscripts/fig

v7_single (docs/seam_audit/v7_single.md, measurement C): Figure A redrawn from the v7_single campaign's
S data, written under another name (the old single_gpu_v0_throughput stays):
    .venv/Scripts/python.exe _plot_single_baseline_paper.py --dir logs/v7_single/throughput \
        --solver v7_single --figure-a-name single_gpu_throughput --no-figure-b
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from _summarize_single_baseline import (CURVE_SIZES, aggregate, load_results,  # noqa: E402
                                        trialwise_ratio)

COLOR_V0 = "#2a78d6"
INK = "#000000"
INK_2 = "#333333"
GRID = "#d9d9d9"
LABELED = ("1m", "2m", "4m", "4m_gen", "8m", "16m", "32m")


def size_label(tag: str) -> str:
    """Tick label of a size tag: 4m -> 4M, 4m_gen -> 4M (the generator-built 4M replaces the hand-made one)."""
    return tag.upper().replace("_GEN", "")


def setup_style(font_size: float) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "Nimbus Roman",
                       "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": font_size,
        "axes.labelsize": font_size,
        "legend.fontsize": font_size - 1,
        "xtick.labelsize": font_size - 0.5,
        "ytick.labelsize": font_size - 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.edgecolor": INK_2, "axes.linewidth": 0.6,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "axes.labelcolor": INK,
        "grid.color": GRID, "grid.linewidth": 0.4,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def log_size_axis(ax, sizes: list[str], n_values: dict[str, int]) -> None:
    from matplotlib import ticker
    ax.set_xscale("log")
    labeled = [tag for tag in sizes if tag in LABELED]
    ax.set_xticks([n_values[tag] for tag in labeled])
    ax.set_xticklabels([size_label(tag) for tag in labeled])
    ax.set_xticks([n_values[tag] for tag in sizes if tag not in labeled], minor=True)
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())


def tidy(ax) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)


def figure_a(rows, sizes, n_values, out: pathlib.Path, width: float, height: float, solver: str = "v0") -> None:
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    style = {1: dict(color=COLOR_V0, linestyle=(0, (4, 2)), marker="o", markerfacecolor="white"),
             2: dict(color=COLOR_V0, linestyle="-", marker="o")}
    fig, (ax_tp, ax_fps) = plt.subplots(
        2, 1, figsize=(width, height), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.10})
    for depth in (1, 2):
        xs, tps, fps = [], [], []
        for tag in sizes:
            row = rows.get((tag, solver, depth))
            if row:
                xs.append(n_values[tag]); tps.append(row["mparticle_steps_per_s"]); fps.append(row["fps_mean"])
        label = f"{depth} frame{'s' if depth > 1 else ''} in flight"
        ax_tp.plot(xs, tps, linewidth=1.0, markersize=4, markeredgewidth=0.9, label=label, **style[depth])
        ax_fps.plot(xs, fps, linewidth=1.0, markersize=4, markeredgewidth=0.9, label=label, **style[depth])
    tp_all = [rows[(t, solver, d)]["mparticle_steps_per_s"] for t in sizes for d in (1, 2) if (t, solver, d) in rows]
    ax_tp.set_ylim(min(tp_all) * 0.97, max(tp_all) * 1.02)
    ax_tp.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax_tp.set_ylabel("throughput (M particle-steps / s)")
    ax_tp.grid(True, axis="y")
    ax_tp.legend(frameon=False, loc="lower right", handlelength=2.6)

    ax_fps.set_yscale("log")
    fps_ticks = [10, 20, 50, 100, 200, 500, 1000]
    lo, hi = ax_fps.get_ylim()
    fps_ticks = [t for t in fps_ticks if lo <= t <= hi * 1.05]
    ax_fps.set_yticks(fps_ticks)
    ax_fps.set_yticklabels([str(t) for t in fps_ticks])
    ax_fps.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax_fps.set_ylabel("steps per second (fps)")
    ax_fps.set_xlabel("particles (fluid + wall)")
    ax_fps.grid(True, axis="y")
    log_size_axis(ax_fps, sizes, n_values)
    tidy(ax_tp); tidy(ax_fps)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_a_single_panel(rows, sizes, n_values, out: pathlib.Path, width: float, height: float,
                          solver: str = "v7_single") -> None:
    """Figure 1 of Sec. 3 (v7_single, "§3 补测" item 4): one panel, particle throughput against the particle count
    at 1 and 2 frames in flight, no title; single-column width, every label >= 7 pt at scale 1."""
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    style = {1: dict(color=COLOR_V0, linestyle=(0, (4, 2)), marker="o", markerfacecolor="white"),
             2: dict(color=COLOR_V0, linestyle="-", marker="o")}
    fig, ax = plt.subplots(figsize=(width, height))
    for depth in (1, 2):
        xs, tps = [], []
        for tag in sizes:
            row = rows.get((tag, solver, depth))
            if row:
                xs.append(n_values[tag]); tps.append(row["mparticle_steps_per_s"])
        label = f"{depth} frame{'s' if depth > 1 else ''} in flight"
        ax.plot(xs, tps, linewidth=1.0, markersize=4, markeredgewidth=0.9, label=label, **style[depth])
    tp_all = [rows[(t, solver, d)]["mparticle_steps_per_s"] for t in sizes for d in (1, 2) if (t, solver, d) in rows]
    ax.set_ylim(min(tp_all) * 0.97, max(tp_all) * 1.02)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.set_ylabel("throughput (M particle-steps / s)")
    ax.set_xlabel("particles (fluid + wall)")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, loc="lower right", handlelength=2.6)
    log_size_axis(ax, sizes, n_values)
    tidy(ax)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_b(rows, sizes, n_values, out: pathlib.Path, width: float, height: float) -> None:
    import matplotlib.pyplot as plt
    style = {1: dict(color=INK_2, linestyle=(0, (4, 2)), marker="o", markerfacecolor="white"),
             2: dict(color=INK, linestyle="-", marker="o")}
    fig, ax = plt.subplots(figsize=(width, height))
    lo_all, hi_all = [], []
    for depth in (1, 2):
        xs, ys, es = [], [], []
        for tag in sizes:
            mean, std, n = trialwise_ratio(rows, tag, ("v5", depth), ("v0", depth))
            if n:
                xs.append(n_values[tag]); ys.append(100 * mean); es.append(100 * std)
        lo_all += [y - e for y, e in zip(ys, es)]; hi_all += [y + e for y, e in zip(ys, es)]
        ax.errorbar(xs, ys, yerr=es, linewidth=1.0, markersize=4, markeredgewidth=0.9,
                    capsize=2, elinewidth=0.7,
                    label=f"{depth} frame{'s' if depth > 1 else ''} in flight", **style[depth])
    ax.axhline(100, color=INK_2, linewidth=0.6)
    ax.set_ylim(min(99.0, min(lo_all) - 0.5), max(101.0, max(hi_all) + 0.5))
    ax.set_ylabel("V5 single-GPU mode / V0 (fps, %)")
    ax.set_xlabel("particles (fluid + wall)")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, loc="lower right", handlelength=2.6)
    log_size_axis(ax, sizes, n_values)
    tidy(ax)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="logs/single_baseline_20260923_v5fix")
    parser.add_argument("--out-dir", default="manuscripts/fig")
    parser.add_argument("--font-size", type=float, default=9.0)
    parser.add_argument("--width", type=float, default=3.5, help="inches (single column)")
    parser.add_argument("--solver", default="v0", help="the solver whose rows Figure A plots (v0, v7_single, ...)")
    parser.add_argument("--figure-a-name", default="single_gpu_v0_throughput")
    parser.add_argument("--no-figure-b", action="store_true", help="skip the V5 / V0 ratio figure")
    parser.add_argument("--sizes", default=None, help="comma-separated size tags to plot (default: the 2-D curve)")
    parser.add_argument("--single-panel", action="store_true",
                        help="Figure A as one throughput panel (no fps panel), height = 0.66 x width")
    parser.add_argument("--exclude-sizes", default="",
                        help="comma-separated size tags to leave out (e.g. 4m when 4m_gen replaces it)")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    setup_style(args.font_size)

    ok, failed = load_results(_REPO / args.dir / "results.jsonl")
    rows = aggregate(ok, [])
    wanted = args.sizes.split(",") if args.sizes else [tag for tag in CURVE_SIZES if tag != "3d_8m"]
    excluded = {tag for tag in args.exclude_sizes.split(",") if tag}
    wanted = [tag for tag in wanted if tag not in excluded]
    sizes = [tag for tag in wanted if any(k[0] == tag and k[1] == args.solver for k in rows)]   # excludes 1m_orig
    n_values = {tag: next(rows[k]["particles"] for k in rows if k[0] == tag) for tag in sizes}

    out_dir = _REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.single_panel:
        figure_a_single_panel(rows, sizes, n_values, out_dir / args.figure_a_name, args.width, args.width * 0.66,
                              solver=args.solver)
    else:
        figure_a(rows, sizes, n_values, out_dir / args.figure_a_name, args.width, args.width * 1.25, solver=args.solver)
    if not args.no_figure_b:
        figure_b(rows, sizes, n_values, out_dir / "v5_single_vs_v0", args.width, args.width * 0.72)
    print(f"[paper_fig] {len(ok)} ok runs ({len(failed)} failed) -> {out_dir / (args.figure_a_name + '.pdf')}"
          + ("" if args.no_figure_b else f", {out_dir / 'v5_single_vs_v0.pdf'}") + " (+ .png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
