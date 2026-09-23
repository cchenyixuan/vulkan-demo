"""
_summarize_single_baseline.py — tables + figure for the single-GPU baseline
campaign (_run_single_baseline_campaign.py).

Reads <dir>/results.jsonl (+ <dir>/telemetry.csv for the clock/power/temp
covariates) and writes:
  <dir>/summary.md       per-size fps (mean ± std over trials, n), ratio tables
                         (V5/V0 at each in-flight depth, depth-2 gain per solver),
                         covariate table
  <dir>/summary.csv      the per-(size, solver, in_flight) aggregate rows
  <dir>/single_gpu_baseline.png   three stacked panels: fps, particle
                         throughput, V5/V0 ratio (all vs N, log x)
Optionally copies the figure to --docs-png.

Ratios are computed trial-wise (trial t's V5 over trial t's V0, both measured
back-to-back on the same card) and reported as mean ± std over trials.

Usage:
    .venv/Scripts/python.exe _summarize_single_baseline.py
        --dir logs/single_baseline_20260923 --docs-png docs/single_gpu_baseline_v0_vs_v5.png
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import pathlib
import shutil
import statistics
import sys

_REPO = pathlib.Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from _run_single_baseline_campaign import SIZES  # noqa: E402  (size order)

SOLVER_LABEL = {"v0": "V0 reference (single-GPU solver)",
                "v5": "V5 multi-GPU solver, single-GPU mode"}
SOLVER_SHORT = {"v0": "V0", "v5": "V5"}
CURVE_SIZES = [tag for tag, _, _ in SIZES if tag != "1m_orig"]

# Validated default categorical palette (dataviz skill, slots 1-2) + chrome.
COLOR_V0 = "#2a78d6"
COLOR_V5 = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def load_results(path: pathlib.Path) -> tuple[list[dict], list[dict]]:
    ok, failed = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        (ok if record.get("status") == "ok" else failed).append(record)
    return ok, failed


def load_telemetry(path: pathlib.Path, gpu_index: int) -> list[tuple[float, float, float, float]]:
    """(epoch, sm_mhz, power_w, temp_c) for one nvidia-smi index."""
    samples = []
    if not path.exists():
        return samples
    for line in path.read_text(errors="replace").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 8 or parts[0].startswith("timestamp"):
            continue
        try:
            if int(parts[1]) != gpu_index:
                continue
            stamp = datetime.datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f")
            samples.append((stamp.timestamp(),
                            float(parts[2].split()[0]),
                            float(parts[4].split()[0]),
                            float(parts[5].split()[0])))
        except (ValueError, IndexError):
            continue
    samples.sort()
    return samples


def covariates(samples, epoch_start: float, epoch_end: float) -> dict:
    window = [s for s in samples if epoch_start <= s[0] <= epoch_end]
    if not window:
        return {"sm_mhz": None, "power_w": None, "temp_c": None, "n_samples": 0}
    return {
        "sm_mhz": statistics.fmean(s[1] for s in window),
        "power_w": statistics.fmean(s[2] for s in window),
        "temp_c": max(s[3] for s in window),
        "n_samples": len(window),
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def aggregate(ok: list[dict], samples) -> dict[tuple, dict]:
    groups: dict[tuple, list[dict]] = {}
    for record in ok:
        groups.setdefault((record["size"], record["solver"], record["in_flight"]), []).append(record)
    rows = {}
    for key, records in groups.items():
        fps_values = [r["fps"] for r in records]
        mean, std = mean_std(fps_values)
        cov = [covariates(samples, r["epoch_start"], r["epoch_end"]) for r in records]
        def cov_mean(field):
            values = [c[field] for c in cov if c[field] is not None]
            return statistics.fmean(values) if values else None
        rows[key] = {
            "size": key[0], "solver": key[1], "in_flight": key[2],
            "particles": records[0]["particles"],
            "n": len(records),
            "fps_mean": mean, "fps_std": std,
            "fps_min": min(fps_values), "fps_max": max(fps_values),
            "us_per_step_mean": statistics.fmean(r["us_per_step"] for r in records),
            "mparticle_steps_per_s": records[0]["particles"] * mean / 1e6,
            "sm_mhz": cov_mean("sm_mhz"), "power_w": cov_mean("power_w"),
            "temp_c": cov_mean("temp_c"),
            "drift_max_abs": max(abs(r["drift"]) for r in records),
            "by_trial": {r["trial"]: r["fps"] for r in records},
        }
    return rows


def trialwise_ratio(rows, size, num_key, den_key) -> tuple[float, float, int]:
    """mean ± std over trials of rows[num]/rows[den], trials present in both."""
    num = rows.get((size,) + num_key)
    den = rows.get((size,) + den_key)
    if not num or not den:
        return float("nan"), float("nan"), 0
    common = sorted(set(num["by_trial"]) & set(den["by_trial"]))
    ratios = [num["by_trial"][t] / den["by_trial"][t] for t in common]
    mean, std = mean_std(ratios)
    return mean, std, len(ratios)


def fmt(value, digits=1, suffix=""):
    if value is None or value != value:
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def write_summary(out_dir: pathlib.Path, rows, failed, sizes_present) -> str:
    lines = []
    lines.append("# Single-GPU baseline: V0 reference vs V5 (single-GPU mode)\n")
    lines.append("One headless RTX 5090 (nvidia-smi index 1, PCI 03:00.0), 2-D lid-driven cavity, "
                 "validation layer off, no GPU timestamps. fps = measured steps / wall time with an "
                 "empty queue at both ends; warmup 1000 steps; defrag every 1000 steps inside the "
                 "window. Values are mean ± std over interleaved trials (n).\n")
    configs = [("v0", 1), ("v0", 2), ("v5", 1), ("v5", 2)]
    header = "| size | particles | " + " | ".join(f"{SOLVER_SHORT[s]} in-flight {d} (fps)" for s, d in configs) + " |"
    lines.append("## Throughput (fps)\n")
    lines.append(header)
    lines.append("|" + "---|" * (2 + len(configs)))
    for size in sizes_present:
        cells = []
        particles = None
        for s, d in configs:
            row = rows.get((size, s, d))
            if row:
                particles = row["particles"]
                cells.append(f"{row['fps_mean']:.1f} ± {row['fps_std']:.1f} (n={row['n']})")
            else:
                cells.append("n/a")
        lines.append(f"| {size} | {particles:,} | " + " | ".join(cells) + " |" if particles
                     else f"| {size} | n/a | " + " | ".join(cells) + " |")

    lines.append("\n## Ratios (trial-wise, mean ± std)\n")
    lines.append("| size | V5/V0 @ in-flight 1 | V5/V0 @ in-flight 2 | V0: in-flight 2 / 1 | V5: in-flight 2 / 1 |")
    lines.append("|---|---|---|---|---|")
    for size in sizes_present:
        r1 = trialwise_ratio(rows, size, ("v5", 1), ("v0", 1))
        r2 = trialwise_ratio(rows, size, ("v5", 2), ("v0", 2))
        g0 = trialwise_ratio(rows, size, ("v0", 2), ("v0", 1))
        g5 = trialwise_ratio(rows, size, ("v5", 2), ("v5", 1))
        def cell(r):
            return "n/a" if r[2] == 0 else f"{100 * r[0]:.1f}% ± {100 * r[1]:.1f} (n={r[2]})"
        lines.append(f"| {size} | {cell(r1)} | {cell(r2)} | {cell(g0)} | {cell(g5)} |")

    lines.append("\n## Particle throughput (million particle-steps per second, from mean fps)\n")
    lines.append("| size | " + " | ".join(f"{SOLVER_SHORT[s]} d{d}" for s, d in configs) + " |")
    lines.append("|" + "---|" * (1 + len(configs)))
    for size in sizes_present:
        cells = [fmt(rows[(size, s, d)]["mparticle_steps_per_s"], 1) if (size, s, d) in rows else "n/a"
                 for s, d in configs]
        lines.append(f"| {size} | " + " | ".join(cells) + " |")

    lines.append("\n## Covariates (nvidia-smi, 1 Hz, inside each measured window; mean over trials)\n")
    lines.append("| size | config | SM clock (MHz) | power (W) | max temp (°C) | max |drift| |")
    lines.append("|---|---|---|---|---|---|")
    for size in sizes_present:
        for s, d in configs:
            row = rows.get((size, s, d))
            if not row:
                continue
            lines.append(f"| {size} | {SOLVER_SHORT[s]} d{d} | {fmt(row['sm_mhz'], 0)} | "
                         f"{fmt(row['power_w'], 0)} | {fmt(row['temp_c'], 0)} | {row['drift_max_abs']} |")

    if failed:
        lines.append(f"\n## Failed / non-ok runs: {len(failed)}\n")
        for record in failed:
            lines.append(f"- {record.get('size')} {record.get('solver')} d{record.get('in_flight')} "
                         f"trial {record.get('trial')}: {record.get('status')} "
                         f"(rc={record.get('returncode')}, {record.get('run_log')})")
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    return text


def write_csv(out_dir: pathlib.Path, rows) -> None:
    fields = ["size", "solver", "in_flight", "particles", "n", "fps_mean", "fps_std",
              "fps_min", "fps_max", "us_per_step_mean", "mparticle_steps_per_s",
              "sm_mhz", "power_w", "temp_c", "drift_max_abs"]
    with open(out_dir / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(rows, key=lambda k: (CURVE_SIZES.index(k[0]) if k[0] in CURVE_SIZES else 99, k[1], k[2])):
            writer.writerow({field: rows[key][field] for field in fields})


def plot(out_dir: pathlib.Path, rows, docs_png: pathlib.Path | None) -> pathlib.Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY, "text.color": INK_PRIMARY,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
        "grid.color": GRIDLINE, "grid.linewidth": 0.6,
    })
    sizes = [tag for tag in CURVE_SIZES if any(k[0] == tag for k in rows)]
    n_values = {tag: next(rows[k]["particles"] for k in rows if k[0] == tag) for tag in sizes}
    x = [n_values[tag] for tag in sizes]

    series = [("v0", 1), ("v0", 2), ("v5", 1), ("v5", 2)]
    style = {
        ("v0", 1): dict(color=COLOR_V0, linestyle="--", marker="o", markerfacecolor=SURFACE),
        ("v0", 2): dict(color=COLOR_V0, linestyle="-", marker="o"),
        ("v5", 1): dict(color=COLOR_V5, linestyle="--", marker="s", markerfacecolor=SURFACE),
        ("v5", 2): dict(color=COLOR_V5, linestyle="-", marker="s"),
    }
    label = {(s, d): f"{SOLVER_SHORT[s]}, {d} frame{'s' if d > 1 else ''} in flight"
             for s, d in series}

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 9.0), sharex=True,
                             gridspec_kw={"height_ratios": [1.25, 1.0, 0.9], "hspace": 0.12})
    ax_fps, ax_tp, ax_ratio = axes

    for key in series:
        ys, es, xs, tps = [], [], [], []
        for tag in sizes:
            row = rows.get((tag,) + key)
            if not row:
                continue
            xs.append(n_values[tag]); ys.append(row["fps_mean"]); es.append(row["fps_std"])
            tps.append(row["mparticle_steps_per_s"])
        if not xs:
            continue
        ax_fps.errorbar(xs, ys, yerr=es, linewidth=1.5, markersize=6, capsize=2,
                        markeredgewidth=1.5, label=label[key], **style[key])
        ax_tp.plot(xs, tps, linewidth=1.5, markersize=6, markeredgewidth=1.5, **style[key])

    ax_fps.set_yscale("log")
    ax_fps.set_xscale("log")
    fps_ticks = [tick for tick in (10, 20, 50, 100, 200, 500, 1000)
                 if ax_fps.get_ylim()[0] <= tick <= ax_fps.get_ylim()[1] * 1.05]
    ax_fps.set_yticks(fps_ticks)
    ax_fps.set_yticklabels([str(tick) for tick in fps_ticks])
    ax_fps.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax_fps.set_ylabel("steps per second (fps)")
    ax_fps.set_title("Single RTX 5090: V0 reference solver vs V5 multi-GPU solver in single-GPU mode",
                     fontsize=10, loc="left")
    ax_fps.legend(frameon=False, fontsize=8, loc="lower left")
    ax_fps.grid(True, which="major", axis="y")

    ax_tp.set_ylabel("million particle-steps / s\n(axis zoomed to the 3% band)")
    ax_tp.grid(True, which="major", axis="y")
    tp_all = [rows[k]["mparticle_steps_per_s"] for k in rows if k[0] in sizes]
    ax_tp.set_ylim(min(tp_all) * 0.96, max(tp_all) * 1.03)

    ratio_styles = {1: dict(color=INK_SECONDARY, linestyle="--", marker="o", markerfacecolor=SURFACE),
                    2: dict(color=INK_PRIMARY, linestyle="-", marker="o")}
    for depth in (1, 2):
        xs, ys, es = [], [], []
        for tag in sizes:
            mean, std, n = trialwise_ratio(rows, tag, ("v5", depth), ("v0", depth))
            if n == 0:
                continue
            xs.append(n_values[tag]); ys.append(100 * mean); es.append(100 * std)
        if xs:
            ax_ratio.errorbar(xs, ys, yerr=es, linewidth=1.5, markersize=6, capsize=2,
                              markeredgewidth=1.5, label=f"V5 / V0 at {depth} in flight",
                              **ratio_styles[depth])
    ax_ratio.axhline(100, color=AXIS, linewidth=1.0)
    ax_ratio.set_ylabel("V5 / V0 fps (%)")
    ax_ratio.set_xlabel("particles (fluid + wall)")
    ax_ratio.legend(frameon=False, fontsize=8, loc="lower right")
    ax_ratio.grid(True, which="major", axis="y")
    ax_ratio.set_ylim(94, 101)
    # Label the power-of-two sizes only (1M..32M); the others get unlabeled
    # minor ticks so 14M and 16M do not collide.
    labeled = [tag for tag in sizes if tag in ("1m", "2m", "4m", "8m", "16m", "32m")]
    ax_ratio.set_xticks([n_values[tag] for tag in labeled])
    ax_ratio.set_xticklabels([tag.upper() for tag in labeled])
    ax_ratio.set_xticks([n_values[tag] for tag in sizes if tag not in labeled], minor=True)
    ax_ratio.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    for ax in axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(length=3, color=AXIS)

    out_png = out_dir / "single_gpu_baseline.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    if docs_png:
        docs_png.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out_png, docs_png)
    return out_png


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="logs/single_baseline_20260923")
    parser.add_argument("--gpu-index", type=int, default=1, help="nvidia-smi index for telemetry")
    parser.add_argument("--docs-png", type=str, default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    out_dir = _REPO / args.dir
    ok, failed = load_results(out_dir / "results.jsonl")
    samples = load_telemetry(out_dir / "telemetry.csv", args.gpu_index)
    rows = aggregate(ok, samples)
    sizes_present = [tag for tag, _, _ in SIZES if any(k[0] == tag for k in rows)]

    text = write_summary(out_dir, rows, failed, sizes_present)
    write_csv(out_dir, rows)
    print(text)
    if not args.no_plot and rows:
        png = plot(out_dir, rows, pathlib.Path(args.docs_png) if args.docs_png else None)
        print(f"[summary] figure -> {png}")
    print(f"[summary] ok runs={len(ok)}  failed={len(failed)}  telemetry samples={len(samples)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
