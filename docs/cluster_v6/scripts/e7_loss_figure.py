"""
e7_loss_figure.py — the paper's fig:loss (section 6.5) redrawn as two panels (user 2026-10-20), from the batch-2
tables JSON (e7_batch2_tables.py) and the existing step traces on wqd10nbj04g2; no new runs, no b2AA data.

Quantities (the definitions of loss_sources.pdf v1 / e7_full_summary.own_chain, the slowest of the 14 links per step;
user 2026-10-20 second message):
  T_B     each card's phase B (b_start -> b_end of its own step), median over the steady steps, mean over the 8 cards
  t_own   per step the largest t_own over the 14 links, median over the steps; t_own of a link-step = t_chain
          (upload_end - send_end minus the receiver's two waits) minus the two transfer-queue idle waits
          (e7_full_summary.transfer_queue_idle): the chain's own work without any wait for another GPU
  t_host  the host hop (readback_end -> worker_copy minus the receiver waits: the worker sees the readback, stamp
          check, memcpy) of that slowest link at that step, median; likewise t_readback (send_end -> readback_end
          minus the readback idle) and t_upload (worker_copy -> upload_end minus the upload idle)
  r_own   per step the largest t_own / T_B over the links (T_B = the link's receiver), median over the steps
          (step_trace_model.worst_link on r_own, as the tables' r_own p50)
eta = the tables' eta without the defrag events, mean over the trials (e7_scaling_figures.select: arm C, window
1500/500, most trials then the later job). Steady steps = step >= warmup with complete rows on every link (as
link_metrics / worst_link; the defrag step stays in, medians).

  (a) T_B (filled circle, solid), t_own (open circle, solid) and the host part of t_own (dashed) against particles
      per device (M, walls included), both axes log, the nine strong K = 8 points.
  (b) eta against r_own (log): the same nine points, filled circles, no legend; dashed line at r_own = 1;
      eta 0..1.05 with the band eta = 1 +- 0.02.
  Style: e7_scaling_figures (Times 8 pt, elsarticle text width), panel labels (a) (b).

Usage:
    python docs/cluster_v6/scripts/e7_loss_figure.py --tables tables_b2.json --logs LOGS_N56 --out FIGURE_DIR
        [--exclude-job e7_b2T_1695731 ...]
Writes <out>/loss_sources.pdf (+ .png), <out>/loss_sources.csv (case, scaling, M_per_device, T_B, t_own, t_host,
r_own, eta; times in ms; the plotted strong points) and <out>/loss_sources_figure.json (per point details, the
weak points with a trace included, and the T_B = t_own crossing).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
import statistics
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from experiment.v7.analysis import step_trace_model  # noqa: E402
import e7_scaling_figures as style  # noqa: E402
from e7_full_summary import transfer_queue_idle  # noqa: E402
from e7_loss_sources import JOB_DIRS  # noqa: E402

WEAK_LOADS = ("cavity2d_n1440", "cavity2d_n2840", "cavity2d_n5640")      # the k members' reference cases
LINK_KEYS = ("t_own", "t_host", "t_readback", "t_upload")


def worst_link_medians(run: dict) -> dict:
    """T_B: per card the median phase B, mean over the cards (per-card values listed); t_own and its three parts:
    per step the slowest link (largest t_own) over all links, medians over the steps; r_own: per step the largest
    t_own / T_B over the links, median (the tables' definition)."""
    metrics = step_trace_model.link_metrics(run)
    idle = transfer_queue_idle(run)
    for name, values in metrics.items():
        pairs = [idle.get((name, int(step)), (math.nan, math.nan)) for step in values["step"]]
        readback_idle = np.array([pair[0] for pair in pairs])
        upload_idle = np.array([pair[1] for pair in pairs])
        values["t_readback"] = values["chain_readback"] - readback_idle
        values["t_upload"] = values["chain_upload"] - upload_idle
        values["t_host"] = values["chain_host"]
        values["t_own"] = values["t_chain"] - readback_idle - upload_idle
        values["r_own"] = values["t_own"] / values["T_B"]
        values["index"] = {int(step): position for position, step in enumerate(values["step"])}
    steps = sorted(set.intersection(*[set(values["index"]) for values in metrics.values()]))
    series = {key: [] for key in LINK_KEYS}
    slowest = []
    for step in steps:
        candidates = [(values["t_own"][values["index"][step]], values) for values in metrics.values()]
        if all(not math.isfinite(t) for t, _ in candidates):
            continue
        _, values = max(candidates, key=lambda item: item[0] if math.isfinite(item[0]) else -math.inf)
        position = values["index"][step]
        for key in LINK_KEYS:
            series[key].append(float(values[key][position]))
        slowest.append(f"{values['sender']}->{values['receiver']}")
    r_own_steps = step_trace_model.worst_link(metrics, "r_own")
    cards = step_trace_model.cycle_components(run)          # per sim medians (us) of the phases, T_B = "b"
    t_b_cards = [entry["b"] for entry in cards]
    out = {"links": len(metrics), "steps": len(slowest), "T_B": statistics.mean(t_b_cards),
           "T_B_cards": t_b_cards, "r_own": float(np.nanmedian(r_own_steps)),
           "r_own_p95": float(np.nanpercentile(r_own_steps, 95)),
           "slowest_link_share": {link: round(slowest.count(link) / len(slowest), 3) for link in sorted(set(slowest))}}
    for key in LINK_KEYS:
        out[key] = float(np.nanmedian(series[key]))
        out[key + "_p95"] = float(np.nanpercentile(series[key], 95))
    return out


def trace_for(tables: dict, row: dict, exclude: set) -> dict | None:
    """The traced C-arm run of the same (case, K) on a g2 job (any job in JOB_DIRS not excluded)."""
    for point in tables["points"]:
        if point["case"] != row["case"] or point["K"] != row["K"] or point["arm"] != "C":
            continue
        if point["job"] in exclude or point["job"] not in JOB_DIRS:
            continue
        trace = point.get("trace") or {}
        if trace.get("status") == "pass" and trace.get("label"):
            return {"job": point["job"], "label": trace["label"], "host": point["host"],
                    "r_own_worst_link_p50": trace.get("r_own_p50"), "r_own_worst_link_p95": trace.get("r_own_p95")}
    return None


def gather(tables: dict, logs: pathlib.Path, exclude: set) -> list:
    chosen = style.select(tables["points"], exclude, "1500/500", "eta_nodefrag")
    rows = []
    for (case, slabs), row in chosen.items():
        if slabs != 8:
            continue
        weak = bool(re.search(r"_k\d+$", case))
        if weak and row["reference"] not in {f"weak:{load}" for load in WEAK_LOADS}:
            continue
        if not weak and row["kind"] not in ("strong", "pairs"):
            continue
        by_key = {(p["job"], p["point"], p["arm"]): p for p in tables["points"]}
        point = by_key[(row["job"], row["point"], "C")]
        entry = {"case": case, "scaling": "weak" if weak else "strong", "K": slabs, "point": row["point"],
                 "eta_job": row["job"], "host": row["host"], "reference": row["reference"], "trials": row["trials"],
                 "particles": row["particles"], "per_gpu": row["per_gpu"], "eta": row["eta"], "eta_min": row["eta_min"],
                 "eta_max": row["eta_max"], "fps": point["fps"]["mean"], "fps_min": point["fps"]["min"],
                 "fps_max": point["fps"]["max"], "fps_nodefrag": (point.get("fps_nodefrag") or {}).get("mean")}
        trace = trace_for(tables, row, exclude)
        if trace is None:
            entry["trace"] = None
            print(f"[trace] {case} K = {slabs}: no traced run on a g2 job -> eta only", flush=True)
        else:
            directory = logs / JOB_DIRS[trace["job"]] / "traces" / trace["label"].replace("_trace_C", "_C")
            run = step_trace_model.load_run(directory)
            medians = worst_link_medians(run)
            entry["trace"] = dict(trace, directory=str(directory), **medians)
            for key in ("T_B", "r_own", "r_own_p95") + LINK_KEYS:
                entry[key] = medians[key]
            print(f"[trace] {case} K = {slabs}: {directory.name} ({trace['job']}), {medians['links']} links x {medians['steps']} steps, "
                  f"T_B {entry['T_B'] / 1e3:.3f} ms (cards {min(medians['T_B_cards']) / 1e3:.3f}-{max(medians['T_B_cards']) / 1e3:.3f}), "
                  f"t_own {entry['t_own'] / 1e3:.3f} ms = readback {entry['t_readback'] / 1e3:.3f} + host {entry['t_host'] / 1e3:.3f} + "
                  f"upload {entry['t_upload'] / 1e3:.3f} (medians), r_own {entry['r_own']:.3f} "
                  f"(tables' p50 {trace['r_own_worst_link_p50']:.3f})", flush=True)
        rows.append(entry)
    rows.sort(key=lambda r: (r["scaling"], r["per_gpu"]))
    return rows


def crossing(strong: list) -> list:
    """Particles per device where t_own = T_B, log-log interpolation between the adjacent points."""
    out = []
    for low, high in zip(strong, strong[1:]):
        f_low = math.log(low["t_own"]) - math.log(low["T_B"])
        f_high = math.log(high["t_own"]) - math.log(high["T_B"])
        if f_low == 0.0 or f_low * f_high > 0:
            continue
        share = f_low / (f_low - f_high)
        u = math.log(low["per_gpu"]) + share * (math.log(high["per_gpu"]) - math.log(low["per_gpu"]))
        value = math.exp(math.log(low["T_B"]) + share * (math.log(high["T_B"]) - math.log(low["T_B"])))
        out.append({"between": [low["case"], high["case"]], "per_device": math.exp(u), "time_us": value})
    return out


def figure(rows: list, out: pathlib.Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullFormatter
    strong = [row for row in rows if row["scaling"] == "strong" and row.get("trace")]
    width, panel, height_panel = style.TEXT_WIDTH, 2.05, 1.85
    left, gap, bottom, top = 0.42, 0.62, 0.40, 0.20
    fig = plt.figure(figsize=(width, bottom + height_panel + top))
    axis_t = style.axes_inches(fig, left, bottom, panel, height_panel)
    axis_r = style.axes_inches(fig, left + panel + gap, bottom, panel, height_panel)
    plain = FuncFormatter(lambda value, _: f"{value:g}")
    for axis in (axis_t, axis_r):
        style.frame(axis)
        axis.set_xscale("log")
        axis.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
        axis.xaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=12))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.xaxis.set_major_formatter(plain)
    # (a) times against particles per GPU
    xs = [row["per_gpu"] / 1e6 for row in strong]
    t_b = [row["T_B"] / 1e3 for row in strong]
    t_own = [row["t_own"] / 1e3 for row in strong]
    t_host = [row["t_host"] / 1e3 for row in strong]
    axis_t.plot(xs, t_b, color=style.COLOR_MAIN, marker="o", ms=3.4, mfc=style.COLOR_MAIN, mec=style.COLOR_MAIN, lw=0.9,
                label=r"$T_B$", zorder=3)
    axis_t.plot(xs, t_own, color=style.COLOR_RED, marker="o", ms=3.4, mfc="white", mec=style.COLOR_RED, mew=0.8, lw=0.9,
                label=r"$t_\mathrm{own}$", zorder=3)
    axis_t.plot(xs, t_host, color=style.COLOR_RED, ls=(0, (3.0, 2.0)), lw=0.9, label=r"host part of $t_\mathrm{own}$",
                zorder=2.5)
    axis_t.set_yscale("log")
    axis_t.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    axis_t.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=12))
    axis_t.yaxis.set_minor_formatter(NullFormatter())
    axis_t.yaxis.set_major_formatter(plain)
    axis_t.set_xlim(min(xs) / 1.7, max(xs) * 1.7)
    ys = t_b + t_own + t_host
    axis_t.set_ylim(min(ys) / 1.8, max(ys) * 2.2)
    axis_t.set_xlabel("particles per device (M)", labelpad=1.5)
    axis_t.set_ylabel("time per step (ms)", labelpad=1.5)
    axis_t.legend(loc="upper left", frameon=False, handlelength=1.6, borderaxespad=0.2)
    # (b) eta against r_own: the same nine strong points, no legend; band eta = 1 +- 0.02 (style.band)
    style.band(axis_r)
    axis_r.axvline(1.0, color="#555555", lw=0.6, ls=(0, (3.0, 2.0)), zorder=0.6)
    axis_r.errorbar([row["r_own"] for row in strong], [row["eta"] for row in strong],
                    yerr=[[row["eta"] - row["eta_min"] for row in strong], [row["eta_max"] - row["eta"] for row in strong]],
                    capsize=1.6, elinewidth=0.6, capthick=0.6, ls="none", color=style.COLOR_MAIN, marker="o", ms=3.4,
                    mfc=style.COLOR_MAIN, mec=style.COLOR_MAIN)
    r_values = [row["r_own"] for row in strong]
    axis_r.set_xlim(min(r_values) / 1.8, max(r_values) * 1.8)
    axis_r.set_ylim(0.0, 1.05)
    axis_r.yaxis.set_major_locator(MultipleLocator(0.2))
    axis_r.set_xlabel(r"$r_\mathrm{own}$", labelpad=1.5)
    axis_r.set_ylabel(r"$\eta$", labelpad=1.5)
    style.panel_label(axis_t, "(a)")
    style.panel_label(axis_r, "(b)")
    style.save(fig, out / "loss_sources.pdf")
    plt.close(fig)


def write_csv(rows: list, path: pathlib.Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("case", "scaling", "M_per_device", "T_B", "t_own", "t_host", "r_own", "eta"))
        for row in rows:
            if row["scaling"] != "strong" or row.get("trace") is None:
                continue                                    # the plotted points only
            writer.writerow((row["case"], row["scaling"], f"{row['per_gpu'] / 1e6:.4f}", f"{row['T_B'] / 1e3:.4f}",
                             f"{row['t_own'] / 1e3:.4f}", f"{row['t_host'] / 1e3:.4f}", f"{row['r_own']:.4f}",
                             f"{row['eta']:.4f}"))


def main() -> int:
    parser = argparse.ArgumentParser(description="fig:loss, two panels (vector PDF)")
    parser.add_argument("--tables", required=True)
    parser.add_argument("--logs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--exclude-job", action="append", default=[])
    arguments = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    style.setup_style()
    tables = json.loads(pathlib.Path(arguments.tables).read_text(encoding="utf-8"))
    out = pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = gather(tables, pathlib.Path(arguments.logs), set(arguments.exclude_job))
    figure(rows, out)
    write_csv(rows, out / "loss_sources.csv")
    strong = [row for row in rows if row["scaling"] == "strong" and row.get("trace")]
    crossings = crossing(strong)
    (out / "loss_sources_figure.json").write_text(json.dumps({"points": rows, "t_own_equals_T_B": crossings}, indent=1),
                                                  encoding="utf-8")
    print(f"{len(strong)} plotted rows -> {out / 'loss_sources.csv'}")
    print("| case | scaling | M/device | T_B ms (cards min-max) | t_own ms | readback | host | upload | t_own p95 | r_own | "
          "r_own p95 | tables' r_own p50 | eta | fps (min-max) | trace |")
    for row in rows:
        if row.get("trace"):
            trace = row["trace"]
            print(f"| {row['case']} | {row['scaling']} | {row['per_gpu'] / 1e6:.3f} | {row['T_B'] / 1e3:.3f} "
                  f"({min(trace['T_B_cards']) / 1e3:.3f}-{max(trace['T_B_cards']) / 1e3:.3f}) | {row['t_own'] / 1e3:.3f} | "
                  f"{row['t_readback'] / 1e3:.3f} | {row['t_host'] / 1e3:.3f} | {row['t_upload'] / 1e3:.3f} | {trace['t_own_p95'] / 1e3:.3f} | "
                  f"{row['r_own']:.3f} | {row['r_own_p95']:.3f} | {trace['r_own_worst_link_p50']:.3f} | "
                  f"{row['eta']:.3f} | {row['fps']:.1f} ({row['fps_min']:.1f}-{row['fps_max']:.1f}) | "
                  f"{pathlib.Path(trace['directory']).name} ({trace['job']}) |")
        else:
            print(f"| {row['case']} | {row['scaling']} | {row['per_gpu'] / 1e6:.3f} | - | - | - | - | - | - | - | - | - | "
                  f"{row['eta']:.3f} | {row['fps']:.1f} ({row['fps_min']:.1f}-{row['fps_max']:.1f}) | no trace |")
    for item in crossings:
        print(f"t_own = T_B between {item['between'][0]} and {item['between'][1]}: {item['per_device'] / 1e6:.3f}M per device, "
              f"{item['time_us'] / 1e3:.3f} ms (log-log interpolation)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
