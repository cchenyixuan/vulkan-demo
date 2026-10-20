"""
e7_loss_figure.py — the paper's fig:loss (section 6.5) redrawn as two panels (user 2026-10-20), from the batch-2
tables JSON (e7_batch2_tables.py) and the existing step traces on wqd10nbj04g2; no new runs, no b2AA data.

Quantities (the definitions of loss_sources.pdf v1 / e7_full_summary.own_chain, now per card):
  T_B     the card's phase B (b_start -> b_end of its own step), median over the steady steps
  t_own   the transport chain into the card without any wait for another GPU: t_chain (upload_end - send_end minus
          the receiver's two waits) minus the two transfer-queue idle waits (e7_full_summary.transfer_queue_idle);
          a card has one or two inbound links, per step the slower one (larger t_own) counts; median over the steps
  t_host  the host hop of that link-step (readback_end -> worker_copy minus the receiver waits: the worker sees
          the readback, stamp check, memcpy) = t_own minus its readback part (send_end -> readback_end minus the
          readback idle) and its upload part (worker_copy -> upload_end minus the upload idle); median
  r_own   t_own / T_B per step on that link, median over the steps
Each quantity is the median per card, then the mean over the 8 cards. eta = the tables' eta without the defrag
events, mean over the trials (e7_scaling_figures.select: arm C, window 1500/500, most trials then the later job).
Steady steps = step >= warmup with complete rows (as link_metrics; the defrag step stays in, medians).

  (a) T_B (filled circle, solid), t_own (open circle, solid) and the host part of t_own (dashed) against particles
      per GPU (M, walls included), both axes log, the strong K = 8 points; a light vertical line at 2M per GPU.
  (b) eta against r_own (log): strong filled circles, weak open triangles; dashed line at r_own = 1; eta 0..1.05.
  Style: e7_scaling_figures (Times 8 pt, elsarticle text width), panel labels (a) (b).

Usage:
    python docs/cluster_v6/scripts/e7_loss_figure.py --tables tables_b2.json --logs LOGS_N56 --out FIGURE_DIR
        [--exclude-job e7_b2T_1695731 ...]
Writes <out>/loss_sources.pdf (+ .png), <out>/loss_sources.csv (case, scaling, M_per_device, T_B, t_own, t_host,
r_own, eta; times in ms) and <out>/loss_sources_figure.json (per card and per point details).
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
CARD_KEYS = ("T_B", "t_own", "t_host", "t_readback", "t_upload", "r_own")


def per_card(run: dict) -> list:
    """Per card (receiver): the medians over the steady steps of T_B, t_own, its three parts and r_own, the slower
    inbound link per step."""
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
    cards = []
    for card in sorted({values["receiver"] for values in metrics.values()}):
        inbound = [values for values in metrics.values() if values["receiver"] == card]
        steps = sorted(set.intersection(*[set(values["index"]) for values in inbound]))
        series = {key: [] for key in CARD_KEYS}
        chosen = []
        for step in steps:
            candidates = [(values["t_own"][values["index"][step]], values) for values in inbound]
            if all(not math.isfinite(t) for t, _ in candidates):
                continue
            t_own, values = max(candidates, key=lambda item: item[0] if math.isfinite(item[0]) else -math.inf)
            position = values["index"][step]
            for key in CARD_KEYS:
                series[key].append(float(values[key][position]))
            chosen.append(f"{values['sender']}->{values['receiver']}")
        entry = {"card": card, "inbound_links": sorted(f"{v['sender']}->{v['receiver']}" for v in inbound),
                 "steps": len(chosen), "slower_link_share": {link: chosen.count(link) / len(chosen) for link in set(chosen)}}
        for key in CARD_KEYS:
            entry[key] = float(np.nanmedian(series[key])) if series[key] else math.nan
        cards.append(entry)
    return cards


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
            cards = per_card(run)
            entry["trace"] = dict(trace, directory=str(directory), cards=cards)
            for key in CARD_KEYS:
                values = [card[key] for card in cards if math.isfinite(card[key])]
                entry[key] = statistics.mean(values)
                entry[key + "_min"], entry[key + "_max"] = min(values), max(values)
            print(f"[trace] {case} K = {slabs}: {directory.name} ({trace['job']}), {len(cards)} cards, "
                  f"T_B {entry['T_B'] / 1e3:.3f} ms, t_own {entry['t_own'] / 1e3:.3f} ms, host {entry['t_host'] / 1e3:.3f}, "
                  f"readback {entry['t_readback'] / 1e3:.3f}, upload {entry['t_upload'] / 1e3:.3f}, r_own {entry['r_own']:.3f} "
                  f"(worst-link p50 {trace['r_own_worst_link_p50']:.3f})", flush=True)
        rows.append(entry)
    rows.sort(key=lambda r: (r["scaling"], r["per_gpu"]))
    return rows


def figure(rows: list, out: pathlib.Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullFormatter
    strong = [row for row in rows if row["scaling"] == "strong" and row.get("trace")]
    weak = [row for row in rows if row["scaling"] == "weak" and row.get("trace")]
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
    axis_t.axvline(2.0, color="#bdbdbd", lw=0.7, zorder=0.5)
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
    axis_t.set_xlabel("particles per GPU (M)", labelpad=1.5)
    axis_t.set_ylabel("time per step (ms)", labelpad=1.5)
    axis_t.legend(loc="upper left", frameon=False, handlelength=1.6, borderaxespad=0.2)
    # (b) eta against r_own
    style.band(axis_r)
    axis_r.axvline(1.0, color="#555555", lw=0.6, ls=(0, (3.0, 2.0)), zorder=0.6)

    def draw(selection, label, **marker):
        axis_r.errorbar([row["r_own"] for row in selection], [row["eta"] for row in selection],
                        yerr=[[row["eta"] - row["eta_min"] for row in selection], [row["eta_max"] - row["eta"] for row in selection]],
                        capsize=1.6, elinewidth=0.6, capthick=0.6, ls="none", label=label, **marker)
    draw(strong, r"strong, $K = 8$", color=style.COLOR_MAIN, marker="o", ms=3.4, mfc=style.COLOR_MAIN, mec=style.COLOR_MAIN)
    if weak:
        draw(weak, r"weak, $K = 8$", color=style.COLOR_RED, marker="^", ms=3.8, mfc="white", mec=style.COLOR_RED, mew=0.8)
    r_values = [row["r_own"] for row in strong + weak]
    axis_r.set_xlim(min(r_values) / 1.8, max(r_values) * 1.8)
    axis_r.set_ylim(0.0, 1.05)
    axis_r.yaxis.set_major_locator(MultipleLocator(0.2))
    axis_r.set_xlabel(r"$r_\mathrm{own}$", labelpad=1.5)
    axis_r.set_ylabel(r"$\eta$", labelpad=1.5)
    axis_r.legend(loc="lower left", frameon=False, handlelength=1.4, borderaxespad=0.2)
    style.panel_label(axis_t, "(a)")
    style.panel_label(axis_r, "(b)")
    style.save(fig, out / "loss_sources.pdf")
    plt.close(fig)


def write_csv(rows: list, path: pathlib.Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("case", "scaling", "M_per_device", "T_B", "t_own", "t_host", "r_own", "eta"))
        for row in rows:
            traced = row.get("trace") is not None
            writer.writerow((row["case"], row["scaling"], f"{row['per_gpu'] / 1e6:.4f}",
                             f"{row['T_B'] / 1e3:.4f}" if traced else "", f"{row['t_own'] / 1e3:.4f}" if traced else "",
                             f"{row['t_host'] / 1e3:.4f}" if traced else "", f"{row['r_own']:.4f}" if traced else "",
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
    (out / "loss_sources_figure.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"{len(rows)} rows -> {out / 'loss_sources.csv'}")
    print("| case | scaling | M/GPU | T_B ms | t_own ms (cards min-max) | readback | host | upload | r_own (cards min-max) | "
          "r_own worst-link p50 (v1) | eta | fps (min-max) | trace |")
    for row in rows:
        if row.get("trace"):
            print(f"| {row['case']} | {row['scaling']} | {row['per_gpu'] / 1e6:.3f} | {row['T_B'] / 1e3:.3f} | "
                  f"{row['t_own'] / 1e3:.3f} ({row['t_own_min'] / 1e3:.3f}-{row['t_own_max'] / 1e3:.3f}) | "
                  f"{row['t_readback'] / 1e3:.3f} | {row['t_host'] / 1e3:.3f} | {row['t_upload'] / 1e3:.3f} | "
                  f"{row['r_own']:.3f} ({row['r_own_min']:.3f}-{row['r_own_max']:.3f}) | {row['trace']['r_own_worst_link_p50']:.3f} | "
                  f"{row['eta']:.3f} | {row['fps']:.1f} ({row['fps_min']:.1f}-{row['fps_max']:.1f}) | "
                  f"{pathlib.Path(row['trace']['directory']).name} ({row['trace']['job']}) |")
        else:
            print(f"| {row['case']} | {row['scaling']} | {row['per_gpu'] / 1e6:.3f} | - | - | - | - | - | - | - | "
                  f"{row['eta']:.3f} | {row['fps']:.1f} ({row['fps_min']:.1f}-{row['fps_max']:.1f}) | no trace |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
