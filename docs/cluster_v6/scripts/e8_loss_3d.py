"""
e8_loss_3d.py — the paper's section 5.5 (sources of the loss), 3-D data (user 2026-10-26 night; post-processing only,
E8 mirrors 02 / 03 / 04 / 09, no new runs).

  1. ledger    the seven K = 8 cubes (4M .. 128M) and the weak 8 x 8M K8: the chain period split into five terms, each a
               percentage of the period:
                 ideal      tau(per-card-size cube) / P: the K = 1 trace of the cube whose size is the per-card load
                            (128M -> 16M cube, 64M -> 8M, 32M -> 4M, weak -> 8M; stand-ins: 76M (9.5M per card) -> 8M
                            cube, 16M / 8M / 4M (2M / 1M / 0.5M per card) -> 4M cube, flagged);
                 seam work  (mean over the cards of the chain's busy time T_A + T_B + T_C) - tau(per-card cube);
                 reference working point   tau(N) / K - tau(per-card cube), sign kept (tau(N): the K = 1 trace of the
                            whole cube; 128M: the K = 2 pair trace / 4; weak: the reference is the 8M cube itself, 0);
                 transfer exposed   per card and step the largest (t_tr - T_B)+ over its incoming links (t_tr = the
                            chain's own three hops = t_chain minus the transfer queues' idle waits, E7's t_own; T_B =
                            the receiver's phase B of that step), per card the mean over the steps, mean over the cards;
                 imbalance  P - busy - exposed (the remainder: waiting for a slower neighbour, phase coupling).
               P, busy, exposed and the taus are MEANS over the steady steps (step >= warmup, next step present, the
               defrag step dropped: the "skip" convention of e8_phases_table), so that tau(N) / K / P is the trace's
               eta (fps is a mean). Identities: ideal + seam + exposed + imbalance = 100 (the period); eta_trace =
               ideal + reference working point; so the five terms add up to 100 + reference working point (negative in
               3-D: the per-card cube is dearer per particle than the whole cube).
               Alongside: eta of the timed trials (tables.json, no defrag, mean (min-max)), the G1 redundancy share
               (e8_kernel_table), the median of r (per step the largest t_tr / T_B over the links, E7's r_own p50) and
               the chain / reference SM clock ratio (per trial: mean over the chain's cards of the loop-window SM clock
               median / mean over the reference members' cards; mean over the trials).
  2. phases    tab:phases row blocks of 64M K8 and 8M K8 in the 2-D table's format (A / B / C medians per card, mean
               over the cards (min-max); step median; step mean window - defrag; step mean skip), the ideal-chain row
               = the K = 1 trace of the per-card-size cube, NOT divided (64M: the 8M cube; 8M: the 4M cube as the
               stand-in for 1M per card, flagged); the whole cube's K = 1 trace / 8 as a third row for comparison.
  3. figure    fig:loss with the 3-D series: (a) T_B, t_tr and its host part of the seven cubes (squares) beside the
               nine 2-D points (circles, from the E7 figure JSON, unchanged), log-log; (b) eta against r, the 2-D
               circles and the seven 3-D squares, log x (4M's r = 205). Style e7_loss_figure / e7_scaling_figures.
  4. 2-D checks   the G1 redundancy share of 2-D 64M K8 from the anatomy run's per-kernel samples (the boundary
               correction + density kernel per sim) and the chain trace (period, host copy bytes -> n0), by
               e8_kernel_table's formula; the SM clock ratio chain / reference of the nine strong K = 8 2-D points
               (tables_b2.json, per trial, mean (min-max)).
  5. memcpy    64M under K = 2 and K = 8: the worker memcpy of the 41 MB packet, median over every link and steady
               step, and the bandwidth at that median (K = 4 alongside).

Usage:
    python docs/cluster_v6/scripts/e8_loss_3d.py --job <02> --job <03> --job <04> --job <09>
        --tables docs/cluster_v6/data/e8/phase2/tables.json
        --two-d-json <e7>/figures/loss_sources_figure.json --two-d-tables <e7>/tables_b2.json
        --two-d-figure-csv <e7>/figures/scaling_figures.csv --two-d-logs ../vulkan-demo/logs/n56
        --out-dir docs/cluster_v6/data/e8/loss_3d --figure-out DIR [--figure-out DIR ...]
Writes OUT/ledger.{md,json,csv}, phases_3d.{md,json,csv}, two_d_checks.{md,json}, memcpy.{md,json},
loss_sources_figure.json and, in every --figure-out directory, loss_sources.{pdf,png,csv} (the previous files are
kept as loss_sources_2d.* when none exist yet).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
import shutil
import statistics
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e8_common as common  # noqa: E402
import e8_phases_table as phases  # noqa: E402
import e8_kernel_table as kernel_table  # noqa: E402
import e7_full_summary as summary  # noqa: E402
import e7_scaling_figures as style  # noqa: E402
from e7_loss_figure import worst_link_medians  # noqa: E402
from experiment.v7.analysis import step_trace_model  # noqa: E402

# point -> (per-card-size cube, note); the ledger / figure order
LEDGER = (("3d_n152_K8", "cavity3d_n152", "0.51M per card -> 4M cube (stand-in)"),
          ("3d_n192_K8", "cavity3d_n152", "1.0M per card -> 4M cube (stand-in)"),
          ("3d_n244_K8", "cavity3d_n152", "2.0M per card -> 4M cube (stand-in)"),
          ("3d_n312_K8", "cavity3d_n152", "4.1M per card = 4M cube"),
          ("3d_n392_K8", "cavity3d_n192", "8.0M per card = 8M cube"),
          ("3d_n416_K8", "cavity3d_n192", "9.5M per card -> 8M cube (stand-in)"),
          ("3d_n496_K8", "cavity3d_n244", "16.0M per card = 16M cube"),
          ("3d_n192_k8_K8", "cavity3d_n192", "weak: 7.7M per card = 8M cube (the reference itself)"))
PHASE_BLOCKS = (("3d_n392_K8", "cavity3d_n192", "8M cube = the per-card size"),
                ("3d_n192_K8", "cavity3d_n152", "4M cube as the stand-in for 1M per card"))
MEMCPY_POINTS = ("3d_n392_K2", "3d_n392_K4", "3d_n392_K8")
TWO_D_CHAIN = ("2026-10-12/03_e7_b2B_1694328", "x_2d_n8000_K8_C", "x_2d_n8000_K8_trace_C.log", "anatomy_2d_n8000_K8.log")
ANATOMY_LINE = re.compile(r"^\[anatomy\] f(\d+) s(\d+): (.*)$")
GRID_LINE = re.compile(r"\[case_loader_v7\] grid: dim=\((\d+), (\d+), (\d+)\)")


def fmt(value, digits=3):
    return common.fmt(value, digits)


def pct(value, digits=1):
    return "-" if value is None or not np.isfinite(value) else f"{value * 100:+.{digits}f}" if value < 0 else f"{value * 100:.{digits}f}"


# ----------------------------------------------------------------------------------------- traces: cards, exposure
def card_table(run: dict, events: list) -> list:
    """Per card over the kept steady steps (step >= warmup, next step present, defrag step dropped): the mean and the
    median of the period, of busy = T_A + T_B + T_C and of the three phases (ms)."""
    warmup = run["meta"]["warmup"]
    defrag_steps = {frame - 1 for frame, _ in events}
    out = []
    for sim in common.sims_of(run):
        rows = common.sim_rows(run, sim)
        steps = rows["step"].astype(int)
        keep = common.steady_mask(rows, warmup) & ~np.isin(steps, list(defrag_steps))
        next_a = np.concatenate([rows["a_start"][1:], [np.nan]])
        period = (next_a - rows["a_start"]) / common.NS_PER_MS
        a = (rows["a_end"] - rows["a_start"]) / common.NS_PER_MS
        b = (rows["b_end"] - rows["b_start"]) / common.NS_PER_MS
        c = (rows["c_end"] - rows["c_start"]) / common.NS_PER_MS
        busy = a + b + c
        entry = {"sim": sim, "steps": int(keep.sum())}
        for key, value in (("period", period), ("busy", busy), ("a", a), ("b", b), ("c", c)):
            entry[key + "_mean"] = float(np.mean(value[keep])) if keep.any() else None
            entry[key + "_p50"] = float(np.median(value[keep])) if keep.any() else None
        out.append(entry)
    return out


def exposure_table(run: dict, events: list) -> dict:
    """Per receiver card: per step the largest (t_tr - T_B)+ over its incoming links (t_tr = t_chain minus the two
    transfer-queue idle waits; T_B = the receiver's phase B of that step), then the mean and the median over the steps
    (us); the defrag step dropped."""
    metrics = step_trace_model.link_metrics(run)
    idle = summary.transfer_queue_idle(run)
    defrag_steps = {frame - 1 for frame, _ in events}
    per_card: dict = {}
    for name, values in metrics.items():
        pairs = [idle.get((name, int(step)), (math.nan, math.nan)) for step in values["step"]]
        readback_idle = np.array([pair[0] for pair in pairs])
        upload_idle = np.array([pair[1] for pair in pairs])
        t_tr = values["t_chain"] - readback_idle - upload_idle
        exposure = np.maximum(0.0, t_tr - values["T_B"])
        card = per_card.setdefault(int(values["receiver"]), {})
        for step, value in zip(values["step"].astype(int), exposure):
            if int(step) in defrag_steps or not np.isfinite(value):
                continue
            card[int(step)] = max(card.get(int(step), 0.0), float(value))
    return {card: {"steps": len(values), "mean_us": statistics.fmean(values.values()) if values else None,
                   "p50_us": float(np.median(list(values.values()))) if values else None}
            for card, values in sorted(per_card.items())}


def trace_events(run: dict, row: dict) -> list:
    meta = run["meta"]
    return common.defrag_events_in_window(row, meta["warmup"], meta.get("max_steps") or 10 ** 9)


# ----------------------------------------------------------------------------------------- jobs, tables
def collect_jobs(directories: list) -> dict:
    jobs = [common.load_job(directory) for directory in directories]
    points, k1, k2 = {}, {}, {}
    for job in jobs:
        for name, point in common.points_of(job).items():
            point["job"] = job
            points[name] = point
        for name, path in job["traces"].items():
            if name.startswith("trace_") and name.endswith("_K1"):
                k1[f"cavity{name[len('trace_'):-len('_K1')]}"] = (path, job["rows"].get(name), job["name"])
            elif name.startswith("trace_") and name.endswith("_K2"):
                k2[f"cavity{name[len('trace_'):-len('_K2')]}"] = (path, job["rows"].get(name), job["name"])
    return {"jobs": jobs, "points": points, "k1": k1, "k2": k2}


def tables_point(tables: dict, name: str) -> dict:
    hits = [p for p in tables["points"] if p["point"] == name]
    if not hits:
        raise KeyError(name)
    return hits[-1]


def clock_ratio(entry: dict) -> dict:
    """Per trial: mean over the chain's cards of the loop-window SM clock median / mean over the reference members'
    cards of theirs; mean (min-max) over the trials."""
    ratios = []
    for trial in entry.get("trials") or []:
        chain = [v["sm_clock_median"] for v in trial["k_clocks"].values() if v.get("sm_clock_median") is not None]
        reference = [clock["sm_clock_median"] for ref in trial["references"] for member in ref.get("members", [])
                     for clock in (member.get("clocks") or {}).values() if clock.get("sm_clock_median") is not None]
        if chain and reference:
            ratios.append(statistics.fmean(chain) / statistics.fmean(reference))
    if not ratios:
        return {"n": 0}
    return {"n": len(ratios), "mean": statistics.fmean(ratios), "min": min(ratios), "max": max(ratios), "values": ratios}


# ----------------------------------------------------------------------------------------- 1. ledger
def ledger_row(name: str, pc_case: str, note: str, data: dict, tables: dict) -> dict:
    point = data["points"][name]
    job = point["job"]
    case, slabs = point["case"], point["slabs"]
    weak = case not in common.CUBES
    run = common.load_trace(point["trace_dir"])
    trace_row = (point["traces"].get("C") or point["traces"].get("E") or {}).get("row")
    events = trace_events(run, trace_row)
    cards = card_table(run, events)
    exposure = exposure_table(run, events)
    medians = worst_link_medians(run)
    period = statistics.fmean(c["period_mean"] for c in cards)
    busy = statistics.fmean(c["busy_mean"] for c in cards)
    exposed = statistics.fmean(v["mean_us"] for v in exposure.values()) / 1e3
    chain_phases = phases.analyze_trace(point["trace_dir"], trace_row)["aggregate"]
    # tau of the per-card load at the per-card-size cube's cost: its K = 1 trace's step time x (N / K) / (the cube's
    # particles) -- 1.000 where the cube IS the per-card size (32M -> 4M, 64M -> 8M, 128M -> 16M), the per-particle
    # cost of the stand-in cube otherwise (76M: 9.53M at the 8M cube's cost; 16M / 8M / 4M: 2M / 1M / 0.5M at the 4M's)
    per_card = common.particles_of(case) / slabs
    pc_path, pc_row, pc_job = data["k1"][pc_case]
    pc = phases.analyze_trace(pc_path, pc_row, divide=1)
    pc_scale = per_card / common.particles_of(pc_case)
    tau_pc_cube = pc["aggregate"]["step_mean_skip"]
    tau_pc = tau_pc_cube * pc_scale
    # tau(N) / K: the whole cube's K = 1 trace / K, the 128M pair trace / (K / 2); weak: the per-card cube itself
    if weak:
        tau_ref, ref_kind, ref_source = tau_pc_cube, "weak: the 8M cube's K = 1 trace, not divided (the reference itself)", pc_job
    elif case in data["k1"]:
        path, row, source = data["k1"][case]
        tau_ref, ref_kind, ref_source = phases.analyze_trace(path, row, divide=slabs)["aggregate"]["step_mean_skip"], f"K = 1 trace / {slabs}", source
    elif case in data["k2"]:
        path, row, source = data["k2"][case]
        tau_ref, ref_kind, ref_source = phases.analyze_trace(path, row, divide=slabs // 2)["aggregate"]["step_mean_skip"], f"K = 2 pair trace / {slabs // 2}", source
    else:
        raise KeyError(f"{name}: no K = 1 / pair trace of {case}")
    ideal, seam, reference = tau_pc / period, (busy - tau_pc) / period, (tau_ref - tau_pc) / period
    exposed_share, imbalance = exposed / period, (period - busy - exposed) / period
    entry = tables_point(tables, name)
    eta = entry["stats"]["eta"]
    try:
        g1 = kernel_table.analyze(point["trace_dir"], case, point.get("raw_dir"), point.get("partition"))
        g1_share, g1_range = g1.get("g1_share_mean"), g1.get("g1_share_range")
    except Exception as error:                                              # noqa: BLE001
        g1_share, g1_range = None, repr(error)
    return {"point": name, "job": job["name"], "host": entry.get("host"), "case": case, "K": slabs, "kind": "weak" if weak else "strong",
            "per_card": per_card, "per_card_cube": pc_case, "per_card_cube_note": note, "per_card_cube_particles": common.particles_of(pc_case),
            "per_card_cube_scale": pc_scale, "tau_pc_cube_ms": tau_pc_cube, "c_pc_ns": tau_pc_cube * 1e6 / common.particles_of(pc_case),
            "per_card_cube_trace_job": pc_job, "reference_kind": ref_kind, "reference_trace_job": ref_source,
            "period_ms": period, "period_window_ms": chain_phases["step_mean_window"], "period_p50_ms": (chain_phases["period"] or {}).get("mean"),
            "busy_ms": busy, "exposed_ms": exposed, "tau_pc_ms": tau_pc, "tau_ref_ms": tau_ref,
            "ideal": ideal, "seam": seam, "reference_point": reference, "exposed": exposed_share, "imbalance": imbalance,
            "sum_five": ideal + seam + reference + exposed_share + imbalance, "eta_trace": tau_ref / period,
            "eta": eta["mean"], "eta_min": eta["min"], "eta_max": eta["max"], "eta_trials": eta["n"],
            "g1_share": g1_share, "g1_range": g1_range, "r_p50": medians["r_own"], "r_p95": medians["r_own_p95"],
            "clock_ratio": clock_ratio(entry), "cards": cards, "exposure": exposure,
            "T_B_us": medians["T_B"], "t_tr_us": medians["t_own"], "t_host_us": medians["t_host"], "t_readback_us": medians["t_readback"],
            "t_upload_us": medians["t_upload"], "t_tr_p95_us": medians["t_own_p95"], "links": medians["links"], "steps": medians["steps"],
            "trace": str(point["trace_dir"])}


def ledger_markdown(rows: list) -> str:
    lines = ["## 1. Ledger: the chain period split into five terms (% of the period; E8 K = 8 points, chain traces)", "",
             "P, busy, exposed and the taus are means over the kept steady steps (defrag step dropped); ideal = tau_pc / P with tau_pc = the "
             "per-card load N / K at the per-card-size cube's K = 1 cost (that cube's K = 1 trace step time x (N / K) / its particles; the "
             "factor is 1 where the cube is the per-card size: 32M -> 4M, 64M -> 8M, 128M -> 16M; the stand-ins scale the 4M / 8M cube's "
             "per-particle cost); seam = busy - tau_pc; reference working point = tau(N) / K - tau_pc (sign kept; tau(N) = the whole cube's "
             "K = 1 trace, 128M: the pair trace / 4; weak: the 8M cube itself, not scaled); exposed = per card and step the largest (t_tr - T_B)+ "
             "over its incoming links, mean over steps and cards; imbalance = P - busy - exposed. Identities: ideal + seam + exposed + "
             "imbalance = 100; eta_trace = ideal + reference working point; the five add up to 100 + reference working point.", "",
             "| point | per card | per-card cube (x factor) | P (ms) | ideal | seam work | reference working point | transfer exposed | imbalance | "
             "sum of five | eta_trace = ideal + ref | eta timed (min–max; trials) | G1 redundancy | r median (p95) | clock ratio chain / reference |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        clock = r["clock_ratio"]
        lines.append(f"| {r['point']} | {r['per_card'] / 1e6:.2f}M | {common.CASE_LABELS.get(r['per_card_cube'], r['per_card_cube'])}"
                     f"{' *' if 'stand-in' in r['per_card_cube_note'] else ''} (x {r['per_card_cube_scale']:.3f}) | {r['period_ms']:.3f} | {pct(r['ideal'])} | {pct(r['seam'])} | "
                     f"{pct(r['reference_point'])} | {pct(r['exposed'])} | {pct(r['imbalance'])} | {pct(r['sum_five'])} | {r['eta_trace']:.3f} | "
                     f"{r['eta']:.3f} ({r['eta_min']:.3f}–{r['eta_max']:.3f}; {r['eta_trials']}) | "
                     f"{pct(r['g1_share'], 2) if r['g1_share'] is not None else '-'} | {r['r_p50']:.3f} ({r['r_p95']:.3f}) | "
                     f"{clock.get('mean', float('nan')):.4f} ({clock.get('min', float('nan')):.4f}–{clock.get('max', float('nan')):.4f}) |")
    lines += ["", "\\* stand-in: " + "; ".join(f"{r['point']}: {r['per_card_cube_note']}" for r in rows if "stand-in" in r["per_card_cube_note"]), "",
              "### The terms in ms", "",
              "| point | P (window − defrag; median) | busy (mean of cards) | tau_pc = c_pc x N/K (c_pc ns/particle, cube trace ms) | tau(N) / K | exposed | P − busy − exposed | T_B (mean of cards' medians) | "
              "t_tr worst link median = readback + host + upload | t_tr p95 | reference trace |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['point']} | {r['period_ms']:.3f} ({r['period_window_ms']:.3f}; {r['period_p50_ms']:.3f}) | {r['busy_ms']:.3f} | {r['tau_pc_ms']:.3f} "
                     f"({r['c_pc_ns']:.3f}, {r['tau_pc_cube_ms']:.3f}) | {r['tau_ref_ms']:.3f} | {r['exposed_ms']:.3f} | "
                     f"{r['period_ms'] - r['busy_ms'] - r['exposed_ms']:.3f} | {r['T_B_us'] / 1e3:.3f} | {r['t_tr_us'] / 1e3:.3f} = "
                     f"{r['t_readback_us'] / 1e3:.3f} + {r['t_host_us'] / 1e3:.3f} + {r['t_upload_us'] / 1e3:.3f} | {r['t_tr_p95_us'] / 1e3:.3f} | "
                     f"{r['reference_kind']} ({r['reference_trace_job']}) |")
    lines += ["", "### Per card (ms): period mean / busy mean / exposed mean (p50) / period − busy − exposed", ""]
    for r in rows:
        cells = []
        for c in r["cards"]:
            e = r["exposure"].get(c["sim"], {})
            e_mean = (e.get("mean_us") or 0.0) / 1e3
            cells.append(f"s{c['sim']}: {c['period_mean']:.3f} / {c['busy_mean']:.3f} / {e_mean:.3f} ({(e.get('p50_us') or 0.0) / 1e3:.3f}) / "
                         f"{c['period_mean'] - c['busy_mean'] - e_mean:.3f}")
        lines.append(f"- {r['point']}: " + "; ".join(cells))
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------- 2. phases blocks
def phase_blocks(data: dict, tables: dict) -> tuple:
    blocks, results, csv_rows = [], [], []
    header = ("| case | row | A | B | C | step median | step mean (window − defrag) | step mean (skip defrag step) |", "|---|---|---|---|---|---|---|---|")
    for name, pc_case, note in PHASE_BLOCKS:
        point = data["points"][name]
        case, slabs = point["case"], point["slabs"]
        trace_row = (point["traces"].get("C") or point["traces"].get("E") or {}).get("row")
        chain = phases.analyze_trace(point["trace_dir"], trace_row)
        pc_path, pc_row, pc_job = data["k1"][pc_case]
        scale = (common.particles_of(case) / slabs) / common.particles_of(pc_case)      # 1.000 for 64M -> 8M, 0.244 for 8M -> 4M
        ideal = phases.analyze_trace(pc_path, pc_row, divide=1.0 / scale)
        raw = phases.analyze_trace(pc_path, pc_row, divide=1)
        full_path, full_row, full_job = data["k1"][case]
        divided = phases.analyze_trace(full_path, full_row, divide=slabs)
        entry = tables_point(tables, name)
        label = f"strong {common.CASE_LABELS[case]}"

        def cell(stat, single=False):
            if not stat:
                return "-"
            return f"{stat['mean']:.3f}" if single else f"{stat['mean']:.3f} ({stat['min']:.3f}–{stat['max']:.3f})"

        def row(text, agg, single=False):
            return (f"| {label} | {text} | {cell(agg['a'], single)} | {cell(agg['b'], single)} | {cell(agg['c'], single)} | {cell(agg['period'], single)} | "
                    f"{agg['step_mean_window']:.3f} | {agg['step_mean_skip']:.3f} |")
        ratio_ideal = ideal["aggregate"]["step_mean_window"] / chain["aggregate"]["step_mean_window"]
        ratio_divided = divided["aggregate"]["step_mean_window"] / chain["aggregate"]["step_mean_window"]
        scale_text = "not divided" if abs(scale - 1.0) < 1e-9 else f"÷ {1.0 / scale:.3f} = x {common.particles_of(case) // slabs:,} / {common.particles_of(pc_case):,} (the per-card load at the cube's per-particle cost)"
        lines = [f"**{name}** — {label} K = {slabs} ({point['job']['name']}, {entry.get('host')}); ideal chain = K = 1 trace of the per-card-size cube "
                 f"{common.CASE_LABELS[pc_case]} ({note}; trace_{common.short_case(pc_case)}_K1 of {pc_job}, one card, {scale_text})", "", *header,
                 row(f"ideal chain: K = 1 trace of {common.CASE_LABELS[pc_case]}, {scale_text} ({note})", ideal["aggregate"], single=True),
                 row(f"chain K = {slabs} ({point['job']['name']}, {entry.get('host')})", chain["aggregate"]),
                 row(f"for comparison: K = 1 trace of {common.CASE_LABELS[case]} ÷ {slabs} (the 7.3 ideal row)", divided["aggregate"], single=True)]
        if abs(scale - 1.0) >= 1e-9:
            lines.append(row(f"for reference: the {common.CASE_LABELS[pc_case]} K = 1 trace raw, not divided", raw["aggregate"], single=True))
        lines += ["", f"ratio ideal / chain (window − defrag means) = **{ratio_ideal:.4f}** (per-card-size reference, the eta' of the three-factor "
                  f"decomposition); whole-cube ÷ {slabs} / chain = {ratio_divided:.4f} (the trace's eta; timed eta {entry['stats']['eta']['mean']:.4f}); "
                  f"{len(chain['aggregate']['defrag_events'])} defrag event(s) in the chain window, {len(ideal['aggregate']['defrag_events'])} in the ideal's."]
        blocks.append("\n".join(lines))
        results.append({"point": name, "case": case, "K": slabs, "per_card_cube": pc_case, "note": note, "scale": scale, "chain": chain, "ideal": ideal,
                        "ideal_raw": raw, "whole_cube_divided": divided, "ratio_ideal_over_chain": ratio_ideal, "ratio_divided_over_chain": ratio_divided,
                        "eta_timed": entry["stats"]["eta"]["mean"]})
        for side, agg_of in (("ideal_per_card_cube", ideal), ("chain", chain), ("whole_cube_divided", divided), ("per_card_cube_raw", raw)):
            a = agg_of["aggregate"]
            csv_rows.append([name, case, slabs, side, pc_case if side.startswith("ideal") else "", a["divide"], (a["a"] or {}).get("mean"),
                             (a["b"] or {}).get("mean"), (a["c"] or {}).get("mean"), (a["period"] or {}).get("mean"), a["step_mean_window"], a["step_mean_skip"]])
    return blocks, results, csv_rows


# ----------------------------------------------------------------------------------------- 3. figure
def figure_rows_3d(rows: list, tables: dict) -> list:
    out = []
    for r in rows:
        if r["kind"] != "strong":
            continue
        out.append({"dimension": "3D", "case": r["case"], "scaling": "strong", "K": r["K"], "point": r["point"], "job": r["job"],
                    "per_gpu": r["per_card"], "eta": r["eta"], "eta_min": r["eta_min"], "eta_max": r["eta_max"],
                    "T_B": r["T_B_us"], "t_tr": r["t_tr_us"], "t_host": r["t_host_us"], "r": r["r_p50"], "r_p95": r["r_p95"]})
    out.sort(key=lambda r: r["per_gpu"])
    return out


def figure_rows_2d(path: pathlib.Path) -> list:
    document = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for p in document["points"]:
        if p["scaling"] != "strong" or not p.get("trace"):
            continue
        out.append({"dimension": "2D", "case": p["case"], "scaling": "strong", "K": p["K"], "point": p["point"], "job": p["eta_job"],
                    "per_gpu": p["per_gpu"], "eta": p["eta"], "eta_min": p["eta_min"], "eta_max": p["eta_max"],
                    "T_B": p["T_B"], "t_tr": p["t_own"], "t_host": p["t_host"], "r": p["r_own"], "r_p95": p["r_own_p95"]})
    out.sort(key=lambda r: r["per_gpu"])
    return out


def crossing(points: list) -> list:
    out = []
    for low, high in zip(points, points[1:]):
        f_low = math.log(low["t_tr"]) - math.log(low["T_B"])
        f_high = math.log(high["t_tr"]) - math.log(high["T_B"])
        if f_low == 0.0 or f_low * f_high > 0:
            continue
        share = f_low / (f_low - f_high)
        u = math.log(low["per_gpu"]) + share * (math.log(high["per_gpu"]) - math.log(low["per_gpu"]))
        value = math.exp(math.log(low["T_B"]) + share * (math.log(high["T_B"]) - math.log(low["T_B"])))
        out.append({"between": [low["case"], high["case"]], "per_device": math.exp(u), "time_us": value})
    return out


def figure(two_d: list, three_d: list, out: pathlib.Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullFormatter
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
    # (a) times against particles per device: 2-D circles (as before), 3-D squares
    series = ((two_d, "o", 3.4, 0.8), (three_d, "s", 3.1, 0.8))
    for rows, marker, size, edge in series:
        xs = [row["per_gpu"] / 1e6 for row in rows]
        axis_t.plot(xs, [row["T_B"] / 1e3 for row in rows], color=style.COLOR_MAIN, marker=marker, ms=size, mfc=style.COLOR_MAIN,
                    mec=style.COLOR_MAIN, lw=0.9, zorder=3)
        axis_t.plot(xs, [row["t_tr"] / 1e3 for row in rows], color=style.COLOR_RED, marker=marker, ms=size, mfc="white",
                    mec=style.COLOR_RED, mew=edge, lw=0.9, zorder=3)
        host = dict(color=style.COLOR_RED, ls=(0, (3.0, 2.0)), lw=0.9, zorder=2.5)
        if marker == "s":
            host.update(marker="s", ms=2.3, mfc="white", mec=style.COLOR_RED, mew=0.6)
        axis_t.plot(xs, [row["t_host"] / 1e3 for row in rows], **host)
    axis_t.set_yscale("log")
    axis_t.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    axis_t.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=12))
    axis_t.yaxis.set_minor_formatter(NullFormatter())
    axis_t.yaxis.set_major_formatter(plain)
    xs_all = [row["per_gpu"] / 1e6 for row in two_d + three_d]
    ys_all = [row[key] / 1e3 for row in two_d + three_d for key in ("T_B", "t_tr", "t_host")]
    axis_t.set_xlim(min(xs_all) / 1.7, max(xs_all) * 1.7)
    axis_t.set_ylim(min(ys_all) / 1.8, max(ys_all) * 2.2)
    axis_t.set_xlabel("particles per device (M)", labelpad=1.5)
    axis_t.set_ylabel("time per step (ms)", labelpad=1.5)
    quantity_handles = [Line2D([], [], color=style.COLOR_MAIN, marker="o", ms=3.4, mfc=style.COLOR_MAIN, mec=style.COLOR_MAIN, lw=0.9, label=r"$T_B$"),
                        Line2D([], [], color=style.COLOR_RED, marker="o", ms=3.4, mfc="white", mec=style.COLOR_RED, mew=0.8, lw=0.9, label=r"$t_\mathrm{tr}$"),
                        Line2D([], [], color=style.COLOR_RED, ls=(0, (3.0, 2.0)), lw=0.9, label=r"host part of $t_\mathrm{tr}$")]
    dimension_handles = [Line2D([], [], color="#444444", marker="o", ms=3.4, mfc="#444444", mec="#444444", lw=0, label="2D"),
                         Line2D([], [], color="#444444", marker="s", ms=3.1, mfc="#444444", mec="#444444", lw=0, label="3D")]
    first = axis_t.legend(handles=quantity_handles, loc="upper left", frameon=False, handlelength=1.6, borderaxespad=0.2)
    axis_t.add_artist(first)
    axis_t.legend(handles=dimension_handles, loc="lower right", frameon=False, handlelength=1.0, borderaxespad=0.2, handletextpad=0.5)
    # (b) eta against r: 2-D circles, 3-D squares; band eta = 1 +- 0.02, dashed line at r = 1
    style.band(axis_r)
    axis_r.axvline(1.0, color="#555555", lw=0.6, ls=(0, (3.0, 2.0)), zorder=0.6)
    for rows, marker, size, label in ((two_d, "o", 3.4, "2D"), (three_d, "s", 3.1, "3D")):
        axis_r.errorbar([row["r"] for row in rows], [row["eta"] for row in rows],
                        yerr=[[row["eta"] - row["eta_min"] for row in rows], [row["eta_max"] - row["eta"] for row in rows]],
                        capsize=1.6, elinewidth=0.6, capthick=0.6, ls="none", color=style.COLOR_MAIN, marker=marker, ms=size,
                        mfc=style.COLOR_MAIN, mec=style.COLOR_MAIN, label=label)
    r_values = [row["r"] for row in two_d + three_d]
    axis_r.set_xlim(min(r_values) / 1.8, max(r_values) * 1.8)
    axis_r.set_ylim(0.0, 1.05)
    axis_r.yaxis.set_major_locator(MultipleLocator(0.2))
    axis_r.set_xlabel(r"$r$", labelpad=1.5)
    axis_r.set_ylabel(r"$\eta$", labelpad=1.5)
    axis_r.legend(handles=[Line2D([], [], color=style.COLOR_MAIN, marker="o", ms=3.4, mfc=style.COLOR_MAIN, mec=style.COLOR_MAIN, lw=0, label="2D"),
                           Line2D([], [], color=style.COLOR_MAIN, marker="s", ms=3.1, mfc=style.COLOR_MAIN, mec=style.COLOR_MAIN, lw=0, label="3D")],
                  loc="lower right", frameon=False, handlelength=1.0, borderaxespad=0.2, handletextpad=0.5)
    style.panel_label(axis_t, "(a)")
    style.panel_label(axis_r, "(b)")
    style.save(fig, out / "loss_sources.pdf")
    plt.close(fig)


def write_figure_csv(rows: list, path: pathlib.Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("dimension", "case", "scaling", "K", "M_per_device", "T_B", "t_tr", "t_host", "r", "eta", "eta_min", "eta_max"))
        for row in rows:
            writer.writerow((row["dimension"], row["case"], row["scaling"], row["K"], f"{row['per_gpu'] / 1e6:.4f}", f"{row['T_B'] / 1e3:.4f}",
                             f"{row['t_tr'] / 1e3:.4f}", f"{row['t_host'] / 1e3:.4f}", f"{row['r']:.4f}", f"{row['eta']:.4f}",
                             f"{row['eta_min']:.4f}", f"{row['eta_max']:.4f}"))


# ----------------------------------------------------------------------------------------- 4. 2-D checks
def two_d_g1(logs: pathlib.Path) -> dict:
    """G1 redundancy share of 2-D 64M K8 (x_2d_n8000_K8): per sim the boundary correction + density kernel's us from
    the anatomy run's sampled frames (the chain trace has phase ticks only), the step period from the chain trace
    (mean over the kept steady steps), n0 (the inner replica column) from the trace's host copy bytes:
    B = 32 (n0 + n1) + 44 n_mig + 16 Ny Nz + 16 with n0 = n1 and n_mig = 0 at the median; the band per peer side =
    2 own columns + n0 = 3 n0 (uniform columns: own column = replica column); share = sides x n0 x (boundary us /
    (sides x 3 n0)) / period = boundary us / (3 period)."""
    job_dir, trace_name, chain_log, anatomy_log = TWO_D_CHAIN
    directory = logs / job_dir
    run = common.load_trace(directory / "traces" / trace_name)
    meta = run["meta"]
    grid = GRID_LINE.search((directory / chain_log).read_text(encoding="utf-8", errors="replace"))
    nx, ny, nz = (int(v) for v in grid.groups())
    fixed = 16 * ny * nz + 16
    metrics = step_trace_model.link_metrics(run)
    bytes_p50 = statistics.median(float(np.nanmedian(values["host_copy_bytes"])) for values in metrics.values())
    n0 = (bytes_p50 - fixed) / 64.0                     # B - fixed = 32 (n0 + n1), n0 = n1
    cards = card_table(run, [])
    samples: dict = {}
    for line in (directory / anatomy_log).read_text(encoding="utf-8", errors="replace").splitlines():
        match = ANATOMY_LINE.match(line.strip())
        if not match:
            continue
        frame, sim, rest = int(match.group(1)), int(match.group(2)), match.group(3)
        values = {key: float(value) for key, value in re.findall(r"(\w+)=([\d.]+)", rest)}
        samples.setdefault(sim, []).append({"frame": frame, **values})
    sims = []
    for card in cards:
        sim = card["sim"]
        frames = samples.get(sim, [])
        boundary = [f.get("correction_density_boundary") for f in frames if f.get("correction_density_boundary") is not None]
        if not boundary:
            continue
        boundary_us = statistics.fmean(boundary)
        sides = 1 if sim in (0, len(cards) - 1) else 2
        band = sides * 3 * n0
        cost = boundary_us / band
        redundant = sides * n0 * cost
        period_us = card["period_mean"] * 1e3
        sims.append({"sim": sim, "frames": len(boundary), "boundary_us": boundary_us, "sides": sides, "n0": n0, "band_particles": band,
                     "cost_ns_per_particle": cost * 1e3, "redundant_us": redundant, "period_us": period_us, "share": redundant / period_us,
                     "boundary_share_of_step": boundary_us / period_us})
    shares = [s["share"] for s in sims]
    return {"trace": str(directory / "traces" / trace_name), "anatomy_log": str(directory / anatomy_log), "grid": [nx, ny, nz], "fixed_bytes": fixed,
            "host_copy_bytes_p50": bytes_p50, "n0": n0, "own_particles_interior": 200 * n0, "particle_share_interior": 2 * n0 / (200 * n0),
            "sims": sims, "share_mean": statistics.fmean(shares) if shares else None, "share_min": min(shares) if shares else None,
            "share_max": max(shares) if shares else None, "detail": (meta.get("clock") or {}).get("detail")}


def two_d_clocks(tables_path: pathlib.Path, figure_csv: pathlib.Path) -> list:
    tables = json.loads(tables_path.read_text(encoding="utf-8"))
    chosen, seen = [], set()
    for row in csv.DictReader(open(figure_csv, encoding="utf-8", newline="")):
        if row["figure"] == "strong_scaling" and row["K"] == "8" and row["case"] not in seen:      # one row per case (both panels list 8000)
            chosen.append(row)
            seen.add(row["case"])
    by_key = {(p["job"], p["point"], p["arm"]): p for p in tables["points"]}
    out = []
    for row in chosen:
        point = by_key[(row["job"], row["point"], "C")]
        reference = row["reference"]
        values = [trial["references"][reference]["clock_ratio"] for trial in point["trials"]
                  if trial.get("references", {}).get(reference, {}).get("clock_ratio") is not None]
        clocks_k = [trial["sm_clock"] for trial in point["trials"]]
        clocks_1 = [trial["references"][reference]["sm_clock_1"] for trial in point["trials"] if reference in trial.get("references", {})]
        out.append({"case": row["case"], "point": row["point"], "job": row["job"], "host": row["host"], "reference": reference,
                    "per_gpu": float(row["per_gpu"]), "trials": len(values), "ratio_mean": statistics.fmean(values), "ratio_min": min(values),
                    "ratio_max": max(values), "ratios": values, "sm_clock_chain": statistics.fmean(clocks_k), "sm_clock_reference": statistics.fmean(clocks_1)})
    out.sort(key=lambda r: r["per_gpu"])
    return out


K_SERIES_JOBS = ("e7_b2A_1694327", "e7_b2V_1703604", "e7_b2B_1694328")      # the A line of the E7 report: 2d_n8000 K = 2 .. 8 on g2


def two_d_clocks_k_series(tables_path: pathlib.Path) -> list:
    """The 64M (2d_n8000) K = 2 .. 8 series on wqd10nbj04g2 (K = 2, 4 from 1694327; 3, 5, 6, 7 from 1703604; 8 from 1694328),
    the same per-trial ratio."""
    tables = json.loads(tables_path.read_text(encoding="utf-8"))
    out = []
    for point in tables["points"]:
        if point["case"] != "cavity2d_n8000" or point["arm"] != "C" or point["job"] not in K_SERIES_JOBS or not point.get("trials"):
            continue
        reference = next((name for name in point["references"] if name.startswith("strong:")), None)
        if reference is None:
            continue
        values = [trial["references"][reference]["clock_ratio"] for trial in point["trials"]
                  if trial.get("references", {}).get(reference, {}).get("clock_ratio") is not None]
        if not values:
            continue
        out.append({"case": point["case"], "K": point["K"], "point": point["point"], "job": point["job"], "host": point["host"], "trials": len(values),
                    "ratio_mean": statistics.fmean(values), "ratio_min": min(values), "ratio_max": max(values), "ratios": values,
                    "sm_clock_chain": statistics.fmean(trial["sm_clock"] for trial in point["trials"]),
                    "sm_clock_reference": statistics.fmean(trial["references"][reference]["sm_clock_1"] for trial in point["trials"] if reference in trial.get("references", {}))})
    out.sort(key=lambda r: r["K"])
    return out


# ----------------------------------------------------------------------------------------- 5. memcpy
def memcpy_of(data: dict, name: str) -> dict:
    point = data["points"][name]
    run = common.load_trace(point["trace_dir"])
    metrics = step_trace_model.link_metrics(run)
    pooled = np.concatenate([values["memcpy"] for values in metrics.values()])
    pooled = pooled[np.isfinite(pooled)]
    bytes_pooled = np.concatenate([values["host_copy_bytes"] for values in metrics.values()])
    bytes_p50 = float(np.nanmedian(bytes_pooled))
    p50, p95 = float(np.percentile(pooled, 50)), float(np.percentile(pooled, 95))
    per_link = {name_: {"sender": int(values["sender"]), "receiver": int(values["receiver"]), "p50_us": float(np.nanmedian(values["memcpy"])),
                        "p95_us": float(np.nanpercentile(values["memcpy"], 95)), "bytes_p50": float(np.nanmedian(values["host_copy_bytes"]))}
                for name_, values in metrics.items()}
    return {"point": name, "K": point["slabs"], "job": point["job"]["name"], "links": len(metrics), "samples": int(len(pooled)),
            "bytes_p50": bytes_p50, "memcpy_p50_us": p50, "memcpy_p95_us": p95, "gb_per_s_at_p50": bytes_p50 / (p50 * 1e3),
            "link_p50_min_us": min(v["p50_us"] for v in per_link.values()), "link_p50_max_us": max(v["p50_us"] for v in per_link.values()),
            "per_link": per_link}


# ----------------------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--job", action="append", required=True)
    parser.add_argument("--tables", required=True)
    parser.add_argument("--two-d-json", required=True)
    parser.add_argument("--two-d-tables", required=True)
    parser.add_argument("--two-d-figure-csv", required=True)
    parser.add_argument("--two-d-logs", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--figure-out", action="append", default=[])
    arguments = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    style.setup_style()
    out = pathlib.Path(arguments.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = json.loads(pathlib.Path(arguments.tables).read_text(encoding="utf-8"))
    data = collect_jobs(arguments.job)
    # 1. ledger
    rows = []
    for name, pc_case, note in LEDGER:
        rows.append(ledger_row(name, pc_case, note, data, tables))
        r = rows[-1]
        print(f"[ledger] {name}: P {r['period_ms']:.3f} ms, ideal {r['ideal'] * 100:.1f} seam {r['seam'] * 100:.1f} ref {r['reference_point'] * 100:+.1f} "
              f"exposed {r['exposed'] * 100:.1f} imbalance {r['imbalance'] * 100:.1f} (sum {r['sum_five'] * 100:.1f}); eta_trace {r['eta_trace']:.3f} "
              f"eta {r['eta']:.3f}; G1 {pct(r['g1_share'], 2)}; r {r['r_p50']:.3f}; clock {r['clock_ratio'].get('mean', float('nan')):.4f}", flush=True)
    ledger_text = ledger_markdown(rows)
    common.write_text(out / "ledger.md", ledger_text)
    common.write_json(out / "ledger.json", rows)
    with (out / "ledger.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["point", "case", "K", "kind", "per_card", "per_card_cube", "period_ms", "busy_ms", "tau_pc_ms", "tau_ref_ms", "exposed_ms",
                         "ideal", "seam", "reference_point", "exposed", "imbalance", "sum_five", "eta_trace", "eta", "eta_min", "eta_max", "g1_share",
                         "r_p50", "r_p95", "clock_ratio_mean", "clock_ratio_min", "clock_ratio_max"])
        for r in rows:
            writer.writerow([r["point"], r["case"], r["K"], r["kind"], r["per_card"], r["per_card_cube"], r["period_ms"], r["busy_ms"], r["tau_pc_ms"],
                             r["tau_ref_ms"], r["exposed_ms"], r["ideal"], r["seam"], r["reference_point"], r["exposed"], r["imbalance"], r["sum_five"],
                             r["eta_trace"], r["eta"], r["eta_min"], r["eta_max"], r["g1_share"], r["r_p50"], r["r_p95"],
                             r["clock_ratio"].get("mean"), r["clock_ratio"].get("min"), r["clock_ratio"].get("max")])
    # 2. phases blocks
    blocks, phase_results, phase_csv = phase_blocks(data, tables)
    common.write_text(out / "phases_3d.md", "## 2. tab:phases, 3-D row blocks (ideal chain = the per-card-size cube's K = 1 trace, not divided)\n\n"
                      "Rules of e8_phases_table / e7_phases_table: steady steps = step >= warmup with the next step present, minus the defrag step; "
                      "A / B / C and the step median = each card's median, then the mean over the cards (min–max); step mean (window) = (sum of the "
                      "steady periods − in-window defrag wall time) / steps; step mean (skip) = the mean of the kept steps. ms.\n\n" + "\n\n".join(blocks))
    common.write_json(out / "phases_3d.json", phase_results)
    with (out / "phases_3d.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["point", "case", "K", "row", "per_card_cube", "divide", "T_A_ms", "T_B_ms", "T_C_ms", "period_median_ms", "step_mean_window_ms", "step_mean_skip_ms"])
        writer.writerows(phase_csv)
    print("\n".join(blocks), flush=True)
    # 3. figure
    two_d = figure_rows_2d(pathlib.Path(arguments.two_d_json))
    three_d = figure_rows_3d(rows, tables)
    crossings = {"2D": crossing(two_d), "3D": crossing(three_d)}
    figure_document = {"points": two_d + three_d, "t_tr_equals_T_B": crossings,
                       "notes": "t_tr = E7's t_own (t_chain minus the transfer queues' idle waits), r = E7's r_own (worst link per step, median)"}
    common.write_json(out / "loss_sources_figure.json", figure_document)
    for directory in arguments.figure_out:
        target = pathlib.Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png", ".csv"):
            old, keep = target / f"loss_sources{suffix}", target / f"loss_sources_2d{suffix}"
            if old.exists() and not keep.exists():
                shutil.copy2(old, keep)
        figure(two_d, three_d, target)
        write_figure_csv(two_d + three_d, target / "loss_sources.csv")
    # 4. 2-D checks
    g1 = two_d_g1(pathlib.Path(arguments.two_d_logs))
    clocks = two_d_clocks(pathlib.Path(arguments.two_d_tables), pathlib.Path(arguments.two_d_figure_csv))
    lines = ["## 4. 2-D checks", "", "### G1 redundancy share, 2-D 64M K8 (x_2d_n8000_K8, job 1694328, wqd10nbj04g2)", "",
             f"Chain trace detail '{g1['detail']}' (phase ticks only), so the boundary correction + density kernel comes from the anatomy run's sampled "
             f"frames (3 per sim); period = the chain trace's per-sim mean over the kept steady steps; grid {g1['grid'][0]} x {g1['grid'][1]} x {g1['grid'][2]}, "
             f"16 Ny Nz + 16 = {g1['fixed_bytes']:,} B, host copy bytes p50 {g1['host_copy_bytes_p50']:,.0f} B -> n0 = n1 = {g1['n0']:,.0f} particles "
             f"(one voxel column = 5 layers x 8022); band per side = 3 n0; share = sides n0 c_b / period = boundary us / (3 period). "
             f"Particle share of the redundant work on an interior card: 2 n0 / (200 columns x n0) = {g1['particle_share_interior'] * 100:.2f} %.", "",
             "| sim | sides | boundary kernel (us, mean of frames) | boundary share of the step | c_b (ns / band particle) | redundant (us) | period (us) | G1 share |",
             "|---|---|---|---|---|---|---|---|"]
    for s in g1["sims"]:
        lines.append(f"| {s['sim']} | {s['sides']} | {s['boundary_us']:.1f} ({s['frames']}) | {s['boundary_share_of_step'] * 100:.2f} % | "
                     f"{s['cost_ns_per_particle']:.3f} | {s['redundant_us']:.1f} | {s['period_us']:,.0f} | {s['share'] * 100:.2f} % |")
    lines += ["", f"**G1 share, mean over the 8 sims: {g1['share_mean'] * 100:.2f} % ({g1['share_min'] * 100:.2f}–{g1['share_max'] * 100:.2f} %)**; "
              f"the paper's 'about 1 %' is the particle share ({g1['particle_share_interior'] * 100:.1f} %); in time it is {g1['share_mean'] * 100:.1f} %.", "",
              "### SM clock ratio chain / reference, the nine strong K = 8 2-D points (tables_b2.json: per trial the chain's loop-window SM clock mean "
              "over its 8 cards / the mean of the 8 reference runs' (4 pairs' for 11320) SM clock; mean over the trials)", "",
              "| case | per device | job (node) | trials | ratio mean (min–max) | ratio − 1 | chain SM MHz | reference SM MHz |", "|---|---|---|---|---|---|---|---|"]
    for c in clocks:
        lines.append(f"| {c['case']} | {c['per_gpu'] / 1e6:.2f}M | {c['job']} ({c['host']}) | {c['trials']} | {c['ratio_mean']:.4f} ({c['ratio_min']:.4f}–{c['ratio_max']:.4f}) | "
                     f"{(c['ratio_mean'] - 1) * 100:+.2f} % | {c['sm_clock_chain']:.0f} | {c['sm_clock_reference']:.0f} |")
    means = [c["ratio_mean"] for c in clocks]
    large = [c["ratio_mean"] for c in clocks if c["per_gpu"] >= 2e5]
    lines += ["", f"**Range of the point means: {(min(means) - 1) * 100:+.2f} % .. {(max(means) - 1) * 100:+.2f} %** over the nine sizes; "
              f"{(min(large) - 1) * 100:+.2f} % .. {(max(large) - 1) * 100:+.2f} % without the two smallest cases (520², 1000²: 0.04M / 0.13M per device, "
              "where the references' own clocks are off the 575 W regime)."]
    k_series = two_d_clocks_k_series(pathlib.Path(arguments.two_d_tables))
    lines += ["", "### The same ratio along the 64M K = 2 .. 8 series (2d_n8000, wqd10nbj04g2; the E7 report's A line)", "",
              "| K | point | job | trials | ratio mean (min–max) | ratio − 1 | chain SM MHz | reference SM MHz |", "|---|---|---|---|---|---|---|---|"]
    for c in k_series:
        lines.append(f"| {c['K']} | {c['point']} | {c['job']} | {c['trials']} | {c['ratio_mean']:.4f} ({c['ratio_min']:.4f}–{c['ratio_max']:.4f}) | "
                     f"{(c['ratio_mean'] - 1) * 100:+.2f} % | {c['sm_clock_chain']:.0f} | {c['sm_clock_reference']:.0f} |")
    if k_series:
        k_means = [c["ratio_mean"] for c in k_series]
        lines += ["", f"**K-series range: {(min(k_means) - 1) * 100:+.2f} % .. {(max(k_means) - 1) * 100:+.2f} %** — this is the paper's 0.9–2.7 %."]
    common.write_text(out / "two_d_checks.md", "\n".join(lines))
    common.write_json(out / "two_d_checks.json", {"g1": g1, "clocks": clocks, "k_series": k_series})
    print("\n".join(lines), flush=True)
    # 5. memcpy
    memcpy = [memcpy_of(data, name) for name in MEMCPY_POINTS if name in data["points"]]
    lines = ["## 5. memcpy of the 64M packet (41 MB) by K: median over every link and steady step; bandwidth at that median", "",
             "| point | K | links x samples | bytes p50 | memcpy p50 (us) | p95 | GB/s at p50 | per-link p50 range (us) |", "|---|---|---|---|---|---|---|---|"]
    for m in memcpy:
        lines.append(f"| {m['point']} | {m['K']} | {m['links']} x {m['samples'] // m['links']} | {m['bytes_p50']:,.0f} | {m['memcpy_p50_us']:,.0f} | {m['memcpy_p95_us']:,.0f} | "
                     f"{m['gb_per_s_at_p50']:.2f} | {m['link_p50_min_us']:,.0f}–{m['link_p50_max_us']:,.0f} |")
    common.write_text(out / "memcpy.md", "\n".join(lines))
    common.write_json(out / "memcpy.json", memcpy)
    print("\n".join(lines), flush=True)
    print(ledger_text[:6000])
    for dimension, items in crossings.items():
        for item in items:
            print(f"{dimension}: t_tr = T_B between {item['between'][0]} and {item['between'][1]}: {item['per_device'] / 1e6:.3f}M per device, {item['time_us'] / 1e3:.3f} ms")
    print(f"[e8_loss_3d] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
