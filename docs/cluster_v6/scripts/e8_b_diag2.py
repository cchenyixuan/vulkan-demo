"""
e8_b_diag2.py — part 2 of the E8 line-B diagnosis (user 2026-10-22 supplement, post-processing only):
  7. 4M K8 and 8M K8: the per-card phase table and the per-kernel table of part 1 (sections 4 / 4b), and for every K8 chain
     the per-link t_tr (= t_chain of E7: the three hops from the sender's send_end to the receiver's upload_end), t_own
     (t_chain minus the transfer queues' idle = waiting for the sender), T_B and r_own = t_own / T_B (E7's fig:loss
     convention); the per-card wait (period - busy) split into transfer exposure max(0, t_own - T_B) and phase coupling
     (the rest: waiting for a slower neighbour, including t_chain - t_own).
  8. the three-factor decomposition eta = balance x size effect x seam work of the five K8 points and the weak K8 (the
     per-card-size reference of 16M / 8M / 4M K8 replaced by the 4M cube, flagged).
  9. SM clock / power / temperature of the K = 1 references (8-card means) beside the chain's cards; the E7 2-D K = 1
     references (2.14M, 8.19M, 32M, 64M) per-particle cost on the cluster, and whether 2-D shows the size trend.
 10. the post-defrag slowdown: per-kernel medians of steps 500-999 vs 1000-1499 of the 8M and 64M K = 1 traces; the
     code paths of the bootstrap defrag and the frame-1000 defrag; the local 7900 XTX 2-D 1M K = 1 runs with defrag
     cadence 500 / 1000 / off; the E7 2-D K = 1 traces on the cluster binned the same way.
 11. what is missing.

Usage:
    python docs/cluster_v6/scripts/e8_b_diag2.py --job DIR [--xtx DIR] [--e7-results GLOB ...] [--e7-trace DIR ...]
        --md claude/e8_b_diag_part2.md [--json OUT.json] [--append-to claude/e8_b_diag.md]
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
import statistics
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e8_common as common  # noqa: E402
import e8_b_diag as diag  # noqa: E402
import e8_kernel_table as kernel_table  # noqa: E402
import e8_loss_sources as loss  # noqa: E402

K8_ALL = ("3d_n152_K8", "3d_n192_K8", "3d_n244_K8", "3d_n312_K8", "3d_n392_K8", "3d_n416_K8", "3d_n496_K8")
STANDIN = {"cavity3d_n244": "cavity3d_n152", "cavity3d_n192": "cavity3d_n152", "cavity3d_n152": "cavity3d_n152", "cavity3d_n416": "cavity3d_n192"}
E7_PARTICLES = {"cavity2d_n1000": 1_044_484, "cavity2d_n1440": 2_137_444, "cavity2d_n2000": 4_088_484, "cavity2d_n2840": 8_191_044,
                "cavity2d_n4000": 16_176_484, "cavity2d_n5640": 32_058_244, "cavity2d_n8000": 64_352_484}
E7_SHORT = {"cavity2d_n1000": "2-D 1.04M", "cavity2d_n1440": "2-D 2.14M", "cavity2d_n2000": "2-D 4.09M", "cavity2d_n2840": "2-D 8.19M",
            "cavity2d_n4000": "2-D 16.2M", "cavity2d_n5640": "2-D 32.1M", "cavity2d_n8000": "2-D 64.4M"}
E7_RESULTS_DEFAULT = ["C:/Users/cchen/PycharmProjects/vulkan-demo-e7/docs/cluster_v6/data/e7/batch2/results_*.jsonl",
                      "C:/Users/cchen/PycharmProjects/vulkan-demo-e7/docs/cluster_v6/data/e7/batch1/results_*.jsonl"]
E7_TRACES_DEFAULT = ["C:/Users/cchen/PycharmProjects/vulkan-demo/logs/n56/2026-10-12/02_e7_b2A_1694327/traces/trace_2d_n8000_K1",
                     "C:/Users/cchen/PycharmProjects/vulkan-demo/logs/n56/2026-10-19/01_e7_b2AA_1716356/traces/reftrace_2d_n1440_K1_g0",
                     "C:/Users/cchen/PycharmProjects/vulkan-demo/logs/n56/2026-10-19/01_e7_b2AA_1716356/traces/reftrace_2d_n1440_K1_g3"]
PART2_MARKER = "<!-- e8_b_diag part 2 -->"


def f(value, digits=3):
    return common.fmt(value, digits)


def pct(value, digits=1):
    return "-" if value is None else f"{value * 100:.{digits}f} %"


# ---------------------------------------------------------------- 7. chains: phases, kernels, links, wait split
def chain_phases(job: dict, point: str) -> dict:
    trace_dir = job["traces"].get(f"{point}_C")
    if not trace_dir:
        return {}
    phases = diag.trace_phases(trace_dir)
    partition = job["partitions"].get(point) or {}
    by_slab = {s["slab"]: s for s in partition.get("slabs") or []}
    for sim in phases["sims"]:
        slab = by_slab.get(sim["sim"], {})
        sim["own_columns"], sim["fluid_columns"], sim["own_particles"] = slab.get("own_columns"), slab.get("fluid_columns"), slab.get("own_particles")
        sim["seams"] = 1 if sim["sim"] in (0, len(phases["sims"]) - 1) else 2
        sim["ns_per_own"] = sim["busy_ms"] * 1e6 / sim["own_particles"] if sim.get("own_particles") else None
    if phases["sims"]:
        slowest = max(phases["sims"], key=lambda s: s["busy_ms"])
        period = statistics.fmean(s["period_ms"] for s in phases["sims"])
        busy_mean = statistics.fmean(s["busy_ms"] for s in phases["sims"])
        phases["slowest"] = {k: slowest.get(k) for k in ("sim", "device", "busy_ms", "own_columns", "fluid_columns", "seams", "own_particles", "ns_per_own")}
        phases["period_mean_of_cards_ms"], phases["busy_mean_of_cards_ms"] = period, busy_mean
        phases["period_minus_slowest_busy_ms"] = period - slowest["busy_ms"]
        phases["balance"] = busy_mean / period
        own_total = sum(s["own_particles"] for s in phases["sims"] if s.get("own_particles"))
        phases["ns_per_own_mean"] = busy_mean * 1e6 / (own_total / len(phases["sims"])) if own_total else None
    phases["fps_nodefrag"] = common.fps_pair(job["rows"].get(f"{point}_trace_C"))[1]
    return phases


def chain_kernels(job: dict, point: str, k1_kernels: dict) -> dict:
    case, slabs = common.case_of_point(point)
    trace_dir = job["traces"].get(f"{point}_C")
    if not trace_dir:
        return {}
    try:
        k8 = kernel_table.analyze(trace_dir, case, job["raw"].get(f"{point}_trace_C"), job["partitions"].get(point))
    except Exception as error:                                              # noqa: BLE001
        return {"error": repr(error)}
    k1 = k1_kernels.get(case)
    if not k1 or k1.get("error"):
        name = f"trace_{common.short_case(case)}_K1"
        try:
            k1 = kernel_table.analyze(job["traces"][name], case, None, None) if name in job["traces"] else {"error": "no K = 1 trace"}
        except Exception as error:                                          # noqa: BLE001
            k1 = {"error": repr(error)}
        k1_kernels[case] = k1
    names = list(k8["kernel_order"]) + [n for n in (k1.get("kernel_order") or []) if n not in k8["kernel_order"]]
    rows = []
    for name in names:
        per_card = [entry for sim in k8["sims"] for entry in sim["kernels"].values() if entry["kernel"] == name and entry.get("ns_per_particle") is not None]
        k1_entry = (k1.get("mean_over_sims") or {}).get(name) if not k1.get("error") else None
        rows.append({"kernel": name, "phase": (per_card[0]["phase"] if per_card else k1_entry["phase"]), "k1_ns": k1_entry["ns_per_particle_mean"] if k1_entry else None,
                     "k8_ns_min": min(e["ns_per_particle"] for e in per_card) if per_card else None,
                     "k8_ns_mean": statistics.fmean(e["ns_per_particle"] for e in per_card) if per_card else None,
                     "k8_ns_max": max(e["ns_per_particle"] for e in per_card) if per_card else None,
                     "k8_us_per_card": [round(e["us_p50"], 1) for e in per_card]})
    phase_of = lambda n: (k8["mean_over_sims"].get(n) or {}).get("phase")           # noqa: E731
    combos = [("correction_density（interior + boundary）", ["correction_density_interior", "correction_density_boundary", "correction_boundary", "density_boundary"]),
              ("force（deep interior + boundary）", ["force_deep_interior", "force"]),
              ("A 段合计", [n for n in names if phase_of(n) == "A"]), ("B 段合计", [n for n in names if phase_of(n) == "B"]), ("C 段合计", [n for n in names if phase_of(n) == "C"]),
              ("G1 簿记", ["ghost_leading", "ghost_trailing", "expand", "install_leading", "install_trailing", "append_departed", "band_compact", "deep_wall_marker", "wall_extrapolate"]),
              ("全部核之和", names)]
    for title, members in combos:
        per_card = [sum(e["ns_per_particle"] for e in sim["kernels"].values() if e["kernel"] in members and e.get("ns_per_particle") is not None) for sim in k8["sims"]]
        per_card = [v for v in per_card if v]
        k1_values = [(k1.get("mean_over_sims") or {}).get(n, {}).get("ns_per_particle_mean") for n in members] if not k1.get("error") else []
        k1_values = [v for v in k1_values if v is not None]
        rows.append({"kernel": title, "phase": "Σ", "combo": True, "k1_ns": sum(k1_values) if k1_values else None,
                     "k8_ns_min": min(per_card) if per_card else None, "k8_ns_mean": statistics.fmean(per_card) if per_card else None,
                     "k8_ns_max": max(per_card) if per_card else None, "k8_us_per_card": []})
    return {"case": case, "rows": rows, "g1_share_mean": k8.get("g1_share_mean"), "g1_share_range": k8.get("g1_share_range"), "sims_own": [s.get("own_particles") for s in k8["sims"]]}


def chain_links(job: dict, point: str, phases: dict) -> dict:
    """E7's per-link metrics (medians over the steady steps) and the per-card wait split."""
    trace_dir = job["traces"].get(f"{point}_C")
    if not trace_dir:
        return {}
    try:
        result = loss.analyze(trace_dir)
    except Exception as error:                                              # noqa: BLE001
        return {"error": repr(error)}
    links = {}
    for name, values in result["links"].items():
        links[name] = {"sender": values["sender"], "receiver": values["receiver"],
                       **{key: values.get(key + "_p50") for key in ("t_chain", "t_own", "T_B", "r_chain", "r_own", "own_readback", "own_host", "memcpy", "own_upload", "host_copy_bytes")}}
        # the link metrics are in us (step_trace_model); T_B of the receiver in us too
        links[name]["t_chain_minus_T_B_us"] = (values["t_chain_p50"] - values["T_B_p50"]) if values.get("t_chain_p50") is not None and values.get("T_B_p50") is not None else None
        links[name]["t_own_minus_T_B_us"] = (values["t_own_p50"] - values["T_B_p50"]) if values.get("t_own_p50") is not None and values.get("T_B_p50") is not None else None
    cards = []
    for sim in phases.get("sims", []):
        incoming = [l for l in links.values() if l["receiver"] == sim["sim"]]
        exposure_own = max([max(0.0, l["t_own_minus_T_B_us"]) for l in incoming if l["t_own_minus_T_B_us"] is not None] or [0.0]) / 1e3
        exposure_chain = max([max(0.0, l["t_chain_minus_T_B_us"]) for l in incoming if l["t_chain_minus_T_B_us"] is not None] or [0.0]) / 1e3
        total = sim["gap_mean_ms"]
        cards.append({"sim": sim["sim"], "device": sim["device"], "wait_total_ms": total, "wait_total_median_ms": sim["gap_ms"],
                      "wait_BC_ms": sim["wait_BC_ms"], "wait_CA_ms": sim["wait_CA_ms"],
                      "exposure_own_ms": exposure_own, "coupling_ms": max(0.0, total - exposure_own), "exposure_chain_ms": exposure_chain,
                      "incoming": [l["sender"] for l in incoming]})
    return {"links": links, "cards": cards, "r_own_p50": result.get("r_own_p50"), "r_own_p95": result.get("r_own_p95"),
            "r_chain_p50": result.get("r_chain_p50"), "r_chain_p95": result.get("r_chain_p95"), "steps": result.get("steps")}


# ---------------------------------------------------------------- 8. three factors
def three_factors(s1: dict, s2r: dict, chains: dict, weak_eta: list, weak_phases: dict) -> list:
    rows = []
    for point in K8_ALL:
        data = s1["points"].get(point)
        chain = chains.get(point) or {}
        if not data or not chain.get("sims"):
            continue
        case = data["case"]
        complete = [t for t in data["trials"] if t["complete"]]
        eta = statistics.fmean(t["eta_mean_nodefrag"] for t in complete) if complete else None
        per_card_cube = diag.PER_CARD_CUBE.get(case)
        standin = per_card_cube not in s2r
        cube = STANDIN.get(case) if standin else per_card_cube
        ns_same = (s2r.get(case) or {}).get("ns_mean")
        ns_pc = (s2r.get(cube) or {}).get("ns_mean")
        ns_own = chain.get("ns_per_own_mean")
        balance = chain.get("balance")
        tau_pc = (s1["cube_tau"].get(cube) or {}).get("mean_nodefrag")
        rows.append({"point": point, "case": case, "eta": eta, "balance": balance, "reference_cube": cube, "standin": standin,
                     "ns_same": ns_same, "ns_per_card": ns_pc, "ns_own": ns_own,
                     "size": ns_same / ns_pc if ns_same and ns_pc else None, "seam": ns_pc / ns_own if ns_pc and ns_own else None,
                     "seam_time": (tau_pc / chain["busy_mean_of_cards_ms"]) if tau_pc and not standin and chain.get("busy_mean_of_cards_ms") else None,
                     "product": balance * ns_same / ns_own if balance and ns_same and ns_own else None,
                     "busy_mean_ms": chain.get("busy_mean_of_cards_ms"), "period_ms": chain.get("period_mean_of_cards_ms")})
    if weak_phases and weak_phases.get("sims"):
        tau_8m = (s1["cube_tau"].get("cavity3d_n192") or {}).get("mean_nodefrag")
        ns_8m = (s2r.get("cavity3d_n192") or {}).get("ns_mean")
        rows.append({"point": "3d_n192_k8_K8", "case": "cavity3d_n192_k8", "eta": statistics.fmean(weak_eta) if weak_eta else None, "balance": weak_phases.get("balance"),
                     "reference_cube": "cavity3d_n192", "standin": False, "ns_same": None, "ns_per_card": ns_8m, "ns_own": weak_phases.get("ns_per_own_mean"),
                     "size": 1.0, "seam": (ns_8m / weak_phases["ns_per_own_mean"]) if ns_8m and weak_phases.get("ns_per_own_mean") else None,
                     "seam_time": (tau_8m / weak_phases["busy_mean_of_cards_ms"]) if tau_8m and weak_phases.get("busy_mean_of_cards_ms") else None,
                     "product": (weak_phases["balance"] * tau_8m / weak_phases["busy_mean_of_cards_ms"]) if tau_8m and weak_phases.get("busy_mean_of_cards_ms") else None,
                     "busy_mean_ms": weak_phases.get("busy_mean_of_cards_ms"), "period_ms": weak_phases.get("period_mean_of_cards_ms")})
    return rows


# ---------------------------------------------------------------- 9. clocks, E7 2-D
def clocks(job: dict, points: dict, s2r: dict) -> dict:
    out = {}
    for point in K8_ALL + ("3d_n192_k8_K8",):
        trials = points.get(point) or {}
        case, slabs = common.case_of_point(point)
        ref_cards, chain_cards = {}, {}
        for trial, runs in sorted(trials.items()):
            for card, label in runs["refs"].items():
                row = job["rows"].get(label) or {}
                clock = common.card_clock(row, card)
                if clock.get("sm_clock_median") is not None:
                    ref_cards.setdefault(card, []).append(clock)
            if runs["C"]:
                row = job["rows"].get(runs["C"]) or {}
                for card in row.get("device_map") or []:
                    clock = common.card_clock(row, card)
                    if clock.get("sm_clock_median") is not None:
                        chain_cards.setdefault(card, []).append(clock)
        trace_row = job["rows"].get(f"{point}_trace_C") or {}
        trace_cards = {card: common.card_clock(trace_row, card) for card in trace_row.get("device_map") or []}
        mean = lambda cards, key: statistics.fmean(v[key] for values in cards.values() for v in values if v.get(key) is not None) if any(v.get(key) is not None for values in cards.values() for v in values) else None  # noqa: E731
        per_card = lambda cards, key: {card: statistics.fmean(v[key] for v in values if v.get(key) is not None) for card, values in sorted(cards.items()) if any(v.get(key) is not None for v in values)}  # noqa: E731
        if ref_cards or chain_cards:
            out[point] = {"case": case, "ref": {key: mean(ref_cards, key) for key in ("sm_clock_median", "sm_clock_min", "power_median", "temperature_max")},
                          "ref_per_card": per_card(ref_cards, "sm_clock_median"),
                          "chain": {key: mean(chain_cards, key) for key in ("sm_clock_median", "sm_clock_min", "power_median", "temperature_max")},
                          "chain_per_card": per_card(chain_cards, "sm_clock_median"), "chain_power_per_card": per_card(chain_cards, "power_median"),
                          "trace_per_card": {card: v.get("sm_clock_median") for card, v in trace_cards.items()},
                          "ref_ns": (s2r.get(case) or {}).get("ns_mean")}
    return out


def e7_references(globs: list) -> dict:
    files = []
    for pattern in globs:
        files += sorted(glob.glob(pattern))
    by_case = {}
    for path in files:
        job = re.search(r"results_(\d+)", path)
        job = job.group(1) if job else pathlib.Path(path).parent.name
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            match = re.search(r"_R1_(2d_n\d+)_g(\d+)$", row.get("label", ""))
            if not match or row.get("status") != "pass":
                continue
            case = "cavity" + match.group(1)
            if case not in E7_PARTICLES:
                continue
            fps = row.get("steady_fps_nodefrag") or row.get("steady_fps")
            if not fps:
                continue
            card = str(row["device_map"][0]) if row.get("device_map") else match.group(2)
            telemetry = (row.get("telemetry_loop") or {}).get(card) or {}
            by_case.setdefault(case, []).append({"job": job, "label": row["label"], "node": row.get("node"), "steps": row.get("total_steps"),
                                                 "ns": 1e9 / fps / E7_PARTICLES[case], "fps_nodefrag": row.get("steady_fps_nodefrag"),
                                                 "clock": telemetry.get("sm_clock_median"), "power": telemetry.get("power_median"), "temperature": telemetry.get("temperature_max")})
    out = {}
    for case, rows in by_case.items():
        ns = [r["ns"] for r in rows]
        out[case] = {"particles": E7_PARTICLES[case], "runs": len(rows), "jobs": sorted({r["job"] for r in rows}), "nodes": sorted({r["node"] for r in rows if r["node"]}),
                     "steps": sorted({r["steps"] for r in rows if r["steps"]}), "ns_mean": statistics.fmean(ns), "ns_min": min(ns), "ns_max": max(ns), "ns_std": statistics.pstdev(ns),
                     "clock": statistics.fmean(r["clock"] for r in rows if r["clock"]) if any(r["clock"] for r in rows) else None,
                     "power": statistics.fmean(r["power"] for r in rows if r["power"]) if any(r["power"] for r in rows) else None}
    return out


def trace_bins(trace_dir, bins: int = 100) -> dict:
    """Per-100-step means of the step period (a_start differences) of sim 0 of a trace, the defrag step removed, normalised
    to 100-199; busy alongside."""
    run = common.load_trace(pathlib.Path(trace_dir))
    rows = common.sim_rows(run, 0)
    steps = rows["step"].astype(int)[:-1]
    period = np.diff(rows["a_start"]) / 1e6
    busy = ((rows["a_end"] - rows["a_start"]) + (rows["b_end"] - rows["b_start"]) + (rows["c_end"] - rows["c_start"]))[:-1] / 1e6
    median = float(np.median(period))
    out = []
    for start in range(0, int(steps.max()) + 1, bins):
        in_bin = (steps >= start) & (steps < start + bins)
        keep = in_bin & (period <= 2.0 * median)
        if keep.any():
            out.append({"bin": f"{start}-{start + bins - 1}", "period": float(np.mean(period[keep])), "busy": float(np.mean(busy[keep])), "removed": int(in_bin.sum() - keep.sum())})
    base = next((b["period"] for b in out if b["bin"].startswith("100-")), out[0]["period"] if out else None)
    for b in out:
        b["normalised"] = b["period"] / base if base else None
    return {"trace": str(trace_dir), "case": run["meta"].get("case"), "K": run["meta"].get("K"), "steps": int(len(steps) + 1), "warmup": run["meta"].get("warmup"),
            "defrag_cadence": run["meta"].get("defrag_cadence"), "bins": out}


# ---------------------------------------------------------------- 10. post-defrag
def kernel_halves(trace_dir, case: str, halves=diag.HALVES) -> dict:
    """Per-kernel medians (us) of sim 0 over the two halves of the window (the defrag step excluded), ns per particle."""
    run = common.load_trace(pathlib.Path(trace_dir))
    meta = run["meta"]
    labels = ((meta.get("labels") or {}).get("compute") or [[]])[0] or sorted(k for k in run["device"] if k.endswith("_end"))
    rows = common.sim_rows(run, 0)
    steps = rows["step"].astype(int)
    keep_all = common.steady_mask(rows, meta["warmup"])
    particles = common.particles_of(case) if case in common.CASE_LABELS else None
    next_a = np.concatenate([rows["a_start"][1:], [np.nan]])
    period = next_a - rows["a_start"]
    busy = (rows["a_end"] - rows["a_start"]) + (rows["b_end"] - rows["b_start"]) + (rows["c_end"] - rows["c_start"])
    gap = period - busy
    defrag_step = int(np.argmax(np.where(keep_all, gap, -np.inf))) if keep_all.any() else None
    out = {"trace": str(trace_dir), "case": case, "kernels": [], "phases": []}
    masks = []
    for lo, hi in halves:
        mask = keep_all & (steps >= lo) & (steps < hi)
        if defrag_step is not None:
            mask[defrag_step] = False
        masks.append(mask)
    previous = None
    for label in labels:
        if label not in rows:
            previous = label if label in kernel_table.PHASE_STARTS else previous
            continue
        if label in kernel_table.PHASE_STARTS:
            previous = label
            continue
        if previous is None or previous not in rows:
            previous = label
            continue
        duration = (rows[label] - rows[previous]) / common.NS_PER_US
        if label not in kernel_table.PHASE_ENDS:
            values = []
            for mask in masks:
                valid = mask & np.isfinite(duration) & (duration >= 0)
                values.append(float(np.median(duration[valid])) if valid.any() else None)
            if any(v is not None for v in values):
                out["kernels"].append({"kernel": kernel_table.kernel_name(label), "phase": kernel_table.phase_of(label), "us": values,
                                       "ns": [(v * 1e3 / particles if v is not None and particles else None) for v in values],
                                       "ratio": (values[1] / values[0]) if values[0] and values[1] else None})
        previous = label
    for name, value in (("A", rows["a_end"] - rows["a_start"]), ("B", rows["b_end"] - rows["b_start"]), ("C", rows["c_end"] - rows["c_start"]), ("busy", busy), ("period", period)):
        values = [float(np.median(value[mask & np.isfinite(value)]) / common.NS_PER_US) if (mask & np.isfinite(value)).any() else None for mask in masks]
        out["phases"].append({"phase": name, "us": values, "ratio": (values[1] / values[0]) if values[0] and values[1] else None})
    out["defrag_step_removed"] = defrag_step
    return out


def xtx_runs(directory) -> dict:
    """The local 7900 XTX runs: raw/<label>/loop_trace.csv per run directory (cad500, cad1000, nodefrag)."""
    out = {}
    if not directory:
        return out
    directory = pathlib.Path(directory)
    for label in ("cad500", "cad1000", "nodefrag"):
        raw = directory / label
        trace = diag.read_loop_trace(raw) if (raw / "loop_trace.csv").exists() else {}
        if not trace:
            out[label] = {"missing": True}
            continue
        total = max(trace) + 1
        values = diag.periods_ms(trace, 1, total)
        median = float(np.nanmedian(values))
        defrag_frames = sorted(frame for frame, entry in trace.items() if entry["after_defrag"])
        bins = []
        for start in range(0, total, 100):
            chunk = values[max(start - 1, 0): start + 99]
            chunk = chunk[np.isfinite(chunk)]
            kept = chunk[chunk <= 2.0 * median]
            bins.append({"bin": f"{start}-{start + 99}", "mean": float(np.mean(kept)) if len(kept) else None, "removed": int(len(chunk) - len(kept))})
        base = next((b["mean"] for b in bins if b["bin"].startswith("100-")), None)
        for b in bins:
            b["normalised"] = b["mean"] / base if base and b["mean"] else None
        log = (raw / "run.log").read_text(encoding="utf-8", errors="replace") if (raw / "run.log").exists() else ""
        fps = re.search(r"STEADY.*?=\s*([\d.]+) fps", log)
        device = re.search(r"(Radeon[^\n]*|7900[^\n]*)", log)
        out[label] = {"frames": total, "median_ms": median, "defrag_frames_after": defrag_frames, "bins": bins, "steady_fps": float(fps.group(1)) if fps else None,
                      "device": device.group(1)[:60] if device else None,
                      "mean_500_999": float(np.nanmean([v for v in values[499:999] if v <= 2 * median])), "mean_1000_1499": float(np.nanmean([v for v in values[999:1499] if v <= 2 * median]))}
    return out


# ---------------------------------------------------------------- markdown
def markdown(job: dict, chains: dict, kernels: dict, links: dict, factors: list, clock_rows: dict, e7_refs: dict, e7_bins: list, halves: dict, xtx: dict) -> str:
    L = [PART2_MARKER, "", "# 第二部分：补充（你 10-22 上午的消息；只做后处理）", "",
         "数据与口径同第一部分（同一快照）；本部分的脚本 `docs/cluster_v6/scripts/e8_b_diag2.py`，数字在 `docs/cluster_v6/data/e8/b_diag/b_diag2_1721839.json`。", ""]
    # 7
    L += ["## 7. 4M K8 与 8M K8 的各卡相位、逐核，以及每条链路的 t_tr / T_B / r_own 与等待拆分", "",
          "各卡表的口径同第 4 节（A / B / C / busy / 周期 = 稳态步去 defrag 步的均值，等待 = 每步中位数，ms）。", ""]
    for point in ("3d_n152_K8", "3d_n192_K8"):
        data = chains.get(point)
        if not data or not data.get("sims"):
            L += [f"### {point}: 缺 trace", ""]
            continue
        L += [f"### {diag.SHORT.get(data['case'], point)} K8（{data['name']}；fps 去 defrag {f(data.get('fps_nodefrag'), 2)}）", "",
              "| sim (卡) | own 列 / 流体列 | seam 数 | own 粒子 | A | B | C | busy | ns/own 粒子 | A→B 等待 | B→C 等待 | C→A 等待 | 周期 − busy（每步中位） | 周期 | busy h1 / h2 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in data["sims"]:
            L.append(f"| {s['sim']} ({s['device']}) | {s['own_columns']} / {s['fluid_columns']} | {s['seams']} | {f(s['own_particles'], 0)} | {f(s['A_ms'])} | {f(s['B_ms'])} | {f(s['C_ms'])} | **{f(s['busy_ms'])}** | "
                     f"{f(s['ns_per_own'])} | {f(s['wait_AB_ms'])} | {f(s['wait_BC_ms'])} | {f(s['wait_CA_ms'])} | {f(s['gap_ms'])} | {f(s['period_ms'])} | {f(s['busy_h1_ms'], 2)} / {f(s['busy_h2_ms'], 2)} |")
        slow = data["slowest"]
        L += ["", f"最慢的卡：sim {slow['sim']}（卡 {slow['device']}），busy {slow['busy_ms']:.3f} ms，own 列 {slow['own_columns']}（流体 {slow['fluid_columns']}），seam {slow['seams']} 个；"
              f"周期（各卡平均）{data['period_mean_of_cards_ms']:.3f} ms，周期 − 最慢卡 busy = **{data['period_minus_slowest_busy_ms']:.3f} ms**（{data['period_minus_slowest_busy_ms'] / data['period_mean_of_cards_ms'] * 100:.1f} %）；"
              f"busy 平均 {data['busy_mean_of_cards_ms']:.3f} ms，平衡度 **{data['balance']:.3f}**；每 own 粒子 busy（各卡平均）**{f(data['ns_per_own_mean'])} ns**。", ""]
        k = kernels.get(point) or {}
        if k.get("rows"):
            L += [f"逐核（ns / own 粒子；8 卡 min / 平均 / max；K = 1 同规模 trace 的 ns / N_total；G1 冗余份额 {pct(k['g1_share_mean'])}，范围 "
                  f"{pct(k['g1_share_range'][0]) if k.get('g1_share_range') else '-'}–{pct(k['g1_share_range'][1]) if k.get('g1_share_range') else '-'}）：", "",
                  "| 核 | 段 | K = 1 ns/粒子 | K8 min | K8 平均 | K8 max | K8 平均 / K1 | K8 各卡 µs/步 |", "|---|---|---|---|---|---|---|---|"]
            for r in k["rows"]:
                ratio = f(r["k8_ns_mean"] / r["k1_ns"]) if r.get("k1_ns") and r["k1_ns"] >= 0.1 and r.get("k8_ns_mean") else "-"
                L.append(f"| {'**' + r['kernel'] + '**' if r.get('combo') else r['kernel']} | {r['phase']} | {f(r['k1_ns'])} | {f(r['k8_ns_min'])} | {f(r['k8_ns_mean'])} | {f(r['k8_ns_max'])} | {ratio} | {r['k8_us_per_card'] or ''} |")
            L.append("")
        elif k.get("error"):
            L += [f"逐核：失败 {k['error']}", ""]
    L += ["### 7.1 每条链路的三跳（E7 口径；trace 的稳态步中位数，µs）", "",
          "t_tr = t_chain：发送卡 send_end → 接收卡 upload_end 的三跳总时间（含传输队列等发送方的空闲）；t_own = t_chain − 队列空闲（只算自己的读回 / 主机 / 上传，E7 的 r_own 分子）；"
          "T_B = 接收卡的 B 段；r_own = t_own / T_B，r_chain = t_chain / T_B；t_own − T_B > 0 = 自己的传输暴露在 B 段之外。", "",
          "| 点 | 链路 | t_chain | t_own | 读回 | 主机 | memcpy | 上传 | T_B | r_chain | r_own | t_chain − T_B | t_own − T_B | 主机拷贝字节 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for point in K8_ALL:
        data = links.get(point) or {}
        if data.get("error"):
            L.append(f"| {point} | 失败 {data['error']} | | | | | | | | | | | | |")
            continue
        for name, l in (data.get("links") or {}).items():
            L.append(f"| {diag.SHORT.get(common.case_of_point(point)[0], point)} K8 | {name} | {f(l['t_chain'], 0)} | {f(l['t_own'], 0)} | {f(l['own_readback'], 0)} | {f(l['own_host'], 0)} | {f(l['memcpy'], 0)} | {f(l['own_upload'], 0)} | "
                     f"{f(l['T_B'], 0)} | {f(l['r_chain'])} | {f(l['r_own'])} | {f(l['t_chain_minus_T_B_us'], 0)} | {f(l['t_own_minus_T_B_us'], 0)} | {f(l['host_copy_bytes'], 0)} |")
    L += ["", "### 7.2 各卡等待的拆分（ms）", "",
          "总等待 = 周期 − busy（稳态步去 defrag 步的均值；括号内每步中位数）= B→C 等待 + C→A 等待（+ A→B，≈ 0）。传输暴露 = 该卡接收链路里 max(0, t_own − T_B) 的最大值（自己的三跳超出 B 段的部分）；"
          "相位耦合 = 总等待 − 传输暴露（等慢的邻居：包括 t_chain − t_own 的队列空闲和帧边界上等最慢卡）。t_chain − T_B 的版本一并列出。", "",
          "| 点 | sim (卡) | 接收自 | 总等待（中位） | B→C | C→A | 传输暴露 (t_own − T_B)⁺ | 相位耦合 | (t_chain − T_B)⁺ |", "|---|---|---|---|---|---|---|---|---|"]
    for point in K8_ALL:
        data = links.get(point) or {}
        for c in data.get("cards") or []:
            L.append(f"| {diag.SHORT.get(common.case_of_point(point)[0], point)} K8 | {c['sim']} ({c['device']}) | {c['incoming']} | {f(c['wait_total_ms'])} ({f(c['wait_total_median_ms'])}) | {f(c['wait_BC_ms'])} | {f(c['wait_CA_ms'])} | "
                     f"**{f(c['exposure_own_ms'])}** | **{f(c['coupling_ms'])}** | {f(c['exposure_chain_ms'])} |")
    L.append("")
    # 8
    L += ["## 8. 三因子分解：η = 平衡度 × 尺寸效应 × seam 工作", "",
          "平衡度 = 各卡 busy 平均 / 周期；尺寸效应 = K = 1 同总规模的 ns/粒子 / K = 1 每卡尺寸立方体的 ns/粒子（参考从整算例换成每卡尺寸时 η 变成 η' 的那一步）；"
          "seam 工作 = K = 1 每卡尺寸立方体的 ns/粒子 / 链路每 own 粒子的 busy ns（同样多的粒子在链路里多花的时间：ghost、带核、G1 冗余、窄 slab 的内核效率）。"
          "三者之积 = 平衡度 × ns_同规模 / ns_own = η（own 粒子之和 = 总粒子数时严格相等）。16M / 8M / 4M K8 的每卡尺寸立方体（2M / 1M / 0.5M）没有，用 4M 立方体代替（标 *），其尺寸效应与 seam 工作各自不准、乘积仍准。"
          "弱 K8 的尺寸效应为 1（参考就是每卡的 8M 立方体），seam 工作按时间口径 τ(8M K1) / busy 平均（弱 slab 的 own 粒子 7.7M ≠ 8M 立方体的 8.0M，ns 口径见注）。", "",
          "| 点 | η（nd） | 平衡度 | 每卡尺寸参考 | ns 同规模 | ns 每卡尺寸 | ns/own（链路） | 尺寸效应 | seam 工作（ns 口径） | seam 工作（时间口径） | 三者之积 | busy 平均 / 周期 (ms) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in factors:
        L.append(f"| {diag.SHORT.get(r['case'], r['point'])}{' K8' if 'k8' not in r['case'] else ''} | {f(r['eta'])} | {f(r['balance'])} | {diag.SHORT.get(r['reference_cube'], r['reference_cube'])}{' *' if r['standin'] else ''} | "
                 f"{f(r['ns_same'])} | {f(r['ns_per_card'])} | {f(r['ns_own'])} | {f(r['size'])} | {f(r['seam'])} | {f(r['seam_time'])} | **{f(r['product'])}** | {f(r['busy_mean_ms'], 2)} / {f(r['period_ms'], 2)} |")
    L += ["", "注：弱 K8 的 ns 口径 seam 工作 = 6.637 / 6.886 = 0.964 是因为弱 slab 每卡 own 7.72M（少两个侧壁面）而 8M 立方体 8.0M；时间口径 τ_8M / busy = 0.999 才是与 η 一致的因子。", ""]
    # 9
    L += ["## 9. 时钟与功耗：K = 1 参考对链路各卡；E7 2-D K = 1 参考的每粒子成本", "",
          "| 点 | 参考 K1：SM 时钟中位（8 卡平均） | 参考 SM 最小 | 参考功耗 (W) | 参考最高温 (°C) | 参考 ns/粒子 | 链路 K 运行：SM 时钟中位（8 卡平均） | 链路各卡 SM 中位 | 链路各卡功耗 | 链路功耗平均 | 链路最高温 | trace 运行各卡 SM |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for point, c in clock_rows.items():
        L.append(f"| {diag.SHORT.get(c['case'], point)}{' K8' if 'k8' not in c['case'] else ''} | {f(c['ref']['sm_clock_median'], 0)} | {f(c['ref']['sm_clock_min'], 0)} | {f(c['ref']['power_median'], 1)} | {f(c['ref']['temperature_max'], 0)} | {f(c['ref_ns'])} | "
                 f"{f(c['chain']['sm_clock_median'], 0)} | {[round(v) for v in c['chain_per_card'].values()]} | {[round(v) for v in c['chain_power_per_card'].values()]} | {f(c['chain']['power_median'], 1)} | {f(c['chain']['temperature_max'], 0)} | "
                 f"{[round(v) for v in c['trace_per_card'].values() if v]} |")
    L += ["", "（遥测 ≥ 1 Hz，nvidia-smi：SM 时钟 / 功耗中位数、温度最大值取各运行的计时窗口；参考 = 该点所有试验的 8 个 K = 1 进程，链路 = 该点所有试验的 K 运行的 8 张卡。）", "",
          "### 9.1 E7 2-D K = 1 参考（集群同款 5090 节点，575 W，1500 / 500 窗口，fps 去 defrag）", "",
          "| 算例 | 粒子数 | 参考运行数 | 作业 | 节点 | ns/粒子·步 均值 | min / max | std | SM 时钟中位均值 | 功耗均值 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for case in ("cavity2d_n1000", "cavity2d_n1440", "cavity2d_n2000", "cavity2d_n2840", "cavity2d_n4000", "cavity2d_n5640", "cavity2d_n8000"):
        r = e7_refs.get(case)
        if not r:
            continue
        L.append(f"| {E7_SHORT[case]} ({case}) | {r['particles']:,} | {r['runs']} | {', '.join(r['jobs'])} | {', '.join(r['nodes'])} | **{r['ns_mean']:.4f}** | {r['ns_min']:.4f} / {r['ns_max']:.4f} | {r['ns_std']:.4f} | "
                 f"{f(r['clock'], 0)} | {f(r['power'], 1)} |")
    three_d = [(diag.SHORT.get(c["case"], p), c["ref_ns"]) for p, c in clock_rows.items() if c.get("ref_ns") and "k8" not in c["case"]]
    L += ["", "对照 3-D（本作业的 K = 1 参考，8 卡平均，nd）：" + "，".join(f"{name} {ns:.3f}" for name, ns in three_d) + " ns：2-D 从 2.14M 到 64M 在 ±3 % 内无趋势（8.19M 最高），3-D 从 4M 到 64M 降 8 %。", ""]
    # 10
    L += ["## 10. defrag 之后变慢", "", "### 10.1 8M K1 与 64M K1 trace：每核在第 500–999 步与第 1000–1499 步的中位数（µs；去掉第 1000 帧的 defrag 步）", ""]
    for case in ("cavity3d_n192", "cavity3d_n392"):
        h = halves.get(case)
        if not h:
            L += [f"{diag.SHORT.get(case, case)}：缺 K = 1 trace", ""]
            continue
        if h.get("error"):
            L += [f"{diag.SHORT.get(case, case)}：失败 {h['error']}", ""]
            continue
        L += [f"**{diag.SHORT[case]} K1**（{pathlib.Path(h['trace']).name}）", "", "| 核 / 段 | 段 | 500–999 µs | 1000–1499 µs | 后 / 前 | 500–999 ns/粒子 | 1000–1499 ns/粒子 |", "|---|---|---|---|---|---|---|"]
        for k in h["kernels"]:
            L.append(f"| {k['kernel']} | {k['phase']} | {f(k['us'][0], 1)} | {f(k['us'][1], 1)} | **{f(k['ratio'])}** | {f(k['ns'][0])} | {f(k['ns'][1])} |")
        for p in h["phases"]:
            L.append(f"| **{p['phase']}** | Σ | {f(p['us'][0], 1)} | {f(p['us'][1], 1)} | **{f(p['ratio'])}** | | |")
        L.append("")
    L += ["### 10.2 启动 defrag 与第 1000 帧 defrag 的代码路径", "",
          "- 两者是同一个调用：`ChainOrchestratorV7.bootstrap_all` 在 bootstrap_compute、prepare_step_cmd_buffers 之后对每个 sim 调 `sim.submit_defrag_and_wait()`（注释 2026-09-16：生成器的初始顺序不是按 voxel 排的，不做这一步前 1000 帧的内部核慢 5–9 倍）；"
          "`run_pipelined` 在 `n % defrag_cadence == 0` 时排空流水线、收集迁移报告后调同一个 `sim.submit_defrag_and_wait()`；bench 结束时再调一次。`submit_defrag_and_wait` 复用同一条录好的 `defrag_cmd`（`_record_defrag_cmd`：清 scratch 计数器与 scratch SoA → defrag 派发 → scratch → primary 拷回 → 计数器写回 alive）。",
          "- 排序键相同：`defrag.comp` 按 voxel 派发（一个线程一个 voxel），把该 voxel 的 inside_particle_index 里的粒子依次拷到 scratch 的 [base, base + count)；"
          "base 由 `USE_PREFIX_SUM_DEFRAG` 决定——默认 false（common.glsl constant_id 46；numerics.use_prefix_sum_defrag 默认 False，E8 作业没有改）→ base = atomicAdd(defrag_scratch_counter, count)，"
          "即 **voxel 块在内存里的先后顺序是原子到达顺序（非确定，近似派发顺序），不是 voxel id 顺序**；voxel 内的顺序 = inside_particle_index 的顺序。",
          "- 两次 defrag 的差别只在输入状态：启动时 inside_particle_index 由 initialize_voxelization 按格点顺序建立（voxel 内 = 格点顺序）；到第 1000 帧，update_voxel 每步压缩旧表（保留留下的粒子的相对顺序）并把本步迁入的粒子按到达顺序追加到表尾，"
          "所以 voxel 内的顺序已被 1000 步的迁入迁出打乱，而 voxel 块的放置两次都是原子到达顺序。换言之：同一条代码、同一个键，但第 1000 帧 defrag 把\"打乱后的 voxel 内顺序\"固化成了内存顺序；启动 defrag 固化的是格点顺序。",
          "- 2-D 的集群 K = 1 trace（下表）没有这一跳，本机 2-D 1M 的三次运行见 10.3。", ""]
    L += ["### 10.3 本机 7900 XTX，2-D 1M K = 1（cavity2d_n1000，1500 步，暖机 500，每 100 步均值归一到第 100–199 步）", ""]
    present = [k for k in ("cad500", "cad1000", "nodefrag") if xtx.get(k) and not xtx[k].get("missing")]
    if present:
        L.append("| 步段 | " + " | ".join(f"defrag {('每 500' if k == 'cad500' else '每 1000' if k == 'cad1000' else '关闭')}：ms | 归一" for k in present) + " |")
        L.append("|---|" + "---|---|" * len(present))
        n_bins = max(len(xtx[k]["bins"]) for k in present)
        for index in range(n_bins):
            cells = []
            for k in present:
                b = xtx[k]["bins"][index] if index < len(xtx[k]["bins"]) else {}
                cells.append(f"{f(b.get('mean'), 3)}{' (去 ' + str(b['removed']) + ' 步)' if b.get('removed') else ''} | {f(b.get('normalised'), 4)}")
            L.append(f"| {xtx[present[0]]['bins'][index]['bin']} | " + " | ".join(cells) + " |")
        L.append("")
        L.append("; ".join(f"{k}: defrag 发生在帧 {[x - 1 for x in xtx[k]['defrag_frames_after']]} 之后（after_defrag 标记），整体中位 {xtx[k]['median_ms']:.3f} ms，"
                           f"500–999 均值 {xtx[k]['mean_500_999']:.3f} / 1000–1499 均值 {xtx[k]['mean_1000_1499']:.3f}（比 {xtx[k]['mean_1000_1499'] / xtx[k]['mean_500_999']:.4f}）"
                           + (f"，STEADY {xtx[k]['steady_fps']:.1f} fps" if xtx[k].get("steady_fps") else "") for k in present))
        L.append("")
    else:
        L += ["缺：本机运行的 loop trace 不在（三次运行在后台跑，完成后重跑本脚本）。", ""]
    L += ["### 10.4 E7 2-D K = 1 trace（集群）同样分段", ""]
    if e7_bins:
        L.append("| 步段 | " + " | ".join(f"{pathlib.Path(b['trace']).name}（{E7_SHORT.get('cavity' + re.search(r'(2d_n\d+)', b['trace']).group(1), '')}）：周期 ms | 归一" for b in e7_bins) + " |")
        L.append("|---|" + "---|---|" * len(e7_bins))
        n_bins = max(len(b["bins"]) for b in e7_bins)
        for index in range(n_bins):
            cells = []
            for b in e7_bins:
                x = b["bins"][index] if index < len(b["bins"]) else {}
                cells.append(f"{f(x.get('period'), 3)}{' (去 ' + str(x['removed']) + ' 步)' if x.get('removed') else ''} | {f(x.get('normalised'), 4)}")
            L.append(f"| {e7_bins[0]['bins'][index]['bin']} | " + " | ".join(cells) + " |")
        L += ["", "（trace 的 a_start 之差 = 周期；defrag 间隔 1000；去掉的步 = 周期 > 2 × 中位数的 defrag 步。）", ""]
    else:
        L += ["缺：E7 的 2-D K = 1 trace 不在本机。", ""]
    # 11
    L += ["## 11. 缺的数据", "",
          "- 16M / 8M / 4M K8 的每卡尺寸参考（2M / 1M / 0.5M 立方体）没有算例，第 8 节用 4M 代替；128M K8 的各节等 e8_C 跑完后补。",
          "- 阶段 1 的 8M K = 1 没有 loop trace，主机提交 / 等待只有 B 线的。",
          "- 逐粒子邻居数统计求解器没有（第一部分 2.5 节）。",
          "- 本机 2-D 1M 的三次运行只有墙钟 loop trace（没有逐核 trace）；集群 2-D 的 defrag 间隔只有 1000 这一档。", ""]
    return "\n".join(L) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--phase1", default=None)
    parser.add_argument("--xtx", default=None, help="directory with cad500 / cad1000 / nodefrag run directories (loop_trace.csv, run.log)")
    parser.add_argument("--e7-results", action="append", default=None, help="glob(s) of E7 results.jsonl files (default: batch 2 + batch 1 of the e7 worktree)")
    parser.add_argument("--e7-trace", action="append", default=None, help="E7 2-D K = 1 trace directories (default: three in logs/n56)")
    parser.add_argument("--md", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--append-to", default=None, help="the part-1 document: part 2 replaces everything after its marker")
    arguments = parser.parse_args()
    job = common.load_job(arguments.job)
    points = diag.point_runs(job)
    s1 = diag.section1(job, points)
    s2r = diag.section2_references(job, points)
    k1_kernels = diag.section2_kernels(job, cases=("cavity3d_n152", "cavity3d_n192", "cavity3d_n244", "cavity3d_n312", "cavity3d_n392"))
    chains = {point: chain_phases(job, point) for point in K8_ALL}
    weak = chain_phases(job, "3d_n192_k8_K8")
    kernels = {point: chain_kernels(job, point, k1_kernels) for point in ("3d_n152_K8", "3d_n192_K8")}
    links = {point: chain_links(job, point, chains[point]) for point in K8_ALL if chains.get(point)}
    weak_eta = []
    for trial, runs in (points.get("3d_n192_k8_K8") or {}).items():
        chain_fps = common.fps_pair(job["rows"].get(runs["C"]))[1] if runs["C"] else None
        refs = [common.fps_pair(job["rows"].get(label))[1] for label in runs["refs"].values()]
        if chain_fps and len(refs) == 8 and all(refs):
            weak_eta.append(chain_fps / statistics.fmean(refs))
    factors = three_factors(s1, s2r, chains, weak_eta, weak)
    clock_rows = clocks(job, points, s2r)
    e7_refs = e7_references(arguments.e7_results or E7_RESULTS_DEFAULT)
    e7_bins = []
    for trace in (arguments.e7_trace or E7_TRACES_DEFAULT):
        if pathlib.Path(trace, "run_meta.json").exists():
            try:
                e7_bins.append(trace_bins(trace))
            except Exception as error:                                      # noqa: BLE001
                print(f"[e8_b_diag2] E7 trace {trace}: {error!r}")
    halves = {}
    for case in ("cavity3d_n192", "cavity3d_n392"):
        name = f"trace_{common.short_case(case)}_K1"
        if name in job["traces"]:
            try:
                halves[case] = kernel_halves(job["traces"][name], case)
            except Exception as error:                                      # noqa: BLE001
                halves[case] = {"error": repr(error), "trace": name}
    xtx = xtx_runs(arguments.xtx)
    text = markdown(job, chains, kernels, links, factors, clock_rows, e7_refs, e7_bins, halves, xtx)
    common.write_text(arguments.md, text)
    if arguments.json:
        common.write_json(arguments.json, {"job": job["name"], "chains": chains, "kernels": kernels, "links": links, "factors": factors, "clocks": clock_rows,
                                           "e7_references": e7_refs, "e7_bins": e7_bins, "kernel_halves": halves, "xtx": xtx, "weak_eta": weak_eta})
    if arguments.append_to:
        path = pathlib.Path(arguments.append_to)
        base = path.read_text(encoding="utf-8")
        if PART2_MARKER in base:
            base = base[:base.index(PART2_MARKER)].rstrip("\n") + "\n"
        path.write_text(base + "\n" + text, encoding="utf-8", newline="\n")
        print(f"[e8_b_diag2] appended part 2 to {path}")
    print(f"[e8_b_diag2] wrote {arguments.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
