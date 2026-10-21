"""
e8_kernel_table.py — E8 phase 2, deliverable 7: per-kernel time per step (tab:kernelcost format) from the step traces
recorded with --step-trace-detail full, and the G1 redundancy share.

A trace's run_meta.json lists every compute tick label in recording order per sim ("labels": {"compute": [[...], ...]});
a kernel's duration on a step = its tick minus the previous tick of the same phase (the phase start is a_start /
b_start / c_start; a K = 1 run records its own labels, e.g. predict_end, voxel_end, correction_density_end, density_end,
force_end). Per sim: the median over the steady steps (us), the share of the step period, ns per particle = us x 1000 /
the sim's own particles (alive incl. walls); then the mean over the sims. A kernel absent on a step (its tick not
recorded) is skipped for that step.

G1 redundancy (phase C's boundary correction_density recomputes the inner ghost column G1 as self, GHOST_SELF_LAYER = 1):
  share = sum over the sim's peer sides of n0_side x c_boundary / period, with n0_side = the inner replica count the sim
  RECEIVES on that side (the packet counters of the links into the sim: raw pool_demand of the trace run, column inner,
  median over the steady steps; else the link's host copy bytes), c_boundary = the boundary correction_density kernel's
  us per particle it processes = its median us / (particles of the band = per peer side 2 own columns x 4 (n + 8)^2 + n0_side),
  period = the sim's median step period. The band's own-column count assumes the 2/2/3 bands (c = 2) and full columns.

Usage:
    python docs/cluster_v6/scripts/e8_kernel_table.py --job DIR [--job DIR ...] --out-dir OUT [--cases cavity3d_n192,cavity3d_n392]
Writes OUT/kernel_table.md, kernel_table.json, kernel_table.csv.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e8_common as common  # noqa: E402

PHASE_STARTS = ("a_start", "b_start", "c_start")
PHASE_ENDS = ("a_end", "b_end", "c_end")


def kernel_name(label: str) -> str:
    for prefix in ("a_", "b_", "c_"):
        if label.startswith(prefix) and label not in PHASE_STARTS:
            label = label[len(prefix):]
            break
    return label[:-len("_end")] if label.endswith("_end") else label


def phase_of(label: str) -> str:
    return label[0].upper() if label[:2] in ("a_", "b_", "c_") else "-"


def kernels_of_sim(run: dict, sim: int, labels: list) -> dict:
    rows = common.sim_rows(run, sim)
    keep = common.steady_mask(rows, run["meta"]["warmup"])
    period = (np.diff(rows["a_start"]) / common.NS_PER_MS)[keep[:-1]]
    period_p50 = float(np.median(period)) if len(period) else None
    out, previous = {}, None
    for label in labels:
        if label not in rows:
            previous = label if label in PHASE_STARTS else previous
            continue
        if label in PHASE_STARTS:
            previous = label
            continue
        if previous is None or previous not in rows:
            previous = label
            continue
        duration = (rows[label] - rows[previous]) / common.NS_PER_US
        valid = keep & np.isfinite(duration) & (duration >= 0)
        if valid.any() and label not in PHASE_ENDS:
            out[label] = {"kernel": kernel_name(label), "phase": phase_of(label), "us_p50": float(np.median(duration[valid])),
                          "us_mean": float(np.mean(duration[valid])), "steps": int(valid.sum()), "after": previous}
        previous = label
    return {"period_p50_ms": period_p50, "kernels": out}


def analyze(trace_dir, case: str, raw_dir=None, partition=None) -> dict:
    run = common.load_trace(trace_dir)
    meta = run["meta"]
    labels_per_sim = (meta.get("labels") or {}).get("compute") or []
    slabs = (meta.get("slabs") or {}).get("per_sim") or []
    lattice = common.lattice_of(case)
    column_particles = 4 * (lattice["n"] + 8) ** 2
    demand = common.read_pool_demand(raw_dir) if raw_dir else {}
    sims = []
    for sim in common.sims_of(run):
        labels = labels_per_sim[sim] if sim < len(labels_per_sim) else sorted(k for k in run["device"] if k.endswith("_end"))
        result = kernels_of_sim(run, sim, labels)
        own = slabs[sim].get("own_particles") if sim < len(slabs) else None
        if own is None and partition and sim < len(partition.get("slabs") or []):
            own = partition["slabs"][sim].get("own_particles")           # the point's partition record
        if own is None and meta.get("K") == 1:
            own = lattice["total"]
        for entry in result["kernels"].values():
            entry["ns_per_particle"] = entry["us_p50"] * 1e3 / own if own else None
            entry["share_of_period"] = entry["us_p50"] / (result["period_p50_ms"] * 1e3) if result["period_p50_ms"] else None
        # G1 redundancy: the inner replica counts received by this sim (links <sender>_to_<sim>)
        n0 = {}
        for link, counts in demand.items():
            if link.startswith("_") or not link.endswith(f"_to_s{sim}"):
                continue
            inner = counts.get("inner")
            if inner is not None and len(inner):
                n0[link] = float(np.median(inner[meta["warmup"]:])) if len(inner) > meta["warmup"] else float(np.median(inner))
        # the boundary correction + density: one fused kernel (V7_FUSED_CORRECTION_DENSITY=1) or two
        by_kernel = {entry["kernel"]: entry for entry in result["kernels"].values() if entry["phase"] == "C"}
        if "correction_density_boundary" in by_kernel:
            boundary_us, boundary_kernels = by_kernel["correction_density_boundary"]["us_p50"], ["correction_density_boundary"]
        elif "correction_boundary" in by_kernel and "density_boundary" in by_kernel:
            boundary_us = by_kernel["correction_boundary"]["us_p50"] + by_kernel["density_boundary"]["us_p50"]
            boundary_kernels = ["correction_boundary", "density_boundary"]
        else:
            boundary_us, boundary_kernels = None, []
        g1 = None
        if boundary_us and n0 and result["period_p50_ms"]:
            band_particles = sum(2 * column_particles + value for value in n0.values())
            cost_per_particle = boundary_us / band_particles
            g1 = {"n0_per_side": n0, "band_particles": band_particles, "boundary_us": boundary_us, "boundary_kernels": boundary_kernels,
                  "cost_per_particle_us": cost_per_particle,
                  "redundant_us": sum(n0.values()) * cost_per_particle,
                  "share_of_step": sum(n0.values()) * cost_per_particle / (result["period_p50_ms"] * 1e3)}
        sims.append({"sim": sim, "own_particles": own, "period_p50_ms": result["period_p50_ms"], "kernels": result["kernels"], "g1": g1})
    # mean over sims per kernel
    names = []
    for sim in sims:
        for entry in sim["kernels"].values():
            if entry["kernel"] not in names:
                names.append(entry["kernel"])
    mean = {}
    for name in names:
        values = [entry for sim in sims for entry in sim["kernels"].values() if entry["kernel"] == name]
        mean[name] = {"phase": values[0]["phase"], "us_p50_mean": statistics.fmean(v["us_p50"] for v in values),
                      "ns_per_particle_mean": statistics.fmean(v["ns_per_particle"] for v in values if v["ns_per_particle"] is not None) if any(v["ns_per_particle"] is not None for v in values) else None,
                      "share_mean": statistics.fmean(v["share_of_period"] for v in values if v["share_of_period"] is not None) if any(v["share_of_period"] is not None for v in values) else None,
                      "sims": len(values)}
    g1_shares = [sim["g1"]["share_of_step"] for sim in sims if sim.get("g1")]
    return {"trace": str(trace_dir), "case": case, "K": meta.get("K"), "detail": (meta.get("clock") or {}).get("detail"), "sims": sims,
            "mean_over_sims": mean, "kernel_order": names, "period_mean_of_sims_ms": statistics.fmean(s["period_p50_ms"] for s in sims if s["period_p50_ms"]) if sims else None,
            "g1_share_mean": statistics.fmean(g1_shares) if g1_shares else None, "g1_share_range": [min(g1_shares), max(g1_shares)] if g1_shares else None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cases", default="cavity3d_n192,cavity3d_n392", help="the cubes whose K = 8 and K = 1 traces are tabulated")
    parser.add_argument("--all", action="store_true", help="every traced point and K = 1 trace")
    arguments = parser.parse_args()
    out = pathlib.Path(arguments.out_dir)
    wanted = set(arguments.cases.split(","))
    targets = []
    for directory in arguments.job:
        job = common.load_job(directory)
        for name, point in common.points_of(job).items():
            if point.get("trace_dir") and (arguments.all or (point["case"] in wanted and point["slabs"] == 8)):
                targets.append((name, point["case"], point["trace_dir"], point.get("raw_dir"), job["name"], point.get("partition")))
        for trace_name, path in job["traces"].items():
            if trace_name.startswith("trace_") and trace_name.endswith("_K1"):
                case = f"cavity{trace_name[len('trace_'):-len('_K1')]}"
                if arguments.all or case in wanted:
                    targets.append((trace_name, case, path, job["raw"].get(trace_name), job["name"], None))
    results, csv_rows = [], []
    lines = ["## Per-kernel time per step (E8 phase 2; medians over the steady steps, mean over the sims)", ""]
    for name, case, trace_dir, raw_dir, job_name, partition in targets:
        try:
            result = analyze(trace_dir, case, raw_dir, partition)
        except Exception as error:                                          # noqa: BLE001
            lines.append(f"**{name}**: failed: {error!r}")
            results.append({"point": name, "case": case, "error": repr(error)})
            continue
        result.update({"point": name, "job": job_name})
        results.append(result)
        lines += [f"**{name}** — {common.CASE_LABELS.get(case, case)} K = {result['K']}, detail {result['detail']}, step period mean of sims "
                  f"{common.fmt(result['period_mean_of_sims_ms'])} ms" + (f", G1 redundancy share {result['g1_share_mean'] * 100:.2f} % "
                  f"({result['g1_share_range'][0] * 100:.2f}–{result['g1_share_range'][1] * 100:.2f} % over the sims)" if result.get("g1_share_mean") is not None else ", G1 share: n/a (no pool demand)"),
                  "", "| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |", "|---|---|---|---|---|---|"]
        for kernel in result["kernel_order"]:
            m = result["mean_over_sims"][kernel]
            lines.append(f"| {m['phase']} | {kernel} | {m['us_p50_mean']:,.1f} | {common.fmt(m['ns_per_particle_mean'], 3)} | "
                         f"{(m['share_mean'] * 100) if m['share_mean'] is not None else float('nan'):.1f} % | {m['sims']} |")
            csv_rows.append([name, job_name, case, result["K"], m["phase"], kernel, m["us_p50_mean"], m["ns_per_particle_mean"], m["share_mean"], m["sims"]])
        lines += ["", "per sim (us / step): " + "; ".join(f"sim {s['sim']}: " + ", ".join(f"{e['kernel']} {e['us_p50']:.0f}" for e in s["kernels"].values()) for s in result["sims"]), ""]
    text = "\n".join(lines)
    print(text[:4000])
    common.write_text(out / "kernel_table.md", text)
    common.write_json(out / "kernel_table.json", results)
    with (out / "kernel_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["point", "job", "case", "K", "phase", "kernel", "us_per_step", "ns_per_particle", "share_of_step", "sims"])
        writer.writerows(csv_rows)
    print(f"[e8_kernel_table] {len(results)} traces -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
