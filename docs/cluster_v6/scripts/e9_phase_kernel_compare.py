"""E9 vs E8 (prefix vs atomicAdd defrag order) per traced point: phase durations, per-kernel costs and the ledger terms.

Inputs are the outputs of the E8 tools run on both campaigns:
  * phases_table.json  (e8_phases_table.py)  - per sim A / B / C medians and period of every traced point
  * kernel_table.json  (e8_kernel_table.py --all) - per sim per-kernel us (p50 over the steady steps) and own particles
  * ledger.json        (e8_loss_3d.py)        - the five-term ledger per K = 8 point

Output: one markdown document (and a JSON) with, per point,
  1. T_A / T_B / T_C / period (per card median, mean over the cards) for E8 and E9 and the ratios E9 / E8;
  2. the four sweep kernels (correction_density_interior, force_deep_interior = phase B; correction_density_boundary, force = phase C)
     and the small kernels: us / step mean over the sims, ns per own particle, ratio E9 / E8;
  3. the ledger terms side by side (E8 -> E9).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics

PHASE_B = ("correction_density_interior", "force_deep_interior")
PHASE_C = ("correction_density_boundary", "force")
SMALL = ("predict", "voxel", "ghost_trailing", "ghost_leading", "expand", "install_trailing", "install_leading", "append_departed", "density")
LEDGER_TERMS = (("period_ms", "P (ms)"), ("busy_ms", "busy (ms)"), ("tau_pc_ms", "tau_pc (ms)"), ("tau_ref_ms", "tau(N)/K (ms)"),
                ("ideal", "ideal %"), ("seam", "seam work %"), ("reference_point", "reference working point %"),
                ("exposed", "transfer exposed %"), ("imbalance", "imbalance %"), ("eta_trace", "eta_trace"), ("eta", "eta timed"),
                ("g1_share", "G1 redundancy %"))
LABELS = {"3d_n152_K8": "4M K8", "3d_n192_K8": "8M K8", "3d_n244_K8": "16M K8", "3d_n312_K8": "33M K8", "3d_n392_K8": "64M K8",
          "3d_n416_K8": "76M K8", "3d_n496_K8": "128M K8", "3d_n192_k8_K8": "weak 8 x 8M K8", "3d_n392_K2": "64M K2", "3d_n392_K4": "64M K4",
          "3d_n192_k2_K2": "weak 2 x 8M K2", "3d_n192_k4_K4": "weak 4 x 8M K4", "trace_3d_n152_K1": "4M K1", "trace_3d_n192_K1": "8M K1",
          "trace_3d_n244_K1": "16M K1", "trace_3d_n312_K1": "33M K1", "trace_3d_n392_K1": "64M K1", "trace_3d_n416_K1": "76M K1"}
ORDER = ("trace_3d_n152_K1", "trace_3d_n192_K1", "trace_3d_n244_K1", "trace_3d_n312_K1", "trace_3d_n392_K1", "trace_3d_n416_K1",
         "3d_n152_K8", "3d_n192_K8", "3d_n244_K8", "3d_n312_K8", "3d_n392_K8", "3d_n416_K8", "3d_n496_K8", "3d_n192_k8_K8",
         "3d_n392_K2", "3d_n392_K4", "3d_n192_k2_K2", "3d_n192_k4_K4")


def load(path: str):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def phase_means(entry: dict) -> dict:
    """Per card medians (p50 over the steady steps) averaged over the cards, ms."""
    sims = entry["chain"]["per_sim"]
    out = {}
    for key, name in (("a_p50_ms", "A"), ("b_p50_ms", "B"), ("c_p50_ms", "C"), ("period_p50_ms", "period")):
        values = [s[key] for s in sims if key in s and s[key] is not None]
        out[name] = statistics.fmean(values) if values else None
    for alt in ("step_mean_skip", "step_mean_skip_ms"):
        if alt in entry["chain"]:
            out["step_mean_skip"] = entry["chain"][alt]
    return out


def k1_phase_means(entry: dict) -> dict:
    """The K = 1 trace of a point as stored in the 'ideal' half of the phases table (divided by K there; we undo the division)."""
    ideal = entry.get("ideal") or {}
    sims = ideal.get("per_sim") or []
    if not sims:
        return {}
    out = {}
    for key, name in (("a_p50_ms", "A"), ("b_p50_ms", "B"), ("c_p50_ms", "C"), ("period_p50_ms", "period")):
        values = [s[key] for s in sims if key in s and s[key] is not None]
        out[name] = statistics.fmean(values) if values else None
    return out


def kernel_means(trace: dict) -> dict:
    """Per kernel: us / step (p50 over the steady steps, mean over the sims) and ns per own particle (mean over the sims)."""
    out = {}
    for name in trace["kernel_order"]:
        us, ns = [], []
        for sim in trace["sims"]:
            for entry in sim["kernels"].values():
                if entry["kernel"] == name:
                    us.append(entry["us_p50"])
                    ns.append(entry["us_p50"] * 1e3 / sim["own_particles"])
        if us:
            out[name] = {"us": statistics.fmean(us), "ns": statistics.fmean(ns), "sims": len(us)}
    own = statistics.fmean(s["own_particles"] for s in trace["sims"])
    return {"kernels": out, "own_particles_mean": own, "period_ms": trace["period_mean_of_sims_ms"], "g1_share": trace.get("g1_share_mean")}


def ratio(a, b):
    return None if a in (None, 0) or b is None else b / a


def f(x, digits=3, pct=False):
    if x is None:
        return "-"
    return f"{x * 100:.1f}" if pct else f"{x:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e8-phases", required=True)
    parser.add_argument("--e9-phases", required=True)
    parser.add_argument("--e8-kernels", required=True)
    parser.add_argument("--e9-kernels", required=True)
    parser.add_argument("--e8-ledger", required=True)
    parser.add_argument("--e9-ledger", required=True)
    parser.add_argument("--md", required=True)
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()

    phases8 = {e["point"]: e for e in load(arguments.e8_phases)}
    phases9 = {e["point"]: e for e in load(arguments.e9_phases)}
    kernels8 = {t["point"]: t for t in load(arguments.e8_kernels)}
    kernels9 = {t["point"]: t for t in load(arguments.e9_kernels)}
    ledger8 = {r["point"]: r for r in load(arguments.e8_ledger)}
    ledger9 = {r["point"]: r for r in load(arguments.e9_ledger)}
    points = [p for p in ORDER if p in kernels9 or p in phases9]
    document = {"points": []}
    lines = ["## E8 (atomicAdd defrag order) vs E9 (prefix order): phases, kernels and ledger per traced point", "",
             "Phase durations = per card median over the steady steps, mean over the cards (e8_phases_table.py); kernels = per sim p50 over the "
             "steady steps (per-kernel timestamps, --step-trace-detail full), mean over the sims, and ns per own particle of the sim "
             "(e8_kernel_table.py --all); ratios = E9 / E8 of the same point; the partitions are identical between the campaigns except "
             "64M K2 (E8 cut [50], E9 [51]), where the ns-per-particle columns are the fair comparison.", ""]

    # 1. phases
    lines += ["### 1. Phase durations (ms) and the ratios E9 / E8", "",
              "| point | T_A E8 / E9 (ratio) | T_B E8 / E9 (ratio) | T_C E8 / E9 (ratio) | period E8 / E9 (ratio) |", "|---|---|---|---|---|"]
    for p in points:
        if p.startswith("trace_"):
            # the K = 1 traces appear as the 'ideal' half of their K = 8 point
            key = p.replace("trace_", "").replace("_K1", "_K8")
            a8, a9 = (k1_phase_means(phases8[key]) if key in phases8 else {}), (k1_phase_means(phases9[key]) if key in phases9 else {})
            if not a9:
                continue
            # undo the division by K: the ideal half is stored divided by K (the per-card ideal)
            K = phases9[key]["K"]
            a8 = {k: (v * K if v is not None else None) for k, v in a8.items()}
            a9 = {k: (v * K if v is not None else None) for k, v in a9.items()}
        else:
            if p not in phases9:
                continue
            a8 = phase_means(phases8[p]) if p in phases8 else {}
            a9 = phase_means(phases9[p])
        entry = {"point": p, "label": LABELS.get(p, p), "phases": {"e8": a8, "e9": a9,
                 "ratio": {k: ratio(a8.get(k), a9.get(k)) for k in ("A", "B", "C", "period")}}}
        document["points"].append(entry)
        cells = []
        for k in ("A", "B", "C", "period"):
            cells.append(f"{f(a8.get(k))} / {f(a9.get(k))} ({f(entry['phases']['ratio'][k], 3)})")
        lines.append(f"| {LABELS.get(p, p)} | " + " | ".join(cells) + " |")
    lines.append("")

    # 2. kernels
    lines += ["### 2. Kernels: us / step (mean over the sims) and ns per own particle; ratio E9 / E8", "",
              "| point | own / sim (M) | corr+dens interior us (ns/p) E8 -> E9 (ratio) | force_deep_interior | corr+dens boundary | force (boundary) | "
              "phase-B kernels sum (ratio) | phase-C kernels sum (ratio) | small kernels sum (ratio) | period (ratio) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for entry in document["points"]:
        p = entry["point"]
        if p not in kernels9:
            continue
        k9 = kernel_means(kernels9[p])
        k8 = kernel_means(kernels8[p]) if p in kernels8 else {"kernels": {}, "own_particles_mean": None, "period_ms": None}
        entry["kernels"] = {"e8": k8, "e9": k9, "ratio": {}}

        def cell(name):
            a, b = k8["kernels"].get(name), k9["kernels"].get(name)
            r = ratio(a["ns"], b["ns"]) if a and b else None
            entry["kernels"]["ratio"][name] = r
            if not b:
                return "-"
            left = f"{a['us']:,.0f} ({a['ns']:.3f})" if a else "-"
            return f"{left} -> {b['us']:,.0f} ({b['ns']:.3f}) ({f(r)})"

        def group(names, key):
            a = sum(k8["kernels"][n]["ns"] for n in names if n in k8["kernels"]) if k8["kernels"] else None
            b = sum(k9["kernels"][n]["ns"] for n in names if n in k9["kernels"])
            r = ratio(a, b)
            entry["kernels"]["ratio"][key] = r
            return f"{f(a)} -> {f(b)} ns/p ({f(r)})"

        own = f"{k9['own_particles_mean'] / 1e6:.2f}"
        per = ratio(k8["period_ms"], k9["period_ms"])
        entry["kernels"]["ratio"]["period"] = per
        lines.append(f"| {LABELS.get(p, p)} | {own} | {cell('correction_density_interior')} | {cell('force_deep_interior')} | "
                     f"{cell('correction_density_boundary')} | {cell('force')} | {group(PHASE_B, 'phase_b_kernels')} | {group(PHASE_C, 'phase_c_kernels')} | "
                     f"{group(SMALL, 'small_kernels')} | {f(k8['period_ms'])} -> {f(k9['period_ms'])} ms ({f(per)}) |")
    lines.append("")
    lines += ["Small kernels = predict, voxel, ghost_*, expand, install_*, append_departed, density (the copy).", ""]

    # 3. ledger
    lines += ["### 3. Ledger terms E8 -> E9 (K = 8 points; % of the period unless ms)", "",
              "| point | " + " | ".join(label for _, label in LEDGER_TERMS) + " |", "|---|" + "---|" * len(LEDGER_TERMS)]
    for p in ORDER:
        if p not in ledger9:
            continue
        r8, r9 = ledger8.get(p, {}), ledger9[p]
        cells = []
        led = {}
        for key, label in LEDGER_TERMS:
            a, b = r8.get(key), r9.get(key)
            pct = key in ("ideal", "seam", "reference_point", "exposed", "imbalance", "g1_share")
            led[key] = {"e8": a, "e9": b}
            cells.append(f"{f(a, 3, pct)} -> {f(b, 3, pct)}")
        lines.append(f"| {LABELS.get(p, p)} | " + " | ".join(cells) + " |")
        document.setdefault("ledger", {})[p] = led
    lines.append("")
    pathlib.Path(arguments.md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    if arguments.json:
        pathlib.Path(arguments.json).write_text(json.dumps(document, indent=1), encoding="utf-8")
    print(f"[e9_phase_kernel_compare] {len(document['points'])} points -> {arguments.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
