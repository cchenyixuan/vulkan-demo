"""E9 -> the manuscript: the 3-D figures' replacement and the LaTeX rows of the 3-D tables (user 2026-10-29).

1. Figures: strong_scaling_3d.{pdf,png,csv} (fig:scaling3d, drawn by e8_scaling_figures_3d.py from E9's tables.json, the 2-D
   grey lines still E7's) and loss_sources.{pdf,png,csv} (fig:loss, drawn by e8_loss_3d.py, the 2-D points still E7's) are
   copied into the manuscript's fig/; the E8 (atomicAdd defrag order) versions already there are kept as *_e8_atomic.*.
2. LaTeX rows, all from E9: tab:strong3d (the 64M cube at K = 2 / 4 / 8 and the K = 8 cubes 4M .. 128M), tab:weak3d (8M per
   device at K = 2 / 4 / 8), tab:phases (the 64M and 8M 3-D row blocks: ideal chain = the whole case's K = 1 trace / 8, then
   the chain K = 8; step median and step mean columns), tab:ledger3d (eta, seam, transport, imbalance, r; the ideal chain =
   the whole case / K, so the reference column is gone).
3. The ledger recomputed with that ideal chain and the check |eta_timed + seam + transport + imbalance - 1| <= 0.02.
4. 64M K = 8: the per-particle cost of the interior and the boundary kernels (E9, with E8 beside it).

Everything goes into one markdown file (--out).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import statistics

COLUMN_LAYERS = 4
LABEL = {"3d_n152_K8": ("4M", "160^3", 4_096_000), "3d_n192_K8": ("8M", "200^3", 8_000_000), "3d_n244_K8": ("16M", "252^3", 16_003_008),
         "3d_n312_K8": ("33M", "320^3", 32_768_000), "3d_n392_K8": ("64M", "400^3", 64_000_000), "3d_n416_K8": ("76M", "424^3", 76_225_024),
         "3d_n496_K8": ("128M", "504^3", 128_024_064), "3d_n392_K2": ("64M", "400^3", 64_000_000), "3d_n392_K4": ("64M", "400^3", 64_000_000),
         "3d_n192_k2_K2": ("weak 2 x 8M", "", 15_680_000), "3d_n192_k4_K4": ("weak 4 x 8M", "", 31_040_000), "3d_n192_k8_K8": ("weak 8 x 8M", "", 61_760_000)}


def load(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def mean_minmax(stat, digits=3):
    return f"{stat['mean']:.{digits}f} ({stat['min']:.{digits}f}--{stat['max']:.{digits}f})"


def reference_fps(point):
    """Mean over the trials of the mean over the reference members (K = 1 cards, or K = 2 pairs for 128M): with / without defrag."""
    with_, without = [], []
    for trial in point["trials"]:
        for group in trial["references"]:
            members = [m for m in group["members"] if m["status"] == "pass"]
            with_.append(statistics.fmean(m["fps"] for m in members))
            without.append(statistics.fmean(m["fps_nodefrag"] for m in members))
    return statistics.fmean(with_), statistics.fmean(without), point["reference_kind"]


def scaling_rows(points):
    """tab:strong3d / tab:weak3d: N | per device | K | fps (with defrag) | fps (no defrag) | reference fps with | without | eta (no defrag) | eta_min | eta (with)."""
    rows = []
    for p in points:
        s = p["stats"]
        ref_with, ref_without, kind = reference_fps(p)
        size, cube, n = LABEL[p["point"]]
        per_card = p["particles"] / p["K"]
        ref_label = "K = 2 pair" if kind == "pairs" else ("K = 1, same cube per device" if p["kind"] == "weak" else "K = 1")
        rows.append({"point": p["point"], "size": size, "cube": cube, "particles": p["particles"], "per_card": per_card, "K": p["K"], "kind": p["kind"],
                     "fps_with": s["fps"]["mean"], "fps_nodefrag": s["fps_nodefrag"]["mean"], "ref_with": ref_with, "ref_nodefrag": ref_without,
                     "ref_label": ref_label, "eta": s["eta"], "eta_min": s["eta_min"]["mean"], "eta_with": s["eta_with"]["mean"], "trials": p["trial_count"],
                     "cuts": p["cuts"], "job": p["job"]})
    return rows


def latex_scaling(rows, weak=False):
    lines = []
    for r in rows:
        label = r["size"] if weak else f"{r['size']} ({r['cube'].replace('^3', '$^3$')})"
        lines.append(f"{label} & {r['per_card'] / 1e6:.2f} & {r['K']} & {r['fps_with']:.2f} & {r['fps_nodefrag']:.2f} & {r['ref_with']:.2f} & {r['ref_nodefrag']:.2f} & "
                     f"{mean_minmax(r['eta'])} & {r['eta_min']:.3f} & {r['eta_with']:.3f} \\\\")
    return "\n".join(lines)


def phases_rows(entry, label):
    """tab:phases 3-D block: ideal chain = the whole case's K = 1 trace / 8 (one card, so no range), then the chain K = 8."""
    w, c = entry["whole_cube_divided"]["aggregate"], entry["chain"]["aggregate"]

    def cell(stat, single):
        return f"{stat['mean']:.3f}" if single else f"{stat['mean']:.3f} ({stat['min']:.3f}--{stat['max']:.3f})"

    ideal = (f"{label} & ideal chain: K = 1 of the whole case / 8 & {cell(w['a'], True)} & {cell(w['b'], True)} & {cell(w['c'], True)} & "
             f"{w['period']['mean']:.3f} & {w['step_mean_window']:.3f} \\\\")
    chain = (f"{label} & chain K = 8 & {cell(c['a'], False)} & {cell(c['b'], False)} & {cell(c['c'], False)} & "
             f"{cell(c['period'], False)} & {c['step_mean_window']:.3f} \\\\")
    ratio = w["step_mean_window"] / c["step_mean_window"]
    ratio_median = w["period"]["mean"] / c["period"]["mean"]
    return ideal, chain, ratio, ratio_median, entry["eta_timed"]


def ledger_rows(ledger):
    """Ideal chain = the whole case / K (tau_ref): eta' = tau_ref / P, seam = busy - tau_ref, transport = exposed, imbalance = the rest."""
    out = []
    for r in ledger:
        P, busy, tau_ref, exposed = r["period_ms"], r["busy_ms"], r["tau_ref_ms"], r["exposed_ms"]
        ideal = tau_ref / P
        seam, transport, imbalance = (busy - tau_ref) / P, exposed / P, (P - busy - exposed) / P
        out.append({"point": r["point"], "kind": r["kind"], "per_card": r["per_card"], "period_ms": P, "eta_trace": ideal, "seam": seam, "transport": transport,
                    "imbalance": imbalance, "sum_trace": ideal + seam + transport + imbalance, "eta": r["eta"], "eta_min_trials": r["eta_min"],
                    "eta_max_trials": r["eta_max"], "residual": r["eta"] + seam + transport + imbalance - 1, "r_p50": r["r_p50"], "r_p95": r["r_p95"],
                    "busy_ms": busy, "tau_ref_ms": tau_ref, "exposed_ms": exposed, "reference": r.get("reference_kind") or r.get("tau_ref_kind")})
    return out


def kernel_costs(trace, column_particles):
    """64M K = 8: per sim the four sweep kernels' us and the particles each one processes (bands 2 / 2 / 3 own columns per peer side;
    the boundary correction + density kernel also processes the G1 replicas it receives, n0 per side)."""
    rows = []
    for sim in trace["sims"]:
        ks = {e["kernel"]: e["us_p50"] for e in sim["kernels"].values()}
        g1 = sim.get("g1") or {}
        sides = len(g1.get("n0_per_side") or {})
        n0 = sum((g1.get("n0_per_side") or {}).values())
        own = sim["own_particles"]
        processed = {"correction_density_interior": own - 2 * column_particles * sides, "force_deep_interior": own - 3 * column_particles * sides,
                     "correction_density_boundary": 2 * column_particles * sides + n0, "force": 3 * column_particles * sides}
        rows.append({"sim": sim["sim"], "own": own, "sides": sides, "n0": n0, "period_ms": sim["period_p50_ms"],
                     "us": {k: ks[k] for k in processed}, "processed": processed,
                     "ns_per_processed": {k: ks[k] * 1e3 / processed[k] for k in processed},
                     "ns_per_own": {k: ks[k] * 1e3 / own for k in processed}})
    summary = {}
    for key in ("ns_per_processed", "ns_per_own"):
        summary[key] = {k: {"mean": statistics.fmean(r[key][k] for r in rows), "min": min(r[key][k] for r in rows), "max": max(r[key][k] for r in rows)}
                        for k in rows[0]["processed"]}
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", required=True)
    parser.add_argument("--phases-3d", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--kernels-e9", required=True)
    parser.add_argument("--kernels-e8", required=True)
    parser.add_argument("--scaling-fig-dir", required=True, help="where e8_scaling_figures_3d.py wrote E9's strong_scaling_3d.*")
    parser.add_argument("--loss-fig-dir", required=True, help="where e8_loss_3d.py wrote E9's loss_sources.*")
    parser.add_argument("--manuscript-fig", required=True)
    parser.add_argument("--out", required=True)
    a = parser.parse_args()

    # 1. figures into the manuscript, the E8 versions kept
    fig_dir = pathlib.Path(a.manuscript_fig).resolve()
    copied = []
    for name, source in (("strong_scaling_3d", pathlib.Path(a.scaling_fig_dir)), ("loss_sources", pathlib.Path(a.loss_fig_dir))):
        for suffix in (".pdf", ".png", ".csv"):
            target, keep = fig_dir / f"{name}{suffix}", fig_dir / f"{name}_e8_atomic{suffix}"
            if target.exists() and not keep.exists():
                shutil.copy2(target, keep)
            shutil.copy2(source / f"{name}{suffix}", target)
            copied.append(str(target))

    tables = load(a.tables)
    points = {p["point"]: p for p in tables["points"]}
    strong_k = scaling_rows([points[n] for n in ("3d_n392_K2", "3d_n392_K4", "3d_n392_K8")])
    strong_n = scaling_rows([points[n] for n in ("3d_n152_K8", "3d_n192_K8", "3d_n244_K8", "3d_n312_K8", "3d_n392_K8", "3d_n416_K8", "3d_n496_K8")])
    weak = scaling_rows([points[n] for n in ("3d_n192_k2_K2", "3d_n192_k4_K4", "3d_n192_k8_K8")])
    phases = {e["point"]: e for e in load(a.phases_3d)}
    p64 = phases_rows(phases["3d_n392_K8"], "64M (400$^3$)")
    p8 = phases_rows(phases["3d_n192_K8"], "8M (200$^3$)")
    ledger = ledger_rows(load(a.ledger))
    order = ["3d_n152_K8", "3d_n192_K8", "3d_n244_K8", "3d_n312_K8", "3d_n392_K8", "3d_n416_K8", "3d_n496_K8", "3d_n192_k8_K8"]
    ledger.sort(key=lambda r: order.index(r["point"]))
    k9 = {t["point"]: t for t in load(a.kernels_e9)}
    k8 = {t["point"]: t for t in load(a.kernels_e8)}
    column_particles = COLUMN_LAYERS * 400 ** 2                                   # the 64M cube: 4 layers x (392 + 8)^2 = 640,000 per voxel column
    rows9, sum9 = kernel_costs(k9["3d_n392_K8"], column_particles)
    rows8, sum8 = kernel_costs(k8["3d_n392_K8"], column_particles)
    cube9 = k9["trace_3d_n192_K1"]["mean_over_sims"]
    cube64 = k9["trace_3d_n392_K1"]["mean_over_sims"]

    L = []
    L += ["# E9 -> manuscript: 3-D figures and table rows (prefix defrag order; jobs e9_A/B/C/D 1741917/18/21/24, 2026-10-28)", "",
          "Source data: `docs/cluster_v6/data/e9/phase2/tables.json` (e8_p2_tables.py on the four E9 mirrors), `data/e9/loss_3d/{ledger,phases_3d}.json` "
          "(e8_loss_3d.py), per-kernel tables (e8_kernel_table.py --all). eta = fps_K / (K x mean simultaneous K = 1 reference fps) without the defrag "
          "steps (128M: K = 2 pairs, eta = fps_K / (4 x mean pair fps)); mean (min--max) over the trials (4M, 8M five trials, the rest three). "
          "Generated by `docs/cluster_v6/scripts/e9_paper_rows.py`.", ""]
    L += ["## 1. Figures replaced in the manuscript", ""]
    L += [f"- `{c}`" for c in copied]
    L += ["", "The previous (E8, atomicAdd order) files are kept beside them as `strong_scaling_3d_e8_atomic.*` and `loss_sources_e8_atomic.*`. "
          "fig:scaling3d: the 3-D series (64M at K = 2 / 4 / 8, weak 8M per device at K = 2 / 4 / 8, the seven K = 8 cubes with 76M open) are E9; "
          "the 2-D grey lines are E7's `scaling_figures.csv` unchanged. fig:loss: the 3-D squares (T_B, t_tr, host part, eta against r) are E9's "
          "eight K = 8 points (seven cubes + weak); the 2-D circles are E7's `loss_sources_figure.json` unchanged.", ""]

    L += ["## 2. tab:strong3d", "",
          "Columns: case | particles per device (M) | K | fps with defrag | fps without defrag | reference fps with | reference fps without | "
          "eta without defrag, mean (min--max) | eta_min (against the slowest reference card) | eta with defrag. Reference = the K simultaneous K = 1 runs "
          "of the whole case (128M: the four K = 2 pairs; the pair fps is listed and eta uses K / 2 = 4).", "", "```latex",
          "% 64M cube against K (line A: K = 2, 4; line B: K = 8)", latex_scaling(strong_k),
          "% K = 8 against the cube size (line B: 4M .. 64M; line D: 76M; line C: 128M, K = 2 pair reference)", latex_scaling(strong_n), "```", ""]
    L += ["Cuts (own voxel columns per device boundary): " + "; ".join(f"{r['size']} K{r['K']}: {r['cuts']}" for r in strong_k + strong_n), ""]

    L += ["## 3. tab:weak3d", "",
          "Columns as tab:strong3d; case = 8M (200$^3$) per device, K boxes of 200 x 200 x 200 K stacked along x; reference = each card's K = 1 "
          "of the single 8M cube (not divided).", "", "```latex", latex_scaling(weak, weak=True), "```", ""]

    L += ["## 4. tab:phases, the two 3-D row blocks", "",
          "Columns: case | row | A | B | C | step median | step mean (window - defrag), ms. Chain rows: each card's median over the steady steps "
          "(step >= 500, next step present, the defrag step dropped), mean over the cards (min--max); step mean = (sum of the steady periods - the "
          "in-window defrag wall time) / steps. Ideal chain = the K = 1 trace of the whole case on one card, every column / 8 (one card: no range). "
          "The 'skip the defrag step' mean is in `phases_3d.md` if wanted (64M: 47.191 / 54.069; 8M: 5.792 / 8.909).", "", "```latex",
          "% 64M (400^3) K = 8, trace 3d_n392_K8_C of e9_B (wqd10nbj04g4); ideal = trace_3d_n392_K1 / 8", p64[0], p64[1],
          "% 8M (200^3) K = 8, trace 3d_n192_K8_C of e9_B; ideal = trace_3d_n192_K1 / 8", p8[0], p8[1], "```", "",
          f"ideal / chain (step means, window - defrag): 64M {p64[2]:.4f} (medians {p64[3]:.4f}; timed eta {p64[4]:.4f}), 8M {p8[2]:.4f} "
          f"(medians {p8[3]:.4f}; timed eta {p8[4]:.4f}).", ""]

    L += ["## 5. tab:ledger3d (ideal chain = the whole case / K; four terms + r)", "",
          "Columns: case | particles per device (M) | P (ms) | eta | seam (%) | transport (%) | imbalance (%) | r. From the chain trace of each "
          "point: P = the step period (mean over the steady steps and cards); ideal = tau(N) / K with tau(N) = the whole case's K = 1 trace step time "
          "(128M: the K = 2 pair trace / 4; weak: the 8M cube's K = 1 trace, not divided); seam = (busy - ideal) / P with busy = T_A + T_B + T_C per "
          "card, mean over the cards; transport = the exposed transfer, per card and step the largest (t_tr - T_B)+ over its incoming links, mean "
          "over steps and cards; imbalance = (P - busy - exposed) / P; r = t_tr / T_B of the worst link per step, median. eta in the table is the "
          "timed one (the trials' mean); the trace's own eta' = ideal / P is listed in the check below. With this ideal chain the reference "
          "working-point term of the E8 ledger is zero by construction, so eta' + seam + transport + imbalance = 100 % exactly.", "", "```latex"]
    for r in ledger:
        size = LABEL[r["point"]][0]
        L.append(f"{size} & {r['per_card'] / 1e6:.2f} & {r['period_ms']:.3f} & {r['eta']:.3f} & {r['seam'] * 100:.1f} & {r['transport'] * 100:.1f} & "
                 f"{r['imbalance'] * 100:.1f} & {r['r_p50']:.2f} \\\\")
    L += ["```", "", "### 5.1 Check: the four terms against the timed eta", "",
          "| point | per device | P (ms) | eta' = ideal / P (trace) | seam % | transport % | imbalance % | eta' + three terms | eta timed (min--max) | "
          "eta timed + three terms - 1 | r p50 (p95) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    worst = 0.0
    for r in ledger:
        worst = max(worst, abs(r["residual"]))
        L.append(f"| {LABEL[r['point']][0]} | {r['per_card'] / 1e6:.2f}M | {r['period_ms']:.3f} | {r['eta_trace']:.4f} | {r['seam'] * 100:.2f} | "
                 f"{r['transport'] * 100:.2f} | {r['imbalance'] * 100:.2f} | {r['sum_trace'] * 100:.1f} | {r['eta']:.4f} ({r['eta_min_trials']:.3f}--{r['eta_max_trials']:.3f}) | "
                 f"**{r['residual']:+.4f}** | {r['r_p50']:.3f} ({r['r_p95']:.3f}) |")
    L += ["", f"Largest |eta timed + seam + transport + imbalance - 1| over the eight points: **{worst:.4f}** (limit 0.02): the trace run's eta' sits "
          "0.003-0.013 below the timed trials' mean (different runs; the traced run carries the per-kernel timestamps). In ms: " +
          "; ".join(f"{LABEL[r['point']][0]} P {r['period_ms']:.3f} = ideal {r['tau_ref_ms']:.3f} + seam {r['busy_ms'] - r['tau_ref_ms']:.3f} + exposed "
                    f"{r['exposed_ms']:.3f} + imbalance {r['period_ms'] - r['busy_ms'] - r['exposed_ms']:.3f}" for r in ledger) + ".", ""]

    L += ["## 6. 64M K = 8: per-particle cost of the interior and the boundary sweep kernels (E9; E8 beside it)", "",
          "Per-kernel GPU timestamps of the traced K = 8 run (`--step-trace-detail full`), p50 over the steady steps per card. Particles each kernel "
          "processes on a card with c own voxel columns (640,000 particles per column = 4 layers x 400^2, walls included) and s peer sides (1 for the "
          "end cards, 2 inside): correction+density interior = own - 2 x 640k x s; force deep interior = own - 3 x 640k x s; correction+density "
          "boundary = 2 x 640k x s own band columns + the received G1 replicas (640k per side); force boundary = 3 x 640k x s. "
          "The 'per own particle' columns divide by the card's own particles instead (the kernel table's convention).", "",
          "| campaign | kernel | ns per processed particle, mean over the 8 cards (min--max) | ns per own particle | us per step, mean of the cards |",
          "|---|---|---|---|---|"]
    names = (("correction_density_interior", "correction + density, interior (phase B)"), ("force_deep_interior", "force, deep interior (phase B)"),
             ("correction_density_boundary", "correction + density, boundary (phase C)"), ("force", "force, boundary (phase C)"))
    for tag, rows, summ in (("E9", rows9, sum9), ("E8", rows8, sum8)):
        for key, label in names:
            s, o = summ["ns_per_processed"][key], summ["ns_per_own"][key]
            us = statistics.fmean(r["us"][key] for r in rows)
            L.append(f"| {tag} | {label} | **{s['mean']:.2f}** ({s['min']:.2f}--{s['max']:.2f}) | {o['mean']:.3f} | {us:,.0f} |")
    ci, fi = sum9["ns_per_processed"]["correction_density_interior"]["mean"], sum9["ns_per_processed"]["force_deep_interior"]["mean"]
    cb, fb = sum9["ns_per_processed"]["correction_density_boundary"]["mean"], sum9["ns_per_processed"]["force"]["mean"]
    L += ["", f"E9 ratios boundary / interior per processed particle: correction + density **{cb / ci:.3f}**, force **{fb / fi:.3f}**; both kernels "
          f"together (sum of the two boundary kernels' ns per their particles / sum of the two interior ones) **{(cb + fb) / (ci + fi):.3f}**. "
          f"For reference the K = 1 traces (no bands, whole cube per particle): 8M cube correction + density {cube9['correction_density_interior']['ns_per_particle_mean']:.3f} ns, "
          f"force {cube9['force_deep_interior']['ns_per_particle_mean']:.3f} ns; 64M cube {cube64['correction_density_interior']['ns_per_particle_mean']:.3f} / "
          f"{cube64['force_deep_interior']['ns_per_particle_mean']:.3f} ns. E8 ratios: correction + density "
          f"{sum8['ns_per_processed']['correction_density_boundary']['mean'] / sum8['ns_per_processed']['correction_density_interior']['mean']:.3f}, "
          f"force {sum8['ns_per_processed']['force']['mean'] / sum8['ns_per_processed']['force_deep_interior']['mean']:.3f}.", "",
          "Per card (E9): sim | own (M) | sides | corr+dens interior us (ns/p) | force deep us (ns/p) | corr+dens boundary us (ns/p) | force boundary us (ns/p) | period ms", "",
          "| sim | own | sides | cd interior | force deep | cd boundary | force boundary | period |", "|---|---|---|---|---|---|---|---|"]
    for r in rows9:
        n = r["ns_per_processed"]
        L.append(f"| {r['sim']} | {r['own'] / 1e6:.2f} | {r['sides']} | {r['us']['correction_density_interior']:,.0f} ({n['correction_density_interior']:.2f}) | "
                 f"{r['us']['force_deep_interior']:,.0f} ({n['force_deep_interior']:.2f}) | {r['us']['correction_density_boundary']:,.0f} ({n['correction_density_boundary']:.2f}) | "
                 f"{r['us']['force']:,.0f} ({n['force']:.2f}) | {r['period_ms']:.3f} |")
    L.append("")
    pathlib.Path(a.out).write_text("\n".join(L), encoding="utf-8")
    json_out = pathlib.Path(a.out).with_suffix(".json")
    json_out.write_text(json.dumps({"strong_k": strong_k, "strong_n": strong_n, "weak": weak, "ledger": ledger, "kernels_e9": {"rows": rows9, "summary": sum9},
                                    "kernels_e8": {"rows": rows8, "summary": sum8}, "phases": {"64M": p64, "8M": p8}, "figures": copied}, indent=1),
                        encoding="utf-8")
    print(f"[e9_paper_rows] -> {a.out} (+ .json); figures: {len(copied)} files; ledger check worst residual {worst:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
