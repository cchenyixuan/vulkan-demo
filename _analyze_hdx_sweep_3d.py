"""
_analyze_hdx_sweep_3d.py — metrics, verdicts, table and figure for the 3-D
h/dx sweep (E6). Reads docs/hdx_sweep_3d_<date>/hdx*/ (result.json, run.jsonl,
breakdown.json) and the final snapshots, writes summary.md / summary.json /
profiles.csv into the sweep directory and the paper figure
manuscripts/fig/hdx_sweep_3d.{pdf,png}.

Metrics (all relative to the h/dx = 4 reference where stated):
  cost      us/step, wall seconds per unit physical time (= the full run's
            sim wall time for T), per-kernel ns/particle
  health    correction_fallback_count (cumulative), fluid rho std/rho0 and
            max|rho-rho0|/rho0, interior pressure std / (rho0 U^2), W_sum mean
            and 1 % quantile (interior), min particle spacing/dx and pairs
            closer than 0.5 dx, kinetic-energy series (max relative jump),
            max|v|/U, dead particles, overflow, drift
  accuracy  mid-plane (z = 0) centerlines u(y) at x = 0 and v(x) at y = 0,
            SPH (Shepard) interpolation at 201 points, L2 deviation vs h/dx = 4
            normalised by U; h/dx 4 vs 3.5 is the resolution-sensitivity baseline
Verdict (E6.3): smallest h/dx with fallback = 0, no pair < 0.5 dx, rho std <=
1.5 x reference, both centerline L2 <= 2 % U and <= 2 x baseline, drift 0,
overflow 0, no divergence.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

U_LID = 1.0
SAMPLE_COUNT = 201
FLUID_HALF = 0.5
STAGES = ["predict", "update_voxel", "correction", "density_kernel", "density_copy", "force"]


def wendland_c4_3d(r: np.ndarray, h: float) -> np.ndarray:
    q = r / h
    inside = q < 1.0
    w = np.zeros_like(q)
    qi = q[inside]
    w[inside] = (495.0 / (32.0 * math.pi * h ** 3)) * (1.0 - qi) ** 6 * (1.0 + 6.0 * qi + 35.0 / 3.0 * qi ** 2)
    return w


def shepard_interpolate(tree, position, volume, field, points, h):
    """SPH Shepard interpolation of a per-particle field at the given points."""
    values = np.zeros(points.shape[0])
    neighbor_lists = tree.query_ball_point(points, h)
    for index, neighbors in enumerate(neighbor_lists):
        if not neighbors:
            values[index] = np.nan
            continue
        neighbors = np.asarray(neighbors)
        r = np.linalg.norm(position[neighbors] - points[index], axis=1)
        w = wendland_c4_3d(r, h) * volume[neighbors]
        total = w.sum()
        values[index] = (w * field[neighbors]).sum() / total if total > 0 else np.nan
    return values


def analyze_case(case_dir: pathlib.Path, snapshot_path: pathlib.Path) -> dict:
    from scipy.spatial import cKDTree

    result = json.loads((case_dir / "result.json").read_text())
    records = [json.loads(line) for line in (case_dir / "run.jsonl").read_text().splitlines() if line.strip()]
    breakdown = json.loads((case_dir / "breakdown.json").read_text()) if (case_dir / "breakdown.json").exists() else None
    snap = np.load(snapshot_path)
    h, dx, hdx = float(snap["h"]), float(snap["dx"]), float(snap["hdx"])
    position = snap["s0_position"].astype(np.float64)
    velocity = snap["s0_velocity"].astype(np.float64)
    mass = snap["s0_mass"].astype(np.float64)
    density = snap["s0_density"].astype(np.float64)
    pressure = snap["s0_pressure"].astype(np.float64)
    kernel_sum = snap["s0_kernel_sum"].astype(np.float64)
    material = snap["s0_material"].astype(np.int64)
    kinds = snap["material_kinds"]
    rest = snap["material_rest_density"]
    fluid = kinds[np.minimum(material, kinds.size - 1)] == int(snap["fluid_kind_code"])
    finite = np.isfinite(position).all(axis=1) & np.isfinite(velocity).all(axis=1) & np.isfinite(density)
    nan_count = int((~finite).sum())
    rho0 = float(rest[np.minimum(material, rest.size - 1)][fluid][0]) if fluid.any() else 1000.0

    out = {"hdx": hdx, "h": h, "dx": dx, "steps": result["steps"], "timestep": result["timestep"],
           "physical_time": result["physical_time"], "particles": result["particles"],
           "expected_neighbors": 4.0 / 3.0 * math.pi * hdx ** 3,
           "us_per_step": result["us_per_step"], "s_per_T": result["wall_sim_seconds"],
           "status": result["status"], "nan_count": nan_count}
    final = records[-1]
    out.update({"fallback": final["correction_fallback_count"], "drift": final["drift"],
                "dead": final.get("dead_count", 0),
                "overflow_inside": final["overflow_inside_count"],
                "overflow_incoming": final["overflow_incoming_count"],
                "max_speed_final": final["max_speed"],
                "max_speed_series": max(r["max_speed"] for r in records)})
    ke = np.array([r["kinetic_energy"] for r in records])
    t = np.array([r["time"] for r in records])
    # Spike detector on the second half of the run (the first half is the
    # smooth spin-up, where KE legitimately grows by tens of % per record):
    # max relative change between consecutive 1000-step records.
    later = t > 0.5 * t[-1]
    jumps = np.abs(np.diff(ke)) / np.maximum(ke[:-1], 1e-12)
    out["ke_max_rel_jump"] = float(jumps[later[1:]].max()) if later[1:].any() else float(jumps.max())
    out["ke_final"] = float(ke[-1])
    out["ke_finite"] = bool(np.isfinite(ke).all())
    out["ke_series"] = [[float(a), float(b)] for a, b in zip(t, ke)]

    # Snapshot health (fluid).
    rel = density[fluid & finite] / rho0 - 1.0
    out["rho_std_rel"] = float(rel.std()) if rel.size else float("nan")
    out["rho_max_abs_rel"] = float(np.abs(rel).max()) if rel.size else float("nan")
    wall_positions = position[~fluid & finite]
    fluid_positions = position[fluid & finite]
    wall_tree = cKDTree(wall_positions)
    distance_to_wall, _ = wall_tree.query(fluid_positions, k=1)
    interior = distance_to_wall > h
    p_interior = pressure[fluid & finite][interior]
    out["interior_count"] = int(interior.sum())
    out["p_std_interior_norm"] = float(p_interior.std() / (rho0 * U_LID ** 2)) if p_interior.size else float("nan")
    w_interior = kernel_sum[fluid & finite][interior]
    out["wsum_mean_interior"] = float(w_interior.mean()) if w_interior.size else float("nan")
    out["wsum_q01_interior"] = float(np.quantile(w_interior, 0.01)) if w_interior.size else float("nan")
    all_tree = cKDTree(position[finite])
    nn_distance, _ = all_tree.query(position[finite], k=2)
    out["min_spacing_over_dx"] = float(nn_distance[:, 1].min() / dx)
    out["pairs_below_half_dx"] = int(len(all_tree.query_pairs(0.5 * dx)))

    # Centerlines on the mid-plane z = 0 (fluid cube [-0.5, 0.5]^3).
    s = np.linspace(-FLUID_HALF, FLUID_HALF, SAMPLE_COUNT)
    vertical = np.column_stack([np.zeros_like(s), s, np.zeros_like(s)])      # x = 0, z = 0 -> u(y)
    horizontal = np.column_stack([s, np.zeros_like(s), np.zeros_like(s)])    # y = 0, z = 0 -> v(x)
    volume = mass[finite] / density[finite]
    out["u_of_y"] = shepard_interpolate(all_tree, position[finite], volume, velocity[finite, 0], vertical, h).tolist()
    out["v_of_x"] = shepard_interpolate(all_tree, position[finite], volume, velocity[finite, 1], horizontal, h).tolist()
    out["sample_coordinate"] = s.tolist()

    if breakdown:
        particles = breakdown["alive"]
        out["kernel_ns_per_particle"] = {stage: breakdown["mean_us"][stage] * 1000.0 / particles for stage in STAGES}
        out["kernel_ns_per_particle"]["step"] = breakdown["mean_us"]["gpu_frame"] * 1000.0 / particles
        out["breakdown_frame_us"] = breakdown["mean_us"]["gpu_frame"]
    return out


def l2(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)) / U_LID) if ok.any() else float("nan")


def fmt(x, digits=2):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="docs/hdx_sweep_3d_20260924")
    parser.add_argument("--snapshots", default=None, help="snapshot dir (default <dir>/snapshots)")
    parser.add_argument("--fig", default="manuscripts/fig/hdx_sweep_3d")
    parser.add_argument("--reference", type=float, default=4.0)
    parser.add_argument("--baseline", type=float, default=3.5)
    args = parser.parse_args()

    sweep_dir = _REPO / args.dir
    snapshot_dir = pathlib.Path(args.snapshots) if args.snapshots else sweep_dir / "snapshots"
    cases = {}
    for case_dir in sorted(sweep_dir.glob("hdx*")):
        if not (case_dir / "result.json").exists():
            continue
        tag = case_dir.name
        snapshot_path = snapshot_dir / f"{tag}.npz"
        if not snapshot_path.exists():
            print(f"[hdx] {tag}: no snapshot, skipped", file=sys.stderr)
            continue
        print(f"[hdx] analyzing {tag} ...", flush=True)
        cases[float(tag[3:])] = analyze_case(case_dir, snapshot_path)
    if args.reference not in cases:
        raise SystemExit(f"reference h/dx={args.reference} missing")
    reference = cases[args.reference]
    baseline_u = l2(cases[args.baseline]["u_of_y"], reference["u_of_y"]) if args.baseline in cases else float("nan")
    baseline_v = l2(cases[args.baseline]["v_of_x"], reference["v_of_x"]) if args.baseline in cases else float("nan")

    for hdx, case in cases.items():
        case["l2_u"] = l2(case["u_of_y"], reference["u_of_y"])
        case["l2_v"] = l2(case["v_of_x"], reference["v_of_x"])
        case["speedup_s_per_T"] = reference["s_per_T"] / case["s_per_T"]
        checks = {
            "fallback_zero": case["fallback"] == 0,
            "no_close_pairs": case["pairs_below_half_dx"] == 0,
            "rho_std_within_1.5x": case["rho_std_rel"] <= 1.5 * reference["rho_std_rel"],
            "l2_u_ok": case["l2_u"] <= 0.02 and (not math.isfinite(baseline_u) or case["l2_u"] <= 2.0 * baseline_u or hdx == args.reference),
            "l2_v_ok": case["l2_v"] <= 0.02 and (not math.isfinite(baseline_v) or case["l2_v"] <= 2.0 * baseline_v or hdx == args.reference),
            "conserved": case["drift"] == 0 and case["overflow_inside"] == 0 and case["overflow_incoming"] == 0 and case["dead"] == 0,
            "no_divergence": (case["nan_count"] == 0 and case["ke_finite"]
                              and case["max_speed_series"] <= 3.0 * U_LID
                              and case["ke_max_rel_jump"] <= 0.10),
        }
        case["checks"] = checks
        case["pass"] = all(checks.values())
    passing = [hdx for hdx in sorted(cases) if cases[hdx]["pass"]]
    recommended = min(passing) if passing else None

    # ---- summary.md ------------------------------------------------------
    lines = [f"# 3-D h/dx sweep (E6) — V0, one RTX 5090, {reference['particles']:,} particles (101^3 fluid + 4 wall layers), T = {reference['physical_time']:.3f} s\n"]
    lines.append("| h/dx | neighbors | dt (s) | steps to T | us/step | s per T | speed-up | fallback | rho std/rho0 | min spacing/dx | pairs < 0.5 dx | u(y) L2 | v(x) L2 | verdict |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for hdx in sorted(cases, reverse=True):
        c = cases[hdx]
        lines.append(f"| {hdx:.2f} | {c['expected_neighbors']:.0f} | {c['timestep']:.2e} | {c['steps']:,} | {c['us_per_step']:.0f} | "
                     f"{c['s_per_T']:.1f} | {c['speedup_s_per_T']:.2f}x | {c['fallback']} | {c['rho_std_rel']:.2e} | "
                     f"{fmt(c['min_spacing_over_dx'], 3)} | {c['pairs_below_half_dx']} | {fmt(100 * c['l2_u'])} % | {fmt(100 * c['l2_v'])} % | "
                     f"{'PASS' if c['pass'] else 'FAIL'} |")
    lines.append(f"\nResolution-sensitivity baseline (h/dx {args.reference:g} vs {args.baseline:g}): u L2 = {fmt(100 * baseline_u)} %, v L2 = {fmt(100 * baseline_v)} % of U.")
    lines.append("\n## Secondary metrics\n")
    lines.append("| h/dx | max|rho-rho0|/rho0 | interior p std/(rho0 U^2) | W_sum mean (int.) | W_sum 1 % (int.) | max|v|/U (series / final) | KE max rel. jump | dead | overflow in/inc | drift | NaN |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for hdx in sorted(cases, reverse=True):
        c = cases[hdx]
        lines.append(f"| {hdx:.2f} | {c['rho_max_abs_rel']:.2e} | {fmt(c['p_std_interior_norm'], 4)} | {fmt(c['wsum_mean_interior'], 4)} | "
                     f"{fmt(c['wsum_q01_interior'], 4)} | {c['max_speed_series']:.3f} / {c['max_speed_final']:.3f} | {c['ke_max_rel_jump']:.3f} | "
                     f"{c['dead']} | {c['overflow_inside']}/{c['overflow_incoming']} | {c['drift']} | {c['nan_count']} |")
    lines.append("\n## Per-kernel ns per particle (sync loop, warmup 1000 + 2000 timed steps)\n")
    lines.append("| h/dx | " + " | ".join(STAGES) + " | step |")
    lines.append("|---|" + "---|" * (len(STAGES) + 1))
    for hdx in sorted(cases, reverse=True):
        k = cases[hdx].get("kernel_ns_per_particle")
        if k:
            lines.append(f"| {hdx:.2f} | " + " | ".join(f"{k[s]:.3f}" for s in STAGES) + f" | {k['step']:.3f} |")
    lines.append("\n## Checks per case\n")
    for hdx in sorted(cases, reverse=True):
        c = cases[hdx]
        failed = [name for name, ok in c["checks"].items() if not ok]
        lines.append(f"- h/dx {hdx:.2f}: {'PASS' if c['pass'] else 'FAIL (' + ', '.join(failed) + ')'}")
    if recommended is not None:
        lines.append(f"\n**Recommendation: h/dx = {recommended:g}, {cases[recommended]['speedup_s_per_T']:.2f}x faster per unit physical "
                     f"time than h/dx = {args.reference:g} ({cases[recommended]['s_per_T']:.1f} s vs {reference['s_per_T']:.1f} s for T = {reference['physical_time']:.2f} s).**")
    else:
        lines.append("\n**No h/dx below the reference passed all criteria.**")
    text = "\n".join(lines) + "\n"
    (sweep_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text)

    slim = {str(hdx): {k: v for k, v in c.items() if k not in ("u_of_y", "v_of_x", "sample_coordinate", "ke_series")}
            for hdx, c in cases.items()}
    slim["_baseline"] = {"reference": args.reference, "baseline": args.baseline, "l2_u": baseline_u, "l2_v": baseline_v,
                         "recommended": recommended}
    (sweep_dir / "summary.json").write_text(json.dumps(slim, indent=1), encoding="utf-8")
    with open(sweep_dir / "profiles.csv", "w") as handle:
        handle.write("coordinate," + ",".join(f"u_hdx{hdx:g},v_hdx{hdx:g}" for hdx in sorted(cases, reverse=True)) + "\n")
        for index, coordinate in enumerate(reference["sample_coordinate"]):
            handle.write(f"{coordinate:.5f}," + ",".join(
                f"{cases[hdx]['u_of_y'][index]:.6f},{cases[hdx]['v_of_x'][index]:.6f}" for hdx in sorted(cases, reverse=True)) + "\n")

    # ---- figure -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9.0, "axes.labelsize": 9.0, "legend.fontsize": 8.0,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.edgecolor": "#333333", "axes.linewidth": 0.6, "grid.color": "#d9d9d9", "grid.linewidth": 0.4,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    hdxs = sorted(cases)
    colors = {4.0: "#2a78d6", 3.5: "#1baf7a", 3.0: "#eda100", 2.5: "#eb6834", 2.0: "#e34948"}
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 8.4), gridspec_kw={"hspace": 0.42, "wspace": 0.32})
    (ax_spt, ax_us), (ax_rho, ax_fb), (ax_u, ax_v) = axes
    ax_spt.plot(hdxs, [cases[x]["s_per_T"] for x in hdxs], "o-", color="#2a78d6", linewidth=1.0, markersize=4)
    ax_spt.set_ylabel("wall seconds per T"); ax_spt.set_xlabel("h / Δx"); ax_spt.grid(True, axis="y")
    ax_spt.set_ylim(bottom=0)
    ax_us.plot(hdxs, [cases[x]["us_per_step"] for x in hdxs], "o-", color="#2a78d6", linewidth=1.0, markersize=4)
    ax_us.set_ylabel("µs per step"); ax_us.set_xlabel("h / Δx"); ax_us.grid(True, axis="y"); ax_us.set_ylim(bottom=0)
    ax_rho.plot(hdxs, [cases[x]["rho_std_rel"] for x in hdxs], "o-", color="#2a78d6", linewidth=1.0, markersize=4)
    ax_rho.set_yscale("log"); ax_rho.set_ylabel("fluid ρ std / ρ₀"); ax_rho.set_xlabel("h / Δx"); ax_rho.grid(True, axis="y")
    fallback_values = [cases[x]["fallback"] for x in hdxs]
    ax_fb.plot(hdxs, fallback_values, "o-", color="#2a78d6", linewidth=1.0, markersize=4)
    ax_fb.set_ylabel("correction fallback count (cumulative)")
    ax_fb.set_xlabel("h / Δx"); ax_fb.grid(True, axis="y")
    top = max(1, int(max(fallback_values) * 1.2))
    ax_fb.set_ylim(-0.05 * top, top)
    ax_fb.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    if max(fallback_values) == 0:
        ax_fb.text(0.5, 0.55, "0 for every h/Δx", transform=ax_fb.transAxes, ha="center", fontsize=8.5, color="#333333")
    for x in sorted(hdxs, reverse=True):
        c = cases[x]
        label = f"h/Δx = {x:g}" + ("" if c["pass"] else " (fail)")
        ax_u.plot(c["u_of_y"], c["sample_coordinate"], color=colors.get(x, "#333333"), linewidth=1.0, label=label)
        ax_v.plot(c["sample_coordinate"], c["v_of_x"], color=colors.get(x, "#333333"), linewidth=1.0, label=label)
    ax_u.set_xlabel("u / U on x = 0, z = 0"); ax_u.set_ylabel("y / L"); ax_u.grid(True); ax_u.legend(frameon=False, loc="lower right")
    ax_v.set_xlabel("x / L"); ax_v.set_ylabel("v / U on y = 0, z = 0"); ax_v.grid(True)
    for ax, letter in zip(axes.ravel(), "abcdef"):
        ax.text(-0.18, 1.04, f"({letter})", transform=ax.transAxes, fontsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig_path = _REPO / args.fig
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(sweep_dir / "hdx_sweep_3d.png", dpi=200, bbox_inches="tight")
    print(f"[hdx] figure -> {fig_path.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
