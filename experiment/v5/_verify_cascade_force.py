"""
_verify_cascade_force.py — per-particle A/B of V3.3 cascading force (V5_CASCADE_FORCE=1)
against the legacy force_all path, on a K=2 chain.

The cascade moves force_deep_interior into Phase B (reading this frame's rho/P from
density_pressure_scratch) and leaves force_boundary (4-column band) in Phase C. If the
band arithmetic or the data-dependency argument were wrong, the error would appear as a
LOCALIZED difference at the band edge (columns 3..4 from the seam), far above the
run-to-run floating-point floor (atomicAdd slot ordering -> neighbour sum order).

So we run the SAME case four times (legacy x2, cascade x2), match particles across runs
by position (KD-tree), and compare per-particle acceleration / shift / velocity / density
/ position:
  * legacy-vs-legacy and cascade-vs-cascade = the run-to-run noise floor
  * legacy-vs-cascade                       = the test
and break the acceleration difference down by voxel-column distance to the seam.

Usage (driver, runs 4 subprocesses on GPUs 0,1):
    .venv/Scripts/python.exe experiment/v5/_verify_cascade_force.py \\
        --case cases/lid_driven_cavity_2d/case.yaml --steps 300 --out logs/verify_cascade
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

FIELDS = ("position", "velocity", "density", "acceleration", "shift")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="cases/lid_driven_cavity_2d/case.yaml")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--pool-safety", type=float, default=1.2)
    p.add_argument("--device-map", default="0,1")
    p.add_argument("--slabs", type=int, default=2, help="K (sims cycle over --device-map)")
    p.add_argument("--a-env", default="",
                   help="comma-separated KEY=VAL for the reference runs (A), e.g. '' = legacy")
    p.add_argument("--b-env", default="V5_CASCADE_FORCE=1",
                   help="comma-separated KEY=VAL for the candidate runs (B)")
    p.add_argument("--out", default="logs/verify_cascade")
    p.add_argument("--dump", default=None,
                   help="(worker mode) run one K=2 chain and save per-particle state to this .npz")
    return p.parse_args()


# --------------------------------------------------------------------------- worker
def dump_state(args) -> None:
    from experiment.v5.utils.case_loader_v5 import load_case_v5
    from experiment.v5.utils.orchestrator_v5 import ChainOrchestratorV5
    from experiment.v5.utils.partition_v5 import compute_chain_partition
    from experiment.v5.utils.simulator_v5 import SphSimulatorV5
    from experiment.v5.utils.vulkan_context_v5 import VulkanContextV5

    global_case = load_case_v5(args.case)
    device_map = [int(x) for x in args.device_map.split(",")]
    chain = compute_chain_partition(global_case, [1.0] * args.slabs, pool_safety=args.pool_safety)
    contexts, sims = [], []
    saved = {}
    try:
        for index in range(args.slabs):
            contexts.append(VulkanContextV5.create(
                device_index=device_map[index % len(device_map)],
                application_name=f"verify_cascade_s{index}"))
            # per-direction timelines: the only scheme that serves interior
            # chain nodes (two inbound workers) — same as the production runners.
            sims.append(SphSimulatorV5(contexts[-1], chain.slabs[index],
                                       sync_scheme="per-direction"))
        # defrag would re-order pids and its own scratch copies; keep it out of the window.
        with ChainOrchestratorV5(sims, defrag_cadence=10 ** 9) as orch:
            orch.bootstrap_all()
            orch.run_pipelined(args.steps, depth=args.depth, warmup=0)
            for index, sim in enumerate(sims):
                capacities = sim.case.capacities
                pool = capacities.total_pool_capacity()
                raw = sim.readback_buffers_batch(
                    ["position_voxel_id", "velocity_mass", "density_pressure",
                     "acceleration", "shift", "inside_particle_count"])
                position_voxel = np.frombuffer(raw["position_voxel_id"], np.float32).reshape(pool, 4)
                # Band invariant (V3.4 band-voxel dispatch): the particles reachable
                # through the band voxels' lists must be exactly the own alive
                # particles whose voxel column lies in the band, for every band
                # width the boundary kernels use (2 / 3 / 4 columns).
                grid, ghost = sim.case.grid, sim.case.ghost_grid
                face = grid.grid_dimension_y * grid.grid_dimension_z
                leading_x = ghost.leading_ghost_voxel_count // face
                trailing_x = ghost.trailing_ghost_voxel_count // face
                voxel_counts = np.frombuffer(raw["inside_particle_count"], np.uint32)
                own_slice = slice(sim.own_first_pid(), sim.own_first_pid() + capacities.own_pool_size)
                own_vid = np.rint(position_voxel[own_slice, 3]).astype(np.int64)
                own_alive = (velocity_mass_all := np.frombuffer(raw["velocity_mass"], np.float32)
                             .reshape(pool, 4))[own_slice, 3] > 0
                own_x = (own_vid - 1) // face
                for band in (2, 3, 4):
                    columns = []
                    if leading_x > 0:
                        columns += list(range(leading_x, leading_x + band))
                    if trailing_x > 0:
                        own_last_x = grid.grid_dimension_x - 1 - trailing_x
                        columns += list(range(own_last_x - band + 1, own_last_x + 1))
                    list_count = sum(int(voxel_counts[1 + c * face: 1 + (c + 1) * face].sum()) for c in columns)
                    coordinate_count = int((own_alive & (own_vid > 0) & np.isin(own_x, columns)).sum())
                    saved[f"s{index}_band{band}_list_count"] = np.int64(list_count)
                    saved[f"s{index}_band{band}_coordinate_count"] = np.int64(coordinate_count)
                velocity_mass = np.frombuffer(raw["velocity_mass"], np.float32).reshape(pool, 4)
                density_pressure = np.frombuffer(raw["density_pressure"], np.float32).reshape(pool, 2)
                acceleration = np.frombuffer(raw["acceleration"], np.float32).reshape(pool, 4)
                shift = np.frombuffer(raw["shift"], np.float32).reshape(pool, 4)
                own = slice(sim.own_first_pid(), sim.own_first_pid() + capacities.own_pool_size)
                alive = (velocity_mass[own, 3] > 0) & (position_voxel[own, 3] > 0.5)
                saved[f"s{index}_position"] = position_voxel[own, 0:3][alive]
                saved[f"s{index}_velocity"] = velocity_mass[own, 0:3][alive]
                saved[f"s{index}_density"] = density_pressure[own, 0][alive]
                saved[f"s{index}_pressure"] = density_pressure[own, 1][alive]
                saved[f"s{index}_acceleration"] = acceleration[own, 0:3][alive]
                saved[f"s{index}_shift"] = shift[own, 0:3][alive]
    finally:
        for sim in sims:
            sim.destroy()
        for ctx in contexts:
            ctx.destroy()
    saved["origin_x"] = np.float64(global_case.grid.origin_x)
    saved["h"] = np.float64(global_case.physics.smoothing_length)
    saved["cuts"] = np.array(chain.cuts, dtype=np.int64)
    saved["slabs"] = np.int64(args.slabs)
    saved["cascade"] = np.int64(1 if os.environ.get("V5_CASCADE_FORCE", "0") == "1" else 0)
    np.savez(args.dump, **saved)
    print(f"[verify] dumped {args.dump}: " + ", ".join(
        f"s{i} n={saved[f's{i}_position'].shape[0]}" for i in range(args.slabs)), flush=True)


# --------------------------------------------------------------------------- driver
def match(a_pos: np.ndarray, b_pos: np.ndarray, tolerance: float):
    """Index into b for every a particle (nearest neighbour), and the matched mask."""
    from scipy.spatial import cKDTree
    tree = cKDTree(b_pos)
    distance, index = tree.query(a_pos, k=1)
    return index, distance <= tolerance


def compare(a: dict, b: dict, sim: int, tolerance: float) -> dict:
    a_pos, b_pos = a[f"s{sim}_position"], b[f"s{sim}_position"]
    index, ok = match(a_pos, b_pos, tolerance)
    out = {"n_a": int(a_pos.shape[0]), "n_b": int(b_pos.shape[0]), "unmatched": int((~ok).sum())}
    for field in FIELDS:
        fa = a[f"s{sim}_{field}"][ok].astype(np.float64)
        fb = b[f"s{sim}_{field}"][index[ok]].astype(np.float64)
        diff = fa - fb
        norm = np.linalg.norm(diff, axis=1) if diff.ndim == 2 else np.abs(diff)
        scale = float(np.max(np.linalg.norm(fa, axis=1)) if fa.ndim == 2 else np.max(np.abs(fa)))
        out[field] = {"max": float(norm.max()), "p999": float(np.percentile(norm, 99.9)),
                      "mean": float(norm.mean()), "scale": scale}
    # acceleration difference by voxel-column distance to the seam
    h, origin_x = float(a["h"]), float(a["origin_x"])
    cuts = np.asarray(a["cuts"], dtype=np.int64)
    global_column = np.floor((a_pos[ok, 0] - origin_x) / h).astype(np.int64)
    # 0-based distance to the NEAREST seam: right of a cut = col - cut,
    # left of it = cut - 1 - col. The force band is distance 0..3 on each side.
    right = global_column[:, None] - cuts[None, :]
    col = np.where(right >= 0, right, -right - 1).min(axis=1)
    accel_diff = np.linalg.norm(
        a[f"s{sim}_acceleration"][ok].astype(np.float64)
        - b[f"s{sim}_acceleration"][index[ok]].astype(np.float64), axis=1)
    by_column = {}
    for c in range(0, 16):
        mask = col == c
        if mask.any():
            by_column[c] = {"n": int(mask.sum()), "max": float(accel_diff[mask].max()),
                            "mean": float(accel_diff[mask].mean())}
    out["accel_by_column"] = by_column
    return out


def main() -> int:
    args = parse_args()
    if args.dump:
        dump_state(args)
        return 0

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    def parse_env(text: str) -> dict:
        return dict(item.split("=", 1) for item in text.split(",") if item.strip())

    # "legacy_*" = reference configuration A, "cascade_*" = candidate configuration B
    # (names kept for the report; the actual switches come from --a-env / --b-env).
    a_env, b_env = parse_env(args.a_env), parse_env(args.b_env)
    print(f"[verify] A (reference) env: {a_env or '{legacy}'}   B (candidate) env: {b_env}", flush=True)
    runs = [("legacy_1", a_env), ("legacy_2", a_env), ("cascade_1", b_env), ("cascade_2", b_env)]
    dumps = {}
    for name, run_env in runs:
        path = out_dir / f"{name}.npz"
        env = dict(os.environ, V5_CASCADE_FORCE="0", V5_BAND_VOXEL_DISPATCH="0",
                   VK_LOADER_LAYERS_DISABLE="VK_LAYER_KHRONOS_validation")
        env.update(run_env)
        cascade = env["V5_CASCADE_FORCE"]
        cmd = [sys.executable, __file__, "--case", args.case, "--steps", str(args.steps),
               "--slabs", str(args.slabs),
               "--depth", str(args.depth), "--pool-safety", str(args.pool_safety),
               "--device-map", args.device_map, "--dump", str(path)]
        print(f"[verify] run {name} (V5_CASCADE_FORCE={cascade}) ...", flush=True)
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        tail = [l for l in result.stdout.splitlines() if l.startswith("[verify]")]
        print("   " + (tail[-1] if tail else result.stdout[-300:]), flush=True)
        if result.returncode != 0:
            print(result.stderr[-2000:])
            return 1
        dumps[name] = dict(np.load(path))

    # band invariant: voxel-list reach == coordinate membership, every run, every sim, every band width
    invariant_ok = True
    for name, dump in dumps.items():
        for sim in range(int(dump["slabs"])):
            parts = []
            for band in (2, 3, 4):
                lc, cc = int(dump[f"s{sim}_band{band}_list_count"]), int(dump[f"s{sim}_band{band}_coordinate_count"])
                parts.append(f"band{band}: lists {lc} / coords {cc}")
                invariant_ok &= (lc == cc)
            print(f"[verify] band invariant {name} s{sim}: " + "; ".join(parts), flush=True)
    print(f"[verify] band invariant (voxel lists cover exactly the band particles): "
          f"{'OK' if invariant_ok else '*** VIOLATED ***'}", flush=True)

    h = float(dumps["legacy_1"]["h"])
    tolerance = 0.05 * h  # well below the particle spacing (~h/4), far above FP drift
    pairs = [("legacy_1", "legacy_2", "noise: legacy vs legacy"),
             ("cascade_1", "cascade_2", "noise: cascade vs cascade"),
             ("legacy_1", "cascade_1", "TEST: legacy vs cascade"),
             ("legacy_2", "cascade_2", "TEST: legacy vs cascade (2nd pair)")]
    slabs = int(dumps["legacy_1"]["slabs"])
    report = {"steps": args.steps, "case": args.case, "slabs": slabs,
              "cuts": [int(c) for c in dumps["legacy_1"]["cuts"]], "pairs": {}}
    print(f"\n[verify] steps={args.steps} K={slabs} seam columns={report['cuts']} "
          f"match tolerance={tolerance:.2e} m")
    for a_name, b_name, label in pairs:
        print(f"\n--- {label} ---")
        report["pairs"][label] = {}
        for sim in range(slabs):
            r = compare(dumps[a_name], dumps[b_name], sim, tolerance)
            report["pairs"][label][f"s{sim}"] = r
            print(f"  s{sim}: n={r['n_a']}/{r['n_b']} unmatched={r['unmatched']}  " + "  ".join(
                f"{f}: max {r[f]['max']:.3e} (p99.9 {r[f]['p999']:.2e}, scale {r[f]['scale']:.2e})"
                for f in FIELDS))
    # column table: acceleration max diff per column distance, noise vs test, both sims
    print("\n[verify] max |delta acceleration| by voxel-column distance to the NEAREST seam "
          "(all sims, both sides; force band = distance 0..3, band edge between 3 and 4):")
    print(f"{'dist':>5s} {'n':>7s} {'noise L-L':>12s} {'noise C-C':>12s} {'TEST L-C':>12s} {'TEST ratio':>11s}")
    noise_ll = report["pairs"]["noise: legacy vs legacy"]
    noise_cc = report["pairs"]["noise: cascade vs cascade"]
    test = report["pairs"]["TEST: legacy vs cascade"]
    sims = sorted(noise_ll)

    def aggregate(pair, c, key):
        values = [pair[s]["accel_by_column"][c][key] for s in sims if c in pair[s]["accel_by_column"]]
        if not values:
            return None
        return max(values) if key == "max" else sum(values)

    rows = []
    for c in range(0, 16):
        n, m, t = aggregate(noise_ll, c, "max"), aggregate(noise_cc, c, "max"), aggregate(test, c, "max")
        count = aggregate(test, c, "n")
        if n is None or m is None or t is None:
            continue
        floor = max(n, m)
        ratio = t / floor if floor > 0 else float("inf")
        rows.append((c, count, n, m, t, ratio))
        print(f"{c:>5d} {count:>7d} {n:>12.3e} {m:>12.3e} {t:>12.3e} {ratio:>11.2f}")
    worst = max(rows, key=lambda r: r[5]) if rows else None
    verdict = "PASS" if (invariant_ok and worst is not None and worst[5] < 3.0) else "FAIL"
    report["band_invariant_ok"] = bool(invariant_ok)
    report["verdict"] = verdict
    report["worst_column_ratio"] = worst[5] if worst else None
    print(f"\n[verify] worst test/noise ratio over columns: {worst[5]:.2f} at column {worst[0]} -> {verdict}"
          if worst else "[verify] no columns compared -> FAIL")
    (out_dir / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = [r[0] for r in rows]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(cols, [max(r[2], r[3]) for r in rows], "o-", color="#2a78d6", label="run-to-run noise floor (max of L-L, C-C)")
        ax.plot(cols, [r[4] for r in rows], "s-", color="#eb6834", label="TEST: legacy vs cascade")
        ax.axvline(3.5, color="#888", linestyle="--", linewidth=1)
        ax.set_yscale("log"); ax.set_xlabel("voxel-column distance to the nearest seam (0 = seam column)")
        ax.set_ylabel("max |delta acceleration| in column (m/s^2)")
        ax.set_title(f"Cascading force A/B, {pathlib.Path(args.case).parent.name}, K={slabs}, "
                     f"{args.steps} steps; dashed = force band edge", fontsize=9)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out_dir / "accel_diff_by_column.png", dpi=140)
        print(f"[verify] figure: {out_dir / 'accel_diff_by_column.png'}")
    except Exception as exc:  # matplotlib optional
        print(f"[verify] figure skipped: {exc}")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
