"""
_run_v5_chain_bench.py — N-slab chain runner with virtual-GPU device mapping.

The M3 entry point: runs a ChainPartition of K slabs on any number of
physical devices via --device-map (K sims, each with its own VulkanContext;
sims sharing a physical device timeslice its queues — the virtual-GPU
emulation mode of docs/sph_v5_design.md §3.4). Pipelined (depth-2) wall-clock
fps + conservation + per-sim pool health + an end-of-run numeric
seam-integrity check at every cut (cross-side overshoot, A/B duplicates,
density band, vmax) — the same criteria the 8M snapshot study validated.

Usage:
    # K=3 on 2 physical GPUs (first interior slab live):
    .venv/Scripts/python.exe experiment/v5/_run_v5_chain_bench.py \\
        --weights 1,1,1 --device-map 0,1,0 --max-steps 20000 --warmup 5000
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V5 N-slab chain bench runner")
    p.add_argument("--case", default="cases/lid_driven_cavity_2d/case.yaml")
    p.add_argument("--weights", default="1.0,1.0,1.0",
                   help="K comma-separated slab weights (left to right)")
    p.add_argument("--device-map", default=None,
                   help="K comma-separated physical device indices; default "
                        "round-robin over 0,1")
    p.add_argument("--sync-scheme", default="per-direction",
                   choices=["aggregated", "per-direction"],
                   help="interior slabs require per-direction")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--pool-safety", type=float, default=1.2,
                   help="REQUIRED-ish for chains: None would give every sim "
                        "the full global pool (VRAM explosion at K sims per "
                        "device). Pass 0 to force None/legacy.")
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--warmup", type=int, default=5000)
    p.add_argument("--defrag-cadence", type=int, default=None)
    p.add_argument("--no-defrag", action="store_true")
    p.add_argument("--validation", action="store_true")
    p.add_argument("--seam-check", action="store_true", default=True)
    p.add_argument("--no-seam-check", dest="seam_check", action="store_false")
    return p.parse_args()


def seam_integrity_check(chain, sims, global_case) -> bool:
    """Numeric end-state check at every cut: overshoot, duplicates, fields.

    Reads back each sim's own alive particles (positions are global — the
    partition shifts only voxel-grid origins) and applies the same criteria
    the 8M snapshot study established: cross-side overshoot < 1 dx,
    zero cross-pair duplicates near each seam, density within the weakly-
    compressible band, vmax <= lid speed + tolerance."""
    h = global_case.physics.smoothing_length
    origin_x = global_case.grid.origin_x
    dx = 1.0 / np.sqrt(global_case.initial.positions.shape[0])

    per_sim = []
    for sim in sims:
        capacities = sim.case.capacities
        pool = capacities.total_pool_capacity()
        raw = sim.readback_buffers_batch(
            ["position_voxel_id", "velocity_mass", "density_pressure"])
        position_voxel = np.frombuffer(
            raw["position_voxel_id"], np.float32).reshape(pool, 4)
        velocity_mass = np.frombuffer(
            raw["velocity_mass"], np.float32).reshape(pool, 4)
        density_pressure = np.frombuffer(
            raw["density_pressure"], np.float32).reshape(pool, 2)
        own = slice(sim.own_first_pid(),
                    sim.own_first_pid() + capacities.own_pool_size)
        alive = velocity_mass[own, 3] > 0
        per_sim.append({
            "position": position_voxel[own, 0:2][alive],
            "speed": np.sqrt((velocity_mass[own, 0:3][alive] ** 2).sum(1)),
            "density": density_pressure[own, 0][alive],
        })

    all_ok = True
    for cut_index, cut in enumerate(chain.cuts):
        x_cut = origin_x + cut * h
        left, right = per_sim[cut_index], per_sim[cut_index + 1]
        left_overshoot = (left["position"][:, 0].max() - x_cut) / dx
        right_overshoot = (x_cut - right["position"][:, 0].min()) / dx
        band = 20 * dx
        left_band = left["position"][
            np.abs(left["position"][:, 0] - x_cut) < band]
        right_band = right["position"][
            np.abs(right["position"][:, 0] - x_cut) < band]
        quantized_left = set(map(tuple, np.round(left_band / (dx / 4)).astype(np.int64)))
        quantized_right = set(map(tuple, np.round(right_band / (dx / 4)).astype(np.int64)))
        duplicates = len(quantized_left & quantized_right)
        ok = (left_overshoot < 1.0 and right_overshoot < 1.0
              and duplicates == 0)
        all_ok &= ok
        print(f"[chain_v5] seam {cut_index} (col {cut}): "
              f"L_overshoot={left_overshoot:+.2f}dx "
              f"R_overshoot={right_overshoot:+.2f}dx dup={duplicates} "
              f"{'OK' if ok else '*** FAIL ***'}")

    density_all = np.concatenate([s["density"] for s in per_sim])
    speed_all = np.concatenate([s["speed"] for s in per_sim])
    rest = 1000.0
    density_ok = (density_all.min() > rest * 0.95
                  and density_all.max() < rest * 1.05)
    speed_ok = speed_all.max() <= 1.0 + 1e-3
    all_ok &= density_ok and speed_ok
    print(f"[chain_v5] fields: rho[{density_all.min():.1f},"
          f"{density_all.max():.1f}] vmax={speed_all.max():.4f} "
          f"{'OK' if density_ok and speed_ok else '*** FAIL ***'}")
    return all_ok


def main() -> int:
    args = parse_args()

    from experiment.v5.utils.case_loader_v5 import load_case_v5
    from experiment.v5.utils.orchestrator_v5 import ChainOrchestratorV5
    from experiment.v5.utils.partition_v5 import compute_chain_partition
    from experiment.v5.utils.simulator_v5 import SphSimulatorV5
    from experiment.v5.utils.vulkan_context_v5 import VulkanContextV5

    weights = [float(w) for w in args.weights.split(",")]
    slab_count = len(weights)
    if args.device_map is not None:
        device_map = [int(d) for d in args.device_map.split(",")]
        if len(device_map) != slab_count:
            sys.exit(f"--device-map needs {slab_count} entries")
    else:
        device_map = [index % 2 for index in range(slab_count)]
    pool_safety = None if args.pool_safety == 0 else args.pool_safety

    global_case = load_case_v5(args.case)
    expected_total = int(global_case.initial.positions.shape[0])
    chain = compute_chain_partition(global_case, weights, pool_safety)
    defrag_cadence = (args.defrag_cadence if args.defrag_cadence is not None
                      else global_case.numerics.defrag_cadence)
    if args.no_defrag:
        defrag_cadence = args.max_steps + 1

    print(f"[chain_v5] K={slab_count} weights={weights} "
          f"device_map={device_map} sync={args.sync_scheme} "
          f"depth={args.depth} pool_safety={pool_safety}")

    contexts, sims = [], []
    try:
        for index in range(slab_count):
            ctx = VulkanContextV5.create(
                device_index=device_map[index],
                enable_validation=args.validation,
                application_name=f"chain_v5_s{index}")
            contexts.append(ctx)
            sims.append(SphSimulatorV5(ctx, chain.slabs[index],
                                       sync_scheme=args.sync_scheme))

        def on_defrag(frame_n: int, report: list) -> None:
            migrations = "/".join(str(r["interval_migration"]) for r in report)
            drops = sum(r["overflow_install_tail"] for r in report)
            if drops:
                print(f"[migration] frame {frame_n}: interval {migrations} "
                      f"*** DROPS={drops} ***", file=sys.stderr, flush=True)

        with ChainOrchestratorV5(sims, defrag_cadence=defrag_cadence) as orch:
            orch.bootstrap_all()
            result = orch.run_pipelined(
                args.max_steps, depth=args.depth, warmup=args.warmup,
                on_defrag=on_defrag)
            print(f"[chain_v5] TOTAL: {result['frame_count']} steps in "
                  f"{result['elapsed_s']:.2f}s = {result['fps']:.1f} fps")
            if "steady_fps" in result:
                print(f"[chain_v5] STEADY (post-warmup {args.warmup}): "
                      f"{result['steady_frames']} steps in "
                      f"{result['steady_s']:.2f}s = "
                      f"{result['steady_fps']:.1f} fps")

            for sim in sims:
                sim.submit_defrag_and_wait()
            total = 0
            for index, sim in enumerate(sims):
                status = sim.readback_global_status()
                health = sim.readback_pool_health()
                total += status["alive_particle_count"]
                print(f"[chain_v5] sim{index} (dev{device_map[index]}): "
                      f"alive={status['alive_particle_count']:,} "
                      f"pool_used={health['used_fraction']*100:.1f}% "
                      f"peak_migration={health['peak_migration_count']} "
                      f"drops={status['overflow_install_tail']}")
            drift = total - expected_total
            print(f"[chain_v5] final: total={total:,} "
                  f"(expected {expected_total:,}) drift={drift}")

            seam_ok = True
            if args.seam_check:
                seam_ok = seam_integrity_check(chain, sims, global_case)
            if drift != 0 or not seam_ok:
                print("[chain_v5] *** VALIDATION FAILED ***")
                return 1
    finally:
        for sim in sims:
            sim.destroy()
        for ctx in contexts:
            ctx.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
