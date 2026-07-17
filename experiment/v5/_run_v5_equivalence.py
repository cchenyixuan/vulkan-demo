"""
_run_v5_equivalence.py — M4 numerical-equivalence battery.

Question answered: does a K-slab chain compute the SAME physics as a single
GPU? Slot-level bit-exactness is impossible on this solver (atomicAdd slot
ordering + FP reduction reorder + chaotic amplification — established during
M1), so the criterion is the envelope method: run K=1 TWICE to measure the
run-to-run FP-nondeterminism envelope on aggregate physical quantities, then
require every K>1 config's delta against the K=1 reference to sit within
that envelope (x tolerance factor).

All configs run the same case, same step count, same phased pipeline
(K=1 goes through the chain path too — no peers, phase C waits phase_a).

Usage:
    .venv/Scripts/python.exe experiment/v5/_run_v5_equivalence.py \\
        --case cases/lid_driven_cavity_2d/case.yaml --steps 2000 \\
        --configs 1,1,4,6,8
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

ENVELOPE_FACTOR = 10.0      # cross-config delta must be <= 10x the K=1
                            # run-to-run envelope (or negligible vs scale)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V5 chain numerical-equivalence battery")
    p.add_argument("--case", default="cases/lid_driven_cavity_2d/case.yaml")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--configs", default="1,1,4,6,8",
                   help="comma-separated K values; MUST start with 1,1 "
                        "(the baseline envelope pair)")
    p.add_argument("--pool-safety", type=float, default=1.2)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--sync-scheme", default="per-direction",
                   choices=["aggregated", "per-direction"])
    p.add_argument("--defrag-cadence", type=int, default=None)
    return p.parse_args()


def run_config(global_case, slab_count: int, args) -> dict:
    """Build a K-slab chain, run --steps, read back merged aggregate stats."""
    from experiment.v5.utils.orchestrator_v5 import ChainOrchestratorV5
    from experiment.v5.utils.partition_v5 import compute_chain_partition
    from experiment.v5.utils.simulator_v5 import SphSimulatorV5
    from experiment.v5.utils.vulkan_context_v5 import VulkanContextV5

    chain = compute_chain_partition(
        global_case, [1.0] * slab_count, pool_safety=args.pool_safety)
    device_map = [index % 2 for index in range(slab_count)]
    defrag_cadence = (args.defrag_cadence if args.defrag_cadence is not None
                      else global_case.numerics.defrag_cadence)

    contexts, sims = [], []
    try:
        for index in range(slab_count):
            contexts.append(VulkanContextV5.create(
                device_index=device_map[index],
                application_name=f"equiv_v5_k{slab_count}_s{index}"))
            sims.append(SphSimulatorV5(contexts[-1], chain.slabs[index],
                                       sync_scheme=args.sync_scheme))
        with ChainOrchestratorV5(sims, defrag_cadence=defrag_cadence) as orch:
            orch.bootstrap_all()
            orch.run_pipelined(args.steps, depth=args.depth, warmup=0)
            for sim in sims:
                sim.submit_defrag_and_wait()

            masses, velocities, densities, positions = [], [], [], []
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
                masses.append(velocity_mass[own, 3][alive].astype(np.float64))
                velocities.append(
                    velocity_mass[own, 0:3][alive].astype(np.float64))
                densities.append(
                    density_pressure[own, 0][alive].astype(np.float64))
                positions.append(
                    position_voxel[own, 0:2][alive].astype(np.float64))
    finally:
        for sim in sims:
            sim.destroy()
        for ctx in contexts:
            ctx.destroy()

    mass = np.concatenate(masses)
    velocity = np.concatenate(velocities)
    density = np.concatenate(densities)
    position = np.concatenate(positions)
    return {
        "n": int(mass.shape[0]),
        "kinetic_energy": float(0.5 * (mass * (velocity ** 2).sum(1)).sum()),
        "momentum_x": float((mass * velocity[:, 0]).sum()),
        "momentum_y": float((mass * velocity[:, 1]).sum()),
        "mean_density": float(density.mean()),
        "max_speed": float(np.sqrt((velocity ** 2).sum(1)).max()),
        "center_x": float((mass * position[:, 0]).sum() / mass.sum()),
        "center_y": float((mass * position[:, 1]).sum() / mass.sum()),
    }


def main() -> int:
    args = parse_args()
    configs = [int(k) for k in args.configs.split(",")]
    if len(configs) < 3 or configs[0] != 1 or configs[1] != 1:
        sys.exit("--configs must start with the baseline pair '1,1'")

    from experiment.v5.utils.case_loader_v5 import load_case_v5
    global_case = load_case_v5(args.case)

    results = []
    for run_index, slab_count in enumerate(configs):
        print(f"\n[equiv] === run {run_index}: K={slab_count} "
              f"({args.steps} steps) ===", flush=True)
        results.append(run_config(global_case, slab_count, args))

    reference, rerun = results[0], results[1]
    metrics = [k for k in reference if k != "n"]
    print(f"\n[equiv] baseline envelope (K=1 vs K=1 rerun):")
    envelope = {}
    for metric in metrics:
        envelope[metric] = abs(reference[metric] - rerun[metric])
        print(f"  {metric:<16} {envelope[metric]:.6e}")

    all_ok = reference["n"] == rerun["n"]
    print(f"\n{'metric':<16} " + " ".join(
        f"{'K=' + str(k):>14}" for k in configs[2:]) + "   verdict")
    for metric in metrics:
        deltas = [abs(reference[metric] - result[metric])
                  for result in results[2:]]
        scale = max(abs(reference[metric]), 1e-30)
        ok = all(delta <= max(ENVELOPE_FACTOR * envelope[metric],
                              1e-9 * scale) for delta in deltas)
        all_ok &= ok
        print(f"{metric:<16} " + " ".join(f"{d:>14.6e}" for d in deltas)
              + f"   {'PASS' if ok else 'FAIL'}")
    n_ok = all(result["n"] == reference["n"] for result in results[2:])
    all_ok &= n_ok
    print(f"{'n (exact)':<16} " + " ".join(
        f"{result['n'] - reference['n']:>14}" for result in results[2:])
        + f"   {'PASS' if n_ok else 'FAIL'}")

    print(f"\n[equiv] {'ALL PASS' if all_ok else '*** FAIL ***'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
