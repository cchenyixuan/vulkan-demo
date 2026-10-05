"""
_run_two_hop_equivalence.py — numerical equivalence of the two-hop transport,
by the envelope method of experiment/v5/_run_v5_equivalence.py.

Four runs of the same case and step count:

    K=1            reference                      (V5 run_config)
    K=1  rerun     scheduling nondeterminism      (V5 run_config)
    K=2  three_hop validated decomposition        (V5 run_config)
    K=2  two_hop   the configuration under test   (this file)

The envelope of every aggregate metric is the larger of |K1 - K1 rerun| and
|K1 - K2 three_hop| (floored at float32 noise), exactly as in the V5
battery; the two-hop run must sit within ENVELOPE_FACTOR x envelope of the
K=1 reference and conserve the particle count exactly.

Usage:
    VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation \\
    .venv/Scripts/python.exe experiment/two_hop/_run_two_hop_equivalence.py \\
        --case cases/lid_driven_cavity_2d/case.yaml --steps 2000
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.two_hop._run_two_hop_bench import (  # noqa: E402
    pin_campaign_environment)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="two-hop numerical equivalence")
    parser.add_argument("--case", default="cases/lid_driven_cavity_2d/case.yaml")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--pool-safety", type=float, default=1.2)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--sync-scheme", default="per-direction",
                        choices=["per-direction"])
    parser.add_argument("--defrag-cadence", type=int, default=None)
    parser.add_argument("--output", default=None, help="JSON report path")
    return parser.parse_args()


def run_two_hop_config(global_case, args) -> dict:
    """K=2 two-hop chain; same aggregate statistics as V5's run_config."""
    from experiment.two_hop.shared_host_v5 import (
        ChainOrchestratorTwoHop, SharedHostLinkPool, SphSimulatorTwoHop,
        VulkanContextTwoHop)
    from experiment.v5.utils.partition_v5 import compute_chain_partition

    slab_count = 2
    chain = compute_chain_partition(
        global_case, [1.0] * slab_count, pool_safety=args.pool_safety)
    defrag_cadence = (args.defrag_cadence if args.defrag_cadence is not None
                      else global_case.numerics.defrag_cadence)

    contexts, sims = [], []
    link_pool = None
    try:
        for index in range(slab_count):
            contexts.append(VulkanContextTwoHop.create(
                device_index=index % 2,
                application_name=f"equiv_two_hop_s{index}"))
        link_pool = SharedHostLinkPool(contexts)
        for index in range(slab_count):
            sims.append(SphSimulatorTwoHop(
                contexts[index], chain.slabs[index], link_pool=link_pool,
                slab_index=index, sync_scheme=args.sync_scheme))
        with ChainOrchestratorTwoHop(
                sims, defrag_cadence=defrag_cadence) as orchestrator:
            orchestrator.bootstrap_all()
            orchestrator.run_pipelined(args.steps, depth=args.depth, warmup=0)
            for sim in sims:
                sim.submit_defrag_and_wait()
            error_counts = {
                "stamp_errors_host": sum(
                    worker.stamp_error_count for worker in orchestrator.workers),
                "overwrite_errors_host": sum(
                    worker.overwrite_error_count
                    for worker in orchestrator.workers),
                "stamp_errors_gpu": sum(
                    sim.readback_global_status().get("stamp_error_count", 0)
                    for sim in sims),
            }

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
        if link_pool is not None:
            link_pool.destroy()
        for context in contexts:
            context.destroy()

    mass = np.concatenate(masses)
    velocity = np.concatenate(velocities)
    density = np.concatenate(densities)
    position = np.concatenate(positions)
    statistics_by_metric = {
        "n": int(mass.shape[0]),
        "kinetic_energy": float(0.5 * (mass * (velocity ** 2).sum(1)).sum()),
        "momentum_x": float((mass * velocity[:, 0]).sum()),
        "momentum_y": float((mass * velocity[:, 1]).sum()),
        "mean_density": float(density.mean()),
        "max_speed": float(np.sqrt((velocity ** 2).sum(1)).max()),
        "center_x": float((mass * position[:, 0]).sum() / mass.sum()),
        "center_y": float((mass * position[:, 1]).sum() / mass.sum()),
    }
    return statistics_by_metric, error_counts


def main() -> int:
    args = parse_args()
    environment = pin_campaign_environment()
    print(f"[equiv_two_hop] environment: {environment}")

    from experiment.v5._run_v5_equivalence import (
        ENVELOPE_FACTOR, FLOAT32_NOISE_FLOOR, run_config)
    from experiment.v5.utils.case_loader_v5 import load_case_v5
    global_case = load_case_v5(args.case)

    runs = {}
    for name, slab_count in (("k1_reference", 1), ("k1_rerun", 1),
                             ("k2_three_hop", 2)):
        print(f"\n[equiv_two_hop] === {name} ({args.steps} steps) ===", flush=True)
        runs[name] = run_config(global_case, slab_count, args)
    print(f"\n[equiv_two_hop] === k2_two_hop ({args.steps} steps) ===", flush=True)
    runs["k2_two_hop"], error_counts = run_two_hop_config(global_case, args)

    reference = runs["k1_reference"]
    metrics = [metric for metric in reference if metric != "n"]
    all_ok = True
    table = {}
    print(f"\n{'metric':<16} {'reference':>14} {'|K1-K1_rerun|':>14} "
          f"{'|K1-K2_three|':>14} {'envelope':>14} {'|K1-K2_two|':>14} "
          f"{'|K2three-K2two|':>16}   verdict")
    for metric in metrics:
        rerun_delta = abs(reference[metric] - runs["k1_rerun"][metric])
        three_hop_delta = abs(reference[metric] - runs["k2_three_hop"][metric])
        scale = max(abs(reference[metric]), 1.0)   # SI characteristic floor
        envelope = max(rerun_delta, three_hop_delta, FLOAT32_NOISE_FLOOR * scale)
        two_hop_delta = abs(reference[metric] - runs["k2_two_hop"][metric])
        between_transports = abs(
            runs["k2_three_hop"][metric] - runs["k2_two_hop"][metric])
        ok = two_hop_delta <= ENVELOPE_FACTOR * envelope
        all_ok &= ok
        table[metric] = {
            "reference": reference[metric],
            "rerun_delta": rerun_delta,
            "three_hop_delta": three_hop_delta,
            "envelope": envelope,
            "two_hop_delta": two_hop_delta,
            "three_hop_vs_two_hop": between_transports,
            "pass": ok,
        }
        print(f"{metric:<16} {reference[metric]:>14.6e} {rerun_delta:>14.6e} "
              f"{three_hop_delta:>14.6e} {envelope:>14.6e} "
              f"{two_hop_delta:>14.6e} {between_transports:>16.6e}   "
              f"{'PASS' if ok else 'FAIL'}")

    counts = {name: run["n"] for name, run in runs.items()}
    count_ok = len(set(counts.values())) == 1
    errors_ok = all(value == 0 for value in error_counts.values())
    all_ok &= count_ok and errors_ok
    print(f"{'n (exact)':<16} {counts}   {'PASS' if count_ok else 'FAIL'}")
    print(f"{'stamp checks':<16} {error_counts}   "
          f"{'PASS' if errors_ok else 'FAIL'}")
    print(f"\n[equiv_two_hop] {'ALL PASS' if all_ok else '*** FAIL ***'}")

    if args.output:
        output_path = pathlib.Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({
            "case": args.case, "steps": args.steps, "depth": args.depth,
            "environment": environment, "envelope_factor": ENVELOPE_FACTOR,
            "runs": runs, "metrics": table, "particle_counts": counts,
            "error_counts": error_counts, "all_pass": bool(all_ok),
        }, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
