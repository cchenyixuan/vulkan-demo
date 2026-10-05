"""
pool_peaks.py — measured ghost / departed pool demand for the v6 pool sizing (opt item (e)).

Each job runs one K = 2 chain (local 2 x RTX 5090, production switches + the (1,2) seam
configuration + lean transport) with V6_POOL_PEAKS=1: every frame, the transport worker
reads the sender's allocation counter of each pool region from the sender staging (the
inner / outer replica regions and the migrant region of V6_GHOST_LAYERS=2, or the V5
mixed pool). The counter is the DEMAND of that frame: ghost_send adds every column's
count before its overflow test, so a frame that would overflow is still recorded. The
departed pool's demand is PoolHealth.peak_departed_count (atomicMax per frame, never
reset). The measurement factor is generous (V6_GHOST_POOL_FACTOR=1.0 unless the job
says otherwise) so no region overflows and the flow is never perturbed.

Developed flow: the 2-D 1M / 2M jobs restart from checkpoints of the cavity validation
campaign (logs/cavity_validation, Re 1000 at t = 52.65 s and Re 3200 at t = 99.45 s (1M) / 99.8 s (2M)):
positions, velocities and material groups of every particle replace the case's initial
condition, then the normal bootstrap runs (densities restart at rho0, pressure rebuilds
within a few hundred steps). The other cases start from their initial condition.

Per job the result line holds, per link and region: capacity (slots at the run's factor
and at factor 1), per-frame demand max / p99.9 / p99 / mean / min and the frame of the
max, the required factor (max / capacity at factor 1); per sim the departed peak and
capacity; drift / overflow / far-migration invariants. Per-frame series go to an .npz.

Usage:
    .venv/Scripts/python.exe -m experiment.seam_audit.pool_peaks --out logs/seam_audit/opt/pool_peaks
    .venv/Scripts/python.exe -m experiment.seam_audit.pool_peaks --out ... --jobs 2d_1m_init,3d_8m_init
    .venv/Scripts/python.exe -m experiment.seam_audit.pool_peaks --out ... --summarize-only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.seam_audit.opt_campaign import (  # noqa: E402
    CASES, RESULT_PREFIX, SEAM_L2, production_environment)

_VALIDATION = "logs/cavity_validation"
# name: (case yaml, dimension, checkpoint npz or None, frames)
JOBS = {
    "2d_narrow_init": (CASES["2d_narrow"][0], 2, None, 100000),
    "2d_1m_init": (CASES["2d_1m"][0], 2, None, 20000),
    "2d_1m_re1000_t53": (CASES["2d_1m"][0], 2,
                         f"{_VALIDATION}/campaign_20260929/re1000_1m_k2/checkpoint_007020000.npz", 20000),
    "2d_1m_re3200_t99": ("cases/cavity_validation/re3200_1m/case.yaml", 2,
                         f"{_VALIDATION}/campaign_20260929/re3200_1m_k1/checkpoint_013260000.npz", 20000),
    "2d_2m_re3200_t100": ("cases/cavity_validation/kcgoff_re3200_2m/case.yaml", 2,
                          f"{_VALIDATION}/campaign_20260930_kcgoff/re3200_2m_k1/checkpoint_018810000.npz", 10000),
    "2d_4m_init": (CASES["2d_4m"][0], 2, None, 10000),
    "2d_16m_init": (CASES["2d_16m"][0], 2, None, 4000),
    "3d_8m_init": (CASES["3d_8m"][0], 3, None, 3000),
    "3d_narrow_init": (CASES["3d_narrow"][0], 3, None, 3000),
}


def _percentile(values: np.ndarray, fraction: float) -> float:
    return float(np.quantile(values, fraction)) if values.size else 0.0


# --------------------------------------------------------------------------- worker
def run_worker(args) -> int:
    from experiment.v6.utils.case_loader_v6 import load_case_v6
    from experiment.v6.utils.case_v6 import InitialParticles
    from experiment.v6.utils.orchestrator_v6 import ChainOrchestratorV6
    from experiment.v6.utils.partition_v6 import compute_chain_partition
    from experiment.v6.utils.simulator_v6 import SphSimulatorV6
    from experiment.v6.utils.vulkan_context_v6 import VulkanContextV6

    global_case = load_case_v6(args.case)
    result = {"job": args.job, "case": args.case, "checkpoint": args.checkpoint, "frames": args.frames,
              "switches": {key: value for key, value in sorted(os.environ.items())
                           if key.startswith("V6_")}}
    if args.checkpoint:
        saved = np.load(args.checkpoint)
        count = int(saved["position"].shape[0])
        if count != global_case.initial.positions.shape[0]:
            raise SystemExit(f"checkpoint has {count:,} particles, case {global_case.initial.positions.shape[0]:,}")
        positions = np.zeros((count, 3), dtype=np.float32)
        positions[:, :saved["position"].shape[1]] = saved["position"]
        velocities = np.zeros((count, 3), dtype=np.float32)
        velocities[:, :saved["velocity"].shape[1]] = saved["velocity"]
        global_case.initial = InitialParticles(
            positions=positions, velocities=velocities,
            material_group=saved["material"].astype(np.uint32))
        result["checkpoint_time"] = float(saved["time"])
        result["checkpoint_step"] = int(saved["step"])
    expected_total = int(global_case.initial.positions.shape[0])
    chain = compute_chain_partition(global_case, [1.0, 1.0], 1.2)
    result["cuts"] = list(chain.cuts)
    device_map = [int(device) for device in args.device_map.split(",")]
    contexts, sims = [], []
    try:
        for index in range(2):
            contexts.append(VulkanContextV6.create(device_index=device_map[index],
                                                   application_name=f"pool_peaks_s{index}"))
            sims.append(SphSimulatorV6(contexts[-1], chain.slabs[index], sync_scheme="per-direction"))
        with ChainOrchestratorV6(sims, defrag_cadence=global_case.numerics.defrag_cadence) as orchestrator:
            orchestrator.bootstrap_all()
            workers = list(orchestrator.workers)
            started = time.perf_counter()
            pipelined = orchestrator.run_pipelined(args.frames, depth=2, warmup=0)
            result["fps"] = pipelined.get("steady_fps", pipelined["fps"])
            result["wall_s"] = time.perf_counter() - started
            for sim in sims:
                sim.submit_defrag_and_wait()

            from experiment.v6.utils.partition_v6 import configured_ghost_pool_factor
            factor = configured_ghost_pool_factor(global_case)
            series = {}
            links = {}
            for worker in workers:
                regions = {}
                case = worker.source.case
                face = case.grid.grid_dimension_y * case.grid.grid_dimension_z
                unit = {  # slots of each region at factor 1 (partition_v6._ghost_pool_layout)
                    "mixed": face * (case.capacities.max_particles_per_voxel
                                     + case.capacities.max_incoming_per_voxel),
                    "inner": face * (case.capacities.max_particles_per_voxel
                                     + case.capacities.max_incoming_per_voxel),
                    "outer": face * (case.capacities.max_particles_per_voxel
                                     + case.capacities.max_incoming_per_voxel),
                    "migrant": face * case.capacities.max_incoming_per_voxel,
                }
                for region, counts in worker.region_counts.items():
                    values = np.asarray(counts, dtype=np.int64)
                    series[f"{worker.label}__{region}"] = values.astype(np.int32)
                    capacity = worker.region_capacity[region]
                    peak = int(values.max()) if values.size else 0
                    regions[region] = {
                        "capacity": capacity, "capacity_factor_1": unit[region],
                        "frames": int(values.size), "max": peak,
                        "max_frame": int(values.argmax()) if values.size else None,
                        "p999": _percentile(values, 0.999), "p99": _percentile(values, 0.99),
                        "mean": float(values.mean()) if values.size else 0.0,
                        "min": int(values.min()) if values.size else 0,
                        "occupancy": peak / capacity if capacity else None,
                        "required_factor": peak / unit[region] if unit[region] else None,
                        # the steady level of the last 10 % vs the first 10 %
                        "mean_first_tenth": float(values[: max(1, values.size // 10)].mean()) if values.size else 0.0,
                        "mean_last_tenth": float(values[-max(1, values.size // 10):].mean()) if values.size else 0.0,
                    }
                links[worker.label] = {"face_voxels": face, "regions": regions}
            result["links"] = links
            result["factor"] = factor

            statuses, healths, overflow = [], [], {}
            alive_total = 0
            for sim in sims:
                status = sim.readback_global_status()
                health = sim.readback_pool_health()
                statuses.append(dict(status))
                healths.append(dict(health))
                alive_total += status["alive_particle_count"]
                for name, value in status.items():
                    if name.startswith("overflow_"):
                        overflow[name] = overflow.get(name, 0) + value
            result["departed"] = [{
                "peak": health["peak_departed_count"], "capacity": health["departed_pool_size"],
                "face_voxels": sim.case.grid.grid_dimension_y * sim.case.grid.grid_dimension_z,
                "peer_sides": int(sim.case.transport.has_leading_peer) + int(sim.case.transport.has_trailing_peer),
            } for sim, health in zip(sims, healths)]
            result["pool_health"] = healths
            stamp_errors = sum(worker.stamp_error_count for worker in workers) + sum(
                status.get("stamp_error_count", 0) for status in statuses)
            far_migration = sum(status.get("far_migration_count", 0) for status in statuses)
            result["invariants"] = {
                "drift": alive_total - expected_total, "overflow": overflow,
                "stamp_errors": stamp_errors, "far_migration": far_migration,
                "valid": (alive_total == expected_total and not any(overflow.values())
                          and stamp_errors == 0 and far_migration == 0)}
            np.savez_compressed(pathlib.Path(args.series_out), **series)
    finally:
        for sim in sims:
            sim.destroy()
        for context in contexts:
            context.destroy()
    print(RESULT_PREFIX + json.dumps(result), flush=True)
    return 0 if result["invariants"]["valid"] else 3


# --------------------------------------------------------------------------- driver
def run_driver(args) -> int:
    out_dir = pathlib.Path(args.out).resolve()
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    done = set()
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("ok"):
                done.add(record["job"])
    names = args.jobs.split(",") if args.jobs else list(JOBS)
    for name in names:
        if name in done:
            continue
        case_path, dimension, checkpoint, frames = JOBS[name]
        environment = {key: value for key, value in os.environ.items() if not key.startswith("V6_")}
        environment.update(production_environment(dimension))
        environment.update(SEAM_L2)
        environment.update({"V6_LEAN_TRANSPORT": "1", "V6_POOL_PEAKS": "1",
                            "V6_GHOST_POOL_FACTOR": args.factor,
                            # restarts from arbitrary states: a particle within float32 rounding of
                            # the cut is otherwise lost at bootstrap (a1a1838; 2d_1m_re3200 lost one)
                            "V6_INIT_SEAM_CLAMP": "1"})
        for item in args.env:
            key, _, value = item.partition("=")
            environment[key] = value
        command = [sys.executable, "-m", "experiment.seam_audit.pool_peaks", "--worker",
                   "--job", name, "--case", case_path, "--frames", str(frames),
                   "--device-map", args.device_map,
                   "--series-out", str(out_dir / f"{name}_series.npz")]
        if checkpoint:
            command += ["--checkpoint", checkpoint]
        print(f"[pool_peaks] {name} ...", flush=True)
        started = time.time()
        try:
            completed = subprocess.run(command, env=environment, capture_output=True, text=True,
                                       timeout=args.timeout, cwd=_REPO_ROOT)
            output = completed.stdout + "\n" + completed.stderr
            return_code = completed.returncode
        except subprocess.TimeoutExpired as error:
            output = f"TIMEOUT after {args.timeout}s\n{error.stdout or ''}\n{error.stderr or ''}"
            return_code = -1
        (out_dir / "logs" / f"{name}.log").write_text(str(output), encoding="utf-8")
        lines = [line for line in output.splitlines() if line.startswith(RESULT_PREFIX)]
        record = {"job": name, "return_code": return_code, "wall_s": round(time.time() - started, 1),
                  "ok": bool(lines)}
        if lines:
            record["result"] = json.loads(lines[-1][len(RESULT_PREFIX):])
            print(f"[pool_peaks]   valid={record['result']['invariants']['valid']} "
                  f"({record['wall_s']} s)", flush=True)
        else:
            print(f"[pool_peaks]   FAILED rc={return_code}", flush=True)
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    summarize(out_dir)
    return 0


def summarize(out_dir: pathlib.Path) -> None:
    records = [json.loads(line) for line in (out_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    latest = {}
    for record in records:
        if record.get("ok"):
            latest[record["job"]] = record["result"]
    lines = ["# v6 pool demand (K = 2, (1,2) + lean, measurement factor generous, per frame)", "",
             "Region demand = the sender's allocation counter of the frame (replicas of the inner / outer",
             "ghost layer, migrants). required f = max / (slots at factor 1). Departed: PoolHealth peak.", "",
             "| job | frames | link | region | slots at f=1 | max | frame of max | p99.9 | mean (first / last 10 %) | "
             "required f | valid |",
             "|" + "---|" * 11]
    for name in [job for job in JOBS if job in latest]:
        result = latest[name]
        for link, entry in result["links"].items():
            for region, stats in entry["regions"].items():
                lines.append(
                    f"| {name} | {result['frames']} | {link} | {region} | {stats['capacity_factor_1']:,} | "
                    f"{stats['max']:,} | {stats['max_frame']} | {stats['p999']:,.0f} | "
                    f"{stats['mean']:,.0f} ({stats['mean_first_tenth']:,.0f} / {stats['mean_last_tenth']:,.0f}) | "
                    f"{stats['required_factor']:.4f} | {'yes' if result['invariants']['valid'] else '**NO**'} |")
    lines += ["", "| job | sim | departed peak | capacity | face voxels | peer sides | peak / face |",
              "|---|---|---|---|---|---|---|"]
    for name in [job for job in JOBS if job in latest]:
        for index, entry in enumerate(latest[name]["departed"]):
            lines.append(f"| {name} | s{index} | {entry['peak']} | {entry['capacity']} | {entry['face_voxels']:,} | "
                         f"{entry['peer_sides']} | {entry['peak'] / max(1, entry['face_voxels']):.4f} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="v6 pool demand measurement")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--job", default=None)
    parser.add_argument("--case", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--frames", type=int, default=10000)
    parser.add_argument("--series-out", default=None)
    parser.add_argument("--device-map", default="0,1")
    parser.add_argument("--out", default="logs/seam_audit/opt/pool_peaks")
    parser.add_argument("--jobs", default=None)
    parser.add_argument("--factor", default="1.0")
    parser.add_argument("--env", action="append", default=[], help="extra KEY=VALUE")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.worker:
        return run_worker(args)
    if args.summarize_only:
        summarize(pathlib.Path(args.out).resolve())
        return 0
    return run_driver(args)


if __name__ == "__main__":
    sys.exit(main())
