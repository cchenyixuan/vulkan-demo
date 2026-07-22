"""
_run_v5_3d_longrun.py — dual-GPU 3D long run with screenshots + checkpoints.

Mission (user, 2026-07-23): develop the 3D lid-driven cavity flow for
10-30 s of simulated physics; a slice screenshot every N steps and a
particle checkpoint every ~0.2 s of simulation time for later offline
visualization.

Readback safety: everything happens inside run_pipelined's ``on_defrag``
callback — the only point where the pipelined chain is fully drained.
Both screenshot and checkpoint cadences must therefore be multiples of
the defrag cadence (asserted at startup); the manifest records the EXACT
sim time of each artifact, so a 7000-step (0.21 s) checkpoint grid is as
good as an exact 0.2 s one for later playback.

Checkpoint format (fluid particles only — walls/lid are static and saved
ONCE to walls_static.npz):
    ckpt_step0007000.npz:
        positions  float32 (n_fluid, 3)   world coordinates
        velocities float16 (n_fluid, 3)   viz-grade precision
        density    float16 (n_fluid,)
    manifest.jsonl — one JSON per checkpoint: step, sim_time_s, n_fluid,
        alive_total, drift, vmax, rho range, free_gb, wall-clock stamp.

Disk guard (C: was at 47.5 GB free when this was written): if free space
drops below --degrade-free-gb the checkpoint interval doubles (screenshots
continue); below --stop-free-gb checkpointing stops entirely. The run
itself never stops for disk reasons.

I/O runs on a single worker thread (queue depth 2 = bounded RAM); the GPU
pipeline stalls only for the readback itself (~2-3 s per event).

    .venv/Scripts/python.exe experiment/v5/_run_v5_3d_longrun.py \
        --case cases/cavity3d_8m/case.yaml --weights 1.05,1 \
        --target-sim-seconds 30 --out logs/cavity3d_longrun_20260723
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import queue
import shutil
import sys
import threading
import time

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.v5._run_v5_3d_slice import render_snapshot  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="V5 3D dual long run")
    parser.add_argument("--case", default="cases/cavity3d_8m/case.yaml")
    parser.add_argument("--weights", default="1.05,1")
    parser.add_argument("--device-map", default="0,1")
    parser.add_argument("--sync-scheme", default="per-direction")
    parser.add_argument("--pool-safety", type=float, default=1.2)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--target-sim-seconds", type=float, default=30.0)
    parser.add_argument("--checkpoint-sim-interval", type=float, default=0.2)
    parser.add_argument("--screenshot-steps", type=int, default=10000)
    parser.add_argument("--degrade-free-gb", type=float, default=25.0)
    parser.add_argument("--stop-free-gb", type=float, default=15.0)
    parser.add_argument("--out", default="logs/cavity3d_longrun_20260723")
    return parser.parse_args()


def free_gb(path: pathlib.Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def main() -> int:
    args = parse_args()

    from experiment.v5.utils.case_loader_v5 import load_case_v5
    from experiment.v5.utils.orchestrator_v5 import ChainOrchestratorV5
    from experiment.v5.utils.partition_v5 import compute_chain_partition
    from experiment.v5.utils.simulator_v5 import SphSimulatorV5
    from experiment.v5.utils.vulkan_context_v5 import VulkanContextV5

    out_dir = _REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    weights = [float(w) for w in args.weights.split(",")]
    device_map = [int(d) for d in args.device_map.split(",")]
    assert len(device_map) == len(weights)

    global_case = load_case_v5(args.case)
    expected_total = int(global_case.initial.positions.shape[0])
    timestep = global_case.physics.timestep
    defrag_cadence = global_case.numerics.defrag_cadence

    max_steps = int(round(args.target_sim_seconds / timestep))
    max_steps = ((max_steps + defrag_cadence - 1) // defrag_cadence) * defrag_cadence
    checkpoint_steps = max(
        defrag_cadence,
        int(round(args.checkpoint_sim_interval / timestep / defrag_cadence))
        * defrag_cadence)
    screenshot_steps = ((args.screenshot_steps + defrag_cadence - 1)
                        // defrag_cadence) * defrag_cadence

    print(f"[longrun] dt={timestep:.3e}s  target={args.target_sim_seconds}s "
          f"-> {max_steps:,} steps")
    print(f"[longrun] checkpoint every {checkpoint_steps:,} steps "
          f"({checkpoint_steps * timestep:.3f}s sim), screenshot every "
          f"{screenshot_steps:,} steps; defrag cadence {defrag_cadence}")
    print(f"[longrun] disk: {free_gb(out_dir):.1f} GB free; degrade "
          f"<{args.degrade_free_gb} GB, stop <{args.stop_free_gb} GB")

    chain = compute_chain_partition(global_case, weights, args.pool_safety)

    # ---- I/O worker: renders + npz writes off the GPU-driving thread ----
    jobs: queue.Queue = queue.Queue(maxsize=2)
    state = {"walls_saved": False, "ckpt_interval": checkpoint_steps,
             "ckpt_stopped": False, "errors": 0}

    def worker_loop():
        while True:
            job = jobs.get()
            if job is None:
                return
            try:
                _process_job(job)
            except Exception as error:                     # noqa: BLE001
                state["errors"] += 1
                print(f"[longrun] IO-WORKER ERROR ({job['kind']} "
                      f"step {job['step']}): {error!r}", flush=True)
            finally:
                jobs.task_done()

    def _process_job(job):
        positions = job["positions"]
        velocities = job["velocities"]
        if job["kind"] == "screenshot":
            png = out_dir / f"slice_step{job['step']:07d}.png"
            render_snapshot(png, job["step"], positions, velocities,
                            dx=global_case.physics.smoothing_length / 4.0)
            print(f"[longrun] wrote {png.name}", flush=True)
            return
        # checkpoint
        fluid = job["material_group"] == 0
        if not state["walls_saved"]:
            state["walls_saved"] = True
            np.savez(out_dir / "walls_static.npz",
                     positions=positions[~fluid].astype(np.float32),
                     material_group=job["material_group"][~fluid].astype(np.uint8))
        fluid_positions = positions[fluid]
        fluid_velocities = velocities[fluid]
        fluid_density = job["density"][fluid]
        ckpt = out_dir / f"ckpt_step{job['step']:07d}.npz"
        np.savez(ckpt,
                 positions=fluid_positions.astype(np.float32),
                 velocities=fluid_velocities.astype(np.float16),
                 density=fluid_density.astype(np.float16))
        speed = np.sqrt((fluid_velocities ** 2).sum(axis=1))
        record = {
            "step": job["step"],
            "sim_time_s": round(job["step"] * timestep, 6),
            "file": ckpt.name,
            "n_fluid": int(fluid.sum()),
            "alive_total": job["alive_total"],
            "drift": job["alive_total"] - expected_total,
            "vmax": round(float(speed.max()), 5),
            "rho_min": round(float(fluid_density.min()), 2),
            "rho_max": round(float(fluid_density.max()), 2),
            "free_gb": round(free_gb(out_dir), 1),
            "wall_clock": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        with open(manifest_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"[longrun] ckpt t={record['sim_time_s']:.2f}s "
              f"drift={record['drift']} vmax={record['vmax']} "
              f"rho[{record['rho_min']},{record['rho_max']}] "
              f"free={record['free_gb']}GB", flush=True)
        if record["drift"] != 0:
            print(f"[longrun] *** DRIFT at step {job['step']}: "
                  f"{record['drift']} ***", flush=True)

    worker = threading.Thread(target=worker_loop, name="longrun_io", daemon=True)
    worker.start()

    contexts, sims = [], []
    rc = 0
    try:
        for index in range(len(weights)):
            ctx = VulkanContextV5.create(
                device_index=device_map[index], enable_validation=False,
                application_name=f"longrun3d_s{index}")
            contexts.append(ctx)
            sims.append(SphSimulatorV5(ctx, chain.slabs[index],
                                       sync_scheme=args.sync_scheme))

        readback_names = ["position_voxel_id", "velocity_mass",
                          "density_pressure", "material"]

        def merged_readback():
            merged = {"positions": [], "velocities": [], "density": [],
                      "material_group": []}
            alive_total = 0
            for sim in sims:
                capacities = sim.case.capacities
                pool = capacities.total_pool_capacity()
                raw = sim.readback_buffers_batch(readback_names)
                position_voxel = np.frombuffer(
                    raw["position_voxel_id"], np.float32).reshape(pool, 4)
                velocity_mass = np.frombuffer(
                    raw["velocity_mass"], np.float32).reshape(pool, 4)
                density_pressure = np.frombuffer(
                    raw["density_pressure"], np.float32).reshape(pool, 2)
                material_group = np.frombuffer(raw["material"], np.uint32)
                own = slice(sim.own_first_pid(),
                            sim.own_first_pid() + capacities.own_pool_size)
                alive = velocity_mass[own, 3] > 0
                alive_total += int(alive.sum())
                merged["positions"].append(position_voxel[own, 0:3][alive].copy())
                merged["velocities"].append(velocity_mass[own, 0:3][alive].copy())
                merged["density"].append(density_pressure[own, 0][alive].copy())
                merged["material_group"].append(material_group[own][alive].copy())
            return ({key: np.concatenate(value) for key, value in merged.items()},
                    alive_total)

        run_start = time.perf_counter()

        def on_defrag(step: int, report: list) -> None:
            want_screenshot = step % screenshot_steps == 0
            want_checkpoint = (not state["ckpt_stopped"]
                               and step % state["ckpt_interval"] == 0)
            drops = sum(r["overflow_install_tail"] for r in report)
            if drops:
                print(f"[longrun] *** frame {step}: install drops={drops} ***",
                      flush=True)
            if not (want_screenshot or want_checkpoint):
                return

            # Disk guard (checked before adding new data).
            available = free_gb(out_dir)
            if want_checkpoint and available < args.stop_free_gb:
                state["ckpt_stopped"] = True
                want_checkpoint = False
                print(f"[longrun] disk {available:.1f} GB < stop threshold — "
                      f"CHECKPOINTING STOPPED (screenshots continue)", flush=True)
            elif want_checkpoint and available < args.degrade_free_gb:
                state["ckpt_interval"] *= 2
                print(f"[longrun] disk {available:.1f} GB low — checkpoint "
                      f"interval doubled to {state['ckpt_interval']:,} steps",
                      flush=True)

            merged, alive_total = merged_readback()
            if want_checkpoint:
                jobs.put({"kind": "checkpoint", "step": step,
                          "alive_total": alive_total, **merged})
            if want_screenshot:
                jobs.put({"kind": "screenshot", "step": step,
                          "positions": merged["positions"],
                          "velocities": merged["velocities"]})

            elapsed = time.perf_counter() - run_start
            fps = step / elapsed if elapsed > 0 else 0.0
            eta_h = (max_steps - step) / fps / 3600 if fps > 0 else 0.0
            print(f"[longrun] step {step:,}/{max_steps:,} "
                  f"t_sim={step * timestep:.2f}s fps={fps:.1f} "
                  f"eta={eta_h:.1f}h", flush=True)

        with ChainOrchestratorV5(sims, defrag_cadence=defrag_cadence) as orch:
            orch.bootstrap_all()
            result = orch.run_pipelined(max_steps, depth=args.depth,
                                        warmup=0, on_defrag=on_defrag)
            print(f"[longrun] RUN DONE: {result['frame_count']:,} steps in "
                  f"{result['elapsed_s']/3600:.2f}h = {result['fps']:.1f} fps")

            total = 0
            for index, sim in enumerate(sims):
                status = sim.readback_global_status()
                total += status["alive_particle_count"]
            print(f"[longrun] final: total={total:,} (expected "
                  f"{expected_total:,}) drift={total - expected_total}")
            if total != expected_total:
                rc = 1
    except Exception as error:                              # noqa: BLE001
        print(f"[longrun] *** RUN ABORTED: {error!r} ***", flush=True)
        rc = 2
    finally:
        jobs.put(None)
        worker.join(timeout=300)
        for sim in sims:
            sim.destroy()
        for ctx in contexts:
            ctx.destroy()
    if state["errors"]:
        print(f"[longrun] {state['errors']} IO worker errors — check log")
    return rc


if __name__ == "__main__":
    sys.exit(main())
