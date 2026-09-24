"""
_run_hdx_sweep_3d.py — one full V0 run of a 3-D cavity case to a fixed step
count (= fixed physical time T), with periodic health readbacks and a final
full-field snapshot. Used by the h/dx sweep (E6); V0 production code is not
modified — every hook lives here.

Per --record-every steps (and at the last step) the script reads back V0's
global status (alive, overflow counts, correction_fallback_count, max|v|)
plus the particle buffers, and records for the FLUID particles: kinetic
energy, total momentum, mean density, density std and max |rho - rho0|,
max speed; plus wall-clock of the block (sim time only, readbacks excluded).

At the end it saves snapshot_final.npz in the V5 verifier's layout
(s0_position / s0_velocity / s0_density / s0_pressure / s0_material, plus
s0_kernel_sum = density_gradient_kernel_sum.w, h, dx, material kinds) and
prints one ``RESULT {json}`` line.

Usage:
    .venv/Scripts/python.exe _run_hdx_sweep_3d.py --case cases/hdx_sweep_3d/hdx4.0/case.yaml
        --steps 20000 --device 2 --out-dir docs/hdx_sweep_3d_20260924/hdx4.0 --tag hdx4.0
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from _run_single_baseline_bench import (V0Adapter, install_obj_cache,  # noqa: E402
                                        resolve_device_index_in_subprocess)


def read_array(sim, name: str, dtype, columns: int | None):
    raw = sim._readback_buffer(sim.buffers[name])
    array = np.frombuffer(raw, dtype=dtype)
    return array.reshape(-1, columns) if columns else array


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--gpu-uuid", type=str, default=None)
    parser.add_argument("--record-every", type=int, default=1000)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", default="")
    parser.add_argument("--snapshot", default=None,
                        help="snapshot path (default <out-dir>/snapshot_final.npz)")
    parser.add_argument("--obj-cache", default="logs/_obj_npy_cache")
    args = parser.parse_args()

    install_obj_cache(pathlib.Path(args.obj_cache))
    if args.gpu_uuid:
        device_index = resolve_device_index_in_subprocess(args.gpu_uuid, "v0")
    elif args.device is not None:
        device_index = args.device
    else:
        raise SystemExit("pass --gpu-uuid or --device")

    from utils.sph.case import _KIND_NAME_TO_CODE
    from vulkan import vkGetPhysicalDeviceProperties

    out_dir = _REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = pathlib.Path(args.snapshot) if args.snapshot else out_dir / "snapshot_final.npz"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    adapter = V0Adapter(args.case, device_index)
    sim, case = adapter.sim, adapter.case
    gpu_name = str(vkGetPhysicalDeviceProperties(adapter.physical_device).deviceName)
    fluid_code = _KIND_NAME_TO_CODE["fluid"]
    group_count = len(case.materials)
    kind_of_group = np.zeros(group_count, dtype=np.int64)
    rest_density_of_group = np.zeros(group_count, dtype=np.float64)
    for material in case.materials:
        kind_of_group[material.group_id] = int(material.kind)
        rest_density_of_group[material.group_id] = float(material.rest_density)
    timestep = float(case.timestep)
    smoothing_length = float(case.physics.h)
    dx = 2.0 * float(case.physics.particle_radius)
    print(f"[hdx_sweep] tag={args.tag} gpu={gpu_name} particles={adapter.expected_particle_count:,} "
          f"h={smoothing_length:g} dx={dx:g} h/dx={smoothing_length / dx:.3f} dt={timestep:.3e} "
          f"steps={args.steps} T={args.steps * timestep:.4f}s", flush=True)

    records_path = out_dir / "run.jsonl"
    records_file = open(records_path, "w")
    wall_sim = 0.0
    wall_readback = 0.0
    block_wall = 0.0
    t_run_start = time.time()

    def record(frame: int) -> dict:
        nonlocal wall_readback, block_wall
        t0 = time.perf_counter()
        status = sim.readback_global_status()
        position = read_array(sim, "position_voxel_id", np.float32, 4)
        velocity_mass = read_array(sim, "velocity_mass", np.float32, 4)
        density_pressure = read_array(sim, "density_pressure", np.float32, 2)
        material = read_array(sim, "material", np.uint32, None)[: position.shape[0]]
        has_mass = velocity_mass[:, 3] > 0
        alive = has_mass & (position[:, 3] > 0.5)
        # V0 kills a particle by writing voxel_id 0 into position.w (mass stays);
        # the status alive counter is not decremented, so count kills here.
        dead_count = int((has_mass & ~alive).sum())
        fluid = alive & (kind_of_group[np.minimum(material, group_count - 1)] == fluid_code)
        mass = velocity_mass[fluid, 3].astype(np.float64)
        velocity = velocity_mass[fluid, 0:3].astype(np.float64)
        density = density_pressure[fluid, 0].astype(np.float64)
        rest_density = rest_density_of_group[np.minimum(material[fluid], group_count - 1)]
        speed = np.sqrt((velocity ** 2).sum(axis=1))
        rel = density / rest_density - 1.0
        entry = {
            "step": frame, "time": frame * timestep,
            "alive": int(status["alive_particle_count"]),
            "drift": int(status["alive_particle_count"]) - adapter.expected_particle_count,
            "overflow_inside_count": int(status["overflow_inside_count"]),
            "overflow_incoming_count": int(status["overflow_incoming_count"]),
            "first_overflow_voxel_inside": int(status["first_overflow_voxel_inside"]),
            "first_overflow_voxel_incoming": int(status["first_overflow_voxel_incoming"]),
            "correction_fallback_count": int(status["correction_fallback_count"]),
            "status_maximum_velocity": float(status["maximum_velocity"]),
            "alive_from_buffers": int(alive.sum()),
            "dead_count": dead_count,
            "fluid_count": int(fluid.sum()),
            "kinetic_energy": float(0.5 * (mass * speed ** 2).sum()),
            "momentum": [float(x) for x in (mass[:, None] * velocity).sum(axis=0)],
            "mean_density": float(density.mean()),
            "density_std_rel": float(rel.std()),
            "max_abs_density_dev_rel": float(np.abs(rel).max()),
            "max_speed": float(speed.max()),
            "block_wall_seconds": block_wall,
            "block_us_per_step": block_wall / args.record_every * 1e6,
            "wall_sim_seconds_cumulative": wall_sim,
        }
        wall_readback += time.perf_counter() - t0
        block_wall = 0.0
        records_file.write(json.dumps(entry) + "\n")
        records_file.flush()
        print(f"[hdx_sweep] step {frame:6d} t={entry['time']:.4f}s  {entry['block_us_per_step']:8.1f} us/step  "
              f"alive={entry['alive']:,} drift={entry['drift']:+d} dead={dead_count} "
              f"ovf={entry['overflow_inside_count']}/{entry['overflow_incoming_count']} "
              f"fallback={entry['correction_fallback_count']} "
              f"KE={entry['kinetic_energy']:.5f} rho_std={entry['density_std_rel']:.2e} "
              f"max|v|={entry['max_speed']:.3f}", flush=True)
        return entry

    last = None
    try:
        for frame in range(1, args.steps + 1):
            t0 = time.perf_counter()
            sim.step()                       # sync submit + fence wait; defrag at cadence
            elapsed = time.perf_counter() - t0
            wall_sim += elapsed
            block_wall += elapsed
            if frame % args.record_every == 0 or frame == args.steps:
                last = record(frame)

        # Final full-field snapshot (verifier layout).
        t0 = time.perf_counter()
        position = read_array(sim, "position_voxel_id", np.float32, 4)
        velocity_mass = read_array(sim, "velocity_mass", np.float32, 4)
        density_pressure = read_array(sim, "density_pressure", np.float32, 2)
        kernel_sum = read_array(sim, "density_gradient_kernel_sum", np.float32, 4)
        material = read_array(sim, "material", np.uint32, None)[: position.shape[0]]
        alive = (velocity_mass[:, 3] > 0) & (position[:, 3] > 0.5)
        np.savez_compressed(
            snapshot_path,
            s0_position=position[alive, 0:3], s0_velocity=velocity_mass[alive, 0:3],
            s0_mass=velocity_mass[alive, 3], s0_density=density_pressure[alive, 0],
            s0_pressure=density_pressure[alive, 1], s0_material=material[alive],
            s0_kernel_sum=kernel_sum[alive, 3], s0_voxel_id=position[alive, 3],
            h=np.float64(smoothing_length), dx=np.float64(dx),
            hdx=np.float64(smoothing_length / dx), step=np.int64(args.steps),
            time=np.float64(args.steps * timestep),
            material_kinds=kind_of_group, material_rest_density=rest_density_of_group,
            material_names=np.array([m.name for m in case.materials]),
            fluid_kind_code=np.int64(fluid_code),
            speed_of_sound=np.float64(case.physics.speed_of_sound),
        )
        wall_readback += time.perf_counter() - t0

        result = {
            "tag": args.tag, "case": args.case, "gpu": gpu_name, "device": device_index,
            "particles": adapter.expected_particle_count, "h": smoothing_length, "dx": dx,
            "hdx": smoothing_length / dx, "timestep": timestep, "steps": args.steps,
            "physical_time": args.steps * timestep,
            "wall_sim_seconds": round(wall_sim, 3),
            "us_per_step": round(wall_sim / args.steps * 1e6, 2),
            "fps": round(args.steps / wall_sim, 3),
            "wall_readback_seconds": round(wall_readback, 2),
            "wall_total_seconds": round(time.time() - t_run_start, 1),
            "snapshot": str(snapshot_path),
            "final": last,
            "status": ("ok" if last and last["drift"] == 0 and last["dead_count"] == 0
                       and last["overflow_inside_count"] == 0
                       and last["overflow_incoming_count"] == 0 else "bad_state"),
        }
    finally:
        records_file.close()
        adapter.destroy()

    (out_dir / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("RESULT " + json.dumps(result), flush=True)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
