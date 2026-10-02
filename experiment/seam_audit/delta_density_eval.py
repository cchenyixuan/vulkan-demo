"""
delta_density_eval.py — evaluation of V6_DELTA_DENSITY (opt item 六; report only, default stays off).

Single card (v6 single-buffer path: sim.bootstrap(), one combined step command buffer, fence per
step), 2-D lid-driven cavity 1M (cases/lid_driven_cavity_2d_gen, Re 1000), two variants run on
the two cards at the same time:

  baseline  density_pressure.x = rho (float32; ULP 6.1e-5 kg/m3 at rho0 = 1000)
  delta     V6_DELTA_DENSITY=1: .x = rho - rho_ref, EOS from x = delta / rho0 by its series

Scenarios:
  rest       from the case's initial condition (fluid at rest, lid at U) to --rest-time.
  developed  from the Re 1000 validation checkpoint (t = 52.65 s, K = 2 run): positions,
             velocities and material groups replace the initial condition (densities restart
             at rho0); run --developed-time more.

Per sample (every --sample-every steps): fluid kinetic energy 0.5 sum m |v|^2, fluid pressure
mean / std / extrema, density statistics. At --noise-every steps and at the end: the spatial
pressure noise (rms of P minus its Shepard average over h, fluid particles), the density
quantization measures (one extra step: the fraction of fluid particles whose stored float32
density changed, the median |delta rho| of the changed ones, and the number of distinct pressure
levels among fluid particles with |rho - rho0| < 10 ULP), and at the end the centerline
profiles u(y) at x = 0 and v(x) at y = 0 (Shepard interpolation, 401 points) with their errors
against Ghia et al. 1982 (docs/cavity_validation/reference). Throughput: steps per second of the
stepping loop excluding the sample readbacks.

Usage:
    .venv/Scripts/python.exe -m experiment.seam_audit.delta_density_eval --out logs/seam_audit/opt/delta_density
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

RESULT_PREFIX = "[delta_density] RESULT "
CASE = "cases/lid_driven_cavity_2d_gen/case.yaml"
CHECKPOINT = "logs/cavity_validation/campaign_20260929/re1000_1m_k2/checkpoint_007020000.npz"
REFERENCE_DIR = _REPO_ROOT / "docs" / "cavity_validation" / "reference"
FLUID_HALF = 0.5          # the cavity's fluid spans [-0.5, 0.5] in both axes


def wendland_c4_2d(distance: np.ndarray, smoothing_length: float) -> np.ndarray:
    q = distance / smoothing_length
    weight = np.zeros_like(q)
    inside = q < 1.0
    qi = q[inside]
    weight[inside] = (9.0 / (math.pi * smoothing_length ** 2)) * (1.0 - qi) ** 6 * (1.0 + 6.0 * qi + 35.0 / 3.0 * qi ** 2)
    return weight


def shepard(tree, positions, volumes, values, points, smoothing_length):
    """Shepard-normalised SPH average of `values` (N, k) at `points` (M, 2)."""
    out = np.full((points.shape[0], values.shape[1]), np.nan)
    for index, neighbors in enumerate(tree.query_ball_point(points, smoothing_length)):
        if not neighbors:
            continue
        neighbors = np.asarray(neighbors)
        weight = wendland_c4_2d(np.linalg.norm(positions[neighbors] - points[index], axis=1),
                                smoothing_length) * volumes[neighbors]
        total = weight.sum()
        if total > 0:
            out[index] = (weight[:, None] * values[neighbors]).sum(axis=0) / total
    return out


def load_ghia_1000() -> dict:
    ghia = json.load(open(REFERENCE_DIR / "ghia1982.json"))
    return {"y": np.asarray(ghia["u_vertical_centerline"]["y"])[1:-1],
            "u": np.asarray(ghia["u_vertical_centerline"]["1000"])[1:-1],
            "x": np.asarray(ghia["v_horizontal_centerline"]["x"])[1:-1],
            "v": np.asarray(ghia["v_horizontal_centerline"]["1000"])[1:-1]}


# --------------------------------------------------------------------------- worker
def run_worker(args) -> int:
    from scipy.spatial import cKDTree

    from experiment.v6.utils.case_loader_v6 import load_case_v6
    from experiment.v6.utils.case_v6 import InitialParticles, KIND_FLUID
    from experiment.v6.utils.simulator_v6 import SphSimulatorV6
    from experiment.v6.utils.vulkan_context_v6 import VulkanContextV6

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case = load_case_v6(CASE)
    time_offset = 0.0
    if args.scenario == "developed":
        saved = np.load(_REPO_ROOT / CHECKPOINT)
        count = int(saved["position"].shape[0])
        positions = np.zeros((count, 3), dtype=np.float32)
        positions[:, :2] = saved["position"]
        velocities = np.zeros((count, 3), dtype=np.float32)
        velocities[:, :2] = saved["velocity"]
        case.initial = InitialParticles(positions=positions, velocities=velocities,
                                        material_group=saved["material"].astype(np.uint32))
        time_offset = float(saved["time"])
    timestep = float(case.physics.timestep)
    smoothing_length = float(case.physics.smoothing_length)
    total_steps = int(round(args.duration / timestep))
    kinds = np.asarray([material.kind for material in case.materials])
    fluid_groups = np.flatnonzero(kinds == KIND_FLUID)
    defrag_cadence = case.numerics.defrag_cadence

    context = VulkanContextV6.create(device_index=args.device, application_name="delta_density")
    sim = SphSimulatorV6(context, case)
    offset = sim.stored_density_offset()
    rest_density = sim.reference_density()
    result = {"variant": args.variant, "scenario": args.scenario, "device": args.device,
              "switches": {key: value for key, value in sorted(os.environ.items()) if key.startswith("V6_")},
              "timestep": timestep, "steps": total_steps, "time_offset": time_offset,
              "stored_density_offset": offset, "samples": []}
    pool = case.capacities.total_pool_capacity()
    own = slice(sim.own_first_pid(), sim.own_first_pid() + case.capacities.own_pool_size)

    def read_state():
        raw = sim.readback_buffers_batch(["position_voxel_id", "velocity_mass", "density_pressure", "material"],
                                         density="stored")
        position = np.frombuffer(raw["position_voxel_id"], np.float32).reshape(pool, 4)[own]
        velocity = np.frombuffer(raw["velocity_mass"], np.float32).reshape(pool, 4)[own]
        density_pressure = np.frombuffer(raw["density_pressure"], np.float32).reshape(pool, 2)[own]
        material = np.frombuffer(raw["material"], np.uint32)[:pool][own]
        alive = velocity[:, 3] > 0
        return {"position": position[alive, :2].astype(np.float64), "velocity": velocity[alive, :2].astype(np.float64),
                "mass": velocity[alive, 3].astype(np.float64),
                "stored_density": density_pressure[alive, 0].copy(),
                "density": density_pressure[alive, 0].astype(np.float64) + offset,
                "pressure": density_pressure[alive, 1].astype(np.float64),
                "fluid": np.isin(material[alive], fluid_groups), "slot": np.flatnonzero(alive)}

    def basic_sample(state, step):
        fluid = state["fluid"]
        velocity = state["velocity"][fluid]
        pressure = state["pressure"][fluid]
        density = state["density"][fluid]
        return {"step": step, "time": time_offset + step * timestep,
                "kinetic_energy": float(0.5 * np.sum(state["mass"][fluid] * np.sum(velocity ** 2, axis=1))),
                "pressure_mean": float(pressure.mean()), "pressure_std": float(pressure.std()),
                "pressure_min": float(pressure.min()), "pressure_max": float(pressure.max()),
                "density_mean": float(density.mean()), "density_std": float(density.std()),
                "alive": int(state["position"].shape[0])}

    def noise_and_quantization(state_before, step):
        """Spatial pressure noise, then one more step for the density quantization measures."""
        fluid = state_before["fluid"]
        positions = state_before["position"]
        volumes = state_before["mass"] / state_before["density"]
        tree = cKDTree(positions)
        sample = np.flatnonzero(fluid)[::max(1, int(fluid.sum()) // args.noise_points)]
        smooth = shepard(tree, positions, volumes, state_before["pressure"][:, None], positions[sample],
                         smoothing_length)[:, 0]
        residual = state_before["pressure"][sample] - smooth
        extent = np.maximum(np.abs(positions[sample, 0]), np.abs(positions[sample, 1]))
        near_wall = extent > FLUID_HALF - 4 * smoothing_length
        near_lid = positions[sample, 1] > FLUID_HALF - 4 * smoothing_length
        near_rest = fluid & (np.abs(state_before["density"] - rest_density) < 10 * 6.1035156e-05)
        levels = np.unique(state_before["pressure"][near_rest]).size
        sim.submit_step_single_and_wait()
        state_after = read_state()
        same = state_after["slot"].size == state_before["slot"].size and np.array_equal(
            state_after["slot"], state_before["slot"])
        entry = {"step": step, "pressure_noise_rms": float(np.sqrt(np.nanmean(residual ** 2))),
                 "pressure_noise_interior_rms": float(np.sqrt(np.nanmean(residual[~near_wall] ** 2))),
                 "pressure_noise_wall_rms": float(np.sqrt(np.nanmean(residual[near_wall] ** 2))),
                 "pressure_noise_lid_rms": float(np.sqrt(np.nanmean(residual[near_lid] ** 2))),
                 "pressure_rms": float(np.sqrt(np.mean(state_before["pressure"][fluid] ** 2))),
                 "pressure_levels_near_rest": int(levels), "particles_near_rest": int(near_rest.sum())}
        if same:
            changed = fluid & (state_after["stored_density"] != state_before["stored_density"])
            difference = np.abs(state_after["density"][changed] - state_before["density"][changed])
            entry.update({"fraction_density_changed": float(changed.sum() / fluid.sum()),
                          "median_abs_density_change": float(np.median(difference)) if difference.size else 0.0})
        return entry

    try:
        sim.bootstrap()
        sim.prepare_step_single_cmd_buffer()
        stepping_seconds = 0.0
        noise_entries = []
        step = 0
        next_noise_step = args.noise_every
        result["samples"].append(basic_sample(read_state(), 0))
        while step < total_steps:
            chunk = min(args.sample_every, total_steps - step)
            started = time.perf_counter()
            for _ in range(chunk):
                sim.submit_step_single_and_wait()
                step += 1
                if step % defrag_cadence == 0:
                    sim.submit_defrag_and_wait()
            stepping_seconds += time.perf_counter() - started
            state = read_state()
            result["samples"].append(basic_sample(state, step))
            if step >= next_noise_step or step >= total_steps:
                noise_entries.append(noise_and_quantization(state, step))
                step += 1                       # the quantization step advanced the sim
                next_noise_step += args.noise_every
        result["noise"] = noise_entries
        result["steps_per_second"] = (step - len(noise_entries)) / stepping_seconds if stepping_seconds else None
        # final centerline profiles
        state = read_state()
        positions = state["position"]
        volumes = state["mass"] / state["density"]
        tree = cKDTree(positions)
        coordinate = np.linspace(-FLUID_HALF, FLUID_HALF, 401)
        vertical = np.stack([np.zeros_like(coordinate), coordinate], axis=1)
        horizontal = np.stack([coordinate, np.zeros_like(coordinate)], axis=1)
        u_of_y = shepard(tree, positions, volumes, state["velocity"][:, 0:1], vertical, smoothing_length)[:, 0]
        v_of_x = shepard(tree, positions, volumes, state["velocity"][:, 1:2], horizontal, smoothing_length)[:, 0]
        ghia = load_ghia_1000()
        u_at = np.interp(ghia["y"], coordinate + FLUID_HALF, u_of_y)
        v_at = np.interp(ghia["x"], coordinate + FLUID_HALF, v_of_x)
        result["ghia"] = {"u_rms": float(np.sqrt(np.mean((u_at - ghia["u"]) ** 2))),
                          "u_max": float(np.abs(u_at - ghia["u"]).max()),
                          "v_rms": float(np.sqrt(np.mean((v_at - ghia["v"]) ** 2))),
                          "v_max": float(np.abs(v_at - ghia["v"]).max())}
        result["final_time"] = time_offset + step * timestep
        result["final_alive"] = int(positions.shape[0])
        result["expected"] = int(case.initial.positions.shape[0])
        np.savez_compressed(out_dir / f"{args.scenario}_{args.variant}_profiles.npz",
                            coordinate=coordinate, u_of_y=u_of_y, v_of_x=v_of_x)
        np.savez_compressed(out_dir / f"{args.scenario}_{args.variant}_final_state.npz",
                            position=positions.astype(np.float32), velocity=state["velocity"].astype(np.float32),
                            density=state["density"], pressure=state["pressure"].astype(np.float32),
                            fluid=state["fluid"])
    finally:
        sim.destroy()
        context.destroy()
    (out_dir / f"{args.scenario}_{args.variant}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(RESULT_PREFIX + json.dumps({key: result[key] for key in ("variant", "scenario", "steps_per_second",
                                                                    "ghia", "final_time", "final_alive")}),
          flush=True)
    return 0


# --------------------------------------------------------------------------- driver
def run_driver(args) -> int:
    out_dir = pathlib.Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = {key: value for key, value in os.environ.items() if not key.startswith("V6_")}
    base["VK_LOADER_LAYERS_DISABLE"] = "VK_LAYER_KHRONOS_validation"
    for scenario, duration in (("developed", args.developed_time), ("rest", args.rest_time)):
        if scenario not in args.scenarios.split(","):
            continue
        processes = []
        for device, variant in ((0, "baseline"), (1, "delta")):
            environment = dict(base)
            if variant == "delta":
                environment["V6_DELTA_DENSITY"] = "1"
            command = [sys.executable, "-m", "experiment.seam_audit.delta_density_eval", "--worker",
                       "--variant", variant, "--scenario", scenario, "--duration", str(duration),
                       "--device", str(device), "--out-dir", str(out_dir),
                       "--sample-every", str(args.sample_every), "--noise-every", str(args.noise_every),
                       "--noise-points", str(args.noise_points)]
            log = open(out_dir / f"{scenario}_{variant}.log", "w", encoding="utf-8")
            processes.append((variant, subprocess.Popen(command, env=environment, stdout=log,
                                                        stderr=subprocess.STDOUT, cwd=_REPO_ROOT), log))
            print(f"[delta_density] {scenario}/{variant} on device {device} started", flush=True)
        for variant, process, log in processes:
            code = process.wait()
            log.close()
            print(f"[delta_density] {scenario}/{variant} exit {code}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="V6_DELTA_DENSITY evaluation")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--variant", choices=("baseline", "delta"))
    parser.add_argument("--scenario", choices=("rest", "developed"))
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--out", default="logs/seam_audit/opt/delta_density")
    parser.add_argument("--scenarios", default="developed,rest")
    parser.add_argument("--developed-time", type=float, default=2.0)
    parser.add_argument("--rest-time", type=float, default=5.0)
    parser.add_argument("--sample-every", type=int, default=1000)
    parser.add_argument("--noise-every", type=int, default=50000)
    parser.add_argument("--noise-points", type=int, default=20000)
    args = parser.parse_args()
    if args.worker:
        return run_worker(args)
    return run_driver(args)


if __name__ == "__main__":
    sys.exit(main())
