"""
_run_tgv.py — one Taylor–Green run on one GPU: experiment/v8's solver on a periodic TGV case, sampled against the
analytic solution (experiment/v8/utils/taylor_green.py) every --sample-dt of simulated time.

    .venv/Scripts/python.exe experiment/v8/_run_tgv.py --case cases/tgv_2d/re100_n200/case.yaml --t-end 2 \\
        --out logs/v8/tgv/re100_n200 [--gpu-uuid ae137c668a40f5acaab90a83f7cda175] [--in-flight 2]

At every sample the queue is drained and the per-particle state read back (position, velocity, acceleration,
density / pressure, kernel sum, material); the leapfrog velocity v_{n+1/2} is synchronised to the positions'
time, v_{n+1} = v_{n+1/2} + 1/2 a_{n+1} dt (a_{n+1} is the force of the step just completed), and compared with
u(x_i, t_{n+1}). Per sample (samples.jsonl):

  kinetic_energy        1/2 sum m_i |v_i|^2 over the fluid (m_i = rho0 V_calibrated: the SPH mass)
  kinetic_energy_lattice   1/2 rho0 dx^2 sum |v_i|^2 (the lattice quadrature of the continuum energy)
  velocity_l2_relative  sqrt(sum |v_i - u_i|^2 / sum |u_i|^2);  velocity_l2 / U0, velocity_linf / U0 likewise
  max_speed             max |v_i| (exact: U0 F(t))
  pressure_l2           sqrt(mean (p_i - p_i^exact)^2) / (rho0 U0^2), raw and after removing the mean offset
  density               max |rho / rho0 - 1|, mean rho / rho0 - 1
  kernel_sum            min / max of the Shepard sum (1 on the lattice; < 1 flags a neighbourhood deficit)
  momentum              sum m_i v_i (x, y)

Snapshots (--snapshot-times, simulated time) go to <out>/snapshots/t<time>.npz with the synchronised velocity,
the analytic fields at the particles and the pressure. result.json carries the case, the commit, the GPU, the
status block (drift and every overflow counter must be 0) and the wall time. A defrag runs at the case's cadence,
appended to the step's submission as the throughput runner does.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys
import time

import numpy as np

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

HEADLESS_5090_UUID = "ae137c668a40f5acaab90a83f7cda175"
READ_FIELDS = {"position_voxel_id": 4, "velocity_mass": 4, "acceleration": 4, "density_pressure": 2,
               "density_gradient_kernel_sum": 4, "material": 1}


def git_state() -> dict:
    def run(*arguments: str) -> str:
        return subprocess.run(["git", *arguments], cwd=_REPOSITORY_ROOT, capture_output=True, text=True).stdout.strip()
    tracked = [line for line in run("status", "--porcelain", "--untracked-files=no").splitlines() if line.strip()]
    return {"commit": run("rev-parse", "HEAD"), "short": run("rev-parse", "--short", "HEAD"), "tracked_dirty": tracked}


def read_state(sim) -> dict[str, np.ndarray]:
    raw = sim.readback_buffers_batch(list(READ_FIELDS))
    pool = sim.case.capacities.total_pool_capacity()
    state = {}
    for name, components in READ_FIELDS.items():
        dtype = np.uint32 if name == "material" else np.float32
        array = np.frombuffer(raw[name], dtype=dtype)[:pool * components]
        state[name] = array.reshape(pool, components) if components > 1 else array
    return state


class Sampler:
    def __init__(self, case_v8, sim, out: pathlib.Path, snapshot_times: list[float]) -> None:
        self.case_v8 = case_v8
        self.sim = sim
        self.out = out
        self.field = case_v8.field
        self.dt = float(case_v8.case.physics.timestep)
        self.kinds = np.array([material.kind for material in case_v8.case.materials])
        self.fluid_group_rest_density = float(self.field["rest_density"])
        self.amplitude = float(self.field["amplitude"])
        self.spacing = 2.0 * float(case_v8.case.materials[0].radius)
        self.snapshot_steps = {int(round(t / self.dt)): t for t in snapshot_times}
        self.samples_path = out / "samples.jsonl"
        self.samples_path.write_text("", encoding="utf-8")
        (out / "snapshots").mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []

    def analytic_keywords(self) -> dict:
        return dict(amplitude=self.amplitude, viscosity=float(self.field["viscosity"]), k=float(self.field["wavenumber"]),
                    x0=float(self.field["origin"][0]), y0=float(self.field["origin"][1]))

    def sample(self, step: int) -> dict:
        from experiment.v8.utils import taylor_green
        state = read_state(self.sim)
        alive = (state["position_voxel_id"][:, 3] > 0.5) & (state["velocity_mass"][:, 3] > 0.0)
        fluid = alive & (self.kinds[state["material"]] == 0)
        time_value = step * self.dt
        positions = state["position_voxel_id"][fluid, :3].astype(np.float64)
        mass = state["velocity_mass"][fluid, 3].astype(np.float64)
        velocity = (state["velocity_mass"][fluid, :3].astype(np.float64)
                    + 0.5 * self.dt * state["acceleration"][fluid, :3].astype(np.float64))[:, :2]
        density = state["density_pressure"][fluid, 0].astype(np.float64)
        pressure_value = state["density_pressure"][fluid, 1].astype(np.float64)
        kernel_sum = state["density_gradient_kernel_sum"][fluid, 3].astype(np.float64)
        keywords = self.analytic_keywords()
        exact_velocity = taylor_green.velocity(positions[:, 0], positions[:, 1], time_value, **keywords)
        exact_pressure = taylor_green.pressure(positions[:, 0], positions[:, 1], time_value,
                                               rest_density=self.fluid_group_rest_density, **keywords)
        difference = velocity - exact_velocity
        speed = np.sqrt((velocity ** 2).sum(axis=1))
        exact_norm = math.sqrt((exact_velocity ** 2).sum())
        rho0, u0 = self.fluid_group_rest_density, self.amplitude
        pressure_error = pressure_value - exact_pressure
        row = {
            "step": step, "time": time_value, "fluid": int(fluid.sum()),
            "kinetic_energy": float(0.5 * (mass * (velocity ** 2).sum(axis=1)).sum()),
            "kinetic_energy_lattice": float(0.5 * rho0 * self.spacing ** 2 * (velocity ** 2).sum()),
            "kinetic_energy_exact": taylor_green.kinetic_energy(time_value, amplitude=u0, viscosity=keywords["viscosity"],
                                                               k=keywords["k"], rest_density=rho0,
                                                               length=float(self.field["length"])),
            "decay_factor_exact": taylor_green.decay_factor(time_value, keywords["viscosity"], keywords["k"]),
            "velocity_l2_relative": float(math.sqrt((difference ** 2).sum()) / exact_norm) if exact_norm > 0 else None,
            "velocity_l2": float(math.sqrt((difference ** 2).sum(axis=1).mean()) / u0),
            "velocity_linf": float(np.sqrt((difference ** 2).sum(axis=1)).max() / u0),
            "max_speed": float(speed.max()), "max_speed_exact": float(u0 * taylor_green.decay_factor(time_value, keywords["viscosity"], keywords["k"])),
            "pressure_l2": float(math.sqrt((pressure_error ** 2).mean()) / (rho0 * u0 * u0)),
            "pressure_l2_centered": float(math.sqrt(((pressure_error - pressure_error.mean()) ** 2).mean()) / (rho0 * u0 * u0)),
            "pressure_offset": float(pressure_error.mean() / (rho0 * u0 * u0)),
            "density_max_deviation": float(np.abs(density / rho0 - 1.0).max()),
            "density_mean_deviation": float((density / rho0 - 1.0).mean()),
            "kernel_sum_min": float(kernel_sum.min()), "kernel_sum_max": float(kernel_sum.max()),
            "momentum": [float(value) for value in (mass[:, None] * velocity).sum(axis=0)],
        }
        self.rows.append(row)
        with self.samples_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if step in self.snapshot_steps:
            name = f"t{self.snapshot_steps[step]:g}".replace(".", "p")
            vorticity_exact = taylor_green.vorticity(positions[:, 0], positions[:, 1], time_value, **keywords)
            np.savez_compressed(self.out / "snapshots" / f"{name}.npz", step=step, time=time_value,
                                position=positions[:, :2].astype(np.float32), velocity=velocity.astype(np.float32),
                                exact_velocity=exact_velocity.astype(np.float32), density=density.astype(np.float32),
                                pressure=pressure_value.astype(np.float32), exact_pressure=exact_pressure.astype(np.float32),
                                kernel_sum=kernel_sum.astype(np.float32), exact_vorticity=vorticity_exact.astype(np.float32))
        return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True)
    parser.add_argument("--gpu-uuid", default=HEADLESS_5090_UUID)
    parser.add_argument("--t-end", type=float, default=2.0)
    parser.add_argument("--sample-dt", type=float, default=0.02)
    parser.add_argument("--snapshot-times", default="0,0.5,1,2")
    parser.add_argument("--in-flight", type=int, default=2)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", default="")
    arguments = parser.parse_args()

    from experiment.v7_single._run_single_bench import resolve_device_index
    from experiment.v7_single.utils.vulkan_context_single import VulkanContextSingle
    from experiment.v8.utils.case_loader_v8 import load_case_v8
    from experiment.v8.utils.simulator_v8 import SphSimulatorV8

    out = (_REPOSITORY_ROOT / arguments.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    case_v8 = load_case_v8(arguments.case)
    if case_v8.field.get("type") != "taylor_green":
        sys.exit("the case has no taylor_green initial_field")
    dt = float(case_v8.case.physics.timestep)
    total_steps = int(round(arguments.t_end / dt))
    sample_every = max(1, int(round(arguments.sample_dt / dt)))
    snapshot_times = [float(value) for value in arguments.snapshot_times.split(",") if value.strip()]
    cadence = int(case_v8.case.numerics.defrag_cadence)
    expected = int(case_v8.case.initial.positions.shape[0])
    device_index = resolve_device_index(arguments.gpu_uuid)
    ctx = VulkanContextSingle.create(device_index=device_index, enable_validation=False, application_name="sph_v8_tgv")
    result: dict = {"status": "error"}
    try:
        sim = SphSimulatorV8(ctx, case_v8)
        sampler = Sampler(case_v8, sim, out, snapshot_times)
        print(f"[tgv] {arguments.case}: {expected:,} particles, dt={dt:.4g}, {total_steps:,} steps to t={arguments.t_end:g}, "
              f"sample every {sample_every} steps, in_flight={arguments.in_flight}, gpu={ctx.device_name}", flush=True)
        wall_start = time.perf_counter()
        sim.bootstrap()
        sim.prepare_step(in_flight=arguments.in_flight)
        first = sampler.sample(0)
        print(f"[tgv] t=0: E={first['kinetic_energy']:.6f} (lattice {first['kinetic_energy_lattice']:.6f}, exact "
              f"{first['kinetic_energy_exact']:.6f}) velocity_l2={first['velocity_l2']:.3e} pressure_l2={first['pressure_l2']:.3e} "
              f"kernel_sum [{first['kernel_sum_min']:.4f}, {first['kernel_sum_max']:.4f}]", flush=True)
        step_start = time.perf_counter()
        last_print = step_start
        for frame in range(1, total_steps + 1):
            sim.submit_step(with_defrag=bool(cadence) and frame % cadence == 0)
            if frame % sample_every == 0 or frame == total_steps or frame in sampler.snapshot_steps:
                sim.drain()
                row = sampler.sample(frame)
                now = time.perf_counter()
                if now - last_print > 20.0 or frame == total_steps:
                    print(f"[tgv] step {frame:,}/{total_steps:,} t={row['time']:.3f}: E/E0={row['kinetic_energy'] / first['kinetic_energy']:.5f} "
                          f"(exact {row['decay_factor_exact'] ** 2:.5f}) velocity_l2={row['velocity_l2']:.3e} "
                          f"rel={row['velocity_l2_relative']:.3e} pressure_l2={row['pressure_l2_centered']:.3e} "
                          f"{(now - step_start) / frame * 1e3:.2f} ms/step", flush=True)
                    last_print = now
        sim.drain()
        wall_steps = time.perf_counter() - step_start
        status = sim.readback_global_status()
        overflow = {name: value for name, value in status.items() if name.startswith("overflow_") and value}
        drift = int(status["alive_particle_count"]) - expected
        result = {"status": "ok" if drift == 0 and not overflow else "bad_state", "case": arguments.case, "tag": arguments.tag,
                  "out": str(out), "gpu": ctx.device_name, "gpu_uuid": ctx.device_uuid, "particles": expected,
                  "alive": int(status["alive_particle_count"]), "drift": drift, "overflow": overflow,
                  "correction_fallback": int(status["correction_fallback_count"]), "steps": total_steps,
                  "timestep": dt, "t_end": arguments.t_end, "sample_every": sample_every, "in_flight": arguments.in_flight,
                  "defrag_cadence": cadence, "wall_seconds": round(time.perf_counter() - wall_start, 2),
                  "step_wall_seconds": round(wall_steps, 2), "steps_per_second": round(total_steps / wall_steps, 2),
                  "field": case_v8.field, "periodic": list(case_v8.periodic), "git": git_state(),
                  "spirv_digests": {name: digest[:16] for name, digest in
                                    __import__("experiment.v7_single.utils.spirv_manifest", fromlist=["spirv_digests"])
                                    .spirv_digests(sim.spirv_directory).items()},
                  **sim.resolution_record()}
        print(f"[tgv] done: {total_steps:,} steps in {wall_steps:.1f} s ({total_steps / wall_steps:.0f} steps/s), alive "
              f"{status['alive_particle_count']:,} drift {drift:+d} overflow {overflow or 0} fallback "
              f"{status['correction_fallback_count']}", flush=True)
        sim.destroy()
    finally:
        ctx.destroy()
    (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("RESULT " + json.dumps({key: result[key] for key in ("status", "drift", "overflow", "steps", "steps_per_second")
                                  if key in result}), flush=True)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
