"""
_test_v8_periodic.py — does the periodic domain of experiment/v8 do what it says? One GPU, a small TGV case.

  1. kernel sum: after the bootstrap, the GPU's Shepard sum sum_j V_j W_ij of every particle against a CPU
     float64 sum over the minimum-image pairs (scipy cKDTree with boxsize = L): the pair set and the wrapped
     separations are right iff the difference is float32 rounding (< 1e-5) everywhere, edges included. The
     same lattice loaded as a NON-periodic case (a control written next to it) shows the edge deficit the wrap
     removes (kernel sum ~ 0.5 on the box edges).
  2. correction: the KCG inverse of every particle ~ I (periodic lattice, no edge).
  3. steps: `--steps` steps (fluid streams cross the box edges: the case's origin_offset 0.25), then drift 0,
     every overflow counter 0, every position inside [0, L)^2, no particle killed, the total momentum ~ 0 and
     the kinetic energy at F(t)^2 within a few per cent.

    .venv/Scripts/python.exe experiment/v8/_test_v8_periodic.py --case logs/v8/smoke/re100_n100/case.yaml --gpu-uuid ...
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

HEADLESS_5090_UUID = "ae137c668a40f5acaab90a83f7cda175"


def wendland_c4_2d(distance: np.ndarray, h: float) -> np.ndarray:
    q = distance / h
    inside = q < 1.0
    one_minus_q = np.where(inside, 1.0 - q, 0.0)
    return (9.0 / (math.pi * h * h)) * one_minus_q ** 6 * ((35.0 / 3.0) * q * q + 6.0 * q + 1.0) * inside


def cpu_kernel_sum(positions: np.ndarray, volume: np.ndarray, h: float, box: float | None) -> np.ndarray:
    """sum_j V_j W(r_ij) per particle, V_j per particle (= m / rho_j with the uploaded density)."""
    from scipy.spatial import cKDTree
    points = positions[:, :2].astype(np.float64)
    tree = cKDTree(points, boxsize=box) if box else cKDTree(points)
    pairs = tree.query_pairs(h * (1.0 + 1e-9), output_type="ndarray")
    difference = points[pairs[:, 0]] - points[pairs[:, 1]]
    if box:
        difference -= box * np.round(difference / box)
    distance = np.sqrt((difference ** 2).sum(axis=1))
    kernel = wendland_c4_2d(distance, h)
    result = np.zeros(points.shape[0])
    np.add.at(result, pairs[:, 0], volume[pairs[:, 1]] * kernel)
    np.add.at(result, pairs[:, 1], volume[pairs[:, 0]] * kernel)
    return result


def readback(sim, names: list[str]) -> dict[str, np.ndarray]:
    layout = {"position_voxel_id": (4, np.float32), "velocity_mass": (4, np.float32), "density_pressure": (2, np.float32),
              "acceleration": (4, np.float32), "correction_inverse": (8, np.float32),
              "density_gradient_kernel_sum": (4, np.float32), "material": (1, np.uint32)}
    raw = sim.readback_buffers_batch(names)
    pool = sim.case.capacities.total_pool_capacity()
    out = {}
    for name in names:
        components, dtype = layout[name]
        array = np.frombuffer(raw[name], dtype=dtype)[:pool * components]
        out[name] = array.reshape(pool, components) if components > 1 else array
    return out


def alive_rows(state: dict) -> np.ndarray:
    return (state["position_voxel_id"][:, 3] > 0.5) & (state["velocity_mass"][:, 3] > 0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True)
    parser.add_argument("--gpu-uuid", default=HEADLESS_5090_UUID)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--out", default=None, help="JSON report path")
    arguments = parser.parse_args()

    from experiment.v7_single._run_single_bench import resolve_device_index
    from experiment.v7_single.utils.vulkan_context_single import VulkanContextSingle
    from experiment.v8.utils import taylor_green
    from experiment.v8.utils.case_loader_v8 import load_case_v8
    from experiment.v8.utils.simulator_v8 import SphSimulatorV8

    case_path = pathlib.Path(arguments.case).resolve()
    # the non-periodic control: the same lattice, v7's grid rule, no initial field
    control_dir = case_path.parent / "nonperiodic_control"
    control_dir.mkdir(exist_ok=True)
    text = case_path.read_text(encoding="utf-8")
    assert "periodic: [true, true]" in text and "initial_field:" in text
    control_text = text.replace("periodic: [true, true]", "periodic: [false, false]")
    head, _, tail = control_text.partition("initial_field:")
    control_text = head + "numerics:" + tail.split("numerics:", 1)[1]
    for name in ("domain.obj", "frame.obj", "materials.yaml"):
        (control_dir / name).write_bytes((case_path.parent / name).read_bytes())
    (control_dir / "case.yaml").write_text(control_text, encoding="utf-8")

    report: dict = {"case": str(case_path), "steps": arguments.steps, "checks": {}}
    device_index = resolve_device_index(arguments.gpu_uuid)
    ctx = VulkanContextSingle.create(device_index=device_index, enable_validation=False, application_name="v8_periodic_test")
    try:
        failures = []
        # ---- control: edge deficit without the wrap ----------------------------------------------------------
        control = load_case_v8(control_dir / "case.yaml")
        sim = SphSimulatorV8(ctx, control)
        sim.bootstrap()
        state = readback(sim, ["position_voxel_id", "velocity_mass", "density_gradient_kernel_sum"])
        alive = alive_rows(state)
        control_sum = state["density_gradient_kernel_sum"][alive, 3].astype(np.float64)
        sim.destroy()
        report["checks"]["control_kernel_sum"] = {"min": float(control_sum.min()), "max": float(control_sum.max()),
                                                  "mean": float(control_sum.mean())}
        print(f"[test] non-periodic control: kernel sum min {control_sum.min():.4f} max {control_sum.max():.4f}")

        # ---- periodic: kernel sum against the CPU minimum-image sum ------------------------------------------
        case = load_case_v8(case_path)
        box = float(case.domain_length[0])
        h = float(case.case.physics.smoothing_length)
        mass = float(case.case.materials[0].rest_density * case.case.materials[0].volume)
        sim = SphSimulatorV8(ctx, case)
        sim.bootstrap()
        state = readback(sim, ["position_voxel_id", "velocity_mass", "density_gradient_kernel_sum", "correction_inverse",
                               "density_pressure"])
        alive = alive_rows(state)
        positions = state["position_voxel_id"][alive, :3]
        gpu_sum = state["density_gradient_kernel_sum"][alive, 3].astype(np.float64)
        # the correction read the uploaded density rho_0(x) (density_from_pressure) for V_j = m / rho_j; the
        # slots are the initial order (no defrag yet)
        uploaded_density = (case.initial_density if case.initial_density is not None
                            else np.full(positions.shape[0], case.case.materials[0].rest_density)).astype(np.float64)
        cpu_sum = cpu_kernel_sum(positions, mass / uploaded_density, h, box)
        difference = np.abs(gpu_sum - cpu_sum)
        # the lattice has pairs at r = h exactly (h = 5 dx: (3, 4) dx) whose float32 / float64 inclusion can differ;
        # W(h) = 0 so they change nothing, and the loader's volume calibration excludes them likewise
        report["checks"]["periodic_kernel_sum"] = {
            "gpu_min": float(gpu_sum.min()), "gpu_max": float(gpu_sum.max()), "cpu_min": float(cpu_sum.min()),
            "cpu_max": float(cpu_sum.max()), "max_abs_difference": float(difference.max()),
            "mean_abs_difference": float(difference.mean()), "alive": int(alive.sum())}
        print(f"[test] periodic: kernel sum GPU [{gpu_sum.min():.6f}, {gpu_sum.max():.6f}] CPU [{cpu_sum.min():.6f}, "
              f"{cpu_sum.max():.6f}] max |GPU - CPU| {difference.max():.2e}")
        if difference.max() > 1e-5:
            failures.append(f"kernel sum differs from the CPU minimum-image sum by {difference.max():.2e}")
        if gpu_sum.min() < 0.99:
            failures.append(f"periodic kernel sum min {gpu_sum.min():.4f} (edge deficit?)")
        # the KCG inverse on a full lattice support is c I with c = 1 / 1.129 (the calibrated volume's M, common.glsl
        # REGULARIZATION_XI); an edge particle would get an anisotropic, larger L. Check isotropy and uniformity.
        inverse = state["correction_inverse"][alive].astype(np.float64)
        diagonal = inverse[:, :2]
        off_diagonal = np.abs(inverse[:, 3]).max()
        anisotropy = np.abs(diagonal[:, 0] - diagonal[:, 1]).max()
        spread = float(diagonal.max() - diagonal.min())
        report["checks"]["correction_inverse"] = {"diagonal_mean": float(diagonal.mean()), "diagonal_min": float(diagonal.min()),
                                                 "diagonal_max": float(diagonal.max()), "max_off_diagonal": float(off_diagonal),
                                                 "max_anisotropy": float(anisotropy)}
        print(f"[test] periodic: KCG inverse diagonal {diagonal.mean():.5f} (min {diagonal.min():.5f} max {diagonal.max():.5f}), "
              f"max |off-diagonal| {off_diagonal:.2e}, max |L_xx - L_yy| {anisotropy:.2e}")
        if off_diagonal > 1e-2 or anisotropy > 1e-2 or spread > 0.05:
            failures.append(f"KCG inverse not isotropic / uniform: off {off_diagonal:.3f} aniso {anisotropy:.3f} spread {spread:.3f}")

        # ---- steps across the box edges ---------------------------------------------------------------------
        expected = int(case.case.initial.positions.shape[0])
        sim.prepare_step(in_flight=1)
        cadence = int(case.case.numerics.defrag_cadence)
        for frame in range(1, arguments.steps + 1):
            sim.submit_step(with_defrag=bool(cadence) and frame % cadence == 0)
        sim.drain()
        status = sim.readback_global_status()
        overflow = {name: value for name, value in status.items() if name.startswith("overflow_") and value}
        drift = int(status["alive_particle_count"]) - expected
        state = readback(sim, ["position_voxel_id", "velocity_mass", "acceleration", "density_pressure"])
        alive = alive_rows(state)
        positions = state["position_voxel_id"][alive, :3].astype(np.float64)
        inside = ((positions[:, 0] >= 0.0) & (positions[:, 0] < box) & (positions[:, 1] >= 0.0) & (positions[:, 1] < box))
        dt = float(case.case.physics.timestep)
        velocity = (state["velocity_mass"][alive, :3].astype(np.float64)
                    + 0.5 * dt * state["acceleration"][alive, :3].astype(np.float64))
        mass = state["velocity_mass"][alive, 3].astype(np.float64)
        momentum = (mass[:, None] * velocity).sum(axis=0)
        kinetic = 0.5 * (mass * (velocity ** 2).sum(axis=1)).sum()
        kinetic_0 = 0.5 * (mass * (np.asarray(case.case.initial.velocities, dtype=np.float64) ** 2).sum(axis=1)).sum()
        time_value = arguments.steps * dt
        expected_ratio = taylor_green.decay_factor(time_value, case.field["viscosity"], case.field["wavenumber"]) ** 2
        initial_x = np.asarray(case.case.initial.positions, dtype=np.float64)
        crossings = int((np.abs(positions[:, 0] - initial_x[:, 0]) > box / 2).sum()) if alive.sum() == expected else -1
        report["checks"]["steps"] = {
            "drift": drift, "overflow": overflow, "alive": int(alive.sum()), "outside_box": int((~inside).sum()),
            "momentum": momentum.tolist(), "momentum_scale": float((mass * np.abs(velocity).sum(axis=1)).sum()),
            "kinetic_ratio": float(kinetic / kinetic_0), "expected_ratio": float(expected_ratio), "time": time_value,
            "edge_crossings_x": crossings, "correction_fallback": int(status["correction_fallback_count"]),
            "max_speed": float(np.sqrt((velocity ** 2).sum(axis=1)).max())}
        print(f"[test] {arguments.steps} steps (t = {time_value:.4f}): alive {alive.sum()} drift {drift:+d} overflow "
              f"{overflow or 0} outside {int((~inside).sum())} crossings(x, by slot) {crossings} "
              f"E/E0 {kinetic / kinetic_0:.4f} (exact {expected_ratio:.4f}) |P| {np.abs(momentum).max():.2e}")
        if drift or overflow or (~inside).any():
            failures.append(f"drift {drift}, overflow {overflow}, {int((~inside).sum())} particles outside the box")
        if abs(kinetic / kinetic_0 - expected_ratio) > 0.05 * expected_ratio:
            failures.append(f"kinetic energy ratio {kinetic / kinetic_0:.4f} vs exact {expected_ratio:.4f}")
        if np.abs(momentum).max() > 1e-4 * (mass * np.abs(velocity).sum(axis=1)).sum():
            failures.append(f"total momentum {momentum}")
        sim.destroy()
    finally:
        ctx.destroy()
    report["failures"] = failures
    report["verdict"] = "PASS" if not failures else "FAIL"
    if arguments.out:
        out = pathlib.Path(arguments.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"[test] {report['verdict']}" + ("" if not failures else ": " + "; ".join(failures)))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
