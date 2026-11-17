"""
single_step.py - one GPU evaluation of the fused correction + density pass and of force with
numerics.wall_boundary crossing against the CPU reference (experiment/wall_crossing/reference.py), on a small
box of a few thousand particles with prescribed smooth fields; 2-D and 3-D.

    .venv/Scripts/python.exe -m experiment.wall_crossing.single_step --dimension 2 --device 1 \
        --out logs/wall_crossing/single_step

How one GPU evaluation is obtained: the bootstrap. bootstrap_compute records correction_density_all, force_all
and bootstrap_half_kick on the uploaded state; force reads the uploaded (rho_0, P_0) from primary (there is no
scratch -> primary copy in the bootstrap). So with the initial payload replaced by prescribed fields
(velocity, density, pressure of every fluid particle; walls keep rho0 and their material's velocity):
    density_pressure_scratch.x = rho_0 + dt * (density rate)       -> the density rate of the fused pass
    acceleration, shift                                             -> force
    correction_inverse, density_gradient_kernel_sum,
    wall_crossing_fluid_inverse                                     -> L_i, g_f, the kernel sum, A_f^-1
all read back before any defrag (row k of the case's initial particles is own pid first + k).

Configurations (--configurations, comma list):
  smooth                  jittered lattice, a few fluid particles below the distance floor, a moving lid,
                          non-linear velocity / density / pressure fields with both pressure signs:
                          the GPU against the CPU reference, particle by particle (both continuity rules);
  couette                 lattice, v = U (y - y_bottom) / H e_x, lid at U, uniform rho, P = 0: the no-slip rule
                          is exact for it away from the side walls (viscous acceleration and density rate ~ 0);
  hydrostatic_positive    lattice, P = P_c + rho0 g . (r - r_c) > 0 with the case's gravity g (oblique), v = 0:
  hydrostatic_negative    the same with P < 0 (the other TIC branch): the acceleration is ~ 0 everywhere.
With --compare-modes the three property configurations also run with wall_boundary simple and adami (GPU only),
for the size of the same residuals without the mode.

--density-budget (diagnostic; sets V7_DENSITY_BUDGET=1 before the simulator is imported): the smooth
configuration with wall_boundary crossing and with wall_boundary simple, the fused pass in its DENSITY_BUDGET
build. After the bootstrap density_budget holds dt x the four terms of every fluid particle's density rate in
the particle's region; they are compared with the reference's four terms particle by particle, their sum with
the rate the pass stored, and the region with the host's.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.box_case import BoxCaseSpecification, box_particles, write_box_case   # noqa: E402
from experiment.wall_crossing.reference import ReferenceConstants, evaluate_reference                # noqa: E402
from experiment.wall_crossing.run_log import LoggedRun, vulkan_device_uuid                           # noqa: E402

PROPERTY_CONFIGURATIONS = ("couette", "hydrostatic_positive", "hydrostatic_negative")
ALL_CONFIGURATIONS = ("smooth",) + PROPERTY_CONFIGURATIONS
HYDROSTATIC_GRAVITY = {2: (0.35, -0.80, 0.0), 3: (0.35, -0.80, 0.45)}        # oblique to every wall
# The smooth configuration has no gravity, so the diagnostic delta_wall body_force (a wall neighbour's density
# difference rho_i g . (r_j - r_i) / c^2) would be zero on it and the comparison would say nothing about the
# term. With that value only, the smooth configuration takes this gravity (oblique to every wall, large
# enough for the walls' diffusion to be of the size of the fluid's).
BODY_FORCE_SMOOTH_GRAVITY = {2: (3.5, -8.0, 0.0), 3: (3.5, -8.0, 4.5)}
LID_SPEED = 1.0


def box_specification(dimension: int, configuration: str, wall_boundary: str, continuity: str,
                      distance_floor_factor: float) -> BoxCaseSpecification:
    """The box of one configuration. 2-D: 50 x 40 fluid particles, h / dx = 5; 3-D: 24 x 20 x 16, h / dx = 4.
    Neither is a cube and the box does not sit at the origin, so an axis mix-up shows."""
    if dimension == 2:
        base = dict(dimension=2, fluid_counts=(50, 40), spacing=0.02, support_ratio=5.0)
    else:
        base = dict(dimension=3, fluid_counts=(24, 20, 16), spacing=0.025, support_ratio=4.0)
    specification = BoxCaseSpecification(
        wall_boundary=wall_boundary, continuity=continuity, distance_floor_factor=distance_floor_factor,
        speed_of_sound=10.0, viscosity=1.0e-2, **base)
    if configuration == "smooth":
        specification.jitter_fraction = 0.2
        specification.near_wall_count = 8
        specification.near_wall_fraction = 0.1          # below the floor 0.25 dx
        specification.explicit_box = True               # jitter moves the fluid bounding box
        specification.lid_velocity = (LID_SPEED, 0.0, 0.0)
    elif configuration == "couette":
        specification.lid_velocity = (LID_SPEED, 0.0, 0.0)
    else:
        specification.lid_velocity = (0.0, 0.0, 0.0)
        specification.gravity = HYDROSTATIC_GRAVITY[dimension]
    specification.comment = f"single_step configuration {configuration}"
    return specification


def prescribed_fields(configuration: str, specification: BoxCaseSpecification, positions: np.ndarray) -> dict:
    """Velocity (N, 3), density (N,) and pressure (N,) of the fluid particles at positions (float64)."""
    dimension = specification.dimension
    lower, upper = specification.box_lower(), specification.box_upper()
    extent = np.where(upper > lower, upper - lower, 1.0)
    scaled = (positions - lower) / extent                       # in [0, 1] on the active axes, 0 on the others
    count = positions.shape[0]
    rest_density = specification.rest_density
    velocity = np.zeros((count, 3))
    density = np.full(count, rest_density)
    pressure = np.zeros(count)
    pi = np.pi
    if configuration == "smooth":
        depth = scaled[:, 2] if dimension == 3 else np.zeros(count)
        velocity[:, 0] = np.sin(2.0 * pi * scaled[:, 0] + 0.3) * np.cos(1.5 * pi * scaled[:, 1]) \
            + 0.4 * scaled[:, 1] + 0.3 * np.cos(1.1 * pi * depth)
        velocity[:, 1] = np.cos(pi * scaled[:, 0]) * np.sin(2.0 * pi * scaled[:, 1] + 0.5) \
            - 0.3 * scaled[:, 0] + 0.2 * np.sin(0.7 * pi * depth)
        if dimension == 3:
            velocity[:, 2] = np.sin(1.2 * pi * depth + 0.1) * np.cos(pi * scaled[:, 0]) + 0.25 * scaled[:, 1]
        density = rest_density * (1.0 + 0.03 * np.sin(1.7 * pi * scaled[:, 0] + 0.2)
                                  * np.cos(1.3 * pi * scaled[:, 1] - 0.4) * np.cos(0.9 * pi * depth))
        pressure_scale = 0.02 * rest_density * specification.speed_of_sound ** 2
        pressure = pressure_scale * np.sin(2.2 * pi * scaled[:, 0] - 0.6) * np.cos(1.9 * pi * scaled[:, 1] + 0.3) \
            * np.cos(1.4 * pi * depth + 0.2)
    elif configuration == "couette":
        velocity[:, 0] = LID_SPEED * scaled[:, 1]
    elif configuration in ("hydrostatic_positive", "hydrostatic_negative"):
        gravity = np.asarray(specification.gravity, dtype=np.float64)
        centre = 0.5 * (lower + upper)
        hydrostatic = rest_density * ((positions - centre) * gravity).sum(axis=1)
        swing = float(np.abs(hydrostatic).max())
        offset = 3.0 * swing if configuration == "hydrostatic_positive" else -3.0 * swing
        pressure = offset + hydrostatic
    else:
        raise ValueError(f"unknown configuration {configuration!r}")
    return {"velocity": velocity, "density": density, "pressure": pressure}


def gpu_single_evaluation(case_path: pathlib.Path, device_index: int, configuration: str,
                          specification: BoxCaseSpecification) -> dict:
    """Load the case, replace the fluid's initial velocity / density / pressure by the prescribed fields, run the
    bootstrap (one evaluation of both passes) and read the results back. Returns the float32 inputs as uploaded
    and the outputs, one row per initial particle (the case's order)."""
    from vulkan import vkFreeCommandBuffers

    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7.utils.case_v7 import KIND_BOUNDARY
    from experiment.v7.utils.simulator_v7 import SphSimulatorV7
    from experiment.v7.utils.vulkan_context_v7 import VulkanContextV7

    case = load_case_v7(case_path)
    particle_count = int(case.initial.positions.shape[0])
    kinds = np.asarray([material.kind for material in case.materials])
    is_wall = kinds[case.initial.material_group] == KIND_BOUNDARY
    fields = prescribed_fields(configuration, specification,
                               case.initial.positions[~is_wall].astype(np.float64))

    context = VulkanContextV7.create(device_index=device_index, enable_validation=False,
                                     application_name="wall_crossing_single_step")
    simulator = SphSimulatorV7(context, case)
    try:
        device_uuid = vulkan_device_uuid(context)
        pool_capacity = case.capacities.total_pool_capacity()
        first = simulator.own_first_pid()
        rows = slice(first, first + particle_count)
        initial = simulator._build_initial_data()
        velocity_mass = np.frombuffer(initial["velocity_mass"], dtype=np.float32).reshape(pool_capacity, 4).copy()
        density_pressure = np.frombuffer(initial["density_pressure"], dtype=np.float32) \
            .reshape(pool_capacity, 2).copy()
        offset = simulator.stored_density_offset()
        fluid_rows = first + np.flatnonzero(~is_wall)
        velocity_mass[fluid_rows, 0:3] = fields["velocity"].astype(np.float32)
        density_pressure[fluid_rows, 0] = (fields["density"] - offset).astype(np.float32)
        density_pressure[fluid_rows, 1] = fields["pressure"].astype(np.float32)
        initial["velocity_mass"] = velocity_mass.tobytes()
        initial["density_pressure"] = density_pressure.tobytes()

        simulator._upload_initial_state(initial)
        command = simulator._record_bootstrap_init_cmd()
        context.submit_and_wait(command)
        vkFreeCommandBuffers(context.device, context.command_pool, 1, [command])
        simulator.bootstrap_compute()

        names = ["density_pressure_scratch", "acceleration", "shift", "correction_inverse",
                 "density_gradient_kernel_sum"]
        if simulator._wall_crossing:
            names.append("wall_crossing_fluid_inverse")
        if simulator._wall_adami:
            names.append("wall_dummy_velocity")
        if simulator._density_budget_active():
            names.append("density_budget")
        raw = simulator.readback_buffers_batch(names, density="absolute")
        status = simulator.readback_global_status()

        def table(name: str, components: int) -> np.ndarray:
            return np.frombuffer(raw[name], dtype=np.float32)[:pool_capacity * components] \
                .reshape(pool_capacity, components)[rows].copy()

        result = {
            "case": case,
            "device_uuid": device_uuid,
            "is_wall": is_wall,
            "positions": case.initial.positions.astype(np.float32),
            "velocities": velocity_mass[rows, 0:3].copy(),
            "masses": velocity_mass[rows, 3].copy(),
            "densities": density_pressure[rows, 0].astype(np.float64) + offset,       # rho as float64 of float32
            "densities_float32": density_pressure[rows, 0].copy(),
            "pressures": density_pressure[rows, 1].copy(),
            "scratch": table("density_pressure_scratch", 2),
            "acceleration": table("acceleration", 4)[:, 0:3],
            "shift": table("shift", 4)[:, 0:3],
            "correction_inverse": table("correction_inverse", 8),
            "density_gradient_kernel_sum": table("density_gradient_kernel_sum", 4),
            "status": status,
        }
        if simulator._wall_crossing:
            result["wall_crossing_fluid_inverse"] = table("wall_crossing_fluid_inverse", 8)
        if simulator._density_budget_active():
            budget_entries = table("density_budget", 72).reshape(particle_count, 9, 2, 4)
            result["density_budget"] = budget_entries[:, :, 0, :].copy()          # dt x the four terms
            result["density_budget_applied"] = budget_entries[:, :, 1, 0].copy()  # the stored density's change
            result["density_budget_box"] = simulator.density_budget_box()
        return result
    finally:
        simulator.destroy()
        context.destroy()


def _error_statistics(gpu: np.ndarray, reference: np.ndarray, rounding_scale: np.ndarray | None = None) -> dict:
    """Errors of a per-particle scalar (N,) or vector (N, 3) quantity: the largest |gpu - reference| over the
    largest |reference| ("of the field"), the per-particle relative error |gpu - reference| / |reference| with
    the reference floored at 1 % of its largest value (maximum and 99.9th percentile), and, when the sum of the
    absolute pair terms is given, the error in units of that rounding scale."""
    gpu = np.asarray(gpu, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if gpu.ndim == 1:
        difference = np.abs(gpu - reference)
        magnitude = np.abs(reference)
    else:
        difference = np.sqrt(((gpu - reference) ** 2).sum(axis=1))
        magnitude = np.sqrt((reference ** 2).sum(axis=1))
    largest = float(magnitude.max()) if magnitude.size else 0.0
    floor = 0.01 * largest if largest > 0.0 else 1.0
    relative = difference / np.maximum(magnitude, floor)
    statistics = {
        "reference_maximum": largest,
        "difference_maximum": float(difference.max()) if difference.size else 0.0,
        "error_of_field_maximum": float(difference.max() / largest) if largest > 0.0 else float(difference.max()),
        "relative_error_maximum": float(relative.max()) if relative.size else 0.0,
        "relative_error_percentile_99_9": float(np.percentile(relative, 99.9)) if relative.size else 0.0,
        "relative_error_median": float(np.median(relative)) if relative.size else 0.0,
        "worst_row": int(np.argmax(relative)) if relative.size else -1,
    }
    if rounding_scale is not None:
        scaled = difference / np.maximum(np.asarray(rounding_scale, dtype=np.float64), 1e-300)
        statistics["error_over_pair_term_sum_maximum"] = float(scaled.max())
    return statistics


def compare_with_reference(gpu: dict, tolerance: float) -> dict:
    """The GPU evaluation against the CPU reference on the same float32 inputs. Passes when every compared
    quantity's per-particle relative error (reference floored at 1 % of its largest value) is below tolerance,
    the wall rows hold exactly what the mode writes and the fallback sets agree."""
    case = gpu["case"]
    constants = ReferenceConstants.from_case(case)
    reference = evaluate_reference(gpu["positions"], gpu["velocities"], gpu["densities"], gpu["pressures"],
                                   gpu["masses"], gpu["is_wall"], constants)
    fluid = reference.fluid_index
    timestep = constants.timestep

    # density rate: (rho_1 - rho_0) / dt in float64 of the float32 values; rho_1 itself in ulps of rho
    rate_gpu = (gpu["scratch"][fluid, 0].astype(np.float64) - gpu["densities_float32"][fluid].astype(np.float64)) \
        / timestep
    density_ulp = np.spacing(np.abs(gpu["densities_float32"][fluid])).astype(np.float64)
    rate_resolution = density_ulp / timestep
    predicted = gpu["densities_float32"][fluid].astype(np.float64) + timestep * reference.density_rate
    density_ulp_error = np.abs(gpu["scratch"][fluid, 0].astype(np.float64) - predicted) / density_ulp

    stored_gpu = gpu["correction_inverse"][fluid][:, [0, 1, 2, 3, 4, 5]]
    fluid_inverse_gpu = gpu["wall_crossing_fluid_inverse"][fluid]
    comparisons = {
        "density_rate": _error_statistics(rate_gpu, reference.density_rate, reference.density_rate_scale),
        "acceleration": _error_statistics(gpu["acceleration"][fluid], reference.acceleration,
                                          reference.acceleration_scale),
        "shift": _error_statistics(gpu["shift"][fluid], reference.shift, reference.shift_scale),
        "correction_inverse": _error_statistics(stored_gpu.reshape(-1), reference.correction_inverse.reshape(-1)),
        "kernel_sum": _error_statistics(gpu["density_gradient_kernel_sum"][fluid, 3], reference.kernel_sum),
        "fluid_density_gradient_sum": _error_statistics(gpu["density_gradient_kernel_sum"][fluid, 0:3],
                                                        reference.fluid_density_gradient_sum),
        "fluid_inverse": _error_statistics(fluid_inverse_gpu[:, 0:6].reshape(-1),
                                           reference.fluid_inverse.reshape(-1)),
    }
    rate_floor = max(0.01 * float(np.abs(reference.density_rate).max()), 1e-300)
    comparisons["density_rate"]["resolution_of_rate_maximum"] = float(
        (rate_resolution / np.maximum(np.abs(reference.density_rate), rate_floor)).max())
    comparisons["density_rate"]["new_density_error_ulp_maximum"] = float(density_ulp_error.max())
    comparisons["density_rate"]["new_density_error_ulp_percentile_99_9"] = float(np.percentile(density_ulp_error, 99.9))
    # The rate is only readable through rho_1 = rho_0 + dt * rate stored in float32: one ulp of rho is
    # ulp / dt of the rate, whatever the kernel computed. The pass criterion of the rate therefore takes the
    # error beyond one ulp of the stored density (what remains is the kernel's own error).
    beyond_resolution = np.maximum(np.abs(rate_gpu - reference.density_rate) - rate_resolution, 0.0)
    comparisons["density_rate"]["relative_error_beyond_one_ulp_maximum"] = float(
        (beyond_resolution / np.maximum(np.abs(reference.density_rate), rate_floor)).max())

    # the walls: fixed outputs
    wall = np.flatnonzero(gpu["is_wall"])
    rest_density = np.float32(next(material for material in case.materials if material.kind == 1).rest_density)
    wall_problems = []
    expected_inverse = np.tile(np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32), (wall.size, 1))
    if not np.array_equal(gpu["correction_inverse"][wall], expected_inverse):
        wall_problems.append("correction_inverse of a wall is not the identity")
    if np.any(gpu["density_gradient_kernel_sum"][wall] != 0.0):
        wall_problems.append("density gradient / kernel sum of a wall is not zero")
    if not np.array_equal(gpu["scratch"][wall, 0], np.full(wall.size, rest_density, dtype=np.float32)) \
            or np.any(gpu["scratch"][wall, 1] != 0.0):
        wall_problems.append("scratch of a wall is not (rho0, 0)")
    if np.any(gpu["acceleration"][wall] != 0.0) or np.any(gpu["shift"][wall] != 0.0):
        wall_problems.append("force wrote a wall's acceleration or shift")

    flag_gpu = fluid_inverse_gpu[:, 7] != 0.0
    fallback_problems = []
    if not np.array_equal(flag_gpu, reference.fluid_inverse_valid):
        fallback_problems.append(f"A_f validity differs on {int((flag_gpu != reference.fluid_inverse_valid).sum())} rows")
    fallback_gpu = int(gpu["status"]["wall_crossing_gradient_fallback_count"])
    if fallback_gpu != int((~reference.fluid_inverse_valid).sum()):
        fallback_problems.append(f"fallback count {fallback_gpu} (GPU) != "
                                 f"{int((~reference.fluid_inverse_valid).sum())} (reference)")

    # The pass criterion per quantity (all against the same tolerance, 1e-4 by default):
    #   L, A_f^-1, kernel sum, g_f: the per-particle relative error (reference floored at 1 % of its maximum);
    #   density rate: the same beyond one ulp of the stored density (see above);
    #   acceleration and shift: sums in which the pair terms largely cancel, so a particle's result can be
    #     orders below its terms and a relative error says little there; their error is taken against a tenth of
    #     the summed absolute pair terms (i.e. it must stay below tolerance / 10 of that rounding scale), and
    #     the plain relative errors are reported beside it.
    compared = ("correction_inverse", "kernel_sum", "fluid_density_gradient_sum", "fluid_inverse")
    worst = max([comparisons[name]["relative_error_maximum"] for name in compared]
                + [comparisons["density_rate"]["relative_error_beyond_one_ulp_maximum"]]
                + [10.0 * comparisons[name]["error_over_pair_term_sum_maximum"]
                   for name in ("acceleration", "shift")])
    # what the L A = I step of the derivation is worth (reference only): the stored formulas against the wall
    # neighbours' values written out pair by pair
    scale_pressure = np.sqrt((reference.acceleration_pressure ** 2).sum(axis=1)).max()
    derivation = {
        "pressure_pairwise_difference_of_field": float(
            np.sqrt(((reference.acceleration_pressure - reference.acceleration_pressure_pairwise) ** 2)
                    .sum(axis=1)).max() / scale_pressure) if scale_pressure > 0.0 else 0.0,
        "density_diffusion_pairwise_difference_of_field": float(
            np.abs(reference.density_rate_diffusion_wall - reference.density_rate_diffusion_wall_pairwise).max()
            / max(np.abs(reference.density_rate).max(), 1e-300)),
    }
    near_wall = reference.wall_neighbour_count > 0
    return {
        "passed": bool(worst < tolerance and not wall_problems and not fallback_problems),
        "tolerance": tolerance,
        "worst_relative_error": worst,
        "comparisons": comparisons,
        "wall_problems": wall_problems,
        "fallback_problems": fallback_problems,
        "derivation_step": derivation,
        "counts": {
            "fluid": int(fluid.size), "wall": int(wall.size),
            "fluid_with_wall_neighbours": int(near_wall.sum()),
            "fluid_below_distance_floor": int((reference.wall_distance_raw < constants.distance_floor).sum()),
            "fluid_inverse_fallbacks": int((~reference.fluid_inverse_valid).sum()),
            "fluid_inverse_frobenius_capped": int(reference.fluid_inverse_capped.sum()),
            "correction_frobenius_capped": int(reference.correction_capped.sum()),
            "correction_fallbacks": int(reference.correction_fallback.sum()),
            "pairs": reference.notes["pair_count"],
            "positive_pressure_particles": int((gpu["pressures"][fluid] > 0.0).sum()),
        },
        "terms_of_density_rate_maximum": {
            "drift_fluid": float(np.abs(reference.density_rate_drift_fluid).max()),
            "drift_wall": float(np.abs(reference.density_rate_drift_wall).max()),
            "diffusion_fluid": float(np.abs(reference.density_rate_diffusion_fluid).max()),
            "diffusion_wall": float(np.abs(reference.density_rate_diffusion_wall).max()),
        },
        "reference_notes": {key: value for key, value in reference.notes.items() if key != "pair_count"},
    }


def compare_budget_with_reference(gpu: dict, tolerance: float) -> dict:
    """V7_DENSITY_BUDGET=1: the four terms the fused pass accumulated for every fluid particle (one evaluation:
    density_budget = dt x term, in the particle's region) against the reference's drift fluid, drift wall,
    diffusion fluid, diffusion wall. Any wall mode (simple / adami: the reference under which the terms are that
    mode's, see reference.py). Criterion per term: 10 x |error| / the summed absolute pair terms of the whole
    density rate below the tolerance (each term is a sum of cancelling pairs, as the acceleration is; a term's
    own pair sum is no scale: it vanishes where a float32 distance puts a pair on the other side of the support
    radius); beside it the sum of the four terms against the rate the pass stored (beyond one ulp of the stored
    density) and the region."""
    case = gpu["case"]
    constants = ReferenceConstants.from_case(case)
    reference = evaluate_reference(gpu["positions"], gpu["velocities"], gpu["densities"], gpu["pressures"],
                                   gpu["masses"], gpu["is_wall"], constants)
    fluid = reference.fluid_index
    timestep = constants.timestep
    budget = gpu["density_budget"][fluid].astype(np.float64)                      # (F, 9, 4), dt x term
    terms_gpu = budget.sum(axis=1) / timestep
    terms_reference = np.stack([reference.density_rate_drift_fluid, reference.density_rate_drift_wall,
                                reference.density_rate_diffusion_fluid, reference.density_rate_diffusion_wall],
                               axis=1)
    names = ("drift_fluid", "drift_wall", "diffusion_fluid", "diffusion_wall")
    comparisons = {name: _error_statistics(terms_gpu[:, column], terms_reference[:, column],
                                           reference.density_rate_scale)
                   for column, name in enumerate(names)}
    # the sum of the four terms against the rate the pass stored (rho_1 - rho_0) / dt
    rate_gpu = (gpu["scratch"][fluid, 0].astype(np.float64) - gpu["densities_float32"][fluid].astype(np.float64)) \
        / timestep
    rate_resolution = np.spacing(np.abs(gpu["densities_float32"][fluid])).astype(np.float64) / timestep
    closure = np.maximum(np.abs(terms_gpu.sum(axis=1) - rate_gpu) - rate_resolution, 0.0)
    rate_floor = max(0.01 * float(np.abs(rate_gpu).max()), 1e-300)
    rate_reference_error = np.maximum(np.abs(rate_gpu - reference.density_rate) - rate_resolution, 0.0)
    # the region: exactly one region of a fluid particle is written; the host's in float32 as the shader's
    lower, upper = gpu["density_budget_box"]
    positions = gpu["positions"][fluid]
    support = np.float32(constants.smoothing_length)
    horizontal = np.where(positions[:, 0] - np.float32(lower[0]) < support, 1,
                          np.where(np.float32(upper[0]) - positions[:, 0] < support, 2, 0))
    vertical = np.where(positions[:, 1] - np.float32(lower[1]) < support, 1,
                        np.where(np.float32(upper[1]) - positions[:, 1] < support, 2, 0))
    region_host = horizontal + 3 * vertical
    written = np.abs(budget).sum(axis=2) > 0.0                                       # (F, 9)
    region_problems = int(((written.sum(axis=1) > 1) | (written & (np.arange(9)[None, :] != region_host[:, None]))
                           .any(axis=1)).sum())
    wall_rows_written = int(np.abs(gpu["density_budget"][gpu["is_wall"]]).sum() > 0.0)
    # the applied change: exactly the float32 the pass stored minus the float32 it started from
    applied_gpu = gpu["density_budget_applied"][fluid].astype(np.float64).sum(axis=1)
    applied_expected = gpu["scratch"][fluid, 0].astype(np.float64) - gpu["densities_float32"][fluid].astype(np.float64)
    applied_mismatches = int((applied_gpu != applied_expected).sum())
    worst = max(10.0 * comparisons[name]["error_over_pair_term_sum_maximum"] for name in names)
    closure_worst = float((closure / np.maximum(np.abs(rate_gpu), rate_floor)).max())
    return {
        "passed": bool(worst < tolerance and closure_worst < tolerance and region_problems == 0
                       and wall_rows_written == 0 and applied_mismatches == 0),
        "tolerance": tolerance,
        "worst_term_error_over_pair_terms_times_10": worst,
        "comparisons": comparisons,
        "sum_of_terms_against_stored_rate_beyond_one_ulp": closure_worst,
        "stored_rate_against_reference_beyond_one_ulp": float(
            (rate_reference_error / np.maximum(np.abs(reference.density_rate), rate_floor)).max()),
        "region_problems": region_problems,
        "wall_rows_written": wall_rows_written,
        "applied_change_mismatches": applied_mismatches,
        "applied_change_field_mean_per_second": float(applied_gpu.mean() / timestep),
        "terms_sum_field_mean_per_second": float(terms_gpu.sum(axis=1).mean()),
        "region_counts": {str(region): int((region_host == region).sum()) for region in range(9)},
        "terms_maximum": {name: float(np.abs(terms_reference[:, column]).max())
                          for column, name in enumerate(names)},
        "terms_field_mean_reference": {name: float(terms_reference[:, column].mean())
                                       for column, name in enumerate(names)},
        "terms_field_mean_gpu": {name: float(terms_gpu[:, column].mean()) for column, name in enumerate(names)},
    }


def property_residuals(gpu: dict, configuration: str, specification: BoxCaseSpecification) -> dict:
    """Residuals of the exact-solution configurations on the GPU result alone (any wall mode)."""
    case = gpu["case"]
    dimension = specification.dimension
    fluid = np.flatnonzero(~gpu["is_wall"])
    positions = gpu["positions"][fluid].astype(np.float64)
    lower, upper = specification.box_lower(), specification.box_upper()
    support = specification.support_radius()
    gap = np.minimum(positions - lower, upper - positions)[:, :dimension]
    wall_distance = gap.min(axis=1)
    acceleration = gpu["acceleration"][fluid].astype(np.float64)
    acceleration_norm = np.sqrt((acceleration ** 2).sum(axis=1))
    timestep = float(np.float32(case.physics.timestep))
    rate = (gpu["scratch"][fluid, 0].astype(np.float64) - gpu["densities_float32"][fluid].astype(np.float64)) / timestep
    rate_resolution = float(np.spacing(np.float32(specification.rest_density))) / timestep

    def summary(values: np.ndarray, mask: np.ndarray) -> dict:
        selected = values[mask]
        return {"count": int(mask.sum()),
                "maximum": float(selected.max()) if selected.size else None,
                "root_mean_square": float(np.sqrt((selected ** 2).mean())) if selected.size else None}

    if configuration == "couette":
        height = upper[1] - lower[1]
        shear = LID_SPEED / height
        # away from the side walls (x, and z in 3-D): the linear field is exact there
        side_axes = [axis for axis in range(dimension) if axis != 1]
        away = np.all(gap[:, side_axes] > support, axis=1)
        near_plate = away & (gap[:, 1] < support)
        # scales: the viscous acceleration a wall-normal profile of curvature U / (H h) would have, and rho * shear
        viscous_scale = specification.viscosity * shear / support
        return {
            "selection": "fluid farther than h from the side walls",
            "acceleration_over_nu_shear_over_h": {
                "all": summary(acceleration_norm / viscous_scale, away),
                "within_h_of_a_plate": summary(acceleration_norm / viscous_scale, near_plate)},
            "density_rate_over_rho_shear": {
                "all": summary(np.abs(rate) / (specification.rest_density * shear), away),
                "within_h_of_a_plate": summary(np.abs(rate) / (specification.rest_density * shear), near_plate)},
            "density_rate_resolution_over_rho_shear": rate_resolution / (specification.rest_density * shear),
        }
    gravity_norm = float(np.sqrt((np.asarray(specification.gravity) ** 2).sum()))
    near_wall = wall_distance < support
    in_corner = (gap < support).sum(axis=1) >= 2
    return {
        "selection": "every fluid particle",
        "acceleration_over_gravity": {
            "all": summary(acceleration_norm / gravity_norm, np.ones_like(near_wall)),
            "interior": summary(acceleration_norm / gravity_norm, ~near_wall),
            "within_h_of_a_wall": summary(acceleration_norm / gravity_norm, near_wall),
            "within_h_of_two_walls": summary(acceleration_norm / gravity_norm, in_corner)},
        "density_rate_maximum": float(np.abs(rate).max()),
    }


def run_configuration(out_root: pathlib.Path, dimension: int, device_index: int, configuration: str,
                      wall_boundary: str, continuity: str, distance_floor_factor: float, tolerance: float,
                      delta_wall: str = "extension", continuity_wall_component: str = "full",
                      density_budget: bool = False) -> dict:
    label = f"{dimension}d_{configuration}_{wall_boundary}" + (f"_{continuity}" if wall_boundary == "crossing" else "")
    if wall_boundary == "crossing" and delta_wall != "extension":
        label += f"_delta_wall_{delta_wall}"
    if wall_boundary == "crossing" and continuity_wall_component != "full":
        label += f"_wall_component_{continuity_wall_component}"
    if density_budget:
        label += "_budget"
    directory = out_root / label
    specification = box_specification(dimension, configuration, wall_boundary, continuity, distance_floor_factor)
    specification.delta_wall = delta_wall
    specification.continuity_wall_component = continuity_wall_component
    if wall_boundary == "crossing" and delta_wall == "body_force" and configuration == "smooth":
        specification.gravity = BODY_FORCE_SMOOTH_GRAVITY[dimension]
    case_path = write_box_case(directory / "case", specification)
    with LoggedRun(kind="single_step", case=str(case_path.relative_to(REPOSITORY_ROOT)).replace("\\", "/")) as run:
        gpu = gpu_single_evaluation(case_path, device_index, configuration, specification)
        run.record["gpu_uuids"] = [gpu["device_uuid"]]
        outcome = {"label": label, "run_id": run.run_id, "dimension": dimension, "configuration": configuration,
                   "wall_boundary": wall_boundary, "continuity": continuity if wall_boundary == "crossing" else None,
                   "particles": {"fluid": int((~gpu["is_wall"]).sum()), "wall": int(gpu["is_wall"].sum())},
                   "status": {key: int(value) for key, value in gpu["status"].items()
                              if key.startswith("overflow_") or key.endswith("fallback_count")
                              or key == "alive_particle_count"}}
        if configuration == "smooth" and wall_boundary != "crossing":
            # simple / adami: only the budget is compared (the reference's force is the crossing mode's)
            outcome["budget"] = compare_budget_with_reference(gpu, tolerance)
            outcome["passed"] = outcome["budget"]["passed"]
        elif configuration == "smooth":
            outcome["reference"] = compare_with_reference(gpu, tolerance)
            outcome["passed"] = outcome["reference"]["passed"]
            if density_budget:
                outcome["budget"] = compare_budget_with_reference(gpu, tolerance)
                outcome["passed"] = outcome["passed"] and outcome["budget"]["passed"]
        else:
            outcome["residuals"] = property_residuals(gpu, configuration, specification)
            outcome["passed"] = True
            if wall_boundary == "crossing":
                # the reference agrees with the GPU on these fields too (the same comparison, informational)
                outcome["reference"] = compare_with_reference(gpu, tolerance)
        run.extra["label"] = label
        run.extra["passed"] = outcome["passed"]
        run.extra["status"] = "ok" if outcome["passed"] else "failed: comparison above tolerance"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "result.json").write_text(json.dumps(outcome, indent=2), encoding="utf-8", newline="\n")
    return outcome


def _print_outcome(outcome: dict) -> None:
    print(f"\n[single_step] {outcome['label']}  (run {outcome['run_id']}; fluid {outcome['particles']['fluid']:,}, "
          f"wall {outcome['particles']['wall']:,})")
    reference = outcome.get("reference")
    if reference is not None and outcome["configuration"] != "smooth":
        # an exact-solution configuration: its outputs are ~ 0 (differences of large terms), so only the errors
        # in units of the summed pair terms and of the stored density mean something
        comparisons = reference["comparisons"]
        print(f"  GPU vs CPU reference: acceleration error {comparisons['acceleration']['error_over_pair_term_sum_maximum']:.2e} "
              f"of the summed pair terms, new density within {comparisons['density_rate']['new_density_error_ulp_maximum']:.2f} ulp")
    elif reference is not None:
        counts = reference["counts"]
        print(f"  pairs {counts['pairs']:,}; fluid with wall neighbours {counts['fluid_with_wall_neighbours']:,}; "
              f"below the floor {counts['fluid_below_distance_floor']}; A_f fallbacks "
              f"{counts['fluid_inverse_fallbacks']}, Frobenius-capped {counts['fluid_inverse_frobenius_capped']}; "
              f"p > 0 on {counts['positive_pressure_particles']:,}")
        print(f"  {'quantity':28s} {'max rel err':>12s} {'99.9 %':>12s} {'err / max|ref|':>15s} {'err / sum|terms|':>17s}")
        for name, statistics in reference["comparisons"].items():
            scaled = statistics.get("error_over_pair_term_sum_maximum")
            print(f"  {name:28s} {statistics['relative_error_maximum']:12.3e} "
                  f"{statistics['relative_error_percentile_99_9']:12.3e} {statistics['error_of_field_maximum']:15.3e} "
                  + (f"{scaled:17.3e}" if scaled is not None else f"{'':>17s}"))
        rate = reference["comparisons"]["density_rate"]
        print(f"  new density vs reference: max {rate['new_density_error_ulp_maximum']:.2f} ulp "
              f"(99.9 % {rate['new_density_error_ulp_percentile_99_9']:.2f}); one ulp of rho is up to "
              f"{rate['resolution_of_rate_maximum']:.1e} of the rate; rate error beyond one ulp: "
              f"{rate['relative_error_beyond_one_ulp_maximum']:.3e}")
        print(f"  derivation step L A = I (reference only): pressure {reference['derivation_step']['pressure_pairwise_difference_of_field']:.2e}, "
              f"wall density diffusion {reference['derivation_step']['density_diffusion_pairwise_difference_of_field']:.2e} of the field")
        for problem in reference["wall_problems"] + reference["fallback_problems"]:
            print(f"  PROBLEM: {problem}")
        if outcome["configuration"] == "smooth":
            print(f"  -> {'PASS' if reference['passed'] else 'FAIL'} (worst criterion value "
                  f"{reference['worst_relative_error']:.3e}, tolerance {reference['tolerance']:.0e}; criterion: "
                  "relative error of L, A_f^-1, kernel sum, g_f; density rate beyond one ulp; 10 x error / summed "
                  "pair terms of acceleration and shift)")
    budget = outcome.get("budget")
    if budget is not None:
        print(f"  density budget (V7_DENSITY_BUDGET=1): {'term':18s} {'max |ref|':>12s} {'err / sum|pairs|':>17s} "
              f"{'max rel err':>12s} {'field mean ref':>15s} {'field mean GPU':>15s}")
        for name, statistics in budget["comparisons"].items():
            print(f"  {'':38s} {name:18s} {budget['terms_maximum'][name]:12.4e} "
                  f"{statistics['error_over_pair_term_sum_maximum']:17.3e} {statistics['relative_error_maximum']:12.3e} "
                  f"{budget['terms_field_mean_reference'][name]:15.6e} {budget['terms_field_mean_gpu'][name]:15.6e}")
        print(f"  sum of the four terms vs the stored rate (beyond one ulp): "
              f"{budget['sum_of_terms_against_stored_rate_beyond_one_ulp']:.3e}; stored rate vs reference: "
              f"{budget['stored_rate_against_reference_beyond_one_ulp']:.3e}; region mismatches "
              f"{budget['region_problems']}; wall rows written {budget['wall_rows_written']}; applied change != "
              f"stored - initial on {budget['applied_change_mismatches']} rows (field mean "
              f"{budget['applied_change_field_mean_per_second']:+.6e} per second against the terms' "
              f"{budget['terms_sum_field_mean_per_second']:+.6e}); regions "
              f"{budget['region_counts']}")
        print(f"  -> budget {'PASS' if budget['passed'] else 'FAIL'} (worst 10 x term error / summed pair terms "
              f"{budget['worst_term_error_over_pair_terms_times_10']:.3e}, tolerance {budget['tolerance']:.0e})")
    residuals = outcome.get("residuals")
    if residuals is not None:
        for name, groups in residuals.items():
            if isinstance(groups, dict):
                for group, values in groups.items():
                    if isinstance(values, dict) and values.get("maximum") is not None:
                        print(f"  {name} [{group}, n = {values['count']:,}]: max {values['maximum']:.3e}, "
                              f"rms {values['root_mean_square']:.3e}")
            elif isinstance(groups, float):
                print(f"  {name}: {groups:.3e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="wall_boundary crossing: one GPU step against the CPU reference")
    parser.add_argument("--dimension", type=int, choices=(2, 3), required=True)
    parser.add_argument("--device", type=int, required=True,
                        help="v7 device index (discrete-first order; the record stores the device UUID)")
    parser.add_argument("--out", default="logs/wall_crossing/single_step")
    parser.add_argument("--configurations", default=",".join(ALL_CONFIGURATIONS))
    parser.add_argument("--continuity", default="both", choices=("extrapolate", "wall_velocity", "both"))
    parser.add_argument("--distance-floor-factor", type=float, default=0.25)
    parser.add_argument("--tolerance", type=float, default=1.0e-4,
                        help="largest per-particle relative error allowed in the smooth configuration")
    parser.add_argument("--compare-modes", action="store_true",
                        help="also run the property configurations with wall_boundary simple and adami")
    parser.add_argument("--delta-wall", default="extension",
                        choices=("extension", "rest_density", "body_force"),
                        help="the diagnostic numerics.wall_crossing.delta_wall of the crossing runs (default: the "
                             "mode); another value adds its name to the labels and to the summary file name")
    parser.add_argument("--continuity-wall-component", default="full", choices=("full", "normal"),
                        help="the diagnostic numerics.wall_crossing.continuity_wall_component of the crossing runs")
    parser.add_argument("--density-budget", action="store_true",
                        help="diagnostic: run with V7_DENSITY_BUDGET=1 and compare the four budget terms with the "
                             "reference (crossing and, in the smooth configuration, simple)")
    arguments = parser.parse_args()
    if arguments.density_budget:
        import os
        os.environ["V7_DENSITY_BUDGET"] = "1"       # read once when simulator_v7 is imported (first GPU run below)

    out_root = (REPOSITORY_ROOT / arguments.out).resolve()
    configurations = [name.strip() for name in arguments.configurations.split(",") if name.strip()]
    unknown = [name for name in configurations if name not in ALL_CONFIGURATIONS]
    if unknown:
        parser.error(f"unknown configurations {unknown}; known: {', '.join(ALL_CONFIGURATIONS)}")
    continuities = ("extrapolate", "wall_velocity") if arguments.continuity == "both" else (arguments.continuity,)

    outcomes = []
    for configuration in configurations:
        # the continuity rule only acts where the velocity differs from the wall's: smooth and couette
        rules = continuities if configuration in ("smooth", "couette") else continuities[:1]
        for continuity in rules:
            outcomes.append(run_configuration(out_root, arguments.dimension, arguments.device, configuration,
                                              "crossing", continuity, arguments.distance_floor_factor,
                                              arguments.tolerance, arguments.delta_wall,
                                              arguments.continuity_wall_component, arguments.density_budget))
            _print_outcome(outcomes[-1])
        if arguments.density_budget and configuration == "smooth":
            outcomes.append(run_configuration(out_root, arguments.dimension, arguments.device, configuration,
                                              "simple", "extrapolate", arguments.distance_floor_factor,
                                              arguments.tolerance, density_budget=True))
            _print_outcome(outcomes[-1])
        if arguments.compare_modes and configuration in PROPERTY_CONFIGURATIONS:
            for wall_boundary in ("simple", "adami"):
                outcomes.append(run_configuration(out_root, arguments.dimension, arguments.device, configuration,
                                                  wall_boundary, "extrapolate", arguments.distance_floor_factor,
                                                  arguments.tolerance))
                _print_outcome(outcomes[-1])

    summary_path = out_root / (f"summary_{arguments.dimension}d"
                               + ("" if arguments.delta_wall == "extension" else f"_delta_wall_{arguments.delta_wall}")
                               + ("" if arguments.continuity_wall_component == "full"
                                  else f"_wall_component_{arguments.continuity_wall_component}")
                               + ("_budget" if arguments.density_budget else "")
                               + ".json")
    summary_path.write_text(json.dumps(outcomes, indent=2), encoding="utf-8", newline="\n")
    failed = [outcome["label"] for outcome in outcomes if not outcome["passed"]]
    print(f"\n[single_step] {len(outcomes)} run(s), summary {summary_path}")
    print("[single_step] " + ("ALL PASS" if not failed else f"FAILED: {', '.join(failed)}"))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
