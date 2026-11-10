"""
micropolar_tests.py - the GPU checks of the micropolar terms (docs/micropolar_design.md, tests T1 - T5).

Sub-commands:

  dump        Run a case on a K-slab chain for --steps steps from its initial condition and save the own, alive
              particles' position / velocity / density / pressure / material (+ angular_velocity on a micropolar
              case) as one .npz. --repo-root selects the source tree the solver is imported from, so the same
              command runs the reference build (a checkout of the tag micropolar-base) and this tree.
  compare     Comparison of dumps on fixed sample points (Shepard interpolation of the fluid's u, v, rho and w_z
              with the Wendland C4 kernel of radius h, on a regular grid inside the cavity): the reference
              dumps give the run-to-run rounding floor (runs are not bit-reproducible: atomic appends change the
              summation order, and single particles near the lid corners end up a fraction of a spacing apart
              after a few thousand steps, so particles cannot be matched one to one), each test dump is compared
              with every reference repeat.
  operators   T3: one force evaluation (the bootstrap) on prescribed states with uniform density, on a lattice and
              a jittered lattice; reads back acceleration / angular_acceleration, interior particles only.
  relaxation  T4: v = 0, uniform w, xi = 0: w against the scheme's exact product and against exp(-2 c t / Theta).
  restart     The relaxation state run N steps, saved with cavity_runner.read_restart_state, restarted in a new
              simulator (restart_init: no force pass) and run N more: w must continue the same product, which
              it does only if the state carries angular_acceleration. Also: a state without the two micropolar
              fields is refused.

Run from the repository root:
    .venv/Scripts/python.exe -m experiment.validation.micropolar_tests <sub-command> ...
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

_THIS_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

_OVERFLOW_PREFIX = "overflow_"


def _import_solver(repo_root: pathlib.Path):
    """The solver modules of the tree at ``repo_root`` (this tree, or a checkout of the reference tag)."""
    root = str(repo_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7.utils.orchestrator_v7 import ChainOrchestratorV7
    from experiment.v7.utils.partition_v7 import compute_chain_partition
    from experiment.v7.utils.simulator_v7 import SphSimulatorV7
    from experiment.v7.utils.vulkan_context_v7 import VulkanContextV7
    import experiment.v7.utils.simulator_v7 as simulator_module
    loaded_from = pathlib.Path(simulator_module.__file__).resolve()
    if repo_root.resolve() not in loaded_from.parents:
        raise RuntimeError(f"solver imported from {loaded_from}, not from {repo_root}")
    return load_case_v7, compute_chain_partition, SphSimulatorV7, VulkanContextV7, ChainOrchestratorV7


def _read_own_particles(sim, field_names) -> dict:
    """Own, alive rows of the named set-0 buffers of one sim."""
    component_counts = {"position_voxel_id": 4, "velocity_mass": 4, "density_pressure": 2, "acceleration": 4,
                        "angular_velocity": 4, "angular_acceleration": 4, "shift": 4,
                        "wall_dummy_velocity": 4, "wall_dummy_angular_velocity": 4}
    capacities = sim.case.capacities
    pool_capacity = capacities.total_pool_capacity()
    names = list(dict.fromkeys(["position_voxel_id", "velocity_mass", "material", *field_names]))
    raw = sim.readback_buffers_batch(names)
    own = slice(sim.own_first_pid(), sim.own_first_pid() + capacities.own_pool_size)
    rows = {}
    for name in names:
        if name == "material":
            rows[name] = np.frombuffer(raw[name], np.uint32)[:pool_capacity][own]
        else:
            count = component_counts[name]
            rows[name] = np.frombuffer(raw[name], np.float32)[:pool_capacity * count].reshape(pool_capacity, count)[own]
    alive = (rows["velocity_mass"][:, 3] > 0) & (rows["position_voxel_id"][:, 3] > 0.5)
    return {name: np.array(values[alive]) for name, values in rows.items()}


def _run_chain(solver, global_case, slab_count: int, device_map, weights, steps: int, field_names,
               validation: bool, depth: int = 2) -> tuple[dict, dict]:
    """Bootstrap a K-slab chain of ``global_case``, run ``steps`` steps, return (merged own rows, invariants)."""
    _load_case, compute_chain_partition, SphSimulatorV7, VulkanContextV7, ChainOrchestratorV7 = solver
    chain = compute_chain_partition(global_case, list(weights), pool_safety=1.2)
    contexts, sims = [], []
    try:
        for index in range(slab_count):
            contexts.append(VulkanContextV7.create(device_index=device_map[index], enable_validation=validation,
                                                   application_name=f"micropolar_tests_k{slab_count}_s{index}"))
            sims.append(SphSimulatorV7(contexts[-1], chain.slabs[index], sync_scheme="per-direction"))
        with ChainOrchestratorV7(sims, defrag_cadence=global_case.numerics.defrag_cadence) as orchestrator:
            orchestrator.bootstrap_all()
            if steps > 0:
                orchestrator.run_pipelined(steps, depth=depth, warmup=0)
            parts = [_read_own_particles(sim, field_names) for sim in sims]
            statuses = [sim.readback_global_status() for sim in sims]
    finally:
        for sim in sims:
            sim.destroy()
        for context in contexts:
            context.destroy()
    merged = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
    overflow = {}
    for status in statuses:
        for name, value in status.items():
            if name.startswith(_OVERFLOW_PREFIX) or name in ("far_migration_count", "stamp_error_count"):
                overflow[name] = overflow.get(name, 0) + int(value)
    invariants = {"particles_loaded": int(global_case.initial.positions.shape[0]),
                  "particles_alive": int(merged["material"].shape[0]),
                  "slab_particles": [int(part["material"].shape[0]) for part in parts],
                  "counters": overflow}
    return merged, invariants


# ============================================================================
# dump
# ============================================================================

def command_dump(arguments) -> int:
    repo_root = pathlib.Path(arguments.repo_root).resolve()
    solver = _import_solver(repo_root)
    load_case = solver[0]
    global_case = load_case(repo_root / arguments.case)
    device_map = [int(value) for value in arguments.device_map.split(",")]
    weights = [float(value) for value in arguments.weights.split(",")] if arguments.weights else [1.0] * arguments.slabs
    if len(device_map) != arguments.slabs or len(weights) != arguments.slabs:
        sys.exit("--device-map / --weights need one entry per slab")
    micropolar = bool(getattr(global_case.physics, "micropolar_enabled", False))
    field_names = ["density_pressure"] + (["angular_velocity"] if micropolar else [])
    rows, invariants = _run_chain(solver, global_case, arguments.slabs, device_map, weights, arguments.steps,
                                  field_names, arguments.validation, arguments.depth)
    out_path = pathlib.Path(arguments.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"repo_root": str(repo_root), "case": arguments.case, "slabs": arguments.slabs, "device_map": device_map,
            "weights": weights, "steps": arguments.steps, "depth": arguments.depth, "micropolar": micropolar,
            "timestep": float(global_case.physics.timestep), **invariants}
    np.savez(out_path, _meta=json.dumps(meta), **rows)
    conserved = invariants["particles_alive"] == invariants["particles_loaded"]
    counters_zero = all(value == 0 for value in invariants["counters"].values())
    print(f"[dump] {out_path.name}: {invariants['particles_alive']:,} of {invariants['particles_loaded']:,} particles "
          f"alive ({'conserved' if conserved else 'NOT CONSERVED'}), counters "
          f"{'all zero' if counters_zero else invariants['counters']}")
    return 0 if conserved and counters_zero else 1


# ============================================================================
# compare
# ============================================================================

def _load_dump(path) -> tuple[dict, dict]:
    with np.load(path) as archive:
        meta = json.loads(str(archive["_meta"]))
        rows = {name: np.array(archive[name]) for name in archive.files if name != "_meta"}
    return rows, meta


def _sample_points(half_extent: float, count: int) -> np.ndarray:
    axis = np.linspace(-half_extent, half_extent, count)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    return np.column_stack([grid_x.ravel(), grid_y.ravel()])


def _sampled_fields(rows: dict, points: np.ndarray, smoothing_length: float, fluid_group: int) -> dict:
    """Shepard interpolation (Wendland C4, support radius ``smoothing_length``) of the fluid particles' u, v, rho
    and, when present, w_z at ``points`` (float64)."""
    from scipy.spatial import cKDTree
    fluid = rows["material"] == fluid_group
    positions = rows["position_voxel_id"][fluid, :2].astype(np.float64)
    fields = {"u": rows["velocity_mass"][fluid, 0], "v": rows["velocity_mass"][fluid, 1],
              "rho": rows["density_pressure"][fluid, 0]}
    if "angular_velocity" in rows:
        fields["w_z"] = rows["angular_velocity"][fluid, 2]
    tree = cKDTree(positions)
    pairs = tree.query_ball_point(points, smoothing_length)
    sampled = {name: np.zeros(points.shape[0]) for name in fields}
    for point_index, neighbours in enumerate(pairs):
        neighbours = np.asarray(neighbours, dtype=np.int64)
        q = np.sqrt(((positions[neighbours] - points[point_index]) ** 2).sum(axis=1)) / smoothing_length
        weight = (1.0 - q) ** 6 * (35.0 / 3.0 * q * q + 6.0 * q + 1.0)
        total = weight.sum()
        for name, values in fields.items():
            sampled[name][point_index] = (weight * values[neighbours].astype(np.float64)).sum() / total
    return sampled


def _sampled_difference(reference: dict, other: dict) -> dict:
    result = {}
    for name in reference:
        if name not in other:
            continue
        difference = np.abs(reference[name] - other[name])
        result[name + "_max"] = float(difference.max())
        result[name + "_rms"] = float(np.sqrt((difference ** 2).mean()))
    return result


def command_compare(arguments) -> int:
    """Floor = the pairwise rms differences among the reference repeats (their largest is the floor: the repeats of
    one build fall into more than one group after a few thousand steps, so a single pair can under-estimate it);
    a test dump passes when its rms difference to every reference stays within --ratio-limit times that floor."""
    import itertools
    points = _sample_points(arguments.half_extent, arguments.points_per_side)

    def sample(dump_path):
        rows, meta = _load_dump(dump_path)
        return _sampled_fields(rows, points, arguments.smoothing_length, arguments.fluid_group), rows, meta

    references = [sample(dump_path) for dump_path in arguments.reference]
    meta_reference = references[0][2]
    quantities = ("u", "v", "rho", "w_z")
    floor = {}
    for name in quantities:
        if any(name not in sampled for sampled, _rows, _meta in references):
            continue
        pairs = [_sampled_difference({name: first[0][name]}, {name: second[0][name]})[name + "_rms"]
                 for first, second in itertools.combinations(references, 2)]
        floor[name] = {"pair_rms_max": float(max(pairs)), "pair_rms_median": float(np.median(pairs)),
                       "pair_rms_min": float(min(pairs)), "pairs": len(pairs)}
    report = {"reference": [str(dump_path) for dump_path in arguments.reference], "floor": floor, "tests": {}}
    print(f"reference: {meta_reference['case']}, K = {meta_reference['slabs']}, {meta_reference['steps']} steps, "
          f"tree {pathlib.Path(meta_reference['repo_root']).name}, {len(references)} repeats "
          f"({len(references) * (len(references) - 1) // 2} pairs); rms over {points.shape[0]} sample points")
    exit_code = 0
    for test_path in arguments.test:
        sampled, rows, meta = sample(test_path)
        if rows["material"].shape != references[0][1]["material"].shape:
            raise ValueError(f"{test_path}: {rows['material'].shape[0]} particles, the reference has "
                             f"{references[0][1]['material'].shape[0]}")
        print(f"\ntest {pathlib.Path(test_path).name} ({meta['case']}, K = {meta['slabs']}, tree "
              f"{pathlib.Path(meta['repo_root']).name})")
        print(f"{'quantity':<10}{'floor min':>12}{'floor med':>12}{'floor max':>12}{'test min':>12}{'test med':>12}"
              f"{'test max':>12}{'max ratio':>11}")
        record = {}
        for name in quantities:
            if name not in floor or name not in sampled:
                continue
            against = [_sampled_difference({name: reference[0][name]}, {name: sampled[name]})[name + "_rms"]
                       for reference in references]
            ratio = max(against) / floor[name]["pair_rms_max"] if floor[name]["pair_rms_max"] else float("nan")
            record[name] = {"rms_min": float(min(against)), "rms_median": float(np.median(against)),
                            "rms_max": float(max(against)), "ratio_to_floor_max": float(ratio)}
            print(f"{name:<10}{floor[name]['pair_rms_min']:>12.3e}{floor[name]['pair_rms_median']:>12.3e}"
                  f"{floor[name]['pair_rms_max']:>12.3e}{min(against):>12.3e}{float(np.median(against)):>12.3e}"
                  f"{max(against):>12.3e}{ratio:>11.2f}")
            if ratio > arguments.ratio_limit:
                exit_code = 1
        if "angular_velocity" in rows:
            largest = float(np.abs(rows["angular_velocity"][:, :3]).max())
            record["angular_velocity_abs_max"] = largest
            print(f"max |w| over all particles of the test dump: {largest:.3e}")
        report["tests"][str(test_path)] = record
    if arguments.json:
        pathlib.Path(arguments.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n{'PASS' if exit_code == 0 else 'FAIL'}: limit = {arguments.ratio_limit} x the largest reference-pair rms")
    return exit_code


# ============================================================================
# T3 / T4: prescribed states on a fluid block
# ============================================================================

def _fluid_block_case(solver, case_path: pathlib.Path, half_width: float, jitter: float, seed: int,
                      regularization_xi, viscosity: float, viscosity_transfer: float, viscosity_rotation: float,
                      micro_inertia: float, eps_h_squared=None):
    """The case at ``case_path`` with its particles replaced by a square block of fluid, lattice spacing = the
    case's, centred on the origin, optionally jittered by ``jitter`` spacings (uniform in each direction), and
    with the micropolar terms switched on with the given coefficients. Returns (case, fluid group id, spacing)."""
    load_case = solver[0]
    case = load_case(case_path)
    from experiment.v7.utils.case_v7 import KIND_FLUID
    fluid_group = next(index for index, material in enumerate(case.materials) if material.kind == KIND_FLUID)
    material = case.materials[fluid_group]
    spacing = 2.0 * material.radius
    count = int(round(2.0 * half_width / spacing)) + 1
    axis = (np.arange(count) - (count - 1) / 2.0) * spacing
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    positions = np.zeros((count * count, 3), dtype=np.float64)
    positions[:, 0], positions[:, 1] = grid_x.ravel(), grid_y.ravel()
    if jitter > 0.0:
        generator = np.random.default_rng(seed)
        positions[:, :2] += generator.uniform(-jitter, jitter, size=(count * count, 2)) * spacing
    case.initial = type(case.initial)(
        positions=positions.astype(np.float32),
        velocities=np.zeros((count * count, 3), dtype=np.float32),
        material_group=np.full(count * count, fluid_group, dtype=np.uint32),
        angular_velocities=np.zeros((count * count, 3), dtype=np.float32))
    if regularization_xi is not None:
        case.numerics.regularization_xi = float(regularization_xi)
    if eps_h_squared is not None:
        case.numerics.eps_h_squared = float(eps_h_squared)
    case.numerics.wall_boundary = "simple"
    case.physics.micropolar_enabled = True
    case.physics.micro_inertia = float(micro_inertia)
    material.viscosity = float(viscosity)
    material.viscosity_transfer = float(viscosity_transfer)
    material.viscosity_rotation = float(viscosity_rotation)
    return case, fluid_group, spacing


def _interior(positions: np.ndarray, half_width: float, margin: float) -> np.ndarray:
    return (np.abs(positions[:, 0]) < half_width - margin) & (np.abs(positions[:, 1]) < half_width - margin)


def _relative(values: np.ndarray, scale: float) -> tuple[float, float]:
    magnitude = np.sqrt((values.astype(np.float64) ** 2).sum(axis=-1)) if values.ndim > 1 else np.abs(values)
    return float(magnitude.max() / scale), float(np.sqrt((magnitude ** 2).mean()) / scale)


_OPERATOR_COEFFICIENTS = dict(viscosity=2.5e-3, viscosity_transfer=-0.625e-3, viscosity_rotation=3.75e-6,
                              micro_inertia=1.6e-3)


def _operator_state(solver, arguments, case_path, half_width, jitter, regularization_xi, velocity_of, spin_of,
                    eps_h_squared=None):
    case, _group, spacing = _fluid_block_case(solver, case_path, half_width, jitter, arguments.seed,
                                              regularization_xi, eps_h_squared=eps_h_squared,
                                              **_OPERATOR_COEFFICIENTS)
    positions = case.initial.positions.astype(np.float64)
    case.initial.velocities[:] = velocity_of(positions).astype(np.float32)
    case.initial.angular_velocities[:] = spin_of(positions).astype(np.float32)
    rows, invariants = _run_chain(solver, case, 1, [arguments.device], [1.0], 0,
                                  ["acceleration", "angular_acceleration", "angular_velocity"],
                                  arguments.validation)
    if invariants["particles_alive"] != invariants["particles_loaded"]:
        raise RuntimeError(f"operator state lost particles: {invariants}")
    return case, rows, spacing


def command_operators(arguments) -> int:
    solver = _import_solver(_THIS_REPO_ROOT)
    case_path = _THIS_REPO_ROOT / arguments.case
    half_width = arguments.half_width
    coefficients = _OPERATOR_COEFFICIENTS
    # the float32 coefficients of the shader
    viscosity32 = np.float32(coefficients["viscosity"])
    coupling = float(viscosity32 + np.float32(coefficients["viscosity_transfer"]))
    rotation = float(np.float32(coefficients["viscosity_rotation"]))
    inertia = float(np.float32(coefficients["micro_inertia"]))
    omega = 1.0
    gradient = np.array([0.7, -1.3])                      # w_z = gradient . (x, y) [1/s per m]
    results = []

    def zeros(positions):
        return np.zeros((positions.shape[0], 3))

    def rigid_rotation(positions):
        velocity = np.zeros((positions.shape[0], 3))
        velocity[:, 0], velocity[:, 1] = -omega * positions[:, 1], omega * positions[:, 0]
        return velocity

    def linear_spin(positions):
        spin = np.zeros((positions.shape[0], 3))
        spin[:, 2] = positions[:, :2] @ gradient
        return spin

    configurations = [("lattice", 0.0, 0.0), ("jittered", arguments.jitter, 0.0)]
    if arguments.default_regularization:
        configurations += [("lattice, default xi", 0.0, None), ("jittered, default xi", arguments.jitter, None)]
    for label, jitter, regularization_xi in configurations:
        # --- row 1: rigid rotation, w = 0: Theta alpha / c = +2 Omega z, a = 0
        case, rows, spacing = _operator_state(solver, arguments, case_path, half_width, jitter, regularization_xi,
                                              rigid_rotation, zeros)
        interior = _interior(rows["position_voxel_id"], half_width, arguments.margin * case.physics.smoothing_length)
        scaled = rows["angular_acceleration"][interior, :3].astype(np.float64) * inertia / coupling
        expected = np.array([0.0, 0.0, 2.0 * omega])
        error_max, error_rms = _relative(scaled - expected, 2.0 * omega)
        mean_z = float(scaled[:, 2].mean())
        acceleration_max, acceleration_rms = _relative(rows["acceleration"][interior, :3], 1.0)
        results.append({"state": "rigid rotation", "particles": label, "interior": int(interior.sum()),
                        "regularization_xi": case.numerics.regularization_xi,
                        "theta_alpha_over_c_mean_z": mean_z, "relative_error_max": error_max,
                        "relative_error_rms": error_rms, "acceleration_max": acceleration_max,
                        "acceleration_rms": acceleration_rms})
        print(f"[T3] rigid rotation, {label} (xi_reg = {case.numerics.regularization_xi:g}, {interior.sum():,} "
              f"interior): Theta alpha_z / c mean = {mean_z:+.7f} (expected {2.0 * omega:+.1f}), relative error max "
              f"{error_max:.2e} rms {error_rms:.2e}; |a| max {acceleration_max:.2e} rms {acceleration_rms:.2e} m/s^2")

        # --- row 2: w_z linear, v = 0: a / c = curl w = (dw/dy, -dw/dx, 0); Theta alpha + 2 c w = xi lap w = 0
        case, rows, spacing = _operator_state(solver, arguments, case_path, half_width, jitter, regularization_xi,
                                              zeros, linear_spin)
        interior = _interior(rows["position_voxel_id"], half_width, arguments.margin * case.physics.smoothing_length)
        curl_expected = np.array([gradient[1], -gradient[0], 0.0])
        curl_measured = rows["acceleration"][interior, :3].astype(np.float64) / coupling
        curl_max, curl_rms = _relative(curl_measured - curl_expected, float(np.linalg.norm(curl_expected)))
        spin = rows["angular_velocity"][interior, :3].astype(np.float64)
        diffusion = (rows["angular_acceleration"][interior, :3].astype(np.float64) * inertia
                     + 2.0 * coupling * spin)                                   # = xi lap w
        laplacian = diffusion[:, 2] / rotation
        # scale: |grad w| / dx, the size of one pair's contribution to the Laplacian
        laplacian_scale = float(np.linalg.norm(gradient)) / spacing
        laplacian_max, laplacian_rms = _relative(laplacian, laplacian_scale)
        results.append({"state": "linear w_z", "particles": label, "interior": int(interior.sum()),
                        "regularization_xi": case.numerics.regularization_xi,
                        "curl_relative_error_max": curl_max, "curl_relative_error_rms": curl_rms,
                        "laplacian_over_gradient_per_spacing_max": laplacian_max,
                        "laplacian_over_gradient_per_spacing_rms": laplacian_rms})
        print(f"[T3] linear w_z, {label} (xi_reg = {case.numerics.regularization_xi:g}): a / c against curl w "
              f"relative error max {curl_max:.2e} rms {curl_rms:.2e}; lap w / (|grad w| / dx) max "
              f"{laplacian_max:.2e} rms {laplacian_rms:.2e}")

    # --- row 3: w_z = sin(k x) on the lattice at three resolutions (fixed h / dx): lap w against -k^2 w
    # with the case's EPS_H_SQUARED (a fixed fraction of h^2: a constant factor on the operator) and with 0
    wave_number = 2.0 * np.pi / arguments.wavelength

    def sine_spin(positions):
        spin = np.zeros((positions.shape[0], 3))
        spin[:, 2] = np.sin(wave_number * positions[:, 0])
        return spin

    for eps_label, eps_h_squared in (("case EPS_H_SQUARED", None), ("EPS_H_SQUARED = 0", 0.0)):
      previous = None
      ratios = []
      for resolution_case in arguments.resolution_cases:
        case, rows, spacing = _operator_state(solver, arguments, _THIS_REPO_ROOT / resolution_case,
                                              arguments.sine_half_width, 0.0, 0.0, zeros, sine_spin,
                                              eps_h_squared=eps_h_squared)
        interior = _interior(rows["position_voxel_id"], arguments.sine_half_width,
                             arguments.margin * case.physics.smoothing_length)
        spin = rows["angular_velocity"][interior, 2].astype(np.float64)
        laplacian = (rows["angular_acceleration"][interior, 2].astype(np.float64) * inertia
                     + 2.0 * coupling * spin) / rotation
        exact = -wave_number ** 2 * spin
        # least-squares amplitude ratio (the operator's transfer factor at this k) and the L2 error
        ratio = float((laplacian * exact).sum() / (exact * exact).sum())
        error = float(np.sqrt(((laplacian - exact) ** 2).sum() / (exact ** 2).sum()))
        order = (None if previous is None
                 else float(np.log(previous[1] / error) / np.log(previous[0] / spacing)))
        ratios.append(ratio)
        results.append({"state": "sin(k x)", "eps": eps_label, "case": resolution_case, "spacing": spacing,
                        "smoothing_length": case.physics.smoothing_length,
                        "eps_h_squared": case.numerics.eps_h_squared, "interior": int(interior.sum()),
                        "k_h": wave_number * case.physics.smoothing_length, "amplitude_ratio": ratio,
                        "relative_l2_error": error, "observed_order": order})
        print(f"[T3] sin(k x), {eps_label}, dx = {spacing:.4g} m (k h = "
              f"{wave_number * case.physics.smoothing_length:.3f}, "
              f"{interior.sum():,} interior): lap w / (-k^2 w) = {ratio:.6f}, relative L2 error {error:.3e}"
              + ("" if order is None else f", observed order {order:.2f}"))
        previous = (spacing, error)
      if len(ratios) == 3 and ratios[2] != ratios[1]:
        # three resolutions, spacing halved each time: Richardson on the amplitude ratio R(dx) = R_0 + C dx^p
        observed = float(np.log2(abs((ratios[1] - ratios[0]) / (ratios[2] - ratios[1]))))
        limit = ratios[2] + (ratios[2] - ratios[1]) / (2.0 ** observed - 1.0)
        results.append({"state": "sin(k x) Richardson", "eps": eps_label, "observed_order": observed,
                        "amplitude_ratio_limit": limit})
        print(f"[T3] sin(k x), {eps_label}: amplitude ratio {ratios[0]:.6f} / {ratios[1]:.6f} / {ratios[2]:.6f}, "
              f"Richardson order {observed:.2f}, limit dx -> 0 at fixed h/dx = {limit:.6f} "
              f"(constant factor {limit - 1.0:+.4%})")
    if arguments.json:
        pathlib.Path(arguments.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(arguments.json).write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


def command_relaxation(arguments) -> int:
    solver = _import_solver(_THIS_REPO_ROOT)
    case_path = _THIS_REPO_ROOT / arguments.case
    viscosity, viscosity_transfer = 2.5e-3, -0.625e-3
    coupling = float(np.float32(viscosity) + np.float32(viscosity_transfer))
    initial_spin = 3.0
    results = []
    for steps in arguments.steps:
        case, _group, _spacing = _fluid_block_case(solver, case_path, arguments.half_width, 0.0, arguments.seed, None,
                                                   viscosity=viscosity, viscosity_transfer=viscosity_transfer,
                                                   viscosity_rotation=0.0, micro_inertia=arguments.micro_inertia)
        timestep = float(np.float32(case.physics.timestep))
        inertia = float(np.float32(arguments.micro_inertia))
        case.initial.angular_velocities[:, 2] = initial_spin
        rows, invariants = _run_chain(solver, case, 1, [arguments.device], [1.0], steps,
                                      ["angular_velocity"], arguments.validation)
        if invariants["particles_alive"] != invariants["particles_loaded"]:
            raise RuntimeError(f"relaxation run lost particles: {invariants}")
        spin = rows["angular_velocity"][:, 2].astype(np.float64)
        rate = 2.0 * coupling / inertia
        # stored after N steps: w_{N-1/2} = w_0 (1 - rate dt / 2) (1 - rate dt)^(N-1)
        discrete = initial_spin * (1.0 - 0.5 * rate * timestep) * (1.0 - rate * timestep) ** (steps - 1)
        exponential = initial_spin * np.exp(-rate * (steps - 0.5) * timestep)
        in_plane = float(np.abs(rows["angular_velocity"][:, :2]).max())
        velocity = float(np.abs(rows["velocity_mass"][:, :3]).max())
        record = {"steps": steps, "rate_times_timestep": rate * timestep, "mean": float(spin.mean()),
                  "spread": float(spin.max() - spin.min()), "discrete": discrete, "exponential": exponential,
                  "relative_to_discrete_max": float(np.abs(spin / discrete - 1.0).max()),
                  "relative_to_exponential": float(spin.mean() / exponential - 1.0),
                  "expected_first_order_difference": float(discrete / exponential - 1.0),
                  "in_plane_spin_max": in_plane, "velocity_max": velocity}
        results.append(record)
        print(f"[T4] {steps} steps (2 c dt / Theta = {rate * timestep:.4g}): w_z mean {record['mean']:.7f}, "
              f"spread {record['spread']:.2e}; scheme's product {discrete:.7f} (max relative deviation "
              f"{record['relative_to_discrete_max']:.2e}); exp {exponential:.7f} (relative "
              f"{record['relative_to_exponential']:+.3e}, expected {record['expected_first_order_difference']:+.3e}); "
              f"max |w_xy| {in_plane:.1e}, max |v| {velocity:.1e}")
    if arguments.json:
        pathlib.Path(arguments.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(arguments.json).write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


def command_restart(arguments) -> int:
    solver = _import_solver(_THIS_REPO_ROOT)
    _load_case, compute_chain_partition, SphSimulatorV7, VulkanContextV7, ChainOrchestratorV7 = solver
    from experiment.validation.cavity_runner import read_restart_state
    viscosity, viscosity_transfer, initial_spin = 2.5e-3, -0.625e-3, 3.0
    coupling = float(np.float32(viscosity) + np.float32(viscosity_transfer))
    steps = arguments.steps

    def build():
        case, _group, _spacing = _fluid_block_case(solver, _THIS_REPO_ROOT / arguments.case, arguments.half_width,
                                                   0.0, arguments.seed, None, viscosity=viscosity,
                                                   viscosity_transfer=viscosity_transfer, viscosity_rotation=0.0,
                                                   micro_inertia=arguments.micro_inertia)
        case.initial.angular_velocities[:, 2] = initial_spin
        chain = compute_chain_partition(case, [1.0], pool_safety=1.2)
        context = VulkanContextV7.create(device_index=arguments.device, enable_validation=arguments.validation,
                                         application_name="micropolar_tests_restart")
        return case, context, SphSimulatorV7(context, chain.slabs[0], sync_scheme="per-direction")

    case, context, sim = build()
    try:
        with ChainOrchestratorV7([sim], defrag_cadence=case.numerics.defrag_cadence) as orchestrator:
            orchestrator.bootstrap_all()
            orchestrator.run_pipelined(steps, depth=2, warmup=0)
            state = read_restart_state(sim)
            offset = sim.stored_density_offset()
    finally:
        sim.destroy()
        context.destroy()
    print(f"[restart] state after {steps} steps: fields {sorted(state)}")

    case, context, sim = build()
    refused = False
    try:
        with ChainOrchestratorV7([sim], defrag_cadence=case.numerics.defrag_cadence) as orchestrator:
            newtonian_state = {name: values for name, values in state.items()
                               if name not in ("angular_velocity", "angular_acceleration")}
            try:
                sim.restart_init(newtonian_state, offset)
            except ValueError as error:
                refused = True
                print(f"[restart] a state without the micropolar fields is refused: {error}")
            orchestrator.restart_all([state], offset)
            orchestrator.run_pipelined(steps, depth=2, warmup=0)
            rows = _read_own_particles(sim, ["angular_velocity"])
    finally:
        sim.destroy()
        context.destroy()
    timestep = float(np.float32(case.physics.timestep))
    rate = 2.0 * coupling / float(np.float32(arguments.micro_inertia))
    total = 2 * steps
    discrete = initial_spin * (1.0 - 0.5 * rate * timestep) * (1.0 - rate * timestep) ** (total - 1)
    spin = rows["angular_velocity"][:, 2].astype(np.float64)
    deviation = float(np.abs(spin / discrete - 1.0).max())
    # without alpha_n in the state the first step after the restart would not kick: one factor (1 - rate dt) missing
    missing_kick = rate * timestep
    passed = refused and deviation < 0.01 * missing_kick and rows["material"].shape[0] == state["material"].shape[0]
    print(f"[restart] {steps} + {steps} steps: w_z mean {spin.mean():.7f}, the uninterrupted product {discrete:.7f}, "
          f"max relative deviation {deviation:.2e} (a lost kick would be {missing_kick:.2e}); "
          f"{'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    dump = commands.add_parser("dump")
    dump.add_argument("--repo-root", default=str(_THIS_REPO_ROOT))
    dump.add_argument("--case", required=True, help="case.yaml, relative to --repo-root")
    dump.add_argument("--slabs", type=int, default=1)
    dump.add_argument("--device-map", default="0")
    dump.add_argument("--weights", default=None)
    dump.add_argument("--steps", type=int, required=True)
    dump.add_argument("--depth", type=int, default=2)
    dump.add_argument("--out", required=True)
    dump.add_argument("--validation", action="store_true", help="enable the Vulkan validation layer")
    dump.set_defaults(handler=command_dump)

    compare = commands.add_parser("compare")
    compare.add_argument("--reference", nargs="+", required=True,
                         help="dumps of repeats of the reference run (at least two; ten is cheap and covers both "
                              "groups the repeats fall into)")
    compare.add_argument("--test", nargs="+", required=True)
    compare.add_argument("--smoothing-length", type=float, required=True, help="h of the case (support radius)")
    compare.add_argument("--half-extent", type=float, default=0.48, help="sample grid = [-e, e]^2")
    compare.add_argument("--points-per-side", type=int, default=49)
    compare.add_argument("--fluid-group", type=int, default=0, help="material group id of the fluid")
    compare.add_argument("--ratio-limit", type=float, default=2.0,
                         help="largest accepted (test rms difference) / (reference repeat rms difference) "
                              "on the sample points")
    compare.add_argument("--json", default=None)
    compare.set_defaults(handler=command_compare)

    operators = commands.add_parser("operators")
    operators.add_argument("--case", default="cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025/case.yaml")
    operators.add_argument("--resolution-cases", nargs="+", default=[
        "cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025/case.yaml",
        "cases/lid_driven_cavity_2d_n500_xi0p001_eps0p0025/case.yaml",
        "cases/lid_driven_cavity_2d_n1000_xi0p001_eps0p0025/case.yaml"])
    operators.add_argument("--half-width", type=float, default=0.3)
    operators.add_argument("--sine-half-width", type=float, default=0.15)
    operators.add_argument("--wavelength", type=float, default=0.2)
    operators.add_argument("--jitter", type=float, default=0.1, help="jitter amplitude in particle spacings")
    operators.add_argument("--margin", type=float, default=1.2, help="interior = farther than margin * h from the edge")
    operators.add_argument("--default-regularization", action="store_true",
                           help="also run the linear-field rows with the case's regularization xi")
    operators.add_argument("--seed", type=int, default=20261110)
    operators.add_argument("--device", type=int, default=0)
    operators.add_argument("--validation", action="store_true")
    operators.add_argument("--json", default=None)
    operators.set_defaults(handler=command_operators)

    relaxation = commands.add_parser("relaxation")
    relaxation.add_argument("--case", default="cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025/case.yaml")
    relaxation.add_argument("--half-width", type=float, default=0.2)
    relaxation.add_argument("--micro-inertia", type=float, default=2.25e-5)
    relaxation.add_argument("--steps", type=int, nargs="+", default=[1, 100, 300])
    relaxation.add_argument("--seed", type=int, default=20261110)
    relaxation.add_argument("--device", type=int, default=0)
    relaxation.add_argument("--validation", action="store_true")
    relaxation.add_argument("--json", default=None)
    relaxation.set_defaults(handler=command_relaxation)

    restart = commands.add_parser("restart")
    restart.add_argument("--case", default="cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025/case.yaml")
    restart.add_argument("--half-width", type=float, default=0.2)
    restart.add_argument("--micro-inertia", type=float, default=2.25e-5)
    restart.add_argument("--steps", type=int, default=100)
    restart.add_argument("--seed", type=int, default=20261110)
    restart.add_argument("--device", type=int, default=0)
    restart.add_argument("--validation", action="store_true")
    restart.set_defaults(handler=command_restart)

    arguments = parser.parse_args()
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
