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
  seam        K = 2 against K = 1, particle by particle, on a prescribed smooth state (v, w, wall values) of a
              cavity case: acceleration, angular_acceleration and the wall dummy micro-rotation after the bootstrap
              force (no motion: particles match exactly by position) and after a few steps. A replica, a wall
              dummy value or a ghost layer without the right w shows up at the seam as an O(1) difference;
              agreement is at the rounding level of the neighbour sums.
  advection   A fluid block with uniform w translating across the cut of a K = 2 chain (c > 0, xi = 0): every
              particle, migrants included, must follow the relaxation product of T4. A migrant that lost w, or
              kept a stale alpha, would not.
  cavity      T5: a micropolar cavity case on a K-slab chain to a given step count, sampling the centreline
              profiles u(0, y), v(x, 0) and w_z on both lines (Shepard, radius h) at every --sample-steps (at defrag
              boundaries, pipeline drained); writes samples.npz + run.json.
  cavity-compare  Time-averaged centreline profiles and their extrema of two cavity runs (K = 1 against K = 2) over
              the last --window time units, the steadiness of each (last window against the one before), and the
              sampling noise (standard error of the window mean).
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
               validation: bool, depth: int = 2, pool_safety: float = 1.2) -> tuple[dict, dict]:
    """Bootstrap a K-slab chain of ``global_case``, run ``steps`` steps, return (merged own rows, invariants)."""
    _load_case, compute_chain_partition, SphSimulatorV7, VulkanContextV7, ChainOrchestratorV7 = solver
    chain = compute_chain_partition(global_case, list(weights), pool_safety=pool_safety)
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
    merged["_slab"] = np.concatenate([np.full(part["material"].shape[0], index, dtype=np.int32)
                                      for index, part in enumerate(parts)])
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
    rows.pop("_slab", None)
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


def _prescribed_cavity_case(solver, case_path: pathlib.Path, wall_boundary: str):
    """The cavity case with a smooth prescribed state: v a rigid rotation plus a shear, w_z of the fluid a linear
    plus a sine part, and non-zero prescribed wall micro-rotations (so the stored wall values and the adami dummy
    values enter the sums)."""
    load_case = solver[0]
    case = load_case(case_path)
    from experiment.v7.utils.case_v7 import KIND_FLUID
    case.numerics.wall_boundary = wall_boundary
    positions = case.initial.positions.astype(np.float64)
    kinds = np.asarray([material.kind for material in case.materials])[case.initial.material_group.astype(np.int64)]
    fluid = kinds == KIND_FLUID
    velocities = case.initial.velocities.copy()
    velocities[fluid, 0] = (-0.8 * positions[fluid, 1] + 0.3 * positions[fluid, 0]).astype(np.float32)
    velocities[fluid, 1] = (0.8 * positions[fluid, 0] - 0.3 * positions[fluid, 1]).astype(np.float32)
    case.initial.velocities = velocities
    spin = np.zeros((positions.shape[0], 3), dtype=np.float32)
    spin[fluid, 2] = (1.5 * positions[fluid, 0] - 0.7 * positions[fluid, 1]
                      + 0.4 * np.sin(9.0 * positions[fluid, 0]) * np.cos(7.0 * positions[fluid, 1])).astype(np.float32)
    # walls: a different constant per boundary material
    for group, material in enumerate(case.materials):
        if material.kind != KIND_FLUID:
            spin[case.initial.material_group == group, 2] = 0.6 - 0.9 * group
    case.initial.angular_velocities = spin
    return case


def command_seam(arguments) -> int:
    """Runs per wall option and step count: K = 1 on the first device twice (the same-device rounding floor), K = 1
    on the second device (the cross-device rounding floor: another vendor's code generation), K = 2 on the two
    devices, and K = 2 with both slabs on the first device (the seam alone, no second device). Every run is compared
    with the first K = 1 run, particle by particle, near the cut and elsewhere. Pass: each K = 2 run's rms
    difference near the cut stays within --ratio-limit times the larger K = 1 floor of its devices (or the absolute
    rounding bound when the floors are below it)."""
    from scipy.spatial import cKDTree
    solver = _import_solver(_THIS_REPO_ROOT)
    from experiment.v7.utils.case_v7 import KIND_FLUID
    device_map = [int(value) for value in arguments.device_map.split(",")]
    first, second = device_map[0], device_map[1]
    run_plan = [("k1", 1, [first]), ("k1_repeat", 1, [first]), ("k1_other_device", 1, [second]),
                ("k2", 2, [first, second]), ("k2_one_device", 2, [first, first])]
    exit_code = 0
    results = []
    for wall_boundary in arguments.walls:
        for steps in arguments.steps:
            fields = ["acceleration", "angular_acceleration", "angular_velocity", "density_pressure"]
            # the wall dummy buffers are transient and not permuted by defrag: after the bootstrap defrag (steps = 0)
            # their slots no longer belong to the particles read back, so they are compared after steps only
            dummy_fields = ["wall_dummy_velocity", "wall_dummy_angular_velocity"] \
                if wall_boundary == "adami" and steps > 0 else []
            runs = {}
            for label, slab_count, devices in run_plan:
                case = _prescribed_cavity_case(solver, _THIS_REPO_ROOT / arguments.case, wall_boundary)
                rows, invariants = _run_chain(solver, case, slab_count, devices, [1.0] * slab_count, steps,
                                              fields + dummy_fields, arguments.validation, depth=1)
                if (invariants["particles_alive"] != invariants["particles_loaded"]
                        or any(invariants["counters"].values())):
                    raise RuntimeError(f"seam {wall_boundary} {label}: {invariants}")
                runs[label] = rows
            smoothing_length = float(case.physics.smoothing_length)
            reference = runs["k1"]
            tree = cKDTree(reference["position_voxel_id"][:, :3].astype(np.float64))
            fluid_groups = [group for group, material in enumerate(case.materials) if material.kind == KIND_FLUID]
            k2 = runs["k2"]
            cut = 0.5 * (float(k2["position_voxel_id"][k2["_slab"] == 0, 0].max())
                         + float(k2["position_voxel_id"][k2["_slab"] == 1, 0].min()))
            comparison_names = (["acceleration", "angular_acceleration", "angular_velocity"] + dummy_fields)
            table = {}
            for label, _slab_count, _devices in run_plan[1:]:
                other = runs[label]
                distance, index = tree.query(other["position_voxel_id"][:, :3].astype(np.float64))
                if np.unique(index).size != index.size or float(distance.max()) > 0.01 * smoothing_length:
                    raise RuntimeError(f"seam {wall_boundary} {steps} steps {label}: particles do not match "
                                       f"(max distance {distance.max():.3e})")
                is_fluid = np.isin(other["material"], fluid_groups)
                near = np.abs(other["position_voxel_id"][:, 0].astype(np.float64) - cut) < 4.0 * smoothing_length
                for name in comparison_names:
                    mask = (~is_fluid if name.startswith("wall_dummy")
                            else np.ones_like(is_fluid) if name == "angular_velocity" else is_fluid)
                    scale = float(np.sqrt((reference[name][:, :3].astype(np.float64) ** 2).sum(axis=1)[
                        np.isin(reference["material"], fluid_groups) if not name.startswith("wall_dummy")
                        else ~np.isin(reference["material"], fluid_groups)].mean()))
                    difference = np.sqrt(((reference[name][index, :3].astype(np.float64)
                                           - other[name][:, :3].astype(np.float64)) ** 2).sum(axis=1)) / scale

                    def summary(selection):
                        values = difference[selection]
                        return float(np.sqrt((values ** 2).mean())), float(values.max())
                    near_rms, near_max = summary(mask & near)
                    far_rms, far_max = summary(mask & ~near)
                    table[(name, label)] = (near_rms, near_max, far_rms, far_max)
                    results.append({"walls": wall_boundary, "steps": steps, "run": label, "field": name,
                                    "cut_x": cut, "scale_rms": scale, "near_seam_rms": near_rms,
                                    "near_seam_max": near_max, "elsewhere_rms": far_rms, "elsewhere_max": far_max,
                                    "near_seam_particles": int((mask & near).sum())})
            print(f"[seam] {wall_boundary}, {steps} steps, cut at x = {cut:+.4f}; |difference to the first K = 1 run| "
                  f"/ field rms, near the cut (within 4 h) and elsewhere: rms (max)")
            for name in comparison_names:
                floors = {"k2": max(table[(name, "k1_repeat")][0], table[(name, "k1_other_device")][0],
                                    table[(name, "k1_repeat")][2], table[(name, "k1_other_device")][2]),
                          "k2_one_device": max(table[(name, "k1_repeat")][0], table[(name, "k1_repeat")][2])}
                verdicts = []
                for label in ("k2", "k2_one_device"):
                    bound = max(arguments.ratio_limit * floors[label], arguments.rounding_bound)
                    passed = table[(name, label)][0] <= bound
                    exit_code |= 0 if passed else 1
                    verdicts.append(f"{label} {'PASS' if passed else 'FAIL'}")
                print(f"[seam]   {name}: " + "; ".join(
                    f"{label} near {table[(name, label)][0]:.1e} ({table[(name, label)][1]:.1e}) elsewhere "
                    f"{table[(name, label)][2]:.1e} ({table[(name, label)][3]:.1e})"
                    for label, _count, _devices in run_plan[1:]) + " | " + ", ".join(verdicts))
    if arguments.json:
        pathlib.Path(arguments.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(arguments.json).write_text(json.dumps(results, indent=1), encoding="utf-8")
    return exit_code


def command_advection(arguments) -> int:
    solver = _import_solver(_THIS_REPO_ROOT)
    device_map = [int(value) for value in arguments.device_map.split(",")]
    viscosity, viscosity_transfer, initial_spin = 2.5e-3, -0.625e-3, 3.0
    coupling = float(np.float32(viscosity) + np.float32(viscosity_transfer))
    # jittered: a lattice line would cross the cut in one step (101 migrants at once, more than the migrant region
    # of a link is sized for); with the jitter the crossings spread over about 80 steps
    case, _group, _spacing = _fluid_block_case(solver, _THIS_REPO_ROOT / arguments.case, arguments.half_width,
                                               arguments.jitter, arguments.seed, None, viscosity=viscosity,
                                               viscosity_transfer=viscosity_transfer, viscosity_rotation=0.0,
                                               micro_inertia=arguments.micro_inertia)
    case.initial.angular_velocities[:, 2] = initial_spin
    case.initial.velocities[:, 0] = arguments.speed
    timestep = float(np.float32(case.physics.timestep))
    columns = arguments.speed * timestep * arguments.steps / case.physics.smoothing_length
    # the block's own pools are sized from the initial halves; half a block may end up on one slab
    rows, invariants = _run_chain(solver, case, 2, device_map, [1.0, 1.0], arguments.steps,
                                  ["angular_velocity", "acceleration"], arguments.validation, depth=arguments.depth,
                                  pool_safety=arguments.pool_safety)
    rate = 2.0 * coupling / float(np.float32(arguments.micro_inertia))
    discrete = initial_spin * (1.0 - 0.5 * rate * timestep) * (1.0 - rate * timestep) ** (arguments.steps - 1)
    spin = rows["angular_velocity"][:, 2].astype(np.float64)
    deviation = float(np.abs(spin / discrete - 1.0).max())
    moved = int(invariants["slab_particles"][1]) - case.initial.positions.shape[0] // 2
    conserved = invariants["particles_alive"] == invariants["particles_loaded"]
    counters_zero = all(value == 0 for value in invariants["counters"].values())
    passed = conserved and counters_zero and deviation < 0.05 * rate * timestep and abs(moved) > 0
    print(f"[advection] block at {arguments.speed} m/s for {arguments.steps} steps = {columns:.1f} voxel columns; "
          f"slab particles {invariants['slab_particles']} ({moved:+,} crossed to slab 1), "
          f"{'conserved' if conserved else 'NOT CONSERVED'}, counters "
          f"{'all zero' if counters_zero else invariants['counters']}")
    print(f"[advection] w_z min {spin.min():.7f} max {spin.max():.7f}, the product {discrete:.7f}: max relative "
          f"deviation {deviation:.2e} (one lost kick = {rate * timestep:.2e}; a lost w = 1); max |w_xy| "
          f"{np.abs(rows['angular_velocity'][:, :2]).max():.1e}, max |a| {np.abs(rows['acceleration'][:, :3]).max():.2e}; "
          f"{'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


_CENTRELINE_POINTS = 201


def _centreline_points(half_extent: float) -> dict:
    axis = np.linspace(-half_extent, half_extent, _CENTRELINE_POINTS)
    zeros = np.zeros_like(axis)
    return {"vertical": np.column_stack([zeros, axis]),        # x = 0: u(y), w_z(y)
            "horizontal": np.column_stack([axis, zeros])}      # y = 0: v(x), w_z(x)


def command_cavity(arguments) -> int:
    import time
    solver = _import_solver(_THIS_REPO_ROOT)
    load_case, compute_chain_partition, SphSimulatorV7, VulkanContextV7, ChainOrchestratorV7 = solver
    from experiment.v7.utils.case_v7 import KIND_FLUID
    case = load_case(_THIS_REPO_ROOT / arguments.case)
    device_map = [int(value) for value in arguments.device_map.split(",")]
    weights = [float(value) for value in arguments.weights.split(",")] if arguments.weights else [1.0] * arguments.slabs
    if len(device_map) != arguments.slabs or len(weights) != arguments.slabs:
        sys.exit("--device-map / --weights need one entry per slab")
    cadence = case.numerics.defrag_cadence
    if arguments.sample_steps % cadence or arguments.steps % arguments.sample_steps:
        sys.exit(f"--sample-steps must be a multiple of the defrag cadence {cadence} and divide --steps")
    fluid_group = next(index for index, material in enumerate(case.materials) if material.kind == KIND_FLUID)
    smoothing_length = float(case.physics.smoothing_length)
    timestep = float(case.physics.timestep)
    lines = _centreline_points(arguments.half_extent)
    out_dir = pathlib.Path(arguments.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = int(case.initial.positions.shape[0])
    samples = {"step": [], "u_vertical": [], "w_vertical": [], "v_horizontal": [], "w_horizontal": [],
               "kinetic_energy": [], "spin_square_mean": []}
    chain = compute_chain_partition(case, weights, pool_safety=1.2)
    contexts, sims = [], []
    started = time.perf_counter()
    try:
        for index in range(arguments.slabs):
            contexts.append(VulkanContextV7.create(device_index=device_map[index], enable_validation=False,
                                                   application_name=f"micropolar_cavity_k{arguments.slabs}_s{index}"))
            sims.append(SphSimulatorV7(contexts[-1], chain.slabs[index], sync_scheme="per-direction"))

        def sample(step: int) -> None:
            parts = [_read_own_particles(sim, ["density_pressure", "angular_velocity"]) for sim in sims]
            rows = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
            if rows["material"].shape[0] != expected:
                raise RuntimeError(f"step {step}: {rows['material'].shape[0]} particles alive, {expected} loaded")
            vertical = _sampled_fields(rows, lines["vertical"], smoothing_length, fluid_group)
            horizontal = _sampled_fields(rows, lines["horizontal"], smoothing_length, fluid_group)
            fluid = rows["material"] == fluid_group
            samples["step"].append(step)
            samples["u_vertical"].append(vertical["u"])
            samples["w_vertical"].append(vertical["w_z"])
            samples["v_horizontal"].append(horizontal["v"])
            samples["w_horizontal"].append(horizontal["w_z"])
            samples["kinetic_energy"].append(float(0.5 * (rows["velocity_mass"][fluid, 3].astype(np.float64)
                                                         * (rows["velocity_mass"][fluid, :2].astype(np.float64) ** 2
                                                            ).sum(axis=1)).sum()))
            samples["spin_square_mean"].append(float((rows["angular_velocity"][fluid, 2].astype(np.float64) ** 2).mean()))

        def on_defrag(frame: int, report: list) -> None:
            for index, entry in enumerate(report):
                bad = {key: value for key, value in entry.items()
                       if (key.startswith("overflow_") or key == "far_migration") and value}
                if bad:
                    raise RuntimeError(f"invariant violation at step {frame}, sim {index}: {bad}")
            if frame > 0 and frame % arguments.sample_steps == 0:
                sample(frame)
                if len(samples["step"]) % 20 == 0:
                    rate = frame / (time.perf_counter() - started)
                    print(f"[cavity] step {frame:,} (t = {frame * timestep:.2f}), {rate:,.0f} steps/s, min u "
                          f"{samples['u_vertical'][-1].min():+.5f}, max |w_z| "
                          f"{np.abs(samples['w_vertical'][-1]).max():.5f}", flush=True)

        with ChainOrchestratorV7(sims, defrag_cadence=cadence) as orchestrator:
            orchestrator.bootstrap_all()
            orchestrator.run_pipelined(arguments.steps, depth=arguments.depth, on_defrag=on_defrag)
            if not samples["step"] or samples["step"][-1] != arguments.steps:
                sample(arguments.steps)
            statuses = [sim.readback_global_status() for sim in sims]
    finally:
        for sim in sims:
            sim.destroy()
        for context in contexts:
            context.destroy()
    counters = {}
    for status in statuses:
        for name, value in status.items():
            if name.startswith(_OVERFLOW_PREFIX) or name in ("far_migration_count", "stamp_error_count"):
                counters[name] = counters.get(name, 0) + int(value)
    elapsed = time.perf_counter() - started
    np.savez(out_dir / "samples.npz", coordinate=np.linspace(-arguments.half_extent, arguments.half_extent,
                                                            _CENTRELINE_POINTS),
             **{name: np.asarray(values) for name, values in samples.items()})
    meta = {"case": arguments.case, "slabs": arguments.slabs, "device_map": device_map, "weights": weights,
            "steps": arguments.steps, "sample_steps": arguments.sample_steps, "timestep": timestep,
            "depth": arguments.depth, "particles": expected, "counters": counters, "elapsed_seconds": elapsed,
            "steps_per_second": arguments.steps / elapsed,
            "slab_particles": [int(slab.initial.positions.shape[0]) for slab in chain.slabs]}
    (out_dir / "run.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    zero = all(value == 0 for value in counters.values())
    print(f"[cavity] {arguments.steps:,} steps (t = {arguments.steps * timestep:.2f}) in {elapsed:.0f} s, "
          f"{len(samples['step'])} samples, particle count conserved at every sample, counters "
          f"{'all zero' if zero else counters}")
    return 0 if zero else 1


def command_cavity_compare(arguments) -> int:
    def load(directory):
        with np.load(pathlib.Path(directory) / "samples.npz") as archive:
            data = {name: np.array(archive[name]) for name in archive.files}
        meta = json.loads((pathlib.Path(directory) / "run.json").read_text(encoding="utf-8"))
        return data, meta

    runs = [load(directory) for directory in arguments.runs]
    profiles = ("u_vertical", "v_horizontal", "w_vertical", "w_horizontal")
    extrema = (("u_vertical", "min"), ("v_horizontal", "min"), ("v_horizontal", "max"),
               ("w_vertical", "min"), ("w_vertical", "max"), ("w_horizontal", "min"), ("w_horizontal", "max"))
    summaries = []
    for (data, meta), directory in zip(runs, arguments.runs):
        time_axis = data["step"] * meta["timestep"]
        last = time_axis > time_axis[-1] - arguments.window
        previous = (time_axis > time_axis[-1] - 2.0 * arguments.window) & ~last
        summary = {"label": f"K = {meta['slabs']} (devices {meta['device_map']})", "samples": int(last.sum()),
                   "end_time": float(time_axis[-1])}
        for name in profiles:
            summary[name] = data[name][last].mean(axis=0)
            summary[name + "_previous"] = data[name][previous].mean(axis=0)
            # standard error of the window mean, per point (samples treated as independent: an underestimate
            # when they are correlated)
            summary[name + "_error"] = data[name][last].std(axis=0, ddof=1) / np.sqrt(last.sum())
        summaries.append(summary)
        print(f"{directory}: {summary['label']}, t_end = {summary['end_time']:.2f}, window {arguments.window} "
              f"({summary['samples']} samples), {meta['steps_per_second']:,.0f} steps/s, counters "
              f"{'all zero' if not any(meta['counters'].values()) else meta['counters']}")
    reference, test = summaries[0], summaries[1]
    exit_code = 0
    report = {"window": arguments.window, "extrema": [], "profiles": []}
    print(f"\n{'extremum':<20}{reference['label']:>24}{test['label']:>24}{'rel. diff':>12}{'drift ref':>12}"
          f"{'drift test':>12}{'4 digits':>10}")
    for name, kind in extrema:
        pick = (np.min, np.argmin) if kind == "min" else (np.max, np.argmax)
        first, second = float(pick[0](reference[name])), float(pick[0](test[name]))
        relative = abs(second - first) / abs(first)
        drift_reference = abs(float(pick[0](reference[name + "_previous"])) - first) / abs(first)
        drift_test = abs(float(pick[0](test[name + "_previous"])) - second) / abs(second)
        same = f"{first:.4e}" == f"{second:.4e}" or relative < 5e-5
        four = float(f"{first:.3e}") == float(f"{second:.3e}")
        report["extrema"].append({"profile": name, "kind": kind, "reference": first, "test": second,
                                  "relative_difference": relative, "drift_reference": drift_reference,
                                  "drift_test": drift_test, "equal_to_four_digits": bool(four)})
        exit_code |= 0 if four else 1
        print(f"{kind + ' ' + name:<20}{first:>24.6f}{second:>24.6f}{relative:>12.2e}{drift_reference:>12.2e}"
              f"{drift_test:>12.2e}{'yes' if four else 'NO':>10}")
        del same
    print(f"\n{'profile':<16}{'max |diff|':>12}{'rms diff':>12}{'profile rms':>13}{'max std err':>13}"
          f"{'max drift ref':>15}")
    for name in profiles:
        difference = np.abs(test[name] - reference[name])
        scale = float(np.sqrt((reference[name] ** 2).mean()))
        error = float(max(reference[name + "_error"].max(), test[name + "_error"].max()))
        drift = float(np.abs(reference[name] - reference[name + "_previous"]).max())
        report["profiles"].append({"profile": name, "max_difference": float(difference.max()),
                                   "rms_difference": float(np.sqrt((difference ** 2).mean())), "profile_rms": scale,
                                   "max_standard_error": error, "max_drift_reference": drift})
        print(f"{name:<16}{difference.max():>12.3e}{np.sqrt((difference ** 2).mean()):>12.3e}{scale:>13.4e}"
              f"{error:>13.3e}{drift:>15.3e}")
    if arguments.json:
        pathlib.Path(arguments.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n{'PASS' if exit_code == 0 else 'FAIL'}: every extremum equal to four significant digits")
    return exit_code


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

    seam = commands.add_parser("seam")
    seam.add_argument("--case", default="cases/micropolar_cavity_n250_re400/case.yaml")
    seam.add_argument("--walls", nargs="+", default=["simple", "adami"])
    seam.add_argument("--steps", type=int, nargs="+", default=[0, 20])
    seam.add_argument("--device-map", default="0,1")
    seam.add_argument("--ratio-limit", type=float, default=2.0,
                      help="accepted (K = 2 vs K = 1 rms near the cut) / (K = 1 floor rms)")
    seam.add_argument("--rounding-bound", type=float, default=1e-5,
                      help="accepted rms |difference| / field rms when the K = 1 floor is below it (a K = 1 repeat "
                           "can be bit-identical at the bootstrap)")
    seam.add_argument("--validation", action="store_true")
    seam.add_argument("--json", default=None)
    seam.set_defaults(handler=command_seam)

    advection = commands.add_parser("advection")
    advection.add_argument("--case", default="cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025/case.yaml")
    advection.add_argument("--half-width", type=float, default=0.2)
    advection.add_argument("--micro-inertia", type=float, default=2.25e-4,
                           help="2 c dt / Theta = 5e-4: w halves over the default 1400 steps")
    advection.add_argument("--speed", type=float, default=1.0)
    advection.add_argument("--steps", type=int, default=1400)
    advection.add_argument("--pool-safety", type=float, default=2.2)
    advection.add_argument("--jitter", type=float, default=0.3, help="jitter amplitude in particle spacings")
    advection.add_argument("--depth", type=int, default=2)
    advection.add_argument("--device-map", default="0,1")
    advection.add_argument("--seed", type=int, default=20261110)
    advection.add_argument("--validation", action="store_true")
    advection.set_defaults(handler=command_advection)

    cavity = commands.add_parser("cavity")
    cavity.add_argument("--case", default="cases/micropolar_cavity_n250_re400/case.yaml")
    cavity.add_argument("--slabs", type=int, default=1)
    cavity.add_argument("--device-map", default="0")
    cavity.add_argument("--weights", default=None)
    cavity.add_argument("--steps", type=int, required=True)
    cavity.add_argument("--sample-steps", type=int, default=5000)
    cavity.add_argument("--depth", type=int, default=2)
    cavity.add_argument("--half-extent", type=float, default=0.5)
    cavity.add_argument("--out", required=True)
    cavity.set_defaults(handler=command_cavity)

    cavity_compare = commands.add_parser("cavity-compare")
    cavity_compare.add_argument("runs", nargs=2, help="two cavity run directories: the reference (K = 1), the test")
    cavity_compare.add_argument("--window", type=float, default=5.0, help="averaging window in time units")
    cavity_compare.add_argument("--json", default=None)
    cavity_compare.set_defaults(handler=command_cavity_compare)

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
