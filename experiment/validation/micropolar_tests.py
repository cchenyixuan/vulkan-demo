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
  factors     Check C: the constant factors of the viscous term and of the angular diffusion at fixed h / dx (2-D).
  operators3d Check E: rigid rotation, linear w and the two factors in 3-D (tilted axis / gradient / wave vector).
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
  label       Check A: w as a passive label (c = 0, xi = 0): every fluid particle keeps its id-derived w bit for
              bit through defrags, seam crossings and a restart; matched by the global id in extension_fields.
  paired      Check D: K = 1 against K = 2 as paired continuations of one saved state (groups of cavity runs).
  operators-host  Discretizations of the viscous Laplacian compared on the host (viscous_operators.py): lattice
              symbols by direction and wavelength, and factor / noise / exactness on a developed arrangement.
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

# --viscous-form (before the sub-command): numerics.viscous_form of every case the sub-command builds; None = the
# case file's own value
_VISCOUS_FORM = None


def _apply_viscous_form(case):
    if _VISCOUS_FORM is not None:
        case.numerics.viscous_form = _VISCOUS_FORM
    return case


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
    dimension = int(case.physics.dimension)
    if dimension == 2:
        grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
        positions = np.zeros((count * count, 3), dtype=np.float64)
        positions[:, 0], positions[:, 1] = grid_x.ravel(), grid_y.ravel()
    else:                                              # a cube (check E)
        grid_x, grid_y, grid_z = np.meshgrid(axis, axis, axis, indexing="ij")
        positions = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    total = positions.shape[0]
    if jitter > 0.0:
        generator = np.random.default_rng(seed)
        positions[:, :dimension] += generator.uniform(-jitter, jitter, size=(total, dimension)) * spacing
    case.initial = type(case.initial)(
        positions=positions.astype(np.float32),
        velocities=np.zeros((total, 3), dtype=np.float32),
        material_group=np.full(total, fluid_group, dtype=np.uint32),
        angular_velocities=np.zeros((total, 3), dtype=np.float32))
    if regularization_xi is not None:
        case.numerics.regularization_xi = float(regularization_xi)
    if eps_h_squared is not None:
        case.numerics.eps_h_squared = float(eps_h_squared)
    case.numerics.wall_boundary = "simple"
    _apply_viscous_form(case)
    case.physics.micropolar_enabled = True
    case.physics.micro_inertia = float(micro_inertia)
    material.viscosity = float(viscosity)
    material.viscosity_transfer = float(viscosity_transfer)
    material.viscosity_rotation = float(viscosity_rotation)
    return case, fluid_group, spacing


def _interior(positions: np.ndarray, half_width: float, margin: float) -> np.ndarray:
    """Farther than ``margin`` from every face of the block (a 2-D block lies on z = 0 and passes the z test)."""
    return ((np.abs(positions[:, 0]) < half_width - margin) & (np.abs(positions[:, 1]) < half_width - margin)
            & (np.abs(positions[:, 2]) < half_width - margin))


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


def _richardson(ratios) -> tuple:
    """Three values at a step halved each time: observed order and the limit."""
    observed = float(np.log2(abs((ratios[1] - ratios[0]) / (ratios[2] - ratios[1]))))
    return observed, float(ratios[2] + (ratios[2] - ratios[1]) / (2.0 ** observed - 1.0))


def _least_squares_ratio(measured: np.ndarray, exact: np.ndarray) -> float:
    return float((measured * exact).sum() / (exact * exact).sum())


def command_factors(arguments) -> int:
    """Check C: the constant factor of the two Laplacian-type operators at fixed h / dx, 2-D, regular lattice,
    three resolutions: the viscous term, a_y / (-nu k^2 v_y) for v_y = sin(k x), w = 0, c = 0, and the angular
    diffusion, lap w / (-k^2 w) for w_z = sin(k x), v = 0; each with the case's EPS_H_SQUARED and with 0."""
    solver = _import_solver(_THIS_REPO_ROOT)
    wave_number = 2.0 * np.pi / arguments.wavelength
    coefficients = _OPERATOR_COEFFICIENTS
    viscosity = float(np.float32(coefficients["viscosity"]))
    coupling = float(np.float32(coefficients["viscosity"]) + np.float32(coefficients["viscosity_transfer"]))
    rotation = float(np.float32(coefficients["viscosity_rotation"]))
    inertia = float(np.float32(coefficients["micro_inertia"]))
    results = []

    def zeros(positions):
        return np.zeros((positions.shape[0], 3))

    def sine(component):
        def field(positions):
            values = np.zeros((positions.shape[0], 3))
            values[:, component] = np.sin(wave_number * positions[:, 0])
            return values
        return field

    for eps_label, eps_h_squared in (("case EPS_H_SQUARED", None), ("EPS_H_SQUARED = 0", 0.0)):
        ratios = {"nu": [], "xi": []}
        for resolution_case in arguments.resolution_cases:
            # viscous term: c = 0 (nu_t = -nu), so the acceleration is the viscous term alone (P = 0, rho uniform)
            case, _group, spacing = _fluid_block_case(
                solver, _THIS_REPO_ROOT / resolution_case, arguments.half_width, 0.0, arguments.seed, 0.0,
                eps_h_squared=eps_h_squared, viscosity=coefficients["viscosity"],
                viscosity_transfer=-coefficients["viscosity"], viscosity_rotation=0.0,
                micro_inertia=coefficients["micro_inertia"])
            positions = case.initial.positions.astype(np.float64)
            case.initial.velocities[:] = sine(1)(positions).astype(np.float32)
            rows, _invariants = _run_chain(solver, case, 1, [arguments.device], [1.0], 0, ["acceleration"], False)
            interior = _interior(rows["position_voxel_id"], arguments.half_width,
                                 arguments.margin * case.physics.smoothing_length)
            velocity = rows["velocity_mass"][interior, 1].astype(np.float64)
            # the bootstrap's backward half kick moved the stored velocity by dt a / 2: undo it for the exact field
            acceleration = rows["acceleration"][interior, 1].astype(np.float64)
            velocity = velocity + 0.5 * float(np.float32(case.physics.timestep)) * acceleration
            ratios["nu"].append(_least_squares_ratio(acceleration, -viscosity * wave_number ** 2 * velocity))
            cross = float(np.abs(rows["acceleration"][interior, 0]).max() / np.abs(acceleration).max())
            # angular diffusion
            case, rows, _spacing = _operator_state(solver, arguments, _THIS_REPO_ROOT / resolution_case,
                                                   arguments.half_width, 0.0, 0.0, zeros, sine(2),
                                                   eps_h_squared=eps_h_squared)
            interior = _interior(rows["position_voxel_id"], arguments.half_width,
                                 arguments.margin * case.physics.smoothing_length)
            spin = rows["angular_velocity"][interior, 2].astype(np.float64)
            # (the half kick changed the stored w by dt alpha / 2 as well)
            alpha = rows["angular_acceleration"][interior, 2].astype(np.float64)
            spin = spin + 0.5 * float(np.float32(case.physics.timestep)) * alpha
            laplacian = (alpha * inertia + 2.0 * coupling * spin) / rotation
            ratios["xi"].append(_least_squares_ratio(laplacian, -wave_number ** 2 * spin))
            print(f"[factors] 2-D, {eps_label}, dx = {spacing:.4g} (k h = "
                  f"{wave_number * case.physics.smoothing_length:.3f}): viscous a_y / (-nu k^2 v_y) = "
                  f"{ratios['nu'][-1]:.6f} (|a_x| / |a_y| max {cross:.1e}), lap w / (-k^2 w) = {ratios['xi'][-1]:.6f}")
        for name in ("nu", "xi"):
            order, limit = _richardson(ratios[name])
            results.append({"dimension": 2, "eps": eps_label, "coefficient": name, "ratios": ratios[name],
                            "order": order, "factor": limit})
            print(f"[factors] 2-D, {eps_label}, factor on {name}: {limit:.6f} ({limit - 1.0:+.4%}), order {order:.2f}")

    # Direction and disorder: the same two factors with the wave vector turned against the lattice axes, and on a
    # jittered lattice; one lattice, three wavelengths (k h halved twice = three resolutions at fixed h / dx).
    direction_case = _THIS_REPO_ROOT / arguments.resolution_cases[len(arguments.resolution_cases) // 2]
    for jitter in (0.0, arguments.jitter):
        for angle_degrees in arguments.angles:
            angle = np.radians(angle_degrees)
            direction = np.array([np.cos(angle), np.sin(angle), 0.0])
            transverse = np.array([-np.sin(angle), np.cos(angle), 0.0])
            ratios = {"nu": [], "xi": []}
            for k_times_h in arguments.k_times_h:
                case, _group, spacing = _fluid_block_case(
                    solver, direction_case, arguments.half_width, jitter, arguments.seed, 0.0,
                    viscosity=coefficients["viscosity"], viscosity_transfer=-coefficients["viscosity"],
                    viscosity_rotation=0.0, micro_inertia=coefficients["micro_inertia"])
                wave_vector = k_times_h / case.physics.smoothing_length * direction
                wave_number_squared = float(wave_vector @ wave_vector)
                positions = case.initial.positions.astype(np.float64)
                case.initial.velocities[:] = (np.sin(positions @ wave_vector)[:, None] * transverse[None, :]
                                              ).astype(np.float32)
                rows, _invariants = _run_chain(solver, case, 1, [arguments.device], [1.0], 0, ["acceleration"], False)
                interior = _interior(rows["position_voxel_id"], arguments.half_width,
                                     arguments.margin * case.physics.smoothing_length)
                timestep = float(np.float32(case.physics.timestep))
                acceleration = rows["acceleration"][interior, :3].astype(np.float64)
                velocity = rows["velocity_mass"][interior, :3].astype(np.float64) + 0.5 * timestep * acceleration
                ratios["nu"].append(_least_squares_ratio(acceleration @ transverse,
                                                         -viscosity * wave_number_squared * (velocity @ transverse)))
                case, rows, _spacing = _operator_state(
                    solver, arguments, direction_case, arguments.half_width, jitter, 0.0, zeros,
                    lambda positions: np.column_stack([np.zeros(positions.shape[0]), np.zeros(positions.shape[0]),
                                                       np.sin(positions @ wave_vector)]))
                interior = _interior(rows["position_voxel_id"], arguments.half_width,
                                     arguments.margin * case.physics.smoothing_length)
                alpha = rows["angular_acceleration"][interior, 2].astype(np.float64)
                spin = rows["angular_velocity"][interior, 2].astype(np.float64) + 0.5 * timestep * alpha
                ratios["xi"].append(_least_squares_ratio((alpha * inertia + 2.0 * coupling * spin) / rotation,
                                                         -wave_number_squared * spin))
            record = {"dimension": 2, "eps": "case EPS_H_SQUARED", "angle_degrees": angle_degrees, "jitter": jitter}
            text = []
            for name in ("nu", "xi"):
                order, limit = _richardson(ratios[name])
                record[name] = {"ratios": ratios[name], "order": order, "factor": limit}
                text.append(f"{name} {limit:.5f} ({limit - 1.0:+.3%}, order {order:.2f})")
            results.append(record)
            print(f"[factors] 2-D, case EPS_H_SQUARED, wave vector at {angle_degrees:g} deg to the lattice, jitter "
                  f"{jitter:g} dx: " + ", ".join(text))
    if arguments.json:
        pathlib.Path(arguments.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(arguments.json).write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


def command_operators3d(arguments) -> int:
    """Check E: the operators in 3-D (K = 1, single evaluations on a lattice and a jittered lattice, interior
    particles): rigid rotation about a tilted axis, w linear with nine distinct gradient components, and the two
    Laplacian factors with a tilted wave vector (three wavelengths on one lattice = three resolutions at fixed
    h / dx: the factors depend on k h only)."""
    solver = _import_solver(_THIS_REPO_ROOT)
    case_path = _THIS_REPO_ROOT / arguments.case
    coefficients = _OPERATOR_COEFFICIENTS
    viscosity = float(np.float32(coefficients["viscosity"]))
    coupling = float(np.float32(coefficients["viscosity"]) + np.float32(coefficients["viscosity_transfer"]))
    rotation = float(np.float32(coefficients["viscosity_rotation"]))
    inertia = float(np.float32(coefficients["micro_inertia"]))
    omega = np.array([0.7, -1.1, 1.6])
    gradient = np.array([[0.9, -1.4, 0.5], [1.7, 0.3, -0.8], [-0.6, 1.1, 2.0]])     # w_i = gradient[i, j] x_j
    curl_expected = np.array([gradient[2, 1] - gradient[1, 2], gradient[0, 2] - gradient[2, 0],
                              gradient[1, 0] - gradient[0, 1]])
    results = []
    exit_code = 0

    def zeros(positions):
        return np.zeros((positions.shape[0], 3))

    half_width = None
    for label, jitter in (("lattice", 0.0), ("jittered", arguments.jitter)):
        probe = solver[0](case_path)
        smoothing_length = float(probe.physics.smoothing_length)
        if probe.physics.dimension != 3:
            sys.exit(f"{arguments.case} is not a 3-D case")
        half_width = arguments.half_width_in_h * smoothing_length
        margin = arguments.margin * smoothing_length
        # --- rigid rotation
        case, rows, spacing = _operator_state(solver, arguments, case_path, half_width, jitter, 0.0,
                                              lambda positions: np.cross(omega[None, :], positions), zeros)
        interior = _interior(rows["position_voxel_id"], half_width, margin)
        scaled = rows["angular_acceleration"][interior, :3].astype(np.float64) * inertia / coupling
        component_error = np.abs(scaled - 2.0 * omega).max(axis=0) / (2.0 * np.abs(omega))
        acceleration_max = float(np.abs(rows["acceleration"][interior, :3]).max())
        passed = bool(component_error.max() < 1e-4)
        exit_code |= 0 if passed else 1
        results.append({"state": "rigid rotation", "particles": label, "interior": int(interior.sum()),
                        "h_over_dx": smoothing_length / spacing, "mean": scaled.mean(axis=0).tolist(),
                        "expected": (2.0 * omega).tolist(), "relative_error_max": component_error.tolist(),
                        "acceleration_max": acceleration_max})
        print(f"[3-D] rigid rotation, {label} ({interior.sum():,} interior, h / dx = {smoothing_length / spacing:.0f}): "
              f"Theta alpha / c mean = ({scaled[:, 0].mean():+.6f}, {scaled[:, 1].mean():+.6f}, "
              f"{scaled[:, 2].mean():+.6f}), expected ({2 * omega[0]:+.1f}, {2 * omega[1]:+.1f}, {2 * omega[2]:+.1f}); "
              f"max relative error per component ({component_error[0]:.1e}, {component_error[1]:.1e}, "
              f"{component_error[2]:.1e}); |a| max {acceleration_max:.1e} m/s^2; {'PASS' if passed else 'FAIL'}")
        # --- linear w
        case, rows, spacing = _operator_state(solver, arguments, case_path, half_width, jitter, 0.0, zeros,
                                              lambda positions: positions @ gradient.T)
        interior = _interior(rows["position_voxel_id"], half_width, margin)
        measured = rows["acceleration"][interior, :3].astype(np.float64) / coupling
        component_error = np.abs(measured - curl_expected).max(axis=0) / np.abs(curl_expected)
        passed = bool(component_error.max() < 1e-4)
        exit_code |= 0 if passed else 1
        results.append({"state": "linear w", "particles": label, "interior": int(interior.sum()),
                        "mean": measured.mean(axis=0).tolist(), "expected": curl_expected.tolist(),
                        "relative_error_max": component_error.tolist()})
        print(f"[3-D] linear w, {label}: a / c mean = ({measured[:, 0].mean():+.6f}, {measured[:, 1].mean():+.6f}, "
              f"{measured[:, 2].mean():+.6f}), curl w = ({curl_expected[0]:+.1f}, {curl_expected[1]:+.1f}, "
              f"{curl_expected[2]:+.1f}); max relative error per component ({component_error[0]:.1e}, "
              f"{component_error[1]:.1e}, {component_error[2]:.1e}); {'PASS' if passed else 'FAIL'}")

    # --- the two factors, tilted wave vector, three wavelengths (k h halved twice) on the regular lattice
    direction = np.array([2.0, 3.0, 6.0]) / 7.0
    transverse = np.array([3.0, -2.0, 0.0]) / np.sqrt(13.0)           # perpendicular to the wave vector
    for eps_label, eps_h_squared in (("case EPS_H_SQUARED", None), ("EPS_H_SQUARED = 0", 0.0)):
        ratios = {"nu": [], "xi": []}
        for k_times_h in arguments.k_times_h:
            wave_vector = k_times_h / smoothing_length * direction
            wave_number_squared = float(wave_vector @ wave_vector)
            # viscous term: v = e sin(k . r), e perpendicular to k (divergence free), c = 0
            case, _group, spacing = _fluid_block_case(
                solver, case_path, half_width, 0.0, arguments.seed, 0.0, eps_h_squared=eps_h_squared,
                viscosity=coefficients["viscosity"], viscosity_transfer=-coefficients["viscosity"],
                viscosity_rotation=0.0, micro_inertia=coefficients["micro_inertia"])
            positions = case.initial.positions.astype(np.float64)
            case.initial.velocities[:] = (np.sin(positions @ wave_vector)[:, None] * transverse[None, :]
                                          ).astype(np.float32)
            rows, _invariants = _run_chain(solver, case, 1, [arguments.device], [1.0], 0, ["acceleration"], False)
            interior = _interior(rows["position_voxel_id"], half_width, margin)
            timestep = float(np.float32(case.physics.timestep))
            acceleration = rows["acceleration"][interior, :3].astype(np.float64)
            velocity = rows["velocity_mass"][interior, :3].astype(np.float64) + 0.5 * timestep * acceleration
            ratios["nu"].append(_least_squares_ratio(acceleration @ transverse,
                                                     -viscosity * wave_number_squared * (velocity @ transverse)))
            # angular diffusion: w_x = sin(k . r)
            case, rows, _spacing = _operator_state(
                solver, arguments, case_path, half_width, 0.0, 0.0, zeros,
                lambda positions: np.column_stack([np.sin(positions @ wave_vector),
                                                   np.zeros(positions.shape[0]), np.zeros(positions.shape[0])]),
                eps_h_squared=eps_h_squared)
            interior = _interior(rows["position_voxel_id"], half_width, margin)
            alpha = rows["angular_acceleration"][interior, 0].astype(np.float64)
            spin = rows["angular_velocity"][interior, 0].astype(np.float64) + 0.5 * timestep * alpha
            laplacian = (alpha * inertia + 2.0 * coupling * spin) / rotation
            ratios["xi"].append(_least_squares_ratio(laplacian, -wave_number_squared * spin))
            print(f"[3-D] sin(k . r), {eps_label}, k h = {k_times_h:.3f}: viscous a / (-nu k^2 v) = "
                  f"{ratios['nu'][-1]:.6f}, lap w / (-k^2 w) = {ratios['xi'][-1]:.6f}")
        for name in ("nu", "xi"):
            order, limit = _richardson(ratios[name])
            results.append({"dimension": 3, "eps": eps_label, "coefficient": name, "ratios": ratios[name],
                            "order": order, "factor": limit, "h_over_dx": smoothing_length / spacing})
            print(f"[3-D] {eps_label}, factor on {name} at h / dx = {smoothing_length / spacing:.0f}: {limit:.6f} "
                  f"({limit - 1.0:+.4%}), order {order:.2f}")
    if arguments.json:
        pathlib.Path(arguments.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(arguments.json).write_text(json.dumps(results, indent=1), encoding="utf-8")
    return exit_code


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
    case = _apply_viscous_form(load_case(case_path))
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
    # "k2" / "k2_one_device" name the chain runs for any slab count (check B: --device-map 0,0,0 or 0,0,1)
    slab_count = len(device_map)
    first, second = device_map[0], device_map[-1]
    run_plan = [("k1", 1, [first]), ("k1_repeat", 1, [first]), ("k1_other_device", 1, [second]),
                ("k2", slab_count, device_map), ("k2_one_device", slab_count, [first] * slab_count)]
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
            cuts = [0.5 * (float(k2["position_voxel_id"][k2["_slab"] == index, 0].max())
                           + float(k2["position_voxel_id"][k2["_slab"] == index + 1, 0].min()))
                    for index in range(slab_count - 1)]
            cut = cuts[0]
            comparison_names = (["acceleration", "angular_acceleration", "angular_velocity"] + dummy_fields)
            table = {}
            for label, _slab_count, _devices in run_plan[1:]:
                other = runs[label]
                distance, index = tree.query(other["position_voxel_id"][:, :3].astype(np.float64))
                if np.unique(index).size != index.size or float(distance.max()) > 0.01 * smoothing_length:
                    raise RuntimeError(f"seam {wall_boundary} {steps} steps {label}: particles do not match "
                                       f"(max distance {distance.max():.3e})")
                is_fluid = np.isin(other["material"], fluid_groups)
                near = np.min(np.abs(other["position_voxel_id"][:, 0].astype(np.float64)[:, None]
                                     - np.asarray(cuts)[None, :]), axis=1) < 4.0 * smoothing_length
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


def _save_state(path: pathlib.Path, sims) -> None:
    """The chain's step-boundary state (cavity_runner.read_restart_state rows of every slab, merged) as one .npz."""
    from experiment.validation.cavity_runner import read_restart_state
    parts = [read_restart_state(sim) for sim in sims]
    merged = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, _stored_density_offset=np.float64(sims[0].stored_density_offset()), **merged)


def _load_state(path) -> tuple[dict, float]:
    with np.load(path) as archive:
        rows = {name: np.array(archive[name]) for name in archive.files if not name.startswith("_")}
        offset = float(archive["_stored_density_offset"])
    return rows, offset


def _state_as_initial(case, state: dict) -> None:
    """The state's positions / velocities / materials (and w) stand in for the case's initial condition, so the
    chain is cut by the state's particle distribution (cavity_runner --start-from does the same)."""
    case.initial = type(case.initial)(
        positions=np.ascontiguousarray(state["position_voxel_id"][:, :3], dtype=np.float32),
        velocities=np.ascontiguousarray(state["velocity_mass"][:, :3], dtype=np.float32),
        material_group=np.ascontiguousarray(state["material"], dtype=np.uint32),
        angular_velocities=(np.ascontiguousarray(state["angular_velocity"][:, :3], dtype=np.float32)
                            if "angular_velocity" in state else None))


def _cut_positions(case, chain) -> list:
    """x of every cut of the chain: the left edge of each slab's first own voxel column but the first slab's."""
    return [float(case.grid.origin_x + geometry.own_global_first_column * case.physics.smoothing_length)
            for geometry in chain.geometry[1:]]


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
    case = _apply_viscous_form(load_case(_THIS_REPO_ROOT / arguments.case))
    if arguments.epsilon_squared_factor is not None:
        case.numerics.eps_h_squared = arguments.epsilon_squared_factor * case.physics.smoothing_length ** 2
    device_map = [int(value) for value in arguments.device_map.split(",")]
    weights = [float(value) for value in arguments.weights.split(",")] if arguments.weights else [1.0] * arguments.slabs
    if len(device_map) != arguments.slabs or len(weights) != arguments.slabs:
        sys.exit("--device-map / --weights need one entry per slab")
    start_state = None
    if arguments.start_state:
        # a continuation: the saved step-boundary state, restarted (no bootstrap force, no half kick)
        start_state, start_offset = _load_state(arguments.start_state)
        _state_as_initial(case, start_state)
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
            if start_state is None:
                orchestrator.bootstrap_all()
            else:
                from experiment.v7.utils.partition_v7 import restart_slab_rows
                rows_per_slab = restart_slab_rows(case, chain, start_state["position_voxel_id"][:, 0])
                orchestrator.restart_all([{name: np.ascontiguousarray(values[rows])
                                           for name, values in start_state.items()} for rows in rows_per_slab],
                                         start_offset)
            orchestrator.run_pipelined(arguments.steps, depth=arguments.depth, on_defrag=on_defrag)
            if not samples["step"] or samples["step"][-1] != arguments.steps:
                sample(arguments.steps)
            statuses = [sim.readback_global_status() for sim in sims]
            if arguments.save_state:
                _save_state(pathlib.Path(arguments.save_state).resolve(), sims)
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
            "steps_per_second": arguments.steps / elapsed, "cuts_x": _cut_positions(case, chain),
            "start_state": arguments.start_state,
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


def _label_values(particle_count: int) -> np.ndarray:
    """Check A: a distinct micro-rotation per global id, three non-zero components that differ from each other and
    from every other particle's (exact float32 values)."""
    identifier = np.arange(1, particle_count + 1, dtype=np.float64)
    values = np.column_stack([identifier / 65536.0, -identifier / 32768.0 - 0.5, 0.25 + 3.0 * identifier / 131072.0])
    as_float32 = values.astype(np.float32)
    if not np.array_equal(as_float32.astype(np.float64), values):
        raise ValueError("label values are not exact in float32")
    return as_float32


def command_label(arguments) -> int:
    """Check A: w as a passive label. Micropolar on with c = 0 (nu_t = -nu) and xi = 0: alpha is exactly zero, so
    every fluid particle must keep its initial w bit for bit through predict, defrag, migration, the departed
    copies and a restart. The start is a developed flow (a saved state), each fluid particle's w replaced by a
    label derived from its global id; the id itself rides in extension_fields (V7_TRANSPORT_EXTENSION), the path
    the seam audit uses, independent of the micropolar buffers. Two phases with a state written and restarted in
    between. Velocity and density are compared with Newtonian runs of the same two phases (the T1 floor)."""
    import itertools
    import os
    os.environ["V7_TRANSPORT_EXTENSION"] = "1"
    solver = _import_solver(_THIS_REPO_ROOT)
    load_case, compute_chain_partition, SphSimulatorV7, VulkanContextV7, ChainOrchestratorV7 = solver
    from experiment.v7.utils.case_v7 import KIND_FLUID
    from experiment.v7.utils.partition_v7 import restart_slab_rows
    from experiment.validation.cavity_runner import read_restart_state
    device_map = [int(value) for value in arguments.device_map.split(",")]
    slab_count = len(device_map)
    source_state, source_offset = _load_state(arguments.start_state)
    particle_count = int(source_state["material"].shape[0])
    labels = _label_values(particle_count)
    exit_code = 0
    results = []

    def make_case(wall_boundary: str, micropolar: bool):
        case = load_case(_THIS_REPO_ROOT / arguments.case)
        case.numerics.wall_boundary = wall_boundary
        case.physics.micropolar_enabled = micropolar
        for material in case.materials:
            if material.kind == KIND_FLUID:
                material.viscosity_transfer = -material.viscosity if micropolar else 0.0     # c = 0
                material.viscosity_rotation = 0.0
        return case

    def run_phase(case, state: dict, offset: float, steps: int):
        """Restart the chain from ``state``, run ``steps``; returns (state after, installs, defrags, cuts)."""
        _state_as_initial(case, state)
        chain = compute_chain_partition(case, [1.0] * slab_count, pool_safety=1.2)
        contexts, sims = [], []
        installs, defrags = 0, 1                      # restart_all ends with one defrag
        try:
            for index in range(slab_count):
                contexts.append(VulkanContextV7.create(device_index=device_map[index], enable_validation=False,
                                                       application_name=f"micropolar_label_s{index}"))
                sims.append(SphSimulatorV7(contexts[-1], chain.slabs[index], sync_scheme="per-direction"))

            def on_defrag(frame: int, report: list) -> None:
                nonlocal installs, defrags
                if frame == 0:
                    return
                defrags += 1
                for entry in report:
                    installs += int(entry["interval_migration"])
                    bad = {key: value for key, value in entry.items()
                           if (key.startswith("overflow_") or key == "far_migration") and value}
                    if bad:
                        raise RuntimeError(f"invariant violation at frame {frame}: {bad}")

            with ChainOrchestratorV7(sims, defrag_cadence=case.numerics.defrag_cadence) as orchestrator:
                rows_per_slab = restart_slab_rows(case, chain, state["position_voxel_id"][:, 0])
                orchestrator.restart_all([{name: np.ascontiguousarray(values[rows])
                                           for name, values in state.items()} for rows in rows_per_slab], offset)
                orchestrator.run_pipelined(steps, depth=arguments.depth, on_defrag=on_defrag)
                statuses = [sim.readback_global_status() for sim in sims]
                # installs since the last defrag (the counter is reset by every defrag; a defrag at the last
                # frame was already counted by its report)
                if steps % case.numerics.defrag_cadence:
                    installs += sum(int(status["migration_install_count"]) for status in statuses)
                for status in statuses:
                    bad = {name: value for name, value in status.items()
                           if (name.startswith(_OVERFLOW_PREFIX) or name in ("far_migration_count",
                                                                             "stamp_error_count")) and value}
                    if bad:
                        raise RuntimeError(f"counters after the phase: {bad}")
                parts = [read_restart_state(sim) for sim in sims]
                new_offset = sims[0].stored_density_offset()
        finally:
            for sim in sims:
                sim.destroy()
            for context in contexts:
                context.destroy()
        merged = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
        return merged, new_offset, installs, defrags, _cut_positions(case, chain)

    def two_phases(wall_boundary: str, micropolar: bool):
        case = make_case(wall_boundary, micropolar)
        state = {name: values.copy() for name, values in source_state.items()}
        kinds = np.asarray([material.kind for material in case.materials])[state["material"].astype(np.int64)]
        fluid = kinds == KIND_FLUID
        if micropolar:
            state["angular_velocity"] = np.zeros((particle_count, 4), np.float32)
            state["angular_velocity"][fluid, :3] = labels[fluid]
            state["angular_acceleration"] = np.zeros((particle_count, 4), np.float32)
            state["extension_fields"] = np.zeros((particle_count, 4), np.float32)
            state["extension_fields"][:, 0] = np.arange(1, particle_count + 1, dtype=np.float32)
        else:
            state.pop("angular_velocity", None)
            state.pop("angular_acceleration", None)
        middle, offset, installs_first, defrags_first, cuts_first = run_phase(case, state, source_offset,
                                                                              arguments.steps_before)
        final, _offset, installs_second, defrags_second, cuts_second = run_phase(
            make_case(wall_boundary, micropolar), middle, offset, arguments.steps_after)
        return final, fluid, installs_first + installs_second, defrags_first + defrags_second, cuts_first, cuts_second

    smoothing_length = None
    points = _sample_points(0.48, 49)
    for wall_boundary in arguments.walls:
        final, fluid, installs, defrags, cuts_first, cuts_second = two_phases(wall_boundary, True)
        identifiers = np.rint(final["extension_fields"][:, 0]).astype(np.int64) - 1
        unique = (np.unique(identifiers).size == particle_count and identifiers.min() == 0
                  and identifiers.max() == particle_count - 1 and final["material"].shape[0] == particle_count)
        mismatches = -1
        compared = 0
        wall_mismatches = -1
        if unique:
            is_fluid = fluid[identifiers]
            got = final["angular_velocity"][:, :3]
            expected = np.where(is_fluid[:, None], labels[identifiers], np.float32(0.0))
            different = (got.view(np.uint32) != expected.view(np.uint32)).any(axis=1)
            mismatches = int(different[is_fluid].sum())
            wall_mismatches = int(different[~is_fluid].sum())
            compared = int(is_fluid.sum())
            material_kept = bool(np.array_equal(final["material"], source_state["material"][identifiers]))
        else:
            material_kept = False
        # velocity and density against Newtonian runs of the same two phases
        case = make_case(wall_boundary, False)
        smoothing_length = float(case.physics.smoothing_length)
        fluid_group = next(index for index, material in enumerate(case.materials) if material.kind == KIND_FLUID)
        references = [_sampled_fields(two_phases(wall_boundary, False)[0], points, smoothing_length, fluid_group)
                      for _ in range(arguments.floor_repeats)]
        sampled = _sampled_fields(final, points, smoothing_length, fluid_group)
        ratios = {}
        for name in ("u", "v", "rho"):
            pairs = [_sampled_difference({name: first[name]}, {name: second[name]})[name + "_rms"]
                     for first, second in itertools.combinations(references, 2)]
            against = [_sampled_difference({name: reference[name]}, {name: sampled[name]})[name + "_rms"]
                       for reference in references]
            ratios[name] = (max(pairs), max(against), max(against) / max(pairs))
        passed = unique and material_kept and mismatches == 0 and wall_mismatches == 0
        exit_code |= 0 if passed else 1
        record = {"walls": wall_boundary, "slabs": slab_count, "device_map": device_map,
                  "particles": particle_count, "fluid_compared": compared, "mismatches": mismatches,
                  "wall_particles": particle_count - compared, "wall_mismatches": wall_mismatches,
                  "ids_unique_and_complete": bool(unique), "materials_kept": material_kept,
                  "seam_crossings": installs, "defrags": defrags, "steps": [arguments.steps_before,
                                                                             arguments.steps_after],
                  "cuts_x": [cuts_first, cuts_second], "floor": {name: list(value) for name, value in ratios.items()}}
        results.append(record)
        print(f"[label] K = {slab_count} (devices {device_map}), {wall_boundary}: {compared:,} fluid particles "
              f"matched by global id, {mismatches} with w != w_0 (bitwise); {particle_count - compared:,} wall "
              f"particles, {wall_mismatches} changed; ids unique and complete: {unique}; {installs:,} seam "
              f"crossings, {defrags} defrags, 1 restart from a written state ({arguments.steps_before} + "
              f"{arguments.steps_after} steps, cuts at {cuts_first} / {cuts_second}); "
              f"{'PASS' if passed else 'FAIL'}")
        print("[label]   u, v, rho against Newtonian runs of the same phases (largest rms to a reference / largest "
              "reference pair rms): " + ", ".join(f"{name} {value[1]:.2e} / {value[0]:.2e} = {value[2]:.2f}"
                                                  for name, value in ratios.items()))
    if arguments.json:
        pathlib.Path(arguments.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(arguments.json).write_text(json.dumps(results, indent=1), encoding="utf-8")
    return exit_code


def command_paired(arguments) -> int:
    """Check D: continuations of one saved state, a group of K = 1 runs and one or more groups of K = 2 runs
    (cavity --start-state ...). Per run: the seven centreline extrema and the L2 difference of every profile from
    the mean profile of the K = 1 group, over the window [--discard, --discard + --window] time units after the
    start. Per quantity: group means and standard deviations and the difference of means in K = 1 standard
    deviations. Pass: every quantity of every K = 2 group within --sigma-limit K = 1 standard deviations."""
    profiles = ("u_vertical", "v_horizontal", "w_vertical", "w_horizontal")
    extrema = (("u_vertical", "min"), ("v_horizontal", "min"), ("v_horizontal", "max"),
               ("w_vertical", "min"), ("w_vertical", "max"), ("w_horizontal", "min"), ("w_horizontal", "max"))

    def load(directory):
        with np.load(pathlib.Path(directory) / "samples.npz") as archive:
            data = {name: np.array(archive[name]) for name in archive.files}
        meta = json.loads((pathlib.Path(directory) / "run.json").read_text(encoding="utf-8"))
        time_axis = data["step"] * meta["timestep"]
        window = (time_axis > arguments.discard + 1e-9) & (time_axis <= arguments.discard + arguments.window + 1e-9)
        mean = {name: data[name][window].mean(axis=0) for name in profiles}
        return {"directory": str(directory), "meta": meta, "samples": int(window.sum()), "mean": mean}

    groups = {}
    for specification in arguments.group:
        label, _, pattern = specification.partition("=")
        directories = sorted(pathlib.Path().glob(pattern))
        if not directories:
            sys.exit(f"--group {specification}: no run directory matches")
        groups[label] = [load(directory) for directory in directories]
    reference_label = arguments.group[0].partition("=")[0]
    reference_runs = groups[reference_label]
    reference_mean = {name: np.mean([run["mean"][name] for run in reference_runs], axis=0) for name in profiles}

    def quantities(run) -> dict:
        values = {}
        for name, kind in extrema:
            values[f"{kind} {name}"] = float(run["mean"][name].min() if kind == "min" else run["mean"][name].max())
        for name in profiles:
            values[f"L2 {name}"] = float(np.sqrt(((run["mean"][name] - reference_mean[name]) ** 2).mean()))
        # w on x = 0 at fixed points (the line the cut lies on): its value at the minimum's position of the K = 1 mean
        index = int(np.argmin(reference_mean["w_vertical"]))
        values["w_vertical at the K=1 minimum's point"] = float(run["mean"]["w_vertical"][index])
        return values

    table = {label: [quantities(run) for run in runs] for label, runs in groups.items()}
    names = list(table[reference_label][0])
    report = {"discard": arguments.discard, "window": arguments.window, "groups": {}, "summary": []}
    for label, runs in groups.items():
        print(f"\ngroup {label}: {len(runs)} runs")
        report["groups"][label] = []
        for run, values in zip(runs, table[label]):
            meta = run["meta"]
            print(f"  {pathlib.Path(run['directory']).name}: K = {meta['slabs']}, devices {meta['device_map']}, cuts x = "
                  f"{meta.get('cuts_x')}, {run['samples']} samples, {meta['steps_per_second']:,.0f} steps/s, counters "
                  f"{'zero' if not any(meta['counters'].values()) else meta['counters']}")
            report["groups"][label].append({"run": run["directory"], "cuts_x": meta.get("cuts_x"),
                                            "device_map": meta["device_map"], "values": values})
        for name in names:
            print(f"    {name:<40}" + "  ".join(f"{values[name]:+.6f}" if not name.startswith("L2")
                                               else f"{values[name]:.3e}" for values in table[label]))
    exit_code = 0
    print(f"\n{'quantity':<40}{'K=1 mean':>12}{'K=1 std':>11}" + "".join(
        f"{label + ' mean':>16}{label + ' std':>13}{'diff / std':>12}" for label in groups if label != reference_label))
    for name in names:
        reference_values = np.array([values[name] for values in table[reference_label]])
        mean, deviation = float(reference_values.mean()), float(reference_values.std(ddof=1))
        line = f"{name:<40}{mean:>12.6f}{deviation:>11.2e}"
        for label in groups:
            if label == reference_label:
                continue
            other = np.array([values[name] for values in table[label]])
            other_mean, other_deviation = float(other.mean()), float(other.std(ddof=1)) if other.size > 1 else 0.0
            sigma = (other_mean - mean) / deviation if deviation else float("nan")
            report["summary"].append({"quantity": name, "group": label, "reference_mean": mean,
                                      "reference_std": deviation, "mean": other_mean, "std": other_deviation,
                                      "difference_in_reference_std": sigma})
            if abs(sigma) > arguments.sigma_limit:
                exit_code = 1
            line += f"{other_mean:>16.6f}{other_deviation:>13.2e}{sigma:>12.2f}"
        print(line)
    if arguments.json:
        pathlib.Path(arguments.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n{'PASS' if exit_code == 0 else 'FAIL'}: every quantity within {arguments.sigma_limit} K = 1 standard "
          f"deviations (window: {arguments.window} time units after discarding {arguments.discard})")
    return exit_code


def command_operators_host(arguments) -> int:
    """Section 3 of the 2026-11-11 task: discretizations of the viscous Laplacian compared on the host (numpy,
    float64; experiment/validation/viscous_operators.py). Validates the harness against the shader's check C
    numbers, then the lattice measurements (long transverse wave by direction, the whole spectrum, the
    longitudinal wave) and the developed arrangement (a saved state: mean factor at k -> 0 with uncertainty, noise,
    exactness for linear fields, smallest pair distance). Writes a json and a markdown table file."""
    from experiment.validation import viscous_operators as operators
    solver = _import_solver(_THIS_REPO_ROOT)
    case = solver[0](_THIS_REPO_ROOT / arguments.case)
    from experiment.v7.utils.case_v7 import KIND_FLUID
    smoothing_length = float(case.physics.smoothing_length)
    fluid_group = next(index for index, material in enumerate(case.materials) if material.kind == KIND_FLUID)
    spacing = 2.0 * float(case.materials[fluid_group].radius)
    default_factor = float(case.numerics.eps_h_squared) / smoothing_length ** 2
    epsilon_factors = [default_factor, arguments.small_epsilon_factor]
    out_dir = pathlib.Path(arguments.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(text=""):
        print(text, flush=True)
        lines.append(text)

    def table(header, rows):
        say("| " + " | ".join(header) + " |")
        say("|" + "---|" * len(header))
        for row in rows:
            say("| " + " | ".join(row) + " |")
        say()

    names = operators.OPERATORS
    # ---- validation against the shader (check C: json of micropolar_tests.py factors / operators3d)
    say("## Harness validation against the shader (check C)")
    say()
    reference = json.loads((_THIS_REPO_ROOT / arguments.shader_factors).read_text(encoding="utf-8"))
    reference_3d = json.loads((_THIS_REPO_ROOT / arguments.shader_factors_3d).read_text(encoding="utf-8"))
    rows, worst = [], 0.0
    for dimension, source, length, step, direction, polarisation in (
            (2, reference, smoothing_length, spacing, np.array([1.0, 0.0]), np.array([0.0, 1.0])),
            (3, reference_3d, 0.04, 0.01, np.array([2.0, 3.0, 6.0]) / 7.0, np.array([3.0, -2.0, 0.0]) / np.sqrt(13.0))):
        positions = operators.lattice_patch(step, length, dimension)
        lattice = operators.ParticleSet(positions, np.full(positions.shape[0], step ** dimension), length, dimension)
        for eps_label, factor in (("case EPS_H_SQUARED", default_factor), ("EPS_H_SQUARED = 0", 0.0)):
            for coefficient, name in (("nu", "MG"), ("xi", "BM")):
                entry = next(item for item in source if item.get("dimension") == dimension
                             and item.get("eps") == eps_label and item.get("coefficient") == coefficient)
                host = [operators._plane_wave_ratio(lattice, name, k_times_h / length * direction, polarisation,
                                                    factor * length ** 2).real
                        for k_times_h in (0.6283185, 0.3141593, 0.1570796)]
                difference = max(abs(a - b) for a, b in zip(host, entry["ratios"]))
                worst = max(worst, difference)
                rows.append([f"{dimension}-D", eps_label, f"{coefficient} ({name})",
                             " ".join(f"{value:.6f}" for value in entry["ratios"]),
                             " ".join(f"{value:.6f}" for value in host), f"{difference:.1e}"])
    table(["case", "eps", "operator", "shader (float32, k h = 0.628 / 0.314 / 0.157)", "host (float64)",
           "max difference"], rows)
    validated = worst < 5e-5
    say(f"Largest difference {worst:.1e}: {'the harness reproduces the shader to four digits' if validated else 'MISMATCH'}.")
    say()

    # ---- lattice
    lattice = operators.lattice_measurements(smoothing_length, spacing, 0.04, 0.01, epsilon_factors)
    for factor in epsilon_factors:
        say(f"## Lattice, long transverse wave, 2-D, h/dx = {smoothing_length / spacing:.0f}, "
            f"epsilon_squared_factor = {factor:g}")
        say()
        table(["angle (deg)"] + list(names),
              [[f"{row['angle']:g}"] + [f"{row[name]:.5f}" for name in names]
               for row in lattice["long_wave_2d"] if row["epsilon_factor"] == factor])
        say(f"## Lattice, long transverse wave, 3-D, h/dx = 4, epsilon_squared_factor = {factor:g}")
        say()
        table(["wave vector, polarisation"] + list(names),
              [[row["direction"]] + [f"{row[name]:.5f}" for name in names]
               for row in lattice["long_wave_3d"] if row["epsilon_factor"] == factor])
        say(f"## Lattice, long longitudinal wave (v parallel to k), epsilon_squared_factor = {factor:g}")
        say()
        table(["case"] + list(names),
              [[row["case"]] + [f"{row[name]:.5f}" for name in names]
               for row in lattice["longitudinal"] if row["epsilon_factor"] == factor])
    for direction in ("axis", "diagonal"):
        say(f"## Lattice, transverse wave over the whole spectrum, 2-D, along the {direction}, "
            f"epsilon_squared_factor = {default_factor:g}")
        say()
        table(["k dx / pi"] + list(names),
              [[f"{row['k_dx_over_pi']:.3f}"] + [f"{row[name]:.4f}" for name in names]
               for row in lattice["spectrum_2d"] if row["direction"] == direction])

    # ---- developed arrangement
    state, offset = _load_state(_THIS_REPO_ROOT / arguments.state)
    density = state["density_pressure"][:, 0].astype(np.float64) + offset
    volumes = state["velocity_mass"][:, 3].astype(np.float64) / density
    is_fluid = state["material"] == fluid_group
    developed = operators.developed_measurements(state["position_voxel_id"][:, :2].astype(np.float64), volumes,
                                                 is_fluid, smoothing_length, spacing, epsilon_factors,
                                                 regularization_xi=arguments.regularization_xi)
    say(f"## Developed arrangement ({arguments.state}): {developed['fluid']:,} fluid particles, bulk "
        f"{developed['bulk']:,}, wall bands {developed['band']}; KGC regularisation xi = "
        f"{developed['regularization_xi']:g}")
    say()
    say(f"Smallest pair distance: {developed['smallest_pair_distance_over_dx']:.4f} dx.")
    say()
    for factor in epsilon_factors:
        used = [name for name in names if factor == epsilon_factors[0] or operators.HAS_EPSILON[name]]
        say(f"### Mean factor at k -> 0 (intercept of a fit in k^2 over 2, 4, 8 waves; +- standard error over the "
            f"four quadrants / the four walls), epsilon_squared_factor = {factor:g}")
        say()
        directions = []
        for item in developed["factors"]:
            key = (item["region"], item["direction"])
            if item["epsilon_factor"] == factor and key not in directions:
                directions.append(key)
        rows = []
        for region, direction in directions:
            row = [region, f"{direction:g} deg" if region == "bulk" else str(direction)]
            for name in used:
                item = next(entry for entry in developed["factors"] if entry["operator"] == name
                            and entry["epsilon_factor"] == factor and entry["region"] == region
                            and entry["direction"] == direction)
                row.append(f"{item['intercept']:.4f} +- {item['standard_error']:.4f}")
            rows.append(row)
        # bulk mean over the eight directions and its spread
        row = ["bulk", "mean of 8 (std over directions)"]
        for name in used:
            values = np.array([entry["intercept"] for entry in developed["factors"] if entry["operator"] == name
                               and entry["epsilon_factor"] == factor and entry["region"] == "bulk"])
            row.append(f"{values.mean():.4f} ({values.std(ddof=1):.4f})")
        rows.append(row)
        table(["region", "wave vector"] + used, rows)
    say(f"### Noise: rms of the residual after the projection / rms of the signal, per wave count (2 / 4 / 8), "
        f"epsilon_squared_factor = {default_factor:g}")
    say()
    rows = []
    for region in ("bulk", "wall band"):
        row = [region + (" (mean of 8 directions)" if region == "bulk" else " (normal and parallel)")]
        for name in names:
            values = np.mean([entry["per_wave_count"] for entry in developed["noise"] if entry["operator"] == name
                              and entry["epsilon_factor"] == default_factor and entry["region"] == region], axis=0)
            row.append(" / ".join(f"{value:.3f}" for value in values))
        rows.append(row)
    table(["region"] + list(names), rows)
    say("### Exactness: rms |lap_h v| in units of |grad v| / h (zero is exact)")
    say()
    rows = []
    for field in ("linear", "rigid rotation"):
        for region, key in (("bulk", "bulk_rms"), ("wall bands", "band_rms")):
            rows.append([field, region] + [f"{next(entry for entry in developed['exactness'] if entry['operator'] == name and entry['field'] == field)[key]:.2e}"
                                           for name in names])
    table(["field", "region"] + list(names), rows)
    (out_dir / "operators_host.json").write_text(json.dumps(
        {"case": arguments.case, "state": arguments.state, "smoothing_length": smoothing_length, "spacing": spacing,
         "epsilon_factors": epsilon_factors, "validation_max_difference": worst, "lattice": lattice,
         "developed": developed}, indent=1), encoding="utf-8")
    (out_dir / "operators_host.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if validated else 1


def command_viscous_cross_check(arguments) -> int:
    """The shader's viscous term against the host harness, particle by particle, for both numerics.viscous_form
    values: one force evaluation (the bootstrap: uniform rho_0, P = 0, so the acceleration is the viscous term)
    on (a) a regular lattice block and (b) the particle positions of a saved developed state, with
    v = e sin(k . r) on the fluid. Host: viscous_operators.py with the same ingredients (V = m / rho_0, the case's
    KGC regularisation and EPS_H_SQUARED). Compared where no wall particle is in the support."""
    from scipy.spatial import cKDTree
    from experiment.validation import viscous_operators as operators
    solver = _import_solver(_THIS_REPO_ROOT)
    load_case = solver[0]
    from experiment.v7.utils.case_v7 import KIND_FLUID
    exit_code = 0
    results = []
    state, _offset = _load_state(_THIS_REPO_ROOT / arguments.state)
    for arrangement in ("lattice", "developed"):
        for form, host_name in (("monaghan_gingold", "MG"), ("laplacian", "BM")):
            if arrangement == "lattice":
                case, fluid_group, spacing = _fluid_block_case(
                    solver, _THIS_REPO_ROOT / arguments.block_case, arguments.half_width, 0.0, arguments.seed, None,
                    viscosity=2.5e-3, viscosity_transfer=0.0, viscosity_rotation=0.0, micro_inertia=1.0)
                case.physics.micropolar_enabled = False
                case.initial.angular_velocities = None
                limit = arguments.half_width - 1.2 * case.physics.smoothing_length
            else:
                case = load_case(_THIS_REPO_ROOT / arguments.case)
                fluid_group = next(index for index, material in enumerate(case.materials)
                                   if material.kind == KIND_FLUID)
                case.physics.micropolar_enabled = False
                _state_as_initial(case, {name: state[name] for name in ("position_voxel_id", "velocity_mass",
                                                                         "material")})
                limit = 0.5 - 1.2 * case.physics.smoothing_length
            case.numerics.viscous_form = form
            smoothing_length = float(case.physics.smoothing_length)
            positions = case.initial.positions.astype(np.float64)
            radians = np.radians(arguments.angle)
            wave_vector = 2.0 * np.pi * arguments.waves * np.array([np.cos(radians), np.sin(radians), 0.0])
            transverse = np.array([-np.sin(radians), np.cos(radians), 0.0])
            is_fluid = case.initial.material_group == fluid_group
            velocities = np.zeros_like(positions)
            velocities[is_fluid] = np.sin(positions[is_fluid] @ wave_vector)[:, None] * transverse[None, :]
            case.initial.velocities = velocities.astype(np.float32)
            rows, _invariants = _run_chain(solver, case, 1, [arguments.device], [1.0], 0, ["acceleration"], False)
            # host: the float32 positions and velocities the GPU got, V = m / rho_0 (the bootstrap force's density)
            material = case.materials[fluid_group]
            volume = float(np.float32(material.rest_density * material.volume)) / material.rest_density
            particles = operators.ParticleSet(case.initial.positions.astype(np.float64),
                                              np.full(positions.shape[0], volume), smoothing_length, 2,
                                              float(case.numerics.regularization_xi))
            host = float(np.float32(material.viscosity)) * particles.laplacian(
                host_name, case.initial.velocities[:, :2].astype(np.float64), float(case.numerics.eps_h_squared))
            tree = cKDTree(case.initial.positions[:, :2].astype(np.float64))
            distance, index = tree.query(rows["position_voxel_id"][:, :2].astype(np.float64))
            if float(distance.max()) > 1e-9:
                raise RuntimeError(f"positions moved at the bootstrap: {distance.max():.2e}")
            inside = (is_fluid[index] & (np.abs(rows["position_voxel_id"][:, 0]) < limit)
                      & (np.abs(rows["position_voxel_id"][:, 1]) < limit))
            shader = rows["acceleration"][:, :2].astype(np.float64)
            difference = np.sqrt(((shader - host[index]) ** 2).sum(axis=1))[inside]
            scale = float(np.sqrt((host[index][inside] ** 2).sum(axis=1).mean()))
            record = {"arrangement": arrangement, "viscous_form": form, "host_operator": host_name,
                      "particles": int(inside.sum()), "rms_relative": float(np.sqrt((difference ** 2).mean()) / scale),
                      "max_relative": float(difference.max() / scale)}
            passed = record["rms_relative"] < arguments.tolerance
            exit_code |= 0 if passed else 1
            results.append(record)
            print(f"[cross-check] {arrangement}, viscous_form = {form} against host {host_name}: {inside.sum():,} "
                  f"particles, |shader - host| / rms(host) rms {record['rms_relative']:.2e} max "
                  f"{record['max_relative']:.2e}; {'PASS' if passed else 'FAIL'}")
    if arguments.json:
        pathlib.Path(arguments.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(arguments.json).write_text(json.dumps(results, indent=1), encoding="utf-8")
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
    parser.add_argument("--viscous-form", default=None, choices=("monaghan_gingold", "laplacian"),
                        help="numerics.viscous_form of every case the sub-command builds (default: the case's)")
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

    factors = commands.add_parser("factors")
    factors.add_argument("--resolution-cases", nargs="+", default=[
        "cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025/case.yaml",
        "cases/lid_driven_cavity_2d_n500_xi0p001_eps0p0025/case.yaml",
        "cases/lid_driven_cavity_2d_n1000_xi0p001_eps0p0025/case.yaml"])
    factors.add_argument("--half-width", type=float, default=0.15)
    factors.add_argument("--wavelength", type=float, default=0.2)
    factors.add_argument("--margin", type=float, default=1.2)
    factors.add_argument("--angles", type=float, nargs="+", default=[0.0, 22.5, 45.0])
    factors.add_argument("--k-times-h", type=float, nargs=3, default=[0.6283185, 0.3141593, 0.1570796])
    factors.add_argument("--jitter", type=float, default=0.1)
    factors.add_argument("--seed", type=int, default=20261110)
    factors.add_argument("--device", type=int, default=0)
    factors.add_argument("--validation", action="store_true")
    factors.add_argument("--json", default=None)
    factors.set_defaults(handler=command_factors)

    operators3d = commands.add_parser("operators3d")
    operators3d.add_argument("--case", default="cases/cavity3d_1m/case.yaml", help="a 3-D case (h / dx = 4)")
    operators3d.add_argument("--half-width-in-h", type=float, default=4.0)
    operators3d.add_argument("--k-times-h", type=float, nargs=3, default=[0.6283185, 0.3141593, 0.1570796])
    operators3d.add_argument("--jitter", type=float, default=0.1)
    operators3d.add_argument("--margin", type=float, default=1.2)
    operators3d.add_argument("--seed", type=int, default=20261110)
    operators3d.add_argument("--device", type=int, default=0)
    operators3d.add_argument("--validation", action="store_true")
    operators3d.add_argument("--json", default=None)
    operators3d.set_defaults(handler=command_operators3d)

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
    cavity.add_argument("--save-state", default=None, help="write the final step-boundary state (.npz)")
    cavity.add_argument("--epsilon-squared-factor", type=float, default=None,
                        help="override numerics.epsilon_squared_factor (EPS_H_SQUARED / h^2)")
    cavity.add_argument("--start-state", default=None,
                        help="continue from a saved state (restart) instead of the case's initial condition")
    cavity.set_defaults(handler=command_cavity)

    cavity_compare = commands.add_parser("cavity-compare")
    cavity_compare.add_argument("runs", nargs=2, help="two cavity run directories: the reference (K = 1), the test")
    cavity_compare.add_argument("--window", type=float, default=5.0, help="averaging window in time units")
    cavity_compare.add_argument("--json", default=None)
    cavity_compare.set_defaults(handler=command_cavity_compare)

    label = commands.add_parser("label")
    label.add_argument("--case", default="cases/micropolar_cavity_n250_re400/case.yaml")
    label.add_argument("--start-state", required=True, help="a saved state of the case (a developed flow)")
    label.add_argument("--device-map", default="0", help="one device per slab: 0 | 0,1 | 0,0,0 ...")
    label.add_argument("--walls", nargs="+", default=["simple", "adami"])
    label.add_argument("--steps-before", type=int, default=1500)
    label.add_argument("--steps-after", type=int, default=1500)
    label.add_argument("--floor-repeats", type=int, default=3)
    label.add_argument("--depth", type=int, default=2)
    label.add_argument("--json", default=None)
    label.set_defaults(handler=command_label)

    paired = commands.add_parser("paired")
    paired.add_argument("--group", action="append", required=True, metavar="LABEL=GLOB",
                        help="a group of cavity run directories; the first group is the K = 1 reference")
    paired.add_argument("--discard", type=float, default=1.0)
    paired.add_argument("--window", type=float, default=5.0)
    paired.add_argument("--sigma-limit", type=float, default=3.0)
    paired.add_argument("--json", default=None)
    paired.set_defaults(handler=command_paired)

    host = commands.add_parser("operators-host")
    host.add_argument("--case", default="cases/micropolar_cavity_n250_re400/case.yaml")
    host.add_argument("--state", default="data/micropolar/checks/d/k1_t40.npz")
    host.add_argument("--shader-factors", default="data/micropolar/checks/c_factors.json")
    host.add_argument("--shader-factors-3d", default="data/micropolar/checks/e_operators3d.json")
    host.add_argument("--small-epsilon-factor", type=float, default=1e-6)
    host.add_argument("--regularization-xi", type=float, default=0.0,
                      help="KGC regularisation on the developed arrangement (the solver's default is 0.001, a "
                           "factor 0.999 on every corrected gradient; 0 isolates the operators)")
    host.add_argument("--out", default="data/micropolar/viscous")
    host.set_defaults(handler=command_operators_host)

    cross = commands.add_parser("viscous-cross-check")
    cross.add_argument("--case", default="cases/micropolar_cavity_n250_re400/case.yaml")
    cross.add_argument("--block-case", default="cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025/case.yaml")
    cross.add_argument("--state", default="data/micropolar/checks/d/k1_t40.npz")
    cross.add_argument("--half-width", type=float, default=0.2)
    cross.add_argument("--waves", type=float, default=4.0, help="waves per metre (the cavity is 1 m wide)")
    cross.add_argument("--angle", type=float, default=22.5)
    cross.add_argument("--tolerance", type=float, default=1e-4)
    cross.add_argument("--seed", type=int, default=20261110)
    cross.add_argument("--device", type=int, default=0)
    cross.add_argument("--json", default=None)
    cross.set_defaults(handler=command_viscous_cross_check)

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
    global _VISCOUS_FORM
    _VISCOUS_FORM = arguments.viscous_form
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
