"""
micropolar_periodic.py - Stage 8b: periodic boundaries (docs/micropolar_design.md Section 18). The Taylor-Green
vortex in a doubly periodic box: the check of the port against origin/v8-tgv (Newtonian, its numerics), and the
micropolar Taylor-Green vortex against its exact two-mode solution.

Field (box [0, L)^2, origin of the field at L / 4 as on v8-tgv, so the fastest streams cross the box edges):
    u = U(t) sin(k x') cos(k y'),  v = -U(t) cos(k x') sin(k y'),  w = W(t) sin(k x') sin(k y'),  x' = x - L / 4
Equations of the solver (2-D, w scalar, curl v of a rigid rotation = 2 Omega):
    rho dv/dt = -grad p + (mu + kappa) lap v + kappa curl(w e_z),  rho j dw/dt = gamma lap w + kappa (curl_z v - 2 w)
With curl_z v = 2 k U sin sin, curl(w e_z) = k W (the shape of v), lap = -2 k^2, the nonlinear terms a gradient and
the advection of w zero, the exact solution is the 2 x 2 linear system (kinematic: nu = (mu + kappa) / rho,
c = kappa / rho, xi = gamma / rho, Theta = j):
    dU/dt = -2 k^2 nu U + c k W,   dW/dt = (2 c k U - (2 k^2 xi + 2 c) W) / Theta
The pressure of the field is p = rho U^2 / 4 (cos 2 k x' + cos 2 k y') (the initial density from it).

  cases     the case directories of RUNS (particle files shared per side)
  run       one run: the initial field, samples of U(t), W(t), E(t) at every defrag (1000 steps)
  analyse   the port check against v8-tgv's numbers, the micropolar runs against the exact two-mode solution and
            against their control, the noise floor from repeated runs -> data/micropolar/stage8b/
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

RUNS_DIRECTORY = "logs/micropolar/stage8b"
DATA_DIRECTORY = "data/micropolar/stage8b"
SUPPORT_OVER_SPACING = 5
COUPLING_RATIO = 3.0
STEP_NUMBER, STEP_NUMBER_ANGULAR, STEP_NUMBER_MONAGHAN = 0.0328, 0.0334, 0.0125
TWO_PI = 2.0 * math.pi

# v8-tgv's set v3 (eps^2 = 1e-4 h^2, numerics.delta_density): the port check, Newtonian, box [0, 1), U0 = 1,
# nu = 0.01 (Re = 100); v8's measured nu_eff / nu (docs/tgv/data_eps1e-4_delta/summary.md on origin/v8-tgv)
V8_REFERENCE = {"check_n400_ma0p1": 0.9949, "check_n400_ma0p05": 0.9971, "check_n400_ma0p02": 0.9990,
                "check_n400_ma0p01": 1.0000, "check_n800_ma0p1": 0.9961}

# run -> parameters. kind "v8": the port check; kind "tgv": the micropolar Taylor-Green vortex (thesis numerics,
# box [0, 2 pi), k = 1, U0 = 1, Re = U0 / (k nu_inf) = REYNOLDS); "length_times_modulus" = |k| l with |k| = sqrt 2 k;
# None = the control (nu_inf, micropolar path with nu_t = -nu, xi = 0, passive w). "theta_factor" = Theta / l^2.
REYNOLDS = 20.0
TGV_END_TIME = 20.0
RUNS = {}
for side, mach, end in ((400, 0.1, 2.0), (400, 0.05, 2.0), (400, 0.02, 2.0), (400, 0.01, 2.0), (800, 0.1, 2.0)):
    RUNS[f"check_n{side}_ma{str(mach).replace('.', 'p')}"] = {
        "kind": "v8", "side": side, "mach": mach, "end_time": end, "length_times_modulus": None, "theta_factor": 1.0,
        "initial_spin": "none", "repeat": 0}
for side in (400, 800):
    RUNS[f"tgv_control_n{side}"] = {"kind": "tgv", "side": side, "mach": 0.02, "length_times_modulus": None}
for value, label in ((0.25, "025"), (0.5, "05"), (1.0, "1"), (2.0, "2")):
    RUNS[f"tgv_kl{label}_n400"] = {"kind": "tgv", "side": 400, "mach": 0.02, "length_times_modulus": value}
RUNS["tgv_kl05_n800"] = {"kind": "tgv", "side": 800, "mach": 0.02, "length_times_modulus": 0.5}
# repeats (the spread of repeated runs), the Mach number halved once, the micro-inertia run
RUNS["tgv_control_n400_repeat"] = {"kind": "tgv", "side": 400, "mach": 0.02, "length_times_modulus": None, "repeat": 1}
RUNS["tgv_kl1_n400_repeat"] = {"kind": "tgv", "side": 400, "mach": 0.02, "length_times_modulus": 1.0, "repeat": 1}
# the control with the plain stored density (no V7_DELTA_DENSITY): does the storage matter at rho0 = 1000, Ma 0.02?
RUNS["tgv_control_n400_stored_density"] = {"kind": "tgv", "side": 400, "mach": 0.02, "length_times_modulus": None}
RUNS["tgv_control_n400_ma0p01"] = {"kind": "tgv", "side": 400, "mach": 0.01, "length_times_modulus": None}
RUNS["tgv_kl1_n400_ma0p01"] = {"kind": "tgv", "side": 400, "mach": 0.01, "length_times_modulus": 1.0}
# |k| l = 1 with the micro-inertia that makes the fast rate five times the slow one (0.7549 l^2; j = l^2 gives
# 4.06), started from W(0) = 0
INERTIA_FACTOR = 0.7549
RUNS["tgv_kl1_inertia_n400"] = {"kind": "tgv", "side": 400, "mach": 0.02, "length_times_modulus": 1.0,
                                "theta_factor": INERTIA_FACTOR, "initial_spin": "zero"}
# j = l^2 / 100: the approach to the quasi-steady mu_eff (j -> 0). The angular diffusion xi / Theta is 100 times that
# of j = l^2, so these two runs take their own (smaller) dt and do not shrink the dt of the family above; their
# control runs at their dt (the control of every other run takes the acoustic dt, 8.6 times larger)
RUNS["tgv_control_n400_small_dt"] = {"kind": "tgv", "side": 400, "mach": 0.02, "length_times_modulus": None,
                                     "own_timestep": True}
for value, label in ((0.5, "05"), (1.0, "1")):
    RUNS[f"tgv_kl{label}_j001_n400"] = {"kind": "tgv", "side": 400, "mach": 0.02, "length_times_modulus": value,
                                        "theta_factor": 0.01, "own_timestep": True}
# j = xi / (nu - c / 2) = 2 l^2 kappa / (mu + kappa) = 1.5 l^2 (K = 3): w = omega / 2 is then an exact solution and U
# decays at the rate of the Newtonian fluid mu + kappa / 2 at every |k| l (Section 18); one run checks it
RUNS["tgv_kl1_j150_n400"] = {"kind": "tgv", "side": 400, "mach": 0.02, "length_times_modulus": 1.0,
                             "theta_factor": 1.5}
for entry in RUNS.values():
    entry.setdefault("theta_factor", 1.0)
    entry.setdefault("initial_spin", "curl")
    entry.setdefault("repeat", 0)
    entry.setdefault("end_time", TGV_END_TIME)

CASE_TEMPLATE = """schema_version: 2

# GENERATED by experiment/validation/micropolar_periodic.py cases: {title}
# 2-D doubly periodic box [0, {length:.10f})^2, {side} x {side} = {count:,} fluid particles on the cell centres,
# dx = {spacing:.8e}, h = {support} dx = {smoothing_length:.8e}, {voxels} voxels of h per side. No walls.
# {description}
# Speed of sound {speed_of_sound:g} (Ma = {mach:g} on U0 = 1). dt = {timestep:.6e} ({governing} limit; physics.cfl =
# dt c0 / h); end time {end_time:g} = {steps:,} steps.

time:
  total: null
  max_steps: null
  output_cadence: null

physics:
  dimension: 2
  periodic: [true, true]
  h: {smoothing_length:.10e}
  particle_radius: {particle_radius:.10e}
  lattice: grid
  calibrate_volume: true
  speed_of_sound: {speed_of_sound!r}
  power: 7
  cfl: {cfl:.10f}
  gravity: [0.0, 0.0, 0.0]
{micropolar_block}
numerics:
{numerics}
capacities:
  pool_size: {pool_size}
  max_per_voxel: 64
  max_incoming: 16
  workgroup: 128

material_library: materials.yaml

geometry:
  frame: ../geometry_{length_label}_n{side}/frame.obj
  particles:
    - {{file: ../geometry_{length_label}_n{side}/domain.obj, material: periodic_fluid}}
"""

NUMERICS_V8 = """  use_density_diffusion: true
  delta_coefficient: 0.1
  use_kcg_correction: true
  regularization:
    xi: 0.001
    det_threshold: 1.0e-4
    frobenius_max: 10.0
  epsilon_squared_factor: 0.0001
  use_pst: true
  pst_main: 0.1
  pst_anti: 0.0005
  defrag_enabled: true
  defrag_cadence: 1000
  use_prefix_sum_defrag: false
"""
NUMERICS_THESIS = """  use_density_diffusion: true
  delta_coefficient: 0.1
  use_kcg_correction: true
  regularization:
    xi: 0.001
    det_threshold: 1.0e-4
    frobenius_max: 10.0
  epsilon_squared_factor: 1.0e-6
  use_pst: true
  pst_main: 0.1
  pst_anti: 0.0005
  defrag_enabled: true
  defrag_cadence: 1000
  use_prefix_sum_defrag: false
  viscous_form: laplacian
  wall_boundary: adami
"""
MATERIALS_TEMPLATE = """# GENERATED by experiment/validation/micropolar_periodic.py with the case beside it.
schema_version: 1

periodic_fluid:
  kind: fluid
  rest_density: {density!r}
  viscosity: {viscosity!r}
{micropolar_lines}"""


def far_viscosity(run: str) -> float:
    entry = RUNS[run]
    return 0.01 if entry["kind"] == "v8" else 1.0 / REYNOLDS


def box_length(run: str) -> float:
    return 1.0 if RUNS[run]["kind"] == "v8" else TWO_PI


def wavenumber(run: str) -> float:
    return TWO_PI / box_length(run)


def design(run: str) -> dict:
    """The solver's coefficients (kinematic): nu, nu_t, c, xi, Theta; None for the Newtonian path of the check."""
    entry = RUNS[run]
    far = far_viscosity(run)
    if entry["kind"] == "v8":
        return {"viscosity": far, "micropolar": False}
    k = wavenumber(run)
    modulus = math.sqrt(2.0) * k
    if entry["length_times_modulus"] is None:
        # the control: nu_inf on the micropolar path, c = 0 exactly, xi = 0, a positive Theta (passive w)
        return {"viscosity": far, "viscosity_transfer": -far, "coupling": 0.0, "viscosity_rotation": 0.0,
                "micro_inertia": (0.5 / modulus) ** 2, "length": None, "micropolar": True}
    from experiment.validation.micropolar_couette import design_parameters
    length = entry["length_times_modulus"] / modulus
    parameters = design_parameters(far, COUPLING_RATIO, length)
    parameters["micro_inertia"] *= entry["theta_factor"]
    return {**parameters, "length": length, "micropolar": True}


def timestep_of(run: str) -> dict:
    entry = RUNS[run]
    length, side = box_length(run), entry["side"]
    spacing = length / side
    support = SUPPORT_OVER_SPACING * spacing
    speed_of_sound = 1.0 / entry["mach"]
    if entry["kind"] == "v8":
        dt_acoustic = 0.15 * support / speed_of_sound
        dt_viscous = STEP_NUMBER_MONAGHAN * support ** 2 / far_viscosity(run)      # v8's generator rule
        timestep = min(dt_acoustic, dt_viscous)
        governing = "acoustic" if timestep == dt_acoustic else "viscous"
        return {"timestep": timestep, "governing": governing, "speed_of_sound": speed_of_sound,
                "cfl": timestep * speed_of_sound / support, "steps": int(round(entry["end_time"] / timestep))}
    # the thesis rule over every tgv run of the same side and Mach number: one dt for a control and its runs
    family = [name for name, other in RUNS.items() if other["kind"] == "tgv" and other["side"] == side
              and other["mach"] == entry["mach"]
              and other.get("own_timestep", False) == entry.get("own_timestep", False)]
    designs = [design(name) for name in family]
    dt_acoustic = 0.15 * support / speed_of_sound
    dt_viscous = STEP_NUMBER * support ** 2 / max(item["viscosity"] for item in designs)
    rates = [item["viscosity_rotation"] / item["micro_inertia"] for item in designs if item["viscosity_rotation"]]
    dt_angular = STEP_NUMBER_ANGULAR * support ** 2 / max(rates) if rates else float("inf")
    timestep = min(dt_acoustic, dt_viscous, dt_angular)
    governing = {dt_acoustic: "acoustic", dt_viscous: "viscous", dt_angular: "angular diffusion"}[timestep]
    steps = int(math.ceil(entry["end_time"] / timestep / 1000.0 - 1e-9)) * 1000
    return {"timestep": timestep, "governing": governing, "speed_of_sound": speed_of_sound,
            "cfl": timestep * speed_of_sound / support, "steps": steps, "dt_acoustic": dt_acoustic,
            "dt_viscous": dt_viscous, "dt_angular": dt_angular}


def write_geometry(directory: pathlib.Path, length: float, side: int) -> int:
    spacing = length / side
    coordinate = spacing * (np.arange(side) + 0.5)
    x, y = np.meshgrid(coordinate, coordinate, indexing="ij")
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "domain.obj", "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# {side * side} particles\n")
        np.savetxt(handle, np.column_stack([x.ravel(), y.ravel(), np.zeros(side * side)]), fmt="v %.12f %.12f %.1f")
    depth = 0.5 * spacing
    corners = [(0.0, 0.0, -depth), (0.0, length, -depth), (length, 0.0, -depth), (length, length, -depth),
               (0.0, 0.0, depth), (0.0, length, depth), (length, 0.0, depth), (length, length, depth)]
    faces = ["f 1 2 4", "f 1 4 3", "f 5 7 8", "f 5 8 6", "f 1 5 6", "f 1 6 2", "f 3 4 8", "f 3 8 7", "f 1 3 7",
             "f 1 7 5", "f 2 6 8", "f 2 8 4"]
    (directory / "frame.obj").write_text("# periodic box\n" + "".join(f"v {a:.12f} {b:.12f} {c:.12f}\n"
                                                                      for a, b, c in corners)
                                         + "\n".join(faces) + "\n", encoding="utf-8")
    return side * side


def make_case(run: str) -> pathlib.Path:
    entry = RUNS[run]
    length, side = box_length(run), entry["side"]
    length_label = "unit" if entry["kind"] == "v8" else "two_pi"
    geometry = _REPO_ROOT / f"cases/periodic/geometry_{length_label}_n{side}"
    count = side * side if (geometry / "domain.obj").exists() else write_geometry(geometry, length, side)
    spacing = length / side
    support = SUPPORT_OVER_SPACING * spacing
    parameters = design(run)
    step = timestep_of(run)
    if parameters["micropolar"]:
        micropolar_block = f"  micropolar:\n    enabled: true\n    micro_inertia: {parameters['micro_inertia']!r}\n"
        micropolar_lines = (f"  viscosity_transfer: {parameters['viscosity_transfer']!r}\n"
                            f"  viscosity_rotation: {parameters['viscosity_rotation']!r}\n")
    else:
        micropolar_block, micropolar_lines = "", ""
    if entry["kind"] == "v8":
        description = (f"Port check: v8-tgv's set v3 (eps^2 = 1e-4 h^2, delta density through V7_DELTA_DENSITY=1), "
                       f"Newtonian nu = {far_viscosity(run)} (Re = U0 L / nu = 100), rho0 = 1.")
    elif parameters["length"] is None:
        description = (f"Control: nu = nu_inf = {far_viscosity(run)!r} on the micropolar path, nu_t = -nu (c = 0), xi = "
                       f"0, Theta = {parameters['micro_inertia']:.6g} (passive w); Re = U0 / (k nu_inf) = {REYNOLDS:g}.")
    else:
        description = (f"Micropolar: K = {COUPLING_RATIO:g}, |k| l = {entry['length_times_modulus']:g} (l = "
                       f"{parameters['length']:.6f} = {parameters['length'] / spacing:.2f} dx), Theta = "
                       f"{entry['theta_factor']:g} l^2; nu = {parameters['viscosity']:.6g}, nu_t = "
                       f"{parameters['viscosity_transfer']:.6g}, c = {parameters['coupling']:.6g}, xi = "
                       f"{parameters['viscosity_rotation']:.6g}; Re = {REYNOLDS:g} on nu_inf, "
                       f"{REYNOLDS * (1.0 + COUPLING_RATIO / 2.0):g} on mu.")
    case_dir = _REPO_ROOT / f"cases/periodic/{run}"
    case_dir.mkdir(parents=True, exist_ok=True)
    density = 1.0 if entry["kind"] == "v8" else 1000.0
    (case_dir / "case.yaml").write_text(CASE_TEMPLATE.format(
        title=run, length=length, side=side, count=count, spacing=spacing, support=SUPPORT_OVER_SPACING,
        smoothing_length=support, voxels=side // SUPPORT_OVER_SPACING, description=description,
        speed_of_sound=step["speed_of_sound"], mach=entry["mach"], timestep=step["timestep"],
        governing=step["governing"], end_time=entry["end_time"], steps=step["steps"],
        particle_radius=0.5 * spacing, cfl=step["cfl"], micropolar_block=micropolar_block,
        numerics=NUMERICS_V8 if entry["kind"] == "v8" else NUMERICS_THESIS,
        pool_size=int(math.ceil(1.15 * count / 128.0)) * 128, length_label=length_label), encoding="utf-8",
        newline="\n")
    (case_dir / "materials.yaml").write_text(MATERIALS_TEMPLATE.format(
        density=density, viscosity=float(parameters["viscosity"]), micropolar_lines=micropolar_lines),
        encoding="utf-8", newline="\n")
    (case_dir / "periodic.json").write_text(json.dumps({
        "run": run, **entry, "length": length, "spacing": spacing, "smoothing_length": support,
        "wavenumber": wavenumber(run), "far_viscosity": far_viscosity(run), "rest_density": density,
        "design": parameters, **step}, indent=1), encoding="utf-8")
    return case_dir


def two_mode(parameters: dict, k: float, initial: tuple, times: np.ndarray) -> dict:
    """U(t), W(t) of the exact two-mode solution, its two rates and the slow eigenvector's W / U."""
    nu, c = parameters["viscosity"], parameters["coupling"]
    xi, theta = parameters["viscosity_rotation"], parameters["micro_inertia"]
    matrix = np.array([[-2.0 * k * k * nu, c * k], [2.0 * c * k / theta, -(2.0 * k * k * xi + 2.0 * c) / theta]])
    values, vectors = np.linalg.eig(matrix)
    order = np.argsort(-values.real)                 # slow (least negative) first
    values, vectors = values[order].real, vectors[:, order].real
    coefficients = np.linalg.solve(vectors, np.asarray(initial, dtype=np.float64))
    solution = vectors @ (coefficients[:, None] * np.exp(values[:, None] * times[None, :]))
    return {"U": solution[0], "W": solution[1], "slow_rate": -values[0], "fast_rate": -values[1],
            "slow_spin_over_velocity": vectors[1, 0] / vectors[0, 0]}


def command_cases(arguments) -> int:
    for run in arguments.runs or RUNS:
        print(f"[periodic] {make_case(run).relative_to(_REPO_ROOT)}", flush=True)
    return 0


def command_run(arguments) -> int:
    from experiment.validation import micropolar_tests as tests
    from experiment.validation.micropolar_couette import solver_revision
    from experiment.validation.micropolar_rotor3d import read_own_particles
    solver = tests._import_solver(_REPO_ROOT)
    load_case, compute_chain_partition, SphSimulatorV7, VulkanContextV7, ChainOrchestratorV7 = solver
    from experiment.v7.utils.case_v7 import KIND_FLUID
    run = arguments.run
    entry = RUNS[run]
    case_dir = make_case(run)
    meta = json.loads((case_dir / "periodic.json").read_text(encoding="utf-8"))
    case = load_case(case_dir / "case.yaml")
    if not all(case.physics.periodic):
        sys.exit(f"{run}: the loader did not read physics.periodic")
    timestep = float(case.physics.timestep)
    steps = meta["steps"] if arguments.steps is None else arguments.steps
    cadence = case.numerics.defrag_cadence
    length, k = meta["length"], meta["wavenumber"]
    origin = 0.25 * length
    fluid_group = next(index for index, material in enumerate(case.materials) if material.kind == KIND_FLUID)
    fluid_material = case.materials[fluid_group]
    rest, constant = float(fluid_material.rest_density), float(fluid_material.eos_constant)
    power = float(case.physics.power_parameter)
    micropolar = bool(case.physics.micropolar_enabled)
    parameters = meta["design"]

    def shapes(x, y):
        phase_x, phase_y = k * (x - origin), k * (y - origin)
        return np.sin(phase_x), np.cos(phase_x), np.sin(phase_y), np.cos(phase_y)

    # the initial field at the particles
    position = case.initial.positions[:, :2].astype(np.float64)
    sine_x, cosine_x, sine_y, cosine_y = shapes(position[:, 0], position[:, 1])
    velocities = case.initial.velocities.copy()
    velocities[:, 0] = (sine_x * cosine_y).astype(np.float32)
    velocities[:, 1] = (-cosine_x * sine_y).astype(np.float32)
    case.initial.velocities = velocities
    if micropolar:
        spin = np.zeros((position.shape[0], 3), dtype=np.float32)
        if entry["initial_spin"] == "curl":
            spin[:, 2] = (k * sine_x * sine_y).astype(np.float32)        # W(0) = k U(0): w = curl v / 2
        case.initial.angular_velocities = spin

    def initial_density(x, y):
        sine_x, cosine_x, sine_y, cosine_y = shapes(x, y)
        cosine_2x, cosine_2y = cosine_x ** 2 - sine_x ** 2, cosine_y ** 2 - sine_y ** 2
        pressure = 0.25 * rest * (cosine_2x + cosine_2y)
        density = rest * np.power(1.0 + pressure / constant, 1.0 / power)
        return density, (np.float32(constant) * (np.power(density.astype(np.float32) / np.float32(rest),
                                                          np.float32(power)) - np.float32(1.0))).astype(np.float32)

    chain = compute_chain_partition(case, [1.0], pool_safety=arguments.pool_safety)
    out_dir = (_REPO_ROOT / RUNS_DIRECTORY / run).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = {name: [] for name in ("step", "time", "kinetic_energy", "kinetic_energy_lattice", "velocity_amplitude",
                                     "spin_amplitude", "velocity_shape_residual", "max_speed", "density_mean",
                                     "density_min", "density_max", "outside_box")}
    fields = ["density_pressure", "acceleration"] + (["angular_velocity", "angular_acceleration"] if micropolar else [])
    contexts, sims, statuses = [], [], []
    started = time.perf_counter()
    failure = None
    bootstrap_seconds = None
    try:
        contexts.append(VulkanContextV7.create(device_index=int(arguments.device), enable_validation=False,
                                               application_name="periodic"))
        sim = SphSimulatorV7(contexts[-1], chain.slabs[0], sync_scheme="per-direction")
        sims.append(sim)
        original = sim._build_initial_data
        slab_initial = chain.slabs[0].initial

        def build() -> dict:
            data = original()
            pool_capacity = sim.case.capacities.total_pool_capacity()
            own_first = sim.own_first_pid()
            density_pressure = np.frombuffer(data["density_pressure"], np.float32).reshape(pool_capacity, 2).copy()
            rows = np.flatnonzero(slab_initial.material_group.astype(np.int64) == fluid_group)
            points = slab_initial.positions[rows, :2].astype(np.float64)
            density, pressure = initial_density(points[:, 0], points[:, 1])
            density_pressure[own_first + rows, 0] = (density - sim.stored_density_offset()).astype(np.float32)
            density_pressure[own_first + rows, 1] = pressure
            data["density_pressure"] = density_pressure.tobytes()
            return data
        sim._build_initial_data = build

        def sample(step: int) -> None:
            rows = read_own_particles(sim, fields)
            fluid = rows["material"] == fluid_group
            point = rows["position_voxel_id"][fluid, :2].astype(np.float64)
            mass = rows["velocity_mass"][fluid, 3].astype(np.float64)
            # leapfrog: velocity_mass holds v^{n-1/2}; v^n = v^{n-1/2} + dt / 2 a^n (as v8-tgv's runner)
            velocity = (rows["velocity_mass"][fluid, :2].astype(np.float64)
                        + 0.5 * timestep * rows["acceleration"][fluid, :2].astype(np.float64))
            if not np.isfinite(velocity).all():
                raise RuntimeError(f"step {step}: non-finite velocity")
            sine_x, cosine_x, sine_y, cosine_y = shapes(point[:, 0], point[:, 1])
            shape_u, shape_v = sine_x * cosine_y, -cosine_x * sine_y
            amplitude = 2.0 * float(np.mean(velocity[:, 0] * shape_u + velocity[:, 1] * shape_v))
            residual = velocity - amplitude * np.column_stack([shape_u, shape_v])
            # readback_buffers_batch returns rho (it adds the offset of the delta storage itself). The Stage 8b runs
            # were made before this line lost a second "+ sim.stored_density_offset()": their density_* samples
            # after t = 0 carry rho0 twice with V7_DELTA_DENSITY=1 (a diagnostic only; no rate uses it)
            density = rows["density_pressure"][fluid, 0].astype(np.float64)
            samples["step"].append(step)
            samples["time"].append(step * timestep)
            samples["kinetic_energy"].append(float(0.5 * (mass * (velocity ** 2).sum(axis=1)).sum()))
            samples["kinetic_energy_lattice"].append(float(0.5 * rest * (length / entry["side"]) ** 2
                                                           * (velocity ** 2).sum()))
            samples["velocity_amplitude"].append(amplitude)
            samples["velocity_shape_residual"].append(float(np.sqrt((residual ** 2).sum(axis=1).mean() / 0.5))
                                                      / max(abs(amplitude), 1e-30))
            if micropolar:
                spin = (rows["angular_velocity"][fluid, 2].astype(np.float64)
                        + 0.5 * timestep * rows["angular_acceleration"][fluid, 2].astype(np.float64))
                samples["spin_amplitude"].append(4.0 * float(np.mean(spin * sine_x * sine_y)))
            else:
                samples["spin_amplitude"].append(0.0)
            samples["max_speed"].append(float(np.sqrt((velocity ** 2).sum(axis=1).max())))
            samples["density_mean"].append(float(density.mean()))
            samples["density_min"].append(float(density.min()))
            samples["density_max"].append(float(density.max()))
            samples["outside_box"].append(int(((point < 0.0) | (point >= length)).any(axis=1).sum()))

        def on_defrag(frame: int, report: list) -> None:
            for index, item in enumerate(report):
                bad = {key: value for key, value in item.items()
                       if (key.startswith("overflow_") or key == "far_migration") and value}
                if bad:
                    raise RuntimeError(f"invariant violation at step {frame}: {bad}")
            if frame > 0 and frame % cadence == 0:
                sample(frame)

        with ChainOrchestratorV7(sims, defrag_cadence=cadence) as orchestrator:
            orchestrator.bootstrap_all()
            bootstrap_seconds = time.perf_counter() - started
            try:
                orchestrator.run_pipelined(steps, depth=arguments.depth, on_defrag=on_defrag)
                if not samples["step"] or samples["step"][-1] != steps:
                    sample(steps)
            except RuntimeError as error:
                failure = str(error)
            statuses = [sim.readback_global_status() for sim in sims]
    finally:
        for sim in sims:
            sim.destroy()
        for context in contexts:
            context.destroy()
    counters = {}
    for status in statuses:
        for name, value in status.items():
            if name.startswith("overflow_") or name in ("far_migration_count", "stamp_error_count"):
                counters[name] = counters.get(name, 0) + int(value)
    elapsed = time.perf_counter() - started
    # the initial energy and amplitudes from the host (U(0) = 1; W(0) by the initial spin)
    initial_mass = None
    np.savez_compressed(out_dir / "samples.npz", **{name: np.asarray(values) for name, values in samples.items()})
    import os
    record = {"run": run, "case": str(case_dir.relative_to(_REPO_ROOT)).replace("\\", "/"), "periodic": meta,
              "revision": solver_revision(), "device": int(arguments.device), "steps": steps, "timestep": timestep,
              "particles": int(case.initial.positions.shape[0]), "counters": counters, "elapsed_seconds": elapsed,
              "bootstrap_seconds": bootstrap_seconds, "failure": failure, "delta_density": os.environ.get(
                  "V7_DELTA_DENSITY", "0"), "initial_mass": initial_mass,
              "steps_per_second": steps / max(elapsed - (bootstrap_seconds or 0.0), 1e-9)}
    (out_dir / "run.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    zero = all(value == 0 for value in counters.values())
    print(f"[periodic] {run}: {steps:,} steps (t = {steps * timestep:.4f}) in {elapsed:.0f} s "
          f"({record['steps_per_second']:.1f} steps/s), counters {'all zero' if zero else counters}"
          + (f"; FAILED: {failure}" if failure else ""), flush=True)
    return 0 if zero and failure is None else 1


V8_FIT_WINDOW = (0.1, 2.0)
TGV_FIT_WINDOW = (5.0, 20.0)
CONTROL_OF = {"tgv_kl025_n400": "tgv_control_n400", "tgv_kl05_n400": "tgv_control_n400",
              "tgv_kl1_n400": "tgv_control_n400", "tgv_kl2_n400": "tgv_control_n400",
              "tgv_kl05_n800": "tgv_control_n800", "tgv_kl1_n400_repeat": "tgv_control_n400_repeat",
              "tgv_kl1_n400_ma0p01": "tgv_control_n400_ma0p01", "tgv_kl1_inertia_n400": "tgv_control_n400",
              "tgv_kl05_j001_n400": "tgv_control_n400_small_dt", "tgv_kl1_j001_n400": "tgv_control_n400_small_dt",
              "tgv_kl1_j150_n400": "tgv_control_n400"}


def load_run(run: str):
    directory = _REPO_ROOT / RUNS_DIRECTORY / run
    if not (directory / "run.json").exists():
        return None
    record = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if record.get("failure"):
        return None
    with np.load(directory / "samples.npz") as archive:
        samples = {name: np.array(archive[name]) for name in archive.files}
    return {"record": record, "samples": samples}


def fitted_rate(times: np.ndarray, values: np.ndarray, window: tuple) -> float:
    """-d ln|values| / dt by least squares over the window."""
    chosen = (times >= window[0]) & (times <= window[1]) & (np.abs(values) > 0)
    slope = np.polyfit(times[chosen], np.log(np.abs(values[chosen])), 1)[0]
    return float(-slope)


def mu_effective_ratio(length_times_modulus: float) -> float:
    """mu_eff / (mu + kappa / 2) of the quasi-steady w (j -> 0), K = COUPLING_RATIO."""
    shear = 1.0
    coupling = COUPLING_RATIO * shear
    s = length_times_modulus ** 2 * (2.0 * shear + coupling) / (2.0 * (shear + coupling))
    return (shear + coupling / 2.0 + (coupling / 2.0) * s / (1.0 + s)) / (shear + coupling / 2.0)


def command_analyse(arguments) -> int:
    lines, result = [], {"check": {}, "tgv": {}}

    def say(text: str = "") -> None:
        lines.append(text)
        print(text, flush=True)

    say("Stage 8b: the periodic domain. 1. The port check against v8-tgv (set v3: eps^2 = 1e-4 h^2, delta density; "
        "Re 100, nu_eff / nu = slope of ln E over t in [0.1, 2] against -2 (2 nu k^2))")
    say(f"{'run':<22}{'nu_eff/nu':>11}{'v8-tgv':>9}{'difference':>12}{'steps/s':>9}  counters")
    for run, reference in V8_REFERENCE.items():
        loaded = load_run(run)
        if loaded is None:
            continue
        samples, record = loaded["samples"], loaded["record"]
        rate = fitted_rate(samples["time"], samples["kinetic_energy"], V8_FIT_WINDOW) / 2.0
        exact = 2.0 * far_viscosity(run) * wavenumber(run) ** 2
        ratio = rate / exact
        result["check"][run] = {"nu_eff_over_nu": ratio, "v8_tgv": reference, "difference": ratio - reference,
                                "steps_per_second": record["steps_per_second"], "counters": record["counters"],
                                "delta_density": record["delta_density"]}
        say(f"{run:<22}{ratio:>11.4f}{reference:>9.4f}{ratio - reference:>+12.4f}{record['steps_per_second']:>9.0f}  "
            f"{'zero' if not any(record['counters'].values()) else record['counters']}")
    say()
    say(f"2. The micropolar Taylor-Green vortex (box [0, 2 pi), k = 1, Re = {REYNOLDS:g} on nu_inf, K = "
        f"{COUPLING_RATIO:g}); rates fitted to ln U over t in {TGV_FIT_WINDOW} for the run and for the exact two-mode "
        "solution sampled at the same times")
    for run, entry in RUNS.items():
        if entry["kind"] != "tgv":
            continue
        loaded = load_run(run)
        if loaded is None:
            continue
        samples, record = loaded["samples"], loaded["record"]
        parameters = record["periodic"]["design"]
        k = record["periodic"]["wavenumber"]
        times = samples["time"]
        if parameters["length"] is None:
            exact_rate = 2.0 * k * k * far_viscosity(run)
            measured = fitted_rate(times, samples["velocity_amplitude"], TGV_FIT_WINDOW)
            result["tgv"][run] = {"control": True, "rate": measured, "exact_rate": exact_rate,
                                  "nu_eff_over_nu": measured / exact_rate, "steps_per_second": record["steps_per_second"],
                                  "counters": record["counters"]}
            say(f"  {run:<28} control: rate {measured:.6f}, exact 2 k^2 nu_inf {exact_rate:.6f}, ratio "
                f"{measured / exact_rate:.4f}")
            continue
        initial = (1.0, k if entry["initial_spin"] == "curl" else 0.0)
        exact = two_mode(parameters, k, initial, times)
        exact_fitted = fitted_rate(times, exact["U"], TGV_FIT_WINDOW)
        measured = fitted_rate(times, samples["velocity_amplitude"], TGV_FIT_WINDOW)
        late = times >= TGV_FIT_WINDOW[0]
        spin_ratio = float(np.mean(samples["spin_amplitude"][late] / samples["velocity_amplitude"][late]))
        exact_spin_ratio = float(np.mean(exact["W"][late] / exact["U"][late]))
        error_u = float(np.sqrt(np.mean((samples["velocity_amplitude"] - exact["U"]) ** 2)))
        error_w = float(np.sqrt(np.mean((samples["spin_amplitude"] - exact["W"]) ** 2)))
        item = {"control": False, "length_times_modulus": entry["length_times_modulus"],
                "theta_factor": entry["theta_factor"], "rate": measured, "exact_fitted_rate": exact_fitted,
                "rate_over_exact": measured / exact_fitted, "exact_slow_rate": exact["slow_rate"],
                "exact_fast_rate": exact["fast_rate"], "spin_over_velocity": spin_ratio,
                "exact_spin_over_velocity": exact_spin_ratio, "spin_ratio_over_exact": spin_ratio / exact_spin_ratio,
                "rms_error_U": error_u, "rms_error_W": error_w, "mu_effective_ratio": mu_effective_ratio(
                    entry["length_times_modulus"]), "steps_per_second": record["steps_per_second"],
                "counters": record["counters"]}
        item["series"] = {"time": times.tolist(), "U": samples["velocity_amplitude"].tolist(),
                          "W": samples["spin_amplitude"].tolist(), "exact_U": exact["U"].tolist(),
                          "exact_W": exact["W"].tolist()}
        control = CONTROL_OF.get(run)
        if control and control in result["tgv"]:
            base = result["tgv"][control]
            exact_control = 2.0 * k * k * far_viscosity(run)
            item["over_control"] = measured / base["rate"]
            item["exact_over_control"] = exact_fitted / exact_control
            item["exact_slow_over_control"] = exact["slow_rate"] / exact_control
        result["tgv"][run] = item
        say(f"  {run:<28} |k|l {entry['length_times_modulus']:g}, Theta {entry['theta_factor']:g} l^2: rate {measured:.6f}, "
            f"exact {exact_fitted:.6f} (slow eigenvalue {exact['slow_rate']:.6f}, fast {exact['fast_rate']:.6f}): "
            f"{measured / exact_fitted:.4f}; W/U {spin_ratio:.5f}, exact {exact_spin_ratio:.5f} "
            f"({spin_ratio / exact_spin_ratio:.4f}); rms U {error_u:.2e}, W {error_w:.2e}"
            + (f"; over the control {item['over_control']:.4f} (exact {item['exact_over_control']:.4f}, quasi-steady "
               f"mu_eff {item['mu_effective_ratio']:.4f})" if "over_control" in item else ""))
    # the noise floor: the spread of repeated runs, the Mach number, the resolution pair
    floors = {}
    for first, second, label in (("tgv_control_n400", "tgv_control_n400_repeat", "repeat of the control"),
                                 ("tgv_kl1_n400", "tgv_kl1_n400_repeat", "repeat of |k|l = 1"),
                                 ("tgv_control_n400", "tgv_control_n400_ma0p01", "control at Ma 0.01"),
                                 ("tgv_kl1_n400", "tgv_kl1_n400_ma0p01", "|k|l = 1 at Ma 0.01"),
                                 ("tgv_kl05_n400", "tgv_kl05_n800", "|k|l = 0.5 at 800^2"),
                                 ("tgv_control_n400", "tgv_control_n400_small_dt", "control at dt / 8.6")):
        if first in result["tgv"] and second in result["tgv"]:
            a, b = result["tgv"][first], result["tgv"][second]
            entry = {"rate_difference": b["rate"] / a["rate"] - 1.0}
            if "over_control" in a and "over_control" in b:
                entry["over_control_difference"] = b["over_control"] - a["over_control"]
            floors[label] = entry
            say(f"  {label}: rate {entry['rate_difference']:+.2e}"
                + (f", over control {entry['over_control_difference']:+.2e}" if "over_control_difference" in entry else ""))
    result["floors"] = floors
    out_dir = _REPO_ROOT / DATA_DIRECTORY
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "taylor_green.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (out_dir / "taylor_green.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    cases = commands.add_parser("cases")
    cases.add_argument("runs", nargs="*")
    run = commands.add_parser("run")
    run.add_argument("run", choices=sorted(RUNS))
    run.add_argument("--device", default="0")
    run.add_argument("--steps", type=int, default=None)
    run.add_argument("--depth", type=int, default=2)
    run.add_argument("--pool-safety", type=float, default=1.2)
    commands.add_parser("analyse")
    arguments = parser.parse_args()
    if arguments.command == "analyse":
        return command_analyse(arguments)
    if arguments.command == "cases":
        return command_cases(arguments)
    if arguments.command == "run":
        return command_run(arguments)
    return 1


if __name__ == "__main__":
    sys.exit(main())
