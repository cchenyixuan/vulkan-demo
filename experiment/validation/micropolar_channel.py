"""
micropolar_channel.py - Stage 8c: plane flows between two no-spin walls in a channel periodic in x
(docs/micropolar_design.md Section 19): steady plane Couette and Poiseuille flows against their exact solutions,
and the start-up of plane Couette flow against the host solution of micropolar_startup.py.

Geometry: fluid on the cell centres of [0, L) x [0, H), L = 100 dx (periodic in x), walls of WALL_LAYERS layers
below y = 0 and above y = H (adami, w = 0). Couette: the lower wall carries u = U (stationary wall particles with
the velocity U, as the lid of the cavity), the upper wall is at rest. Poiseuille: both walls at rest, a uniform
body force g = G / rho along x. The start-up is impulsive (u(0, t) = U for t > 0): the run loop of the solver
cannot change a wall's velocity during a run without a solver change, and the host solution uses the same start.

Exact solutions (K = kappa / mu, l^2 = gamma (mu + kappa) / (kappa (2 mu + kappa)), N^2 = kappa / (2 mu + kappa),
delta* = kappa l / (2 (mu + kappa)), w = 0 on both walls):
  Couette    T = (mu + kappa / 2) U / (H - 2 delta* tanh(H / (2 l)))                       (eq. 5.5)
             w(y) = T / (2 mu + kappa) [1 - cosh((y - H / 2) / l) / cosh(H / (2 l))]
             u(y) = U - [T y + kappa int_0^y w] / (mu + kappa)
  Poiseuille (y_c = y - a, a = H / 2, m = a / l)                                           (eqs. 2.24 - 2.26)
             w = G / (2 mu + kappa) [y_c - a sinh(y_c / l) / sinh(m)]
             u = G / (2 mu + kappa) (a^2 - y_c^2) - G N^2 a l / (mu + kappa) (cosh(m) - cosh(y_c / l)) / sinh(m)
             Q / Q_N = 1 - 3 N^2 / (1 + N^2) (coth(m) - 1 / m) / m

Wall stress, two readings: "fit" - u = a s + b s^2 fitted to the fluid within 4 dx of the wall (s the distance),
tau = rho nu a with the solver's nu ((mu + kappa) / rho for the micropolar fluid at a no-spin wall); "pairs" - the
force of the wall on the fluid from the solver's pair terms (micropolar_wall_torque.py's sums for plane walls,
the minimum image across the periodic seam), per unit length of wall, on the saved states.

  cases     the case directories of RUNS (particle files shared per geometry)
  run       one run (samples of the profiles u(y), w(y) and of the wall stress; wall-band states for "pairs")
  analyse   every finished run against the exact solution / the host start-up curves -> data/micropolar/stage8c/
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

RUNS_DIRECTORY = "logs/micropolar/stage8c"
DATA_DIRECTORY = "data/micropolar/stage8c"
SUPPORT_OVER_SPACING = 5
WALL_LAYERS = 11
LENGTH_IN_SPACINGS = 100
SPACING = 1.0e-3                       # m: the length unit of the cases
WALL_SPEED = 1.0                       # m/s: U of Couette, the centreline speed of the Newtonian Poiseuille flow
DENSITY = 1000.0
COUPLING_RATIO = 3.0
STEP_NUMBER, STEP_NUMBER_ANGULAR = 0.0328, 0.0334
FIT_SPACINGS = 4.0                     # the wall fit uses the fluid within 4 dx of the wall
STEADY_REYNOLDS = 10.0
STARTUP_REYNOLDS = 100.0
STEADY_END = 1.5                       # end time of a steady run in units of H^2 / nu_inf


# --- the table of runs -------------------------------------------------------------------------------------------
# flow: couette | poiseuille | startup; m = a / l = H / (2 l); length = l / dx; micropolar: False = the control
# (nu_inf on the micropolar path, nu_t = -nu, c = 0, xi = 0); mach; repeat; theta_factor = j / l^2.
RUNS: dict[str, dict] = {}


def _add(name: str, **entry) -> None:
    entry.setdefault("mach", 0.02)
    entry.setdefault("repeat", 0)
    entry.setdefault("theta_factor", 1.0)
    RUNS[name] = entry


for flow in ("couette", "poiseuille"):
    for m, length in ((1, 10), (2, 10), (5, 10), (10, 10), (2, 20), (5, 20)):
        for micropolar in (True, False):
            base = f"{flow}_m{m}_l{length}" + ("" if micropolar else "_control")
            _add(base, flow=flow, m=m, length=length, micropolar=micropolar)
            _add(base + "_repeat", flow=flow, m=m, length=length, micropolar=micropolar, repeat=1)
            _add(base + "_ma0p01", flow=flow, m=m, length=length, micropolar=micropolar, mach=0.01)
# the start-up: H = 20 l (m = 10), to the steady state at l = 20 dx, to Lambda = H / 2 otherwise
for length in (20, 40, 80):
    for micropolar in (True, False):
        base = f"startup_l{length}" + ("" if micropolar else "_control")
        _add(base, flow="startup", m=10, length=length, micropolar=micropolar)
for micropolar in (True, False):
    base = "startup_l40_j001" + ("" if micropolar else "_control")
    _add(base, flow="startup", m=10, length=40, micropolar=micropolar, theta_factor=0.01, own_timestep=True)
for micropolar in (True, False):
    base = "startup_l20" + ("" if micropolar else "_control")
    _add(base + "_repeat", flow="startup", m=10, length=20, micropolar=micropolar, repeat=1)
    _add(base + "_ma0p01", flow="startup", m=10, length=20, micropolar=micropolar, mach=0.01)


def height_in_spacings(entry: dict) -> int:
    return 2 * entry["m"] * entry["length"]


def far_viscosity(entry: dict) -> float:
    """nu_inf = (mu + kappa / 2) / rho from the Reynolds number: U H / nu_inf (Couette, start-up), U_c H / nu_inf with
    the Newtonian centreline speed U_c = WALL_SPEED (Poiseuille)."""
    height = height_in_spacings(entry) * SPACING
    reynolds = STARTUP_REYNOLDS if entry["flow"] == "startup" else STEADY_REYNOLDS
    return WALL_SPEED * height / reynolds


def design(entry: dict) -> dict:
    far = far_viscosity(entry)
    length = entry["length"] * SPACING
    if not entry["micropolar"]:
        return {"viscosity": far, "viscosity_transfer": -far, "coupling": 0.0, "viscosity_rotation": 0.0,
                "micro_inertia": length ** 2 * entry["theta_factor"], "length": None, "micropolar": False}
    from experiment.validation.micropolar_couette import design_parameters
    parameters = design_parameters(far, COUPLING_RATIO, length)
    parameters["micro_inertia"] *= entry["theta_factor"]
    return {**parameters, "length": length, "micropolar": True}


def body_force(entry: dict) -> float:
    """g = G / rho of the Poiseuille runs: the Newtonian (nu_inf) centreline speed WALL_SPEED."""
    if entry["flow"] != "poiseuille":
        return 0.0
    half = 0.5 * height_in_spacings(entry) * SPACING
    return 2.0 * far_viscosity(entry) * WALL_SPEED / half ** 2


def startup_end_time(entry: dict) -> float:
    far, height = far_viscosity(entry), height_in_spacings(entry) * SPACING
    if entry["length"] == 20 and entry["repeat"] == 0 and entry["mach"] == 0.02:
        return STEADY_END * height ** 2 / far                   # to the steady state (the floor runs: H / 2)
    return (0.5 * height) ** 2 / (math.pi * far)               # Lambda = sqrt(pi nu_inf t) = H / 2


def timestep_of(name: str) -> dict:
    entry = RUNS[name]
    support = SUPPORT_OVER_SPACING * SPACING
    speed_of_sound = WALL_SPEED / entry["mach"]
    # one dt for a run and its control: the family of the same flow, m, l, Mach and timestep class
    family = [other for other in RUNS.values() if other["flow"] == entry["flow"] and other["m"] == entry["m"]
              and other["length"] == entry["length"] and other["mach"] == entry["mach"]
              and other.get("own_timestep", False) == entry.get("own_timestep", False)]
    designs = [design(other) for other in family]
    dt_acoustic = 0.15 * support / speed_of_sound
    dt_viscous = STEP_NUMBER * support ** 2 / max(item["viscosity"] for item in designs)
    rates = [item["viscosity_rotation"] / item["micro_inertia"] for item in designs if item["viscosity_rotation"]]
    dt_angular = STEP_NUMBER_ANGULAR * support ** 2 / max(rates) if rates else float("inf")
    timestep = min(dt_acoustic, dt_viscous, dt_angular)
    governing = {dt_acoustic: "acoustic", dt_viscous: "viscous", dt_angular: "angular diffusion"}[timestep]
    if entry["flow"] == "startup":
        end_time = startup_end_time(entry)
    else:
        end_time = STEADY_END * (height_in_spacings(entry) * SPACING) ** 2 / far_viscosity(entry)
    steps = int(math.ceil(end_time / timestep / 1000.0 - 1e-9)) * 1000
    return {"timestep": timestep, "governing": governing, "speed_of_sound": speed_of_sound,
            "cfl": timestep * speed_of_sound / support, "steps": steps, "end_time": steps * timestep,
            "dt_acoustic": dt_acoustic, "dt_viscous": dt_viscous, "dt_angular": dt_angular}


# --- the exact steady solutions ----------------------------------------------------------------------------------
def coefficients(parameters: dict) -> dict:
    """mu, kappa, gamma (dynamic, rho = DENSITY) of a design; the control has kappa = 0, mu = rho nu_inf."""
    rho = DENSITY
    if not parameters["micropolar"]:
        return {"mu": rho * parameters["viscosity"], "kappa": 0.0, "gamma": 0.0, "length": None}
    kappa = rho * parameters["coupling"]
    mu = rho * parameters["viscosity"] - kappa
    return {"mu": mu, "kappa": kappa, "gamma": rho * parameters["viscosity_rotation"],
            "length": parameters["length"]}


def couette_exact(parameters: dict, height: float, y: np.ndarray) -> dict:
    """u(y), w(y), the stress magnitude T of plane Couette flow (lower wall at U, upper at rest)."""
    c = coefficients(parameters)
    mu, kappa, length = c["mu"], c["kappa"], c["length"]
    if length is None:
        stress = mu * WALL_SPEED / height
        return {"u": WALL_SPEED * (1.0 - y / height), "w": 0.5 * WALL_SPEED / height * np.ones_like(y),
                "stress": stress, "stress_newtonian": stress}
    displacement = kappa * length / (2.0 * (mu + kappa))
    stress = (mu + 0.5 * kappa) * WALL_SPEED / (height - 2.0 * displacement * math.tanh(height / (2.0 * length)))
    half = 0.5 * height / length
    w = stress / (2.0 * mu + kappa) * (1.0 - np.cosh((y - 0.5 * height) / length) / math.cosh(half))
    integral = stress / (2.0 * mu + kappa) * (y - length * (np.sinh((y - 0.5 * height) / length) + math.sinh(half))
                                              / math.cosh(half))
    u = WALL_SPEED - (stress * y + kappa * integral) / (mu + kappa)
    return {"u": u, "w": w, "stress": stress, "stress_newtonian": (mu + 0.5 * kappa) * WALL_SPEED / height}


def poiseuille_exact(parameters: dict, height: float, force: float, y: np.ndarray) -> dict:
    """u(y), w(y), Q of plane Poiseuille flow under the body force rho g = G, walls at y = 0, H."""
    c = coefficients(parameters)
    mu, kappa, length = c["mu"], c["kappa"], c["length"]
    pressure_gradient = DENSITY * force
    half = 0.5 * height
    centred = y - half
    if length is None:
        u = pressure_gradient / (2.0 * mu) * (half ** 2 - centred ** 2)
        return {"u": u, "w": 0.5 * pressure_gradient / mu * centred, "flow_rate": 2.0 * pressure_gradient * half ** 3
                / (3.0 * mu), "flow_rate_ratio": 1.0}
    m = half / length
    n_squared = kappa / (2.0 * mu + kappa)
    w = pressure_gradient / (2.0 * mu + kappa) * (centred - half * np.sinh(centred / length) / math.sinh(m))
    u = (pressure_gradient / (2.0 * mu + kappa) * (half ** 2 - centred ** 2)
         - pressure_gradient * n_squared * half * length / (mu + kappa)
         * (math.cosh(m) - np.cosh(centred / length)) / math.sinh(m))
    newtonian = 2.0 * pressure_gradient * half ** 3 / (3.0 * (mu + 0.5 * kappa))
    ratio = 1.0 - 3.0 * n_squared / (1.0 + n_squared) * (1.0 / math.tanh(m) - 1.0 / m) / m
    return {"u": u, "w": w, "flow_rate": newtonian * ratio, "flow_rate_ratio": ratio}


# --- cases -------------------------------------------------------------------------------------------------------
CASE_TEMPLATE = """schema_version: 2

# GENERATED by experiment/validation/micropolar_channel.py cases: {title}
# {flow_text}
# Channel periodic in x: [0, {length:.6f}) x [0, {height:.6f}) m, {columns} x {rows} = {fluid:,} fluid particles on the
# cell centres, dx = {spacing:g} m, h = {support} dx; walls of {layers} layers below y = 0 and above y = H (adami, w = 0),
# {wall:,} particles. {description}
# Speed of sound {speed_of_sound:g} m/s (Ma = {mach:g} on {speed_label}). dt = {timestep:.6e} s ({governing} limit;
# acoustic {dt_acoustic:.3e}, viscous {dt_viscous:.3e}, angular diffusion {dt_angular:.3e}); end {end_time:.6g} s =
# {steps:,} steps.

time:
  total: null
  max_steps: null
  output_cadence: null

physics:
  dimension: 2
  periodic: [true, false]
  h: {smoothing_length:.10e}
  particle_radius: {particle_radius:.10e}
  lattice: grid
  calibrate_volume: true
  speed_of_sound: {speed_of_sound!r}
  power: 7
  cfl: {cfl:.10f}
  gravity: [{gravity!r}, 0.0, 0.0]
  micropolar:
    enabled: true
    micro_inertia: {micro_inertia!r}

numerics:
  use_density_diffusion: true
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

capacities:
  pool_size: {pool_size}
  max_per_voxel: 64
  max_incoming: 16
  workgroup: 128

material_library: materials.yaml

geometry:
  frame: ../{geometry}/frame.obj
  particles:
    - {{file: ../{geometry}/domain.obj, material: channel_fluid}}
    - {{file: ../{geometry}/lower.obj,  material: {lower_material}}}
    - {{file: ../{geometry}/upper.obj,  material: channel_wall}}
"""
MATERIALS_TEMPLATE = """# GENERATED by experiment/validation/micropolar_channel.py with the case beside it.
schema_version: 1

channel_fluid:
  kind: fluid
  rest_density: {density!r}
  viscosity: {viscosity!r}
  viscosity_transfer: {viscosity_transfer!r}
  viscosity_rotation: {viscosity_rotation!r}

channel_wall:
  kind: boundary
  rest_density: {density!r}
  viscosity: {viscosity!r}
  wall_angular_velocity: [0.0, 0.0, 0.0]

channel_sliding_wall:
  kind: boundary
  rest_density: {density!r}
  viscosity: {viscosity!r}
  initial_velocity: [{wall_speed!r}, 0.0, 0.0]
  wall_angular_velocity: [0.0, 0.0, 0.0]
"""


def write_geometry(directory: pathlib.Path, rows: int) -> dict:
    columns = LENGTH_IN_SPACINGS
    x = SPACING * (np.arange(columns) + 0.5)
    fluid_y = SPACING * (np.arange(rows) + 0.5)
    lower_y = -SPACING * (np.arange(WALL_LAYERS) + 0.5)
    upper_y = SPACING * (rows + np.arange(WALL_LAYERS) + 0.5)
    directory.mkdir(parents=True, exist_ok=True)

    def write(path, ys):
        grid_x, grid_y = np.meshgrid(x, ys, indexing="ij")
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(f"# {grid_x.size} particles\n")
            np.savetxt(handle, np.column_stack([grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)]),
                       fmt="v %.12f %.12f %.1f")
        return grid_x.size
    fluid = write(directory / "domain.obj", fluid_y)
    lower = write(directory / "lower.obj", lower_y)
    upper = write(directory / "upper.obj", upper_y)
    length = columns * SPACING
    bottom = -(WALL_LAYERS + 0.6) * SPACING
    top = (rows + WALL_LAYERS + 0.6) * SPACING
    depth = 0.5 * SPACING
    corners = [(0.0, bottom, -depth), (0.0, top, -depth), (length, bottom, -depth), (length, top, -depth),
               (0.0, bottom, depth), (0.0, top, depth), (length, bottom, depth), (length, top, depth)]
    faces = ["f 1 2 4", "f 1 4 3", "f 5 7 8", "f 5 8 6", "f 1 5 6", "f 1 6 2", "f 3 4 8", "f 3 8 7", "f 1 3 7",
             "f 1 7 5", "f 2 6 8", "f 2 8 4"]
    (directory / "frame.obj").write_text("# channel frame: x = the periodic box, y = walls + 0.6 dx\n" + "".join(
        f"v {a:.12f} {b:.12f} {c:.12f}\n" for a, b, c in corners) + "\n".join(faces) + "\n", encoding="utf-8")
    return {"fluid": fluid, "wall": lower + upper}


def make_case(name: str) -> pathlib.Path:
    entry = RUNS[name]
    rows = height_in_spacings(entry)
    geometry = f"geometry_h{rows}"
    geometry_dir = _REPO_ROOT / "cases/channel" / geometry
    counts = write_geometry(geometry_dir, rows) if not (geometry_dir / "domain.obj").exists() else {
        "fluid": LENGTH_IN_SPACINGS * rows, "wall": 2 * LENGTH_IN_SPACINGS * WALL_LAYERS}
    parameters = design(entry)
    step = timestep_of(name)
    support = SUPPORT_OVER_SPACING * SPACING
    height = rows * SPACING
    far = far_viscosity(entry)
    flow_text = {"couette": "Steady plane Couette flow: the lower wall slides at U = 1 m/s, the upper wall at rest.",
                 "poiseuille": "Steady plane Poiseuille flow: walls at rest, a uniform body force along x.",
                 "startup": "Start-up of plane Couette flow: the lower wall slides at U = 1 m/s from t = 0 "
                            "(impulsive), the upper wall at rest."}[entry["flow"]]
    if parameters["micropolar"]:
        description = (f"Micropolar: K = {COUPLING_RATIO:g}, l = {entry['length']} dx, m = H / (2 l) = {entry['m']}, "
                       f"Theta = {entry['theta_factor']:g} l^2; nu = {parameters['viscosity']:.6g}, nu_t = "
                       f"{parameters['viscosity_transfer']:.6g}, c = {parameters['coupling']:.6g}, xi = "
                       f"{parameters['viscosity_rotation']:.6g}; nu_inf = {far:.6g} m^2/s.")
    else:
        description = (f"Control: nu = nu_inf = {far:.6g} m^2/s on the micropolar path, nu_t = -nu (c = 0), xi = 0, "
                       f"Theta = {parameters['micro_inertia']:.6g} (passive w).")
    case_dir = _REPO_ROOT / "cases/channel" / name
    case_dir.mkdir(parents=True, exist_ok=True)
    count = counts["fluid"] + counts["wall"]
    (case_dir / "case.yaml").write_text(CASE_TEMPLATE.format(
        title=name, flow_text=flow_text, length=LENGTH_IN_SPACINGS * SPACING, height=height,
        columns=LENGTH_IN_SPACINGS, rows=rows, fluid=counts["fluid"], spacing=SPACING, support=SUPPORT_OVER_SPACING,
        layers=WALL_LAYERS, wall=counts["wall"], description=description, speed_of_sound=step["speed_of_sound"],
        mach=entry["mach"], speed_label="U" if entry["flow"] != "poiseuille" else "the Newtonian centreline speed",
        timestep=step["timestep"], governing=step["governing"], dt_acoustic=step["dt_acoustic"],
        dt_viscous=step["dt_viscous"], dt_angular=step["dt_angular"], end_time=step["end_time"], steps=step["steps"],
        smoothing_length=support, particle_radius=0.5 * SPACING, cfl=step["cfl"], gravity=body_force(entry),
        micro_inertia=float(parameters["micro_inertia"]), pool_size=int(math.ceil(1.2 * count / 128.0)) * 128,
        geometry=geometry, lower_material="channel_wall" if entry["flow"] == "poiseuille" else "channel_sliding_wall"),
        encoding="utf-8", newline="\n")
    (case_dir / "materials.yaml").write_text(MATERIALS_TEMPLATE.format(
        density=DENSITY, viscosity=float(parameters["viscosity"]),
        viscosity_transfer=float(parameters["viscosity_transfer"]),
        viscosity_rotation=float(parameters["viscosity_rotation"]), wall_speed=WALL_SPEED), encoding="utf-8",
        newline="\n")
    (case_dir / "channel.json").write_text(json.dumps({
        "run": name, **entry, "spacing": SPACING, "smoothing_length": support, "height": height,
        "length_x": LENGTH_IN_SPACINGS * SPACING, "rows": rows, "far_viscosity": far, "body_force": body_force(entry),
        "wall_speed": WALL_SPEED, "rest_density": DENSITY, "design": parameters, **step, **counts}, indent=1),
        encoding="utf-8")
    return case_dir


def command_cases(arguments) -> int:
    for name in arguments.runs or RUNS:
        print(f"[channel] {make_case(name).relative_to(_REPO_ROOT)}", flush=True)
    return 0


# --- run ---------------------------------------------------------------------------------------------------------
def sample_steps(name: str, steps: int, timestep: float) -> list:
    """Steady runs: 60 samples evenly to the end. Start-up: about 40 per decade of t from the 10th step, and the
    steps nearest the times at which the Newtonian layer reaches Lambda = 0.25 l, l, 4 l (Lambda = sqrt(pi nu_inf
    t), the early form; the analysis reads the exact Lambda(t) from the host solution)."""
    entry = RUNS[name]
    if entry["flow"] != "startup":
        chosen = {int(round(value)) for value in np.linspace(steps / 60.0, steps, 60)}
        return sorted({step - 1 if step % 1000 == 0 else step for step in chosen})
    chosen = set(np.unique(np.round(np.geomspace(10, steps, int(40 * math.log10(steps / 10.0)) + 1)).astype(int)))
    far = far_viscosity(entry)
    for factor in (0.25, 1.0, 4.0):
        layer = factor * entry["length"] * SPACING
        chosen.add(int(round(layer ** 2 / (math.pi * far) / timestep)))
    # never on a defrag step (a multiple of 1000): the step returns after the defrag has reordered the particles,
    # while the auxiliary buffers (acceleration, the wall's dummy values, M^-1) keep the order of the step
    return sorted({step - 1 if step % 1000 == 0 else step for step in chosen if 0 < step <= steps})


def profile_bins(position_y: np.ndarray, values: list, rows: int) -> list:
    index = np.clip(np.floor(position_y / SPACING).astype(np.int64), 0, rows - 1)
    count = np.bincount(index, minlength=rows).astype(np.float64)
    safe = np.where(count > 0, count, 1.0)
    return [np.bincount(index, value, rows) / safe for value in values] + [count]


def wall_fit(position_y: np.ndarray, velocity_x: np.ndarray, wall_y: float, wall_speed: float) -> dict:
    """u - U_wall = a s + b s^2 over the fluid within FIT_SPACINGS dx of the wall, s = |y - wall_y|."""
    distance = np.abs(position_y - wall_y)
    chosen = distance < FIT_SPACINGS * SPACING
    s = distance[chosen]
    design_matrix = np.column_stack([s, s * s])
    solution, *_ = np.linalg.lstsq(design_matrix, velocity_x[chosen] - wall_speed, rcond=None)
    return {"slope": float(solution[0]), "curvature": float(solution[1]), "particles": int(chosen.sum())}


def save_wall_band(path: pathlib.Path, rows_data: dict, fluid_group: int, height: float, micropolar: bool,
                   step: int) -> None:
    """The particles the pair sums need: fluid within 2 h + dx of a wall, wall particles within h + dx of the
    surface. kind: 0 fluid, 1 lower wall, 2 upper wall."""
    support = SUPPORT_OVER_SPACING * SPACING
    position = rows_data["position_voxel_id"][:, :2]
    y = position[:, 1].astype(np.float64)
    is_fluid = rows_data["material"] == fluid_group
    kind = np.where(is_fluid, 0, np.where(y < 0.5 * height, 1, 2)).astype(np.uint8)
    keep = np.where(is_fluid, (y < 2.0 * support + SPACING) | (y > height - 2.0 * support - SPACING),
                    (y > -(support + SPACING)) & (y < height + support + SPACING))
    dummy = rows_data.get("wall_dummy_velocity")
    arrays = {"position": position[keep], "velocity": rows_data["velocity_mass"][keep, :2],
              "density": rows_data["density_pressure"][keep, 0], "pressure": rows_data["density_pressure"][keep, 1],
              "acceleration": rows_data["acceleration"][keep, :2],
              "dummy_velocity": (dummy if dummy is not None else rows_data["velocity_mass"])[keep, :2],
              "kind": kind[keep], "mass": np.float64(rows_data["velocity_mass"][is_fluid, 3][0]),
              "step": np.int64(step)}
    if "correction_inverse" in rows_data:
        # the solver's own M_i^-1 of this step, packed (m00, m11, m22, m01, m02, m12): the pair sums use it
        arrays["correction_inverse"] = rows_data["correction_inverse"][keep, :6]
    if micropolar:
        arrays.update({"spin": rows_data["angular_velocity"][keep, 2],
                       "angular_acceleration": rows_data["angular_acceleration"][keep, 2],
                       "dummy_spin": rows_data.get("wall_dummy_angular_velocity",
                                                   rows_data["angular_velocity"])[keep, 2]})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def command_run(arguments) -> int:
    from experiment.validation import micropolar_tests as tests
    from experiment.validation.micropolar_couette import solver_revision
    from experiment.validation.micropolar_rotor3d import read_own_particles
    solver = tests._import_solver(_REPO_ROOT)
    load_case, compute_chain_partition, SphSimulatorV7, VulkanContextV7, ChainOrchestratorV7 = solver
    from experiment.v7.utils.case_v7 import KIND_FLUID
    import os
    name = arguments.run
    entry = RUNS[name]
    case_dir = make_case(name)
    meta = json.loads((case_dir / "channel.json").read_text(encoding="utf-8"))
    case = load_case(case_dir / "case.yaml")
    if not (case.physics.periodic[0] and not case.physics.periodic[1]):
        sys.exit(f"{name}: the loader did not read physics.periodic [true, false]")
    timestep = float(case.physics.timestep)
    steps = meta["steps"] if arguments.steps is None else arguments.steps
    rows = meta["rows"]
    height = meta["height"]
    fluid_group = next(index for index, material in enumerate(case.materials) if material.kind == KIND_FLUID)
    lower_speed = WALL_SPEED if entry["flow"] != "poiseuille" else 0.0
    micropolar = bool(case.physics.micropolar_enabled)
    chain = compute_chain_partition(case, [1.0], pool_safety=arguments.pool_safety)
    out_dir = (_REPO_ROOT / RUNS_DIRECTORY / name).resolve()
    (out_dir / "states").mkdir(parents=True, exist_ok=True)
    wanted = sorted(set(sample_steps(name, steps, timestep)))
    state_set = set(wanted) if entry["flow"] == "startup" else set(wanted[-5:])
    samples = {key: [] for key in ("step", "time", "u", "w", "count", "lower_fit", "upper_fit", "max_speed",
                                   "density_mean", "outside_channel", "flow_rate")}
    fields = ["density_pressure", "acceleration", "angular_velocity", "angular_acceleration", "wall_dummy_velocity",
              "wall_dummy_angular_velocity", "correction_inverse"]
    contexts, sims, statuses = [], [], []
    started = time.perf_counter()
    failure, bootstrap_seconds = None, None
    try:
        contexts.append(VulkanContextV7.create(device_index=int(arguments.device), enable_validation=False,
                                               application_name="channel"))
        sim = SphSimulatorV7(contexts[-1], chain.slabs[0], sync_scheme="per-direction")
        sims.append(sim)

        def sample(step: int) -> None:
            data = read_own_particles(sim, fields)
            fluid = data["material"] == fluid_group
            point = data["position_voxel_id"][fluid, :2].astype(np.float64)
            velocity = (data["velocity_mass"][fluid, :2].astype(np.float64)
                        + 0.5 * timestep * data["acceleration"][fluid, :2].astype(np.float64))
            if not np.isfinite(velocity).all():
                raise RuntimeError(f"step {step}: non-finite velocity")
            spin = (data["angular_velocity"][fluid, 2].astype(np.float64)
                    + 0.5 * timestep * data["angular_acceleration"][fluid, 2].astype(np.float64))
            u, w, count = profile_bins(point[:, 1], [velocity[:, 0], spin], rows)
            samples["step"].append(step)
            samples["time"].append(step * timestep)
            samples["u"].append(u)
            samples["w"].append(w)
            samples["count"].append(count)
            lower = wall_fit(point[:, 1], velocity[:, 0], 0.0, lower_speed)
            upper = wall_fit(point[:, 1], velocity[:, 0], height, 0.0)
            samples["lower_fit"].append([lower["slope"], lower["curvature"], lower["particles"]])
            samples["upper_fit"].append([upper["slope"], upper["curvature"], upper["particles"]])
            samples["max_speed"].append(float(np.abs(velocity).max()))
            samples["density_mean"].append(float(data["density_pressure"][fluid, 0].astype(np.float64).mean()))
            samples["outside_channel"].append(int(((point[:, 1] < 0.0) | (point[:, 1] >= height)).sum()))
            samples["flow_rate"].append(float(velocity[:, 0].sum() * SPACING * SPACING / meta["length_x"]))
            if step in state_set:
                save_wall_band(out_dir / "states" / f"state_{step:09d}.npz", data, fluid_group, height, micropolar,
                               step)

        with ChainOrchestratorV7(sims, defrag_cadence=case.numerics.defrag_cadence) as orchestrator:
            orchestrator.bootstrap_all()
            bootstrap_seconds = time.perf_counter() - started
            pending = list(wanted)
            try:
                for step in range(1, steps + 1):
                    record = orchestrator.step()
                    for item in record.get("defrag_report", []):
                        bad = {key: value for key, value in item.items()
                               if (key.startswith("overflow_") or key == "far_migration") and value}
                        if bad:
                            raise RuntimeError(f"invariant violation at step {step}: {bad}")
                    if pending and step == pending[0]:
                        pending.pop(0)
                        sample(step)
                    if step % 20000 == 0:
                        (out_dir / "heartbeat.json").write_text(json.dumps(
                            {"step": step, "steps": steps, "elapsed_seconds": time.perf_counter() - started}),
                            encoding="utf-8")
            except RuntimeError as error:
                failure = str(error)
            statuses = [item.readback_global_status() for item in sims]
    finally:
        for item in sims:
            item.destroy()
        for context in contexts:
            context.destroy()
    counters = {}
    for status in statuses:
        for key, value in status.items():
            if key.startswith("overflow_") or key in ("far_migration_count", "stamp_error_count"):
                counters[key] = counters.get(key, 0) + int(value)
    elapsed = time.perf_counter() - started
    np.savez_compressed(out_dir / "samples.npz", **{key: np.asarray(values) for key, values in samples.items()})
    record = {"run": name, "case": str(case_dir.relative_to(_REPO_ROOT)).replace("\\", "/"), "channel": meta,
              "revision": solver_revision(), "device": int(arguments.device), "steps": steps, "timestep": timestep,
              "particles": int(case.initial.positions.shape[0]), "counters": counters, "elapsed_seconds": elapsed,
              "bootstrap_seconds": bootstrap_seconds, "failure": failure,
              "delta_density": os.environ.get("V7_DELTA_DENSITY", "0"),
              "steps_per_second": steps / max(elapsed - (bootstrap_seconds or 0.0), 1e-9)}
    (out_dir / "run.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    zero = all(value == 0 for value in counters.values())
    print(f"[channel] {name}: {steps:,} steps (t = {steps * timestep:.5g} s) in {elapsed:.0f} s "
          f"({record['steps_per_second']:.0f} steps/s), counters {'all zero' if zero else counters}"
          + (f"; FAILED: {failure}" if failure else ""), flush=True)
    return 0 if zero and failure is None else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    cases = commands.add_parser("cases")
    cases.add_argument("runs", nargs="*")
    run = commands.add_parser("run")
    run.add_argument("run", choices=sorted(RUNS))
    run.add_argument("--device", default="0")
    run.add_argument("--steps", type=int, default=None)
    run.add_argument("--pool-safety", type=float, default=1.2)
    commands.add_parser("analyse")
    arguments = parser.parse_args()
    if arguments.command == "cases":
        return command_cases(arguments)
    if arguments.command == "run":
        return command_run(arguments)
    if arguments.command == "analyse":
        from experiment.validation.micropolar_channel_analysis import command_analyse
        return command_analyse(arguments)
    return 1


if __name__ == "__main__":
    sys.exit(main())
