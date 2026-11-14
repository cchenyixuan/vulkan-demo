"""
micropolar_dipole.py - Stage 8: the normal collision of a dipole with a no-slip wall (Clercx & Bruneau, Computers &
Fluids 35 (2006) 245-279), Newtonian against micropolar (docs/micropolar_design.md Section 17).

Box [-1, 1]^2 (half-width W = 1 m) with no-slip walls at rest (adami walls at the reference density, w = 0 on every
wall). Two shielded monopoles omega = +-omega_e (1 - r^2 / r0^2) exp(-r^2 / r0^2), r0 = 0.1, at (0, +0.1) (positive)
and (0, -0.1) (negative); omega_e = 299.528385375226 gives E(0) = 1/2 int u^2 = 2 (rms velocity U = 1 m/s) and
Omega(0) = 1/2 int omega^2 = 800, the reference values of the paper (its text says omega_e of about 320; 320 gives
E(0) = 2.28). The dipole moves in +x and meets the wall x = W first at t of about 0.37.

Reynolds number Re = U W / nu_inf = 625 for every run (nu_inf = (mu + kappa / 2) / rho; Re on mu is 1562.5 at
K = kappa / mu = 3). The control is the Newtonian fluid of viscosity nu_inf on the micropolar code path with
nu_t = -nu (the coupling c = nu + nu_t is exactly 0 in float32, test T2) and xi = 0, so its w is a passive constant.

Thesis configuration of Chapter 5 (the cavity's: laplacian viscous form, epsilon_squared_factor 1e-6, adami walls,
KGC regularization 1e-3, density diffusion on every pair, h = 5 dx, particle shifting as in the cavity cases).

Initial state of every run: the analytic velocity at the particles; w = omega / 2 from the analytic vorticity; the
density from the pressure of the incompressible flow (Poisson problem on the host, written into the simulator's
initial payload before its bootstrap; option --initial-pressure zero keeps the uniform rest density).

  cases      the geometry (particle files) and the case directories of RUNS
  run        one run of RUNS: samples of the diagnostics on a grid of one spacing (energy, enstrophy, palinstrophy,
             vorticity extrema, the wall x = W), vorticity fields at every 0.025, leakage through the walls
  analyse    the runs against the benchmark tables and against each other (data/micropolar/stage8/)

  python -m experiment.validation.micropolar_dipole cases
  python -m experiment.validation.micropolar_dipole run control_n1000 --device-map 0
  python -m experiment.validation.micropolar_dipole analyse
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import time

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---- the benchmark ----------------------------------------------------------------------------------------------
HALF_WIDTH = 1.0                      # W (m)
RMS_VELOCITY = 1.0                    # U (m/s): E(0) = 2 on the box of area 4
REYNOLDS = 625.0                      # U W / nu_inf
FAR_VISCOSITY = RMS_VELOCITY * HALF_WIDTH / REYNOLDS
MONOPOLE_RADIUS = 0.1                 # r0
EXTREMUM_VORTICITY = 299.528385375226  # omega_e: E(0) = 2 on [-1, 1]^2 (host check of Stage 8)
MONOPOLES = ((0.0, 0.1, +1.0), (0.0, -0.1, -1.0))   # (x, y, sign of the core vorticity)
END_TIME = 1.0
DENSITY = 1000.0
SUPPORT_OVER_SPACING = 5              # h / dx of the 2-D thesis cases
WALL_LAYERS = 11                      # as the cavity cases
COUPLING_RATIO = 3.0                  # K = kappa / mu of the micropolar runs
CONTROL_MICRO_INERTIA_LENGTH = 1.0 / 50.0   # Theta = l^2 of the control's passive w (any positive value)
STEP_NUMBER, STEP_NUMBER_ANGULAR, CFL = 0.0328, 0.0334, 0.15

# Clercx & Bruneau (2006), Re = 625: spectral (SM) and finite-difference (FD) values
BENCHMARK = {
    "enstrophy_peaks": {"SM": [(0.3711, 933.6), (0.6479, 305.2)], "FD": [(0.371, 932.8), (0.647, 305.2)]},
    "palinstrophy_peaks": {"SM": [(0.3624, 1.386e7), (0.6521, 6.777e5)], "FD": [(0.363, 1.296e7), (0.653, 6.561e5)]},
    "global": {"SM": {0.25: (1.5022, 472.6, 3.99e5), 0.50: (1.0130, 380.6, 5.53e5), 0.75: (0.7673, 255.2, 4.87e5)},
               "FD": {0.25: (1.502, 472.7, 3.91e5), 0.50: (1.013, 380.4, 5.49e5), 0.75: (0.767, 255.0, 4.73e5)}},
    "initial": {"energy": 2.0, "enstrophy": 800.0, "palinstrophy": 4.42e5},
    # positive primary vortex: (t, x, y, omega_max) by SM, FD
    "primary": {"SM": [(0.6, 0.818, 0.165, 158.9), (1.0, 0.805, 0.254, 102.6)],
                "FD": [(0.625, 0.832, 0.166, 154.2), (1.0, 0.805, 0.254, 102.6)]},
}

# run -> resolution N (dx = W / N), micropolar length l / W (None: the control), speed of sound c0 (m/s),
# shift of the dipole along x in spacings, initial pressure, slabs
RUNS = {
    "control_n250": (250, None, 600.0, 0.0, "incompressible", 1),
    "control_n500": (500, None, 600.0, 0.0, "incompressible", 1),
    "control_n500_zero_pressure": (500, None, 600.0, 0.0, "zero", 1),
    "control_n1000": (1000, None, 600.0, 0.0, "incompressible", 1),
    "control_n1000_half_mach": (1000, None, 1200.0, 0.0, "incompressible", 1),
    "control_n1000_shifted": (1000, None, 600.0, 0.5, "incompressible", 1),
    "mp_l50_n1000": (1000, 1.0 / 50.0, 600.0, 0.0, "incompressible", 1),
    "mp_l100_n1000": (1000, 1.0 / 100.0, 600.0, 0.0, "incompressible", 1),
    "control_n2000": (2000, None, 600.0, 0.0, "incompressible", 2),
    "mp_l100_n2000": (2000, 1.0 / 100.0, 600.0, 0.0, "incompressible", 2),
    # Stage 8 checks (13 Nov task, Part 0): the micro-inertia (j = l^2 / 100), the saturation (l = W / 25), the
    # resolution of l (l = W / 50 at 10 dx)
    "mp_l50_j001_n1000": (1000, 1.0 / 50.0, 600.0, 0.0, "incompressible", 1),
    "mp_l25_j001_n1000": (1000, 1.0 / 25.0, 600.0, 0.0, "incompressible", 1),
    "mp_l50_n500": (500, 1.0 / 50.0, 600.0, 0.0, "incompressible", 1),
}
# run -> Theta / l^2 (absent: 1, the thesis convention j = l^2)
MICRO_INERTIA_FACTOR = {"mp_l50_j001_n1000": 0.01, "mp_l25_j001_n1000": 0.01}
RUNS_DIRECTORY = "logs/micropolar/stage8"
DATA_DIRECTORY = "data/micropolar/stage8"
SNAPSHOT_INTERVAL = 0.025
SAMPLE_INTERVAL = 0.01
WINDOW_INTERVAL = 0.0025
WINDOWS = ((0.30, 0.44), (0.58, 0.72))   # around the first and second collision
WALL_BAND_NODES = 48                      # nodes next to x = W kept at every sample (profiles normal to the wall)
EXTREMA_KEPT = 6

CASE_TEMPLATE = """schema_version: 2

# GENERATED by experiment/validation/micropolar_dipole.py cases: {title}
# Normal dipole-wall collision (Clercx & Bruneau 2006) in the box [-1, 1]^2 m, no-slip walls at rest
# ({wall_layers} layers, adami); dx = {spacing:.6f} m, h = {smoothing_length:.6f} m = {support} dx; fluid {fluid_count:,}
# particles, walls {wall_count:,}.
# Thesis configuration of Chapter 5 (the cavity's numerics: laplacian, epsilon_squared_factor 1e-6, adami walls,
# KGC regularization 1e-3, density diffusion on every pair, particle shifting).
# {parameters}
# Re = U W / nu_inf = {reynolds:g} (U = 1 m/s rms velocity, W = 1 m). Speed of sound {speed_of_sound:g} m/s: Ma =
# {mach_peak:.4f} on the largest initial speed ({peak_speed:.4f} m/s), {mach_rms:.5f} on U.
# Time step: dt = min(CFL h / c0 = {dt_acoustic:.4e}, s h^2 / nu = {dt_viscous:.4e}, s h^2 / (xi / Theta) =
# {dt_angular:.4e}) with CFL = {cfl_nominal:g}, s = {step_number:g} / {step_number_angular:g}: dt = {timestep:.6e} s,
# set by the {governing} limit. physics.cfl below is dt c0 / h = {cfl_effective:.8f}. End time {end_time:g} s =
# {steps:,} steps.

time:
  total: null
  max_steps: null
  output_cadence: null

physics:
  dimension: 2
  h: {smoothing_length:.8f}
  particle_radius: {particle_radius:.8f}
  lattice: grid
  calibrate_volume: true
  speed_of_sound: {speed_of_sound:.6f}
  power: 7
  cfl: {cfl_effective:.8f}
  gravity: [0.0, 0.0, 0.0]
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
  max_per_voxel: 96
  max_incoming: 16
  workgroup: 128

material_library: materials.yaml

geometry:
  frame: {geometry}/frame.obj
  particles:
    - {{file: {geometry}/domain.obj, material: dipole_fluid}}
    - {{file: {geometry}/wall.obj,   material: dipole_wall}}
"""

MATERIALS_TEMPLATE = """# GENERATED by experiment/validation/micropolar_dipole.py with the case beside it.
schema_version: 1

dipole_fluid:
  kind: fluid
  rest_density: {density}
  viscosity: {viscosity!r}
  viscosity_transfer: {viscosity_transfer!r}
  viscosity_rotation: {viscosity_rotation!r}

dipole_wall:
  kind: boundary
  rest_density: {density}
  viscosity: {viscosity!r}
  wall_angular_velocity: [0.0, 0.0, 0.0]
"""


# ---- the analytic initial state --------------------------------------------------------------------------------
def analytic_fields(x: np.ndarray, y: np.ndarray, shift_x: float = 0.0) -> dict:
    """Velocity, vorticity and the velocity gradient of the two isolated monopoles (eq. 7 and 10 of the paper),
    the dipole moved by shift_x along x."""
    u, v, vorticity = (np.zeros_like(x) for _ in range(3))
    du_dx, du_dy, dv_dx, dv_dy = (np.zeros_like(x) for _ in range(4))
    for centre_x, centre_y, sign in MONOPOLES:
        dx, dy = x - (centre_x + shift_x), y - centre_y
        ratio = (dx ** 2 + dy ** 2) / MONOPOLE_RADIUS ** 2
        envelope = np.exp(-ratio)
        amplitude = 0.5 * sign * EXTREMUM_VORTICITY
        u += -amplitude * dy * envelope
        v += amplitude * dx * envelope
        vorticity += sign * EXTREMUM_VORTICITY * (1.0 - ratio) * envelope
        factor = -2.0 / MONOPOLE_RADIUS ** 2
        du_dx += -amplitude * dy * envelope * factor * dx
        du_dy += -amplitude * envelope * (1.0 + factor * dy * dy)
        dv_dx += amplitude * envelope * (1.0 + factor * dx * dx)
        dv_dy += amplitude * dx * envelope * factor * dy
    return {"u": u, "v": v, "vorticity": vorticity, "du_dx": du_dx, "du_dy": du_dy, "dv_dx": dv_dx, "dv_dy": dv_dy}


def incompressible_pressure(shift_x: float, spacing: float) -> tuple:
    """Pressure / rho of the incompressible flow of the analytic velocity: lap p = 2 (u_x v_y - u_y v_x), zero normal
    derivative on the walls (the velocity vanishes there to 1e-34), solved by the cosine transform on the
    cell-centred grid of spacing / 2; p = 0 on the walls (the field decays as exp(-2 r^2 / r0^2)). Returns the
    node coordinates and p / rho (m^2/s^2)."""
    from scipy.fft import dctn, idctn
    fine = 0.5 * spacing
    count = int(round(2.0 * HALF_WIDTH / fine))
    axis = -HALF_WIDTH + fine * (np.arange(count) + 0.5)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    fields = analytic_fields(x, y, shift_x)
    source = 2.0 * (fields["du_dx"] * fields["dv_dy"] - fields["du_dy"] * fields["dv_dx"])
    transformed = dctn(source, type=2, norm="ortho")
    eigenvalue = (2.0 / fine * np.sin(np.pi * np.arange(count) / (2.0 * count))) ** 2
    denominator = -(eigenvalue[:, None] + eigenvalue[None, :])
    denominator[0, 0] = 1.0
    transformed /= denominator
    transformed[0, 0] = 0.0
    pressure = idctn(transformed, type=2, norm="ortho")
    pressure -= np.mean(np.concatenate([pressure[0, :], pressure[-1, :], pressure[:, 0], pressure[:, -1]]))
    return axis, pressure


def bilinear(axis: np.ndarray, grid: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    from scipy.interpolate import RegularGridInterpolator
    interpolator = RegularGridInterpolator((axis, axis), grid, bounds_error=False, fill_value=None)
    return interpolator(np.column_stack([x, y]))


def peak_initial_speed(shift_x: float = 0.0) -> float:
    axis = np.linspace(-0.3, 0.3, 1201)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    fields = analytic_fields(x, y, shift_x)
    return float(np.sqrt(fields["u"] ** 2 + fields["v"] ** 2).max())


# ---- cases ------------------------------------------------------------------------------------------------------
def design(run: str) -> dict:
    from experiment.validation.micropolar_couette import design_parameters
    length = RUNS[run][1]
    if length is None:
        # the control: nu = nu_inf, nu_t = -nu (c = 0 exactly), xi = 0, a positive Theta for the passive w
        return {"viscosity": FAR_VISCOSITY, "viscosity_transfer": -FAR_VISCOSITY, "coupling": 0.0,
                "viscosity_rotation": 0.0, "micro_inertia": CONTROL_MICRO_INERTIA_LENGTH ** 2, "length": None}
    parameters = design_parameters(FAR_VISCOSITY, COUPLING_RATIO, length * HALF_WIDTH)
    parameters["micro_inertia"] *= MICRO_INERTIA_FACTOR.get(run, 1.0)
    return {**parameters, "length": length * HALF_WIDTH}


def timestep_of(run: str) -> dict:
    """The step of a run: the smallest of the acoustic limit and the two diffusive limits, the diffusive ones
    taken over every micropolar run of the same resolution so that the control and its micropolar runs share dt."""
    resolution, _length, speed_of_sound = RUNS[run][:3]
    spacing = HALF_WIDTH / resolution
    support = SUPPORT_OVER_SPACING * spacing
    family = [name for name, entry in RUNS.items() if entry[0] == resolution]
    designs = [design(name) for name in family]
    dt_acoustic = CFL * support / speed_of_sound
    dt_viscous = STEP_NUMBER * support ** 2 / max(entry["viscosity"] for entry in designs)
    rates = [entry["viscosity_rotation"] / entry["micro_inertia"] for entry in designs if entry["viscosity_rotation"]]
    dt_angular = STEP_NUMBER_ANGULAR * support ** 2 / max(rates) if rates else float("inf")
    timestep = min(dt_acoustic, dt_viscous, dt_angular)
    governing = {dt_acoustic: "acoustic", dt_viscous: "viscous", dt_angular: "angular diffusion"}[timestep]
    steps = int(np.ceil(END_TIME / timestep / 1000.0 - 1e-9)) * 1000
    return {"dt_acoustic": dt_acoustic, "dt_viscous": dt_viscous, "dt_angular": dt_angular, "timestep": timestep,
            "governing": governing, "steps": steps, "cfl_effective": timestep * speed_of_sound / support}


def write_geometry(directory: pathlib.Path, resolution: int) -> dict:
    """Fluid on the cell centres of [-1, 1]^2 (2 N per side), WALL_LAYERS of wall particles outside every side
    (corners filled), the frame 0.6 dx outside the walls (as the cavity cases)."""
    spacing = HALF_WIDTH / resolution
    count = 2 * resolution
    index = np.arange(-WALL_LAYERS, count + WALL_LAYERS)
    coordinate = -HALF_WIDTH + spacing * (index + 0.5)
    x, y = np.meshgrid(coordinate, coordinate, indexing="ij")
    inside = (index[:, None] >= 0) & (index[:, None] < count) & (index[None, :] >= 0) & (index[None, :] < count)
    directory.mkdir(parents=True, exist_ok=True)

    def write(path, points):
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(f"# {points.shape[0]} particles\n")
            np.savetxt(handle, np.column_stack([points, np.zeros(points.shape[0])]), fmt="v %.9f %.9f %.9f")
    fluid = np.column_stack([x[inside], y[inside]])
    wall = np.column_stack([x[~inside], y[~inside]])
    write(directory / "domain.obj", fluid)
    write(directory / "wall.obj", wall)
    half = HALF_WIDTH + WALL_LAYERS * spacing + 0.6 * spacing
    depth = 0.5 * spacing
    corners = [(-half, -half, -depth), (-half, half, -depth), (half, -half, -depth), (half, half, -depth),
               (-half, -half, depth), (-half, half, depth), (half, -half, depth), (half, half, depth)]
    faces = ["f 1 2 4", "f 1 4 3", "f 5 7 8", "f 5 8 6", "f 1 5 6", "f 1 6 2", "f 3 4 8", "f 3 8 7", "f 1 3 7",
             "f 1 7 5", "f 2 6 8", "f 2 8 4"]
    (directory / "frame.obj").write_text("# dipole box frame\n" + "".join(
        f"v {a:.6f} {b:.6f} {c:.6f}\n" for a, b, c in corners) + "\n".join(faces) + "\n", encoding="utf-8")
    return {"fluid": int(fluid.shape[0]), "wall": int(wall.shape[0])}


def make_case(run: str) -> pathlib.Path:
    resolution, length, speed_of_sound, shift, initial_pressure, slabs = RUNS[run]
    spacing = HALF_WIDTH / resolution
    support = SUPPORT_OVER_SPACING * spacing
    geometry_dir = _REPO_ROOT / f"cases/dipole/geometry_n{resolution}"
    meta_file = geometry_dir / "geometry.json"
    if meta_file.exists():
        counts = json.loads(meta_file.read_text(encoding="utf-8"))["counts"]
    else:
        counts = write_geometry(geometry_dir, resolution)
        meta_file.write_text(json.dumps({"resolution": resolution, "spacing": spacing, "wall_layers": WALL_LAYERS,
                                         "counts": counts}, indent=1), encoding="utf-8")
    parameters = design(run)
    step = timestep_of(run)
    peak = peak_initial_speed(shift * spacing)
    if length is None:
        text = (f"Control: Newtonian fluid of viscosity nu_inf = {FAR_VISCOSITY:.6e} on the micropolar path, nu_t = -nu "
                f"(c = 0 exactly), xi = 0, Theta = {parameters['micro_inertia']:.3e} (passive w).")
    else:
        text = (f"Micropolar: K = kappa / mu = {COUPLING_RATIO:g}, l = W / {1.0 / length:g} = {length:g} m "
                f"({length / spacing:g} dx), Theta = {MICRO_INERTIA_FACTOR.get(run, 1.0):g} l^2 = "
                f"{parameters['micro_inertia']:.6e}; nu = {parameters['viscosity']:.6e}, nu_t = "
                f"{parameters['viscosity_transfer']:.6e}, c = {parameters['coupling']:.6e}, xi = "
                f"{parameters['viscosity_rotation']:.6e}; w = 0 on every wall.")
    case_dir = _REPO_ROOT / f"cases/dipole/{run}"
    case_dir.mkdir(parents=True, exist_ok=True)
    total = counts["fluid"] + counts["wall"]
    pool_size = int(np.ceil(1.15 * total / 128.0)) * 128
    relative_geometry = pathlib.Path("..") / geometry_dir.name
    (case_dir / "case.yaml").write_text(CASE_TEMPLATE.format(
        title=run, wall_layers=WALL_LAYERS, spacing=spacing, smoothing_length=support, support=SUPPORT_OVER_SPACING,
        fluid_count=counts["fluid"], wall_count=counts["wall"], parameters=text, reynolds=REYNOLDS,
        speed_of_sound=speed_of_sound, mach_peak=peak / speed_of_sound, peak_speed=peak,
        mach_rms=RMS_VELOCITY / speed_of_sound, dt_acoustic=step["dt_acoustic"], dt_viscous=step["dt_viscous"],
        dt_angular=step["dt_angular"], cfl_nominal=CFL, step_number=STEP_NUMBER,
        step_number_angular=STEP_NUMBER_ANGULAR, timestep=step["timestep"], governing=step["governing"],
        cfl_effective=step["cfl_effective"], end_time=END_TIME, steps=step["steps"], particle_radius=0.5 * spacing,
        micro_inertia=float(parameters["micro_inertia"]), pool_size=pool_size,
        geometry=relative_geometry.as_posix()), encoding="utf-8", newline="\n")
    (case_dir / "materials.yaml").write_text(MATERIALS_TEMPLATE.format(
        density=DENSITY, viscosity=float(parameters["viscosity"]),
        viscosity_transfer=float(parameters["viscosity_transfer"]),
        viscosity_rotation=float(parameters["viscosity_rotation"])), encoding="utf-8", newline="\n")
    (case_dir / "dipole.json").write_text(json.dumps({
        "run": run, "resolution": resolution, "spacing": spacing, "smoothing_length": support,
        "micropolar_length": parameters["length"], "coupling_ratio": None if length is None else COUPLING_RATIO,
        "design": {key: value for key, value in parameters.items() if key != "length"},
        "far_viscosity": FAR_VISCOSITY, "reynolds": REYNOLDS,
        "reynolds_on_mu": REYNOLDS * (1.0 + 0.5 * (0.0 if length is None else COUPLING_RATIO)),
        "speed_of_sound": speed_of_sound, "peak_initial_speed": peak, "mach_peak": peak / speed_of_sound,
        "shift_in_spacings": shift, "initial_pressure": initial_pressure, "slabs": slabs,
        "extremum_vorticity": EXTREMUM_VORTICITY, "monopole_radius": MONOPOLE_RADIUS, "counts": counts,
        **step}, indent=1), encoding="utf-8")
    return case_dir


def command_cases(arguments) -> int:
    for run in (arguments.runs or RUNS):
        print(f"[dipole] {make_case(run).relative_to(_REPO_ROOT)}", flush=True)
    return 0


# ---- diagnostics on a grid of one spacing ----------------------------------------------------------------------
class Grid:
    """Cell-centred nodes of [-1, 1]^2 at the particle spacing. Particle values go to the nodes by the quadratic
    B-spline (three nodes per axis) normalised by the deposited volume (Shepard); the fluid within 1.5 dx of a
    wall is mirrored across it with the velocity and w reversed (the no-slip, no-spin extension) and the pressure
    kept, so the nodes next to a wall see a full neighbourhood. Derivatives: central differences, the ghost node
    across a wall being the odd (velocity, w) reflection of the first node."""

    PAD = 3                    # padding nodes on every side: a mirrored particle lies up to 1.5 dx outside

    def __init__(self, resolution: int):
        self.count = 2 * resolution
        self.spacing = HALF_WIDTH / resolution
        self.axis = -HALF_WIDTH + self.spacing * (np.arange(self.count) + 0.5)
        self.padded = self.count + 2 * self.PAD

    def _mirrored(self, position: np.ndarray, odd: list, even: list):
        reach = HALF_WIDTH - 1.5 * self.spacing
        positions, odd_values, even_values = [position], [list(odd)], [list(even)]
        for axis in (0, 1):
            for side in (-1.0, 1.0):
                near = (side * position[:, axis] > reach)
                if not near.any():
                    continue
                mirrored = position[near].copy()
                mirrored[:, axis] = 2.0 * side * HALF_WIDTH - mirrored[:, axis]
                positions.append(mirrored)
                odd_values.append([-values[near] for values in odd])
                even_values.append([values[near] for values in even])
        # corners: across both walls (the reflection through the corner keeps the sign of odd fields)
        for side_x in (-1.0, 1.0):
            for side_y in (-1.0, 1.0):
                near = (side_x * position[:, 0] > reach) & (side_y * position[:, 1] > reach)
                if not near.any():
                    continue
                mirrored = position[near].copy()
                mirrored[:, 0] = 2.0 * side_x * HALF_WIDTH - mirrored[:, 0]
                mirrored[:, 1] = 2.0 * side_y * HALF_WIDTH - mirrored[:, 1]
                positions.append(mirrored)
                odd_values.append([values[near] for values in odd])
                even_values.append([values[near] for values in even])
        position = np.concatenate(positions)
        odd = [np.concatenate([group[index] for group in odd_values]) for index in range(len(odd))]
        even = [np.concatenate([group[index] for group in even_values]) for index in range(len(even))]
        return position, odd, even

    def deposit(self, position: np.ndarray, odd: list, even: list) -> list:
        """Node values (count x count) of the fields `odd` (reversed across a wall) and `even` (kept)."""
        position, odd, even = self._mirrored(position.astype(np.float64), odd, even)
        fields = [values.astype(np.float64) for values in odd + even]
        coordinate = (position + HALF_WIDTH) / self.spacing - 0.5
        nearest = np.rint(coordinate).astype(np.int64)
        offset = coordinate - nearest
        weights = [0.5 * (0.5 - offset) ** 2, 0.75 - offset ** 2, 0.5 * (0.5 + offset) ** 2]
        size = self.padded * self.padded
        volume = np.zeros(size)
        sums = [np.zeros(size) for _ in fields]
        for step_x in range(3):
            column = nearest[:, 0] + (step_x - 1) + self.PAD
            for step_y in range(3):
                row = nearest[:, 1] + (step_y - 1) + self.PAD
                flat = column * self.padded + row
                weight = weights[step_x][:, 0] * weights[step_y][:, 1]
                volume += np.bincount(flat, weight, size)
                for total, values in zip(sums, fields):
                    total += np.bincount(flat, weight * values, size)
        inner = (slice(self.PAD, self.PAD + self.count), slice(self.PAD, self.PAD + self.count))
        volume = volume.reshape(self.padded, self.padded)[inner]
        safe = np.where(volume > 0.0, volume, 1.0)
        return [np.where(volume > 0.0, total.reshape(self.padded, self.padded)[inner] / safe, 0.0) for total in sums]

    @staticmethod
    def odd_padded(values: np.ndarray) -> np.ndarray:
        padded = np.pad(values, 1, mode="symmetric")
        padded[0, :] *= -1.0
        padded[-1, :] *= -1.0
        padded[:, 0] *= -1.0
        padded[:, -1] *= -1.0
        return padded

    def vorticity(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """dv/dx - du/dy at the nodes (index 0 = x, index 1 = y)."""
        u_padded, v_padded = self.odd_padded(u), self.odd_padded(v)
        dv_dx = (v_padded[2:, 1:-1] - v_padded[:-2, 1:-1]) / (2.0 * self.spacing)
        du_dy = (u_padded[1:-1, 2:] - u_padded[1:-1, :-2]) / (2.0 * self.spacing)
        return dv_dx - du_dy


def extrema(field: np.ndarray, axis: np.ndarray, upper_half: bool, sign: float, kept: int = EXTREMA_KEPT,
            threshold: float = 10.0) -> np.ndarray:
    """Local extrema of sign * field above threshold (5 x 5 neighbourhood) in the upper half y > 0: rows (value,
    x, y) refined by a parabola through the node and its two neighbours on each axis, strongest first."""
    from scipy.ndimage import maximum_filter
    signed = sign * field
    peak = (signed == maximum_filter(signed, size=5, mode="nearest")) & (signed > threshold)
    if upper_half:
        peak[:, : field.shape[1] // 2] = False
    rows = []
    for column, row in zip(*np.nonzero(peak)):
        value = signed[column, row]
        position = []
        for axis_index, (low, high) in enumerate(((column - 1, column + 1), (row - 1, row + 1))):
            index = column if axis_index == 0 else row
            if low < 0 or high >= field.shape[axis_index]:
                position.append(axis[index])
                continue
            if axis_index == 0:
                left, right = signed[low, row], signed[high, row]
            else:
                left, right = signed[column, low], signed[column, high]
            curvature = left - 2.0 * value + right
            shift = 0.5 * (left - right) / curvature if curvature < 0.0 else 0.0
            position.append(axis[index] + float(np.clip(shift, -0.5, 0.5)) * (axis[1] - axis[0]))
        rows.append((sign * value, position[0], position[1]))
    rows.sort(key=lambda entry: -abs(entry[0]))
    table = np.full((kept, 3), np.nan)
    if rows:
        table[:min(kept, len(rows))] = np.asarray(rows[:kept])
    return table


def wall_fit(position: np.ndarray, value_list: list, spacing: float) -> tuple:
    """Slopes a of value = a s + b s^2 (value 0 on the wall) at the wall x = W, s = W - x, from the fluid within
    4 dx of it, per bin of 2 dx in y. Returns the bin centres and one array of slopes per value."""
    distance = HALF_WIDTH - position[:, 0]
    chosen = distance < 4.0 * spacing
    s = distance[chosen]
    width = 2.0 * spacing
    bins = int(round(2.0 * HALF_WIDTH / width))
    index = np.clip(((position[chosen, 1] + HALF_WIDTH) / width).astype(np.int64), 0, bins - 1)
    second = np.bincount(index, s ** 2, bins)
    third = np.bincount(index, s ** 3, bins)
    fourth = np.bincount(index, s ** 4, bins)
    determinant = second * fourth - third ** 2
    safe = np.where(np.abs(determinant) > 0.0, determinant, 1.0)
    slopes = []
    for values in value_list:
        chosen_values = values[chosen]
        first_moment = np.bincount(index, chosen_values * s, bins)
        second_moment = np.bincount(index, chosen_values * s ** 2, bins)
        slopes.append(np.where(np.abs(determinant) > 0.0, (first_moment * fourth - second_moment * third) / safe,
                               np.nan))
    centres = -HALF_WIDTH + width * (np.arange(bins) + 0.5)
    return centres, slopes


def diagnostics(grid: Grid, position: np.ndarray, velocity: np.ndarray, pressure: np.ndarray, spin, volume: float,
                speed_of_sound: float, keep_fields: bool) -> dict:
    """Everything a sample records from the fluid rows (position, velocity: (n, 2); pressure / rho: (n,); spin w_z:
    (n,) or None)."""
    u, v = velocity[:, 0].astype(np.float64), velocity[:, 1].astype(np.float64)
    odd = [u, v] + ([spin.astype(np.float64)] if spin is not None else [])
    nodes = grid.deposit(position, odd, [pressure.astype(np.float64)])
    grid_u, grid_v = nodes[0], nodes[1]
    grid_spin = nodes[2] if spin is not None else None
    grid_pressure = nodes[-1]
    vorticity = grid.vorticity(grid_u, grid_v)
    area = grid.spacing ** 2
    gradient_x, gradient_y = np.gradient(vorticity, grid.spacing)
    result = {
        "energy": 0.5 * volume * float((u ** 2 + v ** 2).sum()),
        "energy_grid": 0.5 * area * float((grid_u ** 2 + grid_v ** 2).sum()),
        "enstrophy": 0.5 * area * float((vorticity ** 2).sum()),
        "palinstrophy": 0.5 * area * float((gradient_x ** 2 + gradient_y ** 2).sum()),
        "angular_momentum": volume * float((position[:, 0] * v - position[:, 1] * u).sum()),
        "max_speed": float(np.sqrt((u ** 2 + v ** 2).max())),
        "outside_box": int((np.abs(position) > HALF_WIDTH).any(axis=1).sum()),
        "positive_extrema": extrema(vorticity, grid.axis, True, +1.0),
        "negative_extrema": extrema(vorticity, grid.axis, True, -1.0),
        "lower_negative_extrema": extrema(vorticity[:, ::-1], grid.axis, True, -1.0),
    }
    # the wall x = W: vorticity (u = 0 along the wall, so omega = dv/dx there) from v(s) = a s + b s^2 through the
    # two nodes nearest the wall (s = W - x), and the pressure from p(s) = p_w + c s^2
    last, before = grid_v[-1, :], grid_v[-2, :]
    result["wall_vorticity"] = -(9.0 * last - before) / (3.0 * grid.spacing)
    result["wall_pressure"] = (9.0 * grid_pressure[-1, :] - grid_pressure[-2, :]) / 8.0
    # the same wall from the particles: least squares v = a s + b s^2 (and w = c s + d s^2) over the fluid within
    # 4 dx of the wall, in bins of 2 dx along it (no smoothing of the grid; the primary reading of the wall)
    fit_values = [velocity[:, 1].astype(np.float64)] + ([spin.astype(np.float64)] if spin is not None else [])
    fit_y, slopes = wall_fit(position, fit_values, grid.spacing)
    result["wall_fit_y"] = fit_y
    result["wall_fit_vorticity"] = -slopes[0]
    if spin is not None:
        result["wall_fit_spin_gradient"] = slopes[1]                     # dw/ds at the wall, s = W - x
    result["band_vorticity"] = vorticity[-WALL_BAND_NODES:, :].astype(np.float32)
    result["band_tangential_velocity"] = grid_v[-WALL_BAND_NODES:, :].astype(np.float32)
    if grid_spin is not None:
        result["band_spin"] = grid_spin[-WALL_BAND_NODES:, :].astype(np.float32)
        result["spin_departure"] = area * float(((grid_spin - 0.5 * vorticity) ** 2).sum())
        result["spin_departure_band"] = area * float(((grid_spin - 0.5 * vorticity)[-WALL_BAND_NODES:, :] ** 2).sum())
    # leakage through the walls: outward normal velocity of the fluid within one spacing of a wall against
    # p / (rho c0) (docs: eq. 5.13 of the thesis, v_n / U = k_w Ma p / (rho U^2))
    distance = HALF_WIDTH - np.abs(position)
    near = distance < grid.spacing
    scaled = pressure.astype(np.float64) / speed_of_sound
    sums = {"count": 0, "product": 0.0, "square": 0.0, "normal": 0.0, "scaled": 0.0}
    for axis in (0, 1):
        chosen = near[:, axis]
        outward = np.sign(position[chosen, axis]) * velocity[chosen, axis]
        sums["count"] += int(chosen.sum())
        sums["product"] += float((outward * scaled[chosen]).sum())
        sums["square"] += float((scaled[chosen] ** 2).sum())
        sums["normal"] += float(outward.sum())
        sums["scaled"] += float(scaled[chosen].sum())
    result["leak_sums"] = np.asarray([sums[key] for key in ("count", "product", "square", "normal", "scaled")])
    if keep_fields:
        result["field_vorticity"] = vorticity.astype(np.float16)
        if grid_spin is not None:
            result["field_spin_departure"] = (grid_spin - 0.5 * vorticity).astype(np.float16)
    return result


def sample_times(timestep: float, steps: int, cadence: int) -> list:
    times = set(np.round(np.arange(SAMPLE_INTERVAL, END_TIME + 1e-9, SAMPLE_INTERVAL), 6))
    times |= set(np.round(np.arange(SNAPSHOT_INTERVAL, END_TIME + 1e-9, SNAPSHOT_INTERVAL), 6))
    for low, high in WINDOWS:
        times |= set(np.round(np.arange(low, high + 1e-9, WINDOW_INTERVAL), 6))
    chosen = sorted({int(round(time / timestep / cadence)) * cadence for time in times})
    return [step for step in chosen if 0 < step <= steps]


# ---- run --------------------------------------------------------------------------------------------------------
def command_run(arguments) -> int:
    from experiment.validation import micropolar_tests as tests
    from experiment.validation.micropolar_couette import solver_revision
    from experiment.validation.micropolar_rotor3d import memory_status, read_own_particles
    solver = tests._import_solver(_REPO_ROOT)
    load_case, compute_chain_partition, SphSimulatorV7, VulkanContextV7, ChainOrchestratorV7 = solver
    from experiment.v7.utils.case_v7 import KIND_FLUID
    run = arguments.run
    resolution, length, speed_of_sound, shift, initial_pressure, slab_count = RUNS[run]
    case_dir = make_case(run)
    meta = json.loads((case_dir / "dipole.json").read_text(encoding="utf-8"))
    case = load_case(case_dir / "case.yaml")
    device_map = [int(value) for value in arguments.device_map.split(",")]
    if len(device_map) != slab_count:
        sys.exit(f"{run}: {slab_count} slab(s), --device-map names {len(device_map)}")
    weights = [float(value) for value in arguments.weights.split(",")] if arguments.weights else [1.0] * slab_count
    timestep = float(case.physics.timestep)
    if abs(timestep - meta["timestep"]) > 1e-15:
        sys.exit(f"{run}: the loader's dt {timestep} differs from the case's {meta['timestep']}")
    steps = meta["steps"] if arguments.end_time is None else int(round(arguments.end_time / timestep / 1000.0)) * 1000
    cadence = case.numerics.defrag_cadence
    spacing = meta["spacing"]
    shift_x = shift * spacing
    grid = Grid(resolution)
    volume = (2.0 * HALF_WIDTH) ** 2 / meta["counts"]["fluid"]
    micropolar = length is not None

    # the initial state at the particles: velocity, w = omega / 2 (every run; the control's w is passive)
    kinds = np.asarray([material.kind for material in case.materials])[case.initial.material_group.astype(np.int64)]
    fluid_rows = np.flatnonzero(kinds == KIND_FLUID)
    position = case.initial.positions[fluid_rows, :2].astype(np.float64)
    fields = analytic_fields(position[:, 0], position[:, 1], shift_x)
    velocities = case.initial.velocities.copy()
    velocities[fluid_rows, 0] = fields["u"].astype(np.float32)
    velocities[fluid_rows, 1] = fields["v"].astype(np.float32)
    case.initial.velocities = velocities
    spin = np.zeros((case.initial.positions.shape[0], 3), dtype=np.float32)
    spin[fluid_rows, 2] = (0.5 * fields["vorticity"]).astype(np.float32)
    case.initial.angular_velocities = spin
    fluid_group = next(index for index, material in enumerate(case.materials) if material.kind == KIND_FLUID)
    fluid_material = case.materials[fluid_group]
    rest = float(fluid_material.rest_density)
    power = float(case.physics.power_parameter)
    constant = float(fluid_material.eos_constant)
    if initial_pressure == "incompressible":
        axis, pressure_grid = incompressible_pressure(shift_x, spacing)
    else:
        axis, pressure_grid = None, None

    def initial_density_pressure(x: np.ndarray, y: np.ndarray) -> tuple:
        """(p / rho, rho, p) at the points: rho from p through the equation of state p = B ((rho / rho0)^gamma - 1),
        and p again from that rho in float32, as the solver evaluates it."""
        kinematic = bilinear(axis, pressure_grid, x, y) if pressure_grid is not None else np.zeros(x.shape[0])
        density = (rest * (1.0 + rest * kinematic / constant) ** (1.0 / power)).astype(np.float32)
        pressure = (np.float32(constant) * (np.power(density / np.float32(rest), np.float32(power))
                                            - np.float32(1.0))).astype(np.float32)
        return kinematic, density, pressure
    initial_kinematic_pressure = initial_density_pressure(position[:, 0], position[:, 1])[0]

    initial_diagnostics = diagnostics(grid, position, np.column_stack([fields["u"], fields["v"]]),
                                      initial_kinematic_pressure, 0.5 * fields["vorticity"] if micropolar else None,
                                      volume, speed_of_sound, keep_fields=False)
    print(f"[dipole] {run}: initial E {initial_diagnostics['energy']:.6f} (grid {initial_diagnostics['energy_grid']:.6f}), "
          f"Omega {initial_diagnostics['enstrophy']:.3f}, P {initial_diagnostics['palinstrophy']:.4e}; "
          f"p / rho from {initial_kinematic_pressure.min():.3f} to {initial_kinematic_pressure.max():.3f} m^2/s^2", flush=True)

    chain = compute_chain_partition(case, weights, pool_safety=arguments.pool_safety)
    out_dir = (_REPO_ROOT / RUNS_DIRECTORY / run).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fields").mkdir(exist_ok=True)
    revision = solver_revision()
    planned = sample_times(timestep, steps, cadence)
    snapshot_steps = {int(round(time / timestep / cadence)) * cadence
                      for time in np.round(np.arange(SNAPSHOT_INTERVAL, END_TIME + 1e-9, SNAPSHOT_INTERVAL), 6)}
    state_steps = {int(round(time / timestep / cadence)) * cadence for time in arguments.particle_states}
    samples: dict = {}
    memory = {"peak_working_set": 0, "peak_commit": 0, "least_free_physical": None, "least_free_commit": None}
    gigabyte = 1024.0 ** 3

    def note_memory() -> dict:
        status = memory_status()
        if status:
            memory["peak_working_set"] = max(memory["peak_working_set"], status["peak_working_set"])
            memory["peak_commit"] = max(memory["peak_commit"], status["peak_commit"])
            for key, source in (("least_free_physical", "free_physical"), ("least_free_commit", "free_commit")):
                memory[key] = status[source] if memory[key] is None else min(memory[key], status[source])
        return status

    def record(step: int, values: dict) -> None:
        samples.setdefault("step", []).append(step)
        samples.setdefault("time", []).append(step * timestep)
        for name, value in values.items():
            if name.startswith("field_"):
                continue
            samples.setdefault(name, []).append(value)

    contexts, sims = [], []
    started = time.perf_counter()
    failure = None
    statuses = []
    bootstrap_seconds = None
    try:
        for index in range(slab_count):
            contexts.append(VulkanContextV7.create(device_index=device_map[index], enable_validation=False,
                                                   application_name=f"dipole_k{slab_count}_s{index}"))
            sim = SphSimulatorV7(contexts[-1], chain.slabs[index], sync_scheme="per-direction")
            sims.append(sim)
            if initial_pressure == "incompressible":
                # the simulator's initial payload with (rho, p) of the incompressible pressure on the fluid rows
                slab_initial = chain.slabs[index].initial
                original = sim._build_initial_data

                def build(sim=sim, slab_initial=slab_initial, original=original) -> dict:
                    data = original()
                    capacities = sim.case.capacities
                    pool_capacity = capacities.total_pool_capacity()
                    own_first = sim.own_first_pid()
                    density_pressure = np.frombuffer(data["density_pressure"], np.float32).reshape(pool_capacity, 2).copy()
                    rows = np.flatnonzero(slab_initial.material_group.astype(np.int64) == fluid_group)
                    points = slab_initial.positions[rows, :2].astype(np.float64)
                    _kinematic, density, pressure = initial_density_pressure(points[:, 0], points[:, 1])
                    offset = sim.stored_density_offset()
                    density_pressure[own_first + rows, 0] = (density.astype(np.float64) - offset).astype(np.float32)
                    density_pressure[own_first + rows, 1] = pressure
                    data["density_pressure"] = density_pressure.tobytes()
                    print(f"[dipole] slab {index}: initial (rho, p) of the incompressible pressure on {rows.size:,} "
                          f"fluid rows", flush=True)
                    return data
                sim._build_initial_data = build

        def sample(step: int) -> None:
            status = note_memory()
            if status and min(status["free_physical"], status["free_commit"]) < arguments.memory_floor_gigabytes * gigabyte:
                raise RuntimeError(f"step {step}: free memory below the floor of {arguments.memory_floor_gigabytes:g} GB")
            wanted = ["density_pressure"] + (["angular_velocity"] if micropolar else [])
            parts = [read_own_particles(sim, wanted) for sim in sims]
            rows = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
            del parts
            fluid = rows["material"] == fluid_group
            count = int(fluid.sum())
            if count != meta["counts"]["fluid"]:
                raise RuntimeError(f"step {step}: {count} fluid particles alive, {meta['counts']['fluid']} loaded")
            fluid_position = rows["position_voxel_id"][fluid, :2].astype(np.float64)
            fluid_velocity = rows["velocity_mass"][fluid, :2].astype(np.float64)
            if not np.isfinite(fluid_velocity).all():
                raise RuntimeError(f"step {step}: non-finite velocity")
            density = rows["density_pressure"][fluid, 0].astype(np.float64)
            kinematic_pressure = rows["density_pressure"][fluid, 1].astype(np.float64) / rest
            fluid_spin = rows["angular_velocity"][fluid, 2].astype(np.float64) if micropolar else None
            keep = step in snapshot_steps
            values = diagnostics(grid, fluid_position, fluid_velocity, kinematic_pressure, fluid_spin, volume,
                                 speed_of_sound, keep)
            values.update({"density_mean": float(density.mean()), "density_min": float(density.min()),
                           "density_max": float(density.max()), "pressure_mean": float(kinematic_pressure.mean())})
            if keep:
                arrays = {"vorticity": values["field_vorticity"], "step": np.int64(step), "time": step * timestep}
                if "field_spin_departure" in values:
                    arrays["spin_departure"] = values["field_spin_departure"]
                np.savez_compressed(out_dir / "fields" / f"fields_{step:08d}.npz", **arrays)
            if step in state_steps:
                free = shutil.disk_usage(out_dir).free
                if free > arguments.disk_reserve_gigabytes * gigabyte + 0.2 * gigabyte:
                    arrays = {"position": rows["position_voxel_id"][fluid, :2], "velocity": rows["velocity_mass"][fluid, :2],
                              "density": rows["density_pressure"][fluid, 0], "pressure": rows["density_pressure"][fluid, 1],
                              "step": np.int64(step)}
                    if micropolar:
                        arrays["spin"] = rows["angular_velocity"][fluid, 2]
                    np.savez(out_dir / f"state_{step:08d}.npz", **arrays)
            record(step, values)

        def on_defrag(frame: int, report: list) -> None:
            for index, entry in enumerate(report):
                bad = {key: value for key, value in entry.items()
                       if (key.startswith("overflow_") or key == "far_migration") and value}
                if bad:
                    raise RuntimeError(f"invariant violation at step {frame}, sim {index}: {bad}")
            if frame in planned_set:
                sample(frame)
                (out_dir / "heartbeat.json").write_text(json.dumps(
                    {"step": frame, "steps": steps, "time": frame * timestep,
                     "elapsed_seconds": time.perf_counter() - started, "memory_now": note_memory(),
                     "memory": memory}), encoding="utf-8")
                if len(samples["step"]) % 10 == 0:
                    np.savez_compressed(out_dir / "samples_partial.npz",
                                        **{name: np.asarray(values) for name, values in samples.items()})

        planned_set = set(planned)
        with ChainOrchestratorV7(sims, defrag_cadence=cadence) as orchestrator:
            orchestrator.bootstrap_all()
            bootstrap_seconds = time.perf_counter() - started
            (out_dir / "start.json").write_text(json.dumps(
                {"run": run, "case": str(case_dir.relative_to(_REPO_ROOT)).replace("\\", "/"), "dipole": meta,
                 "revision": revision, "slabs": slab_count, "device_map": device_map, "weights": weights,
                 "steps": steps, "timestep": timestep, "sample_steps": planned,
                 "snapshot_steps": sorted(snapshot_steps), "bootstrap_seconds": bootstrap_seconds,
                 "cuts_x": tests._cut_positions(case, chain),
                 "slab_particles": [int(slab.initial.positions.shape[0]) for slab in chain.slabs],
                 "memory_at_start": note_memory()}, indent=1), encoding="utf-8")
            try:
                orchestrator.run_pipelined(steps, depth=arguments.depth, on_defrag=on_defrag)
                if not samples.get("step") or samples["step"][-1] != steps:
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
    initial = {name: (value.tolist() if isinstance(value, np.ndarray) else value)
               for name, value in initial_diagnostics.items() if not name.startswith(("band_", "field_"))}
    np.savez_compressed(out_dir / "samples.npz", **{name: np.asarray(values) for name, values in samples.items()},
                        initial_wall_vorticity=initial_diagnostics["wall_vorticity"])
    run_meta = {"run": run, "case": str(case_dir.relative_to(_REPO_ROOT)).replace("\\", "/"), "dipole": meta,
                "revision": revision, "slabs": slab_count, "device_map": device_map, "weights": weights,
                "steps": steps, "timestep": timestep, "end_time": steps * timestep, "particles":
                int(case.initial.positions.shape[0]), "fluid_particles": meta["counts"]["fluid"],
                "counters": counters, "elapsed_seconds": elapsed, "bootstrap_seconds": bootstrap_seconds,
                "failure": failure, "memory": memory, "initial": initial,
                "initial_pressure": initial_pressure, "initial_pressure_range": [
                    float(initial_kinematic_pressure.min()), float(initial_kinematic_pressure.max())],
                "steps_per_second": steps / max(elapsed - (bootstrap_seconds or 0.0), 1e-9),
                "cuts_x": tests._cut_positions(case, chain), "sample_count": len(samples.get("step", [])),
                "slab_particles": [int(slab.initial.positions.shape[0]) for slab in chain.slabs]}
    (out_dir / "run.json").write_text(json.dumps(run_meta, indent=1), encoding="utf-8")
    zero = all(value == 0 for value in counters.values())
    print(f"[dipole] {run}: {steps:,} steps (t = {steps * timestep:.4f}) in {elapsed:.0f} s "
          f"({run_meta['steps_per_second']:.1f} steps/s), counters {'all zero' if zero else counters}"
          + (f"; FAILED: {failure}" if failure else ""), flush=True)
    return 0 if zero and failure is None else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    cases = commands.add_parser("cases")
    cases.add_argument("runs", nargs="*")
    run = commands.add_parser("run")
    run.add_argument("run", choices=sorted(RUNS))
    run.add_argument("--device-map", default="0")
    run.add_argument("--weights", default=None)
    run.add_argument("--end-time", type=float, default=None, help="stop earlier (smoke)")
    run.add_argument("--depth", type=int, default=2)
    run.add_argument("--pool-safety", type=float, default=1.2)
    run.add_argument("--particle-states", type=float, nargs="*", default=[], help="times of full fluid states")
    run.add_argument("--memory-floor-gigabytes", type=float, default=2.0)
    run.add_argument("--disk-reserve-gigabytes", type=float, default=6.0)
    arguments = parser.parse_args()
    if arguments.command == "cases":
        return command_cases(arguments)
    if arguments.command == "run":
        return command_run(arguments)
    return 1


if __name__ == "__main__":
    sys.exit(main())
