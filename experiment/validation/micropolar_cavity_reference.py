"""
micropolar_cavity_reference.py - the lid-driven cavity of a micropolar fluid by finite differences (host, float64).

A second method for docs/micropolar_design.md Section 15 (Stage 6, 2.3), independent of the particle solver:
stream function psi, vorticity omega and microrotation w on a uniform grid of nodes, second-order central
differences, the wall vorticity by Jensen's formula, the Poisson equation by a sine transform, explicit steps to
the steady state. The model is the one of Section 1 (nu = (mu + kappa) / rho, c = kappa / rho):

  d omega / dt + v . grad omega = nu lap omega - c lap w            omega = curl v,  lap psi = -omega
  Theta (d w / dt + v . grad w) = xi lap w + c (omega - 2 w)        w = 0 on the four walls

With w = omega / 2 the first line is the Newtonian equation with nu_inf = nu - c / 2. A Newtonian fluid is c = 0.
Unit square, lid at y = 1 moving with U = 1 in +x, the parameters of micropolar_stage6_cavity.py.

  run       Cases on a sequence of grids (a finer grid starts from the coarser solution), extrema of the centre
            lines with their extrapolation to zero spacing, and the comparison with the particle runs.

Cases: newtonian_far (nu_inf, Reynolds number 400), newtonian_wall (nu, Reynolds number 250), micropolar (K = 3,
l = L / 20, Theta = l^2, w = 0 on the walls) and its variants, named micropolar[_l<N>][_no_inertia][_free_spin]:
_l<N> sets l = L / N; _no_inertia lets Theta -> 0 (w follows omega through xi lap w + c (omega - 2 w) = 0, no lag
and no transport of w); _free_spin lets the walls not hold w (its normal derivative is zero there instead);
_lid_spin lets only the lid hold w and _rest_spin only the three walls at rest.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
from scipy.fft import dstn, idstn
from scipy.interpolate import CubicSpline, RectBivariateSpline

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.validation import micropolar_stage6_cavity as cavity                     # noqa: E402
from experiment.validation.micropolar_couette import design_parameters                  # noqa: E402

LID_SPEED = 1.0
PARTICLE_RUNS = {"newtonian_far": "newtonian_re400_n500", "newtonian_wall": "newtonian_re250_n500",
                 "micropolar": "mp_n500", "micropolar_l40": "mp_n500_l40"}


def case_parameters(case: str) -> dict:
    """The coefficients of a case from its name (see the module text)."""
    if case in ("newtonian_far", "newtonian_wall"):
        viscosity = cavity.FAR_VISCOSITY if case == "newtonian_far" else float(cavity.design()["viscosity"])
        return {"viscosity": viscosity, "coupling": 0.0, "rotation": 0.0, "inertia": None, "length": None,
                "spin_equation": None, "spin_walls": None, "particle_run": PARTICLE_RUNS[case]}
    words = case.split("_")
    if words[0] != "micropolar":
        raise ValueError(f"unknown case {case!r}")
    length, spin_equation, spin_walls, rest = 1.0 / 20.0, "full", "held", words[1:]
    while rest:
        if rest[0].startswith("l") and rest[0][1:].isdigit():
            length, rest = 1.0 / int(rest[0][1:]), rest[1:]
        elif rest[:2] == ["no", "inertia"]:
            spin_equation, rest = "no_inertia", rest[2:]
        elif rest[:2] in (["free", "spin"], ["lid", "spin"], ["rest", "spin"]):
            spin_walls, rest = rest[0], rest[2:]
        else:
            raise ValueError(f"unknown case {case!r}")
    parameters = design_parameters(cavity.FAR_VISCOSITY, cavity.COUPLING_RATIO, length)
    return {"viscosity": parameters["viscosity"], "coupling": parameters["coupling"],
            "rotation": parameters["viscosity_rotation"], "inertia": parameters["micro_inertia"],
            "length": length, "spin_equation": spin_equation, "spin_walls": spin_walls,
            "particle_run": PARTICLE_RUNS.get(case)}


def refine(field: np.ndarray, resolution: int) -> np.ndarray:
    """A node field on a coarser grid, on the grid of the given resolution (bicubic)."""
    coarse = np.linspace(0.0, 1.0, field.shape[0])
    fine = np.linspace(0.0, 1.0, resolution + 1)
    return RectBivariateSpline(coarse, coarse, field, kx=3, ky=3)(fine, fine)


def centre_line_velocities(stream: np.ndarray) -> tuple:
    """u on the vertical centre line and v on the horizontal one, at the nodes (central differences)."""
    resolution = stream.shape[0] - 1
    spacing = 1.0 / resolution
    middle = resolution // 2
    horizontal_velocity = np.zeros(resolution + 1)
    horizontal_velocity[1:-1] = (stream[middle, 2:] - stream[middle, :-2]) / (2.0 * spacing)
    horizontal_velocity[-1] = LID_SPEED
    vertical_velocity = np.zeros(resolution + 1)
    vertical_velocity[1:-1] = -(stream[2:, middle] - stream[:-2, middle]) / (2.0 * spacing)
    return horizontal_velocity, vertical_velocity


def extremum(coordinate: np.ndarray, values: np.ndarray, kind: str) -> tuple:
    """The extremum of a centre line and its place, from the parabola through the three nodes around it."""
    interior = values[1:-1]
    index = 1 + int(np.argmin(interior) if kind == "min" else np.argmax(interior))
    left, centre, right = values[index - 1], values[index], values[index + 1]
    curvature = left - 2.0 * centre + right
    offset = 0.5 * (left - right) / curvature if curvature != 0.0 else 0.0
    spacing = coordinate[1] - coordinate[0]
    return float(centre - 0.25 * (left - right) * offset), float(coordinate[index] + offset * spacing)


def measures(stream: np.ndarray) -> dict:
    resolution = stream.shape[0] - 1
    spacing = 1.0 / resolution
    coordinate = np.linspace(0.0, 1.0, resolution + 1)
    horizontal_velocity, vertical_velocity = centre_line_velocities(stream)
    u_min, y_at_u_min = extremum(coordinate, horizontal_velocity, "min")
    v_max, x_at_v_max = extremum(coordinate, vertical_velocity, "max")
    v_min, x_at_v_min = extremum(coordinate, vertical_velocity, "min")
    # kinetic energy per unit density, (1/2) integral of |v|^2: interior nodes plus half the lid row
    velocity_x = (stream[1:-1, 2:] - stream[1:-1, :-2]) / (2.0 * spacing)
    velocity_y = -(stream[2:, 1:-1] - stream[:-2, 1:-1]) / (2.0 * spacing)
    kinetic_energy = 0.5 * float((velocity_x ** 2 + velocity_y ** 2).sum()) * spacing ** 2 \
        + 0.25 * LID_SPEED ** 2 * spacing
    return {"u_min": u_min, "y_at_u_min": y_at_u_min, "v_max": v_max, "x_at_v_max": x_at_v_max,
            "v_min": v_min, "x_at_v_min": x_at_v_min, "kinetic_energy": kinetic_energy,
            "stream_min": float(stream.min()),
            "coordinate": coordinate, "u": horizontal_velocity, "v": vertical_velocity}


def solve(resolution: int, parameters: dict, start: dict | None = None, end_time: float = 400.0,
          tolerance: float = 1e-7, window: float = 5.0, report=None) -> dict:
    """Explicit steps to the steady state: the three extrema of the centre lines change by less than the tolerance
    over a window of time, twice in a row."""
    spacing = 1.0 / resolution
    viscosity, coupling = parameters["viscosity"], parameters["coupling"]
    rotation, inertia = parameters["rotation"], parameters["inertia"]
    micropolar = coupling > 0.0
    no_inertia = micropolar and parameters["spin_equation"] == "no_inertia"
    free_spin = micropolar and parameters["spin_walls"] != "held"
    free_at_rest = micropolar and parameters["spin_walls"] in ("free", "lid")      # bottom and side walls
    free_at_lid = micropolar and parameters["spin_walls"] in ("free", "rest")
    index = np.arange(1, resolution)
    line = (2.0 * np.cos(np.pi * index / resolution) - 2.0) / spacing ** 2
    eigenvalue = line[:, None] + line[None, :]
    shape = (resolution + 1, resolution + 1)
    stream = np.zeros(shape)
    vorticity = np.zeros(shape) if start is None else refine(start["vorticity"], resolution)
    spin = np.zeros(shape) if start is None or not micropolar else refine(start["spin"], resolution)
    far_viscosity = viscosity - 0.5 * coupling
    step = min(0.2 * spacing ** 2 / viscosity, 0.8 * far_viscosity / LID_SPEED ** 2)
    if micropolar and not no_inertia:
        step = min(step, 0.2 * spacing ** 2 * inertia / rotation, 0.1 * inertia / (2.0 * coupling))
    if no_inertia and free_spin:
        raise ValueError("the case without spin inertia is written for walls that hold w only")

    def laplacian(field):
        return (field[2:, 1:-1] + field[:-2, 1:-1] + field[1:-1, 2:] + field[1:-1, :-2]
                - 4.0 * field[1:-1, 1:-1]) / spacing ** 2

    def update_stream_and_walls():
        stream[1:-1, 1:-1] = idstn(dstn(-vorticity[1:-1, 1:-1], type=1) / eigenvalue, type=1)
        scale = 1.0 / (2.0 * spacing ** 2)
        vorticity[1:-1, 0] = (stream[1:-1, 2] - 8.0 * stream[1:-1, 1]) * scale
        vorticity[1:-1, -1] = (stream[1:-1, -3] - 8.0 * stream[1:-1, -2]) * scale - 3.0 * LID_SPEED / spacing
        vorticity[0, 1:-1] = (stream[2, 1:-1] - 8.0 * stream[1, 1:-1]) * scale
        vorticity[-1, 1:-1] = (stream[-3, 1:-1] - 8.0 * stream[-2, 1:-1]) * scale
        # zero normal derivative of w at the walls that do not hold it, second order one-sided
        if free_at_rest:
            spin[1:-1, 0] = (4.0 * spin[1:-1, 1] - spin[1:-1, 2]) / 3.0
            spin[0, :-1] = (4.0 * spin[1, :-1] - spin[2, :-1]) / 3.0
            spin[-1, :-1] = (4.0 * spin[-2, :-1] - spin[-3, :-1]) / 3.0
        if free_at_lid:
            spin[1:-1, -1] = (4.0 * spin[1:-1, -2] - spin[1:-1, -3]) / 3.0

    time_now, steps, last, passes, change = 0.0, 0, None, 0, None
    steps_per_window = max(1, int(round(window / step)))
    started = time.time()
    steady = False
    while time_now < end_time:
        update_stream_and_walls()
        velocity_x = (stream[1:-1, 2:] - stream[1:-1, :-2]) / (2.0 * spacing)
        velocity_y = -(stream[2:, 1:-1] - stream[:-2, 1:-1]) / (2.0 * spacing)

        def advection(field):
            return (velocity_x * (field[2:, 1:-1] - field[:-2, 1:-1])
                    + velocity_y * (field[1:-1, 2:] - field[1:-1, :-2])) / (2.0 * spacing)

        rate = -advection(vorticity) + viscosity * laplacian(vorticity)
        if micropolar:
            rate -= coupling * laplacian(spin)
            if not no_inertia:
                spin_rate = -advection(spin) + (rotation * laplacian(spin) + coupling * (
                    vorticity[1:-1, 1:-1] - 2.0 * spin[1:-1, 1:-1])) / inertia
        vorticity[1:-1, 1:-1] += step * rate
        if micropolar:
            if no_inertia:
                spin[1:-1, 1:-1] = idstn(dstn(-coupling * vorticity[1:-1, 1:-1], type=1)
                                         / (rotation * eigenvalue - 2.0 * coupling), type=1)
            else:
                spin[1:-1, 1:-1] += step * spin_rate
        time_now += step
        steps += 1
        if not np.isfinite(vorticity[resolution // 2, resolution // 2]):
            raise FloatingPointError(f"the solution left the finite numbers at t = {time_now:.3f}")
        if steps % steps_per_window == 0:
            update_stream_and_walls()
            current = measures(stream)
            values = np.array([current["u_min"], current["v_max"], current["v_min"]])
            if last is not None:
                change = float(np.abs(values - last).max())
                passes = passes + 1 if change < tolerance else 0
                if report:
                    report(f"  n = {resolution}: t = {time_now:7.1f}  u_min {values[0]:+.6f}  v_max {values[1]:+.6f}  "
                           f"v_min {values[2]:+.6f}  change {change:.1e}  ({time.time() - started:.0f} s)")
                if passes >= 2:
                    steady = True
                    break
            last = values
    update_stream_and_walls()
    result = measures(stream)
    result.update({"resolution": resolution, "time": time_now, "steps": steps, "step": step, "steady": steady,
                   "last_change": change, "seconds": time.time() - started,
                   "stream": stream, "vorticity": vorticity, "spin": spin})
    if micropolar:
        # the microrotation against half the vorticity: where the two differ the fluid is stiffer than nu_inf
        interior = (slice(1, -1), slice(1, -1))
        result["spin_minus_half_vorticity_rms"] = float(np.sqrt(
            ((spin[interior] - 0.5 * vorticity[interior]) ** 2).mean()))
        result["half_vorticity_rms"] = float(np.sqrt(((0.5 * vorticity[interior]) ** 2).mean()))
    return result


def lid_measures(fields, parameters: dict) -> dict:
    """What the lid does to the fluid, from the fields of the finest grid: the traction on the fluid
    (-nu omega + c w at the lid, over rho U^2; mean over 0.1 < x < 0.9, away from the corner singularities), the
    depth below the lid at which u has fallen to U / 2 (at x = 0.25, 0.5, 0.75), and the flux of the main vortex
    (the least stream function)."""
    stream = np.asarray(fields["stream"], dtype=np.float64)
    vorticity = np.asarray(fields["vorticity"], dtype=np.float64)
    spin = np.asarray(fields["spin"], dtype=np.float64)
    resolution = stream.shape[0] - 1
    spacing = 1.0 / resolution
    middle = slice(int(round(0.1 * resolution)), int(round(0.9 * resolution)) + 1)
    traction = -parameters["viscosity"] * vorticity[middle, -1] + parameters["coupling"] * spin[middle, -1]
    depths = {}
    for position in (0.25, 0.5, 0.75):
        column = int(round(position * resolution))
        velocity = np.zeros(resolution + 1)
        velocity[1:-1] = (stream[column, 2:] - stream[column, :-2]) / (2.0 * spacing)
        velocity[-1] = LID_SPEED
        below = int(np.max(np.nonzero(velocity < 0.5 * LID_SPEED)[0]))
        fraction = (velocity[below + 1] - 0.5 * LID_SPEED) / (velocity[below + 1] - velocity[below])
        depths[f"{position:g}"] = float(1.0 - (below + 1 - fraction) * spacing)
    return {"lid_traction_mean": float(traction.mean()), "half_speed_depth": depths,
            "stream_min": float(stream.min()), "resolution": resolution}


def extrapolate(values: list, resolutions: list) -> dict:
    """Zero-spacing value from the two finest grids with the second order of the scheme, and the order the three
    finest grids show (None if their differences do not have one sign or the ratio is out of range)."""
    if len(values) < 2:
        return {"extrapolated": values[-1], "order": None}
    fine, finer = values[-2], values[-1]
    ratio = (resolutions[-1] / resolutions[-2]) ** 2
    result = {"extrapolated": finer + (finer - fine) / (ratio - 1.0), "order": None}
    if len(values) >= 3 and (values[-3] - fine) * (fine - finer) > 0.0:
        target = (values[-3] - fine) / (fine - finer)
        spacings = [1.0 / resolution for resolution in resolutions[-3:]]

        def mismatch(order):
            return ((spacings[0] ** order - spacings[1] ** order) / (spacings[1] ** order - spacings[2] ** order)
                    - target)
        orders = np.linspace(0.3, 6.0, 571)
        signs = np.sign([mismatch(order) for order in orders])
        crossing = np.nonzero(signs[1:] != signs[:-1])[0]
        if crossing.size:
            result["order"] = float(orders[crossing[0]] + 0.005)
    return result


def command_run(arguments) -> int:
    lines = []

    def say(text=""):
        lines.append(text)
        print(text, flush=True)

    target = _REPO_ROOT / arguments.out
    stored = json.loads(target.with_suffix(".json").read_text(encoding="utf-8")) \
        if target.with_suffix(".json").exists() and not arguments.fresh else {"cases": {}}
    particle = json.loads((_REPO_ROOT / "data/micropolar/stage6/cavity.json").read_text(encoding="utf-8"))["runs"] \
        if (_REPO_ROOT / "data/micropolar/stage6/cavity.json").exists() else {}
    keys = ("u_min", "y_at_u_min", "v_max", "x_at_v_max", "v_min", "x_at_v_min", "kinetic_energy")
    for case in arguments.cases:
        parameters = case_parameters(case)
        entry = stored["cases"].get(case)
        wanted = [int(value) for value in arguments.resolutions]
        if entry is None or entry["resolutions"] != wanted:
            say(f"{case}: nu {parameters['viscosity']:g}, c {parameters['coupling']:g}, xi {parameters['rotation']:g}, "
                f"Theta {parameters['inertia']}")
            grids, start = [], None
            for resolution in wanted:
                result = solve(resolution, parameters, start=start, end_time=arguments.end_time,
                               tolerance=arguments.tolerance, report=say if arguments.verbose else None)
                say(f"  n = {resolution}: steady {result['steady']} at t = {result['time']:.1f} "
                    f"(last change {result['last_change']:.1e}), {result['steps']:,} steps, {result['seconds']:.0f} s")
                start = result
                grids.append(result)
            entry = {"parameters": {key: value for key, value in parameters.items()}, "resolutions": wanted,
                     "grids": [{key: (value.tolist() if isinstance(value, np.ndarray) else value)
                                for key, value in grid.items() if key not in ("stream", "vorticity", "spin")}
                               for grid in grids]}
            entry["extrapolated"] = {key: extrapolate([grid[key] for grid in grids], wanted) for key in keys}
            stored["cases"][case] = entry
            np.savez_compressed(target.parent / f"{target.stem}_{case}.npz",
                                stream=grids[-1]["stream"].astype(np.float32),
                                vorticity=grids[-1]["vorticity"].astype(np.float32),
                                spin=grids[-1]["spin"].astype(np.float32))
            target.with_suffix(".json").write_text(json.dumps(stored, indent=1), encoding="utf-8")
    cases = stored["cases"]
    say()
    say("Lid-driven cavity by finite differences (stream function, vorticity, microrotation; second order; "
        "steady state). Extrema of the centre lines per grid and extrapolated to zero spacing with the second "
        "order of the scheme from the two finest grids; 'order' is what the three finest grids show.")
    say(f"{'case':<24}{'n':>6}{'u_min':>11}{'y':>8}{'v_max':>11}{'x':>8}{'v_min':>11}{'x':>8}{'E_k':>10}"
        f"{'steady at':>11}")
    for case, entry in cases.items():
        for grid in entry["grids"]:
            say(f"{case:<24}{grid['resolution']:>6}{grid['u_min']:>11.5f}{grid['y_at_u_min']:>8.4f}"
                f"{grid['v_max']:>11.5f}{grid['x_at_v_max']:>8.4f}{grid['v_min']:>11.5f}{grid['x_at_v_min']:>8.4f}"
                f"{grid['kinetic_energy']:>10.5f}"
                + (f"{grid['time']:>11.0f}" if grid["steady"] else f"{'(not)':>11}"))
        zero = entry["extrapolated"]
        say(f"{case:<24}{'-> 0':>6}{zero['u_min']['extrapolated']:>11.5f}{zero['y_at_u_min']['extrapolated']:>8.4f}"
            f"{zero['v_max']['extrapolated']:>11.5f}{zero['x_at_v_max']['extrapolated']:>8.4f}"
            f"{zero['v_min']['extrapolated']:>11.5f}{zero['x_at_v_min']['extrapolated']:>8.4f}"
            f"{zero['kinetic_energy']['extrapolated']:>10.5f}")
        orders = [zero[key]["order"] for key in ("u_min", "v_max", "v_min")]
        say(f"{'':<24}{'order':>6}" + "".join(
            (f"{order:>11.2f}{'':>8}" if order is not None else f"{'-':>11}{'':>8}") for order in orders))
    if "newtonian_far" in cases and "newtonian_wall" in cases:
        far, wall = cases["newtonian_far"]["extrapolated"], cases["newtonian_wall"]["extrapolated"]
        say()
        say("Against the two Newtonian bounds (finite differences, extrapolated): the place of each extremum on the "
            "way from the fluid of viscosity mu_inf (0) to the fluid of viscosity mu + kappa (1); negative = beyond "
            "the fluid of viscosity mu_inf; and the value over that of the fluid of viscosity mu_inf")
        say(f"{'case':<24}{'u_min':>9}{'v_max':>9}{'v_min':>9}{'E_k':>9}   {'u_min':>9}{'v_max':>9}{'v_min':>9}{'E_k':>9}")
        for case, entry in cases.items():
            if case in ("newtonian_far", "newtonian_wall"):
                continue
            zero = entry["extrapolated"]
            names = ("u_min", "v_max", "v_min", "kinetic_energy")
            fractions = [(zero[key]["extrapolated"] - far[key]["extrapolated"])
                         / (wall[key]["extrapolated"] - far[key]["extrapolated"]) for key in names]
            ratios = [zero[key]["extrapolated"] / far[key]["extrapolated"] for key in names]
            entry["against_bounds"] = {"fraction": dict(zip(names, fractions)),
                                       "over_far": dict(zip(names, ratios))}
            say(f"{case:<24}" + "".join(f"{value:>9.2f}" for value in fractions) + "   "
                + "".join(f"{value:>9.4f}" for value in ratios))
    say()
    say("What the lid does (finest grid of each case): traction of the lid on the fluid, mean over 0.1 < x < 0.9, "
        "in rho U^2 and over that on the fluid of viscosity mu_inf; depth below the lid at which u = U / 2, at "
        "x = 0.25, 0.5, 0.75 (L); flux of the main vortex (the least stream function, U L) and over that of the "
        "fluid of viscosity mu_inf")
    say(f"{'case':<24}{'n':>6}{'traction':>10}{'ratio':>8}{'x=0.25':>9}{'0.5':>8}{'0.75':>8}{'flux':>10}{'ratio':>8}")
    for case, entry in cases.items():
        fields_path = target.parent / f"{target.stem}_{case}.npz"
        if not fields_path.exists():
            continue
        with np.load(fields_path) as fields:
            entry["lid"] = lid_measures(fields, entry["parameters"])
    base = cases.get("newtonian_far", {}).get("lid")
    for case, entry in cases.items():
        lid = entry.get("lid")
        if not lid:
            continue
        say(f"{case:<24}{lid['resolution']:>6}{lid['lid_traction_mean']:>10.5f}"
            + (f"{lid['lid_traction_mean'] / base['lid_traction_mean']:>8.3f}" if base else f"{'-':>8}")
            + "".join(f"{lid['half_speed_depth'][key]:>{width}.4f}" for key, width in (("0.25", 9), ("0.5", 8),
                                                                                       ("0.75", 8)))
            + f"{lid['stream_min']:>10.5f}"
            + (f"{lid['stream_min'] / base['stream_min']:>8.3f}" if base else f"{'-':>8}"))
    compared = [(case, entry) for case, entry in cases.items() if entry["parameters"]["particle_run"] in particle]
    if compared:
        say()
        say("The particle runs (n = 500, time means) against the finite differences: extrema over the extrapolated "
            "ones, and the centre lines against those of the finest grid (rms and largest difference, U)")
        say(f"{'case':<24}{'particle run':<24}{'u_min':>9}{'v_max':>9}{'v_min':>9}{'E_k':>9}{'u rms':>10}{'u max':>10}"
            f"{'v rms':>10}{'v max':>10}")
        for case, entry in compared:
            run = particle[entry["parameters"]["particle_run"]]
            zero, finest = entry["extrapolated"], entry["grids"][-1]
            coordinate = np.asarray(run["coordinate"])
            difference_u = np.asarray(run["u_mean"]) - CubicSpline(finest["coordinate"], finest["u"])(coordinate)
            difference_v = np.asarray(run["v_mean"]) - CubicSpline(finest["coordinate"], finest["v"])(coordinate)
            ratios = [run["u_min"] / zero["u_min"]["extrapolated"], run["v_max"] / zero["v_max"]["extrapolated"],
                      run["v_min"] / zero["v_min"]["extrapolated"],
                      run["kinetic_energy_mean"] / zero["kinetic_energy"]["extrapolated"]]
            entry["particle"] = {"run": entry["parameters"]["particle_run"],
                                 "over_finite_differences": dict(zip(("u_min", "v_max", "v_min", "kinetic_energy"),
                                                                     ratios)),
                                 "u_difference_rms": float(np.sqrt((difference_u ** 2).mean())),
                                 "u_difference_max": float(np.abs(difference_u).max()),
                                 "v_difference_rms": float(np.sqrt((difference_v ** 2).mean())),
                                 "v_difference_max": float(np.abs(difference_v).max())}
            say(f"{case:<24}{entry['parameters']['particle_run']:<24}"
                + "".join(f"{value:>9.4f}" for value in ratios)
                + f"{entry['particle']['u_difference_rms']:>10.1e}{entry['particle']['u_difference_max']:>10.1e}"
                  f"{entry['particle']['v_difference_rms']:>10.1e}{entry['particle']['v_difference_max']:>10.1e}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    target.with_suffix(".json").write_text(json.dumps(stored, indent=1), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--cases", nargs="+", default=["newtonian_far", "newtonian_wall", "micropolar"])
    run.add_argument("--resolutions", nargs="+", default=["128", "192", "256"])
    run.add_argument("--end-time", type=float, default=400.0)
    run.add_argument("--tolerance", type=float, default=1e-7)
    run.add_argument("--out", default="data/micropolar/stage6/cavity_reference.txt")
    run.add_argument("--fresh", action="store_true", help="do not reuse the stored cases")
    run.add_argument("--verbose", action="store_true")
    run.set_defaults(handler=command_run)
    arguments = parser.parse_args()
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
