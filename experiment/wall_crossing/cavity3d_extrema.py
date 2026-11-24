"""
cavity3d_extrema.py - claude/wall_store_report.md section 15.2 (round 9): how much of the 3-D cavity's extremum deficit
is the post-processing (the brief's section 2).

Methods (each finds the in-plane extrema u_min, v_max, v_min of the centre lines from particles):
  (i)   "linear": the round-8 method (cavity3d_lines.py): linear MLS (Wendland C4, support h, weight W V_j, basis
        [1, (x - p) / h]) at points every dx / 8 along the line, velocity from the fluid and the wall particles, per
        sample; the samples' profiles averaged; the extremum = the discrete one refined by the parabola through it
        and its two neighbours.
  (ii)  "quadratic": the same with the full quadratic basis (10 terms), same support, same points and search.
  (iii) "fit": direct least squares on the particles within radius 1.5 dx of the line (every sample pooled), in a
        window |s - c| <= w around the extremum: a quartic in the along-line coordinate s plus a, b, a^2, b^2, a b in the
        two transverse coordinates; the extremum of the quartic (a = b = 0) inside the window, the window re-centred
        on it once. Windows w (normalised): u_min 0.055, v_max 0.06, v_min 0.03 x a factor chosen on the analytic test.
Coordinates are normalised on the crossing box: s = (x - wall) / L.

Analytic test: a field with the paper's peak shapes,
    f(s, a, b) = g(s) + k_a a^2 / 2 + k_b b^2 / 2     (a, b: transverse offsets from the line, normalised),
g a C2 quintic Hermite curve through the table 5 / 6 points and the table 10 extremum, with g' = 0 and g'' = the
brief's values at the extremum (u_min 28, v_max -20, v_min 150; the cubic spline through the same points gives the
second derivative printed beside them), k_a, k_b the transverse curvatures measured from the simulation's tubes.
It is sampled at particle positions (the real ones of a run's tubes, or the lattice), with or without noise of the
particles' measured scatter, and every method's extremum is compared with g's (= table 10's) value.
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
from scipy.interpolate import BPoly, CubicSpline
from scipy.spatial import cKDTree

from experiment.wall_crossing.cavity3d_benchmark import SEARCH, parabola_extremum, reference

# quantity: (line, along axis, velocity component, kind, transverse axes)
QUANTITIES = {"u_min": ("vertical", 1, 0, "min", (0, 2)), "v_max": ("horizontal", 0, 1, "max", (1, 2)),
              "v_min": ("horizontal", 0, 1, "min", (1, 2))}
# the spanwise |w| peaks (not in the analytic test): w > 0 on the side z < 0.5, w < 0 on the other (table 10 |w|max)
SPANWISE = {"w_first": ("spanwise", 2, 2, "max", (0, 1), (0.02, 0.5)), "w_second": ("spanwise", 2, 2, "min", (0, 1),
                                                                                    (0.5, 0.98))}
REFERENCE_SECOND_DERIVATIVES = {"u_min": 28.0, "v_max": -20.0, "v_min": 150.0}
FIT_WINDOWS = {"u_min": 0.055, "v_max": 0.06, "v_min": 0.03, "w_first": 0.06, "w_second": 0.06}
FIT_RADIUS_SPACINGS = 1.5
FIT_DEGREE = 4


def specification(quantity: str) -> tuple:
    return QUANTITIES[quantity] if quantity in QUANTITIES else SPANWISE[quantity]


def search_range(quantity: str) -> tuple:
    return SEARCH[quantity][3] if quantity in QUANTITIES else SPANWISE[quantity][5]


def wendland_c4(normalized: np.ndarray) -> np.ndarray:
    one_minus = np.clip(1.0 - normalized, 0.0, None)
    return one_minus ** 6 * ((35.0 / 3.0) * normalized ** 2 + 6.0 * normalized + 1.0)


def kernel_second_moment() -> float:
    """sigma^2 / h^2 of the 3-D Wendland C4: the per-direction second moment of W over its support (mean r^2 / 3)."""
    radius = np.linspace(0.0, 1.0, 200001)
    weight = wendland_c4(radius)
    return float(np.trapezoid(radius ** 4 * weight, radius) / np.trapezoid(radius ** 2 * weight, radius) / 3.0)


def analytic_curves(second_derivatives: dict = REFERENCE_SECOND_DERIVATIVES) -> tuple[dict, dict]:
    """({"vertical": g_u(s), "horizontal": g_v(s)} as scipy BPoly, {quantity: the cubic spline's g'' there})."""
    data = reference()
    curves, spline_second = {}, {}
    for line, keys in (("vertical", ["u_min"]), ("horizontal", ["v_max", "v_min"])):
        points = {float(s): float(value) for s, value in zip(data[line]["coordinate"], data[line]["velocity"])}
        extremum_positions = {}
        for key in keys:
            position = float(data["extrema"][key]["position"])
            # a table point closer than 0.004 to the extremum would fight it; the table's points are 0.008 apart there
            for existing in [s for s in points if abs(s - position) < 0.004]:
                del points[existing]
            points[position] = float(data["extrema"][key]["value"])
            extremum_positions[key] = position
        coordinate = np.array(sorted(points))
        values = np.array([points[s] for s in coordinate])
        spline = CubicSpline(coordinate, values, bc_type="not-a-knot")
        first, second = spline(coordinate, 1), spline(coordinate, 2)
        for key, position in extremum_positions.items():
            index = int(np.argmin(np.abs(coordinate - position)))
            spline_second[key] = float(second[index])
            first[index] = 0.0
            second[index] = float(second_derivatives[key])
        curves[line] = BPoly.from_derivatives(coordinate, np.column_stack([values, first, second]))
    return curves, spline_second


def normalised(positions: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return (positions - lower) / (upper - lower)


def analytic_values(positions: np.ndarray, lower, upper, quantity: str, curves: dict, curvatures: dict) -> np.ndarray:
    """f(s, a, b) of quantity's line at positions (physical)."""
    line, along, _component, _kind, transverse = QUANTITIES[quantity]
    unit = normalised(positions, lower, upper)
    s = unit[:, along]
    first, second = unit[:, transverse[0]] - 0.5, unit[:, transverse[1]] - 0.5
    k_first, k_second = curvatures[quantity]
    return curves[line](s) + 0.5 * k_first * first ** 2 + 0.5 * k_second * second ** 2


# ---------------------------------------------------------------------------------------------------------------------
# moving least squares of order 1 or 2 at points
# ---------------------------------------------------------------------------------------------------------------------

def moving_least_squares(positions: np.ndarray, volumes: np.ndarray, values: np.ndarray, points: np.ndarray,
                         support: float, order: int) -> np.ndarray:
    """MLS of values (M,) at points (P, 3), basis of total order 1 (4 terms) or 2 (10 terms); NaN where singular."""
    tree = cKDTree(positions)
    lists = tree.query_ball_point(points, r=support, workers=-1, return_sorted=False)
    lengths = np.fromiter((len(item) for item in lists), dtype=np.int64, count=len(lists))
    particle = np.concatenate([np.asarray(item, dtype=np.int64) for item in lists if len(item)])
    point = np.repeat(np.arange(points.shape[0]), lengths)
    offset = (positions[particle] - points[point]) / support
    weight = wendland_c4(np.sqrt((offset ** 2).sum(axis=1))) * volumes[particle]
    columns = [np.ones(offset.shape[0]), offset[:, 0], offset[:, 1], offset[:, 2]]
    if order == 2:
        columns += [offset[:, 0] ** 2, offset[:, 1] ** 2, offset[:, 2] ** 2, offset[:, 0] * offset[:, 1],
                    offset[:, 1] * offset[:, 2], offset[:, 2] * offset[:, 0]]
    basis = np.column_stack(columns)
    terms = basis.shape[1]
    moment = np.zeros((points.shape[0], terms, terms))
    right = np.zeros((points.shape[0], terms))
    for row in range(terms):
        right[:, row] = np.bincount(point, weights=weight * basis[:, row] * values[particle], minlength=points.shape[0])
        for column in range(row, terms):
            entry = np.bincount(point, weights=weight * basis[:, row] * basis[:, column], minlength=points.shape[0])
            moment[:, row, column] = entry
            moment[:, column, row] = entry
    result = np.full(points.shape[0], np.nan)
    good = lengths >= terms + 2
    if good.any():
        with np.errstate(all="ignore"):
            condition = np.linalg.cond(moment[good])
        usable = np.flatnonzero(good)[np.isfinite(condition) & (condition < 1e10)]
        result[usable] = np.linalg.solve(moment[usable], right[usable][:, :, None])[:, 0, 0]
    return result


def line_points(lower, upper, along: int, coordinates: np.ndarray) -> np.ndarray:
    centre = 0.5 * (lower + upper)
    points = np.tile(centre, (coordinates.size, 1))
    points[:, along] = lower[along] + coordinates * (upper[along] - lower[along])
    return points


def method_mls(samples: list, lower, upper, spacings: int, support: float, quantity: str, order: int,
               walls: tuple | None = None) -> dict:
    """Methods (i) (order 1) and (ii) (order 2): every sample's profile on the fine points of quantity's search range,
    averaged, then the parabola-refined extremum. samples: [{"positions", "volumes", "values"}] (values = the
    velocity component of the line); walls: (positions, volumes, values) or None."""
    line, along, _component, kind, _transverse = specification(quantity)[:5]
    search = search_range(quantity)
    intervals = 8 * (spacings + 1)
    fine = np.arange(intervals + 1) / intervals
    fine = fine[(fine >= search[0] - 1e-12) & (fine <= search[1] + 1e-12)]
    points = line_points(lower, upper, along, fine)
    profiles = []
    for sample in samples:
        positions, volumes, values = sample["positions"], sample["volumes"], sample["values"]
        if walls is not None:
            positions = np.concatenate([positions, walls[0]])
            volumes = np.concatenate([volumes, walls[1]])
            values = np.concatenate([values, walls[2]])
        profiles.append(moving_least_squares(positions, volumes, values, points, support, order))
    mean = np.mean(profiles, axis=0)
    position, value = parabola_extremum(fine, mean, kind, search)
    return {"value": value, "position": position}


def method_fit(samples: list, lower, upper, spacing: float, quantity: str, window: float, centre: float,
               degree: int = FIT_DEGREE, radius_spacings: float = FIT_RADIUS_SPACINGS) -> dict:
    """Method (iii): least squares on every sample's particles within radius radius_spacings x spacing of the line,
    |s - c| <= window; a polynomial of degree in s plus a, b, a^2, b^2, a b; the extremum of the polynomial (a = b = 0)
    in the window; the window re-centred on it once."""
    line, along, _component, kind, transverse = specification(quantity)[:5]
    length = float(upper[along] - lower[along])
    radius = radius_spacings * spacing / length
    pooled = []
    for sample in samples:
        unit = normalised(sample["positions"], lower, upper)
        first, second = unit[:, transverse[0]] - 0.5, unit[:, transverse[1]] - 0.5
        keep = first ** 2 + second ** 2 <= radius ** 2
        pooled.append(np.column_stack([unit[keep, along], first[keep], second[keep], sample["values"][keep]]))
    pooled = np.concatenate(pooled)
    result = {}
    for _iteration in range(2):
        inside = np.abs(pooled[:, 0] - centre) <= window
        s = (pooled[inside, 0] - centre) / window
        first, second, values = pooled[inside, 1] / radius, pooled[inside, 2] / radius, pooled[inside, 3]
        design = np.column_stack([s ** power for power in range(degree + 1)]
                                 + [first, second, first ** 2, second ** 2, first * second])
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        polynomial = np.polynomial.Polynomial(coefficients[:degree + 1])
        grid = np.linspace(-1.0, 1.0, 4001)
        curve = polynomial(grid)
        index = int(np.argmin(curve) if kind == "min" else np.argmax(curve))
        residual = values - design @ coefficients
        result = {"value": float(curve[index]), "position": float(centre + grid[index] * window),
                  "points": int(inside.sum()), "residual": float(residual.std()),
                  "at_edge": bool(index in (0, grid.size - 1))}
        centre = result["position"]
    return result


# ---------------------------------------------------------------------------------------------------------------------
# particle sets
# ---------------------------------------------------------------------------------------------------------------------

def tube_samples(path: pathlib.Path) -> tuple[list, dict]:
    """Samples of a tube file: round-8 frame tubes (one npz, a frame index per particle) or a round-9 run's tubes/
    directory (one npz per record). Returns ([{"positions", "velocities", "volumes", "time"}], info)."""
    path = pathlib.Path(path)
    samples = []
    if path.is_dir():
        for file in sorted(path.glob("t*.npz")):
            with np.load(file) as archive:
                samples.append({"positions": archive["positions"].astype(np.float64),
                                "velocities": archive["velocities"].astype(np.float64),
                                "pressures": archive["pressures"].astype(np.float64),
                                "volumes": archive["masses"].astype(np.float64) / archive["densities"].astype(np.float64),
                                "time": float(archive["time"]), "level_time": float(archive["level_time"])})
        return samples, {"half_width": float(np.load(sorted(path.glob("t*.npz"))[0])["half_width"])}
    with np.load(path) as archive:
        frame = archive["frame"]
        for index, time_value in enumerate(archive["times"]):
            rows = frame == index
            samples.append({"positions": archive["positions"][rows].astype(np.float64),
                            "velocities": archive["velocities"][rows].astype(np.float64),
                            "pressures": archive["pressures"][rows].astype(np.float64),
                            "volumes": archive["masses"][rows].astype(np.float64)
                            / archive["densities"][rows].astype(np.float64),
                            "time": float(time_value)})
        info = {"half_width": float(archive["half_width"]), "support": float(archive["support"])}
    return samples, info


def lattice_tube(spacings: int, half_width: float) -> np.ndarray:
    """The fluid lattice (spacing 1 / N, -0.5 .. 0.5) inside the three square tubes of the given half-width."""
    axis = np.arange(spacings + 1) / spacings - 0.5
    near = axis[np.abs(axis) <= half_width + 1e-12]
    blocks = []
    for along in range(3):
        grids = [near, near, near]
        grids[along] = axis
        mesh = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, 3)
        blocks.append(mesh)
    return np.unique(np.concatenate(blocks), axis=0)


def transverse_curvatures(samples: list, lower, upper, spacing: float, half_width: float) -> dict:
    """k_a, k_b of each quantity from the samples: least squares of the velocity component around the extremum
    (|s - s_ext| <= 0.02, the whole tube) with a quadratic in s and a, b, a^2, b^2, a b."""
    data = reference()["extrema"]
    out = {}
    for quantity, (line, along, component, _kind, transverse) in QUANTITIES.items():
        rows = []
        position = float(data[quantity]["position"])
        for sample in samples:
            unit = normalised(sample["positions"], lower, upper)
            first, second = unit[:, transverse[0]] - 0.5, unit[:, transverse[1]] - 0.5
            keep = (np.abs(unit[:, along] - position) <= 0.02) & (np.abs(first) <= half_width) & \
                (np.abs(second) <= half_width)
            rows.append(np.column_stack([unit[keep, along] - position, first[keep], second[keep],
                                         sample["velocities"][keep, component]]))
        rows = np.concatenate(rows)
        s, first, second, values = rows.T
        design = np.column_stack([np.ones_like(s), s, s ** 2, first, second, first ** 2, second ** 2, first * second])
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        out[quantity] = (2.0 * coefficients[5], 2.0 * coefficients[6])
    return out


def particle_scatter(samples: list, lower, upper, support: float, spacing: float) -> dict:
    """Per quantity: the standard deviation of the particles' velocity component about the quadratic-MLS field at
    their own positions (particles within 1.5 dx of the line, 0.2 <= s <= 0.8), over the samples."""
    out = {}
    for quantity, (line, along, component, _kind, transverse) in QUANTITIES.items():
        if quantity == "v_min":
            continue                                    # same line as v_max
        residuals = []
        for sample in samples[:: max(1, len(samples) // 5)]:
            unit = normalised(sample["positions"], lower, upper)
            first, second = unit[:, transverse[0]] - 0.5, unit[:, transverse[1]] - 0.5
            near = (first ** 2 + second ** 2 <= (1.5 * spacing) ** 2) & (unit[:, along] >= 0.2) & \
                (unit[:, along] <= 0.8)
            field = moving_least_squares(sample["positions"], sample["volumes"], sample["velocities"][:, component],
                                         sample["positions"][near], support, 2)
            residuals.append(sample["velocities"][near, component] - field)
        residual = np.concatenate(residuals)
        out[line] = float(np.nanstd(residual))
    return out


def run_methods(samples: list, lower, upper, spacings: int, support: float, values_of, walls_of=None,
                window_factor: float = 1.0, quantities=None) -> dict:
    """The three methods for the quantities (default the three in-plane extrema). values_of(quantity, sample) -> the
    sample's particle values; walls_of(quantity) -> (positions, volumes, values) or None."""
    spacing = float(upper[0] - lower[0]) / (spacings + 1)
    out = {}
    for quantity in (quantities or list(QUANTITIES)):
        prepared = [{"positions": sample["positions"], "volumes": sample["volumes"],
                     "values": values_of(quantity, sample)} for sample in samples]
        walls = walls_of(quantity) if walls_of is not None else None
        linear = method_mls(prepared, lower, upper, spacings, support, quantity, 1, walls)
        quadratic = method_mls(prepared, lower, upper, spacings, support, quantity, 2, walls)
        fit = method_fit(prepared, lower, upper, spacing, quantity, window_factor * FIT_WINDOWS[quantity],
                         quadratic["position"])
        out[quantity] = {"linear": linear, "quadratic": quadratic, "fit": fit}
    return out


def relative_deviation(value: float, reference_value: float) -> float:
    return (abs(value) - abs(reference_value)) / abs(reference_value)


def box_of(spacings: int) -> tuple[np.ndarray, np.ndarray]:
    """The crossing box of the N lattice: the fluid lattice -0.5 .. 0.5 widened by half a spacing."""
    half = 0.5 + 0.5 / spacings
    return np.full(3, -half), np.full(3, half)


def theory_bias(spacings: int, support_ratio: float, curvatures: dict) -> dict:
    """1/2 sigma^2 (g'' + k_a + k_b) relative to |extremum| (the kernel-smoothing estimate of the brief), along-line
    term alone and with the transverse terms, sigma^2 = kernel_second_moment() h^2 (normalised h = ratio / N / L)."""
    data = reference()["extrema"]
    moment = kernel_second_moment()
    length = 1.0 + 1.0 / spacings
    support = support_ratio / spacings / length
    out = {}
    for quantity in QUANTITIES:
        along = REFERENCE_SECOND_DERIVATIVES[quantity]
        k_first, k_second = curvatures[quantity]
        value = float(data[quantity]["value"])
        # the smoothed value is g + bias; its relative deviation in magnitude is sign(g) bias / |g| (weaker: < 0)
        sign = float(np.sign(value))
        out[quantity] = {"along": sign * 0.5 * moment * support ** 2 * along / abs(value),
                         "total": sign * 0.5 * moment * support ** 2 * (along + k_first + k_second) / abs(value)}
    return out


def boundary_particles(case_yaml: pathlib.Path, volume: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The case's wall and lid particles (every boundary OBJ of case.yaml; the fluid OBJ is not read): positions,
    volumes (= volume, the lattice's), velocities (the material's initial velocity)."""
    import yaml
    from experiment.v7.utils.case_loader_v7 import _parse_obj_vertices
    case_yaml = pathlib.Path(case_yaml).resolve()
    data = yaml.safe_load(case_yaml.read_text(encoding="utf-8"))
    library = yaml.safe_load((case_yaml.parent / data["material_library"]).resolve().read_text(encoding="utf-8"))
    positions, velocities = [], []
    for entry in data["geometry"]["particles"]:
        specification_entry = library[entry["material"]]
        if specification_entry["kind"] != "boundary":
            continue
        vertices = np.asarray(_parse_obj_vertices((case_yaml.parent / entry["file"]).resolve()), dtype=np.float64)
        positions.append(vertices)
        velocities.append(np.broadcast_to(np.asarray(specification_entry.get("initial_velocity", [0.0, 0.0, 0.0]),
                                                     dtype=np.float64), vertices.shape).copy())
    positions = np.concatenate(positions)
    return positions, np.full(positions.shape[0], float(volume)), np.concatenate(velocities)
