"""cavity_sampling.py - CPU-side sampling of SPH particle data for the lid-driven cavity validation.

Pure numpy / scipy (no GPU). Used online by cavity_runner.py (every sample) and offline by
cavity_analysis.py (time averages, extrema, stream function).

Coordinates. The solver's cavity has its fluid particles on a square lattice filling [-0.5, 0.5]^2
(spacing dx), the first wall / lid particle row at +-(0.5 + dx) and the lid moving in +x at the top.
The benchmark is the unit square [0, 1]^2 with the lid at y = 1. A Frame maps between the two:
  'wall'  - the unit square spans the centre lines of the innermost wall / lid rows,
            [-(0.5 + dx), 0.5 + dx], so L = 1 + 2 dx (the case's actual size as defined on 2026-09-30:
            the boundary starts at the innermost fixed boundary particles);
  'fluid' - the unit square spans the fluid lattice [-0.5, 0.5], L = 1.
Both maps are symmetric about the cavity centre, so the two centre lines (x = 0.5, y = 0.5) are the
same physical lines in both frames; only positions along them scale. Velocities are not rescaled
(U = 1 in both).

Interpolation. Kernel = the solver's Wendland C4 profile (helpers.glsl evaluate_kernel) with support
radius h (the case's 'h'), weight w_j = W(|p - x_j| / h) V_j with V_j = m_j / rho_j. Shepard:
f(p) = sum w f / sum w. MLS: linear basis [1, dx/h, dy/h], value = a0 of the weighted least-squares
fit (exact for linear fields); points whose moment matrix is ill-conditioned fall back to Shepard.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.spatial import cKDTree


def wendland_c4(q: np.ndarray) -> np.ndarray:
    """Unnormalised Wendland C4 profile (1 - q)^6 (35/3 q^2 + 6 q + 1) for q < 1, else 0 (the solver's
    kernel up to its constant, which cancels in Shepard and MLS)."""
    q = np.asarray(q, dtype=np.float64)
    one_minus_q = np.clip(1.0 - q, 0.0, None)
    return one_minus_q ** 6 * ((35.0 / 3.0) * q * q + 6.0 * q + 1.0)


@dataclasses.dataclass(frozen=True)
class Frame:
    """Affine map between solver coordinates and the benchmark's unit square (see module doc)."""
    name: str
    half_width: float           # physical half width of the unit square: 0.5 + dx ('wall') or 0.5 ('fluid')

    @staticmethod
    def make(name: str, spacing: float) -> "Frame":
        if name == "wall":
            return Frame(name, 0.5 + spacing)
        if name == "fluid":
            return Frame(name, 0.5)
        raise ValueError(f"unknown frame {name!r}")

    @property
    def length(self) -> float:
        return 2.0 * self.half_width

    def to_physical(self, reference: np.ndarray) -> np.ndarray:
        return np.asarray(reference, dtype=np.float64) * self.length - self.half_width

    def to_reference(self, physical: np.ndarray) -> np.ndarray:
        return (np.asarray(physical, dtype=np.float64) + self.half_width) / self.length


@dataclasses.dataclass
class ParticleSet:
    """The particles a sample is taken from (float64 copies)."""
    positions: np.ndarray       # (N, 2)
    velocities: np.ndarray      # (N, 2)
    volumes: np.ndarray         # (N,)   m / rho


def _pairs(tree: cKDTree, points: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Flattened (point index, particle index) pairs with |p - x_j| < radius."""
    neighbour_lists = tree.query_ball_point(points, r=radius, workers=-1, return_sorted=False)
    lengths = np.fromiter((len(item) for item in neighbour_lists), dtype=np.int64, count=len(neighbour_lists))
    if lengths.sum() == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    particle_index = np.concatenate([np.asarray(item, dtype=np.int64) for item in neighbour_lists if len(item)])
    point_index = np.repeat(np.arange(len(neighbour_lists), dtype=np.int64), lengths)
    return point_index, particle_index


def interpolate(particles: ParticleSet, points: np.ndarray, support: float,
                tree: cKDTree | None = None, condition_limit: float = 1.0e8) -> dict:
    """Shepard and linear-MLS velocity at ``points`` (M, 2) from ``particles``.

    Returns {'shepard': (M, 2), 'mls': (M, 2), 'neighbours': (M,), 'mls_fallback': (M,) bool}; points
    with no neighbour get NaN."""
    points = np.asarray(points, dtype=np.float64)
    if tree is None:
        tree = cKDTree(particles.positions)
    point_count = points.shape[0]
    point_index, particle_index = _pairs(tree, points, support)
    offset = (particles.positions[particle_index] - points[point_index]) / support
    distance = np.sqrt((offset ** 2).sum(axis=1))
    weight = wendland_c4(distance) * particles.volumes[particle_index]
    values = particles.velocities[particle_index]
    weight_sum = np.bincount(point_index, weights=weight, minlength=point_count)
    neighbours = np.bincount(point_index, minlength=point_count)
    with np.errstate(invalid="ignore", divide="ignore"):
        shepard = np.stack([np.bincount(point_index, weights=weight * values[:, component],
                                        minlength=point_count) / weight_sum for component in range(2)], axis=1)
    # MLS, linear basis b = [1, ox, oy]: M = sum w b b^T, r = sum w b f, value = a0.
    basis = np.stack([np.ones_like(distance), offset[:, 0], offset[:, 1]], axis=1)
    moment = np.zeros((point_count, 3, 3))
    for row in range(3):
        for column in range(row, 3):
            entry = np.bincount(point_index, weights=weight * basis[:, row] * basis[:, column], minlength=point_count)
            moment[:, row, column] = entry
            moment[:, column, row] = entry
    right = np.zeros((point_count, 3, 2))
    for row in range(3):
        for component in range(2):
            right[:, row, component] = np.bincount(point_index, weights=weight * basis[:, row] * values[:, component],
                                                   minlength=point_count)
    mls = np.full((point_count, 2), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        condition = np.linalg.cond(moment)
    good = np.isfinite(condition) & (condition < condition_limit) & (neighbours >= 6)
    if good.any():
        solution = np.linalg.solve(moment[good], right[good])
        mls[good] = solution[:, 0, :]
    fallback = ~good & (neighbours > 0)
    mls[fallback] = shepard[fallback]
    return {"shepard": shepard, "mls": mls, "neighbours": neighbours, "mls_fallback": fallback}


def band(particles: ParticleSet, axis: int, coordinate: float, half_width: float) -> ParticleSet:
    """The particles with |x_axis - coordinate| <= half_width (a strip around a centre line)."""
    mask = np.abs(particles.positions[:, axis] - coordinate) <= half_width
    return ParticleSet(particles.positions[mask], particles.velocities[mask], particles.volumes[mask])


def centerline_points(frame: Frame, along: np.ndarray, line: str) -> np.ndarray:
    """Physical points on the vertical centre line ('u': x_ref = 0.5, along = y_ref) or the horizontal
    one ('v': y_ref = 0.5, along = x_ref)."""
    along_physical = frame.to_physical(np.asarray(along, dtype=np.float64))
    centre = frame.to_physical(np.array([0.5]))[0]
    if line == "u":
        return np.stack([np.full_like(along_physical, centre), along_physical], axis=1)
    if line == "v":
        return np.stack([along_physical, np.full_like(along_physical, centre)], axis=1)
    raise ValueError(line)


def sample_centerlines(particles: ParticleSet, support: float, point_sets: dict) -> dict:
    """Velocity on the two centre lines. ``point_sets`` maps a name to (line, (M, 2) physical points);
    each line's points are interpolated from the strip |coordinate| <= 1.05 support around it. Returns
    {name: {'shepard': (M,), 'mls': (M,), 'neighbours': (M,), 'mls_fallback': (M,)}} with the line's
    velocity component (u on the vertical line, v on the horizontal one)."""
    strips = {}
    results = {}
    for name, (line, points) in point_sets.items():
        axis = 0 if line == "u" else 1
        if line not in strips:
            centre = float(points[0, axis])
            strip = band(particles, axis, centre, 1.05 * support)
            strips[line] = (strip, cKDTree(strip.positions))
        strip, tree = strips[line]
        sampled = interpolate(strip, points, support, tree=tree)
        component = 0 if line == "u" else 1
        results[name] = {"shepard": sampled["shepard"][:, component], "mls": sampled["mls"][:, component],
                         "neighbours": sampled["neighbours"], "mls_fallback": sampled["mls_fallback"]}
    return results


def refine_extremum(coordinates: np.ndarray, values: np.ndarray, kind: str, half_window: int = 3,
                    search: tuple[float, float] | None = None) -> tuple[float, float]:
    """(position, value) of the minimum ('min') or maximum ('max') of a sampled profile: the discrete
    extremum inside ``search`` refined by a least-squares parabola through the 2 half_window + 1 points
    around it (the parabola's vertex; the discrete point if the fit is not concave / convex)."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(values)
    if search is not None:
        mask &= (coordinates >= search[0]) & (coordinates <= search[1])
    candidates = np.flatnonzero(mask)
    index = candidates[np.argmin(values[candidates]) if kind == "min" else np.argmax(values[candidates])]
    first, last = max(index - half_window, 0), min(index + half_window + 1, coordinates.size)
    x, y = coordinates[first:last], values[first:last]
    centre = coordinates[index]
    a, b, c = np.polyfit(x - centre, y, 2)
    if (kind == "min" and a <= 0) or (kind == "max" and a >= 0):
        return float(centre), float(values[index])
    vertex = -b / (2 * a)
    if abs(vertex) > (x[-1] - x[0]) / 2:
        return float(centre), float(values[index])
    return float(centre + vertex), float(c - b * b / (4 * a))


def stream_function(u_grid: np.ndarray, y_reference: np.ndarray) -> np.ndarray:
    """psi(x_i, y_j) = integral of u from the bottom wall (y = 0, psi = 0) to y_j along each vertical
    grid line (trapezoid), the definition of Marchi et al. 2021 (Sec. 3: volume fluxes summed upward
    from the bottom wall; the primary vortex has psi < 0). ``u_grid`` is (ny, nx) on ``y_reference``
    (ny,) which starts at 0."""
    increments = 0.5 * (u_grid[1:] + u_grid[:-1]) * np.diff(y_reference)[:, None]
    return np.concatenate([np.zeros((1, u_grid.shape[1])), np.cumsum(increments, axis=0)], axis=0)


def refine_grid_minimum(x_reference: np.ndarray, y_reference: np.ndarray, field: np.ndarray,
                        half_window: int = 2) -> tuple[float, float, float]:
    """(x, y, value) of the minimum of ``field`` (ny, nx): the grid minimum refined by a least-squares
    quadratic surface over the (2 half_window + 1)^2 neighbourhood."""
    j, i = np.unravel_index(np.nanargmin(field), field.shape)
    j0, j1 = max(j - half_window, 0), min(j + half_window + 1, field.shape[0])
    i0, i1 = max(i - half_window, 0), min(i + half_window + 1, field.shape[1])
    xx, yy = np.meshgrid(x_reference[i0:i1] - x_reference[i], y_reference[j0:j1] - y_reference[j])
    zz = field[j0:j1, i0:i1]
    finite = np.isfinite(zz).ravel()
    if finite.sum() < 6:
        return float(x_reference[i]), float(y_reference[j]), float(field[j, i])
    xs, ys = xx.ravel()[finite], yy.ravel()[finite]
    design = np.stack([np.ones(xs.size), xs, ys, xs ** 2, xs * ys, ys ** 2], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, zz.ravel()[finite], rcond=None)
    _, bx, by, axx, axy, ayy = coefficients
    hessian = np.array([[2 * axx, axy], [axy, 2 * ayy]])
    if np.linalg.det(hessian) <= 0 or hessian[0, 0] <= 0:
        return float(x_reference[i]), float(y_reference[j]), float(field[j, i])
    offset = np.linalg.solve(hessian, -np.array([bx, by]))
    if np.any(np.abs(offset) > np.array([x_reference[i1 - 1] - x_reference[i0], y_reference[j1 - 1] - y_reference[j0]]) / 2):
        return float(x_reference[i]), float(y_reference[j]), float(field[j, i])
    value = coefficients @ np.array([1.0, offset[0], offset[1], offset[0] ** 2, offset[0] * offset[1], offset[1] ** 2])
    return float(x_reference[i] + offset[0]), float(y_reference[j] + offset[1]), float(value)
