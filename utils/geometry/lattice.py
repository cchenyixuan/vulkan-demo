"""
lattice.py — N-dimensional particle lattices for geometry discretization.

This is the SINGLE SOURCE OF TRUTH for the two supported packings. By design it
is meant to be shared by both the sampler (particle placement) and the SPH
volume calibration (so placement and calibration can never drift apart). The
lattice geometry matches ``experiment/v4/utils/case_loader_v4._calibrate_particle_volume``
exactly:

  - ``"grid"`` : Cartesian — square in 2D / simple cubic in 3D. Nearest-neighbor
                 distance = ``particle_spacing``. Packing fraction π/4 (2D) / π/6 (3D).
  - ``"hex"``  : closest packing — 2D triangular (6 nearest neighbors) / 3D FCC
                 (12 nearest neighbors). Nearest-neighbor distance = ``particle_spacing``.
                 Packing fraction π/(2√3) ≈ 0.9069 (2D) / π/(3√2) ≈ 0.7405 (3D).

All functions are dimension-generic (``dimension`` ∈ {2, 3}) and vectorized with
numpy so they scale to tens of millions of sites.
"""

from __future__ import annotations

import math

import numpy as np

LATTICE_GRID = "grid"
LATTICE_HEX = "hex"
_VALID_LATTICE_KINDS = (LATTICE_GRID, LATTICE_HEX)

_SQRT_3_OVER_2 = math.sqrt(3.0) / 2.0
_SQRT_2 = math.sqrt(2.0)


def _validate(lattice_kind: str, dimension: int) -> None:
    if lattice_kind not in _VALID_LATTICE_KINDS:
        raise ValueError(f"lattice_kind must be one of {_VALID_LATTICE_KINDS}, got {lattice_kind!r}")
    if dimension not in (2, 3):
        raise ValueError(f"dimension must be 2 or 3, got {dimension}")


def tile_bounding_box(
    bounding_box_min,
    bounding_box_max,
    particle_spacing: float,
    lattice_kind: str,
    dimension: int,
) -> np.ndarray:
    """Return every lattice site whose center falls inside the axis-aligned box.

    ``bounding_box_min`` / ``bounding_box_max`` are length-``dimension`` sequences.
    Returns a ``(N, dimension)`` float64 array. The lattice is anchored to the
    world origin (site indices are integer multiples from 0), so two boxes that
    abut share a consistent lattice — important when sampling adjacent regions
    (e.g. wall shell + fluid interior) that must interlock.
    """
    _validate(lattice_kind, dimension)
    bounding_box_min = np.asarray(bounding_box_min, dtype=np.float64)
    bounding_box_max = np.asarray(bounding_box_max, dtype=np.float64)
    if particle_spacing <= 0.0:
        raise ValueError(f"particle_spacing must be > 0, got {particle_spacing}")

    if lattice_kind == LATTICE_GRID:
        return _tile_grid(bounding_box_min, bounding_box_max, particle_spacing, dimension)
    if dimension == 2:
        return _tile_hex_2d(bounding_box_min, bounding_box_max, particle_spacing)
    return _tile_fcc_3d(bounding_box_min, bounding_box_max, particle_spacing)


def _tile_grid(bounding_box_min, bounding_box_max, particle_spacing, dimension) -> np.ndarray:
    per_axis_coordinates = []
    for axis in range(dimension):
        first_index = math.ceil(bounding_box_min[axis] / particle_spacing)
        last_index = math.floor(bounding_box_max[axis] / particle_spacing)
        per_axis_coordinates.append(
            np.arange(first_index, last_index + 1, dtype=np.float64) * particle_spacing)
    mesh = np.meshgrid(*per_axis_coordinates, indexing="ij")
    return np.stack([component.ravel() for component in mesh], axis=1)


def _tile_hex_2d(bounding_box_min, bounding_box_max, particle_spacing) -> np.ndarray:
    # Primitive vectors a1 = (s, 0), a2 = (s/2, s·√3/2). site(i, j) = i·a1 + j·a2.
    row_spacing = particle_spacing * _SQRT_3_OVER_2
    first_row = math.floor(bounding_box_min[1] / row_spacing) - 1
    last_row = math.ceil(bounding_box_max[1] / row_spacing) + 1
    rows = np.arange(first_row, last_row + 1)
    # The +j·s/2 skew shifts x by up to (|row|·s/2); widen the column range to cover it.
    maximum_skew = max(abs(first_row), abs(last_row)) * particle_spacing * 0.5
    first_column = math.floor((bounding_box_min[0] - maximum_skew) / particle_spacing) - 1
    last_column = math.ceil((bounding_box_max[0] + maximum_skew) / particle_spacing) + 1
    columns = np.arange(first_column, last_column + 1)

    column_index, row_index = np.meshgrid(columns, rows, indexing="ij")
    x = column_index * particle_spacing + row_index * (particle_spacing * 0.5)
    y = row_index * row_spacing
    points = np.stack([x.ravel(), y.ravel()], axis=1)
    return _clip_to_box(points, bounding_box_min, bounding_box_max)


def _tile_fcc_3d(bounding_box_min, bounding_box_max, particle_spacing) -> np.ndarray:
    # FCC: conventional cubic supercell of side a = s·√2 with a 4-atom basis.
    # Nearest-neighbor distance = a/√2 = s; each atom has 12 neighbors at s.
    supercell = particle_spacing * _SQRT_2
    per_axis_indices = []
    for axis in range(3):
        first_cell = math.floor(bounding_box_min[axis] / supercell) - 1
        last_cell = math.ceil(bounding_box_max[axis] / supercell) + 1
        per_axis_indices.append(np.arange(first_cell, last_cell + 1))
    cell_x, cell_y, cell_z = np.meshgrid(*per_axis_indices, indexing="ij")
    cell_origins = np.stack(
        [cell_x.ravel(), cell_y.ravel(), cell_z.ravel()], axis=1).astype(np.float64) * supercell
    basis = np.array(
        [[0.0, 0.0, 0.0],
         [0.5, 0.5, 0.0],
         [0.5, 0.0, 0.5],
         [0.0, 0.5, 0.5]], dtype=np.float64) * supercell
    points = (cell_origins[:, None, :] + basis[None, :, :]).reshape(-1, 3)
    return _clip_to_box(points, bounding_box_min, bounding_box_max)


def _clip_to_box(points, bounding_box_min, bounding_box_max) -> np.ndarray:
    # Half-open-ish inclusive clip with a tiny tolerance so sites exactly on the
    # max face survive floating-point comparison.
    tolerance = 1.0e-9
    inside = np.all(
        (points >= bounding_box_min - tolerance) & (points <= bounding_box_max + tolerance), axis=1)
    return points[inside]


def particle_spacing_for_target_count(
    target_particle_count: int,
    region_measure: float,
    lattice_kind: str,
    dimension: int,
) -> float:
    """Invert the number density to hit a target particle count.

    ``region_measure`` is the region's area (2D) or volume (3D). The returned
    spacing yields approximately ``target_particle_count`` particles when the
    region is filled with ``lattice_kind`` — exact for an infinite lattice, so
    the realized count differs slightly near boundaries (report the actual).
    """
    _validate(lattice_kind, dimension)
    if target_particle_count <= 0:
        raise ValueError("target_particle_count must be positive")
    if region_measure <= 0:
        raise ValueError("region_measure must be positive")
    volume_per_particle = region_measure / target_particle_count
    if lattice_kind == LATTICE_GRID:
        return volume_per_particle ** (1.0 / dimension)
    if dimension == 2:
        # area per triangular-lattice site = s²·√3/2  ->  s = sqrt(2·measure/(√3·N))
        return math.sqrt(volume_per_particle / _SQRT_3_OVER_2)
    # FCC volume per site = s³/√2  ->  s = (√2·measure/N)^(1/3)
    return (volume_per_particle * _SQRT_2) ** (1.0 / 3.0)


def per_particle_volume(lattice_kind: str, dimension: int, particle_spacing: float) -> float:
    """Analytic Voronoi-cell volume of one lattice site (geometric cross-check).

    NOTE: the SPH solver's authoritative per-particle volume is the
    partition-of-unity ``V = 1 / Σ_{j≠i} W(r_ij)``, which is close to but not
    exactly this analytic cell volume. Use this only as a sanity/density check.
    """
    _validate(lattice_kind, dimension)
    if lattice_kind == LATTICE_GRID:
        return particle_spacing ** dimension
    if dimension == 2:
        return particle_spacing * particle_spacing * _SQRT_3_OVER_2
    return particle_spacing ** 3 / _SQRT_2


def neighbor_offsets(
    lattice_kind: str,
    dimension: int,
    particle_spacing: float,
    support_radius: float,
) -> np.ndarray:
    """Lattice-site offsets (relative to a site at the origin, self excluded)
    within ``support_radius``. This is exactly the neighbor set an interior
    particle sees; the SPH volume calibration can consume it instead of
    re-deriving the lattice, guaranteeing placement/calibration agreement.
    """
    _validate(lattice_kind, dimension)
    half = support_radius + 2.0 * particle_spacing      # buffer for hex/FCC skew
    box_min = np.full(dimension, -half)
    box_max = np.full(dimension, half)
    sites = tile_bounding_box(box_min, box_max, particle_spacing, lattice_kind, dimension)
    distance = np.linalg.norm(sites, axis=1)
    keep = (distance <= support_radius + 1.0e-9) & (distance > 1.0e-9)
    return sites[keep]
