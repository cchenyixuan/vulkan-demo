"""
deep_wall_single.py — E39 B4's deep-wall skip decision for one slab without peers.

The auto rule and the candidate count are those of experiment/v7/utils/simulator_v7.py
(deep_wall_candidate_count / _resolve_deep_wall_skip), copied so that v7_single does not import the
multi-GPU simulator module (which reads twenty V7_* switches at import time). The CPU test
experiment/v7_single/_test_single_cpu.py checks that the two functions agree on the acceptance
cases. The reason strings are v7's, so run headers and dump metadata of both builds read alike.
"""

from __future__ import annotations

import numpy as np

from experiment.v7.utils.case_v7 import CaseV7, KIND_BOUNDARY

# The AUTO rule's threshold (simulator_v7._DEEP_WALL_AUTO_MINIMUM_CANDIDATE_FRACTION).
AUTO_MINIMUM_CANDIDATE_FRACTION = 0.01


def deep_wall_candidate_count(case: CaseV7, skip_band_width: int) -> int:
    """Walls deep_wall_marker.comp + the skip decision would skip on the slab's initial particles (the AUTO
    rule's input): bin with initialize_voxelization's float32 formula, presence = a non-BOUNDARY particle in
    an own voxel, deep = every in-grid 3^d neighbour own and without presence (z only in 3-D), skip = BOUNDARY
    in a deep voxel outside the band of width skip_band_width (v7's function, verbatim)."""
    grid, physics = case.grid, case.physics
    dimensions = np.array([grid.grid_dimension_x, grid.grid_dimension_y, grid.grid_dimension_z])
    origin = np.array([grid.origin_x, grid.origin_y, grid.origin_z], dtype=np.float32)
    smoothing_length = np.float32(physics.smoothing_length)
    positions = np.asarray(case.initial.positions, dtype=np.float32)
    if positions.shape[0] == 0:
        return 0
    coordinates = np.floor((positions - origin) / smoothing_length).astype(np.int32)
    kinds = np.array([material.kind for material in case.materials])[np.asarray(case.initial.material_group)]
    inside = np.all((coordinates >= 0) & (coordinates < dimensions), axis=1)
    face = int(dimensions[1] * dimensions[2])
    leading_x = case.ghost_grid.leading_ghost_voxel_count // face
    trailing_x = case.ghost_grid.trailing_ghost_voxel_count // face
    own = np.zeros(dimensions, dtype=bool)
    own[leading_x:int(dimensions[0]) - trailing_x] = True
    present = np.zeros(dimensions, dtype=bool)
    present[tuple(coordinates[inside & (kinds != KIND_BOUNDARY)].T)] = True
    blocked = np.pad(present | ~own, 1, constant_values=False)      # outside the grid: not visited
    neighborhood_blocked = np.zeros(dimensions, dtype=bool)
    z_range = int(physics.neighbor_z_range)
    for delta_x in (-1, 0, 1):
        for delta_y in (-1, 0, 1):
            for delta_z in range(-z_range, z_range + 1):
                neighborhood_blocked |= blocked[1 + delta_x:1 + delta_x + dimensions[0],
                                                 1 + delta_y:1 + delta_y + dimensions[1],
                                                 1 + delta_z:1 + delta_z + dimensions[2]]
    deep = own & ~neighborhood_blocked
    column = coordinates[:, 0]
    own_last_x = int(dimensions[0]) - 1 - trailing_x
    in_band = (((leading_x > 0) & (column < leading_x + skip_band_width))
               | ((trailing_x > 0) & (column > own_last_x - skip_band_width)))
    walls = np.flatnonzero(inside & (kinds == KIND_BOUNDARY) & ~in_band)
    return int(deep[tuple(coordinates[walls].T)].sum())


def resolve_deep_wall_skip(case: CaseV7, mode: str, density_band_width: int) -> tuple[bool, str]:
    """(active, reason) of V7_DEEP_WALL_SKIP for this slab: 0 off, 1 forced on, auto = on iff the slab is 3-D
    and at least AUTO_MINIMUM_CANDIDATE_FRACTION of its initial particles are deep-wall candidates (v7's
    _resolve_deep_wall_skip, same reason strings)."""
    if mode == "0":
        return False, "V7_DEEP_WALL_SKIP=0"
    if mode == "1":
        return True, "forced (V7_DEEP_WALL_SKIP=1)"
    dimension = int(case.physics.dimension)
    if dimension != 3:
        return False, f"auto: {dimension}-D slab"
    candidates = deep_wall_candidate_count(case, density_band_width)
    particles = int(case.initial.positions.shape[0])
    fraction = candidates / particles if particles else 0.0
    verdict = fraction >= AUTO_MINIMUM_CANDIDATE_FRACTION
    return verdict, (f"auto: 3-D, {candidates:,} deep-wall candidates at the initial state = "
                     f"{100.0 * fraction:.2f} % of {particles:,} particles "
                     f"{'>=' if verdict else '<'} {100.0 * AUTO_MINIMUM_CANDIDATE_FRACTION:g} %")
