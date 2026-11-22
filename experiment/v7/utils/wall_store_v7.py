"""
wall_store_v7.py - the static wall store of wall_boundary crossing (plan C, branch v7-crossing-wall-store).

The wall particles of a crossing case do not live in the particle pool. The loader takes them out of the initial
state and keeps them here; every slab uploads the walls of its extended grid (own and ghost columns) once, into
buffers that nothing rewrites:

  wall records (32 B each, sorted by voxel of the slab's grid):
      position_volume  vec4   position, volume (the wall particle's m / rho as the particle pool had it:
                              float32(float32(rest_density x volume) / float32(rest_density)))
      plane_ids        uint   up to three plane ids of WALL_STORE_PLANE_ID_BITS bits each, in ascending order;
                              WALL_STORE_PLANE_ID_NONE fills the unused fields (a face particle has one plane, an
                              edge particle two, a corner particle three: the planes it lies behind)
      body_id          uint   row of the body table
      wall_material_id uint   row of the wall material (rule) table
      flags            uint   reserved, 0
  wall voxel offsets (uint, voxel count + 2): the walls of voxel id v are records [offset[v], offset[v + 1])
  near-wall voxel flags (uint, voxel count + 1): 1 iff a wall particle lies closer than the support radius h
      (with a margin of h / 256 against rounding) to the voxel's cube; a fluid particle elsewhere has no wall
      neighbour. Computed from every wall of the case, not only the slab's.
  plane table (32 B per plane): inward unit normal n (xyz, 0), a point p of the plane (xyz, 0). The axis-aligned
      box gives 4 planes (2-D) or 6 (3-D) in the order x lower, x upper, y lower, y upper, z lower, z upper.
  body table (80 B per body): linear velocity V, angular velocity omega, rotation centre c, linear acceleration A,
      angular acceleration alpha (each xyz, 0). v_w(r) = V + omega x (r - c),
      a_w(r) = A + alpha x (r - c) + omega x (omega x (r - c)). Body 0 is the fixed wall; every other initial
      velocity of the wall materials (the lid of a cavity) is one more body moving with it.
  wall material table (64 B per row): the boundary rule of every field and operator (shaders/wall_boundary.glsl):
      [0] velocity in the continuity equation  (lambda_normal, lambda_tangential, 0, 0)
      [1] velocity in the viscous term         (lambda_normal, lambda_tangential, 0, 0)
      [2] pressure                             (lambda, gradient weight, body-force weight, prescribed value)
      [3] density                              (lambda, gradient weight, body-force weight, prescribed value)
    lambda = 1: prescribed value, lambda = 0: prescribed normal derivative. Row
    WALL_STORE_RULE_SETS.index(name): "default" (the scheme of the particle-pool version's defaults) and
    "continuity_normal" (the continuity equation takes the normal component of the velocity's prescribed value
    only, the tangential component v_j = v_i; the 2026-11-18 scheme's choice, kept for comparisons).

No Vulkan import: the loader, the partition and the tests use it without a device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

WALL_STORE_PLANE_ID_BITS = 10
WALL_STORE_PLANE_ID_NONE = (1 << WALL_STORE_PLANE_ID_BITS) - 1
WALL_STORE_PLANES_PER_PARTICLE = 3
WALL_STORE_RECORD_BYTES = 32
WALL_STORE_PLANE_BYTES = 32
WALL_STORE_BODY_BYTES = 80
WALL_STORE_RULE_ROW_BYTES = 64
# the near-wall flag's margin against rounding (the particle-pool version's marker: 1 / 256 of a voxel)
WALL_STORE_NEAR_WALL_MARGIN = 1.0 / 256.0
# rows of the wall material table (case.yaml numerics.wall_crossing.rule_set)
WALL_STORE_RULE_SETS = ("default", "continuity_normal")
WALL_STORE_DEFAULT_RULE_SET = "default"


def rule_table() -> np.ndarray:
    """(len(WALL_STORE_RULE_SETS), 4, 4) float32: the wall material table, one row per rule set."""
    value, derivative = 1.0, 0.0
    default = np.array([
        [value, value, 0.0, 0.0],            # continuity velocity: prescribed value, normal and tangential
        [value, value, 0.0, 0.0],            # viscous velocity: prescribed value, normal and tangential
        [derivative, 1.0, 1.0, 0.0],         # pressure: prescribed normal derivative, G_p and rho_i (g - a_w)
        [derivative, 1.0, 1.0, 0.0],         # density: prescribed normal derivative, G and rho_i (g - a_w) / c^2
    ], dtype=np.float32)
    continuity_normal = default.copy()
    continuity_normal[0] = [value, derivative, 0.0, 0.0]     # tangential part of the continuity velocity: v_j = v_i
    return np.stack([default, continuity_normal])


@dataclass
class WallSource:
    """Every wall particle the crossing case keeps, in load order, and the three tables. Built by the loader
    (build_wall_source); every slab case shares it and takes its own layout (layout_for_grid)."""
    positions: np.ndarray            # (N, 3) float32
    volumes: np.ndarray              # (N,) float32
    plane_ids: np.ndarray            # (N,) uint32, packed
    body_ids: np.ndarray             # (N,) uint32
    wall_material_ids: np.ndarray    # (N,) uint32
    material_groups: np.ndarray      # (N,) uint32: the case's material group (output only)
    velocities: np.ndarray           # (N, 3) float32: v_w at the particle (output only)
    masses: np.ndarray               # (N,) float32 (output only)
    densities: np.ndarray            # (N,) float32: the rest density (output only)
    planes: np.ndarray               # (P, 8) float32
    bodies: np.ndarray               # (B, 20) float32
    rules: np.ndarray                # (R, 4, 4) float32
    loaded_count: int                # wall particles of the case files
    discarded_count: int             # of those, deeper than h behind the box (not kept)
    rule_set: str = WALL_STORE_DEFAULT_RULE_SET
    smoothing_length: float = 0.0
    dimension: int = 2

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])


@dataclass
class WallStoreLayout:
    """One slab grid's share of the wall store, ready for upload."""
    record_rows: np.ndarray          # (M,) indices into the WallSource, in the uploaded (voxel) order
    records: bytes                   # M x 32 B
    voxel_offsets: np.ndarray        # (voxel count + 2,) uint32
    near_wall_flags: np.ndarray      # (voxel count + 1,) uint32
    near_wall_voxel_count: int = 0
    statistics: dict = field(default_factory=dict)


def box_planes(box_lower, box_upper, dimension: int) -> np.ndarray:
    """(2 d, 8) float32: the planes of the axis-aligned box, x lower, x upper, y lower, y upper (z lower, z upper),
    each (inward normal xyz, 0, a point of the plane xyz, 0)."""
    rows = []
    for axis in range(int(dimension)):
        for side, corner in ((1.0, box_lower), (-1.0, box_upper)):
            normal = [0.0, 0.0, 0.0]
            normal[axis] = side
            rows.append(normal + [0.0] + [float(component) for component in corner[:3]] + [0.0])
    return np.asarray(rows, dtype=np.float32)


def pack_plane_ids(plane_lists: list) -> np.ndarray:
    packed = np.empty(len(plane_lists), dtype=np.uint32)
    for row, ids in enumerate(plane_lists):
        value = 0
        for slot in range(WALL_STORE_PLANES_PER_PARTICLE):
            plane = ids[slot] if slot < len(ids) else WALL_STORE_PLANE_ID_NONE
            value |= int(plane) << (WALL_STORE_PLANE_ID_BITS * slot)
        packed[row] = value
    return packed


def unpack_plane_ids(packed: np.ndarray) -> np.ndarray:
    """(N, 3) int64, WALL_STORE_PLANE_ID_NONE for an unused field."""
    packed = np.asarray(packed, dtype=np.int64)
    mask = (1 << WALL_STORE_PLANE_ID_BITS) - 1
    return np.stack([(packed >> (WALL_STORE_PLANE_ID_BITS * slot)) & mask
                     for slot in range(WALL_STORE_PLANES_PER_PARTICLE)], axis=1)


def plane_depths(positions: np.ndarray, planes: np.ndarray) -> np.ndarray:
    """(N, P) float64: the depth of every position behind every plane, n . (p - r) (> 0 behind it)."""
    positions = np.asarray(positions, dtype=np.float64)
    normals = planes[:, 0:3].astype(np.float64)
    points = planes[:, 4:7].astype(np.float64)
    return (points * normals).sum(axis=1)[None, :] - positions @ normals.T


def build_wall_source(positions: np.ndarray, velocities: np.ndarray, material_groups: np.ndarray, materials: list,
                      box_lower, box_upper, smoothing_length: float, dimension: int,
                      rule_set: str = WALL_STORE_DEFAULT_RULE_SET, keep_all_walls: bool = False) -> WallSource:
    """The wall store of a crossing case from its wall particles (the rows of positions / velocities /
    material_groups that belong to a boundary material). Keeps the walls closer than smoothing_length to the
    fluid box (a deeper one is never within the support of a fluid particle inside the box); the planes each kept
    wall lies behind; the bodies from the walls' initial velocities; the rule set's row of the material table for
    every wall."""
    if rule_set not in WALL_STORE_RULE_SETS:
        raise ValueError(f"numerics.wall_crossing.rule_set: {rule_set!r}, expected one of "
                         f"{', '.join(WALL_STORE_RULE_SETS)}")
    from experiment.v7.utils.wall_crossing_v7 import box_depth_behind_wall
    positions = np.asarray(positions, dtype=np.float32)
    velocities = np.asarray(velocities, dtype=np.float32)
    material_groups = np.asarray(material_groups, dtype=np.uint32)
    loaded = int(positions.shape[0])
    depth = box_depth_behind_wall(positions, box_lower, box_upper, dimension)
    # DIAGNOSTIC keep_all_walls (V7_WALL_STORE_KEEP_ALL_WALLS=1, case_loader_v7): every wall layer of the case files
    kept = np.ones(depth.size, dtype=bool) if keep_all_walls else depth < float(smoothing_length)
    positions, velocities, material_groups = positions[kept], velocities[kept], material_groups[kept]

    planes = box_planes(box_lower, box_upper, dimension)
    depths = plane_depths(positions, planes)
    plane_lists = [list(np.flatnonzero(row > 0.0)) for row in depths]
    if any(not ids for ids in plane_lists):
        raise ValueError("wall store: a wall particle lies behind no plane of the box")
    if any(len(ids) > WALL_STORE_PLANES_PER_PARTICLE for ids in plane_lists):
        raise ValueError("wall store: a wall particle lies behind more than three planes")

    masses = np.empty(positions.shape[0], dtype=np.float32)
    densities = np.empty(positions.shape[0], dtype=np.float32)
    for group in np.unique(material_groups):
        material = materials[int(group)]
        rows = material_groups == group
        masses[rows] = np.float32(material.rest_density * material.volume)
        densities[rows] = np.float32(material.rest_density)
    volumes = (masses / densities).astype(np.float32)

    # bodies: 0 = the fixed wall; one more per distinct nonzero initial velocity of the walls
    body_ids = np.zeros(positions.shape[0], dtype=np.uint32)
    body_rows = [np.zeros(20, dtype=np.float32)]
    distinct = {}
    for row in range(positions.shape[0]):
        velocity = tuple(float(component) for component in velocities[row])
        if velocity == (0.0, 0.0, 0.0):
            continue
        if velocity not in distinct:
            distinct[velocity] = len(body_rows)
            body = np.zeros(20, dtype=np.float32)
            body[0:3] = velocity
            body_rows.append(body)
        body_ids[row] = distinct[velocity]
    rules = rule_table()
    return WallSource(positions=positions, volumes=volumes, plane_ids=pack_plane_ids(plane_lists),
                      body_ids=body_ids,
                      wall_material_ids=np.full(positions.shape[0], WALL_STORE_RULE_SETS.index(rule_set),
                                                dtype=np.uint32),
                      material_groups=material_groups, velocities=velocities, masses=masses, densities=densities,
                      planes=planes, bodies=np.stack(body_rows), rules=rules, loaded_count=loaded,
                      discarded_count=loaded - int(kept.sum()), rule_set=rule_set,
                      smoothing_length=float(smoothing_length), dimension=int(dimension))


def _voxel_coordinates(positions: np.ndarray, grid, smoothing_length: float) -> np.ndarray:
    """(N, 3) int64: floor((r - origin) / h) per axis (initialize_voxelization.comp's binning; 2-D: z = 0)."""
    origin = np.array([grid.origin_x, grid.origin_y, grid.origin_z], dtype=np.float64)
    coordinates = np.floor((np.asarray(positions, dtype=np.float64) - origin) / float(smoothing_length))
    return coordinates.astype(np.int64)


def _inside_grid(coordinates: np.ndarray, grid) -> np.ndarray:
    dims = np.array([grid.grid_dimension_x, grid.grid_dimension_y, grid.grid_dimension_z])
    return ((coordinates >= 0) & (coordinates < dims[None, :])).all(axis=1)


def _voxel_ids(coordinates: np.ndarray, grid) -> np.ndarray:
    """helpers.glsl own_voxel_id_of: y + z NY + x NY NZ + 1 (x slowest)."""
    return (coordinates[:, 1] + coordinates[:, 2] * grid.grid_dimension_y
            + coordinates[:, 0] * grid.grid_dimension_y * grid.grid_dimension_z + 1)


def near_wall_flags(source: WallSource, grid) -> np.ndarray:
    """(voxel count + 1,) uint32 over the grid's voxel ids: 1 iff a wall of the source (every wall of the case)
    lies closer than h (1 + WALL_STORE_NEAR_WALL_MARGIN) to the voxel's cube."""
    h = float(source.smoothing_length)
    total = grid.grid_dimension_x * grid.grid_dimension_y * grid.grid_dimension_z
    flags = np.zeros(total + 1, dtype=np.uint32)
    if source.count == 0:
        return flags
    origin = np.array([grid.origin_x, grid.origin_y, grid.origin_z], dtype=np.float64)
    positions = source.positions.astype(np.float64)
    coordinates = _voxel_coordinates(positions, grid, h)
    active = 3 if source.dimension == 3 else 2
    limit = (h * (1.0 + WALL_STORE_NEAR_WALL_MARGIN)) ** 2
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
               for dz in ((-1, 0, 1) if active == 3 else (0,))]
    for offset in offsets:
        neighbour = coordinates + np.asarray(offset, dtype=np.int64)[None, :]
        inside = _inside_grid(neighbour, grid)
        if not inside.any():
            continue
        lower = origin[None, :] + neighbour[inside] * h
        upper = lower + h
        gap = np.maximum(np.maximum(lower - positions[inside], positions[inside] - upper), 0.0)[:, :active]
        close = (gap * gap).sum(axis=1) < limit
        flags[_voxel_ids(neighbour[inside][close], grid)] = 1
    return flags


def layout_for_grid(source: WallSource, grid) -> WallStoreLayout:
    """The records of the walls inside the grid (a slab's extended grid: own and ghost columns), sorted by voxel id
    (stable: load order within a voxel), the voxel offsets and the near-wall flags of that grid."""
    h = float(source.smoothing_length)
    total = grid.grid_dimension_x * grid.grid_dimension_y * grid.grid_dimension_z
    coordinates = _voxel_coordinates(source.positions, grid, h)
    inside = np.flatnonzero(_inside_grid(coordinates, grid))
    voxel_ids = _voxel_ids(coordinates[inside], grid)
    order = np.argsort(voxel_ids, kind="stable")
    rows = inside[order]
    sorted_ids = voxel_ids[order]
    counts = np.bincount(sorted_ids, minlength=total + 1)[:total + 1]
    offsets = np.zeros(total + 2, dtype=np.uint32)
    offsets[1:] = np.cumsum(counts).astype(np.uint32)
    record_dtype = np.dtype([("position_volume", np.float32, 4), ("plane_ids", np.uint32), ("body_id", np.uint32),
                             ("wall_material_id", np.uint32), ("flags", np.uint32)])
    records = np.zeros(rows.size, dtype=record_dtype)
    records["position_volume"][:, 0:3] = source.positions[rows]
    records["position_volume"][:, 3] = source.volumes[rows]
    records["plane_ids"] = source.plane_ids[rows]
    records["body_id"] = source.body_ids[rows]
    records["wall_material_id"] = source.wall_material_ids[rows]
    assert record_dtype.itemsize == WALL_STORE_RECORD_BYTES
    flags = near_wall_flags(source, grid)
    return WallStoreLayout(record_rows=rows, records=records.tobytes(), voxel_offsets=offsets,
                           near_wall_flags=flags, near_wall_voxel_count=int(flags.sum()),
                           statistics={"walls": int(rows.size), "voxels": int(total),
                                       "near_wall_voxels": int(flags.sum()),
                                       "largest_voxel_wall_count": int(counts.max()) if counts.size else 0})


def describe_wall_source(source: WallSource) -> str:
    return (f"wall store: {source.count:,} wall particles kept of {source.loaded_count:,} "
            f"({source.discarded_count:,} deeper than h behind the box discarded), {source.planes.shape[0]} planes, "
            f"{source.bodies.shape[0]} bodies, rule set {source.rule_set}")


def output_particles(source: Optional[WallSource]) -> Optional[dict]:
    """The kept walls as particles for frames and samples (written once: they do not move): positions, velocities
    (v_w), mass, density (rest), volume, material group. None without a wall store."""
    if source is None:
        return None
    return {"positions": source.positions, "velocities": source.velocities, "mass": source.masses,
            "density": source.densities, "volumes": source.volumes, "material": source.material_groups}
