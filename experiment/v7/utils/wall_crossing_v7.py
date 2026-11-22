"""
wall_crossing_v7.py - host side of case.yaml numerics.wall_boundary: crossing (the crossing-fraction walls of an
axis-aligned box; shaders/wall_boundary.glsl has the rules and the GPU side, utils/wall_store_v7.py the static wall
store the walls live in since plan C, branch v7-crossing-wall-store).

What lives here:
  - the settings of the mode (WallCrossingSettings: distance floor, the fluid box, the rule set of the wall material
    table, the organisation of the two passes) and their parsing from case.yaml's numerics.wall_crossing mapping;
  - the geometry of the box in numpy (the shader's plane-table geometry gives the same values on a box):
        phi(r) = min over the active axes of min(r_a - lower_a, upper_a - r_a)
        d(r)   = | max(lower - r, r - upper, 0) |
    and the pair geometry (box_pair_geometry): the face the segment i -> j leaves the box through, its normal,
    the pair's d_ij / phi_ij and the mirror point of the wall particle;
  - the load-time check: every non-wall particle inside the box, every wall particle outside it;
  - the diagnostics the report asks for (smallest phi of the fluid).

The exploration switches of the particle-pool version (git tag crossing-pool) are gone: the second-order extension,
the nearest_wall geometry, the diagnostic delta_wall values, the summed wall pressure after the loop, continuity
wall_velocity. A case that writes one of those keys with the value the wall store implements is accepted; any other
value is refused with a pointer to the tag.

No Vulkan import: the CPU reference (experiment/wall_crossing/reference.py) and the case loader use it without a
device.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

WALL_CROSSING_DEFAULT_DISTANCE_FLOOR_FACTOR = 0.25
# the scheme the wall store implements, under the names of the particle-pool version's switches (tools import them)
WALL_CROSSING_DEFAULT_GEOMETRY = "pairwise"
WALL_CROSSING_DEFAULT_DELTA_WALL = "reflection"
WALL_CROSSING_DEFAULT_CONTINUITY_WALL_COMPONENT = "full"
WALL_CROSSING_FIRST_ORDER_SETTINGS = (("geometry", "nearest_wall"), ("continuity_wall_component", "normal"),
                                      ("pressure_wall", "extension"), ("delta_wall", "extension"))
# numerics.wall_crossing.rule_set: the row of the wall material table every wall particle takes (wall_store_v7)
WALL_CROSSING_RULE_SETS = ("default", "continuity_normal")
WALL_CROSSING_DEFAULT_RULE_SET = "default"
# numerics.wall_crossing.split_dispatch: the organisation of the two passes (claude/wall_store_report.md).
#   true  = (B) the interior pipeline (no wall segment) and the near-wall pipeline, two dispatches per pass: the
#           default (measured faster than (A) on every case: (A) gives every particle the near-wall registers);
#   false = (A) one pipeline for every particle, the near-wall flag decides whether the wall segment runs.
WALL_CROSSING_DEFAULT_SPLIT_DISPATCH = True
# numerics.wall_crossing.split_order, (B) only: which of a pass's two dispatches is recorded first (no barrier
# between them). Not written: by dimension, the faster order measured on the single-GPU cases of the report:
# near_wall_first in 2-D (the near-wall dispatch is the latency-bound long pole there), interior_first in 3-D.
WALL_CROSSING_DEFAULT_SPLIT_ORDER = {2: "near_wall_first", 3: "interior_first"}
# keys of the particle-pool version and the values the wall store implements (a case writing one is accepted)
WALL_CROSSING_LEGACY_KEYS = {
    "continuity": ("extrapolate",),
    "geometry": ("pairwise",),
    "pressure_wall": ("reflection",),
    "delta_wall": ("reflection",),
    "viscous_extension": ("linear",),
    "continuity_extension": ("linear",),
    "continuity_wall_component": ("full", "normal"),        # normal = rule_set continuity_normal
    "pressure_wall_evaluation": ("summed", "pairwise"),     # the wall store evaluates every wall pair
}

KIND_BOUNDARY = 1       # case_v7.KIND_BOUNDARY (not imported: case_v7 imports this module)


@dataclass(frozen=True)
class WallCrossingSettings:
    """case.yaml numerics.wall_crossing, resolved by the loader (the box is never None after loading).

    distance_floor_factor: phi_ij >= this x the particle spacing (2 x particle_radius).
    box_lower / box_upper: the fluid box in world coordinates, three components (z unused in 2-D); its planes are the
        plane table of the wall store.
    box_from_case: True when case.yaml gave the box, False when the loader derived it from the fluid particles'
        bounding box widened by one particle radius.
    rule_set: the row of the wall material table (wall_store_v7.WALL_STORE_RULE_SETS): "default" or
        "continuity_normal" (the continuity equation takes the normal component of the wall velocity's prescribed
        value only; continuity_wall_component normal of the particle-pool version).
    split_dispatch: the organisation of the two passes: False = (A) one pipeline, True = (B) interior and near-wall
        pipelines; split_dispatch_explicit records whether the case wrote it.
    split_order: (B) only, "near_wall_first" or "interior_first": which dispatch comes first; None (not written) =
        WALL_CROSSING_DEFAULT_SPLIT_ORDER of the case's dimension (resolved_split_order).
    """
    distance_floor_factor: float = WALL_CROSSING_DEFAULT_DISTANCE_FLOOR_FACTOR
    box_lower: Optional[tuple[float, float, float]] = None
    box_upper: Optional[tuple[float, float, float]] = None
    box_from_case: bool = False
    rule_set: str = WALL_CROSSING_DEFAULT_RULE_SET
    split_dispatch: bool = WALL_CROSSING_DEFAULT_SPLIT_DISPATCH
    split_dispatch_explicit: bool = False
    split_order: Optional[str] = None

    def resolved_split_order(self, dimension: int) -> str:
        """The order (B) records its two dispatches in: the case's, or the default of the dimension."""
        return self.split_order or WALL_CROSSING_DEFAULT_SPLIT_ORDER[3 if int(dimension) == 3 else 2]

    def distance_floor(self, particle_radius: float) -> float:
        """The floor of phi_ij in metres: the factor x the particle spacing."""
        return float(self.distance_floor_factor) * 2.0 * float(particle_radius)


def box_distance_to_wall(positions: np.ndarray, box_lower, box_upper, dimension: int) -> np.ndarray:
    """phi(r) for every row of positions (N, 3): the distance of a point of the box to its nearest wall plane,
    negative outside the box. float64."""
    positions = np.asarray(positions, dtype=np.float64)
    lower = np.asarray(box_lower, dtype=np.float64)
    upper = np.asarray(box_upper, dtype=np.float64)
    axis_gap = np.minimum(positions - lower, upper - positions)[:, :int(dimension)]
    return axis_gap.min(axis=1)


def box_depth_behind_wall(positions: np.ndarray, box_lower, box_upper, dimension: int) -> np.ndarray:
    """d(r) for every row of positions (N, 3): the distance of a point outside the box to the box (to the
    edge or corner in an edge or corner block), 0 inside. float64."""
    positions = np.asarray(positions, dtype=np.float64)
    lower = np.asarray(box_lower, dtype=np.float64)
    upper = np.asarray(box_upper, dtype=np.float64)
    outside = np.maximum(np.maximum(lower - positions, positions - upper), 0.0)[:, :int(dimension)]
    return np.sqrt((outside * outside).sum(axis=1))


def box_pair_geometry(fluid_positions: np.ndarray, wall_positions: np.ndarray, box_lower, box_upper,
                      dimension: int, distance_floor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The geometry of P pairs (fluid particle i inside the box, wall particle j outside it), the shader's
    wall_crossing_pair_geometry on a box row by row. float64.

    The wall particle lies outside the box on one, two or three axes; each names a face. The segment i -> j
    leaves the box through the face with the smallest s_a = phi_a / (phi_a + d_a), phi_a = the distance of i to
    the face (0 for a particle that has left the box there), d_a = the depth of j behind it; equal s_a: the first
    axis in the order x, y, z. Returns
        normal (P, 3)               the inward unit normal of that face
        depth_ratio (P,)            d_ij / max(phi_ij, distance_floor)        (1 / t_ij = 1 + depth_ratio)
        mirror_displacement (P, 3)  r_j - r_j', r_j' = r_j reflected in every face it lies outside of
    A wall position that is not outside the box gets zeros."""
    fluid_positions = np.asarray(fluid_positions, dtype=np.float64)
    wall_positions = np.asarray(wall_positions, dtype=np.float64)
    lower = np.asarray(box_lower, dtype=np.float64)
    upper = np.asarray(box_upper, dtype=np.float64)
    nearest_box_point = np.clip(wall_positions, lower, upper)
    outside = wall_positions - nearest_box_point
    outside[:, int(dimension):] = 0.0
    pair_count = wall_positions.shape[0]
    normal = np.zeros((pair_count, 3))
    depth_ratio = np.zeros(pair_count)
    smallest_crossing = np.full(pair_count, 2.0)
    for axis in range(3):
        depth = np.abs(outside[:, axis])
        outward_sign = np.sign(outside[:, axis])
        inward_distance = outward_sign * (nearest_box_point[:, axis] - fluid_positions[:, axis])
        clamped_distance = np.maximum(inward_distance, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            crossing = clamped_distance / (clamped_distance + depth)
        chosen = (depth > 0.0) & (crossing < smallest_crossing)
        smallest_crossing[chosen] = crossing[chosen]
        depth_ratio[chosen] = depth[chosen] / np.maximum(inward_distance[chosen], float(distance_floor))
        normal[chosen] = 0.0
        normal[chosen, axis] = -outward_sign[chosen]
    return normal, depth_ratio, 2.0 * outside


def parse_wall_crossing_settings(numerics_mapping: dict) -> WallCrossingSettings:
    """numerics.wall_crossing of case.yaml (optional mapping; every key optional):
        distance_floor_factor: <float > 0>                 (default 0.25)
        box: {lower: [x, y, z], upper: [x, y, z]}          (default: from the fluid particles)
        rule_set: default | continuity_normal              (default default; the wall material table's row)
        split_dispatch: true | false                       (default true = (B) the interior and the near-wall
                                                            pipelines; false = (A) one pipeline)
        split_order: near_wall_first | interior_first      ((B) only; default by dimension: near_wall_first in
                                                            2-D, interior_first in 3-D)
    and the keys of the particle-pool version (WALL_CROSSING_LEGACY_KEYS) with a value the wall store implements;
    continuity_wall_component normal selects rule_set continuity_normal. Anything else raises ValueError. A 2-D
    case may give two components per corner."""
    mapping = numerics_mapping.get("wall_crossing") or {}
    if not isinstance(mapping, dict):
        raise ValueError(f"numerics.wall_crossing: expected a mapping, got {mapping!r}")
    known = ({"distance_floor_factor", "box", "rule_set", "split_dispatch", "split_order"}
             | set(WALL_CROSSING_LEGACY_KEYS))
    unknown = sorted(set(mapping) - known)
    if unknown:
        raise ValueError(f"numerics.wall_crossing: unknown keys {unknown} (known: {', '.join(sorted(known))})")
    for key, accepted in WALL_CROSSING_LEGACY_KEYS.items():
        if key in mapping and str(mapping[key]) not in accepted:
            raise ValueError(f"numerics.wall_crossing.{key}: {mapping[key]!r} is a switch of the particle-pool "
                             f"version (git tag crossing-pool); the wall store implements {key} "
                             f"{' or '.join(accepted)} only")
    rule_set = str(mapping.get("rule_set", WALL_CROSSING_DEFAULT_RULE_SET))
    if rule_set not in WALL_CROSSING_RULE_SETS:
        raise ValueError(f"numerics.wall_crossing.rule_set: {rule_set!r}, expected one of "
                         f"{', '.join(WALL_CROSSING_RULE_SETS)}")
    if str(mapping.get("continuity_wall_component", "full")) == "normal":
        if "rule_set" in mapping and rule_set != "continuity_normal":
            raise ValueError("numerics.wall_crossing: continuity_wall_component normal contradicts rule_set "
                             f"{rule_set}")
        rule_set = "continuity_normal"
    split_order = mapping.get("split_order")
    if split_order is not None and split_order not in ("interior_first", "near_wall_first"):
        raise ValueError(f"numerics.wall_crossing.split_order: {split_order!r}, expected interior_first or "
                         "near_wall_first")
    split_dispatch_explicit = "split_dispatch" in mapping
    split_dispatch = mapping.get("split_dispatch", WALL_CROSSING_DEFAULT_SPLIT_DISPATCH)
    if not isinstance(split_dispatch, bool):
        raise ValueError(f"numerics.wall_crossing.split_dispatch: {split_dispatch!r}, expected true or false")
    distance_floor_factor = float(mapping.get("distance_floor_factor", WALL_CROSSING_DEFAULT_DISTANCE_FLOOR_FACTOR))
    if not distance_floor_factor > 0.0:
        raise ValueError(f"numerics.wall_crossing.distance_floor_factor: {distance_floor_factor!r}, expected > 0")
    box = mapping.get("box")
    box_lower = box_upper = None
    if box is not None:
        if not isinstance(box, dict) or set(box) != {"lower", "upper"}:
            raise ValueError(f"numerics.wall_crossing.box: expected {{lower: [...], upper: [...]}}, got {box!r}")
        corners = []
        for key in ("lower", "upper"):
            components = [float(component) for component in box[key]]
            if len(components) not in (2, 3):
                raise ValueError(f"numerics.wall_crossing.box.{key}: {box[key]!r}, expected 2 or 3 components")
            corners.append(tuple(components + [0.0] * (3 - len(components))))
        box_lower, box_upper = corners
    return WallCrossingSettings(distance_floor_factor=distance_floor_factor, box_lower=box_lower,
                                box_upper=box_upper, box_from_case=box is not None, rule_set=rule_set,
                                split_dispatch=split_dispatch, split_dispatch_explicit=split_dispatch_explicit,
                                split_order=split_order)


def resolve_wall_crossing_box(settings: WallCrossingSettings, positions: np.ndarray, is_wall: np.ndarray,
                              particle_radius: float, dimension: int) -> WallCrossingSettings:
    """The settings with the box filled in and checked against the particles.

    Without a box in case.yaml: the bounding box of the non-wall particles widened by one particle radius on
    every active axis (a lattice whose outermost fluid row sits half a spacing from the wall plane gives the
    wall planes). Then, for either source of the box: every non-wall particle strictly inside (phi > 0),
    every wall particle strictly outside (d > 0); anything else raises ValueError with the counts and the
    first offender. The unused z of a 2-D box is stored as 0."""
    positions = np.asarray(positions, dtype=np.float64)
    is_wall = np.asarray(is_wall, dtype=bool)
    active = int(dimension)
    if not (~is_wall).any():
        raise ValueError("wall_boundary crossing: the case has no fluid particle (the box is the fluid's)")
    if settings.box_lower is None:
        fluid = positions[~is_wall]
        lower = fluid.min(axis=0) - float(particle_radius)
        upper = fluid.max(axis=0) + float(particle_radius)
    else:
        lower = np.asarray(settings.box_lower, dtype=np.float64)
        upper = np.asarray(settings.box_upper, dtype=np.float64)
    lower, upper = lower.copy(), upper.copy()
    lower[active:] = 0.0
    upper[active:] = 0.0
    if not (upper[:active] > lower[:active]).all():
        raise ValueError(f"wall_boundary crossing: empty box lower {lower.tolist()} upper {upper.tolist()}")

    fluid_distance = box_distance_to_wall(positions[~is_wall], lower, upper, active)
    wall_depth = box_depth_behind_wall(positions[is_wall], lower, upper, active)
    fluid_outside = fluid_distance <= 0.0
    wall_inside = wall_depth <= 0.0
    if fluid_outside.any() or wall_inside.any():
        source = "numerics.wall_crossing.box" if settings.box_from_case else "the fluid bounding box + one radius"
        details = []
        if fluid_outside.any():
            first = positions[~is_wall][np.flatnonzero(fluid_outside)[0]]
            details.append(f"{int(fluid_outside.sum()):,} fluid particle(s) not inside the box "
                           f"(first at {first.tolist()}, phi = {float(fluid_distance[fluid_outside][0]):.6g})")
        if wall_inside.any():
            first = positions[is_wall][np.flatnonzero(wall_inside)[0]]
            details.append(f"{int(wall_inside.sum()):,} wall particle(s) not outside the box "
                           f"(first at {first.tolist()})")
        raise ValueError(f"wall_boundary crossing: box lower {lower[:active].tolist()} upper "
                         f"{upper[:active].tolist()} ({source}) does not separate fluid and walls: "
                         + "; ".join(details) + ". The mode handles an axis-aligned box whose walls lie "
                         "outside and whose fluid lies inside; give numerics.wall_crossing.box if the default "
                         "is wrong.")
    return replace(settings,
                   box_lower=tuple(float(component) for component in lower),
                   box_upper=tuple(float(component) for component in upper))


def describe_wall_crossing(settings: WallCrossingSettings, particle_radius: float, dimension: int,
                           positions: Optional[np.ndarray] = None, is_wall: Optional[np.ndarray] = None) -> str:
    """One line for the loader's log: the box (and where it came from), the floor, the rule set, the organisation
    and, with the particles, the smallest fluid phi and the smallest / largest wall depth."""
    active = int(dimension)
    organisation = (("(B) split dispatch, " + settings.resolved_split_order(dimension)) if settings.split_dispatch
                    else "(A) one pipeline")
    text = (f"wall_boundary crossing: box lower {list(settings.box_lower[:active])} upper "
            f"{list(settings.box_upper[:active])} "
            f"({'case.yaml' if settings.box_from_case else 'fluid bounding box + one particle radius'}), "
            f"distance floor {settings.distance_floor(particle_radius):.6g} m "
            f"(= {settings.distance_floor_factor:g} x spacing), rule set {settings.rule_set}, {organisation}")
    if positions is not None and is_wall is not None:
        is_wall = np.asarray(is_wall, dtype=bool)
        fluid_distance = box_distance_to_wall(np.asarray(positions)[~is_wall], settings.box_lower,
                                              settings.box_upper, active)
        text += f"; fluid phi min {float(fluid_distance.min()):.6g} m"
        if is_wall.any():
            wall_depth = box_depth_behind_wall(np.asarray(positions)[is_wall], settings.box_lower,
                                               settings.box_upper, active)
            text += f", wall depth {float(wall_depth.min()):.6g} .. {float(wall_depth.max()):.6g} m"
    return text


def fluid_distance_statistics(positions: np.ndarray, is_fluid: np.ndarray, settings: WallCrossingSettings,
                              particle_radius: float, dimension: int) -> dict:
    """Diagnostics of a state: the smallest phi over the fluid rows, how many sit below the floor and how many
    have left the box (phi <= 0)."""
    is_fluid = np.asarray(is_fluid, dtype=bool)
    if not is_fluid.any():
        return {"fluid_count": 0, "distance_minimum": None, "below_floor_count": 0, "outside_box_count": 0}
    distance = box_distance_to_wall(np.asarray(positions)[is_fluid], settings.box_lower, settings.box_upper,
                                    int(dimension))
    floor = settings.distance_floor(particle_radius)
    return {"fluid_count": int(is_fluid.sum()),
            "distance_minimum": float(distance.min()),
            "distance_floor": float(floor),
            "below_floor_count": int((distance < floor).sum()),
            "outside_box_count": int((distance <= 0.0).sum())}
