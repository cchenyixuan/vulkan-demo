"""
wall_crossing_v7.py - host side of case.yaml numerics.wall_boundary: crossing
(the crossing-fraction walls, first order, axis-aligned box; shaders/wall_boundary.glsl
has the rules and the GPU side).

What lives here:
  - the settings of the mode (WallCrossingSettings: continuity rule, distance floor,
    the fluid box) and their parsing from case.yaml's numerics.wall_crossing mapping;
  - the two functions of the geometry in numpy, the same expressions as the shader's
    wall_crossing_distance_to_wall / wall_crossing_depth_behind_wall:
        phi(r) = min over the active axes of min(r_a - lower_a, upper_a - r_a)
        d(r)   = | max(lower - r, r - upper, 0) |
    (2-D: x and y only). Another geometry replaces these two and the shader's two;
  - the load-time check: every non-wall particle inside the box, every wall particle
    outside it;
  - the diagnostics the report asks for (smallest phi of the fluid, fallback count).

No Vulkan import: the CPU reference (experiment/wall_crossing/reference.py)
and the case loader use it without a device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

WALL_CROSSING_CONTINUITY_CHOICES = ("extrapolate", "wall_velocity")     # spec const 120: 0, 1
WALL_CROSSING_DEFAULT_CONTINUITY = "extrapolate"
WALL_CROSSING_DEFAULT_DISTANCE_FLOOR_FACTOR = 0.25
# DIAGNOSTIC (spec const 116: 0, 1, 2), not part of the mode: how a wall neighbour enters the density diffusion.
# "extension" is the mode as specified (the density continued across the wall); "rest_density" takes simple's
# pair term rho0 - rho_i instead; "body_force" takes the pair term rho_i g . (r_j - r_i) / c^2 (c^2 = dp/drho
# of the Tait EOS at rho_i, wall acceleration 0), which is zero without gravity. See shaders/wall_boundary.glsl.
WALL_CROSSING_DELTA_WALL_CHOICES = ("extension", "rest_density", "body_force")
WALL_CROSSING_DEFAULT_DELTA_WALL = "extension"

KIND_BOUNDARY = 1       # case_v7.KIND_BOUNDARY (not imported: case_v7 imports this module)


@dataclass(frozen=True)
class WallCrossingSettings:
    """case.yaml numerics.wall_crossing, resolved by the loader (the box is never None after loading).

    continuity: the velocity difference of a wall neighbour in the continuity equation, "extrapolate"
        ((v_w - v_i) (1 + d_j / phi_i)) or "wall_velocity" (v_w - v_i).
    distance_floor_factor: phi_i >= this x the particle spacing (2 x particle_radius).
    box_lower / box_upper: the fluid box in world coordinates, three components (z unused in 2-D).
    box_from_case: True when case.yaml gave the box, False when the loader derived it from the fluid
        particles' bounding box widened by one particle radius.
    delta_wall: DIAGNOSTIC, default "extension" (the mode as specified); "rest_density" lets a wall neighbour
        enter the density diffusion with simple's pair term rho0 - rho_i, "body_force" with the pair term
        rho_i g . (r_j - r_i) / c^2 (nothing without gravity).
    """
    continuity: str = WALL_CROSSING_DEFAULT_CONTINUITY
    distance_floor_factor: float = WALL_CROSSING_DEFAULT_DISTANCE_FLOOR_FACTOR
    box_lower: Optional[tuple[float, float, float]] = None
    box_upper: Optional[tuple[float, float, float]] = None
    box_from_case: bool = False
    delta_wall: str = WALL_CROSSING_DEFAULT_DELTA_WALL

    def continuity_code(self) -> int:
        return WALL_CROSSING_CONTINUITY_CHOICES.index(self.continuity)

    def delta_wall_code(self) -> int:
        return WALL_CROSSING_DELTA_WALL_CHOICES.index(self.delta_wall)

    def distance_floor(self, particle_radius: float) -> float:
        """The floor of phi_i in metres: the factor x the particle spacing."""
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


def parse_wall_crossing_settings(numerics_mapping: dict) -> WallCrossingSettings:
    """numerics.wall_crossing of case.yaml (optional mapping; every key optional):
        continuity: extrapolate | wall_velocity            (default extrapolate)
        distance_floor_factor: <float > 0>                 (default 0.25)
        box: {lower: [x, y, z], upper: [x, y, z]}          (default: from the fluid particles)
        delta_wall: extension | rest_density | body_force  (DIAGNOSTIC; default extension = the mode)
    A 2-D case may give two components per corner."""
    mapping = numerics_mapping.get("wall_crossing") or {}
    if not isinstance(mapping, dict):
        raise ValueError(f"numerics.wall_crossing: expected a mapping, got {mapping!r}")
    unknown = sorted(set(mapping) - {"continuity", "distance_floor_factor", "box", "delta_wall"})
    if unknown:
        raise ValueError(f"numerics.wall_crossing: unknown keys {unknown} (known: continuity, "
                         "distance_floor_factor, box, delta_wall)")
    delta_wall = str(mapping.get("delta_wall", WALL_CROSSING_DEFAULT_DELTA_WALL))
    if delta_wall not in WALL_CROSSING_DELTA_WALL_CHOICES:
        raise ValueError(f"numerics.wall_crossing.delta_wall: {delta_wall!r}, expected one of "
                         f"{', '.join(WALL_CROSSING_DELTA_WALL_CHOICES)}")
    continuity = str(mapping.get("continuity", WALL_CROSSING_DEFAULT_CONTINUITY))
    if continuity not in WALL_CROSSING_CONTINUITY_CHOICES:
        raise ValueError(f"numerics.wall_crossing.continuity: {continuity!r}, expected one of "
                         f"{', '.join(WALL_CROSSING_CONTINUITY_CHOICES)}")
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
    return WallCrossingSettings(continuity=continuity, distance_floor_factor=distance_floor_factor,
                                box_lower=box_lower, box_upper=box_upper, box_from_case=box is not None,
                                delta_wall=delta_wall)


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
    return WallCrossingSettings(
        continuity=settings.continuity, distance_floor_factor=settings.distance_floor_factor,
        box_lower=tuple(float(component) for component in lower),
        box_upper=tuple(float(component) for component in upper),
        box_from_case=settings.box_from_case, delta_wall=settings.delta_wall)


def describe_wall_crossing(settings: WallCrossingSettings, particle_radius: float, dimension: int,
                           positions: Optional[np.ndarray] = None, is_wall: Optional[np.ndarray] = None) -> str:
    """One line for the loader's log: the box (and where it came from), the floor, the continuity rule and,
    with the particles, the smallest fluid phi and the smallest / largest wall depth."""
    active = int(dimension)
    text = (f"wall_boundary crossing: box lower {list(settings.box_lower[:active])} upper "
            f"{list(settings.box_upper[:active])} "
            f"({'case.yaml' if settings.box_from_case else 'fluid bounding box + one particle radius'}), "
            f"distance floor {settings.distance_floor(particle_radius):.6g} m "
            f"(= {settings.distance_floor_factor:g} x spacing), continuity {settings.continuity}")
    if settings.delta_wall != WALL_CROSSING_DEFAULT_DELTA_WALL:
        text += f", DIAGNOSTIC delta_wall {settings.delta_wall}"
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
