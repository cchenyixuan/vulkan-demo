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
  - the pair's geometry of numerics.wall_crossing.geometry pairwise (box_pair_geometry, the shader's
    wall_crossing_pair_geometry): the face the segment i -> j leaves the box through, its normal, the pair's
    d_ij / phi_ij and the mirror point of the wall particle;
  - the load-time check: every non-wall particle inside the box, every wall particle
    outside it;
  - the diagnostics the report asks for (smallest phi of the fluid, fallback count).

No Vulkan import: the CPU reference (experiment/wall_crossing/reference.py)
and the case loader use it without a device.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

WALL_CROSSING_CONTINUITY_CHOICES = ("extrapolate", "wall_velocity")     # spec const 120: 0, 1
WALL_CROSSING_DEFAULT_CONTINUITY = "extrapolate"
WALL_CROSSING_DEFAULT_DISTANCE_FLOOR_FACTOR = 0.25
# DIAGNOSTIC (spec const 116: 0, 1, 2), not part of the mode: how a wall neighbour enters the density diffusion.
# "extension" is the mode as specified (the density continued across the wall); "rest_density" takes simple's
# pair term rho0 - rho_i instead; "body_force" takes the pair term rho_i g . (r_j - r_i) / c^2 (c^2 = dp/drho
# of the Tait EOS at rho_i, wall acceleration 0), which is zero without gravity. See shaders/wall_boundary.glsl.
# "reflection" (3) is not a diagnostic but a variant of the mode (needs geometry pairwise): the density of a wall
# particle is the continued density at its mirror point plus rho_i g . (r_j - r_j') / c^2. The default since
# 2026-11-19 (part of the "complete scheme", below); "extension" was the default until then.
WALL_CROSSING_DELTA_WALL_CHOICES = ("extension", "rest_density", "body_force", "reflection")
WALL_CROSSING_DEFAULT_DELTA_WALL = "reflection"
# Spec const 128 (0, 1): where a pair's crossing fraction and wall normal come from. "nearest_wall" (the mode as it
# was first specified, the default until 2026-11-19): phi_i = the distance of the fluid particle to its nearest
# wall plane, d_j = the distance of the wall particle to the box, n_j = the unit vector from the wall particle to
# the nearest point of the box. "pairwise" (the default since 2026-11-19): from the face the segment i -> j leaves
# the box through (box_pair_geometry below). Not offered together with a quadratic extension.
WALL_CROSSING_GEOMETRY_CHOICES = ("nearest_wall", "pairwise")
WALL_CROSSING_DEFAULT_GEOMETRY = "pairwise"
# Spec const 129 (0, 1): the pressure of a wall neighbour in force. "extension" (the default until 2026-11-19): the
# continuation P_i + G_i . (r_j - r_i). "reflection" (the default since 2026-11-19; needs geometry pairwise): the
# continued pressure at the mirror point plus the body force, P_i + G_i . (r_j' - r_i) + rho_i g . (r_j - r_j').
WALL_CROSSING_PRESSURE_WALL_CHOICES = ("extension", "reflection")
WALL_CROSSING_DEFAULT_PRESSURE_WALL = "reflection"
# report section 27: how force evaluates the wall neighbours' pressure (spec const 135). "summed" (default): the
# sums of the loop and the term after it; "pairwise": p_j per wall pair from G_p the fused pass stored.
WALL_CROSSING_PRESSURE_EVALUATION_CHOICES = ("summed", "pairwise")
WALL_CROSSING_DEFAULT_PRESSURE_EVALUATION = "summed"
# Spec const 117 (0, 1): which part of a wall neighbour's velocity difference enters the continuity equation.
# "full" takes the whole vector (the mode as first specified, the default until 2026-11-18 and again since
# 2026-11-19, as part of the complete scheme); "normal" (the default from 2026-11-18 to 2026-11-19, the user's
# decision after the density budget of the 250^2 cavity) keeps only its component along the wall's inward normal
# before the continuity rule is applied. The viscous term takes the whole velocity difference with either value.
# See shaders/wall_boundary.glsl.
WALL_CROSSING_CONTINUITY_WALL_COMPONENT_CHOICES = ("full", "normal")
WALL_CROSSING_DEFAULT_CONTINUITY_WALL_COMPONENT = "full"
# The defaults of the four switches above are the "complete scheme" since 2026-11-19 (the user's decision after
# sections 14 to 17 of claude/wall_crossing_report.md): geometry pairwise, continuity_wall_component full,
# pressure_wall reflection, delta_wall reflection. Their values from 2026-11-18 to 2026-11-19 (the first-order
# scheme with the normal component) are kept here for the tools that need to name or reproduce them:
WALL_CROSSING_FIRST_ORDER_SETTINGS = (("geometry", "nearest_wall"), ("continuity_wall_component", "normal"),
                                      ("pressure_wall", "extension"), ("delta_wall", "extension"))
# what a quadratic extension needs (it is not offered with the pairwise geometry, and a reflection needs the
# pairwise geometry): named in the error message when a case asks for a quadratic extension alone
WALL_CROSSING_NEAREST_WALL_REQUIREMENTS = (("geometry", "nearest_wall"), ("pressure_wall", "extension"),
                                           ("delta_wall", "extension"))
# Spec consts 118 (viscous_extension) and 119 (continuity_extension), 0 = linear, 1 = quadratic: the order of the
# velocity's continuation across the wall in force's viscous term and in the continuity equation. "linear" (the
# default of both) is the first-order rule (v_w - v_i) (1 + a_j), a_j = d_j / phi_i. "quadratic" is the
# second-order rule (v_w - v_i) (1 + a_j)^2 - a_j grad v_i (r_j - r_i), grad v_i solved per particle in the fused
# pass (shaders/wall_boundary.glsl). continuity_extension quadratic takes the whole velocity difference
# (continuity_wall_component then only names the first-order rule a particle falls back to), needs
# continuity extrapolate and is offered together with viscous_extension quadratic only.
WALL_CROSSING_EXTENSION_CHOICES = ("linear", "quadratic")
WALL_CROSSING_DEFAULT_VISCOUS_EXTENSION = "linear"
WALL_CROSSING_DEFAULT_CONTINUITY_EXTENSION = "linear"
# state of a fluid particle's second-order continuation in a step (wall_crossing_velocity_gradient[pid * 3].w)
WALL_CROSSING_EXTENSION_STATE_NAMES = ("no_wall_neighbour", "valid", "fallback")

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
    delta_wall: "reflection" (default since 2026-11-19; with geometry pairwise): the density of a wall
        neighbour taken at its mirror point plus the body force; "extension" (the default until then): the
        density continued to it; DIAGNOSTICS: "rest_density" lets a wall neighbour enter the density diffusion
        with simple's pair term rho0 - rho_i, "body_force" with the pair term rho_i g . (r_j - r_i) / c^2
        (nothing without gravity).
    continuity_wall_component: "full" (default; also until 2026-11-18) or "normal" (the default from
        2026-11-18 to 2026-11-19): a wall neighbour's velocity difference enters the continuity equation whole,
        or with its component along the wall's inward normal only.
    viscous_extension / continuity_extension: "linear" (default) or "quadratic": the first-order or the
        second-order continuation of the velocity in force's viscous term / in the continuity equation
        (quadratic needs geometry nearest_wall, pressure_wall extension and delta_wall other than reflection).
    geometry: "pairwise" (default since 2026-11-19) or "nearest_wall" (the default until then): the crossing
        fraction and the wall normal pair by pair from the face the segment leaves the box through, or from the
        fluid particle's nearest wall and the wall particle's own wall.
    pressure_wall: "reflection" (default since 2026-11-19; with geometry pairwise) or "extension" (the default
        until then): the pressure of a wall neighbour taken at its mirror point plus the body force, or
        continued to it.
    split_dispatch: True (default since report section 25, 2026-11-20; False until then) or False (report
        section 22): the fused pass and force run as two dispatches each, a build without the mode's code for the
        particles of the voxels with no wall particle in their 3^d neighbourhood and the mode's build for the
        others (shaders/wall_crossing_split.glsl). The same equations; the interior particles have no wall
        neighbour. A case that does not write the key gets the default, which falls back to False on a slab the
        split does not support (a quadratic extension, V7_DENSITY_BUDGET=1, ...; the simulator prints why); a case
        that writes true is refused there (ValueError). split_dispatch_explicit records whether the case wrote it.
    split_order: "near_wall_first" (default since report section 25; "interior_first" until then) or
        "interior_first" (report section 24): which of a split pass's two dispatches comes first (the same
        results; report section 23 measured near_wall_first faster). The debug environment variable
        V7_WALL_CROSSING_SPLIT_ORDER overrides it when set.
    """
    continuity: str = WALL_CROSSING_DEFAULT_CONTINUITY
    distance_floor_factor: float = WALL_CROSSING_DEFAULT_DISTANCE_FLOOR_FACTOR
    box_lower: Optional[tuple[float, float, float]] = None
    box_upper: Optional[tuple[float, float, float]] = None
    box_from_case: bool = False
    delta_wall: str = WALL_CROSSING_DEFAULT_DELTA_WALL
    continuity_wall_component: str = WALL_CROSSING_DEFAULT_CONTINUITY_WALL_COMPONENT
    viscous_extension: str = WALL_CROSSING_DEFAULT_VISCOUS_EXTENSION
    continuity_extension: str = WALL_CROSSING_DEFAULT_CONTINUITY_EXTENSION
    geometry: str = WALL_CROSSING_DEFAULT_GEOMETRY
    pressure_wall: str = WALL_CROSSING_DEFAULT_PRESSURE_WALL
    pressure_wall_evaluation: str = WALL_CROSSING_DEFAULT_PRESSURE_EVALUATION
    split_dispatch: bool = True
    split_dispatch_explicit: bool = False
    split_order: str = "near_wall_first"

    def geometry_code(self) -> int:
        return WALL_CROSSING_GEOMETRY_CHOICES.index(self.geometry)

    def pressure_wall_code(self) -> int:
        return WALL_CROSSING_PRESSURE_WALL_CHOICES.index(self.pressure_wall)

    def pressure_wall_evaluation_code(self) -> int:
        return WALL_CROSSING_PRESSURE_EVALUATION_CHOICES.index(self.pressure_wall_evaluation)

    def viscous_extension_code(self) -> int:
        return WALL_CROSSING_EXTENSION_CHOICES.index(self.viscous_extension)

    def continuity_extension_code(self) -> int:
        return WALL_CROSSING_EXTENSION_CHOICES.index(self.continuity_extension)

    def velocity_gradient_active(self) -> bool:
        """The fused pass solves and stores grad v_i (either extension quadratic)."""
        return "quadratic" in (self.viscous_extension, self.continuity_extension)

    def continuity_wall_component_code(self) -> int:
        return WALL_CROSSING_CONTINUITY_WALL_COMPONENT_CHOICES.index(self.continuity_wall_component)

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


def box_pair_geometry(fluid_positions: np.ndarray, wall_positions: np.ndarray, box_lower, box_upper,
                      dimension: int, distance_floor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """numerics.wall_crossing.geometry pairwise: the geometry of P pairs (fluid particle i inside the box, wall
    particle j outside it), the shader's wall_crossing_pair_geometry row by row. float64.

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
        continuity: extrapolate | wall_velocity            (default extrapolate)
        distance_floor_factor: <float > 0>                 (default 0.25)
        box: {lower: [x, y, z], upper: [x, y, z]}          (default: from the fluid particles)
        geometry: pairwise | nearest_wall                  (default pairwise since 2026-11-19, nearest_wall
                                                            before; pairwise not with a quadratic extension)
        continuity_wall_component: full | normal           (default full; normal from 2026-11-18 to 11-19)
        pressure_wall: reflection | extension              (default reflection since 2026-11-19, extension
                                                            before; reflection needs geometry pairwise)
        pressure_wall_evaluation: summed | pairwise         (default summed; report section 27: pairwise gives
                                                            every wall neighbour p_j from G_p the fused pass
                                                            stored; the numbers change slightly)
        delta_wall: reflection | extension | rest_density | body_force
                                                           (default reflection since 2026-11-19, extension
                                                            before; reflection needs geometry pairwise;
                                                            rest_density and body_force are DIAGNOSTICS)
        split_dispatch: true | false                       (default true since report section 25, false before;
                                                            report section 22: the interior particles in a build
                                                            without the mode's code)
        split_order: near_wall_first | interior_first      (default near_wall_first since report section 25,
                                                            interior_first before; report section 24)
        viscous_extension: linear | quadratic              (default linear; quadratic needs geometry
                                                            nearest_wall, pressure_wall extension and delta_wall
                                                            extension, i.e. these three written in the case)
        continuity_extension: linear | quadratic           (default linear; quadratic needs viscous_extension
                                                            quadratic and continuity extrapolate, and what
                                                            viscous_extension quadratic needs)
    A 2-D case may give two components per corner. A combination that is not offered raises ValueError naming
    every key the case has to write (the defaults changed on 2026-11-19: a case that asks for a quadratic
    extension or for geometry nearest_wall alone now has to write the keys that were the defaults before)."""
    mapping = numerics_mapping.get("wall_crossing") or {}
    if not isinstance(mapping, dict):
        raise ValueError(f"numerics.wall_crossing: expected a mapping, got {mapping!r}")
    unknown = sorted(set(mapping) - {"continuity", "distance_floor_factor", "box", "delta_wall",
                                     "continuity_wall_component", "viscous_extension", "continuity_extension",
                                     "geometry", "pressure_wall", "pressure_wall_evaluation", "split_dispatch",
                                     "split_order"})
    if unknown:
        raise ValueError(f"numerics.wall_crossing: unknown keys {unknown} (known: continuity, "
                         "distance_floor_factor, box, delta_wall, continuity_wall_component, viscous_extension, "
                         "continuity_extension, geometry, pressure_wall, pressure_wall_evaluation, split_dispatch, "
                         "split_order)")
    split_order = str(mapping.get("split_order", "near_wall_first"))
    if split_order not in ("interior_first", "near_wall_first"):
        raise ValueError(f"numerics.wall_crossing.split_order: {split_order!r}, expected interior_first or "
                         "near_wall_first")
    split_dispatch_explicit = "split_dispatch" in mapping
    split_dispatch = mapping.get("split_dispatch", True)
    if not isinstance(split_dispatch, bool):
        raise ValueError(f"numerics.wall_crossing.split_dispatch: {split_dispatch!r}, expected true or false")
    geometry = str(mapping.get("geometry", WALL_CROSSING_DEFAULT_GEOMETRY))
    if geometry not in WALL_CROSSING_GEOMETRY_CHOICES:
        raise ValueError(f"numerics.wall_crossing.geometry: {geometry!r}, expected one of "
                         f"{', '.join(WALL_CROSSING_GEOMETRY_CHOICES)}")
    pressure_wall = str(mapping.get("pressure_wall", WALL_CROSSING_DEFAULT_PRESSURE_WALL))
    if pressure_wall not in WALL_CROSSING_PRESSURE_WALL_CHOICES:
        raise ValueError(f"numerics.wall_crossing.pressure_wall: {pressure_wall!r}, expected one of "
                         f"{', '.join(WALL_CROSSING_PRESSURE_WALL_CHOICES)}")
    pressure_wall_evaluation = str(mapping.get("pressure_wall_evaluation", WALL_CROSSING_DEFAULT_PRESSURE_EVALUATION))
    if pressure_wall_evaluation not in WALL_CROSSING_PRESSURE_EVALUATION_CHOICES:
        raise ValueError(f"numerics.wall_crossing.pressure_wall_evaluation: {pressure_wall_evaluation!r}, expected "
                         f"one of {', '.join(WALL_CROSSING_PRESSURE_EVALUATION_CHOICES)}")
    extensions = {}
    for key, default in (("viscous_extension", WALL_CROSSING_DEFAULT_VISCOUS_EXTENSION),
                         ("continuity_extension", WALL_CROSSING_DEFAULT_CONTINUITY_EXTENSION)):
        extensions[key] = str(mapping.get(key, default))
        if extensions[key] not in WALL_CROSSING_EXTENSION_CHOICES:
            raise ValueError(f"numerics.wall_crossing.{key}: {extensions[key]!r}, expected one of "
                             f"{', '.join(WALL_CROSSING_EXTENSION_CHOICES)}")
    continuity_wall_component = str(mapping.get("continuity_wall_component",
                                                WALL_CROSSING_DEFAULT_CONTINUITY_WALL_COMPONENT))
    if continuity_wall_component not in WALL_CROSSING_CONTINUITY_WALL_COMPONENT_CHOICES:
        raise ValueError(f"numerics.wall_crossing.continuity_wall_component: {continuity_wall_component!r}, "
                         f"expected one of {', '.join(WALL_CROSSING_CONTINUITY_WALL_COMPONENT_CHOICES)}")
    delta_wall = str(mapping.get("delta_wall", WALL_CROSSING_DEFAULT_DELTA_WALL))
    if delta_wall not in WALL_CROSSING_DELTA_WALL_CHOICES:
        raise ValueError(f"numerics.wall_crossing.delta_wall: {delta_wall!r}, expected one of "
                         f"{', '.join(WALL_CROSSING_DELTA_WALL_CHOICES)}")
    continuity = str(mapping.get("continuity", WALL_CROSSING_DEFAULT_CONTINUITY))
    if continuity not in WALL_CROSSING_CONTINUITY_CHOICES:
        raise ValueError(f"numerics.wall_crossing.continuity: {continuity!r}, expected one of "
                         f"{', '.join(WALL_CROSSING_CONTINUITY_CHOICES)}")
    check_wall_crossing_combination(mapping, geometry=geometry, pressure_wall=pressure_wall, delta_wall=delta_wall,
                                    continuity=continuity, **extensions)
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
                                delta_wall=delta_wall, continuity_wall_component=continuity_wall_component,
                                viscous_extension=extensions["viscous_extension"],
                                continuity_extension=extensions["continuity_extension"],
                                geometry=geometry, pressure_wall=pressure_wall,
                                pressure_wall_evaluation=pressure_wall_evaluation, split_dispatch=split_dispatch,
                                split_dispatch_explicit=split_dispatch_explicit, split_order=split_order)


def _value_text(mapping: dict, key: str, value: str) -> str:
    """'key: value' and where the value comes from (the case, or the default)."""
    return f"{key}: {value} ({'written in the case' if key in mapping else 'the default'})"


def check_wall_crossing_combination(mapping: dict, geometry: str, pressure_wall: str, delta_wall: str,
                                    continuity: str, viscous_extension: str, continuity_extension: str) -> None:
    """The combinations that are not offered, each with one ValueError that names every key the case has to
    write (and whether the offending value was written or is the default):
      - a quadratic extension (viscous_extension / continuity_extension) needs geometry nearest_wall,
        pressure_wall extension and delta_wall other than reflection (the reflections need the pairwise geometry,
        which the second-order continuation does not support); continuity_extension quadratic also needs
        viscous_extension quadratic and continuity extrapolate;
      - a reflection (pressure_wall / delta_wall) needs geometry pairwise: geometry nearest_wall without a
        quadratic extension needs pressure_wall extension and delta_wall other than reflection.
    With the defaults of 2026-11-19 (pairwise, reflection, reflection) a case that writes a quadratic extension
    alone, or geometry nearest_wall alone, falls under these."""
    quadratic = "quadratic" in (viscous_extension, continuity_extension)
    if quadratic:
        problems = []
        if continuity_extension == "quadratic":
            if viscous_extension != "quadratic":
                problems.append("continuity_extension quadratic is offered together with viscous_extension "
                                f"quadratic only ({_value_text(mapping, 'viscous_extension', viscous_extension)})")
            if continuity == "wall_velocity":
                problems.append("continuity_extension quadratic continues the velocity across the wall; it cannot "
                                "be combined with continuity wall_velocity")
        offending = []
        if geometry != "nearest_wall":
            offending.append(_value_text(mapping, "geometry", geometry))
        if pressure_wall == "reflection":
            offending.append(_value_text(mapping, "pressure_wall", pressure_wall))
        if delta_wall == "reflection":
            offending.append(_value_text(mapping, "delta_wall", delta_wall))
        if offending:
            requested = [f"{key} quadratic" for key, value in
                         (("viscous_extension", viscous_extension), ("continuity_extension", continuity_extension))
                         if value == "quadratic"]
            needed = ", ".join(f"{key}: {value}" for key, value in WALL_CROSSING_NEAREST_WALL_REQUIREMENTS)
            problems.append(f"{' and '.join(requested)} {'need' if len(requested) > 1 else 'needs'} {needed} written "
                            "in the case as well (the second-order continuation is not offered with the pairwise "
                            "geometry, and a reflection needs it; the defaults are geometry pairwise, pressure_wall "
                            "reflection and delta_wall reflection since 2026-11-19; the second-order runs of "
                            "2026-11-18 also had continuity_wall_component: normal); now " + "; ".join(offending))
        if problems:
            raise ValueError("numerics.wall_crossing: " + " | ".join(problems))
        return
    if geometry != "pairwise":
        # a reflection needs the wall particle's mirror point, which is the pairwise geometry's
        reflected = [(key, value) for key, value in (("pressure_wall", pressure_wall), ("delta_wall", delta_wall))
                     if value == "reflection"]
        if reflected:
            needed = " and ".join(f"{key}: extension" for key, _ in reflected)
            raise ValueError("numerics.wall_crossing: a reflection needs geometry: pairwise; with "
                             f"{_value_text(mapping, 'geometry', geometry)} write {needed} as well (delta_wall may "
                             "also be a diagnostic value; the defaults of pressure_wall and delta_wall are reflection "
                             "since 2026-11-19); now "
                             + "; ".join(_value_text(mapping, key, value) for key, value in reflected))


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
    """One line for the loader's log: the box (and where it came from), the floor, the continuity rule and,
    with the particles, the smallest fluid phi and the smallest / largest wall depth."""
    active = int(dimension)
    text = (f"wall_boundary crossing: box lower {list(settings.box_lower[:active])} upper "
            f"{list(settings.box_upper[:active])} "
            f"({'case.yaml' if settings.box_from_case else 'fluid bounding box + one particle radius'}), "
            f"distance floor {settings.distance_floor(particle_radius):.6g} m "
            f"(= {settings.distance_floor_factor:g} x spacing), continuity {settings.continuity}, "
            f"continuity_wall_component {settings.continuity_wall_component}, geometry {settings.geometry}, "
            f"pressure_wall {settings.pressure_wall}"
            + (f" (evaluated {settings.pressure_wall_evaluation})"
               if settings.pressure_wall_evaluation != WALL_CROSSING_DEFAULT_PRESSURE_EVALUATION else "") + ", "
            + ("DIAGNOSTIC " if settings.delta_wall in ("rest_density", "body_force") else "")
            + f"delta_wall {settings.delta_wall}")
    if settings.velocity_gradient_active():
        text += (f", viscous_extension {settings.viscous_extension}, "
                 f"continuity_extension {settings.continuity_extension}")
        if settings.continuity_extension == "quadratic":
            text += (" (the whole velocity difference of a wall neighbour: continuity_wall_component "
                     f"{settings.continuity_wall_component} only names the first-order rule a particle falls "
                     "back to)")
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
