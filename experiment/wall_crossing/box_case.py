"""
box_case.py - write a small axis-aligned box case (2-D or 3-D) for the wall_boundary tests: a lattice of fluid
particles filling the box, wall particles on a lattice outside it (the +y block above the fluid footprint is a
second boundary material, the "lid", which may move), a material library of its own and a case.yaml in the
schema case_loader_v7 reads.

    from experiment.wall_crossing.box_case import BoxCaseSpecification, write_box_case
    case_path = write_box_case(directory, BoxCaseSpecification(dimension=2, fluid_counts=(50, 40), spacing=0.02))

The fluid particle k of an axis sits at lower + (k + 1/2) spacing, so the wall planes are the box faces and the
first wall layer is half a spacing behind them. Options used by the tests:
  jitter_fraction      uniform displacement of every fluid particle (fraction of the spacing, per active axis);
  near_wall_count      that many fluid particles of the outermost rows are moved to near_wall_fraction x spacing
                       from their wall (below the distance floor of wall_boundary crossing);
  explicit_box         write numerics.wall_crossing.box (needed with jitter: the default box is the fluid
                       particles' bounding box + one radius, which jitter moves).
Nothing here imports Vulkan.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BoxCaseSpecification:
    dimension: int = 2
    fluid_counts: tuple = (50, 40)              # fluid particles per active axis
    spacing: float = 0.02                       # particle spacing dx
    lower: tuple = (0.10, -0.20, 0.05)          # box lower corner (z ignored in 2-D)
    support_ratio: float = 5.0                  # h / dx (support radius)
    wall_layers: Optional[int] = None           # default ceil(h / dx) + 1
    speed_of_sound: float = 10.0
    power: float = 7.0
    cfl: float = 0.15
    gravity: tuple = (0.0, 0.0, 0.0)
    rest_density: float = 1000.0
    viscosity: float = 1.0e-2
    lid_velocity: tuple = (1.0, 0.0, 0.0)
    wall_boundary: str = "crossing"             # simple | adami | crossing
    continuity: str = "extrapolate"             # crossing only
    distance_floor_factor: float = 0.25         # crossing only
    delta_wall: str = "extension"               # crossing only, diagnostic: extension | rest_density | body_force
    continuity_wall_component: str = "normal"   # crossing only: normal (the solver's default) | full
    viscous_extension: str = "linear"           # crossing only: linear (the solver's default) | quadratic
    continuity_extension: str = "linear"        # crossing only: linear (the solver's default) | quadratic
    explicit_box: bool = False
    jitter_fraction: float = 0.0
    near_wall_count: int = 0
    near_wall_fraction: float = 0.1
    seed: int = 20261117
    delta_coefficient: float = 0.1
    regularization_xi: float = 0.001
    epsilon_squared_factor: float = 0.0025
    use_density_diffusion: bool = True
    use_pst: bool = True
    pst_main: float = 0.1
    pst_anti: float = 0.0005
    defrag_cadence: int = 1000
    max_per_voxel: Optional[int] = None
    max_incoming: int = 32
    workgroup: int = 128
    pool_factor: float = 1.15
    comment: str = ""
    extra_numerics: dict = field(default_factory=dict)

    def active_counts(self) -> tuple:
        return tuple(int(count) for count in self.fluid_counts[:self.dimension])

    def support_radius(self) -> float:
        return self.support_ratio * self.spacing

    def box_lower(self) -> np.ndarray:
        lower = np.zeros(3)
        lower[:self.dimension] = np.asarray(self.lower[:self.dimension], dtype=np.float64)
        return lower

    def box_upper(self) -> np.ndarray:
        upper = np.zeros(3)
        upper[:self.dimension] = (self.box_lower()[:self.dimension]
                                  + np.asarray(self.active_counts(), dtype=np.float64) * self.spacing)
        return upper

    def layers(self) -> int:
        return int(self.wall_layers) if self.wall_layers is not None else int(np.ceil(self.support_ratio)) + 1


def _lattice_cells(specification: BoxCaseSpecification) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integer cell indices (N, dimension) of the fluid, the static wall and the lid."""
    counts = specification.active_counts()
    layers = specification.layers()
    axes = [np.arange(-layers, count + layers) for count in counts]
    cells = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, specification.dimension)
    inside = np.ones(cells.shape[0], dtype=bool)
    for axis, count in enumerate(counts):
        inside &= (cells[:, axis] >= 0) & (cells[:, axis] < count)
    # the lid: above the +y face, within the fluid footprint on every other axis
    lid = cells[:, 1] >= counts[1]
    for axis, count in enumerate(counts):
        if axis != 1:
            lid &= (cells[:, axis] >= 0) & (cells[:, axis] < count)
    return cells[inside], cells[~inside & ~lid], cells[lid]


def box_particles(specification: BoxCaseSpecification) -> dict:
    """Positions (float64, (N, 3)) of the fluid, the wall and the lid, in that order, with the options applied."""
    dimension, spacing = specification.dimension, specification.spacing
    lower = specification.box_lower()
    upper = specification.box_upper()
    fluid_cells, wall_cells, lid_cells = _lattice_cells(specification)

    def to_positions(cells: np.ndarray) -> np.ndarray:
        positions = np.zeros((cells.shape[0], 3))
        positions[:, :dimension] = lower[:dimension] + (cells + 0.5) * spacing
        return positions

    fluid = to_positions(fluid_cells)
    generator = np.random.default_rng(specification.seed)
    if specification.jitter_fraction > 0.0:
        fluid[:, :dimension] += generator.uniform(
            -specification.jitter_fraction, specification.jitter_fraction,
            size=(fluid.shape[0], dimension)) * spacing
    if specification.near_wall_count > 0:
        counts = np.asarray(specification.active_counts())
        on_face = np.flatnonzero(((fluid_cells == 0) | (fluid_cells == counts - 1)).any(axis=1))
        chosen = generator.choice(on_face, size=min(specification.near_wall_count, on_face.size), replace=False)
        for row in chosen:
            for axis in range(dimension):
                if fluid_cells[row, axis] == 0:
                    fluid[row, axis] = lower[axis] + specification.near_wall_fraction * spacing
                    break
                if fluid_cells[row, axis] == counts[axis] - 1:
                    fluid[row, axis] = upper[axis] - specification.near_wall_fraction * spacing
                    break
    # every fluid particle stays strictly inside the box
    margin = 0.02 * spacing
    fluid[:, :dimension] = np.clip(fluid[:, :dimension], lower[:dimension] + margin, upper[:dimension] - margin)
    return {"fluid": fluid, "wall": to_positions(wall_cells), "lid": to_positions(lid_cells),
            "box_lower": lower, "box_upper": upper}


def _write_obj(path: pathlib.Path, positions: np.ndarray) -> None:
    lines = [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in np.asarray(positions, dtype=np.float32)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_box_case(directory: pathlib.Path, specification: BoxCaseSpecification) -> pathlib.Path:
    """Write frame.obj, domain.obj, wall.obj, wall_top.obj, materials.yaml and case.yaml into directory
    (created; an existing case.yaml there is overwritten: the directory is the caller's own output).
    Returns the path of case.yaml."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    particles = box_particles(specification)
    dimension, spacing = specification.dimension, specification.spacing
    support = specification.support_radius()
    everything = np.concatenate([particles["fluid"], particles["wall"], particles["lid"]])
    frame_lower, frame_upper = everything.min(axis=0), everything.max(axis=0)
    corners = np.array([[x, y, z] for x in (frame_lower[0], frame_upper[0])
                        for y in (frame_lower[1], frame_upper[1])
                        for z in (frame_lower[2], frame_upper[2])])
    _write_obj(directory / "frame.obj", corners)
    _write_obj(directory / "domain.obj", particles["fluid"])
    _write_obj(directory / "wall.obj", particles["wall"])
    _write_obj(directory / "wall_top.obj", particles["lid"])

    lid_velocity = [float(component) for component in specification.lid_velocity]
    (directory / "materials.yaml").write_text(
        "schema_version: 1\n"
        "# material library of a wall_boundary test box (experiment/wall_crossing/box_case.py)\n"
        f"box_fluid:\n  kind: fluid\n  rest_density: {specification.rest_density!r}\n"
        f"  viscosity: {specification.viscosity!r}\n"
        f"box_wall:\n  kind: boundary\n  rest_density: {specification.rest_density!r}\n"
        f"  viscosity: {specification.viscosity!r}\n"
        f"box_lid:\n  kind: boundary\n  rest_density: {specification.rest_density!r}\n"
        f"  viscosity: {specification.viscosity!r}\n  initial_velocity: {lid_velocity!r}\n",
        encoding="utf-8", newline="\n")

    total = everything.shape[0]
    pool_size = int(np.ceil(total * specification.pool_factor / 64.0)) * 64
    nominal_per_voxel = specification.support_ratio ** dimension
    max_per_voxel = (int(specification.max_per_voxel) if specification.max_per_voxel is not None
                     else int(np.ceil(2.5 * nominal_per_voxel / 32.0)) * 32)
    gravity = [float(component) for component in specification.gravity]
    numerics_lines = [
        f"  use_density_diffusion: {'true' if specification.use_density_diffusion else 'false'}",
        f"  delta_coefficient: {specification.delta_coefficient!r}",
        "  use_kcg_correction: true",
        "  regularization:",
        f"    xi: {specification.regularization_xi!r}",
        "    det_threshold: 1.0e-4",
        "    frobenius_max: 10.0",
        f"  epsilon_squared_factor: {specification.epsilon_squared_factor!r}",
        f"  use_pst: {'true' if specification.use_pst else 'false'}",
        f"  pst_main: {specification.pst_main!r}",
        f"  pst_anti: {specification.pst_anti!r}",
        f"  defrag_cadence: {int(specification.defrag_cadence)}",
        f"  wall_boundary: {specification.wall_boundary}",
    ]
    if specification.wall_boundary == "crossing":
        numerics_lines += ["  wall_crossing:",
                           f"    continuity: {specification.continuity}",
                           f"    distance_floor_factor: {specification.distance_floor_factor!r}"]
        if specification.delta_wall != "extension":
            numerics_lines += [f"    delta_wall: {specification.delta_wall}"]
        # written only when it is not the solver's default, so that a default case exercises the default
        from experiment.v7.utils.wall_crossing_v7 import WALL_CROSSING_DEFAULT_CONTINUITY_WALL_COMPONENT
        if specification.continuity_wall_component != WALL_CROSSING_DEFAULT_CONTINUITY_WALL_COMPONENT:
            numerics_lines += [f"    continuity_wall_component: {specification.continuity_wall_component}"]
        for key in ("viscous_extension", "continuity_extension"):
            if getattr(specification, key) != "linear":
                numerics_lines += [f"    {key}: {getattr(specification, key)}"]
        if specification.explicit_box:
            box_lower = [float(component) for component in particles["box_lower"][:dimension]]
            box_upper = [float(component) for component in particles["box_upper"][:dimension]]
            numerics_lines += [f"    box: {{lower: {box_lower!r}, upper: {box_upper!r}}}"]
    for key, value in specification.extra_numerics.items():
        numerics_lines.append(f"  {key}: {value}")
    counts_text = " x ".join(str(count) for count in specification.active_counts())
    case_text = "\n".join([
        "schema_version: 2",
        "",
        "# GENERATED by experiment/wall_crossing/box_case.py: an axis-aligned box of fluid on a lattice",
        f"# ({counts_text} fluid particles, spacing {spacing!r}, h / dx = {specification.support_ratio!r}), walls on the",
        f"# lattice outside it ({specification.layers()} layers); the +y block above the fluid footprint is box_lid.",
        f"# box lower {particles['box_lower'][:dimension].tolist()} upper {particles['box_upper'][:dimension].tolist()}; "
        f"jitter {specification.jitter_fraction!r} dx, {specification.near_wall_count} particle(s) moved to "
        f"{specification.near_wall_fraction!r} dx from a wall, seed {specification.seed}.",
        *([f"# {specification.comment}"] if specification.comment else []),
        "",
        "time:",
        "  total: null",
        "  max_steps: null",
        "  output_cadence: null",
        "",
        "physics:",
        f"  dimension: {dimension}",
        f"  h: {support!r}",
        f"  particle_radius: {0.5 * spacing!r}",
        "  lattice: grid",
        "  calibrate_volume: true",
        f"  speed_of_sound: {specification.speed_of_sound!r}",
        f"  power: {specification.power!r}",
        f"  cfl: {specification.cfl!r}",
        f"  gravity: {gravity!r}",
        "",
        "numerics:",
        *numerics_lines,
        "",
        "capacities:",
        f"  pool_size: {pool_size}",
        f"  max_per_voxel: {max_per_voxel}",
        f"  max_incoming: {int(specification.max_incoming)}",
        f"  workgroup: {int(specification.workgroup)}",
        "",
        "material_library: materials.yaml",
        "",
        "geometry:",
        "  frame: frame.obj",
        "  particles:",
        "    - {file: domain.obj,   material: box_fluid}",
        "    - {file: wall.obj,     material: box_wall}",
        "    - {file: wall_top.obj, material: box_lid}",
        "",
    ])
    case_path = directory / "case.yaml"
    case_path.write_text(case_text, encoding="utf-8", newline="\n")
    return case_path
