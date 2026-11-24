"""
cavity3d_lines.py - claude/wall_store_report.md section 14 (round 8): the 3-D cavity's centre lines sampled from the
particles, online (box_run.py --lines-time) or offline (from a saved fluid state).

Lines (coordinates normalised on the crossing box: s = (x - wall) / L, L = the wall-to-wall length, the same box for
every wall mode; the lid is the face y = 1 and moves in +x):
  vertical    (0.5, s, 0.5)   u = velocity x   (Albensoeder & Kuhlmann 2005 table 5)
  horizontal  (s, 0.5, 0.5)   v = velocity y   (table 6)
  spanwise    (0.5, 0.5, s)   w = velocity z   (table 10's |w| extremum)
Each line is sampled at s = k / (8 (N + 1)), k = 0 .. 8 (N + 1), i.e. every dx / 8 along the physical line (L = (N +
1) dx), and at the benchmark's table points; the cavity centre is sampled once more for the pressure's zero.

Interpolation (the method of wall_crossing_report.md section 16, checked there on an analytic field, in 3-D): linear
moving least squares with the solver's Wendland C4 profile of support h, weight W(|p - x_j| / h) V_j, V_j = m_j /
rho_j, basis [1, (x - p) / h]; the value is the fit's constant term (exact for linear fields). Points with fewer
than 10 neighbours or a moment matrix of condition >= 1e8 fall back to the Shepard average (counted).
  velocity: the fluid particles and the wall particles (the case's static wall and lid particles with their stored
            velocity: 0 for the walls, the lid speed for the lid), as in section 16, so the support is never cut;
  pressure: the fluid particles only (a wall particle's stored pressure is not a field value in every wall mode);
            the linear fit is one-sided near a wall.
Outputs are float64 arrays; box_run stores them as float32.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

LINES = (("vertical", 1, 0), ("horizontal", 0, 1), ("spanwise", 2, 2))   # (name, axis along, velocity component)
CONDITION_LIMIT = 1.0e8
MINIMUM_NEIGHBOURS = 10


def wendland_c4(q: np.ndarray) -> np.ndarray:
    """The solver's Wendland C4 profile without its constant (it cancels in Shepard and MLS)."""
    one_minus_q = np.clip(1.0 - q, 0.0, None)
    return one_minus_q ** 6 * ((35.0 / 3.0) * q * q + 6.0 * q + 1.0)


def fine_coordinates(spacings: int) -> np.ndarray:
    """s = k / (8 (N + 1)): every dx / 8 along a line of length L = (N + 1) dx (the crossing box)."""
    intervals = 8 * (spacings + 1)
    return np.arange(intervals + 1, dtype=np.float64) / intervals


def line_points(lower: np.ndarray, upper: np.ndarray, axis: int, coordinates: np.ndarray) -> np.ndarray:
    """Physical points of the line along axis through the box centre at normalised coordinates."""
    centre = 0.5 * (lower + upper)
    points = np.tile(centre, (coordinates.size, 1))
    points[:, axis] = lower[axis] + coordinates * (upper[axis] - lower[axis])
    return points


def moving_least_squares(positions: np.ndarray, volumes: np.ndarray, values: np.ndarray, points: np.ndarray,
                         support: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear MLS of values (M_particles, C) at points (P, 3): (values (P, C), neighbours (P,), fallback (P,))."""
    tree = cKDTree(positions)
    neighbour_lists = tree.query_ball_point(points, r=support, workers=-1, return_sorted=False)
    lengths = np.fromiter((len(item) for item in neighbour_lists), dtype=np.int64, count=len(neighbour_lists))
    point_count, components = points.shape[0], values.shape[1]
    result = np.full((point_count, components), np.nan)
    if lengths.sum() == 0:
        return result, lengths, np.zeros(point_count, dtype=bool)
    particle_index = np.concatenate([np.asarray(item, dtype=np.int64) for item in neighbour_lists if len(item)])
    point_index = np.repeat(np.arange(point_count, dtype=np.int64), lengths)
    offset = (positions[particle_index] - points[point_index]) / support
    weight = wendland_c4(np.sqrt((offset ** 2).sum(axis=1))) * volumes[particle_index]
    basis = np.column_stack([np.ones(offset.shape[0]), offset])                      # (pairs, 4)
    moment = np.zeros((point_count, 4, 4))
    for row in range(4):
        for column in range(row, 4):
            entry = np.bincount(point_index, weights=weight * basis[:, row] * basis[:, column], minlength=point_count)
            moment[:, row, column] = entry
            moment[:, column, row] = entry
    right = np.zeros((point_count, 4, components))
    for row in range(4):
        for component in range(components):
            right[:, row, component] = np.bincount(point_index, weights=weight * basis[:, row]
                                                   * values[particle_index, component], minlength=point_count)
    weight_sum = moment[:, 0, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        shepard = right[:, 0, :] / weight_sum[:, None]
        condition = np.linalg.cond(moment)
    good = np.isfinite(condition) & (condition < CONDITION_LIMIT) & (lengths >= MINIMUM_NEIGHBOURS)
    if good.any():
        result[good] = np.linalg.solve(moment[good], right[good])[:, 0, :]
    fallback = ~good & (lengths > 0)
    result[fallback] = shepard[fallback]
    return result, lengths, fallback


def tube(positions: np.ndarray, centre: np.ndarray, axis: int, radius: float) -> np.ndarray:
    """Mask of the particles within radius of the line along axis through centre (a square tube)."""
    mask = np.ones(positions.shape[0], dtype=bool)
    for other in range(3):
        if other != axis:
            mask &= np.abs(positions[:, other] - centre[other]) <= radius
    return mask


class LineSampler:
    """Samples the three centre lines of one case (box, support, walls fixed) from fluid states."""

    def __init__(self, lower, upper, spacings: int, support: float, wall_positions: np.ndarray,
                 wall_velocities: np.ndarray, wall_volumes: np.ndarray, reference_points: dict | None = None):
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.support = float(support)
        self.fine = fine_coordinates(spacings)
        self.reference_points = {name: np.asarray(value, dtype=np.float64)
                                 for name, value in (reference_points or {}).items()}
        self.centre = 0.5 * (self.lower + self.upper)
        self.walls = (np.asarray(wall_positions, dtype=np.float64), np.asarray(wall_velocities, dtype=np.float64),
                      np.asarray(wall_volumes, dtype=np.float64))

    def coordinates(self, name: str) -> np.ndarray:
        """The sample coordinates of a line: the fine ones, then the line's reference points (if any)."""
        return np.concatenate([self.fine, self.reference_points.get(name, np.zeros(0))])

    def sample(self, state: dict) -> dict:
        """{line: {"velocity" (P, 3), "pressure" (P,), "neighbours" (P,), "fallback_velocity", "fallback_pressure"}}
        on the line's coordinates(), and "centre_pressure"."""
        fluid_positions = np.asarray(state["positions"], dtype=np.float64)
        fluid_volumes = np.asarray(state["masses"], dtype=np.float64) / np.asarray(state["densities"], dtype=np.float64)
        wall_positions, wall_velocities, wall_volumes = self.walls
        radius = 1.05 * self.support
        out = {}
        for name, axis, _component in LINES:
            fluid_mask = tube(fluid_positions, self.centre, axis, radius)
            wall_mask = tube(wall_positions, self.centre, axis, radius)
            points = line_points(self.lower, self.upper, axis, self.coordinates(name))
            positions = np.concatenate([fluid_positions[fluid_mask], wall_positions[wall_mask]])
            velocities = np.concatenate([np.asarray(state["velocities"], dtype=np.float64)[fluid_mask],
                                         wall_velocities[wall_mask]])
            volumes = np.concatenate([fluid_volumes[fluid_mask], wall_volumes[wall_mask]])
            velocity, neighbours, fallback_velocity = moving_least_squares(positions, volumes, velocities, points,
                                                                           self.support)
            pressure, _, fallback_pressure = moving_least_squares(
                fluid_positions[fluid_mask], fluid_volumes[fluid_mask],
                np.asarray(state["pressures"], dtype=np.float64)[fluid_mask][:, None], points, self.support)
            out[name] = {"velocity": velocity, "pressure": pressure[:, 0], "neighbours": neighbours,
                         "fallback_velocity": fallback_velocity, "fallback_pressure": fallback_pressure}
        centre_mask = np.all(np.abs(fluid_positions - self.centre) <= radius, axis=1)
        centre_pressure, _, _ = moving_least_squares(
            fluid_positions[centre_mask], fluid_volumes[centre_mask],
            np.asarray(state["pressures"], dtype=np.float64)[centre_mask][:, None], self.centre[None, :], self.support)
        out["centre_pressure"] = float(centre_pressure[0, 0])
        return out


def wall_particles_of_case(case_yaml) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The case's boundary particles as its geometry lists them (every wall OBJ of case.yaml, all layers), whatever
    the wall mode keeps of them (crossing's wall store keeps the layers within h only, and not in case.initial):
    (positions, velocities = the material's initial velocity, volumes = the material's calibrated volume)."""
    import pathlib
    import yaml
    from experiment.v7.utils.case_loader_v7 import _parse_obj_vertices, load_case_v7
    from experiment.v7.utils.case_v7 import KIND_BOUNDARY
    case_yaml = pathlib.Path(case_yaml).resolve()
    data = yaml.safe_load(case_yaml.read_text(encoding="utf-8"))
    library = yaml.safe_load((case_yaml.parent / data["material_library"]).resolve().read_text(encoding="utf-8"))
    case = load_case_v7(case_yaml)
    volume = float(case.materials[0].volume)
    positions, velocities = [], []
    for entry in data["geometry"]["particles"]:
        spec = library[entry["material"]]
        if spec["kind"] != "boundary":
            continue
        vertices = np.asarray(_parse_obj_vertices((case_yaml.parent / entry["file"]).resolve()), dtype=np.float64)
        positions.append(vertices)
        velocities.append(np.broadcast_to(np.asarray(spec.get("initial_velocity", [0.0, 0.0, 0.0]),
                                                     dtype=np.float64), vertices.shape).copy())
    assert all(material.kind != KIND_BOUNDARY or abs(float(material.volume) - volume) < 1e-15
               for material in case.materials)
    positions = np.concatenate(positions)
    return positions, np.concatenate(velocities), np.full(positions.shape[0], volume)


def sampler_of_case(case, lower, upper, reference_points: dict | None, case_yaml) -> LineSampler:
    """A LineSampler for a loaded case (case_loader_v7): N from the box and the fluid radius, the walls from the
    case's geometry (wall_particles_of_case of case_yaml, the case's file)."""
    radius = float(next(material for material in case.materials if material.kind == 0).radius)
    spacings = int(round((float(upper[0]) - float(lower[0])) / (2.0 * radius))) - 1
    wall_positions, wall_velocities, wall_volumes = wall_particles_of_case(case_yaml)
    return LineSampler(lower, upper, spacings, float(case.physics.smoothing_length), wall_positions,
                       wall_velocities, wall_volumes, reference_points)


class LineRecorder:
    """Accumulates the samples of a run and writes them to one npz (rewritten at every sample, atomically)."""

    def __init__(self, path, sampler: LineSampler, existing_until_step: int | None = None):
        self.path = path
        self.sampler = sampler
        self.rows = []
        if existing_until_step is not None and path.exists():
            with np.load(path) as archive:
                keep = archive["step"] <= existing_until_step
                for index in np.flatnonzero(keep):
                    self.rows.append({key: archive[key][index] for key in archive.files
                                      if key.startswith(("step", "time", "centre_pressure"))
                                      or key.endswith(("_velocity", "_pressure", "_fallback_velocity",
                                                       "_fallback_pressure"))})

    def add(self, step: int, time_value: float, state: dict) -> dict:
        """Sample state, append the row, rewrite the file; returns the sample (LineSampler.sample)."""
        sampled = self.sampler.sample(state)
        row = {"step": np.int64(step), "time": np.float64(time_value),
               "centre_pressure": np.float64(sampled["centre_pressure"])}
        for name, _axis, _component in LINES:
            row[f"{name}_velocity"] = sampled[name]["velocity"].astype(np.float32)
            row[f"{name}_pressure"] = sampled[name]["pressure"].astype(np.float32)
            row[f"{name}_fallback_velocity"] = np.int64(sampled[name]["fallback_velocity"].sum())
            row[f"{name}_fallback_pressure"] = np.int64(sampled[name]["fallback_pressure"].sum())
        self.rows.append(row)
        self.write()
        return sampled

    def write(self) -> None:
        arrays = {key: np.stack([row[key] for row in self.rows]) for key in self.rows[0]}
        for name, _axis, _component in LINES:
            arrays[f"{name}_coordinate"] = self.sampler.coordinates(name)
        arrays["fine_count"] = np.int64(self.sampler.fine.size)
        arrays["box_lower"] = self.sampler.lower
        arrays["box_upper"] = self.sampler.upper
        arrays["support"] = np.float64(self.sampler.support)
        temporary = self.path.with_name(self.path.stem + ".partial.npz")
        np.savez(temporary, **arrays)
        temporary.replace(self.path)


def read_acceleration_of_chain(simulators: list, slab_cases: list) -> dict:
    """The alive fluid particles' acceleration buffer with their positions (the same rows as
    hydrostatic.read_fluid_state): {"positions" (M, 3), "accelerations" (M, 3)}."""
    from experiment.v7.utils.case_v7 import KIND_FLUID
    positions, accelerations = [], []
    for simulator, case in zip(simulators, slab_cases):
        raw = simulator.readback_buffers_batch(["position_voxel_id", "velocity_mass", "material", "acceleration"],
                                               density="stored")
        capacity = case.capacities.total_pool_capacity()
        first = simulator.own_first_pid()
        rows = slice(first, first + case.capacities.own_pool_size)
        position = np.frombuffer(raw["position_voxel_id"], dtype=np.float32)[:capacity * 4].reshape(capacity, 4)[rows]
        velocity = np.frombuffer(raw["velocity_mass"], dtype=np.float32)[:capacity * 4].reshape(capacity, 4)[rows]
        acceleration = np.frombuffer(raw["acceleration"], dtype=np.float32)[:capacity * 4].reshape(capacity, 4)[rows]
        material = np.frombuffer(raw["material"], dtype=np.uint32)[:capacity][rows]
        kinds = np.asarray([entry.kind for entry in case.materials])
        alive = (velocity[:, 3] > 0.0) & (position[:, 3] > 0.5)
        fluid = alive & (kinds[np.minimum(material, kinds.size - 1)] == KIND_FLUID)
        positions.append(position[fluid, 0:3].astype(np.float64))
        accelerations.append(acceleration[fluid, 0:3].astype(np.float64))
    return {"positions": np.concatenate(positions), "accelerations": np.concatenate(accelerations)}


def wall_count_within(positions: np.ndarray, lower: np.ndarray, upper: np.ndarray, distance: float) -> np.ndarray:
    """How many of the six box faces lie within distance of each point (0 interior, 1 face, 2 edge, 3 corner)."""
    near_lower = (positions - lower) < distance
    near_upper = (upper - positions) < distance
    return (near_lower | near_upper).sum(axis=1)


def pair_distance_minima(fluid_positions: np.ndarray, wall_positions: np.ndarray) -> tuple[float, float]:
    """(smallest fluid-fluid distance, smallest fluid-to-wall-particle distance)."""
    tree = cKDTree(fluid_positions)
    distance, _ = tree.query(fluid_positions, k=2, workers=-1)
    fluid_fluid = float(distance[:, 1].min())
    wall_tree = cKDTree(wall_positions)
    wall_distance, _ = wall_tree.query(fluid_positions, k=1, workers=-1)
    return fluid_fluid, float(wall_distance.min())
