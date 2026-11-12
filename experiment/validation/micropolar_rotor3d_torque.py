"""
micropolar_rotor3d_torque.py - torque on the rotor of the 3-D case from the saved states of micropolar_rotor3d.py
run, on the host in float64: the two readings of Stage 4 (section 13.3) in three dimensions.

Pair terms of force.comp between a fluid particle i and a neighbour j (a wall neighbour enters with the dummy values
of the wall pass, read back from the GPU; G_ij = L_i grad W_ij with the solver's own L_i, read back too):

    a_ij     = - V_j (p_j + p_i) / rho_i G_ij                               pressure, as an exchange between i and j
               + 2 nu V_j (v_i - v~_j) (r_ij . G_ij) / (r_ij^2 + eps^2)     viscous, viscous_form laplacian
    Theta alpha_ij = 2 xi V_j (w_i - w~_j) (r_ij . G_ij) / (r_ij^2 + eps^2)  xi lap w

wall form    T = sum over the fluid within h of the rotor and its rotor neighbours of
             [ r_mid x m a_ij + m Theta alpha_ij ]_z, in units of the mass rho0 dx^3 a particle stands for, plus the
             wall term mu * (integral of (n x v_wall)_z over the rotor surface) = 2 mu Omega V_rotor that turns the
             stress (mu + kappa) grad v of the laplacian form into the symmetric one (w = 0 on the wall).
circle form  the same exchange terms across the closed surface at distance h around the rotor (the fluid beyond
             it on the fluid within it), plus the field terms on that surface, - mu * integral of (n x v)_z
             + kappa * integral of [n_z (r . w) - w_z (r . n)], and the advective flux of angular momentum.
             The rotor acts on the fluid with minus the sum (steady flow).

  sums    Both readings for every saved state of a run, split into the lateral surface and the end faces; the
          wall form of the housing where its band was saved; the checks: the wall pass recomputed on the host
          against the GPU's dummy values (separately at the edges of the rotor), and the full acceleration and
          angular acceleration of the fluid in a wedge against the GPU's buffers.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.validation.micropolar_rotor3d import DENSITY, housing_distance, rotor_distance   # noqa: E402
from experiment.validation.viscous_operators import wendland_c4                                   # noqa: E402

SECTORS = 12                    # the pair sums are taken sector by sector (memory)
ANGLE_BINS = 36
CAP_BIN_SPACINGS = 4.0
EDGE_BINS = 3


def parameters_of(run: dict) -> dict:
    meta = run["rotor3d"]
    return {"support": meta["smoothing_length"], "spacing": meta["spacing"], "viscosity": meta["design_viscosity"],
            "coupling": meta["design_coupling"], "viscosity_rotation": meta["design_viscosity_rotation"],
            "micro_inertia": meta["design_micro_inertia"], "micropolar": bool(meta["micropolar"]),
            "epsilon_squared": float(meta["epsilon_squared_factor"]) * meta["smoothing_length"] ** 2,
            "rotor_radius": meta["rotor_radius"], "rotor_half_length": meta["rotor_half_length"],
            "housing_radius": meta["housing_radius"], "housing_half_height": meta["housing_half_height"],
            "rotation_rate": meta["rotation_rate"], "end_rotation_rate": meta["end_rotation_rate"]}


def _correction(packed: np.ndarray) -> np.ndarray:
    """(count, 3, 3) from the packed symmetric inverse (m00, m11, m22, m01, m02, m12)."""
    matrix = np.empty((packed.shape[0], 3, 3))
    matrix[:, 0, 0], matrix[:, 1, 1], matrix[:, 2, 2] = packed[:, 0], packed[:, 1], packed[:, 2]
    matrix[:, 0, 1] = matrix[:, 1, 0] = packed[:, 3]
    matrix[:, 0, 2] = matrix[:, 2, 0] = packed[:, 4]
    matrix[:, 1, 2] = matrix[:, 2, 1] = packed[:, 5]
    return matrix


def _pairs(first_position, second_position, support):
    """(first index, second index, offset r_i - r_j, distance) of the pairs closer than the support."""
    from scipy.spatial import cKDTree
    matrix = cKDTree(first_position).sparse_distance_matrix(cKDTree(second_position), support,
                                                           output_type="coo_matrix")
    first, second, distance = matrix.row, matrix.col, matrix.data
    keep = distance > 1e-12
    first, second, distance = first[keep], second[keep], distance[keep]
    return first, second, first_position[first] - second_position[second], distance


class State:
    def __init__(self, state: dict, parameters: dict) -> None:
        self.parameters = parameters
        self.position = state["position"].astype(np.float64)
        self.velocity = state["velocity"].astype(np.float64)
        self.density = state["density"].astype(np.float64)
        self.pressure = state["pressure"].astype(np.float64)
        self.kind = state["kind"]
        self.mass = float(state["mass"])
        self.count = self.position.shape[0]
        self.is_fluid = self.kind == 0
        micropolar = parameters["micropolar"]
        self.spin = state["spin"].astype(np.float64) if micropolar else np.zeros((self.count, 3))
        dummy_spin = state["dummy_spin"].astype(np.float64) if micropolar else np.zeros((self.count, 3))
        # neighbour values as force.comp reads them: a wall neighbour enters with its dummy values
        self.neighbour_velocity = np.where(self.is_fluid[:, None], self.velocity,
                                           state["dummy_velocity"].astype(np.float64))
        self.neighbour_spin = np.where(self.is_fluid[:, None], self.spin, dummy_spin)
        self.correction = _correction(state["correction_inverse"].astype(np.float64))
        self.acceleration = state["acceleration"].astype(np.float64)
        self.angular_acceleration = (state["angular_acceleration"].astype(np.float64) if micropolar
                                     else np.zeros((self.count, 3)))
        self.dummy_velocity = state["dummy_velocity"].astype(np.float64)
        self.dummy_spin = dummy_spin
        self.dummy_weight = state["dummy_weight"].astype(np.float64)
        self.radius = np.hypot(self.position[:, 0], self.position[:, 1])
        self.to_rotor = rotor_distance(self.radius, self.position[:, 2], parameters["rotor_radius"],
                                       parameters["rotor_half_length"])
        self.to_housing = housing_distance(self.radius, self.position[:, 2], parameters["housing_radius"],
                                           parameters["housing_half_height"])
        self.sector = np.minimum((np.arctan2(self.position[:, 1], self.position[:, 0]) % (2.0 * np.pi)
                                  / (2.0 * np.pi) * SECTORS).astype(np.int64), SECTORS - 1)

    def pair_terms(self, first_index, second_index, offset, distance) -> dict:
        """The exchange terms of the pairs (first = fluid particle i, second = neighbour j), per unit mass of i:
        force terms (count, 3) and the couple Theta alpha (count, 3); also the kernel and the corrected gradient."""
        parameters = self.parameters
        kernel, derivative, _second = wendland_c4(distance, parameters["support"], 3)
        gradient = (derivative / distance)[:, None] * offset
        corrected = np.einsum("pab,pb->pa", self.correction[first_index], gradient)
        volume = self.mass / self.density[second_index]
        radial = (offset * corrected).sum(axis=1) / (distance ** 2 + parameters["epsilon_squared"])
        velocity_difference = self.velocity[first_index] - self.neighbour_velocity[second_index]
        spin_difference = self.spin[first_index] - self.neighbour_spin[second_index]
        return {
            "kernel": kernel, "corrected": corrected, "volume": volume, "radial": radial,
            "velocity_difference": velocity_difference, "spin_difference": spin_difference,
            "pressure_flux": -(volume * (self.pressure[second_index] + self.pressure[first_index])
                               / self.density[first_index])[:, None] * corrected,
            "viscous": (2.0 * parameters["viscosity"] * volume * radial)[:, None] * velocity_difference,
            "spin_diffusion": (2.0 * parameters["viscosity_rotation"] * volume * radial)[:, None] * spin_difference,
        }


def _exchange(state: State, first_global, second_global, offset, terms, selected) -> dict:
    """Force and torque about z of the selected pairs: exchange terms, arm at the pair midpoint, times the mass."""
    middle = state.position[first_global[selected]] - 0.5 * offset[selected]
    result = {"pairs": int(selected.sum()), "force": {}, "torque": {}}
    for name in ("pressure_flux", "viscous"):
        force = state.mass * terms[name][selected]
        result["force"][name] = force.sum(axis=0)
        result["torque"][name] = float((middle[:, 0] * force[:, 1] - middle[:, 1] * force[:, 0]).sum())
    result["torque"]["spin_diffusion"] = float(state.mass * terms["spin_diffusion"][selected, 2].sum())
    result["force_total"] = result["force"]["pressure_flux"] + result["force"]["viscous"]
    result["torque_total"] = float(sum(result["torque"].values()))
    return result


def _add(total: dict, part: dict) -> dict:
    if not total:
        return part
    for key, value in part.items():
        if isinstance(value, dict):
            _add(total[key], value)
        else:
            total[key] = total[key] + value
    return total


def wall_and_circle_sums(state: State) -> dict:
    """The exchange sums: rotor wall on the fluid (by lateral surface and end faces), the fluid beyond the surface
    at distance h on the fluid within it, and the housing on the fluid where the housing band was saved."""
    parameters = state.parameters
    support, spacing = parameters["support"], parameters["spacing"]
    result = {"rotor_lateral": {}, "rotor_end": {}, "circle": {}, "housing_lateral": {}, "housing_end": {}}
    inner = state.is_fluid & (state.to_rotor < support)
    outer = np.nonzero(state.is_fluid & (state.to_rotor >= support) & (state.to_rotor < 2.0 * support + spacing))[0]
    rotor_wall = np.nonzero((state.kind == 1) | (state.kind == 2))[0]
    near_housing = state.is_fluid & (state.to_housing < support)
    housing_wall = np.nonzero(state.kind == 3)[0]
    has_housing = housing_wall.size > 0
    for sector in range(SECTORS):
        chosen = np.nonzero(inner & (state.sector == sector))[0]
        if chosen.size:
            first, second, offset, distance = _pairs(state.position[chosen], state.position[rotor_wall], support)
            first_global, second_global = chosen[first], rotor_wall[second]
            terms = state.pair_terms(first_global, second_global, offset, distance)
            lateral = state.kind[second_global] == 1
            result["rotor_lateral"] = _add(result["rotor_lateral"], _exchange(state, first_global, second_global,
                                                                             offset, terms, lateral))
            result["rotor_end"] = _add(result["rotor_end"], _exchange(state, first_global, second_global, offset,
                                                                     terms, ~lateral))
            first, second, offset, distance = _pairs(state.position[chosen], state.position[outer], support)
            first_global, second_global = chosen[first], outer[second]
            terms = state.pair_terms(first_global, second_global, offset, distance)
            result["circle"] = _add(result["circle"], _exchange(state, first_global, second_global, offset, terms,
                                                                np.ones(first.size, dtype=bool)))
        if has_housing:
            chosen = np.nonzero(near_housing & (state.sector == sector))[0]
            first, second, offset, distance = _pairs(state.position[chosen], state.position[housing_wall], support)
            first_global, second_global = chosen[first], housing_wall[second]
            terms = state.pair_terms(first_global, second_global, offset, distance)
            # a housing particle belongs to the end faces where it lies beyond them
            end = np.abs(state.position[second_global, 2]) >= parameters["housing_half_height"]
            result["housing_lateral"] = _add(result["housing_lateral"], _exchange(state, first_global, second_global,
                                                                                 offset, terms, ~end))
            result["housing_end"] = _add(result["housing_end"], _exchange(state, first_global, second_global, offset,
                                                                         terms, end))
    return {key: value for key, value in result.items() if value}


def surface_terms(state: State) -> dict:
    """The field terms on the closed surface at distance h around the rotor: per patch (angle bins on the lateral
    part, radial bins on the two caps, bins of the edge angle on the two rounded edges) a quadratic in the wall
    distance fitted to the fluid within two spacings of the surface, evaluated on it, times the area of the patch.
    vorticity = integral of (n x v)_z, spin = integral of [n_z (r . w) - w_z (r . n)], advection = rho0 times the
    integral of (r x v)_z (v . n - mean v . n); n away from the rotor."""
    parameters = state.parameters
    support, spacing = parameters["support"], parameters["spacing"]
    radius_rotor, half_length = parameters["rotor_radius"], parameters["rotor_half_length"]
    shell = np.nonzero(state.is_fluid & (np.abs(state.to_rotor - support) < 2.0 * spacing))[0]
    position, velocity, spin = state.position[shell], state.velocity[shell], state.spin[shell]
    radius, height = state.radius[shell], position[:, 2]
    unit_radial = position[:, :2] / radius[:, None]
    beyond_radial = np.maximum(radius - radius_rotor, 0.0)
    beyond_axial = np.maximum(np.abs(height) - half_length, 0.0)
    norm = np.hypot(beyond_radial, beyond_axial)
    normal = np.column_stack([unit_radial * (beyond_radial / norm)[:, None], np.sign(height) * beyond_axial / norm])
    coordinate = (state.to_rotor[shell] - support) / spacing
    # patches
    lateral = (np.abs(height) <= half_length)
    cap = (radius <= radius_rotor) & ~lateral
    edge = ~lateral & ~cap
    top = height > 0.0
    angle_index = np.minimum((np.arctan2(position[:, 1], position[:, 0]) % (2.0 * np.pi) / (2.0 * np.pi)
                              * ANGLE_BINS).astype(np.int64), ANGLE_BINS - 1)
    cap_width = CAP_BIN_SPACINGS * spacing
    cap_bins = int(np.ceil(radius_rotor / cap_width - 1.0e-9))
    cap_index = np.minimum((radius / cap_width).astype(np.int64), cap_bins - 1)
    edge_angle = np.arctan2(beyond_axial, beyond_radial)                         # 0 at the lateral side
    edge_index = np.minimum((edge_angle / (0.5 * np.pi) * EDGE_BINS).astype(np.int64), EDGE_BINS - 1)
    patch = np.where(lateral, angle_index,
                     np.where(cap, ANGLE_BINS + cap_index + cap_bins * top,
                              ANGLE_BINS + 2 * cap_bins + edge_index + EDGE_BINS * top))
    patches = ANGLE_BINS + 2 * cap_bins + 2 * EDGE_BINS
    surface_radius = radius_rotor + support
    area = np.empty(patches)
    area[:ANGLE_BINS] = surface_radius * (2.0 * np.pi / ANGLE_BINS) * 2.0 * half_length
    edges = np.minimum(np.arange(cap_bins + 1) * cap_width, radius_rotor)
    area[ANGLE_BINS:ANGLE_BINS + 2 * cap_bins] = np.tile(np.pi * (edges[1:] ** 2 - edges[:-1] ** 2), 2)
    bounds = np.linspace(0.0, 0.5 * np.pi, EDGE_BINS + 1)
    edge_area = 2.0 * np.pi * support * (radius_rotor * (bounds[1:] - bounds[:-1])
                                         + support * (np.sin(bounds[1:]) - np.sin(bounds[:-1])))
    area[ANGLE_BINS + 2 * cap_bins:] = np.tile(edge_area, 2)

    moments = [np.bincount(patch, coordinate ** power, patches) for power in range(5)]
    matrix = np.empty((patches, 3, 3))
    for row in range(3):
        for column in range(3):
            matrix[:, row, column] = moments[row + column]

    def on_surface(values):
        right = np.column_stack([np.bincount(patch, values * coordinate ** power, patches) for power in range(3)])
        return np.linalg.solve(matrix, right[:, :, None])[:, 0, 0]
    vorticity_integrand = normal[:, 0] * velocity[:, 1] - normal[:, 1] * velocity[:, 0]
    spin_integrand = (normal[:, 2] * (position * spin).sum(axis=1) - spin[:, 2] * (position * normal).sum(axis=1))
    normal_velocity = (velocity * normal).sum(axis=1)
    moment_of_momentum = position[:, 0] * velocity[:, 1] - position[:, 1] * velocity[:, 0]
    vorticity = on_surface(vorticity_integrand)
    spin_term = on_surface(spin_integrand)
    outflow = on_surface(normal_velocity)
    lever = on_surface(moment_of_momentum)
    mean_outflow = float((outflow * area).sum() / area.sum())
    regions = {"lateral": slice(0, ANGLE_BINS), "caps": slice(ANGLE_BINS, ANGLE_BINS + 2 * cap_bins),
               "edges": slice(ANGLE_BINS + 2 * cap_bins, patches)}
    result = {"particles_min": int(np.bincount(patch, minlength=patches).min()), "area": float(area.sum()),
              "mean_outflow": mean_outflow}
    for label, values in (("vorticity", vorticity), ("spin", spin_term),
                          ("advection", DENSITY * lever * (outflow - mean_outflow))):
        result[label] = float((values * area).sum())
        result[f"{label}_by_region"] = {name: float((values[region] * area[region]).sum())
                                        for name, region in regions.items()}
    return result


def wall_pass_check(state: State) -> dict:
    """The wall pass again on the host for the rotor's wall particles (dummy velocity 2 v_w - <v>, wall pressure
    <p>, dummy micro-rotation 2 w_w - <w>, <.> the kernel-weighted mean over the fluid neighbours) against the
    GPU's values, for the wall particles within h of both the lateral surface and an end face (the edges) and for
    the others."""
    parameters = state.parameters
    support, spacing = parameters["support"], parameters["spacing"]
    wall = np.nonzero(((state.kind == 1) | (state.kind == 2)) & (state.to_rotor > -support))[0]
    fluid = np.nonzero(state.is_fluid & (state.to_rotor < support + spacing))[0]
    first, second, _offset, distance = _pairs(state.position[wall], state.position[fluid], support)
    kernel, _derivative, _second = wendland_c4(distance, support, 3)
    weight = np.bincount(first, kernel, wall.size)
    safe = np.where(weight > 0.0, weight, 1.0)

    def mean(values):
        return np.column_stack([np.bincount(first, values[fluid[second], column] * kernel, wall.size) / safe
                                for column in range(values.shape[1])])
    host_velocity = 2.0 * state.velocity[wall] - mean(state.velocity)
    host_pressure = np.bincount(first, state.pressure[fluid[second]] * kernel, wall.size) / safe
    host_spin = -mean(state.spin)                                       # w_w = 0 on every wall
    scale = float(wendland_c4(np.array([0.0]), support, 3)[0][0])                 # W(0)
    checked = weight > 1.0e-6 * scale
    depth_lateral = parameters["rotor_radius"] - state.radius[wall]
    depth_end = parameters["rotor_half_length"] - np.abs(state.position[wall, 2])
    at_edge = (depth_lateral < support) & (depth_end < support)
    speed = float(np.abs(state.velocity).max())
    result = {}
    for label, members in (("edges", checked & at_edge), ("elsewhere", checked & ~at_edge)):
        entry = {"wall_particles": int(members.sum()),
                 "dummy_velocity_max_difference_over_max_speed": float(
                     np.abs(host_velocity[members] - state.dummy_velocity[wall][members]).max() / speed),
                 "wall_pressure_max_difference_over_max_pressure": float(
                     np.abs(host_pressure[members] - state.pressure[wall][members]).max()
                     / max(float(np.abs(state.pressure).max()), 1e-30)),
                 "dummy_speed_max_over_wall_speed": float(
                     np.sqrt((state.dummy_velocity[wall][members] ** 2).sum(axis=1)).max() / speed)}
        if parameters["micropolar"]:
            entry["dummy_spin_max_difference_over_max_spin"] = float(
                np.abs(host_spin[members] - state.dummy_spin[wall][members]).max()
                / max(float(np.abs(state.spin).max()), 1e-30))
        result[label] = entry
    return result


def acceleration_check(state: State, wedge_degrees: float = 6.0) -> dict:
    """Every pair term of force.comp (pressure with its sign rule, viscous, c curl w; xi lap w, c curl v, -2 c w)
    summed over all the neighbours of the fluid particles within h of the rotor in a wedge about the x axis,
    against the GPU's acceleration and angular acceleration."""
    parameters = state.parameters
    support = parameters["support"]
    azimuth = np.degrees(np.arctan2(state.position[:, 1], state.position[:, 0]))
    target = np.nonzero(state.is_fluid & (state.to_rotor < support) & (np.abs(azimuth) < 0.5 * wedge_degrees))[0]
    first, second, offset, distance = _pairs(state.position[target], state.position, support)
    first_global = target[first]
    terms = state.pair_terms(first_global, second, offset, distance)
    volume, corrected = terms["volume"], terms["corrected"]
    kernel_sum = np.bincount(first, volume * terms["kernel"], target.size)
    self_pressure = state.pressure[first_global]
    near_surface = kernel_sum[first] < 0.75
    combined = np.where((self_pressure > 0.0) | near_surface, state.pressure[second] + self_pressure,
                        state.pressure[second] - self_pressure)
    pressure = -(volume * combined / state.density[first_global])[:, None] * corrected
    coupling = parameters["coupling"] * volume[:, None] * np.cross(terms["spin_difference"], corrected)
    total = np.column_stack([np.bincount(first, (pressure + terms["viscous"] + coupling)[:, column], target.size)
                             for column in range(3)])
    gpu = state.acceleration[target]
    scale = float(np.sqrt((gpu ** 2).sum(axis=1).mean()))
    difference = np.sqrt(((total - gpu) ** 2).sum(axis=1))
    result = {"target_fluid": int(target.size), "kernel_sum_min": float(kernel_sum.min()),
              "acceleration_rms_difference_over_rms": float(np.sqrt((difference ** 2).mean()) / scale),
              "acceleration_max_difference_over_rms": float(difference.max() / scale)}
    if parameters["micropolar"]:
        inertia = parameters["micro_inertia"]
        spin_rate = (terms["spin_diffusion"] + parameters["coupling"] * volume[:, None]
                     * np.cross(terms["velocity_difference"], corrected)) / inertia
        total_spin = np.column_stack([np.bincount(first, spin_rate[:, column], target.size) for column in range(3)])
        total_spin -= 2.0 * parameters["coupling"] * state.spin[target] / inertia
        gpu_spin = state.angular_acceleration[target]
        spin_scale = float(np.sqrt((gpu_spin ** 2).sum(axis=1).mean()))
        spin_difference = np.sqrt(((total_spin - gpu_spin) ** 2).sum(axis=1))
        result["angular_acceleration_rms_difference_over_rms"] = float(
            np.sqrt((spin_difference ** 2).mean()) / spin_scale)
        result["angular_acceleration_max_difference_over_rms"] = float(spin_difference.max() / spin_scale)
    return result


def readings(sums: dict, surface: dict, parameters: dict, mass: float) -> dict:
    """The torque about z of each wall on the fluid by the two readings, from the sums of one state."""
    factor = mass / (DENSITY * parameters["spacing"] ** 3)
    coupling = parameters["coupling"] if parameters["micropolar"] else 0.0
    shear_viscosity = DENSITY * (parameters["viscosity"] - coupling)              # mu
    coupling_viscosity = DENSITY * coupling                                       # kappa
    rotor_volume = np.pi * parameters["rotor_radius"] ** 2 * 2.0 * parameters["rotor_half_length"]
    wall_term = 2.0 * shear_viscosity * parameters["rotation_rate"] * rotor_volume
    lateral = sums["rotor_lateral"]["torque_total"] / factor
    end = sums["rotor_end"]["torque_total"] / factor
    field_terms = -shear_viscosity * surface["vorticity"] + coupling_viscosity * surface["spin"]
    circle_exchange = sums["circle"]["torque_total"] / factor
    result = {
        "mass_over_lattice_mass": factor, "shear_viscosity": shear_viscosity, "coupling_viscosity": coupling_viscosity,
        "rotor_wall": lateral + end + wall_term, "rotor_wall_lateral": lateral + wall_term, "rotor_wall_end": end,
        "rotor_wall_by_term": {
            "lateral": {name: value / factor for name, value in sums["rotor_lateral"]["torque"].items()},
            "end": {name: value / factor for name, value in sums["rotor_end"]["torque"].items()},
            "wall_term": wall_term},
        "rotor_circle": -(circle_exchange + field_terms),
        "rotor_circle_with_advection": -(circle_exchange + field_terms) + surface["advection"],
        "rotor_circle_by_term": {"pair_sum": circle_exchange,
                                 "vorticity_term": -shear_viscosity * surface["vorticity"],
                                 "spin_term": coupling_viscosity * surface["spin"],
                                 "advection": surface["advection"],
                                 "vorticity_by_region": surface["vorticity_by_region"],
                                 "spin_by_region": surface["spin_by_region"]},
        "rotor_force": ((sums["rotor_lateral"]["force_total"] + sums["rotor_end"]["force_total"]) / factor).tolist(),
    }
    if "housing_lateral" in sums:
        result["housing_wall"] = (sums["housing_lateral"]["torque_total"] + sums["housing_end"]["torque_total"]) / factor
        result["housing_wall_lateral"] = sums["housing_lateral"]["torque_total"] / factor
        result["housing_wall_end"] = sums["housing_end"]["torque_total"] / factor
        result["rotor_plus_housing_over_rotor"] = (result["rotor_wall"] + result["housing_wall"]) / result["rotor_wall"]
    return result


def run_readings(run_directory: pathlib.Path, with_checks: bool = True) -> dict:
    run = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
    parameters = parameters_of(run)
    files = sorted((run_directory / "states").glob("state_*.npz"))
    if not files:
        raise FileNotFoundError(f"no states under {run_directory}")
    per_state, checks = [], {}
    for index, path in enumerate(files):
        with np.load(path) as archive:
            raw = {name: np.array(archive[name]) for name in archive.files}
        state = State(raw, parameters)
        sums = wall_and_circle_sums(state)
        surface = surface_terms(state)
        entry = readings(sums, surface, parameters, state.mass)
        entry["step"] = int(raw["step"])
        entry["surface"] = {key: surface[key] for key in ("particles_min", "area", "mean_outflow")}
        per_state.append(entry)
        if with_checks and index == len(files) - 1:
            checks = {"wall_pass": wall_pass_check(state), "acceleration": acceleration_check(state)}
        print(f"[rotor3d torque] {run_directory.name} step {entry['step']:,}: wall {entry['rotor_wall']:.6f}, "
              f"circle {entry['rotor_circle']:.6f}", flush=True)
    summary = {"states": len(per_state), "steps": [entry["step"] for entry in per_state], "parameters": parameters,
               "checks": checks, "per_state": per_state}
    for key in ("rotor_wall", "rotor_wall_lateral", "rotor_wall_end", "rotor_circle", "rotor_circle_with_advection",
                "housing_wall", "housing_wall_lateral", "housing_wall_end", "rotor_plus_housing_over_rotor"):
        if key in per_state[0]:
            values = np.array([entry[key] for entry in per_state])
            summary[key] = {"mean": float(values.mean()),
                            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0}
    return summary


def command_sums(arguments) -> int:
    for name in arguments.runs:
        run_directory = _REPO_ROOT / arguments.directory / name
        summary = run_readings(run_directory, not arguments.no_checks)
        target = run_directory / "torque.json"
        target.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(json.dumps({key: summary[key] for key in summary if key not in ("per_state", "parameters")}, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    sums = commands.add_parser("sums")
    sums.add_argument("runs", nargs="+")
    sums.add_argument("--directory", default="logs/micropolar/stage6_rotor3d")
    sums.add_argument("--no-checks", action="store_true")
    sums.set_defaults(handler=command_sums)
    arguments = parser.parse_args()
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
