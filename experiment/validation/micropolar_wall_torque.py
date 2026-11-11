"""
micropolar_wall_torque.py - force and torque of a wall on the fluid from pair sums (docs/micropolar_design.md,
Stage 4, section 3), on the host in float64 from the wall states that micropolar_couette.py run --state-count saves.

For the fluid particles i and the particles j of one wall W, with the pair terms of force.comp and the neighbour
values the solver used in that step (wall pressure, dummy velocity and dummy micro-rotation of the wall pass, read
back from the GPU):

    F_W = sum_i sum_{j in W} m a_ij
    T_W = sum_i sum_{j in W} [ (r_i - r_axis) x m a_ij + m Theta alpha_ij ]_z

    a_ij     = - V_j (p_j +- p_i) / rho_i G_ij                              pressure (force.comp's sign rule)
               + 2 nu V_j (v_i - v~_j) (r_ij . G_ij) / (r_ij^2 + eps^2)     viscous, numerics.viscous_form laplacian
               + c V_j (w_i - w~_j) x G_ij                                  c curl w
    alpha_ij = [ 2 xi V_j (w_i - w~_j) (r_ij . G_ij) / (r_ij^2 + eps^2)     xi lap w
               + c V_j (v_i - v~_j) x G_ij ] / Theta                        c curl v

V_j = m / rho_j, G_ij = L_i grad W_ij, L_i = (M_i + xi_reg I)^-1 recomputed here from the saved positions and
densities (the solver's own L_i was built one density step earlier). The local term -2 c w_i of alpha_i is an
exchange inside the fluid and belongs to no wall.

Checks made on every state: the host's total acceleration and angular acceleration of the fluid particles within h
of a wall (all neighbours) against the GPU's buffers, and the wall pass recomputed here against the GPU's dummy
values.

  check     the two checks on the states of one run
  sums      F_W and T_W by term for both walls, mean and scatter over the states of a run
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

from experiment.validation.viscous_operators import wendland_c4   # noqa: E402

REGULARIZATION_XI = 0.001       # numerics.regularization.xi of the generated cases
DENSITY = 1000.0                # rest density of the generated cases
SURFACES = ("rotor", "stator", "rotor_layer", "stator_layer")
TERMS = ("pressure", "viscous", "coupling")
ANGLE_BINS = 72


def run_parameters(run: dict) -> dict:
    meta = run["couette"]
    return {"smoothing_length": meta["smoothing_length"], "spacing": meta["spacing"],
            "viscosity": meta["design_viscosity"], "coupling": meta["design_coupling"],
            "viscosity_rotation": meta["design_viscosity_rotation"], "micro_inertia": meta["design_micro_inertia"],
            "epsilon_squared": float(meta["epsilon_squared_factor"]) * meta["smoothing_length"] ** 2,
            "micropolar": bool(meta["micropolar"]), "inner_radius": meta["inner_radius"],
            "outer_radius": meta["outer_radius"], "eccentricity": meta["eccentricity"],
            "rotation_rate": meta["rotation_rate"], "viscous_form": meta["viscous_form"]}


def load_state(path) -> dict:
    with np.load(path) as archive:
        return {name: np.array(archive[name]) for name in archive.files}


def _scatter(index, values, count):
    if values.ndim == 1:
        return np.bincount(index, values, count)
    return np.column_stack([np.bincount(index, values[:, column], count) for column in range(values.shape[1])])


def wall_sums(state: dict, parameters: dict, angle_bins: int = ANGLE_BINS) -> dict:
    """The pair sums of one state. Returns per wall ("rotor", "stator") the force and the torque about the rotor
    axis by term, their distribution over the angle of the fluid particle about the wall's own axis, and the two
    checks."""
    from scipy.spatial import cKDTree
    if parameters["viscous_form"] != "laplacian":
        raise ValueError("the pair sums are written for numerics.viscous_form laplacian")
    support = parameters["smoothing_length"]
    # a state that holds the whole fluid (the eccentric runs): keep the band the sums need
    all_position = state["position"].astype(np.float64)
    band = np.minimum(np.hypot(all_position[:, 0] - parameters["eccentricity"], all_position[:, 1])
                      - parameters["inner_radius"], parameters["outer_radius"] - np.hypot(*all_position.T)) \
        < 2.0 * support + 2.0 * parameters["spacing"]
    if not band.all():
        state = {name: (values[band] if getattr(values, "ndim", 0) >= 1 and values.shape[0] == band.size else values)
                 for name, values in state.items()}
    position = state["position"].astype(np.float64)
    velocity = state["velocity"].astype(np.float64)
    density = state["density"].astype(np.float64)
    pressure = state["pressure"].astype(np.float64)
    kind = state["kind"]
    mass = float(state["mass"])
    micropolar = parameters["micropolar"]
    count = position.shape[0]
    spin = state["spin"].astype(np.float64) if micropolar else np.zeros(count)
    is_fluid = kind == 0
    is_wall = ~is_fluid
    # neighbour values as force.comp reads them: a wall neighbour enters with its dummy values
    neighbour_velocity = np.where(is_wall[:, None], state["dummy_velocity"].astype(np.float64), velocity)
    neighbour_spin = np.where(is_wall, state["dummy_spin"].astype(np.float64), spin) if micropolar else spin
    rotor_axis = np.array([parameters["eccentricity"], 0.0])
    rotor_distance = np.hypot(*(position - rotor_axis).T) - parameters["inner_radius"]
    stator_distance = parameters["outer_radius"] - np.hypot(*position.T)
    wall_distance = np.minimum(rotor_distance, stator_distance)
    target = is_fluid & (wall_distance < support)          # the fluid that can have a wall neighbour

    tree = cKDTree(position)
    pairs = tree.query_pairs(support, output_type="ndarray")
    first = np.concatenate([pairs[:, 0], pairs[:, 1]])
    second = np.concatenate([pairs[:, 1], pairs[:, 0]])
    del pairs
    keep = target[first] | (is_wall[first] & is_fluid[second])
    first, second = first[keep], second[keep]
    offset = position[first] - position[second]                                   # r_ij
    distance = np.sqrt((offset ** 2).sum(axis=1))
    keep = (distance < support) & (distance >= 1e-12)
    first, second, offset, distance = first[keep], second[keep], offset[keep], distance[keep]
    kernel, derivative, _second = wendland_c4(distance, support, 2)

    # --- the wall pass again (wall particle <- its fluid neighbours), against the GPU's dummy values
    from_wall = is_wall[first]
    wall_first, wall_second, wall_kernel = first[from_wall], second[from_wall], kernel[from_wall]
    weight = np.bincount(wall_first, wall_kernel, count)
    has_fluid = weight > 0.0
    safe = np.where(has_fluid, weight, 1.0)
    host_dummy_velocity = 2.0 * velocity - _scatter(wall_first, velocity[wall_second] * wall_kernel[:, None],
                                                    count) / safe[:, None]
    host_wall_pressure = np.bincount(wall_first, pressure[wall_second] * wall_kernel, count) / safe
    host_dummy_spin = 2.0 * spin - np.bincount(wall_first, spin[wall_second] * wall_kernel, count) / safe
    # walls whose fluid neighbours are all in the saved band: within h of the surface
    checked = is_wall & has_fluid & (wall_distance > -support)
    speed_scale = max(float(np.abs(velocity).max()), 1e-30)
    checks = {
        "dummy_velocity_max_difference_over_max_speed": float(np.abs(
            host_dummy_velocity[checked] - state["dummy_velocity"][checked]).max() / speed_scale),
        "wall_pressure_max_difference_over_max_pressure": float(np.abs(
            host_wall_pressure[checked] - pressure[checked]).max() / max(float(np.abs(pressure).max()), 1e-30)),
    }
    if micropolar:
        checks["dummy_spin_max_difference_over_max_spin"] = float(np.abs(
            host_dummy_spin[checked] - state["dummy_spin"][checked]).max() / max(float(np.abs(spin).max()), 1e-30))

    # --- the pairs of the target fluid
    from_fluid = target[first]
    first, second, offset, distance, derivative, kernel = (first[from_fluid], second[from_fluid], offset[from_fluid],
                                                           distance[from_fluid], derivative[from_fluid],
                                                           kernel[from_fluid])
    volume = mass / density[second]
    gradient = (derivative / distance)[:, None] * offset                           # grad_i W_ij
    moment = np.zeros((count, 2, 2))
    for row in range(2):
        for column in range(2):
            moment[:, row, column] = np.bincount(first, -volume * offset[:, row] * gradient[:, column], count)
    moment[:, 0, 0] += REGULARIZATION_XI
    moment[:, 1, 1] += REGULARIZATION_XI
    moment[~target] = np.eye(2)
    correction = np.linalg.inv(moment)
    corrected = np.einsum("pab,pb->pa", correction[first], gradient)                # G_ij
    kernel_sum = np.bincount(first, volume * kernel, count)
    radial = (offset * corrected).sum(axis=1) / (distance ** 2 + parameters["epsilon_squared"])

    self_pressure = pressure[first]
    near_surface = kernel_sum[first] < 0.75
    combined = np.where((self_pressure > 0.0) | near_surface, pressure[second] + self_pressure,
                        pressure[second] - self_pressure)
    velocity_difference = velocity[first] - neighbour_velocity[second]
    spin_difference = spin[first] - neighbour_spin[second]
    pair = {
        "pressure": -(volume * combined / density[first])[:, None] * corrected,
        # the same pressures in the form that is an exchange between i and j for every sign of p_i
        "pressure_flux": -(volume * (pressure[second] + self_pressure) / density[first])[:, None] * corrected,
        "viscous": (2.0 * parameters["viscosity"] * volume * radial)[:, None] * velocity_difference,
        "coupling": (parameters["coupling"] * volume * spin_difference)[:, None]
        * np.column_stack([-corrected[:, 1], corrected[:, 0]]),
    }
    if micropolar:
        inertia = parameters["micro_inertia"]
        pair_spin = {
            "spin_diffusion": 2.0 * parameters["viscosity_rotation"] * volume * spin_difference * radial / inertia,
            "spin_coupling": parameters["coupling"] * volume * (velocity_difference[:, 0] * corrected[:, 1]
                                                               - velocity_difference[:, 1] * corrected[:, 0]) / inertia,
        }
    else:
        inertia = 0.0
        pair_spin = {}

    # --- check: the total of every neighbour against the GPU's acceleration (fluid within h of a wall)
    total = sum(_scatter(first, pair[name], count) for name in TERMS)
    gpu = state["acceleration"].astype(np.float64)
    scale = float(np.sqrt((gpu[target] ** 2).sum(axis=1).mean()))
    difference = np.sqrt(((total[target] - gpu[target]) ** 2).sum(axis=1))
    checks["acceleration_rms_difference_over_rms"] = float(np.sqrt((difference ** 2).mean()) / scale)
    checks["acceleration_max_difference_over_rms"] = float(difference.max() / scale)
    if micropolar:
        total_spin = (sum(np.bincount(first, values, count) for values in pair_spin.values())
                      - 2.0 * parameters["coupling"] * spin / inertia)
        gpu_spin = state["angular_acceleration"].astype(np.float64)
        spin_scale = float(np.sqrt((gpu_spin[target] ** 2).mean()))
        checks["angular_acceleration_rms_difference_over_rms"] = float(
            np.sqrt(((total_spin[target] - gpu_spin[target]) ** 2).mean()) / spin_scale)
        checks["angular_acceleration_max_difference_over_rms"] = float(
            np.abs(total_spin[target] - gpu_spin[target]).max() / spin_scale)
    checks["kernel_sum_min"] = float(kernel_sum[target].min())
    checks["target_fluid"] = int(target.sum())

    # --- the sums over four classes of pairs of the fluid within h of a wall: its wall neighbours ("rotor",
    # "stator") and, as a control surface inside the fluid, its fluid neighbours further than h from that wall
    # ("rotor_layer", "stator_layer": what the fluid beyond the circle at distance h does to the layer)
    result = {"checks": checks, "mass": mass,
              "mass_over_lattice_mass": mass / (DENSITY * parameters["spacing"] ** 2)}
    arm = position[first] - rotor_axis                                             # r_i - r_axis
    middle = 0.5 * (position[first] + position[second]) - rotor_axis               # the pair's midpoint
    near_rotor = rotor_distance < stator_distance
    classes = (("rotor", kind[second] == 1, rotor_axis), ("stator", kind[second] == 2, np.zeros(2)),
               ("rotor_layer", is_fluid[second] & near_rotor[first] & (rotor_distance[second] >= support),
                rotor_axis),
               ("stator_layer", is_fluid[second] & ~near_rotor[first] & (stator_distance[second] >= support),
                np.zeros(2)))
    for label, selected, axis in classes:
        own_arm = position[first[selected]] - axis
        angle = np.arctan2(own_arm[:, 1], own_arm[:, 0]) % (2.0 * np.pi)
        sector = np.minimum((angle / (2.0 * np.pi) * angle_bins).astype(np.int64), angle_bins - 1)
        unit_radial = own_arm / np.sqrt((own_arm ** 2).sum(axis=1))[:, None]
        unit_tangential = np.column_stack([-unit_radial[:, 1], unit_radial[:, 0]])
        entry = {"pairs": int(selected.sum()), "force": {}, "torque": {}, "torque_midpoint": {},
                 "tangential_by_angle": {}, "radial_by_angle": {}}
        for name in pair:
            force = mass * pair[name][selected]
            entry["force"][name] = force.sum(axis=0).tolist()
            entry["torque"][name] = float((arm[selected, 0] * force[:, 1] - arm[selected, 1] * force[:, 0]).sum())
            entry["torque_midpoint"][name] = float((middle[selected, 0] * force[:, 1]
                                                    - middle[selected, 1] * force[:, 0]).sum())
            entry["tangential_by_angle"][name] = np.bincount(sector, (force * unit_tangential).sum(axis=1),
                                                             angle_bins).tolist()
            entry["radial_by_angle"][name] = np.bincount(sector, (force * unit_radial).sum(axis=1),
                                                         angle_bins).tolist()
        for name, values in pair_spin.items():
            couple = mass * inertia * values[selected]
            entry["torque"][name] = float(couple.sum())
            entry["torque_midpoint"][name] = entry["torque"][name]
            entry["tangential_by_angle"][name] = np.bincount(sector, couple, angle_bins).tolist()   # couple per bin
        # the formula of the task as written: every pair term of the solver, arm at the fluid particle
        entry["force_total"] = np.sum([entry["force"][name] for name in TERMS], axis=0).tolist()
        entry["torque_total"] = float(sum(entry["torque"][name] for name in (*TERMS, *pair_spin)))
        # the exchange terms only (pressure in its flux form, viscous, xi lap w), arm at the pair midpoint
        entry["force_exchange"] = np.sum([entry["force"][name] for name in ("pressure_flux", "viscous")],
                                         axis=0).tolist()
        entry["torque_exchange"] = float(sum(entry["torque_midpoint"][name]
                                             for name in ("pressure_flux", "viscous", "spin_diffusion")
                                             if name in entry["torque_midpoint"]))
        result[label] = entry

    # --- the first layer of fluid at each wall (distance < h): what the translation of the wall sums into the
    # torque of a symmetric stress needs (module docstring). internal_viscous = sum over the layer particles and
    # all their neighbours of (1/2) r_ij x m a_ij (viscous): the torque the non-central viscous pair forces
    # create inside the layer; circulation = the line integral of the velocity along the circle at distance h
    # from the wall (about the wall's own axis, counter-clockwise), from a straight-line fit of the tangential
    # velocity against the wall distance in each angle bin over the fluid within 2 dx of the circle;
    # total = sum over the layer of r_i x m a_i + m Theta alpha_i with the GPU's accelerations (zero when steady).
    half_pair_torque = 0.5 * mass * (offset[:, 0] * pair["viscous"][:, 1] - offset[:, 1] * pair["viscous"][:, 0])
    gpu_spin_rate = state["angular_acceleration"].astype(np.float64) if micropolar else np.zeros(count)
    particle_torque = mass * ((position[:, 0] - rotor_axis[0]) * gpu[:, 1] - (position[:, 1] - rotor_axis[1]) * gpu[:, 0]
                              + inertia * gpu_spin_rate)
    spacing = parameters["spacing"]
    for wall, in_layer, axis, distance_to_wall, circle_radius in (
            ("rotor", target & near_rotor, rotor_axis, rotor_distance, parameters["inner_radius"] + support),
            ("stator", target & ~near_rotor, np.zeros(2), stator_distance, parameters["outer_radius"] - support)):
        shell = is_fluid & (np.abs(distance_to_wall - support) < 2.0 * spacing)
        own = position[shell] - axis
        radius = np.sqrt((own ** 2).sum(axis=1))
        tangential = (-own[:, 1] * velocity[shell, 0] + own[:, 0] * velocity[shell, 1]) / radius
        sector = np.minimum((np.arctan2(own[:, 1], own[:, 0]) % (2.0 * np.pi) / (2.0 * np.pi) * angle_bins)
                            .astype(np.int64), angle_bins - 1)
        coordinate = (distance_to_wall[shell] - support) / spacing
        outward = (own[:, 0] * velocity[shell, 0] + own[:, 1] * velocity[shell, 1]) / radius
        number = np.bincount(sector, minlength=angle_bins).astype(np.float64)
        # value on the circle of a quadratic in the wall distance, fitted per angle bin (normal equations)
        moments = [np.bincount(sector, coordinate ** power, angle_bins) for power in range(5)]
        matrix = np.empty((angle_bins, 3, 3))
        for row in range(3):
            for column in range(3):
                matrix[:, row, column] = moments[row + column]

        def value_on_circle(values):
            right = np.column_stack([np.bincount(sector, values * coordinate ** power, angle_bins)
                                     for power in range(3)])
            return np.linalg.solve(matrix, right[:, :, None])[:, 0, 0]
        on_circle = value_on_circle(tangential)
        outward_on_circle = value_on_circle(outward)
        spin_on_circle = value_on_circle(spin[shell]) if micropolar else np.zeros(angle_bins)
        result[wall]["layer"] = {
            "particles": int(in_layer.sum()),
            "internal_viscous": float(half_pair_torque[in_layer[first]].sum()),
            "total": float(particle_torque[in_layer].sum()),
            "circle_radius": float(circle_radius),
            "circulation": float(on_circle.sum() * circle_radius * 2.0 * np.pi / angle_bins),
            "tangential_velocity_by_angle": on_circle.tolist(),
            "outward_velocity_by_angle": outward_on_circle.tolist(),
            "spin_by_angle": spin_on_circle.tolist(),
            "spin_integral": float(spin_on_circle.sum() * 2.0 * np.pi / angle_bins),      # integral of w d(angle)
            # integral of rho0 r v_t v_n dl along the circle, n = away from the rotor axis / the origin
            "angular_momentum_flux": float(DENSITY * circle_radius ** 2 * (on_circle * outward_on_circle).sum()
                                           * 2.0 * np.pi / angle_bins),
            "shell_particles_min": int(number.min())}
    return result


def state_files(run_directory: pathlib.Path) -> list:
    return sorted((run_directory / "states").glob("state_*.npz"))


def run_sums(run_directory: pathlib.Path, limit: int | None = None, last_fraction: float | None = None) -> dict:
    """wall_sums of every saved state of a run: per wall and term the mean, the standard deviation over the states
    and the standard error; the checks as the worst value over the states."""
    run = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
    parameters = run_parameters(run)
    files = state_files(run_directory)
    if last_fraction is not None:
        files = [path for path in files if int(path.stem.split("_")[1]) > (1.0 - last_fraction) * run["steps"]]
    files = files[:limit]
    if not files:
        raise FileNotFoundError(f"no states under {run_directory}")
    per_state = [wall_sums(load_state(path), parameters) for path in files]
    summary = {"states": len(per_state), "steps": [int(path.stem.split("_")[1]) for path in files],
               "parameters": parameters, "checks": {}}
    for key in per_state[0]["checks"]:
        values = [item["checks"][key] for item in per_state]
        summary["checks"][key] = float(min(values) if key in ("kernel_sum_min", "target_fluid") else max(values))
    summary["mass_over_lattice_mass"] = per_state[0]["mass_over_lattice_mass"]
    for wall in SURFACES:
        entry = {}
        for group in ("force", "torque", "torque_midpoint", "tangential_by_angle", "radial_by_angle"):
            entry[group] = {}
            for name in per_state[0][wall][group]:
                values = np.array([item[wall][group][name] for item in per_state])
                entry[group][name] = {"mean": values.mean(axis=0).tolist(),
                                      "std": values.std(axis=0, ddof=1).tolist() if len(per_state) > 1 else None}
        if "layer" in per_state[0][wall]:
            entry["layer"] = {}
            for name in per_state[0][wall]["layer"]:
                values = np.array([item[wall]["layer"][name] for item in per_state], dtype=np.float64)
                entry["layer"][name] = {"mean": values.mean(axis=0).tolist(),
                                        "std": values.std(axis=0, ddof=1).tolist() if len(per_state) > 1 else None}
        for name in ("force_total", "torque_total", "force_exchange", "torque_exchange"):
            values = np.array([item[wall][name] for item in per_state])
            entry[name] = {"mean": values.mean(axis=0).tolist(), "std": values.std(axis=0, ddof=1).tolist(),
                           "standard_error": (values.std(axis=0, ddof=1) / np.sqrt(len(per_state))).tolist(),
                           "values": values.tolist()}
        summary[wall] = entry
    return summary


def command_check(arguments) -> int:
    run_directory = (_REPO_ROOT / arguments.run).resolve()
    summary = run_sums(run_directory, arguments.limit)
    print(f"[wall-torque] {run_directory.name}: {summary['states']} states")
    for key, value in summary["checks"].items():
        print(f"  {key:<52}{value:.3e}")
    return 0


def command_sums(arguments) -> int:
    results = {}
    for name in arguments.runs:
        run_directory = (_REPO_ROOT / arguments.directory / name).resolve()
        summary = run_sums(run_directory, arguments.limit)
        results[name] = summary
        print(f"\n[wall-torque] {name}: {summary['states']} states; acceleration check "
              f"{summary['checks']['acceleration_rms_difference_over_rms']:.1e} rms")
        for wall in ("rotor", "stator"):
            entry = summary[wall]
            torque = entry["torque_total"]
            force = entry["force_total"]
            print(f"  {wall:<7} T = {torque['mean']:+.8e} (std {torque['std']:.1e}, s.e. {torque['standard_error']:.1e})"
                  f"   F = ({force['mean'][0]:+.4e}, {force['mean'][1]:+.4e})")
            print("          by term: " + ", ".join(f"{term} {values['mean']:+.5e}"
                                                    for term, values in entry["torque"].items()))
    if arguments.json:
        target = pathlib.Path(arguments.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("--run", required=True, help="run directory, relative to the repository root")
    check.add_argument("--limit", type=int, default=None)
    check.set_defaults(handler=command_check)
    sums = commands.add_parser("sums")
    sums.add_argument("runs", nargs="+")
    sums.add_argument("--directory", default="logs/micropolar/stage4")
    sums.add_argument("--limit", type=int, default=None)
    sums.add_argument("--json", default=None)
    sums.set_defaults(handler=command_sums)
    arguments = parser.parse_args()
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
