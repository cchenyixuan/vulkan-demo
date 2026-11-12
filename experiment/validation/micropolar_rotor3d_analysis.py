"""
micropolar_rotor3d_analysis.py - what the 3-D runs of Stage 6 (section 4 of the task) show.

  analyse   Per run: does it settle (angular momentum, kinetic energy, mid-plane fit over the eighths of the run),
            levels and their drift, the mid-plane profiles of v_theta and w_z against the 2-D exact solution for
            the radii of the case, how far the ends reach along the rotor, the wall layers of w on the rotor's end
            face (tangential and normal component), the torque by both readings with its split into lateral
            surface and end faces, speed and memory. Per pair: the change of the torque against the Newtonian
            control; a chain run against the one-slab run.
  sizing    Particles, steps, step limit, memory per card and hours per run of the production case at g = 80 dx on
            eight RTX 5090, from the measured runs.
  maps      The azimuthal means on the (r, z) grid over the last quarter and the fluid within 1.5 dx of the plane
            y = 0 at the last step with w - curl v / 2, as files for the repository (item 15 of the pack).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.validation import micropolar_couette as couette                       # noqa: E402
from experiment.validation import micropolar_couette_reference as reference           # noqa: E402
from experiment.validation.micropolar_rotor3d import DENSITY                          # noqa: E402

RUNS_DIRECTORY = "logs/micropolar/stage6_rotor3d"
# (micropolar run, its Newtonian control)
# The control at g = 40 dx ran on two slabs (g40_newtonian_k2); a one-slab control is taken if one exists.
PAIRS = (("g20_k3_m1", "g20_newtonian"), ("g20s_k3_m1", "g20s_newtonian"),
         ("g40_k3_m2", "g40_newtonian"), ("g40_k3_m2_k2", "g40_newtonian"),
         ("g40_k3_m2", "g40_newtonian_k2"), ("g40_k3_m2_k2", "g40_newtonian_k2"))
CHAINS = (("g40_k3_m2_k2", "g40_k3_m2"),)
# (run on the second lattice, run on the first): the scale for the differences of a chain against one slab
LATTICES = (("g20s_k3_m1", "g20_k3_m1"), ("g20s_newtonian", "g20_newtonian"))
DEVICE_LOCAL = re.compile(r"\[SimV7\] device-local buffers: (\d+), ([0-9.]+) MB")
DEFRAG_SCRATCH = re.compile(r"\[SimV7\] defrag scratch buffers: (\d+), ([0-9.]+) MB")
STAGING = re.compile(r"\[SimV7\] host staging buffers: (\d+)(?:, ([0-9.]+) (KB|MB))?")


def load(name: str, directory: pathlib.Path) -> dict:
    run_dir = directory / name
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    with np.load(run_dir / "samples.npz") as archive:
        data = {key: np.array(archive[key]) for key in archive.files}
    return {"name": name, "run": run, "meta": run["rotor3d"], "data": data, "directory": run_dir}


def _annulus_meta(meta: dict) -> dict:
    """The keys the 2-D fit of micropolar_couette reads."""
    return {"inner_radius": meta["rotor_radius"], "outer_radius": meta["housing_radius"],
            "smoothing_length": meta["smoothing_length"], "micropolar": meta["micropolar"],
            "micropolar_length": meta["micropolar_length"], "design_coupling": meta["design_coupling"],
            "design_viscosity": meta["design_viscosity"], "wall_speed": meta["wall_speed"],
            "rotation_rate": meta["rotation_rate"]}


def _fit(meta: dict, radius, velocity, spin) -> dict:
    return couette._fit({"meta": _annulus_meta(meta)}, {"radius": radius, "velocity": velocity, "spin": spin},
                        reference, 1.0)


def exact_two_dimensional(meta: dict) -> dict:
    """The 2-D solutions between infinite cylinders of the radii of the case."""
    inner, outer, rate = meta["rotor_radius"], meta["housing_radius"], meta["rotation_rate"]
    alpha, beta = reference.newtonian_couette_constants(inner, outer, rate * inner, 0.0)
    result = {"newtonian_alpha": float(alpha), "newtonian_beta": float(beta)}
    if meta["micropolar"]:
        eringen = reference.eringen_parameters_from_solver_parameters(
            DENSITY, meta["design_viscosity"], meta["design_coupling"], meta["design_viscosity_rotation"])
        solution = reference.solve_micropolar_couette(inner, outer, eringen.shear_viscosity,
                                                      eringen.coupling_viscosity, eringen.spin_viscosity,
                                                      rate * inner, 0.0, 0.0, 0.0)
        result["solution"] = solution
        result["torque_change"] = float(reference.relative_torque_change(
            inner, outer, eringen.shear_viscosity, eringen.coupling_viscosity, eringen.spin_viscosity,
            rate * inner, 0.0, 0.0, 0.0))
    return result


def analyse_run(item: dict) -> dict:
    run, meta, data = item["run"], item["meta"], item["data"]
    spacing, support, gap = meta["spacing"], meta["smoothing_length"], meta["gap"]
    inner, outer = meta["rotor_radius"], meta["housing_radius"]
    half_length, half_height = meta["rotor_half_length"], meta["housing_half_height"]
    rate, speed = meta["rotation_rate"], meta["wall_speed"]
    steps = run["steps"]
    step = data["step"]
    record = {"name": item["name"], "micropolar": meta["micropolar"], "gap_in_spacings": gap / spacing,
              "length_in_spacings": (meta["micropolar_length"] / spacing) if meta["micropolar"] else None,
              "coupling_ratio": meta["coupling_ratio"], "particles": run["particles"], "counts": meta["counts"],
              "steps": steps, "timestep": run["timestep"], "governing": meta["governing"],
              "dt_limits": {key: meta[key] for key in ("dt_acoustic", "dt_viscous", "dt_angular")},
              "slabs": run["slabs"], "device_map": run["device_map"], "weights": run["weights"],
              "slab_particles": run["slab_particles"], "cuts_x": run["cuts_x"],
              "steps_per_second": run["steps_per_second"], "elapsed_seconds": run["elapsed_seconds"],
              "bootstrap_seconds": run["bootstrap_seconds"], "failure": run["failure"],
              "counters_zero": not any(run["counters"].values()), "revision": run["revision"],
              "lattice_offset": meta["lattice_offset"]}
    log_path = item["directory"].with_suffix(".log")
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        local = [float(match[1]) for match in DEVICE_LOCAL.findall(text)]
        scratch = [float(match[1]) for match in DEFRAG_SCRATCH.findall(text)]
        if local:
            megabytes = [a + b for a, b in zip(local, scratch)] if len(scratch) == len(local) else local
            record["device_megabytes"] = megabytes
            record["bytes_per_particle"] = sum(megabytes) * 1024.0 * 1024.0 / run["particles"]
    # --- does it settle: the eighths of the run
    eighth = np.minimum(((step - 1) * 8) // steps, 7)
    angular_momentum = np.array([data["angular_momentum"][eighth == index].mean() for index in range(8)])
    energy = np.array([data["kinetic_energy"][eighth == index].mean() for index in range(8)])
    scale = DENSITY * speed ** 2
    density = np.array([data["density_mean"][eighth == index].mean() for index in range(8)]) / DENSITY - 1.0
    pressure = np.array([data["pressure_mean"][eighth == index].mean() for index in range(8)]) / scale
    record.update({
        "angular_momentum_windows": angular_momentum.tolist(), "kinetic_energy_windows": energy.tolist(),
        "angular_momentum_steady_change": float(abs(angular_momentum[7] - angular_momentum[6]) / angular_momentum[7]),
        "kinetic_energy_steady_change": float(abs(energy[7] - energy[6]) / energy[7]),
        "density_level": float(density[7]), "density_drift": float(density[7] - density[6]),
        "pressure_level": float(pressure[7]), "pressure_drift": float(pressure[7] - pressure[6]),
        "density_range": [float(data["density_min"].min()), float(data["density_max"].max())],
        "axial_speed_rms_over_wall_speed": float(data["axial_speed_rms"][eighth >= 6].mean() / speed),
        "history": {"time_over_diffusion_time": (step * run["timestep"] * meta["far_viscosity"] / gap ** 2).tolist(),
                    "angular_momentum_over_last": (data["angular_momentum"] / data["angular_momentum"][-1]).tolist(),
                    "kinetic_energy_over_last": (data["kinetic_energy"] / data["kinetic_energy"][-1]).tolist()}})
    # --- the mid-plane (|z| < 2 dx): mean over the samples of each eighth and of the last quarter
    radial_bins = data["midplane_count"].shape[1]
    centres = (np.arange(radial_bins) + 0.5) * spacing
    in_gap = (centres > inner) & (centres < outer)

    def midplane(selection) -> dict:
        count = data["midplane_count"][selection].sum(axis=0).astype(np.float64)[in_gap]
        return {"radius": data["midplane_radius"][selection].sum(axis=0)[in_gap] / count,
                "velocity": data["midplane_tangential"][selection].sum(axis=0)[in_gap] / count,
                "spin": data["midplane_spin_axial"][selection].sum(axis=0)[in_gap] / count}
    whole = midplane(eighth >= 6)
    fit = _fit(meta, whole["radius"], whole["velocity"], whole["spin"])
    constants = [_fit(meta, **midplane(eighth == index))["potential_vortex_constant"] for index in range(8)]
    exact = exact_two_dimensional(meta)
    record.update({
        "midplane": {key: values.tolist() for key, values in whole.items()},
        "midplane_fit": fit, "midplane_constants": constants,
        "midplane_steady_change": float(abs(constants[7] - constants[6]) / abs(constants[7])),
        "newtonian_beta_two_dimensional": exact["newtonian_beta"]})
    if meta["micropolar"]:
        solution = exact["solution"]
        record["exact_two_dimensional"] = {
            "torque_change": exact["torque_change"],
            "potential_vortex_constant": float(solution.potential_vortex_constant),
            "velocity": reference.velocity(solution, whole["radius"]).tolist(),
            "spin": reference.micro_rotation(solution, whole["radius"]).tolist()}
        core = (whole["radius"] > inner + support) & (whole["radius"] < outer - support)
        exact_velocity = np.asarray(record["exact_two_dimensional"]["velocity"])
        exact_spin = np.asarray(record["exact_two_dimensional"]["spin"])
        record["midplane_velocity_rms_against_two_dimensional"] = float(
            np.sqrt(((whole["velocity"] - exact_velocity) ** 2).mean() / (exact_velocity ** 2).mean()))
        record["midplane_spin_rms_against_two_dimensional"] = float(
            np.sqrt(((whole["spin"] - exact_spin) ** 2).mean() / (exact_spin ** 2).mean()))
        record["midplane_points_in_core"] = int(core.sum())
    else:
        alpha, beta = exact["newtonian_alpha"], exact["newtonian_beta"]
        exact_velocity = alpha * whole["radius"] + beta / whole["radius"]
        record["exact_two_dimensional"] = {"velocity": exact_velocity.tolist()}
        record["midplane_velocity_rms_against_two_dimensional"] = float(
            np.sqrt(((whole["velocity"] - exact_velocity) ** 2).mean() / (exact_velocity ** 2).mean()))
        record["midplane_beta_over_two_dimensional"] = fit["potential_vortex_constant"] / beta
    # --- how far the ends reach: the fitted constant of v_theta(r) in every axial bin along the rotor
    count = data["window_count"][6:].sum(axis=0)
    tangential = data["window_tangential"][6:].sum(axis=0)
    axial_bins = count.shape[1]
    height = -half_height + (np.arange(axial_bins) + 0.5) * spacing
    radius_bins = (np.arange(count.shape[0]) + 0.5) * spacing
    gap_bins = np.nonzero((radius_bins > inner) & (radius_bins < outer))[0]
    along = np.nonzero(np.abs(height) < half_length)[0]
    constants_by_height = []
    for column in along:
        filled = count[gap_bins, column] > 0
        radius = radius_bins[gap_bins][filled]
        velocity = tangential[gap_bins, column][filled] / count[gap_bins, column][filled]
        # the Newtonian form alpha r + beta / r for every run (two well-conditioned constants): a measure of the
        # shape along the rotor, not the micropolar constant
        away = (radius > inner + support) & (radius < outer - support)
        basis = np.column_stack([radius[away], 1.0 / radius[away]])
        constants_by_height.append(np.linalg.lstsq(basis, velocity[away], rcond=None)[0][1])
    constants_by_height = np.array(constants_by_height)
    middle = constants_by_height[np.abs(height[along]) < 0.1 * gap].mean()
    ratio = constants_by_height / middle
    from_end = (half_length - np.abs(height[along])) / gap
    reach = {}
    for tolerance in (0.01, 0.001):
        outside = from_end[np.abs(ratio - 1.0) > tolerance]
        reach[f"{tolerance:g}"] = float(outside.max()) if outside.size else 0.0
    record["end_reach"] = {"distance_from_end_in_gaps": from_end.tolist(), "constant_over_midplane": ratio.tolist(),
                           "reach_in_gaps": reach}
    # --- the wall layers of w on the rotor's end face (micropolar): between r = 0.4 R1 and 0.6 R1, against the
    # distance from the face; tangential component w_r (length l) and normal component w_z (its own length)
    if meta["micropolar"]:
        from scipy.optimize import least_squares
        band = np.nonzero((radius_bins > 0.4 * inner) & (radius_bins < 0.6 * inner))[0]
        above = np.nonzero((height > half_length) & (height < half_height))[0]
        weight = count[np.ix_(band, above)].sum(axis=0)
        distance = height[above] - half_length
        layers = {"distance_in_spacings": (distance / spacing).tolist()}
        length = meta["micropolar_length"]
        normal_length = length * np.sqrt(meta["far_viscosity"] / meta["design_viscosity"])
        for label, key, expected in (("normal", "window_spin_axial", normal_length),
                                     ("tangential", "window_spin_radial", length)):
            values = data[key][6:].sum(axis=0)[np.ix_(band, above)].sum(axis=0) / weight / rate
            end_gap = half_height - half_length

            def model(parameters, decay):
                # core linear in the distance, a layer at the rotor face and one at the housing face
                return (parameters[0] + parameters[1] * distance + parameters[2] * np.exp(-distance / decay)
                        + parameters[3] * np.exp(-(end_gap - distance) / decay))

            def residual(parameters):
                return model(parameters[:4], parameters[4]) - values
            start = np.array([values[len(values) // 2], 0.0, -values[len(values) // 2], 0.0, expected])
            solved = least_squares(residual, start, bounds=([-np.inf] * 4 + [0.2 * expected],
                                                            [np.inf] * 4 + [5.0 * expected]))
            layers[label] = {"values_over_rotation_rate": values.tolist(),
                             "fitted_length_in_spacings": float(solved.x[4] / spacing),
                             "expected_length_in_spacings": float(expected / spacing),
                             "fitted_over_expected": float(solved.x[4] / expected),
                             "fit_rms": float(np.sqrt((solved.fun ** 2).mean()))}
        record["end_face_layers"] = layers
    # --- the torque
    torque_path = item["directory"] / "torque.json"
    if torque_path.exists():
        torque = json.loads(torque_path.read_text(encoding="utf-8"))
        viscosity_far = DENSITY * meta["far_viscosity"]
        length = 2.0 * half_length
        lateral_reference = 4.0 * np.pi * viscosity_far * exact["newtonian_beta"] * length
        end_reference = np.pi * viscosity_far * rate * inner ** 4 / (half_height - half_length)
        record["torque"] = {key: torque[key] for key in torque if key not in ("per_state", "parameters")}
        record["torque"]["lateral_two_dimensional_newtonian"] = float(lateral_reference)
        record["torque"]["end_faces_plane_discs_newtonian"] = float(end_reference)
    return record


def compare_samples(directory: pathlib.Path, name: str, single: str) -> dict | None:
    """A run on several slabs against the run of the same case on one slab, sample by sample at equal steps:
    angular momentum and kinetic energy of the fluid and the mid-plane profiles of v_theta and w_z. The one-slab
    run may still be going (its samples_partial.npz is taken then)."""
    def samples(run: str):
        run_dir = directory / run
        for file_name, finished in (("samples.npz", True), ("samples_partial.npz", False)):
            if (run_dir / file_name).exists():
                with np.load(run_dir / file_name) as archive:
                    return {key: np.array(archive[key]) for key in archive.files}, finished
        return None, False
    chain, chain_finished = samples(name)
    base, base_finished = samples(single)
    if chain is None or base is None:
        return None
    steps = np.intersect1d(chain["step"], base["step"])
    if steps.size == 0:
        return None
    rows_chain = np.searchsorted(chain["step"], steps)
    rows_base = np.searchsorted(base["step"], steps)

    def relative(key):
        return chain[key][rows_chain] / base[key][rows_base] - 1.0

    def profile_difference(key):
        differences = []
        for row_chain, row_base in zip(rows_chain, rows_base):
            count_chain, count_base = chain["midplane_count"][row_chain], base["midplane_count"][row_base]
            filled = (count_chain > 0) & (count_base > 0)
            first = chain[key][row_chain][filled] / count_chain[filled]
            second = base[key][row_base][filled] / count_base[filled]
            scale = float((second ** 2).mean())
            # a Newtonian run has no micro-rotation: both profiles vanish and so does their difference
            differences.append(float(np.sqrt(((first - second) ** 2).mean() / scale)) if scale > 0.0 else 0.0)
        return np.asarray(differences)
    momentum, energy = relative("angular_momentum"), relative("kinetic_energy")
    velocity, spin = profile_difference("midplane_tangential"), profile_difference("midplane_spin_axial")
    last_quarter = steps > 0.75 * steps[-1]
    return {"single": single, "single_finished": bool(base_finished), "chain_finished": bool(chain_finished),
            "common_samples": int(steps.size), "first_step": int(steps[0]), "last_step": int(steps[-1]),
            "angular_momentum_largest": float(np.abs(momentum).max()), "angular_momentum_last": float(momentum[-1]),
            "kinetic_energy_largest": float(np.abs(energy).max()), "kinetic_energy_last": float(energy[-1]),
            "midplane_velocity_largest": float(velocity.max()), "midplane_velocity_last": float(velocity[-1]),
            "midplane_velocity_last_quarter_mean": float(velocity[last_quarter].mean()),
            "midplane_spin_largest": float(spin.max()), "midplane_spin_last": float(spin[-1]),
            "midplane_spin_last_quarter_mean": float(spin[last_quarter].mean()),
            "steps": steps.tolist(), "angular_momentum": momentum.tolist(), "kinetic_energy": energy.tolist(),
            "midplane_velocity": velocity.tolist(), "midplane_spin": spin.tolist()}


def command_analyse(arguments) -> int:
    directory = _REPO_ROOT / arguments.directory
    names = [path.parent.name for path in sorted(directory.glob("*/run.json"))]
    names = [name for name in names if not arguments.names or name in arguments.names]
    records = {}
    for name in names:
        records[name] = analyse_run(load(name, directory))
    lines = []

    def say(text=""):
        lines.append(text)
        print(text)
    say("3-D rotor in a housing (R1 = 3 g, length 3 g, housing radius 4 g, end gaps g; Re = 10 on mu_inf, Ma = 0.02, "
        "h = 4 dx): the runs")
    say(f"{'run':<16}{'g/dx':>5}{'l/dx':>5}{'K':>3}{'particles':>12}{'steps':>8}{'limit':>9}{'steps/s':>8}{'B/part.':>8}"
        f"{'L steady':>10}{'E steady':>10}{'mid steady':>11}{'v_z rms/U':>10}{'rho level':>11}{'drift':>10}"
        f"{'p level':>9}{'drift':>10}  describe")
    for name, record in records.items():
        say(f"{name:<16}{record['gap_in_spacings']:>5g}"
            + (f"{record['length_in_spacings']:>5g}" if record["micropolar"] else f"{'-':>5}")
            + f"{record['slabs']:>3}{record['particles']:>12,}{record['steps']:>8,}{record['governing'][:8]:>9}"
            f"{record['steps_per_second']:>8.1f}"
            + (f"{record['bytes_per_particle']:>8.0f}" if "bytes_per_particle" in record else f"{'-':>8}")
            + f"{record['angular_momentum_steady_change']:>10.1e}{record['kinetic_energy_steady_change']:>10.1e}"
            f"{record['midplane_steady_change']:>11.1e}{record['axial_speed_rms_over_wall_speed']:>10.1e}"
            f"{record['density_level']:>+11.2e}{record['density_drift']:>+10.1e}{record['pressure_level']:>+9.3f}"
            f"{record['pressure_drift']:>+10.1e}  {record['revision']['describe']}"
            + ("" if record["counters_zero"] and not record["failure"] else "  FAILED / counters"))
    say()
    say("mid-plane against the 2-D solution between infinite cylinders of the same radii; how far the ends reach "
        "(distance from the rotor's end, in g, beyond which the fitted constant of v_theta(r) is within 1 % and "
        "0.1 % of its mid-plane value)")
    say(f"{'run':<16}{'C2 mid-plane':>14}{'2-D':>12}{'ratio':>9}{'v rms vs 2-D':>14}{'w rms vs 2-D':>14}"
        f"{'reach 1 %':>11}{'0.1 %':>8}")
    for name, record in records.items():
        fitted = record["midplane_fit"]["potential_vortex_constant"]
        two_dimensional = (record["exact_two_dimensional"]["potential_vortex_constant"] if record["micropolar"]
                           else record["newtonian_beta_two_dimensional"])
        say(f"{name:<16}{fitted:>14.6f}{two_dimensional:>12.6f}{fitted / two_dimensional:>9.4f}"
            f"{record['midplane_velocity_rms_against_two_dimensional']:>14.2e}"
            + (f"{record['midplane_spin_rms_against_two_dimensional']:>14.2e}" if record["micropolar"]
               else f"{'-':>14}")
            + f"{record['end_reach']['reach_in_gaps']['0.01']:>11.2f}{record['end_reach']['reach_in_gaps']['0.001']:>8.2f}")
    if any("end_face_layers" in record for record in records.values()):
        say()
        say("wall layers of w above the rotor's end face (between 0.4 R1 and 0.6 R1): fitted decay length of the "
            "tangential component w_r (expected l) and of the normal component w_z (expected l sqrt(nu_inf / nu))")
        say(f"{'run':<16}{'w_r: fitted / dx':>18}{'expected':>10}{'ratio':>8}{'w_z: fitted / dx':>18}{'expected':>10}"
            f"{'ratio':>8}")
        for name, record in records.items():
            if "end_face_layers" not in record:
                continue
            tangential, normal = record["end_face_layers"]["tangential"], record["end_face_layers"]["normal"]
            say(f"{name:<16}{tangential['fitted_length_in_spacings']:>18.2f}"
                f"{tangential['expected_length_in_spacings']:>10.2f}{tangential['fitted_over_expected']:>8.3f}"
                f"{normal['fitted_length_in_spacings']:>18.2f}{normal['expected_length_in_spacings']:>10.2f}"
                f"{normal['fitted_over_expected']:>8.3f}")
    with_torque = {name: record for name, record in records.items() if "torque" in record}
    if with_torque:
        say()
        say("torque of the rotor on the fluid about its axis (N m; rho0 dx^3 per particle): wall form (pair sums over "
            "the rotor's wall particles, by lateral surface and end faces), circle form (flux across the closed "
            "surface at distance h, without and with the advective term), wall form of the housing; references: "
            "2-D Newtonian value times the rotor length, and two plane discs for the end faces")
        say(f"{'run':<16}{'states':>7}{'wall':>11}{'lateral':>11}{'ends':>10}{'circle':>11}{'+ adv.':>11}"
            f"{'circle/wall':>12}{'housing':>11}{'(R + H) / R':>12}{'lat / 2-D':>10}{'ends / discs':>13}"
            f"{'a check':>9}{'alpha check':>12}")
        for name, record in with_torque.items():
            torque = record["torque"]
            checks = torque["checks"].get("acceleration", {})
            say(f"{name:<16}{torque['states']:>7}{torque['rotor_wall']['mean']:>11.5f}"
                f"{torque['rotor_wall_lateral']['mean']:>11.5f}{torque['rotor_wall_end']['mean']:>10.5f}"
                f"{torque['rotor_circle']['mean']:>11.5f}{torque['rotor_circle_with_advection']['mean']:>11.5f}"
                f"{torque['rotor_circle']['mean'] / torque['rotor_wall']['mean']:>12.4f}"
                + (f"{torque['housing_wall']['mean']:>11.5f}{torque['rotor_plus_housing_over_rotor']['mean']:>+12.4f}"
                   if "housing_wall" in torque else f"{'-':>11}{'-':>12}")
                + f"{torque['rotor_wall_lateral']['mean'] / torque['lateral_two_dimensional_newtonian']:>10.4f}"
                f"{torque['rotor_wall_end']['mean'] / torque['end_faces_plane_discs_newtonian']:>13.4f}"
                + (f"{checks['acceleration_rms_difference_over_rms']:>9.1e}" if checks else f"{'-':>9}")
                + (f"{checks['angular_acceleration_rms_difference_over_rms']:>12.1e}"
                   if "angular_acceleration_rms_difference_over_rms" in checks else f"{'-':>12}"))
    pairs = {}
    for name, control in PAIRS:
        if name not in records or control not in records or name in pairs:
            continue
        item, base = records[name], records[control]
        entry = {"control": control,
                 "midplane_change": item["midplane_fit"]["potential_vortex_constant"]
                 / base["midplane_fit"]["potential_vortex_constant"] - 1.0,
                 "two_dimensional_change": item["exact_two_dimensional"]["torque_change"]}
        if "torque" in item and "torque" in base:
            for key in ("rotor_wall", "rotor_wall_lateral", "rotor_wall_end", "rotor_circle",
                        "rotor_circle_with_advection", "housing_wall"):
                if key in item["torque"] and key in base["torque"]:
                    entry[f"{key}_change"] = item["torque"][key]["mean"] / base["torque"][key]["mean"] - 1.0
            # the plane film between the two discs of an end gap (shear-driven, both walls displaced): exact
            from experiment.validation import micropolar_film as film
            meta_gap = 1.0 / (2.0 * item["gap_in_spacings"] / (2.0 * item["length_in_spacings"]))   # l / g
            displacement = film.displacement_over_length(item["coupling_ratio"]) * meta_gap
            entry["end_faces_plane_film_change"] = float(
                1.0 / (1.0 - 2.0 * displacement * np.tanh(0.5 / meta_gap)) - 1.0)
        pairs[name] = entry
    if pairs:
        say()
        say("micropolar run over its Newtonian control: the change of the mid-plane constant against the 2-D exact D "
            "for the radii, and the change of the torque by reading and by surface; for the end faces the exact "
            "plane film between two discs, 1 / (1 - 2 (delta / g) tanh(g / (2 l))) - 1")
        say(f"{'run':<16}{'D mid-plane':>13}{'D 2-D exact':>13}{'ratio':>8}{'D_T wall':>10}{'lateral':>9}{'ends':>9}"
            f"{'circle':>9}{'+ adv.':>9}{'housing':>9}{'ends, film':>12}")
        for name, entry in pairs.items():
            say(f"{name:<16}{entry['midplane_change']:>13.5f}{entry['two_dimensional_change']:>13.5f}"
                f"{entry['midplane_change'] / entry['two_dimensional_change']:>8.4f}"
                + "".join((f"{entry[key]:>{width}.4f}" if key in entry else f"{'-':>{width}}") for key, width in (
                    ("rotor_wall_change", 10), ("rotor_wall_lateral_change", 9), ("rotor_wall_end_change", 9),
                    ("rotor_circle_change", 9), ("rotor_circle_with_advection_change", 9),
                    ("housing_wall_change", 9), ("end_faces_plane_film_change", 12))))
    chains = {}
    for name, single in CHAINS:
        if name not in records or single not in records:
            continue
        item, base = records[name], records[single]
        difference = {key: float(np.sqrt(((np.asarray(item["midplane"][key]) - np.asarray(base["midplane"][key])) ** 2)
                                         .mean() / (np.asarray(base["midplane"][key]) ** 2).mean()))
                      for key in ("velocity", "spin")}
        entry = {"single": single, "midplane_rms_difference": difference,
                 "midplane_constant_ratio": item["midplane_fit"]["potential_vortex_constant"]
                 / base["midplane_fit"]["potential_vortex_constant"],
                 "angular_momentum_ratio": item["angular_momentum_windows"][7] / base["angular_momentum_windows"][7],
                 "kinetic_energy_ratio": item["kinetic_energy_windows"][7] / base["kinetic_energy_windows"][7]}
        if "torque" in item and "torque" in base:
            entry["rotor_wall_ratio"] = item["torque"]["rotor_wall"]["mean"] / base["torque"]["rotor_wall"]["mean"]
            entry["rotor_circle_ratio"] = (item["torque"]["rotor_circle"]["mean"]
                                           / base["torque"]["rotor_circle"]["mean"])
        chains[name] = entry
        say()
        say(f"chain against one slab ({name} over {single}): mid-plane profiles rms difference v "
            f"{difference['velocity']:.2e}, w {difference['spin']:.2e}; mid-plane constant ratio "
            f"{entry['midplane_constant_ratio']:.6f}; angular momentum {entry['angular_momentum_ratio']:.6f}; kinetic "
            f"energy {entry['kinetic_energy_ratio']:.6f}"
            + (f"; torque wall form {entry['rotor_wall_ratio']:.5f}, circle form {entry['rotor_circle_ratio']:.5f}"
               if "rotor_wall_ratio" in entry else ""))
    chain_samples = {}
    for name, single in CHAINS:
        entry = compare_samples(directory, name, single)
        if entry is None:
            continue
        chain_samples[name] = entry
        say()
        say(f"chain against one slab, sample by sample at equal steps ({name} over {single}"
            + ("" if entry["single_finished"] else ", which is still running") + f"): {entry['common_samples']} "
            f"samples from step {entry['first_step']:,} to {entry['last_step']:,}; angular momentum of the fluid "
            f"differs by at most {entry['angular_momentum_largest']:.1e} (last {entry['angular_momentum_last']:+.1e}), "
            f"kinetic energy by at most {entry['kinetic_energy_largest']:.1e} (last {entry['kinetic_energy_last']:+.1e}); "
            f"mid-plane profile of v_theta: rms difference over rms at most {entry['midplane_velocity_largest']:.1e} "
            f"(last {entry['midplane_velocity_last']:.1e}), of w_z at most {entry['midplane_spin_largest']:.1e} "
            f"(last {entry['midplane_spin_last']:.1e})")
    lattice_samples = {}
    for name, first in LATTICES:
        entry = compare_samples(directory, name, first)
        if entry is None:
            continue
        lattice_samples[name] = entry
        say()
        say(f"for scale, the second lattice against the first, sample by sample ({name} over {first}): angular "
            f"momentum differs by at most {entry['angular_momentum_largest']:.1e}, kinetic energy by at most "
            f"{entry['kinetic_energy_largest']:.1e}; mid-plane profile of v_theta at most "
            f"{entry['midplane_velocity_largest']:.1e}, of w_z at most {entry['midplane_spin_largest']:.1e}")
    if arguments.json:
        for record in records.values():
            record.get("exact_two_dimensional", {}).pop("solution", None)
        target = _REPO_ROOT / arguments.json
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"runs": records, "pairs": pairs, "chains": chains,
                                      "chain_samples": chain_samples, "lattice_samples": lattice_samples},
                                     indent=1), encoding="utf-8")
        target.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return 0


def command_maps(arguments) -> int:
    from experiment.validation.micropolar_stage6 import velocity_gradient
    directory = _REPO_ROOT / arguments.directory
    item = load(arguments.name, directory)
    run, meta, data = item["run"], item["meta"], item["data"]
    spacing = meta["spacing"]
    count = data["window_count"][6:].sum(axis=0)
    safe = np.where(count > 0, count, 1.0)
    maps = {key[len("window_"):]: (data[key][6:].sum(axis=0) / safe).astype(np.float32)
            for key in data if key.startswith("window_") and key != "window_count"}
    maps["count"] = count.astype(np.float32)
    # the slab through the axis at the last step: gradient by weighted least squares over the neighbours within
    # 3 dx, then the particles within 1.5 dx of the plane y = 0
    slab = np.load(sorted((item["directory"] / "states").glob("slab_axis_*.npz"))[-1])
    position = slab["position"].astype(np.float64)
    velocity = slab["velocity"].astype(np.float64)
    gradient = velocity_gradient(position, velocity, arguments.radius_in_spacings * spacing)
    curl = np.column_stack([gradient[:, 2, 1] - gradient[:, 1, 2], gradient[:, 0, 2] - gradient[:, 2, 0],
                            gradient[:, 1, 0] - gradient[:, 0, 1]])
    spin = slab["spin"].astype(np.float64) if "spin" in slab.files else np.zeros_like(velocity)
    keep = np.abs(position[:, 1]) < 1.5 * spacing
    side = np.sign(position[keep, 0])
    side[side == 0.0] = 1.0
    excess = spin[keep] - 0.5 * curl[keep]
    # in the plane y = 0: radial = x, azimuthal = y (times the side), axial = z
    target = _REPO_ROOT / arguments.out / f"{arguments.name}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target, **{f"map_{key}": values for key, values in maps.items()},
        x=position[keep, 0].astype(np.float32), z=position[keep, 2].astype(np.float32),
        tangential=(side * velocity[keep, 1]).astype(np.float32),
        pressure=slab["pressure"][keep].astype(np.float32),
        spin_radial=(side * spin[keep, 0]).astype(np.float32), spin_tangential=(side * spin[keep, 1]).astype(np.float32),
        spin_axial=spin[keep, 2].astype(np.float32),
        excess_radial=(side * excess[:, 0]).astype(np.float32),
        excess_tangential=(side * excess[:, 1]).astype(np.float32), excess_axial=excess[:, 2].astype(np.float32))
    target.with_suffix(".json").write_text(json.dumps(
        {"run": arguments.name, "rotor3d": meta, "revision": run["revision"], "steps": run["steps"],
         "slabs": run["slabs"], "cuts_x": run["cuts_x"],
         "gradient": f"weighted linear least squares over the fluid neighbours within "
                     f"{arguments.radius_in_spacings:g} dx", "particles_in_plane": int(keep.sum())}, indent=1),
        encoding="utf-8")
    print(f"[rotor3d] maps of {arguments.name}: {int(keep.sum()):,} particles in the plane -> {target}")
    return 0


def count_particles(gap_in_spacings: int, wall_layers: int = 9, offset_z: float = 0.5) -> dict:
    """Particle counts of the nominal case at a resolution, by the inequalities of the generator (lattice offset
    (0, 0, 0.5) dx), without writing a file."""
    rotor_radius, rotor_half = 3.0 * gap_in_spacings, 1.5 * gap_in_spacings
    housing_radius, housing_half = 4.0 * gap_in_spacings, 2.5 * gap_in_spacings
    reach = int(np.ceil(housing_radius + wall_layers)) + 2
    index = np.arange(-reach, reach + 1, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(index, index, indexing="ij")
    radius = np.hypot(grid_x, grid_y).ravel()
    radius = radius[radius < housing_radius + wall_layers]
    counts = {"domain": 0, "rotor": 0, "housing": 0}
    levels = np.arange(-int(np.ceil(housing_half + wall_layers)) - 2, int(np.ceil(housing_half + wall_layers)) + 3) \
        + offset_z
    for height in levels:
        if abs(height) >= housing_half + wall_layers:
            continue
        in_rotor = (radius <= rotor_radius) & (abs(height) <= rotor_half)
        in_cavity = (radius < housing_radius) & (abs(height) < housing_half)
        shell = in_rotor & ((radius > rotor_radius - wall_layers) | (abs(height) > rotor_half - wall_layers))
        counts["domain"] += int((in_cavity & ~in_rotor).sum())
        counts["rotor"] += int(shell.sum())
        counts["housing"] += int((~in_cavity).sum())
    return counts


def command_sizing(arguments) -> int:
    """What the measured runs say about the production run at g = 80 dx on eight RTX 5090: particles, steps, the
    step limit, memory per card, hours per run."""
    analysis = json.loads((_REPO_ROOT / arguments.analysis).read_text(encoding="utf-8"))
    runs = analysis["runs"]
    lines, result = [], {}

    def say(text=""):
        lines.append(text)
        print(text)
    check = count_particles(40)
    stored = next((runs[name].get("counts") for name in ("g40_newtonian", "g40_newtonian_k2", "g40_k3_m2",
                                                           "g40_k3_m2_k2") if name in runs), None)
    counts = count_particles(arguments.gap_in_spacings)
    total = sum(counts.values())
    say(f"Production case at g = {arguments.gap_in_spacings} dx: fluid {counts['domain']:,}, rotor {counts['rotor']:,}, "
        f"housing {counts['housing']:,}, total {total:,} particles"
        + (f" (the same count at g = 40 dx reproduces the generated case: fluid {check['domain']:,} = "
           f"{stored['domain']:,}, rotor {check['rotor']:,} = {stored['rotor_lateral'] + stored['rotor_end']:,}, "
           f"housing {check['housing']:,} = {stored['housing']:,})" if stored else ""))
    spacing = 0.1 / arguments.gap_in_spacings
    support, sound, far_viscosity = 4.0 * spacing, 50.0, 0.01
    dt_acoustic = 0.15 * support / sound
    dt_viscous = 0.0328 * support ** 2 / (1.6 * far_viscosity)
    dt_angular = 0.0334 * support ** 2 / (1.5 * far_viscosity)
    steps = {factor: int(np.ceil(round(factor * 0.1 ** 2 / far_viscosity / dt_acoustic / 1000.0, 6))) * 1000
             for factor in (1.5, 2.0)}
    say(f"Step: acoustic limit {dt_acoustic:.3e} s; viscous limit of K = 3 {dt_viscous:.3e} s ({dt_viscous / dt_acoustic:.1f} "
        f"times larger), angular limit {dt_angular:.3e} s ({dt_angular / dt_acoustic:.1f} times), with the step numbers "
        "measured in 2-D (not measured in 3-D); the acoustic limit governs. Steps: "
        + ", ".join(f"{count:,} to {factor:g} g^2 / nu_inf" for factor, count in steps.items()))
    result.update({"counts": counts, "total": total, "dt_acoustic": dt_acoustic, "dt_viscous": dt_viscous,
                   "dt_angular": dt_angular, "steps": {str(key): value for key, value in steps.items()}})
    partition_path = _REPO_ROOT / arguments.partition
    share, ghost_slots = 1.0 / 8.0, None
    if partition_path.exists():
        partition = json.loads(partition_path.read_text(encoding="utf-8"))
        particles = [slab["particles"] for slab in partition["slab"]]
        share = max(particles) / sum(particles)
        three = max(max(cut["three_columns_left"], cut["three_columns_right"]) for cut in partition["cut"])
        say(f"Eight slabs along x (partition of the g = 40 dx case, {partition['columns']} voxel columns): "
            f"{min(slab['columns'] for slab in partition['slab'])} to {max(slab['columns'] for slab in partition['slab'])} "
            f"columns per slab, particles per slab {min(particles):,} to {max(particles):,} (largest share "
            f"{share:.4f} against 0.125), wall share {min(slab['wall_share'] for slab in partition['slab']):.2f} to "
            f"{max(slab['wall_share'] for slab in partition['slab']):.2f}; the three ghost columns of a cut hold up to "
            f"{three:,} particles, {three / (sum(particles) / 8.0):.2f} of a slab. At g = 80 dx the columns double: "
            f"about {2 * partition['columns'] // 8} per slab, and the three ghost columns are half that share.")
        result["partition"] = {"largest_share": share, "three_ghost_columns_over_slab": three / (sum(particles) / 8.0)}
        ghost_slots = 4 * three                         # the cut face grows fourfold from g = 40 to 80 dx
    per_card = share * total
    say(f"Largest slab at g = {arguments.gap_in_spacings} dx: {per_card / 1e6:.1f} million particles")
    # the measured one-slab runs on the 5090: time per step = a + b n from the two sizes
    table = []
    for label, small, large in (("Newtonian", "g20_newtonian", "g40_newtonian"),
                                ("micropolar", "g20_k3_m1", "g40_k3_m2")):
        if small not in runs or large not in runs:
            continue
        one, two = runs[small], runs[large]
        time_one, time_two = 1.0 / one["steps_per_second"], 1.0 / two["steps_per_second"]
        slope = (time_two - time_one) / (two["particles"] - one["particles"])
        intercept = time_one - slope * one["particles"]
        step_time = intercept + slope * per_card
        bytes_per_particle = two.get("bytes_per_particle", one.get("bytes_per_particle"))
        memory = bytes_per_particle * per_card / 1024.0 ** 3 if bytes_per_particle else None
        ghost_memory = (2 * ghost_slots * bytes_per_particle / 1.25 / 1024.0 ** 3
                        if ghost_slots and bytes_per_particle else None)
        entry = {"steps_per_second_small": one["steps_per_second"], "steps_per_second_large": two["steps_per_second"],
                 "time_per_step_intercept": intercept, "time_per_particle": slope,
                 "one_slab_step_time_at_largest_slab": step_time, "bytes_per_particle": bytes_per_particle,
                 "memory_gigabytes_own": memory, "memory_gigabytes_ghost": ghost_memory,
                 "hours_without_chain_overhead": {str(factor): count * step_time / 3600.0
                                                  for factor, count in steps.items()}}
        result[label] = entry
        table.append((label, entry))
        say(f"{label}: {one['steps_per_second']:.1f} steps/s at {one['particles']:,} and {two['steps_per_second']:.2f} "
            f"at {two['particles']:,} particles on one 5090: {1e3 * intercept:.1f} ms + {1e9 * slope:.2f} ns per "
            f"particle; a slab of {per_card / 1e6:.1f} million: {1e3 * step_time:.0f} ms per step before any cost of "
            f"the chain; {bytes_per_particle:.0f} bytes per particle: {memory:.1f} GB for the slab's own particles"
            + (f" and about {ghost_memory:.1f} GB for the ghost regions of two cuts" if ghost_memory else "")
            + "; " + ", ".join(f"{count * step_time / 3600.0:.1f} h to {factor:g} g^2 / nu_inf"
                               for factor, count in steps.items()) + " without chain overhead")
    chain = runs.get("g40_k3_m2_k2")
    single = runs.get("g40_k3_m2")
    if chain and single:
        ratio = chain["steps_per_second"] / single["steps_per_second"]
        result["two_slabs_over_one_slab_rate"] = ratio
        say(f"The chain as measured: two slabs (5090 + XTX, weights {chain['weights']}, slabs of "
            f"{chain['slab_particles']}) run at {chain['steps_per_second']:.2f} steps/s against "
            f"{single['steps_per_second']:.2f} on the 5090 alone ({ratio:.2f} times).")
    if arguments.out:
        target = _REPO_ROOT / arguments.out
        target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        target.with_suffix(".json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    analyse = commands.add_parser("analyse")
    analyse.add_argument("names", nargs="*")
    analyse.add_argument("--directory", default=RUNS_DIRECTORY)
    analyse.add_argument("--json", default="data/micropolar/stage6/rotor3d.json")
    analyse.set_defaults(handler=command_analyse)
    maps = commands.add_parser("maps")
    maps.add_argument("name")
    maps.add_argument("--directory", default=RUNS_DIRECTORY)
    maps.add_argument("--radius-in-spacings", type=float, default=3.0)
    maps.add_argument("--out", default="data/micropolar/stage6/rotor3d")
    maps.set_defaults(handler=command_maps)
    sizing = commands.add_parser("sizing")
    sizing.add_argument("--gap-in-spacings", type=int, default=80)
    sizing.add_argument("--analysis", default="data/micropolar/stage6/rotor3d.json")
    sizing.add_argument("--partition", default="data/micropolar/stage6/rotor3d_partition_g40_k8.json")
    sizing.add_argument("--out", default="data/micropolar/stage6/rotor3d_sizing.txt")
    sizing.set_defaults(handler=command_sizing)
    arguments = parser.parse_args()
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
