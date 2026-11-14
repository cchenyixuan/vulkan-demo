"""
micropolar_channel_analysis.py - Stage 8c: the analysis of the channel runs of micropolar_channel.py
(docs/micropolar_design.md Section 19).

Steady runs (Couette, Poiseuille): the last samples against the exact solution - the rms of u - u_exact and of
w - w_exact over the rms of the exact profile, the wall stress (Couette) in three readings - "fit" (u = a s + b s^2
within 4 dx of each wall), "pairs" (the wall's force on the fluid from the solver's pair terms on the saved states,
per unit length), "profile" (the amplitude of the exact profile shape fitted to the whole binned profile, the
reading of items 01 to 03) -, the flow rate (Poiseuille); the micropolar over the control value against the exact
ratio; the floor from the repeat and from half the Mach number.

Start-up: tau(t) on the sliding wall (fit, pairs), Lambda(t) = (mu + kappa / 2) U / tau_N(t) of the host's Newtonian
solution at the run's sample times, E(t) = tau_mp / tau_control - 1 of the run pair against the host curve, the
profiles at Lambda = 0.25 l, l, 4 l against the host solution at the same times, the floors (repeat, half Mach) and
the smallest Lambda / l from which a resolution follows the host curve within its floor.

  python -m experiment.validation.micropolar_channel analyse
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np

from experiment.validation import micropolar_channel as channel
from experiment.validation.viscous_operators import wendland_c4

_REPO_ROOT = channel._REPO_ROOT
REGULARIZATION_XI = 0.001


# --- the pair sums of a plane wall ---------------------------------------------------------------------------------
def plane_wall_forces(state: dict, parameters: dict, length_x: float, height: float) -> dict:
    """Force of each wall (1 lower, 2 upper) on the fluid within h of it, by term, from the pair terms of force.comp
    (laplacian viscous form) on one saved state, the minimum image across the periodic seam in x."""
    from scipy.spatial import cKDTree
    support = channel.SUPPORT_OVER_SPACING * channel.SPACING
    epsilon_squared = 1.0e-6 * support ** 2
    position = state["position"].astype(np.float64)
    velocity = state["velocity"].astype(np.float64)
    density = state["density"].astype(np.float64)
    pressure = state["pressure"].astype(np.float64)
    kind = state["kind"]
    mass = float(state["mass"])
    micropolar = "spin" in state
    count = position.shape[0]
    spin = state["spin"].astype(np.float64) if micropolar else np.zeros(count)
    is_fluid = kind == 0
    is_wall = ~is_fluid
    neighbour_velocity = np.where(is_wall[:, None], state["dummy_velocity"].astype(np.float64), velocity)
    neighbour_spin = np.where(is_wall, state["dummy_spin"].astype(np.float64), spin) if micropolar else spin
    wall_distance = np.minimum(position[:, 1], height - position[:, 1])
    target = is_fluid & (wall_distance < support)
    wrapped = position.copy()
    wrapped[:, 0] = np.mod(wrapped[:, 0], length_x)
    shift = 1.0 - wrapped[:, 1].min()
    wrapped[:, 1] += shift
    tree = cKDTree(wrapped, boxsize=[length_x, 10.0 * (wrapped[:, 1].max() + 1.0)])
    pairs = tree.query_pairs(support, output_type="ndarray")
    first = np.concatenate([pairs[:, 0], pairs[:, 1]])
    second = np.concatenate([pairs[:, 1], pairs[:, 0]])
    keep = target[first]
    first, second = first[keep], second[keep]
    offset = position[first] - position[second]
    offset[:, 0] -= length_x * np.round(offset[:, 0] / length_x)
    distance = np.sqrt((offset ** 2).sum(axis=1))
    keep = (distance < support) & (distance >= 1e-12)
    first, second, offset, distance = first[keep], second[keep], offset[keep], distance[keep]
    kernel, derivative, _second = wendland_c4(distance, support, 2)
    volume = mass / density[second]
    gradient = (derivative / distance)[:, None] * offset
    moment = np.zeros((count, 2, 2))
    for row in range(2):
        for column in range(2):
            moment[:, row, column] = np.bincount(first, -volume * offset[:, row] * gradient[:, column], count)
    moment[:, 0, 0] += REGULARIZATION_XI
    moment[:, 1, 1] += REGULARIZATION_XI
    moment[~target] = np.eye(2)
    correction = np.linalg.inv(moment)
    if "correction_inverse" in state:
        # the solver's own M_i^-1 of the step (the host's, from the saved positions, is one density step later)
        packed = state["correction_inverse"].astype(np.float64)
        correction = np.stack([np.column_stack([packed[:, 0], packed[:, 3]]),
                               np.column_stack([packed[:, 3], packed[:, 1]])], axis=1)
    corrected = np.einsum("pab,pb->pa", correction[first], gradient)
    kernel_sum = np.bincount(first, volume * kernel, count)
    radial = (offset * corrected).sum(axis=1) / (distance ** 2 + epsilon_squared)
    self_pressure = pressure[first]
    near_surface = kernel_sum[first] < 0.75
    combined = np.where((self_pressure > 0.0) | near_surface, pressure[second] + self_pressure,
                        pressure[second] - self_pressure)
    velocity_difference = velocity[first] - neighbour_velocity[second]
    spin_difference = spin[first] - neighbour_spin[second]
    pair = {"pressure": -(volume * combined / density[first])[:, None] * corrected,
            "pressure_flux": -(volume * (pressure[second] + self_pressure) / density[first])[:, None] * corrected,
            "viscous": (2.0 * parameters["viscosity"] * volume * radial)[:, None] * velocity_difference,
            "coupling": (parameters["coupling"] * volume * spin_difference)[:, None]
            * np.column_stack([-corrected[:, 1], corrected[:, 0]])}
    total = sum(np.column_stack([np.bincount(first, pair[name][:, column], count) for column in range(2)])
                for name in ("pressure", "viscous", "coupling"))
    gpu = state["acceleration"].astype(np.float64)
    # the check: host against GPU acceleration of the fluid within h of a wall, over the rms of the wall part
    # (the total itself is near zero in a steady flow)
    from_wall = is_wall[second]
    wall_part = np.column_stack([np.bincount(first[from_wall], pair["viscous"][from_wall, column], count)
                                 for column in range(2)])
    scale = float(np.sqrt((wall_part[target] ** 2).sum(axis=1).mean())) or 1.0
    check = float(np.sqrt(((total[target] - gpu[target]) ** 2).sum(axis=1).mean()) / scale)
    result = {"acceleration_rms_difference_over_wall_rms": check,
              "used_solver_correction": "correction_inverse" in state}
    # the wall form of Stage 4 (Section 13.3): the exchange terms viscous + pressure in the flux form, in the
    # mass rho0 dx^2 (the calibrated particle mass is that times the volume calibration)
    lattice_mass = channel.DENSITY * channel.SPACING ** 2
    for label, wall_kind in (("lower", 1), ("upper", 2)):
        selected = kind[second] == wall_kind
        force = {name: (lattice_mass * values[selected]).sum(axis=0) / length_x for name, values in pair.items()}
        result[label] = {"by_term": {name: value.tolist() for name, value in force.items()},
                         "stress_x": float(force["viscous"][0] + force["pressure_flux"][0])}
    return result


def state_stresses(run_directory: pathlib.Path, parameters: dict, length_x: float, height: float) -> dict:
    """step -> pair-sum stresses of every saved state of a run."""
    out = {}
    for path in sorted((run_directory / "states").glob("state_*.npz")):
        if int(path.stem.split("_")[-1]) % 1000 == 0:
            continue        # read after the defrag: the auxiliary buffers keep the old order (micropolar_channel)
        with np.load(path) as archive:
            state = {name: np.array(archive[name]) for name in archive.files}
        result = plane_wall_forces(state, parameters, length_x, height)
        out[int(state["step"])] = result
    return out


# --- loading -------------------------------------------------------------------------------------------------------
def load(name: str):
    directory = _REPO_ROOT / channel.RUNS_DIRECTORY / name
    if not (directory / "run.json").exists():
        return None
    record = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    with np.load(directory / "samples.npz") as archive:
        samples = {key: np.array(archive[key]) for key in archive.files}
    return {"record": record, "samples": samples, "directory": directory, "meta": record["channel"]}


def rms_ratio(measured: np.ndarray, exact: np.ndarray, weights: np.ndarray) -> float:
    chosen = weights > 0
    return float(np.sqrt(np.mean((measured[chosen] - exact[chosen]) ** 2)) / np.sqrt(np.mean(exact[chosen] ** 2)))


def steady_result(item: dict) -> dict:
    meta, samples = item["meta"], item["samples"]
    parameters = meta["design"]
    rows, height = meta["rows"], meta["height"]
    y = channel.SPACING * (np.arange(rows) + 0.5)
    last = slice(-5, None)                                   # the last five samples (the last twelfth of the run)
    u = samples["u"][last].mean(axis=0)
    w = samples["w"][last].mean(axis=0)
    count = samples["count"][-1]
    flow = meta["flow"]
    viscosity_solver = channel.DENSITY * parameters["viscosity"]   # (mu + kappa) or mu_inf
    out = {"flow": flow, "m": meta["m"], "length": meta["length"], "micropolar": meta["micropolar"],
           "mach": meta["mach"], "repeat": meta["repeat"], "steps": item["record"]["steps"],
           "steps_per_second": item["record"]["steps_per_second"], "counters_zero": not any(
               item["record"]["counters"].values()), "outside_channel": int(samples["outside_channel"][-1])}
    if flow == "couette":
        exact = channel.couette_exact(parameters, height, y)
        out["u_rms_error"] = rms_ratio(u, exact["u"], count)
        out["w_rms_error"] = rms_ratio(w, exact["w"], count) if meta["micropolar"] else None
        out["stress_exact"] = exact["stress"]
        out["stress_newtonian"] = exact["stress_newtonian"]
        out["exact_ratio"] = exact["stress"] / exact["stress_newtonian"]
        lower = -viscosity_solver * samples["lower_fit"][last, 0].mean()
        upper = viscosity_solver * samples["upper_fit"][last, 0].mean()
        out["fit_lower"] = lower / exact["stress"]
        out["fit_upper"] = upper / exact["stress"]
        # the profile reading: u - U = s (u_exact - U), least squares over the bins
        shape = exact["u"] - channel.WALL_SPEED
        scale = float(np.sum((u - channel.WALL_SPEED) * shape) / np.sum(shape * shape))
        out["profile"] = scale
        stresses = state_stresses(item["directory"], parameters, meta["length_x"], height)
        if stresses:
            lower_pairs = np.mean([value["lower"]["stress_x"] for value in stresses.values()])
            upper_pairs = np.mean([value["upper"]["stress_x"] for value in stresses.values()])
            out["pairs_lower"] = float(lower_pairs) / exact["stress"]
            out["pairs_upper"] = float(-upper_pairs) / exact["stress"]
            out["pairs_check"] = float(max(value["acceleration_rms_difference_over_wall_rms"] for value in stresses.values()))
        # steadiness: the change of the fit reading over the last fifth of the samples
        series = -viscosity_solver * samples["lower_fit"][:, 0]
        out["steadiness"] = float(abs(series[-1] - series[-12]) / exact["stress"])
    else:
        force = meta["body_force"]
        exact = channel.poiseuille_exact(parameters, height, force, y)
        out["u_rms_error"] = rms_ratio(u, exact["u"], count)
        out["w_rms_error"] = rms_ratio(w, exact["w"], count) if meta["micropolar"] else None
        flow_rate = float(np.mean(samples["flow_rate"][last]))
        out["flow_rate_exact"] = exact["flow_rate"]
        out["flow_rate"] = flow_rate / exact["flow_rate"]
        out["exact_ratio"] = exact["flow_rate_ratio"]
        series = samples["flow_rate"]
        out["steadiness"] = float(abs(series[-1] - series[-12]) / exact["flow_rate"])
    return out


# --- the start-up ---------------------------------------------------------------------------------------------------
def host_curves() -> dict | None:
    path = _REPO_ROOT / channel.DATA_DIRECTORY / "startup_host.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def startup_series(item: dict, with_pairs: bool = True) -> dict:
    meta, samples = item["meta"], item["samples"]
    parameters = meta["design"]
    viscosity_solver = channel.DENSITY * parameters["viscosity"]
    out = {"time": samples["time"], "step": samples["step"], "fit": -viscosity_solver * samples["lower_fit"][:, 0]}
    if with_pairs:
        stresses = state_stresses(item["directory"], parameters, meta["length_x"], meta["height"])
        steps = sorted(stresses)
        out["pairs_step"] = np.asarray(steps)
        out["pairs"] = np.asarray([stresses[step]["lower"]["stress_x"] for step in steps])
        out["pairs_check"] = float(max((value["acceleration_rms_difference_over_wall_rms"] for value in stresses.values()),
                                       default=float("nan")))
    return out


def command_analyse(arguments) -> int:
    lines, result = [], {"steady": {}, "startup": {}}

    def say(text: str = "") -> None:
        lines.append(text)
        print(text, flush=True)

    items = {name: load(name) for name in channel.RUNS}
    items = {name: item for name, item in items.items() if item is not None}
    # --- steady
    say("Stage 8c: steady plane flows in the channel periodic in x (L = 100 dx, h = 5 dx, Re = 10, Ma 0.02)")
    for name, item in items.items():
        if item["meta"]["flow"] == "startup":
            continue
        result["steady"][name] = steady_result(item)
    for flow in ("couette", "poiseuille"):
        say()
        if flow == "couette":
            say(f"{'run':<30}{'u rms':>9}{'w rms':>9}{'fit lo':>9}{'fit up':>9}{'pairs lo':>9}{'pairs up':>9}"
                f"{'profile':>9}{'steady':>9}  over control (exact)")
        else:
            say(f"{'run':<30}{'u rms':>9}{'w rms':>9}{'Q/exact':>9}{'steady':>9}  over control (exact)")
        for name, entry in result["steady"].items():
            if entry["flow"] != flow:
                continue
            control = name.replace(f"_l{entry['length']}", f"_l{entry['length']}_control", 1) \
                if entry["micropolar"] else None
            ratio_text = ""
            if control and control in result["steady"]:
                other = result["steady"][control]
                if flow == "couette":
                    for key in ("fit_lower", "pairs_lower", "profile"):
                        if key in entry and key in other:
                            entry[f"{key}_over_control"] = (entry[key] * entry["stress_exact"]) / (
                                other[key] * other["stress_exact"])
                    ratio_text = " ".join(f"{key.split('_')[0]} {entry[f'{key}_over_control']:.4f}"
                                          for key in ("fit_lower", "pairs_lower", "profile")
                                          if f"{key}_over_control" in entry) + f" ({entry['exact_ratio']:.4f})"
                else:
                    entry["flow_rate_over_control"] = (entry["flow_rate"] * entry["flow_rate_exact"]) / (
                        other["flow_rate"] * other["flow_rate_exact"])
                    ratio_text = f"{entry['flow_rate_over_control']:.4f} ({entry['exact_ratio']:.4f})"
            w_text = f"{entry['w_rms_error']:>9.2e}" if entry["w_rms_error"] is not None else f"{'-':>9}"
            if flow == "couette":
                say(f"{name:<30}{entry['u_rms_error']:>9.2e}{w_text}{entry['fit_lower']:>9.4f}{entry['fit_upper']:>9.4f}"
                    f"{entry.get('pairs_lower', float('nan')):>9.4f}{entry.get('pairs_upper', float('nan')):>9.4f}"
                    f"{entry['profile']:>9.4f}{entry['steadiness']:>9.1e}  {ratio_text}")
            else:
                say(f"{name:<30}{entry['u_rms_error']:>9.2e}{w_text}{entry['flow_rate']:>9.4f}"
                    f"{entry['steadiness']:>9.1e}  {ratio_text}")
    # floors: the repeat and half the Mach number against the base run
    floors = {}
    for name, entry in result["steady"].items():
        for suffix in ("_repeat", "_ma0p01"):
            base = name[: -len(suffix)] if name.endswith(suffix) else None
            if base and base in result["steady"]:
                reference = result["steady"][base]
                keys = ("fit_lower", "pairs_lower", "profile", "u_rms_error") if entry["flow"] == "couette" else (
                    "flow_rate", "u_rms_error")
                floors.setdefault(base, {})[suffix[1:]] = {key: entry[key] - reference[key] for key in keys
                                                          if key in entry and key in reference}
                for key in ("fit_lower_over_control", "pairs_lower_over_control", "profile_over_control",
                            "flow_rate_over_control"):
                    if key in entry and key in reference:
                        floors[base][suffix[1:]][key] = entry[key] - reference[key]
    result["steady_floors"] = floors
    say()
    say("floors (repeat, half Mach): largest |change| over the runs of each flow")
    for flow in ("couette", "poiseuille"):
        values = {}
        for base, entry in floors.items():
            if result["steady"][base]["flow"] != flow:
                continue
            for variant, changes in entry.items():
                for key, value in changes.items():
                    values.setdefault(f"{key} ({variant})", []).append(abs(value))
        say(f"  {flow}: " + "; ".join(f"{key} {max(value):.1e}" for key, value in sorted(values.items())))

    # --- start-up
    host = host_curves()
    if host:
        say()
        say("Stage 8c: the start-up of plane Couette flow (H = 20 l, impulsive start) against the host solution")
        result["startup"] = startup_analysis(items, host, say)
    out_dir = _REPO_ROOT / channel.DATA_DIRECTORY
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "channel.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (out_dir / "channel.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    return 0


def startup_analysis(items: dict, host: dict, say) -> dict:
    from experiment.validation import micropolar_startup as startup
    out = {}
    pairs = []
    for name, item in items.items():
        meta = item["meta"]
        if meta["flow"] != "startup" or not meta["micropolar"]:
            continue
        control = name.replace(f"_l{meta['length']}", f"_l{meta['length']}_control", 1)
        if "_j001" in name:
            control = name.replace("_j001", "_j001_control")
        if control in items:
            pairs.append((name, control))
    for name, control in pairs:
        item, other = items[name], items[control]
        meta = item["meta"]
        length = meta["length"] * channel.SPACING
        far = meta["far_viscosity"]
        time_unit = length ** 2 / far                          # the host's time unit l^2 / nu_inf
        stress_unit = channel.DENSITY * far * channel.WALL_SPEED / length
        run_series, control_series = startup_series(item), startup_series(other)
        times = run_series["time"]
        host_times = times / time_unit
        # the host's Newtonian stress at the run's times -> Lambda / l; the host's excess for this j
        theta = meta["theta_factor"]
        key = f"theta_{theta:g}"
        newtonian = host["curves"]["newtonian"]
        newtonian_stress = np.interp(np.log(host_times), np.log(np.asarray(newtonian["times"])),
                                     np.asarray(newtonian["stress"]))
        layer = 1.0 / newtonian_stress                         # Lambda / l
        curve = host["curves"][key]
        host_excess = np.interp(np.log(layer), np.log(np.asarray(curve["layer_over_length"])),
                                np.asarray(curve["excess"]))
        excess_fit = run_series["fit"] / control_series["fit"] - 1.0
        entry = {"pair": [name, control], "length_in_spacings": meta["length"], "theta": theta,
                 "steps": item["record"]["steps"], "timestep": item["record"]["timestep"],
                 "steps_per_second": [item["record"]["steps_per_second"], other["record"]["steps_per_second"]],
                 "elapsed_seconds": [item["record"]["elapsed_seconds"], other["record"]["elapsed_seconds"]],
                 "time": times.tolist(), "layer_over_length": layer.tolist(), "host_excess": host_excess.tolist(),
                 "excess_fit": excess_fit.tolist(),
                 "stress_fit_over_host": (run_series["fit"] / stress_unit / np.interp(
                     np.log(host_times), np.log(np.asarray(curve["times"])), np.asarray(curve["stress"]))).tolist(),
                 "control_stress_fit_over_host": (control_series["fit"] / stress_unit / newtonian_stress).tolist()}
        if len(run_series.get("pairs", [])) and len(control_series.get("pairs", [])):
            common = np.intersect1d(run_series["pairs_step"], control_series["pairs_step"])
            run_pairs = run_series["pairs"][np.searchsorted(run_series["pairs_step"], common)]
            control_pairs = control_series["pairs"][np.searchsorted(control_series["pairs_step"], common)]
            pair_times = common * item["record"]["timestep"] / time_unit
            pair_layer = 1.0 / np.interp(np.log(pair_times), np.log(np.asarray(newtonian["times"])),
                                         np.asarray(newtonian["stress"]))
            entry["pairs_layer_over_length"] = pair_layer.tolist()
            entry["excess_pairs"] = (run_pairs / control_pairs - 1.0).tolist()
            entry["pairs_check"] = max(run_series["pairs_check"], control_series["pairs_check"])
        # the profiles where the Newtonian layer has Lambda / l = 0.25, 1, 4: the run's sample nearest in Lambda and
        # the host solution at that sample's time (units: y / l, u / U, w l / U)
        chosen = {}
        for value in (0.25, 1.0, 4.0):
            index = int(np.argmin(np.abs(np.log(layer / value))))
            chosen[f"{value:g}"] = index
        targets = {float(host_times[index]): label for label, index in chosen.items()}
        host_run = startup.solve(theta, 2.0e-5, 1.02, 0.02, 1.004, max(targets) * (1.0 + 1e-9), targets)
        host_newtonian = startup.solve(None, 2.0e-5, 1.02, 0.02, 1.004, max(targets) * (1.0 + 1e-9), targets)
        rows = meta["rows"]
        y = (np.arange(rows) + 0.5) * channel.SPACING / length
        profiles = {}
        for label, index in chosen.items():
            profiles[label] = {"layer_over_length": float(layer[index]), "time_host": float(host_times[index]),
                               "y": y.tolist(), "u": (item["samples"]["u"][index] / channel.WALL_SPEED).tolist(),
                               "w": (item["samples"]["w"][index] * length / channel.WALL_SPEED).tolist(),
                               "u_control": (other["samples"]["u"][index] / channel.WALL_SPEED).tolist(),
                               "host": host_run["profiles"].get(label), "host_newtonian": host_newtonian["profiles"].get(label)}
            host_profile = host_run["profiles"].get(label)
            if host_profile:
                host_y = np.asarray(host_profile["y"])
                for key in ("u", "w"):
                    exact = np.interp(y, host_y, np.asarray(host_profile[key]))
                    measured = np.asarray(profiles[label][key])
                    scale = np.sqrt(np.mean(exact ** 2)) or 1.0
                    profiles[label][f"{key}_rms_error"] = float(np.sqrt(np.mean((measured - exact) ** 2)) / scale)
        entry["profiles"] = profiles
        out[name] = entry
    # floors of l = 20 dx: E of the repeat pair and of the half-Mach pair against E of the base pair, by Lambda / l,
    # for both readings (fit, wall form); the runs at l = 40 and 80 dx are held against the same floor at their
    # Lambda / l
    base = out.get("startup_l20")
    newtonian = host["curves"]["newtonian"]

    def layer_of(times, meta):
        time_unit = (meta["length"] * channel.SPACING) ** 2 / meta["far_viscosity"]
        return 1.0 / np.interp(np.log(np.asarray(times) / time_unit), np.log(np.asarray(newtonian["times"])),
                               np.asarray(newtonian["stress"]))
    for variant in ("repeat", "ma0p01"):
        name = f"startup_l20_{variant}"
        control = f"startup_l20_control_{variant}"
        if base and name in items and control in items:
            run_series, control_series = startup_series(items[name]), startup_series(items[control])
            meta = items[name]["meta"]
            layer = layer_of(run_series["time"], meta)
            excess = run_series["fit"] / control_series["fit"] - 1.0
            base_on = np.interp(np.log(layer), np.log(np.asarray(base["layer_over_length"])),
                                np.asarray(base["excess_fit"]))
            floor_entry = {"layer_over_length": layer.tolist(), "excess_change": (excess - base_on).tolist()}
            common = np.intersect1d(run_series["pairs_step"], control_series["pairs_step"])
            if common.size and "excess_pairs" in base:
                run_pairs = run_series["pairs"][np.searchsorted(run_series["pairs_step"], common)]
                control_pairs = control_series["pairs"][np.searchsorted(control_series["pairs_step"], common)]
                pair_layer = layer_of(common * items[name]["record"]["timestep"], meta)
                base_pairs = np.interp(np.log(pair_layer), np.log(np.asarray(base["pairs_layer_over_length"])),
                                       np.asarray(base["excess_pairs"]))
                floor_entry["pairs_layer_over_length"] = pair_layer.tolist()
                floor_entry["pairs_excess_change"] = (run_pairs / control_pairs - 1.0 - base_pairs).tolist()
            out.setdefault("floors", {})[variant] = floor_entry
    floors = out.get("floors", {})

    def smallest_within(layer, miss, floor):
        within = miss <= np.maximum(floor, 1e-12)
        for index in range(len(layer)):
            if within[index:].all():
                return float(layer[index])
        return None
    for name, entry in out.items():
        if name == "floors":
            continue
        layer = np.asarray(entry["layer_over_length"])
        miss = np.abs(np.asarray(entry["excess_fit"]) - np.asarray(entry["host_excess"]))
        floor = np.zeros_like(layer)
        for variant in floors.values():
            floor = np.maximum(floor, np.abs(np.interp(np.log(layer), np.log(np.asarray(variant["layer_over_length"])),
                                                       np.asarray(variant["excess_change"]))))
        entry["floor"] = floor.tolist()
        entry["smallest_resolved_layer_over_length"] = smallest_within(layer, miss, floor)
        text_pairs = "-"
        if "excess_pairs" in entry:
            pair_layer = np.asarray(entry["pairs_layer_over_length"])
            host_on_pairs = np.interp(np.log(pair_layer), np.log(layer), np.asarray(entry["host_excess"]))
            pair_miss = np.abs(np.asarray(entry["excess_pairs"]) - host_on_pairs)
            pair_floor = np.zeros_like(pair_layer)
            for variant in floors.values():
                if "pairs_layer_over_length" in variant:
                    pair_floor = np.maximum(pair_floor, np.abs(np.interp(
                        np.log(pair_layer), np.log(np.asarray(variant["pairs_layer_over_length"])),
                        np.asarray(variant["pairs_excess_change"]))))
            entry["pairs_floor"] = pair_floor.tolist()
            entry["pairs_miss"] = (np.asarray(entry["excess_pairs"]) - host_on_pairs).tolist()
            resolved_pairs = smallest_within(pair_layer, pair_miss, pair_floor)
            entry["smallest_resolved_layer_over_length_pairs"] = resolved_pairs
            # the smallest Lambda / l from which the wall form stays within 0.005 of the host curve (a tolerance
            # beside the floor: the floor of a repeat is near zero where the run is deterministic)
            entry["smallest_layer_within_0p005_pairs"] = smallest_within(pair_layer, pair_miss,
                                                                       np.full_like(pair_floor, 0.005))
            text_pairs = (f"{'-' if resolved_pairs is None else f'{resolved_pairs:.3g}'} (wall form), "
                          f"within 0.005 from {entry['smallest_layer_within_0p005_pairs'] or float('nan'):.3g}")
        resolved = entry["smallest_resolved_layer_over_length"]
        say(f"  {name:<22} l = {entry['length_in_spacings']} dx, j = {entry['theta']:g} l^2: "
            f"{entry['steps']:,} steps; E within the floor from Lambda / l = "
            f"{'-' if resolved is None else f'{resolved:.3g}'} (fit), {text_pairs}; largest |E_fit - E_host| beyond "
            f"Lambda = l: {float(miss[layer >= 1.0].max()) if (layer >= 1.0).any() else float('nan'):.4f}")
    return out
