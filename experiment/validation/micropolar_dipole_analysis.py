"""
micropolar_dipole_analysis.py - Stage 8 analysis of the dipole-wall collision runs (micropolar_dipole.py):
the runs against the benchmark of Clercx & Bruneau (2006) at Re = 625, the micropolar runs against the control,
the noise floor from the shifted lattice (and the halved Mach number), into data/micropolar/stage8/.

  python -m experiment.validation.micropolar_dipole_analysis analyse
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

from experiment.validation import micropolar_dipole as dipole        # noqa: E402

CONTROL = "control_n1000"
FLOOR_RUNS = ("control_n1000_shifted", "control_n1000_half_mach")
MICROPOLAR = ("mp_l100_n1000", "mp_l50_n1000")
CONVERGENCE = ("control_n250", "control_n500", "control_n1000", "control_n2000")
DETACHED = 0.05                        # a negative extremum this far from x = W counts as detached (secondary)
SECONDARY_START = 0.45                 # the wall layer of the first collision has rolled up by then (item 20)
EDGE_DISTANCE = 0.03                   # where the reference velocity of the shear length is read (about 3 delta)


def load(run: str, partial: bool = False):
    directory = _REPO_ROOT / dipole.RUNS_DIRECTORY / run
    record_path = directory / "run.json"
    samples_path = directory / "samples.npz"
    if not record_path.exists() or not samples_path.exists():
        if partial and (directory / "samples_partial.npz").exists():
            samples_path = directory / "samples_partial.npz"
            record = json.loads((directory / "start.json").read_text(encoding="utf-8"))
            record["partial"] = True
        else:
            return None
    else:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("failure"):
            return None
    with np.load(samples_path) as archive:
        samples = {name: np.array(archive[name]) for name in archive.files}
    return {"run": run, "record": record, "samples": samples}


def peak(times: np.ndarray, values: np.ndarray, window: tuple) -> tuple:
    """The largest local maximum inside the window (a sample above its left and not below its right neighbour; the
    edges of the window do not count: the second peak sits on the flank of the first), refined by the parabola
    through it and its two neighbours."""
    inside = np.flatnonzero((times >= window[0]) & (times <= window[1]))
    inside = inside[(inside > 0) & (inside < times.size - 1)]
    local = [index for index in inside if values[index] > values[index - 1] and values[index] >= values[index + 1]
             and index != inside[0] and index != inside[-1]]
    if not local:
        return float("nan"), float("nan")
    index = max(local, key=lambda candidate: values[candidate])
    left, middle, right = values[index - 1:index + 2]
    step_left, step_right = times[index] - times[index - 1], times[index + 1] - times[index]
    if abs(step_left - step_right) > 1e-9 * step_left:
        return float(times[index]), float(middle)
    curvature = left - 2.0 * middle + right
    if curvature >= 0.0:
        return float(times[index]), float(middle)
    shift = 0.5 * (left - right) / curvature
    return float(times[index] + shift * step_left), float(middle - 0.25 * (left - right) * shift)


def value_at(times: np.ndarray, values: np.ndarray, time: float) -> float:
    return float(np.interp(time, times, values))


def track(times: np.ndarray, table: np.ndarray, start_time: float, start_choice, reach: float = 0.06,
          strongest: bool = False):
    """Follow one extremum through the samples: table (samples, kept, 3) rows (value, x, y). The first sample at or
    after start_time takes the row chosen by start_choice(rows); later samples the row nearest the previous
    position within `reach` (a gap ends the track)."""
    track_time, track_x, track_y, track_value = [], [], [], []
    previous = None
    for time, rows in zip(times, table):
        valid = rows[np.isfinite(rows[:, 0])]
        if time < start_time - 1e-9 or valid.size == 0:
            continue
        if previous is None:
            chosen = start_choice(valid)
            if chosen is None:
                continue
        else:
            distance = np.hypot(valid[:, 1] - previous[0], valid[:, 2] - previous[1])
            if distance.min() > reach:
                break
            if strongest:
                # the strongest extremum within reach (the nearest one can be a weak extremum of the wake: the
                # n = 500 run's primary jumped to one at t = 0.09 with the nearest rule); used by the checks only,
                # table 22 keeps the nearest rule
                near = valid[distance <= reach]
                chosen = near[np.argmax(np.abs(near[:, 0]))]
            else:
                chosen = valid[np.argmin(distance)]
        previous = (chosen[1], chosen[2])
        track_time.append(time)
        track_value.append(chosen[0])
        track_x.append(chosen[1])
        track_y.append(chosen[2])
    path = {"time": np.asarray(track_time), "x": np.asarray(track_x), "y": np.asarray(track_y),
            "value": np.asarray(track_value)}
    return smoothed(path)


def smoothed(path: dict, half_width: float = 0.0125) -> dict:
    """The extremum of a flat core hops between grid nodes (about 0.005 W): every quantity of the track is replaced
    by a local quadratic fit in time over +-half_width (at least three samples); the raw values and the rms of the
    residual of the positions stay in the track."""
    times = path["time"]
    if times.size < 3:
        return {**path, "raw_x": path["x"], "raw_y": path["y"], "position_noise": float("nan")}
    out = {"time": times, "raw_x": path["x"], "raw_y": path["y"], "raw_value": path["value"]}
    for key in ("x", "y", "value"):
        values = np.empty_like(path[key])
        for index, time in enumerate(times):
            chosen = np.flatnonzero(np.abs(times - time) <= half_width + 1e-9)
            if chosen.size < 3:
                order = np.argsort(np.abs(times - time))[:3]
                chosen = np.sort(order)
            offset = times[chosen] - time
            degree = 2 if chosen.size >= 4 else 1
            coefficients = np.polyfit(offset, path[key][chosen], degree)
            values[index] = coefficients[-1]
        out[key] = values
    out["position_noise"] = float(np.sqrt(np.mean((out["x"] - path["x"]) ** 2 + (out["y"] - path["y"]) ** 2)))
    return out


def field_track(run: str, start_time: float, sign: float, start_region: tuple, wall_band: float = 0.02,
                reach: float = 0.06) -> dict:
    """A core followed through the vorticity fields saved every 0.025 (all runs have them): the field smoothed by a
    Gaussian of two nodes, the upper half without the band next to x = W; at start_time the extremum of sign inside
    start_region ((x0, x1), (y0, y1)), later the extremum within `reach` of the previous position, refined by a
    parabola on each axis. (The extrema kept at every sample are the six strongest of the upper half, and from
    the first collision on the wall layer fills them: a detached secondary of |omega| = 100 drops out.)"""
    from scipy.ndimage import gaussian_filter
    directory = _REPO_ROOT / dipole.RUNS_DIRECTORY / run / "fields"
    paths = sorted(directory.glob("fields_*.npz"))
    track_time, track_x, track_y, track_value = [], [], [], []
    previous = None
    axis = None
    for path in paths:
        with np.load(path) as archive:
            time = float(archive["time"])
            if time < start_time - 1e-9:
                continue
            field = sign * gaussian_filter(np.array(archive["vorticity"], dtype=np.float32), 2.0)
        if axis is None:
            axis = -dipole.HALF_WIDTH + (2.0 * dipole.HALF_WIDTH / field.shape[0]) * (np.arange(field.shape[0]) + 0.5)
        x, y = np.meshgrid(axis, axis, indexing="ij")
        allowed = (y > 0.0) & (x < dipole.HALF_WIDTH - wall_band)
        if previous is None:
            allowed &= ((x >= start_region[0][0]) & (x <= start_region[0][1]) & (y >= start_region[1][0])
                        & (y <= start_region[1][1]))
        else:
            allowed &= np.hypot(x - previous[0], y - previous[1]) < reach
        if not allowed.any():
            break
        masked = np.where(allowed, field, -np.inf)
        column, row = np.unravel_index(np.argmax(masked), masked.shape)
        position = []
        for axis_index, index in ((0, column), (1, row)):
            if 0 < index < field.shape[axis_index] - 1:
                left = field[index - 1, row] if axis_index == 0 else field[column, index - 1]
                right = field[index + 1, row] if axis_index == 0 else field[column, index + 1]
                middle = field[column, row]
                curvature = left - 2.0 * middle + right
                shift = float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5)) if curvature < 0.0 else 0.0
            else:
                shift = 0.0
            position.append(axis[index] + shift * (axis[1] - axis[0]))
        previous = position
        track_time.append(time)
        track_x.append(position[0])
        track_y.append(position[1])
        track_value.append(sign * float(field[column, row]))
    return {"time": np.asarray(track_time), "x": np.asarray(track_x), "y": np.asarray(track_y),
            "value": np.asarray(track_value), "source": "fields every 0.025, Gaussian of two nodes"}


def summary(entry: dict, strongest: bool = False) -> dict:
    samples, record = entry["samples"], entry["record"]
    times = samples["time"]
    initial = record.get("initial", {})
    spacing = record["dipole"]["spacing"]
    first, second = (peak(times, samples["enstrophy"], window) for window in dipole.WINDOWS)
    palinstrophy_first, palinstrophy_second = (peak(times, samples["palinstrophy"], window)
                                               for window in dipole.WINDOWS)
    primary = track(times, samples["positive_extrema"], 0.0, lambda rows: rows[0], strongest=strongest)
    # the secondary vortex: the strongest negative extremum of the upper half detached from the wall after the
    # first collision
    # the secondary from the saved fields (see field_track): at t = 0.45 the negative core above the primary
    secondary = field_track(entry["run"], SECONDARY_START, -1.0, ((0.6, dipole.HALF_WIDTH - DETACHED), (0.2, 0.5)))
    lower = track(times, samples["lower_negative_extrema"], 0.0, lambda rows: rows[0], strongest=strongest)
    result = {
        "run": entry["run"], "partial": bool(record.get("partial")), "spacing": spacing,
        "end_time": float(times[-1]), "samples": int(times.size),
        "initial": {key: initial.get(key) for key in ("energy", "energy_grid", "enstrophy", "palinstrophy")},
        "enstrophy_peaks": [first, second], "palinstrophy_peaks": [palinstrophy_first, palinstrophy_second],
        "global": {f"{time:.2f}": [value_at(times, samples[name], time)
                                   for name in ("energy", "enstrophy", "palinstrophy")]
                   for time in (0.25, 0.50, 0.75) if times[-1] >= time - 1e-9},
        "leak": leak(samples), "outside_box_max": int(samples["outside_box"].max()),
        "density_mean_change": float(samples["density_mean"][-1] / dipole.DENSITY - 1.0),
        "energy_grid_over_particles": float(np.median(samples["energy_grid"] / samples["energy"])),
    }
    primary_rows = {}
    for time in (0.6, 0.625, 1.0):
        if primary["time"].size and primary["time"][-1] >= time - 1e-9:
            primary_rows[f"{time:g}"] = [value_at(primary["time"], primary[key], time) for key in ("x", "y", "value")]
    result["primary"] = primary_rows
    if primary["time"].size and primary["time"][-1] >= 0.25:
        # the strength of the primary core before the collision (Stage 8 checks: the bulk decay of the cores)
        result["primary_value_at_0.25"] = value_at(primary["time"], primary["value"], 0.25)
    # the symmetry of the collision: the negative primary of the lower half against the positive one
    if lower["time"].size and primary["time"].size:
        common = min(lower["time"].size, primary["time"].size)
        result["symmetry_largest_offset"] = float(np.max(np.hypot(lower["x"][:common] - primary["x"][:common],
                                                                  lower["y"][:common] - primary["y"][:common])))
    # closest approach to x = W and the rebound (the largest distance between the first and the second approach)
    if primary["time"].size:
        distance = dipole.HALF_WIDTH - primary["x"]
        first_window = (primary["time"] >= 0.2) & (primary["time"] <= 0.5)
        if first_window.any():
            index = np.flatnonzero(first_window)[np.argmin(distance[first_window])]
            result["closest_approach"] = [float(primary["time"][index]), float(distance[index]),
                                          float(primary["y"][index])]
            # the second approach: the closest point in 0.55 <= t <= 0.85; the rebound: the largest distance
            # between the two approaches (a micropolar run need not have a second enstrophy maximum)
            second_window = (primary["time"] >= 0.55) & (primary["time"] <= 0.85)
            if second_window.any():
                second_index = np.flatnonzero(second_window)[np.argmin(distance[second_window])]
                result["second_approach"] = [float(primary["time"][second_index]), float(distance[second_index]),
                                             float(primary["y"][second_index])]
                later = np.arange(index + 1, second_index)
                if later.size:
                    rebound = later[np.argmax(distance[later])]
                    result["rebound"] = [float(primary["time"][rebound]), float(distance[rebound]),
                                         float(primary["y"][rebound])]
    result["tracks"] = {name: {key: (values.tolist() if isinstance(values, np.ndarray)
                                     else values if isinstance(values, str) else float(values))
                               for key, values in path.items()}
                        for name, path in (("primary", primary), ("secondary", secondary), ("lower", lower))}
    result["series"] = {name: samples[name].tolist() for name in ("time", "energy", "enstrophy", "palinstrophy",
                                                                  "spin_departure", "max_speed", "density_mean")
                        if name in samples}
    result["series"]["time"] = [0.0] + result["series"]["time"]
    for name, key in (("energy", "energy"), ("enstrophy", "enstrophy"), ("palinstrophy", "palinstrophy")):
        result["series"][name] = [initial.get(key)] + result["series"][name]
    for name in ("spin_departure", "max_speed", "density_mean"):
        if name in result["series"]:
            result["series"][name] = [None] + result["series"][name]
    # the energy budget: -dE/dt over 2 nu_inf Omega (1 for the incompressible Newtonian fluid; the excess of the
    # control is its numerical dissipation; a micropolar run adds kappa (omega - 2 w)^2 and xi |grad w|^2)
    rate = -np.diff(samples["energy"]) / np.diff(times)
    middle_enstrophy = 0.5 * (samples["enstrophy"][1:] + samples["enstrophy"][:-1])
    middle_time = 0.5 * (times[1:] + times[:-1])
    budget = rate / (2.0 * dipole.FAR_VISCOSITY * middle_enstrophy)
    result["energy_budget"] = {label: float(np.mean(budget[(middle_time >= low) & (middle_time <= high)]))
                               for label, (low, high) in (("0.05-0.30", (0.05, 0.30)), ("0.30-0.45", (0.30, 0.45)),
                                                          ("0.45-1.00", (0.45, 1.00)))}
    result["max_speed_peak"] = float(samples["max_speed"].max())
    result["wall"] = wall_at(samples, spacing, first[0])
    if "spin_departure" in samples:
        result["spin_departure_peak"] = list(peak(times, samples["spin_departure"], (0.0, dipole.END_TIME)))
        result["spin_departure_at_first_peak"] = value_at(times, samples["spin_departure"], first[0])
    return result


def leak(samples: dict) -> dict:
    """k_w of v_n = k_w p / (rho c0) from all samples (fluid within one spacing of a wall), and the mean normal
    velocity."""
    totals = samples["leak_sums"].sum(axis=0)
    count, product, square, normal, scaled = totals
    late = samples["leak_sums"][samples["time"] > 0.25].sum(axis=0)
    return {"coefficient": float(product / square) if square > 0 else float("nan"),
            "coefficient_after_quarter": float(late[1] / late[2]) if late[2] > 0 else float("nan"),
            "mean_normal_velocity": float(normal / count) if count else float("nan"),
            "mean_scaled_pressure": float(scaled / count) if count else float("nan"), "rows": int(count)}


def wall_at(samples: dict, spacing: float, time: float) -> dict:
    """The wall x = W at the sample nearest `time`: the vorticity (particle fit) along the upper half, its peak and
    integral, the pressure difference along the wall, the vorticity flux d omega / ds at the wall from the grid
    band, and the reference velocity at EDGE_DISTANCE."""
    index = int(np.argmin(np.abs(samples["time"] - time)))
    fit_y = samples["wall_fit_y"][index]
    wall_vorticity = samples["wall_fit_vorticity"][index]
    upper = fit_y > 0.0
    width = fit_y[1] - fit_y[0]
    peak_index = np.flatnonzero(upper)[np.argmax(np.abs(wall_vorticity[upper]))]
    band = samples["band_vorticity"][index]                         # (nodes, ny), last row next to the wall
    nodes_y = -dipole.HALF_WIDTH + spacing * (np.arange(band.shape[1]) + 0.5)
    distance = spacing * (np.arange(band.shape[0])[::-1] + 0.5)      # of each band row from the wall
    # d omega / ds at s = 0 from omega = a + b s + c s^2 through the four rows next to the wall
    rows = slice(band.shape[0] - 4, band.shape[0])
    design = np.column_stack([np.ones(4), distance[rows], distance[rows] ** 2])
    coefficients = np.linalg.lstsq(design, band[rows, :].astype(np.float64), rcond=None)[0]
    flux = coefficients[1]
    node_upper = nodes_y > 0.0
    tangential = samples["band_tangential_velocity"][index]
    edge_row = band.shape[0] - 1 - int(round(EDGE_DISTANCE / spacing - 0.5))
    edge_velocity = np.abs(tangential[edge_row, :])
    pressure = samples["wall_pressure"][index]
    extra = {}
    if "wall_fit_spin_gradient" in samples:
        # dw/ds at the wall (particle fit, s = W - x) integrated over the upper half: the couple term of the
        # tangential momentum balance at the wall, dp/dy = (mu + kappa) d omega / dx - kappa d w / dx
        extra["spin_flux_integral_upper"] = float(np.nansum(samples["wall_fit_spin_gradient"][index][upper]) * width)
    return {**extra, "time": float(samples["time"][index]),
            "vorticity_peak": float(wall_vorticity[peak_index]), "vorticity_peak_y": float(fit_y[peak_index]),
            "vorticity_integral_upper": float(np.nansum(wall_vorticity[upper]) * width),
            "grid_vorticity_peak": float(samples["wall_vorticity"][index][node_upper][
                np.argmax(np.abs(samples["wall_vorticity"][index][node_upper]))]),
            "flux_integral_upper": float(flux[node_upper].sum() * spacing),
            "flux_peak": float(flux[node_upper][np.argmax(np.abs(flux[node_upper]))]),
            "pressure_difference_along_wall": float(pressure[np.argmin(np.abs(nodes_y))] - pressure[-1]),
            "edge_velocity_max": float(edge_velocity[node_upper].max()),
            "edge_velocity_y": float(nodes_y[node_upper][np.argmax(edge_velocity[node_upper])])}


def compare(run: dict, base: dict) -> dict:
    """Relative changes of the numbers of a run against a base run (absolute for times and positions)."""
    def relative(a, b):
        return float(a / b - 1.0) if b else float("nan")
    out = {}
    for label, index in (("first", 0), ("second", 1)):
        out[f"enstrophy_peak_{label}_time"] = run["enstrophy_peaks"][index][0] - base["enstrophy_peaks"][index][0]
        out[f"enstrophy_peak_{label}_value"] = relative(run["enstrophy_peaks"][index][1],
                                                        base["enstrophy_peaks"][index][1])
    # the enstrophy at the times of the base's two peaks (a run without a second maximum has a value there)
    for label in ("first", "second"):
        key = f"enstrophy_at_base_{label}_peak"
        if key in run and key in base:
            out[key] = relative(run[key], base[key])
    for key in ("closest_approach", "rebound", "second_approach"):
        if key in run and key in base:
            out[f"{key}_time"] = run[key][0] - base[key][0]
            out[f"{key}_distance"] = run[key][1] - base[key][1]
            out[f"{key}_distance_relative"] = relative(run[key][1], base[key][1])
    # the wall at the time of the base's first enstrophy peak (wall_common, set by command_analyse)
    wall, base_wall = run.get("wall_common", run["wall"]), base.get("wall_common", base["wall"])
    for key in ("vorticity_peak", "vorticity_integral_upper", "flux_integral_upper", "flux_peak",
                "pressure_difference_along_wall"):
        out[f"wall_{key}"] = relative(wall[key], base_wall[key])
    for time in ("0.25", "0.50", "0.75"):
        if time in run["global"] and time in base["global"]:
            out[f"energy_{time}"] = relative(run["global"][time][0], base["global"][time][0])
            out[f"enstrophy_{time}"] = relative(run["global"][time][1], base["global"][time][1])
    return out


# Stage 8 checks (13 Nov task, Part 0): (run, the run it is compared with, the control whose peak times are used)
CHECK_PAIRS = (("mp_l50_j001_n1000", "control_n1000", "control_n1000"),
               ("mp_l50_j001_n1000", "mp_l50_n1000", "control_n1000"),
               ("mp_l25_j001_n1000", "control_n1000", "control_n1000"),
               ("mp_l50_n500", "control_n500", "control_n500"),
               ("mp_l50_n1000", "control_n1000", "control_n1000"))


def check_pairs(results: dict, entries: dict, say) -> dict:
    """The rows of table 22 for the check runs: each against its base at the peak times of its family's control
    (the wall at the control's first enstrophy peak, the enstrophy at both), with the wall traction and the
    momentum balance at the wall for micropolar runs, and the strength of the primary core at t = 0.25."""
    checks, results_strongest = {}, {}
    for run, base, control in CHECK_PAIRS:
        if not all(name in results for name in (run, base, control)):
            continue
        first_time, second_time = (results[control]["enstrophy_peaks"][index][0] for index in (0, 1))
        pair = {}
        for name in (run, base):
            samples = entries[name]["samples"]
            item = summary(entries[name], strongest=True)
            results_strongest[name] = item
            item["wall_common"] = wall_at(samples, entries[name]["record"]["dipole"]["spacing"], first_time)
            for label, time in (("first", first_time), ("second", second_time)):
                item[f"enstrophy_at_base_{label}_peak"] = value_at(samples["time"], samples["enstrophy"], time)
            pair[name] = item
        values = compare(pair[run], pair[base])
        designs = {name: json.loads((_REPO_ROOT / f"cases/dipole/{name}/dipole.json").read_text(encoding="utf-8"))
                   ["design"] for name in (run, base)}
        wall_run, wall_base = pair[run]["wall_common"], pair[base]["wall_common"]
        values["wall_traction_peak"] = ((designs[run]["viscosity"] * wall_run["vorticity_peak"])
                                        / (designs[base]["viscosity"] * wall_base["vorticity_peak"]) - 1.0)
        values["wall_traction_integral"] = ((designs[run]["viscosity"] * wall_run["vorticity_integral_upper"])
                                            / (designs[base]["viscosity"] * wall_base["vorticity_integral_upper"]) - 1.0)
        def balance(name, wall):
            spin = designs[name]["coupling"] * wall.get("spin_flux_integral_upper", 0.0) if designs[name]["coupling"] else 0.0
            return designs[name]["viscosity"] * wall["flux_integral_upper"] - spin
        values["wall_momentum_balance"] = balance(run, wall_run) / balance(base, wall_base) - 1.0
        values["primary_value_at_0.25"] = (pair[run].get("primary_value_at_0.25", float("nan"))
                                           / pair[base].get("primary_value_at_0.25", float("nan")) - 1.0)
        # the shear length of the family's control and delta* / Lambda of the run
        control_wall = pair[base]["wall_common"] if base == control else wall_at(
            entries[control]["samples"], entries[control]["record"]["dipole"]["spacing"], first_time)
        shear_length = control_wall["edge_velocity_max"] / abs(control_wall["vorticity_peak"])
        length = dipole.RUNS[run][1] * dipole.HALF_WIDTH
        displacement = dipole.COUPLING_RATIO * length / (2.0 * (1.0 + dipole.COUPLING_RATIO))
        key = f"{run}__against__{base}"
        checks[key] = {"run": run, "base": base, "control": control, "changes": values,
                       "base_values": {"enstrophy_peak_first_value": pair[base]["enstrophy_peaks"][0][1],
                                       "wall_vorticity_peak": wall_base["vorticity_peak"],
                                       "primary_value_at_0.25": pair[base].get("primary_value_at_0.25")},
                       "run_values": {"primary_value_at_0.25": pair[run].get("primary_value_at_0.25"),
                                      "enstrophy_peaks": results[run]["enstrophy_peaks"]},
                       "shear_length": shear_length, "displacement_over_shear_length": displacement / shear_length,
                       "shear_length_over_length": shear_length / length}
        say()
        say(f"check {run} against {base} (times of {control}): first enstrophy peak "
            f"{values['enstrophy_peak_first_value']:+.4f}, wall vorticity {values['wall_vorticity_peak']:+.4f}, traction "
            f"{values['wall_traction_peak']:+.4f}, flux {values['wall_flux_integral_upper']:+.4f}, primary core at t = 0.25 "
            f"{values['primary_value_at_0.25']:+.4f}; delta* / Lambda {displacement / shear_length:.3f}")
    return checks


def command_analyse(arguments) -> int:
    lines = []

    def say(text: str = "") -> None:
        lines.append(text)
        print(text, flush=True)

    results, entries = {}, {}
    for run in dipole.RUNS:
        entry = load(run, partial=arguments.partial)
        if entry is not None:
            entries[run] = entry
            results[run] = summary(entry)
    if CONTROL in results:
        # every run read at the control's peak times: the wall at its first peak, the enstrophy at both
        first_time, second_time = (results[CONTROL]["enstrophy_peaks"][index][0] for index in (0, 1))
        for run, entry in entries.items():
            samples = entry["samples"]
            results[run]["wall_common"] = wall_at(samples, entry["record"]["dipole"]["spacing"], first_time)
            for label, time in (("first", first_time), ("second", second_time)):
                if samples["time"][-1] >= time:
                    results[run][f"enstrophy_at_base_{label}_peak"] = value_at(samples["time"], samples["enstrophy"],
                                                                               time)
    benchmark = dipole.BENCHMARK
    say("Stage 8: normal dipole-wall collision (Clercx & Bruneau 2006), Re = 625 on nu_inf")
    say()
    say(f"{'run':<28}{'dx':>8}{'end t':>7}{'E(0)':>9}{'Om(0)':>9}{'t1':>8}{'Omega1':>9}{'t2':>8}{'Omega2':>9}"
        f"{'k_w':>8}{'out':>5}")
    for run, item in results.items():
        say(f"{run:<28}{item['spacing']:>8.4f}{item['end_time']:>7.3f}{item['initial']['energy']:>9.5f}"
            f"{item['initial']['enstrophy']:>9.2f}{item['enstrophy_peaks'][0][0]:>8.4f}{item['enstrophy_peaks'][0][1]:>9.2f}"
            f"{item['enstrophy_peaks'][1][0]:>8.4f}{item['enstrophy_peaks'][1][1]:>9.2f}"
            f"{item['leak']['coefficient']:>8.3f}{item['outside_box_max']:>5}" + ("  (partial)" if item["partial"] else ""))
    validation = {}
    for run in [name for name in CONVERGENCE + FLOOR_RUNS if name in results]:
        item = results[run]
        rows = []

        def row(label, reference_sm, reference_fd, value, kind="relative"):
            difference = (value / reference_sm - 1.0) if kind == "relative" else (value - reference_sm)
            rows.append({"quantity": label, "SM": reference_sm, "FD": reference_fd, "run": value,
                         "difference": difference, "kind": kind})
        for index, label in ((0, "first"), (1, "second")):
            (time_sm, value_sm), (time_fd, value_fd) = (benchmark["enstrophy_peaks"][method][index]
                                                        for method in ("SM", "FD"))
            row(f"time of the {label} enstrophy peak", time_sm, time_fd, item["enstrophy_peaks"][index][0], "absolute")
            row(f"{label} enstrophy peak", value_sm, value_fd, item["enstrophy_peaks"][index][1])
            (time_sm, value_sm), (time_fd, value_fd) = (benchmark["palinstrophy_peaks"][method][index]
                                                        for method in ("SM", "FD"))
            row(f"time of the {label} palinstrophy peak", time_sm, time_fd, item["palinstrophy_peaks"][index][0],
                "absolute")
            row(f"{label} palinstrophy peak", value_sm, value_fd, item["palinstrophy_peaks"][index][1])
        for time, values in benchmark["global"]["SM"].items():
            key = f"{time:.2f}"
            if key not in item["global"]:
                continue
            for position, name in enumerate(("E", "Omega", "P")):
                row(f"{name}({time:g})", values[position], benchmark["global"]["FD"][time][position],
                    item["global"][key][position])
        for method in ("SM", "FD"):
            for time, x, y, strength in benchmark["primary"][method]:
                key = f"{time:g}"
                if key not in item["primary"]:
                    continue
                value_x, value_y, value_strength = item["primary"][key]
                row(f"primary core x at t = {time:g} ({method})", x, x if method == "FD" else float("nan"), value_x,
                    "absolute")
                row(f"primary core y at t = {time:g} ({method})", y, y if method == "FD" else float("nan"), value_y,
                    "absolute")
                row(f"primary core omega_max at t = {time:g} ({method})", strength, float("nan"), value_strength)
        validation[run] = rows
        say()
        say(f"{run} against the benchmark (SM: spectral, FD: finite differences; difference relative to SM, "
            "absolute for times and positions)")
        say(f"  {'quantity':<44}{'SM':>12}{'FD':>12}{'run':>12}{'difference':>12}")
        for entry in rows:
            say(f"  {entry['quantity']:<44}{entry['SM']:>12.5g}{entry['FD']:>12.5g}{entry['run']:>12.5g}"
                f"{entry['difference']:>+12.4f}")
    comparisons, floors = {}, {}
    if CONTROL in results:
        base = results[CONTROL]
        for run in FLOOR_RUNS:
            if run in results:
                floors[run] = compare(results[run], base)
        for run in MICROPOLAR:
            if run in results:
                comparisons[run] = compare(results[run], base)
        if floors or comparisons:
            keys = sorted({key for values in list(floors.values()) + list(comparisons.values()) for key in values})
            names = list(floors) + list(comparisons)
            say()
            say("against the control " + CONTROL + " (relative changes; absolute for times and distances)")
            say(f"  {'quantity':<40}" + "".join(f"{name:>26}" for name in names))
            for key in keys:
                say(f"  {key:<40}" + "".join(
                    f"{(floors.get(name) or comparisons.get(name)).get(key, float('nan')):>+26.4f}" for name in names))
        # the shear length of the wall at the first collision, from the control
        wall = base["wall"]
        shear_length = wall["edge_velocity_max"] / abs(wall["vorticity_peak"])
        rules = {}
        for run in MICROPOLAR:
            length = dipole.RUNS[run][1] * dipole.HALF_WIDTH
            displacement = dipole.COUPLING_RATIO * length / (2.0 * (1.0 + dipole.COUPLING_RATIO))
            rules[run] = {"micropolar_length": length, "displacement": displacement,
                          "displacement_over_shear_length": displacement / shear_length,
                          "shear_length_over_length": shear_length / length}
        say()
        say(f"shear length of the wall x = W at the first collision (control, t = {wall['time']:.4f}): reference velocity "
            f"{wall['edge_velocity_max']:.3f} m/s (largest |v| at {EDGE_DISTANCE:g} m from the wall, y = "
            f"{wall['edge_velocity_y']:.3f}), wall vorticity {wall['vorticity_peak']:.1f} 1/s at y = "
            f"{wall['vorticity_peak_y']:.3f}: Lambda = {shear_length:.5f} m")
        for run, values in rules.items():
            say(f"  {run}: l = {values['micropolar_length']:g}, delta* = {values['displacement']:.5f}, delta* / Lambda = "
                f"{values['displacement_over_shear_length']:.3f}, Lambda / l = {values['shear_length_over_length']:.3f}")
        results["_shear_length"] = {"value": shear_length, "wall": wall, "rules": rules}
        # the wall shear traction (mu + kappa) omega_w (w = 0 on the wall) over mu_inf omega_w of the control, and
        # the tangential momentum balance at the wall: (mu + kappa) d omega / dn - kappa d w / dn against
        # mu_inf d omega / dn of the control (both kinematic, over rho)
        base_wall = base["wall_common"]
        for run, values in floors.items():
            # a control has the viscosity of the control: traction and balance change as the vorticity and its flux
            values["wall_traction_peak"] = values.get("wall_vorticity_peak")
            values["wall_traction_integral"] = values.get("wall_vorticity_integral_upper")
            values["wall_momentum_balance"] = values.get("wall_flux_integral_upper")
        say()
        say("the wall x = W at the control's first enstrophy peak: shear traction and momentum balance")
        for run in MICROPOLAR:
            if run not in results:
                continue
            design = json.loads((_REPO_ROOT / f"cases/dipole/{run}/dipole.json").read_text(encoding="utf-8"))["design"]
            wall_run = results[run]["wall_common"]
            traction = (design["viscosity"] * wall_run["vorticity_peak"]) / (dipole.FAR_VISCOSITY * base_wall["vorticity_peak"])
            traction_integral = ((design["viscosity"] * wall_run["vorticity_integral_upper"])
                                 / (dipole.FAR_VISCOSITY * base_wall["vorticity_integral_upper"]))
            vorticity_term = design["viscosity"] * wall_run["flux_integral_upper"]
            spin_term = design["coupling"] * wall_run.get("spin_flux_integral_upper", float("nan"))
            control_term = dipole.FAR_VISCOSITY * base_wall["flux_integral_upper"]
            balance = {"traction_peak_change": traction - 1.0, "traction_integral_change": traction_integral - 1.0,
                       "vorticity_term": vorticity_term, "spin_term": spin_term, "control_term": control_term,
                       "balance_over_control": (vorticity_term - spin_term) / control_term,
                       "spin_term_share": -spin_term / vorticity_term,
                       "pressure_difference_over_control": wall_run["pressure_difference_along_wall"]
                       / base_wall["pressure_difference_along_wall"]}
            results[run]["wall_balance"] = balance
            comparisons[run]["wall_traction_peak"] = balance["traction_peak_change"]
            comparisons[run]["wall_traction_integral"] = balance["traction_integral_change"]
            comparisons[run]["wall_momentum_balance"] = balance["balance_over_control"] - 1.0
            say(f"  {run}: (mu + kappa) omega_w over mu_inf omega_w of the control {traction:.4f} (peak), "
                f"{traction_integral:.4f} (integral); (mu + kappa) d omega/dn = {vorticity_term:.3f}, kappa dw/dn = "
                f"{spin_term:.3f}, their difference over mu_inf d omega/dn of the control {balance['balance_over_control']:.4f} "
                f"(the pressure difference along the wall {balance['pressure_difference_over_control']:.4f}); the couple "
                f"term is {balance['spin_term_share']:.3f} of the vorticity term")
    checks = check_pairs(results, entries, say)
    out_dir = _REPO_ROOT / dipole.DATA_DIRECTORY
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dipole.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    def plain(value):
        if isinstance(value, dict):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        return value
    (out_dir / "dipole.json").write_text(json.dumps(
        plain({"benchmark": benchmark, "runs": results, "validation": validation, "floors": floors,
               "comparisons": comparisons, "checks": checks}), indent=1, default=float), encoding="utf-8")
    return 0


def command_checks(arguments) -> int:
    """Host checks of the set-up and of the estimators (no card): omega_e for E(0) = 2 and the reference values of
    the paper at t = 0, the largest initial speed, the Poisson pressure against the exact pressure of one monopole,
    the grid estimator on the initial lattice, and on a boundary layer of thickness 0.01 W (the wall vorticity of
    the grid and of the particle fit) -> data/micropolar/stage8/host_checks.{txt,json}."""
    from scipy.optimize import brentq
    lines, result = [], {}

    def say(text: str = "") -> None:
        lines.append(text)
        print(text, flush=True)

    def integrals(resolution: int, extremum: float) -> dict:
        spacing = 2.0 / resolution
        axis = -1.0 + spacing * (np.arange(resolution) + 0.5)
        x, y = np.meshgrid(axis, axis, indexing="ij")
        saved = dipole.EXTREMUM_VORTICITY
        dipole.EXTREMUM_VORTICITY = extremum
        fields = dipole.analytic_fields(x, y)
        dipole.EXTREMUM_VORTICITY = saved
        gradient_x, gradient_y = np.gradient(fields["vorticity"], spacing)
        return {"energy": 0.5 * float((fields["u"] ** 2 + fields["v"] ** 2).sum()) * spacing ** 2,
                "enstrophy": 0.5 * float((fields["vorticity"] ** 2).sum()) * spacing ** 2,
                "palinstrophy": 0.5 * float((gradient_x ** 2 + gradient_y ** 2).sum()) * spacing ** 2}
    extremum = brentq(lambda value: integrals(4000, value)["energy"] - 2.0, 250.0, 350.0, xtol=1e-12)
    at_extremum, at_320 = integrals(4000, extremum), integrals(4000, 320.0)
    result["initial"] = {"extremum_vorticity_for_energy_2": extremum, "at_extremum": at_extremum,
                         "at_320": at_320, "peak_initial_speed": dipole.peak_initial_speed()}
    say(f"1. omega_e giving E(0) = 2 on [-1, 1]^2 (grid of 4000^2 cells): {extremum:.9f}; there Omega(0) = "
        f"{at_extremum['enstrophy']:.4f}, P(0) = {at_extremum['palinstrophy']:.5e} (paper: E = 2, Omega = 800, "
        f"P = 4.42e5). With omega_e = 320 (the paper's 'about 320'): E {at_320['energy']:.4f}, Omega "
        f"{at_320['enstrophy']:.2f}, P {at_320['palinstrophy']:.4e}. Largest initial speed "
        f"{result['initial']['peak_initial_speed']:.4f} m/s (at the centre of the dipole).")
    saved = dipole.MONOPOLES
    dipole.MONOPOLES = ((0.0, 0.0, +1.0),)
    pressure_rows = {}
    for resolution in (250, 1000):
        axis, pressure = dipole.incompressible_pressure(0.0, 1.0 / resolution)
        x, y = np.meshgrid(axis, axis, indexing="ij")
        exact = -dipole.EXTREMUM_VORTICITY ** 2 * dipole.MONOPOLE_RADIUS ** 2 / 16.0 * np.exp(
            -2.0 * (x ** 2 + y ** 2) / dipole.MONOPOLE_RADIUS ** 2)
        pressure_rows[resolution] = {"largest_difference": float(np.abs(pressure - exact).max()),
                                     "depth": float(-exact.min())}
    dipole.MONOPOLES = saved
    axis, pressure = dipole.incompressible_pressure(0.0, 1.0 / 1000)
    result["pressure"] = {"one_monopole": pressure_rows, "dipole_range": [float(pressure.min()), float(pressure.max())]}
    say("2. Pressure / rho of the incompressible flow (cosine transform on the grid of dx / 2): one monopole against "
        "-omega_e^2 r0^2 / 16 exp(-2 r^2 / r0^2): largest difference " + ", ".join(
            f"{row['largest_difference']:.1e} at n = {resolution}" for resolution, row in pressure_rows.items())
        + f" (depth {pressure_rows[1000]['depth']:.2f} m^2/s^2); the dipole: {pressure.min():.3f} to "
        f"{pressure.max():.3f} m^2/s^2.")
    lattice = {}
    for resolution in (250, 500, 1000):
        spacing = 1.0 / resolution
        coordinate = -1.0 + spacing * (np.arange(2 * resolution) + 0.5)
        x, y = np.meshgrid(coordinate, coordinate, indexing="ij")
        position = np.column_stack([x.ravel(), y.ravel()])
        fields = dipole.analytic_fields(position[:, 0], position[:, 1])
        values = dipole.diagnostics(dipole.Grid(resolution), position, np.column_stack([fields["u"], fields["v"]]),
                                    np.zeros(position.shape[0]), None, spacing ** 2, 600.0, False)
        lattice[resolution] = {key: float(values[key]) for key in ("energy", "energy_grid", "enstrophy", "palinstrophy")}
    result["lattice"] = lattice
    say("3. The grid estimator on the initial lattice (E of the particles; E, Omega, P of the grid): " + "; ".join(
        f"n = {resolution}: E {row['energy']:.6f} / {row['energy_grid']:.6f}, Omega {row['enstrophy']:.3f}, P "
        f"{row['palinstrophy']:.4e}" for resolution, row in lattice.items()) + ".")
    layer = {}
    thickness, speed = 0.01, 5.0
    for resolution in (500, 1000):
        spacing = 1.0 / resolution
        coordinate = -1.0 + spacing * (np.arange(2 * resolution) + 0.5)
        x, y = np.meshgrid(coordinate, coordinate, indexing="ij")
        position = np.column_stack([x.ravel(), y.ravel()])
        distance = 1.0 - position[:, 0]
        envelope = 0.5 * (1.0 + np.tanh((position[:, 0] - 0.5) / 0.05))
        v = speed * (1.0 - np.exp(-distance / thickness)) * np.sin(np.pi * position[:, 1]) \
            * (position[:, 1] > 0.0) * envelope
        values = dipole.diagnostics(dipole.Grid(resolution), position, np.column_stack([np.zeros_like(v), v]),
                                    np.zeros(v.size), None, spacing ** 2, 600.0, False)
        fine = np.linspace(0.0, 1.0, 200001)
        smooth = 0.5 * (1.0 + np.tanh((1.0 - fine - 0.5) / 0.05))
        derivative = np.gradient(speed * (1.0 - np.exp(-fine / thickness)) * smooth, fine)
        exact = 0.5 * np.trapezoid(derivative ** 2, fine) * 0.5
        middle_grid = values["wall_vorticity"][np.argmin(np.abs(coordinate - 0.5))]
        middle_fit = values["wall_fit_vorticity"][np.argmin(np.abs(values["wall_fit_y"] - 0.5))]
        layer[resolution] = {"enstrophy_over_exact": float(values["enstrophy"] / exact),
                             "grid_wall_vorticity_over_exact": float(middle_grid / (-speed / thickness)),
                             "fit_wall_vorticity_over_exact": float(middle_fit / (-speed / thickness))}
    result["boundary_layer"] = layer
    say("4. A boundary layer v = 5 (1 - exp(-s / 0.01)) sin(pi y) next to x = 1: enstrophy over exact, wall vorticity "
        "of the grid and of the particle fit over exact: " + "; ".join(
            f"n = {resolution}: {row['enstrophy_over_exact']:.4f}, {row['grid_wall_vorticity_over_exact']:.4f}, "
            f"{row['fit_wall_vorticity_over_exact']:.4f}" for resolution, row in layer.items()) + ".")
    out_dir = _REPO_ROOT / dipole.DATA_DIRECTORY
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "host_checks.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (out_dir / "host_checks.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return 0


FIELD_WINDOW = ((0.4, 1.0), (0.0, 0.6))     # x, y of the cropped fields (the window of Fig. 7 of the paper)


def command_fields(arguments) -> int:
    """The vorticity (and w - omega / 2) of the runs at the times, cropped to FIELD_WINDOW, as float16 files of
    data/micropolar/stage8/fields/ for the pack."""
    out_dir = _REPO_ROOT / dipole.DATA_DIRECTORY / "fields"
    out_dir.mkdir(parents=True, exist_ok=True)
    for run in arguments.runs:
        directory = _REPO_ROOT / dipole.RUNS_DIRECTORY / run
        record = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        timestep, resolution = record["timestep"], record["dipole"]["resolution"]
        grid = dipole.Grid(resolution)
        columns = np.flatnonzero((grid.axis >= FIELD_WINDOW[0][0]) & (grid.axis <= FIELD_WINDOW[0][1]))
        rows = np.flatnonzero((grid.axis >= FIELD_WINDOW[1][0]) & (grid.axis <= FIELD_WINDOW[1][1]))
        for time in arguments.times:
            step = int(round(time / timestep / 1000.0)) * 1000
            source = directory / "fields" / f"fields_{step:08d}.npz"
            with np.load(source) as archive:
                arrays = {name: np.array(archive[name])[np.ix_(columns, rows)]
                          for name in ("vorticity", "spin_departure") if name in archive.files}
            target = out_dir / f"{run}_t{time:.3f}.npz"
            np.savez_compressed(target, x=grid.axis[columns], y=grid.axis[rows], time=step * timestep, run=run,
                                **arrays)
            print(f"[dipole] {target.relative_to(_REPO_ROOT)}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    analyse = commands.add_parser("analyse")
    analyse.add_argument("--partial", action="store_true", help="include unfinished runs (samples_partial.npz)")
    commands.add_parser("checks")
    fields = commands.add_parser("fields")
    fields.add_argument("--runs", nargs="+", required=True)
    fields.add_argument("--times", type=float, nargs="+", required=True)
    arguments = parser.parse_args()
    if arguments.command == "analyse":
        return command_analyse(arguments)
    if arguments.command == "fields":
        return command_fields(arguments)
    if arguments.command == "checks":
        return command_checks(arguments)
    return 1


if __name__ == "__main__":
    sys.exit(main())
