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
DETACHED = 0.015                       # a negative extremum this far from x = W counts as detached (secondary)
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
    """Largest value inside the window, refined by the parabola through it and its two neighbours."""
    inside = np.flatnonzero((times >= window[0]) & (times <= window[1]))
    if inside.size < 3:
        return float("nan"), float("nan")
    index = inside[np.argmax(values[inside])]
    if index == 0 or index == times.size - 1:
        return float(times[index]), float(values[index])
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


def track(times: np.ndarray, table: np.ndarray, start_time: float, start_choice, reach: float = 0.06):
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
            chosen = valid[np.argmin(distance)]
        previous = (chosen[1], chosen[2])
        track_time.append(time)
        track_value.append(chosen[0])
        track_x.append(chosen[1])
        track_y.append(chosen[2])
    return {"time": np.asarray(track_time), "x": np.asarray(track_x), "y": np.asarray(track_y),
            "value": np.asarray(track_value)}


def summary(entry: dict) -> dict:
    samples, record = entry["samples"], entry["record"]
    times = samples["time"]
    initial = record.get("initial", {})
    spacing = record["dipole"]["spacing"]
    first, second = (peak(times, samples["enstrophy"], window) for window in dipole.WINDOWS)
    palinstrophy_first, palinstrophy_second = (peak(times, samples["palinstrophy"], window)
                                               for window in dipole.WINDOWS)
    primary = track(times, samples["positive_extrema"], 0.0, lambda rows: rows[0])
    # the secondary vortex: the strongest negative extremum of the upper half detached from the wall after the
    # first collision
    secondary = track(times, samples["negative_extrema"], first[0],
                      lambda rows: next((row for row in rows if dipole.HALF_WIDTH - row[1] > DETACHED), None))
    lower = track(times, samples["lower_negative_extrema"], 0.0, lambda rows: rows[0])
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
            later = np.flatnonzero((primary["time"] > primary["time"][index]) & (primary["time"] <= second[0]))
            if later.size:
                rebound = later[np.argmax(distance[later])]
                result["rebound"] = [float(primary["time"][rebound]), float(distance[rebound]),
                                     float(primary["y"][rebound])]
    result["tracks"] = {name: {key: values.tolist() for key, values in path.items()}
                        for name, path in (("primary", primary), ("secondary", secondary), ("lower", lower))}
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
    return {"time": float(samples["time"][index]),
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
    for key in ("closest_approach", "rebound"):
        if key in run and key in base:
            out[f"{key}_time"] = run[key][0] - base[key][0]
            out[f"{key}_distance"] = run[key][1] - base[key][1]
            out[f"{key}_distance_relative"] = relative(run[key][1], base[key][1])
    for key in ("vorticity_peak", "vorticity_integral_upper", "flux_integral_upper", "flux_peak",
                "pressure_difference_along_wall"):
        out[f"wall_{key}"] = relative(run["wall"][key], base["wall"][key])
    for time in ("0.25", "0.50", "0.75"):
        if time in run["global"] and time in base["global"]:
            out[f"energy_{time}"] = relative(run["global"][time][0], base["global"][time][0])
            out[f"enstrophy_{time}"] = relative(run["global"][time][1], base["global"][time][1])
    return out


def command_analyse(arguments) -> int:
    lines = []

    def say(text: str = "") -> None:
        lines.append(text)
        print(text, flush=True)

    results = {}
    for run in dipole.RUNS:
        entry = load(run, partial=arguments.partial)
        if entry is not None:
            results[run] = summary(entry)
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
               "comparisons": comparisons}), indent=1, default=float), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    analyse = commands.add_parser("analyse")
    analyse.add_argument("--partial", action="store_true", help="include unfinished runs (samples_partial.npz)")
    arguments = parser.parse_args()
    if arguments.command == "analyse":
        return command_analyse(arguments)
    return 1


if __name__ == "__main__":
    sys.exit(main())
