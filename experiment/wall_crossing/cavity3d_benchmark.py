"""
cavity3d_benchmark.py - claude/wall_store_report.md section 14 (round 8): the 3-D cubic cavity at Re = 1000 against
the original tables of Albensoeder & Kuhlmann (2005): reference data, the extrema and profile errors of a run's MLS
centre lines (box_run.py --lines-time, cavity3d_lines.py), and the re-evaluation of the earlier runs (3.1).

    .venv/Scripts/python.exe -m experiment.wall_crossing.cavity3d_benchmark old        # 3.1: the earlier runs
    .venv/Scripts/python.exe -m experiment.wall_crossing.cavity3d_benchmark lines --run <dir> [--window 35 45]

Reference (experiment/validation/reference/cavity3d_re1000_albensoeder_kuhlmann_2005_original_*.csv, our
coordinates: lid = the face y = 1 moving in +x): table 10's extrema (u_min on the vertical line, v_min / v_max on the
horizontal line, |w| on the spanwise line at two positions), tables 5 / 6's 15 inner points of the two lines (velocity
and p / (rho U^2) minus the centre value).

Extrema: the time-mean profile on the dx / 8 points; the discrete extremum inside the search range, refined by the
parabola through it and its two neighbours (its vertex). Search ranges (normalised coordinate): u_min y in
[0.02, 0.5], v_max x in [0.02, 0.5], v_min x in [0.5, 0.98]; |w| on z in [0.02, 0.5] and [0.5, 0.98] (the two
sides; their mean and difference are given). Deviation = (|ours| - |reference|) / |reference| (+ = stronger).
Profile L2: the root mean square over the 15 inner points of (time-mean value - reference), velocity in lid speeds,
pressure p / (rho U^2) with each sample's centre pressure subtracted (rho = the rest density, U = 1).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

REFERENCE_DIRECTORY = REPOSITORY_ROOT / "experiment" / "validation" / "reference"
REFERENCE_PREFIX = "cavity3d_re1000_albensoeder_kuhlmann_2005_original_"
REST_DENSITY = 1000.0
LID_SPEED = 1.0
SEARCH = {"u_min": ("vertical", 0, "min", (0.02, 0.5)), "v_max": ("horizontal", 1, "max", (0.02, 0.5)),
          "v_min": ("horizontal", 1, "min", (0.5, 0.98))}
SPANWISE_SIDES = ((0.02, 0.5), (0.5, 0.98))


def read_table(path: pathlib.Path) -> dict:
    """A csv with '#' comment lines and one header line: {column: list of str}."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    header = lines[0].split(",")
    rows = [line.split(",") for line in lines[1:]]
    return {name: [row[index] if index < len(row) else "" for row in rows] for index, name in enumerate(header)}


def reference() -> dict:
    vertical = read_table(REFERENCE_DIRECTORY / f"{REFERENCE_PREFIX}vertical.csv")
    horizontal = read_table(REFERENCE_DIRECTORY / f"{REFERENCE_PREFIX}horizontal.csv")
    extrema_table = read_table(REFERENCE_DIRECTORY / f"{REFERENCE_PREFIX}extrema.csv")
    extrema = {}
    for index, name in enumerate(extrema_table["quantity"]):
        extrema[name] = {"value": float(extrema_table["value"][index]),
                         "position": float(extrema_table["position"][index]),
                         "position_2": float(extrema_table["position_2"][index]) if extrema_table["position_2"][index]
                         else None}
    as_array = {key: np.asarray([float(value) for value in values]) for key, values in vertical.items()}
    as_array_h = {key: np.asarray([float(value) for value in values]) for key, values in horizontal.items()}
    return {"vertical": {"coordinate": as_array["y"], "velocity": as_array["u"], "pressure": as_array["pressure"]},
            "horizontal": {"coordinate": as_array_h["x"], "velocity": as_array_h["v"],
                           "pressure": as_array_h["pressure"]},
            "extrema": extrema}


def inner(coordinate: np.ndarray) -> np.ndarray:
    """The 15 inner table points (not the wall rows 0 and 1)."""
    return (coordinate > 1e-9) & (coordinate < 1.0 - 1e-9)


def reference_points() -> dict:
    """The coordinates box_run samples besides the dx / 8 points: the table points of the two lines, and on the
    spanwise line table 10's |w| positions."""
    data = reference()
    spanwise = data["extrema"]["w_abs_max"]
    return {"vertical": data["vertical"]["coordinate"], "horizontal": data["horizontal"]["coordinate"],
            "spanwise": np.asarray([spanwise["position"], spanwise["position_2"]])}


def parabola_extremum(coordinates: np.ndarray, values: np.ndarray, kind: str, search: tuple) -> tuple[float, float]:
    """(position, value): the discrete extremum inside search refined by the parabola through it and its two
    neighbours (the discrete point when the three are not convex / concave or the vertex leaves them)."""
    mask = np.isfinite(values) & (coordinates >= search[0]) & (coordinates <= search[1])
    candidates = np.flatnonzero(mask)
    index = candidates[np.argmin(values[candidates]) if kind == "min" else np.argmax(values[candidates])]
    if index == 0 or index == coordinates.size - 1:
        return float(coordinates[index]), float(values[index])
    x0, x1, x2 = coordinates[index - 1:index + 2]
    y0, y1, y2 = values[index - 1:index + 2]
    denominator = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denominator
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denominator
    c = (x1 * x2 * (x1 - x2) * y0 + x2 * x0 * (x2 - x0) * y1 + x0 * x1 * (x0 - x1) * y2) / denominator
    if (kind == "min" and a <= 0) or (kind == "max" and a >= 0):
        return float(x1), float(y1)
    vertex = -b / (2 * a)
    if not x0 <= vertex <= x2:
        return float(x1), float(y1)
    return float(vertex), float(c - b * b / (4 * a))


def deviation(value: float, reference_value: float) -> float:
    return (abs(value) - abs(reference_value)) / abs(reference_value)


def extrema_of_profiles(profiles: dict) -> dict:
    """profiles: {"vertical": (coordinate, u), "horizontal": (coordinate, v), "spanwise": (coordinate, w)} (fine
    points). Returns {name: {"value", "position", "deviation"}} with the table-10 quantities; w_abs_max holds both
    sides ("sides": [(position, |w|, w)], value = their mean |w|, "side_difference")."""
    data = reference()["extrema"]
    out = {}
    for name, (line, _component, kind, search) in SEARCH.items():
        coordinate, values = profiles[line]
        position, value = parabola_extremum(coordinate, values, kind, search)
        out[name] = {"value": value, "position": position, "deviation": deviation(value, data[name]["value"]),
                     "position_error": position - data[name]["position"]}
    coordinate, values = profiles["spanwise"]
    sides = []
    for search in SPANWISE_SIDES:
        position, magnitude = parabola_extremum(coordinate, np.abs(values), "max", search)
        nearest = int(np.argmin(np.abs(coordinate - position)))
        sides.append({"position": position, "magnitude": magnitude, "sign": float(np.sign(values[nearest]))})
    mean = 0.5 * (sides[0]["magnitude"] + sides[1]["magnitude"])
    out["w_abs_max"] = {"value": mean, "deviation": deviation(mean, data["w_abs_max"]["value"]), "sides": sides,
                        "side_difference": sides[0]["magnitude"] - sides[1]["magnitude"],
                        "position": sides[0]["position"], "position_2": sides[1]["position"],
                        "position_error": sides[0]["position"] - data["w_abs_max"]["position"],
                        "position_2_error": sides[1]["position"] - data["w_abs_max"]["position_2"]}
    return out


def profile_errors(coordinate: np.ndarray, values: np.ndarray, reference_coordinate: np.ndarray,
                   reference_values: np.ndarray) -> dict:
    """L2 / Linf over the 15 inner table points; values are at the table points already (same order) or are
    interpolated linearly from coordinate when the two differ."""
    keep = inner(reference_coordinate)
    if coordinate.shape == reference_coordinate.shape and np.allclose(coordinate, reference_coordinate):
        ours = values
    else:
        ours = np.interp(reference_coordinate, coordinate, values)
    difference = ours[keep] - reference_values[keep]
    return {"l2": float(np.sqrt(np.mean(difference ** 2))), "linf": float(np.max(np.abs(difference))),
            "points": int(keep.sum()), "values": ours.tolist()}


# ---- lines.npz of a run ------------------------------------------------------------------------------------------

def load_lines(path: pathlib.Path) -> dict:
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def window_mask(lines: dict, window: tuple) -> np.ndarray:
    return (lines["time"] >= window[0] - 1e-9) & (lines["time"] <= window[1] + 1e-9)


def analyse_lines(lines: dict, window: tuple) -> dict:
    """Extrema, profile L2 (velocity, pressure) of the time-mean lines over window."""
    mask = window_mask(lines, window)
    fine = int(lines["fine_count"])
    data = reference()
    component = {"vertical": 0, "horizontal": 1, "spanwise": 2}
    mean = {}
    for name in ("vertical", "horizontal", "spanwise"):
        velocity = lines[f"{name}_velocity"][mask].astype(np.float64)[:, :, component[name]]
        pressure = (lines[f"{name}_pressure"][mask].astype(np.float64)
                    - lines["centre_pressure"][mask][:, None]) / (REST_DENSITY * LID_SPEED ** 2)
        mean[name] = {"coordinate": lines[f"{name}_coordinate"], "velocity": velocity.mean(axis=0),
                      "pressure": pressure.mean(axis=0)}
    profiles = {name: (mean[name]["coordinate"][:fine], mean[name]["velocity"][:fine]) for name in mean}
    result = {"window": list(window), "samples": int(mask.sum()),
              "times": [float(lines["time"][mask].min()), float(lines["time"][mask].max())] if mask.any() else None,
              "extrema": extrema_of_profiles(profiles)}
    for name in ("vertical", "horizontal"):
        table_coordinate = mean[name]["coordinate"][fine:]
        result[f"{name}_velocity_error"] = profile_errors(table_coordinate, mean[name]["velocity"][fine:],
                                                          data[name]["coordinate"], data[name]["velocity"])
        result[f"{name}_pressure_error"] = profile_errors(table_coordinate, mean[name]["pressure"][fine:],
                                                          data[name]["coordinate"], data[name]["pressure"])
    result["mean_profiles"] = {name: {"coordinate": mean[name]["coordinate"][:fine].tolist(),
                                      "velocity": mean[name]["velocity"][:fine].tolist(),
                                      "pressure": mean[name]["pressure"][:fine].tolist()} for name in mean}
    result["fallbacks"] = {key: int(lines[key][mask].sum()) for key in lines if "fallback" in key}
    return result


# ---- 3.1: the earlier runs ---------------------------------------------------------------------------------------

OLD_RUNS = [
    # (label, method, N, run directory under logs/, case for the walls and the box)
    ("crossing N = 50 (s25)", "crossing", 50, "wall_crossing/box_run/s25_cavity3d_n50_crossing", "cases/cavity3d_n50_crossing"),
    ("crossing N = 70 (s25)", "crossing", 70, "wall_crossing/box_run/s25_cavity3d_n70_crossing", "cases/cavity3d_n70_crossing"),
    ("crossing N = 100 (s24, particle pool)", "crossing", 100, "wall_crossing/box_run/s24_cavity3d_full_integral",
     "cases/cavity3d_1m_crossing_integral_split"),
    ("crossing N = 100 (wall store, round 3, remote)", "crossing", 100,
     "wall_store/round3/remote/store/box_run/r3_cavity3d_1m_store", "cases/cavity3d_1m_crossing"),
    ("crossing N = 100 (final defaults, round 5, remote)", "crossing", 100,
     "wall_store/round5/remote/box_run/r5_cavity3d_1m", "cases/cavity3d_1m_crossing"),
    ("simple N = 50 (s25)", "simple", 50, "wall_crossing/box_run/s25_cavity3d_n50_simple", "cases/cavity3d_n50"),
    ("simple N = 70 (s25)", "simple", 70, "wall_crossing/box_run/s25_cavity3d_n70_simple", "cases/cavity3d_n70"),
    ("simple N = 100 (section 19)", "simple", 100, "wall_crossing/box_run/cavity3d_1m_simple_long", "cases/cavity3d_1m"),
]


def old_window_profiles(profiles: list, window: tuple) -> dict:
    chosen = [entry for entry in profiles if window[0] - 1e-9 <= entry["time"] <= window[1] + 1e-9]
    out = {"count": len(chosen), "times": [chosen[0]["time"], chosen[-1]["time"]] if chosen else None}
    for name in ("u_along_y", "v_along_x"):
        out[name] = (np.asarray(chosen[0][name]["coordinate"]),
                     np.mean([np.asarray(entry[name]["velocity"], dtype=np.float64) for entry in chosen], axis=0))
    return out


def snapshot_states(run_directory: pathlib.Path, case) -> list:
    """The saved fluid states of an old run: final.npz (t = end) and the kept checkpoint(s), as read_fluid_state
    dicts (masses = rest density x the material's volume where the file has none)."""
    from experiment.v7.utils.case_v7 import KIND_FLUID
    fluid_material = next(index for index, material in enumerate(case.materials) if material.kind == KIND_FLUID)
    mass = float(case.materials[fluid_material].rest_density) * float(case.materials[fluid_material].volume)
    states = []
    for checkpoint in sorted((run_directory / "checkpoints").glob("c*.npz")):
        with np.load(checkpoint) as archive:
            position = archive["position_voxel_id"]
            velocity = archive["velocity_mass"]
            density = archive["density_pressure"]
            material = archive["material"]
            offset = float(archive["_stored_density_offset"]) if "_stored_density_offset" in archive.files else 0.0
        kinds = np.asarray([entry.kind for entry in case.materials])
        fluid = kinds[np.minimum(material.astype(np.int64), kinds.size - 1)] == KIND_FLUID
        step = int(checkpoint.stem[1:])
        states.append({"label": f"checkpoint step {step}", "time": step * float(case.physics.timestep),
                       "positions": position[fluid, :3].astype(np.float64),
                       "velocities": velocity[fluid, :3].astype(np.float64),
                       "masses": velocity[fluid, 3].astype(np.float64),
                       "densities": density[fluid, 0].astype(np.float64) + offset,
                       "pressures": density[fluid, 1].astype(np.float64)})
    final = run_directory / "final.npz"
    if final.exists():
        with np.load(final) as archive:
            summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
            states.append({"label": "final.npz", "time": float(summary["time"]),
                           "positions": archive["positions"].astype(np.float64),
                           "velocities": archive["velocities"].astype(np.float64),
                           "masses": np.full(archive["positions"].shape[0], mass),
                           "densities": archive["densities"].astype(np.float64),
                           "pressures": archive["pressures"].astype(np.float64)})
    return states


def old_main(arguments) -> int:
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.wall_crossing import cavity3d_lines
    from experiment.wall_crossing.hydrostatic import initial_box
    data = reference()
    table10 = data["extrema"]
    results = []
    for label, method, size, directory, case_path in OLD_RUNS:
        run_directory = REPOSITORY_ROOT / "logs" / directory
        record = {"label": label, "method": method, "N": size, "run": directory}
        profiles = json.loads((run_directory / "profiles.json").read_text(encoding="utf-8"))
        window = old_window_profiles(profiles, (35.0, 45.0))
        coordinate_u, mean_u = window["u_along_y"]
        coordinate_v, mean_v = window["v_along_x"]
        record["window_profiles"] = window["count"]
        # (a) as reported then: the discrete extremum of the 101-point Shepard mean profile, against the tables'
        # grid-point extrema (tables 5 / 6)
        record["as_reported"] = {
            "u_min": float(np.nanmin(mean_u)), "u_min_position": float(coordinate_u[np.nanargmin(mean_u)]),
            "v_max": float(np.nanmax(mean_v)), "v_max_position": float(coordinate_v[np.nanargmax(mean_v)]),
            "v_min": float(np.nanmin(mean_v)), "v_min_position": float(coordinate_v[np.nanargmin(mean_v)])}
        grid = {"u_min": float(data["vertical"]["velocity"].min()), "v_max": float(data["horizontal"]["velocity"].max()),
                "v_min": float(data["horizontal"]["velocity"].min())}
        record["as_reported_deviation_grid"] = {key: deviation(record["as_reported"][key], grid[key]) for key in grid}
        record["as_reported_deviation_table10"] = {key: deviation(record["as_reported"][key], table10[key]["value"])
                                                    for key in grid}
        # (b) the same 101-point mean profiles, extremum refined by the 3-point parabola, against table 10
        refined = {}
        for key, (line, coordinate, values, kind, search) in {
                "u_min": ("vertical", coordinate_u, mean_u, "min", SEARCH["u_min"][3]),
                "v_max": ("horizontal", coordinate_v, mean_v, "max", SEARCH["v_max"][3]),
                "v_min": ("horizontal", coordinate_v, mean_v, "min", SEARCH["v_min"][3])}.items():
            position, value = parabola_extremum(coordinate, values, kind, search)
            refined[key] = {"value": value, "position": position, "deviation": deviation(value, table10[key]["value"]),
                            "position_error": position - table10[key]["position"]}
        record["refined_shepard101"] = refined
        # (c) the saved snapshots (MLS on dx / 8 lines, as the new runs): |w| and, for comparison, the other three
        case = load_case_v7((REPOSITORY_ROOT / case_path / "case.yaml").resolve())
        lower, upper, _radius, _dimension = initial_box(case)
        sampler = cavity3d_lines.sampler_of_case(case, lower, upper, reference_points(),
                                                  REPOSITORY_ROOT / case_path / "case.yaml")
        fine = sampler.fine.size
        snapshots = []
        for state in snapshot_states(run_directory, case):
            sampled = sampler.sample(state)
            profiles_fine = {name: (sampler.fine, sampled[name]["velocity"][:fine, index])
                             for name, index in (("vertical", 0), ("horizontal", 1), ("spanwise", 2))}
            snapshots.append({"label": state["label"], "time": state["time"],
                              "extrema": extrema_of_profiles(profiles_fine),
                              "fallbacks": int(sum(sampled[name]["fallback_velocity"].sum()
                                                   for name in ("vertical", "horizontal", "spanwise")))})
        record["snapshots"] = snapshots
        results.append(record)
        print(f"{label}: window {window['count']} profiles; snapshots {[item['label'] for item in snapshots]}",
              flush=True)
    out = REPOSITORY_ROOT / "logs" / "wall_store" / "round8" / "old_runs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8", newline="\n")
    print(f"-> {out}")
    return 0


def lines_main(arguments) -> int:
    lines = load_lines(pathlib.Path(arguments.run) / "lines.npz")
    result = analyse_lines(lines, tuple(arguments.window))
    result.pop("mean_profiles")
    print(json.dumps(result, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("old")
    lines_parser = sub.add_parser("lines")
    lines_parser.add_argument("--run", required=True)
    lines_parser.add_argument("--window", type=float, nargs=2, default=[35.0, 45.0])
    arguments = parser.parse_args()
    return {"old": old_main, "lines": lines_main}[arguments.command](arguments)


if __name__ == "__main__":
    sys.exit(main())
