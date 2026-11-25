"""
cavity3d_round9.py - tables and figures of claude/wall_store_report.md section 15 (round 9): the measurement error of
the 3-D cavity's extrema and the level-by-level refinement (cavity3d_extrema.py, cavity3d_refine.py, cavity3d_levels.py).

    python -m experiment.wall_crossing.cavity3d_round9 measurement     # section 15.1-15.2 (brief section 2)
    python -m experiment.wall_crossing.cavity3d_round9 levels          # section 15.3-15.5 (brief section 3, 5)

Inputs (logs/wall_store/round9/):
  frame_tubes/<hdx4|hdx3p5>_n<N>.npz  the centre-line tubes of round 8's saved frames (t = 35-45; cut on the remote);
  levels.json                         {"<N>": {"directory", "machine", "chain"}} of the round-9 levels (box_run dirs
                                      with tubes/, lines.npz, series.csv, convergence.jsonl, summary.json);
  smi_local.csv, remote/smi_remote.csv, levels.jsonl, remote/levels.jsonl.
The chosen method of the brief's 2.4 is METHOD ("fit", the direct least squares); the round-8 method ("linear") is
given beside it everywhere.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing import cavity3d_extrema as extrema_methods                     # noqa: E402
from experiment.wall_crossing.cavity3d_benchmark import (analyse_lines, inner, load_lines,   # noqa: E402
                                                         reference)
from experiment.wall_crossing.cavity_tables import markdown_table as plain_markdown_table    # noqa: E402

ROUND = REPOSITORY_ROOT / "logs" / "wall_store" / "round9"
TABLES = REPOSITORY_ROOT / "claude" / "wall_store" / "tables"
FIGURES = REPOSITORY_ROOT / "claude" / "wall_store" / "figures"
METHOD = "fit"
METHOD_TEXT = {"linear": "(i) 线性 MLS（现在的方法）", "quadratic": "(ii) 二次 MLS", "fit": "(iii) 直接拟合"}
QUANTITY_TEXT = {"u_min": "u_min", "v_max": "v_max", "v_min": "v_min", "w_abs_max": "|w|max"}
TRANSVERSE_CURVATURE_SOURCE = "hdx3p5_n100"
WINDOW = 4.0                      # the brief: a level's result = the mean over its last 4 units of level time
REST_DENSITY = 1000.0


def escape_bars(text) -> str:
    return re.sub(r"(?<!\\)\|", r"\\|", str(text))


def markdown_table(header: list, rows: list) -> str:
    return plain_markdown_table([escape_bars(cell) for cell in header],
                                [[escape_bars(cell) for cell in row] for row in rows])


def write(name: str, text: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / name).write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"-> {TABLES / name}")


def percent(value: float, digits: int = 2) -> str:
    return f"{100.0 * value:+.{digits}f} %"


def deviation(value: float, reference_value: float) -> float:
    return (abs(value) - abs(reference_value)) / abs(reference_value)


# ---------------------------------------------------------------------------------------------------------------------
# brief section 2: the measurement error
# ---------------------------------------------------------------------------------------------------------------------

FRAME_RUNS = [("hdx4_n50", 50, 4.0), ("hdx4_n70", 70, 4.0), ("hdx3p5_n50", 50, 3.5), ("hdx3p5_n70", 70, 3.5),
              ("hdx3p5_n100", 100, 3.5)]


def round9_level_tubes(size: int) -> list:
    """The tube records of a round-9 level in its result window (empty when the level is not there)."""
    levels = json.loads((ROUND / "levels.json").read_text(encoding="utf-8")) if (ROUND / "levels.json").exists() else {}
    entry = levels.get(str(size))
    if entry is None:
        return []
    directory = REPOSITORY_ROOT / entry["directory"]
    samples, _ = extrema_methods.tube_samples(directory / "tubes")
    end = max(sample["level_time"] for sample in samples)
    return [sample for sample in samples if sample["level_time"] >= end - WINDOW - 1e-9 and sample["level_time"] > 0]


def measurement_main() -> int:
    data = reference()["extrema"]
    curves, spline_second = extrema_methods.analytic_curves()
    source, _ = extrema_methods.tube_samples(ROUND / "frame_tubes" / f"{TRANSVERSE_CURVATURE_SOURCE}.npz")
    lower, upper = extrema_methods.box_of(100)
    curvatures = extrema_methods.transverse_curvatures(source, lower, upper, 0.01, 1.5 * 0.035 / float(upper[0] - lower[0]))
    scatter = extrema_methods.particle_scatter(source, lower, upper, 0.035, 0.01)
    moment = extrema_methods.kernel_second_moment()
    results = {"kernel_second_moment": moment, "spline_second_derivatives": spline_second,
               "transverse_curvatures": {key: list(value) for key, value in curvatures.items()},
               "particle_scatter": scatter, "analytic": [], "data": []}
    # analytic field: the lattice for every N, the real positions where a run has them
    rows = []
    for size in (50, 70, 100, 140, 200):
        lower, upper = extrema_methods.box_of(size)
        support = 4.0 / size
        sets = [("点阵", [{"positions": extrema_methods.lattice_tube(size, 1.5 * support),
                           "volumes": None}])]
        real = []
        if size in (50, 70):
            real, _ = extrema_methods.tube_samples(ROUND / "frame_tubes" / f"hdx4_n{size}.npz")
            sets.append((f"真实位置（第八轮帧，{len(real)} 个）", real))
        else:
            real = round9_level_tubes(size)
            if real:
                sets.append((f"真实位置（第九轮细管，{len(real)} 个）", real))
        theory = extrema_methods.theory_bias(size, 4.0, curvatures)
        for label, samples in sets:
            prepared = [{"positions": sample["positions"],
                         "volumes": sample["volumes"] if sample["volumes"] is not None
                         else np.ones(sample["positions"].shape[0])} for sample in samples]
            out = extrema_methods.run_methods(
                prepared, lower, upper, size, support,
                lambda quantity, sample: extrema_methods.analytic_values(sample["positions"], lower, upper, quantity,
                                                                         curves, curvatures))
            entry = {"N": size, "positions": label, "methods": {}}
            cells = [str(size), label]
            for quantity in extrema_methods.QUANTITIES:
                values = {method: deviation(out[quantity][method]["value"], data[quantity]["value"])
                          for method in ("linear", "quadratic", "fit")}
                entry["methods"][quantity] = values
                entry["methods"][quantity]["theory"] = theory[quantity]["total"]
                cells.append(f"{percent(values['linear'])}（估计 {percent(theory[quantity]['total'])}）/ "
                             f"{percent(values['quadratic'], 3)} / {percent(values['fit'], 3)}")
            results["analytic"].append(entry)
            rows.append(cells)
    write("round9_analytic.md", markdown_table(
        ["N（h/dx = 4）", "粒子位置", "u_min：(i)（½σ²f'' 估计）/ (ii) / (iii)", "v_max：同左", "v_min：同左"], rows))
    # the round-8 data (and round 9's N = 100 level) by the three methods
    accuracy = json.loads((REPOSITORY_ROOT / "logs" / "wall_store" / "round8" / "accuracy.json").read_text(
        encoding="utf-8"))
    rows = []
    entries = [(name, size, ratio, f"第八轮 crossing N = {size}（h/dx = {ratio:g}），帧",
                f"crossing_hdx{'4' if ratio == 4.0 else '3p5'}_n{size}") for name, size, ratio in FRAME_RUNS]
    entries.insert(2, ("level", 100, 4.0, "第九轮 N = 100 级（h/dx = 4），细管", None))
    for name, size, ratio, label, round8_label in entries:
        if name == "level":
            samples = round9_level_tubes(100)
            if not samples:
                continue
        else:
            samples, _ = extrema_methods.tube_samples(ROUND / "frame_tubes" / f"{name}.npz")
        lower, upper = extrema_methods.box_of(size)
        out = extrema_methods.run_methods(samples, lower, upper, size, ratio / size,
                                          lambda quantity, sample: sample["velocities"][:, extrema_methods.QUANTITIES[
                                              quantity][2]])
        entry = {"label": label, "N": size, "hdx": ratio, "samples": len(samples), "methods": {}}
        cells = [label, str(len(samples))]
        for quantity in extrema_methods.QUANTITIES:
            values = {method: deviation(out[quantity][method]["value"], data[quantity]["value"])
                      for method in ("linear", "quadratic", "fit")}
            if round8_label is not None:
                values["round8"] = accuracy[round8_label]["window"]["extrema"][quantity]["deviation"]
            entry["methods"][quantity] = values
            cells.append(" / ".join([percent(values["round8"]) if "round8" in values else "—"]
                                    + [percent(values[method]) for method in ("linear", "quadratic", "fit")]))
        results["data"].append(entry)
        rows.append(cells)
    write("round9_data_methods.md", markdown_table(
        ["运行", "样本", "u_min：第八轮报的 / (i) 重算 / (ii) / (iii)", "v_max：同左", "v_min：同左"], rows))
    write("round9_measurement_facts.md", markdown_table(
        ["量", "值"],
        [["Wendland C4 单方向二阶矩 σ² / H²", f"{moment:.4f}"],
         ["三次样条在极值处的 f''（u_min / v_max / v_min）",
          " / ".join(f"{spline_second[key]:.1f}" for key in ("u_min", "v_max", "v_min"))],
         ["解析场用的 f''（任务书的值）", "28 / −20 / 150"],
         ["横向曲率 k_a / k_b（第八轮 h/dx = 3.5 N = 100 的帧，拟合）",
          "；".join(f"{key} {value[0]:.1f} / {value[1]:.1f}" for key, value in curvatures.items())],
         ["粒子速度相对光滑场的散布（标准差，U）",
          " / ".join(f"{key} {value:.1e}" for key, value in scatter.items())]]))
    # the profiles at the 15 table points on the analytic field: linear MLS with walls (round 8), quadratic MLS with
    # walls and on the fluid alone (the wall particles carry the wall velocity: lid 1, the rest 0)
    rows = []
    for size in (100, 200):
        lower, upper = extrema_methods.box_of(size)
        support = 4.0 / size
        positions = extrema_methods.lattice_tube(size, 1.5 * support)
        walls = extrema_methods.boundary_particles(
            REPOSITORY_ROOT / f"cases/cavity3d_re1000_n{size}_hdx4_crossing/case.yaml", 1.0)
        near = (np.abs(walls[0]) <= 1.6 * support).sum(axis=1) >= 2
        wall_positions, wall_velocities = walls[0][near], walls[2][near]
        data_reference = reference()
        for line, along, component, quantity in (("vertical", 1, 0, "u_min"), ("horizontal", 0, 1, "v_max")):
            coordinate = data_reference[line]["coordinate"][inner(data_reference[line]["coordinate"])]
            points = extrema_methods.line_points(lower, upper, along, coordinate)
            field = extrema_methods.analytic_values(positions, lower, upper, quantity, curves, curvatures)
            exact = curves[line](coordinate)
            every = np.concatenate([positions, wall_positions])
            values = np.concatenate([field, wall_velocities[:, component]])
            variants = {"linear_walls": extrema_methods.moving_least_squares(every, np.ones(len(every)), values, points,
                                                                             support, 1),
                        "quadratic_walls": extrema_methods.moving_least_squares(every, np.ones(len(every)), values,
                                                                                points, support, 2),
                        "quadratic_fluid": extrema_methods.moving_least_squares(positions, np.ones(len(positions)),
                                                                                field, points, support, 2)}
            near_wall = np.abs(coordinate - 0.5) > 0.5 - 1.05 * support / float(upper[0] - lower[0])
            cells = [str(size), "竖线 u" if line == "vertical" else "横线 v"]
            for key, value in variants.items():
                error = value - exact
                near_text = f"{np.abs(error[near_wall]).max():.4f}" if near_wall.any() else "—"
                cells.append(f"{np.sqrt(np.mean(error ** 2)):.5f}（近壁 {near_text}，"
                             f"内部 {np.abs(error[~near_wall]).max():.4f}）")
            rows.append(cells)
    write("round9_analytic_profiles.md", markdown_table(
        ["N", "线", "(i) 线性 MLS，含壁粒子：L2（近壁最大 / 内部最大）", "(ii) 二次 MLS，含壁粒子", "(ii) 二次 MLS，只用流体"],
        rows))
    (ROUND / "measurement.json").write_text(json.dumps(results, indent=1), encoding="utf-8", newline="\n")
    print(f"-> {ROUND / 'measurement.json'}")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# brief section 3: the levels
# ---------------------------------------------------------------------------------------------------------------------

def level_entries() -> dict:
    return {int(key): value for key, value in
            json.loads((ROUND / "levels.json").read_text(encoding="utf-8")).items()}


def window_samples(samples: list, end: float, length: float) -> list:
    return [sample for sample in samples if end - length - 1e-9 <= sample["level_time"] <= end + 1e-9
            and sample["level_time"] > 0]


def table_points_profile(samples: list, lower, upper, support: float, walls: tuple, order: int) -> dict:
    """Mean velocity and pressure (minus the centre's, / rho U^2) at the 15 interior table points of both lines, by
    MLS of the given order (velocity: fluid and walls; pressure: fluid only)."""
    data = reference()
    out = {}
    centre = 0.5 * (lower + upper)
    for line, along, component in (("vertical", 1, 0), ("horizontal", 0, 1)):
        coordinate = data[line]["coordinate"][inner(data[line]["coordinate"])]
        points = extrema_methods.line_points(lower, upper, along, coordinate)
        velocities, pressures = [], []
        for sample in samples:
            positions = np.concatenate([sample["positions"], walls[0]])
            volumes = np.concatenate([sample["volumes"], walls[1]])
            values = np.concatenate([sample["velocities"][:, component], walls[2][:, component]])
            velocities.append(extrema_methods.moving_least_squares(positions, volumes, values, points, support, order))
            both = np.concatenate([points, centre[None, :]])
            pressure = extrema_methods.moving_least_squares(sample["positions"], sample["volumes"],
                                                            sample["pressures"], both, support, order)
            pressures.append((pressure[:-1] - pressure[-1]) / REST_DENSITY)
        mean_velocity, mean_pressure = np.mean(velocities, axis=0), np.mean(pressures, axis=0)
        reference_velocity = data[line]["velocity"][inner(data[line]["coordinate"])]
        reference_pressure = data[line]["pressure"][inner(data[line]["coordinate"])]
        out[line] = {"velocity_l2": float(np.sqrt(np.mean((mean_velocity - reference_velocity) ** 2))),
                     "pressure_l2": float(np.sqrt(np.mean((mean_pressure - reference_pressure) ** 2))),
                     "coordinate": coordinate.tolist(), "velocity": mean_velocity.tolist(),
                     "pressure": mean_pressure.tolist()}
    return out


def fine_profiles(samples: list, lower, upper, spacings: int, support: float, walls: tuple) -> dict:
    """Quadratic-MLS mean profiles of u on the vertical and v on the horizontal line every dx / 2 (for the figure)."""
    out = {}
    for line, along, component in (("vertical", 1, 0), ("horizontal", 0, 1)):
        coordinate = np.arange(2 * (spacings + 1) + 1) / (2 * (spacings + 1))
        points = extrema_methods.line_points(lower, upper, along, coordinate)
        profiles = []
        for sample in samples:
            positions = np.concatenate([sample["positions"], walls[0]])
            volumes = np.concatenate([sample["volumes"], walls[1]])
            values = np.concatenate([sample["velocities"][:, component], walls[2][:, component]])
            profiles.append(extrema_methods.moving_least_squares(positions, volumes, values, points, support, 2))
        out[line] = {"coordinate": coordinate.tolist(), "velocity": np.nanmean(profiles, axis=0).tolist()}
    return out


def extrema_of(samples: list, lower, upper, size: int, support: float) -> dict:
    """The chosen method and the linear one on samples: {quantity: {"fit", "quadratic", "linear"}} incl. w_abs_max."""
    quantities = list(extrema_methods.QUANTITIES) + list(extrema_methods.SPANWISE)
    out = extrema_methods.run_methods(samples, lower, upper, size, support,
                                      lambda quantity, sample: sample["velocities"][:, extrema_methods.specification(
                                          quantity)[2]], quantities=quantities)
    out["w_abs_max"] = {method: {"value": 0.5 * (abs(out["w_first"][method]["value"])
                                                 + abs(out["w_second"][method]["value"])),
                                 "side_difference": abs(out["w_first"][method]["value"])
                                 - abs(out["w_second"][method]["value"])}
                        for method in ("linear", "quadratic", "fit")}
    return out


def log_number(path: pathlib.Path, pattern: str):
    if not path.exists():
        return None
    match = re.search(pattern, path.read_text(encoding="utf-8", errors="replace"))
    return match.group(1).replace(",", "") if match else None


def smi_memory(path: pathlib.Path, start: float, end: float) -> dict:
    """Median memory.used (MiB) per GPU index while start < unix < end (the first 60 s skipped)."""
    if not path.exists():
        return {}
    values = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, skipinitialspace=True):
            try:
                unix = float(row["unix"])
            except (KeyError, ValueError):
                continue
            if start + 60.0 < unix < end:
                values.setdefault(int(row["index"]), []).append(float(row["memory.used"]))
    return {index: float(np.median(memory)) for index, memory in values.items()}


def level_main() -> int:
    data = reference()["extrema"]
    entries = level_entries()
    results = {}
    for size, entry in sorted(entries.items()):
        directory = REPOSITORY_ROOT / entry["directory"]
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        samples, _ = extrema_methods.tube_samples(directory / "tubes")
        lower, upper = np.asarray(summary["box_lower"]), np.asarray(summary["box_upper"])
        support = 4.0 / size
        end = float(summary["level_time"])
        window = window_samples(samples, end, WINDOW)
        halves = [window_samples(samples, end - WINDOW / 2, WINDOW / 2), window_samples(samples, end, WINDOW / 2)]
        volume = float(np.median(window[-1]["volumes"]))
        walls = extrema_methods.boundary_particles(REPOSITORY_ROOT / f"cases/cavity3d_re1000_n{size}_hdx4_crossing/case.yaml",
                                                   volume)
        tube = 1.6 * support
        walls = tuple(array[(np.abs(walls[0]) <= tube).sum(axis=1) >= 2] for array in walls)   # near a centre line
        result = {"N": size, **entry, "summary": summary, "samples": len(window)}
        result["extrema"] = extrema_of(window, lower, upper, size, support)
        result["halves"] = [extrema_of(half, lower, upper, size, support) if half else None for half in halves]
        result["profiles"] = table_points_profile(window, lower, upper, support, walls, 2)
        result["fine_profiles"] = fine_profiles(window, lower, upper, size, support, walls)
        lines = load_lines(directory / "lines.npz")
        old = analyse_lines(lines, (float(summary["time"]) - WINDOW - 1e-6, float(summary["time"]) + 1e-6))
        old.pop("mean_profiles")
        result["old"] = old
        # history: the chosen method on every record (one record each)
        history = []
        for sample in samples:
            if sample["level_time"] <= 0:
                continue
            single = extrema_methods.run_methods([sample], lower, upper, size, support,
                                                 lambda quantity, item: item["velocities"][:, extrema_methods.QUANTITIES[
                                                     quantity][2]])
            history.append({"time": sample["time"], "level_time": sample["level_time"],
                            **{quantity: deviation(single[quantity][METHOD]["value"], data[quantity]["value"])
                               for quantity in extrema_methods.QUANTITIES},
                            **{f"{quantity}_linear": deviation(single[quantity]["linear"]["value"],
                                                               data[quantity]["value"])
                               for quantity in extrema_methods.QUANTITIES}})
        result["history"] = history
        # the start: the first records of the level (energy, speed, density) and the volume constraint's I
        series = list(csv.DictReader(open(directory / "series.csv", newline="", encoding="utf-8")))
        level_time_key = "level_time" if "level_time" in series[0] else "time"
        start_rows = []
        for target in (0.0, 0.1, 0.5, 1.0, 2.0):
            row = min(series, key=lambda item: abs(float(item[level_time_key]) - target))
            start_rows.append({"level_time": float(row[level_time_key]),
                               "specific_energy": 0.5 * float(row["speed_root_mean_square"]) ** 2,
                               "speed_maximum": float(row["speed_maximum"]),
                               "density_minimum": float(row["density_minimum"]),
                               "density_maximum": float(row["density_maximum"])})
        result["start_rows"] = start_rows
        first_unit = [row for row in series if 0.0 < float(row[level_time_key]) <= 1.0]
        later = [row for row in series if float(row[level_time_key]) > 2.0]
        result["density_range_first_unit"] = [min(float(row["density_minimum"]) for row in first_unit),
                                              max(float(row["density_maximum"]) for row in first_unit)] if first_unit else None
        result["density_range_later"] = [min(float(row["density_minimum"]) for row in later),
                                         max(float(row["density_maximum"]) for row in later)] if later else None
        result["initial_state"] = summary.get("initial_state")
        constraint = [json.loads(line) for line in (directory / "volume_constraint.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip() and "integral" in line]
        result["integral"] = {"first": constraint[0]["integral"] if constraint else None,
                              "last": constraint[-1]["integral"] if constraint else None,
                              "theta_maximum_after_1": max((abs(record["theta"]) for record in constraint
                                                            if record["time"] > 1.0), default=None),
                              "theta_last": constraint[-1]["theta"] if constraint else None,
                              "series": [[record["time"], record["integral"], record["theta"]] for record in constraint]}
        if samples and samples[0]["level_time"] == 0.0:
            first = extrema_methods.run_methods([samples[0]], lower, upper, size, support,
                                                lambda quantity, item: item["velocities"][:, extrema_methods.QUANTITIES[
                                                    quantity][2]])
            result["start_extrema"] = {quantity: deviation(first[quantity][METHOD]["value"], data[quantity]["value"])
                                       for quantity in extrema_methods.QUANTITIES}
        out_log = ROUND / ("remote" if entry.get("chain") == "remote" else "") / f"{pathlib.Path(entry['directory']).name}.out"
        result["wall_store"] = log_number(out_log, r"wall store: ([\d,]+) wall particles kept")
        result["wall_generated"] = log_number(out_log, r"kept of ([\d,]+)")
        span = (float(summary.get("unix_start", 0.0)), float(summary.get("unix_end", 0.0)))
        result["memory"] = entry.get("memory", {})
        results[size] = result
        print(f"[round9] level N = {size}: {len(window)} records in the window", flush=True)
    (ROUND / "levels_results.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8",
                                               newline="\n")
    continuations = continuation_analysis(results)
    long = long_run_analysis(results, continuations)
    if long is not None:
        long["continuations"] = continuations
        (ROUND / "long_run.json").write_text(json.dumps(long, indent=1, default=float), encoding="utf-8", newline="\n")
        print(f"[round9] long N = 100 run: reached t = {long['reached']:.2f}, windows of N = {sorted(long['windows'])}",
              flush=True)
    level_tables(results, long)
    level_figures(results, long)
    return 0


def fit_limit(sizes: np.ndarray, values: np.ndarray, errors: np.ndarray, samples: int = 4000, seed: int = 1) -> dict:
    """deviation = limit + C N^-p: least squares (exact for three sizes), and a Monte Carlo over the levels'
    uncertainties (normal, errors); returns the central fit and the spread (16th-84th percentile half-width)."""
    from scipy.optimize import least_squares

    def solve(observed):
        def residual(parameters):
            limit, coefficient, power = parameters
            return limit + coefficient * sizes.astype(float) ** (-power) - observed
        start = [observed[-1], (observed[0] - observed[-1]) * sizes[0] ** 1.5, 1.5]
        fit = least_squares(residual, start, bounds=([-1.0, -1e6, 0.1], [1.0, 1e6, 6.0]))
        return fit.x

    central = solve(values)
    generator = np.random.default_rng(seed)
    draws = np.array([solve(values + generator.normal(0.0, errors)) for _ in range(samples)])
    low, high = np.percentile(draws, [16, 84], axis=0)
    return {"limit": float(central[0]), "coefficient": float(central[1]), "power": float(central[2]),
            "limit_spread": float(0.5 * (high[0] - low[0])), "power_spread": float(0.5 * (high[2] - low[2])),
            "limit_range": [float(low[0]), float(high[0])], "power_range": [float(low[2]), float(high[2])]}


def at_bound(fit: dict) -> bool:
    return fit["power"] <= 0.1001 or fit["power"] >= 5.999


def power_text(fit: dict) -> str:
    """The order p with its 16-84 % range; a central fit at a bound of fit_limit (0.1 or 6) means the three levels
    do not converge like N^-p (the differences do not shrink), so there is no limit to give."""
    text = f"{fit['power']:.2f}（{fit['power_range'][0]:.2f}–{fit['power_range'][1]:.2f}）"
    if at_bound(fit):
        text += "；到边界，不像 N^−p 收敛"
    return text


DRIFT_RUN = REPOSITORY_ROOT / "logs" / "wall_store" / "round8" / "comparison" / "box_run" / \
    "0001_C_crossing_hdx4_n100_dual" / "lines.npz"
DRIFT_SPAN = (30.0, 45.0)
HISTORY_FROM = 20.0


def drift_rates() -> dict:
    """The slow drift after the level rule is met: the slope (deviation per unit time) of every sample's extrema
    (round-8 method, the same case as the N = 100 level: h/dx = 4, Re = 1000) over DRIFT_SPAN, by least squares."""
    from experiment.wall_crossing.cavity3d_benchmark import extrema_of_profiles
    lines = load_lines(DRIFT_RUN)
    fine = int(lines["fine_count"])
    coordinate = lines["vertical_coordinate"][:fine]
    data = reference()["extrema"]
    times, values = [], {key: [] for key in QUANTITY_TEXT}
    for index, time_value in enumerate(lines["time"]):
        if not DRIFT_SPAN[0] <= time_value <= DRIFT_SPAN[1]:
            continue
        found = extrema_of_profiles({"vertical": (coordinate, lines["vertical_velocity"][index, :fine, 0]),
                                     "horizontal": (coordinate, lines["horizontal_velocity"][index, :fine, 1]),
                                     "spanwise": (coordinate, lines["spanwise_velocity"][index, :fine, 2])})
        times.append(float(time_value))
        for key in QUANTITY_TEXT:
            values[key].append(deviation(found[key]["value"], data[key]["value"]))
    out = {}
    for key, series in values.items():
        slope, intercept = np.polyfit(times, series, 1)
        residual = np.asarray(series) - (slope * np.asarray(times) + intercept)
        error = float(residual.std(ddof=2) / (np.std(times) * math.sqrt(len(times))))
        out[key] = {"slope": float(slope), "slope_error": error, "samples": len(times)}
    return out


LONG_RUN = "L100_long"            # report 15.4: N = 100 from zero to t = 70 without the convergence stop
LONG_FROM = 32.0                  # the long run's drift is fitted from the N = 100 level's convergence on
PLATEAU_FROM = 44.0               # the long run levels off here (its 4-unit means, report 15.4)
# report 15.4, plateau check: levels resumed from their last checkpoint without the convergence stop (box_run --resume
# in a copy of the level's directory, until the round's deadline); relative to ROUND
CONTINUATIONS = {170: "remote/box_run/L170_cont", 200: "box_run/L200_cont"}
QUANTITIES_ALL = ("u_min", "v_max", "v_min", "w_abs_max")
# an interpolated start jumps toward the paper about 4 units into the level and relaxes back over the next ~8 (report
# 15.4, figure round9_history): delta(N) is taken from the windows that start at this level time or later
RELAXED_FROM = 12.0


def window_deviations(samples: list, lower, upper, size: int, support: float) -> dict:
    data = reference()["extrema"]
    values = extrema_of(samples, lower, upper, size, support)
    return {quantity: deviation(values[quantity][METHOD]["value"], data[quantity]["value"]) for quantity in QUANTITIES_ALL}


def continuation_analysis(results: dict) -> dict:
    """The plateau check: every continued level's 4-unit windows after its result window (accumulated time
    (end, end + 4], (end + 4, end + 8], ...; complete windows only), record by record values, and the drift rate over
    the level's records from level time 6 on plus the continuation's."""
    data = reference()["extrema"]
    out = {}
    for size, relative in sorted(CONTINUATIONS.items()):
        directory = ROUND / relative
        if size not in results or not (directory / "tubes").exists():
            continue
        summary = results[size]["summary"]
        end_level = float(summary["time"])
        lower, upper = np.asarray(summary["box_lower"]), np.asarray(summary["box_upper"])
        support = 4.0 / size
        samples, _ = extrema_methods.tube_samples(directory / "tubes")
        later = [sample for sample in samples if sample["time"] > end_level + 1e-6]
        if not later:
            continue
        reached = max(sample["time"] for sample in later)
        windows = []
        end = end_level + WINDOW
        while end <= reached + 1e-6:
            chunk = [sample for sample in later if end - WINDOW + 1e-9 < sample["time"] <= end + 1e-9]
            windows.append({"end": end, "records": len(chunk),
                            **window_deviations(chunk, lower, upper, size, support)})
            end += WINDOW
        series = []
        for sample in later:
            single = extrema_methods.run_methods([sample], lower, upper, size, support,
                                                 lambda quantity, item: item["velocities"][:, extrema_methods.QUANTITIES[
                                                     quantity][2]])
            series.append({"time": sample["time"], "level_time": sample["level_time"],
                           **{quantity: deviation(single[quantity][METHOD]["value"], data[quantity]["value"])
                              for quantity in extrema_methods.QUANTITIES},
                           **{f"{quantity}_linear": deviation(single[quantity]["linear"]["value"], data[quantity]["value"])
                              for quantity in extrema_methods.QUANTITIES}})
        rows = [row for row in results[size]["history"] if row["level_time"] >= 6.0] + series
        times = np.array([row["time"] for row in rows])
        rates = {}
        for quantity in ("u_min", "v_max", "v_min"):
            values = np.array([row[quantity] for row in rows])
            slope, intercept = np.polyfit(times, values, 1)
            residual = values - (slope * times + intercept)
            rates[quantity] = {"slope": float(slope),
                               "slope_error": float(residual.std(ddof=2) / (times.std() * math.sqrt(len(times)))),
                               "span": [float(times.min()), float(times.max())]}
        out[size] = {"reached": reached, "windows": windows, "series": series, "rates": rates,
                     "directory": str(directory.relative_to(REPOSITORY_ROOT)).replace("\\", "/")}
        print(f"[round9] continuation of N = {size}: reached t = {reached:.2f}, {len(windows)} complete windows",
              flush=True)
    return out


def long_run_analysis(results: dict, continuations: dict | None = None) -> dict | None:
    """The drift diagnostic: the long N = 100 run's extrema (chosen method) in the same accumulated-time window as every
    level's result window (and every continuation window), its plateau (non-overlapping 4-unit windows from
    PLATEAU_FROM: their mean and scatter), and record by record (its drift rate). deviation(N, t) = d_100(t) + delta(N):
    delta is the refinement effect at equal time, independent of the drift when the drift is common to the levels."""
    directory = ROUND / "box_run" / LONG_RUN
    if not (directory / "tubes").exists():
        return None
    data = reference()["extrema"]
    samples, _ = extrema_methods.tube_samples(directory / "tubes")
    summary = json.loads((REPOSITORY_ROOT / level_entries()[100]["directory"] / "summary.json").read_text(encoding="utf-8"))
    lower, upper = np.asarray(summary["box_lower"]), np.asarray(summary["box_upper"])
    support = 4.0 / 100
    reached = max(sample["time"] for sample in samples)
    if reached < LONG_FROM + WINDOW:
        print(f"[round9] long N = 100 run only at t = {reached:.2f}; not used yet", flush=True)
        return None
    out = {"reached": reached, "windows": {}, "series": [], "rates": {}, "continuation_windows": {}}
    last = extrema_of([sample for sample in samples if sample["time"] >= reached - WINDOW - 1e-9], lower, upper, 100,
                      support)
    out["last_window"] = {"end": reached, **{quantity: deviation(last[quantity][METHOD]["value"], data[quantity]["value"])
                                             for quantity in ("u_min", "v_max", "v_min", "w_abs_max")}}
    # non-overlapping 4-unit windows from HISTORY_FROM (the approach to the plateau and the plateau itself)
    evolution = []
    start = HISTORY_FROM
    while start + WINDOW <= reached + 1e-6:
        chunk = [sample for sample in samples if start - 1e-9 <= sample["time"] < start + WINDOW - 1e-9]
        evolution.append({"start": start, "records": len(chunk), "plateau": start >= PLATEAU_FROM - 1e-9,
                          **window_deviations(chunk, lower, upper, 100, support)})
        start += WINDOW
    out["evolution"] = evolution
    plateau = [row for row in evolution if row["plateau"]]
    out["plateau"] = {"windows": plateau,
                      "mean": {quantity: float(np.mean([row[quantity] for row in plateau])) for quantity in QUANTITIES_ALL},
                      "scatter": {quantity: float(np.std([row[quantity] for row in plateau], ddof=1))
                                  for quantity in QUANTITIES_ALL}}
    for size, continuation in sorted((continuations or {}).items()):
        out["continuation_windows"][size] = []
        for window in continuation["windows"]:
            end = window["end"]
            if end > reached + 1e-6:
                continue
            chunk = [sample for sample in samples if end - WINDOW + 1e-9 < sample["time"] <= end + 1e-9]
            out["continuation_windows"][size].append({"end": end, "records": len(chunk),
                                                      **window_deviations(chunk, lower, upper, 100, support)})
    for size, result in sorted(results.items()):
        end = float(result["summary"]["time"])
        if end > reached + 1e-6:
            continue
        window = [sample for sample in samples if end - WINDOW - 1e-9 <= sample["time"] <= end + 1e-9]
        halves = [[sample for sample in window if sample["time"] <= end - WINDOW / 2 + 1e-9],
                  [sample for sample in window if sample["time"] > end - WINDOW / 2 + 1e-9]]
        values = extrema_of(window, lower, upper, 100, support)
        half_values = [extrema_of(half, lower, upper, 100, support) for half in halves]
        out["windows"][size] = {
            "end": end, "records": len(window),
            **{quantity: deviation(values[quantity][METHOD]["value"], data[quantity]["value"])
               for quantity in ("u_min", "v_max", "v_min", "w_abs_max")},
            "halves": {quantity: deviation(half_values[1][quantity][METHOD]["value"], data[quantity]["value"])
                       - deviation(half_values[0][quantity][METHOD]["value"], data[quantity]["value"])
                       for quantity in ("u_min", "v_max", "v_min", "w_abs_max")}}
    for sample in samples:
        if sample["time"] < HISTORY_FROM:
            continue
        single = extrema_of([sample], lower, upper, 100, support)
        out["series"].append({"time": sample["time"], **{quantity: deviation(single[quantity][METHOD]["value"],
                                                                             data[quantity]["value"])
                                                         for quantity in ("u_min", "v_max", "v_min", "w_abs_max")}})
    # the drift rate before the plateau (from the N = 100 level's stop) and on it
    for span_key, (first, last_time) in (("approach", (LONG_FROM, PLATEAU_FROM)), ("plateau", (PLATEAU_FROM, reached))):
        rows = [row for row in out["series"] if first - 1e-9 <= row["time"] <= last_time + 1e-9]
        times = np.array([row["time"] for row in rows])
        out["rates"][span_key] = {}
        for quantity in ("u_min", "v_max", "v_min", "w_abs_max"):
            values = np.array([row[quantity] for row in rows])
            slope, intercept = np.polyfit(times, values, 1)
            residual = values - (slope * times + intercept)
            out["rates"][span_key][quantity] = {
                "slope": float(slope), "intercept": float(intercept),
                "slope_error": float(residual.std(ddof=2) / (times.std() * math.sqrt(len(times)))),
                "span": [float(times.min()), float(times.max())], "records": len(rows)}
    # each level's own drift after level time 6 (the earliest the rule may stop), same fit
    for size, result in sorted(results.items()):
        own = [row for row in result["history"] if row["level_time"] >= 6.0]
        if size == 100 or len(own) < 5:
            continue
        own_times = np.array([row["time"] for row in own])
        out.setdefault("level_rates", {})[size] = {}
        for quantity in ("u_min", "v_max", "v_min"):
            values = np.array([row[quantity] for row in own])
            slope, intercept = np.polyfit(own_times, values, 1)
            residual = values - (slope * own_times + intercept)
            out["level_rates"][size][quantity] = {
                "slope": float(slope), "slope_error": float(residual.std(ddof=2) / (own_times.std()
                                                                                   * math.sqrt(len(own_times)))),
                "span": [float(own_times.min()), float(own_times.max())]}
    return out


def level_tables(results: dict, long: dict | None = None) -> None:
    data = reference()["extrema"]
    rows = []
    for size, result in sorted(results.items()):
        summary = result["summary"]
        cells = [str(size), result.get("machine", ""), f"{summary['fluid_particles']:,}",
                 f"{int(result['wall_store']):,}（生成 {int(result['wall_generated']):,}）" if result["wall_store"] else "—",
                 " / ".join(f"{value:.0f}" for value in result["memory"].values()) if result["memory"] else "—",
                 f"{1.0 / (summary['steps_per_second'] * summary['timestep']):.0f}",
                 f"{summary['level_time']:.2f}（{'收敛' if summary.get('converged') else summary.get('end_reason', '')}）"]
        for quantity in ("u_min", "v_max", "v_min", "w_abs_max"):
            value = result["extrema"][quantity][METHOD]["value"]
            cells.append(f"{value:+.4f}（{percent(deviation(value, data[quantity]['value']))}）")
        profiles = result["profiles"]
        cells.append(f"{profiles['vertical']['velocity_l2']:.4f} / {profiles['horizontal']['velocity_l2']:.4f}")
        cells.append(f"{profiles['vertical']['pressure_l2']:.5f} / {profiles['horizontal']['pressure_l2']:.5f}")
        rows.append(cells)
    rows.append(["原文", "—", "—", "—", "—", "—", "—"] + [f"{data[key]['value']:+.4f}" for key in
                                                           ("u_min", "v_max", "v_min", "w_abs_max")] + ["—", "—"])
    write("round9_levels.md", markdown_table(
        ["N", "机器", "流体粒子", "壁面存储里的壁粒子", "显存 MiB（每张卡，运行中中位减空闲）", "每单位模拟时间的墙钟秒",
         "收敛所用本级时间", "u_min（偏差）", "v_max", "v_min", "|w|max（两侧平均）", "速度 L2：u / v",
         "压力 L2：竖线 / 横线"], rows))
    # old method beside, and the steadiness of the window
    rows = []
    for size, result in sorted(results.items()):
        cells = [str(size), str(result["samples"])]
        for quantity in ("u_min", "v_max", "v_min", "w_abs_max"):
            new = deviation(result["extrema"][quantity][METHOD]["value"], data[quantity]["value"])
            quadratic = deviation(result["extrema"][quantity]["quadratic"]["value"], data[quantity]["value"])
            old = result["old"]["extrema"][quantity]["deviation"]
            if None in result["halves"]:
                cells.append(f"{percent(old)} / {percent(quadratic)} / {percent(new)}；半窗 —")
                continue
            first, second = (deviation(half[quantity][METHOD]["value"], data[quantity]["value"])
                             for half in result["halves"])
            cells.append(f"{percent(old)} / {percent(quadratic)} / {percent(new)}；半窗 {100 * (second - first):+.2f}")
        cells.append(f"{result['old']['vertical_velocity_error']['l2']:.4f} / "
                     f"{result['old']['horizontal_velocity_error']['l2']:.4f}")
        rows.append(cells)
    write("round9_levels_methods.md", markdown_table(
        ["N", "窗口内记录", "u_min：(i) / (ii) / (iii)；后半窗减前半窗（百分点）", "v_max：同左", "v_min：同左",
         "|w|max：同左", "(i) 的速度 L2：u / v"], rows))
    # the starts of the levels and the volume constraint's I
    rows = []
    for size, result in sorted(results.items()):
        initial = result.get("initial_state") or {}
        start = result["start_rows"]
        extrema_start = result.get("start_extrema")
        rows.append([
            str(size), f"{initial['support']:.4f}" if initial else "从零",
            f"{initial['previous_specific_energy']:.5f} → {initial['specific_energy']:.5f}" if initial else "—",
            " / ".join(f"{row['specific_energy']:.5f}" for row in start),
            " / ".join(f"{row['speed_maximum']:.3f}" for row in start),
            (f"{result['density_range_first_unit'][0]:.2f}–{result['density_range_first_unit'][1]:.2f}"
             f"（以后 {result['density_range_later'][0]:.2f}–{result['density_range_later'][1]:.2f}）")
            if result["density_range_first_unit"] and result["density_range_later"] else "—",
            " / ".join(percent(extrema_start[key]) for key in ("u_min", "v_max", "v_min")) if extrema_start and initial
            else "—",
            (f"{result['integral']['first']:.3e} → {result['integral']['last']:.3e}；|θ| ≤ "
             f"{result['integral']['theta_maximum_after_1']:.1e}") if result["integral"]["first"] is not None else "—"])
    write("round9_starts.md", markdown_table(
        ["N", "插值支撑", "单位质量动能：插值前 → 插值后", "本级时间 0 / 0.1 / 0.5 / 1 / 2 的单位质量动能",
         "同上的最大速度", "第一个单位内的密度范围（以后）", "起步时的平面极值（直接拟合）",
         "体积修正的积分项 I：第一次更新 → 最后；t > 1 后的 |θ| 最大"], rows))
    # extrapolation
    sizes = np.array(sorted(results))
    rows, fits = [], {}
    if sizes.size >= 3:
        for quantity in ("u_min", "v_max", "v_min", "w_abs_max"):
            for method_key, label in ((METHOD, METHOD_TEXT[METHOD]), ("old", METHOD_TEXT["linear"])):
                if method_key == "old":
                    values = np.array([results[size]["old"]["extrema"][quantity]["deviation"] for size in sizes])
                    halves = np.array([0.0 for _ in sizes])
                else:
                    values = np.array([deviation(results[size]["extrema"][quantity][METHOD]["value"],
                                                 data[quantity]["value"]) for size in sizes])
                    halves = np.array([abs(deviation(results[size]["halves"][1][quantity][METHOD]["value"],
                                                     data[quantity]["value"])
                                           - deviation(results[size]["halves"][0][quantity][METHOD]["value"],
                                                       data[quantity]["value"]))
                                       if None not in results[size]["halves"] else 0.0 for size in sizes])
                # each level's uncertainty: the scatter of 4-unit means on the long N = 100 run's plateau (one window
                # of a level is one such mean); without the long run, half the difference of the two half-windows
                if long is not None:
                    errors = np.full(sizes.size, max(long["plateau"]["scatter"][quantity], 0.0005))
                else:
                    errors = np.maximum(0.5 * halves, 0.0005)
                fit = fit_limit(sizes, values, errors)
                fits[f"{quantity}_{method_key}"] = {**fit, "values": values.tolist(), "errors": errors.tolist()}
                rows.append([QUANTITY_TEXT[quantity], label, " / ".join(percent(value) for value in values),
                             power_text(fit),
                             f"{percent(fit['limit'])} ± {100 * fit['limit_spread']:.2f}"])
        write("round9_extrapolation.md", markdown_table(
            [f"量", "方法", f"偏差（N = {' / '.join(str(size) for size in sizes)}）", "阶 p（16–84 % 区间）",
             "外推极限（± 半宽）"], rows))
    if long is not None:
        long_tables(results, long, fits)
    (ROUND / "extrapolation.json").write_text(json.dumps(fits, indent=1), encoding="utf-8", newline="\n")


def fixed_power_limit(sizes: np.ndarray, deltas: np.ndarray, errors: np.ndarray, power: float) -> dict:
    """delta(N) = C (N^-p - 100^-p) for a given p (delta(100) = 0 exactly), weighted least squares over N > 100:
    delta(infinity) = -C 100^-p with its standard error, and the chi-square per degree of freedom."""
    mask = sizes > 100
    basis = sizes[mask].astype(float) ** (-power) - 100.0 ** (-power)
    weights = 1.0 / errors[mask] ** 2
    coefficient = float(np.sum(weights * basis * deltas[mask]) / np.sum(weights * basis ** 2))
    coefficient_error = float(1.0 / math.sqrt(np.sum(weights * basis ** 2)))
    residual = deltas[mask] - coefficient * basis
    degrees = max(int(mask.sum()) - 1, 1)
    return {"power": power, "coefficient": coefficient, "limit": -coefficient * 100.0 ** (-power),
            "limit_error": coefficient_error * 100.0 ** (-power),
            "chi_square": float(np.sum(weights * residual ** 2) / degrees)}


def long_tables(results: dict, long: dict, fits: dict) -> None:
    """report 15.4: the levels (and their continuations) against the long N = 100 run at equal accumulated time, the
    long run's approach to its plateau, the drift rates, and the extrapolation of the equal-time refinement effect
    delta(N) (delta(100) = 0) from the N = 100 plateau."""
    data = reference()["extrema"]
    quantities = QUANTITIES_ALL
    scatter = long["plateau"]["scatter"]
    continuations = long.get("continuations") or {}
    # 1. the long run's 4-unit means: approach and plateau
    rows = []
    for row in long["evolution"]:
        rows.append([f"{row['start']:.0f}–{row['start'] + WINDOW:.0f}" + ("（平台）" if row["plateau"] else ""),
                     str(row["records"])] + [percent(row[quantity]) for quantity in quantities])
    rows.append([f"平台 t ≥ {PLATEAU_FROM:.0f} 的平均（{len(long['plateau']['windows'])} 个窗口）", "—"]
                + [percent(long["plateau"]["mean"][quantity]) for quantity in quantities])
    rows.append(["平台上 4 单位平均值的散布（标准差）", "—"]
                + [f"{100 * scatter[quantity]:.2f} 个百分点" for quantity in quantities])
    write("round9_long_run.md", markdown_table(
        ["N = 100 长跑的窗口（累计时间 t）", "记录数", "u_min", "v_max", "v_min", "|w|max"], rows))
    # 2. every window of a level or a continuation against the long run in the same window
    rows = []
    per_level = {}
    for size, result in sorted(results.items()):
        window = long["windows"].get(size)
        if window is None:
            rows.append([str(size), "—", "长跑没有覆盖这个窗口", "", ""])
            continue
        level = {quantity: deviation(result["extrema"][quantity][METHOD]["value"], data[quantity]["value"])
                 for quantity in quantities}
        label = "第 0 级" if size == 100 else "这一级的结果"
        offset = float(result["summary"].get("time_offset") or 0.0)
        entries = [(f"{size}（{label}）", window["end"], level, window)]
        for continued, long_window in zip((continuations.get(size) or {}).get("windows", []),
                                          long.get("continuation_windows", {}).get(size, [])):
            entries.append((f"{size} 续跑", continued["end"], continued, long_window))
        per_level[size] = []
        every = [{quantity: values[quantity] - long_values[quantity] for quantity in quantities}
                 for _, _, values, long_values in entries]
        for text, end, values, long_values in entries:
            delta = {quantity: values[quantity] - long_values[quantity] for quantity in quantities}
            level_start = end - WINDOW - offset
            used = size == 100 or level_start >= RELAXED_FROM - 0.05
            if used:
                per_level[size].append(delta)
            rows.append([text, f"{end - WINDOW:.1f}–{end:.1f}", f"{level_start:.1f}–{level_start + WINDOW:.1f}",
                         " / ".join(percent(values[quantity]) for quantity in quantities),
                         " / ".join(percent(long_values[quantity]) for quantity in quantities),
                         " / ".join(f"{100 * delta[quantity]:+.2f}" for quantity in quantities),
                         "—" if size == 100 else ("是" if used else "否（本级时间 < 12，还在回落）")])
        if not per_level[size]:              # no relaxed window (no continuation): every window, marked in the log
            print(f"[round9] N = {size}: no window from level time {RELAXED_FROM:g}; delta from all windows", flush=True)
            per_level[size] = every
    write("round9_drift.md", markdown_table(
        ["N", "窗口（累计时间 t）", "本级时间", "这一级：u_min / v_max / v_min / |w|max",
         "同一窗口的 N = 100 长跑", "差 δ = 这一级 − N = 100（百分点）", "用于外推"], rows))
    # 3. drift rates
    rates_round8 = drift_rates()
    rows = []
    for quantity in quantities:
        cells = [QUANTITY_TEXT[quantity]]
        for span_key in ("approach", "plateau"):
            rate = long["rates"][span_key][quantity]
            cells.append(f"{100 * rate['slope']:+.4f} ± {100 * rate['slope_error']:.4f}")
        cells.append(f"{100 * rates_round8[quantity]['slope']:+.4f} ± {100 * rates_round8[quantity]['slope_error']:.4f}")
        for size in sorted(long.get("level_rates", {})):
            entry = (continuations.get(size) or {}).get("rates", {}).get(quantity) \
                or long["level_rates"][size].get(quantity)
            cells.append(f"{100 * entry['slope']:+.4f} ± {100 * entry['slope_error']:.4f}（t {entry['span'][0]:.0f}–"
                         f"{entry['span'][1]:.0f}）" if entry else "—")
        rows.append(cells)
    approach = long["rates"]["approach"]["u_min"]["span"]
    plateau_span = long["rates"]["plateau"]["u_min"]["span"]
    write("round9_drift_rates.md", markdown_table(
        ["量（百分点 / 单位时间）", f"N = 100 长跑 t = {approach[0]:.0f}–{approach[1]:.0f}",
         f"N = 100 长跑 t = {plateau_span[0]:.0f}–{plateau_span[1]:.0f}", "第八轮同一算例 t = 30–45，现在的方法"]
        + [f"N = {size}：本级时间 ≥ 6" + ("，含续跑" if size in continuations else "")
           for size in sorted(long.get("level_rates", {}))], rows))
    # 4. delta(N): the mean over a level's windows (result window and continuation windows); one window's delta has
    # the uncertainty sqrt(2) x the plateau scatter (the level's window and the long run's are one 4-unit mean each)
    sizes = np.array(sorted(per_level))
    if sizes.size < 3:
        return
    rows = []
    for quantity in quantities:
        deltas, errors, counts = [], [], []
        for size in sizes:
            if size == 100:
                deltas.append(0.0)
                errors.append(0.0005)
                counts.append(1)
                continue
            values = [entry[quantity] for entry in per_level[size]]
            deltas.append(float(np.mean(values)))
            errors.append(max(math.sqrt(2.0) * scatter[quantity] / math.sqrt(len(values)), 0.0005))
            counts.append(len(values))
        deltas, errors = np.array(deltas), np.array(errors)
        fit = fit_limit(sizes, deltas, errors)
        fixed = {power: fixed_power_limit(sizes, deltas, errors, power) for power in (1.0, 2.0)}
        base = long["plateau"]["mean"][quantity]
        base_error = scatter[quantity] / math.sqrt(len(long["plateau"]["windows"]))
        fits[f"{quantity}_matched"] = {**fit, "sizes": sizes.tolist(), "values": deltas.tolist(),
                                       "errors": errors.tolist(), "windows": counts, "base": base,
                                       "base_error": base_error,
                                       "fixed": {str(key): value for key, value in fixed.items()}}
        free_text = (f"{percent(base + fit['limit'])} ± {100 * math.hypot(fit['limit_spread'], base_error):.2f}"
                     if not at_bound(fit) else "—")
        delta_texts = []
        for size, value, count in zip(sizes, deltas, counts):
            delta_texts.append(f"{100 * value:+.2f}" + (f"（{count} 窗）" if size != 100 else ""))
        fixed_texts = []
        for power in (1.0, 2.0):
            entry = fixed[power]
            fixed_texts.append(f"{percent(base + entry['limit'])} ± {100 * math.hypot(entry['limit_error'], base_error):.2f}"
                               f"（χ² {entry['chi_square']:.1f}）")
        rows.append([QUANTITY_TEXT[quantity], " / ".join(delta_texts),
                     " / ".join(f"{100 * error:.2f}" for size, error in zip(sizes, errors) if size != 100),
                     power_text(fit), f"{100 * fit['limit']:+.2f} ± {100 * fit['limit_spread']:.2f}",
                     f"{percent(base)} ± {100 * base_error:.2f}", free_text, " / ".join(fixed_texts)])
    write("round9_extrapolation_matched.md", markdown_table(
        ["量（直接拟合）", f"δ(N)，N = {' / '.join(str(size) for size in sizes)}（百分点）",
         f"δ 的不确定度，N = {' / '.join(str(size) for size in sizes if size != 100)}",
         "阶 p（16–84 %）", "δ(∞)（百分点）", f"N = 100 长跑的平台（t ≥ {PLATEAU_FROM:.0f}）",
         "极限 = 平台 + δ(∞)（p 自由）", "p 取 1 / 2 时的极限（χ² / 自由度）"], rows))


def level_figures(results: dict, long: dict | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiment.wall_crossing.report_figures import INK_PRIMARY, INK_SECONDARY, save, setup_style
    setup_style()
    data = reference()
    colours = {100: "#2a78d6", 140: "#1baf7a", 170: "#9a5ac8", 200: "#eb6834", 280: "#b2441a"}

    def header(figure, title, handles, labels, columns=4):
        height = figure.get_figheight()
        figure.suptitle(title, x=0.01, y=1.0 - 0.05 / height, ha="left", va="top", fontsize=12, fontweight="bold",
                        color=INK_PRIMARY)
        figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0 - 0.40 / height), ncol=columns,
                      handlelength=3.0, labelcolor=INK_PRIMARY, borderaxespad=0.0)
        rows = math.ceil(len(labels) / columns)
        figure.tight_layout(rect=(0, 0, 1.0, 1.0 - (0.45 + 0.27 * rows) / height), w_pad=2.0, h_pad=1.5)

    # 1. history: the three in-plane extrema against the accumulated time, the levels joined
    figure, axes = plt.subplots(3, 1, figsize=(12.0, 10.0), sharex=True)
    for size, result in sorted(results.items()):
        time_values = np.array([row["time"] for row in result["history"]])
        for axis, quantity in zip(axes, ("u_min", "v_max", "v_min")):
            axis.plot(time_values, [100 * row[quantity] for row in result["history"]], color=colours.get(size, "#666"),
                      linewidth=1.5, marker="o", markersize=3, label=f"N = {size}（{result.get('machine', '')}）")
            axis.plot(time_values, [100 * row[f"{quantity}_linear"] for row in result["history"]],
                      color=colours.get(size, "#666"), linewidth=1.0, linestyle=":")
        start = float(result["summary"].get("time_offset") or 0.0)
        for axis in axes:
            if start > 0:
                axis.axvline(start, color=INK_SECONDARY, linewidth=0.8, linestyle="--")
    if long is not None:
        long_times = np.array([row["time"] for row in long["series"]])
        for axis, quantity in zip(axes, ("u_min", "v_max", "v_min")):
            axis.plot(long_times, [100 * row[quantity] for row in long["series"]], color=INK_SECONDARY, linewidth=1.2,
                      label="N = 100 长跑（不按判据停，漂移诊断）", zorder=1)
        for size, continuation in sorted((long.get("continuations") or {}).items()):
            continued_times = np.array([row["time"] for row in continuation["series"]])
            for axis, quantity in zip(axes, ("u_min", "v_max", "v_min")):
                axis.plot(continued_times, [100 * row[quantity] for row in continuation["series"]],
                          color=colours.get(size, "#666"), linewidth=1.5, linestyle="--", marker="o", markersize=2.5,
                          markerfacecolor="none", label=f"N = {size} 续跑（不按判据停，平台检查）")
    for axis, quantity in zip(axes, ("u_min", "v_max", "v_min")):
        axis.axhline(0.0, color=INK_SECONDARY, linewidth=0.8)
        axis.set_ylabel(f"{QUANTITY_TEXT[quantity]} 偏差（%）")
        # the from-zero spin-up of N = 100 (deviations down to -90 %) would hide the levels' differences
        axis.set_xlim(left=HISTORY_FROM)
        axis.set_ylim(-4.0, 0.5)
    axes[-1].set_xlabel(f"累计时间 t（L / U）；虚竖线 = 换级；N = 100 从零起步的 t < {HISTORY_FROM:g} 在图外")
    handles, labels = axes[0].get_legend_handles_labels()
    header(figure, "逐级加密：三个平面极值相对原文的偏差（实线 = 直接拟合，点线 = 现在的方法）", handles, labels)
    save(figure, FIGURES / "round9_history.png")

    # 2. deviation against N, both methods, log-log of |deviation|
    fits = json.loads((ROUND / "extrapolation.json").read_text(encoding="utf-8"))         if (ROUND / "extrapolation.json").exists() else {}
    sizes = np.array(sorted(results))
    figure, axes = plt.subplots(1, 4, figsize=(14.0, 4.6))
    for axis, quantity in zip(axes, ("u_min", "v_max", "v_min", "w_abs_max")):
        for method_key, marker, style, label in ((METHOD, "o", "-", "直接拟合"), ("old", "s", ":", "现在的方法")):
            entry = fits.get(f"{quantity}_{method_key}")
            if entry is None:
                continue
            values = np.abs(np.asarray(entry["values"]))
            axis.loglog(sizes, values * 100, marker=marker, linestyle=style, color=INK_PRIMARY if method_key == "old"
                        else "#2a78d6", markersize=6, linewidth=1.5, label=label)
            if at_bound(entry):
                continue                       # no N^-p convergence: no curve (table: at the bound)
            dense = np.linspace(sizes.min(), sizes.max() * 1.6, 100)
            model = entry["limit"] + entry["coefficient"] * dense ** (-entry["power"])
            axis.loglog(dense, np.abs(model) * 100, color="#2a78d6" if method_key != "old" else INK_SECONDARY,
                        linewidth=0.8, linestyle="--")
        matched = fits.get(f"{quantity}_matched")
        if matched is not None:
            # equal time: the long N = 100 run's plateau plus each level's delta(N)
            matched_sizes = np.asarray(matched["sizes"])
            base = matched["base"]
            axis.loglog(matched_sizes, np.abs(base + np.asarray(matched["values"])) * 100, marker="^", linestyle="-",
                        color="#eb6834", markersize=6, linewidth=1.5,
                        label=f"直接拟合，按 N = 100 长跑对齐到它的平台（t ≥ {PLATEAU_FROM:.0f}）")
            if not at_bound(matched):
                dense = np.linspace(sizes.min(), sizes.max() * 1.6, 100)
                model = base + matched["limit"] + matched["coefficient"] * dense ** (-matched["power"])
                axis.loglog(dense, np.abs(model) * 100, color="#eb6834", linewidth=0.8, linestyle="--")
        # guide slopes N^-1 and N^-2 through the chosen method's N = 100 value
        entry = fits.get(f"{quantity}_{METHOD}")
        if entry is not None:
            guide = np.array([sizes.min(), sizes.max() * 1.6])
            start = abs(entry["values"][0]) * 100
            for power, style in ((1, (0, (6, 3))), (2, (0, (2, 2)))):
                axis.loglog(guide, start * (guide / guide[0]) ** (-power), color=INK_SECONDARY, linewidth=0.7,
                            linestyle=style, label=f"N^−{power}（参考斜率）")
        axis.set_title(QUANTITY_TEXT[quantity])
        axis.set_xlabel("N")
        axis.set_xticks(sizes)
        axis.set_xticklabels([str(size) for size in sizes])
        axis.minorticks_off()
        ticks = [value for value in (0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5) ]
        axis.set_yticks(ticks)
        axis.set_yticklabels([f"{value:g}" for value in ticks])
        axis.set_ylim(0.1, 5.0)
    axes[0].set_ylabel("|偏差|（%）")
    handles, labels = axes[0].get_legend_handles_labels()
    header(figure, "偏差随 N（双对数；彩色虚线 = 极限 + C N^−p 的拟合，到边界的不画）", handles, labels)
    save(figure, FIGURES / "round9_convergence.png")

    # 3. the largest level's centre-line velocity against the paper (top), every level's difference from the paper at
    # the 15 interior table points (bottom; quadratic MLS, the profile L2 of round9_levels)
    largest = max(results)
    profiles = results[largest]["fine_profiles"]
    figure, grid = plt.subplots(2, 2, figsize=(12.0, 9.6), gridspec_kw={"height_ratios": [1.5, 1.0]})
    axes = grid[0]
    axes[0].plot(profiles["vertical"]["velocity"], profiles["vertical"]["coordinate"], color="#2a78d6", linewidth=1.5,
                 label=f"N = {largest}（二次 MLS，最后 {WINDOW:g} 个单位的平均）")
    axes[1].plot(profiles["horizontal"]["coordinate"], profiles["horizontal"]["velocity"], color="#2a78d6",
                 linewidth=1.5)
    marker = {"marker": "o", "linestyle": "none", "color": INK_PRIMARY, "markersize": 4.5, "markerfacecolor": "none",
              "markeredgewidth": 1.2}
    axes[0].plot(data["vertical"]["velocity"], data["vertical"]["coordinate"], label="Albensoeder & Kuhlmann 2005",
                 **marker)
    axes[1].plot(data["horizontal"]["coordinate"], data["horizontal"]["velocity"], **marker)
    for key, axis in (("u_min", axes[0]), ("v_max", axes[1]), ("v_min", axes[1])):
        extremum = data["extrema"][key]
        if key == "u_min":
            axis.plot([extremum["value"]], [extremum["position"]], marker="D", color=INK_PRIMARY, markersize=6,
                      linestyle="none")
        else:
            axis.plot([extremum["position"]], [extremum["value"]], marker="D", color=INK_PRIMARY, markersize=6,
                      linestyle="none")
    axes[0].set_xlabel("u / U"); axes[0].set_ylabel("y / L"); axes[0].set_title("竖线 (0.5, y, 0.5)")
    axes[1].set_xlabel("x / L"); axes[1].set_ylabel("v / U"); axes[1].set_title("横线 (x, 0.5, 0.5)")
    handles, labels = axes[0].get_legend_handles_labels()
    for axis, line in zip(grid[1], ("vertical", "horizontal")):
        reference_velocity = data[line]["velocity"][inner(data[line]["coordinate"])]
        for size, result in sorted(results.items()):
            entry = result["profiles"][line]
            axis.plot(entry["coordinate"], 1000.0 * (np.asarray(entry["velocity"]) - reference_velocity),
                      color=colours.get(size, "#666"), marker="o", markersize=4, linewidth=1.2,
                      label=f"N = {size}")
        axis.axhline(0.0, color=INK_SECONDARY, linewidth=0.8)
        axis.set_xlabel("y / L" if line == "vertical" else "x / L")
        axis.set_ylabel(("u" if line == "vertical" else "v") + " − 原文（× 10⁻³ U）")
        axis.set_title(("竖线" if line == "vertical" else "横线") + "：表格点上的差（各级）")
    level_handles, level_labels = grid[1][0].get_legend_handles_labels()
    header(figure, f"N = {largest} 的中心线速度（圆圈 = 原文表 5、6，菱形 = 表 10 的极值）；下排 = 各级在表格点上的差",
           handles + level_handles, labels + level_labels, columns=3)
    save(figure, FIGURES / "round9_profiles.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("measurement", "levels"))
    arguments = parser.parse_args()
    return measurement_main() if arguments.command == "measurement" else level_main()


if __name__ == "__main__":
    sys.exit(main())
