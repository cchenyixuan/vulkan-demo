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
    level_tables(results)
    level_figures(results)
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


def level_tables(results: dict) -> None:
    data = reference()["extrema"]
    rows = []
    for size, result in sorted(results.items()):
        summary = result["summary"]
        cells = [str(size), result.get("machine", ""), f"{summary['fluid_particles']:,}",
                 f"{int(result['wall_store']):,}（生成 {int(result['wall_generated']):,}）" if result["wall_store"] else "—",
                 " / ".join(f"{value / 1024:.1f}" for value in result["memory"].values()) if result["memory"] else "—",
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
        ["N", "机器", "流体粒子", "壁面存储里的壁粒子", "显存 GiB（每张卡，运行中中位）", "每单位模拟时间的墙钟秒",
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
                # each level's uncertainty: half the difference of its two half-windows, at least 0.05 points
                errors = np.maximum(0.5 * halves, 0.0005)
                fit = fit_limit(sizes, values, errors)
                fits[f"{quantity}_{method_key}"] = {**fit, "values": values.tolist(), "errors": errors.tolist()}
                rows.append([QUANTITY_TEXT[quantity], label, " / ".join(percent(value) for value in values),
                             f"{fit['power']:.2f}（{fit['power_range'][0]:.2f}–{fit['power_range'][1]:.2f}）",
                             f"{percent(fit['limit'])} ± {100 * fit['limit_spread']:.2f}"])
        write("round9_extrapolation.md", markdown_table(
            [f"量", "方法", f"偏差（N = {' / '.join(str(size) for size in sizes)}）", "阶 p（16–84 % 区间）",
             "外推极限（± 半宽）"], rows))
    (ROUND / "extrapolation.json").write_text(json.dumps(fits, indent=1), encoding="utf-8", newline="\n")


def level_figures(results: dict) -> None:
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
    for axis, quantity in zip(axes, ("u_min", "v_max", "v_min")):
        axis.axhline(0.0, color=INK_SECONDARY, linewidth=0.8)
        axis.set_ylabel(f"{QUANTITY_TEXT[quantity]} 偏差（%）")
    axes[-1].set_xlabel("累计时间 t（L / U）；虚竖线 = 换级")
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
            dense = np.linspace(sizes.min(), sizes.max() * 1.6, 100)
            model = entry["limit"] + entry["coefficient"] * dense ** (-entry["power"])
            axis.loglog(dense, np.abs(model) * 100, color="#2a78d6" if method_key != "old" else INK_SECONDARY,
                        linewidth=0.8, linestyle="--")
        axis.set_title(QUANTITY_TEXT[quantity])
        axis.set_xlabel("N")
        axis.set_xticks(sizes)
        axis.set_xticklabels([str(size) for size in sizes])
        axis.minorticks_off()
    axes[0].set_ylabel("|偏差|（%）")
    handles, labels = axes[0].get_legend_handles_labels()
    header(figure, "偏差随 N（双对数；虚线 = 极限 + C N^−p 的拟合）", handles, labels)
    save(figure, FIGURES / "round9_convergence.png")

    # 3. the largest level's centre-line velocity against the paper
    largest = max(results)
    profiles = results[largest]["fine_profiles"]
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.6))
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
    header(figure, f"N = {largest} 的中心线速度（圆圈 = 原文表 5、6，菱形 = 表 10 的极值）", handles, labels, columns=2)
    save(figure, FIGURES / "round9_profiles.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("measurement", "levels"))
    arguments = parser.parse_args()
    return measurement_main() if arguments.command == "measurement" else level_main()


if __name__ == "__main__":
    sys.exit(main())
