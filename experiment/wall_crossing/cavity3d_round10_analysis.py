"""
cavity3d_round10_analysis.py - claude/wall_store_report.md section 16 (round 10): every level of the 3-D cavity at
Re = 1000 on its steady window, the extrapolation over N = 100 .. 280, the report tables, the thesis tables and figures.

    python -m experiment.wall_crossing.cavity3d_round10 analysis

Steady windows (all by the direct fit of report 15.2, pooled over the window's tube records):
  N = 100  the long run's plateau, t = 44 .. 70 (round 9, L100_long)
  N = 140  its result window, level time 12.5 .. 16.5 (round 9; its state was deleted, no continuation)
  N = 170  the continuation, level time 12.4 .. 20.4 (round 9, L170_cont)
  N = 200  the continuation, level time 12.2 .. 20.2 (round 9, L200_cont)
  N = 280  the last 4 units at the steadiness stop (or at the cap of level time 20)
The uncertainty of a level = the scatter of 4-unit means on N = 100's plateau / sqrt(number of 4-unit windows).
Fit: deviation = limit + C N^-p, p free (Monte Carlo, cavity3d_round9.fit_limit) and p = 1, 2 (weighted least
squares, chi-square per degree of freedom and its probability).
Outputs: claude/wall_store/tables/round10_*.md, claude/wall_store/thesis/ (CSV, PDF, PNG), logs/wall_store/round10/
section16.json.
"""
from __future__ import annotations

import csv
import datetime
import json
import math
import re
import time

import numpy as np

from experiment.wall_crossing import cavity3d_extrema as extrema_methods
from experiment.wall_crossing.cavity3d_round10 import (LEVEL_CAP, LOGS, QUANTITIES, REPOSITORY_ROOT, lid_analysis,
                                                       lid_two_dimensional, predictions, steady_samples)
from experiment.wall_crossing.cavity3d_round9 import METHOD, deviation, extrema_of, reference

ROUND9 = LOGS / "round9"
ROUND10 = LOGS / "round10"
THESIS = REPOSITORY_ROOT / "claude" / "wall_store" / "thesis"
SIZES = (100, 140, 170, 200, 280)
WINDOW = 4.0
RECORD_INTERVAL = 0.5
QUANTITY_TEXT = {"u_min": "u_min", "v_max": "v_max", "v_min": "v_min", "w_abs_max": "\\|w\\|max"}
MACHINE_TEXT = {"local": "本机双卡", "remote": "远端单卡"}
MACHINE_ENGLISH = {"local": "2 x RTX 5090", "remote": "1 x RTX 5090"}
STEADY_KIND = {100: "长跑平台", 140: "结果窗口", 170: "续跑", 200: "续跑", 280: "最后 4 个单位"}


def steady_text(size: int, level: dict) -> str:
    """The steady window of a level: accumulated time for N = 100's long run, level time for the others (the window
    starts half a record interval before its first record)."""
    if size == 100:
        return f"{STEADY_KIND[size]} t = {level['time'][0] - RECORD_INTERVAL:.1f}–{level['time'][1]:.1f}"
    return (f"{STEADY_KIND[size]}，本级时间 {level['level_time'][0] - RECORD_INTERVAL:.1f}–"
            f"{level['level_time'][1]:.1f}（{level['records']} 个记录）")
LEVEL_SWITCHES = (32.16, 48.62, 53.58)


def final_window_280() -> tuple:
    """(first, last, reason): N = 280's last 4 units of level time at the steadiness stop (the stop file's record), or
    at the cap, or (still running) at its newest record."""
    stop = ROUND10 / "L280.stop"
    if stop.exists():
        end = float(json.loads(stop.read_text(encoding="utf-8"))["level_time"])
        return end - WINDOW, end, "steady"
    series = [json.loads(line) for line in (ROUND10 / "box_run" / "L280" / "extrema.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    end = series[-1]["level_time"]
    return end - WINDOW, end, "cap" if end >= LEVEL_CAP - RECORD_INTERVAL else "running"


def level_samples(size: int) -> list:
    if size != 280:
        return steady_samples(size)
    first, last, _reason = final_window_280()
    return [sample for sample in steady_samples(280, (first - 1.0, last))
            if first + 1e-6 < sample["level_time"] <= last + 1e-6]


def window_count(samples: list) -> int:
    span = max(sample["level_time"] for sample in samples) - min(sample["level_time"] for sample in samples) \
        + RECORD_INTERVAL
    return max(1, int(round(span / WINDOW)))


def smi_memory_280() -> dict:
    """Median memory.used per card while N = 280 runs (from 10 minutes after its start: the interpolation is done)
    minus the median before its start (the display card's own use)."""
    log = (ROUND10 / "master.log").read_text(encoding="utf-8").splitlines()
    start = datetime.datetime.strptime(next(line[:19] for line in log if "L280 starts" in line),
                                       "%Y-%m-%d %H:%M:%S").timestamp()
    ends = [line[:19] for line in log if "L280 ended" in line]
    end = datetime.datetime.strptime(ends[0], "%Y-%m-%d %H:%M:%S").timestamp() if ends else time.time()
    idle, running = {}, {}
    with open(ROUND10 / "smi_local.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, skipinitialspace=True):
            try:
                unix, index, used = float(row["unix"]), int(row["index"]), float(row["memory.used"])
            except (KeyError, TypeError, ValueError):  # the logger's ERROR rows (no memory column)
                continue
            if unix < start:
                idle.setdefault(index, []).append(used)
            elif start + 600.0 < unix < end:
                running.setdefault(index, []).append(used)
    return {index: float(np.median(values)) - float(np.median(idle.get(index, [0.0])))
            for index, values in running.items()}


def level_costs(size: int) -> dict:
    """Particles, memory, wall clock: round 9's level entries (levels_results.json) and, for N = 280, this round's run
    (summary.json or, while it runs, the tube records' clock; the nvidia-smi log)."""
    if size != 280:
        entry = json.loads((ROUND9 / "levels_results.json").read_text(encoding="utf-8"))[str(size)]
        summary = entry["summary"]
        memory = {int(key): float(value) for key, value in entry["memory"].items()}
        machine, wall_store = entry["chain"], int(entry["wall_store"])
        seconds_per_unit = 1.0 / (float(summary["steps_per_second"]) * float(summary["timestep"]))
        fluid = int(summary["fluid_particles"])
    else:
        run = ROUND10 / "box_run" / "L280"
        text = (ROUND10 / "L280.out").read_text(encoding="utf-8", errors="replace")
        wall_store = int(re.search(r"wall store: ([\d,]+) wall particles kept", text).group(1).replace(",", ""))
        fluid = int(re.search(r"particle pool: ([\d,]+) fluid particles", text).group(1).replace(",", ""))
        machine, memory = "local", smi_memory_280()
        series = [json.loads(line) for line in (run / "extrema.jsonl").read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        # the records' clock after the first one (the start, interpolation and bootstrap excluded)
        seconds_per_unit = (series[-1]["written"] - series[0]["written"]) / (series[-1]["level_time"]
                                                                           - series[0]["level_time"])
    total_memory = sum(memory.values())
    return {"machine": machine, "fluid": fluid, "wall_store": wall_store, "memory": memory,
            "bytes_per_particle": total_memory * 2 ** 20 / (fluid + wall_store) if total_memory else None,
            "seconds_per_unit": seconds_per_unit}


def level_steady_values(size: int) -> dict:
    """Direct-fit extrema of the pooled steady window, quadratic-MLS profiles at the table points (L2) and, for the
    finest level, the fine profiles of the figure."""
    from experiment.wall_crossing.cavity3d_round9 import fine_profiles, table_points_profile
    samples = level_samples(size)
    data = reference()["extrema"]
    lower, upper = extrema_methods.box_of(size)
    support = 4.0 / size
    values = extrema_of(samples, lower, upper, size, support)
    volume = float(np.median(samples[-1]["volumes"]))
    walls = extrema_methods.boundary_particles(
        REPOSITORY_ROOT / f"cases/cavity3d_re1000_n{size}_hdx4_crossing/case.yaml", volume)
    tube = 1.6 * support
    walls = tuple(array[(np.abs(walls[0]) <= tube).sum(axis=1) >= 2] for array in walls)
    profiles = table_points_profile(samples, lower, upper, support, walls, 2)
    out = {"records": len(samples), "windows": window_count(samples),
           "time": [min(sample["time"] for sample in samples), max(sample["time"] for sample in samples)],
           "level_time": [min(sample["level_time"] for sample in samples),
                          max(sample["level_time"] for sample in samples)],
           "values": {quantity: values[quantity][METHOD]["value"] for quantity in QUANTITIES},
           "deviations": {quantity: 100.0 * deviation(values[quantity][METHOD]["value"], data[quantity]["value"])
                          for quantity in QUANTITIES},
           "velocity_l2": {"u": profiles["vertical"]["velocity_l2"], "v": profiles["horizontal"]["velocity_l2"]},
           "pressure_l2": {"vertical": profiles["vertical"]["pressure_l2"],
                           "horizontal": profiles["horizontal"]["pressure_l2"]},
           "profiles": profiles}
    if size == max(SIZES):
        out["fine_profiles"] = fine_profiles(samples, lower, upper, size, support, walls)
    print(f"[round10] N = {size}: {len(samples)} records ({out['windows']} windows), "
          + ", ".join(f"{quantity} {out['deviations'][quantity]:+.3f} %" for quantity in QUANTITIES), flush=True)
    return out


def plateau_scatter() -> dict:
    """Scatter of the 4-unit means on N = 100's plateau (round 9, long_run.json), percentage points."""
    long = json.loads((ROUND9 / "long_run.json").read_text(encoding="utf-8"))
    return {quantity: 100.0 * float(long["plateau"]["scatter"][quantity]) for quantity in QUANTITIES}


def weighted_power_fit(sizes: np.ndarray, values: np.ndarray, errors: np.ndarray, power: float) -> dict:
    """deviation = limit + C N^-p at a fixed p by weighted least squares: limit, its standard error, chi-square per
    degree of freedom."""
    design = np.column_stack([np.ones(sizes.size), sizes.astype(float) ** -power])
    weights = 1.0 / errors ** 2
    covariance = np.linalg.inv(design.T @ (weights[:, None] * design))
    parameters = covariance @ (design.T @ (weights * values))
    residual = values - design @ parameters
    degrees = max(sizes.size - 2, 1)
    return {"power": power, "limit": float(parameters[0]), "coefficient": float(parameters[1]),
            "limit_error": float(math.sqrt(covariance[0, 0])),
            "chi_square": float(np.sum(weights * residual ** 2) / degrees), "degrees": degrees,
            "residuals": residual.tolist()}


def extrapolations(levels: dict) -> dict:
    from scipy.stats import chi2
    from experiment.wall_crossing.cavity3d_round9 import fit_limit
    scatter = plateau_scatter()
    prediction = predictions(280)
    sizes = np.array(SIZES)
    out = {}
    for quantity in QUANTITIES:
        values = np.array([levels[size]["deviations"][quantity] for size in SIZES])
        errors = np.array([scatter[quantity] / math.sqrt(levels[size]["windows"]) for size in SIZES])
        raw = fit_limit(sizes, values / 100.0, errors / 100.0)
        free = {"power": raw["power"], "power_range": raw["power_range"], "limit": 100.0 * raw["limit"],
                "coefficient": 100.0 * raw["coefficient"], "limit_range": [100.0 * value for value in raw["limit_range"]]}
        fixed = {}
        for power in (1.0, 2.0):
            fit = weighted_power_fit(sizes, values, errors, power)
            fit["probability"] = float(chi2.sf(fit["chi_square"] * fit["degrees"], fit["degrees"]))
            fixed[str(power)] = fit
        entry = {"values": values.tolist(), "errors": errors.tolist(), "free": free, "fixed": fixed}
        if quantity in ("u_min", "v_max"):
            measured, error = values[-1], errors[-1]
            entry["prediction_test"] = {power: {"predicted": prediction[power][quantity],
                                                "difference": float(measured - prediction[power][quantity]),
                                                "sigmas": float((measured - prediction[power][quantity]) / error)}
                                        for power in ("1.0", "2.0")}
        out[quantity] = entry
    return out


def signed(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}".replace("-", "−")


def at_bound(free: dict) -> bool:
    return free["power"] <= 0.1001 or free["power"] >= 5.999


def report_tables(levels: dict, costs: dict, fits: dict, steady: tuple, lid: dict) -> None:
    from experiment.wall_crossing.cavity3d_round9 import markdown_table, write
    data = reference()["extrema"]
    _first, last, reason = steady
    reason_text = {"steady": "平稳", "cap": "到 20，没平稳", "running": "运行中"}[reason]

    def memory_text(cost):
        return " / ".join(f"{cost['memory'][index]:,.0f}" for index in sorted(cost["memory"]))

    level, cost = levels[280], costs[280]
    row = [f"{cost['fluid']:,}", f"{cost['wall_store']:,}", memory_text(cost),
           f"{cost['bytes_per_particle']:.0f}" if cost["bytes_per_particle"] else "—",
           f"{cost['seconds_per_unit']:,.0f}", f"{last:.2f}（{reason_text}）"]
    row += [f"{level['values'][quantity]:+.4f}（{signed(level['deviations'][quantity])} %）" for quantity in QUANTITIES]
    row += [f"{level['velocity_l2']['u']:.4f} / {level['velocity_l2']['v']:.4f}",
            f"{level['pressure_l2']['vertical']:.5f} / {level['pressure_l2']['horizontal']:.5f}"]
    write("round10_level280.md", markdown_table(
        ["流体粒子", "壁面存储的壁粒子", "显存 MiB（每张卡，运行中中位减空闲）", "每粒子字节", "每单位时间墙钟秒",
         "平稳所用本级时间", "u_min（偏差）", "v_max", "v_min", "\\|w\\|max（两侧平均）", "速度 L2：u / v",
         "压力 L2：竖线 / 横线"], [row]))

    rows = []
    for size in SIZES:
        level, cost = levels[size], costs[size]
        rows.append([str(size), MACHINE_TEXT[cost["machine"]], f"{cost['fluid']:,}", f"{cost['wall_store']:,}",
                     memory_text(cost), f"{cost['seconds_per_unit']:,.0f}", steady_text(size, level)]
                    + [f"{signed(level['deviations'][quantity])} %" for quantity in QUANTITIES]
                    + [f"{level['velocity_l2']['u']:.4f} / {level['velocity_l2']['v']:.4f}",
                       f"{level['pressure_l2']['vertical']:.5f} / {level['pressure_l2']['horizontal']:.5f}"])
    write("round10_summary.md", markdown_table(
        ["N", "机器", "流体粒子", "壁面存储的壁粒子", "显存 MiB（每张卡）", "每单位时间墙钟秒", "平稳窗口",
         "u_min", "v_max", "v_min", "\\|w\\|max", "速度 L2：u / v", "压力 L2：竖线 / 横线"], rows))

    rows = []
    for quantity in QUANTITIES:
        fit = fits[quantity]
        free, one, two = fit["free"], fit["fixed"]["1.0"], fit["fixed"]["2.0"]
        bound = at_bound(free)
        rows.append([QUANTITY_TEXT[quantity], " / ".join(signed(value) for value in fit["values"]),
                     " / ".join(f"{value:.3f}" for value in fit["errors"]),
                     f"{free['power']:.2f}（{free['power_range'][0]:.2f}–{free['power_range'][1]:.2f}）"
                     + ("；到边界" if bound else ""),
                     "—" if bound else f"{signed(free['limit'])} %（{signed(free['limit_range'][0])} – "
                                       f"{signed(free['limit_range'][1])}）",
                     f"{signed(one['limit'])} ± {one['limit_error']:.2f} %（χ² {one['chi_square']:.2f}，"
                     f"P {one['probability']:.2f}）",
                     f"{signed(two['limit'])} ± {two['limit_error']:.2f} %（χ² {two['chi_square']:.2f}，"
                     f"P {two['probability']:.2f}）"])
    write("round10_extrapolation.md", markdown_table(
        ["量", "偏差 N = 100 / 140 / 170 / 200 / 280（%）", "不确定度（百分点）", "阶 p（16–84 %）",
         "极限（p 自由，16–84 %）", "p = 1 的极限（χ² / 自由度，概率）", "p = 2 的极限（χ² / 自由度，概率）"], rows))

    rows = []
    for quantity in ("u_min", "v_max"):
        test, fit = fits[quantity]["prediction_test"], fits[quantity]
        rows.append([QUANTITY_TEXT[quantity], f"{signed(fit['values'][-1])} ± {fit['errors'][-1]:.2f} %"]
                    + [f"{signed(test[power]['predicted'])} %（差 {signed(test[power]['difference'])}，"
                       f"{test[power]['sigmas']:+.1f} σ）" for power in ("1.0", "2.0")])
    write("round10_prediction.md", markdown_table(
        ["量", "N = 280 实测", "第九轮 1 阶外推的预测", "第九轮 2 阶外推的预测"], rows))

    rows = []
    for quantity in QUANTITIES:
        reference_value = data[quantity]["value"]
        fit = fits[quantity]
        rows.append([QUANTITY_TEXT[quantity], f"{reference_value:+.4f}",
                     f"{levels[280]['values'][quantity]:+.4f}（{signed(levels[280]['deviations'][quantity])} %）"]
                    + [f"{reference_value * (1.0 + fit['fixed'][power]['limit'] / 100.0):+.4f}"
                       f"（{signed(fit['fixed'][power]['limit'])} ± {fit['fixed'][power]['limit_error']:.2f} %）"
                       for power in ("1.0", "2.0")])
    write("round10_paper.md", markdown_table(
        ["量", "原文表 10", "N = 280（最细一级）", "外推极限，p = 1", "外推极限，p = 2"], rows))

    rows = []
    for size, entry in sorted(lid["levels"].items(), key=lambda item: int(item[0])):
        rows.append([str(size), str(entry["samples"]), " / ".join(f"{1000 * value:+.1f}" for value in entry["errors"]),
                     f"{1000 * entry['shift_mean']:.2f}（{entry['shift_spacings']:.3f} dx）",
                     f"{1000 * entry['wall_layer_difference']:+.1f}",
                     f"{1000 * entry['deficit_peak']:+.1f}（y = {entry['deficit_peak_y']:.4f}，离盖子 "
                     f"{entry['deficit_peak_spacings']:.1f} dx）"])
    write("round10_lid.md", markdown_table(
        ["N", "记录数", "u − 原文，y = 0.9766 / 0.9688 / 0.9609 / 0.9531（× 10⁻³）", "等效上移 e（× 10⁻³ L）",
         "盖子下 2.5 dx 以内的差（× 10⁻³）", "最大亏损（× 10⁻³）"], rows))


def robustness(fits: dict) -> dict:
    """How much the fixed-order fits depend on one level and on the per-level errors: every fit again without one
    level, with both orders together (limit + a / N + b / N^2), and with the errors doubled (round 9 found two runs of
    the same N at the same time 0.02-0.07 pp apart, above the plateau scatter / sqrt(windows) used here)."""
    from scipy.stats import chi2
    sizes = np.array(SIZES)
    out = {}
    for quantity in ("u_min", "v_max", "v_min"):
        values, errors = np.array(fits[quantity]["values"]), np.array(fits[quantity]["errors"])
        cases = [("all", np.ones(sizes.size, bool), 1.0)]
        cases += [(f"without {size}", sizes != size, 1.0) for size in SIZES]
        cases += [("errors x 2", np.ones(sizes.size, bool), 2.0)]
        rows = {}
        for name, keep, scale in cases:
            entry = {}
            for powers in ((1.0,), (2.0,), (1.0, 2.0)):
                if len(powers) == 2 and name != "all":
                    continue
                design = np.column_stack([np.ones(int(keep.sum()))] + [sizes[keep].astype(float) ** -power
                                                                     for power in powers])
                weights = 1.0 / (scale * errors[keep]) ** 2
                covariance = np.linalg.inv(design.T @ (weights[:, None] * design))
                parameters = covariance @ (design.T @ (weights * values[keep]))
                residual = values[keep] - design @ parameters
                degrees = int(keep.sum()) - design.shape[1]
                chi_square = float(np.sum(weights * residual ** 2))
                entry["+".join(f"{power:.0f}" for power in powers)] = {
                    "limit": float(parameters[0]), "limit_error": float(math.sqrt(covariance[0, 0])),
                    "chi_square": chi_square / degrees, "probability": float(chi2.sf(chi_square, degrees)),
                    "residual_sigmas": (residual * np.sqrt(weights)).tolist()}
            rows[name] = entry
        out[quantity] = rows
    return out


def robustness_table(checks: dict) -> None:
    from experiment.wall_crossing.cavity3d_round9 import markdown_table, write

    def cell(fit):
        return (f"{signed(fit['limit'])} ± {fit['limit_error']:.2f}（χ² {fit['chi_square']:.2f}，"
                f"P {fit['probability']:.3f}）")
    rows = []
    for quantity, cases in checks.items():
        for name, entry in cases.items():
            rows.append([QUANTITY_TEXT[quantity], {"all": "五级", "errors x 2": "五级，不确定度加倍"}.get(
                name, name.replace("without", "去掉 N ="))]
                + [cell(entry[key]) if key in entry else "" for key in ("1", "2", "1+2")]
                + [" / ".join(f"{value:+.1f}" for value in entry["1"]["residual_sigmas"])])
    write("round10_robustness.md", markdown_table(
        ["量", "用哪些级", "p = 1 的极限 %（χ² / 自由度，概率）", "p = 2", "1 阶 + 2 阶两项一起",
         "p = 1 的残差 / σ（按 N 从小到大）"], rows))


def history_series() -> dict:
    """{run: (N, times, [{u_min, v_max}])} in %: N = 100's long run, round 9's levels and continuations, N = 280."""
    long = json.loads((ROUND9 / "long_run.json").read_text(encoding="utf-8"))
    results = json.loads((ROUND9 / "levels_results.json").read_text(encoding="utf-8"))
    out = {"N100": (100, [row["time"] for row in long["series"]],
                    [{quantity: 100.0 * row[quantity] for quantity in ("u_min", "v_max")} for row in long["series"]])}
    for size in (140, 170, 200):
        history = results[str(size)]["history"]
        out[f"N{size}"] = (size, [row["time"] for row in history],
                           [{quantity: 100.0 * row[quantity] for quantity in ("u_min", "v_max")} for row in history])
        continuation = (long.get("continuations") or {}).get(str(size))
        if continuation:
            out[f"N{size}_continued"] = (size, [row["time"] for row in continuation["series"]],
                                         [{quantity: 100.0 * row[quantity] for quantity in ("u_min", "v_max")}
                                          for row in continuation["series"]])
    series = [json.loads(line) for line in (ROUND10 / "box_run" / "L280" / "extrema.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    out["N280"] = (280, [row["time"] for row in series],
                   [{quantity: row[quantity] for quantity in ("u_min", "v_max")} for row in series])
    return out


def thesis_outputs(levels: dict, costs: dict, fits: dict) -> None:
    """claude/wall_store/thesis/: the two tables (CSV), three figures (PDF + PNG, the paper's house style: Times 8 pt,
    the elsarticle text width) and every figure's data (CSV)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    THESIS.mkdir(parents=True, exist_ok=True)
    data = reference()
    font_size = 8.0
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": font_size, "axes.labelsize": font_size,
        "legend.fontsize": font_size - 1.0, "xtick.labelsize": font_size - 0.5, "ytick.labelsize": font_size - 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.edgecolor": "#333333", "axes.linewidth": 0.6,
        "xtick.color": "#333333", "ytick.color": "#333333", "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.minor.size": 1.6, "ytick.minor.size": 1.6, "lines.linewidth": 1.0, "figure.facecolor": "white",
        "axes.facecolor": "white", "legend.frameon": False, "axes.unicode_minus": True})
    text_width = 384.0 / 72.0
    colours = {100: "#9ecae1", 140: "#6baed6", 170: "#3182bd", 200: "#08519c", 280: "#08306b"}
    label = {"u_min": r"$u_{\min}$", "v_max": r"$v_{\max}$", "v_min": r"$v_{\min}$", "w_abs_max": r"$|w|_{\max}$"}

    def save(figure, name):
        figure.savefig(THESIS / f"{name}.pdf")
        figure.savefig(THESIS / f"{name}.png", dpi=300)
        plt.close(figure)
        print(f"[round10] wrote {THESIS / name}.pdf / .png", flush=True)

    def write_csv(name, header, rows):
        with open(THESIS / f"{name}.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    def tag(axis, text):
        # above the frame, left-aligned: inside it collides with the data or the zero line
        axis.set_title(f"({text})", loc="left", fontsize=font_size + 0.5, pad=2.5)

    rows = []
    for size in SIZES:
        level, cost = levels[size], costs[size]
        rows.append([size, MACHINE_ENGLISH[cost["machine"]], cost["fluid"], cost["wall_store"],
                     " / ".join(f"{cost['memory'][index]:.0f}" for index in sorted(cost["memory"])),
                     f"{cost['seconds_per_unit']:.0f}"]
                    + [f"{level['deviations'][quantity]:.3f}" for quantity in QUANTITIES]
                    + [f"{level['velocity_l2']['u']:.5f}", f"{level['velocity_l2']['v']:.5f}",
                       f"{level['pressure_l2']['vertical']:.6f}", f"{level['pressure_l2']['horizontal']:.6f}"])
    write_csv("table_levels", ["N", "hardware", "fluid particles", "wall-store particles", "memory MiB per card",
                               "wall seconds per unit time"] + [f"{quantity} deviation %" for quantity in QUANTITIES]
              + ["velocity L2 u (vertical line)", "velocity L2 v (horizontal line)", "pressure L2 vertical",
                 "pressure L2 horizontal"], rows)
    rows = []
    for quantity in QUANTITIES:
        reference_value = data["extrema"][quantity]["value"]
        fit = fits[quantity]
        rows.append([quantity, f"{reference_value:.4f}", f"{levels[280]['values'][quantity]:.5f}",
                     f"{levels[280]['deviations'][quantity]:.3f}"]
                    + [value for power in ("1.0", "2.0") for value in
                       (f"{reference_value * (1 + fit['fixed'][power]['limit'] / 100.0):.5f}",
                        f"{fit['fixed'][power]['limit']:.3f}", f"{fit['fixed'][power]['limit_error']:.3f}",
                        f"{fit['fixed'][power]['chi_square']:.2f}")]
                    + [f"{fit['free']['power']:.2f}", f"{fit['free']['power_range'][0]:.2f}",
                       f"{fit['free']['power_range'][1]:.2f}"])
    write_csv("table_paper_comparison",
              ["quantity", "Albensoeder & Kuhlmann 2005 table 10", "N = 280", "N = 280 deviation %",
               "limit p = 1", "limit p = 1 deviation %", "error", "chi2/dof p = 1",
               "limit p = 2", "limit p = 2 deviation %", "error", "chi2/dof p = 2",
               "free p", "free p 16 %", "free p 84 %"], rows)

    # figure 1: |deviation| against N (log-log) with the fixed-order fits
    figure, axes = plt.subplots(2, 2, figsize=(text_width, 0.74 * text_width))
    curve_sizes = np.geomspace(90.0, 420.0, 241)
    rows = []
    for axis, quantity, letter in zip(axes.flat, QUANTITIES, "abcd"):
        fit = fits[quantity]
        values, errors = np.array(fit["values"]), np.array(fit["errors"])
        signed_panel = bool(np.any(values > 0.0) and np.any(values < 0.0))
        axis.errorbar(SIZES, values if signed_panel else np.abs(values), yerr=errors, color="#08519c", marker="o",
                      markersize=3.2, linestyle="none", elinewidth=0.6, capsize=1.5, label="SPH, steady window",
                      zorder=3)
        for size, value, error in zip(SIZES, values, errors):
            rows.append([quantity, "SPH", size, f"{value:.5f}", f"{error:.5f}"])
        axis.set_xscale("log")
        axis.set_xticks(list(SIZES))
        axis.set_xticklabels([str(size) for size in SIZES])
        axis.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axis.set_xlabel(r"$N$ (particle spacings per side)")
        tag(axis, letter)
        if signed_panel:
            # the deviation changes sign: no power law, a log axis and fit curves would mislead
            axis.axhline(0.0, color="#333333", linewidth=0.5)
            axis.set_xlim(curve_sizes[0], curve_sizes[-1])
            axis.set_ylabel(fr"deviation of {label[quantity]} (%)")
            axis.legend(loc="upper right", handlelength=2.2, title="sign change: no power-law fit",
                        title_fontsize=font_size - 1.0)
            continue
        for power, colour, style in (("1.0", "#b2182b", (0, (4, 2))), ("2.0", "#636363", (0, (1, 1.5)))):
            entry = fit["fixed"][power]
            curve = entry["limit"] + entry["coefficient"] * curve_sizes ** -float(power)
            axis.plot(curve_sizes, np.abs(curve), color=colour, linestyle=style, linewidth=0.9,
                      label=fr"$p = {power[0]}$, limit {entry['limit']:+.2f} %".replace("-", "−"))
            for size_value, value in zip(curve_sizes[::12], curve[::12]):
                rows.append([quantity, f"fit p = {power[0]}", f"{size_value:.1f}", f"{value:.5f}", ""])
        axis.set_yscale("log")
        axis.set_ylabel(fr"|deviation of {label[quantity]}| (%)")
        axis.legend(loc="lower left", handlelength=2.2)
    figure.tight_layout(w_pad=1.2, h_pad=1.0)
    save(figure, "extrema_convergence")
    write_csv("extrema_convergence", ["quantity", "series", "N", "deviation %", "uncertainty"], rows)

    # figure 2: the finest level's centre-line velocity profiles and the paper's tables 5, 6
    fine = levels[max(SIZES)]["fine_profiles"]
    figure, axes = plt.subplots(1, 2, figsize=(text_width, 0.47 * text_width))
    rows = []
    for axis, line, letter in zip(axes, ("vertical", "horizontal"), "ab"):
        coordinate, velocity = np.array(fine[line]["coordinate"]), np.array(fine[line]["velocity"])
        reference_coordinate, reference_velocity = data[line]["coordinate"], data[line]["velocity"]
        simulated = (velocity, coordinate) if line == "vertical" else (coordinate, velocity)
        paper = (reference_velocity, reference_coordinate) if line == "vertical" else (reference_coordinate,
                                                                                         reference_velocity)
        axis.plot(*simulated, color="#08519c", linewidth=1.0, label=f"SPH, N = {max(SIZES)}")
        axis.plot(*paper, linestyle="none", marker="o", markersize=3.2, markerfacecolor="none",
                  markeredgecolor="#000000", markeredgewidth=0.6, label="Albensoeder &\nKuhlmann (2005)")
        if line == "vertical":
            axis.set_xlabel(r"$u/U$ on $(0.5, y, 0.5)$")
            axis.set_ylabel(r"$y/L$")
            axis.legend(loc="center right", bbox_to_anchor=(1.0, 0.42))
        else:
            axis.set_xlabel(r"$x/L$")
            axis.set_ylabel(r"$v/U$ on $(x, 0.5, 0.5)$")
            axis.legend(loc="lower left")
        tag(axis, letter)
        rows += [[line, "SPH", f"{c:.6f}", f"{v:.6f}"] for c, v in zip(coordinate, velocity)]
        rows += [[line, "Albensoeder & Kuhlmann 2005", f"{c:.4f}", f"{v:.7f}"]
                 for c, v in zip(reference_coordinate, reference_velocity)]
    figure.tight_layout(w_pad=1.5)
    save(figure, "centreline_profiles")
    write_csv("centreline_profiles", ["line", "series", "coordinate / L", "velocity / U"], rows)

    # figure 3: u_min and v_max against the accumulated time, the levels joined
    histories = history_series()
    figure, axes = plt.subplots(2, 1, figsize=(text_width, 0.64 * text_width), sharex=True)
    rows = []
    labelled = set()
    last_time = max(times[-1] for _size, times, _values in histories.values())
    for axis, quantity, letter in zip(axes, ("u_min", "v_max"), "ab"):
        for name, (size, times, values) in histories.items():
            show = size not in labelled and axis is axes[0]
            axis.plot(times, [row[quantity] for row in values], color=colours[size], linewidth=0.9,
                      label=f"N = {size}" if show else None)
            if show:
                labelled.add(size)
            rows += [[name, size, f"{t:.3f}", quantity, f"{row[quantity]:.4f}"] for t, row in zip(times, values)]
        for switch in LEVEL_SWITCHES:
            axis.axvline(switch, color="#7f7f7f", linewidth=0.6, linestyle=(0, (3, 2)))
        axis.axhline(0.0, color="#333333", linewidth=0.5)
        axis.set_ylabel(fr"deviation of {label[quantity]} (%)")
        axis.set_ylim(-2.6, 0.2)
        tag(axis, letter)
    axes[0].legend(loc="lower right", ncol=5, handlelength=1.6, columnspacing=1.0)
    axes[1].set_xlabel(r"accumulated time $tU/L$ (dashed: level switches at 32.16, 48.62, 53.58)")
    axes[1].set_xlim(20.0, last_time + 0.5)
    figure.tight_layout(h_pad=0.6)
    save(figure, "extrema_history")
    write_csv("extrema_history", ["run", "N", "accumulated time", "quantity", "deviation %"], rows)


def analysis_main(_arguments=None) -> int:
    steady = final_window_280()
    levels = {size: level_steady_values(size) for size in SIZES}
    costs = {size: level_costs(size) for size in SIZES}
    fits = extrapolations(levels)
    lid = lid_analysis(list(SIZES), (steady[0], steady[1]))
    two_dimensional = lid_two_dimensional()
    if two_dimensional is not None:
        from experiment.validation import cavity_reference
        coordinate, reference_velocity, _uncertainty = cavity_reference.marchi2021()["u"]
        errors = two_dimensional["mean"] - reference_velocity
        lid["two_dimensional"] = {"window": two_dimensional["window"], "records": two_dimensional["records"],
                                  "coordinate": coordinate.tolist(), "errors": errors.tolist(),
                                  "l2": float(np.sqrt(np.mean(errors ** 2)))}
    report_tables(levels, costs, fits, steady, lid)
    checks = robustness(fits)
    robustness_table(checks)
    thesis_outputs(levels, costs, fits)
    serialisable = {str(size): {key: value for key, value in level.items() if key != "profiles"}
                    for size, level in levels.items()}
    (ROUND10 / "section16.json").write_text(json.dumps(
        {"steady": steady, "levels": serialisable, "costs": {str(size): cost for size, cost in costs.items()},
         "fits": fits, "robustness": checks, "lid": lid}, indent=1, default=str), encoding="utf-8")
    for quantity in QUANTITIES:
        fit = fits[quantity]
        print(f"[round10] {quantity}: free p {fit['free']['power']:.2f} ({fit['free']['power_range'][0]:.2f}-"
              f"{fit['free']['power_range'][1]:.2f}), limit {fit['free']['limit']:+.3f}; p = 1 "
              f"{fit['fixed']['1.0']['limit']:+.3f} (chi2 {fit['fixed']['1.0']['chi_square']:.2f}); p = 2 "
              f"{fit['fixed']['2.0']['limit']:+.3f} (chi2 {fit['fixed']['2.0']['chi_square']:.2f})", flush=True)
    return 0
