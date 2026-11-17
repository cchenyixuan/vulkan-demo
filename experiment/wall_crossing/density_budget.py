"""
density_budget.py - tables and figures of the density budget diagnosis: where does the drift of the fluid's mean
density in the 250^2 cavity come from?

    .venv/Scripts/python.exe -m experiment.wall_crossing.density_budget

The runs (box_run.py --density-budget, i.e. V7_DENSITY_BUDGET=1: the fused pass adds, every step and for every
fluid particle, dt x (drift fluid, drift wall, diffusion fluid, diffusion wall) of its density rate, and the change
the stored float32 density actually took in that step, to a buffer binned by the particle's region; a sample every
defrag boundary = 0.03 time units holds the buffer's sums):
    budget_n250_crossing                    wall_boundary crossing as specified
    budget_n250_simple                      wall_boundary simple
    budget_n250_crossing_continuity_normal  crossing with the diagnostic continuity_wall_component normal
What is computed from a run's series.csv, over the samples with 1 <= t <= 5:
  - per region and column the cumulative sum per fluid particle C(t) = buffer sum / fluid particle count (kg/m^3);
    the time mean of that column's field-mean rate over the window is (C(t_last) - C(t_first)) / (t_last - t_first):
    an exact time integral (every step is in it), not a mean of samples;
  - the rounding of the stored density = applied change - the sum of the four terms. The stored density is a
    float32 near 1000 kg/m^3 (grid 6.1e-5 kg/m^3), and rho_n + dt x rate is rounded to that grid every step;
  - the uncertainty of every number: the window cut into BATCHES consecutive batches of (almost) equal length, the
    same mean rate in each, standard error = standard deviation of the batch rates / sqrt(BATCHES). It measures
    how much the rate varies inside the window (the slow change of the developing flow included), i.e. how far
    another window of this length could be off; the integration itself has no sampling error;
  - closure: the applied change (all regions) against the change of the fluid's mean density between the two
    samples (float64 means of the stored float32 densities read back), which must agree to rounding of the sums.
Regions (shaders/density_budget.glsl; support radius h = the kernel radius): interior (no wall plane within h),
the lid band (top wall within h, no side wall), the static-wall bands (left, right or bottom wall within h, one
wall only), the four corner squares h x h (two walls within h), given also as the two lid corners and the two
bottom corners.

Output: claude/wall_crossing/tables/density_<run>.md, density_budget_checks.md,
claude/wall_crossing/density_budget/summary.json, claude/wall_crossing/figures/density_budget.png (the cumulative
columns against time) and density_budget_regions.png (the contributions by region).
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.report_figures import INK_PRIMARY, INK_SECONDARY, setup_style   # noqa: E402
from experiment.wall_crossing.report_tables import markdown_table, run_records                # noqa: E402
from experiment.wall_crossing.run_log import LOG_ROOT                                         # noqa: E402

REPORT_ROOT = REPOSITORY_ROOT / "claude" / "wall_crossing"
REST_DENSITY = 1000.0
WINDOW = (1.0, 5.011)
BATCHES = 8
REGION_NAMES = ("interior", "left", "right", "bottom", "bottom_left", "bottom_right", "top", "top_left", "top_right")
TERM_NAMES = ("drift_fluid", "drift_wall", "diffusion_fluid", "diffusion_wall")
APPLIED_NAME = "applied"
# the rows of a group's result: the four terms, their sum, the two partial sums, the applied change, the rounding
ROW_SUM, ROW_DRIFT, ROW_DIFFUSION, ROW_APPLIED, ROW_ROUNDING = 4, 5, 6, 7, 8
# the regions of the tables: name -> (its text, the shader's regions it is made of)
REGION_GROUPS = (
    ("lid", "盖带", ("top",)),
    ("static", "静止壁带（左、右、底）", ("left", "right", "bottom")),
    ("corners", "四个角", ("bottom_left", "bottom_right", "top_left", "top_right")),
    ("interior", "内部", ("interior",)),
)
CORNER_GROUPS = (("lid_corners", "其中：盖的两个角", ("top_left", "top_right")),
                 ("bottom_corners", "其中：底的两个角", ("bottom_left", "bottom_right")))
ALL_REGIONS = ("all", "全场", REGION_NAMES)
RUNS = {
    "budget_n250_crossing": ("crossing（`delta_wall: extension`，缺省）",
                             ("连续性项，流体邻居", "连续性项，壁邻居", "δ 项，流体邻居", "δ 项，壁延拓")),
    "budget_n250_simple": ("simple",
                           ("连续性项，流体邻居", "连续性项，壁邻居", "δ 项，流体邻居", "δ 项，壁邻居（ρ0）")),
    "budget_n250_crossing_continuity_normal": ("crossing，连续性的壁项只取法向分量（诊断）",
                                               ("连续性项，流体邻居", "连续性项，壁邻居", "δ 项，流体邻居", "δ 项，壁延拓")),
}
# the default-pipeline run each budget run is compared with (the earlier tables' run of the same case)
DEFAULT_PIPELINE_RUNS = {"budget_n250_crossing": "dw_n250_crossing_extension", "budget_n250_simple": "dw_n250_simple",
                         "budget_n250_crossing_continuity_normal": "dw_n250_crossing_continuity_normal"}
TERM_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")       # the palette's first four slots, in order
ROUNDING_COLOUR = "#6250d6"
MEASURED_COLOUR = "#e34948"


def load_series(name: str) -> dict:
    with open(LOG_ROOT / "box_run" / name / "series.csv", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def cumulative_per_particle(series: dict) -> np.ndarray:
    """(samples, regions, 5): the buffer's sums (four terms, applied change) / the fluid particle count, kg/m^3."""
    values = np.stack([np.stack([series[f"budget_{region}_{column}"] for column in (*TERM_NAMES, APPLIED_NAME)], axis=1)
                       for region in REGION_NAMES], axis=1)
    return values / series["fluid_particles"][:, None, None]


def with_derived_rows(grouped: np.ndarray) -> np.ndarray:
    """(samples, 5) -> (samples, 9): the four terms, their sum, drift sum, diffusion sum, applied, rounding."""
    four, applied = grouped[:, 0:4], grouped[:, 4:5]
    total = four.sum(axis=1, keepdims=True)
    return np.concatenate([four, total, four[:, 0:2].sum(axis=1, keepdims=True),
                           four[:, 2:4].sum(axis=1, keepdims=True), applied, applied - total], axis=1)


def mean_rate(time: np.ndarray, cumulative: np.ndarray) -> tuple:
    """(mean rate over the samples given, its standard error from BATCHES consecutive batches, the batch rates)."""
    rate = (cumulative[-1] - cumulative[0]) / (time[-1] - time[0])
    edges = np.linspace(0, len(time) - 1, BATCHES + 1).round().astype(int)
    batch_rates = np.array([(cumulative[last] - cumulative[first]) / (time[last] - time[first])
                            for first, last in zip(edges[:-1], edges[1:])])
    return rate, batch_rates.std(axis=0, ddof=1) / np.sqrt(BATCHES), batch_rates


def analyse(name: str) -> dict:
    series = load_series(name)
    time = series["time"]
    inside = (time >= WINDOW[0]) & (time <= WINDOW[1])
    window_time = time[inside]
    everything = cumulative_per_particle(series)                           # (samples, 9, 5)
    cumulative = everything[inside]
    region_index = {region: index for index, region in enumerate(REGION_NAMES)}
    counts = np.stack([series[f"region_count_{region}"] for region in REGION_NAMES], axis=1)[inside]
    fluid_particles = float(series["fluid_particles"][0])

    groups = {}
    for key, _, members in (*REGION_GROUPS, *CORNER_GROUPS, ALL_REGIONS):
        columns = [region_index[member] for member in members]
        rows = with_derived_rows(cumulative[:, columns, :].sum(axis=1))
        rate, error, batch_rates = mean_rate(window_time, rows)
        half = len(window_time) // 2
        groups[key] = {
            "rate": rate.tolist(), "standard_error": error.tolist(),
            "rate_first_half": ((rows[half] - rows[0]) / (window_time[half] - window_time[0])).tolist(),
            "rate_second_half": ((rows[-1] - rows[half]) / (window_time[-1] - window_time[half])).tolist(),
            "batch_rates": batch_rates.tolist(),
            "particle_fraction": float(counts[:, columns].sum(axis=1).mean() / fluid_particles),
            "particles_mean": float(counts[:, columns].sum(axis=1).mean())}

    density_mean = series["density_mean"][inside]
    actual_rate = float((density_mean[-1] - density_mean[0]) / (window_time[-1] - window_time[0]))
    fit_slope = float(np.polyfit(window_time, density_mean, 1)[0])
    all_rows = with_derived_rows(cumulative.sum(axis=1))                   # (samples, 9)
    applied = all_rows[:, ROW_APPLIED]
    # the applied change against the mean density read back, both relative to the window's first sample
    closure_path = (applied - applied[0]) - (density_mean - density_mean[0])
    fit_slopes = [float(np.polyfit(window_time, all_rows[:, row], 1)[0]) for row in range(all_rows.shape[1])]
    whole = with_derived_rows(everything.sum(axis=1))                      # every sample, for the figure
    return {
        "name": name, "window": [float(window_time[0]), float(window_time[-1])], "samples": int(inside.sum()),
        "batches": BATCHES, "fluid_particles": fluid_particles, "groups": groups,
        "closure": {
            "applied_rate": groups["all"]["rate"][ROW_APPLIED], "density_mean_rate": actual_rate,
            "difference": groups["all"]["rate"][ROW_APPLIED] - actual_rate,
            "largest_cumulative_difference": float(np.abs(closure_path).max()),
            "density_mean_change": float(density_mean[-1] - density_mean[0]),
            "density_mean_fit_slope": fit_slope, "fit_slopes": fit_slopes},
        "final": {"kinetic_energy": float(series["kinetic_energy"][-1]), "alive": int(series["alive"][-1]),
                  "pressure_mean": float(series["pressure_mean"][-1]),
                  "pressure_negative_fraction": float(series["pressure_negative_fraction"][-1]),
                  "time": float(time[-1]), "speed_maximum": float(series["speed_maximum"].max())},
        "series": {"time": time.tolist(), "cumulative_rows": whole.tolist(),
                   "density_mean_minus_rest": (series["density_mean"] - REST_DENSITY).tolist()}}


def value_text(rate: float, error: float) -> str:
    return f"{rate:+.5f} ± {error:.5f}"


def budget_table(result: dict, title: str, term_texts: tuple, record: dict) -> str:
    groups = result["groups"]
    columns = ["项", "全场"] + [text for _, text, _ in REGION_GROUPS] + [text for _, text, _ in CORNER_GROUPS]
    keys = ["all"] + [key for key, _, _ in REGION_GROUPS] + [key for key, _, _ in CORNER_GROUPS]
    rows = []
    for index, text in ((0, term_texts[0]), (1, term_texts[1]), (2, term_texts[2]), (3, term_texts[3]),
                        (ROW_SUM, "**四项之和**"), (ROW_ROUNDING, "float32 存储舍入（实际写入 − 四项之和）"),
                        (ROW_APPLIED, "**实际写入的密度变化**"), (ROW_DRIFT, "连续性两项之和"),
                        (ROW_DIFFUSION, "δ 两项之和")):
        rows.append([text] + [value_text(groups[key]["rate"][index], groups[key]["standard_error"][index])
                              for key in keys])
    rows.append(["区域内的流体粒子（窗口平均）"]
                + [f"{groups[key]['particles_mean']:,.0f}（{100 * groups[key]['particle_fraction']:.2f} %）" for key in keys])
    closure = result["closure"]
    slopes = closure["fit_slopes"]
    lines = [
        f"**{title}**：各项对流体平均密度变化率的贡献，kg/m³ 每时间单位，t ∈ [{result['window'][0]:.2f}, "
        f"{result['window'][1]:.2f}] 的时间平均（每一步都累加进去，不是样本平均）。每格 = 该区域内粒子的该项之和 ÷ "
        f"全部流体粒子数（{result['fluid_particles']:,.0f}），所以各区域相加等于“全场”。“实际写入” = 每一步存进去的 "
        f"float32 密度减去这一步开始时的密度；它等于四项之和加上存储舍入。± 后面是把窗口切成 {result['batches']} 段、"
        "各段平均值的标准误（含流动发展带来的慢变化）。",
        "",
        markdown_table(columns, rows),
        "",
        f"- 闭合：实际写入（全场）{closure['applied_rate']:+.6f}；同一窗口两端样本读回的平均密度之差 ÷ 时长 "
        f"{closure['density_mean_rate']:+.6f}；两者相差 {closure['difference']:+.1e}。窗口内逐样本相比，累计量与平均密度"
        f"最大相差 {closure['largest_cumulative_difference']:.1e} kg/m³（平均密度在窗口内变化了 "
        f"{closure['density_mean_change']:+.5f}）。",
        f"- 与之前的“漂移率”（平均密度的线性拟合斜率）对应：本次运行的拟合斜率是 {closure['density_mean_fit_slope']:+.5f}。"
        f"对各累计曲线做同样的拟合：四项依次 {slopes[0]:+.5f}、{slopes[1]:+.5f}、{slopes[2]:+.5f}、{slopes[3]:+.5f}，"
        f"四项之和 {slopes[ROW_SUM]:+.5f}，存储舍入 {slopes[ROW_ROUNDING]:+.5f}，实际写入 {slopes[ROW_APPLIED]:+.5f}。",
        f"- 前半窗和后半窗（全场）：四项之和 {groups['all']['rate_first_half'][ROW_SUM]:+.5f}、"
        f"{groups['all']['rate_second_half'][ROW_SUM]:+.5f}；存储舍入 {groups['all']['rate_first_half'][ROW_ROUNDING]:+.5f}、"
        f"{groups['all']['rate_second_half'][ROW_ROUNDING]:+.5f}；实际写入 "
        f"{groups['all']['rate_first_half'][ROW_APPLIED]:+.5f}、{groups['all']['rate_second_half'][ROW_APPLIED]:+.5f}。",
        f"- 记录：{record.get('run_id', '?')}（{record.get('commit', '?')}）。",
    ]
    return "\n".join(lines) + "\n"


def end_labels(axis, entries: list, x_data: float, x_text: float) -> None:
    """Values at the right edge, spread so that those that end together do not overlap."""
    low, high = axis.get_ylim()
    gap = 0.06 * (high - low)
    placed = []
    for value, text_value in sorted(entries):
        position = value if not placed or value - placed[-1] >= gap else placed[-1] + gap
        placed.append(position)
        axis.annotate(text_value, (x_data, value), xytext=(x_text, position), textcoords="data", fontsize=8,
                      color=INK_SECONDARY, va="center", annotation_clip=False)


def cumulative_figure(results: dict, names: tuple, out: pathlib.Path) -> None:
    """Per run: the four cumulative terms (per fluid particle) against time, and beside it their sum, the rounding
    of the stored density, the applied change and the mean density read back (the terms are far above their sum)."""
    setup_style()
    figure, axes = plt.subplots(len(names), 2, figsize=(13.6, 4.4 * len(names)), squeeze=False)
    for row, name in enumerate(names):
        result = results[name]
        time = np.array(result["series"]["time"])
        rows = np.array(result["series"]["cumulative_rows"])              # (samples, 9)
        rows = rows - rows[0]                                             # from the first sample (t = 0)
        title, term_texts = RUNS[name]
        title = title.replace("`", "")
        axis = axes[row][0]
        for index, (colour, text) in enumerate(zip(TERM_COLOURS, term_texts)):
            axis.plot(time, rows[:, index], color=colour, zorder=3, label=text)
        axis.axhline(0.0, color=INK_SECONDARY, linewidth=0.7, zorder=2)
        axis.set_title(f"{title}：四项各自的累计贡献（kg/m³）")
        axis.set_xlim(0, 5.01)
        end_labels(axis, [(float(rows[-1, index]), f"{rows[-1, index]:+.3f}") for index in range(4)], 5.01, 5.08)
        axis.legend(loc="best", ncol=2, handlelength=2.2, labelcolor=INK_PRIMARY, fontsize=8.5)
        closure_axis = axes[row][1]
        density = np.array(result["series"]["density_mean_minus_rest"])
        closure_axis.plot(time, rows[:, ROW_SUM], color=INK_PRIMARY, linewidth=1.6, zorder=3, label="四项之和")
        closure_axis.plot(time, rows[:, ROW_ROUNDING], color=ROUNDING_COLOUR, zorder=3, label="float32 存储舍入")
        closure_axis.plot(time, rows[:, ROW_APPLIED], color=MEASURED_COLOUR, linewidth=2.6, alpha=0.45, zorder=4,
                          label="实际写入（四项之和 + 舍入）")
        closure_axis.plot(time, density - density[0], color=MEASURED_COLOUR, linestyle=(0, (4, 2)), linewidth=1.2,
                          zorder=5, label="读回的流体平均密度的变化")
        closure_axis.axhline(0.0, color=INK_SECONDARY, linewidth=0.7, zorder=2)
        closure_axis.set_title(f"{title}：合计（kg/m³）")
        closure_axis.set_xlim(0, 5.01)
        end_labels(closure_axis, [(float(rows[-1, ROW_SUM]), f"{rows[-1, ROW_SUM]:+.4f}"),
                                  (float(rows[-1, ROW_ROUNDING]), f"{rows[-1, ROW_ROUNDING]:+.4f}"),
                                  (float(rows[-1, ROW_APPLIED]), f"{rows[-1, ROW_APPLIED]:+.4f}")], 5.01, 5.08)
        closure_axis.legend(loc="best", handlelength=2.6, labelcolor=INK_PRIMARY, fontsize=8.5)
    for axis in axes[-1]:
        axis.set_xlabel("时间（模拟单位）")
    height = figure.get_figheight()
    figure.suptitle("250² 方腔：流体平均密度的收支（每步累加，从 t = 0 起，每个流体粒子平均）", x=0.01,
                    y=1.0 - 0.05 / height, ha="left", va="top", fontsize=12, fontweight="bold", color=INK_PRIMARY)
    figure.tight_layout(rect=(0, 0, 0.985, 1.0 - 0.32 / height), w_pad=3.6)
    figure.savefig(out / "density_budget.png")
    plt.close(figure)


def region_figure(results: dict, names: tuple, out: pathlib.Path) -> None:
    """The mean rates by region, one panel per run: the sum of the two continuity terms, the sum of the two delta
    terms, the rounding of the stored density (bars with their standard errors)."""
    setup_style()
    region_keys = [key for key, _, _ in REGION_GROUPS]
    region_texts = ["盖带", "静止壁带", "四个角", "内部"]
    figure, axes = plt.subplots(1, len(names), figsize=(6.8 * len(names), 4.8), squeeze=False)
    width = 0.24
    for axis, name in zip(axes[0], names):
        result = results[name]
        title = RUNS[name][0].replace("`", "")
        positions = np.arange(len(region_keys))
        for offset, (index, colour, text) in enumerate(((ROW_DRIFT, TERM_COLOURS[0], "连续性两项之和"),
                                                        (ROW_DIFFUSION, TERM_COLOURS[2], "δ 两项之和"),
                                                        (ROW_ROUNDING, ROUNDING_COLOUR, "float32 存储舍入"))):
            values = [result["groups"][key]["rate"][index] for key in region_keys]
            errors = [result["groups"][key]["standard_error"][index] for key in region_keys]
            bars = axis.bar(positions + (offset - 1) * (width + 0.02), values, width=width, color=colour, zorder=3,
                            label=text, yerr=errors, error_kw={"ecolor": INK_PRIMARY, "elinewidth": 0.9, "capsize": 2})
            for bar, value in zip(bars, values):
                axis.annotate(f"{value:+.4f}", (bar.get_x() + bar.get_width() / 2, value),
                              xytext=(0, 5 if value >= 0 else -5), textcoords="offset points", ha="center",
                              va="bottom" if value >= 0 else "top", fontsize=7.5, color=INK_SECONDARY)
        axis.axhline(0.0, color=INK_SECONDARY, linewidth=0.8, zorder=2)
        axis.set_xticks(positions)
        axis.set_xticklabels(region_texts)
        axis.set_title(f"{title}（kg/m³ 每时间单位）")
        axis.legend(loc="best", handlelength=1.4, labelcolor=INK_PRIMARY, fontsize=8.5)
        axis.grid(axis="x", visible=False)
        low, high = axis.get_ylim()
        axis.set_ylim(low - 0.08 * (high - low), high + 0.08 * (high - low))
    height = figure.get_figheight()
    figure.suptitle("250² 方腔 t ∈ [1, 5]：对平均密度变化率的贡献，按区域", x=0.01, y=1.0 - 0.05 / height,
                    ha="left", va="top", fontsize=12, fontweight="bold", color=INK_PRIMARY)
    figure.tight_layout(rect=(0, 0, 1.0, 1.0 - 0.32 / height), w_pad=3.0)
    figure.savefig(out / "density_budget_regions.png")
    plt.close(figure)


def main() -> int:
    records = {}
    for record in run_records():
        if record.get("kind") == "box_run" and record.get("name"):
            records[record["name"]] = record
    available = [name for name in RUNS if (LOG_ROOT / "box_run" / name / "series.csv").exists()]
    results = {name: analyse(name) for name in available}
    (REPORT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    for name in available:
        title, term_texts = RUNS[name]
        text = budget_table(results[name], title, term_texts, records.get(name, {}))
        (REPORT_ROOT / "tables" / f"density_{name}.md").write_text(text, encoding="utf-8", newline="\n")
        print(text)

    # the budget runs against the default-pipeline runs of the same cases (the earlier tables' measurements)
    from experiment.wall_crossing.delta_wall_experiment import FIT_WINDOW, load_series as load_plain, measure
    rows = []
    comparison = {}
    for name in available:
        default_name = DEFAULT_PIPELINE_RUNS[name]
        if not (LOG_ROOT / "box_run" / default_name / "series.csv").exists():
            continue
        budget_measure = measure(load_plain(name), FIT_WINDOW)
        default_measure = measure(load_plain(default_name), FIT_WINDOW)
        comparison[name] = {"budget_run": budget_measure, "default_run": default_measure, "default_name": default_name}
        rows.append([RUNS[name][0], f"{budget_measure['drift_rate']:+.5f}", f"{default_measure['drift_rate']:+.5f}",
                     f"{budget_measure['pressure_mean']:+.1f} / {default_measure['pressure_mean']:+.1f}",
                     f"{budget_measure['kinetic_energy']:.3f} / {default_measure['kinetic_energy']:.3f}",
                     f"{records.get(name, {}).get('run_id', '?')} / {records.get(default_name, {}).get('run_id', '?')}"])
    if rows:
        text = markdown_table(["运行", "漂移率，打开收支", "漂移率，不打开（之前的表）", "t = 5 的平均压力（Pa），打开 / 不打开",
                               "t = 5 的动能，打开 / 不打开", "记录，打开 / 不打开"], rows)
        (REPORT_ROOT / "tables" / "density_budget_checks.md").write_text(text + "\n", encoding="utf-8", newline="\n")
        print(text)

    out = REPORT_ROOT / "density_budget"
    out.mkdir(parents=True, exist_ok=True)
    summary = {name: {key: value for key, value in result.items() if key != "series"} for name, result in results.items()}
    summary["comparison_with_default_pipeline"] = comparison
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    main_names = tuple(name for name in ("budget_n250_crossing", "budget_n250_simple") if name in results)
    if main_names:
        (REPORT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
        cumulative_figure(results, main_names, REPORT_ROOT / "figures")
        region_figure(results, main_names, REPORT_ROOT / "figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
