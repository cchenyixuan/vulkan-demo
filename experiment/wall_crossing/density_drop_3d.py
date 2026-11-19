"""
density_drop_3d.py - report section 22 C: the density drop of the 3-D cavity (1M, wall_boundary crossing, the
volume constraint off) split by term and region, from a box_run run with --density-budget.

    .venv/Scripts/python.exe -m experiment.wall_crossing.density_drop_3d --run s22_cavity3d_1m_density_budget \
        [--window-start 0.2] [--window-end 1.0]

Reads logs/wall_crossing/box_run/<run>/series.csv. The window is the first sample at or after --window-start to the
last sample at or before --window-end (samples are at multiples of the defrag cadence). Writes
claude/wall_crossing/tables/density_drop_3d_<run>.md and logs/wall_crossing/density_drop_3d/<run>.json:

  checks       every sample: alive particles, alive fluid particles, each overflow counter, the fluid outside the box
  mean density the series' density_mean (float64 mean of the fluid's true density) over the run
  rates        over the window, in kg/m^3/s per fluid particle:
                   d(mean density)/dt from density_mean;
                   per term (continuity / delta term, fluid / wall neighbours; and the change the stored float32
                   density actually took) the budget's increase over the window / (fluid count x window length);
  regions      the same per region of density_budget.glsl's 3-D binning (which box planes lie within the support
               radius h of the particle; the lid is the upper y plane) and per group of the task: the lid's four
               edges, the other eight edges, the eight corners, the lid's middle, the other five faces' middles,
               the interior; each with its fluid particle count (at the window's ends).
A region's contribution is its budget increase divided by ALL fluid particles, so the regions add up to the totals.

A 2-D run (report section 23, B4) is recognised by its columns and uses the 2-D regions (density_budget.glsl: the lid
is the upper y side): the lid, the other three sides, the lid's two corners, the two bottom corners, the interior.
--interval-steps (default 4000): the length of the parts of the run in the table of rates over parts of the run and
the spacing of the mean-density rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.run_log import LOG_ROOT     # noqa: E402

TERM_NAMES = ("drift_fluid", "drift_wall", "diffusion_fluid", "diffusion_wall", "applied")
TERM_LABELS = {"drift_fluid": "连续性项·流体邻居", "drift_wall": "连续性项·壁面粒子",
               "diffusion_fluid": "δ 项·流体邻居", "diffusion_wall": "δ 项·壁面粒子",
               "applied": "存储密度实际变化"}
REGION_LABELS = {"lid_edge": "盖子的边（盖 + 一个侧面）", "bottom_edge": "底面的边（底 + 一个侧面）",
                 "vertical_edge": "竖棱（两个侧面）", "lid_corner": "盖子的角（盖 + 两个侧面）",
                 "bottom_corner": "底面的角（底 + 两个侧面）", "lid_middle": "盖子中部", "bottom_middle": "底面中部",
                 "side_middle": "四个侧面的中部", "interior_1h_2h": "内部 h–2h", "interior_2h_4h": "内部 2h–4h",
                 "interior_4h_8h": "内部 4h–8h", "interior_beyond_8h": "内部 ≥ 8h"}
REGION_LABELS_2D = {"top": "盖子（上边）", "left": "左边", "right": "右边", "bottom": "底边",
                    "top_left": "盖子的左角", "top_right": "盖子的右角", "bottom_left": "左下角",
                    "bottom_right": "右下角", "interior_1h_2h": "内部 h–2h", "interior_2h_4h": "内部 2h–4h",
                    "interior_4h_8h": "内部 4h–8h", "interior_beyond_8h": "内部 ≥ 8h"}
TASK_GROUPS_2D = (("盖子（上边）", ("top",)),
                  ("其余三条边", ("left", "right", "bottom")),
                  ("盖子的两个角", ("top_left", "top_right")),
                  ("底部的两个角", ("bottom_left", "bottom_right")),
                  ("内部", ("interior_1h_2h", "interior_2h_4h", "interior_4h_8h", "interior_beyond_8h")))
FINE_ORDER_2D = ("top", "left", "right", "bottom", "top_left", "top_right", "bottom_left", "bottom_right",
                 "interior_1h_2h", "interior_2h_4h", "interior_4h_8h", "interior_beyond_8h")
TASK_GROUPS = (("盖子的四条边", ("lid_edge",)),
               ("其余八条棱", ("bottom_edge", "vertical_edge")),
               ("八个角", ("lid_corner", "bottom_corner")),
               ("盖子中部", ("lid_middle",)),
               ("其余五个面的中部", ("bottom_middle", "side_middle")),
               ("内部", ("interior_1h_2h", "interior_2h_4h", "interior_4h_8h", "interior_beyond_8h")))


def read_series(path: pathlib.Path) -> list:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append({key: (float(value) if value not in ("", None) else None) for key, value in row.items()})
    return rows


def main() -> int:
    from experiment.v7.utils.simulator_v7 import DENSITY_BUDGET_REGION_NAMES, DENSITY_BUDGET_REGION_NAMES_3D
    from experiment.wall_crossing.report_tables import markdown_table, write
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--window-start", type=float, default=0.2)
    parser.add_argument("--window-end", type=float, default=1.0)
    parser.add_argument("--interval-steps", type=int, default=4000)
    arguments = parser.parse_args()
    run_directory = LOG_ROOT / "box_run" / arguments.run
    series = read_series(run_directory / "series.csv")
    summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8")) \
        if (run_directory / "summary.json").exists() else {}
    tolerance = 1e-9
    start = next(row for row in series if row["time"] >= arguments.window_start - tolerance)
    end = [row for row in series if row["time"] <= arguments.window_end + tolerance][-1]
    window = end["time"] - start["time"]
    fluid_count = int(start["fluid_alive"])
    overflow_keys = sorted(key for key in series[0] if key.startswith("overflow_"))
    # report section 23: a 2-D run has the 2-D regions
    three_dimensional = "budget_lid_middle_drift_fluid" in series[0]
    region_names = DENSITY_BUDGET_REGION_NAMES_3D if three_dimensional else DENSITY_BUDGET_REGION_NAMES
    task_groups = TASK_GROUPS if three_dimensional else TASK_GROUPS_2D
    region_labels = REGION_LABELS if three_dimensional else REGION_LABELS_2D
    fine_order = (("lid_edge", "bottom_edge", "vertical_edge", "lid_corner", "bottom_corner", "lid_middle",
                   "bottom_middle", "side_middle", "interior_1h_2h", "interior_2h_4h", "interior_4h_8h",
                   "interior_beyond_8h") if three_dimensional else FINE_ORDER_2D)

    # ---- checks over every sample
    alive_values = sorted({int(row["alive"]) for row in series})
    fluid_values = sorted({int(row["fluid_alive"]) for row in series})
    overflow_maximum = {key: max(int(row[key] or 0) for row in series) for key in overflow_keys}
    outside_maximum = max(int(row["outside_box_count"]) for row in series)
    check_rows = [["采样点数", str(len(series)), f"t = {series[0]['time']:.2f} … {series[-1]['time']:.2f}"],
                  ["全部粒子数（各采样点出现过的值）", ", ".join(f"{value:,}" for value in alive_values), ""],
                  ["活着的流体粒子数（各采样点出现过的值）", ", ".join(f"{value:,}" for value in fluid_values), ""],
                  *[[f"`{key}` 最大值", str(value), ""] for key, value in overflow_maximum.items()],
                  ["盒外流体粒子数最大值", str(outside_maximum), ""],
                  ["correction 回退次数（最后一个采样点）", str(int(series[-1]["correction_fallbacks"])), ""],
                  ["梯度回退次数（最后一个采样点）", str(int(series[-1]["gradient_fallbacks"])), ""]]

    # ---- the mean density over the run
    rest_density = 1000.0
    density_rows = [[f"{row['time']:.2f}", str(int(row["step"])), f"{row['density_mean']:.6f}",
                     f"{(row['density_mean'] / rest_density - 1.0) * 100:+.4f} %", f"{row['kinetic_energy']:.4e}"]
                    for row in series if int(row["step"]) % arguments.interval_steps == 0 or row is series[-1]]

    # ---- rates over the window
    def increase(region: str, term: str) -> float:
        key = f"budget_{region}_{term}"
        return (end[key] - start[key]) / (fluid_count * window)

    mean_density_rate = (end["density_mean"] - start["density_mean"]) / window
    term_totals = {term: sum(increase(region, term) for region in region_names)
                   for term in TERM_NAMES}
    four_terms = sum(term_totals[term] for term in TERM_NAMES[:4])
    rate_rows = [["平均密度的变化率（`density_mean` 两端之差 / 时长）", f"{mean_density_rate:+.6e}"],
                 *[[TERM_LABELS[term], f"{term_totals[term]:+.6e}"] for term in TERM_NAMES[:4]],
                 ["四项之和", f"{four_terms:+.6e}"],
                 [TERM_LABELS["applied"], f"{term_totals['applied']:+.6e}"],
                 ["存储密度实际变化 − 四项之和（float32 存储的舍入，累计）",
                  f"{term_totals['applied'] - four_terms:+.3e}"]]

    def region_counts(region: str) -> tuple:
        return int(start[f"region_count_{region}"]), int(end[f"region_count_{region}"])

    def region_row(label: str, regions: tuple) -> list:
        values = [sum(increase(region, term) for region in regions) for term in TERM_NAMES]
        counts_start = sum(region_counts(region)[0] for region in regions)
        counts_end = sum(region_counts(region)[1] for region in regions)
        return [label, *[f"{value:+.4e}" for value in values[:4]], f"{sum(values[:4]):+.4e}", f"{values[4]:+.4e}",
                f"{counts_start:,} / {counts_end:,}"]

    header = ["区域", *[TERM_LABELS[term] for term in TERM_NAMES[:4]], "四项之和", TERM_LABELS["applied"],
              f"流体粒子数（t = {start['time']:.2f} / {end['time']:.2f}）"]
    group_rows = [region_row(label, regions) for label, regions in task_groups]
    group_rows.append(region_row("合计", tuple(region_names)))
    fine_rows = [region_row(region_labels[region], (region,)) for region in fine_order]
    # per particle of the region (the group's sum of the four terms / the region's mean count, times the fluid count)
    per_particle_rows = []
    for label, regions in task_groups:
        values = sum(increase(region, term) for region in regions for term in TERM_NAMES[:4])
        mean_count = 0.5 * sum(region_counts(region)[0] + region_counts(region)[1] for region in regions)
        per_particle_rows.append([label, f"{values * fluid_count / mean_count:+.4e}" if mean_count else "—",
                                  f"{mean_count / fluid_count * 100:.2f} %"])

    # the rate over consecutive parts of the run (every fourth sample), all fluid and the interior / the rest
    interior_regions = dict(task_groups)["内部"]
    marks = [row for row in series if int(row["step"]) % arguments.interval_steps == 0]
    interval_rows = []
    for first, second in zip(marks, marks[1:]):
        length = second["time"] - first["time"]

        def interval_sum(regions, terms, first=first, second=second, length=length):
            return sum(second[f"budget_{region}_{term}"] - first[f"budget_{region}_{term}"]
                       for region in regions for term in terms) / (fluid_count * length)
        near_regions = tuple(region for region in region_names if region not in interior_regions)
        interval_rows.append([f"{first['time']:.2f} – {second['time']:.2f}",
                              f"{(second['density_mean'] - first['density_mean']) / length:+.4e}",
                              f"{interval_sum(region_names, TERM_NAMES[:4]):+.4e}",
                              f"{interval_sum(interior_regions, TERM_NAMES[:4]):+.4e}",
                              f"{interval_sum(near_regions, TERM_NAMES[:4]):+.4e}",
                              f"{interval_sum(region_names, ('applied',)):+.4e}"])

    text = "\n".join([
        f"run `{arguments.run}`, window t = {start['time']:.2f} – {end['time']:.2f} (steps {int(start['step'])} – "
        f"{int(end['step'])}), {fluid_count:,} fluid particles; rates in kg/m³/s per fluid particle "
        "(region contributions divided by all fluid particles)", "",
        "**checks**", "", markdown_table(["项目", "值", "备注"], check_rows), "",
        "**mean density**", "", markdown_table(["t", "步", "平均密度 kg/m³", "相对 ρ0", "动能 J"], density_rows), "",
        "**rates over the window**", "", markdown_table(["量", "kg/m³/s（每个流体粒子平均）"], rate_rows), "",
        "**rates over parts of the run** (kg/m³/s per fluid particle; four terms = continuity + δ, fluid + wall)", "",
        markdown_table(["t", "平均密度的变化率", "四项之和：全部", "其中内部", "其中近壁（其余区域）",
                        "存储密度实际变化"], interval_rows), "",
        "**task groups**", "", markdown_table(header, group_rows), "",
        "**fine regions**", "", markdown_table(header, fine_rows), "",
        "**per particle of the region** (four terms, divided by the region's own mean count)", "",
        markdown_table(["区域", "kg/m³/s（每个该区域粒子）", "占流体粒子"], per_particle_rows), ""])
    write(f"density_drop_3d_{arguments.run}.md", text)
    out = LOG_ROOT / "density_drop_3d"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{arguments.run}.json").write_text(json.dumps({
        "run": arguments.run, "window": [start["time"], end["time"]], "steps": [start["step"], end["step"]],
        "fluid_count": fluid_count, "mean_density_rate": mean_density_rate, "term_totals": term_totals,
        "alive_values": alive_values, "fluid_values": fluid_values, "overflow_maximum": overflow_maximum,
        "outside_maximum": outside_maximum, "summary": summary,
        "regions": {region: {"rates": {term: increase(region, term) for term in TERM_NAMES},
                             "counts": region_counts(region)} for region in region_names}},
        indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
