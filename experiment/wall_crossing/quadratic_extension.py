"""
quadratic_extension.py - the second-order continuation of the velocity of wall_boundary crossing
(numerics.wall_crossing.viscous_extension / continuity_extension, shaders/wall_boundary.glsl): its cases and the
tables and figures of section 10 of claude/wall_crossing_report.md.

    .venv/Scripts/python.exe -m experiment.wall_crossing.quadratic_extension make-cases
    .venv/Scripts/python.exe -m experiment.wall_crossing.quadratic_extension tables

Configurations (the names used in case directories, run names and tables):
    quadratic_viscous   viscous_extension quadratic, continuity_extension linear (the default continuity:
                        first order, normal component only)
    quadratic_full      viscous_extension quadratic, continuity_extension quadratic (the whole vector)
make-cases derives, from an existing crossing case, the same case with the two keys added (and, for the
screening's second attempt, distance_floor_factor 0.5): the cavity at 250^2 and 500^2 and the two 2-D
hydrostatic boxes. Existing case files are left alone (these directories hold only what this function writes).

tables writes, from the logs (every part is skipped while its runs do not exist):
    claude/wall_crossing/tables/quadratic_*.md, density_budget_n250_crossing_quadratic_*.md,
    claude/wall_crossing/figures/quadratic_*.png, claude/wall_crossing/quadratic_extension/summary.json

Colours: one per entity, as in normal_default.py (first order = its NEW_COLOUR red, the old default full blue,
simple aqua, adami yellow: that set and order passed the palette checks there). The second-order configuration
is drawn in the primary ink (no hue): no JavaScript runtime was at hand to run the palette validator on a
five-hue set, and a neutral dark line beside four validated hues needs none. Every figure has its table.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

CONFIGURATIONS = {
    "quadratic_viscous": ("quadratic", "linear"),
    "quadratic_full": ("quadratic", "quadratic"),
}
CONFIGURATION_TEXT = {"quadratic_viscous": "二阶粘性", "quadratic_full": "全二阶"}
# source case (relative to cases/) -> the suffix-free name of its variants
SOURCE_CASES = (
    "lid_driven_cavity_2d_n250_xi0p001_eps0p0025_crossing",
    "lid_driven_cavity_2d_n500_xi0p001_eps0p0025_crossing",
    "wall_crossing/hydrostatic_2d_n250_gy_crossing",
    "wall_crossing/hydrostatic_2d_n250_g30_crossing",
)
FLOOR_VARIANT_SOURCES = ("lid_driven_cavity_2d_n250_xi0p001_eps0p0025_crossing",)
SECOND_FLOOR_FACTOR = 0.5


def variant_text(source_text: str, source_name: str, configuration: str, floor_factor: float | None) -> str:
    """The source case with the configuration's keys (and the floor factor) in numerics.wall_crossing."""
    viscous_extension, continuity_extension = CONFIGURATIONS[configuration]
    lines = source_text.split("\n")
    mapping_rows = [index for index, line in enumerate(lines) if line.startswith("  wall_crossing:")]
    if len(mapping_rows) != 1:
        raise ValueError(f"{source_name}: expected exactly one numerics.wall_crossing mapping")
    if any(line.strip().startswith(("viscous_extension", "continuity_extension", "distance_floor_factor"))
           for line in lines):
        raise ValueError(f"{source_name}: the source already sets an extension or the floor factor")
    added = [f"    viscous_extension: {viscous_extension}       # linear | quadratic (second-order continuation)",
             f"    continuity_extension: {continuity_extension}    # linear | quadratic"]
    if floor_factor is not None:
        added.append(f"    distance_floor_factor: {floor_factor!r}   # default 0.25")
    lines[mapping_rows[0] + 1:mapping_rows[0] + 1] = added
    if lines[0] != "schema_version: 2":
        raise ValueError(f"{source_name}: does not start with the schema line")
    header = [f"# VARIANT of cases/{source_name} (experiment/wall_crossing/quadratic_extension.py): the second-order",
              f"# continuation of the velocity, viscous_extension {viscous_extension}, continuity_extension "
              f"{continuity_extension}"
              + (f", distance_floor_factor {floor_factor!r}" if floor_factor is not None else "")
              + "; everything else identical."]
    lines[2:2] = header + [""]
    return "\n".join(lines)


def variant_name(source_name: str, configuration: str, floor_factor: float | None) -> str:
    return f"{source_name}_{configuration}" + ("" if floor_factor is None
                                               else "_floor" + repr(floor_factor).replace(".", "p"))


def make_cases() -> list:
    written = []
    for source_name in SOURCE_CASES:
        source = REPOSITORY_ROOT / "cases" / source_name / "case.yaml"
        source_text = source.read_text(encoding="utf-8")
        floors = (None, SECOND_FLOOR_FACTOR) if source_name in FLOOR_VARIANT_SOURCES else (None,)
        for configuration in CONFIGURATIONS:
            for floor_factor in floors:
                target = REPOSITORY_ROOT / "cases" / variant_name(source_name, configuration, floor_factor) \
                    / "case.yaml"
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(variant_text(source_text, source_name, configuration, floor_factor),
                                  encoding="utf-8", newline="\n")
                written.append(target)
    return written


# ======================================================================================================== tables
# (matplotlib and the other analysis modules are imported inside tables(): make-cases needs none of them)

SCREENING_RUNS = (
    ("budget_n250_crossing_quadratic_viscous", "二阶粘性", True),
    ("budget_n250_crossing_quadratic_full", "全二阶", True),
    ("budget_n250_crossing_quadratic_full_floor0p5", "全二阶，距离下限 0.5", True),
    ("budget_n250_crossing_normal_r1", "一阶（现缺省），第 1 次", False),
    ("budget_n250_crossing_normal_r2", "一阶（现缺省），第 2 次", False),
    ("budget_n250_crossing", "旧缺省（一阶，完整向量）", False),
)
BUDGET_TERM_TEXTS = {
    "budget_n250_crossing_quadratic_viscous": ("连续性项，流体邻居", "连续性项，壁邻居（一阶，只取法向）", "δ 项，流体邻居",
                                               "δ 项，壁延拓"),
    "budget_n250_crossing_quadratic_full": ("连续性项，流体邻居", "连续性项，壁邻居（二阶，完整向量）", "δ 项，流体邻居",
                                            "δ 项，壁延拓"),
    "budget_n250_crossing_quadratic_full_floor0p5": ("连续性项，流体邻居", "连续性项，壁邻居（二阶，完整向量）",
                                                     "δ 项，流体邻居", "δ 项，壁延拓"),
}
SCREENING_LIMITS = {"speed_maximum": 1.05, "interior_continuity": 0.005, "drift_rate": 0.008}
DIAGNOSTIC_REGION_GROUPS = (("静止壁带（左、右、下）", ("left", "right", "bottom")), ("盖带", ("top",)),
                            ("下面两个角", ("bottom_left", "bottom_right")), ("盖的左角（上游）", ("top_left",)),
                            ("盖的右角（下游）", ("top_right",)))
DIFFUSION_NUMBER_FLAG = 0.1
HYDROSTATIC_DIRECTIONS = {"gy": "重力沿 −y", "g30": "重力偏离 −y 30°"}
LONG_RUN_INSTANTS = (0.2, 1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0)
RATE_WINDOWS = ((1.0, 5.0), (20.0, 60.0))
# the last four: the fallback test (single_step.py --determinant-threshold: the determinant threshold of L at the
# median of the box's determinants, 1.27 in 2-D and 1.26 in 3-D, so about half of the particles fall back)
SINGLE_STEP_SUFFIXES = (("二阶粘性", "_extension_quadratic_viscous"), ("全二阶", "_extension_quadratic_full"),
                        ("一阶（两个开关都是 linear）", ""),
                        ("二阶粘性，回退测试", "_extension_quadratic_viscous_det_threshold_1p27"),
                        ("全二阶，回退测试", "_extension_quadratic_full_det_threshold_1p27"),
                        ("二阶粘性，回退测试", "_extension_quadratic_viscous_det_threshold_1p26"),
                        ("全二阶，回退测试", "_extension_quadratic_full_det_threshold_1p26"))


def tables() -> int:
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from experiment.wall_crossing import delta_wall_experiment, density_budget, normal_default
    from experiment.wall_crossing.report_figures import INK_PRIMARY, INK_SECONDARY, SURFACE, setup_style
    from experiment.wall_crossing.report_tables import markdown_table, run_records
    from experiment.wall_crossing.run_log import LOG_ROOT

    report_root = REPOSITORY_ROOT / "claude" / "wall_crossing"
    write_table = normal_default.write_table
    record_text = normal_default.record_text
    quadratic_colour = INK_PRIMARY
    first_colour, old_colour = normal_default.NEW_COLOUR, normal_default.OLD_COLOUR
    records = run_records()
    records_by_id = {record["run_id"]: record for record in records}
    records_by_name = {}
    for record in records:
        if record.get("name"):
            records_by_name[record["name"]] = record
    summary = {}

    # ---------------------------------------------------------------- 5.1 bit-level regression
    rows = []
    manifest_now = normal_default.manifest_rows((REPOSITORY_ROOT / normal_default.MANIFEST).read_text(encoding="utf-8"))
    manifest_base = normal_default.manifest_rows(normal_default.git_show(normal_default.BASE_COMMIT,
                                                                         normal_default.MANIFEST))
    manifest_before = normal_default.manifest_rows(normal_default.git_show("398f70a", normal_default.MANIFEST))
    same_as_base = sorted(name for name in manifest_base if manifest_now.get(name) == manifest_base[name])
    changed_since = sorted(name for name in manifest_now if manifest_before.get(name) != manifest_now[name])
    rows.append(["simple、adami 用的 SPIR-V", f"基点（{normal_default.BASE_COMMIT}）清单里的 {len(manifest_base)} 个文件",
                 f"{len(same_as_base)} 个逐字节相同", "通过" if len(same_as_base) == len(manifest_base) else "**不通过**",
                 "MANIFEST.txt"])
    rows.append(["相对 398f70a 变了的 SPIR-V", "全部 " + str(len(manifest_now)) + " 个文件",
                 "、".join(name.replace(".comp.spv", "") for name in changed_since) or "无", "—", "MANIFEST.txt"])
    summary["regression"] = {"manifest_same_as_base": len(same_as_base), "manifest_base": len(manifest_base),
                             "changed_since_398f70a": changed_since}
    for name, text, reference_text in (
            ("simple_against_base_s999", "simple，250²，999 步", "基点构建的 2 个转储（2026-11-17）"),
            ("adami_against_base_s200", "adami，250²，200 步", "基点构建的 2 个转储（2026-11-17）"),
            ("linear_against_398f70a_s700", "crossing，两个开关都是 linear（缺省），250²，700 步",
             "398f70a 的求解器的 8 个转储（6c8a55a 时产生；之后到 398f70a 求解器没有改动）")):
        path = LOG_ROOT / "outcome_classes" / name / "classes.json"
        if not path.exists():
            continue
        classes = json.loads(path.read_text(encoding="utf-8"))
        members = "；".join(f"类 {index}：参照 {group['reference_runs']} 次，本次 {group['case_runs']} 次"
                           for index, group in enumerate(classes["classes"]))
        rows.append([text, reference_text, f"{len(classes['classes'])} 个结果类。{members}",
                     "逐位相同" if classes["case_within_reference_classes"] else "**不同**", classes["run_id"]])
        summary["regression"][name] = {"classes": len(classes["classes"]),
                                       "identical": classes["case_within_reference_classes"],
                                       "run_id": classes["run_id"]}
    failures = LOG_ROOT / "tests" / "failures_quadratic_extension.txt"
    base_failures = LOG_ROOT / "tests" / "failures_base.txt"
    if failures.exists() and base_failures.exists():
        now = sorted(line.strip() for line in failures.read_text(encoding="utf-8").splitlines() if line.strip())
        base = sorted(line.strip() for line in base_failures.read_text(encoding="utf-8").splitlines() if line.strip())
        seam = [record for record in records if record.get("kind") == "seam_layout_test"]
        rows.append(["`_test_seam_layout.py` 的失败清单", "基点的清单（858 条，基点自身就有）",
                     f"{len(now)} 条，" + ("与基点的清单逐行相同" if now == base else "**与基点不同**"),
                     "通过" if now == base else "**不通过**", seam[-1]["run_id"] if seam else "?"])
        summary["regression"]["seam_failures"] = {"count": len(now), "same_as_base": now == base}
    write_table("quadratic_regression.md", markdown_table(["检查", "参照", "结果", "结论", "记录"], rows))

    # ---------------------------------------------------------------- 5.3 GPU single step against numpy
    rows = []
    summary["single_step"] = {}
    for dimension in (2, 3):
        for text, suffix in SINGLE_STEP_SUFFIXES:
            for budget_build in (False, True):
                path = LOG_ROOT / "single_step" / (f"summary_{dimension}d_wall_component_normal{suffix}"
                                                   + ("_budget" if budget_build else "") + ".json")
                if not path.exists():
                    continue
                outcomes = json.loads(path.read_text(encoding="utf-8"))
                smooth = [outcome for outcome in outcomes if outcome["configuration"] == "smooth"
                          and outcome["wall_boundary"] == "crossing"]
                for outcome in smooth:
                    reference = outcome["reference"]
                    comparisons = reference["comparisons"]
                    second_order = reference.get("second_order") or {}
                    budget = outcome.get("budget")
                    gradient = comparisons.get("velocity_gradient")
                    rows.append([
                        f"{dimension}-D", text, "收支构建" if budget_build else "缺省构建", outcome["continuity"],
                        f"{outcome['particles']['fluid']:,} / {reference['counts']['fluid_with_wall_neighbours']:,}",
                        "—" if not gradient else f"{gradient['relative_error_maximum']:.1e}",
                        f"{comparisons['acceleration']['error_over_pair_term_sum_maximum']:.1e}",
                        f"{comparisons['density_rate']['relative_error_beyond_one_ulp_maximum']:.1e}",
                        "—" if not budget else f"{budget['worst_term_error_over_pair_terms_times_10']:.1e}",
                        "—" if not second_order else
                        f"{second_order['second_order_fallbacks']}（解出 {second_order['second_order_solved']}）",
                        f"{reference['worst_relative_error']:.1e}",
                        "通过" if outcome["passed"] else "**不通过**", outcome["run_id"]])
                    summary["single_step"][outcome["label"]] = {
                        "passed": outcome["passed"], "worst": reference["worst_relative_error"],
                        "run_id": outcome["run_id"], "second_order": second_order}
    if rows:
        write_table("quadratic_single_step.md", markdown_table(
            ["维数", "配置", "构建", "连续性规则", "流体粒子 / 其中有壁邻居的", "∇v_i 的相对误差（最大）",
             "加速度误差 ÷ 逐对项绝对值之和（最大）", "密度变化率的相对误差（超出 1 ulp 的部分，最大）",
             "收支四项：10 × 误差 ÷ 逐对项之和（最大）", "二阶回退的粒子", "判据值（容差 1e-4）", "结论", "记录"], rows))

    # ---------------------------------------------------------------- 5.4 screening
    def load_series(name: str) -> dict:
        with open(LOG_ROOT / "box_run" / name / "series.csv", encoding="utf-8", newline="") as handle:
            series_rows = list(csv.DictReader(handle))
        return {key: np.array([float(row[key]) if row[key] != "" else np.nan for row in series_rows])
                for key in series_rows[0]}

    rows, shell_rows, diagnostic_rows = [], [], []
    summary["screening"] = {}
    for name, text, screened in SCREENING_RUNS:
        if not (LOG_ROOT / "box_run" / name / "series.csv").exists():
            continue
        result = density_budget.analyse(name)
        series = load_series(name)
        groups = result["groups"]
        interior = groups["interior"]["rate"][density_budget.ROW_DRIFT]
        interior_error = groups["interior"]["standard_error"][density_budget.ROW_DRIFT]
        drift_rate = result["closure"]["density_mean_fit_slope"]
        conserved = bool(series["alive"].min() == series["alive"].max())
        outside = int(series["outside_box_count"].max())
        speed = float(series["speed_maximum"].max())
        checks = {"conserved": conserved and outside == 0, "speed": speed <= SCREENING_LIMITS["speed_maximum"],
                  "interior": abs(interior) <= SCREENING_LIMITS["interior_continuity"],
                  "drift": abs(drift_rate) <= SCREENING_LIMITS["drift_rate"]}
        passed = all(checks.values())
        record = records_by_name.get(name, {})
        rows.append([
            f"**{text}**" if screened else text,
            f"{int(series['alive'][0]):,}，" + ("守恒" if conserved else "**不守恒**") + f"；越出盒 {outside}",
            f"{speed:.3f}",
            f"{interior:+.5f} ± {interior_error:.5f}" + ("" if checks["interior"] else "（**超过 0.005**）"),
            f"{drift_rate:+.5f}" + ("" if checks["drift"] else "（**超过 0.008**）"),
            f"{groups['all']['rate'][density_budget.ROW_SUM]:+.5f} ± "
            f"{groups['all']['standard_error'][density_budget.ROW_SUM]:.5f}",
            f"{groups['all']['rate'][density_budget.ROW_ROUNDING]:+.5f}",
            f"{groups['all']['rate'][density_budget.ROW_APPLIED]:+.5f} ± "
            f"{groups['all']['standard_error'][density_budget.ROW_APPLIED]:.5f}",
            f"{series['pressure_mean'][-1]:+.1f} Pa；{100 * series['pressure_negative_fraction'][-1]:.2f} %",
            f"{series['kinetic_energy'][-1]:.3f}",
            ("**通过**" if passed else "**不通过**") if screened else "（参照）",
            record_text(record)])
        shell_rows.append([text] + [
            f"{groups[key]['rate'][density_budget.ROW_DRIFT]:+.5f} ± "
            f"{groups[key]['standard_error'][density_budget.ROW_DRIFT]:.5f}"
            for key in ("lid", "static", "corners", "shell_1h_2h", "shell_2h_4h", "shell_4h_8h", "shell_beyond_8h")])
        summary["screening"][name] = {
            "interior_continuity": interior, "interior_continuity_error": interior_error, "drift_rate": drift_rate,
            "checks": checks, "passed": passed if screened else None, "speed_maximum": speed,
            "pressure_mean_last": float(series["pressure_mean"][-1]),
            "pressure_negative_fraction_last": float(series["pressure_negative_fraction"][-1]),
            "run_id": record.get("run_id")}
        if name in BUDGET_TERM_TEXTS:
            title = f"crossing，{text}"
            write_table(f"density_{name}.md", density_budget.budget_table(result, title, BUDGET_TERM_TEXTS[name], record))
        if "extension_left_wall_steps" in series:
            entry = {"regions": {}}
            fallback_total, steps_total = 0.0, 0.0
            for group_text, regions in DIAGNOSTIC_REGION_GROUPS:
                fallbacks = sum(series[f"extension_{region}_fallbacks"][-1] for region in regions)
                steps = sum(series[f"extension_{region}_wall_steps"][-1] for region in regions)
                determinant = min(series[f"extension_{region}_determinant_minimum"][-1] for region in regions)
                factor = max(series[f"extension_{region}_factor_maximum"][-1] for region in regions)
                number = max(series[f"extension_{region}_diffusion_maximum"][-1] for region in regions)
                fallback_total += fallbacks
                steps_total += steps
                entry["regions"][group_text] = {"fallbacks": fallbacks, "wall_steps": steps,
                                                "determinant_minimum": determinant, "factor_maximum": factor,
                                                "diffusion_maximum": number}
                diagnostic_rows.append([
                    text, group_text, f"{fallbacks:.0f} / {steps:.3g}", f"{determinant:.4f}", f"{factor:.0f}",
                    f"{number:.3f}" + ("（**超过 0.1**）" if number > DIFFUSION_NUMBER_FLAG else "")])
            entry["fallbacks"] = fallback_total
            entry["wall_steps"] = steps_total
            if "extension_fallback" in series:
                entry["fallback_at_samples_maximum"] = int(np.nanmax(series["extension_fallback"]))
                entry["valid_at_samples"] = [int(np.nanmin(series["extension_valid"])),
                                             int(np.nanmax(series["extension_valid"]))]
            summary["screening"][name]["diagnostics"] = entry
    if rows:
        write_table("quadratic_screening.md", markdown_table(
            ["运行", "粒子数（含壁）", "流体最大速度（盖速 = 1；线 1.05）", "内部的连续性项（线 ±0.005）",
             "漂移率（平均密度的线性拟合斜率；线 ±0.008）", "四项之和（全场）", "float32 存储舍入", "实际写入",
             "t = 5 的平均压力；负压粒子比例", "t = 5 的动能", "筛查结论", "记录（提交）"], rows))
        write_table("quadratic_screening_regions.md", markdown_table(
            ["运行", "盖带", "静止壁带", "四个角", "内部：h 到 2h", "内部：2h 到 4h", "内部：4h 到 8h", "内部：8h 以外"],
            shell_rows))
    if diagnostic_rows:
        write_table("quadratic_diagnostics.md", markdown_table(
            ["运行", "区域", "回退的粒子·步 / 有壁邻居的粒子·步", "det(I − K L) 的最小值", "(1 + a_j)² 的最大值",
             "壁面粘性扩散数的最大值"], diagnostic_rows))

    # ---------------------------------------------------------------- 5.5 hydrostatic boxes
    hydrostatic_runs = {}
    for direction, direction_text in HYDROSTATIC_DIRECTIONS.items():
        entries = ((f"2d_n250_{direction}_crossing_normal_r1", "一阶（现缺省），第 1 次", 0),
                   (f"2d_n250_{direction}_crossing_normal_r2", "一阶（现缺省），第 2 次", 0),
                   (f"2d_n250_{direction}_crossing_quadratic_viscous", "二阶粘性", 1))
        present = tuple(entry for entry in entries if (LOG_ROOT / "hydrostatic" / entry[0] / "summary.json").exists())
        if any(form == 1 for _, _, form in present):
            hydrostatic_runs[direction] = (direction_text, present)
    if len(hydrostatic_runs) == len(HYDROSTATIC_DIRECTIONS):
        introduction = (
            "2-D 静水盒（250² 方腔的几何，盖静止，|g| = 10，c0 = 100，从均匀密度、零速度开始，到 t = 10.02）。速度以 "
            "√(|g| H) 为单位，压力以 ρ0 |g| H 为单位。“去电平” = 减去全场平均偏差后的误差；“电平” = 压力减去 "
            "ρ0 g·(r − r_中心) 之后对全部流体粒子的平均。一阶的两次是第 9 节的运行，二阶粘性每个方向一次。")
        hydrostatic = delta_wall_experiment.hydrostatic_comparison(
            records_by_id, runs_by_direction=hydrostatic_runs,
            forms=(("一阶（现缺省）", first_colour, "solid"), ("二阶粘性", quadratic_colour, "solid")),
            introduction=introduction, table_name="quadratic_hydrostatic.md", figure_name="quadratic_hydrostatic.png",
            figure_title="2-D 静水盒：一阶（现缺省）与二阶粘性（压力以 ρ0 g H 为单位）")
        ratio_rows = []
        summary["hydrostatic"] = {"runs": hydrostatic, "ratios": {}}
        for direction, direction_text in HYDROSTATIC_DIRECTIONS.items():
            entries = hydrostatic.get(direction, {})
            first = [entries[name]["speed_root_mean_square_over_scale_last_fifth"] for name in entries
                     if "normal_r" in name]
            second = [entries[name]["speed_root_mean_square_over_scale_last_fifth"] for name in entries
                      if name.endswith("quadratic_viscous")]
            if not first or not second:
                continue
            ratio = second[0] / (sum(first) / len(first))
            summary["hydrostatic"]["ratios"][direction] = {"first_order": first, "second_order": second[0],
                                                          "ratio_to_first_order_mean": ratio,
                                                          "ratio_to_first_order_largest": second[0] / max(first)}
            ratio_rows.append([direction_text, " / ".join(f"{value:.2e}" for value in first), f"{second[0]:.2e}",
                               f"{ratio:.2f}", "通过" if ratio <= 2.0 else "**不通过**"])
        write_table("quadratic_hydrostatic_ratio.md", markdown_table(
            ["方向", "一阶的残余速度 rms / √(gH)（末 1/5 段；两次）", "二阶粘性", "二阶粘性 ÷ 一阶两次的平均",
             "不超过 2 倍"], ratio_rows))

    # ---------------------------------------------------------------- 5.6 cavity against Marchi 2021
    cavity = normal_default.cavity(
        modes=(("quadratic_viscous", "crossing，二阶粘性", quadratic_colour, "P"),
               ("crossing_normal", "crossing，一阶（现缺省）", first_colour, "o"),
               ("crossing", "crossing，旧缺省（一阶，完整向量）", old_colour, "s"),
               ("simple", "simple", normal_default.SIMPLE_COLOUR, "^"),
               ("adami", "adami", normal_default.ADAMI_COLOUR, "D")),
        extra_runs=(("n250_crossing_quadratic_viscous", "crossing，二阶粘性，另一次运行（诊断读回每 1000 步一次）",
                     "quadratic_viscous"),
                    ("n250_crossing_normal_rho_minus", "crossing，一阶，初始密度 ρ0 − 0.1（对照）", "crossing_normal"),
                    ("n250_crossing_normal_rho_minus_0p45", "crossing，一阶，初始密度 ρ0 − 0.45（对照）",
                     "crossing_normal"),
                    ("n250_crossing_r2", "crossing，旧缺省，第二次运行", "crossing")),
        prefix="quadratic", required_mode="quadratic_viscous", labelled_modes=("quadratic_viscous", "crossing_normal"),
        labels_beside=True,
        figure_title="Re = 1000 方腔：四个极值相对 Marchi 2021 的偏差（ψ_min 用 wall 帧；数字 = 二阶粘性和一阶）")
    if cavity:
        summary["cavity"] = cavity

    # ---------------------------------------------------------------- 5.6 histories of the long runs
    long_runs = (("n250_quadratic_viscous", "250² 二阶粘性", quadratic_colour, "solid"),
                 ("n500_quadratic_viscous", "500² 二阶粘性", quadratic_colour, (0, (5, 2))),
                 ("n250_crossing_quadratic_viscous", "250² 二阶粘性，另一次运行", quadratic_colour, (0, (1, 1.5))),
                 ("n250_crossing_normal", "250² 一阶", first_colour, "solid"),
                 ("n500_crossing_normal", "500² 一阶", first_colour, (0, (5, 2))))
    histories = {name: normal_default.load_field_history(name) for name, _, _, _ in long_runs}
    histories = {name: history for name, history in histories.items() if history and len(history["time"]) > 5}
    tables_path = report_root / "cavity" / "tables.json"
    analysed = json.loads(tables_path.read_text(encoding="utf-8"))["runs"] if tables_path.exists() else {}
    if "n250_quadratic_viscous" in histories or "n500_quadratic_viscous" in histories:
        summary["long_runs"] = {}
        rows, summary_rows = [], []
        for name, text, _, _ in long_runs:
            history = histories.get(name)
            if not history:
                continue
            time = history["time"]
            entry = {"t_last": float(time[-1]), "at_times": {}}
            for instant in LONG_RUN_INSTANTS:
                if instant > time[-1] + 0.3:
                    continue
                index = int(np.argmin(np.abs(time - instant)))
                entry["at_times"][f"{instant:g}"] = {
                    "time": float(time[index]),
                    "density_mean_minus_rest": float(history["density_mean"][index] - normal_default.REST_DENSITY),
                    "pressure_mean": float(history["pressure_mean"][index]),
                    "pressure_negative_fraction": float(history["pressure_negative_fraction"][index])}
                rows.append([text, f"{time[index]:.2f}",
                             f"{history['density_mean'][index] - normal_default.REST_DENSITY:+.5f}",
                             f"{history['pressure_mean'][index] / normal_default.DYNAMIC_PRESSURE:+.3f} ρ0U²",
                             f"{100.0 * history['pressure_negative_fraction'][index]:.3f} %",
                             f"{history['speed_maximum'][index]:.3f}", f"{int(history['alive'][index]):,}"])
            early = (time >= 1.0) & (time <= 5.011)
            late = time >= 20.0
            last = time >= time[-1] - 20.0
            crossing = (analysed.get(name, {}).get("diagnostics") or {}).get("crossing") or {}
            status_path = LOG_ROOT / "cavity" / name / "wall_crossing_status.jsonl"
            extension_fallbacks = None
            if status_path.exists():
                status_rows = [json.loads(line) for line in status_path.read_text(encoding="utf-8").splitlines()
                               if line.strip()]
                counts = [row["extension_fallback"] for row in status_rows if "extension_fallback" in row]
                if counts:
                    extension_fallbacks = (max(counts), len(counts))
            result_path = LOG_ROOT / "cavity" / name / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
            entry.update({
                "alive_constant": bool((history["alive"] == history["alive"][0]).all()),
                "speed_maximum": float(history["speed_maximum"].max()),
                "density_mean_rate_1_to_5": float(np.polyfit(time[early], history["density_mean"][early], 1)[0]),
                "density_mean_rate_20_to_end": float(np.polyfit(time[late], history["density_mean"][late], 1)[0])
                if late.sum() >= 3 else None,
                "density_mean_rate_last_20": float(np.polyfit(time[last], history["density_mean"][last], 1)[0]),
                "pressure_negative_fraction_maximum_after_5": float(
                    history["pressure_negative_fraction"][time >= 5.0].max()),
                "extension_fallbacks_at_reads": extension_fallbacks, "crossing_status": crossing,
                "invariant_segments": len(result.get("segments_with_invariant_violation", [])) if result else None})
            summary["long_runs"][name] = entry
            summary_rows.append([
                text, f"t = {time[-1]:.2f}",
                f"{int(history['alive'][0]):,}，" + ("每个样本都相同" if entry["alive_constant"] else "**有变化**"),
                f"{entry['speed_maximum']:.3f}",
                f"{entry['density_mean_rate_1_to_5']:+.5f}",
                "—" if entry["density_mean_rate_20_to_end"] is None else f"{entry['density_mean_rate_20_to_end']:+.6f}",
                f"{entry['density_mean_rate_last_20']:+.6f}",
                f"{100.0 * entry['pressure_negative_fraction_maximum_after_5']:.3f} %",
                "—" if not crossing else str(crossing["gradient_fallbacks"]),
                "—（一阶）" if extension_fallbacks is None else f"{extension_fallbacks[0]}（{extension_fallbacks[1]} 次读回）",
                "—" if not crossing or crossing.get("wall_distance_minimum_over_spacing") is None
                else f"{crossing['wall_distance_minimum_over_spacing']:.3f}",
                "—" if not crossing or crossing.get("below_floor_maximum") is None
                else f"{crossing['below_floor_maximum']} / {crossing['outside_box_maximum']}",
                "运行中" if not result else f"不变量违反的段：{entry['invariant_segments']}",
                record_text(records_by_name.get(name, {}))])
        write_table("quadratic_long_runs.md", markdown_table(
            ["运行", "t", "平均密度 − ρ0（kg/m³）", "平均压力", "负压粒子比例", "流体最大速度", "粒子数（含壁）"], rows))
        write_table("quadratic_long_runs_summary.md", markdown_table(
            ["运行", "到达", "粒子数（含壁）", "流体最大速度（整个运行）", "平均密度变化率，t ∈ [1, 5]", "同，t ∈ [20, 末]",
             "同，最后 20 个时间单位", "负压粒子比例的最大值（t ≥ 5）", "A_f 回退（累计）", "二阶回退的粒子（各次读回的最大值）",
             "φ_min / dx", "低于下限 / 越出盒（最大粒子数）", "运行器的结论", "记录（提交）"], summary_rows))
        setup_style()
        figure, axes = plt.subplots(1, 3, figsize=(13.6, 4.2))
        panels = (("density_mean", lambda values: values - normal_default.REST_DENSITY, "流体平均密度 − ρ0（kg/m³）"),
                  ("pressure_mean", lambda values: values / normal_default.DYNAMIC_PRESSURE, "流体平均压力 / (ρ0 U²)"),
                  ("pressure_negative_fraction", lambda values: 100.0 * values, "负压粒子比例（%），t ≥ 1，5 个时间单位的滑动平均"))
        handles, labels = [], []
        for axis, (key, transform, title) in zip(axes, panels):
            for name, text, colour, style in long_runs:
                history = histories.get(name)
                if not history or name == "n250_crossing_quadratic_viscous":
                    continue
                # the share of negative pressures from t = 1 on: the first samples (the start from rest; the
                # table has t = 0.2) are tens of times larger than everything after them
                shown = history["time"] >= (1.0 if key == "pressure_negative_fraction" else 0.0)
                values = transform(history[key][shown])
                instants = history["time"][shown]
                if key == "pressure_negative_fraction":
                    # a few particles out of 63,001 or 251,001: the samples jump between 0 and one or two
                    # particles, so the running mean over 25 samples (5 time units) is drawn, where the window
                    # is complete
                    values = np.convolve(values, np.ones(25) / 25.0, mode="valid")
                    instants = instants[12:len(instants) - 12]
                line, = axis.plot(instants, values, color=colour, linestyle=style, zorder=3, linewidth=1.6)
                if axis is axes[0]:
                    handles.append(line)
                    labels.append(text)
            if key != "pressure_negative_fraction":
                axis.axhline(0.0, color=INK_SECONDARY, linewidth=0.7, zorder=2)
            axis.set_xlim(0, 100)
            axis.set_title(title)
            axis.set_xlabel("时间（模拟单位）")
        height = figure.get_figheight()
        figure.suptitle("Re = 1000 方腔：统计运行全程的密度与压力，二阶粘性与一阶", x=0.01, y=1.0 - 0.05 / height,
                        ha="left", va="top", fontsize=12, fontweight="bold", color=INK_PRIMARY)
        figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0 - 0.40 / height),
                      ncol=len(handles), handlelength=3.0, columnspacing=1.6, labelcolor=INK_PRIMARY,
                      borderaxespad=0.0)
        figure.tight_layout(rect=(0, 0, 1.0, 1.0 - 0.66 / height), w_pad=2.6)
        (report_root / "figures").mkdir(parents=True, exist_ok=True)
        figure.savefig(report_root / "figures" / "quadratic_long_runs.png")
        plt.close(figure)

    # ---------------------------------------------------------------- 5.7 the pressure-branch control
    controls = (("n250_crossing_normal", "一阶（现缺省），正常起步", None),
                ("n250_crossing_normal_rho_minus", "一阶，初始密度 ρ0 − 0.1", -0.1),
                ("n250_crossing_normal_rho_minus_0p45", "一阶，初始密度 ρ0 − 0.45", -0.45),
                ("n250_crossing", "旧缺省（一阶，完整向量），正常起步", None))
    control_histories = {name: normal_default.load_field_history(name) for name, _, _ in controls}
    # a run that has not reached t = 99 yet (still running) is left out
    control_histories = {name: history for name, history in control_histories.items()
                         if history and history["time"][-1] >= 99.0}
    if control_histories.get("n250_crossing_normal_rho_minus"):
        summary["control"] = {}
        rows, history_rows = [], []
        for name, text, offset in controls:
            history = control_histories.get(name)
            run = analysed.get(name)
            entry = {}
            if history:
                time, fraction = history["time"], history["pressure_negative_fraction"]
                below = time[fraction < 0.95]
                window = time >= 80.0
                entry = {"first_time_below_95_percent": float(below.min()) if below.size else None,
                         "negative_fraction_minimum_from_50": float(fraction[time >= 50.0].min()),
                         "negative_fraction_window_mean": float(fraction[window].mean()),
                         "pressure_mean_window": float(history["pressure_mean"][window].mean()),
                         "density_rate_1_to_20": float(np.polyfit(time[(time >= 1.0) & (time <= 20.0)],
                                                                  history["density_mean"][(time >= 1.0) & (time <= 20.0)],
                                                                  1)[0])}
                for instant in (0.2, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 80.0, 100.0):
                    index = int(np.argmin(np.abs(time - instant)))
                    history_rows.append([text, f"{time[index]:.2f}",
                                         f"{history['density_mean'][index] - normal_default.REST_DENSITY:+.5f}",
                                         f"{history['pressure_mean'][index] / normal_default.DYNAMIC_PRESSURE:+.3f} ρ0U²",
                                         f"{100.0 * fraction[index]:.2f} %"])
            if run and not run["failed"]:
                extrema = run["extrema"]
                entry["psi_min_deviation_wall_percent"] = extrema["psi_min"]["deviation_wall_percent"]
                valid = None if offset is None or not history else entry["negative_fraction_minimum_from_50"] >= 0.95
                entry["valid"] = valid
                rows.append([
                    text, f"{extrema['psi_min']['deviation_wall_percent']:+.2f} %",
                    " / ".join(f"{extrema[key]['deviation_percent']:+.2f} %" for key in ("u_min", "v_max", "v_min")),
                    f"{run['errors']['wall_u']['L2']:.4f} / {run['errors']['wall_v']['L2']:.4f}",
                    "—（没有记历史）" if not history else
                    f"{100.0 * entry['negative_fraction_window_mean']:.2f} %；"
                    f"{entry['pressure_mean_window'] / normal_default.DYNAMIC_PRESSURE:+.3f} ρ0U²",
                    "—" if not history else
                    ("一直在 95 % 以上" if entry["first_time_below_95_percent"] is None
                     else f"t = {entry['first_time_below_95_percent']:.1f} 起低于 95 %"),
                    "（参照）" if offset is None else ("**有效**" if valid else "**无效**（t ≥ 50 时负压比例低于 95 %）"),
                    record_text(records_by_name.get(name, {}))])
            summary["control"][name] = entry
        write_table("quadratic_control.md", markdown_table(
            ["运行（250²，到 t = 100）", "ψ_min 偏差（wall 帧）", "u_min / v_max / v_min 偏差", "u / v 剖面 L2（wall 帧）",
             "平均窗口内（t ≥ 80）：负压粒子比例；平均压力", "负压粒子比例的历史", "对照是否有效", "记录（提交）"], rows))
        write_table("quadratic_control_history.md", markdown_table(
            ["运行", "t", "平均密度 − ρ0（kg/m³）", "平均压力", "负压粒子比例"], history_rows))

    # ---------------------------------------------------------------- 5.7 distance of the first fluid layer to the walls
    # Snapshots of the 250^2 cavity runs with t in [90, 100] (the runner keeps the last 21, one per time unit):
    # the fluid particles' distance to each of the four wall planes of the crossing box (the fluid bounding box
    # of the lattice widened by one particle radius: +-0.502), away from the corners (within the support radius
    # of another wall), in units of the spacing. First layer = distance below one spacing.
    layer_runs = (("n250_crossing", "旧缺省（一阶，完整向量）", old_colour, "solid"),
                  ("n250_crossing_r2", "旧缺省，第二次运行", old_colour, (0, (4, 2))),
                  ("n250_crossing_normal", "一阶（现缺省）", first_colour, "solid"),
                  ("n250_quadratic_viscous", "二阶粘性", quadratic_colour, "solid"),
                  ("n250_crossing_quadratic_viscous", "二阶粘性，另一次运行", quadratic_colour, (0, (4, 2))),
                  ("n250_simple", "simple", normal_default.SIMPLE_COLOUR, "solid"),
                  ("n250_adami", "adami", normal_default.ADAMI_COLOUR, "solid"))
    walls = (("左壁", 0, -1), ("右壁", 0, +1), ("底壁", 1, -1), ("盖（上壁）", 1, +1))
    half_box, bins = 0.502, np.linspace(0.0, 3.0, 61)
    layer_data = {}
    for name, text, _, _ in layer_runs:
        directory = LOG_ROOT / "cavity" / name
        if not (directory / "meta.json").exists() or not (directory / "snapshots").is_dir():
            continue
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        spacing, support, timestep = float(meta["spacing"]), float(meta["support_radius"]), float(meta["dt"])
        snapshots = [(int(path.stem[1:]) * timestep, path) for path in sorted((directory / "snapshots").glob("t*.npz"))]
        snapshots = [(instant, path) for instant, path in snapshots if 89.5 <= instant <= 100.5]
        if not snapshots:
            continue
        per_wall = {wall_text: {"first": [], "second": [], "counts": [], "histogram": np.zeros(len(bins) - 1)}
                    for wall_text, _, _ in walls}
        for _, path in snapshots:
            data = np.load(path)
            fluid = np.isin(data["material"], meta["fluid_groups"])
            positions = data["positions"][fluid].astype(np.float64)
            for wall_text, axis_index, side in walls:
                distance = (half_box - side * positions[:, axis_index]) / spacing
                along = np.abs(positions[:, 1 - axis_index]) <= half_box - support
                selected = distance[along]
                per_wall[wall_text]["first"].append(selected[selected < 1.0])
                per_wall[wall_text]["second"].append(selected[(selected >= 1.0) & (selected < 2.0)])
                per_wall[wall_text]["counts"].append(int((selected < 1.0).sum()))
                per_wall[wall_text]["histogram"] += np.histogram(selected, bins=bins)[0]
        lattice_count = 2.0 * (half_box - support) / spacing
        layer_data[name] = {"snapshots": len(snapshots), "times": [snapshots[0][0], snapshots[-1][0]], "walls": {}}
        for wall_text, _, _ in walls:
            first = np.concatenate(per_wall[wall_text]["first"])
            second = np.concatenate(per_wall[wall_text]["second"])
            layer_data[name]["walls"][wall_text] = {
                "count_per_snapshot": float(np.mean(per_wall[wall_text]["counts"])),
                "count_over_lattice": float(np.mean(per_wall[wall_text]["counts"]) / lattice_count),
                "mean": float(first.mean()), "standard_deviation": float(first.std()),
                "percentiles_5_50_95": [float(value) for value in np.percentile(first, (5, 50, 95))],
                "minimum": float(first.min()), "second_layer_mean": float(second.mean()),
                "histogram": (per_wall[wall_text]["histogram"] / len(snapshots)).tolist()}
    if "n250_crossing" in layer_data and "n250_crossing_normal" in layer_data:
        rows = []
        for wall_text, _, _ in walls:
            for name, text, _, _ in layer_runs:
                if name not in layer_data:
                    continue
                entry = layer_data[name]["walls"][wall_text]
                low, median, high = entry["percentiles_5_50_95"]
                rows.append([wall_text, text, f"{entry['count_per_snapshot']:.1f}（{100 * entry['count_over_lattice']:.1f} %）",
                             f"{entry['mean']:.3f}", f"{entry['standard_deviation']:.3f}",
                             f"{low:.3f} / {median:.3f} / {high:.3f}", f"{entry['minimum']:.3f}",
                             f"{entry['second_layer_mean']:.3f}"])
        first_entry = layer_data["n250_crossing_normal"]
        write_table("quadratic_wall_layer.md",
                    f"250² 方腔，t ≈ {first_entry['times'][0]:.0f} 到 {first_entry['times'][1]:.0f} 的 "
                    f"{first_entry['snapshots']} 个快照合在一起：流体粒子到各壁面（crossing 的盒面，±0.502）的距离，以粒子间距 "
                    "dx 为单位；只取离另外两面壁超过核半径 h 的那一段（每面壁中间 0.964 的长度，点阵上是 241 个粒子）。"
                    "第一层 = 距离小于 1 dx 的粒子；点阵上第一层在 0.5 dx，第二层在 1.5 dx。\n\n"
                    + markdown_table(["壁", "运行", "第一层的粒子数（每个快照；相对点阵）", "第一层距离的平均（dx）",
                                      "标准差", "5 % / 中位 / 95 % 分位", "最小", "第二层（1 到 2 dx）的平均"], rows))
        setup_style()
        figure, axes = plt.subplots(1, 4, figsize=(13.6, 4.0), sharey=True)
        centres = 0.5 * (bins[:-1] + bins[1:])
        handles, labels = [], []
        for axis, (wall_text, _, _) in zip(axes, walls):
            for name, text, colour, style in layer_runs:
                if name not in layer_data or name in ("n250_simple", "n250_adami", "n250_crossing_r2",
                                                      "n250_crossing_quadratic_viscous"):
                    continue
                line, = axis.plot(centres, layer_data[name]["walls"][wall_text]["histogram"], color=colour,
                                  linestyle=style, linewidth=1.5, zorder=3)
                if axis is axes[0]:
                    handles.append(line)
                    labels.append(text)
            axis.set_title(wall_text)
            axis.set_xlabel("到壁面的距离 / dx")
            axis.set_xlim(0.0, 3.0)
        axes[0].set_ylabel("每个快照、每 0.05 dx 一格里的粒子数")
        height = figure.get_figheight()
        figure.suptitle("250² 方腔，t ≈ 90 到 100：近壁流体粒子到壁面的距离分布（四面壁分开）", x=0.01,
                        y=1.0 - 0.05 / height, ha="left", va="top", fontsize=12, fontweight="bold", color=INK_PRIMARY)
        figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0 - 0.40 / height),
                      ncol=len(handles), handlelength=3.0, columnspacing=1.6, labelcolor=INK_PRIMARY,
                      borderaxespad=0.0)
        figure.tight_layout(rect=(0, 0, 1.0, 1.0 - 0.66 / height), w_pad=1.6)
        (report_root / "figures").mkdir(parents=True, exist_ok=True)
        figure.savefig(report_root / "figures" / "quadratic_wall_layer.png")
        plt.close(figure)
        summary["wall_layer"] = {name: {key: value for key, value in entry.items()} for name, entry in layer_data.items()}

    # ---------------------------------------------------------------- 5.8 time per step and register counts
    def step_rate(history: dict, window: tuple) -> float | None:
        inside = (history["time"] >= window[0]) & (history["time"] <= window[1])
        if inside.sum() < 3:
            return None
        steps = history["step"][inside]
        return None if "unix" not in history else float((steps[-1] - steps[0]) / (history["unix"][inside][-1]
                                                                                 - history["unix"][inside][0]))

    def load_history_with_clock(name: str) -> dict:
        path = LOG_ROOT / "cavity" / name / "field_history.jsonl"
        if not path.exists():
            return {}
        by_step = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("step") is not None:
                    by_step[int(row["step"])] = row
        steps = sorted(by_step)
        return {key: np.array([by_step[step][key] for step in steps], dtype=float) for key in ("step", "time", "unix")}

    rows = []
    summary["timing"] = {}
    for resolution, card in ((250, "接显示器的 5090"), (500, "无显示器的 5090")):
        first = load_history_with_clock(f"n{resolution}_crossing_normal")
        second = load_history_with_clock(f"n{resolution}_quadratic_viscous")
        if not first or not second or len(second["time"]) < 30:
            continue
        for window in RATE_WINDOWS + ((0.0, float(min(first["time"][-1], second["time"][-1]))),):
            first_rate, second_rate = step_rate(first, window), step_rate(second, window)
            if first_rate is None or second_rate is None:
                continue
            ratio = first_rate / second_rate
            summary["timing"][f"n{resolution}_{window[0]:g}_{window[1]:g}"] = {
                "first_order_steps_per_second": first_rate, "second_order_steps_per_second": second_rate,
                "time_per_step_ratio": ratio}
            rows.append([f"{resolution}²（{card}）", f"t ∈ [{window[0]:g}, {window[1]:.0f}]",
                         f"{first_rate:.0f} 步/秒（{1e6 / first_rate:.0f} µs/步）",
                         f"{second_rate:.0f} 步/秒（{1e6 / second_rate:.0f} µs/步）", f"{ratio:.3f}"])
    if rows:
        write_table("quadratic_timing.md", markdown_table(
            ["算例（卡）", "窗口", "一阶（第 9 节的运行）", "二阶粘性", "每步时间：二阶粘性 ÷ 一阶"], rows))
    rows = []
    summary["registers"] = {}
    for case_suffix, text in (("crossing", "一阶（两个开关都是 linear）"), ("crossing_quadratic_viscous", "二阶粘性"),
                              ("crossing_quadratic_full", "全二阶")):
        path = LOG_ROOT / "statistics" / f"quadratic_n250_{case_suffix}.json"
        if not path.exists():
            continue
        pipelines = json.loads(path.read_text(encoding="utf-8"))["pipelines"]
        cells = [text]
        for pipeline in ("correction_density_all", "force_all"):
            statistics = pipelines[pipeline][0]["statistics"]
            cells += [str(statistics["Register Count"]), f"{statistics['Binary Size']:,}"]
            summary["registers"][f"{case_suffix}_{pipeline}"] = statistics
        rows.append(cells)
    if rows:
        write_table("quadratic_registers.md", markdown_table(
            ["配置（250² 的 2-D 算例）", "融合遍：寄存器数", "融合遍：二进制大小（字节）", "force：寄存器数",
             "force：二进制大小（字节）"], rows))

    # ---------------------------------------------------------------- 5.8 K = 1 against K = 2
    k2_path = LOG_ROOT / "k2_equivalence" / "n250_crossing_quadratic_viscous" / "summary.json"
    if k2_path.exists():
        summary["k2_equivalence"] = json.loads(k2_path.read_text(encoding="utf-8"))

    out = report_root / "quadratic_extension"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8", newline="\n")
    print(f"[quadratic_extension] tables written; parts: {', '.join(summary)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("make-cases")
    subparsers.add_parser("tables")
    arguments = parser.parse_args()
    if arguments.command == "make-cases":
        for path in make_cases():
            print(f"[quadratic_extension] wrote {path.relative_to(REPOSITORY_ROOT)}")
        return 0
    return tables()


if __name__ == "__main__":
    sys.exit(main())
