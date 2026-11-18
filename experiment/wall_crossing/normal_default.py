"""
normal_default.py - tables and figures of the default change of 2026-11-18: numerics.wall_crossing.
continuity_wall_component full -> normal (a wall neighbour's velocity difference enters the continuity equation
with its component along the wall's inward normal only).

    .venv/Scripts/python.exe -m experiment.wall_crossing.normal_default

Reads what the run tools wrote (nothing is typed in here) and skips a part whose runs are not there yet:
  regression      logs/wall_crossing/outcome_classes/*/classes.json, single_step/summary_*.json, tests/, the SPIR-V
                  manifests of the base (eda5b90), of the commit before the change (2351a4d) and the files now
  drift           box_run nd_n250_crossing_r1, _r2 (the default pipeline) and the budget runs below, beside the
                  earlier runs of full, simple and adami (delta_wall_experiment.measure: linear fit, 1 <= t <= 5)
  budget          box_run budget_n250_crossing_normal_r1, _r2 (density_budget.analyse) beside the runs of full
                  and the run of normal made with the switch before the change
  delta density   box_run budget_n250_crossing_normal_delta_density (V7_DELTA_DENSITY=1)
  hydrostatic     hydrostatic 2d_n250_<gy|g30>_crossing_normal_r1, _r2 beside the two runs of full
                  (delta_wall_experiment.hydrostatic_comparison with this run set)
  cavity          claude/wall_crossing/cavity/tables.json (cavity_tables.py): n250 / n500 of crossing_normal,
                  crossing (full), simple, adami against Marchi 2021, and the observed order 250^2 -> 500^2
  long runs       logs/wall_crossing/cavity/n<250|500>_crossing_normal/field_history.jsonl (cavity_runner_entry:
                  the fluid's mean density, mean pressure, share of negative pressure, largest speed, particle
                  count at every sample); the pressure of the earlier runs from their last checkpoints

Output: claude/wall_crossing/tables/normal_default_*.md, claude/wall_crossing/figures/normal_default_*.png,
claude/wall_crossing/normal_default/summary.json.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing import delta_wall_experiment, density_budget                         # noqa: E402
from experiment.wall_crossing.report_figures import INK_PRIMARY, INK_SECONDARY, SURFACE, setup_style   # noqa: E402
from experiment.wall_crossing.report_tables import markdown_table, run_records                    # noqa: E402
from experiment.wall_crossing.run_log import LOG_ROOT                                             # noqa: E402

REPORT_ROOT = REPOSITORY_ROOT / "claude" / "wall_crossing"
REST_DENSITY = 1000.0
DYNAMIC_PRESSURE = 1000.0           # rho0 U^2, lid speed 1
BASE_COMMIT = "eda5b90"
COMMIT_BEFORE_CHANGE = "2351a4d"
MANIFEST = "experiment/v7/shaders/spv/MANIFEST.txt"
# one colour per entity in every figure of this module (validate_palette of the dataviz skill, light surface, in
# this order: all checks pass; aqua and yellow are below 3:1 against the surface, so every figure has its table)
NEW_COLOUR, OLD_COLOUR, SIMPLE_COLOUR, ADAMI_COLOUR = "#e34948", "#2a78d6", "#1baf7a", "#eda100"
NEW_TEXT, OLD_TEXT = "连续性壁项 normal（新缺省）", "连续性壁项 full（旧缺省）"

DRIFT_RUNS = (
    # (box_run name, text, pipeline text)
    ("nd_n250_crossing_r1", "**crossing，新缺省，第 1 次**", "缺省管线"),
    ("nd_n250_crossing_r2", "**crossing，新缺省，第 2 次**", "缺省管线"),
    ("budget_n250_crossing_normal_r1", "**crossing，新缺省，第 1 次**", "收支构建"),
    ("budget_n250_crossing_normal_r2", "**crossing，新缺省，第 2 次**", "收支构建"),
    ("dw_n250_crossing_continuity_normal", "crossing，normal（改缺省之前用开关）", "缺省管线"),
    ("budget_n250_crossing_continuity_normal", "crossing，normal（改缺省之前用开关）", "收支构建"),
    ("dw_n250_crossing_extension", "crossing，full（旧缺省）", "缺省管线"),
    ("budget_n250_crossing", "crossing，full（旧缺省），第 1 次", "收支构建"),
    ("budget_n250_crossing_repeat", "crossing，full（旧缺省），第 2 次", "收支构建"),
    ("dw_n250_simple", "simple", "缺省管线"),
    ("dw_n250_adami", "adami", "缺省管线"),
)
NEW_DEFAULT_DRIFT_RUNS = ("nd_n250_crossing_r1", "nd_n250_crossing_r2", "budget_n250_crossing_normal_r1",
                          "budget_n250_crossing_normal_r2")
BUDGET_COLUMNS = (
    ("budget_n250_crossing", "full（旧缺省），第 1 次"),
    ("budget_n250_crossing_repeat", "full（旧缺省），第 2 次"),
    ("budget_n250_crossing_continuity_normal", "normal，改缺省之前用开关"),
    ("budget_n250_crossing_normal_r1", "**normal（新缺省），第 1 次**"),
    ("budget_n250_crossing_normal_r2", "**normal（新缺省），第 2 次**"),
)
NEW_BUDGET_RUNS = ("budget_n250_crossing_normal_r1", "budget_n250_crossing_normal_r2")
DELTA_DENSITY_RUN = "budget_n250_crossing_normal_delta_density"
HYDROSTATIC_RUNS = {
    "gy": ("重力沿 −y", (
        ("2d_n250_gy_crossing", "full，此前的运行", 0),
        ("2d_n250_gy_crossing_history", "full，重复（带历史）", 0),
        ("2d_n250_gy_crossing_normal_r1", "normal（新缺省），第 1 次", 1),
        ("2d_n250_gy_crossing_normal_r2", "normal（新缺省），第 2 次", 1))),
    "g30": ("重力偏离 −y 30°", (
        ("2d_n250_g30_crossing", "full，此前的运行", 0),
        ("2d_n250_g30_crossing_history", "full，重复（带历史）", 0),
        ("2d_n250_g30_crossing_normal_r1", "normal（新缺省），第 1 次", 1),
        ("2d_n250_g30_crossing_normal_r2", "normal（新缺省），第 2 次", 1)))}
HYDROSTATIC_FORMS = ((OLD_TEXT, OLD_COLOUR, "solid"), (NEW_TEXT, NEW_COLOUR, "solid"))
# cavity modes of cavity_tables.py: (mode key, text, colour, marker)
CAVITY_MODES = (("crossing_normal", "crossing，normal（新缺省）", NEW_COLOUR, "o"),
                ("crossing", "crossing，full（旧缺省）", OLD_COLOUR, "s"),
                ("simple", "simple", SIMPLE_COLOUR, "^"),
                ("adami", "adami", ADAMI_COLOUR, "D"))
# further runs of a mode, listed after that mode's row: (run, text, mode key)
CAVITY_EXTRA_RUNS = (("n250_crossing_r2", "crossing，full（旧缺省），第二次运行", "crossing"),)
LONG_RUNS = (("n250_crossing_normal", "250²", NEW_COLOUR, "solid"), ("n500_crossing_normal", "500²", NEW_COLOUR, (0, (5, 2))))
# the earlier cavity runs whose pressure exists only in their checkpoints: (run, text)
CHECKPOINT_RUNS = (("n250_crossing", "250² crossing，full（旧缺省）"), ("n250_crossing_r2", "250² crossing，full，第二次运行"),
                   ("n250_simple", "250² simple"), ("n250_adami", "250² adami"),
                   ("n500_crossing", "500² crossing，full（旧缺省）"), ("n500_simple", "500² simple"),
                   ("n500_adami", "500² adami"))


def write_table(name: str, text: str) -> None:
    path = REPORT_ROOT / "tables" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8", newline="\n")
    print(f"[normal_default] wrote {path.relative_to(REPOSITORY_ROOT)}")


def box_run_exists(name: str) -> bool:
    return (LOG_ROOT / "box_run" / name / "series.csv").exists()


def record_text(record: dict) -> str:
    return f"{record.get('run_id', '?')}（{record.get('commit', '?')}）" if record else "?"


# =============================================================================
# Regression checks
# =============================================================================

def manifest_rows(text: str) -> dict:
    rows = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and len(parts[0]) == 64 and parts[2].endswith(".spv"):
            rows[parts[2]] = parts[0]
    return rows


def git_show(revision: str, path: str) -> str:
    return subprocess.run(["git", "show", f"{revision}:{path}"], cwd=REPOSITORY_ROOT, capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


def regression(records_by_id: dict) -> dict:
    summary = {}
    rows = []
    # ---- SPIR-V: the files now against the base manifest and against the manifest before the change
    base = manifest_rows(git_show(BASE_COMMIT, MANIFEST))
    before = manifest_rows(git_show(COMMIT_BEFORE_CHANGE, MANIFEST))
    spv_directory = REPOSITORY_ROOT / "experiment" / "v7" / "shaders" / "spv"
    now = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(spv_directory.glob("*.spv"))}
    same_as_base = sorted(name for name in base if now.get(name) == base[name])
    changed_since_before = sorted(name for name in now if before.get(name) != now[name])
    summary["spirv"] = {"base_files": len(base), "identical_to_base": len(same_as_base), "files_now": len(now),
                        "changed_against_commit_before": changed_since_before}
    rows.append(["simple 和 adami 用的 SPIR-V 文件与基点相同",
                 f"基点（{BASE_COMMIT}）清单里的 {len(base)} 个文件，现在 {len(same_as_base)} 个逐字节相同。"
                 f"与改缺省之前（{COMMIT_BEFORE_CHANGE}）相比，变了的只有 "
                 + "、".join(f"`{name}`" for name in changed_since_before)
                 + "（crossing 的融合遍及其收支构建，各差一个常数：spec const 117 的缺省值）",
                 "MANIFEST.txt（sha256）"])
    # ---- the seam layout test: its list of failures against the base's
    tests = LOG_ROOT / "tests"
    if (tests / "failures_default_normal.txt").exists():
        def failures(name: str) -> list:
            return sorted(line.rstrip("\r\n") for line in
                          (tests / name).read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())
        base_failures, new_failures = failures("failures_base.txt"), failures("failures_default_normal.txt")
        seam_record = next((record for record in records_by_id.values()
                            if record.get("kind") == "seam_layout_test"
                            and "default normal" in str(record.get("label", ""))), {})
        summary["seam_layout"] = {"failures": len(new_failures), "failures_base": len(base_failures),
                                  "identical": base_failures == new_failures, "run_id": seam_record.get("run_id")}
        rows.append(["`_test_seam_layout.py` 的失败清单与基点相同",
                     f"{len(new_failures)} 项，基点 {len(base_failures)} 项；逐行"
                     + ("相同" if base_failures == new_failures else "**不同**"), record_text(seam_record)])
    # ---- outcome classes
    for directory_name, check_text in (
            ("explicit_full_against_old_default_s700",
             "显式设 `full` 时，crossing 与改缺省之前逐位相同（250² 方腔，K = 1，700 步）"),
            ("default_against_explicit_normal_s700",
             "不写这个键（缺省）与显式写 `normal` 逐位相同（同上；额外做的）")):
        path = LOG_ROOT / "outcome_classes" / directory_name / "classes.json"
        if not path.exists():
            continue
        classes = json.loads(path.read_text(encoding="utf-8"))
        members = "；".join(f"类 {index}（{entry['digest'][:12]}）：参照 {entry['reference_runs']} 次，本算例 "
                           f"{entry['case_runs']} 次" for index, entry in enumerate(classes["classes"]))
        summary[directory_name] = {"classes": [{key: entry[key] for key in ("digest", "reference_runs", "case_runs")}
                                               for entry in classes["classes"]],
                                   "case_within_reference_classes": classes["case_within_reference_classes"],
                                   "run_id": classes["run_id"]}
        rows.append([check_text,
                     f"{len(classes['classes'])} 个结果类。{members}。"
                     + ("本算例的每次运行都落在含参照运行的类里：逐位相同" if classes["case_within_reference_classes"]
                        else "**本算例有运行落在不含参照运行的类里**"),
                     record_text(records_by_id.get(classes["run_id"], {"run_id": classes["run_id"]}))])
    # ---- single step against the numpy reference
    for suffix, check_text in (("_wall_component_normal", "numpy 单步参考解，新缺省（`normal`）"),
                               ("_wall_component_normal_budget", "同上，打开收支构建（四项逐粒子对参考解；含 simple）"),
                               ("", "numpy 单步参考解，显式 `full`（重跑）")):
        cells = []
        identifiers = []
        entry_summary = {}
        for dimension in (2, 3):
            path = LOG_ROOT / "single_step" / f"summary_{dimension}d{suffix}.json"
            if not path.exists():
                continue
            entries = json.loads(path.read_text(encoding="utf-8"))
            passed = sum(1 for entry in entries if entry["passed"])
            worst = max((entry["reference"]["worst_relative_error"] for entry in entries
                         if entry.get("reference") and entry["configuration"] == "smooth"), default=float("nan"))
            worst_budget = max((entry["budget"]["worst_term_error_over_pair_terms_times_10"] for entry in entries
                                if entry.get("budget")), default=None)
            cells.append(f"{dimension}-D：{passed} / {len(entries)} 个配置通过，光滑场判据值最大 {worst:.1e}"
                         + ("" if worst_budget is None else f"，收支四项的判据值最大 {worst_budget:.1e}")
                         + "（容差 1e-4）")
            identifiers.append(f"{entries[0]['run_id']} … {entries[-1]['run_id'][-4:]}")
            entry_summary[f"{dimension}d"] = {"passed": passed, "configurations": len(entries), "worst": worst,
                                              "worst_budget": worst_budget,
                                              "run_ids": [entry["run_id"] for entry in entries]}
        if cells:
            summary[f"single_step{suffix}"] = entry_summary
            rows.append([check_text, "；".join(cells), "；".join(identifiers)])
    write_table("normal_default_regression.md", markdown_table(["检查", "结果", "记录"], rows))
    return summary


# =============================================================================
# Drift of the mean density, 250^2, t = 0 .. 5
# =============================================================================

def drift(records: dict) -> dict:
    measured = {}
    rows = []
    for name, text, pipeline in DRIFT_RUNS:
        if not box_run_exists(name):
            continue
        entry = delta_wall_experiment.measure(delta_wall_experiment.load_series(name), delta_wall_experiment.FIT_WINDOW)
        measured[name] = entry
        rows.append([text, pipeline, f"{entry['drift_rate']:+.5f}", f"{entry['density_mean_minus_rest']:+.5f}",
                     f"{entry['pressure_mean']:+.1f} Pa（{entry['pressure_mean_over_dynamic']:+.3f} ρ0U²）",
                     f"{100.0 * entry['pressure_negative_fraction']:.2f} %", f"{entry['kinetic_energy']:.3f}",
                     f"{entry['speed_maximum']:.3f}", str(entry["gradient_fallbacks"]),
                     f"{entry['wall_distance_minimum_over_spacing']:.3f}", record_text(records.get(name, {}))])
    new_rates = [measured[name]["drift_rate"] for name in NEW_DEFAULT_DRIFT_RUNS if name in measured]
    summary = {"runs": measured}
    text = ("250² 方腔，从静止到 t = 5.01，每 0.03 个时间单位一个样本。漂移率 = 流体平均密度对 t ∈ [1.02, 5.01] 的线性拟合"
            "斜率；其余各列取最后一个样本（t = 5.01），“最大速度”“φ_min”是整个运行的。“缺省管线”与“收支构建”是同一个算例的"
            "两种运行（后者多一遍只读的收支累加，算出的状态相同）。\n\n"
            + markdown_table(["运行", "管线", "漂移率（kg/m³ 每时间单位）", "平均密度 − ρ0（kg/m³）", "平均压力", "负压粒子比例",
                              "动能", "最大速度", "A_f 回退", "φ_min / dx", "记录（提交）"], rows))
    if new_rates:
        summary["new_default"] = {"drift_rates": new_rates, "mean": float(np.mean(new_rates)),
                                  "minimum": float(np.min(new_rates)), "maximum": float(np.max(new_rates)),
                                  "standard_deviation": float(np.std(new_rates, ddof=1)) if len(new_rates) > 1 else None}
        text += (f"\n新缺省的 {len(new_rates)} 次运行：漂移率 {min(new_rates):+.5f} 到 {max(new_rates):+.5f}，平均 "
                 f"{np.mean(new_rates):+.5f}"
                 + (f"，标准差 {np.std(new_rates, ddof=1):.5f}" if len(new_rates) > 1 else "") + "。\n")
    write_table("normal_default_drift.md", text)
    return summary


# =============================================================================
# Density budget, 250^2, t in [1, 5]
# =============================================================================

def budget(records: dict) -> dict:
    available = [(name, text) for name, text in BUDGET_COLUMNS if box_run_exists(name)]
    if not any(name in NEW_BUDGET_RUNS for name, _ in available):
        return {}
    results = {name: density_budget.analyse(name) for name, _ in available}
    row_sum, row_drift, row_diffusion = density_budget.ROW_SUM, density_budget.ROW_DRIFT, density_budget.ROW_DIFFUSION
    row_applied, row_rounding = density_budget.ROW_APPLIED, density_budget.ROW_ROUNDING
    rows = []
    for label, group, index in (
            ("连续性项，流体邻居（全场）", "all", 0), ("连续性项，壁邻居（全场）", "all", 1),
            ("δ 项，流体邻居（全场）", "all", 2), ("δ 项，壁延拓（全场）", "all", 3),
            ("**四项之和（全场）**", "all", row_sum), ("float32 存储舍入（全场）", "all", row_rounding),
            ("**实际写入的密度变化（全场）**", "all", row_applied),
            ("连续性两项之和：全场", "all", row_drift), ("连续性两项之和：盖带", "lid", row_drift),
            ("连续性两项之和：静止壁带", "static", row_drift), ("连续性两项之和：四个角", "corners", row_drift),
            ("**连续性两项之和：内部**", "interior", row_drift),
            ("连续性（流体邻居）：内部 h 到 2h", "shell_1h_2h", 0), ("连续性（流体邻居）：内部 2h 到 4h", "shell_2h_4h", 0),
            ("连续性（流体邻居）：内部 4h 到 8h", "shell_4h_8h", 0), ("连续性（流体邻居）：内部 8h 以外", "shell_beyond_8h", 0),
            ("δ 两项之和：全场", "all", row_diffusion), ("δ 两项之和：四个角", "corners", row_diffusion),
            ("δ 两项之和：内部", "interior", row_diffusion),
            ("float32 存储舍入：内部", "interior", row_rounding), ("四项之和：内部", "interior", row_sum),
            ("实际写入：内部", "interior", row_applied)):
        rows.append([label] + [density_budget.value_text(results[name]["groups"][group]["rate"][index],
                                                         results[name]["groups"][group]["standard_error"][index])
                               for name, _ in available])
    rows.append(["平均密度的线性拟合斜率（“漂移率”）"]
                + [f"{results[name]['closure']['density_mean_fit_slope']:+.5f}" for name, _ in available])
    rows.append(["闭合：实际写入与读回的平均密度，窗口内最大相差（kg/m³）"]
                + [f"{results[name]['closure']['largest_cumulative_difference']:.1e}" for name, _ in available])
    rows.append(["记录（提交）"] + [record_text(records.get(name, {})) for name, _ in available])
    write_table("normal_default_budget_compare.md",
                "250² 方腔，t ∈ [1.02, 5.01] 的时间平均，kg/m³ 每时间单位；每格 = 该区域内粒子的该项之和 ÷ 全部流体粒子数；"
                "± 是把窗口切成 8 段、各段平均值的标准误。\n\n"
                + markdown_table(["量"] + [text for _, text in available], rows))
    summary = {name: {key: value for key, value in result.items() if key != "series"}
               for name, result in results.items()}

    # ---- figure: the continuity terms by region, old against new default; and the whole-field rows
    new_names = [name for name in NEW_BUDGET_RUNS if name in results]
    old_names = [name for name in ("budget_n250_crossing", "budget_n250_crossing_repeat") if name in results]
    if new_names and old_names:
        setup_style()
        figure, axes = plt.subplots(1, 2, figsize=(13.6, 4.9))
        width = 0.36

        def mean_and_spread(names: list, group: str, index: int) -> tuple:
            values = [results[name]["groups"][group]["rate"][index] for name in names]
            errors = [results[name]["groups"][group]["standard_error"][index] for name in names]
            # the two runs' mean; the bar's whisker = the larger of their half difference and their standard error
            spread = max(0.5 * (max(values) - min(values)), max(errors))
            return float(np.mean(values)), float(spread)

        panels = (
            (axes[0], "连续性两项之和，按区域", (("盖带", "lid", row_drift), ("静止壁带", "static", row_drift),
                                       ("四个角", "corners", row_drift), ("内部", "interior", row_drift))),
            (axes[1], "全场的各项", (("连续性两项", "all", row_drift), ("δ 两项", "all", row_diffusion),
                                ("四项之和", "all", row_sum), ("float32 舍入", "all", row_rounding),
                                ("实际写入", "all", row_applied))))
        for axis, title, entries in panels:
            positions = np.arange(len(entries))
            for offset, (names, colour, text) in enumerate(((old_names, OLD_COLOUR, OLD_TEXT),
                                                           (new_names, NEW_COLOUR, NEW_TEXT))):
                statistics = [mean_and_spread(names, group, index) for _, group, index in entries]
                values = [value for value, _ in statistics]
                bars = axis.bar(positions + (offset - 0.5) * (width + 0.03), values, width=width, color=colour,
                                zorder=3, label=text, yerr=[spread for _, spread in statistics],
                                error_kw={"ecolor": INK_PRIMARY, "elinewidth": 0.9, "capsize": 2})
                # the value beyond the whisker's end, so that the two do not overlap
                for bar, (value, spread) in zip(bars, statistics):
                    anchor = value + spread if value >= 0 else value - spread
                    axis.annotate(f"{value:+.4f}", (bar.get_x() + bar.get_width() / 2, anchor),
                                  xytext=(0, 4 if value >= 0 else -4), textcoords="offset points", ha="center",
                                  va="bottom" if value >= 0 else "top", fontsize=8, color=INK_SECONDARY)
            axis.axhline(0.0, color=INK_SECONDARY, linewidth=0.8, zorder=2)
            axis.set_xticks(positions)
            axis.set_xticklabels([text for text, _, _ in entries])
            axis.set_title(f"{title}（kg/m³ 每时间单位）")
            axis.grid(axis="x", visible=False)
            low, high = axis.get_ylim()
            axis.set_ylim(low - 0.12 * (high - low), high + 0.12 * (high - low))
        handles, labels = axes[0].get_legend_handles_labels()
        height = figure.get_figheight()
        figure.suptitle("250² 方腔 t ∈ [1, 5]：平均密度变化率的收支，旧缺省与新缺省（各两次运行的平均）", x=0.01,
                        y=1.0 - 0.05 / height, ha="left", va="top", fontsize=12, fontweight="bold", color=INK_PRIMARY)
        figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0 - 0.40 / height), ncol=2,
                      handlelength=1.4, columnspacing=1.6, labelcolor=INK_PRIMARY, borderaxespad=0.0)
        figure.tight_layout(rect=(0, 0, 1.0, 1.0 - 0.66 / height), w_pad=3.0)
        (REPORT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
        figure.savefig(REPORT_ROOT / "figures" / "normal_default_budget.png")
        plt.close(figure)
    return summary


def delta_density(records: dict) -> dict:
    """The run with the density stored as rho - rho_ref (V7_DELTA_DENSITY=1) beside the two runs of the default
    storage: does the rounding term go, and what is the drift then?"""
    if not box_run_exists(DELTA_DENSITY_RUN):
        return {}
    columns = [(name, text) for name, text in (("budget_n250_crossing_normal_r1", "float32 存 ρ（缺省），第 1 次"),
                                               ("budget_n250_crossing_normal_r2", "float32 存 ρ（缺省），第 2 次"),
                                               (DELTA_DENSITY_RUN, "**float32 存 ρ − ρ_ref（`V7_DELTA_DENSITY=1`）**"))
               if box_run_exists(name)]
    results = {name: density_budget.analyse(name) for name, _ in columns}
    measured = {name: delta_wall_experiment.measure(delta_wall_experiment.load_series(name),
                                                    delta_wall_experiment.FIT_WINDOW) for name, _ in columns}
    rows = []
    for label, group, index in (("连续性两项之和（全场）", "all", density_budget.ROW_DRIFT),
                                ("δ 两项之和（全场）", "all", density_budget.ROW_DIFFUSION),
                                ("**四项之和（全场）**", "all", density_budget.ROW_SUM),
                                ("**存储舍入（实际写入 − 四项之和，全场）**", "all", density_budget.ROW_ROUNDING),
                                ("**实际写入的密度变化（全场）**", "all", density_budget.ROW_APPLIED),
                                ("连续性两项之和：内部", "interior", density_budget.ROW_DRIFT),
                                ("存储舍入：内部", "interior", density_budget.ROW_ROUNDING)):
        rows.append([label] + [density_budget.value_text(results[name]["groups"][group]["rate"][index],
                                                         results[name]["groups"][group]["standard_error"][index])
                               for name, _ in columns])
    rows.append(["**漂移率（平均密度的线性拟合斜率）**"] + [f"{measured[name]['drift_rate']:+.5f}" for name, _ in columns])
    rows.append(["t = 5 的平均密度 − ρ0（kg/m³）"] + [f"{measured[name]['density_mean_minus_rest']:+.5f}" for name, _ in columns])
    rows.append(["t = 5 的平均压力（Pa）"] + [f"{measured[name]['pressure_mean']:+.1f}" for name, _ in columns])
    rows.append(["t = 5 的负压粒子比例"] + [f"{100.0 * measured[name]['pressure_negative_fraction']:.2f} %"
                                      for name, _ in columns])
    rows.append(["t = 5 的动能"] + [f"{measured[name]['kinetic_energy']:.3f}" for name, _ in columns])
    rows.append(["闭合：实际写入与读回的平均密度，窗口内最大相差（kg/m³）"]
                + [f"{results[name]['closure']['largest_cumulative_difference']:.1e}" for name, _ in columns])
    rows.append(["记录（提交）"] + [record_text(records.get(name, {})) for name, _ in columns])
    write_table("normal_default_delta_density.md",
                "250² 方腔，新缺省，t ∈ [1.02, 5.01]，kg/m³ 每时间单位；± 是 8 段的标准误。\n\n"
                + markdown_table(["量"] + [text for _, text in columns], rows))
    return {"budget": {name: {key: value for key, value in result.items() if key != "series"}
                       for name, result in results.items()}, "measured": measured}


# =============================================================================
# Hydrostatic boxes
# =============================================================================

def hydrostatic(records_by_id: dict) -> dict:
    runs = {}
    for direction, (direction_text, entries) in HYDROSTATIC_RUNS.items():
        present = tuple(entry for entry in entries
                        if (LOG_ROOT / "hydrostatic" / entry[0] / "summary.json").exists())
        if not any(form == 1 for _, _, form in present):
            return {}
        runs[direction] = (direction_text, present)
    introduction = (
        "2-D 静水盒（250² 方腔的几何，盖静止，|g| = 10，c0 = 100，从均匀密度、零速度开始，到 t = 10.02）。速度以 "
        "√(|g| H) 为单位，压力以 ρ0 |g| H 为单位（H = 盒沿重力方向的尺寸）。“去电平” = 减去全场平均偏差后的误差；"
        "“电平” = 压力减去 ρ0 g·(r − r_中心) 之后对全部流体粒子的平均。两种设置各有两次运行：两次之间的差别就是单次"
        "运行的可重复程度。`full` 的两次是之前的运行（改缺省之前），其中第一次没有记电平的历史。")
    return delta_wall_experiment.hydrostatic_comparison(
        records_by_id, runs_by_direction=runs, forms=HYDROSTATIC_FORMS, introduction=introduction,
        table_name="normal_default_hydrostatic.md", figure_name="normal_default_hydrostatic.png",
        figure_title="2-D 静水盒：连续性壁项 full（旧缺省）与 normal（新缺省）（压力以 ρ0 g H 为单位）")


# =============================================================================
# Re = 1000 cavity against Marchi 2021
# =============================================================================

def cavity(modes: tuple = CAVITY_MODES, extra_runs: tuple = CAVITY_EXTRA_RUNS, prefix: str = "normal_default",
           required_mode: str = "crossing_normal", labelled_modes: tuple = ("crossing_normal", "crossing"),
           figure_title: str = "Re = 1000 方腔：四个极值相对 Marchi 2021 的偏差（ψ_min 用 wall 帧）") -> dict:
    """The metric tables (<prefix>_cavity.md, _cavity_positions.md, _cavity_order.md) and the figure
    (<prefix>_cavity.png) of the cavity runs of modes ((mode key of cavity_tables.py, text, colour, marker), ...),
    from claude/wall_crossing/cavity/tables.json. Nothing is written unless a run of required_mode exists.
    labelled_modes: the modes whose values are written beside their markers in the figure."""
    path = REPORT_ROOT / "cavity" / "tables.json"
    if not path.exists():
        return {}
    tables = json.loads(path.read_text(encoding="utf-8"))
    runs, convergence = tables["runs"], tables.get("convergence", {})
    if not any(name.endswith(f"_{required_mode}") for name in runs):
        return {}
    ordered = []
    for resolution in (250, 500):
        for mode, text, _, _ in modes:
            name = f"n{resolution}_{mode}"
            if name in runs and not runs[name]["failed"]:
                ordered.append((name, f"{resolution}² {text}" + ("" if runs[name]["complete"] else "（运行中）")))
            for extra, extra_text, extra_mode in extra_runs:
                if extra_mode == mode and extra.startswith(f"n{resolution}_") and extra in runs \
                        and not runs[extra]["failed"]:
                    ordered.append((extra, f"{resolution}² {extra_text}"))

    def percent(value: float) -> str:
        return f"{value:+.2f} %"

    rows = []
    for name, text in ordered:
        record = runs[name]
        extrema, errors = record["extrema"], record["errors"]
        psi = extrema.get("psi_min")
        rows.append([text,
                     "—" if not psi else f"{psi['value_wall']:.5f}（{percent(psi['deviation_wall_percent'])}）",
                     "—" if not psi else percent(psi["deviation_mid_percent"]),
                     *[f"{extrema[key]['value']:.5f}（{percent(extrema[key]['deviation_percent'])}）"
                       for key in ("u_min", "v_max", "v_min")],
                     f"{errors['wall_u']['L2']:.4f} / {errors['wall_u']['Linf']:.4f}",
                     f"{errors['wall_v']['L2']:.4f} / {errors['wall_v']['Linf']:.4f}",
                     f"{errors['mid_u']['L2']:.4f} / {errors['mid_u']['Linf']:.4f}",
                     f"{errors['mid_v']['L2']:.4f} / {errors['mid_v']['Linf']:.4f}"])
    reference = None
    try:
        from experiment.validation import cavity_reference
        reference = cavity_reference.marchi2021()["extrema"]
    except Exception:           # the table stands without the reference line
        pass
    header = ("Re = 1000 方腔对 Marchi 2021，K = 1，最后 20 个时间单位平均。括号内 = 相对参考值的偏差（负 = 偏弱）。"
              "“wall 帧”把单位方腔放在最内层壁粒子行的中线上（已发表的表用的放法，各模式同样处理）；“mid 帧”放在最内层壁"
              "粒子行与最外层流体行的正中（crossing 的盒面）。剖面误差是 28 个 Marchi 点上的 L2 / L∞，速度以盖速为单位。")
    if reference:
        header += (f"参考值：ψ_min {reference['psi_min'][0]:.5f}，u_min {reference['u_min'][0]:.5f}，v_max "
                   f"{reference['v_max'][0]:.5f}，v_min {reference['v_min'][0]:.5f}。")
    text = header + "\n\n" + markdown_table(
        ["运行", "ψ_min（wall 帧）", "ψ_min 偏差（mid 帧）", "u_min", "v_max", "v_min", "u 剖面 L2 / L∞（wall 帧）",
         "v 剖面 L2 / L∞（wall 帧）", "u 剖面 L2 / L∞（mid 帧）", "v 剖面 L2 / L∞（mid 帧）"], rows)
    write_table(f"{prefix}_cavity.md", text)

    rows = []
    for name, text in ordered:
        record = runs[name]
        extrema = record["extrema"]
        psi = extrema.get("psi_min")
        rows.append([text]
                    + [f"{extrema[key]['position_wall']:.4f} / {extrema[key]['position_mid']:.4f}"
                       f"（{extrema[key]['position_mid'] - extrema[key]['position_reference']:+.4f}）"
                       for key in ("u_min", "v_max", "v_min")]
                    + (["—"] if not psi else
                       [f"({psi['x_wall']:.4f}, {psi['y_wall']:.4f}) / ({psi['x_mid']:.4f}, {psi['y_mid']:.4f})"
                        f"（{psi['x_mid'] - psi['x_reference']:+.4f}, {psi['y_mid'] - psi['y_reference']:+.4f}）"])
                    + [f"{record['window'][0]:.1f}–{record['window'][1]:.1f}",
                       "—" if record["t_steady_online"] is None else f"{record['t_steady_online']:.1f}",
                       f"{record['kinetic_energy']:.4f}", (record.get("git") or {}).get("head", "?")])
    position_header = "极值的位置：wall 帧 / mid 帧（括号内 = mid 帧位置减参考位置）。"
    if reference:
        position_header += (f"参考位置：y(u_min) {reference['y_at_u_min'][0]:.4f}，x(v_max) {reference['x_at_v_max'][0]:.4f}，"
                            f"x(v_min) {reference['x_at_v_min'][0]:.4f}，ψ_min ({reference['x_at_psi_min'][0]:.4f}, "
                            f"{reference['y_at_psi_min'][0]:.4f})。")
    write_table(f"{prefix}_cavity_positions.md", position_header + "\n\n" + markdown_table(
        ["运行", "y(u_min)", "x(v_max)", "x(v_min)", "ψ_min 的位置", "平均窗口 t", "定常 t_s", "窗口平均动能", "提交"], rows))

    # ---- observed order 250^2 -> 500^2, every mode
    quantities = ("ψ_min 偏差（绝对值，wall 帧）", "ψ_min 偏差（绝对值，mid 帧）", "u_min 偏差（绝对值）", "v_max 偏差（绝对值）",
                  "v_min 偏差（绝对值）", "u L2 (wall)", "u Linf (wall)", "v L2 (wall)", "v Linf (wall)", "u L2 (mid)",
                  "u Linf (mid)", "v L2 (mid)", "v Linf (mid)")
    rows = []
    order_summary = {}
    for quantity in quantities:
        cells = [quantity.replace("Linf", "L∞")]
        for mode, _, _, _ in modes:
            entry = convergence.get(mode, {}).get("quantities", {}).get(quantity)
            if not entry or len(entry["values"]) < 2:
                cells.append("—")
                continue
            values, pairwise = entry["values"], entry["pairwise"]
            cells.append(f"{values[0]:.5f} → {values[1]:.5f}，p = {pairwise[0]:.2f}")
            order_summary.setdefault(mode, {})[quantity] = {"values": values[:2], "order": pairwise[0]}
        rows.append(cells)
    write_table(f"{prefix}_cavity_order.md",
                "观测收敛阶 p = ln(e₂₅₀ / e₅₀₀) / ln 2，e = 该量相对 Marchi 2021 的误差（绝对值），250² → 500²。只有两档"
                "分辨率，所以每个 p 只是一对数的比值；误差本身已经很小的量（接近运行间的散布）p 没有意义。\n\n"
                + markdown_table(["量"] + [text for _, text, _, _ in modes], rows))

    # ---- figure: the deviations of the four extrema at the two resolutions, one panel per quantity
    setup_style()
    figure, axes = plt.subplots(1, 4, figsize=(13.6, 4.3), sharey=True)
    handles = {}
    for axis, (key, title) in zip(axes, (("psi_min", "ψ_min"), ("u_min", "u_min"), ("v_max", "v_max"), ("v_min", "v_min"))):
        labelled = {}       # resolution -> [(value, mode)] of the two crossing settings, whose values are written
        for mode, text, colour, marker in modes:
            points = []
            for resolution in (250, 500):
                record = runs.get(f"n{resolution}_{mode}")
                if not record or record["failed"] or not record["complete"]:
                    continue
                entry = record["extrema"].get(key)
                if not entry:
                    continue
                points.append((resolution, entry["deviation_wall_percent"] if key == "psi_min"
                               else entry["deviation_percent"]))
            if not points:
                continue
            line, = axis.plot([point[0] for point in points], [point[1] for point in points], color=colour,
                              marker=marker, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
            handles[text] = line
            if mode in labelled_modes:
                for resolution, value in points:
                    labelled.setdefault(resolution, []).append(value)
        # the values of the two crossing settings: the upper one above its marker, the lower one below
        for resolution, values in labelled.items():
            for value in values:
                above = value == max(values) and len(values) > 1
                axis.annotate(f"{value:+.2f}", (resolution, value), xytext=(0, 9 if above else -9),
                              textcoords="offset points", ha="center", va="bottom" if above else "top",
                              fontsize=8, color=INK_SECONDARY, zorder=4,
                              bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.8})
        axis.axhline(0.0, color=INK_SECONDARY, linewidth=0.8, zorder=2)
        axis.set_xscale("log")
        axis.set_xticks([250, 500])
        axis.set_xticklabels(["250²", "500²"])
        axis.minorticks_off()
        axis.set_xlim(210, 600)
        axis.set_title(f"{title}：相对 Marchi 2021 的偏差（%）")
    height = figure.get_figheight()
    figure.suptitle(figure_title, x=0.01, y=1.0 - 0.05 / height,
                    ha="left", va="top", fontsize=12, fontweight="bold", color=INK_PRIMARY)
    figure.legend(list(handles.values()), list(handles.keys()), loc="upper left",
                  bbox_to_anchor=(0.01, 1.0 - 0.40 / height), ncol=len(handles), handlelength=2.2, columnspacing=1.6,
                  labelcolor=INK_PRIMARY, borderaxespad=0.0)
    figure.tight_layout(rect=(0, 0, 1.0, 1.0 - 0.66 / height), w_pad=1.6)
    figure.savefig(REPORT_ROOT / "figures" / f"{prefix}_cavity.png")
    plt.close(figure)
    return {"runs": {name: {"extrema": runs[name]["extrema"], "errors": runs[name]["errors"],
                            "window": runs[name]["window"], "t_steady_online": runs[name]["t_steady_online"],
                            "complete": runs[name]["complete"]} for name, _ in ordered},
            "order": order_summary}


# =============================================================================
# The long runs: histories of the fluid's mean density and pressure
# =============================================================================

def load_field_history(name: str) -> dict:
    """field_history.jsonl of a cavity run as arrays, one row per step (a step with a sample and a snapshot has two
    lines), in time order."""
    path = LOG_ROOT / "cavity" / name / "field_history.jsonl"
    if not path.exists():
        return {}
    by_step = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("step") is not None:
                by_step[int(record["step"])] = record
    steps = sorted(by_step)
    return {key: np.array([by_step[step][key] for step in steps], dtype=float)
            for key in ("step", "time", "alive", "fluid_particles", "density_mean", "density_minimum",
                        "density_maximum", "pressure_mean", "pressure_minimum", "pressure_maximum",
                        "pressure_negative_fraction", "speed_maximum")}


def checkpoint_pressure(name: str) -> dict:
    """The fluid's mean density and pressure in the last checkpoint of an earlier cavity run (its restart state:
    the stored density plus the storage offset, and the pressure)."""
    directory = LOG_ROOT / "cavity" / name
    manifest = directory / "checkpoints" / "manifest.jsonl"
    if not manifest.exists():
        return {}
    entries = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    entries = [entry for entry in entries if (directory / "checkpoints" / entry["file"]).exists()]
    if not entries:
        return {}
    entry = entries[-1]
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    data = np.load(directory / "checkpoints" / entry["file"])
    fluid = np.isin(data["material"], meta["fluid_groups"])
    density = data["density_pressure"][fluid, 0].astype(np.float64) + float(data["_stored_density_offset"])
    pressure = data["density_pressure"][fluid, 1].astype(np.float64)
    speed = np.sqrt((data["velocity_mass"][fluid, 0:3].astype(np.float64) ** 2).sum(axis=1))
    return {"time": float(entry["time"]), "step": int(entry["step"]), "fluid_particles": int(fluid.sum()),
            "density_mean_minus_rest": float(density.mean() - REST_DENSITY), "pressure_mean": float(pressure.mean()),
            "pressure_negative_fraction": float((pressure < 0.0).mean()), "speed_maximum": float(speed.max()),
            "pressure_minimum": float(pressure.min()), "pressure_maximum": float(pressure.max())}


def long_runs(records_by_name: dict) -> dict:
    histories = {name: load_field_history(name) for name, _, _, _ in LONG_RUNS}
    histories = {name: history for name, history in histories.items() if history}
    if not histories:
        return {}
    summary = {"runs": {}, "checkpoints": {}}
    instants = (1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 90.7, 99.4, 100.0)
    rows = []
    for name, text, _, _ in LONG_RUNS:
        history = histories.get(name)
        if not history:
            continue
        time = history["time"]
        expected = history["alive"][0]
        late = time >= 20.0
        entry = {
            "samples": int(len(time)), "t_last": float(time[-1]),
            "alive_constant": bool((history["alive"] == expected).all()), "alive": int(expected),
            "speed_maximum": float(history["speed_maximum"].max()),
            "density_range": [float(history["density_minimum"].min()), float(history["density_maximum"].max())],
            "pressure_mean_last": float(history["pressure_mean"][-1]),
            "pressure_negative_fraction_maximum_after_5": float(history["pressure_negative_fraction"][time >= 5.0].max()),
            "pressure_negative_fraction_last": float(history["pressure_negative_fraction"][-1]),
            "density_mean_minus_rest_last": float(history["density_mean"][-1] - REST_DENSITY),
            "density_mean_rate_20_to_end": float(np.polyfit(time[late], history["density_mean"][late], 1)[0])
            if late.sum() >= 3 else None,
            "density_mean_rate_1_to_5": float(np.polyfit(time[(time >= 1.0) & (time <= 5.011)],
                                                         history["density_mean"][(time >= 1.0) & (time <= 5.011)], 1)[0]),
            "at_times": {}}
        last_fifth = time >= time[-1] - 20.0
        entry["density_mean_rate_last_20"] = float(np.polyfit(time[last_fifth], history["density_mean"][last_fifth], 1)[0])
        entry["pressure_mean_rate_last_20"] = float(np.polyfit(time[last_fifth], history["pressure_mean"][last_fifth], 1)[0])
        for instant in instants:
            if instant > time[-1] + 0.3:
                continue
            index = int(np.argmin(np.abs(time - instant)))
            entry["at_times"][f"{instant:g}"] = {
                "time": float(time[index]), "density_mean_minus_rest": float(history["density_mean"][index] - REST_DENSITY),
                "pressure_mean": float(history["pressure_mean"][index]),
                "pressure_negative_fraction": float(history["pressure_negative_fraction"][index]),
                "speed_maximum": float(history["speed_maximum"][index]), "alive": int(history["alive"][index])}
            rows.append([f"{text}", f"{time[index]:.2f}", f"{history['density_mean'][index] - REST_DENSITY:+.5f}",
                         f"{history['pressure_mean'][index]:+.1f} Pa（{history['pressure_mean'][index] / DYNAMIC_PRESSURE:+.3f} ρ0U²）",
                         f"{history['pressure_minimum'][index]:+.0f} … {history['pressure_maximum'][index]:+.0f}",
                         f"{100.0 * history['pressure_negative_fraction'][index]:.3f} %",
                         f"{history['speed_maximum'][index]:.3f}", f"{int(history['alive'][index]):,}"])
        summary["runs"][name] = entry
    write_table("normal_default_long_runs.md",
                "新缺省的两个方腔统计运行，各时刻的样本（每 0.2 个时间单位一个样本，取最近的一个）：对存活的流体粒子（不含壁）"
                "的 float64 平均。\n\n"
                + markdown_table(["运行", "t", "平均密度 − ρ0（kg/m³）", "平均压力", "压力范围（Pa）", "负压粒子比例",
                                  "流体最大速度", "粒子数（含壁）"], rows))

    # what the runner and the crossing status file say about the walls and the invariants (cavity_tables.py)
    tables_path = REPORT_ROOT / "cavity" / "tables.json"
    analysed = json.loads(tables_path.read_text(encoding="utf-8"))["runs"] if tables_path.exists() else {}
    rows = []
    for name, text, _, _ in LONG_RUNS:
        entry = summary["runs"].get(name)
        if not entry:
            continue
        record = records_by_name.get(name, {})
        crossing = (analysed.get(name, {}).get("diagnostics") or {}).get("crossing") or {}
        result_path = LOG_ROOT / "cavity" / name / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        entry["crossing_status"] = crossing
        entry["result"] = {key: result.get(key) for key in ("status", "segments_with_invariant_violation",
                                                           "steady", "stop_step")} if result else None
        rows.append([text, f"{entry['samples']}，到 t = {entry['t_last']:.2f}",
                     f"{entry['alive']:,}，" + ("每个样本都相同" if entry["alive_constant"] else "**有变化**"),
                     f"{entry['speed_maximum']:.3f}",
                     f"{entry['density_range'][0] - REST_DENSITY:+.3f} … {entry['density_range'][1] - REST_DENSITY:+.3f}",
                     f"{entry['density_mean_rate_1_to_5']:+.5f}",
                     "—" if entry["density_mean_rate_20_to_end"] is None else f"{entry['density_mean_rate_20_to_end']:+.6f}",
                     f"{entry['density_mean_rate_last_20']:+.6f}",
                     f"{entry['pressure_mean_rate_last_20']:+.3f}",
                     f"{100.0 * entry['pressure_negative_fraction_maximum_after_5']:.3f} %",
                     "—" if not crossing else str(crossing["gradient_fallbacks"]),
                     "—" if not crossing or crossing.get("wall_distance_minimum_over_spacing") is None
                     else f"{crossing['wall_distance_minimum_over_spacing']:.3f}",
                     "—" if not crossing or crossing.get("below_floor_maximum") is None
                     else f"{crossing['below_floor_maximum']} / {crossing['outside_box_maximum']}",
                     "运行中" if not result else
                     (f"{result['status']}；不变量违反的段：{len(result['segments_with_invariant_violation'])}；"
                      f"定常 t_s = {(result.get('steady') or {}).get('steady_time', float('nan')):.2f}"),
                     record_text(record)])
    write_table("normal_default_long_runs_summary.md", markdown_table(
        ["运行", "样本数", "粒子数（含壁）", "流体最大速度（整个运行）", "密度范围 − ρ0（整个运行，kg/m³）",
         "平均密度变化率，t ∈ [1, 5]（kg/m³ 每时间单位）", "同，t ∈ [20, 末]", "同，最后 20 个时间单位",
         "平均压力变化率，最后 20 个时间单位（Pa 每时间单位）", "负压粒子比例的最大值（t ≥ 5）", "A_f 回退（累计）",
         "φ_min / dx", "低于下限 / 越出盒（最大粒子数）", "运行器的结论", "记录（提交）"], rows))

    # ---- the earlier runs: what their last checkpoints hold, beside the new runs at the same times
    rows = []
    for name, text in CHECKPOINT_RUNS:
        state = checkpoint_pressure(name)
        if not state:
            continue
        summary["checkpoints"][name] = state
        rows.append([text, f"{state['time']:.2f}", f"{state['density_mean_minus_rest']:+.5f}",
                     f"{state['pressure_mean']:+.1f} Pa（{state['pressure_mean'] / DYNAMIC_PRESSURE:+.3f} ρ0U²）",
                     f"{state['pressure_minimum']:+.0f} … {state['pressure_maximum']:+.0f}",
                     f"{100.0 * state['pressure_negative_fraction']:.2f} %", f"{state['speed_maximum']:.3f}",
                     "最后一个检查点"])
    for name, text, _, _ in LONG_RUNS:
        history = histories.get(name)
        if not history:
            continue
        for instant in ((90.72,) if name.startswith("n250") else (99.45,)):
            if instant > history["time"][-1] + 0.3:
                continue
            index = int(np.argmin(np.abs(history["time"] - instant)))
            rows.append([f"**{text} crossing，normal（新缺省）**", f"{history['time'][index]:.2f}",
                         f"{history['density_mean'][index] - REST_DENSITY:+.5f}",
                         f"{history['pressure_mean'][index]:+.1f} Pa（{history['pressure_mean'][index] / DYNAMIC_PRESSURE:+.3f} ρ0U²）",
                         f"{history['pressure_minimum'][index]:+.0f} … {history['pressure_maximum'][index]:+.0f}",
                         f"{100.0 * history['pressure_negative_fraction'][index]:.2f} %",
                         f"{history['speed_maximum'][index]:.3f}", "样本历史"])
    if rows:
        write_table("normal_default_pressure_level.md",
                    "之前的运行没有记压力的历史，压力只留在检查点里（每 10 个时间单位一个，只保留最后两个）：250² 的最后一个"
                    "检查点在 t = 90.72，500² 的在 t = 99.45。新缺省的运行取同一时刻附近的样本。\n\n"
                    + markdown_table(["运行", "t", "平均密度 − ρ0（kg/m³）", "平均压力", "压力范围（Pa）", "负压粒子比例",
                                      "流体最大速度", "来源"], rows))

    # ---- figure: four quantities against time, one axis each, the two resolutions
    setup_style()
    figure, axes = plt.subplots(1, 4, figsize=(13.6, 4.2))
    panels = (("density_mean", lambda values: values - REST_DENSITY, "流体平均密度 − ρ0（kg/m³）", "{:+.4f}"),
              ("pressure_mean", lambda values: values / DYNAMIC_PRESSURE, "流体平均压力 / (ρ0 U²)", "{:+.3f}"),
              ("pressure_negative_fraction", lambda values: 100.0 * values, "负压粒子比例（%）", "{:.3f}"),
              ("speed_maximum", lambda values: values, "流体最大速度（盖速 = 1）", "{:.3f}"))
    handles, labels = [], []
    for axis, (key, transform, title, value_format) in zip(axes, panels):
        ends = []
        for name, text, colour, style in LONG_RUNS:
            history = histories.get(name)
            if not history:
                continue
            values = transform(history[key])
            line, = axis.plot(history["time"], values, color=colour, linestyle=style, zorder=3)
            ends.append((float(values[-1]), value_format.format(values[-1]), float(history["time"][-1])))
            if axis is axes[0]:
                handles.append(line)
                labels.append(f"{text}，新缺省")
        low, high = axis.get_ylim()
        gap = 0.07 * (high - low)
        placed = []
        for value, text_value, instant in sorted(ends):
            position = value if not placed or value - placed[-1] >= gap else placed[-1] + gap
            placed.append(position)
            axis.annotate(text_value, (instant, value), xytext=(instant + 2.5, position), textcoords="data",
                          fontsize=8, color=INK_SECONDARY, va="center", annotation_clip=False)
        if key in ("density_mean", "pressure_mean"):
            axis.axhline(0.0, color=INK_SECONDARY, linewidth=0.7, zorder=2)
        axis.set_xlim(0, max(float(history["time"][-1]) for history in histories.values()))
        axis.set_title(title)
        axis.set_xlabel("时间（模拟单位）")
    height = figure.get_figheight()
    figure.suptitle("Re = 1000 方腔，新缺省（连续性壁项 normal）：统计运行全程的密度与压力", x=0.01,
                    y=1.0 - 0.05 / height, ha="left", va="top", fontsize=12, fontweight="bold", color=INK_PRIMARY)
    figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0 - 0.40 / height), ncol=len(handles),
                  handlelength=3.0, columnspacing=1.6, labelcolor=INK_PRIMARY, borderaxespad=0.0)
    figure.tight_layout(rect=(0, 0, 0.975, 1.0 - 0.66 / height), w_pad=2.6)
    figure.savefig(REPORT_ROOT / "figures" / "normal_default_long_runs.png")
    plt.close(figure)
    return summary


def main() -> int:
    records_by_id, box_records, cavity_records = {}, {}, {}
    for record in run_records():
        records_by_id[record["run_id"]] = record
        if record.get("kind") == "box_run" and record.get("name"):
            box_records[record["name"]] = record
        if record.get("kind") == "cavity" and record.get("name"):
            cavity_records[record["name"]] = record
    (REPORT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
    summary = {
        "regression": regression(records_by_id),
        "drift": drift(box_records),
        "budget": budget(box_records),
        "delta_density": delta_density(box_records),
        "hydrostatic": hydrostatic(records_by_id),
        "cavity": cavity(),
        "long_runs": long_runs(cavity_records),
    }
    out = REPORT_ROOT / "normal_default"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    print(f"[normal_default] wrote {(out / 'summary.json').relative_to(REPOSITORY_ROOT)}; parts with data: "
          + ", ".join(key for key, value in summary.items() if value))
    return 0


if __name__ == "__main__":
    sys.exit(main())
