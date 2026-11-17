"""
report_tables.py - the tables of claude/wall_crossing_report.md that are not written by a run tool itself, and
the report's assembly.

    .venv/Scripts/python.exe -m experiment.wall_crossing.report_tables            # tables + assemble the report
    .venv/Scripts/python.exe -m experiment.wall_crossing.report_tables --no-build  # tables only

Tables (claude/wall_crossing/tables/*.md), every row with the runs.jsonl record(s) it comes from:
  single_step_reference.md  GPU against the numpy reference (logs/wall_crossing/single_step/summary_<d>d.json)
  single_step_physics.md    the one-step Couette and hydrostatic residuals of every wall mode (same files)
  registers.md              the driver's register counts (logs/wall_crossing/statistics/*.json)
  timing_<set>.md           per-kernel times (logs/wall_crossing/timing/<set>/summary.json)
  bitwise.md                base against branch, bit for bit (the bitwise_compare records of runs.jsonl)
  box_runs_diagnostics.md   the box_run runs of the density diagnostics at 250^2
  box_runs_cavity3d.md      the box_run runs of the 3-D cavity; cavity3d_modes.md / cavity3d_differences.md: their
                            centre-line profiles at the latest common time, per mode and pair by pair
  seam_all.md               every seam audit (logs/wall_crossing/seam_audit/*/summary.json) in one table
  bitwise_classes.md, seam_*.md, k2_*.md   copies of summaries the run tools wrote under logs/
  (and claude/wall_crossing/runs.jsonl, runs_<machine>.jsonl: copies of the run records the report cites)
  hydrostatic_other.md      hydrostatic runs that a later run of the same name replaced
  throughput.md             K = 2 throughput (logs/wall_crossing/throughput/*.json), when measured

Assembly: claude/wall_crossing/report_source.md is the hand-written text; a line `<!-- include: FILE -->` (FILE
relative to claude/wall_crossing/) is replaced by that file's content (its first-level title line dropped), and
the result is written to claude/wall_crossing_report.md. A missing file leaves a visible "（未生成：FILE）" line.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.run_log import LOG_ROOT     # noqa: E402

REPORT_ROOT = REPOSITORY_ROOT / "claude" / "wall_crossing"
MODE_TEXT = {("simple", None): "simple", ("adami", None): "adami", ("crossing", "extrapolate"): "crossing（extrapolate）",
             ("crossing", "wall_velocity"): "crossing（wall_velocity）"}


def run_records() -> list:
    """Every record of this machine's runs.jsonl and of the copies fetched from the second machine."""
    records = []
    for path in [LOG_ROOT / "runs.jsonl", *sorted((LOG_ROOT / "remote").glob("*.jsonl"))]:
        if path.exists():
            records += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return records


def markdown_table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def write(name: str, text: str) -> None:
    path = REPORT_ROOT / "tables" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"[report_tables] wrote {path.relative_to(REPOSITORY_ROOT)}")


def scientific(value: float) -> str:
    return "0" if value == 0 else f"{value:.1e}"


# =============================================================================
# One-step comparison
# =============================================================================

def single_step_tables() -> None:
    entries = []
    for dimension in (2, 3):
        path = LOG_ROOT / "single_step" / f"summary_{dimension}d.json"
        if path.exists():
            entries += json.loads(path.read_text(encoding="utf-8"))
    if not entries:
        return
    configuration_text = {"smooth": "光滑场", "couette": "Couette", "hydrostatic_positive": "静水（p > 0）",
                          "hydrostatic_negative": "静水（p < 0）"}
    rows = []
    for entry in entries:
        reference = entry.get("reference")
        if not reference:
            continue
        comparisons, counts = reference["comparisons"], reference["counts"]
        rate = comparisons["density_rate"]
        rows.append([
            f"{entry['dimension']}-D {configuration_text[entry['configuration']]}，{entry['continuity']}",
            f"{counts['fluid']:,} / {counts['wall']:,}（近壁 {counts['fluid_with_wall_neighbours']:,}，"
            f"低于下限 {counts['fluid_below_distance_floor']}）",
            f"{entry['status']['wall_crossing_gradient_fallback_count']} = {counts['fluid_inverse_fallbacks']}；"
            f"截断 {counts['fluid_inverse_frobenius_capped']}",
            f"{rate['new_density_error_ulp_maximum']:.2f}",
            scientific(rate["relative_error_beyond_one_ulp_maximum"]),
            f"{scientific(comparisons['acceleration']['relative_error_maximum'])} / "
            f"{scientific(comparisons['acceleration']['error_over_pair_term_sum_maximum'])}",
            f"{scientific(comparisons['shift']['relative_error_maximum'])} / "
            f"{scientific(comparisons['shift']['error_over_pair_term_sum_maximum'])}",
            scientific(comparisons["correction_inverse"]["relative_error_maximum"]),
            scientific(comparisons["fluid_inverse"]["relative_error_maximum"]),
            scientific(comparisons["kernel_sum"]["relative_error_maximum"]),
            scientific(comparisons["fluid_density_gradient_sum"]["relative_error_maximum"]),
            f"{scientific(reference['derivation_step']['pressure_pairwise_difference_of_field'])} / "
            f"{scientific(reference['derivation_step']['density_diffusion_pairwise_difference_of_field'])}",
            f"{scientific(reference['worst_relative_error'])}（容差 {reference['tolerance']:.0e}）",
            "通过" if reference["passed"] else "未通过", entry["run_id"]])
    write("single_step_reference.md", markdown_table(
        ["配置", "流体 / 壁粒子", "A_f 回退（GPU = 参考）", "新密度误差（ulp，最大）", "密度变化率：超出 1 ulp 的相对误差",
         "加速度：最大相对误差 / 误差 ÷ 对项绝对值之和", "位移修正：同左", "L", "A_f⁻¹", "核和", "g_f",
         "推导恒等式（仅参考解）：压力 / 壁面密度扩散", "判据值", "结论", "记录"], rows))

    rows = []
    for entry in entries:
        residuals = entry.get("residuals")
        if not residuals:
            continue
        mode = MODE_TEXT[(entry["wall_boundary"], entry["continuity"])]
        if entry["configuration"] == "couette":
            value = residuals["acceleration_over_nu_shear_over_h"]
            rows.append([f"{entry['dimension']}-D", "Couette：粘性加速度残差 ÷ (ν · 剪切率 / h)", mode,
                         scientific(value["all"]["maximum"]), scientific(value["within_h_of_a_plate"]["root_mean_square"]),
                         "—", entry["run_id"]])
        else:
            value = residuals["acceleration_over_gravity"]
            sign = "p > 0" if entry["configuration"].endswith("positive") else "p < 0"
            rows.append([f"{entry['dimension']}-D", f"静水（{sign}）：加速度大小 ÷ 重力大小", mode,
                         scientific(value["all"]["maximum"]), scientific(value["within_h_of_a_wall"]["root_mean_square"]),
                         scientific(value["interior"]["maximum"]), entry["run_id"]])
    write("single_step_physics.md", markdown_table(
        ["维度", "配置与量", "壁面", "全部流体：最大", "近壁（h 内）：均方根", "内部：最大", "记录"], rows))


# =============================================================================
# Register counts
# =============================================================================

def register_table(records: list) -> None:
    directory = LOG_ROOT / "statistics"
    if not directory.exists():
        return
    by_output = {}
    for record in records:
        if record["kind"] == "pipeline_statistics":
            match = re.search(r"--out (\S+)", record.get("command", ""))
            if match:
                by_output[pathlib.Path(match.group(1)).name] = record

    def registers(report: dict, name: str) -> str:
        executables = report["pipelines"].get(name)
        if not executables:
            return "—"
        statistics = executables[0]["statistics"]
        return f"{statistics['Register Count']}（{statistics['Binary Size']:,} B）"

    rows = []
    for case_name, case_text in (("n1000", "2-D 方腔 1000²"), ("cavity3d_1m", "3-D 方腔 1M")):
        for tree, tree_text in (("base", "基线 eda5b90"), ("branch", "本分支")):
            for mode in ("simple", "adami", "crossing", "crossing_wall_velocity"):
                path = directory / f"{tree}_{case_name}_{mode}.json"
                if not path.exists():
                    continue
                report = json.loads(path.read_text(encoding="utf-8"))
                record = by_output.get(path.name, {})
                rows.append([case_text, tree_text, mode.replace("crossing_wall_velocity", "crossing（wall_velocity）"),
                             registers(report, "correction_density_all"), registers(report, "force_all"),
                             registers(report, "wall_extrapolate"), record.get("run_id", "?")])
    if rows:
        write("registers.md", markdown_table(
            ["算例", "代码", "壁面", "第一遍 correction_density：寄存器（二进制大小）", "第二遍 force", "壁面遍 wall_extrapolate",
             "记录"], rows))


# =============================================================================
# Per-kernel timing
# =============================================================================

def timing_tables() -> None:
    for path in sorted((LOG_ROOT / "timing").glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        kernels = ("predict_us", "update_voxel_us", "correction_density_us", "density_copy_us", "wall_extrapolate_us",
                   "force_us", "step_total_us")
        reference = summary["labels"].get("branch_simple", {}).get("kernels", {}).get("step_total_us", {})
        rows = []
        for label, entry in summary["labels"].items():
            cells = []
            for kernel in kernels:
                value = entry["kernels"].get(kernel)
                cells.append("—" if value is None else
                             f"{value['median_us']:.1f}（{value['minimum_us']:.1f}–{value['maximum_us']:.1f}）")
            total = entry["kernels"]["step_total_us"]["median_us"]
            ratio = "—" if not reference else f"{100.0 * (total / reference['median_us'] - 1.0):+.1f} %"
            rows.append([label, *cells, ratio, f"{entry['run_ids'][0]} … {entry['run_ids'][-1][-4:]}"])
        text = (f"每步 GPU 时间（µs）：{summary['rounds']} 轮交替测量的中位数（最小–最大）；每次 {summary['max_steps']} 步，"
                f"前 {summary['warmup']} 步不计；设备 {summary['uuid']}。“相对 simple” = 整步时间相对本分支 simple。\n\n"
                + markdown_table(["配置", "predict", "update_voxel", "correction_density", "density_copy",
                                  "wall_extrapolate", "force", "整步", "相对 simple", "记录"], rows))
        write(f"timing_{path.parent.name}.md", text)


# =============================================================================
# Bit comparisons
# =============================================================================

def bitwise_table(records: list) -> None:
    latest = {}
    for record in records:
        if record["kind"] == "bitwise_compare":
            latest[record.get("label", record["run_id"])] = record
    if not latest:
        return
    rows = [[label, record.get("verdict", record.get("status")), record["commit"], record["run_id"]]
            for label, record in sorted(latest.items())]
    write("bitwise.md", markdown_table(["比较（算例、步数窗口、K、两侧）", "结论", "比较时本分支的提交", "记录"], rows))


# =============================================================================
# box_run runs
# =============================================================================

def box_run_table(records: list) -> None:
    latest = {}
    for record in records:
        if record["kind"] == "box_run" and record.get("name"):
            latest[record["name"]] = record
    rows = {"diag_": [], "cavity3d_": []}
    for name, record in sorted(latest.items()):
        group = next((prefix for prefix in rows if name.startswith(prefix)), None)
        if group is None:
            continue
        directory = LOG_ROOT / "box_run" / name
        rate = "—"
        duration = "—"
        if (directory / "series.csv").exists():
            with open(directory / "series.csv", encoding="utf-8", newline="") as handle:
                series = list(csv.DictReader(handle))
            if len(series) >= 6 and "density_mean" in series[0]:
                time = np.array([float(row["time"]) for row in series])
                mean = np.array([float(row["density_mean"]) for row in series])
                half = time >= 0.5 * time[-1]
                rate = f"{float(np.polyfit(time[half], mean[half], 1)[0]):+.5f}"
                duration = f"{time[-1]:.2f}"
        summary = record.get("summary") or {}
        rows[group].append([
            name, duration, str(record.get("status")), rate,
            "—" if "density_deviation_maximum" not in summary else f"{summary['density_deviation_maximum']:.3f}",
            "—" if "wall_distance_minimum_over_spacing" not in summary
            else f"{summary['wall_distance_minimum_over_spacing']:.3f}",
            "—" if "below_floor_count_maximum" not in summary
            else f"{summary['below_floor_count_maximum']} / {summary['outside_box_count_maximum']}",
            str(summary.get("gradient_fallbacks", "—")), str(summary.get("overflow", "—")),
            "—" if "steps_per_second" not in summary else f"{summary['steps_per_second']:.0f}",
            record["commit"], record["run_id"]])
    header = ["运行", "到达的 t", "状态", "平均密度漂移率（后半段，kg/m³ 每时间单位）", "最大密度偏差（kg/m³）", "φ_min / dx",
              "低于下限 / 越出盒（最大粒子数）", "A_f 回退", "体素表溢出", "步 / 秒（与别的作业共用 GPU）", "提交", "记录"]
    if rows["diag_"]:
        write("box_runs_diagnostics.md", markdown_table(header, rows["diag_"]))
    if rows["cavity3d_"]:
        write("box_runs_cavity3d.md", markdown_table(header, rows["cavity3d_"]))


def cavity3d_profile_tables(records: list) -> None:
    """The 3-D cavity runs at their latest common profile time: per mode the kinetic energy and the extrema of
    the two centre-line profiles; per pair of modes the rms and the largest difference of the profiles."""
    latest = {}
    for record in records:
        if record["kind"] == "box_run" and str(record.get("name", "")).startswith("cavity3d_1m_"):
            latest[record["name"]] = record
    runs = {}
    for name, record in latest.items():
        directory = LOG_ROOT / "box_run" / name
        if not (directory / "profiles.json").exists() or not (directory / "series.csv").exists():
            continue
        profiles = json.loads((directory / "profiles.json").read_text(encoding="utf-8"))
        with open(directory / "series.csv", encoding="utf-8", newline="") as handle:
            series = list(csv.DictReader(handle))
        if profiles and series:
            runs[name.replace("cavity3d_1m_", "")] = (profiles, series, record)
    if len(runs) < 2:
        return
    common_time = min(profiles[-1]["time"] for profiles, _, _ in runs.values())
    chosen = {}
    rows = []
    order = [mode for mode in ("simple", "adami", "crossing", "crossing_wall_velocity") if mode in runs]
    for mode in order:
        profiles, series, record = runs[mode]
        profile = min(profiles, key=lambda entry: abs(entry["time"] - common_time))
        sample = min(series, key=lambda row: abs(float(row["time"]) - common_time))
        chosen[mode] = profile
        coordinate_u = np.array(profile["u_along_y"]["coordinate"])
        velocity_u = np.array(profile["u_along_y"]["velocity"], dtype=float)
        coordinate_v = np.array(profile["v_along_x"]["coordinate"])
        velocity_v = np.array(profile["v_along_x"]["velocity"], dtype=float)
        index_u, index_max, index_min = np.nanargmin(velocity_u), np.nanargmax(velocity_v), np.nanargmin(velocity_v)
        rows.append([mode, f"{profile['time']:.2f}", f"{float(sample['kinetic_energy']):.4f}",
                     f"{velocity_u[index_u]:.4f}（y = {coordinate_u[index_u]:.2f}）",
                     f"{velocity_v[index_max]:.4f}（x = {coordinate_v[index_max]:.2f}）",
                     f"{velocity_v[index_min]:.4f}（x = {coordinate_v[index_min]:.2f}）",
                     f"{float(series[-1]['time']):.2f}", str(record.get("status")), record["run_id"]])
    write("cavity3d_modes.md", markdown_table(
        ["壁面", "剖面时刻 t", "该时刻动能", "u 最小值（竖直中线）", "v 最大值（水平中线）", "v 最小值（水平中线）",
         "运行到达的 t", "状态", "记录"], rows))
    rows = []
    for first_index, first in enumerate(order):
        for second in order[first_index + 1:]:
            cells = [f"{first} − {second}"]
            for line in ("u_along_y", "v_along_x"):
                difference = (np.array(chosen[first][line]["velocity"], dtype=float)
                              - np.array(chosen[second][line]["velocity"], dtype=float))
                difference = difference[np.isfinite(difference)]
                cells += [f"{float(np.sqrt((difference ** 2).mean())):.4f}", f"{float(np.abs(difference).max()):.4f}"]
            rows.append(cells)
    write("cavity3d_differences.md", markdown_table(
        ["两种壁面", "u(y) 差的均方根", "u(y) 差的最大值", "v(x) 差的均方根", "v(x) 差的最大值"], rows))


def seam_audit_table() -> None:
    """Every seam audit of logs/wall_crossing/seam_audit in one table: per case and horizon, the rms ratios at the
    cut (d = 0) of the three pairings the audit makes, for the acceleration and the density."""
    rows = []
    for path in sorted((LOG_ROOT / "seam_audit").glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        for horizon, row in summary["horizons"].items():
            first = row["d0_rms_ratio"]
            second = row.get("second_pair_d0_rms_ratio") or {}
            noise = row.get("test_noise_d0_rms_ratio") or {}
            far = row["far_bin_worst_rms_ratio"]

            def cell(value) -> str:
                return "—" if value is None else f"{value:.2f}"

            invariants = row["invariants"]
            clean = row["all_runs_valid"] and all(
                (values.get("drift") or 0) == 0 and (values.get("overflow_total") or 0) == 0
                and (values.get("missing_ids") or 0) == 0 and (values.get("duplicate_ids") or 0) == 0
                and (values.get("stamp_errors_gpu") or 0) == 0 and (values.get("stamp_errors_host") or 0) == 0
                for values in invariants.values())
            rows.append([
                summary["label"], horizon, f"{row['triplets']:,}",
                f"{cell(row['seam_excess'])} / {cell(second.get('acceleration'))} / {cell(noise.get('acceleration'))}",
                f"{cell(first['density']['all'])} / {cell(second.get('density'))} / {cell(noise.get('density'))}",
                f"{cell(first['shift']['all'])} / {cell(second.get('shift'))} / {cell(noise.get('shift'))}",
                f"{cell(far['acceleration']['value'])} / {cell(far['density']['value'])} / {cell(far['shift']['value'])}",
                "全部为 0" if clean else json.dumps(invariants),
                f"{summary['run_ids'][0]} … {summary['run_ids'][-1][-4:]}" if summary.get("run_ids") else "?"])
    if rows:
        write("seam_all.md", markdown_table(
            ["算例", "步数 N", "配对粒子", "加速度 d = 0：(B1, A1) / (B2, A2) / 链自身 (B1, B2)", "密度 d = 0：同左",
             "位移修正 d = 0：同左", "远区最坏（加速度 / 密度 / 位移修正）", "四个运行的不变量（漂移、编号、戳、溢出）", "记录"],
            rows))


def commits_table() -> None:
    """The branch's commits since the base, oldest first."""
    import subprocess
    completed = subprocess.run(["git", "log", "--reverse", "--format=%h%x09%s", "eda5b90..HEAD"], cwd=REPOSITORY_ROOT,
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
    rows = [line.split("\t", 1) for line in completed.stdout.splitlines() if "\t" in line]
    if rows:
        write("commits.md", markdown_table(["提交", "标题"], [[f"`{commit}`", title.replace("|", "/")]
                                                         for commit, title in rows])
              + "\n（加入本报告的那个提交排在这些之后。）\n")


def copies() -> None:
    """Summaries the run tools wrote under logs/ (not tracked), copied beside the report; and the run records."""
    for source, target in (("bitwise/classes/classes_k1.md", "bitwise_classes.md"),
                           ("seam_audit/n250_crossing/summary.md", "seam_n250_crossing.md"),
                           ("seam_audit/n250_simple/summary.md", "seam_n250_simple.md"),
                           ("seam_audit/cavity3d_1m_crossing/summary.md", "seam_cavity3d_1m_crossing.md"),
                           ("seam_audit/cavity3d_1m_simple/summary.md", "seam_cavity3d_1m_simple.md"),
                           ("k2_equivalence/n250_simple/summary.md", "k2_n250_simple.md"),
                           ("k2_equivalence/hydrostatic_g30_crossing/summary.md", "k2_hydrostatic_g30_crossing.md"),
                           ("k2_equivalence/n250_crossing_700/summary.md", "k2_n250_crossing_700.md"),
                           ("k2_equivalence/n250_crossing/summary.md", "k2_n250_crossing_999.md")):
        path = LOG_ROOT / source
        if path.exists():
            write(target, path.read_text(encoding="utf-8"))
    for source, target in (("runs.jsonl", "runs.jsonl"),
                           ("remote/runs_DESKTOP-9IN3NPD.jsonl", "runs_DESKTOP-9IN3NPD.jsonl"),
                           ("tests/failures_base.txt", "tests/seam_layout_failures_base_eda5b90.txt"),
                           ("tests/failures_head.txt", "tests/seam_layout_failures_branch.txt")):
        path = LOG_ROOT / source
        if path.exists():
            (REPORT_ROOT / target).parent.mkdir(parents=True, exist_ok=True)
            (REPORT_ROOT / target).write_text(path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8",
                                              newline="\n")
            print(f"[report_tables] copied {source} -> claude/wall_crossing/{target}")


def hydrostatic_other_table(records: list) -> None:
    by_name = {}
    for record in records:
        if record["kind"] == "hydrostatic":
            name = record.get("name") or pathlib.Path(record.get("case", "")).parent.name.replace("hydrostatic_", "")
            by_name.setdefault(name, []).append(record)
    rows = []
    for name, runs in sorted(by_name.items()):
        for record in runs[:-1]:
            rows.append([name, record["commit"], str(record.get("status"))[:150], record["run_id"]])
    if rows:
        write("hydrostatic_other.md", markdown_table(["运行名", "提交", "状态", "记录"], rows))


# =============================================================================
# K = 2 throughput
# =============================================================================

def throughput_table() -> None:
    directory = LOG_ROOT / "throughput"
    rows = []
    for path in sorted(directory.glob("*.json")) if directory.exists() else []:
        entry = json.loads(path.read_text(encoding="utf-8"))
        rows.append([entry["label"], entry["wall_boundary"], str(entry["slabs"]),
                     " / ".join(f"{value:.1f}" for value in entry["steps_per_second_trials"]),
                     f"{entry['steps_per_second_median']:.1f}", str(entry["drift"]),
                     ", ".join(entry["run_ids"])])
    if rows:
        write("throughput.md", markdown_table(
            ["算例", "壁面", "K", "各次（步 / 秒）", "中位数", "粒子数漂移", "记录"], rows))


# =============================================================================
# Assembly
# =============================================================================

def build_report() -> None:
    source = REPORT_ROOT / "report_source.md"
    if not source.exists():
        print("[report_tables] no report_source.md yet")
        return
    output = []
    for line in source.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"\s*<!-- include: (\S+) -->\s*", line)
        if not match:
            output.append(line)
            continue
        path = REPORT_ROOT / match.group(1)
        if not path.exists():
            output.append(f"（**未完成**：这一部分（`{match.group(1)}`）在长运行按用户要求停止时还没有写或没有数据，"
                          "之后没有继续；见第 6 节。）")
            continue
        # the file's own title line is dropped; its other headings become bold lines (they would otherwise
        # break the report's outline)
        included = []
        for text in path.read_text(encoding="utf-8").splitlines():
            if text.startswith("# "):
                continue
            heading = re.fullmatch(r"#{2,6} (.*)", text)
            included.append(f"**{heading.group(1)}**" if heading else text)
        while included and not included[0].strip():
            included.pop(0)
        while included and not included[-1].strip():
            included.pop()
        output += included
    target = REPOSITORY_ROOT / "claude" / "wall_crossing_report.md"
    target.write_text("\n".join(output) + "\n", encoding="utf-8", newline="\n")
    print(f"[report_tables] wrote {target.relative_to(REPOSITORY_ROOT)} ({len(output)} lines)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-build", action="store_true")
    arguments = parser.parse_args()
    records = run_records()
    single_step_tables()
    register_table(records)
    timing_tables()
    bitwise_table(records)
    box_run_table(records)
    cavity3d_profile_tables(records)
    hydrostatic_other_table(records)
    throughput_table()
    seam_audit_table()
    commits_table()
    copies()
    if not arguments.no_build:
        build_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
