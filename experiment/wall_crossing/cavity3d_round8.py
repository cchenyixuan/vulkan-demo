"""
cavity3d_round8.py - the tables and figures of claude/wall_store_report.md section 14 (round 8: the 3-D cubic cavity
at Re = 1000 against Albensoeder & Kuhlmann 2005, h/dx = 3.5 and 4).

    .venv/Scripts/python.exe -m experiment.wall_crossing.cavity3d_round8 old         # 14.1: the earlier runs (3.1)
    .venv/Scripts/python.exe -m experiment.wall_crossing.cavity3d_round8 smoke       # 14.2: smoke tests
    .venv/Scripts/python.exe -m experiment.wall_crossing.cavity3d_round8 timing      # 14.3
    .venv/Scripts/python.exe -m experiment.wall_crossing.cavity3d_round8 accuracy    # 14.4 / 14.5

Inputs: logs/wall_store/round8/{old_runs.json, smoke_summary.json, timing/summary.json, production/...}
(the production runs' lines.npz and series.csv; the remote runs' small files pulled back). Tables go to
claude/wall_store/tables/round8_*.md, figures to claude/wall_store/figures/round8_*.png, numbers to
logs/wall_store/round8/accuracy.json.
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

from experiment.wall_crossing.cavity3d_benchmark import analyse_lines, load_lines, reference   # noqa: E402
from experiment.wall_crossing.cavity_tables import markdown_table as plain_markdown_table       # noqa: E402


def escape_bars(text) -> str:
    """'|' inside a cell (|w|, ⟨|v|²⟩) would split the column: escape every bare one."""
    return re.sub(r"(?<!\\)\|", r"\\|", str(text))


def markdown_table(header: list, rows: list) -> str:
    return plain_markdown_table([escape_bars(cell) for cell in header],
                                [[escape_bars(cell) for cell in row] for row in rows])

ROUND = REPOSITORY_ROOT / "logs" / "wall_store" / "round8"
TABLES = REPOSITORY_ROOT / "claude" / "wall_store" / "tables"
FIGURES = REPOSITORY_ROOT / "claude" / "wall_store" / "figures"
WINDOW = (35.0, 45.0)
HALVES = ((35.0, 40.0), (40.0, 45.0))
QUANTITY_TEXT = {"u_min": "u_min", "v_max": "v_max", "v_min": "v_min", "w_abs_max": "|w|max"}


def write(name: str, text: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / name).write_text(text, encoding="utf-8", newline="\n")
    print(f"-> {TABLES / name}")


def percent(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:+.{digits}f} %"


# ---- 14.1 ------------------------------------------------------------------------------------------------------

def old_tables() -> None:
    records = json.loads((ROUND / "old_runs.json").read_text(encoding="utf-8"))
    table10 = reference()["extrema"]
    rows, snapshot_rows = [], []
    for record in records:
        reported = record["as_reported"]
        grid = record["as_reported_deviation_grid"]
        refined = record["refined_shepard101"]
        cells = [record["label"], str(record["window_profiles"])]
        for key in ("u_min", "v_max", "v_min"):
            cells.append(f"{reported[key]:+.4f}（{reported[key + '_position']:.2f}）：{percent(grid[key])} → "
                         f"{refined[key]['value']:+.4f}（{refined[key]['position']:.4f}）：**{percent(refined[key]['deviation'])}**")
        rows.append(cells)
        for snapshot in record["snapshots"]:
            extrema = snapshot["extrema"]
            w = extrema["w_abs_max"]
            snapshot_rows.append([record["label"], f"{snapshot['label']}（t = {snapshot['time']:.2f}）"]
                                 + [f"{extrema[key]['value']:+.4f}（{extrema[key]['position']:.4f}），"
                                    f"{percent(extrema[key]['deviation'])}" for key in ("u_min", "v_max", "v_min")]
                                 + [f"{w['value']:.4f}（{w['position']:.4f} / {w['position_2']:.4f}），"
                                    f"{percent(w['deviation'])}；两侧 {w['sides'][0]['magnitude']:.4f} / "
                                    f"{w['sides'][1]['magnitude']:.4f}"])
    reference_row = ["表 10", "—"] + [f"{table10[key]['value']:+.4f}（{table10[key]['position']:.5f}）"
                                      for key in ("u_min", "v_max", "v_min")]
    write("round8_old_extrema.md", markdown_table(
        ["运行（h/dx = 4，Re = 1000 (1 + dx)）", "窗口 35–45 的剖面数",
         "u_min：以前的值（位置）：对表 5 格点的偏差 → 抛物线细化（位置）：对表 10 的偏差",
         "v_max：同左", "v_min：同左"], rows + [reference_row + [""] * 0]))
    write("round8_old_snapshots.md", markdown_table(
        ["运行", "快照", "u_min（位置），对表 10", "v_max", "v_min",
         "|w|max（两侧平均；位置），对表 10 的 0.0303398；两侧的值"], snapshot_rows))


# ---- 14.2 ------------------------------------------------------------------------------------------------------

def smoke_tables() -> None:
    data = json.loads((ROUND / "smoke_summary.json").read_text(encoding="utf-8"))
    runs = data["runs"]
    order = sorted(runs, key=lambda label: int(label[1]))
    rows = []
    for label in order:
        record = runs[label]
        rows.append([label.split("_")[0], label.split("_", 1)[1].replace("_", " "),
                     f"{record['time_reached']:.2f}" + ("" if record["completed"] else f"（停：{record['stop']}）"),
                     "是" if record["finite"] else "**否**",
                     f"{record['speed_maximum']:.3f}（t {record['speed_maximum_time']:.2f}）",
                     "、".join(f"{value:,}" for value in record["alive_readback"]) + " / "
                     + "、".join(f"{value:,}" for value in record["alive_gpu"]),
                     f"{record['overflow_maximum']:.0f} / {record['far_migrations']:.0f} / {record['stamp_errors']:.0f} / "
                     f"{record['zombie_samples']}",
                     f"{record['outside_box_maximum']:.0f}（{record['below_floor_maximum']:.0f}）",
                     f"{record['pair_distance_minimum_over_spacing']:.3f} / "
                     f"{record['wall_particle_distance_minimum_over_spacing']:.3f}",
                     f"{record['correction_fallbacks']:.0f} / {record['gradient_fallbacks']:.0f}",
                     f"{0.5 * 0 + record['specific_energy']['3.0']:.5f}" if math.isfinite(record["specific_energy"]["3.0"])
                     else "—"])
    write("round8_smoke.md", markdown_table(
        ["运行", "设置", "跑到 t", "全有限", "最大速度 / U（时刻）", "粒子数：读回 / GPU",
         "溢出 / 远距迁移 / 时间戳错误 / 僵尸（读回 ≠ GPU 的采样数）", "盒外流体（低于距离下限）",
         "最小间距 / Δx：流体–流体 / 流体–壁粒子（t > 0）", "修正矩阵回退 / A_f 回退（累计）", "t = 3 的 ½⟨|v|²⟩"], rows))
    energy_rows = [[name] + [f"{percent(value, 2)}（t = {float(t):g}）" for t, value in values.items()]
                   for name, values in data["energy_relative_difference_3p5_vs_4"].items()]
    write("round8_smoke_energy.md", markdown_table(["对比（3.5 相对 4）", "比能 ½⟨|v|²⟩ 的相对差"],
                                                   [[row[0], "；".join(row[1:])] for row in energy_rows]))
    correction_rows = []
    for label in order:
        record = runs[label]
        for when in ("correction_initial", "correction_final"):
            if when not in record:
                continue
            for class_name in ("corner", "edge", "face", "interior"):
                statistics = record[when][class_name]
                correction_rows.append([
                    label.split("_")[0], "t = 0" if when == "correction_initial" else f"末态 t = {record['time_reached']:.2f}",
                    {"corner": "角（3 面在 h 内）", "edge": "棱（2 面）", "face": "面（1 面）", "interior": "内部（抽 2 万）"}[class_name],
                    f"{statistics['count']:,}",
                    f"{statistics['smallest_eigenvalue']['min']:.4f} / {statistics['smallest_eigenvalue']['p01']:.4f} / "
                    f"{statistics['smallest_eigenvalue']['p50']:.4f}",
                    f"{statistics['condition']['p50']:.2f} / {statistics['condition']['p99']:.2f} / {statistics['condition']['max']:.2f}",
                    f"{statistics['determinant_min']:.4f}",
                    f"{statistics['inverse_norm_max']:.2f}",
                    f"{statistics['fallback_count']} / {statistics['frobenius_clamped_count']}",
                    "(" + ", ".join(f"{value:.2f}" for value in statistics["worst_distance_to_faces_over_dx"]) + ")"])
    write("round8_smoke_correction.md", markdown_table(
        ["运行", "状态", "粒子类", "个数", "最小特征值：最小 / 1 % / 中位", "条件数：中位 / 99 % / 最大",
         "det(M + ξI) 最小", "‖(M + ξI)⁻¹‖_F 最大", "det ≤ 1e-4（回退）/ 逆的范数 > 10（缩放）",
         "最差粒子离三个面的距离 / Δx"], correction_rows))
    acceleration_rows = []
    for label in order:
        record = runs[label]
        counts = record["acceleration_faces_within_h_counts"]
        position = record["acceleration_maximum_position"]
        top = record.get("final_top_accelerations")
        acceleration_rows.append([
            label.split("_")[0], f"{record['acceleration_maximum']:.1f}（t {record['acceleration_maximum_time']:.2f}，"
                                 f"({position[0]:+.3f}, {position[1]:+.3f}, {position[2]:+.3f})，"
                                 f"{record['acceleration_maximum_faces_within_h']} 面在 h 内）",
            f"{record['acceleration_p999_maximum']:.1f}",
            f"角 {counts['3']} / 棱 {counts['2']} / 面 {counts['1']} / 内部 {counts['0']}",
            ("—" if top is None else
             f"中位 {top['median']:.2f}；前 20 个：{min(top['values']):.1f}–{max(top['values']):.1f}，"
             f"在 h 内的面数 {sorted(top['faces_within_h'], reverse=True)}，离最近面 "
             f"{min(top['distance_to_nearest_face_over_dx']):.2f}–{max(top['distance_to_nearest_face_over_dx']):.2f} Δx")])
    write("round8_smoke_acceleration.md", markdown_table(
        ["运行", "全程最大加速度（时刻，位置，几面在 h 内）", "各采样 99.9 % 分位的最大",
         "每次采样最大加速度粒子所在的类（采样数）", "末态：加速度最大的 20 个粒子"], acceleration_rows))


# ---- 14.3 ------------------------------------------------------------------------------------------------------

TIMESTEP = {"3p5": 5.25e-5, "4": 6.0e-5}       # N = 100


def timing_tables() -> None:
    """(a) the brief's protocol: kernel_timing (the single-GPU bench, one frame in flight, GPU timestamps of the whole
    step), 2,000 steps, the first 200 discarded, three interleaved rounds, GPU 1 with GPU 0 idle; the per-frame csv
    also gives the frames after the first defrag (the bench sorts the particles only at step 1,000, box_run at the
    bootstrap). (b) the production path: box_run (depth 2) on GPU 1 with GPU 0 idle, 6,000 steps, steps per second of
    the defrag intervals after the first (stress.py defrag.csv, sampling excluded), two interleaved rounds."""
    rows_a, per_step = [], {}
    for method in ("crossing", "simple"):
        for hdx in ("3p5", "4"):
            label = f"{method}_hdx{hdx}"
            whole, frame, after = [], [], []
            for round_index in range(3):
                with open(ROUND / "timing" / f"{label}_r{round_index}.csv", newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                frames = np.asarray([int(row["frame_n"]) for row in rows])
                step = np.asarray([float(row["step_total_us"]) for row in rows])
                wall = np.asarray([float(row["frame_total_us"]) for row in rows])
                whole.append(step.mean()); frame.append(wall.mean()); after.append(step[frames > 1000].mean())
            per_step[(method, hdx, "a")] = float(np.median(whole))
            per_step[(method, hdx, "a_after")] = float(np.median(after))
            rows_a.append([method, "3.5" if hdx == "3p5" else "4",
                           f"{np.median(whole) / 1000:.3f}（{min(whole) / 1000:.3f}–{max(whole) / 1000:.3f}）",
                           f"{np.median(frame) / 1000:.3f}",
                           f"{np.median(whole) * 1e-6 / TIMESTEP[hdx]:.1f}",
                           f"{np.median(after) / 1000:.3f}", f"{np.median(after) * 1e-6 / TIMESTEP[hdx]:.1f}"])
    rows_b = []
    for method in ("crossing", "simple"):
        for hdx in ("3p5", "4"):
            values = []
            for directory in sorted((ROUND / "timing_production_path" / "box_run").glob(f"*_T_{method}_hdx{hdx}_r*")):
                with open(directory / "defrag.csv", newline="", encoding="utf-8") as handle:
                    rates = [float(row["steps_per_second"]) for row in csv.DictReader(handle) if int(row["frame"]) > 1000]
                values.append(float(np.mean([1e6 / rate for rate in rates])))
            per_step[(method, hdx, "b")] = float(np.mean(values))
            rows_b.append([method, "3.5" if hdx == "3p5" else "4",
                           " / ".join(f"{value / 1000:.3f}" for value in values),
                           f"{np.mean(values) * 1e-6 / TIMESTEP[hdx]:.1f}"])
    write("round8_timing_bench.md", markdown_table(
        ["方法", "h/dx", "每步 GPU ms：第 201–2,000 步（三轮中位，最小–最大）", "每步墙钟 ms（同一段）",
         "每单位模拟时间的秒数（GPU）", "每步 GPU ms：第 1,001–2,000 步（第一次整理之后）", "每单位模拟时间的秒数"], rows_a))
    write("round8_timing_production.md", markdown_table(
        ["方法", "h/dx", "每步 ms：第 1,001–6,000 步（两轮）", "每单位模拟时间的墙钟秒"], rows_b))
    ratio_rows = []
    for key, text in (("a", "基准程序，第 201–2,000 步（任务书的做法）"), ("a_after", "基准程序，第 1,001–2,000 步"),
                      ("b", "生产路径 box_run，第 1,001–6,000 步")):
        cells = [text]
        for hdx in ("3p5", "4"):
            cells.append(f"{per_step[('crossing', hdx, key)] / per_step[('simple', hdx, key)]:.3f}")
        for method in ("crossing", "simple"):
            step_ratio = per_step[(method, "3p5", key)] / per_step[(method, "4", key)]
            cells.append(f"{step_ratio:.3f} / {step_ratio * TIMESTEP['4'] / TIMESTEP['3p5']:.3f}")
        ratio_rows.append(cells)
    write("round8_timing_ratios.md", markdown_table(
        ["测法", "crossing ÷ simple（h/dx = 3.5，每步）", "crossing ÷ simple（h/dx = 4）",
         "crossing：3.5 ÷ 4，每步 / 每单位模拟时间", "simple：3.5 ÷ 4，每步 / 每单位模拟时间"], ratio_rows))


# ---- 14.4 / 14.5 -----------------------------------------------------------------------------------------------

def read_series(path: pathlib.Path) -> dict:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.asarray([float(row[key]) if row[key] not in ("", None) else np.nan for row in rows])
            for key in rows[0]}


def fit_order(spacings: list, values: list) -> dict:
    """Least-squares p of |e| = C dx^p (dx = 1 / N) and the pairwise orders; None when e changes sign."""
    values = np.asarray(values, dtype=np.float64)
    if np.any(values == 0) or not (np.all(values > 0) or np.all(values < 0)):
        return {"order": None, "pairwise": None}
    dx = 1.0 / np.asarray(spacings, dtype=np.float64)
    slope, _ = np.polyfit(np.log(dx), np.log(np.abs(values)), 1)
    pairwise = [float(np.log(abs(values[index + 1] / values[index])) / np.log(dx[index + 1] / dx[index]))
                for index in range(len(values) - 1)]
    return {"order": float(slope), "pairwise": pairwise}


CHOSEN_HDX = 3.5
RUN_COLOUR = {("crossing", 50): "#9ec5f4", ("crossing", 70): "#5a9be6", ("crossing", 100): "#2a78d6",
              ("crossing", 140): "#0b3f86", ("simple", 100): "#eb6834", ("simple", 140): "#b2441a"}


def run_text(entry: dict) -> str:
    return f"{entry['method']} N = {entry['N']}（h/dx = {entry['hdx']:g}）"


def accuracy_main(arguments) -> int:
    """production/runs.json: {label: {"directory", "method", "N", "hdx", "machine"}}."""
    runs = json.loads((ROUND / "production" / "runs.json").read_text(encoding="utf-8"))
    results = {}
    for label, entry in runs.items():
        directory = REPOSITORY_ROOT / entry["directory"]
        if not (directory / "lines.npz").exists():
            print(f"{label}: no lines.npz in {entry['directory']}; skipped")
            continue
        lines = load_lines(directory / "lines.npz")
        if lines["time"].max() < WINDOW[1] - 1e-6:
            print(f"{label}: lines end at t = {lines['time'].max():.2f}, before the window's end; skipped")
            continue
        result = {"window": analyse_lines(lines, WINDOW),
                  "halves": [analyse_lines(lines, half) for half in HALVES], **entry}
        for half in result["halves"]:
            half.pop("mean_profiles")
        series = read_series(directory / "series.csv")
        result["series"] = {"time": series["time"].tolist(),
                            "specific_energy": (0.5 * series["speed_root_mean_square"] ** 2).tolist(),
                            "speed_maximum": series["speed_maximum"].tolist(),
                            "density_mean": series["density_mean"].tolist(),
                            "alive": sorted(set(int(value) for value in series["alive"])),
                            "overflow_maximum": float(np.nanmax(series["overflow"])),
                            "outside_box_maximum": float(np.nanmax(series["outside_box_count"]))}
        summary_path = directory / "summary.json"
        result["summary"] = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
        results[label] = result
    (ROUND / "accuracy.json").write_text(json.dumps(results, indent=1), encoding="utf-8", newline="\n")
    print(f"-> {ROUND / 'accuracy.json'}")
    accuracy_tables(results)
    accuracy_figures(results)
    return 0


def accuracy_tables(results: dict) -> None:
    table10 = reference()["extrema"]
    ordered = sorted(results.items(), key=lambda item: (item[1]["hdx"] != CHOSEN_HDX, item[1]["method"] != "crossing",
                                                        item[1]["N"]))
    extrema_rows, profile_rows, steady_rows, run_rows = [], [], [], []
    for label, result in ordered:
        window = result["window"]
        extrema = window["extrema"]
        cells = [run_text(result), str(window["samples"])]
        for key in ("u_min", "v_max", "v_min"):
            cells.append(f"{extrema[key]['value']:+.4f}，{percent(extrema[key]['deviation'], 2)}，"
                         f"{extrema[key]['position']:.4f}（{extrema[key]['position_error']:+.4f}）")
        w = extrema["w_abs_max"]
        cells.append(f"{w['value']:.5f}，{percent(w['deviation'], 2)}，{w['position']:.4f} / {w['position_2']:.4f}"
                     f"（{w['position_error']:+.4f} / {w['position_2_error']:+.4f}）；两侧差 {w['side_difference']:+.5f}")
        extrema_rows.append(cells)
        profile_rows.append([run_text(result)] + [
            f"{window[f'{line}_{kind}_error']['l2']:.{4 if kind == 'velocity' else 5}f} / "
            f"{window[f'{line}_{kind}_error']['linf']:.{4 if kind == 'velocity' else 5}f}"
            for line in ("vertical", "horizontal") for kind in ("velocity", "pressure")])
        first, second = result["halves"]
        steady_rows.append([run_text(result)] + [
            f"{second['extrema'][key]['value'] - first['extrema'][key]['value']:+.5f}"
            f"（{100 * (abs(second['extrema'][key]['value']) - abs(first['extrema'][key]['value'])) / abs(table10[key]['value']):+.2f} 个百分点）"
            for key in ("u_min", "v_max", "v_min", "w_abs_max")]
            + [f"{second['vertical_velocity_error']['l2'] - first['vertical_velocity_error']['l2']:+.5f} / "
               f"{second['horizontal_velocity_error']['l2'] - first['horizontal_velocity_error']['l2']:+.5f}"])
        series = result["series"]
        time = np.asarray(series["time"])
        energy = np.asarray(series["specific_energy"])
        inside = (time >= WINDOW[0] - 1e-9) & (time <= WINDOW[1] + 1e-9)
        slope = np.polyfit(time[inside], energy[inside], 1)[0] if inside.sum() > 2 else float("nan")
        summary = result["summary"] or {}
        whole = summary.get("whole_run") or {}
        run_rows.append([run_text(result), result.get("machine", ""), f"{time.max():.2f}",
                         f"{float(np.max(series['speed_maximum'])):.3f}",
                         "、".join(f"{value:,}" for value in series["alive"]),
                         f"{series['overflow_maximum']:.0f} / {series['outside_box_maximum']:.0f}",
                         f"{whole.get('correction_fallbacks', '—')} / {whole.get('gradient_fallbacks', '—')}",
                         f"{whole.get('wall_distance_minimum_over_spacing', float('nan')):.3f}"
                         f"（{whole.get('below_floor_count_maximum', '—')}）",
                         f"{whole.get('density_deviation_maximum', float('nan')):.2f}",
                         f"{energy[inside].mean():.5f}", f"{100 * slope / energy[inside].mean():+.3f} %",
                         f"{summary.get('steps_per_second', float('nan')):.1f}",
                         f"{1.0 / summary.get('steps_per_second', float('nan')) / summary.get('timestep', float('nan')):.0f}"
                         if summary else "—"])
    reference_row = ["表 10", "—"] + [f"{table10[key]['value']:+.4f}，—，{table10[key]['position']:.5f}"
                                      for key in ("u_min", "v_max", "v_min")] + [
        f"{table10['w_abs_max']['value']:.5f}，—，{table10['w_abs_max']['position']:.5f} / "
        f"{table10['w_abs_max']['position_2']:.5f}"]
    write("round8_accuracy_extrema.md", markdown_table(
        ["运行", "窗口内样本", "u_min：值，偏差，位置（位置差）", "v_max：同左", "v_min：同左",
         "|w|max：两侧平均，偏差，两侧位置（位置差）；两侧差"], extrema_rows + [reference_row]))
    write("round8_accuracy_profiles.md", markdown_table(
        ["运行", "竖线 u：L2 / L∞", "竖线 p：L2 / L∞", "横线 v：L2 / L∞", "横线 p：L2 / L∞"], profile_rows))
    write("round8_accuracy_steady.md", markdown_table(
        ["运行（t = 40–45 减 35–40）", "u_min 的变化（相当于表 10 值的百分点）", "v_max", "v_min", "|w|max",
         "速度剖面 L2 的变化：u / v"], steady_rows))
    write("round8_accuracy_runs.md", markdown_table(
        ["运行", "机器", "跑到 t", "全程最大速度 / U", "粒子数（全程）", "溢出 / 盒外（最大）",
         "修正矩阵回退 / A_f 回退（累计）", "流体离盒面的最小距离 / dx（低于 0.25 dx 下限的最多个数，只有 crossing 计）", "密度最大偏差（kg/m³）",
         "窗口内 ½⟨|v|²⟩",
         "窗口内每单位时间的变化", "步 / 秒（含采样）", "每单位模拟时间的墙钟秒"], run_rows))
    # convergence orders, crossing over N
    crossing = sorted(((result["N"], result) for result in results.values()
                       if result["method"] == "crossing" and result["hdx"] == CHOSEN_HDX), key=lambda item: item[0])
    comparison_table(results)
    equal_accuracy_table(results)
    if len(crossing) >= 2:
        spacings = [item[0] for item in crossing]
        order_rows = []
        quantities = {f"{QUANTITY_TEXT[key]} 偏差": [item[1]["window"]["extrema"][key]["deviation"] for item in crossing]
                      for key in ("u_min", "v_max", "v_min", "w_abs_max")}
        for line, letter in (("vertical", "u"), ("horizontal", "v")):
            quantities[f"{letter} 剖面 L2"] = [item[1]["window"][f"{line}_velocity_error"]["l2"] for item in crossing]
            quantities[f"{'竖' if line == 'vertical' else '横'}线 p 剖面 L2"] = [
                item[1]["window"][f"{line}_pressure_error"]["l2"] for item in crossing]
        for name, values in quantities.items():
            fitted = fit_order(spacings, values)
            text = ("误差变号，不给阶" if fitted["order"] is None else
                    f"p = {fitted['order']:.2f}（两两：{'、'.join(f'{value:.2f}' for value in fitted['pairwise'])}）")
            order_rows.append([name, " / ".join(percent(value, 2) if "偏差" in name else
                                                f"{value:.5f}" if " p " in name else f"{value:.4f}"
                                                for value in values), text])
        write("round8_accuracy_orders.md", markdown_table(
            [f"crossing 的量（N = {' / '.join(str(value) for value in spacings)}）", "值", "表观阶（误差 ∝ dx^p）"],
            order_rows))


def comparison_table(results: dict) -> None:
    """h/dx = 3.5 against 4 at the same N (crossing; the 4 runs are the comparison runs of section 14.6)."""
    rows = []
    for size in sorted({result["N"] for result in results.values()}):
        pair = {result["hdx"]: result for result in results.values()
                if result["method"] == "crossing" and result["N"] == size}
        if len(pair) < 2:
            continue
        for hdx in sorted(pair, reverse=True):
            window = pair[hdx]["window"]
            extrema = window["extrema"]
            rows.append([str(size), f"{hdx:g}"]
                        + [percent(extrema[key]["deviation"], 2) for key in ("u_min", "v_max", "v_min", "w_abs_max")]
                        + [f"{window['vertical_velocity_error']['l2']:.4f} / {window['horizontal_velocity_error']['l2']:.4f}",
                           f"{window['vertical_pressure_error']['l2']:.5f} / {window['horizontal_pressure_error']['l2']:.5f}"])
    if rows:
        write("round8_accuracy_hdx.md", markdown_table(
            ["N", "h/dx", "u_min 偏差", "v_max 偏差", "v_min 偏差", "|w|max 偏差", "速度 L2：u / v", "压力 L2：竖线 / 横线"],
            rows))


def seconds_per_unit_time(result: dict) -> float:
    summary = result["summary"]
    return 1.0 / (summary["steps_per_second"] * summary["timestep"])


def equal_accuracy_table(results: dict) -> None:
    """What h/dx = CHOSEN_HDX costs to reach the error of each other-h/dx crossing run on one card. The chosen runs'
    error is fitted as C N^-p over every N; their cost (wall seconds per unit simulated time, single-card runs only) is
    interpolated piecewise linearly in log-log (end segments extended)."""
    chosen = sorted((result for result in results.values()
                     if result["method"] == "crossing" and result["hdx"] == CHOSEN_HDX), key=lambda result: result["N"])
    single = [result for result in chosen if result.get("machine") == "远端单卡"]
    others = sorted((result for result in results.values() if result["method"] == "crossing"
                     and result["hdx"] != CHOSEN_HDX and result.get("machine") == "远端单卡"),
                    key=lambda result: result["N"])
    if len(chosen) < 2 or len(single) < 2 or not others:
        return
    sizes = np.log([result["N"] for result in chosen])
    cost_sizes = np.log([result["N"] for result in single])
    costs = np.log([seconds_per_unit_time(result) for result in single])

    def cost_at(size: float) -> float:
        position = math.log(size)
        index = int(np.clip(np.searchsorted(cost_sizes, position) - 1, 0, cost_sizes.size - 2))
        slope = (costs[index + 1] - costs[index]) / (cost_sizes[index + 1] - cost_sizes[index])
        return float(math.exp(costs[index] + slope * (position - cost_sizes[index])))

    quantities = [(QUANTITY_TEXT[key] + " 偏差", lambda result, key=key: abs(result["window"]["extrema"][key]["deviation"]),
                   True) for key in ("u_min", "v_max", "v_min", "w_abs_max")]
    quantities += [(f"{letter} 剖面 L2", lambda result, line=line: result["window"][f"{line}_velocity_error"]["l2"], False)
                   for letter, line in (("u", "vertical"), ("v", "horizontal"))]
    rows = []
    for name, measure, is_percent in quantities:
        slope, intercept = np.polyfit(sizes, np.log([measure(result) for result in chosen]), 1)
        for other in others:
            target = measure(other)
            needed = math.exp((math.log(target) - intercept) / slope)
            ours, theirs = cost_at(needed), seconds_per_unit_time(other)
            rows.append([name, f"h/dx = {other['hdx']:g}，N = {other['N']}",
                         percent(target, 2) if is_percent else f"{target:.4f}", f"{theirs:.0f}",
                         f"{needed:.0f}", f"{ours:.0f}", f"{ours / theirs:.2f}"])
    write("round8_accuracy_cost.md", markdown_table(
        ["量", "对照运行（远端单卡）", "它的误差（大小）", "它每单位模拟时间的墙钟秒",
         f"h/dx = {CHOSEN_HDX:g} 达到同样误差要的 N（按 {CHOSEN_HDX:g} 的拟合）", "那时每单位模拟时间的墙钟秒（插值）",
         f"{CHOSEN_HDX:g} ÷ 对照"], rows))


def accuracy_figures(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiment.wall_crossing.report_figures import INK_PRIMARY, INK_SECONDARY, save, setup_style
    setup_style()
    data = reference()
    everything = results
    results = {label: result for label, result in results.items() if result["hdx"] == CHOSEN_HDX}
    ordered = sorted(results.values(), key=lambda result: (result["method"] != "crossing", result["N"]))

    def style(result):
        return {"color": RUN_COLOUR.get((result["method"], result["N"]), INK_SECONDARY), "linewidth": 1.5,
                "linestyle": "-" if result["method"] == "crossing" else "--", "label": run_text(result)}

    def header(figure, title, handles, labels, columns=3):
        height = figure.get_figheight()
        figure.suptitle(title, x=0.01, y=1.0 - 0.05 / height, ha="left", va="top", fontsize=12, fontweight="bold",
                        color=INK_PRIMARY)
        figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0 - 0.40 / height), ncol=columns,
                      handlelength=3.0, labelcolor=INK_PRIMARY, borderaxespad=0.0)
        rows = math.ceil(len(labels) / columns)
        figure.tight_layout(rect=(0, 0, 1.0, 1.0 - (0.45 + 0.27 * rows) / height), w_pad=2.0, h_pad=1.5)

    # profiles: velocity and pressure on both centre lines
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 9.6))
    for result in ordered:
        mean = result["window"]["mean_profiles"]
        kw = style(result)
        axes[0, 0].plot(mean["vertical"]["velocity"], mean["vertical"]["coordinate"], **kw)
        axes[0, 1].plot(mean["horizontal"]["coordinate"], mean["horizontal"]["velocity"], **{**kw, "label": None})
        axes[1, 0].plot(mean["vertical"]["pressure"], mean["vertical"]["coordinate"], **{**kw, "label": None})
        axes[1, 1].plot(mean["horizontal"]["coordinate"], mean["horizontal"]["pressure"], **{**kw, "label": None})
    marker = {"marker": "o", "linestyle": "none", "color": INK_PRIMARY, "markersize": 4.5,
              "markerfacecolor": "none", "markeredgewidth": 1.2}
    axes[0, 0].plot(data["vertical"]["velocity"], data["vertical"]["coordinate"], label="Albensoeder & Kuhlmann 2005",
                    **marker)
    axes[0, 1].plot(data["horizontal"]["coordinate"], data["horizontal"]["velocity"], **marker)
    axes[1, 0].plot(data["vertical"]["pressure"], data["vertical"]["coordinate"], **marker)
    axes[1, 1].plot(data["horizontal"]["coordinate"], data["horizontal"]["pressure"], **marker)
    axes[0, 0].set_xlabel("u / U"); axes[0, 0].set_ylabel("y / L"); axes[0, 0].set_title("竖线 (0.5, y, 0.5)：u")
    axes[0, 1].set_xlabel("x / L"); axes[0, 1].set_ylabel("v / U"); axes[0, 1].set_title("横线 (x, 0.5, 0.5)：v")
    axes[1, 0].set_xlabel("(p − p_中心) / (ρU²)"); axes[1, 0].set_ylabel("y / L"); axes[1, 0].set_title("竖线：压力")
    axes[1, 1].set_xlabel("x / L"); axes[1, 1].set_ylabel("(p − p_中心) / (ρU²)"); axes[1, 1].set_title("横线：压力")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    header(figure, f"3-D 方腔 Re = 1000：t = {WINDOW[0]:g}–{WINDOW[1]:g} 的平均中心线剖面（符号 = 原文表 5、6）",
           handles, labels)
    save(figure, FIGURES / "round8_profiles.png")

    # spanwise w
    figure, axis = plt.subplots(1, 1, figsize=(12.0, 4.8))
    for result in ordered:
        mean = result["window"]["mean_profiles"]["spanwise"]
        axis.plot(mean["coordinate"], mean["velocity"], **style(result))
    w = data["extrema"]["w_abs_max"]
    signs = [side["sign"] for side in next(iter(results.values()))["window"]["extrema"]["w_abs_max"]["sides"]]
    axis.plot([w["position"], w["position_2"]], [signs[0] * w["value"], signs[1] * w["value"]],
              label="表 10 的 |w|max（符号取我们的）", **{**marker, "marker": "D"})
    axis.set_xlabel("z / L"); axis.set_ylabel("w / U"); axis.set_title("展向中心线 (0.5, 0.5, z)")
    handles, labels = axis.get_legend_handles_labels()
    header(figure, f"展向速度 w：t = {WINDOW[0]:g}–{WINDOW[1]:g} 的平均", handles, labels)
    save(figure, FIGURES / "round8_spanwise.png")

    # kinetic energy history
    figure, axis = plt.subplots(1, 1, figsize=(12.0, 4.8))
    for result in ordered:
        axis.plot(result["series"]["time"], result["series"]["specific_energy"], **style(result))
    axis.axvspan(WINDOW[0], WINDOW[1], color=INK_SECONDARY, alpha=0.08, linewidth=0)
    axis.set_xlabel("t（L / U）"); axis.set_ylabel("½⟨|v|²⟩ / U²"); axis.set_title("单位质量的动能（灰带 = 统计窗口）")
    handles, labels = axis.get_legend_handles_labels()
    header(figure, "动能历史", handles, labels)
    save(figure, FIGURES / "round8_energy.png")

    # convergence of the extrema deviations
    crossing = sorted((result for result in results.values() if result["method"] == "crossing"),
                      key=lambda result: result["N"])
    if len(crossing) >= 2:
        figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
        spacings = np.asarray([result["N"] for result in crossing], dtype=np.float64)
        colours = {"u_min": "#2a78d6", "v_max": "#eb6834", "v_min": "#1baf7a", "w_abs_max": "#9a5ac8"}
        for key, colour in colours.items():
            values = np.asarray([result["window"]["extrema"][key]["deviation"] for result in crossing])
            axes[0].plot(spacings, 100 * values, marker="o", color=colour, linewidth=1.5, markersize=6,
                         label=QUANTITY_TEXT[key])
            simple = [result for result in results.values() if result["method"] == "simple"]
            for result in simple:
                axes[0].plot([result["N"]], [100 * result["window"]["extrema"][key]["deviation"]], marker="s",
                             color=colour, markerfacecolor="none", linestyle="none", markersize=7)
        others = sorted((result for result in everything.values()
                         if result["method"] == "crossing" and result["hdx"] != CHOSEN_HDX), key=lambda result: result["N"])
        for key, colour in colours.items():
            if others:
                axes[0].plot([result["N"] for result in others],
                             [100 * result["window"]["extrema"][key]["deviation"] for result in others], marker="o",
                             markerfacecolor="none", color=colour, linewidth=1.0, linestyle=":", markersize=7)
        axes[0].axhline(0.0, color=INK_SECONDARY, linewidth=0.8)
        axes[0].set_xlabel("每边间距数 N"); axes[0].set_ylabel("极值大小相对表 10 的偏差（%）")
        axes[0].set_title("极值偏差（实心 = crossing 3.5，空心圆点线 = crossing 4，空心方块 = simple 3.5）")
        axes[0].set_xticks(spacings); axes[0].set_xticklabels([str(int(value)) for value in spacings])
        for key, colour, line in (("u L2", "#2a78d6", "vertical_velocity"), ("v L2", "#eb6834", "horizontal_velocity"),
                                  ("p 竖线 L2", "#1baf7a", "vertical_pressure"),
                                  ("p 横线 L2", "#9a5ac8", "horizontal_pressure")):
            values = np.asarray([result["window"][f"{line}_error"]["l2"] for result in crossing])
            axes[1].loglog(spacings, values, marker="o", color=colour, linewidth=1.5, markersize=6, label=key)
            if others:
                axes[1].loglog([result["N"] for result in others],
                               [result["window"][f"{line}_error"]["l2"] for result in others], marker="o",
                               markerfacecolor="none", color=colour, linewidth=1.0, linestyle=":", markersize=7)
        guide = values[0] * (spacings / spacings[0]) ** -2.0
        axes[1].loglog(spacings, guide, color=INK_SECONDARY, linewidth=0.8, linestyle=":", label="∝ N⁻²")
        axes[1].set_xlabel("每边间距数 N"); axes[1].set_ylabel("15 个内部点上的 L2")
        axes[1].set_title(f"剖面误差（实心 = crossing {CHOSEN_HDX:g}，空心圆点线 = crossing 4）")
        axes[1].set_xticks(spacings); axes[1].set_xticklabels([str(int(value)) for value in spacings])
        axes[1].minorticks_off()
        # one legend per panel: the colours name a different quantity in each panel
        axes[0].legend(loc="lower right", ncol=2, labelcolor=INK_PRIMARY)
        axes[1].legend(loc="upper right", ncol=2, labelcolor=INK_PRIMARY)
        height = figure.get_figheight()
        figure.suptitle("随分辨率的收敛", x=0.01, y=1.0 - 0.05 / height, ha="left", va="top", fontsize=12,
                        fontweight="bold", color=INK_PRIMARY)
        figure.tight_layout(rect=(0, 0, 1.0, 1.0 - 0.45 / height), w_pad=2.0)
        save(figure, FIGURES / "round8_convergence.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("old", "smoke", "timing", "accuracy"))
    arguments = parser.parse_args()
    if arguments.command == "old":
        old_tables()
    elif arguments.command == "smoke":
        smoke_tables()
    elif arguments.command == "timing":
        timing_tables()
    else:
        return accuracy_main(arguments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
