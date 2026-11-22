"""
wall_store_check.py - the wall store (branch v7-crossing-wall-store, claude/wall_store_report.md) against the particle-
pool version (git tag crossing-pool): the physical quantities of short runs of both builds.

    .venv/Scripts/python.exe -m experiment.wall_crossing.wall_store_check [--part all|cavity|hydrostatic|drift|cavity3d|
                                                                           checkpoint|dual]

The runs (each build's own tools, logs under logs/wall_store/{new,reference}/; the summation and its order differ, so
nothing is compared bit for bit):
  cavity       cavity_run.py, 2-D 250^2 cavity with the volume constraint, from rest to t = 10 (the average over 8-10):
               psi_min, u_min, v_max, v_min, the profile L2 against Marchi et al. (2021) (wall frame, MLS), the two
               builds' dense centre lines against each other (experiment/validation/cavity_analysis.analyse_run);
  hydrostatic  hydrostatic.py run, the 2-D boxes (gravity -y and tilted 30 degrees) to t = 10: residual speed, speed
               of the first layer, level-free pressure error, pressure level at the box centre;
  drift        cavity_run.py, the 250^2 cavity WITHOUT the volume constraint to t = 10: the mean density's slope
               (least squares over 2 <= t <= 10 of field_history.jsonl);
  cavity3d     box_run.py, the 3-D 1M cavity to t = 2.04: the kinetic energy and the three centre-line extrema at
               that time (profiles.json, Shepard average of the fluid), the mean density;
  checkpoint   box_run.py, the 250^2 cavity to t = 4 with a checkpoint at t = 2.01 and the same run resumed from it:
               theta, I (volume_constraint.jsonl) and the mean density at the end;
  dual         box_run.py / cavity_run.py with K = 2 against K = 1 (2-D 250^2 to t = 5, 3-D 1M to t = 2).
Writes claude/wall_store/tables/check_*.md and logs/wall_store/checks/physics.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

LOG_ROOT = REPOSITORY_ROOT / "logs" / "wall_store"
TABLE_ROOT = REPOSITORY_ROOT / "claude" / "wall_store" / "tables"
BUILDS = (("reference", "标签 crossing-pool"), ("new", "壁面独立存储"))


def markdown_table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def write_table(name: str, text: str) -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    (TABLE_ROOT / name).write_text(text, encoding="utf-8", newline="\n")
    print(f"[wall_store_check] wrote {TABLE_ROOT / name}")


def read_series(path: pathlib.Path) -> dict:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(row[key]) if row[key] not in ("", None) else np.nan for row in rows])
            for key in rows[0]}


def read_jsonl(path: pathlib.Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def relative(new: float, reference: float) -> str:
    return f"{100.0 * (new / reference - 1.0):+.3f} %" if reference else "—"


def cavity(result: dict) -> None:
    from experiment.validation.cavity_analysis import analyse_run, load_run
    names = {"reference": "ref_n250_t10", "new": "ws_n250_t10"}
    analyses = {}
    for build, _ in BUILDS:
        run = load_run(names[build], root=LOG_ROOT / build / "cavity")
        analyses[build] = analyse_run(run, average_span=2.0, tolerance=1.0e-3, window=10.0, with_stream_function=True)
    entries = {}
    rows = []
    for build, text in BUILDS:
        analysis = analyses[build]
        stream = analysis["stream"] or {}
        extrema = analysis["extrema"]
        errors = analysis["errors"]
        entries[build] = {"psi_min": stream.get("psi_min_wall"), "u_min": extrema["u_min"]["value"],
                          "v_max": extrema["v_max"]["value"], "v_min": extrema["v_min"]["value"],
                          "l2_u": errors[("wall", "mls", "u")]["L2"], "l2_v": errors[("wall", "mls", "v")]["L2"],
                          "window": analysis["window"], "samples": analysis["samples_in_window"]}
        entry = entries[build]
        rows.append([text, f"{entry['psi_min']:.6f}", f"{entry['u_min']:.5f}", f"{entry['v_max']:.5f}",
                     f"{entry['v_min']:.5f}", f"{entry['l2_u']:.5f}", f"{entry['l2_v']:.5f}",
                     f"{entry['window'][0]:.1f}–{entry['window'][1]:.1f}（{entry['samples']}）"])
    new, reference = entries["new"], entries["reference"]
    rows.append(["差（新 − 标签）",
                 f"{new['psi_min'] - reference['psi_min']:+.6f}（{relative(new['psi_min'], reference['psi_min'])}）",
                 *[f"{new[key] - reference[key]:+.5f}" for key in ("u_min", "v_max", "v_min", "l2_u", "l2_v")], ""])
    difference = {}
    for line in ("u", "v"):
        delta = (analyses["new"]["statistics"][f"{line}_dense_mls"]["mean"]
                 - analyses["reference"]["statistics"][f"{line}_dense_mls"]["mean"])
        difference[line] = {"l2": float(np.sqrt(np.mean(delta ** 2))), "linf": float(np.abs(delta).max())}
    entries["profile_difference"] = difference
    result["cavity"] = entries
    write_table("check_cavity_n250.md", markdown_table(
        ["2-D 250² 方腔（有总体积修正），t = 10（8–10 平均）", "ψ_min（wall 帧）", "u_min", "v_max", "v_min",
         "u(y) L2 对 Marchi", "v(x) L2 对 Marchi", "平均窗口（样本数）"], rows)
        + f"\n两个版本的密中心线剖面之差：u(y) L2 {difference['u']['l2']:.5f}、L∞ {difference['u']['linf']:.5f}；"
          f"v(x) L2 {difference['v']['l2']:.5f}、L∞ {difference['v']['linf']:.5f}（盖速为单位）。\n")


def hydrostatic(result: dict) -> None:
    names = {"reference": "ref_2d_n250_{}", "new": "ws_2d_n250_{}_b"}
    rows = []
    result["hydrostatic"] = {}
    for gravity, text in (("gy", "重力沿 −y"), ("g30", "重力倾斜 30°")):
        entries = {}
        for build, build_text in BUILDS:
            directory = LOG_ROOT / build / "hydrostatic" / names[build].format(gravity)
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            series = read_series(directory / "series.csv")
            final = summary["final"]
            entries[build] = {"speed_maximum_over_scale": final["speed_maximum_over_scale"],
                              "speed_root_mean_square_over_scale": final["speed_root_mean_square_over_scale"],
                              "first_layer_speed_maximum": final["layers"][0]["speed_maximum"],
                              "pressure_error_root_mean_square_over_scale":
                                  final["pressure_error_root_mean_square_over_scale"],
                              "pressure_level_at_centre": final["pressure_level_at_centre"],
                              "pressure_level_over_scale": final["pressure_level_over_scale"],
                              "density_mean": float(series["density_mean"][-1]) if "density_mean" in series
                              else float("nan"),
                              "time": summary["time"], "run_id": summary["run_id"]}
            entry = entries[build]
            rows.append([text, build_text, f"{entry['speed_maximum_over_scale']:.5f}",
                         f"{entry['speed_root_mean_square_over_scale']:.6f}",
                         f"{entry['first_layer_speed_maximum']:.4f}",
                         f"{entry['pressure_error_root_mean_square_over_scale']:.5f}",
                         f"{entry['pressure_level_at_centre']:+.1f}（{entry['pressure_level_over_scale']:+.5f}）",
                         f"{entry['density_mean'] - 1000.0:+.5f}"])
        result["hydrostatic"][gravity] = entries
    write_table("check_hydrostatic.md", markdown_table(
        ["静水盒（2-D 250²，t = 10）", "版本", "最大速度 / √(gH)", "均方根速度 / √(gH)", "离壁第一层的最大速度（m/s）",
         "压力误差均方根 / ρ0gH（去掉水平）", "盒中心的平均压力 Pa（/ ρ0gH）", "平均密度 − ρ0（kg/m³）"], rows))


def drift(result: dict) -> None:
    names = {"reference": "ref_n250_no_constraint_t10", "new": "ws_n250_no_constraint_t10"}
    rows = []
    result["drift"] = {}
    for build, text in BUILDS:
        history = read_jsonl(LOG_ROOT / build / "cavity" / names[build] / "field_history.jsonl")
        times = np.array([row["time"] for row in history])
        means = np.array([row["density_mean"] for row in history])
        chosen = times >= 2.0 - 1e-9
        slope, intercept = np.polyfit(times[chosen], means[chosen], 1)
        entry = {"slope_per_time": float(slope), "relative_slope_per_time": float(slope / 1000.0),
                 "density_mean_end": float(means[-1]), "time_end": float(times[-1]),
                 "fit_window": [float(times[chosen].min()), float(times[chosen].max())], "samples": int(chosen.sum())}
        result["drift"][build] = entry
        rows.append([text, f"{entry['slope_per_time']:+.3e}", f"{entry['relative_slope_per_time']:+.3e}",
                     f"{entry['density_mean_end'] - 1000.0:+.5f}",
                     f"{entry['fit_window'][0]:.2f}–{entry['fit_window'][1]:.2f}（{entry['samples']}）"])
    write_table("check_drift.md", markdown_table(
        ["2-D 250² 方腔（无总体积修正），到 t = 10", "平均密度的漂移率（kg/m³ 每单位时间）", "相对漂移率（每单位时间）",
         "t = 10 的平均密度 − ρ0", "拟合窗口（样本数）"], rows))


def profile_extrema(entry: dict) -> dict:
    vertical = entry["u_along_y"]
    horizontal = entry["v_along_x"]
    u = np.asarray(vertical["velocity"], dtype=np.float64)
    v = np.asarray(horizontal["velocity"], dtype=np.float64)
    return {"u_min": float(np.nanmin(u)), "v_max": float(np.nanmax(v)), "v_min": float(np.nanmin(v)),
            "u": u, "v": v}


def cavity3d(result: dict) -> None:
    names = {"reference": "ref_cavity3d_1m_t2", "new": "ws_cavity3d_1m_t2"}
    entries = {}
    rows = []
    for build, text in BUILDS:
        directory = LOG_ROOT / build / "box_run" / names[build]
        series = read_series(directory / "series.csv")
        profiles = json.loads((directory / "profiles.json").read_text(encoding="utf-8"))
        last = profiles[-1]
        extrema = profile_extrema(last)
        index = int(np.argmin(np.abs(series["time"] - last["time"])))
        entries[build] = {"time": float(last["time"]), "kinetic_energy": float(series["kinetic_energy"][index]),
                          "density_mean": float(series["density_mean"][index]),
                          "speed_maximum": float(series["speed_maximum"][index]),
                          "u_min": extrema["u_min"], "v_max": extrema["v_max"], "v_min": extrema["v_min"],
                          "profile_u": extrema["u"], "profile_v": extrema["v"],
                          "steps_per_second": float(np.nanmedian(series["steps_per_second"]))}
        entry = entries[build]
        rows.append([text, f"{entry['time']:.2f}", f"{entry['kinetic_energy']:.5f}", f"{entry['u_min']:.5f}",
                     f"{entry['v_max']:.5f}", f"{entry['v_min']:.5f}", f"{entry['density_mean'] - 1000.0:+.5f}"])
    new, reference = entries["new"], entries["reference"]
    rows.append(["差（新 − 标签）", "", f"{new['kinetic_energy'] - reference['kinetic_energy']:+.5f}"
                 f"（{relative(new['kinetic_energy'], reference['kinetic_energy'])}）",
                 *[f"{new[key] - reference[key]:+.5f}" for key in ("u_min", "v_max", "v_min")],
                 f"{new['density_mean'] - reference['density_mean']:+.5f}"])
    difference = {line: float(np.sqrt(np.nanmean((new[f"profile_{line}"] - reference[f"profile_{line}"]) ** 2)))
                  for line in ("u", "v")}
    result["cavity3d"] = {build: {key: value for key, value in entry.items() if not key.startswith("profile_")}
                          for build, entry in entries.items()}
    result["cavity3d"]["profile_difference_l2"] = difference
    write_table("check_cavity3d.md", markdown_table(
        ["3-D 1M 方腔（有总体积修正），单卡", "时刻", "动能（J）", "u_min（竖直中心线）", "v_max（水平中心线）", "v_min",
         "平均密度 − ρ0"], rows)
        + f"\n两个版本 101 点中心线剖面之差（同一时刻）：u(y) L2 {difference['u']:.5f}，v(x) L2 {difference['v']:.5f}"
          "（盖速为单位）。\n")


def constraint_records(path: pathlib.Path) -> list:
    """volume_constraint.jsonl with absolute steps and times: a resumed segment's records count from 0 after its
    marker line {"segment", "segment_start_step"} (volume_constraint_v7: the steps of the process)."""
    records, offset = [], 0
    for record in read_jsonl(path):
        if "segment_start_step" in record:
            offset = int(record["segment_start_step"])
            continue
        step = offset + int(record["step"])
        timestep = record["time"] / record["step"] if record["step"] else None
        records.append({**record, "absolute_step": step, "absolute_time": None if timestep is None else step * timestep})
    if records and records[0]["absolute_time"] is None:
        records[0]["absolute_time"] = 0.0
    return records


def checkpoint(result: dict) -> None:
    rows = []
    entries = {}
    for build, build_text, name in (("new", "壁面独立存储", "ws_checkpoint_n250"),
                                    ("reference", "标签 crossing-pool", "ref_checkpoint_n250")):
        directory = LOG_ROOT / build / "box_run" / name
        if not directory.exists():
            continue
        for label, series_name, constraint_name, text in (
                ("continuous", "series_continuous.csv", "volume_constraint_continuous.jsonl", "连续运行到 t = 4"),
                ("resumed", "series.csv", "volume_constraint.jsonl", "t = 2.01 的检查点续跑到 t = 4")):
            series = read_series(directory / series_name)
            records = constraint_records(directory / constraint_name)
            last = records[-1]
            entry = {"time": float(series["time"][-1]), "density_mean": float(series["density_mean"][-1]),
                     "kinetic_energy": float(series["kinetic_energy"][-1]), "theta": float(last["theta"]),
                     "integral": float(last["integral"]), "constraint_time": float(last["absolute_time"])}
            entries[f"{build}_{label}"] = entry
            rows.append([build_text, text, f"{entry['time']:.2f}", f"{entry['constraint_time']:.2f}",
                         f"{entry['theta']:+.4e}", f"{entry['integral']:+.4e}",
                         f"{entry['density_mean'] - 1000.0:+.6f}", f"{entry['kinetic_energy']:.5f}"])
        continuous, resumed = entries[f"{build}_continuous"], entries[f"{build}_resumed"]
        rows.append([build_text, "差（续跑 − 连续）", "", "", f"{resumed['theta'] - continuous['theta']:+.2e}",
                     f"{resumed['integral'] - continuous['integral']:+.2e}",
                     f"{resumed['density_mean'] - continuous['density_mean']:+.6f}",
                     relative(resumed["kinetic_energy"], continuous["kinetic_energy"])])
    result["checkpoint"] = entries
    old = LOG_ROOT / "checks" / "old_checkpoint_refusal.txt"
    write_table("check_checkpoint.md", markdown_table(
        ["2-D 250² 方腔", "运行", "末时刻", "最后一次体积修正的时刻", "θ", "I", "平均密度 − ρ0", "动能"], rows)
        + ("\n旧检查点的报错（标签 crossing-pool 存的检查点；上：box_run --resume，下：直接交给 "
           "SphSimulatorV7.restart_init）：\n\n```\n"
           + old.read_text(encoding="utf-8").strip() + "\n```\n" if old.exists() else ""))


def dual(result: dict) -> None:
    rows = []
    result["dual"] = {}
    for label, build, names, text in (
            ("n250", "new", ("ws_dual_n250_k1", "ws_dual_n250_k2"), "2-D 250²，t = 5"),
            ("n250_reference", "reference", ("ref_dual_n250_k1", "ref_dual_n250_k2"), "2-D 250²，t = 5，标签（对照）"),
            ("cavity3d", "new", ("ws_dual_cavity3d_1m_k1", "ws_dual_cavity3d_1m_k2"), "3-D 1M，t = 2.04")):
        entries = {}
        for slabs, name in zip((1, 2), names):
            directory = LOG_ROOT / build / "box_run" / name
            if not directory.exists():
                continue
            series = read_series(directory / "series.csv")
            profiles = json.loads((directory / "profiles.json").read_text(encoding="utf-8"))
            extrema = profile_extrema(profiles[-1])
            entries[slabs] = {"time": float(series["time"][-1]), "kinetic_energy": float(series["kinetic_energy"][-1]),
                              "density_mean": float(series["density_mean"][-1]), "u_min": extrema["u_min"],
                              "v_max": extrema["v_max"], "v_min": extrema["v_min"], "u": extrema["u"],
                              "v": extrema["v"], "alive": float(series["alive"][-1]),
                              "overflow": float(np.nansum(series["overflow"]))}
        if len(entries) < 2:
            continue
        one, two = entries[1], entries[2]
        difference = {line: float(np.sqrt(np.nanmean((two[line] - one[line]) ** 2))) for line in ("u", "v")}
        result["dual"][label] = {slabs: {key: value for key, value in entry.items() if key not in ("u", "v")}
                                 for slabs, entry in entries.items()}
        result["dual"][label]["profile_difference_l2"] = difference
        for slabs in (1, 2):
            entry = entries[slabs]
            rows.append([text, f"K = {slabs}", f"{entry['time']:.2f}", f"{entry['kinetic_energy']:.6f}",
                         f"{entry['u_min']:.5f}", f"{entry['v_max']:.5f}", f"{entry['v_min']:.5f}",
                         f"{entry['density_mean'] - 1000.0:+.6f}", f"{entry['alive']:.0f}", f"{entry['overflow']:.0f}"])
        rows.append([text, "差（K = 2 − K = 1）", "",
                     f"{two['kinetic_energy'] - one['kinetic_energy']:+.2e}"
                     f"（{relative(two['kinetic_energy'], one['kinetic_energy'])}）",
                     *[f"{two[key] - one[key]:+.2e}" for key in ("u_min", "v_max", "v_min")],
                     f"{two['density_mean'] - one['density_mean']:+.2e}",
                     f"剖面差 L2 u {difference['u']:.1e}、v {difference['v']:.1e}", ""])
    write_table("check_dual.md", markdown_table(
        ["算例", "卡数", "时刻", "动能", "u_min", "v_max", "v_min", "平均密度 − ρ0", "粒子数", "溢出"], rows))


def theoretical_occupancy(registers: int, workgroup_warps: int = 4, warps_per_sm: int = 48,
                          registers_per_sm: int = 65536) -> float:
    """GB202: registers allocated per thread in steps of 8 (256 per warp), whole workgroups of 4 warps."""
    allocated = ((registers + 7) // 8) * 8
    warps = min(warps_per_sm, registers_per_sm // (allocated * 32))
    return (warps // workgroup_warps) * workgroup_warps / warps_per_sm


def registers(result: dict) -> None:
    """Register counts (VK_KHR_pipeline_executable_properties, pipeline_statistics.py) of the passes' pipelines:
    simple, the tag's (split dispatch, its default), the wall store's (A) and (B)."""
    directory = LOG_ROOT / "statistics"
    rows = []
    result["registers"] = {}
    for size, case_name, text in (("2d", "lid_driven_cavity_2d_n250_xi0p001_eps0p0025", "2-D"),
                                  ("3d", "cavity3d_1m", "3-D")):
        sources = (("simple", f"{case_name}_simple.json", (("correction_density_all", "融合遍"),
                                                              ("force_all", "受力"))),
                   ("pool", f"{case_name}_pool.json",
                    (("correction_density_all_split_interior", "融合遍 内部"),
                     ("correction_density_all_split_near_wall", "融合遍 近壁"),
                     ("force_all_split_interior", "受力 内部"), ("force_all_split_near_wall", "受力 近壁"))),
                   ("store_a", f"{case_name}_crossing_org0.json", (("correction_density_all", "融合遍（一条管线）"),
                                                                   ("force_all", "受力（一条管线）"))),
                   ("store_b", f"{case_name}_crossing_org1.json",
                    (("correction_density_all_split_interior", "融合遍 内部（类 1）"),
                     ("correction_density_all_split_near_wall", "融合遍 近壁（类 2）"),
                     ("force_all_split_interior", "受力 内部（类 1）"),
                     ("force_all_split_near_wall", "受力 近壁（类 2）"))))
        labels = {"simple": "simple", "pool": "标签 crossing-pool", "store_a": "壁面独立存储 (A)",
                  "store_b": "壁面独立存储 (B)"}
        for build, file_name, pipelines in sources:
            path = directory / file_name
            if not path.exists():
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            for key, pipeline_text in pipelines:
                count = report["pipelines"][key][0]["statistics"]["Register Count"]
                occupancy = theoretical_occupancy(int(count))
                result["registers"][f"{size}_{build}_{key}"] = {"registers": count, "occupancy": occupancy}
                rows.append([text, labels[build], pipeline_text, f"{count}", f"{100 * occupancy:.1f} %"])
    write_table("check_registers.md", markdown_table(
        ["维数", "版本", "管线", "每线程寄存器", "理论占用率"], rows))


def timing(result: dict) -> None:
    """Whole-step timing tables: single GPU (kernel_timing.py, V7_BENCH_TICK_DETAIL=step, logs/wall_store/
    timing_single_r2) and two GPUs (dual_timing.py steady rate and anatomy, logs/wall_store/timing_dual)."""
    single_labels = (("pool", "标签 crossing-pool（(B)，近壁先）"), ("store_a", "壁面独立存储 (A)"),
                     ("store_b", "壁面独立存储 (B)，内部先"),
                     ("store_b_near_wall_first", "壁面独立存储 (B)，近壁先"))
    rows = []
    result["timing_single"] = {}
    for size, text in (("n250", "2-D 250²"), ("n500", "2-D 500²"), ("n1000", "2-D 1000²"), ("c3d", "3-D 1M")):
        summary = json.loads((LOG_ROOT / "timing_single_r2" / size / "summary.json").read_text(encoding="utf-8"))
        labels = summary["labels"]
        simple = labels["simple"]["kernels"]["step_total_us"]
        entry = {"simple": simple["median_us"]}
        cells = [text, f"{simple['median_us']:.1f}（{simple['minimum_us']:.1f}–{simple['maximum_us']:.1f}）"]
        for label, _ in single_labels:
            value = labels[label]["kernels"]["step_total_us"]
            entry[label] = value["median_us"]
            cells.append(f"{value['median_us']:.1f}（{value['median_us'] / simple['median_us']:.3f}）")
        default = entry["store_b_near_wall_first"] if size != "c3d" else entry["store_b"]
        entry["default"] = default
        cells.append(f"{100 * (default / entry['pool'] - 1):+.1f} %")
        result["timing_single"][size] = entry
        rows.append(cells)
    single_table = markdown_table(
        ["单卡（无显示器的 5090）", "simple 每步 µs（三轮范围）"]
        + [f"{text} µs（/ simple）" for _, text in single_labels] + ["缺省（2-D 近壁先，3-D 内部先）对标签"], rows)
    dual_labels = (("pool", "标签 crossing-pool"), ("store_mode1", "壁面独立存储 (B)，分界带方式 1"),
                   ("store_mode2", "壁面独立存储 (B)，分界带方式 2（缺省）"))
    rows = []
    result["timing_dual"] = {}
    for size, text in (("c3d", "3-D 1M"), ("n1000", "2-D 1000²")):
        summary = json.loads((LOG_ROOT / "timing_dual" / f"{size}_steady" / "summary.json").read_text(encoding="utf-8"))
        labels = summary["labels"]
        simple = labels["simple"]["steady_fps"]
        entry = {"simple": simple["median"]}
        cells = [text, f"{simple['median']:.1f}（{simple['minimum']:.1f}–{simple['maximum']:.1f}）"]
        for label, _ in dual_labels:
            value = labels[label]["steady_fps"]
            entry[label] = value["median"]
            cells.append(f"{value['median']:.1f}（{simple['median'] / value['median']:.3f}）")
        cells.append(f"{100 * (entry['pool'] / entry['store_mode2'] - 1):+.1f} %")
        result["timing_dual"][size] = entry
        rows.append(cells)
    dual_table = markdown_table(
        ["双卡（两张 5090），稳态", "simple 步/秒（三轮范围）"]
        + [f"{text} 步/秒（每步时间 / simple）" for _, text in dual_labels] + ["缺省对标签的每步时间"], rows)
    rows = []
    result["timing_anatomy"] = {}
    for size, text in (("c3d", "3-D 1M"), ("n1000", "2-D 1000²")):
        summary = json.loads((LOG_ROOT / "timing_dual" / f"{size}_anatomy" / "summary.json").read_text(encoding="utf-8"))
        for label, label_text in (("simple", "simple"),) + dual_labels:
            slabs = summary["labels"][label]["slabs"]
            values = {slab: {key: slabs[slab][key]["median_us"] for key in ("phase_a", "phase_b", "phase_c",
                                                                            "b_to_c_gap")}
                      for slab in ("0", "1")}
            result["timing_anatomy"][f"{size}_{label}"] = values
            rows.append([text, label_text] + [f"{values['0'][key]:.0f} / {values['1'][key]:.0f}"
                                               for key in ("phase_a", "phase_b", "phase_c", "b_to_c_gap")])
    anatomy_table = markdown_table(["双卡分相计时（µs，0 号卡 / 1 号卡）", "版本", "phase A", "phase B", "phase C",
                                    "B→C 等待"], rows)
    write_table("timing_single.md", single_table)
    write_table("timing_dual.md", dual_table)
    write_table("timing_dual_anatomy.md", anatomy_table)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", default="all",
                        choices=("all", "cavity", "hydrostatic", "drift", "cavity3d", "checkpoint", "dual",
                                 "registers", "timing"))
    arguments = parser.parse_args()
    out = LOG_ROOT / "checks" / "physics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    result = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    parts = {"cavity": cavity, "hydrostatic": hydrostatic, "drift": drift, "cavity3d": cavity3d,
             "checkpoint": checkpoint, "dual": dual, "registers": registers,
             "timing": timing}
    for name, function in parts.items():
        if arguments.part in ("all", name):
            function(result)
    out.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
