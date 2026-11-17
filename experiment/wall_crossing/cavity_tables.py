"""
cavity_tables.py - tables of the Re = 1000 lid-driven cavity runs of logs/wall_crossing/cavity against
Marchi et al. 2021 (docs/validation/data/marchi2021_re1000.csv), K = 1, every wall mode side by side.

    .venv/Scripts/python.exe -m experiment.wall_crossing.cavity_tables [--root logs/wall_crossing/cavity] \
        [--out claude/wall_crossing/cavity] [--runs name,name] [--no-stream-function]

The analysis of one run is experiment/validation/cavity_analysis.analyse_run (the published protocol of
docs/validation/cavity_re1000.md: average over the last 20 time units, MLS samples, extrema from the dense
centre lines with a parabola, psi_min from the snapshots on a 513^2 grid). That module's own tables only read
K = 2 runs of logs/validation/cavity_re1000; this driver reads the runs of this work (names
n<resolution>_<mode>) and writes:
  - per run: psi_min, u_min, v_max, v_min with their deviation from the reference and their positions, the L2 and
    L-infinity errors of the two centre-line profiles (28 points each);
  - per mode: the observed order of every error over the available resolutions (pairwise and least squares).

Frames. The benchmark's unit square has to be placed on the particles. Two placements are reported:
  wall  the centre lines of the innermost wall rows (half width 0.5 + dx): the frame of the published tables,
        used for simple and adami, kept here for every mode so the numbers compare like for like;
  mid   half way between the innermost wall row and the outermost fluid row (half width 0.5 + dx / 2): for
        wall_boundary crossing this is the box, the plane on which the mode imposes the wall condition.
Velocities are in units of the lid speed in both; psi scales with the frame length (psi_mid = psi_wall
(1 + 2 dx) / (1 + dx): the stream function between the wall row and the mid plane is of order dx^2 and is
neglected); positions are converted. The profile errors of the mid frame come from the time-averaged dense
lines (cubic spline), cavity_analysis' own "mid" frame.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import yaml

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.validation import cavity_analysis, cavity_reference      # noqa: E402
from experiment.validation.cavity_tables import fit_order, order          # noqa: E402

RESOLUTIONS = (250, 500, 1000)
MODES = ("simple", "adami", "crossing", "crossing_wall_velocity", "crossing_floor0p5")
MODE_TEXT = {"simple": "simple", "adami": "adami", "crossing": "crossing (extrapolate)",
             "crossing_wall_velocity": "crossing (wall_velocity)", "crossing_floor0p5": "crossing (floor 0.5)"}
AVERAGE_SPAN = 20.0
STEADY_TOLERANCE = 1.0e-3
WINDOW = 10.0


def deviation_percent(value: float, reference: float) -> float:
    """100 (|value| / |reference| - 1): negative = weaker than the reference."""
    return 100.0 * (abs(value) / abs(reference) - 1.0)


def run_diagnostics(run: dict) -> dict:
    """What the samples and the crossing status file say about the density and the walls over the whole run:
    the fluid's density range at the first and last sample, the rate of its mid-range over the second half of
    the run (the secular drift of the level), the largest deviation from rho0 and the largest speed at a sample;
    for a crossing run (wall_crossing_status.jsonl, written by cavity_runner_entry): the cumulative A_f fallback
    count, the smallest fluid wall distance and how many fluid particles were below the floor or outside the box
    at any position readback."""
    rows = run["rows"]
    times = np.array([row["time"] for row in rows])
    low = np.array([row["density_min"] for row in rows])
    high = np.array([row["density_max"] for row in rows])
    middle = 0.5 * (low + high)
    half = times >= 0.5 * times.max()
    rate = float(np.polyfit(times[half], middle[half], 1)[0]) if half.sum() >= 3 else float("nan")
    case_copy = yaml.safe_load((run["dir"] / "case.yaml").read_text(encoding="utf-8"))
    speed_of_sound = float(case_copy["physics"]["speed_of_sound"])
    rest_density = 1000.0           # stick_water / stick_wall of materials/standard.yaml (the cavity series)
    diagnostics = {
        "density_range_first": [float(low[0]), float(high[0])], "density_range_last": [float(low[-1]), float(high[-1])],
        "density_mid_range_rate_per_time": rate,
        "density_deviation_maximum": float(max(np.abs(low - rest_density).max(), np.abs(high - rest_density).max())),
        # c0^2 (rho_mid - rho0) / (rho0 U^2) with U = 1 (the lid speed the runner checks through Re = 1000)
        "pressure_level_last_over_rho_u2": float((middle[-1] - rest_density) * speed_of_sound ** 2 / rest_density),
        "speed_maximum": float(max(row["max_speed"] for row in rows)),
        "samples": len(rows), "t_last": float(times.max())}
    status_path = run["dir"] / "wall_crossing_status.jsonl"
    if status_path.exists():
        records = [json.loads(line) for line in status_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        with_distance = [record for record in records if "wall_distance_minimum" in record]
        spacing = float(run["spacing"])
        # the counter restarts with every segment (a resumed run starts a new process): sum the segments' maxima
        segment_totals, previous = [], -1
        for record in records:
            if record["frame_stamp"] < previous:
                segment_totals.append(0)
            if not segment_totals:
                segment_totals.append(0)
            segment_totals[-1] = max(segment_totals[-1], record["gradient_fallbacks"])
            previous = record["frame_stamp"]
        diagnostics["crossing"] = {
            "status_records": len(records), "position_readbacks": len(with_distance),
            "gradient_fallbacks": int(sum(segment_totals)),
            "correction_fallbacks": int(max((record["correction_fallbacks"] for record in records), default=0)),
            "wall_distance_minimum_over_spacing": (min(record["wall_distance_minimum"] for record in with_distance)
                                                   / spacing) if with_distance else None,
            "below_floor_maximum": max((record["below_floor_count"] for record in with_distance), default=None),
            "outside_box_maximum": max((record["outside_box_count"] for record in with_distance), default=None)}
    return diagnostics


def analyse(name: str, root: pathlib.Path, cache_directory: pathlib.Path, with_stream_function: bool):
    """The numbers of one run (None when it has no samples yet). Cached per run by its sample count."""
    run = cavity_analysis.load_run(name, root)
    if run is None:
        return None
    complete = bool(run["result"])
    diagnostics = run_diagnostics(run)
    last_segment = run["segments"][-1] if run["segments"] else {}
    if not complete and last_segment.get("status") == "error":
        # the run ended in an error (an invariant violation: particles lost): no steady state to analyse
        return {"name": name, "case": run["case"], "resolution": int(run["resolution"]), "failed": True,
                "complete": False, "error": last_segment.get("error"), "t_last": float(run["times"].max()),
                "git": run["meta"].get("git"), "diagnostics": diagnostics,
                "device_uuids": sorted({uuid for segment in run["segments"]
                                        for uuid in (segment.get("device_uuids") or [])})}
    cache_path = cache_directory / f"{name}.json"
    signature = {"version": 2, "samples": len(run["rows"]), "t_last": float(run["times"].max()), "complete": complete,
                 "stream": with_stream_function}
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("signature") == signature:
            return cached
    analysis = cavity_analysis.analyse_run(run, AVERAGE_SPAN, STEADY_TOLERANCE, WINDOW, with_stream_function)
    reference = cavity_reference.marchi2021()["extrema"]
    spacing = float(run["spacing"])
    wall_length, mid_length = 1.0 + 2.0 * spacing, 1.0 + spacing

    def to_mid(position_wall: float) -> float:
        physical = position_wall * wall_length - (0.5 + spacing)
        return (physical + 0.5 + 0.5 * spacing) / mid_length

    record = {
        "signature": signature, "name": name, "case": run["case"], "resolution": int(run["resolution"]),
        "spacing": spacing, "wall_boundary": run["meta"].get("wall_boundary"),
        "git": run["meta"].get("git"), "complete": complete,
        "window": list(analysis["window"]), "samples_in_window": analysis["samples_in_window"],
        "t_steady_online": analysis["t_steady_online"], "t_steady_sliding": analysis["t_steady_sliding"],
        "wall_hours": sum(float(segment.get("wall_s", 0.0)) for segment in run["segments"]) / 3600.0,
        "device_uuids": sorted({uuid for segment in run["segments"] for uuid in (segment.get("device_uuids") or [])}),
        "kinetic_energy": float(analysis["kinetic_energy_window"]["mean"][0]),
        "failed": False, "diagnostics": diagnostics,
        "extrema": {}, "errors": {},
    }
    for key, position_key in (("u_min", "y_at_u_min"), ("v_max", "x_at_v_max"), ("v_min", "x_at_v_min")):
        entry = analysis["extrema"][key]
        record["extrema"][key] = {
            "value": entry["value"], "sem": entry["sem"],
            "deviation_percent": deviation_percent(entry["value"], reference[key][0]),
            "position_wall": entry["position_wall"], "position_mid": to_mid(entry["position_wall"]),
            "position_reference": reference[position_key][0]}
    stream = analysis.get("stream")
    if stream:
        psi_wall = stream["psi_min_wall"]
        psi_mid = psi_wall * wall_length / mid_length
        record["extrema"]["psi_min"] = {
            "snapshots": stream["snapshots"],
            "value_wall": psi_wall, "deviation_wall_percent": deviation_percent(psi_wall, reference["psi_min"][0]),
            "value_mid": psi_mid, "deviation_mid_percent": deviation_percent(psi_mid, reference["psi_min"][0]),
            "value_fluid": stream["psi_min_fluid"],
            "deviation_fluid_percent": deviation_percent(stream["psi_min_fluid"], reference["psi_min"][0]),
            "x_wall": stream["x_wall"], "y_wall": stream["y_wall"],
            "x_mid": to_mid(stream["x_wall"]), "y_mid": to_mid(stream["y_wall"]),
            "x_reference": reference["x_at_psi_min"][0], "y_reference": reference["y_at_psi_min"][0]}
    for frame, method in (("wall", "mls"), ("mid", "dense"), ("fluid", "mls")):
        for line in ("u", "v"):
            error = analysis["errors"][(frame, method, line)]
            record["errors"][f"{frame}_{line}"] = {"L2": error["L2"], "Linf": error["Linf"],
                                                   "L2_relative": error["L2_relative"]}
    cache_directory.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(record, indent=1), encoding="utf-8", newline="\n")
    return record


def markdown_table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="logs/wall_crossing/cavity")
    parser.add_argument("--out", default="claude/wall_crossing/cavity")
    parser.add_argument("--runs", default=None, help="comma-separated run names (default: every n<res>_<mode>)")
    parser.add_argument("--no-stream-function", action="store_true")
    arguments = parser.parse_args()

    root = (REPOSITORY_ROOT / arguments.root).resolve()
    out = (REPOSITORY_ROOT / arguments.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    names = arguments.runs.split(",") if arguments.runs else [
        f"n{resolution}_{mode}" for mode in MODES for resolution in RESOLUTIONS]
    records = {}
    for name in names:
        record = analyse(name, root, out / "analysis", not arguments.no_stream_function)
        if record is None:
            continue
        records[name] = record
        if record["failed"]:
            print(f"[cavity_tables] {name}: FAILED at t = {record['t_last']:.2f}: {record['error']}", flush=True)
        else:
            print(f"[cavity_tables] {name}: window {record['window'][0]:.2f}-{record['window'][1]:.2f}, "
                  f"{'complete' if record['complete'] else 'IN PROGRESS'}", flush=True)
    if not records:
        print("[cavity_tables] no run with samples")
        return 1
    reference = cavity_reference.marchi2021()["extrema"]

    def label(record: dict) -> str:
        mode = record["name"].split("_", 1)[1]
        state = "（失败）" if record["failed"] else ("" if record["complete"] else "（运行中）")
        return f"{record['resolution']}² {MODE_TEXT.get(mode, mode)}{state}"

    def sort_key(record: dict) -> tuple:
        mode = record["name"].split("_", 1)[1]
        return (MODES.index(mode) if mode in MODES else 99, record["resolution"])

    everything = sorted(records.values(), key=sort_key)
    ordered = [record for record in everything if not record["failed"]]
    failed = [record for record in everything if record["failed"]]
    sections = []
    rows = []
    for record in ordered:
        psi = record["extrema"].get("psi_min")
        extrema = record["extrema"]
        rows.append([label(record), record["name"],
                     f"{record['window'][0]:.1f}–{record['window'][1]:.1f}",
                     "—" if record["t_steady_online"] is None else f"{record['t_steady_online']:.1f}",
                     "—" if not psi else f"{psi['value_wall']:.5f} ({psi['deviation_wall_percent']:+.2f} %)",
                     "—" if not psi else f"{psi['value_mid']:.5f} ({psi['deviation_mid_percent']:+.2f} %)",
                     *[f"{extrema[key]['value']:.5f} ({extrema[key]['deviation_percent']:+.2f} %)"
                       for key in ("u_min", "v_max", "v_min")]])
    sections.append("### 极值与 ψ_min（括号内为相对 Marchi 2021 的偏差，负 = 偏弱）\n\n"
                    f"参考值：ψ_min {reference['psi_min'][0]:.5f}，u_min {reference['u_min'][0]:.5f}，"
                    f"v_max {reference['v_max'][0]:.5f}，v_min {reference['v_min'][0]:.5f}。\n\n"
                    + markdown_table(["运行", "run 目录", "平均窗口 t", "定常 t_s", "ψ_min（wall 帧）", "ψ_min（mid 帧）",
                                      "u_min", "v_max", "v_min"], rows))
    rows = []
    for record in ordered:
        errors = record["errors"]
        rows.append([label(record)]
                    + [f"{errors[f'wall_{line}'][norm]:.4f}" for line in ("u", "v") for norm in ("L2", "Linf")]
                    + [f"{100.0 * errors[f'wall_{line}']['L2_relative']:.2f} %" for line in ("u", "v")]
                    + [f"{errors[f'mid_{line}'][norm]:.4f}" for line in ("u", "v") for norm in ("L2", "Linf")]
                    + [f"{100.0 * errors[f'mid_{line}']['L2_relative']:.2f} %" for line in ("u", "v")])
    sections.append("### 中线剖面误差（28 个 Marchi 点；速度以盖速为单位）\n\n"
                    + markdown_table(["运行", "u L2 (wall)", "u L∞ (wall)", "v L2 (wall)", "v L∞ (wall)",
                                      "u 相对 L2 (wall)", "v 相对 L2 (wall)", "u L2 (mid)", "u L∞ (mid)",
                                      "v L2 (mid)", "v L∞ (mid)", "u 相对 L2 (mid)", "v 相对 L2 (mid)"], rows))
    rows = []
    for record in ordered:
        extrema = record["extrema"]
        psi = extrema.get("psi_min")
        rows.append([label(record)]
                    + [f"{extrema[key]['position_wall']:.4f} / {extrema[key]['position_mid']:.4f} "
                       f"({extrema[key]['position_mid'] - extrema[key]['position_reference']:+.4f})"
                       for key in ("u_min", "v_max", "v_min")]
                    + (["—"] if not psi else
                       [f"({psi['x_wall']:.4f}, {psi['y_wall']:.4f}) / ({psi['x_mid']:.4f}, {psi['y_mid']:.4f})"]))
    sections.append("### 极值位置（wall 帧 / mid 帧；括号内为 mid 帧位置减参考位置）\n\n"
                    f"参考位置：y(u_min) {reference['y_at_u_min'][0]:.4f}，x(v_max) {reference['x_at_v_max'][0]:.4f}，"
                    f"x(v_min) {reference['x_at_v_min'][0]:.4f}，ψ_min ({reference['x_at_psi_min'][0]:.4f}, "
                    f"{reference['y_at_psi_min'][0]:.4f})。\n\n"
                    + markdown_table(["运行", "y(u_min)", "x(v_max)", "x(v_min)", "ψ_min 位置"], rows))

    # ---- observed order per mode over the complete resolutions
    convergence = {}
    rows = []
    for mode in MODES:
        series = [records.get(f"n{resolution}_{mode}") for resolution in RESOLUTIONS]
        series = [record for record in series if record is not None and record["complete"] and not record["failed"]]
        if len(series) < 2:
            continue
        spacings = [record["spacing"] for record in series]
        quantities = {}
        for frame in ("wall", "mid"):
            for line in ("u", "v"):
                for norm in ("L2", "Linf"):
                    quantities[f"{line} {norm} ({frame})"] = [record["errors"][f"{frame}_{line}"][norm]
                                                               for record in series]
        for key in ("u_min", "v_max", "v_min"):
            quantities[f"|{key} 偏差|"] = [abs(record["extrema"][key]["value"] - reference[key][0])
                                         for record in series]
        if all(record["extrema"].get("psi_min") for record in series):
            for frame in ("wall", "mid"):
                quantities[f"|ψ_min 偏差| ({frame})"] = [
                    abs(record["extrema"]["psi_min"][f"value_{frame}"] - reference["psi_min"][0])
                    for record in series]
        convergence[mode] = {"resolutions": [record["resolution"] for record in series], "quantities": {}}
        for quantity, values in quantities.items():
            pairwise = order(values, spacings)
            fitted = fit_order(values, spacings)
            convergence[mode]["quantities"][quantity] = {"values": values, "pairwise": pairwise, "fit": fitted}
            rows.append([MODE_TEXT[mode], quantity, " → ".join(f"{value:.5f}" for value in values),
                         ", ".join("—" if math.isnan(value) else f"{value:.2f}" for value in pairwise),
                         "—" if math.isnan(fitted) or len(values) < 3 else f"{fitted:.2f}"])
    if rows:
        sections.append("### 观测收敛阶（p = ln(e₁/e₂)/ln(dx₁/dx₂)；三点时另给最小二乘拟合）\n\n"
                        + markdown_table(["壁面", "量", "误差（分辨率由低到高）", "相邻两档的阶", "拟合阶"], rows))

    # ---- density and wall diagnostics of every run (failed ones included)
    rows = []
    for record in everything:
        diagnostics = record["diagnostics"]
        crossing = diagnostics.get("crossing")
        rows.append([
            label(record), f"{diagnostics['t_last']:.1f}",
            f"{diagnostics['density_range_first'][0]:.3f} … {diagnostics['density_range_first'][1]:.3f}",
            f"{diagnostics['density_range_last'][0]:.3f} … {diagnostics['density_range_last'][1]:.3f}",
            f"{diagnostics['density_mid_range_rate_per_time']:+.5f}",
            f"{diagnostics['density_deviation_maximum']:.3f}",
            f"{diagnostics['pressure_level_last_over_rho_u2']:+.2f}",
            f"{diagnostics['speed_maximum']:.3f}",
            "—" if not crossing else str(crossing["gradient_fallbacks"]),
            "—" if not crossing or crossing["wall_distance_minimum_over_spacing"] is None
            else f"{crossing['wall_distance_minimum_over_spacing']:.3f}",
            "—" if not crossing or crossing["below_floor_maximum"] is None
            else f"{crossing['below_floor_maximum']} / {crossing['outside_box_maximum']}"])
    sections.append(
        "### 密度与壁面诊断（整个运行）\n\n"
        "密度范围取自采样时刻的流体粒子；“中值漂移率” = 运行后半段 (ρ_min + ρ_max) / 2 对时间的线性拟合斜率；"
        "“压力电平” = 末样本的 c₀² (ρ_mid − ρ₀) / (ρ₀ U²)；A_f 回退 = 整个运行的累计次数，φ_min 与越界计数来自每 "
        "20 000 步一次的位置回读（只有 crossing 运行有这两项；n250_crossing 在加入该记录之前启动，没有）。\n\n"
        + markdown_table(["运行", "末时刻 t", "ρ 范围（首样本）", "ρ 范围（末样本）", "中值漂移率 kg/m³ 每时间单位",
                          "max |ρ − ρ₀|", "压力电平 / ρU²", "最大速度", "A_f 回退", "φ_min / dx",
                          "低于下限 / 越出盒（最大粒子数）"], rows))
    if failed:
        rows = [[label(record), record["name"], f"{record['t_last']:.2f}", record["error"],
                 (record.get("git") or {}).get("head", "?"), ", ".join(record["device_uuids"]) or "?"]
                for record in failed]
        sections.append("### 失败的运行（不变量被破坏，runner 退出码 3）\n\n"
                        + markdown_table(["运行", "run 目录", "最后样本 t", "错误", "提交", "GPU UUID"], rows))

    rows = [[label(record), record["name"], (record.get("git") or {}).get("head", "?"),
             ", ".join(record["device_uuids"]) or "?", f"{record['wall_hours']:.2f}",
             record["samples_in_window"], f"{record['kinetic_energy']:.4f}"] for record in ordered]
    sections.append("### 运行信息\n\n" + markdown_table(["运行", "run 目录", "提交", "GPU UUID", "墙钟 h",
                                                       "窗口内样本数", "窗口平均动能"], rows))

    text = ("# Re = 1000 方腔：各壁面模式对 Marchi 2021（K = 1）\n\n"
            f"数据：`{arguments.root}/<run 目录>`；分析 = experiment/validation/cavity_analysis.analyse_run，"
            f"最后 {AVERAGE_SPAN:g} 个时间单位平均。帧的定义见 experiment/wall_crossing/cavity_tables.py 文档串。\n\n"
            + "\n\n".join(sections) + "\n")
    (out / "tables.md").write_text(text, encoding="utf-8", newline="\n")
    (out / "tables.json").write_text(json.dumps({"runs": records, "convergence": convergence}, indent=1),
                                     encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
