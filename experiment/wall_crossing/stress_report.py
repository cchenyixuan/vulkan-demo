"""
stress_report.py - the tables and figures of the round-6 stress test (claude/wall_store_report.md section 12) from
the queues of experiment/wall_crossing/stress.py.

    python -m experiment.wall_crossing.stress_report

Reads logs/wall_store/round6/{L1,L2,L3,L4} (local), logs/wall_store/round6/followup (the three follow-ups after L4:
reruns, host-memory control, the 3-D hydrostatic box to t = 5 against the particle-pool tag) and
logs/wall_store/round6/remote/run (the remote queue's small files, pulled), logs/wall_store/round6/smi_local.csv and
remote/run/smi_remote.csv. Writes
claude/wall_store/tables/round6_*.md, claude/wall_store/figures/round6_*.png and logs/wall_store/round6/summary.json.

The checks of one run (box_run under the stress watchdog):
  outcome      ok | ok (deadline) | stop: <box_run stop condition> | stall (watchdog.json) | killed (supervisor) |
               exit <code>
  conserved    every sample's alive particles (readback) and every defrag boundary's alive count (the solver's global
               status) equal the first sample's
  counters     every overflow counter, far migrations, GPU and host stamp errors 0 at every defrag boundary and sample
  zombie       the solver has no counter of that name; counted here: samples where the readback rows differ from the
               status's alive count at the same step, and fluid particles outside the box (outside_box_count)
  finite       no NaN / Inf in any sampled quantity
  speed        cavity: largest fluid speed / lid speed <= 1.5; hydrostatic box: largest speed against earlier runs of
               the same case at the sample times they share (no interpolation: the instantaneous largest speed rings
               with the start-up pressure waves), within 0.5x the lowest and 1.5x the highest of those runs
  volume       volume constraint on: |theta|, I and the mean density over the run (long runs: trend over the second
               half against the scatter)
  step time    steps per second of the defrag intervals (sampling excluded): median of the last 10 % against the
               first 10 % (the first interval, start-up, left out)
  memory       resident memory of the process and its device-local memory, after the first 10 % against the end
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

ROUND6 = REPOSITORY_ROOT / "logs" / "wall_store" / "round6"
REMOTE6 = ROUND6 / "remote" / "run"
ROUND5_REMOTE = REPOSITORY_ROOT / "logs" / "wall_store" / "round5" / "remote"
TABLES = REPOSITORY_ROOT / "claude" / "wall_store" / "tables"
FIGURES = REPOSITORY_ROOT / "claude" / "wall_store" / "figures"
REST_DENSITY = 1000.0
LID_SPEED = 1.0
NON_FINITE = {"nan", "inf", "-inf", "+inf", "infinity", "-infinity"}
HYDRO_REFERENCES = {
    "h2d": sorted((REPOSITORY_ROOT / "logs" / "wall_store" / "new" / "hydrostatic").glob("ws_2d_n250_g30*"))
    + [REPOSITORY_ROOT / "logs" / "wall_store" / "round4" / "hydrostatic" / name for name in ("g30_default", "g30_final")],
    "h3d": [REPOSITORY_ROOT / "logs" / "wall_crossing" / "hydrostatic" / name
            for name in ("3d_n48_gtilt_crossing", "3d_n48_gtilt_crossing_default_volume")],
}
CASE_TEXT = {"c250": "2-D 方腔 250²", "c500": "2-D 方腔 500²", "c3d50": "3-D 方腔 每边 50", "c3d70": "3-D 方腔 每边 70",
             "h2d": "2-D 静水盒（倾斜重力）", "h3d": "3-D 48³ 静水盒（倾斜重力）"}
LANES_TEXT = {"kauto": "规则", "k1": "k = 1", "k4": "k = 4"}
CHECK_TEXT = {"conserved": "粒子守恒", "counters_zero": "计数为 0", "zombie_zero": "僵尸粒子为 0",
              "finite": "无 NaN/Inf", "speed": "最大速度", "no samples": "没有采样"}
HYDRO_TIME_TOLERANCE = 1e-6                 # a common sample time of a run and an earlier run of the same case
THROTTLE_BITS = {0x1: "空闲", 0x2: "应用频率设置", 0x4: "软件功率上限", 0x8: "硬件降速", 0x10: "同步加速",
                 0x20: "软件过热降速", 0x40: "硬件过热降速", 0x80: "硬件功率制动", 0x100: "显示时钟"}


# ---- reading --------------------------------------------------------------------------------------------------------

def read_table(path: pathlib.Path) -> tuple:
    """(columns as float arrays, count of non-finite cells) of a csv file; ({}, 0) when missing or empty."""
    if not path.exists():
        return {}, 0
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}, 0
    columns, bad = {}, 0
    for key in rows[0]:
        values = []
        for row in rows:
            cell = (row.get(key) or "").strip()
            if cell.lower() in NON_FINITE:
                bad += 1
            try:
                values.append(float(cell))
            except ValueError:
                values.append(math.nan)
        columns[key] = np.asarray(values)
    return columns, bad


def read_jsonl(path: pathlib.Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def local_run_dir(queue_dir: pathlib.Path, record: dict) -> pathlib.Path:
    """The run directory of a status record (a remote path is mapped into the pulled copy)."""
    text = record["run_dir"].replace("\\", "/")
    for marker in ("/box_run/", "/cavity/"):
        if marker in text:
            return queue_dir / (marker.strip("/") + "/" + text.split(marker, 1)[1])
    return pathlib.Path(text)


def queue_runs(queue_dir: pathlib.Path) -> list:
    return [(record, local_run_dir(queue_dir, record)) for record in read_jsonl(queue_dir / "status.jsonl")
            if "exit_code" in record]


# ---- the checks of one run ------------------------------------------------------------------------------------------

def hydro_envelope(key: str) -> list:
    runs = []
    for directory in HYDRO_REFERENCES.get(key, []):
        series, _ = read_table(directory / "series.csv")
        if series:
            runs.append((series["time"], series["speed_maximum"]))
    return runs


def ratio_drift(values: np.ndarray) -> float | None:
    values = values[np.isfinite(values)]
    if values.size < 3:
        return None
    values = values[1:]                     # the first interval holds the start-up
    count = max(1, values.size // 10)
    first, last = float(np.median(values[:count])), float(np.median(values[-count:]))
    return last / first - 1.0 if first > 0 else None


def growth(values: np.ndarray) -> float | None:
    values = values[np.isfinite(values)]
    if values.size < 3:
        return None
    start = values[max(1, values.size // 10)]
    return float(values[-1] - start)


def analyse_box(run_dir: pathlib.Path, record: dict, kind: str, envelopes: dict) -> dict:
    series, bad_series = read_table(run_dir / "series.csv")
    defrag, bad_defrag = read_table(run_dir / "defrag.csv")
    stop = read_json(run_dir / "watch_stop.json")
    constraint = [entry for entry in read_jsonl(run_dir / "volume_constraint.jsonl") if "theta" in entry]
    watchdog = (run_dir / "watchdog.json").exists()
    external = record.get("external_kill")
    code = record.get("exit_code")
    reason = stop["reason"] if stop else None
    if watchdog:
        outcome = "stall"
    elif external:
        outcome = "killed"
    elif code == 0:
        outcome = "ok"
    elif code == 3 and reason and "deadline" in reason:
        outcome = "ok (deadline)"
    elif code == 3 and reason:
        outcome = f"stop: {reason}"
    else:
        outcome = f"exit {code}"
    result = {"name": record["name"], "label": record.get("label"), "combo": record.get("combo"), "outcome": outcome,
              "exit_code": code, "seconds": record.get("seconds"), "start_unix": record.get("start_unix"),
              "watchdog": watchdog, "external_kill": external, "run_dir": str(run_dir)}
    frames = int(np.nanmax(defrag["frame"])) if defrag else 0
    result["frames"] = frames
    if not series:
        result["problems"] = ["no samples"]
        return result
    alive = series["alive"]
    result["time_end"] = float(series["time"][-1])
    checks = {}
    status_alive = defrag.get("alive", np.array([]))
    checks["conserved"] = bool(np.all(alive == alive[0]) and np.all(series["fluid_alive"] == series["fluid_alive"][0])
                               and np.all(status_alive == alive[0]))
    counters = [float(np.nanmax(series["overflow"]))]
    for key in ("overflow", "report_overflow", "far_migration", "report_far_migration", "stamp_error_gpu",
                "stamp_error_host"):
        if key in defrag:
            counters.append(float(np.nanmax(defrag[key])))
    checks["counters_zero"] = bool(max(counters) == 0)
    mismatch = 0
    if defrag:
        by_frame = dict(zip(defrag["frame"].astype(int), defrag["alive"]))
        mismatch = int(sum(1 for step, value in zip(series["step"].astype(int), alive)
                           if step in by_frame and by_frame[step] != value))
    result["zombie"] = {"status_readback_mismatch": mismatch,
                        "outside_box_maximum": int(np.nanmax(series["outside_box_count"]))}
    checks["zombie_zero"] = mismatch == 0 and result["zombie"]["outside_box_maximum"] == 0
    checks["finite"] = bad_series == 0 and bad_defrag == 0
    speed = float(np.nanmax(series["speed_maximum"]))
    result["speed_maximum"] = speed
    if kind == "cavity":
        result["speed_over_lid"] = speed / LID_SPEED
        checks["speed"] = bool(speed <= 1.5 * LID_SPEED)
    else:
        # the largest speed is an instantaneous value that rings with the start-up pressure waves: compared only at
        # the times the earlier runs sampled too (no interpolation between their samples)
        runs = envelopes.get(kind, [])
        ratios = []
        for time_value, value in zip(series["time"], series["speed_maximum"]):
            if time_value <= 0:
                continue
            reference = []
            for times, speeds in runs:
                nearest = int(np.argmin(np.abs(times - time_value)))
                if abs(times[nearest] - time_value) <= HYDRO_TIME_TOLERANCE:
                    reference.append(float(speeds[nearest]))
            if reference:
                ratios.append((float(time_value), value / max(reference), value / min(reference)))
        result["speed_reference_times"] = len(ratios)
        if ratios:
            result["speed_over_reference_max"] = float(max(ratio[1] for ratio in ratios))
            result["speed_over_reference_min"] = float(min(ratio[2] for ratio in ratios))
            checks["speed"] = bool(result["speed_over_reference_max"] <= 1.5
                                   and result["speed_over_reference_min"] >= 0.5)
        else:
            checks["speed"] = None
    if constraint:
        theta = np.array([entry["theta"] for entry in constraint])
        integral = np.array([entry["integral"] for entry in constraint])
        result["volume"] = {"theta_abs_max": float(np.max(np.abs(theta))), "integral_min": float(integral.min()),
                            "integral_max": float(integral.max()), "updates": len(constraint)}
    result["density_mean_range"] = [float(np.nanmin(series["density_mean"])), float(np.nanmax(series["density_mean"]))]
    if defrag:
        result["steps_per_second_median"] = float(np.nanmedian(defrag["steps_per_second"][1:])) \
            if defrag["steps_per_second"].size > 1 else float(defrag["steps_per_second"][0])
        result["step_time_drift"] = ratio_drift(defrag["steps_per_second"])
        result["rss_mb"] = [float(np.nanmin(defrag["rss_mb"])), float(np.nanmax(defrag["rss_mb"]))]
        result["rss_growth_mb"] = growth(defrag["rss_mb"])
        result["vram_mb"] = [float(np.nanmin(defrag["vram_process_mb"])), float(np.nanmax(defrag["vram_process_mb"]))]
        result["vram_growth_mb"] = growth(defrag["vram_process_mb"])
    result["checks"] = checks
    result["problems"] = [name for name, value in checks.items() if value is False]
    if outcome not in ("ok", "ok (deadline)"):
        result["problems"].insert(0, outcome)
    return result


def kind_of(combo: str | None, label: str | None) -> str:
    text = combo or label or ""
    for key in ("h2d", "h3d"):
        if text.startswith(key) or f"_{key}_" in text:
            return key
    return "cavity"


# ---- long runs --------------------------------------------------------------------------------------------------------

def long_history(run_dir: pathlib.Path) -> dict:
    series, _ = read_table(run_dir / "series.csv")
    defrag, _ = read_table(run_dir / "defrag.csv")
    constraint = [entry for entry in read_jsonl(run_dir / "volume_constraint.jsonl") if "theta" in entry]
    return {"series": series, "defrag": defrag, "constraint": constraint}


def trend(times: np.ndarray, values: np.ndarray) -> dict | None:
    """Second half of the run: linear change over that half against the residual scatter."""
    keep = np.isfinite(times) & np.isfinite(values)
    times, values = times[keep], values[keep]
    if times.size < 10:
        return None
    half = times >= times[0] + 0.5 * (times[-1] - times[0])
    slope, intercept = np.polyfit(times[half], values[half], 1)
    residual = values[half] - (slope * times[half] + intercept)
    change = float(slope * (times[half][-1] - times[half][0]))
    scatter = float(np.std(residual))
    return {"change_over_second_half": change, "scatter": scatter,
            "ratio": change / scatter if scatter > 0 else math.inf,
            "first_half_range": [float(values[~half].min()), float(values[~half].max())],
            "second_half_range": [float(values[half].min()), float(values[half].max())]}


def long_summary(history: dict) -> dict:
    series, defrag, constraint = history["series"], history["defrag"], history["constraint"]
    out = {"time_end": float(series["time"][-1]), "samples": int(series["time"].size),
           "kinetic_energy": trend(series["time"], series["kinetic_energy"]),
           "density_mean": trend(series["time"], series["density_mean"])}
    if constraint:
        times = np.array([entry["time"] for entry in constraint])
        out["theta"] = trend(times, np.array([entry["theta"] for entry in constraint]))
        out["integral"] = trend(times, np.array([entry["integral"] for entry in constraint]))
    if defrag:
        out["steps_per_second"] = trend(defrag["unix"] - defrag["unix"][0], defrag["steps_per_second"])
    return out


def long_figure(histories: dict, smi: dict, out: pathlib.Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiment.wall_crossing.report_figures import save, setup_style
    setup_style()
    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"]
    figure, axes = plt.subplots(3, 2, figsize=(12.0, 10.0))
    for index, (label, history) in enumerate(histories.items()):
        colour = colours[index % len(colours)]
        series, defrag, constraint = history["series"], history["defrag"], history["constraint"]
        axes[0, 0].plot(series["time"], series["kinetic_energy"], color=colour, linewidth=1.2, label=label)
        axes[0, 1].plot(series["time"], series["density_mean"] - REST_DENSITY, color=colour, linewidth=1.0)
        if constraint:
            times = [entry["time"] for entry in constraint]
            axes[1, 0].plot(times, [entry["theta"] for entry in constraint], color=colour, linewidth=1.0)
            axes[1, 1].plot(times, [entry["integral"] for entry in constraint], color=colour, linewidth=1.0)
        if defrag:
            # the defrag intervals' time: the sample just before each interval end
            times = np.interp(defrag["frame"], series["step"], series["time"])
            axes[2, 0].plot(times, 1000.0 / defrag["steps_per_second"], color=colour, linewidth=0.6)
            axes[2, 1].plot(times, defrag["rss_mb"], color=colour, linewidth=1.0)
    axes[0, 0].set_ylabel("动能（J）")
    axes[0, 1].set_ylabel("平均密度 − ρ0（kg/m³）")
    axes[1, 0].set_ylabel("θ")
    axes[1, 1].set_ylabel("I")
    axes[2, 0].set_ylabel("每步 ms（重排间隔，不含采样）")
    axes[2, 1].set_ylabel("进程驻留内存（MiB）")
    for axis in axes.ravel():
        axis.set_xlabel("t")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    figure.suptitle(title)
    FIGURES.mkdir(parents=True, exist_ok=True)
    save(figure, out)


# ---- nvidia-smi -----------------------------------------------------------------------------------------------------

def read_smi(path: pathlib.Path) -> list:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) < 12 or "ERROR" in line:
            continue
        try:
            rows.append({"unix": float(cells[0]), "index": int(cells[2]), "uuid": cells[3],
                         "temperature": float(cells[4]), "power": float(cells[5]), "sm_clock": float(cells[6]),
                         "utilization": float(cells[8]), "memory_used": float(cells[9]), "pstate": cells[10],
                         "throttle": int(cells[11], 16)})
        except ValueError:
            continue
    return rows


def smi_window(rows: list, start: float, end: float) -> dict:
    out = {}
    for index in sorted({row["index"] for row in rows}):
        chosen = [row for row in rows if row["index"] == index and start <= row["unix"] <= end]
        busy = [row for row in chosen if row["utilization"] >= 50]
        if not chosen:
            continue
        reasons = {}
        for row in busy:
            for bit, text in THROTTLE_BITS.items():
                if row["throttle"] & bit:
                    reasons[text] = reasons.get(text, 0) + 1
        values = lambda key, source=busy or chosen: np.array([row[key] for row in source])  # noqa: E731
        out[index] = {"samples": len(chosen), "busy_samples": len(busy),
                      "temperature_median": float(np.median(values("temperature"))),
                      "temperature_max": float(np.max([row["temperature"] for row in chosen])),
                      "power_median": float(np.median(values("power"))),
                      "sm_clock_median": float(np.median(values("sm_clock"))),
                      "sm_clock_min": float(np.min(values("sm_clock"))),
                      "memory_used": [float(np.min(values("memory_used"))), float(np.max(values("memory_used")))],
                      "throttle_share": {text: count / max(1, len(busy)) for text, count in reasons.items()}}
    return out


def clock_effect(defrag: dict, smi_rows: list, card_indices: list) -> dict | None:
    """Steps per second of the defrag intervals against the slowest participating card's SM clock (nearest sample)."""
    if not defrag or not smi_rows:
        return None
    rows = [row for row in smi_rows if row["index"] in card_indices]
    if not rows:
        return None
    by_time = {}
    for row in rows:
        by_time.setdefault(row["unix"], []).append(row["sm_clock"])
    stamps = np.array(sorted(by_time))
    clocks = np.array([min(by_time[stamp]) for stamp in stamps])
    nearest = np.searchsorted(stamps, defrag["unix"]).clip(0, stamps.size - 1)
    clock = clocks[nearest]
    rate = defrag["steps_per_second"]
    keep = np.isfinite(rate) & np.isfinite(clock)
    keep[0] = False
    if keep.sum() < 20:
        return None
    clock, rate = clock[keep], rate[keep]
    low, high = np.percentile(clock, 25), np.percentile(clock, 75)
    return {"correlation": float(np.corrcoef(clock, rate)[0, 1]) if np.std(clock) > 0 else None,
            "clock_quartiles": [float(low), float(high)],
            "rate_low_clock": float(np.median(rate[clock <= low])), "rate_high_clock": float(np.median(rate[clock >= high]))}


# ---- tables ---------------------------------------------------------------------------------------------------------

def yes(value) -> str:
    return "—" if value is None else ("是" if value else "**否**")


def percent(value) -> str:
    return "—" if value is None else f"{100.0 * value:+.1f} %"


def run_row(run: dict, text: str) -> str:
    checks = run.get("checks", {})
    speed = (f"{run['speed_over_lid']:.3f} 盖速" if "speed_over_lid" in run else
             (f"{run['speed_over_reference_max']:.2f}×参考上沿" if "speed_over_reference_max" in run else
              f"{run.get('speed_maximum', math.nan):.4f}"))
    volume = run.get("volume")
    volume_text = (f"θ 的绝对值 ≤ {volume['theta_abs_max']:.1e}，I {volume['integral_min']:+.1e}…{volume['integral_max']:+.1e}"
                   if volume else "—")
    density = run.get("density_mean_range", [math.nan, math.nan])
    return (f"| {text} | {run['outcome']} | {run['frames']:,} | {run.get('time_end', math.nan):.2f} | "
            f"{yes(checks.get('conserved'))} | {yes(checks.get('counters_zero'))} | {yes(checks.get('zombie_zero'))} | "
            f"{yes(checks.get('finite'))} | {speed} | {volume_text}；ρ̄ − ρ0 {density[0] - REST_DENSITY:+.4f}…"
            f"{density[1] - REST_DENSITY:+.4f} | {run.get('steps_per_second_median', math.nan):.1f} | "
            f"{percent(run.get('step_time_drift'))} | {run.get('rss_mb', [math.nan] * 2)[0]:.0f} → "
            f"{run.get('rss_mb', [math.nan] * 2)[1]:.0f}（{run.get('rss_growth_mb') or 0:+.0f}） | "
            f"{run.get('vram_mb', [math.nan] * 2)[1]:.0f}（{run.get('vram_growth_mb') or 0:+.0f}） |")


RUN_HEADER = ("| 运行 | 结果 | 帧数 | 跑到 t | 粒子守恒 | 溢出等计数为 0 | 僵尸粒子为 0 | 无 NaN/Inf | 最大速度 | 总体积修正 θ、I；平均密度 | "
              "步 / 秒（中位） | 每步时间漂移（末 10 % 对首 10 %，步/秒） | 进程内存 MiB（增长） | 显存 MiB（增长） |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")


def mixed_table(runs: list) -> tuple:
    """Per combination: runs, frames, outcomes, failed checks, steps/s, drift."""
    groups = {}
    for run in runs:
        groups.setdefault(run["combo"], []).append(run)
    rows, totals = [], {"runs": 0, "frames": 0, "problems": 0, "stalls": 0, "abnormal": 0}
    for combo in sorted(groups):
        group = groups[combo]
        name, volume, lanes = combo.split("_")
        stalls = sum(1 for run in group if run["outcome"] == "stall")
        abnormal = sum(1 for run in group if run["outcome"] not in ("ok", "ok (deadline)", "stall"))
        failed = {}
        for run in group:
            for problem in run.get("problems", []):
                failed[problem] = failed.get(problem, 0) + 1
        frames = sum(run["frames"] for run in group)
        rates = [run["steps_per_second_median"] for run in group if run.get("steps_per_second_median")]
        drifts = [run["step_time_drift"] for run in group if run.get("step_time_drift") is not None]
        speeds = [run.get("speed_over_lid", run.get("speed_over_reference_max", math.nan)) for run in group]
        speed_text = f"{np.nanmax(speeds):.3f}"
        if name.startswith("h"):
            low = [run["speed_over_reference_min"] for run in group if "speed_over_reference_min" in run]
            common = sum(run.get("speed_reference_times", 0) for run in group)
            no_common = sum(1 for run in group if not run.get("speed_reference_times"))
            speed_text = (f"{np.nanmax(speeds):.2f} / {min(low):.2f}（{common} 个共同时刻）" if low else "无共同时刻") + (
                f"；{no_common} 次运行没有共同时刻" if low and no_common else "")
        rows.append(f"| {CASE_TEXT[name]} | {'开' if volume == 'von' else '关'} | {LANES_TEXT[lanes]} | {len(group)} | "
                    f"{frames:,} | {stalls} | {abnormal} | "
                    + ("无" if not failed else "；".join(f"{CHECK_TEXT.get(key, key)} ×{value}" for key, value in failed.items()))
                    + f" | {speed_text} | {np.median(rates) if rates else math.nan:.0f} | "
                    + (f"{100 * np.median(drifts):+.1f} %（{100 * min(drifts):+.1f}…{100 * max(drifts):+.1f}）"
                       if drifts else "—") + " |")
        totals["runs"] += len(group)
        totals["frames"] += frames
        totals["stalls"] += stalls
        totals["abnormal"] += abnormal
        totals["problems"] += sum(1 for run in group if run.get("problems"))
    header = ("| 算例 | 总体积修正 | k | 次数 | 总帧数 | 停住 | 异常退出 / 停止 | 未通过的检查（次数） | "
              "最大速度（方腔：对盖速；静水：同一时刻对以前运行的上沿 / 下沿） | "
              "步 / 秒（中位） | 每步时间漂移，中位（范围） |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
    return header + "\n".join(rows) + "\n", totals


def segment_drift(runs: list) -> dict | None:
    """Over a segment of short runs on one card: every run's median steps/s against the median of its combination on
    that card, the runs started in the last quarter of the segment against those of the first quarter (a run's own
    first-vs-last 10 % holds its start-up ramp, this does not)."""
    rates = {}
    for run in runs:
        if run.get("steps_per_second_median") and run.get("start_unix"):
            rates.setdefault(run["combo"], []).append(run["steps_per_second_median"])
    medians = {combo: float(np.median(values)) for combo, values in rates.items()}
    points = sorted((run["start_unix"], run["steps_per_second_median"] / medians[run["combo"]]) for run in runs
                    if run.get("steps_per_second_median") and run.get("start_unix"))
    if len(points) < 8:
        return None
    start, end = points[0][0], points[-1][0]
    first = [value for when, value in points if when <= start + 0.25 * (end - start)]
    last = [value for when, value in points if when >= end - 0.25 * (end - start)]
    return {"first_runs": len(first), "last_runs": len(last), "first": float(np.median(first)),
            "last": float(np.median(last)), "change": float(np.median(last) / np.median(first) - 1.0)}


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    envelopes = {key: hydro_envelope(key) for key in HYDRO_REFERENCES}
    summary = {"segments": {}}
    smi_local = read_smi(ROUND6 / "smi_local.csv")
    smi_remote = read_smi(REMOTE6 / "smi_remote.csv")

    def analyse_queue(queue_dir: pathlib.Path) -> list:
        return [analyse_box(run_dir, record, kind_of(record.get("combo"), record.get("label")), envelopes)
                for record, run_dir in queue_runs(queue_dir) if "/cavity/" not in record["run_dir"].replace("\\", "/")]

    # L1, L2, R1: one long run each
    long_runs = {"L1": (ROUND6 / "L1", "3-D 1M 方腔，双卡", smi_local, [0, 1]),
                 "L2": (ROUND6 / "L2", "2-D 1000² 方腔，双卡", smi_local, [0, 1])}
    remote_runs = analyse_queue(REMOTE6) if (REMOTE6 / "status.jsonl").exists() else []
    long_rows, histories = [], {}
    for key, (queue_dir, text, smi_rows, cards) in long_runs.items():
        runs = analyse_queue(queue_dir)
        if not runs:
            continue
        run = runs[0]
        history = long_history(pathlib.Path(run["run_dir"]))
        run["long"] = long_summary(history)
        run["clock_effect"] = clock_effect(history["defrag"], smi_rows, cards)
        histories[key] = (text, history)
        summary["segments"][key] = run
        long_rows.append(run_row(run, f"{key}：{text}"))
    r1 = [run for run in remote_runs if (run["label"] or "").startswith("R1")]
    if r1:
        run = r1[0]
        history = long_history(pathlib.Path(run["run_dir"]))
        run["long"] = long_summary(history)
        run["clock_effect"] = clock_effect(history["defrag"], smi_remote, [0])
        histories["R1"] = ("3-D 1M 方腔，远端单卡", history)
        summary["segments"]["R1"] = run
        long_rows.append(run_row(run, "R1：3-D 1M 方腔，远端单卡"))
    (TABLES / "round6_long.md").write_text(RUN_HEADER + "\n".join(long_rows) + "\n", encoding="utf-8", newline="\n")
    trend_rows = []
    for key, run in summary["segments"].items():
        long = run.get("long", {})
        for quantity, text in (("kinetic_energy", "动能"), ("density_mean", "平均密度 − ρ0"), ("theta", "θ"),
                               ("integral", "I"), ("steps_per_second", "步 / 秒")):
            entry = long.get(quantity)
            if not entry:
                continue
            offset = REST_DENSITY if quantity == "density_mean" else 0.0
            first, second = entry["first_half_range"], entry["second_half_range"]
            trend_rows.append(f"| {key} | {text} | {first[0] - offset:+.4g} … {first[1] - offset:+.4g} | "
                              f"{second[0] - offset:+.4g} … {second[1] - offset:+.4g} | "
                              f"{entry['change_over_second_half']:+.3g} | {entry['scatter']:.3g} | {entry['ratio']:+.1f} |")
    (TABLES / "round6_long_trend.md").write_text(
        "| 运行 | 量 | 前半程范围 | 后半程范围 | 后半程线性变化 | 后半程残差标准差 | 变化 / 残差 |\n"
        "|---|---|---|---|---|---|---|\n" + "\n".join(trend_rows) + "\n", encoding="utf-8", newline="\n")
    for key, (text, history) in histories.items():
        long_figure({text: history}, {}, FIGURES / f"round6_{key.lower()}_history.png",
                    f"{key}：{text} —— 动能、平均密度、θ、I、每步时间、进程内存")

    # L3, R3: mixed short runs
    for key, queue_dirs, filter_label in (("L3", [ROUND6 / "L3" / "device0", ROUND6 / "L3" / "device1"], None),
                                          ("R3", [REMOTE6], "R3")):
        runs = []
        for queue_dir in queue_dirs:
            if key == "R3":
                runs += [run for run in remote_runs if not (run["label"] or "").startswith(("R1", "R2"))]
            else:
                for run in analyse_queue(queue_dir):
                    run["device"] = queue_dir.name
                    runs.append(run)
        if not runs:
            continue
        table, totals = mixed_table(runs)
        (TABLES / f"round6_{key.lower()}.md").write_text(table, encoding="utf-8", newline="\n")
        summary["segments"][key] = {"totals": totals, "problem_runs": [
            {field: run.get(field) for field in ("name", "device", "outcome", "problems", "run_dir")}
            for run in runs if run.get("problems")]}
        if key == "L3":
            summary["segments"][key]["per_device"] = {
                device: {"runs": sum(1 for run in runs if run["device"] == device),
                         "frames": sum(run["frames"] for run in runs if run["device"] == device)}
                for device in ("device0", "device1")}
            drifts = {device: segment_drift([run for run in runs if run["device"] == device])
                      for device in ("device0", "device1")}
        else:
            drifts = {"remote": segment_drift(runs)}
        summary["segments"][key]["segment_drift"] = drifts
        summary.setdefault("mixed_runs", {})[key] = runs
    drift_rows = []
    for key in ("L3", "R3"):
        for device, entry in summary["segments"].get(key, {}).get("segment_drift", {}).items():
            if entry:
                card = {"device0": "本机卡 0", "device1": "本机卡 1", "remote": "远端"}[device]
                drift_rows.append(f"| {key} | {card} | {entry['first_runs']} | {entry['last_runs']} | "
                                  f"{entry['first']:.3f} | {entry['last']:.3f} | {100 * entry['change']:+.1f} % |")
    if drift_rows:
        (TABLES / "round6_mixed_drift.md").write_text(
            "| 段 | 卡 | 前四分之一时段的运行数 | 后四分之一时段的运行数 | 前段：步 / 秒对本组合中位之比（中位） | "
            "后段：同左 | 后段对前段 |\n|---|---|---|---|---|---|---|\n" + "\n".join(drift_rows) + "\n",
            encoding="utf-8", newline="\n")

    # L4: two runs at once on each card
    alone = {}
    for run in summary.get("mixed_runs", {}).get("L3", []):
        alone.setdefault((run["device"][-1], run["combo"]), []).append(run.get("steps_per_second_median") or math.nan)
    l4_rows, l4_totals = [], {"runs": 0, "frames": 0, "stalls": 0, "abnormal": 0, "problems": 0}
    for queue_dir in sorted((ROUND6 / "L4").glob("slot*_device*_*")):
        runs = analyse_queue(queue_dir)
        if not runs:
            continue
        slot, device, part = queue_dir.name.split("_")
        combo = runs[0]["combo"]
        rates = [run["steps_per_second_median"] for run in runs if run.get("steps_per_second_median")]
        reference = np.nanmedian(alone.get((device[-1], combo), [math.nan])) if alone.get((device[-1], combo)) else math.nan
        stalls = sum(1 for run in runs if run["outcome"] == "stall")
        abnormal = sum(1 for run in runs if run["outcome"] not in ("ok", "ok (deadline)", "stall"))
        problems = {}
        for run in runs:
            for problem in run.get("problems", []):
                problems[problem] = problems.get(problem, 0) + 1
        name, volume, lanes = combo.split("_")
        l4_rows.append(f"| {slot[-1]} | {device[-1]} | {part} | {CASE_TEXT[name]}，{LANES_TEXT[lanes]} | {len(runs)} | "
                       f"{sum(run['frames'] for run in runs):,} | {stalls} | {abnormal} | "
                       + ("无" if not problems else "；".join(f"{CHECK_TEXT.get(key, key)} ×{value}" for key, value in problems.items()))
                       + f" | {np.median(rates) if rates else math.nan:.0f} | {reference:.0f} | "
                       + (f"{np.median(rates) / reference:.2f}" if rates and np.isfinite(reference) else "—") + " |")
        l4_totals["runs"] += len(runs)
        l4_totals["frames"] += sum(run["frames"] for run in runs)
        l4_totals["stalls"] += stalls
        l4_totals["abnormal"] += abnormal
        l4_totals["problems"] += sum(1 for run in runs if run.get("problems"))
        summary.setdefault("L4_runs", []).extend(runs)
    if l4_rows:
        (TABLES / "round6_l4.md").write_text(
            "| 时段 | 卡 | 队列 | 算例 | 次数 | 总帧数 | 停住 | 异常退出 / 停止 | 未通过的检查 | 并发时步 / 秒（中位） | "
            "同卡单独跑（L3 同一组合）步 / 秒 | 并发 / 单独 |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n"
            + "\n".join(l4_rows) + "\n", encoding="utf-8", newline="\n")
        summary["segments"]["L4"] = {"totals": l4_totals}

    r2_section(summary)
    r1_section(summary)

    # smi per segment
    segment_windows = {}
    master = (ROUND6 / "master.log").read_text(encoding="utf-8").splitlines() if (ROUND6 / "master.log").exists() else []
    import datetime
    stamps = {}
    for line in master:
        date_text, time_text, *event = line.split()
        stamps[" ".join(event)] = datetime.datetime.strptime(f"{date_text} {time_text}", "%Y-%m-%d %H:%M:%S").timestamp()
    order = [("L1", "L1 start", "L1 end"), ("L2", "L1 end", "L2 end"), ("L3", "L2 end", "L3 end"),
             ("L4", "L3 end", "L4 slot 4 end")]
    smi_rows = []
    for key, start, end in order:
        if start in stamps:
            window = smi_window(smi_local, stamps[start], stamps.get(end, math.inf))
            segment_windows[key] = window
            for card, entry in window.items():
                smi_rows.append(smi_row(key, f"本机卡 {card}", entry))
    remote_status = read_jsonl(REMOTE6 / "status.jsonl")
    for key, prefix in (("R2", "R2"), ("R1", "R1"), ("R3", None)):
        chosen = [record for record in remote_status if "exit_code" in record
                  and ((record["label"].startswith(prefix)) if prefix else not record["label"].startswith(("R1", "R2")))]
        if chosen:
            window = smi_window(smi_remote, chosen[0]["start_unix"], chosen[-1]["start_unix"] + chosen[-1]["seconds"])
            segment_windows[key] = window
            for card, entry in window.items():
                smi_rows.append(smi_row(key, "远端", entry))
    (TABLES / "round6_smi.md").write_text(
        "| 段 | 卡 | 采样数（忙 ≥ 50 %） | 温度中位 / 最高 °C | 功耗中位 W | SM 频率中位 / 最低 MHz | 显存占用 MiB | 忙时降频原因（占忙采样） |\n"
        "|---|---|---|---|---|---|---|---|\n" + "\n".join(smi_rows) + "\n", encoding="utf-8", newline="\n")
    summary["smi"] = segment_windows
    followup_section(summary, envelopes)
    overview_table(summary)
    (ROUND6 / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps({key: value.get("totals", value.get("outcome")) for key, value in summary["segments"].items()},
                     indent=1, default=str))
    return 0


SEGMENT_TEXT = {"L1": "L1：3-D 1M 方腔，双卡，连续", "L2": "L2：2-D 1000² 方腔，双卡，连续",
                "L3": "L3：两张卡各自单卡混合短运行", "L4": "L4：每张卡同时两个单卡运行",
                "R1": "R1：3-D 1M 方腔，远端单卡，到 t = 90", "R2": "R2：2-D 500²，远端分段续跑到 t = 100",
                "R3": "R3：远端 3-D 小算例混合短运行"}


def overview_table(summary: dict) -> None:
    """One row per segment: runs, frames, stalls, abnormal exits, runs with a failed check (which checks)."""
    rows = []
    for key in ("L1", "L2", "L3", "L4", "R1", "R2", "R3"):
        entry = summary["segments"].get(key)
        if not entry:
            continue
        if key in ("L1", "L2", "R1"):
            runs = [entry]
        elif key in ("L3", "R3"):
            runs = summary.get("mixed_runs", {}).get(key, [])
        elif key == "L4":
            runs = summary.get("L4_runs", [])
        else:
            runs = None
        if runs is not None:
            failed = {}
            for run in runs:
                for problem in run.get("problems", []):
                    failed[problem] = failed.get(problem, 0) + 1
            stalls = sum(1 for run in runs if run.get("outcome") == "stall")
            abnormal = sum(1 for run in runs if run.get("outcome") not in ("ok", "ok (deadline)", "stall"))
            rows.append(f"| {SEGMENT_TEXT[key]} | {len(runs)} | {sum(run.get('frames', 0) for run in runs):,} | "
                        f"{stalls} | {abnormal} | {sum(1 for run in runs if run.get('problems'))} | "
                        + ("无" if not failed else "；".join(f"{CHECK_TEXT.get(name, name)} ×{count}" for name, count in failed.items())) + " |")
        else:
            checks = entry.get("checks", {})
            failed = [name for name, value in checks.items() if value is False]
            rows.append(f"| {SEGMENT_TEXT[key]} | {entry.get('segments', 0)} 段 | {entry.get('frames', 0):,} | 0 | "
                        f"{entry.get('abnormal', 0)} | {1 if failed else 0} | " + ("无" if not failed else "；".join(CHECK_TEXT.get(name, name) for name in failed)) + " |")
    (TABLES / "round6_overview.md").write_text(
        "| 段 | 运行次数 | 总帧数 | 停住 | 异常退出 / 停止 | 有检查未通过的运行 | 未通过的检查（次数） |\n"
        "|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n", encoding="utf-8", newline="\n")


def absolute_times(records: list) -> np.ndarray:
    """Times of a log whose clock may restart with each segment: a drop starts a new offset (the previous end)."""
    times, offset, previous = [], 0.0, -math.inf
    for record in records:
        value = float(record["time"])
        if value + offset < previous - 1e-9:
            offset = previous
        times.append(value + offset)
        previous = value + offset
    return np.asarray(times)


def r2_section(summary: dict) -> None:
    """R2: 2-D 500^2 in segments (one checkpoint interval each, resumed) against the section-11 continuous run."""
    run_dir = REMOTE6 / "cavity" / "r6_r2_n500"
    continuous = ROUND5_REMOTE / "cavity" / "r5_n500"
    if not (run_dir / "samples.jsonl").exists():
        return
    status = [record for record in read_jsonl(REMOTE6 / "status.jsonl")
              if "exit_code" in record and record["label"].startswith("R2")]
    runner_segments = read_jsonl(run_dir / "segments.jsonl")
    defrag, bad_defrag = read_table(run_dir / "defrag.csv")
    samples = read_jsonl(run_dir / "samples.jsonl")
    wrappers = [record for record in read_jsonl(run_dir / "wrapper.jsonl") if record.get("event") == "start"]
    pids = [record["pid"] for record in wrappers]
    rows = []
    for index, record in enumerate(status):
        runner = runner_segments[index] if index < len(runner_segments) else {}
        pid = pids[index] if index < len(pids) else None
        chosen = defrag["pid"] == pid if defrag and pid is not None else np.zeros(0, bool)
        rates = defrag["steps_per_second"][chosen][1:] if chosen.any() else np.array([])
        rss = defrag["rss_mb"][chosen] if chosen.any() else np.array([math.nan])
        totals = runner.get("totals", {})
        rows.append(f"| {index} | {record['exit_code']} | {runner.get('status', '—')} | {runner.get('start_step', 0):,} | "
                    f"{record['seconds']:.0f} | {np.median(rates) if rates.size else math.nan:.0f} | "
                    f"{np.nanmax(rss):.0f} | " + ("0" if totals and all(value == 0 for value in totals.values())
                                                 else json.dumps(totals)) + " |")
    alive = np.array([row["alive"] for row in samples])
    speed = np.array([row["max_speed"] for row in samples])
    checks = {"conserved": bool(np.all(alive == alive[0]) and (not defrag or np.all(defrag["alive"] == alive[0]))),
              "counters_zero": bool(not defrag or all(float(np.nanmax(defrag[key])) == 0 for key in
                                    ("overflow", "report_overflow", "far_migration", "stamp_error_gpu", "stamp_error_host"))),
              "finite": bad_defrag == 0 and bool(np.all(np.isfinite(speed))),
              "speed": bool(speed.max() <= 1.5 * LID_SPEED), "speed_maximum": float(speed.max()),
              "runner_segments_complete": all(segment.get("status") == "complete" for segment in runner_segments)}
    (TABLES / "round6_r2_segments.md").write_text(
        "| 段 | 退出码 | 运行器的段状态 | 起步 | 墙钟 s | 步 / 秒（中位） | 进程内存峰值 MiB | 溢出 / 远距迁移 / 时间戳错误 / 粒子数漂移 |\n"
        "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    # the window analysis (cavity_window_analysis.py on the remote box) against the section-11 run's
    analysis = read_json(ROUND6 / "remote" / "r6_r2_n500_analysis.json")
    latest = read_json(ROUND5_REMOTE / "r5_n500_analysis.json")
    result = read_json(run_dir / "result.json") or {}
    out = {"checks": checks, "segments": len(status), "outcome": result.get("status"),
           "frames": int(result.get("stop_step") or max(int(samples[-1]["step"]) if samples else 0,
                                                        int(np.nanmax(defrag["frame"])) if defrag else 0)),
           "abnormal": sum(1 for record in status if record["exit_code"] != 0 or record.get("watchdog")
                           or record.get("external_kill"))}
    if analysis and latest:
        from experiment.validation import cavity_reference
        reference = cavity_reference.marchi2021()["extrema"]
        psi_reference = float(reference["psi_min"][0])

        def deviation(value, reference_value):
            return 100.0 * (abs(value) - abs(reference_value)) / abs(reference_value)
        table = []
        for text, entry in (("分段续跑（本轮 R2）", analysis), ("连续运行（第 11 节 r5_n500）", latest)):
            psi = entry["psi_min"]
            cells = [text, f"{entry['window'][0]:.2f}–{entry['window'][1]:.2f}（{entry['samples_in_window']}）",
                     f"{psi['mid']:.5f}（{deviation(psi['mid'], psi_reference):+.2f} %）",
                     f"{psi['wall']:.5f}（{deviation(psi['wall'], psi_reference):+.2f} %）"]
            for name in ("u_min", "v_max", "v_min"):
                value = entry["extrema"][name]["value"]
                cells.append(f"{value:+.5f}（{deviation(value, float(reference[name][0])):+.2f} %）")
            errors = entry["errors"]
            cells.append(f"{errors['wall_mls_u']['L2']:.4f} / {errors['wall_mls_v']['L2']:.4f}")
            cells.append(f"{entry['kinetic_energy_window']['mean']:.4f}")
            cells.append(f"{entry['density_mean_window']['mean'] - REST_DENSITY:+.5f}")
            table.append("| " + " | ".join(cells) + " |")
        differences = {"psi_mid": analysis["psi_min"]["mid"] - latest["psi_min"]["mid"],
                       "psi_wall": analysis["psi_min"]["wall"] - latest["psi_min"]["wall"],
                       **{name: analysis["extrema"][name]["value"] - latest["extrema"][name]["value"]
                          for name in ("u_min", "v_max", "v_min")},
                       "l2_u": analysis["errors"]["wall_mls_u"]["L2"] - latest["errors"]["wall_mls_u"]["L2"],
                       "l2_v": analysis["errors"]["wall_mls_v"]["L2"] - latest["errors"]["wall_mls_v"]["L2"],
                       "kinetic_energy": analysis["kinetic_energy_window"]["mean"] - latest["kinetic_energy_window"]["mean"],
                       "kinetic_energy_sem": [analysis["kinetic_energy_window"]["sem"], latest["kinetic_energy_window"]["sem"]]}
        table.append(f"| 差（分段 − 连续） | — | {differences['psi_mid']:+.5f} | {differences['psi_wall']:+.5f} | "
                     f"{differences['u_min']:+.5f} | {differences['v_max']:+.5f} | {differences['v_min']:+.5f} | "
                     f"{differences['l2_u']:+.4f} / {differences['l2_v']:+.4f} | {differences['kinetic_energy']:+.4f} | — |")
        table.append(f"| Marchi et al. 2021 | — | {psi_reference:.5f} | {psi_reference:.5f} | "
                     + " | ".join(f"{float(reference[name][0]):+.5f}" for name in ("u_min", "v_max", "v_min"))
                     + " | — | — | — |")
        (TABLES / "round6_r2_compare.md").write_text(
            "| 2-D 500² 方腔，t = 80–100 平均 | 窗口（样本数） | ψ_min，mid 帧 | ψ_min，wall 帧 | u_min | v_max | v_min | "
            "剖面 L2 对 Marchi，wall 帧 MLS：u / v | 窗口内动能 | 窗口内平均密度 − ρ0 |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(table) + "\n", encoding="utf-8", newline="\n")
        out["differences"] = differences
    # histories against the continuous run at the same times
    histories = {}
    for key, directory in (("segmented", run_dir), ("continuous", continuous)):
        sample_rows = read_jsonl(directory / "samples.jsonl")
        field = [row for row in read_jsonl(directory / "field_history.jsonl")]
        constraint = [row for row in read_jsonl(directory / "volume_constraint.jsonl") if "theta" in row]
        histories[key] = {"time": np.array([row["time"] for row in sample_rows]),
                          "kinetic_energy": np.array([row["kinetic_energy"] for row in sample_rows]),
                          "field_time": np.array([row["time"] for row in field]),
                          "density_mean": np.array([row["density_mean"] for row in field]),
                          "constraint_time": absolute_times(constraint),
                          "theta": np.array([row["theta"] for row in constraint]),
                          "integral": np.array([row["integral"] for row in constraint])}
    compare = {}
    for quantity, time_key in (("kinetic_energy", "time"), ("density_mean", "field_time"), ("theta", "constraint_time"),
                               ("integral", "constraint_time")):
        first, second = histories["segmented"], histories["continuous"]
        late = first[time_key] >= 80.0
        if late.sum() == 0 or second[time_key].size == 0:
            continue
        other = np.interp(first[time_key][late], second[time_key], second[quantity])
        compare[quantity] = {"segmented_mean_80_100": float(np.mean(first[quantity][late])),
                             "continuous_mean_80_100": float(np.mean(second[quantity][second[time_key] >= 80.0])),
                             "segmented_std_80_100": float(np.std(first[quantity][late])),
                             "continuous_std_80_100": float(np.std(second[quantity][second[time_key] >= 80.0])),
                             "largest_pointwise_difference_80_100": float(np.max(np.abs(first[quantity][late] - other)))}
    out["history_compare"] = compare
    hist_rows = [f"| {name} | {entry['segmented_mean_80_100']:+.6g} ± {entry['segmented_std_80_100']:.2g} | "
                 f"{entry['continuous_mean_80_100']:+.6g} ± {entry['continuous_std_80_100']:.2g} | "
                 f"{entry['largest_pointwise_difference_80_100']:.3g} |"
                 for name, entry in compare.items()]
    (TABLES / "round6_r2_history.md").write_text(
        "| 量 | 分段续跑，t = 80–100 平均 ± 标准差 | 连续运行（第 11 节） | 同一时刻之差的最大值 |\n|---|---|---|---|\n"
        + "\n".join(hist_rows) + "\n", encoding="utf-8", newline="\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiment.wall_crossing.report_figures import save, setup_style
    setup_style()
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 7.6))
    boundaries = [segment.get("start_step", 0) * 1.5e-5 for segment in runner_segments[1:]]
    for key, colour, width in (("continuous", "#eb6834", 1.6), ("segmented", "#2a78d6", 1.0)):
        history = histories[key]
        text = "连续运行（第 11 节）" if key == "continuous" else "分段续跑（R2）"
        axes[0, 0].plot(history["time"], history["kinetic_energy"], color=colour, linewidth=width, label=text)
        axes[0, 1].plot(history["field_time"], history["density_mean"] - REST_DENSITY, color=colour, linewidth=width)
        axes[1, 0].plot(history["constraint_time"], history["theta"], color=colour, linewidth=width)
        axes[1, 1].plot(history["constraint_time"], history["integral"], color=colour, linewidth=width)
    for axis in axes.ravel():
        for boundary in boundaries:
            axis.axvline(boundary, color="#bbbbbb", linewidth=0.6, zorder=0)
        axis.set_xlabel("t")
    axes[0, 0].set_ylabel("动能（J）")
    axes[0, 1].set_ylabel("平均密度 − ρ0（kg/m³）")
    axes[1, 0].set_ylabel("θ")
    axes[1, 1].set_ylabel("I")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    figure.suptitle("R2：2-D 500² 方腔分段续跑与第 11 节的连续运行（灰线：续跑处）")
    save(figure, FIGURES / "round6_r2_history.png")
    summary["segments"]["R2"] = out


def r1_section(summary: dict) -> None:
    """R1: 3-D 1M to t = 90; windows t = 35-45 (against section 11's r5_cavity3d_1m) and t = 80-90."""
    from experiment.wall_crossing.cavity3d_reference import (REFERENCE_U, REFERENCE_V, errors, read_reference,
                                                             window_mean_profiles)
    status = [record for record in read_jsonl(REMOTE6 / "status.jsonl")
              if "exit_code" in record and record["label"].startswith("R1")]
    if not status:
        return
    run_dir = local_run_dir(REMOTE6, status[0])
    if not (run_dir / "profiles.json").exists():
        return
    reference_y, reference_u = read_reference(REFERENCE_U)
    reference_x, reference_v = read_reference(REFERENCE_V)
    reference = {"u_min": float(reference_u.min()), "v_max": float(reference_v.max()), "v_min": float(reference_v.min())}
    previous = ROUND5_REMOTE / "box_run" / "r5_cavity3d_1m"
    entries = {}
    for key, directory, window, text in (("r6_35", run_dir, (35.0, 45.0), "本轮 R1，t = 35–45"),
                                         ("r6_80", run_dir, (80.0, 90.0), "本轮 R1，t = 80–90"),
                                         ("r5_35", previous, (35.0, 45.0), "第 11 节（r5_cavity3d_1m），t = 35–45")):
        profiles = json.loads((directory / "profiles.json").read_text(encoding="utf-8"))
        mean = window_mean_profiles(profiles, window)
        if mean is None:
            continue
        series, _ = read_table(directory / "series.csv")
        constraint = [entry for entry in read_jsonl(directory / "volume_constraint.jsonl") if "theta" in entry]
        rows = (series["time"] >= window[0] - 1e-9) & (series["time"] <= window[1] + 1e-9)
        chosen = [entry for entry in constraint if window[0] - 1e-9 <= entry["time"] <= window[1] + 1e-9]
        u, v = mean["u_along_y"], mean["v_along_x"]
        extrema = {"u_min": float(np.nanmin(u["velocity"])), "v_max": float(np.nanmax(v["velocity"])),
                   "v_min": float(np.nanmin(v["velocity"]))}
        entries[key] = {"text": text, "count": mean["count"], "mean": mean, "extrema": extrema,
                        "deviation": {name: (abs(value) - abs(reference[name])) / abs(reference[name])
                                      for name, value in extrema.items()},
                        "errors_u": errors(u["coordinate"], u["velocity"], reference_y, reference_u),
                        "errors_v": errors(v["coordinate"], v["velocity"], reference_x, reference_v),
                        "kinetic_energy": float(np.mean(series["kinetic_energy"][rows])),
                        "kinetic_energy_slope": float(np.polyfit(series["time"][rows], series["kinetic_energy"][rows], 1)[0]),
                        "density_mean": float(np.mean(series["density_mean"][rows])),
                        "theta": float(np.mean([entry["theta"] for entry in chosen])) if chosen else math.nan,
                        "integral": float(np.mean([entry["integral"] for entry in chosen])) if chosen else math.nan}
    table = []
    for key, entry in entries.items():
        table.append(f"| {entry['text']} | {entry['count']} | "
                     + " | ".join(f"{entry['extrema'][name]:+.4f}（{100 * entry['deviation'][name]:+.1f} %）"
                                  for name in ("u_min", "v_max", "v_min"))
                     + f" | {entry['errors_u']['l2']:.4f} / {entry['errors_v']['l2']:.4f} | {entry['kinetic_energy']:.3f}"
                     f"（{entry['kinetic_energy_slope']:+.4f} / 单位时间） | {entry['density_mean'] - REST_DENSITY:+.4f} | "
                     f"{entry['theta']:+.2e} | {entry['integral']:+.2e} |")
    table.append(f"| 基准（Albensoeder & Kuhlmann 2005） | — | {reference['u_min']:+.4f} | {reference['v_max']:+.4f} | "
                 f"{reference['v_min']:+.4f} | — | — | — | — | — |")
    pairs = []

    def compare(first: str, second: str, text: str) -> None:
        if first not in entries or second not in entries:
            return
        extrema = {name: entries[first]["extrema"][name] - entries[second]["extrema"][name]
                   for name in ("u_min", "v_max", "v_min")}
        profiles = {}
        for line in ("u_along_y", "v_along_x"):
            delta = (np.asarray(entries[first]["mean"][line]["velocity"])
                     - np.asarray(entries[second]["mean"][line]["velocity"]))
            profiles[line] = {"l2": float(np.sqrt(np.nanmean(delta ** 2))), "linf": float(np.nanmax(np.abs(delta)))}
        pairs.append({"pair": text, "extrema": extrema, "profiles": profiles})
    compare("r6_35", "r5_35", "本轮 t = 35–45 − 第 11 节 t = 35–45")
    compare("r6_80", "r6_35", "本轮 t = 80–90 − 本轮 t = 35–45")
    compare("r6_80", "r5_35", "本轮 t = 80–90 − 第 11 节 t = 35–45")
    pair_rows = [f"| {pair['pair']} | {pair['extrema']['u_min']:+.4f} | {pair['extrema']['v_max']:+.4f} | "
                 f"{pair['extrema']['v_min']:+.4f} | {pair['profiles']['u_along_y']['l2']:.4f} / "
                 f"{pair['profiles']['v_along_x']['l2']:.4f} | {pair['profiles']['u_along_y']['linf']:.4f} / "
                 f"{pair['profiles']['v_along_x']['linf']:.4f} |" for pair in pairs]
    (TABLES / "round6_r1_windows.md").write_text(
        "| 3-D 1M 方腔 | 窗口内剖面数 | u_min（大小相对基准） | v_max | v_min | 剖面 L2 对基准：u(y) / v(x) | 窗口内动能（斜率） | "
        "平均密度 − ρ0 | θ 平均 | I 平均 |\n|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(table)
        + "\n\n| 两个窗口的平均剖面之差 | Δu_min | Δv_max | Δv_min | 剖面之差 L2：u(y) / v(x) | L∞ |\n|---|---|---|---|---|---|\n"
        + "\n".join(pair_rows) + "\n", encoding="utf-8", newline="\n")
    summary.setdefault("r1_windows", {})
    summary["r1_windows"] = {"entries": {key: {name: value for name, value in entry.items() if name != "mean"}
                                         for key, entry in entries.items()}, "pairs": pairs}


def smi_row(key: str, card: str, entry: dict) -> str:
    reasons = "；".join(f"{text} {100 * share:.0f} %" for text, share in sorted(entry["throttle_share"].items(),
                                                                              key=lambda item: -item[1])) or "无"
    return (f"| {key} | {card} | {entry['samples']}（{entry['busy_samples']}） | {entry['temperature_median']:.0f} / "
            f"{entry['temperature_max']:.0f} | {entry['power_median']:.0f} | {entry['sm_clock_median']:.0f} / "
            f"{entry['sm_clock_min']:.0f} | {entry['memory_used'][0]:.0f}–{entry['memory_used'][1]:.0f} | {reasons} |")


# ---- the follow-ups after L4 (logs/wall_store/round6/followup) ------------------------------------------------------

def queue_glob(pattern: str) -> list:
    return [(record, run_dir) for queue in sorted(ROUND6.glob(pattern))
            for record, run_dir in queue_runs(queue)]


def followup_section(summary: dict, envelopes: dict) -> None:
    """(1) the 20 reruns of the L3 combination that failed the hydrostatic speed check; (2) the host-memory control
    (two 300 s dual runs, the transport workers' timestamp dicts cleared at every defrag or not); (3) the 3-D
    hydrostatic box to t = 5, the final defaults against the particle-pool tag crossing-pool (both one run per card at
    a time) and the L4 concurrent runs, and every round-6 3-D hydrostatic run checked again against the pool runs."""
    out = {}
    reruns = [analyse_box(run_dir, record, "h3d", envelopes) for record, run_dir in queue_glob("followup/repro_h3d_device*")]
    if reruns:
        out["reruns"] = {"runs": len(reruns), "failed": sum(1 for run in reruns if run.get("problems")),
                         "ratio_high": [min(run["speed_over_reference_max"] for run in reruns),
                                        max(run["speed_over_reference_max"] for run in reruns)]}
    rows = []
    for record, run_dir in queue_glob("followup/memory_control"):
        defrag, _ = read_table(run_dir / "defrag.csv")
        keep = np.isfinite(defrag["rss_mb"])
        frames, rss = defrag["frame"][keep], defrag["rss_mb"][keep]
        slope = float(np.polyfit(frames, rss, 1)[0]) * 1048576.0
        cleared = record["label"].endswith("clear1")
        rows.append(f"| {'每次整理时清空 `worker.timestamps`（`STRESS_CLEAR_WORKER_TIMESTAMPS=1`）' if cleared else '不清空（与 L1、L2 相同）'} | "
                    f"{int(frames.max()):,} | {rss[0]:.0f} → {rss[-1]:.0f} | {slope:.0f} B | "
                    f"{np.nanmedian(defrag['steps_per_second'][1:]):.0f} |")
        out.setdefault("memory_control", {})[record["label"]] = {"frames": int(frames.max()), "rss": [float(rss[0]), float(rss[-1])],
                                                                 "bytes_per_frame": slope}
    if rows:
        (TABLES / "round6_memory_control.md").write_text(
            "| 运行 | 帧数 | 进程内存 MB | 每帧增长 | 步 / 秒（中位） |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n",
            encoding="utf-8", newline="\n")
    pool_members = queue_glob("followup/h3d_t5_pool_device*")
    if not pool_members:
        summary["followup"] = out
        return
    pool_series = [read_table(run_dir / "series.csv")[0] for _, run_dir in pool_members]
    pool_envelope = {"h3d": [(series["time"], series["speed_maximum"]) for series in pool_series]}
    groups = {"墙存储（最终缺省），单独跑": queue_glob("followup/h3d_t5_ws_device*"),
              "粒子池版本（crossing-pool），单独跑": pool_members,
              "墙存储，L4 同卡并发": queue_glob("L4/slot2_device*_b")}

    def value_at(series: dict, column: str, time_value: float) -> float:
        if column not in series:
            return math.nan
        index = np.where(np.abs(series["time"] - time_value) < HYDRO_TIME_TOLERANCE)[0]
        return float(series[column][index[0]]) if index.size else math.nan

    loaded = {name: [read_table(run_dir / "series.csv")[0] for _, run_dir in members] for name, members in groups.items()}
    old = [read_table(directory / "series.csv")[0] for directory in HYDRO_REFERENCES["h3d"]]
    columns = (("speed_maximum", 4), ("kinetic_energy", 5), ("density_mean", 3))
    rows = []
    for time_value in (1.0, 2.0, 3.0, 4.0, 5.0):
        for index, (name, series_list) in enumerate(loaded.items()):
            cells = [f"{time_value:.0f}" if index == 0 else "", f"{name}（{len(series_list)} 次）"]
            for column, digits in columns:
                values = np.array([value_at(series, column, time_value) for series in series_list])
                values = values[np.isfinite(values)]
                cells.append("—" if values.size == 0 else
                             f"{np.median(values):.{digits}f}（{values.min():.{digits}f}–{values.max():.{digits}f}）")
            rows.append("| " + " | ".join(cells) + " |")
        cells = ["", "以前两次（11-17 / 11-19，粒子池，旧配置）"]
        for column, digits in columns:
            values = [value_at(series, column, time_value) for series in old]
            cells.append(" / ".join("—" if not math.isfinite(value) else f"{value:.{digits}f}" for value in values))
        rows.append("| " + " | ".join(cells) + " |")
    (TABLES / "round6_h3d_t5.md").write_text(
        "| t | 运行 | 最大速度，中位（范围） | 动能，中位（范围） | 平均密度 kg/m³，中位（范围） |\n|---|---|---|---|---|\n"
        + "\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    t5 = {}
    for name, members in groups.items():
        results = [analyse_box(run_dir, record, "h3d", envelopes) for record, run_dir in members]
        t5[name] = {"runs": len(results), "failed_old": sum(1 for run in results if "speed" in run.get("problems", [])),
                    "other_problems": sum(1 for run in results if [p for p in run.get("problems", []) if p != "speed"]),
                    "ratio_high_old": [round(run.get("speed_over_reference_max", math.nan), 3) for run in results],
                    "steps_per_second": [run.get("steps_per_second_median") for run in results]}
    out["t5"] = t5
    # every round-6 wall-store 3-D hydrostatic run against the pool runs (same rule)
    recheck = {"L3（本机单卡，到 t ≤ 2.5）": [(run, pathlib.Path(run["run_dir"])) for run in summary["mixed_runs"].get("L3", [])
                                         if (run.get("combo") or "").startswith("h3d")],
               "R3（远端单卡，到 t ≤ 2.5）": [(run, pathlib.Path(run["run_dir"])) for run in summary["mixed_runs"].get("R3", [])
                                         if (run.get("combo") or "").startswith("h3d")],
               "L4（同卡并发，到 t = 5）": groups["墙存储，L4 同卡并发"],
               "重跑 0117 的组合（到 t = 1.5）": queue_glob("followup/repro_h3d_device*"),
               "最终缺省单独跑（到 t = 5）": groups["墙存储（最终缺省），单独跑"]}
    rows, out["recheck"] = [], {}
    for name, members in recheck.items():
        old_failed = new_failed = 0
        high, low = [], []
        for record, run_dir in members:
            old_result = analyse_box(run_dir, dict(record), "h3d", envelopes)
            new_result = analyse_box(run_dir, dict(record), "h3d", pool_envelope)
            old_failed += "speed" in old_result.get("problems", [])
            new_failed += "speed" in new_result.get("problems", [])
            if new_result.get("speed_over_reference_max") is not None:
                high.append(new_result["speed_over_reference_max"])
                low.append(new_result["speed_over_reference_min"])
        rows.append(f"| {name} | {len(members)} | {old_failed} | {new_failed} | {max(high):.2f} / {min(low):.2f} |")
        out["recheck"][name] = {"runs": len(members), "failed_old": old_failed, "failed_pool": new_failed,
                                "ratio_high": max(high), "ratio_low": min(low)}
    (TABLES / "round6_h3d_recheck.md").write_text(
        "| 运行 | 次数 | 对以前两次（11-17 / 11-19）未通过 | 对粒子池版本 6 次（同一时期的缺省）未通过 | "
        "对粒子池版本：最大的上沿比 / 最小的下沿比 |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n",
        encoding="utf-8", newline="\n")
    summary["followup"] = out


if __name__ == "__main__":
    sys.exit(main())
