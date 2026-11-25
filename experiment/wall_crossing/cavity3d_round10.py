"""
cavity3d_round10.py - claude/wall_store_report.md section 16 (round 10): N = 280 of the 3-D cavity at Re = 1000,
interpolated from round 9's continued N = 200 state (L200_cont, t = 53.58) and stopped by the steadiness rule of the
round-10 brief; the extrapolation over N = 100 .. 280; the thesis tables and figures.

    python -m experiment.wall_crossing.cavity3d_round10 monitor --run <box_run directory> --stop-file <file> [--commit]

monitor (runs beside box_run --stop-file, until the run's summary.json is written):
  - every new tube record of the run (tubes/t*.npz, every 0.5 units of time): the extrema by the direct fit of report
    15.2 (cavity3d_round9.extrema_of; the linear MLS beside it) appended to <run>/extrema.jsonl;
  - the steadiness rule: a straight line through the instantaneous u_min and v_max deviations (percentage points of
    the paper's table 10) of the records in the last 6 units of level time; both |slope| < 0.01 points per unit and
    level time >= 12 -> the stop file is written (box_run ends at its next line record, its checkpoint written);
  - every 2 units of level time one row of claude/wall_store/round10_progress.md: level time, wall clock, the last 2
    units' mean u_min / v_max deviations, the 6-unit slopes, the two extrapolations' predictions for N = 280 (round 9's
    fits with p = 1 and p = 2, logs/wall_store/round9/extrapolation.json) and which is closer, the expected end;
    with --commit the file is committed alone after every row;
  - the C drive keeps one checkpoint: older checkpoints of the run are deleted once a newer one is complete.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re
import subprocess
import sys
import time

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing import cavity3d_extrema as extrema_methods      # noqa: E402
from experiment.wall_crossing.cavity3d_round9 import METHOD, deviation, extrema_of, reference      # noqa: E402

PROGRESS = REPOSITORY_ROOT / "claude" / "wall_store" / "round10_progress.md"
ROUND9_EXTRAPOLATION = REPOSITORY_ROOT / "logs" / "wall_store" / "round9" / "extrapolation.json"
QUANTITIES = ("u_min", "v_max", "v_min", "w_abs_max")
STEADY_SPAN = 6.0              # the brief: the straight line through the last 6 units
STEADY_SLOPE = 0.01            # percentage points per unit of time, both u_min and v_max
STEADY_MINIMUM = 12.0          # level time
LEVEL_CAP = 20.0               # box_run --time 20 ends the level there
PROGRESS_EVERY = 2.0
MEAN_SPAN = 2.0
CHECKPOINT_NAME = re.compile(r"^c\d{10}\.npz$")
ATTRIBUTION = ("Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
               "Claude-Session: https://claude.ai/code/session_01Bm1HYXBKtvVtXUBSQW4W5E")


def predictions(size: int) -> dict:
    """{power: {quantity: deviation in %}} of round 9's fixed-power fits at N = size (delta = C (N^-p - 100^-p) on
    N = 100's plateau)."""
    data = json.loads(ROUND9_EXTRAPOLATION.read_text(encoding="utf-8"))
    out = {}
    for power in ("1.0", "2.0"):
        out[power] = {}
        for quantity in ("u_min", "v_max"):
            entry = data[f"{quantity}_matched"]
            fixed = entry["fixed"][power]
            delta = fixed["coefficient"] * (size ** -fixed["power"] - 100.0 ** -fixed["power"])
            out[power][quantity] = 100.0 * (entry["base"] + delta)
    return out


def load_tube(path: pathlib.Path) -> dict:
    with np.load(path) as archive:
        return {"positions": archive["positions"].astype(np.float64),
                "velocities": archive["velocities"].astype(np.float64),
                "pressures": archive["pressures"].astype(np.float64),
                "volumes": archive["masses"].astype(np.float64) / archive["densities"].astype(np.float64),
                "time": float(archive["time"]), "level_time": float(archive["level_time"]),
                "step": int(archive["step"])}


def record_of(path: pathlib.Path, size: int) -> dict:
    data = reference()["extrema"]
    sample = load_tube(path)
    lower, upper = extrema_methods.box_of(size)
    values = extrema_of([sample], lower, upper, size, 4.0 / size)
    record = {"file": path.name, "step": sample["step"], "time": sample["time"], "level_time": sample["level_time"],
              "written": path.stat().st_mtime}
    for quantity in QUANTITIES:
        record[quantity] = 100.0 * deviation(values[quantity][METHOD]["value"], data[quantity]["value"])
        record[f"{quantity}_linear"] = 100.0 * deviation(values[quantity]["linear"]["value"], data[quantity]["value"])
    return record


def slopes(records: list, end: float) -> dict | None:
    """Straight-line slopes (points per unit) of u_min and v_max over the records with level time in
    [end - 6, end]; None before 6 units exist."""
    rows = [row for row in records if end - STEADY_SPAN - 1e-9 <= row["level_time"] <= end + 1e-9]
    if not rows or rows[0]["level_time"] > end - STEADY_SPAN + 0.55 or len(rows) < 5:
        return None
    times = np.array([row["level_time"] for row in rows])
    return {quantity: float(np.polyfit(times, np.array([row[quantity] for row in rows]), 1)[0])
            for quantity in ("u_min", "v_max")} | {"records": len(rows)}


def seconds_per_unit(records: list) -> float | None:
    rows = [row for row in records if row["level_time"] >= records[-1]["level_time"] - 2.0 - 1e-9]
    if len(rows) < 2 or rows[-1]["level_time"] - rows[0]["level_time"] < 0.9:
        return None
    return (rows[-1]["written"] - rows[0]["written"]) / (rows[-1]["level_time"] - rows[0]["level_time"])


def clock(unix: float) -> str:
    return datetime.datetime.fromtimestamp(unix).strftime("%m-%d %H:%M")


def progress_header(arguments, prediction: dict) -> str:
    p1, p2 = prediction["1.0"], prediction["2.0"]
    return (
        "# 第十轮进度：N = 280（本机双卡，从第九轮 N = 200 的续跑末态 t = 53.58 插值起步）\n\n"
        "- 每过 2 个本级时间单位追加一行，每行后单独提交本文件。偏差 = 直接拟合的极值相对原文表 10，单位 %；"
        "斜率单位是百分点 / 单位时间。\n"
        "- 平稳判据：最近 6 个单位里 u_min、v_max 的瞬时值各做直线拟合，两个 |斜率| 都 < 0.01，并且本级时间 ≥ 12，"
        "就停；本级时间到 20 还没平稳也停。\n"
        f"- 第九轮外推对 N = 280 的预测：1 阶 u_min {p1['u_min']:+.2f} % / v_max {p1['v_max']:+.2f} %；"
        f"2 阶 {p2['u_min']:+.2f} % / {p2['v_max']:+.2f} %。两者差 {abs(p1['u_min'] - p2['u_min']):.2f} / "
        f"{abs(p1['v_max'] - p2['v_max']):.2f} 个百分点。\n"
        "- 插值起步以后约 4 个单位有一次跳变、再回落约 8 个单位（第九轮 15.4），本级时间 12 以前的值不能拿来比预测。\n"
        f"- 运行目录 `{arguments.run}`。\n\n"
        "| 本级时间 | 累计时间 t | 墙钟 | u_min（最近 2 单位平均） | v_max（最近 2 单位平均） | 斜率 u_min / v_max（最近 6 单位） "
        "| 离 1 阶 / 2 阶预测（u_min；v_max） | 更接近 | 预计 |\n"
        "|---|---|---|---|---|---|---|---|---|\n")


def progress_row(records: list, mark: float, prediction: dict, note: str = "") -> str:
    last = records[-1]
    window = [row for row in records if mark - MEAN_SPAN - 1e-9 < row["level_time"] <= mark + 1e-9 and row["level_time"] > 0]
    means = {quantity: float(np.mean([row[quantity] for row in window])) for quantity in ("u_min", "v_max")}
    fitted = slopes(records, mark)
    slope_text = f"{fitted['u_min']:+.4f} / {fitted['v_max']:+.4f}" if fitted else "（不足 6 单位）"
    distance = {power: {quantity: means[quantity] - prediction[power][quantity] for quantity in ("u_min", "v_max")}
                for power in ("1.0", "2.0")}
    gap = {power: 0.5 * (abs(distance[power]["u_min"]) + abs(distance[power]["v_max"])) for power in ("1.0", "2.0")}
    closer = "1 阶" if gap["1.0"] < gap["2.0"] else "2 阶"
    if mark < STEADY_MINIMUM:
        closer += "（本级时间 < 12，仅参考）"
    distance_text = (f"{distance['1.0']['u_min']:+.2f} / {distance['2.0']['u_min']:+.2f}；"
                     f"{distance['1.0']['v_max']:+.2f} / {distance['2.0']['v_max']:+.2f}")
    rate = seconds_per_unit(records)
    if note:
        expected = note
    elif rate is None:
        expected = "—"
    else:
        earliest = last["written"] + max(STEADY_MINIMUM - last["level_time"], 0.5) * rate
        latest = last["written"] + max(LEVEL_CAP - last["level_time"], 0.0) * rate
        expected = (f"每单位 {rate / 3600.0:.2f} 小时；最早 {clock(earliest)}（本级时间 "
                    f"{max(STEADY_MINIMUM, last['level_time'] + 0.5):.1f}），最迟 {clock(latest)}（20）")
    return (f"| {mark:.1f} | {last['time']:.2f} | {clock(last['written'])} | {means['u_min']:+.2f} % "
            f"| {means['v_max']:+.2f} % | {slope_text} | {distance_text} | {closer} | {expected} |\n")


def commit_progress(message: str) -> None:
    relative = str(PROGRESS.relative_to(REPOSITORY_ROOT)).replace("\\", "/")
    for attempt in range(6):
        added = subprocess.run(["git", "add", "--", relative], cwd=REPOSITORY_ROOT, capture_output=True, text=True)
        done = subprocess.run(["git", "commit", "-q", "-m", f"{message}\n\n{ATTRIBUTION}", "--", relative],
                              cwd=REPOSITORY_ROOT, capture_output=True, text=True)
        if added.returncode == 0 and done.returncode == 0:
            return
        if "nothing to commit" in (done.stdout + done.stderr):
            return
        time.sleep(5.0 * (attempt + 1))
    print(f"[round10] commit of the progress file failed: {done.stderr.strip()}", flush=True)


def trim_checkpoints(run: pathlib.Path) -> None:
    """Keep the newest complete checkpoint (listed in the manifest) of the run; delete the older ones."""
    directory = run / "checkpoints"
    manifest = directory / "manifest.jsonl"
    if not manifest.exists():
        return
    listed = {json.loads(line)["file"] for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()}
    complete = sorted(path for path in directory.glob("c*.npz") if CHECKPOINT_NAME.match(path.name)
                      and path.name in listed)
    for old in complete[:-1]:
        old.unlink()
        print(f"[round10] deleted the older checkpoint {old.name} (kept {complete[-1].name})", flush=True)


def monitor_main(arguments) -> int:
    run = (REPOSITORY_ROOT / arguments.run).resolve()
    stop_file = (REPOSITORY_ROOT / arguments.stop_file).resolve()
    series_path = run / "extrema.jsonl"
    prediction = predictions(arguments.size)
    started = time.time()
    records = []
    if series_path.exists():
        records = [json.loads(line) for line in series_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = {row["file"] for row in records}
    next_mark = PROGRESS_EVERY * (int(records[-1]["level_time"] // PROGRESS_EVERY) + 1) if records else PROGRESS_EVERY
    if not PROGRESS.exists():
        PROGRESS.write_text(progress_header(arguments, prediction), encoding="utf-8", newline="\n")
        if arguments.commit:
            commit_progress("round 10 progress: header (N = 280 started)")
    stopped = stop_file.exists()
    while True:
        summary = run / "summary.json"
        finished = summary.exists() and summary.stat().st_mtime > started
        fresh = sorted(path for path in (run / "tubes").glob("t*.npz")
                       if not path.name.endswith(".partial.npz") and path.name not in seen) \
            if (run / "tubes").exists() else []
        for path in fresh:
            record = record_of(path, arguments.size)
            seen.add(path.name)
            if record["level_time"] <= 0.0:
                continue                                    # the interpolated start itself
            records.append(record)
            with open(series_path, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"[round10] level time {record['level_time']:.2f}: u_min {record['u_min']:+.3f} %, "
                  f"v_max {record['v_max']:+.3f} %", flush=True)
            fitted = slopes(records, record["level_time"])
            if not stopped and record["level_time"] >= STEADY_MINIMUM - 1e-9 and fitted is not None \
                    and abs(fitted["u_min"]) < STEADY_SLOPE and abs(fitted["v_max"]) < STEADY_SLOPE:
                stop_file.write_text(json.dumps({"level_time": record["level_time"], "time": record["time"],
                                                 "slopes": fitted, "unix": time.time()}), encoding="utf-8")
                stopped = True
                with open(PROGRESS, "a", encoding="utf-8", newline="\n") as handle:
                    handle.write(progress_row(records, record["level_time"], prediction,
                                              note="**平稳，停**（box_run 在下一个记录停下，写检查点）"))
                if arguments.commit:
                    commit_progress(f"round 10 progress: steady at level time {record['level_time']:.1f}")
                next_mark = float("inf")
            while record["level_time"] >= next_mark - 1e-9:
                with open(PROGRESS, "a", encoding="utf-8", newline="\n") as handle:
                    handle.write(progress_row(records, next_mark, prediction))
                if arguments.commit:
                    commit_progress(f"round 10 progress: level time {next_mark:.0f}")
                next_mark += PROGRESS_EVERY
        trim_checkpoints(run)
        if finished and not fresh:
            final = json.loads(summary.read_text(encoding="utf-8"))
            with open(PROGRESS, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(f"\n运行结束：{clock(summary.stat().st_mtime)}，本级时间 {final.get('level_time', 0):.2f}"
                             f"（t = {final.get('time', 0):.2f}），结束原因 {final.get('end_reason')}。"
                             "这一级的结果和外推见报告第 16 节。\n")
            if arguments.commit:
                commit_progress("round 10 progress: run ended")
            return 0
        time.sleep(arguments.poll)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    monitor = sub.add_parser("monitor")
    monitor.add_argument("--run", required=True, help="the box_run directory (relative to the repository)")
    monitor.add_argument("--stop-file", required=True)
    monitor.add_argument("--size", type=int, default=280)
    monitor.add_argument("--poll", type=float, default=30.0)
    monitor.add_argument("--commit", action="store_true", help="commit the progress file alone after every row")
    arguments = parser.parse_args()
    if arguments.action == "monitor":
        return monitor_main(arguments)
    return 1


if __name__ == "__main__":
    sys.exit(main())
