"""
_summarize_low_load.py — tables for the low-load sweep (10k - 1M).

Reads campaign directories written by _run_two_hop_campaign.py and prints
three Markdown tables:

  1. cases: particle counts, voxel columns, K=2 split, force-band share.
  2. fps per (case, configuration): both transports, ratio, and the CPU
     accounting of the pipelined loop (is the main thread or the GPU pacing
     the pipeline?).
  3. mechanism per (case, transport): b_to_c_gap distribution, exposed
     frames, upload -> phase C slack, transfer-chain legs, worker segments.

Usage:
    .venv/Scripts/python.exe experiment/two_hop/_summarize_low_load.py \\
        --main logs/two_hop_experiment/low_load_main_20260928 \\
        --cascade-off logs/two_hop_experiment/low_load_cascade_off_20260928 \\
        --output summary.md
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_TRANSPORTS = ("three_hop", "two_hop")
_FORCE_BAND_COLUMNS = 4
# The main thread paces the pipeline when it finds the awaited frame already
# finished on most frames; below this share the GPU is the pacer.
_CPU_BOUND_FRAMES_WITHOUT_WAIT_PERCENT = 50.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", required=True,
                        help="campaign directory of the main configuration")
    parser.add_argument("--cascade-off", default=None)
    parser.add_argument("--extra", action="append", default=[],
                        metavar="NAME=DIRECTORY",
                        help="additional fps group shown in table 2 "
                             "(e.g. depth3=..., switch_5ms=...)")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_results(directories: str) -> list:
    """``directories``: one campaign directory, or several joined by ','
    (a group measured in more than one campaign invocation)."""
    results = []
    for directory in directories.split(","):
        path = pathlib.Path(directory) / "results.jsonl"
        results += [json.loads(line) for line in path.read_text().splitlines()
                    if line.strip()]
    return results


def fluid_particle_count(case_path: str) -> int:
    domain = (_REPO_ROOT / case_path).parent / "domain.obj"
    with open(domain) as handle:
        return sum(1 for line in handle if line.startswith("v "))


def median_or_none(values: list):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def text(value, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def fps_trials(results: list, case_label: str, transport: str) -> list:
    return [result for result in results
            if result["case_label"] == case_label
            and result["transport"] == transport
            and not result["instrumented"] and result.get("steady_fps")]


def instrumented_run(results: list, case_label: str, transport: str) -> dict:
    runs = [result for result in results
            if result["case_label"] == case_label
            and result["transport"] == transport
            and result["instrumented"] and "gpu" in result]
    return runs[-1] if runs else {}


def fps_cells(results: list, case_label: str) -> dict:
    """Median / range per transport, ratio, overlap, loop accounting."""
    cells: dict = {}
    for transport in _TRANSPORTS:
        trials = fps_trials(results, case_label, transport)
        if not trials:
            continue
        values = [trial["steady_fps"] for trial in trials]
        loops = [trial["loop"] for trial in trials if trial.get("loop")]
        cells[transport] = {
            "trials": len(values),
            "median": statistics.median(values),
            "low": min(values), "high": max(values),
            "invalid": sum(1 for trial in trials if not trial.get("valid")),
            "submit_p50": median_or_none(
                [loop["submit_us"]["p50"] for loop in loops]),
            "submit_mean": median_or_none(
                [loop["submit_us"]["mean"] for loop in loops]),
            "wait_p50": median_or_none(
                [loop["wait_us"]["p50"] for loop in loops]),
            "period_p50": median_or_none(
                [loop["period_us"]["p50"] for loop in loops]),
            "submit_share": median_or_none(
                [loop["submit_share_of_period"] for loop in loops]),
            "without_wait": median_or_none(
                [loop["frames_without_wait_percent"] for loop in loops]),
        }
    if len(cells) == 2:
        three, two = cells["three_hop"], cells["two_hop"]
        cells["ratio_percent"] = two["median"] / three["median"] * 100.0
        cells["overlap"] = (two["low"] <= three["high"]
                            and three["low"] <= two["high"])
    return cells


def pacer(cell: dict) -> str:
    if cell.get("without_wait") is None:
        return "—"
    return ("CPU" if cell["without_wait"]
            >= _CPU_BOUND_FRAMES_WITHOUT_WAIT_PERCENT else "GPU")


def worker_statistic(trials: list, segment: str, statistic: str):
    """Median over trials of (median over the two workers)."""
    per_trial = []
    for trial in trials:
        values = [segments[segment][statistic]
                  for segments in trial.get("worker_segments_us", {}).values()
                  if segment in segments]
        if values:
            per_trial.append(statistics.median(values))
    return statistics.median(per_trial) if per_trial else None


def gpu_statistic(run: dict, metric: str, statistic: str = "p50"):
    values = [sim[metric][statistic] for sim in run.get("gpu", [])
              if metric in sim]
    return statistics.median(values) if values else None


def main() -> int:
    args = parse_args()
    groups = {"main": load_results(args.main)}
    if args.cascade_off:
        groups["cascade_off"] = load_results(args.cascade_off)
    for entry in args.extra:
        name, _, directory = entry.partition("=")
        groups[name] = load_results(directory)
    main_results = groups["main"]
    case_labels = list(dict.fromkeys(
        result["case_label"] for result in main_results))
    # Frame period of the pipelined loop and GPU busy time per frame, for the
    # "what caps fps at low load" question.
    period_lines = [
        "",
        "### 帧周期构成（three_hop，µs）",
        "",
        "周期与提交来自 depth 2 的 fps trial（各 trial 的中位数）；GPU 三个 "
        "phase 来自 depth 1 的 anatomy（两张卡 p50 的中位数）。",
        "",
        "| case | 配置 | 周期 p50 | 周期均值 | 提交 p50 | 等待 p50 | "
        "phase A | phase B | phase C | A+B+C | A+B+C 占周期 p50 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case_label in case_labels:
        for group_name in ("main", "cascade_off"):
            if group_name not in groups:
                continue
            trials = [trial for trial in fps_trials(
                groups[group_name], case_label, "three_hop")
                if trial.get("loop")]
            run = instrumented_run(groups[group_name], case_label, "three_hop")
            if not trials or not run:
                continue
            period = statistics.median(
                trial["loop"]["period_us"]["p50"] for trial in trials)
            period_mean = statistics.median(
                trial["loop"]["period_us"]["mean"] for trial in trials)
            submit = statistics.median(
                trial["loop"]["submit_us"]["p50"] for trial in trials)
            wait = statistics.median(
                trial["loop"]["wait_us"]["p50"] for trial in trials)
            phases = [gpu_statistic(run, f"phase_{name}_us")
                      for name in ("a", "b", "c")]
            busy = sum(phases)
            period_lines.append(
                f"| {case_label} | {group_name} | {period:.0f} | "
                f"{period_mean:.0f} | {submit:.0f} | {wait:.0f} | "
                f"{phases[0]:.0f} | {phases[1]:.0f} | {phases[2]:.0f} | "
                f"{busy:.0f} | {busy / period * 100:.0f} % |")
    lines: list = []

    # ---- Table 1: cases -----------------------------------------------------
    lines += [
        "### 表 1：算例与切分",
        "",
        "| case | 流体粒子 | 总粒子 | voxel 列数 | own 列数（slab 0 / 1） | "
        "own 粒子（slab 0 / 1） | force band 占比 | minimum_own_columns | "
        "staging KB/方向 | warmup / 总步数 | 测量窗口 (s) |",
        "|---|---:|---:|---:|---|---|---|---:|---:|---|---:|",
    ]
    for case_label in case_labels:
        trials = fps_trials(main_results, case_label, "three_hop")
        if not trials:
            continue
        first = trials[0]
        partition = first["partition"]
        own = partition["own_columns"]
        staging = [size for per_sim in first["staging_bytes_per_direction"]
                   for size in per_sim.values()]
        window = statistics.median(
            trial["steady_frames"] / trial["steady_fps"] for trial in trials)
        relaxed = "（放宽）" if partition["minimum_own_columns_relaxed"] else ""
        lines.append(
            f"| {case_label} | {fluid_particle_count(first['case']):,} | "
            f"{first['particles']:,} | {partition['grid_columns']} | "
            f"{own[0]} / {own[1]} | "
            f"{partition['own_particles'][0]:,} / "
            f"{partition['own_particles'][1]:,} | "
            f"{_FORCE_BAND_COLUMNS / own[0] * 100:.1f} % / "
            f"{_FORCE_BAND_COLUMNS / own[1] * 100:.1f} % | "
            f"{partition['minimum_own_columns']}{relaxed} | "
            f"{statistics.median(staging) / 1024:.0f} | "
            f"{first['warmup']} / {first['max_steps']} | {window:.1f} |")

    # ---- Table 2: fps -------------------------------------------------------
    lines += [
        "",
        "### 表 2：fps（depth 以各组为准，不挂 GPU 计时器）",
        "",
        "提交 = 主线程在 `_submit_frame` 里的时间；无等待帧 = 主线程去等 "
        "frame_done 时该帧已经完成的比例。",
        "",
        "| case | 配置 | three_hop 中位数 | three_hop 极差 | two_hop 中位数 | "
        "two_hop 极差 | two / three | 极差重叠 | 提交 p50 (µs) | "
        "提交占周期 | 无等待帧 | 节拍由谁定 |",
        "|---|---|---:|---|---:|---|---:|---|---:|---:|---:|---|",
    ]
    for case_label in case_labels:
        for group_name, results in groups.items():
            cells = fps_cells(results, case_label)
            if "three_hop" not in cells or "two_hop" not in cells:
                continue
            three, two = cells["three_hop"], cells["two_hop"]
            share = (f"{three['submit_share'] * 100:.1f} %"
                     if three["submit_share"] is not None else "—")
            without_wait = (f"{three['without_wait']:.2f} %"
                            if three["without_wait"] is not None else "—")
            flag = ""
            if three["invalid"] or two["invalid"]:
                flag = f" ⚠ {three['invalid'] + two['invalid']} 次无效"
            lines.append(
                f"| {case_label} | {group_name}{flag} | "
                f"{three['median']:.1f} | "
                f"{three['low']:.1f}–{three['high']:.1f} | "
                f"{two['median']:.1f} | {two['low']:.1f}–{two['high']:.1f} | "
                f"{cells['ratio_percent']:.2f} % | "
                f"{'是' if cells['overlap'] else '否'} | "
                f"{text(three['submit_p50'], 0)} | {share} | {without_wait} | "
                f"{pacer(three)} |")

    # ---- Table 3: mechanism -------------------------------------------------
    for group_name, results in groups.items():
        if not any(result["instrumented"] for result in results):
            continue
        lines += [
            "",
            f"### 表 3（{group_name}）：暴露量与传输链（µs）",
            "",
            "GPU 列来自 depth 1 的逐帧 anatomy：gap 统计两张卡的全部帧，其余是"
            "两张卡各自 p50 的中位数。worker 列来自 depth 2 的 fps trial，"
            "是各 trial、两个 worker 的中位数。",
            "",
            "| case | transport | anatomy 帧数 | gap p50 | gap p95 | gap p99 | "
            "暴露帧 >50 µs | 暴露帧 >20 µs | slack p50 | slack p5 | phase B | "
            "链长 p50 | readback 调度间隔 | readback DMA | host 段 | "
            "upload DMA |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            "---:|---:|---:|",
        ]
        for case_label in case_labels:
            for transport in _TRANSPORTS:
                run = instrumented_run(results, case_label, transport)
                if not run:
                    continue
                gap = run["b_to_c_gap_us"]
                lines.append(
                    f"| {case_label} | {transport} | {gap['samples'] // 2} | "
                    f"{gap['p50']:.1f} | {gap['p95']:.1f} | {gap['p99']:.1f} | "
                    f"{gap['exposed_over_50us_percent']:.2f} % | "
                    f"{gap['exposed_over_20us_percent']:.2f} % | "
                    f"{text(gpu_statistic(run, 'upload_to_c_gap_us'))} | "
                    f"{text(gpu_statistic(run, 'upload_to_c_gap_us', 'p5'))} | "
                    f"{text(gpu_statistic(run, 'phase_b_us'))} | "
                    f"{text(gpu_statistic(run, 'chain_us'))} | "
                    f"{text(gpu_statistic(run, 'readback_sched_gap_us'))} | "
                    f"{text(gpu_statistic(run, 'readback_dma_us'))} | "
                    f"{text(gpu_statistic(run, 'chain_host_share_us'))} | "
                    f"{text(gpu_statistic(run, 'upload_dma_us'))} |")
        lines += [
            "",
            f"worker 分段（{group_name}，p50 / p95，µs）：",
            "",
            "| case | transport | dest guard | upload guard | copy | signal | "
            "等 upload | consumed signal |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for case_label in case_labels:
            for transport in _TRANSPORTS:
                trials = fps_trials(results, case_label, transport)
                if not trials:
                    continue

                def cell(segment: str) -> str:
                    median = worker_statistic(trials, segment, "p50")
                    tail = worker_statistic(trials, segment, "p95")
                    if median is None:
                        return "—"
                    return f"{median:.1f} / {tail:.1f}"

                lines.append(
                    f"| {case_label} | {transport} | {cell('dest_guard')} | "
                    f"{cell('upload_guard')} | {cell('copy')} | "
                    f"{cell('signal')} | {cell('upload_wait')} | "
                    f"{cell('consumed_signal')} |")

    lines += period_lines

    # ---- Validity -----------------------------------------------------------
    lines += ["", "### 正确性", "",
              "| 组 | 运行数 | drift ≠ 0 | stamp 错误 | upload 期间覆盖 | "
              "install drop | seam 失败 | 无效运行 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group_name, results in groups.items():
        lines.append(
            f"| {group_name} | {len(results)} | "
            f"{sum(1 for r in results if r.get('drift') != 0)} | "
            f"{sum((r.get('stamp_errors_gpu') or 0) + (r.get('stamp_errors_host') or 0) for r in results)} | "
            f"{sum(r.get('overwrite_errors_host') or 0 for r in results)} | "
            f"{sum(r.get('install_tail_drops') or 0 for r in results)} | "
            f"{sum(1 for r in results if r.get('seam_ok') is False)} | "
            f"{sum(1 for r in results if not r.get('valid'))} |")

    output = "\n".join(lines) + "\n"
    sys.stdout.reconfigure(encoding="utf-8")
    print(output)
    if args.output:
        pathlib.Path(args.output).write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
