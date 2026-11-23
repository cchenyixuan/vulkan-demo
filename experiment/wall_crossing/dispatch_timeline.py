"""
dispatch_timeline.py - claude/wall_store_report.md section 10.4: when the near-wall and the interior dispatch of a
wall_boundary crossing split pass run, per workgroup, without timestamps between them (they keep overlapping as in
the step). The *_timeline builds (V7_DISPATCH_TIMELINE=1, compile_shaders_v7.TIMELINE_VARIANTS) write, per workgroup,
its SM id and the device's realtime clock (ns) at its start and end; this script runs the single-GPU step (one
command buffer, no bench timestamps, defrag at its cadence), and after the warmup reads the records back after
several steps.

    .venv/Scripts/python.exe experiment/wall_crossing/dispatch_timeline.py run --case <crossing case.yaml> \
        --label n500_default --device 1 [--env NAME=VALUE ...] [--warmup 2000] [--samples 20]
    .venv/Scripts/python.exe experiment/wall_crossing/dispatch_timeline.py tables

run    writes logs/wall_store/round4/timeline/<label>.npz (per sample and slot: SM id, start, end per workgroup) and
       <label>.json (the per-sample measures below, medians over the samples).
tables the summary tables claude/wall_store/tables/round4_timeline.md and the figure
       claude/wall_store/figures/round4_timeline_<label>.png (one sample: every workgroup of the fused pass as a bar on
       its SM's row, by dispatch).

Measures per pass (fused, force) and sample: each dispatch's first start and last end (relative to the pass's first
start), the overlap of the two spans, the pass span; the near-wall workgroups that did work (longer than 10 us: the
others hold no near-wall particle and leave at once) with their count and median duration; per SM, the time a busy
near-wall workgroup and an interior workgroup were resident together, as a share of the busy near-wall time; how many
SMs ran a busy near-wall workgroup.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = REPOSITORY_ROOT / "logs" / "wall_store" / "round4" / "timeline"
TABLES = REPOSITORY_ROOT / "claude" / "wall_store" / "tables"
FIGURES = REPOSITORY_ROOT / "claude" / "wall_store" / "figures"
BUSY_NANOSECONDS = 10000.0   # a near-wall workgroup without a near-wall particle returns within ~3 us
NSYS = "C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.5.2/target-windows-x64/nsys.exe"
SLOTS = {"fused_interior": 0, "fused_near_wall": 1, "force_interior": 2, "force_near_wall": 3}


def run(arguments) -> int:
    for item in arguments.env:
        name, _, value = item.partition("=")
        os.environ[name] = value
    os.environ["V7_DISPATCH_TIMELINE"] = "0" if arguments.plain else "1"
    os.environ.setdefault("VK_LOADER_LAYERS_DISABLE", "VK_LAYER_KHRONOS_validation")
    sys.path.insert(0, str(REPOSITORY_ROOT))
    os.chdir(REPOSITORY_ROOT)
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7.utils.simulator_v7 import SphSimulatorV7
    from experiment.v7.utils.vulkan_context_v7 import VulkanContextV7
    case = load_case_v7(arguments.case)
    cadence = int(case.numerics.defrag_cadence)
    context = VulkanContextV7.create(device_index=arguments.device, enable_validation=False,
                                     application_name="dispatch_timeline")
    simulator = SphSimulatorV7(context, case)
    groups = simulator._per_own_particle_dispatch_count()
    simulator.bootstrap()
    simulator.prepare_step_single_cmd_buffer()
    samples = []
    step = 0
    total = arguments.warmup + arguments.samples * arguments.spacing
    nsys_started = False
    while step < total:
        if arguments.nsys_session and step == arguments.warmup and not nsys_started:
            # Nsight Systems' collection of the steady steps only (the run was started with nsys launch)
            import subprocess
            subprocess.run([NSYS, "start", f"--session={arguments.nsys_session}", "-o", arguments.nsys_output,
                            "--gpu-metrics-devices=0", "--gpu-metrics-set=gb20x", "--gpu-metrics-frequency=200000",
                            "--sample=none", "--cpuctxsw=none",
                            "--force-overwrite=true"], check=True)
            nsys_started = True
            total = step + arguments.nsys_steps
        simulator.submit_step_single_and_wait()
        step += 1
        if step % cadence == 0:
            simulator.submit_defrag_and_wait()
        if arguments.plain:
            continue
        if step > arguments.warmup and (step - arguments.warmup) % arguments.spacing == 0 and step % cadence != 0:
            raw = simulator.readback_buffers_batch(["dispatch_timeline"])["dispatch_timeline"]
            words = np.frombuffer(raw, dtype=np.uint32).reshape(4, -1, 8)
            start = words[:, :, 1].astype(np.uint64) | (words[:, :, 2].astype(np.uint64) << np.uint64(32))
            end = words[:, :, 3].astype(np.uint64) | (words[:, :, 4].astype(np.uint64) << np.uint64(32))
            samples.append({"step": step, "sm": words[:, :, 0].copy(), "start": start, "end": end})
    if nsys_started:
        import subprocess
        subprocess.run([NSYS, "stop", f"--session={arguments.nsys_session}"], check=True)
    simulator.destroy()
    context.destroy()
    if arguments.plain:
        print(f"[dispatch_timeline] plain build, {step} steps (for Nsight Systems)")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{arguments.label}.npz", steps=np.array([sample["step"] for sample in samples]),
                        sm=np.stack([sample["sm"] for sample in samples]),
                        start=np.stack([sample["start"] for sample in samples]),
                        end=np.stack([sample["end"] for sample in samples]), groups=groups)
    measures = [measure(sample, groups) for sample in samples]
    summary = {"label": arguments.label, "case": arguments.case, "environment": arguments.env, "groups": groups,
               "samples": len(samples), "median": {key: float(np.median([entry[key] for entry in measures]))
                                                   for key in measures[0]}}
    (OUT / f"{arguments.label}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary["median"], indent=1))
    return 0


def intervals(sample: dict, slot: int, valid_groups: int) -> tuple:
    start = sample["start"][slot][:valid_groups].astype(np.int64)
    end = sample["end"][slot][:valid_groups].astype(np.int64)
    sm = sample["sm"][slot][:valid_groups].astype(np.int64)
    written = (start > 0) & (end >= start)
    return start[written], end[written], sm[written]


def merge(starts: np.ndarray, ends: np.ndarray) -> list:
    """The union of intervals as a sorted list of disjoint (start, end)."""
    if starts.size == 0:
        return []
    order = np.argsort(starts)
    merged = [[starts[order[0]], ends[order[0]]]]
    for begin, finish in zip(starts[order[1:]], ends[order[1:]]):
        if begin > merged[-1][1]:
            merged.append([begin, finish])
        else:
            merged[-1][1] = max(merged[-1][1], finish)
    return [(float(begin), float(finish)) for begin, finish in merged]


def union_length(starts: np.ndarray, ends: np.ndarray) -> float:
    if starts.size == 0:
        return 0.0
    order = np.argsort(starts)
    total, current_start, current_end = 0.0, starts[order[0]], ends[order[0]]
    for begin, finish in zip(starts[order[1:]], ends[order[1:]]):
        if begin > current_end:
            total += current_end - current_start
            current_start, current_end = begin, finish
        else:
            current_end = max(current_end, finish)
    return float(total + current_end - current_start)


def measure(sample: dict, groups: int) -> dict:
    result = {}
    for kernel, interior_slot, near_slot in (("fused", 0, 1), ("force", 2, 3)):
        near_capacity = sample["start"].shape[1]
        interior = intervals(sample, interior_slot, groups)
        near = intervals(sample, near_slot, near_capacity)
        origin = min(interior[0].min(), near[0].min())
        interior_start, interior_end = interior[0].min() - origin, interior[1].max() - origin
        near_start, near_end = near[0].min() - origin, near[1].max() - origin
        busy = (near[1] - near[0]) > BUSY_NANOSECONDS
        overlap = max(0.0, min(interior_end, near_end) - max(interior_start, near_start))
        co_resident, busy_time = 0.0, 0.0
        for sm in np.unique(near[2][busy]):
            near_on = busy & (near[2] == sm)
            interior_on = interior[2] == sm
            merged = merge(near[0][near_on], near[1][near_on])
            busy_time += sum(finish - begin for begin, finish in merged)
            # time when a busy near-wall workgroup and an interior workgroup were both resident on this SM
            pieces_low, pieces_high = [], []
            for begin, finish in merged:
                lo = np.maximum(interior[0][interior_on], begin)
                hi = np.minimum(interior[1][interior_on], finish)
                keep = hi > lo
                pieces_low.append(lo[keep])
                pieces_high.append(hi[keep])
            if pieces_low:
                co_resident += union_length(np.concatenate(pieces_low), np.concatenate(pieces_high))
        busy_starts = near[0][busy] - origin
        result.update({
            f"{kernel}_interior_start_us": interior_start / 1e3, f"{kernel}_interior_end_us": interior_end / 1e3,
            f"{kernel}_near_wall_start_us": near_start / 1e3, f"{kernel}_near_wall_end_us": near_end / 1e3,
            f"{kernel}_busy_near_wall_first_start_us": float(busy_starts.min()) / 1e3 if busy_starts.size else 0.0,
            f"{kernel}_busy_near_wall_last_start_us": float(busy_starts.max()) / 1e3 if busy_starts.size else 0.0,
            f"{kernel}_overlap_us": overlap / 1e3,
            f"{kernel}_span_us": (max(interior_end, near_end)) / 1e3,
            f"{kernel}_busy_near_wall_groups": float(busy.sum()),
            f"{kernel}_busy_near_wall_duration_us": float(np.median(near[1][busy] - near[0][busy])) / 1e3
            if busy.any() else 0.0,
            f"{kernel}_busy_near_wall_sms": float(np.unique(near[2][busy]).size),
            f"{kernel}_co_resident_share": co_resident / busy_time if busy_time else 0.0,
            f"{kernel}_interior_groups_per_sm": float(np.bincount(interior[2]).max()),
        })
    return result


# the builds of round 4's runs (section 10.3): label suffix -> text; "default" was the default when the runs were made
# (items (a), (d), the wall pairs of S after the store), "combined" is the default since section 10.4
BUILD_TEXT = {"base": "262156d 的缺省", "default": "(a)+(d)，S 在存储后（ad）", "ae": "(a)+(e)+(d)，S 在存储后",
              "ae_scope_interior": "(e) 只在内部管线（scope interior）", "combined": "(a)+(d)，S 合进循环（现缺省）"}


def tables(arguments) -> int:
    labels = [f"n{size}_{build}" for size in (500, 1000) for build in BUILD_TEXT
              if (OUT / f"n{size}_{build}.npz").exists()]
    lines = [f"| 算例 | 构建 | 遍 | 内部调度：开始–结束 µs | 近壁调度：开始–结束 µs | 干活的近壁工作组（> {BUSY_NANOSECONDS / 1e3:.0f} µs）："
             "个数 · 中位时长 µs · 开始于 µs（最早–最晚） | 两次调度的时间重叠 µs | 整遍 µs | 干活的近壁工作组与内部工作组同驻一个 SM 的时间占比 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for label in labels:
        # the measures again from the records (the run's json holds them with the threshold of its day)
        data = np.load(OUT / f"{label}.npz")
        groups = int(data["groups"])
        measures = [measure({"sm": data["sm"][index], "start": data["start"][index], "end": data["end"][index]}, groups)
                    for index in range(data["sm"].shape[0])]
        median = {key: float(np.median([entry[key] for entry in measures])) for key in measures[0]}
        size, build = label.split("_", 1)
        for kernel, text in (("fused", "融合遍"), ("force", "受力")):
            lines.append(
                f"| {size[1:]}² | {BUILD_TEXT[build]} | {text} | "
                f"{median[f'{kernel}_interior_start_us']:.1f}–{median[f'{kernel}_interior_end_us']:.1f} | "
                f"{median[f'{kernel}_near_wall_start_us']:.1f}–{median[f'{kernel}_near_wall_end_us']:.1f} | "
                f"{median[f'{kernel}_busy_near_wall_groups']:.0f} · {median[f'{kernel}_busy_near_wall_duration_us']:.1f} · "
                f"{median[f'{kernel}_busy_near_wall_first_start_us']:.1f}–{median[f'{kernel}_busy_near_wall_last_start_us']:.1f} | "
                f"{median[f'{kernel}_overlap_us']:.1f} | {median[f'{kernel}_span_us']:.1f} | "
                f"{100 * median[f'{kernel}_co_resident_share']:.0f} % |")
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "round4_timeline.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    if arguments.figures:
        draw([label for label in arguments.figures.split(",") if label in labels], arguments.figure_name)
    return 0


WARPS_PER_WORKGROUP = 2          # 64 invocations per workgroup (spec const 51)
STREAMING_MULTIPROCESSORS = 170  # RTX 5090 (GB202)


def resident_warps(starts: np.ndarray, ends: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Warps resident per SM (mean over the SMs) at the grid times, from workgroup start / end."""
    events = np.concatenate([starts, ends])
    steps = np.concatenate([np.ones_like(starts), -np.ones_like(ends)])
    order = np.argsort(events, kind="stable")
    cumulative = np.cumsum(steps[order])
    index = np.searchsorted(events[order], grid, side="right") - 1
    counts = np.where(index >= 0, cumulative[np.clip(index, 0, None)], 0)
    return counts * WARPS_PER_WORKGROUP / STREAMING_MULTIPROCESSORS


def draw(labels: list, name: str) -> None:
    """Per build (row) and pass (column): the warps resident per SM over the pass, of the interior dispatch and of
    the near-wall dispatch's working workgroups (longer than 10 us), from one sample's records."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(REPOSITORY_ROOT))
    from experiment.wall_crossing.report_figures import INK_PRIMARY, INK_SECONDARY, save, setup_style
    setup_style()
    colours = {"interior": "#2a78d6", "near_wall": "#eb6834"}
    figure, axes = plt.subplots(len(labels), 2, figsize=(12.0, 1.9 * len(labels) + 0.9), sharey=True, squeeze=False)
    for row, label in enumerate(labels):
        data = np.load(OUT / f"{label}.npz")
        groups = int(data["groups"])
        sample = {"sm": data["sm"][0], "start": data["start"][0], "end": data["end"][0]}
        for column, (kernel, interior_slot, near_slot, text) in enumerate((("fused", 0, 1, "融合遍"),
                                                                           ("force", 2, 3, "受力"))):
            axis = axes[row][column]
            interior = intervals(sample, interior_slot, groups)
            near = intervals(sample, near_slot, sample["start"].shape[1])
            busy = (near[1] - near[0]) > BUSY_NANOSECONDS
            origin = min(interior[0].min(), near[0].min())
            last = max(interior[1].max(), near[1].max())
            grid = np.linspace(origin, last, 600)
            time = (grid - origin) / 1e3
            axis.plot(time, resident_warps(interior[0], interior[1], grid), color=colours["interior"], linewidth=1.6)
            axis.plot(time, resident_warps(near[0][busy], near[1][busy], grid), color=colours["near_wall"],
                      linewidth=1.6)
            axis.set_xlim(0, time[-1])
            axis.set_ylim(0, 48)
            size, build = label.split("_", 1)
            axis.set_title(f"{size[1:]}² · {BUILD_TEXT.get(build, build)} · {text}（{time[-1]:.0f} µs）", fontsize=9.5,
                           loc="left", color=INK_PRIMARY)
            if column == 0:
                axis.set_ylabel("线程束 / SM")
            if row == len(labels) - 1:
                axis.set_xlabel("从这一遍第一个工作组开始的时间 µs")
    from matplotlib.lines import Line2D
    figure.suptitle("两次调度在一遍里的驻留：每个 SM 平均驻留的线程束（着色器内的实时时钟，调度之间没有计时）",
                    x=0.01, y=0.995, ha="left", va="top", fontsize=12, fontweight="bold", color=INK_PRIMARY)
    figure.legend([Line2D([], [], color=colours["interior"], linewidth=1.6),
                   Line2D([], [], color=colours["near_wall"], linewidth=1.6)],
                  ["内部调度", "近壁调度里干活的工作组（> 10 µs；其余立即返回）"], loc="upper left",
                  bbox_to_anchor=(0.01, 1.0 - 0.42 / figure.get_figheight()), ncol=2, labelcolor=INK_PRIMARY)
    figure.text(0.99, 1.0 - 0.45 / figure.get_figheight(), "上限 48 线程束 / SM；每个工作组 2 个线程束",
                ha="right", va="top", fontsize=8.5, color=INK_SECONDARY)
    figure.tight_layout(rect=(0, 0, 1, 1.0 - 0.65 / figure.get_figheight()))
    FIGURES.mkdir(parents=True, exist_ok=True)
    save(figure, FIGURES / f"round4_timeline_{name}.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--case", required=True)
    run_parser.add_argument("--label", required=True)
    run_parser.add_argument("--device", type=int, default=1)
    run_parser.add_argument("--env", action="append", default=[])
    run_parser.add_argument("--warmup", type=int, default=2000)
    run_parser.add_argument("--samples", type=int, default=20)
    run_parser.add_argument("--spacing", type=int, default=37)
    run_parser.add_argument("--plain", action="store_true",
                            help="the plain builds (no records): the steps only, to run under Nsight Systems")
    run_parser.add_argument("--nsys-session", default="",
                            help="with --plain under nsys launch --session-new=<name>: nsys start after the warmup, "
                                 "--nsys-steps steps, nsys stop")
    run_parser.add_argument("--nsys-output", default="")
    run_parser.add_argument("--nsys-steps", type=int, default=400)
    tables_parser = sub.add_parser("tables")
    tables_parser.add_argument("--figures", default="")
    tables_parser.add_argument("--figure-name", default="builds")
    arguments = parser.parse_args()
    return run(arguments) if arguments.action == "run" else tables(arguments)


if __name__ == "__main__":
    sys.exit(main())
