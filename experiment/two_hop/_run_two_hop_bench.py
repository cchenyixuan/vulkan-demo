"""
_run_two_hop_bench.py — ONE runner for both ghost transports.

    --transport three_hop   V5 as deployed: private stagings + worker memcpy
    --transport two_hop     shared host blocks + semaphore-relay worker

Everything else is the same code path for both: K=2 ChainPartition,
per-direction sync scheme, ChainOrchestratorV5 frame loop (the two-hop
orchestrator subclass only swaps the worker class), same environment
switches, same validation at the end.

Two measurement modes:

  default          run_pipelined at --depth (2): wall-clock steady fps, no
                   GPU timers attached (the pipelined loop cannot read
                   per-frame GPU timestamps — the frame in flight overwrites
                   the query slots). Worker segment timestamps are host-side
                   and are collected for every frame.
  --instrumented   synchronous depth-1 step() with compute + transfer
                   BenchTimers: per-frame b_to_c_gap, DMA durations,
                   upload -> phase C slack, readback scheduling gap.

Every trial ends with: final defrag, alive conservation (drift), GPU + host
frame-stamp error counts, and the seam-integrity check.

Usage:
    VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation \\
    .venv/Scripts/python.exe experiment/two_hop/_run_two_hop_bench.py \\
        --transport two_hop --case cases/lid_driven_cavity_2d/case.yaml \\
        --warmup 1000 --max-steps 3000
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The V5 campaign switches. simulator_v5 / transport_v5 / partition_v5 read
# them at import or construction time, so they are pinned here, before any
# V5 import, identically for both transports. V5_WORKER_COUNT_AWARE has no
# effect on two_hop (nothing is copied) but stays set.
CAMPAIGN_ENVIRONMENT = {
    "V5_WORKER_COUNT_AWARE": "1",
    "V5_GHOST_POOL_FACTOR": "0.25",
    "V5_SPLIT_TRANSFER_QUEUES": "1",
    "V5_CASCADE_FORCE": "1",
    "V5_BAND_VOXEL_DISPATCH": "1",
}

TRANSPORTS = ("three_hop", "two_hop")

# Worker segments: (name, start key, end key). The last two exist only in
# the relay worker's records.
WORKER_SEGMENTS = (
    ("source_wait", "dequeue_ns", "source_wait_ns"),
    ("dest_guard", "source_wait_ns", "dest_guard_ns"),
    ("upload_guard", "dest_guard_ns", "wait_ns"),
    ("copy", "wait_ns", "copy_ns"),
    ("signal", "copy_ns", "signal_ns"),
    # readback observed -> upload released: the host's share of the chain.
    ("relay_total", "source_wait_ns", "signal_ns"),
    ("upload_wait", "signal_ns", "upload_done_ns"),
    ("consumed_signal", "upload_done_ns", "consumed_ns"),
)

GPU_METRICS = (
    "phase_a_us", "phase_b_us", "phase_c_us",
    "a_to_b_gap_us", "b_to_c_gap_us", "c_to_a_gap_us",
    "readback_dma_us", "readback_sched_gap_us",
    "upload_dma_us", "upload_to_c_gap_us",
    # Derived (gpu_frame_metrics): end of this GPU's phase A -> inbound
    # upload landed, and what is left of it after the three GPU-timed legs.
    "chain_us", "chain_host_share_us",
)

# b_to_c_gap above which a frame counts as exposed. The hidden-chain floor
# is ~6 us on this rig; 20 us is the threshold of the first campaign.
EXPOSED_GAP_THRESHOLDS_US = (50.0, 20.0)

# Python's default interpreter switch interval, for the control runs.
PYTHON_DEFAULT_SWITCH_INTERVAL_MS = 5.0

# GlobalStatusBuffer overflow counters (a non-zero one means particles or
# neighbour-list entries were dropped) and their names in the defrag report.
OVERFLOW_COUNTERS = (
    "overflow_inside_count", "overflow_incoming_count",
    "overflow_ghost_count", "overflow_install_tail",
    "overflow_install_inside")
OVERFLOW_REPORT_KEYS = {
    "overflow_inside": "overflow_inside_count",
    "overflow_incoming": "overflow_incoming_count",
    "overflow_ghost": "overflow_ghost_count",
    "overflow_install_tail": "overflow_install_tail",
    "overflow_install_inside": "overflow_install_inside",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="two-hop vs three-hop bench runner")
    parser.add_argument("--transport", required=True, choices=TRANSPORTS)
    parser.add_argument("--case", default="cases/lid_driven_cavity_2d/case.yaml")
    parser.add_argument("--weights", default="1.0,1.0")
    parser.add_argument("--device-map", default="0,1")
    parser.add_argument("--sync-scheme", default="per-direction",
                        choices=["aggregated", "per-direction"])
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--pool-safety", type=float, default=1.2)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--defrag-cadence", type=int, default=None)
    parser.add_argument("--trials", type=int, default=1,
                        help="repeat the run in this process (fresh "
                             "contexts + simulators each time). Interleaving "
                             "ACROSS transports is the campaign script's job.")
    parser.add_argument("--first-trial-index", type=int, default=0,
                        help="trial number recorded for the first trial")
    parser.add_argument("--instrumented", action="store_true",
                        help="depth-1 synchronous loop with GPU timers")
    parser.add_argument("--bench-csv", default=None,
                        help="per-frame CSV (trial index is appended to the "
                             "stem when --trials > 1)")
    parser.add_argument("--result-json", default=None,
                        help="append one JSON line per trial")
    parser.add_argument("--seam-check", action="store_true", default=True)
    parser.add_argument("--no-seam-check", dest="seam_check", action="store_false")
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--loop-trace", action="store_true",
                        help="V5_LOOP_TRACE=1 — per-frame CPU accounting of "
                             "the pipelined loop: time inside _submit_frame "
                             "(all sims' submits + worker notifies) vs time "
                             "blocked in the frame_done waits. Costs three "
                             "perf_counter calls per frame, in both arms.")
    parser.add_argument("--minimum-own-columns", type=int, default=None,
                        help="per-slab own-column floor handed to "
                             "compute_chain_partition (default: V5's "
                             "MINIMUM_OWN_COLUMNS_HARD)")
    parser.add_argument("--switch-interval-ms", type=float, default=None,
                        help="sys.setswitchinterval, in ms (default: leave "
                             "Python's 5 ms). The N56 production stack uses 0.2.")
    parser.add_argument("--variant", default=None,
                        help="free-text tag written into the result line")
    parser.add_argument("--max-per-voxel", type=int, default=None,
                        help="override the case's max_particles_per_voxel "
                             "(capacity experiments)")
    parser.add_argument("--max-incoming", type=int, default=None,
                        help="override the case's max_incoming_per_voxel")
    parser.add_argument("--ghost-pool-factor", type=float, default=None,
                        help="override V5_GHOST_POOL_FACTOR (the ghost pool is "
                             "face voxels x (max_per_voxel + max_incoming) x "
                             "factor, so it must follow a capacity override)")
    parser.add_argument("--lean-layout", action="store_true",
                        help="run without the extension_fields buffer "
                             "(experiment/memory_audit/lean_layout.py)")
    parser.add_argument("--host-import-extension-on-baseline", action="store_true",
                        help="control: three_hop with VK_EXT_external_memory_host "
                             "enabled on the device (unused), to show the "
                             "extension itself does not move the baseline")
    return parser.parse_args()


def pin_campaign_environment() -> dict:
    for name, value in CAMPAIGN_ENVIRONMENT.items():
        existing = os.environ.get(name)
        if existing is None:
            os.environ[name] = value
        elif existing != value:
            print(f"[two_hop_bench] WARNING: {name}={existing} overrides the "
                  f"campaign value {value}", file=sys.stderr)
    return {name: os.environ[name] for name in CAMPAIGN_ENVIRONMENT}


def percentile_summary(values: list) -> dict:
    ordered = sorted(values)
    count = len(ordered)

    def at(fraction: float) -> float:
        return round(ordered[min(count - 1, int(count * fraction))], 2)

    return {
        "p50": round(statistics.median(ordered), 2),
        "mean": round(statistics.fmean(ordered), 2),
        "p1": at(0.01),
        "p5": at(0.05),
        "p10": at(0.10),
        "p90": at(0.90),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": round(ordered[-1], 2),
        "samples": count,
    }


def exposure_summary(gaps: list) -> dict:
    """Distribution of b_to_c_gap plus the share of exposed frames."""
    summary = percentile_summary(gaps)
    for threshold in EXPOSED_GAP_THRESHOLDS_US:
        summary[f"exposed_over_{threshold:.0f}us_percent"] = round(
            sum(1 for gap in gaps if gap > threshold) / len(gaps) * 100.0, 3)
    return summary


def loop_trace_summary(rows: list, first_frame: int) -> dict:
    """CPU accounting of the pipelined loop over frames >= first_frame.
    ``rows`` = ChainOrchestratorV5's (loop start s, submit s, wait s) per
    frame. The period is the spacing of consecutive loop starts, so it
    contains the defrag pauses; submit and wait do not."""
    selected = rows[first_frame:]
    if len(selected) < 3:
        return {}
    starts = [row[0] for row in selected]
    periods = [(later - earlier) * 1e6
               for earlier, later in zip(starts, starts[1:])]
    submits = [row[1] * 1e6 for row in selected]
    waits = [row[2] * 1e6 for row in selected]
    mean_period = statistics.fmean(periods)
    return {
        "submit_us": percentile_summary(submits),
        "wait_us": percentile_summary(waits),
        "period_us": percentile_summary(periods),
        # Share of the frame period the main thread spends submitting; near
        # 1 (and waits near 0) means the CPU loop paces the pipeline.
        "submit_share_of_period": round(
            statistics.fmean(submits) / mean_period, 4),
        "frames_without_wait_percent": round(
            sum(1 for wait in waits if wait < 5.0) / len(waits) * 100.0, 3),
    }


def worker_segment_rows(worker, first_frame: int) -> dict:
    """{frame_n: {segment: microseconds}} for frames >= first_frame."""
    rows = {}
    for frame_n, stamps in worker.timestamps.items():
        if frame_n < first_frame or "dequeue_ns" not in stamps:
            continue
        rows[frame_n] = {
            name: (stamps[end_key] - stamps[start_key]) / 1000.0
            for name, start_key, end_key in WORKER_SEGMENTS
            if start_key in stamps and end_key in stamps}
    return rows


def summarize_worker(worker, first_frame: int) -> dict:
    rows = worker_segment_rows(worker, first_frame)
    samples: dict = {}
    for segments in rows.values():
        for name, value in segments.items():
            samples.setdefault(name, []).append(value)
    return {name: percentile_summary(values) for name, values in samples.items()}


def peer_direction(slab_index: int) -> str:
    """K=2: slab 0 talks through its trailing side, slab 1 through leading."""
    return "trailing" if slab_index == 0 else "leading"


def gpu_frame_metrics(durations: dict, slab_index: int) -> dict:
    direction = peer_direction(slab_index)
    renamed = {
        "readback_dma_us": f"readback_{direction}_dma_us",
        "readback_sched_gap_us": f"readback_{direction}_sched_gap_us",
        "upload_dma_us": f"upload_{direction}_dma_us",
        "upload_to_c_gap_us": f"upload_{direction}_to_c_gap_us",
    }
    metrics = {metric: durations.get(renamed.get(metric, metric))
               for metric in GPU_METRICS}
    legs = [metrics[name] for name in (
        "a_to_b_gap_us", "phase_b_us", "b_to_c_gap_us", "upload_to_c_gap_us")]
    if all(value is not None for value in legs):
        # Phase C starts a_to_b_gap + phase B + b_to_c_gap after phase A
        # ended; the upload landed upload_to_c_gap before that.
        metrics["chain_us"] = legs[0] + legs[1] + legs[2] - legs[3]
        timed = [metrics[name] for name in (
            "readback_sched_gap_us", "readback_dma_us", "upload_dma_us")]
        if all(value is not None for value in timed):
            # What the GPU-timed legs do not account for: semaphore round
            # trips, host thread wake-ups and (three-hop) the byte copy.
            # Uses this GPU's own readback legs as the peer's (K=2, equal
            # weights).
            metrics["chain_host_share_us"] = metrics["chain_us"] - sum(timed)
    return metrics


def run_trial(args, trial_index: int, environment: dict) -> dict:
    from experiment.two_hop.shared_host_v5 import (
        ChainOrchestratorTwoHop, SharedHostLinkPool, SphSimulatorTwoHop,
        VulkanContextTwoHop)
    from experiment.v5._run_v5_chain_bench import seam_integrity_check
    from experiment.v5.utils.bench_v5 import (
        BenchTimer, compute_durations, split_parity_ticks)
    from experiment.v5.utils.case_loader_v5 import load_case_v5
    from experiment.v5.utils.orchestrator_v5 import ChainOrchestratorV5
    from experiment.v5.utils.partition_v5 import (
        MINIMUM_OWN_COLUMNS_HARD, compute_chain_partition)
    from experiment.v5.utils.simulator_v5 import SphSimulatorV5
    from experiment.v5.utils.vulkan_context_v5 import VulkanContextV5

    weights = [float(text) for text in args.weights.split(",")]
    device_map = [int(text) for text in args.device_map.split(",")]
    if len(weights) != 2 or len(device_map) != 2:
        sys.exit("this experiment is K=2: --weights and --device-map take 2 values")
    two_hop = args.transport == "two_hop"
    if two_hop and args.sync_scheme != "per-direction":
        sys.exit("two_hop requires --sync-scheme per-direction")
    pool_safety = None if args.pool_safety == 0 else args.pool_safety

    global_case = load_case_v5(args.case)
    expected_total = int(global_case.initial.positions.shape[0])
    if args.max_per_voxel is not None:
        global_case.capacities.max_particles_per_voxel = args.max_per_voxel
    if args.max_incoming is not None:
        global_case.capacities.max_incoming_per_voxel = args.max_incoming
    minimum_own_columns = (args.minimum_own_columns
                           if args.minimum_own_columns is not None
                           else MINIMUM_OWN_COLUMNS_HARD)
    chain = compute_chain_partition(global_case, weights, pool_safety,
                                    minimum_own_columns=minimum_own_columns)
    defrag_cadence = (args.defrag_cadence if args.defrag_cadence is not None
                      else global_case.numerics.defrag_cadence)

    host_import_extension = two_hop or args.host_import_extension_on_baseline
    context_class = VulkanContextTwoHop if host_import_extension else VulkanContextV5
    orchestrator_class = ChainOrchestratorTwoHop if two_hop else ChainOrchestratorV5
    three_hop_simulator_class = SphSimulatorV5
    two_hop_simulator_class = SphSimulatorTwoHop
    if args.lean_layout:
        from experiment.memory_audit.lean_layout import LeanLayoutMixin

        class SphSimulatorLean(LeanLayoutMixin, SphSimulatorV5):
            pass

        class SphSimulatorTwoHopLean(LeanLayoutMixin, SphSimulatorTwoHop):
            pass

        three_hop_simulator_class = SphSimulatorLean
        two_hop_simulator_class = SphSimulatorTwoHopLean
    print(f"[two_hop_bench] trial {trial_index}: transport={args.transport} "
          f"case={args.case} weights={weights} device_map={device_map} "
          f"sync={args.sync_scheme} depth={1 if args.instrumented else args.depth} "
          f"instrumented={args.instrumented} pool_safety={pool_safety} "
          f"context={context_class.__name__}")

    result: dict = {
        "case": args.case,
        "particles": expected_total,
        "transport": args.transport,
        "variant": args.variant,
        "lean_layout": args.lean_layout,
        "capacities": [{
            "own_pool_size": slab.capacities.own_pool_size,
            "ghost_pool_size": (slab.capacities.leading_ghost_pool_size
                                + slab.capacities.trailing_ghost_pool_size),
            "max_particles_per_voxel": slab.capacities.max_particles_per_voxel,
            "max_incoming_per_voxel": slab.capacities.max_incoming_per_voxel,
        } for slab in chain.slabs],
        "trial": trial_index,
        "instrumented": args.instrumented,
        "depth": 1 if args.instrumented else args.depth,
        "weights": weights,
        "sync_scheme": args.sync_scheme,
        "pool_safety": pool_safety,
        "warmup": args.warmup,
        "max_steps": args.max_steps,
        "defrag_cadence": defrag_cadence,
        "validation": args.validation,
        "host_import_extension": host_import_extension,
        "environment": environment,
        "switch_interval_ms": round(sys.getswitchinterval() * 1000.0, 4),
        "partition": {
            "grid_columns": int(global_case.grid.grid_dimension_x),
            "cut_columns": [int(cut) for cut in chain.cuts],
            "own_columns": [int(geometry.own_column_count)
                            for geometry in chain.geometry],
            "own_particles": [int(geometry.own_particle_count)
                              for geometry in chain.geometry],
            "minimum_own_columns": minimum_own_columns,
            "minimum_own_columns_relaxed": (
                minimum_own_columns < MINIMUM_OWN_COLUMNS_HARD),
        },
        "unix_time": round(time.time(), 1),
    }

    contexts, sims, timers = [], [], []
    link_pool = None
    frame_rows: list = []
    try:
        for slab_index in range(2):
            contexts.append(context_class.create(
                device_index=device_map[slab_index],
                enable_validation=args.validation,
                application_name=f"two_hop_bench_s{slab_index}"))
        if two_hop:
            link_pool = SharedHostLinkPool(contexts)
        for slab_index in range(2):
            if two_hop:
                sims.append(two_hop_simulator_class(
                    contexts[slab_index], chain.slabs[slab_index],
                    link_pool=link_pool, slab_index=slab_index,
                    sync_scheme=args.sync_scheme))
            else:
                sims.append(three_hop_simulator_class(
                    contexts[slab_index], chain.slabs[slab_index],
                    sync_scheme=args.sync_scheme))
        result["staging_bytes_per_direction"] = [
            dict(sim._transport_total_bytes) for sim in sims]
        result["device_buffer_bytes"] = [
            sum(buffer.size for buffer in sim.buffers.values())
            + sum(buffer.size for buffer in sim.scratch_buffers.values())
            for sim in sims]

        if args.instrumented:
            for slab_index, sim in enumerate(sims):
                compute_timer = BenchTimer(sim.ctx, label=f"s{slab_index}")
                transfer_timer = BenchTimer(
                    sim.ctx, label=f"s{slab_index}_transfer",
                    queue_family_index=sim.ctx.transfer_queue_family_index)
                sim.bench = compute_timer
                sim.bench_transfer = transfer_timer
                timers.append((compute_timer, transfer_timer))

        defrag_drops = []
        # The overflow counters are cumulative since the last defrag, so they
        # are summed at every defrag boundary and once more at the end.
        overflow_totals = {name: 0 for name in OVERFLOW_COUNTERS}

        def on_defrag(frame_n: int, report: list) -> None:
            drops = sum(entry["overflow_install_tail"] for entry in report)
            for report_key, name in OVERFLOW_REPORT_KEYS.items():
                overflow_totals[name] += sum(
                    entry.get(report_key, 0) for entry in report)
            if drops:
                defrag_drops.append((frame_n, drops))
                print(f"[migration] frame {frame_n}: *** DROPS={drops} ***",
                      file=sys.stderr, flush=True)

        with orchestrator_class(sims, defrag_cadence=defrag_cadence) as orchestrator:
            orchestrator.bootstrap_all()

            if args.instrumented:
                gpu_samples: list = [dict() for _ in sims]
                # Every duration compute_durations emits (per-kernel split
                # included); summarized only, not written to the CSV.
                kernel_samples: list = [dict() for _ in sims]
                frame_times = []
                while orchestrator.frame_count < args.max_steps:
                    record = orchestrator.step()
                    frame_n = record["frame_n"]
                    if "defrag_report" in record:
                        on_defrag(record["defrag_frame"], record["defrag_report"])
                    if frame_n < args.warmup:
                        continue
                    row = {"frame_n": frame_n,
                           "frame_time_us": record["frame_time_us"]}
                    frame_times.append(record["frame_time_us"])
                    for slab_index, (compute_timer, transfer_timer) in enumerate(timers):
                        ticks = compute_timer.read_frame(include_defrag=False)
                        ticks.update(transfer_timer.read_frame(include_defrag=False))
                        previous_c_end = None
                        if compute_timer.parity_regions:
                            ticks, previous_c_end = split_parity_ticks(
                                ticks, frame_n % 2)
                        durations = compute_durations(ticks)
                        if previous_c_end is not None and "a_start" in ticks:
                            durations["c_to_a_gap_us"] = (
                                ticks["a_start"] - previous_c_end) / 1000.0
                        for name, value in durations.items():
                            if value is not None:
                                kernel_samples[slab_index].setdefault(
                                    name, []).append(value)
                        for metric, value in gpu_frame_metrics(
                                durations, slab_index).items():
                            row[f"s{slab_index}_{metric}"] = value
                            if value is not None:
                                gpu_samples[slab_index].setdefault(
                                    metric, []).append(value)
                    for label, stamps in record["workers"].items():
                        for name, start_key, end_key in WORKER_SEGMENTS:
                            if start_key in stamps and end_key in stamps:
                                row[f"worker_{label}_{name}_us"] = (
                                    stamps[end_key] - stamps[start_key]) / 1000.0
                    frame_rows.append(row)
                median_frame_us = statistics.median(frame_times)
                result["steady_fps"] = round(1e6 / median_frame_us, 2)
                result["frame_time_us"] = percentile_summary(frame_times)
                result["gpu"] = [
                    {metric: percentile_summary(values)
                     for metric, values in samples.items()}
                    for samples in gpu_samples]
                result["gpu_durations_p50_us"] = [
                    {name: round(statistics.median(values), 2)
                     for name, values in samples.items()}
                    for samples in kernel_samples]
                all_gaps = [gap for samples in gpu_samples
                            for gap in samples.get("b_to_c_gap_us", [])]
                if all_gaps:
                    # Both GPUs pooled: a frame is late if either waited.
                    result["b_to_c_gap_us"] = exposure_summary(all_gaps)
            else:
                pipelined = orchestrator.run_pipelined(
                    args.max_steps, depth=args.depth, warmup=args.warmup,
                    on_defrag=on_defrag)
                result["total_fps"] = round(pipelined["fps"], 2)
                result["steady_fps"] = round(pipelined.get("steady_fps", 0.0), 2)
                result["steady_frames"] = pipelined.get("steady_frames")
                if args.loop_trace:
                    result["loop"] = loop_trace_summary(
                        list(orchestrator._loop_trace), args.warmup)
                    # Covers every frame of the run, warmup included.
                    result["loop_trace"] = orchestrator.loop_trace_stats()
                    if result["loop"]:
                        loop = result["loop"]
                        print(f"[two_hop_bench] loop (post-warmup): submit "
                              f"p50={loop['submit_us']['p50']:.0f}us "
                              f"mean={loop['submit_us']['mean']:.0f}us  wait "
                              f"p50={loop['wait_us']['p50']:.0f}us  period "
                              f"p50={loop['period_us']['p50']:.0f}us  "
                              f"submit share={loop['submit_share_of_period']:.3f}  "
                              f"frames without wait="
                              f"{loop['frames_without_wait_percent']:.2f}%")
            print(f"[two_hop_bench] STEADY {result['steady_fps']:.1f} fps "
                  f"({args.transport}, post-warmup {args.warmup})")

            for sim in sims:
                sim.submit_defrag_and_wait()
            alive_total, stamp_errors_gpu = 0, 0
            pool_used = []
            for sim in sims:
                status = sim.readback_global_status()
                health = sim.readback_pool_health()
                alive_total += status["alive_particle_count"]
                stamp_errors_gpu += status.get("stamp_error_count", 0)
                pool_used.append(round(health["used_fraction"], 4))
                for name in OVERFLOW_COUNTERS:
                    overflow_totals[name] += int(status[name])
            workers = orchestrator.workers
            result["alive_total"] = alive_total
            result["drift"] = alive_total - expected_total
            result["stamp_errors_gpu"] = stamp_errors_gpu
            result["stamp_errors_host"] = sum(
                worker.stamp_error_count for worker in workers)
            result["overwrite_errors_host"] = sum(
                getattr(worker, "overwrite_error_count", 0) for worker in workers)
            result["install_tail_drops"] = sum(drops for _, drops in defrag_drops)
            result["overflow"] = overflow_totals
            result["pool_used_fraction"] = pool_used
            result["worker_copy_bytes_last_frame"] = {
                worker.label: worker.last_copy_bytes for worker in workers}
            result["worker_segments_us"] = {
                worker.label: summarize_worker(worker, args.warmup)
                for worker in workers}
            for worker in workers:
                parts = "  ".join(
                    f"{name}={values['p50']:.0f}/{values['p90']:.0f}/{values['max']:.0f}"
                    for name, values in result["worker_segments_us"][worker.label].items())
                print(f"[worker {worker.label}] us p50/p90/max: {parts}")
            if not args.instrumented:
                for frame_n in sorted(workers[0].timestamps):
                    if frame_n < args.warmup:
                        continue
                    row = {"frame_n": frame_n}
                    for worker in workers:
                        stamps = worker.timestamps.get(frame_n, {})
                        for name, start_key, end_key in WORKER_SEGMENTS:
                            if start_key in stamps and end_key in stamps:
                                row[f"worker_{worker.label}_{name}_us"] = (
                                    stamps[end_key] - stamps[start_key]) / 1000.0
                    frame_rows.append(row)

            print(f"[two_hop_bench] final: total={alive_total:,} "
                  f"(expected {expected_total:,}) drift={result['drift']} "
                  f"stamp_errors gpu={stamp_errors_gpu} "
                  f"host={result['stamp_errors_host']} "
                  f"overwrite={result['overwrite_errors_host']}")

            result["seam_ok"] = None
            if args.seam_check:
                result["seam_ok"] = bool(
                    seam_integrity_check(chain, sims, global_case))
        result["valid"] = (
            result["drift"] == 0
            and result["stamp_errors_gpu"] == 0
            and result["stamp_errors_host"] == 0
            and result["overwrite_errors_host"] == 0
            and result["install_tail_drops"] == 0
            and not any(result["overflow"].values())
            and result["seam_ok"] is not False)
        if not result["valid"]:
            print("[two_hop_bench] *** VALIDATION FAILED ***")
    finally:
        for compute_timer, transfer_timer in timers:
            compute_timer.destroy()
            transfer_timer.destroy()
        for sim in sims:
            sim.destroy()
        if link_pool is not None:
            link_pool.destroy()
        for context in contexts:
            context.destroy()

    if args.bench_csv and frame_rows:
        csv_path = pathlib.Path(args.bench_csv)
        if args.trials > 1:
            csv_path = csv_path.with_name(
                f"{csv_path.stem}_trial{trial_index}{csv_path.suffix}")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        columns = list(dict.fromkeys(
            column for row in frame_rows for column in row))
        with open(csv_path, "w") as handle:
            handle.write(",".join(columns) + "\n")
            for row in frame_rows:
                handle.write(",".join(
                    "" if row.get(column) is None
                    else (str(row[column]) if column == "frame_n"
                          else f"{row[column]:.3f}")
                    for column in columns) + "\n")
        result["bench_csv"] = str(csv_path)
        print(f"[two_hop_bench] CSV -> {csv_path}")
    return result


def main() -> int:
    args = parse_args()
    if args.switch_interval_ms is not None:
        # Before any worker thread exists.
        sys.setswitchinterval(args.switch_interval_ms / 1000.0)
    print(f"[two_hop_bench] interpreter switch interval: "
          f"{sys.getswitchinterval() * 1000.0:g} ms")
    if args.ghost_pool_factor is not None:
        os.environ["V5_GHOST_POOL_FACTOR"] = f"{args.ghost_pool_factor:g}"
    environment = pin_campaign_environment()
    if args.lean_layout:
        from experiment.memory_audit.lean_layout import apply_lean_layout
        apply_lean_layout()
    if args.loop_trace:
        os.environ["V5_LOOP_TRACE"] = "1"
        environment["V5_LOOP_TRACE"] = "1"
    print(f"[two_hop_bench] environment: {environment}")
    exit_code = 0
    for offset in range(args.trials):
        result = run_trial(args, args.first_trial_index + offset, environment)
        if args.result_json:
            result_path = pathlib.Path(args.result_json)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            with open(result_path, "a") as handle:
                handle.write(json.dumps(result) + "\n")
        if not result["valid"]:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
