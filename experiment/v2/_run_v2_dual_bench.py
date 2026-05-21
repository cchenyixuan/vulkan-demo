"""
_run_v2_dual_bench.py — V2 dual-GPU headless BENCHMARK runner.

Same bootstrap as _run_v2_dual.py with no viewer and no debug-log path, plus
GPU-timestamp instrumentation: attaches one BenchTimer per simulator BEFORE
bootstrap_all (so the SIMULTANEOUS_USE phase A/B/C/defrag cmd buffers get
ticks baked in), then per frame extracts:

  - per-kernel µs from each GPU (predict / update_voxel / ghost_send /
    correction_interior / install_migrations / correction_boundary /
    density / force / defrag)
  - phase A/B/C totals + a_to_b_gap + b_to_c_gap (sync-hiding KPI)
  - worker memcpy µs (already recorded by transport_v2's worker thread)
  - frame_total µs (orchestrator wall-clock)

Two outputs:
  1. Full CSV (--bench-csv PATH) — one row per frame, all metrics
  2. Periodic stderr summary every --bench-window frames — rolling mean
     of the key KPIs

Usage:
    .venv/Scripts/python.exe experiment/v2/_run_v2_dual_bench.py \\
        --case cases/lid_driven_cavity_2d/case.yaml \\
        --max-steps 5000 \\
        --bench-csv logs/bench_001.csv \\
        --bench-window 500

Design note: this runner exists separate from _run_v2_dual.py / _viewer
because clean benchmark numbers require zero swapchain present overhead,
no camera matrix work, no debug snapshot writes — anything that runs on
the CPU thread between orch.step() calls pollutes the wall-clock.
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time
from typing import Optional, TextIO

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V2 dual-GPU benchmark runner")
    p.add_argument("--case", default="cases/lid_driven_cavity_2d/case.yaml")
    p.add_argument("--device-a", type=int, default=0)
    p.add_argument("--device-b", type=int, default=1)
    p.add_argument("--weights", default="1.0,1.0")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--defrag-cadence", type=int, default=None)
    p.add_argument("--validation", action="store_true")
    p.add_argument("--disable-pst", action="store_true")
    p.add_argument("--no-defrag", action="store_true")
    p.add_argument("--bench-csv", type=str, default=None,
                   help="path to per-frame CSV output (created if absent)")
    p.add_argument("--bench-window", type=int, default=500,
                   help="frames between stderr aggregate prints")
    p.add_argument("--warmup", type=int, default=50,
                   help="frames to discard before any aggregate / CSV row")
    return p.parse_args()


# ============================================================================
# Per-frame row assembly
# ============================================================================

# Ordered list of per-GPU duration keys. Used both for CSV header and stderr.
_PER_GPU_KEYS = [
    "predict_us",
    "update_voxel_us",
    "ghost_send_leading_us",
    "ghost_send_leading_dispatch_us",      # split: compute portion
    "ghost_send_leading_readback_us",      # split: device→host DMA
    "ghost_send_leading_host_barrier_us",  # split: host-coherence barrier
    "ghost_send_trailing_us",
    "ghost_send_trailing_dispatch_us",
    "ghost_send_trailing_readback_us",
    "ghost_send_trailing_host_barrier_us",
    "phase_a_us",
    "a_to_b_gap_us",
    "correction_interior_us",
    "phase_b_us",
    "b_to_c_gap_us",
    "install_leading_us",
    "install_leading_upload_us",           # split: host→device DMA
    "install_leading_dispatch_us",         # split: install_migrations compute
    "install_trailing_us",
    "install_trailing_upload_us",
    "install_trailing_dispatch_us",
    "correction_boundary_us",
    "density_us",
    "force_us",
    "phase_c_us",
    "defrag_us",
]

_WORKER_KEYS = [
    "worker_a_to_b_memcpy_us",
    "worker_a_to_b_wait_us",
    "worker_b_to_a_memcpy_us",
    "worker_b_to_a_wait_us",
]


def _csv_header() -> str:
    cols = ["frame_n", "frame_total_us"]
    for tag in ("a", "b"):
        cols.extend(f"gpu_{tag}_{k}" for k in _PER_GPU_KEYS)
    cols.extend(_WORKER_KEYS)
    return ",".join(cols) + "\n"


def _format_cell(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.3f}"


def _csv_row(frame_n: int, frame_total_us: float,
             gpu_a: dict[str, float], gpu_b: dict[str, float],
             worker: dict[str, float]) -> str:
    parts = [str(frame_n), f"{frame_total_us:.3f}"]
    for d in (gpu_a, gpu_b):
        parts.extend(_format_cell(d.get(k)) for k in _PER_GPU_KEYS)
    parts.extend(_format_cell(worker.get(k)) for k in _WORKER_KEYS)
    return ",".join(parts) + "\n"


def _extract_worker_metrics(orch_record: dict) -> dict[str, float]:
    """Pull memcpy / wait durations out of orchestrator.step()'s record.
    Worker timestamps_for_frame returns {wait_ns, copy_ns, signal_ns}; we
    derive memcpy = copy - wait and wait = wait - frame_start."""
    frame_start_ns = orch_record["frame_start_ns"]
    out: dict[str, float] = {}
    for tag in ("a_to_b", "b_to_a"):
        ts = orch_record.get(f"worker_{tag}", {})
        if not ts:
            continue
        wait_ns = ts.get("wait_ns")
        copy_ns = ts.get("copy_ns")
        if wait_ns is not None and copy_ns is not None:
            out[f"worker_{tag}_memcpy_us"] = (copy_ns - wait_ns) / 1000.0
            out[f"worker_{tag}_wait_us"] = (wait_ns - frame_start_ns) / 1000.0
    return out


# ============================================================================
# Window aggregator (rolling stats for stderr)
# ============================================================================


class WindowAggregator:
    """Accumulates per-frame metric dicts inside a sliding window of N frames.
    On flush() emits one summary line per key with mean / p50 / p99."""

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.samples: dict[str, list[float]] = {}
        self.frame_count = 0

    def add(self, name: str, value: Optional[float]) -> None:
        if value is None:
            return
        self.samples.setdefault(name, []).append(value)

    def tick(self) -> bool:
        """Increment frame counter; returns True when window is full."""
        self.frame_count += 1
        return self.frame_count >= self.window_size

    def flush_summary(self, header: str) -> str:
        """Build a multi-line summary string and reset the window."""
        lines = [header]
        for name in sorted(self.samples):
            values = self.samples[name]
            if not values:
                continue
            mean = statistics.fmean(values)
            p50 = statistics.median(values)
            p99 = sorted(values)[max(0, int(len(values) * 0.99) - 1)]
            lines.append(
                f"  {name:38s}  mean={mean:8.1f}us  p50={p50:8.1f}us  p99={p99:8.1f}us  n={len(values)}")
        self.samples.clear()
        self.frame_count = 0
        return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    args = parse_args()

    from experiment.v2.utils.bench_v2 import BenchTimer, compute_durations
    from experiment.v2.utils.case_loader_v2 import load_case_v2
    from experiment.v2.utils.orchestrator_v2 import DualGpuOrchestratorV2
    from experiment.v2.utils.partition_v2 import compute_dual_gpu_partition
    from experiment.v2.utils.simulator_v2 import SphSimulatorV2
    from experiment.v2.utils.vulkan_context_v2 import VulkanContextV2

    weights = [float(w) for w in args.weights.split(",")]
    if len(weights) != 2:
        sys.exit(f"--weights must have 2 values; got {args.weights!r}")

    print(f"[bench_v2_dual] case={args.case}")
    print(f"[bench_v2_dual] device_a={args.device_a} device_b={args.device_b} "
          f"weights={weights}  max_steps={args.max_steps}  warmup={args.warmup}")

    global_case = load_case_v2(args.case)
    if args.disable_pst:
        print("[bench_v2_dual] use_pst overridden to False")
        global_case.numerics.use_pst = False

    expected_total = int(global_case.initial.positions.shape[0])

    slab0, slab1, _k_split = compute_dual_gpu_partition(global_case, weights)
    defrag_cadence = (args.defrag_cadence if args.defrag_cadence is not None
                      else global_case.numerics.defrag_cadence)
    if args.no_defrag:
        defrag_cadence = args.max_steps + 1  # never triggers

    ctx_a = VulkanContextV2.create(
        device_index=args.device_a, enable_validation=args.validation,
        application_name="sph_v2_bench_a")
    ctx_b = VulkanContextV2.create(
        device_index=args.device_b, enable_validation=args.validation,
        application_name="sph_v2_bench_b")

    sim_a = SphSimulatorV2(ctx_a, slab0)
    sim_b = SphSimulatorV2(ctx_b, slab1)

    # Attach BenchTimer BEFORE bootstrap_all so phase A/B/C recordings see
    # a live bench and bake in vkCmdWriteTimestamp calls.
    bench_a = BenchTimer(ctx_a, label="gpu_a")
    bench_b = BenchTimer(ctx_b, label="gpu_b")
    sim_a.bench = bench_a
    sim_b.bench = bench_b
    print(f"[bench_v2_dual] timestampPeriod a={bench_a.ns_per_tick:.4f}ns  "
          f"b={bench_b.ns_per_tick:.4f}ns")

    rc = 0
    csv_file: Optional[TextIO] = None
    if args.bench_csv:
        csv_path = pathlib.Path(args.bench_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "w")
        csv_file.write(_csv_header())
        print(f"[bench_v2_dual] CSV → {csv_path}")

    aggregator = WindowAggregator(window_size=args.bench_window)

    try:
        with DualGpuOrchestratorV2(sim_a, sim_b,
                                    defrag_cadence=defrag_cadence) as orch:
            orch.bootstrap_all()

            t_start = time.perf_counter()
            while orch.frame_count < args.max_steps:
                record = orch.step()
                # Defrag fires when frame_count just became a multiple of cadence.
                defrag_ran = (orch.frame_count > 0
                              and orch.frame_count % defrag_cadence == 0)

                # Skip the warmup period entirely.
                if orch.frame_count <= args.warmup:
                    continue

                gpu_a_ticks = bench_a.read_frame(include_defrag=defrag_ran)
                gpu_b_ticks = bench_b.read_frame(include_defrag=defrag_ran)
                gpu_a = compute_durations(gpu_a_ticks)
                gpu_b = compute_durations(gpu_b_ticks)
                worker = _extract_worker_metrics(record)

                # Persist
                if csv_file is not None:
                    csv_file.write(_csv_row(
                        orch.frame_count, record["frame_time_us"],
                        gpu_a, gpu_b, worker))

                # Aggregate
                aggregator.add("frame_total_us", record["frame_time_us"])
                for k in _PER_GPU_KEYS:
                    aggregator.add(f"gpu_a_{k}", gpu_a.get(k))
                    aggregator.add(f"gpu_b_{k}", gpu_b.get(k))
                for k in _WORKER_KEYS:
                    aggregator.add(k, worker.get(k))

                if aggregator.tick():
                    elapsed = time.perf_counter() - t_start
                    fps = orch.frame_count / elapsed
                    header = (f"[bench] frames {orch.frame_count-args.bench_window+1}"
                              f"..{orch.frame_count}  fps={fps:6.1f}  "
                              f"(wall {elapsed:.1f}s)")
                    print(aggregator.flush_summary(header),
                          file=sys.stderr, flush=True)
                    if csv_file is not None:
                        csv_file.flush()

            total_elapsed = time.perf_counter() - t_start
            print(f"[bench_v2_dual] {args.max_steps} steps in "
                  f"{total_elapsed:.2f}s = {args.max_steps / total_elapsed:.1f} fps")

            # Final defrag to validate alive count (sanity check, not a metric)
            if not args.no_defrag:
                sim_a.submit_defrag_and_wait()
                sim_b.submit_defrag_and_wait()
            s_a = sim_a.readback_global_status()
            s_b = sim_b.readback_global_status()
            total = s_a["alive_particle_count"] + s_b["alive_particle_count"]
            print(f"[bench_v2_dual] final: a={s_a['alive_particle_count']:,} "
                  f"b={s_b['alive_particle_count']:,} "
                  f"total={total:,} (expected {expected_total:,})")
            if total != expected_total:
                print(f"[bench_v2_dual] WARN: alive drift = "
                      f"{total - expected_total}", file=sys.stderr)

    finally:
        if csv_file is not None:
            csv_file.close()
        bench_a.destroy()
        bench_b.destroy()
        sim_a.destroy()
        sim_b.destroy()
        ctx_a.destroy()
        ctx_b.destroy()

    return rc


if __name__ == "__main__":
    sys.exit(main())
