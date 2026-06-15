"""
_run_v3_single_bench.py — V3 single-GPU headless BENCHMARK runner.

Single-GPU baseline for dual-GPU comparison: one sim, one ctx, one GPU.
Uses the combined step cmd path (predict + update_voxel + correction_all
+ density + force) — no ghost flow, no timeline semaphores, no worker
threads, one submit + one fence wait per frame. See SphSimulatorV3.
_record_step_single_cmd for the rationale on why ghost_send / install_
migrations / the correction interior/boundary split are all skipped.

Outputs match the dual-GPU bench runner format:
  1. Full CSV (--bench-csv PATH) — one row per frame, all per-kernel µs
  2. Periodic stderr summary every --bench-window frames — rolling
     mean / p50 / p99 of the key KPIs

Usage:
    .venv/Scripts/python.exe experiment/v3/_run_v3_single_bench.py \\
        --case cases/lid_driven_cavity_2d/case.yaml \\
        --device 0 \\
        --max-steps 5000 \\
        --bench-csv logs/single_a.csv \\
        --bench-window 1000
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
    p = argparse.ArgumentParser(description="V3 single-GPU benchmark runner")
    p.add_argument("--case", default="cases/lid_driven_cavity_2d/case.yaml")
    p.add_argument("--device", type=int, default=0,
                   help="physical device index for the single sim")
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
    p.add_argument("--validate-split", action="store_true",
                   help="P3.C: use split pipeline variants (correction_interior+"
                        "_boundary, density_deep_interior+_boundary, force_deep_"
                        "interior+_boundary) in place of *_all. Single-GPU mode's"
                        " empty boundary band makes the two paths bit-equivalent;"
                        " divergent alive / readback proves a shader bug.")
    return p.parse_args()


# ============================================================================
# Per-frame row assembly
# ============================================================================

# Ordered list of per-frame duration keys for the single-GPU bench. Used
# both for CSV header and stderr. Mirrors dual-GPU bench's _PER_GPU_KEYS
# but flat (no phase A/B/C aggregation — single mode has no phases).
_KERNEL_KEYS = [
    "predict_us",
    "update_voxel_us",
    "correction_us",
    "density_us",
    "force_us",
    "step_total_us",
    "defrag_us",
]


def _csv_header() -> str:
    return ",".join(["frame_n", "frame_total_us"] + _KERNEL_KEYS) + "\n"


def _format_cell(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.3f}"


def _csv_row(frame_n: int, frame_total_us: float,
             kernels: dict[str, float]) -> str:
    parts = [str(frame_n), f"{frame_total_us:.3f}"]
    parts.extend(_format_cell(kernels.get(k)) for k in _KERNEL_KEYS)
    return ",".join(parts) + "\n"


# ============================================================================
# Window aggregator (rolling stats for stderr) — identical to dual runner.
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
        self.frame_count += 1
        return self.frame_count >= self.window_size

    def flush_summary(self, header: str) -> str:
        lines = [header]
        for name in sorted(self.samples):
            values = self.samples[name]
            if not values:
                continue
            mean = statistics.fmean(values)
            p50 = statistics.median(values)
            p99 = sorted(values)[max(0, int(len(values) * 0.99) - 1)]
            lines.append(
                f"  {name:24s}  mean={mean:8.1f}us  p50={p50:8.1f}us  p99={p99:8.1f}us  n={len(values)}")
        self.samples.clear()
        self.frame_count = 0
        return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    args = parse_args()

    from experiment.v3.utils.bench_v3 import BenchTimer, compute_durations
    from experiment.v3.utils.case_loader_v3 import load_case_v3
    from experiment.v3.utils.simulator_v3 import SphSimulatorV3
    from experiment.v3.utils.vulkan_context_v3 import VulkanContextV3

    print(f"[bench_v3_single] case={args.case}")
    print(f"[bench_v3_single] device={args.device}  max_steps={args.max_steps}  "
          f"warmup={args.warmup}")

    # load_case_v3 returns a degenerate CaseV3 with no ghost / no transport
    # (see feedback_loader_partition_decoupling.md — partition_v3 is the
    # sole producer of ghost segments, and we're bypassing it here).
    case = load_case_v3(args.case)
    if args.disable_pst:
        print("[bench_v3_single] use_pst overridden to False")
        case.numerics.use_pst = False

    expected_total = int(case.initial.positions.shape[0])

    defrag_cadence = (args.defrag_cadence if args.defrag_cadence is not None
                      else case.numerics.defrag_cadence)
    if args.no_defrag:
        defrag_cadence = args.max_steps + 1  # never triggers

    ctx = VulkanContextV3.create(
        device_index=args.device, enable_validation=args.validation,
        application_name="sph_v3_bench_single")

    sim = SphSimulatorV3(ctx, case)

    # Attach BenchTimer BEFORE prepare_step_single_cmd_buffer so the single
    # cmd recording sees a live bench and bakes in vkCmdWriteTimestamp calls.
    bench = BenchTimer(ctx, label="gpu")
    sim.bench = bench
    print(f"[bench_v3_single] timestampPeriod={bench.ns_per_tick:.4f}ns")

    rc = 0
    csv_file: Optional[TextIO] = None
    if args.bench_csv:
        csv_path = pathlib.Path(args.bench_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "w")
        csv_file.write(_csv_header())
        print(f"[bench_v3_single] CSV → {csv_path}")

    aggregator = WindowAggregator(window_size=args.bench_window)

    try:
        # Single-GPU bootstrap (sim.bootstrap() asserts no peer) + record
        # the combined step cmd buffer. step_single_use_split flag selects
        # _all vs split pipeline variants (P3.C validation harness).
        sim.bootstrap()
        sim.step_single_use_split = args.validate_split
        if args.validate_split:
            print(f"[bench_v3_single] VALIDATION MODE: using split pipeline variants")
        sim.prepare_step_single_cmd_buffer()

        t_start = time.perf_counter()
        frame_n = 0
        while frame_n < args.max_steps:
            t_frame_start = time.perf_counter_ns()
            sim.submit_step_single_and_wait()
            t_frame_end = time.perf_counter_ns()
            frame_n += 1

            # Defrag at cadence (independent submit + fence wait).
            defrag_ran = (frame_n % defrag_cadence == 0
                          and not args.no_defrag)
            if defrag_ran:
                sim.submit_defrag_and_wait()

            # Skip the warmup period entirely (covers JIT pipeline cache
            # warm-up + first-defrag cache-locality jump observed in dual).
            if frame_n <= args.warmup:
                continue

            ticks = bench.read_frame(include_defrag=defrag_ran)
            kernels = compute_durations(ticks)
            frame_total_us = (t_frame_end - t_frame_start) / 1000.0

            if csv_file is not None:
                csv_file.write(_csv_row(frame_n, frame_total_us, kernels))

            aggregator.add("frame_total_us", frame_total_us)
            for k in _KERNEL_KEYS:
                aggregator.add(k, kernels.get(k))

            if aggregator.tick():
                elapsed = time.perf_counter() - t_start
                fps = frame_n / elapsed
                header = (f"[bench] frames {frame_n-args.bench_window+1}"
                          f"..{frame_n}  fps={fps:6.1f}  "
                          f"(wall {elapsed:.1f}s)")
                print(aggregator.flush_summary(header),
                      file=sys.stderr, flush=True)
                if csv_file is not None:
                    csv_file.flush()

        total_elapsed = time.perf_counter() - t_start
        print(f"[bench_v3_single] {args.max_steps} steps in "
              f"{total_elapsed:.2f}s = {args.max_steps / total_elapsed:.1f} fps")

        # Final defrag to validate alive count (sanity check, not a metric)
        if not args.no_defrag:
            sim.submit_defrag_and_wait()
        status = sim.readback_global_status()
        alive = status["alive_particle_count"]
        print(f"[bench_v3_single] final: alive={alive:,} "
              f"(expected {expected_total:,})")
        if alive != expected_total:
            print(f"[bench_v3_single] WARN: alive drift = "
                  f"{alive - expected_total}", file=sys.stderr)

    finally:
        if csv_file is not None:
            csv_file.close()
        bench.destroy()
        sim.destroy()
        ctx.destroy()

    return rc


if __name__ == "__main__":
    sys.exit(main())
