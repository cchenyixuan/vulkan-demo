"""
_run_v2_dual.py — V2 dual-GPU headless runner.

Runs the full V2 dual-GPU pipeline (3-submit + worker threads + sync2 timeline
+ ghost transport) on a YAML case, prints per-N-step status, ends with a
defrag-validated alive count.

Usage:
    .venv/Scripts/python.exe experiment/v2/_run_v2_dual.py [options]

Options (all optional):
    --case PATH          case.yaml path  (default: cases/lid_driven_cavity_2d/case.yaml)
    --device-a N         physical device index for sim_a (default: 0)
    --device-b N         physical device index for sim_b (default: 1)
    --weights W0,W1      partition weights  (default: 3.2,1.0 — empirically
                         optimal for V2 Path A+ on AMD 7900 XTX + NV 4060 Ti
                         cavity 1M; override for other hardware/cases)
    --max-steps N        steps to run  (default: 1000)
    --status-every N     print global_status every N steps  (default: 100)
    --defrag-cadence N   override case's defrag_cadence  (default: from case.yaml)
    --validation         enable Vulkan validation layer  (slow; default off)
    --disable-pst        override case to use_pst=False
    --no-defrag          skip defrag entirely

No GLFW / viewer here. For visualization a separate _run_v2_dual_viewer.py
is needed (Phase 5+ work; ~1500 LoC for renderer + swapchain integration).
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V2 dual-GPU headless runner")
    p.add_argument("--case", default="cases/lid_driven_cavity_2d/case.yaml")
    p.add_argument("--device-a", type=int, default=0)
    p.add_argument("--device-b", type=int, default=1)
    p.add_argument("--weights", default="3.2,1.0",
                   help="partition weights (W_AMD, W_NV). Default 3.2,1.0 is "
                        "the V2 Path A+ optimum for cross-vendor AMD 7900 XTX "
                        "+ NV 4060 Ti cavity 1M, found by sweep (277 fps peak). "
                        "Override for other hardware: same-vendor should try "
                        "1.0,1.0 first; pure single-GPU use 1.0,1e6 to force "
                        "all particles onto sim_a.")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--status-every", type=int, default=100)
    p.add_argument("--defrag-cadence", type=int, default=None)
    p.add_argument("--validation", action="store_true")
    p.add_argument("--disable-pst", action="store_true")
    p.add_argument("--no-defrag", action="store_true")
    # Debug logging (opt-in)
    p.add_argument("--debug-log", type=str, default=None,
                   help="enable debug log; arg = output directory (e.g. logs/run_001)")
    p.add_argument("--debug-log-every", type=int, default=50,
                   help="frames between CSV log writes")
    p.add_argument("--debug-snapshot-every", type=int, default=500,
                   help="frames between full buffer snapshots; 0 disables")
    p.add_argument("--debug-snapshot-format", choices=["npz", "h5"], default="npz")
    p.add_argument("--debug-snapshot-no-compress", action="store_true",
                   help="disable npz/h5 compression (~3-5× larger files)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from experiment.v2.utils.case_loader_v2 import load_case_v2
    from experiment.v2.utils.debug_log_v2 import DebugLogger
    from experiment.v2.utils.orchestrator_v2 import DualGpuOrchestratorV2
    from experiment.v2.utils.partition_v2 import compute_dual_gpu_partition
    from experiment.v2.utils.simulator_v2 import SphSimulatorV2
    from experiment.v2.utils.vulkan_context_v2 import VulkanContextV2

    weights = [float(w) for w in args.weights.split(",")]
    if len(weights) != 2:
        sys.exit(f"--weights must have 2 values; got {args.weights!r}")

    print(f"[run_v2_dual] case={args.case}")
    print(f"[run_v2_dual] device_a={args.device_a} device_b={args.device_b} "
          f"weights={weights}")

    global_case = load_case_v2(args.case)
    if args.disable_pst:
        print("[run_v2_dual] use_pst overridden to False")
        global_case.numerics.use_pst = False

    expected_total = int(global_case.initial.positions.shape[0])

    slab0, slab1, k_split = compute_dual_gpu_partition(global_case, weights)
    defrag_cadence = (args.defrag_cadence if args.defrag_cadence is not None
                      else global_case.numerics.defrag_cadence)
    if args.no_defrag:
        defrag_cadence = args.max_steps + 1  # never triggers

    ctx_a = VulkanContextV2.create(
        device_index=args.device_a, enable_validation=args.validation,
        application_name="sph_v2_dual_a")
    ctx_b = VulkanContextV2.create(
        device_index=args.device_b, enable_validation=args.validation,
        application_name="sph_v2_dual_b")

    sim_a = SphSimulatorV2(ctx_a, slab0)
    sim_b = SphSimulatorV2(ctx_b, slab1)

    rc = 0
    try:
        with DualGpuOrchestratorV2(sim_a, sim_b,
                                    defrag_cadence=defrag_cadence) as orch:
            orch.bootstrap_all()

            logger = None
            if args.debug_log:
                logger = DebugLogger(
                    output_dir=args.debug_log,
                    sims={"a": sim_a, "b": sim_b},
                    log_every=args.debug_log_every,
                    snapshot_every=(args.debug_snapshot_every
                                    if args.debug_snapshot_every > 0 else None),
                    snapshot_format=args.debug_snapshot_format,
                    snapshot_compressed=not args.debug_snapshot_no_compress,
                    meta_extra={
                        "case": args.case,
                        "weights": weights,
                        "max_steps": args.max_steps,
                        "defrag_cadence": defrag_cadence,
                        "runner": "_run_v2_dual",
                    },
                )

            try:
                t_start = time.perf_counter()
                next_status = args.status_every
                while orch.frame_count < args.max_steps:
                    orch.step()
                    if logger is not None:
                        logger.tick(orch.frame_count)
                    if orch.frame_count >= next_status:
                        s_a = sim_a.readback_global_status()
                        s_b = sim_b.readback_global_status()
                        elapsed = time.perf_counter() - t_start
                        fps = orch.frame_count / elapsed
                        print(f"  step {orch.frame_count:5d}  fps={fps:6.1f}  "
                              f"a: alive={s_a['alive_particle_count']:7d} "
                              f"ofl_in={s_a['overflow_inside_count']:5d} "
                              f"@v={s_a['first_overflow_voxel_inside']:5d} "
                              f"ofl_inc={s_a['overflow_incoming_count']:6d} "
                              f"@v={s_a['first_overflow_voxel_incoming']:5d} "
                              f"install={s_a['migration_install_count']:5d}  "
                              f"|  b: alive={s_b['alive_particle_count']:7d} "
                              f"ofl_in={s_b['overflow_inside_count']:5d} "
                              f"@v={s_b['first_overflow_voxel_inside']:5d} "
                              f"ofl_inc={s_b['overflow_incoming_count']:6d} "
                              f"@v={s_b['first_overflow_voxel_incoming']:5d} "
                              f"install={s_b['migration_install_count']:5d}",
                              flush=True)
                        next_status += args.status_every

                total_elapsed = time.perf_counter() - t_start
                print(f"[run_v2_dual] {args.max_steps} steps in {total_elapsed:.2f}s "
                      f"= {args.max_steps / total_elapsed:.1f} fps")

                # Final defrag to validate alive count
                if not args.no_defrag:
                    print("[run_v2_dual] final defrag …")
                    sim_a.submit_defrag_and_wait()
                    sim_b.submit_defrag_and_wait()
                s_a = sim_a.readback_global_status()
                s_b = sim_b.readback_global_status()
                total = s_a["alive_particle_count"] + s_b["alive_particle_count"]
                print(f"[run_v2_dual] final: a={s_a['alive_particle_count']:,} "
                      f"b={s_b['alive_particle_count']:,} "
                      f"total={total:,} (expected {expected_total:,})")
                if total != expected_total:
                    print(f"[run_v2_dual] WARN: alive count drift = "
                          f"{total - expected_total}", file=sys.stderr)
                    rc = 1
            finally:
                if logger is not None:
                    logger.close()
    finally:
        # Destroy sims AFTER orchestrator stops its workers (handled by
        # orchestrator's destroy() inside the `with`).
        sim_a.destroy()
        sim_b.destroy()
        ctx_a.destroy()
        ctx_b.destroy()

    return rc


if __name__ == "__main__":
    sys.exit(main())
