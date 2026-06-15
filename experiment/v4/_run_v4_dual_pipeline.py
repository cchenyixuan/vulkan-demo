"""
_run_v4_dual_pipeline.py — V4 dual-GPU SUBMIT-AHEAD (pipelined) runner.

Drives DualGpuOrchestratorV4.run_pipelined(depth), which keeps `depth` frames in
flight so the GPU never idles on the CPU submit/wait round-trip that the
synchronous depth-1 step() pays every frame. On cavity 1M this recovered ~+12%
(the inter-frame GPU bubble) in the V2-architecture prototype; depth-2 captures
essentially all of it (depth-3 adds <1%).

Safety (single-buffered state): the 5N timeline makes worker(n)'s host-signal a
prerequisite for frame n+1's transfer-queue activity, so no staging buffer is
reused before its reader finishes and no host-signal goes backwards — validated
drift=0 at depth 2 and 3. See orchestrator_v4.run_pipelined docstring.

This runner does NOT collect per-kernel GPU timestamps (the in-flight next frame
overwrites the query-pool slots); it measures wall-clock fps + the per-defrag
migration series. For per-stage timing use _run_v4_dual_bench.py (depth-1).

Usage:
    .venv/Scripts/python.exe experiment/v4/_run_v4_dual_pipeline.py \\
        --depth 2 --weights 2.9,1.0 --pool-safety 1.2 \\
        --max-steps 20000 --warmup 5000
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V4 dual-GPU submit-ahead pipeline runner")
    p.add_argument("--case", default="cases/lid_driven_cavity_2d/case.yaml")
    p.add_argument("--device-a", type=int, default=0)
    p.add_argument("--device-b", type=int, default=1)
    p.add_argument("--weights", default="2.9,1.0")
    p.add_argument("--depth", type=int, default=2, help="frames in flight (1=synchronous baseline)")
    p.add_argument("--pool-safety", type=float, default=None,
                   help="per-slab own_pool_size = ceil(slab_particles*FACTOR); None=global pool")
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--warmup", type=int, default=5000)
    p.add_argument("--defrag-cadence", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from experiment.v4.utils.case_loader_v4 import load_case_v4
    from experiment.v4.utils.orchestrator_v4 import DualGpuOrchestratorV4
    from experiment.v4.utils.partition_v4 import compute_dual_gpu_partition
    from experiment.v4.utils.simulator_v4 import SphSimulatorV4
    from experiment.v4.utils.vulkan_context_v4 import VulkanContextV4

    weights = [float(w) for w in args.weights.split(",")]
    global_case = load_case_v4(args.case)
    expected_total = int(global_case.initial.positions.shape[0])
    slab0, slab1, _k = compute_dual_gpu_partition(
        global_case, weights, pool_safety=args.pool_safety)

    defrag_cadence = (args.defrag_cadence if args.defrag_cadence is not None
                      else global_case.numerics.defrag_cadence)

    ctx_a = VulkanContextV4.create(device_index=args.device_a, application_name="pipe_v4_a")
    ctx_b = VulkanContextV4.create(device_index=args.device_b, application_name="pipe_v4_b")
    sim_a = SphSimulatorV4(ctx_a, slab0)
    sim_b = SphSimulatorV4(ctx_b, slab1)

    print(f"[pipe_v4] depth={args.depth} weights={weights} "
          f"pool_safety={args.pool_safety} max_steps={args.max_steps} "
          f"warmup={args.warmup} defrag_cadence={defrag_cadence}")

    def on_defrag(frame_n: int, report: list) -> None:
        a, b = report[0], report[1]
        print(f"[migration] frame {frame_n:>6}  interval "
              f"AMD={a['interval_migration']:>5} NV={b['interval_migration']:>5}  "
              f"|  peak/run AMD={a['peak_migration']:>5} NV={b['peak_migration']:>5}  "
              f"|  peak_tail AMD={a['peak_tail']:,} ({a['used_fraction']*100:.1f}%) "
              f"NV={b['peak_tail']:,} ({b['used_fraction']*100:.1f}%)  "
              f"|  drops AMD={a['overflow_install_tail']} NV={b['overflow_install_tail']}",
              file=sys.stderr, flush=True)

    try:
        with DualGpuOrchestratorV4(sim_a, sim_b, defrag_cadence=defrag_cadence) as orch:
            orch.bootstrap_all()
            result = orch.run_pipelined(
                args.max_steps, depth=args.depth, warmup=args.warmup, on_defrag=on_defrag)

            print(f"[pipe_v4] TOTAL: {result['frame_count']} steps in "
                  f"{result['elapsed_s']:.2f}s = {result['fps']:.1f} fps")
            if "steady_fps" in result:
                print(f"[pipe_v4] STEADY (post-warmup {args.warmup}): "
                      f"{result['steady_frames']} steps in {result['steady_s']:.2f}s "
                      f"= {result['steady_fps']:.1f} fps")

            sim_a.submit_defrag_and_wait()
            sim_b.submit_defrag_and_wait()
            s_a = sim_a.readback_global_status()
            s_b = sim_b.readback_global_status()
            total = s_a["alive_particle_count"] + s_b["alive_particle_count"]
            print(f"[pipe_v4] final: a={s_a['alive_particle_count']:,} "
                  f"b={s_b['alive_particle_count']:,} total={total:,} "
                  f"(expected {expected_total:,}) drift={total - expected_total}")
            for tag, sim, status in (("AMD/a", sim_a, s_a), ("NV/b", sim_b, s_b)):
                ph = sim.readback_pool_health()
                flag = ("  *** OVERFLOWED ***" if ph["free_margin"] <= 0
                        else "  ** WARN <10% margin **" if ph["used_fraction"] > 0.9
                        else "")
                print(f"[pipe_v4] pool_health[{tag}]: "
                      f"peak_tail={ph['peak_tail_high_water']:,} / "
                      f"own_pool={ph['own_pool_size']:,}  used={ph['used_fraction']*100:.1f}%  "
                      f"margin={ph['free_margin']:,}  peak_migration={ph['peak_migration_count']:,}"
                      f"  install_tail_drops={status['overflow_install_tail']}{flag}")
    finally:
        sim_a.destroy()
        sim_b.destroy()
        ctx_a.destroy()
        ctx_b.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
