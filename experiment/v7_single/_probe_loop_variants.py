"""
_probe_loop_variants.py — diagnostic: wall time per step of v7_single's production step under three one-in-flight
host loops on one case, back to back in one process (measurement B's production loop read 7 % slower than its timed
loop on cavity3d_8m_b4; this isolates the loop from the command buffer):
  a. step_and_wait(): submit_step() + drain() per step (B's production loop);
  b. submit_step() per step with one fence (the wait happens at the top of the next submit; A's depth-1 loop);
  c. ctx.submit_and_wait(step_cmd): a fresh fence per step (B's timed loop, without the timestamps).
Each variant: ``--steps`` steps after a warmup, the defrag at the case's cadence appended / drained as the loop
does, repeated ``--rounds`` times interleaved (a b c a b c ...). Prints us/step per variant and round.

    .venv/Scripts/python.exe experiment/v7_single/_probe_loop_variants.py --case cases/cavity3d_8m_b4/case.yaml \\
        --gpu-uuid ae137c668a40f5acaab90a83f7cda175 --steps 300 --rounds 2
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True)
    parser.add_argument("--gpu-uuid", default="ae137c668a40f5acaab90a83f7cda175")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=2)
    arguments = parser.parse_args()
    from _run_single_baseline_bench import install_obj_cache
    install_obj_cache(_REPOSITORY_ROOT / "logs" / "_obj_npy_cache")
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7_single._run_single_bench import resolve_device_index
    from experiment.v7_single.utils.simulator_single import SphSimulatorSingle
    from experiment.v7_single.utils.vulkan_context_single import VulkanContextSingle
    case = load_case_v7(arguments.case)
    cadence = int(case.numerics.defrag_cadence)
    ctx = VulkanContextSingle.create(device_index=resolve_device_index(arguments.gpu_uuid), application_name="probe_loops")
    sim = SphSimulatorSingle(ctx, case)
    sim.bootstrap()
    sim.prepare_step(in_flight=1)
    frame = 0

    def loop_a(count: int) -> None:
        nonlocal frame
        for _ in range(count):
            frame += 1
            sim.step_and_wait()
            if frame % cadence == 0:
                sim.submit_defrag_and_wait()

    def loop_b(count: int) -> None:
        nonlocal frame
        for _ in range(count):
            frame += 1
            sim.submit_step(with_defrag=frame % cadence == 0)
        sim.drain()

    def loop_c(count: int) -> None:
        nonlocal frame
        for _ in range(count):
            frame += 1
            ctx.submit_and_wait(sim.step_cmd)
            if frame % cadence == 0:
                ctx.submit_and_wait(sim.defrag_cmd)

    loops = {"a step_and_wait": loop_a, "b submit_step ring(1)": loop_b, "c ctx.submit_and_wait": loop_c}
    loop_a(arguments.warmup)
    results = {name: [] for name in loops}
    for round_index in range(arguments.rounds):
        for name, loop in loops.items():
            sim.drain()
            started = time.perf_counter()
            loop(arguments.steps)
            elapsed = time.perf_counter() - started
            results[name].append(elapsed / arguments.steps * 1e6)
            print(f"[probe] round {round_index + 1} {name:24s} {elapsed / arguments.steps * 1e6:9.1f} us/step", flush=True)
    status = sim.readback_global_status()
    print(f"[probe] alive {status['alive_particle_count']:,} / {case.initial.positions.shape[0]:,}; "
          + "; ".join(f"{name}: {sum(values) / len(values):.1f} us/step" for name, values in results.items()))
    sim.destroy()
    ctx.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
