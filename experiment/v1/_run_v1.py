"""
_run_v1.py — V1 dual-GPU smoke test entry point.

V1.0 scaffolding milestone (#9 in the task list):
  * MultiGPUContext brings up two VulkanContext instances pinned to different
    physical devices.
  * Two V0 SphSimulators run, one per context, each given the FULL case
    (no partition yet — this is wasteful but proves dual-instance works).
  * Step both sequentially for a small fixed budget; assert no crash and
    that each sim's alive_count matches the expected initial value.

What this does NOT yet do:
  * No partition (each GPU runs the whole 1M particles redundantly)
  * No ghost / migration / cross-GPU sync
  * No per-step timing (V0 single-GPU values already established)

Once #10 (partition) lands, this entry point grows: each sim gets its own
slab subset and the alive counts sum to total. After #11/#12 (ghost +
migration shaders), #13's 9-phase blocking sync loop replaces the trivial
"step both sequentially" loop here.

Usage:
    .venv/Scripts/python.exe experiment/v1/_run_v1.py \\
        --case cases/lid_driven_cavity_2d/case.yaml \\
        --device-indices 0,1 \\
        --steps 100
"""

import argparse
import pathlib
import sys
import time

# Repo root on sys.path so 'utils.sph.*' and 'experiment.v1.utils.*' resolve.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import compile_shaders
from vulkan import vkGetPhysicalDeviceProperties

from utils.sph.case import load_case
from utils.sph.simulator import SphSimulator

from experiment.v1.utils.multi_gpu import MultiGPUContext


DEFAULT_CASE = "cases/lid_driven_cavity_2d/case.yaml"


def _parse_device_indices(text: str) -> list[int]:
    parts = [piece.strip() for piece in text.split(",") if piece.strip()]
    if len(parts) < 1:
        raise ValueError("--device-indices must contain at least one index")
    return [int(piece) for piece in parts]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V1 dual-GPU smoke test (#9 scaffolding).")
    parser.add_argument("--case", default=DEFAULT_CASE,
                        help=f"case yaml path (default: {DEFAULT_CASE})")
    parser.add_argument("--device-indices", default="0,1",
                        type=_parse_device_indices,
                        help="comma-separated physical device indices "
                             "(default: 0,1)")
    parser.add_argument("--steps", type=int, default=100,
                        help="number of step()s to run on each sim (default: 100)")
    parser.add_argument("--no-validation", action="store_true",
                        help="disable Vulkan validation layers")
    args = parser.parse_args()

    compile_shaders.compile_all(include_phase1=False)

    case = load_case(args.case)
    n_particles = sum(s.vertices.shape[0] for s in case.particle_sources)
    print()
    print(f"[v1-smoke] case        : {args.case}")
    print(f"[v1-smoke] particles   : {n_particles:,} (replicated across all GPUs in this test)")
    print(f"[v1-smoke] device_idx  : {args.device_indices}")
    print(f"[v1-smoke] steps_each  : {args.steps}")
    print()

    with MultiGPUContext.create(
        device_indices=args.device_indices,
        enable_validation=(not args.no_validation),
    ) as multi_ctx:
        # Per-GPU sanity print of the device names actually selected.
        for slot_index, ctx in enumerate(multi_ctx.contexts):
            props = vkGetPhysicalDeviceProperties(ctx.physical_device)
            print(f"[v1-smoke] slot {slot_index} : {str(props.deviceName)}")
        print()

        # ---- Build one SphSimulator per context ----------------------------
        # V0 SphSimulator unchanged. Each one independently allocates buffers,
        # uploads case state, and runs the full 1M-particle simulation on
        # its GPU. This is wasteful (no partition) but proves dual-instance.
        simulators: list[SphSimulator] = []
        try:
            for slot_index, ctx in enumerate(multi_ctx.contexts):
                print(f"[v1-smoke] building simulator on slot {slot_index}...")
                sim = SphSimulator(ctx, case)
                simulators.append(sim)
                sim.bootstrap()

            # ---- Step each simulator sequentially --------------------------
            print()
            print(f"[v1-smoke] stepping each simulator {args.steps} times "
                  f"(sequential, no inter-GPU sync)...")
            per_slot_wall = []
            for slot_index, sim in enumerate(simulators):
                wall_t0 = time.perf_counter()
                for _ in range(args.steps):
                    sim.step()
                wall_elapsed = time.perf_counter() - wall_t0
                per_slot_wall.append(wall_elapsed)
                fps = args.steps / max(wall_elapsed, 1e-9)
                ms = wall_elapsed / args.steps * 1000.0
                print(f"[v1-smoke]   slot {slot_index}: "
                      f"{args.steps} steps in {wall_elapsed:.2f}s "
                      f"({fps:.1f} fps, {ms:.2f} ms/step)")

            # ---- Final sanity: per-sim alive_count and overflow checks ----
            print()
            print(f"[v1-smoke] post-step sanity:")
            for slot_index, sim in enumerate(simulators):
                status = sim.readback_global_status()
                drift = status["alive_particle_count"] - n_particles
                ok = (drift == 0
                      and status["overflow_inside_count"] == 0
                      and status["overflow_incoming_count"] == 0)
                marker = "OK" if ok else "FAIL"
                print(f"[v1-smoke]   slot {slot_index} [{marker}] "
                      f"alive={status['alive_particle_count']:,} "
                      f"(drift={drift:+d}), "
                      f"overflow_inside={status['overflow_inside_count']}, "
                      f"overflow_incoming={status['overflow_incoming_count']}")
        finally:
            for sim in simulators:
                sim.destroy()

    print()
    print("[v1-smoke] done — both contexts torn down cleanly.")


if __name__ == "__main__":
    main()
