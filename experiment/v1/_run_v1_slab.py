"""
_run_v1_slab.py — V1 simplified: ONE GPU runs ITS slab WITHOUT ghost.

Use case: prove the slab partition mechanics (case filtering + shrunken
grid) work end-to-end on V0 SphSimulator. No multi-GPU coordination, no
ghost particles, no migration. Particles drifting past the slab edge are
killed (V0 out-of-grid behaviour). Density at the K_split column will
under-count because peer neighbours are absent — boundary physics is
expected to look weird; that's the point.

This is the baseline for "what does no ghost cost?". Next increment will
populate ghost SoA from peer's t=0 state and freeze it ("frozen ghost"),
then compare alive-drift / energy curves.

Usage:
    .venv/Scripts/python.exe experiment/v1/_run_v1_slab.py \\
        --device-index 0 \\
        --slot-index 0 \\
        --steps 500
"""

import argparse
import pathlib
import sys
import time

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import compile_shaders
from vulkan import vkGetPhysicalDeviceProperties

from utils.sph.case import load_case
from utils.sph.simulator import SphSimulator
from utils.sph.vulkan_context import VulkanContext

from experiment.v1.utils.case_slab import build_slab_case
from experiment.v1.utils.partition import compute_partition


def _run_headless(sim: SphSimulator, args, initial_alive: int) -> None:
    """Original headless step loop with periodic readback + drift print."""
    wall_start = time.perf_counter()
    for step_index in range(args.steps):
        sim.step()
        if (step_index + 1) % args.report_interval == 0:
            status = sim.readback_global_status()
            drift = status["alive_particle_count"] - initial_alive
            elapsed = time.perf_counter() - wall_start
            fps = (step_index + 1) / max(elapsed, 1e-9)
            print(f"[v1-slab]   step {step_index+1:>5}  "
                  f"alive={status['alive_particle_count']:>9,} "
                  f"(drift={drift:+d})  "
                  f"max_v={status['maximum_velocity']:.4f}  "
                  f"fps={fps:.1f}")
    wall_total = time.perf_counter() - wall_start
    final_status = sim.readback_global_status()
    print()
    print(f"[v1-slab] FINAL after {args.steps} steps in {wall_total:.2f}s "
          f"({args.steps / wall_total:.1f} fps):")
    print(f"[v1-slab]   alive            : {final_status['alive_particle_count']:,}")
    print(f"[v1-slab]   alive_drift      : "
          f"{final_status['alive_particle_count'] - initial_alive:+d}")
    print(f"[v1-slab]   max_velocity     : {final_status['maximum_velocity']:.4f}")
    print(f"[v1-slab]   overflow_inside  : "
          f"{final_status['overflow_inside_count']}")
    print(f"[v1-slab]   overflow_incoming: "
          f"{final_status['overflow_incoming_count']}")
    print(f"[v1-slab]   correction_fallback: "
          f"{final_status['correction_fallback_count']}")


DEFAULT_CASE = "cases/lid_driven_cavity_2d/case.yaml"
DEFAULT_GPU_NAMES = "NVIDIA GeForce RTX 4060 Ti,AMD Radeon RX 7900 XTX"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V1 single-GPU slab smoke test (no ghost, no migration).")
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--device-index", type=int, default=0,
                        help="physical device index to run on")
    parser.add_argument("--slot-index", type=int, default=0,
                        help="which partition slot's slab to run "
                             "(0 .. len(--gpu-names)-1)")
    parser.add_argument("--gpu-names", default=DEFAULT_GPU_NAMES,
                        help="comma-separated GPU names for partition lookup")
    parser.add_argument("--weights", default=None,
                        help="comma-separated explicit weights "
                             "(skips KNOWN_GPU_SPH_WEIGHT lookup)")
    parser.add_argument("--steps", type=int, default=500,
                        help="(headless mode only) number of steps to run")
    parser.add_argument("--report-interval", type=int, default=100,
                        help="(headless mode only) print readback every N steps")
    parser.add_argument("--viewer", action="store_true",
                        help="launch GLFW window with V0 SphRenderer (interactive); "
                             "starts paused — press SPACE to step")
    parser.add_argument("--no-validation", action="store_true")
    args = parser.parse_args()

    compile_shaders.compile_all(include_phase1=False)

    case = load_case(args.case)
    gpu_names = [name.strip() for name in args.gpu_names.split(",")]
    weights_override = None
    if args.weights:
        weights_override = [float(w.strip()) for w in args.weights.split(",")]

    partition = compute_partition(
        case, gpu_names, weights_override=weights_override)

    print()
    print(f"[v1-slab] case             : {args.case}")
    print(f"[v1-slab] partition K_split: {partition.voxel_x_split} / "
          f"{partition.grid_nx}")
    print(f"[v1-slab] running slot     : {args.slot_index} "
          f"({gpu_names[args.slot_index]!r})")

    slab_case = build_slab_case(case, partition, args.slot_index)
    n_slab = sum(s.vertices.shape[0] for s in slab_case.particle_sources)
    print(f"[v1-slab] slab particles   : {n_slab:,}")
    print(f"[v1-slab] slab grid        : "
          f"origin={tuple(round(v, 5) for v in slab_case.grid['origin'])}, "
          f"dim={slab_case.grid['dimension']}")
    print()

    extra_instance_extensions = None
    extra_device_extensions = None
    if args.viewer:
        import glfw
        if not glfw.init():
            raise RuntimeError("glfw.init() failed")
        extra_instance_extensions = list(glfw.get_required_instance_extensions())
        extra_device_extensions = ["VK_KHR_swapchain"]

    with VulkanContext.create(
        application_name="sph_v1_slab",
        enable_validation=(not args.no_validation),
        extra_instance_extensions=extra_instance_extensions,
        extra_device_extensions=extra_device_extensions,
        device_index=args.device_index,
    ) as ctx:
        device_props = vkGetPhysicalDeviceProperties(ctx.physical_device)
        gpu_name = str(device_props.deviceName)
        print(f"[v1-slab] device           : {gpu_name}")
        print()

        sim = SphSimulator(ctx, slab_case)
        try:
            sim.bootstrap()
            initial_alive = sim.readback_global_status()["alive_particle_count"]
            print(f"[v1-slab] initial alive    : {initial_alive:,}")
            print()

            if args.viewer:
                from utils.sph.renderer import SphRenderer
                with SphRenderer(sim, window_width=1280, window_height=720) as viewer:
                    viewer.run()
            else:
                _run_headless(sim, args, initial_alive)
        finally:
            sim.destroy()


if __name__ == "__main__":
    main()
