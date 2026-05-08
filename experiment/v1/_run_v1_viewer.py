"""
_run_v1_viewer.py — Live Vulkan + GLFW SPH viewer driving SphSimulatorV1.

Default V1.0a configuration: SINGLE-GPU V0-collapse mode
(leading_ghost_pool_size = trailing_ghost_pool_size = 0). The V1 buffer
layout collapses to V0's pid range, ghost_send / install_migrations
pipelines are not created, and the per-step kernel sequence is
numerically equivalent to V0 — useful for visually verifying that the
new V1 pipeline framework hasn't perturbed the physics.

Usage (run from repo root):
    .venv/Scripts/python.exe experiment/v1/_run_v1_viewer.py [case_path] [--log-fps PATH]

Default case: cases/lid_driven_cavity_2d/case.yaml.

Hotkeys / mouse / colour modes: same as V0's _run_viewer.py — handled by
SphRenderer (we reuse it; renderer reads only sim.{ctx,case,buffers,step,
step_count,simulation_time}, all provided by SphSimulatorV1).
"""

import argparse
import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import glfw

from utils.sph.case import load_case
from utils.sph.vulkan_context import VulkanContext
from utils.sph.renderer import SphRenderer

from experiment.v1 import compile_shaders_v1
from experiment.v1.utils.simulator_v1 import SphSimulatorV1


DEFAULT_CASE = "cases/lid_driven_cavity_2d/case.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live Vulkan + GLFW SPH viewer (V1.0a single-GPU).")
    parser.add_argument(
        "case",
        nargs="?",
        default=DEFAULT_CASE,
        help=f"case yaml path (default: {DEFAULT_CASE})",
    )
    parser.add_argument(
        "--log-fps",
        type=str,
        default=None,
        metavar="PATH",
        help="append per-window fps samples (CSV) to this file for benchmarking",
    )
    args = parser.parse_args()

    # Recompile V1 shaders on each run (same UX as V0 viewer).
    compile_shaders_v1.compile_v1_shaders()

    if not glfw.init():
        raise RuntimeError("glfw.init() failed")

    case = load_case(args.case)
    print(f"\n[v1-viewer] loaded {args.case}")
    print(f"[v1-viewer]   active particles: "
          f"{sum(s.vertices.shape[0] for s in case.particle_sources):,}")
    if args.log_fps:
        print(f"[v1-viewer]   logging fps to {args.log_fps}")

    required_extensions = list(glfw.get_required_instance_extensions())
    with VulkanContext.create(
        application_name="sph_v1_viewer",
        enable_validation=True,
        extra_instance_extensions=required_extensions,
        extra_device_extensions=["VK_KHR_swapchain"],
    ) as ctx:
        # V0-collapse mode: leading=trailing=0 → identical pid/voxel layout to V0.
        sim = SphSimulatorV1(ctx, case)
        try:
            sim.bootstrap()
            with SphRenderer(sim, window_width=1280, window_height=720) as viewer:
                viewer.run(log_fps_path=args.log_fps)
        finally:
            sim.destroy()


if __name__ == "__main__":
    main()
