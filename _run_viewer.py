"""Live Vulkan + GLFW SPH viewer.

Usage:
    .venv/Scripts/python.exe _run_viewer.py [case_path] [--log-fps PATH]

Default case: cases/lid_driven_cavity_2d/case.yaml.

Hotkeys:
    SPACE       pause/resume
    0..4        color mode: speed / accel / density / voxel_id / kernel_sum
    , / .       scale current color mode ÷1.5 / ×1.5
    P           perspective ↔ orthogonal
    F           re-frame to fit case bbox
    +/- or [/]  steps_per_frame ±1 / ÷×2
    ESC         quit
Mouse:
    left drag   orbit
    middle drag pan
    scroll      zoom

--log-fps PATH:
    Append per-window (~0.5s) fps samples to PATH as CSV
    (`wallclock_s,step_count,fps_window_avg`). Useful for back-to-back
    benchmark comparison runs (master vs. experiment branches).
"""

import argparse

import glfw

import compile_shaders

from utils.sph.case import load_case
from utils.sph.vulkan_context import VulkanContext
from utils.sph.simulator import SphSimulator
from utils.sph.renderer import SphRenderer


DEFAULT_CASE = "cases/lid_driven_cavity_2d/case.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Vulkan + GLFW SPH viewer.")
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

    # Recompile every shader on each run for fast iteration.
    compile_shaders.compile_all(include_phase1=False)

    if not glfw.init():
        raise RuntimeError("glfw.init() failed")

    case = load_case(args.case)
    print(f"\n[viewer] loaded {args.case}")
    print(f"[viewer]   active particles: "
          f"{sum(s.vertices.shape[0] for s in case.particle_sources):,}")
    if args.log_fps:
        print(f"[viewer]   logging fps to {args.log_fps}")

    required_extensions = list(glfw.get_required_instance_extensions())
    with VulkanContext.create(
        application_name="sph_v0_viewer",
        enable_validation=True,
        extra_instance_extensions=required_extensions,
        extra_device_extensions=["VK_KHR_swapchain"],
    ) as ctx:
        sim = SphSimulator(ctx, case)
        try:
            sim.bootstrap()
            with SphRenderer(sim, window_width=1280, window_height=720) as viewer:
                viewer.run(log_fps_path=args.log_fps)
        finally:
            sim.destroy()


if __name__ == "__main__":
    main()
