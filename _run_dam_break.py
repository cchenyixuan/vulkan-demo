"""Live Vulkan + GLFW viewer for SPH dam_break_2d.

Hotkeys:
    SPACE       pause/resume
    0..3        color mode: speed / accel / density / voxel_id
    P           perspective ↔ orthogonal
    F           re-frame to fit case bbox
    +/- or [/]  steps_per_frame ±1 / ÷×2
    ESC         quit
Mouse:
    left drag   orbit
    middle drag pan
    scroll      zoom
"""

import glfw

import compile_shaders

from utils.sph.case import load_case
from utils.sph.vulkan_context import VulkanContext
from utils.sph.simulator import SphSimulator
from utils.sph.renderer import SphRenderer


def main() -> None:
    # Recompile every shader on each run — convenient when iterating on
    # shader code. Skip phase1 demo shaders (irrelevant to the SPH viewer).
    compile_shaders.compile_all(include_phase1=False)

    # GLFW must be initialized before querying its required Vulkan extensions.
    if not glfw.init():
        raise RuntimeError("glfw.init() failed")

    case = load_case("cases/dam_break_2d/case.yaml")
    required_instance_extensions = list(glfw.get_required_instance_extensions())

    with VulkanContext.create(
        application_name="sph_v0_viewer",
        enable_validation=True,
        extra_instance_extensions=required_instance_extensions,
        extra_device_extensions=["VK_KHR_swapchain"],
    ) as ctx:
        sim = SphSimulator(ctx, case)
        try:
            sim.bootstrap()
            with SphRenderer(sim, window_width=1280, window_height=720) as viewer:
                viewer.run()
        finally:
            sim.destroy()


if __name__ == "__main__":
    main()
