"""
_run_v2_dual_viewer.py — V2 dual-GPU viewer (GLFW window renders ONE slab).

Drives the full V2 dual-GPU pipeline (3-submit per frame + worker threads +
sync2 timeline + ghost transport) via DualGpuOrchestratorV2; renders the
chosen slab's particles in a GLFW window via SphRendererV2.

The non-rendered GPU still runs all its compute every step so the cross-GPU
ghost transport produces correct neighbors on the rendered side.

Usage:
    .venv/Scripts/python.exe experiment/v2/_run_v2_dual_viewer.py [options]

Options:
    --case PATH          case.yaml path
    --device-a/-b N      physical device indices (default 0 / 1)
    --weights W0,W1      partition weights  (default: 1.0,1.0)
    --render-slot 0|1    which slab to render in the window (default: 0)
    --max-steps N        cap total steps (default: unlimited; ESC to quit)
    --defrag-cadence N   override case.yaml
    --validation         enable Vulkan validation layer
    --disable-pst        override use_pst=False
    --no-defrag          skip defrag entirely

Hotkeys (handled by SphRendererV2): SPACE pause, 0..4 colour modes,
, / .  scale tune, P perspective ↔ ortho, F frame-fit, +/- steps_per_frame,
ESC quit, mouse drag = orbit / pan, scroll = zoom.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V2 dual-GPU viewer (renders one slab)")
    p.add_argument("--case", default="cases/lid_driven_cavity_2d/case.yaml")
    p.add_argument("--device-a", type=int, default=0)
    p.add_argument("--device-b", type=int, default=1)
    p.add_argument("--weights", default="1.0,1.0")
    p.add_argument("--render-slot", type=int, default=0, choices=(0, 1))
    p.add_argument("--max-steps", type=int, default=None,
                   help="cap total simulation steps; default = no cap (ESC quits)")
    p.add_argument("--defrag-cadence", type=int, default=None)
    p.add_argument("--validation", action="store_true")
    p.add_argument("--disable-pst", action="store_true")
    p.add_argument("--no-defrag", action="store_true")
    p.add_argument("--window-width", type=int, default=1280)
    p.add_argument("--window-height", type=int, default=720)
    p.add_argument("--auto-quit", type=float, default=None,
                   help="auto-close window after N seconds (for smoke tests)")
    return p.parse_args()


def _glfw_required_extensions() -> list[str]:
    import glfw
    if not glfw.init():
        raise RuntimeError("glfw.init() failed")
    raw = glfw.get_required_instance_extensions()
    if raw is None:
        return []
    # glfw returns either list of bytes or a single ctypes ptr; normalize to str list
    out = []
    for item in raw:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


def main() -> int:
    args = parse_args()

    from experiment.v2.utils.case_loader_v2 import load_case_v2
    from experiment.v2.utils.orchestrator_v2 import DualGpuOrchestratorV2
    from experiment.v2.utils.partition_v2 import compute_dual_gpu_partition
    from experiment.v2.utils.renderer_v2 import SphRendererV2
    from experiment.v2.utils.simulator_v2 import SphSimulatorV2
    from experiment.v2.utils.vulkan_context_v2 import VulkanContextV2

    weights = [float(w) for w in args.weights.split(",")]
    if len(weights) != 2:
        sys.exit(f"--weights must have 2 values; got {args.weights!r}")
    print(f"[run_v2_viewer] case={args.case} render_slot={args.render_slot}")

    global_case = load_case_v2(args.case)
    if args.disable_pst:
        global_case.numerics.use_pst = False
    slab0, slab1, k_split = compute_dual_gpu_partition(global_case, weights)
    defrag_cadence = (args.defrag_cadence if args.defrag_cadence is not None
                      else global_case.numerics.defrag_cadence)
    if args.no_defrag:
        defrag_cadence = 10**9

    # The rendered slot's context needs surface/swapchain extensions; the other
    # is compute-only (no swapchain). GLFW must be initialized before we ask
    # for required extensions.
    glfw_exts = _glfw_required_extensions()
    print(f"[run_v2_viewer] GLFW required instance extensions: {glfw_exts}")

    render_device_index = args.device_a if args.render_slot == 0 else args.device_b
    compute_only_device_index = args.device_b if args.render_slot == 0 else args.device_a

    ctx_render = VulkanContextV2.create(
        device_index=render_device_index,
        enable_validation=args.validation,
        application_name="sph_v2_dual_render",
        extra_instance_extensions=glfw_exts,
        extra_device_extensions=["VK_KHR_swapchain"],
    )
    ctx_compute = VulkanContextV2.create(
        device_index=compute_only_device_index,
        enable_validation=args.validation,
        application_name="sph_v2_dual_compute",
    )

    # Map back to sim_a / sim_b based on render_slot. Orchestrator always wants
    # sim_a = leftmost slab (trailing peer), sim_b = rightmost (leading peer).
    if args.render_slot == 0:
        sim_a = SphSimulatorV2(ctx_render, slab0)
        sim_b = SphSimulatorV2(ctx_compute, slab1)
        render_sim = sim_a
    else:
        sim_a = SphSimulatorV2(ctx_compute, slab0)
        sim_b = SphSimulatorV2(ctx_render, slab1)
        render_sim = sim_b

    rc = 0
    try:
        with DualGpuOrchestratorV2(sim_a, sim_b,
                                   defrag_cadence=defrag_cadence) as orch:
            orch.bootstrap_all()

            renderer = SphRendererV2(
                render_sim,
                window_width=args.window_width,
                window_height=args.window_height,
            )

            try:
                # step_fn: drive one orchestrator step (= one full dual-GPU
                # 3-submit frame, synchronous). Renderer calls this every
                # frame iteration when not paused. Respects --max-steps.
                def step_fn() -> None:
                    if args.max_steps is not None and orch.frame_count >= args.max_steps:
                        return
                    orch.step()

                renderer.run(
                    step_fn=step_fn,
                    step_count_fn=lambda: orch.frame_count,
                    auto_quit_seconds=args.auto_quit,
                )
            finally:
                renderer.destroy()

            s_a = sim_a.readback_global_status()
            s_b = sim_b.readback_global_status()
            print(f"[run_v2_viewer] final alive: "
                  f"a={s_a['alive_particle_count']:,} b={s_b['alive_particle_count']:,} "
                  f"total={s_a['alive_particle_count'] + s_b['alive_particle_count']:,}")
    finally:
        sim_a.destroy()
        sim_b.destroy()
        ctx_render.destroy()
        ctx_compute.destroy()
        import glfw
        glfw.terminate()

    return rc


if __name__ == "__main__":
    sys.exit(main())
