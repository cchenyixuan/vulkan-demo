"""
_run_v3_dual_viewer.py — V3 dual-GPU viewer (GLFW window renders ONE slab).

Drives the full V3 dual-GPU pipeline (3-submit per frame + worker threads +
sync2 timeline + ghost transport) via DualGpuOrchestratorV3; renders the
chosen slab's particles in a GLFW window via SphRendererV3.

The non-rendered GPU still runs all its compute every step so the cross-GPU
ghost transport produces correct neighbors on the rendered side.

Usage:
    .venv/Scripts/python.exe experiment/v3/_run_v3_dual_viewer.py [options]

Options:
    --case PATH          case.yaml path
    --device-a/-b N      physical device indices (default 0 / 1)
    --weights W0,W1      partition weights  (default: 3.2,1.0 — V3 Path A+
                         optimum for AMD+NV cross-vendor cavity 1M)
    --render-slot 0|1    which slab to render in the window (default: 0)
    --max-steps N        cap total steps (default: unlimited; ESC to quit)
    --defrag-cadence N   override case.yaml
    --validation         enable Vulkan validation layer
    --disable-pst        override use_pst=False
    --no-defrag          skip defrag entirely

Hotkeys (handled by SphRendererV3): SPACE pause, 0..4 color modes,
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
    p = argparse.ArgumentParser(description="V3 dual-GPU viewer (renders one slab)")
    p.add_argument("--case", default="cases/lid_driven_cavity_2d/case.yaml")
    p.add_argument("--device-a", type=int, default=0)
    p.add_argument("--device-b", type=int, default=1)
    p.add_argument("--weights", default="3.2,1.0")
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

    from experiment.v3.utils.case_loader_v3 import load_case_v3
    from experiment.v3.utils.debug_log_v3 import DebugLogger
    from experiment.v3.utils.orchestrator_v3 import DualGpuOrchestratorV3
    from experiment.v3.utils.partition_v3 import compute_dual_gpu_partition
    from experiment.v3.utils.renderer_v3 import SphRendererV3
    from experiment.v3.utils.simulator_v3 import SphSimulatorV3
    from experiment.v3.utils.vulkan_context_v3 import VulkanContextV3

    weights = [float(w) for w in args.weights.split(",")]
    if len(weights) != 2:
        sys.exit(f"--weights must have 2 values; got {args.weights!r}")
    print(f"[run_v3_viewer] case={args.case} render_slot={args.render_slot}")

    global_case = load_case_v3(args.case)
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
    print(f"[run_v3_viewer] GLFW required instance extensions: {glfw_exts}")

    render_device_index = args.device_a if args.render_slot == 0 else args.device_b
    compute_only_device_index = args.device_b if args.render_slot == 0 else args.device_a

    ctx_render = VulkanContextV3.create(
        device_index=render_device_index,
        enable_validation=args.validation,
        application_name="sph_v3_dual_render",
        extra_instance_extensions=glfw_exts,
        extra_device_extensions=["VK_KHR_swapchain"],
        enable_shared_host_transport=True,
    )
    ctx_compute = VulkanContextV3.create(
        device_index=compute_only_device_index,
        enable_validation=args.validation,
        application_name="sph_v3_dual_compute",
        enable_shared_host_transport=True,
    )

    # Map back to sim_a / sim_b based on render_slot. Orchestrator always wants
    # sim_a = leftmost slab (trailing peer), sim_b = rightmost (leading peer).
    if args.render_slot == 0:
        sim_a = SphSimulatorV3(ctx_render, slab0)
        sim_b = SphSimulatorV3(ctx_compute, slab1)
        render_sim = sim_a
    else:
        sim_a = SphSimulatorV3(ctx_compute, slab0)
        sim_b = SphSimulatorV3(ctx_render, slab1)
        render_sim = sim_b

    rc = 0
    try:
        with DualGpuOrchestratorV3(sim_a, sim_b,
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
                        "render_slot": args.render_slot,
                        "max_steps": args.max_steps,
                        "defrag_cadence": defrag_cadence,
                        "runner": "_run_v3_dual_viewer",
                    },
                )

            try:
                renderer = SphRendererV3(
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
                        if logger is not None:
                            logger.tick(orch.frame_count)

                    renderer.run(
                        step_fn=step_fn,
                        step_count_fn=lambda: orch.frame_count,
                        auto_quit_seconds=args.auto_quit,
                    )
                finally:
                    renderer.destroy()

                s_a = sim_a.readback_global_status()
                s_b = sim_b.readback_global_status()
                print(f"[run_v3_viewer] final alive: "
                      f"a={s_a['alive_particle_count']:,} b={s_b['alive_particle_count']:,} "
                      f"total={s_a['alive_particle_count'] + s_b['alive_particle_count']:,}")
            finally:
                if logger is not None:
                    logger.close()
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
