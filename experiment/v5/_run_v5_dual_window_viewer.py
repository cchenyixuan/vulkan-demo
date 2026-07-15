"""
_run_v5_dual_window_viewer.py — V5 dual-GPU viewer, TWO windows (one per GPU).

Shows the multi-GPU domain decomposition live: GPU A renders its slab (left
half of the cavity) in window A, GPU B renders its slab (right half) in window
B, side by side. Together the two windows show the whole cavity split at the
partition line — you can watch each GPU's own particles plus (with the 'G'
hotkey) the 1-voxel ghost band each receives from its peer every step.

Contrast with the sibling runners:
  * _run_v5_single_viewer.py       — one GPU, whole cavity, ONE window (simplest
                                      way to just watch the physics).
  * _run_v5_dual_viewer.py         — dual-GPU compute but renders only ONE slab.
  * this file                      — dual-GPU compute AND renders BOTH slabs,
                                      one window per GPU.

Both GPUs run the full V5 3-submit pipeline via DualGpuOrchestratorV5 (ghost
transport + timeline sync + worker threads). Each frame:
    orch.step()                    # advance both GPUs one synchronous step
    renderer_a._draw_frame()       # present GPU A's slab
    renderer_b._draw_frame()       # present GPU B's slab

Requirement: BOTH physical devices must be present-capable for their window's
surface. On the 2x5090 rig this holds (NV WSI presents from either GPU); if a
device reports no present support the renderer raises with a clear message —
fall back to _run_v5_single_viewer.py or swap --device-a/-b.

Usage:
    VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation \\
        .venv/Scripts/python.exe experiment/v5/_run_v5_dual_window_viewer.py

Options:
    --case PATH          case.yaml (default: cavity 1M)
    --device-a/-b N      physical device indices (default 0 / 1). device-a
                         renders the left slab, device-b the right slab.
    --weights W0,W1      partition weights (default 1.0,1.0 — symmetric, natural
                         for the 2x5090 rig; use 3.2,1.0 for AMD+NV cross-vendor).
    --max-steps N        cap total steps (default: unlimited; ESC quits either window)
    --defrag-cadence N   override case.yaml
    --no-defrag          skip defrag entirely
    --disable-pst        override use_pst=False
    --validation         enable Vulkan validation layer
    --window-width/-height  per-window size (default 900x720)
    --auto-quit SECONDS  auto-close after N seconds (smoke test)

Control model: window A is the "control" window. SPACE on EITHER window toggles
a global pause (both sims halt together, since they are one simulation). Window
A's steps_per_frame (+/- and [ / ]) sets how many sim steps advance per rendered
frame. Camera / color-mode hotkeys are independent per window, so you can view
each slab from a different angle or in a different color mode.
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
    p = argparse.ArgumentParser(
        description="V5 dual-GPU viewer (two windows, one slab per GPU)")
    p.add_argument("--case", default="cases/lid_driven_cavity_2d/case.yaml")
    p.add_argument("--device-a", type=int, default=0)
    p.add_argument("--device-b", type=int, default=1)
    p.add_argument("--weights", default="1.0,1.0")
    p.add_argument("--max-steps", type=int, default=None,
                   help="cap total simulation steps; default = no cap (ESC quits)")
    p.add_argument("--defrag-cadence", type=int, default=None)
    p.add_argument("--no-defrag", action="store_true")
    p.add_argument("--disable-pst", action="store_true")
    p.add_argument("--validation", action="store_true")
    p.add_argument("--window-width", type=int, default=900)
    p.add_argument("--window-height", type=int, default=720)
    p.add_argument("--auto-quit", type=float, default=None,
                   help="auto-close both windows after N seconds (for smoke tests)")
    return p.parse_args()


def _glfw_required_extensions() -> list[str]:
    import glfw
    if not glfw.init():
        raise RuntimeError("glfw.init() failed")
    raw = glfw.get_required_instance_extensions()
    if raw is None:
        return []
    out = []
    for item in raw:
        out.append(item.decode("utf-8") if isinstance(item, bytes) else str(item))
    return out


def _run_two_window_loop(orch, renderer_a, renderer_b, *,
                         max_steps, auto_quit_seconds) -> None:
    """Combined event loop: poll both windows, advance the orchestrator, draw
    both slabs. Blocks until either window is closed (ESC / window-close) or
    auto_quit elapses."""
    import glfw
    from vulkan import vkDeviceWaitIdle

    win_a, win_b = renderer_a.window, renderer_b.window

    run_start = time.perf_counter()
    last_t = run_start
    frame_counter = 0

    # Global pause driven by SPACE on either window. Each renderer's own key
    # callback flips its .paused; we detect the edge and toggle a shared flag,
    # then mirror it back so both titles stay consistent.
    sim_paused = False
    prev_a, prev_b = renderer_a.paused, renderer_b.paused

    while not (glfw.window_should_close(win_a) or glfw.window_should_close(win_b)):
        glfw.poll_events()

        if (auto_quit_seconds is not None
                and time.perf_counter() - run_start >= auto_quit_seconds):
            break

        if renderer_a.paused != prev_a or renderer_b.paused != prev_b:
            sim_paused = not sim_paused
        renderer_a.paused = renderer_b.paused = sim_paused
        prev_a = prev_b = sim_paused

        if not sim_paused:
            # Window A is the control window for stepping cadence.
            for _ in range(renderer_a.steps_per_frame):
                if max_steps is not None and orch.frame_count >= max_steps:
                    break
                orch.step()

        renderer_a._draw_frame()
        renderer_b._draw_frame()

        frame_counter += 1
        now = time.perf_counter()
        if now - last_t > 0.5:
            fps = frame_counter / (now - last_t)
            step = orch.frame_count
            sim_time = step * renderer_a.case.physics.timestep
            pause_tag = "  [PAUSED]" if sim_paused else ""
            mode_names = ["speed", "accel", "density", "voxel_id", "kernel_sum"]
            glfw.set_window_title(
                win_a,
                f"SPH V5  GPU A (slab 0, left)   step={step}   "
                f"t={sim_time:.3e}s   {fps:.0f} fps   "
                f"spf={renderer_a.steps_per_frame}   "
                f"color={mode_names[renderer_a.color_mode]}{pause_tag}")
            glfw.set_window_title(
                win_b,
                f"SPH V5  GPU B (slab 1, right)   step={step}   "
                f"t={sim_time:.3e}s   {fps:.0f} fps   "
                f"color={mode_names[renderer_b.color_mode]}{pause_tag}")
            frame_counter = 0
            last_t = now

    vkDeviceWaitIdle(renderer_a.ctx.device)
    vkDeviceWaitIdle(renderer_b.ctx.device)


def _teardown_two_renderers(renderer_a, renderer_b) -> None:
    """Destroy both renderers' Vulkan + window resources without terminating
    GLFW between them. SphRendererV5.destroy() ends with glfw.terminate(),
    which nukes ALL windows — so the first destroy would orphan the second
    window. Neuter terminate for the per-renderer destroys, then let the caller
    terminate GLFW exactly once."""
    import glfw
    real_terminate = glfw.terminate
    glfw.terminate = lambda: None
    try:
        for r in (renderer_a, renderer_b):
            if r is None:            # skip a renderer that failed to construct
                continue
            try:
                r.destroy()
            except Exception as exc:  # teardown is best-effort at process exit
                print(f"[dual_window] renderer teardown warning: {exc}",
                      file=sys.stderr)
    finally:
        glfw.terminate = real_terminate


def main() -> int:
    args = parse_args()

    from experiment.v5.utils.case_loader_v5 import load_case_v5
    from experiment.v5.utils.orchestrator_v5 import DualGpuOrchestratorV5
    from experiment.v5.utils.partition_v5 import compute_dual_gpu_partition
    from experiment.v5.utils.renderer_v5 import SphRendererV5
    from experiment.v5.utils.simulator_v5 import SphSimulatorV5
    from experiment.v5.utils.vulkan_context_v5 import VulkanContextV5

    weights = [float(w) for w in args.weights.split(",")]
    if len(weights) != 2:
        sys.exit(f"--weights must have 2 values; got {args.weights!r}")
    print(f"[run_v5_dual_window] case={args.case} "
          f"device-a={args.device_a} device-b={args.device_b} weights={weights}")

    global_case = load_case_v5(args.case)
    if args.disable_pst:
        global_case.numerics.use_pst = False
    slab0, slab1, k_split = compute_dual_gpu_partition(global_case, weights)
    defrag_cadence = (args.defrag_cadence if args.defrag_cadence is not None
                      else global_case.numerics.defrag_cadence)
    if args.no_defrag:
        defrag_cadence = 10 ** 9

    # Both windows present, so both contexts need surface + swapchain extensions.
    glfw_exts = _glfw_required_extensions()
    print(f"[run_v5_dual_window] GLFW required instance extensions: {glfw_exts}")

    ctx_a = VulkanContextV5.create(
        device_index=args.device_a,
        enable_validation=args.validation,
        application_name="sph_v5_dualwin_a",
        extra_instance_extensions=glfw_exts,
        extra_device_extensions=["VK_KHR_swapchain"],
    )
    ctx_b = VulkanContextV5.create(
        device_index=args.device_b,
        enable_validation=args.validation,
        application_name="sph_v5_dualwin_b",
        extra_instance_extensions=glfw_exts,
        extra_device_extensions=["VK_KHR_swapchain"],
    )

    # Orchestrator requires sim_a = leftmost (trailing peer), sim_b = rightmost
    # (leading peer). slab0/slab1 come out of the partition in that order.
    sim_a = SphSimulatorV5(ctx_a, slab0)
    sim_b = SphSimulatorV5(ctx_b, slab1)

    rc = 0
    try:
        with DualGpuOrchestratorV5(sim_a, sim_b,
                                   defrag_cadence=defrag_cadence) as orch:
            orch.bootstrap_all()

            # Init to None and build BOTH renderers INSIDE the try so the
            # finally always tears down whichever were created. If renderer_b's
            # __init__ raises (e.g. device-b not present-capable), a fully-built
            # renderer_a must still be destroyed before the outer finally's
            # ctx_a.destroy() — otherwise vkDestroyDevice runs with live child
            # objects (a Vulkan usage violation).
            renderer_a = renderer_b = None
            try:
                renderer_a = SphRendererV5(sim_a,
                                           window_width=args.window_width,
                                           window_height=args.window_height)
                renderer_b = SphRendererV5(sim_b,
                                           window_width=args.window_width,
                                           window_height=args.window_height)
                # Offset window B to the right of window A so they don't stack.
                import glfw
                pos_a = glfw.get_window_pos(renderer_a.window)
                glfw.set_window_pos(
                    renderer_b.window,
                    int(pos_a[0]) + args.window_width + 30, int(pos_a[1]))

                _run_two_window_loop(
                    orch, renderer_a, renderer_b,
                    max_steps=args.max_steps,
                    auto_quit_seconds=args.auto_quit)
            finally:
                _teardown_two_renderers(renderer_a, renderer_b)

            s_a = sim_a.readback_global_status()
            s_b = sim_b.readback_global_status()
            total = s_a["alive_particle_count"] + s_b["alive_particle_count"]
            print(f"[run_v5_dual_window] final alive: "
                  f"a={s_a['alive_particle_count']:,} "
                  f"b={s_b['alive_particle_count']:,} total={total:,} "
                  f"(steps={orch.frame_count})")
    finally:
        sim_a.destroy()
        sim_b.destroy()
        ctx_a.destroy()
        ctx_b.destroy()
        import glfw
        glfw.terminate()

    return rc


if __name__ == "__main__":
    sys.exit(main())
