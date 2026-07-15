"""
_run_v5_single_viewer.py — V5 single-GPU viewer (whole cavity, ONE window).

The simplest way to *see* a case: one GPU runs the ENTIRE domain (no slab
split, no ghost flow, no worker threads) and a GLFW window renders every
particle via SphRendererV5. Use this when you want to watch the physics —
e.g. the lid-driven cavity's corner vortices spin up — rather than study the
multi-GPU decomposition (for that, see _run_v5_dual_window_viewer.py, which
shows each GPU's slab in its own window).

Pipeline per frame (all synchronous, one queue):
    sim.submit_step_single_and_wait()   # predict+voxel+correction+density+force
    (defrag at cadence)                 # sim.submit_defrag_and_wait()
    renderer._draw_frame()              # point-sprite render of all pids

This reuses the exact single-GPU step path validated by
_run_v5_single_bench.py (SphSimulatorV5.submit_step_single_and_wait) and the
unmodified SphRendererV5 — the degenerate no-partition case has ghost pools of
size 0, so own_first_pid()=1 .. own_last_pid()=pool_size and the renderer's
"own" draw call paints the full cavity.

Usage:
    .venv/Scripts/python.exe experiment/v5/_run_v5_single_viewer.py [options]

    # 2x5090 rig: device 0 is the display GPU, so keep --device 0 for the
    # window (present-capable queue), and disable validation for clean fps:
    VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation \\
        .venv/Scripts/python.exe experiment/v5/_run_v5_single_viewer.py

Options:
    --case PATH          case.yaml (default: cavity 1M)
    --device N           physical device index for the sim + window (default 0;
                         on the 2x5090 rig device 0 drives the display so its
                         graphics/compute queue is present-capable)
    --max-steps N        cap total simulation steps (default: unlimited; ESC quits)
    --defrag-cadence N   override case.yaml defrag cadence
    --no-defrag          skip defrag entirely
    --disable-pst        override use_pst=False
    --validation         enable Vulkan validation layer
    --window-width/-height  initial window size (default 1280x720)
    --auto-quit SECONDS  auto-close after N seconds (smoke test)

Hotkeys (SphRendererV5): SPACE pause, 0..4 color mode
    (0 speed, 1 accel, 2 density dev, 3 voxel id, 4 kernel sum),
    , / .  tune current mode's color scale, P perspective<->ortho,
    F frame-fit to case bbox, +/- steps_per_frame +-1, [ / ] halve/double it,
    ESC quit. Mouse: left drag orbit, middle drag pan, scroll zoom.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="V5 single-GPU viewer (whole cavity, one window)")
    p.add_argument("--case", default="cases/lid_driven_cavity_2d/case.yaml")
    p.add_argument("--device", type=int, default=0,
                   help="physical device index for the sim + window (default 0)")
    p.add_argument("--max-steps", type=int, default=None,
                   help="cap total simulation steps; default = no cap (ESC quits)")
    p.add_argument("--defrag-cadence", type=int, default=None,
                   help="override case.yaml defrag cadence")
    p.add_argument("--no-defrag", action="store_true",
                   help="skip defrag entirely")
    p.add_argument("--disable-pst", action="store_true",
                   help="override use_pst=False")
    p.add_argument("--validation", action="store_true",
                   help="enable Vulkan validation layer")
    p.add_argument("--window-width", type=int, default=1280)
    p.add_argument("--window-height", type=int, default=720)
    p.add_argument("--auto-quit", type=float, default=None,
                   help="auto-close window after N seconds (for smoke tests)")
    return p.parse_args()


def _glfw_required_extensions() -> list[str]:
    """GLFW's required instance extensions (VK_KHR_surface + platform surface).
    glfw.init() must run before this; the returned list feeds VulkanContextV5."""
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


def main() -> int:
    args = parse_args()

    from experiment.v5.utils.case_loader_v5 import load_case_v5
    from experiment.v5.utils.renderer_v5 import SphRendererV5
    from experiment.v5.utils.simulator_v5 import SphSimulatorV5
    from experiment.v5.utils.vulkan_context_v5 import VulkanContextV5

    print(f"[run_v5_single_viewer] case={args.case} device={args.device}")

    case = load_case_v5(args.case)
    if args.disable_pst:
        print("[run_v5_single_viewer] use_pst overridden to False")
        case.numerics.use_pst = False

    expected_total = int(case.initial.positions.shape[0])
    defrag_cadence = (args.defrag_cadence if args.defrag_cadence is not None
                      else case.numerics.defrag_cadence)
    if args.no_defrag:
        defrag_cadence = 10 ** 9   # never triggers

    # The window's context needs the surface + swapchain extensions. GLFW must
    # be initialized before we can query its required instance extensions.
    glfw_exts = _glfw_required_extensions()
    print(f"[run_v5_single_viewer] GLFW required instance extensions: {glfw_exts}")

    ctx = VulkanContextV5.create(
        device_index=args.device,
        enable_validation=args.validation,
        application_name="sph_v5_single_viewer",
        extra_instance_extensions=glfw_exts,
        extra_device_extensions=["VK_KHR_swapchain"],
    )

    sim = SphSimulatorV5(ctx, case)

    rc = 0
    try:
        # Single-GPU bootstrap (asserts no peer) + record the combined step cmd
        # — identical to _run_v5_single_bench.py's setup.
        sim.bootstrap()
        sim.prepare_step_single_cmd_buffer()

        # Frame counter drives --max-steps + defrag cadence + the title's
        # step/sim-time readout (single-GPU sim carries no step counter).
        state = {"n": 0}

        def step_fn() -> None:
            if args.max_steps is not None and state["n"] >= args.max_steps:
                return
            sim.submit_step_single_and_wait()
            state["n"] += 1
            if (not args.no_defrag) and (state["n"] % defrag_cadence == 0):
                sim.submit_defrag_and_wait()

        renderer = SphRendererV5(
            sim,
            window_width=args.window_width,
            window_height=args.window_height,
        )
        try:
            renderer.run(
                step_fn=step_fn,
                step_count_fn=lambda: state["n"],
                auto_quit_seconds=args.auto_quit,
            )
        finally:
            renderer.destroy()

        # Sanity: defrag once more so alive count is exact, then report drift.
        if not args.no_defrag:
            sim.submit_defrag_and_wait()
        status = sim.readback_global_status()
        alive = status["alive_particle_count"]
        print(f"[run_v5_single_viewer] final: steps={state['n']} "
              f"alive={alive:,} (expected {expected_total:,})")
        if alive != expected_total:
            print(f"[run_v5_single_viewer] WARN: alive drift = "
                  f"{alive - expected_total}", file=sys.stderr)
    finally:
        sim.destroy()
        ctx.destroy()
        import glfw
        glfw.terminate()

    return rc


if __name__ == "__main__":
    sys.exit(main())
