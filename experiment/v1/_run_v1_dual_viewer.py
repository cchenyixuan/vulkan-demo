"""
_run_v1_dual_viewer.py — V1.0a dual-GPU viewer (renders ONE GPU's slab).

Runs the FULL dual-GPU pipeline (predict + update_voxel + ghost_send on each
GPU → cross-GPU CPU staging transport → install_migrations + correction +
density + force on each GPU), and shows the chosen slab's particles in a
GLFW window for visual sanity check.

The non-rendered GPU is still doing all its compute every step (so the cross-
GPU sync produces correct ghost data on the rendered side); we just don't
draw its particles.

Usage (run from repo root):
    .venv/Scripts/python.exe experiment/v1/_run_v1_dual_viewer.py [case] [options]

Options:
    --render-slot 0|1     which slab to render (default 0 = leftmost)
    --device-a / -b N     physical device index for slot 0 / 1
                          (defaults pick first 2 discrete GPUs)
    --gpu-names "X,Y"     for partition.compute_partition (default = current pair)
    --log-fps PATH        same as V0 viewer

Hotkeys (handled by SphRenderer): SPACE pause, 0..4 colour modes, , / .
scale, P perspective, F frame-fit, +/- steps_per_frame, ESC quit.

Caveats (V1.0a + minimal driver):
  - Defrag is DISABLED for the dual-GPU run (single-GPU defrag still works
    if you change the slab path back to _run_v1_viewer.py).
  - step(wait=False) optimisation is NOT used; every step waits on 5 fences
    sequentially (3 fences for transport + 2 for compute pre/post). FPS is
    therefore lower than full-overlap V2 will deliver.
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

from experiment.v1 import compile_shaders_v1
from experiment.v1.utils.case_slab import build_slab_case
from experiment.v1.utils.multi_gpu import MultiGPUContext
from experiment.v1.utils.partition import compute_partition
from experiment.v1.utils.renderer_v1 import SphRendererV1
from experiment.v1.utils.simulator_v1 import SphSimulatorV1
from experiment.v1.utils.transport_config import build_per_gpu_layouts
from experiment.v1.utils.transport_cpu_staging import CpuStagingMultiGpuTransport


DEFAULT_CASE = "cases/lid_driven_cavity_2d/case.yaml"
DEFAULT_GPU_NAMES = "NVIDIA GeForce RTX 4060 Ti,AMD Radeon RX 7900 XTX"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V1.0a dual-GPU viewer (renders one slab).")
    parser.add_argument("case", nargs="?", default=DEFAULT_CASE,
                        help=f"case yaml path (default: {DEFAULT_CASE})")
    parser.add_argument("--render-slot", type=int, default=0, choices=(0, 1),
                        help="which slab to render: 0 = leftmost, 1 = rightmost")
    parser.add_argument("--device-a", type=int, default=1,
                        help="physical device index for slot 0 (default 1 = NVIDIA)")
    parser.add_argument("--device-b", type=int, default=0,
                        help="physical device index for slot 1 (default 0 = AMD)")
    parser.add_argument("--gpu-names", default=DEFAULT_GPU_NAMES,
                        help="comma-separated GPU names for partition computation")
    parser.add_argument("--weights", default=None,
                        help="comma-separated explicit weights override")
    parser.add_argument("--log-fps", type=str, default=None, metavar="PATH",
                        help="append per-window fps samples (CSV) to PATH")
    args = parser.parse_args()

    # Recompile V1 shaders fresh (matches single-GPU viewer UX).
    compile_shaders_v1.compile_v1_shaders()

    if not glfw.init():
        raise RuntimeError("glfw.init() failed")

    # ---- Load case + partition ------------------------------------------
    case = load_case(args.case)
    gpu_names = [n.strip() for n in args.gpu_names.split(",")]
    weights_override = (
        [float(w.strip()) for w in args.weights.split(",")] if args.weights else None
    )
    partition = compute_partition(case, gpu_names, weights_override=weights_override)
    layouts = build_per_gpu_layouts(partition, case)

    print(f"\n[v1-dual] loaded {args.case}")
    print(f"[v1-dual]   K_split = {partition.voxel_x_split} / {partition.grid_nx}")
    for slot, gp in enumerate(partition.gpu_partitions):
        print(f"[v1-dual]   slot {slot} {gp.gpu_name!r}: own_x=[{gp.own_voxel_x_range[0]},"
              f"{gp.own_voxel_x_range[1]})  particles={gp.particle_count():,}")
    print(f"[v1-dual]   rendering slot {args.render_slot}")

    # ---- Two VulkanContexts (one per device) with swapchain extension ---
    required_extensions = list(glfw.get_required_instance_extensions())
    multi_ctx = MultiGPUContext.create(
        device_indices=[args.device_a, args.device_b],
        application_name_prefix="sph_v1_dual",
        enable_validation=True,
        extra_instance_extensions=required_extensions,
        extra_device_extensions=["VK_KHR_swapchain"],
    )
    sims: list[SphSimulatorV1] = []
    transport: CpuStagingMultiGpuTransport | None = None
    try:
        # ---- Build per-slot SphSimulatorV1 ------------------------------
        for slot, layout in enumerate(layouts):
            slab_case = build_slab_case(case, partition, slot)
            # V1.0a smoke: disable per-step defrag (dual-GPU defrag coordination
            # is V1.x). Mutate a private copy of numerics to clear the flag.
            slab_case.numerics.defrag_enabled = False
            sims.append(SphSimulatorV1(
                multi_ctx[slot], slab_case,
                leading_ghost_pool_size=layout.leading_ghost_pool_size,
                trailing_ghost_pool_size=layout.trailing_ghost_pool_size,
                leading_ghost_voxel_count=layout.leading_ghost_voxel_count,
                trailing_ghost_voxel_count=layout.trailing_ghost_voxel_count,
                ghost_voxel_x_thickness_leading=layout.ghost_voxel_x_thickness_leading,
                ghost_voxel_x_thickness_trailing=layout.ghost_voxel_x_thickness_trailing,
                leading_transport_config=layout.leading_transport_config,
                trailing_transport_config=layout.trailing_transport_config,
            ))

        transport = CpuStagingMultiGpuTransport(sims)

        # ---- Dual-GPU bootstrap ----------------------------------------
        print(f"\n[v1-dual] bootstrap...")
        for sim in sims:
            sim.ctx.submit_and_wait(sim.bootstrap_pre_sync_cmd)
        transport.transfer()
        for sim in sims:
            sim.ctx.submit_and_wait(sim.bootstrap_post_sync_cmd)
        for slot, sim in enumerate(sims):
            s = sim.readback_global_status()
            print(f"[v1-dual]   slot {slot} alive={s['alive_particle_count']:,} "
                  f"ovf_inside={s['overflow_inside_count']} "
                  f"ovf_install={s['overflow_install_inside']}")

        # ---- Monkeypatch rendered sim's step() with dual-GPU dance -----
        rendered = sims[args.render_slot]

        def _dual_step(wait: bool = True) -> None:
            for sim in sims:
                sim.ctx.submit_and_wait(sim.step_pre_sync_cmd)
            transport.transfer()
            for sim in sims:
                sim.ctx.submit_and_wait(sim.step_post_sync_cmd)
            for sim in sims:
                sim.simulation_time += sim.case.timestep
                sim.step_count += 1

        rendered.step = _dual_step

        print(f"[v1-dual] starting renderer on slot {args.render_slot}...")

        # ---- Hand off to SphRendererV1 ---------------------------------
        with SphRendererV1(rendered, window_width=1280, window_height=720) as viewer:
            viewer.run(log_fps_path=args.log_fps)
        print(f"[v1-dual] renderer exited cleanly")

    finally:
        # Explicit cleanup ordering: transport first (uses both ctxs),
        # then sims (per-ctx state), then multi_ctx (devices).
        print(f"[v1-dual] cleanup: transport...")
        if transport is not None:
            transport.destroy()
        print(f"[v1-dual] cleanup: {len(sims)} sims...")
        for sim in sims:
            sim.destroy()
        print(f"[v1-dual] cleanup: multi_ctx...")
        multi_ctx.destroy()
        print(f"[v1-dual] cleanup done")


if __name__ == "__main__":
    main()
