"""
_run_v5_3d_slice.py — single-GPU 3D run with periodic mid-plane slice renders.

First visual validation of the dimension:3 solver path (2026-07-23): runs a 3D
case on one GPU, and at each snapshot step reads back positions + velocities
and renders three panels into one PNG per snapshot:

  1. mid-z slice (x-y): speed field — the primary lid-driven vortex plane
  2. mid-z slice (x-y): velocity quiver (subsampled) — vortex topology
  3. mid-x slice (z-y): speed field — secondary/end-wall structure plane

Alive extraction mirrors _run_v5_snapshot_movie.py: velocity_mass[:, 3] > 0
over the own range. Single-GPU stepping is fence-waited per frame, so any
between-steps point is a safe readback point (no pipelining).

    .venv/Scripts/python.exe experiment/v5/_run_v5_3d_slice.py \
        --case cases/cavity3d_1m/case.yaml --device 1 \
        --snapshots 2000,4000,8000,12000 --out logs/cavity3d_first_light
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="V5 single-GPU 3D slice snapshots")
    parser.add_argument("--case", default="cases/cavity3d_1m/case.yaml")
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--snapshots", default="2000,4000,8000,12000",
                        help="comma-separated step numbers to render")
    parser.add_argument("--slice-half-width", type=float, default=1.5,
                        help="slice half-thickness in units of dx")
    parser.add_argument("--out", default="logs/cavity3d_first_light")
    parser.add_argument("--defrag-cadence", type=int, default=None)
    parser.add_argument("--validation", action="store_true")
    return parser.parse_args()


def render_snapshot(out_path, step, positions, velocities, dx, lid_speed=1.0,
                    slice_half_width=1.5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    speed = np.sqrt((velocities ** 2).sum(axis=1))
    window = slice_half_width * dx
    mid_z = np.abs(positions[:, 2]) < window
    mid_x = np.abs(positions[:, 0]) < window

    figure, axes = plt.subplots(1, 3, figsize=(19, 6))

    scatter = axes[0].scatter(positions[mid_z, 0], positions[mid_z, 1],
                              c=speed[mid_z], s=0.5, cmap="viridis",
                              vmin=0.0, vmax=lid_speed)
    axes[0].set_title(f"step {step}: mid-z slice speed (x-y), "
                      f"n={int(mid_z.sum()):,}")
    axes[0].set_aspect("equal")
    figure.colorbar(scatter, ax=axes[0], fraction=0.046)

    # Quiver: subsample the mid-z slice on a coarse grid for readability.
    slice_positions = positions[mid_z]
    slice_velocities = velocities[mid_z]
    if slice_positions.shape[0] > 0:
        stride = max(1, slice_positions.shape[0] // 1400)
        order = np.argsort(slice_positions[:, 0], kind="stable")
        pick = order[::stride]
        axes[1].quiver(slice_positions[pick, 0], slice_positions[pick, 1],
                       slice_velocities[pick, 0], slice_velocities[pick, 1],
                       np.sqrt((slice_velocities[pick, :2] ** 2).sum(axis=1)),
                       cmap="viridis", scale=lid_speed * 25.0, width=0.0022,
                       clim=(0.0, lid_speed))
    axes[1].set_title("mid-z velocity quiver (vortex topology)")
    axes[1].set_aspect("equal")
    axes[1].set_xlim(axes[0].get_xlim())
    axes[1].set_ylim(axes[0].get_ylim())

    scatter2 = axes[2].scatter(positions[mid_x, 2], positions[mid_x, 1],
                               c=speed[mid_x], s=0.5, cmap="viridis",
                               vmin=0.0, vmax=lid_speed)
    axes[2].set_title("mid-x slice speed (z-y): end-wall structure")
    axes[2].set_aspect("equal")
    figure.colorbar(scatter2, ax=axes[2], fraction=0.046)

    figure.tight_layout()
    figure.savefig(out_path, dpi=110)
    plt.close(figure)


def main() -> int:
    args = parse_args()

    from experiment.v5.utils.case_loader_v5 import load_case_v5
    from experiment.v5.utils.simulator_v5 import SphSimulatorV5
    from experiment.v5.utils.vulkan_context_v5 import VulkanContextV5

    snapshot_steps = sorted(int(v) for v in args.snapshots.split(","))
    out_dir = _REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    case = load_case_v5(args.case)
    if case.physics.dimension != 3:
        print(f"WARNING: case dimension={case.physics.dimension}, expected 3")
    expected_total = int(case.initial.positions.shape[0])
    # dx is not stored on PhysicsConstants; the 3D generator uses h/dx=4, and
    # the slice window is forgiving — derive an effective dx from h/4.
    dx = case.physics.smoothing_length / 4.0
    defrag_cadence = (args.defrag_cadence if args.defrag_cadence is not None
                      else case.numerics.defrag_cadence)

    ctx = VulkanContextV5.create(device_index=args.device,
                                 enable_validation=args.validation,
                                 application_name="sph_v5_3d_slice")
    sim = SphSimulatorV5(ctx, case)
    rc = 0
    try:
        sim.bootstrap()
        sim.prepare_step_single_cmd_buffer()

        pool = case.capacities.total_pool_capacity()
        own_first = sim.own_first_pid()
        own_slice = slice(own_first, own_first + case.capacities.own_pool_size)

        t_start = time.perf_counter()
        frame_n = 0
        for target in snapshot_steps:
            while frame_n < target:
                sim.submit_step_single_and_wait()
                frame_n += 1
                if frame_n % defrag_cadence == 0:
                    sim.submit_defrag_and_wait()

            raw = sim.readback_buffers_batch(["position_voxel_id", "velocity_mass"])
            position_voxel = np.frombuffer(raw["position_voxel_id"],
                                           np.float32).reshape(pool, 4)
            velocity_mass = np.frombuffer(raw["velocity_mass"],
                                          np.float32).reshape(pool, 4)
            alive = velocity_mass[own_slice, 3] > 0
            positions = position_voxel[own_slice, 0:3][alive]
            velocities = velocity_mass[own_slice, 0:3][alive]
            alive_count = int(alive.sum())
            status = sim.readback_global_status()
            elapsed = time.perf_counter() - t_start
            print(f"[3d_slice] step {frame_n}: alive={alive_count:,} "
                  f"(status {status['alive_particle_count']:,} / "
                  f"expected {expected_total:,}) "
                  f"fps_so_far={frame_n/elapsed:.1f}", flush=True)
            if status["alive_particle_count"] != expected_total:
                print(f"[3d_slice] *** DRIFT: "
                      f"{status['alive_particle_count'] - expected_total} ***")
                rc = 1

            out_path = out_dir / f"slice_step{frame_n:06d}.png"
            render_snapshot(out_path, frame_n, positions, velocities, dx,
                            slice_half_width=args.slice_half_width)
            print(f"[3d_slice] wrote {out_path}", flush=True)

        elapsed = time.perf_counter() - t_start
        print(f"[3d_slice] total: {frame_n} steps in {elapsed:.1f}s "
              f"= {frame_n/elapsed:.1f} fps (includes readback+render stalls)")
    finally:
        sim.destroy()
        ctx.destroy()
    return rc


if __name__ == "__main__":
    sys.exit(main())
