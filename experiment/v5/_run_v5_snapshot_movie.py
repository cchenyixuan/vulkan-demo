"""
_run_v5_snapshot_movie.py — dual-GPU run with periodic GLOBAL particle
snapshots rendered to a PNG frame sequence, for visual bug hunting.

Every defrag boundary (pipeline drained — the only safe readback point),
both sims' {position_voxel_id, velocity_mass, density_pressure} are read
back, own-range alive particles from the two slabs are merged into the
global frame (slab positions ARE global — partition only shifts each
slab's voxel-grid origin), and a 3-panel frame is rendered off-thread:

  1. mean-speed field  (fixed color scale 0..lid speed)
  2. particle count per pixel (holes / clumps / ghost double-draw detector)
  3. slab-seam zoom scatter, colored by source GPU (migration/install bugs)

Outputs under --out-dir:
    frames/frame_%05d.png      one per defrag interval (~8.3 s at 8M)
    snaps/snap_%05d.npz        merged raw arrays every --npz-every frames
    meta.json / run.log

Usage (30 min, 8M):
    .venv/Scripts/python.exe experiment/v5/_run_v5_snapshot_movie.py \\
        --case cases/lid_driven_cavity_2d_8m/case.yaml --minutes 30
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import queue
import sys
import threading
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

BLUE = "#2a78d6"
GREEN = "#008300"


class _RunDone(Exception):
    pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V5 dual-GPU snapshot-movie runner")
    p.add_argument("--case", default="cases/lid_driven_cavity_2d_8m/case.yaml")
    p.add_argument("--device-a", type=int, default=0)
    p.add_argument("--device-b", type=int, default=1)
    p.add_argument("--weights", default="1.0,1.0")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--sync-scheme", default="per-direction",
                   choices=["aggregated", "per-direction"])
    p.add_argument("--minutes", type=float, default=30.0)
    p.add_argument("--pool-safety", type=float, default=None)
    p.add_argument("--defrag-cadence", type=int, default=None,
                   help="also the snapshot cadence (readback at defrag drain)")
    p.add_argument("--npz-every", type=int, default=36,
                   help="save merged raw npz every Nth snapshot (~5 min at 8M)")
    p.add_argument("--bins", type=int, default=900)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


class SnapshotRenderer(threading.Thread):
    """Off-thread renderer: consumes merged snapshot dicts, writes PNGs.
    Bounded queue gives backpressure instead of unbounded RAM growth."""

    def __init__(self, out_dir: pathlib.Path, extent: tuple, x_cut: float,
                 seam_half_width: float, bins: int, lid_speed: float,
                 expected_total: int):
        super().__init__(name="snapshot-renderer", daemon=False)
        self.jobs: queue.Queue = queue.Queue(maxsize=2)
        self.out_dir = out_dir
        self.extent = extent          # (x0, x1, y0, y1)
        self.x_cut = x_cut
        self.seam_half_width = seam_half_width
        self.bins = bins
        self.lid_speed = lid_speed
        self.expected_total = expected_total
        self.last_error = None
        self.rendered = 0

    def run(self) -> None:
        try:
            while True:
                job = self.jobs.get()
                if job is None:
                    return
                self._render(job)
                self.rendered += 1
        except BaseException as e:  # noqa: BLE001
            self.last_error = e
            print(f"[renderer] DIED: {e!r}", file=sys.stderr, flush=True)

    def _render(self, job: dict) -> None:
        index = job["index"]
        x0, x1, y0, y1 = self.extent
        position = job["position"]
        speed = job["speed"]
        source = job["source"]        # 0 = sim A, 1 = sim B

        fig, axes = plt.subplots(1, 3, figsize=(19, 6.4), dpi=110)

        # Panel 1: mean speed field
        count, xe, ye = np.histogram2d(
            position[:, 0], position[:, 1], bins=self.bins,
            range=[[x0, x1], [y0, y1]])
        speed_sum, _, _ = np.histogram2d(
            position[:, 0], position[:, 1], bins=self.bins,
            range=[[x0, x1], [y0, y1]], weights=speed)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_speed = speed_sum / count
        ax = axes[0]
        # Empty pixels (count=0) render light gray — a hole in the fluid
        # interior shows up immediately.
        speed_cmap = plt.get_cmap("viridis").copy()
        speed_cmap.set_bad("#dddddd")
        im = ax.imshow(np.ma.masked_invalid(mean_speed.T), origin="lower",
                       extent=(x0, x1, y0, y1),
                       cmap=speed_cmap, vmin=0.0, vmax=self.lid_speed,
                       interpolation="nearest")
        ax.axvline(self.x_cut, color="white", linewidth=0.6, alpha=0.6)
        fig.colorbar(im, ax=ax, shrink=0.85, label="|v| (m/s)")
        ax.set_title(f"mean speed   frame {index}   "
                     f"t={job['sim_time']:.3f}s   step={job['step']:,}")

        # Panel 2: particle count per pixel (holes / clumps / duplicates)
        fluid_pixels = max((count > 0).sum(), 1)
        expected_per_pixel = len(position) / fluid_pixels
        ax = axes[1]
        im = ax.imshow(count.T, origin="lower", extent=(x0, x1, y0, y1),
                       cmap="Blues", vmin=0.0, vmax=3.0 * expected_per_pixel,
                       interpolation="nearest")
        ax.axvline(self.x_cut, color="red", linewidth=0.6, alpha=0.6)
        fig.colorbar(im, ax=ax, shrink=0.85, label="particles / pixel")
        alive_a = int((source == 0).sum())
        alive_b = int((source == 1).sum())
        drift = len(position) - self.expected_total
        ax.set_title(f"count map   A={alive_a:,}  B={alive_b:,}  drift={drift}")

        # Panel 3: seam zoom, colored by source GPU
        seam_mask = np.abs(position[:, 0] - self.x_cut) < self.seam_half_width
        seam_position = position[seam_mask]
        seam_source = source[seam_mask]
        ax = axes[2]
        for value, color, label in ((0, BLUE, "sim A"), (1, GREEN, "sim B")):
            sel = seam_source == value
            ax.scatter(seam_position[sel, 0], seam_position[sel, 1],
                       s=0.3, c=color, alpha=0.45, linewidths=0, label=label)
        ax.axvline(self.x_cut, color="red", linewidth=0.8, alpha=0.7)
        ax.set_xlim(self.x_cut - self.seam_half_width,
                    self.x_cut + self.seam_half_width)
        ax.set_ylim(y0, y1)
        ax.set_aspect("auto")
        ax.legend(fontsize=8, frameon=False, markerscale=12, loc="upper right")
        ax.set_title(f"seam zoom (±{self.seam_half_width * 1000:.1f} mm), "
                     f"seam particles={seam_mask.sum():,}")

        fig.tight_layout()
        fig.savefig(self.out_dir / "frames" / f"frame_{index:05d}.png",
                    facecolor="white")
        plt.close(fig)


def main() -> int:
    args = parse_args()

    from experiment.v5.utils.case_loader_v5 import load_case_v5
    from experiment.v5.utils.orchestrator_v5 import DualGpuOrchestratorV5
    from experiment.v5.utils.partition_v5 import compute_dual_gpu_partition
    from experiment.v5.utils.simulator_v5 import SphSimulatorV5
    from experiment.v5.utils.vulkan_context_v5 import VulkanContextV5

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (pathlib.Path(args.out_dir) if args.out_dir
               else pathlib.Path("logs") / f"snapmovie_{stamp}")
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    (out_dir / "snaps").mkdir(parents=True, exist_ok=True)

    weights = [float(w) for w in args.weights.split(",")]
    global_case = load_case_v5(args.case)
    expected_total = int(global_case.initial.positions.shape[0])
    slab0, slab1, k_split = compute_dual_gpu_partition(
        global_case, weights, pool_safety=args.pool_safety)
    defrag_cadence = (args.defrag_cadence if args.defrag_cadence is not None
                      else global_case.numerics.defrag_cadence)

    grid = global_case.grid
    h = global_case.physics.smoothing_length
    extent = (grid.origin_x, grid.origin_x + grid.grid_dimension_x * h,
              grid.origin_y, grid.origin_y + grid.grid_dimension_y * h)
    x_cut = grid.origin_x + k_split * h
    seam_half_width = 8.0 * h
    lid_speed = 1.0
    timestep = global_case.physics.timestep
    budget_s = args.minutes * 60.0

    meta = {
        "case": args.case, "weights": weights, "depth": args.depth,
        "sync_scheme": args.sync_scheme, "minutes": args.minutes,
        "defrag_cadence": defrag_cadence, "expected_total": expected_total,
        "k_split": int(k_split), "x_cut": x_cut, "extent": extent,
        "timestep": timestep,
        "started_iso": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[snapmovie] out_dir={out_dir}")
    print(f"[snapmovie] {json.dumps(meta)}")

    renderer = SnapshotRenderer(out_dir, extent, x_cut, seam_half_width,
                                args.bins, lid_speed, expected_total)
    renderer.start()

    ctx_a = VulkanContextV5.create(device_index=args.device_a, application_name="snap_v5_a")
    ctx_b = VulkanContextV5.create(device_index=args.device_b, application_name="snap_v5_b")
    sim_a = SphSimulatorV5(ctx_a, slab0, sync_scheme=args.sync_scheme)
    sim_b = SphSimulatorV5(ctx_b, slab1, sync_scheme=args.sync_scheme)

    state = {"t_start": None, "snapshot_index": 0}
    READBACK_NAMES = ["position_voxel_id", "velocity_mass", "density_pressure"]

    def take_snapshot(frame_n: int) -> None:
        index = state["snapshot_index"]
        state["snapshot_index"] += 1
        merged_position, merged_speed, merged_source = [], [], []
        merged_velocity, merged_density = [], []
        for source_value, sim in ((0, sim_a), (1, sim_b)):
            capacities = sim.case.capacities
            pool = capacities.total_pool_capacity()
            raw = sim.readback_buffers_batch(READBACK_NAMES)
            position_voxel = np.frombuffer(
                raw["position_voxel_id"], np.float32).reshape(pool, 4)
            velocity_mass = np.frombuffer(
                raw["velocity_mass"], np.float32).reshape(pool, 4)
            density_pressure = np.frombuffer(
                raw["density_pressure"], np.float32).reshape(pool, 2)
            own_first = sim.own_first_pid()
            own_slice = slice(own_first, own_first + capacities.own_pool_size)
            alive = velocity_mass[own_slice, 3] > 0
            merged_position.append(position_voxel[own_slice, 0:2][alive].copy())
            velocity = velocity_mass[own_slice, 0:2][alive]
            merged_velocity.append(velocity.copy())
            merged_speed.append(np.sqrt((velocity ** 2).sum(axis=1)))
            merged_density.append(density_pressure[own_slice, 0][alive].copy())
            merged_source.append(np.full(int(alive.sum()), source_value, np.int8))

        job = {
            "index": index,
            "step": frame_n,
            "sim_time": frame_n * timestep,
            "position": np.concatenate(merged_position),
            "speed": np.concatenate(merged_speed),
            "source": np.concatenate(merged_source),
        }
        if renderer.last_error is not None:
            raise RuntimeError(f"renderer died: {renderer.last_error!r}")
        renderer.jobs.put(job)   # blocks if renderer is 2 frames behind

        if index % args.npz_every == 0:
            np.savez_compressed(
                out_dir / "snaps" / f"snap_{index:05d}.npz",
                step=frame_n,
                position=job["position"],
                velocity=np.concatenate(merged_velocity),
                density=np.concatenate(merged_density),
                source=job["source"])

    try:
        with DualGpuOrchestratorV5(sim_a, sim_b, defrag_cadence=defrag_cadence) as orch:
            orch.bootstrap_all()

            def on_defrag(frame_n: int, report: list) -> None:
                take_snapshot(frame_n)
                elapsed = time.perf_counter() - state["t_start"]
                if state["snapshot_index"] % 10 == 1:
                    print(f"[snapmovie] {datetime.datetime.now().isoformat(timespec='seconds')} "
                          f"snapshot {state['snapshot_index']} frame={frame_n:,} "
                          f"elapsed={elapsed/60:.1f}min", flush=True)
                if elapsed >= budget_s:
                    raise _RunDone()

            state["t_start"] = time.perf_counter()
            max_steps_bound = int(budget_s * 250)
            try:
                orch.run_pipelined(max_steps_bound, depth=args.depth,
                                   warmup=0, on_defrag=on_defrag)
            except _RunDone:
                pass

            s_a = sim_a.readback_global_status()
            s_b = sim_b.readback_global_status()
            total = s_a["alive_particle_count"] + s_b["alive_particle_count"]
            print(f"[snapmovie] final: a={s_a['alive_particle_count']:,} "
                  f"b={s_b['alive_particle_count']:,} total={total:,} "
                  f"(expected {expected_total:,}) drift={total - expected_total}")
            print(f"[snapmovie] snapshots taken: {state['snapshot_index']}")
    finally:
        renderer.jobs.put(None)
        renderer.join(timeout=120)
        print(f"[snapmovie] frames rendered: {renderer.rendered}")
        sim_a.destroy()
        sim_b.destroy()
        ctx_a.destroy()
        ctx_b.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
