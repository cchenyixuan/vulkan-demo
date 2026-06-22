"""
debug_log_v4.py — periodic CSV log + ckpt-style buffer snapshots for V4 sims.

Purpose: long-running debug instrumentation. Every ``log_every`` frames write
a CSV row per sim with all global_status counters + voxel inside/incoming sums.
Every ``snapshot_every`` frames additionally dump all sim buffers to disk
(npz or h5) so post-mortem analysis (or simulation resume) is possible without
having to re-run.

Usage:

    from experiment.v4.utils.debug_log_v4 import DebugLogger

    with DualGpuOrchestratorV4(sim_a, sim_b, ...) as orch:
        orch.bootstrap_all()

        logger = DebugLogger(
            output_dir="logs/run_001",
            sims={"a": sim_a, "b": sim_b},
            log_every=50,
            snapshot_every=500,
        )

        try:
            while orch.frame_count < max_steps:
                orch.step()
                logger.tick(orch.frame_count)
        finally:
            logger.close()

Output layout:

    output_dir/
    ├── meta.json                          one-shot write at init
    ├── status.csv                         one row per sim per log tick
    └── snapshots/
        ├── sim_a/
        │   ├── frame_000500.npz           full buffer dump
        │   └── frame_001000.npz
        └── sim_b/
            └── ...

Each snapshot includes EVERY buffer in sim.buffers — enough to resume the
simulation from this frame (combined with the case yaml + partition weights
recorded in meta.json). Resume logic itself is out of scope here; this module
only produces the artifacts.

Disk usage: cavity 1M ~50 MB / snapshot (npz_compressed), ~225 MB raw.
10 000 step run with snapshot_every=500 = 20 snapshots × 2 sims ≈ 2 GB.
"""

from __future__ import annotations

import csv
import json
import pathlib
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable, Literal, Optional

import numpy as np

if TYPE_CHECKING:
    from experiment.v4.utils.simulator_v4 import SphSimulatorV4


# ============================================================================
# Buffer view registry — (dtype, shape_fn(case)) per buffer name
# ============================================================================
# shape_fn returns the desired numpy shape; (-1,) means "treat as flat".
# These reinterpret raw bytes from readback into typed numpy arrays for
# saving + downstream analysis. The pid / vid axis is always the LEADING
# dimension so slicing buf[pid] / buf[vid] reads naturally.

_BufferView = tuple[type, Callable[..., tuple]]

_BUFFER_VIEWS: dict[str, _BufferView] = {
    # ---- Set 0: per-particle SoA ------------------------------------------
    "position_voxel_id":           (np.float32, lambda c: (c.capacities.total_pool_capacity(), 4)),
    "density_pressure":            (np.float32, lambda c: (c.capacities.total_pool_capacity(), 2)),
    "density_pressure_scratch":    (np.float32, lambda c: (c.capacities.total_pool_capacity(), 2)),
    "velocity_mass":               (np.float32, lambda c: (c.capacities.total_pool_capacity(), 4)),
    "acceleration":                (np.float32, lambda c: (c.capacities.total_pool_capacity(), 4)),
    "shift":                       (np.float32, lambda c: (c.capacities.total_pool_capacity(), 4)),
    "material":                    (np.uint32,  lambda c: (c.capacities.total_pool_capacity(),)),
    "correction_inverse":          (np.float32, lambda c: (c.capacities.total_pool_capacity() * 2, 4)),
    "density_gradient_kernel_sum": (np.float32, lambda c: (c.capacities.total_pool_capacity(), 4)),
    "extension_fields":            (np.float32, lambda c: (c.capacities.total_pool_capacity(), 4)),

    # ---- Set 1: per-voxel --------------------------------------------------
    "inside_particle_count":       (np.uint32, lambda c: (1 + c.grid.total_voxel_count(),)),
    "incoming_particle_count":     (np.uint32, lambda c: (1 + c.grid.total_voxel_count(),)),
    "inside_particle_index":       (np.uint32, lambda c: (1 + c.grid.total_voxel_count(),
                                                          c.capacities.max_particles_per_voxel)),
    "incoming_particle_index":     (np.uint32, lambda c: (1 + c.grid.total_voxel_count(),
                                                          c.capacities.max_incoming_per_voxel)),
    "voxel_base_offset":           (np.uint32, lambda c: (1 + c.grid.total_voxel_count(),)),

    # ---- Set 3: globals + pool-health + materials -------------------------
    "global_status":               (np.uint32, lambda c: (16,)),
    "pool_health":                 (np.uint32, lambda c: (4,)),    # V4 watermark (reclaims overflow_log)
    # V4 cleanup: inlet_template / dispatch_indirect / ghost_out_packet /
    # ghost_in_staging / diagnostic removed (never read/written by any kernel).
    "material_parameters":         (np.uint8,  lambda c: (-1,)),   # 48 B per material; treat as raw
    "defrag_scratch_counter":      (np.uint32, lambda c: (1,)),
}


def reinterpret_buffer(raw_bytes: bytes, name: str, case) -> np.ndarray:
    """Wrap raw readback bytes as a typed numpy array based on buffer name."""
    if name not in _BUFFER_VIEWS:
        return np.frombuffer(raw_bytes, dtype=np.uint8)
    dtype, shape_fn = _BUFFER_VIEWS[name]
    arr = np.frombuffer(raw_bytes, dtype=dtype)
    shape = shape_fn(case)
    if shape == (-1,):
        return arr
    return arr.reshape(shape)


# ============================================================================
# Per-sim voxel range bookkeeping
# ============================================================================

@dataclass
class _SimVoxelRanges:
    """Cached voxel_id range partitioning per sim's extended grid.

    Voxel_id layout (1-based, with vid 0 sentinel unused):
        [1, leading_ghost_end]                  = leading ghost
        (leading_ghost_end, own_end]            = own
        (own_end, total]                        = trailing ghost
    """
    total: int
    leading_ghost_end: int     # vid range upper bound (inclusive)
    own_end: int               # vid range upper bound (inclusive)

    @classmethod
    def from_sim(cls, sim: "SphSimulatorV4") -> "_SimVoxelRanges":
        case = sim.case
        total = case.grid.total_voxel_count()
        leading_count = case.ghost_grid.leading_ghost_voxel_count
        trailing_count = case.ghost_grid.trailing_ghost_voxel_count
        return cls(
            total=total,
            leading_ghost_end=leading_count,
            own_end=total - trailing_count,
        )

    def sum_inside(self, inside_count: np.ndarray) -> dict[str, int]:
        """Slice and sum a (1+total,) inside_particle_count array per region.

        Slot 0 is the sentinel (always 0) and is excluded from all sums.
        Returns {leading_ghost, own, trailing_ghost} integer sums.
        """
        leading = int(inside_count[1:self.leading_ghost_end + 1].sum())
        own = int(inside_count[self.leading_ghost_end + 1:self.own_end + 1].sum())
        trailing = int(inside_count[self.own_end + 1:self.total + 1].sum())
        return {
            "leading_ghost": leading,
            "own": own,
            "trailing_ghost": trailing,
        }


# ============================================================================
# DebugLogger
# ============================================================================

class DebugLogger:
    """Periodic CSV log + checkpoint-style snapshots.

    Per ``log_every`` frames: read global_status + inside_particle_count +
    incoming_particle_count from each sim, write one CSV row per sim.
    Per ``snapshot_every`` frames: read every sim buffer and dump to npz
    (or h5) under snapshots/sim_<label>/frame_<n>.<ext>.

    Args:
        output_dir: base directory for log + snapshots; created if missing.
        sims: dict[label, SphSimulatorV4] — label appears in CSV + dir names.
        log_every: frames between CSV writes (0 = every frame).
        snapshot_every: frames between buffer snapshots. Set to ``None`` or
            very large to disable snapshots.
        snapshot_format: "npz" (numpy native, no extra deps) or "h5" (HDF5,
            requires ``h5py``; supports selective loading).
        snapshot_compressed: True → gzip / lz77 inside the format. Cuts disk
            ~3-5× at cost of ~10ms CPU per buffer.
        snapshot_buffers: iterable of buffer names to snapshot; default None
            = ALL buffers in sim.buffers (full ckpt-style).
        meta_extra: additional dict merged into meta.json (e.g. case path,
            git commit). Caller-provided.
    """

    def __init__(
        self,
        output_dir: str | pathlib.Path,
        sims: dict[str, "SphSimulatorV4"],
        log_every: int = 50,
        snapshot_every: Optional[int] = 500,
        snapshot_format: Literal["npz", "h5"] = "npz",
        snapshot_compressed: bool = True,
        snapshot_buffers: Optional[Iterable[str]] = None,
        meta_extra: Optional[dict] = None,
    ) -> None:
        if not sims:
            raise ValueError("sims must be non-empty")
        if log_every <= 0:
            raise ValueError(f"log_every must be > 0, got {log_every}")
        if snapshot_format not in ("npz", "h5"):
            raise ValueError(f"snapshot_format must be 'npz' or 'h5', got {snapshot_format!r}")

        self.output_dir = pathlib.Path(output_dir)
        self.sims = dict(sims)
        self.log_every = log_every
        self.snapshot_every = snapshot_every
        self.snapshot_format = snapshot_format
        self.snapshot_compressed = snapshot_compressed
        self.snapshot_buffers = (tuple(snapshot_buffers)
                                 if snapshot_buffers is not None else None)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = self.output_dir / "snapshots"
        if snapshot_every is not None:
            for label in self.sims:
                (self.snapshot_dir / f"sim_{label}").mkdir(parents=True, exist_ok=True)

        # h5py is optional — only imported if user requests h5 output
        self._h5py = None
        if snapshot_format == "h5":
            try:
                import h5py
                self._h5py = h5py
            except ImportError as e:
                raise ImportError(
                    "snapshot_format='h5' requires h5py: pip install h5py") from e

        # Cache per-sim voxel ranges
        self._voxel_ranges = {
            label: _SimVoxelRanges.from_sim(sim)
            for label, sim in self.sims.items()
        }

        # Write meta.json once
        self._write_meta(meta_extra or {})

        # Open CSV
        csv_path = self.output_dir / "status.csv"
        self._csv_file = open(csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=self._csv_fieldnames())
        self._csv_writer.writeheader()
        self._csv_file.flush()

        self._closed = False
        print(f"[DebugLoggerV4] output={self.output_dir} "
              f"sims={list(self.sims)} "
              f"log_every={log_every} snapshot_every={snapshot_every} "
              f"snapshot_format={snapshot_format}")

    # ========================================================================
    # Public
    # ========================================================================

    def tick(self, frame_n: int) -> None:
        """Call once per frame; internal modulo-check decides whether to act."""
        if self._closed:
            raise RuntimeError("DebugLogger.tick() after close()")

        do_log = (frame_n % self.log_every == 0)
        do_snapshot = (self.snapshot_every is not None
                       and frame_n % self.snapshot_every == 0)

        if not (do_log or do_snapshot):
            return

        for label, sim in self.sims.items():
            if do_log:
                self._log_one_sim(label, sim, frame_n)
            if do_snapshot:
                self._snapshot_one_sim(label, sim, frame_n)

        if do_log:
            self._csv_file.flush()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._csv_file.close()
        except Exception:
            pass
        self._closed = True

    def __enter__(self) -> "DebugLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ========================================================================
    # CSV
    # ========================================================================

    @staticmethod
    def _csv_fieldnames() -> list[str]:
        return [
            "frame_n", "sim_label",
            # 16 global_status fields
            "alive_particle_count", "maximum_velocity",
            "overflow_inside_count", "overflow_incoming_count",
            "first_overflow_voxel_inside", "first_overflow_voxel_incoming",
            "correction_fallback_count", "overflow_ghost_count",
            "ghost_send_leading_count", "ghost_send_trailing_count",
            "ghost_recv_leading_count", "ghost_recv_trailing_count",
            "migration_install_count",
            "overflow_install_tail", "overflow_install_inside",
            "first_overflow_voxel_install",
            # derived from inside_particle_count
            "inside_sum_own", "inside_sum_leading_ghost", "inside_sum_trailing_ghost",
            # derived from incoming_particle_count
            "incoming_sum_own", "incoming_sum_leading_ghost", "incoming_sum_trailing_ghost",
        ]

    def _log_one_sim(self, label: str, sim: "SphSimulatorV4", frame_n: int) -> None:
        # Batch-read 3 buffers in one fence wait
        raws = sim.readback_buffers_batch([
            "global_status",
            "inside_particle_count",
            "incoming_particle_count",
        ])
        case = sim.case
        ranges = self._voxel_ranges[label]

        inside_arr = reinterpret_buffer(raws["inside_particle_count"],
                                        "inside_particle_count", case)
        incoming_arr = reinterpret_buffer(raws["incoming_particle_count"],
                                          "incoming_particle_count", case)
        inside_sums = ranges.sum_inside(inside_arr)
        incoming_sums = ranges.sum_inside(incoming_arr)

        # Parse global_status by re-using sim's struct unpacker for consistency
        status = sim.readback_global_status()
        # Note: readback_global_status() runs another readback — could optimize
        # by parsing raws["global_status"] directly. Left as-is for clarity
        # since 64 B readback is negligible.

        row = {
            "frame_n": frame_n,
            "sim_label": label,
            **status,
            "inside_sum_own": inside_sums["own"],
            "inside_sum_leading_ghost": inside_sums["leading_ghost"],
            "inside_sum_trailing_ghost": inside_sums["trailing_ghost"],
            "incoming_sum_own": incoming_sums["own"],
            "incoming_sum_leading_ghost": incoming_sums["leading_ghost"],
            "incoming_sum_trailing_ghost": incoming_sums["trailing_ghost"],
        }
        self._csv_writer.writerow(row)

    # ========================================================================
    # Snapshot
    # ========================================================================

    def _snapshot_one_sim(self, label: str, sim: "SphSimulatorV4",
                          frame_n: int) -> None:
        # Choose buffer set: user-provided list or all sim buffers
        if self.snapshot_buffers is None:
            names = list(sim.buffers.keys())
        else:
            names = [n for n in self.snapshot_buffers if n in sim.buffers]
            missing = set(self.snapshot_buffers) - set(sim.buffers.keys())
            if missing:
                print(f"[DebugLoggerV4] WARN: snapshot_buffers unknown to sim "
                      f"{label}: {missing}")

        t0 = time.perf_counter()
        raws = sim.readback_buffers_batch(names)
        t_readback = time.perf_counter() - t0

        # Reinterpret to typed arrays
        arrays = {name: reinterpret_buffer(raws[name], name, sim.case)
                  for name in names}

        # Write
        t1 = time.perf_counter()
        ext = self.snapshot_format
        path = self.snapshot_dir / f"sim_{label}" / f"frame_{frame_n:06d}.{ext}"
        if ext == "npz":
            self._write_npz(path, arrays, frame_n, label, sim)
        else:
            self._write_h5(path, arrays, frame_n, label, sim)
        t_write = time.perf_counter() - t1

        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"[DebugLoggerV4] snapshot sim_{label} frame={frame_n} "
              f"→ {path.name} ({size_mb:.1f} MB, "
              f"readback={t_readback*1000:.1f}ms, write={t_write*1000:.1f}ms)")

    def _write_npz(self, path: pathlib.Path, arrays: dict[str, np.ndarray],
                   frame_n: int, label: str, sim: "SphSimulatorV4") -> None:
        # Embed metadata as a json-encoded uint8 array so np.load gets it
        # back without a separate sidecar.
        meta = self._snapshot_meta(frame_n, label, sim)
        meta_bytes = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
        save_kwargs = dict(arrays, _meta=meta_bytes)
        if self.snapshot_compressed:
            np.savez_compressed(path, **save_kwargs)
        else:
            np.savez(path, **save_kwargs)

    def _write_h5(self, path: pathlib.Path, arrays: dict[str, np.ndarray],
                  frame_n: int, label: str, sim: "SphSimulatorV4") -> None:
        compression = "gzip" if self.snapshot_compressed else None
        with self._h5py.File(path, "w") as f:
            for name, arr in arrays.items():
                f.create_dataset(name, data=arr, compression=compression)
            for k, v in self._snapshot_meta(frame_n, label, sim).items():
                # h5 attrs require simple types; cast collections to lists
                f.attrs[k] = v if not isinstance(v, dict) else json.dumps(v)

    def _snapshot_meta(self, frame_n: int, label: str,
                       sim: "SphSimulatorV4") -> dict:
        case = sim.case
        return {
            "frame_n": int(frame_n),
            "sim_label": label,
            "wall_clock_unix": time.time(),
            "case_dimension": case.physics.dimension,
            "case_grid": [case.grid.grid_dimension_x,
                          case.grid.grid_dimension_y,
                          case.grid.grid_dimension_z],
            "case_origin": [case.grid.origin_x,
                            case.grid.origin_y,
                            case.grid.origin_z],
            "own_pool_size": int(case.capacities.own_pool_size),
            "leading_ghost_pool_size": int(case.capacities.leading_ghost_pool_size),
            "trailing_ghost_pool_size": int(case.capacities.trailing_ghost_pool_size),
            "leading_ghost_voxel_count": int(case.ghost_grid.leading_ghost_voxel_count),
            "trailing_ghost_voxel_count": int(case.ghost_grid.trailing_ghost_voxel_count),
            "smoothing_length": float(case.physics.smoothing_length),
            "timestep": float(case.physics.timestep),
        }

    # ========================================================================
    # meta.json (one-shot)
    # ========================================================================

    def _write_meta(self, extra: dict) -> None:
        meta = {
            "log_every": self.log_every,
            "snapshot_every": self.snapshot_every,
            "snapshot_format": self.snapshot_format,
            "snapshot_compressed": self.snapshot_compressed,
            "snapshot_buffers": list(self.snapshot_buffers) if self.snapshot_buffers else None,
            "sims": {
                label: self._snapshot_meta(0, label, sim)
                for label, sim in self.sims.items()
            },
            **extra,
        }
        path = self.output_dir / "meta.json"
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)
