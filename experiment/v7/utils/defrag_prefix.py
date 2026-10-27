"""defrag_prefix.py — the host-side producer of ``voxel_base_offset`` for defrag.comp's prefix branch (E9, 2026-10-27).

``defrag.comp`` has two destination-base strategies behind the specialization constant 46 ``USE_PREFIX_SUM_DEFRAG``
(common.glsl):

* ``atomic``  (the default until E9): ``base = atomicAdd(defrag_scratch_counter, count)``. Every voxel's particles
  stay contiguous, but the voxel BLOCKS land in the arrival order of the defrag workgroups. The first wave of the
  dispatch (all SMs filling at once, ~2,000 workgroups of 128 voxels, ~16M particles) completes in a scrambled order,
  the rest nearly in dispatch order; an 8M slab is therefore scrambled everywhere and its two sweep kernels pay
  +12 % per particle, a 64M slab only in its first quarter (E8 report 7.18, local 5090).
* ``prefix``  (the default since E9): ``base = voxel_base_offset[voxel_id]`` = the exclusive prefix sum of
  ``inside_particle_count`` over the own voxels in voxel-id order, so the pid order IS the voxel-id order (x slowest).
  The tree never had a producer of ``voxel_base_offset`` (no prefix_sum.comp); this module is it: a host scan run
  right before every defrag submit (bootstrap and periodic), while the queue is drained. It also writes the
  own-particle total into ``defrag_scratch_counter`` (which the prefix branch never increments) so that the defrag
  command's existing tail — copy the counter into ``global_status.alive_particle_count``, reset the install tail
  (``migration_install_count``) — keeps working unchanged. Ghost voxels contribute nothing (defrag.comp skips them,
  ``is_own_voxel``); the departed region is untouched (defrag never reads it).

Cost: one readback of ``inside_particle_count`` + two uploads, ~5 ms at 8M and ~27 ms at 64M per defrag, i.e. once at
bootstrap and once every ``defrag_cadence`` (1000) frames, outside every timed window. The SPIR-V is unchanged:
46 is a specialization constant.

Used by ``SphSimulatorV7`` (experiment/v7) and ``SphSimulatorSingle`` (experiment/v7_single): both call
``upload_defrag_prefix(simulator)``; the simulator must offer ``case``, ``buffers``, ``_readback_buffer(buffer)`` and
``_staging_upload(buffer, payload)``.
"""
from __future__ import annotations

import struct
import time

import numpy as np


def own_voxel_range(case) -> tuple[int, int, int]:
    """(first, last, total): the 1-based ids of the own voxels, helpers.glsl ``is_own_voxel``:
    ``LEADING_GHOST_VOXEL_COUNT < voxel_id <= extended_voxel_count() - TRAILING_GHOST_VOXEL_COUNT``."""
    total_voxel_count = int(case.grid.total_voxel_count())
    first_own_voxel = int(case.ghost_grid.leading_ghost_voxel_count) + 1
    last_own_voxel = total_voxel_count - int(case.ghost_grid.trailing_ghost_voxel_count)
    return first_own_voxel, last_own_voxel, total_voxel_count


def prefix_bases(inside_particle_count: np.ndarray, first_own_voxel: int, last_own_voxel: int) -> tuple[np.ndarray, int]:
    """The exclusive prefix sum of the own voxels' counts in voxel-id order (uint32, one entry per voxel slot, slot 0
    and the ghost voxels 0) and the own-particle total. ``voxel_base_offset[v] = sum of count[u] for own u < v``;
    the first own voxel's base is 0 (defrag.comp adds own_first_pid())."""
    counts = np.asarray(inside_particle_count, dtype=np.uint64)
    own_counts = np.zeros(counts.shape[0], dtype=np.uint64)
    own_counts[first_own_voxel:last_own_voxel + 1] = counts[first_own_voxel:last_own_voxel + 1]
    bases = np.zeros_like(own_counts)
    bases[1:] = np.cumsum(own_counts)[:-1]
    total = int(own_counts.sum())
    if total >= 2 ** 32:
        raise RuntimeError(f"defrag prefix: {total:,} own particles do not fit the uint32 bases")
    return bases.astype(np.uint32), total


def upload_defrag_prefix(simulator, verbose: bool = True) -> dict:
    """Before one defrag submit: read back inside_particle_count, scan, upload voxel_base_offset and the total.
    Returns the record (total, non-empty own voxels, milliseconds)."""
    started = time.perf_counter()
    counts_buffer = simulator.buffers["inside_particle_count"]
    counts = np.frombuffer(simulator._readback_buffer(counts_buffer), dtype=np.uint32)
    first_own_voxel, last_own_voxel, total_voxel_count = own_voxel_range(simulator.case)
    if counts.shape[0] != total_voxel_count + 1:
        raise RuntimeError(f"defrag prefix: inside_particle_count has {counts.shape[0]} slots, "
                           f"expected {total_voxel_count + 1}")
    bases, total = prefix_bases(counts, first_own_voxel, last_own_voxel)
    own_pool_size = int(simulator.case.capacities.own_pool_size)
    if total > own_pool_size:
        raise RuntimeError(f"defrag prefix: {total:,} own particles in the voxel lists exceed the own pool {own_pool_size:,}")
    base_buffer = simulator.buffers["voxel_base_offset"]
    payload = bases.tobytes()
    if len(payload) != base_buffer.size:
        raise RuntimeError(f"defrag prefix: {len(payload)} bytes of bases for a {base_buffer.size}-byte voxel_base_offset")
    simulator._staging_upload(base_buffer, payload)
    simulator._staging_upload(simulator.buffers["defrag_scratch_counter"], struct.pack("<I", total))
    record = {"total": total, "own_voxels": last_own_voxel - first_own_voxel + 1,
              "nonempty_own_voxels": int(np.count_nonzero(counts[first_own_voxel:last_own_voxel + 1])),
              "milliseconds": (time.perf_counter() - started) * 1e3}
    if verbose:
        print(f"[defrag_prefix] own voxels {first_own_voxel}..{last_own_voxel} ({record['own_voxels']:,}, "
              f"{record['nonempty_own_voxels']:,} non-empty) -> voxel_base_offset in voxel-id order, "
              f"{total:,} own particles, {record['milliseconds']:.1f} ms", flush=True)
    return record


def audit_pid_order(simulator, own_total: int | None = None) -> dict:
    """Diagnostic (V7_DEFRAG_AUDIT=1): after a defrag, read back position_voxel_id and check that the own pids are in
    voxel-id order: ``descents`` = block transitions to a lower voxel id (0 under the prefix order; tens of thousands
    under the atomic order), ``blocks`` = contiguous voxel blocks (= the non-empty own voxels either way),
    ``zero_voxel_id`` = own pids without a voxel (must be 0). Prints one line and returns the record."""
    started = time.perf_counter()
    own_first_pid = int(simulator.own_first_pid())
    if own_total is None:
        records = getattr(simulator, "_defrag_prefix_records", None)
        own_total = int(records[-1]["total"]) if records else int(simulator.readback_global_status()["alive_particle_count"])
    raw = simulator._readback_buffer(simulator.buffers["position_voxel_id"])
    voxel_ids = np.rint(np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)[own_first_pid:own_first_pid + own_total, 3]).astype(np.int64)
    differences = np.diff(voxel_ids)
    record = {"own_total": own_total, "descents": int(np.count_nonzero(differences < 0)),
              "blocks": 1 + int(np.count_nonzero(differences != 0)) if voxel_ids.size else 0,
              "nonempty_voxels": int(np.unique(voxel_ids[voxel_ids > 0]).shape[0]),
              "zero_voxel_id": int(np.count_nonzero(voxel_ids <= 0)), "milliseconds": (time.perf_counter() - started) * 1e3}
    record["exact_voxel_order"] = record["descents"] == 0 and record["zero_voxel_id"] == 0
    print(f"[defrag_prefix] audit: {own_total:,} own pids, {record['blocks']:,} voxel blocks for {record['nonempty_voxels']:,} "
          f"non-empty voxels, {record['descents']:,} block descents, {record['zero_voxel_id']:,} pids without a voxel -> "
          f"{'EXACT voxel order' if record['exact_voxel_order'] else 'NOT voxel order'} ({record['milliseconds']:.0f} ms)", flush=True)
    return record
