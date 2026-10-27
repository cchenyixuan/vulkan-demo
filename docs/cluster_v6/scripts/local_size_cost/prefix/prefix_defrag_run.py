"""Scratch monkeypatch (user 2026-10-27): run the e8 chain harness (run_chain_v6.py --solver v7 ...) with
numerics.use_prefix_sum_defrag = true, i.e. defrag.comp's USE_PREFIX_SUM_DEFRAG branch (spec constant 46), and measure
the per-particle cost + the m = 8 piece profile against the default (atomicAdd-ordered) defrag.

What the tree does with the switch today: defrag.comp reads voxel_base_offset[voxel_id] as the destination base, but
NOTHING in experiment/v7 writes that buffer (no prefix_sum.comp - the shader list has none - and no host scan; the
simulator only allocates it, set 1 binding 4; docs/paper_material_map.md: "prefix-sum defrag not implemented"). The
defrag cmd also refreshes global_status.alive_particle_count from defrag_scratch_counter, which the prefix branch never
increments (DCE'd) and the cmd fills with 0. So the raw switch collapses every voxel onto base 0 and sets alive = 0.

PREFIX_MODE (environment):
  off  = default defrag (control; the switch stays false)
  raw  = the switch alone, exactly what the tree does (expected broken; the audit below shows how)
  host = the switch + a host-side exclusive scan of inside_particle_count[] over the own voxels in voxel-id order,
         uploaded to voxel_base_offset before EVERY defrag submit (bootstrap + periodic), and alive_particle_count
         restored to the scanned total after the submit. experiment/v7 code and SPIR-V untouched: the tree's own
         defrag.comp prefix branch does the scatter; only the missing producer runs on the host (one readback + one
         upload per defrag; the chain runs one bootstrap defrag + one every defrag_cadence = 1000 frames).
LAYOUT_AUDIT=1: after every defrag, read back position_voxel_id (+ material) of the own particles and report whether
  the pid order is the voxel-id order (descents, voxel blocks), plus the x-column span and material composition of
  the 8 equal pid pieces (the m = 8 split pieces of 7.17) -> appended to PREFIX_AUDIT_OUT (jsonl).
The split pieces + per-piece ticks of split/split_dispatch_run_ticks.py are imported unchanged (SPLIT_CD / SPLIT_FO /
SPLIT_N / SPLIT_HASH_OUT as there; 0 / 0 = plain kernels). Usage: python prefix_defrag_run.py <run_chain_v6.py args...>
"""
import json
import os
import pathlib
import runpy
import struct
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SPLIT_DIR = HERE.parent / "split"
sys.path.insert(0, str(SPLIT_DIR))
import split_dispatch_run_ticks as split  # noqa: E402  (applies the split / ticks patches at import; no-ops at m = 0)

sim = split.sim
REPO = split.REPO
MODE = os.environ.get("PREFIX_MODE", "off")
AUDIT = os.environ.get("LAYOUT_AUDIT", "0") == "1"
AUDIT_OUT = os.environ.get("PREFIX_AUDIT_OUT")
PIECES = 8
if MODE not in ("off", "raw", "host"):
    raise SystemExit(f"PREFIX_MODE={MODE!r} must be off | raw | host")
_state = {"defrags": 0}

_orig_init = sim.SphSimulatorV7.__init__


def init(self, ctx, case, *args, **kwargs):
    if MODE in ("raw", "host"):
        case.numerics.use_prefix_sum_defrag = True
    _orig_init(self, ctx, case, *args, **kwargs)
    entry = [e for e in self._global_entries() if e[0] == 46]
    print(f"[prefix] mode={MODE} numerics.use_prefix_sum_defrag={case.numerics.use_prefix_sum_defrag} "
          f"spec constant 46 entry={entry} defrag_cadence={case.numerics.defrag_cadence}", flush=True)


sim.SphSimulatorV7.__init__ = init

_orig_defrag = sim.SphSimulatorV7.submit_defrag_and_wait


def own_voxel_range(self):
    """1-based voxel ids of the own voxels: (lead, V - trail] (helpers.glsl is_own_voxel)."""
    total = self.case.grid.total_voxel_count()
    lead = self.case.ghost_grid.leading_ghost_voxel_count
    trail = self.case.ghost_grid.trailing_ghost_voxel_count
    return lead + 1, total - trail, total


def submit_defrag_and_wait(self):
    index = _state["defrags"]
    _state["defrags"] += 1
    t0 = time.perf_counter()
    total_own = None
    if MODE == "host":
        counts = np.frombuffer(self.readback_buffer_by_name("inside_particle_count"), dtype=np.uint32)
        first, last, total_voxels = own_voxel_range(self)
        assert counts.shape[0] == total_voxels + 1, (counts.shape, total_voxels)
        own = np.zeros(counts.shape[0], dtype=np.uint64)
        own[first:last + 1] = counts[first:last + 1]               # non-own voxels (ghosts, slot 0) contribute 0
        base = np.zeros_like(own)
        base[1:] = np.cumsum(own)[:-1]                              # exclusive scan in voxel-id order
        total_own = int(own.sum())
        if base[-1] + own[-1] != total_own or base.max() >= 2 ** 32:
            raise RuntimeError("prefix: scan overflow / inconsistency")
        payload = base.astype(np.uint32).tobytes()
        buf = self.buffers["voxel_base_offset"]
        assert len(payload) == buf.size, (len(payload), buf.size)
        self._staging_upload(buf, payload)
        t_scan = time.perf_counter() - t0
        nonempty = int(np.count_nonzero(own))
        print(f"[prefix] defrag #{index}: host scan of inside_particle_count over own voxels {first}..{last} "
              f"({last - first + 1:,} voxels, {nonempty:,} non-empty) -> voxel_base_offset uploaded; total {total_own:,} "
              f"particles; readback + scan + upload {t_scan * 1e3:.1f} ms", flush=True)
    _orig_defrag(self)
    t_defrag = time.perf_counter() - t0
    status_before = self.readback_global_status()
    if MODE == "host":
        # the defrag cmd copied defrag_scratch_counter (never incremented by the prefix branch, filled 0) into
        # alive_particle_count: restore the scanned total
        raw = self._readback_buffer(self.buffers["global_status"])
        self._staging_upload(self.buffers["global_status"], struct.pack("<I", total_own) + raw[4:])
    status = self.readback_global_status()
    restored = f" -> restored {status['alive_particle_count']:,}" if MODE == "host" else ""
    print(f"[prefix] defrag #{index} done in {t_defrag * 1e3:.1f} ms: alive after the cmd "
          f"{status_before['alive_particle_count']:,}{restored}; overflow_inside {status['overflow_inside_count']} "
          f"overflow_incoming {status['overflow_incoming_count']}", flush=True)
    if AUDIT:
        try:
            layout_audit(self, index, status, total_own)
        except Exception as error:  # noqa: BLE001 - diagnostic only (a 64M readback is ~1.5 GB of staging)
            print(f"[prefix] layout audit #{index} failed: {error!r}", flush=True)


sim.SphSimulatorV7.submit_defrag_and_wait = submit_defrag_and_wait


def layout_audit(self, index, status, total_own):
    """pid order vs voxel-id order of the own particles after this defrag + the 8 equal pid pieces' columns."""
    t0 = time.perf_counter()
    first_pid = self.own_first_pid()
    live = int(os.environ.get("SPLIT_N", "0")) or total_own or status["alive_particle_count"]
    names = ["position_voxel_id", "material"]
    data = self.readback_buffers_batch(names)
    pos = np.frombuffer(data["position_voxel_id"], dtype=np.float32).reshape(-1, 4)
    material = np.frombuffer(data["material"], dtype=np.uint32)
    vid = np.rint(pos[first_pid:first_pid + live, 3]).astype(np.int64)
    mat = material[first_pid:first_pid + live]
    grid = self.case.grid
    yz = grid.grid_dimension_y * grid.grid_dimension_z
    x_col = (vid - 1) // yz                                         # own_voxel_id_of: x slowest
    zero_vid = int(np.count_nonzero(vid <= 0))
    d = np.diff(vid)
    descents = int(np.count_nonzero(d < 0))
    blocks = 1 + int(np.count_nonzero(d != 0))
    nonempty_voxels = int(np.unique(vid[vid > 0]).shape[0])
    running_max = np.maximum.accumulate(vid)
    behind = int(np.count_nonzero(vid < running_max))              # pids whose voxel is behind an earlier pid's voxel
    tail_end = min(pos.shape[0], first_pid + live + 1024)
    tail = np.rint(pos[first_pid + live:tail_end, 3]).astype(np.int64)
    pieces = []
    for k in range(PIECES):
        lo, hi = live * k // PIECES, live * (k + 1) // PIECES
        xs = x_col[lo:hi]
        counts = np.bincount(mat[lo:hi], minlength=3).tolist()
        valid = xs[xs >= 0]
        cols, col_counts = np.unique(valid, return_counts=True)
        pieces.append({"piece": k, "range": [lo, hi], "x_first": int(xs[0]) if xs.size else None,
                       "x_last": int(xs[-1]) if xs.size else None,
                       "x_min": int(cols.min()) if cols.size else None, "x_max": int(cols.max()) if cols.size else None,
                       "material_counts": counts, "columns": {int(c): int(n) for c, n in zip(cols, col_counts)}})
    totals = np.bincount(mat, minlength=3).tolist()
    record = {"mode": MODE, "defrag_index": index, "live": live, "first_pid": first_pid, "zero_voxel_id": zero_vid,
              "descents": descents, "pids_behind_running_max": behind, "voxel_blocks": blocks,
              "nonempty_voxels": nonempty_voxels, "sorted_exactly": descents == 0 and zero_vid == 0,
              "tail_nonzero": int(np.count_nonzero(tail > 0)),
              "grid": [grid.grid_dimension_x, grid.grid_dimension_y, grid.grid_dimension_z], "material_totals": totals,
              "alive_status": status["alive_particle_count"], "pieces": pieces, "audit_s": time.perf_counter() - t0,
              "time": time.time()}
    verdict = "EXACT voxel order" if record["sorted_exactly"] else "NOT voxel order"
    print(f"[prefix] layout audit #{index}: live {live:,} pids from {first_pid}; voxel id 0 on {zero_vid:,} pids; "
          f"descents {descents:,}; pids behind the running max {behind:,}; voxel blocks {blocks:,} vs non-empty voxels "
          f"{nonempty_voxels:,}; {verdict}; material totals {totals}; pieces x-span "
          f"{[(p['x_first'], p['x_last']) for p in pieces]}; {record['audit_s']:.1f} s", flush=True)
    if AUDIT_OUT:
        with open(AUDIT_OUT, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    harness = REPO / "docs/cluster_v6/scripts/run_chain_v6.py"
    sys.argv = [str(harness)] + sys.argv[1:]
    runpy.run_path(str(harness), run_name="__main__")
