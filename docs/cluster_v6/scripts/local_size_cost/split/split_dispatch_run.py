"""Scratch monkeypatch (user 2026-10-27, approved): run the e8 chain harness (run_chain_v6.py --solver v7 ...) with the two phase-B
sweep dispatches split into m pieces, to measure the per-dispatch fill + drain cost by timing: T(m) - T(1) = (m - 1) x tail.

Nothing in experiment/v7 (code or SPIR-V) is modified. vkCmdDispatchBase is NOT used: on this RTX 5090 / driver 576.88 a
non-zero base group runs the wrong workgroups in this solver (E39's observation, reproduced here: the four-piece run left most
particles unprocessed). The pieces are instead dispatches of SEGMENT pipelines whose spec constant 115 = the piece's first
thread, the mechanism the tree already uses for force.comp (FORCE_SEGMENT, spv/force_segment.comp.spv); the fused
correction + density kernel gets the same two-line variant compiled from a scratch copy of correction_density.comp
(split/shaders/correction_density_segment.comp[.spv], the tree's source and SPIR-V untouched; diff = the constant + the
thread-id add). In this process only, the script patches
  * SphSimulatorV7._load_shader_modules   -> also loads force_segment (tree) and correction_density_segment (scratch);
  * SphSimulatorV7._build_compute_pipelines -> adds one SEGMENT pipeline per piece (same spec constants as the production
    pipeline + 115 = first thread); the piece boundaries divide the LIVE particle range [0, SPLIT_N) evenly in particle order,
    the last piece also covers the pool slack (the production dispatch covers the whole pool);
  * SphSimulatorV7._bind_pipeline_and_sets + simulator_v7.vkCmdDispatch -> for a split kernel records the pieces back to back
    with the simulator's own compute->compute barrier between them;
  * SphSimulatorV7.destroy                -> optionally hashes / aggregates the state first (SPLIT_HASH_OUT).

Environment: SPLIT_CD = m for correction_density_interior, SPLIT_FO = m for force_deep_interior_scratch
  (0 = production path untouched; 1 = one SEGMENT dispatch with first thread 0 = the variant's own cost; m > 1 = m pieces),
  SPLIT_N = live own particles of the single slab (the case total at K = 1), SPLIT_HASH_OUT = file for the state record.
Usage: python split_dispatch_run.py <run_chain_v6.py arguments...>   (cwd = the e8 worktree)
"""
import hashlib
import json
import os
import pathlib
import runpy
import sys
import time

REPO = pathlib.Path("C:/Users/cchen/PycharmProjects/vulkan-demo-e8")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import vulkan as vk  # noqa: E402
import experiment.v7.utils.simulator_v7 as sim  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
KEYS = ("correction_density_interior", "force_deep_interior_scratch")
SPLIT = {"correction_density_interior": int(os.environ.get("SPLIT_CD", "0")),
         "force_deep_interior_scratch": int(os.environ.get("SPLIT_FO", "0"))}
LIVE_N = int(os.environ.get("SPLIT_N", "0"))
HASH_OUT = os.environ.get("SPLIT_HASH_OUT")
SEGMENT_CONSTANT = 115
MODULE_OF = {"correction_density_interior": "correction_density_segment", "force_deep_interior_scratch": "force_segment"}
_printed = set()


def piece_key(key, index):
    return f"split__{key}__{index}"


_orig_load = sim.SphSimulatorV7._load_shader_modules


def load_shader_modules(self):
    modules = _orig_load(self)
    tree_spv = REPO / "experiment/v7/shaders/spv"
    sources = {"force_segment": tree_spv / "force_segment.comp.spv",
               "correction_density_segment": HERE / "shaders/correction_density_segment.comp.spv"}
    for name, path in sources.items():
        if name in modules:
            continue
        code = path.read_bytes()
        info = vk.VkShaderModuleCreateInfo(codeSize=len(code), pCode=code)
        modules[name] = vk.vkCreateShaderModule(self.ctx.device, info, None)
        print(f"[split] shader module {name} <- {path} ({len(code):,} B)", flush=True)
    return modules


sim.SphSimulatorV7._load_shader_modules = load_shader_modules

_orig_build = sim.SphSimulatorV7._build_compute_pipelines


def build_compute_pipelines(self):
    pipelines = _orig_build(self)
    per_p = self._per_own_particle_dispatch_count()
    wg = self.case.capacities.workgroup_size
    live_groups = min(per_p, (LIVE_N + wg - 1) // wg) if LIVE_N > 0 else per_p
    self.__dict__["_split_pieces"] = {}
    for key, m in SPLIT.items():
        if m <= 0:
            continue
        if key == "correction_density_interior":
            if not self._fused_correction_density_active():
                raise RuntimeError("split: the slab does not fuse correction + density")
            if key in self._fused_deep_wall_skip_pipeline_keys():
                raise RuntimeError("split: the deep-wall-skip variant of the fused kernel is active (no scratch variant for it)")
            entries = self._global_entries() + self._correction_mode_entries(1, 0)
        else:
            entries = self._global_entries() + self._force_mode_entries(1, density_source=1)
        bounds = [live_groups * k // m for k in range(m)] + [per_p]
        pieces = [(bounds[k], bounds[k + 1] - bounds[k]) for k in range(m) if bounds[k + 1] > bounds[k]]
        self._split_pieces[key] = pieces
        for index, (first_group, count) in enumerate(pieces):
            pipelines[piece_key(key, index)] = self._create_pipeline(
                shader=self.shader_modules[MODULE_OF[key]], entries=entries + [(SEGMENT_CONSTANT, 'I', first_group * wg)])
        print(f"[split] {key}: {per_p:,} workgroups ({live_groups:,} live, {wg} lanes) -> {len(pieces)} SEGMENT pieces "
              f"{pieces} (spec {SEGMENT_CONSTANT} = first thread), {len(pieces) - 1} barriers", flush=True)
    print(f"[split] pipelines: SPLIT_CD={SPLIT['correction_density_interior']} SPLIT_FO={SPLIT['force_deep_interior_scratch']} "
          f"SPLIT_N={LIVE_N:,}", flush=True)
    return pipelines


sim.SphSimulatorV7._build_compute_pipelines = build_compute_pipelines

_orig_bind = sim.SphSimulatorV7._bind_pipeline_and_sets
_pending = {"sim": None, "key": None, "cmd": None}


def bind_pipeline_and_sets(self, cmd, pipeline_key):
    if SPLIT.get(pipeline_key):
        _pending.update(sim=self, key=pipeline_key, cmd=cmd)      # the pieces bind their own pipelines in dispatch()
        return None
    _pending.update(sim=None, key=None, cmd=None)
    return _orig_bind(self, cmd, pipeline_key)


sim.SphSimulatorV7._bind_pipeline_and_sets = bind_pipeline_and_sets

_orig_dispatch = sim.vkCmdDispatch


def dispatch(cmd, x, y, z):
    self, key = _pending["sim"], _pending["key"]
    if self is None or _pending["cmd"] is not cmd or (y, z) != (1, 1):
        return _orig_dispatch(cmd, x, y, z)
    _pending.update(sim=None, key=None, cmd=None)
    pieces = self._split_pieces[key]
    if sum(count for _, count in pieces) != x:
        raise RuntimeError(f"split: {key} dispatch of {x:,} workgroups != pieces {pieces}")
    for index, (first_group, count) in enumerate(pieces):
        _orig_bind(self, cmd, piece_key(key, index))
        _orig_dispatch(cmd, count, 1, 1)
        if index < len(pieces) - 1:
            self._record_compute_barrier(cmd)
    tag = (key, x)
    if tag not in _printed:
        _printed.add(tag)
        print(f"[split] recorded {key}: {len(pieces)} pieces of {x:,} workgroups", flush=True)


sim.vkCmdDispatch = dispatch

_orig_destroy = sim.SphSimulatorV7.destroy


def destroy(self):
    if HASH_OUT:
        try:
            names = ["position_voxel_id", "velocity_mass", "density_pressure"]
            data = self.readback_buffers_batch(names)
            digests = {name: hashlib.sha1(data[name]).hexdigest() for name in names}
            digests["all"] = hashlib.sha1(b"".join(data[name] for name in names)).hexdigest()
            # physical aggregates over the live particles (plain runs are not bit-reproducible: atomics order the
            # voxel lists, so the hash differs run to run; a skipped / duplicated workgroup range shows up here)
            import numpy as np
            live = LIVE_N if LIVE_N > 0 else None
            pos = np.frombuffer(data["position_voxel_id"], dtype=np.float32).reshape(-1, 4)[:live]
            vel = np.frombuffer(data["velocity_mass"], dtype=np.float32).reshape(-1, 4)[:live]
            rho = np.frombuffer(data["density_pressure"], dtype=np.float32).reshape(-1, 2)[:live]
            speed = np.sqrt((vel[:, :3].astype(np.float64) ** 2).sum(axis=1))
            q = [0, 1, 10, 50, 90, 99, 100]
            stats = {"rho_mean": float(rho[:, 0].astype(np.float64).mean()), "rho_std": float(rho[:, 0].astype(np.float64).std()),
                     "rho_pct": [float(v) for v in np.percentile(rho[:, 0], q)],
                     "p_mean": float(rho[:, 1].astype(np.float64).mean()),
                     "speed_mean": float(speed.mean()), "speed_pct": [float(v) for v in np.percentile(speed, q)],
                     "pos_mean": [float(v) for v in pos[:, :3].astype(np.float64).mean(axis=0)],
                     "pos_abs_sum": float(np.abs(pos[:, :3].astype(np.float64)).sum()),
                     "n": int(pos.shape[0])}
            record = {"split": SPLIT, "live_n": LIVE_N, "sizes": {name: len(data[name]) for name in names}, "sha1": digests,
                      "stats": stats, "time": time.time()}
            with open(HASH_OUT, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"[split] state hash {digests['all'][:16]} -> {HASH_OUT}", flush=True)
        except Exception as error:  # noqa: BLE001 - diagnostic only
            print(f"[split] state hash failed: {error!r}", flush=True)
    return _orig_destroy(self)


sim.SphSimulatorV7.destroy = destroy

if __name__ == "__main__":
    harness = REPO / "docs/cluster_v6/scripts/run_chain_v6.py"
    sys.argv = [str(harness)] + sys.argv[1:]
    runpy.run_path(str(harness), run_name="__main__")
