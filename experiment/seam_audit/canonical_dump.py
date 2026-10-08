"""
canonical_dump.py — bit-identity harness for the v6 release (E37): run a case with K slabs from its initial state
and dump the full per-particle state after the last step, merged over the slabs and sorted by a global particle id;
--compare checks two dumps field by field, bit for bit.

Two runs of the same build are bitwise equal only when the GPU's atomic orders cannot reach the arithmetic. The
voxel lists are built by atomic appends, so with --canonical-lists every list is sorted by pid after the initial
voxelization, the replicas of the bootstrap ghost round are packed from the sorted lists, and the bootstrap defrag
is skipped (the E36 k1_dump harness of the v7-wall-bc branch, generalised to K slabs); the run must end before the
first defrag. Later list appends (a particle changing voxel, a migrant installed, a departed particle re-registered)
are atomic too: two arrivals in one voxel in one step can still swap. Run a build twice and compare (the self
check) before reading a cross-build comparison.

Global particle id (no solver change, as experiment/seam_audit/dump_state.py): the initial upload also fills
extension_fields (z = id // 2**20, w = id % 2**20) with the particle's row in the global case; no kernel reads it
and defrag moves it with the particle. Across a cut, the release set's lean packets (V6_LEAN_TRANSPORT=1) carry it
only with V6_TRANSPORT_EXTENSION=1 (spec constant 88, one more copy; dump_state.py's audit workers set it): pass
--transport-extension to both runs of a comparison when particles cross a cut. Without it a migrant arrives with
id 0; the harness stops when the slabs show a crossing (migration_install_count, cumulative until a defrag, or the
pool-health peaks, never reset).

Switches: cavity_runner.RELEASE_ENVIRONMENT (the validation release set, equal to the code defaults since E6b),
set before the solver import; --env V6_X=VALUE adds or overrides one (e.g. V6_CASCADE_FORCE=0). --repo runs another
checkout's experiment/v6 (e.g. a v6-rc1 worktree) with this harness; nothing from experiment/ is imported before
that checkout is first on sys.path. The dump's meta records what ran: the imported simulator module, sha256 (taken
before the run, of the working-tree bytes: CRLF in a Windows checkout with core.autocrlf) of
experiment/v6/utils/*.py, of every SPIR-V file of the shader directory, of cavity_runner.py and of this file; the
case file; the loaded initial state and the loaded case (materials, physics, numerics, grid).

Invariants (exit 3, and a PROBLEM for --compare): alive == expected, no overflow_* counter, and no transport error
(GPU stamp_error_count, far_migration_count, the host workers' stale-readback count). A non-canonical run whose
last step is followed by a defrag (steps % cadence == 0) does not dump density_pressure_scratch and
wall_dummy_velocity: defrag does not move them (DEFRAG_SET0_BINDINGS).

--monitor / --timestamps (K = 1, without --canonical-lists): the E36 section 8.3 timing method - the normal
bootstrap, a BenchTimer on the sim, and at every defrag boundary (drained) the E36 k1_dump monitor (status / state
readback and its numpy reductions, unchanged so that fps is measured as in E36) plus the GPU timestamps of the
last frame before it, written to <out>.monitor.jsonl.

--compare: fails on a field present in one dump only (unless --ignore names it; a field present in both is always
compared), on a broken invariant in either dump (also read from the v7 k1_dump format's status), on different
steps / canonical_lists, and on a switch both environments set differently (V6_ / V7_ prefixes stripped); it prints
both sides' provenance.

    .venv/Scripts/python.exe experiment/seam_audit/canonical_dump.py --case CASE.yaml --device-map 1 --steps 200 \\
        --canonical-lists --out logs/e37/equivalence/k1_a.npz [--repo ../vulkan-demo-e34] [--transport-extension]
    (by path: with -m, experiment/ would already be imported from this checkout, and --repo is refused)
    .venv/Scripts/python.exe -m experiment.seam_audit.canonical_dump --compare A.npz B.npz [--ignore FIELD ...]

--solver v7 (E39) runs experiment/v7 instead (modules *_v7, classes *V7, switches V7_*): the release set and --env
use the V7_ prefix, and the dump's meta records the solver. Dumps of the two solvers compare directly (--compare
strips the V6_ / V7_ prefixes).

E39 B4 (v7 V7_DEEP_WALL_SKIP): a skipped deep wall keeps its previous correction_inverse and
density_gradient_kernel_sum (nothing reads them). The harness sets simulator_v7._DEEP_WALL_RECORD_DECISIONS (unless
--no-deep-wall-record) so every slab that skips records its GPU decisions, and the dump carries per particle
deep_wall_skip_correction / deep_wall_skip_density (1 = skipped in the last step; not compared as fields; dropped
after a final defrag, which does not move the record) plus meta['deep_wall'] (per-slab resolution, the material
kinds, the support radius) and a CPU check independent of the GPU marker: every row skipped by correction is a
BOUNDARY particle with no particle of another kind within the support radius (float64, r <= h, every slab's
particles), and density skipped the same rows (simple walls; with adami, density never reaches a wall). A failed
check is an invariant problem (exit 3). --compare masks the rows skipped by correction (in either dump) in exactly
correction_inverse and density_gradient_kernel_sum, prints the masked count per field, repeats the CPU check and
prints correction_fallback_count / overflow_deep_wall_skip_count of both dumps; every other row and field must be
bit-identical. --no-mask compares those two fields in full (two dumps that skipped the same rows). Dumps without
skip data compare exactly as before.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

GLOBAL_ID_LOW_BASE = 2 ** 20
# set-0 buffers dumped per particle: (name, components, dtype); the same names as the E36 k1_dump
DUMPED_FIELDS = (("position_voxel_id", 4, np.float32), ("velocity_mass", 4, np.float32),
                 ("density_pressure", 2, np.float32), ("density_pressure_scratch", 2, np.float32),
                 ("acceleration", 4, np.float32), ("shift", 4, np.float32), ("material", 1, np.uint32),
                 ("correction_inverse", 8, np.float32), ("density_gradient_kernel_sum", 4, np.float32),
                 ("extension_fields", 4, np.float32))
ADAMI_FIELDS = (("wall_dummy_velocity", 4, np.float32),)
NOT_MOVED_BY_DEFRAG = ("density_pressure_scratch", "wall_dummy_velocity")     # set 0 bindings 2 and 10
TRANSPORT_ERROR_COUNTERS = ("stamp_error_count", "far_migration_count")      # global status, cumulative
GLOBAL_IDS: dict = {}          # id(slab case) -> global ids of the slab's initial particles
# E39 B4: per-particle skip decisions of the last step (not fields: excluded from the comparison) and the two fields a
# skipped deep wall keeps from an earlier step
DEEP_WALL_SKIP_ARRAYS = ("deep_wall_skip_correction", "deep_wall_skip_density")
DEEP_WALL_MASKED_FIELDS = ("correction_inverse", "density_gradient_kernel_sum")
DEEP_WALL_COUNTERS = ("correction_fallback_count", "overflow_deep_wall_skip_count")
BOUNDARY_KIND = 1               # common.glsl MATERIAL_BOUNDARY


def sha256_files(paths, root: pathlib.Path) -> str:
    """One digest over the files' paths (relative to root) and contents, in path order."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def canonical(value):
    """JSON-able, deterministic image of a loaded case object (arrays by digest, floats exactly)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: canonical(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, np.ndarray):
        return [str(value.dtype), list(value.shape), hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()]
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, float):
        return value.hex()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    text = repr(value)
    return type(value).__name__ if " at 0x" in text else text


def digest_of(value) -> str:
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True).encode()).hexdigest()


def _status_records(meta: dict) -> list:
    status = meta.get("status")
    return status if isinstance(status, list) else [status] if isinstance(status, dict) else []


def _invariant_problems(meta: dict) -> list:
    """Broken invariants of one dump; the v7 k1_dump format keeps its counters only in meta['status']."""
    problems = []
    if meta.get("alive") is None or meta.get("alive") != meta.get("expected"):
        problems.append(f"alive {meta.get('alive')} != expected {meta.get('expected')}")
    overflow = meta.get("overflow")
    if overflow is None:
        overflow = {f"slab{index}.{key}": value for index, record in enumerate(_status_records(meta))
                    for key, value in record.items() if key.startswith("overflow_") and value}
    if overflow:
        problems.append(f"overflow {overflow}")
    transport = meta.get("transport_errors")
    if transport is None:
        transport = {key: [int(record.get(key, 0)) for record in _status_records(meta)]
                     for key in TRANSPORT_ERROR_COUNTERS}
    if any(value for values in transport.values() for value in values):
        problems.append(f"transport errors {transport}")
    return problems


def _environment(meta: dict) -> dict:
    return {(key[3:] if key.startswith(("V6_", "V7_")) else key): value
            for key, value in (meta.get("environment") or {}).items()}


def deep_wall_cpu_check(state: dict, material_kinds: list, support_radius: float) -> dict:
    """E39 B4, independent of the GPU marker: the rows skipped by correction must be BOUNDARY particles with no
    particle of another kind within the support radius (float64 distances, r <= h, over every dumped particle of
    every slab); density must have skipped the same rows (or none, with adami walls)."""
    from scipy.spatial import cKDTree
    correction = state["deep_wall_skip_correction"].astype(bool)
    density = state["deep_wall_skip_density"].astype(bool)
    kinds = np.asarray(material_kinds)[state["material"].astype(np.int64)]
    positions = state["position_voxel_id"][:, :3].astype(np.float64)
    skipped = np.flatnonzero(correction)
    others = positions[kinds != BOUNDARY_KIND]
    near = 0
    if skipped.size and others.shape[0]:
        counts = cKDTree(others).query_ball_point(positions[skipped], r=float(support_radius), return_length=True)
        near = int((np.asarray(counts) > 0).sum())
    return {"skipped_correction": int(skipped.size), "skipped_density": int(density.sum()),
            "skipped_not_boundary": int((kinds[skipped] != BOUNDARY_KIND).sum()),
            "skipped_with_other_kind_within_h": near,
            "density_rows_equal_correction_rows": bool(np.array_equal(correction, density))}


def _deep_wall_cpu_problems(check: dict, wall_boundary: str) -> list:
    problems = []
    if check["skipped_not_boundary"] or check["skipped_with_other_kind_within_h"]:
        problems.append(f"deep-wall skip of a non-wall or of a wall with a particle of another kind within h: {check}")
    density_ok = (check["skipped_density"] == 0 if wall_boundary == "adami"
                  else check["density_rows_equal_correction_rows"])
    if not density_ok:
        problems.append(f"density and correction skipped different rows ({wall_boundary} walls): {check}")
    return problems


def compare(first_path: str, second_path: str, ignore=(), no_mask: bool = False) -> int:
    with np.load(first_path) as first_archive, np.load(second_path) as second_archive:
        first = {name: first_archive[name] for name in first_archive.files if name != "meta"}
        second = {name: second_archive[name] for name in second_archive.files if name != "meta"}
        first_meta, second_meta = json.loads(str(first_archive["meta"])), json.loads(str(second_archive["meta"]))
    problems = []
    for label, path, meta in (("first ", first_path, first_meta), ("second", second_path, second_meta)):
        print(f"{label}: {path}")
        print(f"        repo={meta.get('repo')} head={meta.get('repo_head')} dirty_v6={meta.get('repo_dirty_v6')} "
              f"K={meta.get('slabs', 1)} wall={meta.get('wall_boundary', meta.get('wall_bc'))} "
              f"steps={meta.get('steps')} canonical={meta.get('canonical_lists')} "
              f"alive={meta.get('alive')}/{meta.get('expected')} crossings={meta.get('crossings')}")
        if "code_sha256" in meta:
            print(f"        simulator {meta.get('simulator_module')}; code {meta['code_sha256']}; "
                  f"initial {str(meta.get('initial_sha256'))[:16]}; materials {str(meta.get('materials_sha256'))[:16]}")
        if meta.get("solver_switches"):
            print(f"        {meta.get('solver')} switches {meta['solver_switches']}")
        if meta.get("fused_correction_density"):
            print(f"        fused correction + density per slab {meta['fused_correction_density']}")
        if meta.get("band_overlap"):
            print(f"        band overlap per slab {meta['band_overlap']}")
        if meta.get("band_overlap_chain"):
            print(f"        band overlap chain verdict {meta['band_overlap_chain']}")
        problems += [f"{label.strip()}: {problem}" for problem in _invariant_problems(meta)]
        if meta.get("dropped_fields"):
            print(f"        not dumped (not moved by the final defrag): {meta['dropped_fields']}")
    for key in ("steps", "canonical_lists"):
        if first_meta.get(key) != second_meta.get(key):
            problems.append(f"{key} differs: {first_meta.get(key)} vs {second_meta.get(key)}")
    first_environment, second_environment = _environment(first_meta), _environment(second_meta)
    for key in sorted(set(first_environment) | set(second_environment)):
        if key in first_environment and key in second_environment:
            if first_environment[key] != second_environment[key]:
                problems.append(f"switch {key}: {first_environment[key]} vs {second_environment[key]}")
        else:
            print(f"  switch {key} set in the {'first' if key in first_environment else 'second'} run only "
                  f"({first_environment.get(key, second_environment.get(key))})")
    if not np.array_equal(first["global_id"], second["global_id"]):
        problems.append("global ids differ (alive set or order)")
    elif first["global_id"].size == 0:
        problems.append("no particles")
    # E39 B4: the skip decisions are not fields; the rows skipped by correction (either dump) are masked in
    # DEEP_WALL_MASKED_FIELDS only
    first_skip = {name: first.pop(name) for name in DEEP_WALL_SKIP_ARRAYS if name in first}
    second_skip = {name: second.pop(name) for name in DEEP_WALL_SKIP_ARRAYS if name in second}
    mask = None
    if first_skip or second_skip:
        if not np.array_equal(first["global_id"], second["global_id"]):
            problems.append("deep-wall skip data with different particle sets: no mask")
        else:
            mask = np.zeros(first["global_id"].size, dtype=bool)
        for label, skip, values, meta in (("first ", first_skip, first, first_meta),
                                          ("second", second_skip, second, second_meta)):
            if not skip:
                print(f"  deep-wall skip {label}: no skip data")
                continue
            if mask is not None:
                mask |= skip["deep_wall_skip_correction"].astype(bool)
            information = meta.get("deep_wall") or {}
            check = deep_wall_cpu_check(dict(values, **skip), information["material_kinds"],
                                        information["support_radius"])
            print(f"  deep-wall skip {label}: {information.get('resolution')}; CPU check {check}")
            problems += [f"{label.strip()}: {problem}"
                         for problem in _deep_wall_cpu_problems(check, meta.get("wall_boundary", "simple"))]
        if first_skip and second_skip:
            same = np.array_equal(first_skip["deep_wall_skip_correction"], second_skip["deep_wall_skip_correction"])
            print(f"  deep-wall skip: the two dumps skipped {'the same' if same else 'DIFFERENT'} rows")
        for counter in DEEP_WALL_COUNTERS:
            values = [[int(record.get(counter, 0)) for record in _status_records(meta)]
                      for meta in (first_meta, second_meta)]
            print(f"  {counter:30s} {values[0]} vs {values[1]}"
                  + ("" if values[0] == values[1] else "   DIFFERENT (reported, not masked)"))
        if no_mask:
            print(f"  --no-mask: {', '.join(DEEP_WALL_MASKED_FIELDS)} compared in full")
            mask = None
    compared = 0
    identical = True
    masked_note = ""
    for name in sorted(set(first) | set(second)):
        if name == "global_id":
            continue
        if name not in first or name not in second:
            where = "first" if name in first else "second"
            print(f"  {name:30s} only in {where}{' (ignored)' if name in ignore else ''}")
            if name not in ignore:
                problems.append(f"{name} only in the {where} dump")
            continue
        a, b = first[name], second[name]
        if a.shape != b.shape:
            print(f"  {name:30s} shapes differ {a.shape} {b.shape}")
            identical = False
            continue
        raw_a, raw_b = a.view(np.uint32), b.view(np.uint32)          # bitwise: -0.0 != 0.0, NaN by its bits
        rows_equal = (raw_a == raw_b).reshape(a.shape[0], -1).all(axis=1)
        note = "   (--ignore applies to one-sided fields only)" if name in ignore else ""
        if mask is not None and name in DEEP_WALL_MASKED_FIELDS:
            kept = ~mask
            kept_equal = rows_equal[kept]
            largest = (float(np.abs(a[kept].astype(np.float64) - b[kept].astype(np.float64)).max())
                       if not kept_equal.all() else 0.0)
            print(f"  {name:30s} {'identical' if kept_equal.all() else 'DIFFERENT'} {int(kept_equal.sum())} / "
                  f"{kept_equal.size} outside the mask   max |diff| {largest:.3e}; masked {int(mask.sum())} rows "
                  f"(deep walls skipped by correction; {int((~rows_equal[mask]).sum())} of them differ){note}")
            identical &= bool(kept_equal.all())
            masked_note = f"; {', '.join(DEEP_WALL_MASKED_FIELDS)}: {int(mask.sum())} deep-wall rows masked"
            compared += 1
            continue
        largest = (float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
                   if a.dtype.kind == "f" and not rows_equal.all() else 0.0)
        print(f"  {name:30s} {'identical' if rows_equal.all() else 'DIFFERENT'} {int(rows_equal.sum())} / "
              f"{rows_equal.size}   max |diff| {largest:.3e}{note}")
        identical &= bool(rows_equal.all())
        compared += 1
    if compared == 0:
        problems.append("no field compared")
    for problem in problems:
        print(f"  PROBLEM: {problem}")
    verdict = identical and not problems
    print(f"BIT-IDENTICAL ({compared} fields{masked_note})" if verdict
          else "NOT bit-identical" if not identical else "NOT COMPARABLE")
    return 0 if verdict else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compare", nargs=2, metavar=("FIRST", "SECOND"), default=None)
    parser.add_argument("--ignore", nargs="*", default=[],
                        help="with --compare: fields allowed in one dump only (fields in both are always compared)")
    parser.add_argument("--case")
    parser.add_argument("--device-map", default="1", help="one physical device index per slab (K = entries)")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--out")
    parser.add_argument("--canonical-lists", action="store_true")
    parser.add_argument("--transport-extension", action="store_true",
                        help="V6_TRANSPORT_EXTENSION=1: migrants keep their global id across a cut")
    parser.add_argument("--env", action="append", default=[], metavar="V6_X=VALUE",
                        help="add / override one V6_* switch on top of the release set (repeatable)")
    parser.add_argument("--monitor", action="store_true", help="state checks at every defrag boundary (K = 1)")
    parser.add_argument("--timestamps", action="store_true", help="with --monitor: per-kernel GPU timestamps")
    parser.add_argument("--repo", default=str(pathlib.Path(__file__).resolve().parents[2]),
                        help="checkout whose experiment/v6 runs (default: this one)")
    parser.add_argument("--solver", choices=("v6", "v7"), default="v6",
                        help="solver directory experiment/<solver> (E39: v7 = the v7-perf fork, V7_* switches)")
    parser.add_argument("--no-deep-wall-record", action="store_true",
                        help="E39 B4: do not record the deep-wall skip decisions (the production specialisation; "
                             "the dump then has no skip data)")
    parser.add_argument("--no-mask", action="store_true",
                        help="with --compare: compare the deep-wall rows of correction_inverse / "
                             "density_gradient_kernel_sum too")
    arguments = parser.parse_args()
    if arguments.compare:
        return compare(*arguments.compare, ignore=tuple(arguments.ignore), no_mask=arguments.no_mask)
    if not arguments.case or not arguments.out:
        parser.error("--case and --out are required (or --compare)")
    device_map = [int(device) for device in arguments.device_map.split(",")]
    if (arguments.monitor or arguments.timestamps) and (len(device_map) != 1 or arguments.canonical_lists):
        parser.error("--monitor / --timestamps: K = 1 and the normal bootstrap (the E36 timing method)")
    solver = arguments.solver
    prefix = solver.upper() + "_"
    overrides = {}
    for item in arguments.env:
        key, separator, value = item.partition("=")
        if not separator or not key.startswith(prefix):
            parser.error(f"--env takes {prefix}X=VALUE with --solver {solver}, got {item!r}")
        overrides[key] = value

    harness = pathlib.Path(__file__).resolve()
    repo = pathlib.Path(arguments.repo).resolve()
    out = pathlib.Path(arguments.out).resolve()
    loaded = sys.modules.get("experiment")
    if loaded is not None and repo not in pathlib.Path(loaded.__path__[0]).resolve().parents:
        sys.exit(f"experiment/ is already imported from {loaded.__path__[0]}: run this file by its path "
                 "(not with -m) to use --repo")
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    from experiment.validation.cavity_runner import RELEASE_ENVIRONMENT          # noqa: E402 (repo first)
    for key in [key for key in os.environ if key.startswith(("V6_", "V7_"))]:
        del os.environ[key]
    os.environ.update({prefix + key[len("V6_"):]: value for key, value in RELEASE_ENVIRONMENT.items()})
    if arguments.transport_extension:
        os.environ[prefix + "TRANSPORT_EXTENSION"] = "1"
    os.environ.update(overrides)
    os.environ["VK_LOADER_LAYERS_DISABLE"] = "VK_LAYER_KHRONOS_validation"
    environment = {key: value for key, value in os.environ.items() if key.startswith(prefix)}
    from vulkan import (VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, VkCommandBufferBeginInfo,  # noqa: E402
                        vkBeginCommandBuffer, vkCmdDispatch, vkEndCommandBuffer, vkFreeCommandBuffers)
    import importlib                                                              # noqa: E402

    def solver_module(name):
        return importlib.import_module(f"experiment.{solver}.utils.{name}_{solver}")

    load_case_v6 = getattr(solver_module("case_loader"), f"load_case_{solver}")
    KIND_FLUID = solver_module("case").KIND_FLUID
    ChainOrchestratorV6 = getattr(solver_module("orchestrator"), f"ChainOrchestrator{solver.upper()}")
    compute_chain_partition = solver_module("partition").compute_chain_partition
    SphSimulatorV6 = getattr(solver_module("simulator"), f"SphSimulator{solver.upper()}")
    VulkanContextV6 = getattr(solver_module("vulkan_context"), f"VulkanContext{solver.upper()}")

    # E39 B4: record the deep-wall skip decisions (a test hook of simulator_v7, read when the simulators are built)
    if hasattr(sys.modules[SphSimulatorV6.__module__], "_DEEP_WALL_RECORD_DECISIONS"):
        sys.modules[SphSimulatorV6.__module__]._DEEP_WALL_RECORD_DECISIONS = not arguments.no_deep_wall_record
    simulator_module = pathlib.Path(sys.modules[SphSimulatorV6.__module__].__file__).resolve()
    if repo not in simulator_module.parents:
        sys.exit(f"the solver was imported from {simulator_module}, not from {repo}")
    utils_directory = simulator_module.parent
    shader_directory = utils_directory.parent / "shaders" / "spv"        # the simulator's default (<prefix>SPV_DIR unset)
    if prefix + "SPV_DIR" in environment:
        shader_directory = pathlib.Path(environment[prefix + "SPV_DIR"]).resolve()
    cavity_runner_file = pathlib.Path(sys.modules["experiment.validation.cavity_runner"].__file__).resolve()
    code = {f"experiment/{solver}/utils": sha256_files(list(utils_directory.glob("*.py")), utils_directory),
            "spv": sha256_files(list(shader_directory.glob("*.spv")), shader_directory),
            "cavity_runner": hashlib.sha256(cavity_runner_file.read_bytes()).hexdigest(),
            "harness": hashlib.sha256(harness.read_bytes()).hexdigest()}
    spv_files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(shader_directory.glob("*.spv"))}

    original_build = SphSimulatorV6._build_initial_data

    def build_initial_data_with_global_ids(self):
        data = original_build(self)
        if "extension_fields" in data:
            raise RuntimeError("the solver uploads extension_fields itself")
        ids = GLOBAL_IDS[id(self.case)]
        fields = np.zeros((self.case.capacities.total_pool_capacity(), 4), dtype=np.float32)
        first = self.own_first_pid()
        fields[first:first + ids.size, 2] = (ids // GLOBAL_ID_LOW_BASE).astype(np.float32)
        fields[first:first + ids.size, 3] = (ids % GLOBAL_ID_LOW_BASE).astype(np.float32)
        data["extension_fields"] = fields.tobytes()
        return data

    SphSimulatorV6._build_initial_data = build_initial_data_with_global_ids

    case_path = pathlib.Path(arguments.case).resolve()
    case_sha256 = hashlib.sha256(case_path.read_bytes()).hexdigest()
    global_case = load_case_v6(str(case_path))
    loaded_case = {"initial_sha256": digest_of([global_case.initial.positions, global_case.initial.velocities,
                                                global_case.initial.material_group]),
                   "materials_sha256": digest_of(global_case.materials),
                   "physics_sha256": digest_of(global_case.physics),
                   "case_state_sha256": digest_of(global_case)}
    defrag_cadence = int(global_case.numerics.defrag_cadence)
    if arguments.canonical_lists and arguments.steps >= defrag_cadence:
        sys.exit(f"--canonical-lists: the run must end before the first defrag (steps < {defrag_cadence})")
    final_defrag = not arguments.canonical_lists and arguments.steps % defrag_cadence == 0
    wall_boundary = getattr(global_case.numerics, "wall_boundary", "simple")      # rc1 has no wall option
    chain = compute_chain_partition(global_case, [1.0] * len(device_map), 1.2)
    global_positions = np.ascontiguousarray(global_case.initial.positions)
    row_of = {row.tobytes(): index for index, row in enumerate(global_positions)}
    if len(row_of) != global_positions.shape[0]:
        sys.exit("the case has coincident particles: positions cannot key the global id")
    for slab in chain.slabs:
        GLOBAL_IDS[id(slab)] = np.array([row_of[row.tobytes()] for row in np.ascontiguousarray(slab.initial.positions)],
                                        dtype=np.int64)
    total = int(global_positions.shape[0])
    if sum(ids.size for ids in GLOBAL_IDS.values()) != total:
        sys.exit("the slabs do not partition the case's particles")

    contexts, sims = [], []
    for index, slab in enumerate(chain.slabs):
        context = VulkanContextV6.create(device_index=device_map[index], enable_validation=False,
                                         application_name=f"canonical_dump_s{index}")
        contexts.append(context)
        sims.append(SphSimulatorV6(context, slab, sync_scheme="per-direction"))
    orchestrator = ChainOrchestratorV6(sims, defrag_cadence=defrag_cadence)
    bench = None
    if arguments.timestamps:
        bench_v6 = solver_module("bench")
        bench = bench_v6.BenchTimer(contexts[0], label="canonical_dump")
        sims[0].bench = bench            # before bootstrap_all records the step command buffers

    def submit(sim, record) -> None:
        cmd = sim._allocate_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
        record(cmd)
        vkEndCommandBuffer(cmd)
        sim.ctx.submit_and_wait(cmd)
        vkFreeCommandBuffers(sim.ctx.device, sim.ctx.command_pool, 1, [cmd])

    def record_voxelization(sim, cmd) -> None:          # the first half of _record_bootstrap_init_cmd
        sim._record_compute_barrier(cmd)
        sim._bind_pipeline_and_sets(cmd, "initialize_voxelization")
        vkCmdDispatch(cmd, sim._per_own_particle_dispatch_count(), 1, 1)

    def record_ghost_send(sim, cmd) -> None:            # its second half (replicas from the sorted lists)
        for direction in ("leading", "trailing"):
            if direction not in sim._transport_segments:
                continue
            sim._record_compute_barrier(cmd)
            sim._record_reset_ghost_send_count(cmd, direction)
            sim._record_transfer_to_compute_barrier(cmd)
            if hasattr(sim, "_record_ghost_send_dispatch"):
                # E39 B6 (v7): the solver binds and sizes its own ghost_send (the lane-group kernel's geometry with
                # V7_GHOST_SEND_LANES > 0, exactly the two commands below with 0)
                sim._record_ghost_send_dispatch(cmd, direction)
            else:
                sim._bind_pipeline_and_sets(cmd, f"ghost_send_{direction}")
                vkCmdDispatch(cmd, sim._per_yz_face_dispatch_count(), 1, 1)
            sim._record_compute_to_transfer_barrier(cmd)
            sim._record_readback_for_direction(cmd, direction)
            sim._record_compute_to_host_barrier(cmd)

    def sort_voxel_lists(sim) -> None:
        slots = sim.case.capacities.max_particles_per_voxel
        raw = sim.readback_buffers_batch(["inside_particle_count", "inside_particle_index"])
        counts = np.frombuffer(raw["inside_particle_count"], np.uint32)
        index = np.frombuffer(raw["inside_particle_index"], np.uint32).copy()
        rows = index[:counts.size * slots].reshape(counts.size, slots)
        for voxel in np.flatnonzero(counts):
            rows[voxel, :counts[voxel]] = np.sort(rows[voxel, :counts[voxel]])
        sim._staging_upload(sim.buffers["inside_particle_index"], index.tobytes())

    if arguments.canonical_lists:
        # ChainOrchestratorV6.bootstrap_all with the lists sorted between the voxelization and the ghost round and
        # without its closing defrag
        for sim in sims:
            sim._upload_initial_state()
            submit(sim, lambda cmd, sim=sim: record_voxelization(sim, cmd))
            sort_voxel_lists(sim)
            submit(sim, lambda cmd, sim=sim: record_ghost_send(sim, cmd))
        for index in range(len(sims) - 1):
            left, right = sims[index], sims[index + 1]
            right.receiver_staging_view("leading")[:] = left.sender_staging_view("trailing")
            left.receiver_staging_view("trailing")[:] = right.sender_staging_view("leading")
        for sim in sims:
            sim.bootstrap_compute()
        for sim in sims:
            sim.prepare_step_cmd_buffers()
    else:
        orchestrator.bootstrap_all()

    out.parent.mkdir(parents=True, exist_ok=True)
    monitor_path = out.with_suffix(".monitor.jsonl")
    if arguments.monitor and monitor_path.exists():
        monitor_path.unlink()
    fluid_groups = [index for index, material in enumerate(global_case.materials) if material.kind == KIND_FLUID]
    spacing = 2.0 * min(float(material.radius) for material in global_case.materials if float(material.radius) > 0)
    status_log = []
    wall_start = time.perf_counter()

    def on_defrag(frame: int, report: list) -> None:      # the E36 k1_dump monitor (K = 1), body unchanged
        if not arguments.monitor:
            return
        sim = sims[0]
        status = sim.readback_global_status()
        overflow = {key: value for key, value in status.items() if key.startswith("overflow_") and value}
        raw = sim.readback_buffers_batch(["position_voxel_id", "velocity_mass", "material", "density_pressure"])
        capacity = sim.case.capacities.total_pool_capacity()
        first, stop = sim.own_first_pid(), sim.own_first_pid() + sim.case.capacities.own_pool_size
        position = np.frombuffer(raw["position_voxel_id"], np.float32)[:capacity * 4].reshape(capacity, 4)[first:stop]
        velocity = np.frombuffer(raw["velocity_mass"], np.float32)[:capacity * 4].reshape(capacity, 4)[first:stop]
        material = np.frombuffer(raw["material"], np.uint32)[:capacity][first:stop]
        alive = (velocity[:, 3] > 0) & (position[:, 3] > 0.5)
        fluid = alive & np.isin(material, fluid_groups)
        reach = np.abs(position[fluid, :2].astype(np.float64)).max(axis=1)
        density_pressure = np.frombuffer(raw["density_pressure"], np.float32)[:capacity * 2].reshape(capacity, 2)[first:stop]
        record = {"step": frame, "alive": int(alive.sum()), "overflow": overflow,
                  "fluid_density_mean": float(density_pressure[fluid, 0].astype(np.float64).mean()),
                  "fluid_pressure_mean": float(density_pressure[fluid, 1].astype(np.float64).mean()),
                  "fluid_pressure_p01_p99": [float(value) for value in np.percentile(density_pressure[fluid, 1], [1, 99])],
                  "fluid_beyond_effective_wall": int((reach > 0.5 + 0.5 * spacing).sum()),
                  "fluid_max_reach": float(reach.max()), "fluid_beyond_fluid_box": int((reach > 0.5).sum()),
                  "wall_density_floor_count": int(status.get("wall_density_floor_count", 0)),
                  "correction_fallback_count": int(status["correction_fallback_count"]),
                  "wall_s": time.perf_counter() - wall_start}
        if bench is not None:
            # drained: the query pool holds the last executed frame's ticks (ns); with parity regions phase C's
            # ticks of that frame carry its parity's labels
            ticks = bench.read_frame(include_defrag=False)
            if bench.parity_regions:
                ticks, _ = bench_v6.split_parity_ticks(ticks, (frame - 1) % 2)
            record["ticks_ns"] = {label: float(value) for label, value in ticks.items()}
        status_log.append(record)
        with open(monitor_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if record["alive"] != total or overflow:
            raise RuntimeError(f"invariant violation at step {frame}: {record}")

    result = orchestrator.run_pipelined(arguments.steps, depth=arguments.depth, on_defrag=on_defrag)

    status = [sim.readback_global_status() for sim in sims]
    pool_health = [sim.readback_pool_health() for sim in sims]
    # migration_install_count: installs since the last defrag (the whole run in canonical mode); the peaks: largest
    # per-step install / departure count, never reset (departed_count itself is zeroed every frame)
    crossings = {"migration_install_count": [int(record.get("migration_install_count", 0)) for record in status],
                 "peak_migration_count": [int(record.get("peak_migration_count", 0)) for record in pool_health],
                 "peak_departed_count": [int(record.get("peak_departed_count", 0)) for record in pool_health]}
    crossed = any(value for values in crossings.values() for value in values)
    transport_errors = {key: [int(record.get(key, 0)) for record in status] for key in TRANSPORT_ERROR_COUNTERS}
    transport_errors["host_stamp_error_count"] = [int(getattr(worker, "stamp_error_count", 0))
                                                  for worker in getattr(orchestrator, "workers", ())]
    # E39 B4: the slabs that skip deep walls record their decisions (a final defrag moves the rows, not the record)
    deep_wall_resolution = [list(getattr(sim, "deep_wall_skip_resolution", (False, "no V7_DEEP_WALL_SKIP")))
                            for sim in sims]
    deep_wall_active = any(active for active, _ in deep_wall_resolution)
    deep_wall_recorded = (deep_wall_active and not final_defrag
                          and all(sim.buffers["deep_wall_skip_record"].size > 16
                                  for sim, (active, _) in zip(sims, deep_wall_resolution) if active))
    # E39 B1: whether each slab ran the fused kernel (a slab that falls back records the separate kernels although
    # the switch says 1); None for a solver without the switch
    fused_resolution = ([list(sim.fused_correction_density_resolution) for sim in sims]
                        if all(hasattr(sim, "fused_correction_density_resolution") for sim in sims) else None)
    # E39 B3: whether each slab recorded the V7_BAND_OVERLAP layout ([on, reason with the plan] per slab) and the chain
    # verdict auto decides once for every slab ({verdict, reason, every_slab_alike}; a verdict, the slabs' state is
    # per slab); None for a solver without the switch
    band_overlap_record = getattr(solver_module("simulator"), "band_overlap_record", lambda simulators: None)(sims)
    band_overlap_resolution = band_overlap_record["slabs"] if band_overlap_record else None
    band_overlap_chain = band_overlap_record["chain"] if band_overlap_record else None
    rows: dict = {}
    for sim in sims:
        fields = DUMPED_FIELDS
        if wall_boundary == "adami":
            if "wall_dummy_velocity" not in sim.buffers:
                raise RuntimeError("wall_boundary adami, but the simulator has no wall_dummy_velocity buffer")
            fields = DUMPED_FIELDS + ADAMI_FIELDS
        if final_defrag:
            fields = tuple(field for field in fields if field[0] not in NOT_MOVED_BY_DEFRAG)
        raw = sim.readback_buffers_batch([name for name, _, _ in fields], density="stored")
        capacity = sim.case.capacities.total_pool_capacity()
        first, stop = sim.own_first_pid(), sim.own_first_pid() + sim.case.capacities.own_pool_size
        part = {}
        for name, components, dtype in fields:
            flat = np.frombuffer(raw[name], dtype=dtype)[:capacity * components]
            part[name] = np.array((flat.reshape(capacity, components) if components > 1 else flat)[first:stop])
        if deep_wall_recorded:
            # E39 B4: [pid * 2 + kernel] = frame_stamp + 1 of the last step that kernel skipped pid (a slab that does
            # not skip has no record: all 0)
            last_step = np.uint32(int(status[sims.index(sim)]["frame_stamp"]) + 1)
            record = np.zeros((stop - first, 2), dtype=np.uint32)
            if "deep_wall_skip_record" in sim.buffers:
                record = np.frombuffer(sim.readback_buffer_by_name("deep_wall_skip_record"),
                                       dtype=np.uint32)[:capacity * 2].reshape(capacity, 2)[first:stop]
            part["deep_wall_skip_correction"] = (record[:, 0] == last_step).astype(np.uint8)
            part["deep_wall_skip_density"] = (record[:, 1] == last_step).astype(np.uint8)
        alive = (part["velocity_mass"][:, 3] > 0) & (part["position_voxel_id"][:, 3] > 0.5)
        for name, values in part.items():
            rows.setdefault(name, []).append(values[alive])
    state = {name: np.concatenate(parts) for name, parts in rows.items()}
    extension = state["extension_fields"].astype(np.float64)
    ids = np.rint(extension[:, 2]).astype(np.int64) * GLOBAL_ID_LOW_BASE + np.rint(extension[:, 3]).astype(np.int64)
    if crossed and environment.get(prefix + "TRANSPORT_EXTENSION") != "1":
        raise RuntimeError(f"particles crossed a cut ({crossings}): the migrants' global ids were not transported; "
                           "rerun with --transport-extension")
    if np.any(extension[:, :2] != 0) or len(np.unique(ids)) != ids.size:
        raise RuntimeError("extension_fields no longer hold unique global ids")
    order = np.argsort(ids)
    state = {name: values[order] for name, values in state.items()}
    state["global_id"] = ids[order]
    deep_wall = None
    if deep_wall_active:
        deep_wall = {"resolution": deep_wall_resolution, "recorded": deep_wall_recorded,
                     "material_kinds": [int(material.kind) for material in global_case.materials],
                     "support_radius": float(global_case.physics.smoothing_length),
                     "dimension": int(global_case.physics.dimension)}
        if deep_wall_recorded:
            deep_wall["cpu_check"] = deep_wall_cpu_check(state, deep_wall["material_kinds"],
                                                          deep_wall["support_radius"])
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                          cwd=repo).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", f"experiment/{solver}"], capture_output=True,
                                text=True, cwd=repo).stdout.strip())
    overflow = {f"slab{index}.{key}": value for index, record in enumerate(status)
                for key, value in record.items() if key.startswith("overflow_") and value}
    # E39: the solver's own switch registry (v7: configured_v7_switches, defaults included), so a dump names its build
    switches_function = getattr(solver_module("simulator"), f"configured_{solver}_switches", None)
    solver_switches = switches_function() if switches_function else {}
    meta = {"repo": str(repo), "repo_head": head, "repo_dirty_v6": dirty, "solver": solver,
            "solver_switches": solver_switches, "case": arguments.case,
            "case_path": str(case_path), "case_sha256": case_sha256, **loaded_case,
            "simulator_module": str(simulator_module),
            "code_sha256": {key: value[:16] for key, value in code.items()}, "code_sha256_full": code,
            "spv_files": spv_files, "device_local_buffers": len(sims[0].buffers),
            "slabs": len(sims), "device_map": device_map, "steps": arguments.steps, "depth": arguments.depth,
            "defrag_cadence": defrag_cadence, "canonical_lists": arguments.canonical_lists,
            "dropped_fields": list(NOT_MOVED_BY_DEFRAG) if final_defrag else [],
            "wall_boundary": wall_boundary, "environment": environment, "cuts": [int(cut) for cut in chain.cuts],
            "alive": int(ids.size), "expected": total, "overflow": overflow, "transport_errors": transport_errors,
            "crossings": crossings, "status": status, "pool_health": pool_health, "deep_wall": deep_wall,
            "fused_correction_density": fused_resolution,
            "band_overlap": band_overlap_resolution, "band_overlap_chain": band_overlap_chain,
            "fps": result.get("fps"),
            "elapsed_s": result.get("elapsed_s"), "dt": float(global_case.physics.timestep),
            "device_names": [context.device_name for context in contexts]}
    np.savez(out, meta=json.dumps(meta), **state)
    problems = _invariant_problems(meta)
    if deep_wall and deep_wall.get("cpu_check"):
        problems += _deep_wall_cpu_problems(deep_wall["cpu_check"], wall_boundary)
    fps = meta["fps"]
    print(f"[canonical_dump] {solver} {head}{' (dirty)' if dirty else ''} K={len(sims)} wall={wall_boundary} "
          f"{arguments.steps} steps: alive {meta['alive']}/{total}, crossings {crossings}, "
          f"fps {fps if fps is None else round(fps)}, buffers {meta['device_local_buffers']}, "
          f"code {meta['code_sha256']}, invariants {'ok' if not problems else problems}"
          + (f", switches {solver_switches}" if solver_switches else "")
          + (f", band overlap on {sum(1 for active, _ in band_overlap_resolution if active)}/{len(sims)} slab(s)"
             if band_overlap_resolution else "")
          + (f" (chain verdict {'on' if band_overlap_chain['verdict'] else 'off'}"
             f"{'' if band_overlap_chain['every_slab_alike'] else ', NOT alike on every slab'})"
             if band_overlap_chain else "")
          + (f", deep walls {deep_wall['resolution']} {deep_wall.get('cpu_check', 'not recorded')}"
             if deep_wall else "") + f" -> {out}", flush=True)
    orchestrator.destroy()
    for sim in sims:
        sim.destroy()
    for context in contexts:
        context.destroy()
    return 0 if not problems else 3


if __name__ == "__main__":
    sys.exit(main())
