"""
dump_state.py — seam-audit GPU worker: run ONE chain configuration and dump the
full per-particle state at a list of horizons, keyed by a global particle id.

One process = one run (one solver version, one case, one slab count K). The
dumps of several runs are compared afterwards on the CPU by analyze.py; the
campaign driver is run_matrix.py.

Global particle id (no solver change): this process monkeypatches the
simulator's ``_build_initial_data`` so the initial upload also fills the
``extension_fields`` buffer (vec4 per pool slot, unused by the V5/V6 physics):

    own slot own_first_pid() + k  ->  z = float(global_id // 2**20)
                                      w = float(global_id %  2**20)

(both exact in float32; x and y stay 0). ghost_send, install_migrations and
defrag copy extension_fields bit-exactly, so the id follows every particle
through migration and defrag. The field is never interpreted as anything else;
a non-zero x/y or a non-integral z/w at readback is reported as an invariant
violation (it would mean the solver started using the field itself).

The global id of a particle is its row index in the degenerate global case
returned by load_case (the order of the case's OBJ files). Each slab's own
particles are recovered with exactly the mask of
partition_*._filter_particles_by_x_range, and the result is asserted to be
bit-identical to the slab's initial arrays.

Frame loop: run_frames() mirrors ChainOrchestratorV5.run_pipelined's default
loop (submit-ahead with bounded in-flight depth, drain + defrag every
defrag_cadence frames) but with CONTINUING frame numbers, so the run can stop at
every horizon (run_pipelined restarts at frame 0 and cannot be called twice: the
timeline values and the workers' frame-stamp check need monotonic frame
numbers). For each horizon N: run to N-1, snapshot ownership (id -> slab), run
frame N-1, dump. The pipeline drains and readbacks add no GPU work that changes
state; a defrag boundary that coincides with a horizon runs AFTER the dump
(read-only), and the one that coincides with the final horizon is skipped.

Usage (GPU — run only when the GPUs are free):
    .venv/Scripts/python.exe -m experiment.seam_audit.dump_state --version v5 \\
        --case cases/lid_driven_cavity_2d_gen/case.yaml --slabs 2 --device-map 0,1 \\
        --horizons 300,2000 --out-dir logs/seam_audit/manual --run-name v5_K2_t1

CPU-only check of the case load, partition and global-id mask (no Vulkan):
    .venv/Scripts/python.exe -m experiment.seam_audit.dump_state --version v5 \\
        --case cases/lid_driven_cavity_2d_gen/case.yaml --slabs 4 --dry-run

The last stdout line is always "[seam_audit] RESULT {json}". Exit code: 0 =
every horizon valid, 3 = completed but an invariant failed, 1 = error. On an
error (a frame stall, a dead transport worker, Ctrl+C, ...) the worker prints
the traceback and the RESULT line and leaves through os._exit WITHOUT Vulkan
teardown: frames still in flight would block vkDeviceWaitIdle forever.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import platform
import sys
import time
import traceback

import numpy as np

try:
    from experiment.seam_audit.crossing_window import (
        CrossingWindowRecorder, cut_line_positions)
except ImportError:  # run as a script from this directory
    from crossing_window import CrossingWindowRecorder, cut_line_positions

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from experiment.seam_audit.solver_adapter import (  # noqa: E402
    SUPPORTED_VERSIONS,
    load_solver,
)

RESULT_PREFIX = "[seam_audit] RESULT "
LOG_PREFIX = "[seam_audit]"
DUMP_FORMAT_VERSION = 1



def enable_audit_transport() -> None:
    """Audit mode: the global ids ride in extension_fields, so a v6 build with
    lean ghost packets (V6_LEAN_TRANSPORT=1) must still carry that field across
    the link. Called first thing by the audit workers (dump_state, single_step);
    the physics does not read extension_fields."""
    os.environ["V6_TRANSPORT_EXTENSION"] = "1"

# Global id encoding inside extension_fields (z = high part, w = low part).
GLOBAL_ID_LOW_BASE = 2 ** 20
GLOBAL_ID_HIGH_COMPONENT = 2
GLOBAL_ID_LOW_COMPONENT = 3
GLOBAL_ID_ATTRIBUTE = "_seam_audit_global_ids"
INJECTION_MARKER = "_seam_audit_injection"

UNKNOWN_SLAB = 255          # ownership value of an id not seen alive
DISABLED_DEFRAG_CADENCE = 10 ** 12

EXIT_VALID = 0
EXIT_ERROR = 1
EXIT_INVALID = 3

# Components per pool slot of each buffer read back (std430 layouts, common.glsl).
BUFFER_COMPONENT_COUNTS = {
    "position_voxel_id": 4,             # x, y, z, voxel_id (0 = dead)
    "velocity_mass": 4,                 # vx, vy, vz, mass (0 = unallocated)
    "density_pressure": 2,              # rho, p
    "acceleration": 4,                  # ax, ay, az, reserved
    "shift": 4,                         # shift xyz, reserved
    "density_gradient_kernel_sum": 4,   # grad rho xyz, kernel_sum
    "extension_fields": 4,              # x, y unused (must stay 0); z, w = global id
    "material": 1,                      # uint material group
}
STATE_BUFFER_NAMES = (
    "position_voxel_id", "velocity_mass", "density_pressure", "acceleration",
    "shift", "density_gradient_kernel_sum", "extension_fields", "material",
)
OWNERSHIP_BUFFER_NAMES = ("position_voxel_id", "velocity_mass", "extension_fields")
RECORDED_ENVIRONMENT_PREFIXES = ("V5_", "V6_", "VK_")
# Recorded and warned about when non-zero, but NOT part of 'valid' (the validity
# rule is: every overflow_* counter + stamp errors). v6: migrants found in the
# outer ghost column, a two-column jump that CFL should make impossible.
WATCHED_STATUS_KEYS = ("far_migration_count",)


# =============================================================================
# Command line + file layout (also imported by run_matrix.py — keep this module
# free of solver imports at top level)
# =============================================================================

def parse_arguments(argument_list=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seam audit worker: run one chain configuration and dump "
                    "per-particle state at each horizon (see module docstring).")
    parser.add_argument("--version", choices=SUPPORTED_VERSIONS, default="v5")
    parser.add_argument("--case", required=True, help="case.yaml path")
    parser.add_argument("--slabs", type=int, required=True, help="K, number of slabs")
    parser.add_argument("--weights", default=None,
                        help="K comma-separated slab weights (default: all 1.0)")
    parser.add_argument("--device-map", default="0,1",
                        help="comma-separated physical device indices, cycled over "
                             "the sims (discrete-first order of the V5/V6 context)")
    parser.add_argument("--horizons", default="300,2000",
                        help="comma-separated frame counts at which to dump")
    parser.add_argument("--depth", type=int, default=2, help="frames in flight")
    parser.add_argument("--pool-safety", type=float, default=1.2,
                        help="per-slab own pool = ceil(particles * safety); 0 = global pool")
    parser.add_argument("--sync-scheme", default="per-direction",
                        choices=["aggregated", "per-direction"])
    parser.add_argument("--defrag-cadence", type=int, default=None,
                        help="frames between defrags (default: case numerics."
                             "defrag_cadence; 0 disables defrag)")
    parser.add_argument("--out-dir", default="logs/seam_audit/manual")
    parser.add_argument("--run-name", default=None,
                        help="file stem of the dumps (default: <version>_K<slabs>)")
    parser.add_argument("--keep-previous", action="store_true",
                        help="also save the N-1 ownership snapshot (id, slab) per horizon")
    parser.add_argument("--validation", action="store_true",
                        help="enable the Vulkan validation layer (default off)")
    parser.add_argument("--stall-timeout", type=float, default=120.0,
                        help="seconds a frame may take before the orchestrator autopsy")
    parser.add_argument("--window", type=int, default=0,
                        help="crossing-capture window: for the last W frames before each "
                             "horizon, step one frame at a time and store every particle "
                             "within 1.5 h of a particle that crossed an audit cut line "
                             "during that frame (crossing_window.py). 0 = off")
    parser.add_argument("--window-horizons", default="",
                        help="comma-separated horizons that get the window (default: all)")
    parser.add_argument("--capture-requests-dir", default="",
                        help="reference pass of the two-pass window: directory with "
                             "requests_N<horizon>.npz (frame, id) built from every test run's "
                             "window; the window then captures exactly those particles at "
                             "those frames instead of detecting crossings itself")
    parser.add_argument("--audit-slab-counts", default="",
                        help="comma-separated K values whose equal-weight cut lines the "
                             "window watches (a K=1 reference must name the K of the test "
                             "runs it serves); this run's own cuts are always included")
    parser.add_argument("--dry-run", action="store_true",
                        help="CPU only: load + partition + global-id mask check, "
                             "then stop before creating any Vulkan context")
    return parser.parse_args(argument_list)


def parse_horizons(text: str) -> list[int]:
    horizons = sorted({int(item) for item in text.split(",") if item.strip()})
    if not horizons or horizons[0] < 1:
        raise ValueError(f"--horizons needs positive frame counts, got {text!r}")
    return horizons


def parse_device_map(text: str) -> list[int]:
    devices = [int(item) for item in text.split(",") if item.strip()]
    if not devices:
        raise ValueError(f"--device-map is empty: {text!r}")
    return devices


def parse_weights(text, slab_count: int) -> list[float]:
    if text is None or not str(text).strip():
        return [1.0] * slab_count
    weights = [float(item) for item in str(text).split(",") if item.strip()]
    if len(weights) != slab_count:
        raise ValueError(f"--weights has {len(weights)} entries, --slabs is {slab_count}")
    return weights


def window_file_path(out_dir, run_name: str, horizon: int) -> pathlib.Path:
    return pathlib.Path(out_dir) / f"{run_name}_N{horizon}_window.npz"


def audit_cut_lines(solver, global_case, chain, arguments):
    """World x of every cut line the window watches: this run's own cuts plus
    the equal-weight cuts of every K in --audit-slab-counts (same partition
    code and pool safety as the test runs, so the columns are identical)."""
    cut_columns = set(int(cut) for cut in chain.cuts)
    for item in str(arguments.audit_slab_counts).split(","):
        item = item.strip()
        if not item:
            continue
        slab_count = int(item)
        if slab_count > 1:
            audit_chain = solver.compute_chain_partition(
                global_case, [1.0] * slab_count, arguments.pool_safety)
            cut_columns.update(int(cut) for cut in audit_chain.cuts)
    return cut_line_positions(float(global_case.grid.origin_x),
                              float(global_case.physics.smoothing_length),
                              sorted(cut_columns))


class OwnPositionReader:
    """Per-frame reader for the crossing window. readback_buffers_batch
    allocates and maps a staging buffer per buffer per call (~0.4 s per frame
    at 1M); this keeps ONE host-cached staging buffer per sim mapped for the
    whole run and replays a pre-recorded copy of the own range of
    position_voxel_id and extension_fields. Alive = voxel id > 0.5: every kill
    path of the solvers (predict out of grid, incoming overflow, ghost_send
    migration, install rollback, update_voxel overflow) zeroes the position
    vec4, and dead slots are zero. Only used between drained frames, when the
    compute queue is otherwise idle."""

    def __init__(self, sim) -> None:
        from vulkan import (VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                            VkBufferCopy, VkCommandBufferBeginInfo, vkBeginCommandBuffer,
                            vkCmdCopyBuffer, vkEndCommandBuffer, vkMapMemory)
        self.sim = sim
        own_first = sim.own_first_pid()
        self.own_count = sim.case.capacities.own_pool_size
        byte_count = 32 * self.own_count                    # 16 B position + 16 B extension
        self.staging = sim._allocate_buffer(
            byte_count, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
        mapped = vkMapMemory(sim.ctx.device, self.staging.memory, 0, byte_count, 0)
        self.view = np.frombuffer(mapped, dtype=np.float32, count=byte_count // 4)
        self.command_buffer = sim._allocate_oneshot_cmd()
        vkBeginCommandBuffer(self.command_buffer, VkCommandBufferBeginInfo())
        for index, name in enumerate(("position_voxel_id", "extension_fields")):
            vkCmdCopyBuffer(self.command_buffer, sim.buffers[name].handle, self.staging.handle, 1, [
                VkBufferCopy(srcOffset=16 * own_first, dstOffset=index * 16 * self.own_count,
                             size=16 * self.own_count)])
        vkEndCommandBuffer(self.command_buffer)

    def read(self, initial_total: int) -> tuple:
        self.sim.ctx.submit_and_wait(self.command_buffer)
        positions = self.view[:4 * self.own_count].reshape(self.own_count, 4)
        extension = self.view[4 * self.own_count:].reshape(self.own_count, 4)
        alive = positions[:, 3] > 0.5
        ids, _foreign = decode_global_ids(extension[alive])
        valid = (ids >= 0) & (ids < initial_total)
        return ids[valid], positions[alive][valid, 0].astype(np.float64)

    def destroy(self) -> None:
        from vulkan import vkDestroyBuffer, vkFreeCommandBuffers, vkFreeMemory, vkUnmapMemory
        device = self.sim.ctx.device
        vkFreeCommandBuffers(device, self.sim.ctx.command_pool, 1, [self.command_buffer])
        vkUnmapMemory(device, self.staging.memory)
        vkDestroyBuffer(device, self.staging.handle, None)
        vkFreeMemory(device, self.staging.memory, None)


def load_capture_requests(directory, horizon: int):
    """{frame label -> sorted global ids} for one horizon, or None."""
    if not directory:
        return None
    request_path = pathlib.Path(directory) / f"requests_N{horizon}.npz"
    if not request_path.exists():
        return None
    with np.load(request_path) as archive:
        frames = archive["frame"].astype(np.int64)
        ids = archive["id"].astype(np.int64)
    requests = {}
    for frame in np.unique(frames):
        requests[int(frame)] = np.unique(ids[frames == frame])
    return requests


def requested_rows(state: dict, requested_ids: np.ndarray, frame_label: int) -> dict:
    rows = np.flatnonzero(np.isin(state["id"].astype(np.int64), requested_ids))
    record = {key: value[rows] for key, value in state.items()
              if isinstance(value, np.ndarray) and value.shape[:1] == state["id"].shape}
    record["crossed_this_frame"] = np.zeros(rows.size, dtype=bool)
    record["frame"] = np.full(rows.size, frame_label, dtype=np.int32)
    return record


def collect_positions(readers, initial_total: int) -> tuple:
    """(global ids, x) of every alive own particle over all sims."""
    id_parts, x_parts = [], []
    for reader in readers:
        ids, x = reader.read(initial_total)
        id_parts.append(ids)
        x_parts.append(x)
    return np.concatenate(id_parts), np.concatenate(x_parts)


def dump_file_paths(out_dir, run_name: str, horizon: int) -> dict:
    """Canonical dump file names of one run at one horizon."""
    directory = pathlib.Path(out_dir)
    stem = f"{run_name}_N{int(horizon)}"
    return {
        "npz": directory / f"{stem}.npz",
        "json": directory / f"{stem}.json",
        "previous": directory / f"{stem}_previous.npz",
    }


def json_default(value):
    """json.dumps hook for numpy scalars / arrays and paths."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pathlib.Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def recorded_environment() -> dict:
    return {key: value for key, value in sorted(os.environ.items())
            if key.startswith(RECORDED_ENVIRONMENT_PREFIXES)}


def save_npz_atomically(path: pathlib.Path, arrays: dict) -> None:
    temporary_path = path.with_name(path.stem + ".partial.npz")
    np.savez(temporary_path, **arrays)          # uncompressed on purpose
    os.replace(temporary_path, path)


def save_json_atomically(path: pathlib.Path, document: dict) -> None:
    temporary_path = path.with_name(path.stem + ".partial.json")
    temporary_path.write_text(json.dumps(document, indent=1, default=json_default),
                              encoding="utf-8")
    os.replace(temporary_path, path)


# =============================================================================
# Global ids: slab membership (CPU), encoding, injection, decoding
# =============================================================================

def slab_global_indices(global_case, chain) -> list[np.ndarray]:
    """Global row indices of each slab's own particles, in the slab's initial
    order — exactly the mask of partition_*._filter_particles_by_x_range
    (x_index = floor((x - origin_x) / h) clipped to [0, nx-1], own columns
    [own_global_first_column, own_global_last_column]). Asserted bit-identical
    against every slab's initial arrays, and the slabs must partition the case."""
    positions = global_case.initial.positions
    smoothing_length = global_case.physics.smoothing_length
    origin_x = global_case.grid.origin_x
    # Same expression and dtypes as the partitioner, so the floor matches bit for bit.
    x_indices = np.floor((positions[:, 0] - origin_x) / smoothing_length).astype(np.int64)
    grid_column_count = global_case.grid.grid_dimension_x
    np.clip(x_indices, 0, grid_column_count - 1, out=x_indices)

    indices_per_slab = []
    for slab_index, (geometry, slab_case) in enumerate(zip(chain.geometry, chain.slabs)):
        first_column = geometry.own_global_first_column
        last_column = geometry.own_global_last_column
        mask = (x_indices >= first_column) & (x_indices < last_column + 1)
        global_indices = np.flatnonzero(mask).astype(np.int64)
        initial = slab_case.initial
        checks = {
            "positions": np.array_equal(positions[global_indices], initial.positions),
            "velocities": np.array_equal(global_case.initial.velocities[global_indices],
                                         initial.velocities),
            "material_group": np.array_equal(global_case.initial.material_group[global_indices],
                                             initial.material_group),
        }
        global_densities = getattr(global_case.initial, "densities", None)
        if global_densities is not None:
            checks["densities"] = np.array_equal(global_densities[global_indices],
                                                 initial.densities)
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise AssertionError(
                f"slab {slab_index}: global-id mask does not reproduce the partition's "
                f"initial arrays ({', '.join(failed)}); the x-range filter changed?")
        indices_per_slab.append(global_indices)

    membership_count = np.zeros(positions.shape[0], dtype=np.int64)
    for global_indices in indices_per_slab:
        membership_count[global_indices] += 1
    if not np.all(membership_count == 1):
        raise AssertionError(
            f"slabs do not partition the case: {int((membership_count == 0).sum())} "
            f"particles in no slab, {int((membership_count > 1).sum())} in several")
    return indices_per_slab


def encode_global_ids(global_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(high, low) float32 components; exact while high < 2**24."""
    global_ids = np.asarray(global_ids, dtype=np.int64)
    if global_ids.size and (global_ids.min() < 0
                            or global_ids.max() >= GLOBAL_ID_LOW_BASE * 2 ** 24):
        raise ValueError("global id out of the exactly representable range")
    high = (global_ids // GLOBAL_ID_LOW_BASE).astype(np.float32)
    low = (global_ids % GLOBAL_ID_LOW_BASE).astype(np.float32)
    return high, low


def decode_global_ids(extension_fields: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode (N, 4) extension_fields rows. Returns (ids, foreign): ids is int64
    with -1 where z/w are not a valid encoding; foreign marks rows whose unused
    x/y components are non-zero (the solver wrote the field)."""
    high = extension_fields[:, GLOBAL_ID_HIGH_COMPONENT].astype(np.float64)
    low = extension_fields[:, GLOBAL_ID_LOW_COMPONENT].astype(np.float64)
    well_formed = (np.isfinite(high) & np.isfinite(low)
                   & (high == np.rint(high)) & (low == np.rint(low))
                   & (high >= 0) & (low >= 0) & (low < GLOBAL_ID_LOW_BASE))
    ids = np.full(extension_fields.shape[0], -1, dtype=np.int64)
    ids[well_formed] = (np.rint(high[well_formed]).astype(np.int64) * GLOBAL_ID_LOW_BASE
                        + np.rint(low[well_formed]).astype(np.int64))
    foreign = np.any(extension_fields[:, 0:2] != 0, axis=1)
    return ids, foreign


def check_encoding_round_trip(particle_count: int) -> None:
    global_ids = np.arange(particle_count, dtype=np.int64)
    high, low = encode_global_ids(global_ids)
    extension_fields = np.zeros((particle_count, 4), dtype=np.float32)
    extension_fields[:, GLOBAL_ID_HIGH_COMPONENT] = high
    extension_fields[:, GLOBAL_ID_LOW_COMPONENT] = low
    decoded, foreign = decode_global_ids(extension_fields)
    if not np.array_equal(decoded, global_ids) or foreign.any():
        raise AssertionError("global id float32 encoding does not round-trip")


def install_global_id_injection(simulator_class) -> None:
    """Wrap ``simulator_class._build_initial_data`` (process-local monkeypatch)
    so the initial payload also carries ``extension_fields`` with the global
    ids that dump_state attached to the sim object (GLOBAL_ID_ATTRIBUTE)."""
    original_build = simulator_class._build_initial_data
    if getattr(original_build, INJECTION_MARKER, False):
        return

    def build_initial_data_with_global_ids(self):
        data = original_build(self)
        global_ids = getattr(self, GLOBAL_ID_ATTRIBUTE, None)
        if global_ids is None:
            raise RuntimeError("seam audit: simulator has no global ids attached "
                               f"(set {GLOBAL_ID_ATTRIBUTE} right after construction)")
        if "extension_fields" in data:
            raise RuntimeError("seam audit: the solver now uploads extension_fields "
                               "itself; the global id cannot ride in it any more")
        particle_count = self.case.initial.positions.shape[0]
        if len(global_ids) != particle_count:
            raise RuntimeError(f"seam audit: {len(global_ids)} global ids for "
                               f"{particle_count} initial particles")
        pool_capacity = self.case.capacities.total_pool_capacity()
        own_first = self.own_first_pid()
        extension_fields = np.zeros((pool_capacity, 4), dtype=np.float32)
        high, low = encode_global_ids(global_ids)
        extension_fields[own_first:own_first + particle_count, GLOBAL_ID_HIGH_COMPONENT] = high
        extension_fields[own_first:own_first + particle_count, GLOBAL_ID_LOW_COMPONENT] = low
        data["extension_fields"] = extension_fields.tobytes()
        return data

    setattr(build_initial_data_with_global_ids, INJECTION_MARKER, True)
    simulator_class._build_initial_data = build_initial_data_with_global_ids


# =============================================================================
# Readback
# =============================================================================

def read_own_slots(sim, buffer_names) -> tuple[dict, np.ndarray]:
    """One batched readback; returns the OWN pool range of every buffer
    (copies, so the raw bytes can be freed) and the alive mask
    (mass > 0 and voxel id > 0.5)."""
    raw = sim.readback_buffers_batch(list(buffer_names))
    capacities = sim.case.capacities
    pool_capacity = capacities.total_pool_capacity()
    own_first = sim.own_first_pid()
    own_stop = own_first + capacities.own_pool_size
    arrays = {}
    for name in buffer_names:
        component_count = BUFFER_COMPONENT_COUNTS[name]
        element_type = np.uint32 if name == "material" else np.float32
        flat = np.frombuffer(raw[name], dtype=element_type)[:pool_capacity * component_count]
        if component_count > 1:
            flat = flat.reshape(pool_capacity, component_count)
        arrays[name] = np.array(flat[own_first:own_stop])
    del raw
    alive = ((arrays["velocity_mass"][:, 3] > 0)
             & (arrays["position_voxel_id"][:, 3] > 0.5))
    return arrays, alive


def collect_ownership(sims, initial_total: int) -> tuple[np.ndarray, dict]:
    """id -> owning slab index (UNKNOWN_SLAB if not alive anywhere)."""
    owner = np.full(initial_total, UNKNOWN_SLAB, dtype=np.uint8)
    valid_ids_per_slab = []
    invalid_count = 0
    foreign_count = 0
    alive_total = 0
    for slab_index, sim in enumerate(sims):
        arrays, alive = read_own_slots(sim, OWNERSHIP_BUFFER_NAMES)
        ids, foreign = decode_global_ids(arrays["extension_fields"][alive])
        valid = (ids >= 0) & (ids < initial_total)
        owner[ids[valid]] = slab_index
        valid_ids_per_slab.append(ids[valid])
        invalid_count += int((~valid).sum())
        foreign_count += int(foreign.sum())
        alive_total += int(alive.sum())
    counts = np.bincount(np.concatenate(valid_ids_per_slab), minlength=initial_total)
    statistics = {
        "alive_total": alive_total,
        "missing_ids": int((counts == 0).sum()),
        "duplicate_ids": int(np.maximum(counts - 1, 0).sum()),
        "invalid_ids": invalid_count,
        "foreign_extension_values": foreign_count,
    }
    return owner, statistics


def collect_state(sims, dimension: int, initial_total: int,
                  previous_owner: np.ndarray) -> tuple[dict, dict]:
    """Full dump arrays (sorted by global id) + id bookkeeping."""
    parts = []
    invalid_count = 0
    foreign_count = 0
    alive_total = 0
    alive_per_slab = []
    for slab_index, sim in enumerate(sims):
        arrays, alive = read_own_slots(sim, STATE_BUFFER_NAMES)
        alive_count = int(alive.sum())
        alive_total += alive_count
        alive_per_slab.append(alive_count)
        alive_slots = np.flatnonzero(alive)
        ids, foreign = decode_global_ids(arrays["extension_fields"][alive_slots])
        foreign_count += int(foreign.sum())
        valid = (ids >= 0) & (ids < initial_total)
        invalid_count += int((~valid).sum())
        selected_slots = alive_slots[valid]
        density_pressure = arrays["density_pressure"][selected_slots]
        parts.append({
            "id": ids[valid],
            "slab": np.full(selected_slots.size, slab_index, dtype=np.uint8),
            "position": arrays["position_voxel_id"][selected_slots, :dimension],
            "velocity": arrays["velocity_mass"][selected_slots, :dimension],
            "acceleration": arrays["acceleration"][selected_slots, :dimension],
            "shift": arrays["shift"][selected_slots, :dimension],
            "density": density_pressure[:, 0],
            "pressure": density_pressure[:, 1],
            "kernel_sum": arrays["density_gradient_kernel_sum"][selected_slots, 3],
            "material": arrays["material"][selected_slots].astype(np.uint16),
        })
        del arrays

    merged = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
    order = np.argsort(merged["id"], kind="stable")
    state = {name: np.ascontiguousarray(values[order]) for name, values in merged.items()}
    ids = state["id"]
    counts = np.bincount(ids, minlength=initial_total)
    state["id"] = ids.astype(np.uint32)
    state["previous_slab"] = previous_owner[ids]
    state["crossed_last_step"] = ((state["previous_slab"] != UNKNOWN_SLAB)
                                  & (state["previous_slab"] != state["slab"]))
    for name in ("position", "velocity", "acceleration", "shift",
                 "density", "pressure", "kernel_sum"):
        state[name] = state[name].astype(np.float32, copy=False)

    crossed = state["crossed_last_step"]
    crossings_by_direction: dict = {}
    for source_slab, destination_slab in zip(state["previous_slab"][crossed],
                                             state["slab"][crossed]):
        key = f"{int(source_slab)}->{int(destination_slab)}"
        crossings_by_direction[key] = crossings_by_direction.get(key, 0) + 1
    bookkeeping = {
        "alive_total": alive_total,
        "alive_per_slab": alive_per_slab,
        "missing_ids": int((counts == 0).sum()),
        "duplicate_ids": int(np.maximum(counts - 1, 0).sum()),
        "invalid_ids": invalid_count,
        "foreign_extension_values": foreign_count,
        "crossed_last_step_count": int(crossed.sum()),
        "crossings_by_direction": dict(sorted(crossings_by_direction.items())),
        "unknown_previous_owner": int((state["previous_slab"] == UNKNOWN_SLAB).sum()),
    }
    return state, bookkeeping


def read_statuses(sims) -> tuple[list[dict], list[dict]]:
    statuses = [dict(sim.readback_global_status()) for sim in sims]
    healths = [dict(sim.readback_pool_health()) for sim in sims]
    return statuses, healths


# =============================================================================
# Frame loop (mirrors ChainOrchestratorV5.run_pipelined's default loop)
# =============================================================================

def perform_defrag(orchestrator, frame_number: int, defrag_log: list) -> None:
    """The run_pipelined defrag boundary body (pipeline already drained):
    snapshot the migration report, then defrag every sim. The extra
    readback_global_status per sim is read-only (overflow / stamp invariants)."""
    report = orchestrator._collect_defrag_report()
    statuses = [dict(sim.readback_global_status()) for sim in orchestrator.sims]
    defrag_log.append({"frame": int(frame_number), "report": report,
                       "global_status": statuses})
    for sim in orchestrator.sims:
        sim.submit_defrag_and_wait()


def run_frames(orchestrator, start_frame: int, stop_frame: int, depth: int,
               defrag_cadence: int, defrag_log: list, *, defrag_at_stop: bool = True,
               stall_timeout_seconds: float = 120.0) -> None:
    """Run frames [start_frame, stop_frame) exactly like run_pipelined's default
    loop: submit frame n, keep at most ``depth`` frames in flight, and when the
    submitted count reaches a multiple of ``defrag_cadence`` drain + defrag.
    Continues the caller's frame numbering; drains at the end. With
    ``defrag_at_stop=False`` a boundary that falls exactly on ``stop_frame`` is
    left to the caller (it must not run before the horizon dump)."""
    depth = max(1, depth)
    frame_number = start_frame
    next_wait = start_frame
    while frame_number < stop_frame:
        orchestrator._submit_frame(frame_number)
        frame_number += 1
        orchestrator._frame_count = frame_number
        while frame_number - next_wait >= depth:
            orchestrator._wait_frame(next_wait, stall_timeout_seconds)
            next_wait += 1
        if frame_number % defrag_cadence == 0:
            if frame_number == stop_frame and not defrag_at_stop:
                continue
            while next_wait < frame_number:
                orchestrator._wait_frame(next_wait, stall_timeout_seconds)
                next_wait += 1
            perform_defrag(orchestrator, frame_number, defrag_log)
    while next_wait < stop_frame:
        orchestrator._wait_frame(next_wait, stall_timeout_seconds)
        next_wait += 1
    orchestrator._frame_count = max(stop_frame, start_frame)


# =============================================================================
# Invariants
# =============================================================================

def compute_invariants(bookkeeping: dict, initial_total: int, statuses: list[dict],
                       defrag_log: list, worker_stamp_errors: dict,
                       previous_statistics: dict) -> dict:
    """Every global_status key starting with 'overflow_' is a must-be-zero
    counter (the field list is NOT hard-coded: v6 adds counters). Each counter
    is the per-sim maximum over every observation so far (defrag boundaries +
    this horizon), summed over sims — correct whether the solver accumulates or
    resets it. WATCHED_STATUS_KEYS (v6 far_migration_count) are recorded the
    same way and raise a warning when non-zero, without changing 'valid'."""
    observations = [entry["global_status"] for entry in defrag_log] + [statuses]
    overflow_maximum: dict = {}
    watched_maximum: dict = {}
    stamp_maximum: dict = {}
    for observation in observations:
        for slab_index, status in enumerate(observation):
            for key, value in status.items():
                if key.startswith("overflow_"):
                    target = overflow_maximum
                elif key in WATCHED_STATUS_KEYS:
                    target = watched_maximum
                else:
                    continue
                slot = target.setdefault(key, {})
                slot[slab_index] = max(slot.get(slab_index, 0), int(value))
            stamp_maximum[slab_index] = max(stamp_maximum.get(slab_index, 0),
                                            int(status.get("stamp_error_count", 0)))
    invariants = {
        "drift": int(bookkeeping["alive_total"] - initial_total),
        "missing_ids": int(bookkeeping["missing_ids"]),
        "duplicate_ids": int(bookkeeping["duplicate_ids"]),
        "invalid_ids": int(bookkeeping["invalid_ids"]),
        "foreign_extension_values": int(bookkeeping["foreign_extension_values"]),
        "previous_duplicate_ids": int(previous_statistics["duplicate_ids"]),
        "previous_invalid_ids": int(previous_statistics["invalid_ids"]),
        "stamp_errors_gpu": int(sum(stamp_maximum.values())),
        "stamp_errors_host": int(sum(worker_stamp_errors.values())),
    }
    for key in sorted(overflow_maximum):
        invariants[key] = int(sum(overflow_maximum[key].values()))
    must_be_zero = list(invariants)
    invariants["overflow_total"] = int(sum(invariants[key] for key in overflow_maximum))
    warnings = []
    for key in sorted(watched_maximum):
        invariants[key] = int(sum(watched_maximum[key].values()))
        if invariants[key]:
            warnings.append(f"{key}={invariants[key]}")
    invariants["warnings"] = warnings
    invariants["valid"] = all(invariants[key] == 0 for key in must_be_zero)
    return invariants


# =============================================================================
# Case description
# =============================================================================

def material_names_from_case(case_path: pathlib.Path) -> list:
    """Material group names in the loader's order (first use in geometry)."""
    try:
        import yaml
        case_data = yaml.safe_load(case_path.read_text(encoding="utf-8"))
        names = []
        for entry in case_data["geometry"]["particles"]:
            if entry["material"] not in names:
                names.append(entry["material"])
        return names
    except Exception:  # descriptive only
        return []


def describe_case(arguments, global_case, chain, indices_per_slab, weights,
                  device_map, defrag_cadence, horizons) -> dict:
    case_path = pathlib.Path(arguments.case)
    grid = global_case.grid
    return {
        "dump_format_version": DUMP_FORMAT_VERSION,
        "tool": "experiment/seam_audit/dump_state.py",
        "run_name": arguments.run_name,
        "version": arguments.version,
        "case": str(arguments.case),
        "case_resolved": str(case_path.resolve()),
        "case_name": case_path.resolve().parent.name,
        "slabs": arguments.slabs,
        "weights": weights,
        "device_map": [device_map[index % len(device_map)] for index in range(arguments.slabs)],
        "depth": arguments.depth,
        "pool_safety": arguments.pool_safety,
        "sync_scheme": arguments.sync_scheme,
        "defrag_cadence": defrag_cadence,
        "horizons": horizons,
        "cuts": [int(cut) for cut in chain.cuts],
        "smoothing_length": float(global_case.physics.smoothing_length),
        "origin_x": float(grid.origin_x),
        "origin_y": float(grid.origin_y),
        "origin_z": float(grid.origin_z),
        "grid_nx": int(grid.grid_dimension_x),
        "grid_dimension": [int(grid.grid_dimension_x), int(grid.grid_dimension_y),
                           int(grid.grid_dimension_z)],
        "dimension": int(global_case.physics.dimension),
        "own_columns": [[int(geometry.own_global_first_column),
                         int(geometry.own_global_last_column)] for geometry in chain.geometry],
        "own_pool_sizes": [int(slab.capacities.own_pool_size) for slab in chain.slabs],
        "ghost_pool_sizes": [[int(slab.capacities.leading_ghost_pool_size),
                              int(slab.capacities.trailing_ghost_pool_size)]
                             for slab in chain.slabs],
        "initial_particle_counts": [int(indices.size) for indices in indices_per_slab],
        "initial_total": int(global_case.initial.positions.shape[0]),
        "material_kinds": [int(material.kind) for material in global_case.materials],
        "material_names": material_names_from_case(case_path),
        "id_encoding": "extension_fields.z = id // 2**20, extension_fields.w = id % 2**20 "
                       "(id = row index in the degenerate global case)",
        "environment": recorded_environment(),
        "validation_layer": bool(arguments.validation),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }


# =============================================================================
# Run
# =============================================================================

def prepare(arguments, include_gpu_modules: bool):
    """CPU part shared by --dry-run and the real run."""
    horizons = parse_horizons(arguments.horizons)
    device_map = parse_device_map(arguments.device_map)
    if arguments.slabs < 1:
        raise ValueError("--slabs must be >= 1")
    weights = parse_weights(arguments.weights, arguments.slabs)
    if arguments.run_name is None:
        arguments.run_name = f"{arguments.version}_K{arguments.slabs}"
    solver = load_solver(arguments.version, include_gpu_modules=include_gpu_modules)
    print(f"{LOG_PREFIX} {arguments.run_name}: loading {arguments.case} "
          f"({arguments.version}, K={arguments.slabs})", flush=True)
    global_case = solver.load_case(arguments.case)
    pool_safety = None if arguments.pool_safety == 0 else arguments.pool_safety
    chain = solver.compute_chain_partition(global_case, weights, pool_safety)
    indices_per_slab = slab_global_indices(global_case, chain)
    initial_total = int(global_case.initial.positions.shape[0])
    check_encoding_round_trip(initial_total)
    if arguments.defrag_cadence is None:
        defrag_cadence = int(global_case.numerics.defrag_cadence)
    else:
        defrag_cadence = int(arguments.defrag_cadence)
    if defrag_cadence <= 0:
        defrag_cadence = DISABLED_DEFRAG_CADENCE
    description = describe_case(arguments, global_case, chain, indices_per_slab, weights,
                                device_map, defrag_cadence, horizons)
    for slab_index, (geometry, indices) in enumerate(zip(chain.geometry, indices_per_slab)):
        print(f"{LOG_PREFIX}   slab {slab_index}: own columns "
              f"[{geometry.own_global_first_column}, {geometry.own_global_last_column}] "
              f"particles {indices.size:,} pool {chain.slabs[slab_index].capacities.own_pool_size:,} "
              f"device {description['device_map'][slab_index]}; global-id mask "
              f"bit-identical to the partition", flush=True)
    print(f"{LOG_PREFIX}   cuts {description['cuts']}  total {initial_total:,}  "
          f"h={description['smoothing_length']}  origin_x={description['origin_x']}  "
          f"nx={description['grid_nx']}  dimension={description['dimension']}  "
          f"defrag_cadence={defrag_cadence}  horizons={horizons}", flush=True)
    return solver, global_case, chain, indices_per_slab, horizons, device_map, \
        defrag_cadence, description


def verify_bootstrap_ids(sims, indices_per_slab) -> None:
    """Right after bootstrap (no migration yet) every slab must own exactly
    the ids it was given — proves the injection reached the GPU intact."""
    for slab_index, (sim, expected) in enumerate(zip(sims, indices_per_slab)):
        arrays, alive = read_own_slots(sim, OWNERSHIP_BUFFER_NAMES)
        ids, foreign = decode_global_ids(arrays["extension_fields"][alive])
        if foreign.any() or not np.array_equal(np.sort(ids), np.sort(expected)):
            raise RuntimeError(
                f"slab {slab_index}: ids after bootstrap do not match the injected ids "
                f"(alive {int(alive.sum())}, expected {expected.size}, "
                f"foreign rows {int(foreign.sum())})")
    print(f"{LOG_PREFIX} bootstrap id check passed on {len(sims)} sim(s)", flush=True)


def run(arguments, summary: dict) -> int:
    if arguments.dry_run:
        prepare(arguments, include_gpu_modules=False)
        summary.update({"run_name": arguments.run_name, "dry_run": True, "valid": True})
        return EXIT_VALID

    (solver, global_case, chain, indices_per_slab, horizons, device_map,
     defrag_cadence, description) = prepare(arguments, include_gpu_modules=True)
    summary["run_name"] = arguments.run_name
    if os.environ.get(solver.env_prefix + "PER_SIM_PIPELINE", "0") not in ("", "0"):
        print(f"{LOG_PREFIX} WARNING: {solver.env_prefix}PER_SIM_PIPELINE is set; "
              f"run_frames always uses the default pipelined loop", file=sys.stderr)
    out_directory = pathlib.Path(arguments.out_dir)
    out_directory.mkdir(parents=True, exist_ok=True)
    initial_total = description["initial_total"]
    dimension = description["dimension"]
    smoothing_length = float(global_case.physics.smoothing_length)
    window_horizon_set = {int(item) for item in str(arguments.window_horizons).split(",")
                          if item.strip()}
    cut_lines = audit_cut_lines(solver, global_case, chain, arguments) if arguments.window > 0 else None
    if cut_lines is not None:
        print(f"{LOG_PREFIX} crossing window {arguments.window} frames over cut lines "
              f"{[round(float(line), 6) for line in cut_lines]}", flush=True)
    install_global_id_injection(solver.Simulator)

    # The Vulkan objects are torn down only after a successful run, on purpose
    # without try/finally or a 'with' on the orchestrator. After a failure (a
    # frame stall, a dead transport worker, Ctrl+C) frames can still be in
    # flight whose batches wait on timeline values that will never be signalled:
    # the upload waits for worker_done(n) of a stuck or dead worker, phase C for
    # upload_done(n). sim.destroy() starts with vkDeviceWaitIdle and would block
    # forever, and the orchestrator's worker.stop() joins threads stuck in
    # vkWaitSemaphores, so the RESULT line would never be printed. Every
    # exception therefore goes straight to main(), which reports it and leaves
    # through os._exit; process exit releases the devices, as the matrix
    # runner's kill does.
    run_start = time.perf_counter()
    contexts, sims = [], []
    for slab_index in range(arguments.slabs):
        device_index = device_map[slab_index % len(device_map)]
        contexts.append(solver.Context.create(
            device_index=device_index,
            enable_validation=arguments.validation,
            application_name=f"seam_audit_{arguments.run_name}_s{slab_index}"))
        sim = solver.Simulator(contexts[-1], chain.slabs[slab_index],
                               sync_scheme=arguments.sync_scheme)
        sims.append(sim)
        if "extension_fields" not in sim.buffers:
            raise RuntimeError("solver has no extension_fields buffer; the global id "
                               "cannot be carried")
        setattr(sim, GLOBAL_ID_ATTRIBUTE, indices_per_slab[slab_index])

    orchestrator = solver.Orchestrator(sims, defrag_cadence=defrag_cadence)
    bootstrap_start = time.perf_counter()
    orchestrator.bootstrap_all()
    bootstrap_seconds = time.perf_counter() - bootstrap_start
    verify_bootstrap_ids(sims, indices_per_slab)

    defrag_log: list = []
    frames_seconds = 0.0
    current_frame = 0
    horizon_results = []
    position_readers = None
    for horizon_index, horizon in enumerate(horizons):
        is_final_horizon = horizon_index == len(horizons) - 1
        # Crossing window (--window W > 0): the last W frames before the
        # horizon run one at a time; after each, the cheap readback finds the
        # particles that crossed an audit cut line, and only then the full
        # state is read and the crossing neighbourhoods captured. W = 0 keeps
        # the original sequence (pipelined to N-1, ownership, frame N-1).
        window_enabled = arguments.window > 0 and (
            not window_horizon_set or horizon in window_horizon_set)
        capture_requests = (load_capture_requests(arguments.capture_requests_dir, horizon)
                            if window_enabled else None)
        window_frames = (min(arguments.window, horizon - current_frame)
                         if window_enabled else 1)
        window_start = horizon - window_frames
        frames_start = time.perf_counter()
        run_frames(orchestrator, current_frame, window_start, arguments.depth,
                   defrag_cadence, defrag_log,
                   stall_timeout_seconds=arguments.stall_timeout)
        frames_seconds += time.perf_counter() - frames_start

        recorder = None
        requested_records = []
        window_seconds = 0.0
        if window_enabled and capture_requests is None:
            if position_readers is None:
                position_readers = [OwnPositionReader(sim) for sim in sims]
            recorder = CrossingWindowRecorder(cut_lines, smoothing_length)
            window_ids, window_x = collect_positions(position_readers, initial_total)
            recorder.begin(initial_total, window_ids, window_x)
        unknown_owner = np.full(initial_total, UNKNOWN_SLAB, dtype=np.uint8)
        previous_owner, previous_statistics = None, None
        for window_frame in range(window_start, horizon):
            is_dump_frame = window_frame + 1 == horizon
            if is_dump_frame:
                previous_owner, previous_statistics = collect_ownership(sims, initial_total)
            frames_start = time.perf_counter()
            run_frames(orchestrator, window_frame, window_frame + 1, arguments.depth,
                       defrag_cadence, defrag_log, defrag_at_stop=not is_dump_frame,
                       stall_timeout_seconds=arguments.stall_timeout)
            frames_seconds += time.perf_counter() - frames_start
            if capture_requests is not None and not is_dump_frame:
                label = window_frame + 1
                if label in capture_requests:
                    capture_start = time.perf_counter()
                    frame_state, _ = collect_state(sims, dimension, initial_total, unknown_owner)
                    requested_records.append(requested_rows(frame_state, capture_requests[label], label))
                    del frame_state
                    window_seconds += time.perf_counter() - capture_start
            if recorder is not None and not is_dump_frame:
                capture_start = time.perf_counter()
                window_ids, window_x = collect_positions(position_readers, initial_total)
                crossing_ids = recorder.crossings(initial_total, window_ids, window_x)
                if crossing_ids.size:
                    frame_state, _ = collect_state(sims, dimension, initial_total,
                                                   unknown_owner)
                    recorder.add_frame(window_frame + 1, initial_total, frame_state,
                                       crossing_ids)
                    del frame_state
                else:
                    recorder.advance(initial_total, window_ids, window_x)
                window_seconds += time.perf_counter() - capture_start

        state, bookkeeping = collect_state(sims, dimension, initial_total,
                                           previous_owner)
        statuses, healths = read_statuses(sims)
        worker_stamp_errors = {
            worker.label: int(getattr(worker, "stamp_error_count", 0))
            for worker in getattr(orchestrator, "workers", ())}
        invariants = compute_invariants(bookkeeping, initial_total, statuses,
                                        defrag_log, worker_stamp_errors,
                                        previous_statistics)
        window_summary = None
        if capture_requests is not None:
            if horizon in capture_requests:
                requested_records.append(requested_rows(state, capture_requests[horizon], horizon))
            requested_records = [record for record in requested_records if record["id"].size]
            window_arrays = ({key: np.concatenate([record[key] for record in requested_records])
                              for key in requested_records[0]} if requested_records else
                             {"frame": np.zeros(0, dtype=np.int32)})
            window_path = window_file_path(out_directory, arguments.run_name, horizon)
            save_npz_atomically(window_path, window_arrays)
            requested_total = int(sum(ids.size for ids in capture_requests.values()))
            captured_total = int(window_arrays["frame"].size)
            window_summary = {"mode": "requested", "window_frames": int(window_frames),
                              "requested_frames": len(capture_requests),
                              "requested_rows": requested_total, "captured_rows": captured_total,
                              "missing_rows": requested_total - captured_total,
                              "npz": window_path.name, "capture_seconds": window_seconds,
                              "first_frame": int(window_start + 1), "last_frame": int(horizon)}
            print(f"{LOG_PREFIX} N={horizon} window (requested): {len(capture_requests)} frames, "
                  f"{captured_total}/{requested_total} rows captured ({window_seconds:.1f} s)",
                  flush=True)
        if recorder is not None:
            crossing_ids = recorder.crossings(initial_total, state["id"].astype(np.int64),
                                              state["position"][:, 0].astype(np.float64))
            if crossing_ids.size:
                recorder.add_frame(horizon, initial_total, state, crossing_ids)
            else:
                recorder.frame_count += 1
            window_arrays = recorder.arrays()
            window_path = window_file_path(out_directory, arguments.run_name, horizon)
            save_npz_atomically(window_path, window_arrays if window_arrays
                                else {"frame": np.zeros(0, dtype=np.int32)})
            window_summary = dict(recorder.summary(), npz=window_path.name,
                                  capture_seconds=window_seconds,
                                  first_frame=int(window_start + 1),
                                  last_frame=int(horizon))
            print(f"{LOG_PREFIX} N={horizon} window: {window_summary['window_frames']} frames, "
                  f"{window_summary['frames_with_crossings']} with crossings, "
                  f"{window_summary['crossing_events']} crossing particles, "
                  f"{window_summary['captured_rows']} captured rows "
                  f"({window_seconds:.1f} s)", flush=True)
        paths = dump_file_paths(out_directory, arguments.run_name, horizon)
        save_npz_atomically(paths["npz"], state)
        if arguments.keep_previous:
            known = np.flatnonzero(previous_owner != UNKNOWN_SLAB)
            save_npz_atomically(paths["previous"], {
                "id": known.astype(np.uint32), "slab": previous_owner[known]})
        document = dict(description)
        document.update({
            "horizon": int(horizon),
            "frame_count": int(horizon),
            "npz": paths["npz"].name,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "wall_time_s": time.perf_counter() - run_start,
            "bootstrap_time_s": bootstrap_seconds,
            "frames_time_s": frames_seconds,
            "global_status": statuses,
            "pool_health": healths,
            "worker_stamp_errors": worker_stamp_errors,
            "defrag_reports": defrag_log,
            "ownership_previous": {"frame": int(horizon - 1), **previous_statistics},
            "bookkeeping": bookkeeping,
            "invariants": invariants,
            "window": window_summary,
        })
        save_json_atomically(paths["json"], document)
        del state
        horizon_results.append({
            "horizon": int(horizon),
            "valid": invariants["valid"],
            "drift": invariants["drift"],
            "missing_ids": invariants["missing_ids"],
            "duplicate_ids": invariants["duplicate_ids"],
            "stamp_errors_gpu": invariants["stamp_errors_gpu"],
            "stamp_errors_host": invariants["stamp_errors_host"],
            "overflow_total": invariants["overflow_total"],
            "crossed_last_step": bookkeeping["crossed_last_step_count"],
            "alive": bookkeeping["alive_total"],
            "warnings": invariants["warnings"],
            "npz": str(paths["npz"]),
            "json": str(paths["json"]),
        })
        print(f"{LOG_PREFIX} N={horizon}: alive {bookkeeping['alive_total']:,} "
              f"drift {invariants['drift']} missing {invariants['missing_ids']} "
              f"duplicates {invariants['duplicate_ids']} crossed "
              f"{bookkeeping['crossed_last_step_count']} "
              f"{bookkeeping['crossings_by_direction']} stamps "
              f"{invariants['stamp_errors_gpu']}/{invariants['stamp_errors_host']} "
              f"overflow {invariants['overflow_total']} -> "
              f"{'VALID' if invariants['valid'] else '*** INVALID ***'} "
              f"({paths['npz'].name})"
              + (f" WARNING {invariants['warnings']}" if invariants["warnings"] else ""),
              flush=True)

        if (not is_final_horizon) and horizon % defrag_cadence == 0:
            # the boundary run_frames left to us: same place in the GPU
            # sequence as in run_pipelined, after the read-only dump
            perform_defrag(orchestrator, horizon, defrag_log)
        current_frame = horizon

    # Every submitted frame has been waited for: the workers are idle in their
    # queues and every queue is empty, so this teardown cannot block.
    orchestrator.destroy()
    for reader in position_readers or ():
        reader.destroy()
    for sim in sims:
        sim.destroy()
    for context in contexts:
        context.destroy()

    all_valid = all(result["valid"] for result in horizon_results)
    summary.update({
        "valid": all_valid,
        "wall_time_s": time.perf_counter() - run_start,
        "per_horizon": horizon_results,
    })
    return EXIT_VALID if all_valid else EXIT_INVALID


def main(argument_list=None) -> int:
    enable_audit_transport()
    arguments = parse_arguments(argument_list)
    summary = {
        "run_name": arguments.run_name,
        "version": arguments.version,
        "case": arguments.case,
        "slabs": arguments.slabs,
        "device_map": arguments.device_map,
        "horizons": arguments.horizons,
        "out_dir": arguments.out_dir,
        "dry_run": bool(arguments.dry_run),
        "valid": False,
        "error": None,
    }
    try:
        exit_code = run(arguments, summary)
    except (Exception, KeyboardInterrupt) as error:
        # Report, then leave at once through os._exit: no Vulkan teardown and
        # no join of the solver's non-daemon worker threads, either of which
        # can block forever after a failure (see run()).
        traceback.print_exc()
        summary["valid"] = False
        summary["error"] = f"{type(error).__name__}: {error}"
        print(RESULT_PREFIX + json.dumps(summary, default=json_default), flush=True)
        sys.stderr.flush()
        os._exit(EXIT_ERROR)
    print(RESULT_PREFIX + json.dumps(summary, default=json_default), flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
