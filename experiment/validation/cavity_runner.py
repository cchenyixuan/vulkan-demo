"""cavity_runner.py - one segment of a lid-driven cavity validation run with the v6 solver.

Runs the case from rest (or resumes from the newest checkpoint) with ChainOrchestratorV6 (K = 1 or 2,
production depth-2 loop) and does all I/O in the drained on_defrag hook:
  every sample (default: the defrag boundary nearest 0.2 time units):
      cached readback of every alive own particle (position, velocity / mass, density, material);
      checks: alive total = expected (drift 0), every GPU overflow_* counter, far_migration and the GPU
      and host frame-stamp error counts = 0 (any violation aborts the run);
      kinetic energy of the fluid, max |v|, mean fluid velocity, fluid density range;
      MLS and Shepard velocity at the 28 + 28 Marchi 2021 points in both frames (wall-row and fluid-box,
      see cavity_sampling) and on two 1001-point dense centre lines;
      -> samples.jsonl (scalars) + samples/s<step>.npz (profiles);
  every window (default 10 time units): steady-state test on the window means;
  every time unit from t >= t_end - average_span on: light snapshot of every particle (for the stream
      function and the vortex centre, offline) -> snapshots/t<step>.npz;
  every checkpoint interval (default 10 time units): full restart state (the nine step-boundary fields,
      stored density) -> checkpoints/c<step>.npz (newest two kept), for --resume.
The run stops at t_stop = max(t_end, t_steady + average_span) (t_steady = end of the first window whose
mean profiles and mean kinetic energy changed by less than --steady-tol relative to the previous window),
or at --t-max. The release switches arrive through the environment (cavity_campaign.py builds it); this
runner refuses to start if they do not match --expect. A fresh run keeps a copy of its case.yaml in the run
directory: the analysis reads the run's numerics (xi, epsilon_squared_factor) from that copy, not from cases/.

    .venv/Scripts/python.exe -m experiment.validation.cavity_runner --case cases/lid_driven_cavity_2d_n250/case.yaml \
        --run-dir logs/validation/cavity_re1000/n250_k2_float32 --slabs 2 --device-map 0,1 --expect release
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import time

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.validation import cavity_reference, cavity_sampling as sampling  # noqa: E402

RELEASE_ENVIRONMENT = {
    "V6_KEEP_DEPARTED": "1", "V6_GHOST_LAYERS": "2", "V6_LEAN_TRANSPORT": "1", "V6_COMPACT_GHOST_LISTS": "1",
    "V6_PACKED_REPLICAS": "1", "V6_BAND_SLOT_LANES": "64", "V6_GHOST_POOL_FACTOR": "0.29",
    "V6_MIGRANT_POOL_FACTOR": "0.05", "V6_DEPARTED_FACE_FRACTION": "0.8", "V6_INIT_SEAM_CLAMP": "1",
    "V6_WORKER_COUNT_AWARE": "1", "V6_SPLIT_TRANSFER_QUEUES": "1",
}
EXPECTED = {"release": dict(RELEASE_ENVIRONMENT),
            "release_delta": dict(RELEASE_ENVIRONMENT, V6_DELTA_DENSITY="1")}
DENSE_POINTS = 1001
FRAMES = ("wall", "fluid")


class StopRun(Exception):
    pass


class InvariantViolation(RuntimeError):
    """drift, an overflow_* counter, far_migration or a frame-stamp error: the run is invalid (exit code 3;
    the campaign does not resume it, a resume from an earlier checkpoint would hide the violation)."""


INVARIANT_EXIT_CODE = 3
STEP_FILE = re.compile(r"[stc](\d{10})\.npz")


def step_of(path: pathlib.Path) -> int | None:
    match = STEP_FILE.fullmatch(path.name)
    return int(match.group(1)) if match else None


def check_environment(expect: str) -> dict:
    """The V6_* environment must be exactly the expected set (read before any v6 import)."""
    wanted = EXPECTED[expect]
    actual = {key: value for key, value in os.environ.items() if key.startswith("V6_")}
    if actual != wanted:
        missing = {key: value for key, value in wanted.items() if actual.get(key) != value}
        extra = {key: value for key, value in actual.items() if key not in wanted}
        sys.exit(f"[cavity] environment does not match --expect {expect}: wrong/missing {missing}, extra {extra}")
    if os.environ.get("VK_LOADER_LAYERS_DISABLE", "") != "VK_LAYER_KHRONOS_validation":
        sys.exit("[cavity] set VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation")
    return actual


class CachedStateReader:
    """Persistent host-cached staging copy of one sim's own range of the four light fields
    (position_voxel_id, velocity_mass, density_pressure, material). The pre-recorded command buffer starts
    with a full memory barrier (shader / transfer writes -> transfer reads) and is replayed with
    submit_and_wait between drained frames only."""

    FIELDS = (("position_voxel_id", 16), ("velocity_mass", 16), ("density_pressure", 8), ("material", 4))

    def __init__(self, sim) -> None:
        from vulkan import (VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_STRUCTURE_TYPE_MEMORY_BARRIER, VkBufferCopy,
                            VkCommandBufferBeginInfo, VkMemoryBarrier, vkBeginCommandBuffer, vkCmdCopyBuffer,
                            vkCmdPipelineBarrier, vkEndCommandBuffer, vkMapMemory)
        self.sim = sim
        own_first = sim.own_first_pid()
        self.count = sim.case.capacities.own_pool_size
        self.offsets = {}
        byte_count = 0
        for name, stride in self.FIELDS:
            self.offsets[name] = byte_count
            byte_count += stride * self.count
        self.staging = sim._allocate_buffer(
            byte_count, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
        mapped = vkMapMemory(sim.ctx.device, self.staging.memory, 0, byte_count, 0)
        self.view = np.frombuffer(mapped, dtype=np.uint8, count=byte_count)
        self.command_buffer = sim._allocate_oneshot_cmd()
        vkBeginCommandBuffer(self.command_buffer, VkCommandBufferBeginInfo())
        barrier = VkMemoryBarrier(sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER, srcAccessMask=VK_ACCESS_MEMORY_WRITE_BIT,
                                  dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT)
        vkCmdPipelineBarrier(self.command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, [barrier], 0, None, 0, None)
        for name, stride in self.FIELDS:
            vkCmdCopyBuffer(self.command_buffer, sim.buffers[name].handle, self.staging.handle, 1, [
                VkBufferCopy(srcOffset=stride * own_first, dstOffset=self.offsets[name], size=stride * self.count)])
        vkEndCommandBuffer(self.command_buffer)

    def read(self) -> dict:
        """Alive own particles: positions (n, 2), velocities (n, 2), mass (n,), density (n,) absolute
        (float64; the stored density plus the sim's stored_density_offset), material (n,)."""
        self.sim.ctx.submit_and_wait(self.command_buffer)
        count = self.count
        position = np.frombuffer(self.view, np.float32, 4 * count, self.offsets["position_voxel_id"]).reshape(count, 4)
        velocity = np.frombuffer(self.view, np.float32, 4 * count, self.offsets["velocity_mass"]).reshape(count, 4)
        density = np.frombuffer(self.view, np.float32, 2 * count, self.offsets["density_pressure"]).reshape(count, 2)
        material = np.frombuffer(self.view, np.uint32, count, self.offsets["material"])
        alive = (velocity[:, 3] > 0) & (position[:, 3] > 0.5)
        offset = float(self.sim.stored_density_offset())
        return {"positions": position[alive, :2].astype(np.float64),
                "velocities": velocity[alive, :2].astype(np.float64),
                "mass": velocity[alive, 3].astype(np.float64),
                "density": density[alive, 0].astype(np.float64) + offset,
                "material": material[alive].copy()}

    def destroy(self) -> None:
        from vulkan import vkDestroyBuffer, vkFreeCommandBuffers, vkFreeMemory, vkUnmapMemory
        device = self.sim.ctx.device
        vkFreeCommandBuffers(device, self.sim.ctx.command_pool, 1, [self.command_buffer])
        vkUnmapMemory(device, self.staging.memory)
        vkDestroyBuffer(device, self.staging.handle, None)
        vkFreeMemory(device, self.staging.memory, None)


def read_restart_state(sim) -> dict:
    """Alive own rows of the nine restart fields, density stored as on the GPU (restart_init converts)."""
    layout = sim.RESTART_FIELD_LAYOUT
    raw = sim.readback_buffers_batch(list(layout), density="stored")
    capacities = sim.case.capacities
    pool_capacity = capacities.total_pool_capacity()
    own_first = sim.own_first_pid()
    own_stop = own_first + capacities.own_pool_size
    rows = {}
    for name, (component_count, element_type) in layout.items():
        flat = np.frombuffer(raw[name], dtype=element_type)[:pool_capacity * component_count]
        if component_count > 1:
            flat = flat.reshape(pool_capacity, component_count)
        rows[name] = np.array(flat[own_first:own_stop])
    alive = (rows["velocity_mass"][:, 3] > 0) & (rows["position_voxel_id"][:, 3] > 0.5)
    return {name: values[alive] for name, values in rows.items()}


def atomic_savez(path: pathlib.Path, **arrays) -> None:
    """Write to <name>.tmp (fsynced), then rename; a crash leaves at most a .tmp file, never a torn .npz."""
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def safe_unlink(path: pathlib.Path) -> bool:
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return True
    except OSError as error:
        print(f"[cavity] could not delete {path} ({error!r}); retrying later", flush=True)
        return False


def file_hash(paths: list[pathlib.Path]) -> str:
    digest = hashlib.sha256()
    for item in sorted(paths):
        digest.update(str(item.relative_to(_REPO_ROOT)).replace("\\", "/").encode())
        digest.update(item.read_bytes())
    return digest.hexdigest()[:16]


def code_hashes(case_path: pathlib.Path) -> dict:
    """physics = the v6 solver (python + SPIR-V) + the case files + the material library; sampling = the
    sampling / reference code and the reference CSV. A resume refuses to continue under different hashes."""
    case_directory = case_path.parent
    physics = (list((_REPO_ROOT / "experiment" / "v6" / "utils").glob("*.py"))
               + list((_REPO_ROOT / "experiment" / "v6" / "shaders" / "spv").glob("*.spv"))
               + [case_path] + list(case_directory.glob("*.obj")) + [_REPO_ROOT / "materials" / "standard.yaml"])
    validation = _REPO_ROOT / "experiment" / "validation"
    sampling_files = [validation / "cavity_sampling.py", validation / "cavity_reference.py",
                      _REPO_ROOT / "docs" / "validation" / "data" / "marchi2021_re1000.csv"]
    return {"physics": file_hash(physics), "sampling": file_hash(sampling_files),
            "runner": file_hash([validation / "cavity_runner.py"])}


def device_uuid(ctx) -> str:
    # core in Vulkan 1.1 (the v6 instance is 1.3); python-vulkan's vkGetInstanceProcAddr only resolves extensions
    from vulkan import VkPhysicalDeviceIDProperties, VkPhysicalDeviceProperties2, vkGetPhysicalDeviceProperties2
    id_properties = VkPhysicalDeviceIDProperties()
    vkGetPhysicalDeviceProperties2(ctx.physical_device, VkPhysicalDeviceProperties2(pNext=id_properties))
    return bytes(id_properties.deviceUUID).hex()


def process_alive(pid: int) -> bool:
    import ctypes
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(handle)
    return code.value == 259                                                 # STILL_ACTIVE


def read_jsonl(path: pathlib.Path) -> list[dict]:
    """Rows of a JSON-lines file; a torn last line (crash during a write) is skipped."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[cavity] skipped an unreadable line in {path.name}", flush=True)
    return rows


def atomic_write_text(path: pathlib.Path, text_value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text_value, encoding="utf-8")
    os.replace(temporary, path)


def git_state() -> dict:
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                          cwd=_REPO_ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "experiment/v6", "experiment/validation"],
                           capture_output=True, text=True, cwd=_REPO_ROOT).stdout.strip()
    return {"head": head, "dirty": bool(dirty)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--slabs", type=int, choices=(1, 2), required=True)
    parser.add_argument("--device-map", required=True, help="e.g. 0,1 (K = 2) or 1 (K = 1)")
    parser.add_argument("--expect", choices=sorted(EXPECTED), required=True)
    parser.add_argument("--t-end", type=float, default=100.0, help="earliest stop time")
    parser.add_argument("--t-max", type=float, default=200.0, help="latest stop time (steady or not)")
    parser.add_argument("--window", type=float, default=10.0, help="steady-state window length")
    parser.add_argument("--steady-tol", type=float, default=1.0e-3)
    parser.add_argument("--average-span", type=float, default=20.0, help="averaging window after steady state")
    parser.add_argument("--sample-time", type=float, default=0.2)
    parser.add_argument("--snapshot-time", type=float, default=1.0)
    parser.add_argument("--checkpoint-time", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None, help="stop after this many steps (calibration)")
    parser.add_argument("--pool-safety", type=float, default=1.2)
    parser.add_argument("--require-uuid", default=None,
                        help="comma-separated Vulkan deviceUUIDs the slabs' GPUs must have (e.g. the headless 5090)")
    arguments = parser.parse_args()
    environment = check_environment(arguments.expect)
    if arguments.t_end + arguments.average_span > arguments.t_max:
        sys.exit(f"[cavity] --t-end {arguments.t_end} + --average-span {arguments.average_span} exceeds --t-max {arguments.t_max}")

    import ctypes
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)      # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    except Exception:
        pass

    from experiment.v6.utils.case_loader_v6 import load_case_v6
    from experiment.v6.utils.case_v6 import KIND_FLUID
    from experiment.v6.utils.orchestrator_v6 import ChainOrchestratorV6
    from experiment.v6.utils.partition_v6 import compute_chain_partition, restart_slab_rows
    from experiment.v6.utils.simulator_v6 import SphSimulatorV6
    from experiment.v6.utils.vulkan_context_v6 import VulkanContextV6

    run_dir = pathlib.Path(arguments.run_dir).resolve()
    for sub in ("samples", "snapshots", "checkpoints"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
        for leftover in list((run_dir / sub).glob("*.tmp")) + list((run_dir / sub).glob("*.tmp.npz")):
            safe_unlink(leftover)
    lock_path = run_dir / "runner.pid"
    if lock_path.exists():
        try:
            other = int(lock_path.read_text(encoding="utf-8").split()[0])
        except (ValueError, IndexError):
            other = 0
        if other and other != os.getpid() and process_alive(other):
            sys.exit(f"[cavity] another runner (pid {other}) is using {run_dir}")
    lock_path.write_text(f"{os.getpid()} {time.time():.0f}", encoding="utf-8")
    if not arguments.resume and ((run_dir / "samples.jsonl").exists()
                                 or (run_dir / "checkpoints" / "manifest.jsonl").exists()):
        sys.exit(f"[cavity] {run_dir} already holds a run; pass --resume or use a new --run-dir")
    global_case = load_case_v6(arguments.case)
    if global_case.numerics.wall_boundary == "adami" and arguments.slabs != 1:
        sys.exit(f"[cavity] {arguments.case}: wall_boundary adami supports one GPU (K = 1) only in this release; "
                 "run it with --slabs 1")
    expected_total = int(global_case.initial.positions.shape[0])
    dt = float(global_case.physics.timestep)
    support = float(global_case.physics.smoothing_length)
    radii = [float(material.radius) for material in global_case.materials if float(material.radius) > 0]
    spacing = 2.0 * min(radii)
    fluid_groups = np.array([index for index, material in enumerate(global_case.materials)
                             if material.kind == KIND_FLUID], dtype=np.uint32)
    fluid_viscosity = [float(material.viscosity) for material in global_case.materials if material.kind == KIND_FLUID]
    lid_speed = [float(np.linalg.norm(material.initial_velocity)) for material in global_case.materials
                 if np.linalg.norm(material.initial_velocity) > 0]
    if len(set(fluid_viscosity)) != 1 or len(set(lid_speed)) != 1:
        sys.exit(f"[cavity] expected one fluid viscosity and one lid speed: {fluid_viscosity}, {lid_speed}")
    reynolds = lid_speed[0] * 1.0 / fluid_viscosity[0]
    if abs(reynolds - 1000.0) > 1e-6:
        sys.exit(f"[cavity] nominal Re = U L / nu = {reynolds} (expected 1000; L = 1 is the fluid box)")
    cadence = int(global_case.numerics.defrag_cadence)

    def snap(time_value: float) -> int:
        """Steps of the defrag boundary nearest time_value (at least one cadence)."""
        return max(cadence, int(round(time_value / dt / cadence)) * cadence)

    sample_steps = snap(arguments.sample_time)
    snapshot_steps = snap(arguments.snapshot_time)
    # checkpoints coincide with samples, so every checkpoint follows a full invariant check
    checkpoint_steps = sample_steps * max(1, int(round(arguments.checkpoint_time / (sample_steps * dt))))
    # the window is a whole number of sample intervals, so every window end is a sample
    window_steps = sample_steps * max(1, int(round(arguments.window / (sample_steps * dt))))
    max_total_steps = int(math.ceil(arguments.t_max / dt / cadence)) * cadence

    reference = cavity_reference.marchi2021()
    frames = {name: sampling.Frame.make(name, spacing) for name in FRAMES}
    dense_reference = np.linspace(0.0, 1.0, DENSE_POINTS)
    point_sets = {}
    for frame_name, frame in frames.items():
        point_sets[f"u_marchi_{frame_name}"] = ("u", sampling.centerline_points(frame, reference["u"][0], "u"))
        point_sets[f"v_marchi_{frame_name}"] = ("v", sampling.centerline_points(frame, reference["v"][0], "v"))
    point_sets["u_dense"] = ("u", sampling.centerline_points(frames["wall"], dense_reference, "u"))
    point_sets["v_dense"] = ("v", sampling.centerline_points(frames["wall"], dense_reference, "v"))

    weights = [1.0] * arguments.slabs
    chain = compute_chain_partition(global_case, weights, arguments.pool_safety)
    device_map = [int(item) for item in arguments.device_map.split(",")]
    if len(device_map) != arguments.slabs:
        sys.exit("--device-map needs one entry per slab")

    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
        current = {"case": str(pathlib.Path(arguments.case).resolve().relative_to(_REPO_ROOT)),
                   "slabs": arguments.slabs, "expect": arguments.expect, "dt": dt, "spacing": spacing,
                   "support_radius": support, "sample_steps": sample_steps, "window_steps": window_steps,
                   "defrag_cadence": cadence, "environment": environment}
        hashes = code_hashes(pathlib.Path(arguments.case).resolve())
        for key in ("physics", "sampling"):
            if stored.get("code_hashes", {}).get(key, hashes[key]) != hashes[key]:
                current[f"code_hash_{key}"] = hashes[key]
                stored[f"code_hash_{key}"] = stored["code_hashes"][key]
        mismatch = {key: (stored.get(key), value) for key, value in current.items() if stored.get(key) != value}
        if mismatch:
            sys.exit(f"[cavity] {run_dir} was started with different settings: {mismatch}")
    else:
        meta = {"case": str(pathlib.Path(arguments.case).resolve().relative_to(_REPO_ROOT)),
                "slabs": arguments.slabs, "device_map": device_map, "expect": arguments.expect,
                "environment": environment, "expected_total": expected_total, "dt": dt, "support_radius": support,
                "spacing": spacing, "reynolds_nominal": reynolds, "defrag_cadence": cadence,
                "cuts": list(getattr(chain, "cuts", [])), "pool_safety": arguments.pool_safety,
                "sample_steps": sample_steps, "snapshot_steps": snapshot_steps, "checkpoint_steps": checkpoint_steps,
                "window_steps": window_steps, "t_end": arguments.t_end, "t_max": arguments.t_max,
                "steady_tol": arguments.steady_tol, "average_span": arguments.average_span,
                "wall_boundary": global_case.numerics.wall_boundary,
                "frames": {name: frame.half_width for name, frame in frames.items()},
                "dense_reference": "linspace(0, 1, 1001) in the wall frame",
                "fluid_groups": fluid_groups.tolist(), "git": git_state(),
                "code_hashes": code_hashes(pathlib.Path(arguments.case).resolve())}
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        (run_dir / "case.yaml").write_bytes(pathlib.Path(arguments.case).read_bytes())
        np.savez(run_dir / "points.npz", **{name: points for name, (_, points) in point_sets.items()},
                 dense_reference=dense_reference, u_reference_y=reference["u"][0], v_reference_x=reference["v"][0])

    # ---- resume bookkeeping ------------------------------------------------------------------------------
    start_step = 0
    state = None
    stored_offset = 0.0
    manifest_path = run_dir / "checkpoints" / "manifest.jsonl"
    if arguments.resume and manifest_path.exists():
        entries = read_jsonl(manifest_path)
        for entry in reversed(entries):
            path = run_dir / "checkpoints" / entry["file"]
            if not path.exists():
                continue
            try:
                with np.load(path) as archive:
                    candidate = {name: archive[name] for name in archive.files if not name.startswith("_")}
                    candidate_offset = float(archive["_stored_density_offset"])
            except Exception as error:
                print(f"[cavity] checkpoint {path.name} unreadable ({error!r}); trying an older one", flush=True)
                continue
            if candidate["position_voxel_id"].shape[0] != expected_total:
                print(f"[cavity] checkpoint {path.name} holds {candidate['position_voxel_id'].shape[0]} rows, "
                      f"expected {expected_total}; trying an older one", flush=True)
                continue
            state, stored_offset, start_step = candidate, candidate_offset, int(entry["step"])
            break
        if state is None:
            sys.exit(f"[cavity] --resume: no usable checkpoint in {manifest_path.parent}")
    samples_path = run_dir / "samples.jsonl"
    previous_rows = []
    if samples_path.exists():
        previous_rows = read_jsonl(samples_path)
    kept_rows = [row for row in previous_rows if row["step"] <= start_step]
    atomic_write_text(samples_path, "".join(json.dumps(row) + "\n" for row in kept_rows))
    for path in list((run_dir / "samples").glob("s*.npz")) + list((run_dir / "snapshots").glob("t*.npz")):
        step = step_of(path)
        if step is not None and step > start_step:
            safe_unlink(path)
    windows_path = run_dir / "windows.jsonl"
    if windows_path.exists():
        kept_windows = [row for row in read_jsonl(windows_path) if row["step"] <= start_step]
        atomic_write_text(windows_path, "".join(json.dumps(row) + "\n" for row in kept_windows))
    segments_path = run_dir / "segments.jsonl"
    segment_index = len(read_jsonl(segments_path)) if segments_path.exists() else 0
    steady_path = run_dir / "steady.json"
    steady = json.loads(steady_path.read_text(encoding="utf-8")) if steady_path.exists() else {}
    if steady.get("decided_at_step", 0) > start_step:
        steady = {}
        safe_unlink(steady_path)

    # window means from the kept samples (steady-state test state)
    window_sums: dict[int, dict] = {}

    def add_to_window(step: int, kinetic_energy: float, profile: np.ndarray) -> None:
        index = (step - 1) // window_steps
        entry = window_sums.setdefault(index, {"count": 0, "ke": 0.0, "profile": np.zeros_like(profile)})
        entry["count"] += 1
        entry["ke"] += kinetic_energy
        entry["profile"] += profile

    for row in kept_rows:
        with np.load(run_dir / "samples" / f"s{row['step']:010d}.npz") as archive:
            profile = np.concatenate([archive["u_marchi_wall_mls"], archive["v_marchi_wall_mls"]])
        add_to_window(row["step"], row["kinetic_energy"], profile)

    def stop_step() -> int:
        if steady.get("steady_time") is not None:
            t_stop = max(arguments.t_end, steady["steady_time"] + arguments.average_span)
        else:
            t_stop = arguments.t_max
        steps = int(math.ceil(min(t_stop, arguments.t_max) / dt / cadence)) * cadence
        if arguments.max_steps is not None:
            steps = min(steps, start_step + arguments.max_steps)
        return min(steps, max_total_steps)

    # ---- GPU setup -----------------------------------------------------------------------------------------
    contexts, sims, readers = [], [], []
    orchestrator = None
    totals = {"overflow": 0, "far_migration": 0, "stamp_gpu": 0, "stamp_host": 0}
    log = open(run_dir / "run.log", "a", encoding="utf-8")

    def say(message: str) -> None:
        line = f"[cavity {time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    say(f"segment {segment_index}: case {arguments.case} K={arguments.slabs} devices {device_map} expect "
        f"{arguments.expect}; start step {start_step} (t = {start_step * dt:.3f}); dt {dt:.3e}; "
        f"sample/snapshot/checkpoint/window every {sample_steps}/{snapshot_steps}/{checkpoint_steps}/{window_steps} steps")
    status = "error"
    failure: dict = {}
    wall_start = time.perf_counter()
    try:
        for index in range(arguments.slabs):
            contexts.append(VulkanContextV6.create(device_index=device_map[index], enable_validation=False,
                                                   application_name=f"cavity_s{index}"))
            sims.append(SphSimulatorV6(contexts[-1], chain.slabs[index], sync_scheme="per-direction"))
        uuids = [device_uuid(ctx) for ctx in contexts]
        failure["device_uuids"] = uuids
        say(f"device uuids {uuids}")
        if arguments.require_uuid is not None:
            wanted = [item.replace("-", "").lower() for item in arguments.require_uuid.split(",")]
            if wanted != uuids:
                raise RuntimeError(f"device uuids {uuids} differ from --require-uuid {wanted}")
        orchestrator = ChainOrchestratorV6(sims, defrag_cadence=cadence)
        if state is None:
            if start_step != 0:
                raise RuntimeError("no checkpoint to resume from")
            orchestrator.bootstrap_all()
        else:
            rows_per_slab = restart_slab_rows(global_case, chain, state["position_voxel_id"][:, 0])
            states = [{name: np.ascontiguousarray(values[rows]) for name, values in state.items()}
                      for rows in rows_per_slab]
            orchestrator.restart_all(states, stored_offset)
            del states
        state = None
        readers = [CachedStateReader(sim) for sim in sims]

        last = {"wall": time.perf_counter(), "step": start_step}

        def light_state() -> dict:
            parts = [reader.read() for reader in readers]
            return {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}

        def check_status(report_step: int) -> None:
            for index, sim in enumerate(sims):
                status_values = sim.readback_global_status()
                overflow = {name: value for name, value in status_values.items() if name.startswith("overflow_") and value}
                bad = dict(overflow)
                if status_values.get("far_migration_count", 0):
                    bad["far_migration_count"] = status_values["far_migration_count"]
                if status_values.get("stamp_error_count", 0):
                    bad["stamp_error_count"] = status_values["stamp_error_count"]
                if bad:
                    raise InvariantViolation(f"invariant violation at step {report_step}, sim {index}: {bad}")
            host = sum(getattr(worker, "stamp_error_count", 0) for worker in orchestrator.workers)
            if host:
                raise InvariantViolation(f"host stamp errors at step {report_step}: {host}")

        def on_defrag(frame: int, report: list) -> None:
            step = start_step + frame
            for worker in orchestrator.workers:
                worker.timestamps.clear()
            for index, entry in enumerate(report):
                bad = {key: value for key, value in entry.items()
                       if (key.startswith("overflow_") or key == "far_migration") and value}
                if bad:
                    raise InvariantViolation(f"invariant violation at step {step}, sim {index}: {bad}")
            check_status(step)              # stamp errors and overflow_initialization_outside, every defrag
            now = time.perf_counter()
            (run_dir / "heartbeat.json").write_text(json.dumps({
                "step": step, "time": step * dt, "unix": time.time(), "segment": segment_index,
                "fps": (step - last["step"]) / max(now - last["wall"], 1e-9)}), encoding="utf-8")
            if step % sample_steps == 0:
                sample_started = time.perf_counter()
                light = light_state()
                alive = int(light["mass"].size)
                if alive != expected_total:
                    raise InvariantViolation(f"drift at step {step}: alive {alive} != expected {expected_total}")
                fluid = np.isin(light["material"], fluid_groups)
                speed2 = (light["velocities"][fluid] ** 2).sum(axis=1)
                kinetic_energy = 0.5 * float((light["mass"][fluid] * speed2).sum())
                fluid_mass = float(light["mass"][fluid].sum())
                particles = sampling.ParticleSet(light["positions"], light["velocities"],
                                                 light["mass"] / light["density"])
                sampled = sampling.sample_centerlines(particles, support, point_sets)
                arrays = {}
                for name, result in sampled.items():
                    arrays[f"{name}_mls"] = result["mls"].astype(np.float64)
                    arrays[f"{name}_shepard"] = result["shepard"].astype(np.float64)
                    arrays[f"{name}_neighbours"] = result["neighbours"].astype(np.int32)
                atomic_savez(run_dir / "samples" / f"s{step:010d}.npz", **arrays)
                profile = np.concatenate([arrays["u_marchi_wall_mls"], arrays["v_marchi_wall_mls"]])
                fps = (step - last["step"]) / max(sample_started - last["wall"], 1e-9)
                row = {"step": step, "time": step * dt, "kinetic_energy": kinetic_energy,
                       "max_speed": float(np.sqrt(speed2.max())),
                       "mean_u": float((light["mass"][fluid] * light["velocities"][fluid, 0]).sum() / fluid_mass),
                       "mean_v": float((light["mass"][fluid] * light["velocities"][fluid, 1]).sum() / fluid_mass),
                       "density_min": float(light["density"][fluid].min()),
                       "density_max": float(light["density"][fluid].max()),
                       "alive": alive, "mls_fallbacks": int(sum(result["mls_fallback"].sum() for result in sampled.values())),
                       "min_neighbours": int(min(result["neighbours"].min() for result in sampled.values())),
                       "fps": fps, "sample_s": time.perf_counter() - sample_started, "segment": segment_index,
                       "wall_s": time.perf_counter() - wall_start}
                with open(samples_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                add_to_window(step, kinetic_energy, profile)
                last["wall"], last["step"] = time.perf_counter(), step
                if step % window_steps == 0 and steady.get("steady_time") is None:
                    index = step // window_steps - 1
                    current, previous = window_sums.get(index), window_sums.get(index - 1)
                    if current and previous and current["count"] and previous["count"]:
                        mean_now = current["profile"] / current["count"]
                        mean_before = previous["profile"] / previous["count"]
                        profile_change = float(np.linalg.norm(mean_now - mean_before) / np.linalg.norm(mean_now))
                        ke_now, ke_before = current["ke"] / current["count"], previous["ke"] / previous["count"]
                        ke_change = abs(ke_now - ke_before) / ke_now
                        record = {"step": step, "time": step * dt, "profile_change": profile_change,
                                  "kinetic_energy_change": ke_change}
                        with open(run_dir / "windows.jsonl", "a", encoding="utf-8") as handle:
                            handle.write(json.dumps(record) + "\n")
                        say(f"window ending t = {step * dt:.2f}: profile change {profile_change:.2e}, "
                            f"KE change {ke_change:.2e}")
                        if profile_change < arguments.steady_tol and ke_change < arguments.steady_tol:
                            steady.update({"steady_time": step * dt, "decided_at_step": step,
                                           "profile_change": profile_change, "kinetic_energy_change": ke_change})
                            steady_path.write_text(json.dumps(steady, indent=1), encoding="utf-8")
                            say(f"steady at t = {step * dt:.2f}; stop at t = {stop_step() * dt:.2f}")
            if step * dt >= arguments.t_end - arguments.average_span - 1e-9 and step % snapshot_steps == 0:
                light = light_state()
                atomic_savez(run_dir / "snapshots" / f"t{step:010d}.npz",
                             positions=light["positions"].astype(np.float32),
                             velocities=light["velocities"].astype(np.float32),
                             volumes=(light["mass"] / light["density"]).astype(np.float32),
                             material=light["material"].astype(np.uint8))
            if step % checkpoint_steps == 0:
                parts = [read_restart_state(sim) for sim in sims]
                merged = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
                if merged["position_voxel_id"].shape[0] != expected_total:
                    raise InvariantViolation(f"drift at checkpoint step {step}: "
                                             f"{merged['position_voxel_id'].shape[0]} rows != {expected_total}")
                name = f"c{step:010d}.npz"
                atomic_savez(run_dir / "checkpoints" / name, _stored_density_offset=np.float64(sims[0].stored_density_offset()),
                             **merged)
                with open(manifest_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": step, "time": step * dt, "file": name}) + "\n")
                checkpoints = sorted(path for path in (run_dir / "checkpoints").glob("c*.npz")
                                     if step_of(path) is not None and step_of(path) <= step)
                for old in checkpoints[:-2]:
                    safe_unlink(old)
            if step >= stop_step():
                raise StopRun()

        try:
            orchestrator.run_pipelined(max_total_steps - start_step, depth=2, on_defrag=on_defrag)
        except StopRun:
            pass
        final_step = start_step + orchestrator._frame_count
        for sim in sims:
            sim.submit_defrag_and_wait()
        alive_total = 0
        for index, sim in enumerate(sims):
            status_values = sim.readback_global_status()
            alive_total += status_values["alive_particle_count"]
            totals["overflow"] += sum(value for key, value in status_values.items() if key.startswith("overflow_"))
            totals["far_migration"] += status_values.get("far_migration_count", 0)
            totals["stamp_gpu"] += status_values.get("stamp_error_count", 0)
        totals["stamp_host"] = sum(getattr(worker, "stamp_error_count", 0) for worker in orchestrator.workers)
        totals["drift"] = alive_total - expected_total
        status = "complete" if all(value == 0 for value in totals.values()) else "invalid"
        if status == "invalid":
            failure["invariant_violation"] = True
        say(f"segment {segment_index} end at step {final_step}: {totals} -> {status}")
    except Exception as error:
        failure["error"] = repr(error)
        failure["invariant_violation"] = isinstance(error, InvariantViolation)
        say(f"segment {segment_index} failed: {error!r}")
    finally:
        record = {"segment": segment_index, "start_step": start_step, "status": status, "totals": totals,
                  "error": failure.get("error"), "invariant_violation": failure.get("invariant_violation", False),
                  "device_uuids": failure.get("device_uuids"),
                  "code_hashes": code_hashes(pathlib.Path(arguments.case).resolve()),
                  "wall_s": time.perf_counter() - wall_start, "git": git_state()}
        with open(segments_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if failure.get("error"):
            # frames may be in flight (stall, worker death): skip the Vulkan teardown, which can block forever
            # in vkDeviceWaitIdle; process exit releases the devices.
            log.close()
            os._exit(INVARIANT_EXIT_CODE if failure.get("invariant_violation") else 1)
        for reader in readers:
            reader.destroy()
        if orchestrator is not None:
            orchestrator.destroy()
        for sim in sims:
            sim.destroy()
        for ctx in contexts:
            ctx.destroy()
        log.close()
    segments = read_jsonl(segments_path)
    violated = [segment["segment"] for segment in segments if segment.get("invariant_violation")]
    if status == "complete" and arguments.max_steps is None:
        (run_dir / "result.json").write_text(json.dumps({
            "status": "invalid" if violated else status, "segments_with_invariant_violation": violated,
            "segments": [{key: segment.get(key) for key in ("segment", "start_step", "status", "error")}
                         for segment in segments],
            "steady": steady, "stop_step": stop_step(), "dt": dt}, indent=1), encoding="utf-8")
    if status == "invalid" or violated:
        return INVARIANT_EXIT_CODE
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
