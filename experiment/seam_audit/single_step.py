"""
single_step.py - single-step seam test: the true size of v5 seam defect (1).

Defect (1): in the phase C force pass (C5) a seam-column particle reads its
ghost neighbours' rho^n, P^n (the replicas are packed in phase A, before this
step's density) where the single-GPU step reads rho^{n+1}, P^{n+1}. The
300 / 2000-step audit (docs/seam_audit/v6.md) cannot resolve it: after hundreds
of steps a K=2 run and a K=1 run have diverged chaotically, and that noise floor
hides a one-step lag. Here every run starts from the SAME saved state:

  original  v6 K=1 from the case's initial condition (global ids injected into
            extension_fields exactly as dump_state does). At every snapshot step
            N it saves the full step-boundary state - the nine set-0 SoA fields
            of every particle, sorted by global id - and keeps running, dumping
            the state after N+1, N+2, N+5, N+10, N+50 steps (the self-check
            reference: the same physics with the original run's voxel lists).
  restart   loads one snapshot, splits it into K slabs with the cuts of a
            K-slab run of the case (partition_v6.restart_slab_rows) and starts
            the chain with ChainOrchestratorV6.restart_all: full-state upload +
            voxel lists, NO bootstrap correction / density / force and NO
            backward half kick; the first frame's phase A rebuilds the ghosts.
            Runs the frames one at a time, records every migration (slab
            change, by global id) and dumps after 1, 2, 5, 10, 50 steps.
            V6_KEEP_DEPARTED / V6_GHOST_LAYERS come from the environment;
            --shuffle-seed permutes the upload order (a different neighbour
            summation order, same physics).
  analyze   matches the dumps by global id (no KD-tree: at one step every run
            holds the same particles) and bins by column distance to the cut.
  campaign  runs the matrix (one subprocess per run) and then the analysis.

Dumps hold FLUID particles inside a column window around the K=2 cut (the disk
cannot hold every particle of every run): --step1-window columns per side at
step 1 with every field, --window columns per side at later steps without the
correction matrix and density gradient.

Usage (GPU):
  .venv/Scripts/python.exe -m experiment.seam_audit.single_step campaign \\
      --out logs/seam_audit/single_step
  .venv/Scripts/python.exe -m experiment.seam_audit.single_step analyze \\
      --out logs/seam_audit/single_step
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
import traceback

import numpy as np

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from experiment.seam_audit.dump_state import (  # noqa: E402
    GLOBAL_ID_ATTRIBUTE,
    OwnPositionReader,
    check_encoding_round_trip,
    decode_global_ids,
    install_global_id_injection,
    json_default,
    perform_defrag,
    read_statuses,
    recorded_environment,
    run_frames,
    save_json_atomically,
    save_npz_atomically,
    slab_global_indices,
)
from experiment.seam_audit.solver_adapter import load_solver  # noqa: E402

LOG_PREFIX = "[single_step]"
RESULT_PREFIX = "[single_step] RESULT "
SOLVER_VERSION = "v6"
EXIT_VALID, EXIT_ERROR, EXIT_INVALID = 0, 1, 3

# The nine set-0 fields of a step boundary (SphSimulatorV6.RESTART_FIELD_LAYOUT).
STATE_FIELDS = ("position_voxel_id", "velocity_mass", "density_pressure", "acceleration",
                "shift", "material", "correction_inverse", "density_gradient_kernel_sum",
                "extension_fields")
# correction_inverse rows: (m00, m11, m22, m01) (m02, m12, 0, 0) -> 6 unique entries.
CORRECTION_COMPONENTS = (0, 1, 2, 3, 4, 5)
KIND_FLUID = 0

# Campaign matrix.
CASES = (
    {"name": "cavity2d_1m", "path": "cases/lid_driven_cavity_2d_gen/case.yaml",
     "step1_window": 12, "window": 12, "long_window": 8},
    {"name": "cavity2d_4m", "path": "cases/lid_driven_cavity_2d_4m/case.yaml",
     "step1_window": 12, "window": 12, "long_window": 8},
    {"name": "cavity3d_1m", "path": "cases/cavity3d_1m/case.yaml",
     "step1_window": 8, "window": 5, "long_window": 4},
)
# Long-horizon pass (how fast the defect-(1) difference meets the chaos noise):
# the reference, the noise run and one trial of each seam configuration, run
# to 2000 steps from the same snapshots, in <case>_long/.
LONG_STEPS = (1, 10, 50, 100, 200, 500, 1000, 2000)
LONG_RUN_NAMES = ("k1_a", "k1_shuffled", "keep0_layers1_t1", "keep1_layers1_t1",
                  "keep1_layers2_t1")
SNAPSHOT_STEPS = (300, 2000)
STEPS_AFTER_RESTART = (1, 2, 5, 10, 50)
REFERENCE_DEVICE = "1"
# name -> (slabs, device map, environment, shuffle seed)
RESTART_RUNS = (
    ("k1_a", 1, REFERENCE_DEVICE, {}, 0),
    ("k1_b", 1, REFERENCE_DEVICE, {}, 0),
    ("k1_shuffled", 1, REFERENCE_DEVICE, {}, 1),
    ("keep0_layers1_t1", 2, "0,1", {"V6_KEEP_DEPARTED": "0", "V6_GHOST_LAYERS": "1"}, 0),
    ("keep1_layers1_t1", 2, "0,1", {"V6_KEEP_DEPARTED": "1", "V6_GHOST_LAYERS": "1"}, 0),
    ("keep1_layers1_t2", 2, "0,1", {"V6_KEEP_DEPARTED": "1", "V6_GHOST_LAYERS": "1"}, 0),
    ("keep1_layers2_t1", 2, "0,1", {"V6_KEEP_DEPARTED": "1", "V6_GHOST_LAYERS": "2"}, 0),
    ("keep1_layers2_t2", 2, "0,1", {"V6_KEEP_DEPARTED": "1", "V6_GHOST_LAYERS": "2"}, 0),
)
SEAM_ENVIRONMENT_KEYS = ("V6_KEEP_DEPARTED", "V6_GHOST_LAYERS")


# =============================================================================
# File layout
# =============================================================================

def snapshot_path(directory, snapshot_step: int) -> pathlib.Path:
    return pathlib.Path(directory) / f"snapshot_N{snapshot_step}.npz"


def dump_path(directory, run_name: str, snapshot_step: int, step: int) -> pathlib.Path:
    return pathlib.Path(directory) / f"{run_name}_N{snapshot_step}_k{step}.npz"


def run_json_path(directory, run_name: str, snapshot_step: int) -> pathlib.Path:
    return pathlib.Path(directory) / f"{run_name}_N{snapshot_step}.json"


def parse_integers(text: str) -> list[int]:
    return sorted({int(item) for item in str(text).split(",") if item.strip()})


# =============================================================================
# State readback (GPU side)
# =============================================================================

def read_slab_state(sim, initial_total: int) -> tuple[dict, np.ndarray]:
    """Raw rows of every alive own particle of one sim for the nine
    step-boundary fields, plus their global ids."""
    layout = sim.RESTART_FIELD_LAYOUT
    raw = sim.readback_buffers_batch(list(layout))
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
    del raw
    alive = (rows["velocity_mass"][:, 3] > 0) & (rows["position_voxel_id"][:, 3] > 0.5)
    rows = {name: values[alive] for name, values in rows.items()}
    ids, foreign = decode_global_ids(rows["extension_fields"])
    if foreign.any() or (ids < 0).any() or (ids >= initial_total).any():
        raise RuntimeError(f"invalid global ids in extension_fields ({int(foreign.sum())} "
                           f"foreign rows, {int(((ids < 0) | (ids >= initial_total)).sum())} "
                           "out of range)")
    return rows, ids


def gather_state(sims, initial_total: int) -> dict:
    """Every alive own particle of every sim, sorted by global id: the raw
    fields plus 'id' (int64) and 'slab' (uint8)."""
    parts = []
    for slab_index, sim in enumerate(sims):
        rows, ids = read_slab_state(sim, initial_total)
        rows["id"] = ids
        rows["slab"] = np.full(ids.size, slab_index, dtype=np.uint8)
        parts.append(rows)
    merged = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
    order = np.argsort(merged["id"], kind="stable")
    state = {name: np.ascontiguousarray(values[order]) for name, values in merged.items()}
    counts = np.bincount(state["id"], minlength=initial_total)
    state["_missing"] = int((counts == 0).sum())
    state["_duplicates"] = int(np.maximum(counts - 1, 0).sum())
    return state


def global_columns(positions_x: np.ndarray, origin_x: float, smoothing_length: float,
                   column_count: int) -> np.ndarray:
    """Global voxel column of each x (the partitioner's expression and clamp)."""
    columns = np.floor((np.asarray(positions_x, dtype=np.float64) - origin_x)
                       / smoothing_length).astype(np.int64)
    np.clip(columns, 0, column_count - 1, out=columns)
    return columns


def compact_dump(state: dict, geometry: dict, window: int, full: bool) -> dict:
    """Fluid particles with signed column t = column - cut in [-window, window).
    With ``full`` (step 1) also every non-fluid particle of the two seam
    columns t = -1, 0: the walls a seam particle sees across the cut, which
    the stale-ghost predictor of the analysis needs."""
    columns = global_columns(state["position_voxel_id"][:, 0], geometry["origin_x"],
                             geometry["smoothing_length"], geometry["column_count"])
    signed = columns - geometry["cut_column"]
    fluid = np.isin(state["material"], np.asarray(geometry["fluid_groups"], dtype=np.uint32))
    keep = fluid & (signed >= -window) & (signed < window)
    if full:
        keep |= (signed >= -1) & (signed <= 0)
    density_pressure = state["density_pressure"][keep]
    gradient_kernel_sum = state["density_gradient_kernel_sum"][keep]
    dump = {
        "id": state["id"][keep].astype(np.uint32),
        "slab": state["slab"][keep],
        "material": state["material"][keep].astype(np.uint16),
        "position": state["position_voxel_id"][keep, :3],
        "velocity": state["velocity_mass"][keep, :3],
        "acceleration": state["acceleration"][keep, :3],
        "shift": state["shift"][keep, :3],
        "density": density_pressure[:, 0].copy(),
        "pressure": density_pressure[:, 1].copy(),
        "kernel_sum": gradient_kernel_sum[:, 3].copy(),
    }
    if full:
        dump["density_gradient"] = gradient_kernel_sum[:, :3].copy()
        dump["correction_inverse"] = state["correction_inverse"][keep][:, CORRECTION_COMPONENTS]
    return {name: np.ascontiguousarray(values) for name, values in dump.items()}


def case_geometry(solver, global_case, pool_safety) -> dict:
    """Cut column of the equal-weight K=2 chain (the seam every run is binned
    against) and the fluid material groups."""
    chain_two = solver.compute_chain_partition(global_case, [1.0, 1.0], pool_safety)
    fluid_groups = [index for index, material in enumerate(global_case.materials)
                    if int(material.kind) == KIND_FLUID]
    physics = global_case.physics
    numerics = global_case.numerics
    return {
        "cut_column": int(chain_two.cuts[0]),
        "origin_x": float(global_case.grid.origin_x),
        "smoothing_length": float(physics.smoothing_length),
        "column_count": int(global_case.grid.grid_dimension_x),
        "dimension": int(physics.dimension),
        "fluid_groups": fluid_groups,
        # Constants of the force pass (force.comp), for the analysis' CPU
        # reconstruction of the stale-ghost difference.
        "force_constants": {
            "kernel_coefficient": float(physics.kernel_coefficient),
            "kernel_gradient_coefficient": float(physics.kernel_gradient_coefficient),
            "eps_h_squared": float(numerics.eps_h_squared),
            "cfl_number": float(physics.cfl_number),
            "timestep": float(physics.timestep),
            "pst_main_shift_coefficient": float(numerics.pst_main_shift_coefficient),
            "pst_anti_shift_coefficient": float(numerics.pst_anti_shift_coefficient),
            "use_kcg_correction": bool(numerics.use_kcg_correction),
            "use_pst": bool(numerics.use_pst),
        },
        "materials": [{"kind": int(material.kind), "rest_density": float(material.rest_density),
                       "viscosity": float(material.viscosity), "radius": float(material.radius),
                       "volume": float(material.volume),
                       "eos_constant": float(material.eos_constant)}
                      for material in global_case.materials],
    }


class StagingMigrantReader:
    """Ids of this frame's migrants, read from the sender stagings.

    Every slab change goes through a ghost_send migrant packet, so the sender
    staging of a drained frame (host-visible, valid until the next readback)
    lists exactly the particles that changed slab in that frame: the rows of
    the region that carries extension_fields (the V5 mixed pool, or the
    LAYERS=2 migrant region) whose .w is an OWN voxel of the receiver.
    O(migrants) per frame instead of a readback of every own particle."""

    def __init__(self, sims) -> None:
        self.entries = []
        for slab_index, sim in enumerate(sims):
            for direction, segments in sim._transport_segments.items():
                peer_index = slab_index + 1 if direction == "trailing" else slab_index - 1
                peer_case = sims[peer_index].case
                extension = next(segment for segment in segments
                                 if segment.buffer_name == "extension_fields")
                position = next(segment for segment in segments
                                if segment.buffer_name == "position_voxel_id"
                                and segment.count_staging_offset == extension.count_staging_offset)
                first_own_voxel = peer_case.ghost_grid.leading_ghost_voxel_count + 1
                last_own_voxel = (peer_case.grid.total_voxel_count()
                                  - peer_case.ghost_grid.trailing_ghost_voxel_count)
                self.entries.append((slab_index, peer_index, sim.sender_staging_view(direction),
                                     position, extension, first_own_voxel, last_own_voxel))

    def read(self, initial_total: int) -> list:
        """[(id, from slab, to slab)] of the last drained frame."""
        out = []
        for (source, destination, view, position, extension, first_own_voxel,
             last_own_voxel) in self.entries:
            count = int(np.frombuffer(view, dtype=np.uint32, count=1,
                                      offset=extension.count_staging_offset)[0])
            count = min(count, extension.size // 16)
            if count == 0:
                continue
            rows = np.frombuffer(view, dtype=np.float32, count=4 * count,
                                 offset=position.staging_offset).reshape(count, 4)
            voxel_ids = np.rint(rows[:, 3]).astype(np.int64)
            migrant = (voxel_ids >= first_own_voxel) & (voxel_ids <= last_own_voxel)
            if not migrant.any():
                continue
            extension_rows = np.frombuffer(view, dtype=np.float32, count=4 * count,
                                           offset=extension.staging_offset).reshape(count, 4)
            ids, foreign = decode_global_ids(extension_rows[migrant])
            if foreign.any() or (ids < 0).any() or (ids >= initial_total).any():
                raise RuntimeError("invalid global id in a migrant packet")
            out.extend((int(particle_id), source, destination) for particle_id in ids)
        return out


def must_be_zero_status(statuses: list[dict]) -> dict:
    """overflow_* counters + stamp errors + v6 far_migration, summed over sims."""
    totals: dict = {}
    for status in statuses:
        for key, value in status.items():
            if key.startswith("overflow_") or key in ("stamp_error_count", "far_migration_count"):
                totals[key] = totals.get(key, 0) + int(value)
    return totals


def teardown(orchestrator, readers, sims, contexts) -> None:
    orchestrator.destroy()
    for reader in readers:
        reader.destroy()
    for sim in sims:
        sim.destroy()
    for context in contexts:
        context.destroy()


# =============================================================================
# original: K=1 from the initial condition, snapshots + continuation dumps
# =============================================================================

def run_original(arguments, summary: dict) -> int:
    solver = load_solver(SOLVER_VERSION)
    out_directory = pathlib.Path(arguments.out_dir)
    out_directory.mkdir(parents=True, exist_ok=True)
    global_case = solver.load_case(arguments.case)
    pool_safety = arguments.pool_safety or None
    geometry = case_geometry(solver, global_case, pool_safety)
    chain = solver.compute_chain_partition(global_case, [1.0], pool_safety)
    indices_per_slab = slab_global_indices(global_case, chain)
    initial_total = int(global_case.initial.positions.shape[0])
    check_encoding_round_trip(initial_total)
    defrag_cadence = int(global_case.numerics.defrag_cadence) or 10 ** 12
    snapshots = parse_integers(arguments.snapshots)
    steps = parse_integers(arguments.steps)
    horizons = sorted(set(snapshots) | {snapshot + step for snapshot in snapshots for step in steps})
    print(f"{LOG_PREFIX} original K=1 {arguments.case}: snapshots {snapshots}, dumps after "
          f"{steps} more steps; cut column {geometry['cut_column']}, defrag every "
          f"{defrag_cadence}", flush=True)

    install_global_id_injection(solver.Simulator)
    context = solver.Context.create(device_index=int(arguments.device), enable_validation=False,
                                    application_name="single_step_original")
    sim = solver.Simulator(context, chain.slabs[0], sync_scheme=arguments.sync_scheme)
    setattr(sim, GLOBAL_ID_ATTRIBUTE, indices_per_slab[0])
    orchestrator = solver.Orchestrator([sim], defrag_cadence=defrag_cadence)
    orchestrator.bootstrap_all()

    defrag_log: list = []
    current_frame = 0
    records = []
    start = time.perf_counter()
    for horizon in horizons:
        run_frames(orchestrator, current_frame, horizon, arguments.depth, defrag_cadence,
                   defrag_log, defrag_at_stop=False, stall_timeout_seconds=arguments.stall_timeout)
        state = gather_state([sim], initial_total)
        alive = int(state["id"].size)
        if alive != initial_total or state["_missing"] or state["_duplicates"]:
            raise RuntimeError(f"step {horizon}: {alive} alive of {initial_total}, missing "
                               f"{state['_missing']}, duplicates {state['_duplicates']}")
        if horizon in snapshots:
            arrays = {name: state[name] for name in STATE_FIELDS}
            arrays["id"] = state["id"].astype(np.uint32)
            save_npz_atomically(snapshot_path(out_directory, horizon), arrays)
            print(f"{LOG_PREFIX} snapshot after {horizon} steps: {alive:,} particles", flush=True)
        for snapshot in snapshots:
            step = horizon - snapshot
            if step in steps:
                dump = compact_dump(state, geometry,
                                    arguments.step1_window if step == 1 else arguments.window,
                                    full=step == 1)
                save_npz_atomically(dump_path(out_directory, "original", snapshot, step), dump)
        records.append({"step": horizon, "alive": alive})
        del state
        if horizon % defrag_cadence == 0:
            perform_defrag(orchestrator, horizon, defrag_log)
        current_frame = horizon

    statuses, healths = read_statuses([sim])
    status_totals = must_be_zero_status(statuses)
    valid = all(value == 0 for value in status_totals.values())
    document = {
        "tool": "experiment/seam_audit/single_step.py original",
        "case": arguments.case, "device": arguments.device, "snapshots": snapshots,
        "steps": steps, "geometry": geometry, "initial_total": initial_total,
        "defrag_cadence": defrag_cadence, "records": records, "global_status": statuses,
        "pool_health": healths, "must_be_zero": status_totals, "valid": valid,
        "wall_time_s": time.perf_counter() - start,
        "environment": recorded_environment(), "host": platform.node(),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    save_json_atomically(out_directory / "original.json", document)
    teardown(orchestrator, [], [sim], [context])
    summary.update({"valid": valid, "must_be_zero": status_totals})
    return EXIT_VALID if valid else EXIT_INVALID


# =============================================================================
# restart: K slabs from one snapshot
# =============================================================================

def run_restart(arguments, summary: dict) -> int:
    solver = load_solver(SOLVER_VERSION)
    out_directory = pathlib.Path(arguments.out_dir)
    global_case = solver.load_case(arguments.case)
    pool_safety = arguments.pool_safety or None
    geometry = case_geometry(solver, global_case, pool_safety)
    slab_count = int(arguments.slabs)
    chain = solver.compute_chain_partition(global_case, [1.0] * slab_count, pool_safety)
    initial_total = int(global_case.initial.positions.shape[0])
    steps = parse_integers(arguments.steps)
    last_step = max(steps)
    snapshot_step = int(arguments.snapshot_step)

    snapshot_directory = pathlib.Path(arguments.snapshot_dir or out_directory)
    with np.load(snapshot_path(snapshot_directory, snapshot_step)) as archive:
        snapshot = {name: archive[name] for name in STATE_FIELDS}
        snapshot_ids = archive["id"].astype(np.int64)
    if snapshot_ids.size != initial_total:
        raise RuntimeError(f"snapshot holds {snapshot_ids.size} of {initial_total} particles")
    order = np.arange(snapshot_ids.size)
    if arguments.shuffle_seed:
        order = np.random.default_rng(int(arguments.shuffle_seed)).permutation(snapshot_ids.size)
        snapshot = {name: values[order] for name, values in snapshot.items()}
        snapshot_ids = snapshot_ids[order]
    rows_per_slab = solver.partition_module.restart_slab_rows(
        global_case, chain, snapshot["position_voxel_id"][:, 0])
    states = [{name: np.ascontiguousarray(snapshot[name][rows]) for name in STATE_FIELDS}
              for rows in rows_per_slab]
    owner = np.full(initial_total, 255, dtype=np.uint8)
    for slab_index, rows in enumerate(rows_per_slab):
        owner[snapshot_ids[rows]] = slab_index
    del snapshot
    device_map = [int(item) for item in str(arguments.device_map).split(",")]
    print(f"{LOG_PREFIX} restart {arguments.run_name}: snapshot N={snapshot_step}, K={slab_count} "
          f"(cuts {list(chain.cuts)}, rows {[int(rows.size) for rows in rows_per_slab]}), "
          f"shuffle seed {arguments.shuffle_seed}, seam switches "
          f"{ {key: os.environ.get(key, '') for key in SEAM_ENVIRONMENT_KEYS} }", flush=True)

    contexts, sims = [], []
    for slab_index in range(slab_count):
        contexts.append(solver.Context.create(
            device_index=device_map[slab_index % len(device_map)], enable_validation=False,
            application_name=f"single_step_{arguments.run_name}_s{slab_index}"))
        sims.append(solver.Simulator(contexts[-1], chain.slabs[slab_index],
                                     sync_scheme=arguments.sync_scheme))
    orchestrator = solver.Orchestrator(sims, defrag_cadence=10 ** 12)
    orchestrator.restart_all(states)
    del states
    tracking = arguments.migration_tracking if slab_count > 1 else "none"
    readers = [OwnPositionReader(sim) for sim in sims] if tracking == "ownership" else []
    staging_reader = StagingMigrantReader(sims) if tracking == "staging" else None

    migrations = []            # (step, id, from slab, to slab)
    migration_counts = {}
    start = time.perf_counter()
    for frame in range(last_step):
        orchestrator._submit_frame(frame)
        orchestrator._frame_count = frame + 1
        orchestrator._wait_frame(frame, arguments.stall_timeout)
        step = frame + 1
        if readers:
            new_owner = np.full(initial_total, 255, dtype=np.uint8)
            for slab_index, reader in enumerate(readers):
                ids, _x = reader.read(initial_total)
                new_owner[ids] = slab_index
            changed = np.flatnonzero(new_owner != owner)
            for particle_id in changed:
                migrations.append((step, int(particle_id), int(owner[particle_id]),
                                   int(new_owner[particle_id])))
            migration_counts[step] = int(changed.size)
            owner = new_owner
        elif staging_reader is not None:
            frame_migrations = staging_reader.read(initial_total)
            for particle_id, source, destination in frame_migrations:
                migrations.append((step, particle_id, source, destination))
            migration_counts[step] = len(frame_migrations)
        if step in steps:
            state = gather_state(sims, initial_total)
            if state["id"].size != initial_total or state["_missing"] or state["_duplicates"]:
                raise RuntimeError(f"step {step}: {state['id'].size} alive of {initial_total}, "
                                   f"missing {state['_missing']}, duplicates {state['_duplicates']}")
            dump = compact_dump(state, geometry,
                                arguments.step1_window if step == 1 else arguments.window,
                                full=step == 1)
            save_npz_atomically(dump_path(out_directory, arguments.run_name, snapshot_step, step),
                                dump)
            del state
    statuses, healths = read_statuses(sims)
    status_totals = must_be_zero_status(statuses)
    worker_stamp_errors = {worker.label: int(getattr(worker, "stamp_error_count", 0))
                           for worker in getattr(orchestrator, "workers", ())}
    valid = (all(value == 0 for value in status_totals.values())
             and not any(worker_stamp_errors.values()))
    document = {
        "tool": "experiment/seam_audit/single_step.py restart",
        "run_name": arguments.run_name, "case": arguments.case, "snapshot_step": snapshot_step,
        "slabs": slab_count, "cuts": [int(cut) for cut in chain.cuts],
        "device_map": device_map, "shuffle_seed": int(arguments.shuffle_seed),
        "rows_per_slab": [int(rows.size) for rows in rows_per_slab], "steps": steps,
        "migration_tracking": tracking,
        "geometry": geometry, "migration_counts": migration_counts,
        "migrations": migrations, "global_status": statuses, "pool_health": healths,
        "worker_stamp_errors": worker_stamp_errors, "must_be_zero": status_totals,
        "valid": valid, "wall_time_s": time.perf_counter() - start,
        "seam_switches": {key: os.environ.get(key, "") for key in SEAM_ENVIRONMENT_KEYS},
        "environment": recorded_environment(), "host": platform.node(),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    save_json_atomically(run_json_path(out_directory, arguments.run_name, snapshot_step), document)
    teardown(orchestrator, readers, sims, contexts)
    total_migrations = sum(migration_counts.values())
    print(f"{LOG_PREFIX} {arguments.run_name} N={snapshot_step}: {last_step} steps, "
          f"{total_migrations} migrations, must-be-zero {status_totals} -> "
          f"{'VALID' if valid else '*** INVALID ***'}", flush=True)
    summary.update({"valid": valid, "migrations": total_migrations, "must_be_zero": status_totals})
    return EXIT_VALID if valid else EXIT_INVALID


# =============================================================================
# analyze
# =============================================================================

VECTOR_QUANTITIES = ("acceleration", "shift", "velocity", "position", "density_gradient")
REPORTED_QUANTITIES = ("acceleration", "shift", "density", "pressure", "kernel_sum",
                       "correction_inverse", "density_gradient", "velocity", "position")
REFERENCE_RUN = "k1_a"
NOISE_RUN = "k1_shuffled"
COMPARED_RUNS = ("original", "k1_b", "k1_shuffled", "keep0_layers1_t1", "keep1_layers1_t1",
                 "keep1_layers1_t2", "keep1_layers2_t1", "keep1_layers2_t2")
TEST_RUNS = ("keep0_layers1_t1", "keep1_layers1_t1", "keep1_layers1_t2",
             "keep1_layers2_t1", "keep1_layers2_t2")
TRIAL_PAIRS = (("keep1_layers1_t1", "keep1_layers1_t2"), ("keep1_layers2_t1", "keep1_layers2_t2"))
# A departed migrant missing from the sender's lists (defect (2)) corrupts the
# correction / density of every particle within h of it, and the force pass
# then reads those values for every particle within h of THOSE: 2h in one step.
CROSSING_FLAG_RADIUS_IN_H = 2.0


def load_dump(path: pathlib.Path) -> dict:
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def magnitude(name: str, values: np.ndarray) -> np.ndarray:
    """Per-particle size of a quantity (or of a difference of it)."""
    values = values.astype(np.float64)
    if name in VECTOR_QUANTITIES:
        return np.sqrt(np.sum(values * values, axis=1))
    if name == "correction_inverse":    # symmetric matrix, off-diagonals count twice
        weights = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        return np.sqrt(np.sum(weights * values * values, axis=1))
    return np.abs(values)


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values * values))) if values.size else float("nan")


def median(values: np.ndarray) -> float:
    return float(np.median(values)) if values.size else float("nan")


def safe_ratio(test: float, noise: float) -> float:
    """test / noise; 0 when both are exactly 0 (identical), inf when only the
    noise is exactly 0."""
    if noise > 0:
        return test / noise
    return 0.0 if test == 0 else float("inf")


def relative_median(difference: np.ndarray, reference: np.ndarray) -> float:
    """Median of |diff| / |ref| over particles whose |ref| is not tiny."""
    if not difference.size:
        return float("nan")
    floor = 1e-3 * rms(reference)
    usable = reference > floor
    return float(np.median(difference[usable] / reference[usable])) if usable.any() else float("nan")


def flag_crossing_neighbourhood(positions: np.ndarray, crossing_rows: np.ndarray,
                                smoothing_length: float) -> np.ndarray:
    """Rows within h of any crossing row (crossing rows included)."""
    flagged = np.zeros(positions.shape[0], dtype=bool)
    if crossing_rows.size == 0:
        return flagged
    from scipy.spatial import cKDTree
    tree = cKDTree(positions[crossing_rows].astype(np.float64))
    distances, _ = tree.query(positions.astype(np.float64), k=1,
                              distance_upper_bound=smoothing_length)
    flagged[np.isfinite(distances)] = True
    flagged[crossing_rows] = True
    return flagged


def correction_matrices(packed: np.ndarray) -> np.ndarray:
    """(n, 6) packed (m00, m11, m22, m01, m02, m12) -> (n, 3, 3)."""
    packed = packed.astype(np.float64)
    matrices = np.empty((packed.shape[0], 3, 3))
    matrices[:, 0, 0], matrices[:, 1, 1], matrices[:, 2, 2] = packed[:, 0], packed[:, 1], packed[:, 2]
    matrices[:, 0, 1] = matrices[:, 1, 0] = packed[:, 3]
    matrices[:, 0, 2] = matrices[:, 2, 0] = packed[:, 4]
    matrices[:, 1, 2] = matrices[:, 2, 1] = packed[:, 5]
    return matrices


def stale_ghost_prediction(reference: dict, signed: np.ndarray, self_rows: np.ndarray,
                           snapshot: dict, geometry: dict) -> dict:
    """CPU reconstruction of defect (1) at step 1.

    For every self row i (a clean fluid particle of a seam column), sum over
    its neighbours j ACROSS the cut (the ghost replicas of a K=2 run) the
    force-pass terms of force.comp evaluated twice: with the replica's
    rho_j^n, P_j^n (what C5 reads at layers = 1) and with rho_j^{n+1},
    P_j^{n+1} (what the K=1 step reads). Positions, velocities and every
    self quantity are the reference's step-1 values (identical in every run
    at step 1). Returns the predicted (stale - fresh) acceleration and shift
    per self row, plus the size of the ghost density / pressure change."""
    from scipy.spatial import cKDTree
    constants = geometry["force_constants"]
    materials = geometry["materials"]
    smoothing_length = geometry["smoothing_length"]
    dimension = geometry["dimension"]
    positions = reference["position"].astype(np.float64)
    ids = reference["id"].astype(np.int64)
    left = np.flatnonzero(signed == -1)
    right = np.flatnonzero(signed == 0)
    pairs = cKDTree(positions[left]).sparse_distance_matrix(
        cKDTree(positions[right]), smoothing_length, output_type="coo_matrix")
    left_rows, right_rows = left[pairs.row], right[pairs.col]
    # each cross pair serves both ends as self
    self_index = np.concatenate([left_rows, right_rows])
    neighbour_index = np.concatenate([right_rows, left_rows])
    is_self = np.zeros(positions.shape[0], dtype=bool)
    is_self[self_rows] = True
    keep = is_self[self_index]
    self_index, neighbour_index = self_index[keep], neighbour_index[keep]

    relative = positions[self_index] - positions[neighbour_index]          # x_ij = r_i - r_j
    distance = np.sqrt(np.sum(relative * relative, axis=1))
    usable = (distance < smoothing_length) & (distance >= 1e-12)
    self_index, neighbour_index = self_index[usable], neighbour_index[usable]
    relative, distance = relative[usable], distance[usable]
    normalized = distance / smoothing_length
    one_minus = 1.0 - normalized
    gradient = (constants["kernel_gradient_coefficient"] * one_minus ** 5
                * normalized * ((-280.0 / 3.0) * normalized + (-56.0 / 3.0)))[:, None] \
        * relative / distance[:, None]
    if constants["use_kcg_correction"]:
        matrices = correction_matrices(reference["correction_inverse"][self_index])
        gradient = np.einsum("nab,nb->na", matrices, gradient)

    self_material = reference["material"][self_index].astype(np.int64)
    viscosity = np.array([materials[group]["viscosity"] for group in self_material])
    radius = np.array([materials[group]["radius"] for group in self_material])
    volume = np.array([materials[group]["volume"] for group in self_material])
    self_ids = ids[self_index]
    neighbour_ids = ids[neighbour_index]
    self_mass = snapshot["velocity_mass"][self_ids, 3].astype(np.float64)
    self_density = reference["density"][self_index].astype(np.float64)
    self_pressure = reference["pressure"][self_index].astype(np.float64)
    self_kernel_sum = reference["kernel_sum"][self_index].astype(np.float64)
    near_surface = self_kernel_sum < 0.75
    velocity_difference = (reference["velocity"][self_index].astype(np.float64)
                           - reference["velocity"][neighbour_index].astype(np.float64))
    morris = 2.0 * (dimension + 2.0)
    viscous_scalar = (morris * viscosity * np.sum(velocity_difference * relative, axis=1)
                      / (distance * distance + constants["eps_h_squared"]))
    delta_x = 2.0 * radius
    distance_ratio = (distance - delta_x) / delta_x

    def terms(neighbour_density, neighbour_pressure):
        neighbour_volume = self_mass / neighbour_density
        combined = np.where((self_pressure > 0.0) | near_surface,
                            neighbour_pressure + self_pressure,
                            neighbour_pressure - self_pressure)
        acceleration = ((-neighbour_volume * combined / self_density
                         + neighbour_volume * viscous_scalar)[:, None] * gradient)
        disorder = np.where(distance_ratio > 0.0, 0.0,
                            distance_ratio * np.minimum(neighbour_density / self_density, 1.0))
        return acceleration, disorder

    stale_density = snapshot["density_pressure"][neighbour_ids, 0].astype(np.float64)
    stale_pressure = snapshot["density_pressure"][neighbour_ids, 1].astype(np.float64)
    fresh_density = reference["density"][neighbour_index].astype(np.float64)
    fresh_pressure = reference["pressure"][neighbour_index].astype(np.float64)
    stale_acceleration, stale_disorder = terms(stale_density, stale_pressure)
    fresh_acceleration, fresh_disorder = terms(fresh_density, fresh_pressure)
    pair_acceleration = stale_acceleration - fresh_acceleration
    pst_base = (constants["cfl_number"] * constants["pst_main_shift_coefficient"] * 2.0
                * smoothing_length * smoothing_length)
    blend = np.clip(self_kernel_sum, 0.0, 1.0) ** 4
    pair_shift = ((blend * pst_base * volume * (stale_disorder - fresh_disorder))[:, None]
                  * gradient) if constants["use_pst"] else np.zeros_like(gradient)

    row_of = np.full(positions.shape[0], -1, dtype=np.int64)
    row_of[self_rows] = np.arange(self_rows.size)
    predicted_acceleration = np.zeros((self_rows.size, 3))
    predicted_shift = np.zeros((self_rows.size, 3))
    np.add.at(predicted_acceleration, row_of[self_index], pair_acceleration)
    np.add.at(predicted_shift, row_of[self_index], pair_shift)

    # Size of the one-step change the stale replica misses (per ghost pair).
    unique_neighbours = np.unique(neighbour_index)
    neighbour_ids_unique = ids[unique_neighbours]
    density_change = (reference["density"][unique_neighbours].astype(np.float64)
                      - snapshot["density_pressure"][neighbour_ids_unique, 0])
    pressure_change = (reference["pressure"][unique_neighbours].astype(np.float64)
                       - snapshot["density_pressure"][neighbour_ids_unique, 1])
    fresh_pressure_unique = reference["pressure"][unique_neighbours].astype(np.float64)
    close_pairs = distance_ratio <= 0.0
    return {
        "acceleration": predicted_acceleration,
        "shift": predicted_shift,
        "statistics": {
            "self_particles": int(self_rows.size),
            "ghost_pairs": int(self_index.size),
            "ghost_pairs_per_particle": float(self_index.size / max(self_rows.size, 1)),
            "close_pair_fraction": float(close_pairs.mean()) if close_pairs.size else 0.0,
            "ghost_particles": int(unique_neighbours.size),
            "ghost_density_change_relative_rms": rms(density_change / fresh_density.mean()
                                                     if fresh_density.size else density_change),
            "ghost_density_change_relative_max": float(np.max(np.abs(density_change))
                                                       / fresh_density.mean()) if density_change.size else 0.0,
            "ghost_density_changed_fraction": float(np.mean(density_change != 0))
                                              if density_change.size else 0.0,
            "ghost_pressure_change_rms": rms(pressure_change),
            "ghost_pressure_change_max": float(np.max(np.abs(pressure_change)))
                                         if pressure_change.size else 0.0,
            "ghost_pressure_rms": rms(fresh_pressure_unique),
            "ghost_pressure_change_relative_median": relative_median(
                np.abs(pressure_change), np.abs(fresh_pressure_unique)),
        },
    }


def analyze_step(directory: pathlib.Path, snapshot_step: int, step: int, geometry: dict,
                 migrations: dict, snapshot: dict | None) -> dict:
    dumps = {}
    for run_name in (REFERENCE_RUN,) + COMPARED_RUNS:
        path = dump_path(directory, run_name, snapshot_step, step)
        if path.exists():
            dumps[run_name] = load_dump(path)
    if REFERENCE_RUN not in dumps or NOISE_RUN not in dumps:
        return {"missing": True}
    common = dumps[REFERENCE_RUN]["id"]
    for dump in dumps.values():
        common = np.intersect1d(common, dump["id"], assume_unique=True)
    aligned = {}
    for run_name, dump in dumps.items():
        rows = np.searchsorted(dump["id"], common)
        aligned[run_name] = {name: values[rows] for name, values in dump.items()}
    reference = aligned[REFERENCE_RUN]
    smoothing_length = geometry["smoothing_length"]
    columns = global_columns(reference["position"][:, 0], geometry["origin_x"],
                             smoothing_length, geometry["column_count"])
    signed = columns - geometry["cut_column"]
    distance = np.where(signed >= 0, signed, -1 - signed)
    fluid = np.isin(reference["material"], np.asarray(geometry["fluid_groups"]))

    crossing_ids = set()
    for run_name in TEST_RUNS:
        for migration_step, particle_id, _source, _destination in migrations.get(run_name, ()):
            if migration_step <= step:
                crossing_ids.add(particle_id)
    crossing_rows = np.flatnonzero(np.isin(common, np.fromiter(crossing_ids, dtype=np.int64,
                                                               count=len(crossing_ids))))
    flagged = flag_crossing_neighbourhood(reference["position"], crossing_rows,
                                          CROSSING_FLAG_RADIUS_IN_H * smoothing_length)
    crossing = np.zeros(common.size, dtype=bool)
    crossing[crossing_rows] = True
    groups = {"clean": fluid & ~flagged, "crossing_neighbourhood": fluid & flagged & ~crossing,
              "crossing": fluid & crossing}

    quantities = [name for name in REPORTED_QUANTITIES if name in reference]
    bin_count = int(distance[fluid].max()) + 1
    result = {"crossing_flag_radius_in_h": CROSSING_FLAG_RADIUS_IN_H,
              "particles": int(fluid.sum()), "crossing_particles": int(groups["crossing"].sum()),
              "crossing_neighbourhood": int(groups["crossing_neighbourhood"].sum()),
              "bins": bin_count, "runs": {}}
    noise_differences = {name: magnitude(name, aligned[NOISE_RUN][name] - reference[name])
                         for name in quantities}
    reference_magnitudes = {name: magnitude(name, reference[name]) for name in quantities}

    def summarize(differences: dict, include_relative: bool) -> dict:
        summary = {}
        for name in quantities:
            per_group = {}
            for group_name, group_mask in groups.items():
                per_bin = []
                for bin_index in range(bin_count):
                    mask = group_mask & (distance == bin_index)
                    test_values = differences[name][mask]
                    noise_values = noise_differences[name][mask]
                    entry = {"n": int(mask.sum()), "rms": rms(test_values),
                             "noise_rms": rms(noise_values),
                             "median": median(test_values), "noise_median": median(noise_values),
                             "max": float(test_values.max()) if test_values.size else float("nan"),
                             "differing_fraction": float(np.mean(test_values > 0))
                             if test_values.size else float("nan"),
                             "noise_differing_fraction": float(np.mean(noise_values > 0))
                             if noise_values.size else float("nan")}
                    entry["ratio"] = safe_ratio(entry["rms"], entry["noise_rms"])
                    entry["median_ratio"] = safe_ratio(entry["median"], entry["noise_median"])
                    if include_relative:
                        reference_values = reference_magnitudes[name][mask]
                        entry["reference_rms"] = rms(reference_values)
                        entry["relative_rms"] = (entry["rms"] / entry["reference_rms"]
                                                 if entry["reference_rms"] > 0 else float("nan"))
                        entry["relative_median"] = relative_median(test_values, reference_values)
                    per_bin.append(entry)
                per_group[group_name] = per_bin
            summary[name] = per_group
        return summary

    for run_name in COMPARED_RUNS:
        if run_name not in aligned:
            continue
        differences = {name: magnitude(name, aligned[run_name][name] - reference[name])
                       for name in quantities}
        result["runs"][run_name] = summarize(differences, include_relative=True)
    for first, second in TRIAL_PAIRS:
        if first in aligned and second in aligned:
            differences = {name: magnitude(name, aligned[second][name] - aligned[first][name])
                           for name in quantities}
            result["runs"][f"{first}~{second}"] = summarize(differences, include_relative=False)
    # Determinism of the column-0 difference: correlation of (t1 - ref) and
    # (t2 - ref) over the clean column-0 fluid particles.
    column_zero = groups["clean"] & (distance == 0)
    determinism = {}
    for first, second in TRIAL_PAIRS:
        if first not in aligned or second not in aligned:
            continue
        entry = {}
        for name in quantities:
            first_difference = (aligned[first][name] - reference[name])[column_zero].astype(np.float64).ravel()
            second_difference = (aligned[second][name] - reference[name])[column_zero].astype(np.float64).ravel()
            if first_difference.size > 1 and first_difference.std() > 0 and second_difference.std() > 0:
                entry[name] = float(np.corrcoef(first_difference, second_difference)[0, 1])
            else:
                entry[name] = None
        determinism[f"{first}~{second}"] = entry
    result["determinism_column0"] = determinism
    # Signed-column profile of acceleration (clean fluid particles).
    profile = {}
    window = int(max(np.abs(signed[fluid]).max(), 1))
    for run_name in TEST_RUNS + ("original", "k1_b"):
        if run_name not in aligned or "acceleration" not in quantities:
            continue
        test = magnitude("acceleration", aligned[run_name]["acceleration"] - reference["acceleration"])
        rows_out = []
        for column in range(-window, window):
            mask = groups["clean"] & (signed == column)
            if not mask.any():
                continue
            noise_value = rms(noise_differences["acceleration"][mask])
            rows_out.append({"column": column, "n": int(mask.sum()), "rms": rms(test[mask]),
                             "noise_rms": noise_value,
                             "ratio": safe_ratio(rms(test[mask]), noise_value)})
        profile[run_name] = rows_out
    result["acceleration_profile"] = profile

    # Step 1: CPU reconstruction of defect (1) for the clean column-0 particles.
    if step == 1 and snapshot is not None and "correction_inverse" in reference:
        self_rows = np.flatnonzero(column_zero)
        prediction = stale_ghost_prediction(reference, signed, self_rows, snapshot, geometry)
        reconstruction = {"statistics": prediction["statistics"],
                          "predicted_acceleration_rms": rms(magnitude("acceleration", prediction["acceleration"])),
                          "predicted_shift_rms": rms(magnitude("shift", prediction["shift"])),
                          "noise_acceleration_rms": rms(noise_differences["acceleration"][self_rows]),
                          "noise_shift_rms": rms(noise_differences["shift"][self_rows]),
                          "reference_acceleration_rms": rms(reference_magnitudes["acceleration"][self_rows]),
                          "reference_shift_rms": rms(reference_magnitudes["shift"][self_rows]),
                          "runs": {}}
        for run_name in TEST_RUNS:
            if run_name not in aligned:
                continue
            entry = {}
            for name in ("acceleration", "shift"):
                measured = (aligned[run_name][name][self_rows].astype(np.float64)
                            - reference[name][self_rows].astype(np.float64))
                predicted = prediction[name]
                residual = measured - predicted
                flat_measured, flat_predicted = measured.ravel(), predicted.ravel()
                correlation = (float(np.corrcoef(flat_measured, flat_predicted)[0, 1])
                               if flat_measured.std() > 0 and flat_predicted.std() > 0 else None)
                entry[name] = {"measured_rms": rms(magnitude(name, measured)),
                               "residual_rms": rms(magnitude(name, residual)),
                               "correlation": correlation}
            reconstruction["runs"][run_name] = entry
        result["stale_ghost_reconstruction"] = reconstruction
    return result


def case_geometry_from_directory(directory: pathlib.Path) -> dict:
    """Geometry block of original.json, or of any restart run's json (the
    long-horizon pass has no original run of its own)."""
    for path in [directory / "original.json"] + sorted(directory.glob("*_N*.json")):
        if path.exists():
            document = json.loads(path.read_text(encoding="utf-8"))
            if "geometry" in document:
                return document["geometry"]
    raise FileNotFoundError(f"{directory}: no run json with a geometry block")


def analyze_case(directory: pathlib.Path, snapshots, steps, snapshot_directory=None) -> dict:
    geometry = case_geometry_from_directory(directory)
    snapshot_directory = pathlib.Path(snapshot_directory or directory)
    analysis = {"geometry": geometry, "snapshots": {}}
    for snapshot_step in snapshots:
        migrations = {}
        run_documents = {}
        for run_name in COMPARED_RUNS + (REFERENCE_RUN,):
            path = run_json_path(directory, run_name, snapshot_step)
            if path.exists():
                document = json.loads(path.read_text(encoding="utf-8"))
                run_documents[run_name] = {key: document.get(key) for key in (
                    "valid", "must_be_zero", "migration_counts", "rows_per_slab", "cuts",
                    "worker_stamp_errors", "pool_health", "wall_time_s")}
                migrations[run_name] = [tuple(item) for item in document.get("migrations", [])]
        snapshot = None
        if snapshot_path(snapshot_directory, snapshot_step).exists():
            with np.load(snapshot_path(snapshot_directory, snapshot_step)) as archive:
                snapshot = {"velocity_mass": archive["velocity_mass"],
                            "density_pressure": archive["density_pressure"]}
                if not np.array_equal(archive["id"], np.arange(archive["id"].size, dtype=np.uint32)):
                    raise RuntimeError("snapshot ids are not 0..N-1 in order")
        per_step = {}
        for step in steps:
            print(f"{LOG_PREFIX} analyze {directory.name} N={snapshot_step} k={step}", flush=True)
            per_step[str(step)] = analyze_step(directory, snapshot_step, step, geometry,
                                               migrations, snapshot)
        analysis["snapshots"][str(snapshot_step)] = {"runs": run_documents, "steps": per_step}
        del snapshot
    return analysis


def run_analyze(arguments) -> int:
    root = pathlib.Path(arguments.out)
    case_names = [item for item in arguments.cases.split(",") if item] or [case["name"] for case in CASES]
    for case_name in case_names:
        directory = root / case_name
        if not directory.is_dir():
            print(f"{LOG_PREFIX} {case_name}: no directory, skipped", flush=True)
            continue
        snapshot_directory = getattr(arguments, "snapshot_root", "") or ""
        snapshot_directory = (pathlib.Path(snapshot_directory) / case_name.removesuffix("_long")
                              if snapshot_directory else None)
        analysis = analyze_case(directory, parse_integers(arguments.snapshots),
                                parse_integers(arguments.steps), snapshot_directory)
        save_json_atomically(root / f"analysis_{case_name}.json", analysis)
        print(f"{LOG_PREFIX} wrote {root / f'analysis_{case_name}.json'}", flush=True)
    return EXIT_VALID


# =============================================================================
# campaign driver
# =============================================================================

def run_child(command: list, environment: dict, log_path: pathlib.Path, timeout: float) -> dict:
    start = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT,
                                   env=environment, cwd=str(_REPOSITORY_ROOT))
        try:
            exit_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            exit_code = -9
    result_line = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(RESULT_PREFIX):
            result_line = line[len(RESULT_PREFIX):]
    return {"exit_code": exit_code, "seconds": time.perf_counter() - start,
            "result": json.loads(result_line) if result_line else None}


def run_long_campaign(arguments) -> int:
    """LONG_RUN_NAMES restarted from the main pass's snapshots and run to
    max(LONG_STEPS) steps, dumped at LONG_STEPS into <case>_long/."""
    root = pathlib.Path(arguments.out).resolve()
    case_names = [item for item in arguments.cases.split(",") if item]
    cases = [case for case in CASES if not case_names or case["name"] in case_names]
    base_environment = {key: value for key, value in os.environ.items()
                        if not key.startswith("V6_")}
    base_environment["VK_LOADER_LAYERS_DISABLE"] = "VK_LAYER_KHRONOS_validation"
    base_environment["PYTHONUNBUFFERED"] = "1"
    steps = ",".join(str(step) for step in LONG_STEPS)
    ledger_path = root / "campaign_ledger.jsonl"
    all_valid = True
    for case in cases:
        source = root / case["name"]
        directory = root / f"{case['name']}_long"
        directory.mkdir(parents=True, exist_ok=True)
        common = ["--case", case["path"], "--out-dir", str(directory), "--snapshot-dir", str(source),
                  "--step1-window", str(case["long_window"]), "--window", str(case["long_window"]),
                  "--steps", steps]
        for snapshot_step in SNAPSHOT_STEPS:
            if not snapshot_path(source, snapshot_step).exists():
                print(f"{LOG_PREFIX} [{case['name']}] no snapshot N={snapshot_step}, skipped", flush=True)
                all_valid = False
                continue
            for run_name, slab_count, device_map, environment, shuffle_seed in RESTART_RUNS:
                if run_name not in LONG_RUN_NAMES:
                    continue
                if run_json_path(directory, run_name, snapshot_step).exists() and not arguments.force:
                    continue
                command = [sys.executable, "-m", "experiment.seam_audit.single_step", "restart",
                           "--run-name", run_name, "--slabs", str(slab_count),
                           "--device-map", device_map, "--snapshot-step", str(snapshot_step),
                           "--shuffle-seed", str(shuffle_seed)] + common
                child_environment = dict(base_environment)
                child_environment.update(environment)
                print(f"{LOG_PREFIX} [{case['name']}_long] N={snapshot_step} {run_name}", flush=True)
                outcome = run_child(command, child_environment,
                                    directory / f"{run_name}_N{snapshot_step}.log", arguments.timeout)
                with open(ledger_path, "a", encoding="utf-8") as ledger:
                    ledger.write(json.dumps({"case": f"{case['name']}_long", "snapshot": snapshot_step,
                                             "run": run_name, **outcome},
                                            default=json_default) + "\n")
                print(f"{LOG_PREFIX}   exit {outcome['exit_code']} in {outcome['seconds']:.0f} s",
                      flush=True)
                all_valid &= outcome["exit_code"] == 0
    if not arguments.skip_analysis:
        run_analyze(argparse.Namespace(
            out=str(root), cases=",".join(f"{case['name']}_long" for case in cases),
            snapshots=",".join(str(step) for step in SNAPSHOT_STEPS), steps=steps,
            snapshot_root=str(root)))
    return EXIT_VALID if all_valid else EXIT_INVALID


def run_campaign(arguments) -> int:
    if arguments.long:
        return run_long_campaign(arguments)
    root = pathlib.Path(arguments.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    case_names = [item for item in arguments.cases.split(",") if item]
    cases = [case for case in CASES if not case_names or case["name"] in case_names]
    runs = [run for run in RESTART_RUNS
            if not arguments.runs or run[0] in arguments.runs.split(",")]
    base_environment = {key: value for key, value in os.environ.items()
                        if not key.startswith("V6_")}
    base_environment["VK_LOADER_LAYERS_DISABLE"] = "VK_LAYER_KHRONOS_validation"
    base_environment["PYTHONUNBUFFERED"] = "1"
    python = sys.executable
    ledger_path = root / "campaign_ledger.jsonl"
    snapshots = ",".join(str(step) for step in SNAPSHOT_STEPS)
    steps = ",".join(str(step) for step in STEPS_AFTER_RESTART)
    all_valid = True
    for case in cases:
        directory = root / case["name"]
        directory.mkdir(parents=True, exist_ok=True)
        common = ["--case", case["path"], "--out-dir", str(directory),
                  "--step1-window", str(case["step1_window"]), "--window", str(case["window"]),
                  "--steps", steps]
        if not (directory / "original.json").exists() or arguments.force:
            command = [python, "-m", "experiment.seam_audit.single_step", "original",
                       "--device", REFERENCE_DEVICE, "--snapshots", snapshots] + common
            print(f"{LOG_PREFIX} [{case['name']}] original", flush=True)
            outcome = run_child(command, dict(base_environment), directory / "original.log",
                                arguments.timeout)
            with open(ledger_path, "a", encoding="utf-8") as ledger:
                ledger.write(json.dumps({"case": case["name"], "run": "original", **outcome},
                                        default=json_default) + "\n")
            print(f"{LOG_PREFIX}   exit {outcome['exit_code']} in {outcome['seconds']:.0f} s",
                  flush=True)
            if outcome["exit_code"] != 0:
                all_valid = False
                continue
        for snapshot_step in SNAPSHOT_STEPS:
            for run_name, slab_count, device_map, environment, shuffle_seed in runs:
                if run_json_path(directory, run_name, snapshot_step).exists() and not arguments.force:
                    continue
                command = [python, "-m", "experiment.seam_audit.single_step", "restart",
                           "--run-name", run_name, "--slabs", str(slab_count),
                           "--device-map", device_map, "--snapshot-step", str(snapshot_step),
                           "--shuffle-seed", str(shuffle_seed)] + common
                child_environment = dict(base_environment)
                child_environment.update(environment)
                print(f"{LOG_PREFIX} [{case['name']}] N={snapshot_step} {run_name}", flush=True)
                outcome = run_child(command, child_environment,
                                    directory / f"{run_name}_N{snapshot_step}.log", arguments.timeout)
                with open(ledger_path, "a", encoding="utf-8") as ledger:
                    ledger.write(json.dumps({"case": case["name"], "snapshot": snapshot_step,
                                             "run": run_name, **outcome},
                                            default=json_default) + "\n")
                print(f"{LOG_PREFIX}   exit {outcome['exit_code']} in {outcome['seconds']:.0f} s",
                      flush=True)
                all_valid &= outcome["exit_code"] == 0
    if not arguments.skip_analysis:
        run_analyze(argparse.Namespace(out=str(root), cases=",".join(case["name"] for case in cases),
                                       snapshots=snapshots, steps=steps))
    return EXIT_VALID if all_valid else EXIT_INVALID


# =============================================================================
# command line
# =============================================================================

def parse_arguments(argument_list=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def add_run_options(command_parser) -> None:
        command_parser.add_argument("--case", required=True)
        command_parser.add_argument("--out-dir", required=True)
        command_parser.add_argument("--steps", default=",".join(map(str, STEPS_AFTER_RESTART)))
        command_parser.add_argument("--step1-window", type=int, default=12)
        command_parser.add_argument("--window", type=int, default=12)
        command_parser.add_argument("--pool-safety", type=float, default=1.2)
        command_parser.add_argument("--sync-scheme", default="per-direction",
                                    choices=["aggregated", "per-direction"])
        command_parser.add_argument("--stall-timeout", type=float, default=120.0)

    original = commands.add_parser("original", help="K=1 from t=0: snapshots + continuation")
    add_run_options(original)
    original.add_argument("--device", default=REFERENCE_DEVICE)
    original.add_argument("--snapshots", default=",".join(map(str, SNAPSHOT_STEPS)))
    original.add_argument("--depth", type=int, default=2)

    restart = commands.add_parser("restart", help="K slabs from one snapshot")
    add_run_options(restart)
    restart.add_argument("--run-name", required=True)
    restart.add_argument("--slabs", type=int, required=True)
    restart.add_argument("--device-map", default="0,1")
    restart.add_argument("--snapshot-step", type=int, required=True)
    restart.add_argument("--shuffle-seed", type=int, default=0)
    restart.add_argument("--migration-tracking", default="staging",
                         choices=["staging", "ownership"],
                         help="how K>1 runs record slab changes per frame: migrant packets in "
                              "the sender staging (default, O(migrants)) or a readback of every "
                              "own particle's id (the main pass of 2026-10-02)")
    restart.add_argument("--snapshot-dir", default="",
                         help="directory of snapshot_N<step>.npz (default: --out-dir)")

    analyze = commands.add_parser("analyze", help="statistics from the dumps")
    analyze.add_argument("--out", required=True)
    analyze.add_argument("--cases", default="")
    analyze.add_argument("--snapshots", default=",".join(map(str, SNAPSHOT_STEPS)))
    analyze.add_argument("--steps", default=",".join(map(str, STEPS_AFTER_RESTART)))
    analyze.add_argument("--snapshot-root", default="",
                         help="root holding <case>/snapshot_N*.npz when the analysed "
                              "directories have none (long-horizon pass)")

    campaign = commands.add_parser("campaign", help="run the matrix + analysis")
    campaign.add_argument("--out", required=True)
    campaign.add_argument("--cases", default="")
    campaign.add_argument("--runs", default="")
    campaign.add_argument("--timeout", type=float, default=3600.0)
    campaign.add_argument("--force", action="store_true")
    campaign.add_argument("--skip-analysis", action="store_true")
    campaign.add_argument("--long", action="store_true",
                          help="long-horizon pass (LONG_RUN_NAMES to LONG_STEPS) into <case>_long/")
    return parser.parse_args(argument_list)


def main(argument_list=None) -> int:
    arguments = parse_arguments(argument_list)
    if arguments.command == "analyze":
        return run_analyze(arguments)
    if arguments.command == "campaign":
        return run_campaign(arguments)
    summary = {"command": arguments.command, "case": arguments.case,
               "run_name": getattr(arguments, "run_name", "original"), "valid": False,
               "error": None}
    try:
        if arguments.command == "original":
            exit_code = run_original(arguments, summary)
        else:
            exit_code = run_restart(arguments, summary)
    except (Exception, KeyboardInterrupt) as error:
        # Same policy as dump_state: report and leave through os._exit, no
        # Vulkan teardown (frames still in flight would block it forever).
        traceback.print_exc()
        summary["error"] = f"{type(error).__name__}: {error}"
        print(RESULT_PREFIX + json.dumps(summary, default=json_default), flush=True)
        sys.stderr.flush()
        os._exit(EXIT_ERROR)
    print(RESULT_PREFIX + json.dumps(summary, default=json_default), flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
