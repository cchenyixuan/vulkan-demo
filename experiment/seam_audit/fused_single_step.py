"""
fused_single_step.py - E39 B1 single-step check of V7_FUSED_CORRECTION_DENSITY (experiment/v7): from the SAME
developed input state, ONE step with the separate correction / density kernels (the recording of
V7_FUSED_CORRECTION_DENSITY=0) and ONE step with the fused kernel (correction_density.comp), compared per particle.

Input state: a saved step boundary in experiment/seam_audit/single_step.py's snapshot format (the nine set-0 restart
fields of every particle + id, rho absolute), --state PATH; or --develop N: the case's initial state run N steps with
the code defaults (+ --env) at K = 1 on the first device of --device-map, saved to --save-state PATH when given.

Method (one process, so that both steps see the same device state, voxel lists and ghost pools):
  1. the chain of K = len(--device-map) slabs (equal weights) is built with the code defaults + --env (every slab must
     resolve fused; a fused slab builds the fused and the separate pipelines) and restarted from the state
     (ChainOrchestratorV7.restart_all: upload, voxel lists, one defrag);
  2. every device-local buffer of every slab is read back: the input state, byte for byte;
  3. step A: every buffer is restored from (2), every slab records the separate kernels and runs one frame
     (--path chain: phases A / B / C with the transport; --path single, K = 1: the single-cmd step); every buffer
     is read back;
  4. step B: the same with the fused kernels (the next frame number: the timelines only increase);
  5. the inputs of the correction / density passes after phase A and the transport must be the same in A and B:
     every voxel list (own and ghost) holds the same particles in the same order, a particle identified by its
     position bits (atomic slot allocations - ghost_send's replica blocks, install_migrations' migrant pids, the
     departed pool - may give a particle another pool slot in the other step, so rows are paired through the
     lists, not by pid), with the same velocity, mass, material and (own and outer ghost) stored density; a
     difference voids the comparison. Then the outputs are compared on the rows the passes wrote: the own alive
     particles (interior / band column, fluid / wall) and, with two ghost layers, the particles listed in the inner
     ghost column (recomputed as self by the band pass): correction_inverse (6 entries), kernel sum and density
     gradient - expected bit-identical; rho and P in density_pressure_scratch and density_pressure (after the
     copy) - summation rounding. E39 B4: a deep wall a kernel skipped (the decision record, which must be the same
     in both steps) keeps its previous L / gradient / kernel sum: those rows are left out of the three fields
     (counted), their scratch is compared. acceleration / shift (force, downstream) are reported.
  ULP distance: |ordered(a) - ordered(b)| on the float32 bit patterns (+0 and -0 at distance 0; bit-identical rows are
  counted separately); per row the largest over the field's components; relative difference |a - b| / max(|a|, |b|)
  per component, maximum over the components of the row.

Usage (from the worktree root, by path):
    .venv/Scripts/python.exe experiment/seam_audit/fused_single_step.py --case cases/lid_driven_cavity_2d_gen/case.yaml \\
        --state ../vulkan-demo/logs/seam_audit/single_step/cavity2d_1m/snapshot_N2000.npz --device-map 1 \\
        --out logs/e39/b1/single_step/k1_2d1m.json
    ... --device-map 0,1 (K = 2) | --env V7_CASCADE_FORCE=0 | --env V7_DELTA_DENSITY=1 | --path single (K = 1)
    ... --case cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025_adami/case.yaml --develop 2000 \\
        --save-state logs/e39/b1/single_step/n250_adami_N2000.npz --device-map 1 --out logs/e39/b1/single_step/adami.json
Exit 0 = L / kernel sum / density gradient bit-identical on every compared row, inputs identical, invariants ok;
1 = a field or the inputs differ; 2 = no fused slab or bad arguments; 3 = an invariant (alive, overflow, stamps).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

LOG_PREFIX = "[fused_single_step]"
EXIT_PASS, EXIT_DIFFERENT, EXIT_ERROR, EXIT_INVALID = 0, 1, 2, 3
# the nine set-0 fields of a step boundary (SphSimulatorV7.RESTART_FIELD_LAYOUT, single_step.py's snapshots)
STATE_FIELDS = ("position_voxel_id", "velocity_mass", "density_pressure", "acceleration", "shift", "material",
                "correction_inverse", "density_gradient_kernel_sum", "extension_fields")
# compared fields: name -> (buffer, components per pool slot, compared columns, written by correction)
FIELDS = {
    "correction_inverse": ("correction_inverse", 8, (0, 1, 2, 3, 4, 5), True),   # (m00 m11 m22 m01) (m02 m12 - -)
    "kernel_sum": ("density_gradient_kernel_sum", 4, (3,), True),
    "density_gradient": ("density_gradient_kernel_sum", 4, (0, 1, 2), True),
    "scratch_density": ("density_pressure_scratch", 2, (0,), False),
    "scratch_pressure": ("density_pressure_scratch", 2, (1,), False),
    "primary_density": ("density_pressure", 2, (0,), False),
    "primary_pressure": ("density_pressure", 2, (1,), False),
    "acceleration": ("acceleration", 4, (0, 1, 2), False),
    "shift": ("shift", 4, (0, 1, 2), False),
}
BIT_IDENTICAL_FIELDS = ("correction_inverse", "kernel_sum", "density_gradient")
DOWNSTREAM_FIELDS = ("acceleration", "shift")
BOUNDARY_KIND = 1                       # common.glsl MATERIAL_BOUNDARY
DEEP_WALL_KERNELS = ("correction", "density")


def ordered_integers(values: np.ndarray) -> np.ndarray:
    """float32 bit patterns as int64 that order like the floats (+0 and -0 both 0)."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.int32).astype(np.int64)
    return np.where(bits < 0, -(bits & 0x7FFFFFFF), bits)


def row_statistics(first: np.ndarray, second: np.ndarray) -> dict:
    """first / second: (rows, components) float32 of the two steps. Per row: bit identity, the largest component
    ULP distance, absolute and relative difference; then counts, percentiles and maxima over the rows."""
    rows = int(first.shape[0])
    if rows == 0:
        return {"rows": 0}
    identical = (first.view(np.uint32) == second.view(np.uint32)).all(axis=1)
    ulp = np.abs(ordered_integers(first) - ordered_integers(second)).max(axis=1)
    first64, second64 = first.astype(np.float64), second.astype(np.float64)
    finite = np.isfinite(first64).all(axis=1) & np.isfinite(second64).all(axis=1)
    difference = np.abs(first64 - second64)
    scale = np.maximum(np.abs(first64), np.abs(second64))
    relative = np.divide(difference, scale, out=np.zeros_like(difference), where=scale > 0)
    difference_row, relative_row = difference.max(axis=1), relative.max(axis=1)
    worst = int(np.argmax(np.where(finite, ulp, -1)))
    return {"rows": rows, "bit_identical": int(identical.sum()), "max_ulp": int(ulp[finite].max()) if finite.any() else None,
            "ulp_0": int((ulp == 0).sum()), "ulp_1": int((ulp == 1).sum()), "ulp_2": int((ulp == 2).sum()),
            "ulp_3_or_more": int((ulp >= 3).sum()),
            "ulp_p50": float(np.percentile(ulp, 50)), "ulp_p99": float(np.percentile(ulp, 99)),
            "ulp_p99_9": float(np.percentile(ulp, 99.9)),
            "max_abs": float(difference_row[finite].max()) if finite.any() else None,
            "max_rel": float(relative_row[finite].max()) if finite.any() else None,
            "non_finite_rows": int((~finite).sum()), "worst_row": worst,
            "worst_values": [first[worst].tolist(), second[worst].tolist()]}


def field_rows(buffers: dict, capacity: int, field: str, rows: np.ndarray) -> np.ndarray:
    buffer, components, columns, _ = FIELDS[field]
    values = np.frombuffer(buffers[buffer], dtype=np.float32)[:capacity * components].reshape(capacity, components)
    return np.ascontiguousarray(values[rows][:, list(columns)])


def list_entries(sim, buffers: dict) -> tuple[np.ndarray, np.ndarray]:
    """Every voxel-list entry of one slab, own and ghost voxels, in (voxel id, slot) order: (voxel ids, pids)."""
    slots = sim.case.capacities.max_particles_per_voxel
    voxel_count = sim.case.grid.total_voxel_count()
    counts = np.frombuffer(buffers["inside_particle_count"], np.uint32)[:voxel_count + 1].astype(np.int64)
    index = np.frombuffer(buffers["inside_particle_index"], np.uint32)[:(voxel_count + 1) * slots]
    filled = np.arange(slots)[None, :] < np.minimum(counts, slots)[:, None]
    voxel_ids = np.nonzero(filled)[0].astype(np.int64)
    pids = index.reshape(voxel_count + 1, slots)[filled].astype(np.int64)
    return voxel_ids, pids


def input_differences_and_pairing(sim, first: dict, second: dict) -> tuple[list, np.ndarray]:
    """Compare what the correction / density passes of one slab read in the two steps, through the voxel lists:
    per voxel the same count, per list entry the same particle (position bits) with the same velocity, mass and
    material, and (outer ghost entries: the uploaded rho_n; own and inner ghost rows hold rho_{n+1} after the copy)
    the same stored density. Returns the differences and pair[pid in A] = pid in B (-1 = unlisted)."""
    capacity = sim.case.capacities.total_pool_capacity()
    voxel_count = sim.case.grid.total_voxel_count()
    differences = []
    first_counts = np.frombuffer(first["inside_particle_count"], np.uint32)[:voxel_count + 1]
    second_counts = np.frombuffer(second["inside_particle_count"], np.uint32)[:voxel_count + 1]
    pair = np.full(capacity, -1, dtype=np.int64)
    if not np.array_equal(first_counts, second_counts):
        differences.append(f"inside_particle_count ({int((first_counts != second_counts).sum())} voxels)")
        return differences, pair
    voxel_ids, first_pids = list_entries(sim, first)
    _, second_pids = list_entries(sim, second)

    def view(buffers, name, components, dtype=np.uint32):
        return np.frombuffer(buffers[name], dtype)[:capacity * components].reshape(capacity, components)

    for name, components, columns in (("position_voxel_id", 4, (0, 1, 2)), ("velocity_mass", 4, (0, 1, 2, 3)),
                                      ("material", 1, (0,))):
        first_values = view(first, name, components)[first_pids][:, list(columns)]
        second_values = view(second, name, components)[second_pids][:, list(columns)]
        different = (first_values != second_values).any(axis=1)
        if different.any():
            differences.append(f"{name} of {int(different.sum())} list entries")
    face = sim.case.grid.grid_dimension_y * sim.case.grid.grid_dimension_z
    leading_x = sim.case.ghost_grid.leading_ghost_voxel_count // face
    trailing_x = sim.case.ghost_grid.trailing_ghost_voxel_count // face
    own_last_x = sim.case.grid.grid_dimension_x - 1 - trailing_x
    column = (voxel_ids - 1) // face
    outer_ghost = ((column < leading_x - 1) if leading_x > 1 else np.zeros(column.shape, dtype=bool)) \
        | ((column > own_last_x + 1) if trailing_x > 1 else np.zeros(column.shape, dtype=bool))
    if outer_ghost.any():
        first_density = view(first, "density_pressure", 2)[first_pids[outer_ghost], 0]
        second_density = view(second, "density_pressure", 2)[second_pids[outer_ghost], 0]
        if not np.array_equal(first_density, second_density):
            differences.append(f"outer ghost rho_n of {int((first_density != second_density).sum())} entries")
    if np.unique(first_pids).size != first_pids.size:
        differences.append("a pid listed twice")
    pair[first_pids] = second_pids
    return differences, pair


def slab_rows(sim, buffers: dict) -> dict:
    """The rows (step A pids) the correction / density passes of one slab wrote: own alive particles by column
    (interior / band of the slab's correction = density band width) and the particles listed in the inner ghost
    column(s) (the band pass's self rows, two ghost layers), each split into fluid (any kind but BOUNDARY) and
    wall."""
    case = sim.case
    capacity = case.capacities.total_pool_capacity()
    material = np.frombuffer(buffers["material"], np.uint32)[:capacity]
    kinds = np.array([material_entry.kind for material_entry in case.materials])
    face = case.grid.grid_dimension_y * case.grid.grid_dimension_z
    leading_x = case.ghost_grid.leading_ghost_voxel_count // face
    trailing_x = case.ghost_grid.trailing_ghost_voxel_count // face
    own_last_x = case.grid.grid_dimension_x - 1 - trailing_x
    band_width = sim.band_widths[0]
    voxel_ids, pids = list_entries(sim, buffers)
    column = (voxel_ids - 1) // face
    own = (column >= leading_x) & (column <= own_last_x)
    band = own & (((leading_x > 0) & (column < leading_x + band_width))
                  | ((trailing_x > 0) & (column > own_last_x - band_width)))
    inner_ghost = np.zeros(column.shape, dtype=bool)
    if sim._ghost_self_layer(2, 1, "correction"):
        inner_ghost = ((leading_x > 0) & (column == leading_x - 1)) | ((trailing_x > 0) & (column == own_last_x + 1))
    groups = {"interior": pids[own & ~band], "band": pids[band], "ghost_self": pids[inner_ghost]}
    out = {}
    for name, rows in groups.items():
        is_wall = kinds[material[rows]] == BOUNDARY_KIND if rows.size else np.zeros(0, dtype=bool)
        out[f"{name}_fluid"] = rows[~is_wall]
        out[f"{name}_wall"] = rows[is_wall]
    return out


def deep_wall_skipped(sim, buffers: dict, status: dict) -> dict:
    """Per kernel, the pids the B4 variants skipped in this step (record == frame_stamp + 1); empty without B4."""
    if not sim._deep_wall_skip_active() or buffers.get("deep_wall_skip_record") is None:
        return {kernel: np.zeros(0, dtype=np.int64) for kernel in DEEP_WALL_KERNELS}
    capacity = sim.case.capacities.total_pool_capacity()
    record = np.frombuffer(buffers["deep_wall_skip_record"], np.uint32)
    if record.size < 2 * capacity:
        raise RuntimeError("deep_wall_skip_record is the 16-byte stub: the decisions were not recorded")
    record = record[:2 * capacity].reshape(capacity, 2)
    stamp = np.uint32(int(status["frame_stamp"]) + 1)
    return {kernel: np.flatnonzero(record[:, index] == stamp) for index, kernel in enumerate(DEEP_WALL_KERNELS)}


def global_status(sim, buffers: dict) -> dict:
    import struct
    names = sys.modules[type(sim).__module__]._GLOBAL_STATUS_FIELD_NAMES
    fields = struct.unpack("<I f " + "I " * 38, buffers["global_status"][:160])
    return {name: fields[index] for index, name in enumerate(names)}


def invariant_problems(status: dict, expected_alive: int) -> list:
    problems = []
    if status["alive_particle_count"] != expected_alive:
        problems.append(f"alive {status['alive_particle_count']} != {expected_alive}")
    problems += [f"{key}={value}" for key, value in status.items()
                 if key.startswith("overflow_") and value]
    problems += [f"{key}={status[key]}" for key in ("stamp_error_count", "far_migration_count") if status.get(key)]
    return problems


def develop_state(solver, global_case, device: int, steps: int) -> dict:
    """The case's initial state run ``steps`` steps at K = 1 (bootstrap + the default step loop); own alive rows of
    the nine restart fields, rho absolute."""
    chain = solver["compute_chain_partition"](global_case, [1.0], None)
    context = solver["Context"].create(device_index=device, enable_validation=False,
                                       application_name="fused_single_step_develop")
    sim = solver["Simulator"](context, chain.slabs[0], sync_scheme="per-direction")
    orchestrator = solver["Orchestrator"]([sim], defrag_cadence=int(global_case.numerics.defrag_cadence))
    try:
        orchestrator.bootstrap_all()
        result = orchestrator.run_pipelined(steps, depth=2)
        status = sim.readback_global_status()
        problems = invariant_problems(status, int(global_case.initial.positions.shape[0]))
        if problems:
            raise RuntimeError(f"developing the state: {problems}")
        raw = sim.readback_buffers_batch(list(STATE_FIELDS), density="absolute")
        capacity = sim.case.capacities.total_pool_capacity()
        first, stop = sim.own_first_pid(), sim.own_first_pid() + sim.case.capacities.own_pool_size
        layout = solver["Simulator"].RESTART_FIELD_LAYOUT
        state = {}
        for name, (components, dtype) in layout.items():
            flat = np.frombuffer(raw[name], dtype=dtype)[:capacity * components]
            state[name] = np.array((flat.reshape(capacity, components) if components > 1 else flat)[first:stop])
        alive = (state["velocity_mass"][:, 3] > 0) & (state["position_voxel_id"][:, 3] > 0.5)
        state = {name: np.ascontiguousarray(values[alive]) for name, values in state.items()}
        state["id"] = np.arange(int(alive.sum()), dtype=np.uint32)
        print(f"{LOG_PREFIX} developed {steps} steps ({result.get('fps', 0):.0f} fps): {int(alive.sum()):,} particles",
              flush=True)
        return state
    finally:
        orchestrator.destroy()
        sim.destroy()
        context.destroy()


def run_one_step(sims, orchestrator, frame: int, path: str, stall_timeout: float) -> None:
    if path == "single":
        sims[0].prepare_step_single_cmd_buffer()
        sims[0].submit_step_single_and_wait()
        return
    for sim in sims:
        sim.prepare_step_cmd_buffers()
    orchestrator._submit_frame(frame)
    orchestrator._frame_count = frame + 1
    orchestrator._wait_frame(frame, stall_timeout)


def restore(sim, saved: dict) -> None:
    for name, payload in saved.items():
        sim._staging_upload(sim.buffers[name], payload)


def markdown_table(document: dict) -> str:
    lines = ["| category | field | rows | bit-identical | max ULP | 0 / 1 / 2 / >=3 ULP | p50 / p99 / p99.9 | "
             "max abs | max rel |", "|---|---|---|---|---|---|---|---|---|"]
    for category, fields in document["statistics"].items():
        for field, item in fields.items():
            if not item.get("rows"):
                continue
            lines.append(f"| {category} | {field} | {item['rows']:,} | {item['bit_identical']:,} | {item['max_ulp']} | "
                         f"{item['ulp_0']:,} / {item['ulp_1']:,} / {item['ulp_2']:,} / {item['ulp_3_or_more']:,} | "
                         f"{item['ulp_p50']:g} / {item['ulp_p99']:g} / {item['ulp_p99_9']:g} | "
                         f"{item['max_abs']:.3e} | {item['max_rel']:.3e} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True)
    parser.add_argument("--state", help="saved step boundary (single_step.py snapshot format)")
    parser.add_argument("--develop", type=int, default=0, help="develop the state from the initial one: N steps, K = 1")
    parser.add_argument("--save-state", help="with --develop: save the developed state here (npz)")
    parser.add_argument("--device-map", default="1", help="one physical device per slab (K = entries)")
    parser.add_argument("--path", choices=("chain", "single"), default="chain",
                        help="chain: phases A / B / C (any K); single: the K = 1 single-cmd step")
    parser.add_argument("--env", action="append", default=[], metavar="V7_X=VALUE",
                        help="V7_* switch on top of the code defaults (repeatable)")
    parser.add_argument("--pool-safety", type=float, default=0.0, help="chain partition pool safety (0 = default)")
    parser.add_argument("--stall-timeout", type=float, default=120.0)
    parser.add_argument("--out", required=True, help="JSON result (a .md table is written next to it)")
    arguments = parser.parse_args()
    if bool(arguments.state) == bool(arguments.develop):
        parser.error("give exactly one of --state and --develop")
    device_map = [int(device) for device in arguments.device_map.split(",")]
    if arguments.path == "single" and len(device_map) != 1:
        parser.error("--path single needs K = 1")
    overrides = {}
    for item in arguments.env:
        key, separator, value = item.partition("=")
        if not separator or not key.startswith("V7_"):
            parser.error(f"--env takes V7_X=VALUE, got {item!r}")
        overrides[key] = value
    if overrides.get("V7_FUSED_CORRECTION_DENSITY", "1").strip() != "1":
        parser.error("the tool compares the fused kernel with the separate ones: V7_FUSED_CORRECTION_DENSITY must stay 1")

    # the switches are read at the solver's import
    for key in [key for key in os.environ if key.startswith("V7_")]:
        del os.environ[key]
    os.environ.update(overrides)
    os.environ["VK_LOADER_LAYERS_DISABLE"] = "VK_LAYER_KHRONOS_validation"
    from experiment.v7.utils import simulator_v7                                  # noqa: E402
    from experiment.v7.utils.case_loader_v7 import load_case_v7                   # noqa: E402
    from experiment.v7.utils.orchestrator_v7 import ChainOrchestratorV7           # noqa: E402
    from experiment.v7.utils.partition_v7 import compute_chain_partition, restart_slab_rows   # noqa: E402
    from experiment.v7.utils.vulkan_context_v7 import VulkanContextV7             # noqa: E402
    solver = {"Simulator": simulator_v7.SphSimulatorV7, "Orchestrator": ChainOrchestratorV7,
              "Context": VulkanContextV7, "compute_chain_partition": compute_chain_partition}
    # E39 B4: record the deep-wall decisions (the test hook canonical_dump uses), read when the simulators are built
    simulator_v7._DEEP_WALL_RECORD_DECISIONS = True
    start = time.perf_counter()

    global_case = load_case_v7(arguments.case)
    pool_safety = arguments.pool_safety or None
    if arguments.develop:
        state = develop_state(solver, global_case, device_map[0], arguments.develop)
        if arguments.save_state:
            pathlib.Path(arguments.save_state).parent.mkdir(parents=True, exist_ok=True)
            np.savez(arguments.save_state, **state)
            print(f"{LOG_PREFIX} saved the developed state -> {arguments.save_state}", flush=True)
        state_source = f"--develop {arguments.develop}"
    else:
        with np.load(arguments.state) as archive:
            state = {name: archive[name] for name in STATE_FIELDS}
        state_source = str(arguments.state)
    particle_count = int(state["position_voxel_id"].shape[0])

    chain = compute_chain_partition(global_case, [1.0] * len(device_map), pool_safety)
    rows_per_slab = restart_slab_rows(global_case, chain, state["position_voxel_id"][:, 0])
    states = [{name: np.ascontiguousarray(state[name][rows]) for name in STATE_FIELDS} for rows in rows_per_slab]
    del state
    contexts, sims = [], []
    for index, slab in enumerate(chain.slabs):
        contexts.append(VulkanContextV7.create(device_index=device_map[index], enable_validation=False,
                                               application_name=f"fused_single_step_s{index}"))
        sims.append(simulator_v7.SphSimulatorV7(contexts[-1], slab, sync_scheme="per-direction"))
    orchestrator = ChainOrchestratorV7(sims, defrag_cadence=10 ** 12)
    try:
        resolutions = [sim.fused_correction_density_resolution for sim in sims]
        if not all(active for active, _ in resolutions):
            print(f"{LOG_PREFIX} a slab does not fuse, nothing to compare: {resolutions}", flush=True)
            return EXIT_ERROR
        orchestrator.restart_all(states, 0.0)       # the state holds absolute rho
        del states
        saved = [sim.readback_buffers_batch(list(sim.buffers), density="stored") for sim in sims]
        steps, statuses = {}, {}
        frame = 0
        for label, fused in (("separate", False), ("fused", True)):
            for sim, saved_buffers, resolution in zip(sims, saved, resolutions):
                restore(sim, saved_buffers)
                sim.fused_correction_density_resolution = ((True, resolution[1]) if fused
                                                           else (False, "fused_single_step: the separate kernels"))
            run_one_step(sims, orchestrator, frame, arguments.path, arguments.stall_timeout)
            frame += 1
            steps[label] = [sim.readback_buffers_batch(list(sim.buffers), density="stored") for sim in sims]
            statuses[label] = [global_status(sim, buffers) for sim, buffers in zip(sims, steps[label])]
            print(f"{LOG_PREFIX} step with the {label} kernels done", flush=True)
        for sim, resolution in zip(sims, resolutions):
            sim.fused_correction_density_resolution = resolution

        # ---- invariants and inputs
        problems = {label: [f"slab {index}: {problem}" for index, status in enumerate(statuses[label])
                            for problem in invariant_problems(status, int(rows_per_slab[index].size))]
                    for label in steps}
        status_differences = [f"slab {index}: {key} {first[key]} vs {second[key]}"
                              for index, (first, second) in enumerate(zip(statuses["separate"], statuses["fused"]))
                              for key in first if first[key] != second[key]]
        input_differences = []
        deep_wall = []
        statistics: dict = {}
        excluded = {}
        for index, sim in enumerate(sims):
            first, second = steps["separate"][index], steps["fused"][index]
            capacity = sim.case.capacities.total_pool_capacity()
            differences, pair = input_differences_and_pairing(sim, first, second)
            input_differences += [f"slab {index}: {difference}" for difference in differences]
            if differences:
                continue
            skipped = {label: deep_wall_skipped(sim, steps[label][index], statuses[label][index]) for label in steps}
            same_decisions = all(np.array_equal(np.sort(pair[skipped["separate"][kernel]]),
                                                np.sort(skipped["fused"][kernel]))
                                 and (pair[skipped["separate"][kernel]] >= 0).all() for kernel in DEEP_WALL_KERNELS)
            deep_wall.append({"active": bool(sim._deep_wall_skip_active()),
                              "skipped": {kernel: int(skipped["fused"][kernel].size) for kernel in DEEP_WALL_KERNELS},
                              "same_rows_in_both_steps": bool(same_decisions)})
            not_written = skipped["separate"]["correction"]
            for category, pids in slab_rows(sim, first).items():
                written = pids[~np.isin(pids, not_written)]
                excluded[category] = excluded.get(category, 0) + int(pids.size - written.size)
                for field in FIELDS:
                    rows = written if FIELDS[field][3] else pids
                    statistics.setdefault(category, {}).setdefault(field, []).append(
                        (field_rows(first, capacity, field, rows), field_rows(second, capacity, field, pair[rows])))
        merged = {}
        for category, fields in statistics.items():
            merged[category] = {}
            for field, parts in fields.items():
                first_rows = np.concatenate([part[0] for part in parts])
                second_rows = np.concatenate([part[1] for part in parts])
                merged[category][field] = row_statistics(first_rows, second_rows)
        bit_identical = all(item.get("rows", 0) == item.get("bit_identical", 0)
                            for fields in merged.values() for field, item in fields.items()
                            if field in BIT_IDENTICAL_FIELDS)
        verdict_invalid = any(problems.values())
        verdict_pass = (bit_identical and not input_differences and not status_differences
                        and all(entry["same_rows_in_both_steps"] for entry in deep_wall) and not verdict_invalid)
        document = {
            "tool": "experiment/seam_audit/fused_single_step.py", "case": arguments.case, "state": state_source,
            "particles": particle_count, "device_map": device_map, "path": arguments.path,
            "slabs": len(sims), "cuts": [int(cut) for cut in chain.cuts],
            "environment": {key: value for key, value in os.environ.items() if key.startswith("V7_")},
            "switches": simulator_v7.configured_v7_switches(),
            "resolutions": [list(resolution) for resolution in resolutions],
            "band_widths": [list(sim.band_widths) for sim in sims],
            "invariant_problems": problems, "global_status_differences": status_differences,
            "input_differences": input_differences, "deep_wall": deep_wall,
            "rows_left_out_of_correction_fields": excluded,
            "statistics": merged, "bit_identical_correction_fields": bit_identical,
            "verdict": "PASS" if verdict_pass else ("INVALID" if verdict_invalid else "DIFFERENT"),
            "wall_s": time.perf_counter() - start}
        out = pathlib.Path(arguments.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(document, indent=1), encoding="utf-8")
        table = markdown_table(document)
        out.with_suffix(".md").write_text(table + "\n", encoding="utf-8")
        print(table)
        print(f"{LOG_PREFIX} inputs {'identical' if not input_differences else input_differences}; global status "
              f"{'identical' if not status_differences else status_differences}; deep walls {deep_wall}; invariants "
              f"{problems}; L / kernel sum / gradient {'bit-identical' if bit_identical else 'DIFFERENT'} -> "
              f"{document['verdict']} ({document['wall_s']:.0f} s) -> {out}", flush=True)
        if verdict_invalid:
            return EXIT_INVALID
        return EXIT_PASS if verdict_pass else EXIT_DIFFERENT
    finally:
        orchestrator.destroy()
        for sim in sims:
            sim.destroy()
        for context in contexts:
            context.destroy()


if __name__ == "__main__":
    sys.exit(main())
