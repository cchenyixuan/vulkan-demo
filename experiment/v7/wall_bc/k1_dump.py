"""k1_dump.py - E36 GPU worker: run a cavity case with ONE slab (K = 1) on one GPU with the v6 or the v7 solver,
from the case's initial state, and dump the full per-particle state after the last step, sorted by a global
particle id.

The global id (the particle's row in the case) rides in extension_fields (z = id // 2**20, w = id % 2**20), as in
experiment/seam_audit/dump_state.py: defrag copies that field bit for bit and no kernel reads it, so two runs can
be compared particle by particle although defrag orders the pids by atomic counters.

Solver switches: the validation release set (experiment/validation/cavity_runner.RELEASE_ENVIRONMENT) under the
solver's prefix (V6_ or V7_), plus V7_WALL_BC for v7, set here before the solver import (the solvers read their
switches at import). The frame loop is ChainOrchestrator.run_pipelined at depth 2 with the case's defrag cadence,
as in the validation runner. With --monitor every defrag boundary (drained) checks alive count, every overflow_*
counter and the fluid particles beyond the effective wall (max(|x|, |y|) > 0.5 + dx / 2 in the cavity frame),
and records them in <out>.monitor.jsonl.

    .venv/Scripts/python.exe -m experiment.v7.wall_bc.k1_dump --solver v7 --wall-bc 1 \\
        --case cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025/case.yaml --device 0 --steps 200 \\
        --out logs/e36/equivalence/v7_bc0_a.npz
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib
import sys
import time

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

GLOBAL_ID_LOW_BASE = 2 ** 20
# set-0 buffers dumped per particle: (name, components, dtype)
DUMPED_FIELDS = (("position_voxel_id", 4, np.float32), ("velocity_mass", 4, np.float32),
                 ("density_pressure", 2, np.float32), ("density_pressure_scratch", 2, np.float32),
                 ("acceleration", 4, np.float32), ("shift", 4, np.float32), ("material", 1, np.uint32),
                 ("correction_inverse", 8, np.float32), ("density_gradient_kernel_sum", 4, np.float32),
                 ("extension_fields", 4, np.float32))
V7_ONLY_FIELDS = (("wall_dummy_velocity", 4, np.float32),)


def solver_environment(solver: str, wall_bc: int | None) -> dict:
    from experiment.validation.cavity_runner import RELEASE_ENVIRONMENT
    prefix = solver.upper() + "_"
    environment = {key.replace("V6_", prefix, 1): value for key, value in RELEASE_ENVIRONMENT.items()}
    if solver == "v7":
        environment["V7_WALL_BC"] = str(wall_bc)
    return environment


def set_environment(solver: str, wall_bc: int | None) -> dict:
    for key in [key for key in os.environ if key.startswith(("V6_", "V7_"))]:
        del os.environ[key]
    environment = solver_environment(solver, wall_bc)
    os.environ.update(environment)
    os.environ["VK_LOADER_LAYERS_DISABLE"] = "VK_LAYER_KHRONOS_validation"
    return environment


def load_solver(solver: str):
    package = f"experiment.{solver}.utils"
    upper = solver.upper()
    return {"load_case": getattr(importlib.import_module(f"{package}.case_loader_{solver}"), f"load_case_{solver}"),
            "kind_fluid": importlib.import_module(f"{package}.case_{solver}").KIND_FLUID,
            "partition": importlib.import_module(f"{package}.partition_{solver}"),
            "Simulator": getattr(importlib.import_module(f"{package}.simulator_{solver}"), f"SphSimulator{upper}"),
            "Orchestrator": getattr(importlib.import_module(f"{package}.orchestrator_{solver}"),
                                    f"ChainOrchestrator{upper}"),
            "Context": getattr(importlib.import_module(f"{package}.vulkan_context_{solver}"), f"VulkanContext{upper}")}


def install_global_ids(simulator_class) -> None:
    """The initial upload also fills extension_fields with the global ids (row index in the case)."""
    original = simulator_class._build_initial_data

    def build_initial_data_with_global_ids(self):
        data = original(self)
        if "extension_fields" in data:
            raise RuntimeError("the solver uploads extension_fields itself")
        count = self.case.initial.positions.shape[0]
        fields = np.zeros((self.case.capacities.total_pool_capacity(), 4), dtype=np.float32)
        ids = np.arange(count, dtype=np.int64)
        first = self.own_first_pid()
        fields[first:first + count, 2] = (ids // GLOBAL_ID_LOW_BASE).astype(np.float32)
        fields[first:first + count, 3] = (ids % GLOBAL_ID_LOW_BASE).astype(np.float32)
        data["extension_fields"] = fields.tobytes()
        return data

    simulator_class._build_initial_data = build_initial_data_with_global_ids


def sort_voxel_lists(sim) -> None:
    """Sort the entries of every voxel's inside list by pid (host round trip of inside_particle_index)."""
    slots = sim.case.capacities.max_particles_per_voxel
    raw = sim.readback_buffers_batch(["inside_particle_count", "inside_particle_index"])
    counts = np.frombuffer(raw["inside_particle_count"], np.uint32)
    index = np.frombuffer(raw["inside_particle_index"], np.uint32).copy()
    rows = index[:counts.size * slots].reshape(counts.size, slots)
    for voxel in np.flatnonzero(counts):
        rows[voxel, :counts[voxel]] = np.sort(rows[voxel, :counts[voxel]])
    sim._staging_upload(sim.buffers["inside_particle_index"], index.tobytes())


def read_state(sim, solver: str) -> dict:
    """Alive own particles of every dumped field, sorted by global id (stored density, no offset added)."""
    fields = DUMPED_FIELDS + (V7_ONLY_FIELDS if solver == "v7" else ())
    raw = sim.readback_buffers_batch([name for name, _, _ in fields], density="stored")
    capacity = sim.case.capacities.total_pool_capacity()
    first, stop = sim.own_first_pid(), sim.own_first_pid() + sim.case.capacities.own_pool_size
    rows = {}
    for name, components, dtype in fields:
        flat = np.frombuffer(raw[name], dtype=dtype)[:capacity * components]
        rows[name] = np.array((flat.reshape(capacity, components) if components > 1 else flat)[first:stop])
    alive = (rows["velocity_mass"][:, 3] > 0) & (rows["position_voxel_id"][:, 3] > 0.5)
    rows = {name: values[alive] for name, values in rows.items()}
    extension = rows["extension_fields"].astype(np.float64)
    ids = np.rint(extension[:, 2]).astype(np.int64) * GLOBAL_ID_LOW_BASE + np.rint(extension[:, 3]).astype(np.int64)
    if np.any(extension[:, :2] != 0) or len(np.unique(ids)) != ids.size:
        raise RuntimeError("extension_fields no longer hold unique global ids")
    order = np.argsort(ids)
    rows = {name: values[order] for name, values in rows.items()}
    rows["global_id"] = ids[order]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solver", choices=("v6", "v7"), required=True)
    parser.add_argument("--wall-bc", type=int, choices=(0, 1, 2, 3), default=None, help="v7 only: V7_WALL_BC")
    parser.add_argument("--case", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--out", required=True)
    parser.add_argument("--monitor", action="store_true", help="check invariants at every defrag boundary")
    parser.add_argument("--timestamps", action="store_true",
                        help="with --monitor: GPU timestamps of the last frame before every defrag boundary "
                             "(per-kernel durations in the monitor log)")
    parser.add_argument("--canonical-lists", action="store_true",
                        help="equivalence harness: sort every voxel list by pid after the initial voxelization and skip "
                             "the bootstrap defrag (the solver's lists are built by atomic appends, whose order varies "
                             "from run to run)")
    arguments = parser.parse_args()
    if (arguments.solver == "v7") != (arguments.wall_bc is not None):
        sys.exit("--wall-bc is required with --solver v7 and not allowed with v6")
    environment = set_environment(arguments.solver, arguments.wall_bc)
    solver = load_solver(arguments.solver)
    install_global_ids(solver["Simulator"])

    global_case = solver["load_case"](arguments.case)
    chain = solver["partition"].compute_chain_partition(global_case, [1.0], 1.2)
    if len(chain.slabs) != 1:
        sys.exit("expected one slab")
    slab = chain.slabs[0]
    if not np.array_equal(slab.initial.positions, global_case.initial.positions):
        sys.exit("the K = 1 slab does not hold the case's particles in the case's order")
    total = int(global_case.initial.positions.shape[0])
    radii = [float(material.radius) for material in global_case.materials if float(material.radius) > 0]
    spacing = 2.0 * min(radii)
    fluid_groups = [index for index, material in enumerate(global_case.materials) if material.kind == solver["kind_fluid"]]
    out = pathlib.Path(arguments.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    monitor_path = out.with_suffix(".monitor.jsonl")
    if arguments.monitor and monitor_path.exists():
        monitor_path.unlink()

    context = solver["Context"].create(device_index=arguments.device, enable_validation=False,
                                       application_name=f"e36_{arguments.solver}")
    sim = solver["Simulator"](context, slab, sync_scheme="per-direction")
    orchestrator = solver["Orchestrator"]([sim], defrag_cadence=int(global_case.numerics.defrag_cadence))
    bench = None
    if arguments.timestamps:
        # attached before the step cmds are recorded (bootstrap_all records them)
        bench_module = importlib.import_module(f"experiment.{arguments.solver}.utils.bench_{arguments.solver}")
        bench = bench_module.BenchTimer(context, label="e36")
        sim.bench = bench
    if arguments.canonical_lists:
        # bootstrap_all of a one-slab chain without its defrag, the lists sorted between the voxelization and the
        # bootstrap passes
        sim.bootstrap_init()
        sort_voxel_lists(sim)
        sim.bootstrap_compute()
        sim.prepare_step_cmd_buffers()
    else:
        orchestrator.bootstrap_all()
    status_log = []
    wall_start = time.perf_counter()

    def on_defrag(frame: int, report: list) -> None:
        if not arguments.monitor:
            return
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
                ticks, _ = bench_module.split_parity_ticks(ticks, (frame - 1) % 2)
            record["ticks_ns"] = {label: float(value) for label, value in ticks.items()}
        status_log.append(record)
        with open(monitor_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if record["alive"] != total or overflow:
            raise RuntimeError(f"invariant violation at step {frame}: {record}")

    result = orchestrator.run_pipelined(arguments.steps, depth=arguments.depth, on_defrag=on_defrag)
    state = read_state(sim, arguments.solver)
    status = sim.readback_global_status()
    meta = {"solver": arguments.solver, "wall_bc": arguments.wall_bc, "case": arguments.case,
            "device": arguments.device, "steps": arguments.steps, "depth": arguments.depth,
            "canonical_lists": arguments.canonical_lists, "environment": environment, "dt": float(global_case.physics.timestep),
            "support_radius": float(global_case.physics.smoothing_length), "spacing": spacing,
            "stored_density_offset": float(sim.stored_density_offset()), "alive": int(state["global_id"].size),
            "expected": total, "status": status, "fps": result.get("fps"), "elapsed_s": result.get("elapsed_s"),
            "device_name": context.device_name}
    np.savez(out, meta=json.dumps(meta), **state)
    print(f"[e36] {arguments.solver} wall_bc={arguments.wall_bc} {arguments.steps} steps: alive "
          f"{meta['alive']}/{total}, fps {meta['fps']:.0f}, overflow "
          f"{ {key: value for key, value in status.items() if key.startswith('overflow_') and value} } -> {out}",
          flush=True)
    orchestrator.destroy()
    sim.destroy()
    context.destroy()
    return 0 if meta["alive"] == total else 3


if __name__ == "__main__":
    sys.exit(main())
