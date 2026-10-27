"""
_run_kernel_breakdown.py — per-kernel GPU time for the V0 reference solver and
the V5 solver in single-GPU mode, on the same case and card, with the same
query-pool method.

Records a copy of each solver's own single-step command sequence (leading
barrier, predict, update_voxel, correction, density kernel, density
scratch->primary copy, force) with a BOTTOM_OF_PIPE timestamp after every
stage, then submits it with a fence wait per frame (1 frame in flight) and
reads the timestamps each frame. The solvers are not modified; the recorder
calls the same private helpers their own recorders use. The density kernel
and the scratch->primary copy are timed separately (the copy stage
dispatches nothing).

Prints one ``RESULT {json}`` line: mean and p50 microseconds per stage plus
the GPU frame total, over the measured frames (defrag frames excluded).

Usage:
    .venv/Scripts/python.exe _run_kernel_breakdown.py --solver v5 --device 1
        --case cases/lid_driven_cavity_2d_8m/case.yaml --warmup 1000 --measure 2000

--solver v7_single (docs/seam_audit/v7_single.md, measurement B): the timed command buffer is the
solver's own production recording (SphSimulatorSingle.record_step with a tick callback), so its
barriers are the production step's; the stages are the recording's tick labels (fused: predict,
update_voxel, correction_density, density_copy, force; unfused: predict, update_voxel, correction,
density_kernel, density_copy, force; plus deep_wall_marker / wall_extrapolate where the case has them).
The production step (no timestamps) is interleaved with the timed one after the warmup - odd frames
timed, even frames production, 2 x --measure frames in all - so both see the same flow state (the step
cost of a case drifts between defrags and with the developing flow: on cavity3d_8m_b4 by ~8 % between
frames 1100 and 2900, experiment/v7_single/_probe_loop_variants.py); the result reports the timed GPU
frame, the timed frames' wall time per step (submit + fence) and the production frames' wall time per
step (one fence, no timestamps), defrag frames excluded from all three.

--interleave 0 ("§3 补测", Table 2 and the per-size GPU step): no production frames, every frame after the
warmup is timed (frames warmup+1 .. warmup+measure, the throughput campaign's window); the record then also
carries the GPU frame of every measured frame including the defrag frames, the defrag commands' wall time and
'gpu_step_amortized_us' = (sum of all GPU frames + defrag wall) / measure, the GPU-side cost per step of that
window. --dump-state writes the end state (positions, fluid mask, voxel ids) for _neighbor_stats.py; the
candidate record also carries per-fluid / per-wall candidates and the voxel occupancy. --expect-gpu refuses any
other device name.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

_REPO = pathlib.Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from _run_single_baseline_bench import (ADAPTERS, install_obj_cache,  # noqa: E402
                                        resolve_device_index_in_subprocess)

STAGES = ["predict", "update_voxel", "correction", "density_kernel",
          "density_copy", "force"]
# v7_single: tick label -> stage name (the stage ends at the label)
SINGLE_STAGE_OF_LABEL = {"predict_end": "predict", "voxel_end": "update_voxel",
                         "deep_wall_marker_end": "deep_wall_marker", "correction_density_end": "correction_density",
                         "correction_end": "correction", "density_kernel_end": "density_kernel",
                         "density_end": "density_copy", "wall_extrapolate_end": "wall_extrapolate",
                         "force_end": "force"}
# Pipeline per solver for the dispatching stages (density_copy has none).
PIPELINES = {
    "v0": {"predict": "predict", "update_voxel": "update_voxel",
           "correction": "correction", "density_kernel": "density", "force": "force"},
    "v5": {"predict": "predict", "update_voxel": "update_voxel",
           "correction": "correction_all", "density_kernel": "density_all",
           "force": "force_all"},
}
QUERY_COUNT = len(STAGES) + 1


def record_timed_step(adapter, pool):
    from vulkan import (VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
                        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                        VkCommandBufferBeginInfo, vkBeginCommandBuffer,
                        vkCmdDispatch, vkCmdResetQueryPool, vkCmdWriteTimestamp,
                        vkEndCommandBuffer)
    sim = adapter.sim
    if adapter.name == "v0":
        per_particle = sim._per_particle_dispatch_count()
        per_voxel = sim._per_voxel_dispatch_count()
    else:
        per_particle = sim._per_own_particle_dispatch_count()
        per_voxel = sim._per_extended_voxel_dispatch_count()
    pipelines = PIPELINES[adapter.name]

    cmd = sim._allocate_oneshot_cmd()
    vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
        flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT))
    vkCmdResetQueryPool(cmd, pool, 0, QUERY_COUNT)
    sim._record_compute_barrier(cmd)                       # both solvers' leading barrier
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, 0)
    for slot, stage in enumerate(STAGES, start=1):
        if stage == "density_copy":
            # Both solvers copy density scratch -> primary before force; the
            # V5 single path adds a compute barrier after the copy.
            sim._record_density_scratch_to_primary_copy(cmd)
            if adapter.name == "v5":
                sim._record_compute_barrier(cmd)
        else:
            sim._bind_pipeline_and_sets(cmd, pipelines[stage])
            vkCmdDispatch(cmd, per_voxel if stage == "update_voxel" else per_particle, 1, 1)
            if stage not in ("density_kernel", "force"):
                sim._record_compute_barrier(cmd)
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, slot)
    vkEndCommandBuffer(cmd)
    return cmd


def record_timed_step_single(adapter, pool):
    """v7_single: the production recording with a timestamp at every tick label; returns (cmd, stages)."""
    from vulkan import (VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                        VkCommandBufferBeginInfo, vkBeginCommandBuffer, vkCmdResetQueryPool, vkCmdWriteTimestamp,
                        vkEndCommandBuffer)
    from experiment.v7_single.utils.simulator_single import tick_labels
    sim = adapter.sim
    labels = tick_labels(sim.switches, sim.deep_wall_active, sim.wall_adami)
    slots = {label: index for index, label in enumerate(labels)}
    stages = [SINGLE_STAGE_OF_LABEL[label] for label in labels[1:]]

    def tick(cmd, label):
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, slots[label])
    cmd = sim._allocate_oneshot_cmd()
    vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT))
    vkCmdResetQueryPool(cmd, pool, 0, len(labels))
    sim.record_step(cmd, tick=tick)
    vkEndCommandBuffer(cmd)
    return cmd, stages


def candidate_pairs_single(adapter, dump_state: str | None = None, step: int | None = None) -> dict:
    """v7_single: the neighbour loops' candidate count from the voxel lists at the end of the run: every alive
    particle visits the particles listed in the 3^d voxels around its own (z only in 3-D), so
    candidates = sum over voxels v of count(v) * sum over the stencil of count(v'); per particle = candidates /
    alive. 'ps per candidate' of a neighbour kernel = its microseconds per step * 1e6 / candidates.
    The statistics come from experiment/v7_single/_neighbor_stats.py (per fluid / wall particle too, occupied
    voxels, mean and max per voxel); ``dump_state`` writes the read-back float32 positions, the fluid mask and the
    GPU voxel ids to an .npz for the neighbour count (_neighbor_stats.py --state)."""
    import numpy as np
    from experiment.v7_single._neighbor_stats import KIND_FLUID, candidate_statistics, kinds_of
    sim = adapter.sim
    case = sim.case
    grid = case.grid
    dimensions = (grid.grid_dimension_x, grid.grid_dimension_y, grid.grid_dimension_z)
    if int(getattr(grid, "voxel_order", 0)) != 0:
        raise RuntimeError("candidate_pairs_single assumes voxel_order 0 (x slowest, y fastest)")
    raw = sim.readback_buffers_batch(["inside_particle_count", "position_voxel_id", "velocity_mass", "material"])
    counts = np.frombuffer(raw["inside_particle_count"], dtype=np.uint32)[1:1 + grid.total_voxel_count()].astype(np.int64)
    capacity = case.capacities.total_pool_capacity()
    first, stop = sim.own_first_pid(), sim.own_first_pid() + case.capacities.own_pool_size
    position = np.frombuffer(raw["position_voxel_id"], np.float32)[:capacity * 4].reshape(capacity, 4)[first:stop]
    velocity = np.frombuffer(raw["velocity_mass"], np.float32)[:capacity * 4].reshape(capacity, 4)[first:stop]
    material = np.frombuffer(raw["material"], np.uint32)[:capacity][first:stop]
    alive = (velocity[:, 3] > 0) & (position[:, 3] > 0.5)
    kinds = kinds_of(case)[material[alive].astype(np.int64)]
    fluid_mask = kinds == KIND_FLUID
    voxel_linear = np.rint(position[alive, 3]).astype(np.int64) - 1
    if voxel_linear.min() < 0 or voxel_linear.max() >= counts.size:
        raise RuntimeError("voxel ids outside the grid")
    if not np.array_equal(np.bincount(voxel_linear, minlength=counts.size), counts):
        raise RuntimeError("the voxel lists' counts differ from the particles' voxel ids")
    result = candidate_statistics(counts, dimensions, int(case.physics.neighbor_z_range), voxel_linear, fluid_mask)
    result["alive_rows"] = int(alive.sum())
    if dump_state:
        path = pathlib.Path(dump_state)
        path.parent.mkdir(parents=True, exist_ok=True)
        dx = 2.0 * min(float(m.radius) for m in case.materials if float(m.radius) > 0)
        np.savez(path, positions=np.ascontiguousarray(position[alive, :3]), fluid_mask=fluid_mask,
                 voxel_linear=voxel_linear, origin=np.array([grid.origin_x, grid.origin_y, grid.origin_z]),
                 dims=np.array(dimensions), h=float(case.physics.smoothing_length),
                 z_range=int(case.physics.neighbor_z_range), dimension=int(case.physics.dimension), dx=dx,
                 step=int(step if step is not None else -1))
        result["state_dump"] = str(path)
    return result


def case_dimension(adapter) -> int:
    physics = getattr(adapter.case, "physics", None)
    return int(getattr(physics, "dimension", 2))


def prepare_defrag(sim) -> None:
    """E9 (v7 / v7_single, numerics.defrag_order = prefix): the host scan of experiment.v7.utils.defrag_prefix must run right
    before ANY submission of the defrag command (voxel_base_offset in voxel-id order + the own total in the scratch
    counter). The simulators do it inside submit_defrag_and_wait; this driver submits the adapter's own defrag command
    (the queue is idle: the timed step just completed) and therefore does it here. Solvers without the flag (V0, V5,
    V6, atomic order) are untouched. The scan's wall time counts in defrag_wall like the command it belongs to."""
    numerics = getattr(getattr(sim, "case", None), "numerics", None)
    if numerics is not None and getattr(numerics, "use_prefix_sum_defrag", False):
        from experiment.v7.utils.defrag_prefix import upload_defrag_prefix
        upload_defrag_prefix(sim, verbose=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=sorted(ADAPTERS), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--gpu-uuid", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--measure", type=int, default=2000)
    parser.add_argument("--obj-cache", type=str, default="logs/_obj_npy_cache")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--run", type=int, default=0, help="independent run index")
    parser.add_argument("--expect-gpu", type=str, default=None, help="abort unless the device name contains this")
    parser.add_argument("--interleave", type=int, default=None, choices=(0, 1),
                        help="v7_single: 1 (default) = timed and production frames interleaved after the warmup "
                             "(2 x --measure frames); 0 = every measured frame timed (frames warmup+1 .. "
                             "warmup+measure, continuous, the throughput campaign's window)")
    parser.add_argument("--dump-state", type=str, default=None,
                        help="v7_single: write the final positions / fluid mask / voxel ids to this .npz")
    args = parser.parse_args()

    install_obj_cache(pathlib.Path(args.obj_cache))
    if args.gpu_uuid:
        device_index = resolve_device_index_in_subprocess(args.gpu_uuid, args.solver)
    elif args.device is not None:
        device_index = args.device
    else:
        raise SystemExit("pass --gpu-uuid or --device")

    from vulkan import (VK_QUERY_RESULT_64_BIT, VK_QUERY_RESULT_WAIT_BIT,
                        VK_QUERY_TYPE_TIMESTAMP, VkQueryPoolCreateInfo, ffi,
                        vkCreateQueryPool, vkDestroyQueryPool,
                        vkFreeCommandBuffers, vkGetPhysicalDeviceProperties,
                        vkGetQueryPoolResults)

    adapter = ADAPTERS[args.solver](args.case, device_index)
    device = adapter.device
    properties = vkGetPhysicalDeviceProperties(adapter.physical_device)
    if args.expect_gpu and args.expect_gpu not in str(properties.deviceName):
        adapter.destroy()
        raise SystemExit(f"selected device '{properties.deviceName}' does not contain '{args.expect_gpu}'; "
                         f"refusing to measure")
    ns_per_tick = float(properties.limits.timestampPeriod)
    stages = list(STAGES)
    query_count = QUERY_COUNT
    if args.solver == "v7_single":
        from experiment.v7_single.utils.simulator_single import tick_labels
        query_count = len(tick_labels(adapter.sim.switches, adapter.sim.deep_wall_active, adapter.sim.wall_adami))
    pool = vkCreateQueryPool(device, VkQueryPoolCreateInfo(
        queryType=VK_QUERY_TYPE_TIMESTAMP, queryCount=query_count), None)
    if args.solver == "v7_single":
        cmd, stages = record_timed_step_single(adapter, pool)
    else:
        cmd = record_timed_step(adapter, pool)
    print(f"[kernel_breakdown] solver={args.solver} gpu={str(properties.deviceName)} "
          f"particles={adapter.expected_particle_count:,} dimension={case_dimension(adapter)} "
          f"timestampPeriod={ns_per_tick}ns", flush=True)

    samples: dict[str, list[float]] = {stage: [] for stage in stages}
    samples["gpu_frame"] = []
    cadence = adapter.defrag_cadence
    data = ffi.new(f"uint64_t[{query_count}]")
    interleave = args.solver == "v7_single" and args.interleave != 0
    total = args.warmup + (2 * args.measure if interleave else args.measure)
    t_start = None
    epoch_start = epoch_end = None
    production = None
    timed_wall: list[float] = []
    production_wall: list[float] = []
    gpu_frame_all: list[float] = []        # every measured timed frame, defrag frames included
    defrag_wall: list[float] = []          # the defrag command's submit + wait on the measured defrag frames
    try:
        for frame in range(1, total + 1):
            defrag_frame = bool(cadence) and frame % cadence == 0
            measured = frame > args.warmup
            production_frame = interleave and measured and frame % 2 == 0
            if production_frame:
                t_frame = time.perf_counter()
                adapter.sim.step_and_wait()
                wall = time.perf_counter() - t_frame
                if defrag_frame:
                    adapter.sim.submit_defrag_and_wait()
                elif measured:
                    production_wall.append(wall * 1e6)
                continue
            t_frame = time.perf_counter()
            adapter.sim.ctx.submit_and_wait(cmd)
            wall = time.perf_counter() - t_frame
            if defrag_frame:
                t_defrag = time.perf_counter()
                prepare_defrag(adapter.sim)      # E9: the prefix order's host scan before the adapter's own defrag command
                adapter.sim.ctx.submit_and_wait(adapter.defrag_cmd)
                if measured:
                    defrag_wall.append((time.perf_counter() - t_defrag) * 1e6)
            if frame == args.warmup:
                t_start = time.perf_counter()
                epoch_start = time.time()
            if not measured:
                continue
            vkGetQueryPoolResults(device, pool, 0, query_count, query_count * 8, data, 8,
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT)
            ticks = [int(data[i]) for i in range(query_count)]
            frame_us = (ticks[-1] - ticks[0]) * ns_per_tick / 1000.0
            gpu_frame_all.append(frame_us)
            if defrag_frame:
                continue
            timed_wall.append(wall * 1e6)
            for slot, stage in enumerate(stages, start=1):
                samples[stage].append((ticks[slot] - ticks[slot - 1]) * ns_per_tick / 1000.0)
            samples["gpu_frame"].append(frame_us)
        elapsed = time.perf_counter() - t_start
        epoch_end = time.time()
        if interleave:
            production = {"steps": len(production_wall), "interleaved": True,
                          "us_per_step": statistics.fmean(production_wall),
                          "cpu_fps": 1e6 / statistics.fmean(production_wall)}
        status = adapter.readback_status()
        alive = int(status["alive_particle_count"])
        candidates = (candidate_pairs_single(adapter, args.dump_state, step=total)
                      if args.solver == "v7_single" else None)
        # the GPU time of the whole window per step: every timed frame's GPU frame (defrag frames included) plus
        # the defrag commands' wall time, over the measured steps (only meaningful without interleaving, where
        # the window is the throughput campaign's: frames warmup+1 .. warmup+measure)
        amortized = ((sum(gpu_frame_all) + sum(defrag_wall)) / args.measure
                     if (not interleave and gpu_frame_all) else None)
        result = {
            "solver": args.solver, "case": args.case, "tag": args.tag, "run": args.run,
            "device": device_index, "gpu": str(properties.deviceName),
            "dimension": case_dimension(adapter),
            "particles": adapter.expected_particle_count, "alive": alive,
            "drift": alive - adapter.expected_particle_count,
            "overflow": int(status.get("overflow_inside_count", 0))
                        + int(status.get("overflow_incoming_count", 0)),
            "overflow_all": {key: int(value) for key, value in status.items() if key.startswith("overflow_") and value},
            "warmup": args.warmup, "measure": args.measure, "interleave": interleave,
            "frames_timed": len(samples["gpu_frame"]),
            "cpu_fps": args.measure / elapsed,
            "epoch_start": epoch_start, "epoch_end": epoch_end,
            "timestamp_period_ns": ns_per_tick,
            "stages": stages,
            "candidates": candidates,
            "switches": getattr(adapter, "switches", None),
            "production": production,
            "frames_total": total,
            "timed_us_per_step": (statistics.fmean(timed_wall) if timed_wall else elapsed / args.measure * 1e6),
            "timed_wall_frames": len(timed_wall),
            "gpu_frame_all_mean_us": statistics.fmean(gpu_frame_all) if gpu_frame_all else None,
            "gpu_frame_all_frames": len(gpu_frame_all),
            "defrag_wall_us": [round(value, 1) for value in defrag_wall],
            "gpu_step_amortized_us": amortized,
            "mean_us": {key: round(statistics.fmean(values), 2) for key, values in samples.items()},
            "p50_us": {key: round(statistics.median(values), 2) for key, values in samples.items()},
            "std_us": {key: round(statistics.pstdev(values), 2) for key, values in samples.items()},
        }
    finally:
        vkFreeCommandBuffers(device, adapter.sim.ctx.command_pool, 1, [cmd])
        vkDestroyQueryPool(device, pool, None)
        adapter.destroy()

    print("RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
