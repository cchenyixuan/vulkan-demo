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


def case_dimension(adapter) -> int:
    physics = getattr(adapter.case, "physics", None)
    return int(getattr(physics, "dimension", 2))


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
    ns_per_tick = float(properties.limits.timestampPeriod)
    pool = vkCreateQueryPool(device, VkQueryPoolCreateInfo(
        queryType=VK_QUERY_TYPE_TIMESTAMP, queryCount=QUERY_COUNT), None)
    cmd = record_timed_step(adapter, pool)
    print(f"[kernel_breakdown] solver={args.solver} gpu={str(properties.deviceName)} "
          f"particles={adapter.expected_particle_count:,} dimension={case_dimension(adapter)} "
          f"timestampPeriod={ns_per_tick}ns", flush=True)

    samples: dict[str, list[float]] = {stage: [] for stage in STAGES}
    samples["gpu_frame"] = []
    cadence = adapter.defrag_cadence
    data = ffi.new(f"uint64_t[{QUERY_COUNT}]")
    total = args.warmup + args.measure
    t_start = None
    try:
        for frame in range(1, total + 1):
            adapter.sim.ctx.submit_and_wait(cmd)
            defrag_frame = bool(cadence) and frame % cadence == 0
            if defrag_frame:
                adapter.sim.ctx.submit_and_wait(adapter.defrag_cmd)
            if frame == args.warmup:
                t_start = time.perf_counter()
            if frame <= args.warmup or defrag_frame:
                continue
            vkGetQueryPoolResults(device, pool, 0, QUERY_COUNT, QUERY_COUNT * 8, data, 8,
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT)
            ticks = [int(data[i]) for i in range(QUERY_COUNT)]
            for slot, stage in enumerate(STAGES, start=1):
                samples[stage].append((ticks[slot] - ticks[slot - 1]) * ns_per_tick / 1000.0)
            samples["gpu_frame"].append((ticks[-1] - ticks[0]) * ns_per_tick / 1000.0)
        elapsed = time.perf_counter() - t_start
        status = adapter.readback_status()
        alive = int(status["alive_particle_count"])
        result = {
            "solver": args.solver, "case": args.case, "tag": args.tag, "run": args.run,
            "device": device_index, "gpu": str(properties.deviceName),
            "dimension": case_dimension(adapter),
            "particles": adapter.expected_particle_count, "alive": alive,
            "drift": alive - adapter.expected_particle_count,
            "overflow": int(status.get("overflow_inside_count", 0))
                        + int(status.get("overflow_incoming_count", 0)),
            "warmup": args.warmup, "measure": args.measure,
            "frames_timed": len(samples["gpu_frame"]),
            "cpu_fps": args.measure / elapsed,
            "timestamp_period_ns": ns_per_tick,
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
