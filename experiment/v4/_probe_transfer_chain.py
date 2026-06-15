"""
_probe_transfer_chain.py — measure the real cross-GPU transfer-chain length to
decide whether phase_b (correction_interior + density_deep_interior) still has
compression room.

Phase B's job is to HIDE the transfer chain (readback DMA -> worker memcpy ->
upload DMA) behind compute. So phase_b cannot usefully shrink below the chain
length: once phase_b == chain, shrinking the kernels only grows b_to_c_gap (the
GPU waits for the transfer instead) and the critical path is unchanged.

This probe times each DMA leg directly with transfer-queue GPU timestamps, in
ISOLATION (no concurrent compute), so it is a clean LOWER bound on the in-flight
chain (production readback contends with phase_b for VRAM bandwidth, so the real
chain is >= this). Worker memcpy is timed as the numpy slice copy.

Per-GPU phase_c gate:
  AMD phase_c waits AMD's upload, fed by NV_readback(leading) + memcpy(b->a):
      chain_gate_AMD = NV_readback + memcpy + AMD_upload
  NV  phase_c waits NV's upload,  fed by AMD_readback(trailing) + memcpy(a->b):
      chain_gate_NV  = AMD_readback + memcpy + NV_upload

Usage:
    .venv/Scripts/python.exe experiment/v4/_probe_transfer_chain.py \\
        --weights 2.9,1.0 --pool-safety 1.2 --iters 200
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="cases/lid_driven_cavity_2d/case.yaml")
    p.add_argument("--device-a", type=int, default=0)
    p.add_argument("--device-b", type=int, default=1)
    p.add_argument("--weights", default="2.9,1.0")
    p.add_argument("--pool-safety", type=float, default=1.2)
    p.add_argument("--iters", type=int, default=200)
    return p.parse_args()


def _time_dma(sim, direction: str, record_fn, iters: int) -> float:
    """Submit a timestamped (reset, tick0, copy, tick1) transfer cmd on the
    transfer queue `iters` times; return the median GPU-measured DMA µs."""
    from vulkan import (
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_QUERY_TYPE_TIMESTAMP,
        VK_QUERY_RESULT_64_BIT, VK_QUERY_RESULT_WAIT_BIT,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        VkQueryPoolCreateInfo, VkCommandBufferBeginInfo, VkSubmitInfo, VkFenceCreateInfo,
        vkCreateQueryPool, vkDestroyQueryPool, vkCmdResetQueryPool, vkCmdWriteTimestamp,
        vkGetQueryPoolResults, vkBeginCommandBuffer, vkEndCommandBuffer,
        vkQueueSubmit, vkCreateFence, vkDestroyFence, vkWaitForFences, vkResetFences,
        vkGetPhysicalDeviceProperties, vkFreeCommandBuffers,
    )
    from vulkan._vulkancache import ffi
    dev = sim.ctx.device
    period = vkGetPhysicalDeviceProperties(sim.ctx.physical_device).limits.timestampPeriod
    pool = vkCreateQueryPool(dev, VkQueryPoolCreateInfo(queryType=VK_QUERY_TYPE_TIMESTAMP, queryCount=2), None)
    fence = vkCreateFence(dev, VkFenceCreateInfo(), None)
    samples = []
    try:
        for _ in range(iters):
            cmd = sim._allocate_transfer_oneshot_cmd()
            vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
                flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
            vkCmdResetQueryPool(cmd, pool, 0, 2)
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, 0)
            record_fn(cmd, direction)
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, 1)
            vkEndCommandBuffer(cmd)
            vkResetFences(dev, 1, [fence])
            vkQueueSubmit(sim.ctx.transfer_queue, 1, [VkSubmitInfo(
                commandBufferCount=1, pCommandBuffers=[cmd])], fence)
            vkWaitForFences(dev, 1, [fence], True, 0xFFFFFFFFFFFFFFFF)
            data = ffi.new("uint64_t[2]")
            vkGetQueryPoolResults(dev, pool, 0, 2, 16, data, 8,
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT)
            samples.append((int(data[1]) - int(data[0])) * period / 1000.0)
            vkFreeCommandBuffers(dev, sim.ctx.transfer_command_pool, 1, [cmd])
    finally:
        vkDestroyQueryPool(dev, pool, None)
        vkDestroyFence(dev, fence, None)
    return statistics.median(samples)


def main() -> int:
    args = parse_args()
    from experiment.v4.utils.case_loader_v4 import load_case_v4
    from experiment.v4.utils.orchestrator_v4 import DualGpuOrchestratorV4
    from experiment.v4.utils.partition_v4 import compute_dual_gpu_partition
    from experiment.v4.utils.simulator_v4 import SphSimulatorV4
    from experiment.v4.utils.vulkan_context_v4 import VulkanContextV4

    weights = [float(w) for w in args.weights.split(",")]
    case = load_case_v4(args.case)
    slab0, slab1, _ = compute_dual_gpu_partition(case, weights, pool_safety=args.pool_safety)
    ctx_a = VulkanContextV4.create(device_index=args.device_a, application_name="probe_a")
    ctx_b = VulkanContextV4.create(device_index=args.device_b, application_name="probe_b")
    sim_a = SphSimulatorV4(ctx_a, slab0)
    sim_b = SphSimulatorV4(ctx_b, slab1)
    try:
        with DualGpuOrchestratorV4(sim_a, sim_b, defrag_cadence=1000) as orch:
            orch.bootstrap_all()  # populate staging + device buffers with real data

            # DMA legs (GPU timestamps, isolated)
            amd_readback = _time_dma(sim_a, "trailing", sim_a._record_readback_for_direction, args.iters)
            amd_upload   = _time_dma(sim_a, "trailing", sim_a._record_upload_for_direction,   args.iters)
            nv_readback  = _time_dma(sim_b, "leading",  sim_b._record_readback_for_direction, args.iters)
            nv_upload    = _time_dma(sim_b, "leading",  sim_b._record_upload_for_direction,   args.iters)

            # Worker memcpy (numpy slice copy, both directions)
            def _time_memcpy(src_view, dst_view, iters):
                s = []
                for _ in range(iters):
                    t0 = time.perf_counter()
                    dst_view[:] = src_view
                    s.append((time.perf_counter() - t0) * 1e6)
                return statistics.median(s)
            memcpy_ab = _time_memcpy(sim_a.sender_staging_view("trailing"),
                                     sim_b.receiver_staging_view("leading"), args.iters)
            memcpy_ba = _time_memcpy(sim_b.sender_staging_view("leading"),
                                     sim_a.receiver_staging_view("trailing"), args.iters)

            staging_kb = sim_a.sender_staging_view("trailing").nbytes / 1024
            print("\n" + "=" * 64)
            print(f"TRANSFER-CHAIN PROBE  (staging {staging_kb:.0f} KB/dir, iters={args.iters}, "
                  f"isolated GPU-timestamp DMA = LOWER bound)")
            print("=" * 64)
            print(f"  AMD readback (dev->sender)  = {amd_readback:7.1f} us")
            print(f"  AMD upload   (recv->dev)    = {amd_upload:7.1f} us")
            print(f"  NV  readback (dev->sender)  = {nv_readback:7.1f} us")
            print(f"  NV  upload   (recv->dev)    = {nv_upload:7.1f} us")
            print(f"  worker memcpy a->b          = {memcpy_ab:7.1f} us")
            print(f"  worker memcpy b->a          = {memcpy_ba:7.1f} us")
            chain_gate_amd = nv_readback + memcpy_ba + amd_upload
            chain_gate_nv  = amd_readback + memcpy_ab + nv_upload
            print("  " + "-" * 60)
            print(f"  chain gating AMD phase_c = NV_rb + memcpy(b->a) + AMD_up "
                  f"= {chain_gate_amd:7.1f} us")
            print(f"  chain gating NV  phase_c = AMD_rb + memcpy(a->b) + NV_up "
                  f"= {chain_gate_nv:7.1f} us")
            print("=" * 64)
            print("  Compare vs phase_b (from depth-1 bench @ 2.9:1.0,pool1.2):")
            print("    AMD phase_b ~= 1547 us   NV phase_b ~= 1630 us")
            print("    => phase_b compressible only down to its chain length above.")
    finally:
        sim_a.destroy(); sim_b.destroy(); ctx_a.destroy(); ctx_b.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
