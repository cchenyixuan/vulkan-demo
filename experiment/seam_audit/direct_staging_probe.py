"""
direct_staging_probe.py — evaluation of opt item (f): ghost_send writing its packets straight
into host-visible staging instead of device-local memory + a readback DMA.

Synthetic stand-in for the replica part of ghost_send (one RTX 5090, device 1 by default):
THREADS threads (one per (y, z) face voxel and ghost layer) each allocate RECORDS consecutive
slots with one atomicAdd and write them as the lean 44 B packet in SoA segments (position +
vid vec4, velocity + mass vec4, rho / P vec2, material uint), reading the source values from a
device-local buffer, plus one count word per thread. Targets:

  vram     device-local target, then a vkCmdCopyBuffer of the region (capacity = live / 0.8,
           the occupancy the sized pools run at) into a HOST_VISIBLE | HOST_CACHED staging on
           the dedicated transfer queue — today's path (kernel + readback DMA).
  cached   the same kernel writing straight into HOST_VISIBLE | HOST_CACHED memory (what the
           sender staging uses today) bound as a storage buffer.
  coherent the same into HOST_VISIBLE | HOST_COHERENT (uncached, write-combined) memory.

GPU timestamps: kernel duration, DMA duration (the comparable numbers); host: submit-to-fence
wall for the whole chain (kernel [+ DMA]; for vram these are TWO host-serialized submit + fence
round trips, which production replaces by one semaphore hop, so that column overstates today's
latency) and a CPU read of the live bytes afterwards (what the worker does). Medians over
--iterations. The writer is today's thread-per-voxel pattern (each lane writes its own run of
records, so a warp's host writes do not coalesce); the result covers that writer only. Core-Vulkan only (host-visible memory as a storage buffer is guaranteed by
the spec: every non-sparse buffer's memoryTypeBits includes a HOST_VISIBLE | HOST_COHERENT type).

Usage:
    .venv/Scripts/python.exe -m experiment.seam_audit.direct_staging_probe --out logs/seam_audit/opt/direct_staging
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import time

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SHADER_SOURCE = r"""
#version 460
layout(local_size_x = 64) in;
layout(constant_id = 0) const uint THREAD_COUNT = 1024u;
layout(constant_id = 1) const uint RECORDS_PER_THREAD = 32u;
layout(constant_id = 2) const uint CAPACITY = 32768u;      // slots per segment
layout(constant_id = 3) const uint SOURCE_COUNT = 1048576u;
layout(std430, set = 0, binding = 0) readonly buffer SourceBuffer { vec4 source_values[]; };
layout(std430, set = 0, binding = 1) buffer TargetVec4 { vec4 target_vec4[]; };
layout(std430, set = 0, binding = 2) buffer TargetVec2 { vec2 target_vec2[]; };
layout(std430, set = 0, binding = 3) buffer TargetWord { uint target_word[]; };
layout(std430, set = 0, binding = 4) buffer CounterBuffer { uint allocation; };

void main() {
    uint thread_index = gl_GlobalInvocationID.x;
    if (thread_index >= THREAD_COUNT) return;
    uint base_slot = atomicAdd(allocation, RECORDS_PER_THREAD);
    // segment byte layout: [vec4 position | vec4 velocity | vec2 rho/P | uint material | uint count]
    uint velocity_vec4 = CAPACITY;                       // in vec4 units
    uint density_vec2 = 4u * CAPACITY;                   // in vec2 units (= 32 B * CAPACITY)
    uint material_word = 10u * CAPACITY;                 // in words (= 40 B * CAPACITY)
    uint count_word = 11u * CAPACITY;
    for (uint record_index = 0u; record_index < RECORDS_PER_THREAD; record_index++) {
        uint slot = base_slot + record_index;
        uint source_index = (thread_index * 977u + record_index * 131u) % SOURCE_COUNT;
        vec4 source_value = source_values[source_index];
        target_vec4[slot] = source_value;
        target_vec4[velocity_vec4 + slot] = source_value.yzwx;
        target_vec2[density_vec2 + slot] = source_value.xy;
        target_word[material_word + slot] = floatBitsToUint(source_value.z);
    }
    target_word[count_word + thread_index] = RECORDS_PER_THREAD;
}
"""

# name: (threads, records per thread) ~ (face voxels x 2 ghost layers, replicas per voxel)
WORKLOADS = {
    "2d_narrow": (40, 16),
    "2d_1m": (420, 25),
    "2d_4m": (840, 25),
    "2d_16m": (1640, 25),
    "3d_narrow": (5400, 64),
    "3d_8m": (6500, 64),
}


def compile_shader(out_dir: pathlib.Path) -> bytes:
    source = out_dir / "direct_staging_probe.comp"
    target = out_dir / "direct_staging_probe.comp.spv"
    source.write_text(SHADER_SOURCE, encoding="utf-8")
    from experiment.v6.compile_shaders_v6 import _find_glslc
    subprocess.run([_find_glslc(), "-O", "--target-env=vulkan1.3",
                    str(source), "-o", str(target)], check=True)
    return target.read_bytes()


def run(args) -> dict:
    from vulkan import (
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        VK_PIPELINE_BIND_POINT_COMPUTE, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
        VK_QUERY_RESULT_64_BIT, VK_QUERY_RESULT_WAIT_BIT, VK_QUERY_TYPE_TIMESTAMP,
        VK_SHADER_STAGE_COMPUTE_BIT, VK_SHARING_MODE_CONCURRENT, VK_SHARING_MODE_EXCLUSIVE,
        VK_WHOLE_SIZE,
        VkBufferCopy, VkBufferCreateInfo, VkCommandBufferAllocateInfo, VkCommandBufferBeginInfo,
        VkComputePipelineCreateInfo, VkDescriptorBufferInfo, VkDescriptorPoolCreateInfo,
        VkDescriptorPoolSize, VkDescriptorSetAllocateInfo, VkDescriptorSetLayoutBinding,
        VkDescriptorSetLayoutCreateInfo, VkFenceCreateInfo, VkMappedMemoryRange,
        VkMemoryAllocateInfo, VkMemoryBarrier, VkPipelineLayoutCreateInfo,
        VkPipelineShaderStageCreateInfo, VkQueryPoolCreateInfo, VkShaderModuleCreateInfo,
        VkSpecializationInfo, VkSpecializationMapEntry, VkSubmitInfo, VkWriteDescriptorSet,
        vkAllocateCommandBuffers, vkAllocateDescriptorSets, vkAllocateMemory, vkBeginCommandBuffer,
        vkBindBufferMemory, vkCmdBindDescriptorSets, vkCmdBindPipeline, vkCmdCopyBuffer,
        vkCmdDispatch, vkCmdFillBuffer, vkCmdPipelineBarrier, vkCmdResetQueryPool,
        vkCmdWriteTimestamp, vkCreateBuffer, vkCreateComputePipelines, vkCreateDescriptorPool,
        vkCreateDescriptorSetLayout, vkCreateFence, vkCreatePipelineLayout, vkCreateQueryPool,
        vkCreateShaderModule, vkDestroyBuffer, vkDestroyFence, vkEndCommandBuffer,
        vkFreeMemory, vkGetBufferMemoryRequirements, vkGetQueryPoolResults,
        vkInvalidateMappedMemoryRanges, vkMapMemory, vkQueueSubmit, vkResetFences,
        vkUpdateDescriptorSets, vkWaitForFences, VK_NULL_HANDLE, ffi,
    )
    from experiment.v6.utils.vulkan_context_v6 import VulkanContextV6

    out_dir = pathlib.Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    code = compile_shader(out_dir)
    context = VulkanContextV6.create(device_index=args.device, application_name="direct_staging_probe")
    device = context.device
    period_ns = context.timestamp_period if hasattr(context, "timestamp_period") else None
    if period_ns is None:
        from vulkan import vkGetPhysicalDeviceProperties
        period_ns = vkGetPhysicalDeviceProperties(context.physical_device).limits.timestampPeriod
    families = sorted({context.compute_queue_family_index, context.transfer_queue_family_index})

    def make_buffer(size, usage, required, preferred=0, concurrent=False):
        if concurrent and len(families) > 1:
            info = VkBufferCreateInfo(size=size, usage=usage, sharingMode=VK_SHARING_MODE_CONCURRENT,
                                      queueFamilyIndexCount=len(families), pQueueFamilyIndices=families)
        else:
            info = VkBufferCreateInfo(size=size, usage=usage, sharingMode=VK_SHARING_MODE_EXCLUSIVE)
        handle = vkCreateBuffer(device, info, None)
        requirements = vkGetBufferMemoryRequirements(device, handle)
        type_index = context.find_memory_type(requirements.memoryTypeBits, required, preferred)
        memory = vkAllocateMemory(device, VkMemoryAllocateInfo(allocationSize=requirements.size,
                                                              memoryTypeIndex=type_index), None)
        vkBindBufferMemory(device, handle, memory, 0)
        return handle, memory, type_index, int(requirements.size)

    storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
    source_count = 1 << 20
    source = make_buffer(16 * source_count, storage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    counter = make_buffer(256, storage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)

    layout_bindings = [VkDescriptorSetLayoutBinding(binding=index, descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                    descriptorCount=1, stageFlags=VK_SHADER_STAGE_COMPUTE_BIT)
                       for index in range(5)]
    set_layout = vkCreateDescriptorSetLayout(device, VkDescriptorSetLayoutCreateInfo(
        bindingCount=5, pBindings=layout_bindings), None)
    pipeline_layout = vkCreatePipelineLayout(device, VkPipelineLayoutCreateInfo(
        setLayoutCount=1, pSetLayouts=[set_layout]), None)
    module = vkCreateShaderModule(device, VkShaderModuleCreateInfo(codeSize=len(code), pCode=code), None)
    descriptor_pool = vkCreateDescriptorPool(device, VkDescriptorPoolCreateInfo(
        maxSets=64, poolSizeCount=1,
        pPoolSizes=[VkDescriptorPoolSize(type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=320)]), None)
    query_pool = vkCreateQueryPool(device, VkQueryPoolCreateInfo(queryType=VK_QUERY_TYPE_TIMESTAMP,
                                                                 queryCount=8), None)
    fence = vkCreateFence(device, VkFenceCreateInfo(), None)

    def command_buffer(pool):
        return vkAllocateCommandBuffers(device, VkCommandBufferAllocateInfo(
            commandPool=pool, level=VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1))[0]

    def submit_and_wait(queue, cmd):
        vkResetFences(device, 1, [fence])
        started = time.perf_counter_ns()
        vkQueueSubmit(queue, 1, [VkSubmitInfo(commandBufferCount=1, pCommandBuffers=[cmd])], fence)
        vkWaitForFences(device, 1, [fence], True, 0xFFFFFFFFFFFFFFFF)
        return (time.perf_counter_ns() - started) / 1000.0

    def timestamps(count):
        data = ffi.new(f"uint64_t[{count}]")
        vkGetQueryPoolResults(device, query_pool, 0, count, 8 * count, data, 8,
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT)
        return [int(data[index]) for index in range(count)]

    results = {"device": args.device, "timestamp_period_ns": period_ns, "workloads": {}}
    for name, (threads, records) in WORKLOADS.items():
        live = threads * records
        capacity = int(np.ceil(live / 0.8))
        region_bytes = 44 * capacity + 4 * threads + 64
        target_bytes = 48 * capacity + 4 * threads + 64
        entry = {"threads": threads, "records": records, "live_bytes": 44 * live + 4 * threads,
                 "dma_bytes": region_bytes, "modes": {}}
        specialization_values = np.array([threads, records, capacity, source_count], dtype=np.uint32)
        map_entries = [VkSpecializationMapEntry(constantID=index, offset=4 * index, size=4) for index in range(4)]
        specialization_data = ffi.new("uint32_t[4]", specialization_values.tolist())
        specialization = VkSpecializationInfo(mapEntryCount=4, pMapEntries=map_entries, dataSize=16,
                                              pData=specialization_data)
        stage = VkPipelineShaderStageCreateInfo(stage=VK_SHADER_STAGE_COMPUTE_BIT, module=module, pName="main",
                                                pSpecializationInfo=specialization)
        pipeline = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, [VkComputePipelineCreateInfo(
            stage=stage, layout=pipeline_layout)], None)[0]
        groups = (threads + 63) // 64
        for mode in ("vram", "cached", "coherent"):
            if mode == "vram":
                target = make_buffer(target_bytes, storage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, concurrent=True)
                staging = make_buffer(target_bytes, storage,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                      VK_MEMORY_PROPERTY_HOST_CACHED_BIT, concurrent=True)
            elif mode == "cached":
                target = make_buffer(target_bytes, storage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                     VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
                staging = target
            else:
                target = make_buffer(target_bytes, storage,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                staging = target
            flags = context.memory_type_flags(staging[2])
            mapped = vkMapMemory(device, staging[1], 0, staging[3], 0)
            host_view = np.frombuffer(mapped, dtype=np.uint8, count=target_bytes)
            descriptor_set = vkAllocateDescriptorSets(device, VkDescriptorSetAllocateInfo(
                descriptorPool=descriptor_pool, descriptorSetCount=1, pSetLayouts=[set_layout]))[0]
            writes = []
            for binding, (buffer, size) in enumerate(((source[0], 16 * source_count), (target[0], target_bytes),
                                                       (target[0], target_bytes), (target[0], target_bytes),
                                                       (counter[0], 256))):
                writes.append(VkWriteDescriptorSet(
                    dstSet=descriptor_set, dstBinding=binding, dstArrayElement=0, descriptorCount=1,
                    descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[VkDescriptorBufferInfo(buffer=buffer, offset=0, range=size)]))
            vkUpdateDescriptorSets(device, len(writes), writes, 0, None)

            kernel_us, dma_us, wall_us, read_us = [], [], [], []
            for iteration in range(args.iterations):
                cmd = command_buffer(context.command_pool)
                vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
                # all four queries are reset here: a transfer-only queue may not reset queries
                vkCmdResetQueryPool(cmd, query_pool, 0, 4)
                vkCmdFillBuffer(cmd, counter[0], 0, 4, 0)
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                     1, [VkMemoryBarrier(srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
                                                         dstAccessMask=VK_ACCESS_SHADER_WRITE_BIT)], 0, None, 0, None)
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, query_pool, 0)
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                                        [descriptor_set], 0, None)
                vkCmdDispatch(cmd, groups, 1, 1)
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, query_pool, 1)
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                                     1, [VkMemoryBarrier(srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
                                                         dstAccessMask=VK_ACCESS_HOST_READ_BIT)],
                                     0, None, 0, None)
                vkEndCommandBuffer(cmd)
                wall = submit_and_wait(context.compute_queue, cmd)
                if mode == "vram":
                    # today's readback: the dedicated transfer queue copies the region
                    transfer_cmd = command_buffer(context.transfer_command_pool)
                    vkBeginCommandBuffer(transfer_cmd, VkCommandBufferBeginInfo(
                        flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
                    vkCmdWriteTimestamp(transfer_cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, query_pool, 2)
                    vkCmdCopyBuffer(transfer_cmd, target[0], staging[0], 1,
                                    [VkBufferCopy(srcOffset=0, dstOffset=0, size=region_bytes)])
                    vkCmdWriteTimestamp(transfer_cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, query_pool, 3)
                    vkCmdPipelineBarrier(transfer_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                                         1, [VkMemoryBarrier(srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
                                                             dstAccessMask=VK_ACCESS_HOST_READ_BIT)],
                                         0, None, 0, None)
                    vkEndCommandBuffer(transfer_cmd)
                    wall += submit_and_wait(context.transfer_queue, transfer_cmd)
                if not (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT):
                    vkInvalidateMappedMemoryRanges(device, 1, [VkMappedMemoryRange(
                        memory=staging[1], offset=0, size=staging[3])])
                started = time.perf_counter_ns()
                copy = np.array(host_view[:44 * live], copy=True)
                read_us.append((time.perf_counter_ns() - started) / 1000.0)
                ticks = timestamps(4 if mode == "vram" else 2)
                if iteration >= args.warmup:
                    kernel_us.append((ticks[1] - ticks[0]) * period_ns / 1000.0)
                    if mode == "vram":
                        dma_us.append((ticks[3] - ticks[2]) * period_ns / 1000.0)
                    wall_us.append(wall)
                del copy
            entry["modes"][mode] = {
                "memory_flags": int(flags),
                "kernel_us": statistics.median(kernel_us),
                "dma_us": statistics.median(dma_us) if dma_us else 0.0,
                "gpu_chain_us": statistics.median(kernel_us) + (statistics.median(dma_us) if dma_us else 0.0),
                "submit_to_fence_us": statistics.median(wall_us),
                "host_read_us": statistics.median(read_us[args.warmup:]),
            }
            print(f"[direct_staging] {name} {mode}: {entry['modes'][mode]}", flush=True)
            from vulkan import vkUnmapMemory
            vkUnmapMemory(device, staging[1])
            for buffer in ({target[0]: target, staging[0]: staging}).values():
                vkDestroyBuffer(device, buffer[0], None)
                vkFreeMemory(device, buffer[1], None)
        results["workloads"][name] = entry
    vkDestroyFence(device, fence, None)
    context.destroy()
    (out_dir / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    lines = ["| workload | live KiB | DMA KiB | vram: kernel + DMA µs | cached: kernel µs | coherent: kernel µs | "
             "submit→fence vram / cached / coherent µs | host read of live bytes vram / cached / coherent µs |",
             "|---|---|---|---|---|---|---|---|"]
    for name, entry in results["workloads"].items():
        modes = entry["modes"]
        lines.append(
            f"| {name} | {entry['live_bytes'] / 1024:,.1f} | {entry['dma_bytes'] / 1024:,.1f} | "
            f"{modes['vram']['kernel_us']:.1f} + {modes['vram']['dma_us']:.1f} | {modes['cached']['kernel_us']:.1f} | "
            f"{modes['coherent']['kernel_us']:.1f} | {modes['vram']['submit_to_fence_us']:.0f} / "
            f"{modes['cached']['submit_to_fence_us']:.0f} / {modes['coherent']['submit_to_fence_us']:.0f} | "
            f"{modes['vram']['host_read_us']:.1f} / {modes['cached']['host_read_us']:.1f} / "
            f"{modes['coherent']['host_read_us']:.1f} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="opt item (f): direct host-visible staging writes")
    parser.add_argument("--out", default="logs/seam_audit/opt/direct_staging")
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=10)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    sys.exit(main())
