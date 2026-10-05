"""
probe_shared_host.py — Step 0 gate of the two-hop transport experiment.

Question: can both RTX 5090s import the SAME host allocation through
VK_EXT_external_memory_host and move bytes GPU A -> shared host block ->
GPU B on their transfer queues, with no CPU copy in between?

What it does, per test size:
  1. VirtualAlloc a page-aligned host block; both devices import the same
     pointer and bind it to a VkBuffer (TRANSFER_SRC | TRANSFER_DST).
  2. Prints which memory types each driver accepts for the import.
  3. Correctness, both directions, several rounds with a fresh random
     pattern each round (a driver that snapshots the pages instead of
     pinning them would pass round 1 and fail round 2):
         source VRAM -[transfer Q]-> shared block      (timeline signal)
         host waits the timeline, checks the block's bytes directly
         shared block -[transfer Q]-> destination VRAM (timeline signal)
         destination VRAM is read back and compared
  4. DMA timing with transfer-queue GPU timestamps: shared block versus
     the staging buffers V5 allocates today (same memory-type preferences as
     SphSimulatorV5._allocate_staging_buffers), same size, interleaved.

Any failure stops the experiment (exit code 1); the result goes into the
README either way.

Usage:
    VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation \\
    V5_SPLIT_TRANSFER_QUEUES=1 \\
    .venv/Scripts/python.exe experiment/two_hop/probe_shared_host.py
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import traceback
from dataclasses import dataclass
from typing import Optional

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vulkan import *  # noqa: E402, F401, F403
from vulkan._vulkancache import ffi  # noqa: E402

from experiment.two_hop.shared_host_v5 import (  # noqa: E402
    VulkanContextTwoHop,
    allocate_host_block,
    describe_memory_property_flags,
    destroy_imported_host_buffer,
    free_host_block,
    import_host_block,
    query_host_pointer_memory_type_bits,
    query_min_imported_host_pointer_alignment,
)
from experiment.v5.utils.sync_scheme_v5 import _create_timeline_semaphore  # noqa: E402

_WAIT_TIMEOUT_NS = 10 * 1_000_000_000
_STAGING_USAGE = (VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                  | VK_BUFFER_USAGE_TRANSFER_DST_BIT)
_HOST_STAGING_PROPERTIES = (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
_PRE_FILL_BYTE = 0xEE


class ProbeFailure(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="two-hop shared host memory probe")
    parser.add_argument("--device-a", type=int, default=0)
    parser.add_argument("--device-b", type=int, default=1)
    parser.add_argument("--sizes-mb", default="0.25,1,4,16,64",
                        help="comma-separated test sizes in MiB")
    parser.add_argument("--rounds", type=int, default=6,
                        help="correctness rounds per size (each round runs "
                             "both directions with a fresh pattern)")
    parser.add_argument("--iterations", type=int, default=120,
                        help="timed copies per (device, memory kind, size)")
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--output",
                        default="logs/two_hop_experiment/probe_shared_host.json")
    return parser.parse_args()


# ============================================================================
# Small Vulkan helpers (no simulator involved)
# ============================================================================

@dataclass
class PlainBuffer:
    handle: object
    memory: object
    size: int
    memory_type_index: int
    view: Optional[np.ndarray] = None


def allocate_plain_buffer(context, size: int, usage: int, required: int,
                          preferred: int = 0, *, concurrent: bool = False,
                          mapped: bool = False) -> PlainBuffer:
    """Mirror of SphSimulatorV5._allocate_buffer (which needs a simulator)."""
    if (concurrent and context.transfer_queue_family_index
            != context.compute_queue_family_index):
        family_indices = [context.compute_queue_family_index,
                          context.transfer_queue_family_index]
        create_info = VkBufferCreateInfo(
            size=size, usage=usage, sharingMode=VK_SHARING_MODE_CONCURRENT,
            queueFamilyIndexCount=len(family_indices),
            pQueueFamilyIndices=family_indices)
    else:
        create_info = VkBufferCreateInfo(
            size=size, usage=usage, sharingMode=VK_SHARING_MODE_EXCLUSIVE)
    handle = vkCreateBuffer(context.device, create_info, None)
    requirements = vkGetBufferMemoryRequirements(context.device, handle)
    type_index = context.find_memory_type(
        requirements.memoryTypeBits, required, preferred)
    memory = vkAllocateMemory(context.device, VkMemoryAllocateInfo(
        allocationSize=requirements.size, memoryTypeIndex=type_index), None)
    vkBindBufferMemory(context.device, handle, memory, 0)
    view = None
    if mapped:
        pointer = vkMapMemory(context.device, memory, 0, size, 0)
        view = np.frombuffer(pointer, dtype=np.uint8, count=size)
    return PlainBuffer(handle=handle, memory=memory, size=size,
                       memory_type_index=type_index, view=view)


def destroy_plain_buffer(context, buffer: PlainBuffer) -> None:
    if buffer.view is not None:
        vkUnmapMemory(context.device, buffer.memory)
    vkDestroyBuffer(context.device, buffer.handle, None)
    vkFreeMemory(context.device, buffer.memory, None)


def allocate_command_buffer(context, command_pool):
    return vkAllocateCommandBuffers(context.device, VkCommandBufferAllocateInfo(
        commandPool=command_pool,
        level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        commandBufferCount=1))[0]


def record_host_read_barrier(command_buffer) -> None:
    """Same barrier V5's readback cmd ends with
    (SphSimulatorV5._record_compute_to_host_barrier)."""
    barrier = VkMemoryBarrier2(
        sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        srcStageMask=VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        srcAccessMask=VK_ACCESS_2_TRANSFER_WRITE_BIT,
        dstStageMask=VK_PIPELINE_STAGE_2_HOST_BIT,
        dstAccessMask=VK_ACCESS_2_HOST_READ_BIT)
    vkCmdPipelineBarrier2(command_buffer, VkDependencyInfo(
        sType=VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        memoryBarrierCount=1, pMemoryBarriers=[barrier]))


def wait_timeline(context, semaphore, value: int) -> None:
    wait_info = VkSemaphoreWaitInfo(
        sType=VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        semaphoreCount=1, pSemaphores=[semaphore], pValues=[value])
    try:
        vkWaitSemaphores(context.device, wait_info, _WAIT_TIMEOUT_NS)
    except Exception as error:
        if type(error).__name__ == "VkTimeout":
            raise ProbeFailure(
                f"timeline wait for value {value} on "
                f"'{context.device_name}' timed out") from error
        raise


# ============================================================================
# Per-device resources for one test size
# ============================================================================

@dataclass
class DeviceKit:
    label: str
    context: object
    size: int
    imported: object                    # ImportedHostBuffer
    video_memory_source: PlainBuffer
    video_memory_destination: PlainBuffer
    sender_staging: PlainBuffer         # V5 sender_staging preferences
    receiver_staging: PlainBuffer       # V5 receiver_staging preferences
    verification_staging: PlainBuffer   # pattern upload / readback only
    timeline: object
    timeline_value: int = 0
    fence: object = None

    def next_timeline_value(self) -> int:
        self.timeline_value += 1
        return self.timeline_value


def build_device_kit(label: str, context, block, size: int) -> DeviceKit:
    imported = import_host_block(context, block, _STAGING_USAGE)
    device_local_usage = (_STAGING_USAGE | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
    receiver_preferred = (
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT
        if os.environ.get("V5_RECEIVER_CACHED", "0") == "1" else 0)
    return DeviceKit(
        label=label,
        context=context,
        size=size,
        imported=imported,
        # CONCURRENT across compute + transfer families, like the 12 V5
        # buffers the transfer queue copies from / into.
        video_memory_source=allocate_plain_buffer(
            context, size, device_local_usage,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, concurrent=True),
        video_memory_destination=allocate_plain_buffer(
            context, size, device_local_usage,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, concurrent=True),
        sender_staging=allocate_plain_buffer(
            context, size, _STAGING_USAGE, _HOST_STAGING_PROPERTIES,
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT, mapped=True),
        receiver_staging=allocate_plain_buffer(
            context, size, _STAGING_USAGE, _HOST_STAGING_PROPERTIES,
            receiver_preferred, mapped=True),
        verification_staging=allocate_plain_buffer(
            context, size, _STAGING_USAGE, _HOST_STAGING_PROPERTIES,
            mapped=True),
        timeline=_create_timeline_semaphore(context.device),
        fence=vkCreateFence(context.device, VkFenceCreateInfo(), None),
    )


def destroy_device_kit(kit: DeviceKit) -> None:
    context = kit.context
    vkDeviceWaitIdle(context.device)
    vkDestroyFence(context.device, kit.fence, None)
    vkDestroySemaphore(context.device, kit.timeline, None)
    for buffer in (kit.video_memory_source, kit.video_memory_destination,
                   kit.sender_staging, kit.receiver_staging,
                   kit.verification_staging):
        destroy_plain_buffer(context, buffer)
    destroy_imported_host_buffer(context, kit.imported)


def copy_and_wait_fence(kit: DeviceKit, source, destination, queue=None) -> None:
    """One transfer-queue copy of kit.size bytes, fence wait."""
    context = kit.context
    command_buffer = allocate_command_buffer(context, context.transfer_command_pool)
    vkBeginCommandBuffer(command_buffer, VkCommandBufferBeginInfo(
        flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
    vkCmdCopyBuffer(command_buffer, source, destination, 1, [
        VkBufferCopy(srcOffset=0, dstOffset=0, size=kit.size)])
    vkEndCommandBuffer(command_buffer)
    vkResetFences(context.device, 1, [kit.fence])
    vkQueueSubmit(queue or context.transfer_queue, 1, [VkSubmitInfo(
        commandBufferCount=1, pCommandBuffers=[command_buffer])], kit.fence)
    vkWaitForFences(context.device, 1, [kit.fence], VK_TRUE, _WAIT_TIMEOUT_NS)
    vkFreeCommandBuffers(context.device, context.transfer_command_pool, 1,
                         [command_buffer])


def copy_and_signal_timeline(kit: DeviceKit, source, destination,
                             *, host_read_barrier: bool, queue=None) -> int:
    """One transfer-queue copy that signals the kit's timeline; returns the
    signalled value after the host observed it."""
    context = kit.context
    value = kit.next_timeline_value()
    command_buffer = allocate_command_buffer(context, context.transfer_command_pool)
    vkBeginCommandBuffer(command_buffer, VkCommandBufferBeginInfo(
        flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
    vkCmdCopyBuffer(command_buffer, source, destination, 1, [
        VkBufferCopy(srcOffset=0, dstOffset=0, size=kit.size)])
    if host_read_barrier:
        record_host_read_barrier(command_buffer)
    vkEndCommandBuffer(command_buffer)
    submit_info = VkSubmitInfo2(
        sType=VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        waitSemaphoreInfoCount=0,
        pWaitSemaphoreInfos=None,
        commandBufferInfoCount=1,
        pCommandBufferInfos=[VkCommandBufferSubmitInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            commandBuffer=command_buffer)],
        signalSemaphoreInfoCount=1,
        pSignalSemaphoreInfos=[VkSemaphoreSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            semaphore=kit.timeline, value=value,
            stageMask=VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT)])
    vkQueueSubmit2(queue or context.transfer_queue, 1, [submit_info],
                   VK_NULL_HANDLE)
    wait_timeline(context, kit.timeline, value)
    vkFreeCommandBuffers(context.device, context.transfer_command_pool, 1,
                         [command_buffer])
    return value


def upload_queue(context):
    """The queue V5 submits uploads on (second transfer queue when
    V5_SPLIT_TRANSFER_QUEUES=1)."""
    return getattr(context, "transfer_queue_upload", None) or context.transfer_queue


# ============================================================================
# Correctness
# ============================================================================

def check_direction(source: DeviceKit, destination: DeviceKit, block,
                    pattern: np.ndarray) -> dict:
    size = source.size
    # Pattern into the source device's VRAM.
    source.verification_staging.view[:] = pattern
    copy_and_wait_fence(source, source.verification_staging.handle,
                        source.video_memory_source.handle)
    # Poison the shared block so stale bytes cannot pass.
    block.view[:size] = _PRE_FILL_BYTE

    # Hop 1: source VRAM -> shared block, timeline-signalled.
    copy_and_signal_timeline(
        source, source.video_memory_source.handle, source.imported.handle,
        host_read_barrier=True)
    host_mismatches = int(np.count_nonzero(block.view[:size] != pattern))

    # Hop 2: shared block -> destination VRAM, timeline-signalled.
    copy_and_signal_timeline(
        destination, destination.imported.handle,
        destination.video_memory_destination.handle,
        host_read_barrier=False, queue=upload_queue(destination.context))

    # Read the destination VRAM back through an ordinary staging buffer.
    destination.verification_staging.view[:] = 0
    copy_and_wait_fence(destination,
                        destination.video_memory_destination.handle,
                        destination.verification_staging.handle)
    device_mismatches = int(np.count_nonzero(
        destination.verification_staging.view != pattern))
    return {
        "direction": f"{source.label}->{destination.label}",
        "host_visible_mismatched_bytes": host_mismatches,
        "destination_mismatched_bytes": device_mismatches,
    }


def run_correctness(kit_a: DeviceKit, kit_b: DeviceKit, block, rounds: int,
                    generator: np.random.Generator) -> list[dict]:
    results = []
    for round_index in range(rounds):
        for source, destination in ((kit_a, kit_b), (kit_b, kit_a)):
            pattern = generator.integers(
                0, 256, size=source.size, dtype=np.uint8)
            outcome = check_direction(source, destination, block, pattern)
            outcome["round"] = round_index
            results.append(outcome)
            if (outcome["host_visible_mismatched_bytes"]
                    or outcome["destination_mismatched_bytes"]):
                raise ProbeFailure(
                    f"byte mismatch, size {source.size}, round {round_index}, "
                    f"{outcome['direction']}: host view "
                    f"{outcome['host_visible_mismatched_bytes']} B, "
                    f"destination VRAM "
                    f"{outcome['destination_mismatched_bytes']} B")
    return results


# ============================================================================
# DMA timing (transfer-queue GPU timestamps)
# ============================================================================

def time_copies(kit: DeviceKit, iterations: int) -> dict:
    """Interleaved timed copies on one device. Returns
    {job label: {median_us, p10_us, p90_us, gigabytes_per_second}}."""
    context = kit.context
    device = context.device
    family = vkGetPhysicalDeviceQueueFamilyProperties(
        context.physical_device)[context.transfer_queue_family_index]
    if family.timestampValidBits == 0:
        print(f"  [{kit.label}] transfer queue family has no timestamps; "
              f"timing skipped")
        return {}
    period_ns = vkGetPhysicalDeviceProperties(
        context.physical_device).limits.timestampPeriod

    readback_queue = context.transfer_queue
    jobs = [
        ("shared_block_readback", kit.video_memory_source.handle,
         kit.imported.handle, readback_queue),
        ("v5_sender_staging_readback", kit.video_memory_source.handle,
         kit.sender_staging.handle, readback_queue),
        ("shared_block_upload", kit.imported.handle,
         kit.video_memory_destination.handle, upload_queue(context)),
        ("v5_receiver_staging_upload", kit.receiver_staging.handle,
         kit.video_memory_destination.handle, upload_queue(context)),
    ]
    query_count = 2 * len(jobs) * iterations
    query_pool = vkCreateQueryPool(device, VkQueryPoolCreateInfo(
        queryType=VK_QUERY_TYPE_TIMESTAMP, queryCount=query_count), None)
    try:
        # vkCmdResetQueryPool is not valid on a transfer-only queue (V5's
        # 2026-07-22 audit fix), so the one reset goes through the compute
        # queue.
        reset_command = allocate_command_buffer(context, context.command_pool)
        vkBeginCommandBuffer(reset_command, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
        vkCmdResetQueryPool(reset_command, query_pool, 0, query_count)
        vkEndCommandBuffer(reset_command)
        context.submit_and_wait(reset_command)
        vkFreeCommandBuffers(device, context.command_pool, 1, [reset_command])

        for iteration in range(iterations):
            for job_index, (_label, source, destination, queue) in enumerate(jobs):
                first_query = 2 * (iteration * len(jobs) + job_index)
                command_buffer = allocate_command_buffer(
                    context, context.transfer_command_pool)
                vkBeginCommandBuffer(command_buffer, VkCommandBufferBeginInfo(
                    flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
                vkCmdWriteTimestamp(
                    command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                    query_pool, first_query)
                vkCmdCopyBuffer(command_buffer, source, destination, 1, [
                    VkBufferCopy(srcOffset=0, dstOffset=0, size=kit.size)])
                vkCmdWriteTimestamp(
                    command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                    query_pool, first_query + 1)
                vkEndCommandBuffer(command_buffer)
                vkResetFences(device, 1, [kit.fence])
                vkQueueSubmit(queue, 1, [VkSubmitInfo(
                    commandBufferCount=1,
                    pCommandBuffers=[command_buffer])], kit.fence)
                vkWaitForFences(device, 1, [kit.fence], VK_TRUE,
                                _WAIT_TIMEOUT_NS)
                vkFreeCommandBuffers(device, context.transfer_command_pool, 1,
                                     [command_buffer])

        data = ffi.new(f"uint64_t[{query_count}]")
        vkGetQueryPoolResults(
            device, query_pool, 0, query_count, 8 * query_count, data, 8,
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT)
    finally:
        vkDestroyQueryPool(device, query_pool, None)

    discarded = max(1, iterations // 10)     # first tenth = warm-up
    summary = {}
    for job_index, (label, _source, _destination, _queue) in enumerate(jobs):
        samples = []
        for iteration in range(discarded, iterations):
            first_query = 2 * (iteration * len(jobs) + job_index)
            ticks = int(data[first_query + 1]) - int(data[first_query])
            samples.append(ticks * period_ns / 1000.0)
        samples.sort()
        median_us = statistics.median(samples)
        summary[label] = {
            "median_us": round(median_us, 2),
            "p10_us": round(samples[len(samples) // 10], 2),
            "p90_us": round(samples[(len(samples) * 9) // 10], 2),
            "gigabytes_per_second": (
                round(kit.size / (median_us * 1e-6) / 1e9, 2)
                if median_us > 0 else None),
            "samples": len(samples),
        }
    return summary


# ============================================================================
# Reporting
# ============================================================================

def describe_device(context, probe_block) -> dict:
    memory_properties = vkGetPhysicalDeviceMemoryProperties(
        context.physical_device)
    host_pointer_bits = query_host_pointer_memory_type_bits(
        context, probe_block.address)
    memory_types = []
    print(f"  memory types of '{context.device_name}' "
          f"(physical device index {context.physical_device_index}); "
          f"* = accepts the host pointer:")
    for type_index in range(memory_properties.memoryTypeCount):
        memory_type = memory_properties.memoryTypes[type_index]
        heap = memory_properties.memoryHeaps[memory_type.heapIndex]
        importable = bool(host_pointer_bits & (1 << type_index))
        flags = describe_memory_property_flags(memory_type.propertyFlags)
        memory_types.append({
            "index": type_index,
            "flags": flags,
            "heap_index": int(memory_type.heapIndex),
            "heap_gigabytes": round(heap.size / 2**30, 1),
            "host_pointer_importable": importable,
        })
        print(f"    {'*' if importable else ' '} type[{type_index}] "
              f"heap[{memory_type.heapIndex}] "
              f"({heap.size / 2**30:.1f} GiB)  {flags}")
    return {
        "name": context.device_name,
        "physical_device_index": context.physical_device_index,
        "min_imported_host_pointer_alignment":
            query_min_imported_host_pointer_alignment(context),
        "host_pointer_memory_type_bits": f"0x{host_pointer_bits:x}",
        "memory_types": memory_types,
        "transfer_queue_family_index": context.transfer_queue_family_index,
        "split_transfer_queues": (
            upload_queue(context) is not context.transfer_queue),
    }


def staging_type_description(context, buffer: PlainBuffer) -> str:
    flags = context.memory_type_flags(buffer.memory_type_index)
    return (f"type[{buffer.memory_type_index}] "
            f"{describe_memory_property_flags(flags)}")


def main() -> int:
    args = parse_args()
    sizes = [int(float(text) * 2**20) for text in args.sizes_mb.split(",")]
    report: dict = {
        "arguments": vars(args),
        "environment": {
            name: os.environ.get(name) for name in (
                "V5_SPLIT_TRANSFER_QUEUES", "V5_RECEIVER_CACHED",
                "VK_LOADER_LAYERS_DISABLE")},
        "devices": {}, "sizes": {}, "verdict": "FAIL", "failure": None,
    }
    contexts = []
    exit_code = 1
    try:
        for label, device_index in (("a", args.device_a), ("b", args.device_b)):
            contexts.append(VulkanContextTwoHop.create(
                device_index=device_index,
                enable_validation=args.validation,
                application_name=f"two_hop_probe_{label}"))
        context_a, context_b = contexts
        if context_a.physical_device_index == context_b.physical_device_index:
            raise ProbeFailure("both contexts landed on the same physical device")

        alignment = max(query_min_imported_host_pointer_alignment(context)
                        for context in contexts)
        print(f"\n[probe] minImportedHostPointerAlignment: "
              + ", ".join(
                  f"{label}={query_min_imported_host_pointer_alignment(context)}"
                  for label, context in zip("ab", contexts))
              + f"  -> using {alignment}")

        probe_block = allocate_host_block(sizes[0], alignment)
        try:
            for label, context in zip("ab", contexts):
                report["devices"][label] = describe_device(context, probe_block)
        finally:
            free_host_block(probe_block)

        generator = np.random.default_rng(20260928)
        for size in sizes:
            print(f"\n[probe] ===== size {size / 2**20:.2f} MiB =====")
            block = allocate_host_block(size, alignment)
            kits: list[DeviceKit] = []
            try:
                for label, context in zip("ab", contexts):
                    kits.append(build_device_kit(label, context, block, size))
                kit_a, kit_b = kits
                size_report: dict = {
                    "aligned_bytes": block.aligned_size, "import": {}}
                for kit in kits:
                    imported = kit.imported
                    size_report["import"][kit.label] = {
                        "memory_type_index": imported.memory_type_index,
                        "memory_type_flags": describe_memory_property_flags(
                            imported.memory_type_flags),
                        "host_pointer_type_bits":
                            f"0x{imported.host_pointer_type_bits:x}",
                        "buffer_type_bits": f"0x{imported.buffer_type_bits:x}",
                        "v5_sender_staging": staging_type_description(
                            kit.context, kit.sender_staging),
                        "v5_receiver_staging": staging_type_description(
                            kit.context, kit.receiver_staging),
                    }
                    print(f"  [{kit.label}] imported as "
                          f"type[{imported.memory_type_index}] "
                          f"{describe_memory_property_flags(imported.memory_type_flags)}"
                          f"  | v5 sender "
                          f"{size_report['import'][kit.label]['v5_sender_staging']}"
                          f"  | v5 receiver "
                          f"{size_report['import'][kit.label]['v5_receiver_staging']}")

                correctness = run_correctness(
                    kit_a, kit_b, block, args.rounds, generator)
                size_report["correctness"] = {
                    "transfers_checked": len(correctness),
                    "mismatched_transfers": 0,
                }
                print(f"  correctness: {len(correctness)} transfers "
                      f"({args.rounds} rounds x 2 directions), 0 mismatches")

                size_report["timing"] = {}
                for kit in kits:
                    timing = time_copies(kit, args.iterations)
                    size_report["timing"][kit.label] = timing
                    for label, values in timing.items():
                        print(f"  [{kit.label}] {label:<28} "
                              f"median {values['median_us']:>9.1f} us  "
                              f"(p10 {values['p10_us']:.1f} / "
                              f"p90 {values['p90_us']:.1f})  "
                              f"{values['gigabytes_per_second']} GB/s")
                report["sizes"][str(size)] = size_report
            finally:
                for kit in kits:
                    destroy_device_kit(kit)
                free_host_block(block)

        report["verdict"] = "PASS"
        exit_code = 0
        print("\n[probe] VERDICT: PASS — both devices import the same host "
              "block and the two-hop path is byte-exact in both directions")
    except BaseException as error:  # noqa: BLE001 — every failure is a result
        report["failure"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        print(f"\n[probe] VERDICT: FAIL — {report['failure']}", file=sys.stderr)
        traceback.print_exc()
    finally:
        for context in contexts:
            context.destroy()
        output_path = _REPO_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
        print(f"[probe] report -> {output_path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
