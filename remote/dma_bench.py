"""
dma_bench.py — raw device<->host DMA bandwidth per GPU (transfer queue).

Answers the K>=3 collapse hardware question directly: allocates a 64 MB
device-local buffer + host-visible staging, runs timed vkCmdCopyBuffer
batches on the DEDICATED transfer queue (same path as V5 ghost transport),
both directions, and reports effective GB/s per visible GPU — plus the
PCIe link generation nvidia-smi sees DURING the load.

    source ~/swq/env.sh && python remote/dma_bench.py
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import time

_REPO = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vulkan import *  # noqa: F401,F403

from experiment.v5.utils.vulkan_context_v5 import VulkanContextV5  # noqa: E402

COPY_BYTES = 64 * 1024 * 1024
REPEATS = 24


def make_buffer(ctx, size, usage, host_visible):
    buffer_handle = vkCreateBuffer(ctx.device, VkBufferCreateInfo(
        size=size, usage=usage, sharingMode=VK_SHARING_MODE_EXCLUSIVE), None)
    requirements = vkGetBufferMemoryRequirements(ctx.device, buffer_handle)
    if host_visible:
        required = (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                    | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
    else:
        required = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    type_index = ctx.find_memory_type(requirements.memoryTypeBits, required)
    memory = vkAllocateMemory(ctx.device, VkMemoryAllocateInfo(
        allocationSize=requirements.size, memoryTypeIndex=type_index), None)
    vkBindBufferMemory(ctx.device, buffer_handle, memory, 0)
    return buffer_handle, memory


def timed_copies(ctx, src, dst, label):
    pool = vkCreateCommandPool(ctx.device, VkCommandPoolCreateInfo(
        flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        queueFamilyIndex=ctx.transfer_queue_family_index), None)
    cmd = vkAllocateCommandBuffers(ctx.device, VkCommandBufferAllocateInfo(
        commandPool=pool, level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        commandBufferCount=1))[0]
    fence = vkCreateFence(ctx.device, VkFenceCreateInfo(), None)

    def one_copy():
        vkResetFences(ctx.device, 1, [fence])
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
        vkCmdCopyBuffer(cmd, src, dst, 1,
                        [VkBufferCopy(srcOffset=0, dstOffset=0, size=COPY_BYTES)])
        vkEndCommandBuffer(cmd)
        vkQueueSubmit(ctx.transfer_queue, 1, VkSubmitInfo(
            commandBufferCount=1, pCommandBuffers=[cmd]), fence)
        vkWaitForFences(ctx.device, 1, [fence], VK_TRUE, 10**10)

    one_copy()                                  # warm up link speed
    t_start = time.perf_counter()
    for _ in range(REPEATS):
        one_copy()
    elapsed = time.perf_counter() - t_start
    rate = COPY_BYTES * REPEATS / elapsed / 1e9
    print(f"    {label}: {rate:6.2f} GB/s "
          f"({elapsed/REPEATS*1000:.2f} ms per {COPY_BYTES>>20} MB)", flush=True)
    vkDestroyFence(ctx.device, fence, None)
    vkDestroyCommandPool(ctx.device, pool, None)
    return rate


def link_gen_now():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,pcie.link.gen.current",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return result.stdout.strip().replace("\n", " | ")
    except OSError:
        return "n/a"


def main() -> int:
    device_index = 0
    while True:
        try:
            ctx = VulkanContextV5.create(device_index=device_index,
                                         enable_validation=False,
                                         application_name="dma_bench")
        except RuntimeError:
            break
        print(f"[dma] GPU {device_index}: {ctx.device_name}", flush=True)
        device_buf, device_mem = make_buffer(
            ctx, COPY_BYTES,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            host_visible=False)
        host_buf, host_mem = make_buffer(
            ctx, COPY_BYTES,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            host_visible=True)
        timed_copies(ctx, device_buf, host_buf, "device->host (readback)")
        print(f"    link gen during load: {link_gen_now()}", flush=True)
        timed_copies(ctx, host_buf, device_buf, "host->device (upload)")
        for handle, memory in ((device_buf, device_mem), (host_buf, host_mem)):
            vkDestroyBuffer(ctx.device, handle, None)
            vkFreeMemory(ctx.device, memory, None)
        ctx.destroy()
        device_index += 1
    print(f"[dma] tested {device_index} GPUs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
