"""
_test_shared_host_transport_v3.py — standalone smoke test for SharedHostTransport.

Runs entirely without the simulator / orchestrator. Validates:

  1. Two VulkanContextV3 with enable_shared_host_transport=True both come up
     with the required instance and device extensions enabled.
  2. SharedHostTransport.create() succeeds for two directions of realistic size.
  3. Sub-test A: ctx_a fills region_a_to_b with vkCmdFillBuffer (byte 0xa5)
     and signals semaphore_a_to_b on its compute queue; ctx_b waits the
     semaphore on its compute queue, copies region_a_to_b.dst_view.buffer
     into a HOST_VISIBLE staging buffer, and verifies all bytes are 0xa5.
  4. Sub-test B: same as A in the reverse direction (byte 0x5a, b -> a).
  5. destroy() is exception-safe (wraps in finally).

Run:

    .venv/Scripts/python.exe experiment/v3/_test_shared_host_transport_v3.py

Mirrors the structure of probe_external_memory_host.py + probe_interop_full.py,
but exercises the production SharedHostTransport API exactly as simulator_v3
will use it. If this passes on AMD+NV, the simulator-side integration is
strictly a matter of recording the right vkCmdCopyBuffer / signal / wait
instructions in phase_a / phase_c command buffers.
"""

from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vulkan import *  # noqa: F401, F403, E402

from experiment.v3.utils.vulkan_context_v3 import VulkanContextV3  # noqa: E402
from experiment.v3.utils.shared_host_transport_v3 import SharedHostTransport  # noqa: E402


# Realistic-ish single-direction ghost-region size: 6.4 MB matches 4M cavity.
_REGION_BYTES = 6_400_000


# ============================================================================
# Helpers
# ============================================================================

def _make_host_visible_staging(ctx: VulkanContextV3, size: int):
    """One-shot HOST_VISIBLE | HOST_COHERENT staging buffer for CPU readback
    of dst-side reads. Returns (buffer, memory)."""
    buffer = vkCreateBuffer(ctx.device, VkBufferCreateInfo(
        size=size,
        usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        sharingMode=VK_SHARING_MODE_EXCLUSIVE,
    ), None)
    requirements = vkGetBufferMemoryRequirements(ctx.device, buffer)
    memory_type_index = ctx.find_memory_type(
        type_bits=requirements.memoryTypeBits,
        required_properties=(
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
    )
    memory = vkAllocateMemory(ctx.device, VkMemoryAllocateInfo(
        allocationSize=requirements.size, memoryTypeIndex=memory_type_index,
    ), None)
    vkBindBufferMemory(ctx.device, buffer, memory, 0)
    return buffer, memory


def _record_and_submit_fill_and_signal(
    src_ctx: VulkanContextV3,
    target_buffer,
    fill_byte: int,
    fill_bytes: int,
    signal_semaphore,
) -> None:
    """Record (fill_buffer + signal) into a one-shot command buffer on src
    compute pool, submit on src compute queue with pSignalSemaphores=
    [signal_semaphore], wait for queue idle. Verifies the fill commits to
    host RAM before this returns."""
    command_buffer = vkAllocateCommandBuffers(
        src_ctx.device, VkCommandBufferAllocateInfo(
            commandPool=src_ctx.command_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1))[0]
    vkBeginCommandBuffer(command_buffer, VkCommandBufferBeginInfo(
        flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
    # vkCmdFillBuffer takes a 32-bit dword; replicate the byte pattern.
    dword = int.from_bytes(bytes([fill_byte]) * 4, "little")
    vkCmdFillBuffer(command_buffer, target_buffer, 0, fill_bytes, dword)
    vkEndCommandBuffer(command_buffer)

    vkQueueSubmit(src_ctx.compute_queue, 1, [VkSubmitInfo(
        commandBufferCount=1,
        pCommandBuffers=[command_buffer],
        signalSemaphoreCount=1,
        pSignalSemaphores=[signal_semaphore],
    )], None)
    # vkQueueWaitIdle here is the smoke-test equivalent of a fence wait;
    # in production simulator_v3 the semaphore wait on dst's queue makes
    # this idle wait unnecessary.
    vkQueueWaitIdle(src_ctx.compute_queue)
    vkFreeCommandBuffers(src_ctx.device, src_ctx.command_pool, 1, [command_buffer])


def _record_and_submit_wait_and_copy(
    dst_ctx: VulkanContextV3,
    wait_semaphore,
    source_buffer,
    destination_buffer,
    copy_bytes: int,
) -> None:
    """Record (copy_buffer) into a one-shot command buffer on dst compute
    pool, submit with pWaitSemaphores=[wait_semaphore]+pWaitDstStageMask=
    TRANSFER_BIT, wait for queue idle. The semaphore wait ensures the src
    fill is fully ordered before this copy starts."""
    command_buffer = vkAllocateCommandBuffers(
        dst_ctx.device, VkCommandBufferAllocateInfo(
            commandPool=dst_ctx.command_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1))[0]
    vkBeginCommandBuffer(command_buffer, VkCommandBufferBeginInfo(
        flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
    vkCmdCopyBuffer(command_buffer, source_buffer, destination_buffer, 1,
                    [VkBufferCopy(srcOffset=0, dstOffset=0, size=copy_bytes)])
    vkEndCommandBuffer(command_buffer)

    vkQueueSubmit(dst_ctx.compute_queue, 1, [VkSubmitInfo(
        commandBufferCount=1,
        pCommandBuffers=[command_buffer],
        waitSemaphoreCount=1,
        pWaitSemaphores=[wait_semaphore],
        pWaitDstStageMask=[VK_PIPELINE_STAGE_TRANSFER_BIT],
    )], None)
    vkQueueWaitIdle(dst_ctx.compute_queue)
    vkFreeCommandBuffers(dst_ctx.device, dst_ctx.command_pool, 1, [command_buffer])


def _verify_bytes(staging_memory, ctx: VulkanContextV3,
                  expected_byte: int, size: int, label: str) -> bool:
    """Map staging, compare to constant byte, return True on match."""
    mapped = vkMapMemory(ctx.device, staging_memory, 0, size, 0)
    try:
        readback = bytes(mapped[:size])
    finally:
        vkUnmapMemory(ctx.device, staging_memory)

    expected = bytes([expected_byte]) * size
    if readback == expected:
        print(f"  [PASS] {label}: all {size} bytes = 0x{expected_byte:02x}")
        return True

    first_mismatch_index = next(
        (index for index in range(size) if readback[index] != expected_byte), -1)
    print(f"  [FAIL] {label}: mismatch at byte {first_mismatch_index}: "
          f"expected 0x{expected_byte:02x}, got 0x{readback[first_mismatch_index]:02x}")
    return False


# ============================================================================
# Sub-test bodies
# ============================================================================

def run_direction(
    label: str,
    src_ctx: VulkanContextV3,
    dst_ctx: VulkanContextV3,
    region,
    cross_device_semaphore,
    fill_byte: int,
) -> bool:
    """One direction of the smoke test. Returns True if PASS."""
    src_buffer = region.src_view.buffer
    dst_buffer = region.dst_view.buffer
    region_bytes = region.size

    print(f"\n--- Sub-test {label}: src fills + signals, dst waits + reads ---")

    # 1. src fills shared region via vkCmdFillBuffer + signals semaphore.
    _record_and_submit_fill_and_signal(
        src_ctx=src_ctx,
        target_buffer=src_buffer,
        fill_byte=fill_byte,
        fill_bytes=region_bytes,
        signal_semaphore=cross_device_semaphore.src_semaphore,
    )

    # 2. dst waits semaphore + copies shared region into a HOST_VISIBLE staging.
    staging_buffer, staging_memory = _make_host_visible_staging(
        dst_ctx, region_bytes)
    try:
        _record_and_submit_wait_and_copy(
            dst_ctx=dst_ctx,
            wait_semaphore=cross_device_semaphore.dst_semaphore,
            source_buffer=dst_buffer,
            destination_buffer=staging_buffer,
            copy_bytes=region_bytes,
        )
        return _verify_bytes(
            staging_memory, dst_ctx, fill_byte, region_bytes,
            label=f"{label} ({src_ctx.device_name} -> {dst_ctx.device_name})")
    finally:
        vkDestroyBuffer(dst_ctx.device, staging_buffer, None)
        vkFreeMemory(dst_ctx.device, staging_memory, None)


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    print("=" * 72)
    print("V3.3 SharedHostTransport smoke test")
    print("=" * 72)

    print("\n[1/4] Creating two VulkanContextV3 with enable_shared_host_transport=True")
    ctx_a = VulkanContextV3.create(
        application_name="sht_test_a",
        enable_validation=True,
        device_index=0,
        enable_shared_host_transport=True,
    )
    ctx_b = VulkanContextV3.create(
        application_name="sht_test_b",
        enable_validation=True,
        device_index=1,
        enable_shared_host_transport=True,
    )

    if ctx_a.device_name == ctx_b.device_name:
        print(f"  WARNING: both contexts ended up on the same physical device "
              f"({ctx_a.device_name}); this still exercises the import path "
              f"but doesn't validate the cross-vendor case.")

    transport = None
    overall_pass = True
    try:
        print("\n[2/4] Allocating SharedHostTransport (two directions, "
              f"{_REGION_BYTES} bytes each)")
        transport = SharedHostTransport.create(
            ctx_a=ctx_a,
            ctx_b=ctx_b,
            a_to_b_bytes=_REGION_BYTES,
            b_to_a_bytes=_REGION_BYTES,
        )

        print("\n[3/4] Running per-direction GPU fill + GPU read tests")
        overall_pass &= run_direction(
            label="a_to_b",
            src_ctx=ctx_a,
            dst_ctx=ctx_b,
            region=transport.region_a_to_b,
            cross_device_semaphore=transport.semaphore_a_to_b,
            fill_byte=0xA5,
        )
        overall_pass &= run_direction(
            label="b_to_a",
            src_ctx=ctx_b,
            dst_ctx=ctx_a,
            region=transport.region_b_to_a,
            cross_device_semaphore=transport.semaphore_b_to_a,
            fill_byte=0x5A,
        )

        print("\n[4/4] Cleanup")
    finally:
        if transport is not None:
            transport.destroy()
        ctx_b.destroy()
        ctx_a.destroy()

    print("\n" + "=" * 72)
    if overall_pass:
        print("RESULT: PASS — SharedHostTransport works end-to-end on this hardware pair.")
        print("Ready to integrate into simulator_v3 phase_a / phase_c.")
    else:
        print("RESULT: FAIL — see per-test FAIL lines above.")
    print("=" * 72)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
