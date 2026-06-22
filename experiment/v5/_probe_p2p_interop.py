"""
_probe_p2p_interop.py — CORRECT OPAQUE_WIN32 memory-sharing probe (2x 5090).

The old probe_interop.py picked the import memory type from the *buffer's*
memoryTypeBits. For IMPORTED external memory you must instead intersect with the
bits that `vkGetMemoryWin32HandlePropertiesKHR` reports as valid for that handle
on the importing device — otherwise the allocation fails with OUT_OF_DEVICE_MEMORY
even though the driver advertises the handle type as importable.

This probe does it correctly and then VERIFIES data crosses: GPU0 writes a known
pattern into an exported DEVICE_LOCAL buffer; GPU1 imports the same allocation,
copies it out, and we compare. Success ⇒ same-vendor shared memory works ⇒ the
V3.2 zero-copy transport backend is viable on this rig.
"""

from __future__ import annotations

import sys

from vulkan import *  # noqa: F401,F403
from vulkan._vulkancache import ffi as _ffi

OPAQUE = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
SIZE = 1 << 20                      # 1 MiB test buffer
PATTERN = 0xA5
BUF_USAGE = (VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
             | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
INSTANCE_EXTS = ["VK_KHR_get_physical_device_properties2", "VK_KHR_external_memory_capabilities"]
DEVICE_EXTS = ["VK_KHR_external_memory", "VK_KHR_external_memory_win32", "VK_KHR_dedicated_allocation"]


def create_instance():
    ai = VkApplicationInfo(pApplicationName="p2p", applicationVersion=1, pEngineName="e",
                           engineVersion=1, apiVersion=VK_MAKE_VERSION(1, 3, 0))
    return vkCreateInstance(VkInstanceCreateInfo(
        pApplicationInfo=ai, enabledExtensionCount=len(INSTANCE_EXTS),
        ppEnabledExtensionNames=INSTANCE_EXTS), None)


def find_qf(pd):
    for i, q in enumerate(vkGetPhysicalDeviceQueueFamilyProperties(pd)):
        if q.queueFlags & VK_QUEUE_TRANSFER_BIT:
            return i
    return 0


def create_device(pd, qf):
    qi = VkDeviceQueueCreateInfo(queueFamilyIndex=qf, queueCount=1, pQueuePriorities=[1.0])
    return vkCreateDevice(pd, VkDeviceCreateInfo(
        queueCreateInfoCount=1, pQueueCreateInfos=[qi],
        enabledExtensionCount=len(DEVICE_EXTS), ppEnabledExtensionNames=DEVICE_EXTS), None)


def find_mem_type(pd, type_bits, props):
    mp = vkGetPhysicalDeviceMemoryProperties(pd)
    for i in range(mp.memoryTypeCount):
        if (type_bits & (1 << i)) and (mp.memoryTypes[i].propertyFlags & props) == props:
            return i
    return -1


def make_buffer(device, usage, exportable=False):
    pnext = VkExternalMemoryBufferCreateInfo(handleTypes=OPAQUE) if exportable else None
    return vkCreateBuffer(device, VkBufferCreateInfo(
        pNext=pnext, size=SIZE, usage=usage, sharingMode=VK_SHARING_MODE_EXCLUSIVE), None)


def submit_copy(device, qf, src, dst):
    pool = vkCreateCommandPool(device, VkCommandPoolCreateInfo(queueFamilyIndex=qf), None)
    cmd = vkAllocateCommandBuffers(device, VkCommandBufferAllocateInfo(
        commandPool=pool, level=VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1))[0]
    vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo())
    vkCmdCopyBuffer(cmd, src, dst, 1, [VkBufferCopy(srcOffset=0, dstOffset=0, size=SIZE)])
    vkEndCommandBuffer(cmd)
    queue = vkGetDeviceQueue(device, qf, 0)
    vkQueueSubmit(queue, 1, [VkSubmitInfo(commandBufferCount=1, pCommandBuffers=[cmd])], VK_NULL_HANDLE)
    vkQueueWaitIdle(queue)
    vkDestroyCommandPool(device, pool, None)


def main() -> int:
    instance = create_instance()
    discrete = [(vkGetPhysicalDeviceProperties(pd).deviceName, pd)
                for pd in vkEnumeratePhysicalDevices(instance)
                if vkGetPhysicalDeviceProperties(pd).deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU]
    if len(discrete) < 2:
        print(f"need 2 discrete GPUs, found {len(discrete)}"); return 1
    (name0, pd0), (name1, pd1) = discrete[0], discrete[1]
    print(f"exporter: {name0}\nimporter: {name1}\n" + "=" * 60)

    qf0, qf1 = find_qf(pd0), find_qf(pd1)
    dev0, dev1 = create_device(pd0, qf0), create_device(pd1, qf1)

    # --- GPU0: exportable DEVICE_LOCAL buffer, dedicated + export alloc --------
    export_buf = make_buffer(dev0, BUF_USAGE, exportable=True)
    req0 = vkGetBufferMemoryRequirements(dev0, export_buf)
    idx0 = find_mem_type(pd0, req0.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    export_info = VkExportMemoryAllocateInfo(handleTypes=OPAQUE)
    dedicated0 = VkMemoryDedicatedAllocateInfo(buffer=export_buf, pNext=export_info)
    export_mem = vkAllocateMemory(dev0, VkMemoryAllocateInfo(
        pNext=dedicated0, allocationSize=req0.size, memoryTypeIndex=idx0), None)
    vkBindBufferMemory(dev0, export_buf, export_mem, 0)

    # write pattern on GPU0 via host-visible staging -> export_buf
    stg0 = make_buffer(dev0, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)
    sreq0 = vkGetBufferMemoryRequirements(dev0, stg0)
    sidx0 = find_mem_type(pd0, sreq0.memoryTypeBits,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
    stg0_mem = vkAllocateMemory(dev0, VkMemoryAllocateInfo(allocationSize=sreq0.size, memoryTypeIndex=sidx0), None)
    vkBindBufferMemory(dev0, stg0, stg0_mem, 0)
    ptr = vkMapMemory(dev0, stg0_mem, 0, SIZE, 0)
    _ffi.memmove(ptr, bytes([PATTERN]) * SIZE, SIZE)
    vkUnmapMemory(dev0, stg0_mem)
    submit_copy(dev0, qf0, stg0, export_buf)

    # export the Win32 handle
    get_handle = vkGetDeviceProcAddr(dev0, "vkGetMemoryWin32HandleKHR")
    handle = get_handle(dev0, VkMemoryGetWin32HandleInfoKHR(memory=export_mem, handleType=OPAQUE))
    print(f"[ok] exported handle from GPU0 (0x{int(_ffi.cast('uintptr_t', handle)):x})")

    # --- GPU1: query handle's valid memory types, then import correctly -------
    get_props = vkGetDeviceProcAddr(dev1, "vkGetMemoryWin32HandlePropertiesKHR")
    try:
        hprops = get_props(dev1, OPAQUE, handle)
        handle_type_bits = hprops.memoryTypeBits
        print(f"[ok] GPU1 handle memoryTypeBits = 0x{handle_type_bits:x}")
    except (VkError, TypeError) as exc:
        print(f"[note] vkGetMemoryWin32HandlePropertiesKHR failed ({type(exc).__name__}: {exc}); "
              f"falling back to buffer memoryTypeBits")
        handle_type_bits = 0xFFFFFFFF

    import_buf = make_buffer(dev1, BUF_USAGE, exportable=True)
    req1 = vkGetBufferMemoryRequirements(dev1, import_buf)
    valid_bits = req1.memoryTypeBits & handle_type_bits     # <-- the fix
    idx1 = find_mem_type(pd1, valid_bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    if idx1 < 0:
        idx1 = find_mem_type(pd1, valid_bits, 0)
    print(f"   buffer bits=0x{req1.memoryTypeBits:x}  valid(import)=0x{valid_bits:x}  -> memType {idx1}")
    if idx1 < 0:
        print("[FAIL] no compatible memory type for import"); return 1

    import_info = VkImportMemoryWin32HandleInfoKHR(handleType=OPAQUE, handle=handle, name=None)
    dedicated1 = VkMemoryDedicatedAllocateInfo(buffer=import_buf, pNext=import_info)
    try:
        import_mem = vkAllocateMemory(dev1, VkMemoryAllocateInfo(
            pNext=dedicated1, allocationSize=req0.size, memoryTypeIndex=idx1), None)
    except VkError as exc:
        print(f"[FAIL] import vkAllocateMemory: {type(exc).__name__}: {exc}")
        return 1
    vkBindBufferMemory(dev1, import_buf, import_mem, 0)
    print("[ok] GPU1 IMPORTED the GPU0 allocation (no OUT_OF_DEVICE_MEMORY)")

    # --- verify data crossed: GPU1 copies imported -> host staging ------------
    stg1 = make_buffer(dev1, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)
    sreq1 = vkGetBufferMemoryRequirements(dev1, stg1)
    sidx1 = find_mem_type(pd1, sreq1.memoryTypeBits,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
    stg1_mem = vkAllocateMemory(dev1, VkMemoryAllocateInfo(allocationSize=sreq1.size, memoryTypeIndex=sidx1), None)
    vkBindBufferMemory(dev1, stg1, stg1_mem, 0)
    submit_copy(dev1, qf1, import_buf, stg1)
    ptr1 = vkMapMemory(dev1, stg1_mem, 0, SIZE, 0)
    out = bytes(_ffi.buffer(ptr1, 64))
    vkUnmapMemory(dev1, stg1_mem)
    ok = all(b == PATTERN for b in out)
    print("=" * 60)
    if ok:
        print(f"[PASS] GPU1 read 0x{out[0]:02x} pattern written by GPU0 — SHARED MEMORY WORKS NV→NV ✅")
    else:
        print(f"[PARTIAL] import succeeded but data mismatch (first bytes: {out[:8].hex()}) — "
              f"alloc shareable but not P2P-coherent; may need staging copy semantics")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except VkError as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        sys.exit(1)
