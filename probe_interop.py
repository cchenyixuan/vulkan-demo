"""Cross-vendor OPAQUE_WIN32 memory interop test (DEVICE_LOCAL + staging).

External memory is typically restricted to DEVICE_LOCAL. Test flow:
  1. Source: create exportable DEVICE_LOCAL buffer + HOST_VISIBLE staging buffer;
     write pattern to staging, cmdCopyBuffer to the exportable buffer.
  2. Export Win32 HANDLE from source's memory.
  3. Destination: create a DEVICE_LOCAL buffer bound to imported memory + staging buffer;
     cmdCopyBuffer from imported buffer to staging, map & verify on CPU.
"""
from vulkan import *


INSTANCE_EXTS = [
    "VK_KHR_get_physical_device_properties2",
    "VK_KHR_external_memory_capabilities",
]
DEVICE_EXTS = [
    "VK_KHR_external_memory",
    "VK_KHR_external_memory_win32",
]

HANDLE_TYPE = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
BUF_SIZE = 1024
BUF_USAGE = (
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
    | VK_BUFFER_USAGE_TRANSFER_DST_BIT
)


def create_instance():
    info = VkInstanceCreateInfo(
        pApplicationInfo=VkApplicationInfo(
            pApplicationName="interop",
            applicationVersion=VK_MAKE_VERSION(1, 0, 0),
            pEngineName="interop",
            engineVersion=VK_MAKE_VERSION(1, 0, 0),
            apiVersion=VK_MAKE_VERSION(1, 1, 0),
        ),
        enabledExtensionCount=len(INSTANCE_EXTS),
        ppEnabledExtensionNames=INSTANCE_EXTS,
    )
    return vkCreateInstance(info, None)


def find_queue_family(pd):
    for i, q in enumerate(vkGetPhysicalDeviceQueueFamilyProperties(pd)):
        if q.queueFlags & VK_QUEUE_GRAPHICS_BIT:
            return i
    return 0


def create_device(pd, qf):
    q_info = VkDeviceQueueCreateInfo(
        queueFamilyIndex=qf, queueCount=1, pQueuePriorities=[1.0],
    )
    info = VkDeviceCreateInfo(
        queueCreateInfoCount=1, pQueueCreateInfos=[q_info],
        enabledExtensionCount=len(DEVICE_EXTS), ppEnabledExtensionNames=DEVICE_EXTS,
    )
    return vkCreateDevice(pd, info, None)


def find_mem_type(pd, type_filter, properties):
    mp = vkGetPhysicalDeviceMemoryProperties(pd)
    for i in range(mp.memoryTypeCount):
        if (type_filter & (1 << i)) and (mp.memoryTypes[i].propertyFlags & properties) == properties:
            return i
    return None


def make_buffer(device, size, usage, p_next=None):
    info = VkBufferCreateInfo(
        pNext=p_next,
        size=size,
        usage=usage,
        sharingMode=VK_SHARING_MODE_EXCLUSIVE,
    )
    return vkCreateBuffer(device, info, None)


def alloc_mem(device, mem_size, mem_type_idx, p_next=None):
    info = VkMemoryAllocateInfo(
        pNext=p_next,
        allocationSize=mem_size,
        memoryTypeIndex=mem_type_idx,
    )
    return vkAllocateMemory(device, info, None)


def make_staging(device, pd, size):
    buf = make_buffer(device, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)
    req = vkGetBufferMemoryRequirements(device, buf)
    idx = find_mem_type(
        pd, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    )
    if idx is None:
        raise RuntimeError("no HOST_VISIBLE|COHERENT staging memory")
    mem = alloc_mem(device, req.size, idx)
    vkBindBufferMemory(device, buf, mem, 0)
    return buf, mem


def make_exportable(device, pd, size):
    ext_buf = VkExternalMemoryBufferCreateInfo(handleTypes=HANDLE_TYPE)
    buf = make_buffer(device, size, BUF_USAGE, p_next=ext_buf)
    req = vkGetBufferMemoryRequirements(device, buf)
    idx = find_mem_type(pd, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    if idx is None:
        vkDestroyBuffer(device, buf, None)
        return None, None, 0, "source has no DEVICE_LOCAL memory type matching buffer requirements"

    export = VkExportMemoryAllocateInfo(handleTypes=HANDLE_TYPE)
    mem = alloc_mem(device, req.size, idx, p_next=export)
    vkBindBufferMemory(device, buf, mem, 0)
    return buf, mem, req.size, None


def make_imported(device, pd, size, handle, alloc_size):
    ext_buf = VkExternalMemoryBufferCreateInfo(handleTypes=HANDLE_TYPE)
    buf = make_buffer(device, size, BUF_USAGE, p_next=ext_buf)
    req = vkGetBufferMemoryRequirements(device, buf)
    idx = find_mem_type(pd, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    if idx is None:
        vkDestroyBuffer(device, buf, None)
        return None, None, "destination has no DEVICE_LOCAL memory type matching imported buffer"

    imp = VkImportMemoryWin32HandleInfoKHR(handleType=HANDLE_TYPE, handle=handle, name=None)
    mem = alloc_mem(device, alloc_size, idx, p_next=imp)
    vkBindBufferMemory(device, buf, mem, 0)
    return buf, mem, None


def export_handle(device, mem):
    fn = vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR")
    if fn is None:
        raise RuntimeError("vkGetMemoryWin32HandleKHR not loaded")
    return fn(device, VkMemoryGetWin32HandleInfoKHR(memory=mem, handleType=HANDLE_TYPE))


def gpu_copy(device, queue, qf, src, dst, size):
    pool = vkCreateCommandPool(device, VkCommandPoolCreateInfo(queueFamilyIndex=qf), None)
    cmd = vkAllocateCommandBuffers(device, VkCommandBufferAllocateInfo(
        commandPool=pool, level=VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1,
    ))[0]
    vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
    vkCmdCopyBuffer(cmd, src, dst, 1, [VkBufferCopy(size=size)])
    vkEndCommandBuffer(cmd)
    vkQueueSubmit(queue, 1, [VkSubmitInfo(commandBufferCount=1, pCommandBuffers=[cmd])], None)
    vkQueueWaitIdle(queue)
    vkFreeCommandBuffers(device, pool, 1, [cmd])
    vkDestroyCommandPool(device, pool, None)


def test_direction(instance, src_name, src_pd, dst_name, dst_pd):
    print(f"  {src_name}  ->  {dst_name}")

    src_qf = find_queue_family(src_pd)
    dst_qf = find_queue_family(dst_pd)
    src_dev = create_device(src_pd, src_qf)
    dst_dev = create_device(dst_pd, dst_qf)
    src_q = vkGetDeviceQueue(src_dev, src_qf, 0)
    dst_q = vkGetDeviceQueue(dst_dev, dst_qf, 0)

    try:
        # 1. allocate exportable on source
        src_buf, src_mem, alloc_size, err = make_exportable(src_dev, src_pd, BUF_SIZE)
        if err:
            print(f"    [FAIL] {err}")
            return

        # 2. write pattern to a staging buffer and copy into exportable buffer
        stg_buf, stg_mem = make_staging(src_dev, src_pd, BUF_SIZE)
        pattern = bytes((i * 37 + 13) & 0xFF for i in range(BUF_SIZE))
        mapped = vkMapMemory(src_dev, stg_mem, 0, BUF_SIZE, 0)
        mapped[:BUF_SIZE] = pattern
        vkUnmapMemory(src_dev, stg_mem)
        gpu_copy(src_dev, src_q, src_qf, stg_buf, src_buf, BUF_SIZE)
        vkDestroyBuffer(src_dev, stg_buf, None)
        vkFreeMemory(src_dev, stg_mem, None)

        # 3. export handle
        try:
            handle = export_handle(src_dev, src_mem)
        except Exception as e:
            print(f"    [FAIL] export: {type(e).__name__}: {e}")
            return

        # 4. import on destination
        try:
            dst_buf, dst_mem, err = make_imported(dst_dev, dst_pd, BUF_SIZE, handle, alloc_size)
        except Exception as e:
            print(f"    [FAIL] import: {type(e).__name__}: {e}")
            return
        if err:
            print(f"    [FAIL] {err}")
            return

        # 5. copy imported buffer to dst staging and verify
        dst_stg_buf, dst_stg_mem = make_staging(dst_dev, dst_pd, BUF_SIZE)
        gpu_copy(dst_dev, dst_q, dst_qf, dst_buf, dst_stg_buf, BUF_SIZE)
        mapped = vkMapMemory(dst_dev, dst_stg_mem, 0, BUF_SIZE, 0)
        readback = bytes(mapped[:BUF_SIZE])
        vkUnmapMemory(dst_dev, dst_stg_mem)

        if readback == pattern:
            print("    [OK] cross-vendor memory sharing WORKS in this direction")
        else:
            first_diff = next((i for i in range(BUF_SIZE) if readback[i] != pattern[i]), -1)
            print(f"    [FAIL] data mismatch at offset {first_diff}: "
                  f"expected 0x{pattern[first_diff]:02x}, got 0x{readback[first_diff]:02x}")

        vkDestroyBuffer(dst_dev, dst_stg_buf, None)
        vkFreeMemory(dst_dev, dst_stg_mem, None)
        vkDestroyBuffer(dst_dev, dst_buf, None)
        vkFreeMemory(dst_dev, dst_mem, None)
        vkDestroyBuffer(src_dev, src_buf, None)
        vkFreeMemory(src_dev, src_mem, None)
    finally:
        vkDestroyDevice(dst_dev, None)
        vkDestroyDevice(src_dev, None)


def main():
    instance = create_instance()
    devs = []
    for pd in vkEnumeratePhysicalDevices(instance):
        props = vkGetPhysicalDeviceProperties(pd)
        if props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            devs.append((props.deviceName, pd))

    if len(devs) < 2:
        print(f"need 2 discrete GPUs, found {len(devs)}")
        return

    print("OPAQUE_WIN32 cross-vendor interop test (DEVICE_LOCAL + staging)")
    print("=" * 60)
    test_direction(instance, devs[0][0], devs[0][1], devs[1][0], devs[1][1])
    print()
    test_direction(instance, devs[1][0], devs[1][1], devs[0][0], devs[0][1])

    vkDestroyInstance(instance, None)


if __name__ == "__main__":
    main()
