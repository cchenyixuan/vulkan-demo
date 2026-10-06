"""
p2p_probe_linux.py — Linux counterpart of experiment/v6/_probe_p2p_interop.py (OPAQUE_WIN32 there).

Can two discrete GPUs of this node share device memory through Vulkan external memory?
  1. physical-device groups (vkEnumeratePhysicalDeviceGroups): GPUs in one group (NVLink/SLI-style
     peer access) could use VK_KHR_device_group; separate groups of one = no Vulkan peer path;
  2. OPAQUE_FD export / import between the first two discrete GPUs, then a data check: GPU 0 writes a
     pattern into an exported DEVICE_LOCAL buffer, GPU 1 imports the allocation, copies it to host-visible
     memory, and the bytes are compared. (The spec allows OPAQUE_FD import only between devices with the
     same deviceUUID / driverUUID, so a refusal between two physical GPUs is the expected outcome.)
"""

from __future__ import annotations

import os
import sys

from vulkan import *  # noqa: F401,F403
from vulkan._vulkancache import ffi

OPAQUE_FD = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
SIZE = 1 << 20
PATTERN = 0xA5
BUFFER_USAGE = (VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
DEVICE_EXTENSIONS = ["VK_KHR_external_memory", "VK_KHR_external_memory_fd", "VK_KHR_dedicated_allocation"]


def as_text(value) -> str:
    return value if isinstance(value, str) else ffi.string(value).decode("utf-8", "replace")


def queue_family_with_transfer(physical_device) -> int:
    for index, family in enumerate(vkGetPhysicalDeviceQueueFamilyProperties(physical_device)):
        if family.queueFlags & VK_QUEUE_TRANSFER_BIT:
            return index
    return 0


def create_device(physical_device, family):
    queue_info = VkDeviceQueueCreateInfo(queueFamilyIndex=family, queueCount=1, pQueuePriorities=[1.0])
    return vkCreateDevice(physical_device, VkDeviceCreateInfo(
        queueCreateInfoCount=1, pQueueCreateInfos=[queue_info],
        enabledExtensionCount=len(DEVICE_EXTENSIONS), ppEnabledExtensionNames=DEVICE_EXTENSIONS), None)


def memory_type(physical_device, type_bits, required_flags) -> int:
    properties = vkGetPhysicalDeviceMemoryProperties(physical_device)
    for index in range(properties.memoryTypeCount):
        if (type_bits & (1 << index)) and (properties.memoryTypes[index].propertyFlags & required_flags) == required_flags:
            return index
    return -1


def make_buffer(device, usage, exportable=False):
    next_structure = VkExternalMemoryBufferCreateInfo(handleTypes=OPAQUE_FD) if exportable else None
    return vkCreateBuffer(device, VkBufferCreateInfo(pNext=next_structure, size=SIZE, usage=usage,
                                                     sharingMode=VK_SHARING_MODE_EXCLUSIVE), None)


def host_buffer(physical_device, device):
    buffer = make_buffer(device, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)
    requirements = vkGetBufferMemoryRequirements(device, buffer)
    index = memory_type(physical_device, requirements.memoryTypeBits,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
    memory = vkAllocateMemory(device, VkMemoryAllocateInfo(allocationSize=requirements.size,
                                                           memoryTypeIndex=index), None)
    vkBindBufferMemory(device, buffer, memory, 0)
    return buffer, memory


def copy_and_wait(device, family, source, destination):
    pool = vkCreateCommandPool(device, VkCommandPoolCreateInfo(queueFamilyIndex=family), None)
    command_buffer = vkAllocateCommandBuffers(device, VkCommandBufferAllocateInfo(
        commandPool=pool, level=VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1))[0]
    vkBeginCommandBuffer(command_buffer, VkCommandBufferBeginInfo())
    vkCmdCopyBuffer(command_buffer, source, destination, 1, [VkBufferCopy(srcOffset=0, dstOffset=0, size=SIZE)])
    vkEndCommandBuffer(command_buffer)
    queue = vkGetDeviceQueue(device, family, 0)
    vkQueueSubmit(queue, 1, [VkSubmitInfo(commandBufferCount=1, pCommandBuffers=[command_buffer])], VK_NULL_HANDLE)
    vkQueueWaitIdle(queue)
    vkDestroyCommandPool(device, pool, None)


def main() -> int:
    application_info = VkApplicationInfo(pApplicationName="p2p_linux", applicationVersion=1, pEngineName="e30",
                                         engineVersion=1, apiVersion=VK_MAKE_VERSION(1, 3, 0))
    instance = vkCreateInstance(VkInstanceCreateInfo(pApplicationInfo=application_info), None)
    groups = vkEnumeratePhysicalDeviceGroups(instance)
    print(f"[p2p] physical-device groups: {len(groups)} -> sizes "
          f"{[int(group.physicalDeviceCount) for group in groups]}", flush=True)
    discrete = [physical_device for physical_device in vkEnumeratePhysicalDevices(instance)
                if vkGetPhysicalDeviceProperties(physical_device).deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU]
    if len(discrete) < 2:
        print(f"[p2p] need 2 discrete GPUs, found {len(discrete)}")
        return 0
    exporter, importer = discrete[0], discrete[1]
    print(f"[p2p] exporter {as_text(vkGetPhysicalDeviceProperties(exporter).deviceName)} -> importer "
          f"{as_text(vkGetPhysicalDeviceProperties(importer).deviceName)}", flush=True)
    exporter_family, importer_family = queue_family_with_transfer(exporter), queue_family_with_transfer(importer)
    exporter_device, importer_device = create_device(exporter, exporter_family), create_device(importer, importer_family)

    export_buffer = make_buffer(exporter_device, BUFFER_USAGE, exportable=True)
    export_requirements = vkGetBufferMemoryRequirements(exporter_device, export_buffer)
    export_index = memory_type(exporter, export_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    export_memory = vkAllocateMemory(exporter_device, VkMemoryAllocateInfo(
        pNext=VkMemoryDedicatedAllocateInfo(buffer=export_buffer, pNext=VkExportMemoryAllocateInfo(handleTypes=OPAQUE_FD)),
        allocationSize=export_requirements.size, memoryTypeIndex=export_index), None)
    vkBindBufferMemory(exporter_device, export_buffer, export_memory, 0)
    staging, staging_memory = host_buffer(exporter, exporter_device)
    pointer = vkMapMemory(exporter_device, staging_memory, 0, SIZE, 0)
    ffi.memmove(pointer, bytes([PATTERN]) * SIZE, SIZE)
    vkUnmapMemory(exporter_device, staging_memory)
    copy_and_wait(exporter_device, exporter_family, staging, export_buffer)
    get_fd = vkGetDeviceProcAddr(exporter_device, "vkGetMemoryFdKHR")
    file_descriptor = get_fd(exporter_device, VkMemoryGetFdInfoKHR(memory=export_memory, handleType=OPAQUE_FD))
    print(f"[p2p] exported OPAQUE_FD {file_descriptor} from GPU 0", flush=True)

    import_buffer = make_buffer(importer_device, BUFFER_USAGE, exportable=True)
    import_requirements = vkGetBufferMemoryRequirements(importer_device, import_buffer)
    import_index = memory_type(importer, import_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    try:
        import_memory = vkAllocateMemory(importer_device, VkMemoryAllocateInfo(
            pNext=VkMemoryDedicatedAllocateInfo(buffer=import_buffer, pNext=VkImportMemoryFdInfoKHR(
                handleType=OPAQUE_FD, fd=os.dup(file_descriptor))),
            allocationSize=export_requirements.size, memoryTypeIndex=import_index), None)
    except VkError as error:
        print(f"[p2p] RESULT import refused: {type(error).__name__} — no Vulkan memory sharing between these GPUs")
        return 0
    vkBindBufferMemory(importer_device, import_buffer, import_memory, 0)
    readback, readback_memory = host_buffer(importer, importer_device)
    copy_and_wait(importer_device, importer_family, import_buffer, readback)
    pointer = vkMapMemory(importer_device, readback_memory, 0, SIZE, 0)   # python-vulkan returns a cffi buffer
    data = bytes(pointer[:SIZE])
    vkUnmapMemory(importer_device, readback_memory)
    matching = sum(1 for byte in data if byte == PATTERN)
    verdict = ("data crossed (shared memory works)" if matching == SIZE else
               f"import accepted but data mismatch ({matching:,} of {SIZE:,} bytes match; first bytes {data[:8].hex()})")
    print(f"[p2p] RESULT import accepted; {verdict}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except VkError as error:
        print(f"[p2p] RESULT error {type(error).__name__}: {error}")
        sys.exit(0)
