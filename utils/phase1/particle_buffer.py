from vulkan import *
from vulkan._vulkancache import ffi


def find_memory_type(physical_device, type_filter, properties):
    mem_props = vkGetPhysicalDeviceMemoryProperties(physical_device)
    for i in range(mem_props.memoryTypeCount):
        if (type_filter & (1 << i)) and (mem_props.memoryTypes[i].propertyFlags & properties) == properties:
            return i
    raise RuntimeError(f"No suitable memory type (filter={type_filter:#x}, props={properties:#x})")


def create_buffer(device, physical_device, size, usage, mem_props):
    buf_info = VkBufferCreateInfo(
        size=size,
        usage=usage,
        sharingMode=VK_SHARING_MODE_EXCLUSIVE,
    )
    buf = vkCreateBuffer(device, buf_info, None)
    req = vkGetBufferMemoryRequirements(device, buf)
    alloc_info = VkMemoryAllocateInfo(
        allocationSize=req.size,
        memoryTypeIndex=find_memory_type(physical_device, req.memoryTypeBits, mem_props),
    )
    mem = vkAllocateMemory(device, alloc_info, None)
    vkBindBufferMemory(device, buf, mem, 0)
    return buf, mem


def create_device_local_buffer(device, physical_device, size, usage):
    return create_buffer(
        device, physical_device, size,
        usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    )


def create_host_visible_buffer(device, physical_device, size, usage):
    """Creates a HOST_VISIBLE | HOST_COHERENT buffer and maps it permanently.
    Returns (buf, mem, mapped_ptr)."""
    buf, mem = create_buffer(
        device, physical_device, size, usage,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    )
    mapped = vkMapMemory(device, mem, 0, size, 0)
    return buf, mem, mapped


def upload_to_device_local(device, physical_device, command_pool, queue, device_buf, data):
    """One-shot staging copy from CPU bytes to a device-local buffer."""
    size = len(data)
    staging_buf, staging_mem = create_buffer(
        device, physical_device, size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    )
    mapped = vkMapMemory(device, staging_mem, 0, size, 0)
    mapped[:size] = data
    vkUnmapMemory(device, staging_mem)

    alloc_info = VkCommandBufferAllocateInfo(
        commandPool=command_pool,
        level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        commandBufferCount=1,
    )
    cmd = vkAllocateCommandBuffers(device, alloc_info)[0]
    vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
    vkCmdCopyBuffer(cmd, staging_buf, device_buf, 1, [VkBufferCopy(size=size)])
    vkEndCommandBuffer(cmd)
    vkQueueSubmit(queue, 1, [VkSubmitInfo(commandBufferCount=1, pCommandBuffers=[cmd])], None)
    vkQueueWaitIdle(queue)
    vkFreeCommandBuffers(device, command_pool, 1, [cmd])

    vkDestroyBuffer(device, staging_buf, None)
    vkFreeMemory(device, staging_mem, None)


def destroy_buffer(device, buf, mem):
    vkDestroyBuffer(device, buf, None)
    vkFreeMemory(device, mem, None)
