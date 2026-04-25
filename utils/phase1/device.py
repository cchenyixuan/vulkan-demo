from vulkan import *


def find_queue_families(physical_device, surface, surface_support_fn):
    queue_families = vkGetPhysicalDeviceQueueFamilyProperties(physical_device)
    graphics_family = None
    present_family = None
    for i, qf in enumerate(queue_families):
        if qf.queueFlags & VK_QUEUE_GRAPHICS_BIT:
            graphics_family = i
        if surface_support_fn(physical_device, i, surface):
            present_family = i
        if graphics_family is not None and present_family is not None:
            break
    if graphics_family is None or present_family is None:
        raise RuntimeError("Could not find suitable queue families")
    return graphics_family, present_family


def create_logical_device(physical_device, graphics_family, present_family):
    unique_families = set([graphics_family, present_family])
    queue_create_infos = [
        VkDeviceQueueCreateInfo(
            queueFamilyIndex=family,
            queueCount=1,
            pQueuePriorities=[1.0],
        )
        for family in unique_families
    ]
    device_extensions = ["VK_KHR_swapchain"]
    create_info = VkDeviceCreateInfo(
        queueCreateInfoCount=len(queue_create_infos),
        pQueueCreateInfos=queue_create_infos,
        enabledExtensionCount=len(device_extensions),
        ppEnabledExtensionNames=device_extensions,
    )
    device = vkCreateDevice(physical_device, create_info, None)
    graphics_queue = vkGetDeviceQueue(device, graphics_family, 0)
    present_queue = vkGetDeviceQueue(device, present_family, 0)
    return device, graphics_queue, present_queue


def load_device_functions(device):
    names = [
        "vkCreateSwapchainKHR",
        "vkDestroySwapchainKHR",
        "vkGetSwapchainImagesKHR",
        "vkAcquireNextImageKHR",
        "vkQueuePresentKHR",
    ]
    fns = {}
    for name in names:
        fns[name] = vkGetDeviceProcAddr(device, name)
    return fns
