import glfw
from vulkan import *


def query_swapchain_support(physical_device, surface, inst_fns):
    capabilities = inst_fns["vkGetPhysicalDeviceSurfaceCapabilitiesKHR"](physical_device, surface)
    formats = inst_fns["vkGetPhysicalDeviceSurfaceFormatsKHR"](physical_device, surface)
    present_modes = inst_fns["vkGetPhysicalDeviceSurfacePresentModesKHR"](physical_device, surface)
    return capabilities, formats, present_modes


def choose_surface_format(formats):
    for f in formats:
        if f.format == VK_FORMAT_B8G8R8A8_SRGB and f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR:
            return f
    return formats[0]


def choose_present_mode(present_modes):
    for mode in present_modes:
        if mode == VK_PRESENT_MODE_MAILBOX_KHR:
            return mode
    return VK_PRESENT_MODE_FIFO_KHR


def choose_extent(capabilities, window):
    if capabilities.currentExtent.width != 0xFFFFFFFF:
        w, h = capabilities.currentExtent.width, capabilities.currentExtent.height
    else:
        w, h = glfw.get_framebuffer_size(window)
        w = max(capabilities.minImageExtent.width, min(capabilities.maxImageExtent.width, w))
        h = max(capabilities.minImageExtent.height, min(capabilities.maxImageExtent.height, h))
    return VkExtent2D(width=w, height=h)


def create_swapchain(device, physical_device, surface, window,
                     graphics_family, present_family, inst_fns, dev_fns):
    capabilities, formats, present_modes = query_swapchain_support(physical_device, surface, inst_fns)
    surface_format = choose_surface_format(formats)
    present_mode = choose_present_mode(present_modes)
    extent = choose_extent(capabilities, window)

    image_count = capabilities.minImageCount + 1
    if capabilities.maxImageCount > 0 and image_count > capabilities.maxImageCount:
        image_count = capabilities.maxImageCount

    if graphics_family != present_family:
        sharing_mode = VK_SHARING_MODE_CONCURRENT
        family_indices = [graphics_family, present_family]
    else:
        sharing_mode = VK_SHARING_MODE_EXCLUSIVE
        family_indices = []

    create_info = VkSwapchainCreateInfoKHR(
        surface=surface,
        minImageCount=image_count,
        imageFormat=surface_format.format,
        imageColorSpace=surface_format.colorSpace,
        imageExtent=extent,
        imageArrayLayers=1,
        imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        imageSharingMode=sharing_mode,
        queueFamilyIndexCount=len(family_indices),
        pQueueFamilyIndices=family_indices if family_indices else None,
        preTransform=capabilities.currentTransform,
        compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        presentMode=present_mode,
        clipped=VK_TRUE,
        oldSwapchain=None,
    )
    swapchain = dev_fns["vkCreateSwapchainKHR"](device, create_info, None)
    images = dev_fns["vkGetSwapchainImagesKHR"](device, swapchain)
    return swapchain, images, surface_format.format, extent


def create_image_views(device, images, image_format):
    views = []
    for image in images:
        create_info = VkImageViewCreateInfo(
            image=image,
            viewType=VK_IMAGE_VIEW_TYPE_2D,
            format=image_format,
            components=VkComponentMapping(
                r=VK_COMPONENT_SWIZZLE_IDENTITY,
                g=VK_COMPONENT_SWIZZLE_IDENTITY,
                b=VK_COMPONENT_SWIZZLE_IDENTITY,
                a=VK_COMPONENT_SWIZZLE_IDENTITY,
            ),
            subresourceRange=VkImageSubresourceRange(
                aspectMask=VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0,
                levelCount=1,
                baseArrayLayer=0,
                layerCount=1,
            ),
        )
        views.append(vkCreateImageView(device, create_info, None))
    return views
