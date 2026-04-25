from vulkan import *


def create_sync_objects(device, max_frames, image_count):
    sem_info = VkSemaphoreCreateInfo()
    fence_info = VkFenceCreateInfo(flags=VK_FENCE_CREATE_SIGNALED_BIT)

    image_available = [vkCreateSemaphore(device, sem_info, None) for _ in range(max_frames)]
    render_finished = [vkCreateSemaphore(device, sem_info, None) for _ in range(image_count)]
    in_flight = [vkCreateFence(device, fence_info, None) for _ in range(max_frames)]

    return image_available, render_finished, in_flight
