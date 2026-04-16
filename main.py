import glfw
from vulkan import *

from utils.instance import create_instance, create_surface, load_instance_functions
from utils.device import select_physical_device, find_queue_families, create_logical_device, load_device_functions
from utils.swapchain import create_swapchain, create_image_views
from utils.pipeline import create_render_pass, create_graphics_pipeline
from utils.commands import create_command_pool, create_framebuffers, allocate_command_buffers, record_command_buffer
from utils.sync import create_sync_objects

MAX_FRAMES_IN_FLIGHT = 2


class VulkanApp:
    def __init__(self):
        self.window = None
        self.instance = None
        self.surface = None
        self.inst_fns = {}
        self.dev_fns = {}
        self.physical_device = None
        self.device = None
        self.graphics_queue = None
        self.present_queue = None
        self.graphics_family = None
        self.present_family = None
        self.swapchain = None
        self.swapchain_images = None
        self.swapchain_image_format = None
        self.swapchain_extent = None
        self.image_views = None
        self.render_pass = None
        self.pipeline_layout = None
        self.graphics_pipeline = None
        self.framebuffers = None
        self.command_pool = None
        self.command_buffers = None
        self.image_available_semaphores = None
        self.render_finished_semaphores = None
        self.in_flight_fences = None
        self.current_frame = 0
        self.framebuffer_resized = False

    def init_window(self):
        glfw.init()
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        self.window = glfw.create_window(800, 600, "Vulkan Triangle", None, None)
        glfw.set_framebuffer_size_callback(self.window, self._framebuffer_resize_callback)

    def _framebuffer_resize_callback(self, window, width, height):
        self.framebuffer_resized = True

    def init_vulkan(self):
        self.instance = create_instance()
        self.surface = create_surface(self.instance, self.window)
        self.inst_fns = load_instance_functions(self.instance)
        self.physical_device = select_physical_device(self.instance)
        self.graphics_family, self.present_family = find_queue_families(
            self.physical_device, self.surface, self.inst_fns["vkGetPhysicalDeviceSurfaceSupportKHR"]
        )
        self.device, self.graphics_queue, self.present_queue = create_logical_device(
            self.physical_device, self.graphics_family, self.present_family
        )
        self.dev_fns = load_device_functions(self.device)
        self._create_swapchain_resources()
        self.command_pool = create_command_pool(self.device, self.graphics_family)
        self.command_buffers = allocate_command_buffers(self.device, self.command_pool, MAX_FRAMES_IN_FLIGHT)
        self.image_available_semaphores, self.render_finished_semaphores, self.in_flight_fences = (
            create_sync_objects(self.device, MAX_FRAMES_IN_FLIGHT, len(self.swapchain_images))
        )

    def _create_swapchain_resources(self):
        self.swapchain, self.swapchain_images, self.swapchain_image_format, self.swapchain_extent = (
            create_swapchain(
                self.device, self.physical_device, self.surface, self.window,
                self.graphics_family, self.present_family, self.inst_fns, self.dev_fns
            )
        )
        self.image_views = create_image_views(self.device, self.swapchain_images, self.swapchain_image_format)
        self.render_pass = create_render_pass(self.device, self.swapchain_image_format)
        self.pipeline_layout, self.graphics_pipeline = create_graphics_pipeline(
            self.device, self.render_pass, self.swapchain_extent
        )
        self.framebuffers = create_framebuffers(
            self.device, self.render_pass, self.image_views, self.swapchain_extent
        )

    def _cleanup_swapchain_resources(self):
        for fb in self.framebuffers:
            vkDestroyFramebuffer(self.device, fb, None)
        vkDestroyPipeline(self.device, self.graphics_pipeline, None)
        vkDestroyPipelineLayout(self.device, self.pipeline_layout, None)
        vkDestroyRenderPass(self.device, self.render_pass, None)
        for view in self.image_views:
            vkDestroyImageView(self.device, view, None)
        self.dev_fns["vkDestroySwapchainKHR"](self.device, self.swapchain, None)

    def recreate_swapchain(self):
        width, height = glfw.get_framebuffer_size(self.window)
        while width == 0 or height == 0:
            glfw.wait_events()
            width, height = glfw.get_framebuffer_size(self.window)
        vkDeviceWaitIdle(self.device)
        self._cleanup_swapchain_resources()
        for sem in self.render_finished_semaphores:
            vkDestroySemaphore(self.device, sem, None)
        self._create_swapchain_resources()
        sem_info = VkSemaphoreCreateInfo()
        self.render_finished_semaphores = [
            vkCreateSemaphore(self.device, sem_info, None)
            for _ in range(len(self.swapchain_images))
        ]

    def draw_frame(self):
        vkWaitForFences(self.device, 1, [self.in_flight_fences[self.current_frame]], VK_TRUE, UINT64_MAX)

        try:
            image_index = self.dev_fns["vkAcquireNextImageKHR"](
                self.device, self.swapchain, UINT64_MAX,
                self.image_available_semaphores[self.current_frame], None
            )
        except (VkErrorOutOfDateKhr, VkSuboptimalKhr):
            self.recreate_swapchain()
            return

        vkResetFences(self.device, 1, [self.in_flight_fences[self.current_frame]])
        vkResetCommandBuffer(self.command_buffers[self.current_frame], 0)
        record_command_buffer(
            self.command_buffers[self.current_frame],
            self.render_pass,
            self.framebuffers[image_index],
            self.swapchain_extent,
            self.graphics_pipeline,
        )

        submit_info = VkSubmitInfo(
            waitSemaphoreCount=1,
            pWaitSemaphores=[self.image_available_semaphores[self.current_frame]],
            pWaitDstStageMask=[VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT],
            commandBufferCount=1,
            pCommandBuffers=[self.command_buffers[self.current_frame]],
            signalSemaphoreCount=1,
            pSignalSemaphores=[self.render_finished_semaphores[image_index]],
        )
        vkQueueSubmit(self.graphics_queue, 1, [submit_info], self.in_flight_fences[self.current_frame])

        present_info = VkPresentInfoKHR(
            waitSemaphoreCount=1,
            pWaitSemaphores=[self.render_finished_semaphores[image_index]],
            swapchainCount=1,
            pSwapchains=[self.swapchain],
            pImageIndices=[image_index],
        )
        try:
            self.dev_fns["vkQueuePresentKHR"](self.present_queue, present_info)
        except (VkErrorOutOfDateKhr, VkSuboptimalKhr):
            self.framebuffer_resized = False
            self.recreate_swapchain()
            return

        if self.framebuffer_resized:
            self.framebuffer_resized = False
            self.recreate_swapchain()
            return

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT

    def main_loop(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.draw_frame()
        vkDeviceWaitIdle(self.device)

    def cleanup(self):
        for sem in self.image_available_semaphores:
            vkDestroySemaphore(self.device, sem, None)
        for sem in self.render_finished_semaphores:
            vkDestroySemaphore(self.device, sem, None)
        for fence in self.in_flight_fences:
            vkDestroyFence(self.device, fence, None)
        vkDestroyCommandPool(self.device, self.command_pool, None)
        self._cleanup_swapchain_resources()
        vkDestroyDevice(self.device, None)
        self.inst_fns["vkDestroySurfaceKHR"](self.instance, self.surface, None)
        vkDestroyInstance(self.instance, None)
        glfw.destroy_window(self.window)
        glfw.terminate()

    def run(self):
        self.init_window()
        self.init_vulkan()
        self.main_loop()
        self.cleanup()


if __name__ == "__main__":
    app = VulkanApp()
    app.run()
