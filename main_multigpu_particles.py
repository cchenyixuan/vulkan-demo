"""Phase 1: two GPUs, two windows, particles cross the X=1 boundary via CPU-staged migration.

Per-frame pipeline:
  - wait both GPU fences
  - read each GPU's outgoing buffer (host-visible)
  - route: NV.incoming <- AMD.outgoing, AMD.incoming <- NV.outgoing
  - update counters/indirect for next frame
  - record+submit+present on both GPUs (compute -> barrier -> drawIndirect)
"""
import math
import struct
import time

import glfw
import numpy as np
from vulkan import *
from vulkan._vulkancache import ffi

from utils.instance import create_instance, create_debug_messenger, load_instance_functions, create_surface
from utils.device import find_queue_families, create_logical_device, load_device_functions
from utils.swapchain import create_swapchain, create_image_views
from utils.pipeline import (
    create_render_pass,
    create_particle_graphics_pipeline,
    PARTICLE_PUSH_CONSTANT_SIZE,
)
from utils.compute_pipeline import (
    create_compute_descriptor_set_layout,
    create_graphics_descriptor_set_layout,
    create_compute_pipeline,
    create_descriptor_pool,
    allocate_compute_descriptor_set,
    allocate_graphics_descriptor_set,
    COMPUTE_PUSH_CONSTANT_SIZE,
)
from utils.commands import create_command_pool, create_framebuffers, allocate_command_buffers
from utils.sync import create_sync_objects
from utils.particle_buffer import (
    create_device_local_buffer,
    create_host_visible_buffer,
    upload_to_device_local,
    destroy_buffer,
)


VENDOR_NVIDIA = 0x10DE
VENDOR_AMD = 0x1002

PARTICLE_STRUCT_SIZE = 32       # vec2 pos + vec2 vel + uint gen + 3 pad uints
N_MAX = 11000000                # per-GPU buffer capacity (safety margin over N_INITIAL_NV)
N_INITIAL_NV = 10000000         # all particles start on NV
WORKGROUP_SIZE = 256
DISPATCH_GROUPS = (N_MAX + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE
HALF_SIZE = 0.002
HUE_STEP = 0.15                 # colour shift per crossing
MAX_DT = 1.0 / 60.0

PARTICLES_BUF_BYTES   = N_MAX * PARTICLE_STRUCT_SIZE
MIGRATION_BUF_BYTES   = N_MAX * PARTICLE_STRUCT_SIZE
COUNTERS_BYTES        = 16      # 4 uints
INDIRECT_BYTES        = 16      # VkDrawIndirectCommand


def find_discrete_gpu(instance, vendor_id):
    for dev in vkEnumeratePhysicalDevices(instance):
        props = vkGetPhysicalDeviceProperties(dev)
        if props.vendorID == vendor_id and props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            return dev, props.deviceName
    return None, None


def make_grouped_particles(n, x_center, y_center, half_extent,
                           angle_deg, speed, angular_spread_deg, speed_spread, seed):
    """Particles in a square around (x_center, y_center), moving at angle_deg with small spread."""
    rng = np.random.default_rng(seed)
    pos = np.stack([
        rng.uniform(x_center - half_extent, x_center + half_extent, n),
        rng.uniform(y_center - half_extent, y_center + half_extent, n),
    ], axis=1).astype(np.float32)
    angles = np.radians(
        rng.uniform(angle_deg - angular_spread_deg, angle_deg + angular_spread_deg, n)
    )
    speeds = rng.uniform(speed * (1 - speed_spread), speed * (1 + speed_spread), n)
    vel = np.stack([speeds * np.cos(angles), speeds * np.sin(angles)], axis=1).astype(np.float32)

    arr = np.zeros((n, 8), dtype=np.uint32)
    arr[:, 0] = pos[:, 0].view(np.uint32)
    arr[:, 1] = pos[:, 1].view(np.uint32)
    arr[:, 2] = vel[:, 0].view(np.uint32)
    arr[:, 3] = vel[:, 1].view(np.uint32)
    # gen (col 4) and padding (5..7) stay zero
    buf = np.zeros((N_MAX, 8), dtype=np.uint32)
    buf[:n] = arr
    return buf.tobytes()


def push_ptr(data):
    return ffi.cast("void*", ffi.from_buffer(data))


class ParticleRenderer:
    def __init__(self, instance, physical_device, window, inst_fns,
                 x_min, x_max, y_min, y_max, is_left, base_hue,
                 n_initial, initial_bytes):
        self.instance = instance
        self.physical_device = physical_device
        self.window = window
        self.inst_fns = inst_fns
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.is_left = 1 if is_left else 0
        self.base_hue = base_hue
        self.n_initial = n_initial
        self.initial_bytes = initial_bytes

        self.surface = None
        self.device = None
        self.dev_fns = {}
        self.graphics_queue = None
        self.present_queue = None
        self.graphics_family = None

        self.swapchain = None
        self.swapchain_images = None
        self.swapchain_image_format = None
        self.swapchain_extent = None
        self.image_views = None
        self.render_pass = None
        self.framebuffers = None

        self.particles = [None, None]      # ping, pong  (buf, mem)
        self.outgoing = None               # (buf, mem, mapped)
        self.incoming = None
        self.counters = None
        self.indirect = None

        self.compute_ssbo_layout = None
        self.graphics_ssbo_layout = None
        self.desc_pool = None
        self.compute_sets = [None, None]   # even, odd parity
        self.graphics_sets = [None, None]

        self.compute_pipeline_layout = None
        self.compute_pipeline = None
        self.graphics_pipeline_layout = None
        self.graphics_pipeline = None

        self.command_pool = None
        self.command_buffer = None
        self.image_available_semaphore = None
        self.render_finished_semaphores = None
        self.in_flight_fence = None

        self.parity = 0
        self.framebuffer_resized = False
        self.pending_incoming = 0       # particles buffered in `incoming` not yet consumed by compute

    def init(self):
        self.surface = create_surface(self.instance, self.window)
        gfx, present = find_queue_families(
            self.physical_device, self.surface,
            self.inst_fns["vkGetPhysicalDeviceSurfaceSupportKHR"],
        )
        self.graphics_family = gfx
        self.device, self.graphics_queue, self.present_queue = create_logical_device(
            self.physical_device, gfx, present,
        )
        self.dev_fns = load_device_functions(self.device)
        self.command_pool = create_command_pool(self.device, gfx)

        # ---- buffers ----
        ping_buf, ping_mem = create_device_local_buffer(
            self.device, self.physical_device, PARTICLES_BUF_BYTES,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )
        pong_buf, pong_mem = create_device_local_buffer(
            self.device, self.physical_device, PARTICLES_BUF_BYTES,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )
        self.particles[0] = (ping_buf, ping_mem)
        self.particles[1] = (pong_buf, pong_mem)

        if self.n_initial > 0 and self.initial_bytes is not None:
            upload_to_device_local(
                self.device, self.physical_device, self.command_pool, self.graphics_queue,
                ping_buf, self.initial_bytes,
            )

        self.outgoing = create_host_visible_buffer(
            self.device, self.physical_device, MIGRATION_BUF_BYTES,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )
        self.incoming = create_host_visible_buffer(
            self.device, self.physical_device, MIGRATION_BUF_BYTES,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )
        self.counters = create_host_visible_buffer(
            self.device, self.physical_device, COUNTERS_BYTES,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )
        self.indirect = create_host_visible_buffer(
            self.device, self.physical_device, INDIRECT_BYTES,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        )

        # Initial counter values: in=n_initial, incoming=0, out=0, outgoing=0
        self.counters[2][:COUNTERS_BYTES] = struct.pack("4I", self.n_initial, 0, 0, 0)
        # Initial indirect: vertexCount=4, instanceCount=0, firstVertex=0, firstInstance=0
        self.indirect[2][:INDIRECT_BYTES] = struct.pack("4I", 4, 0, 0, 0)

        # ---- descriptors ----
        self.compute_ssbo_layout = create_compute_descriptor_set_layout(self.device)
        self.graphics_ssbo_layout = create_graphics_descriptor_set_layout(self.device)
        # 4 sets (2 compute + 2 graphics); SSBO descriptors: 2*6 + 2*1 = 14
        self.desc_pool = create_descriptor_pool(self.device, max_sets=4, ssbo_descriptors=14)

        sizes = {
            "particles": PARTICLES_BUF_BYTES,
            "migration": MIGRATION_BUF_BYTES,
            "counters":  COUNTERS_BYTES,
            "indirect":  INDIRECT_BYTES,
        }
        # parity 0: in=ping, out=pong
        self.compute_sets[0] = allocate_compute_descriptor_set(
            self.device, self.desc_pool, self.compute_ssbo_layout,
            ping_buf, pong_buf,
            self.outgoing[0], self.incoming[0], self.counters[0], self.indirect[0], sizes,
        )
        # parity 1: in=pong, out=ping
        self.compute_sets[1] = allocate_compute_descriptor_set(
            self.device, self.desc_pool, self.compute_ssbo_layout,
            pong_buf, ping_buf,
            self.outgoing[0], self.incoming[0], self.counters[0], self.indirect[0], sizes,
        )
        # graphics reads whatever compute wrote:
        # parity 0 output is pong, parity 1 output is ping
        self.graphics_sets[0] = allocate_graphics_descriptor_set(
            self.device, self.desc_pool, self.graphics_ssbo_layout, pong_buf, PARTICLES_BUF_BYTES,
        )
        self.graphics_sets[1] = allocate_graphics_descriptor_set(
            self.device, self.desc_pool, self.graphics_ssbo_layout, ping_buf, PARTICLES_BUF_BYTES,
        )

        # ---- pipelines ----
        self.compute_pipeline_layout, self.compute_pipeline = create_compute_pipeline(
            self.device, self.compute_ssbo_layout,
        )

        self._create_swapchain_resources()

        # ---- command & sync ----
        self.command_buffer = allocate_command_buffers(self.device, self.command_pool, 1)[0]
        ia, rf, ifs = create_sync_objects(self.device, 1, len(self.swapchain_images))
        self.image_available_semaphore = ia[0]
        self.render_finished_semaphores = rf
        self.in_flight_fence = ifs[0]

        glfw.set_framebuffer_size_callback(
            self.window, lambda _w, _width, _height: setattr(self, "framebuffer_resized", True),
        )

    def _create_swapchain_resources(self):
        self.swapchain, self.swapchain_images, self.swapchain_image_format, self.swapchain_extent = (
            create_swapchain(
                self.device, self.physical_device, self.surface, self.window,
                self.graphics_family, self.graphics_family, self.inst_fns, self.dev_fns,
            )
        )
        self.image_views = create_image_views(
            self.device, self.swapchain_images, self.swapchain_image_format,
        )
        self.render_pass = create_render_pass(self.device, self.swapchain_image_format)
        self.graphics_pipeline_layout, self.graphics_pipeline = create_particle_graphics_pipeline(
            self.device, self.render_pass, self.graphics_ssbo_layout,
        )
        self.framebuffers = create_framebuffers(
            self.device, self.render_pass, self.image_views, self.swapchain_extent,
        )

    def _cleanup_swapchain_resources(self):
        for fb in self.framebuffers:
            vkDestroyFramebuffer(self.device, fb, None)
        vkDestroyPipeline(self.device, self.graphics_pipeline, None)
        vkDestroyPipelineLayout(self.device, self.graphics_pipeline_layout, None)
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
        vkDestroySemaphore(self.device, self.image_available_semaphore, None)
        self._create_swapchain_resources()
        sem_info = VkSemaphoreCreateInfo()
        self.render_finished_semaphores = [
            vkCreateSemaphore(self.device, sem_info, None)
            for _ in range(len(self.swapchain_images))
        ]
        self.image_available_semaphore = vkCreateSemaphore(self.device, sem_info, None)

    def read_counters(self):
        return struct.unpack("4I", bytes(self.counters[2][:COUNTERS_BYTES]))

    def read_and_clear_outgoing(self):
        """Returns (bytes, count) and zeros outgoing_count so the same batch is not re-routed."""
        _, _, _, og = self.read_counters()
        if og == 0:
            return b"", 0
        data = bytes(self.outgoing[2][:og * PARTICLE_STRUCT_SIZE])
        # Zero only the outgoing_count field (offset 12)
        self.counters[2][12:16] = struct.pack("I", 0)
        return data, og

    def append_incoming(self, data, count):
        """Append `count` particles to the incoming buffer, preserving any already pending
        (un-consumed) particles from a previous failed submit. Returns running total."""
        if count > 0:
            offset = self.pending_incoming * PARTICLE_STRUCT_SIZE
            self.incoming[2][offset:offset + count * PARTICLE_STRUCT_SIZE] = data
            self.pending_incoming += count
        return self.pending_incoming

    def _set_counters(self, in_count, incoming_count):
        self.counters[2][:COUNTERS_BYTES] = struct.pack("4I", in_count, incoming_count, 0, 0)

    def _reset_indirect(self):
        self.indirect[2][:INDIRECT_BYTES] = struct.pack("4I", 4, 0, 0, 0)

    def _record(self, cmd, image_index, dt):
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo())

        compute_pc = struct.pack(
            "5fI", dt, self.x_min, self.x_max, self.y_min, self.y_max, self.is_left,
        )
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, self.compute_pipeline)
        vkCmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_COMPUTE, self.compute_pipeline_layout,
            0, 1, [self.compute_sets[self.parity]], 0, None,
        )
        vkCmdPushConstants(
            cmd, self.compute_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, COMPUTE_PUSH_CONSTANT_SIZE, push_ptr(compute_pc),
        )
        vkCmdDispatch(cmd, DISPATCH_GROUPS, 1, 1)

        out_buf = self.particles[1 - self.parity][0]  # compute writes to "out"
        barriers = [
            VkBufferMemoryBarrier(
                srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
                dstAccessMask=VK_ACCESS_SHADER_READ_BIT,
                srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
                buffer=out_buf, offset=0, size=PARTICLES_BUF_BYTES,
            ),
            VkBufferMemoryBarrier(
                srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
                dstAccessMask=VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
                srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
                buffer=self.indirect[0], offset=0, size=INDIRECT_BYTES,
            ),
        ]
        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            0, 0, None, len(barriers), barriers, 0, None,
        )

        clear = VkClearValue(color=VkClearColorValue(float32=[0.04, 0.04, 0.07, 1.0]))
        vkCmdBeginRenderPass(
            cmd,
            VkRenderPassBeginInfo(
                renderPass=self.render_pass,
                framebuffer=self.framebuffers[image_index],
                renderArea=VkRect2D(offset=VkOffset2D(x=0, y=0), extent=self.swapchain_extent),
                clearValueCount=1,
                pClearValues=[clear],
            ),
            VK_SUBPASS_CONTENTS_INLINE,
        )
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, self.graphics_pipeline)
        vkCmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, self.graphics_pipeline_layout,
            0, 1, [self.graphics_sets[self.parity]], 0, None,
        )
        vkCmdSetViewport(cmd, 0, 1, [VkViewport(
            x=0.0, y=0.0,
            width=float(self.swapchain_extent.width),
            height=float(self.swapchain_extent.height),
            minDepth=0.0, maxDepth=1.0,
        )])
        vkCmdSetScissor(cmd, 0, 1, [VkRect2D(
            offset=VkOffset2D(x=0, y=0), extent=self.swapchain_extent,
        )])

        graphics_pc = struct.pack(
            "8f",
            self.x_min, self.y_min, self.x_max, self.y_max,
            HALF_SIZE, self.base_hue, HUE_STEP, 0.0,
        )
        vkCmdPushConstants(
            cmd, self.graphics_pipeline_layout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0, PARTICLE_PUSH_CONSTANT_SIZE, push_ptr(graphics_pc),
        )
        vkCmdDrawIndirect(cmd, self.indirect[0], 0, 1, 0)
        vkCmdEndRenderPass(cmd)
        vkEndCommandBuffer(cmd)

    def submit(self, dt, in_count):
        """Submit one frame. Counters are reset AFTER acquire succeeds, so acquire-failure
        leaves counters intact (preserving the last successful compute's out/outgoing values
        for the next retry). Consumes `self.pending_incoming` if submit proceeds."""
        try:
            image_index = self.dev_fns["vkAcquireNextImageKHR"](
                self.device, self.swapchain, UINT64_MAX,
                self.image_available_semaphore, None,
            )
        except (VkErrorOutOfDateKhr, VkSuboptimalKhr):
            self.recreate_swapchain()
            return False  # compute did NOT run; counters/parity/pending_incoming unchanged

        # Committed: compute will run this frame. Prepare state.
        self._set_counters(in_count, self.pending_incoming)
        self._reset_indirect()
        self.pending_incoming = 0

        vkResetFences(self.device, 1, [self.in_flight_fence])
        vkResetCommandBuffer(self.command_buffer, 0)
        self._record(self.command_buffer, image_index, dt)

        submit_info = VkSubmitInfo(
            waitSemaphoreCount=1,
            pWaitSemaphores=[self.image_available_semaphore],
            pWaitDstStageMask=[VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT],
            commandBufferCount=1,
            pCommandBuffers=[self.command_buffer],
            signalSemaphoreCount=1,
            pSignalSemaphores=[self.render_finished_semaphores[image_index]],
        )
        vkQueueSubmit(self.graphics_queue, 1, [submit_info], self.in_flight_fence)
        # Compute WILL run now. Flip parity so next frame reads from the buffer we just wrote.
        self.parity ^= 1

        present = VkPresentInfoKHR(
            waitSemaphoreCount=1,
            pWaitSemaphores=[self.render_finished_semaphores[image_index]],
            swapchainCount=1,
            pSwapchains=[self.swapchain],
            pImageIndices=[image_index],
        )
        try:
            self.dev_fns["vkQueuePresentKHR"](self.present_queue, present)
        except (VkErrorOutOfDateKhr, VkSuboptimalKhr):
            self.framebuffer_resized = False
            self.recreate_swapchain()
            return True  # compute DID run; caller should treat state as advanced

        if self.framebuffer_resized:
            self.framebuffer_resized = False
            self.recreate_swapchain()
            return True

        return True

    def wait_done(self):
        vkWaitForFences(self.device, 1, [self.in_flight_fence], VK_TRUE, UINT64_MAX)

    def cleanup(self):
        vkDeviceWaitIdle(self.device)
        vkDestroySemaphore(self.device, self.image_available_semaphore, None)
        for sem in self.render_finished_semaphores:
            vkDestroySemaphore(self.device, sem, None)
        vkDestroyFence(self.device, self.in_flight_fence, None)
        self._cleanup_swapchain_resources()
        vkDestroyPipeline(self.device, self.compute_pipeline, None)
        vkDestroyPipelineLayout(self.device, self.compute_pipeline_layout, None)
        vkDestroyDescriptorPool(self.device, self.desc_pool, None)
        vkDestroyDescriptorSetLayout(self.device, self.compute_ssbo_layout, None)
        vkDestroyDescriptorSetLayout(self.device, self.graphics_ssbo_layout, None)
        for buf, mem in self.particles:
            destroy_buffer(self.device, buf, mem)
        for triple in (self.outgoing, self.incoming, self.counters, self.indirect):
            vkUnmapMemory(self.device, triple[1])
            destroy_buffer(self.device, triple[0], triple[1])
        vkDestroyCommandPool(self.device, self.command_pool, None)
        vkDestroyDevice(self.device, None)
        self.inst_fns["vkDestroySurfaceKHR"](self.instance, self.surface, None)


def main():
    glfw.init()
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)

    instance = create_instance()
    messenger = create_debug_messenger(instance)
    inst_fns = load_instance_functions(instance)

    nv_dev, nv_name = find_discrete_gpu(instance, VENDOR_NVIDIA)
    amd_dev, amd_name = find_discrete_gpu(instance, VENDOR_AMD)
    if not nv_dev or not amd_dev:
        raise RuntimeError("Need both NVIDIA and AMD discrete GPUs")
    print(f"NV:  {nv_name}")
    print(f"AMD: {amd_name}")

    win_w, win_h = 600, 600
    nv_window = glfw.create_window(win_w, win_h, "NVIDIA [0, 1]", None, None)
    amd_window = glfw.create_window(win_w, win_h, "AMD [1, 2]", None, None)
    glfw.set_window_pos(nv_window, 100, 200)
    glfw.set_window_pos(amd_window, 100 + win_w, 200)

    nv_initial = make_grouped_particles(
        n=N_INITIAL_NV,
        x_center=0.2, y_center=0.5, half_extent=0.15,
        angle_deg=0, speed=0.5,
        angular_spread_deg=3, speed_spread=0.05,
        seed=11,
    )
    nv = ParticleRenderer(
        instance, nv_dev, nv_window, inst_fns,
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
        is_left=True, base_hue=0.33,
        n_initial=N_INITIAL_NV, initial_bytes=nv_initial,
    )
    amd = ParticleRenderer(
        instance, amd_dev, amd_window, inst_fns,
        x_min=1.0, x_max=2.0, y_min=0.0, y_max=1.0,
        is_left=False, base_hue=0.05,
        n_initial=0, initial_bytes=None,
    )

    nv.init()
    amd.init()
    print("Both GPUs initialized. Entering render loop...")

    last = time.perf_counter()
    fps_t0 = last
    fps_frames = 0
    while not glfw.window_should_close(nv_window) and not glfw.window_should_close(amd_window):
        glfw.poll_events()
        now = time.perf_counter()
        dt = min(now - last, MAX_DT)
        last = now

        # Always wait for both GPUs' previous frame to finish.
        # On the very first iteration fences are pre-signaled, so this is a no-op.
        nv.wait_done()
        amd.wait_done()

        # Read counters from the last successful compute (or initial values on first iter).
        nv_in, _, nv_out, _ = nv.read_counters()
        amd_in, _, amd_out, _ = amd.read_counters()

        # Read & clear outgoing atomically on the CPU side — prevents double-routing if the
        # next submit happens to acquire-fail and we loop back to read counters again.
        nv_outgoing_data, nv_og = nv.read_and_clear_outgoing()
        amd_outgoing_data, amd_og = amd.read_and_clear_outgoing()

        # Cross-route: append the other's outgoing to this GPU's incoming. Append (not overwrite)
        # preserves any particles that were pending from a previous failed submit.
        nv.append_incoming(amd_outgoing_data, amd_og)
        amd.append_incoming(nv_outgoing_data, nv_og)

        # Sentinel: compute has run iff out>0 or og>0 was observed in this read. First iter:
        # both 0, fall back to the initial in_count that was written into the counter buffer.
        nv_next_in = nv_out if (nv_out + nv_og > 0) else nv_in
        amd_next_in = amd_out if (amd_out + amd_og > 0) else amd_in

        # submit() internally sets counters AFTER acquire succeeds, so on acquire failure the
        # pending-incoming buffer and counters stay valid for retry.
        nv.submit(dt, nv_next_in)
        amd.submit(dt, amd_next_in)

        fps_frames += 1
        if now - fps_t0 >= 1.0:
            nv_c = nv.read_counters()
            amd_c = amd.read_counters()
            total_nv = nv_c[0] + nv_c[1]
            total_amd = amd_c[0] + amd_c[1]
            print(f"fps={fps_frames / (now - fps_t0):5.1f}  NV={total_nv:7d}  AMD={total_amd:7d}  total={total_nv+total_amd}")
            fps_t0 = now
            fps_frames = 0

    # Drain before cleanup
    nv.wait_done()
    amd.wait_done()

    amd.cleanup()
    nv.cleanup()
    if messenger and "vkDestroyDebugUtilsMessengerEXT" in inst_fns:
        inst_fns["vkDestroyDebugUtilsMessengerEXT"](instance, messenger, None)
    vkDestroyInstance(instance, None)
    glfw.destroy_window(amd_window)
    glfw.destroy_window(nv_window)
    glfw.terminate()


if __name__ == "__main__":
    main()
