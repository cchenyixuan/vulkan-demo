"""
simulator_v2.py — V2 per-GPU SPH simulator.

V2 differences vs V1 (see docs/sph_v2_design.md):

  - Timeline semaphore (uint64 counter, 3 values per frame: 3N+1 / 3N+2 / 3N+3)
    replaces V1's fence-wait round trips.

  - 3 pre-recorded cmd buffers per frame (§14.1 / §5):
        phase_a: predict → update_voxel → ghost_send → readback   (signal 3N+1)
        phase_b: correction(INTERIOR)                              (queue-ordered)
        phase_c: upload → install_migration → correction(BOUNDARY) → density →
                 force                                              (wait 3N+2 signal 3N+3)

  - correction.comp split into INTERIOR + BOUNDARY pipelines (CORRECTION_MODE
    spec const id=47). Phase B runs INTERIOR; Phase C runs BOUNDARY after
    install_migration.

  - 3-hop CPU-staged transport (§14.5). Sender staging HOST_CACHED, receiver
    HOST_COHERENT; sim owns both stagings per direction. Readback / upload
    vkCmdCopyBuffer folded into phase_a / phase_c.

  - apiVersion=1.3 (sync2 core), shader target-env=vulkan1.2 (SPIR-V 1.5).

V2 is fully self-contained: no imports from experiment/v1/ or utils/sph/.
"""

from __future__ import annotations

import pathlib
import struct
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from vulkan import *  # noqa: F401, F403
from vulkan._vulkancache import ffi

from experiment.v2.utils.case_v2 import (
    CaseV2,
    KIND_FLUID,
)
from experiment.v2.utils.vulkan_context_v2 import VulkanContextV2


# ============================================================================
# Constants — descriptor binding layout (matches shaders/v2/common.glsl)
# ============================================================================

# Buffers that defrag.comp writes into a scratch SoA before copy-back.
# Same set as V1: set 0 bindings 0/1/3/4/5/6/7/8/9 (everything except scratch
# 2 which IS the scratch).
DEFRAG_SET0_BINDINGS = (0, 1, 3, 4, 5, 6, 7, 8, 9)

# Ghost transport: 9 SoA fields (same set as defrag — skip binding 2 scratch
# which is transient) + 2 set-1 buffers (inside_particle_count + index) + 1
# set-3 count field (ghost_send_*_count → ghost_recv_*_count). Total 12
# segments per direction; matches V1's transport_cpu_staging.py.
TRANSPORT_SET0_BINDINGS = DEFRAG_SET0_BINDINGS

# Buffer name → byte stride per particle (set 0 SoA).
_SET0_BYTE_STRIDES = {
    "position_voxel_id":           16,
    "density_pressure":             8,
    "velocity_mass":               16,
    "acceleration":                16,
    "shift":                       16,
    "material":                     4,
    "correction_inverse":          32,
    "density_gradient_kernel_sum": 16,
    "extension_fields":            16,
}
_SET0_BINDING_TO_NAME = {
    0: "position_voxel_id",
    1: "density_pressure",
    3: "velocity_mass",
    4: "acceleration",
    5: "shift",
    6: "material",
    7: "correction_inverse",
    8: "density_gradient_kernel_sum",
    9: "extension_fields",
}

# GlobalStatusBuffer field offsets (16 × uint, see common.glsl §3)
_OFFSET_GHOST_SEND_LEADING   = 32   # field [8]
_OFFSET_GHOST_SEND_TRAILING  = 36   # field [9]
_OFFSET_GHOST_RECV_LEADING   = 40   # field [10]
_OFFSET_GHOST_RECV_TRAILING  = 44   # field [11]


@dataclass
class _TransportSegment:
    """One vkCmdCopyBuffer region for ghost transport.

    For READBACK direction: src = device buffer, dst = sender_staging
    For UPLOAD direction:   src = recv_staging, dst = device buffer
    Both sides use the same staging_offset for a given segment index.
    """
    buffer_name: str        # 'position_voxel_id', 'inside_particle_count', 'global_status', etc.
    device_offset: int      # byte offset in the device buffer
    staging_offset: int     # byte offset in the staging buffer
    size: int               # byte count


# ============================================================================
# Internal helpers: Buffer + BufferSpec
# ============================================================================

@dataclass
class _Buffer:
    handle: object
    memory: object
    size: int
    # Only populated for HOST_VISIBLE staging buffers (persistent map).
    mapped: Optional[object] = None
    mapped_view: Optional[np.ndarray] = None    # numpy uint8 view


@dataclass
class _BufferSpec:
    name: str
    set_index: int
    binding: int
    size: int
    usage: int


# ============================================================================
# SphSimulatorV2
# ============================================================================

class SphSimulatorV2:
    """Per-GPU SPH simulator using timeline semaphore + 3-submit-per-frame pattern.

    Construction (Phase 2 scope): allocates buffers, builds descriptors,
    builds pipelines, creates timeline semaphore. Does NOT yet record cmd
    buffers, run bootstrap, or submit anything — those are Phase 3.

    Lifecycle:
        ctx = VulkanContextV2.create(device_index=...)
        case = load_case_v2("cases/lid_driven_cavity_2d/case.yaml")
        with SphSimulatorV2(ctx, case) as sim:
            sim.bootstrap()                  # single-GPU path
            sim.prepare_step_cmd_buffers()
            for n in range(max_steps):
                sim.submit_phase_a(n); sim.submit_phase_b(n); sim.submit_phase_c(n)
                sim.wait_frame_done(n)
    """

    # ========================================================================
    # Construction / destruction
    # ========================================================================

    def __init__(self, ctx: VulkanContextV2, case: CaseV2) -> None:
        self.ctx = ctx
        self.case = case

        # Keep-alive bag for cffi cdata referenced by VkSpecializationInfo.
        # Python GC would otherwise free the cdata before pipeline creation.
        # Must be initialized BEFORE pipeline build (which calls _make_spec_info).
        self._spec_keepalive: list = []

        self._check_workgroup_limit()

        # Buffer allocation
        self._buffer_specs = self._build_buffer_specs()
        self.buffers = self._allocate_buffers()
        self.scratch_buffers = self._allocate_scratch_buffers()
        self.staging_buffers = self._allocate_staging_buffers()

        # Descriptors
        self.descriptor_layouts = self._build_descriptor_layouts()
        self.descriptor_pool = self._create_descriptor_pool()
        self.descriptor_sets = self._allocate_descriptor_sets()
        self._wire_descriptor_sets()

        # Pipelines
        self.pipeline_layout = self._build_pipeline_layout()
        self.shader_modules = self._load_shader_modules()
        self.pipelines = self._build_compute_pipelines()

        # Defrag has its own 5-set pipeline layout (set 4 = scratch SoA dst)
        # and its own descriptor pool + set 4 descriptor.
        self.defrag_set4_layout = self._build_defrag_set4_layout()
        self.defrag_descriptor_pool, self.defrag_set4 = self._allocate_defrag_set4()
        self._wire_defrag_set4()
        self.defrag_pipeline_layout = self._build_defrag_pipeline_layout()
        self.pipelines["defrag"] = self._build_defrag_pipeline()

        # Timeline semaphore (Phase 3 uses it; creation is here)
        self.timeline = self._create_timeline_semaphore()

        # cmd buffer slots reserved for Phase 3
        self.phase_a_cmd: Any = None
        self.phase_b_cmd: Any = None
        self.phase_c_cmd: Any = None
        self.defrag_cmd: Any = None
        # Single-GPU baseline path: one combined cmd buffer replacing the
        # 3-submit phase A/B/C pattern. Used only when this sim has no peer
        # (see prepare_step_single_cmd_buffer()). Cannot coexist with the
        # dual-GPU path on the same sim — the two recordings would clash on
        # SIMULTANEOUS_USE replay scheduling.
        self.step_single_cmd: Any = None
        # P3.C validation flag: when True, _record_step_single_cmd uses the
        # split pipeline variants (correction_interior + _boundary, density_
        # deep_interior + _boundary, force_deep_interior + _boundary) in
        # place of the *_all variants. Single-GPU mode's empty boundary band
        # makes the two paths bit-equivalent — any divergence is a shader
        # bug. Set BEFORE prepare_step_single_cmd_buffer() to take effect.
        self.step_single_use_split: bool = False

        # Optional GPU-timestamp collector. Attached by the benchmark runner
        # BEFORE prepare_step_cmd_buffers() (and BEFORE the first defrag) so
        # that ticks get baked into the pre-recorded SIMULTANEOUS_USE cmds.
        # When None, _bench_tick / _bench_reset_step / _bench_reset_defrag
        # are pure no-ops; the production runner pays zero per-frame cost.
        self.bench: Any = None

        self._destroyed = False
        print(f"[SimV2] init complete on {ctx.device_name} "
              f"(own={case.capacities.own_pool_size}, "
              f"ghost L={case.capacities.leading_ghost_pool_size} "
              f"T={case.capacities.trailing_ghost_pool_size})")

    def destroy(self) -> None:
        if self._destroyed:
            return
        device = self.ctx.device
        vkDeviceWaitIdle(device)

        # cmd buffers (only if Phase 3 recorded them)
        cmd_pool = self.ctx.command_pool
        for cmd in (self.phase_a_cmd, self.phase_b_cmd,
                    self.phase_c_cmd, self.defrag_cmd):
            if cmd is not None:
                vkFreeCommandBuffers(device, cmd_pool, 1, [cmd])

        if self.timeline is not None:
            vkDestroySemaphore(device, self.timeline, None)
            self.timeline = None

        for pipeline in self.pipelines.values():
            vkDestroyPipeline(device, pipeline, None)
        self.pipelines = {}

        for module in self.shader_modules.values():
            vkDestroyShaderModule(device, module, None)
        self.shader_modules = {}

        if self.defrag_pipeline_layout is not None:
            vkDestroyPipelineLayout(device, self.defrag_pipeline_layout, None)
            self.defrag_pipeline_layout = None
        if self.pipeline_layout is not None:
            vkDestroyPipelineLayout(device, self.pipeline_layout, None)
            self.pipeline_layout = None

        if self.defrag_descriptor_pool is not None:
            vkDestroyDescriptorPool(device, self.defrag_descriptor_pool, None)
            self.defrag_descriptor_pool = None
        if self.descriptor_pool is not None:
            vkDestroyDescriptorPool(device, self.descriptor_pool, None)
            self.descriptor_pool = None

        if self.defrag_set4_layout is not None:
            vkDestroyDescriptorSetLayout(device, self.defrag_set4_layout, None)
            self.defrag_set4_layout = None
        for layout in self.descriptor_layouts:
            vkDestroyDescriptorSetLayout(device, layout, None)
        self.descriptor_layouts = []

        # Unmap + destroy staging
        for buf in self.staging_buffers.values():
            if buf.mapped is not None:
                vkUnmapMemory(device, buf.memory)
            vkDestroyBuffer(device, buf.handle, None)
            vkFreeMemory(device, buf.memory, None)
        self.staging_buffers = {}

        for buf in self.scratch_buffers.values():
            vkDestroyBuffer(device, buf.handle, None)
            vkFreeMemory(device, buf.memory, None)
        self.scratch_buffers = {}

        for buf in self.buffers.values():
            vkDestroyBuffer(device, buf.handle, None)
            vkFreeMemory(device, buf.memory, None)
        self.buffers = {}

        self._destroyed = True

    def __enter__(self) -> "SphSimulatorV2":
        return self

    def __exit__(self, *_: Any) -> None:
        self.destroy()

    # ========================================================================
    # Timeline value source (public; workers + instrumentation read these)
    # ========================================================================

    def value_phase_a_done(self, frame_n: int) -> int:
        return 3 * frame_n + 1

    def value_cpu_sync_done(self, frame_n: int) -> int:
        return 3 * frame_n + 2

    def value_frame_done(self, frame_n: int) -> int:
        return 3 * frame_n + 3

    def current_timeline_value(self) -> int:
        return vkGetSemaphoreCounterValue(self.ctx.device, self.timeline)

    # ========================================================================
    # High-level frame API
    # bootstrap() implemented in Section 8; submit_phase_* + wait_* in Section 10
    # ========================================================================

    def submit_defrag_and_wait(self) -> None:
        if self.defrag_cmd is None:
            self.defrag_cmd = self._record_defrag_cmd()
        self.ctx.submit_and_wait(self.defrag_cmd)

    # ========================================================================
    # Worker-facing accessors
    # ========================================================================

    def sender_staging_view(self, direction: str):
        """numpy.uint8 view over sender_staging_<direction>. Worker reads from
        this view via slice copy (read side of the worker memcpy)."""
        return self.staging_buffers[f"sender_staging_{direction}"].mapped_view

    def receiver_staging_view(self, direction: str):
        """numpy.uint8 view over receiver_staging_<direction>. Worker writes
        to this view via slice assignment (write side of the worker memcpy)."""
        return self.staging_buffers[f"receiver_staging_{direction}"].mapped_view

    def timeline_semaphore(self):
        return self.timeline

    def device(self):
        return self.ctx.device

    # ========================================================================
    # Readback (Phase 3 — implemented in Section 9 below)
    # ========================================================================

    def readback_positions(self):
        raise NotImplementedError("Phase 3 — TODO")

    def get_render_buffers(self) -> dict:
        """Buffer handles the renderer needs to bind as descriptor inputs.

        Vert shader reads:
          - position_voxel_id  (.xyz = position, .w = voxel_id_as_float)
          - velocity_mass      (.xyz = v_{n+1/2})
          - density_pressure   (.x = ρ, .y = P)
        Used by viewer pipeline; sim itself never reads these as descriptor
        bindings — they live in set 0 for the compute pipeline.
        """
        return {
            "position_voxel_id": self.buffers["position_voxel_id"].handle,
            "velocity_mass":     self.buffers["velocity_mass"].handle,
            "density_pressure":  self.buffers["density_pressure"].handle,
            "global_status":     self.buffers["global_status"].handle,
        }

    # ========================================================================
    # Section 1: Validation helpers
    # ========================================================================

    def _check_workgroup_limit(self) -> None:
        props = vkGetPhysicalDeviceProperties(self.ctx.physical_device)
        max_x = props.limits.maxComputeWorkGroupSize[0]
        wg = self.case.capacities.workgroup_size
        if wg > max_x:
            raise RuntimeError(
                f"WORKGROUP_SIZE {wg} > device limit {max_x} on {self.ctx.device_name}")

    # ========================================================================
    # Section 2: Buffer specs + allocation
    # ========================================================================

    def _build_buffer_specs(self) -> list[_BufferSpec]:
        case = self.case
        pool_capacity = case.capacities.total_pool_capacity()
        voxel_capacity = 1 + case.grid.total_voxel_count()
        cap_inside = case.capacities.max_particles_per_voxel
        cap_incoming = case.capacities.max_incoming_per_voxel
        n_materials = max(len(case.materials), 1)

        BSU = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        TRANSFER = (VK_BUFFER_USAGE_TRANSFER_DST_BIT
                    | VK_BUFFER_USAGE_TRANSFER_SRC_BIT)
        VERT = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT

        return [
            # Set 0: particle SoA
            _BufferSpec("position_voxel_id",            0, 0, 16 * pool_capacity, BSU | TRANSFER | VERT),
            _BufferSpec("density_pressure",             0, 1,  8 * pool_capacity, BSU | TRANSFER | VERT),
            _BufferSpec("density_pressure_scratch",     0, 2,  8 * pool_capacity, BSU | TRANSFER | VERT),
            _BufferSpec("velocity_mass",                0, 3, 16 * pool_capacity, BSU | TRANSFER | VERT),
            _BufferSpec("acceleration",                 0, 4, 16 * pool_capacity, BSU | TRANSFER),
            _BufferSpec("shift",                        0, 5, 16 * pool_capacity, BSU | TRANSFER),
            _BufferSpec("material",                     0, 6,  4 * pool_capacity, BSU | TRANSFER),
            _BufferSpec("correction_inverse",           0, 7, 32 * pool_capacity, BSU | TRANSFER),
            _BufferSpec("density_gradient_kernel_sum",  0, 8, 16 * pool_capacity, BSU | TRANSFER),
            _BufferSpec("extension_fields",             0, 9, 16 * pool_capacity, BSU | TRANSFER),

            # Set 1: voxel cells
            _BufferSpec("inside_particle_count",        1, 0,  4 * voxel_capacity,                BSU | TRANSFER),
            _BufferSpec("incoming_particle_count",      1, 1,  4 * voxel_capacity,                BSU | TRANSFER),
            _BufferSpec("inside_particle_index",        1, 2,  4 * voxel_capacity * cap_inside,   BSU | TRANSFER),
            _BufferSpec("incoming_particle_index",      1, 3,  4 * voxel_capacity * cap_incoming, BSU | TRANSFER),
            _BufferSpec("voxel_base_offset",            1, 4,  4 * voxel_capacity,                BSU | TRANSFER),

            # Set 3: global / transport / materials
            _BufferSpec("global_status",                3, 0,  64,                      BSU | TRANSFER),
            _BufferSpec("overflow_log",                 3, 1,  64,                      BSU | TRANSFER),
            _BufferSpec("inlet_template",               3, 2,  32,                      BSU | TRANSFER),
            _BufferSpec("dispatch_indirect",            3, 3,  16,                      BSU | TRANSFER | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT),
            _BufferSpec("ghost_out_packet",             3, 4,  16,                      BSU | TRANSFER),
            _BufferSpec("ghost_in_staging",             3, 5,  16,                      BSU | TRANSFER),
            _BufferSpec("diagnostic",                   3, 6,  16,                      BSU | TRANSFER),
            _BufferSpec("material_parameters",          3, 7,  48 * n_materials,        BSU | TRANSFER),
            _BufferSpec("defrag_scratch_counter",       3, 8,   4,                      BSU | TRANSFER),
        ]

    def _allocate_buffer(
        self,
        size: int,
        usage: int,
        required_properties: int,
        preferred_properties: int = 0,
    ) -> _Buffer:
        if size == 0:
            raise ValueError("buffer size must be > 0")
        bci = VkBufferCreateInfo(
            size=size, usage=usage, sharingMode=VK_SHARING_MODE_EXCLUSIVE)
        handle = vkCreateBuffer(self.ctx.device, bci, None)
        reqs = vkGetBufferMemoryRequirements(self.ctx.device, handle)
        type_index = self.ctx.find_memory_type(
            reqs.memoryTypeBits, required_properties, preferred_properties)
        alloc_info = VkMemoryAllocateInfo(
            allocationSize=reqs.size, memoryTypeIndex=type_index)
        memory = vkAllocateMemory(self.ctx.device, alloc_info, None)
        vkBindBufferMemory(self.ctx.device, handle, memory, 0)
        return _Buffer(handle=handle, memory=memory, size=size)

    def _allocate_buffers(self) -> dict[str, _Buffer]:
        buffers: dict[str, _Buffer] = {}
        total = 0
        for spec in self._buffer_specs:
            buffers[spec.name] = self._allocate_buffer(
                spec.size, spec.usage,
                required_properties=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            )
            total += spec.size
        print(f"[SimV2] device-local buffers: {len(buffers)}, "
              f"{total / (1024 * 1024):.2f} MB")
        return buffers

    def _allocate_scratch_buffers(self) -> dict[str, _Buffer]:
        """Per-set-0 SoA, allocate a scratch twin for defrag.comp's writes
        (copied back via vkCmdCopyBuffer in defrag cmd)."""
        scratch: dict[str, _Buffer] = {}
        total = 0
        for spec in self._buffer_specs:
            if spec.set_index != 0:
                continue
            if spec.binding not in DEFRAG_SET0_BINDINGS:
                continue
            scratch[spec.name] = self._allocate_buffer(
                spec.size,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                required_properties=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            )
            total += spec.size
        print(f"[SimV2] defrag scratch buffers: {len(scratch)}, "
              f"{total / (1024 * 1024):.2f} MB")
        return scratch

    def _compute_transport_segments(self, direction: str) -> tuple[list[_TransportSegment], int]:
        """Build the V1-equivalent 12-segment list for one direction. Returns
        (segments, total_staging_bytes). direction ∈ {'leading','trailing'}."""
        case = self.case
        cap_inside = case.capacities.max_particles_per_voxel

        if direction == "leading":
            pool_size = case.capacities.leading_ghost_pool_size
            ghost_voxel_count = case.ghost_grid.leading_ghost_voxel_count
            pid_first = 1                           # leading pid range starts at 1
            vid_first = 1                           # leading vid range starts at 1
            send_count_offset = _OFFSET_GHOST_SEND_LEADING
            recv_count_offset = _OFFSET_GHOST_RECV_LEADING
        elif direction == "trailing":
            pool_size = case.capacities.trailing_ghost_pool_size
            ghost_voxel_count = case.ghost_grid.trailing_ghost_voxel_count
            # Trailing pid range = [leading + own + 1, leading + own + trailing]
            pid_first = (case.capacities.leading_ghost_pool_size
                         + case.capacities.own_pool_size + 1)
            # Trailing vid range = [total - trailing + 1, total]
            vid_first = case.grid.total_voxel_count() - ghost_voxel_count + 1
            send_count_offset = _OFFSET_GHOST_SEND_TRAILING
            recv_count_offset = _OFFSET_GHOST_RECV_TRAILING
        else:
            raise ValueError(f"bad direction: {direction}")

        segments: list[_TransportSegment] = []
        staging_offset = 0
        if pool_size == 0 or ghost_voxel_count == 0:
            return segments, 0

        # 1-9. Nine SoA fields × ghost-pid range
        for binding in TRANSPORT_SET0_BINDINGS:
            name = _SET0_BINDING_TO_NAME[binding]
            stride = _SET0_BYTE_STRIDES[name]
            size = stride * pool_size
            device_offset = stride * pid_first
            segments.append(_TransportSegment(name, device_offset, staging_offset, size))
            staging_offset += size

        # 10. set 1 inside_particle_count × ghost-vid range
        size = 4 * ghost_voxel_count
        segments.append(_TransportSegment(
            "inside_particle_count", 4 * vid_first, staging_offset, size))
        staging_offset += size

        # 11. set 1 inside_particle_index × ghost-vid range × MAX_PARTICLES_PER_VOXEL
        size = 4 * ghost_voxel_count * cap_inside
        segments.append(_TransportSegment(
            "inside_particle_index",
            4 * vid_first * cap_inside, staging_offset, size))
        staging_offset += size

        # 12. set 3 ghost_send_*_count (sender side: read device→staging at
        #     send_count_offset; receiver side: write staging→device at
        #     recv_count_offset). Buffer is global_status; both sides use the
        #     same staging slot but different device offsets. We store the
        #     SENDER's device offset here; recv side overrides at cmd record time.
        segments.append(_TransportSegment(
            "global_status", send_count_offset, staging_offset, 4))
        staging_offset += 4

        # Store the receiver-side count offset for later use during upload cmd record.
        self._recv_count_offsets = getattr(self, "_recv_count_offsets", {})
        self._recv_count_offsets[direction] = recv_count_offset
        return segments, staging_offset

    def _allocate_staging_buffers(self) -> dict[str, _Buffer]:
        """Per-direction host-visible stagings (sender CACHED, receiver COHERENT)
        — see docs/sph_v2_design.md §14.5. Persistent-mapped at construction.

        Diagnostic print below shows the actual memory type the driver
        picked for sender and receiver, including whether DEVICE_LOCAL
        was granted (ReBAR-style VRAM exposed to CPU). Useful for
        future memory-type experiments.

        Attempted optimization 2026-05-21 (experiment B3): preferring
        DEVICE_LOCAL for sender and/or receiver. Result: receiver-as-
        ReBAR cut NV's install upload DMA 556→23 µs (24× faster) BUT
        broke worker memcpy — CPU write to ReBAR via numpy[:] runs at
        ~2.3 GB/s (vs theoretical 12 GB/s WC), so worker time grew
        220→1384 µs and stopped fitting inside correction_interior →
        sync hiding collapsed → net fps 228→175 (-23%). Sender-as-
        ReBAR was worse (CPU read of ReBAR is uncached PCIe BAR at
        ~10 MB/s, fps 228→2.7). The worker-bridge architecture
        requires sender_staging in cached host RAM, full stop.
        Real fix requires moving the worker memcpy off the CPU
        (Path A: cross-queue device→device transfer)."""
        case = self.case
        sender_required = (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                           | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        sender_preferred = VK_MEMORY_PROPERTY_HOST_CACHED_BIT
        receiver_required = (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                             | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        receiver_preferred = 0
        usage = (VK_BUFFER_USAGE_TRANSFER_DST_BIT
                 | VK_BUFFER_USAGE_TRANSFER_SRC_BIT)

        # Compute segments + total size per direction; cache on self for cmd recording.
        self._transport_segments: dict[str, list[_TransportSegment]] = {}
        self._transport_total_bytes: dict[str, int] = {}
        self._recv_count_offsets = {}
        stagings: dict[str, _Buffer] = {}
        for direction_name, peer_attr in (
            ("leading",  "has_leading_peer"),
            ("trailing", "has_trailing_peer"),
        ):
            if not getattr(case.transport, peer_attr):
                continue
            segments, total = self._compute_transport_segments(direction_name)
            if total == 0:
                continue
            self._transport_segments[direction_name] = segments
            self._transport_total_bytes[direction_name] = total

            sender_buf = self._allocate_buffer(
                total, usage, sender_required, sender_preferred)
            mapped = vkMapMemory(self.ctx.device, sender_buf.memory, 0, total, 0)
            sender_buf.mapped = mapped
            sender_buf.mapped_view = np.frombuffer(mapped, dtype=np.uint8, count=total)
            stagings[f"sender_staging_{direction_name}"] = sender_buf

            recv_buf = self._allocate_buffer(
                total, usage, receiver_required, receiver_preferred)
            mapped_r = vkMapMemory(self.ctx.device, recv_buf.memory, 0, total, 0)
            recv_buf.mapped = mapped_r
            recv_buf.mapped_view = np.frombuffer(mapped_r, dtype=np.uint8, count=total)
            stagings[f"receiver_staging_{direction_name}"] = recv_buf

        total_bytes = sum(b.size for b in stagings.values())
        # Diagnostic: re-query the chosen memory type for one sender and one
        # receiver buffer. Sender and receiver have asymmetric preferred
        # properties (see docstring), so we report both. Same args →
        # find_memory_type returns the same index as the actual allocation.
        if stagings:
            def _flag_str(flags: int) -> str:
                names = []
                for bit, name in (
                    (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,  "DEVICE_LOCAL"),
                    (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,  "HOST_VISIBLE"),
                    (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, "HOST_COHERENT"),
                    (VK_MEMORY_PROPERTY_HOST_CACHED_BIT,   "HOST_CACHED"),
                ):
                    if flags & bit:
                        names.append(name)
                return "|".join(names) if names else "(none)"

            def _probe(buf, required, preferred) -> str:
                reqs = vkGetBufferMemoryRequirements(self.ctx.device, buf.handle)
                idx = self.ctx.find_memory_type(reqs.memoryTypeBits, required, preferred)
                flags = self.ctx._memory_properties.memoryTypes[idx].propertyFlags
                rebar = "[OK ReBAR]" if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) else "[host RAM]"
                return f"type[{idx}]={_flag_str(flags)} {rebar}"

            sender_example = next(b for k, b in stagings.items() if k.startswith("sender_"))
            recv_example   = next(b for k, b in stagings.items() if k.startswith("receiver_"))
            print(f"[SimV2] host staging buffers: {len(stagings)}, "
                  f"{total_bytes / 1024:.1f} KB (persistent-mapped)")
            print(f"  sender   {_probe(sender_example, sender_required, sender_preferred)}")
            print(f"  receiver {_probe(recv_example, receiver_required, receiver_preferred)}")
        else:
            print(f"[SimV2] host staging buffers: 0 (single-GPU mode, no peer)")
        return stagings

    # ========================================================================
    # Section 3: Descriptor sets
    # ========================================================================

    def _build_descriptor_layouts(self) -> list:
        """4 layouts: set 0/1/3 hold buffers; set 2 is empty (V2 merged-buffer
        scheme stores ghost in set 0/1)."""
        layouts = []
        for set_index in range(4):
            specs_in_set = [s for s in self._buffer_specs if s.set_index == set_index]
            bindings = [
                VkDescriptorSetLayoutBinding(
                    binding=spec.binding,
                    descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1,
                    stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
                )
                for spec in specs_in_set
            ]
            ci = VkDescriptorSetLayoutCreateInfo(
                bindingCount=len(bindings),
                pBindings=bindings if bindings else None,
            )
            layouts.append(vkCreateDescriptorSetLayout(self.ctx.device, ci, None))
        return layouts

    def _create_descriptor_pool(self):
        n_buffers = len(self._buffer_specs)
        pool_size = VkDescriptorPoolSize(
            type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=n_buffers)
        ci = VkDescriptorPoolCreateInfo(
            maxSets=4, poolSizeCount=1, pPoolSizes=[pool_size])
        return vkCreateDescriptorPool(self.ctx.device, ci, None)

    def _allocate_descriptor_sets(self) -> list:
        alloc = VkDescriptorSetAllocateInfo(
            descriptorPool=self.descriptor_pool,
            descriptorSetCount=4,
            pSetLayouts=self.descriptor_layouts,
        )
        return vkAllocateDescriptorSets(self.ctx.device, alloc)

    def _wire_descriptor_sets(self) -> None:
        writes = []
        for spec in self._buffer_specs:
            buf_info = VkDescriptorBufferInfo(
                buffer=self.buffers[spec.name].handle,
                offset=0,
                range=spec.size,
            )
            writes.append(VkWriteDescriptorSet(
                dstSet=self.descriptor_sets[spec.set_index],
                dstBinding=spec.binding,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[buf_info],
            ))
        vkUpdateDescriptorSets(self.ctx.device, len(writes), writes, 0, None)

    # ========================================================================
    # Section 4: Pipelines
    # ========================================================================

    def _build_pipeline_layout(self):
        ci = VkPipelineLayoutCreateInfo(
            setLayoutCount=4,
            pSetLayouts=self.descriptor_layouts,
            pushConstantRangeCount=0,
        )
        return vkCreatePipelineLayout(self.ctx.device, ci, None)

    def _load_shader_modules(self) -> dict[str, object]:
        shader_dir = (pathlib.Path(__file__).resolve().parents[1]
                      / "shaders" / "spv")
        modules: dict[str, object] = {}
        for shader_name in (
            "bootstrap_half_kick", "initialize_voxelization",
            "predict", "update_voxel", "ghost_send", "install_migrations",
            "correction", "density", "force", "defrag",
        ):
            spv_path = shader_dir / f"{shader_name}.comp.spv"
            if not spv_path.exists():
                raise FileNotFoundError(
                    f"shader not compiled: {spv_path}. "
                    f"Run experiment/v2/compile_shaders_v2.py first.")
            code = spv_path.read_bytes()
            ci = VkShaderModuleCreateInfo(codeSize=len(code), pCode=code)
            modules[shader_name] = vkCreateShaderModule(self.ctx.device, ci, None)
        return modules

    # ----- Spec const blob helpers -----

    def _pack_spec(self, entries: list[tuple[int, str, Any]]
                   ) -> tuple[bytes, list[VkSpecializationMapEntry]]:
        """Pack a list of (constant_id, fmt, value) into a contiguous data blob
        + VkSpecializationMapEntry list. fmt is struct-style 'f'/'I'/'i'/'B'.
        All fields are 4 B in our use."""
        blob = bytearray()
        map_entries: list[VkSpecializationMapEntry] = []
        for const_id, fmt, value in entries:
            if fmt == 'B':
                packed = struct.pack('<I', 1 if value else 0)
            else:
                packed = struct.pack('<' + fmt, value)
            assert len(packed) == 4
            map_entries.append(VkSpecializationMapEntry(
                constantID=const_id, offset=len(blob), size=4))
            blob.extend(packed)
        return bytes(blob), map_entries

    def _make_spec_info(self, entries: list[tuple[int, str, Any]]
                        ) -> Optional[object]:
        if not entries:
            return None
        blob, map_entries = self._pack_spec(entries)
        # pData is const void *; python-vulkan can't auto-convert bytes,
        # need a typed cdata pointer. Keep blob + cdata + map_entries alive
        # until vkCreateComputePipelines returns.
        cdata = ffi.new("uint8_t[]", blob)
        info = VkSpecializationInfo(
            mapEntryCount=len(map_entries),
            pMapEntries=map_entries,
            dataSize=len(blob),
            pData=cdata,
        )
        self._spec_keepalive.append({
            "blob": blob, "cdata": cdata, "entries": map_entries, "info": info
        })
        return info

    def _global_entries(self) -> list[tuple[int, str, Any]]:
        p = self.case.physics
        n = self.case.numerics
        cap = self.case.capacities
        g = self.case.grid
        gh = self.case.ghost_grid
        return [
            (0,  'f', p.smoothing_length),
            (1,  'f', p.speed_of_sound),
            (2,  'f', p.delta_coefficient),
            (4,  'f', p.power_parameter),
            (5,  'f', p.cfl_number),
            (6,  'f', p.timestep),
            (7,  'f', g.origin_x),
            (8,  'f', g.origin_y),
            (9,  'f', g.origin_z),
            (10, 'B', 0),                       # STRICT_BIT_EXACT — 目前版本不再需要 bit-exact
            (11, 'I', g.grid_dimension_x),
            (12, 'I', g.grid_dimension_y),
            (13, 'I', g.grid_dimension_z),
            (14, 'f', n.regularization_xi),
            (15, 'f', n.regularization_determinant_threshold),
            (16, 'f', n.regularization_max_frobenius_norm),
            (17, 'f', p.gravity[0]),
            (18, 'f', p.gravity[1]),
            (19, 'f', p.gravity[2]),
            (20, 'I', g.voxel_order),
            (30, 'I', p.dimension),
            (31, 'I', p.neighbor_z_range),
            (32, 'f', p.kernel_coefficient),
            (33, 'f', p.kernel_gradient_coefficient),
            (40, 'f', n.eps_h_squared),
            (41, 'f', n.pst_main_shift_coefficient),
            (42, 'f', n.pst_anti_shift_coefficient),
            (43, 'B', int(n.use_kcg_correction)),
            (44, 'B', int(n.use_density_diffusion)),
            (45, 'B', int(n.use_pst)),
            (46, 'B', int(n.use_prefix_sum_defrag)),
            (50, 'I', cap.max_particles_per_voxel),
            (51, 'I', cap.workgroup_size),
            (52, 'I', cap.max_incoming_per_voxel),
            (53, 'I', cap.own_pool_size),
            (54, 'I', cap.leading_ghost_pool_size),
            (55, 'I', cap.trailing_ghost_pool_size),
            (80, 'I', gh.leading_ghost_voxel_count),
            (81, 'I', gh.trailing_ghost_voxel_count),
            # NEIGHBOR_X_RANGE (id=82) is NOT global anymore — Path A+ needs
            # different widths per kernel (correction=2, density=3, force=4
            # for the cascading interior/boundary split). Each split-kernel
            # pipeline appends its own (82, 'I', width) via the helpers
            # below. Non-split kernels (predict / update_voxel / ghost_send /
            # install_migrations / defrag) don't use in_boundary_band at all,
            # so they rely on the GLSL default (NEIGHBOR_X_RANGE = 0u).
        ]

    # ----- Per-pipeline mode entries (kernel-specific spec const overrides) ---

    def _correction_mode_entries(self, mode: int) -> list[tuple[int, str, Any]]:
        """CORRECTION_MODE (id=47) + NEIGHBOR_X_RANGE (id=82).
        Boundary band = 2 voxels (column 0 reaches ghost; column 1 reaches
        column 0 where migrants land after install_migration)."""
        return [(47, 'I', mode), (82, 'I', 2)]

    def _density_mode_entries(self, mode: int) -> list[tuple[int, str, Any]]:
        """DENSITY_MODE (id=48) + NEIGHBOR_X_RANGE (id=82).
        Boundary band = 3 voxels (= correction's 2 + 1 for neighbor reach
        into stale-correction). Used by Path A+ density split."""
        return [(48, 'I', mode), (82, 'I', 3)]

    def _force_mode_entries(self, mode: int) -> list[tuple[int, str, Any]]:
        """FORCE_MODE (id=49) + NEIGHBOR_X_RANGE (id=82).
        Boundary band = 4 voxels (= density's 3 + 1 for neighbor reach into
        stale-density). Used by Path A+ force split."""
        return [(49, 'I', mode), (82, 'I', 4)]

    def _ghost_direction_entries(
        self, direction: int
    ) -> list[tuple[int, str, Any]]:
        """Per-direction spec consts (ids 90-94) for ghost_send / install_migrations."""
        spec = (self.case.transport.leading if direction == 0
                else self.case.transport.trailing)
        # If direction has no peer, we still need to compile the pipeline
        # (descriptor wiring) but the spec consts can be defaults — the
        # pipeline will never actually be dispatched.
        if spec is None:
            return [
                (90, 'I', direction),
                (91, 'I', 0),
                (92, 'I', 0),
                (93, 'i', 0),
                (94, 'i', 0),
            ]
        return [
            (90, 'I', spec.direction),
            (91, 'I', spec.boundary_voxel_x_local),
            (92, 'I', spec.ghost_voxel_x_local),
            (93, 'i', spec.ghost_pid_offset_to_receiver),
            (94, 'i', spec.ghost_voxel_id_offset_to_receiver),
        ]

    def _build_compute_pipelines(self) -> dict[str, object]:
        """Build the compute pipelines:

            1 × initialize_voxelization
            1 × bootstrap_half_kick
            1 × predict
            1 × update_voxel
            2 × ghost_send (leading, trailing)
            2 × install_migrations (leading, trailing)
            3 × correction (ALL, INTERIOR, BOUNDARY)             ← V2 split
            3 × density    (ALL, DEEP_INTERIOR, BOUNDARY)        ← Path A+ split
            3 × force      (ALL, DEEP_INTERIOR, BOUNDARY)        ← Path A+ split
            1 × defrag                                            (built later)

        Total = 16 (+ defrag built in Section 11 = 17).

        Naming convention: `<kernel>_all` for the V1-equivalent single-
        pipeline variant (used by bootstrap + single-GPU step + dual Phase
        C while Path A+ wiring is pending); `<kernel>_interior` and
        `<kernel>_boundary` for correction's 2-voxel split; density and
        force use `_deep_interior` to mark the larger boundary band
        (3 and 4 voxels respectively, accounting for cascading neighbor
        reach into stale-output regions).

        ghost_send + install_migrations are always built for BOTH directions
        even if this GPU has no peer on that side; phase A/C cmd recording
        skips dispatch on the unused direction (cf. V1)."""
        pipelines: dict[str, object] = {}

        # Pipelines that only need global spec consts (and the shared 4-set
        # pipeline layout). These kernels don't use in_boundary_band, so they
        # rely on the GLSL default NEIGHBOR_X_RANGE = 0u. defrag is excluded
        # here — it uses a 5-set layout (set 4 = destination scratch SoA) and
        # is built in Section 11 alongside its cmd buffer.
        for key in ("initialize_voxelization", "bootstrap_half_kick",
                    "predict", "update_voxel"):
            pipelines[key] = self._create_pipeline(
                shader=self.shader_modules[key],
                entries=self._global_entries(),
            )

        # ghost_send per direction
        for direction, dir_name in ((0, "leading"), (1, "trailing")):
            pipelines[f"ghost_send_{dir_name}"] = self._create_pipeline(
                shader=self.shader_modules["ghost_send"],
                entries=self._global_entries() + self._ghost_direction_entries(direction),
            )

        # install_migrations per direction
        for direction, dir_name in ((0, "leading"), (1, "trailing")):
            pipelines[f"install_migrations_{dir_name}"] = self._create_pipeline(
                shader=self.shader_modules["install_migrations"],
                entries=self._global_entries() + self._ghost_direction_entries(direction),
            )

        # correction × 3 modes (V2 #1 — boundary band = 2 voxels)
        for mode, mode_name in ((0, "all"), (1, "interior"), (2, "boundary")):
            pipelines[f"correction_{mode_name}"] = self._create_pipeline(
                shader=self.shader_modules["correction"],
                entries=self._global_entries() + self._correction_mode_entries(mode),
            )

        # density × 3 modes (Path A+ — boundary band = 3 voxels)
        for mode, mode_name in ((0, "all"), (1, "deep_interior"), (2, "boundary")):
            pipelines[f"density_{mode_name}"] = self._create_pipeline(
                shader=self.shader_modules["density"],
                entries=self._global_entries() + self._density_mode_entries(mode),
            )

        # force × 3 modes (Path A+ — boundary band = 4 voxels)
        for mode, mode_name in ((0, "all"), (1, "deep_interior"), (2, "boundary")):
            pipelines[f"force_{mode_name}"] = self._create_pipeline(
                shader=self.shader_modules["force"],
                entries=self._global_entries() + self._force_mode_entries(mode),
            )

        print(f"[SimV2] compute pipelines: {len(pipelines)}")
        return pipelines

    def _create_pipeline(
        self,
        shader,
        entries: list[tuple[int, str, Any]],
    ):
        spec_info = self._make_spec_info(entries)
        stage = VkPipelineShaderStageCreateInfo(
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            module=shader,
            pName="main",
            pSpecializationInfo=spec_info,
        )
        ci = VkComputePipelineCreateInfo(
            stage=stage,
            layout=self.pipeline_layout,
        )
        return vkCreateComputePipelines(
            self.ctx.device, VK_NULL_HANDLE, 1, [ci], None)[0]

    # ========================================================================
    # Section 5: Timeline semaphore
    # ========================================================================

    def _create_timeline_semaphore(self):
        type_info = VkSemaphoreTypeCreateInfo(
            sType=VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
            semaphoreType=VK_SEMAPHORE_TYPE_TIMELINE,
            initialValue=0,
        )
        ci = VkSemaphoreCreateInfo(pNext=type_info)
        return vkCreateSemaphore(self.ctx.device, ci, None)

    # ========================================================================
    # Section 6: Cmd buffer helpers (Phase 3)
    # ========================================================================

    def _allocate_oneshot_cmd(self):
        info = VkCommandBufferAllocateInfo(
            commandPool=self.ctx.command_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        return vkAllocateCommandBuffers(self.ctx.device, info)[0]

    def _per_own_particle_dispatch_count(self) -> int:
        wg = self.case.capacities.workgroup_size
        own = self.case.capacities.own_pool_size
        return (own + wg - 1) // wg

    def _per_extended_voxel_dispatch_count(self) -> int:
        wg = self.case.capacities.workgroup_size
        v = self.case.grid.total_voxel_count()
        return (v + wg - 1) // wg

    def _per_yz_face_dispatch_count(self) -> int:
        """ghost_send dispatches one thread per (y,z) face slot = NY*NZ threads."""
        wg = self.case.capacities.workgroup_size
        face = self.case.grid.grid_dimension_y * self.case.grid.grid_dimension_z
        return (face + wg - 1) // wg

    def _per_ghost_pid_dispatch_count(self, direction: str) -> int:
        wg = self.case.capacities.workgroup_size
        if direction == "leading":
            pool = self.case.capacities.leading_ghost_pool_size
        elif direction == "trailing":
            pool = self.case.capacities.trailing_ghost_pool_size
        else:
            raise ValueError(direction)
        return (pool + wg - 1) // wg if pool > 0 else 0

    # ----- bench timestamp helpers (no-op when self.bench is None) ----------

    def _bench_tick(self, cmd, label: str) -> None:
        """Insert vkCmdWriteTimestamp into ``cmd`` if a BenchTimer is attached."""
        if self.bench is not None:
            self.bench.tick(cmd, label)

    def _bench_reset_step(self, cmd, start_label: str) -> None:
        """First action of phase_a_cmd: reset step query slots + first tick."""
        if self.bench is not None:
            self.bench.record_step_reset_and_start(cmd, start_label)

    def _bench_reset_defrag(self, cmd, start_label: str) -> None:
        """First action of defrag_cmd: reset defrag slots + defrag start tick."""
        if self.bench is not None:
            self.bench.record_defrag_reset_and_start(cmd, start_label)

    # ----- sync2 barriers ---------------------------------------------------

    def _record_compute_barrier(self, cmd) -> None:
        """Global compute→compute memory barrier (sync2)."""
        mb = VkMemoryBarrier2(
            sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            srcStageMask=VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            srcAccessMask=(VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                           | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT),
            dstStageMask=VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            dstAccessMask=(VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                           | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT),
        )
        info = VkDependencyInfo(
            sType=VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            memoryBarrierCount=1,
            pMemoryBarriers=[mb],
        )
        vkCmdPipelineBarrier2(cmd, info)

    def _record_transfer_to_compute_barrier(self, cmd) -> None:
        mb = VkMemoryBarrier2(
            sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            srcStageMask=VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            srcAccessMask=VK_ACCESS_2_TRANSFER_WRITE_BIT,
            dstStageMask=VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            dstAccessMask=(VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                           | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT),
        )
        info = VkDependencyInfo(
            sType=VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            memoryBarrierCount=1,
            pMemoryBarriers=[mb],
        )
        vkCmdPipelineBarrier2(cmd, info)

    def _record_compute_to_transfer_barrier(self, cmd) -> None:
        mb = VkMemoryBarrier2(
            sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            srcStageMask=VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            srcAccessMask=VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            dstStageMask=VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            dstAccessMask=VK_ACCESS_2_TRANSFER_READ_BIT,
        )
        info = VkDependencyInfo(
            sType=VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            memoryBarrierCount=1,
            pMemoryBarriers=[mb],
        )
        vkCmdPipelineBarrier2(cmd, info)

    def _record_compute_to_host_barrier(self, cmd) -> None:
        """End of Phase A: GPU finished writing sender_staging; host (worker
        thread) about to read it via mapped pointer. HOST_COHERENT alone is
        insufficient — need an explicit access-scope barrier per spec."""
        mb = VkMemoryBarrier2(
            sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            srcStageMask=VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            srcAccessMask=VK_ACCESS_2_TRANSFER_WRITE_BIT,
            dstStageMask=VK_PIPELINE_STAGE_2_HOST_BIT,
            dstAccessMask=VK_ACCESS_2_HOST_READ_BIT,
        )
        info = VkDependencyInfo(
            sType=VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            memoryBarrierCount=1,
            pMemoryBarriers=[mb],
        )
        vkCmdPipelineBarrier2(cmd, info)

    def _record_reset_ghost_send_count(self, cmd, direction: str) -> None:
        """Zero the ghost_send_<direction>_count field of GlobalStatusBuffer
        before ghost_send.comp's atomicAdd. Without reset, the previous step's
        count corrupts slot allocation."""
        offset = (_OFFSET_GHOST_SEND_LEADING if direction == "leading"
                  else _OFFSET_GHOST_SEND_TRAILING)
        vkCmdFillBuffer(
            cmd, self.buffers["global_status"].handle, offset, 4, 0)

    def _record_readback_for_direction(self, cmd, direction: str) -> None:
        """Scatter device-local ghost bytes → sender_staging_<direction>."""
        staging = self.staging_buffers[f"sender_staging_{direction}"]
        for seg in self._transport_segments[direction]:
            src_buf = self.buffers[seg.buffer_name]
            region = VkBufferCopy(
                srcOffset=seg.device_offset,
                dstOffset=seg.staging_offset,
                size=seg.size,
            )
            vkCmdCopyBuffer(cmd, src_buf.handle, staging.handle, 1, [region])

    def _record_upload_for_direction(self, cmd, direction: str) -> None:
        """Gather receiver_staging_<direction> → device-local ghost bytes.
        For the count segment specifically, dst is global_status[recv_count_offset]
        not [send_count_offset] — sender's count slot becomes receiver's
        ghost_recv_*_count slot."""
        staging = self.staging_buffers[f"receiver_staging_{direction}"]
        recv_count_offset = self._recv_count_offsets[direction]
        for seg in self._transport_segments[direction]:
            dst_buf = self.buffers[seg.buffer_name]
            if seg.buffer_name == "global_status":
                dst_offset = recv_count_offset
            else:
                dst_offset = seg.device_offset
            region = VkBufferCopy(
                srcOffset=seg.staging_offset,
                dstOffset=dst_offset,
                size=seg.size,
            )
            vkCmdCopyBuffer(cmd, staging.handle, dst_buf.handle, 1, [region])

    # ----- Pipeline binding -------------------------------------------------

    def _bind_pipeline_and_sets(self, cmd, pipeline_key: str) -> None:
        vkCmdBindPipeline(
            cmd, VK_PIPELINE_BIND_POINT_COMPUTE, self.pipelines[pipeline_key])
        vkCmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout,
            0, 4, self.descriptor_sets, 0, None)

    # ----- Density scratch copy-back (inside step cmd) ---------------------

    def _record_density_scratch_to_primary_copy(self, cmd) -> None:
        """After density.comp writes density_pressure_scratch, copy that back
        to density_pressure (primary) inside the same submit. Force.comp's
        next dispatch reads primary.

        Copy ONLY the own pid range. The ghost-pid range of primary holds
        ρ values uploaded from the peer GPU this step; density.comp doesn't
        dispatch on ghost pids so scratch's ghost range is stale zero — a
        full-buffer copy would zero out the uploaded ghost density and make
        force.comp read ρ=0 for ghost neighbours (→ NaN pressure)."""
        scratch = self.buffers["density_pressure_scratch"]
        primary = self.buffers["density_pressure"]
        density_stride = 8  # vec2 floats
        own_first = self.own_first_pid()
        own_pool = self.case.capacities.own_pool_size
        own_byte_offset = own_first * density_stride
        own_byte_size = own_pool * density_stride
        # compute→transfer (density wrote scratch)
        self._record_compute_to_transfer_barrier(cmd)
        vkCmdCopyBuffer(cmd, scratch.handle, primary.handle, 1, [
            VkBufferCopy(srcOffset=own_byte_offset,
                         dstOffset=own_byte_offset,
                         size=own_byte_size)
        ])
        # transfer→compute (force will read primary)
        self._record_transfer_to_compute_barrier(cmd)

    # ========================================================================
    # Section 7: Initial data upload (Phase 3)
    # ========================================================================

    def own_first_pid(self) -> int:
        return self.case.capacities.leading_ghost_pool_size + 1

    def own_last_pid(self) -> int:
        return (self.case.capacities.leading_ghost_pool_size
                + self.case.capacities.own_pool_size)

    def _build_initial_data(self) -> dict[str, bytes]:
        """Build CPU-side payloads keyed by buffer name. Caller uploads each
        via _staging_upload + ctx.submit_and_wait. Buffers not in this dict
        get zeroed via _zero_buffer.

        V2 layout: own particles start at own_first_pid (V1.0a merged-buffer
        scheme); ghost-pid slots stay zero until next step's ghost_send fills
        them.
        """
        case = self.case
        pool_capacity = case.capacities.total_pool_capacity()
        own_first = self.own_first_pid()
        n_initial = case.initial.positions.shape[0]
        data: dict[str, bytes] = {}

        # position_voxel_id (vec4: xyz, voxel_id_as_float=0 initially)
        position_voxel_id = np.zeros((pool_capacity, 4), dtype=np.float32)
        position_voxel_id[own_first:own_first + n_initial, 0:3] = case.initial.positions
        data["position_voxel_id"] = position_voxel_id.tobytes()

        # velocity_mass (vec4: vx, vy, vz, mass)
        velocity_mass = np.zeros((pool_capacity, 4), dtype=np.float32)
        velocity_mass[own_first:own_first + n_initial, 0:3] = case.initial.velocities
        for i in range(n_initial):
            group = int(case.initial.material_group[i])
            mat = case.materials[group]
            velocity_mass[own_first + i, 3] = mat.rest_density * mat.volume
        data["velocity_mass"] = velocity_mass.tobytes()

        # density_pressure (vec2: ρ₀, 0)
        density_pressure = np.zeros((pool_capacity, 2), dtype=np.float32)
        for i in range(n_initial):
            group = int(case.initial.material_group[i])
            mat = case.materials[group]
            density_pressure[own_first + i, 0] = mat.rest_density
        data["density_pressure"] = density_pressure.tobytes()

        # material (uint group_id, 0 for empty slots)
        material_arr = np.zeros(pool_capacity, dtype=np.uint32)
        if n_initial > 0:
            material_arr[own_first:own_first + n_initial] = case.initial.material_group
        data["material"] = material_arr.tobytes()

        # material_parameters (48 B per row)
        mp_blob = bytearray()
        for mat in case.materials:
            row = struct.pack(
                "<I f f f f f f f f f I I",
                mat.kind, mat.rest_density, mat.viscosity, mat.eos_constant,
                mat.smoothing_length, mat.radius, mat.volume,
                mat.rotor_angular_velocity,
                mat.viscosity_transfer, mat.viscosity_rotation,
                mat.reserved_material_0, mat.reserved_material_1,
            )
            assert len(row) == 48
            mp_blob.extend(row)
        if not mp_blob:
            mp_blob = b"\x00" * 48
        data["material_parameters"] = bytes(mp_blob)

        return data

    def _staging_upload(self, dest: _Buffer, payload: bytes) -> None:
        if len(payload) > dest.size:
            raise ValueError(
                f"payload {len(payload)} > buffer {dest.size}")
        staging = self._allocate_buffer(
            size=len(payload),
            usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            required_properties=(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                 | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
        )
        try:
            mapped = vkMapMemory(self.ctx.device, staging.memory, 0, len(payload), 0)
            view = np.frombuffer(mapped, dtype=np.uint8, count=len(payload))
            view[:] = np.frombuffer(payload, dtype=np.uint8)
            vkUnmapMemory(self.ctx.device, staging.memory)
            cmd = self._allocate_oneshot_cmd()
            vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
                flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
            vkCmdCopyBuffer(cmd, staging.handle, dest.handle, 1, [
                VkBufferCopy(srcOffset=0, dstOffset=0, size=len(payload))
            ])
            vkEndCommandBuffer(cmd)
            self.ctx.submit_and_wait(cmd)
            vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])
        finally:
            vkDestroyBuffer(self.ctx.device, staging.handle, None)
            vkFreeMemory(self.ctx.device, staging.memory, None)

    def _zero_buffer(self, dest: _Buffer) -> None:
        cmd = self._allocate_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
        vkCmdFillBuffer(cmd, dest.handle, 0, dest.size, 0)
        vkEndCommandBuffer(cmd)
        self.ctx.submit_and_wait(cmd)
        vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])

    def _upload_initial_state(self) -> None:
        initial = self._build_initial_data()
        for name, buf in self.buffers.items():
            if name in initial:
                payload = initial[name]
                if len(payload) < buf.size:
                    payload = payload + b"\x00" * (buf.size - len(payload))
                self._staging_upload(buf, payload)
            else:
                self._zero_buffer(buf)
        for buf in self.scratch_buffers.values():
            self._zero_buffer(buf)
        print(f"[SimV2] uploaded initial state ({len(initial)} payload buffers)")

    # ========================================================================
    # Section 8: Bootstrap (Phase 3)
    # ========================================================================

    def _record_bootstrap_init_cmd(self):
        """Bootstrap stage 1: initialize_voxelization + (if peer) ghost_send +
        readback. Outbox staging ready after this submit. Fence-wait submit.
        For sims with no peer this is just init_voxelization."""
        cmd = self._allocate_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
        self._record_compute_barrier(cmd)

        per_p = self._per_own_particle_dispatch_count()
        per_yz = self._per_yz_face_dispatch_count()

        self._bind_pipeline_and_sets(cmd, "initialize_voxelization")
        vkCmdDispatch(cmd, per_p, 1, 1)

        for direction in ("leading", "trailing"):
            if direction not in self._transport_segments:
                continue
            self._record_compute_barrier(cmd)
            self._record_reset_ghost_send_count(cmd, direction)
            self._record_transfer_to_compute_barrier(cmd)
            self._bind_pipeline_and_sets(cmd, f"ghost_send_{direction}")
            vkCmdDispatch(cmd, per_yz, 1, 1)
            self._record_compute_to_transfer_barrier(cmd)
            self._record_readback_for_direction(cmd, direction)
            self._record_compute_to_host_barrier(cmd)

        vkEndCommandBuffer(cmd)
        return cmd

    def _record_bootstrap_compute_cmd(self):
        """Bootstrap stage 2: (if peer) upload + install_migrations →
        correction(ALL) → density → force → bootstrap_half_kick. Runs after
        host memcpy completes the cross-GPU ghost transport."""
        cmd = self._allocate_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))

        per_p = self._per_own_particle_dispatch_count()

        for direction in ("leading", "trailing"):
            if direction not in self._transport_segments:
                continue
            self._record_upload_for_direction(cmd, direction)
            self._record_transfer_to_compute_barrier(cmd)
            self._bind_pipeline_and_sets(cmd, f"install_migrations_{direction}")
            per_ghost_pid = self._per_ghost_pid_dispatch_count(direction)
            vkCmdDispatch(cmd, per_ghost_pid, 1, 1)
            self._record_compute_barrier(cmd)

        self._bind_pipeline_and_sets(cmd, "correction_all")
        vkCmdDispatch(cmd, per_p, 1, 1)
        self._record_compute_barrier(cmd)

        self._bind_pipeline_and_sets(cmd, "density_all")
        vkCmdDispatch(cmd, per_p, 1, 1)
        self._record_density_scratch_to_primary_copy(cmd)

        self._bind_pipeline_and_sets(cmd, "force_all")
        vkCmdDispatch(cmd, per_p, 1, 1)
        self._record_compute_barrier(cmd)

        self._bind_pipeline_and_sets(cmd, "bootstrap_half_kick")
        vkCmdDispatch(cmd, per_p, 1, 1)

        vkEndCommandBuffer(cmd)
        return cmd

    def bootstrap_init(self) -> None:
        """First half of split bootstrap: upload initial state + run
        initialize_voxelization + ghost_send + readback. After this returns,
        sender_staging_view(<dir>) holds the boundary replicas ready for
        host memcpy. Orchestrator calls this on both sims, then bridges via
        memcpy, then calls bootstrap_compute on both sims."""
        self._upload_initial_state()
        cmd = self._record_bootstrap_init_cmd()
        self.ctx.submit_and_wait(cmd)
        vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])

    def bootstrap_compute(self) -> None:
        """Second half of split bootstrap: upload ghost (host memcpy already
        landed in receiver_staging by caller) + install_migrations + correction
        + density + force + bootstrap_half_kick. After this returns, a_0 and
        v_{-1/2} are set with valid ghost-neighbor SPH contributions."""
        cmd = self._record_bootstrap_compute_cmd()
        self.ctx.submit_and_wait(cmd)
        vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])
        status = self.readback_global_status()
        print(f"[SimV2] bootstrap done: alive={status['alive_particle_count']} "
              f"overflow_inside={status['overflow_inside_count']} "
              f"overflow_incoming={status['overflow_incoming_count']}")

    def bootstrap(self) -> None:
        """Single-GPU bootstrap convenience: combine init + compute. No ghost
        transport needed when this sim has no peer."""
        if self._transport_segments:
            raise RuntimeError(
                "bootstrap() is single-GPU only; for dual-GPU call "
                "bootstrap_init() + (host memcpy) + bootstrap_compute() via "
                "DualGpuOrchestratorV2.bootstrap_all()")
        self.bootstrap_init()
        self.bootstrap_compute()

    # ========================================================================
    # Section 9: Readback (Phase 3)
    # ========================================================================

    def _readback_buffer(self, buf: _Buffer) -> bytes:
        staging = self._allocate_buffer(
            size=buf.size,
            usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            required_properties=(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                 | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
        )
        try:
            cmd = self._allocate_oneshot_cmd()
            vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
                flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
            vkCmdCopyBuffer(cmd, buf.handle, staging.handle, 1, [
                VkBufferCopy(srcOffset=0, dstOffset=0, size=buf.size)
            ])
            vkEndCommandBuffer(cmd)
            self.ctx.submit_and_wait(cmd)
            vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])
            mapped = vkMapMemory(self.ctx.device, staging.memory, 0, buf.size, 0)
            payload = bytes(np.frombuffer(mapped, dtype=np.uint8, count=buf.size))
            vkUnmapMemory(self.ctx.device, staging.memory)
            return payload
        finally:
            vkDestroyBuffer(self.ctx.device, staging.handle, None)
            vkFreeMemory(self.ctx.device, staging.memory, None)

    def readback_global_status(self) -> dict:
        """GlobalStatusBuffer is 16 × 4 B = 64 B. Layout per common.glsl §3."""
        payload = self._readback_buffer(self.buffers["global_status"])
        fields = struct.unpack("<I f I I I I I I I I I I I I I I", payload)
        return {
            "alive_particle_count":         fields[0],
            "maximum_velocity":             fields[1],
            "overflow_inside_count":        fields[2],
            "overflow_incoming_count":      fields[3],
            "first_overflow_voxel_inside":  fields[4],
            "first_overflow_voxel_incoming":fields[5],
            "correction_fallback_count":    fields[6],
            "overflow_ghost_count":         fields[7],
            "ghost_send_leading_count":     fields[8],
            "ghost_send_trailing_count":    fields[9],
            "ghost_recv_leading_count":     fields[10],
            "ghost_recv_trailing_count":    fields[11],
            "migration_install_count":      fields[12],
            "overflow_install_tail":        fields[13],
            "overflow_install_inside":      fields[14],
            "first_overflow_voxel_install": fields[15],
        }

    def readback_buffer_by_name(self, name: str) -> bytes:
        """Public wrapper of _readback_buffer for debug-log helpers.

        Issues one vkCmdCopyBuffer device→staging + fence-wait. Use for ad-hoc
        single-buffer inspection. For multi-buffer snapshot, prefer
        readback_buffers_batch() to amortize the fence cost.
        """
        if name not in self.buffers:
            raise KeyError(f"unknown buffer: {name!r}")
        return self._readback_buffer(self.buffers[name])

    def readback_buffers_batch(self, names: list[str]) -> dict[str, bytes]:
        """Read N buffers via a single cmd buffer + fence wait.

        Per-buffer fence overhead is the dominant cost when reading many small
        buffers; batching N copies into one submit drops 22 fence waits to 1
        for a 23-buffer snapshot (~100-200ms saved on cavity scale). Stagings
        are allocated/destroyed per call (debug-only path; not hot).
        """
        if not names:
            return {}
        for name in names:
            if name not in self.buffers:
                raise KeyError(f"unknown buffer: {name!r}")

        device = self.ctx.device
        # 1. Allocate one staging per source buffer
        stagings: dict[str, _Buffer] = {}
        for name in names:
            buf = self.buffers[name]
            stagings[name] = self._allocate_buffer(
                size=buf.size,
                usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                required_properties=(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                     | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
            )

        try:
            # 2. Record one cmd buffer with all copies
            cmd = self._allocate_oneshot_cmd()
            vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
                flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
            for name in names:
                buf = self.buffers[name]
                staging = stagings[name]
                vkCmdCopyBuffer(cmd, buf.handle, staging.handle, 1, [
                    VkBufferCopy(srcOffset=0, dstOffset=0, size=buf.size)
                ])
            vkEndCommandBuffer(cmd)

            # 3. Single fence-wait submit
            self.ctx.submit_and_wait(cmd)
            vkFreeCommandBuffers(device, self.ctx.command_pool, 1, [cmd])

            # 4. Map each staging, copy bytes out
            result: dict[str, bytes] = {}
            for name in names:
                staging = stagings[name]
                mapped = vkMapMemory(device, staging.memory, 0, staging.size, 0)
                result[name] = bytes(np.frombuffer(mapped, dtype=np.uint8,
                                                   count=staging.size))
                vkUnmapMemory(device, staging.memory)
            return result
        finally:
            for staging in stagings.values():
                vkDestroyBuffer(device, staging.handle, None)
                vkFreeMemory(device, staging.memory, None)

    # ========================================================================
    # Section 10: Step cmd buffers + sync2 timeline submit/wait (Phase 3b)
    #
    # V2 v1.0 single-GPU simplification: phase A only does predict + update_voxel;
    # phase B does correction(ALL); phase C does density + force. ghost_send /
    # install_migration / readback / upload are skipped (no peer; ghost-pid pool
    # is sized 0 in case construction). Phase 4 wires per-direction ghost flow.
    # ========================================================================

    def _record_phase_a_cmd(self):
        """V2 phase A: predict + update_voxel + (per-direction: reset ghost_send_count
        + ghost_send dispatch + readback vkCmdCopyBuffer + compute→host barrier).
        signal 3N+1 at submit."""
        cmd = self._allocate_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT))
        # Bench: phase A is the first cmd of the frame, so it owns the
        # per-frame step-slot reset. No-op when bench is unattached.
        self._bench_reset_step(cmd, "a_start")
        self._record_compute_barrier(cmd)

        per_p = self._per_own_particle_dispatch_count()
        per_v = self._per_extended_voxel_dispatch_count()
        per_yz = self._per_yz_face_dispatch_count()

        self._bind_pipeline_and_sets(cmd, "predict")
        vkCmdDispatch(cmd, per_p, 1, 1)
        self._bench_tick(cmd, "a_predict_end")
        self._record_compute_barrier(cmd)

        self._bind_pipeline_and_sets(cmd, "update_voxel")
        vkCmdDispatch(cmd, per_v, 1, 1)
        self._bench_tick(cmd, "a_voxel_end")

        # Per-direction ghost flow (skipped if this GPU has no peer on that side).
        # Three bench ticks split the block into (setup+dispatch) / (readback DMA)
        # / (host coherence barrier) so the bench runner can pinpoint where the
        # cross-vendor cost lives — see _run_v2_dual_bench.py output.
        for direction in ("leading", "trailing"):
            if direction not in self._transport_segments:
                continue
            self._record_compute_barrier(cmd)
            self._record_reset_ghost_send_count(cmd, direction)
            self._record_transfer_to_compute_barrier(cmd)
            self._bind_pipeline_and_sets(cmd, f"ghost_send_{direction}")
            vkCmdDispatch(cmd, per_yz, 1, 1)
            # Split tick 1: after pure compute dispatch (still device-local).
            self._bench_tick(cmd, f"a_ghost_{direction}_dispatch_end")
            self._record_compute_to_transfer_barrier(cmd)
            # Readback: device → sender_staging (the suspected NV bottleneck).
            self._record_readback_for_direction(cmd, direction)
            # Split tick 2: after readback DMA, before the host-visibility barrier.
            self._bench_tick(cmd, f"a_ghost_{direction}_readback_end")
            # compute→host barrier so worker's mapped read sees device writes
            self._record_compute_to_host_barrier(cmd)
            self._bench_tick(cmd, f"a_ghost_{direction}_end")

        vkEndCommandBuffer(cmd)
        return cmd

    def _record_phase_b_cmd(self):
        """V2 Phase B: correction(INTERIOR) over own pid range. Interior
        particles' support radius does NOT touch the ghost zone (by definition
        of NEIGHBOR_X_RANGE band), so they can safely read the ghost-vid
        inside_particle_index even though it still holds Phase A's ghost_send-
        written peer-frame pid scratch values — interior just never queries
        it. Boundary particles early-return in this kernel (CORRECTION_MODE_
        INTERIOR + in_boundary_band(coord)=true).

        Runs concurrently with the CPU sync window (worker memcpy). After
        Phase B's exit barrier, Phase C density/force reads correction_inverse
        writes (which only cover interior pids; boundary pids get filled by
        correction_boundary in Phase C). See docs/sph_v2_design.md §5.2 + §7.
        """
        cmd = self._allocate_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT))
        self._bench_tick(cmd, "b_start")
        # Entry: cross-submit memory visibility (Phase A writes → here reads)
        self._record_compute_barrier(cmd)

        per_p = self._per_own_particle_dispatch_count()
        self._bind_pipeline_and_sets(cmd, "correction_interior")
        vkCmdDispatch(cmd, per_p, 1, 1)

        # Exit: Phase B writes correction_inverse for interior pids; Phase C's
        # density.comp reads them. No semaphore bridges B→C, so explicit barrier
        # required (docs §5.2 step 3 — flagged 「存疑待排查」, included for safety).
        self._record_compute_barrier(cmd)
        self._bench_tick(cmd, "b_correction_interior_end")
        vkEndCommandBuffer(cmd)
        return cmd

    def _record_phase_c_cmd(self):
        """V2 Phase C: per-direction upload + install_migration → correction
        (BOUNDARY) → density → force. wait 3N+2 at entry; signal 3N+3 at end.

        correction_boundary runs ONLY on boundary-band particles (interior
        already covered by Phase B). It runs AFTER install_migrations so that:
          (a) newly arrived migrants (now own pids in the boundary column)
              get their M⁻¹ + density_gradient_kernel_sum computed this frame
          (b) all boundary particles see *uploaded* ghost data (set 1 ghost-vid
              inside_particle_index now has own-frame pid values, not the
              peer-frame scratch left over from Phase A's ghost_send).
        See docs/sph_v2_design.md §5.3 + §7."""
        cmd = self._allocate_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT))
        self._bench_tick(cmd, "c_start")

        per_p = self._per_own_particle_dispatch_count()

        # Per-direction upload + install_migration (skipped if no peer).
        # Split tick after upload DMA isolates host→device transfer cost from
        # the install_migrations.comp dispatch — symmetric to ghost_send's
        # readback split. See _run_v2_dual_bench.py output for derived metrics.
        for direction in ("leading", "trailing"):
            if direction not in self._transport_segments:
                continue
            self._record_upload_for_direction(cmd, direction)
            # Split tick: after upload DMA (host_staging → device).
            self._bench_tick(cmd, f"c_install_{direction}_upload_end")
            self._record_transfer_to_compute_barrier(cmd)
            self._bind_pipeline_and_sets(cmd, f"install_migrations_{direction}")
            per_ghost_pid = self._per_ghost_pid_dispatch_count(direction)
            vkCmdDispatch(cmd, per_ghost_pid, 1, 1)
            self._record_compute_barrier(cmd)
            self._bench_tick(cmd, f"c_install_{direction}_end")

        # Correction on boundary band only (interior was covered by Phase B).
        # Includes any new migrants installed above (they land in boundary
        # columns by construction).
        self._bind_pipeline_and_sets(cmd, "correction_boundary")
        vkCmdDispatch(cmd, per_p, 1, 1)
        self._record_compute_barrier(cmd)
        self._bench_tick(cmd, "c_correction_boundary_end")

        self._bind_pipeline_and_sets(cmd, "density_all")
        vkCmdDispatch(cmd, per_p, 1, 1)
        self._record_density_scratch_to_primary_copy(cmd)
        self._bench_tick(cmd, "c_density_end")

        self._bind_pipeline_and_sets(cmd, "force_all")
        vkCmdDispatch(cmd, per_p, 1, 1)
        self._bench_tick(cmd, "c_force_end")

        vkEndCommandBuffer(cmd)
        return cmd

    def prepare_step_cmd_buffers(self) -> None:
        """Record phase A/B/C cmd buffers once. Caller invokes after bootstrap.
        Re-call to re-record (e.g. when switching CORRECTION_MODE_ALL → split
        in Phase 6)."""
        device = self.ctx.device
        pool = self.ctx.command_pool
        for old in (self.phase_a_cmd, self.phase_b_cmd, self.phase_c_cmd):
            if old is not None:
                vkFreeCommandBuffers(device, pool, 1, [old])
        self.phase_a_cmd = self._record_phase_a_cmd()
        self.phase_b_cmd = self._record_phase_b_cmd()
        self.phase_c_cmd = self._record_phase_c_cmd()
        print(f"[SimV2] step cmd buffers recorded (phase A/B/C)")

    def _record_step_single_cmd(self):
        """Single-GPU combined cmd: predict + update_voxel + correction_all
        + density + force. No ghost flow, no timeline semaphores — caller
        submits with a plain fence wait per frame.

        Skips ghost_send and install_migrations because they are cross-GPU
        exclusive: no peer means nothing to replicate and nothing to install
        (predict's drift-to-ghost branch never fires when both ghost x
        thicknesses are 0; particles that leave the domain hit !in_own_grid
        and are killed locally). Uses correction_all instead of the
        interior/boundary split because there is no sync window to hide and
        in_boundary_band is empty when both ghost pools = 0 (split would
        run interior over all particles + boundary as a per-particle
        early-return no-op, equivalent but with one extra dispatch)."""
        cmd = self._allocate_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT))
        # Bench: this is the only step cmd on this path, so it owns the
        # per-frame step-slot reset. No-op when bench is unattached.
        self._bench_reset_step(cmd, "step_start")
        self._record_compute_barrier(cmd)

        per_p = self._per_own_particle_dispatch_count()
        per_v = self._per_extended_voxel_dispatch_count()

        self._bind_pipeline_and_sets(cmd, "predict")
        vkCmdDispatch(cmd, per_p, 1, 1)
        self._bench_tick(cmd, "predict_end")
        self._record_compute_barrier(cmd)

        self._bind_pipeline_and_sets(cmd, "update_voxel")
        vkCmdDispatch(cmd, per_v, 1, 1)
        self._bench_tick(cmd, "voxel_end")
        self._record_compute_barrier(cmd)

        if self.step_single_use_split:
            # P3.C validation: substitute _all variants with their split
            # equivalents. In single-GPU mode in_boundary_band always returns
            # false (LEADING/TRAILING_GHOST_VOXEL_COUNT = 0), so:
            #   _interior / _deep_interior cover ALL particles (identical to _all)
            #   _boundary covers ZERO particles (every thread early-returns)
            # Output must therefore be bit-identical to the non-split path —
            # any divergence in alive count or per-buffer state proves a
            # shader-side bug.
            self._bind_pipeline_and_sets(cmd, "correction_interior")
            vkCmdDispatch(cmd, per_p, 1, 1)
            self._record_compute_barrier(cmd)
            self._bind_pipeline_and_sets(cmd, "correction_boundary")
            vkCmdDispatch(cmd, per_p, 1, 1)
        else:
            self._bind_pipeline_and_sets(cmd, "correction_all")
            vkCmdDispatch(cmd, per_p, 1, 1)
        self._bench_tick(cmd, "correction_end")
        self._record_compute_barrier(cmd)

        if self.step_single_use_split:
            self._bind_pipeline_and_sets(cmd, "density_deep_interior")
            vkCmdDispatch(cmd, per_p, 1, 1)
            self._record_compute_barrier(cmd)
            self._bind_pipeline_and_sets(cmd, "density_boundary")
            vkCmdDispatch(cmd, per_p, 1, 1)
        else:
            self._bind_pipeline_and_sets(cmd, "density_all")
            vkCmdDispatch(cmd, per_p, 1, 1)
        self._record_density_scratch_to_primary_copy(cmd)
        self._bench_tick(cmd, "density_end")
        self._record_compute_barrier(cmd)

        if self.step_single_use_split:
            self._bind_pipeline_and_sets(cmd, "force_deep_interior")
            vkCmdDispatch(cmd, per_p, 1, 1)
            self._record_compute_barrier(cmd)
            self._bind_pipeline_and_sets(cmd, "force_boundary")
            vkCmdDispatch(cmd, per_p, 1, 1)
        else:
            self._bind_pipeline_and_sets(cmd, "force_all")
            vkCmdDispatch(cmd, per_p, 1, 1)
        self._bench_tick(cmd, "force_end")

        vkEndCommandBuffer(cmd)
        return cmd

    def prepare_step_single_cmd_buffer(self) -> None:
        """Record the single-GPU combined step cmd buffer once. Caller invokes
        after bootstrap. Mutually exclusive with prepare_step_cmd_buffers()
        — the dual-GPU 3-submit path requires this sim to have at least one
        peer direction, which conflicts with the single-GPU assumption."""
        if self._transport_segments:
            raise RuntimeError(
                "prepare_step_single_cmd_buffer() requires a no-peer sim; "
                "this sim has transport segments for "
                f"{sorted(self._transport_segments)}. Use "
                "prepare_step_cmd_buffers() (dual-GPU 3-submit path) instead.")
        device = self.ctx.device
        pool = self.ctx.command_pool
        if self.step_single_cmd is not None:
            vkFreeCommandBuffers(device, pool, 1, [self.step_single_cmd])
        self.step_single_cmd = self._record_step_single_cmd()
        print(f"[SimV2] step cmd buffer recorded (single-GPU combined)")

    def submit_step_single_and_wait(self) -> None:
        """Submit the single-GPU step cmd buffer and block on its fence.
        One submit + one wait per frame — no timeline semaphores, no peer
        sync. Used by _run_v2_single_bench.py and any future single-GPU
        runner."""
        if self.step_single_cmd is None:
            raise RuntimeError(
                "step_single_cmd not recorded; "
                "call prepare_step_single_cmd_buffer()")
        self.ctx.submit_and_wait(self.step_single_cmd)

    # ----- sync2 timeline submit / wait / host signal -----------------------

    def submit_with_timeline(
        self,
        cmd,
        *,
        wait_value: Optional[int],
        signal_value: Optional[int],
        wait_stage: int = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    ) -> None:
        wait_infos = []
        if wait_value is not None:
            wait_infos.append(VkSemaphoreSubmitInfo(
                sType=VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                semaphore=self.timeline,
                value=wait_value,
                stageMask=wait_stage,
            ))
        signal_infos = []
        if signal_value is not None:
            signal_infos.append(VkSemaphoreSubmitInfo(
                sType=VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                semaphore=self.timeline,
                value=signal_value,
                stageMask=VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            ))
        cmd_info = VkCommandBufferSubmitInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            commandBuffer=cmd,
        )
        submit_info = VkSubmitInfo2(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            waitSemaphoreInfoCount=len(wait_infos),
            pWaitSemaphoreInfos=wait_infos if wait_infos else None,
            commandBufferInfoCount=1,
            pCommandBufferInfos=[cmd_info],
            signalSemaphoreInfoCount=len(signal_infos),
            pSignalSemaphoreInfos=signal_infos if signal_infos else None,
        )
        vkQueueSubmit2(self.ctx.compute_queue, 1, [submit_info], VK_NULL_HANDLE)

    def wait_timeline(self, value: int, timeout_ns: int = 0xFFFFFFFFFFFFFFFF) -> bool:
        """vkWaitSemaphores with optional timeout. Default INFINITE.
        Returns True if value was reached, False if timed out.

        python-vulkan raises an exception on VK_TIMEOUT rather than returning
        it as a value, so we catch the timeout case explicitly. Other
        VkResult errors (DEVICE_LOST etc.) propagate.
        """
        info = VkSemaphoreWaitInfo(
            sType=VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            semaphoreCount=1,
            pSemaphores=[self.timeline],
            pValues=[value],
        )
        try:
            vkWaitSemaphores(self.ctx.device, info, timeout_ns)
            return True
        except Exception as e:
            # python-vulkan maps VK_TIMEOUT (=2) to VkTimeout class
            if type(e).__name__ == "VkTimeout":
                return False
            raise

    def host_signal_timeline(self, value: int) -> None:
        """vkSignalSemaphore for host-side timeline advance. Workers call this
        on the *destination* sim. Single-GPU smoke tests call it on self
        between phase B and phase C to substitute for the absent worker."""
        info = VkSemaphoreSignalInfo(
            sType=VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
            semaphore=self.timeline,
            value=value,
        )
        vkSignalSemaphore(self.ctx.device, info)

    # ----- High-level frame API (replaces earlier stubs) --------------------

    def submit_phase_a(self, frame_n: int) -> None:
        if self.phase_a_cmd is None:
            raise RuntimeError("phase_a_cmd not recorded; call prepare_step_cmd_buffers()")
        wait_value = (self.value_frame_done(frame_n - 1) if frame_n > 0 else 0)
        self.submit_with_timeline(
            self.phase_a_cmd,
            wait_value=wait_value,
            signal_value=self.value_phase_a_done(frame_n),
        )

    def submit_phase_b(self, frame_n: int) -> None:
        # Queue-ordered after phase A on same queue; no semaphore wait/signal.
        if self.phase_b_cmd is None:
            raise RuntimeError("phase_b_cmd not recorded; call prepare_step_cmd_buffers()")
        self.submit_with_timeline(
            self.phase_b_cmd, wait_value=None, signal_value=None)

    def submit_phase_c(self, frame_n: int) -> None:
        if self.phase_c_cmd is None:
            raise RuntimeError("phase_c_cmd not recorded; call prepare_step_cmd_buffers()")
        self.submit_with_timeline(
            self.phase_c_cmd,
            wait_value=self.value_cpu_sync_done(frame_n),
            signal_value=self.value_frame_done(frame_n),
        )

    def wait_frame_done(self, frame_n: int) -> None:
        self.wait_timeline(self.value_frame_done(frame_n))

    # ========================================================================
    # Section 11: Defrag pipeline + cmd (Phase 3c)
    # 5-set pipeline layout: sets 0..3 reused + set 4 for scratch destination
    # SoA. Skips binding 2 (density_pressure_scratch) — transient, no need
    # to migrate.
    # ========================================================================

    def _build_defrag_set4_layout(self):
        bindings = [
            VkDescriptorSetLayoutBinding(
                binding=b,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            )
            for b in DEFRAG_SET0_BINDINGS
        ]
        ci = VkDescriptorSetLayoutCreateInfo(
            bindingCount=len(bindings), pBindings=bindings)
        return vkCreateDescriptorSetLayout(self.ctx.device, ci, None)

    def _allocate_defrag_set4(self):
        pool_size = VkDescriptorPoolSize(
            type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=len(DEFRAG_SET0_BINDINGS),
        )
        pool_ci = VkDescriptorPoolCreateInfo(
            maxSets=1, poolSizeCount=1, pPoolSizes=[pool_size])
        pool = vkCreateDescriptorPool(self.ctx.device, pool_ci, None)
        alloc = VkDescriptorSetAllocateInfo(
            descriptorPool=pool,
            descriptorSetCount=1,
            pSetLayouts=[self.defrag_set4_layout],
        )
        descriptor_set = vkAllocateDescriptorSets(self.ctx.device, alloc)[0]
        return pool, descriptor_set

    def _wire_defrag_set4(self) -> None:
        # Names match set 0 SoA — same buffer keys for scratch and primary.
        # Skip binding 2 (density_pressure_scratch); it's not in
        # DEFRAG_SET0_BINDINGS so no scratch twin exists.
        writes = []
        for spec in self._buffer_specs:
            if spec.set_index != 0:
                continue
            if spec.binding not in DEFRAG_SET0_BINDINGS:
                continue
            scratch_buf = self.scratch_buffers[spec.name]
            buf_info = VkDescriptorBufferInfo(
                buffer=scratch_buf.handle, offset=0, range=scratch_buf.size)
            writes.append(VkWriteDescriptorSet(
                dstSet=self.defrag_set4,
                dstBinding=spec.binding,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[buf_info],
            ))
        vkUpdateDescriptorSets(self.ctx.device, len(writes), writes, 0, None)

    def _build_defrag_pipeline_layout(self):
        # 5 sets: existing 4 (0..3) + defrag's set 4
        layouts = list(self.descriptor_layouts) + [self.defrag_set4_layout]
        ci = VkPipelineLayoutCreateInfo(
            setLayoutCount=5, pSetLayouts=layouts, pushConstantRangeCount=0)
        return vkCreatePipelineLayout(self.ctx.device, ci, None)

    def _build_defrag_pipeline(self):
        spec_info = self._make_spec_info(self._global_entries())
        stage = VkPipelineShaderStageCreateInfo(
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            module=self.shader_modules["defrag"],
            pName="main",
            pSpecializationInfo=spec_info,
        )
        ci = VkComputePipelineCreateInfo(
            stage=stage, layout=self.defrag_pipeline_layout)
        return vkCreateComputePipelines(
            self.ctx.device, VK_NULL_HANDLE, 1, [ci], None)[0]

    def _record_defrag_cmd(self):
        """V2 defrag cmd buffer (mirrors V1 _record_defrag_cmd):
            1. fill defrag_scratch_counter = 0
            2. transfer→compute barrier
            3. defrag dispatch (per extended voxel)
            4. compute→transfer barrier
            5. copy each scratch SoA → primary (9 copies, set 0 except scratch)
            6. copy defrag_scratch_counter → global_status.alive_particle_count
            7. fill migration_install_count = 0
            8. transfer→compute barrier (next step's predict reads set 0)
        """
        cmd = self._allocate_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT))
        # Bench: defrag owns its own slot-range reset (the step reset in
        # phase_a_cmd doesn't cover defrag slots — defrag may not run that
        # frame, and we don't want stale defrag readings).
        self._bench_reset_defrag(cmd, "defrag_start")
        self._record_compute_barrier(cmd)

        # 1. Reset scratch counter
        scratch_counter = self.buffers["defrag_scratch_counter"]
        vkCmdFillBuffer(cmd, scratch_counter.handle, 0, scratch_counter.size, 0)

        # 1b. Zero each defrag scratch SoA before dispatch.
        # defrag.comp only writes scratch[own_first_pid .. own_first_pid + alive - 1].
        # Without this fill, the dead-tail (slots beyond alive) keeps stale bytes
        # from PREVIOUS defrag runs — when vkCmdCopyBuffer scratch→primary
        # overwrites primary wholesale, those stale bytes resurrect "ghost
        # particles" with mass>0 and valid voxel_id but no inside_particle_index
        # reference (orphans). Investigation 2026-05-19 traced the alive-count
        # drift bug to exactly this; see logs/test_run/snapshots analysis.
        # Cost: ~167 MB total fill per defrag → ~3 ms at VRAM bandwidth, runs
        # every defrag_cadence (1000) frames → <0.05% wall-clock impact.
        for spec in self._buffer_specs:
            if spec.set_index != 0:
                continue
            if spec.binding not in DEFRAG_SET0_BINDINGS:
                continue
            scratch_buf = self.scratch_buffers[spec.name]
            vkCmdFillBuffer(cmd, scratch_buf.handle, 0, scratch_buf.size, 0)

        # 2. transfer→compute (defrag dispatch will atomicAdd the counter)
        self._record_transfer_to_compute_barrier(cmd)

        # 3. defrag dispatch — uses defrag_pipeline_layout (5 sets)
        vkCmdBindPipeline(
            cmd, VK_PIPELINE_BIND_POINT_COMPUTE, self.pipelines["defrag"])
        all_sets = list(self.descriptor_sets) + [self.defrag_set4]
        vkCmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_COMPUTE, self.defrag_pipeline_layout,
            0, 5, all_sets, 0, None)
        vkCmdDispatch(cmd, self._per_extended_voxel_dispatch_count(), 1, 1)

        # 4. compute→transfer (copy scratch→primary)
        self._record_compute_to_transfer_barrier(cmd)

        # 5. Copy each scratch SoA back to primary
        for spec in self._buffer_specs:
            if spec.set_index != 0:
                continue
            if spec.binding not in DEFRAG_SET0_BINDINGS:
                continue
            src = self.scratch_buffers[spec.name].handle
            dst = self.buffers[spec.name].handle
            vkCmdCopyBuffer(cmd, src, dst, 1, [
                VkBufferCopy(srcOffset=0, dstOffset=0, size=spec.size)
            ])

        # 6. Refresh alive_particle_count from defrag_scratch_counter.
        # GlobalStatusBuffer layout: alive_particle_count at offset 0, 4 B.
        vkCmdCopyBuffer(
            cmd, scratch_counter.handle,
            self.buffers["global_status"].handle, 1, [
                VkBufferCopy(srcOffset=0, dstOffset=0, size=4)
            ])

        # 7. Reset migration_install_count (offset 12 × 4 = 48 in global_status,
        # cf. common.glsl GlobalStatusBuffer field order).
        vkCmdFillBuffer(cmd, self.buffers["global_status"].handle, 48, 4, 0)

        # 8. transfer→compute (next step's predict reads set 0)
        self._record_transfer_to_compute_barrier(cmd)
        self._bench_tick(cmd, "defrag_end")

        vkEndCommandBuffer(cmd)
        return cmd
