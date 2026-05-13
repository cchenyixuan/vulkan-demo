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
        case = make_minimal_test_case(...)
        with SphSimulatorV2(ctx, case) as sim:
            # Phase 3: sim.bootstrap(); sim.submit_phase_a(n); ...
            pass
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

        # Timeline semaphore (Phase 3 uses it; creation is here)
        self.timeline = self._create_timeline_semaphore()

        # cmd buffer slots reserved for Phase 3
        self.phase_a_cmd: Any = None
        self.phase_b_cmd: Any = None
        self.phase_c_cmd: Any = None
        self.defrag_cmd: Any = None

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

        if self.pipeline_layout is not None:
            vkDestroyPipelineLayout(device, self.pipeline_layout, None)
            self.pipeline_layout = None

        if self.descriptor_pool is not None:
            vkDestroyDescriptorPool(device, self.descriptor_pool, None)
            self.descriptor_pool = None

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
    # High-level frame API — Phase 3 fills bodies
    # ========================================================================

    def bootstrap(self) -> None:
        raise NotImplementedError("Phase 3")

    def submit_phase_a(self, frame_n: int) -> None:
        raise NotImplementedError("Phase 3")

    def submit_phase_b(self, frame_n: int) -> None:
        raise NotImplementedError("Phase 3")

    def submit_phase_c(self, frame_n: int) -> None:
        raise NotImplementedError("Phase 3")

    def wait_frame_done(self, frame_n: int) -> None:
        raise NotImplementedError("Phase 3")

    def submit_defrag_and_wait(self) -> None:
        raise NotImplementedError("Phase 3")

    def submit_with_timeline(
        self, cmd, *, wait_value: Optional[int], signal_value: Optional[int]
    ) -> None:
        raise NotImplementedError("Phase 3")

    def wait_timeline(self, value: int) -> None:
        raise NotImplementedError("Phase 3")

    # ========================================================================
    # Worker-facing accessors
    # ========================================================================

    def sender_staging_view(self):
        """numpy.uint8 view over the *combined* sender staging blob (leading
        + trailing if both peers). Worker indexes by direction-relative offset."""
        # Phase 2 returns the leading staging view if exists, else trailing.
        # Phase 4 worker will use direction-specific accessors.
        raise NotImplementedError("Phase 4: needs direction parameter")

    def receiver_staging_view(self):
        raise NotImplementedError("Phase 4: needs direction parameter")

    def timeline_semaphore(self):
        return self.timeline

    def device(self):
        return self.ctx.device

    # ========================================================================
    # Readback (Phase 3+; placeholder)
    # ========================================================================

    def readback_global_status(self) -> dict:
        raise NotImplementedError("Phase 3")

    def readback_positions(self):
        raise NotImplementedError("Phase 3")

    def get_render_buffers(self) -> dict:
        raise NotImplementedError("Phase 5+ (renderer)")

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

    def _allocate_staging_buffers(self) -> dict[str, _Buffer]:
        """Per-direction host-visible stagings (sender CACHED, receiver COHERENT)
        — see docs/sph_v2_design.md §14.5. Persistent-mapped at construction.

        Naming: {sender, receiver}_staging_{leading, trailing}. Only allocated
        for directions where THIS GPU has a peer (transport.has_*_peer)."""
        # Per-direction staging size = sum of segment sizes. Segments mirror
        # V1's transport: 10 SoA × ghost-pid range + 2 set-1 ghost-vid buffers
        # + 1 set-3 count field. Detailed segment math is Phase 4 (worker uses
        # it to compute offsets); for Phase 2 we just need a size upper bound.
        # Conservative bound: 10 vec4 + 2 set-1 buffers + 4 B count, per pool.
        case = self.case

        def staging_size_for_pool(pool_size: int) -> int:
            if pool_size == 0:
                return 0
            voxels_in_dir = case.ghost_grid.leading_ghost_voxel_count  # same per dir
            # Use the max of leading/trailing for simplicity (Phase 4 will refine)
            voxels_in_dir = max(case.ghost_grid.leading_ghost_voxel_count,
                                case.ghost_grid.trailing_ghost_voxel_count)
            cap_inside = case.capacities.max_particles_per_voxel
            # Conservative: each ghost-pid carries ~136 B of SoA (V1 packet).
            packet_bytes = 16 + 8 + 16 + 16 + 16 + 4 + 32 + 16 + 16  # 140 B
            per_pid_bytes = packet_bytes * pool_size
            voxel_bytes = (4 * voxels_in_dir) + (4 * voxels_in_dir * cap_inside)
            count_bytes = 4
            return per_pid_bytes + voxel_bytes + count_bytes

        stagings: dict[str, _Buffer] = {}
        # Required + preferred for sender (HOST_CACHED preferred)
        sender_required = (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                           | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        sender_preferred = VK_MEMORY_PROPERTY_HOST_CACHED_BIT
        receiver_required = (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                             | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        usage = (VK_BUFFER_USAGE_TRANSFER_DST_BIT
                 | VK_BUFFER_USAGE_TRANSFER_SRC_BIT)

        for direction_name, peer_attr, pool_attr in (
            ("leading",  "has_leading_peer",  "leading_ghost_pool_size"),
            ("trailing", "has_trailing_peer", "trailing_ghost_pool_size"),
        ):
            if not getattr(case.transport, peer_attr):
                continue
            pool_size = getattr(case.capacities, pool_attr)
            size = staging_size_for_pool(pool_size)
            if size == 0:
                continue
            # Sender side
            sender_buf = self._allocate_buffer(
                size, usage, sender_required, sender_preferred)
            mapped = vkMapMemory(self.ctx.device, sender_buf.memory, 0, size, 0)
            sender_buf.mapped = mapped
            sender_buf.mapped_view = np.frombuffer(mapped, dtype=np.uint8, count=size)
            stagings[f"sender_staging_{direction_name}"] = sender_buf

            # Receiver side
            recv_buf = self._allocate_buffer(size, usage, receiver_required)
            mapped_r = vkMapMemory(self.ctx.device, recv_buf.memory, 0, size, 0)
            recv_buf.mapped = mapped_r
            recv_buf.mapped_view = np.frombuffer(mapped_r, dtype=np.uint8, count=size)
            stagings[f"receiver_staging_{direction_name}"] = recv_buf

        total = sum(b.size for b in stagings.values())
        print(f"[SimV2] host staging buffers: {len(stagings)}, "
              f"{total / 1024:.1f} KB (persistent-mapped)")
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

    def _pack_global_spec(self) -> tuple[bytes, list[VkSpecializationMapEntry]]:
        """Build (data_blob, map_entries) for global spec consts (ids 0-83
        roughly, excluding ghost/correction per-pipeline overrides).

        Layout matches common.glsl id table. Encoded as little-endian
        per the SPIR-V spec const ABI (matches vulkan host byte order on x86)."""
        p = self.case.physics
        n = self.case.numerics
        cap = self.case.capacities
        g = self.case.grid
        gh = self.case.ghost_grid

        entries: list[tuple[int, str, Any]] = [
            # (id, fmt, value); fmt = 'f'=float32, 'I'=uint32, 'i'=int32, 'B'=bool-as-uint32
            (0,  'f', p.smoothing_length),
            (1,  'f', p.speed_of_sound),
            (2,  'f', p.delta_coefficient),
            (4,  'f', p.power_parameter),
            (5,  'f', p.cfl_number),
            (6,  'f', p.timestep),
            (7,  'f', g.origin_x),
            (8,  'f', g.origin_y),
            (9,  'f', g.origin_z),
            (10, 'B', 0),                       # STRICT_BIT_EXACT (V2 v1.0 = false)
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
            (82, 'I', 0),                       # NEIGHBOR_X_RANGE; V2 overlap mode sets this
        ]

        return self._pack_spec(entries)

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
            (10, 'B', 0),
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
            (82, 'I', 0),
        ]

    def _correction_mode_entry(self, mode: int) -> list[tuple[int, str, Any]]:
        # constant_id=47 CORRECTION_MODE, 82 NEIGHBOR_X_RANGE (overlap mode)
        # V2 v1.0 first cut: keep NEIGHBOR_X_RANGE=0 → in_boundary_band always
        # false → CORRECTION_MODE_ALL behaves identically to V1.
        # Phase 6 will set NEIGHBOR_X_RANGE=2 + INTERIOR/BOUNDARY modes.
        return [(47, 'I', mode)]

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
        """Build the 12 compute pipelines:

            1 × initialize_voxelization
            1 × bootstrap_half_kick
            1 × predict
            1 × update_voxel
            2 × ghost_send (leading, trailing)
            2 × install_migrations (leading, trailing)
            3 × correction (ALL, INTERIOR, BOUNDARY)
            1 × density
            1 × force
            1 × defrag

        ghost_send + install_migrations are always built for BOTH directions
        even if this GPU has no peer on that side; phase A/C cmd recording
        skips dispatch on the unused direction (cf. V1)."""
        pipelines: dict[str, object] = {}

        # Pipelines that only need global spec consts (and the shared 4-set
        # pipeline layout). defrag is excluded here — it uses set 4 for its
        # destination SoA buffers and needs a 5-set layout; built in Phase 3
        # together with its cmd buffer.
        for key in ("initialize_voxelization", "bootstrap_half_kick",
                    "predict", "update_voxel", "density", "force"):
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

        # correction × 3 modes (V2 #1)
        # mode values: 0 = ALL (v1.0 default; V1-equivalent), 1 = INTERIOR, 2 = BOUNDARY
        for mode, mode_name in ((0, "all"), (1, "interior"), (2, "boundary")):
            pipelines[f"correction_{mode_name}"] = self._create_pipeline(
                shader=self.shader_modules["correction"],
                entries=self._global_entries() + self._correction_mode_entry(mode),
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
