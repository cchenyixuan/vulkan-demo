"""
simulator.py — SPH simulator full stack: SPV loading, buffer allocation,
descriptor sets, compute pipelines, command buffer recording, and the run
loop. Consumes a Case (utils.sph.case) + a VulkanContext (utils.sph.vulkan_context).

V0 scope:
  - Headless single-GPU compute. No swapchain / surface; a future renderer
    pulls VkBuffer handles via get_render_buffers().
  - Fully synchronous fence-per-step (no CPU/GPU overlap; V2 work).
  - One command buffer per parity (ping-pong A↔B for density_pressure).
"""

import pathlib
import struct
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
from vulkan import *
from vulkan._vulkancache import ffi

from utils.sph.case import Case, build_specialization_info
from utils.sph.vulkan_context import VulkanContext


# ============================================================================
# Module-level constants
# ============================================================================

SHADER_DIR = pathlib.Path(__file__).resolve().parents[2] / "shaders" / "spv" / "sph"

# Order matters for log readability; not for correctness.
SHADER_NAMES = [
    "initialize_voxelization",
    "predict",
    "update_voxel",
    "correction",
    "density",
    "force",
    "bootstrap_half_kick",
]

# Fill value for set-2 ghost dummy buffers. uint = 3735928559; if any shader
# accidentally reads it (e.g. DCE failure when V1 work is half-done), it will
# blow up immediately rather than silently behave like a dead particle (0).
GHOST_DUMMY_FILL = 0xDEADBEEF


# ============================================================================
# Helpers / dataclasses
# ============================================================================


@dataclass
class Buffer:
    """One VkBuffer + VkDeviceMemory pair, plus byte size for debug."""
    handle: object                      # VkBuffer
    memory: object                      # VkDeviceMemory
    size: int


@dataclass
class _BufferSpec:
    """Internal spec used by SphSimulator._allocate_buffers."""
    name: str                           # dict key in self.buffers
    set_index: int                      # 0/1/2/3
    binding: int                        # binding in that set
    size: int                           # bytes
    usage: int                          # VkBufferUsageFlags


# ============================================================================
# SphSimulator
# ============================================================================


class SphSimulator:
    """SPH simulator. Owns all Vulkan resources except the platform context.

    Lifecycle:
        with VulkanContext.create() as ctx:
            sim = SphSimulator(ctx, case)
            sim.bootstrap()
            sim.run_until(total_time=1.0)
            sim.destroy()             # or use as context manager
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, ctx: VulkanContext, case: Case):
        self.ctx = ctx
        self.case = case
        self._destroyed = False

        # Run state
        self.parity = 0
        self.simulation_time = 0.0
        self.step_count = 0

        # Decision #3: verify WORKGROUP_SIZE fits the device's compute limits.
        self._check_workgroup_limit()

        # Section 1: SPV → VkShaderModule
        self.shader_modules: dict = self._load_shader_modules()

        # Section 2 + 3: allocate every buffer, then upload initial state.
        self.buffers: dict[str, Buffer] = self._allocate_buffers()
        self._upload_initial_state()

        # Section 4: descriptor layouts + pool + sets, then wire buffers.
        self.descriptor_layouts: list = self._build_descriptor_layouts()
        self.descriptor_pool = self._create_descriptor_pool()
        self.descriptor_sets: dict = self._allocate_descriptor_sets()
        self._wire_descriptor_sets()

        # Section 5: pipeline layout + spec info + 7 compute pipelines.
        self.pipeline_layout = self._build_pipeline_layout()
        self._spec_info_data = None             # kept alive for the pipelines' lifetime
        self._spec_info_entries = None          # ditto
        self.spec_info = self._build_vk_specialization_info()
        self.pipelines: dict = self._build_compute_pipelines()

        # Section 6: pre-recorded command buffers (bootstrap + step×2 parity).
        self.bootstrap_cmd = self._record_bootstrap_cmd()
        self.step_cmds = [self._record_step_cmd(parity=0),
                          self._record_step_cmd(parity=1)]

    # ==================================================================
    # Section 0: pre-flight checks
    # ==================================================================

    def _check_workgroup_limit(self) -> None:
        properties = vkGetPhysicalDeviceProperties(self.ctx.physical_device)
        limits = properties.limits
        requested = self.case.capacities.workgroup

        max_size_x = limits.maxComputeWorkGroupSize[0]
        max_size_y = limits.maxComputeWorkGroupSize[1]
        max_size_z = limits.maxComputeWorkGroupSize[2]
        max_invocations = limits.maxComputeWorkGroupInvocations

        print(f"[Simulator] WORKGROUP_SIZE check:")
        print(f"  requested workgroup x = {requested}")
        print(f"  device maxComputeWorkGroupSize  = ({max_size_x}, {max_size_y}, {max_size_z})")
        print(f"  device maxComputeWorkGroupInvocations = {max_invocations}")

        if requested > max_size_x:
            raise RuntimeError(
                f"capacities.workgroup ({requested}) exceeds device limit "
                f"maxComputeWorkGroupSize.x ({max_size_x})")
        if requested > max_invocations:
            raise RuntimeError(
                f"capacities.workgroup ({requested}) exceeds device limit "
                f"maxComputeWorkGroupInvocations ({max_invocations})")
        print(f"  ok\n")

    # ==================================================================
    # Section 1: SPV loading
    # ==================================================================

    def _load_shader_modules(self) -> dict:
        modules = {}
        for name in SHADER_NAMES:
            spv_path = SHADER_DIR / f"{name}.comp.spv"
            if not spv_path.exists():
                raise FileNotFoundError(
                    f"compiled shader missing: {spv_path}. Run compile_shaders.py.")
            spv_bytes = spv_path.read_bytes()
            create_info = VkShaderModuleCreateInfo(
                codeSize=len(spv_bytes),
                pCode=spv_bytes,
            )
            modules[name] = vkCreateShaderModule(self.ctx.device, create_info, None)
        print(f"[Simulator] loaded {len(modules)} shader modules from {SHADER_DIR}")
        return modules

    # ==================================================================
    # Section 2: Buffer allocation
    # ==================================================================

    def _build_buffer_specs(self) -> list[_BufferSpec]:
        """Return spec list for every buffer the simulator owns. Sizes are
        derived purely from Case (closest-packing-validated); usage flags
        leave room for V0+ visualization (VERTEX bit on render-relevant
        particle buffers)."""
        case = self.case
        pool_capacity = case.capacities.pool_size + 1                 # 1-based slot count
        voxel_capacity = (case.grid["dimension"][0]
                          * case.grid["dimension"][1]
                          * case.grid["dimension"][2]) + 1            # 1-based voxel count
        cap_inside = case.capacities.max_per_voxel
        cap_incoming = case.capacities.max_incoming
        n_materials = len(case.materials)

        BSU = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        TRANSFER = (VK_BUFFER_USAGE_TRANSFER_DST_BIT
                    | VK_BUFFER_USAGE_TRANSFER_SRC_BIT)
        VERT = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT

        return [
            # ---- Set 0: particle SoA (10 bindings) ----------------
            _BufferSpec("position_voxel_id",            0, 0, 16 * pool_capacity,        BSU | TRANSFER | VERT),
            _BufferSpec("density_pressure_a",           0, 1,  8 * pool_capacity,        BSU | TRANSFER | VERT),
            _BufferSpec("density_pressure_b",           0, 2,  8 * pool_capacity,        BSU | TRANSFER | VERT),
            _BufferSpec("velocity_mass",                0, 3, 16 * pool_capacity,        BSU | TRANSFER | VERT),
            _BufferSpec("acceleration",                 0, 4, 16 * pool_capacity,        BSU | TRANSFER),
            _BufferSpec("shift",                        0, 5, 16 * pool_capacity,        BSU | TRANSFER),
            _BufferSpec("material",                     0, 6,  4 * pool_capacity,        BSU | TRANSFER),
            _BufferSpec("correction_inverse",           0, 7, 32 * pool_capacity,        BSU | TRANSFER),
            _BufferSpec("density_gradient_kernel_sum", 0, 8, 16 * pool_capacity,        BSU | TRANSFER),
            _BufferSpec("extension_fields",             0, 9, 16 * pool_capacity,        BSU | TRANSFER),

            # ---- Set 1: voxel cells (4 bindings) ------------------
            _BufferSpec("inside_particle_count",        1, 0,  4 * voxel_capacity,                       BSU | TRANSFER),
            _BufferSpec("incoming_particle_count",      1, 1,  4 * voxel_capacity,                       BSU | TRANSFER),
            _BufferSpec("inside_particle_index",        1, 2,  4 * voxel_capacity * cap_inside,          BSU | TRANSFER),
            _BufferSpec("incoming_particle_index",      1, 3,  4 * voxel_capacity * cap_incoming,        BSU | TRANSFER),

            # ---- Set 2: ghost dummies (V0: minimum size, 0xDEADBEEF fill) ----
            _BufferSpec("ghost_position_voxel_id",      2, 0,  16,                                       BSU | TRANSFER),
            _BufferSpec("ghost_density_pressure",       2, 1,   8,                                       BSU | TRANSFER),
            _BufferSpec("ghost_velocity_mass",          2, 3,  16,                                       BSU | TRANSFER),
            _BufferSpec("ghost_acceleration",           2, 4,  16,                                       BSU | TRANSFER),
            _BufferSpec("ghost_shift",                  2, 5,  16,                                       BSU | TRANSFER),
            _BufferSpec("ghost_material",               2, 6,   4,                                       BSU | TRANSFER),
            _BufferSpec("ghost_inside_particle_count",  2, 10,  4,                                       BSU | TRANSFER),
            _BufferSpec("ghost_inside_particle_index",  2, 12,  4 * cap_inside,                          BSU | TRANSFER),

            # ---- Set 3: global / transport / materials (8 bindings) ----
            _BufferSpec("global_status",                3, 0,  64,                                       BSU | TRANSFER),
            _BufferSpec("overflow_log",                 3, 1,  64,                                       BSU | TRANSFER),
            _BufferSpec("inlet_template",               3, 2,  32,                                       BSU | TRANSFER),
            _BufferSpec("dispatch_indirect",            3, 3,  16,                                       BSU | TRANSFER),
            _BufferSpec("ghost_out_packet",             3, 4,  16,                                       BSU | TRANSFER),
            _BufferSpec("ghost_in_staging",             3, 5,  16,                                       BSU | TRANSFER),
            _BufferSpec("diagnostic",                   3, 6,  16,                                       BSU | TRANSFER),
            _BufferSpec("material_parameters",          3, 7,  48 * max(n_materials, 1),                 BSU | TRANSFER),
        ]

    def _allocate_buffer(
        self,
        size: int,
        usage: int,
        memory_properties: int,
    ) -> Buffer:
        """One-shot helper: VkBuffer + VkDeviceMemory of the given properties.
        Each buffer gets its own VkDeviceMemory in V0 (no sub-allocator)."""
        if size == 0:
            raise ValueError("Buffer size must be > 0")
        buffer_create_info = VkBufferCreateInfo(
            size=size,
            usage=usage,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE,
        )
        handle = vkCreateBuffer(self.ctx.device, buffer_create_info, None)
        memory_requirements = vkGetBufferMemoryRequirements(self.ctx.device, handle)
        memory_type_index = self.ctx.find_memory_type(
            memory_requirements.memoryTypeBits, memory_properties)
        allocate_info = VkMemoryAllocateInfo(
            allocationSize=memory_requirements.size,
            memoryTypeIndex=memory_type_index,
        )
        memory = vkAllocateMemory(self.ctx.device, allocate_info, None)
        vkBindBufferMemory(self.ctx.device, handle, memory, 0)
        return Buffer(handle=handle, memory=memory, size=size)

    def _allocate_buffers(self) -> dict[str, Buffer]:
        """Allocate all SSBOs as device-local. Initial data uploaded separately
        via staging in _upload_initial_state."""
        specs = self._build_buffer_specs()
        buffers: dict[str, Buffer] = {}
        total_bytes = 0
        for spec in specs:
            buffers[spec.name] = self._allocate_buffer(
                size=spec.size,
                usage=spec.usage,
                memory_properties=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            )
            total_bytes += spec.size
        # Stash the spec list for later (descriptor wiring uses set/binding).
        self._buffer_specs = specs
        print(f"[Simulator] allocated {len(buffers)} buffers, "
              f"{total_bytes / (1024*1024):.2f} MB total (device-local)")
        return buffers

    # ==================================================================
    # Section 3: Initial data upload
    # ==================================================================

    def _build_initial_data(self) -> dict[str, bytes]:
        """Return name → bytes for every buffer that needs non-zero data at start.
        Buffers absent from this dict are zero-initialized.

        Per design decision #4: alive_particle_count is NOT pre-set here. The
        initialize_voxelization shader atomically counts it after placement,
        so global_status starts at all zeros and the GPU produces the
        authoritative count.
        """
        case = self.case
        pool_capacity = case.capacities.pool_size + 1
        data: dict[str, bytes] = {}

        # ---- positions: (x, y, z, 0) per particle ; voxel_id .w stays 0 -----
        positions = np.zeros((pool_capacity, 4), dtype=np.float32)
        cursor = 1                              # 1-based: slot 0 unused
        for source in case.particle_sources:
            n = int(source.vertices.shape[0])
            positions[cursor:cursor + n, 0:3] = source.vertices
            cursor += n
        data["position_voxel_id"] = positions.tobytes()

        # ---- velocity_mass: v from material.initial_velocity (default 0);
        # mass = ρ₀ × volume (per-particle material). For BOUNDARY kind with
        # nonzero initial_velocity (e.g. lid-driven cavity top), predict.comp
        # skips → velocity persists for the entire run, modelling a moving
        # wall. For FLUID kind, initial_velocity is just the IC and predict
        # then evolves it normally.
        velocity_mass = np.zeros((pool_capacity, 4), dtype=np.float32)
        cursor = 1
        for source in case.particle_sources:
            n = int(source.vertices.shape[0])
            material = case.materials[source.material_group_id]
            mass = material.rest_density * material.volume
            velocity_mass[cursor:cursor + n, 0] = material.initial_velocity[0]
            velocity_mass[cursor:cursor + n, 1] = material.initial_velocity[1]
            velocity_mass[cursor:cursor + n, 2] = material.initial_velocity[2]
            velocity_mass[cursor:cursor + n, 3] = mass
            cursor += n
        data["velocity_mass"] = velocity_mass.tobytes()

        # ---- density_pressure_a: (ρ₀, 0) per particle  -----------------------
        # IC declaration: every particle starts at rest at its material's
        # rest_density. With ρ=ρ₀, the EOS gives P=0 — bootstrap's force pass
        # then has no spurious pressure burst (a divide-by-zero would occur
        # if ρ were left at the zero-init default).
        # Padding slots stay at (0, 0) — the init shader's mass-sentinel keeps
        # them out of any neighbor loop, so 0 is fine there.
        density_pressure_a = np.zeros((pool_capacity, 2), dtype=np.float32)
        cursor = 1
        for source in case.particle_sources:
            n = int(source.vertices.shape[0])
            material = case.materials[source.material_group_id]
            density_pressure_a[cursor:cursor + n, 0] = material.rest_density
            # column 1 (pressure) stays 0
            cursor += n
        data["density_pressure_a"] = density_pressure_a.tobytes()

        # ---- material: per-particle group_id ; 0 for unallocated slots ------
        material_array = np.zeros(pool_capacity, dtype=np.uint32)
        cursor = 1
        for source in case.particle_sources:
            n = int(source.vertices.shape[0])
            material_array[cursor:cursor + n] = source.material_group_id
            cursor += n
        data["material"] = material_array.tobytes()

        # ---- material_parameters: 48 B per material, common.glsl struct order
        material_blob = bytearray()
        for material in case.materials:
            row = struct.pack(
                "I f f f f f f f f f I I",
                int(material.kind),
                float(material.rest_density),
                float(material.viscosity),
                float(material.eos_constant),
                float(material.smoothing_length),
                float(material.radius),
                float(material.volume),
                float(material.rotor_angular_velocity),
                float(material.viscosity_transfer),
                float(material.viscosity_rotation),
                int(material.reserved_material_0),
                int(material.reserved_material_1),
            )
            assert len(row) == 48
            material_blob.extend(row)
        # Pad to allocated buffer size if N_materials == 0 (sanity, V0 always >0)
        if len(material_blob) == 0:
            material_blob = b"\x00" * 48
        data["material_parameters"] = bytes(material_blob)

        # ---- ghost dummies: 0xDEADBEEF fill (decision #4 / Set 2 strategy) --
        ghost_names = [s.name for s in self._buffer_specs if s.set_index == 2]
        for name in ghost_names:
            buffer_size = self.buffers[name].size
            num_uints = buffer_size // 4
            arr = np.full(num_uints, GHOST_DUMMY_FILL, dtype=np.uint32)
            data[name] = arr.tobytes()

        return data

    def _staging_upload(self, dest: Buffer, payload: bytes) -> None:
        """Upload bytes into a device-local buffer via a one-shot staging copy."""
        if len(payload) > dest.size:
            raise ValueError(
                f"upload payload ({len(payload)}B) exceeds dest buffer size "
                f"({dest.size}B)")

        # 1. host-visible staging
        staging = self._allocate_buffer(
            size=len(payload),
            usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            memory_properties=(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                               | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
        )
        try:
            # 2. slice-assign into the mapped staging memory
            mapped = vkMapMemory(self.ctx.device, staging.memory, 0, len(payload), 0)
            mapped[:len(payload)] = payload
            vkUnmapMemory(self.ctx.device, staging.memory)

            # 3. record + submit a one-shot copy command
            cmd = self._allocate_oneshot_cmd()
            vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
                flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
            region = VkBufferCopy(srcOffset=0, dstOffset=0, size=len(payload))
            vkCmdCopyBuffer(cmd, staging.handle, dest.handle, 1, [region])
            vkEndCommandBuffer(cmd)
            self.ctx.submit_and_wait(cmd)
            vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])
        finally:
            # 4. release staging
            vkDestroyBuffer(self.ctx.device, staging.handle, None)
            vkFreeMemory(self.ctx.device, staging.memory, None)

    def _zero_buffer(self, dest: Buffer) -> None:
        """Issue vkCmdFillBuffer to zero a device-local buffer."""
        cmd = self._allocate_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
        vkCmdFillBuffer(cmd, dest.handle, 0, dest.size, 0)
        vkEndCommandBuffer(cmd)
        self.ctx.submit_and_wait(cmd)
        vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])

    def _upload_initial_state(self) -> None:
        initial_data = self._build_initial_data()
        for name, buffer in self.buffers.items():
            if name in initial_data:
                payload = initial_data[name]
                # Pad if shorter than the buffer (e.g. material_parameters)
                if len(payload) < buffer.size:
                    payload = payload + b"\x00" * (buffer.size - len(payload))
                self._staging_upload(buffer, payload)
            else:
                # Default: zero-initialize via vkCmdFillBuffer (no staging needed)
                self._zero_buffer(buffer)
        print(f"[Simulator] uploaded initial state to {len(self.buffers)} buffers")

    # ==================================================================
    # Section 4: Descriptor sets
    # ==================================================================

    def _build_descriptor_layouts(self) -> list:
        """One VkDescriptorSetLayout per descriptor set (4 total). All
        bindings are STORAGE_BUFFER usable from compute shaders."""
        layouts = []
        for set_index in range(4):
            specs_in_set = [s for s in self._buffer_specs if s.set_index == set_index]
            bindings = []
            for spec in specs_in_set:
                bindings.append(VkDescriptorSetLayoutBinding(
                    binding=spec.binding,
                    descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1,
                    stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
                ))
            create_info = VkDescriptorSetLayoutCreateInfo(
                bindingCount=len(bindings),
                pBindings=bindings,
            )
            layouts.append(vkCreateDescriptorSetLayout(self.ctx.device, create_info, None))
        return layouts

    def _create_descriptor_pool(self):
        """Pool sized for: set 0 × 2 (ping-pong) + set 1, 2, 3 = 5 sets total.
        Counts the storage buffer descriptors across all 5 sets."""
        # Across set 0 (×2), set 1, set 2, set 3:
        per_set_binding_count = {
            i: sum(1 for s in self._buffer_specs if s.set_index == i)
            for i in range(4)
        }
        total_descriptors = (
            2 * per_set_binding_count[0]
            + per_set_binding_count[1]
            + per_set_binding_count[2]
            + per_set_binding_count[3]
        )
        pool_size = VkDescriptorPoolSize(
            type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=total_descriptors,
        )
        create_info = VkDescriptorPoolCreateInfo(
            maxSets=5,
            poolSizeCount=1,
            pPoolSizes=[pool_size],
        )
        return vkCreateDescriptorPool(self.ctx.device, create_info, None)

    def _allocate_descriptor_sets(self) -> dict:
        """Allocate 5 descriptor sets:
            set_0_even, set_0_odd  — ping-pong
            set_1, set_2, set_3
        """
        # Order: even, odd, set1, set2, set3 (must match key order below)
        layouts_to_allocate = [
            self.descriptor_layouts[0],     # set_0_even
            self.descriptor_layouts[0],     # set_0_odd
            self.descriptor_layouts[1],     # set_1
            self.descriptor_layouts[2],     # set_2
            self.descriptor_layouts[3],     # set_3
        ]
        allocate_info = VkDescriptorSetAllocateInfo(
            descriptorPool=self.descriptor_pool,
            descriptorSetCount=len(layouts_to_allocate),
            pSetLayouts=layouts_to_allocate,
        )
        allocated = vkAllocateDescriptorSets(self.ctx.device, allocate_info)
        return {
            "set_0_even": allocated[0],
            "set_0_odd":  allocated[1],
            "set_1":      allocated[2],
            "set_2":      allocated[3],
            "set_3":      allocated[4],
        }

    def _wire_descriptor_sets(self) -> None:
        """vkUpdateDescriptorSets: point every (set, binding) at its buffer.
        For ping-pong: set_0_even reads A→1, B→2; set_0_odd reads B→1, A→2."""
        writes = []

        def add_write(descriptor_set, binding, buffer):
            # range=VK_WHOLE_SIZE would be ideal but python-vulkan represents
            # it as a signed -1 sentinel, triggering OverflowError. Use the
            # explicit byte size — equivalent for offset=0.
            buffer_info = VkDescriptorBufferInfo(
                buffer=buffer.handle,
                offset=0,
                range=buffer.size,
            )
            writes.append(VkWriteDescriptorSet(
                dstSet=descriptor_set,
                dstBinding=binding,
                dstArrayElement=0,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                pBufferInfo=[buffer_info],
            ))

        # ---- Set 0 (× 2 instances) ----------------------------------------
        for spec in self._buffer_specs:
            if spec.set_index != 0:
                continue
            buffer = self.buffers[spec.name]
            # Even = "natural" wiring: density_pressure_a@1, density_pressure_b@2
            # Odd  = swapped:           density_pressure_b@1, density_pressure_a@2
            if spec.name == "density_pressure_a":
                add_write(self.descriptor_sets["set_0_even"], 1, buffer)
                add_write(self.descriptor_sets["set_0_odd"],  2, buffer)
            elif spec.name == "density_pressure_b":
                add_write(self.descriptor_sets["set_0_even"], 2, buffer)
                add_write(self.descriptor_sets["set_0_odd"],  1, buffer)
            else:
                add_write(self.descriptor_sets["set_0_even"], spec.binding, buffer)
                add_write(self.descriptor_sets["set_0_odd"],  spec.binding, buffer)

        # ---- Set 1, 2, 3 (single instance each) ----------------------------
        for set_index, set_key in [(1, "set_1"), (2, "set_2"), (3, "set_3")]:
            for spec in self._buffer_specs:
                if spec.set_index != set_index:
                    continue
                add_write(self.descriptor_sets[set_key], spec.binding, self.buffers[spec.name])

        vkUpdateDescriptorSets(self.ctx.device, len(writes), writes, 0, None)

    # ==================================================================
    # Section 5: Pipelines
    # ==================================================================

    def _build_pipeline_layout(self):
        create_info = VkPipelineLayoutCreateInfo(
            setLayoutCount=len(self.descriptor_layouts),
            pSetLayouts=self.descriptor_layouts,
            pushConstantRangeCount=0,
            pPushConstantRanges=[],
        )
        return vkCreatePipelineLayout(self.ctx.device, create_info, None)

    def _build_vk_specialization_info(self):
        """Convert case.SpecializationInfo → VkSpecializationInfo. The data
        blob and entries list must outlive pipeline creation (we keep refs
        on self._spec_info_data / self._spec_info_entries).

        ``pData`` is declared ``const void *`` — python-vulkan can't infer the
        element size of ``void``, so we pre-allocate a uint8_t cffi array and
        pass that. Keeping the cffi cdata as an instance attribute prevents
        Python GC from freeing it before the pipeline is created.
        """
        info = build_specialization_info(self.case)
        self._spec_info_data = info.data
        self._spec_info_data_cdata = ffi.new("uint8_t[]", self._spec_info_data)
        self._spec_info_entries = [
            VkSpecializationMapEntry(constantID=cid, offset=off, size=sz)
            for cid, off, sz in info.map_entries
        ]
        return VkSpecializationInfo(
            mapEntryCount=len(self._spec_info_entries),
            pMapEntries=self._spec_info_entries,
            dataSize=len(self._spec_info_data),
            pData=self._spec_info_data_cdata,
        )

    def _build_compute_pipelines(self) -> dict:
        """One VkPipeline per shader, all sharing pipeline_layout + spec_info."""
        create_infos = []
        for name in SHADER_NAMES:
            stage = VkPipelineShaderStageCreateInfo(
                stage=VK_SHADER_STAGE_COMPUTE_BIT,
                module=self.shader_modules[name],
                pName="main",
                pSpecializationInfo=self.spec_info,
            )
            create_infos.append(VkComputePipelineCreateInfo(
                stage=stage,
                layout=self.pipeline_layout,
            ))
        # vkCreateComputePipelines returns a list when creating multiple
        result = vkCreateComputePipelines(
            self.ctx.device,
            VK_NULL_HANDLE,                 # no pipeline cache in V0
            len(create_infos),
            create_infos,
            None,
        )
        return dict(zip(SHADER_NAMES, result))

    # ==================================================================
    # Section 6: Command buffer recording
    # ==================================================================

    def _allocate_oneshot_cmd(self):
        allocate_info = VkCommandBufferAllocateInfo(
            commandPool=self.ctx.command_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        return vkAllocateCommandBuffers(self.ctx.device, allocate_info)[0]

    def _record_compute_barrier(self, cmd) -> None:
        """Global memory barrier between two compute dispatches."""
        barrier = VkMemoryBarrier(
            sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        )
        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,       # src
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,       # dst
            0,                                          # dependency flags
            1, [barrier],                               # memory
            0, None,                                    # buffer
            0, None,                                    # image
        )

    def _bind_pipeline_and_sets(self, cmd, pipeline_name: str, set_0_key: str) -> None:
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          self.pipelines[pipeline_name])
        sets = [
            self.descriptor_sets[set_0_key],
            self.descriptor_sets["set_1"],
            self.descriptor_sets["set_2"],
            self.descriptor_sets["set_3"],
        ]
        vkCmdBindDescriptorSets(
            cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            self.pipeline_layout,
            0,                                          # first set
            len(sets), sets,
            0, None,                                    # dynamic offsets
        )

    def _per_particle_dispatch_count(self) -> int:
        wg = self.case.capacities.workgroup
        pool = self.case.capacities.pool_size
        return (pool + wg - 1) // wg

    def _per_voxel_dispatch_count(self) -> int:
        wg = self.case.capacities.workgroup
        voxel_count = (self.case.grid["dimension"][0]
                       * self.case.grid["dimension"][1]
                       * self.case.grid["dimension"][2])
        return (voxel_count + wg - 1) // wg

    def _record_bootstrap_cmd(self):
        """One-time startup: initialize_voxelization → correction → density →
        force → bootstrap_half_kick. Bound to set_0_even (parity 0)."""
        cmd = self._allocate_oneshot_cmd()
        # SIMULTANEOUS_USE_BIT: this cmd buffer is pre-recorded once and may be
        # submitted multiple times without waiting for prior submissions to
        # complete (renderer's wait=False path keeps several copies in flight).
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT))

        # Leading barrier: makes this cmd buffer self-protective when callers
        # submit it without waiting on an explicit fence (renderer fast path
        # — see step(wait=False)). Harmless on the wait-fenced path.
        self._record_compute_barrier(cmd)

        per_p = self._per_particle_dispatch_count()

        # 1. Bucket-sort particles into voxels (writes inside_*, voxel_id, alive_count)
        self._bind_pipeline_and_sets(cmd, "initialize_voxelization", "set_0_even")
        vkCmdDispatch(cmd, per_p, 1, 1)
        self._record_compute_barrier(cmd)

        # 2. KCG correction matrix + ∇ρ + kernel_sum
        self._bind_pipeline_and_sets(cmd, "correction", "set_0_even")
        vkCmdDispatch(cmd, per_p, 1, 1)
        self._record_compute_barrier(cmd)

        # 3. Density continuity + Tait EOS → density_pressure_b
        self._bind_pipeline_and_sets(cmd, "density", "set_0_even")
        vkCmdDispatch(cmd, per_p, 1, 1)
        self._record_compute_barrier(cmd)

        # 4. Force (pressure + viscosity + PST shift) → acceleration, shift
        # Note: force reads density_pressure_b which density just wrote → barrier above.
        self._bind_pipeline_and_sets(cmd, "force", "set_0_even")
        vkCmdDispatch(cmd, per_p, 1, 1)
        self._record_compute_barrier(cmd)

        # 5. Backward half-kick: v_0 → v_{-1/2}
        self._bind_pipeline_and_sets(cmd, "bootstrap_half_kick", "set_0_even")
        vkCmdDispatch(cmd, per_p, 1, 1)

        vkEndCommandBuffer(cmd)
        return cmd

    def _record_step_cmd(self, parity: int):
        """One main step: predict → update_voxel → correction → density → force.
        parity=0 reads density_pressure_a (set_0_even); parity=1 reads B."""
        cmd = self._allocate_oneshot_cmd()
        # SIMULTANEOUS_USE_BIT: this cmd buffer is pre-recorded once and may be
        # submitted multiple times without waiting for prior submissions to
        # complete (renderer's wait=False path keeps several copies in flight).
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT))

        # Leading barrier: when callers submit successive step cmds without
        # a CPU fence between them (renderer fast path), this barrier flushes
        # the previous submission's compute writes so this step's first
        # dispatch (predict, reading position/velocity/acceleration/shift)
        # sees a coherent view. Harmless when callers use submit_and_wait.
        self._record_compute_barrier(cmd)

        per_p = self._per_particle_dispatch_count()
        per_v = self._per_voxel_dispatch_count()
        set_0_key = "set_0_even" if parity == 0 else "set_0_odd"

        # 1. predict (per-particle): kick + drift + crossing detection
        self._bind_pipeline_and_sets(cmd, "predict", set_0_key)
        vkCmdDispatch(cmd, per_p, 1, 1)
        self._record_compute_barrier(cmd)

        # 2. update_voxel (per-voxel): compact inside + merge incoming
        self._bind_pipeline_and_sets(cmd, "update_voxel", set_0_key)
        vkCmdDispatch(cmd, per_v, 1, 1)
        self._record_compute_barrier(cmd)

        # 3. correction (per-particle): KCG matrix + ∇ρ + kernel_sum
        self._bind_pipeline_and_sets(cmd, "correction", set_0_key)
        vkCmdDispatch(cmd, per_p, 1, 1)
        self._record_compute_barrier(cmd)

        # 4. density (per-particle): writes the OPPOSITE ping-pong slot
        self._bind_pipeline_and_sets(cmd, "density", set_0_key)
        vkCmdDispatch(cmd, per_p, 1, 1)
        self._record_compute_barrier(cmd)

        # 5. force (per-particle)
        self._bind_pipeline_and_sets(cmd, "force", set_0_key)
        vkCmdDispatch(cmd, per_p, 1, 1)

        vkEndCommandBuffer(cmd)
        return cmd

    # ==================================================================
    # Section 7: Public run API
    # ==================================================================

    def bootstrap(self) -> None:
        """Run the one-time startup sequence. Must be called before step()."""
        self.ctx.submit_and_wait(self.bootstrap_cmd)
        # Optional sanity check: GPU should report no overflow during init.
        status = self.readback_global_status()
        if status["overflow_inside_count"] != 0 or status["overflow_incoming_count"] != 0:
            raise RuntimeError(
                f"GPU initial voxelization overflowed despite closest-packing "
                f"validation: overflow_inside={status['overflow_inside_count']}, "
                f"overflow_incoming={status['overflow_incoming_count']}, "
                f"first_offending_inside_voxel={status['first_overflow_voxel_inside']}. "
                f"Likely a CPU/GPU floor() mismatch near a voxel boundary.")
        print(f"[Simulator] bootstrap ok: alive={status['alive_particle_count']}")

    def step(self, *, wait: bool = True) -> None:
        """Run one main step (predict → update_voxel → correction → density → force).

        wait=True  (default): submit and block on a fresh fence; the function
            returns only after the GPU finishes. Safe for any caller; what
            you want when you're going to readback right after.
        wait=False: fire-and-forget submit. Compute work joins the queue but
            the CPU does not wait — appropriate for the renderer's hot path,
            where the next vkAcquireNextImage / vkWaitForFences on the render
            fence pulls in all pending compute work. Step's leading barrier
            keeps inter-submission memory dependencies satisfied.
        """
        cmd = self.step_cmds[self.parity]
        if wait:
            self.ctx.submit_and_wait(cmd)
        else:
            submit_info = VkSubmitInfo(
                sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
                commandBufferCount=1,
                pCommandBuffers=[cmd],
            )
            vkQueueSubmit(self.ctx.compute_queue, 1, submit_info, VK_NULL_HANDLE)
        self.parity ^= 1
        self.simulation_time += self.case.timestep
        self.step_count += 1

    def run_until(
        self,
        total_time: Optional[float] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        """Step until any of: argument total_time, argument max_steps,
        case.time.total, case.time.max_steps is exhausted."""
        time_budget = self.case.time
        while True:
            if total_time is not None and self.simulation_time >= total_time:
                return
            if max_steps is not None and self.step_count >= max_steps:
                return
            if time_budget.is_time_exceeded(self.simulation_time):
                return
            if time_budget.is_step_exceeded(self.step_count):
                return
            self.step()

    # ==================================================================
    # Section 8: Readback + render hook
    # ==================================================================

    def _readback_buffer(self, buffer: Buffer) -> bytes:
        """Copy a device-local buffer back via host-visible staging."""
        staging = self._allocate_buffer(
            size=buffer.size,
            usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            memory_properties=(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                               | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
        )
        try:
            cmd = self._allocate_oneshot_cmd()
            vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
                flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
            region = VkBufferCopy(srcOffset=0, dstOffset=0, size=buffer.size)
            vkCmdCopyBuffer(cmd, buffer.handle, staging.handle, 1, [region])
            vkEndCommandBuffer(cmd)
            self.ctx.submit_and_wait(cmd)
            vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])

            # python-vulkan vkMapMemory already returns a buffer-like cffi
            # object (indexable, sliceable). bytes() copies it out.
            mapped = vkMapMemory(self.ctx.device, staging.memory, 0, buffer.size, 0)
            data = bytes(mapped[0:buffer.size])
            vkUnmapMemory(self.ctx.device, staging.memory)
            return data
        finally:
            vkDestroyBuffer(self.ctx.device, staging.handle, None)
            vkFreeMemory(self.ctx.device, staging.memory, None)

    def readback_global_status(self) -> dict:
        """Decode the 64 B GlobalStatus block. Field order matches common.glsl."""
        raw = self._readback_buffer(self.buffers["global_status"])
        unpacked = struct.unpack("I I f I I I I I I I I I I I I I", raw)
        return {
            "alive_particle_count":          unpacked[0],
            "frame_counter":                 unpacked[1],
            "maximum_velocity":              unpacked[2],
            "inlet_spawn_count":             unpacked[3],
            "overflow_inside_count":         unpacked[4],
            "overflow_incoming_count":       unpacked[5],
            "first_overflow_voxel_inside":   unpacked[6],
            "first_overflow_voxel_incoming": unpacked[7],
            "correction_fallback_count":     unpacked[8],
        }

    def readback_positions(self) -> np.ndarray:
        """Return (POOL_SIZE+1, 4) float32 array of position_voxel_id."""
        raw = self._readback_buffer(self.buffers["position_voxel_id"])
        return np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)

    def get_render_buffers(self) -> dict:
        """Buffer handles for a future renderer (V0+ visualization layer).
        The renderer creates its own graphics pipeline and reads these as
        instanced vertex sources / SSBOs. Buffer USAGE flags already include
        VERTEX_BUFFER_BIT for the keys returned here."""
        # Note: density_pressure_a/b switches semantics each step (parity).
        # For correctness, renderer should use parity to pick the read side.
        return {
            "position_voxel_id":    self.buffers["position_voxel_id"].handle,
            "velocity_mass":        self.buffers["velocity_mass"].handle,
            "density_pressure_a":   self.buffers["density_pressure_a"].handle,
            "density_pressure_b":   self.buffers["density_pressure_b"].handle,
            "global_status":        self.buffers["global_status"].handle,
        }

    # ==================================================================
    # Section 9: Cleanup
    # ==================================================================

    def destroy(self) -> None:
        if self._destroyed:
            return
        device = self.ctx.device
        if device is None:
            self._destroyed = True
            return

        vkDeviceWaitIdle(device)

        # Cmd buffers freed when their pool is destroyed (ctx owns the pool).
        # We allocated them from ctx.command_pool, so explicit free is optional
        # but cleaner.
        cmds = [self.bootstrap_cmd] + self.step_cmds
        vkFreeCommandBuffers(device, self.ctx.command_pool, len(cmds), cmds)

        for pipeline in self.pipelines.values():
            vkDestroyPipeline(device, pipeline, None)

        vkDestroyPipelineLayout(device, self.pipeline_layout, None)
        # spec info (no destroy fn — refs released when self goes out of scope)

        vkDestroyDescriptorPool(device, self.descriptor_pool, None)
        for layout in self.descriptor_layouts:
            vkDestroyDescriptorSetLayout(device, layout, None)

        for buffer in self.buffers.values():
            vkDestroyBuffer(device, buffer.handle, None)
            vkFreeMemory(device, buffer.memory, None)

        for module in self.shader_modules.values():
            vkDestroyShaderModule(device, module, None)

        self._destroyed = True

    def __enter__(self) -> "SphSimulator":
        return self

    def __exit__(self, *_) -> None:
        self.destroy()
