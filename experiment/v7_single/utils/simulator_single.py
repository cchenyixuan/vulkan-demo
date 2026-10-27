"""
simulator_single.py — the v7 kernels on one GPU, with nothing of the multi-GPU host side.

SphSimulatorSingle runs the same SPIR-V as experiment/v7 (loaded from experiment/v7/shaders/spv
after the manifest check, spirv_manifest.py) with the same specialization constants as v7's
single-GPU mode (docs/seam_audit/v7_single.md, inventory 6) and records the same step, bootstrap,
defrag and restart command streams (inventory 7). What it does not have (inventory 1-3, 9-10): a
transfer queue, CONCURRENT buffers, timeline semaphores, submit locks, fast-submit batches, ghost /
migrant / departed pools, the phase A / B / C recordings, worker threads, the chain orchestrator,
the BenchTimer, and every V7_* switch but the three kernel switches (switches.py).

Host loop: one pre-recorded SIMULTANEOUS_USE step command buffer, a ring of ``in_flight`` persistent
fences (in_flight = 1: one fence, submit + wait per step), the defrag command buffer appended to the
step's submission at the case's cadence or submitted alone (drained) by the caller.

The case is v7's CaseV7 from experiment.v7.utils.case_loader_v7.load_case_v7 (a degenerate slab:
no peer, no ghost columns), which this class checks on construction.
"""

from __future__ import annotations

import pathlib
import struct
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from vulkan import *  # noqa: F401, F403
from vulkan._vulkancache import ffi

from experiment.v7.utils.case_v7 import CaseV7, KIND_FLUID
from experiment.v7.utils.defrag_prefix import upload_defrag_prefix
from experiment.v7_single.utils import spirv_manifest
from experiment.v7_single.utils.deep_wall_single import resolve_deep_wall_skip
from experiment.v7_single.utils.switches import Switches, configured_switches
from experiment.v7_single.utils.vulkan_context_single import VulkanContextSingle

# Test hook, not a switch (experiment/seam_audit/canonical_dump.py sets it before building the simulator, as
# simulator_v7._DEEP_WALL_RECORD_DECISIONS): record every deep-wall skip decision (spec 113, the per-particle
# record buffer) so a dump can name the rows whose L / kernel sum were kept. Production runs never record.
_DEEP_WALL_RECORD_DECISIONS = False

# The release band widths (v7's V7_BAND_WIDTHS default 2,2,3): NEIGHBOR_X_RANGE (spec 82) of the correction /
# density / force pipelines. On a slab without ghost columns no thread is ever in a band; the values are
# fixed here only so the specialized SPIR-V is the same as v7's single-GPU mode (inventory 6).
BAND_WIDTHS = (2, 2, 3)
DEEP_WALL_SKIP_BAND_WIDTH = BAND_WIDTHS[1]
# density_scratch_copy.comp: four (first slot, slot count) regions (spec 101-108) and a fixed local size of 256.
DENSITY_COPY_REGION_LIMIT = 4
DENSITY_COPY_LOCAL_SIZE = 256

# Set 0 bindings defrag.comp writes into the scratch twins (everything but binding 2, which IS the scratch, and
# binding 10, the transient wall dummy velocity).
DEFRAG_SET0_BINDINGS = (0, 1, 3, 4, 5, 6, 7, 8, 9)
GLOBAL_STATUS_BYTES = 160
GLOBAL_STATUS_FIELD_NAMES = (
    "alive_particle_count", "maximum_velocity", "overflow_inside_count", "overflow_incoming_count",
    "first_overflow_voxel_inside", "first_overflow_voxel_incoming", "correction_fallback_count",
    "overflow_ghost_count", "ghost_send_leading_count", "ghost_send_trailing_count", "ghost_recv_leading_count",
    "ghost_recv_trailing_count", "migration_install_count", "overflow_install_tail", "overflow_install_inside",
    "first_overflow_voxel_install", "frame_stamp", "ghost_stamp_leading", "ghost_stamp_trailing",
    "stamp_error_count", "stamp_error_sample", "departed_count", "overflow_departed_count", "far_migration_count",
    "replica_inner_send_leading_count", "replica_inner_send_trailing_count", "replica_outer_send_leading_count",
    "replica_outer_send_trailing_count", "replica_inner_recv_leading_count", "replica_inner_recv_trailing_count",
    "replica_outer_recv_leading_count", "replica_outer_recv_trailing_count", "initialization_seam_clamp_count",
    "overflow_initialization_outside", "overflow_deep_wall_skip_count",
)
UINT64_MAX = 0xFFFFFFFFFFFFFFFF


@dataclass
class _Buffer:
    handle: object
    memory: object
    size: int


@dataclass
class _BufferSpec:
    name: str
    set_index: int
    binding: int
    size: int
    usage: int


def tick_labels(switches: Switches, deep_wall_active: bool, wall_adami: bool) -> tuple[str, ...]:
    """The tick labels record_step emits, in order (the kernel breakdown's stage boundaries)."""
    labels = ["step_start", "predict_end", "voxel_end"]
    if deep_wall_active:
        labels.append("deep_wall_marker_end")
    if switches.fused_correction_density:
        labels.append("correction_density_end")
    else:
        labels += ["correction_end", "density_kernel_end"]
    labels.append("density_end")
    if wall_adami:
        labels.append("wall_extrapolate_end")
    labels.append("force_end")
    return tuple(labels)


class SphSimulatorSingle:
    """The v7 kernels on one GPU (module docstring)."""

    RESTART_FIELD_LAYOUT = {
        "position_voxel_id":           (4, np.float32),
        "velocity_mass":               (4, np.float32),
        "density_pressure":            (2, np.float32),
        "acceleration":                (4, np.float32),
        "shift":                       (4, np.float32),
        "material":                    (1, np.uint32),
        "correction_inverse":          (8, np.float32),
        "density_gradient_kernel_sum": (4, np.float32),
        "extension_fields":            (4, np.float32),
    }

    def __init__(self, ctx: VulkanContextSingle, case: CaseV7, *, switches: Optional[Switches] = None,
                 spirv_directory: Optional[pathlib.Path] = None) -> None:
        self.ctx = ctx
        self.case = case
        self.switches = switches if switches is not None else configured_switches()
        self.spirv_directory = pathlib.Path(spirv_directory or spirv_manifest.V7_SHADER_DIRECTORY)
        self._check_degenerate_case()
        self._check_workgroup_limit()
        self.wall_boundary = case.numerics.wall_boundary
        self.deep_wall_skip_resolution = resolve_deep_wall_skip(case, self.switches.deep_wall_skip,
                                                                DEEP_WALL_SKIP_BAND_WIDTH)
        self.fused_correction_density_resolution = (
            (True, f"one neighbour traversal for correction + density, band {BAND_WIDTHS[0]}")
            if self.switches.fused_correction_density else (False, "V7_FUSED_CORRECTION_DENSITY=0"))
        self.deep_wall_record = bool(_DEEP_WALL_RECORD_DECISIONS)
        print(f"[SimSingle] switches: {self.switches.header()}; V7_DEEP_WALL_SKIP -> "
              f"{'on' if self.deep_wall_active else 'off'} ({self.deep_wall_skip_resolution[1]}); "
              f"V7_FUSED_CORRECTION_DENSITY -> {'fused' if self.fused_active else 'separate kernels'}; "
              f"wall_boundary={self.wall_boundary}")

        self._spec_keepalive: list = []
        self._buffer_specs = self._build_buffer_specs()
        self.buffers = self._allocate_buffers()
        self.scratch_buffers = self._allocate_scratch_buffers()
        self.descriptor_layouts = self._build_descriptor_layouts()
        self.descriptor_pool = self._create_descriptor_pool()
        self.descriptor_sets = self._allocate_descriptor_sets()
        self._wire_descriptor_sets()
        self.pipeline_layout = self._build_pipeline_layout()
        self.shader_modules = self._load_shader_modules()
        self.pipelines = self._build_compute_pipelines()
        self.defrag_set4_layout = self._build_defrag_set4_layout()
        self.defrag_descriptor_pool, self.defrag_set4 = self._allocate_defrag_set4()
        self._wire_defrag_set4()
        self.defrag_pipeline_layout = self._build_defrag_pipeline_layout()
        self.pipelines["defrag"] = self._build_defrag_pipeline()

        self.step_cmd: Any = None
        self.defrag_cmd: Any = None
        self._defrag_prefix_records: list = []             # E9: one record per host scan (utils/defrag_prefix.py)
        self._fences: list = []
        self._submitted_count = 0
        self._destroyed = False
        print(f"[SimSingle] init complete on {ctx.device_name} (own pool {case.capacities.own_pool_size:,}, "
              f"{case.initial.positions.shape[0]:,} particles, {case.grid.total_voxel_count():,} voxels)")

    # ------------------------------------------------------------------ properties

    @property
    def wall_adami(self) -> bool:
        return self.case.numerics.wall_boundary == "adami"

    @property
    def deep_wall_active(self) -> bool:
        return self.deep_wall_skip_resolution[0]

    @property
    def fused_active(self) -> bool:
        return self.fused_correction_density_resolution[0]

    def own_first_pid(self) -> int:
        return 1

    def own_last_pid(self) -> int:
        return self.case.capacities.own_pool_size

    def stored_density_offset(self) -> float:
        return 0.0

    def reference_density(self) -> float:
        fluids = [material for material in self.case.materials if material.kind == KIND_FLUID]
        chosen = fluids[0] if fluids else self.case.materials[0]
        return float(chosen.rest_density)

    def resolution_record(self) -> dict:
        """Run metadata: the switch table and both per-slab resolutions."""
        return {"switches": self.switches.table(),
                "deep_wall_skip": list(self.deep_wall_skip_resolution),
                "fused_correction_density": list(self.fused_correction_density_resolution),
                "wall_boundary": self.wall_boundary, "band_widths": list(BAND_WIDTHS),
                "spirv_directory": str(self.spirv_directory)}

    # ------------------------------------------------------------------ checks

    def _check_degenerate_case(self) -> None:
        case = self.case
        if (case.ghost_grid.leading_ghost_voxel_count or case.ghost_grid.trailing_ghost_voxel_count
                or case.capacities.leading_ghost_pool_size or case.capacities.trailing_ghost_pool_size
                or case.capacities.departed_pool_size or case.capacities.replica_region_size
                or case.transport.has_leading_peer or case.transport.has_trailing_peer):
            raise ValueError("SphSimulatorSingle takes the loader's degenerate case (no peer, no ghost columns, "
                             "no ghost / departed pool); this case is a chain slab")
        if int(case.initial.positions.shape[0]) > case.capacities.own_pool_size:
            raise ValueError(f"{case.initial.positions.shape[0]:,} particles > own pool {case.capacities.own_pool_size:,}")

    def _check_workgroup_limit(self) -> None:
        properties = vkGetPhysicalDeviceProperties(self.ctx.physical_device)
        maximum = properties.limits.maxComputeWorkGroupSize[0]
        workgroup = self.case.capacities.workgroup_size
        if workgroup > maximum:
            raise RuntimeError(f"WORKGROUP_SIZE {workgroup} > device limit {maximum} on {self.ctx.device_name}")

    # ------------------------------------------------------------------ buffers

    def _build_buffer_specs(self) -> list[_BufferSpec]:
        """The set / binding layout of v7 (inventory 3): the same bindings so the same SPIR-V binds the same
        descriptor set layouts; the ghost-only buffers are stubs."""
        case = self.case
        pool_capacity = case.capacities.total_pool_capacity()
        voxel_capacity = 1 + case.grid.total_voxel_count()
        cap_inside = case.capacities.max_particles_per_voxel
        cap_incoming = case.capacities.max_incoming_per_voxel
        material_count = max(len(case.materials), 1)
        storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        transfer = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
        vertex = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
        specs = [
            _BufferSpec("position_voxel_id",            0, 0, 16 * pool_capacity, storage | transfer | vertex),
            _BufferSpec("density_pressure",             0, 1,  8 * pool_capacity, storage | transfer | vertex),
            _BufferSpec("density_pressure_scratch",     0, 2,  8 * pool_capacity, storage | transfer | vertex),
            _BufferSpec("velocity_mass",                0, 3, 16 * pool_capacity, storage | transfer | vertex),
            _BufferSpec("acceleration",                 0, 4, 16 * pool_capacity, storage | transfer),
            _BufferSpec("shift",                        0, 5, 16 * pool_capacity, storage | transfer),
            _BufferSpec("material",                     0, 6,  4 * pool_capacity, storage | transfer),
            _BufferSpec("correction_inverse",           0, 7, 32 * pool_capacity, storage | transfer),
            _BufferSpec("density_gradient_kernel_sum",  0, 8, 16 * pool_capacity, storage | transfer),
            _BufferSpec("extension_fields",             0, 9, 16 * pool_capacity, storage | transfer),
            _BufferSpec("wall_dummy_velocity",          0, 10, 16 * pool_capacity if self.wall_adami else 16,
                        storage | transfer),
            _BufferSpec("inside_particle_count",        1, 0, 4 * voxel_capacity,                storage | transfer),
            _BufferSpec("incoming_particle_count",      1, 1, 4 * voxel_capacity,                storage | transfer),
            _BufferSpec("inside_particle_index",        1, 2, 4 * voxel_capacity * cap_inside,   storage | transfer),
            _BufferSpec("incoming_particle_index",      1, 3, 4 * voxel_capacity * cap_incoming, storage | transfer),
            _BufferSpec("voxel_base_offset",            1, 4, 4 * voxel_capacity,                storage | transfer),
            # ghost-only buffers (ghost_send / expand_ghost_lists / the compact band dispatch): stubs
            _BufferSpec("ghost_voxel_first_particle_id", 1, 5, 4,                                storage | transfer),
            _BufferSpec("band_compact_list",            1, 6, 4,                                storage | transfer),
            _BufferSpec("ghost_packed_words",           1, 7, 4,                                storage | transfer),
            _BufferSpec("global_status",                3, 0, GLOBAL_STATUS_BYTES,               storage | transfer),
            _BufferSpec("pool_health",                  3, 1, 16,                               storage | transfer),
            _BufferSpec("material_parameters",          3, 7, 48 * material_count,              storage | transfer),
            _BufferSpec("defrag_scratch_counter",       3, 8, 4,                                storage | transfer),
            _BufferSpec("band_compact_meta",            3, 9, 16,
                        storage | transfer | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT),
        ]
        if self.deep_wall_active:
            specs += [
                _BufferSpec("deep_wall_voxel_flag",     1, 8, 4 * 2 * voxel_capacity, storage | transfer),
                _BufferSpec("deep_wall_skip_record",    3, 10, 8 * pool_capacity if self.deep_wall_record else 16,
                            storage | transfer),
            ]
        return specs

    def _allocate_buffer(self, size: int, usage: int, required_properties: int,
                         preferred_properties: int = 0) -> _Buffer:
        """One EXCLUSIVE buffer with its own allocation (v7's _allocate_buffer without the CONCURRENT branch)."""
        if size <= 0:
            raise ValueError("buffer size must be > 0")
        handle = vkCreateBuffer(self.ctx.device, VkBufferCreateInfo(
            size=size, usage=usage, sharingMode=VK_SHARING_MODE_EXCLUSIVE), None)
        requirements = vkGetBufferMemoryRequirements(self.ctx.device, handle)
        type_index = self.ctx.find_memory_type(requirements.memoryTypeBits, required_properties, preferred_properties)
        memory = vkAllocateMemory(self.ctx.device, VkMemoryAllocateInfo(
            allocationSize=requirements.size, memoryTypeIndex=type_index), None)
        vkBindBufferMemory(self.ctx.device, handle, memory, 0)
        return _Buffer(handle=handle, memory=memory, size=size)

    def _allocate_buffers(self) -> dict[str, _Buffer]:
        buffers = {}
        total = 0
        for spec in self._buffer_specs:
            buffers[spec.name] = self._allocate_buffer(spec.size, spec.usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            total += spec.size
        print(f"[SimSingle] device-local buffers: {len(buffers)}, {total / (1024 * 1024):.2f} MB, all EXCLUSIVE")
        return buffers

    def _allocate_scratch_buffers(self) -> dict[str, _Buffer]:
        scratch = {}
        total = 0
        for spec in self._buffer_specs:
            if spec.set_index == 0 and spec.binding in DEFRAG_SET0_BINDINGS:
                scratch[spec.name] = self._allocate_buffer(
                    spec.size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                    | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                total += spec.size
        print(f"[SimSingle] defrag scratch buffers: {len(scratch)}, {total / (1024 * 1024):.2f} MB")
        return scratch

    def buffer_bytes(self) -> dict[str, int]:
        """{name: bytes} of the device-local buffers, scratch twins as 'scratch:<name>' (run metadata)."""
        sizes = {spec.name: spec.size for spec in self._buffer_specs}
        sizes.update({f"scratch:{name}": buffer.size for name, buffer in self.scratch_buffers.items()})
        return sizes

    # ------------------------------------------------------------------ descriptors + pipelines

    def _build_descriptor_layouts(self) -> list:
        layouts = []
        for set_index in range(4):
            bindings = [VkDescriptorSetLayoutBinding(binding=spec.binding,
                                                     descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                     descriptorCount=1, stageFlags=VK_SHADER_STAGE_COMPUTE_BIT)
                        for spec in self._buffer_specs if spec.set_index == set_index]
            layouts.append(vkCreateDescriptorSetLayout(self.ctx.device, VkDescriptorSetLayoutCreateInfo(
                bindingCount=len(bindings), pBindings=bindings if bindings else None), None))
        return layouts

    def _create_descriptor_pool(self):
        pool_size = VkDescriptorPoolSize(type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                         descriptorCount=len(self._buffer_specs))
        return vkCreateDescriptorPool(self.ctx.device, VkDescriptorPoolCreateInfo(
            maxSets=4, poolSizeCount=1, pPoolSizes=[pool_size]), None)

    def _allocate_descriptor_sets(self) -> list:
        return vkAllocateDescriptorSets(self.ctx.device, VkDescriptorSetAllocateInfo(
            descriptorPool=self.descriptor_pool, descriptorSetCount=4, pSetLayouts=self.descriptor_layouts))

    def _wire_descriptor_sets(self) -> None:
        writes = []
        for spec in self._buffer_specs:
            writes.append(VkWriteDescriptorSet(
                dstSet=self.descriptor_sets[spec.set_index], dstBinding=spec.binding, dstArrayElement=0,
                descriptorCount=1, descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[VkDescriptorBufferInfo(buffer=self.buffers[spec.name].handle, offset=0,
                                                    range=spec.size)]))
        vkUpdateDescriptorSets(self.ctx.device, len(writes), writes, 0, None)

    def _build_pipeline_layout(self):
        return vkCreatePipelineLayout(self.ctx.device, VkPipelineLayoutCreateInfo(
            setLayoutCount=4, pSetLayouts=self.descriptor_layouts, pushConstantRangeCount=0), None)

    def shader_names(self) -> tuple[str, ...]:
        """The SPIR-V files this build loads (inventory 5)."""
        names = ["initialize_voxelization", "bootstrap_half_kick", "predict", "update_voxel", "force", "defrag"]
        if self.fused_active:
            names.append("correction_density")
            if self.deep_wall_active:
                names.append("correction_density_deep_wall_skip")
        else:
            names += ["correction", "density"]
            if self.deep_wall_active:
                names += ["correction_deep_wall_skip", "density_deep_wall_skip"]
        if self.deep_wall_active:
            names.append("deep_wall_marker")
        if self.switches.density_copy_compute:
            names.append("density_scratch_copy")
        if self.wall_adami:
            names.append("wall_extrapolate")
        return tuple(names)

    def _load_shader_modules(self) -> dict[str, object]:
        modules = {}
        for name in self.shader_names():
            code = spirv_manifest.load_spirv(name, self.spirv_directory)
            modules[name] = vkCreateShaderModule(self.ctx.device, VkShaderModuleCreateInfo(
                codeSize=len(code), pCode=code), None)
        print(f"[SimSingle] SPIR-V: {len(modules)} modules from {self.spirv_directory} (manifest verified)")
        return modules

    def _pack_spec(self, entries: list[tuple[int, str, Any]]) -> tuple[bytes, list]:
        blob = bytearray()
        map_entries = []
        for constant_id, format_code, value in entries:
            packed = struct.pack("<I", 1 if value else 0) if format_code == "B" else struct.pack("<" + format_code, value)
            assert len(packed) == 4
            map_entries.append(VkSpecializationMapEntry(constantID=constant_id, offset=len(blob), size=4))
            blob.extend(packed)
        return bytes(blob), map_entries

    def _make_spec_info(self, entries: list[tuple[int, str, Any]]):
        if not entries:
            return None
        blob, map_entries = self._pack_spec(entries)
        cdata = ffi.new("uint8_t[]", blob)
        info = VkSpecializationInfo(mapEntryCount=len(map_entries), pMapEntries=map_entries,
                                    dataSize=len(blob), pData=cdata)
        self._spec_keepalive.append({"blob": blob, "cdata": cdata, "entries": map_entries, "info": info})
        return info

    def global_entries(self) -> list[tuple[int, str, Any]]:
        """The specialization constants every pipeline gets: v7's _global_entries of a degenerate slab under
        v7's default switch environment (inventory 6; the transport-related ids are the constants v7 passes
        in its single-GPU mode, inert without ghost columns)."""
        physics, numerics, capacities, grid = self.case.physics, self.case.numerics, self.case.capacities, self.case.grid
        return [
            (0, "f", physics.smoothing_length), (1, "f", physics.speed_of_sound), (2, "f", physics.delta_coefficient),
            (4, "f", physics.power_parameter), (5, "f", physics.cfl_number), (6, "f", physics.timestep),
            (7, "f", grid.origin_x), (8, "f", grid.origin_y), (9, "f", grid.origin_z),
            (10, "B", 0),
            (11, "I", grid.grid_dimension_x), (12, "I", grid.grid_dimension_y), (13, "I", grid.grid_dimension_z),
            (14, "f", numerics.regularization_xi), (15, "f", numerics.regularization_determinant_threshold),
            (16, "f", numerics.regularization_max_frobenius_norm),
            (17, "f", physics.gravity[0]), (18, "f", physics.gravity[1]), (19, "f", physics.gravity[2]),
            (20, "I", grid.voxel_order),
            (30, "I", physics.dimension), (31, "I", physics.neighbor_z_range),
            (32, "f", physics.kernel_coefficient), (33, "f", physics.kernel_gradient_coefficient),
            (40, "f", numerics.eps_h_squared), (41, "f", numerics.pst_main_shift_coefficient),
            (42, "f", numerics.pst_anti_shift_coefficient),
            (43, "B", int(numerics.use_kcg_correction)), (44, "B", int(numerics.use_density_diffusion)),
            (45, "B", int(numerics.use_pst)), (46, "B", int(numerics.use_prefix_sum_defrag)),
            (50, "I", capacities.max_particles_per_voxel), (51, "I", capacities.workgroup_size),
            (52, "I", capacities.max_incoming_per_voxel), (53, "I", capacities.own_pool_size),
            (54, "I", 0), (55, "I", 0),
            (80, "I", 0), (81, "I", 0),
            (83, "I", 1),               # GHOST_LAYERS: the loader's GhostGridParams default
            (84, "I", 0), (86, "I", 0),
            (87, "B", 1),               # LEAN_TRANSPORT
            (88, "B", 0),               # TRANSPORT_EXTENSION
            (89, "B", 1),               # COMPACT_GHOST_LISTS
            (95, "B", 0),               # DELTA_DENSITY
            (96, "f", self.reference_density()),
            (98, "B", 1),               # PACKED_REPLICAS (v7's unset default resolves to 1)
            (99, "I", 0),               # DIAG_POISON_G1
            (97, "B", 1),               # INIT_SEAM_CLAMP
            (100, "I", 1 if self.wall_adami else 0),
        ]

    def correction_entries(self) -> list[tuple[int, str, Any]]:
        return [(47, "I", 0), (82, "I", BAND_WIDTHS[0]), (57, "I", 0), (58, "I", 0), (59, "I", 0), (85, "I", 0)]

    def density_entries(self) -> list[tuple[int, str, Any]]:
        return [(48, "I", 0), (82, "I", BAND_WIDTHS[1]), (57, "I", 0), (58, "I", 0), (59, "I", 0), (85, "I", 0)]

    def force_entries(self) -> list[tuple[int, str, Any]]:
        return [(49, "I", 0), (82, "I", BAND_WIDTHS[2]), (56, "I", 0), (57, "I", 0), (58, "I", 0), (59, "I", 0)]

    def deep_wall_skip_entries(self) -> list[tuple[int, str, Any]]:
        return [(111, "I", DEEP_WALL_SKIP_BAND_WIDTH), (112, "B", 0), (113, "B", self.deep_wall_record)]

    def density_copy_slot_regions(self) -> list[tuple[int, int]]:
        """(first slot, slot count) of the scratch -> primary copy: the own range only."""
        return [(self.own_first_pid(), self.case.capacities.own_pool_size)]

    def density_copy_entries(self) -> list[tuple[int, str, Any]]:
        regions = self.density_copy_slot_regions() + [(0, 0)] * (DENSITY_COPY_REGION_LIMIT - 1)
        entries = []
        for region_index, (first_slot, slot_count) in enumerate(regions):
            entries += [(101 + 2 * region_index, "I", first_slot), (102 + 2 * region_index, "I", slot_count)]
        return entries

    def density_copy_group_count(self) -> int:
        slots = sum(count for _, count in self.density_copy_slot_regions())
        return (slots + DENSITY_COPY_LOCAL_SIZE - 1) // DENSITY_COPY_LOCAL_SIZE

    def pipeline_specs(self) -> dict[str, tuple[str, list]]:
        """{pipeline key: (shader name, specialization entries)} of every pipeline this build creates, defrag
        excluded (its own layout, global entries only): the CPU comparison tool reads this table."""
        global_entries = self.global_entries()
        specs: dict[str, tuple[str, list]] = {}
        for key in ("initialize_voxelization", "bootstrap_half_kick", "predict", "update_voxel"):
            specs[key] = (key, list(global_entries))
        skip = self.deep_wall_active
        if self.fused_active:
            specs["correction_density_all"] = (
                "correction_density_deep_wall_skip" if skip else "correction_density",
                global_entries + self.correction_entries() + (self.deep_wall_skip_entries() if skip else []))
        else:
            specs["correction_all"] = ("correction_deep_wall_skip" if skip else "correction",
                                       global_entries + self.correction_entries()
                                       + (self.deep_wall_skip_entries() if skip else []))
            specs["density_all"] = ("density_deep_wall_skip" if skip else "density",
                                    global_entries + self.density_entries()
                                    + (self.deep_wall_skip_entries() if skip else []))
        if skip:
            specs["deep_wall_marker_presence"] = ("deep_wall_marker", global_entries + [(114, "I", 0)])
            specs["deep_wall_marker_deep"] = ("deep_wall_marker", global_entries + [(114, "I", 1)])
        specs["force_all"] = ("force", global_entries + self.force_entries())
        if self.wall_adami:
            specs["wall_extrapolate"] = ("wall_extrapolate", global_entries + [(56, "I", 0)])
        if self.switches.density_copy_compute:
            specs["density_scratch_copy"] = ("density_scratch_copy", self.density_copy_entries())
        return specs

    def _create_pipeline(self, shader_name: str, entries: list[tuple[int, str, Any]], layout=None):
        stage = VkPipelineShaderStageCreateInfo(stage=VK_SHADER_STAGE_COMPUTE_BIT,
                                                module=self.shader_modules[shader_name], pName="main",
                                                pSpecializationInfo=self._make_spec_info(entries))
        create_info = VkComputePipelineCreateInfo(stage=stage, layout=layout or self.pipeline_layout)
        return vkCreateComputePipelines(self.ctx.device, VK_NULL_HANDLE, 1, [create_info], None)[0]

    def _build_compute_pipelines(self) -> dict[str, object]:
        pipelines = {key: self._create_pipeline(shader_name, entries)
                     for key, (shader_name, entries) in self.pipeline_specs().items()}
        print(f"[SimSingle] compute pipelines: {len(pipelines)} ({', '.join(pipelines)})")
        return pipelines

    def _build_defrag_set4_layout(self):
        bindings = [VkDescriptorSetLayoutBinding(binding=binding, descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                 descriptorCount=1, stageFlags=VK_SHADER_STAGE_COMPUTE_BIT)
                    for binding in DEFRAG_SET0_BINDINGS]
        return vkCreateDescriptorSetLayout(self.ctx.device, VkDescriptorSetLayoutCreateInfo(
            bindingCount=len(bindings), pBindings=bindings), None)

    def _allocate_defrag_set4(self):
        pool = vkCreateDescriptorPool(self.ctx.device, VkDescriptorPoolCreateInfo(
            maxSets=1, poolSizeCount=1,
            pPoolSizes=[VkDescriptorPoolSize(type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                             descriptorCount=len(DEFRAG_SET0_BINDINGS))]), None)
        descriptor_set = vkAllocateDescriptorSets(self.ctx.device, VkDescriptorSetAllocateInfo(
            descriptorPool=pool, descriptorSetCount=1, pSetLayouts=[self.defrag_set4_layout]))[0]
        return pool, descriptor_set

    def _wire_defrag_set4(self) -> None:
        writes = []
        for spec in self._buffer_specs:
            if spec.set_index != 0 or spec.binding not in DEFRAG_SET0_BINDINGS:
                continue
            scratch = self.scratch_buffers[spec.name]
            writes.append(VkWriteDescriptorSet(
                dstSet=self.defrag_set4, dstBinding=spec.binding, dstArrayElement=0, descriptorCount=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[VkDescriptorBufferInfo(buffer=scratch.handle, offset=0, range=scratch.size)]))
        vkUpdateDescriptorSets(self.ctx.device, len(writes), writes, 0, None)

    def _build_defrag_pipeline_layout(self):
        return vkCreatePipelineLayout(self.ctx.device, VkPipelineLayoutCreateInfo(
            setLayoutCount=5, pSetLayouts=list(self.descriptor_layouts) + [self.defrag_set4_layout],
            pushConstantRangeCount=0), None)

    def _build_defrag_pipeline(self):
        return self._create_pipeline("defrag", self.global_entries(), layout=self.defrag_pipeline_layout)

    # ------------------------------------------------------------------ recording helpers

    def _allocate_oneshot_cmd(self):
        return vkAllocateCommandBuffers(self.ctx.device, VkCommandBufferAllocateInfo(
            commandPool=self.ctx.command_pool, level=VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1))[0]

    def _free_cmd(self, cmd) -> None:
        vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])

    def per_particle_dispatch_count(self) -> int:
        workgroup = self.case.capacities.workgroup_size
        return (self.case.capacities.own_pool_size + workgroup - 1) // workgroup

    def per_voxel_dispatch_count(self) -> int:
        workgroup = self.case.capacities.workgroup_size
        return (self.case.grid.total_voxel_count() + workgroup - 1) // workgroup

    @staticmethod
    def _memory_barrier(cmd, source_stage, source_access, destination_stage, destination_access) -> None:
        barrier = VkMemoryBarrier2(sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                                   srcStageMask=source_stage, srcAccessMask=source_access,
                                   dstStageMask=destination_stage, dstAccessMask=destination_access)
        vkCmdPipelineBarrier2(cmd, VkDependencyInfo(sType=VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                                    memoryBarrierCount=1, pMemoryBarriers=[barrier]))

    def _record_compute_barrier(self, cmd) -> None:
        storage = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
        self._memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, storage,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, storage)

    def _record_transfer_to_compute_barrier(self, cmd) -> None:
        self._memory_barrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT)

    def _record_compute_to_transfer_barrier(self, cmd) -> None:
        self._memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT)

    def _bind(self, cmd, pipeline_key: str) -> None:
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, self.pipelines[pipeline_key])
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout, 0, 4,
                                self.descriptor_sets, 0, None)

    def _record_deep_wall_marker(self, cmd) -> None:
        """deep_wall_marker.comp's two passes over the voxels, each followed by a compute barrier (B4)."""
        if not self.deep_wall_active:
            return
        per_voxel = self.per_voxel_dispatch_count()
        for key in ("deep_wall_marker_presence", "deep_wall_marker_deep"):
            self._bind(cmd, key)
            vkCmdDispatch(cmd, per_voxel, 1, 1)
            self._record_compute_barrier(cmd)

    def _record_density_copy(self, cmd) -> None:
        """The scratch -> primary density copy: B9's compute pass between two compute barriers, or (switch 0)
        v7's vkCmdCopyBuffer between compute -> transfer and transfer -> compute barriers."""
        if self.switches.density_copy_compute:
            self._record_compute_barrier(cmd)
            self._bind(cmd, "density_scratch_copy")
            vkCmdDispatch(cmd, self.density_copy_group_count(), 1, 1)
            self._record_compute_barrier(cmd)
            return
        stride = 8
        first, count = self.density_copy_slot_regions()[0]
        region = VkBufferCopy(srcOffset=first * stride, dstOffset=first * stride, size=count * stride)
        self._record_compute_to_transfer_barrier(cmd)
        vkCmdCopyBuffer(cmd, self.buffers["density_pressure_scratch"].handle,
                        self.buffers["density_pressure"].handle, 1, [region])
        self._record_transfer_to_compute_barrier(cmd)

    def _record_wall_extrapolate(self, cmd) -> None:
        """E37 adami: the wall pass over the own range (reading the fluid's rho / P from primary), then a barrier."""
        if not self.wall_adami:
            return
        self._bind(cmd, "wall_extrapolate")
        vkCmdDispatch(cmd, self.per_particle_dispatch_count(), 1, 1)
        self._record_compute_barrier(cmd)

    # ------------------------------------------------------------------ the recordings (inventory 7)

    def record_step(self, cmd, tick: Optional[Callable[[Any, str], None]] = None) -> None:
        """The time step (v7's _record_step_single_cmd, command for command): predict -> update_voxel ->
        [deep-wall marker] -> correction_density_all (or correction_all -> density_all) -> density copy ->
        [wall_extrapolate] -> force_all. ``tick(cmd, label)`` is called at the stage boundaries (tick_labels)
        when given — the kernel breakdown's timestamps inside the production recording; None records nothing."""
        def mark(label: str) -> None:
            if tick is not None:
                tick(cmd, label)
        per_particle = self.per_particle_dispatch_count()
        self._record_compute_barrier(cmd)
        mark("step_start")
        self._bind(cmd, "predict")
        vkCmdDispatch(cmd, per_particle, 1, 1)
        mark("predict_end")
        self._record_compute_barrier(cmd)
        self._bind(cmd, "update_voxel")
        vkCmdDispatch(cmd, self.per_voxel_dispatch_count(), 1, 1)
        mark("voxel_end")
        self._record_compute_barrier(cmd)
        if self.deep_wall_active:
            self._record_deep_wall_marker(cmd)
            mark("deep_wall_marker_end")
        if self.fused_active:
            self._bind(cmd, "correction_density_all")
            vkCmdDispatch(cmd, per_particle, 1, 1)
            mark("correction_density_end")
        else:
            self._bind(cmd, "correction_all")
            vkCmdDispatch(cmd, per_particle, 1, 1)
            mark("correction_end")
            self._record_compute_barrier(cmd)
            self._bind(cmd, "density_all")
            vkCmdDispatch(cmd, per_particle, 1, 1)
            mark("density_kernel_end")
        self._record_density_copy(cmd)
        mark("density_end")
        self._record_compute_barrier(cmd)          # v7 records this second barrier after the copy too
        if self.wall_adami:
            # the wall pass inline so the tick sits before its barrier, where v7's BenchTimer puts it
            self._bind(cmd, "wall_extrapolate")
            vkCmdDispatch(cmd, per_particle, 1, 1)
            mark("wall_extrapolate_end")
            self._record_compute_barrier(cmd)
        self._bind(cmd, "force_all")
        vkCmdDispatch(cmd, per_particle, 1, 1)
        mark("force_end")

    def record_bootstrap_init(self, cmd) -> None:
        """Bootstrap stage 1 (v7's _record_bootstrap_init_cmd without the per-direction ghost round, which
        records nothing on a slab without peers): the initial voxelization."""
        self._record_compute_barrier(cmd)
        self._bind(cmd, "initialize_voxelization")
        vkCmdDispatch(cmd, self.per_particle_dispatch_count(), 1, 1)

    def record_bootstrap_compute(self, cmd) -> None:
        """Bootstrap stage 2 (v7's _record_bootstrap_compute_cmd on a slab without peers): [adami wall pass]
        [deep-wall marker] correction + density (fused or separate) -> density copy -> [wall pass] -> force
        -> bootstrap_half_kick."""
        per_particle = self.per_particle_dispatch_count()
        if self.wall_adami:
            self._record_compute_barrier(cmd)
            self._record_wall_extrapolate(cmd)
        if self.deep_wall_active:
            self._record_compute_barrier(cmd)
            self._record_deep_wall_marker(cmd)
        if self.fused_active:
            self._bind(cmd, "correction_density_all")
            vkCmdDispatch(cmd, per_particle, 1, 1)
        else:
            self._bind(cmd, "correction_all")
            vkCmdDispatch(cmd, per_particle, 1, 1)
            self._record_compute_barrier(cmd)
            self._bind(cmd, "density_all")
            vkCmdDispatch(cmd, per_particle, 1, 1)
        self._record_density_copy(cmd)
        self._record_wall_extrapolate(cmd)
        self._bind(cmd, "force_all")
        vkCmdDispatch(cmd, per_particle, 1, 1)
        self._record_compute_barrier(cmd)
        self._bind(cmd, "bootstrap_half_kick")
        vkCmdDispatch(cmd, per_particle, 1, 1)

    def record_restart_init(self, cmd) -> None:
        """The restart's voxelization (v7's restart_init command): voxel lists from the uploaded state, then
        with adami the wall pass once (the dummy velocity is not part of the state)."""
        self._record_compute_barrier(cmd)
        self._bind(cmd, "initialize_voxelization")
        vkCmdDispatch(cmd, self.per_particle_dispatch_count(), 1, 1)
        self._record_compute_barrier(cmd)
        self._record_wall_extrapolate(cmd)

    def record_defrag(self, cmd, tick: Optional[Callable[[Any, str], None]] = None) -> None:
        """The defrag (v7's _record_defrag_cmd without the migration_install_count fill, inventory 7): zero
        the counter and the nine scratch twins, defrag over the voxels, copy the twins back, copy the counter
        into global_status.alive_particle_count."""
        self._record_compute_barrier(cmd)
        counter = self.buffers["defrag_scratch_counter"]
        if not self.case.numerics.use_prefix_sum_defrag:     # prefix order: the host wrote the total (E9)
            vkCmdFillBuffer(cmd, counter.handle, 0, counter.size, 0)
        twins = [(spec, self.scratch_buffers[spec.name]) for spec in self._buffer_specs
                 if spec.set_index == 0 and spec.binding in DEFRAG_SET0_BINDINGS]
        for _spec, scratch in twins:
            vkCmdFillBuffer(cmd, scratch.handle, 0, scratch.size, 0)
        self._record_transfer_to_compute_barrier(cmd)
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, self.pipelines["defrag"])
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, self.defrag_pipeline_layout, 0, 5,
                                list(self.descriptor_sets) + [self.defrag_set4], 0, None)
        vkCmdDispatch(cmd, self.per_voxel_dispatch_count(), 1, 1)
        self._record_compute_to_transfer_barrier(cmd)
        for spec, scratch in twins:
            vkCmdCopyBuffer(cmd, scratch.handle, self.buffers[spec.name].handle, 1,
                            [VkBufferCopy(srcOffset=0, dstOffset=0, size=spec.size)])
        vkCmdCopyBuffer(cmd, counter.handle, self.buffers["global_status"].handle, 1,
                        [VkBufferCopy(srcOffset=0, dstOffset=0, size=4)])
        self._record_transfer_to_compute_barrier(cmd)
        if tick is not None:
            tick(cmd, "defrag_end")

    # ------------------------------------------------------------------ uploads, bootstrap, restart

    def _build_initial_data(self) -> dict[str, bytes]:
        """Payloads keyed by buffer name (v7's _build_initial_data: the same bytes); buffers not listed are
        zeroed. extension_fields is not uploaded here (canonical_dump adds the global ids)."""
        case = self.case
        pool_capacity = case.capacities.total_pool_capacity()
        own_first = self.own_first_pid()
        count = int(case.initial.positions.shape[0])
        groups = np.asarray(case.initial.material_group, dtype=np.int64)
        data = {}
        position_voxel_id = np.zeros((pool_capacity, 4), dtype=np.float32)
        position_voxel_id[own_first:own_first + count, 0:3] = case.initial.positions
        data["position_voxel_id"] = position_voxel_id.tobytes()
        velocity_mass = np.zeros((pool_capacity, 4), dtype=np.float32)
        velocity_mass[own_first:own_first + count, 0:3] = case.initial.velocities
        masses = np.array([material.rest_density * material.volume for material in case.materials], dtype=np.float64)
        velocity_mass[own_first:own_first + count, 3] = masses[groups].astype(np.float32)
        data["velocity_mass"] = velocity_mass.tobytes()
        density_pressure = np.zeros((pool_capacity, 2), dtype=np.float32)
        rest = np.array([material.rest_density for material in case.materials], dtype=np.float64)
        density_pressure[own_first:own_first + count, 0] = rest[groups].astype(np.float32)
        data["density_pressure"] = density_pressure.tobytes()
        material = np.zeros(pool_capacity, dtype=np.uint32)
        if count:
            material[own_first:own_first + count] = case.initial.material_group
        data["material"] = material.tobytes()
        data["material_parameters"] = self._material_parameters_payload()
        return data

    def _material_parameters_payload(self) -> bytes:
        blob = bytearray()
        for material in self.case.materials:
            row = struct.pack("<I f f f f f f f f f f I", material.kind, material.rest_density, material.viscosity,
                              material.eos_constant, material.smoothing_length, material.radius, material.volume,
                              material.rotor_angular_velocity, material.viscosity_transfer,
                              material.viscosity_rotation, material.rest_density * material.volume,
                              material.reserved_material_1)
            assert len(row) == 48
            blob.extend(row)
        return bytes(blob) if blob else b"\x00" * 48

    def _staging_upload(self, destination: _Buffer, payload: bytes) -> None:
        if len(payload) > destination.size:
            raise ValueError(f"payload {len(payload)} > buffer {destination.size}")
        staging = self._allocate_buffer(len(payload), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        try:
            mapped = vkMapMemory(self.ctx.device, staging.memory, 0, len(payload), 0)
            np.frombuffer(mapped, dtype=np.uint8, count=len(payload))[:] = np.frombuffer(payload, dtype=np.uint8)
            vkUnmapMemory(self.ctx.device, staging.memory)
            cmd = self._allocate_oneshot_cmd()
            vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
            vkCmdCopyBuffer(cmd, staging.handle, destination.handle, 1,
                            [VkBufferCopy(srcOffset=0, dstOffset=0, size=len(payload))])
            vkEndCommandBuffer(cmd)
            self.ctx.submit_and_wait(cmd)
            self._free_cmd(cmd)
        finally:
            vkDestroyBuffer(self.ctx.device, staging.handle, None)
            vkFreeMemory(self.ctx.device, staging.memory, None)

    def _zero_buffer(self, destination: _Buffer) -> None:
        cmd = self._allocate_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
        vkCmdFillBuffer(cmd, destination.handle, 0, destination.size, 0)
        vkEndCommandBuffer(cmd)
        self.ctx.submit_and_wait(cmd)
        self._free_cmd(cmd)

    def _upload_initial_state(self, payload: Optional[dict[str, bytes]] = None) -> None:
        if payload is None:
            payload = self._build_initial_data()
        for name, buffer in self.buffers.items():
            if name in payload:
                data = payload[name]
                if len(data) < buffer.size:
                    data = data + b"\x00" * (buffer.size - len(data))
                self._staging_upload(buffer, data)
            else:
                self._zero_buffer(buffer)
        for buffer in self.scratch_buffers.values():
            self._zero_buffer(buffer)
        print(f"[SimSingle] uploaded initial state ({len(payload)} payload buffers)")

    def _submit_recording(self, record) -> None:
        cmd = self._allocate_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
        record(cmd)
        vkEndCommandBuffer(cmd)
        self.ctx.submit_and_wait(cmd)
        self._free_cmd(cmd)

    def bootstrap_init(self) -> None:
        """Upload the initial state and voxelize it (one-shot submit)."""
        self._upload_initial_state()
        self._submit_recording(self.record_bootstrap_init)

    def bootstrap_compute(self) -> None:
        """The bootstrap's correction / density / force / half kick (one-shot submit), then the status print."""
        self._submit_recording(self.record_bootstrap_compute)
        status = self.readback_global_status()
        print(f"[SimSingle] bootstrap done: alive={status['alive_particle_count']} "
              f"overflow_inside={status['overflow_inside_count']} overflow_incoming={status['overflow_incoming_count']}")

    def bootstrap(self) -> None:
        self.bootstrap_init()
        self.bootstrap_compute()

    def restart_init(self, state: dict, stored_density_offset: float = 0.0) -> None:
        """Start from a saved step-boundary state (v7's restart_init): the nine fields of RESTART_FIELD_LAYOUT,
        one row per particle, go to pids 1 + k; every other buffer is zeroed; initialize_voxelization rebuilds
        the lists. No bootstrap kernels, no half kick: the next step is the step after the saved one."""
        layout = self.RESTART_FIELD_LAYOUT
        missing = [name for name in layout if name not in state]
        if missing:
            raise ValueError(f"restart state lacks {missing}")
        row_count = int(np.asarray(state["position_voxel_id"]).shape[0])
        own_pool = self.case.capacities.own_pool_size
        if row_count > own_pool:
            raise ValueError(f"restart state has {row_count:,} particles for an own pool of {own_pool:,}")
        pool_capacity = self.case.capacities.total_pool_capacity()
        own_first = self.own_first_pid()
        payload = {}
        for name, (component_count, element_type) in layout.items():
            rows = np.asarray(state[name])
            expected_shape = (row_count,) if component_count == 1 else (row_count, component_count)
            if rows.shape != expected_shape or rows.dtype != element_type:
                raise ValueError(f"restart field {name}: {rows.dtype} {rows.shape}, expected "
                                 f"{np.dtype(element_type)} {expected_shape}")
            full = np.zeros((pool_capacity,) + expected_shape[1:], dtype=element_type)
            full[own_first:own_first + row_count] = rows
            shift = float(stored_density_offset) - self.stored_density_offset()
            if name == "density_pressure" and shift != 0.0:
                rows64 = full[own_first:own_first + row_count, 0].astype(np.float64) + shift
                full[own_first:own_first + row_count, 0] = rows64.astype(np.float32)
            payload[name] = full.tobytes()
        if row_count and float(np.asarray(state["velocity_mass"])[:, 3].min()) <= 0.0:
            raise ValueError("restart state has rows with mass <= 0 (initialize_voxelization would skip them)")
        payload["material_parameters"] = self._material_parameters_payload()
        self._upload_initial_state(payload)
        self._submit_recording(self.record_restart_init)
        status = self.readback_global_status()
        positions = np.frombuffer(self.readback_buffer_by_name("position_voxel_id"),
                                  dtype=np.float32)[:pool_capacity * 4].reshape(pool_capacity, 4)
        voxel_ids = np.rint(positions[own_first:own_first + row_count, 3]).astype(np.int64)
        outside = int(((voxel_ids < 1) | (voxel_ids > self.case.grid.total_voxel_count())).sum())
        if status["alive_particle_count"] != row_count or status["overflow_inside_count"] or outside:
            raise RuntimeError(f"restart: {status['alive_particle_count']:,} of {row_count:,} particles placed, "
                               f"overflow_inside={status['overflow_inside_count']}, {outside} outside the grid")
        print(f"[SimSingle] restart state loaded: {row_count:,} particles, voxel lists rebuilt")

    # ------------------------------------------------------------------ the step loop

    def prepare_step(self, in_flight: int = 1) -> None:
        """Record the step and defrag command buffers (once) and create the ring of ``in_flight`` fences."""
        if in_flight < 1:
            raise ValueError("in_flight must be >= 1")
        if self.step_cmd is None:
            self.step_cmd = self._allocate_oneshot_cmd()
            vkBeginCommandBuffer(self.step_cmd, VkCommandBufferBeginInfo(
                flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT))
            self.record_step(self.step_cmd)
            vkEndCommandBuffer(self.step_cmd)
        if self.defrag_cmd is None:
            self.defrag_cmd = self._allocate_oneshot_cmd()
            vkBeginCommandBuffer(self.defrag_cmd, VkCommandBufferBeginInfo(
                flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT))
            self.record_defrag(self.defrag_cmd)
            vkEndCommandBuffer(self.defrag_cmd)
        self.drain()
        for fence in self._fences:
            vkDestroyFence(self.ctx.device, fence, None)
        self._fences = [vkCreateFence(self.ctx.device, VkFenceCreateInfo(
            sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, flags=VK_FENCE_CREATE_SIGNALED_BIT), None)
            for _ in range(in_flight)]
        self._submitted_count = 0
        print(f"[SimSingle] step + defrag command buffers recorded; {in_flight} fence(s) in flight")

    @property
    def in_flight(self) -> int:
        return len(self._fences)

    def submit_step(self, with_defrag: bool = False) -> None:
        """Submit one step (and, with ``with_defrag``, the defrag command buffer in the same submission)
        after the submission ``in_flight`` steps back has completed: at most ``in_flight`` steps are queued."""
        if not self._fences:
            raise RuntimeError("call prepare_step() first")
        if with_defrag and self.case.numerics.use_prefix_sum_defrag:
            # E9 prefix order: the scan needs the voxel counts of the step before the defrag, so the undrained
            # step + defrag submission becomes the step alone followed by the drained defrag (decided BEFORE this
            # slot's fence is waited on and reset: the inner submit_step owns the fence handling)
            self.submit_step(with_defrag=False)
            self.submit_defrag_and_wait()
            return
        slot = self._submitted_count % len(self._fences)
        fence = self._fences[slot]
        vkWaitForFences(self.ctx.device, 1, [fence], VK_TRUE, UINT64_MAX)
        vkResetFences(self.ctx.device, 1, [fence])
        command_buffers = [self.step_cmd] + ([self.defrag_cmd] if with_defrag else [])
        try:
            vkQueueSubmit(self.ctx.compute_queue, 1, VkSubmitInfo(
                sType=VK_STRUCTURE_TYPE_SUBMIT_INFO, commandBufferCount=len(command_buffers),
                pCommandBuffers=command_buffers), fence)
        except Exception:
            # a refused submission leaves the slot's fence unsignaled and every later drain would block forever:
            # replace it by a signaled fence before re-raising
            vkDestroyFence(self.ctx.device, fence, None)
            self._fences[slot] = vkCreateFence(self.ctx.device, VkFenceCreateInfo(
                sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, flags=VK_FENCE_CREATE_SIGNALED_BIT), None)
            raise
        self._submitted_count += 1

    def drain(self) -> None:
        """Wait for every submitted step (and defrag) to complete."""
        if self._fences:
            vkWaitForFences(self.ctx.device, len(self._fences), self._fences, VK_TRUE, UINT64_MAX)

    def step_and_wait(self) -> None:
        """One step with one fence: submit, then wait (the in_flight = 1 loop)."""
        if len(self._fences) != 1:
            raise RuntimeError("step_and_wait is the one-fence loop; prepare_step(in_flight=1)")
        self.submit_step()
        self.drain()

    def submit_defrag_and_wait(self) -> None:
        """The defrag alone, after a drain (the drained defrag boundary of the runners)."""
        if self.defrag_cmd is None:
            self.defrag_cmd = self._allocate_oneshot_cmd()
            vkBeginCommandBuffer(self.defrag_cmd, VkCommandBufferBeginInfo(
                flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT))
            self.record_defrag(self.defrag_cmd)
            vkEndCommandBuffer(self.defrag_cmd)
        self.drain()
        if self.case.numerics.use_prefix_sum_defrag:        # E9: the same host scan as SphSimulatorV7
            self._defrag_prefix_records.append(upload_defrag_prefix(self, verbose=len(self._defrag_prefix_records) < 2))
        self.ctx.submit_and_wait(self.defrag_cmd)

    # ------------------------------------------------------------------ readback

    def _readback_buffer(self, buffer: _Buffer) -> bytes:
        return self.readback_buffers_batch_raw({"_": buffer})["_"]

    def readback_buffers_batch_raw(self, buffers: dict[str, _Buffer]) -> dict[str, bytes]:
        """Copy the given buffers into host-visible stagings with one command buffer + one fence."""
        device = self.ctx.device
        stagings = {name: self._allocate_buffer(buffer.size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                    for name, buffer in buffers.items()}
        try:
            cmd = self._allocate_oneshot_cmd()
            vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
            for name, buffer in buffers.items():
                vkCmdCopyBuffer(cmd, buffer.handle, stagings[name].handle, 1,
                                [VkBufferCopy(srcOffset=0, dstOffset=0, size=buffer.size)])
            vkEndCommandBuffer(cmd)
            self.ctx.submit_and_wait(cmd)
            self._free_cmd(cmd)
            result = {}
            for name, staging in stagings.items():
                mapped = vkMapMemory(device, staging.memory, 0, staging.size, 0)
                result[name] = bytes(np.frombuffer(mapped, dtype=np.uint8, count=staging.size))
                vkUnmapMemory(device, staging.memory)
            return result
        finally:
            for staging in stagings.values():
                vkDestroyBuffer(device, staging.handle, None)
                vkFreeMemory(device, staging.memory, None)

    def readback_global_status(self) -> dict:
        """The 40-uint GlobalStatusBuffer (alive count, every overflow_* counter, ...)."""
        self.drain()
        fields = struct.unpack("<I f " + "I " * 38, self._readback_buffer(self.buffers["global_status"]))
        return {name: fields[index] for index, name in enumerate(GLOBAL_STATUS_FIELD_NAMES)}

    def readback_pool_health(self) -> dict:
        """install_migrations' watermarks, always 0 here (kept for the dump tools' metadata)."""
        peak_tail, peak_migration, peak_departed, _ = struct.unpack("<I I I I", self._readback_buffer(self.buffers["pool_health"]))
        own_pool = self.case.capacities.own_pool_size
        return {"peak_tail_high_water": peak_tail, "peak_migration_count": peak_migration,
                "peak_departed_count": peak_departed, "departed_pool_size": 0, "own_pool_size": own_pool,
                "free_margin": own_pool - peak_tail, "used_fraction": peak_tail / own_pool if own_pool else 0.0,
                "sizing_target": peak_tail}

    def readback_buffer_by_name(self, name: str) -> bytes:
        if name not in self.buffers:
            raise KeyError(f"unknown buffer: {name!r}")
        self.drain()
        return self._readback_buffer(self.buffers[name])

    def readback_buffers_batch(self, names: list[str], density: str = "absolute") -> dict[str, bytes]:
        """Read several buffers with one submit (v7's interface; ``density`` is a no-op without delta storage)."""
        if not names:
            return {}
        for name in names:
            if name not in self.buffers:
                raise KeyError(f"unknown buffer: {name!r}")
        self.drain()
        return self.readback_buffers_batch_raw({name: self.buffers[name] for name in names})

    # ------------------------------------------------------------------ teardown

    def destroy(self) -> None:
        if self._destroyed:
            return
        device = self.ctx.device
        vkDeviceWaitIdle(device)
        for fence in self._fences:
            vkDestroyFence(device, fence, None)
        self._fences = []
        for cmd in (self.step_cmd, self.defrag_cmd):
            if cmd is not None:
                self._free_cmd(cmd)
        self.step_cmd = self.defrag_cmd = None
        for pipeline in self.pipelines.values():
            vkDestroyPipeline(device, pipeline, None)
        self.pipelines = {}
        for module in self.shader_modules.values():
            vkDestroyShaderModule(device, module, None)
        self.shader_modules = {}
        vkDestroyPipelineLayout(device, self.defrag_pipeline_layout, None)
        vkDestroyPipelineLayout(device, self.pipeline_layout, None)
        vkDestroyDescriptorPool(device, self.defrag_descriptor_pool, None)
        vkDestroyDescriptorPool(device, self.descriptor_pool, None)
        vkDestroyDescriptorSetLayout(device, self.defrag_set4_layout, None)
        for layout in self.descriptor_layouts:
            vkDestroyDescriptorSetLayout(device, layout, None)
        self.descriptor_layouts = []
        for collection in (self.scratch_buffers, self.buffers):
            for buffer in collection.values():
                vkDestroyBuffer(device, buffer.handle, None)
                vkFreeMemory(device, buffer.memory, None)
        self.scratch_buffers = {}
        self.buffers = {}
        self._destroyed = True

    def __enter__(self) -> "SphSimulatorSingle":
        return self

    def __exit__(self, *_: Any) -> None:
        self.destroy()
