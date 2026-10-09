"""
_v7_cpu_fake.py — build v7's and v7_single's simulators on the CPU (object.__new__, no Vulkan device) so
their specialization constants and command streams can be read without a GPU.

The approach is experiment/v7/_test_seam_layout.py's: the simulator object gets only the attributes its
spec-constant builders and recorders read (case, band widths, fake buffers named "buffer:<name>" and
"scratch:<name>", a pipeline table that names any key, fake descriptor sets and layouts), and the vkCmd*
entry points of the simulator module are replaced by recorders for the duration of a capture
(capture_v7 / capture_single). Pipeline specs come from each simulator's own pipeline builder with
_create_pipeline replaced by a function that returns (shader name, entries).

Nothing here touches the GPU; the modules are imported (python-vulkan loads the loader library only).
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any

RECORDED_FUNCTIONS = ("vkCmdPipelineBarrier2", "vkCmdCopyBuffer", "vkCmdDispatch", "vkCmdDispatchIndirect",
                      "vkCmdBindPipeline", "vkCmdBindDescriptorSets", "vkCmdFillBuffer", "vkBeginCommandBuffer",
                      "vkEndCommandBuffer", "vkFreeCommandBuffers")


class PipelineNames(dict):
    """A pipeline table that names every key it is asked for (the recorder sees the key, not a handle)."""

    def __missing__(self, key):
        return f"pipeline:{key}"


def _recorders(events: list) -> dict[str, Any]:
    def barrier(cmd, info):
        for index in range(info.memoryBarrierCount):
            item = info.pMemoryBarriers[index]
            events.append(("barrier", int(item.srcStageMask), int(item.srcAccessMask),
                           int(item.dstStageMask), int(item.dstAccessMask)))
        if info.bufferMemoryBarrierCount or info.imageMemoryBarrierCount:
            events.append(("barrier_other", int(info.bufferMemoryBarrierCount), int(info.imageMemoryBarrierCount)))

    def copy(cmd, source, destination, count, regions):
        events.append(("copy", source, destination,
                       tuple((int(regions[index].srcOffset), int(regions[index].dstOffset), int(regions[index].size))
                             for index in range(count))))

    return {
        "vkCmdPipelineBarrier2": barrier,
        "vkCmdCopyBuffer": copy,
        "vkCmdDispatch": lambda cmd, x, y, z: events.append(("dispatch", int(x), int(y), int(z))),
        "vkCmdDispatchIndirect": lambda cmd, buffer, offset: events.append(("dispatch_indirect", buffer, int(offset))),
        "vkCmdBindPipeline": lambda cmd, point, pipeline: events.append(("bind", pipeline)),
        "vkCmdBindDescriptorSets": lambda cmd, point, layout, first, count, sets, *rest:
            events.append(("sets", layout, int(first), int(count))),
        "vkCmdFillBuffer": lambda cmd, buffer, offset, size, value:
            events.append(("fill", buffer, int(offset), int(size), int(value))),
        "vkBeginCommandBuffer": lambda cmd, info: events.append(("begin",)),
        "vkEndCommandBuffer": lambda cmd: events.append(("end",)),
        "vkFreeCommandBuffers": lambda device, pool, count, cmds: None,
    }


@contextlib.contextmanager
def recording(module, events: list):
    """Replace the module's vkCmd* / vkBegin / vkEnd / vkFree entry points by recorders appending to ``events``."""
    saved = {name: getattr(module, name) for name in RECORDED_FUNCTIONS}
    try:
        for name, function in _recorders(events).items():
            setattr(module, name, function)
        yield events
    finally:
        for name, function in saved.items():
            setattr(module, name, function)


@contextlib.contextmanager
def v7_switch_values(simulator_v7, *, fused: int, copy_compute: int, deep_wall: str, record: bool = False):
    """Set simulator_v7's import-time switch constants for the duration of the block."""
    saved = (simulator_v7._FUSED_CORRECTION_DENSITY, simulator_v7._DENSITY_COPY_COMPUTE,
             simulator_v7._DEEP_WALL_SKIP, simulator_v7._DEEP_WALL_RECORD_DECISIONS)
    simulator_v7._FUSED_CORRECTION_DENSITY = int(fused)
    simulator_v7._DENSITY_COPY_COMPUTE = bool(copy_compute)
    simulator_v7._DEEP_WALL_SKIP = str(deep_wall)
    simulator_v7._DEEP_WALL_RECORD_DECISIONS = bool(record)
    try:
        yield
    finally:
        (simulator_v7._FUSED_CORRECTION_DENSITY, simulator_v7._DENSITY_COPY_COMPUTE,
         simulator_v7._DEEP_WALL_SKIP, simulator_v7._DEEP_WALL_RECORD_DECISIONS) = saved


def _fake_context():
    return SimpleNamespace(device="device", command_pool="pool", physical_device="physical_device",
                           compute_queue="queue", device_name="cpu", transfer_command_pool="transfer_pool",
                           compute_queue_family_index=0, transfer_queue_family_index=1,
                           submit_and_wait=lambda cmd: None)


def fake_v7_simulator(simulator_v7, case):
    """An SphSimulatorV7 of ``case`` without a device: enough for _build_buffer_specs, _global_entries, the
    resolutions, _build_compute_pipelines (with _create_pipeline replaced) and the recorders."""
    simulator = object.__new__(simulator_v7.SphSimulatorV7)
    simulator.case = case
    simulator.ctx = _fake_context()
    simulator.band_widths = simulator._configured_band_widths()
    simulator._spec_keepalive = []
    simulator._transport_segments = {}
    simulator._transport_total_bytes = {}
    simulator.staging_buffers = {}
    simulator._recv_status_overrides = {}
    simulator.bench = None
    simulator.bench_transfer = None
    simulator.step_single_use_split = False
    simulator.step_trace_parity = False
    specs = simulator._build_buffer_specs()
    simulator._buffer_specs = specs
    simulator.buffers = {spec.name: SimpleNamespace(handle=f"buffer:{spec.name}", size=spec.size) for spec in specs}
    simulator.scratch_buffers = {spec.name: SimpleNamespace(handle=f"scratch:{spec.name}", size=spec.size)
                                 for spec in specs if spec.set_index == 0
                                 and spec.binding in simulator_v7.DEFRAG_SET0_BINDINGS}
    simulator.pipelines = PipelineNames()
    simulator.pipeline_layout = "pipeline_layout"
    simulator.defrag_pipeline_layout = "defrag_pipeline_layout"
    simulator.descriptor_sets = ["set0", "set1", "set2", "set3"]
    simulator.defrag_set4 = "set4"
    simulator._allocate_oneshot_cmd = lambda: "cmd"
    return simulator


def fake_single_simulator(simulator_single, case, switches, record: bool = False):
    """An SphSimulatorSingle of ``case`` without a device (the same attributes as fake_v7_simulator)."""
    from experiment.v7_single.utils.deep_wall_single import resolve_deep_wall_skip
    simulator = object.__new__(simulator_single.SphSimulatorSingle)
    simulator.case = case
    simulator.ctx = _fake_context()
    simulator.switches = switches
    simulator.spirv_directory = simulator_single.spirv_manifest.V7_SHADER_DIRECTORY
    simulator.wall_boundary = case.numerics.wall_boundary
    simulator.deep_wall_skip_resolution = resolve_deep_wall_skip(case, switches.deep_wall_skip,
                                                                 simulator_single.DEEP_WALL_SKIP_BAND_WIDTH)
    simulator.fused_correction_density_resolution = (bool(switches.fused_correction_density), "")
    simulator.deep_wall_record = bool(record)
    simulator._spec_keepalive = []
    specs = simulator._build_buffer_specs()
    simulator._buffer_specs = specs
    simulator.buffers = {spec.name: SimpleNamespace(handle=f"buffer:{spec.name}", size=spec.size) for spec in specs}
    simulator.scratch_buffers = {spec.name: SimpleNamespace(handle=f"scratch:{spec.name}", size=spec.size)
                                 for spec in specs if spec.set_index == 0
                                 and spec.binding in simulator_single.DEFRAG_SET0_BINDINGS}
    simulator.pipelines = PipelineNames()
    simulator.pipeline_layout = "pipeline_layout"
    simulator.defrag_pipeline_layout = "defrag_pipeline_layout"
    simulator.descriptor_sets = ["set0", "set1", "set2", "set3"]
    simulator.defrag_set4 = "set4"
    simulator.step_cmd = None
    simulator.defrag_cmd = None
    simulator._fences = []
    simulator._submitted_count = 0
    return simulator


def entries_to_dict(entries) -> dict[int, tuple[str, Any]]:
    """{constant id: (format, value)} of a specialization entry list; the value is the exact Python value."""
    return {int(identifier): (fmt, (1 if value else 0) if fmt == "B" else value) for identifier, fmt, value in entries}


def v7_pipeline_specs(simulator_v7, case, *, fused: int, copy_compute: int, deep_wall: str,
                      record: bool = False) -> dict[str, tuple[str, dict]]:
    """{pipeline key: (shader name, {id: (format, value)})} of every compute pipeline v7 builds for ``case``
    under the given switch values (the defrag pipeline, built apart with the global entries, is added as
    'defrag')."""
    with v7_switch_values(simulator_v7, fused=fused, copy_compute=copy_compute, deep_wall=deep_wall, record=record):
        simulator = fake_v7_simulator(simulator_v7, case)
        shader_names = set()

        class ShaderNames(dict):
            def __missing__(self, key):
                shader_names.add(key)
                return key
        simulator.shader_modules = ShaderNames()
        records = {}

        def create_pipeline(shader, entries):
            return ("pipeline", shader, list(entries))
        simulator._create_pipeline = create_pipeline
        pipelines = simulator._build_compute_pipelines()
        for key, (_tag, shader, entries) in pipelines.items():
            records[key] = (shader, entries_to_dict(entries))
        records["defrag"] = ("defrag", entries_to_dict(simulator._global_entries()))
    return records


def single_pipeline_specs(simulator_single, case, switches, record: bool = False) -> dict[str, tuple[str, dict]]:
    simulator = fake_single_simulator(simulator_single, case, switches, record=record)
    records = {key: (shader, entries_to_dict(entries)) for key, (shader, entries) in simulator.pipeline_specs().items()}
    records["defrag"] = ("defrag", entries_to_dict(simulator.global_entries()))
    return records
