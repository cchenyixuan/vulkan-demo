"""
voxel_list_probe.py - the voxel neighbour lists of chosen voxels after N steps of a canonical run, as global
particle ids in list order, for a chain of K slabs (K = the number of devices).

    .venv/Scripts/python.exe experiment/wall_crossing/voxel_list_probe.py <case.yaml> <device map> <steps> \
        <out.json> <ix,iy> [<ix,iy> ...]

The run is the flow of experiment/seam_audit/canonical_dump.py --canonical-lists --transport-extension: upload,
voxelization, every voxel list sorted by particle id, the ghost round from the sorted lists, bootstrap compute, no
bootstrap defrag, then N pipelined steps (N below the defrag cadence). Every particle carries its row in the
global case in extension_fields (it travels with a migrant), so a list read back from any slab is a list of
global ids.

<ix,iy> are voxel coordinates of the ONE-slab grid (the global case's origin, voxel size h); a slab reports the
voxels it holds (own or ghost columns). The output also carries, per slab, the float32 grid origin the kernels
bin with, and per listed particle its float32 position: two layouts bin a particle with
floor((x - origin_x) / h) in float32 from different origins, so a particle within rounding of a voxel boundary
can sit in different voxels on the two layouts for a step.

This is a diagnostic of the K = 1 / K = 2 comparison (experiment/wall_crossing/k2_equivalence.py); 2-D cases.
"""
import json
import os
import pathlib
import sys

os.environ["V7_TRANSPORT_EXTENSION"] = "1"
os.environ["VK_LOADER_LAYERS_DISABLE"] = "VK_LAYER_KHRONOS_validation"
REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np                                                                # noqa: E402
from vulkan import (VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, VkCommandBufferBeginInfo,    # noqa: E402
                    vkBeginCommandBuffer, vkCmdDispatch, vkEndCommandBuffer, vkFreeCommandBuffers)

from experiment.v7.utils.case_loader_v7 import load_case_v7                       # noqa: E402
from experiment.v7.utils.orchestrator_v7 import ChainOrchestratorV7               # noqa: E402
from experiment.v7.utils.partition_v7 import compute_chain_partition              # noqa: E402
from experiment.v7.utils.simulator_v7 import SphSimulatorV7                       # noqa: E402
from experiment.v7.utils.vulkan_context_v7 import VulkanContextV7                 # noqa: E402


def main() -> int:
    case_path, steps, out_path = sys.argv[1], int(sys.argv[3]), sys.argv[4]
    device_map = [int(item) for item in sys.argv[2].split(",")]
    wanted = [tuple(int(value) for value in item.split(",")) for item in sys.argv[5:]]

    global_case = load_case_v7(case_path)
    support = np.float32(global_case.physics.smoothing_length)
    global_origin = np.array([global_case.grid.origin_x, global_case.grid.origin_y], dtype=np.float64)
    row_of_position = {global_case.initial.positions[row].tobytes(): row
                       for row in range(global_case.initial.positions.shape[0])}
    chain = compute_chain_partition(global_case, [1.0] * len(device_map), 1.2)
    contexts, simulators = [], []
    for device, slab in zip(device_map, chain.slabs):
        context = VulkanContextV7.create(device_index=device, enable_validation=False,
                                         application_name="voxel_list_probe")
        contexts.append(context)
        simulator = SphSimulatorV7(context, slab, sync_scheme="per-direction")
        ids = np.array([row_of_position[slab.initial.positions[row].tobytes()]
                        for row in range(slab.initial.positions.shape[0])], dtype=np.int64)
        original = simulator._build_initial_data

        def build(original=original, simulator=simulator, ids=ids):
            data = original()
            capacity = simulator.case.capacities.total_pool_capacity()
            extension = np.zeros((capacity, 4), dtype=np.float32)
            first = simulator.own_first_pid()
            extension[first:first + ids.size, 2] = ids // 2 ** 20
            extension[first:first + ids.size, 3] = ids % 2 ** 20 + 1          # + 1: 0 marks "no id"
            data["extension_fields"] = extension.tobytes()
            return data
        simulator._build_initial_data = build
        simulators.append(simulator)
    orchestrator = ChainOrchestratorV7(simulators, defrag_cadence=int(global_case.numerics.defrag_cadence))

    def submit(simulator, record):
        command = simulator._allocate_oneshot_cmd()
        vkBeginCommandBuffer(command, VkCommandBufferBeginInfo(flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
        record(command)
        vkEndCommandBuffer(command)
        simulator.ctx.submit_and_wait(command)
        vkFreeCommandBuffers(simulator.ctx.device, simulator.ctx.command_pool, 1, [command])

    def record_voxelization(simulator, command):
        simulator._record_compute_barrier(command)
        simulator._bind_pipeline_and_sets(command, "initialize_voxelization")
        vkCmdDispatch(command, simulator._per_own_particle_dispatch_count(), 1, 1)

    def record_ghost_send(simulator, command):
        for direction in ("leading", "trailing"):
            if direction not in simulator._transport_segments:
                continue
            simulator._record_compute_barrier(command)
            simulator._record_reset_ghost_send_count(command, direction)
            simulator._record_transfer_to_compute_barrier(command)
            simulator._record_ghost_send_dispatch(command, direction)
            simulator._record_compute_to_transfer_barrier(command)
            simulator._record_readback_for_direction(command, direction)
            simulator._record_compute_to_host_barrier(command)

    def sort_voxel_lists(simulator):
        slots = simulator.case.capacities.max_particles_per_voxel
        raw = simulator.readback_buffers_batch(["inside_particle_count", "inside_particle_index"])
        counts = np.frombuffer(raw["inside_particle_count"], np.uint32)
        index = np.frombuffer(raw["inside_particle_index"], np.uint32).copy()
        rows = index[:counts.size * slots].reshape(counts.size, slots)
        for voxel in np.flatnonzero(counts):
            rows[voxel, :counts[voxel]] = np.sort(rows[voxel, :counts[voxel]])
        simulator._staging_upload(simulator.buffers["inside_particle_index"], index.tobytes())

    for simulator in simulators:
        simulator._upload_initial_state()
        submit(simulator, lambda command, simulator=simulator: record_voxelization(simulator, command))
        sort_voxel_lists(simulator)
        submit(simulator, lambda command, simulator=simulator: record_ghost_send(simulator, command))
    for index in range(len(simulators) - 1):
        left, right = simulators[index], simulators[index + 1]
        right.receiver_staging_view("leading")[:] = left.sender_staging_view("trailing")
        left.receiver_staging_view("trailing")[:] = right.sender_staging_view("leading")
    for simulator in simulators:
        simulator.bootstrap_compute()
    for simulator in simulators:
        simulator.prepare_step_cmd_buffers()
    orchestrator.run_pipelined(steps, depth=2)

    result = {"case": case_path, "steps": steps, "device_map": device_map,
              "support": float(support), "global_origin": [float(value) for value in global_origin],
              "cuts": [int(cut) for cut in getattr(chain, "cuts", [])], "slabs": [], "voxels": {}}
    for slab_index, simulator in enumerate(simulators):
        case = simulator.case
        capacity = case.capacities.total_pool_capacity()
        slots = case.capacities.max_particles_per_voxel
        raw = simulator.readback_buffers_batch(["inside_particle_count", "inside_particle_index",
                                                "extension_fields", "position_voxel_id"])
        counts = np.frombuffer(raw["inside_particle_count"], np.uint32)
        index = np.frombuffer(raw["inside_particle_index"], np.uint32)[:counts.size * slots] \
            .reshape(counts.size, slots)
        extension = np.frombuffer(raw["extension_fields"], np.float32)[:capacity * 4].reshape(capacity, 4)
        position = np.frombuffer(raw["position_voxel_id"], np.float32)[:capacity * 4].reshape(capacity, 4)
        global_id = (extension[:, 2].astype(np.int64) * 2 ** 20 + extension[:, 3].astype(np.int64)) - 1
        origin = np.array([case.grid.origin_x, case.grid.origin_y], dtype=np.float64)
        column_shift = int(round((origin[0] - global_origin[0]) / float(support)))
        result["slabs"].append({"origin_x_float32": float(np.float32(case.grid.origin_x)),
                                "origin_y_float32": float(np.float32(case.grid.origin_y)),
                                "column_shift": column_shift,
                                "grid_dimension_x": int(case.grid.grid_dimension_x)})
        for ix, iy in wanted:
            local_x = ix - column_shift
            if not (0 <= local_x < case.grid.grid_dimension_x):
                continue
            voxel = iy + local_x * case.grid.grid_dimension_y * case.grid.grid_dimension_z + 1
            pids = index[voxel, :counts[voxel]]
            result["voxels"].setdefault(f"{ix},{iy}", {})[f"slab{slab_index}"] = {
                "local_voxel": int(voxel), "count": int(counts[voxel]),
                "global_ids": [int(global_id[pid]) for pid in pids],
                "x_float32": [float(position[pid, 0]) for pid in pids],
                "y_float32": [float(position[pid, 1]) for pid in pids]}
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle)
    orchestrator.destroy()
    for simulator in simulators:
        simulator.destroy()
    for context in contexts:
        context.destroy()
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
