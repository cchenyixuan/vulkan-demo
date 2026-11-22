"""
frozen_evaluate.py - one evaluation of the fused correction + density pass and of force on a saved state, without
advancing time, by the solver of a given checkout (claude/wall_store_report.md, the hydrostatic regression).

    .venv/Scripts/python.exe experiment/wall_crossing/frozen_evaluate.py --repo <checkout> --case <case.yaml> \
        --state <checkpoint.npz> --device 1 --out <evaluation.npz> [--drop-walls] [--sort-voxel-lists]

The checkout goes first on sys.path (start the script by its path). K = 1: the state's rows go to the own pids in
order (SphSimulatorV7.restart_init, voxel lists rebuilt), then one command buffer records correction_density_all and,
after a barrier, force_all (the pipelines the bootstrap records: the one-pipeline builds of both checkouts), and
the buffers are read back. Force reads the state's (rho_n, P_n) (no scratch -> primary copy), the same for every
checkout. --drop-walls removes the state's wall rows (the wall store keeps its walls out of the pool). Output per
row of the (possibly reduced) state: positions, the density rate (scratch - stored) / dt, acceleration, shift,
correction_inverse (8 floats), density gradient and kernel sum, and the row index into the input state.
"""
import argparse
import os
import pathlib
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--drop-walls", action="store_true")
    parser.add_argument("--sort-voxel-lists", action="store_true",
                        help="sort every voxel's particle list by pid after the rebuild (the atomic fill order varies "
                             "from run to run, and with it the summation order): a bit-reproducible evaluation")
    arguments = parser.parse_args()
    repo = pathlib.Path(arguments.repo).resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    os.environ.setdefault("VK_LOADER_LAYERS_DISABLE", "VK_LAYER_KHRONOS_validation")
    import numpy as np
    from vulkan import (VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, VkCommandBufferBeginInfo, vkBeginCommandBuffer,
                        vkCmdDispatch, vkEndCommandBuffer, vkFreeCommandBuffers)
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7.utils.partition_v7 import compute_chain_partition
    from experiment.v7.utils.simulator_v7 import SphSimulatorV7
    from experiment.v7.utils.vulkan_context_v7 import VulkanContextV7

    case = load_case_v7(pathlib.Path(arguments.case).resolve())
    with np.load(arguments.state) as archive:
        state = {name: archive[name] for name in archive.files if not name.startswith("_")}
        stored_offset = float(archive["_stored_density_offset"])
    rows = np.arange(state["position_voxel_id"].shape[0])
    if arguments.drop_walls:
        kinds = np.asarray([material.kind for material in case.materials])
        rows = np.flatnonzero(kinds[state["material"].astype(np.int64)] != 1)
        state = {name: np.ascontiguousarray(values[rows]) for name, values in state.items()}
    slab = compute_chain_partition(case, [1.0], 1.2).slabs[0]
    context = VulkanContextV7.create(device_index=arguments.device, enable_validation=False,
                                     application_name="frozen_evaluate")
    simulator = SphSimulatorV7(context, slab, sync_scheme="per-direction")
    simulator.restart_init(state, stored_offset)
    if arguments.sort_voxel_lists:
        per_voxel = int(slab.capacities.max_particles_per_voxel)
        raw = simulator.readback_buffers_batch(["inside_particle_index", "inside_particle_count"])
        index = np.frombuffer(raw["inside_particle_index"], dtype=np.uint32).reshape(-1, per_voxel)
        counts = np.frombuffer(raw["inside_particle_count"], dtype=np.uint32)[:index.shape[0]].astype(np.int64)
        occupied = np.arange(per_voxel)[None, :] < counts[:, None]
        ordered = np.sort(np.where(occupied, index, np.uint32(0xFFFFFFFF)), axis=1)
        simulator._staging_upload(simulator.buffers["inside_particle_index"],
                                  np.ascontiguousarray(np.where(occupied, ordered, index)).tobytes())
    command = simulator._allocate_oneshot_cmd()
    vkBeginCommandBuffer(command, VkCommandBufferBeginInfo(flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
    groups = simulator._per_own_particle_dispatch_count()
    simulator._record_compute_barrier(command)
    simulator._bind_pipeline_and_sets(command, "correction_density_all")
    vkCmdDispatch(command, groups, 1, 1)
    simulator._record_compute_barrier(command)
    simulator._bind_pipeline_and_sets(command, "force_all")
    vkCmdDispatch(command, groups, 1, 1)
    simulator._record_compute_barrier(command)
    vkEndCommandBuffer(command)
    context.submit_and_wait(command)
    vkFreeCommandBuffers(context.device, context.command_pool, 1, [command])
    names = ["position_voxel_id", "density_pressure", "density_pressure_scratch", "acceleration", "shift",
             "correction_inverse", "density_gradient_kernel_sum", "velocity_mass", "material"]
    raw = simulator.readback_buffers_batch(names, density="stored")
    capacity = slab.capacities.total_pool_capacity()
    first = simulator.own_first_pid()
    count = rows.size

    def field(name, width, dtype=np.float32):
        values = np.frombuffer(raw[name], dtype=dtype)[:capacity * width]
        values = values.reshape(capacity, width) if width > 1 else values
        return np.array(values[first:first + count])

    stored = field("density_pressure", 2)
    scratch = field("density_pressure_scratch", 2)
    timestep = float(np.float32(case.physics.timestep))
    np.savez(arguments.out, state_rows=rows, positions=field("position_voxel_id", 4)[:, :3],
             density_rate=(scratch[:, 0].astype(np.float64) - stored[:, 0].astype(np.float64)) / timestep,
             acceleration=field("acceleration", 4)[:, :3], shift=field("shift", 4)[:, :3],
             correction_inverse=field("correction_inverse", 8),
             density_gradient_kernel_sum=field("density_gradient_kernel_sum", 4),
             material=field("material", 1, np.uint32), timestep=timestep)
    status = simulator.readback_global_status()
    print(f"[frozen_evaluate] {repo.name}: {count:,} rows evaluated, correction fallbacks "
          f"{status.get('correction_fallback_count')}, gradient fallbacks {status.get('wall_crossing_gradient_fallback_count')}"
          f" -> {arguments.out}")
    simulator.destroy()
    context.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
