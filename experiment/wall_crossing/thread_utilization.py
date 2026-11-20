"""
thread_utilization.py - report section 27, 2.2: how many threads of the split dispatches' warps do work, counted on
the host from the GPU's own particle order after one step.

    .venv/Scripts/python.exe -m experiment.wall_crossing.thread_utilization --device 1 [--steps 1]

Per case: the K = 1 simulator is bootstrapped (its initial defrag puts the pids in voxel-id order), runs --steps steps,
and the own slots' position (voxel id), mass (alive) and material are read back together with the near-wall flags.
Both dispatches of a split pass launch one thread per own slot (thread t = pid own_first_pid + t), 32 per warp. A
thread WORKS in the near-wall dispatch when its particle is an alive fluid particle of a near-wall voxel (it enters the
neighbour loop); a wall particle (fixed outputs, then return), a particle of the other class and an empty slot idle.
Likewise for the interior dispatch. Per dispatch: the warps with at least one working thread, the working threads per
such warp (mean, distribution), and the warps whose working threads lie in two or more voxels (they walk different
neighbour lists; the mean number of distinct voxels too). --defrag: after the steps a defrag (the steady order, voxel-id sorted) and one more step.
Writes claude/wall_crossing/tables/thread_utilization[_after_defrag]*.md and
logs/wall_crossing/section27/thread_utilization[_after_defrag].json.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.run_log import LOG_ROOT, LoggedRun     # noqa: E402

CASES = (("2-D 250²", "lid_driven_cavity_2d_n250_xi0p001_eps0p0025_crossing"),
         ("2-D 1000²", "lid_driven_cavity_2d_n1000_xi0p001_eps0p0025_crossing"),
         ("3-D 1M", "cavity3d_1m_crossing"))
WARP = 32
BINS = ((1, 8), (9, 16), (17, 24), (25, 31), (32, 32))


def warp_statistics(working: np.ndarray, voxel: np.ndarray) -> dict:
    padded = int(np.ceil(working.size / WARP) * WARP)
    lanes = np.zeros(padded, dtype=bool)
    lanes[:working.size] = working
    voxels = np.full(padded, -1, dtype=np.int64)
    voxels[:working.size] = np.where(working, voxel, -1)
    lanes, voxels = lanes.reshape(-1, WARP), voxels.reshape(-1, WARP)
    per_warp = lanes.sum(axis=1)
    busy = per_warp > 0
    distinct = np.array([np.unique(row[row >= 0]).size for row in voxels[busy]])
    histogram = {f"{low}-{high}": int(((per_warp >= low) & (per_warp <= high)).sum()) for low, high in BINS}
    return {"warps_launched": int(per_warp.size), "warps_working": int(busy.sum()),
            "working_threads": int(per_warp.sum()),
            "mean_working_per_busy_warp": float(per_warp[busy].mean()) if busy.any() else 0.0,
            "histogram": histogram,
            "warps_spanning_two_or_more_voxels": int((distinct >= 2).sum()),
            "mean_distinct_voxels": float(distinct.mean()) if distinct.size else 0.0}


def measure(case_directory: str, steps: int, device: int, uuid: str, defrag: bool = False) -> dict:
    os.environ.setdefault("VK_LOADER_LAYERS_DISABLE", "VK_LAYER_KHRONOS_validation")
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7.utils.case_v7 import KIND_BOUNDARY, KIND_FLUID
    from experiment.v7.utils.simulator_v7 import SphSimulatorV7
    from experiment.v7.utils.vulkan_context_v7 import VulkanContextV7
    with LoggedRun(kind="thread_utilization", case=f"cases/{case_directory}/case.yaml", gpu_uuids=[uuid]) as run:
        case = load_case_v7(REPOSITORY_ROOT / "cases" / case_directory / "case.yaml")
        context = VulkanContextV7.create(device_index=device, enable_validation=False,
                                         application_name="thread_utilization")
        simulator = SphSimulatorV7(context, case)
        simulator.bootstrap()
        simulator.prepare_step_single_cmd_buffer()
        for _ in range(steps):
            simulator.submit_step_single_and_wait()
        if defrag:
            # the order of steady running: the defrag (every 1,000 steps) sorts the pids by voxel id, then one step
            simulator.submit_defrag_and_wait()
            simulator.submit_step_single_and_wait()
        raw = simulator.readback_buffers_batch(["wall_crossing_near_wall_voxel_flag", "position_voxel_id",
                                                "velocity_mass", "material"])
        total = case.grid.total_voxel_count()
        flags = np.frombuffer(raw["wall_crossing_near_wall_voxel_flag"], dtype=np.uint32)
        near_flag = flags[total + 2:2 * total + 2]
        capacity = case.capacities.total_pool_capacity()
        first = simulator.own_first_pid()
        rows = slice(first, first + case.capacities.own_pool_size)
        position = np.frombuffer(raw["position_voxel_id"], dtype=np.float32)[:capacity * 4].reshape(capacity, 4)[rows]
        mass = np.frombuffer(raw["velocity_mass"], dtype=np.float32)[:capacity * 4].reshape(capacity, 4)[rows, 3]
        material = np.frombuffer(raw["material"], dtype=np.uint32)[:capacity][rows]
        simulator.destroy()
        context.destroy()
        kinds = np.asarray([entry.kind for entry in case.materials])[np.minimum(material, len(case.materials) - 1)]
        voxel = np.rint(position[:, 3]).astype(np.int64)
        alive = (voxel >= 1) & (voxel <= total) & (mass > 0.0)
        near = np.zeros(voxel.size, dtype=bool)
        near[alive] = near_flag[voxel[alive] - 1] != 0
        fluid, wall = alive & (kinds == KIND_FLUID), alive & (kinds == KIND_BOUNDARY)
        result = {"case": case_directory, "steps": steps, "defrag": defrag, "run_id": run.run_id,
                  "own_slots": int(voxel.size),
                  "fluid": int(fluid.sum()), "wall": int(wall.sum()),
                  "particles_per_voxel_fluid_median": float(np.median(np.bincount(voxel[fluid])[np.bincount(voxel[fluid]) > 0])),
                  "near_wall": warp_statistics(fluid & near, voxel), "interior": warp_statistics(fluid & ~near, voxel)}
        run.extra["thread_utilization"] = {key: value for key, value in result.items() if key != "run_id"}
    return result


def main() -> int:
    from experiment.wall_crossing.report_tables import markdown_table, write
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--uuid", default="ae137c668a40f5acaab90a83f7cda175")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--defrag", action="store_true", help="after --steps steps a defrag and one more step")
    arguments = parser.parse_args()
    results = [(label, measure(case, arguments.steps, arguments.device, arguments.uuid, arguments.defrag))
               for label, case in CASES]
    suffix = "_after_defrag" if arguments.defrag else ""
    out = LOG_ROOT / "section27"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"thread_utilization{suffix}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    rows, histogram_rows = [], []
    for label, result in results:
        for name, text in (("near_wall", "近壁"), ("interior", "内部")):
            entry = result[name]
            rows.append([label, text, f"{entry['working_threads']:,}",
                         f"{entry['warps_working']:,} / {entry['warps_launched']:,}",
                         f"{entry['mean_working_per_busy_warp']:.1f}（{100 * entry['mean_working_per_busy_warp'] / WARP:.0f} %）",
                         f"{100 * entry['warps_spanning_two_or_more_voxels'] / max(entry['warps_working'], 1):.0f} %"
                         f"（平均 {entry['mean_distinct_voxels']:.2f} 个体素）",
                         f"{result['particles_per_voxel_fluid_median']:.0f}"])
            histogram_rows.append([label, text] + [f"{entry['histogram'][key]:,}" for key in entry["histogram"]])
    write(f"thread_utilization{suffix}.md", markdown_table(
        ["算例", "调度", "干活的线程", "有活的线程束 / 发出的线程束", "有活的线程束里平均干活的线程",
         "干活线程跨两个以上体素的线程束", "每个体素的流体粒子（中位）"], rows))
    write(f"thread_utilization{suffix}_histogram.md", markdown_table(
        ["算例", "调度"] + [f"{low}–{high} 个干活" if low != high else f"{low} 个全干活" for low, high in BINS],
        histogram_rows))
    print(json.dumps(results, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
