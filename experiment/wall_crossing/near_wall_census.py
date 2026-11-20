"""
near_wall_census.py - report section 26, 2.1 / 3.1: who runs the near-wall build of numerics.wall_crossing.split_dispatch,
counted on the host from a case's initial particles (single slab, the whole grid).

    .venv/Scripts/python.exe -m experiment.wall_crossing.near_wall_census [--case DIR ...]

Per case: the own pool (the threads each of the two dispatches of a split pass launches, one per pid), the alive
fluid and wall particles; the voxels holding a wall particle; the near-wall voxels and the fluid particles in them
under the two marker rules (wall_crossing_near_wall_marker.comp):
  - neighbourhood (rule 0, sections 22-25): a voxel of the 3^d neighbourhood holds a wall particle;
  - distance (rule 1, section 26): a wall particle lies closer than 1 + 1/256 voxel to the voxel's cube, in the
    voxel units u = (x - origin) / h that bin a particle (float32, as predict.comp);
and the fluid particles that have a wall neighbour (a wall particle closer than h, float32 distance as the pair
loops test it). Every one of those must lie in a near-wall voxel under both rules (checked; the run fails if not).
Selection-level warp occupancy of the near-wall dispatch: pids in voxel-id order (the prefix defrag's order; inside a
voxel the loader's order), 32 per warp; a thread is busy in the near-wall build when it is a fluid particle of a
near-wall voxel (a wall particle leaves after its fixed outputs, a free slot or an interior particle at once).
Writes claude/wall_crossing/tables/near_wall_census.md and logs/wall_crossing/near_wall_census/summary.json.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.run_log import LOG_ROOT     # noqa: E402

CASES = ("lid_driven_cavity_2d_n250_xi0p001_eps0p0025_crossing", "lid_driven_cavity_2d_n500_xi0p001_eps0p0025_crossing",
         "lid_driven_cavity_2d_n1000_xi0p001_eps0p0025_crossing", "cavity3d_1m_crossing")
MARGIN = np.float32(1.0 / 256.0)
WARP = 32


def census(case_directory: str) -> dict:
    from scipy.spatial import cKDTree
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7.utils.case_v7 import KIND_BOUNDARY, KIND_FLUID
    case = load_case_v7(REPOSITORY_ROOT / "cases" / case_directory / "case.yaml")
    dimension = int(case.physics.dimension)
    support = np.float32(case.physics.smoothing_length)
    origin = np.array([case.grid.origin_x, case.grid.origin_y, case.grid.origin_z], dtype=np.float32)
    dimensions = np.array([case.grid.grid_dimension_x, case.grid.grid_dimension_y, case.grid.grid_dimension_z])
    positions = case.initial.positions.astype(np.float32)
    kinds = np.asarray([material.kind for material in case.materials])[case.initial.material_group.astype(np.int64)]
    fluid, wall = kinds == KIND_FLUID, kinds == KIND_BOUNDARY
    voxel_space = (positions - origin) / support                           # float32, predict.comp's expression
    coordinates = np.floor(voxel_space).astype(np.int64)
    if dimension == 2:
        coordinates[:, 2] = 0
    # x-slowest voxel id (helpers.glsl own_voxel_id_of, 0-based here)
    def voxel_id(coordinate: np.ndarray) -> np.ndarray:
        return coordinate[..., 1] + coordinate[..., 2] * dimensions[1] + coordinate[..., 0] * dimensions[1] * dimensions[2]

    total = int(np.prod(dimensions))
    holds_wall = np.zeros(total, dtype=bool)
    holds_wall[voxel_id(coordinates[wall])] = True
    offsets = np.array([(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                        for dz in ((-1, 0, 1) if dimension == 3 else (0,))], dtype=np.int64)
    near_neighbourhood = np.zeros(total, dtype=bool)
    near_distance = np.zeros(total, dtype=bool)
    wall_coordinates, wall_voxel_space = coordinates[wall], voxel_space[wall]
    reach_squared = np.float32((np.float32(1.0) + MARGIN) ** 2)
    for offset in offsets:
        target = wall_coordinates + offset
        inside = np.all((target >= 0) & (target < dimensions), axis=1)
        target, source = target[inside], wall_voxel_space[inside]
        near_neighbourhood[voxel_id(target)] = True
        lower = target.astype(np.float32)
        gap = np.maximum(np.maximum(lower - source, source - (lower + np.float32(1.0))), np.float32(0.0))
        if dimension == 2:
            gap[:, 2] = 0.0
        close = np.einsum("ij,ij->i", gap, gap) < reach_squared
        near_distance[voxel_id(target[close])] = True
    particle_voxel = voxel_id(coordinates)
    # fluid particles with a wall neighbour: a wall closer than h (float32 pair distance, as the loops test it)
    tree = cKDTree(positions[wall].astype(np.float64))
    candidates = tree.query_ball_point(positions[fluid].astype(np.float64), r=float(support) * 1.001)
    fluid_index = np.flatnonzero(fluid)
    wall_positions = positions[wall]
    has_wall_neighbour = np.zeros(positions.shape[0], dtype=bool)
    for row, neighbours in zip(fluid_index, candidates):
        if not neighbours:
            continue
        distance = np.sqrt(np.sum((positions[row] - wall_positions[neighbours]) ** 2, axis=1, dtype=np.float32))
        if np.any((distance < support) & (distance >= np.float32(1e-12))):
            has_wall_neighbour[row] = True
    result = {"case": case_directory, "dimension": dimension, "own_pool": int(case.capacities.own_pool_size),
              "fluid": int(fluid.sum()), "wall": int(wall.sum()), "voxels": total,
              "voxels_holding_wall": int(holds_wall.sum()), "fluid_with_wall_neighbour": int(has_wall_neighbour.sum())}
    order = np.argsort(particle_voxel, kind="stable")                     # the prefix defrag's pid order
    for rule, near in (("neighbourhood", near_neighbourhood), ("distance", near_distance)):
        particle_near = near[particle_voxel]
        missed = int((has_wall_neighbour & ~particle_near).sum()) + int((wall & ~particle_near).sum())
        if missed:
            raise RuntimeError(f"{case_directory}: rule {rule} leaves {missed} particles with a wall neighbour "
                               "(or walls) outside the near-wall voxels")
        busy_near = (fluid & particle_near)[order]
        busy_interior = (fluid & ~particle_near)[order]
        padded = int(np.ceil(case.capacities.own_pool_size / WARP) * WARP)
        rows = {}
        for name, busy in (("near_wall", busy_near), ("interior", busy_interior)):
            lanes = np.zeros(padded, dtype=bool)
            lanes[:busy.size] = busy
            per_warp = lanes.reshape(-1, WARP).sum(axis=1)
            working = per_warp > 0
            rows[name] = {"busy_threads": int(busy.sum()), "warps_working": int(working.sum()),
                          "warps_launched": int(per_warp.size),
                          "active_fraction_in_working_warps": float(per_warp[working].sum() / (WARP * working.sum()))
                          if working.any() else 0.0}
        result[rule] = {"voxels_near_wall": int(near.sum()), "fluid_near_wall": int((fluid & particle_near).sum()),
                        "wall_near_wall": int((wall & particle_near).sum()),
                        "fluid_near_wall_without_wall_neighbour": int((fluid & particle_near & ~has_wall_neighbour).sum()),
                        "warps": rows}
    return result


def gpu_flags(cases: list, device: int, uuid: str) -> int:
    """The GPU marker's flags after the bootstrap (SphSimulatorV7.readback_wall_crossing_near_wall) under the rule of
    V7_WALL_CROSSING_NEAR_WALL_RULE (set by the caller; read when the simulator module is imported):
    logs/wall_crossing/near_wall_census/gpu_<rule>.json."""
    import os
    os.environ.setdefault("VK_LOADER_LAYERS_DISABLE", "VK_LAYER_KHRONOS_validation")
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7.utils import simulator_v7
    from experiment.v7.utils.vulkan_context_v7 import VulkanContextV7
    from experiment.wall_crossing.run_log import LoggedRun
    rule = simulator_v7._WALL_CROSSING_NEAR_WALL_RULE
    records = []
    for case_directory in cases:
        case_path = REPOSITORY_ROOT / "cases" / case_directory / "case.yaml"
        with LoggedRun(kind="near_wall_census_gpu", case=f"cases/{case_directory}/case.yaml",
                       gpu_uuids=[uuid] if uuid else []) as run:
            case = load_case_v7(case_path)
            context = VulkanContextV7.create(device_index=device, enable_validation=False,
                                             application_name="near_wall_census")
            simulator = simulator_v7.SphSimulatorV7(context, case)
            simulator.bootstrap()
            statistics = simulator.readback_wall_crossing_near_wall()
            statistics.update({"case": case_directory, "rule": rule, "run_id": run.run_id})
            run.extra["near_wall"] = statistics
            simulator.destroy()
            context.destroy()
        records.append(statistics)
        print(json.dumps(statistics), flush=True)
    out = LOG_ROOT / "near_wall_census"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"gpu_{rule}.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    return 0


def main() -> int:
    from experiment.wall_crossing.report_tables import markdown_table, write
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", action="append", default=None)
    parser.add_argument("--gpu", action="store_true", help="read the GPU marker's flags instead (gpu_flags)")
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--uuid", default="ae137c668a40f5acaab90a83f7cda175")
    arguments = parser.parse_args()
    if arguments.gpu:
        return gpu_flags(arguments.case or list(CASES), arguments.device, arguments.uuid)
    results = [census(case) for case in (arguments.case or CASES)]
    out = LOG_ROOT / "near_wall_census"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    label = {"lid_driven_cavity_2d_n250_xi0p001_eps0p0025_crossing": "2-D 250²",
             "lid_driven_cavity_2d_n500_xi0p001_eps0p0025_crossing": "2-D 500²",
             "lid_driven_cavity_2d_n1000_xi0p001_eps0p0025_crossing": "2-D 1000²", "cavity3d_1m_crossing": "3-D 1M"}
    rows, warp_rows = [], []
    for result in results:
        fluid = result["fluid"]
        old, new = result["neighbourhood"], result["distance"]
        rows.append([label.get(result["case"], result["case"]), f"{result['own_pool']:,}", f"{fluid:,}",
                     f"{result['wall']:,}",
                     f"{old['fluid_near_wall']:,}（{100 * old['fluid_near_wall'] / fluid:.2f} %）",
                     f"{new['fluid_near_wall']:,}（{100 * new['fluid_near_wall'] / fluid:.2f} %）",
                     f"{result['fluid_with_wall_neighbour']:,}（{100 * result['fluid_with_wall_neighbour'] / fluid:.2f} %）",
                     f"{old['voxels_near_wall']:,} → {new['voxels_near_wall']:,}（共 {result['voxels']:,}）"])
        for rule, text in (("neighbourhood", "旧（3^d 邻域）"), ("distance", "新（距离）")):
            for name, build in (("near_wall", "近壁"), ("interior", "内部")):
                entry = result[rule]["warps"][name]
                warp_rows.append([label.get(result["case"], result["case"]), text, build, f"{entry['busy_threads']:,}",
                                  f"{entry['warps_working']:,} / {entry['warps_launched']:,}",
                                  f"{100 * entry['active_fraction_in_working_warps']:.1f} %"])
    write("near_wall_census.md", markdown_table(
        ["算例", "每次调度的线程（自有池）", "流体粒子", "壁面粒子", "近壁流体：旧规则", "近壁流体：新规则",
         "真正有壁面邻居的流体", "近壁体素：旧 → 新"], rows))
    write("near_wall_census_warps.md", markdown_table(
        ["算例", "规则", "版本", "干活的线程（流体）", "有活的线程束 / 发出的线程束", "有活的线程束里干活线程的比例"],
        warp_rows))
    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
