"""check_wall_pass.py - recompute the E36 wall pass (wall_extrapolate.comp) in float64 numpy from a k1_dump.py state of a
v7 WALL_BC = 1, 3 or 4 run and compare it with the GPU's float32 result (3 = adami_rho0 and 4 = pressure only store
rho0 instead of rho_w; the pass is otherwise the same).

The dump is taken after a step with no defrag between that step's wall pass and the readback, so it holds exactly the
pass's inputs and outputs: positions and the fluid's stored half-step velocity (unchanged after phase A), the fluid's
rho / P of the step (density_pressure_scratch, which the cascading phase B reads; equal to the copied primary), the
walls' stored, prescribed velocity, and the outputs (rho_w, p_w) in density_pressure and (u_dummy, sum_f W_wf) in
wall_dummy_velocity. For every wall particle, over the fluid neighbours within the support radius h (Wendland C4,
2-D: W = 9 / (pi h^2) (1 - q)^6 (35/3 q^2 + 6 q + 1)):
    u~ = sum u_f W / sum W,  p_w = [sum p_f W + g . sum rho_f (r_w - r_f) W] / sum W,
    rho_w = rho0 (1 + p_w / B)^(1 / gamma)  (B = c0^2 rho0 / gamma),  u_dummy = 2 u_w - u~;
no fluid neighbour: p_w = 0, rho_w = rho0, u_dummy = u_w.
Writes a JSON report next to the dump (<dump>.check.json) and prints it.

    .venv/Scripts/python.exe -m experiment.v7.wall_bc.check_wall_pass logs/e36/numpy_check/v7_bc1_t1.npz
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.spatial import cKDTree

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

FLOAT32_EPSILON = float(np.finfo(np.float32).eps)       # 1.19e-7


def wendland_c4_2d(distance: np.ndarray, support: float) -> np.ndarray:
    q = distance / support
    return 9.0 / (np.pi * support ** 2) * (1.0 - q) ** 6 * ((35.0 / 3.0) * q ** 2 + 6.0 * q + 1.0)


def main() -> int:
    path = pathlib.Path(sys.argv[1]).resolve()
    with np.load(path) as archive:
        state = {name: archive[name] for name in archive.files if name != "meta"}
        meta = json.loads(str(archive["meta"]))
    if meta["solver"] != "v7" or meta["wall_bc"] not in (1, 3, 4):
        sys.exit("needs a v7 WALL_BC = 1, 3 or 4 dump")
    stores_rest_density = meta["wall_bc"] in (3, 4)
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7.utils.case_v7 import KIND_BOUNDARY, KIND_FLUID
    case = load_case_v7(str(_REPO_ROOT / meta["case"]))
    support = float(case.physics.smoothing_length)
    gamma = float(case.physics.power_parameter)
    gravity = np.array(case.physics.gravity[:2], dtype=np.float64)
    offset = float(meta["stored_density_offset"])
    kinds = np.array([material.kind for material in case.materials])
    rest_density = np.array([material.rest_density for material in case.materials], dtype=np.float64)
    eos_constant = np.array([material.eos_constant for material in case.materials], dtype=np.float64)

    material = state["material"].astype(np.int64)
    positions = state["position_voxel_id"][:, :2].astype(np.float64)
    velocities = state["velocity_mass"][:, :2].astype(np.float64)
    fluid = kinds[material] == KIND_FLUID
    walls = np.flatnonzero(kinds[material] == KIND_BOUNDARY)
    fluid_index = np.flatnonzero(fluid)
    fluid_density = state["density_pressure_scratch"][fluid_index, 0].astype(np.float64) + offset
    fluid_pressure = state["density_pressure_scratch"][fluid_index, 1].astype(np.float64)
    copy_mismatch = int(np.any(state["density_pressure_scratch"][fluid_index] != state["density_pressure"][fluid_index],
                               axis=1).sum())

    tree = cKDTree(positions[fluid_index])
    count = walls.size
    weight_sum = np.zeros(count)
    velocity_sum = np.zeros((count, 2))
    pressure_sum = np.zeros(count)
    absolute_pressure_sum = np.zeros(count)
    offset_sum = np.zeros((count, 2))
    neighbour_count = np.zeros(count, dtype=np.int64)
    for row, (wall, found) in enumerate(zip(walls, tree.query_ball_point(positions[walls], support))):
        if not found:
            continue
        found = np.asarray(found)
        difference = positions[wall] - positions[fluid_index[found]]          # r_wf = r_w - r_f
        distance = np.linalg.norm(difference, axis=1)
        keep = (distance < support) & (distance >= 1e-12)
        found, difference, distance = found[keep], difference[keep], distance[keep]
        weight = wendland_c4_2d(distance, support)
        neighbour_count[row] = found.size
        weight_sum[row] = weight.sum()
        velocity_sum[row] = (velocities[fluid_index[found]] * weight[:, None]).sum(axis=0)
        pressure_sum[row] = (fluid_pressure[found] * weight).sum()
        absolute_pressure_sum[row] = (np.abs(fluid_pressure[found]) * weight).sum()
        offset_sum[row] = (fluid_density[found, None] * difference * weight[:, None]).sum(axis=0)

    wall_material = material[walls]
    prescribed = velocities[walls]
    has_fluid = weight_sum > 0
    pressure = np.zeros(count)
    pressure[has_fluid] = (pressure_sum[has_fluid] + offset_sum[has_fluid] @ gravity) / weight_sum[has_fluid]
    base = 1.0 + pressure / eos_constant[wall_material]
    density = rest_density[wall_material] * np.maximum(base, 1.0e-3) ** (1.0 / gamma)
    density[~has_fluid] = rest_density[wall_material[~has_fluid]]
    if stores_rest_density:
        density = rest_density[wall_material].copy()
    dummy = prescribed.copy()
    dummy[has_fluid] = 2.0 * prescribed[has_fluid] - velocity_sum[has_fluid] / weight_sum[has_fluid, None]

    gpu_density = state["density_pressure"][walls, 0].astype(np.float64) + offset
    gpu_pressure = state["density_pressure"][walls, 1].astype(np.float64)
    gpu_dummy = state["wall_dummy_velocity"][walls, :2].astype(np.float64)
    gpu_weight_sum = state["wall_dummy_velocity"][walls, 3].astype(np.float64)
    gpu_has_fluid = gpu_weight_sum > 0
    # error scales: |p_w| can cancel to ~0, so its error is measured against sum |p_f| W / sum W (the size of the
    # terms the float32 sum adds); velocities against max(|u~|, U = 1)
    pressure_scale = np.where(has_fluid, absolute_pressure_sum / np.where(has_fluid, weight_sum, 1.0), 1.0)
    velocity_scale = np.maximum(np.abs(velocity_sum / np.where(has_fluid, weight_sum, 1.0)[:, None]).max(axis=1), 1.0)
    # a wall whose fluid neighbours all have P = 0 exactly (float32 rho = rho0) has scale 0: absolute error there
    pressure_error = np.abs(gpu_pressure - pressure) / np.where(pressure_scale > 0, pressure_scale, 1.0)
    density_error = np.abs(gpu_density - density) / density
    dummy_error = np.abs(gpu_dummy - dummy).max(axis=1) / velocity_scale
    weight_error = np.where(has_fluid, np.abs(gpu_weight_sum - weight_sum) / np.where(has_fluid, weight_sum, 1.0), 0.0)
    # tiny weight sums (a fluid particle at the edge of the support) are fine relatively but differ in float32 by the
    # edge rounding of the distance; report the bulk (sum W >= 1e-3 of the full-support value) separately
    full_support = wendland_c4_2d(np.array([0.0]), support)[0]
    bulk = has_fluid & (weight_sum >= 1e-3 * full_support)

    def summary(values: np.ndarray, mask: np.ndarray) -> dict:
        selected = values[mask]
        return {"max": float(selected.max()) if selected.size else 0.0,
                "p99": float(np.percentile(selected, 99)) if selected.size else 0.0,
                "median": float(np.median(selected)) if selected.size else 0.0,
                "max_over_float32_epsilon": float(selected.max() / FLOAT32_EPSILON) if selected.size else 0.0}

    report = {
        "dump": str(path), "wall_bc": meta["wall_bc"], "steps": meta["steps"], "time": meta["steps"] * meta["dt"],
        "wall_particles": int(count),
        "walls_with_fluid_neighbours_numpy": int(has_fluid.sum()),
        "walls_with_fluid_neighbours_gpu": int(gpu_has_fluid.sum()),
        "walls_with_fluid_neighbours_by_material": {
            case.materials[group].name if hasattr(case.materials[group], "name") else str(group):
                int((has_fluid & (wall_material == group)).sum()) for group in np.unique(wall_material)},
        "walls_with_fluid_neighbours_disagree": int((has_fluid != gpu_has_fluid).sum()),
        "fluid_neighbours_per_wall": {"min": int(neighbour_count[has_fluid].min()),
                                      "median": float(np.median(neighbour_count[has_fluid])),
                                      "max": int(neighbour_count[has_fluid].max())},
        "bulk_walls": int(bulk.sum()),
        "walls_with_zero_pressure_scale": int((has_fluid & (pressure_scale == 0)).sum()),
        "relative_error": {
            "pressure_over_sum_abs_p_W": {"all": summary(pressure_error, has_fluid), "bulk": summary(pressure_error, bulk)},
            "density": {"all": summary(density_error, has_fluid), "bulk": summary(density_error, bulk)},
            "dummy_velocity_over_max_u_1": {"all": summary(dummy_error, has_fluid), "bulk": summary(dummy_error, bulk)},
            "kernel_weight_sum": {"all": summary(weight_error, has_fluid), "bulk": summary(weight_error, bulk)},
        },
        "no_fluid_walls_exact": bool(np.all(gpu_pressure[~has_fluid & ~gpu_has_fluid] == 0.0)
                                     and np.all(gpu_density[~has_fluid & ~gpu_has_fluid]
                                                == rest_density[wall_material[~has_fluid & ~gpu_has_fluid]])
                                     and np.all(gpu_dummy[~has_fluid & ~gpu_has_fluid]
                                                == prescribed[~has_fluid & ~gpu_has_fluid])),
        "wall_density_floor_count_gpu": int(meta["status"].get("wall_density_floor_count", -1)),
        "base_below_floor_numpy": int((base[has_fluid] < 1.0e-3).sum()),
        "fluid_scratch_vs_primary_mismatch": copy_mismatch,
        "pressure_range_walls": [float(gpu_pressure[has_fluid].min()), float(gpu_pressure[has_fluid].max())],
        "density_range_walls": [float(gpu_density[has_fluid].min()), float(gpu_density[has_fluid].max())],
    }
    out = path.with_suffix(".check.json")
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
