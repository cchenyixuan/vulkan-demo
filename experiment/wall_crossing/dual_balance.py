"""
dual_balance.py - report section 27, 2.3: what each slab of a two-slab 3-D cavity run holds (host, initial particles).

    .venv/Scripts/python.exe -m experiment.wall_crossing.dual_balance

Per case (cavity3d_1m_crossing and its frame_shift_3 variant, section27_cases.py) and slab of the equal-weight
cut (partition_v7.compute_chain_cuts, the chain's own rule): its voxel columns, fluid particles, near-wall fluid
particles (voxels in the 3^d neighbourhood of a voxel holding a wall particle: the marker's rule, identical to rule 1
on these cases, report section 26), wall particles, and the own particles in the fused band (correction / density
width) and the force band (force width) next to the seam. Writes claude/wall_crossing/tables/dual_balance.md and
logs/wall_crossing/section27/dual_balance.json.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.run_log import LOG_ROOT     # noqa: E402

CASES = (("缺省", "cavity3d_1m_crossing"), ("frame 外移 3 层", "cavity3d_1m_crossing_frame_shift_3"))


def slab_counts(case_directory: str) -> dict:
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7.utils.case_v7 import KIND_BOUNDARY, KIND_FLUID
    from experiment.v7.utils.partition_v7 import compute_chain_cuts, configured_band_widths
    case = load_case_v7(REPOSITORY_ROOT / "cases" / case_directory / "case.yaml")
    cuts = list(compute_chain_cuts(case, [1.0, 1.0], 6))
    fused_band, _, force_band = configured_band_widths()
    support = np.float32(case.physics.smoothing_length)
    origin = np.array([case.grid.origin_x, case.grid.origin_y, case.grid.origin_z], dtype=np.float32)
    dimensions = np.array([case.grid.grid_dimension_x, case.grid.grid_dimension_y, case.grid.grid_dimension_z])
    positions = case.initial.positions.astype(np.float32)
    kinds = np.asarray([material.kind for material in case.materials])[case.initial.material_group.astype(np.int64)]
    coordinates = np.floor((positions - origin) / support).astype(np.int64)
    fluid, wall = kinds == KIND_FLUID, kinds == KIND_BOUNDARY
    voxel = coordinates[:, 1] + coordinates[:, 2] * dimensions[1] + coordinates[:, 0] * dimensions[1] * dimensions[2]
    total = int(np.prod(dimensions))
    holds_wall = np.zeros(total, dtype=bool)
    holds_wall[voxel[wall]] = True
    near = np.zeros(total, dtype=bool)
    wall_coordinates = coordinates[wall]
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                target = wall_coordinates + np.array([dx, dy, dz])
                inside = np.all((target >= 0) & (target < dimensions), axis=1)
                target = target[inside]
                near[target[:, 1] + target[:, 2] * dimensions[1] + target[:, 0] * dimensions[1] * dimensions[2]] = True
    near_particle = near[voxel]
    edges = [0] + cuts + [int(dimensions[0])]
    slabs = []
    for index in range(len(edges) - 1):
        low, high = edges[index], edges[index + 1]
        own = (coordinates[:, 0] >= low) & (coordinates[:, 0] < high)
        seam_low = low if index > 0 else None
        seam_high = high if index < len(edges) - 2 else None
        def band(width: int) -> np.ndarray:
            mask = np.zeros(own.size, dtype=bool)
            if seam_low is not None:
                mask |= own & (coordinates[:, 0] < seam_low + width)
            if seam_high is not None:
                mask |= own & (coordinates[:, 0] >= seam_high - width)
            return mask
        slabs.append({"columns": [low, high], "column_count": high - low, "fluid": int((own & fluid).sum()),
                      "near_wall_fluid": int((own & fluid & near_particle).sum()), "wall": int((own & wall).sum()),
                      "fused_band": int(band(fused_band).sum()), "force_band": int(band(force_band).sum()),
                      "fused_band_fluid": int((band(fused_band) & fluid).sum())})
    return {"case": case_directory, "cuts": cuts, "grid_x": int(dimensions[0]), "origin_x": float(case.grid.origin_x),
            "band_widths": [fused_band, force_band], "slabs": slabs}


def main() -> int:
    from experiment.wall_crossing.report_tables import markdown_table, write
    results, rows = [], []
    for label, case in CASES:
        result = slab_counts(case)
        results.append(result)
        for index, slab in enumerate(result["slabs"]):
            rows.append([label, f"{index}（{index} 号卡）", f"[{slab['columns'][0]}, {slab['columns'][1]})（{slab['column_count']} 列）",
                         f"{slab['fluid']:,}", f"{slab['near_wall_fluid']:,}", f"{slab['wall']:,}",
                         f"{slab['fused_band']:,}（流体 {slab['fused_band_fluid']:,}）", f"{slab['force_band']:,}"])
    out = LOG_ROOT / "section27"
    out.mkdir(parents=True, exist_ok=True)
    (out / "dual_balance.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    write("dual_balance.md", markdown_table(
        ["算例", "片", "体素列", "流体粒子", "近壁流体", "壁面粒子", "fused 分界带的粒子", "受力分界带的粒子"], rows))
    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
