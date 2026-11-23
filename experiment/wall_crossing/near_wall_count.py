"""
near_wall_count.py - who runs which dispatch in the wall store's organisation (B) (claude/wall_store_report.md section
9.5), counted on the host from a case's initial particles (one slab): fluid particles, the near-wall voxels of the wall
store (wall_store_v7.near_wall_flags, the flags the near-wall pipelines test) and the fluid particles in them (the
near-wall dispatch's working threads), the wall records kept; and for simple the particles of its pool (fluid + walls).
CPU only.

    .venv/Scripts/python.exe -m experiment.wall_crossing.near_wall_count

Writes claude/wall_store/tables/round3_near_wall_count.md and logs/wall_store/round3/near_wall_count.json.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

CASES = (("2-D 250²", "lid_driven_cavity_2d_n250_xi0p001_eps0p0025"),
         ("2-D 500²", "lid_driven_cavity_2d_n500_xi0p001_eps0p0025"),
         ("2-D 1000²", "lid_driven_cavity_2d_n1000_xi0p001_eps0p0025"), ("3-D 1M", "cavity3d_1m"))


def main() -> int:
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7.utils.partition_v7 import compute_chain_partition
    from experiment.v7.utils.wall_store_v7 import _voxel_coordinates, _voxel_ids, layout_for_grid
    rows, result = [], {}
    for text, base in CASES:
        simple = load_case_v7(REPOSITORY_ROOT / "cases" / base / "case.yaml")
        kinds = np.asarray([material.kind for material in simple.materials])
        simple_kinds = kinds[np.asarray(simple.initial.material_group, dtype=np.int64)]
        crossing = load_case_v7(REPOSITORY_ROOT / "cases" / f"{base}_crossing" / "case.yaml")
        slab = compute_chain_partition(crossing, [1.0], 1.2).slabs[0]
        layout = layout_for_grid(slab.wall_source, slab.grid)
        positions = np.asarray(crossing.initial.positions, dtype=np.float64)
        coordinates = _voxel_coordinates(positions, slab.grid, float(slab.wall_source.smoothing_length))
        near_wall = layout.near_wall_flags[_voxel_ids(coordinates, slab.grid)] != 0
        total_voxels = slab.grid.grid_dimension_x * slab.grid.grid_dimension_y * slab.grid.grid_dimension_z
        entry = {"simple_pool_particles": int(simple.initial.positions.shape[0]),
                 "simple_fluid": None if simple_kinds is None else int((simple_kinds == 0).sum()),
                 "simple_walls": None if simple_kinds is None else int((simple_kinds == 1).sum()),
                 "crossing_fluid": int(positions.shape[0]), "near_wall_fluid": int(near_wall.sum()),
                 "near_wall_voxels": int(layout.near_wall_voxel_count), "voxels": int(total_voxels),
                 "wall_records": int(slab.wall_source.count),
                 "own_pool_threads": int(slab.capacities.own_pool_size)}
        result[text] = entry
        rows.append(f"| {text} | {entry['simple_pool_particles']:,} | {entry['crossing_fluid']:,} | "
                    f"{entry['near_wall_fluid']:,}（{100 * entry['near_wall_fluid'] / entry['crossing_fluid']:.2f} %） | "
                    f"{entry['near_wall_voxels']:,} / {entry['voxels']:,} | {entry['wall_records']:,} | "
                    f"{entry['own_pool_threads']:,} |")
        print(text, entry, flush=True)
    table = ("| 算例 | simple 的池（流体 + 壁面） | crossing 的流体 | 近壁流体（近壁调度真正干活的线程） | 近壁体素 / 体素 | "
             "壁面记录 | 每次调度的线程（自有池） |\n|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
    (REPOSITORY_ROOT / "claude" / "wall_store" / "tables" / "round3_near_wall_count.md").write_text(
        table, encoding="utf-8", newline="\n")
    (REPOSITORY_ROOT / "logs" / "wall_store" / "round3" / "near_wall_count.json").write_text(
        json.dumps(result, indent=1), encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
