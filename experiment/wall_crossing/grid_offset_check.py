"""
grid_offset_check.py - report section 21: where the voxel boundaries sit against the particle layers of a case, and
the grid origin shift case_loader_v7 applies (case_loader_v7._place_grid_origin).

    .venv/Scripts/python.exe -m experiment.wall_crossing.grid_offset_check [CASE.yaml ...]

Loads each case (default: the three of section 21) and writes claude/wall_crossing/tables/grid_offset_cases.md and
claude/wall_crossing/grid_offset/cases.json: per axis the smallest distance of an initial particle (fluid and walls)
to a voxel boundary before and after, in particle spacings, the shift, the voxels added, and the origin and voxel
counts before and after.
"""
from __future__ import annotations

import json
import pathlib
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.report_tables import REPORT_ROOT, markdown_table, write     # noqa: E402

CASES = ("cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025_crossing/case.yaml",
         "cases/cavity3d_1m_crossing/case.yaml",
         "cases/wall_crossing/hydrostatic_3d_n48_gtilt_crossing/case.yaml")


def main() -> int:
    from experiment.v7.utils import case_loader_v7
    cases = sys.argv[1:] or list(CASES)
    rows, records = [], []
    for case in cases:
        loaded = case_loader_v7.load_case_v7(REPOSITORY_ROOT / case)
        placement = json.loads(json.dumps(case_loader_v7.LAST_GRID_PLACEMENT))
        placement["case"] = case
        placement["particles"] = int(loaded.initial.positions.shape[0])
        records.append(placement)
        for axis, item in placement["axes"].items():
            index = "xyz".index(axis)
            rows.append([pathlib.Path(case).parent.name, axis, f"{item['clearance_before']:.4f}",
                         f"{item['clearance_after']:.4f}", f"{item['shift']:.4f}", str(item["voxels_added"]),
                         f"{placement['origin_before'][index]:.6f} → {placement['origin_after'][index]:.6f}",
                         f"{placement['dims_before'][index]} → {placement['dims_after'][index]}",
                         f"{placement['smoothing_length'] / placement['particle_spacing']:.2f}"])
    write("grid_offset_cases.md", markdown_table(
        ["算例", "方向", "改前：离体素边界最近（dx）", "改后（dx）", "δ（dx，向负方向）", "加的体素", "原点（m）",
         "体素数", "h / dx"], rows))
    target = REPORT_ROOT / "grid_offset"
    target.mkdir(parents=True, exist_ok=True)
    (target / "cases.json").write_text(json.dumps(records, indent=1), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
