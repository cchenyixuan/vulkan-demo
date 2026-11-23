"""
lanes_rule_inputs.py - the inputs and the choice of wall_boundary crossing's near-wall lanes rule
(simulator_v7.SphSimulatorV7._near_wall_lanes; claude/wall_store_report.md sections 10.6 and 11) for a list of
crossing cases, on the host only (no GPU): per slab, the fluid particles in the wall store's near-wall voxels at the
start, the near-wall warps per SM, and k. The machine is given (default: one RTX 5090, 170 SMs x 48 warps).

    .venv/Scripts/python.exe -m experiment.wall_crossing.lanes_rule_inputs --case cases/cavity3d_n50_crossing/case.yaml \
        --case cases/cavity3d_n70_crossing/case.yaml --slabs 1
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


def rule_inputs(case_path: str, slabs: int, streaming_multiprocessors: int, warps_per_streaming_multiprocessor: int,
                lanes: int = 4, pool_safety: float = 1.2) -> list:
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7.utils.partition_v7 import compute_chain_partition
    from experiment.v7.utils.wall_store_v7 import _voxel_coordinates, _voxel_ids, layout_for_grid
    case = load_case_v7(case_path)
    chain = compute_chain_partition(case, [1.0] * slabs, pool_safety)
    wave = warps_per_streaming_multiprocessor * 32 * streaming_multiprocessors
    result = []
    for slab in chain.slabs:
        layout = layout_for_grid(slab.wall_source, slab.grid)
        positions = np.asarray(slab.initial.positions, dtype=np.float64)
        coordinates = _voxel_coordinates(positions, slab.grid, float(slab.wall_source.smoothing_length))
        near_wall = int((layout.near_wall_flags[_voxel_ids(coordinates, slab.grid)] != 0).sum())
        result.append({"fluid": int(positions.shape[0]), "near_wall_fluid": near_wall,
                       "near_wall_warps_per_sm": near_wall / 32 / streaming_multiprocessors,
                       "rule_k": lanes if lanes * near_wall <= wave else 1, "wave": wave})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--slabs", type=int, default=1)
    parser.add_argument("--sms", type=int, default=170)
    parser.add_argument("--warps-per-sm", type=int, default=48)
    parser.add_argument("--out", default="")
    arguments = parser.parse_args()
    table = {}
    for case_path in arguments.case:
        table[case_path] = rule_inputs(case_path, arguments.slabs, arguments.sms, arguments.warps_per_sm)
        for index, entry in enumerate(table[case_path]):
            print(f"{case_path} slab {index}: fluid {entry['fluid']:,}, near-wall {entry['near_wall_fluid']:,} "
                  f"({entry['near_wall_warps_per_sm']:.1f} warps / SM), rule k = {entry['rule_k']}", flush=True)
    if arguments.out:
        pathlib.Path(arguments.out).write_text(json.dumps(table, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
