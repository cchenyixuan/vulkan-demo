"""cavity_k8_partition_check.py (CPU only): the chain partition the runner will build for --start-from at K slabs:
the case loaded, the checkpoint's positions / materials put in place of the rest state (exactly as cavity_runner does),
compute_chain_partition with equal weights -> the cut columns, the seam x positions, every slab's particle count and
own columns, the per-column fluid histogram's total. Also checks the checkpoint against the case (row count, fields,
dt of the source run).

    python docs/validation/cluster/cavity_k8_partition_check.py --case cases/.../case.yaml --checkpoint <c<step>.npz> --slabs 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--slabs", type=int, default=8)
    parser.add_argument("--pool-safety", type=float, default=1.2)
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7.utils.case_v7 import InitialParticles
    from experiment.v7.utils.partition_v7 import _bin_fluid_counts, compute_chain_partition, restart_slab_rows
    from experiment.v7.utils.simulator_v7 import SphSimulatorV7

    case = load_case_v7(arguments.case)
    expected_total = int(case.initial.positions.shape[0])
    dt = float(case.physics.timestep)
    support = float(case.physics.smoothing_length)
    path = pathlib.Path(arguments.checkpoint).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path) as archive:
        rows = {name: np.array(archive[name]) for name in archive.files if not name.startswith("_")}
        offset = float(archive["_stored_density_offset"])
    layout = SphSimulatorV7.RESTART_FIELD_LAYOUT
    assert set(rows) == set(layout), (sorted(rows), sorted(layout))
    count = int(rows["position_voxel_id"].shape[0])
    assert count == expected_total, (count, expected_total)
    for name, (components, element_type) in layout.items():
        shape = (count,) if components == 1 else (count, components)
        assert rows[name].shape == shape and rows[name].dtype == element_type, (name, rows[name].shape, rows[name].dtype)
    source_meta = json.loads((path.parent.parent / "meta.json").read_text(encoding="utf-8"))
    assert abs(float(source_meta["dt"]) - dt) < 1e-15, (source_meta["dt"], dt)
    step = int(path.stem[1:])
    rest_histogram = _bin_fluid_counts(case)
    case.initial = InitialParticles(positions=np.ascontiguousarray(rows["position_voxel_id"][:, :3], dtype=np.float32),
                                    velocities=np.ascontiguousarray(rows["velocity_mass"][:, :3], dtype=np.float32),
                                    material_group=np.ascontiguousarray(rows["material"], dtype=np.uint32))
    state_histogram = _bin_fluid_counts(case)
    weights = [1.0] * arguments.slabs
    chain = compute_chain_partition(case, weights, arguments.pool_safety)
    rest_chain = None
    per_slab = restart_slab_rows(case, chain, rows["position_voxel_id"][:, 0])
    cuts = list(chain.cuts)
    result = {"checkpoint": str(path), "sha256": digest, "step": step, "time": step * dt, "rows": count,
              "stored_density_offset": offset, "grid_nx": int(case.grid.grid_dimension_x), "origin_x": float(case.grid.origin_x),
              "support": support, "slabs": arguments.slabs, "cuts": cuts,
              "cuts_x": [float(case.grid.origin_x + cut * support) for cut in cuts],
              "cuts_x_wall_frame": [float(case.grid.origin_x + cut * support) for cut in cuts],
              "own_columns": [int(geometry.own_column_count) for geometry in chain.geometry],
              "own_particles": [int(geometry.own_particle_count) for geometry in chain.geometry],
              "own_pool": [int(geometry.own_pool_size) for geometry in chain.geometry],
              "restart_rows_per_slab": [int(part.size) for part in per_slab],
              "fluid_total_rest": int(rest_histogram.sum()), "fluid_total_state": int(state_histogram.sum()),
              "columns_changed_between_rest_and_state": int(np.count_nonzero(rest_histogram != state_histogram)),
              "max_column_difference": int(np.abs(rest_histogram - state_histogram).max()),
              "ghost_layers": int(chain.geometry[0].ghost_layers)}
    print(json.dumps(result, indent=1))
    if arguments.json:
        pathlib.Path(arguments.json).write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
