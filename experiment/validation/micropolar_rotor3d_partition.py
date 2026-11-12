"""micropolar_rotor3d_partition.py <case.yaml> <K>: the chain partition of a 3-D rotor case for K slabs along x
(no GPU): particles, wall share and own columns per slab, and the particles in the three columns on each side of
every cut (what the 3-layer ghost of an adami chain carries). Prints one line: PARTITION <json>."""
import json
import os
import pathlib
import sys

import numpy as np

root = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
os.chdir(root)
from experiment.validation import micropolar_tests as tests   # noqa: E402

case_path = root / sys.argv[1]
slab_count = int(sys.argv[2])
solver = tests._import_solver(root)
load_case, compute_chain_partition = solver[0], solver[1]
from experiment.v7.utils.case_v7 import KIND_FLUID   # noqa: E402
case = load_case(case_path)
chain = compute_chain_partition(case, [1.0] * slab_count, pool_safety=1.2)
support = float(case.physics.smoothing_length)
positions = case.initial.positions
kinds = np.asarray([material.kind for material in case.materials])[case.initial.material_group.astype(np.int64)]
column = np.floor((positions[:, 0] - case.grid.origin_x) / support).astype(np.int64)
per_column = np.bincount(column)
fluid_per_column = np.bincount(column[kinds == KIND_FLUID], minlength=per_column.size)
result = {"case": sys.argv[1], "slabs": slab_count, "columns": int(per_column.size), "slab": [], "cut": []}
for index, (slab, geometry) in enumerate(zip(chain.slabs, chain.geometry)):
    first = int(geometry.own_global_first_column)
    last = int(chain.geometry[index + 1].own_global_first_column) if index + 1 < slab_count else int(per_column.size)
    total = int(per_column[first:last].sum())
    fluid = int(fluid_per_column[first:last].sum())
    result["slab"].append({"first_column": first, "columns": last - first, "particles": total, "fluid": fluid,
                           "wall_share": 1.0 - fluid / total})
    if index + 1 < slab_count:
        result["cut"].append({"x": float(case.grid.origin_x + last * support),
                              "three_columns_left": int(per_column[last - 3:last].sum()),
                              "three_columns_right": int(per_column[last:last + 3].sum())})
print("PARTITION " + json.dumps(result))
