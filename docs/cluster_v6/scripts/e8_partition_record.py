"""
e8_partition_record.py — E8 phase 2 (user 2026-10-22): the partition and calibration record of one point, written next
to its weights file after the calibration (e7_lib.sh E7_PARTITION_RECORD=1):
  - the per-column fluid histogram of the case (partition_v7._bin_fluid_counts, the histogram the cut rule splits),
  - the weights file's rounds (per round: cuts, per-slab fluid / all particles, per-device busy, omega, next cuts),
  - the final weights and cuts, and per slab: own columns, own particles (fluid + walls), fluid particles, wall
    particles, the shares of the total (particles and fluid), and the device of the slab.
CPU only (the case is read through the job's .obj cache; the chain partition is computed as the bench computes it,
without any Vulkan context).

Usage:
    python docs/cluster_v6/scripts/e8_partition_record.py --obj-cache DIR --case CASE.yaml --weights-file FILE
        --device-map 0,1,... --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[3]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj-cache", default=None)
    parser.add_argument("--case", required=True)
    parser.add_argument("--weights-file", required=True)
    parser.add_argument("--device-map", required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    if arguments.obj_cache:
        import run_chain_v6
        run_chain_v6.install_obj_cache(arguments.obj_cache, "v7")
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7.utils.partition_v7 import (MINIMUM_OWN_COLUMNS_HARD, _bin_fluid_counts, chain_cuts_from_counts,
                                                  compute_chain_partition)
    record = json.loads(pathlib.Path(arguments.weights_file).read_text(encoding="utf-8"))
    weights = [float(value) for value in record["weights"]]
    device_map = [int(item) for item in arguments.device_map.split(",") if item.strip()]
    case = load_case_v7(arguments.case)
    histogram = _bin_fluid_counts(case)
    cuts = chain_cuts_from_counts(histogram, weights, MINIMUM_OWN_COLUMNS_HARD)
    if cuts != [int(cut) for cut in record["cuts"]]:
        print(f"[partition_record] WARNING: recomputed cuts {cuts} differ from the weights file's {record['cuts']}", flush=True)
    chain = compute_chain_partition(case, weights, 1.2)
    boundaries = [0] + cuts + [int(histogram.size)]
    total = int(case.initial.positions.shape[0])
    fluid_total = int(histogram.sum())
    slabs = []
    for index, geometry in enumerate(chain.geometry):
        first, last = boundaries[index], boundaries[index + 1]
        fluid = int(histogram[first:last].sum())
        own = int(geometry.own_particle_count)
        slabs.append({"slab": index, "device": device_map[index % len(device_map)], "first_column": first,
                      "last_column": last - 1, "own_columns": last - first,
                      "fluid_columns": int(np.count_nonzero(histogram[first:last])), "own_particles": own,
                      "fluid_particles": fluid, "wall_particles": own - fluid, "share_of_total": own / total,
                      "share_of_fluid": fluid / fluid_total, "weight": weights[index], "own_pool": int(geometry.own_pool_size)})
    document = {"case": arguments.case, "weights_file": arguments.weights_file, "weights_file_format": record.get("format"),
                "host": record.get("host"), "device_map": device_map, "slab_count": len(weights), "weights": weights,
                "cuts": cuts, "grid_columns": int(histogram.size), "fluid_histogram": [int(value) for value in histogram],
                "total_particles": total, "fluid_total": fluid_total, "rounds": record.get("rounds"),
                "settings": record.get("settings"), "slabs": slabs}
    pathlib.Path(arguments.out).write_text(json.dumps(document, indent=1), encoding="utf-8")
    print(f"[partition_record] {arguments.out}: K={len(weights)} cuts={cuts} own=" + ", ".join(f"{slab['own_particles']:,}" for slab in slabs)
          + " fluid=" + ", ".join(f"{slab['fluid_particles']:,}" for slab in slabs)
          + " walls=" + ", ".join(f"{slab['wall_particles']:,}" for slab in slabs), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
