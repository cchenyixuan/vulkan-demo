"""compare_dumps.py - bitwise comparison of two k1_dump.py states (same case, same step count), particle by particle
through the global id.

For every field present in both dumps: the particles whose every component has the same bits, and the largest
absolute difference. density_pressure_scratch is compared only where the solver writes it every step (all
particles for v6 and v7 with WALL_BC = 0; it is a transient copy of density_pressure). Exit code 0 when every
compared field is bit-identical for every particle, 1 otherwise.

    .venv/Scripts/python.exe -m experiment.v7.wall_bc.compare_dumps logs/e36/equivalence/v6_a.npz logs/e36/equivalence/v7_bc0_a.npz
"""
from __future__ import annotations

import json
import sys

import numpy as np

COMPARED = ("position_voxel_id", "velocity_mass", "density_pressure", "density_pressure_scratch", "acceleration",
            "shift", "material", "correction_inverse", "density_gradient_kernel_sum", "extension_fields")


def load(path: str) -> tuple[dict, dict]:
    with np.load(path) as archive:
        state = {name: archive[name] for name in archive.files if name != "meta"}
        meta = json.loads(str(archive["meta"]))
    return state, meta


def compare(first_path: str, second_path: str) -> dict:
    first, first_meta = load(first_path)
    second, second_meta = load(second_path)
    if not np.array_equal(first["global_id"], second["global_id"]):
        raise SystemExit("the dumps hold different particle sets")
    report = {"first": first_meta["solver"] + (f" WALL_BC={first_meta['wall_bc']}" if first_meta["wall_bc"] is not None else ""),
              "second": second_meta["solver"] + (f" WALL_BC={second_meta['wall_bc']}" if second_meta["wall_bc"] is not None else ""),
              "steps": (first_meta["steps"], second_meta["steps"]), "particles": int(first["global_id"].size), "fields": {}}
    for name in COMPARED:
        a, b = first[name], second[name]
        a_bits = a.view(np.uint32).reshape(a.shape[0], -1)
        b_bits = b.view(np.uint32).reshape(b.shape[0], -1)
        same = np.all(a_bits == b_bits, axis=1)
        difference = (np.abs(a.astype(np.float64) - b.astype(np.float64)).max()
                      if a.dtype != np.uint32 else float(np.abs(a.astype(np.int64) - b.astype(np.int64)).max()))
        report["fields"][name] = {"identical_particles": int(same.sum()), "max_abs_difference": float(difference)}
    report["bit_identical"] = all(entry["identical_particles"] == report["particles"]
                                  for entry in report["fields"].values())
    return report


def main() -> int:
    report = compare(sys.argv[1], sys.argv[2])
    print(f"{report['first']} vs {report['second']}, {report['steps'][0]} steps, {report['particles']} particles")
    for name, entry in report["fields"].items():
        print(f"  {name:30s} identical {entry['identical_particles']:6d} / {report['particles']}"
              f"   max |diff| {entry['max_abs_difference']:.3e}")
    print("BIT-IDENTICAL" if report["bit_identical"] else "NOT bit-identical")
    return 0 if report["bit_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
