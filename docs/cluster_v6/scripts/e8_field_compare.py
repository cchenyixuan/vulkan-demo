"""
e8_field_compare.py — E8 phase 1 items 1 / 2: per-particle comparison of a K-slab chain against a K = 1 run at the
same step, from seam-audit dumps (experiment/seam_audit/dump_state.py, keyed by the global particle id).

Per particle present in both dumps: |dv| / U (U = the lid speed, 1 m/s) and |drho| / rho0 (rho0 = the material's
reference density, 1000 kg/m^3), binned by the unsigned voxel-column distance d to the nearest seam of the test run
(analyze.py's rule: global column c = floor((x - origin_x) / h); for the nearest cut s, d = c - s when c >= s, else
s - 1 - c; bins d = 0..15 and "interior" (d >= 16)). Fluid particles only (material kind 0) unless --all. Per bin:
count, max and rms of both, and the position difference |dx| / dx_lattice (should be of the same order as |dv| dt).
The K = 1 dump's own run-to-run noise is NOT measured here (analyze.py does that with two references).

Usage:
    python docs/cluster_v6/scripts/e8_field_compare.py --reference DUMP_K1.npz --test DUMP_K2.npz [--all]
        [--lid-speed 1.0] [--rho0 1000.0] [--md OUT.md] [--json OUT.json]
(the test dump's JSON sidecar, <stem>.json, must sit next to it: it carries the cuts, origin_x and h)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

DISTANCE_BINS = 16
FLUID_KIND = 0


def load(path: pathlib.Path) -> tuple[dict, dict]:
    arrays = dict(np.load(path))
    sidecar = path.with_suffix(".json")
    meta = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    return arrays, meta


def seam_distance(x: np.ndarray, cuts: list[int], origin_x: float, h: float, grid_nx: int) -> np.ndarray:
    columns = np.clip(np.floor((x - origin_x) / h).astype(np.int64), 0, grid_nx - 1)
    if not cuts:
        return np.full(columns.shape, DISTANCE_BINS, dtype=np.int64)
    distance = np.full(columns.shape, 10 ** 9, dtype=np.int64)
    for cut in cuts:
        candidate = np.where(columns >= cut, columns - cut, cut - 1 - columns)
        distance = np.minimum(distance, candidate)
    return distance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="K = 1 dump (npz)")
    parser.add_argument("--test", required=True, help="K-slab dump (npz) with its .json sidecar")
    parser.add_argument("--all", action="store_true", help="every particle, not only the fluid")
    parser.add_argument("--lid-speed", type=float, default=1.0)
    parser.add_argument("--rho0", type=float, default=1000.0)
    parser.add_argument("--md", default=None)
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()
    reference, reference_meta = load(pathlib.Path(arguments.reference))
    test, meta = load(pathlib.Path(arguments.test))
    if not meta:
        raise SystemExit(f"{arguments.test}: no JSON sidecar next to the dump (cuts / origin / h are needed)")
    cuts = [int(cut) for cut in meta["cuts"]]
    origin_x, h, grid_nx = float(meta["origin_x"]), float(meta["smoothing_length"]), int(meta["grid_nx"])
    kinds = {index: kind for index, kind in enumerate(meta.get("material_kinds", []))}
    reference_ids, test_ids = reference["id"].astype(np.int64), test["id"].astype(np.int64)
    common, reference_index, test_index = np.intersect1d(reference_ids, test_ids, assume_unique=True, return_indices=True)
    missing_in_test = int(reference_ids.size - common.size)
    missing_in_reference = int(test_ids.size - common.size)
    if not arguments.all:
        material = test["material"][test_index].astype(np.int64)
        fluid = np.array([kinds.get(int(group), 0) == FLUID_KIND for group in np.unique(material)])
        fluid_groups = np.unique(material)[fluid]
        keep = np.isin(material, fluid_groups)
        reference_index, test_index = reference_index[keep], test_index[keep]
    velocity_difference = np.linalg.norm(test["velocity"][test_index] - reference["velocity"][reference_index], axis=1)
    density_difference = np.abs(test["density"][test_index].astype(np.float64) - reference["density"][reference_index].astype(np.float64))
    position_difference = np.linalg.norm(test["position"][test_index] - reference["position"][reference_index], axis=1)
    dv = velocity_difference / arguments.lid_speed
    drho = density_difference / arguments.rho0
    distance = seam_distance(test["position"][test_index, 0], cuts, origin_x, h, grid_nx)
    bins = np.minimum(distance, DISTANCE_BINS)
    rows = []
    for bin_index in range(DISTANCE_BINS + 1):
        selected = bins == bin_index
        count = int(selected.sum())
        if count == 0:
            continue
        rows.append({"bin": "interior" if bin_index == DISTANCE_BINS else bin_index, "count": count,
                     "dv_max": float(dv[selected].max()), "dv_rms": float(np.sqrt(np.mean(dv[selected] ** 2))),
                     "drho_max": float(drho[selected].max()), "drho_rms": float(np.sqrt(np.mean(drho[selected] ** 2))),
                     "dx_max_m": float(position_difference[selected].max())})
    summary = {"reference": arguments.reference, "test": arguments.test, "slabs": meta.get("slabs"), "cuts": cuts,
               "horizon": meta.get("horizon"), "reference_horizon": reference_meta.get("horizon"),
               "particles_compared": int(test_index.size), "fluid_only": not arguments.all,
               "missing_in_test": missing_in_test, "missing_in_reference": missing_in_reference,
               "dv_max_over_U": float(dv.max()) if dv.size else None, "drho_max_over_rho0": float(drho.max()) if drho.size else None,
               "dv_rms_over_U": float(np.sqrt(np.mean(dv ** 2))) if dv.size else None,
               "drho_rms_over_rho0": float(np.sqrt(np.mean(drho ** 2))) if drho.size else None,
               "lid_speed": arguments.lid_speed, "rho0": arguments.rho0, "bins": rows,
               "reference_valid": all(item.get("valid", True) for item in [reference_meta.get("invariants", {})]),
               "test_valid": all(item.get("valid", True) for item in [meta.get("invariants", {})])}
    lines = [f"K = {meta.get('slabs')} (cuts {cuts}) vs K = 1 at step {meta.get('horizon')}: {test_index.size:,} "
             f"{'fluid ' if not arguments.all else ''}particles compared (ids missing in the test {missing_in_test}, "
             f"in the reference {missing_in_reference}); max |dv|/U = {summary['dv_max_over_U']:.3e}, "
             f"max |drho|/rho0 = {summary['drho_max_over_rho0']:.3e}, rms |dv|/U = {summary['dv_rms_over_U']:.3e}, "
             f"rms |drho|/rho0 = {summary['drho_rms_over_rho0']:.3e}", "",
             "| distance to the seam (columns) | particles | max dv/U | rms dv/U | max drho/rho0 | rms drho/rho0 | max dx (m) |",
             "|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['bin']} | {row['count']:,} | {row['dv_max']:.3e} | {row['dv_rms']:.3e} | {row['drho_max']:.3e} | "
                     f"{row['drho_rms']:.3e} | {row['dx_max_m']:.3e} |")
    text = "\n".join(lines)
    print(text)
    print(f"[e8_field_compare] RESULT {json.dumps({key: value for key, value in summary.items() if key != 'bins'})}")
    if arguments.md:
        pathlib.Path(arguments.md).write_text(text + "\n", encoding="utf-8")
    if arguments.json:
        pathlib.Path(arguments.json).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
