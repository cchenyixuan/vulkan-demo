"""
e8_field_compare.py — E8 phase 1 / 2: per-particle comparison of a K-slab chain against a K = 1 run at the same step,
from seam-audit dumps (experiment/seam_audit/dump_state.py, keyed by the global particle id).

Per particle present in both dumps: |dv| / U (U = the lid speed, 1 m/s) and |drho| / rho0 (rho0 = the material's
reference density, 1000 kg/m^3), binned by the unsigned voxel-column distance d to the nearest seam of the test run
(analyze.py's rule: global column c = floor((x - origin_x) / h); for the nearest cut s, d = c - s when c >= s, else
s - 1 - c; bins d = 0..15 and "interior" (d >= 16)). Fluid particles only (material kind 0) unless --all. Per bin:
count, max and rms of both, and the position difference |dx|.

--reference2 (phase 2): a second K = 1 dump of the same case; |A1 - A2| on the same particles is the K = 1 run-to-run
noise floor (atomic summation order), reported per bin next to the test's |B - A1| with the rms ratio test / noise.
With a ``device`` array in the test dump (e8_dump_state.py) the per-slab table names the device of every slab.

Usage:
    python docs/cluster_v6/scripts/e8_field_compare.py --reference DUMP_K1.npz [--reference2 DUMP_K1b.npz] --test DUMP_K.npz
        [--all] [--lid-speed 1.0] [--rho0 1000.0] [--md OUT.md] [--json OUT.json]
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


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values ** 2))) if values.size else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="K = 1 dump A1 (npz)")
    parser.add_argument("--reference2", default=None, help="a second K = 1 dump A2 (npz): the noise floor |A1 - A2|")
    parser.add_argument("--test", required=True, help="K-slab dump B (npz) with its .json sidecar")
    parser.add_argument("--all", action="store_true", help="every particle, not only the fluid")
    parser.add_argument("--lid-speed", type=float, default=1.0)
    parser.add_argument("--rho0", type=float, default=1000.0)
    parser.add_argument("--md", default=None)
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()
    reference, reference_meta = load(pathlib.Path(arguments.reference))
    test, meta = load(pathlib.Path(arguments.test))
    reference2, reference2_meta = (load(pathlib.Path(arguments.reference2)) if arguments.reference2 else (None, {}))
    if not meta:
        raise SystemExit(f"{arguments.test}: no JSON sidecar next to the dump (cuts / origin / h are needed)")
    cuts = [int(cut) for cut in meta["cuts"]]
    origin_x, h, grid_nx = float(meta["origin_x"]), float(meta["smoothing_length"]), int(meta["grid_nx"])
    kinds = {index: kind for index, kind in enumerate(meta.get("material_kinds", []))}
    reference_ids, test_ids = reference["id"].astype(np.int64), test["id"].astype(np.int64)
    common, reference_index, test_index = np.intersect1d(reference_ids, test_ids, assume_unique=True, return_indices=True)
    missing_in_test = int(reference_ids.size - common.size)
    missing_in_reference = int(test_ids.size - common.size)
    reference2_index = None
    if reference2 is not None:
        ids2 = reference2["id"].astype(np.int64)
        common2, keep_common, reference2_index = np.intersect1d(common, ids2, assume_unique=True, return_indices=True)
        reference_index, test_index = reference_index[keep_common], test_index[keep_common]
    if not arguments.all:
        material = test["material"][test_index].astype(np.int64)
        groups = np.unique(material)
        fluid_groups = groups[np.array([kinds.get(int(group), 0) == FLUID_KIND for group in groups])]
        keep = np.isin(material, fluid_groups)
        reference_index, test_index = reference_index[keep], test_index[keep]
        if reference2_index is not None:
            reference2_index = reference2_index[keep]
    dv = np.linalg.norm(test["velocity"][test_index] - reference["velocity"][reference_index], axis=1) / arguments.lid_speed
    drho = np.abs(test["density"][test_index].astype(np.float64) - reference["density"][reference_index].astype(np.float64)) / arguments.rho0
    dx = np.linalg.norm(test["position"][test_index] - reference["position"][reference_index], axis=1)
    noise_dv = noise_drho = None
    if reference2_index is not None:
        noise_dv = np.linalg.norm(reference["velocity"][reference_index] - reference2["velocity"][reference2_index], axis=1) / arguments.lid_speed
        noise_drho = np.abs(reference["density"][reference_index].astype(np.float64)
                            - reference2["density"][reference2_index].astype(np.float64)) / arguments.rho0
    distance = seam_distance(test["position"][test_index, 0], cuts, origin_x, h, grid_nx)
    bins = np.minimum(distance, DISTANCE_BINS)
    rows = []
    for bin_index in range(DISTANCE_BINS + 1):
        selected = bins == bin_index
        count = int(selected.sum())
        if count == 0:
            continue
        row = {"bin": "interior" if bin_index == DISTANCE_BINS else bin_index, "count": count,
               "dv_max": float(dv[selected].max()), "dv_rms": rms(dv[selected]),
               "drho_max": float(drho[selected].max()), "drho_rms": rms(drho[selected]),
               "dx_max_m": float(dx[selected].max())}
        if noise_dv is not None:
            row.update({"noise_dv_max": float(noise_dv[selected].max()), "noise_dv_rms": rms(noise_dv[selected]),
                        "noise_drho_max": float(noise_drho[selected].max()), "noise_drho_rms": rms(noise_drho[selected])})
            row["dv_rms_ratio"] = row["dv_rms"] / row["noise_dv_rms"] if row["noise_dv_rms"] > 0 else None
            row["drho_rms_ratio"] = row["drho_rms"] / row["noise_drho_rms"] if row["noise_drho_rms"] > 0 else None
        rows.append(row)
    # per slab (the test's owner), with the device when the dump carries it
    slab_rows = []
    slabs = test["slab"][test_index].astype(np.int64)
    devices = test["device"][test_index].astype(np.int64) if "device" in test else None
    for slab in sorted(set(slabs.tolist())):
        selected = slabs == slab
        slab_rows.append({"slab": int(slab), "device": int(devices[selected][0]) if devices is not None else None,
                          "count": int(selected.sum()), "dv_max": float(dv[selected].max()), "dv_rms": rms(dv[selected]),
                          "drho_max": float(drho[selected].max()), "drho_rms": rms(drho[selected])})
    summary = {"reference": arguments.reference, "reference2": arguments.reference2, "test": arguments.test,
               "slabs": meta.get("slabs"), "cuts": cuts, "device_map": meta.get("device_map"), "horizon": meta.get("horizon"),
               "reference_horizon": reference_meta.get("horizon"), "reference2_horizon": reference2_meta.get("horizon"),
               "particles_compared": int(test_index.size), "fluid_only": not arguments.all,
               "missing_in_test": missing_in_test, "missing_in_reference": missing_in_reference,
               "dv_max_over_U": float(dv.max()) if dv.size else None, "drho_max_over_rho0": float(drho.max()) if drho.size else None,
               "dv_rms_over_U": rms(dv), "drho_rms_over_rho0": rms(drho), "dx_max_m": float(dx.max()) if dx.size else None,
               "lid_speed": arguments.lid_speed, "rho0": arguments.rho0, "bins": rows, "per_slab": slab_rows,
               "reference_valid": reference_meta.get("invariants", {}).get("valid", True),
               "reference2_valid": reference2_meta.get("invariants", {}).get("valid", True) if reference2 is not None else None,
               "test_valid": meta.get("invariants", {}).get("valid", True)}
    if noise_dv is not None:
        summary.update({"noise_dv_max_over_U": float(noise_dv.max()), "noise_dv_rms_over_U": rms(noise_dv),
                        "noise_drho_max_over_rho0": float(noise_drho.max()), "noise_drho_rms_over_rho0": rms(noise_drho),
                        "dv_rms_ratio": rms(dv) / rms(noise_dv) if rms(noise_dv) > 0 else None,
                        "drho_rms_ratio": rms(drho) / rms(noise_drho) if rms(noise_drho) > 0 else None})
    lines = [f"K = {meta.get('slabs')} (cuts {cuts}, devices {meta.get('device_map')}) vs K = 1 at step {meta.get('horizon')}: "
             f"{test_index.size:,} {'fluid ' if not arguments.all else ''}particles compared (ids missing in the test {missing_in_test}, "
             f"in the reference {missing_in_reference}); max |dv|/U = {summary['dv_max_over_U']:.3e}, rms {summary['dv_rms_over_U']:.3e}; "
             f"max |drho|/rho0 = {summary['drho_max_over_rho0']:.3e}, rms {summary['drho_rms_over_rho0']:.3e}; max |dx| {summary['dx_max_m']:.3e} m"]
    if noise_dv is not None:
        lines.append(f"noise floor |A1 - A2| (two K = 1 runs, same particles): max |dv|/U = {summary['noise_dv_max_over_U']:.3e}, "
                     f"rms {summary['noise_dv_rms_over_U']:.3e}; max |drho|/rho0 = {summary['noise_drho_max_over_rho0']:.3e}, rms "
                     f"{summary['noise_drho_rms_over_rho0']:.3e}; rms ratios test / noise dv {summary['dv_rms_ratio']:.2f}, "
                     f"drho {summary['drho_rms_ratio']:.2f}")
    lines.append("")
    if noise_dv is None:
        lines += ["| distance to the seam (columns) | particles | max dv/U | rms dv/U | max drho/rho0 | rms drho/rho0 | max dx (m) |",
                  "|---|---|---|---|---|---|---|"]
        for row in rows:
            lines.append(f"| {row['bin']} | {row['count']:,} | {row['dv_max']:.3e} | {row['dv_rms']:.3e} | {row['drho_max']:.3e} | "
                         f"{row['drho_rms']:.3e} | {row['dx_max_m']:.3e} |")
    else:
        lines += ["| distance to the seam (columns) | particles | max dv/U test / noise | rms dv/U test / noise (ratio) | "
                  "max drho/rho0 test / noise | rms drho/rho0 test / noise (ratio) | max dx (m) |", "|---|---|---|---|---|---|---|"]
        for row in rows:
            lines.append(f"| {row['bin']} | {row['count']:,} | {row['dv_max']:.3e} / {row['noise_dv_max']:.3e} | "
                         f"{row['dv_rms']:.3e} / {row['noise_dv_rms']:.3e} ({row['dv_rms_ratio']:.2f}) | "
                         f"{row['drho_max']:.3e} / {row['noise_drho_max']:.3e} | "
                         f"{row['drho_rms']:.3e} / {row['noise_drho_rms']:.3e} ({row['drho_rms_ratio']:.2f}) | {row['dx_max_m']:.3e} |")
    lines += ["", "| slab | device | particles | max dv/U | rms dv/U | max drho/rho0 | rms drho/rho0 |", "|---|---|---|---|---|---|---|"]
    for row in slab_rows:
        lines.append(f"| {row['slab']} | {row['device'] if row['device'] is not None else '-'} | {row['count']:,} | {row['dv_max']:.3e} | "
                     f"{row['dv_rms']:.3e} | {row['drho_max']:.3e} | {row['drho_rms']:.3e} |")
    text = "\n".join(lines)
    print(text)
    print(f"[e8_field_compare] RESULT {json.dumps({key: value for key, value in summary.items() if key not in ('bins', 'per_slab')})}")
    if arguments.md:
        pathlib.Path(arguments.md).write_text(text + "\n", encoding="utf-8")
    if arguments.json:
        pathlib.Path(arguments.json).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
