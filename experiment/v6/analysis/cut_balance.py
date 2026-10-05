"""
cut_balance.py — E31 (a): per-slab balance of the chain cuts, pre-E31 rule (searchsorted side="left": the
slab left of a cut always ends before the first column whose prefix reaches the target) against the E31 rule
(partition_v6.nearest_cut: the column boundary nearest the target). CPU only.

  --e29       the 9 E29 cases at K = 2: per slab the own column range, own columns, columns holding fluid,
              fluid particles, all particles and n_B at the initial state (own particles outside the force
              band, i.e. the phase B force sweep), for both rules; E29's measured n_B (warmup boundary, pre-E31
              rule) from the scan's run_meta next to it.
  --cluster   the N56 cluster cases (2-D 4M-128M, 3-D stretched 32M / 64M, cube 401^3) at K = 2 / 4 / 8: fluid
              particles per slab for both rules, max / mean - 1. The fluid histogram comes from the generator
              lattice (utils/geometry/_demo_cavity_case*.py: every lattice column holds the same number of fluid
              particles; positions and the frame as written with %.6f, h as written in case.yaml, binned in
              float32 like partition_v6._bin_fluid_counts), so no particle files are needed. --validate checks
              that histogram against the loader on every local case that has its .obj files.

    .venv/Scripts/python.exe -m experiment.v6.analysis.cut_balance --e29 --cluster --validate \\
        --out docs/perf_model/e31
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.v6.utils.partition_v6 import (  # noqa: E402
    MINIMUM_OWN_COLUMNS_HARD,
    chain_cuts_from_counts,
    configured_band_widths,
)

E29_CASES = {   # name: (case yaml, E29 scan trace of the K = 2 run with the trace on)
    "2d_10k": "logs/two_hop_experiment/cases/cavity_2d_10k/case.yaml",
    "2d_62k": "cases/cavity_2d_62k/case.yaml",
    "2d_250k": "cases/cavity_2d_250k/case.yaml",
    "2d_1m": "cases/lid_driven_cavity_2d_gen/case.yaml",
    "2d_4m": "cases/lid_driven_cavity_2d_4m/case.yaml",
    "2d_16m": "cases/lid_driven_cavity_2d_16m/case.yaml",
    "3d_1m": "cases/cavity3d_1m/case.yaml",
    "3d_narrow": "cases/cavity3d_narrow/case.yaml",
    "3d_8m": "cases/cavity3d_8m/case.yaml",
}
E29_SCAN = "logs/e29_step_trace/scan/runs"
# Generator lattices: (dimension, half, half_x, border, h/dx). 2-D: _demo_cavity_case.py (border 11);
# 3-D: _demo_cavity_case_3d.py (the cluster families use border 4, the older local cubes 9).
CLUSTER_CASES = {
    "2d_4m": (2, 1000, 1000, 11, 5.0),
    "2d_8m": (2, 1414, 1414, 11, 5.0),
    "2d_16m": (2, 2000, 2000, 11, 5.0),
    "2d_32m": (2, 2828, 2828, 11, 5.0),
    "2d_64m": (2, 4000, 4000, 11, 5.0),
    "2d_128m": (2, 5656, 5656, 11, 5.0),
    "3d_stretched_32m": (3, 79, 79 * 8 + 4, 4, 4.0),      # cavity3d_weak4_k8_32m (gen_3d_families.sh)
    "3d_stretched_64m": (3, 100, 100 * 8 + 4, 4, 4.0),    # cavity3d_weak8_k8_64m
    "3d_cube_401": (3, 200, 200, 4, 4.0),                  # cavity3d_cube_64m
}
VALIDATION_CASES = {   # local case with .obj files -> its generator lattice
    "cases/lid_driven_cavity_2d_gen/case.yaml": (2, 500, 500, 11, 5.0),
    "cases/cavity_2d_62k/case.yaml": (2, 124, 124, 11, 5.0),
    "cases/cavity_2d_250k/case.yaml": (2, 250, 250, 11, 5.0),
    "cases/lid_driven_cavity_2d_4m/case.yaml": (2, 1000, 1000, 11, 5.0),
    "cases/lid_driven_cavity_2d_8m/case.yaml": (2, 1414, 1414, 11, 5.0),
    "cases/lid_driven_cavity_2d_16m/case.yaml": (2, 2000, 2000, 11, 5.0),
    "cases/cavity3d_1m/case.yaml": (3, 50, 50, 9, 4.0),
    "cases/cavity3d_8m/case.yaml": (3, 100, 100, 9, 4.0),
    "cases/cavity3d_narrow/case.yaml": (3, 100, 45, 4, 4.0),
    "cases/cavity3d_weak4_k2_8m/case.yaml": (3, 79, 159, 9, 4.0),
}


def legacy_chain_cuts(fluid_counts, weights, minimum_own_columns) -> list[int]:
    """The pre-E31 compute_chain_cuts body (searchsorted side="left" taken as is)."""
    fluid_counts = np.asarray(fluid_counts, dtype=np.int64)
    slab_count, grid_nx = len(weights), len(fluid_counts)
    cumulative = np.cumsum(fluid_counts)
    total, weight_total = int(fluid_counts.sum()), sum(weights)
    cuts, cumulative_weight = [], 0.0
    for weight in weights[:-1]:
        cumulative_weight += weight
        target = max(1, int(total * (cumulative_weight / weight_total)))
        cuts.append(int(np.searchsorted(cumulative, target, side="left")))
    for j in range(len(cuts)):
        low = (cuts[j - 1] if j > 0 else 0) + minimum_own_columns
        high = grid_nx - minimum_own_columns * (slab_count - 1 - j)
        cuts[j] = max(low, min(cuts[j], high))
    return cuts


def yaml_h(path) -> str:
    """physics h as written in case.yaml (the loader's smoothing length)."""
    return next(line.split(":", 1)[1].split("#")[0].strip()
                for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip().startswith("h:"))


def lattice_fluid_counts(dimension, half, half_x, border, hdx, h_text=None):
    """Per voxel column fluid count of a generator lattice, binned like the loader + partition_v6:
    positions written '%.6f' and parsed to float32, frame bbox the same, origin = bbox_min - h / 2,
    nx = floor(span / h + 0.5) + 1, column = floor((x - origin) / h) in float32."""
    dx = 0.5 / half
    h = float(h_text) if h_text is not None else float(f"{hdx * dx:.6f}")
    frame_half_x = (half_x + border) * dx + 0.6 * dx
    bbox_min = float(np.float32(float(f"{-frame_half_x:.6f}")))
    bbox_max = float(np.float32(float(f"{frame_half_x:.6f}")))
    origin = bbox_min - 0.5 * h
    nx = int(math.floor((bbox_max - bbox_min) / h + 0.5)) + 1
    lattice_x = np.arange(-half_x, half_x + 1, dtype=np.int64)
    x32 = np.array([float(f"{value:.6f}") for value in lattice_x * dx], dtype=np.float32)
    columns = np.floor((x32 - origin) / h).astype(np.int64)
    np.clip(columns, 0, nx - 1, out=columns)
    per_lattice_column = (2 * half + 1) ** (dimension - 1)
    return np.bincount(columns, minlength=nx).astype(np.int64) * per_lattice_column


def loaded_histograms(case_path):
    """(fluid, all particles) per voxel column from the loader (partition_v6's own binning)."""
    from experiment.v6.utils.case_loader_v6 import load_case_v6
    from experiment.v6.utils.partition_v6 import _bin_fluid_counts
    case = load_case_v6(str(_REPO_ROOT / case_path))
    grid_nx = case.grid.grid_dimension_x
    h = case.physics.smoothing_length
    columns = np.floor((case.initial.positions[:, 0] - case.grid.origin_x) / h).astype(np.int64)
    np.clip(columns, 0, grid_nx - 1, out=columns)
    return _bin_fluid_counts(case), np.bincount(columns, minlength=grid_nx).astype(np.int64)


def slab_rows(fluid, everything, cuts, force_band):
    boundaries = [0] + list(cuts) + [len(fluid)]
    rows = []
    for index in range(len(boundaries) - 1):
        first, last = boundaries[index], boundaries[index + 1] - 1
        own = np.arange(first, last + 1)
        leading, trailing = index > 0, index < len(boundaries) - 2
        outside = own[~((leading & (own < first + force_band)) | (trailing & (own > last - force_band)))]
        rows.append({"first": int(first), "last": int(last), "own_columns": int(last - first + 1),
                     "fluid_columns": int(np.count_nonzero(fluid[first:last + 1])),
                     "fluid": int(fluid[first:last + 1].sum()),
                     "all": int(everything[first:last + 1].sum()) if everything is not None else None,
                     "n_B": int(everything[outside].sum()) if everything is not None else None})
    return rows


def imbalance(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(values.max() / values.mean() - 1.0)


def e29_table() -> dict:
    force_band = configured_band_widths()[2]
    out = {}
    for name, case_path in E29_CASES.items():
        fluid, everything = loaded_histograms(case_path)
        entry = {"grid_nx": int(len(fluid)), "fluid_total": int(fluid.sum()), "force_band": force_band}
        for rule, function in (("legacy", legacy_chain_cuts), ("nearest", chain_cuts_from_counts)):
            cuts = function(fluid, [1.0, 1.0], MINIMUM_OWN_COLUMNS_HARD)
            rows = slab_rows(fluid, everything, cuts, force_band)
            entry[rule] = {"cuts": cuts, "slabs": rows, "fluid_imbalance": imbalance([r["fluid"] for r in rows]),
                           "n_B_imbalance": imbalance([r["n_B"] for r in rows])}
        meta_path = _REPO_ROOT / E29_SCAN / f"{name}__k2__on" / "run_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            entry["e29_measured"] = [{"own_columns": s["own_columns"], "n_B": s["n_B"]} for s in meta["slabs"]["per_sim"]]
            entry["e29_partition"] = meta.get("partition")
        out[name] = entry
        legacy, nearest = entry["legacy"], entry["nearest"]
        print(f"{name:10s} nx {len(fluid):4d} legacy cut {legacy['cuts']} fluid "
              f"{[r['fluid'] for r in legacy['slabs']]} ({100 * legacy['fluid_imbalance']:.2f} %) -> nearest cut "
              f"{nearest['cuts']} fluid {[r['fluid'] for r in nearest['slabs']]} ({100 * nearest['fluid_imbalance']:.2f} %)",
              flush=True)
    return out


def cluster_table() -> dict:
    out = {}
    for name, (dimension, half, half_x, border, hdx) in CLUSTER_CASES.items():
        h_text = None
        if dimension == 2:   # the local case.yaml carries the h the cluster run read
            size = name.split("_")[1]
            yaml_path = _REPO_ROOT / f"cases/lid_driven_cavity_2d_{size}/case.yaml"
            h_text = yaml_h(yaml_path)
        fluid = lattice_fluid_counts(dimension, half, half_x, border, hdx, h_text)
        entry = {"grid_nx": int(len(fluid)), "fluid_total": int(fluid.sum()),
                 "column_fluid_max": int(fluid.max()), "lattice": [dimension, half, half_x, border, hdx], "K": {}}
        for slab_count in (2, 4, 8):
            weights = [1.0] * slab_count
            row = {}
            for rule, function in (("legacy", legacy_chain_cuts), ("nearest", chain_cuts_from_counts)):
                cuts = function(fluid, weights, MINIMUM_OWN_COLUMNS_HARD)
                rows = slab_rows(fluid, None, cuts, 0)
                row[rule] = {"cuts": cuts, "fluid": [r["fluid"] for r in rows],
                             "own_columns": [r["own_columns"] for r in rows],
                             "fluid_imbalance": imbalance([r["fluid"] for r in rows])}
            entry["K"][slab_count] = row
        out[name] = entry
        print(f"{name:18s} nx {len(fluid):5d} column {fluid.max():,} of {fluid.sum():,}: " + "; ".join(
            f"K={k} {100 * v['legacy']['fluid_imbalance']:.2f} -> {100 * v['nearest']['fluid_imbalance']:.2f} %"
            for k, v in entry["K"].items()), flush=True)
    return out


def validate() -> dict:
    out = {}
    for case_path, lattice in VALIDATION_CASES.items():
        if not (_REPO_ROOT / case_path).parent.joinpath("domain.obj").exists():
            out[case_path] = "no .obj files"
            continue
        loaded, _ = loaded_histograms(case_path)
        h_text = yaml_h(_REPO_ROOT / case_path)
        analytic = lattice_fluid_counts(*lattice, h_text=h_text)
        out[case_path] = bool(len(loaded) == len(analytic) and np.array_equal(loaded, analytic))
        print(f"validate {case_path}: {'identical' if out[case_path] else 'DIFFERENT'}", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e29", action="store_true")
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()
    result = {}
    if arguments.validate:
        result["validation"] = validate()
    if arguments.e29:
        result["e29"] = e29_table()
    if arguments.cluster:
        result["cluster"] = cluster_table()
    if arguments.out:
        out = _REPO_ROOT / arguments.out
        out.mkdir(parents=True, exist_ok=True)
        (out / "cut_balance.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"wrote {out / 'cut_balance.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
