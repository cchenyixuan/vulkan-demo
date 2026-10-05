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


def lattice_histograms(dimension, half, half_x, border, hdx, h_text=None):
    """(fluid, all particles) per voxel column of a node-lattice generator case, binned like the loader +
    partition_v6: positions written '%.6f' and parsed to float32, frame bbox the same, origin = bbox_min -
    h / 2, nx = floor(span / h + 0.5) + 1, column = floor((x - origin) / h) in float32. Every lattice site of
    the box is a particle (fluid, wall or lid), so each x lattice column holds the full cross-section,
    (2 (half + border) + 1)^(d - 1) particles, of which (2 half + 1)^(d - 1) are fluid inside the fluid range."""
    dx = 0.5 / half
    h = float(h_text) if h_text is not None else float(f"{hdx * dx:.6f}")
    frame_half_x = (half_x + border) * dx + 0.6 * dx
    bbox_min = float(np.float32(float(f"{-frame_half_x:.6f}")))
    bbox_max = float(np.float32(float(f"{frame_half_x:.6f}")))
    origin = bbox_min - 0.5 * h
    nx = int(math.floor((bbox_max - bbox_min) / h + 0.5)) + 1
    lattice_x = np.arange(-half_x - border, half_x + border + 1, dtype=np.int64)
    x32 = np.array([float(f"{value:.6f}") for value in lattice_x * dx], dtype=np.float32)
    columns = np.floor((x32 - origin) / h).astype(np.int64)
    np.clip(columns, 0, nx - 1, out=columns)
    fluid = np.abs(lattice_x) <= half_x
    everything = np.bincount(columns, minlength=nx).astype(np.int64) * (2 * (half + border) + 1) ** (dimension - 1)
    fluid_counts = np.bincount(columns[fluid], minlength=nx).astype(np.int64) * (2 * half + 1) ** (dimension - 1)
    return fluid_counts, everything


def lattice_fluid_counts(dimension, half, half_x, border, hdx, h_text=None):
    return lattice_histograms(dimension, half, half_x, border, hdx, h_text)[0]


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


def cluster_histograms(name: str):
    dimension, half, half_x, border, hdx = CLUSTER_CASES[name]
    h_text = None
    if dimension == 2:      # the local case.yaml carries the h the cluster run read
        h_text = yaml_h(_REPO_ROOT / f"cases/lid_driven_cavity_2d_{name.split('_')[1]}/case.yaml")
    return lattice_histograms(dimension, half, half_x, border, hdx, h_text)


def walls_table() -> dict:
    """E32 2 (c): with the E31 cuts at equal weights, n_B per slab counting the walls (own particles outside
    the force band) and the end slabs against the interior ones."""
    force_band = configured_band_widths()[2]
    out = {}
    for name in CLUSTER_CASES:
        fluid, everything = cluster_histograms(name)
        entry = {}
        for slab_count in (2, 4, 8):
            cuts = chain_cuts_from_counts(fluid, [1.0] * slab_count, MINIMUM_OWN_COLUMNS_HARD)
            rows = slab_rows(fluid, everything, cuts, force_band)
            n_b = [r["n_B"] for r in rows]
            interior = n_b[1:-1] if slab_count > 2 else n_b
            own = [r["all"] for r in rows]
            interior_own = own[1:-1] if slab_count > 2 else own
            entry[slab_count] = {"cuts": cuts, "n_B": n_b, "own": own, "fluid": [r["fluid"] for r in rows],
                                 "own_columns": [r["own_columns"] for r in rows],
                                 "end_over_interior": max(n_b[0], n_b[-1]) / float(np.mean(interior)),
                                 "own_end_over_interior": max(own[0], own[-1]) / float(np.mean(interior_own)),
                                 "n_B_imbalance": imbalance(n_b)}
        out[name] = entry
        print(f"{name:18s} " + "; ".join(f"K={k}: end/interior n_B {v['end_over_interior']:.3f} (own with walls "
                                        f"{v['own_end_over_interior']:.3f}), n_B max/mean-1 {100 * v['n_B_imbalance']:.2f} %"
                                        for k, v in entry.items()), flush=True)
    return out


# E32 3: the aligned E7 cases (utils/geometry/_demo_cavity_case_aligned.py), each next to the node lattice it
# replaces: (name under cases/aligned, aligned (dimension, n, n_x, border, h/dx), old node lattice (dimension,
# half, half_x, border, h/dx), case.yaml whose h the old lattice was binned with or None (h written %.6f by the
# old generators), role). An old lattice marked "same size" is the node lattice the old generator would give
# for that size (the case did not exist before).
ALIGNED_CASES = (
    # 2-D squares, h/dx 5, border 11: E29's per-card points (K = 2 on 62k / 250k / 1M / 4M / 16M; the total
    # grows with K, so K = 4 / 8 use the next / the second-next size), the strong curve 4M-128M with the K
    # sweep at 32M / 64M, and the k1 members of the weak families
    ("cavity2d_n240", (2, 240, 240, 11, 5.0), (2, 124, 124, 11, 5.0), "cases/cavity_2d_62k/case.yaml",
     "E29 per card 32k (K=2)"),
    ("cavity2d_n360", (2, 360, 360, 11, 5.0), (2, 180, 180, 11, 5.0), None, "E29 per card 32k (K=4); same size"),
    ("cavity2d_n520", (2, 520, 520, 11, 5.0), (2, 250, 250, 11, 5.0), "cases/cavity_2d_250k/case.yaml",
     "E29 per card 130k (K=2), 32k (K=8)"),
    ("cavity2d_n720", (2, 720, 720, 11, 5.0), (2, 360, 360, 11, 5.0), None, "E29 per card 130k (K=4); same size"),
    ("cavity2d_n1000", (2, 1000, 1000, 11, 5.0), (2, 500, 500, 11, 5.0), "cases/lid_driven_cavity_2d_gen/case.yaml",
     "E29 per card 0.5M (K=2), 130k (K=8)"),
    ("cavity2d_n1440", (2, 1440, 1440, 11, 5.0), (2, 707, 707, 11, 5.0), "cases/cavity_weak_k1_2m/case.yaml",
     "E29 per card 0.5M (K=4); weak 2M k1"),
    ("cavity2d_n2000", (2, 2000, 2000, 11, 5.0), (2, 1000, 1000, 11, 5.0), "cases/lid_driven_cavity_2d_4m/case.yaml",
     "E29 per card 2M (K=2), 0.5M (K=8); strong 4M; weak4 k1"),
    ("cavity2d_n2840", (2, 2840, 2840, 11, 5.0), (2, 1414, 1414, 11, 5.0), "cases/lid_driven_cavity_2d_8m/case.yaml",
     "E29 per card 2M (K=4); strong 8M; weak8 k1"),
    ("cavity2d_n4000", (2, 4000, 4000, 11, 5.0), (2, 2000, 2000, 11, 5.0), "cases/lid_driven_cavity_2d_16m/case.yaml",
     "E29 per card 8M (K=2), 2M (K=8); strong 16M; weak16 k1"),
    ("cavity2d_n5640", (2, 5640, 5640, 11, 5.0), (2, 2828, 2828, 11, 5.0), "cases/lid_driven_cavity_2d_32m/case.yaml",
     "E29 per card 8M (K=4); strong 32M, K sweep; weak32 k1"),
    ("cavity2d_n8000", (2, 8000, 8000, 11, 5.0), (2, 4000, 4000, 11, 5.0), "cases/lid_driven_cavity_2d_64m/case.yaml",
     "E29 per card 8M (K=8); strong 64M, K sweep"),
    ("cavity2d_n11320", (2, 11320, 11320, 11, 5.0), (2, 5656, 5656, 11, 5.0), "cases/lid_driven_cavity_2d_128m/case.yaml",
     "strong 128M"),
    ("cavity2d_n9000", (2, 9000, 9000, 11, 5.0), (2, 4500, 4500, 11, 5.0), None,
     "strong top point whose K=1 reference fits (instead of 128M); same size"),
    # 2-D weak families: member k = K slabs of the k1 square side by side (n_x = K n)
    *[(f"cavity2d_n{n}_k{k}", (2, n, k * n, 11, 5.0), (2, half, half * k + k // 2, 11, 5.0),
       local.format(k=k) if local else None, f"weak {label} k{k}")
      for n, half, label, local in ((1440, 707, "2M", None), (2000, 1000, "4M", None), (2840, 1414, "8M", None),
                                    (4000, 2000, "16M", None), (5640, 2828, "32M", None))
      for k in (2, 4, 8)],
    # 3-D, h/dx 4: E29's per-card points (border 9 as the E29 cases), the narrow slab, the weak families
    # (border 4; k8 = the stretched 32M / 64M of the strong matrix) and the cube
    ("cavity3d_n104_b9", (3, 104, 104, 9, 4.0), (3, 50, 50, 9, 4.0), "cases/cavity3d_1m/case.yaml",
     "E29 per card 0.7M (K=2)"),
    ("cavity3d_n200_b9", (3, 200, 200, 9, 4.0), (3, 100, 100, 9, 4.0), "cases/cavity3d_8m/case.yaml",
     "E29 per card 4.7M (K=2)"),
    ("cavity3d_n200_x96", (3, 200, 96, 4, 4.0), (3, 100, 45, 4, 4.0), "cases/cavity3d_narrow/case.yaml",
     "E29 narrow slab (K=2)"),
    *[(f"cavity3d_n{n}" + (f"_k{k}" if k > 1 else ""), (3, n, k * n, 4, 4.0), (3, half, half * k + k // 2, 4, 4.0),
       None, f"3-D weak {label} k{k}" + (f"; strong stretched {32 if n == 160 else 64}M" if k == 8 else ""))
      for n, half, label in ((160, 79, "4M"), (200, 100, "8M")) for k in (1, 2, 4, 8)],
    ("cavity3d_n416", (3, 416, 416, 4, 4.0), (3, 200, 200, 4, 4.0), None, "cube (401^3 before)"),
)
ALIGNED_DIRECTORY = "cases/aligned"
# K = 1 memory: v6 release defaults measured at K = 1 (E29 / opt logs) ~350 B per particle in 2-D, ~345 B in
# 3-D (pool 1.15 N + defrag twin + voxel structures); the single-GPU ceiling measured on the 5090 is ~90M
# particles (90.4M ran, 96.5M out of device memory; docs/sph_v4_summary.md)
K1_BYTES_PER_PARTICLE = {2: 350.0, 3: 345.0}
K1_PARTICLE_CEILING = 90_000_000


def aligned_entry(name, new, old, old_yaml, role) -> dict:
    from utils.geometry._demo_cavity_case_aligned import counts as aligned_counts
    from utils.geometry._demo_cavity_case_aligned import layout, loader_histogram
    old_fluid, old_all = lattice_histograms(*old, h_text=yaml_h(_REPO_ROOT / old_yaml) if old_yaml else None)
    spec = layout(*new)
    new_fluid, new_all, _ = loader_histogram(spec)
    fluid_count, wall_count, lid_count = aligned_counts(spec)
    if int(new_fluid.sum()) != fluid_count or int(new_all.sum()) != fluid_count + wall_count + lid_count:
        raise AssertionError(f"{name}: loader histogram {int(new_fluid.sum())} / {int(new_all.sum())} != lattice "
                             f"counts {(fluid_count, wall_count, lid_count)}")
    dimension = new[0]
    entry = {"name": name, "role": role, "old_yaml": old_yaml,
             "old": {"lattice": list(old), "fluid": int(old_fluid.sum()), "total": int(old_all.sum()),
                     "columns": int(len(old_fluid)), "fluid_columns": int(np.count_nonzero(old_fluid))},
             "new": {"lattice": list(new), "fluid": fluid_count, "wall": wall_count, "lid": lid_count,
                     "total": fluid_count + wall_count + lid_count, "columns": int(len(new_fluid)),
                     "fluid_columns": int(np.count_nonzero(new_fluid)), "wall_columns": spec["wall_columns"]}}
    for label, histogram in (("old", old_fluid), ("new", new_fluid)):
        for slab_count in (2, 4, 8):
            try:
                cuts = chain_cuts_from_counts(histogram, [1.0] * slab_count, MINIMUM_OWN_COLUMNS_HARD)
            except ValueError:
                entry[label][f"K{slab_count}"] = None
                continue
            rows = slab_rows(histogram, None, cuts, 0)
            entry[label][f"K{slab_count}"] = imbalance([r["fluid"] for r in rows])
            entry[label][f"K{slab_count}_fluid_columns"] = [r["fluid_columns"] for r in rows]
    entry["fluid_change"] = entry["new"]["fluid"] / entry["old"]["fluid"] - 1.0
    entry["total_change"] = entry["new"]["total"] / entry["old"]["total"] - 1.0
    total = entry["new"]["total"]
    entry["k1_gib"] = total * K1_BYTES_PER_PARTICLE[dimension] / 2 ** 30
    entry["k1_fits"] = total <= K1_PARTICLE_CEILING
    return entry


def aligned_table() -> dict:
    """E32 3: every ALIGNED_CASES entry: counts, columns, the equal-weight fluid imbalance at K = 2 / 4 / 8
    (None: fewer columns than K x the minimum own columns) for the old and the aligned lattice, and the
    K = 1 memory estimate."""
    out = {}
    for name, new, old, old_yaml, role in ALIGNED_CASES:
        entry = aligned_entry(name, new, old, old_yaml, role)
        out[name] = entry

        def percents(side, digits):
            return "/".join("—" if entry[side][f"K{k}"] is None else f"{100 * entry[side][f'K{k}']:.{digits}f}"
                            for k in (2, 4, 8))
        print(f"{name:22s} fluid {entry['old']['fluid']:>12,} -> {entry['new']['fluid']:>12,} "
              f"({100 * entry['fluid_change']:+5.1f} %), fluid columns {entry['old']['fluid_columns']:>4} -> "
              f"{entry['new']['fluid_columns']:>4}; K=2/4/8 old {percents('old', 2)} new {percents('new', 3)} %; "
              f"K=1 {entry['k1_gib']:.1f} GiB{'' if entry['k1_fits'] else ' DOES NOT FIT'}", flush=True)
    return out


def write_aligned_cases(names=None) -> None:
    """case.yaml + generate.txt of every aligned case (the generator with --yaml-only; no particle files)."""
    import subprocess
    for name, new, _old, _old_yaml, _role in ALIGNED_CASES:
        if names and name not in names:
            continue
        dimension, n, n_x, border, _hdx = new
        command = [sys.executable, "utils/geometry/_demo_cavity_case_aligned.py", "--dimension", str(dimension),
                   "--n", str(n), "--n-x", str(n_x), "--border", str(border), "--out", f"{ALIGNED_DIRECTORY}/{name}",
                   "--yaml-only"]
        subprocess.run(command, cwd=_REPO_ROOT, check=True)


def validate() -> dict:
    out = {}
    for case_path, lattice in VALIDATION_CASES.items():
        if not (_REPO_ROOT / case_path).parent.joinpath("domain.obj").exists():
            out[case_path] = "no .obj files"
            continue
        loaded, loaded_all = loaded_histograms(case_path)
        h_text = yaml_h(_REPO_ROOT / case_path)
        analytic, analytic_all = lattice_histograms(*lattice, h_text=h_text)
        out[case_path] = bool(len(loaded) == len(analytic) and np.array_equal(loaded, analytic)
                              and np.array_equal(loaded_all, analytic_all))
        print(f"validate {case_path}: fluid and all particles {'identical' if out[case_path] else 'DIFFERENT'}",
              flush=True)
    # E32: every aligned case whose particle files were generated (--objs-only): the loader against the
    # generator's emulation, and the real partition's per-slab fluid at K = 2 / 4 / 8
    from experiment.v6.utils.case_loader_v6 import load_case_v6
    from experiment.v6.utils.partition_v6 import compute_chain_partition
    from utils.geometry._demo_cavity_case_aligned import layout, loader_histogram
    for name, new, _old, _old_yaml, _role in ALIGNED_CASES:
        case_path = f"{ALIGNED_DIRECTORY}/{name}/case.yaml"
        if not (_REPO_ROOT / case_path).parent.joinpath("domain.obj").exists():
            continue
        loaded, loaded_all = loaded_histograms(case_path)
        emulated, emulated_all, _ = loader_histogram(layout(*new))
        identical = bool(len(loaded) == len(emulated) and np.array_equal(loaded, emulated)
                         and np.array_equal(loaded_all, emulated_all))
        case = load_case_v6(str(_REPO_ROOT / case_path))
        slabs = {}
        for slab_count in (2, 4, 8):
            try:
                chain = compute_chain_partition(case, [1.0] * slab_count, 1.2)
            except ValueError:
                continue
            fluid = [row["fluid"] for row in slab_rows(loaded, None, chain.cuts, 0)]
            slabs[slab_count] = {"cuts": [int(cut) for cut in chain.cuts], "fluid": fluid,
                                 "own_particles": [int(g.own_particle_count) for g in chain.geometry],
                                 "imbalance": imbalance(fluid)}
        out[case_path] = {"identical": identical, "slabs": slabs}
        print(f"validate {case_path}: loader {'identical to' if identical else 'DIFFERENT from'} the generator; "
              + "; ".join(f"K={k} fluid/slab {sorted(set(v['fluid']))} imbalance {100 * v['imbalance']:.3f} %"
                          for k, v in slabs.items()), flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e29", action="store_true")
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--walls", action="store_true", help="E32: n_B with walls per slab, end vs interior slabs")
    parser.add_argument("--aligned", action="store_true", help="E32: old / aligned E7 cases, counts and imbalance")
    parser.add_argument("--write-aligned-cases", action="store_true",
                        help="E32: write case.yaml + generate.txt of every aligned case under cases/aligned")
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()
    result = {}
    if arguments.validate:
        result["validation"] = validate()
    if arguments.e29:
        result["e29"] = e29_table()
    if arguments.cluster:
        result["cluster"] = cluster_table()
    if arguments.walls:
        result["walls"] = walls_table()
    if arguments.aligned:
        result["aligned"] = aligned_table()
    if arguments.write_aligned_cases:
        write_aligned_cases()
    if arguments.out:
        out = _REPO_ROOT / arguments.out
        out.mkdir(parents=True, exist_ok=True)
        (out / "cut_balance.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"wrote {out / 'cut_balance.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
