"""
_neighbor_stats.py — candidate and neighbour counts of a particle state (docs/seam_audit/v7_single.md, "§3 补测").

Candidates (measurement B's definition): the particles listed in the 3^d voxels around a particle's own voxel,
self included. With c(v) the count of voxel v and s(v) = sum of c over the 3^d stencil around v, every particle
in v visits s(v) candidates: candidate pairs = sum_v c(v) s(v); per particle = pairs / N, or, over the fluid
particles only, mean of s(v(i)) over the fluid i.

Neighbours: |{j != i : r_ij < h}| with h the support radius (common.glsl: Wendland C4 support = SMOOTHING_LENGTH;
the neighbour loops skip distance >= SMOOTHING_LENGTH and distance < 1e-12). Counted on the CPU with scipy's
cKDTree in float64 at three radii — h (1 - 1e-5), the largest double below h (strict r < h in double arithmetic)
and h (1 + 1e-5) — and with the shader's float32 arithmetic for the pairs in the shell between the first and the
last radius, on the float32 positions the GPU holds. A lattice whose h / dx is an integer has pairs at exactly
r = h (axis pairs and Pythagorean triples: 12 of the 80 lattice points within 5 dx in 2-D) whose verdict depends
on the rounding of the coordinates themselves; the float32 count is the one the kernels' loops see (up to the
last-bit difference of a fused multiply-add in the GPU's dot product, counted as 'ambiguous' pairs), the two
double counts bracket it.

Voxel binning on the CPU (the initial state, no GPU): initialize_voxelization.comp's
ivec3(floor((position - origin) / SMOOTHING_LENGTH)) in float32 and helpers.glsl's voxel id
y + z NY + x NY NZ + 1 (x slowest, y fastest).

    python experiment/v7_single/_neighbor_stats.py --case cases/lid_driven_cavity_2d_8m/case.yaml --out initial.json
    python experiment/v7_single/_neighbor_stats.py --state logs/.../state.npz --out end.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

KIND_FLUID = 0
RADIUS_MARGIN = 1e-5


def voxel_coordinates(positions, origin, h: float) -> np.ndarray:
    """floor((position - origin) / h) in float32, the shader's arithmetic (spec constants are float32)."""
    pos = np.ascontiguousarray(positions, dtype=np.float32)
    origin32 = np.asarray(origin, dtype=np.float32)
    return np.floor((pos - origin32) / np.float32(h)).astype(np.int64)


def bin_particles(positions, origin, dims, h: float) -> tuple[np.ndarray, np.ndarray]:
    """(0-based linear voxel index of every particle, counts per voxel) on the loader's grid."""
    dims = tuple(int(d) for d in dims)
    coordinate = voxel_coordinates(positions, origin, h)
    inside = np.all((coordinate >= 0) & (coordinate < np.asarray(dims)), axis=1)
    if not inside.all():
        raise ValueError(f"{int((~inside).sum())} particles bin outside the grid {dims}")
    linear = coordinate[:, 1] + coordinate[:, 2] * dims[1] + coordinate[:, 0] * dims[1] * dims[2]
    counts = np.bincount(linear, minlength=dims[0] * dims[1] * dims[2]).astype(np.int64)
    return linear, counts


def stencil_sums(counts: np.ndarray, dims, z_range: int) -> tuple[np.ndarray, np.ndarray]:
    """(count field (x, z, y), stencil sum field): s(v) = sum of counts over the 3 x (2 z_range + 1) x 3 voxels."""
    dims = tuple(int(d) for d in dims)
    field = counts.reshape(dims[0], dims[2], dims[1])          # voxel id - 1 = y + z NY + x NY NZ
    padded = np.pad(field, 1)
    stencil = np.zeros_like(field)
    for dx in (-1, 0, 1):
        for dz in range(-z_range, z_range + 1):
            for dy in (-1, 0, 1):
                stencil += padded[1 + dx:1 + dx + dims[0], 1 + dz:1 + dz + dims[2], 1 + dy:1 + dy + dims[1]]
    return field, stencil


def candidate_statistics(counts: np.ndarray, dims, z_range: int, linear_of_particle: np.ndarray,
                         fluid_mask: np.ndarray) -> dict:
    field, stencil = stencil_sums(counts, dims, z_range)
    pairs = int((field.astype(np.int64) * stencil.astype(np.int64)).sum())
    particles = int(counts.sum())
    seen = stencil.reshape(-1)[linear_of_particle]
    occupied = counts[counts > 0]
    walls = ~fluid_mask
    return {"candidate_pairs": pairs, "listed_particles": particles,
            "candidates_per_particle": pairs / particles if particles else None,
            "candidates_per_fluid_particle": float(seen[fluid_mask].mean()) if fluid_mask.any() else None,
            "candidates_per_wall_particle": float(seen[walls].mean()) if walls.any() else None,
            "fluid_particles": int(fluid_mask.sum()), "wall_particles": int(walls.sum()),
            "grid_dimensions": [int(d) for d in dims], "total_voxels": int(counts.size),
            "occupied_voxels": int(occupied.size),
            "mean_particles_per_voxel": float(occupied.mean()) if occupied.size else None,
            "max_particles_per_voxel": int(occupied.max()) if occupied.size else None,
            "stencil": f"{3 ** (2 + z_range)} voxels"}


def neighbor_statistics(positions32: np.ndarray, fluid_mask: np.ndarray, h: float, *, workers: int = -1,
                        chunk_pairs: float = 2.0e7, log=print) -> dict:
    """Mean neighbour counts (all particles, fluid, walls) at the three double radii and in float32."""
    from scipy.spatial import cKDTree
    t0 = time.perf_counter()
    positions32 = np.ascontiguousarray(positions32, dtype=np.float32)
    pos64 = positions32.astype(np.float64)
    count = pos64.shape[0]
    tree = cKDTree(pos64, leafsize=32, balanced_tree=False, compact_nodes=False)
    radii = {"lo": h * (1.0 - RADIUS_MARGIN), "strict": float(np.nextafter(h, 0.0)), "hi": h * (1.0 + RADIUS_MARGIN)}
    counts = {}
    for name, radius in radii.items():
        counts[name] = tree.query_ball_point(pos64, radius, workers=workers, return_length=True).astype(np.int64) - 1
        log(f"[neighbors] r={name} ({radius:.9g}): mean {counts[name].mean():.3f} ({time.perf_counter() - t0:.0f} s)")
    # the shell (lo, hi]: the shader's float32 distance test on the float32 positions
    h32 = np.float32(h)
    ulp = np.spacing(h32)
    counts["f32"] = counts["lo"].copy()
    ambiguous = 0
    shell_pairs = 0
    per_point = max(1.0, float(counts["hi"].mean()) + 1.0)
    chunk = int(max(20000, min(count, chunk_pairs / per_point)))
    for start in range(0, count, chunk):
        stop = min(count, start + chunk)
        sub = cKDTree(pos64[start:stop], leafsize=32, balanced_tree=False, compact_nodes=False)
        pairs = sub.sparse_distance_matrix(tree, radii["hi"], output_type="ndarray")
        shell = pairs[pairs["v"] > radii["lo"]]
        del pairs
        if shell.size:
            shell_pairs += int(shell.size)
            i = shell["i"].astype(np.int64)
            j = shell["j"].astype(np.int64)
            difference = positions32[j] - positions32[start + i]
            distance = np.sqrt((difference * difference).sum(axis=1, dtype=np.float32))
            inside = (distance < h32) & (distance >= np.float32(1e-12))
            counts["f32"][start:stop] += np.bincount(i[inside], minlength=stop - start)
            ambiguous += int((np.abs(distance - h32) <= 2 * ulp).sum())
    log(f"[neighbors] float32 shell: {shell_pairs:,} pairs in (lo, hi], {ambiguous:,} within 2 ulp of h; "
        f"mean {counts['f32'].mean():.3f} ({time.perf_counter() - t0:.0f} s)")
    walls = ~fluid_mask
    out = {"h": float(h), "radii": radii, "particles": int(count), "fluid_particles": int(fluid_mask.sum()),
           "shell_pairs": shell_pairs, "ambiguous_pairs_2ulp": ambiguous, "seconds": time.perf_counter() - t0}
    for name, values in counts.items():
        out[f"neighbors_{name}_all"] = float(values.mean())
        out[f"neighbors_{name}_fluid"] = float(values[fluid_mask].mean()) if fluid_mask.any() else None
        out[f"neighbors_{name}_wall"] = float(values[walls].mean()) if walls.any() else None
        out[f"neighbors_{name}_max"] = int(values.max())
    return out


def kinds_of(case) -> np.ndarray:
    return np.asarray([int(material.kind) for material in case.materials], dtype=np.int64)


def initial_state(case_path: str, obj_cache: str | None) -> dict:
    """Positions, fluid mask, grid and h of the loaded case (CPU only)."""
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    if obj_cache:
        from _run_single_baseline_bench import install_obj_cache
        install_obj_cache(pathlib.Path(obj_cache))
    case = load_case_v7(case_path)
    grid = case.grid
    kinds = kinds_of(case)[np.asarray(case.initial.material_group, dtype=np.int64)]
    return {"positions": np.ascontiguousarray(case.initial.positions, dtype=np.float32),
            "fluid_mask": kinds == KIND_FLUID,
            "origin": (grid.origin_x, grid.origin_y, grid.origin_z),
            "dims": (grid.grid_dimension_x, grid.grid_dimension_y, grid.grid_dimension_z),
            "h": float(case.physics.smoothing_length), "z_range": int(case.physics.neighbor_z_range),
            "dimension": int(case.physics.dimension), "particles": int(case.initial.positions.shape[0]),
            "dx": 2.0 * min(float(m.radius) for m in case.materials if float(m.radius) > 0)}


def load_state(path: str) -> dict:
    with np.load(path) as data:
        state = {key: data[key] for key in data.files}
    return {"positions": np.ascontiguousarray(state["positions"], dtype=np.float32),
            "fluid_mask": state["fluid_mask"].astype(bool),
            "voxel_linear": state["voxel_linear"].astype(np.int64) if "voxel_linear" in state else None,
            "origin": tuple(float(v) for v in state["origin"]), "dims": tuple(int(v) for v in state["dims"]),
            "h": float(state["h"]), "z_range": int(state["z_range"]), "dimension": int(state["dimension"]),
            "particles": int(state["positions"].shape[0]), "dx": float(state["dx"]) if "dx" in state else None,
            "step": int(state["step"]) if "step" in state else None}


def analyze(state: dict, *, neighbors: bool = True, workers: int = -1, log=print) -> dict:
    positions = state["positions"]
    fluid = state["fluid_mask"]
    linear, counts = bin_particles(positions, state["origin"], state["dims"], state["h"])
    result = {"particles": state["particles"], "fluid_particles": int(fluid.sum()), "dimension": state["dimension"],
              "h": state["h"], "dx": state.get("dx"), "h_over_dx": (state["h"] / state["dx"]) if state.get("dx") else None,
              "step": state.get("step"), "grid_origin": list(state["origin"]),
              "candidates": candidate_statistics(counts, state["dims"], state["z_range"], linear, fluid)}
    if state.get("voxel_linear") is not None:
        gpu = state["voxel_linear"]
        mismatch = int((gpu != linear).sum())
        result["cpu_binning_vs_gpu_voxel_ids_mismatch"] = mismatch
        if mismatch:
            gpu_counts = np.bincount(gpu, minlength=counts.size).astype(np.int64)
            result["candidates_gpu_lists"] = candidate_statistics(gpu_counts, state["dims"], state["z_range"], gpu, fluid)
    if neighbors:
        result["neighbors"] = neighbor_statistics(positions, fluid, state["h"], workers=workers, log=log)
        c = result["candidates"]
        n = result["neighbors"]
        result["ratio_candidates_to_neighbors_f32_all"] = c["candidates_per_particle"] / n["neighbors_f32_all"]
        result["ratio_candidates_to_neighbors_f32_fluid"] = (c["candidates_per_fluid_particle"] / n["neighbors_f32_fluid"]
                                                             if n.get("neighbors_f32_fluid") else None)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default=None, help="case.yaml: the loaded initial state, binned on the CPU")
    parser.add_argument("--state", default=None, help="state.npz written by _run_kernel_breakdown.py --dump-state")
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-neighbors", action="store_true")
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--obj-cache", default="logs/_obj_npy_cache")
    arguments = parser.parse_args()
    if bool(arguments.case) == bool(arguments.state):
        raise SystemExit("pass exactly one of --case / --state")
    t0 = time.perf_counter()
    state = initial_state(arguments.case, arguments.obj_cache) if arguments.case else load_state(arguments.state)
    result = analyze(state, neighbors=not arguments.no_neighbors, workers=arguments.workers)
    result["source"] = arguments.case or arguments.state
    result["seconds"] = time.perf_counter() - t0
    text = json.dumps(result, indent=1)
    if arguments.out:
        pathlib.Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(arguments.out).write_text(text, encoding="utf-8")
    c = result["candidates"]
    line = (f"[neighbor_stats] {result['source']}: N={result['particles']:,} grid {c['grid_dimensions']} "
            f"occupied {c['occupied_voxels']:,}/{c['total_voxels']:,} voxels, per voxel mean {c['mean_particles_per_voxel']:.2f} "
            f"max {c['max_particles_per_voxel']}; candidates/particle {c['candidates_per_particle']:.2f} "
            f"(fluid {c['candidates_per_fluid_particle']:.2f})")
    if "neighbors" in result:
        n = result["neighbors"]
        line += (f"; neighbors f32 {n['neighbors_f32_all']:.3f} (fluid {n['neighbors_f32_fluid']:.3f}), "
                 f"double [{n['neighbors_lo_all']:.3f}, {n['neighbors_hi_all']:.3f}]")
    print(line + f" ({result['seconds']:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
