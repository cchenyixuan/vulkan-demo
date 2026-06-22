"""
_demo_relax.py — fill + constrained relaxation, before/after, for the test cases.

For each case: load the mesh, fill it (interior + boundary shell, cached as .obj),
relax the near-surface band, and emit a before/after preview + quality metrics
(kernel-sum density deviation dq and nearest-neighbor uniformity) + relaxed .obj.

    .venv/Scripts/python.exe utils/geometry/_demo_relax.py --only 2d
    .venv/Scripts/python.exe utils/geometry/_demo_relax.py            # all cases
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from scipy.spatial import cKDTree                              # noqa: E402

from utils.geometry import (                                   # noqa: E402
    LATTICE_GRID, MeshRegion, PlanarMeshRegion,
    load_obj_triangles, project_planar_mesh, sample_with_boundary,
)
from utils.geometry.relax import relax_particles                # noqa: E402

PARTICLE_DIAMETER = 0.02
BOUNDARY_LAYERS = 3
SMOOTHING_LENGTH = 5.0 * PARTICLE_DIAMETER           # h/dx = 5 (matches the SPH cases)
_HERE = pathlib.Path(__file__).resolve().parent
_CASE_DIR = _HERE / "test_cases"
_OUT_DIR = _CASE_DIR / "out"

# name -> (obj file, is_planar)
CASES = {
    "2d":        ("test_2d.obj", True),
    "3d":        ("test_3d.obj", False),
    "3d_hole":   ("test_3d_hole.obj", False),
    "3d_2_hole": ("test_3d_2_hole.obj", False),
    "3d_tube":   ("test_3d_tube.obj", False),
}


def build_region(obj_file, is_planar):
    vertices, triangles = load_obj_triangles(_CASE_DIR / obj_file)
    if is_planar:
        vertices_2d, _, _ = project_planar_mesh(vertices)
        return PlanarMeshRegion(vertices_2d, triangles)
    return MeshRegion(vertices, triangles)


def load_or_fill(stem, region):
    interior_path = _OUT_DIR / f"{stem}_grid_interior.obj"
    boundary_path = _OUT_DIR / f"{stem}_grid_boundary.obj"
    if interior_path.exists() and boundary_path.exists():
        interior = _read_obj(interior_path)[:, :region.dimension]
        boundary = _read_obj(boundary_path)[:, :region.dimension]
        print(f"  loaded cached fill: interior={interior.shape[0]} boundary={boundary.shape[0]}")
        return interior, boundary
    t0 = time.perf_counter()
    interior, boundary = sample_with_boundary(
        region, PARTICLE_DIAMETER, LATTICE_GRID, boundary_layers=BOUNDARY_LAYERS)
    print(f"  filled: interior={interior.shape[0]} boundary={boundary.shape[0]} ({time.perf_counter()-t0:.0f}s)")
    _write_obj(interior_path, interior)
    _write_obj(boundary_path, boundary)
    return interior, boundary


def _read_obj(path):
    rows = [tuple(map(float, line.split()[1:4])) for line in open(path) if line.startswith("v ")]
    return np.array(rows)


def _write_obj(path, points):
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(points.shape[0])])
    np.savetxt(path, points, fmt="v %.6f %.6f %.6f", header=f"# {points.shape[0]}", comments="")


# --- quality metrics -------------------------------------------------------

def _wendland(distance, smoothing_length, dimension):
    q = distance / smoothing_length
    coefficient = (9.0 / (math.pi * smoothing_length ** 2) if dimension == 2
                   else 495.0 / (32.0 * math.pi * smoothing_length ** 3))
    value = coefficient * np.clip(1.0 - q, 0.0, None) ** 6 * ((35.0 / 3.0) * q * q + 6.0 * q + 1.0)
    return np.where(q < 1.0, value, 0.0)


def kernel_density_deviation(points, dimension, sample_count=3000):
    """Per-particle Σ V·W (partition of unity ~1 in the bulk); returns the sample
    kernel-sum values. Lower spread = more uniform. Evaluated on a random subset."""
    tree = cKDTree(points)
    rng_index = np.linspace(0, points.shape[0] - 1, min(sample_count, points.shape[0])).astype(int)
    sample = points[rng_index]
    # calibrated volume V = 1 / Σ_lattice W on an ideal grid at this spacing
    from utils.geometry.lattice import neighbor_offsets
    offsets = neighbor_offsets("grid", dimension, PARTICLE_DIAMETER, SMOOTHING_LENGTH)
    lattice_sum = float(np.sum(_wendland(np.linalg.norm(offsets, axis=1), SMOOTHING_LENGTH, dimension)))
    volume = 1.0 / lattice_sum
    kernel_sum = np.empty(sample.shape[0])
    neighbor_lists = tree.query_ball_point(sample, SMOOTHING_LENGTH)
    for row, neighbors in enumerate(neighbor_lists):
        distance = np.linalg.norm(points[neighbors] - sample[row], axis=1)
        distance = distance[distance > 1e-12]
        kernel_sum[row] = volume * float(np.sum(_wendland(distance, SMOOTHING_LENGTH, dimension)))
    return kernel_sum


def nearest_neighbor_distance(points):
    tree = cKDTree(points)
    distance, _ = tree.query(points, k=2)
    return distance[:, 1]


def report_metrics(tag, points, dimension):
    nn = nearest_neighbor_distance(points)
    ks = kernel_density_deviation(points, dimension)
    print(f"    {tag:6s}: NN dist mean={nn.mean()/PARTICLE_DIAMETER:.3f}dx "
          f"min={nn.min()/PARTICLE_DIAMETER:.3f}dx CV={nn.std()/nn.mean():.3f}  |  "
          f"kernel_sum mean={ks.mean():.3f} std={ks.std():.3f}")
    return nn, ks


# --- per-case driver -------------------------------------------------------

def run_case(name):
    obj_file, is_planar = CASES[name]
    stem = pathlib.Path(obj_file).stem
    print(f"\n=== {name} ({obj_file}) ===")
    region = build_region(obj_file, is_planar)
    interior, boundary = load_or_fill(stem, region)
    before = np.vstack([interior, boundary])
    dimension = region.dimension

    print("  metrics BEFORE:")
    nn_before, ks_before = report_metrics("before", before, dimension)

    t0 = time.perf_counter()
    after, constraint = relax_particles(
        before, region, PARTICLE_DIAMETER, iterations=30, active_band=3.0, verbose=True)
    print(f"  relaxed in {time.perf_counter()-t0:.0f}s")

    print("  metrics AFTER:")
    nn_after, ks_after = report_metrics("after", after, dimension)
    _write_obj(_OUT_DIR / f"{stem}_grid_relaxed.obj", after)

    _make_figure(name, before, after, dimension, nn_before, nn_after, ks_before, ks_after)


def _make_figure(name, before, after, dimension, nn_before, nn_after, ks_before, ks_after):
    figure = plt.figure(figsize=(16, 9))
    if dimension == 2:
        for position, (cloud, title) in ((1, (before, "before")), (2, (after, "after"))):
            ax = figure.add_subplot(2, 3, position)
            ax.scatter(cloud[:, 0], cloud[:, 1], s=2.0, color="tab:blue")
            ax.set_title(f"{name} {title}"); ax.set_aspect("equal")
        # edge zoom on the slanted right bulge: staircase (orange) -> conformed (green)
        ax = figure.add_subplot(2, 3, 3)
        ax.scatter(before[:, 0], before[:, 1], s=10, color="tab:orange", label="before")
        ax.scatter(after[:, 0], after[:, 1], s=10, color="tab:green", label="after")
        ax.set_xlim(0.55, 1.25); ax.set_ylim(-0.45, 0.15); ax.set_aspect("equal")
        ax.set_title("slanted-edge zoom"); ax.legend(fontsize=7)
    else:
        half = 0.6 * PARTICLE_DIAMETER
        zc = 0.5 * (before[:, 2].min() + before[:, 2].max())
        for position, (cloud, title) in ((1, (before, "before")), (2, (after, "after"))):
            ax = figure.add_subplot(2, 3, position)
            sliced = cloud[np.abs(cloud[:, 2] - zc) < half]
            ax.scatter(sliced[:, 0], sliced[:, 1], s=4.0, color="tab:blue")
            ax.set_title(f"{name} {title}  z~{zc:.2f} slice"); ax.set_aspect("equal")
        ax = figure.add_subplot(2, 3, 3, projection="3d")
        shell = after[np.linspace(0, after.shape[0]-1, min(8000, after.shape[0])).astype(int)]
        ax.scatter(shell[:, 0], shell[:, 1], shell[:, 2], s=1.0, color="tab:red")
        ax.set_title("after (subsampled)")

    ax = figure.add_subplot(2, 3, 4)
    ax.hist(nn_before / PARTICLE_DIAMETER, bins=60, range=(0.3, 1.5), alpha=0.5, label="before", color="tab:orange")
    ax.hist(nn_after / PARTICLE_DIAMETER, bins=60, range=(0.3, 1.5), alpha=0.5, label="after", color="tab:green")
    ax.set_title("nearest-neighbor distance / dx"); ax.legend()
    ax = figure.add_subplot(2, 3, 5)
    ax.hist(ks_before, bins=50, alpha=0.5, label=f"before std={ks_before.std():.3f}", color="tab:orange")
    ax.hist(ks_after, bins=50, alpha=0.5, label=f"after std={ks_after.std():.3f}", color="tab:green")
    ax.set_title("kernel-sum (partition of unity ~1)"); ax.legend()
    figure.tight_layout()
    figure.savefig(_OUT_DIR / f"relax_{name}.png", dpi=105)
    plt.close(figure)
    print(f"  -> {_OUT_DIR / f'relax_{name}.png'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="subset of case names")
    args = parser.parse_args()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = args.only if args.only else list(CASES)
    for name in names:
        run_case(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
