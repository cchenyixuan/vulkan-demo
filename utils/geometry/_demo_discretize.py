"""
_demo_discretize.py — discretize the provided 2D/3D test cases and preview.

Loads utils/geometry/test_cases/{test_2d,test_3d}.obj, fills each at the given
particle spacing with BOTH lattices (grid + hex/FCC) into labeled INTERIOR +
BOUNDARY particles (an n-layer surface shell), writes the resulting clouds out
as .obj, and renders preview PNGs (interior vs boundary colored).

Run:  .venv/Scripts/python.exe utils/geometry/_demo_discretize.py
"""

from __future__ import annotations

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

from utils.geometry import (                                   # noqa: E402
    LATTICE_GRID, LATTICE_HEX, MeshRegion, PlanarMeshRegion,
    load_obj_triangles, project_planar_mesh, sample_with_boundary,
)

PARTICLE_DIAMETER = 0.02            # nearest-neighbor spacing = particle diameter
BOUNDARY_LAYERS = 3                 # surface shell thickness in lattice layers
_HERE = pathlib.Path(__file__).resolve().parent
_CASE_DIR = _HERE / "test_cases"
_OUT_DIR = _CASE_DIR / "out"


def write_obj(path: pathlib.Path, points: np.ndarray) -> None:
    if points.shape[1] == 2:                                   # lift 2D -> 3D (z=0)
        points = np.column_stack([points, np.zeros(points.shape[0])])
    header = f"# {points.shape[0]} particles, spacing={PARTICLE_DIAMETER}"
    np.savetxt(path, points, fmt="v %.6f %.6f %.6f", header=header, comments="")


def discretize_2d() -> None:
    print("\n=== 2D case: test_2d.obj ===")
    vertices_3d, triangles = load_obj_triangles(_CASE_DIR / "test_2d.obj")
    vertices_2d, flat_axis, flat_value = project_planar_mesh(vertices_3d)
    region = PlanarMeshRegion(vertices_2d, triangles)
    print(f"  {vertices_3d.shape[0]} verts, {triangles.shape[0]} tris; "
          f"flat axis {flat_axis}; {region._boundary_segments.shape[0]} boundary edges")

    figure, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for axis_plot, lattice in zip(axes, (LATTICE_GRID, LATTICE_HEX)):
        t0 = time.perf_counter()
        interior, boundary = sample_with_boundary(
            region, PARTICLE_DIAMETER, lattice, boundary_layers=BOUNDARY_LAYERS)
        print(f"  {lattice:5s}: interior={interior.shape[0]:>6} boundary={boundary.shape[0]:>5}  "
              f"({time.perf_counter()-t0:.2f}s)")
        write_obj(_OUT_DIR / f"test_2d_{lattice}_interior.obj", interior)
        write_obj(_OUT_DIR / f"test_2d_{lattice}_boundary.obj", boundary)
        for start, end in region._boundary_segments:
            axis_plot.plot([start[0], end[0]], [start[1], end[1]], color="k", linewidth=0.8)
        axis_plot.scatter(interior[:, 0], interior[:, 1], s=2.0, color="tab:blue", label="interior")
        axis_plot.scatter(boundary[:, 0], boundary[:, 1], s=3.0, color="tab:red", label="boundary")
        axis_plot.set_title(f"2D {lattice}  interior={interior.shape[0]} boundary={boundary.shape[0]}")
        axis_plot.set_aspect("equal")
        axis_plot.legend(loc="upper right", markerscale=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(_OUT_DIR / "preview_2d.png", dpi=110)
    plt.close(figure)
    print(f"  -> {_OUT_DIR / 'preview_2d.png'}")


def discretize_3d() -> None:
    print("\n=== 3D case: test_3d.obj ===")
    vertices, triangles = load_obj_triangles(_CASE_DIR / "test_3d.obj")
    print(f"  {vertices.shape[0]} verts, {triangles.shape[0]} tris")
    region = MeshRegion(vertices, triangles)

    figure = plt.figure(figsize=(13, 11))
    for column, lattice in enumerate((LATTICE_GRID, LATTICE_HEX)):
        t0 = time.perf_counter()
        interior, boundary = sample_with_boundary(
            region, PARTICLE_DIAMETER, lattice, boundary_layers=BOUNDARY_LAYERS)
        print(f"  {lattice:5s}: interior={interior.shape[0]:>8} boundary={boundary.shape[0]:>7}  "
              f"({time.perf_counter()-t0:.1f}s)")
        write_obj(_OUT_DIR / f"test_3d_{lattice}_interior.obj", interior)
        write_obj(_OUT_DIR / f"test_3d_{lattice}_boundary.obj", boundary)

        # top: thin z-slice (cross-section shows the boundary shell ringing the interior)
        ax_slice = figure.add_subplot(2, 2, column + 1)
        for cloud, color, name in ((interior, "tab:blue", "interior"), (boundary, "tab:red", "boundary")):
            sliced = cloud[np.abs(cloud[:, 2]) < PARTICLE_DIAMETER * 0.6]
            ax_slice.scatter(sliced[:, 0], sliced[:, 1], s=4.0, color=color, label=name)
        ax_slice.set_title(f"3D {lattice}: z~0 slice")
        ax_slice.set_aspect("equal")
        ax_slice.legend(loc="upper right", markerscale=2, fontsize=8)

        # bottom: the BOUNDARY shell only (subsampled) — the surface skin
        ax_3d = figure.add_subplot(2, 2, column + 3, projection="3d")
        shell = boundary[np.linspace(0, boundary.shape[0] - 1, min(8000, boundary.shape[0])).astype(int)]
        ax_3d.scatter(shell[:, 0], shell[:, 1], shell[:, 2], s=1.5, color="tab:red")
        ax_3d.set_title(f"3D {lattice}: boundary shell ({boundary.shape[0]} pts)")
    figure.tight_layout()
    figure.savefig(_OUT_DIR / "preview_3d.png", dpi=110)
    plt.close(figure)
    print(f"  -> {_OUT_DIR / 'preview_3d.png'}")


def main() -> int:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"particle diameter (spacing) = {PARTICLE_DIAMETER} m, "
          f"boundary_layers = {BOUNDARY_LAYERS}")
    discretize_2d()
    discretize_3d()
    print(f"\noutputs in {_OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
