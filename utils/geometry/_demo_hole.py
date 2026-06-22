"""
_demo_hole.py — discretize the genus-1 (through-hole) test case and preview.

Renders three orthogonal mid-slices (so the tunnel shows up whichever axis it
runs along) plus the boundary shell, to confirm the generalized winding number
correctly carves the hole (tunnel interior = outside) and the boundary shell
wraps both the outer surface and the tunnel walls.

Grid lattice only (502 triangles makes the winding pass ~10x the simple cube;
hex is identical quality, just denser). Run:
    .venv/Scripts/python.exe utils/geometry/_demo_hole.py
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
    LATTICE_GRID, MeshRegion, load_obj_triangles, sample_with_boundary,
)

PARTICLE_DIAMETER = 0.02
BOUNDARY_LAYERS = 3
_HERE = pathlib.Path(__file__).resolve().parent
_OUT_DIR = _HERE / "test_cases" / "out"


def main() -> int:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    vertices, triangles = load_obj_triangles(_HERE / "test_cases" / "test_3d_hole.obj")
    print(f"loaded {vertices.shape[0]} verts, {triangles.shape[0]} triangles")
    region = MeshRegion(vertices, triangles)

    t0 = time.perf_counter()
    interior, boundary = sample_with_boundary(
        region, PARTICLE_DIAMETER, LATTICE_GRID, boundary_layers=BOUNDARY_LAYERS)
    print(f"grid: interior={interior.shape[0]} boundary={boundary.shape[0]} "
          f"({time.perf_counter()-t0:.1f}s)")
    np.savetxt(_OUT_DIR / "test_3d_hole_grid_interior.obj", interior,
               fmt="v %.6f %.6f %.6f", header=f"# {interior.shape[0]}", comments="")
    np.savetxt(_OUT_DIR / "test_3d_hole_grid_boundary.obj", boundary,
               fmt="v %.6f %.6f %.6f", header=f"# {boundary.shape[0]}", comments="")

    half = 0.6 * PARTICLE_DIAMETER
    slice_axes = [(0, "x~0", (1, 2)), (1, "y~0", (0, 2)), (2, "z~0", (0, 1))]
    figure = plt.figure(figsize=(14, 11))
    for index, (axis, label, plot_axes) in enumerate(slice_axes):
        ax = figure.add_subplot(2, 2, index + 1)
        for cloud, color, name in ((interior, "tab:blue", "interior"),
                                   (boundary, "tab:red", "boundary")):
            sliced = cloud[np.abs(cloud[:, axis]) < half]
            ax.scatter(sliced[:, plot_axes[0]], sliced[:, plot_axes[1]], s=4.0, color=color, label=name)
        ax.set_title(f"slice {label}")
        ax.set_aspect("equal")
        ax.legend(loc="upper right", markerscale=2, fontsize=8)
    ax_3d = figure.add_subplot(2, 2, 4, projection="3d")
    shell = boundary[np.linspace(0, boundary.shape[0] - 1, min(9000, boundary.shape[0])).astype(int)]
    ax_3d.scatter(shell[:, 0], shell[:, 1], shell[:, 2], s=1.5, color="tab:red")
    ax_3d.set_title(f"boundary shell ({boundary.shape[0]} pts)")
    figure.tight_layout()
    figure.savefig(_OUT_DIR / "preview_3d_hole.png", dpi=110)
    plt.close(figure)
    print(f"-> {_OUT_DIR / 'preview_3d_hole.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
