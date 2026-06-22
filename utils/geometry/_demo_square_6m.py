"""
_demo_square_6m.py — discretize a unit square to ~6M particles (scale demo).

Generic 2D geometry (analytic Box, NOT coupled to any solver case). Reverse-solves
the spacing to hit ~6,000,000 particles, fills with interior + a 3-layer boundary
shell, reports counts / spacing / timing / memory, and previews a corner zoom
(individual lattice + shell) alongside the full downsampled view.

    .venv/Scripts/python.exe utils/geometry/_demo_square_6m.py
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
    Box, LATTICE_GRID, particle_spacing_for_target_count, sample_with_boundary,
)

TARGET_PARTICLES = 6_000_000
BOUNDARY_LAYERS = 3
_OUT_DIR = pathlib.Path(__file__).resolve().parent / "test_cases" / "out"


def main() -> int:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    square = Box([0.0, 0.0], [1.0, 1.0])
    spacing = particle_spacing_for_target_count(TARGET_PARTICLES, 1.0, LATTICE_GRID, 2)
    print(f"unit square, target {TARGET_PARTICLES:,} -> spacing = {spacing:.6e} "
          f"(~{round(1/spacing)} per axis)")

    t0 = time.perf_counter()
    interior, boundary = sample_with_boundary(square, spacing, LATTICE_GRID, boundary_layers=BOUNDARY_LAYERS)
    fill_seconds = time.perf_counter() - t0
    total = interior.shape[0] + boundary.shape[0]
    print(f"filled in {fill_seconds:.1f}s: interior={interior.shape[0]:,} "
          f"boundary={boundary.shape[0]:,}  total={total:,}")
    print(f"memory: {(interior.nbytes + boundary.nbytes)/1e6:.0f} MB "
          f"({total*2*8/1e6:.0f} MB of float64 positions)")

    # write the cloud (interior fluid + boundary wall), lifted 2D->3D (z=0)
    t1 = time.perf_counter()
    combined = np.vstack([interior, boundary])
    combined3 = np.column_stack([combined, np.zeros(combined.shape[0])])
    np.savetxt(_OUT_DIR / "square_6m_grid.obj", combined3,
               fmt="v %.6f %.6f %.6f", header=f"# {combined.shape[0]} particles dx={spacing:.6e}", comments="")
    print(f"wrote square_6m_grid.obj ({combined.shape[0]:,} verts) in {time.perf_counter()-t1:.0f}s")

    # --- preview ---------------------------------------------------------
    figure, axes = plt.subplots(1, 2, figsize=(14, 7))
    # left: full square, downsampled (6M is solid when fully plotted)
    sub = np.linspace(0, interior.shape[0]-1, 60_000).astype(int)
    axes[0].scatter(interior[sub, 0], interior[sub, 1], s=0.5, color="tab:blue", label="interior (sampled)")
    bsub = np.linspace(0, boundary.shape[0]-1, min(40_000, boundary.shape[0])).astype(int)
    axes[0].scatter(boundary[bsub, 0], boundary[bsub, 1], s=0.5, color="tab:red", label="boundary")
    axes[0].set_title(f"unit square, {total:,} particles (downsampled view)")
    axes[0].set_aspect("equal"); axes[0].legend(markerscale=8, loc="upper right", fontsize=8)
    # right: corner zoom showing individual lattice + the 3-layer shell
    window = 30 * spacing
    cint = interior[(interior[:, 0] < window) & (interior[:, 1] < window)]
    cbnd = boundary[(boundary[:, 0] < window) & (boundary[:, 1] < window)]
    axes[1].scatter(cint[:, 0], cint[:, 1], s=14, color="tab:blue", label="interior")
    axes[1].scatter(cbnd[:, 0], cbnd[:, 1], s=14, color="tab:red", label="boundary shell")
    axes[1].set_title(f"corner zoom (dx={spacing:.2e}, {BOUNDARY_LAYERS}-layer shell)")
    axes[1].set_aspect("equal"); axes[1].legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    figure.savefig(_OUT_DIR / "square_6m.png", dpi=110)
    plt.close(figure)
    print(f"-> {_OUT_DIR / 'square_6m.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
