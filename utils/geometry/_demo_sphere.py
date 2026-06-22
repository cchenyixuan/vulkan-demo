"""
_demo_sphere.py — relaxation on a strongly curved surface (analytic sphere).

This is where relaxation earns its keep: a sphere's surface staircases badly under
a Cartesian lattice cookie-cutter. Fill is instant (analytic SDF), relaxation uses
the exact SDF + normal (RegionSurfaceProxy). Shows before/after z-slice, an arc
zoom (staircase -> conformed), and a curved-surface metric: the radial spread of
the outer surface skin (should collapse toward r=R).

    .venv/Scripts/python.exe utils/geometry/_demo_sphere.py
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
from scipy.spatial import cKDTree                              # noqa: E402

from utils.geometry import LATTICE_GRID, Sphere, sample_with_boundary   # noqa: E402
from utils.geometry.relax import relax_particles                        # noqa: E402

RADIUS = 1.0
PARTICLE_DIAMETER = 0.05
_OUT_DIR = pathlib.Path(__file__).resolve().parent / "test_cases" / "out"


def surface_skin_radial_std(points, dx):
    """std of |point| for the outer surface skin (|r - R| < 0.6 dx) — measures how
    staircased vs conformed the surface is."""
    radius = np.linalg.norm(points, axis=1)
    skin = np.abs(radius - RADIUS) < 0.6 * dx
    return float(radius[skin].std()), int(skin.sum())


def nn_stats(points):
    distance, _ = cKDTree(points).query(points, k=2)
    nn = distance[:, 1]
    return nn, nn.mean(), nn.min(), nn.std() / nn.mean()


def main() -> int:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    region = Sphere([0.0, 0.0, 0.0], RADIUS)
    interior, boundary = sample_with_boundary(region, PARTICLE_DIAMETER, LATTICE_GRID, boundary_layers=3)
    before = np.vstack([interior, boundary])
    print(f"sphere R={RADIUS} dx={PARTICLE_DIAMETER}: interior={interior.shape[0]} boundary={boundary.shape[0]}")

    std_b, n_skin = surface_skin_radial_std(before, PARTICLE_DIAMETER)
    nn_b, m_b, min_b, cv_b = nn_stats(before)
    print(f"  BEFORE: surface-skin radial std = {std_b/PARTICLE_DIAMETER:.3f} dx  (skin n={n_skin})  "
          f"NN min={min_b/PARTICLE_DIAMETER:.3f}dx CV={cv_b:.3f}")

    t0 = time.perf_counter()
    after, _ = relax_particles(before, region, PARTICLE_DIAMETER, iterations=50, active_band=3.0, verbose=True)
    print(f"  relaxed in {time.perf_counter()-t0:.1f}s")

    std_a, _ = surface_skin_radial_std(after, PARTICLE_DIAMETER)
    nn_a, m_a, min_a, cv_a = nn_stats(after)
    print(f"  AFTER : surface-skin radial std = {std_a/PARTICLE_DIAMETER:.3f} dx  "
          f"NN min={min_a/PARTICLE_DIAMETER:.3f}dx CV={cv_a:.3f}")
    print(f"  >>> surface conformance improved {std_b/std_a:.1f}x "
          f"(radial std {std_b/PARTICLE_DIAMETER:.3f} -> {std_a/PARTICLE_DIAMETER:.3f} dx)")

    half = 0.6 * PARTICLE_DIAMETER
    figure = plt.figure(figsize=(16, 9))
    for position, (cloud, title) in ((1, (before, "before")), (2, (after, "after"))):
        ax = figure.add_subplot(2, 3, position)
        sliced = cloud[np.abs(cloud[:, 2]) < half]
        ax.scatter(sliced[:, 0], sliced[:, 1], s=6.0, color="tab:blue")
        ax.add_patch(plt.Circle((0, 0), RADIUS, fill=False, color="k", linewidth=0.8))
        ax.set_title(f"sphere {title}  z~0 slice"); ax.set_aspect("equal")
    ax = figure.add_subplot(2, 3, 3)
    for cloud, color, name in ((before, "tab:orange", "before"), (after, "tab:green", "after")):
        sliced = cloud[np.abs(cloud[:, 2]) < half]
        ax.scatter(sliced[:, 0], sliced[:, 1], s=18, color=color, label=name)
    ax.add_patch(plt.Circle((0, 0), RADIUS, fill=False, color="k", linewidth=0.8))
    ax.set_xlim(0.6, 1.08); ax.set_ylim(-0.25, 0.25); ax.set_aspect("equal")
    ax.set_title("surface arc zoom (staircase -> conformed)"); ax.legend(fontsize=8)

    ax = figure.add_subplot(2, 3, 4)
    ax.hist(nn_b / PARTICLE_DIAMETER, bins=60, range=(0.3, 1.5), alpha=0.5, color="tab:orange", label="before")
    ax.hist(nn_a / PARTICLE_DIAMETER, bins=60, range=(0.3, 1.5), alpha=0.5, color="tab:green", label="after")
    ax.set_title("nearest-neighbor distance / dx"); ax.legend()
    ax = figure.add_subplot(2, 3, 5)
    rb = np.linalg.norm(before, axis=1); ra = np.linalg.norm(after, axis=1)
    sel_b = np.abs(rb - RADIUS) < 0.6 * PARTICLE_DIAMETER
    sel_a = np.abs(ra - RADIUS) < 0.6 * PARTICLE_DIAMETER
    ax.hist((rb[sel_b] - RADIUS) / PARTICLE_DIAMETER, bins=50, alpha=0.5, color="tab:orange",
            label=f"before std={std_b/PARTICLE_DIAMETER:.3f}dx")
    ax.hist((ra[sel_a] - RADIUS) / PARTICLE_DIAMETER, bins=50, alpha=0.5, color="tab:green",
            label=f"after std={std_a/PARTICLE_DIAMETER:.3f}dx")
    ax.set_title("surface-skin radial offset (r-R)/dx"); ax.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(_OUT_DIR / "relax_sphere.png", dpi=105)
    plt.close(figure)
    print(f"  -> {_OUT_DIR / 'relax_sphere.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
