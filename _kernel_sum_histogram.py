"""Compute kernel_sum for every particle in the initial layout and plot
histograms by source (domain / boundary / all).

Mirrors correction.comp's accumulator exactly:
    kernel_sum_i = Σ_{j != i}  V_j · W_C4(|x_i - x_j|)

V_j is taken from the calibrated material volume (matches simulator's
upload), so this is a faithful CPU replica of the GPU's kernel_sum
output at the very first step (before any movement).
"""

import pathlib
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from utils.sph.case import load_case


CASE_PATH = "cases/lid_driven_cavity_2d/case.yaml"
OUTPUT_PATH = "output/lid_driven_cavity_2d/kernel_sum_histograms.png"


def wendland_c4_2d(r: np.ndarray, h: float) -> np.ndarray:
    """Vectorized 2D Wendland C4. Same formula as helpers.glsl::evaluate_kernel."""
    q = r / h
    one_minus_q = 1.0 - q
    coefficient = 9.0 / (np.pi * h * h)
    inside = q < 1.0
    profile = (35.0 / 3.0) * q * q + 6.0 * q + 1.0
    return np.where(inside, coefficient * (one_minus_q ** 6) * profile, 0.0)


def compute_kernel_sums(positions: np.ndarray, volumes: np.ndarray, h: float) -> np.ndarray:
    """Σ_{j != i} V_j · W(|x_i - x_j|) for every particle, with KD-tree neighbor search.

    Uses cKDTree.sparse_distance_matrix to enumerate ALL pairs within radius
    h (including self at d=0), then subtracts the self contribution V_i·W(0)
    so the result matches correction.comp's `if (j == i) continue;` semantics.
    """
    n = positions.shape[0]
    tree = cKDTree(positions)

    print(f"  building pairs (sparse_distance_matrix)...")
    t0 = time.perf_counter()
    pairs = tree.sparse_distance_matrix(tree, h, output_type="coo_matrix")
    print(f"  pairs: {pairs.nnz:,d}   ({time.perf_counter()-t0:.2f}s)")

    print(f"  vectorized W + bincount accumulate...")
    t0 = time.perf_counter()
    W = wendland_c4_2d(pairs.data.astype(np.float64), h)
    contributions = volumes[pairs.col] * W
    kernel_sum = np.bincount(pairs.row, weights=contributions, minlength=n)
    # Subtract self: V_i × W(0). W(0) = coefficient × 1 × 1.
    W_at_zero = 9.0 / (np.pi * h * h)
    kernel_sum -= volumes * W_at_zero
    print(f"  accumulate done ({time.perf_counter()-t0:.2f}s)")
    return kernel_sum


def plot_histograms(kernel_sum, masks, n_bins=120):
    """3 panels: domain / boundary (wall+wall_top) / all."""
    bin_edges = np.linspace(0.0, 1.2, n_bins + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    panels = [
        ("domain",   "tab:blue",   masks["domain"]),
        ("boundary", "0.45",       masks["boundary"]),
        ("all",      "tab:purple", masks["all"]),
    ]

    for ax, (title, color, mask) in zip(axes, panels):
        sub = kernel_sum[mask]
        ax.hist(sub, bins=bin_edges, color=color, edgecolor="black", linewidth=0.3)
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0,
                   label="kernel_sum = 1")
        ax.axvline(sub.mean(), color="red", linestyle=":", linewidth=1.5,
                   label=f"mean = {sub.mean():.4f}")
        ax.set_title(f"{title}  (n = {len(sub):,d})")
        ax.set_xlabel("kernel_sum  (Σ_{j != i} V_j · W_ij)")
        ax.set_ylabel("count")
        ax.set_xlim(0.0, 1.2)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.25)

        # Stats overlay
        text = (f"min  = {sub.min():.4f}\n"
                f"max  = {sub.max():.4f}\n"
                f"std  = {sub.std():.4f}")
        ax.text(0.98, 0.95, text, transform=ax.transAxes,
                ha="right", va="top",
                fontsize=9, family="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    fig.suptitle(
        f"Initial kernel_sum distribution — lid_driven_cavity_2d  "
        f"(h={h}, dx=2r={2*case.physics.particle_radius}, h/dx={h/(2*case.physics.particle_radius):.0f})",
        fontsize=11,
    )
    fig.tight_layout()
    pathlib.Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved: {OUTPUT_PATH}")


# --- main ---------------------------------------------------------------

print(f"loading case...")
case = load_case(CASE_PATH)
h = case.physics.h

print(f"  h = {h}")
print(f"  particle_radius = {case.physics.particle_radius}")
print(f"  per-material volume:")
for m in case.materials:
    print(f"    {m.name:8s}  V = {m.volume:.6e}  (calibrated)")

# Concatenate all particles + per-particle volume + source label.
all_xy = []
all_vol = []
labels = []
for src in case.particle_sources:
    n = src.vertices.shape[0]
    mat = case.materials[src.material_group_id]
    all_xy.append(src.vertices[:, :2])         # 2D positions only
    all_vol.append(np.full(n, mat.volume, dtype=np.float64))
    labels.extend([src.obj_path.stem] * n)

positions = np.concatenate(all_xy).astype(np.float64)
volumes = np.concatenate(all_vol)
labels = np.array(labels)
print(f"\ntotal particles = {positions.shape[0]:,d}")
for stem in np.unique(labels):
    print(f"  {stem:12s} {(labels == stem).sum():,d}")

print(f"\ncomputing kernel_sum...")
t0 = time.perf_counter()
kernel_sum = compute_kernel_sums(positions, volumes, h)
print(f"total: {time.perf_counter()-t0:.2f}s")

print(f"\noverall kernel_sum stats:")
print(f"  min   = {kernel_sum.min():.6f}")
print(f"  max   = {kernel_sum.max():.6f}")
print(f"  mean  = {kernel_sum.mean():.6f}")
print(f"  std   = {kernel_sum.std():.6f}")

masks = {
    "domain":   labels == "domain",
    "boundary": (labels == "wall") | (labels == "wall_top"),
    "all":      np.ones(len(labels), dtype=bool),
}

print(f"\nper-source mean:")
for k, m in masks.items():
    sub = kernel_sum[m]
    print(f"  {k:9s}  n={m.sum():>8,d}  mean={sub.mean():.6f}  "
          f"min={sub.min():.4f}  max={sub.max():.4f}")

plot_histograms(kernel_sum, masks)
