"""
_render_tgv.py — pictures of the Taylor–Green runs from their particle snapshots (experiment/v8/_run_tgv.py's
<run>/snapshots/t<time>.npz: position, velocity (synchronised), pressure, kernel sum, exact fields).

    .venv/Scripts/python.exe experiment/v8/_render_tgv.py --runs logs/v8/tgv_eps1e-4_delta --out docs/tgv/render

  fields     one run, one time: speed, pressure (mean removed), vorticity of the SPH velocity (curl of the pixel-binned
             field), vorticity error against the exact field; every particle contributes to its pixel (mean per pixel).
  zoom       the same physical window at several resolutions, every particle a dot coloured by speed: the centre of the
             box and a window across the periodic edge x = L.
  kernelsum  the Shepard kernel sum per particle, binned: the particle arrangement's regularity (1 on a regular lattice).
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


def load_snapshot(run_dir: pathlib.Path, time_label: str) -> dict:
    path = run_dir / "snapshots" / f"{time_label}.npz"
    data = np.load(path)
    return {key: data[key] for key in data.files}


def bin_mean(position: np.ndarray, values: np.ndarray, pixels: int, length: float = 1.0) -> np.ndarray:
    """Mean of `values` per pixel (pixels x pixels over [0, L)^2), NaN where no particle; image axis 0 = y (up)."""
    ix = np.clip((position[:, 0] / length * pixels).astype(np.int64), 0, pixels - 1)
    iy = np.clip((position[:, 1] / length * pixels).astype(np.int64), 0, pixels - 1)
    flat = iy * pixels + ix
    count = np.bincount(flat, minlength=pixels * pixels).astype(np.float64)
    total = np.bincount(flat, weights=values.astype(np.float64), minlength=pixels * pixels)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    return mean.reshape(pixels, pixels)


def fill_nan(image: np.ndarray) -> np.ndarray:
    """Fill empty pixels from the nearest filled one (coarse runs leave gaps at high pixel counts)."""
    if not np.isnan(image).any():
        return image
    from scipy import ndimage
    mask = np.isnan(image)
    index = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    return image[tuple(index)]


def wendland_c4_gradient_scalar(distance: np.ndarray, h: float) -> np.ndarray:
    """dW/dr of the 2-D Wendland C4 with support h (helpers.glsl): gradient = this * (x_ij / r)."""
    q = distance / h
    inside = q < 1.0
    one_minus = np.where(inside, 1.0 - q, 0.0)
    return (9.0 / (math.pi * h * h) / h) * one_minus ** 5 * q * ((-280.0 / 3.0) * q - 56.0 / 3.0) * inside


def sph_vorticity(position: np.ndarray, velocity: np.ndarray, density: np.ndarray, h: float, dx: float,
                  box: float = 1.0, xi: float = 0.001, block: int = 100_000) -> np.ndarray:
    """omega_i = sum_j V_j (v_j - v_i) x (L_i grad W_ij) with the solver's kernel, calibrated volume (V_j = m / rho_j,
    m = rho0 V_lattice), KCG matrix L_i = (M_i + xi I)^-1 and the minimum-image periodic box: the vorticity the
    solver's own operators see on the particles (float64, blocks of `block` particles against a periodic cKDTree)."""
    from scipy.spatial import cKDTree
    from experiment.v7.utils.case_loader_v7 import _calibrate_particle_volume
    points = position[:, :2].astype(np.float64)
    mass = _calibrate_particle_volume(h, 0.5 * dx, 2, "grid")       # rho0 = 1
    volume = mass / density.astype(np.float64)
    tree = cKDTree(points, boxsize=box)
    omega = np.zeros(points.shape[0])
    for start in range(0, points.shape[0], block):
        stop = min(start + block, points.shape[0])
        block_tree = cKDTree(points[start:stop], boxsize=box)
        pairs = block_tree.sparse_distance_matrix(tree, h * (1 + 1e-9), output_type="ndarray")
        i = pairs["i"] + start
        j = pairs["j"]
        keep = (i != j) & (pairs["v"] > 1e-12)
        i, j, distance = i[keep], j[keep], pairs["v"][keep]
        difference = points[i] - points[j]                             # x_ij = r_i - r_j
        difference -= box * np.round(difference / box)
        gradient = (wendland_c4_gradient_scalar(distance, h) / distance)[:, None] * difference
        weighted = gradient * volume[j][:, None]
        # M_i = sum_j V_j (r_j - r_i) (x) grad W_ij, then L_i = (M_i + xi I)^-1
        matrix = np.zeros((stop - start, 2, 2))
        np.add.at(matrix, i - start, (-difference)[:, :, None] * weighted[:, None, :])
        matrix += xi * np.eye(2)[None]
        inverse = np.linalg.inv(matrix)
        corrected = np.einsum("nab,nb->na", inverse[i - start], gradient)
        relative = velocity[j] - velocity[i]
        # curl: d v_y / d x - d v_x / d y with grad v = sum_j V_j (v_j - v_i) (x) L grad W
        contribution = volume[j] * (relative[:, 1] * corrected[:, 0] - relative[:, 0] * corrected[:, 1])
        np.add.at(omega, i, contribution)
    return omega


def render_fields(run_dir: pathlib.Path, time_label: str, out: pathlib.Path, pixels: int, dpi: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    snapshot = load_snapshot(run_dir, time_label)
    position = snapshot["position"].astype(np.float64)
    velocity = snapshot["velocity"].astype(np.float64)
    speed = np.sqrt((velocity ** 2).sum(axis=1))
    pressure = snapshot["pressure"].astype(np.float64)
    pressure -= pressure.mean()
    side = int(round(np.sqrt(position.shape[0])))
    dx = 1.0 / side
    h = 5.0 * dx
    exact_velocity = snapshot["exact_velocity"].astype(np.float64)
    amplitude = float(np.sqrt((exact_velocity ** 2).sum(axis=1)).max())      # U0 F(t): the instantaneous amplitude
    error = np.sqrt(((velocity - exact_velocity) ** 2).sum(axis=1)) / amplitude
    vorticity_particles = sph_vorticity(position, velocity, snapshot["density"], h, dx)
    exact_vorticity = snapshot["exact_vorticity"].astype(np.float64)
    vorticity_scale = float(np.abs(exact_vorticity).max())
    panels = [
        (fill_nan(bin_mean(position, speed, pixels)), "speed $|v|/U_0$", "viridis", None),
        (fill_nan(bin_mean(position, pressure, pixels)), "pressure $p/(\\rho_0 U_0^2)$ (mean removed)", "RdBu_r", "sym"),
        (fill_nan(bin_mean(position, vorticity_particles, pixels)), "vorticity $\\omega L/U_0$ (SPH gradient on the particles)", "RdBu_r", "sym"),
        (fill_nan(bin_mean(position, (vorticity_particles - exact_vorticity) / vorticity_scale, pixels)),
         "vorticity error (SPH − exact) / max|ω_exact|", "RdBu_r", "sym"),
        (fill_nan(bin_mean(position, error, pixels)), "velocity error $|v-u| / (U_0 F(t))$ (relative to the amplitude)", "magma", None),
        (fill_nan(bin_mean(position, snapshot["kernel_sum"].astype(np.float64), pixels)), "kernel sum $\\sum_j V_j W_{ij}$", "coolwarm", "ks"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 11))
    time_value = float(snapshot["time"])
    relative_l2 = float(np.sqrt(((velocity - exact_velocity) ** 2).sum() / (exact_velocity ** 2).sum()))
    vorticity_l2 = float(np.sqrt(((vorticity_particles - exact_vorticity) ** 2).mean()) / vorticity_scale)
    for ax, (image, title, cmap, kind) in zip(axes.ravel(), panels):
        if kind == "sym":
            limit = float(np.nanpercentile(np.abs(image), 99.5))
            vmin, vmax = -limit, limit
        elif kind == "ks":
            spread = float(max(abs(np.nanpercentile(image, 0.5) - 1.0), abs(np.nanpercentile(image, 99.5) - 1.0), 1e-3))
            vmin, vmax = 1.0 - spread, 1.0 + spread
        else:
            vmin, vmax = None, None
        im = ax.imshow(image, origin="lower", extent=(0, 1, 0, 1), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    particles = position.shape[0]
    fig.suptitle(f"{run_dir.name}: {particles:,} particles, t = {time_value:.2f}, {pixels}² pixels (mean per pixel); "
                 f"relative L2 velocity error {100 * relative_l2:.2f} %, vorticity rms error {100 * vorticity_l2:.2f} % of max|ω|",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"[render] {out}")


def render_zoom(runs: list[tuple[str, pathlib.Path]], time_label: str, out: pathlib.Path, window: float, dpi: int) -> None:
    """Rows: two windows (the box centre, across the periodic edge x = L); columns: the runs. Every particle a dot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    centres = [((0.5, 0.5), "box centre"), ((1.0, 0.25), "across the periodic edge $x = L$")]
    fig, axes = plt.subplots(len(centres), len(runs), figsize=(4.2 * len(runs), 4.4 * len(centres)), squeeze=False)
    for column, (label, run_dir) in enumerate(runs):
        snapshot = load_snapshot(run_dir, time_label)
        position = snapshot["position"].astype(np.float64)
        speed = np.sqrt((snapshot["velocity"].astype(np.float64) ** 2).sum(axis=1))
        side = int(round(np.sqrt(position.shape[0])))
        dx = 1.0 / side
        for row, ((cx, cy), name) in enumerate(centres):
            ax = axes[row, column]
            shifted = position.copy()
            if cx >= 1.0:            # the right half of the window is the box's left edge, drawn at x + L
                shifted[:, 0] = np.where(shifted[:, 0] < 0.5, shifted[:, 0] + 1.0, shifted[:, 0])
            inside = ((np.abs(shifted[:, 0] - cx) < window / 2) & (np.abs(shifted[:, 1] - cy) < window / 2))
            dot_points = (4.2 * 72 * 0.8) * dx / window         # dot diameter in points ~ 0.8 of a spacing
            ax.scatter(shifted[inside, 0], shifted[inside, 1], c=speed[inside], s=max(dot_points ** 2 * 0.55, 0.15),
                       cmap="viridis", linewidths=0, vmin=0, vmax=float(np.percentile(speed, 99.9)))
            ax.set_xlim(cx - window / 2, cx + window / 2); ax.set_ylim(cy - window / 2, cy + window / 2)
            ax.set_aspect("equal")
            if cx >= 1.0:
                ax.axvline(1.0, color="black", linewidth=0.8, linestyle=(0, (3, 2)))
            ax.set_title(f"{label}: {side}², dx = L/{side}, {int(inside.sum()):,} particles in {window:g} L", fontsize=10)
            ax.tick_params(labelsize=8)
            if column == 0:
                ax.set_ylabel(name, fontsize=10)
    fig.suptitle(f"the same window of {window:g} L × {window:g} L at every resolution (t = {time_label[1:].replace('p', '.')}, colour = speed)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"[render] {out}")


def render_kernel_sum(runs: list[tuple[str, pathlib.Path, str]], out: pathlib.Path, pixels: int, dpi: int, spread: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(runs), figsize=(6.2 * len(runs), 5.6), squeeze=False)
    for ax, (label, run_dir, time_label) in zip(axes[0], runs):
        snapshot = load_snapshot(run_dir, time_label)
        position = snapshot["position"].astype(np.float64)
        kernel_sum = snapshot["kernel_sum"].astype(np.float64)
        image = fill_nan(bin_mean(position, kernel_sum, pixels))
        im = ax.imshow(image, origin="lower", extent=(0, 1, 0, 1), cmap="coolwarm", vmin=1 - spread, vmax=1 + spread, interpolation="nearest")
        ax.set_title(f"{label}\nkernel sum [{kernel_sum.min():.3f}, {kernel_sum.max():.3f}], t = {float(snapshot['time']):g}", fontsize=10)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"[render] {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="logs/v8/tgv_eps1e-4_delta")
    parser.add_argument("--out", default="docs/tgv/render")
    parser.add_argument("--pixels", type=int, default=2048)
    parser.add_argument("--dpi", type=int, default=150)
    arguments = parser.parse_args()
    root = _REPOSITORY_ROOT / arguments.runs
    out = _REPOSITORY_ROOT / arguments.out
    out.mkdir(parents=True, exist_ok=True)
    render_fields(root / "re1000_n3200", "t2", out / "fields_re1000_n3200_t2.png", arguments.pixels, arguments.dpi)
    render_fields(root / "re100_n3200", "t1", out / "fields_re100_n3200_t1.png", arguments.pixels, arguments.dpi)
    render_fields(root / "re100_n800", "t1", out / "fields_re100_n800_t1.png", arguments.pixels, arguments.dpi)
    zoom_runs = [(f"Re 1000", root / f"re1000_n{side}") for side in (200, 800, 1600, 3200)]
    render_zoom(zoom_runs, "t2", out / "zoom_re1000_t2.png", 0.04, arguments.dpi)
    render_zoom(zoom_runs, "t2", out / "zoom_re1000_t2_wide.png", 0.1, arguments.dpi)
    render_kernel_sum([("Re 1000, 3200², acoustic dt (85 k steps)", root / "re1000_n3200", "t1"),
                       ("Re 100, 3200², viscous dt (328 k steps)", root / "re100_n3200", "t1"),
                       ("Re 100, 1600², viscous dt (164 k steps)", root / "re100_n1600", "t1"),
                       ("Re 100, 1600², centred box", root / "re100_n1600_center", "t1")],
                      out / "kernel_sum_float32.png", 1024, arguments.dpi, 0.03)
    return 0


if __name__ == "__main__":
    sys.exit(main())
