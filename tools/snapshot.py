"""
2D scatter snapshot tool for SPH simulation debug.

Pure CPU; reads numpy arrays produced by SphSimulator.readback_*. Saves a PNG
showing every active particle, frame bbox, and (optionally) the voxel grid.
By default colors particles by material; if a density array is supplied,
fluid particles are colored by relative density deviation (ρ−ρ₀)/ρ₀ instead.

Use:
    from tools.snapshot import save_snapshot
    positions = sim.readback_positions()
    save_snapshot(sim.case, positions, "snap.png", title="step 100")
"""

import pathlib
from typing import Optional

import matplotlib
matplotlib.use("Agg")           # headless; PNG only, no GUI backend
import matplotlib.pyplot as plt
import numpy as np

from utils.sph.case import (
    KIND_BOUNDARY,
    KIND_FLUID,
    KIND_INLET,
    KIND_ROTOR,
    Case,
)


# Color per material kind (default scheme; overridable via plot_kwargs).
_KIND_COLORS = {
    KIND_FLUID:    "tab:blue",
    KIND_BOUNDARY: "0.55",      # gray
    KIND_INLET:    "tab:green",
    KIND_ROTOR:    "tab:red",
}


def save_snapshot(
    case: Case,
    positions: np.ndarray,                              # (POOL_SIZE+1, 4)
    output_path,
    *,
    title: str = "",
    density: Optional[np.ndarray] = None,               # (POOL_SIZE+1, 2) optional
    show_voxel_grid: bool = False,
    point_size: float = 6.0,
    figsize: tuple = (10, 6),
    dpi: int = 120,
    deviation_clip: float = 0.05,                       # ±5% colorbar range
) -> pathlib.Path:
    """Save a 2D PNG snapshot of the simulation state. Returns the Path written.

    `positions` is the full (POOL_SIZE+1, 4) readback. Slot 0 and slots beyond
    the active pool are skipped automatically (filtered by voxel_id != 0).

    `density`, if given, must be the (POOL_SIZE+1, 2) readback of whichever
    density buffer is the read side at the moment of the snapshot. Fluid
    particles will be color-mapped by (ρ−ρ₀)/ρ₀ in [-deviation_clip, +deviation_clip].
    Boundary / rotor / inlet particles always use their material color.
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=figsize)

    # ---- Frame bbox (recovered from grid origin + dimension) ----
    origin = np.array(case.grid["origin"], dtype=np.float64)
    dimension = np.array(case.grid["dimension"], dtype=np.int64)
    h = case.physics.smoothing_length
    # bbox.min sits at center of voxel (0,0,0) by grid.compute_grid convention,
    # so bbox.min = origin + h/2 and bbox.max = origin + (dim - 0.5) * h.
    bbox_min = origin + 0.5 * h
    bbox_max = origin + (dimension.astype(np.float64) - 0.5) * h

    rect = plt.Rectangle(
        (bbox_min[0], bbox_min[1]),
        bbox_max[0] - bbox_min[0],
        bbox_max[1] - bbox_min[1],
        edgecolor="black", facecolor="none",
        linewidth=1.2, linestyle="--", zorder=1,
    )
    axis.add_patch(rect)

    # ---- Optional voxel grid ----
    if show_voxel_grid:
        for i in range(int(dimension[0]) + 1):
            axis.axvline(origin[0] + i * h, color="0.93", linewidth=0.4, zorder=0)
        for j in range(int(dimension[1]) + 1):
            axis.axhline(origin[1] + j * h, color="0.93", linewidth=0.4, zorder=0)

    # ---- Plot particles per source (preserve material grouping) ----
    cursor = 1                                          # 1-based slot 0 unused
    last_density_scatter = None                         # for shared colorbar
    for source in case.particle_sources:
        n = int(source.vertices.shape[0])
        slot_slice = slice(cursor, cursor + n)
        sub_pos = positions[slot_slice]

        # Filter alive (voxel_id stored in pos.w; killed particles have w == 0)
        alive_mask = sub_pos[:, 3] >= 0.5
        n_alive = int(alive_mask.sum())
        sub_alive = sub_pos[alive_mask]

        material = case.materials[source.material_group_id]
        label = f"{material.name} ({n_alive}/{n})"

        if density is not None and material.kind == KIND_FLUID and n_alive > 0:
            sub_density = density[slot_slice][alive_mask, 0]
            rho0 = material.rest_density
            color_values = (sub_density - rho0) / rho0
            last_density_scatter = axis.scatter(
                sub_alive[:, 0], sub_alive[:, 1],
                c=color_values, cmap="RdBu_r", s=point_size,
                vmin=-deviation_clip, vmax=+deviation_clip,
                label=label, zorder=2,
            )
        else:
            color = _KIND_COLORS.get(material.kind, "tab:purple")
            axis.scatter(
                sub_alive[:, 0], sub_alive[:, 1],
                c=color, s=point_size, label=label, zorder=2,
            )

        cursor += n

    axis.set_aspect("equal")
    pad_x = 0.04 * (bbox_max[0] - bbox_min[0])
    pad_y = 0.04 * (bbox_max[1] - bbox_min[1])
    axis.set_xlim(bbox_min[0] - pad_x, bbox_max[0] + pad_x)
    axis.set_ylim(bbox_min[1] - pad_y, bbox_max[1] + pad_y)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title(title)
    axis.legend(loc="upper right", fontsize=8, framealpha=0.85)

    if last_density_scatter is not None:
        cbar = figure.colorbar(last_density_scatter, ax=axis, shrink=0.9, pad=0.02)
        cbar.set_label(r"$(\rho - \rho_0) / \rho_0$")

    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    return output_path
