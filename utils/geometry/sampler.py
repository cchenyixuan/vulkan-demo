"""
sampler.py — fill a Region with lattice particles (SDF membership).

``fill_region`` tiles the region's bounding box with the chosen lattice and
keeps the sites that lie inside by at least ``inset_fraction · particle_spacing``.
The inset is the cheap anti-staircase measure: requiring sites to be a consistent
fraction of a spacing inside the surface gives the outermost particle layer a
uniform offset from the true boundary instead of a 0…spacing ragged edge.

For mesh regions the membership test is O(sites × triangles), so sites are
evaluated in batches to bound peak memory (important once you tile millions of
candidate sites against a multi-thousand-triangle mesh).

This produces the APPROXIMATE distribution. The boundary-conforming polish
(constrained relaxation, feature-aware projection) is a separate stage that runs
on top of this; see the package docstring. For axis-aligned lattice-conforming
geometry (e.g. the lid-driven cavity box) this approximate fill is already exact
and needs no relaxation.
"""

from __future__ import annotations

import numpy as np

from utils.geometry.lattice import tile_bounding_box
from utils.geometry.region import Region


def fill_region(
    region: Region,
    particle_spacing: float,
    lattice_kind: str,
    inset_fraction: float = 0.5,
    boundary_pad: float = 2.0,
    batch_size: int = 1_000_000,
) -> np.ndarray:
    """Return interior particle positions, shape ``(N, region.dimension)``.

    ``inset_fraction`` : keep sites with ``signed_distance <= -inset_fraction·spacing``.
                         0.0 keeps everything strictly inside the surface; 0.5
                         gives a consistent half-spacing boundary offset.
    ``boundary_pad``    : expand the tiling box by this many spacings per side so
                         a site whose center is just inside a slightly-outside
                         bounding box is not missed (cheap insurance).
    ``batch_size``      : candidate sites per signed-distance evaluation.
    """
    bounding_box_min, bounding_box_max = region.bounds()
    pad = boundary_pad * particle_spacing
    candidates = tile_bounding_box(
        np.asarray(bounding_box_min) - pad,
        np.asarray(bounding_box_max) + pad,
        particle_spacing,
        lattice_kind,
        region.dimension,
    )
    if candidates.shape[0] == 0:
        return candidates

    inset = inset_fraction * particle_spacing
    kept_chunks = []
    for start in range(0, candidates.shape[0], batch_size):
        chunk = candidates[start:start + batch_size]
        # Use the region's own membership test: regions (e.g. MeshRegion) can take
        # a cheaper path for inset == 0 (winding number only, no distance pass).
        inside = region.contains(chunk, inset=inset)
        if np.any(inside):
            kept_chunks.append(chunk[inside])
    if not kept_chunks:
        return candidates[:0]
    return np.concatenate(kept_chunks, axis=0)


def sample_with_boundary(
    region: Region,
    particle_spacing: float,
    lattice_kind: str,
    boundary_layers: int = 3,
    boundary_pad: float = 2.0,
    batch_size: int = 1_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Discretize a SOLID region into labeled (interior, boundary) particles.

    Both sets lie on the SAME lattice, so they interlock with no interface
    spacing mismatch. ``boundary`` is the outermost ``boundary_layers`` lattice
    layers just inside the surface — an SPH dummy/ghost wall shell that gives the
    surface a particle skin and lets fluid near the wall keep full kernel support.
    ``interior`` is the deeper fill.

    Banding is by signed distance: a lattice site is in the solid when its center
    is within half a spacing of the surface (``signed_distance <= ½·spacing``);
    of those, the outermost ``boundary_layers·spacing`` band is boundary, the rest
    interior. For FULL kernel support at the wall, set
    ``boundary_layers >= ceil(smoothing_length / particle_spacing)`` (h/dx, ≈5).

    Returns ``(interior_points, boundary_points)``, each ``(N, dimension)``.

    NOTE: this is the lattice-shell boundary — exact for axis-aligned / lattice-
    conforming surfaces. On strongly curved or angled surfaces the shell inherits
    mild lattice staircasing; the constrained-relaxation stage (boundary particles
    projected onto the surface manifold) is what conforms it. See package docstring.
    """
    if boundary_layers < 1:
        raise ValueError("boundary_layers must be >= 1")
    bounding_box_min, bounding_box_max = region.bounds()
    pad = (boundary_pad + boundary_layers) * particle_spacing
    candidates = tile_bounding_box(
        np.asarray(bounding_box_min) - pad,
        np.asarray(bounding_box_max) + pad,
        particle_spacing,
        lattice_kind,
        region.dimension,
    )
    if candidates.shape[0] == 0:
        return candidates, candidates

    surface_band = 0.5 * particle_spacing                 # site center within ½dx of surface = "in solid"
    wall_thickness = boundary_layers * particle_spacing
    interior_chunks, boundary_chunks = [], []
    for start in range(0, candidates.shape[0], batch_size):
        chunk = candidates[start:start + batch_size]
        signed = region.signed_distance(chunk)
        in_solid = signed <= surface_band
        boundary_mask = in_solid & (signed > -wall_thickness)
        interior_mask = signed <= -wall_thickness
        if np.any(boundary_mask):
            boundary_chunks.append(chunk[boundary_mask])
        if np.any(interior_mask):
            interior_chunks.append(chunk[interior_mask])
    interior = np.concatenate(interior_chunks, axis=0) if interior_chunks else candidates[:0]
    boundary = np.concatenate(boundary_chunks, axis=0) if boundary_chunks else candidates[:0]
    return interior, boundary
