"""
utils.geometry — general-purpose geometry discretization for particle methods.

A standalone (no solver dependency) toolkit that turns continuous geometry into
particle point clouds for SPH / mesh-free methods:

  - lattice.py     : N-D packings (Cartesian "grid" / closest-packed "hex" =
                     2D triangular / 3D FCC). Single source of truth, matched to
                     the SPH volume calibration.
  - region.py      : SDF-backed regions — analytic primitives (Box, Sphere,
                     HalfSpace, Cylinder) + exact CSG booleans.
  - mesh_region.py : a Region from a triangle mesh, ROBUST TO DIRTY MESHES via
                     generalized winding numbers (holes / non-manifold / soup).
  - sampler.py     : fill a Region with a lattice (SDF inset membership, batched).

Planned next stages (see design discussion): feature-network extraction
(dihedral-angle edges + corner vertices), constrained relaxation (coupled
packing with boundary particles projected onto the surface manifold; corners
pinned, edges 1D-constrained), per-particle volume, and the .obj / case writer.
"""

from utils.geometry.lattice import (
    LATTICE_GRID,
    LATTICE_HEX,
    neighbor_offsets,
    particle_spacing_for_target_count,
    per_particle_volume,
    tile_bounding_box,
)
from utils.geometry.mesh_region import (
    MeshRegion,
    PlanarMeshRegion,
    load_obj_triangles,
    project_planar_mesh,
)
from utils.geometry.region import (
    Box,
    Cylinder,
    Difference,
    HalfSpace,
    Intersection,
    Region,
    Sphere,
    Union,
)
from utils.geometry.sampler import fill_region, sample_with_boundary

__all__ = [
    "LATTICE_GRID", "LATTICE_HEX", "tile_bounding_box", "neighbor_offsets",
    "particle_spacing_for_target_count", "per_particle_volume",
    "Region", "Box", "Sphere", "HalfSpace", "Cylinder",
    "Union", "Intersection", "Difference",
    "MeshRegion", "PlanarMeshRegion", "load_obj_triangles", "project_planar_mesh",
    "fill_region", "sample_with_boundary",
]
