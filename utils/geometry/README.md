# utils.geometry — particle geometry discretization

A standalone, general-purpose toolkit that turns continuous geometry into particle
point clouds for SPH / mesh-free methods. It has **no dependency on the SPH solver**
(it is pure geometry), but its lattice definitions are bit-for-bit consistent with
the V4 solver's volume calibration, so a cloud it produces drops straight into the
pipeline.

It does four things:

1. **Lattices** — Cartesian (`grid`) and closest-packed (`hex` = 2D triangular /
   3D FCC), N-dimensional, vectorized to tens of millions of sites.
2. **Regions** — geometry defined by a signed-distance function: analytic primitives
   + CSG booleans, **and triangle meshes that are robust to dirty input** (holes,
   non-manifold edges, soup) via generalized winding numbers.
3. **Sampling** — fill a region with a lattice, including a labeled **boundary shell**
   (interior fluid + N-layer surface particles).
4. **Relaxation (optional)** — constrained packing that conforms the boundary to
   curved surfaces. *Only useful for strongly curved geometry — see the note below.*

---

## Install / dependencies

Core (`lattice`, `region`, `mesh_region`, `sampler`): **numpy** only.
Relaxation (`relax`): **scipy** (`cKDTree`). Demos also use **matplotlib**.

```bash
.venv/Scripts/python.exe -m pip install numpy scipy matplotlib
```

## Conventions

- **Particle spacing** `particle_spacing` = particle diameter = `2·particle_radius`
  = nearest-neighbor distance. (The SPH cases use smoothing length `h = 5·dx`.)
- Points are `(N, dimension)` float64 arrays; `dimension ∈ {2, 3}`.
- A region's signed distance is `< 0` strictly inside, `0` on the surface, `> 0` outside.
- **Full words, no abbreviations** in identifiers (`particle_spacing`, not `dx`),
  matching the repo convention; math symbols appear only in comments.

---

## Module map

| file | role |
|---|---|
| `lattice.py`     | N-D lattices: `tile_bounding_box`, `particle_spacing_for_target_count`, `per_particle_volume`, `neighbor_offsets`. **Single source of truth**, matched to the solver's `_calibrate_particle_volume`. |
| `region.py`      | SDF `Region` base + analytic `Box`, `Sphere`, `HalfSpace`, `Cylinder` + CSG `Union`/`Intersection`/`Difference`. `surface_normal` via central differences (works for any region). |
| `mesh_region.py` | `MeshRegion` (3D, generalized-winding-number, **dirty-mesh robust**) + `PlanarMeshRegion` (2D filled polygon) + OBJ loader + `project_planar_mesh`. |
| `sampler.py`     | `fill_region` (interior only) and `sample_with_boundary` (interior + boundary shell). SDF-inset membership, batched for large clouds. |
| `relax.py`       | `relax_particles` — constrained packing relaxation (optional stage). |
| `_test_geometry.py` | 41 unit tests (lattice geometry, calibrator agreement, SDF/CSG, dirty-mesh robustness, sampler). |
| `_demo_*.py`     | runnable demos (see below). |
| `test_cases/`    | input `.obj` meshes; outputs go to `test_cases/out/` (gitignored). |

---

## Core concepts

### Lattices (`grid` / `hex`)

| lattice | 2D | 3D | nn distance | packing |
|---|---|---|---|---|
| `grid` | square | simple cubic | `particle_spacing` | π/4 · / · π/6 |
| `hex`  | triangular (6 nn) | FCC (12 nn) | `particle_spacing` | 0.9069 / 0.7405 |

These match `experiment/v4/.../case_loader_v4._calibrate_particle_volume` exactly
(`lattice: grid|hex` in the case YAML), so the SPH partition-of-unity volume
`V = 1/Σ W` is correct. Test `[3]` asserts the agreement to 1e-9. HCP is **not**
implemented — `hex`/FCC already gives closest packing.

### Regions (SDF + dirty-mesh robustness)

Everything derives from `signed_distance(points)`. Analytic primitives compose with
exact CSG (`Union` = min, `Intersection` = max, `Difference` = max(A, −B)). Triangle
meshes use the **generalized winding number** (Jacobson et al. 2013): `w ≈ 1` inside,
`≈ 0` outside, and it stays correct through holes / non-manifold / inconsistent
orientation, so **no mesh repair is required**. Validated on a cube with holes,
degenerate triangles, and duplicate-vertex soup, and on the supplied `test_3d_hole`
(32-edge open mesh — solid filled across the hole, no leak).

> Particles are **never** placed at mesh vertices — the mesh only defines the manifold
> (and, for relaxation, the feature network). Tessellation quality therefore does not
> leak into the particle distribution.

### Sampling: interior + boundary shell

`fill_region` returns interior particles (inset by ½ spacing from the surface to avoid
boundary aliasing). `sample_with_boundary` additionally returns a labeled **boundary
shell** — the outermost `boundary_layers` lattice layers, an SPH dummy/ghost wall on
the same lattice as the interior (no interface mismatch). Set
`boundary_layers ≥ ceil(h/dx) ≈ 5` for full kernel support at the wall.

### Relaxation (optional — a curved-surface tool)

`relax_particles` does constrained packing: short-range repulsion equalizes spacing
while each particle is projected by its constraint class (`FREE` interior / `SURFACE`
slides on the manifold / `FEATURE_EDGE` slides along a sharp edge / `FROZEN` deep
interior). It never calls the expensive winding-number SDF in the loop (a fast surface
proxy answers closest-point/normal queries) and only relaxes the near-surface band.

**When to use it — measured:**
- **Strongly curved surface** (e.g. a sphere): dramatic. Surface-skin radial spread
  collapses `0.318·dx → 0` (exact conform to the true surface).
- **Axis-aligned / mildly-angled** geometry (the lid-driven cavity, and the bundled
  test meshes): **not worth it.** The grid fill is already a perfect lattice
  (NN = 1.000·dx, CV = 0); relaxation only cosmetically smooths angled-boundary
  staircasing, while slightly glassifying the bulk (CV → ~0.04) and introducing mild
  surface clumping (min NN ~0.6·dx), with no density-uniformity gain.

So relaxation is **off the default path**: use it only for curved geometry. The simple
repulsion has a residual clumping limitation; a production version would need
surface-CVT + surface particle-count management.

---

## Public API

```python
from utils.geometry import (
    LATTICE_GRID, LATTICE_HEX,
    tile_bounding_box, particle_spacing_for_target_count, per_particle_volume, neighbor_offsets,
    Region, Box, Sphere, HalfSpace, Cylinder, Union, Intersection, Difference,
    MeshRegion, PlanarMeshRegion, load_obj_triangles, project_planar_mesh,
    fill_region, sample_with_boundary,
)
from utils.geometry.relax import relax_particles, FROZEN, FREE, SURFACE, FEATURE_EDGE, PINNED
```

Key signatures:

```python
tile_bounding_box(bounding_box_min, bounding_box_max, particle_spacing, lattice_kind, dimension) -> (N, dim)
particle_spacing_for_target_count(target_particle_count, region_measure, lattice_kind, dimension) -> float
per_particle_volume(lattice_kind, dimension, particle_spacing) -> float

Region.signed_distance(points) -> (N,)          # < 0 inside
Region.contains(points, inset=0.0) -> (N,) bool
Region.surface_normal(points) -> (N, dim)       # outward unit normal

Box(minimum_corner, maximum_corner); Sphere(center, radius)
HalfSpace(anchor, outward_normal); Cylinder(base_center, axis_vector, radius)
Union(*children); Intersection(*children); Difference(base, subtracted)

MeshRegion(vertices, triangles)        # or MeshRegion.from_obj(path)
PlanarMeshRegion(vertices_2d, triangles)
load_obj_triangles(path) -> (vertices(V,3), triangles(F,3))
project_planar_mesh(vertices_3d) -> (vertices_2d, flat_axis, flat_value)

fill_region(region, particle_spacing, lattice_kind, inset_fraction=0.5) -> (N, dim)
sample_with_boundary(region, particle_spacing, lattice_kind, boundary_layers=3) -> (interior, boundary)

relax_particles(points, region, particle_spacing, *, iterations=40, active_band=4.0,
                surface_band=0.8, feature_radius=0.7, cutoff_factor=1.8, step_fraction=0.18,
                dihedral_degrees=35.0, use_features=True, pin_corners=False, verbose=False)
    -> (relaxed_points, constraint_classes)
```

---

## Examples

**Analytic region — solid box with a spherical bite, hex packed, with a wall shell:**

```python
from utils.geometry import Box, Sphere, Difference, sample_with_boundary, LATTICE_HEX

region = Difference(Box([-1, -1, -1], [1, 1, 1]), Sphere([1, 1, 1], 0.6))
interior, boundary = sample_with_boundary(region, 0.02, LATTICE_HEX, boundary_layers=5)
```

**Hit a target particle count (grid, 2D):**

```python
from utils.geometry import particle_spacing_for_target_count, Box, fill_region, LATTICE_GRID

spacing = particle_spacing_for_target_count(16_000_000, region_measure=1.0,
                                            lattice_kind=LATTICE_GRID, dimension=2)
points = fill_region(Box([0, 0], [1, 1]), spacing, LATTICE_GRID)
```

**Dirty triangle mesh (holes / non-manifold OK):**

```python
from utils.geometry import MeshRegion, sample_with_boundary, LATTICE_GRID

region = MeshRegion.from_obj("model.obj")           # winding-number robust
interior, boundary = sample_with_boundary(region, 0.02, LATTICE_GRID)
```

**2D planar mesh (Blender plane in the X-Z plane):**

```python
from utils.geometry import load_obj_triangles, project_planar_mesh, PlanarMeshRegion, sample_with_boundary, LATTICE_GRID

vertices_3d, triangles = load_obj_triangles("plane.obj")
vertices_2d, flat_axis, flat_value = project_planar_mesh(vertices_3d)
region = PlanarMeshRegion(vertices_2d, triangles)
interior, boundary = sample_with_boundary(region, 0.02, LATTICE_GRID)
```

**Relaxation (curved geometry only):**

```python
from utils.geometry.relax import relax_particles
relaxed, classes = relax_particles(points, region, 0.02, iterations=40)
```

---

## Tests & demos

```bash
# unit tests (41) — lattice geometry, calibrator agreement, SDF/CSG, dirty-mesh robustness
.venv/Scripts/python.exe utils/geometry/_test_geometry.py

# discretize the bundled 2D + 3D meshes (grid + hex, interior + boundary shell) -> out/preview_*.png
.venv/Scripts/python.exe utils/geometry/_demo_discretize.py

# open-mesh hole case (winding number bridges it) -> out/preview_3d_hole.png + hole_robustness.png
.venv/Scripts/python.exe utils/geometry/_demo_hole.py

# fill + relaxation, before/after, on the test cases  (--only NAME for a subset)
.venv/Scripts/python.exe utils/geometry/_demo_relax.py --only 2d

# relaxation on a curved surface (analytic sphere) — where it actually helps
.venv/Scripts/python.exe utils/geometry/_demo_sphere.py
```

Outputs (point-cloud `.obj` + preview `.png`) are written to `test_cases/out/`,
which is gitignored.

---

## Design notes / decisions

- **Lattice is the single source of truth**, shared in spirit with the solver's volume
  calibration so placement and calibration cannot drift (asserted by a unit test).
- **Dirty meshes need no repair** — generalized winding number handles holes,
  non-manifold edges, and soup; degenerate triangles are dropped.
- **Particles never sit on mesh vertices** — avoids inheriting a bad tessellation.
- **Quality comes from sampling + (optional) constrained relaxation, not perfect
  placement.** For axis-aligned/lattice-conforming geometry the cookie-cutter fill is
  already exact, so relaxation is reserved for curved surfaces.
- **N-D core, 2D validated first.** Grid-2D reproduces an existing Blender reference
  fill exactly (1,002,001 particles); 3D primitives are present and the solver's 3D
  volume calibration is ready.

## Limitations & roadmap

- Mesh winding number / point-triangle distance is `O(points × triangles)` — fine for
  moderate meshes (≈1 M points over a few hundred triangles in tens of seconds). A
  BVH / hierarchical winding-number evaluator is the optimization for large meshes.
- Relaxation uses simple repulsion → mild residual clumping (min NN ~0.6–0.7·dx). A
  surface-CVT variant with surface particle-count management would remove it; only
  worth building when strongly-curved production geometry arrives.
- Per-particle volume `V = 1/ΣW` (for relaxed / non-lattice clouds) and an
  `.obj` + `case.yaml` writer / CLI are the next pieces (the latter to generate the
  1M/4M/8M/16M scaling cases — which use analytic boxes and need no relaxation).
```
