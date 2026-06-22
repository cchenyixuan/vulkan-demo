"""
_test_geometry.py — self-contained tests for utils.geometry.

Run:  .venv/Scripts/python.exe utils/geometry/_test_geometry.py

Covers: lattice geometry + counts, lattice/calibrator agreement (the
single-source-of-truth claim), analytic SDF regions + CSG, dirty-mesh
robustness via generalized winding number, and the sampler.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.geometry import (                                   # noqa: E402
    LATTICE_GRID, LATTICE_HEX, tile_bounding_box, neighbor_offsets,
    particle_spacing_for_target_count, per_particle_volume,
    Box, Sphere, Difference, Intersection, fill_region, MeshRegion,
)

_passed = 0
_failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}   {detail}")


# ---------------------------------------------------------------------------
# 1. Lattice geometry
# ---------------------------------------------------------------------------

def test_lattice_geometry():
    print("[1] lattice geometry")
    spacing = 0.1

    # grid 2D: square unit area -> ~ (1/s + 1)^2 sites
    grid2 = tile_bounding_box([0, 0], [1, 1], spacing, LATTICE_GRID, 2)
    check("grid2 count", abs(grid2.shape[0] - 11 * 11) <= 0, f"got {grid2.shape[0]}")

    # grid 3D
    grid3 = tile_bounding_box([0, 0, 0], [1, 1, 1], spacing, LATTICE_GRID, 3)
    check("grid3 count", abs(grid3.shape[0] - 11 ** 3) <= 0, f"got {grid3.shape[0]}")

    # hex 2D: 6 nearest neighbors at exactly the spacing
    hex2 = tile_bounding_box([-1, -1], [1, 1], spacing, LATTICE_HEX, 2)
    origin_index = np.argmin(np.linalg.norm(hex2, axis=1))
    distances = np.linalg.norm(hex2 - hex2[origin_index], axis=1)
    distances = np.sort(distances)[1:]            # drop self (0)
    nearest_six = distances[:6]
    check("hex2 nn distance == spacing", np.allclose(nearest_six, spacing, atol=1e-9),
          f"{nearest_six}")
    check("hex2 has 6 neighbors at spacing", np.sum(np.abs(distances - spacing) < 1e-9) == 6)

    # FCC 3D: 12 nearest neighbors at exactly the spacing
    fcc = tile_bounding_box([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], spacing, LATTICE_HEX, 3)
    origin_index = np.argmin(np.linalg.norm(fcc, axis=1))
    distances = np.sort(np.linalg.norm(fcc - fcc[origin_index], axis=1))[1:]
    check("fcc nn distance == spacing", abs(distances[0] - spacing) < 1e-9, f"{distances[0]}")
    check("fcc has 12 neighbors at spacing", np.sum(np.abs(distances - spacing) < 1e-9) == 12,
          f"{np.sum(np.abs(distances - spacing) < 1e-9)}")

    # packing density: hex 2D number density = 2/√3 x grid (exact, analytic cells)
    density_ratio = (per_particle_volume(LATTICE_GRID, 2, spacing)
                     / per_particle_volume(LATTICE_HEX, 2, spacing))
    check("hex2 number density = 2/sqrt(3) x grid (analytic)",
          abs(density_ratio - 2 / math.sqrt(3)) < 1e-12, f"{density_ratio:.6f}")
    # finite fill approaches that ratio on a large domain (boundary effects shrink)
    fine = 0.05
    big_hex = tile_bounding_box([-3, -3], [3, 3], fine, LATTICE_HEX, 2).shape[0]
    big_grid = tile_bounding_box([-3, -3], [3, 3], fine, LATTICE_GRID, 2).shape[0]
    check("hex2 finite fill ~15% denser", abs(big_hex / big_grid - 2 / math.sqrt(3)) < 0.03,
          f"{big_hex / big_grid:.4f}")


def test_target_count_inversion():
    print("[2] particle_spacing_for_target_count round-trip")
    # The inversion's contract is analytic: measure / cell-volume == target exactly.
    # (Realized fill count differs by boundary/fencepost effects — negligible at the
    # millions-of-particles scale we generate, ~6-9% only at tiny targets.)
    for lattice in (LATTICE_GRID, LATTICE_HEX):
        for dimension, measure in ((2, 1.0), (3, 1.0)):
            target = 10_000
            spacing = particle_spacing_for_target_count(target, measure, lattice, dimension)
            predicted = measure / per_particle_volume(lattice, dimension, spacing)
            check(f"{lattice} {dimension}D analytic round-trip", abs(predicted - target) < 1e-6,
                  f"predicted {predicted:.3f}")
    # realized fill converges to target as the boundary fraction shrinks (large N)
    spacing = particle_spacing_for_target_count(1_000_000, 1.0, LATTICE_GRID, 2)
    count = tile_bounding_box([0, 0], [1, 1], spacing, LATTICE_GRID, 2).shape[0]
    check(f"grid 2D realized fill ~ target at 1M ({count})", abs(count - 1_000_000) / 1e6 < 0.01,
          f"err {abs(count - 1_000_000) / 1e6:.4f}")


def test_lattice_calibrator_agreement():
    """The lattice module and the V4 solver's volume calibration must encode the
    SAME geometry. Independently sum the Wendland kernel over neighbor_offsets and
    compare 1/Σ to _calibrate_particle_volume."""
    print("[3] lattice <-> V4 calibrator agreement")
    from experiment.v4.utils.case_loader_v4 import _calibrate_particle_volume

    smoothing_length = 0.005
    particle_radius = 0.0005
    spacing = 2.0 * particle_radius

    def wendland(distance, dimension):
        q = distance / smoothing_length
        if q >= 1.0:
            return 0.0
        coefficient = (9.0 / (math.pi * smoothing_length ** 2) if dimension == 2
                       else 495.0 / (32.0 * math.pi * smoothing_length ** 3))
        return coefficient * (1.0 - q) ** 6 * ((35.0 / 3.0) * q * q + 6.0 * q + 1.0)

    for lattice_kind, calibrator_name in ((LATTICE_GRID, "grid"), (LATTICE_HEX, "hex")):
        for dimension in (2, 3):
            offsets = neighbor_offsets(lattice_kind, dimension, spacing, smoothing_length)
            kernel_sum = sum(wendland(float(np.linalg.norm(o)), dimension) for o in offsets)
            my_volume = 1.0 / kernel_sum
            reference = _calibrate_particle_volume(
                smoothing_length, particle_radius, dimension, calibrator_name)
            relative_error = abs(my_volume - reference) / reference
            check(f"{calibrator_name} {dimension}D volume matches calibrator",
                  relative_error < 1e-9, f"mine={my_volume:.6e} ref={reference:.6e}")


# ---------------------------------------------------------------------------
# 4. Analytic SDF regions
# ---------------------------------------------------------------------------

def test_analytic_regions():
    print("[4] analytic SDF regions + CSG")
    box = Box([0, 0, 0], [2, 2, 2])
    check("box inside center sdf", abs(box.signed_distance(np.array([[1, 1, 1.0]]))[0] - (-1.0)) < 1e-9)
    check("box outside sdf", abs(box.signed_distance(np.array([[3, 1, 1.0]]))[0] - 1.0) < 1e-9)
    check("box contains center", bool(box.contains(np.array([[1, 1, 1.0]]))[0]))
    check("box rejects outside", not bool(box.contains(np.array([[3, 1, 1.0]]))[0]))

    sphere = Sphere([0, 0, 0], 1.0)
    check("sphere exact sdf", abs(sphere.signed_distance(np.array([[2, 0, 0.0]]))[0] - 1.0) < 1e-9)

    # normal via central difference ~ analytic (outward radial) on the sphere
    point = np.array([[0.7, 0.1, -0.2]])
    numeric = sphere.surface_normal(point)[0]
    analytic = point[0] / np.linalg.norm(point[0])
    check("sphere normal ~ radial", np.allclose(numeric, analytic, atol=1e-4), f"{numeric}")

    # Difference: box with a spherical bite taken out of one corner
    carved = Difference(Box([0, 0, 0], [2, 2, 2]), Sphere([0, 0, 0], 1.0))
    check("difference removes carved point", not bool(carved.contains(np.array([[0.2, 0.2, 0.2]]))[0]))
    check("difference keeps far point", bool(carved.contains(np.array([[1.8, 1.8, 1.8]]))[0]))


# ---------------------------------------------------------------------------
# 5. Dirty-mesh robustness (generalized winding number)
# ---------------------------------------------------------------------------

def _unit_cube_mesh():
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1.0]])
    triangles = np.array([
        [0, 3, 2], [0, 2, 1],     # bottom (-z)
        [4, 5, 6], [4, 6, 7],     # top (+z)
        [0, 1, 5], [0, 5, 4],     # front (-y)
        [3, 7, 6], [3, 6, 2],     # back (+y)
        [0, 4, 7], [0, 7, 3],     # left (-x)
        [1, 2, 6], [1, 6, 5]])    # right (+x)
    return vertices, triangles


def test_mesh_robustness():
    print("[5] dirty-mesh robustness (generalized winding number)")
    vertices, triangles = _unit_cube_mesh()
    inside_points = np.array([[0.5, 0.5, 0.5], [0.2, 0.2, 0.8], [0.9, 0.5, 0.1]])
    outside_points = np.array([[2.0, 0.5, 0.5], [-1.0, 0.5, 0.5], [0.5, 0.5, 3.0]])

    clean = MeshRegion(vertices, triangles)
    w_in = clean.winding_number(inside_points)
    w_out = clean.winding_number(outside_points)
    check("clean: inside winding ~ 1", np.allclose(w_in, 1.0, atol=1e-6), f"{w_in}")
    check("clean: outside winding ~ 0", np.allclose(w_out, 0.0, atol=1e-6), f"{w_out}")
    check("clean: contains inside", bool(np.all(clean.contains(inside_points))))
    check("clean: rejects outside", bool(not np.any(clean.contains(outside_points))))
    # signed distance magnitude sanity: center is 0.5 from nearest face
    center_distance = clean.signed_distance(np.array([[0.5, 0.5, 0.5]]))[0]
    check("clean: center sdf ~ -0.5", abs(center_distance + 0.5) < 1e-6, f"{center_distance}")

    # (a) HOLE: drop both triangles of the top (+z) face -> open box.
    holed = MeshRegion(vertices, np.delete(triangles, [2, 3], axis=0))
    w_in_hole = holed.winding_number(inside_points)
    check("HOLE: deep-inside still > 0.5", bool(np.all(w_in_hole > 0.5)), f"{w_in_hole}")
    check("HOLE: outside still < 0.5", bool(np.all(holed.winding_number(outside_points) < 0.5)))
    check("HOLE: contains() still right",
          bool(np.all(holed.contains(inside_points)) and not np.any(holed.contains(outside_points))))

    # (b) DEGENERATE triangle (repeated index) appended -> must be dropped, no NaN.
    dirty = np.vstack([triangles, np.array([[5, 5, 5]])])
    degen = MeshRegion(vertices, dirty)
    check("DEGENERATE: dropped, count unchanged", degen.triangles.shape[0] == triangles.shape[0])
    check("DEGENERATE: no NaN, inside ~ 1", np.allclose(degen.winding_number(inside_points), 1.0, atol=1e-6))

    # (c) DUPLICATE vertices (mesh soup) -> add a duplicate vertex, reindex one triangle to it.
    vertices_dup = np.vstack([vertices, vertices[6]])
    triangles_dup = triangles.copy()
    triangles_dup[10] = [1, 2, 8]                   # vertex 8 == duplicate of 6
    duped = MeshRegion(vertices_dup, triangles_dup)
    check("DUPLICATE-VERT: inside ~ 1", np.allclose(duped.winding_number(inside_points), 1.0, atol=1e-6),
          f"{duped.winding_number(inside_points)}")


# ---------------------------------------------------------------------------
# 6. Sampler
# ---------------------------------------------------------------------------

def test_sampler():
    print("[6] sampler fill_region")
    box = Box([0, 0], [1, 1])
    spacing = 0.02
    points = fill_region(box, spacing, LATTICE_GRID, inset_fraction=0.0)
    check("fill: all strictly inside", bool(np.all(box.signed_distance(points) <= 1e-12)))
    # ~ (1/0.02 + 1)^2 = 51^2 = 2601 with inset 0 and pad covering the faces
    check("fill: count plausible", 2400 < points.shape[0] < 2800, f"{points.shape[0]}")

    # inset by half a spacing removes the outermost ring
    inset_points = fill_region(box, spacing, LATTICE_GRID, inset_fraction=0.5)
    check("fill: inset reduces count", inset_points.shape[0] < points.shape[0],
          f"{inset_points.shape[0]} vs {points.shape[0]}")

    # mesh fill: carve particles inside the unit cube, all must be inside
    vertices, triangles = _unit_cube_mesh()
    cube = MeshRegion(vertices, triangles)
    # inset by half a spacing so no particle lands exactly on a face (winding ~0.5,
    # the genuinely-ambiguous boundary) — the recommended sampling default.
    cube_points = fill_region(cube, 0.1, LATTICE_GRID, inset_fraction=0.5)
    check("mesh fill: nonempty", cube_points.shape[0] > 0, f"{cube_points.shape[0]}")
    check("mesh fill: all inside", bool(np.all(cube.contains(cube_points))))


def main() -> int:
    test_lattice_geometry()
    test_target_count_inversion()
    test_lattice_calibrator_agreement()
    test_analytic_regions()
    test_mesh_robustness()
    test_sampler()
    print(f"\n{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
