"""
mesh_region.py — a Region backed by a triangle mesh, ROBUST TO DIRTY MESHES.

The whole point: real geometry arrives as imperfect triangle soup — holes,
non-manifold edges, duplicated/degenerate vertices, inconsistent orientation,
self-intersections. A naive ray-cast inside/outside test breaks on all of these.

Inside/outside here uses the GENERALIZED WINDING NUMBER (Jacobson, Kavan &
Sorkine-Hornung, SIGGRAPH 2013):

    w(p) = (1 / 4π) · Σ_triangles  Ω(triangle, p)

where Ω is the SIGNED solid angle the triangle subtends at p (Van Oosterom &
Strackee). For a watertight, consistently-oriented mesh w(p) is exactly 1 inside
and 0 outside; for a DIRTY mesh w is a smooth function that stays ≈1 deep inside
and ≈0 far outside, so thresholding at ½ recovers a sensible interior even
through holes and across non-manifold junk. It does not require repair.

The signed distance combines:
  - magnitude : unsigned distance to the nearest triangle (exact closest point);
  - sign      : −1 where the winding number ≥ ½ (inside), +1 outside.

Both are O(query_points × triangles); fine for moderate meshes and for the
sampler's batched evaluation. A BVH / hierarchical winding-number evaluator is
the optimization path for very large meshes (noted, not yet built).

Mesh particles are NEVER placed at mesh vertices — this class only defines the
manifold (for membership/normal) and is the input to feature detection. Tessellation
quality therefore does not leak into the particle distribution.
"""

from __future__ import annotations

import pathlib

import numpy as np

from utils.geometry.region import Region

_FOUR_PI = 4.0 * np.pi


def load_obj_triangles(path) -> tuple[np.ndarray, np.ndarray]:
    """Minimal Wavefront OBJ reader → ``(vertices (V,3), triangles (F,3) int)``.

    Triangulates polygon faces by fan. Ignores everything but ``v`` / ``f``.
    Tolerant of vertex/uv/normal face syntax (``f a/b/c``); negative (relative)
    indices supported. Deliberately dependency-free and forgiving — dirty input
    is the expected case.
    """
    path = pathlib.Path(path)
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    with open(path, "r") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                tokens = line.split()[1:]
                face_indices = []
                for token in tokens:
                    raw = token.split("/")[0]
                    index = int(raw)
                    # OBJ is 1-based; negative indexes count back from the end.
                    face_indices.append(index - 1 if index > 0 else len(vertices) + index)
                for corner in range(1, len(face_indices) - 1):
                    triangles.append((face_indices[0], face_indices[corner], face_indices[corner + 1]))
    return (np.asarray(vertices, dtype=np.float64),
            np.asarray(triangles, dtype=np.int64).reshape(-1, 3))


class MeshRegion(Region):
    """Region from a triangle mesh; dirty-mesh-robust via generalized winding number."""

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray, dimension: int = 3):
        if dimension != 3:
            raise ValueError("MeshRegion is 3D only")
        self.dimension = 3
        self.vertices = np.asarray(vertices, dtype=np.float64)
        triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
        # Drop degenerate triangles (repeated indices) — common dirty-mesh defect
        # that contributes a NaN/zero solid angle. Winding number tolerates the
        # resulting gaps.
        non_degenerate = ~((triangles[:, 0] == triangles[:, 1])
                           | (triangles[:, 1] == triangles[:, 2])
                           | (triangles[:, 0] == triangles[:, 2]))
        self.triangles = triangles[non_degenerate]
        if self.triangles.shape[0] == 0:
            raise ValueError("mesh has no non-degenerate triangles")
        # Cache the three corner-vertex arrays.
        self._corner_a = self.vertices[self.triangles[:, 0]]
        self._corner_b = self.vertices[self.triangles[:, 1]]
        self._corner_c = self.vertices[self.triangles[:, 2]]

    @classmethod
    def from_obj(cls, path) -> "MeshRegion":
        vertices, triangles = load_obj_triangles(path)
        return cls(vertices, triangles)

    # -- inside/outside -----------------------------------------------------

    def winding_number(self, points: np.ndarray) -> np.ndarray:
        """Generalized winding number ``(N,)``; ≈1 inside, ≈0 outside (dirty-robust)."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        accumulated = np.zeros(points.shape[0], dtype=np.float64)
        # Loop over triangles (typically far fewer than query points), vectorizing
        # the solid angle over all points each step.
        for corner_a, corner_b, corner_c in zip(self._corner_a, self._corner_b, self._corner_c):
            accumulated += _signed_solid_angle(points, corner_a, corner_b, corner_c)
        return accumulated / _FOUR_PI

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        unsigned = self._unsigned_distance(points)
        inside = self.winding_number(points) >= 0.5
        return np.where(inside, -unsigned, unsigned)

    def contains(self, points: np.ndarray, inset: float = 0.0) -> np.ndarray:
        # For inset == 0 the winding number alone decides (avoids a distance pass).
        if inset == 0.0:
            return self.winding_number(points) >= 0.5
        return self.signed_distance(points) <= -inset

    # -- distance -----------------------------------------------------------

    def _unsigned_distance(self, points: np.ndarray) -> np.ndarray:
        nearest = np.full(points.shape[0], np.inf, dtype=np.float64)
        for corner_a, corner_b, corner_c in zip(self._corner_a, self._corner_b, self._corner_c):
            nearest = np.minimum(nearest, _point_triangle_distance(points, corner_a, corner_b, corner_c))
        return nearest

    def bounds(self):
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


class PlanarMeshRegion(Region):
    """2D region = the filled area of a planar triangle mesh.

    Membership is point-in-ANY-triangle (a union of the triangles), robust to how
    the polygon was tessellated; holes are naturally empty. The signed-distance
    sign comes from membership; the magnitude is the distance to the nearest
    BOUNDARY edge (an undirected edge incident to exactly one triangle), which
    gives a usable inset for anti-staircase sampling. This is the 2D analogue of
    ``MeshRegion`` (3D winding number + triangle distance).

    ``vertices`` are 2D ``(V, 2)``; project your planar 3D mesh onto its in-plane
    axes before constructing (see ``project_planar_mesh``).
    """

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray):
        self.dimension = 2
        self.vertices = np.asarray(vertices, dtype=np.float64)[:, :2]
        triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
        non_degenerate = ~((triangles[:, 0] == triangles[:, 1])
                           | (triangles[:, 1] == triangles[:, 2])
                           | (triangles[:, 0] == triangles[:, 2]))
        self.triangles = triangles[non_degenerate]
        if self.triangles.shape[0] == 0:
            raise ValueError("planar mesh has no non-degenerate triangles")
        self._corner_a = self.vertices[self.triangles[:, 0]]
        self._corner_b = self.vertices[self.triangles[:, 1]]
        self._corner_c = self.vertices[self.triangles[:, 2]]
        self._boundary_segments = self._extract_boundary_segments()

    def _extract_boundary_segments(self) -> np.ndarray:
        incidence: dict[tuple[int, int], int] = {}
        for triangle in self.triangles:
            for first, second in ((0, 1), (1, 2), (2, 0)):
                edge = (int(triangle[first]), int(triangle[second]))
                key = (min(edge), max(edge))
                incidence[key] = incidence.get(key, 0) + 1
        boundary = [key for key, count in incidence.items() if count == 1]
        if not boundary:                            # closed surface (shouldn't happen for a filled patch)
            boundary = list(incidence.keys())
        return np.array([[self.vertices[a], self.vertices[b]] for a, b in boundary])

    def contains(self, points: np.ndarray, inset: float = 0.0) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)[:, :2]
        if inset == 0.0:
            return self._inside_any_triangle(points)
        return self.signed_distance(points) <= -inset

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)[:, :2]
        inside = self._inside_any_triangle(points)
        distance = self._distance_to_boundary(points)
        return np.where(inside, -distance, distance)

    def bounds(self):
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def _inside_any_triangle(self, points: np.ndarray) -> np.ndarray:
        inside = np.zeros(points.shape[0], dtype=bool)
        for corner_a, corner_b, corner_c in zip(self._corner_a, self._corner_b, self._corner_c):
            cross_ab = _cross_2d(corner_b - corner_a, points - corner_a)
            cross_bc = _cross_2d(corner_c - corner_b, points - corner_b)
            cross_ca = _cross_2d(corner_a - corner_c, points - corner_c)
            # Inside if all cross products share a sign (works for either winding).
            all_non_negative = (cross_ab >= -1e-12) & (cross_bc >= -1e-12) & (cross_ca >= -1e-12)
            all_non_positive = (cross_ab <= 1e-12) & (cross_bc <= 1e-12) & (cross_ca <= 1e-12)
            inside |= all_non_negative | all_non_positive
        return inside

    def _distance_to_boundary(self, points: np.ndarray) -> np.ndarray:
        nearest = np.full(points.shape[0], np.inf, dtype=np.float64)
        for start, end in self._boundary_segments:
            nearest = np.minimum(nearest, _point_segment_distance(points, start, end))
        return nearest


def _cross_2d(edge_vectors, point_vectors) -> np.ndarray:
    """z-component of the 2D cross product. ``edge_vectors`` is (2,) or (N,2)."""
    edge_vectors = np.asarray(edge_vectors, dtype=np.float64)
    if edge_vectors.ndim == 1:
        return edge_vectors[0] * point_vectors[:, 1] - edge_vectors[1] * point_vectors[:, 0]
    return edge_vectors[:, 0] * point_vectors[:, 1] - edge_vectors[:, 1] * point_vectors[:, 0]


def project_planar_mesh(vertices_3d: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Drop the (near-)constant axis of a planar 3D mesh → 2D vertices.

    Returns ``(vertices_2d (V,2), flat_axis, flat_value)`` so the discretized
    points can be lifted back to 3D for output. The flat axis is the one with the
    smallest coordinate range (the plane normal).
    """
    vertices_3d = np.asarray(vertices_3d, dtype=np.float64)
    spans = vertices_3d.max(axis=0) - vertices_3d.min(axis=0)
    flat_axis = int(np.argmin(spans))
    flat_value = float(np.median(vertices_3d[:, flat_axis]))
    in_plane_axes = [axis for axis in range(3) if axis != flat_axis]
    return vertices_3d[:, in_plane_axes], flat_axis, flat_value


# ---------------------------------------------------------------------------
# Vectorized triangle math (fixed triangle, many points)
# ---------------------------------------------------------------------------

def _signed_solid_angle(points, corner_a, corner_b, corner_c) -> np.ndarray:
    """Signed solid angle of one triangle at each point (Van Oosterom–Strackee)."""
    vector_a = corner_a - points
    vector_b = corner_b - points
    vector_c = corner_c - points
    length_a = np.linalg.norm(vector_a, axis=1)
    length_b = np.linalg.norm(vector_b, axis=1)
    length_c = np.linalg.norm(vector_c, axis=1)
    numerator = np.einsum("ij,ij->i", vector_a, np.cross(vector_b, vector_c))
    denominator = (length_a * length_b * length_c
                   + np.einsum("ij,ij->i", vector_a, vector_b) * length_c
                   + np.einsum("ij,ij->i", vector_b, vector_c) * length_a
                   + np.einsum("ij,ij->i", vector_c, vector_a) * length_b)
    return 2.0 * np.arctan2(numerator, denominator)


def _point_triangle_distance(points, corner_a, corner_b, corner_c) -> np.ndarray:
    """Exact distance from each point to the triangle (a, b, c).

    Robust formulation: minimum over the perpendicular in-face projection (when
    the foot is inside the triangle) and the three edge segments. Handles obtuse
    triangles, edges, and vertices without the 7-region branch tangle.
    """
    edge_ab = corner_b - corner_a
    edge_ac = corner_c - corner_a
    normal = np.cross(edge_ab, edge_ac)
    normal_length = np.linalg.norm(normal)

    candidate = np.minimum(
        _point_segment_distance(points, corner_a, corner_b),
        np.minimum(
            _point_segment_distance(points, corner_b, corner_c),
            _point_segment_distance(points, corner_c, corner_a)))

    if normal_length > 1.0e-30:
        unit_normal = normal / normal_length
        relative = points - corner_a
        perpendicular = relative @ unit_normal
        foot = points - np.outer(perpendicular, unit_normal)
        inside = _point_in_triangle(foot, corner_a, corner_b, corner_c, unit_normal)
        candidate = np.where(inside, np.minimum(candidate, np.abs(perpendicular)), candidate)
    return candidate


def _point_segment_distance(points, end_p, end_q) -> np.ndarray:
    segment = end_q - end_p
    length_squared = float(segment @ segment)
    if length_squared < 1.0e-30:
        return np.linalg.norm(points - end_p, axis=1)
    parameter = np.clip(((points - end_p) @ segment) / length_squared, 0.0, 1.0)
    closest = end_p + np.outer(parameter, segment)
    return np.linalg.norm(points - closest, axis=1)


def _point_in_triangle(points, corner_a, corner_b, corner_c, unit_normal) -> np.ndarray:
    """Whether each (assumed coplanar) point is inside triangle a,b,c (same-side test)."""
    def same_side(edge_start, edge_end):
        edge = edge_end - edge_start
        to_point = points - edge_start
        cross = np.cross(np.broadcast_to(edge, to_point.shape), to_point)
        return cross @ unit_normal >= -1.0e-12
    return same_side(corner_a, corner_b) & same_side(corner_b, corner_c) & same_side(corner_c, corner_a)
