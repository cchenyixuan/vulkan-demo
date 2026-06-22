"""
relax.py — constrained packing relaxation of a particle cloud.

Takes the approximate lattice fill (interior + boundary shell) and relaxes it to a
uniform, boundary-conforming, isotropic distribution: equalizes spacing (flattens
the per-particle density deviation) and conforms the boundary shell to angled /
curved surfaces, removing the lattice staircasing.

Algorithm = damped short-range repulsion + per-particle constrained projection
(the design we agreed):
  - FREE      (interior near-surface band): repelled freely, clamped to stay inside.
  - SURFACE   (outer skin on a smooth face): projected onto the surface manifold
              each iteration -> slides tangentially, equalizing surface spacing.
  - FEATURE   (near a sharp edge): projected onto the feature curve (1D slide).
  - PINNED    (a true corner): one particle frozen at the corner vertex.
  - FROZEN    (deep interior): already a perfect lattice; not moved, but still acts
              as a repulsion source so the active band packs seamlessly against it.

Performance: the loop NEVER calls the expensive winding-number SDF. A "surface
proxy" (dense surface samples + KDTree + per-sample outward normal) answers
closest-point / normal / signed-offset queries in O(log N). Only the near-surface
band moves; the deep interior is frozen. Neighbor search uses scipy cKDTree.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from utils.geometry.mesh_region import MeshRegion, PlanarMeshRegion
from utils.geometry.region import Region

FROZEN, FREE, SURFACE, FEATURE_EDGE, PINNED = 0, 1, 2, 3, 4


# ===========================================================================
# Surface proxy — fast closest-point / normal / signed-offset (no SDF in loop)
# ===========================================================================

class SurfaceProxy:
    """Dense surface samples + outward normals + KDTree for O(log N) queries."""

    def __init__(self, samples: np.ndarray, normals: np.ndarray):
        self.samples = np.asarray(samples, dtype=np.float64)
        self.normals = np.asarray(normals, dtype=np.float64)
        self.tree = cKDTree(self.samples)

    def query(self, points: np.ndarray):
        """Return (closest_sample, outward_normal, signed_offset, distance)."""
        distance, index = self.tree.query(points)
        closest = self.samples[index]
        normal = self.normals[index]
        signed_offset = np.einsum("ij,ij->i", points - closest, normal)
        return closest, normal, signed_offset, distance


def _triangle_surface_samples(vertices, triangles, sample_spacing):
    """Dense points + outward normals sampled over every triangle (barycentric)."""
    sample_points, sample_normals = [], []
    for triangle in triangles:
        corner_a, corner_b, corner_c = vertices[triangle]
        normal = np.cross(corner_b - corner_a, corner_c - corner_a)
        norm_length = np.linalg.norm(normal)
        if norm_length < 1e-30:
            continue
        normal = normal / norm_length
        longest_edge = max(np.linalg.norm(corner_b - corner_a),
                           np.linalg.norm(corner_c - corner_a),
                           np.linalg.norm(corner_c - corner_b))
        divisions = max(1, int(np.ceil(longest_edge / sample_spacing)))
        index_i, index_j = np.meshgrid(np.arange(divisions + 1), np.arange(divisions + 1), indexing="ij")
        keep = (index_i + index_j) <= divisions
        weight_a = index_i[keep] / divisions
        weight_b = index_j[keep] / divisions
        weight_c = 1.0 - weight_a - weight_b
        points = (weight_a[:, None] * corner_a + weight_b[:, None] * corner_b
                  + weight_c[:, None] * corner_c)
        sample_points.append(points)
        sample_normals.append(np.broadcast_to(normal, points.shape))
    return np.concatenate(sample_points, axis=0), np.concatenate(sample_normals, axis=0)


def _planar_surface_samples(region: PlanarMeshRegion, sample_spacing):
    """Sample points + outward normals along the 2D boundary segments."""
    sample_points, sample_normals = [], []
    for start, end in region._boundary_segments:
        edge = end - start
        length = np.linalg.norm(edge)
        if length < 1e-30:
            continue
        outward = np.array([edge[1], -edge[0]]) / length          # perpendicular
        midpoint = 0.5 * (start + end)
        if region.contains(np.array([midpoint + 0.05 * length * outward]))[0]:
            outward = -outward                                    # flip to point outside
        divisions = max(1, int(np.ceil(length / sample_spacing)))
        parameter = np.linspace(0.0, 1.0, divisions + 1)[:, None]
        points = start + parameter * edge
        sample_points.append(points)
        sample_normals.append(np.broadcast_to(outward, points.shape))
    return np.concatenate(sample_points, axis=0), np.concatenate(sample_normals, axis=0)


class RegionSurfaceProxy:
    """Surface proxy for an ANALYTIC region (Sphere, Box, CSG, ...): uses the
    region's own exact, cheap SDF + normal — no sampling/KDTree needed."""

    def __init__(self, region: Region):
        self.region = region

    def query(self, points):
        signed = self.region.signed_distance(points)
        normal = self.region.surface_normal(points)
        closest = points - signed[:, None] * normal
        return closest, normal, signed, np.abs(signed)


def build_surface_proxy(region: Region, particle_spacing: float):
    sample_spacing = 0.5 * particle_spacing
    if isinstance(region, MeshRegion):
        samples, normals = _triangle_surface_samples(region.vertices, region.triangles, sample_spacing)
        return SurfaceProxy(samples, normals)
    if isinstance(region, PlanarMeshRegion):
        samples, normals = _planar_surface_samples(region, sample_spacing)
        return SurfaceProxy(samples, normals)
    return RegionSurfaceProxy(region)          # analytic region: exact SDF + normal


# ===========================================================================
# Feature extraction (sharp edges + corners)
# ===========================================================================

def feature_edges_mesh(vertices, triangles, dihedral_degrees=35.0):
    """Sharp feature edges (segments) + corner vertices, by dihedral angle."""
    face_normal = np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                           vertices[triangles[:, 2]] - vertices[triangles[:, 0]])
    lengths = np.linalg.norm(face_normal, axis=1, keepdims=True)
    face_normal = face_normal / np.where(lengths < 1e-30, 1.0, lengths)
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, triangle in enumerate(triangles):
        for first, second in ((0, 1), (1, 2), (2, 0)):
            key = (min(int(triangle[first]), int(triangle[second])),
                   max(int(triangle[first]), int(triangle[second])))
            edge_faces.setdefault(key, []).append(face_index)
    cosine_threshold = np.cos(np.radians(dihedral_degrees))
    segments, corner_indices = [], set()
    for (vertex_a, vertex_b), faces in edge_faces.items():
        sharp = False
        if len(faces) == 1:                                       # open boundary edge = feature
            sharp = True
        elif len(faces) == 2:
            if float(np.dot(face_normal[faces[0]], face_normal[faces[1]])) < cosine_threshold:
                sharp = True
        if sharp:
            segments.append([vertices[vertex_a], vertices[vertex_b]])
            corner_indices.update((vertex_a, vertex_b))
    segments = np.array(segments) if segments else np.zeros((0, 2, 3))
    # A corner = a vertex where >=3 sharp edges meet (true geometric corner).
    incidence: dict[int, int] = {}
    for (vertex_a, vertex_b), faces in edge_faces.items():
        if (len(faces) == 1
                or (len(faces) == 2 and float(np.dot(face_normal[faces[0]], face_normal[faces[1]])) < cosine_threshold)):
            incidence[vertex_a] = incidence.get(vertex_a, 0) + 1
            incidence[vertex_b] = incidence.get(vertex_b, 0) + 1
    corners = np.array([vertices[v] for v, count in incidence.items() if count >= 3]) \
        if any(c >= 3 for c in incidence.values()) else np.zeros((0, 3))
    return segments, corners


def feature_corners_planar(region: PlanarMeshRegion, angle_degrees=35.0):
    """Polygon corners (2D): boundary vertices where the edge direction turns sharply.

    Returns (segments(0,2,2) empty — 2D boundary itself is the 'surface', not the
    feature; sharp corners are the features), corners (K,2).
    """
    # Build ordered boundary loops from segments.
    segments = region._boundary_segments
    adjacency: dict[tuple, list] = {}
    for start, end in segments:
        adjacency.setdefault(tuple(np.round(start, 9)), []).append(tuple(np.round(end, 9)))
        adjacency.setdefault(tuple(np.round(end, 9)), []).append(tuple(np.round(start, 9)))
    cosine_threshold = np.cos(np.radians(180.0 - angle_degrees))
    corners = []
    for vertex, neighbors in adjacency.items():
        if len(neighbors) != 2:
            corners.append(vertex)                                # non-manifold boundary -> treat as corner
            continue
        point = np.array(vertex)
        edge_one = np.array(neighbors[0]) - point
        edge_two = np.array(neighbors[1]) - point
        edge_one /= (np.linalg.norm(edge_one) + 1e-30)
        edge_two /= (np.linalg.norm(edge_two) + 1e-30)
        if float(np.dot(edge_one, edge_two)) > cosine_threshold:  # turn sharper than threshold
            corners.append(vertex)
    corners = np.array(corners) if corners else np.zeros((0, 2))
    return np.zeros((0, 2, 2)), corners


# ===========================================================================
# Main relaxation
# ===========================================================================

def relax_particles(
    points: np.ndarray,
    region: Region,
    particle_spacing: float,
    *,
    iterations: int = 40,
    active_band: float = 4.0,
    surface_band: float = 0.8,
    feature_radius: float = 0.7,
    cutoff_factor: float = 1.8,
    step_fraction: float = 0.18,
    dihedral_degrees: float = 35.0,
    use_features: bool = True,
    pin_corners: bool = False,
    verbose: bool = False,
):
    """Relax ``points`` in-place-style (returns a new array) and a per-particle
    constraint-class array. Bands are in units of ``particle_spacing``."""
    points = np.array(points, dtype=np.float64)
    dimension = points.shape[1]
    proxy = build_surface_proxy(region, particle_spacing)

    # --- features --------------------------------------------------------
    feature_segments = np.zeros((0, 2, dimension))
    feature_corners = np.zeros((0, dimension))
    if use_features:
        if isinstance(region, MeshRegion):
            feature_segments, feature_corners = feature_edges_mesh(
                region.vertices, region.triangles, dihedral_degrees)
        elif isinstance(region, PlanarMeshRegion):
            feature_segments, feature_corners = feature_corners_planar(region, dihedral_degrees)

    # --- classify --------------------------------------------------------
    _, _, signed, _ = proxy.query(points)
    constraint = np.full(points.shape[0], FROZEN, dtype=np.int32)
    movable = signed > -active_band * particle_spacing
    is_surface = signed > -surface_band * particle_spacing
    constraint[movable] = FREE
    constraint[movable & is_surface] = SURFACE

    # feature edges: surface particles near a sharp segment -> FEATURE_EDGE
    if feature_segments.shape[0] > 0:
        surface_mask = constraint == SURFACE
        near_edge = _near_segments(points[surface_mask], feature_segments,
                                   feature_radius * particle_spacing)
        surface_indices = np.where(surface_mask)[0]
        constraint[surface_indices[near_edge]] = FEATURE_EDGE

    # corners: pin the single nearest particle to each true corner.
    # Off by default: snapping a particle exactly onto a corner vertex can clump it
    # with a neighbor < dx away (the corner vertex is not on the lattice). Sharp
    # features are kept by the 1D feature-edge constraint instead.
    pinned_targets = {}
    if pin_corners and feature_corners.shape[0] > 0:
        candidate_mask = np.isin(constraint, (SURFACE, FEATURE_EDGE))
        candidate_indices = np.where(candidate_mask)[0]
        if candidate_indices.size:
            candidate_tree = cKDTree(points[candidate_indices])
            for corner in feature_corners:
                distance, local = candidate_tree.query(corner)
                if distance < feature_radius * particle_spacing:
                    global_index = candidate_indices[local]
                    constraint[global_index] = PINNED
                    pinned_targets[int(global_index)] = corner

    if verbose:
        names = {FROZEN: "frozen", FREE: "free", SURFACE: "surface",
                 FEATURE_EDGE: "feature", PINNED: "pinned"}
        counts = {names[k]: int(np.sum(constraint == k)) for k in names}
        print(f"    classes: {counts}")

    movable_mask = constraint != FROZEN
    movable_index = np.where(movable_mask)[0]
    movable_class = constraint[movable_index]
    surface_local = movable_class == SURFACE
    free_local = movable_class == FREE
    pinned_index = np.array(sorted(pinned_targets), dtype=np.int64)
    pinned_position = (np.array([pinned_targets[int(i)] for i in pinned_index])
                       if pinned_index.size else np.zeros((0, dimension)))
    if pinned_index.size:
        points[pinned_index] = pinned_position

    # --- iterate ---------------------------------------------------------
    cutoff = cutoff_factor * particle_spacing
    step = step_fraction * particle_spacing
    maximum_step = 0.25 * particle_spacing
    for iteration in range(iterations):
        displacement = _repulsion_displacement(points, cutoff)
        magnitude = np.linalg.norm(displacement, axis=1, keepdims=True)
        scale = np.minimum(1.0, maximum_step / (step * magnitude + 1e-30))
        delta = step * displacement * scale
        delta[~movable_mask] = 0.0
        points = points + delta

        # constrained projection — query the surface proxy only for movable particles
        # (frozen ones don't move, so their classification/offset is constant).
        _, normal, signed, _ = proxy.query(points[movable_index])
        points[movable_index[surface_local]] -= (
            signed[surface_local, None] * normal[surface_local])                     # onto surface manifold
        strayed = free_local & (signed > 0.0)                                        # strayed outside -> back in
        points[movable_index[strayed]] -= (
            (signed[strayed] + 0.3 * particle_spacing)[:, None] * normal[strayed])
        if feature_segments.shape[0] > 0:
            edge_like = np.where(constraint == FEATURE_EDGE)[0]
            if edge_like.size:
                points[edge_like] = _project_to_segments(points[edge_like], feature_segments)
        if pinned_index.size:
            points[pinned_index] = pinned_position

        step = max(step * 0.985, 0.04 * particle_spacing)            # mild annealing
        if verbose and (iteration + 1) % 10 == 0:
            print(f"    iter {iteration+1}/{iterations}  max|delta|={float(np.max(np.linalg.norm(delta,axis=1)))/particle_spacing:.3f}dx")

    return points, constraint


def _repulsion_displacement(points, cutoff):
    """Sum of short-range repulsion (1 - r/cutoff)^2 along pair directions."""
    tree = cKDTree(points)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")
    displacement = np.zeros_like(points)
    if pairs.shape[0] == 0:
        return displacement
    delta = points[pairs[:, 0]] - points[pairs[:, 1]]
    distance = np.linalg.norm(delta, axis=1)
    valid = distance > 1e-12
    delta, distance, pairs = delta[valid], distance[valid], pairs[valid]
    weight = (1.0 - distance / cutoff) ** 2
    contribution = (weight / distance)[:, None] * delta
    np.add.at(displacement, pairs[:, 0], contribution)
    np.add.at(displacement, pairs[:, 1], -contribution)
    return displacement


def _near_segments(points, segments, radius):
    """Boolean mask: each point within ``radius`` of any segment."""
    if points.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    nearest = np.full(points.shape[0], np.inf)
    for start, end in segments:
        nearest = np.minimum(nearest, _point_segment_distance(points, start, end))
    return nearest <= radius


def _project_to_segments(points, segments):
    """Project each point onto the nearest of the feature segments (1D constraint)."""
    best = points.copy()
    best_distance = np.full(points.shape[0], np.inf)
    for start, end in segments:
        edge = end - start
        length_squared = float(edge @ edge)
        if length_squared < 1e-30:
            continue
        parameter = np.clip(((points - start) @ edge) / length_squared, 0.0, 1.0)
        projected = start + parameter[:, None] * edge
        distance = np.linalg.norm(points - projected, axis=1)
        closer = distance < best_distance
        best[closer] = projected[closer]
        best_distance[closer] = distance[closer]
    return best


def _point_segment_distance(points, start, end):
    edge = end - start
    length_squared = float(edge @ edge)
    if length_squared < 1e-30:
        return np.linalg.norm(points - start, axis=1)
    parameter = np.clip(((points - start) @ edge) / length_squared, 0.0, 1.0)
    closest = start + parameter[:, None] * edge
    return np.linalg.norm(points - closest, axis=1)
