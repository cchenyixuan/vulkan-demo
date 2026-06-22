"""
region.py — SDF-backed geometric regions for particle sampling.

A ``Region`` is defined by a signed distance function: ``signed_distance(points)``
returns < 0 strictly inside, 0 on the surface, > 0 outside. Everything the
discretizer needs derives from it:

  - ``contains(points, inset)``  : membership for placement (inset by ½ spacing
                                   gives a consistent boundary offset, anti-staircase).
  - ``surface_normal(points)``   : ∇(signed_distance), for boundary-layer offset
                                   and the constrained-relaxation projection.
  - ``bounds()``                 : axis-aligned box to tile before masking.

Analytic primitives (Box, Sphere, HalfSpace, Cylinder) compose with exact CSG
booleans (Union = min, Intersection = max, Difference = max(A, -B)). Dirty meshes
get a robust SDF via generalized winding numbers in ``mesh_region.py`` — a
``MeshRegion`` is just another ``Region`` and composes with these the same way.

All primitives are dimension-generic where it makes sense (Box / Sphere /
HalfSpace work in 2D and 3D); Cylinder is 3D-only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Region(ABC):
    """Abstract SDF region. ``dimension`` is 2 or 3; points are ``(N, dimension)``."""

    dimension: int

    @abstractmethod
    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance, shape ``(N,)``; negative inside, positive outside."""

    @abstractmethod
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned ``(minimum, maximum)`` corners, each shape ``(dimension,)``."""

    def contains(self, points: np.ndarray, inset: float = 0.0) -> np.ndarray:
        """Boolean mask of points at least ``inset`` inside the surface."""
        return self.signed_distance(points) <= -inset

    def surface_normal(self, points: np.ndarray, finite_difference_step: float = 1.0e-5) -> np.ndarray:
        """Outward unit normal = ∇(signed_distance), by central differences.

        Works for ANY region (primitive, CSG composite, or mesh) without an
        analytic gradient. Primitives may override with a closed form.
        """
        points = np.asarray(points, dtype=np.float64)
        gradient = np.empty_like(points)
        for axis in range(self.dimension):
            step = np.zeros(self.dimension)
            step[axis] = finite_difference_step
            forward = self.signed_distance(points + step)
            backward = self.signed_distance(points - step)
            gradient[:, axis] = (forward - backward) / (2.0 * finite_difference_step)
        norm = np.linalg.norm(gradient, axis=1, keepdims=True)
        norm = np.where(norm < 1.0e-30, 1.0, norm)
        return gradient / norm


# ---------------------------------------------------------------------------
# Analytic primitives
# ---------------------------------------------------------------------------

class Box(Region):
    """Axis-aligned box ``[minimum_corner, maximum_corner]`` (exact SDF)."""

    def __init__(self, minimum_corner, maximum_corner):
        self.minimum_corner = np.asarray(minimum_corner, dtype=np.float64)
        self.maximum_corner = np.asarray(maximum_corner, dtype=np.float64)
        if self.minimum_corner.shape != self.maximum_corner.shape:
            raise ValueError("box corners must have the same shape")
        if np.any(self.maximum_corner <= self.minimum_corner):
            raise ValueError("box maximum_corner must exceed minimum_corner on every axis")
        self.dimension = int(self.minimum_corner.shape[0])
        self._center = 0.5 * (self.minimum_corner + self.maximum_corner)
        self._half_extent = 0.5 * (self.maximum_corner - self.minimum_corner)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        offset = np.abs(points - self._center) - self._half_extent
        outside = np.linalg.norm(np.maximum(offset, 0.0), axis=1)
        inside = np.minimum(np.max(offset, axis=1), 0.0)
        return outside + inside

    def bounds(self):
        return self.minimum_corner.copy(), self.maximum_corner.copy()


class Sphere(Region):
    """Sphere (3D) / disk (2D): ``|point - center| - radius`` (exact SDF)."""

    def __init__(self, center, radius):
        self.center = np.asarray(center, dtype=np.float64)
        self.radius = float(radius)
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        self.dimension = int(self.center.shape[0])

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return np.linalg.norm(points - self.center, axis=1) - self.radius

    def bounds(self):
        return self.center - self.radius, self.center + self.radius


class HalfSpace(Region):
    """Half-space inside ``outward_normal · (point - anchor) <= 0``.

    Useful as a CSG cutting plane. Unbounded — ``bounds()`` raises; only use it
    inside an Intersection/Difference with a bounded region.
    """

    def __init__(self, anchor, outward_normal):
        self.anchor = np.asarray(anchor, dtype=np.float64)
        normal = np.asarray(outward_normal, dtype=np.float64)
        self.outward_normal = normal / np.linalg.norm(normal)
        self.dimension = int(self.anchor.shape[0])

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return (points - self.anchor) @ self.outward_normal

    def bounds(self):
        raise NotImplementedError("HalfSpace is unbounded; intersect it with a bounded region")


class Cylinder(Region):
    """Finite axis-aligned cylinder (3D only), exact SDF."""

    def __init__(self, base_center, axis_vector, radius):
        self.base_center = np.asarray(base_center, dtype=np.float64)
        self.axis_vector = np.asarray(axis_vector, dtype=np.float64)
        self.radius = float(radius)
        self.dimension = 3
        self._height = float(np.linalg.norm(self.axis_vector))
        if self._height <= 0 or self.radius <= 0:
            raise ValueError("cylinder needs positive height and radius")
        self._axis_unit = self.axis_vector / self._height

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        relative = points - self.base_center
        along = relative @ self._axis_unit
        radial = np.linalg.norm(relative - np.outer(along, self._axis_unit), axis=1)
        distance_radial = radial - self.radius
        distance_axial = np.abs(along - 0.5 * self._height) - 0.5 * self._height
        outside = np.linalg.norm(
            np.stack([np.maximum(distance_radial, 0.0), np.maximum(distance_axial, 0.0)], axis=1), axis=1)
        inside = np.minimum(np.maximum(distance_radial, distance_axial), 0.0)
        return outside + inside

    def bounds(self):
        # Conservative axis-aligned box around the (possibly tilted) cylinder.
        end_center = self.base_center + self.axis_vector
        lo = np.minimum(self.base_center, end_center) - self.radius
        hi = np.maximum(self.base_center, end_center) + self.radius
        return lo, hi


# ---------------------------------------------------------------------------
# CSG boolean composition
# ---------------------------------------------------------------------------

class Union(Region):
    """Union of solids: inside if inside ANY child.  SDF = min(children)."""

    def __init__(self, *children: Region):
        if not children:
            raise ValueError("Union needs at least one child")
        self.children = children
        self.dimension = children[0].dimension

    def signed_distance(self, points):
        return np.min(np.stack([c.signed_distance(points) for c in self.children], axis=0), axis=0)

    def bounds(self):
        corners = [c.bounds() for c in self.children]
        lo = np.min(np.stack([c[0] for c in corners], axis=0), axis=0)
        hi = np.max(np.stack([c[1] for c in corners], axis=0), axis=0)
        return lo, hi


class Intersection(Region):
    """Intersection: inside if inside ALL children.  SDF = max(children)."""

    def __init__(self, *children: Region):
        if not children:
            raise ValueError("Intersection needs at least one child")
        self.children = children
        self.dimension = children[0].dimension

    def signed_distance(self, points):
        return np.max(np.stack([c.signed_distance(points) for c in self.children], axis=0), axis=0)

    def bounds(self):
        # Bounded by any bounded child; intersect the boxes that are available.
        boxes = []
        for child in self.children:
            try:
                boxes.append(child.bounds())
            except NotImplementedError:
                continue
        if not boxes:
            raise NotImplementedError("Intersection has no bounded child")
        lo = np.max(np.stack([b[0] for b in boxes], axis=0), axis=0)
        hi = np.min(np.stack([b[1] for b in boxes], axis=0), axis=0)
        return lo, hi


class Difference(Region):
    """``base`` minus ``subtracted``.  SDF = max(base, -subtracted)."""

    def __init__(self, base: Region, subtracted: Region):
        self.base = base
        self.subtracted = subtracted
        self.dimension = base.dimension

    def signed_distance(self, points):
        return np.maximum(self.base.signed_distance(points), -self.subtracted.signed_distance(points))

    def bounds(self):
        return self.base.bounds()       # subtracting can only shrink the region
