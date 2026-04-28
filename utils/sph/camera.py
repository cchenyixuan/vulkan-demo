"""
camera.py — orbit camera with pyrr math, perspective + orthogonal switch.

Adapted from the user's reference camera. Conventions:
  - Right-handed world space, Y-up.
  - View matrix = look_at(position, position + front, up).
  - All matrices float32, OpenGL convention (Y-up in clip space).
  - Caller (renderer) applies Vulkan Y-flip via viewport (negative height),
    so we don't need to bake the flip into the projection matrix.

Public surface:
    cam = Camera(projection_type="orthogonal")
    cam.update_aspect(width, height)
    cam.rotate(dx, dy)            # mouse left drag
    cam.translate(dx, dy)         # mouse middle drag
    cam.zoom(dy)                  # mouse wheel
    cam.switch_projection()       # toggle perspective ↔ orthogonal
    vp = cam.view_projection()    # 4×4 float32, ready for push constant
"""

from typing import Tuple

import numpy as np
import pyrr


class Camera:
    """Orbit camera. Position rotates / pans / zooms relative to a lookat point."""

    def __init__(self, projection_type: str = "orthogonal"):
        self.position = pyrr.Vector3([0.25, 0.15, 0.5], dtype=np.float32)
        self.front    = pyrr.Vector3([0.0, 0.0, -1.0], dtype=np.float32)
        self.up       = pyrr.Vector3([0.0, 1.0, 0.0], dtype=np.float32)
        self.lookat   = pyrr.Vector3([0.25, 0.15, 0.0], dtype=np.float32)

        self.projection_type = projection_type
        self.fov = 45.0                                 # degrees
        self.near = 0.001
        self.far = 100.0
        self.aspect = 16.0 / 9.0

    # ------------------------------------------------------------------
    # Geometry queries
    # ------------------------------------------------------------------

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.position - self.lookat))

    def update_aspect(self, width: int, height: int) -> None:
        if height <= 0 or width <= 0:
            return
        self.aspect = float(width) / float(height)

    # ------------------------------------------------------------------
    # Matrix builders
    # ------------------------------------------------------------------

    def view_matrix(self) -> np.ndarray:
        target = self.position + self.front
        return pyrr.matrix44.create_look_at(
            self.position, target, self.up, dtype=np.float32)

    def projection_matrix(self) -> np.ndarray:
        if self.projection_type == "perspective":
            return pyrr.matrix44.create_perspective_projection_matrix(
                self.fov, self.aspect, self.near, self.far, dtype=np.float32)
        # orthogonal — half-extent = current camera distance, aspect-corrected.
        d = max(self.distance, 1e-4)
        return pyrr.matrix44.create_orthogonal_projection_matrix(
            -d, d,
            -d / self.aspect, d / self.aspect,
            -100.0, 100.0,
            dtype=np.float32)

    def view_projection(self) -> np.ndarray:
        # pyrr matrix multiplication: result = a @ b means apply `a` first.
        # In OpenGL convention, gl_Position = projection * view * world.
        # pyrr.matrix44.multiply(a, b) returns b @ a in mathematical sense, so
        # multiply(view, projection) gives projection * view as required.
        return pyrr.matrix44.multiply(self.view_matrix(), self.projection_matrix())

    # ------------------------------------------------------------------
    # Mouse / scroll inputs
    # ------------------------------------------------------------------

    def rotate(self, dx: float, dy: float) -> None:
        """Mouse-left drag: orbit around lookat point.

        dx rotates around world up; dy rotates around camera-right.
        """
        right = pyrr.vector3.normalize(np.cross(self.front, self.up))
        # Sensitivity 0.005 rad/pixel — feels natural at 1280×720.
        rotation = (
            pyrr.matrix44.create_from_axis_rotation(self.up, -dx * 0.005)
            @ pyrr.matrix44.create_from_axis_rotation(right, -dy * 0.005)
        )
        # Rotate position around lookat
        offset = np.array([*(self.position - self.lookat), 0.0], dtype=np.float32)
        rotated_offset = rotation @ offset
        self.position = pyrr.Vector3(
            self.lookat + rotated_offset[:3], dtype=np.float32)
        # Rotate basis vectors
        front_h = np.array([*self.front, 0.0], dtype=np.float32)
        up_h    = np.array([*self.up,    0.0], dtype=np.float32)
        self.front = pyrr.Vector3((rotation @ front_h)[:3], dtype=np.float32)
        self.up    = pyrr.Vector3((rotation @ up_h)[:3], dtype=np.float32)

    def translate(self, dx: float, dy: float) -> None:
        """Mouse-middle drag: pan in screen plane (move both position and lookat)."""
        right = pyrr.vector3.normalize(np.cross(self.front, self.up))
        # Pan speed scales with current zoom level so it feels uniform.
        scale = self.distance * 0.0015
        delta = (-dx * scale) * right + (dy * scale) * self.up
        self.position = pyrr.Vector3(self.position + delta, dtype=np.float32)
        self.lookat   = pyrr.Vector3(self.lookat   + delta, dtype=np.float32)

    def zoom(self, dy: float) -> None:
        """Scroll wheel: dolly along front direction.

        Positive dy = scroll up = zoom in (move closer to lookat).
        Stops at minimum distance to avoid clipping through the lookat point.
        """
        step_factor = dy * 0.1
        candidate = self.position + step_factor * self.distance * self.front
        new_dist = float(np.linalg.norm(candidate - self.lookat))
        if new_dist < 1e-3 and dy > 0:
            return                                       # don't dive into the lookat
        self.position = pyrr.Vector3(candidate, dtype=np.float32)

    def switch_projection(self) -> None:
        self.projection_type = (
            "orthogonal" if self.projection_type == "perspective" else "perspective"
        )

    # ------------------------------------------------------------------
    # Utility presets
    # ------------------------------------------------------------------

    def frame_bbox(self, bbox_min, bbox_max, margin: float = 1.2) -> None:
        """Re-center the camera so the given bbox is fully visible.

        Useful at startup so the user sees the whole frame without panning."""
        center = 0.5 * (np.asarray(bbox_min, dtype=np.float32)
                        + np.asarray(bbox_max, dtype=np.float32))
        diagonal = float(np.linalg.norm(
            np.asarray(bbox_max) - np.asarray(bbox_min)))
        self.lookat = pyrr.Vector3(center, dtype=np.float32)
        # Place camera in front (+Z) at distance proportional to bbox size.
        self.position = pyrr.Vector3(
            [center[0], center[1], diagonal * margin], dtype=np.float32)
        self.front = pyrr.Vector3([0.0, 0.0, -1.0], dtype=np.float32)
        self.up    = pyrr.Vector3([0.0, 1.0,  0.0], dtype=np.float32)
