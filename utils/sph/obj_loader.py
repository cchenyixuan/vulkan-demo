"""
Minimal vertex-only OBJ parser.

V0 scope: each particle obj (domain, boundary) and frame.obj is treated as a
point cloud — only `v x y z [w]` lines matter. Faces, normals, texcoords,
groups, and material directives are silently skipped. Future versions may
swap in a richer loader (mesh topology for irregular frame voxelization)
behind the same call site.
"""

import pathlib

import numpy as np


def load_obj_vertices(path: str | pathlib.Path) -> np.ndarray:
    """Parse OBJ file → ``(N, 3)`` float32 array of vertex positions.

    - Reads only ``v x y z [w]`` lines; the optional w-component is ignored.
    - Silently skips comments (``#``), blank lines, and every other OBJ
      directive (``f``, ``vn``, ``vt``, ``o``, ``g``, ``s``, ``usemtl``,
      ``mtllib``, ...).
    - Returns an empty ``(0, 3)`` array if no ``v`` lines are found.

    Raises:
        FileNotFoundError: path does not exist.
        ValueError:        malformed ``v`` line (wrong field count or
                           non-numeric coordinate).
    """
    path = pathlib.Path(path)
    vertices: list[tuple[float, float, float]] = []

    with path.open(encoding="utf-8") as handle:
        for line_index, raw_line in enumerate(handle, start=1):
            # Strip inline comments first ('#' can appear mid-line per OBJ spec).
            hash_position = raw_line.find("#")
            if hash_position != -1:
                raw_line = raw_line[:hash_position]
            stripped = raw_line.strip()
            if not stripped:
                continue

            tokens = stripped.split()
            if tokens[0] != "v":
                continue

            # Expect 3 coords (x y z) or 4 (x y z w); w ignored.
            if len(tokens) not in (4, 5):
                raise ValueError(
                    f"{path}:{line_index}: malformed `v` line, "
                    f"expected 3 or 4 numbers after `v`, got {len(tokens) - 1}: "
                    f"{stripped!r}")
            try:
                x = float(tokens[1])
                y = float(tokens[2])
                z = float(tokens[3])
            except ValueError as parse_error:
                raise ValueError(
                    f"{path}:{line_index}: non-numeric coordinate in `v` line: "
                    f"{stripped!r}") from parse_error
            vertices.append((x, y, z))

    if not vertices:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(vertices, dtype=np.float32)
