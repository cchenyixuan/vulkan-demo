"""
Voxel grid derivation: bbox + smoothing_length → grid origin / dimension dict.

Convention (must match shaders/sph/predict.comp's voxel coord computation):
    voxel_coord = floor((position - origin) / smoothing_length)

We anchor the grid so that ``bbox.min`` lies exactly at the **center** of voxel
(0, 0, 0) along each axis. Therefore:

    origin = bbox.min - h / 2

For dimension count we want the minimum voxel grid that fully contains
``[bbox.min, bbox.max]`` under that anchoring — i.e. for every point ``p`` in
the bbox, the integer voxel index must satisfy ``0 ≤ floor((p − origin)/h) ≤
dim − 1``. Solving the upper bound gives:

    dim = floor((bbox.max − bbox.min) / h + 0.5) + 1

The trailing ``+1`` is not a margin — it is required for correctness when
``span / h`` is an exact integer (``bbox.max`` lands at a voxel center, not the
upper boundary). Edge cases:

  - ``span = 0`` (degenerate axis, e.g. z in 2D) → ``dim = 1``.
  - ``span = k·h`` (exact multiple) → ``dim = k + 1``.
  - generic non-integer ``span/h``         → ``dim = ceil(span/h + 0.5)``.

The output dict is consumed by ``utils.sph.case.Case`` (stored on its
``grid`` field) and read by ``utils.sph.case.build_specialization_info``
when packing spec constants for shader pipelines.
"""

from typing import Sequence

import numpy as np


def compute_grid(
    bbox_min: Sequence[float] | np.ndarray,
    bbox_max: Sequence[float] | np.ndarray,
    smoothing_length: float,
    dimension: int,
) -> dict:
    """
    Args:
        bbox_min:          world-space lower corner of bbox (3 components).
        bbox_max:          world-space upper corner of bbox (3 components).
        smoothing_length:  voxel side length (== Wendland C4 support radius).
        dimension:         2 or 3. In 2D the z-axis must collapse to span 0.

    Returns:
        dict with keys
            'origin':    tuple[float, float, float] — corner of voxel (0,0,0)
            'dimension': tuple[int,   int,   int]   — voxel counts per axis

    Raises:
        ValueError on bad input (bad shape, h ≤ 0, dim ∉ {2,3}, max < min,
                   2D with nonzero z extent).
    """
    bbox_min_array = np.asarray(bbox_min, dtype=np.float64)
    bbox_max_array = np.asarray(bbox_max, dtype=np.float64)

    if bbox_min_array.shape != (3,) or bbox_max_array.shape != (3,):
        raise ValueError(
            f"bbox must have 3 components: got min shape {bbox_min_array.shape}, "
            f"max shape {bbox_max_array.shape}")
    if smoothing_length <= 0:
        raise ValueError(f"smoothing_length must be > 0, got {smoothing_length}")
    if dimension not in (2, 3):
        raise ValueError(f"dimension must be 2 or 3, got {dimension}")
    if np.any(bbox_max_array < bbox_min_array):
        raise ValueError(
            f"bbox_max < bbox_min on some axis: "
            f"min={bbox_min_array.tolist()} max={bbox_max_array.tolist()}")
    if dimension == 2 and bbox_max_array[2] != bbox_min_array[2]:
        raise ValueError(
            f"dimension=2 requires zero z extent, got "
            f"bbox z range [{bbox_min_array[2]}, {bbox_max_array[2]}]")

    # bbox.min sits at the center of voxel (0,0,0).
    origin_array = bbox_min_array - 0.5 * smoothing_length

    # Minimum voxel count s.t. floor((bbox.max - origin)/h) ≤ dim - 1.
    span_array = bbox_max_array - bbox_min_array
    dim_array = np.floor(span_array / smoothing_length + 0.5).astype(int) + 1

    return {
        'origin':    tuple(float(component) for component in origin_array),
        'dimension': tuple(int(count) for count in dim_array),
    }
