"""
cavity_fields.py - smooth velocity and vorticity fields of the Re = 1000 cavity validation runs, with the boundary
band drawn at its true thickness and per-particle zooms of the four cavity corners.

Adapted from docs/cavity_validation/scripts/plot_cavity_fields.py (2026-10-02, the field-figure style approved for
the cavity runs); the computation and the rendering are that script's. Changes:
  * input: a run directory of experiment/validation/cavity_runner.py (logs/validation/cavity_re1000/<run>/), the
    newest checkpoint (full particle state: position, velocity, mass, density, material; float32) and meta.json;
  * reference: the primary vortex of Marchi, Santiago & Carvalho Jr. (2021) Table 19 (Tc, from
    docs/validation/data/marchi2021_re1000.csv, checked against the PDF). Table 19 gives the primary vortex only
    (psi_min and its position, no vorticity); the secondary eddies are searched and reported without a reference
    marker. No values transcribed from memory are used;
  * the scatter is validated against experiment/validation/cavity_sampling.interpolate (Shepard);
  * the titles name the numerics setting of the run (KCG regularisation xi, eps^2 of the viscous term) and quote the
    time-averaged errors of the validation analysis (docs/validation/data/profile_<run>.csv);
  * --difference takes any pair of runs (e.g. two numerics settings), repeatable.

Method (unchanged):
  1. interpolates ALL particles (fluid + static wall + moving lid, with their stored velocities, which carry the
     no-slip condition and the lid speed) onto a uniform grid by Shepard-normalised SPH interpolation with the 2-D
     Wendland C4 kernel of support h and particle volume m/rho (vectorised particle-to-grid scatter);
  2. classifies every grid cell by its nearest particle: cells whose nearest particle is a wall or lid particle are
     drawn in the boundary-class colour, so the band is visible at its true thickness;
  3. derives the speed |u|/U, the vorticity omega_z = dv/dx - du/dy (central differences) and two stream functions:
     "velocity-integral" psi = integral of u dy from the bottom wall, and "poisson" laplacian(psi) = -omega_z with
     psi = 0 on the effective walls (half-way between the outermost fluid row and the first boundary row), the
     stream function of the solenoidal part of the interpolated field (drawn by default; both are reported);
  4. locates the primary and secondary vortex centres (psi extrema, local search + quadratic refinement);
  5. renders <run>_velocity.png and <run>_vorticity.png (main panel + four corner zooms that show every particle;
     each zoom has its own colour scale) and the one-pixel-per-cell maps <run>_velocity_full.png /
     <run>_vorticity_full.png;
  6. writes <run>_check.json and regenerates fields_check.md from every *_check.json in the output directory;
  7. with --difference RUN_A RUN_B: renders RUN_A_minus_RUN_B.png (|delta u| and delta omega_z of the two smooth
     grid fields, the slab cut and Y-averaged difference profiles) and its numbers.

Per-particle vorticity in the zooms is the first-order renormalised SPH gradient
    grad f_i = B_i^-T sum_j V_j (f_j - f_i) grad_i W_ij,   B_i = sum_j V_j (x_j - x_i) (x) grad_i W_ij.
The plain sum (without B_i^-1) is 13 % too large: the solver calibrates the particle volume with the self term
excluded (V = 1 / sum_{j != i} W = 1.1293 dx^2), so sum_j V_j (x_j - x_i) (x) grad W = 1.129 I.

Coordinates in all figures are cavity coordinates X = x/L + 1/2, Y = y/L + 1/2 (lid on top, moving +X); velocity in
units of U, psi in U L, vorticity in U/L (L = 1 m, U = 1 m/s in these cases).

CPU only. Usage (from the repository root):
    .venv/Scripts/python.exe -m experiment.validation.cavity_fields \
        --jobs n1000_k2_float32 n1000_k2_float32_xi0p001 \
        --difference n1000_k2_float32_xi0p001 n1000_k2_float32 [--cache-directory <scratch>]
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import math
import pathlib
import sys
import time

import numpy as np
import yaml
from scipy.fft import dstn, idstn
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import binary_erosion, gaussian_filter, map_coordinates
from scipy.spatial import cKDTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize, SymLogNorm, to_rgba  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.ticker import FormatStrFormatter  # noqa: E402

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from experiment.validation import cavity_reference, cavity_sampling  # noqa: E402
from experiment.validation.cavity_analysis import run_numerics  # noqa: E402

DEFAULT_CAMPAIGN_DIRECTORY = REPOSITORY_ROOT / "logs" / "validation" / "cavity_re1000"
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "docs" / "validation" / "fields"
DATA_DIRECTORY = REPOSITORY_ROOT / "docs" / "validation" / "data"

# Unit square of the figures (--frame): "wall" = the square of the innermost boundary rows (the bottom-left innermost
# boundary particle is (0, 0), the top-right one (1, 1)); load_job scales every position, h and dx by 1 / (1 + 2 dx)
# so that those rows sit at |x| = 0.5, the same frame as the validation analysis (cavity_sampling.Frame "wall").
# "fluid" = the fluid lattice box, positions unscaled. Velocities are never scaled (U = 1).
FRAME = "wall"
FLUID_HALF_WIDTH = 0.5            # fluid box [-0.5, 0.5]^2 m; L = 1 m and U = 1 m/s, so SI values are already
                                  # in units of L and U

MATERIAL_FLUID = 0
MATERIAL_WALL = 1
MATERIAL_LID = 2
PIXEL_CLASS_OUTSIDE = 3           # grid cell beyond the outermost boundary row (zoom windows only)
MATERIAL_NAMES = {MATERIAL_FLUID: "fluid", MATERIAL_WALL: "wall", MATERIAL_LID: "lid"}
EXPECTED_MATERIAL_COUNTS_2M = {MATERIAL_FLUID: 2_002_225, MATERIAL_WALL: 47_179, MATERIAL_LID: 15_565}

WALL_COLOR = "#8c8c8c"
LID_COLOR = "#1a1a1a"
OUTSIDE_COLOR = "#ffffff"
INTERFACE_COLOR = "#ffffff"       # fluid / boundary interface line (effective wall), both figures
ANNOTATION_COLOR = "#e6007e"      # zoom rectangles, zoom labels, nominal cavity edge
TEXT_COLOR = "#222222"
MUTED_TEXT_COLOR = "#555555"

VORTICITY_CLIP = 5.0
VORTICITY_CONTOUR_SMOOTHING_CELLS = 3.0    # Gaussian sigma (grid cells) of the copy that the omega_z contours trace
STREAM_FUNCTION_WALL_MASK_SPACINGS = 1.0   # psi is not contoured within this many dx of the effective wall

# Reference vortex centre: Marchi, Santiago & Carvalho Jr. (2021), Table 19 (p. 041004-10), Tc, from
# docs/validation/data/marchi2021_re1000.csv (checked against the PDF by verify_reference.py). Table 19 gives the
# primary vortex only (psi_min and its position); the secondary eddies are searched without a reference.
_MARCHI_EXTREMA = cavity_reference.marchi2021()["extrema"]
REFERENCE_VORTEX_TABLE = {
    1000: {"primary": {"source": "Marchi et al. 2021, Table 19",
                       "position": (_MARCHI_EXTREMA["x_at_psi_min"][0], _MARCHI_EXTREMA["y_at_psi_min"][0]),
                       "stream_function": _MARCHI_EXTREMA["psi_min"][0]}},
}
SEARCHED_VORTICES = ("primary", "BR1", "BL1")
# Contour levels: the psi and omega level sets of Ghia, Ghia & Shin (1982) (levels only, not used as reference values).
GHIA_STREAM_FUNCTION_LEVELS = [-0.1175, -0.115, -0.11, -0.1, -0.09, -0.07, -0.05, -0.03, -0.01, -1e-4, -1e-5,
                               -1e-7, -1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 1.5e-3, 3e-3]
GHIA_VORTICITY_LEVELS_GHIA_SIGN = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]

# Search windows (cavity coordinates X range, Y range) for the vortex centres and the marker shape per vortex.
VORTEX_SEARCH = {
    "primary": {"window": ((0.2, 0.8), (0.2, 0.8)), "extremum": "minimum", "marker": "o", "label": "primary"},
    "BR1": {"window": ((0.6, 1.0), (0.0, 0.45)), "extremum": "maximum", "marker": "s", "label": "bottom-right BR1"},
    "BL1": {"window": ((0.0, 0.35), (0.0, 0.35)), "extremum": "maximum", "marker": "^", "label": "bottom-left BL1"},
    "TL1": {"window": ((0.0, 0.35), (0.6, 1.0)), "extremum": "maximum", "marker": "D", "label": "top-left TL1"},
}

# Corner zoom windows: about 0.009 L of boundary band + 0.026 L of fluid, in cavity coordinates.
ZOOM_BOUNDARY_SPAN = 0.009
ZOOM_FLUID_SPAN = 0.026
ZOOM_CORNERS = {                   # name: (corner X, corner Y, long name)
    "TL": (0.0, 1.0, "top-left corner"),
    "BL": (0.0, 0.0, "bottom-left corner"),
    "TR": (1.0, 1.0, "top-right corner"),
    "BR": (1.0, 0.0, "bottom-right corner"),
}
ZOOM_GRID_CELLS_PER_PARTICLE_SPACING = 6   # local background grid of the zooms: dx / 6
ZOOM_BACKGROUND_LIGHTENING = 0.4           # zoom background field blended 40 % towards white so particles stand out


# ----------------------------------------------------------------------------------------------------------
# process helpers (Windows: lower our own priority so the running GPU campaign's host threads are not disturbed)
# ----------------------------------------------------------------------------------------------------------

def lower_own_process_priority() -> None:
    if sys.platform != "win32":
        return
    try:
        below_normal_priority_class = 0x00004000
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), below_normal_priority_class)
    except Exception:  # pragma: no cover - best effort only
        pass


def peak_memory_bytes() -> int:
    """Peak resident set (Windows: PeakWorkingSetSize) of this process."""
    if sys.platform == "win32":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCounters),
                                                     wintypes.DWORD]
        kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return int(counters.PeakWorkingSetSize)
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


# ----------------------------------------------------------------------------------------------------------
# kernel and interpolation
# ----------------------------------------------------------------------------------------------------------

def wendland_c4_kernel(distance: np.ndarray, smoothing_length: float) -> np.ndarray:
    """2-D Wendland C4, support smoothing_length; same expression as wendland_c4_2d of the validation runner.
    Callers pass distances already < smoothing_length."""
    normalized_distance = distance / smoothing_length
    return (9.0 / (math.pi * smoothing_length * smoothing_length)) * (1.0 - normalized_distance) ** 6 * (
        1.0 + 6.0 * normalized_distance + 35.0 / 3.0 * normalized_distance ** 2)


def wendland_c4_kernel_radial_derivative(distance: np.ndarray, smoothing_length: float) -> np.ndarray:
    """dW/dr of the 2-D Wendland C4: -(9/(pi h^2)) (56/3) q (1 + 5q) (1 - q)^5 / h, q = r/h < 1."""
    normalized_distance = distance / smoothing_length
    return (-(9.0 / (math.pi * smoothing_length * smoothing_length)) * (56.0 / 3.0) * normalized_distance
            * (1.0 + 5.0 * normalized_distance) * (1.0 - normalized_distance) ** 5 / smoothing_length)


class UniformGrid:
    """Cell-centred uniform grid. Arrays on it are indexed [row (y), column (x)]."""

    def __init__(self, x_minimum: float, y_minimum: float, spacing: float, column_count: int, row_count: int):
        self.x_minimum = x_minimum          # left edge of the first cell
        self.y_minimum = y_minimum          # bottom edge of the first cell
        self.spacing = spacing
        self.column_count = column_count
        self.row_count = row_count

    @property
    def x_centres(self) -> np.ndarray:
        return self.x_minimum + (np.arange(self.column_count) + 0.5) * self.spacing

    @property
    def y_centres(self) -> np.ndarray:
        return self.y_minimum + (np.arange(self.row_count) + 0.5) * self.spacing

    @property
    def extent_cavity(self) -> list[float]:
        """imshow extent in cavity coordinates (X = x + 0.5)."""
        return [self.x_minimum + 0.5, self.x_minimum + self.column_count * self.spacing + 0.5,
                self.y_minimum + 0.5, self.y_minimum + self.row_count * self.spacing + 0.5]


def scatter_particles_to_grid(particle_position: np.ndarray, particle_volume: np.ndarray,
                              particle_fields: np.ndarray, grid: UniformGrid, smoothing_length: float,
                              pair_budget: int = 4_000_000):
    """Shepard-normalised SPH interpolation onto the grid cell centres, as a particle-to-grid scatter.

    value(g) = sum_j V_j W(|x_g - x_j|) f_j / sum_j V_j W(|x_g - x_j|).
    Every particle sits in a base cell; one pass per integer cell offset (column_offset, row_offset) whose
    cell can lie inside the support evaluates the weights of all particles for that offset at once.
    Offsets are batched so that one np.bincount call handles about pair_budget particle-cell pairs.
    Returns (interpolated fields [row, column, field] with NaN where no particle reaches, weight sum [row, column]).
    """
    particle_count = particle_position.shape[0]
    field_count = particle_fields.shape[1]
    cell_total = grid.row_count * grid.column_count
    base_column = np.floor((particle_position[:, 0] - grid.x_minimum) / grid.spacing).astype(np.int64)
    base_row = np.floor((particle_position[:, 1] - grid.y_minimum) / grid.spacing).astype(np.int64)

    # A particle lies anywhere in its base cell, so the grid point of offset k is between (|k| - 1/2) and
    # (|k| + 1/2) cells away along each axis: keep offsets whose closest possible distance is inside the support.
    offset_reach = int(math.ceil(smoothing_length / grid.spacing + 0.5))
    offsets = []
    for row_offset in range(-offset_reach, offset_reach + 1):
        for column_offset in range(-offset_reach, offset_reach + 1):
            closest_column = max(0.0, abs(column_offset) - 0.5) * grid.spacing
            closest_row = max(0.0, abs(row_offset) - 0.5) * grid.spacing
            if closest_column * closest_column + closest_row * closest_row < smoothing_length * smoothing_length:
                offsets.append((column_offset, row_offset))

    weight_sum = np.zeros(cell_total)
    weighted_field_sums = np.zeros((field_count, cell_total))
    offsets_per_batch = max(1, pair_budget // max(1, particle_count))
    smoothing_length_squared = smoothing_length * smoothing_length
    for batch_start in range(0, len(offsets), offsets_per_batch):
        batch_cell_indices = []
        batch_weights = []
        batch_particle_indices = []
        for column_offset, row_offset in offsets[batch_start:batch_start + offsets_per_batch]:
            target_column = base_column + column_offset
            target_row = base_row + row_offset
            separation_x = grid.x_minimum + (target_column + 0.5) * grid.spacing - particle_position[:, 0]
            separation_y = grid.y_minimum + (target_row + 0.5) * grid.spacing - particle_position[:, 1]
            distance_squared = separation_x * separation_x + separation_y * separation_y
            inside = ((distance_squared < smoothing_length_squared)
                      & (target_column >= 0) & (target_column < grid.column_count)
                      & (target_row >= 0) & (target_row < grid.row_count))
            selected = np.flatnonzero(inside)
            if selected.size == 0:
                continue
            weight = wendland_c4_kernel(np.sqrt(distance_squared[selected]), smoothing_length) \
                * particle_volume[selected]
            batch_cell_indices.append(target_row[selected] * grid.column_count + target_column[selected])
            batch_weights.append(weight)
            batch_particle_indices.append(selected)
        if not batch_cell_indices:
            continue
        cell_indices = np.concatenate(batch_cell_indices)
        weights = np.concatenate(batch_weights)
        particle_indices = np.concatenate(batch_particle_indices)
        weight_sum += np.bincount(cell_indices, weights=weights, minlength=cell_total)
        for field_index in range(field_count):
            weighted_field_sums[field_index] += np.bincount(
                cell_indices, weights=weights * particle_fields[particle_indices, field_index], minlength=cell_total)
        del cell_indices, weights, particle_indices

    with np.errstate(invalid="ignore", divide="ignore"):
        interpolated = np.where(weight_sum > 0, weighted_field_sums / weight_sum, np.nan)
    interpolated = interpolated.T.reshape(grid.row_count, grid.column_count, field_count)
    return interpolated, weight_sum.reshape(grid.row_count, grid.column_count)


def classify_grid_cells(particle_tree: cKDTree, particle_material: np.ndarray, grid: UniformGrid,
                        boundary_half_extent: float, rows_per_chunk: int = 256) -> np.ndarray:
    """Material of the nearest particle (Euclidean) for every cell centre; PIXEL_CLASS_OUTSIDE beyond the
    outermost boundary row + dx/2 (the band is a square frame: max(|x|, |y|) > boundary_half_extent)."""
    pixel_class = np.empty((grid.row_count, grid.column_count), dtype=np.int8)
    x_centres = grid.x_centres
    y_centres = grid.y_centres
    for row_start in range(0, grid.row_count, rows_per_chunk):
        row_stop = min(grid.row_count, row_start + rows_per_chunk)
        chunk_x, chunk_y = np.meshgrid(x_centres, y_centres[row_start:row_stop])
        _, nearest = particle_tree.query(np.column_stack([chunk_x.ravel(), chunk_y.ravel()]), k=1, workers=4)
        chunk_class = particle_material[nearest].astype(np.int8)
        outside = np.maximum(np.abs(chunk_x.ravel()), np.abs(chunk_y.ravel())) > boundary_half_extent
        chunk_class[outside] = PIXEL_CLASS_OUTSIDE
        pixel_class[row_start:row_stop] = chunk_class.reshape(row_stop - row_start, grid.column_count)
    return pixel_class


def particle_vorticity(particle_tree: cKDTree, particle_position: np.ndarray, particle_velocity: np.ndarray,
                       particle_volume: np.ndarray, target_indices: np.ndarray, smoothing_length: float):
    """Per-particle SPH vorticity omega_i = dv/dx_i - du/dy_i from the first-order renormalised gradient

        grad f_i = B_i^-T g_i,  g_i = sum_j V_j (f_j - f_i) grad_i W_ij,  B_i = sum_j V_j (x_j - x_i) (x) grad_i W_ij,

    grad_i W_ij = dW/dr (x_i - x_j)/r, all particles (fluid, wall, lid) as neighbours. The renormalisation makes
    the estimate exact for linear velocity fields whatever the particle volume convention (here V = m/rho =
    1.133 dx^2, so the plain sum omega_i = sum_j V_j [(v_j - v_i) dW/dx_i - (u_j - u_i) dW/dy_i] reads 13 % high).
    Returns (renormalised omega, plain-sum omega)."""
    neighbor_lists = particle_tree.query_ball_point(particle_position[target_indices], smoothing_length, workers=4)
    neighbor_counts = np.array([len(neighbors) for neighbors in neighbor_lists])
    owner_slot = np.repeat(np.arange(target_indices.size), neighbor_counts)
    neighbor_index = np.concatenate([np.asarray(neighbors, dtype=np.int64) for neighbors in neighbor_lists])
    owner_index = target_indices[owner_slot]
    separation = particle_position[owner_index] - particle_position[neighbor_index]      # x_i - x_j
    distance = np.hypot(separation[:, 0], separation[:, 1])
    keep = (distance > 0.0) & (distance < smoothing_length)
    owner_slot, owner_index, neighbor_index = owner_slot[keep], owner_index[keep], neighbor_index[keep]
    separation, distance = separation[keep], distance[keep]
    gradient_scale = wendland_c4_kernel_radial_derivative(distance, smoothing_length) / distance
    weighted_gradient_x = gradient_scale * separation[:, 0] * particle_volume[neighbor_index]   # V_j dW/dx_i
    weighted_gradient_y = gradient_scale * separation[:, 1] * particle_volume[neighbor_index]   # V_j dW/dy_i
    velocity_difference = particle_velocity[neighbor_index] - particle_velocity[owner_index]

    def total(weights):
        return np.bincount(owner_slot, weights=weights, minlength=target_indices.size)

    # B_i[a, b] = sum_j V_j (x_j - x_i)_a (grad_i W_ij)_b, with x_j - x_i = -separation.
    moment = np.empty((target_indices.size, 2, 2))
    moment[:, 0, 0] = total(-separation[:, 0] * weighted_gradient_x)
    moment[:, 0, 1] = total(-separation[:, 0] * weighted_gradient_y)
    moment[:, 1, 0] = total(-separation[:, 1] * weighted_gradient_x)
    moment[:, 1, 1] = total(-separation[:, 1] * weighted_gradient_y)
    raw_gradient_u = np.column_stack([total(velocity_difference[:, 0] * weighted_gradient_x),
                                      total(velocity_difference[:, 0] * weighted_gradient_y)])
    raw_gradient_v = np.column_stack([total(velocity_difference[:, 1] * weighted_gradient_x),
                                      total(velocity_difference[:, 1] * weighted_gradient_y)])
    # For f linear, g_b = sum_a (grad f)_a B[a, b], i.e. g = B^T grad f.
    moment_transposed = moment.transpose(0, 2, 1)
    gradient_u = np.linalg.solve(moment_transposed, raw_gradient_u[:, :, None])[:, :, 0]
    gradient_v = np.linalg.solve(moment_transposed, raw_gradient_v[:, :, None])[:, :, 0]
    renormalised = gradient_v[:, 0] - gradient_u[:, 1]
    plain_sum = raw_gradient_v[:, 0] - raw_gradient_u[:, 1]
    return renormalised, plain_sum


# ----------------------------------------------------------------------------------------------------------
# derived fields and vortex centres
# ----------------------------------------------------------------------------------------------------------

def stream_function_from_vorticity(vorticity: np.ndarray, grid: UniformGrid, wall_half_extent: float) -> np.ndarray:
    """Solve laplacian(psi) = -omega_z in the square |x|, |y| < wall_half_extent with psi = 0 on its edge.

    The square is discretised with a node grid aligned to the effective walls (node spacing ~ grid spacing,
    5-point Laplacian, fast sine transform DST-I); omega_z is resampled onto the nodes bilinearly and psi is
    resampled back onto the grid (psi = 0 outside the square)."""
    interval_count = int(round(2.0 * wall_half_extent / grid.spacing))
    node_spacing = 2.0 * wall_half_extent / interval_count
    interior_nodes = -wall_half_extent + np.arange(1, interval_count) * node_spacing
    node_columns, node_rows = np.meshgrid((interior_nodes - grid.x_centres[0]) / grid.spacing,
                                          (interior_nodes - grid.y_centres[0]) / grid.spacing)
    vorticity_on_nodes = map_coordinates(vorticity, [node_rows, node_columns], order=1, mode="nearest")
    del node_columns, node_rows
    wave_numbers = np.arange(1, interval_count)
    eigenvalues = (2.0 * np.cos(np.pi * wave_numbers / interval_count) - 2.0) / node_spacing ** 2
    stream_function_on_nodes = idstn(dstn(-vorticity_on_nodes, type=1)
                                     / (eigenvalues[:, None] + eigenvalues[None, :]), type=1)
    padded = np.zeros((interval_count + 1, interval_count + 1))
    padded[1:-1, 1:-1] = stream_function_on_nodes
    del stream_function_on_nodes, vorticity_on_nodes
    grid_columns, grid_rows = np.meshgrid((grid.x_centres + wall_half_extent) / node_spacing,
                                          (grid.y_centres + wall_half_extent) / node_spacing)
    return map_coordinates(padded, [grid_rows, grid_columns], order=1, mode="constant", cval=0.0)

def sample_on_effective_walls(field: np.ndarray, grid: UniformGrid, wall_half_extent: float,
                              corner_margin: float, sample_count: int = 2001) -> dict:
    """Bilinear samples of a grid field along the four effective walls (corners within corner_margin left out)."""
    along_wall = np.linspace(-wall_half_extent + corner_margin, wall_half_extent - corner_margin, sample_count)
    on_wall = np.full_like(along_wall, wall_half_extent)

    def sample(x_coordinates, y_coordinates):
        return map_coordinates(field, [(y_coordinates - grid.y_centres[0]) / grid.spacing,
                                       (x_coordinates - grid.x_centres[0]) / grid.spacing], order=1)

    return {"left": sample(-on_wall, along_wall), "right": sample(on_wall, along_wall),
            "top": sample(along_wall, on_wall), "bottom": sample(along_wall, -on_wall)}


def refine_extremum(field: np.ndarray, row: int, column: int, half_size: int = 2):
    """Least-squares quadratic fit on a (2 half_size + 1)^2 patch; returns (row shift, column shift, value)
    in cell units, or the grid point itself when the fit is unusable."""
    patch = field[row - half_size:row + half_size + 1, column - half_size:column + half_size + 1]
    if patch.shape != (2 * half_size + 1, 2 * half_size + 1) or not np.all(np.isfinite(patch)):
        return 0.0, 0.0, float(field[row, column])
    row_offsets, column_offsets = np.mgrid[-half_size:half_size + 1, -half_size:half_size + 1]
    column_offsets = column_offsets.ravel().astype(float)
    row_offsets = row_offsets.ravel().astype(float)
    design = np.column_stack([np.ones_like(column_offsets), column_offsets, row_offsets, column_offsets ** 2,
                              column_offsets * row_offsets, row_offsets ** 2])
    coefficients, *_ = np.linalg.lstsq(design, patch.ravel(), rcond=None)
    hessian = np.array([[2.0 * coefficients[3], coefficients[4]], [coefficients[4], 2.0 * coefficients[5]]])
    gradient = np.array([coefficients[1], coefficients[2]])
    try:
        column_shift, row_shift = -np.linalg.solve(hessian, gradient)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, float(field[row, column])
    if abs(column_shift) > 1.0 or abs(row_shift) > 1.0:
        return 0.0, 0.0, float(field[row, column])
    value = float(coefficients @ np.array([1.0, column_shift, row_shift, column_shift ** 2,
                                           column_shift * row_shift, row_shift ** 2]))
    return float(row_shift), float(column_shift), value


def find_vortex_centres(stream_function: np.ndarray, vorticity: np.ndarray, valid_mask: np.ndarray,
                        grid: UniformGrid, reynolds_number: int) -> dict:
    """psi extremum inside each search window (primary = minimum, secondary eddies = maximum). A centre is
    accepted only when it is a strict interior extremum (not on the edge of the window nor on the edge of the
    valid mask, and not beaten by any of its 8 neighbours) with the right sign."""
    reference = REFERENCE_VORTEX_TABLE.get(reynolds_number, {})
    cavity_x = grid.x_centres + 0.5
    cavity_y = grid.y_centres + 0.5
    results = {}
    for vortex_name in SEARCHED_VORTICES:
        search = VORTEX_SEARCH[vortex_name]
        (x_low, x_high), (y_low, y_high) = search["window"]
        column_slice = np.flatnonzero((cavity_x >= x_low) & (cavity_x <= x_high))
        row_slice = np.flatnonzero((cavity_y >= y_low) & (cavity_y <= y_high))
        window_field = stream_function[row_slice[0]:row_slice[-1] + 1, column_slice[0]:column_slice[-1] + 1]
        window_mask = valid_mask[row_slice[0]:row_slice[-1] + 1, column_slice[0]:column_slice[-1] + 1]
        sign = -1.0 if search["extremum"] == "minimum" else 1.0
        scored = np.where(window_mask, sign * window_field, -np.inf)
        flat_best = int(np.argmax(scored))
        window_row, window_column = np.unravel_index(flat_best, scored.shape)
        mask_interior = binary_erosion(window_mask, structure=np.ones((3, 3), dtype=bool))
        on_edge = ((window_row in (0, scored.shape[0] - 1)) or (window_column in (0, scored.shape[1] - 1))
                   or not bool(mask_interior[window_row, window_column]))
        if not on_edge:
            neighbourhood = scored[window_row - 1:window_row + 2, window_column - 1:window_column + 2]
            on_edge = bool(np.count_nonzero(neighbourhood > scored[window_row, window_column]))
        row = row_slice[0] + window_row
        column = column_slice[0] + window_column
        row_shift, column_shift, value = refine_extremum(stream_function, row, column)
        position = (float(cavity_x[column] + column_shift * grid.spacing),
                    float(cavity_y[row] + row_shift * grid.spacing))
        found = bool(np.isfinite(scored[window_row, window_column]) and sign * value > 0 and not on_edge)
        vorticity_at_centre = float(map_coordinates(vorticity, [[row + row_shift], [column + column_shift]],
                                                    order=1, mode="nearest")[0])
        results[vortex_name] = {"found": found, "position": position, "stream_function": value,
                                "vorticity_z": vorticity_at_centre,
                                "reference_position": reference.get(vortex_name, {}).get("position"),
                                "reference_stream_function": reference.get(vortex_name, {}).get("stream_function"),
                                "reference_source": reference.get(vortex_name, {}).get("source")}
    return results


# ----------------------------------------------------------------------------------------------------------
# data loading and the main per-job computation
# ----------------------------------------------------------------------------------------------------------

def read_case_information(meta: dict) -> dict:
    """Grid origin and speed of sound of the job's case (read-only): the solver anchors voxel (0, 0) at the
    frame.obj bounding-box minimum, origin = bbox_min - h/2 (case_loader_v5._compute_grid), and a particle's
    global voxel-x column is floor((x - origin_x) / h) (partition_v5)."""
    case_path = REPOSITORY_ROOT / meta["case"]
    case = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    frame_path = (case_path.parent / case["geometry"]["frame"]).resolve()
    vertices = np.array([line.split()[1:4] for line in frame_path.read_text().splitlines() if line.startswith("v ")],
                        dtype=float)
    smoothing_length = float(case["physics"]["h"])
    return {"case_path": str(case_path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
            "frame_bounding_box_minimum_x": float(vertices[:, 0].min()),
            "grid_origin_x": float(vertices[:, 0].min() - 0.5 * smoothing_length),
            "speed_of_sound": float(meta.get("speed_of_sound", case["physics"]["speed_of_sound"]))}


def relative_errors(run_id: str) -> dict:
    """Time-averaged rel-L2 errors of the validation analysis: L2 / rms(Tc) per line (profile_<run>.csv)."""
    path = DATA_DIRECTORY / f"profile_{run_id}.csv"
    if not path.exists():
        return {}
    rows = [line.split(",") for line in path.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")]
    header, rows = rows[0], rows[1:]
    column = {name: index for index, name in enumerate(header)}
    result = {}
    for key, prefix in (("rel_l2_marchi2021_u", "u_"), ("rel_l2_marchi2021_v", "v_")):
        selected = [row for row in rows if row[column["line"]].startswith(prefix)]
        errors = np.array([float(row[column["error"]]) for row in selected])
        references = np.array([float(row[column["marchi2021_Tc"]]) for row in selected])
        result[key] = float(np.sqrt(np.mean(errors ** 2)) / np.sqrt(np.mean(references ** 2)))
    return result


def load_job(job_directory: pathlib.Path, snapshot_name: str = "final") -> dict:
    """A run of experiment/validation/cavity_runner.py: the newest checkpoint ('final') or the named checkpoint
    ('c<step>'), i.e. the full particle state in float32, plus meta.json / steady.json of the run."""
    checkpoints = sorted((job_directory / "checkpoints").glob("c*.npz"))
    path = checkpoints[-1] if snapshot_name == "final" else job_directory / "checkpoints" / f"{snapshot_name}.npz"
    with np.load(path) as checkpoint:
        data = {"position": checkpoint["position_voxel_id"][:, :2].astype(np.float64),
                "velocity": checkpoint["velocity_mass"][:, :2].astype(np.float64),
                "mass": checkpoint["velocity_mass"][:, 3].astype(np.float64),
                "density": (checkpoint["density_pressure"][:, 0].astype(np.float64)
                            + float(checkpoint["_stored_density_offset"])),
                "material": checkpoint["material"].astype(np.int8)}
    run_meta = json.loads((job_directory / "meta.json").read_text(encoding="utf-8"))
    steady_path = job_directory / "steady.json"
    steady = json.loads(steady_path.read_text(encoding="utf-8")) if steady_path.exists() else {}
    xi, epsilon_factor = run_numerics(job_directory, run_meta)
    spacing = float(run_meta["spacing"])
    step = int(path.stem[1:])
    data["meta"] = {"case": run_meta["case"].replace("\\", "/"), "re": int(round(run_meta["reynolds_nominal"])),
                    "k": int(run_meta["slabs"]), "dx": spacing, "cuts": run_meta.get("cuts") or [],
                    "expected_total": int(run_meta["expected_total"]),
                    # the fluid lattice of these cases has (1/dx + 1)^2 particles
                    "fluid_count_initial": (int(round(1.0 / spacing)) + 1) ** 2}
    data["result"] = {"converged": steady.get("steady_time") is not None, "steady_time": steady.get("steady_time"),
                      "final": relative_errors(job_directory.name)}
    data["case_information"] = read_case_information(data["meta"])
    data["h"] = float(run_meta["support_radius"])
    data["time"] = step * float(run_meta["dt"])
    data["step"] = step
    data["job"] = job_directory.name
    data["snapshot"] = path.stem
    data["setting"] = (f"ξ = {xi:g}, ε² = {epsilon_factor:g} h²"
                       + (" (release)" if xi == 0.1 and epsilon_factor == 0.01 else ""))
    data["frame_scale"] = 1.0
    if FRAME == "wall":
        # innermost boundary rows at +-(0.5 + dx) -> +-0.5: a similarity transform of positions and kernel support,
        # so every interpolated value is unchanged; lengths (psi, positions) are then in units of L = 1 + 2 dx
        scale = 1.0 / (1.0 + 2.0 * spacing)
        data["position"] = data["position"] * scale
        data["h"] = data["h"] * scale
        data["meta"]["dx"] = spacing * scale
        data["case_information"]["grid_origin_x"] *= scale
        data["frame_scale"] = scale
    return data


def measure_particle_spacing(position: np.ndarray, material: np.ndarray, fallback: float) -> float:
    """Lattice spacing = median nearest-neighbour distance of the static wall particles (an exact square
    lattice). meta['dx'] (2 x particle_radius, rounded) is 0.15 % too large and only used as a fallback."""
    wall_position = position[material == MATERIAL_WALL]
    if wall_position.shape[0] < 2:
        return fallback
    return float(np.median(cKDTree(wall_position).query(wall_position, k=2)[0][:, 1]))


def colorize(field_values: np.ndarray, pixel_class: np.ndarray, colormap, normalization,
             mark_interface: bool = False, rows_per_chunk: int = 512) -> np.ndarray:
    """uint8 RGBA [row, column, 4]: field where the nearest particle is fluid, boundary-class colours elsewhere.
    mark_interface paints the boundary-class pixels that touch a fluid pixel (4-neighbourhood) in
    INTERFACE_COLOR, so the band edge stays visible where the field colour is close to the band colour
    (e.g. clipped omega_z = -5 next to the lid); no field pixel is changed."""
    class_colors = {MATERIAL_WALL: WALL_COLOR, MATERIAL_LID: LID_COLOR, PIXEL_CLASS_OUTSIDE: OUTSIDE_COLOR}
    rgba = np.empty(field_values.shape + (4,), dtype=np.uint8)
    for row_start in range(0, field_values.shape[0], rows_per_chunk):
        row_stop = min(field_values.shape[0], row_start + rows_per_chunk)
        chunk_class = pixel_class[row_start:row_stop]
        chunk_rgba = colormap(normalization(field_values[row_start:row_stop]), bytes=True)
        for class_value, color in class_colors.items():
            chunk_rgba[chunk_class == class_value] = np.round(np.array(to_rgba(color)) * 255).astype(np.uint8)
        rgba[row_start:row_stop] = chunk_rgba
    if mark_interface:
        fluid = pixel_class == MATERIAL_FLUID
        touches_fluid = np.zeros_like(fluid)
        touches_fluid[1:, :] |= fluid[:-1, :]
        touches_fluid[:-1, :] |= fluid[1:, :]
        touches_fluid[:, 1:] |= fluid[:, :-1]
        touches_fluid[:, :-1] |= fluid[:, 1:]
        interface = touches_fluid & ((pixel_class == MATERIAL_WALL) | (pixel_class == MATERIAL_LID))
        rgba[interface] = np.round(np.array(to_rgba(INTERFACE_COLOR)) * 255).astype(np.uint8)
    return rgba


def grid_for_job(position: np.ndarray, particle_spacing: float, grid_cells: int):
    """Grid over the full particle extent + dx/2: every cell inside the outermost boundary row's own square."""
    boundary_half_extent = float(np.max(np.abs(position))) + 0.5 * particle_spacing
    spacing = 2.0 * boundary_half_extent / grid_cells
    return UniformGrid(-boundary_half_extent, -boundary_half_extent, spacing, grid_cells, grid_cells), \
        boundary_half_extent


def cache_path_for(cache_directory, job: str, grid_cells: int, snapshot_name: str = "final"):
    if cache_directory is None:
        return None
    suffix = ("" if snapshot_name == "final" else f"_{snapshot_name}") + f"_{FRAME}"
    return pathlib.Path(cache_directory) / f"{job}{suffix}_grid{grid_cells}.npz"


def interpolate_job_onto_grid(data: dict, grid_cells: int, cache_directory, reuse_cache: bool, log,
                              forced_grid: UniformGrid | None = None) -> dict:
    """Smooth velocity field + nearest-particle class on the main grid, loaded from / written to the optional
    cache (one compressed npz per job, snapshot and grid size; float64 velocities, int8 classes). forced_grid
    puts a second snapshot on exactly the grid of a first one (float32 checkpoints would otherwise give a grid
    that differs by round-off)."""
    position = data["position"]
    material = data["material"]
    particle_spacing = measure_particle_spacing(position, material, float(data["meta"]["dx"]))
    if forced_grid is None:
        grid, boundary_half_extent = grid_for_job(position, particle_spacing, grid_cells)
    else:
        grid, boundary_half_extent = forced_grid, -forced_grid.x_minimum
    cache_path = cache_path_for(cache_directory, data["job"], grid_cells, data.get("snapshot", "final"))
    if reuse_cache and cache_path is not None and cache_path.exists():
        with np.load(cache_path) as cached:
            if (abs(float(cached["grid_spacing"]) - grid.spacing) < 1e-15 and "grid_x_minimum" in cached.files
                    and abs(float(cached["grid_x_minimum"]) - grid.x_minimum) < 1e-15):
                log(f"  grid fields loaded from {cache_path}")
                return {"grid": grid, "boundary_half_extent": boundary_half_extent,
                        "particle_spacing": particle_spacing, "velocity_x": cached["velocity_x"],
                        "velocity_y": cached["velocity_y"], "pixel_class": cached["pixel_class"],
                        "weight_sum_zero_cells": int(cached["weight_sum_zero_cells"]), "particle_tree": None}
    volume = data["mass"] / data["density"]
    smoothing_length = float(data["h"])
    timer = time.perf_counter()
    interpolated, weight_sum = scatter_particles_to_grid(position, volume, data["velocity"], grid, smoothing_length)
    log(f"  scatter {grid_cells}^2: {time.perf_counter() - timer:.1f} s")
    velocity_x = np.ascontiguousarray(interpolated[:, :, 0])
    velocity_y = np.ascontiguousarray(interpolated[:, :, 1])
    del interpolated
    timer = time.perf_counter()
    particle_tree = cKDTree(position)
    pixel_class = classify_grid_cells(particle_tree, material, grid, boundary_half_extent)
    log(f"  tree + nearest-particle classification: {time.perf_counter() - timer:.1f} s")
    weight_sum_zero_cells = int(np.count_nonzero(weight_sum <= 0))
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, velocity_x=velocity_x, velocity_y=velocity_y, pixel_class=pixel_class,
                            grid_spacing=grid.spacing, grid_x_minimum=grid.x_minimum,
                            weight_sum_zero_cells=weight_sum_zero_cells)
    return {"grid": grid, "boundary_half_extent": boundary_half_extent, "particle_spacing": particle_spacing,
            "velocity_x": velocity_x, "velocity_y": velocity_y, "pixel_class": pixel_class,
            "weight_sum_zero_cells": weight_sum_zero_cells, "particle_tree": particle_tree}


def slab_cuts_for_job(data: dict) -> list:
    """K - 1 slab cuts in x [m]: global voxel-x column index (meta 'cuts') x h + grid origin, cross-checked
    against the snapshot's 'source' (slab that owned each particle): particles left of a cut must belong to
    a lower slab."""
    cut_columns = data["meta"].get("cuts") or []
    smoothing_length = float(data["h"])
    origin_x = data["case_information"]["grid_origin_x"]
    position_x = data["position"][:, 0]
    if "source" not in data:
        # the merged checkpoint does not record which slab owned a particle: no ownership cross-check
        return [{"column": int(cut_column), "x": origin_x + cut_column * smoothing_length,
                 "ownership_mismatches": None, "owned_left_maximum_x": None, "owned_right_minimum_x": None}
                for cut_column in cut_columns]
    source = data["source"].astype(np.int64)
    column = np.floor((position_x - origin_x) / smoothing_length).astype(np.int64)
    cuts = []
    for slab_index, cut_column in enumerate(cut_columns):
        owned_left = source <= slab_index
        cuts.append({"column": int(cut_column), "x": origin_x + cut_column * smoothing_length,
                     "ownership_mismatches": int(np.count_nonzero((column < cut_column) != owned_left)),
                     "owned_left_maximum_x": float(position_x[owned_left].max()),
                     "owned_right_minimum_x": float(position_x[~owned_left].min())})
    return cuts


def compute_job(data: dict, grid_cells: int, validate_points: int, plotted_stream_function: str, log,
                cache_directory=None, reuse_cache: bool = False) -> dict:
    position = data["position"]
    velocity = data["velocity"]
    material = data["material"]
    volume = data["mass"] / data["density"]
    smoothing_length = float(data["h"])
    reynolds_number = int(data["meta"]["re"])

    grid_fields = interpolate_job_onto_grid(data, grid_cells, cache_directory, reuse_cache, log)
    grid = grid_fields["grid"]
    spacing = grid.spacing
    boundary_half_extent = grid_fields["boundary_half_extent"]
    particle_spacing = grid_fields["particle_spacing"]
    velocity_x = grid_fields["velocity_x"]
    velocity_y = grid_fields["velocity_y"]
    pixel_class = grid_fields["pixel_class"]
    particle_tree = grid_fields["particle_tree"] if grid_fields["particle_tree"] is not None else cKDTree(position)

    validation = None
    if validate_points > 0:
        validation = validate_against_reference(particle_tree, position, volume, velocity, grid, velocity_x,
                                                velocity_y, smoothing_length, validate_points)
        log(f"  validation vs cavity_sampling Shepard at {validation['points']} grid points: max |du| "
            f"{validation['max_abs_difference_u']:.2e}, max |dv| {validation['max_abs_difference_v']:.2e}")

    speed = np.hypot(velocity_x, velocity_y)
    vorticity = np.gradient(velocity_y, spacing, axis=1) - np.gradient(velocity_x, spacing, axis=0)
    fluid_pixels = pixel_class == MATERIAL_FLUID

    # Effective wall: half-way between the outermost fluid row (nominal edge, |x| = 0.5) and the first
    # boundary row (the innermost boundary particle in the max-norm).
    boundary_particles = material != MATERIAL_FLUID
    first_boundary_row = float(np.min(np.max(np.abs(position[boundary_particles]), axis=1)))
    effective_wall_half_extent = 0.5 * (FLUID_HALF_WIDTH + first_boundary_row)
    timer = time.perf_counter()
    stream_functions = {
        "velocity-integral": cumulative_trapezoid(np.nan_to_num(velocity_x), dx=spacing, axis=0, initial=0.0),
        "poisson": stream_function_from_vorticity(vorticity, grid, effective_wall_half_extent),
    }
    log(f"  stream functions: {time.perf_counter() - timer:.1f} s")

    interior_margin = 2.0 * smoothing_length
    absolute_x = np.abs(grid.x_centres)[None, :]
    absolute_y = np.abs(grid.y_centres)[:, None]
    interior = (absolute_x < FLUID_HALF_WIDTH - interior_margin) & (absolute_y < FLUID_HALF_WIDTH - interior_margin)
    vortex_centres = {name: find_vortex_centres(stream_function, vorticity, fluid_pixels & interior, grid,
                                                reynolds_number)
                      for name, stream_function in stream_functions.items()}

    # Diagnostics of the two stream functions.
    #  * velocity integral sampled ON the effective walls (left, right, top; the bottom is its start): ideally 0,
    #    corners within 2h excluded;
    #  * Poisson: |u - curl psi| = the non-solenoidal part of the interpolated field (away from the walls);
    #  * divergence of the interpolated field.
    wall_closure = sample_on_effective_walls(stream_functions["velocity-integral"], grid, effective_wall_half_extent,
                                             corner_margin=2.0 * smoothing_length)
    away_from_walls = fluid_pixels & (absolute_x < FLUID_HALF_WIDTH - 0.05) & (absolute_y < FLUID_HALF_WIDTH - 0.05)
    poisson = stream_functions["poisson"]
    solenoidal_mismatch = np.hypot(velocity_x - np.gradient(poisson, spacing, axis=0),
                                   velocity_y + np.gradient(poisson, spacing, axis=1))[away_from_walls]
    divergence = (np.gradient(velocity_x, spacing, axis=1) + np.gradient(velocity_y, spacing, axis=0))[away_from_walls]
    stream_function_diagnostics = {
        "effective_wall": effective_wall_half_extent,
        "velocity_integral_on_left_wall_maximum_absolute": float(np.max(np.abs(wall_closure["left"]))),
        "velocity_integral_on_right_wall_maximum_absolute": float(np.max(np.abs(wall_closure["right"]))),
        "velocity_integral_on_lid_wall_maximum_absolute": float(np.max(np.abs(wall_closure["top"]))),
        "velocity_integral_on_lid_wall_mean": float(np.mean(wall_closure["top"])),
        "velocity_integral_minus_poisson_maximum_absolute_interior": float(np.max(np.abs(
            stream_functions["velocity-integral"] - poisson)[fluid_pixels & interior])),
        "poisson_velocity_mismatch_median": float(np.median(solenoidal_mismatch)),
        "poisson_velocity_mismatch_percentile_99": float(np.percentile(solenoidal_mismatch, 99)),
        "poisson_velocity_mismatch_maximum": float(np.max(solenoidal_mismatch)),
        "divergence_root_mean_square": float(np.sqrt(np.mean(divergence ** 2))),
        "divergence_maximum_absolute": float(np.max(np.abs(divergence))),
    }
    del solenoidal_mismatch, divergence, away_from_walls

    slab_cuts = slab_cuts_for_job(data) if int(data["meta"].get("k", 1)) > 1 else []
    fluid_particles = material == MATERIAL_FLUID
    mean_fluid_velocity = np.average(velocity[fluid_particles], axis=0, weights=data["mass"][fluid_particles])

    return {"grid": grid, "velocity_x": velocity_x, "velocity_y": velocity_y, "speed": speed,
            "stream_function": stream_functions[plotted_stream_function],
            "plotted_stream_function": plotted_stream_function,
            "vorticity": vorticity, "pixel_class": pixel_class,
            "weight_sum_zero_cells": grid_fields["weight_sum_zero_cells"],
            "particle_tree": particle_tree, "volume": volume, "smoothing_length": smoothing_length,
            "particle_spacing": particle_spacing, "boundary_half_extent": boundary_half_extent,
            "effective_wall_half_extent": effective_wall_half_extent,
            "reynolds_number": reynolds_number, "vortex_centres": vortex_centres[plotted_stream_function],
            "vortex_centres_by_stream_function": vortex_centres, "validation": validation,
            "stream_function_diagnostics": stream_function_diagnostics, "slab_cuts": slab_cuts,
            "slab_cut_x": [cut["x"] for cut in slab_cuts], "mean_fluid_velocity": [float(value) for value in
                                                                                  mean_fluid_velocity]}


def validate_against_reference(particle_tree, position, volume, velocity, grid, velocity_x, velocity_y,
                               smoothing_length, point_count: int) -> dict:
    """Compare the scatter result with the Shepard interpolation of experiment/validation/cavity_sampling.py at
    point_count grid points (2/3 uniform random, 1/3 inside the four corner zoom windows)."""

    generator = np.random.default_rng(20261002)
    uniform_count = (2 * point_count) // 3
    rows = list(generator.integers(0, grid.row_count, uniform_count))
    columns = list(generator.integers(0, grid.column_count, uniform_count))
    corner_cells = int(math.ceil((ZOOM_BOUNDARY_SPAN + ZOOM_FLUID_SPAN) / grid.spacing))
    per_corner = (point_count - uniform_count) // 4
    for corner_row_low in (0, grid.row_count - corner_cells):
        for corner_column_low in (0, grid.column_count - corner_cells):
            rows += list(generator.integers(corner_row_low, corner_row_low + corner_cells, per_corner))
            columns += list(generator.integers(corner_column_low, corner_column_low + corner_cells, per_corner))
    rows = np.asarray(rows)
    columns = np.asarray(columns)
    points = np.column_stack([grid.x_centres[columns], grid.y_centres[rows]])
    particles = cavity_sampling.ParticleSet(position, velocity, volume)
    reference = cavity_sampling.interpolate(particles, points, smoothing_length, tree=particle_tree)["shepard"]
    difference_u = np.abs(reference[:, 0] - velocity_x[rows, columns])
    difference_v = np.abs(reference[:, 1] - velocity_y[rows, columns])
    return {"points": int(points.shape[0]), "max_abs_difference_u": float(np.nanmax(difference_u)),
            "max_abs_difference_v": float(np.nanmax(difference_v)),
            "nan_mismatch": int(np.count_nonzero(np.isnan(reference).any(axis=1)
                                                 != np.isnan(velocity_x[rows, columns])))}


def compute_zoom(data: dict, fields: dict, corner_name: str) -> dict:
    """Local fine grid (dx/6) Shepard field + every particle of one corner window."""
    corner_x, corner_y, _ = ZOOM_CORNERS[corner_name]
    window_size = ZOOM_BOUNDARY_SPAN + ZOOM_FLUID_SPAN
    cavity_x_low = corner_x - ZOOM_BOUNDARY_SPAN if corner_x == 0.0 else corner_x - ZOOM_FLUID_SPAN
    cavity_y_low = corner_y - ZOOM_BOUNDARY_SPAN if corner_y == 0.0 else corner_y - ZOOM_FLUID_SPAN
    x_low, y_low = cavity_x_low - 0.5, cavity_y_low - 0.5

    position = data["position"]
    velocity = data["velocity"]
    material = data["material"]
    volume = fields["volume"]
    smoothing_length = fields["smoothing_length"]
    local_spacing = fields["particle_spacing"] / ZOOM_GRID_CELLS_PER_PARTICLE_SPACING
    local_cells = int(math.ceil(window_size / local_spacing))
    local_grid = UniformGrid(x_low, y_low, window_size / local_cells, local_cells, local_cells)

    reach = smoothing_length + 2.0 * local_grid.spacing
    near = np.flatnonzero((position[:, 0] > x_low - reach) & (position[:, 0] < x_low + window_size + reach)
                          & (position[:, 1] > y_low - reach) & (position[:, 1] < y_low + window_size + reach))
    interpolated, _ = scatter_particles_to_grid(position[near], volume[near], velocity[near], local_grid,
                                                smoothing_length)
    local_velocity_x, local_velocity_y = interpolated[:, :, 0], interpolated[:, :, 1]
    local_class = classify_grid_cells(fields["particle_tree"], material, local_grid, fields["boundary_half_extent"])

    margin = 0.6 * fields["particle_spacing"]
    in_window = np.flatnonzero((position[:, 0] > x_low - margin) & (position[:, 0] < x_low + window_size + margin)
                               & (position[:, 1] > y_low - margin) & (position[:, 1] < y_low + window_size + margin))
    window_fluid = in_window[material[in_window] == MATERIAL_FLUID]
    fluid_vorticity, fluid_vorticity_plain_sum = particle_vorticity(fields["particle_tree"], position, velocity, volume,
                                                                    window_fluid, smoothing_length)
    local_speed = np.hypot(local_velocity_x, local_velocity_y)
    local_vorticity = (np.gradient(local_velocity_y, local_grid.spacing, axis=1)
                       - np.gradient(local_velocity_x, local_grid.spacing, axis=0))
    # Per-particle check: renormalised particle vorticity vs the local smooth field at the particle position
    # (bilinear), for particles at least h from the boundary band so the smooth field is not wall-affected.
    deep = np.max(np.abs(position[window_fluid]), axis=1) < FLUID_HALF_WIDTH - smoothing_length
    if np.any(deep):
        sample_columns = (position[window_fluid[deep], 0] - local_grid.x_centres[0]) / local_grid.spacing
        sample_rows = (position[window_fluid[deep], 1] - local_grid.y_centres[0]) / local_grid.spacing
        smooth_at_particles = map_coordinates(local_vorticity, [sample_rows, sample_columns], order=1, mode="nearest")
        vorticity_check = {"particles": int(np.count_nonzero(deep)),
                           "median_absolute_difference": float(np.median(np.abs(fluid_vorticity[deep]
                                                                                 - smooth_at_particles))),
                           "median_absolute_smooth": float(np.median(np.abs(smooth_at_particles)))}
    else:
        vorticity_check = {"particles": 0}
    with np.errstate(divide="ignore", invalid="ignore"):
        plain_over_renormalised = fluid_vorticity_plain_sum / fluid_vorticity
    meaningful = np.abs(fluid_vorticity) > 0.5 * np.percentile(np.abs(fluid_vorticity), 90)
    vorticity_check["plain_sum_over_renormalised_median"] = (float(np.median(plain_over_renormalised[meaningful]))
                                                             if np.any(meaningful) else float("nan"))
    return {"name": corner_name, "grid": local_grid, "cavity_window": (cavity_x_low, cavity_y_low, window_size),
            "speed": local_speed,
            "vorticity": local_vorticity,
            "pixel_class": local_class,
            "fluid_indices": window_fluid,
            "fluid_speed": np.hypot(velocity[window_fluid, 0], velocity[window_fluid, 1]),
            "fluid_vorticity": fluid_vorticity,
            "vorticity_check": vorticity_check,
            "wall_indices": in_window[material[in_window] == MATERIAL_WALL],
            "lid_indices": in_window[material[in_window] == MATERIAL_LID]}


def nice_ceiling(value: float) -> float:
    """Smallest 'round' number >= value from the sequence 1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8 x 10^n."""
    if not np.isfinite(value) or value <= 0.0:
        return 1.0
    exponent = math.floor(math.log10(value))
    for mantissa in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        candidate = mantissa * 10.0 ** exponent
        if candidate >= value * (1.0 - 1e-12):
            return candidate
    return 10.0 ** (exponent + 1)


def zoom_color_scale(field_name: str, zoom: dict) -> dict:
    """Colour scale of one corner zoom, shared by its smooth background and its particles.

    The corner values span five decades (speed 1e-3 U in the bottom corners, |omega_z| up to ~400 at the lid
    corners), so one shared scale would leave most zooms uniform. Velocity: linear 0 .. nice ceiling of the
    99.5th percentile of the window's raw particle speeds. Vorticity: symmetric linear +-(nice ceiling of the
    99.5th percentile of |omega_z|) when that is <= VORTICITY_CLIP, otherwise symmetric log (linear within
    +-1 U/L) so the lid-corner singularity keeps its structure."""
    values = zoom["fluid_vorticity"] if field_name == "vorticity" else zoom["fluid_speed"]
    if field_name == "velocity":
        top = nice_ceiling(float(np.percentile(values, 99.5)))
        normalization = Normalize(0.0, top)
        clipped = int(np.count_nonzero(values > top))
        description = f"colour 0 – {top:g} linear"
        extend = "max" if clipped else "neither"
    else:
        top = nice_ceiling(float(np.percentile(np.abs(values), 99.5)))
        if top <= VORTICITY_CLIP:
            normalization = Normalize(-top, top)
            description = f"colour ±{top:g} linear"
        else:
            normalization = SymLogNorm(linthresh=1.0, linscale=1.0, vmin=-top, vmax=top, base=10)
            description = f"colour ±{top:g} symlog"
        clipped = int(np.count_nonzero(np.abs(values) > top))
        extend = "both" if clipped else "neither"
    if clipped:
        description += f", {100.0 * clipped / values.size:.1f} % clipped"
    return {"normalization": normalization, "description": description, "extend": extend,
            "minimum": float(values.min()), "maximum": float(values.max()), "top": top}


# ----------------------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------------------

# Layout in inches (figure 18 x 10 in at 300 dpi): title block on top, main square panel in the middle,
# TL/BL zooms left and TR/BR zooms right of it (each with its own thin colorbar on the side facing the main
# panel), the main colorbar right of the main panel, legends along the bottom edge.
FIGURE_WIDTH = 18.0
FIGURE_HEIGHT = 10.0
TOP_MARGIN = 1.36
INSET_SIZE = 3.2
INSET_GAP = 0.72
INSET_COLORBAR_GAP = 0.07
INSET_COLORBAR_WIDTH = 0.09
INSET_LEFT_COLUMN = 0.56
INSET_RIGHT_COLUMN = 14.02
INSET_TOP_ROW_BOTTOM = FIGURE_HEIGHT - TOP_MARGIN - INSET_SIZE
INSET_BOTTOM_ROW_BOTTOM = INSET_TOP_ROW_BOTTOM - INSET_GAP - INSET_SIZE
MAIN_SIZE = 2.0 * INSET_SIZE + INSET_GAP
MAIN_LEFT = 5.05
MAIN_BOTTOM = FIGURE_HEIGHT - TOP_MARGIN - MAIN_SIZE
COLORBAR_LEFT = MAIN_LEFT + MAIN_SIZE + 0.12
COLORBAR_WIDTH = 0.17


def axes_in_inches(figure, left, bottom, width, height):
    return figure.add_axes([left / FIGURE_WIDTH, bottom / FIGURE_HEIGHT, width / FIGURE_WIDTH,
                            height / FIGURE_HEIGHT])


FIELD_STYLE = {
    "velocity": {"colormap": "viridis", "normalization": Normalize(0.0, 1.0),
                 "colorbar_label": "speed |u| / U   (U = lid speed = 1 m/s)", "extend": "neither",
                 "measured_marker_face": "#ff3b30", "slab_cut_color": "#ffffff",
                 "zoom_quantity": "|u|/U"},
    "vorticity": {"colormap": "RdBu_r", "normalization": Normalize(-VORTICITY_CLIP, VORTICITY_CLIP),
                  "colorbar_label": f"vorticity  ω_z = ∂v/∂x − ∂u/∂y   [U/L],  "
                                    f"colour clipped at ±{VORTICITY_CLIP:g}",
                  "extend": "both", "measured_marker_face": "#ffd400",
                  "slab_cut_color": "#333333", "zoom_quantity": "ω_z"},
}
NEGATIVE_STREAM_FUNCTION_COLOR = "#ffffff"
POSITIVE_STREAM_FUNCTION_COLOR = "#ff9e7a"
GHIA_MARKER_EDGE = "#000000"                                   # open Ghia markers: black ring with a white halo
GHIA_MARKER_HALO = [patheffects.withStroke(linewidth=3.2, foreground="#ffffff")]


def zoom_window(corner_name: str):
    """(X low, Y low, size) of a corner zoom window in cavity coordinates."""
    corner_x, corner_y, _ = ZOOM_CORNERS[corner_name]
    window_size = ZOOM_BOUNDARY_SPAN + ZOOM_FLUID_SPAN
    x_low = corner_x - ZOOM_BOUNDARY_SPAN if corner_x == 0.0 else corner_x - ZOOM_FLUID_SPAN
    y_low = corner_y - ZOOM_BOUNDARY_SPAN if corner_y == 0.0 else corner_y - ZOOM_FLUID_SPAN
    return x_low, y_low, window_size


def draw_main_panel(axes, field_name: str, data: dict, fields: dict, rgba: np.ndarray):
    style = FIELD_STYLE[field_name]
    grid = fields["grid"]
    axes.imshow(rgba, origin="lower", extent=grid.extent_cavity, interpolation="antialiased",
                interpolation_stage="rgba")
    cavity_x = grid.x_centres + 0.5
    cavity_y = grid.y_centres + 0.5
    fluid_pixels = fields["pixel_class"] == MATERIAL_FLUID
    effective_wall = fields["effective_wall_half_extent"]
    if field_name == "velocity":
        # psi = 0 on the effective wall, so the smallest levels can only trace the wall itself: leave out the
        # cells within STREAM_FUNCTION_WALL_MASK_SPACINGS dx of it.
        wall_distance = np.minimum(effective_wall - np.abs(grid.x_centres)[None, :],
                                   effective_wall - np.abs(grid.y_centres)[:, None])
        near_wall = wall_distance < STREAM_FUNCTION_WALL_MASK_SPACINGS * fields["particle_spacing"]
        stream_function = np.ma.masked_where(~fluid_pixels | near_wall, fields["stream_function"])
        negative_levels = [level for level in GHIA_STREAM_FUNCTION_LEVELS if level < 0]
        positive_levels = [level for level in GHIA_STREAM_FUNCTION_LEVELS if level > 0]
        axes.contour(cavity_x, cavity_y, stream_function, levels=negative_levels,
                     colors=NEGATIVE_STREAM_FUNCTION_COLOR, linewidths=0.45, linestyles="solid")
        axes.contour(cavity_x, cavity_y, stream_function, levels=positive_levels,
                     colors=POSITIVE_STREAM_FUNCTION_COLOR, linewidths=0.55, linestyles="solid")
    else:
        # Contours trace a lightly smoothed copy (the core is a flat omega_z ~ -2 plateau whose small
        # fluctuations break the -2 level into speckles); the colours show the unsmoothed field.
        smoothed = gaussian_filter(np.nan_to_num(fields["vorticity"]), sigma=VORTICITY_CONTOUR_SMOOTHING_CELLS)
        vorticity = np.ma.masked_where(~fluid_pixels, smoothed)
        vorticity_levels = sorted(-level for level in GHIA_VORTICITY_LEVELS_GHIA_SIGN)
        axes.contour(cavity_x, cavity_y, vorticity, levels=vorticity_levels, colors="#000000",
                     linewidths=0.4, negative_linestyles="dashed")
        del smoothed, vorticity

    # Fluid / boundary interface (effective wall), so the band edge stays visible against any field colour.
    axes.add_patch(Rectangle((0.5 - effective_wall, 0.5 - effective_wall), 2.0 * effective_wall, 2.0 * effective_wall,
                             fill=False, edgecolor=INTERFACE_COLOR, linewidth=0.6, zorder=4))

    for vortex_name, centre in fields["vortex_centres"].items():
        if field_name == "vorticity" and vortex_name != "primary":
            continue
        marker = VORTEX_SEARCH[vortex_name]["marker"]
        if centre["reference_position"] is not None:
            axes.plot(*centre["reference_position"], marker=marker, markersize=9, markerfacecolor="none",
                      markeredgecolor=GHIA_MARKER_EDGE, markeredgewidth=1.2, linestyle="none", zorder=6,
                      path_effects=GHIA_MARKER_HALO)
        if centre["found"]:
            axes.plot(*centre["position"], marker=marker, markersize=5.5, markerfacecolor=style["measured_marker_face"],
                      markeredgecolor="#000000", markeredgewidth=0.6, linestyle="none", zorder=7)

    extent = grid.extent_cavity
    label_offset = 0.006
    for cut_x in fields["slab_cut_x"]:
        axes.axvline(cut_x + 0.5, color=style["slab_cut_color"], linewidth=0.7, linestyle=(0, (1.5, 2.5)),
                     alpha=0.9, zorder=5)
        axes.text(cut_x + 0.5, extent[3] + label_offset, f"K = {data['meta']['k']} slab cut  X = {cut_x + 0.5:.4f}",
                  color=TEXT_COLOR, fontsize=7.5, va="bottom", ha="center", clip_on=False, zorder=8)

    # Zoom windows; their labels sit outside the axes (above the top edge / in the tick-label row below the
    # bottom edge), next to the rectangle on the side facing the cavity centre, so they cover no eddy.
    for corner_name, (corner_x, corner_y, _) in ZOOM_CORNERS.items():
        x_low, y_low, window_size = zoom_window(corner_name)
        axes.add_patch(Rectangle((x_low, y_low), window_size, window_size, fill=False, edgecolor=ANNOTATION_COLOR,
                                 linewidth=1.3, zorder=9, clip_on=False))
        label_x = x_low + window_size + 0.008 if corner_x == 0.0 else x_low - 0.008
        label_y = extent[3] + label_offset if corner_y == 1.0 else extent[2] - label_offset
        label_vertical = "bottom" if corner_y == 1.0 else "top"
        if corner_y == 0.0 and window_size > 0.05:
            # wide windows (coarse cases) would put the bottom labels on the X tick labels: label above the box
            label_x = x_low + 0.004 if corner_x == 0.0 else x_low + window_size - 0.004
            label_y, label_vertical = y_low + window_size + 0.006, "bottom"
        axes.text(label_x, label_y, corner_name, color=ANNOTATION_COLOR, fontsize=10, fontweight="bold",
                  ha="left" if corner_x == 0.0 else "right", va=label_vertical, zorder=9,
                  clip_on=False)

    axes.set_xlim(extent[0], extent[1])
    axes.set_ylim(extent[2], extent[3])
    axes.set_aspect("equal")
    axes.set_xticks(np.linspace(0.0, 1.0, 11))
    axes.set_yticks(np.linspace(0.0, 1.0, 11))
    axes.tick_params(labelsize=9, colors=TEXT_COLOR)
    axes.set_xlabel("X = x/L + 1/2", fontsize=10, color=TEXT_COLOR)
    axes.set_ylabel("Y = y/L + 1/2", fontsize=10, color=TEXT_COLOR)


def draw_zoom_panel(axes, colorbar_axes, field_name: str, data: dict, fields: dict, zoom: dict,
                    inset_size_inches: float, colorbar_on_left: bool):
    style = FIELD_STYLE[field_name]
    colormap = plt.get_cmap(style["colormap"])
    color_scale = zoom_color_scale(field_name, zoom)
    normalization = color_scale["normalization"]
    local_grid = zoom["grid"]
    rgba = colorize(zoom[field_name if field_name == "vorticity" else "speed"], zoom["pixel_class"], colormap,
                    normalization)
    # Lighten the smooth field (fluid cells only) so the particles, drawn in full colour on top, stay
    # distinguishable where the field is uniform (e.g. the dark, almost still bottom corners).
    fluid_pixels = zoom["pixel_class"] == MATERIAL_FLUID
    rgba[fluid_pixels, :3] = np.round(rgba[fluid_pixels, :3] * (1.0 - ZOOM_BACKGROUND_LIGHTENING)
                                      + 255.0 * ZOOM_BACKGROUND_LIGHTENING).astype(np.uint8)
    axes.imshow(rgba, origin="lower", extent=local_grid.extent_cavity, interpolation="nearest")

    cavity_x_low, cavity_y_low, window_size = zoom["cavity_window"]
    points_per_cavity_unit = inset_size_inches * 72.0 / window_size
    particle_spacing_points = fields["particle_spacing"] * points_per_cavity_unit
    circle_diameter = 0.78 * particle_spacing_points
    square_side = 0.80 * particle_spacing_points

    position = data["position"]
    for indices, color in ((zoom["wall_indices"], WALL_COLOR), (zoom["lid_indices"], LID_COLOR)):
        axes.scatter(position[indices, 0] + 0.5, position[indices, 1] + 0.5, s=square_side ** 2, marker="s",
                     c=color, edgecolors="#ffffff", linewidths=0.35, zorder=3)
    fluid_values = zoom["fluid_vorticity"] if field_name == "vorticity" else zoom["fluid_speed"]
    fluid_indices = zoom["fluid_indices"]
    axes.scatter(position[fluid_indices, 0] + 0.5, position[fluid_indices, 1] + 0.5, s=circle_diameter ** 2,
                 marker="o", c=fluid_values, cmap=colormap, norm=normalization, edgecolors="#1a1a1a",
                 linewidths=0.35, zorder=4)

    corner_x, corner_y, long_name = ZOOM_CORNERS[zoom["name"]]
    # Nominal cavity edge: above the background, below the particles, so it shows only between particles.
    axes.axvline(corner_x, color=ANNOTATION_COLOR, linewidth=0.8, zorder=2.5)
    axes.axhline(corner_y, color=ANNOTATION_COLOR, linewidth=0.8, zorder=2.5)
    axes.set_xlim(cavity_x_low, cavity_x_low + window_size)
    axes.set_ylim(cavity_y_low, cavity_y_low + window_size)
    axes.set_aspect("equal")
    tick_step = 0.01 if window_size <= 0.06 else 0.05 if window_size <= 0.3 else 0.1   # ~4-7 ticks per window
    tick_values = np.arange(-0.1, 1.1001, tick_step)
    axes.set_xticks([value for value in tick_values if cavity_x_low <= value <= cavity_x_low + window_size + 1e-9])
    axes.set_yticks([value for value in tick_values if cavity_y_low <= value <= cavity_y_low + window_size + 1e-9])
    axes.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axes.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axes.tick_params(labelsize=8, colors=TEXT_COLOR, length=3)
    if colorbar_on_left:
        axes.yaxis.tick_right()
    for spine in axes.spines.values():
        spine.set_edgecolor(ANNOTATION_COLOR)
        spine.set_linewidth(1.6)
    raw_format = ".4f" if field_name == "velocity" else "+.3g"
    axes.set_title(f"{zoom['name']}  {long_name}:  {fluid_indices.size} fluid, {zoom['wall_indices'].size} wall, "
                   f"{zoom['lid_indices'].size} lid\nparticles {style['zoom_quantity']} "
                   f"{color_scale['minimum']:{raw_format}} … {color_scale['maximum']:{raw_format}};  "
                   f"{color_scale['description']}",
                   fontsize=8, color=TEXT_COLOR, pad=4, linespacing=1.3)

    colorbar = plt.colorbar(ScalarMappable(norm=normalization, cmap=colormap), cax=colorbar_axes,
                            extend=color_scale["extend"])
    colorbar.ax.tick_params(labelsize=7, length=2)
    if colorbar_on_left:
        colorbar.ax.yaxis.set_ticks_position("left")
        colorbar.ax.yaxis.set_label_position("left")
    if isinstance(normalization, SymLogNorm):
        top = color_scale["top"]
        ticks = [0.0] + [sign * 10.0 ** power for power in range(0, 6) for sign in (-1.0, 1.0)
                         if 10.0 ** power <= top]
        colorbar.set_ticks(sorted(ticks))
        colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        colorbar.ax.minorticks_off()
    else:
        colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))   # plain numbers, no offset text


def render_field_figure(field_name: str, data: dict, fields: dict, zooms: dict, output_directory: pathlib.Path,
                        log) -> list[pathlib.Path]:
    style = FIELD_STYLE[field_name]
    colormap = plt.get_cmap(style["colormap"])
    normalization = style["normalization"]
    field_values = fields["speed"] if field_name == "velocity" else fields["vorticity"]

    job = data["job"]
    full_path = output_directory / f"{job}_{field_name}_full.png"
    rgba = colorize(field_values, fields["pixel_class"], colormap, normalization, mark_interface=True)
    plt.imsave(full_path, rgba, origin="lower")
    del rgba

    figure = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    figure.patch.set_facecolor("#ffffff")
    main_axes = axes_in_inches(figure, MAIN_LEFT, MAIN_BOTTOM, MAIN_SIZE, MAIN_SIZE)
    rgba = colorize(field_values, fields["pixel_class"], colormap, normalization)
    draw_main_panel(main_axes, field_name, data, fields, rgba)
    del rgba

    colorbar_axes = axes_in_inches(figure, COLORBAR_LEFT, MAIN_BOTTOM, COLORBAR_WIDTH, MAIN_SIZE)
    colorbar = figure.colorbar(ScalarMappable(norm=normalization, cmap=colormap), cax=colorbar_axes,
                               extend=style["extend"])
    colorbar.set_label(style["colorbar_label"], fontsize=10, color=TEXT_COLOR)
    colorbar.ax.tick_params(labelsize=9)
    if field_name == "vorticity":
        colorbar.set_ticks(np.arange(-5, 6, 1))

    inset_positions = {"TL": (INSET_LEFT_COLUMN, INSET_TOP_ROW_BOTTOM), "BL": (INSET_LEFT_COLUMN, INSET_BOTTOM_ROW_BOTTOM),
                       "TR": (INSET_RIGHT_COLUMN, INSET_TOP_ROW_BOTTOM),
                       "BR": (INSET_RIGHT_COLUMN, INSET_BOTTOM_ROW_BOTTOM)}
    for corner_name, (left, bottom) in inset_positions.items():
        inset_axes = axes_in_inches(figure, left, bottom, INSET_SIZE, INSET_SIZE)
        colorbar_on_left = left > MAIN_LEFT
        colorbar_left = (left - INSET_COLORBAR_GAP - INSET_COLORBAR_WIDTH if colorbar_on_left
                         else left + INSET_SIZE + INSET_COLORBAR_GAP)
        inset_colorbar_axes = axes_in_inches(figure, colorbar_left, bottom, INSET_COLORBAR_WIDTH, INSET_SIZE)
        draw_zoom_panel(inset_axes, inset_colorbar_axes, field_name, data, fields, zooms[corner_name], INSET_SIZE,
                        colorbar_on_left)

    result = data["result"]
    final = result.get("final", {})
    meta = data["meta"]
    particle_count = data["position"].shape[0]
    steady_time = result.get("steady_time")
    title = (f"Lid-driven cavity Re {meta['re']},  {data['setting']},  N = {particle_count:,},  K = {meta['k']},  "
             f"t* = tU/L = {float(data['time']):.2f}"
             + (f" (steady from t* = {steady_time:.1f})" if steady_time is not None else " (NOT steady)"))
    frame_text = ("unit square = innermost boundary rows (X = (x + 0.5 + dx) / (1 + 2 dx))" if FRAME == "wall"
                  else "unit square = fluid lattice box (X = x + 0.5)")
    subtitle_lines = [
        f"run {data['job']} (v6 release switches), newest checkpoint {data['snapshot']};  {frame_text};  rel-L2 vs Marchi et al. 2021 "
        f"of the time-averaged centre lines (t* 80-100):  u(Y) {100 * final.get('rel_l2_marchi2021_u', float('nan')):.2f} %,"
        f"  v(X) {100 * final.get('rel_l2_marchi2021_v', float('nan')):.2f} %",
        "smooth field = Shepard interpolation (2-D Wendland C4, support h = "
        f"{fields['smoothing_length'] * 1000:.3f} mm = {fields['smoothing_length'] / fields['particle_spacing']:.2f} dx, "
        f"V = m/ρ) of ALL particles (fluid + wall + lid) on a {fields['grid'].column_count}² grid; "
        "cells whose nearest particle is a wall / lid particle show the boundary class at its true thickness",
    ]
    zoom_note = ("zooms: every particle drawn; each zoom has its OWN colour scale (thin colorbar beside it), "
                 "shared by its smooth background and its particles")
    if field_name == "velocity":
        if fields["plotted_stream_function"] == "poisson":
            subtitle_lines.append(
                "streamlines: ψ from ∇²ψ = −ω_z, ψ = 0 on the effective walls (white line, "
                + ("on the innermost boundary rows" if FRAME == "wall" else "half-way between fluid edge and first boundary row")
                + "; "
                "and first boundary row; not contoured within 1 dx of it); ∫u dy not used: the stored velocities carry "
                f"a net flux, mean fluid u = {fields['mean_fluid_velocity'][0]:+.1e} U")
        else:
            subtitle_lines.append("streamlines: ψ = ∫u dy from the bottom wall (no wall closure enforced)")
        subtitle_lines.append(zoom_note + "; particle colour = its stored speed")
    else:
        subtitle_lines.append(
            "ω_z from central differences of the smooth field; contour lines drawn on a Gaussian-smoothed copy "
            f"(σ = {VORTICITY_CONTOUR_SMOOTHING_CELLS * fields['grid'].spacing / fields['particle_spacing']:.1f} dx), "
            "colours unsmoothed; lid-corner zooms use a symmetric-log colour scale, linear for |ω_z| < 1 U/L")
        subtitle_lines.append(zoom_note + "; particle colour = renormalised SPH vorticity "
                              "∇f_i = B_i⁻ᵀ Σ_j V_j (f_j − f_i) ∇_i W_ij,  B_i = Σ_j V_j (x_j − x_i) ⊗ ∇_i W_ij")
    figure.text(0.5, 1.0 - 0.14 / FIGURE_HEIGHT, title, ha="center", va="top", fontsize=13, color=TEXT_COLOR,
                fontweight="bold")
    figure.text(0.5, 1.0 - 0.42 / FIGURE_HEIGHT, "\n".join(subtitle_lines), ha="center", va="top", fontsize=8.5,
                color=MUTED_TEXT_COLOR, linespacing=1.35)

    # Particle / annotation legend (bottom left) and field-specific legend (bottom right).
    middle_color = colormap(normalization(0.5 if field_name == "velocity" else -1.5))
    lightened_color = tuple(channel * (1.0 - ZOOM_BACKGROUND_LIGHTENING) + ZOOM_BACKGROUND_LIGHTENING
                            for channel in middle_color[:3])
    particle_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=8, markerfacecolor=middle_color,
               markeredgecolor="#1a1a1a", markeredgewidth=0.6, label="fluid (interior) particle, own raw value"),
        Line2D([], [], marker="s", linestyle="none", markersize=8, markerfacecolor=WALL_COLOR,
               markeredgecolor=WALL_COLOR, label="wall particle / wall band (static, u = 0)"),
        Line2D([], [], marker="s", linestyle="none", markersize=8, markerfacecolor=LID_COLOR,
               markeredgecolor=LID_COLOR, label="lid particle / lid band (u = U, +X)"),
        Line2D([], [], color=INTERFACE_COLOR, linewidth=1.0,
               path_effects=[patheffects.withStroke(linewidth=2.4, foreground="#777777")],
               label="fluid / boundary interface (effective wall, main panel)"),
        Patch(facecolor=lightened_color, edgecolor="none",
              label=f"zoom background: smooth field, lightened {100 * ZOOM_BACKGROUND_LIGHTENING:.0f} %"),
        Line2D([], [], color=ANNOTATION_COLOR, linewidth=1.2,
               label=("unit-square edge X or Y = 0, 1 = innermost boundary row (zooms)" if FRAME == "wall"
                      else "nominal cavity edge X or Y = 0, 1 (zooms)")),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor=ANNOTATION_COLOR, linewidth=1.3,
                  label=f"zoom window ({ZOOM_BOUNDARY_SPAN + ZOOM_FLUID_SPAN:.3f} L wide)"),
    ]
    if fields["slab_cut_x"]:
        particle_handles.append(Line2D([], [], color="#333333", linewidth=0.9, linestyle=(0, (1.5, 2.5)),
                                       label="K = 2 slab cut (voxel column × h + grid origin)"))
    figure.legend(handles=particle_handles, loc="lower left", bbox_to_anchor=(0.1 / FIGURE_WIDTH, 0.03 / FIGURE_HEIGHT),
                  ncol=2, fontsize=8.5, frameon=False, handlelength=1.6, columnspacing=1.4, labelspacing=0.45)

    centres = fields["vortex_centres"]
    if field_name == "velocity":
        white_line_outline = [patheffects.withStroke(linewidth=2.6, foreground="#555555")]
        vortex_handles = [
            Line2D([], [], color=NEGATIVE_STREAM_FUNCTION_COLOR, linewidth=1.2, path_effects=white_line_outline,
                   label="ψ < 0 (primary circulation)"),
            Line2D([], [], color=POSITIVE_STREAM_FUNCTION_COLOR, linewidth=1.2,
                   label="ψ > 0 (counter-rotating eddies)"),
        ]
        for vortex_name, centre in centres.items():
            marker = VORTEX_SEARCH[vortex_name]["marker"]
            short_name = "primary" if vortex_name == "primary" else vortex_name
            vortex_handles.append(Line2D(
                [], [], marker=marker, linestyle="none", markersize=7, markerfacecolor=style["measured_marker_face"],
                markeredgecolor="#000000", markeredgewidth=0.6,
                label=f"{short_name} SPH ({centre['position'][0]:.4f}, {centre['position'][1]:.4f}) "
                      f"ψ {centre['stream_function']:+.4g}" + ("" if centre["found"] else " [not found]")))
            if centre["reference_position"] is not None:
                reference_label = (f"{short_name} Marchi 2021 ({centre['reference_position'][0]:.4f}, "
                                   f"{centre['reference_position'][1]:.4f}) ψ {centre['reference_stream_function']:+.4g}")
                vortex_handles.append(Line2D([], [], marker=marker, linestyle="none", markersize=8,
                                             markerfacecolor="none", markeredgecolor=GHIA_MARKER_EDGE,
                                             markeredgewidth=1.1, path_effects=GHIA_MARKER_HALO, label=reference_label))
        legend_title = ("streamlines = ψ contours at the 24 levels of Ghia et al. (1982);  vortex centres = ψ extrema:  "
                        "filled = SPH,  open = Marchi et al. 2021 Table 19 (primary only)")
        # "[not found]" labels (undeveloped flow) are longer: one column fewer keeps clear of the particle legend
        legend_columns = 4 if all(centre["found"] for centre in centres.values()) else 3
    else:
        primary = centres.get("primary")
        vortex_handles = [Line2D([], [], color="#000000", linewidth=0.8,
                                 label="ω_z contours at 0, ±0.5, ±1, ±2, ±3 "
                                       "(negative dashed)")]
        if primary is not None:
            vortex_handles.append(
                Line2D([], [], marker="o", linestyle="none", markersize=7, markerfacecolor=style["measured_marker_face"],
                       markeredgecolor="#000000", markeredgewidth=0.6,
                       label=f"primary centre SPH ({primary['position'][0]:.4f}, {primary['position'][1]:.4f}): "
                             f"ω_z = {primary['vorticity_z']:+.4f}"))
            if primary["reference_position"] is not None:
                vortex_handles.append(Line2D(
                    [], [], marker="o", linestyle="none", markersize=8, markerfacecolor="none",
                    markeredgecolor=GHIA_MARKER_EDGE, markeredgewidth=1.1, path_effects=GHIA_MARKER_HALO,
                    label=f"Marchi 2021 Table 19 ψ_min at ({primary['reference_position'][0]:.4f}, "
                          f"{primary['reference_position'][1]:.4f}) (no ω tabulated)"))
        legend_title = "ω_z = ∂v/∂x − ∂u/∂y  (negative in the clockwise primary vortex)"
        legend_columns = 1
    field_legend = figure.legend(handles=vortex_handles, loc="lower right",
                                 bbox_to_anchor=(1.0 - 0.12 / FIGURE_WIDTH, 0.03 / FIGURE_HEIGHT),
                                 ncol=legend_columns, fontsize=8, frameon=False, handlelength=1.6, columnspacing=1.2,
                                 labelspacing=0.45, title=legend_title, title_fontsize=8.5)
    field_legend._legend_box.align = "left"

    figure_path = output_directory / f"{job}_{field_name}.png"
    figure.savefig(figure_path, dpi=300, facecolor="#ffffff")
    plt.close(figure)
    log(f"  wrote {figure_path.name} and {full_path.name}")
    return [figure_path, full_path]


# ----------------------------------------------------------------------------------------------------------
# difference of two runs (dual minus single GPU)
# ----------------------------------------------------------------------------------------------------------

DIFFERENCE_SPEED_COLORMAP = "YlOrRd"      # sequential, light at zero; distinct from the gray / near-black bands
DIFFERENCE_VORTICITY_COLORMAP = "RdBu_r"


def load_grid_fields_for_difference(job_directory: pathlib.Path, grid_cells: int, cache_directory, log,
                                    snapshot_name: str = "final", forced_grid: UniformGrid | None = None) -> dict:
    data = load_job(job_directory, snapshot_name)
    grid_fields = interpolate_job_onto_grid(data, grid_cells, cache_directory, True, log, forced_grid)
    position = data["position"]
    material = data["material"]
    boundary_particles = material != MATERIAL_FLUID
    first_boundary_row = float(np.min(np.max(np.abs(position[boundary_particles]), axis=1)))
    summary = {"job": data["job"], "snapshot": snapshot_name, "time": float(data["time"]), "step": int(data["step"]),
               "k": int(data["meta"]["k"]), "re": int(data["meta"]["re"]),
               "converged": bool(data["result"].get("converged")), "setting": data["setting"],
               "smoothing_length": float(data["h"]),
               "effective_wall_half_extent": 0.5 * (FLUID_HALF_WIDTH + first_boundary_row),
               "slab_cuts": slab_cuts_for_job(data) if int(data["meta"].get("k", 1)) > 1 else []}
    grid_fields.pop("particle_tree", None)
    del data
    return {**grid_fields, **summary}


def compute_difference(campaign_directory: pathlib.Path, job_minuend: str, job_subtrahend: str, grid_cells: int,
                       cache_directory, log, snapshot_minuend: str = "final",
                       snapshot_subtrahend: str = "final", forced_grid: UniformGrid | None = None) -> dict:
    """Smooth grid fields of job_minuend minus job_subtrahend (same particle set, same grid: forced_grid, or the
    minuend's own grid)."""
    log(f"difference {job_minuend}/{snapshot_minuend} - {job_subtrahend}/{snapshot_subtrahend}")
    minuend = load_grid_fields_for_difference(campaign_directory / job_minuend, grid_cells, cache_directory, log,
                                              snapshot_minuend, forced_grid)
    subtrahend = load_grid_fields_for_difference(campaign_directory / job_subtrahend, grid_cells, cache_directory, log,
                                                 snapshot_subtrahend, minuend["grid"])
    grid = minuend["grid"]
    other_grid = subtrahend["grid"]
    if (abs(grid.spacing - other_grid.spacing) > 1e-15 or abs(grid.x_minimum - other_grid.x_minimum) > 1e-15
            or grid.column_count != other_grid.column_count):
        raise ValueError("the two jobs do not share the same grid (different boundary particle extent)")
    spacing = grid.spacing
    delta_velocity_x = minuend["velocity_x"] - subtrahend["velocity_x"]
    delta_velocity_y = minuend["velocity_y"] - subtrahend["velocity_y"]
    delta_vorticity = np.gradient(delta_velocity_y, spacing, axis=1) - np.gradient(delta_velocity_x, spacing, axis=0)
    delta_magnitude = np.hypot(delta_velocity_x, delta_velocity_y)
    fluid_both = (minuend["pixel_class"] == MATERIAL_FLUID) & (subtrahend["pixel_class"] == MATERIAL_FLUID)
    smoothing_length = minuend["smoothing_length"]
    absolute_x = np.abs(grid.x_centres)[None, :]
    absolute_y = np.abs(grid.y_centres)[:, None]
    interior = fluid_both & (absolute_x < FLUID_HALF_WIDTH - 2.0 * smoothing_length) \
        & (absolute_y < FLUID_HALF_WIDTH - 2.0 * smoothing_length)
    cavity_x = grid.x_centres + 0.5
    cavity_y = grid.y_centres + 0.5

    def statistics(mask):
        location = np.unravel_index(int(np.argmax(np.where(mask, delta_magnitude, -1.0))), mask.shape)
        return {"cells": int(np.count_nonzero(mask)),
                "max_abs_delta_u": float(np.max(np.abs(delta_velocity_x[mask]))),
                "rms_delta_u": float(np.sqrt(np.mean(delta_velocity_x[mask] ** 2))),
                "max_abs_delta_v": float(np.max(np.abs(delta_velocity_y[mask]))),
                "rms_delta_v": float(np.sqrt(np.mean(delta_velocity_y[mask] ** 2))),
                "max_delta_magnitude": float(delta_magnitude[location]),
                "max_delta_magnitude_at": [float(cavity_x[location[1]]), float(cavity_y[location[0]])],
                "rms_delta_magnitude": float(np.sqrt(np.mean(delta_magnitude[mask] ** 2))),
                "max_abs_delta_vorticity": float(np.max(np.abs(delta_vorticity[mask]))),
                "rms_delta_vorticity": float(np.sqrt(np.mean(delta_vorticity[mask] ** 2)))}

    # Column profiles (rms over the interior rows of each column) to test whether anything lines up with the cut.
    interior_rows = absolute_y[:, 0] < FLUID_HALF_WIDTH - 2.0 * smoothing_length
    interior_columns = absolute_x[0] < FLUID_HALF_WIDTH - 2.0 * smoothing_length
    with np.errstate(invalid="ignore"):
        column_mask = fluid_both[interior_rows]
        column_count = np.maximum(column_mask.sum(axis=0), 1)
        profile_magnitude = np.sqrt((np.where(column_mask, delta_magnitude[interior_rows], 0.0) ** 2).sum(axis=0)
                                    / column_count)
        profile_vorticity = np.sqrt((np.where(column_mask, delta_vorticity[interior_rows], 0.0) ** 2).sum(axis=0)
                                    / column_count)
    profile_magnitude[~interior_columns] = np.nan
    profile_vorticity[~interior_columns] = np.nan

    # Seam tests. (1) Jump of the difference between neighbouring grid columns, rms over the interior rows: a
    # slab-cut artefact would make the jump across the cut stand out among all interior column pairs.
    # (2) The same on the K = 2 field alone, with the second x-difference (a kink or step in u, v).
    jump = np.sqrt(np.mean(np.diff(delta_velocity_x[interior_rows], axis=1) ** 2
                           + np.diff(delta_velocity_y[interior_rows], axis=1) ** 2, axis=0))
    jump_positions = 0.5 * (cavity_x[1:] + cavity_x[:-1])
    jump_interior = np.abs(jump_positions - 0.5) < FLUID_HALF_WIDTH - 2.0 * smoothing_length
    second_difference = np.sqrt(np.mean(np.diff(minuend["velocity_x"][interior_rows], n=2, axis=1) ** 2
                                        + np.diff(minuend["velocity_y"][interior_rows], n=2, axis=1) ** 2, axis=0))
    second_difference_positions = cavity_x[1:-1]
    second_difference_interior = (np.abs(second_difference_positions - 0.5)
                                  < FLUID_HALF_WIDTH - 2.0 * smoothing_length)
    seam = []
    for cut in minuend["slab_cuts"]:
        cut_cavity_x = cut["x"] + 0.5
        distance_to_cut = np.abs(cavity_x - cut_cavity_x)
        at_cut = distance_to_cut <= smoothing_length
        reference_band = (distance_to_cut > 4.0 * smoothing_length) & (distance_to_cut < 0.2)
        jump_at_cut = float(jump[int(np.argmin(np.abs(jump_positions - cut_cavity_x)))])
        near_cut = np.abs(second_difference_positions - cut_cavity_x) <= 2.0 * spacing
        seam.append({"cut_x": cut_cavity_x,
                     "magnitude_at_cut_over_neighbourhood_median": float(np.nanmean(profile_magnitude[at_cut])
                                                                         / np.nanmedian(profile_magnitude[reference_band])),
                     "vorticity_at_cut_over_neighbourhood_median": float(np.nanmean(profile_vorticity[at_cut])
                                                                         / np.nanmedian(profile_vorticity[reference_band])),
                     "magnitude_profile_maximum_within_0.2_of_cut_at": float(cavity_x[distance_to_cut < 0.2][
                         int(np.nanargmax(profile_magnitude[distance_to_cut < 0.2]))]),
                     "vorticity_profile_maximum_within_0.2_of_cut_at": float(cavity_x[distance_to_cut < 0.2][
                         int(np.nanargmax(profile_vorticity[distance_to_cut < 0.2]))]),
                     "difference_column_jump_at_cut": jump_at_cut,
                     "difference_column_jump_interior_median": float(np.median(jump[jump_interior])),
                     "difference_column_jump_interior_percentile_99": float(np.percentile(jump[jump_interior], 99)),
                     "difference_column_jump_fraction_larger_than_at_cut": float(np.mean(jump[jump_interior]
                                                                                         > jump_at_cut)),
                     "dual_second_difference_near_cut_maximum": float(second_difference[near_cut].max()),
                     "dual_second_difference_interior_median": float(np.median(
                         second_difference[second_difference_interior])),
                     "dual_second_difference_interior_percentile_99": float(np.percentile(
                         second_difference[second_difference_interior], 99))})

    return {"minuend": {key: minuend[key] for key in ("job", "snapshot", "time", "step", "k", "re", "converged",
                                                      "slab_cuts", "setting")},
            "subtrahend": {key: subtrahend[key] for key in ("job", "snapshot", "time", "step", "k", "re", "converged",
                                                            "setting")},
            "grid": grid, "particle_spacing": minuend["particle_spacing"], "smoothing_length": smoothing_length,
            "effective_wall_half_extent": minuend["effective_wall_half_extent"],
            "pixel_class": minuend["pixel_class"], "delta_magnitude": delta_magnitude,
            "delta_vorticity": delta_vorticity, "profile_magnitude": profile_magnitude,
            "profile_vorticity": profile_vorticity,
            "statistics_all_fluid": statistics(fluid_both), "statistics_interior": statistics(interior),
            "seam": seam}


def context_lines(difference: dict) -> list[str]:
    """Interior rms |delta u| of this difference next to the same-step and same-run comparisons."""
    context = difference.get("context") or {}
    if not context:
        return []
    this = difference["statistics_interior"]["rms_delta_magnitude"]
    lines = ["interior rms |Δu vector| for comparison:",
             f"  this figure (end times {abs(difference['minuend']['time'] - difference['subtrahend']['time']):.2f} t* "
             f"apart): {this:.2e} U"]
    if "same_step" in context:
        same_step = context["same_step"]
        lines.append(f"  K = 2 − K = 1 at the same step {same_step['minuend']['step']:,} (t* "
                     f"{same_step['minuend']['time']:.2f}): {same_step['statistics_interior']['rms_delta_magnitude']:.2e} U")
    if "same_run" in context:
        same_run = context["same_run"]
        lines.append(f"  K = 2 final − K = 2 at t* {same_run['subtrahend']['time']:.2f} (same run, "
                     f"{same_run['minuend']['time'] - same_run['subtrahend']['time']:.2f} t* apart): "
                     f"{same_run['statistics_interior']['rms_delta_magnitude']:.2e} U")
    return lines


def seam_detected(seam: dict) -> bool:
    """A slab-cut feature: the jump across the cut, or the K = 2 kink at the cut, above the interior 99th percentile."""
    return (seam["difference_column_jump_at_cut"] > seam["difference_column_jump_interior_percentile_99"]
            or seam["dual_second_difference_near_cut_maximum"] > seam["dual_second_difference_interior_percentile_99"])


def render_difference_figure(difference: dict, output_directory: pathlib.Path, log) -> pathlib.Path:
    minuend = difference["minuend"]
    subtrahend = difference["subtrahend"]
    grid = difference["grid"]
    pixel_class = difference["pixel_class"]
    interior_statistics = difference["statistics_interior"]
    all_statistics = difference["statistics_all_fluid"]
    fluid = pixel_class == MATERIAL_FLUID
    absolute_x = np.abs(grid.x_centres)[None, :]
    absolute_y = np.abs(grid.y_centres)[:, None]
    interior = fluid & (absolute_x < FLUID_HALF_WIDTH - 2.0 * difference["smoothing_length"]) \
        & (absolute_y < FLUID_HALF_WIDTH - 2.0 * difference["smoothing_length"])
    magnitude_top = nice_ceiling(float(np.percentile(difference["delta_magnitude"][interior], 99.5)))
    vorticity_top = nice_ceiling(float(np.percentile(np.abs(difference["delta_vorticity"][interior]), 99.5)))
    panels = [
        ("delta_magnitude", DIFFERENCE_SPEED_COLORMAP, Normalize(0.0, magnitude_top), "max",
         f"|Δu| = |u_A − u_B| / U   (colour 0 – {magnitude_top:g}, linear)"),
        ("delta_vorticity", DIFFERENCE_VORTICITY_COLORMAP, Normalize(-vorticity_top, vorticity_top), "both",
         f"Δω_z = ω_z,A − ω_z,B  [U/L]   (colour clipped at ±{vorticity_top:g})"),
    ]
    figure_width, figure_height = 18.0, 10.6
    map_size, map_bottom = 6.4, 3.1
    map_lefts = (0.85, 9.25)
    figure = plt.figure(figsize=(figure_width, figure_height))
    figure.patch.set_facecolor("#ffffff")

    def inches(left, bottom, width, height):
        return figure.add_axes([left / figure_width, bottom / figure_height, width / figure_width,
                                height / figure_height])

    effective_wall = difference["effective_wall_half_extent"]
    for (field_key, colormap_name, normalization, extend, label), map_left in zip(panels, map_lefts):
        axes = inches(map_left, map_bottom, map_size, map_size)
        colormap = plt.get_cmap(colormap_name)
        rgba = colorize(difference[field_key], pixel_class, colormap, normalization)
        axes.imshow(rgba, origin="lower", extent=grid.extent_cavity, interpolation="antialiased",
                    interpolation_stage="rgba")
        del rgba
        axes.add_patch(Rectangle((0.5 - effective_wall, 0.5 - effective_wall), 2.0 * effective_wall,
                                 2.0 * effective_wall, fill=False, edgecolor=INTERFACE_COLOR, linewidth=0.6, zorder=4))
        for cut in minuend["slab_cuts"]:
            axes.axvline(cut["x"] + 0.5, color="#1f4e79", linewidth=0.9, linestyle=(0, (4, 3)), zorder=5)
            axes.text(cut["x"] + 0.5, grid.extent_cavity[3] + 0.006, f"K = 2 slab cut  X = {cut['x'] + 0.5:.4f}",
                      color="#1f4e79", fontsize=8, ha="center", va="bottom", clip_on=False)
        extent = grid.extent_cavity
        axes.set_xlim(extent[0], extent[1])
        axes.set_ylim(extent[2], extent[3])
        axes.set_aspect("equal")
        axes.set_xticks(np.linspace(0.0, 1.0, 11))
        axes.set_yticks(np.linspace(0.0, 1.0, 11))
        axes.tick_params(labelsize=9, colors=TEXT_COLOR)
        axes.set_xlabel("X = x/L + 1/2", fontsize=10, color=TEXT_COLOR)
        axes.set_ylabel("Y = y/L + 1/2", fontsize=10, color=TEXT_COLOR)
        colorbar_axes = inches(map_left + map_size + 0.12, map_bottom, 0.17, map_size)
        colorbar = figure.colorbar(ScalarMappable(norm=normalization, cmap=colormap), cax=colorbar_axes,
                                   extend=extend)
        colorbar.set_label(label, fontsize=10, color=TEXT_COLOR)
        colorbar.ax.tick_params(labelsize=9)
        colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))

    profile_axes = inches(map_lefts[0], 0.55, 11.7, 1.45)
    cavity_x = grid.x_centres + 0.5
    profile_axes.plot(cavity_x, difference["profile_magnitude"], color="#d95f02", linewidth=0.9,
                      label="rms over Y of |Δu| (left axis)")
    profile_axes.set_ylabel("rms |Δu| / U", fontsize=9, color="#d95f02")
    profile_axes.tick_params(axis="y", labelsize=8, colors="#d95f02")
    profile_axes.tick_params(axis="x", labelsize=8, colors=TEXT_COLOR)
    profile_axes.set_ylim(0.0, None)
    vorticity_axes = profile_axes.twinx()
    vorticity_axes.plot(cavity_x, difference["profile_vorticity"], color="#1b6ca8", linewidth=0.9,
                        label="rms over Y of Δω_z (right axis)")
    vorticity_axes.set_ylabel("rms Δω_z [U/L]", fontsize=9, color="#1b6ca8")
    vorticity_axes.tick_params(axis="y", labelsize=8, colors="#1b6ca8")
    vorticity_axes.set_ylim(0.0, None)
    for cut in minuend["slab_cuts"]:
        profile_axes.axvline(cut["x"] + 0.5, color="#1f4e79", linewidth=0.9, linestyle=(0, (4, 3)))
    profile_axes.set_xlim(grid.extent_cavity[0], grid.extent_cavity[1])
    profile_axes.set_xticks(np.linspace(0.0, 1.0, 11))
    profile_axes.set_xlabel("X = x/L + 1/2   (interior rows and columns only: > 2h from every wall)", fontsize=9,
                            color=TEXT_COLOR)
    handles = profile_axes.get_lines()[:1] + vorticity_axes.get_lines()[:1]
    if minuend["slab_cuts"]:
        handles = handles + [Line2D([], [], color="#1f4e79", linewidth=0.9, linestyle=(0, (4, 3)),
                                    label="K = 2 slab cut")]
    profile_axes.legend(handles=handles, loc="upper left", fontsize=8, frameon=True, framealpha=0.85, ncol=3)

    seam_lines = []
    for seam in difference["seam"]:
        seam_lines += [
            f"slab cut X = {seam['cut_x']:.4f}: column-to-column jump of (Δu, Δv), rms over Y",
            f"  across the cut {seam['difference_column_jump_at_cut']:.1e}  vs  interior median "
            f"{seam['difference_column_jump_interior_median']:.1e}, p99 "
            f"{seam['difference_column_jump_interior_percentile_99']:.1e}",
            f"  ({100 * seam['difference_column_jump_fraction_larger_than_at_cut']:.0f} % of the column pairs jump more)",
            f"run A alone, 2nd x-difference of (u, v) at the cut "
            f"{seam['dual_second_difference_near_cut_maximum']:.1e}",
            f"  vs interior median {seam['dual_second_difference_interior_median']:.1e}, p99 "
            f"{seam['dual_second_difference_interior_percentile_99']:.1e}  ->  "
            + ("NO step or kink at the cut" if not seam_detected(seam) else "FEATURE AT THE CUT")]
    statistics_text = "\n".join([
        "interior fluid (> 2h from every wall): max / rms",
        f"  |Δu| {interior_statistics['max_abs_delta_u']:.2e} / {interior_statistics['rms_delta_u']:.2e} U,   "
        f"|Δv| {interior_statistics['max_abs_delta_v']:.2e} / {interior_statistics['rms_delta_v']:.2e} U",
        f"  |Δω_z| {interior_statistics['max_abs_delta_vorticity']:.2e} / "
        f"{interior_statistics['rms_delta_vorticity']:.2e} U/L",
        "all fluid cells (incl. the singular lid corners): max / rms",
        f"  |Δu| {all_statistics['max_abs_delta_u']:.2e} / {all_statistics['rms_delta_u']:.2e} U,   "
        f"|Δv| {all_statistics['max_abs_delta_v']:.2e} / {all_statistics['rms_delta_v']:.2e} U",
        f"  largest |Δu vector| at (X, Y) = ({all_statistics['max_delta_magnitude_at'][0]:.3f}, "
        f"{all_statistics['max_delta_magnitude_at'][1]:.3f})",
    ] + seam_lines + context_lines(difference))
    figure.text(13.35 / figure_width, 2.62 / figure_height, statistics_text,
                ha="left", va="top", fontsize=7.4, color=TEXT_COLOR, linespacing=1.3)

    title = (f"Lid-driven cavity Re {minuend['re']}:  A = {minuend['job']} ({minuend['setting']})  minus  "
             f"B = {subtrahend['job']} ({subtrahend['setting']})")
    subtitle = "\n".join([
        f"A at t* = {minuend['time']:.2f} ({'steady' if minuend['converged'] else 'NOT steady'}), B at t* = "
        f"{subtrahend['time']:.2f} ({'steady' if subtrahend['converged'] else 'NOT steady'}); both K = "
        f"{minuend['k']} / {subtrahend['k']}, newest checkpoint of each run",
        f"both fields: Shepard interpolation of all particles on the same {grid.column_count}² grid; gray / near-black = "
        "wall / lid band; white line = effective wall; difference shown on cells that are fluid in both runs",
        "dashed line = slab cut of run A (a slab-cut artefact would show as a line along it and a spike in the profiles)",
    ])
    figure.text(0.5, 1.0 - 0.14 / figure_height, title, ha="center", va="top", fontsize=13, color=TEXT_COLOR,
                fontweight="bold")
    figure.text(0.5, 1.0 - 0.45 / figure_height, subtitle, ha="center", va="top", fontsize=8.5,
                color=MUTED_TEXT_COLOR, linespacing=1.35)
    figure_path = output_directory / f"{minuend['job']}_minus_{subtrahend['job']}.png"
    figure.savefig(figure_path, dpi=300, facecolor="#ffffff")
    plt.close(figure)
    log(f"  wrote {figure_path.name}")
    return figure_path


def difference_summary(difference: dict) -> dict:
    return {"minuend": difference["minuend"], "subtrahend": difference["subtrahend"],
            "grid_cells": difference["grid"].column_count,
            "statistics_all_fluid": difference["statistics_all_fluid"],
            "statistics_interior": difference["statistics_interior"], "seam": difference["seam"]}


# ----------------------------------------------------------------------------------------------------------
# sanity numbers
# ----------------------------------------------------------------------------------------------------------

def sanity_numbers(data: dict, fields: dict, zooms: dict | None = None) -> dict:
    position = data["position"]
    velocity = data["velocity"]
    material = data["material"]
    density = data["density"]
    particle_spacing = fields["particle_spacing"]
    fluid = material == MATERIAL_FLUID
    boundary = ~fluid

    counts = {MATERIAL_NAMES[value]: int(np.count_nonzero(material == value)) for value in MATERIAL_NAMES}
    fluid_position = position[fluid]
    fluid_extent = np.max(np.abs(fluid_position), axis=1)
    band_threshold = FLUID_HALF_WIDTH + 0.5 * particle_spacing
    in_band = fluid_extent > band_threshold
    boundary_tree = cKDTree(position[boundary])
    fluid_to_boundary_distance, _ = boundary_tree.query(fluid_position, k=1, workers=4)
    closest = int(np.argmin(fluid_to_boundary_distance))

    fluid_pixels = fields["pixel_class"] == MATERIAL_FLUID
    raw_speed = np.hypot(velocity[fluid, 0], velocity[fluid, 1])
    density_by_material = {MATERIAL_NAMES[value]: [float(density[material == value].min()),
                                                   float(density[material == value].max())]
                           for value in MATERIAL_NAMES}
    # Fluid density: percentiles, the two extremes with their positions, and the linearised pressure
    # coefficient p* = c0^2 (rho - rho0) / (rho0 U^2) (rho0 = rest density of the static wall particles).
    reference_density = float(np.median(density[material == MATERIAL_WALL]))
    speed_of_sound = data["case_information"]["speed_of_sound"]
    fluid_density = density[fluid]
    order = np.argsort(fluid_density)

    def pressure_coefficient(value):
        return speed_of_sound ** 2 * (value - reference_density) / reference_density

    density_extremes = {}
    for label, rank in (("lowest", order[0]), ("second lowest", order[1]), ("second highest", order[-2]),
                        ("highest", order[-1])):
        density_extremes[label] = {"density": float(fluid_density[rank]),
                                   "pressure_coefficient": float(pressure_coefficient(fluid_density[rank])),
                                   "at": [float(fluid_position[rank, 0] + 0.5), float(fluid_position[rank, 1] + 0.5)]}
    density_percentiles = {f"{percent:g}": float(np.percentile(fluid_density, percent))
                           for percent in (0.01, 1.0, 50.0, 99.0, 99.99)}
    zoom_summary = {}
    for corner_name, zoom in (zooms or {}).items():
        zoom_summary[corner_name] = {
            "fluid_particles": int(zoom["fluid_indices"].size),
            "speed_range": [float(zoom["fluid_speed"].min()), float(zoom["fluid_speed"].max())],
            "vorticity_range": [float(zoom["fluid_vorticity"].min()), float(zoom["fluid_vorticity"].max())],
            "velocity_colour": zoom_color_scale("velocity", zoom)["description"],
            "vorticity_colour": zoom_color_scale("vorticity", zoom)["description"],
            "vorticity_check": zoom["vorticity_check"]}
    nan_counts = {"position": int(np.count_nonzero(~np.isfinite(position))),
                  "velocity": int(np.count_nonzero(~np.isfinite(velocity))),
                  "density": int(np.count_nonzero(~np.isfinite(density))),
                  "grid_cells_without_particles": fields["weight_sum_zero_cells"],
                  "grid_speed_in_fluid": int(np.count_nonzero(~np.isfinite(fields["speed"][fluid_pixels]))),
                  "grid_vorticity_in_fluid": int(np.count_nonzero(~np.isfinite(fields["vorticity"][fluid_pixels])))}

    centres = {}
    for stream_function_name, centre_set in fields["vortex_centres_by_stream_function"].items():
        centres[stream_function_name] = {}
        for vortex_name, centre in centre_set.items():
            entry = dict(centre)
            entry["distance_to_reference"] = (float(math.dist(centre["position"], centre["reference_position"]))
                                              if centre["reference_position"] is not None else None)
            centres[stream_function_name][vortex_name] = entry

    return {"job": data["job"], "reynolds_number": fields["reynolds_number"], "k": int(data["meta"]["k"]),
            "time": float(data["time"]), "step": int(data["step"]), "particle_count": int(position.shape[0]),
            "counts": counts,
            # the run's own initial counts (meta.json), so any case size is checked, not only 2M
            "expected_counts": [int(data["meta"].get("fluid_count_initial", -1)),
                                int(data["meta"].get("expected_total", -1))],
            "expected_counts_match": (counts["fluid"] == int(data["meta"].get("fluid_count_initial", -1))
                                      and sum(counts.values()) == int(data["meta"].get("expected_total", -1))),
            "particle_spacing": particle_spacing, "smoothing_length": fields["smoothing_length"],
            "fluid_in_band": {"threshold": band_threshold, "count": int(np.count_nonzero(in_band)),
                              "deepest_penetration_dx": (float((fluid_extent[in_band].max() - FLUID_HALF_WIDTH)
                                                               / particle_spacing) if in_band.any() else 0.0),
                              "outermost_fluid_extent_dx": float((fluid_extent.max() - FLUID_HALF_WIDTH)
                                                                 / particle_spacing)},
            "smallest_fluid_to_boundary_distance_dx": float(fluid_to_boundary_distance[closest] / particle_spacing),
            "smallest_fluid_to_boundary_at": [float(fluid_position[closest, 0] + 0.5),
                                              float(fluid_position[closest, 1] + 0.5)],
            "fluid_to_boundary_distance_below_half_dx": int(np.count_nonzero(fluid_to_boundary_distance
                                                                             < 0.5 * particle_spacing)),
            "density_by_material": density_by_material,
            "reference_density": reference_density, "speed_of_sound": speed_of_sound,
            "density_percentiles_fluid": density_percentiles, "density_extremes_fluid": density_extremes,
            "particle_volume_over_spacing_squared": float(np.mean(data["mass"][fluid] / density[fluid])
                                                          / particle_spacing ** 2),
            "meta_dx": float(data["meta"]["dx"]),
            "mean_fluid_velocity": fields["mean_fluid_velocity"],
            "max_speed_fluid_raw": float(raw_speed.max()),
            "max_speed_fluid_grid": float(np.nanmax(fields["speed"][fluid_pixels])),
            "nan_counts": nan_counts,
            "vortex_centres_by_stream_function": centres, "plotted_stream_function": fields["plotted_stream_function"],
            "setting": data["setting"], "steady_time": data["result"].get("steady_time"),
            "stream_function_diagnostics": fields["stream_function_diagnostics"],
            "slab_cuts": fields["slab_cuts"], "slab_cut_x": fields["slab_cut_x"], "validation": fields["validation"],
            "zooms": zoom_summary, "case": data["case_information"]["case_path"],
            "converged": bool(data["result"].get("converged")),
            "rel_l2_marchi2021": [data["result"].get("final", {}).get("rel_l2_marchi2021_u"),
                                  data["result"].get("final", {}).get("rel_l2_marchi2021_v")],
            "grid_cells": fields["grid"].column_count, "grid_spacing_dx": fields["grid"].spacing / particle_spacing}


def format_position(position) -> str:
    return f"({position[0]:.4f}, {position[1]:.4f})"


def relative_difference(value, reference) -> str:
    """Relative difference of the magnitudes (negative = SPH weaker than the reference)."""
    if value is None or reference is None or reference == 0:
        return "-"
    return f"{100.0 * (abs(value) - abs(reference)) / abs(reference):+.1f} %"


def job_markdown(check: dict) -> list[str]:
    counts = check["counts"]
    band = check["fluid_in_band"]
    validation = check.get("validation") or {}
    spacing = check["particle_spacing"]
    rel_l2_u, rel_l2_v = check.get("rel_l2_marchi2021") or [None, None]
    lines = [f"## {check['job']}  ({check.get('setting', '')}, K = {check['k']}, t* = {check['time']:.3f}, "
             f"step {check['step']:,}, " + (f"steady from t* = {check['steady_time']:.1f})" if check.get("steady_time")
                                             else "NOT steady)"), "",
             f"Case `{check.get('case', '?')}`. Figures: `{check['job']}_velocity.png`, `{check['job']}_vorticity.png`, "
             f"`{check['job']}_velocity_full.png`, `{check['job']}_vorticity_full.png`."
             + (f" rel-L2 vs Marchi 2021 (time average t* 80-100, `docs/validation/data/profile_{check['job']}.csv`): "
                f"u(Y) {100 * rel_l2_u:.2f} %, v(X) {100 * rel_l2_v:.2f} %."
                if rel_l2_u is not None else ""), ""]
    lines += ["| quantity | value |", "|---|---|",
              f"| particles total | {check['particle_count']:,} |",
              f"| fluid / wall / lid | {counts['fluid']:,} / {counts['wall']:,} / {counts['lid']:,} "
              + (f"({'match' if check['expected_counts_match'] else 'DIFFER from'} the initial fluid / total "
               f"{check['expected_counts'][0]:,} / {check['expected_counts'][1]:,} in meta.json) |"
               if "expected_counts" in check else
               f"({'match' if check['expected_counts_match'] else 'DIFFER from'} 2,002,225 / 47,179 / 15,565) |"),
              f"| dx (measured: median nearest-neighbour distance of the wall lattice), h | {spacing:.8f} m "
              f"(meta.json dx = {check.get('meta_dx', float('nan')):.6f} is 2 x particle_radius rounded), "
              f"{check['smoothing_length']:.6g} m (h = {check['smoothing_length'] / spacing:.3f} dx) |",
              f"| particle volume m/rho (fluid mean) | {check.get('particle_volume_over_spacing_squared', float('nan')):.4f}"
              " dx^2 (solver calibrates V = 1 / sum_{j != i} W, self excluded; see notes) |",
              f"| fluid particles inside the boundary band (max(\\|x\\|,\\|y\\|) > 0.5 + dx/2) | {band['count']} "
              + (f"(deepest {band['deepest_penetration_dx']:.3f} dx beyond the nominal edge) |" if band['count']
                 else "(none beyond 0.5 + dx/2) |"),
              f"| outermost fluid particle | {band['outermost_fluid_extent_dx']:+.3f} dx from the nominal edge "
              "(first boundary row sits at +1 dx) |",
              f"| smallest fluid-to-boundary particle distance | {check['smallest_fluid_to_boundary_distance_dx']:.3f}"
              f" dx at (X, Y) = {format_position(check['smallest_fluid_to_boundary_at'])}; pairs closer than 0.5 dx: "
              f"{check['fluid_to_boundary_distance_below_half_dx']} |"]
    percentiles = check.get("density_percentiles_fluid")
    if percentiles:
        lines.append("| fluid density percentiles 0.01 / 1 / 50 / 99 / 99.99 % | "
                     + " / ".join(f"{percentiles[key]:.3f}" for key in ("0.01", "1", "50", "99", "99.99"))
                     + f" kg/m^3 (rho0 = {check['reference_density']:.1f}, c0 = {check['speed_of_sound']:g} m/s; "
                     f"p* = c0^2 (rho - rho0) / (rho0 U^2) = "
                     f"{check['speed_of_sound'] ** 2 * (percentiles['0.01'] - check['reference_density']) / check['reference_density']:+.2f}"
                     f" / {check['speed_of_sound'] ** 2 * (percentiles['99.99'] - check['reference_density']) / check['reference_density']:+.2f}"
                     " at 0.01 / 99.99 %) |")
        extremes = check["density_extremes_fluid"]
        lines.append("| fluid density extremes (single particles) | "
                     + "; ".join(f"{label} {entry['density']:.3f} (p* {entry['pressure_coefficient']:+.1f}) at "
                                 f"{format_position(entry['at'])}" for label, entry in extremes.items())
                     + " - the min / max are the lid-corner singularity particles |")
    for material_name, (density_minimum, density_maximum) in check["density_by_material"].items():
        if material_name != "fluid":
            lines.append(f"| density {material_name} min / max | {density_minimum:.3f} / {density_maximum:.3f} kg/m^3 |")
    if check.get("mean_fluid_velocity"):
        mean_u, mean_v = check["mean_fluid_velocity"]
        lines.append(f"| mass-weighted mean fluid velocity (u, v) | ({mean_u:+.3e}, {mean_v:+.3e}) U - a steady net "
                     "flux of the stored particle velocities (zero in a closed box if the particles moved with it) |")
    lines += [f"| max speed, fluid particles (raw) | {check['max_speed_fluid_raw']:.4f} U |",
              f"| max speed, smooth field on fluid cells | {check['max_speed_fluid_grid']:.4f} U |",
              "| NaN / non-finite | " + ", ".join(f"{name} {value}" for name, value in check["nan_counts"].items())
              + " |"]
    diagnostics = check["stream_function_diagnostics"]
    lines += [f"| effective wall (half-way fluid edge / first boundary row) | \\|x\\|, \\|y\\| = "
              f"{diagnostics['effective_wall']:.6f} m (X, Y = {0.5 - diagnostics['effective_wall']:+.6f} / "
              f"{0.5 + diagnostics['effective_wall']:.6f}) |",
              f"| psi = integral u dy from the bottom wall, sampled ON the effective walls (ideally 0; corners "
              f"within 2h left out): max \\|psi\\| left / right / lid, mean on the lid | "
              f"{diagnostics['velocity_integral_on_left_wall_maximum_absolute']:.2e} / "
              f"{diagnostics['velocity_integral_on_right_wall_maximum_absolute']:.2e} / "
              f"{diagnostics['velocity_integral_on_lid_wall_maximum_absolute']:.2e}, "
              f"{diagnostics['velocity_integral_on_lid_wall_mean']:+.2e} U L (the lid mean equals the net flux: "
              "mean fluid u x 1 L^2) |",
              f"| max \\|psi_integral - psi_poisson\\| (fluid, > 2h from the walls) | "
              f"{diagnostics['velocity_integral_minus_poisson_maximum_absolute_interior']:.2e} U L |",
              f"| \\|u - curl psi_poisson\\| (non-solenoidal part, > 0.05 L from the walls): median / p99 / max | "
              f"{diagnostics['poisson_velocity_mismatch_median']:.2e} / "
              f"{diagnostics['poisson_velocity_mismatch_percentile_99']:.2e} / "
              f"{diagnostics['poisson_velocity_mismatch_maximum']:.2e} U |",
              f"| divergence of the smooth field (> 0.05 L from the walls): rms / max | "
              f"{diagnostics['divergence_root_mean_square']:.2e} / {diagnostics['divergence_maximum_absolute']:.2e}"
              " U/L |",
              f"| stream function drawn in the velocity figure | {check['plotted_stream_function']} |"]
    for cut in check.get("slab_cuts") or []:
        ownership = ("particle ownership not recorded in the checkpoint" if cut["ownership_mismatches"] is None else
                     f"particles owned by slab 0 reach x = {cut['owned_left_maximum_x']:.6f}, slab 1 starts at "
                     f"{cut['owned_right_minimum_x']:.6f}; ownership vs column mismatches: {cut['ownership_mismatches']}")
        lines.append(f"| K = 2 slab cut | global voxel column {cut['column']} -> x = origin + {cut['column']} h = "
                     f"{cut['x']:.6f} m, X = {cut['x'] + 0.5:.6f} (origin = frame bbox min - h/2); {ownership} |")
    if validation:
        lines.append(f"| scatter vs `cavity_sampling.interpolate` Shepard ({validation['points']} grid points) | max "
                     f"\\|du\\| {validation['max_abs_difference_u']:.2e}, max \\|dv\\| "
                     f"{validation['max_abs_difference_v']:.2e}, NaN mismatches {validation['nan_mismatch']} |")
    lines.append(f"| grid | {check['grid_cells']}^2 cells, spacing {check['grid_spacing_dx']:.3f} dx |")
    if "runtime_seconds" in check:
        lines.append(f"| runtime / peak memory (this job) | {check['runtime_seconds']:.0f} s / "
                     f"{check['peak_memory_bytes'] / 2 ** 30:.2f} GiB (process peak working set) |")

    if check.get("zooms"):
        lines += ["", "Corner zooms (raw values of the fluid particles in each window; every zoom has its own colour "
                  "scale; the particle vorticity is the renormalised estimate):", "",
                  "| zoom | fluid particles | speed min / max [U] | velocity colour | omega_z min / max [U/L] | "
                  "vorticity colour | plain-sum / renormalised omega (median) | particle vs smooth omega_z, "
                  "particles > h from the band: median \\|diff\\| / median \\|smooth\\| |",
                  "|---|---|---|---|---|---|---|---|"]
        for corner_name, zoom in check["zooms"].items():
            vorticity_check = zoom["vorticity_check"]
            comparison = (f"{vorticity_check['median_absolute_difference']:.3g} / "
                          f"{vorticity_check['median_absolute_smooth']:.3g} ({vorticity_check['particles']} particles)"
                          if vorticity_check.get("particles") else "-")
            lines.append(f"| {corner_name} | {zoom['fluid_particles']} | {zoom['speed_range'][0]:.2e} / "
                         f"{zoom['speed_range'][1]:.4f} | {zoom['velocity_colour']} | "
                         f"{zoom['vorticity_range'][0]:+.3g} / {zoom['vorticity_range'][1]:+.3g} | "
                         f"{zoom['vorticity_colour']} | {vorticity_check['plain_sum_over_renormalised_median']:.4f} | "
                         f"{comparison} |")

    lines += ["", "Vortex centres (psi extrema in the fluid > 2h from the walls, quadratic sub-cell refinement; "
              "omega_z = smooth-field vorticity interpolated at the centre); reference: Marchi et al. 2021 Table 19 "
              "(primary vortex only):", "",
              "| vortex | psi definition | SPH (X, Y) | SPH psi | SPH omega_z | reference (X, Y) | reference psi | "
              "SPH \\|psi\\| vs reference | distance to reference |", "|---|---|---|---|---|---|---|---|---|"]
    centre_sets = check["vortex_centres_by_stream_function"]
    for vortex_name in centre_sets[check["plotted_stream_function"]]:
        for stream_function_name in ("poisson", "velocity-integral"):
            centre = centre_sets[stream_function_name][vortex_name]
            plotted_mark = " (plotted)" if stream_function_name == check["plotted_stream_function"] else ""
            has_reference = centre.get("reference_position") is not None
            lines.append(
                f"| {vortex_name}{'' if centre['found'] else ' (NOT FOUND)'} | {stream_function_name}{plotted_mark} "
                f"| {format_position(centre['position'])} | {centre['stream_function']:+.4e} "
                f"| {centre['vorticity_z']:+.4f} | "
                + (f"{format_position(centre['reference_position'])} | {centre['reference_stream_function']:+.6e} | "
                   f"{relative_difference(centre['stream_function'], centre['reference_stream_function'])} | "
                   f"{centre['distance_to_reference']:.4f} L |" if has_reference else "- | - | - | - |"))
    lines.append("")
    return lines


def difference_markdown(difference: dict) -> list[str]:
    minuend = difference["minuend"]
    subtrahend = difference["subtrahend"]
    lines = [f"## Difference {minuend['job']} minus {subtrahend['job']}", "",
             f"Figure `{minuend['job']}_minus_{subtrahend['job']}.png`: A = {minuend['setting']} at t* = "
             f"{minuend['time']:.3f}, B = {subtrahend['setting']} at t* = {subtrahend['time']:.3f} (newest checkpoint "
             f"of each run, both after the steady criterion); both smooth fields on the same "
             f"{difference['grid_cells']}^2 grid.", "",
             "| region | max \\|du\\| | rms du | max \\|dv\\| | rms dv | max \\|du vector\\| (at X, Y) | max \\|d omega_z\\| | "
             "rms d omega_z |", "|---|---|---|---|---|---|---|---|"]
    for label, key in (("interior fluid (> 2h from every wall)", "statistics_interior"),
                       ("all fluid cells", "statistics_all_fluid")):
        statistics = difference[key]
        lines.append(f"| {label} | {statistics['max_abs_delta_u']:.2e} | {statistics['rms_delta_u']:.2e} | "
                     f"{statistics['max_abs_delta_v']:.2e} | {statistics['rms_delta_v']:.2e} | "
                     f"{statistics['max_delta_magnitude']:.2e} {format_position(statistics['max_delta_magnitude_at'])} | "
                     f"{statistics['max_abs_delta_vorticity']:.2e} | {statistics['rms_delta_vorticity']:.2e} |")
    lines.append("")
    for seam in difference["seam"]:
        verdict = "a feature at the cut" if seam_detected(seam) else "nothing lines up with the cut"
        lines += [f"Slab cut at X = {seam['cut_x']:.6f} (meta cuts x h + grid origin, matches particle ownership): "
                  f"**{verdict}**.", "",
                  f"- Column-to-column jump of the difference (du, dv), rms over the interior rows: "
                  f"{seam['difference_column_jump_at_cut']:.2e} across the cut vs interior median "
                  f"{seam['difference_column_jump_interior_median']:.2e} and p99 "
                  f"{seam['difference_column_jump_interior_percentile_99']:.2e}; "
                  f"{100 * seam['difference_column_jump_fraction_larger_than_at_cut']:.0f} % of all interior column "
                  "pairs jump more than the pair across the cut.",
                  f"- The K = 2 field alone, rms second x-difference of (u, v) within 2 grid cells of the cut: "
                  f"{seam['dual_second_difference_near_cut_maximum']:.2e} vs interior median "
                  f"{seam['dual_second_difference_interior_median']:.2e} and p99 "
                  f"{seam['dual_second_difference_interior_percentile_99']:.2e} (no step or kink).",
                  f"- Column rms of \\|du\\| and d omega_z within +-h of the cut divided by their median over 4h-0.2 L "
                  f"on both sides: {seam['magnitude_at_cut_over_neighbourhood_median']:.2f} and "
                  f"{seam['vorticity_at_cut_over_neighbourhood_median']:.2f}. This ratio follows the smooth large-scale "
                  "variation of the difference: the largest column rms within 0.2 L of the cut is at X = "
                  f"{seam['magnitude_profile_maximum_within_0.2_of_cut_at']:.4f} (\\|du\\|) and "
                  f"{seam['vorticity_profile_maximum_within_0.2_of_cut_at']:.4f} (d omega_z), not at the cut.",
                  "- The difference is a smooth large-scale pattern, largest in the lid and wall shear layers."]
    context = difference.get("context") or {}
    if context:
        lines += ["", "Same quantities for comparison (interior = fluid > 2h from every wall):", "",
                  "| comparison | t* | interior rms \\|du vector\\| | interior max \\|du vector\\| | interior rms d omega_z | "
                  "all-fluid max \\|du\\| / \\|dv\\| |", "|---|---|---|---|---|---|"]
        rows = [("this figure: K = 2 final minus K = 1 final", difference)]
        if "same_step" in context:
            rows.append((f"K = 2 minus K = 1 at the common checkpoint step {context['same_step']['minuend']['step']:,}",
                         context["same_step"]))
        if "same_run" in context:
            rows.append(("K = 2 final minus K = 2 at that checkpoint (same run)", context["same_run"]))
        for label, entry in rows:
            interior = entry["statistics_interior"]
            everything = entry["statistics_all_fluid"]
            lines.append(f"| {label} | {entry['minuend']['time']:.2f} vs {entry['subtrahend']['time']:.2f} | "
                         f"{interior['rms_delta_magnitude']:.2e} | {interior['max_delta_magnitude']:.2e} | "
                         f"{interior['rms_delta_vorticity']:.2e} | {everything['max_abs_delta_u']:.2e} / "
                         f"{everything['max_abs_delta_v']:.2e} |")
    lines.append("")
    return lines


NOTES_MARKDOWN = [
    "## Notes", "",
    "- **Particle volume.** m = rho0 V with V = 1 / sum_{j != i} W = 1.1293 dx^2 (self excluded, "
    "`case_loader_v6._calibrate_particle_volume`), so sum_j V_j (x_j - x_i) (x) grad W = 1.129 I in the bulk. "
    "Shepard-normalised quantities (all smooth fields, the vortex centres) are unaffected; the zoom particle vorticity "
    "uses the renormalised gradient B^-T sum_j V_j (f_j - f_i) grad W ('plain-sum / renormalised' column above).",
    "- **Numerics settings.** The release cases (xi = 0.1, eps^2 = 0.01 h^2) give a viscous operator of 0.85 nu "
    "(Re_eff ~ 1170); xi = 0.001 gives 0.93 nu, xi = 0.001 with eps^2 = 0.0025 h^2 gives 0.98 nu "
    "(`docs/validation/cavity_re1000.md`, effective viscosity).",
    "- **Net flux of the stored velocities.** The mass-weighted mean fluid velocity is listed per run; a non-zero "
    "mean u cannot be carried by the stored velocities in a closed box, so the figures draw the Poisson stream "
    "function, and both stream functions' vortex centres are reported.",
    "- **Plot choices.** Each zoom has its own colour scale (the corner values span five decades); the lid-corner "
    "vorticity zooms use a symmetric-log scale. The omega_z contour lines trace a Gaussian-smoothed copy; the colours "
    "are unsmoothed. Psi is not contoured within 1 dx of the effective wall.", ""]


def write_check_markdown(output_directory: pathlib.Path) -> pathlib.Path:
    lines = ["# Cavity field plots - sanity numbers", "",
             "Generated by `experiment/validation/cavity_fields.py` from the newest checkpoint of each run in "
             "`logs/validation/cavity_re1000/<run>/checkpoints/` (Re = 1000 validation, v6). "
             + ("Frame: the unit square is the square of the innermost boundary rows (bottom-left innermost boundary "
                "particle = (0, 0), top-right = (1, 1)); positions, h and dx are scaled by 1 / (1 + 2 dx), velocities "
                "unscaled, so psi and omega_z are in units of U L and U / L with L = 1 + 2 dx. "
                if FRAME == "wall" else "Frame: the fluid lattice box, X = x + 1/2. ")
             + "dx = particle spacing in the frame; omega_z = dv/dx - du/dy. Reference: Marchi, "
             "Santiago & Carvalho Jr. (2021) Table 19 Tc (primary vortex; `docs/validation/data/marchi2021_re1000.csv`).",
             ""]
    for check_path in sorted(output_directory.glob("*_check.json")):
        lines += job_markdown(json.loads(check_path.read_text()))
    for difference_path in sorted(output_directory.glob("difference_*.json")):
        lines += difference_markdown(json.loads(difference_path.read_text()))
    lines += NOTES_MARKDOWN
    markdown_path = output_directory / "fields_check.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path


# ----------------------------------------------------------------------------------------------------------

def to_json_ready(value):
    if isinstance(value, dict):
        return {str(key): to_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_ready(item) for item in value]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", nargs="*", default=[], help="run directory names in --campaign")
    parser.add_argument("--campaign", default=str(DEFAULT_CAMPAIGN_DIRECTORY),
                        help="campaign directory holding the job directories")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIRECTORY), help="output directory")
    parser.add_argument("--grid", type=int, default=2048, help="grid cells per side of the main field (>= 2048)")
    parser.add_argument("--stream-function", choices=["poisson", "velocity-integral"], default="poisson",
                        help="stream function drawn as streamlines and used for the plotted vortex centres "
                             "(both are computed and reported in fields_check.md)")
    parser.add_argument("--validate-points", type=int, default=300,
                        help="grid points checked against cavity_sampling.interpolate (0 = skip)")
    parser.add_argument("--normal-priority", action="store_true",
                        help="do not lower this process's scheduling priority (Windows)")
    parser.add_argument("--difference", nargs=2, action="append", metavar=("JOB_MINUEND", "JOB_SUBTRAHEND"),
                        help="also render JOB_MINUEND minus JOB_SUBTRAHEND (repeatable)")
    parser.add_argument("--difference-checkpoint", type=int, default=None, metavar="STEP",
                        help="checkpoint step present in both difference jobs (e.g. 7790000): also report JOB_MINUEND "
                             "minus JOB_SUBTRAHEND at that common step and JOB_MINUEND final minus that step")
    parser.add_argument("--cache-directory", default=None,
                        help="directory for the per-job grid fields (npz, ~70 MB per job at 2048^2); written after "
                             "every scatter, read by --difference and by --reuse-cache")
    parser.add_argument("--reuse-cache", action="store_true",
                        help="load the main-grid fields from --cache-directory instead of recomputing the scatter")
    parser.add_argument("--frame", choices=["wall", "fluid"], default=FRAME,
                        help="unit square of the figures: innermost boundary rows (wall, default) or the fluid box")
    parser.add_argument("--zoom-fluid-span", type=float, default=ZOOM_FLUID_SPAN,
                        help=f"fluid part of each corner zoom window in L (default {ZOOM_FLUID_SPAN}, ~37 dx at 2M)")
    parser.add_argument("--zoom-boundary-span", type=float, default=ZOOM_BOUNDARY_SPAN,
                        help=f"boundary part of each corner zoom window in L (default {ZOOM_BOUNDARY_SPAN})")
    return parser.parse_args()


def main():
    global ZOOM_FLUID_SPAN, ZOOM_BOUNDARY_SPAN, FRAME
    arguments = parse_arguments()
    FRAME = arguments.frame
    # coarse cases (e.g. 25k, dx = L/160) need wider windows to show more than a few particles per corner
    ZOOM_FLUID_SPAN, ZOOM_BOUNDARY_SPAN = arguments.zoom_fluid_span, arguments.zoom_boundary_span
    if not arguments.normal_priority:
        lower_own_process_priority()
    output_directory = pathlib.Path(arguments.out)
    output_directory.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        print(message, flush=True)

    for job in arguments.jobs:
        job_start = time.perf_counter()
        job_directory = pathlib.Path(arguments.campaign) / job
        log(f"{job}: loading {job_directory}")
        data = load_job(job_directory)
        fields = compute_job(data, arguments.grid, arguments.validate_points, arguments.stream_function, log,
                             arguments.cache_directory, arguments.reuse_cache)
        for stream_function_name, centre_set in fields["vortex_centres_by_stream_function"].items():
            for vortex_name, centre in centre_set.items():
                log(f"  {stream_function_name:17s} {vortex_name:8s} SPH ({centre['position'][0]:.4f}, "
                    f"{centre['position'][1]:.4f}) psi {centre['stream_function']:+.5e} omega_z "
                    f"{centre['vorticity_z']:+.4f}"
                    + (f"   reference ({centre['reference_position'][0]:.4f}, {centre['reference_position'][1]:.4f})"
                       if centre["reference_position"] is not None else "")
                    + ("" if centre["found"] else "   [NOT FOUND]"))
        for name, value in fields["stream_function_diagnostics"].items():
            log(f"  {name}: {value:.4g}")
        timer = time.perf_counter()
        zooms = {corner_name: compute_zoom(data, fields, corner_name) for corner_name in ZOOM_CORNERS}
        log(f"  corner zooms: {time.perf_counter() - timer:.1f} s")
        for field_name in ("velocity", "vorticity"):
            timer = time.perf_counter()
            render_field_figure(field_name, data, fields, zooms, output_directory, log)
            log(f"  {field_name} figure: {time.perf_counter() - timer:.1f} s")
        check = sanity_numbers(data, fields, zooms)
        check["runtime_seconds"] = time.perf_counter() - job_start
        check["peak_memory_bytes"] = peak_memory_bytes()
        (output_directory / f"{job}_check.json").write_text(json.dumps(to_json_ready(check), indent=1))
        log(f"  runtime {check['runtime_seconds']:.1f} s, peak memory {check['peak_memory_bytes'] / 2 ** 30:.2f} GiB")
        del data, fields, zooms
    for job_minuend, job_subtrahend in arguments.difference or []:
        timer = time.perf_counter()
        campaign_directory = pathlib.Path(arguments.campaign)
        difference = compute_difference(campaign_directory, job_minuend, job_subtrahend, arguments.grid,
                                        arguments.cache_directory, log)
        context = {}
        if arguments.difference_checkpoint is not None:
            # Separate the effect of K from the effect of the different end times: the same two runs at a common
            # checkpoint step, and the K = 2 run against itself (final minus that checkpoint).
            checkpoint = f"c{arguments.difference_checkpoint:010d}"
            for label, (first_job, first_snapshot, second_job, second_snapshot) in {
                    "same_step": (job_minuend, checkpoint, job_subtrahend, checkpoint),
                    "same_run": (job_minuend, "final", job_minuend, checkpoint)}.items():
                comparison = compute_difference(campaign_directory, first_job, second_job, arguments.grid,
                                                arguments.cache_directory, log, first_snapshot, second_snapshot,
                                                forced_grid=difference["grid"])
                context[label] = difference_summary(comparison)
                del comparison
        difference["context"] = context
        render_difference_figure(difference, output_directory, log)
        summary = difference_summary(difference)
        summary["context"] = context
        log("  interior: " + ", ".join(f"{key} {value:.3e}" for key, value in summary["statistics_interior"].items()
                                       if isinstance(value, float)))
        for seam in summary["seam"]:
            log(f"  seam: {seam}")
        difference_path = output_directory / f"difference_{job_minuend}_minus_{job_subtrahend}.json"
        difference_path.write_text(json.dumps(to_json_ready(summary), indent=1))
        log(f"  difference: {time.perf_counter() - timer:.1f} s, peak memory {peak_memory_bytes() / 2 ** 30:.2f} GiB")
        del difference
    markdown_path = write_check_markdown(output_directory)
    log(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
