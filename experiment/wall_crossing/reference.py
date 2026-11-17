"""
reference.py - CPU (numpy, float64) reference of one evaluation of the solver's two neighbour traversals with
numerics.wall_boundary crossing, 2-D and 3-D: the density rate of the fused correction + density pass and the
acceleration (and shift) of force, for every fluid particle, written from the scheme's formulas:

    t_ij = phi_i / (phi_i + d_j),  phi_i = max(phi(r_i), floor),  d_j = d(r_j)
    velocity (a wall value v_w exists):  v_j - v_i -> (v_w - v_i) (1 + d_j / phi_i)
    pressure, density (no wall value):   theta_j - theta_i -> G_i . (r_j - r_i),  G_i = A_f^-1 g_f
    A_f = sum_{j fluid} V_j (r_j - r_i) (x) grad W_ij,   g_f = sum_{j fluid} V_j (theta_j - theta_i) grad W_ij

and, for the terms that do not change, from the shader conventions (experiment/v7/shaders):
    grad W_ij = grad_i W(|r_i - r_j|) (Wendland C4, support radius h), V_j = m_j / rho_j,
    L_i = the six stored entries of (sum_all V_j (r_j - r_i) (x) grad W_ij + xi I)^-1 (the lower triangle of the
    inverse, mirrored; identity below the determinant threshold; Frobenius cap), kernel sum = sum_all V_j W_ij.

Density rate of a fluid particle i (pass 1; rho, v of step n):
    sum_{j fluid} [ -rho_i V_j (v_j - v_i) . L grad W + 2 delta h c0 (rho_j - rho_i) V_j / (r^2 + eps^2) (r_j - r_i) . L grad W ]
  + sum_{j wall}  [ -rho_i V_j c_ij (v_w,j - v_i) . L grad W ],   c_ij = 1 + d_j / phi_i (extrapolate) or 1 (wall_velocity)
  + 2 delta h c0 G_i . (L sum_{j wall} V_j r^2 / (r^2 + eps^2) grad W),   G_i = A_f^-1 g_f(rho)  (0 on a fallback)

Acceleration of a fluid particle i (pass 2; V_j = m_i / rho_j as force.comp takes it):
    a_p = -(1 / rho_i) [ sum_{j fluid} V_j F_ij L grad W + s p_i L sum_{j wall} V_j grad W + (A_f^-1 - L) g_f(p) ]
          F_ij = p_j + p_i and s = 2 where p_i > 0 or the kernel sum < 0.75, else p_j - p_i and s = 0;
          the last term is dropped on a fallback of A_f
    a_v = sum_j 2 (d + 2) nu V_j c'_ij (v_i - v_j) . (r_i - r_j) / (r^2 + eps^2) L grad W,  c'_ij = 1 + d_j / phi_i for a wall j
    a   = a_p + a_v + g

Nothing here imports Vulkan. The inputs are the float32 values the GPU holds (positions, velocities, densities,
pressures, masses) and the float32 specialization constants, promoted to float64.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from experiment.v7.utils.wall_crossing_v7 import (
    WallCrossingSettings,
    box_depth_behind_wall,
    box_distance_to_wall,
)


def float32_value(value: float) -> float:
    """The float64 of the float32 a specialization constant carries."""
    return float(np.float32(value))


@dataclass
class ReferenceConstants:
    """The constants of one case as the pipelines get them (float32, see simulator_v7._global_entries)."""
    dimension: int
    smoothing_length: float
    kernel_coefficient: float
    kernel_gradient_coefficient: float
    speed_of_sound: float
    delta_coefficient: float
    timestep: float
    cfl_number: float
    eps_h_squared: float
    regularization_xi: float
    regularization_determinant_threshold: float
    regularization_max_frobenius_norm: float
    use_density_diffusion: bool
    use_pst: bool
    pst_main_shift_coefficient: float
    pst_anti_shift_coefficient: float
    gravity: tuple
    viscosity: float                # the fluid material's kinematic viscosity (force.comp: self_params.viscosity)
    particle_radius: float          # self_params.radius
    particle_volume: float          # self_params.volume
    distance_floor: float           # spec const 121
    box_lower: tuple                # spec consts 122-124
    box_upper: tuple                # spec consts 125-127
    continuity: str                 # "extrapolate" | "wall_velocity"

    @classmethod
    def from_case(cls, case) -> "ReferenceConstants":
        """From a loaded CaseV7 (case_loader_v7.load_case_v7) with wall_boundary crossing."""
        physics, numerics = case.physics, case.numerics
        settings: WallCrossingSettings = numerics.wall_crossing
        fluid = next(material for material in case.materials if material.kind == 0)
        return cls(
            dimension=int(physics.dimension),
            smoothing_length=float32_value(physics.smoothing_length),
            kernel_coefficient=float32_value(physics.kernel_coefficient),
            kernel_gradient_coefficient=float32_value(physics.kernel_gradient_coefficient),
            speed_of_sound=float32_value(physics.speed_of_sound),
            delta_coefficient=float32_value(physics.delta_coefficient),
            timestep=float32_value(physics.timestep),
            cfl_number=float32_value(physics.cfl_number),
            eps_h_squared=float32_value(numerics.eps_h_squared),
            regularization_xi=float32_value(numerics.regularization_xi),
            regularization_determinant_threshold=float32_value(numerics.regularization_determinant_threshold),
            regularization_max_frobenius_norm=float32_value(numerics.regularization_max_frobenius_norm),
            use_density_diffusion=bool(numerics.use_density_diffusion),
            use_pst=bool(numerics.use_pst),
            pst_main_shift_coefficient=float32_value(numerics.pst_main_shift_coefficient),
            pst_anti_shift_coefficient=float32_value(numerics.pst_anti_shift_coefficient),
            gravity=tuple(float32_value(component) for component in physics.gravity),
            viscosity=float32_value(fluid.viscosity),
            particle_radius=float32_value(fluid.radius),
            particle_volume=float32_value(fluid.volume),
            distance_floor=float32_value(settings.distance_floor(fluid.radius)),
            box_lower=tuple(float32_value(component) for component in settings.box_lower),
            box_upper=tuple(float32_value(component) for component in settings.box_upper),
            continuity=settings.continuity,
        )


@dataclass
class ReferenceResult:
    """Per fluid particle (rows in the order of fluid_index = the fluid rows of the input)."""
    fluid_index: np.ndarray                 # rows of the input arrays
    wall_distance: np.ndarray               # phi_i with the floor
    wall_distance_raw: np.ndarray           # phi(r_i)
    correction_inverse: np.ndarray          # (F, 6): xx, yy, zz, xy, xz, yz of the stored L
    correction_fallback: np.ndarray         # (F,) bool
    kernel_sum: np.ndarray
    fluid_density_gradient_sum: np.ndarray  # g_f of the density (the gradient accumulator)
    fluid_inverse: np.ndarray               # (F, 6): the stored A_f^-1
    fluid_inverse_valid: np.ndarray         # (F,) bool
    fluid_inverse_capped: np.ndarray        # (F,) bool: the Frobenius cap scaled A_f^-1 down
    correction_capped: np.ndarray           # (F,) bool: the Frobenius cap scaled L down
    density_rate: np.ndarray
    density_rate_drift_fluid: np.ndarray
    density_rate_drift_wall: np.ndarray
    density_rate_diffusion_fluid: np.ndarray
    density_rate_diffusion_wall: np.ndarray
    density_rate_scale: np.ndarray          # sum of |pair terms| (rounding scale of the rate)
    acceleration: np.ndarray                # (F, 3)
    acceleration_pressure: np.ndarray
    acceleration_viscosity: np.ndarray
    acceleration_scale: np.ndarray          # sum of |pair terms| (norms; rounding scale of the acceleration)
    shift: np.ndarray                       # (F, 3)
    shift_scale: np.ndarray                 # sum of |pair terms| (norms; rounding scale of the shift)
    # the same sums with the wall neighbours' values written out pair by pair (no L A = I step):
    density_rate_diffusion_wall_pairwise: np.ndarray
    acceleration_pressure_pairwise: np.ndarray
    wall_neighbour_count: np.ndarray
    fluid_neighbour_count: np.ndarray
    notes: dict = field(default_factory=dict)


def kernel_value(distance: np.ndarray, constants: ReferenceConstants) -> np.ndarray:
    normalized = distance / constants.smoothing_length
    one_minus = np.clip(1.0 - normalized, 0.0, None)
    return constants.kernel_coefficient * one_minus ** 6 * ((35.0 / 3.0) * normalized * normalized
                                                           + 6.0 * normalized + 1.0)


def kernel_gradient(relative_position: np.ndarray, distance: np.ndarray,
                    constants: ReferenceConstants) -> np.ndarray:
    """grad_i W_ij for relative_position = r_i - r_j (helpers.glsl evaluate_kernel_gradient)."""
    normalized = distance / constants.smoothing_length
    one_minus = np.clip(1.0 - normalized, 0.0, None)
    magnitude = (constants.kernel_gradient_coefficient * one_minus ** 5
                 * normalized * ((-280.0 / 3.0) * normalized + (-56.0 / 3.0)))
    return (magnitude / distance)[:, None] * relative_position


def regularized_inverse(matrices: np.ndarray, constants: ReferenceConstants
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """correction.comp's epilogue on a stack of matrices M[a, b] = sum V (r_j - r_i)[a] grad W[b]:
    xi on the diagonal, the 2-D block, |det| > threshold -> inverse with the Frobenius cap, else the identity.
    Returns (the six stored entries xx, yy, zz, xy, xz, yz of every inverse, valid, capped): valid = the
    determinant passed, capped = the Frobenius cap scaled the inverse down."""
    regularized = np.array(matrices, dtype=np.float64, copy=True)
    regularized[:, 0, 0] += constants.regularization_xi
    regularized[:, 1, 1] += constants.regularization_xi
    if constants.dimension == 2:
        regularized[:, 0, 2] = 0.0
        regularized[:, 1, 2] = 0.0
        regularized[:, 2, 0] = 0.0
        regularized[:, 2, 1] = 0.0
        regularized[:, 2, 2] = 1.0
    else:
        regularized[:, 2, 2] += constants.regularization_xi
    determinant = np.linalg.det(regularized)
    valid = np.abs(determinant) > constants.regularization_determinant_threshold
    inverse = np.tile(np.eye(3), (regularized.shape[0], 1, 1))
    capped = np.zeros(regularized.shape[0], dtype=bool)
    if valid.any():
        inverse[valid] = np.linalg.inv(regularized[valid])
        frobenius = np.sqrt((inverse[valid] ** 2).sum(axis=(1, 2)))
        scale = np.where(frobenius > constants.regularization_max_frobenius_norm,
                         constants.regularization_max_frobenius_norm / frobenius, 1.0)
        inverse[valid] *= scale[:, None, None]
        capped[np.flatnonzero(valid)[scale < 1.0]] = True
    # GLSL m[column][row]: the stored (m[0][1], m[0][2], m[1][2]) are the inverse's (row 1, column 0),
    # (row 2, column 0), (row 2, column 1) entries
    stored = np.stack([inverse[:, 0, 0], inverse[:, 1, 1], inverse[:, 2, 2],
                       inverse[:, 1, 0], inverse[:, 2, 0], inverse[:, 2, 1]], axis=1)
    return stored, valid, capped


def symmetric_matrices(stored: np.ndarray) -> np.ndarray:
    """(F, 3, 3) symmetric matrices of the six stored entries xx, yy, zz, xy, xz, yz."""
    matrices = np.empty((stored.shape[0], 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = stored[:, 0]
    matrices[:, 1, 1] = stored[:, 1]
    matrices[:, 2, 2] = stored[:, 2]
    matrices[:, 0, 1] = matrices[:, 1, 0] = stored[:, 3]
    matrices[:, 0, 2] = matrices[:, 2, 0] = stored[:, 4]
    matrices[:, 1, 2] = matrices[:, 2, 1] = stored[:, 5]
    return matrices


def _sum_by_particle(particle_of_pair: np.ndarray, values: np.ndarray, particle_count: int) -> np.ndarray:
    """Sum the pair values (P,) or (P, ...) by their self particle."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        return np.bincount(particle_of_pair, weights=values, minlength=particle_count)
    flat = values.reshape(values.shape[0], -1)
    summed = np.stack([np.bincount(particle_of_pair, weights=flat[:, column], minlength=particle_count)
                       for column in range(flat.shape[1])], axis=1)
    return summed.reshape((particle_count,) + values.shape[1:])


def evaluate_reference(positions: np.ndarray, velocities: np.ndarray, densities: np.ndarray,
                       pressures: np.ndarray, masses: np.ndarray, is_wall: np.ndarray,
                       constants: ReferenceConstants) -> ReferenceResult:
    """One evaluation of both passes on one state: the density pass and force read the same positions,
    velocities, densities and pressures (the bootstrap's situation: force reads the uploaded (rho_0, P_0))."""
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    densities = np.asarray(densities, dtype=np.float64)
    pressures = np.asarray(pressures, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    is_wall = np.asarray(is_wall, dtype=bool)
    dimension = constants.dimension
    support = constants.smoothing_length

    fluid_index = np.flatnonzero(~is_wall)
    fluid_count = fluid_index.size
    all_tree = cKDTree(positions)
    fluid_tree = cKDTree(positions[fluid_index])
    pairs = fluid_tree.sparse_distance_matrix(all_tree, support, output_type="coo_matrix")
    self_of_pair = pairs.row.astype(np.int64)             # index into fluid_index
    neighbour_of_pair = pairs.col.astype(np.int64)        # row of the input
    self_row = fluid_index[self_of_pair]
    displacement = positions[neighbour_of_pair] - positions[self_row]          # r_j - r_i
    distance = np.sqrt((displacement * displacement).sum(axis=1))
    keep = (neighbour_of_pair != self_row) & (distance < support) & (distance >= 1e-12)
    self_of_pair, neighbour_of_pair, self_row = self_of_pair[keep], neighbour_of_pair[keep], self_row[keep]
    displacement, distance = displacement[keep], distance[keep]
    neighbour_is_wall = is_wall[neighbour_of_pair]
    neighbour_is_fluid = ~neighbour_is_wall

    gradient = kernel_gradient(-displacement, distance, constants)             # grad_i W_ij
    value = kernel_value(distance, constants)
    distance_squared = distance * distance
    softened = distance_squared + constants.eps_h_squared

    # ---- geometry -----------------------------------------------------------------------------------------
    wall_distance_raw = box_distance_to_wall(positions[fluid_index], constants.box_lower, constants.box_upper,
                                             dimension)
    wall_distance = np.maximum(wall_distance_raw, constants.distance_floor)
    wall_depth = box_depth_behind_wall(positions, constants.box_lower, constants.box_upper, dimension)
    extrapolation = np.where(neighbour_is_wall,
                             1.0 + wall_depth[neighbour_of_pair] / wall_distance[self_of_pair], 1.0)

    # ---- pass 1: correction, A_f, g_f, density rate ---------------------------------------------------------
    volume = masses[neighbour_of_pair] / densities[neighbour_of_pair]           # V_j = m_j / rho_j
    outer = volume[:, None, None] * displacement[:, :, None] * gradient[:, None, :]
    correction_matrix = _sum_by_particle(self_of_pair, outer, fluid_count)
    fluid_matrix = _sum_by_particle(self_of_pair[neighbour_is_fluid], outer[neighbour_is_fluid], fluid_count)
    correction_stored, correction_valid, correction_capped = regularized_inverse(correction_matrix, constants)
    fluid_stored, fluid_valid, fluid_capped = regularized_inverse(fluid_matrix, constants)
    correction = symmetric_matrices(correction_stored)
    fluid_inverse = symmetric_matrices(fluid_stored)
    kernel_sum = _sum_by_particle(self_of_pair, volume * value, fluid_count)

    corrected_gradient = np.einsum("pab,pb->pa", correction[self_of_pair], gradient)          # L_i grad W_ij
    self_density = densities[self_row]
    density_difference = densities[neighbour_of_pair] - self_density
    velocity_difference = velocities[neighbour_of_pair] - velocities[self_row]              # v_j - v_i
    diffusion_coefficient = 2.0 * constants.delta_coefficient * support * constants.speed_of_sound

    drift_pair = -self_density * volume * (velocity_difference * corrected_gradient).sum(axis=1)
    continuity_factor = extrapolation if constants.continuity == "extrapolate" else np.ones_like(extrapolation)
    drift_pair = drift_pair * np.where(neighbour_is_wall, continuity_factor, 1.0)
    drift_fluid = _sum_by_particle(self_of_pair[neighbour_is_fluid], drift_pair[neighbour_is_fluid], fluid_count)
    drift_wall = _sum_by_particle(self_of_pair[neighbour_is_wall], drift_pair[neighbour_is_wall], fluid_count)

    fluid_density_gradient_sum = _sum_by_particle(
        self_of_pair[neighbour_is_fluid],
        (volume * density_difference)[neighbour_is_fluid, None] * gradient[neighbour_is_fluid], fluid_count)
    one_sided_density_gradient = np.einsum("fab,fb->fa", fluid_inverse, fluid_density_gradient_sum)
    one_sided_density_gradient[~fluid_valid] = 0.0

    diffusion_fluid = np.zeros(fluid_count)
    diffusion_wall = np.zeros(fluid_count)
    diffusion_wall_pairwise = np.zeros(fluid_count)
    diffusion_scale = np.zeros(fluid_count)
    if constants.use_density_diffusion:
        diffusion_pair = (diffusion_coefficient * density_difference * volume / softened
                          * (displacement * corrected_gradient).sum(axis=1))
        diffusion_fluid = _sum_by_particle(self_of_pair[neighbour_is_fluid], diffusion_pair[neighbour_is_fluid],
                                           fluid_count)
        wall_vector = _sum_by_particle(
            self_of_pair[neighbour_is_wall],
            (volume * distance_squared / softened)[neighbour_is_wall, None] * gradient[neighbour_is_wall],
            fluid_count)
        diffusion_wall = diffusion_coefficient * (
            one_sided_density_gradient * np.einsum("fab,fb->fa", correction, wall_vector)).sum(axis=1)
        # the same continuation written pair by pair: psi_ij = G_i . (r_j - r_i)
        continued_difference = (one_sided_density_gradient[self_of_pair] * displacement).sum(axis=1)
        pairwise = (diffusion_coefficient * continued_difference * volume / softened
                    * (displacement * corrected_gradient).sum(axis=1))
        diffusion_wall_pairwise = _sum_by_particle(self_of_pair[neighbour_is_wall], pairwise[neighbour_is_wall],
                                                   fluid_count)
        diffusion_scale = _sum_by_particle(self_of_pair[neighbour_is_fluid],
                                           np.abs(diffusion_pair[neighbour_is_fluid]), fluid_count) \
            + np.abs(diffusion_wall)
    density_rate = drift_fluid + drift_wall + diffusion_fluid + diffusion_wall
    density_rate_scale = _sum_by_particle(self_of_pair, np.abs(drift_pair), fluid_count) + diffusion_scale

    # ---- pass 2: force ------------------------------------------------------------------------------------
    force_volume = masses[self_row] / densities[neighbour_of_pair]             # V_j = m_i / rho_j (force.comp)
    self_pressure = pressures[self_row]
    plus_branch_particle = (pressures[fluid_index] > 0.0) | (kernel_sum < 0.75)
    plus_branch = plus_branch_particle[self_of_pair]
    pressure_combined = np.where(plus_branch, pressures[neighbour_of_pair] + self_pressure,
                                 pressures[neighbour_of_pair] - self_pressure)
    pressure_pair = -(force_volume * pressure_combined / self_density)[:, None] * corrected_gradient
    pressure_fluid = _sum_by_particle(self_of_pair[neighbour_is_fluid], pressure_pair[neighbour_is_fluid],
                                      fluid_count)
    wall_kernel_gradient_sum = _sum_by_particle(
        self_of_pair[neighbour_is_wall], force_volume[neighbour_is_wall, None] * gradient[neighbour_is_wall],
        fluid_count)
    fluid_pressure_gradient_sum = _sum_by_particle(
        self_of_pair[neighbour_is_fluid],
        (force_volume * (pressures[neighbour_of_pair] - self_pressure))[neighbour_is_fluid, None]
        * gradient[neighbour_is_fluid], fluid_count)
    fluid_density = densities[fluid_index]
    fluid_pressure = pressures[fluid_index]
    branch_factor = np.where(plus_branch_particle, 2.0, 0.0)
    wall_pressure_sum = (branch_factor * fluid_pressure)[:, None] * np.einsum(
        "fab,fb->fa", correction, wall_kernel_gradient_sum)
    gradient_part = np.einsum("fab,fb->fa", fluid_inverse - correction, fluid_pressure_gradient_sum)
    gradient_part[~fluid_valid] = 0.0
    acceleration_pressure = pressure_fluid - (wall_pressure_sum + gradient_part) / fluid_density[:, None]

    # the wall neighbours' pressure written pair by pair: p_j = p_i + G_i . (r_j - r_i), G_i = A_f^-1 g_f(p)
    one_sided_pressure_gradient = np.einsum("fab,fb->fa", fluid_inverse, fluid_pressure_gradient_sum)
    one_sided_pressure_gradient[~fluid_valid] = 0.0
    continued_pressure = self_pressure + (one_sided_pressure_gradient[self_of_pair] * displacement).sum(axis=1)
    pairwise_combined = np.where(plus_branch, continued_pressure + self_pressure, continued_pressure - self_pressure)
    pairwise_pair = -(force_volume * pairwise_combined / self_density)[:, None] * corrected_gradient
    acceleration_pressure_pairwise = pressure_fluid + _sum_by_particle(
        self_of_pair[neighbour_is_wall], pairwise_pair[neighbour_is_wall], fluid_count)

    morris_coefficient = 2.0 * (dimension + 2.0)
    viscous_scalar = (morris_coefficient * constants.viscosity * force_volume * extrapolation
                      * ((-velocity_difference) * (-displacement)).sum(axis=1) / softened)
    viscous_pair = viscous_scalar[:, None] * corrected_gradient
    acceleration_viscosity = _sum_by_particle(self_of_pair, viscous_pair, fluid_count)
    gravity = np.asarray(constants.gravity, dtype=np.float64)
    acceleration = acceleration_pressure + acceleration_viscosity + gravity[None, :]
    acceleration_scale = (
        _sum_by_particle(self_of_pair[neighbour_is_fluid],
                         np.sqrt((pressure_pair[neighbour_is_fluid] ** 2).sum(axis=1)), fluid_count)
        + np.sqrt((wall_pressure_sum ** 2).sum(axis=1)) / fluid_density
        + np.sqrt((gradient_part ** 2).sum(axis=1)) / fluid_density
        + _sum_by_particle(self_of_pair, np.sqrt((viscous_pair ** 2).sum(axis=1)), fluid_count)
        + np.sqrt((gravity ** 2).sum()))

    # ---- particle shifting (unchanged by the wall mode; kept so the whole of force is compared) ------------
    shift = np.zeros((fluid_count, 3))
    if constants.use_pst:
        spacing = 2.0 * constants.particle_radius
        kernel_at_spacing = float(kernel_value(np.array([spacing]), constants)[0])
        base_factor = (constants.cfl_number * constants.pst_main_shift_coefficient * 2.0 * support * support)
        distance_ratio = (distance - spacing) / spacing
        disorder = np.where(distance_ratio > 0.0, 0.0,
                            distance_ratio * np.minimum(densities[neighbour_of_pair] / self_density, 1.0))
        crowding = (value / kernel_at_spacing) ** 4 if kernel_at_spacing > 1e-12 else np.zeros_like(value)
        main_scalar = 1.0 - disorder + 0.1 * kernel_sum[self_of_pair] + 0.2 * crowding
        shift_main = _sum_by_particle(
            self_of_pair, (-base_factor * main_scalar * constants.particle_volume)[:, None] * corrected_gradient,
            fluid_count)
        anti_scalar = (distance / spacing) * constants.pst_anti_shift_coefficient
        shift_anti = _sum_by_particle(
            self_of_pair, (base_factor * anti_scalar * constants.particle_volume)[:, None] * corrected_gradient,
            fluid_count)
        blend = np.clip(kernel_sum, 0.0, 1.0) ** 4
        shift = blend[:, None] * shift_main + (1.0 - blend)[:, None] * shift_anti
        gradient_norm = np.sqrt((corrected_gradient ** 2).sum(axis=1))
        shift_scale = (blend * _sum_by_particle(
            self_of_pair, np.abs(base_factor * main_scalar * constants.particle_volume) * gradient_norm, fluid_count)
            + (1.0 - blend) * _sum_by_particle(
                self_of_pair, np.abs(base_factor * anti_scalar * constants.particle_volume) * gradient_norm,
                fluid_count))
    else:
        shift_scale = np.ones(fluid_count)

    return ReferenceResult(
        fluid_index=fluid_index,
        wall_distance=wall_distance,
        wall_distance_raw=wall_distance_raw,
        correction_inverse=correction_stored,
        correction_fallback=~correction_valid,
        kernel_sum=kernel_sum,
        fluid_density_gradient_sum=fluid_density_gradient_sum,
        fluid_inverse=fluid_stored,
        fluid_inverse_valid=fluid_valid,
        fluid_inverse_capped=fluid_capped,
        correction_capped=correction_capped,
        density_rate=density_rate,
        density_rate_drift_fluid=drift_fluid,
        density_rate_drift_wall=drift_wall,
        density_rate_diffusion_fluid=diffusion_fluid,
        density_rate_diffusion_wall=diffusion_wall,
        density_rate_scale=density_rate_scale,
        acceleration=acceleration,
        acceleration_pressure=acceleration_pressure,
        acceleration_viscosity=acceleration_viscosity,
        acceleration_scale=acceleration_scale,
        shift=shift,
        shift_scale=shift_scale,
        density_rate_diffusion_wall_pairwise=diffusion_wall_pairwise,
        acceleration_pressure_pairwise=acceleration_pressure_pairwise,
        wall_neighbour_count=np.bincount(self_of_pair[neighbour_is_wall], minlength=fluid_count),
        fluid_neighbour_count=np.bincount(self_of_pair[neighbour_is_fluid], minlength=fluid_count),
        notes={"pair_count": int(self_of_pair.size)},
    )
