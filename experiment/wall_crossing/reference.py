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
With continuity_wall_component normal (the default from 2026-11-18 to 2026-11-19), (v_w,j - v_i) of a wall
neighbour is replaced by its component along n_j, the unit vector from the wall particle to the nearest point of
the box, before c_ij; with full (the default) it is taken whole.
The last line is delta_wall extension (the default until 2026-11-19; reflection, the default since then, adds a
term below). The diagnostic values replace it by a sum over the wall
neighbours of 2 delta h c0 psi_ij V_j / (r^2 + eps^2) (r_j - r_i) . L grad W with
    rest_density:  psi_ij = rho_j - rho_i (a wall stores rho0),
    body_force:    psi_ij = rho_i g . (r_j - r_i) / c_i^2,  c_i^2 = dp/drho of the Tait EOS at rho_i
                   = gamma B / rho0 (rho_i / rho0)^(gamma - 1)  (wall acceleration 0).

Acceleration of a fluid particle i (pass 2; V_j = m_i / rho_j as force.comp takes it):
    a_p = -(1 / rho_i) [ sum_{j fluid} V_j F_ij L grad W + s p_i L sum_{j wall} V_j grad W + (A_f^-1 - L) g_f(p) ]
          F_ij = p_j + p_i and s = 2 where p_i > 0 or the kernel sum < 0.75, else p_j - p_i and s = 0;
          the last term is dropped on a fallback of A_f
    a_v = sum_j 2 (d + 2) nu V_j c'_ij (v_i - v_j) . (r_i - r_j) / (r^2 + eps^2) L grad W,  c'_ij = 1 + d_j / phi_i for a wall j
    a   = a_p + a_v + g

Second-order continuation of the velocity (viscous_extension / continuity_extension quadratic; both linear by
default, which is everything above). With a_j = d_j / phi_i a wall neighbour's velocity difference is
    v_j - v_i = (v_w - v_i) (1 + a_j)^2 - a_j grad v_i (r_j - r_i),
grad v_i ((grad v)_ab = d v_a / d x_b) solved per particle from the sums of pass 1 (V_j = m_j / rho_j):
    M_0   = sum_fluid V_j (v_j - v_i) (x) grad W + sum_wall V_j (v_w - v_i) (1 + a_j)^2 (x) grad W
    K_raw = sum_wall V_j (-a_j) (r_j - r_i) (x) grad W
    grad v_i = M_0 L (I - K_raw L)^-1
  viscous_extension quadratic:    a_v's wall pairs take (v_i - v_w) (1 + a_j)^2 + a_j grad v_i (r_j - r_i);
  continuity_extension quadratic: the wall neighbours' drift is -rho_i tr[(M_w + grad v_i K_raw) L], M_w the
                                  wall part of M_0 (the whole vector, whatever continuity_wall_component).
State of a particle: 0 no wall neighbour (nothing solved), 1 solved, 2 fallback to the first-order rule in both
terms (L itself fell back, det(I - K_raw L) < 0.5, or a solution that is not finite).

Pairwise geometry and the two reflections (geometry pairwise, pressure_wall reflection, delta_wall reflection;
with continuity_wall_component full the defaults since 2026-11-19, the "complete scheme"; geometry nearest_wall,
pressure_wall extension and delta_wall extension are everything above). With geometry pairwise the pair's own quantities replace phi_i, d_j and n_j everywhere above
(wall_crossing_v7.box_pair_geometry): the face the segment i -> j leaves the box through gives n_ij, phi_ij
(floored) and d_ij, so c_ij = c'_ij = 1 + d_ij / phi_ij, and the wall particle has a mirror point r_j' (r_j
reflected in every face it lies outside of).
  pressure_wall reflection: p_j - p_i = G_i . (r_j - r_i) - (G_i - rho_i g) . (r_j - r_j'), i.e. a_p gains
      (1 / rho_i) L M_d (G_i - rho_i g),  M_d = sum_{j wall} V_j grad W (x) (r_j - r_j')   (0 on a fallback of A_f)
  delta_wall reflection:    rho_j - rho_i = G_i . (r_j - r_i) - (G_i - rho_i g / c_i^2) . (r_j - r_j'), i.e. the
      density rate gains, beside the term of delta_wall extension,
      -2 delta h c0 (G_i - rho_i g / c_i^2) . sum_{j wall} V_j |grad W| r / (r^2 + eps^2) (r_j - r_j')
      (0 on a fallback of A_f; c_i^2 as with delta_wall body_force)

A case with wall_boundary simple or adami: from_case takes continuity wall_velocity, continuity_wall_component
full, delta_wall rest_density, geometry nearest_wall and pressure_wall extension (written out: they do not follow
the defaults of wall_boundary crossing),
with which the four terms of the density rate above are those of simple's (and adami's) fluid particles (a wall
neighbour enters with its stored velocity and its stored rho0, pair by pair); the force part is then not that
mode's and must not be used.

Nothing here imports Vulkan. The inputs are the float32 values the GPU holds (positions, velocities, densities,
pressures, masses) and the float32 specialization constants, promoted to float64.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from experiment.v7.utils.wall_crossing_v7 import (
    WALL_CROSSING_DEFAULT_CONTINUITY_WALL_COMPONENT,
    WALL_CROSSING_DEFAULT_DELTA_WALL,
    WALL_CROSSING_DEFAULT_GEOMETRY,
    WallCrossingSettings,
    box_depth_behind_wall,
    box_distance_to_wall,
    box_pair_geometry,
)

# the wall store (branch v7-crossing-wall-store) implements one scheme; the particle-pool version's switches keep their
# names here so the reference can still evaluate that version's variants
WALL_CROSSING_DEFAULT_PRESSURE_WALL = "reflection"


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
    # spec const 116: "reflection" (the default) | "extension" | the diagnostics "rest_density" | "body_force"
    delta_wall: str = WALL_CROSSING_DEFAULT_DELTA_WALL
    power_parameter: float = 7.0    # the EOS gamma (spec const 4)
    eos_constant: float = 0.0       # the fluid material's B = c0^2 rho0 / gamma (delta_wall body_force only)
    rest_density: float = 0.0       # the fluid material's rho0 (delta_wall body_force only)
    continuity_wall_component: str = WALL_CROSSING_DEFAULT_CONTINUITY_WALL_COMPONENT   # 117: "full" | "normal"
    viscous_extension: str = "linear"           # spec const 118: "linear" (the default) | "quadratic"
    continuity_extension: str = "linear"        # spec const 119: "linear" (the default) | "quadratic"
    geometry: str = WALL_CROSSING_DEFAULT_GEOMETRY              # spec const 128: "pairwise" | "nearest_wall"
    pressure_wall: str = WALL_CROSSING_DEFAULT_PRESSURE_WALL    # spec const 129: "reflection" | "extension"
    # numerics.volume_constraint: the rate the fused pass adds as rho_i * rate (0 = off, or before the first
    # update); a test sets it to the value it wrote to the GPU's volume_constraint_state
    volume_constraint_rate: float = 0.0

    @classmethod
    def from_case(cls, case) -> "ReferenceConstants":
        """From a loaded CaseV7 (case_loader_v7.load_case_v7). wall_boundary crossing: its settings. simple or
        adami: the settings under which the density rate's terms are that mode's (see the module docstring),
        with the box crossing would take."""
        physics, numerics = case.physics, case.numerics
        fluid = next(material for material in case.materials if material.kind == 0)
        settings: WallCrossingSettings = numerics.wall_crossing
        # the scheme's switches: the wall store's single scheme for crossing (continuity_wall_component normal with rule
        # set continuity_normal); simple and adami take a wall neighbour's whole velocity difference and its stored
        # rho0, pair by pair
        switches = {"continuity": "extrapolate", "delta_wall": "reflection", "geometry": "pairwise",
                    "pressure_wall": "reflection", "viscous_extension": "linear", "continuity_extension": "linear",
                    "continuity_wall_component": ("normal" if getattr(settings, "rule_set", "default")
                                                  == "continuity_normal" else "full")}
        if numerics.wall_boundary != "crossing":
            from experiment.v7.utils.wall_crossing_v7 import resolve_wall_crossing_box
            kinds = np.asarray([material.kind for material in case.materials])
            settings = resolve_wall_crossing_box(
                WallCrossingSettings(), np.asarray(case.initial.positions),
                kinds[np.asarray(case.initial.material_group)] == 1, float(fluid.radius), int(physics.dimension))
            switches = {"continuity": "wall_velocity", "delta_wall": "rest_density", "geometry": "nearest_wall",
                        "pressure_wall": "extension", "viscous_extension": "linear", "continuity_extension": "linear",
                        "continuity_wall_component": "full"}
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
            continuity=switches["continuity"],
            delta_wall=switches["delta_wall"],
            power_parameter=float32_value(physics.power_parameter),
            eos_constant=float32_value(fluid.eos_constant),
            rest_density=float32_value(fluid.rest_density),
            continuity_wall_component=switches["continuity_wall_component"],
            viscous_extension=switches["viscous_extension"],
            continuity_extension=switches["continuity_extension"],
            geometry=switches["geometry"],
            pressure_wall=switches["pressure_wall"],
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
    # (F, 4): the sum of the absolute pair terms of drift fluid, drift wall, diffusion fluid, diffusion wall
    # (the rounding scale of each term)
    density_rate_term_scales: np.ndarray | None = None
    # the second-order continuation of the velocity (computed for every case, used by the quadratic extensions)
    velocity_gradient: np.ndarray | None = None         # (F, 3, 3): [a, b] = d v_a / d x_b; zero unless state 1
    extension_state: np.ndarray | None = None           # (F,): 0 no wall neighbour, 1 solved, 2 fallback
    system_determinant: np.ndarray | None = None        # (F,): det(I - K_raw L) (1 without a wall neighbour)
    quadratic_factor_maximum: np.ndarray | None = None  # (F,): the largest (1 + a_j)^2 over the wall neighbours
    wall_viscous_diffusion_number: np.ndarray | None = None   # (F,): density_budget.glsl's number (pass 1 volumes)
    # the two reflections' own parts (zero without them): what pressure_wall reflection adds to the pressure
    # acceleration, what delta_wall reflection adds to the density rate
    acceleration_pressure_reflection: np.ndarray | None = None    # (F, 3)
    density_rate_diffusion_reflection: np.ndarray | None = None   # (F,)
    extrapolation_factor_maximum: np.ndarray | None = None        # (F,): the largest 1 / t_ij over the wall neighbours
    density_rate_volume_constraint: np.ndarray | None = None      # (F,): rho_i * rate (zero with the constraint off)
    # the fluid neighbours' pressure term and the wall neighbours' viscous term on their own (the same in both forms)
    acceleration_pressure_fluid: np.ndarray | None = None         # (F, 3)
    acceleration_viscosity_wall: np.ndarray | None = None         # (F, 3)


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
    depth_ratio = np.where(neighbour_is_wall, wall_depth[neighbour_of_pair] / wall_distance[self_of_pair], 0.0)
    pair_normal = None
    mirror_displacement = np.zeros_like(displacement)           # r_j - r_j' of a wall neighbour
    if constants.geometry == "pairwise":
        # the pair's own geometry: the face the segment i -> j leaves the box through
        pair_normal, pair_depth_ratio, pair_mirror = box_pair_geometry(
            positions[self_row], positions[neighbour_of_pair], constants.box_lower, constants.box_upper,
            dimension, constants.distance_floor)
        depth_ratio = np.where(neighbour_is_wall, pair_depth_ratio, 0.0)
        mirror_displacement = np.where(neighbour_is_wall[:, None], pair_mirror, 0.0)
    extrapolation = 1.0 + depth_ratio
    gravity = np.asarray(constants.gravity, dtype=np.float64)

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

    drift_velocity_difference = velocity_difference
    if constants.continuity_wall_component == "normal":
        # the default: a wall neighbour's velocity difference keeps its component along the wall's inward
        # normal only (the unit vector from the wall particle to the nearest point of the box)
        to_box = np.clip(positions, np.asarray(constants.box_lower), np.asarray(constants.box_upper)) - positions
        to_box[:, dimension:] = 0.0
        to_box_length = np.sqrt((to_box * to_box).sum(axis=1))
        inward_normal = np.divide(to_box, to_box_length[:, None], out=np.zeros_like(to_box),
                                  where=to_box_length[:, None] > 0.0)
        normal_of_pair = inward_normal[neighbour_of_pair] if pair_normal is None else pair_normal
        projected = (velocity_difference * normal_of_pair).sum(axis=1)[:, None] * normal_of_pair
        drift_velocity_difference = np.where(neighbour_is_wall[:, None], projected, velocity_difference)
    drift_pair = -self_density * volume * (drift_velocity_difference * corrected_gradient).sum(axis=1)
    continuity_factor = extrapolation if constants.continuity == "extrapolate" else np.ones_like(extrapolation)
    drift_pair = drift_pair * np.where(neighbour_is_wall, continuity_factor, 1.0)
    drift_fluid = _sum_by_particle(self_of_pair[neighbour_is_fluid], drift_pair[neighbour_is_fluid], fluid_count)
    drift_wall = _sum_by_particle(self_of_pair[neighbour_is_wall], drift_pair[neighbour_is_wall], fluid_count)

    # ---- the second-order continuation of the velocity: grad v_i = M_0 L (I - K_raw L)^-1 ------------------------
    quadratic_factor = (1.0 + depth_ratio) ** 2
    velocity_outer = volume[:, None, None] * velocity_difference[:, :, None] * gradient[:, None, :]
    fluid_velocity_matrix = _sum_by_particle(self_of_pair[neighbour_is_fluid], velocity_outer[neighbour_is_fluid],
                                             fluid_count)
    wall_velocity_matrix = _sum_by_particle(
        self_of_pair[neighbour_is_wall],
        quadratic_factor[neighbour_is_wall, None, None] * velocity_outer[neighbour_is_wall], fluid_count)
    wall_displacement_matrix = _sum_by_particle(
        self_of_pair[neighbour_is_wall], -depth_ratio[neighbour_is_wall, None, None] * outer[neighbour_is_wall],
        fluid_count)
    wall_neighbour_count = np.bincount(self_of_pair[neighbour_is_wall], minlength=fluid_count)
    system_matrix = np.eye(3)[None, :, :] - np.einsum("fab,fbc->fac", wall_displacement_matrix, correction)
    system_determinant = np.linalg.det(system_matrix)
    solvable = (wall_neighbour_count > 0) & correction_valid & (system_determinant >= 0.5)
    velocity_gradient = np.zeros((fluid_count, 3, 3))
    if solvable.any():
        velocity_gradient[solvable] = np.einsum(
            "fab,fbc,fcd->fad", (fluid_velocity_matrix + wall_velocity_matrix)[solvable], correction[solvable],
            np.linalg.inv(system_matrix[solvable]))
        solvable &= np.isfinite(velocity_gradient).all(axis=(1, 2))
        velocity_gradient[~solvable] = 0.0
    extension_state = np.where(wall_neighbour_count == 0, 0, np.where(solvable, 1, 2))
    second_order_of_pair = neighbour_is_wall & solvable[self_of_pair]
    # grad v_i (r_j - r_i) of every pair
    gradient_displacement = np.einsum("pab,pb->pa", velocity_gradient[self_of_pair], displacement)
    if constants.continuity_extension == "quadratic":
        quadratic_drift = -densities[fluid_index] * np.einsum(
            "fab,fba->f",
            wall_velocity_matrix + np.einsum("fab,fbc->fac", velocity_gradient, wall_displacement_matrix),
            correction)
        drift_wall = np.where(solvable, quadratic_drift, drift_wall)
    # the diagnostics of the DENSITY_BUDGET builds (density_budget.glsl): the largest (1 + a_j)^2 and the wall
    # viscous diffusion number, with pass 1's V_j
    quadratic_factor_maximum = np.zeros(fluid_count)
    np.maximum.at(quadratic_factor_maximum, self_of_pair[neighbour_is_wall], quadratic_factor[neighbour_is_wall])
    viscous_factor = quadratic_factor if constants.viscous_extension == "quadratic" else 1.0 + depth_ratio
    wall_viscous_rate = _sum_by_particle(
        self_of_pair[neighbour_is_wall],
        (2.0 * (dimension + 2.0) * constants.viscosity * volume
         * np.abs((displacement * corrected_gradient).sum(axis=1)) / softened * viscous_factor)[neighbour_is_wall],
        fluid_count)

    fluid_density_gradient_sum = _sum_by_particle(
        self_of_pair[neighbour_is_fluid],
        (volume * density_difference)[neighbour_is_fluid, None] * gradient[neighbour_is_fluid], fluid_count)
    one_sided_density_gradient = np.einsum("fab,fb->fa", fluid_inverse, fluid_density_gradient_sum)
    one_sided_density_gradient[~fluid_valid] = 0.0

    diffusion_fluid = np.zeros(fluid_count)
    diffusion_wall = np.zeros(fluid_count)
    diffusion_wall_pairwise = np.zeros(fluid_count)
    diffusion_reflection = np.zeros(fluid_count)
    diffusion_scale = np.zeros(fluid_count)
    body_force_notes = {}
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
        if constants.delta_wall == "rest_density":
            # the diagnostic: a wall neighbour takes the pair term with its stored density (rho0), as in simple,
            # and there is no term after the loop
            diffusion_wall = _sum_by_particle(self_of_pair[neighbour_is_wall], diffusion_pair[neighbour_is_wall],
                                              fluid_count)
            diffusion_wall_pairwise = diffusion_wall.copy()
        elif constants.delta_wall == "body_force":
            # the diagnostic: a wall neighbour takes the pair term with the density difference of a fluid at
            # rest against the wall, rho_i g . (r_j - r_i) / c_i^2 (wall acceleration 0), c_i^2 = dp/drho of
            # the Tait EOS at rho_i; there is no term after the loop
            gravity_vector = np.asarray(constants.gravity, dtype=np.float64)
            sound_speed_squared_rest = constants.power_parameter * constants.eos_constant / constants.rest_density
            sound_speed_squared = sound_speed_squared_rest * (
                self_density / constants.rest_density) ** (constants.power_parameter - 1.0)
            geometry_pair = volume / softened * (displacement * corrected_gradient).sum(axis=1)
            body_force_difference = self_density * (displacement @ gravity_vector) / sound_speed_squared
            diffusion_pair = np.where(neighbour_is_wall,
                                      diffusion_coefficient * body_force_difference * geometry_pair,
                                      diffusion_pair)
            diffusion_wall = _sum_by_particle(self_of_pair[neighbour_is_wall], diffusion_pair[neighbour_is_wall],
                                              fluid_count)
            diffusion_wall_pairwise = diffusion_wall.copy()
            # what the other choice of c^2 (c0^2 for every particle) would give: how far the two are apart on
            # this state, i.e. whether a comparison on it tells them apart
            with_rest_sound_speed = _sum_by_particle(
                self_of_pair[neighbour_is_wall],
                (diffusion_coefficient * self_density * (displacement @ gravity_vector) / sound_speed_squared_rest
                 * geometry_pair)[neighbour_is_wall], fluid_count)
            body_force_notes = {
                "wall_diffusion_maximum": float(np.abs(diffusion_wall).max()),
                "wall_diffusion_difference_with_rest_sound_speed_maximum":
                    float(np.abs(diffusion_wall - with_rest_sound_speed).max())}
        elif constants.delta_wall == "reflection":
            # the density of a wall particle = the continued density at its mirror point + the density difference
            # of a fluid at rest between the mirror point and the particle:
            #     psi_ij = G_i . (r_j - r_i) - (G_i - rho_i g / c_i^2) . (r_j - r_j')
            # The first part is delta_wall extension's term (above). The second, as the pass applies it: the
            # pair's (r_j - r_i) . grad W = |grad W| r, without the correction matrix. Nothing on a fallback of A_f.
            sound_speed_squared_rest = constants.power_parameter * constants.eos_constant / constants.rest_density
            fluid_density_of_particle = densities[fluid_index]
            sound_speed_squared = sound_speed_squared_rest * (
                fluid_density_of_particle / constants.rest_density) ** (constants.power_parameter - 1.0)
            density_gradient_excess = one_sided_density_gradient \
                - (fluid_density_of_particle / sound_speed_squared)[:, None] * gravity[None, :]
            density_gradient_excess[~fluid_valid] = 0.0
            gradient_norm = np.sqrt((gradient * gradient).sum(axis=1))
            reflection_vector = _sum_by_particle(
                self_of_pair[neighbour_is_wall],
                (volume * gradient_norm * distance / softened)[neighbour_is_wall, None]
                * mirror_displacement[neighbour_is_wall], fluid_count)
            diffusion_reflection = -diffusion_coefficient * (density_gradient_excess * reflection_vector).sum(axis=1)
            diffusion_wall = diffusion_wall + diffusion_reflection
            # pair by pair with the corrected gradient (what the two sums after the loop stand for)
            continued_difference = continued_difference \
                - (density_gradient_excess[self_of_pair] * mirror_displacement).sum(axis=1)
            pairwise = (diffusion_coefficient * continued_difference * volume / softened
                        * (displacement * corrected_gradient).sum(axis=1))
            diffusion_wall_pairwise = _sum_by_particle(self_of_pair[neighbour_is_wall], pairwise[neighbour_is_wall],
                                                       fluid_count)
        diffusion_scale = _sum_by_particle(self_of_pair[neighbour_is_fluid],
                                           np.abs(diffusion_pair[neighbour_is_fluid]), fluid_count) \
            + (np.abs(diffusion_wall) if constants.delta_wall in ("extension", "reflection") else
               _sum_by_particle(self_of_pair[neighbour_is_wall], np.abs(diffusion_pair[neighbour_is_wall]),
                                fluid_count))
    density_rate = drift_fluid + drift_wall + diffusion_fluid + diffusion_wall
    # numerics.volume_constraint: rho_i * rate for every fluid particle
    volume_constraint_term = densities[fluid_index] * constants.volume_constraint_rate
    density_rate = density_rate + volume_constraint_term
    wall_diffusion_scale = np.zeros(fluid_count)
    fluid_diffusion_scale = np.zeros(fluid_count)
    if constants.use_density_diffusion:
        fluid_diffusion_scale = _sum_by_particle(self_of_pair[neighbour_is_fluid],
                                                 np.abs(diffusion_pair[neighbour_is_fluid]), fluid_count)
        wall_diffusion_scale = _sum_by_particle(
            self_of_pair[neighbour_is_wall],
            np.abs(pairwise if constants.delta_wall in ("extension", "reflection")
                   else diffusion_pair)[neighbour_is_wall],
            fluid_count)
    term_scales = np.stack([
        _sum_by_particle(self_of_pair[neighbour_is_fluid], np.abs(drift_pair[neighbour_is_fluid]), fluid_count),
        _sum_by_particle(self_of_pair[neighbour_is_wall], np.abs(drift_pair[neighbour_is_wall]), fluid_count),
        fluid_diffusion_scale, wall_diffusion_scale], axis=1)
    density_rate_scale = _sum_by_particle(self_of_pair, np.abs(drift_pair), fluid_count) + diffusion_scale \
        + np.abs(volume_constraint_term)

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
    one_sided_pressure_gradient = np.einsum("fab,fb->fa", fluid_inverse, fluid_pressure_gradient_sum)
    one_sided_pressure_gradient[~fluid_valid] = 0.0
    # pressure_wall reflection: every wall neighbour's pressure is lowered by (G_i - rho_i g) . (r_j - r_j'), the
    # pair sum by L M_d (G_i - rho_i g), M_d = sum_wall V_j grad W (x) (r_j - r_j'); nothing on a fallback of A_f
    reflection_part = np.zeros((fluid_count, 3))
    pressure_gradient_excess = np.zeros((fluid_count, 3))
    if constants.pressure_wall == "reflection":
        pressure_gradient_excess = one_sided_pressure_gradient - fluid_density[:, None] * gravity[None, :]
        pressure_gradient_excess[~fluid_valid] = 0.0
        mirror_matrix = _sum_by_particle(
            self_of_pair[neighbour_is_wall],
            force_volume[neighbour_is_wall, None, None] * gradient[neighbour_is_wall, :, None]
            * mirror_displacement[neighbour_is_wall, None, :], fluid_count)
        reflection_part = -np.einsum("fab,fb->fa", correction,
                                     np.einsum("fab,fb->fa", mirror_matrix, pressure_gradient_excess))
    acceleration_pressure = pressure_fluid \
        - (wall_pressure_sum + gradient_part + reflection_part) / fluid_density[:, None]

    # the wall neighbours' pressure written pair by pair: p_j = p_i + G_i . (r_j - r_i), G_i = A_f^-1 g_f(p)
    # (pressure_wall reflection: - (G_i - rho_i g) . (r_j - r_j'))
    continued_pressure = self_pressure + (one_sided_pressure_gradient[self_of_pair] * displacement).sum(axis=1) \
        - (pressure_gradient_excess[self_of_pair] * mirror_displacement).sum(axis=1)
    pairwise_combined = np.where(plus_branch, continued_pressure + self_pressure, continued_pressure - self_pressure)
    pairwise_pair = -(force_volume * pairwise_combined / self_density)[:, None] * corrected_gradient
    acceleration_pressure_pairwise = pressure_fluid + _sum_by_particle(
        self_of_pair[neighbour_is_wall], pairwise_pair[neighbour_is_wall], fluid_count)

    morris_coefficient = 2.0 * (dimension + 2.0)
    # v_i - v_j of a pair: a wall neighbour's first-order continuation (v_i - v_w) (1 + a_j), or with
    # viscous_extension quadratic and a solved grad v_i the second-order one
    # (v_i - v_w) (1 + a_j)^2 + a_j grad v_i (r_j - r_i)
    viscous_velocity_difference = (-velocity_difference) * extrapolation[:, None]
    if constants.viscous_extension == "quadratic":
        viscous_velocity_difference = np.where(
            second_order_of_pair[:, None],
            (-velocity_difference) * quadratic_factor[:, None] + depth_ratio[:, None] * gradient_displacement,
            viscous_velocity_difference)
    viscous_scalar = (morris_coefficient * constants.viscosity * force_volume
                      * (viscous_velocity_difference * (-displacement)).sum(axis=1) / softened)
    viscous_pair = viscous_scalar[:, None] * corrected_gradient
    acceleration_viscosity = _sum_by_particle(self_of_pair, viscous_pair, fluid_count)
    acceleration = acceleration_pressure + acceleration_viscosity + gravity[None, :]
    acceleration_scale = (
        _sum_by_particle(self_of_pair[neighbour_is_fluid],
                         np.sqrt((pressure_pair[neighbour_is_fluid] ** 2).sum(axis=1)), fluid_count)
        + np.sqrt((wall_pressure_sum ** 2).sum(axis=1)) / fluid_density
        + np.sqrt((gradient_part ** 2).sum(axis=1)) / fluid_density
        + np.sqrt((reflection_part ** 2).sum(axis=1)) / fluid_density
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
        wall_neighbour_count=wall_neighbour_count,
        fluid_neighbour_count=np.bincount(self_of_pair[neighbour_is_fluid], minlength=fluid_count),
        notes={"pair_count": int(self_of_pair.size), **body_force_notes},
        density_rate_term_scales=term_scales,
        velocity_gradient=velocity_gradient,
        extension_state=extension_state,
        system_determinant=np.where(wall_neighbour_count > 0, system_determinant, 1.0),
        quadratic_factor_maximum=quadratic_factor_maximum,
        wall_viscous_diffusion_number=constants.timestep * wall_viscous_rate,
        acceleration_pressure_reflection=-reflection_part / fluid_density[:, None],
        density_rate_volume_constraint=volume_constraint_term,
        density_rate_diffusion_reflection=diffusion_reflection,
        extrapolation_factor_maximum=np.sqrt(quadratic_factor_maximum),
        acceleration_pressure_fluid=pressure_fluid,
        acceleration_viscosity_wall=_sum_by_particle(self_of_pair[neighbour_is_wall], viscous_pair[neighbour_is_wall],
                                                     fluid_count),
    )
