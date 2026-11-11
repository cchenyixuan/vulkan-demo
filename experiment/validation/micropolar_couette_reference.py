"""micropolar_couette_reference.py - closed-form steady circular Couette flow of a micropolar fluid (Eringen).

Flow between two concentric cylinders, v = v(r) e_theta, w = w(r) e_z (micro-rotation). Material constants
mu (shear_viscosity), kappa (coupling_viscosity), gamma (spin_viscosity); the density does not enter.

    momentum (theta):   (mu + kappa) (v'' + v'/r - v/r^2) - kappa w' = 0
    angular momentum:   gamma (w'' + w'/r) + kappa (v' + v/r - 2 w) = 0

Solution. v'' + v'/r - v/r^2 = d/dr (v' + v/r), so the momentum equation has the first integral
(mu + kappa) (v' + v/r) - kappa w = C1. Put into the angular equation it gives the modified Bessel equation
w'' + w'/r - (w - C1 / (2 mu + kappa)) / l^2 = 0 with

    l^2 = gamma (mu + kappa) / (kappa (2 mu + kappa))                          (characteristic_length l)

    w(r) = C1 / (2 mu + kappa) + A I0(r/l) + B K0(r/l)
    v(r) = C1 r / (2 mu + kappa) + C2 / r + (kappa l / (mu + kappa)) (A I1(r/l) - B K1(r/l))

(v from (r v)' = r (kappa w + C1) / (mu + kappa); the antiderivatives of x I0(x) and x K0(x) are x I1(x) and
-x K1(x)). C1 is the value of the first integral, C2 the strength of the potential vortex.
C1, C2, A, B follow from the four wall values v(R1), v(R2), w(R1), w(R2).

Torque per unit length through the circle of radius r (the moment of the outer fluid on the inner one):

    M = 2 pi (r^2 t + r gamma w'),    t = t_(r theta) = (mu + kappa) v' - mu v / r - kappa w
    M = -2 pi (2 mu + kappa) C2       for every r (the Bessel parts of r^2 t and of r gamma w' cancel)

Newtonian reference of viscosity mu_inf = mu + kappa / 2: v = alpha r + beta / r, M = -4 pi mu_inf beta, so
M / M_Newtonian = C2 / beta and the relative torque change is D = C2 / beta - 1. With w = alpha on both walls the
micropolar solution is the Newtonian one for every l (A = B = 0): half the vorticity of v = alpha r + beta / r is alpha
everywhere. For other wall micro-rotations w1, w2 the two wall layers of thickness l act as wall slips
+(kappa l / (mu + kappa)) (w1 - alpha) at R1 and -(kappa l / (mu + kappa)) (w2 - alpha) at R2, so for l << gap

    D ~ (kappa l / (mu + kappa)) R1 R2 (R2 (w1 - alpha) + R1 (w2 - alpha)) / ((R2^2 - R1^2) beta)   -> 0 like l.

Numerics. I0 and K0 differ by exp(2 r / l), so the constants are solved and stored in the scaled form

    scaled_first_kind_amplitude  = A exp(+R2 / l),    scaled I_n(r) = ive(n, r/l) exp(+(r - R2) / l)
    scaled_second_kind_amplitude = B exp(-R1 / l),    scaled K_n(r) = kve(n, r/l) exp(-(r - R1) / l)

(both scaled functions are of order 1 or smaller between the walls once r / l > 1). The 4 x 4 system then stays
well conditioned for any small l: condition number 32 to 43 for R1 = 0.4, R2 = 0.5 and l from 0.0001 to 0.05.
The other end is the weak one: for l larger than the radii the Bessel columns approach the algebraic ones
(I0 -> 1, I1 -> r / (2 l), K1 -> l / r) and digits are lost (wall values reproduced to 1e-12 at l = 20 R2, to
1e-8 at l = 2000 R2). The fit of C2 from a velocity profile alone degrades earlier, because K1 and 1 / r are
hard to tell apart over a narrow gap: condition number of the normalized velocity basis 23 at gap / l = 20,
490 at gap / l = 2, 2000 at gap / l = 1 (R1 = 0.4, R2 = 0.5). system_condition_number is returned with the
solution.

Solver parameters (docs/micropolar_design.md, materials/micropolar.yaml): viscosity nu = (mu + kappa) / rho,
viscosity_transfer nu_t = -mu / rho, coupling coefficient c = nu + nu_t = kappa / rho, viscosity_rotation
xi = gamma / rho, micro_inertia Theta.

Run the checks from the repository root:
    .venv/Scripts/python.exe -m experiment.validation.micropolar_couette_reference
"""
from __future__ import annotations

import dataclasses
import sys

import numpy as np
from scipy import special


@dataclasses.dataclass(frozen=True)
class MicropolarCouetteSolution:
    """The constants of one solution. first_integral_constant (C1, a stress) and potential_vortex_constant (C2,
    the coefficient of 1 / r in v) are the physical ones; the two Bessel amplitudes are stored scaled (module
    docstring)."""
    inner_radius: float
    outer_radius: float
    shear_viscosity: float
    coupling_viscosity: float
    spin_viscosity: float
    characteristic_length: float
    first_integral_constant: float            # C1 = (mu + kappa) (v' + v/r) - kappa w
    potential_vortex_constant: float          # C2; M = -2 pi (2 mu + kappa) C2
    scaled_first_kind_amplitude: float        # A exp(+outer_radius / l)
    scaled_second_kind_amplitude: float       # B exp(-inner_radius / l)
    system_condition_number: float            # of the equilibrated 4 x 4 system that was solved
    unequilibrated_system_condition_number: float   # of the same system before the row / column equilibration

    @property
    def background_rotation_rate(self) -> float:
        """C1 / (2 mu + kappa): the constant part of w and the coefficient of r in v."""
        return self.first_integral_constant / (2.0 * self.shear_viscosity + self.coupling_viscosity)

    @property
    def unscaled_first_kind_amplitude(self) -> float:
        """A of the docstring formula (underflows to 0 for outer_radius / l above about 745)."""
        return self.scaled_first_kind_amplitude * float(np.exp(-self.outer_radius / self.characteristic_length))

    @property
    def unscaled_second_kind_amplitude(self) -> float:
        """B of the docstring formula (overflows to inf for inner_radius / l above about 709)."""
        with np.errstate(over="ignore"):
            return self.scaled_second_kind_amplitude * float(np.exp(self.inner_radius / self.characteristic_length))


@dataclasses.dataclass(frozen=True)
class EringenParameters:
    """mu, kappa, gamma and l of Eringen's notation."""
    shear_viscosity: float
    coupling_viscosity: float
    spin_viscosity: float
    characteristic_length: float


@dataclasses.dataclass(frozen=True)
class SolverParameters:
    """The micropolar keys of the SPH solver: nu, nu_t, c = nu + nu_t, xi and Theta."""
    viscosity: float
    viscosity_transfer: float
    coupling_coefficient: float
    viscosity_rotation: float
    micro_inertia: float


# ---------------------------------------------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------------------------------------------
def characteristic_length_from_viscosities(shear_viscosity: float, coupling_viscosity: float,
                                           spin_viscosity: float) -> float:
    """l = sqrt(gamma (mu + kappa) / (kappa (2 mu + kappa))). Needs kappa > 0, gamma > 0 and 2 mu + kappa > 0."""
    if not (coupling_viscosity > 0.0 and spin_viscosity > 0.0 and 2.0 * shear_viscosity + coupling_viscosity > 0.0):
        raise ValueError("the closed form needs coupling_viscosity > 0, spin_viscosity > 0 and "
                         "2 shear_viscosity + coupling_viscosity > 0, got "
                         f"{shear_viscosity}, {coupling_viscosity}, {spin_viscosity}")
    return float(np.sqrt(spin_viscosity * (shear_viscosity + coupling_viscosity)
                         / (coupling_viscosity * (2.0 * shear_viscosity + coupling_viscosity))))


def eringen_parameters_from_solver_parameters(density: float, viscosity: float, coupling_coefficient: float,
                                              viscosity_rotation: float) -> EringenParameters:
    """Solver -> Eringen: nu = (mu + kappa) / rho, c = kappa / rho, xi = gamma / rho give
    mu = rho (nu - c), kappa = rho c, gamma = rho xi and l^2 = xi nu / (c (2 nu - c))."""
    shear_viscosity = density * (viscosity - coupling_coefficient)
    coupling_viscosity = density * coupling_coefficient
    spin_viscosity = density * viscosity_rotation
    return EringenParameters(shear_viscosity=shear_viscosity, coupling_viscosity=coupling_viscosity,
                             spin_viscosity=spin_viscosity,
                             characteristic_length=characteristic_length_from_viscosities(
                                 shear_viscosity, coupling_viscosity, spin_viscosity))


def solver_parameters_from_design(newtonian_limit_kinematic_viscosity: float,
                                  coupling_to_shear_viscosity_ratio: float,
                                  characteristic_length: float) -> SolverParameters:
    """Design -> solver: nu_inf = (mu + kappa / 2) / rho, K = kappa / mu and l give
    mu / rho = nu_inf / (1 + K / 2), nu = (1 + K) mu / rho, nu_t = -mu / rho, c = K mu / rho,
    xi = l^2 c (2 nu - c) / nu and Theta = l^2."""
    shear_kinematic_viscosity = newtonian_limit_kinematic_viscosity / (1.0 + 0.5 * coupling_to_shear_viscosity_ratio)
    coupling_coefficient = coupling_to_shear_viscosity_ratio * shear_kinematic_viscosity
    viscosity = shear_kinematic_viscosity + coupling_coefficient
    micro_inertia = characteristic_length ** 2
    return SolverParameters(viscosity=viscosity, viscosity_transfer=-shear_kinematic_viscosity,
                            coupling_coefficient=coupling_coefficient,
                            viscosity_rotation=micro_inertia * coupling_coefficient
                            * (2.0 * viscosity - coupling_coefficient) / viscosity,
                            micro_inertia=micro_inertia)


# ---------------------------------------------------------------------------------------------------------------
# basis functions (for the wall system and for least-squares fits of measured profiles)
# ---------------------------------------------------------------------------------------------------------------
def _scaled_modified_bessel_functions(radius, characteristic_length: float, inner_reference_radius: float,
                                      outer_reference_radius: float):
    """(I0, I1, K0, K1)(r / l), I_n times exp(-outer_reference_radius / l) and K_n times
    exp(+inner_reference_radius / l)."""
    radius = np.asarray(radius, dtype=np.float64)
    argument = radius / characteristic_length
    growth = np.exp((radius - outer_reference_radius) / characteristic_length)
    decay = np.exp(-(radius - inner_reference_radius) / characteristic_length)
    return (special.ive(0, argument) * growth, special.ive(1, argument) * growth,
            special.kve(0, argument) * decay, special.kve(1, argument) * decay)


def velocity_basis_functions(radius, characteristic_length: float, inner_reference_radius: float,
                             outer_reference_radius: float) -> np.ndarray:
    """(n, 4) columns [r, 1 / r, scaled I1(r / l), scaled K1(r / l)] at the n radii. A linear least-squares fit of a
    measured v_theta(r) on these columns returns the coefficients
        [C1 / (2 mu + kappa),  C2,  +(kappa l / (mu + kappa)) scaled_first_kind_amplitude,
                                    -(kappa l / (mu + kappa)) scaled_second_kind_amplitude],
    so the coefficient of the 1 / r column (index 1) is C2 (torque M = -2 pi (2 mu + kappa) C2). Only l enters,
    not the viscosities; the reference radii only set the scale of the last two coefficients."""
    radius = np.asarray(radius, dtype=np.float64).reshape(-1)
    _, first_kind_order_one, _, second_kind_order_one = _scaled_modified_bessel_functions(
        radius, characteristic_length, inner_reference_radius, outer_reference_radius)
    return np.stack([radius, 1.0 / radius, first_kind_order_one, second_kind_order_one], axis=1)


def micro_rotation_basis_functions(radius, characteristic_length: float, inner_reference_radius: float,
                                   outer_reference_radius: float) -> np.ndarray:
    """(n, 3) columns [1, scaled I0(r / l), scaled K0(r / l)] at the n radii. A fit of a measured w_z(r) returns
    [C1 / (2 mu + kappa), scaled_first_kind_amplitude, scaled_second_kind_amplitude]."""
    radius = np.asarray(radius, dtype=np.float64).reshape(-1)
    first_kind_order_zero, _, second_kind_order_zero, _ = _scaled_modified_bessel_functions(
        radius, characteristic_length, inner_reference_radius, outer_reference_radius)
    return np.stack([np.ones_like(radius), first_kind_order_zero, second_kind_order_zero], axis=1)


# ---------------------------------------------------------------------------------------------------------------
# the four constants
# ---------------------------------------------------------------------------------------------------------------
def _equilibrate(system: np.ndarray, sweep_count: int = 12):
    """Row and column scales (Ruiz sweeps) that bring every row and column maximum of the system to 1, so that the
    condition number does not depend on the units of the radii or on the size of the scaled Bessel values."""
    row_scale = np.ones(system.shape[0])
    column_scale = np.ones(system.shape[1])
    for _ in range(sweep_count):
        scaled = np.abs(system) * row_scale[:, None] * column_scale[None, :]
        row_scale /= np.sqrt(scaled.max(axis=1))
        column_scale /= np.sqrt(scaled.max(axis=0))
    return row_scale, column_scale


def solve_micropolar_couette(inner_radius: float, outer_radius: float, shear_viscosity: float,
                             coupling_viscosity: float, spin_viscosity: float, inner_velocity: float,
                             outer_velocity: float, inner_micro_rotation: float,
                             outer_micro_rotation: float) -> MicropolarCouetteSolution:
    """The constants for the wall values v(R1), v(R2), w(R1), w(R2) (any four values).

    Unknowns [C1 / (2 mu + kappa), C2, A exp(R2 / l), B exp(-R1 / l)]; rows v(R1), v(R2), w(R1), w(R2) built from
    velocity_basis_functions / micro_rotation_basis_functions, equilibrated and solved by LU."""
    if not 0.0 < inner_radius < outer_radius:
        raise ValueError(f"need 0 < inner_radius < outer_radius, got {inner_radius}, {outer_radius}")
    length = characteristic_length_from_viscosities(shear_viscosity, coupling_viscosity, spin_viscosity)
    bessel_velocity_factor = coupling_viscosity * length / (shear_viscosity + coupling_viscosity)
    wall_radii = np.array([inner_radius, outer_radius], dtype=np.float64)
    velocity_rows = velocity_basis_functions(wall_radii, length, inner_radius, outer_radius)
    micro_rotation_rows = micro_rotation_basis_functions(wall_radii, length, inner_radius, outer_radius)

    system = np.zeros((4, 4))
    system[0:2, 0] = velocity_rows[:, 0]
    system[0:2, 1] = velocity_rows[:, 1]
    system[0:2, 2] = bessel_velocity_factor * velocity_rows[:, 2]
    system[0:2, 3] = -bessel_velocity_factor * velocity_rows[:, 3]
    system[2:4, 0] = micro_rotation_rows[:, 0]
    system[2:4, 2] = micro_rotation_rows[:, 1]
    system[2:4, 3] = micro_rotation_rows[:, 2]
    right_hand_side = np.array([inner_velocity, outer_velocity, inner_micro_rotation, outer_micro_rotation],
                               dtype=np.float64)

    row_scale, column_scale = _equilibrate(system)
    equilibrated_system = system * row_scale[:, None] * column_scale[None, :]
    unknowns = column_scale * np.linalg.solve(equilibrated_system, row_scale * right_hand_side)
    return MicropolarCouetteSolution(
        inner_radius=float(inner_radius), outer_radius=float(outer_radius),
        shear_viscosity=float(shear_viscosity), coupling_viscosity=float(coupling_viscosity),
        spin_viscosity=float(spin_viscosity), characteristic_length=length,
        first_integral_constant=float(unknowns[0] * (2.0 * shear_viscosity + coupling_viscosity)),
        potential_vortex_constant=float(unknowns[1]),
        scaled_first_kind_amplitude=float(unknowns[2]), scaled_second_kind_amplitude=float(unknowns[3]),
        system_condition_number=float(np.linalg.cond(equilibrated_system)),
        unequilibrated_system_condition_number=float(np.linalg.cond(system)))


# ---------------------------------------------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------------------------------------------
def _bessel_parts(solution: MicropolarCouetteSolution, radius: np.ndarray):
    """(A I0, A I1, B K0, B K1)(r / l) from the scaled amplitudes and the scaled functions."""
    first_kind_order_zero, first_kind_order_one, second_kind_order_zero, second_kind_order_one = \
        _scaled_modified_bessel_functions(radius, solution.characteristic_length, solution.inner_radius,
                                          solution.outer_radius)
    return (solution.scaled_first_kind_amplitude * first_kind_order_zero,
            solution.scaled_first_kind_amplitude * first_kind_order_one,
            solution.scaled_second_kind_amplitude * second_kind_order_zero,
            solution.scaled_second_kind_amplitude * second_kind_order_one)


def _coupling_fraction(solution: MicropolarCouetteSolution) -> float:
    """kappa / (mu + kappa)."""
    return solution.coupling_viscosity / (solution.shear_viscosity + solution.coupling_viscosity)


def micro_rotation(solution: MicropolarCouetteSolution, radius) -> np.ndarray:
    """w(r) = C1 / (2 mu + kappa) + A I0(r/l) + B K0(r/l)."""
    radius = np.asarray(radius, dtype=np.float64)
    first_kind_zero, _, second_kind_zero, _ = _bessel_parts(solution, radius)
    return solution.background_rotation_rate + first_kind_zero + second_kind_zero


def micro_rotation_derivative(solution: MicropolarCouetteSolution, radius) -> np.ndarray:
    """w'(r) = (A I1(r/l) - B K1(r/l)) / l."""
    radius = np.asarray(radius, dtype=np.float64)
    _, first_kind_one, _, second_kind_one = _bessel_parts(solution, radius)
    return (first_kind_one - second_kind_one) / solution.characteristic_length


def micro_rotation_second_derivative(solution: MicropolarCouetteSolution, radius) -> np.ndarray:
    """w''(r) = (A (I0 - (l/r) I1) + B (K0 + (l/r) K1)) / l^2   (I1' = I0 - I1/x, K1' = -K0 - K1/x)."""
    radius = np.asarray(radius, dtype=np.float64)
    length = solution.characteristic_length
    first_kind_zero, first_kind_one, second_kind_zero, second_kind_one = _bessel_parts(solution, radius)
    return (first_kind_zero - first_kind_one * length / radius
            + second_kind_zero + second_kind_one * length / radius) / length ** 2


def velocity(solution: MicropolarCouetteSolution, radius) -> np.ndarray:
    """v(r) = C1 r / (2 mu + kappa) + C2 / r + (kappa l / (mu + kappa)) (A I1(r/l) - B K1(r/l))."""
    radius = np.asarray(radius, dtype=np.float64)
    _, first_kind_one, _, second_kind_one = _bessel_parts(solution, radius)
    return (solution.background_rotation_rate * radius + solution.potential_vortex_constant / radius
            + _coupling_fraction(solution) * solution.characteristic_length * (first_kind_one - second_kind_one))


def velocity_derivative(solution: MicropolarCouetteSolution, radius) -> np.ndarray:
    """v'(r) = C1 / (2 mu + kappa) - C2 / r^2 + (kappa / (mu + kappa)) (A (I0 - (l/r) I1) + B (K0 + (l/r) K1))."""
    radius = np.asarray(radius, dtype=np.float64)
    length = solution.characteristic_length
    first_kind_zero, first_kind_one, second_kind_zero, second_kind_one = _bessel_parts(solution, radius)
    return (solution.background_rotation_rate - solution.potential_vortex_constant / radius ** 2
            + _coupling_fraction(solution)
            * (first_kind_zero - first_kind_one * length / radius
               + second_kind_zero + second_kind_one * length / radius))


def velocity_second_derivative(solution: MicropolarCouetteSolution, radius) -> np.ndarray:
    """v''(r) = 2 C2 / r^3 + (kappa / (mu + kappa)) (A (I1 / l - I0 / r + 2 l I1 / r^2)
                                                     - B (K1 / l + K0 / r + 2 l K1 / r^2))."""
    radius = np.asarray(radius, dtype=np.float64)
    length = solution.characteristic_length
    first_kind_zero, first_kind_one, second_kind_zero, second_kind_one = _bessel_parts(solution, radius)
    return (2.0 * solution.potential_vortex_constant / radius ** 3
            + _coupling_fraction(solution)
            * (first_kind_one / length - first_kind_zero / radius + 2.0 * length * first_kind_one / radius ** 2
               - second_kind_one / length - second_kind_zero / radius - 2.0 * length * second_kind_one / radius ** 2))


def shear_stress(solution: MicropolarCouetteSolution, radius) -> np.ndarray:
    """t = t_(r theta) = (mu + kappa) v' - mu v / r - kappa w (force per area in theta on the surface of normal r)."""
    radius = np.asarray(radius, dtype=np.float64)
    return ((solution.shear_viscosity + solution.coupling_viscosity) * velocity_derivative(solution, radius)
            - solution.shear_viscosity * velocity(solution, radius) / radius
            - solution.coupling_viscosity * micro_rotation(solution, radius))


def torque_per_unit_length(solution: MicropolarCouetteSolution, radius) -> np.ndarray:
    """M(r) = 2 pi (r^2 t + r gamma w'), evaluated from this definition (force stress + couple stress); it equals
    torque_per_unit_length_from_constant(solution) at every r up to rounding."""
    radius = np.asarray(radius, dtype=np.float64)
    return 2.0 * np.pi * (radius ** 2 * shear_stress(solution, radius)
                          + radius * solution.spin_viscosity * micro_rotation_derivative(solution, radius))


def torque_per_unit_length_from_constant(solution: MicropolarCouetteSolution) -> float:
    """M = -2 pi (2 mu + kappa) C2."""
    return (-2.0 * np.pi * (2.0 * solution.shear_viscosity + solution.coupling_viscosity)
            * solution.potential_vortex_constant)


# ---------------------------------------------------------------------------------------------------------------
# Newtonian reference and the relative torque change
# ---------------------------------------------------------------------------------------------------------------
def newtonian_couette_constants(inner_radius: float, outer_radius: float, inner_velocity: float,
                                outer_velocity: float) -> tuple[float, float]:
    """(alpha, beta) of v = alpha r + beta / r with v(R1), v(R2) given. alpha is also half the vorticity."""
    difference_of_squares = outer_radius ** 2 - inner_radius ** 2
    rigid_rotation_rate = (outer_radius * outer_velocity - inner_radius * inner_velocity) / difference_of_squares
    potential_vortex_constant = (inner_radius * outer_radius
                                 * (outer_radius * inner_velocity - inner_radius * outer_velocity)
                                 / difference_of_squares)
    return float(rigid_rotation_rate), float(potential_vortex_constant)


def newtonian_couette_velocity(radius, inner_radius: float, outer_radius: float, inner_velocity: float,
                               outer_velocity: float) -> np.ndarray:
    """v(r) = alpha r + beta / r."""
    radius = np.asarray(radius, dtype=np.float64)
    rigid_rotation_rate, potential_vortex_constant = newtonian_couette_constants(
        inner_radius, outer_radius, inner_velocity, outer_velocity)
    return rigid_rotation_rate * radius + potential_vortex_constant / radius


def newtonian_torque_per_unit_length(dynamic_viscosity: float, inner_radius: float, outer_radius: float,
                                     inner_velocity: float, outer_velocity: float) -> float:
    """M = -4 pi mu beta (same sign convention as torque_per_unit_length)."""
    _, potential_vortex_constant = newtonian_couette_constants(inner_radius, outer_radius, inner_velocity,
                                                               outer_velocity)
    return -4.0 * np.pi * dynamic_viscosity * potential_vortex_constant


def relative_torque_change_from_constant(potential_vortex_constant: float, inner_radius: float, outer_radius: float,
                                         inner_velocity: float, outer_velocity: float) -> float:
    """D = C2 / beta - 1 for a C2 from solve_micropolar_couette or from a fit on velocity_basis_functions:
    the torque relative to the Newtonian fluid of viscosity mu + kappa / 2 with the same radii and wall velocities."""
    _, newtonian_potential_vortex_constant = newtonian_couette_constants(inner_radius, outer_radius, inner_velocity,
                                                                         outer_velocity)
    if newtonian_potential_vortex_constant == 0.0:
        raise ValueError("the Newtonian torque is zero for these wall velocities (rigid rotation): D is undefined")
    return potential_vortex_constant / newtonian_potential_vortex_constant - 1.0


def relative_torque_change(inner_radius: float, outer_radius: float, shear_viscosity: float,
                           coupling_viscosity: float, spin_viscosity: float, inner_velocity: float,
                           outer_velocity: float, inner_micro_rotation: float, outer_micro_rotation: float) -> float:
    """D = M / M_Newtonian(mu + kappa / 2) - 1 = C2 / beta - 1 of the closed-form solution."""
    solution = solve_micropolar_couette(inner_radius, outer_radius, shear_viscosity, coupling_viscosity,
                                        spin_viscosity, inner_velocity, outer_velocity, inner_micro_rotation,
                                        outer_micro_rotation)
    return relative_torque_change_from_constant(solution.potential_vortex_constant, inner_radius, outer_radius,
                                                inner_velocity, outer_velocity)


# ---------------------------------------------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------------------------------------------
def equation_residuals(solution: MicropolarCouetteSolution, radius) -> tuple[np.ndarray, np.ndarray]:
    """Residuals of the two equations with the analytic derivatives, each divided by the sum of the absolute values
    of its individual terms ((mu + kappa) v'', (mu + kappa) v'/r, (mu + kappa) v/r^2, kappa w' and gamma w'',
    gamma w'/r, kappa v', kappa v/r, 2 kappa w) at the same radius."""
    radius = np.asarray(radius, dtype=np.float64)
    total_viscosity = solution.shear_viscosity + solution.coupling_viscosity
    momentum_terms = [total_viscosity * velocity_second_derivative(solution, radius),
                      total_viscosity * velocity_derivative(solution, radius) / radius,
                      -total_viscosity * velocity(solution, radius) / radius ** 2,
                      -solution.coupling_viscosity * micro_rotation_derivative(solution, radius)]
    angular_terms = [solution.spin_viscosity * micro_rotation_second_derivative(solution, radius),
                     solution.spin_viscosity * micro_rotation_derivative(solution, radius) / radius,
                     solution.coupling_viscosity * velocity_derivative(solution, radius),
                     solution.coupling_viscosity * velocity(solution, radius) / radius,
                     -2.0 * solution.coupling_viscosity * micro_rotation(solution, radius)]
    return (np.abs(sum(momentum_terms)) / sum(np.abs(term) for term in momentum_terms),
            np.abs(sum(angular_terms)) / sum(np.abs(term) for term in angular_terms))


def _random_case(random_generator: np.random.Generator, radius_over_length=None) -> dict:
    """Random radii, viscosities and wall values; l log-uniform in [R2 / 500, 2 R2] unless R2 / l is given."""
    inner_radius = random_generator.uniform(0.2, 1.0)
    outer_radius = inner_radius + random_generator.uniform(0.05, 0.6)
    shear_viscosity = 10.0 ** random_generator.uniform(-3.0, 1.0)
    coupling_viscosity = shear_viscosity * 10.0 ** random_generator.uniform(-1.0, 1.0)
    if radius_over_length is None:
        radius_over_length = 10.0 ** random_generator.uniform(np.log10(0.5), np.log10(500.0))
    length = outer_radius / radius_over_length
    spin_viscosity = (length ** 2 * coupling_viscosity * (2.0 * shear_viscosity + coupling_viscosity)
                      / (shear_viscosity + coupling_viscosity))
    wall_values = random_generator.uniform(-3.0, 3.0, size=4)
    return dict(inner_radius=inner_radius, outer_radius=outer_radius, shear_viscosity=shear_viscosity,
                coupling_viscosity=coupling_viscosity, spin_viscosity=spin_viscosity,
                inner_velocity=wall_values[0], outer_velocity=wall_values[1],
                inner_micro_rotation=wall_values[2], outer_micro_rotation=wall_values[3])


def _integrate_boundary_value_problem(case: dict, tolerance: float = 1.0e-9):
    """The two equations integrated by scipy.integrate.solve_bvp as a first-order system in (v, v', w, w'): no Bessel
    function and none of the constants of the closed form enter. (At tol 1e-10 the mesh refinement of some cases
    never ends: the residual estimate reaches its rounding floor.)"""
    from scipy.integrate import solve_bvp

    total_viscosity = case["shear_viscosity"] + case["coupling_viscosity"]
    coupling_viscosity, spin_viscosity = case["coupling_viscosity"], case["spin_viscosity"]

    def right_hand_side(radius, state):
        return np.stack([state[1],
                         -state[1] / radius + state[0] / radius ** 2 + coupling_viscosity * state[3] / total_viscosity,
                         state[3],
                         -state[3] / radius
                         - coupling_viscosity * (state[1] + state[0] / radius - 2.0 * state[2]) / spin_viscosity])

    def boundary_residual(inner_state, outer_state):
        return np.array([inner_state[0] - case["inner_velocity"], outer_state[0] - case["outer_velocity"],
                         inner_state[2] - case["inner_micro_rotation"],
                         outer_state[2] - case["outer_micro_rotation"]])

    mesh = np.linspace(case["inner_radius"], case["outer_radius"], 401)
    fraction = (mesh - mesh[0]) / (mesh[-1] - mesh[0])
    guess = np.zeros((4, mesh.size))
    guess[0] = case["inner_velocity"] + (case["outer_velocity"] - case["inner_velocity"]) * fraction
    guess[2] = case["inner_micro_rotation"] + (case["outer_micro_rotation"] - case["inner_micro_rotation"]) * fraction
    result = solve_bvp(right_hand_side, boundary_residual, mesh, guess, tol=tolerance, max_nodes=100000)
    if not result.success:
        raise RuntimeError(f"solve_bvp failed: {result.message}")
    return result


def _test_case(length: float, inner_micro_rotation: float) -> dict:
    """R1 = 0.4, R2 = 0.5, rho = 1000, nu_inf = 0.01, K = 3, inner wall 1 m/s, outer wall at rest with w = 0."""
    design = solver_parameters_from_design(0.01, 3.0, length)
    eringen = eringen_parameters_from_solver_parameters(1000.0, design.viscosity, design.coupling_coefficient,
                                                        design.viscosity_rotation)
    return dict(inner_radius=0.4, outer_radius=0.5, shear_viscosity=eringen.shear_viscosity,
                coupling_viscosity=eringen.coupling_viscosity, spin_viscosity=eringen.spin_viscosity,
                inner_velocity=1.0, outer_velocity=0.0, inner_micro_rotation=inner_micro_rotation,
                outer_micro_rotation=0.0)


def run_checks() -> bool:
    """Run every check, print one line per check, return True if all passed."""
    random_generator = np.random.default_rng(20261111)
    all_passed = True

    def report(name: str, passed: bool, text: str):
        nonlocal all_passed
        all_passed = all_passed and bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {text}")

    # (a1) residuals of both equations, analytic derivatives, random wall values, R2 / l from 0.5 to 500
    worst_momentum = worst_angular = worst_wall = worst_first_integral = 0.0
    cases = [_random_case(random_generator) for _ in range(200)]
    cases += [_random_case(random_generator, radius_over_length=500.0) for _ in range(20)]
    for case in cases:
        solution = solve_micropolar_couette(**case)
        radius = np.linspace(case["inner_radius"], case["outer_radius"], 401)
        momentum_residual, angular_residual = equation_residuals(solution, radius)
        worst_momentum = max(worst_momentum, float(momentum_residual.max()))
        worst_angular = max(worst_angular, float(angular_residual.max()))
        walls = np.array([case["inner_radius"], case["outer_radius"]])
        wall_error = np.concatenate([
            velocity(solution, walls) - [case["inner_velocity"], case["outer_velocity"]],
            micro_rotation(solution, walls) - [case["inner_micro_rotation"], case["outer_micro_rotation"]]])
        worst_wall = max(worst_wall, float(np.abs(wall_error).max()))
        total_viscosity = case["shear_viscosity"] + case["coupling_viscosity"]
        first_integral_terms = [total_viscosity * velocity_derivative(solution, radius),
                                total_viscosity * velocity(solution, radius) / radius,
                                -case["coupling_viscosity"] * micro_rotation(solution, radius)]
        first_integral_error = (np.abs(sum(first_integral_terms) - solution.first_integral_constant)
                                / sum(np.abs(term) for term in first_integral_terms))
        worst_first_integral = max(worst_first_integral, float(first_integral_error.max()))
    report("a1 equation residuals (analytic derivatives, 220 random cases, R2/l 0.5..500)",
           worst_momentum < 1e-11 and worst_angular < 1e-11 and worst_wall < 1e-11 and worst_first_integral < 1e-11,
           f"max relative residual momentum {worst_momentum:.2e}, angular {worst_angular:.2e}; "
           f"wall values reproduced to {worst_wall:.2e} (absolute, values up to 3); "
           f"(mu+kappa)(v'+v/r) - kappa w = C1 to {worst_first_integral:.2e}")

    # (a2) the scaled evaluation is the literal formula with unscaled I0, I1, K0, K1 (moderate r / l only)
    worst_literal = 0.0
    for _ in range(50):
        case = _random_case(random_generator,
                            radius_over_length=10.0 ** random_generator.uniform(np.log10(0.5), np.log10(100.0)))
        solution = solve_micropolar_couette(**case)
        radius = np.linspace(case["inner_radius"], case["outer_radius"], 101)
        length = solution.characteristic_length
        doubled_viscosity = 2.0 * case["shear_viscosity"] + case["coupling_viscosity"]
        first_amplitude = solution.unscaled_first_kind_amplitude
        second_amplitude = solution.unscaled_second_kind_amplitude
        literal_micro_rotation_terms = [np.full_like(radius, solution.first_integral_constant / doubled_viscosity),
                                        first_amplitude * special.iv(0, radius / length),
                                        second_amplitude * special.kv(0, radius / length)]
        literal_velocity_terms = [solution.first_integral_constant * radius / doubled_viscosity,
                                  solution.potential_vortex_constant / radius,
                                  _coupling_fraction(solution) * length * first_amplitude
                                  * special.iv(1, radius / length),
                                  -_coupling_fraction(solution) * length * second_amplitude
                                  * special.kv(1, radius / length)]
        worst_literal = max(
            worst_literal,
            float((np.abs(sum(literal_micro_rotation_terms) - micro_rotation(solution, radius))
                   / sum(np.abs(term) for term in literal_micro_rotation_terms)).max()),
            float((np.abs(sum(literal_velocity_terms) - velocity(solution, radius))
                   / sum(np.abs(term) for term in literal_velocity_terms)).max()))
    report("a2 scaled evaluation = literal formula with iv / kv (50 random cases, R2/l 0.5..100)",
           worst_literal < 1e-12, f"max relative difference {worst_literal:.2e}")

    # (a3) the four derivative functions against 8th-order central differences of v and w
    first_weights = np.array([4.0 / 5.0, -1.0 / 5.0, 4.0 / 105.0, -1.0 / 280.0])
    second_weights = np.array([8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0])
    worst_difference = 0.0
    for _ in range(50):
        case = _random_case(random_generator,
                            radius_over_length=10.0 ** random_generator.uniform(np.log10(0.5), np.log10(100.0)))
        solution = solve_micropolar_couette(**case)
        step = 0.02 * min(solution.characteristic_length, case["inner_radius"])
        radius = np.linspace(case["inner_radius"] + 4.0 * step, case["outer_radius"] - 4.0 * step, 101)
        for function, derivative, second_derivative in (
                (velocity, velocity_derivative, velocity_second_derivative),
                (micro_rotation, micro_rotation_derivative, micro_rotation_second_derivative)):
            scale = np.abs(function(solution, radius)).max()
            difference_first = sum(first_weights[index] * (function(solution, radius + (index + 1) * step)
                                                           - function(solution, radius - (index + 1) * step))
                                   for index in range(4)) / step
            difference_second = (-205.0 / 72.0 * function(solution, radius)
                                 + sum(second_weights[index] * (function(solution, radius + (index + 1) * step)
                                                                + function(solution, radius - (index + 1) * step))
                                       for index in range(4))) / step ** 2
            worst_difference = max(
                worst_difference,
                float(np.abs(difference_first - derivative(solution, radius)).max() * step / scale),
                float(np.abs(difference_second - second_derivative(solution, radius)).max() * step ** 2 / scale))
    report("a3 analytic derivatives against 8th-order central differences (50 random cases)",
           worst_difference < 1e-9,
           f"max |difference| of v', w' (times step) and v'', w'' (times step^2) over max |function| "
           f"{worst_difference:.2e}")

    # (a4) independent integration of the equations (solve_bvp), profiles and torque
    worst_profile = worst_torque = 0.0
    boundary_value_cases = [_random_case(random_generator,
                                         radius_over_length=10.0 ** random_generator.uniform(np.log10(1.0),
                                                                                             np.log10(60.0)))
                            for _ in range(6)]
    boundary_value_cases += [_test_case(0.05, 2.5), _test_case(0.01, 2.5), _test_case(0.01, 0.0)]
    for case in boundary_value_cases:
        solution = solve_micropolar_couette(**case)
        result = _integrate_boundary_value_problem(case)
        radius = np.linspace(case["inner_radius"], case["outer_radius"], 301)
        state = result.sol(radius)
        closed_form = [velocity(solution, radius), velocity_derivative(solution, radius),
                       micro_rotation(solution, radius), micro_rotation_derivative(solution, radius)]
        for row_index in range(4):
            worst_profile = max(worst_profile, float(np.abs(state[row_index] - closed_form[row_index]).max()
                                                     / np.abs(closed_form[row_index]).max()))
        integrated_torque = 2.0 * np.pi * (
            radius ** 2 * ((case["shear_viscosity"] + case["coupling_viscosity"]) * state[1]
                           - case["shear_viscosity"] * state[0] / radius - case["coupling_viscosity"] * state[2])
            + radius * case["spin_viscosity"] * state[3])
        worst_torque = max(worst_torque, float(np.abs(integrated_torque / torque_per_unit_length_from_constant(solution)
                                                      - 1.0).max()))
    report("a4 closed form against scipy.integrate.solve_bvp (6 random cases + 3 test cases, tol 1e-9)",
           worst_profile < 1e-9 and worst_torque < 1e-9,
           f"max relative difference of v, v', w, w' {worst_profile:.2e}; "
           f"M(r) of the integrated solution / (-2 pi (2 mu + kappa) C2) - 1 max {worst_torque:.2e}")

    # (b) rigid rotation
    worst_velocity = worst_micro_rotation = worst_torque = 0.0
    for case_index in range(60):
        case = _random_case(random_generator, radius_over_length=500.0 if case_index < 10 else None)
        rotation_rate = random_generator.uniform(-3.0, 3.0)
        case.update(inner_velocity=rotation_rate * case["inner_radius"],
                    outer_velocity=rotation_rate * case["outer_radius"],
                    inner_micro_rotation=rotation_rate, outer_micro_rotation=rotation_rate)
        solution = solve_micropolar_couette(**case)
        radius = np.linspace(case["inner_radius"], case["outer_radius"], 201)
        worst_velocity = max(worst_velocity, float(np.abs(velocity(solution, radius) / (rotation_rate * radius)
                                                          - 1.0).max()))
        worst_micro_rotation = max(worst_micro_rotation,
                                   float(np.abs(micro_rotation(solution, radius) / rotation_rate - 1.0).max()))
        # scale: the Newtonian torque of the same fluid with the outer wall at rest
        torque_scale = abs(newtonian_torque_per_unit_length(
            case["shear_viscosity"] + 0.5 * case["coupling_viscosity"], case["inner_radius"], case["outer_radius"],
            case["inner_velocity"], 0.0))
        worst_torque = max(worst_torque,
                           float(np.abs(torque_per_unit_length(solution, radius)).max() / torque_scale),
                           abs(torque_per_unit_length_from_constant(solution)) / torque_scale)
    report("b rigid rotation v = Omega r, w = Omega on both walls (60 random cases, 10 at R2/l = 500)",
           worst_velocity < 1e-12 and worst_micro_rotation < 1e-12 and worst_torque < 1e-11,
           f"max |v / (Omega r) - 1| {worst_velocity:.2e}, max |w / Omega - 1| {worst_micro_rotation:.2e}, "
           f"max |M| / |M Newtonian with the outer wall at rest| {worst_torque:.2e}")

    # (c) torque independent of r and equal to -2 pi (2 mu + kappa) C2
    worst_spread = worst_constant = 0.0
    for case in cases:
        solution = solve_micropolar_couette(**case)
        radius = np.linspace(case["inner_radius"], case["outer_radius"], 401)
        torque = torque_per_unit_length(solution, radius)
        # scale: the sizes of the two parts of M, so that a case with C2 near zero does not blow the ratio up
        torque_scale = float(2.0 * np.pi * (np.abs(radius ** 2 * shear_stress(solution, radius))
                                            + np.abs(radius * solution.spin_viscosity
                                                     * micro_rotation_derivative(solution, radius))).max())
        worst_spread = max(worst_spread, float((torque.max() - torque.min()) / torque_scale))
        worst_constant = max(worst_constant,
                             float(np.abs(torque - torque_per_unit_length_from_constant(solution)).max()
                                   / torque_scale))
    report("c torque M(r) = 2 pi (r^2 t + r gamma w') (the 220 cases of a1, 401 radii each)",
           worst_spread < 1e-11 and worst_constant < 1e-11,
           f"max (max M - min M) / scale {worst_spread:.2e}, max |M(r) + 2 pi (2 mu + kappa) C2| / scale "
           f"{worst_constant:.2e} (scale = max over r of 2 pi (|r^2 t| + |r gamma w'|))")

    # (d1) wall micro-rotation = Newtonian rotation rate alpha: the Newtonian solution of viscosity mu + kappa / 2
    lengths = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    inner_radius, outer_radius, inner_velocity, outer_velocity = 0.4, 0.5, 1.0, -0.3
    shear_viscosity, coupling_viscosity = 4.0, 12.0
    newtonian_rotation_rate, _ = newtonian_couette_constants(inner_radius, outer_radius, inner_velocity,
                                                             outer_velocity)
    radius = np.linspace(inner_radius, outer_radius, 401)
    newtonian_velocity = newtonian_couette_velocity(radius, inner_radius, outer_radius, inner_velocity, outer_velocity)
    newtonian_torque = newtonian_torque_per_unit_length(shear_viscosity + 0.5 * coupling_viscosity, inner_radius,
                                                        outer_radius, inner_velocity, outer_velocity)

    def limit_deviations(wall_micro_rotation: float):
        velocity_deviations, torque_deviations = [], []
        for length in lengths:
            spin_viscosity = (length ** 2 * coupling_viscosity * (2.0 * shear_viscosity + coupling_viscosity)
                              / (shear_viscosity + coupling_viscosity))
            solution = solve_micropolar_couette(inner_radius, outer_radius, shear_viscosity, coupling_viscosity,
                                                spin_viscosity, inner_velocity, outer_velocity,
                                                wall_micro_rotation, wall_micro_rotation)
            velocity_deviations.append(float(np.abs(velocity(solution, radius) - newtonian_velocity).max()
                                             / np.abs(newtonian_velocity).max()))
            torque_deviations.append(float(np.abs(torque_per_unit_length(solution, radius) / newtonian_torque
                                                  - 1.0).max()))
        return np.array(velocity_deviations), np.array(torque_deviations)

    velocity_deviations, torque_deviations = limit_deviations(newtonian_rotation_rate)
    report("d1 w_wall = Newtonian rotation rate alpha, l = 0.05 .. 0.001 (exact for every l, not only in the limit)",
           velocity_deviations.max() < 1e-12 and torque_deviations.max() < 1e-11,
           f"max |v - v_Newtonian| / max |v_Newtonian| {velocity_deviations.max():.2e}, "
           f"max |M(r) / M_Newtonian(mu + kappa/2) - 1| {torque_deviations.max():.2e}")

    # (d2) the limit proper: other wall micro-rotations (here 0 on both walls), deviation from Newtonian -> 0 like l,
    # with the slope of the thin-layer estimate of the module docstring. The velocity deviation is not monotone
    # while l is comparable to the gap (0.1), so its decrease is asked for gap / l >= 10 only.
    velocity_deviations, torque_deviations = limit_deviations(0.0)
    length_ratios = np.array(lengths[:-1]) / np.array(lengths[1:])
    torque_orders = np.log(torque_deviations[:-1] / torque_deviations[1:]) / np.log(length_ratios)
    velocity_orders = np.log(velocity_deviations[:-1] / velocity_deviations[1:]) / np.log(length_ratios)
    _, newtonian_potential_vortex_constant = newtonian_couette_constants(inner_radius, outer_radius, inner_velocity,
                                                                         outer_velocity)
    thin_layer_slope = (coupling_viscosity / (shear_viscosity + coupling_viscosity) * inner_radius * outer_radius
                        * (outer_radius + inner_radius) * (0.0 - newtonian_rotation_rate)
                        / ((outer_radius ** 2 - inner_radius ** 2) * newtonian_potential_vortex_constant))
    measured_slope = torque_deviations[-1] / lengths[-1]
    report("d2 w_wall = 0, l -> 0 at fixed mu, kappa and wall velocities: Newtonian limit of viscosity mu + kappa/2",
           bool(np.all(np.diff(torque_deviations) < 0.0) and np.all(np.diff(velocity_deviations[2:]) < 0.0)
                and abs(torque_orders[-1] - 1.0) < 0.05 and abs(velocity_orders[-1] - 1.0) < 0.15
                and abs(measured_slope / thin_layer_slope - 1.0) < 0.02),
           "l = " + ", ".join(f"{length:g}" for length in lengths)
           + ": |D| = " + ", ".join(f"{value:.3e}" for value in torque_deviations)
           + "; max |v - v_Newtonian| / max |v_Newtonian| = " + ", ".join(f"{value:.3e}"
                                                                            for value in velocity_deviations)
           + f"; order in l between the last two: |D| {torque_orders[-1]:.3f}, v {velocity_orders[-1]:.3f}; "
           f"D / l at l = {lengths[-1]:g}: {measured_slope:.4f} against the thin-layer estimate "
           f"{thin_layer_slope:.4f}")

    # (e) conditioning of the 4 x 4 system down to r / l = 5000
    condition_text = []
    worst_condition = 0.0
    for length in (0.05, 0.001, 0.0001):
        solution = solve_micropolar_couette(**_test_case(length, 2.5))
        worst_condition = max(worst_condition, solution.system_condition_number)
        condition_text.append(f"l = {length:g}: {solution.system_condition_number:.2f} equilibrated, "
                              f"{solution.unequilibrated_system_condition_number:.1f} before equilibration")
    report("e condition number of the 4 x 4 system, R1 = 0.4, R2 = 0.5 (test case)", worst_condition < 100.0,
           "; ".join(condition_text))

    # (f) least-squares fits on the basis functions recover C2 and C1 / (2 mu + kappa)
    worst_fit = 0.0
    fit_text = []
    for length in (0.05, 0.025, 0.01, 0.005):
        case = _test_case(length, 2.5)
        solution = solve_micropolar_couette(**case)
        radius = np.linspace(case["inner_radius"], case["outer_radius"], 200)
        velocity_basis = velocity_basis_functions(radius, length, case["inner_radius"], case["outer_radius"])
        micro_rotation_basis = micro_rotation_basis_functions(radius, length, case["inner_radius"],
                                                              case["outer_radius"])
        velocity_coefficients = np.linalg.lstsq(velocity_basis, velocity(solution, radius), rcond=None)[0]
        micro_rotation_coefficients = np.linalg.lstsq(micro_rotation_basis, micro_rotation(solution, radius),
                                                      rcond=None)[0]
        bessel_velocity_factor = _coupling_fraction(solution) * length
        expected_velocity_coefficients = np.array([solution.background_rotation_rate,
                                                   solution.potential_vortex_constant,
                                                   bessel_velocity_factor * solution.scaled_first_kind_amplitude,
                                                   -bessel_velocity_factor * solution.scaled_second_kind_amplitude])
        expected_micro_rotation_coefficients = np.array([solution.background_rotation_rate,
                                                         solution.scaled_first_kind_amplitude,
                                                         solution.scaled_second_kind_amplitude])
        worst_fit = max(worst_fit,
                        float(np.abs(velocity_coefficients / expected_velocity_coefficients - 1.0).max()),
                        float(np.abs(micro_rotation_coefficients / expected_micro_rotation_coefficients - 1.0).max()))
        normalized_basis = velocity_basis / np.linalg.norm(velocity_basis, axis=0)
        fit_text.append(f"l = {length:g}: {np.linalg.cond(normalized_basis):.0f}")
    report("f least-squares fit on velocity_basis_functions / micro_rotation_basis_functions (test case, 200 radii)",
           worst_fit < 1e-8,
           f"max relative error of the 4 + 3 fitted coefficients (C2 included) {worst_fit:.2e}; condition number "
           "of the column-normalized velocity basis " + ", ".join(fit_text))

    # (g) parameter conversions
    design = solver_parameters_from_design(0.01, 3.0, 0.025)
    eringen = eringen_parameters_from_solver_parameters(1000.0, design.viscosity, design.coupling_coefficient,
                                                        design.viscosity_rotation)
    conversion_errors = [
        (eringen.shear_viscosity + 0.5 * eringen.coupling_viscosity) / 1000.0 / 0.01 - 1.0,
        eringen.coupling_viscosity / eringen.shear_viscosity / 3.0 - 1.0,
        eringen.characteristic_length / 0.025 - 1.0,
        (design.viscosity + design.viscosity_transfer) / design.coupling_coefficient - 1.0,
        design.viscosity_transfer / (-eringen.shear_viscosity / 1000.0) - 1.0,
        design.micro_inertia / 0.025 ** 2 - 1.0,
        design.viscosity / 0.016 - 1.0, design.coupling_coefficient / 0.012 - 1.0,
        design.viscosity_rotation / (0.015 * 0.025 ** 2) - 1.0]
    worst_conversion = float(np.abs(conversion_errors).max())
    report("g conversions design (nu_inf 0.01, K 3, l 0.025) -> solver -> Eringen (rho 1000)",
           worst_conversion < 1e-13,
           f"nu {design.viscosity:.6g}, nu_t {design.viscosity_transfer:.6g}, c {design.coupling_coefficient:.6g}, "
           f"xi {design.viscosity_rotation:.6g}, Theta {design.micro_inertia:.6g}; mu {eringen.shear_viscosity:.6g}, "
           f"kappa {eringen.coupling_viscosity:.6g}, gamma {eringen.spin_viscosity:.6g}, "
           f"l {eringen.characteristic_length:.6g}; max relative round-trip error {worst_conversion:.2e}")

    # the numbers of the test case (information, no criterion)
    for length, inner_micro_rotation in ((0.05, 2.5), (0.025, 2.5), (0.01, 2.5), (0.005, 2.5), (0.01, 0.0)):
        case = _test_case(length, inner_micro_rotation)
        solution = solve_micropolar_couette(**case)
        print(f"[INFO] test case l = {length:g}, inner_micro_rotation = {inner_micro_rotation:g}: "
              f"D = {relative_torque_change(**case):+.10f}, C2 = {solution.potential_vortex_constant:.10f}, "
              f"C1 = {solution.first_integral_constant:.8f}, M = {torque_per_unit_length_from_constant(solution):.8f}"
              f" N, condition number {solution.system_condition_number:.2f}")
    return all_passed


if __name__ == "__main__":
    sys.exit(0 if run_checks() else 1)
