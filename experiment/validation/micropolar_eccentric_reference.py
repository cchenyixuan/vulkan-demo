"""
micropolar_eccentric_reference.py - the Stokes flow of a Newtonian fluid between two eccentric rotating cylinders:
force and torque in closed form, and the checks of the formulas (Stage 5 of docs/micropolar_design.md, section 14.3).

The solution is Jeffery's (1922) stream function in bipolar coordinates (also Wannier 1950; Ballal & Rivlin 1976).
I did not have the source texts: the formulas below are derived here (the derivation is in the design note) and
checked three ways by `check`: the concentric limit, the thin-film limit against the Sommerfeld formulas, and an
independent numerical Stokes solution (method of fundamental solutions: point forces inside the inner and outside
the outer cylinder, velocity collocation on the two circles, least squares; no bipolar coordinates).

Geometry and signs. Inner cylinder (radius R1, angular velocity Omega_1) with its axis at x = +e from the axis of the
outer one (radius R2, angular velocity Omega_2), e < R2 - R1: the narrowest gap is on the +x side. Angular
velocities and torques are positive counter-clockwise.

Bipolar parameters: with d1 = (R2^2 - R1^2 - e^2) / (2 e) and a = sqrt(d1^2 - R1^2),
    sinh xi_1 = a / R1,   sinh xi_2 = a / R2,   t = xi_1 - xi_2,   s = xi_1 + xi_2,
    Delta = t tanh t - sinh^2 t - (t - tanh t) cosh s sinh t                                   (negative),
    C = a [ - (Omega_1 - Omega_2) (t + cosh s sinh t) + sinh s sinh t (Omega_1 coth xi_1 - Omega_2 coth xi_2) ] / Delta,
    D = a [ (tanh t - cosh s sinh t) (Omega_1 coth xi_1 - Omega_2 coth xi_2) + sinh s sinh t (Omega_1 - Omega_2) ] / Delta.
The fluid acts on the inner cylinder with
    force    F = (0, - 4 pi mu D)                 per unit depth (perpendicular to the line of centres),
    torque   T_1 = - 4 pi mu (a C + d1 D)         about its own axis,
and on the outer cylinder with -F and, about the outer axis, T_2 = 4 pi mu (a C + (d1 + e) D).
With the outer cylinder at rest: C = a Omega_1 [ (R1 / R2) sinh t - t ] / Delta, D = a Omega_1 (R1 / R2) sinh t tanh t / Delta.

  check       The three checks, printed (and written with --out).
  values      Force and torque for given radii, offset, viscosity and rates.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def bipolar_parameters(inner_radius: float, outer_radius: float, offset: float) -> dict:
    """a (half the distance between the two limiting points), the distance d1 of the inner axis from the midpoint
    of the limiting points, and xi_1 > xi_2 > 0 of the two circles."""
    if not 0.0 < offset < outer_radius - inner_radius:
        raise ValueError("the offset must lie strictly between 0 and R2 - R1")
    inner_distance = (outer_radius ** 2 - inner_radius ** 2 - offset ** 2) / (2.0 * offset)
    half_distance = np.sqrt(inner_distance ** 2 - inner_radius ** 2)
    return {"half_distance": float(half_distance), "inner_distance": float(inner_distance),
            "inner_coordinate": float(np.arcsinh(half_distance / inner_radius)),
            "outer_coordinate": float(np.arcsinh(half_distance / outer_radius))}


def stokes_force_and_torque(inner_radius: float, outer_radius: float, offset: float, viscosity: float,
                            inner_rate: float, outer_rate: float = 0.0) -> dict:
    """Force and torque per unit depth of the Stokes flow between eccentric rotating cylinders (module docstring).
    viscosity is the dynamic viscosity mu. Returns the force of the fluid on the inner cylinder (x along the line
    of centres towards the narrowest gap), its torque on the inner cylinder about the inner axis, and its torque on
    the outer cylinder about the outer axis and about the inner axis."""
    if offset == 0.0:
        torque = (-4.0 * np.pi * viscosity * (inner_rate - outer_rate) * inner_radius ** 2 * outer_radius ** 2
                  / (outer_radius ** 2 - inner_radius ** 2))
        return {"force": [0.0, 0.0], "inner_torque": float(torque), "outer_torque": float(-torque),
                "outer_torque_about_inner_axis": float(-torque)}
    parameters = bipolar_parameters(inner_radius, outer_radius, offset)
    half_distance, inner_distance = parameters["half_distance"], parameters["inner_distance"]
    inner_coordinate, outer_coordinate = parameters["inner_coordinate"], parameters["outer_coordinate"]
    difference = inner_coordinate - outer_coordinate
    total = inner_coordinate + outer_coordinate
    determinant = (difference * np.tanh(difference) - np.sinh(difference) ** 2
                   - (difference - np.tanh(difference)) * np.cosh(total) * np.sinh(difference))
    rate_term = inner_rate / np.tanh(inner_coordinate) - outer_rate / np.tanh(outer_coordinate)
    coefficient_cosh = half_distance * (
        -(inner_rate - outer_rate) * (difference + np.cosh(total) * np.sinh(difference))
        + np.sinh(total) * np.sinh(difference) * rate_term) / determinant
    coefficient_sinh = half_distance * (
        (np.tanh(difference) - np.cosh(total) * np.sinh(difference)) * rate_term
        + np.sinh(total) * np.sinh(difference) * (inner_rate - outer_rate)) / determinant
    force_y = -4.0 * np.pi * viscosity * coefficient_sinh
    inner_torque = -4.0 * np.pi * viscosity * (half_distance * coefficient_cosh + inner_distance * coefficient_sinh)
    outer_torque = 4.0 * np.pi * viscosity * (half_distance * coefficient_cosh
                                              + (inner_distance + offset) * coefficient_sinh)
    # the fluid acts on the outer cylinder with -F at the outer axis, which lies at -offset from the inner axis
    # along x: moving the reference point from the outer to the inner axis adds (-offset) x-arm times (-F_y)
    outer_torque_about_inner_axis = outer_torque + (-offset) * (-force_y)
    return {"force": [0.0, float(force_y)], "inner_torque": float(inner_torque), "outer_torque": float(outer_torque),
            "outer_torque_about_inner_axis": float(outer_torque_about_inner_axis), **parameters}


def sommerfeld_force_and_torque(radius: float, clearance: float, eccentricity_ratio: float, viscosity: float,
                                rate: float) -> dict:
    """The full-film Sommerfeld solution of the Reynolds equation for a journal of radius R turning at Omega in a
    bearing at rest, clearance c << R: load 12 pi mu Omega R^3 eps / (c^2 (2 + eps^2) sqrt(1 - eps^2)), perpendicular
    to the line of centres, and friction torque on the journal 4 pi mu Omega R^3 (1 + 2 eps^2) /
    (c (2 + eps^2) sqrt(1 - eps^2))."""
    root = np.sqrt(1.0 - eccentricity_ratio ** 2)
    load = (12.0 * np.pi * viscosity * rate * radius ** 3 * eccentricity_ratio
            / (clearance ** 2 * (2.0 + eccentricity_ratio ** 2) * root))
    torque = (4.0 * np.pi * viscosity * rate * radius ** 3 * (1.0 + 2.0 * eccentricity_ratio ** 2)
              / (clearance * (2.0 + eccentricity_ratio ** 2) * root))
    return {"load": float(load), "journal_torque": float(-torque)}


def numerical_stokes_force_and_torque(inner_radius: float, outer_radius: float, offset: float, viscosity: float,
                                      inner_rate: float, outer_rate: float = 0.0, source_count: int = 240,
                                      collocation_factor: int = 3) -> dict:
    """An independent numerical solution of the same Stokes problem by the method of fundamental solutions: the
    velocity is a sum of two-dimensional point-force solutions, u_i = sum_k G_ij(x - y_k) f_kj / (4 pi mu),
    G_ij = - delta_ij ln r + r_i r_j / r^2, with the points y_k on a circle inside the inner cylinder and on a
    circle outside the outer one, plus a constant; the strengths follow from the wall velocities at collocation
    points on the two circles by least squares. The fluid acts on the inner cylinder with minus the sum of the
    point forces inside it (and minus the sum of their moments). Returns force, torque and the largest remaining
    wall-velocity error at points between the collocation points."""
    inner_centre = np.array([offset, 0.0])
    outer_centre = np.zeros(2)
    # The point forces must lie between the wall and the singular points of the continued solution (the two
    # limiting points of the bipolar system): inside the inner cylinder beyond the inner limiting point, outside
    # the outer cylinder short of the outer one.
    parameters = bipolar_parameters(inner_radius, outer_radius, offset) if offset > 0.0 else None
    if parameters:
        inner_limit = parameters["inner_distance"] - parameters["half_distance"]          # from the inner axis
        outer_limit = parameters["inner_distance"] + offset + parameters["half_distance"]  # from the outer axis
    else:
        inner_limit, outer_limit = 0.0, np.inf
    inner_source_radius = max(0.5 * (inner_limit + inner_radius), 0.75 * inner_radius)
    outer_source_radius = min(0.5 * (outer_limit + outer_radius), 1.3 * outer_radius)

    def circle(centre, radius, count, phase=0.0):
        angle = 2.0 * np.pi * (np.arange(count) + phase) / count
        return centre + radius * np.column_stack([np.cos(angle), np.sin(angle)])
    sources = np.vstack([circle(inner_centre, inner_source_radius, source_count),
                         circle(outer_centre, outer_source_radius, source_count)])
    inside = np.arange(2 * source_count) < source_count

    def wall_points(count, phase):
        inner_points = circle(inner_centre, inner_radius, count, phase)
        outer_points = circle(outer_centre, outer_radius, count, phase)
        relative = inner_points - inner_centre
        inner_velocity = inner_rate * np.column_stack([-relative[:, 1], relative[:, 0]])
        outer_velocity = outer_rate * np.column_stack([-outer_points[:, 1], outer_points[:, 0]])
        return np.vstack([inner_points, outer_points]), np.vstack([inner_velocity, outer_velocity])

    def matrix(points):
        separation = points[:, None, :] - sources[None, :, :]
        distance_squared = (separation ** 2).sum(axis=2)
        logarithm = -0.5 * np.log(distance_squared)
        rows = np.zeros((2 * points.shape[0], 2 * sources.shape[0] + 2))
        for component in range(2):
            for direction in range(2):
                block = separation[:, :, component] * separation[:, :, direction] / distance_squared
                if component == direction:
                    block = block + logarithm
                rows[component::2, direction:2 * sources.shape[0]:2] = block / (4.0 * np.pi * viscosity)
            rows[component::2, 2 * sources.shape[0] + component] = 1.0
        return rows
    points, velocity = wall_points(collocation_factor * source_count, 0.0)
    system = matrix(points)
    scale = np.sqrt((system ** 2).sum(axis=0))
    solution = np.linalg.lstsq(system / scale, velocity.ravel(), rcond=1e-14)[0] / scale
    strengths = solution[:2 * sources.shape[0]].reshape(-1, 2)
    test_points, test_velocity = wall_points(collocation_factor * source_count, 0.5)
    residual = matrix(test_points) @ solution - test_velocity.ravel()
    speed = max(abs(inner_rate) * inner_radius, abs(outer_rate) * outer_radius)
    force = -strengths[inside].sum(axis=0)
    arm = sources[inside] - inner_centre
    torque = -(arm[:, 0] * strengths[inside, 1] - arm[:, 1] * strengths[inside, 0]).sum()
    outer_force = -strengths[~inside].sum(axis=0)       # by the fictitious fluid outside; not the wall force
    return {"force": force.tolist(), "inner_torque": float(torque),
            "wall_velocity_error_over_wall_speed": float(np.abs(residual).max() / speed),
            "outer_point_force_sum": outer_force.tolist()}


def command_check(arguments) -> int:
    lines = []

    def say(text=""):
        lines.append(text)
        print(text)
    say("Stokes flow between eccentric rotating cylinders: checks of the closed form")
    say()
    say("1. concentric limit (R1 = 4, R2 = 5, mu = 1, Omega_1 = 1): torque on the inner cylinder over "
        "-4 pi mu Omega R1^2 R2^2 / (R2^2 - R1^2), and the force over mu Omega R1")
    exact = -4.0 * np.pi * 16.0 * 25.0 / 9.0
    for offset in (1e-1, 1e-2, 1e-3, 1e-4):
        result = stokes_force_and_torque(4.0, 5.0, offset, 1.0, 1.0)
        say(f"   e = {offset:<8g} T_1 / T_concentric - 1 = {result['inner_torque'] / exact - 1.0:+.3e}   "
            f"F_y / (mu Omega R1) = {result['force'][1] / 4.0:+.4e}   F_y / e = {result['force'][1] / offset:.6f}")
    say()
    say("2. thin-film limit (R1 = 1, mu = 1, Omega_1 = 1): load and journal torque over the Sommerfeld values")
    say(f"   {'c / R':>8}{'eps':>6}{'load ratio':>14}{'torque ratio':>14}")
    for clearance in (0.1, 0.01, 0.001, 0.0001):
        for ratio in (0.3, 0.6, 0.9):
            result = stokes_force_and_torque(1.0, 1.0 + clearance, ratio * clearance, 1.0, 1.0)
            reference = sommerfeld_force_and_torque(1.0, clearance, ratio, 1.0, 1.0)
            say(f"   {clearance:>8g}{ratio:>6g}{result['force'][1] / reference['load']:>14.6f}"
                f"{result['inner_torque'] / reference['journal_torque']:>14.6f}")
    say()
    say("3. independent numerical Stokes solution (point forces, least squares): closed form against it")
    say(f"   {'R1':>9}{'R2':>9}{'e':>6}{'Om_1':>8}{'Om_2':>6}{'F_y closed':>16}{'F_y numerical':>16}{'rel. diff.':>12}"
        f"{'F_x num. / F_y':>16}{'T_1 closed':>16}{'T_1 numerical':>16}{'rel. diff.':>12}{'wall error':>12}")
    cases = [(4.0, 5.0, 0.3, 1.0, 0.0), (4.0, 5.0, 0.6, 1.0, 0.0), (4.0, 5.0, 0.9, 1.0, 0.0),
             (1.0, 2.0, 0.5, 1.0, 0.0), (1.0, 3.0, 1.2, 1.0, 0.0), (4.0, 5.0, 0.6, 0.0, 1.0),
             (4.0, 5.0, 0.6, 1.0, 1.0), (4.0, 5.0, 0.6, 1.0, -0.5), (4.0375, 4.9625, 0.6, 0.9814, 0.0),
             (4.09375, 4.90625, 0.6, 0.9542, 0.0)]
    worst = 0.0
    for inner_radius, outer_radius, offset, inner_rate, outer_rate in cases:
        closed = stokes_force_and_torque(inner_radius, outer_radius, offset, 1.0, inner_rate, outer_rate)
        numerical = numerical_stokes_force_and_torque(inner_radius, outer_radius, offset, 1.0, inner_rate,
                                                      outer_rate)
        force_difference = numerical["force"][1] / closed["force"][1] - 1.0
        torque_difference = numerical["inner_torque"] / closed["inner_torque"] - 1.0
        worst = max(worst, abs(force_difference), abs(torque_difference))
        say(f"   {inner_radius:>9g}{outer_radius:>9g}{offset:>6g}{inner_rate:>8g}{outer_rate:>6g}"
            f"{closed['force'][1]:>16.9f}{numerical['force'][1]:>16.9f}{force_difference:>12.1e}"
            f"{numerical['force'][0] / numerical['force'][1]:>16.1e}{closed['inner_torque']:>16.9f}"
            f"{numerical['inner_torque']:>16.9f}{torque_difference:>12.1e}"
            f"{numerical['wall_velocity_error_over_wall_speed']:>12.1e}")
    say(f"   largest relative difference: {worst:.1e}")
    say()
    say("4. the cases of the runs (R1 = 0.4 m, R2 = 0.5 m, Omega = 2.5 rad/s, per metre of depth)")
    say(f"   {'eps':>5}{'mu (Pa s)':>11}{'load (N/m)':>14}{'T rotor (N)':>14}{'T stator about the rotor axis':>32}"
        f"{'T rotor / concentric':>22}{'Sommerfeld load':>17}")
    for ratio in (0.1, 0.3, 0.6):
        for viscosity in (10.0, 100.0):
            result = stokes_force_and_torque(0.4, 0.5, ratio * 0.1, viscosity, 2.5)
            concentric = stokes_force_and_torque(0.4, 0.5, 0.0, viscosity, 2.5)["inner_torque"]
            thin = sommerfeld_force_and_torque(0.4, 0.1, ratio, viscosity, 2.5)["load"]
            say(f"   {ratio:>5g}{viscosity:>11g}{result['force'][1]:>14.4f}{result['inner_torque']:>14.4f}"
                f"{result['outer_torque_about_inner_axis']:>32.4f}{result['inner_torque'] / concentric:>22.6f}"
                f"{thin:>17.2f}")
    if arguments.out:
        target = _REPO_ROOT / arguments.out
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return 0


def command_values(arguments) -> int:
    result = stokes_force_and_torque(arguments.inner_radius, arguments.outer_radius, arguments.offset,
                                     arguments.viscosity, arguments.inner_rate, arguments.outer_rate)
    for key, value in result.items():
        print(f"{key:<32}{value}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("--out", default=None, help="text file, relative to the repository root")
    check.set_defaults(handler=command_check)
    values = commands.add_parser("values")
    values.add_argument("--inner-radius", type=float, required=True)
    values.add_argument("--outer-radius", type=float, required=True)
    values.add_argument("--offset", type=float, required=True)
    values.add_argument("--viscosity", type=float, required=True, help="dynamic viscosity")
    values.add_argument("--inner-rate", type=float, required=True)
    values.add_argument("--outer-rate", type=float, default=0.0)
    values.set_defaults(handler=command_values)
    arguments = parser.parse_args()
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
