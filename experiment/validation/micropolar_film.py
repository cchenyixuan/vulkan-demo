"""
micropolar_film.py - the plane film of the micropolar model on the host (Stage 6 of docs/micropolar_design.md).

The steady plane film between two walls, 0 < y < h, flow along x:

    nu u'' + c w' = G            (G = (dp/dx) / rho)
    xi w'' - c (u' + 2 w) = 0

with nu_inf = nu - c / 2, l^2 = xi nu / (c (2 nu - c)), delta = c l / (2 nu). Its general solution is

    w(y) = - (G y / nu_inf + a) / 2 + B exp(-y / l) + C exp(-(h - y) / l)
    u(y) = G y^2 / (2 nu_inf) + a y + 2 delta (B exp(-y / l) - C exp(-(h - y) / l)) + E

and the shear stress on a plane y = const is rho nu_inf (G y / nu_inf + a): the wall at y = 0 feels mu_inf a.
The four constants follow from u and w on the two walls. Everything here is linear in (G, U1, U2, w1, w2).

  check   The closed forms against this solution: the film function of micropolar lubrication theory
          f(h) = h^3 - 6 delta h^2 coth(h / (2 l)) + 12 delta l h, the gap of the shear-driven part
          h - 2 delta tanh(h / (2 l)); the sharper form of the wall rule (task of Stage 6, 3.1): core = Newtonian
          profile with u = - delta du/dn at the nominal wall, each wall layer carrying delta l times the wall shear
          rate more: f = h^3 - 6 delta h^2 + 12 delta l h up to exponentially small terms.
  flux    The wedge term dQ/dh of the exact concentric solution against the plane film: does the load see the
          slip term of the rule (it does not).
  bearing The journal bearing in the thin-film approximation with the film of the model: change of load and
          torque against the Newtonian film; the torque also with the curvature term (the slip term of the rule
          on the rotor, and the shear-driven stress of the exact concentric solution at the local gap).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
NEWLINE = chr(10)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def displacement_over_length(coupling_ratio: float) -> float:
    """delta / l = c / (2 nu) = K / (2 (1 + K)) (= N^2 of micropolar lubrication theory)."""
    return coupling_ratio / (2.0 * (1.0 + coupling_ratio))


def plane_film(gap, length: float, displacement: float, gradient=0.0, lower_velocity=0.0, upper_velocity=0.0,
               lower_spin=0.0, upper_spin=0.0) -> dict:
    """The constants of the plane film for arrays of gaps: G' = G / nu_inf, wall velocities and wall
    micro-rotations given. Returns a (wall shear rate of the far viscosity: stress on the lower wall / mu_inf),
    B, C, E and the flux Q = integral of u over the gap."""
    gap = np.atleast_1d(np.asarray(gap, dtype=np.float64))
    shape = gap.shape
    gradient, lower_velocity, upper_velocity, lower_spin, upper_spin = (
        np.broadcast_to(np.asarray(value, dtype=np.float64), shape) for value in
        (gradient, lower_velocity, upper_velocity, lower_spin, upper_spin))
    decay = np.exp(-gap / length)
    matrix = np.zeros(shape + (4, 4))
    right = np.zeros(shape + (4,))
    # unknowns (a, B, C, E)
    matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2] = -0.5, 1.0, decay
    right[..., 0] = lower_spin
    matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2] = -0.5, decay, 1.0
    right[..., 1] = upper_spin + 0.5 * gradient * gap
    matrix[..., 2, 1], matrix[..., 2, 2], matrix[..., 2, 3] = 2.0 * displacement, -2.0 * displacement * decay, 1.0
    right[..., 2] = lower_velocity
    matrix[..., 3, 0], matrix[..., 3, 1], matrix[..., 3, 2], matrix[..., 3, 3] = (
        gap, 2.0 * displacement * decay, -2.0 * displacement, 1.0)
    right[..., 3] = upper_velocity - 0.5 * gradient * gap ** 2
    solution = np.linalg.solve(matrix, right[..., None])[..., 0]
    shear, first, second, constant = (solution[..., index] for index in range(4))
    flux = (gradient * gap ** 3 / 6.0 + shear * gap ** 2 / 2.0
            + 2.0 * displacement * length * (1.0 - decay) * (first - second) + constant * gap)
    return {"shear": shear, "first": first, "second": second, "constant": constant, "flux": flux,
            "upper_shear": shear + gradient * gap}


def profile(coordinate, gap: float, length: float, displacement: float, constants: dict, gradient: float = 0.0):
    """u(y) and w(y) of one film from the constants of plane_film (scalars)."""
    shear, first, second, constant = (float(np.ravel(constants[key])[0]) for key in
                                      ("shear", "first", "second", "constant"))
    near, far = np.exp(-coordinate / length), np.exp(-(gap - coordinate) / length)
    velocity = (0.5 * gradient * coordinate ** 2 + shear * coordinate
                + 2.0 * displacement * (first * near - second * far) + constant)
    spin = -0.5 * (gradient * coordinate + shear) + first * near + second * far
    return velocity, spin


def film_function(kind: str, gap, length: float, displacement: float):
    """f(h): the pressure-driven flux between walls with w = 0 is - f(h) G / (12 nu_inf).
    lubrication: the film function of micropolar lubrication theory, h^3 - 6 delta h^2 coth(h / (2 l)) + 12 delta l h
    rule:        the wall rule, both walls displaced by delta: (h - 2 delta)^3
    first_order: h^3 - 6 delta h^2
    sharper:     core with the slip length - delta plus the flux of the two wall layers: h^3 - 6 delta h^2
                 + 12 delta l h
    newtonian:   h^3"""
    gap = np.asarray(gap, dtype=np.float64)
    if kind == "newtonian":
        return gap ** 3
    if kind == "rule":
        return (gap - 2.0 * displacement) ** 3
    if kind == "first_order":
        return gap ** 3 - 6.0 * displacement * gap ** 2
    if kind == "sharper":
        return gap ** 3 - 6.0 * displacement * gap ** 2 + 12.0 * displacement * length * gap
    if kind == "lubrication":
        return (gap ** 3 - 6.0 * displacement * gap ** 2 / np.tanh(0.5 * gap / length)
                + 12.0 * displacement * length * gap)
    raise ValueError(kind)


def concentric_change(coupling_ratio: float, length: float, radius: float):
    """h -> D of the exact micropolar solution between concentric cylinders of radii R1 and R1 + h (rotor at R1,
    w = 0 on both walls): the factor 1 + D on the shear-driven wall stress of a film that is locally concentric."""
    from experiment.validation import micropolar_couette_reference as reference
    from experiment.validation.micropolar_couette import design_parameters
    design = design_parameters(0.01, coupling_ratio, length)
    eringen = reference.eringen_parameters_from_solver_parameters(1000.0, design["viscosity"], design["coupling"],
                                                                  design["viscosity_rotation"])

    def change(gap):
        gap = np.atleast_1d(gap)
        grid = np.linspace(gap.min(), gap.max(), 257) if gap.size > 257 else gap
        values = np.array([reference.relative_torque_change(radius, radius + value, eringen.shear_viscosity,
                                                            eringen.coupling_viscosity, eringen.spin_viscosity,
                                                            1.0, 0.0, 0.0, 0.0) for value in grid])
        return np.interp(gap, grid, values) if gap.size > 257 else values
    return change


def thin_film_bearing(eccentricity_ratio: float, clearance: float, radius: float, length: float,
                      displacement: float, rotor_spin: float = 0.0, shear_change=None, points: int = 20000) -> dict:
    """The journal bearing in the thin-film approximation with the plane film of the model at every angle: rotor
    (y = 0) moving at U = 1, stator at rest, w = 0 on both walls. displacement = 0: the Newtonian film.
    rotor_spin: the effective micro-rotation of the rotor wall in the plane problem (0: the plane film as it is;
    Omega = U / R: the slip term of the wall rule, 2 delta (w_b - Omega_b), which a plane film lacks).
    shear_change: h -> D; the shear-driven stress on the rotor is then the Newtonian one times 1 + D(h) (the film
    taken as locally concentric), the pressure and the pressure-driven stress stay those of the plane film.
    Returns load / (mu_inf U) per unit depth, journal torque / (mu_inf U R), the flux, and the pressure by angle
    (from the widest gap, in the direction of rotation) in units of mu_inf U / R."""
    angle = 2.0 * np.pi * np.arange(points) / points
    step = 2.0 * np.pi / points
    gap = clearance * (1.0 + eccentricity_ratio * np.cos(angle))
    # unit responses: flux and rotor shear for G' = 1 alone and for the walls alone
    driven = plane_film(gap, length, displacement, gradient=1.0)
    walls = plane_film(gap, length, displacement, lower_velocity=1.0, lower_spin=rotor_spin)
    flux_constant = (walls["flux"] / driven["flux"]).sum() / (1.0 / driven["flux"]).sum()
    gradient = (flux_constant - walls["flux"]) / driven["flux"]            # G' = (dp/dx) / mu_inf, U = 1
    pressure = np.cumsum(gradient) * step * radius
    pressure -= pressure.mean()
    load = np.hypot((pressure * np.cos(angle)).sum(), (pressure * np.sin(angle)).sum()) * step * radius
    wall_shear = walls["shear"] if shear_change is None else -(1.0 + shear_change(gap)) / gap
    shear = wall_shear + gradient * driven["shear"]                        # stress on the rotor / mu_inf
    torque = -(shear.sum() * step * radius)                                # / (mu_inf U R), positive = resisting
    return {"load": float(load), "torque": float(torque), "flux": float(flux_constant),
            "angle": angle, "pressure": pressure * radius}


def bearing_changes(eccentricity_ratio: float, clearance: float, radius: float, length: float,
                    coupling_ratio: float) -> dict:
    """Change of load and journal torque of the micropolar film against the Newtonian film of viscosity mu_inf:
    plane = the plane film as it is; slip = with the slip term of the rule on the rotor (w1 = Omega); concentric =
    the shear-driven stress from the exact concentric solution at the local gap (torque only)."""
    displacement = displacement_over_length(coupling_ratio) * length
    base = thin_film_bearing(eccentricity_ratio, clearance, radius, length, 0.0)
    variants = {"plane": {}, "slip": {"rotor_spin": 1.0 / radius},
                "concentric": {"shear_change": concentric_change(coupling_ratio, length, radius)}}
    result = {}
    for label, keywords in variants.items():
        film = thin_film_bearing(eccentricity_ratio, clearance, radius, length, displacement, **keywords)
        result[f"load_change_{label}"] = (film["load"] / base["load"] - 1.0 if eccentricity_ratio > 0.0
                                          else float("nan"))
        result[f"torque_change_{label}"] = film["torque"] / base["torque"] - 1.0
    return result


def command_check(arguments) -> int:
    lines, records = [], {}

    def say(text=""):
        lines.append(text)
        print(text)
    coupling_ratio = arguments.coupling_ratio
    ratio = displacement_over_length(coupling_ratio)
    say(f"Plane film of the model, K = {coupling_ratio:g}: delta / l = K / (2 (1 + K)) = {ratio:.4f}")
    say()
    say("1. closed forms against the solution of the four-constant system (pressure-driven film between walls with "
        "w = 0; shear-driven film)")
    say(f"{'h / l':>7}{'f / h^3, system':>18}{'lubrication':>14}{'difference':>12}{'g_s / h, system':>18}"
        f"{'closed form':>14}{'difference':>12}")
    worst = 0.0
    for gap in (0.5, 1.0, 1.6, 2.0, 2.8, 4.0, 7.0, 10.0, 14.0, 20.0, 40.0, 100.0):
        driven = plane_film(gap, 1.0, ratio, gradient=-1.0)
        film = 12.0 * float(driven["flux"][0]) / gap ** 3
        closed = float(film_function("lubrication", gap, 1.0, ratio)) / gap ** 3
        sheared = plane_film(gap, 1.0, ratio, lower_velocity=1.0)
        shear_gap = -1.0 / float(sheared["shear"][0]) / gap
        closed_gap = (gap - 2.0 * ratio * np.tanh(0.5 * gap)) / gap
        worst = max(worst, abs(film / closed - 1.0), abs(shear_gap / closed_gap - 1.0))
        say(f"{gap:>7g}{film:>18.10f}{closed:>14.10f}{film - closed:>12.1e}{shear_gap:>18.10f}{closed_gap:>14.10f}"
            f"{shear_gap - closed_gap:>12.1e}")
    records["closed_forms_largest_relative_difference"] = worst
    say(f"   largest relative difference {worst:.1e}; limit h -> 0: f / h^3 -> nu_inf / nu = "
        f"{(1.0 + 0.5 * coupling_ratio) / (1.0 + coupling_ratio):.4f}")
    say()
    say("2. the sharper form of the wall rule for the pressure-driven film (3.1 of the task)")
    say("   core: the Newtonian profile u_c = G' (y^2 - h y) / 2 + delta G' h / 2, which satisfies u = - delta du/dn "
        "at both nominal walls (n into the fluid); its flux is - G' (h^3 - 6 delta h^2) / 12")
    say("   wall layers: each carries delta l times the wall shear rate of the core, - G' h / 2, more than the core "
        "does there; together - G' (12 delta l h) / 12")
    say(f"{'h / l':>7}{'core: max |u - u_c| / max u, middle half':>42}{'layer excess / (delta l shear rate)':>37}"
        f"{'f sharper / f':>15}{'f rule / f':>12}{'f first order / f':>19}{'exp(-h / l)':>13}")
    records["sharper"] = {}
    for gap in (2.0, 4.0, 7.0, 10.0, 14.0, 20.0, 40.0):
        driven = plane_film(gap, 1.0, ratio, gradient=-1.0)
        coordinate = np.linspace(0.0, gap, 40001)
        velocity, _spin = profile(coordinate, gap, 1.0, ratio, driven, gradient=-1.0)
        core = -0.5 * (coordinate ** 2 - gap * coordinate) - 0.5 * ratio * gap
        middle = (coordinate > 0.25 * gap) & (coordinate < 0.75 * gap)
        core_error = float(np.abs(velocity - core)[middle].max() / velocity.max())
        lower = coordinate <= 0.5 * gap
        excess = float(np.trapezoid((velocity - core)[lower], coordinate[lower]))
        shear_rate = 0.5 * gap                                        # du_c/dy at y = 0 for G' = -1
        exact = float(film_function("lubrication", gap, 1.0, ratio))
        entry = {"core_error": core_error, "layer_excess_over_prediction": excess / (ratio * shear_rate),
                 "sharper_over_exact": float(film_function("sharper", gap, 1.0, ratio)) / exact,
                 "rule_over_exact": float(film_function("rule", gap, 1.0, ratio)) / exact,
                 "first_order_over_exact": float(film_function("first_order", gap, 1.0, ratio)) / exact}
        records["sharper"][f"{gap:g}"] = entry
        say(f"{gap:>7g}{core_error:>42.2e}{entry['layer_excess_over_prediction']:>37.6f}"
            f"{entry['sharper_over_exact']:>15.6f}{entry['rule_over_exact']:>12.4f}"
            f"{entry['first_order_over_exact']:>19.4f}{np.exp(-gap):>13.1e}")
    say("   f sharper - f = 6 delta h^2 (coth(h / (2 l)) - 1) = 12 delta h^2 exp(-h / l) + ...: exponentially small.")
    if arguments.out:
        target = _REPO_ROOT / arguments.out
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        target.with_suffix(".json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    return 0


def rule_changes(eccentricity_ratio: float, clearance: float, radius: float, length: float,
                 coupling_ratio: float) -> dict:
    """The wall rule in its slip form on the exact Stokes solution: rotor radius R1 + delta turning at
    Omega (R1 - delta) / (R1 + delta), stator radius R2 - delta, the same offset; over the nominal geometry."""
    from experiment.validation import micropolar_eccentric_reference as exact_reference
    from experiment.validation.micropolar_stage5 import slip_rule_torque_change
    displacement = displacement_over_length(coupling_ratio) * length
    if eccentricity_ratio <= 0.0:
        return {"load_change_rule": float("nan"),
                "torque_change_rule": slip_rule_torque_change(radius, radius + clearance, displacement)}
    nominal = exact_reference.stokes_force_and_torque(radius, radius + clearance, eccentricity_ratio * clearance,
                                                      1.0, 1.0)
    slip = exact_reference.stokes_force_and_torque(radius + displacement, radius + clearance - displacement,
                                                   eccentricity_ratio * clearance, 1.0,
                                                   (radius - displacement) / (radius + displacement))
    return {"load_change_rule": slip["force"][1] / nominal["force"][1] - 1.0,
            "torque_change_rule": slip["inner_torque"] / nominal["inner_torque"] - 1.0}


def predictions(eccentricity_ratio: float, clearance: float, radius: float, length: float,
                coupling_ratio: float) -> dict:
    """Everything the host predicts for one case: film function in the thin-film calculation, and the rule."""
    return {**bearing_changes(eccentricity_ratio, clearance, radius, length, coupling_ratio),
            **rule_changes(eccentricity_ratio, clearance, radius, length, coupling_ratio)}


def command_bearing(arguments) -> int:
    lines, records = [], {}

    def say(text=""):
        lines.append(text)
        print(text)
    radius, clearance = 0.4, 0.1
    say("Journal bearing in the thin-film approximation with the plane film of the model (R1 = 4 g; ratios to the "
        "Newtonian film of viscosity mu_inf). Load: the plane film as it is, and with the slip term of the rule on "
        "the rotor. Torque: the plane film as it is; with the slip term (factor about 1 - 2 delta / R1 on the "
        "shear-driven stress); with the shear-driven stress of the exact concentric solution at the local gap. "
        "Beside them the wall rule evaluated on the exact Stokes solution of its slip-form geometry.")
    say(f"{'K':>4}{'eps':>5}{'l / g':>7}{'h_min / l':>10}{'load: film':>12}{'+ slip term':>12}{'rule, Stokes':>14}"
        f"{'torque: film':>14}{'+ slip term':>12}{'loc. conc.':>12}{'rule, Stokes':>14}{'exact (eps = 0)':>17}")
    for coupling_ratio in (3.0, 1.0):
        for eccentricity_ratio in (0.0, 0.3, 0.6):
            for length_over_gap in (0.05, 0.1, 0.2, 0.25):
                length = length_over_gap * clearance
                entry = predictions(eccentricity_ratio, clearance, radius, length, coupling_ratio)
                exact = (float(concentric_change(coupling_ratio, length, radius)(np.array([clearance]))[0])
                         if eccentricity_ratio == 0.0 else None)
                records[f"K{coupling_ratio:g}_eps{eccentricity_ratio:g}_l{length_over_gap:g}"] = entry
                say(f"{coupling_ratio:>4g}{eccentricity_ratio:>5g}{length_over_gap:>7g}"
                    f"{(1.0 - eccentricity_ratio) / length_over_gap:>10.1f}"
                    f"{100.0 * entry['load_change_plane']:>+11.2f}%{100.0 * entry['load_change_slip']:>+11.2f}%"
                    f"{100.0 * entry['load_change_rule']:>+13.2f}%{entry['torque_change_plane']:>14.4f}"
                    f"{entry['torque_change_slip']:>12.4f}{entry['torque_change_concentric']:>12.4f}"
                    f"{entry['torque_change_rule']:>14.4f}" + (f"{exact:>17.4f}" if exact is not None else ""))
    say("The exact concentric flux supports the plane film for the load: between concentric cylinders the change of "
        "dQ/dh against the Newtonian flow is -0.1 to -0.3 % (K = 3, l = g / 10 and g / 20, h from 0.4 g to 1.6 g), "
        "where the slip term alone would give -2 delta / R1 = -1.9 % and -0.9 %.")
    if arguments.out:
        target = _REPO_ROOT / arguments.out
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(NEWLINE.join(lines) + NEWLINE, encoding="utf-8", newline=NEWLINE)
        target.with_suffix(".json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    return 0


def command_flux(arguments) -> int:
    """Does the load see the slip term of the rule? The flux Q = integral of v_theta dr between concentric cylinders
    (rotor at R1, stator at R1 + h, w = 0 on both), exact micropolar against Newtonian: the wedge term of a
    bearing is dQ/dh. The plane film with w = 0 on both walls leaves dQ/dh unchanged; with the slip term on the
    rotor (w1 = Omega in the plane problem) it would fall by 2 delta / R1."""
    from experiment.validation import micropolar_couette_reference as reference
    from experiment.validation.micropolar_couette import design_parameters
    lines, records = [], {}

    def say(text=""):
        lines.append(text)
        print(text)
    radius, speed, far_viscosity = 0.4, 1.0, 0.01
    rate = speed / radius
    say("Change of dQ/dh (Q = flux between concentric cylinders, rotor radius R1 = 4 g fixed, gap h) of the exact "
        "micropolar solution against the Newtonian one, and what the plane film gives with the effective rotor "
        "micro-rotation w1 = 0 (the plane film as it is) and w1 = Omega (the slip term of the rule)")
    say(f"{'K':>4}{'l / g':>7}{'h / g':>7}{'h / l':>7}{'exact':>11}{'plane, w1 = 0':>15}{'w1 = Omega':>12}"
        f"{'2 delta / R1':>14}")
    for coupling_ratio, length in ((3.0, 0.01), (3.0, 0.005), (1.0, 0.01)):
        design = design_parameters(far_viscosity, coupling_ratio, length)
        eringen = reference.eringen_parameters_from_solver_parameters(1000.0, design["viscosity"], design["coupling"],
                                                                      design["viscosity_rotation"])
        displacement = displacement_over_length(coupling_ratio) * length

        def flux(gap, micropolar):
            coordinate = np.linspace(radius, radius + gap, 20001)
            if micropolar:
                solution = reference.solve_micropolar_couette(
                    radius, radius + gap, eringen.shear_viscosity, eringen.coupling_viscosity,
                    eringen.spin_viscosity, speed, 0.0, 0.0, 0.0)
                velocity = reference.velocity(solution, coordinate)
            else:
                alpha, beta = reference.newtonian_couette_constants(radius, radius + gap, speed, 0.0)
                velocity = alpha * coordinate + beta / coordinate
            return np.trapezoid(velocity, coordinate)
        for gap in (0.04, 0.07, 0.1, 0.13, 0.16):
            step = 1.0e-4
            exact = ((flux(gap + step, True) - flux(gap - step, True))
                     / (flux(gap + step, False) - flux(gap - step, False)) - 1.0)

            def plane(spin):
                gaps = np.array([gap - step, gap + step])
                film = plane_film(gaps, length, displacement, lower_velocity=speed, lower_spin=spin)["flux"]
                base = plane_film(gaps, length, 0.0, lower_velocity=speed)["flux"]
                return (film[1] - film[0]) / (base[1] - base[0]) - 1.0
            entry = {"exact": float(exact), "plane": float(plane(0.0)), "plane_with_slip_term": float(plane(rate)),
                     "two_displacements_over_radius": 2.0 * displacement / radius}
            records[f"K{coupling_ratio:g}_l{length / 0.1:g}_h{gap / 0.1:g}"] = entry
            say(f"{coupling_ratio:>4g}{length / 0.1:>7g}{gap / 0.1:>7g}{gap / length:>7.1f}{entry['exact']:>+11.5f}"
                f"{entry['plane']:>+15.5f}{entry['plane_with_slip_term']:>+12.5f}"
                f"{-entry['two_displacements_over_radius']:>+14.5f}")
    if arguments.out:
        target = _REPO_ROOT / arguments.out
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(NEWLINE.join(lines) + NEWLINE, encoding="utf-8", newline=NEWLINE)
        target.with_suffix(".json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("--coupling-ratio", type=float, default=3.0)
    check.add_argument("--out", default=None)
    check.set_defaults(handler=command_check)
    bearing = commands.add_parser("bearing")
    bearing.add_argument("--out", default=None)
    bearing.set_defaults(handler=command_bearing)
    flux = commands.add_parser("flux")
    flux.add_argument("--out", default=None)
    flux.set_defaults(handler=command_flux)
    arguments = parser.parse_args()
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
