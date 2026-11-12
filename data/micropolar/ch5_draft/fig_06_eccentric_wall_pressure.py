"""
Item 06 - eccentric cylinders: the wall pressure around the rotor against the Stokes solution (the point-force
solution of micropolar_eccentric_reference.py, mean removed), eps = 0.3 and 0.6, with the micropolar runs at K = 3
beside it.

Run from the repository root:  python data/micropolar/ch5_draft/fig_06_eccentric_wall_pressure.py
"""
import matplotlib.pyplot as plt
import numpy as np

import sources
import style
from experiment.validation import micropolar_eccentric_reference as exact_reference

SOURCE = sources.ECCENTRIC
PREFIX = "nd_" if SOURCE is sources.ECCENTRIC_STAGE5 else ""
# eps -> (Newtonian control, [micropolar runs at K = 3: l = 10 dx, l = 25 dx])
RUNS = {0.3: (f"{PREFIX}e30_newtonian", [f"{PREFIX}e30_k3_m5", f"{PREFIX}e30_k3_m2"]),
        0.6: (f"{PREFIX}e60_newtonian", [f"{PREFIX}e60_k3_m5", f"{PREFIX}e60_k3_m2"])}
DENSITY = 1000.0


def centred(angle, values):
    """Angles in degrees from -180 to 180 about the narrowest gap, values minus their mean, sorted."""
    degrees = (np.degrees(np.asarray(angle)) + 180.0) % 360.0 - 180.0
    order = np.argsort(degrees)
    values = np.asarray(values, dtype=np.float64)
    return degrees[order], (values - values.mean())[order]


def main() -> int:
    style.setup()
    analysis = style.load_json(SOURCE["analysis"])
    runs = analysis["runs"]
    figure, axes = plt.subplots(1, 2, figsize=(style.WIDE, 0.42 * style.WIDE))
    used, numbers = [], {}
    for axis, letter, (ratio, (control_name, micropolar_names)) in zip(axes, "ab", RUNS.items()):
        control = runs[control_name]
        speed = control["rotation_rate"] * control["inner_radius"]
        viscosity = DENSITY * control["far_viscosity"]
        exact = exact_reference.numerical_stokes_rotor_wall(
            control["inner_radius"], control["outer_radius"], ratio * (control["outer_radius"]
                                                                       - control["inner_radius"]),
            viscosity, control["rotation_rate"])
        degrees, pressure = centred(exact["angle"], exact["pressure"] / (DENSITY * speed ** 2))
        axis.plot(degrees, pressure, **style.EXACT, label="Stokes, exact")
        walls = control["walls"]
        degrees_run, pressure_run = centred(walls["angle"], walls["rotor"]["wall_pressure_over_dynamic_pressure"])
        axis.plot(degrees_run[::2], pressure_run[::2], **style.run_markers(0, filled=False), label="Newtonian run")
        used.append(control_name)
        entry = {"exact_extremum": float(np.abs(pressure).max()),
                 "newtonian_rms_difference": float(np.sqrt(np.mean(
                     (pressure_run - np.interp(degrees_run, degrees, pressure)) ** 2)))}
        for index, name in enumerate(micropolar_names, start=1):
            if name not in runs:
                continue
            item = runs[name]
            degrees_run, pressure_run = centred(item["walls"]["angle"],
                                                item["walls"]["rotor"]["wall_pressure_over_dynamic_pressure"])
            axis.plot(degrees_run[index % 2::2], pressure_run[index % 2::2], **style.run_markers(index),
                      label=rf"$K = 3$, ${style.LENGTH_IN_SPACINGS} = {item['length_in_spacings']:g}$")
            used.append(name)
            entry[name] = {"extremum_over_exact": float(np.abs(pressure_run).max() / np.abs(pressure).max())}
        numbers[f"eps = {ratio:g}"] = entry
        axis.set_xlim(-180.0, 180.0)
        axis.set_xticks([-180, -90, 0, 90, 180])
        axis.set_xlabel(r"angle from the narrowest gap (degrees)")
        axis.axhline(0.0, color=style.GRID, linewidth=0.5, zorder=0)
        style.panel_letter(axis, letter)
        axis.text(1.0, 1.03, rf"$\varepsilon = {ratio:g}$", transform=axis.transAxes, ha="right", va="bottom",
                  fontsize=style.FONT_SIZE - 1.0)
    axes[0].set_ylabel(r"$(p - \bar p)\,/\,(\rho U^2)$")
    axes[0].legend(loc="lower left", handletextpad=0.4)
    figure.tight_layout(pad=0.4, w_pad=1.2)
    style.save(figure, "fig_06_eccentric_wall_pressure")

    first = runs[RUNS[0.3][0]]
    caption = (
        "Pressure on the rotor between eccentric cylinders, (a) eps = 0.3 and (b) eps = 0.6, against the angle about "
        "the rotor axis measured from the narrowest gap in the direction of rotation; mean over the angle removed. "
        "Line: Stokes solution of the Newtonian flow (sum of point forces; its wall pressure and wall vorticity "
        "integrate to the exact force and torque to 1e-13). Open circles: Newtonian run. Filled symbols: micropolar "
        f"runs, K = 3, l = 10 dx (m = 5) and l = 25 dx (m = 2), w = 0 on both walls. g = {first['gap_in_spacings']:g} "
        f"dx, Ma = {first['mach']:g}, Re = {first['reynolds']:g} on mu_inf (25 on mu for K = 3). The pressure of a "
        "run is the fluid pressure extrapolated to the wall surface (cubic in the wall distance over 0.5 to 12 dx, "
        "72 angle bins, states of the last quarter; every second bin shown). Runs: "
        + ", ".join(f"`{name}`" for name in used) + f"; {SOURCE['note']}. "
        + " ".join(
            f"eps = {ratio:g}: extremum of the Stokes pressure {numbers[f'eps = {ratio:g}']['exact_extremum']:.2f} "
            f"rho U^2, rms difference of the Newtonian run "
            f"{numbers[f'eps = {ratio:g}']['newtonian_rms_difference']:.3f} rho U^2; micropolar extremum over the "
            "Stokes one "
            + ", ".join(f"{numbers[f'eps = {ratio:g}'][name]['extremum_over_exact']:.2f}"
                        for name in RUNS[ratio][1] if name in numbers[f'eps = {ratio:g}']) + "."
            for ratio in RUNS))
    style.record("06", "Eccentric cylinders: wall pressure around the rotor", SOURCE["status"], caption,
                 ["fig_06_eccentric_wall_pressure.pdf", "fig_06_eccentric_wall_pressure.png"],
                 [SOURCE["analysis"]] + used, numbers, SOURCE["tag"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
