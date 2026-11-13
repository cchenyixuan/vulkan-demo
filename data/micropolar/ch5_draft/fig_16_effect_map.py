"""
Item 16 - the map of the micropolar effect (host only): in the plane (N, m), with N = sqrt(K / (2 + K)) and
m = g / (2 l), the contours D = 1 %, 5 % and 20 % of (a) the change of the torque between the concentric cylinders
of the chapter, from the exact solution, with the rule in its slip form dashed, and (b) the change of the load of
the bearing at an eccentricity ratio of 0.6, from the film function of micropolar lubrication theory in the
thin-film calculation, with m built on the mean gap. Every run of items 04 and 09 is marked; the region where D is
below the noise floor of the runs is shaded.

Run from the repository root:  python data/micropolar/ch5_draft/fig_16_effect_map.py
"""
import csv

import matplotlib.pyplot as plt
import numpy as np

import concentric_data as data
import style
from experiment.validation import micropolar_film as film

LEVELS = (0.01, 0.05, 0.20)
ECCENTRICITY_RATIO = 0.6
BEARING_RADIUS, BEARING_CLEARANCE = 0.4, 0.1          # R1 = 4 g, mean gap g: the bearing of items 05 to 09
CONCENTRIC = "data/micropolar/stage6/concentric.json"
ECCENTRIC = "data/micropolar/stage6/eccentric.json"
COUPLING_NUMBERS = np.linspace(0.02, 0.98, 49)
GAP_RATIOS = np.geomspace(0.8, 100.0, 64)      # the frame starts a little below m = 1, where runs sit
# where the labels of the contours go (N), away from the points of the runs
LABEL_PLACES = {"a": (0.92, 0.92, 0.92), "b": (0.30, 0.92, 0.92)}


def coupling_ratio(coupling_number):
    """K = kappa / mu from N = sqrt(K / (2 + K))."""
    return 2.0 * coupling_number ** 2 / (1.0 - coupling_number ** 2)


def coupling_number(ratio):
    return np.sqrt(ratio / (2.0 + ratio))


def load_change(ratio: float, gap_over_two_lengths: float) -> float:
    """Load of the micropolar film over that of the Newtonian film of viscosity mu_inf, minus one (thin film, the
    plane film function, eccentricity ratio 0.6); l from m on the mean gap."""
    length = BEARING_CLEARANCE / (2.0 * gap_over_two_lengths)
    displacement = film.displacement_over_length(ratio) * length
    micropolar = film.thin_film_bearing(ECCENTRICITY_RATIO, BEARING_CLEARANCE, BEARING_RADIUS, length, displacement)
    newtonian = film.thin_film_bearing(ECCENTRICITY_RATIO, BEARING_CLEARANCE, BEARING_RADIUS, length, 0.0)
    return micropolar["load"] / newtonian["load"] - 1.0


def crossing(function, level: float) -> float | None:
    """m at which a change that falls with m passes a level, between 1 and 100 (bisection in log m)."""
    low, high = 1.0, 100.0
    if not function(low) > level > function(high):
        return None
    for _ in range(60):
        middle = np.sqrt(low * high)
        low, high = (middle, high) if function(middle) > level else (low, middle)
    return float(np.sqrt(low * high))


def rows_of(name):
    with open(style.HERE / name, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    style.setup()
    ratios = coupling_ratio(COUPLING_NUMBERS)
    exact = np.array([[data.exact_change(ratio, value) for ratio in ratios] for value in GAP_RATIOS])
    rule = np.array([[data.slip_rule_change(ratio, value) for ratio in ratios] for value in GAP_RATIOS])
    load = np.array([[load_change(ratio, value) for ratio in ratios] for value in GAP_RATIOS])
    torque_floor = style.load_json(CONCENTRIC)["noise_floors"]["200"]["relative_spread"]
    load_floor = style.load_json(ECCENTRIC)["floors"]["e60"]["load_change_spread"]
    concentric_rows = rows_of("tab_04_concentric_rows.csv")
    eccentric_rows = [row for row in rows_of("tab_09_eccentric_rows.csv")
                      if not row["run"].endswith(("_shifted", "_rotated"))]

    figure, axes = plt.subplots(1, 2, figsize=(style.WIDE, 0.50 * style.WIDE), sharey=True)
    labels = {level: f"{100.0 * level:g} %" for level in LEVELS}
    for axis, letter, field, look, floor in ((axes[0], "a", exact, style.EXACT, torque_floor),
                                             (axes[1], "b", load, {"color": style.INK, **style.FILM}, load_floor)):
        axis.contourf(COUPLING_NUMBERS, GAP_RATIOS, field, levels=[-1.0, floor], colors=[style.GRID])
        lines = axis.contour(COUPLING_NUMBERS, GAP_RATIOS, field, levels=LEVELS, colors=[look["color"]],
                             linestyles=[look["linestyle"]], linewidths=look["linewidth"])
        axis.set_yscale("log")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.8, 100.0)
        # a label on each contour at a chosen N: the value of m there from the field itself
        positions = []
        for level, place in zip(LEVELS, LABEL_PLACES[letter]):
            column = int(np.argmin(np.abs(COUPLING_NUMBERS - place)))
            profile = field[:, column]
            below = np.nonzero(profile < level)[0]
            if below.size and below[0] > 0:
                first = below[0]
                fraction = (profile[first - 1] - level) / (profile[first - 1] - profile[first])
                positions.append((COUPLING_NUMBERS[column],
                                  GAP_RATIOS[first - 1] * (GAP_RATIOS[first] / GAP_RATIOS[first - 1]) ** fraction))
        for label in axis.clabel(lines, fmt=labels, fontsize=style.FONT_SIZE - 1.5, inline_spacing=2,
                                 manual=positions):
            # the dashed contour of the rule runs under the label of the exact one: a white ground keeps it legible
            label.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": 0.8})
            label.set_zorder(7)
        axis.set_xlabel(style.math(r"N = \sqrt{K\,/\,(2 + K)}"))
        style.panel_letter(axis, letter, y=1.16)
        upper = axis.twiny()
        upper.set_xlim(0.0, 1.0)
        marks = (0.1, 0.5, 1.0, 3.0, 10.0)
        upper.set_xticks([float(coupling_number(value)) for value in marks])
        upper.set_xticklabels([f"{value:g}" for value in marks])
        upper.set_xlabel(style.math("K"), labelpad=2.0)
        upper.tick_params(labelsize=style.FONT_SIZE - 1.0)
    axes[0].contour(COUPLING_NUMBERS, GAP_RATIOS, rule, levels=LEVELS, colors=[style.INK],
                    linestyles=[style.RULE["linestyle"]], linewidths=style.RULE["linewidth"])
    axes[0].set_ylabel(style.math(style.GAP_OVER_TWO_LENGTHS))
    # the runs: concentric rows on (a), bearing rows on (b) (filled: eccentricity ratio 0.6, open: 0.3)
    series = {3.0: 0, 1.0: 1, 0.5: 2}
    marked = {"a": set(), "b": set()}
    for row in concentric_rows:
        ratio, value = float(row["K"]), float(row["m"])
        if (ratio, value) not in marked["a"]:
            axes[0].plot(coupling_number(ratio), value, zorder=5, **style.run_markers(series[ratio]))
            marked["a"].add((ratio, value))
    for row in eccentric_rows:
        ratio = float(row["K"])
        value = float(row["g/dx"]) / (2.0 * float(row["l/dx"]))
        if (ratio, value) not in marked["b"]:
            axes[1].plot(coupling_number(ratio), value, zorder=5, **style.run_markers(series[ratio]))
            marked["b"].add((ratio, value))
    figure.tight_layout(pad=0.4, w_pad=1.2)
    style.save(figure, "fig_16_effect_map")

    numbers = {"torque_noise_floor": torque_floor, "load_noise_floor": load_floor, "crossings": {}}
    text = []
    for ratio in (3.0, 1.0, 0.5):
        entry = {}
        for label, function in (("torque, exact", lambda value: data.exact_change(ratio, value)),
                                ("torque, slip-form rule", lambda value: data.slip_rule_change(ratio, value)),
                                ("load, film function", lambda value: load_change(ratio, value))):
            entry[label] = {f"{level:g}": crossing(function, level) for level in LEVELS}
        numbers["crossings"][f"K = {ratio:g}"] = entry

        def places(label):
            return ", ".join("beyond the frame" if entry[label][f"{level:g}"] is None
                             else f"{entry[label][f'{level:g}']:.1f}" for level in LEVELS)
        text.append(f"K = {ratio:g} (N = {coupling_number(ratio):.3f}): torque {places('torque, exact')} "
                    f"(rule {places('torque, slip-form rule')}), load {places('load, film function')}")
    caption = (
        "The size of the micropolar effect in the plane of the coupling number N = sqrt(K / (2 + K)) and of "
        "m = g / (2 l) (upper axis: K = kappa / mu); D is the relative change against the Newtonian fluid of "
        "viscosity mu_inf. Host only, no run. (a) Torque between the concentric cylinders of the chapter "
        f"(R1 = {data.INNER_RADIUS / data.GAP:g} g, R2 = {data.OUTER_RADIUS / data.GAP:g} g, w = 0 on both walls): "
        "contours D = 1 %, 5 % and 20 % of the exact solution (solid) and of the wall rule in its slip form "
        "(dashed). (b) Load of the bearing at an eccentricity ratio of 0.6 (R1 = 4 g, mean gap g, m built on the "
        "mean gap): the same contours from the film function of micropolar lubrication theory in the thin-film "
        "calculation (dash-dotted). Points: the runs of the chapter, in (a) the "
        f"{len(marked['a'])} places of the {len(concentric_rows)} rows of item 04, in (b) the {len(marked['b'])} "
        f"places of the {len(eccentric_rows)} rows of item 09 (every place holds runs at the eccentricity ratios 0.3 "
        "and 0.6; the contours are those of 0.6). Shaded: D below the noise floor of the runs, "
        f"{torque_floor:.1e} for the torque (three lattices at g = 200 dx) and {load_floor:.1e} for the change of the "
        "load (three lattices at eps = 0.6, measured at g = 100 dx; no three-lattice set of the bearing exists at "
        "g = 200 dx). m at which D equals 1 %, 5 % and 20 %: " + "; ".join(text) + ".")
    style.record("16", "The map of the effect: contours of D in the plane (N, m) for the torque and the load",
                 "final (host only)", caption, ["fig_16_effect_map.pdf", "fig_16_effect_map.png"],
                 [CONCENTRIC, ECCENTRIC, "tab_04_concentric_rows.csv", "tab_09_eccentric_rows.csv"], numbers, "host")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
