"""
Item 01 - concentric cylinders: v_theta(r) and w(r) of the runs against the exact solution, K = 3 at m = 2 and
m = 10, with insets of both wall layers.

Run from the repository root:  python data/micropolar/ch5_draft/fig_01_concentric_profiles.py
"""
import matplotlib.pyplot as plt
import numpy as np

import concentric_data as data
import sources
import style

SOURCE = sources.CONCENTRIC
RUNS = ["g100_k3_m2", "g200_k3_m10"]            # K = 3: m = 2 (l = 25 dx) and m = 10 (l = 10 dx)
INSET_WIDTH = 0.08                               # in gaps


def main() -> int:
    style.setup()
    runs = data.load(SOURCE)["runs"]
    present = [name for name in RUNS if name in runs and "exact_profile" in runs[name]]
    if len(present) < len(RUNS):
        print("[pack] item 01: runs missing:", sorted(set(RUNS) - set(present)))
        return 1
    figure, axes = plt.subplots(2, 2, figsize=(style.WIDE, 0.72 * style.WIDE), sharex=True)
    numbers = {}
    for column, name in enumerate(RUNS):
        record = runs[name]
        profile, exact = record["profile"], record["exact_profile"]
        radius = (np.asarray(profile["radius"]) - data.INNER_RADIUS) / data.GAP
        exact_radius = (np.asarray(exact["radius"]) - data.INNER_RADIUS) / data.GAP
        stride = max(1, radius.size // 40)
        for row, (key, scale, label) in enumerate((("velocity", data.WALL_SPEED, style.VELOCITY),
                                                   ("spin", data.ROTATION_RATE, style.SPIN))):
            axis = axes[row][column]
            measured = np.asarray(profile[key]) / scale
            expected = np.asarray(exact[key]) / scale
            axis.plot(exact_radius, expected, **style.EXACT, label="exact")
            axis.plot(radius[stride // 2::stride], measured[stride // 2::stride],
                      **style.run_markers(0, filled=False), label="run")
            axis.set_xlim(0.0, 1.0)
            if column == 0:
                axis.set_ylabel(style.math(label))
            if row == 1:
                axis.set_xlabel(style.math(style.RADIUS_IN_GAP))
            style.panel_letter(axis, "abcd"[2 * row + column])
            axis.text(1.0, 1.03, rf"$m = {record['gap_over_two_lengths']:g}$, "
                      rf"${style.LENGTH_IN_SPACINGS} = {record['length_in_spacings']:g}$",
                      transform=axis.transAxes, ha="right", va="bottom", fontsize=style.FONT_SIZE - 1.0)
            # insets: the layer at the rotor and the layer at the stator, every bin (one per dx)
            if key == "velocity":
                places = ([0.17, 0.10, 0.28, 0.34], [0.66, 0.58, 0.28, 0.34])
            else:
                places = ([0.20, 0.56, 0.26, 0.36], [0.56, 0.56, 0.26, 0.36])
            for place, (low, high) in zip(places, ((0.0, INSET_WIDTH), (1.0 - INSET_WIDTH, 1.0))):
                inset = axis.inset_axes(place)
                chosen = (radius >= low) & (radius <= high)
                chosen_exact = (exact_radius >= low) & (exact_radius <= high)
                inset.plot(exact_radius[chosen_exact], expected[chosen_exact], **style.EXACT)
                inset.plot(radius[chosen], measured[chosen], **{**style.run_markers(0, filled=False),
                                                                "markersize": 2.5, "markeredgewidth": 0.6})
                inset.set_xlim(low, high)
                inset.set_xticks([low, high])
                inset.set_xticklabels([f"{low:g}", f"{high:g}"])
                inset.tick_params(labelsize=style.FONT_SIZE - 2.5, length=2.0, top=False, right=False)
                inset.yaxis.set_major_locator(plt.MaxNLocator(3))
                for spine in inset.spines.values():
                    spine.set_linewidth(0.4)
        numbers[name] = {key: record[key] for key in ("velocity_error_core", "velocity_error_wall_layers",
                                                      "spin_error_core", "spin_error_wall_layers",
                                                      "relative_torque_change", "exact_hydrodynamic")}
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.tight_layout(pad=0.4, w_pad=1.2, h_pad=1.0, rect=(0.0, 0.0, 1.0, 0.95))
    figure.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    style.save(figure, "fig_01_concentric_profiles")

    first, second = (runs[name] for name in RUNS)
    caption = (
        "Circular Couette flow of the micropolar fluid between concentric cylinders (R1 = 4 g, R2 = 5 g, rotor at "
        "Omega, w = 0 on both walls): bin means of the azimuthal velocity (a, b) and of the micro-rotation (c, d) "
        "over the last quarter of the run (bins of one dx, about forty shown; every bin in the insets, which cover "
        f"the {INSET_WIDTH:g} g next to the rotor and to the stator) against the exact solution (line). "
        f"(a, c) {data.parameters(first)}, run `{RUNS[0]}`; (b, d) {data.parameters(second)}, run `{RUNS[1]}`. "
        f"Tag {data.tags(runs, RUNS)}. Rms deviation from the exact profile, relative to the rms of the profile: "
        + "; ".join(
            f"m = {runs[name]['gap_over_two_lengths']:g}: v {runs[name]['velocity_error_core']:.1e} in the core and "
            f"{runs[name]['velocity_error_wall_layers']:.1e} within h of a wall, w "
            f"{runs[name]['spin_error_core']:.1e} and {runs[name]['spin_error_wall_layers']:.1e}" for name in RUNS)
        + ".")
    style.record("01", "Concentric cylinders: profiles against the exact solution", SOURCE["status"], caption,
                 ["fig_01_concentric_profiles.pdf", "fig_01_concentric_profiles.png"],
                 [SOURCE["analysis"]] + RUNS, numbers, SOURCE["tag"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
