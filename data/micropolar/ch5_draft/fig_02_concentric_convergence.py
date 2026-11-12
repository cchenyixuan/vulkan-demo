"""
Item 02 - concentric cylinders: deviation of D from the exact value against l / dx for the two resolution series
(log-log, fitted order), and the profile errors of v and w in the core and in the wall layers.

Run from the repository root:  python data/micropolar/ch5_draft/fig_02_concentric_convergence.py
"""
import matplotlib.pyplot as plt
import numpy as np

import concentric_data as data
import sources
import style

SOURCE = sources.CONCENTRIC
SERIES = {"m = 5": ["g050_k3_m5", "g100_k3_m5", "g200_k3_m5"],
          "m = 10": ["g100_k3_m10", "g200_k3_m10", "g400_k3_m10"]}
SUPPORT_IN_SPACINGS = 5.0


def main() -> int:
    style.setup()
    analysis = data.load(SOURCE)
    runs = analysis["runs"]
    figure, axes = plt.subplots(1, 3, figsize=(style.WIDE, 0.36 * style.WIDE))
    pooled_length, pooled_deviation, used = [], [], []
    numbers = {}
    for index, (label, names) in enumerate(SERIES.items()):
        present = [name for name in names if name in runs and "difference" in runs[name]]
        used += present
        length = np.array([runs[name]["length_in_spacings"] for name in present])
        deviation = np.array([abs(runs[name]["difference"]) / runs[name]["exact_hydrodynamic"] for name in present])
        floor = np.array([runs[name]["noise_floor_of_change"] / runs[name]["exact_hydrodynamic"] for name in present])
        filled = index == 0
        line = {"color": style.SERIES[index][0], "linewidth": 0.6, "linestyle": "-" if filled else (0, (3, 2))}
        axes[0].plot(length, deviation, **{**style.run_markers(index, filled),
                                           "markerfacecolor": line["color"] if filled else "none",
                                           "markersize": style.MARKER_SIZE + (0.0 if filled else 2.0)},
                     label=rf"$m = {label[4:]}$", zorder=3)
        # the noise floor of D (spread of beta over the three lattices), as a short bar under each point
        axes[0].plot(length, floor, linestyle="none", marker="_", markersize=9.0, color=style.SERIES[index][0])
        above = deviation > floor
        pooled_length += list(length[above])
        pooled_deviation += list(deviation[above])
        for axis, core, wall in ((axes[1], "velocity_error_core", "velocity_error_wall_layers"),
                                 (axes[2], "spin_error_core", "spin_error_wall_layers")):
            axis.plot(length, [runs[name][core] for name in present], marker="o", markersize=style.MARKER_SIZE,
                      markerfacecolor=line["color"] if filled else "none", markeredgecolor=line["color"],
                      markeredgewidth=0.8, **line)
            axis.plot(length, [runs[name][wall] for name in present], marker="s", markersize=style.MARKER_SIZE,
                      markerfacecolor=line["color"] if filled else "none", markeredgecolor=line["color"],
                      markeredgewidth=0.8, **line)
        numbers[label] = {name: {"length_in_spacings": runs[name]["length_in_spacings"],
                                 "deviation": runs[name]["difference"] / runs[name]["exact_hydrodynamic"],
                                 "noise_floor": runs[name]["noise_floor_of_change"] / runs[name]["exact_hydrodynamic"]}
                          for name in present}
    order, intercept = np.polyfit(np.log(pooled_length), np.log(pooled_deviation), 1)
    span = np.array([4.0, 25.0])
    axes[0].plot(span, np.exp(intercept) * span ** order, color=style.INK, linewidth=0.6, linestyle=(0, (1, 1.5)))
    axes[0].text(0.95, 0.93, f"order {-order:.1f}", transform=axes[0].transAxes, ha="right", va="top",
                 fontsize=style.FONT_SIZE - 1.0)
    numbers["fitted_order"] = float(-order)
    axes[1].set_ylim(8.0e-5, 3.0e-3)
    axes[2].set_ylim(1.0e-3, 3.0e-2)
    for axis, letter, label in zip(axes, "abc", (r"$|D - D_{\mathrm{exact}}|\,/\,D_{\mathrm{exact}}$",
                                                 r"rms error of $v_\theta$", r"rms error of $w$")):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks([5, 10, 20])
        axis.set_xticklabels(["5", "10", "20"])
        axis.set_xticks([], minor=True)
        axis.set_xlim(4.0, 26.0)
        axis.set_xlabel(style.math(style.LENGTH_IN_SPACINGS))
        axis.set_ylabel(label)
        axis.yaxis.set_minor_formatter(plt.NullFormatter())
        style.panel_letter(axis, letter)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=style.MUTED, marker="_", markersize=9.0, linestyle="none"))
    axes[0].legend(handles, labels + ["noise floor"], loc="lower left", handletextpad=0.2)
    handles = [plt.Line2D([], [], color=style.MUTED, marker="o", markersize=style.MARKER_SIZE, linestyle="none",
                          label="core"),
               plt.Line2D([], [], color=style.MUTED, marker="s", markersize=style.MARKER_SIZE, linestyle="none",
                          label="wall layers")]
    axes[1].legend(handles=handles, loc="upper right", handletextpad=0.2)
    figure.tight_layout(pad=0.4, w_pad=1.0)
    style.save(figure, "fig_02_concentric_convergence")

    def listing(key, form):
        return "; ".join(f"{label}: " + ", ".join(
            form.format(runs[name][key] if not callable(key) else key(runs[name])) for name in names if name in runs
            and "difference" in runs[name]) for label, names in SERIES.items())
    caption = (
        "Convergence between concentric cylinders, K = 3, w = 0 on both walls, in the micropolar length resolved "
        "(l / dx = 5, 10, 20; h = 5 dx; Ma = 0.02, Re = 10 on mu_inf, 25 on mu). (a) Deviation of the relative "
        "torque change D (velocity fit, against the run's Newtonian control) from the exact value; horizontal bars: "
        "the noise floor of D (spread of the control's beta over three lattices); dotted: power law through the "
        f"points above their floor, order {-order:.1f}. (b), (c) Rms deviation of the bin means of v_theta and of w "
        "from the exact profiles, relative to the rms of the profile, further than h from both walls (circles) and "
        "within h of a wall (squares). Filled symbols, solid lines: m = 5 (runs "
        + ", ".join(f"`{name}`" for name in SERIES["m = 5"] if name in used) + "); open symbols, dashed lines: "
        "m = 10 (" + ", ".join(f"`{name}`" for name in SERIES["m = 10"] if name in used) + "). "
        f"Tag {data.tags(runs, used)}. Deviation of D: "
        + listing(lambda record: 100.0 * record["difference"] / record["exact_hydrodynamic"], "{:+.2f} %") + ".")
    missing = [name for names in SERIES.values() for name in names if name not in used]
    if missing:
        caption += " Not in the figure yet: " + ", ".join(f"`{name}`" for name in missing) + "."
    style.record("02", "Concentric cylinders: convergence of D and of the profiles", SOURCE["status"], caption,
                 ["fig_02_concentric_convergence.pdf", "fig_02_concentric_convergence.png"],
                 [SOURCE["analysis"]] + used, numbers, SOURCE["tag"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
