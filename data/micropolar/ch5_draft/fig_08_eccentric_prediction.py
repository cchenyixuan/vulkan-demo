"""
Item 08 - eccentric cylinders: measured change over predicted change against the narrowest gap over l, for the load
and for the torque; prediction by the film function of micropolar lubrication theory (thin-film calculation) and by
the wall rule (exact Stokes solution of the slip-form geometry); every micropolar run of 2.2, and the concentric
rows for the rule.

Run from the repository root:  python data/micropolar/ch5_draft/fig_08_eccentric_prediction.py
"""
import matplotlib.pyplot as plt
import numpy as np

import concentric_data
import sources
import style

SOURCE = sources.ECCENTRIC
CONCENTRIC = sources.CONCENTRIC
CASES = {0.3: "o", 0.6: "s"}                 # eccentricity ratio -> marker
CONCENTRIC_MARKER = "^"
RATIO_COLOUR = {3: style.SERIES[0][0], 1: style.SERIES[1][0], 0.5: style.SERIES[2][0]}
RATIO_SIZE = {3: style.MARKER_SIZE + 1.0, 1: style.MARKER_SIZE - 0.5, 0.5: style.MARKER_SIZE - 1.5}


def main() -> int:
    style.setup()
    analysis = style.load_json(SOURCE["analysis"])
    pairs = {name: entry for name, entry in analysis["pairs"].items() if not entry.get("lattice")}
    figure, axes = plt.subplots(1, 2, figsize=(style.WIDE, 0.44 * style.WIDE), sharey=True)
    numbers = {"load": {}, "torque": {}}
    for name, entry in pairs.items():
        ratio, position = entry["coupling_ratio"], entry["narrowest_gap_over_length"]
        look = {"linestyle": "none", "marker": CASES[round(entry["eccentricity_ratio"], 2)],
                "markersize": RATIO_SIZE[ratio], "markeredgecolor": RATIO_COLOUR[ratio], "markeredgewidth": 0.8}
        for axis, key, film_key, rule_key in ((axes[0], "load", "load_over_film", "load_over_rule"),
                                              (axes[1], "torque", "torque_over_film", "torque_over_rule")):
            axis.plot(position, entry[film_key], markerfacecolor=RATIO_COLOUR[ratio], **look)
            axis.plot(position, entry[rule_key], markerfacecolor="white", **look)
            numbers[key][name] = {"narrowest_gap_over_length": position, "over_film": entry[film_key],
                                  "over_rule": entry[rule_key]}
    # the concentric rows: D of the run over D of the rule in its slip form, at g / l = 2 m
    concentric = concentric_data.load(CONCENTRIC)["runs"]
    used_concentric = []
    for name, record in concentric.items():
        if "relative_torque_change" not in record or "sweep" not in record["sections"]:
            continue
        ratio = record["coupling_ratio"]
        key = 3 if ratio == 3 else (1 if ratio == 1 else 0.5)
        axes[1].plot(2.0 * record["gap_over_two_lengths"], record["relative_torque_change"] / record["slip_rule"],
                     linestyle="none", marker=CONCENTRIC_MARKER, markersize=RATIO_SIZE[key], markerfacecolor="white",
                     markeredgecolor=RATIO_COLOUR[key], markeredgewidth=0.8)
        used_concentric.append(name)
        numbers["torque"][name] = {"narrowest_gap_over_length": 2.0 * record["gap_over_two_lengths"],
                                   "over_rule": record["relative_torque_change"] / record["slip_rule"]}
    for axis, letter, label in zip(axes, "ab", ("load", "torque")):
        axis.axhline(1.0, **style.EXACT)
        axis.set_xscale("log")
        axis.set_xlim(1.3, 50.0)
        axis.set_xticks([2, 4, 8, 16, 32])
        axis.set_xticklabels(["2", "4", "8", "16", "32"])
        axis.set_xticks([], minor=True)
        axis.set_xlabel(rf"narrowest gap$\,/\,\ell$")
        style.panel_letter(axis, letter)
        axis.text(1.0, 1.03, label, transform=axis.transAxes, ha="right", va="bottom", fontsize=style.FONT_SIZE - 1.0)
    axes[0].set_ylabel("measured change / predicted change")
    axes[0].set_ylim(0.3, 1.12)
    grey = style.MUTED
    handles = [plt.Line2D([], [], linestyle="none", marker="o", markersize=style.MARKER_SIZE, markerfacecolor=grey,
                          markeredgecolor=grey, label="against the film function"),
               plt.Line2D([], [], linestyle="none", marker="o", markersize=style.MARKER_SIZE, markerfacecolor="white",
                          markeredgecolor=grey, label="against the rule"),
               plt.Line2D([], [], linestyle="none", marker="o", markersize=style.MARKER_SIZE, markerfacecolor="white",
                          markeredgecolor=grey, label=r"$\varepsilon = 0.3$"),
               plt.Line2D([], [], linestyle="none", marker="s", markersize=style.MARKER_SIZE, markerfacecolor="white",
                          markeredgecolor=grey, label=r"$\varepsilon = 0.6$"),
               plt.Line2D([], [], linestyle="none", marker="^", markersize=style.MARKER_SIZE, markerfacecolor="white",
                          markeredgecolor=grey, label="concentric")]
    ratios = sorted({entry["coupling_ratio"] for entry in pairs.values()}
                    | {3, 1, 0.5} & {concentric[name]["coupling_ratio"] for name in used_concentric}, reverse=True)
    for ratio in ratios:
        key = 3 if ratio == 3 else (1 if ratio == 1 else 0.5)
        handles.append(plt.Line2D([], [], linestyle="none", marker="o", markersize=RATIO_SIZE[key],
                                  markerfacecolor=RATIO_COLOUR[key], markeredgecolor=RATIO_COLOUR[key],
                                  label=rf"$K = {ratio:g}$"))
    axes[0].legend(handles=handles[:5], loc="lower right", handletextpad=0.2)
    axes[1].legend(handles=handles[5:], loc="lower right", handletextpad=0.2)
    figure.tight_layout(pad=0.4, w_pad=1.0)
    style.save(figure, "fig_08_eccentric_prediction")

    def span(key, which):
        values = [entry[which] for entry in numbers[key].values() if which in entry and entry.get(which) is not None]
        return f"{min(values):.3f} to {max(values):.3f}"
    eccentric_torque = {name: entry for name, entry in numbers["torque"].items() if "over_film" in entry}
    caption = (
        "What the two predictions of the host make of the micropolar effect between eccentric cylinders: measured "
        "change over predicted change, (a) of the load and (b) of the rotor torque, against the narrowest gap in "
        "units of l (concentric rows: g / l). Change = micropolar run over its Newtonian control, minus one; the "
        "torque change is the mean of the four readings (rotor and stator, wall form and circle form) with the error "
        "of each reading between concentric cylinders at the same K, l and g taken off where that row exists. "
        "Filled: against the film function of micropolar lubrication theory in the thin-film calculation (load: the "
        "plane film; torque: the shear-driven stress of the exact concentric solution at the local gap, the "
        "pressure-driven stress of the plane film). Open: against the wall rule, evaluated on the exact Stokes "
        "solution of its slip-form geometry (rotor radius R1 + delta turning at Omega (R1 - delta) / (R1 + delta), "
        "stator radius R2 - delta); triangles: concentric rows, D of the run over D of the rule. "
        f"Runs: the rows of tab_09 on the lattice as generated ({SOURCE['note']}) and the sweep rows of tab_04. "
        f"Load over film {span('load', 'over_film')}, load over rule {span('load', 'over_rule')}; torque over film "
        f"{min(e['over_film'] for e in eccentric_torque.values()):.3f} to "
        f"{max(e['over_film'] for e in eccentric_torque.values()):.3f}, torque over rule (eccentric) "
        f"{min(e['over_rule'] for e in eccentric_torque.values()):.3f} to "
        f"{max(e['over_rule'] for e in eccentric_torque.values()):.3f}.")
    style.record("08", "Eccentric cylinders: measured over predicted change against narrowest gap / l",
                 SOURCE["status"], caption, ["fig_08_eccentric_prediction.pdf", "fig_08_eccentric_prediction.png"],
                 [SOURCE["analysis"], CONCENTRIC["analysis"]] + list(pairs) + used_concentric, numbers,
                 f"{SOURCE['tag']}; concentric: {CONCENTRIC['tag']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
