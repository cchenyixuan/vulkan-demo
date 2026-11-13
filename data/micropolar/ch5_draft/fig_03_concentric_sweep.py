"""
Item 03 - concentric cylinders: D against m for the three K: the exact solution as lines, the runs as points with
the noise floor, the rule in its slip form dashed; below it the ratios to the exact value.

Run from the repository root:  python data/micropolar/ch5_draft/fig_03_concentric_sweep.py
"""
import matplotlib.pyplot as plt
import numpy as np

import concentric_data as data
import sources
import style

SOURCE = sources.CONCENTRIC


def main() -> int:
    style.setup()
    runs = data.load(SOURCE)["runs"]
    figure, (upper, lower) = plt.subplots(2, 1, figsize=(style.WIDE * 0.72, style.WIDE * 0.80), sharex=True,
                                          gridspec_kw={"height_ratios": [1.5, 1.0]})
    values = np.geomspace(0.8, 60.0, 240)
    used, numbers = [], {}
    for index, coupling_ratio in enumerate(data.RATIOS):
        exact = np.array([data.exact_change(coupling_ratio, value) for value in values])
        rule = np.array([data.slip_rule_change(coupling_ratio, value) for value in values])
        colour = style.SERIES[index][0]
        upper.plot(values, exact, **style.EXACT)
        upper.plot(values, rule, color=colour, **style.RULE)
        lower.plot(values, rule / exact, color=colour, **style.RULE)
        members = sorted((record for record in runs.values() if record.get("coupling_ratio") == coupling_ratio
                          and "sweep" in record.get("sections", []) and "relative_torque_change" in record),
                         key=lambda record: record["gap_over_two_lengths"])
        used += [record["name"] for record in members]
        position = [record["gap_over_two_lengths"] for record in members]
        floor = [record["noise_floor_of_change"] or 0.0 for record in members]
        upper.errorbar(position, [record["relative_torque_change"] for record in members], yerr=floor,
                       ecolor=style.INK, elinewidth=0.6, capsize=1.5, zorder=3,
                       **style.run_markers(index), label=rf"$K = {coupling_ratio:g}$")
        lower.errorbar(position, [record["sph_over_exact"] for record in members],
                       yerr=[value / record["exact_hydrodynamic"] for value, record in zip(floor, members)],
                       ecolor=style.INK, elinewidth=0.6, capsize=1.5, zorder=3, **style.run_markers(index))
        upper.annotate(rf"$K = {coupling_ratio:g}$", (values[-1], exact[-1]), xytext=(3, 0),
                       textcoords="offset points", va="center", fontsize=style.FONT_SIZE - 1.0)
        numbers[f"K = {coupling_ratio:g}"] = {
            f"m = {record['gap_over_two_lengths']:g}": {
                "run": record["name"], "D": record["relative_torque_change"], "D_exact": record["exact_hydrodynamic"],
                "ratio": record["sph_over_exact"], "rule_ratio": record["slip_rule_over_exact"],
                "floor": (record["noise_floor_of_change"] or 0.0) / record["exact_hydrodynamic"]}
            for record in members}
    lower.axhline(1.0, **style.EXACT)
    upper.set_xscale("log")
    upper.set_yscale("log")
    upper.set_xlim(0.8, 130.0)
    upper.set_ylabel(style.math(style.TORQUE_CHANGE))
    lower.set_ylim(0.955, 1.065)
    lower.set_xlabel(style.math(style.GAP_OVER_TWO_LENGTHS))
    lower.set_ylabel(r"ratio to $D_{\mathrm{exact}}$")
    lower.set_xticks([1, 2, 5, 10, 20, 40])
    lower.set_xticklabels(["1", "2", "5", "10", "20", "40"])
    handles = [plt.Line2D([], [], **style.EXACT, label="exact"),
               plt.Line2D([], [], color=style.MUTED, **style.RULE, label="rule, slip form"),
               plt.Line2D([], [], **{**style.run_markers(0), "color": style.MUTED, "markeredgecolor": style.MUTED,
                                     "markerfacecolor": style.MUTED}, label="runs")]
    upper.legend(handles=handles, loc="lower left")
    style.panel_letter(upper, "a")
    style.panel_letter(lower, "b")
    figure.tight_layout(pad=0.4, h_pad=0.8)
    style.save(figure, "fig_03_concentric_sweep")

    def ratios(gap_over_two_lengths):
        found = [entry[f"m = {gap_over_two_lengths:g}"] for entry in numbers.values()
                 if f"m = {gap_over_two_lengths:g}" in entry]
        if not found:
            return None
        return (min(item["ratio"] for item in found), max(item["ratio"] for item in found),
                min(item["rule_ratio"] for item in found), max(item["rule_ratio"] for item in found))
    parts = []
    for value in (1, 2, 5, 10, 20, 40):
        found = ratios(value)
        if found:
            parts.append(f"m = {value}: runs {found[0]:.3f} to {found[1]:.3f}, rule {found[2]:.3f} to {found[3]:.3f}")
    lengths = sorted({(record["gap_over_two_lengths"], record["length_in_spacings"], record["gap_in_spacings"])
                      for record in runs.values() if record["name"] in used})
    caption = (
        "Relative change D of the torque between concentric cylinders (R1 = 4 g, R2 = 5 g, w = 0 on both walls) "
        "against m = g / (2 l) for K = 3, 1 and 0.5 (Re = 10 on mu_inf; 25, 15 and 12.5 on mu; Ma = 0.02). "
        "(a) Exact solution (solid), the wall rule in its slip form (dashed; every wall displaced by "
        "delta = K l / (2 (1 + K)) into the fluid, the rotor's apparent speed reduced by 2 delta Omega) and the "
        "runs (points; bars: noise floor of D from three lattices, smaller than the symbols; the row at g = 400 dx "
        "has one lattice and carries the floor of g = 200 dx). (b) The same as "
        "ratios to the exact value; the rule leaves the frame below m of about 1.9 (1.36 to 1.49 at m = 1). "
        "Resolution of the rows (m, l / dx, g / dx): "
        + ", ".join(f"({m:g}, {length:g}, {gap})" for m, length, gap in lengths)
        + f". Runs: the sweep rows of tab_04; tag {data.tags(runs, used)}. Ratio to exact: " + "; ".join(parts) + ".")
    style.record("03", "Concentric cylinders: D against m for three K, with the slip-form rule", SOURCE["status"],
                 caption, ["fig_03_concentric_sweep.pdf", "fig_03_concentric_sweep.png"],
                 [SOURCE["analysis"]] + used, numbers, SOURCE["tag"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
