"""
Item 11 - micropolar lid-driven cavity: centre-line profiles with the two Newtonian bounds, fields of w and of
w - curl v / 2 with the slab cut of the run on two slabs marked, and a small table (tab_11_micropolar_cavity).

Run from the repository root:  python data/micropolar/ch5_draft/fig_11_micropolar_cavity.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

import style

SOURCE = "data/micropolar/stage6/cavity.json"
FIELD = "data/micropolar/stage6/fields/cavity_{run}"
# the runs of the figure: the micropolar run, its two bounds, the coarser run, the run on two slabs
RUNS = {"micropolar": "mp_n500", "far": "newtonian_re400_n500", "wall": "newtonian_re250_n500",
        "coarse": "mp_n250", "chain": "mp_n500_k2"}
TAG = "thesis-ch5-rc2"


def main() -> int:
    style.setup()
    if not style.exists(SOURCE):
        print("[pack] item 11: no cavity analysis yet")
        return 1
    analysis = style.load_json(SOURCE)
    runs = analysis["runs"]
    if RUNS["micropolar"] not in runs:
        print("[pack] item 11: the micropolar run is missing")
        return 1
    main_run = runs[RUNS["micropolar"]]
    coordinate = np.asarray(main_run["coordinate"])
    field_run = next((RUNS[key] for key in ("chain", "micropolar") if style.exists(FIELD.format(run=RUNS[key]) + ".npz")),
                     None)
    figure = plt.figure(figsize=(style.WIDE, 0.92 * style.WIDE))
    grid = figure.add_gridspec(2, 2, height_ratios=[0.8, 1.0])
    upper = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    lower = [figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])]
    used = []
    for axis, letter, name, label, horizontal in ((upper[0], "a", "u", style.CAVITY_U, r"y\,/\,L"),
                                                  (upper[1], "b", "v", style.CAVITY_V, r"x\,/\,L")):
        for key, look, text in (("far", {"color": style.INK, "linewidth": 0.8, "linestyle": "-"},
                                 r"Newtonian, $\mu_\infty$ ($\mathit{Re} = 400$)"),
                                ("wall", {"color": style.INK, "linewidth": 0.8, "linestyle": (0, (4, 2))},
                                 r"Newtonian, $\mu + \kappa$ ($\mathit{Re} = 250$)")):
            if RUNS[key] in runs:
                axis.plot(coordinate, runs[RUNS[key]][f"{name}_mean"], label=text, **look)
                used.append(RUNS[key])
        axis.plot(coordinate[10::25], np.asarray(main_run[f"{name}_mean"])[10::25],
                  **style.run_markers(0, filled=False), label=r"micropolar, $n = 500$")
        if RUNS["coarse"] in runs:
            axis.plot(coordinate[22::25], np.asarray(runs[RUNS["coarse"]][f"{name}_mean"])[22::25],
                      **{**style.run_markers(1), "marker": "x", "markersize": 3.5}, label=r"micropolar, $n = 250$")
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel(style.math(horizontal))
        axis.set_ylabel(style.math(label))
        style.panel_letter(axis, letter)
    used += [RUNS["micropolar"]] + ([RUNS["coarse"]] if RUNS["coarse"] in runs else [])
    upper[0].legend(loc="upper left", handletextpad=0.4)
    numbers = {}
    if field_run:
        data = np.load(style.REPO_ROOT / (FIELD.format(run=field_run) + ".npz"))
        summary = json.loads((style.REPO_ROOT / (FIELD.format(run=field_run) + ".json")).read_text(encoding="utf-8"))
        spin = data["spin"]
        excess = data["spin"] - 0.5 * data["curl"]
        points_per_length = 0.5 * style.WIDE * 72.0 * 0.72
        marker_area = (1.35 * summary["spacing"] * points_per_length) ** 2
        for axis, letter, values, label in ((lower[0], "c", spin, style.CAVITY_SPIN),
                                            (lower[1], "d", excess, style.CAVITY_EXCESS_SPIN)):
            bound = float(np.percentile(np.abs(values), 99.5))
            image = axis.scatter(data["x"], data["y"], c=values, s=marker_area, marker="s", linewidths=0.0,
                                 cmap=style.FIELD_SIGNED, norm=TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound),
                                 rasterized=True)
            for cut in summary.get("cuts_x") or []:
                axis.axvline(cut, color=style.INK, linewidth=0.6, linestyle=(0, (3, 2)))
            axis.set_aspect("equal")
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.set_xlabel(style.math(r"x\,/\,L"))
            axis.set_ylabel(style.math(r"y\,/\,L"))
            bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
            bar.set_label(style.math(label))
            bar.ax.tick_params(labelsize=style.FONT_SIZE - 1.5)
            bar.outline.set_linewidth(0.4)
            style.panel_letter(axis, letter)
            numbers[letter] = {"min": float(values.min()), "max": float(values.max()), "colour_bound": bound}
        used.append(field_run)
    else:
        for axis in lower:
            axis.set_axis_off()
    figure.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
    style.save(figure, "fig_11_micropolar_cavity")

    # the table
    header = ["run", "fluid", "n", "slabs", "end time", "steady at", "u_min", "y at u_min", "v_max", "x at v_max",
              "v_min", "x at v_min", "kinetic energy", "density level", "density drift per L/U",
              "pressure level", "pressure drift per L/U", "git describe"]
    tex_header = ["run", "fluid", "$n$", "slabs", r"$u_{\min}/U$", r"$y/L$", r"$v_{\max}/U$", r"$x/L$",
                  r"$v_{\min}/U$", r"$x/L$", r"$E_k$", r"$\dot{\bar\rho}/\rho_0$", r"$\dot{\bar p}/(\rho U^2)$"]
    labels = {"mp_n500": "micropolar", "mp_n250": "micropolar", "mp_n500_k2": "micropolar",
              "mp_n500_l40": "micropolar, l = L / 40",
              "newtonian_re400_n500": "Newtonian, mu_inf (Re 400)", "newtonian_re250_n500": "Newtonian, mu + kappa (Re 250)"}
    plain, tex = [], []
    for name, item in runs.items():
        steady = (item.get("steady") or {}).get("steady_time")
        describe = (item.get("git") or {}).get("describe", "")
        cells = [name, labels.get(name, ""), f"{item['resolution']}", f"{item['slabs']}", f"{item['end_time']:.1f}",
                 f"{steady:.1f}" if steady is not None else "-", f"{item['u_min']:.5f}", f"{item['y_at_u_min']:.4f}",
                 f"{item['v_max']:.5f}", f"{item['x_at_v_max']:.4f}", f"{item['v_min']:.5f}",
                 f"{item['x_at_v_min']:.4f}", f"{item['kinetic_energy_mean']:.5f}",
                 f"{item['density_level']:+.1e}" if "density_level" in item else "-",
                 f"{item['density_drift_per_time_unit']:+.1e}" if "density_level" in item else "-",
                 f"{item['pressure_level']:+.4f}" if "density_level" in item else "-",
                 f"{item['pressure_drift_per_time_unit']:+.1e}" if "density_level" in item else "-", describe]
        plain.append(cells)
        tex.append([style.texttt(name), style.tex_escape(cells[1]).replace("mu\\_inf", r"$\mu_\infty$")
                    .replace("mu + kappa", r"$\mu + \kappa$"), cells[2], cells[3], *cells[6:13], cells[14], cells[16]])
    style.write_table("tab_11_micropolar_cavity", header, plain, tex_header, "ll" + "r" * (len(tex_header) - 2), tex)

    comparisons = analysis.get("comparisons", {})
    bounds = comparisons.get("bounds")
    text = ""
    if bounds:
        text += (" Between the two bounds: u_min " + ("yes" if bounds["u_min"] else "no")
                 + f" ({bounds['u_min_fraction']:.2f} of the way from the Re 400 value to the Re 250 value), v_max "
                 + ("yes" if bounds["v_max"] else "no") + f" ({bounds['v_max_fraction']:.2f}), v_min "
                 + ("yes" if bounds["v_min"] else "no") + f" ({bounds['v_min_fraction']:.2f}); along the centre lines "
                 f"the micropolar profile leaves the band of the two bounds at {100.0 * bounds['u_outside_share']:.0f} % "
                 f"(u) and {100.0 * bounds['v_outside_share']:.0f} % (v) of the points, by at most "
                 f"{bounds['u_outside_max']:.1e} and {bounds['v_outside_max']:.1e} U.")
    for key, words in (("resolution", "n = 250 minus n = 500"), ("chain", "two slabs minus one slab")):
        if key in comparisons:
            entry = comparisons[key]
            text += (f" {words.capitalize()}: rms difference of the centre lines {entry['u_difference_rms']:.1e} (u) and "
                     f"{entry['v_difference_rms']:.1e} (v) U, extrema {entry['u_min']:+.1e}, {entry['v_max']:+.1e}, "
                     f"{entry['v_min']:+.1e} U (standard error of the time means {entry['u_noise_rms']:.0e}).")
    parameters = analysis["parameters"]
    caption = (
        "Lid-driven cavity filled with the micropolar fluid: K = 3, Reynolds number 400 on mu_inf (1,000 on mu), "
        "l = L / 20 (25 dx at n = 500, 12.5 dx at n = 250), Theta = l^2, w = 0 on the four walls, Ma = 0.01, thesis "
        "configuration with the default walls. (a) u on the vertical centre line and (b) v on the horizontal centre "
        "line, time means over the last 20 L / U (MLS samples at 1,001 points, every 25th shown): the micropolar run "
        "(circles; crosses: n = 250) between two Newtonian runs on the same particles, one with the viscosity "
        "mu_inf that the micropolar fluid has far from walls (solid) and one with mu + kappa that it has inside a "
        "wall layer (dashed). (c) Micro-rotation and (d) micro-rotation minus half the curl of the velocity at the "
        f"last step of run `{field_run}`"
        + (" (two slabs, 5090 + XTX; dashed line: the cut)" if field_run == RUNS["chain"] else "")
        + "; the second vanishes in the core and marks the wall layers of thickness l. Colour scales end at the "
        "99.5th percentile of the magnitude. Runs: " + ", ".join(f"`{name}`" for name in dict.fromkeys(used))
        + f"; tag `{TAG}`." + text
        + " No reference table for Reynolds number 400 is in the repository (the Marchi et al. files hold 1,000, "
        "3,200 and 5,000), so the Newtonian bound at 400 is not compared with a reference here.")
    style.record("11", "Micropolar cavity: centre lines with the two Newtonian bounds, fields, the slab cut",
                 "final" if all(RUNS[key] in runs for key in RUNS) and field_run == RUNS["chain"]
                 else "final (rc2 runs), incomplete: " + ", ".join(
                     RUNS[key] for key in RUNS if RUNS[key] not in runs) + (
                     "" if field_run == RUNS["chain"] else "; fields of the two-slab run"),
                 caption, ["fig_11_micropolar_cavity.pdf", "fig_11_micropolar_cavity.png",
                           "tab_11_micropolar_cavity.csv", "tab_11_micropolar_cavity.tex"],
                 [SOURCE] + list(dict.fromkeys(used)), {"fields": numbers, "parameters": parameters}, TAG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
