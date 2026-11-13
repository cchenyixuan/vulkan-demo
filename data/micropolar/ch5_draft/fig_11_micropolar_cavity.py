"""
Item 11 - micropolar lid-driven cavity: centre-line profiles against the finite-difference solution, fields of w and of
w - curl v / 2 with the slab cut of the run on two slabs marked, and a small table (tab_11_micropolar_cavity).
A second table (tab_11b_cavity_finite_differences) and a dotted line in (a) and (b) carry the same cases solved by
finite differences on the host (experiment/validation/micropolar_cavity_reference.py).

Run from the repository root:  python data/micropolar/ch5_draft/fig_11_micropolar_cavity.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

import style

SOURCE = "data/micropolar/stage6/cavity.json"
FIELD = "data/micropolar/stage6/fields/cavity_{run}"
REFERENCE = "data/micropolar/stage6/cavity_reference.json"
LID = "data/micropolar/stage6/cavity_reference_lid.json"
# finite-difference case -> what the table calls it
REFERENCE_CASES = {"newtonian_far": "Newtonian, mu_inf (Re 400)", "newtonian_wall": "Newtonian, mu + kappa (Re 250)",
                   "micropolar": "micropolar, l = L / 20", "micropolar_l40": "micropolar, l = L / 40",
                   "micropolar_l80": "micropolar, l = L / 80", "micropolar_l160": "micropolar, l = L / 160",
                   "micropolar_l10": "micropolar, l = L / 10",
                   "micropolar_no_inertia": "micropolar, l = L / 20, Theta -> 0",
                   "micropolar_free_spin": "micropolar, l = L / 20, walls do not hold w",
                   "micropolar_lid_spin": "micropolar, l = L / 20, only the lid holds w",
                   "micropolar_rest_spin": "micropolar, l = L / 20, only the walls at rest hold w"}
# the runs of the figure: the micropolar run, two Newtonian runs on the same particles (viscosities mu_inf and
# mu + kappa), the coarser run, the run on two slabs
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
    reference = style.load_json(REFERENCE)["cases"] if style.exists(REFERENCE) else {}
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
        if "micropolar" in reference:
            finest = reference["micropolar"]["grids"][-1]
            axis.plot(finest["coordinate"], finest[name], color=style.INK, linewidth=0.8, linestyle=(0, (1, 1.5)),
                      label="micropolar, finite\ndifferences")
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
    # the lines in the legend of (a), the markers in the one of (b): together they do not fit into one panel
    handles, labels_ = upper[0].get_legend_handles_labels()
    lines_ = [(handle, label) for handle, label in zip(handles, labels_) if handle.get_marker() in ("None", "", None)]
    markers_ = [(handle, label) for handle, label in zip(handles, labels_) if (handle, label) not in lines_]
    upper[0].legend(*zip(*lines_), loc="upper left", handletextpad=0.4, handlelength=1.7,
                    fontsize=style.FONT_SIZE - 1.0, borderaxespad=0.3)
    if markers_:
        upper[1].legend(*zip(*markers_), loc="lower left", handletextpad=0.2, fontsize=style.FONT_SIZE - 1.0,
                        borderaxespad=0.3)
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
              "v_min", "x at v_min", "kinetic energy over fluid mass / U^2", "density level", "density drift per L/U",
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
    style.write_table("tab_11_micropolar_cavity", header, plain, tex_header, "ll" + "r" * (len(tex_header) - 2), tex,
                      note="$E_k$: kinetic energy of the fluid over its mass, in $U^2$ (the kinetic energy per unit "
                           "depth in $\\rho U^2 L^2$ of a fluid that fills the square).")

    # the same cases by finite differences: the check of the case
    reference_numbers, check_text = {}, ""
    if reference:
        header_b = ["case", "fluid", "grids", "u_min (finest grid)", "u_min (extrapolated)", "order",
                    "v_max (finest grid)", "v_max (extrapolated)", "order", "v_min (finest grid)",
                    "v_min (extrapolated)", "order", "kinetic energy (extrapolated)",
                    "u_min over that of the Newtonian fluid of viscosity mu_inf", "v_max", "v_min", "particle run",
                    "u_min particle / finite differences", "v_max", "v_min", "kinetic energy",
                    "rms difference of u on the centre line", "of v"]
        tex_header_b = ["fluid", r"$u_{\min}/U$", r"$v_{\max}/U$", r"$v_{\min}/U$", r"$E_k$",
                        r"ratio $u_{\min}$", r"ratio $v_{\max}$", r"ratio $v_{\min}$", r"ratio $E_k$"]
        plain_b, tex_b = [], []
        for case, label in REFERENCE_CASES.items():
            if case not in reference:
                continue
            entry = reference[case]
            zero, finest = entry["extrapolated"], entry["grids"][-1]

            def order(key):
                return f"{zero[key]['order']:.2f}" if zero[key]["order"] is not None else "-"
            over_far = (entry.get("against_bounds") or {}).get("over_far")
            particle = entry.get("particle")
            ratios = particle["over_finite_differences"] if particle else None
            plain_b.append(
                [case, label, " ".join(str(value) for value in entry["resolutions"]),
                 f"{finest['u_min']:.5f}", f"{zero['u_min']['extrapolated']:.5f}", order("u_min"),
                 f"{finest['v_max']:.5f}", f"{zero['v_max']['extrapolated']:.5f}", order("v_max"),
                 f"{finest['v_min']:.5f}", f"{zero['v_min']['extrapolated']:.5f}", order("v_min"),
                 f"{zero['kinetic_energy']['extrapolated']:.5f}"]
                + ([f"{over_far[key]:.4f}" for key in ("u_min", "v_max", "v_min")] if over_far else ["-"] * 3)
                + ([particle["run"]] + [f"{ratios[key]:.4f}" for key in ("u_min", "v_max", "v_min", "kinetic_energy")]
                   + [f"{particle['u_difference_rms']:.1e}", f"{particle['v_difference_rms']:.1e}"]
                   if particle else ["-"] * 7))
            tex_b.append(
                [style.tex_escape(label).replace("mu\\_inf", r"$\mu_\infty$").replace("mu + kappa", r"$\mu + \kappa$")
                 .replace("Theta -> 0", r"$\Theta \to 0$").replace("hold w", "hold $w$").replace("holds w", "holds $w$")
                 .replace("l = L / ", "$l = L/$"),
                 f"{zero['u_min']['extrapolated']:.4f}", f"{zero['v_max']['extrapolated']:.4f}",
                 f"{zero['v_min']['extrapolated']:.4f}", f"{zero['kinetic_energy']['extrapolated']:.4f}"]
                + ([f"{ratios[key]:.4f}" for key in ("u_min", "v_max", "v_min", "kinetic_energy")]
                   if particle else ["--"] * 4))
            reference_numbers[case] = {"extrapolated": {key: zero[key]["extrapolated"] for key in zero},
                                       "over_newtonian_far": over_far,
                                       "particle": {key: value for key, value in particle.items()}
                                       if particle else None}
        style.write_table("tab_11b_cavity_finite_differences", header_b, plain_b, tex_header_b,
                          "l" + "r" * (len(tex_header_b) - 1), tex_b,
                          note="Finite differences, extrapolated to zero spacing; ratio: the particle run at "
                               "$n = 500$ over the finite-difference value. $E_k$: half the mean square speed, in "
                               "$U^2$.")
        if all(case in reference and "particle" in reference[case]
               for case in ("newtonian_far", "newtonian_wall", "micropolar")):
            far, wall, polar = (reference[case] for case in ("newtonian_far", "newtonian_wall", "micropolar"))

            def triple(entry):
                return ", ".join(f"{entry['extrapolated'][key]['extrapolated']:.4f}"
                                 for key in ("u_min", "v_max", "v_min"))

            def excess(entry):
                ratios = entry["particle"]["over_finite_differences"]
                return ", ".join(f"{100.0 * (ratios[key] - 1.0):+.1f}" for key in ("u_min", "v_max", "v_min"))
            check_text = (
                " The check of the case is the dotted line and table 11b: the same equations solved by finite "
                "differences on the host (stream function, vorticity and micro-rotation on a uniform grid, second "
                "order, steady state; grids of " + ", ".join(str(value) for value in polar["resolutions"])
                + " intervals, extrema extrapolated to zero spacing). Extrema (u_min, v_max, v_min) of the "
                f"micropolar fluid: particle run {main_run['u_min']:.4f}, {main_run['v_max']:.4f}, "
                f"{main_run['v_min']:.4f} U, finite differences " + triple(polar) + " U; the particle run is "
                "stronger by " + excess(polar) + " %, and its centre lines differ from the finite-difference ones "
                f"by {polar['particle']['u_difference_rms']:.1e} (u) and {polar['particle']['v_difference_rms']:.1e} "
                "(v) U rms. The two Newtonian runs against their finite-difference solutions: " + excess(far)
                + " % (mu_inf), " + excess(wall) + " % (mu + kappa). Table 11b also lists finite-difference "
                "variants of the micropolar case (other lengths l, no spin inertia, walls that do not hold w, w "
                "held on the lid only or on the walls at rest only).")

    # table 11c: the traction of the lid against delta / lambda
    files_c = []
    if style.exists(LID):
        lid = style.load_json(LID)
        header_c = ["case", "l / L", "delta / L", "delta / lambda", "traction of the lid over that on the Newtonian "
                    "fluid of viscosity mu_inf, minus one (finest common grid)", "grid (intervals)",
                    "spread over the common grids", "on every common grid", "excess over delta / lambda"]
        tex_header_c = [r"$l/L$", r"$\delta/L$", r"$\delta/\lambda$", r"$\tau/\tau_\infty - 1$",
                        r"$(\tau/\tau_\infty - 1)\,/\,(\delta/\lambda)$"]
        plain_c, tex_c = [], []
        for row in lid["rows"]:
            like = ", ".join(f"{value:.4f} (n = {grid})" for grid, value in row["excess_like_grids"].items())
            plain_c.append([row["case"], f"1/{1.0 / row['length']:.0f}", f"{row['displacement']:.5f}",
                            f"{row['displacement_over_shear_length']:.4f}", f"{row['excess']:.4f}",
                            f"{row['excess_grid']}", f"{row['excess_spread_over_like_grids']:.4f}", like,
                            f"{row['excess_over_first_order']:.3f}"])
            tex_c.append([f"$1/{1.0 / row['length']:.0f}$", f"{row['displacement']:.4f}",
                          f"{row['displacement_over_shear_length']:.3f}", f"{row['excess']:.3f}",
                          f"{row['excess_over_first_order']:.2f}"])
        style.write_table("tab_11c_lid_traction", header_c, plain_c, tex_header_c, "rrrrr", tex_c,
                          note="Finite differences. $\\tau$: traction of the lid on the micropolar fluid (mean over "
                               "$0.1 \\le x/L \\le 0.9$), $\\tau_\\infty$: on the Newtonian fluid of viscosity "
                               "$\\mu_\\infty$ on the same grid; $\\lambda = \\mu_\\infty U / \\tau_\\infty = "
                               f"{lid['shear_length']:.4f}\\,L$; $\\delta = K l / (2 (1 + K))$.")
        files_c = ["tab_11c_lid_traction.csv", "tab_11c_lid_traction.tex"]
        reference_numbers["lid"] = {"shear_length": lid["shear_length"], "rows": lid["rows"]}
        check_text += (
            " Table 11c: the traction of the lid on the micropolar fluid over that on the Newtonian fluid of "
            "viscosity mu_inf, minus one, is "
            + ", ".join(f"{row['excess']:.3f}" for row in lid["rows"]) + " at l = L / "
            + ", ".join(f"{1.0 / row['length']:.0f}" for row in lid["rows"]) + ", against delta / lambda = "
            + ", ".join(f"{row['displacement_over_shear_length']:.3f}" for row in lid["rows"])
            + f", where lambda = mu_inf U / traction = {lid['shear_length']:.3f} L is the shear length of the lid in "
            "that Newtonian fluid (finite differences; both tractions on the same grid, 256 intervals, 384 for "
            "l = L / 80): the first-order rule with the shear length in the place of the gap, within "
            + f"{100.0 * max(abs(row['excess_over_first_order'] - 1.0) for row in lid['rows'][-2:]):.0f} % from "
            "l = L / 40 down.")

    comparisons = analysis.get("comparisons", {})
    text = ""
    for key, words in (("resolution", "The run at n = 250 minus the run at n = 500"),
                       ("chain", "Two slabs minus one slab")):
        if key in comparisons:
            entry = comparisons[key]
            text += (f" {words}: rms difference of the centre lines {entry['u_difference_rms']:.1e} (u) and "
                     f"{entry['v_difference_rms']:.1e} (v) U, extrema {entry['u_min']:+.1e}, {entry['v_max']:+.1e}, "
                     f"{entry['v_min']:+.1e} U (standard error of the time means {entry['u_noise_rms']:.0e}).")
    parameters = analysis["parameters"]
    caption = (
        "Lid-driven cavity filled with the micropolar fluid: K = 3, Reynolds number 400 on mu_inf (1,000 on mu), "
        "l = L / 20 (25 dx at n = 500, 12.5 dx at n = 250), Theta = l^2, w = 0 on the four walls, Ma = 0.01, thesis "
        "configuration with the default walls. (a) u on the vertical centre line and (b) v on the horizontal centre "
        "line, time means over the last 20 L / U (MLS samples at 1,001 points, every 25th shown): the micropolar run "
        "(circles; crosses: n = 250) and the same fluid by finite differences (dotted); solid and dashed: Newtonian "
        "runs on the same particles with the viscosities mu_inf (Reynolds number 400) and mu + kappa (Reynolds "
        "number 250). (c) Micro-rotation and (d) micro-rotation minus half the curl of the velocity at the "
        f"last step of run `{field_run}`"
        + (" (two slabs, 5090 + XTX; dashed line: the cut)" if field_run == RUNS["chain"] else "")
        + "; the second vanishes in the core and marks the wall layers of thickness l. Colour scales end at the "
        "99.5th percentile of the magnitude. Runs: " + ", ".join(f"`{name}`" for name in dict.fromkeys(used))
        + f"; tag `{TAG}`." + check_text + text)
    style.record("11", "Micropolar cavity: centre lines against the finite-difference solution, fields, the slab cut",
                 "final" if all(RUNS[key] in runs for key in RUNS) and field_run == RUNS["chain"]
                 else "final (rc2 runs), incomplete: " + ", ".join(
                     RUNS[key] for key in RUNS if RUNS[key] not in runs) + (
                     "" if field_run == RUNS["chain"] else "; fields of the two-slab run"),
                 caption, ["fig_11_micropolar_cavity.pdf", "fig_11_micropolar_cavity.png",
                           "tab_11_micropolar_cavity.csv", "tab_11_micropolar_cavity.tex"]
                 + (["tab_11b_cavity_finite_differences.csv", "tab_11b_cavity_finite_differences.tex"]
                    if reference else []) + files_c,
                 [SOURCE] + ([REFERENCE] if reference else []) + ([LID] if files_c else [])
                 + list(dict.fromkeys(used)),
                 {"fields": numbers, "parameters": parameters, "finite_differences": reference_numbers}, TAG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
