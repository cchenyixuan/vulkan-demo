"""
Item 27 - the start-up of plane Couette flow as a test of the saturation of the wall-stress excess (Stage 8c):
(a) the excess E = tau_mp / tau_N - 1 on the sliding wall against Lambda / l (log axis): the host curves (j = l^2,
l^2 / 100, the quasi-steady limit), the runs with their floors, the two limits (sqrt(1.6) - 1 and the steady value
of eq. 5.5), and the peak wall-stress excess of the dipole runs of tables 22 and 26 at their Lambda / l (another
flow); (b) u and (c) w at Lambda = 0.25 l, l, 4 l against the host solution.

Run from the repository root:  python data/micropolar/ch5_draft/fig_27_startup.py
"""
import matplotlib.pyplot as plt
import numpy as np

import sources
import style

SOURCE = sources.CHANNEL
HOST_CURVES = (("theta_1", r"host, $j = \ell^2$", style.INK, "-"),
               ("theta_0.01", r"host, $j = \ell^2/100$", style.SERIES[2][0], (0, (6, 2, 1, 2))),
               ("theta_0.0001", r"host, $j = \ell^2/10^4$", style.MUTED, (0, (1, 1.2))))
RUNS = (("startup_l20", r"run, $\ell = 20\,\Delta x$", 0), ("startup_l40", r"run, $\ell = 40\,\Delta x$", 1),
        ("startup_l80", r"run, $\ell = 80\,\Delta x$", 3), ("startup_l40_j001", r"run, $\ell = 40\,\Delta x$, $j = \ell^2/100$", 2))
DIPOLE_CHECKS = (("mp_l100_n1000__against__control_n1000", r"$\ell = W/100$"),
                 ("mp_l50_n1000__against__control_n1000", r"$\ell = W/50$"),
                 ("mp_l50_n500__against__control_n500", r"$\ell = W/50$, $W/\Delta x = 500$"),
                 ("mp_l50_j001_n1000__against__control_n1000", r"$\ell = W/50$, $j = \ell^2/100$"),
                 ("mp_l25_j001_n1000__against__control_n1000", r"$\ell = W/25$, $j = \ell^2/100$"),
                 ("mp_l100_n2000__against__control_n2000", r"$\ell = W/100$, $W/\Delta x = 2000$"),
                 ("mp_l100_re1250_n1000__against__control_re1250_n1000", r"$\ell = W/100$, Re 1250"),
                 ("mp_l50_re1250_n1000__against__control_re1250_n1000", r"$\ell = W/50$, Re 1250"))
PROFILE_RUN = "startup_l40"
# the j = l^2 / 100 pair (dt 21 times smaller, 9.15 million steps) drifts, run and control alike, beyond Lambda ~ 4 l
# (the control's stress 1.10 of the host at Lambda = 10 l; design note 19.4): drawn to Lambda = 4 l
LAYER_LIMIT = {"startup_l40_j001": 4.0}
MINIMUM_LAYER_IN_SPACINGS = 2.0


def main() -> int:
    style.setup()
    if not style.exists(SOURCE["host"]):
        print("[pack] item 27: no host solution yet")
        return 0
    host = style.load_json(SOURCE["host"])
    analysis = style.load_json(SOURCE["analysis"]) if style.exists(SOURCE["analysis"]) else {}
    startup = analysis.get("startup", {})
    dipole = style.load_json(SOURCE["dipole"]) if style.exists(SOURCE["dipole"]) else {}
    figure = plt.figure(figsize=(style.WIDE, 0.92 * style.WIDE))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.25, 1.0), hspace=1.10, wspace=0.30)
    axis = figure.add_subplot(grid[0, :])
    numbers = {"early_limit": host["early_limit"], "steady_excess_eq_5_5": host["steady_excess_eq_5_5"],
               "convergence": host["convergence"]}
    for key, label, colour, dashes in HOST_CURVES:
        curve = host["curves"].get(key)
        if curve:
            axis.plot(curve["layer_over_length"], curve["excess"], color=colour, linestyle=dashes, linewidth=1.0,
                      label=label)
    axis.axhline(host["early_limit"], color=style.MUTED, linewidth=0.6, linestyle=(0, (4, 3)))
    axis.text(0.0105, host["early_limit"] + 0.004, r"$\sqrt{1.6} - 1$", fontsize=style.FONT_SIZE - 1.5,
              color=style.MUTED, va="bottom")
    steady_layer = host["gap_over_length"]
    axis.plot([steady_layer], [host["steady_excess_eq_5_5"]], linestyle="none", marker="*", markersize=9,
              color=style.INK, label="steady, eq. 5.5")
    for name, label, series_index in RUNS:
        entry = startup.get(name)
        if not entry:
            continue
        colour, marker = style.SERIES[series_index]
        layer = np.asarray(entry["layer_over_length"])
        excess = np.asarray(entry["excess_fit"])
        # a layer thinner than two spacings is not resolved by any reading: drawn from Lambda = 2 dx
        upper = LAYER_LIMIT.get(name, float("inf"))
        shown = (layer >= MINIMUM_LAYER_IN_SPACINGS / entry["length_in_spacings"]) & (layer <= upper)
        if "excess_pairs" in entry:
            pair_layer = np.asarray(entry["pairs_layer_over_length"])
            pair_shown = (pair_layer >= MINIMUM_LAYER_IN_SPACINGS / entry["length_in_spacings"]) & (pair_layer <= upper)
            axis.plot(pair_layer[pair_shown], np.asarray(entry["excess_pairs"])[pair_shown], color=colour,
                      linewidth=1.3, alpha=0.85, label=label + ", wall form")
        layer_shown, excess_shown = layer[shown], excess[shown]
        every = max(1, layer_shown.size // 30)
        axis.plot(layer_shown[::every], excess_shown[::every], linestyle="none", marker=marker,
                  markersize=style.MARKER_SIZE - 0.5, markerfacecolor="none", markeredgewidth=0.7, color=colour,
                  label=label + ", 4 dx fit")
        floor = np.asarray(entry.get("floor", []))
        if floor.size == layer.size and name == "startup_l20" and "excess_pairs" in entry:
            pairs = np.interp(np.log(layer), np.log(pair_layer), np.asarray(entry["excess_pairs"]))
            axis.fill_between(layer[shown], (pairs - floor)[shown], (pairs + floor)[shown], color=style.MUTED,
                              alpha=0.3, linewidth=0.0, label=r"floor ($\ell = 20\,\Delta x$)")
        numbers[name] = {"smallest_resolved_layer_over_length": entry.get("smallest_resolved_layer_over_length")}
    checks = dipole.get("checks", {})
    dipole_points = []
    for key, label in DIPOLE_CHECKS:
        check = checks.get(key)
        if check and check.get("shear_length_over_length") == check.get("shear_length_over_length"):
            dipole_points.append((check["shear_length_over_length"], check["changes"]["wall_traction_peak"], label))
    if dipole_points:
        axis.plot([point[0] for point in dipole_points], [point[1] for point in dipole_points], linestyle="none",
                  marker="X", markersize=style.MARKER_SIZE + 2.0, color=style.INK,
                  label="dipole-wall collision, peak wall stress")
        numbers["dipole"] = {label: [layer, excess] for layer, excess, label in dipole_points}
    axis.set_xscale("log")
    axis.set_xlim(0.01, 30.0)
    axis.set_ylim(-0.02, 0.32)
    axis.set_xlabel(r"$\Lambda / \ell$,  $\Lambda = (\mu + \kappa/2)\,U / \tau_N$")
    axis.set_ylabel(r"$E = \tau_\mathrm{mp} / \tau_N - 1$")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), fontsize=style.FONT_SIZE - 3.0, ncol=3,
                frameon=False, handletextpad=0.4, columnspacing=1.2)
    style.panel_letter(axis, "a")
    # (b) u and (c) w at three layer thicknesses
    entry = startup.get(PROFILE_RUN, {})
    profiles = entry.get("profiles", {})
    for column, (key, ylabel, letter) in enumerate((("u", r"$u / U$", "b"), ("w", r"$w\,\ell / U$", "c"))):
        axis = figure.add_subplot(grid[1, column])
        for index, (label, profile) in enumerate(sorted(profiles.items(), key=lambda item: float(item[0]))):
            colour = style.SERIES[index][0]
            y = np.asarray(profile["y"])
            every = max(1, int(round(y.size / 160)))
            axis.plot(y[::every], np.asarray(profile[key])[::every], linestyle="none", marker="o",
                      markersize=style.MARKER_SIZE - 1.5, color=colour, alpha=0.8,
                      label=rf"run, $\Lambda = {label}\,\ell$")
            host_profile = profile.get("host")
            if host_profile:
                axis.plot(host_profile["y"], host_profile[key], color=colour, linewidth=1.0)
            numbers[f"profile_{label}"] = {name: profile.get(f"{name}_rms_error") for name in ("u", "w")}
        axis.set_xlim(0.0, 12.0)
        axis.set_xlabel(r"$y / \ell$ (sliding wall at 0)")
        axis.set_ylabel(ylabel)
        if column == 0:
            axis.plot([], [], color=style.INK, linewidth=1.0, label="host")
            axis.legend(loc="upper right", fontsize=style.FONT_SIZE - 2.5, frameon=False)
        style.panel_letter(axis, letter)
    style.save(figure, "fig_27_startup")
    caption = (
        "The start-up of plane Couette flow (gap H = 20 l, the lower wall set impulsively in motion at U, w = 0 on "
        "both walls, K = 3) as a test of the saturation of the wall-stress excess. (a) E = tau_mp / tau_N - 1 on "
        "the sliding wall against Lambda / l, Lambda = (mu + kappa / 2) U / tau_N of the Newtonian fluid mu + kappa "
        "/ 2 (for the Rayleigh layer sqrt(pi nu_inf t)): the host solution (implicit, stretched grid, converged to "
        "3e-5 in E) for j = l^2, l^2 / 100 and l^2 / 10^4 (the quasi-steady limit), the runs (l = 20, 40, 80 dx, "
        "j = l^2; l = 40 dx with j = l^2 / 100, drawn to Lambda = 4 l, beyond which that pair at its 21 times smaller dt "
        "drifts, its control as well; lines: the wall form, open markers: the 4 dx fit; both from Lambda = 2 dx) with the floor "
        "of l = 20 dx (repeat, half Mach; grey), the early limit sqrt(1.6) - 1 (a layer thinner than l is a "
        "Newtonian layer of viscosity mu + kappa) and the steady value of eq. 5.5 (star). Crosses: the peak "
        "wall-stress excess of the dipole-wall collision runs (tables 22, 26, 29) at their Lambda / l - another flow, "
        "whose outer flow changes with l. (b, c) u and w across the gap at Lambda = 0.25 l, l and 4 l: the run at "
        "l = 40 dx (markers) against the host solution (lines).")
    style.record("27", "Start-up of plane Couette flow: the wall-stress excess against Lambda / l, and the profiles",
                 SOURCE["status"], caption, ["fig_27_startup.pdf", "fig_27_startup.png"],
                 [SOURCE["host"], SOURCE["analysis"], SOURCE["dipole"]], numbers, SOURCE["tag"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
