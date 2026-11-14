"""
Item 23 - the micropolar Taylor-Green vortex in the doubly periodic box: (a) the decay rate of the micropolar run
over that of its Newtonian control (viscosity mu + kappa / 2) against |k| l: the quasi-steady mu_eff / (mu + kappa / 2)
(j -> 0, solid), the exact two-mode solution with the run's j = l^2 (dashed) and the runs (400^2; the 800^2 run of
the resolution pair open); (b) U(t) and W(t) of the micro-inertia run (|k| l = 1, j = 0.755 l^2, W(0) = 0) against
the exact two-mode solution.

Run from the repository root:  python data/micropolar/ch5_draft/fig_23_taylor_green.py
"""
import math

import matplotlib.pyplot as plt
import numpy as np

import sources
import style

SOURCE = sources.PERIODIC
COUPLING_RATIO = 3.0
RUNS = (("tgv_kl025_n400", 0.25), ("tgv_kl05_n400", 0.5), ("tgv_kl1_n400", 1.0), ("tgv_kl2_n400", 2.0))
RESOLUTION = ("tgv_kl05_n800", 0.5)
INERTIA = "tgv_kl1_inertia_n400"
SMALL_INERTIA = (("tgv_kl05_j001_n400", 0.5), ("tgv_kl1_j001_n400", 1.0))
NEUTRAL = ("tgv_kl1_j150_n400", 1.0)


def quasi_steady(length_times_modulus):
    shear, coupling = 1.0, COUPLING_RATIO
    s = length_times_modulus ** 2 * (2.0 * shear + coupling) / (2.0 * (shear + coupling))
    return (shear + coupling / 2.0 + (coupling / 2.0) * s / (1.0 + s)) / (shear + coupling / 2.0)


def exact_ratio(length_times_modulus, theta_factor=1.0):
    """Slow rate of the two-mode system over the control's 2 k^2 nu_inf (k = 1, nu_inf = 1: the ratio does not
    depend on the Reynolds number)."""
    shear = 1.0 / (1.0 + COUPLING_RATIO / 2.0)
    coupling = COUPLING_RATIO * shear
    viscosity = shear + coupling
    length = length_times_modulus / math.sqrt(2.0)
    rotation = length ** 2 * coupling * (2.0 * viscosity - coupling) / viscosity
    theta = theta_factor * length ** 2
    matrix = np.array([[-2.0 * viscosity, coupling], [2.0 * coupling / theta, -(2.0 * rotation + 2.0 * coupling) / theta]])
    return float(-np.max(np.linalg.eigvals(matrix).real)) / 2.0


def main() -> int:
    style.setup()
    analysis = style.load_json(SOURCE["analysis"])
    tgv = analysis.get("tgv", {})
    present = [(run, value) for run, value in RUNS if run in tgv and "over_control" in tgv[run]]
    if not present:
        print("[pack] item 23: no micropolar Taylor-Green runs yet")
        return 0
    numbers = {}
    floors = analysis.get("floors", {})
    floor = max([abs(entry.get("over_control_difference", 0.0)) for entry in floors.values()] + [0.0])
    figure, axes = plt.subplots(1, 2, figsize=(style.WIDE, 0.42 * style.WIDE))
    axis = axes[0]
    curve = np.geomspace(0.1, 10.0, 200)
    # the curves carry direct labels; the legend holds the markers
    axis.plot(curve, [quasi_steady(value) for value in curve], **style.EXACT)
    axis.text(9.5, 1.555, r"quasi-steady ($j \to 0$)", ha="right", va="top", fontsize=style.FONT_SIZE - 1.5)
    axis.plot(curve, [exact_ratio(value) for value in curve], color=style.INK, linewidth=1.0, linestyle=(0, (4, 2)))
    axis.text(3.2, 1.22, r"$j = \ell^2$", ha="left", va="top", fontsize=style.FONT_SIZE - 1.5)
    axis.plot(curve, [exact_ratio(value, 1.5) for value in curve], color=style.MUTED, linewidth=0.7,
              linestyle=(0, (1, 1.5)))
    axis.text(9.0, 1.01, r"$j = 1.5\,\ell^2$", ha="right", va="bottom", fontsize=style.FONT_SIZE - 1.5,
              color=style.MUTED)
    axis.axhline(1.6, color=style.MUTED, linewidth=0.6, linestyle=(0, (1, 1.5)))
    small = [(run, value) for run, value in SMALL_INERTIA if run in tgv and "over_control" in tgv[run]]
    if small:
        axis.plot(curve, [exact_ratio(value, 0.01) for value in curve], color=style.SERIES[2][0], linewidth=0.8,
                  linestyle=(0, (6, 2, 1, 2)))
    colour, marker = style.SERIES[0]
    axis.errorbar([value for _run, value in present], [tgv[run]["over_control"] for run, _value in present],
                  yerr=floor, linestyle="none", marker=marker, markersize=style.MARKER_SIZE + 2.0, color=colour,
                  markerfacecolor="none", markeredgewidth=0.9, capsize=2.0, elinewidth=0.6,
                  label=r"runs, $j = \ell^2$, $400^2$")
    if small:
        axis.plot([value for _run, value in small], [tgv[run]["over_control"] for run, _value in small],
                  linestyle="none", marker=style.SERIES[2][1], markersize=style.MARKER_SIZE + 2.0,
                  markerfacecolor="none", markeredgewidth=0.9, color=style.SERIES[2][0],
                  label=r"runs, $j = \ell^2/100$")
    if NEUTRAL[0] in tgv and "over_control" in tgv[NEUTRAL[0]]:
        axis.plot([NEUTRAL[1]], [tgv[NEUTRAL[0]]["over_control"]], linestyle="none", marker=style.SERIES[3][1],
                  markersize=style.MARKER_SIZE + 1.5, markerfacecolor="none", markeredgewidth=0.9,
                  color=style.SERIES[3][0], label=r"run, $j = 1.5\,\ell^2$")
        numbers[NEUTRAL[0]] = {key: tgv[NEUTRAL[0]][key] for key in ("over_control", "exact_over_control",
                                                                      "rate_over_exact", "spin_ratio_over_exact")}
    if RESOLUTION[0] in tgv and "over_control" in tgv[RESOLUTION[0]]:
        axis.plot([RESOLUTION[1]], [tgv[RESOLUTION[0]]["over_control"]], linestyle="none", marker="s",
                  markersize=style.MARKER_SIZE + 3.5, markerfacecolor="none", markeredgecolor=style.SERIES[1][0],
                  markeredgewidth=0.9, label=r"run, $j = \ell^2$, $800^2$")
    fitted = present + [item for item in SMALL_INERTIA + (NEUTRAL,) if item[0] in tgv and "over_control" in tgv[item[0]]]
    axis.plot([value for _run, value in fitted], [tgv[run]["exact_over_control"] for run, _value in fitted],
              linestyle="none", marker="x", markersize=style.MARKER_SIZE, color=style.INK, markeredgewidth=0.8,
              zorder=5, label="exact, same fit")
    axis.set_xscale("log")
    axis.set_xlim(0.1, 10.0)
    axis.set_ylim(0.95, 1.65)
    axis.set_xlabel(r"$|\mathbf{k}|\,\ell$")
    axis.set_ylabel(r"rate over control, $\mu_\mathrm{eff}\,/\,(\mu + \kappa/2)$")
    handles, labels = axis.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda index: (not labels[index].startswith("runs, $j = \\ell^2$"),
                                                         "800" not in labels[index], "same fit" in labels[index]))
    axis.legend([handles[index] for index in order], [labels[index] for index in order],
                loc="upper left", bbox_to_anchor=(0.0, 0.92), handletextpad=0.4, fontsize=style.FONT_SIZE - 1.5,
                labelspacing=0.3, frameon=False)
    style.panel_letter(axis, "a")
    axis = axes[1]
    numbers.update({run: {key: tgv[run][key] for key in ("over_control", "exact_over_control", "mu_effective_ratio",
                                                          "rate_over_exact", "spin_ratio_over_exact")}
                    for run, _value in present + small})
    if INERTIA in tgv and "series" in tgv[INERTIA]:
        series = tgv[INERTIA]["series"]
        time = np.asarray(series["time"])
        every = max(1, time.size // 40)
        axis.plot(time, series["exact_U"], color=style.SERIES[0][0], linewidth=1.0, label=r"$U$, two-mode solution")
        axis.plot(time[::every], np.asarray(series["U"])[::every], linestyle="none", marker="o",
                  markersize=style.MARKER_SIZE - 0.5, color=style.SERIES[0][0], label=r"$U$, run")
        axis.plot(time, series["exact_W"], color=style.SERIES[1][0], linewidth=1.0, label=r"$W$, two-mode solution")
        axis.plot(time[::every], np.asarray(series["W"])[::every], linestyle="none", marker="s",
                  markersize=style.MARKER_SIZE - 0.5, color=style.SERIES[1][0], label=r"$W$, run")
        axis.set_xlabel(r"$t\,U_0\,k$")
        axis.set_ylabel(r"$U\,/\,U_0$,  $W\,/\,(k\,U_0)$")
        axis.set_xlim(0.0, float(time[-1]))
        axis.legend(loc="upper right", handletextpad=0.4)
        item = tgv[INERTIA]
        numbers[INERTIA] = {key: item[key] for key in ("rate_over_exact", "rms_error_U", "rms_error_W",
                                                       "exact_slow_rate", "exact_fast_rate", "spin_ratio_over_exact")}
    style.panel_letter(axis, "b")
    figure.tight_layout(pad=0.4, w_pad=1.4)
    style.save(figure, "fig_23_taylor_green")
    caption = (
        "The micropolar Taylor-Green vortex in the doubly periodic box [0, 2 pi)^2 (k = 1, |k| = sqrt 2, U0 = 1, "
        "Re = U0 / (k nu_inf) = 20, 50 on mu; K = 3; Ma 0.02; 400^2 particles, h = 5 dx; thesis numerics of "
        "Chapter 5). The field v = U(t) (sin kx cos ky, -cos kx sin ky), w = W(t) sin kx sin ky is an exact solution; "
        "U and W obey a linear 2 x 2 system (Section 18). (a) Decay rate of U of the micropolar run over that of its "
        "Newtonian control (viscosity mu + kappa / 2, the same particles and dt), fitted over 5 <= t <= 20, against "
        "|k| l: the quasi-steady mu_eff / (mu + kappa / 2) = 1 + 0.6 s / (1 + s), s = 0.625 (|k| l)^2 (solid; it "
        "tends to 1.6, dotted), the slow eigenvalue of the two-mode system with j = l^2 (dashed) and with "
        "j = l^2 / 100 (dash-dotted), the runs at 400^2 with j = l^2 (filled circles; bars: the largest floor of "
        "table 25), with j = l^2 / 100 (triangles) and with j = 1.5 l^2 (diamond), |k| l = 0.5 at 800^2 (open "
        "circle), and the exact two-mode solution fitted over the same window (crosses; at |k| l = 2 the fast mode, "
        "only 2.1 times faster, still enters the window). In a decaying flow the micro-rotation keeps spin from the "
        "earlier, larger vorticity: with j = l^2 the excess over mu + kappa / 2 is a third of the quasi-steady one "
        "at small |k| l, and with j = 1.5 l^2 (rho j (mu + kappa / 2) = gamma) w = omega / 2 is exact and the ratio "
        "is 1 at every |k| l. (b) The run |k| l = 1 with j = 0.755 l^2 (fast rate five times the slow one), "
        "started from W(0) = 0: U(t) and W(t) from the particles (projections on the two shapes; markers) against "
        "the exact two-mode solution (lines).")
    style.record("23", "Micropolar Taylor-Green vortex: mu_eff against |k| l, and the two-mode solution",
                 SOURCE["status"], caption, ["fig_23_taylor_green.pdf", "fig_23_taylor_green.png"],
                 [SOURCE["analysis"]], numbers, SOURCE["tag"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
