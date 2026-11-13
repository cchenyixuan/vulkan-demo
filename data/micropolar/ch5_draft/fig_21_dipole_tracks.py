"""
Item 21 - the dipole-wall collision (Clercx & Bruneau 2006, Re = 625 on nu_inf): (a) trajectories of the primary
(positive) and the secondary (negative, detached) vortex cores of the upper half for the control and the two
micropolar runs, with the primary positions of the benchmark; (b) the enstrophy Omega(t) of the three runs with
the benchmark values (the two peaks of its Table 2, the values of its Table 5).

Run from the repository root:  python data/micropolar/ch5_draft/fig_21_dipole_tracks.py
"""
import matplotlib.pyplot as plt
import numpy as np

import sources
import style

SOURCE = sources.DIPOLE
RUNS = (("control_n1000", "control"),
        ("mp_l100_n1000", r"micropolar $\ell = W/100$"),
        ("mp_l50_n1000", r"micropolar $\ell = W/50$"))


def main() -> int:
    style.setup()
    analysis = style.load_json(SOURCE["analysis"])
    runs = analysis["runs"]
    present = [(name, label) for name, label in RUNS if name in runs]
    if not present:
        print("[pack] item 21: no dipole runs yet")
        return 0
    benchmark = analysis["benchmark"]
    figure, axes = plt.subplots(1, 2, figsize=(style.WIDE, 0.43 * style.WIDE),
                                gridspec_kw={"width_ratios": [1.0, 1.25]})
    tracks_axis, enstrophy_axis = axes
    # (a) the tracks
    tracks_axis.axvline(1.0, color=style.INK, linewidth=1.2)
    for index, (name, label) in enumerate(present):
        colour = style.SERIES[index][0]
        tracks = runs[name]["tracks"]
        primary, secondary = tracks["primary"], tracks["secondary"]
        tracks_axis.plot(primary["x"], primary["y"], color=colour, linewidth=0.9, label=label)
        if secondary["x"]:
            tracks_axis.plot(secondary["x"], secondary["y"], color=colour, linewidth=0.8, linestyle=(0, (3, 1.5)))
        # the positions at the times of the benchmark's table
        for time in (0.6, 0.625, 1.0):
            key = f"{time:g}"
            if key in runs[name]["primary"]:
                x, y, _value = runs[name]["primary"][key]
                tracks_axis.plot(x, y, marker=style.SERIES[index][1], markersize=3.0, color=colour,
                                 markerfacecolor="white", markeredgewidth=0.7, linestyle="none")
    for method, marker in (("SM", "s"), ("FD", "D")):
        points = np.asarray(benchmark["primary"][method])
        tracks_axis.plot(points[:, 1], points[:, 2], marker=marker, markersize=3.5, color=style.INK,
                         linestyle="none", label=f"benchmark ({method})")
    tracks_axis.set_xlim(0.3, 1.02)
    tracks_axis.set_ylim(0.0, 0.5)
    tracks_axis.set_aspect("equal")
    tracks_axis.set_xlabel(r"$x\,/\,W$")
    tracks_axis.set_ylabel(r"$y\,/\,W$")
    style.panel_letter(tracks_axis, "a")
    # (b) the enstrophy
    numbers = {}
    for index, (name, label) in enumerate(present):
        colour = style.SERIES[index][0]
        series = runs[name]["series"]
        enstrophy_axis.plot(series["time"], series["enstrophy"], color=colour, linewidth=0.9, label=label)
        numbers[name] = {"enstrophy_peaks": runs[name]["enstrophy_peaks"]}
    peaks = np.asarray(benchmark["enstrophy_peaks"]["SM"])
    enstrophy_axis.plot(peaks[:, 0], peaks[:, 1], marker="s", markersize=3.5, color=style.INK, linestyle="none",
                        label="benchmark (SM)")
    table = benchmark["global"]["SM"]
    times = sorted(float(key) for key in table)
    enstrophy_axis.plot(times, [table[f"{time}" if f"{time}" in table else str(time)][1] for time in times],
                        marker="o", markersize=3.0, color=style.INK, linestyle="none", markerfacecolor="white")
    enstrophy_axis.plot([0.0], [benchmark["initial"]["enstrophy"]], marker="o", markersize=3.0, color=style.INK,
                        markerfacecolor="white", linestyle="none")
    enstrophy_axis.set_xlim(0.0, 1.0)
    enstrophy_axis.set_xlabel(r"$t\,U\,/\,W$")
    enstrophy_axis.set_ylabel(r"$\Omega\,/\,U^2$")
    # lower left: before t = 0.3 every curve is above 450
    enstrophy_axis.legend(loc="lower left", handlelength=1.4, handletextpad=0.3, fontsize=style.FONT_SIZE - 1.5)
    style.panel_letter(enstrophy_axis, "b")
    figure.tight_layout(pad=0.4, w_pad=1.2)
    style.save(figure, "fig_21_dipole_tracks")
    control = runs.get("control_n1000")
    caption = (
        "Normal collision of a dipole with the no-slip wall x = W (Clercx & Bruneau 2006; box [-W, W]^2, Re = U W / "
        "nu_inf = 625, U the rms velocity), the Newtonian control (viscosity mu_inf) and the micropolar fluid "
        "(K = 3, l = W / 100 and W / 50, Theta = l^2, w = 0 on every wall, w = omega / 2 at t = 0), dx = W / 1000. "
        "(a) Trajectories of the cores of the upper half: the primary (positive) vortex (solid) and the secondary "
        "(negative) vortex once detached from the wall (dashed); open symbols: the runs at t = 0.6, 0.625 and 1.0; "
        "black: the primary positions of the benchmark (spectral SM at t = 0.6 and 1.0, finite differences FD at "
        "t = 0.625 and 1.0). The primary core is the local maximum of the vorticity on a grid of one spacing at "
        "every sample, smoothed in time by a local quadratic fit over +-0.0125 W / U (the maximum of a flat core "
        "moves by about 0.005 W between nodes); the secondary core is the minimum of the vorticity field saved "
        "every 0.025 W / U (Gaussian of two spacings, more than 0.02 W from the wall) from t = 0.45 on. (b) "
        "Enstrophy "
        "Omega = 1/2 int omega^2 against time; black squares: the two peaks of the benchmark (its Table 2), open "
        "circles: its values at t = 0, 0.25, 0.5 and 0.75 (Table 5).")
    if control:
        caption += (f" Control: peaks {control['enstrophy_peaks'][0][1]:.1f} at t = {control['enstrophy_peaks'][0][0]:.4f} "
                    f"and {control['enstrophy_peaks'][1][1]:.1f} at t = {control['enstrophy_peaks'][1][0]:.4f} "
                    "(benchmark 933.6 at 0.3711 and 305.2 at 0.6479).")
    style.record("21", "Dipole-wall collision: core trajectories and enstrophy against the benchmark", SOURCE["status"],
                 caption, ["fig_21_dipole_tracks.pdf", "fig_21_dipole_tracks.png"],
                 [SOURCE["analysis"]] + [name for name, _label in present], numbers, SOURCE["tag"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
