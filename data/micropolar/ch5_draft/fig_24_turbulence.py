"""
Item 24 - decaying 2-D turbulence in the doubly periodic box [0, 2 pi)^2: (a, b) the vorticity of the Newtonian and
of the micropolar run at one time; (c) the energy E(t) and the enstrophy Z(t) of the runs; (d) the energy spectra;
(e) the spectrum ratio micropolar / Newtonian against k with the shifted-lattice noise floor and the linear estimates.

Run from the repository root:  python data/micropolar/ch5_draft/fig_24_turbulence.py
"""
import matplotlib.pyplot as plt
import numpy as np

import sources
import style

SOURCE = sources.PERIODIC
NEWTONIAN = "turbulence_newtonian_n2050"
MICROPOLAR = "turbulence_micropolar_n2050"
SHIFTED = "turbulence_newtonian_n2050_shifted"
COARSE = "turbulence_newtonian_n1025"
FIELD_TIME = 2.0
SPECTRUM_TIMES = ("1", "2", "4")
LENGTH_TIMES_WAVENUMBER_ONE = 7.0          # |k| l = 1 at k = 7 (l = 1 / 7)


def field(run: str, time: float):
    path = style.REPO_ROOT / SOURCE["fields"] / f"{run}_t{time:.3f}.npz"
    if not path.exists():
        candidates = sorted((style.REPO_ROOT / SOURCE["fields"]).glob(f"{run}_t*.npz"))
        if not candidates:
            return None
        path = min(candidates, key=lambda item: abs(float(item.stem.split("_t")[-1]) - time))
    with np.load(path) as archive:
        return np.array(archive["vorticity"], dtype=np.float32), float(archive["time"])


def main() -> int:
    style.setup()
    if not style.exists(SOURCE["turbulence"]):
        print("[pack] item 24: no turbulence analysis yet")
        return 0
    analysis = style.load_json(SOURCE["turbulence"])
    series = analysis.get("series", {})
    if NEWTONIAN not in series:
        print("[pack] item 24: no Newtonian run yet")
        return 0
    figure = plt.figure(figsize=(style.WIDE, 0.92 * style.WIDE))
    grid = figure.add_gridspec(2, 6, height_ratios=(1.0, 0.9), hspace=0.42, wspace=1.6)
    # (a, b) vorticity at FIELD_TIME
    maps = [(NEWTONIAN, "Newtonian, $\\mu + \\kappa/2$"), (MICROPOLAR, "micropolar, $K = 3$")]
    loaded = [(run, title, field(run, FIELD_TIME)) for run, title in maps]
    limit = max([float(np.percentile(np.abs(item[0]), 99.5)) for _run, _title, item in loaded if item is not None] + [1.0])
    image = None
    for column, (letter, (run, title, item)) in enumerate(zip("ab", loaded)):
        axis = figure.add_subplot(grid[0, 2 * column:2 * column + 2])
        if item is not None:
            vorticity, actual = item
            image = axis.imshow(vorticity.T, origin="lower", extent=(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi),
                                cmap=style.FIELD_SIGNED, vmin=-limit, vmax=limit, interpolation="nearest")
            axis.set_title(f"{title}, $t = {actual:.0f}$", fontsize=style.FONT_SIZE)
        axis.set_xticks([0.0, np.pi, 2.0 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
        axis.set_yticks([0.0, np.pi, 2.0 * np.pi], ["0", r"$\pi$", r"$2\pi$"] if column == 0 else [])
        axis.set_xlabel("$x$")
        if column == 0:
            axis.set_ylabel("$y$")
        style.panel_letter(axis, letter, y=1.12)
    # (c) E(t) and Z(t), normalised by their initial values
    axis = figure.add_subplot(grid[0, 4:6])
    for index, (run, label) in enumerate(((NEWTONIAN, "Newtonian"), (MICROPOLAR, "micropolar"),
                                          (SHIFTED, "Newtonian, shifted lattice"), (COARSE, r"Newtonian, $1025^2$"))):
        if run not in series:
            continue
        time = np.asarray(series[run]["time"])
        energy = np.asarray(series[run]["energy"])
        enstrophy = np.asarray(series[run]["enstrophy"])
        colour = style.SERIES[index][0]
        dashes = "-" if index < 2 else (0, (3, 1.5))
        axis.plot(time, energy / energy[0], color=colour, linestyle=dashes, linewidth=1.0, label=f"$E$, {label}")
        axis.plot(time, enstrophy / enstrophy[0], color=colour, linestyle=dashes, linewidth=1.0, alpha=0.55,
                  label=f"$Z$, {label}" if index < 2 else None)
    axis.set_yscale("log")
    axis.set_xlabel("$t$")
    axis.set_ylabel(r"$E/E_0$, $Z/Z_0$")
    axis.legend(loc="lower left", fontsize=style.FONT_SIZE - 2.0, handlelength=1.6)
    style.panel_letter(axis, "c", y=1.12)
    # (d) energy spectra
    shells = np.asarray(analysis["shells"], dtype=np.float64)
    spectra = analysis.get("spectra", {})
    axis = figure.add_subplot(grid[1, 0:3])
    chosen_times = [time for time in SPECTRUM_TIMES if time in spectra]
    shades = np.linspace(1.0, 0.45, max(len(chosen_times), 1))
    for shade, time in zip(shades, chosen_times):
        entry = spectra[time]
        newtonian = np.asarray(entry["newtonian"])
        keep = shells >= 1
        axis.plot(shells[keep], newtonian[keep], color=style.SERIES[0][0], linewidth=1.0, alpha=shade,
                  label=f"Newtonian, $t = {float(entry['time']):.0f}$")
        if MICROPOLAR in entry:
            axis.plot(shells[keep], np.asarray(entry[MICROPOLAR])[keep], color=style.SERIES[1][0], linewidth=1.0,
                      alpha=shade, label=f"micropolar, $t = {float(entry['time']):.0f}$")
        if COARSE in entry and time == chosen_times[-1]:
            axis.plot(shells[keep], np.asarray(entry[COARSE])[keep], color=style.MUTED, linewidth=0.8,
                      linestyle=(0, (3, 1.5)), label=r"Newtonian, $1025^2$")
    reference = np.array([8.0, 80.0])
    axis.plot(reference, 2e-2 * (reference / 8.0) ** -3.0, color=style.INK, linewidth=0.7, linestyle=(0, (1, 1.5)))
    axis.text(30.0, 2e-2 * (30.0 / 8.0) ** -3.0 * 1.8, "$k^{-3}$", fontsize=style.FONT_SIZE - 1.0)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(1.0, shells[-1])
    axis.set_ylim(1e-9, 1.0)
    axis.set_xlabel("$k$")
    axis.set_ylabel("$E(k)$")
    axis.legend(loc="lower left", fontsize=style.FONT_SIZE - 2.0, handlelength=1.6, ncol=1)
    style.panel_letter(axis, "d")
    # (e) spectrum ratio with floor and linear estimates
    axis = figure.add_subplot(grid[1, 3:6])
    numbers = {}
    for index, time in enumerate(chosen_times):
        entry = spectra[time]
        newtonian = np.asarray(entry["newtonian"])
        keep = (shells >= 1) & (newtonian > 0)
        colour = style.SERIES[index][0]
        if MICROPOLAR in entry:
            ratio = np.asarray(entry[MICROPOLAR])[keep] / newtonian[keep]
            axis.plot(shells[keep], ratio, color=colour, linewidth=1.0, label=f"$t = {float(entry['time']):.0f}$")
        axis.plot(shells[keep], np.asarray(entry["linear_inertial"])[keep], color=colour, linewidth=0.8,
                  linestyle=(0, (4, 2)))
        axis.plot(shells[keep], np.asarray(entry["linear_quasi_steady"])[keep], color=colour, linewidth=0.7,
                  linestyle=(0, (1, 1.5)))
        if SHIFTED in entry:
            floor = np.abs(np.asarray(entry[SHIFTED])[keep] / newtonian[keep] - 1.0)
            axis.fill_between(shells[keep], 1.0 - floor, 1.0 + floor, color=colour, alpha=0.12, linewidth=0.0)
    axis.axvline(LENGTH_TIMES_WAVENUMBER_ONE, color=style.MUTED, linewidth=0.6, linestyle=(0, (1, 1.5)))
    axis.text(LENGTH_TIMES_WAVENUMBER_ONE * 1.05, 1.02, r"$k\ell = 1$", fontsize=style.FONT_SIZE - 1.5,
              color=style.MUTED, va="bottom")
    axis.set_xscale("log")
    axis.set_xlim(1.0, shells[-1])
    axis.set_ylim(0.0, 1.2)
    axis.set_xlabel("$k$")
    axis.set_ylabel(r"$E_\mathrm{micropolar}(k)\,/\,E_\mathrm{Newtonian}(k)$")
    axis.plot([], [], color=style.INK, linewidth=0.8, linestyle=(0, (4, 2)), label=r"linear, $j = \ell^2$")
    axis.plot([], [], color=style.INK, linewidth=0.7, linestyle=(0, (1, 1.5)), label=r"linear, $j \to 0$")
    axis.legend(loc="lower left", fontsize=style.FONT_SIZE - 2.0, handlelength=1.8)
    style.panel_letter(axis, "e")
    if image is not None:
        figure.colorbar(image, ax=figure.axes[0:2], orientation="vertical", fraction=0.03, pad=0.02,
                        label=r"$\omega$")
    style.save(figure, "fig_24_turbulence")
    runs = analysis.get("runs", {})
    for name, record in runs.items():
        numbers[name] = {key: record[key] for key in ("particles", "steps", "timestep", "elapsed_seconds")}
    for time, rows in analysis.get("bands", {}).items():
        numbers[f"bands_t{time}"] = rows
    for run in (NEWTONIAN, MICROPOLAR):
        if run in series:
            time = np.asarray(series[run]["time"])
            numbers[f"{run}_end"] = {"time": float(time[-1]), "energy": float(series[run]["energy"][-1]),
                                     "enstrophy": float(series[run]["enstrophy"][-1])}
    caption = (
        "Decaying two-dimensional turbulence in the doubly periodic box [0, 2 pi)^2: 2050^2 particles (h = 5 dx), "
        "initial energy spectrum E(k) ~ k^4 exp(-2 (k / k0)^2) with k0 = 10 and random phases (fixed seed), u' = 1, "
        "Ma 0.02 on the initial peak speed; Newtonian viscosity mu + kappa / 2 = 1e-3 (Re = u' / (k0 nu) = 100), "
        "micropolar K = 3, j = l^2, l = 1/7 (|k| l = 1 at k = 7) with the same far-field viscosity. Z is the "
        "enstrophy (1/2) <omega^2> = sum k^2 E(k). (a, b) Vorticity of the Newtonian and the micropolar run at "
        "t = 2 (grid of 1024^2 from a quadratic-spline deposit, shown at 512^2). (c) E(t) / E(0) and Z(t) / Z(0); "
        "the shifted-lattice Newtonian run and the Newtonian run at 1025^2 (dashed) give the noise floor and the "
        "resolution check. (d) Energy spectra at t = 1, 2, 4 (darker = later) with the 1025^2 run at t = 4. "
        "(e) The ratio of the spectra micropolar / Newtonian (solid) against the linear estimates for a single "
        "Fourier mode (dashed: the slow eigenvalue with j = l^2; dotted: quasi-steady mu_eff, j -> 0); the "
        "shaded band is |ratio - 1| of the shifted-lattice Newtonian run, the noise floor.")
    style.record("24", "Decaying 2-D turbulence: vorticity, E(t) and the enstrophy Z(t), spectra and their ratio",
                 SOURCE["status"], caption, ["fig_24_turbulence.pdf", "fig_24_turbulence.png"],
                 [SOURCE["turbulence"], SOURCE["fields"]], numbers, SOURCE["tag"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
