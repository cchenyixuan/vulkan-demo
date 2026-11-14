"""
Item 24 - decaying 2-D turbulence in the doubly periodic box [0, 2 pi)^2: (a, b) the vorticity of the Newtonian and
of the micropolar run at one time, (c) their difference; (d) the energy E(t) and the enstrophy Z(t) of the micropolar
run and of the floor runs against the Newtonian run; (e) the energy spectra; (f) the spectrum ratio micropolar /
Newtonian against k with the shifted-lattice floor and the linear estimate.

Run from the repository root:  python data/micropolar/ch5_draft/fig_24_turbulence.py
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sources
import style

SOURCE = sources.PERIODIC
NEWTONIAN = "turbulence_newtonian_n2050"
MICROPOLAR = "turbulence_micropolar_n2050"
SHIFTED = "turbulence_newtonian_n2050_shifted"
COARSE = "turbulence_newtonian_n1025"
SMALL_INERTIA = "turbulence_micropolar_j001_n2050"
FIELD_TIME = 2.0
SPECTRUM_TIMES = ("1", "4")
RATIO_TIMES = ("2",)
WAVENUMBER_OF_LENGTH = 7.0          # |k| l = 1 at k = 7 (l = 1 / 7)


def field(run: str, time: float):
    """The vorticity of a run at the snapshot nearest the time (the repeats carry t = 2, 3, 4; the first runs only
    0, 0.5, 1); None if no snapshot lies within 0.05."""
    for name in (f"{run}_repeat", run):
        for path in sorted((style.REPO_ROOT / SOURCE["fields"]).glob(f"{name}_t*.npz")):
            if abs(float(path.stem.split("_t")[-1]) - time) < 0.05:
                with np.load(path) as archive:
                    return np.array(archive["vorticity"], dtype=np.float32), float(archive["time"]), name
    return None


def map_panel(axis, values, limit, colour_map, title, left, with_bar=True):
    image = axis.imshow(values.T, origin="lower", extent=(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi), cmap=colour_map,
                        vmin=-limit, vmax=limit, interpolation="nearest")
    axis.set_title(title, fontsize=style.FONT_SIZE - 1.0, loc="left")
    axis.set_xticks([0.0, np.pi, 2.0 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
    axis.set_yticks([0.0, np.pi, 2.0 * np.pi], ["0", r"$\pi$", r"$2\pi$"] if left else [])
    axis.set_xlabel("$x$")
    if left:
        axis.set_ylabel("$y$")
    bar_axis = make_axes_locatable(axis).append_axes("right", size="5%", pad=0.04)
    if with_bar:
        bar = axis.figure.colorbar(image, cax=bar_axis)
        bar.ax.tick_params(labelsize=style.FONT_SIZE - 2.0)
    else:
        bar_axis.set_visible(False)
    return image


def smoothed(values, cutoff=100.0):
    """The field with a Gaussian low-pass exp(-(k / cutoff)^2) (the grid-scale noise of the particle deposit
    otherwise covers the difference of two runs)."""
    modes = np.fft.fftfreq(values.shape[0], d=1.0 / values.shape[0])
    kx, ky = np.meshgrid(modes, modes, indexing="ij")
    return np.real(np.fft.ifft2(np.fft.fft2(values) * np.exp(-(kx ** 2 + ky ** 2) / cutoff ** 2)))


def binned(shells, spectrum):
    """Sums of the spectrum over single shells up to k = 10 and bins of about 8 % in k above."""
    edges = np.unique(np.concatenate([np.arange(1, 11), np.rint(np.geomspace(10, shells[-1] + 1, 26))]).astype(int))
    centres, sums = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        chosen = (shells >= low) & (shells < high)
        centres.append(float(np.sqrt(low * max(high - 1, low))))
        sums.append(float(np.sum(spectrum[chosen])))
    return np.asarray(centres), np.asarray(sums)


def main() -> int:
    style.setup()
    if not style.exists(SOURCE["turbulence"]):
        print("[pack] item 24: no turbulence analysis yet")
        return 0
    analysis = style.load_json(SOURCE["turbulence"])
    series = analysis.get("series", {})
    if NEWTONIAN not in series or MICROPOLAR not in series:
        print("[pack] item 24: the 2050^2 runs are not analysed yet")
        return 0
    figure = plt.figure(figsize=(style.WIDE, 0.78 * style.WIDE))
    grid = figure.add_gridspec(2, 3, height_ratios=(1.0, 1.0), hspace=0.45, wspace=0.50)
    numbers = {}
    # (a, b, c) vorticity at FIELD_TIME and the difference
    newtonian_field, micropolar_field = field(NEWTONIAN, FIELD_TIME), field(MICROPOLAR, FIELD_TIME)
    if newtonian_field is not None and micropolar_field is not None:
        limit = float(np.percentile(np.abs(newtonian_field[0]), 99.5))
        difference = smoothed(micropolar_field[0] - newtonian_field[0])
        difference_limit = float(np.percentile(np.abs(difference), 99.5))
        map_panel(figure.add_subplot(grid[0, 0]), newtonian_field[0], limit, style.FIELD_SIGNED,
                  rf"(a) $\omega$, Newtonian, $t = {newtonian_field[1]:.0f}$", True, with_bar=False)
        map_panel(figure.add_subplot(grid[0, 1]), micropolar_field[0], limit, style.FIELD_SIGNED,
                  rf"(b) $\omega$, micropolar, $t = {micropolar_field[1]:.0f}$", False)
        map_panel(figure.add_subplot(grid[0, 2]), difference, difference_limit, "PuOr_r",
                  r"(c) $\omega_\mathrm{micropolar} - \omega_\mathrm{Newtonian}$", False)
        numbers["fields"] = {"time": newtonian_field[1], "runs": [newtonian_field[2], micropolar_field[2]],
                             "vorticity_99_5_percentile": limit, "difference_99_5_percentile": difference_limit,
                             "difference_rms_over_vorticity_rms": float(np.sqrt(np.mean(difference ** 2))
                                                                        / np.sqrt(np.mean(newtonian_field[0] ** 2)))}
    # (d) E and Z against the Newtonian run: the two micropolar runs and the floors
    axis = figure.add_subplot(grid[1, 0])
    for run, reference, colour, dashes, label in (
            (MICROPOLAR, NEWTONIAN, style.SERIES[1][0], "-", r"$j = \ell^2$"),
            (SMALL_INERTIA, NEWTONIAN, style.SERIES[2][0], (0, (6, 1.5, 1, 1.5)), r"$j = \ell^2/100$"),
            (SHIFTED, NEWTONIAN, style.MUTED, (0, (3, 1.5)), "shifted lattice"),
            (COARSE, NEWTONIAN, style.SERIES[3][0], (0, (1, 1.2)), r"Newtonian $1025^2$")):
        if run not in series or reference not in series:
            continue
        reference_time = np.asarray(series[reference]["time"])
        inside = np.asarray(series[run]["time"]) <= reference_time[-1] + 1e-9
        time = np.asarray(series[run]["time"])[inside]
        for key, width in (("energy", 1.3), ("enstrophy", 0.7)):
            base = np.interp(time, reference_time, np.asarray(series[reference][key]))
            axis.plot(time, 100.0 * (np.asarray(series[run][key])[inside] / base - 1.0), color=colour, linestyle=dashes,
                      linewidth=width, label=label if key == "energy" else None)
    axis.axhline(0.0, color=style.INK, linewidth=0.5)
    axis.set_xlabel("$t$")
    axis.set_ylabel(r"change of $E$, $Z$ (%)")
    axis.set_ylim(-30.0, 2.0)
    axis.text(4.0, -9.3, "$E$", fontsize=style.FONT_SIZE - 2.0, ha="right", va="bottom")
    axis.text(4.0, -19.8, "$Z$", fontsize=style.FONT_SIZE - 2.0, ha="right", va="top")
    axis.legend(loc="lower left", fontsize=style.FONT_SIZE - 2.5, handlelength=1.8, frameon=False)
    style.panel_letter(axis, "d")
    # (e) energy spectra at t = 1 and 4
    shells = np.asarray(analysis["shells"], dtype=np.float64)
    spectra = analysis.get("spectra", {})
    chosen_times = [time for time in SPECTRUM_TIMES if time in spectra]
    axis = figure.add_subplot(grid[1, 1])
    shades = np.linspace(1.0, 0.45, max(len(chosen_times), 1))
    keep = shells >= 1
    for position, (shade, time) in enumerate(zip(shades, chosen_times)):
        entry = spectra[time]
        for run, colour, dashes, label in (("newtonian", style.SERIES[0][0], "-", "Newtonian"),
                                           (MICROPOLAR, style.SERIES[1][0], (0, (4, 1.5)), r"$j = \ell^2$"),
                                           (SMALL_INERTIA, style.SERIES[2][0], (0, (6, 1.5, 1, 1.5)),
                                            r"$j = \ell^2/100$")):
            if run in entry:
                axis.plot(shells[keep], np.asarray(entry[run])[keep], color=colour, linewidth=1.0, alpha=shade,
                          linestyle=dashes, label=label if position == 0 else None)
        if COARSE in entry and time == chosen_times[-1]:
            axis.plot(shells[keep], np.asarray(entry[COARSE])[keep], color=style.SERIES[3][0], linewidth=0.8,
                      linestyle=(0, (1, 1.2)), label=r"$1025^2$")
    reference = np.array([12.0, 70.0])
    axis.plot(reference, 4e-2 * (reference / 12.0) ** -3.0, color=style.INK, linewidth=0.7)
    axis.text(30.0, 4e-2 * (30.0 / 12.0) ** -3.0 * 2.5, "$k^{-3}$", fontsize=style.FONT_SIZE - 1.5)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(1.0, shells[-1])
    axis.set_ylim(1e-10, 1.0)
    axis.set_xlabel("$k$")
    axis.set_ylabel("$E(k)$")
    axis.legend(loc="lower left", fontsize=style.FONT_SIZE - 2.5, handlelength=1.8, frameon=False)
    style.panel_letter(axis, "e")
    # (f) spectrum ratio (bins of about 8 % in k) with the shifted-lattice floor and the linear estimates
    axis = figure.add_subplot(grid[1, 2])
    for time in [time for time in RATIO_TIMES if time in spectra]:
        entry = spectra[time]
        colour = style.SERIES[0][0]
        centres, newtonian = binned(shells, np.asarray(entry["newtonian"]))
        if SHIFTED in entry:
            floor = np.abs(binned(shells, np.asarray(entry[SHIFTED]))[1] / newtonian - 1.0)
            axis.fill_between(centres, 1.0 - floor, 1.0 + floor, color=colour, alpha=0.15, linewidth=0.0)
        if MICROPOLAR in entry:
            axis.plot(centres, binned(shells, np.asarray(entry[MICROPOLAR]))[1] / newtonian, color=colour,
                      linewidth=1.1, label=r"$j = \ell^2$")
        if SMALL_INERTIA in entry:
            axis.plot(centres, binned(shells, np.asarray(entry[SMALL_INERTIA]))[1] / newtonian,
                      color=style.SERIES[2][0], linewidth=1.1, linestyle=(0, (6, 1.5, 1, 1.5)),
                      label=r"$j = \ell^2/100$")
        axis.plot(shells[keep], np.asarray(entry["linear_inertial"])[keep], color=colour, linewidth=0.7,
                  linestyle=(0, (4, 2)), label=r"one mode, $j = \ell^2$")
        axis.plot(shells[keep], np.asarray(entry["linear_quasi_steady"])[keep], color=style.SERIES[2][0],
                  linewidth=0.7, linestyle=(0, (1, 1.2)), label=r"one mode, $j \to 0$")
        axis.set_title(rf"$t = {float(entry['time']):.0f}$", fontsize=style.FONT_SIZE - 1.0, loc="right")
    axis.axvline(WAVENUMBER_OF_LENGTH, color=style.MUTED, linewidth=0.6, linestyle=(0, (1, 1.5)))
    axis.text(WAVENUMBER_OF_LENGTH * 1.06, 1.12, r"$|\mathbf{k}|\ell = 1$", fontsize=style.FONT_SIZE - 2.0,
              color=style.MUTED, va="bottom")
    axis.set_xscale("log")
    axis.set_xlim(1.0, shells[-1])
    axis.set_ylim(0.0, 1.25)
    axis.set_xlabel("$k$")
    axis.set_ylabel(r"$E_\mathrm{micropolar}(k)\,/\,E_\mathrm{Newtonian}(k)$")
    axis.legend(loc="lower left", fontsize=style.FONT_SIZE - 2.5, handlelength=1.8, frameon=False)
    style.panel_letter(axis, "f")
    style.save(figure, "fig_24_turbulence")
    for name, record in analysis.get("runs", {}).items():
        numbers[name] = {key: record[key] for key in ("particles", "steps", "timestep", "elapsed_seconds")}
    for time, rows in analysis.get("bands", {}).items():
        numbers[f"bands_t{time}"] = rows
    for run in (NEWTONIAN, MICROPOLAR):
        numbers[f"{run}_end"] = {"time": float(series[run]["time"][-1]), "energy": float(series[run]["energy"][-1]),
                                 "enstrophy": float(series[run]["enstrophy"][-1])}
    caption = (
        "Decaying two-dimensional turbulence in the doubly periodic box [0, 2 pi)^2: 2050^2 particles (h = 5 dx), "
        "initial energy spectrum E(k) ~ k^4 exp(-2 (k / k0)^2) with k0 = 10 and random phases (fixed seed), u' = 1, "
        "Ma 0.02 on the initial peak speed; the Newtonian fluid of viscosity mu + kappa / 2 = 1e-3 (Re = u' / (k0 "
        "nu) = 100) and the micropolar fluid K = 3, j = l^2, l = 1/7 (|k| l = 1 at k = 7) with the same far-field "
        "viscosity, both from the same field (w = omega / 2). Z is the enstrophy (1/2) <omega^2> = sum k^2 E(k). "
        "(a, b) Vorticity of the Newtonian and the micropolar run at t = 2 and (c) their difference (the repeats of "
        "the two runs on one card, which carry the snapshots at t >= 2; deposit on a 1024^2 grid, shown at 512^2; the "
        "difference low-passed with exp(-(k / 100)^2) against the grid-scale noise of the deposit). "
        "(d) E and Z of the micropolar runs (j = l^2, and j = l^2 / 100 with the same dt) against the Newtonian "
        "run (thick: E, thin: Z), with the shifted-lattice floor and the Newtonian run at 1025^2 (resolution); the "
        "repeats on one card and the Newtonian run on the other card lie on the shifted-lattice floor (Section 18.5). "
        "(e) Energy spectra at t = 1 and 4 (paler = later) with the 1025^2 run at t = 4. (f) The ratio of the spectra "
        "micropolar / Newtonian at t = 2 in bins of about 8 % in k (solid: j = l^2, dash-dotted: j = l^2 / 100) with the "
        "floor |ratio - 1| of the shifted-lattice run (shaded), and the linear estimates for a single Fourier mode "
        "decaying alone at the slow rate of the two-mode system with j = l^2 (dashed) and at the quasi-steady mu_eff "
        "(j -> 0, dotted).")
    style.record("24", "Decaying 2-D turbulence: vorticity, E(t) and the enstrophy Z(t), spectra and their ratio",
                 SOURCE["status"], caption, ["fig_24_turbulence.pdf", "fig_24_turbulence.png"],
                 [SOURCE["turbulence"], SOURCE["fields"]], numbers, SOURCE["tag"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
