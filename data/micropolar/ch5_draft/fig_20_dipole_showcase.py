"""
Item 20 - the showcase of the dipole-wall collision: vorticity on one colour scale in the window 0.4 <= x / W <= 1,
0 <= y / W <= 0.6 (the upper half; the flow is symmetric), rows = the Newtonian control, the micropolar fluid with
l = W / 100 and with l = W / 50, columns = four times around the first and the second collision (chosen from the
control); a fourth row: w - omega / 2 of l = W / 50 at the same times (the layer on the wall and where it is shed).

Run from the repository root:  python data/micropolar/ch5_draft/fig_20_dipole_showcase.py
(the cropped fields come from: python -m experiment.validation.micropolar_dipole_analysis fields --runs ... --times ...)
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

import sources
import style

SOURCE = sources.DIPOLE
ROWS = (("control_n1000", "control"), ("mp_l100_n1000", r"$\ell = W/100$"), ("mp_l50_n1000", r"$\ell = W/50$"))
SPIN_ROW = "mp_l50_n1000"
TIMES = (0.350, 0.450, 0.600, 0.700)
VORTICITY_LIMIT = 300.0            # one colour scale for the vorticity rows (saturates inside the wall layer)
SPIN_LIMIT = 60.0
LEVELS = (-50.0, -30.0, -10.0, 10.0, 30.0, 50.0)   # the contour levels of the paper's figures


def field(run: str, time: float):
    path = style.REPO_ROOT / SOURCE["fields"] / f"{run}_t{time:.3f}.npz"
    if not path.exists():
        return None
    with np.load(path) as archive:
        return {name: np.array(archive[name]) for name in archive.files}


def main() -> int:
    style.setup()
    rows = [(run, label) for run, label in ROWS if field(run, TIMES[0]) is not None]
    if not rows:
        print("[pack] item 20: no cropped fields yet")
        return 0
    spin = field(SPIN_ROW, TIMES[0]) is not None and "spin_departure" in field(SPIN_ROW, TIMES[0])
    row_count = len(rows) + (1 if spin else 0)
    panel = 0.205 * style.WIDE
    figure, axes = plt.subplots(row_count, len(TIMES), figsize=(style.WIDE, panel * row_count + 0.35),
                                squeeze=False, sharex=True, sharey=True)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-VORTICITY_LIMIT, vmax=VORTICITY_LIMIT)
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    numbers = {}
    image = spin_image = None
    for row_index, (run, label) in enumerate(rows):
        for column, time in enumerate(TIMES):
            axis = axes[row_index, column]
            data = field(run, time)
            vorticity = data["vorticity"].astype(np.float32)
            extent = (data["x"][0], data["x"][-1], data["y"][0], data["y"][-1])
            image = axis.imshow(vorticity.T, origin="lower", extent=extent, cmap=style.FIELD_SIGNED, norm=norm,
                                interpolation="bilinear", rasterized=True)
            axis.contour(data["x"], data["y"], vorticity.T, levels=LEVELS, colors=style.INK, linewidths=0.25)
            numbers.setdefault(run, {})[f"{time:.3f}"] = {"min": float(vorticity.min()), "max": float(vorticity.max())}
            style.panel_letter(axis, next(letters), x=0.03, y=0.97, va="top")
    if spin:
        for column, time in enumerate(TIMES):
            axis = axes[len(rows), column]
            data = field(SPIN_ROW, time)
            departure = data["spin_departure"].astype(np.float32)
            extent = (data["x"][0], data["x"][-1], data["y"][0], data["y"][-1])
            spin_image = axis.imshow(departure.T, origin="lower", extent=extent, cmap=style.FIELD_SIGNED,
                                     vmin=-SPIN_LIMIT, vmax=SPIN_LIMIT, interpolation="bilinear", rasterized=True)
            numbers.setdefault("spin_departure", {})[f"{time:.3f}"] = {"min": float(departure.min()),
                                                                       "max": float(departure.max())}
            style.panel_letter(axis, next(letters), x=0.03, y=0.97, va="top")
    for axis in axes.ravel():
        axis.axvline(1.0, color=style.INK, linewidth=1.0)
        axis.set_xlim(0.4, 1.0)
        axis.set_ylim(0.0, 0.6)
        axis.set_aspect("equal")
        axis.set_xticks([0.4, 0.6, 0.8, 1.0])
        axis.set_yticks([0.0, 0.2, 0.4, 0.6])
    for axis in axes[-1, :]:
        axis.set_xlabel(r"$x\,/\,W$")
    for row_index, axis in enumerate(axes[:, 0]):
        axis.set_ylabel(r"$y\,/\,W$")
    figure.subplots_adjust(left=0.08, right=0.86, bottom=0.07, top=0.98, wspace=0.06, hspace=0.08)
    top = axes[0, -1].get_position()
    bottom = axes[len(rows) - 1, -1].get_position()
    bar_axis = figure.add_axes([0.88, bottom.y0, 0.015, top.y1 - bottom.y0])
    bar = figure.colorbar(image, cax=bar_axis, extend="both")
    bar.set_label(r"$\omega\,W\,/\,U$")
    if spin_image is not None:
        last = axes[len(rows), -1].get_position()
        spin_axis = figure.add_axes([0.88, last.y0, 0.015, last.y1 - last.y0])
        spin_bar = figure.colorbar(spin_image, cax=spin_axis, extend="both")
        spin_bar.set_label(r"$(w - \omega/2)\,W\,/\,U$")
    style.save(figure, "fig_20_dipole_showcase")
    analysis = style.load_json(SOURCE["analysis"])
    control = analysis["runs"].get("control_n1000", {})
    caption = (
        "Normal collision of a dipole with the no-slip wall x = W (thick line; Clercx & Bruneau 2006, box [-W, W]^2, "
        "Re = U W / nu_inf = 625; the upper half 0.4 <= x / W <= 1, 0 <= y / W <= 0.6 is shown, the flow being "
        "symmetric). Vorticity omega W / U on one colour scale (saturated at +-"
        f"{VORTICITY_LIMIT:g}; contours at +-10, +-30, +-50 as in the paper's figures) at t U / W = "
        + ", ".join(f"{time:g}" for time in TIMES)
        + " (columns): before and after the first and the second collision"
        + (f" (enstrophy peaks of the control at t = {control['enstrophy_peaks'][0][0]:.3f} and "
           f"{control['enstrophy_peaks'][1][0]:.3f})" if control else "")
        + ". Rows: " + "; ".join(f"({chr(97 + index * len(TIMES))}-{chr(96 + (index + 1) * len(TIMES))}) {name}"
                                 for index, name in enumerate(
                                     ["the Newtonian control of viscosity mu_inf", "the micropolar fluid, K = 3, "
                                      "l = W / 100 (10 dx)", "the micropolar fluid, l = W / 50 (20 dx)"][:len(rows)]))
        + ("; (" + chr(97 + len(rows) * len(TIMES)) + "-" + chr(96 + (len(rows) + 1) * len(TIMES))
           + f") w - omega / 2 of l = W / 50 (saturated at +-{SPIN_LIMIT:g}): the micro-rotation layer on the wall "
             "and where it is shed" if spin else "")
        + ". w = 0 on every wall, w = omega / 2 at t = 0. dx = W / 1000 (4 million fluid particles), h = 5 dx; "
          "the fields are the particle velocities on a grid of one spacing (quadratic B-spline, no-slip "
          "reflection at the wall), the vorticity by central differences.")
    style.record("20", "Dipole-wall collision: vorticity of the control and of the two micropolar runs at four times",
                 SOURCE["status"], caption, ["fig_20_dipole_showcase.pdf", "fig_20_dipole_showcase.png"],
                 [SOURCE["fields"]] + [run for run, _label in rows], numbers, SOURCE["tag"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
