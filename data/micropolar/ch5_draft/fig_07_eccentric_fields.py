"""
Item 07 - eccentric fields, eps = 0.6, K = 3, l = 10 dx: (a) pressure, (b) speed with a few streamlines,
(c) micro-rotation, (d) w - curl v / 2 (where the micro-rotation departs from half the vorticity: the wall layers).

Run from the repository root:  python data/micropolar/ch5_draft/fig_07_eccentric_fields.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle
from scipy.interpolate import griddata

import sources
import style

SOURCE = sources.ECCENTRIC_FIELD            # {"fields": path without suffix, "status", "tag", "note"}
ZOOM_HALF_WIDTH = 0.55                      # of the inset of panel (d), in gaps, about the narrowest gap


def main() -> int:
    style.setup()
    data = np.load(style.REPO_ROOT / (SOURCE["fields"] + ".npz"))
    summary = json.loads((style.REPO_ROOT / (SOURCE["fields"] + ".json")).read_text(encoding="utf-8"))
    meta = summary["couette"]
    gap, inner, outer, offset = meta["gap"], meta["inner_radius"], meta["outer_radius"], meta["eccentricity"]
    speed_scale, rate = meta["wall_speed"], meta["rotation_rate"]
    x, y = data["x"] / gap, data["y"] / gap
    speed = np.hypot(data["u"], data["v"]) / speed_scale
    pressure = (data["pressure"] - data["pressure"].mean()) / (1000.0 * speed_scale ** 2)
    spin = data["spin"] / rate
    excess = (data["spin"] - 0.5 * data["curl"]) / rate
    panels = [("a", pressure, style.FIELD_SIGNED, True, style.math(r"(p - \bar p)\,/\,(\rho U^2)")),
              ("b", speed, style.FIELD_MAGNITUDE, False, style.math(r"|\mathbf{v}|\,/\,U")),
              ("c", spin, style.FIELD_SIGNED, True, style.math(style.SPIN)),
              ("d", excess, style.FIELD_SIGNED, True, style.math(style.EXCESS_SPIN))]
    figure, axes = plt.subplots(2, 2, figsize=(style.WIDE, 0.96 * style.WIDE))
    extent = outer / gap + 0.15
    points_per_gap = (0.5 * style.WIDE * 72.0 * 0.80) / (2.0 * extent)       # axes are about 80 % of half the width
    marker_area = (1.35 * meta["spacing"] / gap * points_per_gap) ** 2
    numbers = {}
    for axis, (letter, values, colour_map, signed, label) in zip(axes.ravel(), panels):
        if signed:
            bound = float(np.percentile(np.abs(values), 99.8))
            keywords = {"norm": TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)}
        else:
            keywords = {"vmin": 0.0, "vmax": float(values.max())}
        image = axis.scatter(x, y, c=values, s=marker_area, marker="s", linewidths=0.0, cmap=colour_map,
                             rasterized=True, **keywords)
        axis.add_patch(Circle((offset / gap, 0.0), inner / gap, facecolor="white", edgecolor=style.INK,
                              linewidth=0.5, zorder=3))
        axis.add_patch(Circle((0.0, 0.0), outer / gap, facecolor="none", edgecolor=style.INK, linewidth=0.5,
                              zorder=3))
        axis.set_aspect("equal")
        axis.set_xlim(-extent, extent)
        axis.set_ylim(-extent, extent)
        axis.set_xticks([-4, -2, 0, 2, 4])
        axis.set_yticks([-4, -2, 0, 2, 4])
        axis.set_xlabel(style.math(rf"x\,/\,{style.GAP}"))
        axis.set_ylabel(style.math(rf"y\,/\,{style.GAP}"))
        bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
        bar.set_label(label)
        bar.ax.tick_params(labelsize=style.FONT_SIZE - 1.5)
        bar.outline.set_linewidth(0.4)
        style.panel_letter(axis, letter)
        numbers[letter] = {"min": float(values.min()), "max": float(values.max())}
        if letter == "b":
            # streamlines of the velocity interpolated linearly to a grid; seeds across the widest gap
            grid = np.linspace(-extent, extent, 420)
            grid_x, grid_y = np.meshgrid(grid, grid)
            u = griddata((x, y), data["u"], (grid_x, grid_y), method="linear")
            v = griddata((x, y), data["v"], (grid_x, grid_y), method="linear")
            outside = (np.hypot(grid_x, grid_y) > outer / gap) | (np.hypot(grid_x - offset / gap, grid_y)
                                                                  < inner / gap)
            u[outside], v[outside] = np.nan, np.nan
            widest = np.linspace(-(outer / gap) + 0.12, -(inner - offset) / gap - 0.12, 6)
            axis.streamplot(grid, grid, u, v, start_points=np.column_stack([widest, np.zeros_like(widest)]),
                            color="white", linewidth=0.5, arrowsize=0.5, density=6, broken_streamlines=False,
                            zorder=2)
        if letter == "d":
            # inset inside the rotor: the narrowest gap, where the two wall layers are closest
            centre = 0.5 * ((offset + inner) + outer) / gap
            near = (np.abs(x - centre) < ZOOM_HALF_WIDTH) & (np.abs(y) < 1.6 * ZOOM_HALF_WIDTH)
            inset = axis.inset_axes([0.335, 0.26, 0.30, 0.48], zorder=5)
            inset.scatter(x[near], y[near], c=values[near], s=(1.35 * meta["spacing"] / gap * 0.30 * points_per_gap
                                                               * 2.0 * extent / (2.0 * ZOOM_HALF_WIDTH)) ** 2,
                          marker="s", linewidths=0.0, cmap=colour_map, rasterized=True, **keywords)
            inset.add_patch(Circle((offset / gap, 0.0), inner / gap, facecolor="white", edgecolor=style.INK,
                                   linewidth=0.5, zorder=3))
            inset.add_patch(Circle((0.0, 0.0), outer / gap, facecolor="none", edgecolor=style.INK, linewidth=0.5,
                                   zorder=3))
            inset.set_xlim(centre - ZOOM_HALF_WIDTH, centre + ZOOM_HALF_WIDTH)
            inset.set_ylim(-1.6 * ZOOM_HALF_WIDTH, 1.6 * ZOOM_HALF_WIDTH)
            inset.set_aspect("equal")
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_linewidth(0.5)
            axis.indicate_inset_zoom(inset, edgecolor=style.INK, linewidth=0.4, alpha=1.0)
    figure.tight_layout(pad=0.4, w_pad=0.8, h_pad=0.8)
    style.save(figure, "fig_07_eccentric_fields")

    length = meta["micropolar_length"] / meta["spacing"]
    core = np.abs(excess[(np.hypot(data["x"] - offset, data["y"]) > inner + 5.0 * meta["micropolar_length"])
                         & (np.hypot(data["x"], data["y"]) < outer - 5.0 * meta["micropolar_length"])])
    caption = (
        f"Micropolar fluid between eccentric cylinders, eps = {offset / gap:g} (rotor axis at x = {offset / gap:g} g, "
        f"rotor turning counter-clockwise at Omega, stator at rest), K = {meta['coupling_ratio']:g}, l = {length:g} dx "
        f"(m = {gap / (2.0 * meta['micropolar_length']):g}), g = {gap / meta['spacing']:g} dx, w = 0 on both walls, "
        f"Ma = {meta['mach']:g}, Re = {meta['reynolds']:g} on mu_inf "
        f"({meta['reynolds'] * (1.0 + 0.5 * meta['coupling_ratio']):g} on mu); every fluid particle at the last step "
        f"of run `{summary['run']}` (step {summary['step']:,}; {summary['revision']['describe']}; {SOURCE['note']}). "
        "(a) Pressure, mean removed. (b) Speed, with six streamlines started across the widest gap. "
        "(c) Micro-rotation. (d) Micro-rotation minus half the curl of the velocity (curl by weighted linear least "
        "squares over the fluid neighbours within 3 dx): it vanishes in the core and is carried by the two wall "
        "layers of thickness l, where the walls hold w = 0; inset: the narrowest gap (4 l wide). "
        f"Ranges: pressure {numbers['a']['min']:.2f} to {numbers['a']['max']:.2f} rho U^2; micro-rotation "
        f"{numbers['c']['min']:.2f} to {numbers['c']['max']:.2f} Omega; w - curl v / 2 "
        f"{numbers['d']['min']:.2f} to {numbers['d']['max']:.2f} Omega, with a median magnitude of "
        f"{np.median(core):.1e} Omega further than 5 l from both walls. Colour scales of the signed fields are "
        "symmetric about zero and end at the 99.8th percentile of the magnitude.")
    style.record("07", "Eccentric fields: pressure, speed, micro-rotation, w - curl v / 2", SOURCE["status"], caption,
                 ["fig_07_eccentric_fields.pdf", "fig_07_eccentric_fields.png"],
                 [SOURCE["fields"] + ".npz", summary["run"]], numbers, SOURCE["tag"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
