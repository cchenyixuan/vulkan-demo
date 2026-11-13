"""
Item 15 - the 3-D case (rotor of radius 3 g and length 3 g in a housing of radius 4 g with end gaps g, K = 3):
fields in a plane through the axis (v_theta, two components of w, |w - curl v / 2|) and the mid-plane profiles of
v_theta and w_z against the 2-D exact solution for the radii.

Run from the repository root:  python data/micropolar/ch5_draft/fig_15_three_dimensional.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

import style

ANALYSIS = "data/micropolar/stage6/rotor3d.json"
MAPS = "data/micropolar/stage6/rotor3d/{run}"
# the micropolar run of the figure: the finest that exists
RUNS = ["g40_k3_m2", "g40_k3_m2_k2", "g20_k3_m1"]
TAG = "thesis-ch5-rc2"


def rule_sentence(analysis, pair) -> str:
    """The rows 'the wall rule' of table 15b: the Newtonian run in the slip-form geometry over the same control."""
    rule = next((entry for entry in analysis.get("rules", {}).values()
                 if pair and entry["control"] == pair["control"]), None)
    if not rule or "housing_wall_change" not in rule:
        return ""
    return (" Its rows 'the wall rule' are the Newtonian fluid of viscosity mu_inf in the slip-form geometry of the "
            "rule (lateral surface of the rotor at R1 + delta turning at Omega (R1 - delta) / (R1 + delta), its end "
            "faces moved out by delta and turning at Omega, every face of the housing moved in by delta, delta = "
            "3.75 dx; at the rim each wall particle of the rotor takes the rate of the surface it is nearer to) "
            f"over the same control: mid-plane {rule['midplane_change']:.4f}, torque "
            f"{rule['rotor_circle_with_advection_change']:.4f} (flux form), {rule['housing_wall_change']:.4f} "
            f"(housing), {rule['rotor_wall_change']:.4f} (wall form).")


def main() -> int:
    style.setup()
    if not style.exists(ANALYSIS):
        print("[pack] item 15: no 3-D analysis yet")
        return 1
    analysis = style.load_json(ANALYSIS)
    # a run that reached its end, or one that was stopped (its record is under "partial_runs": the part that
    # exists, with azimuthal means in place of the particles of a plane)
    records = {**analysis.get("partial_runs", {}), **analysis["runs"]}
    name = next((run for run in RUNS if run in records and style.exists(MAPS.format(run=run) + ".npz")), None)
    if name is None:
        print("[pack] item 15: no micropolar 3-D run with its maps yet")
        return 1
    record = records[name]
    partial = record.get("partial")
    data = np.load(style.REPO_ROOT / (MAPS.format(run=name) + ".npz"))
    summary = json.loads((style.REPO_ROOT / (MAPS.format(run=name) + ".json")).read_text(encoding="utf-8"))
    meta = summary["rotor3d"]
    gap, rate, speed = meta["gap"], meta["rotation_rate"], meta["wall_speed"]
    x, z = data["x"] / gap, data["z"] / gap
    excess = np.sqrt(data["excess_radial"] ** 2 + data["excess_tangential"] ** 2 + data["excess_axial"] ** 2) / rate
    panels = [("a", data["tangential"] / speed, style.FIELD_MAGNITUDE, False, style.math(style.VELOCITY)),
              ("b", data["spin_axial"] / rate, style.FIELD_SIGNED, True, style.math(r"w_z\,/\,\Omega")),
              ("c", data["spin_radial"] / rate, style.FIELD_SIGNED, True, style.math(r"w_r\,/\,\Omega")),
              ("d", excess, style.FIELD_MAGNITUDE, False,
               style.math(r"|\mathbf{w} - \frac{1}{2}\,\nabla\!\times\!\mathbf{v}|\,/\,\Omega"))]
    figure = plt.figure(figsize=(style.WIDE, 0.88 * style.WIDE))
    grid = figure.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.25])
    half_width = meta["housing_radius"] / gap + 0.15
    half_height = meta["housing_half_height"] / gap + 0.15
    points_per_gap = (0.5 * style.WIDE * 72.0 * 0.74) / (2.0 * half_width)
    marker_area = (1.4 * meta["spacing"] / gap * points_per_gap) ** 2
    numbers = {}
    for index, (letter, values, colour_map, signed, label) in enumerate(panels):
        axis = figure.add_subplot(grid[index // 2, index % 2])
        if signed:
            bound = float(np.percentile(np.abs(values), 99.8))
            keywords = {"norm": TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)}
        else:
            keywords = {"vmin": 0.0, "vmax": float(np.percentile(values, 99.9))}
        image = axis.scatter(x, z, c=values, s=marker_area, marker="s", linewidths=0.0, cmap=colour_map,
                             rasterized=True, **keywords)
        rotor_radius, rotor_half = meta["rotor_radius"] / gap, meta["rotor_half_length"] / gap
        housing_radius, housing_half = meta["housing_radius"] / gap, meta["housing_half_height"] / gap
        axis.add_patch(Rectangle((-rotor_radius, -rotor_half), 2.0 * rotor_radius, 2.0 * rotor_half,
                                 facecolor="white", edgecolor=style.INK, linewidth=0.5, zorder=3))
        axis.add_patch(Rectangle((-housing_radius, -housing_half), 2.0 * housing_radius, 2.0 * housing_half,
                                 facecolor="none", edgecolor=style.INK, linewidth=0.5, zorder=3))
        for cut in summary.get("cuts_x") or []:
            axis.axvline(cut / gap, color=style.INK, linewidth=0.6, linestyle=(0, (3, 2)), zorder=4)
        axis.set_aspect("equal")
        axis.set_xlim(-half_width, half_width)
        axis.set_ylim(-half_height, half_height)
        axis.set_xlabel(style.math(rf"x\,/\,{style.GAP}"))
        axis.set_ylabel(style.math(rf"z\,/\,{style.GAP}"))
        bar = figure.colorbar(image, ax=axis, fraction=0.03, pad=0.03)
        bar.set_label(label)
        bar.ax.tick_params(labelsize=style.FONT_SIZE - 1.5)
        bar.outline.set_linewidth(0.4)
        style.panel_letter(axis, letter)
        numbers[letter] = {"min": float(values.min()), "max": float(values.max())}
    middle = record["midplane"]
    exact = record["exact_two_dimensional"]
    radius = (np.asarray(middle["radius"]) - meta["rotor_radius"]) / gap
    for index, (letter, key, scale, label) in enumerate((("e", "velocity", speed, style.VELOCITY),
                                                         ("f", "spin", rate, r"w_z\,/\,\Omega"))):
        axis = figure.add_subplot(grid[2, index])
        axis.plot(radius, np.asarray(exact[key]) / scale, **style.EXACT, label="2-D exact")
        axis.plot(radius, np.asarray(middle[key]) / scale, **style.run_markers(0, filled=False),
                  label="run, mid-plane")
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel(style.math(style.RADIUS_IN_GAP))
        axis.set_ylabel(style.math(label))
        style.panel_letter(axis, letter)
        if index == 0:
            axis.legend(loc="upper right", handletextpad=0.4)
    figure.tight_layout(pad=0.4, w_pad=1.0, h_pad=0.9)
    style.save(figure, "fig_15_three_dimensional")

    pair = analysis.get("pairs", {}).get(name, {})
    reach = record["end_reach"]["reach_in_gaps"]
    layers = record.get("end_face_layers")
    caption = (
        f"The 3-D case at g = {record['gap_in_spacings']:g} dx: a rotor of radius 3 g and length 3 g turning at Omega "
        "about z in a coaxial housing of radius 4 g with end gaps g, micropolar fluid with K = 3, "
        f"l = {record['length_in_spacings']:g} dx, w = 0 on every wall, Ma = {meta['mach']:g}, Re = {meta['reynolds']:g} "
        f"on mu_inf ({meta['reynolds'] * 2.5:g} on mu), h = 4 dx; run `{name}` "
        f"({record['particles']:,} particles, {record['slabs']} slab" + ("s" if record["slabs"] > 1 else "")
        + f", {summary['revision']['describe']}). "
        + ((f"The run was stopped from outside at step {partial['last_heartbeat_step']:,} of "
            f"{partial['planned_steps']:,} (the machine ran out of memory); its samples reach step "
            f"{partial['samples_to_step']:,}, t = {partial['time_reached_over_diffusion_time']:.2f} g^2 / nu_inf, "
            "where the angular momentum of the fluid has changed by 3e-5 over the last 10,000 steps. No particle "
            "state was saved, so (a) to (d) show the azimuthal means on the (r, z) grid of one spacing over the "
            f"samples of steps {partial['window_steps'][0]:,} to {partial['window_steps'][1]:,}, mirrored about "
            "the axis: (a) azimuthal velocity, (b) axial and (c) radial component of the micro-rotation, (d) "
            "magnitude of w - curl v / 2 (curl from finite differences of the means): the wall layers on every "
            "face of the rotor and of the housing. (e), (f): bin means in the mid-plane (|z| < 2 dx, samples of "
            f"steps {partial['midplane_steps'][0]:,} to {partial['midplane_steps'][1]:,}) against the exact "
            "solution between infinite cylinders of the same radii. ") if partial else
           ("(a) to (d): the fluid particles within 1.5 dx of the plane y = 0 at "
            "the last step: (a) azimuthal velocity, (b) axial and (c) radial component of the micro-rotation (radial: "
            "along x, with the sign of x), (d) magnitude of w - curl v / 2 (curl by weighted least squares over the "
            "neighbours within 3 dx): the wall layers on every face of the rotor and of the housing"
            + ("; dashed line: the cut between the two slabs (5090 on the left, XTX on the right)"
               if summary.get("cuts_x") else "")
            + ". (e), (f): bin means in the mid-plane (|z| < 2 dx, last quarter of the run) against the exact "
            "solution between infinite cylinders of the same radii. "))
        +
        f"Mid-plane constant of v_theta over the 2-D one: "
        f"{record['midplane_fit']['potential_vortex_constant'] / exact['potential_vortex_constant']:.4f}; rms "
        f"deviation of the mid-plane profiles from the 2-D ones {record['midplane_velocity_rms_against_two_dimensional']:.1e} "
        f"(v) and {record['midplane_spin_rms_against_two_dimensional']:.1e} (w); the fitted constant of v_theta(r) is "
        f"within 1 % of its mid-plane value further than {reach['0.01']:.2f} g from the rotor's ends and within "
        f"0.1 % further than {reach['0.001']:.2f} g."
        + (f" Change against the Newtonian control `{pair['control']}`: mid-plane {pair['midplane_change']:.4f} "
           f"(2-D exact {pair['two_dimensional_change']:.4f})"
           + (f", torque {pair['rotor_wall_change']:.4f} by the wall form (lateral surface "
              f"{pair['rotor_wall_lateral_change']:.4f}, end faces {pair['rotor_wall_end_change']:.4f}) and "
              f"{pair['rotor_circle_change']:.4f} by the flux across the surface at distance h."
              if "rotor_wall_change" in pair else ".") if pair else "")
        + (f" On the housing the torque changes by {pair['housing_wall_change']:.4f}; with the advective term the "
           f"flux form gives {pair['rotor_circle_with_advection_change']:.4f}. Table 15b sets this run beside the "
           "one at g = 20 dx." if pair and "housing_wall_change" in pair else "")
        + rule_sentence(analysis, pair)
        + (f" Decay length of the wall layer on the rotor's end face: {layers['tangential']['fitted_over_expected']:.2f} l "
           f"for the tangential component and {layers['normal']['fitted_over_expected']:.2f} of l sqrt(nu_inf / nu) "
           "for the normal one." if layers else ""))
    # table 15b: the two resolutions side by side (each micropolar run with its Newtonian control)
    pairs = analysis.get("pairs", {})
    columns = [(run, label) for run, label in (("g20_k3_m1", "g = 20 dx"), ("g40_k3_m2_k2", "g = 40 dx"),
                                               ("g40_k3_m2", "g = 40 dx, one slab"))
               if run in analysis["runs"] and run in pairs and "torque" in analysis["runs"][run]
               and "rotor_wall_change" in pairs[run]]
    files_b = []
    if columns:
        runs_ = analysis["runs"]

        def torque(run, key, sign=1.0):
            return f"{sign * runs_[run]['torque'][key]['mean']:.3f}" if key in runs_[run]["torque"] else "-"

        def change(run, key):
            return f"{pairs[run][key]:.4f}" if key in pairs[run] else "-"
        rules = analysis.get("rules", {})

        def rule(run, key):
            # the Newtonian run in the slip-form geometry of the rule that shares this column's control
            entry = next((entry for entry in rules.values() if entry["control"] == pairs[run]["control"]), None)
            return f"{entry[key]:.4f}" if entry and key in entry else "-"
        specification = [
            ("g / dx; l / dx; m = g / (2 l)", r"$g/\Delta x$; $l/\Delta x$; $m$",
             lambda run: f"{runs_[run]['gap_in_spacings']:g}; {runs_[run]['length_in_spacings']:g}; "
                         f"{runs_[run]['gap_in_spacings'] / (2.0 * runs_[run]['length_in_spacings']):g}"),
            ("particles; slabs", "particles; slabs",
             lambda run: f"{runs_[run]['particles']:,}; {runs_[run]['slabs']}"),
            ("Newtonian control", "Newtonian control", lambda run: pairs[run]["control"]),
            ("torque, Newtonian: flux form (N m)", r"$T_N$, flux form",
             lambda run: torque(pairs[run]["control"], "rotor_circle_with_advection")),
            ("torque, Newtonian: on the housing", r"$T_N$, housing",
             lambda run: torque(pairs[run]["control"], "housing_wall", -1.0)),
            ("torque, micropolar: flux form", r"$T$, flux form",
             lambda run: torque(run, "rotor_circle_with_advection")),
            ("torque, micropolar: on the housing", r"$T$, housing", lambda run: torque(run, "housing_wall", -1.0)),
            ("torque change D_T, flux form", r"$D_T$, flux form",
             lambda run: change(run, "rotor_circle_with_advection_change")),
            ("D_T on the housing", r"$D_T$, housing", lambda run: change(run, "housing_wall_change")),
            ("D_T, wall form", r"$D_T$, wall form", lambda run: change(run, "rotor_wall_change")),
            ("D_T, wall form, lateral surface", r"$D_T$, lateral surface",
             lambda run: change(run, "rotor_wall_lateral_change")),
            ("D_T, wall form, end faces", r"$D_T$, end faces", lambda run: change(run, "rotor_wall_end_change")),
            ("D_T on the housing, lateral face", r"$D_T$, housing, lateral face",
             lambda run: change(run, "housing_wall_lateral_change")),
            ("D_T on the housing, end faces", r"$D_T$, housing, end faces",
             lambda run: change(run, "housing_wall_end_change")),
            ("exact 2-D D for the radii (lateral gap)", r"$D$, exact 2-D", lambda run: change(run, "two_dimensional_change")),
            ("the wall rule (Newtonian run in the slip-form geometry over the control): D_T, flux form",
             r"rule: $D_T$, flux form", lambda run: rule(run, "rotor_circle_with_advection_change")),
            ("the wall rule: D_T on the housing", r"rule: $D_T$, housing", lambda run: rule(run, "housing_wall_change")),
            ("the wall rule: D_T, wall form", r"rule: $D_T$, wall form", lambda run: rule(run, "rotor_wall_change")),
            ("the wall rule: D_T, wall form, lateral surface", r"rule: $D_T$, lateral surface",
             lambda run: rule(run, "rotor_wall_lateral_change")),
            ("the wall rule: D_T, wall form, end faces", r"rule: $D_T$, end faces",
             lambda run: rule(run, "rotor_wall_end_change")),
            ("the wall rule: D_T on the housing, lateral face", r"rule: $D_T$, housing, lateral face",
             lambda run: rule(run, "housing_wall_lateral_change")),
            ("the wall rule: D_T on the housing, end faces", r"rule: $D_T$, housing, end faces",
             lambda run: rule(run, "housing_wall_end_change")),
            ("the wall rule: change of the mid-plane constant", r"rule: $D$, mid-plane",
             lambda run: rule(run, "midplane_change")),
            ("exact plane film between two discs (end faces)", "plane film, end faces",
             lambda run: change(run, "end_faces_plane_film_change")),
            ("mid-plane constant over the 2-D one, Newtonian", "mid-plane constant / 2-D, Newtonian",
             lambda run: f"{runs_[pairs[run]['control']]['midplane_beta_over_two_dimensional']:.4f}"),
            ("mid-plane constant over the 2-D one, micropolar", "mid-plane constant / 2-D, micropolar",
             lambda run: f"{runs_[run]['midplane_fit']['potential_vortex_constant'] / runs_[run]['exact_two_dimensional']['potential_vortex_constant']:.4f}"),
            ("micropolar effect in the mid-plane (change of the constant)", r"$D$, mid-plane",
             lambda run: change(run, "midplane_change")),
            ("the same over the exact 2-D D", r"$D$, mid-plane / exact",
             lambda run: f"{pairs[run]['midplane_change'] / pairs[run]['two_dimensional_change']:.4f}"),
            ("reach of the ends, 1 % (g)", r"reach of the ends, 1\,\% ($g$)",
             lambda run: f"{runs_[run]['end_reach']['reach_in_gaps']['0.01']:.2f}"),
            ("reach of the ends, 0.1 % (g)", r"reach of the ends, 0.1\,\% ($g$)",
             lambda run: f"{runs_[run]['end_reach']['reach_in_gaps']['0.001']:.2f}"),
            ("wall layer on the end face, normal component: length over l sqrt(nu_inf / nu)",
             r"layer of $w_z$ / ($l\sqrt{\nu_\infty/\nu}$)",
             lambda run: f"{runs_[run]['end_face_layers']['normal']['fitted_over_expected']:.3f}"),
            ("wall layer on the end face, tangential component: length over l", r"layer of $w_r$ / $l$",
             lambda run: f"{runs_[run]['end_face_layers']['tangential']['fitted_over_expected']:.3f}"),
            ("mean density - rho0, over rho0 (Newtonian; micropolar)", r"$\bar\rho/\rho_0 - 1$ (N; M)",
             lambda run: f"{runs_[pairs[run]['control']]['density_level']:+.1e}; {runs_[run]['density_level']:+.1e}"),
            ("mean pressure in rho U^2 (Newtonian; micropolar)", r"$\bar p/(\rho U^2)$ (N; M)",
             lambda run: f"{runs_[pairs[run]['control']]['pressure_level']:+.3f}; {runs_[run]['pressure_level']:+.3f}"),
            ("its drift per eighth of the run (Newtonian; micropolar)", "drift per eighth (N; M)",
             lambda run: f"{runs_[pairs[run]['control']]['pressure_drift']:+.1e}; {runs_[run]['pressure_drift']:+.1e}"),
            ("steps per second (Newtonian; micropolar)", "steps/s (N; M)",
             lambda run: f"{runs_[pairs[run]['control']]['steps_per_second']:.1f}; {runs_[run]['steps_per_second']:.1f}"),
        ]
        header_b = ["quantity"] + [f"{label} ({run})" for run, label in columns]
        tex_header_b = ["quantity"] + [label.replace("g = ", "$g = ").replace(" dx", r"\,\Delta x$")
                                       for _run, label in columns]
        plain_b = [[plain] + [function(run) for run, _label in columns] for plain, _tex, function in specification]
        tex_b = [[tex] + [style.tex_escape(function(run)) for run, _label in columns]
                 for _plain, tex, function in specification]
        style.write_table("tab_15b_three_dimensional", header_b, plain_b, tex_header_b, "l" + "r" * len(columns),
                          tex_b, note="Torques in N m for the case's units ($g = 0.1$ m, $U = 1$ m/s, "
                                      "$\\mu_\\infty = 10$ Pa s); N: Newtonian control, M: micropolar run.")
        files_b = ["tab_15b_three_dimensional.csv", "tab_15b_three_dimensional.tex"]
    final = name.startswith("g40") and not partial
    status = ("final" if final else
              "provisional (the g = 40 dx run on two slabs, stopped at step "
              f"{partial['last_heartbeat_step']:,} of {partial['planned_steps']:,}: azimuthal means near its end; "
              "no Newtonian control and no torque at g = 40 dx)" if partial else
              "provisional (the g = 20 dx run; the g = 40 dx run replaces it)")
    style.record("15", "3-D: fields in a plane through the axis and the mid-plane profiles", status, caption,
                 ["fig_15_three_dimensional.pdf", "fig_15_three_dimensional.png"] + files_b,
                 [ANALYSIS, MAPS.format(run=name) + ".npz", name], numbers, TAG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
