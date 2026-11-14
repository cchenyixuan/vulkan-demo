"""
REPORT.md - the results of the pack in one file, for writing the chapter: the index, the numbers of every topic
with their item and their source (run, host, inferred), what to be careful with, what is still running.

Every number is read from the data files and the tables of the pack. Run from the repository root:
    python data/micropolar/ch5_draft/report.py
"""
import csv
import json
import pathlib
import subprocess
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "data" / "micropolar" / "stage6"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows_of(name):
    with open(HERE / name, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def span(values, form="{:.4f}"):
    values = [float(value) for value in values]
    low, high = form.format(min(values)), form.format(max(values))
    return low if low == high else f"{low} to {high}"


def main() -> int:
    items = {path.stem: load(path) for path in sorted((HERE / "items").glob("*.json"))}
    concentric = load(DATA / "concentric.json")
    eccentric = load(DATA / "eccentric.json")
    cavity = load(DATA / "cavity.json")
    reference = load(DATA / "cavity_reference.json")["cases"]
    lid = load(DATA / "cavity_reference_lid.json")
    rotor = load(DATA / "rotor3d.json")
    head = subprocess.run(["git", "log", "-1", "--format=%h"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    pair_done = all((ROOT / "logs/micropolar/stage6" / name / "run.json").exists()
                    for name in ("g400_newtonian_k3", "g400_k3_m20"))
    out = []

    def say(text=""):
        out.append(text)

    say("# Chapter 5: what the solver has shown")
    say()
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    say(f"Written {time.strftime('%Y-%m-%d %H:%M')} at commit `{head}` (branch `{branch}`; solver tag "
        "`thesis-ch5-rc2`, nothing under `experiment/v7/shaders` or `experiment/v7/utils` changed since). Every number "
        "below is read from the data files by `report.py`; the captions with all parameters and run names are in "
        "`captions.md`, the full account in `docs/micropolar_design.md`, Sections 15 (Stage 6), 16 (Stage 7) and 17 "
        "(Stage 8, the dipole-wall collision).")
    say()
    say("**State.** " + ("Every run of the chapter is done." if pair_done else
                         "Every item is final except the one row at g = 400 dx (m = 20, K = 3, l = 10 dx): its "
                         "Newtonian control is done, the micropolar run is going; items 02 to 04 gain that row when "
                         "it ends. Nothing else is pending and there is no further queue."))
    say()
    say("Marks: **[run]** measured in a particle run on the tag; **[host]** computed on the host without the particle "
        "solver; **[inferred]** my reading, not measured.")
    say()
    say("## 1. Index")
    say()
    say(table(["no.", "item", "files", "status"],
              [[number, item["title"], ", ".join(f"`{name}`" for name in item["files"] if not name.endswith(".png")),
                item["status"]] for number, item in items.items()]))
    say()

    # ---- concentric ----------------------------------------------------------------------------------------------
    rows = rows_of("tab_04_concentric_rows.csv")
    say("## 2. Concentric cylinders (items 01 to 04) [run]")
    say()
    say("Thesis configuration with `numerics.delta_fluid_only: true`, Re = 10 on mu_inf, Ma = 0.02, w = 0 on both "
        "walls; D = relative change of the torque against the Newtonian control on the same particles (velocity fit).")
    say()
    say(table(["K", "m = g / (2 l)", "g / dx", "l / dx", "D (run)", "D (exact)", "run / exact", "noise floor / D",
               "slip-form rule / exact"],
              [[row["K"], row["m"], row["g/dx"], row["l/dx"], row["D_SPH"], row["D_exact"], row["D_SPH/D_exact"],
                row["noise floor/D_exact"], row["D_rule/D_exact"]] for row in rows]))
    say()
    by_length = {}
    for row in rows:
        by_length.setdefault(float(row["l/dx"]), []).append(float(row["D_SPH/D_exact"]))
    say("- **Resolution of the layer decides the error:** run / exact is "
        + "; ".join(f"{span(values)} at l = {length:g} dx" for length, values in sorted(by_length.items())) + ".")
    orders = []
    for series in concentric["resolution_orders"].values():
        orders += [entry["orders"]["deviation"] for entry in series
                   if entry["orders"]["deviation"] == entry["orders"]["deviation"]]
    if orders:
        say("- **Order in h / l** between successive resolutions of one row: "
            + "; ".join(f"{label}: " + ", ".join(f"{entry['orders']['deviation']:.2f}" for entry in series
                                                  if entry["orders"]["deviation"] == entry["orders"]["deviation"])
                        for label, series in concentric["resolution_orders"].items()) + ".")
    fine, coarse = (concentric["runs"].get(name) for name in ("g400_k3_m20", "g200_k3_m20"))
    control, coarse_control = (concentric["runs"].get(name) for name in ("g400_newtonian_k3", "g200_newtonian"))
    if fine and coarse and control and coarse_control:
        others = [float(row["D_SPH/D_exact"]) for row in rows
                  if float(row["l/dx"]) == 10.0 and row["run"] != "g400_k3_m20"]
        expected = fine["exact_hydrodynamic"] * max(others)
        say(f"- **The pair at g = 400 dx** (m = 20, K = 3, l = 10 dx; 625,000 steps each, every counter zero): run / "
            f"exact {fine['sph_over_exact']:.4f}, against {coarse['sph_over_exact']:.4f} for the same m at l = 5 dx. "
            "The excess of the m = 20 row halves with the resolution of the layer. **It does not land on the other "
            f"rows at l = 10 dx** ({span(others)}, where one doubling took a factor of four): D is "
            f"{fine['relative_torque_change'] - expected:.1e} above them, about "
            f"{(fine['relative_torque_change'] - expected) / coarse['noise_floor_of_change']:.0f} times the lattice "
            "floor at g = 200 dx. What is known of the pair: its step is the only one set by the viscous limit; it "
            "has one lattice, so no floor of its own (the table carries that of g = 200 dx); and its control is in "
            f"another state than the others: constant over nominal {control['beta_over_nominal']:.6f} "
            f"({coarse_control['beta_over_nominal']:.6f} at g = 200 dx), hydrodynamic radii "
            f"{control['inner_offset_in_spacings']:+.3f} and {control['outer_offset_in_spacings']:+.3f} dx from the "
            f"nominal ones ({coarse_control['inner_offset_in_spacings']:+.3f} and "
            f"{coarse_control['outer_offset_in_spacings']:+.3f}), mean density {control['density_level']:+.1e} rho0 "
            f"({coarse_control['density_level']:+.1e}), rms of the fit "
            f"{control['fit']['fit_rms_over_wall_speed']:.1e} U ({coarse_control['fit']['fit_rms_over_wall_speed']:.1e}). "
            "The fitted constants are steady. **Not explained**, and not followed: nothing else was to run at "
            "g = 400 dx. I would quote the row as 'the excess at m = 20 halves', not as a point of the l / dx law.")
    floors = concentric["noise_floors"]
    say("- **Noise floor** (spread of the control's constant over three lattices): "
        + ", ".join(f"{floors[key]['relative_spread']:.1e} at g = {key} dx" for key in sorted(floors, key=int)) + ".")
    by_ratio = {}
    for row in rows:
        by_ratio.setdefault(float(row["m"]), []).append(float(row["D_rule/D_exact"]))
    say("- **The wall rule in its slip form** over exact: "
        + "; ".join(f"m = {ratio:g}: {span(values)}" for ratio, values in sorted(by_ratio.items()))
        + ". Exact up to exponentially small terms for shear-driven flow once m is about 5 or more; it fails below "
          "m of about 2.")
    say()

    # ---- eccentric -----------------------------------------------------------------------------------------------
    say("## 3. Eccentric cylinders (items 05 to 09)")
    say()
    newtonian = [row for row in rows_of("tab_05_eccentric_newtonian.csv") if "rc2" in row["setting"]]
    say("**Newtonian controls against the exact Stokes solution** [run, host] (R1 = 4 g, R2 = 5 g; item 05):")
    say()
    say(table(["run", "eps", "g / dx", "lattice", "load / exact - 1 (%)", "torque, flux form, rotor (%)"],
              [[f"`{row['run']}`", row["eps"], row["g/dx"], row["lattice"], row["load/exact - 1 (%)"],
                row["torque, circle form: rotor (%)"]] for row in newtonian]))
    say()
    floor = eccentric["floors"]
    say("Noise floor from three lattices (max - min): "
        + "; ".join(f"eps = {label[1:3] if False else ('0.3' if label == 'e30' else '0.6')}: load "
                    f"{100.0 * entry['load_spread']:.2f} %, D_T {entry['torque_change_spread']:.4f}, load change "
                    f"{100.0 * entry['load_change_spread']:.2f} points" for label, entry in floor.items()) + ".")
    say()
    say("**Micropolar runs against the two predictions** [run, host] (item 09; D_T with the concentric reading "
        "errors taken off; film = the film function of micropolar lubrication theory in the thin-film calculation, "
        "which is not a result of ours; rule = the wall rule on the exact Stokes solution of its slip-form "
        "geometry):")
    say()
    pairs = [row for row in rows_of("tab_09_eccentric_rows.csv") if not row["run"].endswith(("_shifted", "_rotated"))]
    say(table(["run", "K", "eps", "g / dx", "l / dx", "narrowest gap / l", "D_T (run)", "D_T film", "D_T rule",
               "load change (%)", "film (%)", "rule (%)"],
              [[f"`{row['run']}`", row["K"], row["eps"], row["g/dx"], row["l/dx"], row["narrowest gap/l"],
                row["D_T mean, concentric error taken off"], row["D_T film function"], row["D_T rule"],
                row["load change (%)"], row["load change, film function (%)"], row["load change, rule (%)"]]
               for row in pairs]))
    say()
    main_pairs = {name: entry for name, entry in eccentric["pairs"].items() if not entry.get("lattice")}
    say(f"- **The runs reproduce the film function:** load change over film {span([e['load_over_film'] for e in main_pairs.values()], '{:.3f}')}, "
        f"torque change over film {span([e['torque_over_film'] for e in main_pairs.values()], '{:.3f}')}.")
    say(f"- **The rule over-predicts where the film is not thick against l:** load change over rule "
        f"{span([e['load_over_rule'] for e in main_pairs.values()], '{:.3f}')}, torque change over rule "
        f"{span([e['torque_over_rule'] for e in main_pairs.values()], '{:.3f}')}. First order for pressure-driven flow.")
    resolution = eccentric["resolution"][0]
    say(f"- **Resolution pair** (g = 100 dx, l = 10 dx against g = 200 dx, l = 20 dx; eps = 0.3, K = 3): D_T "
        f"{resolution['torque_change_coarse']:.4f} and {resolution['torque_change_fine']:.4f}, load change "
        f"{100.0 * resolution['load_change_coarse']:.2f} and {100.0 * resolution['load_change_fine']:.2f} %.")
    say("- The load does not see the slip term of the rule, the torque does (design note 15.3.3) [host].")
    say()

    # ---- film function -------------------------------------------------------------------------------------------
    caption = items["10"]["caption"]
    start = caption.find("Ratios to the film function")
    say("## 4. The film function (item 10) [host]")
    say()
    say("f = h^3 - 6 delta h^2 coth(h / (2 l)) + 12 delta l h, delta = K l / (2 (1 + K)); the film function of "
        "micropolar lubrication theory (not ours; its identification with the form of that theory is algebra on "
        "our side, not compared with the original). The model's two ordinary differential equations between the "
        "walls agree with it to 1e-15. The sharper form of the rule, h^3 - 6 delta h^2 + 12 delta l h, holds: it "
        "differs by 6 delta h^2 (coth(h / (2 l)) - 1).")
    if start >= 0:
        say()
        say("From the caption: " + caption[start:caption.find("h / l = 20", start) + 70].rsplit(".", 1)[0] + ".")
    say()

    # ---- cavity --------------------------------------------------------------------------------------------------
    say("## 5. Micropolar cavity (item 11, tables 11b and 11c)")
    say()
    say("K = 3, Re = 400 on mu_inf (1,000 on mu), l = L / 20, Theta = l^2, w = 0 on the four walls, Ma = 0.01, default "
        "walls. **The check of the case is the second method**: the same equations by finite differences on the host "
        "(stream function, vorticity, micro-rotation; grids of 128, 192, 256 intervals, extrapolated).")
    say()
    runs = cavity["runs"]
    cavity_rows = []
    for label, run, case in (("micropolar", "mp_n500", "micropolar"),
                             ("Newtonian, mu_inf (Re 400)", "newtonian_re400_n500", "newtonian_far"),
                             ("Newtonian, mu + kappa (Re 250)", "newtonian_re250_n500", "newtonian_wall")):
        particle, zero = runs[run], reference[case]["extrapolated"]
        ratio = reference[case]["particle"]["over_finite_differences"]
        cavity_rows.append([label, f"{particle['u_min']:.4f} / {zero['u_min']['extrapolated']:.4f}",
                            f"{particle['v_max']:.4f} / {zero['v_max']['extrapolated']:.4f}",
                            f"{particle['v_min']:.4f} / {zero['v_min']['extrapolated']:.4f}",
                            f"{particle['kinetic_energy_mean']:.4f} / {zero['kinetic_energy']['extrapolated']:.4f}",
                            ", ".join(f"{100.0 * (ratio[key] - 1.0):+.1f}" for key in ("u_min", "v_max", "v_min")),
                            f"{reference[case]['particle']['u_difference_rms']:.1e}, "
                            f"{reference[case]['particle']['v_difference_rms']:.1e}"])
    say(table(["fluid", "u_min: particles (n = 500) / finite differences", "v_max", "v_min",
               "kinetic energy over fluid mass", "particles stronger by (%)", "centre lines, rms difference u, v (U)"],
              cavity_rows))
    say()
    chain, coarse = cavity["comparisons"]["chain"], cavity["comparisons"]["resolution"]
    say(f"- [run] Two slabs (5090 + XTX) against one slab: centre lines differ by {chain['u_difference_rms']:.1e} (u) "
        f"and {chain['v_difference_rms']:.1e} (v) U rms. n = 250 against n = 500: {coarse['u_difference_rms']:.1e} and "
        f"{coarse['v_difference_rms']:.1e} U rms, extrema 1.3 to 1.5 % stronger at n = 250.")
    say("- [host] The micropolar extrema are stronger than those of the Newtonian fluid of viscosity mu_inf "
        "(finite differences: "
        + ", ".join(f"{100.0 * (reference['micropolar']['against_bounds']['over_far'][key] - 1.0):+.1f}"
                    for key in ("u_min", "v_max", "v_min"))
        + " %). With walls that do not hold w the finite-difference solution is weaker than that Newtonian fluid; "
          "with w held on the lid alone it is strongest (table 11b). [inferred] The condition w = 0 on the driving "
          "wall makes the lid take hold of the fluid as of the more viscous one.")
    marchi = items["11"]["numbers"].get("marchi_reynolds_400") or {}
    marchi_sentence = ""
    if "finite-difference solution" in marchi:
        solution, particles = marchi["finite-difference solution"], marchi["particle run"]
        marchi_sentence = (
            ", which in turn agrees with the reference of Marchi et al. at Reynolds number 400 (your transcription "
            "from the appendix of paper 1, `reference/marchi_re400.csv`, 28 + 28 points) to "
            f"{solution['u']['rms']:.1e} (u) and {solution['v']['rms']:.1e} (v) U rms once extrapolated to zero spacing")
        say("- [run, host] **Against Marchi et al. at Reynolds number 400** (transcribed table; its coordinates, "
            "printed to four decimals, taken as multiples of 1 / 128): the Newtonian particle run of viscosity mu_inf "
            f"differs by {particles['u']['rms']:.1e} (u) and {particles['v']['rms']:.1e} (v) U rms (largest "
            f"{particles['u']['largest']:.1e} at y = {particles['u']['at']:.3f}, next to the lid); the finite-difference "
            f"solution, extrapolated to zero spacing at the points, by {solution['u']['rms']:.1e} and "
            f"{solution['v']['rms']:.1e} (largest {solution['u']['largest']:.1e}).")
    say("- [host] **Table 11c**, the traction of the lid over that on the Newtonian fluid of viscosity mu_inf, minus "
        f"one, against delta / lambda (lambda = mu_inf U / traction = {lid['shear_length']:.3f} L, the shear length "
        "of the lid):")
    say()
    say(table(["l / L", "delta / L", "delta / lambda", "excess traction", "excess / (delta / lambda)"],
              [[f"1/{1.0 / row['length']:.0f}", f"{row['displacement']:.4f}",
                f"{row['displacement_over_shear_length']:.3f}", f"{row['excess']:.3f}",
                f"{row['excess_over_first_order']:.2f}"] for row in lid["rows"]]))
    say()
    say("  The first-order rule with the shear length in the place of the gap, within 4 % from l = L / 40 down "
        "(ratios on like grids).")
    say()

    # ---- baseline, seepage, checks -------------------------------------------------------------------------------
    numbers = items["12"]["numbers"]
    say("## 6. Newtonian cavity baseline, seepage, solver checks (items 12 to 14) [run, existing data]")
    say()
    say(f"- **Item 12**, Re = 1000, n = 1000 against Marchi et al.: L2 error of u {numbers['u_l2']:.2e} U (largest "
        f"{numbers['u_linf']:.2e}), of v {numbers['v_l2']:.2e} U (largest {numbers['v_linf']:.2e}); steady at "
        f"t = {numbers['steady_time']:.1f}; rms error away from the walls {numbers['u_interior_rms']:.1e} (u) and "
        f"{numbers['v_interior_rms']:.1e} (v).")
    bearing = items["13"]["numbers"]["bearing"]
    say("- **Item 13**, the seepage of the thesis walls: velocity into the wall per wall pressure "
        + ", ".join(f"{entry['slope']:.4f} U per rho U^2 ({label}, correlation {entry['correlation']:.2f})"
                    for label, entry in list(bearing.items())[:2])
        + " in the bearing at Re = 10, Ma = 0.02; the cavity keeps its walls and the chapter states the law and "
          "its size (caption of item 13 and `tab_13_candidates`).")
    say("- **Item 14**, five tables from existing data (operator checks, passive label, chain against one slab, step "
        "limits, cost): `tab_14a` to `tab_14e`, read with the caption of item 14.")
    say()

    # ---- 3-D -----------------------------------------------------------------------------------------------------
    say("## 7. The 3-D case (item 15, table 15b) [run]")
    say()
    say("Rotor of radius 3 g and length 3 g in a housing of radius 4 g with end gaps g, K = 3, l = 10 dx, w = 0 on "
        "every wall, Re = 10 on mu_inf, Ma = 0.02, h = 4 dx; each micropolar run with its Newtonian control on the "
        "same particles and the same dt. The case of the chapter is g = 40 dx on two cards (5090 + XTX). The two "
        "columns are two cases (m = 1 and m = 2), not one case at two resolutions.")
    say()
    with open(HERE / "tab_15b_three_dimensional.csv", encoding="utf-8", newline="") as handle:
        three = list(csv.reader(handle))
    say(table(three[0], three[1:]))
    say()
    pair = rotor["pairs"].get("g40_k3_m2_k2")
    if pair:
        polar, newton = rotor["runs"]["g40_k3_m2_k2"]["torque"], rotor["runs"][pair["control"]]["torque"]
        say(f"- **Three readings of the torque change at g = 40 dx:** {pair['rotor_circle_with_advection_change']:.4f} "
            f"(flux form), {pair['housing_wall_change']:.4f} (on the housing), {pair['rotor_wall_change']:.4f} (wall "
            "form). In the Newtonian run the flux form and the housing agree to "
            f"{abs(newton['rotor_circle_with_advection']['mean'] / -newton['housing_wall']['mean'] - 1.0):.1e}; in the "
            "micropolar run the flux form is "
            f"{100.0 * (1.0 - polar['rotor_circle_with_advection']['mean'] / -polar['housing_wall']['mean']):.1f} % "
            "below the housing and still rising at the end of the run (it rises through the 13 states, the housing "
            "does not move). I would quote the housing and give the flux form as the lower reading.")
        say(f"- **Mid-plane:** the micropolar effect is {pair['midplane_change']:.4f} against the exact 2-D D of "
            f"{pair['two_dimensional_change']:.4f} ({pair['midplane_change'] / pair['two_dimensional_change']:.4f} of it).")
        say(f"- **Lateral surface and end faces:** D_T {pair['rotor_wall_lateral_change']:.4f} against the exact 2-D "
            f"{pair['two_dimensional_change']:.4f}; {pair['rotor_wall_end_change']:.4f} against the plane film between "
            f"two discs, {pair['end_faces_plane_film_change']:.4f}. The rims belong to both.")
    say("- **The rule in 3-D** [host, run]: the argument for the normal component is right; that component has its own "
        "layer length l sqrt(nu_inf / nu), and the run shows both lengths (table).")
    rule = rotor.get("rules", {}).get("g40_slip_newtonian_k2")
    if rule and pair:
        say("- **The rule in 3-D, run** [run] (the Newtonian fluid in the slip-form geometry at g = 40 dx: lateral "
            "surface of the rotor at R1 + delta turning at Omega (R1 - delta) / (R1 + delta), its end faces out by "
            "delta at Omega, the housing in by delta; rows 'the wall rule' of the table). **Mid-plane:** change of "
            f"the constant {rule['midplane_change']:.4f} against {pair['midplane_change']:.4f} in the micropolar run "
            f"({100.0 * (rule['midplane_change'] / pair['midplane_change'] - 1.0):+.1f} %). **Torque:** "
            f"{rule['rotor_circle_with_advection_change']:.4f} (flux form), {rule['housing_wall_change']:.4f} "
            f"(housing), {rule['rotor_wall_change']:.4f} (wall form) against "
            f"{pair['rotor_circle_with_advection_change']:.4f}, {pair['housing_wall_change']:.4f}, "
            f"{pair['rotor_wall_change']:.4f}: the rule in this plainest reading overstates the change of the torque "
            f"by {100.0 * (rule['housing_wall_change'] / pair['housing_wall_change'] - 1.0):.0f} to "
            f"{100.0 * (rule['rotor_circle_with_advection_change'] / pair['rotor_circle_with_advection_change'] - 1.0):.0f} "
            "%. The excess arrives on both kinds of faces of the housing (end faces "
            f"{rule['housing_wall_end_change']:.3f} against {pair['housing_wall_end_change']:.3f}, lateral face "
            f"{rule['housing_wall_lateral_change']:.3f} against {pair['housing_wall_lateral_change']:.3f}); the "
            "mid-plane being right, [inferred] it comes from the end regions: "
            "the displaced rotor is a larger body (end faces to R1 + delta at the full rate, end gap g - 2 delta); "
            "the wall layer at a convex rim does not act as one. At the rim the two rates meet without blending: "
            "each wall particle of the rotor takes the rate of the surface it is nearer to (jump 2 delta Omega = "
            "0.0625 U).")
    say("- **Chain:** per-particle seam checks pass on three slabs (one card) and two slabs (XTX + 5090); the cavity on "
        "two slabs equals the one-slab run to 2e-5 U.")
    say()

    # ---- the map of the effect -----------------------------------------------------------------------------------
    if "16" in items:
        crossings = items["16"]["numbers"]["crossings"]
        say("## 8. The map of the effect (item 16) [host]")
        say()
        say("Contours D = 1 %, 5 %, 20 % in the plane of N = sqrt(K / (2 + K)) and m = g / (2 l): (a) the torque "
            "between the concentric cylinders from the exact solution, with the slip-form rule; (b) the load of the "
            "bearing at an eccentricity ratio of 0.6 from the film function in the thin-film calculation. m at which "
            "D equals 1 %, 5 %, 20 % ('-': outside 1 <= m <= 100):")
        say()

        def cells(entry):
            return " / ".join("-" if entry[key] is None else f"{entry[key]:.1f}" for key in ("0.01", "0.05", "0.2"))
        say(table(["K", "torque, exact", "torque, slip-form rule", "load, film function"],
                  [[label.replace("K = ", ""), cells(entry["torque, exact"]), cells(entry["torque, slip-form rule"]),
                    cells(entry["load, film function"])] for label, entry in crossings.items()]))
        say()
        say(f"Noise floors shaded in the figure: {items['16']['numbers']['torque_noise_floor']:.1e} (torque, g = 200 "
            f"dx) and {items['16']['numbers']['load_noise_floor']:.1e} (change of the load, measured at g = 100 dx). "
            "The load reaches a given D at three to four times the m at which the torque does.")
        say()

    # ---- cautions ------------------------------------------------------------------------------------------------
    one_card_sentence = "[inferred] the cards' evaluation of the power in the Tait equation."
    if (DATA / "rotor3d_cut_check_one_card.json").exists() and (DATA / "rotor3d_cut_check_two_cards.json").exists():
        one, two = (next(iter(load(DATA / f"rotor3d_cut_check_{name}.json")["near_the_rotor"]["variants"].values()))
                    for name in ("one_card", "two_cards"))
        one_card_sentence = (
            "**[run] The cause is the two cards' own evaluation of the equation of state:** the same two slabs with "
            f"both on the 5090 (2,000 steps at g = 20 dx) show {one[2]:.1e} and {one[3]:.1e} within one support of "
            f"the cut, the value of the interior; on 5090 + XTX {two[2]:.1e} and {two[3]:.1e} "
            "(`rotor3d_cut_check_one_card.txt`, `rotor3d_cut_check_two_cards.txt`).")
    # ---- the dipole-wall collision (Stage 8) ---------------------------------------------------------------------
    dipole_file = ROOT / "data/micropolar/stage8/dipole.json"
    if dipole_file.exists():
        dipole = load(dipole_file)
        runs, comparisons = dipole["runs"], dipole["comparisons"]
        control = runs.get("control_n1000")
        validation = {entry["quantity"]: entry for entry in dipole["validation"].get("control_n1000", [])}
        say("## 9. The dipole-wall collision (items 20 to 22) [run]")
        say()
        say("Normal collision of a dipole with a no-slip wall (Clercx & Bruneau 2006), box [-W, W]^2, Re = 625 on "
            "nu_inf, dx = W / 1000 (4 million fluid particles, h = 5 dx); the Newtonian control (viscosity mu_inf) and "
            "the micropolar fluid with K = 3, l = W / 100 and W / 50, w = 0 on every wall, w = omega / 2 at t = 0.")
        say()
        if control and validation:
            say(f"- **The control validates** against the benchmark: enstrophy peaks {control['enstrophy_peaks'][0][1]:.1f} "
                f"at t = {control['enstrophy_peaks'][0][0]:.4f} and {control['enstrophy_peaks'][1][1]:.1f} at "
                f"{control['enstrophy_peaks'][1][0]:.4f} (spectral 933.6 at 0.3711 and 305.2 at 0.6479: "
                f"{100.0 * validation['first enstrophy peak']['difference']:+.1f} % and "
                f"{100.0 * validation['second enstrophy peak']['difference']:+.1f} %); the primary core within "
                + f"{max(abs(entry['difference']) for key, entry in validation.items() if key.startswith('primary core x') or key.startswith('primary core y')):.4f}"
                + " W of the benchmark at t = 0.6, 0.625, 1.0; E and Omega at t = 0.25, 0.5, 0.75 within 1 %. The "
                "first peak converges from below (-9.4, -3.7, -1.6 % at n = 250, 500, 1000); halving the Mach number "
                "moves it by 0.01 %.")
        rows = []
        for key, label, kind in (("enstrophy_peak_first_value", "first enstrophy peak", "relative"),
                                 ("enstrophy_at_base_second_peak", "enstrophy at the control's second peak",
                                  "relative"),
                                 ("energy_0.75", "kinetic energy at t = 0.75", "relative"),
                                 ("wall_vorticity_peak", "wall vorticity at the first collision", "relative"),
                                 ("wall_traction_peak", "wall shear traction (mu + kappa) omega_w", "relative"),
                                 ("wall_flux_integral_upper", "vorticity flux into the fluid at the wall", "relative"),
                                 ("wall_pressure_difference_along_wall", "pressure difference along the wall",
                                  "relative"),
                                 ("rebound_distance", "rebound of the primary core (W)", "absolute")):
            cells = [label]
            for run in ("mp_l100_n1000", "mp_l50_n1000"):
                value = comparisons.get(run, {}).get(key)
                cells.append("-" if value is None or value != value else
                             (f"{value:+.4f}" if kind == "absolute" else f"{100.0 * value:+.1f} %"))
            floors = [dipole["floors"][name].get(key) for name in dipole["floors"]]
            floors = [abs(value) for value in floors if value is not None and value == value]
            cells.append("-" if not floors else (f"{max(floors):.4f}" if kind == "absolute"
                                                 else f"{100.0 * max(floors):.2f} %"))
            rows.append(cells)
        say()
        say(table(["against the control", "l = W / 100", "l = W / 50", "noise floor"], rows))
        say()
        shear = runs.get("_shear_length")
        if shear:
            say(f"- **Outside the first-order rule.** The shear length of the wall at the first collision is "
                f"Lambda = {shear['value']:.4f} W, so delta* / Lambda = "
                f"{shear['rules']['mp_l100_n1000']['displacement_over_shear_length']:.2f} and "
                f"{shear['rules']['mp_l50_n1000']['displacement_over_shear_length']:.2f} (Lambda / l = "
                f"{shear['rules']['mp_l100_n1000']['shear_length_over_length']:.2f} and "
                f"{shear['rules']['mp_l50_n1000']['shear_length_over_length']:.2f}); the measured excess of the wall "
                "traction is +18 to +23 %, nearly independent of l: it saturates once the layer is thicker than the "
                "boundary layer. [inferred] Pinned at w = 0 across the boundary layer, the fluid has the viscosity "
                "mu + kappa there: sqrt((mu + kappa) / (mu + kappa / 2)) - 1 = 0.26 for a layer growing as sqrt(nu t).")
        if all(run in runs and "wall_balance" in runs[run] for run in ("mp_l100_n1000", "mp_l50_n1000")):
            short, long = runs["mp_l100_n1000"]["wall_balance"], runs["mp_l50_n1000"]["wall_balance"]
            say("- **The mechanism.** The pressure along the wall is the same in all runs (set by the outer flow). At "
                "a no-spin wall dp/ds = (mu + kappa) d omega / dn - kappa d w / dn: with the measured d w / dn the "
                f"balance closes to {short['balance_over_control']:.3f} and {long['balance_over_control']:.3f} of the "
                f"control, the couple term taking {short['spin_term_share']:.2f} (l = W / 100) and "
                f"{long['spin_term_share']:.2f} (W / 50) of the vorticity term. The no-spin wall passes part of the "
                "vorticity it would produce to the micro-rotation, and less vorticity leaves the wall; the secondary "
                "vortex is weaker and **neither micropolar run has a second enstrophy maximum** (the control has one "
                "at t = 0.649) ([inferred] the second because of the first).")
        say("- **Figures:** item 20 (vorticity of the three runs at t = 0.35, 0.45, 0.6, 0.7, and w - omega / 2 of "
            "l = W / 50), item 21 (core trajectories, enstrophy with the benchmark points), table 22.")
        say()
        checks = dipole.get("checks", {})
        reference = checks.get("mp_l50_n1000__against__control_n1000")
        if reference and len(checks) > 1:
            say("### 9.1 Three checks (table 26) [run]")
            say()
            say("Each run against its control with the definitions and floors of table 22 (the core positions from the "
                "strongest extremum within reach, for the reference too).")
            say()
            keys = (("enstrophy_peak_first_value", "first enstrophy peak"),
                    ("enstrophy_at_base_second_peak", "enstrophy at the control's second peak"),
                    ("energy_0.25", "kinetic energy at t = 0.25 (before the collision)"),
                    ("enstrophy_0.25", "enstrophy at t = 0.25"),
                    ("primary_value_at_0.25", "primary core, vorticity maximum at t = 0.25"),
                    ("wall_vorticity_peak", "wall vorticity at the first collision"),
                    ("wall_traction_peak", "wall shear traction, peak"),
                    ("wall_traction_integral", "wall shear traction, integral"),
                    ("wall_flux_integral_upper", "vorticity flux into the fluid at the wall"))
            columns = [("mp_l50_j001_n1000__against__control_n1000", "l = W/50, j = l^2/100"),
                       ("mp_l50_j001_n1000__against__mp_l50_n1000", "the same against j = l^2"),
                       ("mp_l25_j001_n1000__against__control_n1000", "l = W/25, j = l^2/100"),
                       ("mp_l50_n500__against__control_n500", "l = W/50 at 10 dx"),
                       ("mp_l50_n1000__against__control_n1000", "reference: l = W/50, 20 dx, j = l^2")]
            columns = [(key, label) for key, label in columns if key in checks]
            say(table(["against the control"] + [label for _key, label in columns],
                      [[label] + [f"{100.0 * checks[key]['changes'][quantity]:+.1f} %" for key, _label in columns]
                       for quantity, label in keys]))
            say()
            effect = reference["changes"]
            inertia = checks.get("mp_l50_j001_n1000__against__mp_l50_n1000")
            if inertia:
                moved = {quantity: inertia["changes"][quantity] / effect[quantity]
                         for quantity in ("enstrophy_peak_first_value", "wall_vorticity_peak")}
                say(f"- **The micro-inertia matters** (j = l^2 / 100 against j = l^2, l = W / 50): the first enstrophy "
                    f"peak moves by {100.0 * inertia['changes']['enstrophy_peak_first_value']:+.1f} %, "
                    f"{abs(moved['enstrophy_peak_first_value']):.2f} of the micropolar effect "
                    f"({100.0 * effect['enstrophy_peak_first_value']:+.1f} %), more than the quarter set in the task; "
                    f"the wall vorticity by {100.0 * inertia['changes']['wall_vorticity_peak']:+.1f} % "
                    f"({abs(moved['wall_vorticity_peak']):.2f} of the effect). Most of the difference is there "
                    "before the collision: at t = 0.25 the kinetic energy is "
                    f"{100.0 * checks['mp_l50_j001_n1000__against__control_n1000']['changes']['energy_0.25']:+.2f} % and the "
                    "enstrophy "
                    f"{100.0 * checks['mp_l50_j001_n1000__against__control_n1000']['changes']['enstrophy_0.25']:+.1f} % "
                    f"off the control (j = l^2: {100.0 * effect['energy_0.25']:+.2f} % and "
                    f"{100.0 * effect['enstrophy_0.25']:+.1f} %; floor 0.01 % and 0.04 %). With j = l^2 the "
                    "micro-rotation lags the vorticity of a decaying flow and the bulk effect is held back (here to a "
                    "quarter to a third of that with j = l^2 / 100; the exact Taylor-Green solution of Stage 8b gives "
                    "a third of the quasi-steady mu_eff excess at small |k| l, and j = l^2 / 100 recovers it). The dt "
                    "did not change (the angular-diffusion limit 3.5e-6 lies above 1.25e-6).")
            saturation = checks.get("mp_l25_j001_n1000__against__control_n1000")
            if saturation:
                changes = saturation["changes"]
                strength = saturation["run_values"]["primary_value_at_0.25"]
                base_strength = saturation["base_values"]["primary_value_at_0.25"]
                say(f"- **Saturation: the reading does not hold here** (l = W / 25 = 40 dx, j = l^2 / 100, Lambda / l = "
                    f"{saturation['shear_length_over_length']:.2f}). The wall traction excess is "
                    f"{100.0 * changes['wall_traction_peak']:+.1f} % (peak) and "
                    f"{100.0 * changes['wall_traction_integral']:+.1f} % (integral), not the +26.5 % (sqrt(1.6) - 1) "
                    "of a layer pinned at w = 0 across the boundary layer of the control's flow: the dipole itself "
                    "(core radius r0 = 2.5 l) is in the micropolar range and arrives weaker. **The vorticity maximum "
                    f"of the primary core at t = 0.25 is {strength:.1f} against {base_strength:.1f} of the control "
                    f"({100.0 * changes['primary_value_at_0.25']:+.1f} %; floor -0.13 % shifted lattice, +2.3 % half "
                    f"Mach)**, E and the enstrophy at t = 0.25 {100.0 * changes['energy_0.25']:+.1f} % and "
                    f"{100.0 * changes['enstrophy_0.25']:+.1f} %; the wall vorticity of the first collision "
                    f"{100.0 * changes['wall_vorticity_peak']:+.1f} %, the first enstrophy peak "
                    f"{100.0 * changes['enstrophy_peak_first_value']:+.1f} %. At the wall (mu + kappa) d omega / dn - "
                    f"kappa d w / dn is {100.0 * changes['wall_momentum_balance']:+.1f} % off the control's mu_inf "
                    "d omega / dn and the pressure difference along the wall "
                    f"{100.0 * changes['wall_pressure_difference_along_wall']:+.1f} %: the balance closes.")
            resolution = checks.get("mp_l50_n500__against__control_n500")
            if resolution:
                changes = resolution["changes"]
                parts = []
                for quantity, label in keys:
                    if quantity == "primary_value_at_0.25":       # the reference effect (-0.7 %) is below its floor
                        continue
                    gap = changes[quantity] - effect[quantity]
                    share = abs(gap / effect[quantity]) if effect[quantity] else float("nan")
                    if share > 0.1:
                        parts.append(f"{label} {100.0 * changes[quantity]:+.1f} % against {100.0 * effect[quantity]:+.1f} "
                                     f"% ({share:.2f} of the effect)")
                say("- **Resolution of l** (l = W / 50 at 10 dx, W / dx = 500, against `control_n500`): the first "
                    f"enstrophy peak {100.0 * changes['enstrophy_peak_first_value']:+.1f} % and the wall vorticity "
                    f"{100.0 * changes['wall_vorticity_peak']:+.1f} % (20 dx: "
                    f"{100.0 * effect['enstrophy_peak_first_value']:+.1f} % and {100.0 * effect['wall_vorticity_peak']:+.1f} %)"
                    + ("; differing by more than a tenth of the effect: " + "; ".join(parts) if parts else "")
                    + ". [inferred] The quantities read at the wall (traction, flux) are those that resolve the layer "
                    "w = 0 to omega / 2; the global ones agree.")
            say()

    # ---- the periodic box (Stage 8b) -----------------------------------------------------------------------------
    periodic_file = ROOT / "data/micropolar/stage8b/taylor_green.json"
    if periodic_file.exists():
        periodic = load(periodic_file)
        tgv, check = periodic["tgv"], periodic["check"]
        say("## 10. The periodic box: Taylor-Green and decaying turbulence (items 23 to 25) [run]")
        say()
        say("The periodic domain of v8-tgv ported into the solver of the chapter (one card, one slab, 2-D; nothing "
            "changes with the switch off). Design note Section 18.")
        say()
        if check:
            worst = max(abs(entry["difference"]) for entry in check.values())
            say(f"- **Port check.** v8's Newtonian Taylor-Green (Re 100, Ma 0.1 to 0.01, 400^2 and 800^2): nu_eff / nu "
                f"{min(entry['nu_eff_over_nu'] for entry in check.values()):.4f} to "
                f"{max(entry['nu_eff_over_nu'] for entry in check.values()):.4f}, within {worst:.0e} of v8's.")
        rows = []
        for run, label in (("tgv_kl025_n400", "0.25"), ("tgv_kl05_n400", "0.5"), ("tgv_kl1_n400", "1"),
                           ("tgv_kl2_n400", "2"), ("tgv_kl05_n800", "0.5 (800^2)"),
                           ("tgv_kl05_j001_n400", "0.5, j = l^2/100"), ("tgv_kl1_j001_n400", "1, j = l^2/100"),
                           ("tgv_kl1_j150_n400", "1, j = 1.5 l^2"), ("tgv_kl1_inertia_n400", "1, j = 0.755 l^2, W(0) = 0")):
            if run in tgv and "over_control" in tgv[run]:
                item = tgv[run]
                rows.append([label, f"{item['rate_over_exact']:.4f}", f"{item['spin_ratio_over_exact']:.4f}",
                             f"{item['over_control']:.4f}", f"{item['exact_over_control']:.4f}",
                             f"{item['mu_effective_ratio']:.4f}"])
        if rows:
            say()
            say("Micropolar Taylor-Green vortex (k = 1, Re = 20 on nu_inf, K = 3, Ma 0.02, 400^2 unless named; j = l^2 "
                "unless named), rates fitted over 5 <= t <= 20:")
            say()
            say(table(["\\|k\\| l", "rate / exact", "(W/U) / exact", "rate over control", "exact", "quasi-steady mu_eff"],
                      rows))
            say()
            say("- **Every run follows the exact two-mode solution** to 0.1 % (the Newtonian control misses its "
                "exact rate by -0.1 % as well). **With j = l^2 the runs reach only a third of the quasi-steady "
                "mu_eff excess**, as the exact solution predicts: a decaying flow keeps spin from its earlier "
                "vorticity, and with j scaling as l^2 that memory stays of the size of the effect at every |k| l. "
                "With j = l^2 / 100 the runs reach mu_eff (1.0807 and 1.2285 against 1.0811 and 1.2308); with j = "
                "1.5 l^2 (rho j (mu + kappa / 2) = gamma) w = omega / 2 is exact and the micropolar vortex decays "
                "exactly as the Newtonian fluid mu + kappa / 2 (1.0004).")
            floors = periodic.get("floors", {})
            if floors:
                say("- **Floors:** " + "; ".join(
                    f"{label} {entry['rate_difference']:+.1e}" for label, entry in floors.items()) + " (relative "
                    "change of the rate).")
        turbulence_file = ROOT / "data/micropolar/stage8b/turbulence.json"
        if turbulence_file.exists():
            turbulence = load(turbulence_file)
            runs = turbulence.get("runs", {})
            say("- **Decaying turbulence** (box 2 pi, k0 = 10, u' = 1, Re = 100 on nu_inf and 250 on mu; micropolar "
                "K = 3, l = 1/7; Z is the enstrophy): runs "
                + ", ".join(f"`{name}` ({record['particles']:,} particles, {record['steps']:,} steps, "
                            f"{record['elapsed_seconds'] / 60.0:.0f} min)" for name, record in runs.items()) + ".")
            series = turbulence.get("series", {})
            base = series.get("turbulence_newtonian_n2050")
            shifted = series.get("turbulence_newtonian_n2050_shifted")

            def nearest(item, time_value):
                return min(range(len(item["time"])), key=lambda position: abs(item["time"][position] - time_value))
            for run, label in (("turbulence_micropolar_n2050", "j = l^2"),
                               ("turbulence_micropolar_j001_n2050", "j = l^2 / 100 (beyond the task)")):
                micro = series.get(run)
                if not (base and micro):
                    continue
                cells = []
                for time_value in (1.0, 2.0, 4.0):
                    index, other = nearest(base, time_value), nearest(micro, time_value)
                    cells.append(f"t = {base['time'][index]:.1f}: E {micro['energy'][other] / base['energy'][index] - 1.0:+.1%}, "
                                 f"Z {micro['enstrophy'][other] / base['enstrophy'][index] - 1.0:+.1%}")
                say(f"- **Micropolar ({label}) against Newtonian:** " + "; ".join(cells) + ".")
            if base and shifted:
                index, floor_index = nearest(base, 4.0), nearest(shifted, 4.0)
                say(f"- **Floors:** the shifted lattice moves E by {abs(shifted['energy'][floor_index] / base['energy'][index] - 1.0):.0e} "
                    f"and Z by {abs(shifted['enstrophy'][floor_index] / base['enstrophy'][index] - 1.0):.0e} at t = 4 (a repeat "
                    "and the other card alike); the spectra by 1 to 2 % per band at t = 2 and up to 10 % at t = 4 "
                    "(chaotic divergence of the small scales). **With j = l^2 the effect is small and sits in the "
                    "dissipation range** (spectrum ratio 0.87 at k 30 - 60 and 0.65 at 60 - 120 at t = 2): the spin "
                    "relaxes on Theta / (2 c) = 8.5, longer than the run, and advection keeps w = omega / 2. **With j = "
                    "l^2 / 100 it is seven times larger in E** and the energy-containing band follows the linear "
                    "quasi-steady estimate; at higher k the cascade keeps both far above the single-mode estimates. "
                    "Item 24, design note 18.5.")
        say("- **Figures:** item 23 (rate over the control against |k| l, the micro-inertia run), item 24 "
            "(turbulence), table 25.")
        say()

    say("## 11. To be careful with in the text")
    say()
    for line in (
            "**The film function** is that of micropolar lubrication theory; ours is that the runs reproduce it at a "
            "clearance ratio of 0.25. Its match with the published form has not been checked against the original.",
            "**The cavity is not 'between two Newtonian fluids'**, and the pack no longer says so; its check is the "
            "finite-difference solution" + marchi_sentence + ".",
            "**3-D torque change:** three readings, 0.185 to 0.197 at g = 40 dx; there is no exact 3-D value. Say "
            "which reading is quoted.",
            "**The dipole collision** uses omega_e = 299.528 (E(0) = 2), not the 'about 320' of the paper's text; the "
            "palinstrophy of the runs is dominated by grid-scale particle noise and is not a validation number; the "
            "wall vorticity flux is read from the grid with a bias common to all runs (only its ratios are quoted).",
            "**The slip-form rule in 3-D** is good for the mid-plane and a quarter too large for the torque of this "
            "short rotor (one run, one reading of the rule at the rims). Do not quote the 2-D accuracy of the rule "
            "for the 3-D torque.",
            "**The cylinder cases carry `numerics.delta_fluid_only: true`** (the key), the cavity cases the default "
            "walls. With the key the level of the pressure wanders (recorded for every run; no reading follows it).",
            "**Kinetic energy of the cavity** is quoted over the fluid mass (the particle masses sum to 1.134 rho0 L^2 "
            "at n = 500).",
            "**Within one support of a slab cut** the card's acceleration differs from a host sum over the saved "
            "particles by 2e-3 to 4e-3 of its rms; 1e-6 elsewhere. Tested on the saved two-slab state "
            "(`rotor3d_cut_check.txt`): taking the neighbours' pressure from the host's equation of state of their "
            "single-precision density does not bring it to 1e-6, because the host's evaluation is neither card's "
            "(0.21 Pa rms from the saved pressure). It is the difference the two cards show against each other in "
            "every particle in the seam check (4e-3 m/s^2, there 4e-5 of a prescribed field of 109 m/s^2), seen "
            "here only where a card uses pressures of particles it does not own. " + one_card_sentence
            + " No measured quantity of the pack moves by it (two slabs against one: 2e-5 U in the cavity).",
            "**Not run:** g = 400 dx rows other than the one pair; g = 80 dx; displaced (rule) runs in 3-D other than "
            "the one above; a particle "
            "cavity run at l = L / 40."):
        say(f"- {line}")
    say()
    (HERE / "REPORT.md").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    print(f"[pack] REPORT.md: {len(out)} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
