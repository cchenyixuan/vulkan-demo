"""
Item 26 - table of the three checks of the dipole-wall collision (Stage 8 checks): the micro-inertia
(l = W / 50 with j = l^2 / 100), the saturation (l = W / 25, j = l^2 / 100) and the resolution of l (l = W / 50 at
10 dx, W / dx = 500), each as the rows of table 22 against its control, beside the reference run (l = W / 50 at
20 dx, j = l^2) and the noise floor of table 22.

Run from the repository root:  python data/micropolar/ch5_draft/tab_26_dipole_checks.py
"""
import sources
import style

SOURCE = sources.DIPOLE
COLUMNS = (
    ("mp_l50_j001_n1000__against__control_n1000", "l = W/50, j = l^2/100, against the control",
     r"$\ell = W/50$, $j = \ell^2/100$"),
    ("mp_l50_j001_n1000__against__mp_l50_n1000", "the same against l = W/50 with j = l^2",
     r"the same against $j = \ell^2$"),
    ("mp_l25_j001_n1000__against__control_n1000", "l = W/25, j = l^2/100, against the control",
     r"$\ell = W/25$, $j = \ell^2/100$"),
    ("mp_l50_n500__against__control_n500", "l = W/50 at 10 dx (W/dx = 500), against control_n500",
     r"$\ell = W/50$ at $10\,\Delta x$"),
    ("mp_l50_n1000__against__control_n1000", "reference: l = W/50 at 20 dx, j = l^2 (table 22)",
     r"reference: $\ell = W/50$ at $20\,\Delta x$"),
)
FLOORS = (("control_n1000_shifted", "noise floor: lattice shifted"), ("control_n1000_half_mach", "noise floor: half Mach"))
ROWS = (
    ("enstrophy_peak_first_value", "first enstrophy peak", "relative"),
    ("enstrophy_peak_first_time", "time of the first enstrophy peak", "absolute"),
    ("enstrophy_peak_second_value", "second enstrophy peak", "relative"),
    ("enstrophy_at_base_second_peak", "enstrophy at the time of the control's second peak", "relative"),
    ("energy_0.75", "kinetic energy at t = 0.75", "relative"),
    ("energy_0.25", "kinetic energy at t = 0.25 (before the collision)", "relative"),
    ("enstrophy_0.25", "enstrophy at t = 0.25 (before the collision)", "relative"),
    ("primary_value_at_0.25", "vorticity maximum of the primary core at t = 0.25 (before the collision)", "relative"),
    ("closest_approach_distance", "first approach of the primary core to the wall (W - x)", "absolute"),
    ("rebound_distance", "rebound of the primary core (largest W - x between the approaches)", "absolute"),
    ("second_approach_distance", "second approach of the primary core (W - x)", "absolute"),
    ("wall_vorticity_peak", "wall vorticity at the first collision, peak", "relative"),
    ("wall_vorticity_integral_upper", "wall vorticity at the first collision, integral over 0 < y < W", "relative"),
    ("wall_traction_peak", "wall shear traction, peak", "relative"),
    ("wall_traction_integral", "wall shear traction, integral over 0 < y < W", "relative"),
    ("wall_flux_integral_upper", "wall vorticity flux, integral over 0 < y < W", "relative"),
    ("wall_momentum_balance", "momentum balance at the wall, integral", "relative"),
    ("wall_pressure_difference_along_wall", "pressure difference along the wall", "relative"),
)


def text(value, kind, missing="-"):
    if value is None or value != value:
        return missing
    return f"{value:+.4f}" if kind == "absolute" else f"{100.0 * value:+.2f} %"


def main() -> int:
    analysis = style.load_json(SOURCE["analysis"])
    checks, floors = analysis.get("checks", {}), analysis.get("floors", {})
    present = [(key, plain, tex) for key, plain, tex in COLUMNS if key in checks]
    if len(present) < 2:
        print("[pack] item 26: the check runs are not analysed yet")
        return 0
    header = ["quantity"] + [plain for _key, plain, _tex in present] + [label for _name, label in FLOORS]
    tex_header = ["quantity"] + [tex for _key, _plain, tex in present] + [label for _name, label in FLOORS]
    rows = []
    for key, label, kind in ROWS:
        missing = "no maximum" if key.startswith("enstrophy_peak_second") else "-"
        row = [label] + [text(checks[column]["changes"].get(key), kind, missing) for column, _plain, _tex in present]
        floor_cells = []
        for name, _label in FLOORS:
            values = floors.get(name, {})
            if key == "primary_value_at_0.25":
                runs = analysis["runs"]
                value = (runs[name]["primary_value_at_0.25"] / runs["control_n1000"]["primary_value_at_0.25"] - 1.0
                         if name in runs and "primary_value_at_0.25" in runs[name] else None)
            else:
                value = values.get(key)
            floor_cells.append(text(value, kind))
        rows.append(row + floor_cells)
    rows.append(["delta* / Lambda (Lambda of the run's control)"]
                + [f"{checks[column]['displacement_over_shear_length']:.2f}" if "against__control" in column else "-"
                   for column, _plain, _tex in present] + ["", ""])
    rows.append(["Lambda / l"] + [f"{checks[column]['shear_length_over_length']:.3f}" if "against__control" in column
                                  else "-" for column, _plain, _tex in present] + ["", ""])
    style.write_table("tab_26_dipole_checks", header, rows,
                      tex_header=[cell if "$" in cell else style.tex_escape(cell) for cell in tex_header],
                      note="changes against the named base: relative for values, absolute for times and distances")
    numbers = {column: checks[column]["changes"] for column, _plain, _tex in present}
    caption = (
        "Three checks of the dipole-wall collision of table 22 (Re = 625 on nu_inf, K = 3, w = 0 on every wall), "
        "with the definitions and the noise floor of table 22 (relative changes, absolute ones for times and "
        "distances in W / U and W). (1) The micro-inertia: l = W / 50 with j = l^2 / 100 (the relaxation time "
        "rho j / (2 kappa) of w 0.001 W / U instead of 0.10), against the control and against the run with j = l^2. "
        "(2) The saturation: l = W / 25 (40 dx), j = l^2 / 100, Lambda / l = 0.13; the strength of its primary core "
        "before the collision (t = 0.25) shows the bulk effect on the cores (r0 = 2.5 l). (3) The resolution of l: "
        "l = W / 50 at W / dx = 500 (l = 10 dx) against the control at the same resolution, beside the same l at "
        "20 dx (the reference column, the run of table 22). The kinetic energy and the enstrophy at t = 0.25 show the "
        "effect in the bulk before the collision. The primary core is tracked as the strongest extremum within reach "
        "in every column (the reference too: its core rows can differ from table 22, which follows the nearest one); "
        "its vorticity maximum moves by 2.3 % at half the Mach number, a bias common to runs of the same Mach number. "
        "All runs on the solver trees of thesis-ch5-rc2, the same dt per resolution (the acoustic step; the "
        "angular-diffusion limit of j = l^2 / 100 is 3.5e-6 W / U, above the step 1.25e-6).")
    style.record("26", "Dipole-wall collision: the checks of micro-inertia, saturation and resolution",
                 SOURCE["status"], caption, ["tab_26_dipole_checks.csv", "tab_26_dipole_checks.tex"],
                 [SOURCE["analysis"]], numbers, SOURCE["tag"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
