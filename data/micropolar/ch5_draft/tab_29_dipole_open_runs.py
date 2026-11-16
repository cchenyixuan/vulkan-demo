"""
Item 29 - tables of the open runs of the dipole-wall collision (Stage 8, 14 Nov task, Part B), in the format of item
26: (B1) l = W / 100 at W / dx = 2000 (to t = 0.45) beside the same l at W / dx = 1000; (B2) Re = 1250 with l = W / 100
and W / 50 and the floors of its control; (B3) the oblique collision (30 degrees, Re = 625) with l = W / 50 and its
floors. Each change against the run's own control (tab_29_dipole_open_runs); the controls of B1 to B3 against the
paper (tab_29_dipole_open_validation).

Run from the repository root:  python data/micropolar/ch5_draft/tab_29_dipole_open_runs.py
"""
import sources
import style
from tab_26_dipole_checks import ROWS, text

SOURCE = sources.DIPOLE
COLUMNS = (
    ("mp_l100_n2000__against__control_n2000", "l = W/100, W/dx = 2000 (t <= 0.45)",
     r"$\ell = W/100$, $W/\Delta x = 2000$"),
    ("mp_l100_n1000__against__control_n1000", "l = W/100, W/dx = 1000 (table 22)",
     r"$\ell = W/100$, $W/\Delta x = 1000$"),
    ("mp_l100_re1250_n1000__against__control_re1250_n1000", "Re 1250, l = W/100", r"Re 1250, $\ell = W/100$"),
    ("mp_l50_re1250_n1000__against__control_re1250_n1000", "Re 1250, l = W/50", r"Re 1250, $\ell = W/50$"),
    ("control_re1250_n1000_shifted__against__control_re1250_n1000", "Re 1250 floor: lattice shifted",
     "Re 1250 floor: shifted"),
    ("control_re1250_n1000_half_mach__against__control_re1250_n1000", "Re 1250 floor: half Mach",
     "Re 1250 floor: half Mach"),
    ("oblique_mp_l50_n1000__against__oblique_control_n1000", "oblique, l = W/50", r"oblique, $\ell = W/50$"),
    ("oblique_control_n1000_shifted__against__oblique_control_n1000", "oblique floor: lattice shifted",
     "oblique floor: shifted"),
    ("oblique_control_n1000_half_mach__against__oblique_control_n1000", "oblique floor: half Mach",
     "oblique floor: half Mach"),
)
FLOORS = (("control_n1000_shifted", "Re 625 W/1000 floor: shifted"), ("control_n1000_half_mach",
                                                                       "Re 625 W/1000 floor: half Mach"))
# the rows of item 26 and, for the oblique collision (two wall layers of opposite sign above y = 0, whose signed
# integrals cancel), the integrals of each sign over the whole wall and the negative primary core
EXTRA_ROWS = (
    ("wall_vorticity_integral_negative", "wall vorticity, integral of the negative part (whole wall)", "relative"),
    ("wall_vorticity_integral_positive", "wall vorticity, integral of the positive part (whole wall)", "relative"),
    ("wall_traction_integral_negative", "wall shear traction, integral of the negative part", "relative"),
    ("wall_traction_integral_positive", "wall shear traction, integral of the positive part", "relative"),
    ("negative_primary_value_at_0.6", "negative primary core, vorticity minimum at t = 0.6 (oblique)", "relative"),
)
# the controls against the paper: (title, control, shifted floor, half-Mach floor)
VALIDATION_BLOCKS = (
    ("Re = 625, normal collision, W/dx = 2000 (to t = 0.45)", "control_n2000", None, None),
    ("Re = 1250, normal collision, W/dx = 1000 (Tables 2 to 5 of the paper)", "control_re1250_n1000",
     "control_re1250_n1000_shifted", "control_re1250_n1000_half_mach"),
    ("Re = 625, oblique collision (30 degrees), W/dx = 1000 (Tables 6 to 9)", "oblique_control_n1000",
     "oblique_control_n1000_shifted", "oblique_control_n1000_half_mach"),
)


def validation_cell(entry: dict | None) -> str:
    if entry is None:
        return "-"
    if entry["kind"] == "absolute":
        return f"{entry['run']:.4f} ({entry['difference']:+.4f})"
    return f"{entry['run']:.5g} ({100.0 * entry['difference']:+.1f} %)"


def reference(value) -> str:
    return "-" if value is None or value != value else f"{value:.5g}"


def write_validation(analysis: dict) -> list:
    validation = analysis.get("validation", {})
    header = ["quantity", "SM", "FD", "control", "floor: lattice shifted", "floor: half Mach"]
    rows, tex_rows, written = [], [], []
    for title, control, shifted, half_mach in VALIDATION_BLOCKS:
        if control not in validation:
            continue
        written.append(control)
        rows.append([title, "", "", "", "", ""])
        if tex_rows:
            tex_rows.append("rule")
        tex_rows.append([r"\multicolumn{6}{l}{" + style.tex_escape(title) + "}"])
        floors = {name: {entry["quantity"]: entry for entry in validation.get(name, [])}
                  for name in (shifted, half_mach) if name}
        for entry in validation[control]:
            cells = [entry["quantity"], reference(entry["SM"]), reference(entry["FD"]), validation_cell(entry)]
            for name in (shifted, half_mach):
                cells.append(validation_cell(floors.get(name, {}).get(entry["quantity"])) if name else "-")
            rows.append(cells)
            tex_rows.append([style.tex_escape(cell) for cell in cells])
    if rows:
        style.write_table("tab_29_dipole_open_validation", header, rows, tex_rows=tex_rows,
                          note="the run's value and its difference to the spectral value (SM): relative for values, "
                               "absolute for times and positions")
    return written


def main() -> int:
    analysis = style.load_json(SOURCE["analysis"])
    checks, floors = analysis.get("checks", {}), analysis.get("floors", {})
    present = [(key, plain, tex) for key, plain, tex in COLUMNS if key in checks]
    if not present:
        print("[pack] item 29: the open runs are not analysed yet")
        return 0
    header = ["quantity"] + [plain for _key, plain, _tex in present] + [label for _name, label in FLOORS]
    tex_header = ["quantity"] + [tex for _key, _plain, tex in present] + [label for _name, label in FLOORS]
    runs = analysis.get("runs", {})

    def missing_text(column, key):
        # "no maximum" only for a run that reached the second collision (the floors end at t = 0.40, B1 at 0.45)
        reached = min(runs.get(checks[column][name], {}).get("end_time", 0.0) for name in ("run", "base")) >= 0.72
        return "no maximum" if key.startswith("enstrophy_peak_second") and reached else "-"

    def floor_value(name, key):
        # the floor runs have the control's viscosity: their traction changes as their wall vorticity
        values = floors.get(name, {})
        if key.startswith("wall_traction_integral_"):
            key = key.replace("wall_traction_integral_", "wall_vorticity_integral_")
        return values.get(key)

    rows = []
    for key, label, kind in ROWS + EXTRA_ROWS:
        row = [label] + [text(checks[column]["changes"].get(key), kind, missing_text(column, key))
                         for column, _plain, _tex in present]
        row += [text(floor_value(name, key), kind) for name, _label in FLOORS]
        rows.append(row)
    rows.append(["delta* / Lambda (Lambda of the run's control)"]
                + [f"{checks[column]['displacement_over_shear_length']:.2f}"
                   if checks[column]["displacement_over_shear_length"] == checks[column]["displacement_over_shear_length"]
                   else "-" for column, _plain, _tex in present] + ["", ""])
    rows.append(["Lambda / l"] + [f"{checks[column]['shear_length_over_length']:.3f}"
                                  if checks[column]["shear_length_over_length"] == checks[column]["shear_length_over_length"]
                                  else "-" for column, _plain, _tex in present] + ["", ""])
    rows.append(["Lambda of the control (W)"] + [f"{checks[column]['shear_length']:.5f}"
                                                 for column, _plain, _tex in present] + ["", ""])
    style.write_table("tab_29_dipole_open_runs", header, rows,
                      tex_header=[cell if "$" in cell else style.tex_escape(cell) for cell in tex_header],
                      note="changes against the run's own control: relative for values, absolute for times and distances")
    validated = write_validation(analysis)
    numbers = {column: checks[column]["changes"] for column, _plain, _tex in present}
    numbers["validation"] = {name: {entry["quantity"]: [entry["run"], entry["difference"]]
                                    for entry in analysis["validation"][name]} for name in validated}
    caption = (
        "The open runs of the dipole-wall collision (K = 3, j = l^2, w = 0 on every wall, w = omega / 2 at t = 0), "
        "with the definitions of tables 22 and 26 and each change against the run's own control. (B1) l = W / 100 at "
        "W / dx = 2000 (l = 20 dx; the runs end at t = 0.45, past the first enstrophy peak) beside the same l at "
        "W / dx = 1000 (l = 10 dx, table 22); no floor runs were made at W / dx = 2000: the floors beside the rows "
        "are those of W / dx = 1000 at Re = 625. (B2) Re = 1250 at W / dx = 1000 with l = W / 100 and W / 50 and the "
        "floors of its control (the lattice shifted by dx / 2 along x, half the Mach number); the runs end at "
        "t = 0.75, the floors at 0.40. (B3) the oblique collision of Clercx & Bruneau (2006), 30 degrees to the wall "
        "normal, Re = 625, l = W / 50, with the floors of its control (same end times); the wall rows read the wall "
        "x = W, where both cores of the oblique dipole arrive above y = 0: their wall layers have opposite signs, so "
        "the signed integrals over 0 < y < W, the vorticity flux, the momentum balance and the pressure difference "
        "along the wall are left out for B3 and the integrals of each sign over the whole wall given instead. The "
        "primary core is tracked as the strongest extremum within reach; Lambda = U_ref / |omega_w| of the run's "
        "control at its first enstrophy peak. Second table: the controls of B1 to B3 and their floors against the "
        "spectral (SM) and finite-difference (FD) values of the paper (Tables 2 to 9), with the difference to SM.")
    style.record("29", "Dipole-wall collision: the open runs (resolution of l = W/100, Re = 1250, oblique collision)",
                 SOURCE["status"], caption, ["tab_29_dipole_open_runs.csv", "tab_29_dipole_open_runs.tex",
                                             "tab_29_dipole_open_validation.csv", "tab_29_dipole_open_validation.tex"],
                 [SOURCE["analysis"]], numbers, SOURCE["tag"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
