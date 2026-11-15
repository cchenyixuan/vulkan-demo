"""
Item 29 - table of the open runs of the dipole-wall collision (Stage 8, 14 Nov task, Part B), in the format of item
26: (B1) l = W / 100 at W / dx = 2000 (to t = 0.45) beside the same l at W / dx = 1000; (B2) Re = 1250 with l = W / 100
and W / 50 and the floors of its control; (B3) the oblique collision (30 degrees, Re = 625) with l = W / 50 and its
floors. Each change against the run's own control.

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


def main() -> int:
    analysis = style.load_json(SOURCE["analysis"])
    checks, floors = analysis.get("checks", {}), analysis.get("floors", {})
    present = [(key, plain, tex) for key, plain, tex in COLUMNS if key in checks]
    if not present:
        print("[pack] item 29: the open runs are not analysed yet")
        return 0
    header = ["quantity"] + [plain for _key, plain, _tex in present] + [label for _name, label in FLOORS]
    tex_header = ["quantity"] + [tex for _key, _plain, tex in present] + [label for _name, label in FLOORS]
    rows = []
    for key, label, kind in ROWS:
        missing = "no maximum" if key.startswith("enstrophy_peak_second") else "-"
        row = [label] + [text(checks[column]["changes"].get(key), kind, missing) for column, _plain, _tex in present]
        row += [text(floors.get(name, {}).get(key), kind) for name, _label in FLOORS]
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
    numbers = {column: checks[column]["changes"] for column, _plain, _tex in present}
    caption = (
        "The open runs of the dipole-wall collision (K = 3, j = l^2, w = 0 on every wall, w = omega / 2 at t = 0), "
        "with the definitions of tables 22 and 26 and each change against the run's own control. (B1) l = W / 100 at "
        "W / dx = 2000 (l = 20 dx; the runs end at t = 0.45, past the first enstrophy peak) beside the same l at "
        "W / dx = 1000 (l = 10 dx, table 22); no floor runs were made at W / dx = 2000: the floors beside the rows "
        "are those of W / dx = 1000 at Re = 625. (B2) Re = 1250 at W / dx = 1000 with l = W / 100 and W / 50 and the "
        "floors of its control (the lattice shifted by dx / 2 along x, half the Mach number). (B3) the oblique "
        "collision of Clercx & Bruneau (2006), 30 degrees to the wall normal, Re = 625, l = W / 50, with the floors "
        "of its control; the wall rows read the wall x = W. The primary core is tracked as the strongest extremum "
        "within reach; Lambda = U_ref / |omega_w| of the run's control at its first enstrophy peak.")
    style.record("29", "Dipole-wall collision: the open runs (resolution of l = W/100, Re = 1250, oblique collision)",
                 SOURCE["status"], caption, ["tab_29_dipole_open_runs.csv", "tab_29_dipole_open_runs.tex"],
                 [SOURCE["analysis"]], numbers, SOURCE["tag"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
