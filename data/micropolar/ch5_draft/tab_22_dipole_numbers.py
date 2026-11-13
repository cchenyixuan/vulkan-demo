"""
Item 22 - table of the dipole-wall collision: the numbers of the control and the change of each micropolar run
against it, beside the noise floor (the control on the lattice shifted by half a spacing; the control at half the
Mach number), and delta* / Lambda.

Run from the repository root:  python data/micropolar/ch5_draft/tab_22_dipole_numbers.py
"""
import sources
import style

SOURCE = sources.DIPOLE
CONTROL = "control_n1000"
MICROPOLAR = (("mp_l100_n1000", r"$\ell = W/100$"), ("mp_l50_n1000", r"$\ell = W/50$"))
FLOORS = (("control_n1000_shifted", "lattice shifted"), ("control_n1000_half_mach", "half Mach number"))
# (key of compare(), label, value of the control from its summary, form of the control value, kind of change)
ROWS = (
    ("enstrophy_peak_first_time", "time of the first enstrophy peak", lambda c: c["enstrophy_peaks"][0][0],
     "{:.4f}", "absolute"),
    ("enstrophy_peak_first_value", "first enstrophy peak", lambda c: c["enstrophy_peaks"][0][1], "{:.1f}", "relative"),
    ("enstrophy_peak_second_time", "time of the second enstrophy peak", lambda c: c["enstrophy_peaks"][1][0],
     "{:.4f}", "absolute"),
    ("enstrophy_peak_second_value", "second enstrophy peak", lambda c: c["enstrophy_peaks"][1][1], "{:.1f}",
     "relative"),
    ("enstrophy_at_base_second_peak", "enstrophy at the time of the control's second peak",
     lambda c: c["enstrophy_at_base_second_peak"], "{:.1f}", "relative"),
    ("energy_0.75", "kinetic energy at t = 0.75", lambda c: c["global"]["0.75"][0], "{:.4f}", "relative"),
    ("closest_approach_distance", "first approach of the primary core to the wall (smallest W - x, t < 0.5)",
     lambda c: c["closest_approach"][1], "{:.4f}", "absolute"),
    ("rebound_distance", "rebound of the primary core (largest W - x between the two approaches)",
     lambda c: c["rebound"][1], "{:.4f}", "absolute"),
    ("second_approach_distance", "second approach of the primary core (smallest W - x, 0.55 < t < 0.85)",
     lambda c: c["second_approach"][1], "{:.4f}", "absolute"),
    ("wall_vorticity_peak", "wall vorticity at the first collision, peak",
     lambda c: c["wall_common"]["vorticity_peak"], "{:.1f}", "relative"),
    ("wall_vorticity_integral_upper", "wall vorticity at the first collision, integral over 0 < y < W",
     lambda c: c["wall_common"]["vorticity_integral_upper"], "{:.3f}", "relative"),
    ("wall_traction_peak", "wall shear traction (mu + kappa) omega_w, peak (control: mu_inf omega_w)",
     lambda c: dipole_far_viscosity() * c["wall_common"]["vorticity_peak"], "{:.3f}", "relative"),
    ("wall_traction_integral", "wall shear traction, integral over 0 < y < W",
     lambda c: dipole_far_viscosity() * c["wall_common"]["vorticity_integral_upper"], "{:.4f}", "relative"),
    ("wall_flux_integral_upper", "wall vorticity flux d omega / ds at the first collision, integral over 0 < y < W",
     lambda c: c["wall_common"]["flux_integral_upper"], "{:.1f}", "relative"),
    ("wall_momentum_balance", "momentum balance at the wall: (mu + kappa) d omega / dn - kappa d w / dn, integral "
     "(control: mu_inf d omega / dn)", lambda c: dipole_far_viscosity() * c["wall_common"]["flux_integral_upper"],
     "{:.2f}", "relative"),
    ("wall_pressure_difference_along_wall", "pressure difference along the wall, p(W, 0) - p(W, W), over rho",
     lambda c: c["wall_common"]["pressure_difference_along_wall"], "{:.2f}", "relative"),
)


def dipole_far_viscosity() -> float:
    from experiment.validation import micropolar_dipole
    return micropolar_dipole.FAR_VISCOSITY


def change(value, kind, missing="-"):
    if value is None or value != value:
        return missing
    return f"{value:+.4f}" if kind == "absolute" else f"{100.0 * value:+.2f} %"


def main() -> int:
    analysis = style.load_json(SOURCE["analysis"])
    runs = analysis["runs"]
    if CONTROL not in runs:
        print("[pack] item 22: no control yet")
        return 0
    control = runs[CONTROL]
    comparisons, floors = analysis.get("comparisons", {}), analysis.get("floors", {})
    present = [(name, label) for name, label in MICROPOLAR if name in comparisons]
    floor_present = [(name, label) for name, label in FLOORS if name in floors]
    header = (["quantity", "control"] + [f"change, {label}" for _name, label in present]
              + [f"noise floor: {label}" for _name, label in floor_present])
    tex_header = ([r"quantity", r"control"] + [f"change, {label}" for _name, label in present]
                  + [f"floor: {label}" for _name, label in floor_present])
    rows = []
    for key, label, control_value, form, kind in ROWS:
        try:
            value = form.format(control_value(control))
        except (KeyError, IndexError, TypeError):
            value = "-"
        missing = "no maximum" if key.startswith("enstrophy_peak_second") else "-"
        rows.append([label, value] + [change(comparisons[name].get(key), kind, missing) for name, _label in present]
                    + [change(floors[name].get(key), kind) for name, _label in floor_present])
    shear = runs.get("_shear_length")
    if shear:
        rows.append(["delta* / Lambda (Lambda from the control at the first collision)",
                     f"Lambda = {shear['value']:.4f}"]
                    + [f"{shear['rules'][name]['displacement_over_shear_length']:.3f}" for name, _label in present]
                    + ["" for _floor in floor_present])
    style.write_table("tab_22_dipole_numbers", header, rows, tex_header=[style.tex_escape(cell) if "$" not in cell
                                                                         else cell for cell in tex_header],
                      note="changes against the control: relative for values, absolute for times and distances")
    caption = (
        "The dipole-wall collision at Re = 625 on nu_inf (dx = W / 1000): numbers of the Newtonian control and the "
        "change of each micropolar run (K = 3, w = 0 on every wall) against it; times and distances as absolute "
        "differences (units W / U and W), the rest relative. Noise floor: the same numbers of the control with the "
        "dipole moved by half a spacing along x (lattice shifted against the flow; the dipole is then dx / 2 nearer "
        "the wall) and of the control at half the Mach number. A micropolar change below the floor is not "
        "established. Wall vorticity and its flux on x = W at the time of the control's first enstrophy peak "
        "(vorticity from a fit of the tangential velocity of the particles within 4 dx of the wall; flux from the "
        "grid of one spacing; the absolute flux reads about 1.4 times the pressure difference along the wall in every "
        "run, a bias of the reading common to all). Traction and momentum balance in kinematic units (over rho); "
        "w = 0 on the wall, so the traction of the micropolar fluid is (mu + kappa) omega_w and the tangential "
        "momentum balance at the wall reads dp/ds = (mu + kappa) d omega / dn - kappa d w / dn (dw/dn from the same "
        "particle fit). Lambda = U_ref / |omega_w| of the control, U_ref the largest tangential speed at "
        "0.03 W from the wall; delta* = kappa l / (2 (mu + kappa)) = 0.375 l.")
    style.record("22", "Dipole-wall collision: numbers against the control with the noise floor", SOURCE["status"],
                 caption, ["tab_22_dipole_numbers.csv", "tab_22_dipole_numbers.tex"],
                 [SOURCE["analysis"]], {"rows": len(rows)}, SOURCE["tag"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
