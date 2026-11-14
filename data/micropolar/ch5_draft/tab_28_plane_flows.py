"""
Item 28 - table of the steady plane Couette and Poiseuille runs in the channel periodic in x (Stage 8c): the wall
stress (Couette) in three readings and the flow rate (Poiseuille) over the exact value, the micropolar over the
control value against the exact ratio (eq. 5.5 over (mu + kappa / 2) U / H; eq. 2.26), the rms errors of u and w,
and the floors (a repeat and half the Mach number).

Run from the repository root:  python data/micropolar/ch5_draft/tab_28_plane_flows.py
"""
import sources
import style

SOURCE = sources.CHANNEL
CONFIGURATIONS = ((1, 10), (2, 10), (5, 10), (10, 10), (2, 20), (5, 20))


def number(value, form="{:.4f}"):
    return "-" if value is None or value != value else form.format(value)


def main() -> int:
    if not style.exists(SOURCE["analysis"]):
        print("[pack] item 28: no channel analysis yet")
        return 0
    analysis = style.load_json(SOURCE["analysis"])
    steady, floors = analysis.get("steady", {}), analysis.get("steady_floors", {})
    header = ["flow", "m", "l / dx", "H / dx", "fluid", "fit (lower / upper)", "wall form (lower / upper)", "profile",
              "Q / exact", "u rms / rms", "w rms / rms", "micropolar / control: fit", "wall form", "profile or Q",
              "exact ratio", "floor: repeat", "floor: half Mach"]
    rows = []
    for flow in ("couette", "poiseuille"):
        for m, length in CONFIGURATIONS:
            for micropolar in (False, True):
                name = f"{flow}_m{m}_l{length}" + ("" if micropolar else "_control")
                if name not in steady:
                    continue
                entry = steady[name]
                floor = floors.get(name, {})

                def floor_text(variant):
                    values = floor.get(variant, {})
                    keys = ("profile_over_control", "flow_rate_over_control") if micropolar else ("profile", "flow_rate")
                    chosen = [abs(values[key]) for key in keys if key in values]
                    return number(max(chosen) if chosen else None, "{:.1e}")
                if flow == "couette":
                    cells = [f"{number(entry['fit_lower'])} / {number(entry['fit_upper'])}",
                             f"{number(entry.get('pairs_lower'))} / {number(entry.get('pairs_upper'))}",
                             number(entry["profile"]), "-"]
                    ratios = [number(entry.get("fit_lower_over_control")), number(entry.get("pairs_lower_over_control")),
                              number(entry.get("profile_over_control"))]
                else:
                    cells = ["-", "-", "-", number(entry["flow_rate"])]
                    ratios = ["-", "-", number(entry.get("flow_rate_over_control"))]
                rows.append([flow, str(m), str(length), str(2 * m * length), "micropolar" if micropolar else "control",
                             *cells, number(entry["u_rms_error"], "{:.1e}"), number(entry["w_rms_error"], "{:.1e}"),
                             *ratios, number(entry["exact_ratio"]) if micropolar else "-", floor_text("repeat"),
                             floor_text("ma0p01")])
    if not rows:
        print("[pack] item 28: no steady runs analysed yet")
        return 0
    tex_header = ["flow", "$m$", r"$\ell/\Delta x$", r"$H/\Delta x$", "fluid", "fit", "wall form", "profile",
                  r"$Q/Q_\mathrm{exact}$", r"$u$ rms", r"$w$ rms", "fit", "wall form", "profile, $Q$", "exact",
                  "repeat", r"$\frac{1}{2}$ Ma"]
    style.write_table("tab_28_plane_flows", header, rows, tex_header=tex_header,
                      tex_rows=[[style.tex_escape(cell) for cell in row] for row in rows],
                      note="stress and Q over the exact value; micropolar over control against the exact ratio")
    caption = (
        "Steady plane flows between two no-spin walls in a channel periodic in x (length 100 dx, h = 5 dx, Ma 0.02, "
        "Re = 10 on nu_inf; K = 3, j = l^2; m = H / (2 l)). Couette (the lower wall slides at U): the wall stress "
        "over eq. 5.5 in three readings - the fit u = a s + b s^2 to the fluid within 4 dx of the wall (the reading "
        "of the dipole runs), the wall form (the force of the wall on the fluid from the solver's pair terms, the "
        "reading of the rotor torque, on the states saved at the end; the minimum image across the periodic seam) "
        "and the profile (the amplitude of the exact profile fitted to the whole binned profile, the reading of items "
        "01 to 03). Poiseuille (a uniform body force): the flow rate over eq. 2.26 times the Newtonian rate. The "
        "rms of u - u_exact and w - w_exact over the rms of the exact profile (bins of dx, the last twelfth of the "
        "run). Micropolar over control: the ratio of the two runs' values against the exact ratio (eq. 5.5 over "
        "(mu + kappa / 2) U / H; Q / Q_N of eq. 2.26). Floors: the change of the profile ratio (Couette) or of Q "
        "(Poiseuille) in a repeated run and at half the Mach number.")
    style.record("28", "Table of the steady plane Couette and Poiseuille flows against their exact solutions",
                 SOURCE["status"], caption, ["tab_28_plane_flows.csv", "tab_28_plane_flows.tex"], [SOURCE["analysis"]],
                 {"rows": len(rows)}, SOURCE["tag"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
