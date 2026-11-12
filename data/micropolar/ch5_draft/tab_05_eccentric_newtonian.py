"""
Item 05 - table: the Newtonian flow between eccentric cylinders against the exact Stokes values (load, direction,
torque by both readings), with the spread over three lattices.

Run from the repository root:  python data/micropolar/ch5_draft/tab_05_eccentric_newtonian.py
"""
import sources
import style

SOURCES = [sources.ECCENTRIC] + ([sources.ECCENTRIC_BENCHMARK_STAGE5]
                                 if sources.ECCENTRIC is not sources.ECCENTRIC_BENCHMARK_STAGE5 else [])
SETTINGS = {"nd": "density diffusion off", "fd": "diffusion between fluid pairs (rc2)", "thesis": "thesis walls",
            "ew": "eos walls"}
LATTICE = {"": "as generated", "_shifted": "shifted", "_rotated": "rotated"}


def percent(ratio: float) -> str:
    return f"{100.0 * (ratio - 1.0):+.2f}"


def main() -> int:
    header = ["run", "setting", "eps", "Re", "g/dx", "Ma", "lattice", "load/exact - 1 (%)", "direction (deg)",
              "torque, circle form: rotor (%)", "stator (%)", "torque, wall form: rotor (%)", "stator (%)",
              "torque, 2nd - 1st half (%)", "load, 2nd - 1st half (%)", "density drift", "pressure drift",
              "git describe"]
    tex_header = ["run", "setting", r"$\varepsilon$", r"$\mathit{Re}$", rf"${style.GAP}/{style.SPACING}$",
                  r"$\mathit{Ma}$", "lattice", r"$F/F_{\mathrm{exact}}-1$ (\%)", r"direction ($^\circ$)",
                  r"$T_{\mathrm{circle}}$: rotor (\%)", r"stator (\%)", r"$T_{\mathrm{wall}}$: rotor (\%)",
                  r"stator (\%)"]
    plain, tex, used, numbers = [], [], [], {}
    spreads = []
    for source in SOURCES:
        analysis = style.load_json(source["analysis"])
        for name, item in analysis["runs"].items():
            if item["micropolar"] or "exact" not in item:
                continue
            lattice = next((label for suffix, label in LATTICE.items() if suffix and name.endswith(suffix)),
                           LATTICE[""])
            cells = [name, SETTINGS.get(item["candidate"], item["candidate"]), f"{item['eccentricity_ratio']:g}",
                     f"{item['reynolds']:g}", f"{item['gap_in_spacings']:g}", f"{item['mach']:g}", lattice,
                     percent(item["load_over_exact"]), f"{item['force_angle_degrees']:.1f}",
                     percent(item["rotor_torque_circle_over_exact"]), percent(item["stator_torque_circle_over_exact"]),
                     percent(item["rotor_torque_wall_over_exact"]), percent(item["stator_torque_wall_over_exact"]),
                     f"{100.0 * item['torque_change_between_halves']:+.3f}",
                     f"{100.0 * item['load_change_between_halves']:+.3f}",
                     f"{item['density_drift']:+.1e}" if "density_drift" in item else "-",
                     f"{item['pressure_drift']:+.1e}" if "pressure_drift" in item else "-",
                     item["revision"]["describe"]]
            plain.append(cells)
            tex.append([style.texttt(name)] + cells[1:13])
            used.append(name)
            numbers[name] = {"load_over_exact": item["load_over_exact"], "direction": item["force_angle_degrees"],
                             "rotor_torque_circle_over_exact": item["rotor_torque_circle_over_exact"]}
        for label, floor in analysis.get("floors", {}).items():
            first = analysis["runs"][floor["runs"][0]]
            spreads.append((label, first, floor))
    for label, first, floor in spreads:
        cells = [f"spread of {label} over three lattices", SETTINGS.get(first["candidate"], first["candidate"]),
                 f"{first['eccentricity_ratio']:g}", f"{first['reynolds']:g}", f"{first['gap_in_spacings']:g}",
                 f"{first['mach']:g}", "max - min", f"{100.0 * floor['load_spread']:.2f}",
                 f"{floor['force_angle_spread_degrees']:.2f}", f"{100.0 * floor['rotor_torque_circle_spread']:.2f}",
                 f"{100.0 * floor['stator_torque_circle_spread']:.2f}",
                 f"{100.0 * floor['rotor_torque_wall_spread']:.2f}",
                 f"{100.0 * floor['stator_torque_wall_spread']:.2f}", "", "", "", "", ""]
        plain.append(cells)
        tex.append("rule")
        tex.append([style.tex_escape(cells[0])] + cells[1:13])
        numbers[f"spread_{label}"] = {key: floor[key] for key in ("load_spread", "force_angle_spread_degrees",
                                                                  "rotor_torque_circle_spread",
                                                                  "rotor_torque_wall_spread")}
    style.write_table("tab_05_eccentric_newtonian", header, plain, tex_header, "ll" + "r" * 4 + "l" + "r" * 6, tex)

    loads = [abs(entry["load_over_exact"] - 1.0) for key, entry in numbers.items() if "load_over_exact" in entry]
    circles = [abs(entry["rotor_torque_circle_over_exact"] - 1.0) for entry in numbers.values()
               if "rotor_torque_circle_over_exact" in entry]
    status = SOURCES[0]["status"]
    caption = (
        "Newtonian flow between eccentric rotating cylinders (R1 = 4 g, R2 = 5 g, rotor axis displaced by eps g "
        "towards +x, rotor at Omega, stator at rest) against the exact Stokes solution: load on the rotor "
        "(magnitude; the exact force is along +y, 90 degrees from the line of centres), its direction, and the "
        "rotor and stator torque by the circle form (flux of angular momentum across a circle at distance h from "
        "the wall) and by the wall form (pair sums over the wall particles), each as reading / exact - 1. Mean over "
        "the saved states of the last quarter of the run; the two 'half' columns compare the two halves of that "
        "quarter; drifts: change of the fluid's mean density (over rho0) and mean pressure (over rho U^2) between "
        "the last two eighths of the run. Re = rho U g / mu on the one viscosity of the Newtonian fluid. "
        f"Settings: {'; '.join(source['note'] for source in SOURCES)}. "
        f"Load within {100.0 * max(loads):.2f} % of exact in every row, rotor torque by the circle form within "
        f"{100.0 * max(circles):.2f} %; the departure of the direction from 90 degrees is inertia at Re = 10 and "
        "compressibility at Re = 1 (design note 14.3.2). Exact values for mu = 10 Pa s: load 1,016.0 and 2,145.4 "
        "N/m, rotor torque 158.02 and 223.62 N at eps = 0.3 and 0.6.")
    style.record("05", "Table: Newtonian eccentric flow against the exact Stokes values", status, caption,
                 ["tab_05_eccentric_newtonian.csv", "tab_05_eccentric_newtonian.tex"],
                 [source["analysis"] for source in SOURCES] + used, numbers,
                 "; ".join(source["tag"] for source in SOURCES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
