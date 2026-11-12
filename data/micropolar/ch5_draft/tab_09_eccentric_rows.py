"""
Item 09 - table of every eccentric micropolar row: the change of torque and load against the Newtonian control,
the turn of the force, and the two predictions of the host.

Run from the repository root:  python data/micropolar/ch5_draft/tab_09_eccentric_rows.py
"""
import sources
import style

SOURCE = sources.ECCENTRIC
LATTICE = {"": "", "_shifted": "shifted", "_rotated": "rotated"}


def main() -> int:
    analysis = style.load_json(SOURCE["analysis"])
    runs, pairs = analysis["runs"], analysis["pairs"]
    header = ["run", "control", "K", "eps", "g/dx", "l/dx", "narrowest gap/l", "Re (mu_inf)", "Re (mu)",
              "D_T wall form, rotor", "D_T wall form, stator", "D_T circle form, rotor", "D_T circle form, stator",
              "D_T mean, measured", "D_T mean, concentric error taken off", "spread of the four",
              "D_T film function", "D_T rule", "load change (%)", "load change, film function (%)",
              "load change, rule (%)", "turn of the force (deg)", "torque, 2nd - 1st half (%)",
              "density drift", "pressure drift", "git describe"]
    tex_header = ["run", "$K$", r"$\varepsilon$", rf"${style.GAP}/{style.SPACING}$", rf"$\ell/{style.SPACING}$",
                  r"$h_{\min}/\ell$", r"$\mathit{Re}_\mu$", r"$D_T$ wall R", "wall S", "circle R", "circle S", "mean",
                  "corrected", "film", "rule", r"$\Delta F/F$ (\%)", r"film (\%)", r"rule (\%)", r"turn ($^\circ$)"]
    plain, tex = [], []
    ordered = sorted(pairs.items(), key=lambda pair: (-pair[1]["coupling_ratio"], pair[1]["eccentricity_ratio"],
                                                      pair[1]["gap_in_spacings"], -pair[1]["length_in_spacings"],
                                                      pair[1].get("lattice") or ""))
    previous = None
    for name, entry in ordered:
        item = runs[name]
        changes = entry["changes"]
        predicted = entry["predictions"]
        reynolds = item["reynolds"]
        cells = [name, entry["control"], f"{entry['coupling_ratio']:g}", f"{entry['eccentricity_ratio']:g}",
                 f"{entry['gap_in_spacings']:g}", f"{entry['length_in_spacings']:g}",
                 f"{entry['narrowest_gap_over_length']:.1f}", f"{reynolds:g}",
                 f"{reynolds * (1.0 + 0.5 * entry['coupling_ratio']):g}",
                 f"{changes['rotor_torque_wall']:.4f}", f"{changes['stator_torque_wall']:.4f}",
                 f"{changes['rotor_torque_circle']:.4f}", f"{changes['stator_torque_circle']:.4f}",
                 f"{entry['mean_change']:.4f}",
                 f"{entry['corrected_mean']:.4f}" if entry["corrected_mean"] is not None else "-",
                 f"{entry['spread_of_readings']:.4f}", f"{predicted['torque_change_concentric']:.4f}",
                 f"{predicted['torque_change_rule']:.4f}", f"{100.0 * entry['load_change']:+.2f}",
                 f"{100.0 * predicted['load_change_plane']:+.2f}", f"{100.0 * predicted['load_change_rule']:+.2f}",
                 f"{entry['turn_degrees']:+.2f}", f"{100.0 * item['torque_change_between_halves']:+.3f}",
                 f"{item['density_drift']:+.1e}" if "density_drift" in item else "-",
                 f"{item['pressure_drift']:+.1e}" if "pressure_drift" in item else "-",
                 item["revision"]["describe"]]
        plain.append(cells)
        if previous is not None and previous != entry["coupling_ratio"]:
            tex.append("rule")
        previous = entry["coupling_ratio"]
        tex.append([style.texttt(name), cells[2], cells[3], cells[4], cells[5], cells[6], cells[8], *cells[9:15],
                    cells[16], cells[17], cells[18], cells[19], cells[20], cells[21]])
    style.write_table("tab_09_eccentric_rows", header, plain, tex_header, "l" + "r" * (len(tex_header) - 1), tex)

    main_rows = {name: entry for name, entry in pairs.items() if not entry.get("lattice")}
    floors = analysis.get("floors", {})
    floor_text = "; ".join(
        f"{label}: D_T {floor['torque_change_spread']:.4f}, load change {100.0 * floor['load_change_spread']:.2f} "
        f"points, turn {floor['turn_spread_degrees']:.2f} degrees" for label, floor in floors.items()
        if "pairs" in floor) or "not available in this data set"
    caption = (
        "Every micropolar run between eccentric cylinders (R1 = 4 g, R2 = 5 g, rotor axis displaced by eps g, w = 0 "
        "on both walls, Ma = 0.02, Re = 10 on mu_inf) against its Newtonian control on the same particles. D_T = "
        "T / T_control - 1 by four readings (rotor and stator; wall form = pair sums over the wall particles, circle "
        "form = flux of angular momentum across a circle at distance h from the wall); 'corrected' = their mean with "
        "the error each reading makes between concentric cylinders at the same K, l and g taken off ('-' where the "
        "wall sums of that concentric row do not exist"
        + (": " + ", ".join(f"`{name}`" for name, entry in pairs.items() if entry["corrected_mean"] is None)
           + ", which carry the uncorrected mean in every column that uses D_T"
           if any(entry["corrected_mean"] is None for entry in pairs.values()) else "")
        + "). Predictions: the film function of micropolar lubrication theory in the "
        "thin-film calculation (load: plane film; torque: shear-driven stress of the exact concentric solution at "
        "the local gap) and the wall rule on the exact Stokes solution of its slip-form geometry. "
        f"{len(pairs)} rows ({len(main_rows)} on the lattice as generated). Load change over the film function "
        f"{min(e['load_over_film'] for e in main_rows.values()):.3f} to "
        f"{max(e['load_over_film'] for e in main_rows.values()):.3f}, over the rule "
        f"{min(e['load_over_rule'] for e in main_rows.values()):.3f} to "
        f"{max(e['load_over_rule'] for e in main_rows.values()):.3f}; torque change over the film function "
        f"{min(e['torque_over_film'] for e in main_rows.values()):.3f} to "
        f"{max(e['torque_over_film'] for e in main_rows.values()):.3f}, over the rule "
        f"{min(e['torque_over_rule'] for e in main_rows.values()):.3f} to "
        f"{max(e['torque_over_rule'] for e in main_rows.values()):.3f}. Noise floor of the changes from three "
        f"lattices (max - min): {floor_text}. {SOURCE['note']}.")
    style.record("09", "Table of every eccentric micropolar row", SOURCE["status"], caption,
                 ["tab_09_eccentric_rows.csv", "tab_09_eccentric_rows.tex"], [SOURCE["analysis"]] + list(pairs),
                 {"rows": len(pairs)}, SOURCE["tag"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
