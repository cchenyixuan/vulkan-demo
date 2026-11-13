"""
Item 04 - table of every concentric row: the micropolar rows (tab_04_concentric_rows) and their Newtonian
controls (tab_04b_concentric_controls).

Run from the repository root:  python data/micropolar/ch5_draft/tab_04_concentric_rows.py
"""
import concentric_data as data
import sources
import style

SOURCE = sources.CONCENTRIC


def signed(value, form="{:+.1e}"):
    return "-" if value is None else form.format(value)


def main() -> int:
    runs = data.load(SOURCE)["runs"]
    rows = sorted((record for record in runs.values() if "relative_torque_change" in record
                   and ("sweep" in record["sections"] or "resolution" in record["sections"])),
                  key=lambda record: (-record["coupling_ratio"], record["gap_over_two_lengths"],
                                      record["gap_in_spacings"]))
    header = ["run", "K", "m", "g/dx", "l/dx", "Re (mu_inf)", "Re (mu)", "D_SPH", "D_exact", "D_SPH/D_exact",
              "noise floor/D_exact", "D_rule (slip form)", "D_rule/D_exact", "steady", "density drift",
              "pressure drift", "git describe"]
    tex_header = ["run", "$K$", "$m$", rf"${style.GAP}/{style.SPACING}$", rf"$\ell/{style.SPACING}$",
                  r"$\mathit{Re}_{\mu_\infty}$", r"$\mathit{Re}_{\mu}$", r"$D_{\mathrm{SPH}}$", r"$D_{\mathrm{exact}}$",
                  r"$D_{\mathrm{SPH}}/D_{\mathrm{exact}}$", "floor", r"$D_{\mathrm{rule}}$",
                  r"$D_{\mathrm{rule}}/D_{\mathrm{exact}}$", "steady", r"$\Delta\bar\rho/\rho_0$",
                  r"$\Delta\bar p/(\rho U^2)$"]
    plain, tex = [], []
    previous = None
    for record in rows:
        floor = record["noise_floor_of_change"]
        cells = [record["name"], f"{record['coupling_ratio']:g}", f"{record['gap_over_two_lengths']:g}",
                 f"{record['gap_in_spacings']}", f"{record['length_in_spacings']:g}",
                 f"{record['reynolds_far_viscosity']:g}", f"{record['reynolds_shear_viscosity']:g}",
                 f"{record['relative_torque_change']:.5f}", f"{record['exact_hydrodynamic']:.5f}",
                 f"{record['sph_over_exact']:.4f}",
                 "-" if floor is None else f"{floor / record['exact_hydrodynamic']:.4f}",
                 f"{record['slip_rule']:.5f}", f"{record['slip_rule_over_exact']:.4f}",
                 f"{record['steady_change']:.0e}", signed(record.get("density_drift")),
                 signed(record.get("pressure_drift")), record["revision"]["describe"]]
        plain.append(cells)
        if previous is not None and previous != record["coupling_ratio"]:
            tex.append("rule")
        previous = record["coupling_ratio"]
        tex.append([style.texttt(cells[0])] + [cell.replace("e-0", "e-").replace("e+0", "e+") for cell in cells[1:-1]])
    style.write_table("tab_04_concentric_rows", header, plain, tex_header, "l" + "r" * (len(tex_header) - 1), tex)

    controls = [record for record in runs.values() if not record["micropolar"] and "floor" in record["sections"]
                or record["name"].startswith("g400_newtonian")]
    header = ["run", "g/dx", "dt", "dt set by", "steps", "beta/nominal", "hydrodynamic R1 - R1 (dx)",
              "hydrodynamic R2 - R2 (dx)", "steady", "sector spread", "fit rms/U", "density level", "density drift",
              "pressure level", "pressure drift", "steps/s", "git describe"]
    tex_header = ["run", rf"${style.GAP}/{style.SPACING}$", r"$\Delta t$", "limit", "steps",
                  r"$\beta/\beta_{\mathrm{nominal}}$", rf"$\delta R_1/{style.SPACING}$",
                  rf"$\delta R_2/{style.SPACING}$", "steady", "sectors", "fit rms$/U$",
                  r"$\bar\rho/\rho_0 - 1$", r"$\Delta\bar\rho/\rho_0$", r"$\bar p/(\rho U^2)$",
                  r"$\Delta\bar p/(\rho U^2)$"]
    plain, tex = [], []
    for record in controls:
        cells = [record["name"], f"{record['gap_in_spacings']}", f"{record['timestep']:.4e}", record["governing"],
                 f"{record['steps']:,}", f"{record['beta_over_nominal']:.6f}",
                 f"{record['inner_offset_in_spacings']:+.3f}", f"{record['outer_offset_in_spacings']:+.3f}",
                 f"{record['steady_change']:.0e}", f"{record['sector_spread']:.0e}",
                 f"{record['fit']['fit_rms_over_wall_speed']:.0e}", signed(record.get("density_level")),
                 signed(record.get("density_drift")), signed(record.get("pressure_level"), "{:+.3f}"),
                 signed(record.get("pressure_drift")), f"{record['steps_per_second']:.0f}",
                 record["revision"]["describe"]]
        plain.append(cells)
        tex.append([style.texttt(cells[0])] + cells[1:-2])
    style.write_table("tab_04b_concentric_controls", header, plain, tex_header, "l" + "r" * (len(tex_header) - 1), tex)

    names = [record["name"] for record in rows]
    worst = max(rows, key=lambda record: abs(record["sph_over_exact"] - 1.0))
    by_length = {}
    for record in rows:
        by_length.setdefault(record["length_in_spacings"], []).append(record["sph_over_exact"])
    caption = (
        "Every micropolar row between concentric cylinders (R1 = 4 g, R2 = 5 g, U = Omega R1, w = 0 on both walls, "
        "Ma = 0.02, end time 2 g^2 / nu_inf, last quarter averaged). D_SPH: the potential-vortex constant of the "
        "velocity fit over that of the run's Newtonian control, minus one; D_exact and D_rule (the rule in its slip "
        "form) with the hydrodynamic radii of the control; noise floor: spread of the control's beta over three "
        "lattices, over D_exact (the row at g = 400 dx has one lattice and carries the floor of g = 200 dx; its "
        "step is the only one set by the viscous limit); steady: relative change of the fitted constant between the "
        "last two eighths of the "
        "run; drifts: change of the fluid's mean density (over rho0) and mean pressure (over rho U^2) between the "
        "last two eighths ('-' where the run did not record them). The second table lists the controls. "
        f"{len(rows)} rows; D_SPH / D_exact by l / dx: "
        + "; ".join(f"l = {length:g} dx: {min(values):.4f} to {max(values):.4f}"
                    for length, values in sorted(by_length.items()))
        + f". Largest departure: `{worst['name']}`, {worst['sph_over_exact']:.4f}. Tag {data.tags(runs, names)}.")
    style.record("04", "Table of every concentric row (and of the controls)", SOURCE["status"], caption,
                 ["tab_04_concentric_rows.csv", "tab_04_concentric_rows.tex", "tab_04b_concentric_controls.csv",
                  "tab_04b_concentric_controls.tex"], [SOURCE["analysis"]] + names, {"rows": len(rows)},
                 SOURCE["tag"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
