"""
Item 25 - table of the micropolar Taylor-Green runs: the measured rate against the exact two-mode solution, W / U
against its exact value, the rate over the control against the exact ratio (j of the run) and against the
quasi-steady mu_eff / (mu + kappa / 2), with the noise floors (repeated runs, Mach number halved, resolution).

Run from the repository root:  python data/micropolar/ch5_draft/tab_25_taylor_green.py
"""
import sources
import style

SOURCE = sources.PERIODIC
ORDER = ("tgv_kl025_n400", "tgv_kl05_n400", "tgv_kl1_n400", "tgv_kl2_n400", "tgv_kl05_n800", "tgv_kl1_n400_repeat",
         "tgv_kl1_n400_ma0p01", "tgv_kl1_inertia_n400", "tgv_kl1_j150_n400", "tgv_kl05_j001_n400",
         "tgv_kl1_j001_n400")
CONTROLS = ("tgv_control_n400", "tgv_control_n800", "tgv_control_n400_repeat", "tgv_control_n400_ma0p01",
            "tgv_control_n400_stored_density", "tgv_control_n400_small_dt")


def main() -> int:
    analysis = style.load_json(SOURCE["analysis"])
    tgv = analysis.get("tgv", {})
    rows, tex_rows = [], []
    for run in ORDER:
        if run not in tgv:
            continue
        item = tgv[run]
        label = run.replace("tgv_", "")
        cells = [label, f"{item['length_times_modulus']:g}", f"{item['theta_factor']:g}",
                 f"{item['rate']:.5f}", f"{item['exact_fitted_rate']:.5f}", f"{item['rate_over_exact']:.4f}",
                 f"{item['spin_ratio_over_exact']:.4f}",
                 f"{item.get('over_control', float('nan')):.4f}", f"{item.get('exact_over_control', float('nan')):.4f}",
                 f"{item['mu_effective_ratio']:.4f}"]
        rows.append(cells)
        tex_rows.append([style.texttt(label)] + cells[1:])
    for run in CONTROLS:
        if run not in tgv:
            continue
        item = tgv[run]
        label = run.replace("tgv_", "")
        cells = [label, "-", "-", f"{item['rate']:.5f}", f"{item['exact_rate']:.5f}", f"{item['nu_eff_over_nu']:.4f}",
                 "-", "-", "-", "-"]
        rows.append(cells)
        tex_rows.append([style.texttt(label)] + cells[1:])
    if not rows:
        print("[pack] item 25: no Taylor-Green runs yet")
        return 0
    header = ["run", "|k| l", "Theta / l^2", "rate", "exact rate", "rate / exact", "(W/U) / exact", "over control",
              "exact over control", "mu_eff / (mu + kappa/2)"]
    tex_header = ["run", r"$|\mathbf{k}|\ell$", r"$j/\ell^2$", "rate", "exact", "rate / exact", r"$(W/U)$ / exact",
                  "over control", "exact", r"$\mu_\mathrm{eff}/(\mu+\kappa/2)$"]
    floors = analysis.get("floors", {})
    for label, entry in floors.items():
        rate_cell = f"{entry['rate_difference']:+.1e}"
        ratio_cell = (f"{entry['over_control_difference']:+.1e}" if "over_control_difference" in entry else "-")
        rows.append([f"floor: {label}", "", "", rate_cell, "", "", "", ratio_cell, "", ""])
        tex_rows.append([style.tex_escape(f"floor: {label}"), "", "", rate_cell, "", "", "", ratio_cell, "", ""])
    style.write_table("tab_25_taylor_green", header, rows, tex_header=tex_header, tex_rows=tex_rows,
                      note="rates fitted to ln U over 5 <= t <= 20 for the run and for the exact solution at the same t")
    caption = (
        "The micropolar Taylor-Green vortex (box [0, 2 pi)^2, k = 1, Re = 20 on nu_inf, 50 on mu, K = 3, Ma 0.02 "
        "unless named): decay rate of U fitted over 5 <= t <= 20 (run) and the same fit of the exact two-mode solution "
        "at the same times (exact rate), W / U over the same window against its exact value, the rate over that of "
        "the Newtonian control (viscosity mu + kappa / 2, the same particles and dt) against the exact ratio with the "
        "run's j and against the quasi-steady mu_eff / (mu + kappa / 2) (j -> 0). Controls: their rate over "
        "2 k^2 nu_inf (nu_eff / nu). The two runs with j = l^2 / 100 take the dt of their angular diffusion (2.75e-5, "
        "8.6 times below the acoustic dt of the others) and their own control at that dt (control_n400_small_dt). "
        "Floor rows: the relative change of the rate and the change of the ratio over the control between repeated "
        "runs, at half the Mach number, between 400^2 and 800^2, and of the control between the two dt.")
    style.record("25", "Table of the Taylor-Green rates and ratios with their noise floors", SOURCE["status"], caption,
                 ["tab_25_taylor_green.csv", "tab_25_taylor_green.tex"], [SOURCE["analysis"]],
                 {"rows": len(rows)}, SOURCE["tag"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
