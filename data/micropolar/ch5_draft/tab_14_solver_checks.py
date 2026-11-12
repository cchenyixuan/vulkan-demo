"""
Item 14 - the solver checks of design note Sections 8 to 12 as they stand, built from the data files under
data/micropolar (no number is retyped):

  tab_14a_operator_checks          T3 (rigid rotation, linear w, sin(k x)), T4 (relaxation), the constant factors of
                                   check C, the 3-D operators of check E
  tab_14b_passive_label            checks A and B: w carried as a passive label, compared bitwise
  tab_14c_chain_against_one_slab   K = 2 against K = 1 in the micropolar cavity: T5 and the paired continuations of
                                   check D (five and ten runs per group)
  tab_14d_step_limits              the measured explicit step limits of Section 12.2
  tab_14e_cost                     throughput, per-kernel intervals and memory of Section 12.9

Run from the repository root:  python data/micropolar/ch5_draft/tab_14_solver_checks.py
"""
import re

import numpy as np

import style

SOURCE = {
    "status": "final (existing data)",
    "devices": {0: "RTX 5090", 1: "RX 7900 XTX"},           # the device map of Sections 8 to 12
    # "tree": git describe of the commit that put the data files into the tree (the files record no commit)
    "operators": {
        "t3": "data/micropolar/stage1_2/t3_operators.json", "t4": "data/micropolar/stage1_2/t4_relaxation.json",
        "factors": "data/micropolar/checks/c_factors.json", "three_dimensional": "data/micropolar/checks/e_operators3d.json",
        "tree": "micropolar-base-9-g278a557", "sections": "8 (T3, T4) and 10 (checks C and E)"},
    "label": {
        "files": ["data/micropolar/checks/a_label_0.json", "data/micropolar/checks/a_label_0_1.json",
                  "data/micropolar/checks/a_label_0_0.json", "data/micropolar/checks/a_label_0_0_0.json",
                  "data/micropolar/checks/a_label_0_0_1.json"],
        "ten_references": "data/micropolar/checks/a_label_0_floor10.json",
        "tree": "micropolar-base-9-g278a557", "sections": "10 (checks A and B)"},
    "chain": {
        "t5": {"K = 1 (b)": "data/micropolar/t5/compare_k1_b.json",
               "K = 2, pair": "data/micropolar/t5/compare_k2_cross.json",
               "K = 2, one card": "data/micropolar/t5/compare_k2_same.json"},
        "t5_run": "data/micropolar/t5/k1_a/run.json",
        "five": "data/micropolar/checks/d/paired.json", "ten": "data/micropolar/checks/d/paired10.json",
        "seam": "data/micropolar/stage1_2/seam_k2.json",
        "tree": "micropolar-base-9-g278a557", "tree_ten": "micropolar-base-13-g7e76dcb",
        "sections": "9 (T5), 10 (check D) and 11 (check D with ten runs)"},
    "step_limits": {
        "file": "data/micropolar/stage3/step_limit.json", "bound": "data/micropolar/stage3/lattice_spectral_radius.txt",
        "tree": "micropolar-base-15-g8cc60ea", "sections": "12.2",
        "support_radius_in_spacings": 5.0, "safety": 1.6,    # s = largest stable number / 1.6 (micropolar_couette.py)
        "epsilon_factor": "1e-06"},
    "cost": {"file": "data/micropolar/stage3/cost.json", "tree": "thesis-ch5-rc1", "sections": "12.9"},
}


def short(cell: str) -> str:
    """1.05e-02 -> 1.05e-2, as the other tables of the pack write exponents in their .tex."""
    return cell.replace("e-0", "e-").replace("e+0", "e+")


def signed_percent(value: float) -> str:
    """-4.13 %; a deviation below 0.005 % keeps two significant digits."""
    return f"{100.0 * value:+.2f} %" if abs(value) >= 5.0e-5 else f"{100.0 * value:+.1e} %"


def span(values, form="{:.1e}") -> str:
    low, high = form.format(min(values)), form.format(max(values))
    return low if low == high else f"{low} to {high}"


# ---- (a) operator checks ------------------------------------------------------------------------------------------
def operator_checks() -> tuple:
    source = SOURCE["operators"]
    header = ["check", "case", "quantity", "measured", "expected", "measured / expected - 1",
              "largest relative error over the particles", "note"]
    tex_header = ["check", "case", "quantity", "measured", "expected", "deviation", "largest error"]
    plain, tex, numbers = [], [], {}

    def add(check, case, quantity, measured, expected, deviation, largest, note="", rule=False):
        """case and quantity: (plain text, TeX)."""
        if rule and tex:
            tex.append("rule")
        plain.append([check, case[0], quantity[0], measured, expected, deviation, largest, note])
        tex.append([check, case[1], quantity[1], short(measured), short(expected), short(deviation), short(largest)])

    def arrangement(entry):
        jittered = entry["particles"].startswith("jittered")
        regularization = entry["regularization_xi"]
        text = (f"2-D, {'jittered by 0.1 dx' if jittered else 'lattice'}, regularization xi = {regularization:g}, "
                f"{entry['interior']:,} particles")
        particles = rf"jittered by $0.1\,{style.SPACING}$" if jittered else "lattice"
        tex_text = rf"2-D, {particles}, $\xi_{{\mathrm{{reg}}}} = {regularization:g}$"
        return text, tex_text

    spin_rate = ("Theta alpha_z / c (rigid rotation: curl v = 2 Omega)", r"$\Theta\alpha_z/c$, rigid rotation")
    curl = ("a / c against curl w (linear w_z)", r"$\mathbf{a}/c$ against $\nabla\times\mathbf{w}$, linear $w_z$")
    residual = ("Theta alpha + 2 c w - xi lap w, over grad w / dx (linear w_z)",
                rf"$\Theta\alpha + 2cw - \xi\nabla^2 w$ over $|\nabla w|/{style.SPACING}$")
    amplitude = ("lap w / (-k^2 w), w = sin(k x)", r"$\nabla^2 w/(-k^2 w)$, $w = \sin kx$")
    t3 = style.load_json(source["t3"])
    first = True
    for entry in t3:
        if entry["state"] == "rigid rotation":
            mean = entry["theta_alpha_over_c_mean_z"]
            add("T3", arrangement(entry), spin_rate, f"{mean:.7f}", "2", f"{mean / 2.0 - 1.0:+.1e}",
                f"{entry['relative_error_max']:.1e}", f"largest |a| {entry['acceleration_max']:.1e} m/s^2 (curl w = 0)",
                rule=first)
            first = False
            numbers[f"T3 rigid rotation, {entry['particles']}"] = {
                "mean": mean, "relative_error_max": entry["relative_error_max"],
                "acceleration_max": entry["acceleration_max"]}
    for entry in t3:
        if entry["state"] == "linear w_z":
            add("T3", arrangement({**entry}), curl, "-", "-", "-", f"{entry['curl_relative_error_max']:.1e}",
                f"rms {entry['curl_relative_error_rms']:.1e}")
            add("T3", arrangement({**entry}), residual, f"{entry['laplacian_over_gradient_per_spacing_max']:.1e}", "0",
                "-", "-", f"rms {entry['laplacian_over_gradient_per_spacing_rms']:.1e}; "
                + ("not exact on disordered particles" if entry["particles"].startswith("jittered")
                   else "rounding of the pair differences"))
            numbers[f"T3 linear w_z, {entry['particles']}"] = {
                "curl_relative_error_max": entry["curl_relative_error_max"],
                "laplacian_over_gradient_per_spacing_max": entry["laplacian_over_gradient_per_spacing_max"]}
    epsilon_factor = None                       # the case's EPS_H_SQUARED over h^2 (h: support radius), from the sin(k x) rows
    for entry in t3:
        epsilon_label = entry.get("eps", "")
        epsilon_tex = r"case $\varepsilon^2$" if epsilon_label.startswith("case") else r"$\varepsilon^2 = 0$"
        if entry["state"] == "sin(k x)":
            ratio = entry["amplitude_ratio"]
            if epsilon_label.startswith("case"):
                epsilon_factor = entry["eps_h_squared"] / entry["smoothing_length"] ** 2
                epsilon_label = f"{epsilon_label} = {epsilon_factor:g} h^2"
            add("T3", (f"2-D lattice, dx = {1000.0 * entry['spacing']:g} mm, k h = {entry['k_h']:.2f}, {epsilon_label}",
                       rf"2-D lattice, $kh = {entry['k_h']:.2f}$, {epsilon_tex}"), amplitude, f"{ratio:.6f}", "1",
                f"{ratio - 1.0:+.2e}", "-", f"{entry['interior']:,} particles; relative L2 error "
                                            f"{entry['relative_l2_error']:.3e}")
        elif entry["state"] == "sin(k x) Richardson":
            ratio = entry["amplitude_ratio_limit"]
            numbers[f"T3 sin(k x) limit, {epsilon_label}"] = {"limit": ratio, "order": entry["observed_order"]}
            if epsilon_label.startswith("case"):
                epsilon_label = f"{epsilon_label} = {epsilon_factor:g} h^2"
            add("T3", (f"2-D lattice, limit dx -> 0 at h / dx = 5, {epsilon_label}",
                       rf"2-D lattice, ${style.SPACING} \to 0$ at $h/{style.SPACING} = 5$, {epsilon_tex}"), amplitude,
                f"{ratio:.6f}", "1", f"{ratio - 1.0:+.2e}", "-", f"Richardson order {entry['observed_order']:.2f}")

    product = ("w_z against the product of the scheme, w_0 (1 - lambda dt / 2) (1 - lambda dt)^n",
               r"$w_z$ against the scheme's product")
    exponential = ("w_z against exp(-lambda t): relative difference", r"$w_z$ against $\exp(-\lambda t)$")
    first = True
    for entry in style.load_json(source["t4"]):
        steps = f"{entry['steps']} step" + ("" if entry["steps"] == 1 else "s")
        case = (f"2 c dt / Theta = {entry['rate_times_timestep']:.3f}, after {steps}",
                rf"$2c\,\Delta t/\Theta = {entry['rate_times_timestep']:.3f}$, {steps}")
        add("T4", case, product, f"{entry['mean']:.7f}", f"{entry['discrete']:.7f}",
            f"{entry['mean'] / entry['discrete'] - 1.0:+.1e}", f"{entry['relative_to_discrete_max']:.1e}",
            f"spread over the particles {entry['spread']:g}; largest in-plane w {entry['in_plane_spin_max']:g}, "
            f"largest v {entry['velocity_max']:g}", rule=first)
        add("T4", case, exponential, f"{entry['relative_to_exponential']:+.4e}",
            f"{entry['expected_first_order_difference']:+.4e}", "-", "-",
            "expected: the first-order difference of forward Euler")
        first = False
        numbers[f"T4 after {entry['steps']} steps"] = {
            "mean": entry["mean"], "product": entry["discrete"], "largest_deviation": entry["relative_to_discrete_max"],
            "relative_to_exponential": entry["relative_to_exponential"],
            "expected_first_order_difference": entry["expected_first_order_difference"]}

    # check C: the constant factors on nu (viscous term) and on xi (angular diffusion), Richardson limits. The rows
    # on the jittered lattice are left out: their extrapolation is noise (orders -0.1 to 4.4, design note Section 10).
    factor_nu = ("factor on nu (viscous term)", r"factor on $\nu$")
    factor_xi = ("factor on xi (angular diffusion)", r"factor on $\xi$")
    factors = style.load_json(source["factors"])
    three_dimensional = style.load_json(source["three_dimensional"])
    first = True

    def add_factor(case, coefficient, item):
        nonlocal first
        add("C", case, factor_nu if coefficient == "nu" else factor_xi, f"{item['factor']:.4f}", "1",
            signed_percent(item["factor"] - 1.0), "-", f"Richardson order {item['order']:.2f}; ratios "
            + ", ".join(f"{ratio:.6f}" for ratio in item["ratios"]), rule=first)
        tex[-1][5] = short(signed_percent(item["factor"] - 1.0)).replace(" %", r"\,\%")
        first = False
        numbers[f"C {case[0]}, {coefficient}"] = {"factor": item["factor"], "order": item["order"]}
    for epsilon_label in ("case EPS_H_SQUARED", "EPS_H_SQUARED = 0"):
        epsilon_tex = r"case $\varepsilon^2$" if epsilon_label.startswith("case") else r"$\varepsilon^2 = 0$"
        for entry in factors:
            if entry.get("coefficient") and entry["eps"] == epsilon_label:
                add_factor((f"2-D, h / dx = 5, wave vector along a lattice axis, {epsilon_label}",
                            rf"2-D, $h/{style.SPACING} = 5$, $\mathbf{{k}}$ along an axis, {epsilon_tex}"),
                           entry["coefficient"], entry)
        if epsilon_label.startswith("case"):
            for entry in factors:
                if entry.get("jitter") == 0.0 and entry["angle_degrees"] > 0.0:
                    for coefficient in ("nu", "xi"):
                        add_factor((f"2-D, h / dx = 5, wave vector at {entry['angle_degrees']:g} degrees, {epsilon_label}",
                                    rf"2-D, $\mathbf{{k}}$ at ${entry['angle_degrees']:g}^\circ$, {epsilon_tex}"),
                                   coefficient, entry[coefficient])
    for epsilon_label in ("case EPS_H_SQUARED", "EPS_H_SQUARED = 0"):
        epsilon_tex = r"case $\varepsilon^2$" if epsilon_label.startswith("case") else r"$\varepsilon^2 = 0$"
        for entry in three_dimensional:
            if entry.get("dimension") == 3 and entry["eps"] == epsilon_label:
                add_factor((f"3-D, h / dx = {entry['h_over_dx']:g}, wave vector along (2, 3, 6) / 7, {epsilon_label}",
                            rf"3-D, $h/{style.SPACING} = {entry['h_over_dx']:g}$, {epsilon_tex}"),
                           entry["coefficient"], entry)

    first = True
    for entry in three_dimensional:
        if "state" not in entry:
            continue
        jittered = entry["particles"] == "jittered"
        case = (f"3-D, h / dx = 4, {'jittered by 0.1 dx' if jittered else 'lattice'}, {entry['interior']:,} particles",
                rf"3-D, {'jittered' if jittered else 'lattice'}")
        quantity = (("Theta alpha / c, three components (rigid rotation: curl v = 2 Omega)",
                     r"$\Theta\boldsymbol{\alpha}/c$, rigid rotation") if entry["state"] == "rigid rotation"
                    else ("a / c against curl w, three components (linear w)",
                          r"$\mathbf{a}/c$ against $\nabla\times\mathbf{w}$, linear $\mathbf{w}$"))
        add("E", case, quantity, "(" + ", ".join(f"{value:.6f}" for value in entry["mean"]) + ")",
            "(" + ", ".join(f"{value:g}" for value in entry["expected"]) + ")", "-",
            f"{max(entry['relative_error_max']):.1e}",
            f"largest |a| {entry['acceleration_max']:.1e} m/s^2" if "acceleration_max" in entry else "", rule=first)
        first = False
        numbers[f"E {entry['state']}, {entry['particles']}"] = {"relative_error_max": max(entry["relative_error_max"])}
    style.write_table("tab_14a_operator_checks", header, plain, tex_header, "lllrrrr", tex,
                      note="deviation: measured / expected - 1; largest error: largest relative error over the "
                           "particles; the .csv has a note column (residuals, orders, particle counts)")

    rigid = [value for key, value in numbers.items() if key.startswith("T3 rigid rotation") and "default" not in key]
    default = [value for key, value in numbers.items() if key.startswith("T3 rigid rotation") and "default" in key]
    linear = [value for key, value in numbers.items() if key.startswith("T3 linear") and "default" not in key]
    relaxation = [value for key, value in numbers.items() if key.startswith("T4")]
    three = [value["relative_error_max"] for key, value in numbers.items() if key.startswith("E ")]
    orders = [value["order"] for key, value in numbers.items() if key.startswith("C ")]
    axis_case = "C 2-D, h / dx = 5, wave vector along a lattice axis, case EPS_H_SQUARED"
    diagonal = "C 2-D, h / dx = 5, wave vector at 45 degrees, case EPS_H_SQUARED, nu"
    paragraph = (
        f"(a) tab_14a_operator_checks (design note Sections {source['sections']}; data in the tree since "
        f"`{source['tree']}`). The micropolar operators on prescribed fields, one slab, 2-D at h = 5 dx unless said "
        f"otherwise. Rigid rotation and a linear w_z with the regularization xi = 0: Theta alpha_z / c = 2 to "
        f"{max(item['relative_error_max'] for item in rigid):.1e} and a / c = curl w to "
        f"{max(item['curl_relative_error_max'] for item in linear):.1e} (largest relative error over the particles, "
        f"lattice and jittered); with the default regularization 0.001 both are off by "
        f"{min(item['relative_error_max'] for item in default):.1e} to "
        f"{max(item['relative_error_max'] for item in default):.1e}, the factor M (M + xi I)^-1. Relaxation of a "
        f"uniform w (T4): the scheme's product is met to "
        f"{max(item['largest_deviation'] for item in relaxation):.1e} over 300 steps, and the difference to the "
        f"exponential is the expected first-order one. Constant factors at h = 5 dx (check C, Richardson limits of "
        f"three resolutions, orders {span(orders, '{:.3f}')}): {numbers[axis_case + ', xi']['factor']:.4f} on xi in "
        f"every direction, {numbers[axis_case + ', nu']['factor']:.4f} to {numbers[diagonal]['factor']:.4f} on nu with "
        f"the direction of the wave vector. These are the factors of the viscous form of that stage "
        f"(Monaghan-Gingold) with EPS factor {epsilon_factor:g}; the thesis configuration uses the laplacian form with "
        f"EPS factor 1e-6. 3-D operators at h = 4 dx (check E): exact to {max(three):.1e}.")
    return paragraph, numbers, [source["t3"], source["t4"], source["factors"], source["three_dimensional"]]


# ---- (b) passive label --------------------------------------------------------------------------------------------
def devices_text(device_map: list) -> str:
    names = [SOURCE["devices"][device] for device in device_map]
    if len(set(names)) == 1:
        return names[0] if len(names) == 1 else f"{names[0]} ({len(names)} slabs on one card)"
    parts = []
    for name in dict.fromkeys(names):
        parts.append(name if names.count(name) == 1 else f"{name} ({names.count(name)} slabs)")
    return " + ".join(parts)


def passive_label() -> tuple:
    source = SOURCE["label"]
    header = ["slabs K", "devices", "walls", "fluid particles compared", "bitwise mismatches of w", "wall particles",
              "wall particles changed", "ids unique and complete", "seam crossings", "defrags", "restarts",
              "steps", "cuts at x (L)", "u: largest rms to a Newtonian reference / largest reference pair rms",
              "v: the same", "rho: the same", "Newtonian references"]
    tex_header = ["$K$", "devices", "walls", "fluid particles", "mismatches", "wall particles", "changed",
                  "seam crossings", "defrags", "restarts", "steps", "$u$", "$v$", r"$\rho$"]
    plain, tex, rows = [], [], []

    def add(entry, references):
        cuts = entry["cuts_x"][0]
        cells = [str(entry["slabs"]), devices_text(entry["device_map"]), entry["walls"], f"{entry['fluid_compared']:,}",
                 str(entry["mismatches"]), f"{entry['wall_particles']:,}", str(entry["wall_mismatches"]),
                 "yes" if entry["ids_unique_and_complete"] else "no", f"{entry['seam_crossings']:,}",
                 str(entry["defrags"]), str(len(entry["steps"]) - 1), " + ".join(f"{steps:,}" for steps in entry["steps"]),
                 ", ".join(f"{cut:+.3f}" for cut in cuts) if cuts else "-"]
        cells += [f"{entry['floor'][field][2]:.2f}" for field in ("u", "v", "rho")] + [references]
        plain.append(cells)
        tex.append(cells[:7] + cells[8:12] + cells[13:16])
        rows.append(entry)
    for path in source["files"]:
        for entry in style.load_json(path):
            add(entry, "three")
            if entry["slabs"] == 1 and entry["walls"] == "simple":
                for repeat in style.load_json(source["ten_references"]):
                    add(repeat, "ten (the comparison repeated)")
    style.write_table("tab_14b_passive_label", header, plain, tex_header, "rll" + "r" * 11, tex,
                      note="u, v, rho: largest sampled rms to a Newtonian run of the same two phases over the largest "
                           "rms between two such runs (three references; second row of K = 1, simple: ten)")
    chains = [entry for entry in rows if entry["slabs"] > 1]
    crossings = sorted({entry["seam_crossings"] for entry in chains})
    numbers = {"runs": len(rows), "mismatches": sum(entry["mismatches"] for entry in rows),
               "wall_mismatches": sum(entry["wall_mismatches"] for entry in rows),
               "fluid_compared": rows[0]["fluid_compared"], "wall_particles": rows[0]["wall_particles"],
               "seam_crossings": crossings,
               "largest_ratio_three_references": max(entry["floor"][field][2] for entry in rows[:1] + rows[2:]
                                                     for field in ("u", "v", "rho")),
               "ratios_ten_references": [rows[1]["floor"][field][2] for field in ("u", "v", "rho")]}
    fields = ("u", "v", "rho")
    # the K = 1 run with simple walls is in the table twice (three references, then ten): rows[0] and rows[1]
    up_to_two = [entry["floor"][field][2] for entry in rows[2:] if entry["slabs"] <= 2 for field in fields]
    three_slabs = [entry["floor"][field][2] for entry in rows if entry["slabs"] == 3 for field in fields]
    worst = max((entry for entry in rows if entry["slabs"] == 3),
                key=lambda entry: max(entry["floor"][field][2] for field in fields))
    numbers["ratios_three_references_up_to_two_slabs"] = [min(up_to_two), max(up_to_two)]
    numbers["ratios_three_references_three_slabs"] = [min(three_slabs), max(three_slabs)]
    paragraph = (
        f"(b) tab_14b_passive_label (design note Section {source['sections']}; data in the tree since "
        f"`{source['tree']}`). With c = 0 and xi = 0 the micro-rotation is a label that the solver must only carry: "
        f"w_0 of every fluid particle derived from its global id, the n = 250 cavity started from the K = 1 state at "
        f"t = 40.2, {rows[0]['steps'][0]:,} steps, the state written and restarted, {rows[0]['steps'][1]:,} more. "
        f"{numbers['mismatches']} bitwise mismatches among the {numbers['fluid_compared']:,} fluid particles in each "
        f"of the {len(rows) - 1} runs (K = 1, 2 and 3, simple and adami walls, one card and the pair), "
        f"{numbers['wall_mismatches']} of the {numbers['wall_particles']:,} wall particles changed; per run "
        f"{rows[0]['defrags']} defrags, one restart and {crossings[0]:,} to {crossings[-1]:,} crossings of a cut in "
        f"the chains. u, v and rho against Newtonian runs of the same two phases, largest sampled rms over the "
        f"largest rms between two references (three references per run): {min(up_to_two):.2f} to "
        f"{max(up_to_two):.2f} at K = 1 and 2, {min(three_slabs):.2f} to {max(three_slabs):.2f} at K = 3 (the largest "
        f"on {devices_text(worst['device_map'])}, {worst['walls']} walls); the floor of three references was too "
        f"small once, {max(rows[0]['floor'][field][2] for field in fields):.2f} at K = 1 with simple walls, which "
        f"is {max(numbers['ratios_ten_references']):.2f} when the comparison is repeated with ten references.")
    return paragraph, numbers, source["files"] + [source["ten_references"]]


# ---- (c) chain against one slab -----------------------------------------------------------------------------------
QUANTITIES = [  # key in the data files, plain text, TeX
    ("min u_vertical", "min u, vertical centre line", r"$\min u$, vertical centre line"),
    ("min v_horizontal", "min v, horizontal centre line", r"$\min v$, horizontal centre line"),
    ("max v_horizontal", "max v, horizontal centre line", r"$\max v$, horizontal centre line"),
    ("min w_vertical", "min w, vertical centre line (the line next to the cut)", r"$\min w$, vertical centre line"),
    ("max w_vertical", "max w, vertical centre line", r"$\max w$, vertical centre line"),
    ("min w_horizontal", "min w, horizontal centre line", r"$\min w$, horizontal centre line"),
    ("max w_horizontal", "max w, horizontal centre line", r"$\max w$, horizontal centre line"),
    ("L2 u_vertical", "L2 of u(y) from the K = 1 mean", r"$L_2$ of $u(y)$ from the $K = 1$ mean"),
    ("L2 v_horizontal", "L2 of v(x) from the K = 1 mean", r"$L_2$ of $v(x)$"),
    ("L2 w_vertical", "L2 of w(y) from the K = 1 mean", r"$L_2$ of $w(y)$"),
    ("L2 w_horizontal", "L2 of w(x) from the K = 1 mean", r"$L_2$ of $w(x)$"),
]


def chain_against_one_slab() -> tuple:
    source = SOURCE["chain"]
    from_rest = {}
    for label, path in source["t5"].items():
        for entry in style.load_json(path)["extrema"]:
            key = f"{entry['kind']} {entry['profile']}"
            from_rest.setdefault(key, {"K = 1 (a)": entry["reference"]})[label] = entry["test"]
    five, ten = style.load_json(source["five"]), style.load_json(source["ten"])

    def statistics(analysis, key, group):
        values = np.array([run["values"][key] for run in analysis["groups"][group]])
        return {"runs": int(values.size), "mean": float(values.mean()), "std": float(values.std(ddof=1))}

    def stored(analysis, key, group):
        return next(item for item in analysis["summary"] if item["quantity"] == key and item["group"] == group)

    def value_text(key, value):
        return f"{value:.1e}" if key.startswith("L2") else f"{value:.6f}"
    header = ["quantity",
              "from rest, t = 40.2: K = 1 (a)", "from rest: K = 1 (b)", "from rest: K = 2 on RTX 5090 + RX 7900 XTX",
              "from rest: K = 2 on one card",
              "five continuations: K = 1 mean", "five: K = 1 std", "five: K = 2 pair mean", "five: K = 2 pair std",
              "five: K = 2 pair minus K = 1, in K = 1 std", "five: K = 2 one card mean", "five: K = 2 one card std",
              "five: K = 2 one card minus K = 1, in K = 1 std",
              "ten continuations: K = 1 mean", "ten: K = 1 std", "ten: K = 2 one card mean", "ten: K = 2 one card std",
              "ten: K = 2 one card minus K = 1", "ten: the same in standard errors of the difference",
              "K = 2 pair (its five runs) mean", "K = 2 pair (its five runs) std",
              "K = 2 pair minus the ten K = 1 runs", "K = 2 pair minus the ten K = 1 runs, in standard errors of the "
                                                     "difference"]
    tex_header = ["quantity", r"$K = 1$: mean", "std", r"$K = 2$, one card: mean", "std", r"$\Delta/\sigma_\Delta$",
                  r"$K = 2$, pair: mean", "std", r"$\Delta/\sigma_\Delta$"]
    plain, tex, numbers = [], [], {}
    for key, text, tex_text in QUANTITIES:
        cells = [text]
        rest = from_rest.get(key)
        cells += ([f"{rest[label]:.6f}" for label in ("K = 1 (a)", "K = 1 (b)", "K = 2, pair", "K = 2, one card")]
                  if rest else ["-"] * 4)
        reference = statistics(five, key, "K1")
        cells += [value_text(key, reference["mean"]), f"{reference['std']:.1e}"]
        record = {"five": {}, "ten": {}}
        for group in ("K2x", "K2s"):
            item, check = statistics(five, key, group), stored(five, key, group)
            difference = (item["mean"] - reference["mean"]) / reference["std"]
            if abs(difference - check["difference_in_reference_std"]) > 1e-6:
                raise RuntimeError(f"{key}, {group}: {difference} is not the stored {check['difference_in_reference_std']}")
            cells += [value_text(key, item["mean"]), f"{item['std']:.1e}", f"{difference:+.2f}"]
            record["five"][group] = difference
        reference = statistics(ten, key, "K1")
        cells += [value_text(key, reference["mean"]), f"{reference['std']:.1e}"]
        tex_cells = [tex_text, value_text(key, reference["mean"]), short(f"{reference['std']:.1e}")]
        for group in ("K2s", "K2x"):
            item = statistics(ten, key, group)
            difference = item["mean"] - reference["mean"]
            error = np.sqrt(reference["std"] ** 2 / reference["runs"] + item["std"] ** 2 / item["runs"])
            cells += [value_text(key, item["mean"]), f"{item['std']:.1e}", f"{difference:+.1e}",
                      f"{difference / error:+.2f}"]
            tex_cells += [value_text(key, item["mean"]), short(f"{item['std']:.1e}"), f"{difference / error:+.2f}"]
            record["ten"][group] = {"difference": float(difference), "in_standard_errors": float(difference / error),
                                    "runs": item["runs"]}
        record["ten"]["K1"] = reference
        plain.append(cells)
        if key == "L2 u_vertical":
            tex.append("rule")
        tex.append(tex_cells)
        numbers[text] = record
    groups = {name: len(runs) for name, runs in ten["groups"].items()}
    style.write_table("tab_14c_chain_against_one_slab", header, plain, tex_header, "l" + "r" * 8, tex,
                      note=f"paired continuations from the K = 1 state at t = 40.2: {groups['K1']} runs at K = 1, "
                           f"{groups['K2s']} at K = 2 on one card, {groups['K2x']} at K = 2 on the pair; Delta / "
                           "sigma_Delta: difference of the means to K = 1 in standard errors of that difference; the "
                           ".csv also has the runs from rest (T5) and the statistics of the first five runs")

    seam = style.load_json(source["seam"])

    def near_cut(run, steps=20, field="acceleration"):
        return span([entry["near_seam_rms"] for entry in seam if entry["run"] == run and entry["steps"] == steps
                     and entry["field"] == field])
    run = style.load_json(source["t5_run"])
    names = {"K = 1 (b)": "the K = 1 repeat", "K = 2, pair": "K = 2 on the pair", "K = 2, one card": "K = 2 on one card"}
    relative = {names[label]: [abs(values[label] / values["K = 1 (a)"] - 1.0) for values in from_rest.values()]
                for label in source["t5"]}
    extrema = [text for key, text, _ in QUANTITIES if not key.startswith("L2")]
    worst_five = max(abs(value) for record in numbers.values() for value in record["five"].values())

    def largest(group):
        """The extremum whose mean is furthest from the ten K = 1 runs, in standard errors of the difference."""
        text = max(extrema, key=lambda name: abs(numbers[name]["ten"][group]["in_standard_errors"]))
        item = numbers[text]["ten"][group]
        return text, f"{text.split(',')[0]} on the {text.split(', ')[1].split(' (')[0]}", item
    seam_line = numbers[extrema[3]]["ten"]
    one_card, pair = largest("K2s"), largest("K2x")
    cut = ten["groups"]["K2s"][0]["cuts_x"][0]
    numbers["_summary"] = {
        "relative_differences_from_rest": {label: [min(values), max(values)] for label, values in relative.items()},
        "largest_difference_five_runs_in_reference_std": worst_five, "groups": groups, "cut_x": cut,
        "largest_offset_one_card": {"quantity": one_card[0], **one_card[2]},
        "largest_offset_pair": {"quantity": pair[0], **pair[2]}}
    paragraph = (
        f"(c) tab_14c_chain_against_one_slab (design note Sections {source['sections']}; data in the tree since "
        f"`{source['tree']}`, the ten-run set since `{source['tree_ten']}`). The micropolar cavity "
        f"`{run['case'].split('/')[1]}` (n = 250, Re = 400, K = 3) as a chain of two slabs against one slab, on RTX "
        f"5090 + RX 7900 XTX (the pair) and with both slabs on the RTX 5090 (one card); extrema of the centre-line "
        f"profiles of u, v (in U) and w (in U / L). From rest to t = 40.2 ({run['steps']:,} steps, mean over the "
        f"last 5 time units; in the .csv): relative differences to the first K = 1 run: "
        + "; ".join(f"{label} {values[0]:.0e} to {values[1]:.0e}" for label, values in
                    numbers["_summary"]["relative_differences_from_rest"].items())
        + ". K = 2 agrees with K = 1 as well as K = 1 with its repeat, to three digits; four are not reached by "
        f"either. Paired continuations from the K = 1 state at t = 40.2 (200,000 steps each, first time unit "
        f"discarded, mean of the next five; cut at x = {cut:+.4f} L from the vertical centre line): with five runs "
        f"per group every quantity is within {worst_five:.1f} K = 1 standard deviations. With {groups['K1']} runs at "
        f"K = 1, {groups['K2s']} at K = 2 on one card and the {groups['K2x']} on the pair (the table; differences of "
        f"the means in standard errors of the difference): the minimum of w on the line next to the cut differs by "
        f"{seam_line['K2s']['in_standard_errors']:+.2f} (one card) and {seam_line['K2x']['in_standard_errors']:+.2f} "
        f"(pair); the largest offset of an extremum on one card is {one_card[1]}, "
        f"{one_card[2]['in_standard_errors']:+.1f} ({one_card[2]['difference']:+.1e} U; "
        f"{numbers[one_card[0]]['ten']['K2x']['in_standard_errors']:+.1f} on the pair), and on the pair {pair[1]}, "
        f"{pair[2]['in_standard_errors']:+.1f} ({pair[2]['difference']:+.1e} U; "
        f"{numbers[pair[0]]['ten']['K2s']['in_standard_errors']:+.1f} on one card). Neither is established as a seam "
        f"effect and neither is excluded: the offsets are in the fifth digit and the standard deviations are from "
        f"five and ten runs. The L2 rows are biased low for K = 1 (each K = 1 run is part of the mean it is compared "
        f"with). Per particle, 20 steps after a prescribed smooth state: the acceleration within 4 h of the cut "
        f"differs from a K = 1 run by {near_cut('k2_one_device')} (K = 2 on one card) against {near_cut('k1_repeat')} "
        f"for a K = 1 repeat, and by {near_cut('k2')} on the pair against {near_cut('k1_other_device')} for K = 1 on "
        f"the other card (relative rms, simple and adami walls).")
    return paragraph, numbers, list(source["t5"].values()) + [source["t5_run"], source["five"], source["ten"],
                                                              source["seam"]]


# ---- (d) step limits ----------------------------------------------------------------------------------------------
def step_limits() -> tuple:
    source = SOURCE["step_limits"]
    data = style.load_json(source["file"])
    bounds = {}
    for line in (style.REPO_ROOT / source["bound"]).read_text(encoding="utf-8").splitlines():
        match = re.match(r"EPS factor (\S+) (MG|BM): lambda_max h\^2 = ([\d.]+) .* = ([\d.]+)$", line.strip())
        if match and match.group(1) == source["epsilon_factor"]:
            bounds[match.group(2)] = {"lambda_max": float(match.group(3)), "bound": float(match.group(4))}
    terms = [("laplacian", "viscous term, laplacian form (thesis configuration)",
              "viscous, " + style.texttt("laplacian"), "nu", r"$\nu$", "BM"),
             ("monaghan_gingold", "viscous term, monaghan_gingold form, same EPS factor",
              "viscous, " + style.texttt("monaghan_gingold"), "nu", r"$\nu$", "MG"),
             ("angular", "angular diffusion", "angular diffusion", "xi / Theta", r"$\xi/\Theta$", "BM")]
    header = ["term", "diffusivity D", "largest stable dt D / h^2", "smallest failing dt D / h^2",
              "largest stable dt D / dx^2", "s = largest stable / 1.6", "probes",
              "forward-Euler bound of the lattice symbol, 2 / (lambda_max h^2)", "largest stable / bound"]
    tex_header = ["term", "$D$", r"largest stable $\Delta t\,D/h^2$", "smallest failing",
                  rf"largest stable $\Delta t\,D/{style.SPACING}^2$", "$s$", r"bound $2/(\lambda_{\max} h^2)$",
                  "stable / bound"]
    plain, tex, numbers = [], [], {}
    ratio = source["support_radius_in_spacings"] ** 2
    for kind, text, tex_text, diffusivity, tex_diffusivity, symbol in terms:
        item = data["summary"][kind]
        probes = [probe for probe in data["probes"] if probe["kind"] == kind]
        bound = bounds[symbol]["bound"]
        cells = [text, diffusivity, f"{item['largest_stable']:.5f}", f"{item['smallest_failing']:.5f}",
                 f"{ratio * item['largest_stable']:.2f}", f"{item['largest_stable'] / source['safety']:.4f}",
                 str(len(probes)), f"{bound:.5f}", f"{item['largest_stable'] / bound:.2f}"]
        plain.append(cells)
        tex.append([tex_text, tex_diffusivity] + cells[2:6] + cells[7:])
        numbers[kind] = {**item, "in_spacings_squared": ratio * item["largest_stable"],
                         "step_number": item["largest_stable"] / source["safety"], "bound": bound,
                         "stable_over_bound": item["largest_stable"] / bound, "probes": len(probes)}
    probe = data["probes"][0]
    style.write_table("tab_14d_step_limits", header, plain, tex_header, "ll" + "r" * 6, tex,
                      note="h: support radius (5 dx); bound: forward Euler with the largest eigenvalue of the lattice "
                           "symbol (Brookshaw-Morris sum for the laplacian form and the angular diffusion)")
    paragraph = (
        f"(d) tab_14d_step_limits (design note Section {source['sections']}; data in the tree since "
        f"`{source['tree']}`). Explicit step limits by bisection on runs that stay intact or blow up: annulus with "
        f"{probe['loaded']:,} particles (g = 50 dx), adami walls, EPS factor 1e-6, {probe['steps']:,} steps at the "
        f"acoustic step, the diffusivity raised until the run fails; h is the support radius (5 dx). Largest stable "
        f"dt D / h^2: {numbers['laplacian']['largest_stable']:.4f} for the laplacian form of the viscous term "
        f"({numbers['laplacian']['in_spacings_squared']:.2f} in dx^2), "
        f"{numbers['monaghan_gingold']['largest_stable']:.4f} for the Monaghan-Gingold form, "
        f"{numbers['angular']['largest_stable']:.4f} for the angular diffusion: the laplacian form allows a step "
        f"{numbers['laplacian']['largest_stable'] / numbers['monaghan_gingold']['largest_stable']:.2f} times larger. "
        f"The limits are {numbers['monaghan_gingold']['stable_over_bound']:.2f} and "
        f"{numbers['laplacian']['stable_over_bound']:.2f} of the forward-Euler bound of the lattice symbols. The "
        f"cases use s = limit / {source['safety']:g}: {numbers['laplacian']['step_number']:.4f} (laplacian), "
        f"{numbers['monaghan_gingold']['step_number']:.4f} (Monaghan-Gingold), "
        f"{numbers['angular']['step_number']:.4f} (angular).")
    return paragraph, numbers, [source["file"], source["bound"]]


# ---- (e) cost -----------------------------------------------------------------------------------------------------
CONFIGURATIONS = [  # key in cost.json, text, the row it is a step from
    ("paper", "Newtonian, paper (Monaghan-Gingold, EPS factor 0.0025, simple walls)", None),
    ("paper_adami", "control: paper with adami walls", "paper"),
    ("thesis", "Newtonian, thesis (laplacian, EPS factor 1e-6, adami walls)", "paper_adami"),
    ("thesis_micropolar", "micropolar, thesis", "thesis"),
]
INTERVALS = [  # column, the timestamps summed
    ("predict", ["a_predict_end"]), ("voxel update", ["a_voxel_end"]),
    ("correction + density", ["b_correction_density_interior_end"]), ("wall extrapolation", ["b_wall_extrapolate_end"]),
    ("force", ["b_force_deep_interior_end"]),
    ("phase gaps and phase C", ["a_end", "b_start", "b_end", "c_start", "c_correction_density_boundary_end",
                                "c_density_end", "c_force_end", "c_end"]),
]


def cost() -> tuple:
    source = SOURCE["cost"]
    data = style.load_json(source["file"])
    header = ["size", "configuration", "particles", "trials", "steps/s by trial", "steps/s mean", "steps/s std",
              "M particle-steps/s mean", "M particle-steps/s std", "ratio to the paper row, trial by trial: mean",
              "ratio to the paper row: std", "step from the row above, trial by trial: mean", "step: std",
              "step: trials"]
    header += [f"{name} (microseconds per step, traced run)" for name, _ in INTERVALS]
    header += ["sum of the intervals (microseconds per step)", "device-local buffers (MB)", "defrag scratch (MB)",
               "buffers", "bytes per particle", "bytes per pool slot",
               "K = 2: steps/s over the sum of the simultaneous K = 1 runs: mean", "K = 2 efficiency: std"]
    tex_header = ["size", "configuration", "steps/s", r"$10^6$ particle-steps/s", "ratio to paper", "predict",
                  "corr.\\ + density", "wall", "force", "sum", "bytes per particle"]
    plain, tex, numbers = [], [], {}
    for size in ("1m", "4m"):
        if tex:
            tex.append("rule")
        for key, text, previous in CONFIGURATIONS:
            item = data["throughput"][f"{size}_{key}"]
            trace = data["kernels"][f"{size}_{key}"]
            intervals = [sum(trace["intervals_microseconds"].get(name, 0.0) for name in names)
                         if any(name in trace["intervals_microseconds"] for name in names) else None
                         for _, names in INTERVALS]
            step = None
            if previous:
                before = data["throughput"][f"{size}_{previous}"]
                shared = [trial for trial in item["trials"] if trial in before["trials"]]
                ratios = np.array([item["steps_per_second"][item["trials"].index(trial)]
                                   / before["steps_per_second"][before["trials"].index(trial)] for trial in shared])
                step = {"mean": float(ratios.mean()), "std": float(ratios.std(ddof=1)), "trials": shared}
            cells = [size.upper(), text, f"{item['particles']:,}", str(len(item["trials"])),
                     " ".join(f"{value:.1f}" for value in item["steps_per_second"]),
                     f"{item['steps_per_second_mean']:.1f}", f"{item['steps_per_second_std']:.1f}",
                     f"{item['particle_steps_per_second_mean'] / 1e6:.1f}",
                     f"{item['particle_steps_per_second_std'] / 1e6:.1f}",
                     f"{item['ratio_to_paper']:.4f}", f"{item['ratio_to_paper_std']:.4f}"]
            cells += ([f"{step['mean']:.4f}", f"{step['std']:.4f}", " ".join(str(trial) for trial in step["trials"])]
                      if step else ["-", "-", "-"])
            cells += ["-" if value is None else f"{value:.1f}" for value in intervals]
            cells += [f"{trace['sum_microseconds']:.1f}", f"{item['device_local_megabytes']:.1f}",
                      f"{item['defrag_scratch_megabytes']:.1f}",
                      f"{item['device_local_buffers']} + {item['defrag_scratch_buffers']}",
                      f"{item['bytes_per_particle']:.1f}", f"{item['bytes_per_slot']:.0f}", "-", "-"]
            plain.append(cells)
            tex.append([cells[0], style.tex_escape(text.split(" (")[0]),
                        f"{cells[5]} $\\pm$ {cells[6]}", f"{cells[7]} $\\pm$ {cells[8]}",
                        "1" if previous is None else f"{cells[9]} $\\pm$ {cells[10]}",
                        cells[14], cells[16], cells[17].replace("-", "--"), cells[18], cells[20], cells[24]])
            numbers[f"{size}_{key}"] = {
                "steps_per_second_mean": item["steps_per_second_mean"],
                "steps_per_second_std": item["steps_per_second_std"],
                "particle_steps_per_second_mean": item["particle_steps_per_second_mean"],
                "ratio_to_paper": item["ratio_to_paper"], "ratio_to_paper_std": item["ratio_to_paper_std"],
                "step_from_the_row_above": step, "bytes_per_particle": item["bytes_per_particle"],
                "intervals_microseconds": dict(zip((name for name, _ in INTERVALS), intervals)),
                "sum_microseconds": trace["sum_microseconds"], "drift": item["drift"]}
    pair_key, pair = next(iter(data["pair"].items()))
    megabytes = sum(pair["megabytes_per_slab"])
    cards = " + ".join(name.replace("NVIDIA GeForce ", "").replace("AMD Radeon ", "") for name in pair["devices"])
    cells = ["4M", f"micropolar, thesis, K = 2 on {cards} (weights {pair['weights'][0]:.4f} : {pair['weights'][1]:.4f})",
             f"{pair['particles']:,}", str(len(pair["trials"])),
             " ".join(f"{trial['chain_steps_per_second']:.1f}" for trial in pair["trials"]),
             f"{pair['steps_per_second_mean']:.1f}", f"{pair['steps_per_second_std']:.1f}",
             f"{pair['steps_per_second_mean'] * pair['particles'] / 1e6:.1f}",
             f"{pair['steps_per_second_std'] * pair['particles'] / 1e6:.1f}"]
    cells += ["-"] * (len(header) - len(cells) - 7)
    cells += [" + ".join(f"{value:.1f}" for value in pair["megabytes_per_slab"]), "-", "-",
              f"{megabytes * 1024.0 ** 2 / pair['particles']:.1f}", "-", f"{pair['efficiency_mean']:.3f}",
              f"{pair['efficiency_std']:.3f}"]
    plain.append(cells)
    tex.append("rule")
    tex.append(["4M", r"micropolar, thesis, $K = 2$", f"{cells[5]} $\\pm$ {cells[6]}", f"{cells[7]} $\\pm$ {cells[8]}",
                "--", "--", "--", "--", "--", "--", cells[-4]])
    numbers[pair_key] = {"steps_per_second_mean": pair["steps_per_second_mean"],
                         "efficiency_mean": pair["efficiency_mean"], "efficiency_std": pair["efficiency_std"],
                         "weights": pair["weights"], "megabytes_per_slab": pair["megabytes_per_slab"],
                         "trials": pair["trials"]}
    style.write_table("tab_14e_cost", header, plain, tex_header, "ll" + "r" * 9, tex,
                      note="steps/s: mean and standard deviation of the trials; intervals in microseconds per step "
                           "from one traced run per row; K = 2 row: "
                           f"efficiency {pair['efficiency_mean']:.3f} +- {pair['efficiency_std']:.3f} against the sum "
                           "of the two K = 1 runs made at the same time")

    def step_text(key):
        return " and ".join(f"{numbers[f'{size}_{key}']['step_from_the_row_above']['mean']:.3f}" for size in ("1m", "4m"))

    def force(key, size):
        return numbers[f"{size}_{key}"]["intervals_microseconds"]["force"]

    def added(size):
        """Share of the step time the micropolar terms add that sits in the force kernel and in predict."""
        thesis, micropolar = numbers[f"{size}_thesis"], numbers[f"{size}_thesis_micropolar"]
        kernels = sum(micropolar["intervals_microseconds"][name] - thesis["intervals_microseconds"][name]
                      for name in ("force", "predict"))
        return kernels / (micropolar["sum_microseconds"] - thesis["sum_microseconds"])
    def levels(key):
        """The two levels of the trials of one row: (trials on the lower level, upper mean / lower mean - 1,
        1 - lower mean / upper mean)."""
        values = np.array(data["throughput"][key]["steps_per_second"])
        low = values < 0.5 * (values.min() + values.max())
        upper, lower = values[~low].mean(), values[low].mean()
        return {"trials_on_the_lower_level": [trial for trial, flag in zip(data["throughput"][key]["trials"], low)
                                              if flag],
                "apart": float(upper / lower - 1.0), "lower_below_upper": float(1.0 - lower / upper)}
    runs = sum(len(item["trials"]) for item in data["throughput"].values())
    drift = sum(sum(item["drift"]) for item in data["throughput"].values())
    numbers["levels"] = {key: levels(key) for key in ("1m_paper", "4m_paper")}
    low_one, low_four = numbers["levels"]["1m_paper"], numbers["levels"]["4m_paper"]
    memory = numbers["1m_thesis_micropolar"]["bytes_per_particle"] / numbers["1m_thesis"]["bytes_per_particle"] - 1.0
    best = pair["steps_per_second_mean"] / max(trial["reference_steps_per_second"][pair["devices"][0]]
                                               for trial in pair["trials"])
    paragraph = (
        f"(e) tab_14e_cost (design note Section {source['sections']}; data in the tree since `{source['tree']}`). "
        f"Lid-driven cavity with {data['throughput']['1m_paper']['particles']:,} and "
        f"{data['throughput']['4m_paper']['particles']:,} particles on one RTX 5090 (the rig's display card), K = 1 "
        f"chain bench at depth 2, warm-up {data['protocol']['1m']['warmup']:,} steps, then "
        f"{data['protocol']['1m']['measured']:,} (1M) and {data['protocol']['4m']['measured']:,} (4M) measured steps, "
        f"trials interleaved; drift {drift} in the {runs} runs of the table. Steps trial by trial, 1M and 4M: paper to "
        f"paper with adami walls {step_text('paper_adami')}; to the thesis configuration (form and EPS factor) "
        f"{step_text('thesis')}; thesis to micropolar {step_text('thesis_micropolar')}, so the micropolar terms cost "
        f"{100.0 * (1.0 - numbers['4m_thesis_micropolar']['step_from_the_row_above']['mean']):.0f} % of the "
        f"throughput of the thesis configuration: {numbers['1m_thesis_micropolar']['steps_per_second_mean']:.0f} "
        f"steps/s at 1M and {numbers['4m_thesis_micropolar']['steps_per_second_mean']:.0f} at 4M "
        f"({numbers['1m_thesis_micropolar']['particle_steps_per_second_mean'] / 1e6:.0f} and "
        f"{numbers['4m_thesis_micropolar']['particle_steps_per_second_mean'] / 1e6:.0f} M particle-steps/s). The "
        f"trials of a row sit on two levels (paper row: {100.0 * low_one['apart']:.1f} % apart at 1M, trials "
        f"{', '.join(str(trial) for trial in low_one['trials_on_the_lower_level'])} on the lower one; at 4M trial "
        f"{', '.join(str(trial) for trial in low_four['trials_on_the_lower_level'])} is "
        f"{100.0 * low_four['lower_below_upper']:.1f} % low; cause not found), which is what the standard "
        f"deviations of the rows show; the ratios hold within a trial. Traced runs: the force kernel takes "
        f"{100.0 * (force('thesis_micropolar', '1m') / force('thesis', '1m') - 1.0):.1f} % (1M) and "
        f"{100.0 * (force('thesis_micropolar', '4m') / force('thesis', '4m') - 1.0):.1f} % (4M) more with the "
        f"micropolar terms; force and predict together are {100.0 * added('1m'):.0f} % and {100.0 * added('4m'):.0f} % "
        f"of the added step time. Memory: +{100.0 * memory:.0f} % per particle "
        f"({numbers['1m_thesis']['bytes_per_particle']:.0f} to "
        f"{numbers['1m_thesis_micropolar']['bytes_per_particle']:.0f} bytes). K = 2 on RTX 5090 + RX 7900 XTX, 4M "
        f"micropolar, weights {pair['weights'][0]:.2f} : {pair['weights'][1]:.2f}: "
        f"{pair['steps_per_second_mean']:.1f} steps/s, {pair['efficiency_mean']:.3f} +- {pair['efficiency_std']:.3f} "
        f"of the sum of the two K = 1 runs made at the same time (three trials), {best:.2f} times the RTX 5090 alone.")
    return paragraph, numbers, [source["file"]]


def main() -> int:
    parts = [operator_checks(), passive_label(), chain_against_one_slab(), step_limits(), cost()]
    stems = ["tab_14a_operator_checks", "tab_14b_passive_label", "tab_14c_chain_against_one_slab",
             "tab_14d_step_limits", "tab_14e_cost"]
    caption = ("Solver checks of design note Sections 8 to 12, as they stand; every number is read from the data "
               "files under data/micropolar. Existing data of Stages 1 to 3, not rerun on thesis-ch5-rc2; the cards "
               "are one RTX 5090 and one RX 7900 XTX.\n\n" + "\n\n".join(part[0] for part in parts))
    numbers = dict(zip(("a", "b", "c", "d", "e"), (part[1] for part in parts)))
    sources = [path for part in parts for path in part[2]]
    style.record("14", "Solver checks: operator checks, passive label, chain against one slab, step limits, cost",
                 SOURCE["status"], caption, [f"{stem}.{suffix}" for stem in stems for suffix in ("csv", "tex")],
                 sources, numbers, "Stages 1 to 3, `micropolar-base-9-g278a557` to `thesis-ch5-rc1`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
