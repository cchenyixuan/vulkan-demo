"""
micropolar_stage4.py - Stage 4 of docs/micropolar_design.md: the production runs between concentric cylinders
(resolution study, sweep over K and m), the eccentric pilot, and their analysis. The cases are generated and run by
micropolar_couette.py (generate / run); the wall sums are micropolar_wall_torque.py.

  list       The run table: name, section, generator options, control.
  queue      Generate (if absent) and run the named runs one after the other on one device; a run with a run.json
             is skipped. Every run saves its wall states (--state-count 24) and records its commit.
  analyse    Fits, D against the exact solution and the wall-displacement rule, orders, wall offsets.

All walls carry w = 0 (the rotor included). Thesis configuration everywhere (the generator's defaults).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CASES = "cases/couette_stage4"
RUNS_DIRECTORY = "logs/micropolar/stage4"
STEP_OPTIONS = ["--step-number", "0.0328", "--step-number-angular", "0.0334"]
LATTICES = {"": [], "_shifted": ["--lattice-offset", "0.5", "0.5"], "_rotated": ["--lattice-angle", "15"]}
STATE_COUNT = 24


def _ratio_tag(coupling_ratio: float) -> str:
    return "k" + f"{coupling_ratio:g}".replace(".", "")


def _build_table() -> dict:
    runs = {}

    def newtonian(name, gap, step_from, lattice="", eccentricity=0.0, section=()):
        runs[name] = {
            "generate": ["--gap-in-spacings", str(gap), "--eccentricity-in-spacings", str(eccentricity),
                         "--step-from-coupling-ratio", str(step_from[0]),
                         "--step-from-gap-over-two-lengths", str(step_from[1]), *LATTICES[lattice]],
            "control": None, "gap": gap, "eccentricity": eccentricity, "section": list(section)}

    def micropolar(name, gap, coupling_ratio, gap_over_two_lengths, control, eccentricity=0.0, section=()):
        runs[name] = {
            "generate": ["--gap-in-spacings", str(gap), "--eccentricity-in-spacings", str(eccentricity),
                         "--coupling-ratio", str(coupling_ratio), "--gap-over-two-lengths", str(gap_over_two_lengths),
                         "--rotor-micro-rotation-factor", "0"],
            "control": control, "gap": gap, "eccentricity": eccentricity, "coupling_ratio": coupling_ratio,
            "gap_over_two_lengths": gap_over_two_lengths, "section": list(section)}

    # Newtonian controls. Below g = 400 dx the acoustic limit sets dt for every K of the table, so one control
    # per lattice serves all of them (its step limits are taken from K = 3, the most restrictive). At g = 400 dx
    # the viscous limit of K = 3 is below the acoustic one: K = 3 has its own control.
    for gap in (50, 100, 200):
        for lattice in LATTICES:
            newtonian(f"g{gap:03d}_newtonian{lattice}", gap, (3, 5), lattice, section=("floor",))
    newtonian("g400_newtonian", 400, (1, 40), section=("control",))
    newtonian("g400_newtonian_k3", 400, (3, 10), section=("control",))

    # section 1, resolution study: K = 3, m = 5 and m = 10 at l / dx = 5, 10, 20
    for gap, gap_over_two_lengths in ((50, 5), (100, 5), (200, 5), (100, 10), (200, 10), (400, 10)):
        control = "g400_newtonian_k3" if gap == 400 else f"g{gap:03d}_newtonian"
        micropolar(f"g{gap:03d}_k3_m{gap_over_two_lengths}", gap, 3, gap_over_two_lengths, control,
                   section=("resolution",))
    # section 2, sweep: K = 0.5, 1, 3 and m = 1, 2, 5 (g = 100 dx), 10, 20 (200 dx), 40 (400 dx)
    for coupling_ratio in (3, 1, 0.5):
        for gap, gap_over_two_lengths in ((100, 1), (100, 2), (100, 5), (200, 10), (200, 20), (400, 40)):
            name = f"g{gap:03d}_{_ratio_tag(coupling_ratio)}_m{gap_over_two_lengths}"
            control = f"g{gap:03d}_newtonian"
            if gap == 400 and coupling_ratio == 3:
                control = "g400_newtonian_k3"
            if name in runs:
                runs[name]["section"].append("sweep")
            else:
                micropolar(name, gap, coupling_ratio, gap_over_two_lengths, control, section=("sweep",))

    # section 4, eccentric pilot: g = 100 dx, the rotor axis at x = eps g, K = 3, l = 10 dx (m = 5) and 25 dx
    # (m = 2); noise floor from the three lattices at eps = 0.6
    for eccentricity in (30, 60):
        for lattice in (LATTICES if eccentricity == 60 else [""]):
            newtonian(f"e{eccentricity}_newtonian{lattice}", 100, (3, 5), lattice, eccentricity,
                      section=("eccentric",))
        for gap_over_two_lengths in (5, 2):
            micropolar(f"e{eccentricity}_k3_m{gap_over_two_lengths}", 100, 3, gap_over_two_lengths,
                       f"e{eccentricity}_newtonian", eccentricity, section=("eccentric",))
    return runs


RUNS = _build_table()


def sample_steps_for(steps: int) -> int:
    """Every 1,000 steps up to 300 samples; beyond that the smallest multiple of 1,000 that divides the step count
    and gives at most 180 samples (the sample is a readback of the whole slab)."""
    if steps <= 300_000:
        return 1000
    for multiple in range(1, steps // 1000 + 1):
        if steps % (1000 * multiple) == 0 and steps // (1000 * multiple) <= 180:
            return 1000 * multiple
    return steps


def command_list(arguments) -> int:
    for name, entry in RUNS.items():
        print(f"{name:<28}{','.join(entry['section']):<22}control {entry['control'] or '-':<24}"
              f"{' '.join(entry['generate'])}")
    return 0


def ensure_case(name: str) -> dict:
    case_dir = _REPO_ROOT / CASES / name
    if not (case_dir / "couette.json").exists():
        code = subprocess.run([sys.executable, "-m", "experiment.validation.micropolar_couette", "generate",
                               "--out", f"{CASES}/{name}", *STEP_OPTIONS, *RUNS[name]["generate"]],
                              cwd=_REPO_ROOT).returncode
        if code:
            raise RuntimeError(f"generate {name}: exit {code}")
    return json.loads((case_dir / "couette.json").read_text(encoding="utf-8"))


def command_queue(arguments) -> int:
    (_REPO_ROOT / RUNS_DIRECTORY).mkdir(parents=True, exist_ok=True)
    for name in arguments.names:
        if name not in RUNS:
            print(f"[stage4] {name}: not in the table", flush=True)
            continue
        run_dir = _REPO_ROOT / RUNS_DIRECTORY / name
        if (run_dir / "run.json").exists():
            print(f"[stage4] {name}: done already", flush=True)
            continue
        meta = ensure_case(name)
        steps = arguments.steps if arguments.steps else meta["steps"]
        sample_steps = arguments.sample_steps if arguments.sample_steps else sample_steps_for(steps)
        band = 1000.0 if RUNS[name]["eccentricity"] else 2.0          # eccentric runs keep the whole fluid
        command = [sys.executable, "-m", "experiment.validation.micropolar_couette", "run", "--case",
                   f"{CASES}/{name}", "--out", f"{arguments.out}/{name}", "--device-map", str(arguments.device),
                   "--sample-steps", str(sample_steps), "--state-count", str(STATE_COUNT),
                   "--state-band-supports", str(band)]
        if arguments.steps:
            command += ["--steps", str(arguments.steps)]
        with open(_REPO_ROOT / arguments.out / f"{name}.log", "w", encoding="utf-8") as log:
            code = subprocess.run(command, cwd=_REPO_ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"[stage4] {name}: exit {code}", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------------------------
# analysis of the concentric runs
# ---------------------------------------------------------------------------------------------------------------
DENSITY = 1000.0


def wall_displacement(meta: dict) -> float:
    """delta = c l / (2 nu) (= kappa l / (2 (mu + kappa))): the distance a wall at rest with w = 0 stands further
    into the gap for a Newtonian fluid of viscosity mu + kappa / 2 (thesis section 5.4)."""
    return meta["design_coupling"] * meta["micropolar_length"] / (2.0 * meta["design_viscosity"])


def rule_torque_change(inner_radius: float, outer_radius: float, displacement: float) -> float:
    """D of the displacement rule for the rotor at Omega with w = 0 and the stator at rest with w = 0:
    beta_N(R1 + delta (R1 / R2)^2, R2 - delta) / beta_N(R1, R2) - 1, beta_N(a, b) = Omega a^2 b^2 / (b^2 - a^2)."""
    def potential_vortex(inner, outer):
        return inner ** 2 * outer ** 2 / (outer ** 2 - inner ** 2)
    return (potential_vortex(inner_radius + displacement * (inner_radius / outer_radius) ** 2,
                             outer_radius - displacement) / potential_vortex(inner_radius, outer_radius) - 1.0)


def _root(function, centre: float, half_width: float) -> float:
    from scipy.optimize import brentq
    low, high = centre - half_width, centre + half_width
    if function(low) * function(high) > 0.0:
        return float("nan")
    return float(brentq(function, low, high, xtol=1e-14, rtol=1e-14))


def felt_walls(item: dict, profile: dict, fit: dict, reference, margin: float) -> dict:
    """Where the fitted profiles reach the wall values: v_fit = Omega r near R1 and 0 near R2 (the velocity-only
    fit on the four functions), w_fit = the wall value of the rotor and of the stator (a fit of w alone on its
    three functions over the same bins); and the same two pairs from the joint fit. Radii in metres."""
    meta = item["meta"]
    inner, outer, length = meta["inner_radius"], meta["outer_radius"], meta["micropolar_length"]
    radius = profile["radius"]
    inside = ((radius > inner + margin * meta["smoothing_length"])
              & (radius < outer - margin * meta["smoothing_length"]))
    spin_basis = reference.micro_rotation_basis_functions(radius[inside], length, inner, outer)
    scale = np.sqrt((spin_basis ** 2).sum(axis=0))
    spin_coefficients = np.linalg.lstsq(spin_basis / scale, profile["spin"][inside], rcond=None)[0] / scale
    velocity_coefficients = np.asarray(fit["coefficients"])
    joint = np.asarray(fit["joint_coefficients"])
    factor = meta["design_coupling"] * length / meta["design_viscosity"]

    def velocity_of(coefficients):
        return lambda r: float(reference.velocity_basis_functions([r], length, inner, outer)[0] @ coefficients)

    def spin_of(coefficients):
        return lambda r: float(reference.micro_rotation_basis_functions([r], length, inner, outer)[0] @ coefficients)
    joint_velocity = velocity_of(np.array([joint[0], joint[1], factor * joint[2], -factor * joint[3]]))
    joint_spin = spin_of(np.array([joint[0], joint[2], joint[3]]))
    rate, rotor_spin = meta["rotation_rate"], meta["rotor_micro_rotation"]
    half_width = 4.0 * meta["spacing"]
    result = {"spin_only_coefficients": spin_coefficients.tolist()}
    for label, velocity_function, spin_function in (("separate", velocity_of(velocity_coefficients),
                                                     spin_of(spin_coefficients)),
                                                    ("joint", joint_velocity, joint_spin)):
        result[label] = {
            "velocity_inner": _root(lambda r: velocity_function(r) - rate * r, inner, half_width),
            "velocity_outer": _root(velocity_function, outer, half_width),
            "spin_inner": _root(lambda r: spin_function(r) - rotor_spin, inner, half_width),
            "spin_outer": _root(spin_function, outer, half_width)}
    return result


def analyse_concentric(names, runs_directory: pathlib.Path, margin: float = 1.0) -> dict:
    from experiment.validation import micropolar_couette as couette
    from experiment.validation import micropolar_couette_reference as reference
    loaded = {name: couette._load_run(name, runs_directory) for name in names}
    results = {}
    for name, item in loaded.items():
        meta, run = item["meta"], item["run"]
        end_time = run["steps"] * run["timestep"]
        whole = couette._profile(item, 0.75 * end_time, end_time)
        fit = couette._fit(item, whole, reference, margin)
        window_fits = [couette._fit(item, couette._profile(item, (index / 8.0) * end_time,
                                                           ((index + 1) / 8.0) * end_time), reference, margin)
                       for index in range(8)]
        windows = [entry["potential_vortex_constant"] for entry in window_fits]
        joint_windows = [entry.get("joint_potential_vortex_constant") for entry in window_fits]
        sectors = [couette._fit(item, couette._profile(item, 0.75 * end_time, end_time, sector=sector), reference,
                                margin)["potential_vortex_constant"] for sector in range(couette.SECTOR_COUNT)]
        record = {
            "name": name, "micropolar": meta["micropolar"], "gap_in_spacings": round(meta["gap"] / meta["spacing"]),
            "sections": RUNS[name]["section"] if name in RUNS else [],
            "governing": meta["governing"], "timestep": run["timestep"], "steps": run["steps"],
            "particles": run["particles"], "device_map": run["device_map"],
            "steps_per_second": run["steps_per_second"], "elapsed_seconds": run["elapsed_seconds"],
            "revision": run.get("revision"), "failure": run["failure"],
            "counters_zero": not any(run["counters"].values()), "samples_in_window": whole["samples"],
            "fit": fit, "window_constants": windows, "joint_window_constants": joint_windows,
            "steady_change": float(abs(windows[7] - windows[6]) / abs(windows[7])),
            "joint_steady_change": (float(abs(joint_windows[7] - joint_windows[6]) / abs(joint_windows[7]))
                                    if joint_windows[7] is not None else None),
            "sector_spread": float((max(sectors) - min(sectors)) / abs(np.mean(sectors))),
            "sector_fourfold": float(abs(np.mean(sectors[0::2]) - np.mean(sectors[1::2])) / 2.0
                                     / abs(np.mean(sectors))),
            "radial_velocity_max_over_wall_speed": float(np.abs(whole["radial_velocity"]).max() / meta["wall_speed"]),
            "density_range": [float(item["data"]["density_min"].min()), float(item["data"]["density_max"].max())],
            "profile": {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                        for key, value in whole.items()},
        }
        if not meta["micropolar"]:
            alpha, beta = fit["rigid_rotation_rate"], fit["potential_vortex_constant"]
            rate = meta["rotation_rate"]
            record["hydrodynamic_inner_radius"] = float(np.sqrt(beta / (rate - alpha)))
            record["hydrodynamic_outer_radius"] = float(np.sqrt(-beta / alpha))
            record["inner_offset_in_spacings"] = (record["hydrodynamic_inner_radius"] - meta["inner_radius"]) \
                / meta["spacing"]
            record["outer_offset_in_spacings"] = (record["hydrodynamic_outer_radius"] - meta["outer_radius"]) \
                / meta["spacing"]
            nominal = reference.newtonian_couette_constants(meta["inner_radius"], meta["outer_radius"],
                                                            meta["wall_speed"], 0.0)
            record["nominal_constants"] = list(nominal)
            record["beta_over_nominal"] = beta / nominal[1]
        results[name] = record
        results[name]["_whole"] = whole

    # noise floor per gap: beta of the three lattices
    floors = {}
    for gap in sorted({record["gap_in_spacings"] for record in results.values()}):
        members = [f"g{gap:03d}_newtonian{lattice}" for lattice in LATTICES]
        if all(member in results for member in members):
            betas = np.array([results[member]["fit"]["potential_vortex_constant"] for member in members])
            floors[gap] = {"beta": betas.tolist(),
                           "relative_spread": float((betas.max() - betas.min()) / abs(betas.mean())),
                           "relative_std": float(betas.std(ddof=1) / abs(betas.mean()))}

    for name, record in results.items():
        control_name = RUNS[name]["control"] if name in RUNS else None
        if not record["micropolar"] or control_name not in results:
            continue
        item = loaded[name]
        meta = item["meta"]
        control = results[control_name]
        beta = control["fit"]["potential_vortex_constant"]
        eringen = reference.eringen_parameters_from_solver_parameters(
            DENSITY, meta["design_viscosity"], meta["design_coupling"], meta["design_viscosity_rotation"])
        rate = meta["rotation_rate"]
        exact = {}
        for label, inner, outer in (("nominal", meta["inner_radius"], meta["outer_radius"]),
                                    ("hydrodynamic", control["hydrodynamic_inner_radius"],
                                     control["hydrodynamic_outer_radius"])):
            solution = reference.solve_micropolar_couette(
                inner, outer, eringen.shear_viscosity, eringen.coupling_viscosity, eringen.spin_viscosity,
                rate * inner, 0.0, meta["rotor_micro_rotation"], 0.0)
            exact[label] = (solution, reference.relative_torque_change_from_constant(
                solution.potential_vortex_constant, inner, outer, rate * inner, 0.0))
        solution, exact_change = exact["hydrodynamic"]
        coupling_ratio = meta["coupling_ratio"]
        length = meta["micropolar_length"]
        displacement = wall_displacement(meta)
        record.update({
            "control": control_name,
            "same_timestep_as_control": bool(abs(record["timestep"] / control["timestep"] - 1.0) < 1e-12),
            "coupling_ratio": coupling_ratio, "gap_over_two_lengths": meta["gap"] / (2.0 * length),
            "length_in_spacings": length / meta["spacing"], "support_over_length": meta["smoothing_length"] / length,
            "reynolds_far_viscosity": meta["reynolds"],
            "reynolds_shear_viscosity": meta["reynolds"] * (1.0 + 0.5 * coupling_ratio),
            "relative_torque_change": record["fit"]["potential_vortex_constant"] / beta - 1.0,
            "relative_torque_change_joint": record["fit"]["joint_potential_vortex_constant"] / beta - 1.0,
            "exact_nominal": exact["nominal"][1], "exact_hydrodynamic": exact_change,
            "exact_potential_vortex_constant": solution.potential_vortex_constant,
            "wall_displacement_in_spacings": displacement / meta["spacing"],
            "rule_nominal": rule_torque_change(meta["inner_radius"], meta["outer_radius"], displacement),
            "rule_hydrodynamic": rule_torque_change(control["hydrodynamic_inner_radius"],
                                                    control["hydrodynamic_outer_radius"], displacement),
        })
        record["difference"] = record["relative_torque_change"] - exact_change
        record["difference_joint"] = record["relative_torque_change_joint"] - exact_change
        record["sph_over_exact"] = record["relative_torque_change"] / exact_change
        record["sph_joint_over_exact"] = record["relative_torque_change_joint"] / exact_change
        record["rule_over_exact"] = record["rule_hydrodynamic"] / exact_change
        gap = record["gap_in_spacings"]
        floor_gap = gap if gap in floors else (200 if gap == 400 and 200 in floors else None)
        floor = floors.get(floor_gap)
        record["noise_floor_gap"] = floor_gap
        record["noise_floor_of_change"] = float(floor["relative_spread"] * (1.0 + exact_change)) if floor else None
        # profile errors against the exact solution with the hydrodynamic radii of the control
        profile = record["_whole"]
        radius = profile["radius"]
        valid = (radius > control["hydrodynamic_inner_radius"]) & (radius < control["hydrodynamic_outer_radius"])
        exact_velocity = reference.velocity(solution, radius[valid])
        exact_spin = reference.micro_rotation(solution, radius[valid])
        core = ((radius[valid] > meta["inner_radius"] + meta["smoothing_length"])
                & (radius[valid] < meta["outer_radius"] - meta["smoothing_length"]))

        def relative(measured, expected, mask):
            return float(np.sqrt(((measured[mask] - expected[mask]) ** 2).mean() / (expected ** 2).mean()))
        record["velocity_error_core"] = relative(profile["velocity"][valid], exact_velocity, core)
        record["velocity_error_wall_layers"] = relative(profile["velocity"][valid], exact_velocity, ~core)
        record["spin_error_core"] = relative(profile["spin"][valid], exact_spin, core)
        record["spin_error_wall_layers"] = relative(profile["spin"][valid], exact_spin, ~core)
        record["spin_extremum_over_rotation_rate"] = float(profile["spin"][np.argmax(np.abs(profile["spin"]))] / rate)
        record["exact_profile"] = {"radius": radius[valid].tolist(), "velocity": exact_velocity.tolist(),
                                   "spin": exact_spin.tolist()}
        felt = felt_walls(item, profile, record["fit"], reference, margin)
        record["felt_walls"] = felt
        spacing = meta["spacing"]
        for label in ("separate", "joint"):
            entry = felt[label]
            record[f"spin_minus_velocity_inner_in_spacings_{label}"] = \
                (entry["spin_inner"] - entry["velocity_inner"]) / spacing
            record[f"spin_minus_velocity_outer_in_spacings_{label}"] = \
                (entry["spin_outer"] - entry["velocity_outer"]) / spacing
        separate = felt["separate"]
        record["velocity_inner_offset_in_spacings"] = (separate["velocity_inner"] - meta["inner_radius"]) / spacing
        record["velocity_outer_offset_in_spacings"] = (separate["velocity_outer"] - meta["outer_radius"]) / spacing
        record["spin_inner_offset_in_spacings"] = (separate["spin_inner"] - meta["inner_radius"]) / spacing
        record["spin_outer_offset_in_spacings"] = (separate["spin_outer"] - meta["outer_radius"]) / spacing
    for record in results.values():
        record.pop("_whole")
    return {"fit_margin": margin, "noise_floors": {str(gap): floor for gap, floor in floors.items()},
            "runs": results}


def _order(coarse: float, fine: float) -> float:
    if coarse is None or fine is None or coarse <= 0.0 or fine <= 0.0:
        return float("nan")
    return float(np.log(coarse / fine) / np.log(2.0))


def command_analyse(arguments) -> int:
    runs_directory = _REPO_ROOT / arguments.runs
    names = [name for name in (arguments.names or RUNS)
             if name in RUNS and not RUNS[name]["eccentricity"] and (runs_directory / name / "run.json").exists()]
    analysis = analyse_concentric(names, runs_directory, arguments.fit_margin)
    results = analysis["runs"]
    lines = []

    def say(text=""):
        lines.append(text)
        print(text)
    say("Newtonian controls")
    say(f"{'run':<26}{'steps':>9}{'dt':>11}{'limit':>9}{'alpha':>11}{'beta':>13}{'beta/nom':>10}{'dR1/dx':>8}"
        f"{'dR2/dx':>8}{'steady':>9}{'sectors':>9}{'4-fold':>9}{'fit rms':>9}{'steps/s':>9}")
    for name, record in results.items():
        if record["micropolar"]:
            continue
        say(f"{name:<26}{record['steps']:>9,}{record['timestep']:>11.4e}{record['governing'][:8]:>9}"
            f"{record['fit']['rigid_rotation_rate']:>11.6f}{record['fit']['potential_vortex_constant']:>13.8f}"
            f"{record['beta_over_nominal']:>10.6f}{record['inner_offset_in_spacings']:>8.3f}"
            f"{record['outer_offset_in_spacings']:>8.3f}{record['steady_change']:>9.1e}{record['sector_spread']:>9.1e}"
            f"{record['sector_fourfold']:>9.1e}{record['fit']['fit_rms_over_wall_speed']:>9.1e}"
            f"{record['steps_per_second']:>9.1f}"
            + ("" if record["counters_zero"] and not record["failure"] else "  FAILED / counters"))
    say()
    for gap, floor in analysis["noise_floors"].items():
        say(f"noise floor at g = {gap} dx: beta spread (max - min) / mean = {floor['relative_spread']:.2e}, "
            f"std / mean = {floor['relative_std']:.2e}")
    say()
    say("micropolar runs (w = 0 on both walls); D_exact with the hydrodynamic radii of the control")
    say(f"{'run':<16}{'K':>5}{'m':>4}{'l/dx':>6}{'Re_mu':>7}{'limit':>9}{'D_SPH':>10}{'D_joint':>10}{'D_exact':>10}"
        f"{'D_nominal':>10}{'D_rule':>10}{'SPH/ex':>8}{'joint/ex':>9}{'rule/ex':>8}{'floor/D':>8}{'steady':>9}"
        f"{'j.steady':>9}{'v core':>9}{'v wall':>9}{'w core':>9}{'w wall':>9}{'steps/s':>9}")
    for name, record in results.items():
        if "relative_torque_change" not in record:
            continue
        floor = record["noise_floor_of_change"]
        say(f"{name:<16}{record['coupling_ratio']:>5g}{record['gap_over_two_lengths']:>4g}"
            f"{record['length_in_spacings']:>6g}{record['reynolds_shear_viscosity']:>7g}{record['governing'][:8]:>9}"
            f"{record['relative_torque_change']:>10.5f}{record['relative_torque_change_joint']:>10.5f}"
            f"{record['exact_hydrodynamic']:>10.5f}{record['exact_nominal']:>10.5f}{record['rule_hydrodynamic']:>10.5f}"
            f"{record['sph_over_exact']:>8.4f}{record['sph_joint_over_exact']:>9.4f}{record['rule_over_exact']:>8.4f}"
            + (f"{floor / record['exact_hydrodynamic']:>8.4f}" if floor else f"{'-':>8}")
            + f"{record['steady_change']:>9.1e}{record['joint_steady_change']:>9.1e}"
            f"{record['velocity_error_core']:>9.1e}{record['velocity_error_wall_layers']:>9.1e}"
            f"{record['spin_error_core']:>9.1e}{record['spin_error_wall_layers']:>9.1e}"
            f"{record['steps_per_second']:>9.1f}"
            + ("" if record["same_timestep_as_control"] else "  dt differs from the control")
            + ("" if record["counters_zero"] and not record["failure"] else "  FAILED / counters"))
    say()
    say("where the fitted profiles reach the wall values, in dx from the nominal radius (separate fits of v and w), "
        "and w minus v (separate / joint fit)")
    say(f"{'run':<16}{'v rotor':>9}{'w rotor':>9}{'w - v':>9}{'(joint)':>9}{'v stator':>9}{'w stator':>9}{'w - v':>9}"
        f"{'(joint)':>9}")
    for name, record in results.items():
        if "felt_walls" not in record:
            continue
        say(f"{name:<16}{record['velocity_inner_offset_in_spacings']:>9.3f}"
            f"{record['spin_inner_offset_in_spacings']:>9.3f}"
            f"{record['spin_minus_velocity_inner_in_spacings_separate']:>9.3f}"
            f"{record['spin_minus_velocity_inner_in_spacings_joint']:>9.3f}"
            f"{record['velocity_outer_offset_in_spacings']:>9.3f}{record['spin_outer_offset_in_spacings']:>9.3f}"
            f"{record['spin_minus_velocity_outer_in_spacings_separate']:>9.3f}"
            f"{record['spin_minus_velocity_outer_in_spacings_joint']:>9.3f}")
    # section 1: orders along the two series
    orders = {}
    for series, members in (("m = 5", ["g050_k3_m5", "g100_k3_m5", "g200_k3_m5"]),
                            ("m = 10", ["g100_k3_m10", "g200_k3_m10", "g400_k3_m10"])):
        present = [member for member in members if member in results and "difference" in results[member]]
        if len(present) < 2:
            continue
        say()
        say(f"resolution series {series}: deviation of D and profile errors against h / l, observed orders between "
            f"neighbours")
        say(f"{'run':<16}{'h/l':>6}{'dev D':>11}{'dev/D':>9}{'order':>7}{'joint dev/D':>12}{'order':>7}{'v core':>9}"
            f"{'order':>7}{'v wall':>9}{'order':>7}{'w core':>9}{'order':>7}{'w wall':>9}{'order':>7}")
        previous = None
        orders[series] = []
        for member in present:
            record = results[member]
            values = {"deviation": abs(record["difference"]) / record["exact_hydrodynamic"],
                      "deviation_joint": abs(record["difference_joint"]) / record["exact_hydrodynamic"],
                      "velocity_core": record["velocity_error_core"],
                      "velocity_wall": record["velocity_error_wall_layers"],
                      "spin_core": record["spin_error_core"], "spin_wall": record["spin_error_wall_layers"]}
            observed = {key: (_order(previous[key], values[key]) if previous else float("nan")) for key in values}
            orders[series].append({"run": member, "support_over_length": record["support_over_length"],
                                   "signed_deviation": record["difference"] / record["exact_hydrodynamic"],
                                   "signed_deviation_joint": record["difference_joint"] / record["exact_hydrodynamic"],
                                   "values": values, "orders": observed})
            say(f"{member:<16}{record['support_over_length']:>6.2f}{record['difference']:>+11.2e}"
                f"{record['difference'] / record['exact_hydrodynamic']:>+9.4f}{observed['deviation']:>7.2f}"
                f"{record['difference_joint'] / record['exact_hydrodynamic']:>+12.4f}{observed['deviation_joint']:>7.2f}"
                f"{values['velocity_core']:>9.1e}{observed['velocity_core']:>7.2f}{values['velocity_wall']:>9.1e}"
                f"{observed['velocity_wall']:>7.2f}{values['spin_core']:>9.1e}{observed['spin_core']:>7.2f}"
                f"{values['spin_wall']:>9.1e}{observed['spin_wall']:>7.2f}")
            previous = values
    analysis["resolution_orders"] = orders
    if arguments.json:
        target = _REPO_ROOT / arguments.json
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(analysis, indent=1), encoding="utf-8")
        target.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return 0


# ---------------------------------------------------------------------------------------------------------------
# section 3: force and torque of a wall from pair sums
# ---------------------------------------------------------------------------------------------------------------
def cached_wall_sums(name: str, runs_directory: pathlib.Path, recompute: bool = False) -> dict:
    """micropolar_wall_torque.run_sums over every saved state of a run, kept beside the run as wall_sums.json."""
    from experiment.validation import micropolar_wall_torque as wall_torque
    cache = runs_directory / name / "wall_sums.json"
    if cache.exists() and not recompute:
        return json.loads(cache.read_text(encoding="utf-8"))
    summary = wall_torque.run_sums(runs_directory / name)
    cache.write_text(json.dumps(summary), encoding="utf-8")
    return summary


def wall_torques(summary: dict) -> dict:
    """The three readings of the torque of each wall on the fluid, about the rotor axis, per unit depth.

    as_written   the formula of the task: every pair term of the solver between the fluid and the wall particles
                 (pressure with the solver's sign rule, viscous, c curl w, xi lap w, c curl v), the solver's
                 particle mass, arm at the fluid particle.
    wall         the exchange terms only (viscous, pressure in its flux form, xi lap w), arm at the pair
                 midpoint, the mass rho0 dx^2 a particle stands for, plus the wall term 2 pi R^2 (mu Omega_b +
                 kappa w_b) that turns the stress mu' grad v of the laplacian form into the symmetric one.
    circle       the same exchange terms across the circle at distance h from the wall inside the fluid, plus
                 - mu Gamma - kappa r^2 (integral of w over the angle) from the fields on that circle.
    Also the forces (exchange terms, rho0 dx^2), the parts by term, and the standard errors over the states."""
    parameters = summary["parameters"]
    factor = summary["mass_over_lattice_mass"]
    viscosity = parameters["viscosity"]
    coupling = parameters["coupling"] if parameters["micropolar"] else 0.0
    shear_viscosity = DENSITY * (viscosity - coupling)                    # mu
    coupling_viscosity = DENSITY * coupling                               # kappa
    inner_radius, rate = parameters["inner_radius"], parameters["rotation_rate"]
    result = {"mass_over_lattice_mass": factor, "states": summary["states"], "checks": summary["checks"],
              "shear_viscosity": shear_viscosity, "coupling_viscosity": coupling_viscosity}
    for wall in ("rotor", "stator"):
        entry, layer, circle = summary[wall], summary[wall]["layer"], summary[f"{wall}_layer"]
        wall_term = 2.0 * np.pi * inner_radius ** 2 * shear_viscosity * rate if wall == "rotor" else 0.0
        exchange = entry["torque_exchange"]["mean"] / factor
        circle_radius = layer["circle_radius"]["mean"]
        circle_exchange = circle["torque_exchange"]["mean"] / factor
        field_terms = (-shear_viscosity * layer["circulation"]["mean"]
                       - coupling_viscosity * circle_radius ** 2 * layer["spin_integral"]["mean"])
        # rotor: the pair sum across its circle is the outer fluid on the layer, the torque of the symmetric
        # stress across the circle is that plus the field terms, and the rotor acts on the fluid with minus it.
        # stator: the pair sum is the inner fluid on the layer = minus (outside on inside).
        circle_torque = -(circle_exchange + field_terms) if wall == "rotor" else (-circle_exchange + field_terms)
        result[wall] = {
            "as_written": entry["torque_total"]["mean"],
            "as_written_standard_error": entry["torque_total"]["standard_error"],
            "as_written_by_term": {name: values["mean"] for name, values in entry["torque"].items()
                                   if name != "pressure_flux"},
            "wall": exchange + wall_term,
            "wall_standard_error": entry["torque_exchange"]["standard_error"] / factor,
            "wall_by_term": {**{name: entry["torque_midpoint"][name]["mean"] / factor
                                for name in ("viscous", "pressure_flux", "spin_diffusion")
                                if name in entry["torque_midpoint"]}, "wall_term": wall_term},
            "not_exchange_by_term": {name: entry["torque_midpoint"][name]["mean"] / factor
                                     for name in ("coupling", "spin_coupling") if name in entry["torque_midpoint"]},
            "circle": circle_torque,
            "circle_standard_error": circle["torque_exchange"]["standard_error"] / factor,
            "circle_by_term": {"pair_sum": circle_exchange,
                               "circulation_term": -shear_viscosity * layer["circulation"]["mean"],
                               "spin_term": -coupling_viscosity * circle_radius ** 2 * layer["spin_integral"]["mean"]},
            "layer_total": layer["total"]["mean"] / factor,
            "force": (np.asarray(entry["force_exchange"]["mean"]) / factor).tolist(),
            "force_standard_error": (np.asarray(entry["force_exchange"]["standard_error"]) / factor).tolist(),
            "force_by_term": {name: (np.asarray(entry["force"][name]["mean"]) / factor).tolist()
                              for name in ("viscous", "pressure_flux", "pressure")},
            "force_as_written": entry["force_total"]["mean"],
        }
    return result


def command_torque(arguments) -> int:
    """The wall sums of the concentric runs against the torque of the fits (section 3)."""
    from experiment.validation import micropolar_couette_reference as reference
    runs_directory = _REPO_ROOT / arguments.runs
    names = [name for name in (arguments.names or RUNS)
             if name in RUNS and not RUNS[name]["eccentricity"] and (runs_directory / name / "run.json").exists()
             and (runs_directory / name / "states").exists()]
    analysis = analyse_concentric(names, runs_directory)["runs"]
    lines, records = [], {}

    def say(text=""):
        lines.append(text)
        print(text)
    say("torque of each wall on the fluid about the rotor axis, per unit depth, over the torque of the fit "
        "(4 pi mu beta; 2 pi (2 mu + kappa) C2 of the velocity-only fit); rotor positive, stator negative")
    say(f"{'run':<24}{'states':>7}{'fit torque':>12}{'as written':>22}{'wall form':>20}{'circle form':>20}"
        f"{'rotor+stator':>14}{'|F rotor|':>11}{'check a':>9}")
    say(f"{'':<24}{'':>7}{'':>12}{'rotor':>11}{'stator':>11}{'rotor':>10}{'stator':>10}{'rotor':>10}{'stator':>10}"
        f"{'wall form':>14}{'R1 / T':>11}{'rms':>9}")
    for name in names:
        summary = cached_wall_sums(name, runs_directory, arguments.recompute)
        torques = wall_torques(summary)
        record = analysis[name]
        constant = record["fit"]["potential_vortex_constant"]
        stress_factor = 2.0 * torques["shear_viscosity"] + torques["coupling_viscosity"]
        fit_torque = 2.0 * np.pi * stress_factor * constant
        entry = {"fit_torque": fit_torque, **torques}
        if record["micropolar"]:
            entry["fit_torque_joint"] = 2.0 * np.pi * stress_factor * record["fit"]["joint_potential_vortex_constant"]
            if "exact_potential_vortex_constant" in record:
                # the couple stress at each wall of the exact solution (hydrodynamic radii of the control)
                meta = json.loads((runs_directory / name / "run.json").read_text(encoding="utf-8"))["couette"]
                control = analysis[RUNS[name]["control"]]
                eringen = reference.eringen_parameters_from_solver_parameters(
                    DENSITY, meta["design_viscosity"], meta["design_coupling"], meta["design_viscosity_rotation"])
                inner, outer = control["hydrodynamic_inner_radius"], control["hydrodynamic_outer_radius"]
                solution = reference.solve_micropolar_couette(
                    inner, outer, eringen.shear_viscosity, eringen.coupling_viscosity, eringen.spin_viscosity,
                    meta["rotation_rate"] * inner, 0.0, meta["rotor_micro_rotation"], 0.0)
                exact_torque = -reference.torque_per_unit_length_from_constant(solution)
                slopes = reference.micro_rotation_derivative(solution, np.array([inner, outer]))
                entry["exact_torque"] = float(exact_torque)
                entry["exact_couple"] = {"rotor": float(-2.0 * np.pi * inner * eringen.spin_viscosity * slopes[0]),
                                         "stator": float(2.0 * np.pi * outer * eringen.spin_viscosity * slopes[1])}
        for wall, sign in (("rotor", 1.0), ("stator", -1.0)):
            for form in ("as_written", "wall", "circle"):
                entry[wall][f"{form}_over_fit"] = sign * entry[wall][form] / fit_torque
        entry["balance_wall"] = (entry["rotor"]["wall"] + entry["stator"]["wall"]) / fit_torque
        entry["balance_as_written"] = (entry["rotor"]["as_written"] + entry["stator"]["as_written"]) / fit_torque
        entry["balance_circle"] = (entry["rotor"]["circle"] + entry["stator"]["circle"]) / fit_torque
        meta_inner = summary["parameters"]["inner_radius"]
        entry["rotor_force_over_tangential_scale"] = float(np.hypot(*entry["rotor"]["force"]) * meta_inner / fit_torque)
        records[name] = entry
        say(f"{name:<24}{entry['states']:>7}{fit_torque:>12.4f}{entry['rotor']['as_written_over_fit']:>11.4f}"
            f"{entry['stator']['as_written_over_fit']:>11.4f}{entry['rotor']['wall_over_fit']:>10.4f}"
            f"{entry['stator']['wall_over_fit']:>10.4f}{entry['rotor']['circle_over_fit']:>10.4f}"
            f"{entry['stator']['circle_over_fit']:>10.4f}{entry['balance_wall']:>+14.4f}"
            f"{entry['rotor_force_over_tangential_scale']:>11.1e}"
            f"{entry['checks']['acceleration_rms_difference_over_rms']:>9.1e}")
    say()
    say("parts of the wall form (units of the fit torque) and the couple stress against the exact solution")
    say(f"{'run':<24}{'wall':>7}{'viscous':>9}{'pressure':>9}{'xi lap w':>9}{'wall term':>10}{'exact couple':>13}"
        f"{'c curl w':>10}{'c curl v':>10}{'s.e.':>9}")
    for name, entry in records.items():
        for wall, sign in (("rotor", 1.0), ("stator", -1.0)):
            parts = entry[wall]["wall_by_term"]
            other = entry[wall]["not_exchange_by_term"]
            scale = sign / entry["fit_torque"]
            say(f"{name:<24}{wall:>7}{parts['viscous'] * scale:>9.4f}{parts['pressure_flux'] * scale:>9.4f}"
                + (f"{parts['spin_diffusion'] * scale:>9.4f}" if "spin_diffusion" in parts else f"{'-':>9}")
                + f"{parts['wall_term'] * scale:>10.4f}"
                + (f"{entry['exact_couple'][wall] * scale:>13.4f}" if "exact_couple" in entry else f"{'-':>13}")
                + (f"{other['coupling'] * scale:>10.4f}{other['spin_coupling'] * scale:>10.4f}"
                   if "spin_coupling" in other else f"{'-':>10}{'-':>10}")
                + f"{entry[wall]['wall_standard_error'] / entry['fit_torque']:>9.1e}")
    # D from the wall sums: a micropolar run over its control, both forms and both walls
    say()
    say("D from the wall sums of a run and of its control, against D of the fits and the exact one")
    say(f"{'run':<24}{'D fit':>9}{'D exact':>9}{'wall: rotor':>12}{'stator':>9}{'circle: rotor':>14}{'stator':>9}")
    for name, entry in records.items():
        control_name = RUNS[name]["control"]
        if control_name not in records or "relative_torque_change" not in analysis[name]:
            continue
        control = records[control_name]
        entry["change"] = {f"{form}_{wall}": entry[wall][form] / control[wall][form] - 1.0
                           for form in ("wall", "circle") for wall in ("rotor", "stator")}
        say(f"{name:<24}{analysis[name]['relative_torque_change']:>9.5f}{analysis[name]['exact_hydrodynamic']:>9.5f}"
            f"{entry['change']['wall_rotor']:>12.5f}{entry['change']['wall_stator']:>9.5f}"
            f"{entry['change']['circle_rotor']:>14.5f}{entry['change']['circle_stator']:>9.5f}")
    if arguments.json:
        target = _REPO_ROOT / arguments.json
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(records, indent=1), encoding="utf-8")
        target.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return 0


# ---------------------------------------------------------------------------------------------------------------
# section 4: eccentric pilot
# ---------------------------------------------------------------------------------------------------------------
def wall_shear_profiles(name: str, runs_directory: pathlib.Path, angle_bins: int = 72, layer_spacings=(1.0, 15.0),
                        degree: int = 3) -> dict:
    """The tangential velocity of the fluid near the rotor as a polynomial in the wall distance, per angle bin
    about the rotor axis, from every saved state of a run (the eccentric runs keep the whole fluid): its slope at
    the wall and the vorticity there, omega = dv_t/dn + Omega (v_t = Omega R1 and v_n = 0 along the wall)."""
    from experiment.validation import micropolar_wall_torque as wall_torque
    run = json.loads((runs_directory / name / "run.json").read_text(encoding="utf-8"))
    meta = run["couette"]
    spacing, inner_radius, rate = meta["spacing"], meta["inner_radius"], meta["rotation_rate"]
    count = degree + 1
    matrix = np.zeros((angle_bins, count, count))
    right = np.zeros((angle_bins, count))
    for path in wall_torque.state_files(runs_directory / name):
        state = wall_torque.load_state(path)
        fluid = state["kind"] == 0
        own = state["position"][fluid].astype(np.float64) - np.array([meta["eccentricity"], 0.0])
        radius = np.sqrt((own ** 2).sum(axis=1))
        distance = (radius - inner_radius) / spacing
        layer = (distance > layer_spacings[0]) & (distance < layer_spacings[1])
        own, radius, distance = own[layer], radius[layer], distance[layer]
        velocity = state["velocity"][fluid][layer].astype(np.float64)
        tangential = (-own[:, 1] * velocity[:, 0] + own[:, 0] * velocity[:, 1]) / radius
        sector = np.minimum((np.arctan2(own[:, 1], own[:, 0]) % (2.0 * np.pi) / (2.0 * np.pi) * angle_bins)
                            .astype(np.int64), angle_bins - 1)
        powers = [distance ** power for power in range(2 * count - 1)]
        for row in range(count):
            right[:, row] += np.bincount(sector, tangential * powers[row], angle_bins)
            for column in range(count):
                matrix[:, row, column] += np.bincount(sector, powers[row + column], angle_bins)
    coefficients = np.linalg.solve(matrix, right[:, :, None])[:, :, 0]
    slope = coefficients[:, 1] / spacing                                   # d v_t / d n at the wall
    return {"angle": ((np.arange(angle_bins) + 0.5) * 2.0 * np.pi / angle_bins).tolist(),
            "wall_value_over_wall_speed": (coefficients[:, 0] / meta["wall_speed"]).tolist(),
            "slope": slope.tolist(), "vorticity": (slope + rate).tolist()}


def rotor_displacement_map(vorticity: np.ndarray, rate: float) -> np.ndarray:
    """delta_b / delta around a rotor at Omega that carries w = 0: 1 - (0 - Omega) / (omega_N / 2 - Omega)."""
    return 1.0 + rate / (0.5 * vorticity - rate)


def command_eccentric(arguments) -> int:
    runs_directory = _REPO_ROOT / arguments.runs
    names = [name for name in RUNS if RUNS[name]["eccentricity"] and (runs_directory / name / "run.json").exists()]
    lines, records = [], {}

    def say(text=""):
        lines.append(text)
        print(text)
    for name in names:
        run = json.loads((runs_directory / name / "run.json").read_text(encoding="utf-8"))
        meta = run["couette"]
        summary = cached_wall_sums(name, runs_directory, arguments.recompute)
        torques = wall_torques(summary)
        factor = torques["mass_over_lattice_mass"]
        record = {"eccentricity_ratio": meta["eccentricity"] / meta["gap"], "micropolar": meta["micropolar"],
                  "length_in_spacings": (meta["micropolar_length"] / meta["spacing"]) if meta["micropolar"] else None,
                  "steps": run["steps"], "timestep": run["timestep"], "counters_zero": not any(run["counters"].values()),
                  "failure": run["failure"], "revision": run.get("revision"),
                  "steps_per_second": run["steps_per_second"], **torques}
        # the advective transport of angular momentum through the rotor's circle (not zero off-centre): rho0 r^2
        # integral of v_t (v_n - mean v_n) over the angle; the mean of v_n along the closed circle is a drift
        # that the shifting cancels (no mass crosses on average)
        layer = summary["rotor"]["layer"]
        tangential = np.asarray(layer["tangential_velocity_by_angle"]["mean"])
        outward = np.asarray(layer["outward_velocity_by_angle"]["mean"])
        radius = layer["circle_radius"]["mean"]
        record["rotor"]["circle_advection"] = float(DENSITY * radius ** 2 * (tangential * (outward - outward.mean())).sum()
                                                    * 2.0 * np.pi / tangential.size)
        record["rotor"]["circle_with_advection"] = record["rotor"]["circle"] + record["rotor"]["circle_advection"]
        # time behaviour over the saved states: first against second half
        for wall in ("rotor", "stator"):
            values = np.asarray(summary[wall]["torque_exchange"]["values"]) / factor
            half = values.size // 2
            record[wall]["wall_first_half_minus_second"] = float(values[:half].mean() - values[half:].mean())
        # local wall shear of the rotor from the viscous pair sums: tangential force on the fluid per unit arc
        viscous = np.asarray(summary["rotor"]["tangential_by_angle"]["viscous"]["mean"]) / factor
        bins = viscous.size
        arc = meta["inner_radius"] * 2.0 * np.pi / bins
        shear_viscosity = torques["shear_viscosity"] + torques["coupling_viscosity"]      # of the laplacian term
        record["rotor"]["tangential_traction_by_angle"] = (viscous / arc).tolist()
        record["rotor"]["normal_traction_by_angle"] = (
            (np.asarray(summary["rotor"]["radial_by_angle"]["pressure_flux"]["mean"])
             + np.asarray(summary["rotor"]["radial_by_angle"]["viscous"]["mean"])) / factor / arc).tolist()
        if not meta["micropolar"]:
            # omega_N at the rotor two ways: from the pair sums (omega = Omega - f_t / mu) and from the velocity
            # profile beside the wall (omega = dv_t/dn + Omega)
            profile = wall_shear_profiles(name, runs_directory)
            from_sums = meta["rotation_rate"] - (viscous / arc) / shear_viscosity
            from_profile = np.asarray(profile["vorticity"])
            record["rotor_vorticity"] = {"angle": profile["angle"], "from_pair_sums": from_sums.tolist(),
                                         "from_profile": from_profile.tolist(),
                                         "wall_value_over_wall_speed": profile["wall_value_over_wall_speed"]}
            for label, vorticity in (("from_pair_sums", from_sums), ("from_profile", from_profile)):
                ratio = rotor_displacement_map(vorticity, meta["rotation_rate"])
                angle = np.asarray(profile["angle"])
                record[f"displacement_{label}"] = {
                    "ratio_by_angle": ratio.tolist(), "mean": float(ratio.mean()), "min": float(ratio.min()),
                    "max": float(ratio.max()), "angle_of_max_degrees": float(np.degrees(angle[np.argmax(ratio)])),
                    "angle_of_min_degrees": float(np.degrees(angle[np.argmin(ratio)])),
                    "largest_deviation_from_mean": float(np.abs(ratio / ratio.mean() - 1.0).max())}
        records[name] = record

    say("eccentric runs: rotor axis at x = eps g (narrowest gap on the +x side, angle 0); torque about the rotor "
        "axis and force of each wall on the fluid, per unit depth (the fluid acts on the wall with the opposite)")
    say(f"{'run':<24}{'eps':>5}{'l/dx':>6}{'T rotor':>11}{'T stator':>11}{'sum/T':>9}{'T circle':>11}{'+advect.':>10}"
        f"{'as written':>11}{'Fx rotor':>11}{'Fy rotor':>11}{'|F| R1/T':>9}{'angle':>8}{'F sum/|F|':>10}{'s.e. T':>9}"
        f"{'drift':>9}")
    for name, record in records.items():
        rotor, stator = record["rotor"], record["stator"]
        force = -np.asarray(rotor["force"])                      # the fluid on the rotor
        record["rotor_force_on_wall"] = force.tolist()
        record["rotor_force_magnitude"] = float(np.hypot(*force))
        record["rotor_force_angle_degrees"] = float(np.degrees(np.arctan2(force[1], force[0])))
        total = np.asarray(rotor["force"]) + np.asarray(stator["force"])
        say(f"{name:<24}{record['eccentricity_ratio']:>5.2f}"
            + (f"{record['length_in_spacings']:>6g}" if record["micropolar"] else f"{'-':>6}")
            + f"{rotor['wall']:>11.4f}{stator['wall']:>11.4f}{(rotor['wall'] + stator['wall']) / rotor['wall']:>+9.4f}"
            f"{rotor['circle']:>11.4f}{rotor['circle_with_advection']:>10.4f}{rotor['as_written']:>11.4f}"
            f"{force[0]:>11.4f}{force[1]:>11.4f}"
            f"{record['rotor_force_magnitude'] * 0.4 / rotor['wall']:>9.4f}{record['rotor_force_angle_degrees']:>8.2f}"
            f"{np.hypot(*total) / record['rotor_force_magnitude']:>10.4f}{rotor['wall_standard_error']:>9.1e}"
            f"{rotor['wall_first_half_minus_second']:>+9.1e}"
            + ("" if record["counters_zero"] and not record["failure"] else "  FAILED / counters"))
    # noise floor: the three lattices of the Newtonian case
    floors = {}
    for eccentricity in sorted({RUNS[name]["eccentricity"] for name in names}):
        members = [f"e{eccentricity}_newtonian{lattice}" for lattice in LATTICES]
        if all(member in records for member in members):
            torque_values = np.array([records[member]["rotor"]["wall"] for member in members])
            stator_values = np.array([records[member]["stator"]["wall"] for member in members])
            circle_values = np.array([records[member]["rotor"]["circle"] for member in members])
            magnitudes = np.array([records[member]["rotor_force_magnitude"] for member in members])
            angles = np.array([records[member]["rotor_force_angle_degrees"] for member in members])
            floors[eccentricity] = {
                "rotor_torque": torque_values.tolist(),
                "rotor_torque_spread": float((torque_values.max() - torque_values.min()) / torque_values.mean()),
                "stator_torque_spread": float((stator_values.max() - stator_values.min()) / abs(stator_values.mean())),
                "rotor_circle_torque_spread": float((circle_values.max() - circle_values.min()) / circle_values.mean()),
                "force_magnitude": magnitudes.tolist(),
                "force_magnitude_spread": float((magnitudes.max() - magnitudes.min()) / magnitudes.mean()),
                "force_angle_degrees": angles.tolist(), "force_angle_spread_degrees": float(angles.max() - angles.min())}
            say()
            say(f"noise floor at eps = {eccentricity / 100:g} from the three lattices: rotor torque spread "
                f"{floors[eccentricity]['rotor_torque_spread']:.2e} (wall form), "
                f"{floors[eccentricity]['rotor_circle_torque_spread']:.2e} (circle form), stator "
                f"{floors[eccentricity]['stator_torque_spread']:.2e}; force magnitude "
                f"{floors[eccentricity]['force_magnitude_spread']:.2e}, direction "
                f"{floors[eccentricity]['force_angle_spread_degrees']:.3f} degrees")
    say()
    say("micropolar against Newtonian: D_T = T / T_Newtonian - 1 and the change of the force on the rotor")
    say(f"{'run':<24}{'D_T rotor':>11}{'D_T stator':>11}{'D_T circle':>11}{'|F| change':>11}{'turn (deg)':>11}"
        f"{'Fx / FxN':>10}{'Fy / FyN':>10}")
    for name, record in records.items():
        control_name = RUNS[name]["control"]
        if control_name not in records:
            continue
        control = records[control_name]
        record["change"] = {
            "torque_rotor": record["rotor"]["wall"] / control["rotor"]["wall"] - 1.0,
            "torque_stator": record["stator"]["wall"] / control["stator"]["wall"] - 1.0,
            "torque_rotor_circle": record["rotor"]["circle_with_advection"] / control["rotor"]["circle_with_advection"] - 1.0,
            "force_magnitude": record["rotor_force_magnitude"] / control["rotor_force_magnitude"] - 1.0,
            "force_turn_degrees": record["rotor_force_angle_degrees"] - control["rotor_force_angle_degrees"],
            "force_x_ratio": record["rotor_force_on_wall"][0] / control["rotor_force_on_wall"][0],
            "force_y_ratio": record["rotor_force_on_wall"][1] / control["rotor_force_on_wall"][1]}
        change = record["change"]
        say(f"{name:<24}{change['torque_rotor']:>11.5f}{change['torque_stator']:>11.5f}"
            f"{change['torque_rotor_circle']:>11.5f}{change['force_magnitude']:>11.5f}"
            f"{change['force_turn_degrees']:>11.3f}{change['force_x_ratio']:>10.4f}{change['force_y_ratio']:>10.4f}")
    say()
    say("displacement of the rotor's effective surface, delta_b / delta = 1 + Omega / (omega_N / 2 - Omega), from "
        "the Newtonian control (concentric value (R1 / R2)^2 = 0.64)")
    say(f"{'run':<24}{'omega_N from':>16}{'mean':>9}{'min':>9}{'at (deg)':>9}{'max':>9}{'at (deg)':>9}"
        f"{'max dev. from mean':>20}")
    for name, record in records.items():
        for label in ("from_pair_sums", "from_profile"):
            entry = record.get(f"displacement_{label}")
            if entry:
                say(f"{name:<24}{label:>16}{entry['mean']:>9.4f}{entry['min']:>9.4f}{entry['angle_of_min_degrees']:>9.1f}"
                    f"{entry['max']:>9.4f}{entry['angle_of_max_degrees']:>9.1f}"
                    f"{entry['largest_deviation_from_mean']:>20.4f}")
    if arguments.json:
        target = _REPO_ROOT / arguments.json
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"noise_floors": {str(key): value for key, value in floors.items()},
                                      "runs": records}, indent=1), encoding="utf-8")
        target.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    listing.set_defaults(handler=command_list)
    queue = commands.add_parser("queue")
    queue.add_argument("names", nargs="+")
    queue.add_argument("--device", type=int, required=True)
    queue.add_argument("--out", default=RUNS_DIRECTORY)
    queue.add_argument("--steps", type=int, default=None, help="a short run for a smoke test")
    queue.add_argument("--sample-steps", type=int, default=None)
    queue.set_defaults(handler=command_queue)
    analyse = commands.add_parser("analyse")
    analyse.add_argument("names", nargs="*", help="concentric runs of the table (default: every finished one)")
    analyse.add_argument("--runs", default=RUNS_DIRECTORY)
    analyse.add_argument("--fit-margin", type=float, default=1.0)
    analyse.add_argument("--json", default=None, help="relative to the repository root; a .txt goes beside it")
    analyse.set_defaults(handler=command_analyse)
    torque = commands.add_parser("torque")
    torque.add_argument("names", nargs="*", help="concentric runs of the table (default: every finished one)")
    torque.add_argument("--runs", default=RUNS_DIRECTORY)
    torque.add_argument("--recompute", action="store_true", help="ignore the wall_sums.json kept beside each run")
    torque.add_argument("--json", default=None, help="relative to the repository root; a .txt goes beside it")
    torque.set_defaults(handler=command_torque)
    eccentric = commands.add_parser("eccentric")
    eccentric.add_argument("--runs", default=RUNS_DIRECTORY)
    eccentric.add_argument("--recompute", action="store_true")
    eccentric.add_argument("--json", default=None)
    eccentric.set_defaults(handler=command_eccentric)
    arguments = parser.parse_args()
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
