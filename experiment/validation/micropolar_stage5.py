"""
micropolar_stage5.py - Stage 5 of docs/micropolar_design.md: the wall rule in its slip form, the exact Stokes
benchmark of the Newtonian flow between eccentric rotating cylinders, and the candidates for the wall and the
density-diffusion term.

  list        The run table of this stage: name, section, generator options.
  queue       Generate (if absent) and run the named runs one after the other on one device.
  rule        Section 1 (host): D of the rule in its slip form beside the (R1 / R2)^2 form, for the sweep of Stage 4.
  film        Section 2 (host): the plane film between two walls with w = 0, exact against the rule, and the journal
              bearing in the thin-film approximation with both film functions.
  benchmark   Sections 2 to 4: torque and load of every finished eccentric run of the table against the exact Stokes
              values (micropolar_eccentric_reference.py), the wall-normal velocity per unit wall pressure, the fluid
              behind a wall, the steadiness; the rule between eccentric cylinders; the candidates.
  concentric  Section 4: the concentric control of each candidate (beta, hydrodynamic offsets, residual, drift).

Cases under cases/couette_stage5, raw runs under logs/micropolar/stage5 (not in the repository); the runs of
Stage 4 that this stage reads again stay where they are (logs/micropolar/stage4).

Candidates of section 4 (Newtonian, thesis configuration otherwise):
  thesis   numerics as the thesis configuration                      (prefix none / leak_ of Stage 4, thesis_ here)
  nd       numerics.use_density_diffusion: false                     (prefix nd_)
  fd       numerics.delta_fluid_only: true (diffusion between fluid pairs only)   (prefix fd_)
  ew       numerics.wall_density: eos (walls store the Tait density of their pressure; the part of the v7-wall
           treatment that is in this tree, without its volume anchor)             (prefix ew_)
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

from experiment.validation import micropolar_stage4 as stage4  # noqa: E402

CASES = "cases/couette_stage5"
RUNS_DIRECTORY = "logs/micropolar/stage5"
STAGE4_RUNS_DIRECTORY = stage4.RUNS_DIRECTORY
CANDIDATE_OPTIONS = {"thesis": [], "nd": ["--no-density-diffusion"], "fd": ["--delta-fluid-only"],
                     "ew": ["--wall-density", "eos"]}
LATTICES = stage4.LATTICES
# K = 3: delta = c l / (2 nu) = K l / (2 (1 + K)) = 3 l / 8
DISPLACEMENT_OVER_LENGTH = 0.375


def _newtonian(gap, eccentricity, candidate, reynolds=10.0, lattice="", extra=(), section=(), **keys) -> dict:
    return {"generate": ["--gap-in-spacings", str(gap), "--eccentricity-in-spacings", str(eccentricity),
                         "--step-from-coupling-ratio", "3", "--step-from-gap-over-two-lengths", "5",
                         "--reynolds", repr(float(reynolds)), *LATTICES[lattice], *CANDIDATE_OPTIONS[candidate],
                         *extra],
            "control": None, "gap": gap, "eccentricity": eccentricity, "candidate": candidate,
            "reynolds": float(reynolds), "section": list(section), **keys}


def _micropolar(gap, eccentricity, candidate, gap_over_two_lengths, control, lattice="", section=(), **keys) -> dict:
    return {"generate": ["--gap-in-spacings", str(gap), "--eccentricity-in-spacings", str(eccentricity),
                         "--coupling-ratio", "3", "--gap-over-two-lengths", str(gap_over_two_lengths),
                         "--rotor-micro-rotation-factor", "0", *LATTICES[lattice], *CANDIDATE_OPTIONS[candidate]],
            "control": control, "gap": gap, "eccentricity": eccentricity, "candidate": candidate, "reynolds": 10.0,
            "coupling_ratio": 3, "gap_over_two_lengths": gap_over_two_lengths, "section": list(section), **keys}


def _slip(eccentricity, length, candidate="nd", control=None) -> dict:
    """The Newtonian flow of the rule in its slip form (section 2): rotor radius R1 + delta turning at
    Omega (R1 - delta) / (R1 + delta), stator radius R2 - delta, the same offset of the axes, viscosity nu_inf, the
    sound speed, dt and end time of the micropolar run it stands for. Lengths in dx (dx = 1e-3 m)."""
    displacement = DISPLACEMENT_OVER_LENGTH * length
    inner, outer = 400.0 + displacement, 500.0 - displacement
    gap = outer - inner
    wall_speed = 2.5 * (400.0 - displacement) * 1.0e-3            # Omega (R1 - delta); Omega = 2.5 rad/s
    return {"generate": ["--gap", repr(gap * 1.0e-3), "--gap-in-spacings", repr(gap), "--inner-radius-in-gaps",
                         repr(inner / gap), "--eccentricity-in-spacings", str(eccentricity), "--wall-speed",
                         repr(wall_speed), "--reynolds", repr(wall_speed * gap * 1.0e-3 / 0.01), "--mach",
                         repr(wall_speed / 50.0), "--end-time-factor", repr(2.0 * 0.01 / (gap * 1.0e-3) ** 2),
                         "--step-from-coupling-ratio", "3", "--step-from-gap-over-two-lengths", "5",
                         *CANDIDATE_OPTIONS[candidate]],
            "control": control, "gap": gap, "eccentricity": eccentricity, "candidate": candidate, "reynolds": 10.0,
            "section": ["slip"],
            "slip": {"length_in_spacings": length, "displacement_in_spacings": displacement,
                     "inner_radius_in_spacings": inner, "outer_radius_in_spacings": outer,
                     "rotation_rate": wall_speed / (inner * 1.0e-3)}}


def _build_table() -> dict:
    runs = {}
    whole = {"state_span": 1.0, "state_count": 96, "state_band": 3.0, "last_quarter_only": True}

    # section 3, the benchmark in the nd setting: eps = 0.3 and 0.6 at gap Reynolds numbers 1 and 10 (g = 100 dx;
    # the two at Reynolds number 10 are nd_e30_newtonian and nd_e60_newtonian of Stage 4), eps = 0.6 at Reynolds
    # number 1 with g = 200 dx. Reynolds number 1: nu_inf ten times larger at the same wall speed and sound speed;
    # the step is then the viscous one and the end time 10 g^2 / nu_inf (0.4 turns of the rotor).
    for eccentricity in (30, 60):
        runs[f"nd_e{eccentricity}_re1"] = _newtonian(100, eccentricity, "nd", 1.0, extra=["--end-time-factor", "10"],
                                                     section=("benchmark",), **whole)
    runs["nd_e60_re1_g200"] = _newtonian(200, 120, "nd", 1.0, extra=["--end-time-factor", "10"],
                                         section=("benchmark",), **whole)
    # the same flow at half the Mach number (the pressure of the bearing is 20 rho U^2 at Reynolds number 1: with
    # Ma = 0.02 the density varies by 0.8 %)
    runs["nd_e60_re1_ma001"] = _newtonian(100, 60, "nd", 1.0, extra=["--end-time-factor", "10", "--mach", "0.01"],
                                          section=("benchmark", "mach"), **whole)
    runs["nd_e60_re10_g200"] = _newtonian(200, 120, "nd", 10.0, section=("benchmark",), **whole)
    # noise floor of this setting: the three lattices at eps = 0.6, Newtonian and the micropolar run at l = 10 dx
    for lattice in ("_shifted", "_rotated"):
        runs[f"nd_e60_newtonian{lattice}"] = _newtonian(100, 60, "nd", 10.0, lattice, section=("floor",))
        runs[f"nd_e60_k3_m5{lattice}"] = _micropolar(100, 60, "nd", 5, f"nd_e60_newtonian{lattice}", lattice,
                                                     section=("floor",))

    # section 2, the rule between eccentric cylinders (and between concentric ones as the check of the procedure)
    for eccentricity, prefix in ((0, "nd_g100"), (30, "nd_e30"), (60, "nd_e60")):
        for gap_over_two_lengths, length in ((5, 10.0), (2, 25.0)):
            runs[f"{prefix}_slip_m{gap_over_two_lengths}"] = _slip(eccentricity, length, "nd",
                                                                    control=f"{prefix}_newtonian")

    # section 4, the candidates: benchmark at eps = 0.6, Reynolds numbers 1 and 10; concentric control at g = 100 dx
    runs["thesis_e60_re1"] = _newtonian(100, 60, "thesis", 1.0, extra=["--end-time-factor", "10"],
                                        section=("candidate",), **whole)
    runs["thesis_e60_re10"] = _newtonian(100, 60, "thesis", 10.0, section=("candidate",), **whole)
    runs["nd_e60_re10"] = _newtonian(100, 60, "nd", 10.0, section=("candidate",), **whole)
    runs["fd_e60_re1"] = _newtonian(100, 60, "fd", 1.0, extra=["--end-time-factor", "10"], section=("candidate",),
                                    **whole)
    runs["fd_e60_re10"] = _newtonian(100, 60, "fd", 10.0, section=("candidate",), **whole)
    runs["fd_g100_newtonian"] = _newtonian(100, 0, "fd", 10.0, section=("candidate", "concentric"))
    runs["ew_e60_re1"] = _newtonian(100, 60, "ew", 1.0, extra=["--end-time-factor", "10"], section=("candidate",),
                                    **whole)
    runs["ew_e60_re10"] = _newtonian(100, 60, "ew", 10.0, section=("candidate",), **whole)
    runs["ew_g100_newtonian"] = _newtonian(100, 0, "ew", 10.0, section=("candidate", "concentric"))
    # the concentric controls of the thesis configuration and of nd again (g100_newtonian and nd_g100_newtonian of
    # Stage 4, with the level of density and pressure in the samples)
    runs["thesis_g100_control"] = _newtonian(100, 0, "thesis", 10.0, section=("candidate", "concentric"))
    runs["nd_g100_control"] = _newtonian(100, 0, "nd", 10.0, section=("candidate", "concentric"))
    # beyond the table of the task: the micropolar runs of the pilot with the fd candidate (concentric m = 5 and
    # m = 2 as the check that D is unchanged, eps = 0.6 with its control)
    for prefix, eccentricity, control in (("fd_g100", 0, "fd_g100_newtonian"), ("fd_e60", 60, "fd_e60_re10"),
                                          ("ew_g100", 0, "ew_g100_newtonian"), ("ew_e60", 60, "ew_e60_re10")):
        for gap_over_two_lengths in (5, 2):
            runs[f"{prefix}_k3_m{gap_over_two_lengths}"] = _micropolar(
                100, eccentricity, prefix[:2], gap_over_two_lengths, control, section=("candidate_extra",))
    # beyond the table of the task: the fd candidate five times longer between concentric cylinders (10 g^2 / nu_inf,
    # 670,000 steps), control and m = 5: does the level of the pressure, which nothing ties in this setting, reach D?
    runs["fd_g100_newtonian_long"] = _newtonian(100, 0, "fd", 10.0, extra=["--end-time-factor", "10"],
                                                section=("candidate_extra", "long"))
    runs["fd_g100_k3_m5_long"] = _micropolar(100, 0, "fd", 5, "fd_g100_newtonian_long",
                                             section=("candidate_extra", "long"))
    runs["fd_g100_k3_m5_long"]["generate"] += ["--end-time-factor", "10"]
    # beyond the table of the task: the rule where the wall layers are thin against the narrowest gap. g = 200 dx,
    # l = 10 dx (m = 10; the narrowest gap is 14 l at eps = 0.3 and 8 l at eps = 0.6), nd setting, with controls
    runs["nd_e30_re10_g200"] = _newtonian(200, 60, "nd", 10.0, section=("slip_extra",))
    for eccentricity, control in ((60, "nd_e30_re10_g200"), (120, "nd_e60_re10_g200")):
        runs[f"nd_e{eccentricity // 2}_k3_m10_g200"] = _micropolar(200, eccentricity, "nd", 10, control,
                                                                  section=("slip_extra",))
    return runs


RUNS = _build_table()
# the runs of Stage 4 this stage reads again (their entries keep the Stage 4 form)
STAGE4_NAMES = ("nd_e30_newtonian", "nd_e60_newtonian", "nd_e30_k3_m5", "nd_e30_k3_m2", "nd_e60_k3_m5",
                "nd_e60_k3_m2", "nd_g100_newtonian", "nd_g100_k3_m5", "nd_g100_k3_m2", "g100_newtonian",
                "g100_k3_m5", "g100_k3_m2", "e60_newtonian", "leak_e60", "e30_newtonian")
for _name in STAGE4_NAMES:
    stage4.RUNS[_name].setdefault("candidate", "nd" if _name.startswith("nd_") else "thesis")
    stage4.RUNS[_name].setdefault("reynolds", 10.0)
# the functions of Stage 4 look a run up in its table (last_quarter_only, control)
stage4.RUNS.update(RUNS)


def run_directory(name: str) -> pathlib.Path:
    """The directory that holds the raw run: this stage's, else Stage 4's."""
    own = _REPO_ROOT / RUNS_DIRECTORY
    if (own / name / "run.json").exists() or name in RUNS:
        return own
    return _REPO_ROOT / STAGE4_RUNS_DIRECTORY


def command_list(arguments) -> int:
    for name, entry in RUNS.items():
        print(f"{name:<28}{','.join(entry['section']):<22}control {entry['control'] or '-':<24}"
              f"{' '.join(entry['generate'])}")
    return 0


def ensure_case(name: str) -> dict:
    case_dir = _REPO_ROOT / CASES / name
    if not (case_dir / "couette.json").exists():
        code = subprocess.run([sys.executable, "-m", "experiment.validation.micropolar_couette", "generate",
                               "--out", f"{CASES}/{name}", *stage4.STEP_OPTIONS, *RUNS[name]["generate"]],
                              cwd=_REPO_ROOT).returncode
        if code:
            raise RuntimeError(f"generate {name}: exit {code}")
    return json.loads((case_dir / "couette.json").read_text(encoding="utf-8"))


def command_queue(arguments) -> int:
    (_REPO_ROOT / RUNS_DIRECTORY).mkdir(parents=True, exist_ok=True)
    for name in arguments.names:
        if name not in RUNS:
            print(f"[stage5] {name}: not in the table", flush=True)
            continue
        run_dir = _REPO_ROOT / RUNS_DIRECTORY / name
        if (run_dir / "run.json").exists():
            print(f"[stage5] {name}: done already", flush=True)
            continue
        entry = RUNS[name]
        meta = ensure_case(name)
        steps = arguments.steps if arguments.steps else meta["steps"]
        sample_steps = arguments.sample_steps if arguments.sample_steps else stage4.sample_steps_for(steps)
        # eccentric runs keep the fluid within 3 h of a wall (the wall sums need 2 h, the profiles beside the wall
        # 15 dx, and every particle behind a wall surface is inside the band), concentric runs within 2 h
        band = entry.get("state_band", 3.0 if entry["eccentricity"] else 2.0)
        command = [sys.executable, "-m", "experiment.validation.micropolar_couette", "run", "--case",
                   f"{CASES}/{name}", "--out", f"{RUNS_DIRECTORY}/{name}", "--device-map", str(arguments.device),
                   "--sample-steps", str(sample_steps), "--state-count",
                   str(entry.get("state_count", stage4.STATE_COUNT)), "--state-band-supports", str(band)]
        if arguments.steps:
            command += ["--steps", str(arguments.steps)]
        if "state_span" in entry:
            command += ["--state-span", str(entry["state_span"])]
        with open(_REPO_ROOT / RUNS_DIRECTORY / f"{name}.log", "w", encoding="utf-8") as log:
            code = subprocess.run(command, cwd=_REPO_ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"[stage5] {name}: exit {code}", flush=True)
    return 0

# ---------------------------------------------------------------------------------------------------------------
# section 1 (host): the rule in its slip form
# ---------------------------------------------------------------------------------------------------------------
def potential_vortex(inner: float, outer: float) -> float:
    """beta_N(a, b) / Omega = a^2 b^2 / (b^2 - a^2) of the Newtonian flow between a rotor at a and a stator at b."""
    return inner ** 2 * outer ** 2 / (outer ** 2 - inner ** 2)


def slip_rule_torque_change(inner: float, outer: float, displacement: float, rotor_spin_over_rate: float = 0.0) -> float:
    """D of the rule in its slip form. Every wall stands delta further into the fluid; a wall that carries w_b and
    turns at Omega_b appears to move at its own velocity + 2 delta (w_b - Omega_b) z^ x n. Stator at rest with
    w = 0: displaced only. Rotor at Omega with w_b = f Omega: its displaced surface (radius R1 + delta) moves at
    Omega (R1 + delta) + 2 delta (f - 1) Omega = Omega (R1 + (2 f - 1) delta). With f = 0:
    D = [(R1 - delta) / (R1 + delta)] beta_N(R1 + delta, R2 - delta) / beta_N(R1, R2) - 1."""
    apparent_rate = (inner + (2.0 * rotor_spin_over_rate - 1.0) * displacement) / (inner + displacement)
    return (apparent_rate * potential_vortex(inner + displacement, outer - displacement)
            / potential_vortex(inner, outer) - 1.0)


def command_rule(arguments) -> int:
    from scipy.optimize import brentq
    from experiment.validation import micropolar_couette as couette
    from experiment.validation import micropolar_couette_reference as reference
    inner, outer, gap, wall_speed, far_viscosity = 0.4, 0.5, 0.1, 1.0, 0.01
    rate = wall_speed / inner
    ratios = (3.0, 1.0, 0.5)
    lines, records = [], {"sweep": {}, "host": {}}

    def say(text=""):
        lines.append(text)
        print(text)

    def evaluate(coupling_ratio, gap_over_two_lengths, rotor_spin_over_rate=0.0):
        length = gap / (2.0 * gap_over_two_lengths)
        design = couette.design_parameters(far_viscosity, coupling_ratio, length)
        eringen = reference.eringen_parameters_from_solver_parameters(
            stage4.DENSITY, design["viscosity"], design["coupling"], design["viscosity_rotation"])
        exact = reference.relative_torque_change(inner, outer, eringen.shear_viscosity, eringen.coupling_viscosity,
                                                 eringen.spin_viscosity, wall_speed, 0.0,
                                                 rotor_spin_over_rate * rate, 0.0)
        displacement = design["coupling"] * length / (2.0 * design["viscosity"])
        return {"exact": exact, "displacement": displacement,
                "slip": slip_rule_torque_change(inner, outer, displacement, rotor_spin_over_rate),
                "first_order": stage4.rule_torque_change(inner, outer, displacement)}

    say("The wall rule against the exact concentric solution (R1 = 4 g, R2 = 5 g, w = 0 on both walls)")
    say()
    say("1. by hand in the task (K = 3): D_rule in the slip form and the exact D, nominal radii")
    for value in (2, 5, 10):
        result = evaluate(3.0, value)
        say(f"   m = {value:<3g} delta / g = {result['displacement'] / gap:.6f}   D_slip = {result['slip']:.5f}   "
            f"D_exact = {result['exact']:.5f}   D_(R1/R2)^2 = {result['first_order']:.5f}")
    say()
    say("2. the sweep of Stage 4 (data/micropolar/stage4/couette_stage4.json): exact D and both forms of the rule with "
        "the hydrodynamic radii of the control of each run, as in the table of 13.2")
    say(f"   {'run':<16}{'K':>5}{'m':>5}{'l/dx':>6}{'D_SPH':>10}{'D_exact':>10}{'D_SPH/D_exact':>15}"
        f"{'(R1/R2)^2 form':>16}{'/ D_exact':>11}{'slip form':>11}{'/ D_exact':>11}")
    analysis = json.loads((_REPO_ROOT / arguments.analysis).read_text(encoding="utf-8"))["runs"]
    for name, record in analysis.items():
        sections = record.get("sections", [])
        if "relative_torque_change" not in record or not ("sweep" in sections or "resolution" in sections):
            continue
        control = analysis[record["control"]]
        displacement = record["wall_displacement_in_spacings"] * gap / record["gap_in_spacings"]
        slip = slip_rule_torque_change(control["hydrodynamic_inner_radius"], control["hydrodynamic_outer_radius"],
                                       displacement)
        exact = record["exact_hydrodynamic"]
        records["sweep"][name] = {
            "coupling_ratio": record["coupling_ratio"], "gap_over_two_lengths": record["gap_over_two_lengths"],
            "length_in_spacings": record["length_in_spacings"],
            "relative_torque_change": record["relative_torque_change"], "exact": exact,
            "sph_over_exact": record["sph_over_exact"], "first_order_rule": record["rule_hydrodynamic"],
            "first_order_rule_over_exact": record["rule_over_exact"], "slip_rule": slip,
            "slip_rule_over_exact": slip / exact}
        say(f"   {name:<16}{record['coupling_ratio']:>5g}{record['gap_over_two_lengths']:>5g}"
            f"{record['length_in_spacings']:>6g}{record['relative_torque_change']:>10.5f}{exact:>10.5f}"
            f"{record['sph_over_exact']:>15.4f}{record['rule_hydrodynamic']:>16.5f}{record['rule_over_exact']:>11.4f}"
            f"{slip:>11.5f}{slip / exact:>11.4f}")
    say()
    say("3. on the host over m (nominal radii): D_rule / D_exact, slip form | (R1 / R2)^2 form")
    say(f"   {'m':>7}" + "".join(f"{'K = %g' % ratio:>12}" for ratio in ratios)
        + "   |" + "".join(f"{'K = %g' % ratio:>12}" for ratio in ratios))
    for value in (1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5, 7, 10, 20, 40, 100):
        results = [evaluate(ratio, value) for ratio in ratios]
        records["host"][str(value)] = {f"K{ratio:g}": {"slip_over_exact": result["slip"] / result["exact"],
                                                       "first_order_over_exact": result["first_order"] / result["exact"]}
                                       for ratio, result in zip(ratios, results)}
        say(f"   {value:>7g}" + "".join(f"{result['slip'] / result['exact']:>12.4f}" for result in results)
            + "   |" + "".join(f"{result['first_order'] / result['exact']:>12.4f}" for result in results))
    for ratio in ratios:
        def slip_ratio(value):
            result = evaluate(ratio, value)
            return result["slip"] / result["exact"]
        grid = np.geomspace(1.0, 100.0, 800)
        values = np.array([slip_ratio(value) for value in grid])
        worst_above_two = float(np.abs(values[grid >= 2.0] - 1.0).max())
        crossings = []
        for level in (1.05, 1.02, 1.01):
            above = np.where(np.abs(values - 1.0) > level - 1.0)[0]
            if above.size and above[-1] + 1 < grid.size:
                sign = np.sign(values[above[-1]] - 1.0)
                crossings.append(float(brentq(lambda value: sign * (slip_ratio(value) - 1.0) - (level - 1.0),
                                              grid[above[-1]], grid[above[-1] + 1])))
            else:
                crossings.append(float("nan"))
        say(f"   K = {ratio:g}: slip form within {worst_above_two:.4f} of exact for m >= 2; |D_rule / D_exact - 1| falls "
            f"below 0.05 at m = {crossings[0]:.2f}, below 0.02 at m = {crossings[1]:.2f}, below 0.01 at "
            f"m = {crossings[2]:.2f}; (ratio - 1) m^2 = {(slip_ratio(20.0) - 1.0) * 400.0:+.3f} at m = 20, "
            f"{(slip_ratio(40.0) - 1.0) * 1600.0:+.3f} at m = 40, {(slip_ratio(100.0) - 1.0) * 1.0e4:+.3f} at m = 100")
    say()
    say("4. a rotor that carries w_b = f Omega (the general statement: apparent wall velocity = wall velocity + "
        "2 delta (w_b - Omega_b) z^ x n), K = 3: D_slip / D_exact, and D_exact")
    say(f"   {'m':>7}" + "".join(f"{'f = %g' % factor:>12}" for factor in (0.0, 0.5, 1.0, 2.0))
        + "   |" + "".join(f"{'f = %g' % factor:>12}" for factor in (0.0, 0.5, 1.0, 2.0)))
    for value in (2, 5, 10, 20, 40):
        results = [evaluate(3.0, value, factor) for factor in (0.0, 0.5, 1.0, 2.0)]
        say(f"   {value:>7g}" + "".join(f"{result['slip'] / result['exact']:>12.4f}" for result in results)
            + "   |" + "".join(f"{result['exact']:>12.5f}" for result in results))
    if arguments.out:
        target = _REPO_ROOT / arguments.out
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        target.with_suffix(".json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    return 0

# ---------------------------------------------------------------------------------------------------------------
# sections 2 to 4: eccentric runs against the exact Stokes values, the walls, the rule, the candidates
# ---------------------------------------------------------------------------------------------------------------
def wall_normal_profiles(name: str, runs_directory: pathlib.Path, angle_bins: int = 72, layer_spacings=(0.5, 12.0),
                         degree: int = 3) -> dict:
    """The velocity of the fluid into each wall and its pressure, extrapolated to the wall surface: per angle bin
    (about the axis of that wall) a polynomial in the wall distance fitted to the fluid between 0.5 and 12 dx from
    the surface, over the saved states of the last quarter. Then the straight line of the velocity into the wall
    against the wall pressure over the bins: its slope in units of U per rho0 U^2, and the correlation."""
    from experiment.validation import micropolar_wall_torque as wall_torque
    run = json.loads((runs_directory / name / "run.json").read_text(encoding="utf-8"))
    meta = run["couette"]
    spacing = meta["spacing"]
    count = degree + 1
    walls = {"rotor": {"axis": np.array([meta["eccentricity"], 0.0]), "radius": meta["inner_radius"], "sign": -1.0},
             "stator": {"axis": np.zeros(2), "radius": meta["outer_radius"], "sign": 1.0}}
    sums = {wall: {"matrix": np.zeros((angle_bins, count, count)), "velocity": np.zeros((angle_bins, count)),
                   "pressure": np.zeros((angle_bins, count))} for wall in walls}
    states = 0
    for path in wall_torque.state_files(runs_directory / name):
        if int(path.stem.split("_")[1]) <= 0.75 * run["steps"]:
            continue
        states += 1
        state = wall_torque.load_state(path)
        fluid = state["kind"] == 0
        position = state["position"][fluid].astype(np.float64)
        velocity = state["velocity"][fluid].astype(np.float64)
        pressure = state["pressure"][fluid].astype(np.float64)
        for wall, entry in walls.items():
            own = position - entry["axis"]
            radius = np.sqrt((own ** 2).sum(axis=1))
            # distance from the surface into the fluid: r - R1 at the rotor, R2 - r at the stator
            distance = -entry["sign"] * (radius - entry["radius"]) / spacing
            layer = (distance > layer_spacings[0]) & (distance < layer_spacings[1])
            outward = (own[layer, 0] * velocity[layer, 0] + own[layer, 1] * velocity[layer, 1]) / radius[layer]
            into_wall = entry["sign"] * outward
            sector = np.minimum((np.arctan2(own[layer, 1], own[layer, 0]) % (2.0 * np.pi) / (2.0 * np.pi)
                                 * angle_bins).astype(np.int64), angle_bins - 1)
            powers = [distance[layer] ** power for power in range(2 * count - 1)]
            for row in range(count):
                sums[wall]["velocity"][:, row] += np.bincount(sector, into_wall * powers[row], angle_bins)
                sums[wall]["pressure"][:, row] += np.bincount(sector, pressure[layer] * powers[row], angle_bins)
                for column in range(count):
                    sums[wall]["matrix"][:, row, column] += np.bincount(sector, powers[row + column], angle_bins)
    result = {"states": states, "angle": ((np.arange(angle_bins) + 0.5) * 2.0 * np.pi / angle_bins).tolist()}
    dynamic_pressure = stage4.DENSITY * meta["wall_speed"] ** 2
    for wall in walls:
        velocity_at_wall = np.linalg.solve(sums[wall]["matrix"], sums[wall]["velocity"][:, :, None])[:, 0, 0]
        pressure_at_wall = np.linalg.solve(sums[wall]["matrix"], sums[wall]["pressure"][:, :, None])[:, 0, 0]
        scaled_velocity = velocity_at_wall / meta["wall_speed"]
        scaled_pressure = pressure_at_wall / dynamic_pressure
        slope, intercept = np.polyfit(scaled_pressure, scaled_velocity, 1)
        correlation = float(np.corrcoef(scaled_pressure, scaled_velocity)[0, 1])
        result[wall] = {
            "velocity_into_wall_over_wall_speed": scaled_velocity.tolist(),
            "wall_pressure_over_dynamic_pressure": scaled_pressure.tolist(),
            "slope": float(slope), "intercept": float(intercept), "correlation": correlation,
            "slope_over_mach": float(slope / meta["mach"]),
            "velocity_min": float(scaled_velocity.min()), "velocity_max": float(scaled_velocity.max()),
            "velocity_rms": float(np.sqrt((scaled_velocity ** 2).mean())),
            "pressure_min": float(pressure_at_wall.min()), "pressure_max": float(pressure_at_wall.max())}
    return result


def behind_walls(name: str, runs_directory: pathlib.Path) -> dict:
    """Fluid particles more than half a spacing behind a wall surface in the last saved state, the deepest one, and
    the first saved state that has one."""
    from experiment.validation import micropolar_wall_torque as wall_torque
    run = json.loads((runs_directory / name / "run.json").read_text(encoding="utf-8"))
    meta = run["couette"]
    files = wall_torque.state_files(runs_directory / name)

    def measure(path) -> dict:
        state = wall_torque.load_state(path)
        fluid = state["kind"] == 0
        position = state["position"][fluid].astype(np.float64)
        own = position - np.array([meta["eccentricity"], 0.0])
        rotor = (np.hypot(own[:, 0], own[:, 1]) - meta["inner_radius"]) / meta["spacing"]
        stator = (meta["outer_radius"] - np.hypot(position[:, 0], position[:, 1])) / meta["spacing"]
        return {"step": int(state["step"]), "inside_rotor": int((rotor < -0.5).sum()),
                "beyond_stator": int((stator < -0.5).sum()), "deepest_rotor": float(rotor.min()),
                "deepest_stator": float(stator.min())}
    last = measure(files[-1])
    first_time = None
    whole = stage4.RUNS.get(name, {}).get("state_span") == 1.0
    if whole and (last["inside_rotor"] or last["beyond_stator"]):
        for path in files:
            entry = measure(path)
            if entry["inside_rotor"] or entry["beyond_stator"]:
                first_time = entry["step"] / run["steps"]
                break
    return {**last, "fluid": meta["fluid_count"], "share": (last["inside_rotor"] + last["beyond_stator"])
            / meta["fluid_count"], "first_seen_at_fraction_of_run": first_time, "states_cover_run": bool(whole)}


def sample_levels(name: str, runs_directory: pathlib.Path) -> dict:
    """Density range and, where the run recorded them, the level of the density and the pressure of the fluid:
    over the last quarter, and the change from the second to the last quarter."""
    data = np.load(runs_directory / name / "samples.npz")
    quarter = len(data["step"]) // 4
    result = {"density_min": float(data["density_min"][-quarter:].min()),
              "density_max": float(data["density_max"][-quarter:].max())}
    if "density_mean" in data.files:
        result.update({
            "density_mean": float(data["density_mean"][-quarter:].mean()),
            "density_mean_change": float(data["density_mean"][-quarter:].mean()
                                         - data["density_mean"][quarter:2 * quarter].mean()),
            "pressure_mean": float(data["pressure_mean"][-quarter:].mean()),
            "pressure_mean_change": float(data["pressure_mean"][-quarter:].mean()
                                          - data["pressure_mean"][quarter:2 * quarter].mean()),
            "pressure_min": float(data["pressure_min"][-quarter:].min()),
            "pressure_max": float(data["pressure_max"][-quarter:].max())})
    return result


# the concentric Newtonian control whose hydrodynamic radii stand for a setting and a resolution
CONCENTRIC_CONTROLS = {("thesis", 100): "thesis_g100_control", ("nd", 100): "nd_g100_control",
                       ("fd", 100): "fd_g100_newtonian", ("ew", 100): "ew_g100_newtonian"}
CONCENTRIC_FALLBACKS = {("thesis", 100): "g100_newtonian", ("nd", 100): "nd_g100_newtonian", 200: "g200_newtonian"}
_OFFSETS = {}


def hydrodynamic_offsets(candidate: str, gap_in_spacings: float):
    """(inner, outer) offsets in dx of the hydrodynamic radii from the nominal ones, from the concentric control of
    the candidate at that resolution (g = 200 dx: the thesis control of Stage 4, the only one run); None if no such
    control has finished. The displaced geometries of the rule take the offsets of the control at g = 100 dx."""
    resolution = 200 if gap_in_spacings > 150 else 100
    key = (candidate, resolution)
    if key not in _OFFSETS:
        names = ([CONCENTRIC_FALLBACKS[200]] if resolution == 200
                 else [CONCENTRIC_CONTROLS.get(key), CONCENTRIC_FALLBACKS.get(key)])
        _OFFSETS[key] = None
        for control in names:
            if control and _finished(control):
                record = stage4.analyse_concentric([control], run_directory(control))["runs"][control]
                _OFFSETS[key] = {"control": control, "inner": record["inner_offset_in_spacings"],
                                 "outer": record["outer_offset_in_spacings"]}
                break
    return _OFFSETS[key]


def eccentric_summary(name: str, recompute: bool = False) -> dict:
    """Everything this stage reports of one eccentric run: the record of Stage 4 (torques by the wall and the
    circle form, force), the exact Stokes values of its geometry (Newtonian runs), steadiness, the walls."""
    from experiment.validation import micropolar_eccentric_reference as exact_reference
    directory = run_directory(name)
    record = stage4.eccentric_record(name, directory, recompute)
    run = json.loads((directory / name / "run.json").read_text(encoding="utf-8"))
    meta = run["couette"]
    summary = stage4.cached_wall_sums(name, directory)
    factor = record["mass_over_lattice_mass"]
    entry = stage4.RUNS[name]
    result = {
        "name": name, "candidate": entry.get("candidate"), "micropolar": meta["micropolar"],
        "gap_in_spacings": meta["gap"] / meta["spacing"], "offset_in_spacings": meta["eccentricity"] / meta["spacing"],
        "eccentricity_ratio": meta["eccentricity"] / meta["gap"], "reynolds": meta["reynolds"], "mach": meta["mach"],
        "inner_radius": meta["inner_radius"], "outer_radius": meta["outer_radius"],
        "rotation_rate": meta["rotation_rate"], "far_viscosity": meta["far_viscosity"],
        "length_in_spacings": record["length_in_spacings"], "slip": entry.get("slip"),
        "steps": run["steps"], "timestep": run["timestep"], "end_time": run["steps"] * run["timestep"],
        "steps_per_second": run["steps_per_second"], "device_map": run["device_map"],
        "revision": run.get("revision"), "failure": run["failure"],
        "counters_zero": not any(run["counters"].values()), "states": record["states"],
        "rotor_torque_wall": record["rotor"]["wall"], "stator_torque_wall": record["stator"]["wall"],
        "rotor_torque_circle": record["rotor"]["circle_with_advection"],
        "stator_torque_circle": record["stator"]["circle_with_advection"],
        "rotor_torque_wall_standard_error": record["rotor"]["wall_standard_error"],
        "force": record["rotor_force_on_wall"], "load": record["rotor_force_magnitude"],
        "force_angle_degrees": record["rotor_force_angle_degrees"],
        "force_balance": float(np.hypot(*(np.asarray(record["rotor"]["force"]) + np.asarray(record["stator"]["force"])))
                               / record["rotor_force_magnitude"]),
        "checks": record["checks"],
    }
    # steadiness: second half against first half of the states of the last quarter
    torque_values = np.asarray(summary["rotor"]["torque_exchange"]["values"]) / factor
    force_values = np.asarray(summary["rotor"]["force_exchange"]["values"]) / factor
    half = torque_values.size // 2
    load_values = np.hypot(force_values[:, 0], force_values[:, 1])
    wall_term = record["rotor"]["wall"] - torque_values.mean()
    result["torque_change_between_halves"] = float((torque_values[half:].mean() - torque_values[:half].mean())
                                                   / (torque_values.mean() + wall_term))
    result["load_change_between_halves"] = float((load_values[half:].mean() - load_values[:half].mean())
                                                 / load_values.mean())
    result["load_standard_error"] = float(load_values.std(ddof=1) / np.sqrt(load_values.size))
    if not meta["micropolar"]:
        exact = exact_reference.stokes_force_and_torque(
            meta["inner_radius"], meta["outer_radius"], meta["eccentricity"],
            stage4.DENSITY * meta["far_viscosity"], meta["rotation_rate"])
        # the rotor acts on the fluid with minus the torque the fluid exerts on it
        result["exact"] = {"load": exact["force"][1], "rotor_torque": -exact["inner_torque"]}
        result["load_over_exact"] = result["load"] / exact["force"][1]
        result["force_y_over_exact"] = result["force"][1] / exact["force"][1]
        result["force_x_over_exact_load"] = result["force"][0] / exact["force"][1]
        for key in ("rotor_torque_wall", "rotor_torque_circle"):
            result[f"{key}_over_exact"] = result[key] / (-exact["inner_torque"])
        for key in ("stator_torque_wall", "stator_torque_circle"):
            result[f"{key}_over_exact"] = -result[key] / (-exact["inner_torque"])
        offsets = hydrodynamic_offsets(entry.get("candidate"), meta["gap"] / meta["spacing"])
        if offsets:
            hydrodynamic = exact_reference.stokes_force_and_torque(
                meta["inner_radius"] + offsets["inner"] * meta["spacing"],
                meta["outer_radius"] + offsets["outer"] * meta["spacing"], meta["eccentricity"],
                stage4.DENSITY * meta["far_viscosity"], meta["rotation_rate"])
            result["exact_hydrodynamic"] = {"load": hydrodynamic["force"][1],
                                            "rotor_torque": -hydrodynamic["inner_torque"], **offsets}
            result["load_over_exact_hydrodynamic"] = result["load"] / hydrodynamic["force"][1]
            result["rotor_torque_circle_over_exact_hydrodynamic"] = (result["rotor_torque_circle"]
                                                                     / (-hydrodynamic["inner_torque"]))
            result["stator_torque_circle_over_exact_hydrodynamic"] = (-result["stator_torque_circle"]
                                                                      / (-hydrodynamic["inner_torque"]))
    result["walls"] = wall_normal_profiles(name, directory)
    result["behind_walls"] = behind_walls(name, directory)
    result["levels"] = sample_levels(name, directory)
    if "displacement_from_profile" in record:
        result["displacement_map"] = {label: {key: record[f"displacement_{label}"][key]
                                              for key in ("mean", "min", "max", "largest_deviation_from_mean")}
                                      for label in ("from_pair_sums", "from_profile")}
    return result


BENCHMARK_RUNS = ("nd_e30_re1", "nd_e30_newtonian", "nd_e60_re1", "nd_e60_newtonian", "nd_e60_re1_ma001",
                  "nd_e60_re1_g200", "nd_e60_re10_g200")
FLOOR_RUNS = ("nd_e60_newtonian", "nd_e60_newtonian_shifted", "nd_e60_newtonian_rotated")
CANDIDATE_RUNS = {"thesis": ("thesis_e60_re1", "thesis_e60_re10"), "nd": ("nd_e60_re1", "nd_e60_re10"),
                  "fd": ("fd_e60_re1", "fd_e60_re10"), "ew": ("ew_e60_re1", "ew_e60_re10")}
SLIP_CASES = (("nd_e30", 5, 10.0), ("nd_e30", 2, 25.0), ("nd_e60", 5, 10.0), ("nd_e60", 2, 25.0))
# (micropolar run, its Newtonian control, eps, l / g): the micropolar effect in each setting, and at g = 200 dx
MICROPOLAR_PAIRS = (("nd_e60_k3_m5", "nd_e60_newtonian", 0.6, 0.1), ("fd_e60_k3_m5", "fd_e60_re10", 0.6, 0.1),
                    ("ew_e60_k3_m5", "ew_e60_re10", 0.6, 0.1), ("nd_e60_k3_m2", "nd_e60_newtonian", 0.6, 0.25),
                    ("fd_e60_k3_m2", "fd_e60_re10", 0.6, 0.25), ("ew_e60_k3_m2", "ew_e60_re10", 0.6, 0.25),
                    ("nd_e30_k3_m10_g200", "nd_e30_re10_g200", 0.3, 0.05),
                    ("nd_e60_k3_m10_g200", "nd_e60_re10_g200", 0.6, 0.05))


def _finished(name: str) -> bool:
    return (run_directory(name) / name / "run.json").exists()


def command_benchmark(arguments) -> int:
    lines, records = [], {}

    def say(text=""):
        lines.append(text)
        print(text)

    def summary_of(name):
        if name not in records:
            records[name] = eccentric_summary(name, arguments.recompute)
        return records[name]

    def percent(value):
        return f"{100.0 * (value - 1.0):+.2f} %"

    def benchmark_row(name):
        item = summary_of(name)
        walls, behind, levels = item["walls"], item["behind_walls"], item["levels"]
        return (f"{name:<26}{item['candidate']:>7}{item['eccentricity_ratio']:>6.2f}{item['reynolds']:>5g}"
                f"{item['gap_in_spacings']:>6g}{item['mach']:>7g}"
                f"{item['exact']['load']:>11.2f}{item['load']:>11.2f}{percent(item['load_over_exact']):>10}"
                f"{percent(item['force_y_over_exact']):>10}"
                f"{item['force_angle_degrees']:>8.2f}{item['force'][0]:>9.2f}"
                f"{item['exact']['rotor_torque']:>10.3f}{percent(item['rotor_torque_wall_over_exact']):>10}"
                f"{percent(item['stator_torque_wall_over_exact']):>10}"
                f"{percent(item['rotor_torque_circle_over_exact']):>10}"
                f"{percent(item['stator_torque_circle_over_exact']):>10}"
                + (f"{percent(item['load_over_exact_hydrodynamic']):>10}"
                   f"{percent(item['stator_torque_circle_over_exact_hydrodynamic']):>10}"
                   if "exact_hydrodynamic" in item else f"{'-':>10}{'-':>10}")
                + f"{100.0 * item['torque_change_between_halves']:>+9.3f}{100.0 * item['load_change_between_halves']:>+9.3f}"
                f"{behind['inside_rotor']:>8,}{behind['beyond_stator']:>8,}"
                f"{walls['rotor']['slope']:>+10.5f}{walls['rotor']['correlation']:>+7.2f}"
                f"{walls['stator']['slope']:>+10.5f}{walls['stator']['correlation']:>+7.2f}"
                f"{100.0 * walls['rotor']['velocity_min']:>+8.3f}{100.0 * walls['rotor']['velocity_max']:>+8.3f}"
                f"{walls['rotor']['pressure_min']:>9.0f}{walls['rotor']['pressure_max']:>9.0f}"
                f"{levels['density_min']:>9.2f}{levels['density_max']:>9.2f}"
                + (f"{levels['pressure_mean']:>9.1f}{levels['pressure_mean_change']:>+9.2f}"
                   if "pressure_mean" in levels else f"{'-':>9}{'-':>9}")
                + ("" if item["counters_zero"] and not item["failure"] else "  FAILED / counters"))
    header = (f"{'run':<26}{'cand.':>7}{'eps':>6}{'Re':>5}{'g/dx':>6}{'Ma':>7}{'load exact':>11}{'load':>11}{'':>10}"
              f"{'F_y':>10}"
              f"{'angle':>8}{'F_x':>9}{'T exact':>10}{'T wall R':>10}{'T wall S':>10}{'T circ R':>10}{'T circ S':>10}"
              f"{'load, hyd':>10}{'T cS, hyd':>10}"
              f"{'dT %':>9}{'dF %':>9}{'in rotor':>8}{'out st.':>8}{'slope R':>10}{'r':>7}{'slope S':>10}{'r':>7}"
              f"{'vn min %':>8}{'vn max %':>8}{'p min':>9}{'p max':>9}{'rho min':>9}{'rho max':>9}{'p mean':>9}"
              f"{'change':>9}")
    say("Newtonian flow between eccentric cylinders against the exact Stokes values (load in N/m, torque in N; the "
        "torque columns are reading / exact - 1; 'hyd': against the exact values for the hydrodynamic radii of the "
        "concentric control; dT, dF: second half - first half of the last quarter; slope: velocity "
        "into the wall over U per wall pressure over rho0 U^2, r its correlation over the 72 angle bins)")
    say()
    say("section 3: the benchmark in the nd setting")
    say(header)
    for name in BENCHMARK_RUNS:
        if _finished(name):
            say(benchmark_row(name))
    members = [name for name in FLOOR_RUNS if _finished(name)]
    if len(members) == 3:
        say()
        say("noise floor of the nd setting, three lattices at eps = 0.6, Reynolds number 10")
        say(header)
        for name in members:
            say(benchmark_row(name))
        floor = {}
        for key in ("load", "rotor_torque_wall", "rotor_torque_circle", "stator_torque_wall", "stator_torque_circle"):
            values = np.array([summary_of(name)[key] for name in members])
            floor[f"{key}_spread"] = float((values.max() - values.min()) / abs(values.mean()))
        angles = np.array([summary_of(name)["force_angle_degrees"] for name in members])
        floor["force_angle_spread_degrees"] = float(angles.max() - angles.min())
        records["_floor_newtonian"] = floor
        say("   spread (max - min) / mean: load {load_spread:.2e}, rotor torque wall form {rotor_torque_wall_spread:.2e}, "
            "circle form {rotor_torque_circle_spread:.2e}, stator torque wall form {stator_torque_wall_spread:.2e}, "
            "circle form {stator_torque_circle_spread:.2e}; direction {force_angle_spread_degrees:.3f} degrees"
            .format(**floor))

    # section 2: micropolar run and slip-form Newtonian run of the same case, both against the Newtonian control
    say()
    say("section 2: the rule between eccentric cylinders. D_T = T / T_control - 1 by four readings (rotor and stator, "
        "wall and circle form), the change of the load and the turn of the force, for the micropolar run and for the "
        "Newtonian run in the slip-form geometry; 'corrected' = the same with the error that reading makes between "
        "concentric cylinders at the same l taken off")
    concentric_errors = concentric_reading_errors()
    records["_concentric_reading_errors"] = concentric_errors
    readings = (("rotor_torque_wall", "wall R"), ("stator_torque_wall", "wall S"), ("rotor_torque_circle", "circ R"),
                ("stator_torque_circle", "circ S"))
    say(f"{'case':<10}{'l/dx':>6}{'run':>22}" + "".join(f"{label:>9}" for _, label in readings)
        + "".join(f"{label + ' c.':>10}" for _, label in readings) + f"{'load':>10}{'turn deg':>10}{'F_x':>9}{'F_y':>10}")
    comparison = {}
    for prefix, gap_over_two_lengths, length in SLIP_CASES:
        control_name = f"{prefix}_newtonian"
        rows = {"micropolar": f"{prefix}_k3_m{gap_over_two_lengths}", "slip": f"{prefix}_slip_m{gap_over_two_lengths}"}
        if not all(_finished(name) for name in (control_name, *rows.values())):
            continue
        control = summary_of(control_name)
        entry = {}
        for kind, name in rows.items():
            item = summary_of(name)
            changes = {key: item[key] / control[key] - 1.0 for key, _ in readings}
            errors = concentric_errors.get(f"{kind}_m{gap_over_two_lengths}", {})
            corrected = {key: changes[key] - errors[key] for key in changes} if errors else {}
            entry[kind] = {"run": name, "changes": changes, "corrected": corrected,
                           "load_change": item["load"] / control["load"] - 1.0,
                           "turn_degrees": item["force_angle_degrees"] - control["force_angle_degrees"],
                           "force": item["force"]}
            say(f"{prefix:<10}{length:>6g}{name:>22}" + "".join(f"{changes[key]:>9.4f}" for key, _ in readings)
                + "".join((f"{corrected[key]:>10.4f}" if corrected else f"{'-':>10}") for key, _ in readings)
                + f"{100.0 * entry[kind]['load_change']:>+9.2f}%{entry[kind]['turn_degrees']:>+10.3f}"
                f"{item['force'][0]:>9.2f}{item['force'][1]:>10.2f}")
        comparison[f"{prefix}_m{gap_over_two_lengths}"] = entry
    records["_slip_comparison"] = comparison
    if comparison:
        say()
        say("the rule over the micropolar run: (D_T of the slip-form run) / (D_T of the micropolar run), mean of the "
            "four corrected readings; the same for the change of the load")
        say(f"{'case':<14}{'D_T micropolar':>16}{'D_T rule':>12}{'ratio':>9}{'spread of the four':>20}"
            f"{'load micropolar':>17}{'load rule':>12}{'ratio':>9}")
        for label, entry in comparison.items():
            def mean_of(kind):
                source = entry[kind]["corrected"] or entry[kind]["changes"]
                values = np.array(list(source.values()))
                return float(values.mean()), float(values.max() - values.min())
            micropolar_mean, micropolar_spread = mean_of("micropolar")
            slip_mean, slip_spread = mean_of("slip")
            entry["summary"] = {"micropolar_torque_change": micropolar_mean, "rule_torque_change": slip_mean,
                                "ratio": slip_mean / micropolar_mean,
                                "load_ratio": entry["slip"]["load_change"] / entry["micropolar"]["load_change"]}
            say(f"{label:<14}{micropolar_mean:>16.4f}{slip_mean:>12.4f}{slip_mean / micropolar_mean:>9.3f}"
                f"{micropolar_spread:>10.4f}{slip_spread:>10.4f}"
                f"{100.0 * entry['micropolar']['load_change']:>+16.2f}%{100.0 * entry['slip']['load_change']:>+11.2f}%"
                f"{entry['slip']['load_change'] / entry['micropolar']['load_change']:>9.3f}")
    # the exact Stokes values of the slip-form geometry: what the rule predicts without a run
    say()
    say("the rule from the exact Stokes solution alone (no run): slip-form geometry over the nominal geometry")
    say(f"{'eps':>5}{'l/dx':>6}{'T exact':>11}{'T slip':>11}{'D_T rule':>10}{'load exact':>12}{'load slip':>12}"
        f"{'change':>9}")
    from experiment.validation import micropolar_eccentric_reference as exact_reference
    rule_exact = {}
    for offset in (0.03, 0.06):
        for length in (10.0, 25.0, 5.0, 2.5):
            displacement = DISPLACEMENT_OVER_LENGTH * length * 1.0e-3
            nominal = exact_reference.stokes_force_and_torque(0.4, 0.5, offset, 10.0, 2.5)
            slip = exact_reference.stokes_force_and_torque(
                0.4 + displacement, 0.5 - displacement, offset, 10.0,
                2.5 * (0.4 - displacement) / (0.4 + displacement))
            rule_exact[f"e{round(offset * 1000)}_l{length:g}"] = {
                "torque_change": slip["inner_torque"] / nominal["inner_torque"] - 1.0,
                "load_change": slip["force"][1] / nominal["force"][1] - 1.0}
            say(f"{offset / 0.1:>5g}{length:>6g}{-nominal['inner_torque']:>11.4f}{-slip['inner_torque']:>11.4f}"
                f"{slip['inner_torque'] / nominal['inner_torque'] - 1.0:>10.4f}{nominal['force'][1]:>12.3f}"
                f"{slip['force'][1]:>12.3f}{100.0 * (slip['force'][1] / nominal['force'][1] - 1.0):>+8.2f}%")
    records["_rule_from_exact"] = rule_exact

    # noise floor of D_T: the three lattices of the micropolar run at l = 10 dx with their controls
    lattice_pairs = [(f"nd_e60_k3_m5{lattice}", f"nd_e60_newtonian{lattice}") for lattice in LATTICES]
    if all(_finished(name) for pair in lattice_pairs for name in pair):
        say()
        say("noise floor of the change against the control: nd_e60_k3_m5 over nd_e60_newtonian on the three lattices")
        say(f"{'lattice':<12}" + "".join(f"{label:>9}" for _, label in readings) + f"{'load':>10}{'turn deg':>10}")
        values = []
        for lattice, (name, control_name) in zip(LATTICES, lattice_pairs):
            item, control = summary_of(name), summary_of(control_name)
            row = [item[key] / control[key] - 1.0 for key, _ in readings]
            row += [item["load"] / control["load"] - 1.0, item["force_angle_degrees"] - control["force_angle_degrees"]]
            values.append(row)
            say(f"{lattice or 'as generated':<12}" + "".join(f"{value:>9.4f}" for value in row[:4])
                + f"{100.0 * row[4]:>+9.2f}%{row[5]:>+10.3f}")
        values = np.array(values)
        spread = values.max(axis=0) - values.min(axis=0)
        records["_floor_change"] = {"values": values.tolist(), "spread": spread.tolist()}
        say(f"{'max - min':<12}" + "".join(f"{value:>9.4f}" for value in spread[:4])
            + f"{100.0 * spread[4]:>9.2f}%{spread[5]:>10.3f}")

    # the micropolar effect in each setting (the candidates of section 4) and at g = 200 dx, l = 10 dx, with the
    # rule from the exact Stokes solution beside it
    finished_pairs = [pair for pair in MICROPOLAR_PAIRS if _finished(pair[0]) and _finished(pair[1])]
    if finished_pairs:
        say()
        say("micropolar run over its Newtonian control, by setting and resolution (uncorrected readings), and the "
            "rule from the exact Stokes solution of the slip-form geometry")
        say(f"{'run':<22}{'eps':>5}{'l / g':>7}{'g/dx':>6}" + "".join(f"{label:>9}" for _, label in readings)
            + f"{'mean':>9}{'load':>10}{'turn deg':>10}{'rule D_T':>10}{'rule load':>11}{'D_T ratio':>11}"
            f"{'load ratio':>11}")
        pair_records = {}
        for name, control_name, ratio, length_over_gap in finished_pairs:
            item, control = summary_of(name), summary_of(control_name)
            changes = [item[key] / control[key] - 1.0 for key, _ in readings]
            load_change = item["load"] / control["load"] - 1.0
            displacement = DISPLACEMENT_OVER_LENGTH * length_over_gap * 0.1
            nominal = exact_reference.stokes_force_and_torque(0.4, 0.5, ratio * 0.1, 10.0, 2.5)
            slip = exact_reference.stokes_force_and_torque(0.4 + displacement, 0.5 - displacement, ratio * 0.1, 10.0,
                                                           2.5 * (0.4 - displacement) / (0.4 + displacement))
            rule_torque = slip["inner_torque"] / nominal["inner_torque"] - 1.0
            rule_load = slip["force"][1] / nominal["force"][1] - 1.0
            pair_records[name] = {"control": control_name, "changes": changes, "mean_change": float(np.mean(changes)),
                                  "load_change": load_change,
                                  "turn_degrees": item["force_angle_degrees"] - control["force_angle_degrees"],
                                  "rule_torque_change": rule_torque, "rule_load_change": rule_load}
            say(f"{name:<22}{ratio:>5g}{length_over_gap:>7g}{item['gap_in_spacings']:>6g}"
                + "".join(f"{value:>9.4f}" for value in changes) + f"{np.mean(changes):>9.4f}"
                f"{100.0 * load_change:>+9.2f}%{pair_records[name]['turn_degrees']:>+10.3f}{rule_torque:>10.4f}"
                f"{100.0 * rule_load:>+10.2f}%{rule_torque / np.mean(changes):>11.3f}{rule_load / load_change:>11.3f}")
        records["_micropolar_pairs"] = pair_records

    # section 4: the candidates on the benchmark
    say()
    say("section 4: the candidates on the benchmark at eps = 0.6 (thesis = the thesis configuration; nd = density "
        "diffusion off; fd = density diffusion between fluid pairs only)")
    say(header)
    for candidate, names in CANDIDATE_RUNS.items():
        for name in names:
            if _finished(name):
                say(benchmark_row(name))
    extra = [name for name in ("e60_newtonian", "leak_e60", "fd_e60_newtonian") if _finished(name)]
    if extra:
        say("the same case in earlier runs (states of the last quarter only, or of Stage 4)")
        for name in extra:
            say(benchmark_row(name))
    say()
    say("fluid behind a wall at the end of each candidate run (more than 0.5 dx behind the surface)")
    say(f"{'run':<26}{'inside rotor':>13}{'deepest (dx)':>13}{'beyond stator':>14}{'deepest (dx)':>13}{'share':>9}"
        f"{'first seen at t / t_end':>25}")
    for names in CANDIDATE_RUNS.values():
        for name in names:
            if _finished(name):
                behind = summary_of(name)["behind_walls"]
                first = behind["first_seen_at_fraction_of_run"]
                say(f"{name:<26}{behind['inside_rotor']:>13,}{behind['deepest_rotor']:>13.2f}"
                    f"{behind['beyond_stator']:>14,}{behind['deepest_stator']:>13.2f}{100.0 * behind['share']:>8.2f}%"
                    + (f"{first:>25.3f}" if first is not None else f"{'none':>25}"))
    if arguments.json:
        target = _REPO_ROOT / arguments.json
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(records, indent=1), encoding="utf-8")
        target.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return 0


def concentric_reading_errors() -> dict:
    """What each of the four torque readings makes of D between concentric cylinders, minus D of the fits, for the
    micropolar runs of the nd setting (Stage 4: nd_g100_k3_m5, nd_g100_k3_m2 over nd_g100_newtonian) and for the
    slip-form runs of this stage (nd_g100_slip_m5, nd_g100_slip_m2). Keys: micropolar_m5, micropolar_m2, slip_m5,
    slip_m2 -> {reading: error}."""
    result = {}
    stage4_directory = _REPO_ROOT / STAGE4_RUNS_DIRECTORY
    names = ["nd_g100_newtonian", "nd_g100_k3_m5", "nd_g100_k3_m2"]
    if not all((stage4_directory / name / "run.json").exists() for name in names):
        return result
    analysis = stage4.analyse_concentric(names, stage4_directory)["runs"]
    control_torques = stage4.wall_torques(stage4.cached_wall_sums("nd_g100_newtonian", stage4_directory))
    control_fit = analysis["nd_g100_newtonian"]["fit"]["potential_vortex_constant"]

    def reading_changes(torques):
        return {"rotor_torque_wall": torques["rotor"]["wall"] / control_torques["rotor"]["wall"] - 1.0,
                "stator_torque_wall": torques["stator"]["wall"] / control_torques["stator"]["wall"] - 1.0,
                "rotor_torque_circle": torques["rotor"]["circle"] / control_torques["rotor"]["circle"] - 1.0,
                "stator_torque_circle": torques["stator"]["circle"] / control_torques["stator"]["circle"] - 1.0}
    for gap_over_two_lengths in (5, 2):
        name = f"nd_g100_k3_m{gap_over_two_lengths}"
        changes = reading_changes(stage4.wall_torques(stage4.cached_wall_sums(name, stage4_directory)))
        fit_change = analysis[name]["relative_torque_change"]
        result[f"micropolar_m{gap_over_two_lengths}"] = {key: value - fit_change for key, value in changes.items()}
        result[f"micropolar_m{gap_over_two_lengths}"]["fit_change"] = fit_change
        result[f"micropolar_m{gap_over_two_lengths}"]["exact_change"] = analysis[name]["exact_hydrodynamic"]
        slip_name = f"nd_g100_slip_m{gap_over_two_lengths}"
        if _finished(slip_name):
            directory = run_directory(slip_name)
            slip_analysis = stage4.analyse_concentric([slip_name], directory)["runs"][slip_name]
            # a Newtonian fit: the torque is 4 pi mu beta, whatever the radii and the rate
            fit_change = slip_analysis["fit"]["potential_vortex_constant"] / control_fit - 1.0
            changes = reading_changes(stage4.wall_torques(stage4.cached_wall_sums(slip_name, directory)))
            result[f"slip_m{gap_over_two_lengths}"] = {key: value - fit_change for key, value in changes.items()}
            result[f"slip_m{gap_over_two_lengths}"]["fit_change"] = fit_change
            displacement = RUNS[slip_name]["slip"]["displacement_in_spacings"] * 1.0e-3
            result[f"slip_m{gap_over_two_lengths}"]["rule_change_nominal"] = slip_rule_torque_change(0.4, 0.5,
                                                                                                    displacement)
            result[f"slip_m{gap_over_two_lengths}"]["beta_over_nominal"] = slip_analysis.get("beta_over_nominal")
    return result


# ---------------------------------------------------------------------------------------------------------------
# section 4: the concentric control of each candidate
# ---------------------------------------------------------------------------------------------------------------
CONCENTRIC_SETS = {"thesis": ("thesis_g100_control", "g100_k3_m5", "g100_k3_m2"),
                   "nd": ("nd_g100_control", "nd_g100_k3_m5", "nd_g100_k3_m2"),
                   "fd": ("fd_g100_newtonian", "fd_g100_k3_m5", "fd_g100_k3_m2"),
                   "ew": ("ew_g100_newtonian", "ew_g100_k3_m5", "ew_g100_k3_m2"),
                   # the fd candidate five times longer (10 g^2 / nu_inf)
                   "fd_long": ("fd_g100_newtonian_long", "fd_g100_k3_m5_long")}
STAGE4_CONTROL_OF = {"thesis": "g100_newtonian", "nd": "nd_g100_newtonian"}


def command_concentric(arguments) -> int:
    lines, records = [], {}

    def say(text=""):
        lines.append(text)
        print(text)
    say("Concentric control at g = 100 dx (Newtonian, Reynolds number 10) of each candidate, last quarter of the run")
    say(f"{'candidate':<10}{'run':<22}{'beta / nominal':>15}{'inner offset':>13}{'outer offset':>13}{'fit rms / U':>12}"
        f"{'v_r mean / U':>13}{'v_r max / U':>12}{'sector spread':>14}{'steady':>9}{'rho min':>10}{'rho max':>10}"
        f"{'rho mean':>11}{'p mean':>9}{'change':>9}")
    for candidate, (control_name, *micropolar_names) in CONCENTRIC_SETS.items():
        controls = [name for name in (control_name, STAGE4_CONTROL_OF.get(candidate)) if name and _finished(name)]
        for name in controls:
            directory = run_directory(name)
            record = stage4.analyse_concentric([name], directory)["runs"][name]
            radial = np.asarray(record["profile"]["radial_velocity"]) / 1.0
            levels = sample_levels(name, directory)
            entry = {"candidate": candidate, "beta_over_nominal": record["beta_over_nominal"],
                     "inner_offset_in_spacings": record["inner_offset_in_spacings"],
                     "outer_offset_in_spacings": record["outer_offset_in_spacings"],
                     "fit_rms_over_wall_speed": record["fit"]["fit_rms_over_wall_speed"],
                     "radial_velocity_mean_over_wall_speed": float(radial.mean()),
                     "radial_velocity_max_over_wall_speed": record["radial_velocity_max_over_wall_speed"],
                     "sector_spread": record["sector_spread"], "steady_change": record["steady_change"],
                     "levels": levels, "revision": record["revision"], "counters_zero": record["counters_zero"],
                     "failure": record["failure"]}
            records[name] = entry
            say(f"{candidate:<10}{name:<22}{entry['beta_over_nominal']:>15.6f}{entry['inner_offset_in_spacings']:>13.4f}"
                f"{entry['outer_offset_in_spacings']:>13.4f}{entry['fit_rms_over_wall_speed']:>12.2e}"
                f"{entry['radial_velocity_mean_over_wall_speed']:>+13.2e}"
                f"{entry['radial_velocity_max_over_wall_speed']:>12.2e}{entry['sector_spread']:>14.2e}"
                f"{entry['steady_change']:>9.1e}{levels['density_min']:>10.3f}{levels['density_max']:>10.3f}"
                + (f"{levels['density_mean']:>11.4f}{levels['pressure_mean']:>9.2f}{levels['pressure_mean_change']:>+9.3f}"
                   if "density_mean" in levels else f"{'-':>11}{'-':>9}{'-':>9}"))
    say()
    say("the micropolar runs of the pilot with each candidate (K = 3, w = 0 on both walls): D against the exact "
        "solution with the hydrodynamic radii of the control")
    say(f"{'candidate':<10}{'run':<18}{'l/dx':>6}{'D_SPH':>10}{'D_exact':>10}{'D_SPH / D_exact':>17}{'joint':>10}"
        f"{'steady':>9}")
    for candidate, (control_name, *micropolar_names) in CONCENTRIC_SETS.items():
        # the micropolar runs of Stage 4 go with the control of Stage 4 (the same directory)
        control = STAGE4_CONTROL_OF.get(candidate, control_name)
        names = [name for name in micropolar_names if _finished(name)]
        if not names or not _finished(control):
            continue
        directory = run_directory(control)
        if any(run_directory(name) != directory for name in names):
            continue
        analysis = stage4.analyse_concentric([control, *names], directory)["runs"]
        for name in names:
            record = analysis[name]
            records[name] = {"candidate": candidate, "control": control,
                             "length_in_spacings": record["length_in_spacings"],
                             "relative_torque_change": record["relative_torque_change"],
                             "relative_torque_change_joint": record["relative_torque_change_joint"],
                             "exact_hydrodynamic": record["exact_hydrodynamic"],
                             "sph_over_exact": record["sph_over_exact"], "steady_change": record["steady_change"]}
            say(f"{candidate:<10}{name:<18}{record['length_in_spacings']:>6g}{record['relative_torque_change']:>10.5f}"
                f"{record['exact_hydrodynamic']:>10.5f}{record['sph_over_exact']:>17.4f}"
                f"{record['relative_torque_change_joint']:>10.5f}{record['steady_change']:>9.1e}")
    if arguments.json:
        target = _REPO_ROOT / arguments.json
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(records, indent=1), encoding="utf-8")
        target.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return 0

# ---------------------------------------------------------------------------------------------------------------
# section 2 (host): why the rule over-predicts between eccentric cylinders - the plane film
# ---------------------------------------------------------------------------------------------------------------
def plane_film_function(kind: str, gap, length: float, displacement: float):
    """f(h) of a plane film of thickness h between two walls that carry w = 0: the flux of the pressure-driven
    part is - f(h) (dp/dx) / (12 mu_inf). Newtonian of viscosity mu_inf: h^3. Rule: (h - 2 delta)^3. Micropolar,
    exact (the two ordinary differential equations of the model solved between the walls):
    h^3 - 12 delta h ((h / 2) coth(h / (2 l)) - l) = h^3 - 6 delta h^2 + 12 delta l h - ... for h >> l."""
    gap = np.asarray(gap, dtype=np.float64)
    if kind == "newtonian":
        return gap ** 3
    if kind == "rule":
        return (gap - 2.0 * displacement) ** 3
    return gap ** 3 - 12.0 * displacement * gap * (0.5 * gap / np.tanh(0.5 * gap / length) - length)


def plane_shear_gap(kind: str, gap, length: float, displacement: float):
    """The gap that carries the shear-driven part: wall shear stress = mu_inf U / gap. Newtonian: h. Rule:
    h - 2 delta. Micropolar, exact: h - 2 delta tanh(h / (2 l))."""
    gap = np.asarray(gap, dtype=np.float64)
    if kind == "newtonian":
        return gap
    if kind == "rule":
        return gap - 2.0 * displacement
    return gap - 2.0 * displacement * np.tanh(0.5 * gap / length)


def plane_film_numerical(gap_over_length: float, coupling_ratio: float = 3.0):
    """f(h) and g_s(h) of the plane film from a direct numerical solution of the two ordinary differential
    equations of the model (nu u'' + c w' = -G, xi w'' - c (u' + 2 w) = 0; u = w = 0 on both walls for the
    pressure-driven part, u = U on one wall for the shear-driven part), in units of l: the check of the closed
    forms of plane_film_function and plane_shear_gap."""
    from scipy.integrate import solve_bvp
    shear_viscosity, length, gap = 1.0, 1.0, gap_over_length
    coupling = coupling_ratio * shear_viscosity
    viscosity = shear_viscosity + coupling
    spin_viscosity = length ** 2 * coupling * (2.0 * viscosity - coupling) / viscosity
    far_viscosity = viscosity - 0.5 * coupling

    def equations(source):
        def right_hand_side(_coordinate, state):
            velocity_slope, spin, spin_slope = state[1], state[2], state[3]
            return np.vstack([velocity_slope, (source - coupling * spin_slope) / viscosity, spin_slope,
                              coupling * (velocity_slope + 2.0 * spin) / spin_viscosity])
        return right_hand_side
    mesh = np.linspace(0.0, gap, 2001)
    driven = solve_bvp(equations(-1.0), lambda low, high: np.array([low[0], high[0], low[2], high[2]]), mesh,
                       np.zeros((4, mesh.size)), tol=1e-10, max_nodes=200000)
    fine = np.linspace(0.0, gap, 200001)
    film = 12.0 * far_viscosity * float(np.trapezoid(driven.sol(fine)[0], fine))
    sheared = solve_bvp(equations(0.0), lambda low, high: np.array([low[0] - 1.0, high[0], low[2], high[2]]), mesh,
                        np.vstack([1.0 - mesh / gap, -np.ones_like(mesh) / gap, np.zeros_like(mesh),
                                   np.zeros_like(mesh)]), tol=1e-10, max_nodes=200000)
    stress = -(viscosity * sheared.sol(0.0)[1] + coupling * sheared.sol(0.0)[2])
    return film, float(far_viscosity / stress)


def thin_film_load_and_torque(kind: str, eccentricity_ratio: float, length: float, displacement: float,
                              radius: float = 0.4, clearance: float = 0.1, points: int = 20000):
    """Full-film Reynolds solution of a journal bearing with the film function of `kind`, in units of 6 mu_inf U R
    (pressure) : (load per unit depth) / (6 mu_inf U R^2) and (journal torque) / (6 mu_inf U R^3)."""
    angle = 2.0 * np.pi * np.arange(points) / points                 # from the widest gap
    step = 2.0 * np.pi / points
    gap = clearance * (1.0 + eccentricity_ratio * np.cos(angle))
    conductance = plane_film_function(kind, gap, length, displacement)
    constant = (gap / conductance).sum() / (1.0 / conductance).sum()
    gradient = (gap - constant) / conductance                        # dp/d(angle) / (6 mu_inf U R)
    pressure = np.cumsum(gradient) * step
    pressure -= pressure.mean()
    load = float(np.hypot((pressure * np.cos(angle)).sum(), (pressure * np.sin(angle)).sum()) * step)
    shear = (1.0 / (6.0 * radius * plane_shear_gap(kind, gap, length, displacement)) + 0.5 * gap * gradient / radius)
    return load, float(shear.sum() * step)


def command_film(arguments) -> int:
    from experiment.validation import micropolar_eccentric_reference as exact_reference
    lines, records = [], {}

    def say(text=""):
        lines.append(text)
        print(text)
    say("Plane film between two walls with w = 0 (K = 3, delta = 3 l / 8): the pressure-driven flux is "
        "- f(h) (dp/dx) / (12 mu_inf), the shear stress of the shear-driven part mu_inf U / g_s(h)")
    say("   exact:  f = h^3 - 12 delta h ((h / 2) coth(h / (2 l)) - l),      g_s = h - 2 delta tanh(h / (2 l))")
    say("   rule:   f = (h - 2 delta)^3,                                      g_s = h - 2 delta")
    say()
    say(f"{'h / l':>7}{'f exact / h^3':>15}{'f rule / h^3':>14}{'rule / exact':>14}{'g_s exact / h':>15}{'g_s rule / h':>14}")
    for ratio in (1.0, 1.6, 2.0, 2.8, 4.0, 7.0, 8.0, 10.0, 14.0, 20.0, 40.0):
        length, gap = 1.0, ratio
        displacement = DISPLACEMENT_OVER_LENGTH * length
        exact = float(plane_film_function("exact", gap, length, displacement)) / gap ** 3
        rule = float(plane_film_function("rule", gap, length, displacement)) / gap ** 3
        say(f"{ratio:>7g}{exact:>15.4f}{rule:>14.4f}{rule / exact:>14.4f}"
            f"{float(plane_shear_gap('exact', gap, length, displacement)) / gap:>15.4f}"
            f"{float(plane_shear_gap('rule', gap, length, displacement)) / gap:>14.4f}")
    worst = 0.0
    for ratio in (1.6, 2.8, 4.0, 7.0, 14.0):
        film_numerical, shear_gap_numerical = plane_film_numerical(ratio)
        displacement = DISPLACEMENT_OVER_LENGTH
        worst = max(worst, abs(film_numerical / float(plane_film_function("exact", ratio, 1.0, displacement)) - 1.0),
                    abs(shear_gap_numerical / float(plane_shear_gap("exact", ratio, 1.0, displacement)) - 1.0))
    say(f"   check: the two closed forms against a numerical solution of the two ordinary differential equations at "
        f"h / l = 1.6, 2.8, 4, 7, 14: largest relative difference {worst:.1e}")
    say()
    say("Journal bearing in the thin-film approximation (R = 0.4, c = 0.1; ratios only, the thin-film values are "
        "11 % from the exact Stokes ones at this clearance): change of the load and of the journal torque against "
        "the Newtonian film, with the exact film and with the rule's film; beside it the rule from the exact "
        "Stokes solution of the slip-form geometry")
    say(f"{'eps':>5}{'l / g':>7}{'h_min / l':>10}{'load: exact film':>18}{'rule film':>11}{'rule, Stokes':>14}"
        f"{'torque: exact film':>20}{'rule film':>11}{'rule, Stokes':>14}")
    for ratio in (0.3, 0.6):
        for length_over_gap in (0.05, 0.1, 0.25):
            length = length_over_gap * 0.1
            displacement = DISPLACEMENT_OVER_LENGTH * length
            base = thin_film_load_and_torque("newtonian", ratio, length, displacement)
            exact = thin_film_load_and_torque("exact", ratio, length, displacement)
            rule = thin_film_load_and_torque("rule", ratio, length, displacement)
            nominal = exact_reference.stokes_force_and_torque(0.4, 0.5, ratio * 0.1, 10.0, 2.5)
            slip = exact_reference.stokes_force_and_torque(0.4 + displacement, 0.5 - displacement, ratio * 0.1, 10.0,
                                                           2.5 * (0.4 - displacement) / (0.4 + displacement))
            entry = {"load_exact_film": exact[0] / base[0] - 1.0, "load_rule_film": rule[0] / base[0] - 1.0,
                     "load_rule_stokes": slip["force"][1] / nominal["force"][1] - 1.0,
                     "torque_exact_film": exact[1] / base[1] - 1.0, "torque_rule_film": rule[1] / base[1] - 1.0,
                     "torque_rule_stokes": slip["inner_torque"] / nominal["inner_torque"] - 1.0,
                     "rotor_speed_term": 2.0 * displacement / 0.4}
            records[f"eps{ratio:g}_l{length_over_gap:g}"] = entry
            say(f"{ratio:>5g}{length_over_gap:>7g}{(1.0 - ratio) / length_over_gap:>10.1f}"
                f"{100.0 * entry['load_exact_film']:>+17.2f}%{100.0 * entry['load_rule_film']:>+10.2f}%"
                f"{100.0 * entry['load_rule_stokes']:>+13.2f}%{entry['torque_exact_film']:>20.4f}"
                f"{entry['torque_rule_film']:>11.4f}{entry['torque_rule_stokes']:>14.4f}")
    say("   (the thin film has no curvature: its torque change lacks the factor (R1 - delta) / (R1 + delta) of the "
        "rotor's apparent speed, -2 delta / R1 = -0.009, -0.019, -0.047 for l / g = 0.05, 0.1, 0.25)")
    if arguments.out:
        target = _REPO_ROOT / arguments.out
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        target.with_suffix(".json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    listing.set_defaults(handler=command_list)
    queue = commands.add_parser("queue")
    queue.add_argument("names", nargs="+")
    queue.add_argument("--device", type=int, required=True)
    queue.add_argument("--steps", type=int, default=None, help="a short run for a smoke test")
    queue.add_argument("--sample-steps", type=int, default=None)
    queue.set_defaults(handler=command_queue)
    rule = commands.add_parser("rule")
    rule.add_argument("--analysis", default="data/micropolar/stage4/couette_stage4.json")
    rule.add_argument("--out", default=None, help="text file relative to the repository root; a .json beside it")
    rule.set_defaults(handler=command_rule)
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--recompute", action="store_true", help="ignore the wall_sums.json kept beside each run")
    benchmark.add_argument("--json", default=None, help="relative to the repository root; a .txt goes beside it")
    benchmark.set_defaults(handler=command_benchmark)
    concentric = commands.add_parser("concentric")
    concentric.add_argument("--json", default=None, help="relative to the repository root; a .txt goes beside it")
    concentric.set_defaults(handler=command_concentric)
    film = commands.add_parser("film")
    film.add_argument("--out", default=None, help="text file relative to the repository root; a .json beside it")
    film.set_defaults(handler=command_film)
    arguments = parser.parse_args()
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
