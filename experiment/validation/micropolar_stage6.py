"""
micropolar_stage6.py - Stage 6 of docs/micropolar_design.md: the runs of the chapter on the tag thesis-ch5-rc2.

  list        The run table of this stage: name, section, control, generator options.
  runner      The one queue runner of the stage. It reads a plan file (JSON, a list of jobs in order), starts a job
              when every device it names is idle and the job is the first unfinished one of each of them, and reads
              the plan again before every decision, so the plan can be edited while it runs. A file STOP beside the
              plan lets the running jobs finish and ends the runner; a file KILL terminates them.
  plan        Write or extend a plan file from run names of the table.
  concentric  Analysis of the concentric rows (fits, D, exact solution, rule in its slip form, drift of the levels).
  eccentric   Analysis of the eccentric rows (load, torque, pressure level; film function and rule on the host).

Every cylinder case of this stage carries numerics.delta_fluid_only: true (the decision after Stage 5); everything
else is the thesis configuration. Cases under cases/couette_stage6, raw runs under logs/micropolar/stage6 (not in
the repository). The names of the concentric rows are those of Stage 4.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.validation import micropolar_stage4 as stage4  # noqa: E402

CASES = "cases/couette_stage6"
RUNS_DIRECTORY = "logs/micropolar/stage6"
KEY_OPTIONS = ["--delta-fluid-only"]
LATTICES = stage4.LATTICES
STATE_COUNT = 12                     # states of the last quarter (Stage 4 and 5: 24; the disk)
PLAN = "logs/micropolar/stage6/plan.json"


def _ratio_tag(coupling_ratio: float) -> str:
    return "k" + f"{coupling_ratio:g}".replace(".", "")


def _build_table() -> dict:
    runs = {}

    def newtonian(name, gap, step_from, lattice="", eccentricity=0, section=(), **keys):
        runs[name] = {
            "generate": ["--gap-in-spacings", str(gap), "--eccentricity-in-spacings", str(eccentricity),
                         "--step-from-coupling-ratio", str(step_from[0]),
                         "--step-from-gap-over-two-lengths", str(step_from[1]), *LATTICES[lattice], *KEY_OPTIONS],
            "control": None, "gap": gap, "eccentricity": eccentricity, "candidate": "fd", "reynolds": 10.0,
            "lattice": lattice, "section": list(section), **keys}

    def micropolar(name, gap, coupling_ratio, gap_over_two_lengths, control, lattice="", eccentricity=0, section=(),
                   **keys):
        runs[name] = {
            "generate": ["--gap-in-spacings", str(gap), "--eccentricity-in-spacings", str(eccentricity),
                         "--coupling-ratio", str(coupling_ratio), "--gap-over-two-lengths", str(gap_over_two_lengths),
                         "--rotor-micro-rotation-factor", "0", *LATTICES[lattice], *KEY_OPTIONS],
            "control": control, "gap": gap, "eccentricity": eccentricity, "candidate": "fd", "reynolds": 10.0,
            "coupling_ratio": coupling_ratio, "gap_over_two_lengths": gap_over_two_lengths, "lattice": lattice,
            "section": list(section), **keys}

    # 2.1 concentric. Controls on the three lattices at g = 50, 100, 200 dx; below g = 400 dx the acoustic limit sets
    # dt for every K, so one control per lattice serves all rows. At g = 400 dx the viscous limit of K = 3 is below
    # the acoustic one: the K = 3 rows have their own control.
    for gap in (50, 100, 200):
        for lattice in LATTICES:
            newtonian(f"g{gap:03d}_newtonian{lattice}", gap, (3, 5), lattice, section=("floor",))
    newtonian("g400_newtonian", 400, (1, 40), section=("control",))
    newtonian("g400_newtonian_k3", 400, (3, 10), section=("control",))
    # resolution study: K = 3, m = 5 and m = 10 at l / dx = 5, 10, 20
    for gap, gap_over_two_lengths in ((50, 5), (100, 5), (200, 5), (100, 10), (200, 10), (400, 10)):
        control = "g400_newtonian_k3" if gap == 400 else f"g{gap:03d}_newtonian"
        micropolar(f"g{gap:03d}_k3_m{gap_over_two_lengths}", gap, 3, gap_over_two_lengths, control,
                   section=("resolution",))
    # sweep: K = 0.5, 1, 3 and m = 1, 2, 5 (g = 100 dx), 10, 20 (200 dx), 40 (400 dx)
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
    # the row added by the task of Stage 6: m = 20, K = 3 at l = 10 dx (g = 400 dx)
    micropolar("g400_k3_m20", 400, 3, 20, "g400_newtonian_k3", section=("sweep", "resolution"))

    # beyond the task: the concentric row at K = 1, g = 200 dx, l = 20 dx, which the sweep does not have; it gives
    # the reading errors of the eccentric rows with the same K, l and g
    micropolar("g200_k1_m5", 200, 1, 5, "g200_newtonian", section=("extra",))

    # 2.2 eccentric: K = 1 and 3, eps = 0.3 and 0.6; g = 100 dx with l = 25 and 10 dx (m = 2, 5), g = 200 dx with
    # l = 20 and 10 dx (m = 5, 10); each with its control. The final state of every eccentric run holds the whole
    # fluid (fields); the other states the fluid within 3 h of a wall.
    for gap, tag, lengths in ((100, "", (5, 2)), (200, "_g200", (5, 10))):
        for clearance in (30, 60):
            eccentricity = clearance * gap // 100
            control = f"e{clearance}{tag}_newtonian"
            newtonian(control, gap, (3, 5), "", eccentricity, section=("eccentric",), full_final_state=True)
            for coupling_ratio in (3, 1):
                for gap_over_two_lengths in lengths:
                    micropolar(f"e{clearance}{tag}_{_ratio_tag(coupling_ratio)}_m{gap_over_two_lengths}", gap,
                               coupling_ratio, gap_over_two_lengths, control, "", eccentricity,
                               section=("eccentric",), full_final_state=True)
    # noise floor of the eccentric setting: the three lattices at eps = 0.3 (asked) and at eps = 0.6 (the one of
    # Stage 5 was without the density diffusion term), control and the micropolar run at K = 3, l = 10 dx
    for clearance in (30, 60):
        for lattice in ("_shifted", "_rotated"):
            newtonian(f"e{clearance}_newtonian{lattice}", 100, (3, 5), lattice, clearance, section=("eccentric_floor",))
            micropolar(f"e{clearance}_k3_m5{lattice}", 100, 3, 5, f"e{clearance}_newtonian{lattice}", lattice,
                       clearance, section=("eccentric_floor",))
    return runs


RUNS = _build_table()
# the functions of Stage 4 and 5 look a run up in the table of Stage 4 (control, candidate, last_quarter_only)
stage4.RUNS.update(RUNS)


def command_list(arguments) -> int:
    for name, entry in RUNS.items():
        print(f"{name:<28}{','.join(entry['section']):<22}control {entry['control'] or '-':<24}"
              f"{' '.join(entry['generate'])}")
    return 0


def ensure_case(name: str) -> dict:
    case_dir = _REPO_ROOT / CASES / name
    # the particle files are generated and not in the repository; those of finished runs may have been removed
    if not (case_dir / "couette.json").exists() or not (case_dir / "domain.obj").exists():
        code = subprocess.run([sys.executable, "-m", "experiment.validation.micropolar_couette", "generate",
                               "--out", f"{CASES}/{name}", *stage4.STEP_OPTIONS, *RUNS[name]["generate"]],
                              cwd=_REPO_ROOT).returncode
        if code:
            raise RuntimeError(f"generate {name}: exit {code}")
    return json.loads((case_dir / "couette.json").read_text(encoding="utf-8"))


def run_command(name: str, devices: list, steps: int | None = None) -> list:
    """The command line of one run of the table on the given device(s) (several = a chain)."""
    entry = RUNS[name]
    meta = ensure_case(name)
    total = steps if steps else meta["steps"]
    band = entry.get("state_band", 3.0 if entry["eccentricity"] else 2.0)
    command = [sys.executable, "-m", "experiment.validation.micropolar_couette", "run", "--case",
               f"{CASES}/{name}", "--out", f"{RUNS_DIRECTORY}/{name}", "--device-map",
               ",".join(str(device) for device in devices), "--sample-steps",
               str(stage4.sample_steps_for(total)), "--state-count", str(entry.get("state_count", STATE_COUNT)),
               "--state-band-supports", str(band)]
    if entry.get("full_final_state"):
        command += ["--full-final-state"]
    if steps:
        command += ["--steps", str(steps)]
    return command


# ---------------------------------------------------------------------------------------------------------------
# the queue runner
# ---------------------------------------------------------------------------------------------------------------
def _job_state(job: dict) -> str:
    done = _REPO_ROOT / job["done"]
    if done.exists():
        return "done"
    if done.with_name(done.name + ".failed").exists():
        return "failed"
    return "pending"


def _normalise(job: dict) -> dict:
    """A job of the plan: {"name", "devices": [..], "run": table name} or {"name", "devices", "command": [..],
    "done": path}. A run of the table is done when its run.json exists."""
    job = dict(job)
    if "run" in job:
        job.setdefault("name", job["run"])
        job["done"] = f"{RUNS_DIRECTORY}/{job['run']}/run.json"
        job["log"] = f"{RUNS_DIRECTORY}/{job['run']}.log"
    else:
        job.setdefault("log", str(pathlib.PurePosixPath(job["done"]).with_suffix(".queue.log")))
    return job


def command_runner(arguments) -> int:
    plan_path = _REPO_ROOT / arguments.plan
    stop_path, kill_path = plan_path.with_name("STOP"), plan_path.with_name("KILL")
    running = {}                                          # name -> (process, job, log handle, start time)

    def say(text: str) -> None:
        print(f"[runner {time.strftime('%H:%M:%S')}] {text}", flush=True)

    say(f"plan {arguments.plan}; stop with {stop_path.name}, terminate with {kill_path.name}")
    idle_reported = False
    disk_reported = False
    while True:
        if kill_path.exists():
            for name, (process, job, log, started) in running.items():
                process.terminate()
                say(f"{name}: terminated")
            return 1
        for name in list(running):
            process, job, log, started = running[name]
            code = process.poll()
            if code is None:
                continue
            log.close()
            del running[name]
            out_of_space = False
            if code:
                try:
                    tail = (_REPO_ROOT / job["log"]).read_text(encoding="utf-8", errors="replace")[-4000:]
                    out_of_space = "No space left on device" in tail
                except OSError:
                    pass
                if not out_of_space:
                    done = _REPO_ROOT / job["done"]
                    done.parent.mkdir(parents=True, exist_ok=True)
                    done.with_name(done.name + ".failed").write_text(f"exit {code}\n", encoding="utf-8")
            say(f"{name}: exit {code} after {time.time() - started:.0f} s"
                + (" (the disk was full: the job stays in the plan)" if out_of_space else ""))
        if stop_path.exists():
            if not running:
                say("stopped")
                return 0
            time.sleep(5.0)
            continue
        try:
            plan = [_normalise(job) for job in json.loads(plan_path.read_text(encoding="utf-8"))]
        except (OSError, ValueError) as error:              # the plan is being written
            say(f"plan unreadable ({error}); again in 5 s")
            time.sleep(5.0)
            continue
        busy = {device for process, job, log, started in running.values() for device in job["devices"]}
        # the disk of this machine is shared and nearly full: start nothing below the reserve
        free_gigabytes = shutil.disk_usage(_REPO_ROOT).free / 1024.0 ** 3
        if free_gigabytes < arguments.disk_reserve:
            if not disk_reported:
                say(f"{free_gigabytes:.1f} GB free on the disk, below the reserve of {arguments.disk_reserve:g} GB: "
                    "no job is started until there is room")
                disk_reported = True
            time.sleep(30.0)
            continue
        disk_reported = False
        pending = [job for job in plan if job["name"] not in running and _job_state(job) == "pending"]
        first = {}
        for job in pending:
            for device in job["devices"]:
                first.setdefault(device, job["name"])
        # a device whose first job waits for another device stays idle (a chain job is not overtaken)
        for job in pending:
            if all(first.get(device) == job["name"] and device not in busy for device in job["devices"]):
                try:
                    command = (run_command(job["run"], job["devices"]) if "run" in job
                               else [sys.executable if part == "{python}" else part for part in job["command"]])
                except Exception as error:                # e.g. the case could not be generated; try again later
                    say(f"{job['name']}: not started ({error}); again in 60 s")
                    time.sleep(60.0)
                    break
                log_path = _REPO_ROOT / job["log"]
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = open(log_path, "w", encoding="utf-8")
                process = subprocess.Popen(command, cwd=_REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
                running[job["name"]] = (process, job, log, time.time())
                busy.update(job["devices"])
                say(f"{job['name']}: started on device(s) {job['devices']}")
        if not running and not pending:
            if arguments.exit_when_empty:
                say("plan finished")
                return 0
            if not idle_reported:
                say("plan finished; waiting for more jobs")
                idle_reported = True
        else:
            idle_reported = False
        time.sleep(5.0)


def command_plan(arguments) -> int:
    """Append runs of the table to the plan (or write a new one with --new)."""
    plan_path = _REPO_ROOT / arguments.plan
    plan = [] if arguments.new or not plan_path.exists() else json.loads(plan_path.read_text(encoding="utf-8"))
    known = {job.get("name", job.get("run")) for job in plan}
    devices = [int(value) for value in arguments.devices.split(",")]
    for name in arguments.names:
        if name not in RUNS:
            sys.exit(f"{name}: not in the table")
        if name in known:
            print(f"[stage6] {name}: in the plan already")
            continue
        plan.append({"run": name, "devices": devices})
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = plan_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(plan, indent=1), encoding="utf-8")
    temporary.replace(plan_path)
    print(f"[stage6] plan: {len(plan)} jobs")
    return 0


def command_status(arguments) -> int:
    plan = [_normalise(job) for job in json.loads((_REPO_ROOT / arguments.plan).read_text(encoding="utf-8"))]
    for job in plan:
        state = _job_state(job)
        extra = ""
        if state == "done" and "run" in job:
            run = json.loads((_REPO_ROOT / job["done"]).read_text(encoding="utf-8"))
            extra = (f"  {run['steps']:,} steps, {run['steps_per_second']:.0f} steps/s, "
                     f"{run['elapsed_seconds'] / 60.0:.1f} min" + (f"  FAILED {run['failure']}" if run["failure"] else ""))
        print(f"{job['name']:<34}{str(job['devices']):<8}{state}{extra}")
    return 0


# ---------------------------------------------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------------------------------------------
REDUCED = "data/micropolar/stage6/couette"
DENSITY = stage4.DENSITY


def finished(names=None, eccentric: bool | None = None) -> list:
    runs_directory = _REPO_ROOT / RUNS_DIRECTORY
    return [name for name in (names or RUNS) if name in RUNS and (runs_directory / name / "run.json").exists()
            and (eccentric is None or bool(RUNS[name]["eccentricity"]) == eccentric)]


def levels(name: str, runs_directory: pathlib.Path | None = None) -> dict:
    """Mean density and mean pressure of the fluid over the eight eighths of the run (the samples of the run
    driver), and their change between the last two eighths: the drift the tables carry. Density relative to rho0,
    pressure in rho U^2."""
    from experiment.validation import micropolar_couette as couette
    loaded = couette._load_run(name, runs_directory or _REPO_ROOT / RUNS_DIRECTORY)
    data, run, meta = loaded["data"], loaded["run"], loaded["meta"]
    if "density_mean" not in data:
        return {}
    end_time = run["steps"] * run["timestep"]
    time_axis = data["step"] * run["timestep"]
    scale = DENSITY * meta["wall_speed"] ** 2
    density, pressure = [], []
    for index in range(8):
        window = (time_axis > (index / 8.0) * end_time + 1e-12) & (time_axis <= ((index + 1) / 8.0) * end_time + 1e-12)
        density.append(float(data["density_mean"][window].mean() / DENSITY - 1.0))
        pressure.append(float(data["pressure_mean"][window].mean() / scale))
    return {"density_level_windows": density, "pressure_level_windows": pressure,
            "density_level": density[7], "pressure_level": pressure[7],
            "density_drift": density[7] - density[6], "pressure_drift": pressure[7] - pressure[6]}


def command_concentric(arguments) -> int:
    """The analysis of Stage 4 (fits, D, exact solution, orders) on the concentric runs of this stage, then the rule
    in its slip form and the drift of the levels added to every record."""
    from experiment.validation import micropolar_stage5 as stage5
    stage4.RUNS.update(RUNS)                              # importing Stage 5 put its own entries in
    names = finished(arguments.names, eccentric=False)
    if not names:
        print("[stage6] no finished concentric run")
        return 1
    subprocess.run([sys.executable, "-m", "experiment.validation.micropolar_couette", "reduce", *names, "--runs",
                    RUNS_DIRECTORY, "--out", REDUCED], cwd=_REPO_ROOT, check=True)
    stage4.command_analyse(argparse.Namespace(names=names, runs=RUNS_DIRECTORY, fit_margin=1.0, json=arguments.json))
    target = _REPO_ROOT / arguments.json
    analysis = json.loads(target.read_text(encoding="utf-8"))
    runs = analysis["runs"]
    lines = ["", "rule in its slip form (hydrodynamic radii of the control) and the levels of the fluid: mean density "
             "relative to rho0 and mean pressure in rho U^2 over the last eighth of the run, and their change from "
             "the eighth before (drift)",
             f"{'run':<26}{'D_SPH':>10}{'D_exact':>10}{'D_slip':>10}{'slip/ex':>9}{'rho level':>12}{'rho drift':>12}"
             f"{'p level':>12}{'p drift':>12}  describe"]
    for name, record in runs.items():
        record.update(levels(name))
        text = f"{name:<26}"
        if "relative_torque_change" in record:
            control = runs[record["control"]]
            displacement = record["wall_displacement_in_spacings"] * 0.1 / record["gap_in_spacings"]
            record["slip_rule"] = stage5.slip_rule_torque_change(control["hydrodynamic_inner_radius"],
                                                                 control["hydrodynamic_outer_radius"], displacement)
            record["slip_rule_over_exact"] = record["slip_rule"] / record["exact_hydrodynamic"]
            text += (f"{record['relative_torque_change']:>10.5f}{record['exact_hydrodynamic']:>10.5f}"
                     f"{record['slip_rule']:>10.5f}{record['slip_rule_over_exact']:>9.4f}")
        else:
            text += f"{'-':>10}{'-':>10}{'-':>10}{'-':>9}"
        text += (f"{record['density_level']:>12.2e}{record['density_drift']:>12.2e}{record['pressure_level']:>12.2e}"
                 f"{record['pressure_drift']:>12.2e}  {record['revision']['describe']}"
                 + ("  SOLVER MODIFIED" if record["revision"]["solver_modified"] else ""))
        lines.append(text)
        print(text)
    target.write_text(json.dumps(analysis, indent=1), encoding="utf-8")
    with open(target.with_suffix(".txt"), "a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    return 0


READINGS = ("rotor_torque_wall", "stator_torque_wall", "rotor_torque_circle", "stator_torque_circle")
# the runs of Stages 4 and 5 without the density diffusion term, in the form of this stage (the provisional pack)
STAGE5_NEWTONIAN = ("nd_e30_newtonian", "nd_e60_newtonian", "nd_e60_newtonian_shifted", "nd_e60_newtonian_rotated",
                    "nd_e30_re1", "nd_e60_re1", "nd_e60_re1_ma001", "nd_e60_re1_g200", "nd_e60_re10_g200",
                    "nd_e30_re10_g200")
STAGE5_PAIRS = (("nd_e30_k3_m5", "nd_e30_newtonian"), ("nd_e30_k3_m2", "nd_e30_newtonian"),
                ("nd_e60_k3_m5", "nd_e60_newtonian"), ("nd_e60_k3_m2", "nd_e60_newtonian"),
                ("nd_e60_k3_m5_shifted", "nd_e60_newtonian_shifted"),
                ("nd_e60_k3_m5_rotated", "nd_e60_newtonian_rotated"),
                ("nd_e30_k3_m10_g200", "nd_e30_re10_g200"), ("nd_e60_k3_m10_g200", "nd_e60_re10_g200"))


def _spread(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(values.max() - values.min())


def command_eccentric(arguments) -> int:
    """Every finished eccentric run of the stage (or, with --set stage5, the runs of Stages 4 and 5 without the
    density diffusion term): the record of Stage 5 (load, direction, torque by both readings, wall pressure,
    levels), each micropolar run against its control, the predictions of the host (film function in the thin-film
    calculation; the rule on the exact Stokes solution), the noise floors from the three lattices, and the same
    flow at two resolutions."""
    from experiment.validation import micropolar_stage5 as stage5
    from experiment.validation import micropolar_film as film
    lines = []

    def say(text=""):
        lines.append(text)
        print(text, flush=True)
    own = _REPO_ROOT / RUNS_DIRECTORY
    concentric_fits = {}
    if arguments.set == "stage6":
        stage4.RUNS.update(RUNS)
        original = stage5.run_directory
        stage5.run_directory = lambda name: own if name in RUNS else original(name)
        stage5.CONCENTRIC_CONTROLS[("fd", 100)] = "g100_newtonian"
        stage5.CONCENTRIC_FALLBACKS[200] = "g200_newtonian"
        names = finished(arguments.names, eccentric=True)
        pairs = [(name, RUNS[name]["control"]) for name in names
                 if RUNS[name]["control"] and RUNS[name]["control"] in names]
        concentric_path = _REPO_ROOT / "data/micropolar/stage6/concentric.json"
        if concentric_path.exists():
            concentric_fits = json.loads(concentric_path.read_text(encoding="utf-8"))["runs"]
    else:
        names = [name for name in STAGE5_NEWTONIAN + tuple(pair[0] for pair in STAGE5_PAIRS)
                 if stage5._finished(name)]
        pairs = [pair for pair in STAGE5_PAIRS if pair[0] in names and pair[1] in names]
    records = {}
    for name in names:
        records[name] = stage5.eccentric_summary(name, arguments.recompute)
        directory = stage5.run_directory(name)
        records[name].update(levels(name, directory))
        records[name]["control"] = stage4.RUNS[name]["control"] if arguments.set == "stage6" else None
        print(f"[stage6] {name}: summarised", flush=True)

    def percent(value):
        return f"{100.0 * (value - 1.0):+.2f} %"
    say("Newtonian runs against the exact Stokes values (load in N/m, torque in N; torque columns: reading / exact - 1; "
        "levels: mean density over rho0 - 1 and mean pressure over rho U^2 in the last eighth, and their change from "
        "the eighth before)")
    say(f"{'run':<26}{'eps':>5}{'Re':>4}{'g/dx':>6}{'Ma':>6}{'load exact':>11}{'load':>10}{'':>9}{'angle':>8}"
        f"{'T exact':>9}{'T wall R':>10}{'T wall S':>10}{'T circ R':>10}{'T circ S':>10}{'dT %':>8}{'dF %':>8}"
        f"{'behind':>7}{'rho level':>11}{'drift':>10}{'p level':>9}{'drift':>10}  describe")
    for name in names:
        item = records[name]
        if item["micropolar"]:
            continue
        behind = item["behind_walls"]
        say(f"{name:<26}{item['eccentricity_ratio']:>5.2f}{item['reynolds']:>4g}{item['gap_in_spacings']:>6g}"
            f"{item['mach']:>6g}{item['exact']['load']:>11.2f}{item['load']:>10.2f}{percent(item['load_over_exact']):>9}"
            f"{item['force_angle_degrees']:>8.2f}{item['exact']['rotor_torque']:>9.3f}"
            f"{percent(item['rotor_torque_wall_over_exact']):>10}{percent(item['stator_torque_wall_over_exact']):>10}"
            f"{percent(item['rotor_torque_circle_over_exact']):>10}{percent(item['stator_torque_circle_over_exact']):>10}"
            f"{100.0 * item['torque_change_between_halves']:>+8.3f}{100.0 * item['load_change_between_halves']:>+8.3f}"
            f"{behind['inside_rotor'] + behind['beyond_stator']:>7,}"
            + (f"{item['density_level']:>+11.2e}{item['density_drift']:>+10.1e}{item['pressure_level']:>+9.3f}"
               f"{item['pressure_drift']:>+10.1e}" if "density_level" in item else f"{'-':>11}{'-':>10}{'-':>9}{'-':>10}")
            + f"  {item['revision']['describe']}"
            + ("" if item["counters_zero"] and not item["failure"] else "  FAILED / counters"))

    # what each torque reading makes of D between concentric cylinders at the same K, l and g, minus D of the fit
    def reading_errors(coupling_ratio, gap_over_two_lengths, gap):
        if arguments.set != "stage6":
            errors = stage5.concentric_reading_errors().get(f"micropolar_m{gap_over_two_lengths:g}")
            return {key: errors[key] for key in READINGS} if errors and gap == 100 and coupling_ratio == 3 else None
        name = f"g{gap:03d}_{_ratio_tag(coupling_ratio)}_m{gap_over_two_lengths:g}"
        control = f"g{gap:03d}_newtonian"
        if name not in concentric_fits or "relative_torque_change" not in concentric_fits[name]:
            return None
        torques = stage4.wall_torques(stage4.cached_wall_sums(name, own))
        base = stage4.wall_torques(stage4.cached_wall_sums(control, own))
        fit_change = concentric_fits[name]["relative_torque_change"]
        return {"rotor_torque_wall": torques["rotor"]["wall"] / base["rotor"]["wall"] - 1.0 - fit_change,
                "stator_torque_wall": torques["stator"]["wall"] / base["stator"]["wall"] - 1.0 - fit_change,
                "rotor_torque_circle": torques["rotor"]["circle"] / base["rotor"]["circle"] - 1.0 - fit_change,
                "stator_torque_circle": torques["stator"]["circle"] / base["stator"]["circle"] - 1.0 - fit_change}

    say()
    say("micropolar run over its Newtonian control: D_T by the four readings (measured), their mean with the error "
        "each reading makes between concentric cylinders at the same K, l, g taken off (corrected; '-' = no such "
        "concentric row), the change of the load and the turn of the force; predictions of the host: film = the "
        "film function of micropolar lubrication theory in the thin-film calculation (torque: 'conc.' with the "
        "shear-driven stress of the exact concentric solution at the local gap, 'slip' with the slip term of the "
        "rule on the rotor), rule = the wall rule on the exact Stokes solution of its slip-form geometry")
    say(f"{'run':<22}{'K':>4}{'eps':>5}{'g/dx':>6}{'l/dx':>6}{'hmin/l':>7}" + "".join(
        f"{label:>9}" for label in ("wall R", "wall S", "circ R", "circ S")) + f"{'mean':>9}{'corr.':>9}{'spread':>8}"
        f"{'film c.':>9}{'film s.':>9}{'film pl.':>9}{'rule':>9}{'load':>9}{'film':>9}{'rule':>9}{'turn':>8}"
        f"{'meas/film':>10}{'meas/rule':>10}")
    pair_records = {}
    for name, control_name in pairs:
        item, control = records[name], records[control_name]
        meta_gap = item["outer_radius"] - item["inner_radius"]
        coupling_ratio = stage4.RUNS[name].get("coupling_ratio", 3)
        length_in_spacings = item["length_in_spacings"]
        spacing = meta_gap / item["gap_in_spacings"]
        length = length_in_spacings * spacing
        gap_over_two_lengths = meta_gap / (2.0 * length)
        changes = {key: item[key] / control[key] - 1.0 for key in READINGS}
        errors = reading_errors(coupling_ratio, gap_over_two_lengths, round(item["gap_in_spacings"]))
        corrected = {key: changes[key] - errors[key] for key in READINGS} if errors else None
        mean_change = float(np.mean(list(changes.values())))
        corrected_mean = float(np.mean(list(corrected.values()))) if corrected else None
        best = corrected_mean if corrected_mean is not None else mean_change
        load_change = item["load"] / control["load"] - 1.0
        predicted = film.predictions(item["eccentricity_ratio"], meta_gap, item["inner_radius"], length,
                                     coupling_ratio)
        entry = {
            "control": control_name, "coupling_ratio": coupling_ratio, "eccentricity_ratio": item["eccentricity_ratio"],
            "gap_in_spacings": item["gap_in_spacings"], "length_in_spacings": length_in_spacings,
            "gap_over_two_lengths": gap_over_two_lengths,
            "narrowest_gap_over_length": (1.0 - item["eccentricity_ratio"]) * meta_gap / length,
            "lattice": stage4.RUNS[name].get("lattice", ""),
            "changes": changes, "reading_errors": errors, "corrected": corrected, "mean_change": mean_change,
            "corrected_mean": corrected_mean, "torque_change": best,
            "spread_of_readings": _spread(list((corrected or changes).values())),
            "load_change": load_change,
            "turn_degrees": item["force_angle_degrees"] - control["force_angle_degrees"],
            "predictions": predicted,
            "load_over_film": load_change / predicted["load_change_plane"],
            "load_over_rule": load_change / predicted["load_change_rule"],
            "torque_over_film": best / predicted["torque_change_concentric"],
            "torque_over_film_slip": best / predicted["torque_change_slip"],
            "torque_over_film_plane": best / predicted["torque_change_plane"],
            "torque_over_rule": best / predicted["torque_change_rule"]}
        pair_records[name] = entry
        say(f"{name:<22}{coupling_ratio:>4g}{item['eccentricity_ratio']:>5.2f}{item['gap_in_spacings']:>6g}"
            f"{length_in_spacings:>6g}{entry['narrowest_gap_over_length']:>7.1f}"
            + "".join(f"{changes[key]:>9.4f}" for key in READINGS) + f"{mean_change:>9.4f}"
            + (f"{corrected_mean:>9.4f}" if corrected else f"{'-':>9}") + f"{entry['spread_of_readings']:>8.4f}"
            f"{predicted['torque_change_concentric']:>9.4f}{predicted['torque_change_slip']:>9.4f}"
            f"{predicted['torque_change_plane']:>9.4f}{predicted['torque_change_rule']:>9.4f}"
            f"{100.0 * load_change:>+8.2f}%{100.0 * predicted['load_change_plane']:>+8.2f}%"
            f"{100.0 * predicted['load_change_rule']:>+8.2f}%{entry['turn_degrees']:>+8.3f}"
            f"{entry['load_over_film']:>10.3f}{entry['load_over_rule']:>10.3f}")

    # noise floors: the three lattices of one case
    floors = {}
    for label, members in (("e30", [f"e30_newtonian{lattice}" for lattice in LATTICES]),
                           ("e60", [f"e60_newtonian{lattice}" for lattice in LATTICES]),
                           ("nd_e60", [f"nd_e60_newtonian{lattice}" for lattice in LATTICES])):
        if not all(member in records for member in members):
            continue
        floor = {"runs": members,
                 "load_over_exact": [records[member]["load_over_exact"] for member in members],
                 "force_angle_degrees": [records[member]["force_angle_degrees"] for member in members]}
        for key in READINGS:
            floor[f"{key}_over_exact"] = [records[member][f"{key}_over_exact"] for member in members]
            floor[f"{key}_spread"] = _spread(floor[f"{key}_over_exact"])
        floor["load_spread"] = _spread(floor["load_over_exact"])
        floor["force_angle_spread_degrees"] = _spread(floor["force_angle_degrees"])
        micropolar = [member.replace("newtonian", "k3_m5") for member in members]
        if all(member in pair_records for member in micropolar):
            floor["pairs"] = micropolar
            floor["torque_change"] = [pair_records[member]["torque_change"] for member in micropolar]
            floor["load_change"] = [pair_records[member]["load_change"] for member in micropolar]
            floor["turn_degrees"] = [pair_records[member]["turn_degrees"] for member in micropolar]
            floor["torque_change_spread"] = _spread(floor["torque_change"])
            floor["load_change_spread"] = _spread(floor["load_change"])
            floor["turn_spread_degrees"] = _spread(floor["turn_degrees"])
            floor["readings_change_spread"] = {key: _spread([pair_records[member]["changes"][key]
                                                             for member in micropolar]) for key in READINGS}
        floors[label] = floor
    if floors:
        say()
        say("noise floors from the three lattices (max - min): Newtonian load and torque readings over exact, the "
            "direction; then the changes of the micropolar run (K = 3, l = 10 dx) against its control")
        for label, floor in floors.items():
            say(f"{label}: load {100.0 * floor['load_spread']:.2f} % (values "
                + ", ".join(f"{100.0 * (value - 1.0):+.2f}" for value in floor["load_over_exact"])
                + f" %), direction {floor['force_angle_spread_degrees']:.2f} degrees, torque wall R / S "
                f"{100.0 * floor['rotor_torque_wall_spread']:.2f} / {100.0 * floor['stator_torque_wall_spread']:.2f} %, "
                f"circle R / S {100.0 * floor['rotor_torque_circle_spread']:.2f} / "
                f"{100.0 * floor['stator_torque_circle_spread']:.2f} %"
                + (f"; changes: D_T {floor['torque_change_spread']:.4f} (values "
                   + ", ".join(f"{value:.4f}" for value in floor["torque_change"])
                   + f"), load change {100.0 * floor['load_change_spread']:.2f} points (values "
                   + ", ".join(f"{100.0 * value:+.2f}" for value in floor["load_change"])
                   + f" %), turn {floor['turn_spread_degrees']:.3f} degrees" if "pairs" in floor else ""))

    # the same flow at two resolutions: g = 100 dx with l = 10 dx and g = 200 dx with l = 20 dx (m = 5)
    resolution = []
    for clearance in (30, 60):
        for coupling_ratio in (3, 1):
            coarse = f"e{clearance}_{_ratio_tag(coupling_ratio)}_m5"
            fine = f"e{clearance}_g200_{_ratio_tag(coupling_ratio)}_m5"
            if coarse in pair_records and fine in pair_records:
                resolution.append({"coarse": coarse, "fine": fine, "coupling_ratio": coupling_ratio,
                                   "eccentricity_ratio": clearance / 100.0,
                                   **{f"{key}_{label}": pair_records[run][key] for key in
                                      ("torque_change", "load_change", "turn_degrees", "mean_change")
                                      for label, run in (("coarse", coarse), ("fine", fine))},
                                   "control_load_over_exact_coarse": records[pair_records[coarse]["control"]][
                                       "load_over_exact"],
                                   "control_load_over_exact_fine": records[pair_records[fine]["control"]][
                                       "load_over_exact"]})
    if resolution:
        say()
        say("the same flow at two resolutions (m = 5): g = 100 dx with l = 10 dx against g = 200 dx with l = 20 dx")
        say(f"{'K':>4}{'eps':>5}{'D_T 100':>10}{'D_T 200':>10}{'load 100':>10}{'load 200':>10}{'turn 100':>10}"
            f"{'turn 200':>10}{'control load / exact 100':>26}{'200':>9}")
        for entry in resolution:
            say(f"{entry['coupling_ratio']:>4g}{entry['eccentricity_ratio']:>5.2f}{entry['torque_change_coarse']:>10.4f}"
                f"{entry['torque_change_fine']:>10.4f}{100.0 * entry['load_change_coarse']:>+9.2f}%"
                f"{100.0 * entry['load_change_fine']:>+9.2f}%{entry['turn_degrees_coarse']:>+10.3f}"
                f"{entry['turn_degrees_fine']:>+10.3f}{percent(entry['control_load_over_exact_coarse']):>26}"
                f"{percent(entry['control_load_over_exact_fine']):>9}")
    if arguments.json:
        target = _REPO_ROOT / arguments.json
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"set": arguments.set, "runs": records, "pairs": pair_records, "floors": floors,
                                      "resolution": resolution}, indent=1), encoding="utf-8")
        target.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return 0


def velocity_gradient(position: np.ndarray, velocity: np.ndarray, radius: float) -> np.ndarray:
    """The gradient of a particle field of vectors by weighted linear least squares over the neighbours within
    radius (weight (1 - r / radius)^2): exact for a linear field on any particle arrangement, one-sided at a wall
    (fluid particles only). Any dimension d; returns (count, d, d) with [i, a, b] = d v_a / d x_b."""
    from scipy.spatial import cKDTree
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    count, dimension = position.shape
    pairs = cKDTree(position).query_pairs(radius, output_type="ndarray")
    first, second = pairs[:, 0], pairs[:, 1]
    separation = position[second] - position[first]
    difference = velocity[second] - velocity[first]
    weight = (1.0 - np.sqrt((separation ** 2).sum(axis=1)) / radius) ** 2
    moment = np.zeros((count, dimension, dimension))
    right = np.zeros((count, dimension, dimension))                 # [i, a, b] = sum of W dv_a r_b
    for row in range(dimension):
        for column in range(dimension):
            values = weight * separation[:, row] * separation[:, column]
            moment[:, row, column] = np.bincount(first, values, count) + np.bincount(second, values, count)
            values = weight * difference[:, row] * separation[:, column]
            right[:, row, column] = np.bincount(first, values, count) + np.bincount(second, values, count)
    # grad[i, a, :] = moment^-1 right[i, a, :]
    return np.linalg.solve(moment[:, None, :, :], right[..., None])[..., 0]


def command_fields(arguments) -> int:
    """The whole fluid of one eccentric (or concentric) run at its last saved step, with the curl of the velocity,
    as a file small enough for the repository: x, y, velocity, pressure, micro-rotation, curl (float32)."""
    from experiment.validation import micropolar_stage5 as stage5
    name = arguments.name
    directory = (_REPO_ROOT / RUNS_DIRECTORY) if name in RUNS and arguments.set == "stage6" \
        else stage5.run_directory(name)
    run = json.loads((directory / name / "run.json").read_text(encoding="utf-8"))
    meta = run["couette"]
    candidates = sorted((directory / name / "states").glob("full_*.npz")) \
        or sorted((directory / name / "states").glob("state_*.npz"))
    state = np.load(candidates[-1])
    fluid = state["kind"] == 0
    position = state["position"][fluid].astype(np.float64)
    if position.shape[0] != meta["fluid_count"]:
        sys.exit(f"{name}: the state {candidates[-1].name} holds {position.shape[0]:,} fluid particles of "
                 f"{meta['fluid_count']:,} (a wall band, not the whole fluid)")
    velocity = state["velocity"][fluid].astype(np.float64)
    gradient = velocity_gradient(position, velocity, arguments.radius_in_spacings * meta["spacing"])
    curl = gradient[:, 1, 0] - gradient[:, 0, 1]
    spin = state["spin"][fluid] if "spin" in state.files else np.zeros(position.shape[0], np.float32)
    target = _REPO_ROOT / arguments.out / f"{name}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, x=position[:, 0].astype(np.float32), y=position[:, 1].astype(np.float32),
                        u=velocity[:, 0].astype(np.float32), v=velocity[:, 1].astype(np.float32),
                        pressure=state["pressure"][fluid].astype(np.float32), spin=spin.astype(np.float32),
                        curl=curl.astype(np.float32), step=np.int64(state["step"]))
    summary = {"run": name, "state": candidates[-1].name, "step": int(state["step"]), "steps": run["steps"],
               "revision": run["revision"], "couette": meta,
               "gradient": f"weighted linear least squares over the fluid neighbours within "
                           f"{arguments.radius_in_spacings:g} dx"}
    target.with_suffix(".json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[stage6] fields of {name}: {position.shape[0]:,} particles, step {int(state['step']):,} -> {target}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    listing.set_defaults(handler=command_list)
    runner = commands.add_parser("runner")
    runner.add_argument("--plan", default=PLAN)
    runner.add_argument("--exit-when-empty", action="store_true")
    runner.add_argument("--disk-reserve", type=float, default=3.0, help="GB that must be free before a job starts")
    runner.set_defaults(handler=command_runner)
    plan = commands.add_parser("plan")
    plan.add_argument("names", nargs="*")
    plan.add_argument("--devices", required=True, help="device index, or several for a chain: 0,1")
    plan.add_argument("--plan", default=PLAN)
    plan.add_argument("--new", action="store_true")
    plan.set_defaults(handler=command_plan)
    status = commands.add_parser("status")
    status.add_argument("--plan", default=PLAN)
    status.set_defaults(handler=command_status)
    concentric = commands.add_parser("concentric")
    concentric.add_argument("names", nargs="*", help="default: every finished concentric run of the table")
    concentric.add_argument("--json", default="data/micropolar/stage6/concentric.json")
    concentric.set_defaults(handler=command_concentric)
    eccentric = commands.add_parser("eccentric")
    eccentric.add_argument("names", nargs="*", help="default: every finished eccentric run of the table")
    eccentric.add_argument("--set", default="stage6", choices=("stage6", "stage5"),
                           help="stage5: the runs of Stages 4 and 5 without the density diffusion term")
    eccentric.add_argument("--recompute", action="store_true", help="ignore the wall_sums.json beside each run")
    eccentric.add_argument("--json", default=None)
    eccentric.set_defaults(handler=command_eccentric)
    fields = commands.add_parser("fields")
    fields.add_argument("name")
    fields.add_argument("--set", default="stage6", choices=("stage6", "stage5"))
    fields.add_argument("--radius-in-spacings", type=float, default=3.0)
    fields.add_argument("--out", default="data/micropolar/stage6/fields")
    fields.set_defaults(handler=command_fields)
    arguments = parser.parse_args()
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
