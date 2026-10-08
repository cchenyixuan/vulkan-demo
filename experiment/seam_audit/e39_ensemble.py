"""
e39_ensemble.py - E39 accuracy part B, the ensemble test: is a v7 run (B1 fuses correction and density, so the
numerics change at round-off level) as close to a v6-rc2 run as two v6 runs are to each other? E33's ensemble
method (decomposition_audit) at K = 1, with the solver as the arm.

Runs: 2-D 1M (cases/lid_driven_cavity_2d_gen) and 3-D 1M (cases/cavity3d_1m) restarted from single_step's N = 2000
snapshots (dump_state --restart-snapshot; --snapshot-root, default logs/seam_audit/single_step of this checkout),
K = 1, depth 2, each solver's code defaults plus <PREFIX>DELTA_DENSITY=1 (exact rho in the dumps, as E14 / E32 /
E33; dump_state adds <PREFIX>TRANSPORT_EXTENSION=1), every particle dumped 300 and 2000 steps after the restart.
Arms: v6_k1 (dump_state --version v6: experiment/v6, v6-rc2) and v7_k1 (--version v7: experiment/v7), --trials runs
each (default 6). Trial t runs both arms on GPU (t - 1) % 2 (E33's K = 1 alternation), the arm order rotating by
one per trial (trial 1: v6, v7; trial 2: v7, v6; ...; with two arms the order within a trial goes with the GPU,
which only matters for speed, not for the dumps). Every run's environment is the caller's without any V5_* / V6_* /
V7_* variable, validation off. Before every run nvidia-smi must show no other compute process
(step_trace_campaign.wait_for_idle_gpus, logged); a run that stalls is retried once, an invalid run never. Each
record in OUT/results.jsonl carries the digest of its solver's code (utils/*.py + shaders/spv/*.spv) at run time:
on resume a run counts as done only if it exited 0, its dumps exist and its digest is the current one (a run of
older code is run again), and analyze refuses an arm whose runs ran different code (--allow-mixed-code).
--dry-run prints the plan, the commands, the environment, the GPU time and disk estimates and touches no GPU;
with --preflight it also runs dump_state --dry-run (case load, partition, id mask, snapshot split; CPU) per case
and solver.

analyze: per case and horizon the 2 T runs give C(2T, 2) pairs (T = 6: 66): within v6 (15), within v7 (15) and
across (36). Each pair's id-matched rms difference (every run holds every particle, dumps sorted by global id) over
the fluid particles, for velocity, acceleration, shift, density, pressure and kernel sum, in two bins: all = every
fluid particle, near_wall = the fluid particles with a wall or lid particle within one support radius h, in the
reference run's (the first v6 run's) positions at that horizon (E33's seam-distance bins do not exist at K = 1).
Per (case, horizon, field, bin): the class medians, across / within-v6 and within-v7 / within-v6 with their chance
ranges (2.5-97.5 % over the relabellings), the between-group term D^2 / s_1^2 with its p, and the exact permutation
tests of nowait_audit over the C(2T, T) relabellings (T = 6: 924): cross (across vs within pairs: a systematic
v7 - v6 offset) and one (pairs with a v7 run vs within-v6 pairs: any extra difference). Judgement per (case,
horizon) as in E33: the p < 0.05 count of the 12 tests (6 fields x 2 bins) of each statistic against the joint
permutation null (each relabelling applied to all 12 tests at once); 'indistinguishable' unless P(count >=
observed) < 0.05 for cross or one. Then the same joint test of the between term tells the two ways to differ
apart: 'systematic offset' when it fires too, else 'different scatter, no offset detected' (the cross statistic, a
difference of mean logs, also rises when the v7 runs only scatter more or less than the v6 runs; the between term
does not). The count barely moves when only one or two tests differ, so each statistic also gets its smallest p
of the 12 and that p's family-wise value over the same relabellings (Westfall-Young min-p); an indistinguishable
verdict with a family-wise p < 0.05 reads "indistinguishable; one test stands out" and names the test. Also: the
share of bit-identical particles per pair class (velocity, density), the GPU split of each
arm's own pairs (cross test over the card relabellings, 20 for 3 + 3 runs, smallest p 0.1), and the resolving
power: the velocity / all test again with a synthetic defect of rms f times the within-v6 median added to every v7
run (systematic: the same in every run; random: independent), f = 0.5, 1, 2, and the joint tests recomputed with
it. Every run's invariants (drift, missing /
duplicate ids, every overflow counter, GPU / host frame stamps, far migrations) must be 0 and its last record in
results.jsonl must have exit code 0, and its dump must come
from its arm's solver with exactly <PREFIX>DELTA_DENSITY=1 and <PREFIX>TRANSPORT_EXTENSION=1 set
(--allow-foreign-dumps skips the provenance part, e.g. to re-analyse E33 dumps). Writes OUT/ensemble.json and
OUT/ensemble.md; --docs DIR also writes e39_ensemble_tables.md and e39_ensemble_summary.json (no pair lists).

selftest (CPU only, no solver, no GPU): fake dumps in the dump_state layout (2-D lattice, 12,321 fluid and 1,416
wall / lid particles, T runs per arm, run-to-run noise in every field) for four synthetic cases - null (v7 drawn
like v6), offset (the same offset field in every v7 run, rms 0.5 x the within-pair rms), wall_offset (an offset
only on the fluid near the walls) and spread (v7 runs 30 % noisier) - analyzed by the same code. Passes when the
null is judged indistinguishable at both horizons (and its synthetic systematic defects f >= 0.5 are detected by
their test and by the family-wise min-p), offset and wall_offset are judged systematic offsets (wall_offset in the
near-wall bin above all), spread is judged a different scatter without offset (the 'one' statistic, within-v7 /
within-v6 = 1.3), and the ratios match their constructed values. --null-replicates R adds R more null cases and
reports how often each statistic's joint test fires (expected <= 5 %).

Usage (from the checkout root; GPU for run, CPU for the rest):
  .venv/Scripts/python.exe -m experiment.seam_audit.e39_ensemble run --out logs/e39/ensemble \\
      --snapshot-root C:/Users/cchen/PycharmProjects/vulkan-demo/logs/seam_audit/single_step [--dry-run [--preflight]]
  .venv/Scripts/python.exe -m experiment.seam_audit.e39_ensemble analyze --out logs/e39/ensemble [--docs DIR]
  .venv/Scripts/python.exe -m experiment.seam_audit.e39_ensemble selftest [--out logs/e39/ensemble_selftest]
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import pathlib
import re
import subprocess
import sys
import time

import numpy as np

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from experiment.seam_audit.decomposition_audit import (  # noqa: E402
    class_medians,
    joint_null,
    json_ready,
    pair_rms,
    relabelled_p,
    relabelling_statistics,
    scenario_name,
    sensitivity,
    summarize,
)
from experiment.seam_audit.dump_state import (  # noqa: E402
    dump_file_paths,
    save_json_atomically,
    save_npz_atomically,
)
from experiment.seam_audit.nowait_audit import (  # noqa: E402
    RUN_TIMEOUT_SECONDS,
    STALL_MARKERS,
    log_line,
    permutation_tests,
    result_line,
)
from experiment.seam_audit.solver_adapter import environment_prefix  # noqa: E402

CASES = {
    "cavity2d_1m": {"case": "cases/lid_driven_cavity_2d_gen/case.yaml", "snapshot": "cavity2d_1m/snapshot_N2000.npz",
                    "label": "2-D 1M"},
    "cavity3d_1m": {"case": "cases/cavity3d_1m/case.yaml", "snapshot": "cavity3d_1m/snapshot_N2000.npz",
                    "label": "3-D 1M"},
}
DEFAULT_SNAPSHOT_ROOT = "logs/seam_audit/single_step"
ARMS = {"v6_k1": "v6", "v7_k1": "v7"}          # arm -> dump_state --version; the reference arm first
REFERENCE_ARM, TEST_ARM = "v6_k1", "v7_k1"
TRIALS = 6
HORIZONS = (300, 2000)
FIELDS = ("velocity", "acceleration", "shift", "density", "pressure", "kernel_sum")
BINS = ("all", "near_wall")
BIN_LABELS = {"all": "全部流体", "near_wall": "近壁"}
IDENTICAL_FIELDS = ("velocity", "density")      # share of particles bit-identical in both runs of a pair
CLASS_NAMES = {"within_k1": "within_v6", "within_x": "within_v7", "cross": "across"}
CLASS_ORDER = ("within_v6", "within_v7", "across")
SIGNIFICANCE = 0.05
JOINT_STATISTICS = ("cross", "one", "between")
DEFECT_TEST = ("velocity", "all")               # the test the synthetic defects of the resolving power replace
# GPU time estimate of --dry-run: E33's K = 1 restarts of the same snapshots (v6, 300 + 2000 steps, process wall
# with load and dumps, median of 6); wait_for_idle_gpus settles 5 s before every run
MEASURED_RUN_SECONDS = {"cavity2d_1m": 8.1, "cavity3d_1m": 35.2}
IDLE_SETTLE_SECONDS = 5
LOG_PREFIX = "[e39_ensemble]"


# ----------------------------------------------------------------------------- plan and runs

def gpu_of_trial(trial: int) -> int:
    return (trial - 1) % 2


def run_name(arm: str, trial: int) -> str:
    return f"{arm}_g{gpu_of_trial(trial)}_t{trial}"


def run_order(trials: int) -> list:
    """(trial, arm) in execution order: every trial runs every arm, the arm order rotating by one per trial."""
    arms = list(ARMS)
    order = []
    for trial in range(1, trials + 1):
        shift = (trial - 1) % len(arms)
        order += [(trial, arm) for arm in arms[shift:] + arms[:shift]]
    return order


def run_environment(solver: str) -> dict:
    """The caller's environment without any V5_ / V6_ / V7_ switch (the solver's code defaults), validation off,
    plus <PREFIX>DELTA_DENSITY=1 (dump_state adds <PREFIX>TRANSPORT_EXTENSION=1 itself)."""
    environment = {key: value for key, value in os.environ.items() if not key.startswith(("V5_", "V6_", "V7_"))}
    environment.update({"VK_LOADER_LAYERS_DISABLE": "VK_LAYER_KHRONOS_validation", "PYTHONIOENCODING": "utf-8",
                        "PYTHONUNBUFFERED": "1", environment_prefix(solver) + "DELTA_DENSITY": "1"})
    return environment


def expected_switches(solver: str) -> dict:
    """The V5_ / V6_ / V7_ variables a dump of this driver must record: exactly these two."""
    prefix = environment_prefix(solver)
    return {prefix + "DELTA_DENSITY": "1", prefix + "TRANSPORT_EXTENSION": "1"}


def solver_code_digest(solver: str) -> str:
    """sha256 (16 hex) over experiment/<solver>/utils/*.py and shaders/spv/*.spv, relative path and bytes in sorted
    order: the code a run of that solver executes."""
    directory = _REPOSITORY_ROOT / "experiment" / solver
    digest = hashlib.sha256()
    for path in sorted(list((directory / "utils").glob("*.py")) + list((directory / "shaders" / "spv").glob("*.spv"))):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def git_state() -> dict:
    def git(*arguments) -> str:
        return subprocess.run(["git", *arguments], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", cwd=_REPOSITORY_ROOT).stdout.strip()
    return {"head": git("rev-parse", "--short", "HEAD"),
            "dirty": {solver: bool(git("status", "--porcelain", f"experiment/{solver}"))
                      for solver in sorted(set(ARMS.values()))}}


def snapshot_path(snapshot_root: pathlib.Path, case_name: str) -> pathlib.Path:
    return snapshot_root / CASES[case_name]["snapshot"]


def dump_state_command(case_name: str, arm: str, trial: int, dump_directory: pathlib.Path,
                       snapshot_file: pathlib.Path, dry_run: bool = False) -> list:
    command = [sys.executable, "-m", "experiment.seam_audit.dump_state", "--version", ARMS[arm],
               "--case", CASES[case_name]["case"], "--slabs", "1", "--device-map", str(gpu_of_trial(trial)),
               "--restart-snapshot", str(snapshot_file), "--horizons", ",".join(str(horizon) for horizon in HORIZONS),
               "--depth", "2", "--out-dir", str(dump_directory), "--run-name", run_name(arm, trial)]
    return command + (["--dry-run"] if dry_run else [])


def last_records_by_run(results_path: pathlib.Path) -> dict:
    records = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[(record["case"], record["run"])] = record
    return records


def run_status(record, dumps_exist: bool, digest: str) -> str:
    if record and record["rc"] == 0 and dumps_exist:
        return "done" if record.get("code_digest") == digest else "stale"
    return "pending"


def run_one(case_name: str, arm: str, trial: int, out_directory: pathlib.Path, snapshot_file: pathlib.Path,
            campaign_log: pathlib.Path, digest: str) -> dict:
    from experiment.v6.analysis.step_trace_campaign import wait_for_idle_gpus
    name = run_name(arm, trial)
    solver = ARMS[arm]
    dump_directory = out_directory / "dumps" / case_name
    dump_directory.mkdir(parents=True, exist_ok=True)
    command = dump_state_command(case_name, arm, trial, dump_directory, snapshot_file)
    attempts = []
    summary = {}
    for attempt in (1, 2):
        gpu = wait_for_idle_gpus(lambda message: log_line(campaign_log, message))
        log_line(campaign_log, f"{case_name} {name} ({solver}, K = 1 on GPU {gpu_of_trial(trial)}) attempt {attempt}: "
                               f"nvidia-smi idle after {gpu['waited_s']} s, gpus {gpu['gpus']}")
        log_path = out_directory / "logs" / f"{case_name}__{name}__a{attempt}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with open(log_path, "w", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=_REPOSITORY_ROOT, env=run_environment(solver), stdout=stream,
                                       stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=RUN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = -9
        summary = result_line(log_path.read_text(encoding="utf-8", errors="replace"))
        error = summary.get("error") or ""
        attempts.append({"attempt": attempt, "rc": code, "seconds": round(time.time() - started, 1),
                         "error": error, "log": str(log_path), "gpu_check": gpu})
        log_line(campaign_log, f"  -> exit {code} in {attempts[-1]['seconds']} s" + (f", error {error}" if error else ""))
        if not (code == 1 and any(marker in error for marker in STALL_MARKERS)):
            break
    record = {"case": case_name, "arm": arm, "solver": solver, "trial": trial, "gpu": gpu_of_trial(trial),
              "run": name, "rc": attempts[-1]["rc"], "attempts": attempts, "result": summary,
              "code_digest": digest, "git": git_state(), "snapshot": str(snapshot_file),
              "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(out_directory / "results.jsonl", "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    return record


def snapshot_particle_count(snapshot_file: pathlib.Path):
    try:
        with np.load(snapshot_file) as archive:
            return int(archive["id"].size)
    except (OSError, KeyError, ValueError):
        return None


def print_plan(plan: list, out_directory: pathlib.Path, snapshot_root: pathlib.Path, digests: dict) -> None:
    """The --dry-run report: every planned run with its status, the environment, one command per solver, the GPU
    time and disk estimates."""
    counts = {status: sum(1 for item in plan if item["status"] == status) for status in ("pending", "stale", "done")}
    cases = sorted({item["case"] for item in plan}, key=list(CASES).index)
    trials = max((item["trial"] for item in plan), default=0)
    print(f"{LOG_PREFIX} plan: {len(plan)} runs ({len(cases)} case(s) x {len(ARMS)} arms x {trials} trials) -> "
          f"{counts['pending']} pending, {counts['stale']} to run again (code changed since), {counts['done']} done; "
          f"out {out_directory}", flush=True)
    state = git_state()
    print(f"{LOG_PREFIX} code: " + ", ".join(f"{solver} digest {digest}" + (" (uncommitted changes)"
                                                                          if state["dirty"].get(solver) else "")
                                            for solver, digest in digests.items()) + f"; HEAD {state['head']}")
    print(f"  {'#':>3}  {'case':<12} {'trial':>5}  {'arm':<6} {'GPU':>3}  {'run':<14} status")
    for index, item in enumerate(plan, 1):
        print(f"  {index:>3}  {item['case']:<12} {item['trial']:>5}  {item['arm']:<6} {gpu_of_trial(item['trial']):>3}  "
              f"{item['run']:<14} {item['status']}")
    for arm, solver in ARMS.items():
        pins = " ".join(f"{key}={value}" for key, value in sorted(run_environment(solver).items())
                        if key.startswith(("V5_", "V6_", "V7_", "VK_LOADER")))
        print(f"{LOG_PREFIX} {arm}: environment = the caller's without V5_/V6_/V7_ + {pins}; dump_state adds "
              f"{environment_prefix(solver)}TRANSPORT_EXTENSION=1")
        example = next((item for item in plan if item["arm"] == arm), None)
        if example:
            command = dump_state_command(example["case"], arm, example["trial"],
                                         out_directory / "dumps" / example["case"],
                                         snapshot_path(snapshot_root, example["case"]))
            print(f"{LOG_PREFIX}   e.g. {' '.join(command)}")
    to_run = [item for item in plan if item["status"] != "done"]
    gpu_seconds = sum(MEASURED_RUN_SECONDS[item["case"]] for item in to_run)
    wall_seconds = gpu_seconds + IDLE_SETTLE_SECONDS * len(to_run)
    disk_bytes = 0
    for case_name in cases:
        particles = snapshot_particle_count(snapshot_path(snapshot_root, case_name))
        dimension = 3 if case_name.startswith("cavity3d") else 2
        if particles:
            # id 4 + slab 1 + position / velocity / acceleration / shift 4 x 4 x dimension + density 8 + pressure 4
            # + kernel_sum 4 + material 2 + previous_slab 1 + crossed 1 bytes per particle and dump
            disk_bytes += (sum(1 for item in to_run if item["case"] == case_name) * len(HORIZONS)
                           * particles * (25 + 16 * dimension))
    print(f"{LOG_PREFIX} estimate for the {len(to_run)} run(s) to do: GPU ~{gpu_seconds / 60:.1f} min (per run "
          + ", ".join(f"{case_name} {MEASURED_RUN_SECONDS[case_name]} s" for case_name in cases)
          + f", E33's v6 K = 1 process wall), wall ~{wall_seconds / 60:.1f} min with the {IDLE_SETTLE_SECONDS} s idle "
            f"settle per run when no other job holds a GPU; dumps ~{disk_bytes / 1e9:.1f} GB", flush=True)


def preflight(cases: list, snapshot_root: pathlib.Path, out_directory: pathlib.Path) -> int:
    """dump_state --dry-run for every case and solver (CPU: case load, partition, id mask, restart split), in the
    arm's environment; no Vulkan module is imported."""
    failures = 0
    for case_name in cases:
        for arm, solver in ARMS.items():
            command = dump_state_command(case_name, arm, 1, out_directory / "dumps" / case_name,
                                         snapshot_path(snapshot_root, case_name), dry_run=True)
            started = time.time()
            completed = subprocess.run(command, cwd=_REPOSITORY_ROOT, env=run_environment(solver), capture_output=True,
                                       text=True, encoding="utf-8", errors="replace")
            summary = result_line(completed.stdout)
            lines = [line for line in completed.stdout.splitlines() if "restart from" in line or "slab 0:" in line]
            passed = completed.returncode == 0 and summary.get("valid")
            failures += not passed
            print(f"{LOG_PREFIX} preflight {case_name} {solver}: rc {completed.returncode} in "
                  f"{time.time() - started:.0f} s -> {'OK' if passed else 'FAIL'}", flush=True)
            for line in lines:
                print(f"    {line.strip()}")
            if not passed:
                print(completed.stdout[-2000:] + completed.stderr[-2000:])
    return failures


def run_campaign(out_directory: pathlib.Path, snapshot_root: pathlib.Path, case_names: list, trials: int,
                 dry_run: bool, run_preflight: bool) -> int:
    cases = [case_name for case_name in CASES if not case_names or case_name in case_names]
    missing = [str(path) for case_name in cases
               for path in (snapshot_path(snapshot_root, case_name), _REPOSITORY_ROOT / CASES[case_name]["case"])
               if not path.exists()]
    if missing:
        print(f"{LOG_PREFIX} missing input(s): {missing} (snapshots: --snapshot-root; the cases' .obj files must be "
              f"in this checkout)", flush=True)
        return 2
    digests = {solver: solver_code_digest(solver) for solver in dict.fromkeys(ARMS.values())}
    records = last_records_by_run(out_directory / "results.jsonl")
    plan = []
    for case_name in cases:
        for trial, arm in run_order(trials):
            name = run_name(arm, trial)
            dumps_exist = all(dump_file_paths(out_directory / "dumps" / case_name, name, horizon)["npz"].exists()
                              for horizon in HORIZONS)
            plan.append({"case": case_name, "arm": arm, "trial": trial, "run": name,
                         "status": run_status(records.get((case_name, name)), dumps_exist, digests[ARMS[arm]])})
    if dry_run:
        print_plan(plan, out_directory, snapshot_root, digests)
        return 1 if run_preflight and preflight(cases, snapshot_root, out_directory) else 0
    out_directory.mkdir(parents=True, exist_ok=True)
    campaign_log = out_directory / "campaign.log"
    log_line(campaign_log, f"campaign start: {len(plan)} runs, code " + ", ".join(f"{solver} {digest}"
                                                                                for solver, digest in digests.items())
             + f", snapshots {snapshot_root}")
    failures = 0
    for item in plan:
        if item["status"] == "done":
            log_line(campaign_log, f"{item['case']} {item['run']}: done, skipped")
            continue
        if item["status"] == "stale":
            log_line(campaign_log, f"{item['case']} {item['run']}: dumps of older {ARMS[item['arm']]} code, run again")
        # the digest is taken again right before the run: the record names the code this run executed
        digest = solver_code_digest(ARMS[item["arm"]])
        record = run_one(item["case"], item["arm"], item["trial"], out_directory,
                         snapshot_path(snapshot_root, item["case"]), campaign_log, digest)
        failures += record["rc"] != 0
    log_line(campaign_log, f"campaign finished, {failures} failed run(s)")
    return 1 if failures else 0


# ----------------------------------------------------------------------------- analysis

def discover_runs(dump_directory: pathlib.Path, arm: str, horizon: int) -> list:
    """The runs of an arm with a dump at this horizon, ordered by trial."""
    pattern = re.compile(rf"^({re.escape(arm)}_g\d+_t(\d+))_N{int(horizon)}\.npz$")
    found = []
    for path in dump_directory.glob(f"{arm}_g*_t*_N{int(horizon)}.npz"):
        match = pattern.match(path.name)
        if match:
            found.append((int(match.group(2)), match.group(1)))
    return [name for _, name in sorted(found)]


def class_of(first: str, second: str, test_runs: set) -> str:
    first_test, second_test = first in test_runs, second in test_runs
    if first_test and second_test:
        return "within_v7"
    if first_test or second_test:
        return "across"
    return "within_v6"


def card_of(name: str) -> str:
    match = re.search(r"_g(\d+)_t\d+$", name)
    return f"g{match.group(1)}" if match else "?"


def renamed(summary: dict) -> dict:
    """decomposition_audit.summarize's K = 1 / X vocabulary in the solver's: within_v6, within_v7, across."""
    out = {}
    for key, value in summary.items():
        if key in ("median", "range"):
            out[key] = {CLASS_NAMES[label]: item for label, item in value.items()}
        elif key == "cross_over_within_k1":
            out["across_over_within_v6"] = value
        elif key == "within_x_over_within_k1":
            out["within_v7_over_within_v6"] = value
        else:
            out[key] = value
    return out


def near_wall_mask(positions: np.ndarray, fluid: np.ndarray, support_radius: float) -> np.ndarray:
    """Per fluid particle: is a wall or lid particle (any non-fluid one) within one support radius?"""
    from scipy.spatial import cKDTree
    walls = positions[~fluid].astype(np.float64)
    if walls.shape[0] == 0:
        return np.zeros(int(fluid.sum()), dtype=bool)
    distance, _ = cKDTree(walls).query(positions[fluid].astype(np.float64), k=1,
                                       distance_upper_bound=support_radius)
    return np.isfinite(distance)


def identical_share(values: dict, first: str, second: str, mask: np.ndarray):
    """Share of the bin's particles whose value (every component) is bit-identical in the two runs."""
    equal = values[first] == values[second]
    if equal.ndim == 2:
        equal = equal.all(axis=1)
    return float(np.mean(equal[mask])) if mask.any() else None


def card_split(runs: list, distances: dict) -> dict:
    """An arm's own pairs by GPU (g0xg0, g1xg1, g0xg1): medians, across-card / same-card, and the cross test of the
    GPU 1 runs against the GPU 0 runs over the card relabellings (3 + 3 runs: 20, smallest p 0.1)."""
    groups = {}
    for (first, second), value in distances.items():
        groups.setdefault("x".join(sorted((card_of(first), card_of(second)))), []).append(value)
    median = {key: float(np.median(items)) for key, items in groups.items()}
    same = groups.get("g0xg0", []) + groups.get("g1xg1", [])
    same_median = float(np.median(same)) if same else 0.0
    out = {"median": median, "pairs": {key: len(items) for key, items in groups.items()},
           "across_over_same": median["g0xg1"] / same_median if "g0xg1" in median and same_median > 0 else None}
    second_card = [name for name in runs if card_of(name) == "g1"]
    if second_card and len(second_card) < len(runs):
        tests = permutation_tests(runs, distances, test_runs=second_card)
        out.update({"cross_p": tests["cross_p"], "relabellings": tests["relabellings"]})
    return out


def run_checks(dump_directory: pathlib.Path, run_names: list, horizon: int, arm_of: dict,
               records: dict, case_name: str) -> tuple[dict, dict, list, list]:
    """(invariants, provenance, invalid runs, provenance problems) of every run at this horizon. A run whose last
    record in results.jsonl did not exit 0 is invalid too: a failed repeat can leave one horizon's dump of the new
    run next to the other of the old one."""
    invariants, provenance, invalid, foreign = {}, {}, [], []
    for name in run_names:
        sidecar = json.loads(dump_file_paths(dump_directory, name, horizon)["json"].read_text(encoding="utf-8"))
        item = sidecar.get("invariants", {})
        invariants[name] = {key: value for key, value in item.items() if key != "warnings"}
        if not item.get("valid") or item.get("far_migration_count", 0):
            invalid.append(f"{name}: {invariants[name]}")
        record = records.get((case_name, name), {})
        if record and record.get("rc") != 0:
            invalid.append(f"{name}: its last run exited {record.get('rc')} ({record.get('finished')})")
        solver = ARMS[arm_of[name]]
        switches = {key: value for key, value in (sidecar.get("environment") or {}).items()
                    if key.startswith(("V5_", "V6_", "V7_"))}
        provenance[name] = {"version": sidecar.get("version"), "switches": switches,
                            "density_exact_float64": (sidecar.get("bookkeeping") or {}).get("density_exact_float64"),
                            "slabs": sidecar.get("slabs"), "device_map": sidecar.get("device_map"),
                            "code_digest": record.get("code_digest"), "git": record.get("git")}
        if sidecar.get("version") != solver:
            foreign.append(f"{name}: written by {sidecar.get('version')}, arm {arm_of[name]} is {solver}")
        if switches != expected_switches(solver):
            foreign.append(f"{name}: switches {switches}, expected {expected_switches(solver)}")
        if provenance[name]["density_exact_float64"] is not True:
            foreign.append(f"{name}: density not exact float64 (no {environment_prefix(solver)}DELTA_DENSITY)")
        if sidecar.get("slabs") not in (None, 1):
            foreign.append(f"{name}: K = {sidecar.get('slabs')}, not 1")
    return invariants, provenance, invalid, foreign


def code_problems(provenance: dict, arm_of: dict) -> tuple[dict, list]:
    """Per arm, the code digests its runs recorded (results.jsonl); more than one = the arm mixes code versions."""
    digests = {}
    for name, item in provenance.items():
        digests.setdefault(arm_of[name], {}).setdefault(item.get("code_digest") or "unknown", []).append(name)
    problems = [f"{arm}: runs of {len(by_digest)} code versions {by_digest}" for arm, by_digest in digests.items()
                if len(by_digest) > 1]
    return digests, problems


def joint_counts(statistics: dict) -> dict:
    """decomposition_audit.joint_counts with the between-group term as a third statistic: per statistic the p < 0.05
    count over the tests at every relabelling, P(count >= observed), P(count = 0), the null mean; p_none_both over
    cross and one, as in E33. Also per statistic the smallest p of the tests and its family-wise p over the same
    relabellings (Westfall-Young min-p: the share of relabellings whose smallest p is not larger): the count barely
    moves when one test alone differs (one more significant test), the min-p does."""
    out = {}
    keys = list(statistics)
    observed = statistics[keys[0]]["observed"]
    total = 0
    for kind in JOINT_STATISTICS:
        p_values = np.array([relabelled_p(statistics[key][kind]) for key in keys])     # tests x relabellings
        counts = np.sum(p_values < SIGNIFICANCE, axis=0)
        if kind != "between":
            total = total + counts
        smallest = p_values.min(axis=0)
        strongest = int(np.argmin(p_values[:, observed]))
        out[kind] = {"count": int(counts[observed]), "tests": len(statistics),
                     "p_at_least": float(np.mean(counts >= counts[observed])), "p_none": float(np.mean(counts == 0)),
                     "null_mean_count": float(np.mean(counts)),
                     "min_p": float(smallest[observed]), "min_p_test": "/".join(map(str, keys[strongest])),
                     "min_p_familywise": float(np.mean(smallest <= smallest[observed] + 1e-12))}
    out["p_none_both"] = float(np.mean(total == 0))
    return out


def joint_tests(run_names: list, test_runs: list, tests: dict, scenarios: dict) -> dict:
    """The joint permutation null of one (case, horizon): every relabelling applied to all tests at once (keeps
    their dependence), for cross, one and between; scenarios recompute it with some tests replaced (the synthetic
    defects of the resolving power). The cross / one part must equal decomposition_audit.joint_null (E33)."""
    statistics = {key: relabelling_statistics(run_names, test_runs, distances) for key, distances in tests.items()}
    out = joint_counts(statistics)
    reference = joint_null(run_names, test_runs, tests)
    for kind in ("cross", "one"):
        if any(out[kind][key] != value for key, value in reference[kind].items()):
            raise ValueError(f"joint counts disagree with decomposition_audit.joint_null ({kind})")
    out["scenarios"] = {}
    for name, replacement in scenarios.items():
        changed = dict(statistics)
        for key, distances in replacement.items():
            changed[key] = relabelling_statistics(run_names, test_runs, distances)
        out["scenarios"][name] = joint_counts(changed)
    return out


def judgement(entry: dict) -> dict:
    """E33's configuration-level verdict on the 12 tests of one (case, horizon): indistinguishable unless the joint
    cross or one test fires (P(count >= observed) < 0.05). When one fires, the joint between test (D^2 / s_1^2, which
    a wider or tighter v7 arm alone does not move) separates a systematic offset from a different run-to-run
    scatter: the cross statistic alone also rises when only the scatter differs. An indistinguishable verdict with a
    family-wise min-p < 0.05 for cross or one gets "; one test stands out" (named in 'localized')."""
    joint = entry["joint"]
    items = {(field, label): entry["fields"][field][label] for field in FIELDS for label in BINS
             if label in entry["fields"].get(field, {})}
    ratios = [item["across_over_within_v6"] for item in items.values() if item.get("across_over_within_v6")]
    inside = [key for key, item in items.items() if item.get("across_over_within_v6")
              and item["ratio_null_95"][0] <= item["across_over_within_v6"] <= item["ratio_null_95"][1]]
    fired = {kind: joint[kind]["count"] > 0 and joint[kind]["p_at_least"] < SIGNIFICANCE for kind in JOINT_STATISTICS}
    # a difference confined to one or two tests moves the count by one or two: the family-wise min-p shows it
    localized = {kind: f"{joint[kind]['min_p_test']} (p {joint[kind]['min_p']:.3f}, family-wise "
                       f"{joint[kind]['min_p_familywise']:.3f})"
                 for kind in ("cross", "one") if joint[kind]["min_p_familywise"] < SIGNIFICANCE}
    if not (fired["cross"] or fired["one"]):
        verdict = "indistinguishable" + ("; one test stands out" if localized else "")
    elif fired["between"]:
        verdict = "systematic offset"
    else:
        verdict = "different scatter, no offset detected"
    within_ratios = [item["within_v7_over_within_v6"] for item in items.values() if item.get("within_v7_over_within_v6")]
    return {"verdict": verdict, "cross_detected": fired["cross"], "one_detected": fired["one"],
            "between_detected": fired["between"], "localized": localized, "tests": len(items),
            "ratio_range": [min(ratios), max(ratios)] if ratios else None,
            "ratios_inside_chance_range": len(inside),
            "within_ratio_range": [min(within_ratios), max(within_ratios)] if within_ratios else None,
            "between_range": [min(item["between"] for item in items.values()),
                              max(item["between"] for item in items.values())] if items else None,
            "significant": {kind: [f"{field}/{label}" for (field, label), item in items.items()
                                   if item[f"{kind}_p"] < SIGNIFICANCE] for kind in JOINT_STATISTICS}}


def analyze_case(case_name: str, out_directory: pathlib.Path, records: dict = None, allow_mixed_code: bool = False,
                 allow_foreign_dumps: bool = False, seed_base: int = 39000) -> dict:
    records = records or {}
    dump_directory = out_directory / "dumps" / case_name
    result = {}
    for horizon_index, horizon in enumerate(HORIZONS):
        reference_runs = discover_runs(dump_directory, REFERENCE_ARM, horizon)
        test_runs = discover_runs(dump_directory, TEST_ARM, horizon)
        if len(reference_runs) < 2 or len(test_runs) < 2:
            raise ValueError(f"{case_name} N={horizon}: {len(reference_runs)} {REFERENCE_ARM} and {len(test_runs)} "
                             f"{TEST_ARM} dumps in {dump_directory}; at least 2 of each")
        run_names = reference_runs + test_runs
        test_set = set(test_runs)
        arm_of = {**{name: REFERENCE_ARM for name in reference_runs}, **{name: TEST_ARM for name in test_runs}}
        invariants, provenance, invalid, foreign = run_checks(dump_directory, run_names, horizon, arm_of, records,
                                                              case_name)
        if invalid:
            raise ValueError(f"{case_name} N={horizon}: invalid runs (an invariant, a far migration or the exit code "
                             f"not 0): {invalid}")
        if foreign and not allow_foreign_dumps:
            raise ValueError(f"{case_name} N={horizon}: dumps not of this driver's arms: {foreign} "
                             f"(--allow-foreign-dumps to analyse them anyway)")
        digests, mixed = code_problems(provenance, arm_of)
        if mixed and not allow_mixed_code:
            raise ValueError(f"{case_name} N={horizon}: {mixed} (run the campaign again: runs of older code are "
                             f"repeated; or --allow-mixed-code)")
        reference = reference_runs[0]
        sidecar = json.loads(dump_file_paths(dump_directory, reference, horizon)["json"].read_text(encoding="utf-8"))
        fluid_groups = [index for index, kind in enumerate(sidecar["material_kinds"]) if int(kind) == 0]
        with np.load(dump_file_paths(dump_directory, reference, horizon)["npz"]) as archive:
            identifiers = archive["id"]
            fluid = np.isin(archive["material"], fluid_groups)
            near_wall = near_wall_mask(archive["position"], fluid, float(sidecar["smoothing_length"]))
        masks = {"all": np.ones(int(fluid.sum()), dtype=bool), "near_wall": near_wall}
        entry = {"runs": {REFERENCE_ARM: reference_runs, TEST_ARM: test_runs}, "reference_run": reference,
                 "fluid": int(fluid.sum()), "bin_counts": {label: int(mask.sum()) for label, mask in masks.items()},
                 "near_wall_definition": f"fluid particles with a non-fluid particle within h = "
                                         f"{float(sidecar['smoothing_length'])} (support radius) in {reference}'s "
                                         f"positions at N = {horizon}",
                 "invariants": invariants, "provenance": provenance, "provenance_problems": foreign,
                 "code_digests": digests, "code_problems": mixed,
                 "fields": {}, "identical": {}, "cards": {}, "sensitivity": [], "joint": {}}
        joint_inputs, scenarios = {}, {}
        for field in FIELDS:
            values = {}
            for name in run_names:
                with np.load(dump_file_paths(dump_directory, name, horizon)["npz"]) as archive:
                    if not np.array_equal(archive["id"], identifiers):
                        raise ValueError(f"{case_name} {name} N={horizon}: ids differ from {reference}")
                    values[name] = archive[field][fluid].astype(np.float64)
            pairs = {(first, second): pair_rms(values, first, second, masks)
                     for first, second in itertools.combinations(run_names, 2)}
            per_bin = {}
            for label in BINS:
                if not masks[label].any():
                    continue
                distances = {pair: bins[label] for pair, bins in pairs.items() if bins[label] is not None}
                per_bin[label] = renamed(summarize(run_names, test_set, distances))
                per_bin[label]["pairs"] = {f"{first}-{second}": value for (first, second), value in distances.items()}
                joint_inputs[(field, label)] = distances
                entry["cards"].setdefault(field, {})[label] = {
                    arm: card_split(runs, {pair: value for pair, value in distances.items()
                                           if pair[0] in runs and pair[1] in runs})
                    for arm, runs in ((REFERENCE_ARM, reference_runs), (TEST_ARM, test_runs))}
            entry["fields"][field] = per_bin
            if field in IDENTICAL_FIELDS:
                entry["identical"][field] = {
                    label: class_medians({(first, second): identical_share(values, first, second, masks[label])
                                          for first, second in itertools.combinations(run_names, 2)},
                                         lambda first, second: class_of(first, second, test_set))
                    for label in BINS if masks[label].any()}
            if (field, "all") == DEFECT_TEST:
                case_index = list(CASES).index(case_name) if case_name in CASES else 0
                seed = seed_base + 1000 * horizon_index + 10 * case_index
                rows, defects = sensitivity(values, reference_runs, test_runs, masks["all"],
                                            per_bin["all"]["median"]["within_v6"], seed)
                entry["sensitivity"] = [{("across_over_within_v6" if key == "cross_over_within_k1" else
                                          "within_v7_over_within_v6" if key == "within_x_over_within_k1" else key): value
                                         for key, value in row.items()} for row in rows]
                scenarios = {name: {DEFECT_TEST: distances} for name, distances in defects.items()}
            del values
        entry["joint"] = joint_tests(run_names, test_runs, joint_inputs, scenarios)
        entry["judgement"] = judgement(entry)
        result[str(horizon)] = entry
        verdict = entry["judgement"]
        print(f"{LOG_PREFIX} {case_name} N={horizon}: {len(reference_runs)} v6 + {len(test_runs)} v7 runs, "
              f"near-wall {entry['bin_counts']['near_wall']:,} of {entry['fluid']:,} fluid -> {verdict['verdict']} ("
              + ", ".join(f"{kind} {entry['joint'][kind]['count']}/{verdict['tests']} "
                          f"P {entry['joint'][kind]['p_at_least']:.3f}" for kind in JOINT_STATISTICS)
              + f"; across / within-v6 {verdict['ratio_range'][0]:.3f}-{verdict['ratio_range'][1]:.3f})"
              + "".join(f"; strongest {kind} test {text}" for kind, text in verdict["localized"].items()), flush=True)
    return result


# ----------------------------------------------------------------------------- tables

def case_label(case_name: str) -> str:
    return CASES.get(case_name, {}).get("label", case_name)


def markdown_tables(results: dict) -> str:
    lines = ["## 判定:v7 − v6 的差与 v6 − v6 的差是否可区分(每个算例 × 时刻 12 个检验 = 6 场 × 2 箱)", "",
             "P = 联合置换零分布:每一种重标同时作用于 12 个检验(保持它们的相关),P(≥) = 显著个数不少于观测值的重标所占比例,"
             "P(0) = 一个也不显著的比例;P(0, 两种) = 交叉与含 v7 两种统计量的 24 个检验都不显著的比例(E33 的定义)。判定:交叉与"
             "含 v7 的联合 P(≥) 都不小于 0.05 = 不可区分;否则组间项的联合检验也显著 = 系统偏移,不显著 = 只是起伏大小不同、"
             "看不出偏移(交叉统计量在两组只有起伏不同时也会升高,组间项不会)。比值 = 跨版本 / v6 内的中位数比;"
             "机会范围内 = 比值落在自身重标 2.5–97.5 % 范围内的检验个数。组间项 = D² / s₁²(偏移相对 v6 单次起伏)。"
             "最强单检验 = 12 个检验里 p 最小的一个与它的族校正 p(Westfall–Young min-p,同一组重标):个数统计在只有一两个"
             "检验不同时几乎不动,它会;判定为不可区分而它 < 0.05 时判定后加\"one test stands out\"。", "",
             "| 算例 | N | 运行 v6 + v7 | 交叉 p<0.05 / P(≥) / P(0) | 含 v7 p<0.05 / P(≥) / P(0) | "
             "组间项 p<0.05 / P(≥) / P(0) | P(0, 两种) | 最强单检验 交叉 (p, 族校正) | 最强单检验 含 v7 (p, 族校正) | "
             "比值范围 | 机会范围内 | v7 内 / v6 内 | 组间项范围 | 判定 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for case_name, horizons in results.items():
        for horizon, entry in horizons.items():
            joint, verdict = entry["joint"], entry["judgement"]
            ratio_range = verdict["ratio_range"] or [float("nan")] * 2
            within_range = verdict["within_ratio_range"] or [float("nan")] * 2
            lines.append(f"| {case_label(case_name)} | {horizon} | {len(entry['runs'][REFERENCE_ARM])} + "
                         f"{len(entry['runs'][TEST_ARM])} | "
                         + " | ".join(f"{joint[kind]['count']}/{joint[kind]['tests']} / {joint[kind]['p_at_least']:.3f} / "
                                      f"{joint[kind]['p_none']:.2f}" for kind in JOINT_STATISTICS)
                         + f" | {joint['p_none_both']:.2f} | "
                         + " | ".join(f"{joint[kind]['min_p_test']} ({joint[kind]['min_p']:.3f}, "
                                      f"{joint[kind]['min_p_familywise']:.3f})" for kind in ("cross", "one"))
                         + f" | {ratio_range[0]:.3f}–{ratio_range[1]:.3f} | "
                           f"{verdict['ratios_inside_chance_range']}/{verdict['tests']} | "
                           f"{within_range[0]:.3f}–{within_range[1]:.3f} | "
                           f"{verdict['between_range'][0]:+.3f}…{verdict['between_range'][1]:+.3f} | {verdict['verdict']} |")
    lines += ["", "## 每个场与箱(类中位数:v6 内 / v7 内 / 跨版本;p 交叉 / p 含 v7)", "",
              "| 算例 | N | 场 | 箱 | 粒子 | v6 内 | v7 内 | 跨版本 | 跨 / v6 内 | 机会 95 % | v7 内 / v6 内 | 机会 95 % | "
              "组间项 (p) | p 交叉 | p 含 v7 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for case_name, horizons in results.items():
        for horizon, entry in horizons.items():
            for field in FIELDS:
                for label, item in entry["fields"][field].items():
                    median = item["median"]
                    lines.append(f"| {case_label(case_name)} | {horizon} | {field} | {BIN_LABELS[label]} | "
                                 f"{entry['bin_counts'][label]:,} | {median['within_v6']:.3g} | {median['within_v7']:.3g} | "
                                 f"{median['across']:.3g} | {item['across_over_within_v6']:.3f} | "
                                 f"{item['ratio_null_95'][0]:.3f}–{item['ratio_null_95'][1]:.3f} | "
                                 f"{item['within_v7_over_within_v6']:.3f} | "
                                 f"{item['within_ratio_null_95'][0]:.3f}–{item['within_ratio_null_95'][1]:.3f} | "
                                 f"{item['between']:+.3f} ({item['between_p']:.3f}) | {item['cross_p']:.3f} | "
                                 f"{item['one_p']:.3f} |")
    lines += ["", "## 分辨力:速度 / 全部流体,在每次 v7 运行上加一个 rms 为 f × v6 内中位数的合成偏差", "",
              "系统 = 每次 v7 运行加同一个偏差;随机 = 每次独立。联合 = 12 个检验里速度 / 全部流体换成加偏差后的检验:"
              "p < 0.05 个数与 P(≥)(交叉 / 含 v7 / 组间项),以及交叉与含 v7 的族校正最小 p。", "",
              "| 算例 | N | 偏差 | f | 跨 / v6 内 | v7 内 / v6 内 | 组间项 (p) | p 交叉 | p 含 v7 | 联合 交叉 个数 P(≥) | "
              "联合 含 v7 个数 P(≥) | 联合 组间项 个数 P(≥) | 族校正 最小 p 交叉 / 含 v7 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for case_name, horizons in results.items():
        for horizon, entry in horizons.items():
            for row in entry["sensitivity"]:
                joint = entry["joint"]["scenarios"][scenario_name(row["kind"], row["factor"])]
                lines.append(f"| {case_label(case_name)} | {horizon} | {row['kind']} | {row['factor']:g} | "
                             f"{row['across_over_within_v6']:.3f} | {row['within_v7_over_within_v6']:.3f} | "
                             f"{row['between']:+.3f} ({row['between_p']:.3f}) | {row['cross_p']:.3f} | {row['one_p']:.3f} | "
                             + " | ".join(f"{joint[kind]['count']} {joint[kind]['p_at_least']:.3f}"
                                          for kind in JOINT_STATISTICS)
                             + f" | {joint['cross']['min_p_familywise']:.3f} / {joint['one']['min_p_familywise']:.3f} |")
    lines += ["", "## 逐位相同的流体粒子比例(每类的中位数;速度 / ρ)", "",
              "| 算例 | N | 箱 | v6 内 | v7 内 | 跨版本 |", "|---|---|---|---|---|---|"]
    for case_name, horizons in results.items():
        for horizon, entry in horizons.items():
            for label in BINS:
                if label not in entry["identical"].get(IDENTICAL_FIELDS[0], {}):
                    continue
                cells = [" / ".join(f"{entry['identical'][field][label][kind]['median']:.4f}"
                                    for field in IDENTICAL_FIELDS) for kind in CLASS_ORDER]
                lines.append(f"| {case_label(case_name)} | {horizon} | {BIN_LABELS[label]} | " + " | ".join(cells) + " |")
    lines += ["", "## 换卡:各臂自己的对按 GPU 分(速度;3 + 3 次运行的交叉检验 20 种重标,最小 p 0.1)", "",
              "| 算例 | N | 箱 | 臂 | GPU 0 × GPU 0 | GPU 1 × GPU 1 | GPU 0 × GPU 1 | 跨卡 / 同卡 | p 交叉 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for case_name, horizons in results.items():
        for horizon, entry in horizons.items():
            for label, arms in entry["cards"].get("velocity", {}).items():
                for arm, item in arms.items():
                    cells = [f"{item['median'][key]:.3g}" if key in item["median"] else "—"
                             for key in ("g0xg0", "g1xg1", "g0xg1")]
                    cells.append("—" if item.get("across_over_same") is None else f"{item['across_over_same']:.2f}")
                    cells.append(f"{item['cross_p']:.2f}" if "cross_p" in item else "—")
                    lines.append(f"| {case_label(case_name)} | {horizon} | {BIN_LABELS[label]} | {arm} | "
                                 + " | ".join(cells) + " |")
    lines += ["", "## 运行:不变量与来源", "",
              "| 算例 | N | 运行 | 版本 | 开关 | drift | overflow 合计 | far migration | 帧戳 GPU + 主机 | 代码摘要 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for case_name, horizons in results.items():
        for horizon, entry in horizons.items():
            for name, item in entry["invariants"].items():
                origin = entry["provenance"][name]
                switches = " ".join(f"{key}={value}" for key, value in sorted(origin["switches"].items()))
                lines.append(f"| {case_label(case_name)} | {horizon} | {name} | {origin['version']} | {switches} | "
                             f"{item.get('drift')} | {item.get('overflow_total')} | {item.get('far_migration_count', 0)} | "
                             f"{item.get('stamp_errors_gpu')} + {item.get('stamp_errors_host')} | "
                             f"{origin.get('code_digest') or '—'} |")
    return "\n".join(lines) + "\n"


def write_outputs(results: dict, out_directory: pathlib.Path, docs_directory=None, stem: str = "ensemble") -> None:
    out_directory.mkdir(parents=True, exist_ok=True)
    (out_directory / f"{stem}.json").write_text(json.dumps(json_ready(results), indent=1), encoding="utf-8")
    tables = markdown_tables(results)
    (out_directory / f"{stem}.md").write_text(tables, encoding="utf-8")
    if docs_directory:
        docs = pathlib.Path(docs_directory)
        docs.mkdir(parents=True, exist_ok=True)
        (docs / "e39_ensemble_tables.md").write_text(tables, encoding="utf-8")
        summary = {case_name: {horizon: {**{key: value for key, value in entry.items() if key != "fields"},
                                         "fields": {field: {label: {key: value for key, value in item.items()
                                                                    if key != "pairs"}
                                                            for label, item in bins.items()}
                                                    for field, bins in entry["fields"].items()}}
                               for horizon, entry in horizons.items()}
                   for case_name, horizons in results.items()}
        (docs / "e39_ensemble_summary.json").write_text(json.dumps(json_ready(summary), indent=1), encoding="utf-8")
        print(f"{LOG_PREFIX} wrote {docs / 'e39_ensemble_tables.md'} and e39_ensemble_summary.json", flush=True)
    print(f"{LOG_PREFIX} wrote {out_directory / (stem + '.json')} and {stem}.md", flush=True)


def analyze(out_directory: pathlib.Path, case_names: list, docs_directory=None, allow_mixed_code: bool = False,
            allow_foreign_dumps: bool = False) -> int:
    records = last_records_by_run(out_directory / "results.jsonl")
    results = {}
    for case_name in CASES:
        if case_names and case_name not in case_names:
            continue
        if not (out_directory / "dumps" / case_name).exists():
            print(f"{LOG_PREFIX} {case_name}: no dumps under {out_directory / 'dumps' / case_name}, skipped", flush=True)
            continue
        results[case_name] = analyze_case(case_name, out_directory, records, allow_mixed_code, allow_foreign_dumps)
    if not results:
        print(f"{LOG_PREFIX} nothing to analyze in {out_directory}", flush=True)
        return 1
    write_outputs(results, out_directory, docs_directory)
    return 0


# ----------------------------------------------------------------------------- self-test (synthetic dumps)

SYNTHETIC_FLUID_PER_SIDE = 111                  # 12,321 fluid particles on a lattice in (0, 1)^2
SYNTHETIC_WALL_LAYERS = 3
SYNTHETIC_SUPPORT_FACTOR = 2.5                  # h = 2.5 dx: the two fluid rows next to a wall are near-wall
# run-to-run noise per component (float32 fields: far above the float32 spacing of their base values)
SYNTHETIC_NOISE = {"velocity": 1e-5, "acceleration": 1e-3, "shift": 1e-7, "density": 1e-5, "pressure": 1e-2,
                   "kernel_sum": 1e-5}
SYNTHETIC_SCENARIOS = {"synthetic_null": "null", "synthetic_offset": "offset", "synthetic_wall_offset": "wall_offset",
                       "synthetic_spread": "spread"}
OFFSET_SCALE = 2 ** -0.5          # offset per component sigma / sqrt(2): across / within = sqrt(1.25) = 1.118
WALL_OFFSET_SCALE = 2.0           # near-wall offset per component 2 sigma: across / within there = sqrt(3) = 1.732
SPREAD_SCALE = 1.3                # v7 noise 1.3 sigma: within-v7 / within-v6 = 1.3


def synthetic_geometry() -> tuple:
    """(positions float32 (n, 2), material uint16: 0 fluid, 1 wall, 2 lid, support radius h)."""
    spacing = 1.0 / SYNTHETIC_FLUID_PER_SIDE
    fluid_axis = (np.arange(SYNTHETIC_FLUID_PER_SIDE) + 0.5) * spacing
    layers = SYNTHETIC_WALL_LAYERS
    full_axis = (np.arange(-layers, SYNTHETIC_FLUID_PER_SIDE + layers) + 0.5) * spacing
    grid_x, grid_y = np.meshgrid(full_axis, full_axis, indexing="ij")
    positions = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    inside = ((positions > 0) & (positions < 1)).all(axis=1)
    lid = positions[:, 1] > 1
    material = np.where(inside, 0, np.where(lid, 2, 1)).astype(np.uint16)
    assert int(inside.sum()) == fluid_axis.size ** 2
    return positions.astype(np.float32), material, SYNTHETIC_SUPPORT_FACTOR * spacing


def synthetic_base_fields(positions: np.ndarray) -> dict:
    x, y = positions[:, 0].astype(np.float64), positions[:, 1].astype(np.float64)
    swirl = np.stack([np.sin(np.pi * x) * np.cos(np.pi * y), -np.cos(np.pi * x) * np.sin(np.pi * y)], axis=1)
    return {"velocity": swirl, "acceleration": 3.0 * swirl[:, ::-1], "shift": 1e-4 * swirl,
            "density": 1000.0 + 0.5 * np.cos(np.pi * x) * np.cos(np.pi * y),
            "pressure": 50.0 * np.cos(np.pi * x) * np.cos(np.pi * y), "kernel_sum": 1.0 + 0.01 * x * y}


def write_synthetic_case(out_directory: pathlib.Path, case_name: str, scenario: str, trials: int, seed: int) -> None:
    """Dumps of 2 x trials fake K = 1 runs in the dump_state layout (both horizons): value = base + run noise, plus
    for the v7 runs the scenario's change; sidecars as this driver's runs write them."""
    positions, material, support_radius = synthetic_geometry()
    fluid = material == 0
    near_wall = np.zeros(material.size, dtype=bool)
    near_wall[fluid] = near_wall_mask(positions, fluid, support_radius)
    base = synthetic_base_fields(positions)
    dump_directory = out_directory / "dumps" / case_name
    dump_directory.mkdir(parents=True, exist_ok=True)
    count = material.size
    for horizon_index, horizon in enumerate(HORIZONS):
        generator = np.random.default_rng(seed + horizon_index)
        offsets = {}
        for field, sigma in SYNTHETIC_NOISE.items():
            shape = base[field].shape
            if scenario == "offset":
                offsets[field] = generator.normal(0.0, OFFSET_SCALE * sigma, size=shape)
            elif scenario == "wall_offset":
                offsets[field] = generator.normal(0.0, WALL_OFFSET_SCALE * sigma, size=shape)
                offsets[field][~near_wall] = 0.0
            else:
                offsets[field] = np.zeros(shape)
        for trial in range(1, trials + 1):
            for arm, solver in ARMS.items():
                name = run_name(arm, trial)
                state = {"id": np.arange(count, dtype=np.uint32), "slab": np.zeros(count, dtype=np.uint8),
                         "position": positions, "material": material,
                         "previous_slab": np.zeros(count, dtype=np.uint8),
                         "crossed_last_step": np.zeros(count, dtype=bool)}
                for field, sigma in SYNTHETIC_NOISE.items():
                    scale = SPREAD_SCALE * sigma if (scenario == "spread" and arm == TEST_ARM) else sigma
                    value = base[field] + generator.normal(0.0, scale, size=base[field].shape)
                    if arm == TEST_ARM:
                        value = value + offsets[field]
                    state[field] = value if field == "density" else value.astype(np.float32)
                state["velocity"][~fluid] = 0.0        # walls: prescribed state, identical in every run
                paths = dump_file_paths(dump_directory, name, horizon)
                save_npz_atomically(paths["npz"], state)
                save_json_atomically(paths["json"], {
                    "tool": "e39_ensemble selftest", "synthetic_scenario": scenario, "run_name": name,
                    "version": solver, "slabs": 1, "device_map": [gpu_of_trial(trial)], "horizon": horizon,
                    "cuts": [], "dimension": 2, "origin_x": 0.0, "smoothing_length": support_radius,
                    "material_kinds": [0, 1, 1], "material_names": ["fluid", "wall", "lid"],
                    "environment": expected_switches(solver),
                    "bookkeeping": {"density_exact_float64": True},
                    "invariants": {"drift": 0, "missing_ids": 0, "duplicate_ids": 0, "stamp_errors_gpu": 0,
                                   "stamp_errors_host": 0, "overflow_total": 0, "far_migration_count": 0,
                                   "warnings": [], "valid": True}})


def selftest(out_directory: pathlib.Path, trials: int, null_replicates: int) -> int:
    failures = []
    results = {}
    started = time.time()
    for index, (case_name, scenario) in enumerate(SYNTHETIC_SCENARIOS.items()):
        write_synthetic_case(out_directory, case_name, scenario, trials, seed=3900 + 10 * index)
        results[case_name] = analyze_case(case_name, out_directory, seed_base=39000 + 100 * index)
    write_outputs(results, out_directory)
    minimum_p = 2.0 / len(list(itertools.combinations(range(2 * trials), trials)))   # the symmetric cross statistic

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)
    for horizon, entry in results["synthetic_null"].items():
        check(entry["judgement"]["verdict"] == "indistinguishable", f"null N={horizon}: {entry['judgement']}")
        for row in entry["sensitivity"]:
            if row["kind"] == "systematic":
                check(row["cross_p"] <= minimum_p + 1e-9, f"null N={horizon}: synthetic systematic defect f={row['factor']} "
                                                           f"not detected (cross p {row['cross_p']})")
                # one test changed: the count moves by one, the family-wise min-p must see it
                familywise = entry["joint"]["scenarios"][scenario_name(row["kind"], row["factor"])]["cross"]
                check(familywise["min_p_familywise"] < SIGNIFICANCE and familywise["min_p_test"] == "/".join(DEFECT_TEST),
                      f"null N={horizon}: defect f={row['factor']} family-wise min-p {familywise}")
    expected_ratio = (1 + OFFSET_SCALE ** 2 / 2) ** 0.5
    for horizon, entry in results["synthetic_offset"].items():
        check(entry["judgement"]["verdict"] == "systematic offset", f"offset N={horizon}: {entry['judgement']}")
        for field in FIELDS:
            item = entry["fields"][field]["all"]
            check(item["cross_p"] <= minimum_p + 1e-9, f"offset N={horizon} {field}: cross p {item['cross_p']}")
            check(abs(item["across_over_within_v6"] / expected_ratio - 1) < 0.03,
                  f"offset N={horizon} {field}: ratio {item['across_over_within_v6']:.4f}, built {expected_ratio:.4f}")
            # between = D^2 / s_1^2 per component: offset variance sigma^2 / 2 over the v6 run scatter sigma^2
            check(abs(item["between"] - OFFSET_SCALE ** 2) < 0.1,
                  f"offset N={horizon} {field}: between {item['between']:.3f}, built {OFFSET_SCALE ** 2:.3f}")
    expected_wall_ratio = (1 + WALL_OFFSET_SCALE ** 2 / 2) ** 0.5
    for horizon, entry in results["synthetic_wall_offset"].items():
        check(entry["judgement"]["verdict"] == "systematic offset", f"wall_offset N={horizon}: {entry['judgement']}")
        for field in FIELDS:
            near, everywhere = entry["fields"][field]["near_wall"], entry["fields"][field]["all"]
            check(near["cross_p"] <= minimum_p + 1e-9, f"wall_offset N={horizon} {field}: near-wall cross p {near['cross_p']}")
            check(near["across_over_within_v6"] > everywhere["across_over_within_v6"],
                  f"wall_offset N={horizon} {field}: near-wall ratio not above the all-fluid ratio")
            check(abs(near["across_over_within_v6"] / expected_wall_ratio - 1) < 0.05,
                  f"wall_offset N={horizon} {field}: near-wall ratio {near['across_over_within_v6']:.4f}, "
                  f"built {expected_wall_ratio:.4f}")
    for horizon, entry in results["synthetic_spread"].items():
        check(entry["judgement"]["verdict"] == "different scatter, no offset detected",
              f"spread N={horizon}: {entry['judgement']}")
        for field in FIELDS:
            item = entry["fields"][field]["all"]
            check(abs(item["within_v7_over_within_v6"] / SPREAD_SCALE - 1) < 0.03,
                  f"spread N={horizon} {field}: within-v7 / within-v6 {item['within_v7_over_within_v6']:.4f}")
    false_alarms = {"cross": 0, "one": 0, "between": 0, "verdict": 0, "localized": 0, "judgements": 0}
    for replicate in range(null_replicates):
        case_name = f"synthetic_null_r{replicate + 1}"
        write_synthetic_case(out_directory, case_name, "null", trials, seed=500000 + 10 * replicate)
        for entry in analyze_case(case_name, out_directory, seed_base=600000 + 100 * replicate).values():
            false_alarms["judgements"] += 1
            for kind in JOINT_STATISTICS:
                false_alarms[kind] += entry["judgement"][f"{kind}_detected"]
            false_alarms["verdict"] += not entry["judgement"]["verdict"].startswith("indistinguishable")
            false_alarms["localized"] += bool(entry["judgement"]["localized"])
    if null_replicates:
        print(f"{LOG_PREFIX} selftest null replicates: {false_alarms['judgements']} (case, horizon) judgements; joint "
              f"test fired: cross {false_alarms['cross']}, one {false_alarms['one']}, between {false_alarms['between']}; "
              f"verdict not indistinguishable {false_alarms['verdict']} (each joint test is a 5 % test, the verdict "
              f"fires on cross or one); a family-wise min-p < 0.05 (cross or one) {false_alarms['localized']}",
              flush=True)
    for case_name, horizons in results.items():
        for horizon, entry in horizons.items():
            verdict = entry["judgement"]
            print(f"{LOG_PREFIX} selftest {case_name} N={horizon}: {verdict['verdict']}, ratios "
                  f"{verdict['ratio_range'][0]:.3f}-{verdict['ratio_range'][1]:.3f}, within-v7 / within-v6 "
                  f"{verdict['within_ratio_range'][0]:.3f}-{verdict['within_ratio_range'][1]:.3f}, between "
                  f"{verdict['between_range'][0]:+.3f}..{verdict['between_range'][1]:+.3f}; significant tests: "
                  + ", ".join(f"{kind} {len(verdict['significant'][kind])}" for kind in JOINT_STATISTICS), flush=True)
    print(f"{LOG_PREFIX} selftest {'PASS' if not failures else 'FAIL'} in {time.time() - started:.0f} s"
          + ("".join(f"\n  - {failure}" for failure in failures)), flush=True)
    return 0 if not failures else 1


# ----------------------------------------------------------------------------- command line

def resolved(path_text: str) -> pathlib.Path:
    path = pathlib.Path(path_text)
    return path if path.is_absolute() else (_REPOSITORY_ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=("run", "analyze", "selftest"))
    parser.add_argument("--out", default=None, help="campaign directory (default logs/e39/ensemble; selftest: "
                                                    "logs/e39/ensemble_selftest), relative to the checkout root")
    parser.add_argument("--cases", default="", help=f"comma list of {', '.join(CASES)} (default: both)")
    parser.add_argument("--trials", type=int, default=TRIALS, help=f"runs per arm (default {TRIALS}, at least 3)")
    parser.add_argument("--snapshot-root", default=DEFAULT_SNAPSHOT_ROOT,
                        help="directory holding cavity2d_1m/ and cavity3d_1m/snapshot_N2000.npz (default "
                             f"{DEFAULT_SNAPSHOT_ROOT} of this checkout)")
    parser.add_argument("--dry-run", action="store_true", help="run: print the plan and the estimates, no GPU")
    parser.add_argument("--preflight", action="store_true",
                        help="run --dry-run: also dump_state --dry-run per case and solver (CPU)")
    parser.add_argument("--docs", default=None, help="analyze: also write the tables and a summary here")
    parser.add_argument("--allow-mixed-code", action="store_true",
                        help="analyze: accept an arm whose runs recorded different code digests")
    parser.add_argument("--allow-foreign-dumps", action="store_true",
                        help="analyze: skip the provenance checks (solver version, switches, exact rho, K = 1)")
    parser.add_argument("--null-replicates", type=int, default=0,
                        help="selftest: this many more null cases, to count false alarms")
    arguments = parser.parse_args()
    case_names = [name for name in arguments.cases.split(",") if name]
    unknown = [name for name in case_names if name not in CASES]
    if unknown:
        parser.error(f"unknown case(s) {unknown}; expected {list(CASES)}")
    if arguments.trials < 3:
        parser.error("--trials must be at least 3 (the permutation tests need it)")
    if arguments.mode == "selftest":
        return selftest(resolved(arguments.out or "logs/e39/ensemble_selftest"), arguments.trials,
                        arguments.null_replicates)
    out_directory = resolved(arguments.out or "logs/e39/ensemble")
    if arguments.mode == "run":
        return run_campaign(out_directory, resolved(arguments.snapshot_root), case_names, arguments.trials,
                            arguments.dry_run, arguments.preflight)
    try:
        return analyze(out_directory, case_names, arguments.docs, arguments.allow_mixed_code,
                       arguments.allow_foreign_dumps)
    except ValueError as error:               # a refused input (invalid run, foreign dump, mixed code, too few runs)
        print(f"{LOG_PREFIX} analyze refused: {error}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
