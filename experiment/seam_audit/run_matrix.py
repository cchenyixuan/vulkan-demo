"""
run_matrix.py — sequential seam-audit campaign: one dump_state.py subprocess
per run (GPU), then analyze() for every (case, test configuration, horizon)
(CPU), then summary.md / summary.json.

Matrix file (JSON):
    {
      "cases":     [{"name": "cavity2d_1m", "path": "cases/.../case.yaml", "dimension": 2}, ...],
      "horizons":  [300, 2000],
      "reference": {"name": "v5_K1", "version": "v5", "slabs": 1, "device_map": "1", "trials": 2},
      "tests":     [{"name": "v5", "version": "v5", "slabs": 2, "device_map": "0,1",
                     "env": {}, "trials": 2,
                     "dimensions": [2, 3], "cases": ["cavity2d_1m"]   (optional filters),
                     "weights": [1, 1]                                 (optional)}, ...],
      optional: "depth" (2), "pool_safety" (1.2), "sync_scheme" ("per-direction"),
                "defrag_cadence" (case value), "timeout_s" (1800), "match" ("kdtree"),
                "particles" ("fluid"), "far_bin_start" (8)
    }

Every run's environment = os.environ + VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation
+ the version's production switches (<P> = V5_ / V6_):
    <P>WORKER_COUNT_AWARE=1 <P>SPLIT_TRANSFER_QUEUES=1 <P>CASCADE_FORCE=1
    <P>BAND_VOXEL_DISPATCH=1 <P>GHOST_POOL_FACTOR=0.25 (2-D) / 1.0 (3-D)
+ the configuration's own "env".

Run ids: <case>/<test name>_K<k>_t<trial>; the reference run id is DERIVED from
its content (reference_<version>_dev<devices>[_env<hash>]_K<k>_t<trial>), not
from the matrix, so two matrix files writing to the same --out share the
reference dumps (keep the case names identical across matrix files).

Output layout (--out):
    dumps/<case>/<run>_N<horizon>.npz|.json     (dump_state.py)
    logs/<case>/<run>.log                       (stdout+stderr, appended per attempt)
    runs.jsonl                                  (one record per attempt / skip)
    analysis/<case>/<test>_K<k>_N<horizon>/     (analyze.py: report.json/.md + PNGs)
    summary.md, summary.json                    (+ summary_<matrix stem>.md/.json copies)

Resumable: a run whose JSON sidecars exist for every horizon and say valid is
skipped (with --no-rerun-invalid: skipped if they merely exist). Before each run
the free space of the output drive is checked (stop below --minimum-free-gb).
Each run has a timeout (default 1800 s) after which its process tree is killed
(K=4 chains have stalled in the transport workers on this Windows rig before).
Ctrl+C stops the campaign within about a second: the running worker's process
tree is killed and the attempt is recorded as "interrupted". A relative --out
is taken from the current directory and handed to the workers as an absolute
path; the default is <repository>/logs/seam_audit/<matrix stem>.

Usage:
    .venv/Scripts/python.exe -m experiment.seam_audit.run_matrix \\
        --matrix experiment/seam_audit/matrix_v5_baseline.json --out logs/seam_audit/campaign
    ... --list                 (plan + completion status, no GPU)
    ... --dry-run-workers      (pre-flight, CPU only: every worker with --dry-run)
    ... --analyze-only         (re-run only the CPU analysis + summary)
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import time

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from experiment.seam_audit.dump_state import (  # noqa: E402
    EXIT_INVALID,
    RESULT_PREFIX,
    dump_file_paths,
    json_default,
)
from experiment.seam_audit.solver_adapter import (  # noqa: E402
    SUPPORTED_VERSIONS,
    environment_prefix,
)

LOG_PREFIX = "[seam_audit matrix]"
PRODUCTION_SWITCHES = {
    "WORKER_COUNT_AWARE": "1",
    "SPLIT_TRANSFER_QUEUES": "1",
    "CASCADE_FORCE": "1",
    "BAND_VOXEL_DISPATCH": "1",
}
GHOST_POOL_FACTOR_BY_DIMENSION = {2: "0.25", 3: "1.0"}
DEFAULT_TIMEOUT_SECONDS = 1800.0
WAIT_SLICE_SECONDS = 1.0                 # worker wait slice: Ctrl+C latency
DEFAULT_MINIMUM_FREE_GIGABYTES = 4.0
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
PRIMARY_FIELDS = ("acceleration", "shift", "density")


# =============================================================================
# Matrix + plan
# =============================================================================

@dataclasses.dataclass
class PlannedRun:
    case_name: str
    case_path: str
    dimension: int
    role: str                    # "reference" or "test"
    configuration: str           # test name, or the derived reference tag
    version: str
    slabs: int
    device_map: str
    environment: dict
    weights: list | None
    trial: int

    @property
    def run_name(self) -> str:
        return f"{self.configuration}_K{self.slabs}_t{self.trial}"

    @property
    def run_id(self) -> str:
        return f"{self.case_name}/{self.run_name}"

    @property
    def test_label(self) -> str:
        return f"{self.configuration}_K{self.slabs}"


def parse_environment(value) -> dict:
    """A config's env: a dict, or a string "KEY=VALUE KEY=VALUE" (commas ok)."""
    if not value:
        return {}
    if isinstance(value, dict):
        return {str(key): str(item) for key, item in value.items()}
    pairs = [item for item in re.split(r"[,\s]+", str(value)) if item]
    return dict(pair.split("=", 1) for pair in pairs)


def device_tag(device_map: str) -> str:
    return "-".join(item.strip() for item in str(device_map).split(",") if item.strip())


def reference_configuration(reference: dict) -> str:
    """Matrix-independent reference tag: version + devices (+ env hash)."""
    tag = f"reference_{reference['version']}_dev{device_tag(reference['device_map'])}"
    environment = parse_environment(reference.get("env"))
    if environment:
        digest = hashlib.sha1(json.dumps(environment, sort_keys=True).encode("utf-8"))
        tag += f"_env{digest.hexdigest()[:8]}"
    return tag


def load_matrix(path: pathlib.Path) -> dict:
    matrix = json.loads(path.read_text(encoding="utf-8"))
    for key in ("cases", "horizons", "reference", "tests"):
        if key not in matrix:
            raise ValueError(f"{path}: matrix needs '{key}'")
    for case in matrix["cases"]:
        if not NAME_PATTERN.match(str(case.get("name", ""))):
            raise ValueError(f"case name {case.get('name')!r} must match {NAME_PATTERN.pattern}")
        if int(case.get("dimension", 0)) not in GHOST_POOL_FACTOR_BY_DIMENSION:
            raise ValueError(f"case {case['name']}: dimension must be 2 or 3")
        if not (_REPOSITORY_ROOT / case["path"]).exists() and not pathlib.Path(case["path"]).exists():
            print(f"{LOG_PREFIX} WARNING: case file {case['path']} not found", file=sys.stderr)
    matrix["horizons"] = sorted({int(horizon) for horizon in matrix["horizons"]})
    if not matrix["horizons"] or matrix["horizons"][0] < 1:
        raise ValueError("horizons must be positive frame counts")
    reference = matrix["reference"]
    reference.setdefault("slabs", 1)
    reference.setdefault("device_map", "1")
    reference.setdefault("trials", 2)
    if reference.get("version") not in SUPPORTED_VERSIONS:
        raise ValueError(f"reference version must be one of {SUPPORTED_VERSIONS}")
    if int(reference["trials"]) < 2:
        raise ValueError("the reference needs at least 2 trials (A1, A2 noise floor)")
    for test in matrix["tests"]:
        if not NAME_PATTERN.match(str(test.get("name", ""))):
            raise ValueError(f"test name {test.get('name')!r} must match {NAME_PATTERN.pattern}")
        if test.get("version") not in SUPPORTED_VERSIONS:
            raise ValueError(f"test {test['name']}: version must be one of {SUPPORTED_VERSIONS}")
        test.setdefault("trials", 2)
        test.setdefault("device_map", "0,1")
        if int(test.get("slabs", 0)) < 1:
            raise ValueError(f"test {test['name']}: slabs must be >= 1")
    return matrix


def test_applies(test: dict, case: dict) -> bool:
    if "cases" in test and case["name"] not in test["cases"]:
        return False
    if "dimensions" in test and int(case["dimension"]) not in [int(value) for value
                                                               in test["dimensions"]]:
        return False
    return True


def test_selected(test: dict, only_tests: set) -> bool:
    if not only_tests:
        return True
    return test["name"] in only_tests or f"{test['name']}_K{int(test['slabs'])}" in only_tests


def selected_cases(matrix: dict, only_cases: set) -> list:
    unknown = only_cases - {case["name"] for case in matrix["cases"]}
    if unknown:
        raise ValueError(f"--only-cases: unknown case(s) {sorted(unknown)}")
    return [case for case in matrix["cases"] if not only_cases or case["name"] in only_cases]


def build_plan(matrix: dict, only_cases: set, only_tests: set) -> list:
    reference = matrix["reference"]
    reference_tag = reference_configuration(reference)
    tests = [test for test in matrix["tests"] if test_selected(test, only_tests)]
    if only_tests and not tests:
        raise ValueError(f"--only-tests matched no test in the matrix: {sorted(only_tests)}")
    plan = []
    for case in selected_cases(matrix, only_cases):
        applicable = [test for test in tests if test_applies(test, case)]
        maximum_trials = max([int(reference["trials"])]
                             + [int(test["trials"]) for test in applicable])
        for trial in range(1, maximum_trials + 1):
            if trial <= int(reference["trials"]):
                plan.append(PlannedRun(
                    case_name=case["name"], case_path=case["path"],
                    dimension=int(case["dimension"]), role="reference",
                    configuration=reference_tag, version=reference["version"],
                    slabs=int(reference["slabs"]), device_map=str(reference["device_map"]),
                    environment=parse_environment(reference.get("env")),
                    weights=reference.get("weights"), trial=trial))
            for test in applicable:
                if trial <= int(test["trials"]):
                    plan.append(PlannedRun(
                        case_name=case["name"], case_path=case["path"],
                        dimension=int(case["dimension"]), role="test",
                        configuration=test["name"], version=test["version"],
                        slabs=int(test["slabs"]), device_map=str(test["device_map"]),
                        environment=parse_environment(test.get("env")),
                        weights=test.get("weights"), trial=trial))
    run_ids = [run.run_id for run in plan]
    duplicates = sorted({run_id for run_id in run_ids if run_ids.count(run_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate run ids (same name + K for different configs?): {duplicates}")
    return plan


# =============================================================================
# Paths, status
# =============================================================================

def dump_directory(out_directory: pathlib.Path, run: PlannedRun) -> pathlib.Path:
    return out_directory / "dumps" / run.case_name


def log_path(out_directory: pathlib.Path, run: PlannedRun) -> pathlib.Path:
    return out_directory / "logs" / run.case_name / f"{run.run_name}.log"


def run_completion(run: PlannedRun, out_directory: pathlib.Path, horizons: list) -> str:
    """'complete' (every horizon sidecar valid), 'invalid' (all exist, some
    invalid) or 'incomplete'."""
    all_valid = True
    for horizon in horizons:
        paths = dump_file_paths(dump_directory(out_directory, run), run.run_name, horizon)
        if not paths["json"].exists() or not paths["npz"].exists():
            return "incomplete"
        try:
            sidecar = json.loads(paths["json"].read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return "incomplete"
        all_valid &= bool(sidecar.get("invariants", {}).get("valid", False))
    return "complete" if all_valid else "invalid"


def append_record(out_directory: pathlib.Path, record: dict) -> None:
    with open(out_directory / "runs.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=json_default) + "\n")


# =============================================================================
# Execution
# =============================================================================

def run_environment(run: PlannedRun) -> tuple[dict, dict]:
    prefix = environment_prefix(run.version)
    overrides = {
        "VK_LOADER_LAYERS_DISABLE": "VK_LAYER_KHRONOS_validation",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
    }
    for name, value in PRODUCTION_SWITCHES.items():
        overrides[prefix + name] = value
    overrides[prefix + "GHOST_POOL_FACTOR"] = GHOST_POOL_FACTOR_BY_DIMENSION[run.dimension]
    overrides.update(run.environment)
    environment = dict(os.environ)
    environment.update(overrides)
    return environment, overrides


def run_command(run: PlannedRun, matrix: dict, out_directory: pathlib.Path,
                python_executable: str, dry_run_workers: bool = False) -> list:
    command = [python_executable, "-m", "experiment.seam_audit.dump_state",
               "--version", run.version, "--case", run.case_path,
               "--slabs", str(run.slabs), "--device-map", run.device_map,
               "--horizons", ",".join(str(horizon) for horizon in matrix["horizons"]),
               "--depth", str(matrix.get("depth", 2)),
               "--pool-safety", str(matrix.get("pool_safety", 1.2)),
               "--sync-scheme", str(matrix.get("sync_scheme", "per-direction")),
               "--out-dir", str(dump_directory(out_directory, run)),
               "--run-name", run.run_name]
    if run.weights:
        weights = run.weights if isinstance(run.weights, str) else ",".join(
            str(weight) for weight in run.weights)
        command += ["--weights", weights]
    if matrix.get("defrag_cadence") is not None:
        command += ["--defrag-cadence", str(int(matrix["defrag_cadence"]))]
    window = window_for_case(matrix, run.case_name)
    if window > 0:
        command += ["--window", str(window),
                    "--audit-slab-counts", ",".join(str(count) for count in
                                                    audit_slab_counts(matrix, run.case_name))]
        if matrix.get("window_horizons"):
            command += ["--window-horizons",
                        ",".join(str(int(h)) for h in matrix["window_horizons"])]
        if run.role == "reference":
            requests_directory = capture_requests_directory(out_directory, run.case_name)
            if not requests_directory.exists():
                raise RuntimeError(
                    f"{run.run_id}: the reference pass of the crossing window needs "
                    f"{requests_directory} (run the tests first, then --build-capture-requests)")
            command += ["--capture-requests-dir", str(requests_directory)]
    if dry_run_workers:
        command.append("--dry-run")
    return command


def capture_requests_directory(out_directory: pathlib.Path, case_name: str) -> pathlib.Path:
    return out_directory / "capture_requests" / case_name


def build_capture_requests(matrix: dict, out_directory: pathlib.Path) -> None:
    """Union of the (frame, id) rows every TEST run of a case captured in its
    crossing window, per window horizon -> capture_requests/<case>/requests_N<h>.npz.
    Every test of every matrix sharing this --out contributes (the reference
    runs are shared), so build after ALL test runs and before the references."""
    import numpy as np
    horizons = matrix.get("window_horizons") or matrix["horizons"]
    for case in matrix["cases"]:
        directory = out_directory / "dumps" / case["name"]
        if window_for_case(matrix, case["name"]) <= 0 or not directory.exists():
            continue
        target = capture_requests_directory(out_directory, case["name"])
        target.mkdir(parents=True, exist_ok=True)
        for horizon in horizons:
            frames, ids, sources = [], [], 0
            for window_file in sorted(directory.glob(f"*_N{int(horizon)}_window.npz")):
                if window_file.name.startswith("reference_"):
                    continue
                with np.load(window_file) as archive:
                    if "id" not in archive.files or archive["frame"].size == 0:
                        sources += 1
                        continue
                    frames.append(archive["frame"].astype(np.int64))
                    ids.append(archive["id"].astype(np.int64))
                sources += 1
            frame_array = np.concatenate(frames) if frames else np.zeros(0, np.int64)
            id_array = np.concatenate(ids) if ids else np.zeros(0, np.int64)
            if frame_array.size:
                pairs = np.unique(np.stack([frame_array, id_array], axis=1), axis=0)
                frame_array, id_array = pairs[:, 0], pairs[:, 1]
            np.savez(target / f"requests_N{int(horizon)}.npz",
                     frame=frame_array.astype(np.int32), id=id_array.astype(np.uint32))
            print(f"{LOG_PREFIX} capture requests {case['name']} N={horizon}: "
                  f"{frame_array.size} rows over {np.unique(frame_array).size} frames "
                  f"from {sources} test windows", flush=True)


def case_entry(matrix: dict, case_name: str) -> dict:
    for case in matrix["cases"]:
        if case["name"] == case_name:
            return case
    raise KeyError(case_name)


def window_for_case(matrix: dict, case_name: str) -> int:
    """Crossing-capture window in frames (per case "window" overrides the matrix's)."""
    return int(case_entry(matrix, case_name).get("window", matrix.get("window", 0)))


def audit_slab_counts(matrix: dict, case_name: str) -> list:
    """Every K a test of this case runs with: all runs of the case (the K = 1
    reference included) watch the same equal-weight cut lines."""
    case = case_entry(matrix, case_name)
    counts = {int(test["slabs"]) for test in matrix["tests"] if test_applies(test, case)}
    return sorted(count for count in counts if count > 1)


def process_group_options() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def kill_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()


def wait_for_worker(process: subprocess.Popen, deadline: float) -> tuple:
    """Wait for the worker in short slices until it exits or ``deadline``
    (a time.perf_counter value) passes, then kill its process tree.
    Returns (return code, timed out). One long Popen.wait() is a single
    WaitForSingleObject on Windows that Ctrl+C cannot interrupt: the
    KeyboardInterrupt would surface only when the wait returned, up to the
    whole timeout later (the worker sits in its own process group and never
    sees the Ctrl+C). Between slices it surfaces within a second."""
    while True:
        try:
            return process.wait(timeout=WAIT_SLICE_SECONDS), False
        except subprocess.TimeoutExpired:
            if time.perf_counter() >= deadline:
                kill_process_tree(process)
                return process.returncode, True


def find_result(log_file: pathlib.Path, start_offset: int):
    with open(log_file, "rb") as handle:
        handle.seek(start_offset)
        text = handle.read().decode("utf-8", errors="replace")
    result = None
    for line in text.splitlines():
        if line.startswith(RESULT_PREFIX):
            try:
                result = json.loads(line[len(RESULT_PREFIX):])
            except ValueError:
                result = {"unparsable_result_line": line[:500]}
    return result, text[-2000:]


def execute_run(run: PlannedRun, matrix: dict, out_directory: pathlib.Path,
                timeout_seconds: float, python_executable: str,
                dry_run_workers: bool = False) -> dict:
    environment, overrides = run_environment(run)
    command = run_command(run, matrix, out_directory, python_executable, dry_run_workers)
    log_file = log_path(out_directory, run)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    dump_directory(out_directory, run).mkdir(parents=True, exist_ok=True)
    start_time = datetime.datetime.now().isoformat(timespec="seconds")
    with open(log_file, "ab") as log_handle:
        header = (f"\n===== {start_time} attempt: {' '.join(command)}\n"
                  f"===== environment overrides: {json.dumps(overrides, sort_keys=True)}\n")
        log_handle.write(header.encode("utf-8"))
        log_handle.flush()
        start_offset = log_handle.tell()
        start = time.perf_counter()
        process = subprocess.Popen(command, cwd=str(_REPOSITORY_ROOT), env=environment,
                                   stdout=log_handle, stderr=subprocess.STDOUT,
                                   **process_group_options())
        timed_out = False
        interrupted = False
        finished_before_interrupt = False
        try:
            return_code, timed_out = wait_for_worker(process, start + timeout_seconds)
        except KeyboardInterrupt:
            interrupted = True
            # a Ctrl+C that lands as the worker exits must not relabel a
            # finished run: its status comes from its own exit code and RESULT
            finished_before_interrupt = process.poll() is not None
            kill_process_tree(process)
            return_code = process.returncode
        wall_seconds = time.perf_counter() - start
    result, log_tail = find_result(log_file, start_offset)
    if interrupted and not finished_before_interrupt:
        status = "interrupted"
    elif timed_out:
        status = "timeout"
    elif return_code == 0 and result and result.get("valid"):
        status = "dry_run_ok" if result.get("dry_run") else "valid"
    elif return_code == EXIT_INVALID:
        status = "invalid"
    else:
        status = "failed"
    record = {
        "run_id": run.run_id, "case": run.case_name, "configuration": run.configuration,
        "role": run.role, "trial": run.trial, "version": run.version, "slabs": run.slabs,
        "device_map": run.device_map, "weights": run.weights,
        "environment_overrides": overrides, "command": command, "log": str(log_file),
        "start_time": start_time, "wall_time_s": round(wall_seconds, 2),
        "timeout_s": timeout_seconds, "return_code": return_code, "status": status,
        "result": result,
    }
    if status in ("failed", "timeout"):
        record["log_tail"] = log_tail
    if interrupted:
        record["interrupted"] = True        # Ctrl+C: the campaign stops after this run
    append_record(out_directory, record)
    if interrupted:
        raise KeyboardInterrupt
    return record


def free_bytes(directory: pathlib.Path) -> int:
    probe = directory
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


def run_campaign(plan: list, matrix: dict, out_directory: pathlib.Path, arguments) -> None:
    horizons = matrix["horizons"]
    timeout_seconds = float(arguments.timeout if arguments.timeout is not None
                            else matrix.get("timeout_s", DEFAULT_TIMEOUT_SECONDS))
    minimum_free = arguments.minimum_free_gb * 1e9
    total = len(plan)
    for index, run in enumerate(plan, start=1):
        completion = run_completion(run, out_directory, horizons)
        skip = completion == "complete" or (completion == "invalid"
                                            and arguments.no_rerun_invalid)
        if skip and not arguments.dry_run_workers:
            print(f"{LOG_PREFIX} [{index}/{total}] {run.run_id}: {completion}, skipped", flush=True)
            continue
        available = free_bytes(out_directory)
        if available < minimum_free:
            message = (f"free space {available / 1e9:.1f} GB < {arguments.minimum_free_gb} GB "
                       f"on the output drive; campaign stopped before {run.run_id}")
            print(f"{LOG_PREFIX} STOP: {message}", flush=True)
            append_record(out_directory, {"run_id": run.run_id, "status": "stopped_low_disk",
                                          "free_bytes": available, "message": message})
            return
        print(f"{LOG_PREFIX} [{index}/{total}] {run.run_id} ({run.version}, K={run.slabs}, "
              f"devices {run.device_map}, env {run.environment or '{}'}) "
              f"free {available / 1e9:.1f} GB ...", flush=True)
        record = execute_run(run, matrix, out_directory, timeout_seconds, arguments.python,
                             arguments.dry_run_workers)
        result = record.get("result") or {}
        details = ""
        if result.get("per_horizon"):
            details = " ".join(
                f"N={entry['horizon']}:{'ok' if entry['valid'] else 'INVALID'}"
                f"(drift {entry['drift']}, crossed {entry['crossed_last_step']})"
                for entry in result["per_horizon"])
        elif result.get("error"):
            details = result["error"]
        print(f"{LOG_PREFIX}     -> {record['status']} in {record['wall_time_s']:.0f} s "
              f"{details}", flush=True)


# =============================================================================
# Analysis + summary
# =============================================================================

def analysis_jobs(plan: list, matrix: dict, out_directory: pathlib.Path) -> list:
    """(case, test label, horizon, reference runs, test runs, output dir)."""
    jobs = []
    by_case: dict = {}
    for run in plan:
        by_case.setdefault(run.case_name, []).append(run)
    for case_name, runs in by_case.items():
        references = sorted((run for run in runs if run.role == "reference"),
                            key=lambda run: run.trial)[:2]
        test_labels = []
        for run in runs:
            if run.role == "test" and run.test_label not in test_labels:
                test_labels.append(run.test_label)
        for test_label in test_labels:
            tests = sorted((run for run in runs if run.role == "test"
                            and run.test_label == test_label), key=lambda run: run.trial)[:2]
            for horizon in matrix["horizons"]:
                jobs.append({
                    "case": case_name, "test": test_label, "horizon": horizon,
                    "references": references, "tests": tests,
                    "out_dir": out_directory / "analysis" / case_name / f"{test_label}_N{horizon}",
                })
    return jobs


def existing_dump(out_directory: pathlib.Path, run: PlannedRun, horizon: int):
    paths = dump_file_paths(dump_directory(out_directory, run), run.run_name, horizon)
    if paths["npz"].exists() and paths["json"].exists():
        return paths["npz"]
    return None


def analyze_campaign(plan: list, matrix: dict, out_directory: pathlib.Path) -> list:
    from experiment.seam_audit.analyze import analyze
    rows = []
    for job in analysis_jobs(plan, matrix, out_directory):
        reference_paths = [existing_dump(out_directory, run, job["horizon"])
                           for run in job["references"]]
        test_paths = [existing_dump(out_directory, run, job["horizon"]) for run in job["tests"]]
        row = {"case": job["case"], "test": job["test"], "horizon": job["horizon"],
               "out_dir": str(job["out_dir"])}
        missing = ([run.run_id for run, path in zip(job["references"], reference_paths)
                    if path is None]
                   + [run.run_id for run, path in zip(job["tests"], test_paths) if path is None])
        if len(job["references"]) < 2 or any(path is None for path in reference_paths) \
                or not test_paths or test_paths[0] is None:
            row.update({"status": "missing inputs", "missing": missing})
            print(f"{LOG_PREFIX} analysis {job['case']} {job['test']} N={job['horizon']}: "
                  f"missing inputs {missing}", flush=True)
            rows.append(row)
            continue
        test_paths = [path for path in test_paths if path is not None]
        try:
            report = analyze(reference_paths, test_paths,
                             match=matrix.get("match", "kdtree"), out_dir=job["out_dir"],
                             particle_filter=matrix.get("particles", "fluid"),
                             far_bin_start=int(matrix.get("far_bin_start", 8)))
            row.update(report["summary"])
            row.update({"case": job["case"], "test": job["test"], "horizon": job["horizon"],
                        "status": "ok" if not missing else "ok (partial inputs)",
                        "missing": missing})
            summary = report["summary"]
            print(f"{LOG_PREFIX} analysis {job['case']} {job['test']} N={job['horizon']}: "
                  f"seam_excess {format_ratio(summary['seam_excess'])}, d=0 acceleration "
                  f"flagged/unflagged {format_ratio(summary['d0_rms_ratio']['acceleration']['flagged'])}/"
                  f"{format_ratio(summary['d0_rms_ratio']['acceleration']['unflagged'])}",
                  flush=True)
        except Exception as error:  # keep going; the row records it
            row.update({"status": f"error: {type(error).__name__}: {error}"})
            print(f"{LOG_PREFIX} analysis {job['case']} {job['test']} N={job['horizon']}: "
                  f"ERROR {error}", flush=True)
        try:
            window_row = analyze_window_job(reference_paths, test_paths, job, matrix)
            if window_row:
                row["window"] = window_row
        except Exception as error:  # the window is an add-on; never lose the main row
            row["window_error"] = f"{type(error).__name__}: {error}"
            print(f"{LOG_PREFIX} window analysis {job['case']} {job['test']} N={job['horizon']}: "
                  f"ERROR {error}", flush=True)
        rows.append(row)
    return rows


def analyze_window_job(reference_paths, test_paths, job: dict, matrix: dict):
    """Pool the crossing-capture windows of A1, A2, B1(, B2) (see
    window_analysis.py). Returns the compact per-group ratios, or None when
    the runs have no window files."""
    import json as _json
    from experiment.seam_audit import window_analysis

    def window_path(dump_path):
        if dump_path is None:
            return None
        dump_path = pathlib.Path(dump_path)
        candidate = dump_path.with_name(dump_path.stem + "_window.npz")
        return candidate if candidate.exists() else None

    reference_windows = [window_path(path) for path in reference_paths]
    test_windows = [window_path(path) for path in test_paths]
    if any(path is None for path in reference_windows[:2]) or not test_windows \
            or test_windows[0] is None:
        return None
    test_windows = [path for path in test_windows if path is not None]
    sidecar = _json.loads(pathlib.Path(test_paths[0]).with_suffix(".json").read_text(encoding="utf-8"))
    cuts = sidecar.get("cuts") or sidecar.get("case", {}).get("cuts")
    origin_x = sidecar.get("origin_x")
    smoothing_length = sidecar.get("smoothing_length")
    column_count = sidecar.get("grid_nx") or sidecar.get("grid_dimension_x")
    report = window_analysis.analyze_window(
        reference_windows[:2], test_windows, cuts=cuts, origin_x=float(origin_x),
        smoothing_length=float(smoothing_length), column_count=int(column_count),
        match=matrix.get("match", "kdtree"),
        fluid_material_groups=([index for index, kind in enumerate(sidecar.get("material_kinds", []))
                                if kind == 0]
                               if matrix.get("particles", "fluid") == "fluid" else None))
    window_analysis.write_report(report, job["out_dir"],
                                 f"{job['case']} {job['test']} N={job['horizon']} crossing window")
    compact = {}
    for pair in report["pairs"]:
        if pair.get("empty"):
            continue
        entry = {"counts": pair["counts"]}
        for group in window_analysis.GROUPS:
            statistics = pair["statistics"][group]
            entry[group] = {
                "n": statistics["acceleration"]["test"].get("count", 0),
                "acceleration_rms": statistics["acceleration"]["ratio"]["rms"],
                "acceleration_max": statistics["acceleration"]["ratio"]["max"],
                "shift_rms": statistics["shift"]["ratio"]["rms"],
                "density_rms": statistics["density"]["ratio"]["rms"],
                "kernel_sum_rms": statistics["kernel_sum"]["ratio"]["rms"],
                "kernel_sum_signed_mean": (statistics["kernel_sum"].get("signed_test_minus_reference") or {}).get("mean"),
                "density_signed_mean": (statistics["density"].get("signed_test_minus_reference") or {}).get("mean"),
            }
        compact[pair["label"]] = entry
    first = next(iter(compact.values()), None)
    if first:
        print(f"{LOG_PREFIX} window {job['case']} {job['test']} N={job['horizon']}: "
              f"departed n={first['departed']['n']} accel rms ratio "
              f"{format_ratio(first['departed']['acceleration_rms'])}, arrived "
              f"{format_ratio(first['arrived']['acceleration_rms'])}, control "
              f"{format_ratio(first['control']['acceleration_rms'])}", flush=True)
    return compact


def format_ratio(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if value == float("inf"):
        return "inf"
    return f"{float(value):.3g}"


def far_bin_text(row: dict, field: str) -> str:
    entry = (row.get("far_bin_worst_rms_ratio") or {}).get(field) or {}
    if entry.get("value") is None:
        return "-"
    return f"{format_ratio(entry['value'])} (d={entry.get('bin')})"


def invariant_columns(row: dict) -> list:
    invariants = row.get("invariants") or {}
    drifts = [abs(values.get("drift") or 0) for values in invariants.values()]
    stamps_gpu = sum(values.get("stamp_errors_gpu") or 0 for values in invariants.values())
    stamps_host = sum(values.get("stamp_errors_host") or 0 for values in invariants.values())
    overflow = sum(values.get("overflow_total") or 0 for values in invariants.values())
    return [str(max(drifts) if drifts else "-"), f"{stamps_gpu}/{stamps_host}", str(overflow),
            str(row.get("all_runs_valid", "-"))]


def write_summary(rows: list, matrix_path: pathlib.Path, matrix: dict,
                  out_directory: pathlib.Path) -> None:
    generated = datetime.datetime.now().isoformat(timespec="seconds")
    reference = matrix["reference"]
    lines = [
        "# Seam audit campaign summary", "",
        f"- matrix: `{matrix_path}`; generated {generated}",
        f"- reference: {reference['version']} K={reference['slabs']} devices "
        f"{reference['device_map']} ({reference_configuration(reference)}), "
        f"{reference['trials']} trials; horizons {matrix['horizons']}",
        "- ratio = rms(|B1 - A1|) / rms(|A1 - A2|) on the same matched particles; "
        "d=0 = the seam column on both sides; flagged = a particle that crossed a seam "
        "in the last step lies within h",
        "",
        "## Seam column (d=0) rms ratio test/noise", "",
    ]
    header = ["case", "test", "N", "triplets"]
    for field in PRIMARY_FIELDS:
        header += [f"{field} all", f"{field} flagged", f"{field} unflagged"]
    header += ["far worst acceleration", "B2-A2 acceleration d0", "B1-B2 noise acceleration d0"]
    table = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        if not str(row.get("status", "")).startswith("ok"):
            table.append("| " + " | ".join([row["case"], row["test"], str(row["horizon"]),
                                             str(row.get("status"))]
                                            + ["-"] * (len(header) - 4)) + " |")
            continue
        cells = [row["case"], row["test"], str(row["horizon"]), f"{row.get('triplets', 0):,}"]
        for field in PRIMARY_FIELDS:
            ratios = row["d0_rms_ratio"][field]
            cells += [format_ratio(ratios["all"]), format_ratio(ratios["flagged"]),
                      format_ratio(ratios["unflagged"])]
        second = row.get("second_pair_d0_rms_ratio") or {}
        test_noise = row.get("test_noise_d0_rms_ratio") or {}
        cells += [far_bin_text(row, "acceleration"),
                  format_ratio(second.get("acceleration")),
                  format_ratio(test_noise.get("acceleration"))]
        table.append("| " + " | ".join(cells) + " |")
    lines += table
    lines += ["", "## Matching and invariants", ""]
    header = ["case", "test", "N", "column 0 all/flagged/unflagged/crossed",
              "unmatched B1-A1 (kd)", "unmatched A1-A2 (kd)", "id agreement B1-A1",
              "max abs drift", "stamps gpu/host", "overflow", "all runs valid", "report"]
    table = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        if not str(row.get("status", "")).startswith("ok"):
            continue
        counts = row.get("column0_counts") or {}
        unmatched = row.get("unmatched") or {}
        agreement = (row.get("id_agreement_rate") or {}).get("B1_to_A1")
        report_path = pathlib.Path(row["out_dir"]) / "report.md"
        try:
            report_link = report_path.relative_to(out_directory).as_posix()
        except ValueError:
            report_link = report_path.as_posix()
        cells = [row["case"], row["test"], str(row["horizon"]),
                 "/".join(f"{counts.get(name, 0):,}" for name in
                          ("all", "flagged", "unflagged", "crossed_self")),
                 f"{unmatched.get('B1_to_A1_kdtree', '-'):,}"
                 if isinstance(unmatched.get("B1_to_A1_kdtree"), int) else "-",
                 f"{unmatched.get('A1_to_A2_kdtree', '-'):,}"
                 if isinstance(unmatched.get("A1_to_A2_kdtree"), int) else "-",
                 format_ratio(agreement)] + invariant_columns(row) + [f"[report]({report_link})"]
        table.append("| " + " | ".join(cells) + " |")
    lines += table
    window_rows = [row for row in rows if row.get("window")]
    if window_rows:
        lines += ["", "## Crossing window (column 0, pooled over the window frames)", "",
                  "Groups of test column-0 fluid particles in frames with a seam crossing: departed = a "
                  "particle that LEFT this particle's slab this frame lies within h (V5 defect 2); arrived "
                  "= only arrivals within h; control = between h and 1.5 h of a crossing. Ratio = rms "
                  "|B1-A1| / rms |A1-A2| on the same particles and frame.", ""]
        header = ["case", "test", "N", "frames w/ crossings", "crossing particles",
                  "departed n", "departed accel", "departed density", "departed kernel_sum",
                  "arrived n", "arrived accel", "control n", "control accel", "unmatched"]
        table = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
        for row in window_rows:
            first = next(iter(row["window"].values()))
            counts = first["counts"]
            unmatched = sum(counts.get("unmatched", {}).values())
            table.append("| " + " | ".join([
                row["case"], row["test"], str(row["horizon"]),
                str(counts.get("frames_with_crossings")), str(counts.get("crossing_particles")),
                str(first["departed"]["n"]), format_ratio(first["departed"]["acceleration_rms"]),
                format_ratio(first["departed"]["density_rms"]),
                format_ratio(first["departed"]["kernel_sum_rms"]),
                str(first["arrived"]["n"]), format_ratio(first["arrived"]["acceleration_rms"]),
                str(first["control"]["n"]), format_ratio(first["control"]["acceleration_rms"]),
                str(unmatched)]) + " |")
        lines += table
    problems = [row for row in rows if not str(row.get("status", "")).startswith("ok")
                or row.get("missing")]
    if problems:
        lines += ["", "## Missing or failed", ""]
        lines += [f"- {row['case']} {row['test']} N={row['horizon']}: {row.get('status')} "
                  f"{row.get('missing') or ''}" for row in problems]
    lines.append("")
    text = "\n".join(lines)
    from experiment.seam_audit.analyze import json_safe
    document = json_safe({"matrix": str(matrix_path), "generated": generated,
                          "horizons": matrix["horizons"], "reference": reference,
                          "rows": rows})
    for stem in ("summary", f"summary_{matrix_path.stem}"):
        (out_directory / f"{stem}.md").write_text(text, encoding="utf-8")
        (out_directory / f"{stem}.json").write_text(
            json.dumps(document, indent=1, default=json_default), encoding="utf-8")
    print(f"{LOG_PREFIX} summary: {out_directory / 'summary.md'}", flush=True)


# =============================================================================
# CLI
# =============================================================================

def parse_arguments(argument_list=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seam audit campaign driver.")
    parser.add_argument("--matrix", required=True, help="matrix JSON file")
    parser.add_argument("--out", default=None,
                        help="output directory; a relative path is taken from the current "
                             "directory (default <repository>/logs/seam_audit/<matrix stem>)")
    parser.add_argument("--only-cases", default="", help="comma-separated case names")
    parser.add_argument("--only-tests", default="",
                        help="comma-separated test names (or <name>_K<k> labels)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="skip the GPU runs; redo analysis + summary")
    parser.add_argument("--no-analysis", action="store_true", help="GPU runs only")
    parser.add_argument("--no-rerun-invalid", action="store_true",
                        help="skip runs whose sidecars exist even if invalid")
    parser.add_argument("--timeout", type=float, default=None,
                        help=f"per-run timeout in seconds (default matrix timeout_s or "
                             f"{DEFAULT_TIMEOUT_SECONDS:.0f})")
    parser.add_argument("--minimum-free-gb", type=float, default=DEFAULT_MINIMUM_FREE_GIGABYTES)
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter for the dump_state subprocesses")
    parser.add_argument("--roles", default="all", choices=["all", "test", "reference"],
                        help="run only test runs or only reference runs (two-pass window: "
                             "tests, then --build-capture-requests, then references)")
    parser.add_argument("--build-capture-requests", action="store_true",
                        help="CPU only: build capture_requests/ from the test windows and exit")
    parser.add_argument("--list", action="store_true",
                        help="print the plan with completion status and exit (no GPU)")
    parser.add_argument("--dry-run-workers", action="store_true",
                        help="pre-flight, CPU only: run every planned worker with --dry-run "
                             "(case load + partition + global-id mask check), no analysis")
    return parser.parse_args(argument_list)


def split_names(text: str) -> set:
    return {item.strip() for item in text.split(",") if item.strip()}


def main(argument_list=None) -> int:
    arguments = parse_arguments(argument_list)
    matrix_path = pathlib.Path(arguments.matrix)
    matrix = load_matrix(matrix_path)
    # Absolute, so this process and the workers (started with cwd = repository
    # root and given --out-dir) mean the same directory whatever the cwd is.
    if arguments.out:
        out_directory = pathlib.Path(arguments.out).resolve()
    else:
        out_directory = _REPOSITORY_ROOT / "logs" / "seam_audit" / matrix_path.stem
    plan = build_plan(matrix, split_names(arguments.only_cases),
                      split_names(arguments.only_tests))
    if arguments.roles != "all":
        plan = [run for run in plan if run.role == arguments.roles]
    if arguments.build_capture_requests:
        build_capture_requests(matrix, out_directory)
        return 0
    if arguments.list:
        for index, run in enumerate(plan, start=1):
            completion = run_completion(run, out_directory, matrix["horizons"])
            print(f"{index:3d}. {run.run_id:<55s} {run.role:<9s} {run.version} K={run.slabs} "
                  f"devices {run.device_map:<8s} env {run.environment or '{}'} -> {completion}")
        print(f"{LOG_PREFIX} {len(plan)} runs, horizons {matrix['horizons']}, out {out_directory}")
        return 0
    out_directory.mkdir(parents=True, exist_ok=True)
    print(f"{LOG_PREFIX} matrix {matrix_path}: {len(plan)} runs, horizons {matrix['horizons']}, "
          f"out {out_directory}", flush=True)
    if not arguments.analyze_only:
        run_campaign(plan, matrix, out_directory, arguments)
    if arguments.no_analysis or arguments.dry_run_workers:
        return 0
    rows = analyze_campaign(plan, matrix, out_directory)
    write_summary(rows, matrix_path, matrix, out_directory)
    return 0


if __name__ == "__main__":
    sys.exit(main())
