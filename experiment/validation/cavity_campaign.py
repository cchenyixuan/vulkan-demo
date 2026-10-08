"""cavity_campaign.py - plan, calibrate and run the Re = 1000 cavity validation runs (sequentially).

    .venv/Scripts/python.exe -m experiment.validation.cavity_campaign plan
    .venv/Scripts/python.exe -m experiment.validation.cavity_campaign calibrate [--runs id,id]
    .venv/Scripts/python.exe -m experiment.validation.cavity_campaign run [--runs id,id]
    .venv/Scripts/python.exe -m experiment.validation.cavity_campaign status

Each run is one cavity_runner.py process per segment with an explicit environment: every inherited V6_*
variable dropped, then the v6 release combination (cavity_runner.EXPECTED) and the variant. 'calibrate'
runs each case for a few sample intervals and records the throughput (logs/.../calibration.json); 'run'
prints and records the wall-clock estimate of every run before starting it (campaign.jsonl), holds a lock
file, waits for idle GPUs, watches the runner's heartbeat (a stalled process tree is killed) and resumes a
failed or killed segment from its newest checkpoint (at most --retries times).

--solver v7 (E39) drives experiment/v7 the same way: the inherited V7_* variables are dropped instead and the V7_*
release combination added (cavity_runner.expected_environments), every runner gets --solver v7, and the runs,
calibration, estimates, log and lock live in logs/validation/cavity_re1000_v7/ (the same run ids as v6). A run
directory started with the other solver stops the campaign (the runner refuses to resume it). v7 is calibrated on
its own (calibration.json of its directory):

    .venv/Scripts/python.exe -m experiment.validation.cavity_campaign calibrate --solver v7 \
        --runs n250_k2_float32_xi0p001_eps0p0025,n500_k2_float32_xi0p001_eps0p0025
    .venv/Scripts/python.exe -m experiment.validation.cavity_campaign plan --solver v7 \
        --runs n250_k2_float32_xi0p001_eps0p0025,n500_k2_float32_xi0p001_eps0p0025
    .venv/Scripts/python.exe -m experiment.validation.cavity_campaign run --solver v7 \
        --runs n250_k2_float32_xi0p001_eps0p0025,n500_k2_float32_xi0p001_eps0p0025
    # a run outside RUNS (e.g. an adami case: K = 1 only): the runner under the same environment, started from the
    # repository root (no heartbeat watchdog, no retries, no idle-GPU wait)
    .venv/Scripts/python.exe -c "import subprocess, sys; from experiment.validation import cavity_campaign; \
sys.exit(subprocess.call([sys.executable] + sys.argv[1:], env=cavity_campaign.environment('release', 'v7')))" \
        -m experiment.validation.cavity_runner --solver v7 --expect release --slabs 1 --device-map 1 \
        --require-uuid ae137c668a40f5acaab90a83f7cda175 \
        --case cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025_adami/case.yaml \
        --run-dir logs/validation/cavity_re1000_v7/n250_k1_float32_xi0p001_eps0p0025_adami

In a worktree without its own .venv (e.g. vulkan-demo-v7-perf) the runners are started with this interpreter.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.validation.cavity_runner import SOLVERS, expected_environments, switch_prefix  # noqa: E402

ROOT = _REPO_ROOT / "logs" / "validation" / "cavity_re1000"
PYTHON = _REPO_ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():                 # a worktree without its own .venv: the interpreter running the campaign
    PYTHON = pathlib.Path(sys.executable)
CASES = {"n250": "cases/lid_driven_cavity_2d_n250/case.yaml", "n500": "cases/lid_driven_cavity_2d_n500/case.yaml",
         "n1000": "cases/lid_driven_cavity_2d_n1000/case.yaml", "n2000": "cases/lid_driven_cavity_2d_n2000/case.yaml",
         # 2026-10-04 (user decision): KCG regularization xi 0.1 -> 0.001, plus one 250^2 control with
         # epsilon_squared_factor 0.01 -> 0.0025; same particles as the base cases
         "n250_xi0p001": "cases/lid_driven_cavity_2d_n250_xi0p001/case.yaml",
         "n250_xi0p001_eps0p0025": "cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025/case.yaml",
         "n500_xi0p001": "cases/lid_driven_cavity_2d_n500_xi0p001/case.yaml",
         "n1000_xi0p001": "cases/lid_driven_cavity_2d_n1000_xi0p001/case.yaml",
         # 2026-10-04 (user decision): xi 0.001 + epsilon_squared_factor 0.0025 becomes the main series
         "n500_xi0p001_eps0p0025": "cases/lid_driven_cavity_2d_n500_xi0p001_eps0p0025/case.yaml",
         "n1000_xi0p001_eps0p0025": "cases/lid_driven_cavity_2d_n1000_xi0p001_eps0p0025/case.yaml"}
# "size" (default: "case") selects the time step, the calibration entry and the calibration length; the numerics
# variants cost the same as their base case. Runs execute in this order.
RUNS = [
    {"id": "n250_k2_float32", "case": "n250", "slabs": 2, "devices": "0,1", "expect": "release"},
    {"id": "n250_k2_delta", "case": "n250", "slabs": 2, "devices": "0,1", "expect": "release_delta"},
    {"id": "n500_k2_float32", "case": "n500", "slabs": 2, "devices": "0,1", "expect": "release"},
    {"id": "n500_k2_delta", "case": "n500", "slabs": 2, "devices": "0,1", "expect": "release_delta"},
    {"id": "n1000_k2_float32", "case": "n1000", "slabs": 2, "devices": "0,1", "expect": "release"},
    {"id": "n1000_k2_delta", "case": "n1000", "slabs": 2, "devices": "0,1", "expect": "release_delta"},
    # the K = 1 run with xi = 0.1 (n1000_k1_float32) was stopped at t = 1.65 when xi changed; K = 1 is run with xi = 0.001
    {"id": "n250_k2_float32_xi0p001", "case": "n250_xi0p001", "size": "n250", "slabs": 2, "devices": "0,1",
     "expect": "release"},
    {"id": "n250_k2_float32_xi0p001_eps0p0025", "case": "n250_xi0p001_eps0p0025", "size": "n250", "slabs": 2,
     "devices": "0,1", "expect": "release"},
    {"id": "n500_k2_float32_xi0p001", "case": "n500_xi0p001", "size": "n500", "slabs": 2, "devices": "0,1",
     "expect": "release"},
    {"id": "n1000_k2_float32_xi0p001", "case": "n1000_xi0p001", "size": "n1000", "slabs": 2, "devices": "0,1",
     "expect": "release"},
    {"id": "n1000_k1_float32_xi0p001", "case": "n1000_xi0p001", "size": "n1000", "slabs": 1, "devices": "1",
     "expect": "release", "require_uuid": "ae137c668a40f5acaab90a83f7cda175"},          # the headless 5090
    {"id": "n500_k2_float32_xi0p001_eps0p0025", "case": "n500_xi0p001_eps0p0025", "size": "n500", "slabs": 2,
     "devices": "0,1", "expect": "release"},
    {"id": "n1000_k2_float32_xi0p001_eps0p0025", "case": "n1000_xi0p001_eps0p0025", "size": "n1000", "slabs": 2,
     "devices": "0,1", "expect": "release"},
    {"id": "n1000_k1_float32_xi0p001_eps0p0025", "case": "n1000_xi0p001_eps0p0025", "size": "n1000", "slabs": 1,
     "devices": "1", "expect": "release", "require_uuid": "ae137c668a40f5acaab90a83f7cda175"},
    # 2026-10-05 (user request): the 1000^2 main-series run continued beyond t = 100 to test time convergence; the
    # directory is a copy of n1000_k2_float32_xi0p001_eps0p0025 resumed from its t = 99.2 checkpoint (run --t-end 300)
    {"id": "n1000_k2_float32_xi0p001_eps0p0025_long", "case": "n1000_xi0p001_eps0p0025", "size": "n1000", "slabs": 2,
     "devices": "0,1", "expect": "release"},
]
DT = {"n250": 3.0e-5, "n500": 1.5e-5, "n1000": 7.5e-6, "n2000": 3.75e-6}       # 0.15 * h / c0, h = 5 dx, c0 = 100
CALIBRATION_STEPS = {"n250": 42000, "n500": 52000, "n1000": 108000, "n2000": 159000}   # about 1.2-1.6 time units
HEARTBEAT_TIMEOUT_S = 600
BOOT_TIMEOUT_S = 900


def campaign_root(solver: str) -> pathlib.Path:
    """Run directories, calibration, estimates, log and lock of one solver: ROOT for v6, ROOT_<solver> otherwise."""
    return ROOT if solver == "v6" else ROOT.with_name(f"{ROOT.name}_{solver}")


def environment(expect: str, solver: str = "v6") -> dict:
    env = {key: value for key, value in os.environ.items() if not key.startswith(switch_prefix(solver))}
    env.update(expected_environments(solver)[expect])
    env["VK_LOADER_LAYERS_DISABLE"] = "VK_LAYER_KHRONOS_validation"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_solver(run_dir: pathlib.Path) -> str | None:
    """The solver a run directory was started with (meta.json; v6 if it predates --solver), None before a run."""
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8")).get("solver", "v6")


SOLVER_MARKERS = ("experiment.validation", "experiment.v6", "experiment.v5", "experiment.seam_audit", "_run_")


def command_line(pid: str) -> str:
    try:
        return subprocess.run(["powershell", "-NoProfile", "-Command",
                               f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"],
                              capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return ""


def gpu_compute_processes() -> list[str] | None:
    """Solver processes holding a GPU context, or None when nvidia-smi fails (treated as busy). On Windows
    (WDDM) nvidia-smi lists every process with a GPU context, desktop apps included; only Python processes
    whose command line names one of this repository's solver modules count."""
    try:
        completed = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=60)
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    busy = []
    for line in completed.stdout.splitlines():
        if "python" not in line.lower():
            continue
        pid = line.split(",")[0].strip()
        commandline = command_line(pid)
        if any(marker in commandline for marker in SOLVER_MARKERS):
            busy.append(f"{pid}: {commandline[:160]}")
    return busy


def wait_for_idle_gpus(log=print) -> None:
    """Wait (logging every 10 minutes) until no other solver process holds a GPU."""
    waited = 0
    while True:
        busy = gpu_compute_processes()
        if busy == []:
            return
        if waited % 600 == 0:
            log(f"waiting for idle GPUs: {'nvidia-smi failed' if busy is None else busy}")
        time.sleep(60)
        waited += 60


def kill_tree(pid: int) -> None:
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)


def estimate(run: dict, calibration: dict, t_end: float, t_max: float, start_step: int = 0) -> dict:
    """Wall-clock estimate from the calibrated throughput: to t_end (the earliest stop, a lower bound) and to
    t_max (the latest stop, when the steady rule never fires), from the newest checkpoint on a resume."""
    size = run.get("size", run["case"])
    dt = DT[size]
    steps_end = max(0, int(round(t_end / dt)) - start_step)
    steps_max = max(0, int(round(t_max / dt)) - start_step)
    key = f"{size}_k{run['slabs']}"
    fps = calibration.get(key, {}).get("fps")
    if fps and run["expect"] == "release_delta":
        fps *= calibration.get(key, {}).get("delta_factor", 0.988)
    return {"run": run["id"], "start_step": start_step, "steps_to_t_end": steps_end, "t_end": t_end,
            "steps_to_t_max": steps_max, "t_max": t_max, "fps": fps, "fps_source": key,
            "hours": steps_end / fps / 3600 if fps else None, "hours_max": steps_max / fps / 3600 if fps else None}


def run_segments(run: dict, run_dir: pathlib.Path, extra: list[str], retries: int, log, solver: str = "v6") -> int:
    env = environment(run["expect"], solver)
    attempt = 0
    while True:
        resume = attempt > 0 or (run_dir / "checkpoints" / "manifest.jsonl").exists()
        command = [str(PYTHON), "-m", "experiment.validation.cavity_runner", "--case", CASES[run["case"]],
                   "--run-dir", str(run_dir), "--slabs", str(run["slabs"]), "--device-map", run["devices"],
                   "--expect", run["expect"], "--solver", solver] + (["--resume"] if resume else []) + extra
        if run.get("require_uuid"):
            command += ["--require-uuid", run["require_uuid"]]
        log(f"launch {run['id']} attempt {attempt}: {' '.join(command[2:])}")
        started = time.time()
        with open(run_dir.parent / f"{run['id']}.out", "a", encoding="utf-8") as output:
            process = subprocess.Popen(command, cwd=_REPO_ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
            while process.poll() is None:
                time.sleep(30)
                try:
                    mtime = (run_dir / "heartbeat.json").stat().st_mtime
                except OSError:
                    mtime = None
                fresh = mtime is not None and mtime >= started
                age = time.time() - (mtime if fresh else started)
                limit = HEARTBEAT_TIMEOUT_S if fresh else BOOT_TIMEOUT_S
                if age > limit:
                    log(f"{run['id']}: no heartbeat for {age:.0f} s, killing process tree {process.pid}")
                    kill_tree(process.pid)
                    process.wait(timeout=900)
                    break
        code = process.returncode
        log(f"{run['id']} attempt {attempt} exited {code} after {(time.time() - started) / 3600:.2f} h")
        if code == 0:
            return 0
        if code == 3:
            log(f"{run['id']}: INVARIANT VIOLATION (see {run_dir / 'segments.jsonl'}); not resumed")
            return code
        attempt += 1
        if attempt > retries:
            return code if code else 1
        time.sleep(120)
        wait_for_idle_gpus(log)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("plan", "calibrate", "run", "status"))
    parser.add_argument("--runs", default=None, help="comma-separated run ids (default: all)")
    parser.add_argument("--t-end", type=float, default=100.0)
    parser.add_argument("--average-span", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--solver", choices=SOLVERS, default="v6",
                        help="passed to every cavity_runner; v7 (E39) runs live in logs/validation/cavity_re1000_v7")
    arguments = parser.parse_args()
    root = campaign_root(arguments.solver)
    root.mkdir(parents=True, exist_ok=True)
    selected = [run for run in RUNS if arguments.runs is None or run["id"] in arguments.runs.split(",")]
    calibration_path = root / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.exists() else {}
    campaign_log = open(root / "campaign.log", "a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"[campaign {time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        campaign_log.write(line + "\n")
        campaign_log.flush()

    if arguments.command == "status":
        for run in selected:
            run_dir = root / run["id"]
            rows = (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines() if (run_dir / "samples.jsonl").exists() else []
            last = json.loads(rows[-1]) if rows else {}
            result = (run_dir / "result.json").exists()
            print(f"{run['id']:<20} samples {len(rows):>5}  t = {last.get('time', 0):7.2f}  fps {last.get('fps', 0):7.1f}  "
                  f"{'COMPLETE' if result else ''}")
        return 0
    if arguments.command == "plan":
        total = 0.0
        t_max = max(200.0, arguments.t_end + 2 * arguments.average_span)
        total_max = 0.0
        for run in selected:
            entry = estimate(run, calibration, arguments.t_end, t_max)
            total += entry["hours"] or 0.0
            total_max += entry["hours_max"] or 0.0
            print(json.dumps(entry))
        print(f"total (calibrated runs only): {total:.1f} h to t_end, at most {total_max:.1f} h (t_max)")
        return 0

    lock = root / "campaign.lock"
    if lock.exists():
        other = lock.read_text(encoding="utf-8").strip()
        sys.exit(f"[campaign] lock file {lock} exists (pid {other}); remove it only if that process is gone")
    lock.write_text(str(os.getpid()), encoding="utf-8")
    try:
        if arguments.command == "calibrate":
            for run in selected:
                case = run.get("size", run["case"])
                key = f"{case}_k{run['slabs']}"
                if run["expect"] != "release" or key in calibration:
                    continue
                wait_for_idle_gpus(log)
                run_dir = root / "calibration" / run["id"]
                run_dir.mkdir(parents=True, exist_ok=True)
                code = run_segments(run, run_dir, ["--max-steps", str(CALIBRATION_STEPS[case])], 0, log,
                                    arguments.solver)
                rows = [json.loads(line) for line in (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
                rates = sorted(row["fps"] for row in rows[1:]) if len(rows) > 1 else []
                calibration[key] = {"fps": rates[len(rates) // 2] if rates else None, "samples": len(rows),
                                    "exit": code, "sample_s": [row["sample_s"] for row in rows]}
                calibration_path.write_text(json.dumps(calibration, indent=1), encoding="utf-8")
                log(f"calibrated {key}: {calibration[key]['fps']} steps/s over {len(rows)} samples (exit {code})")
            return 0
        for run in selected:
            run_dir = root / run["id"]
            started_with = run_solver(run_dir)
            if started_with not in (None, arguments.solver):
                log(f"{run['id']}: {run_dir} holds a {started_with} run, not {arguments.solver}; stopping the campaign")
                return 1
            if (run_dir / "result.json").exists():
                log(f"{run['id']}: complete, skipped")
                continue
            t_max = max(200.0, arguments.t_end + 2 * arguments.average_span)
            start_step = 0
            manifest = run_dir / "checkpoints" / "manifest.jsonl"
            if manifest.exists():
                steps = [json.loads(line)["step"] for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
                start_step = max(steps) if steps else 0
            entry = estimate(run, calibration, arguments.t_end, t_max, start_step)
            with open(root / "campaign.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(entry, logged=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
            log(f"{run['id']}: estimate from step {start_step:,}: {entry['steps_to_t_end']:,} steps to t = {arguments.t_end:g} "
                f"at {entry['fps']} steps/s -> {entry['hours'] if entry['hours'] is None else round(entry['hours'], 2)} h "
                f"(at most {entry['hours_max'] if entry['hours_max'] is None else round(entry['hours_max'], 2)} h to t = {t_max:g})")
            wait_for_idle_gpus(log)
            run_dir.mkdir(parents=True, exist_ok=True)
            code = run_segments(run, run_dir, ["--t-end", str(arguments.t_end), "--t-max", str(t_max),
                                               "--average-span", str(arguments.average_span)], arguments.retries, log,
                                arguments.solver)
            if code != 0:
                log(f"{run['id']}: FAILED (exit {code}); stopping the campaign")
                return code
        return 0
    finally:
        lock.unlink(missing_ok=True)
        campaign_log.close()


if __name__ == "__main__":
    sys.exit(main())
