"""
cavity_run.py - start one Re = 1000 lid-driven cavity validation run of the v7 solver on one GPU (K = 1) and
log it to logs/wall_crossing/runs.jsonl.

    .venv/Scripts/python.exe -m experiment.wall_crossing.cavity_run \
        --case cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025_crossing/case.yaml \
        --name n250_crossing --device 1 --uuid ae137c668a40f5acaab90a83f7cda175

The run itself is experiment/validation/cavity_runner.py (--solver v7 --expect release --slabs 1), the protocol of
docs/validation/cavity_re1000.md: from rest, a sample every 0.2 time units, steady test on windows of 10, stop
at max(t_end, t_steady + average_span) (at most t_max), checkpoints every 10. This launcher adds what the
campaign driver gives its own run list (cavity_campaign.py: the release environment, a heartbeat watchdog,
resume from the newest checkpoint after a failure) for a case outside that list, and the run record.

Run directory: logs/wall_crossing/cavity/<name> (the runner's layout: meta.json, samples.jsonl, samples/,
snapshots/, checkpoints/, result.json); the runner's output goes to logs/wall_crossing/cavity/<name>.out.
With WALL_CROSSING_LOG_ROOT set (the code running from a frozen checkout) both and runs.jsonl go under that
directory instead of this checkout's logs/wall_crossing.
A directory that already holds a run is resumed (the runner refuses a changed case or changed settings).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.run_log import LOG_ROOT, LoggedRun     # noqa: E402

HEARTBEAT_TIMEOUT_SECONDS = 600
BOOT_TIMEOUT_SECONDS = 900
POLL_SECONDS = 30
INVARIANT_EXIT_CODE = 3


def kill_process_tree(process_id: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process_id), "/T", "/F"], capture_output=True)
    else:
        subprocess.run(["pkill", "-TERM", "-P", str(process_id)], capture_output=True)
        subprocess.run(["kill", "-TERM", str(process_id)], capture_output=True)


def run_segment(command: list, environment: dict, output_path: pathlib.Path, run_directory: pathlib.Path) -> tuple:
    """One runner process. Returns (exit code, reason): reason "exit", or "stalled" when the heartbeat stopped
    and the process tree was killed."""
    heartbeat_path = run_directory / "heartbeat.json"
    started = time.time()
    with open(output_path, "a", encoding="utf-8", newline="\n") as output:
        output.write(f"\n===== segment started {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        output.flush()
        process = subprocess.Popen(command, cwd=REPOSITORY_ROOT, env=environment, stdout=output,
                                   stderr=subprocess.STDOUT)
        while True:
            try:
                return process.wait(timeout=POLL_SECONDS), "exit"
            except subprocess.TimeoutExpired:
                pass
            now = time.time()
            if heartbeat_path.exists() and heartbeat_path.stat().st_mtime >= started:
                silent = now - heartbeat_path.stat().st_mtime
                limit = HEARTBEAT_TIMEOUT_SECONDS
            else:
                silent = now - started
                limit = BOOT_TIMEOUT_SECONDS
            if silent > limit:
                output.write(f"\n===== no heartbeat for {silent:.0f} s: killing the process tree =====\n")
                output.flush()
                kill_process_tree(process.pid)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    pass
                return -1, "stalled"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True)
    parser.add_argument("--name", required=True, help="run directory name under logs/wall_crossing/cavity")
    parser.add_argument("--device", type=int, required=True, help="v7 device index (discrete-first order)")
    parser.add_argument("--uuid", required=True, help="Vulkan deviceUUID the device must have (hex, no dashes)")
    parser.add_argument("--t-end", type=float, default=100.0)
    parser.add_argument("--average-span", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=None, help="stop after this many steps (a pilot run)")
    parser.add_argument("--root", default=None,
                        help="directory of the run directories (default: <logs/wall_crossing>/cavity, i.e. "
                             "under WALL_CROSSING_LOG_ROOT when that is set)")
    parser.add_argument("--initial-density-offset", type=float, default=0.0,
                        help="control experiment: start every fluid particle at rho0 + this (kg/m3), pressure by "
                             "the EOS (cavity_runner_entry.py, item 4); 0 = the normal start")
    arguments = parser.parse_args()

    from experiment.validation import cavity_campaign
    environment = cavity_campaign.environment("release", "v7")
    environment["OPENBLAS_NUM_THREADS"] = "1"
    if arguments.initial_density_offset != 0.0:
        environment["WALL_CROSSING_INITIAL_DENSITY_OFFSET"] = repr(arguments.initial_density_offset)

    root = (REPOSITORY_ROOT / arguments.root).resolve() if arguments.root else LOG_ROOT / "cavity"
    run_directory = root / arguments.name
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / f"{arguments.name}.out"
    t_max = max(200.0, arguments.t_end + 2.0 * arguments.average_span)
    # the runner through cavity_runner_entry: the same main(), with a git_state that works without git
    base_command = [sys.executable, "-m", "experiment.wall_crossing.cavity_runner_entry", "--solver", "v7",
                    "--expect", "release", "--slabs", "1", "--device-map", str(arguments.device),
                    "--require-uuid", arguments.uuid, "--case", arguments.case, "--run-dir", str(run_directory),
                    "--t-end", str(arguments.t_end), "--t-max", str(t_max),
                    "--average-span", str(arguments.average_span)]
    if arguments.max_steps is not None:
        base_command += ["--max-steps", str(arguments.max_steps)]

    with LoggedRun(kind="cavity", case=arguments.case, gpu_uuids=[arguments.uuid]) as run:
        run.extra["name"] = arguments.name
        if arguments.initial_density_offset != 0.0:
            run.extra["initial_density_offset"] = arguments.initial_density_offset
        run.extra["run_directory"] = (f"cavity/{arguments.name}" if arguments.root is None
                                      else str(run_directory).replace("\\", "/"))
        attempts = []
        exit_code = None
        for attempt in range(arguments.retries + 1):
            holds_run = (run_directory / "samples.jsonl").exists() \
                or (run_directory / "checkpoints" / "manifest.jsonl").exists()
            command = base_command + (["--resume"] if holds_run else [])
            exit_code, reason = run_segment(command, environment, output_path, run_directory)
            attempts.append({"attempt": attempt, "resumed": holds_run, "exit_code": exit_code, "reason": reason})
            print(f"[cavity_run] {arguments.name}: segment {attempt} ended with exit code {exit_code} ({reason})",
                  flush=True)
            if exit_code == 0 or exit_code == INVARIANT_EXIT_CODE:
                break
            time.sleep(30)
        run.extra["segments"] = attempts
        result_path = run_directory / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            run.extra["result"] = {key: result.get(key) for key in
                                   ("steps", "time", "t_steady", "wall_s", "complete", "invariants_ok")
                                   if key in result}
        if exit_code == 0:
            run.extra["status"] = "ok" if arguments.max_steps is None else f"ok (stopped at --max-steps {arguments.max_steps})"
        elif exit_code == INVARIANT_EXIT_CODE:
            run.extra["status"] = "failed: invariant violation (runner exit code 3)"
        else:
            run.extra["status"] = f"failed: runner exit code {exit_code} after {len(attempts)} segment(s)"
        run.record["command"] = " ".join(base_command[1:])
    return 0 if exit_code == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
