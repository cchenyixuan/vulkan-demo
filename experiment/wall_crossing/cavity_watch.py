"""
cavity_watch.py - stop a running cavity statistics run (cavity_run.py) when the fluid leaves the box or runs
away, and only then.

    .venv/Scripts/python.exe -m experiment.wall_crossing.cavity_watch --name n250_reflection

The runner itself ends a run that loses a particle (its invariant check: the alive count at every sample).
This watcher adds the two conditions the runner does not have, read from the files cavity_runner_entry writes
into the run directory while the run goes on:
    field_history.jsonl          "speed_maximum" of the fluid at every sample  >  --speed-limit x the lid speed
    wall_crossing_status.jsonl   "outside_box_count" at every position readback  >  0
(and, as a second line of defence, an "alive" count that differs from the first one seen). On a violation it
writes <run directory>/watch_stop.json with the reason and ends the runner process (the Windows pid in
runner.pid, with its children). Start cavity_run.py with --retries 0, or the launcher resumes the run from its
last checkpoint. Nothing else stops a run here: no density, pressure or budget criterion.

It exits 0 when the run has finished (result.json) or its runner process has gone, 2 after a stop, 3 when no
runner appeared within --start-timeout seconds. Rows written before the watcher started are ignored (a resumed
run's files hold the rows of its earlier segments).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import pathlib
import subprocess
import sys
import time

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.run_log import LOG_ROOT     # noqa: E402


def process_alive(process_id: int) -> bool:
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, process_id)     # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(handle)
    return code.value == 259                                                  # STILL_ACTIVE


def new_rows(path: pathlib.Path, offsets: dict) -> list:
    """The complete lines appended to a jsonl file since the last call."""
    if not path.exists():
        return []
    with open(path, "rb") as handle:
        handle.seek(offsets.get(path, 0))
        data = handle.read()
    end = data.rfind(b"\n")
    if end < 0:
        return []
    offsets[path] = offsets.get(path, 0) + end + 1
    rows = []
    for line in data[:end].decode("utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def runner_process_id(run_directory: pathlib.Path) -> int | None:
    path = run_directory / "runner.pid"
    try:
        return int(path.read_text(encoding="utf-8").split()[0])
    except (OSError, ValueError, IndexError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="run directory name under logs/wall_crossing/cavity")
    parser.add_argument("--lid-speed", type=float, default=1.0)
    parser.add_argument("--speed-limit", type=float, default=1.05, help="in units of the lid speed")
    parser.add_argument("--poll", type=float, default=5.0, help="seconds between two looks")
    parser.add_argument("--start-timeout", type=float, default=300.0)
    arguments = parser.parse_args()

    run_directory = LOG_ROOT / "cavity" / arguments.name
    started = time.time()
    # a result.json of an earlier, complete run of this directory would end the watch at once: none is expected
    # for a run that is started or resumed now (cavity_run.py refuses nothing here, the watcher just reports it)
    offsets: dict = {}
    seen_alive = False
    expected_alive = None
    limit = arguments.speed_limit * arguments.lid_speed
    print(f"[cavity_watch] {arguments.name}: stop when the fluid's largest speed exceeds {limit:g}, a fluid "
          "particle is outside the box, or the particle count changes", flush=True)
    while True:
        reason = None
        for row in new_rows(run_directory / "field_history.jsonl", offsets):
            if row.get("unix", 0.0) < started:
                continue
            if expected_alive is None:
                expected_alive = row.get("alive")
            if row.get("speed_maximum", 0.0) > limit:
                reason = (f"largest fluid speed {row['speed_maximum']:.4f} > {limit:g} at t = {row.get('time')}")
            elif row.get("alive") != expected_alive:
                reason = f"particle count {row.get('alive')} != {expected_alive} at t = {row.get('time')}"
        for row in new_rows(run_directory / "wall_crossing_status.jsonl", offsets):
            if row.get("unix", 0.0) < started:
                continue
            if row.get("outside_box_count", 0) > 0:
                reason = f"{row['outside_box_count']} fluid particle(s) outside the box (status read at unix {row['unix']:.0f})"
        process_id = runner_process_id(run_directory)
        running = process_id is not None and process_alive(process_id)
        seen_alive = seen_alive or running
        if reason is not None:
            record = {"unix": time.time(), "reason": reason, "runner_pid": process_id}
            (run_directory / "watch_stop.json").write_text(json.dumps(record, indent=1), encoding="utf-8",
                                                           newline="\n")
            print(f"[cavity_watch] {arguments.name}: STOP: {reason}", flush=True)
            if running:
                subprocess.run(["taskkill", "/PID", str(process_id), "/T", "/F"], check=False)
            return 2
        if seen_alive and not running:
            finished = (run_directory / "result.json").exists()
            print(f"[cavity_watch] {arguments.name}: the runner has ended "
                  f"({'result.json written' if finished else 'no result.json'}); no stop condition was met",
                  flush=True)
            return 0
        if not seen_alive and time.time() - started > arguments.start_timeout:
            print(f"[cavity_watch] {arguments.name}: no runner within {arguments.start_timeout:g} s", flush=True)
            return 3
        time.sleep(arguments.poll)


if __name__ == "__main__":
    sys.exit(main())
