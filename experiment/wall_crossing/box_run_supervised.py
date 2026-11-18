"""
box_run_supervised.py - a long box_run.py run (2026-11-19: the 3-D cavities) with the watchdog and the resume of
cavity_run.py: the run is restarted from its newest checkpoint when it fails or its heartbeat stops.

    .venv/Scripts/python.exe -m experiment.wall_crossing.box_run_supervised --retries 3 -- \
        --case cases/cavity3d_1m_crossing/case.yaml --name cavity3d_1m_crossing_default_volume --device 1 \
        --uuid ae137c668a40f5acaab90a83f7cda175 --time 40 --sample-time 0.1 --profile-time 1 \
        --checkpoint-time 5 --stop-speed 1.05 --stop-outside

Everything after "--" goes to box_run.py; --checkpoint-time must be given (a resume needs a checkpoint). One
box_run process is one segment (cavity_run.run_segment: heartbeat.json of the run directory, killed after
HEARTBEAT_TIMEOUT_SECONDS without it). Exit code 0: the run is complete; 3: a stop condition of box_run
(watch_stop.json): no retry; anything else (a failure, a stall): resumed with --resume, at most --retries times.
Output: the run's directory logs/wall_crossing/box_run/<name>/ and its box_run.out (every segment's output),
supervisor.jsonl (one line per segment); box_run's runs.jsonl records.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.cavity_run import run_segment     # noqa: E402
from experiment.wall_crossing.run_log import LOG_ROOT            # noqa: E402

STOP_EXIT_CODE = 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("box_run_arguments", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    forwarded = [item for item in arguments.box_run_arguments if item != "--"] \
        if arguments.box_run_arguments[:1] == ["--"] else arguments.box_run_arguments
    if "--checkpoint-time" not in forwarded:
        parser.error("give box_run's --checkpoint-time (a resume needs a checkpoint)")
    name = forwarded[forwarded.index("--name") + 1]
    directory = LOG_ROOT / "box_run" / name
    directory.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", OPENBLAS_NUM_THREADS="1")
    environment.setdefault("VK_LOADER_LAYERS_DISABLE", "VK_LAYER_KHRONOS_validation")
    for attempt in range(arguments.retries + 1):
        command = [sys.executable, "-m", "experiment.wall_crossing.box_run", *forwarded]
        if attempt > 0 or "--resume" in forwarded:
            command = [item for item in command if item != "--resume"] + ["--resume"]
        started = time.time()
        exit_code, reason = run_segment(command, environment, directory / "box_run.out", directory)
        with open(directory / "supervisor.jsonl", "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"attempt": attempt, "exit_code": exit_code, "reason": reason,
                                     "started": started, "ended": time.time(), "command": command[2:]}) + "\n")
        print(f"[box_run_supervised] {name}: attempt {attempt} ended with exit code {exit_code} ({reason})",
              flush=True)
        if exit_code == 0:
            return 0
        if exit_code == STOP_EXIT_CODE and reason == "exit":
            stop = directory / "watch_stop.json"
            print(f"[box_run_supervised] {name}: stop condition: "
                  f"{json.loads(stop.read_text(encoding='utf-8'))['reason'] if stop.exists() else '?'}", flush=True)
            return STOP_EXIT_CODE
    print(f"[box_run_supervised] {name}: giving up after {arguments.retries + 1} attempts", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
