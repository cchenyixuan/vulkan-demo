"""
cavity3d_levels.py - claude/wall_store_report.md section 15 (round 9): run the refinement levels of the 3-D cavity at
Re = 1000 (h/dx = 4, crossing) one after the other, each from the previous level's converged state
(cavity3d_refine.py; box_run --initial-state / --converge / --tube-time).

    python -m experiment.wall_crossing.cavity3d_levels --levels 200,280 --start <run dir of the converged level>
        --device 0,1 --uuid <uuid> --uuid <uuid> --deadline <unix time> [--maximum-size 280]

--start: the box_run directory of the level the chain starts from (its final.npz, summary.json: N, time,
volume_constraint_integral_final). Per level N of --levels:
  case       cases/cavity3d_re1000_n<N>_hdx4_crossing/case.yaml;
  start      the previous level's final.npz, interpolated with support --support-factor x the previous spacing (2.5;
             report 15.2: the previous level's h, 4 spacings, smooths the start's extrema by 0.7-2.7 points more),
             the previous level's I and time;
  run        box_run --time 25 (the brief's cap) --converge --lines-time 0.5 --tube-time 0.5 --sample-time 0.1
             --checkpoint-time 5 --stop-speed 1.5 --stop-outside;
  clean-up   once the new level's heartbeat passes level time 1, the previous level's final.npz and checkpoints are
             deleted (the brief: one level's last checkpoint is kept until the next level has started) - except the
             --start level's when --keep-start (its state is handed to another machine as well);
  watchdog   a heartbeat older than --stall-minutes (20) kills the run (recorded);
  failure    a run that stops (box_run exit 3: speed, particle loss, outside the box) or dies is tried once more with
             the smoother start (support = the previous level's h); a second failure skips the level;
  deadline   every level gets box_run --wall-deadline --deadline: past it the level ends at its next record like a
             converged one, marked not converged (end_reason wall_deadline), so the round ends in time;
  next level the brief's estimate (for the levels not in --required): the level time this level needed x the next level's wall seconds per unit time x 1.2,
             the next level's cost scaled from this level's measured one by (fluid particles) x (1 / dt); the chain
             stops when the estimate passes --deadline, or when N > --maximum-size.
Everything goes to <log root>/levels.jsonl.
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

from experiment.wall_crossing.cavity3d_refine import remove_level_files   # noqa: E402
from experiment.wall_crossing.run_log import LOG_ROOT                      # noqa: E402

STOP_EXIT_CODE = 3
SAFETY = 1.2


def case_path(size: int) -> str:
    return f"cases/cavity3d_re1000_n{size}_hdx4_crossing/case.yaml"


def fluid_count(size: int) -> int:
    return (size + 1) ** 3


def timestep(size: int) -> float:
    import yaml
    data = yaml.safe_load((REPOSITORY_ROOT / case_path(size)).read_text(encoding="utf-8"))
    return float(data["physics"]["timestep"])


def log(entry: dict) -> None:
    entry = {"unix": time.time(), "clock": time.strftime("%Y-%m-%d %H:%M:%S"), **entry}
    with open(LOG_ROOT / "levels.jsonl", "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry) + "\n")
    print(f"[levels] {json.dumps(entry)}", flush=True)


def read_summary(directory: pathlib.Path) -> dict:
    return json.loads((directory / "summary.json").read_text(encoding="utf-8"))


def run_level(arguments, size: int, previous: dict, support: float, attempt: int, cleanup: dict) -> tuple:
    """One box_run of level size; returns (exit code, run directory)."""
    name = f"L{size}" + ("" if attempt == 0 else f"_retry{attempt}")
    directory = LOG_ROOT / "box_run" / name
    command = [sys.executable, "-m", "experiment.wall_crossing.box_run", "--case", case_path(size), "--name", name,
               "--device", arguments.device, "--time", str(arguments.level_time_cap), "--sample-time", "0.1",
               "--lines-time", "0.5", "--tube-time", "0.5", "--converge", "--checkpoint-time", "5",
               "--stop-speed", "1.5", "--stop-outside",
               "--initial-state", str(previous["directory"] / "final.npz"), "--initial-support", f"{support:.9g}",
               "--time-offset", f"{previous['time']:.9g}", "--wall-deadline", f"{arguments.deadline:.3f}"]
    if previous.get("integral") is not None:
        command += ["--initial-integral", repr(float(previous["integral"]))]
    for uuid in arguments.uuid:
        command += ["--uuid", uuid]
    log({"event": "level_start", "size": size, "name": name, "attempt": attempt, "support": support,
         "previous": str(previous["directory"]), "time_offset": previous["time"], "integral": previous.get("integral"),
         "command": command})
    output = open(LOG_ROOT / f"{name}.out", "w", encoding="utf-8")
    process = subprocess.Popen(command, cwd=str(REPOSITORY_ROOT), stdout=output, stderr=subprocess.STDOUT)
    started = time.time()
    while process.poll() is None:
        time.sleep(30)
        heartbeat = directory / "heartbeat.json"
        if heartbeat.exists():
            try:
                beat = json.loads(heartbeat.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if cleanup.get("pending") is not None and beat.get("level_time", 0.0) >= 1.0:
                removed = remove_level_files(cleanup["pending"])
                log({"event": "previous_level_files_deleted", "directory": str(cleanup["pending"]),
                     "removed": removed})
                cleanup["pending"] = None
            if time.time() - float(beat["unix"]) > arguments.stall_minutes * 60.0:
                process.kill()
                log({"event": "stalled_killed", "size": size, "name": name, "heartbeat": beat})
        elif time.time() - started > arguments.stall_minutes * 60.0 + 1800.0:
            process.kill()
            log({"event": "no_heartbeat_killed", "size": size, "name": name})
    output.close()
    return process.returncode, directory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--levels", required=True, help="the sizes N to run, in order (e.g. 200,280)")
    parser.add_argument("--start", required=True, help="box_run directory of the converged level to start from")
    parser.add_argument("--device", required=True)
    parser.add_argument("--uuid", action="append", default=[])
    parser.add_argument("--deadline", type=float, required=True, help="unix time the whole round must end by")
    parser.add_argument("--maximum-size", type=int, default=100000, help="the largest N that fits in memory")
    parser.add_argument("--required", default="",
                        help="sizes run whatever the estimate says (the brief's planned levels; the estimate of 3.5 "
                             "decides only the optional ones)")
    parser.add_argument("--support-factor", type=float, default=2.5)
    parser.add_argument("--level-time-cap", type=float, default=25.0)
    parser.add_argument("--stall-minutes", type=float, default=20.0)
    parser.add_argument("--keep-start", action="store_true", help="do not delete the --start level's final.npz")
    arguments = parser.parse_args()

    start_directory = (REPOSITORY_ROOT / arguments.start).resolve() if not pathlib.Path(arguments.start).is_absolute() \
        else pathlib.Path(arguments.start)
    summary = read_summary(start_directory)
    previous_size = int(round((summary["box_upper"][0] - summary["box_lower"][0]) / summary["spacing"])) - 1
    previous = {"directory": start_directory, "size": previous_size, "time": float(summary["time"]),
                "integral": summary.get("volume_constraint_integral_final"),
                "level_time": float(summary.get("level_time") or summary["time"]),
                "seconds_per_unit": 1.0 / (float(summary["steps_per_second"]) * float(summary["timestep"]))}
    log({"event": "chain_start", "levels": arguments.levels, "start": str(start_directory), "start_size": previous_size,
         "start_time": previous["time"], "deadline": arguments.deadline,
         "deadline_clock": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(arguments.deadline)),
         "maximum_size": arguments.maximum_size})
    cleanup = {"pending": None if arguments.keep_start else start_directory}
    for size in [int(item) for item in arguments.levels.split(",") if item]:
        if size > arguments.maximum_size:
            log({"event": "level_skipped", "size": size, "reason": f"N > the largest that fits ({arguments.maximum_size})"})
            break
        # the brief 3.5: the previous level's converged level time x this level's cost per unit time x 1.2
        cost = previous["seconds_per_unit"] * (fluid_count(size) / fluid_count(previous["size"])) * \
            (timestep(previous["size"]) / timestep(size))
        estimate = previous["level_time"] * cost * SAFETY
        remaining = arguments.deadline - time.time()
        log({"event": "estimate", "size": size, "seconds_per_unit_estimate": cost,
             "previous_level_time": previous["level_time"], "estimate_seconds": estimate,
             "remaining_seconds": remaining})
        required = size in [int(item) for item in arguments.required.split(",") if item]
        if estimate > remaining and not required:
            log({"event": "level_skipped", "size": size, "reason": "estimate exceeds the remaining time"})
            break
        previous_spacing = 1.0 / previous["size"]
        supports = [arguments.support_factor * previous_spacing, 4.0 * previous_spacing]
        directory = None
        for attempt, support in enumerate(supports):
            code, directory = run_level(arguments, size, previous, support, attempt, cleanup)
            done = code == 0 and (directory / "summary.json").exists() and (directory / "final.npz").exists()
            log({"event": "level_end", "size": size, "attempt": attempt, "exit_code": code, "complete": done})
            if done:
                break
            directory = None
        if directory is None:
            log({"event": "level_failed", "size": size})
            break
        summary = read_summary(directory)
        if cleanup.get("pending") is not None:        # a level shorter than 1 unit: delete now
            log({"event": "previous_level_files_deleted", "directory": str(cleanup["pending"]),
                 "removed": remove_level_files(cleanup["pending"])})
        cleanup["pending"] = directory
        previous = {"directory": directory, "size": size, "time": float(summary["time"]),
                    "integral": summary.get("volume_constraint_integral_final"),
                    "level_time": float(summary["level_time"]),
                    "seconds_per_unit": 1.0 / (float(summary["steps_per_second"]) * float(summary["timestep"]))}
        log({"event": "level_summary", "size": size, "converged": summary.get("converged"),
             "level_time": summary["level_time"], "time": summary["time"],
             "seconds_per_unit": previous["seconds_per_unit"], "integral": previous["integral"]})
    log({"event": "chain_end"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
