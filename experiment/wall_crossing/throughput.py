"""
throughput.py - throughput of a chain of K slabs for several cases, measured with the repository's chain bench
(experiment/v7/_run_v7_chain_bench.py: pipelined at depth 2, equal slab weights, its own conservation, stamp,
overflow and seam checks), the cases interleaved round by round.

    .venv/Scripts/python.exe -m experiment.wall_crossing.throughput --device-map 0,1 \
        --uuid fb83f0dafd9e1d38dc05e7ae643e22a8 --uuid ae137c668a40f5acaab90a83f7cda175 \
        --max-steps 20000 --warmup 5000 --rounds 3 \
        --entry n1000_simple=cases/lid_driven_cavity_2d_n1000_xi0p001_eps0p0025/case.yaml \
        --entry n1000_crossing=cases/lid_driven_cavity_2d_n1000_xi0p001_eps0p0025_crossing/case.yaml

The devices must be idle (the tool does not check). Per entry: logs/wall_crossing/throughput/<label>_k<K>.json
(steady steps per second of every round = the bench's STEADY line, their median, the particle drift, the bench's
verdict) and the bench output of every round beside it; one runs.jsonl record per bench run.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys

import yaml

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.run_log import LOG_ROOT, LoggedRun     # noqa: E402


def parse_entry(text: str) -> tuple:
    label, _, case = text.partition("=")
    if not label or not case:
        raise argparse.ArgumentTypeError(f"entry {text!r}: expected LABEL=CASE")
    return label, case


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device-map", required=True)
    parser.add_argument("--uuid", action="append", required=True)
    parser.add_argument("--entry", action="append", type=parse_entry, required=True)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--warmup", type=int, default=5000)
    parser.add_argument("--rounds", type=int, default=3)
    arguments = parser.parse_args()

    slabs = len(arguments.device_map.split(","))
    out = LOG_ROOT / "throughput"
    out.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", PYTHONIOENCODING="utf-8",
                       VK_LOADER_LAYERS_DISABLE="VK_LAYER_KHRONOS_validation")
    results = {label: {"label": label, "case": case, "slabs": slabs, "device_map": arguments.device_map,
                       "uuids": arguments.uuid, "max_steps": arguments.max_steps, "warmup": arguments.warmup,
                       "wall_boundary": None, "steps_per_second_trials": [], "drifts": [], "verdicts": [],
                       "run_ids": []}
               for label, case in arguments.entry}
    for label, case in arguments.entry:
        numerics = yaml.safe_load((REPOSITORY_ROOT / case).read_text(encoding="utf-8")).get("numerics", {})
        crossing = numerics.get("wall_crossing") or {}
        results[label]["wall_boundary"] = numerics.get("wall_boundary", "simple") + (
            f" ({crossing.get('continuity', 'extrapolate')})" if numerics.get("wall_boundary") == "crossing" else "")

    for round_index in range(arguments.rounds):
        for label, case in arguments.entry:
            command = [sys.executable, "experiment/v7/_run_v7_chain_bench.py", "--case", case, "--weights",
                       ",".join(["1"] * slabs), "--device-map", arguments.device_map, "--max-steps",
                       str(arguments.max_steps), "--warmup", str(arguments.warmup)]
            log_path = out / f"{label}_k{slabs}_r{round_index}.log"
            with LoggedRun(kind="throughput", case=case, gpu_uuids=arguments.uuid,
                           command=" ".join(command[1:])) as run:
                run.extra["label"] = f"{label} K={slabs} round {round_index}"
                with open(log_path, "w", encoding="utf-8", newline="\n") as log:
                    exit_code = subprocess.call(command, cwd=REPOSITORY_ROOT, env=environment, stdout=log,
                                                stderr=subprocess.STDOUT)
                text = log_path.read_text(encoding="utf-8", errors="replace")
                steady = re.search(r"STEADY \(post-warmup \d+\): \d+ steps in [\d.]+s = ([\d.]+) fps", text)
                drift = re.search(r"drift=(-?\d+)", text)
                run.extra["exit_code"] = exit_code
                run.extra["steady_steps_per_second"] = float(steady.group(1)) if steady else None
                run.extra["drift"] = int(drift.group(1)) if drift else None
                run.extra["status"] = "ok" if exit_code == 0 and steady else f"failed: exit code {exit_code}"
                entry = results[label]
                entry["run_ids"].append(run.run_id)
                entry["verdicts"].append("ok" if exit_code == 0 else "bench validation failed")
                if steady:
                    entry["steps_per_second_trials"].append(float(steady.group(1)))
                entry["drifts"].append(int(drift.group(1)) if drift else None)
            print(f"[throughput] round {round_index} {label} K={slabs}: "
                  f"{run.extra['steady_steps_per_second']} steps/s, drift {run.extra['drift']}, exit code "
                  f"{exit_code}", flush=True)
    for label, entry in results.items():
        trials = entry["steps_per_second_trials"]
        entry["steps_per_second_median"] = statistics.median(trials) if trials else float("nan")
        entry["drift"] = "0" if entry["drifts"] and all(value == 0 for value in entry["drifts"]) \
            else str(entry["drifts"])
        (out / f"{label}_k{slabs}.json").write_text(json.dumps(entry, indent=1), encoding="utf-8", newline="\n")
        print(f"[throughput] {label} K={slabs}: median {entry['steps_per_second_median']:.1f} steps/s of "
              f"{[round(value, 1) for value in trials]}, drift {entry['drift']}")
    return 0 if all(all(verdict == "ok" for verdict in entry["verdicts"]) for entry in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
