"""
budget_bitwise.py - does V7_DENSITY_BUDGET=1 (the diagnostic density budget) change the state the solver computes?

    .venv/Scripts/python.exe -m experiment.wall_crossing.budget_bitwise --case cases/.../case.yaml --name n250_crossing \
        --device 1 --uuid ae137c668a40f5acaab90a83f7cda175 --steps 999 --repeats 6

Runs the case --repeats times without the switch and --repeats times with it (alternating), each a canonical dump
(experiment/seam_audit/canonical_dump.py --canonical-lists, K = 1, so --steps must stay below the first defrag), and
groups the dumps into outcome classes: two dumps share a class when every field array is byte-identical. One run
is not enough beyond the step at which particles start changing voxels: from there the solver's own results fall
into a few classes from run to run (the order in which a voxel's list is filled), with or without the switch. The
switch changes nothing when every class that holds a run with it also holds a run without it (and the other way
round), or, with a single class, trivially.

Output: logs/wall_crossing/budget_bitwise/<name>_s<steps>/ (the dumps, their logs, classes.md, classes.json); one
runs.jsonl record per dump and one for the classification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.run_log import LOG_ROOT, LoggedRun                                      # noqa: E402


def dump_digest(path: pathlib.Path) -> str:
    """sha256 over every field array of a canonical dump (the meta entry left out)."""
    data = np.load(path, allow_pickle=True)
    digest = hashlib.sha256()
    for key in sorted(data.files):
        if key == "meta":
            continue
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(data[key]).tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--uuid", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=6)
    arguments = parser.parse_args()

    out = LOG_ROOT / "budget_bitwise" / f"{arguments.name}_s{arguments.steps}"
    out.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.setdefault("VK_LOADER_LAYERS_DISABLE", "VK_LAYER_KHRONOS_validation")
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    harness = REPOSITORY_ROOT / "experiment" / "seam_audit" / "canonical_dump.py"
    entries = []
    for repeat in range(arguments.repeats):
        for budget in (False, True):
            label = f"{'budget' if budget else 'plain'}_{repeat}"
            dump_path = out / f"{label}.npz"
            command = [sys.executable, str(harness), "--solver", "v7", "--case", arguments.case, "--device-map",
                       str(arguments.device), "--steps", str(arguments.steps), "--canonical-lists",
                       "--transport-extension", "--out", str(dump_path)]
            if budget:
                command += ["--env", "V7_DENSITY_BUDGET=1"]
            with LoggedRun(kind="bitwise_dump", case=arguments.case, gpu_uuids=[arguments.uuid],
                           command=" ".join(command)) as run:
                run.extra["label"] = (f"{arguments.name} K=1 {arguments.steps} steps canonical, "
                                      f"V7_DENSITY_BUDGET={int(budget)}, repeat {repeat}")
                with open(out / f"{label}.log", "w", encoding="utf-8", newline="\n") as log:
                    exit_code = subprocess.call(command, cwd=REPOSITORY_ROOT, env=environment, stdout=log,
                                                stderr=subprocess.STDOUT)
                run.extra["exit_code"] = exit_code
                run.extra["status"] = "ok" if exit_code == 0 else f"failed: exit code {exit_code}"
                if exit_code != 0:
                    print(f"[budget_bitwise] {label}: canonical_dump failed, see {out / (label + '.log')}")
                    return 1
                entries.append({"label": label, "budget": budget, "run_id": run.run_id,
                                "digest": dump_digest(dump_path)})

    with LoggedRun(kind="bitwise_classes", case=arguments.case, gpu_uuids=[arguments.uuid]) as run:
        classes: dict = {}
        for entry in entries:
            classes.setdefault(entry["digest"], []).append(entry)
        ordered = sorted(classes.values(), key=lambda members: -len(members))
        with_budget = [sum(1 for member in members if member["budget"]) for members in ordered]
        without_budget = [len(members) - count for members, count in zip(ordered, with_budget)]
        # the switch changes nothing: every class holds runs of both kinds (a single class always does here)
        shared = all(count > 0 for count in with_budget) and all(count > 0 for count in without_budget)
        lines = [f"# {arguments.name}, K = 1, {arguments.steps} steps, canonical voxel lists: "
                 f"{arguments.repeats} runs without and {arguments.repeats} with V7_DENSITY_BUDGET=1", "",
                 "| 结果类 | 不打开收支的运行 | 打开收支的运行 | 记录 |", "|---|---|---|---|"]
        for index, members in enumerate(ordered):
            lines.append(f"| {index}（{members[0]['digest'][:12]}） | {without_budget[index]} | {with_budget[index]} | "
                         + ", ".join(member["run_id"] for member in members) + " |")
        verdict = ("每个结果类里都同时有打开和不打开的运行" if shared
                   else "有的结果类只含一种运行（次数不够，或开关改变了结果）")
        lines += ["", f"{len(ordered)} 个结果类；{verdict}。记录 {run.run_id}。"]
        text = "\n".join(lines) + "\n"
        (out / "classes.md").write_text(text, encoding="utf-8", newline="\n")
        (out / "classes.json").write_text(json.dumps({
            "name": arguments.name, "case": arguments.case, "steps": arguments.steps, "repeats": arguments.repeats,
            "run_id": run.run_id, "classes": [[{"label": member["label"], "budget": member["budget"],
                                                "run_id": member["run_id"]} for member in members]
                                              for members in ordered],
            "every_class_shared": shared}, indent=1), encoding="utf-8", newline="\n")
        run.extra["label"] = f"{arguments.name} {arguments.steps} steps: outcome classes with / without the budget"
        run.extra["classes"] = len(ordered)
        run.extra["every_class_shared"] = shared
        run.extra["status"] = "ok" if shared else "failed: a class holds runs of one kind only"
        print(text)
    return 0 if shared else 1


if __name__ == "__main__":
    sys.exit(main())
