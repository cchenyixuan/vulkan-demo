"""
outcome_classes.py - does a case compute the states a reference computes, bit for bit?

    # against dumps that exist (made by the code before a change)
    .venv/Scripts/python.exe -m experiment.wall_crossing.outcome_classes --name explicit_full_against_old_default \
        --case cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025_crossing_continuity_full/case.yaml \
        --reference-dumps logs/wall_crossing/budget_bitwise/before_default_change_s700 --reference-pattern "plain_*.npz" \
        --device 1 --uuid ae137c668a40f5acaab90a83f7cda175 --steps 700 --repeats 6
    # against another case, dumped now (alternating with the case)
    .venv/Scripts/python.exe -m experiment.wall_crossing.outcome_classes --name default_against_explicit_normal \
        --case cases/.../case.yaml --reference-case cases/.../case.yaml --device 1 --uuid ... --steps 700 --repeats 4

Every run is a canonical dump (experiment/seam_audit/canonical_dump.py --canonical-lists, K = 1, so --steps must
stay below the first defrag). Two dumps share an outcome class when every field array is byte-identical. One
pair of runs is not enough for wall_boundary crossing beyond the step at which particles start changing voxels:
from there its results fall into a few classes from run to run (budget_bitwise.py), so the question is put to
the classes: the case computes what the reference computes when every class that holds a run of the case also
holds a reference run. (Whether every reference class is reached by the case depends on the number of runs; it
is reported, not required.)

Output: logs/wall_crossing/outcome_classes/<name>_s<steps>/ (the dumps made here, their logs, classes.md,
classes.json); one runs.jsonl record per dump and one for the classification.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.budget_bitwise import dump_digest                                      # noqa: E402
from experiment.wall_crossing.run_log import LOG_ROOT, LoggedRun                                      # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--case", required=True, help="the case in question")
    parser.add_argument("--reference-dumps", default=None, help="directory of existing canonical dumps")
    parser.add_argument("--reference-pattern", default="*.npz")
    parser.add_argument("--reference-case", default=None, help="a case to dump now as the reference")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--uuid", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=6)
    arguments = parser.parse_args()
    if (arguments.reference_dumps is None) == (arguments.reference_case is None):
        parser.error("give exactly one of --reference-dumps and --reference-case")

    out = LOG_ROOT / "outcome_classes" / f"{arguments.name}_s{arguments.steps}"
    out.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.setdefault("VK_LOADER_LAYERS_DISABLE", "VK_LAYER_KHRONOS_validation")
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    harness = REPOSITORY_ROOT / "experiment" / "seam_audit" / "canonical_dump.py"

    entries = []
    if arguments.reference_dumps is not None:
        reference_directory = (REPOSITORY_ROOT / arguments.reference_dumps).resolve()
        reference_paths = sorted(reference_directory.glob(arguments.reference_pattern))
        if not reference_paths:
            parser.error(f"no dump matches {arguments.reference_pattern} in {reference_directory}")
        for path in reference_paths:
            entries.append({"label": f"reference:{path.name}", "reference": True, "run_id": None,
                            "digest": dump_digest(path)})
        reference_text = f"{len(reference_paths)} existing dumps {arguments.reference_dumps}/{arguments.reference_pattern}"
        kinds = (("case", arguments.case, False),)
    else:
        reference_text = f"{arguments.repeats} runs of {arguments.reference_case}"
        kinds = (("reference", arguments.reference_case, True), ("case", arguments.case, False))

    for repeat in range(arguments.repeats):
        for kind, case, is_reference in kinds:
            label = f"{kind}_{repeat}"
            dump_path = out / f"{label}.npz"
            command = [sys.executable, str(harness), "--solver", "v7", "--case", case, "--device-map",
                       str(arguments.device), "--steps", str(arguments.steps), "--canonical-lists",
                       "--transport-extension", "--out", str(dump_path)]
            with LoggedRun(kind="bitwise_dump", case=case, gpu_uuids=[arguments.uuid],
                           command=" ".join(command)) as run:
                run.extra["label"] = f"{arguments.name}: {kind} K=1 {arguments.steps} steps canonical, repeat {repeat}"
                with open(out / f"{label}.log", "w", encoding="utf-8", newline="\n") as log:
                    exit_code = subprocess.call(command, cwd=REPOSITORY_ROOT, env=environment, stdout=log,
                                                stderr=subprocess.STDOUT)
                run.extra["exit_code"] = exit_code
                run.extra["status"] = "ok" if exit_code == 0 else f"failed: exit code {exit_code}"
                if exit_code != 0:
                    print(f"[outcome_classes] {label}: canonical_dump failed, see {out / (label + '.log')}")
                    return 1
                entries.append({"label": label, "reference": is_reference, "run_id": run.run_id,
                                "digest": dump_digest(dump_path)})

    with LoggedRun(kind="bitwise_classes", case=arguments.case, gpu_uuids=[arguments.uuid]) as run:
        classes: dict = {}
        for entry in entries:
            classes.setdefault(entry["digest"], []).append(entry)
        ordered = sorted(classes.values(), key=lambda members: -len(members))
        of_reference = [sum(1 for member in members if member["reference"]) for members in ordered]
        of_case = [len(members) - count for members, count in zip(ordered, of_reference)]
        # the case computes what the reference computes: no class holds runs of the case only
        covered = all(reference_count > 0 for reference_count, case_count in zip(of_reference, of_case) if case_count)
        every_reference_class_reached = all(case_count > 0 for case_count in of_case)
        lines = [f"# {arguments.name}, K = 1, {arguments.steps} steps, canonical voxel lists: {arguments.repeats} runs of "
                 f"{arguments.case} against {reference_text}", "",
                 "| 结果类 | 参照的运行 | 本算例的运行 | 记录（本次产生的转储） |", "|---|---|---|---|"]
        for index, members in enumerate(ordered):
            lines.append(f"| {index}（{members[0]['digest'][:12]}） | {of_reference[index]} | {of_case[index]} | "
                         + ", ".join(member["run_id"] for member in members if member["run_id"]) + " |")
        verdict = ("本算例的每次运行都落在含参照运行的结果类里（逐位相同）" if covered
                   else "本算例有运行落在不含参照运行的结果类里（结果不同，或次数不够）")
        if covered and not every_reference_class_reached:
            verdict += "；参照的结果类里有本算例这几次没有落到的"
        lines += ["", f"{len(ordered)} 个结果类；{verdict}。记录 {run.run_id}。"]
        text = "\n".join(lines) + "\n"
        (out / "classes.md").write_text(text, encoding="utf-8", newline="\n")
        (out / "classes.json").write_text(json.dumps({
            "name": arguments.name, "case": arguments.case, "reference": reference_text, "steps": arguments.steps,
            "repeats": arguments.repeats, "run_id": run.run_id,
            "classes": [{"digest": members[0]["digest"], "reference_runs": of_reference[index],
                         "case_runs": of_case[index],
                         "members": [{"label": member["label"], "reference": member["reference"],
                                      "run_id": member["run_id"]} for member in members]}
                        for index, members in enumerate(ordered)],
            "case_within_reference_classes": covered,
            "every_reference_class_reached": every_reference_class_reached}, indent=1), encoding="utf-8", newline="\n")
        run.extra["label"] = f"{arguments.name} {arguments.steps} steps: outcome classes of the case against the reference"
        run.extra["classes"] = len(ordered)
        run.extra["case_within_reference_classes"] = covered
        run.extra["status"] = "ok" if covered else "failed: a class holds runs of the case only"
        print(text)
    return 0 if covered else 1


if __name__ == "__main__":
    sys.exit(main())
