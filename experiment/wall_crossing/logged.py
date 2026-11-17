"""
logged.py - run any command as a child process and append one record of it to logs/wall_crossing/runs.jsonl
(machine, GPU UUID, commit, case, command, start and end, status = the exit code).

    .venv/Scripts/python.exe -m experiment.wall_crossing.logged --kind bench --case cases/.../case.yaml \
        --uuid ae137c668a40f5acaab90a83f7cda175 --output logs/wall_crossing/timing/x.log [--cwd OTHER_CHECKOUT] \
        [--commit eda5b90] [--label NAME] [--env KEY=VALUE ...] -- <command and its arguments>

The command's stdout and stderr go to --output (appended) when given, else to this process's own. --cwd runs the
command in another checkout (the base commit's worktree); --commit then states that checkout's commit (default:
git of --cwd, or of this tree). The validation layer is disabled for the child (VK_LOADER_LAYERS_DISABLE) and
OPENBLAS_NUM_THREADS is 1 unless --env says otherwise. The exit code of the command is returned.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.run_log import LoggedRun, git_commit     # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--uuid", action="append", default=[], help="GPU deviceUUID (repeat for several GPUs)")
    parser.add_argument("--label", default=None)
    parser.add_argument("--output", default=None, help="file the command's output is appended to")
    parser.add_argument("--cwd", default=None, help="working directory of the command (another checkout)")
    parser.add_argument("--commit", default=None)
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = arguments.command[1:] if arguments.command[:1] == ["--"] else arguments.command
    if not command:
        parser.error("no command after --")

    working_directory = pathlib.Path(arguments.cwd).resolve() if arguments.cwd else REPOSITORY_ROOT
    environment = dict(os.environ)
    environment.setdefault("VK_LOADER_LAYERS_DISABLE", "VK_LAYER_KHRONOS_validation")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    for item in arguments.env:
        key, _, value = item.partition("=")
        environment[key] = value
    commit = arguments.commit or git_commit(working_directory)

    with LoggedRun(kind=arguments.kind, case=arguments.case, gpu_uuids=arguments.uuid,
                   command=" ".join(command), commit=commit) as run:
        if arguments.label:
            run.extra["label"] = arguments.label
        if arguments.cwd:
            run.extra["checkout"] = str(working_directory).replace("\\", "/")
        if arguments.env:
            run.extra["environment"] = list(arguments.env)
        if arguments.output:
            output_path = (REPOSITORY_ROOT / arguments.output) if not os.path.isabs(arguments.output) \
                else pathlib.Path(arguments.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            run.extra["output"] = str(output_path).replace("\\", "/")
            with open(output_path, "a", encoding="utf-8", newline="\n") as output:
                exit_code = subprocess.call(command, cwd=working_directory, env=environment, stdout=output,
                                            stderr=subprocess.STDOUT)
        else:
            exit_code = subprocess.call(command, cwd=working_directory, env=environment)
        run.extra["exit_code"] = exit_code
        run.extra["status"] = "ok" if exit_code == 0 else f"failed: exit code {exit_code}"
        print(f"[logged] {run.run_id} {arguments.kind} {arguments.label or ''}: exit code {exit_code}", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
