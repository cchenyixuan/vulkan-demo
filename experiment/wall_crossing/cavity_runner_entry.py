"""
cavity_runner_entry.py - experiment/validation/cavity_runner.py started on a machine that may have no git.

The runner records the repository state (git rev-parse / git status) in meta.json and segments.jsonl; on a
deployed copy without git that call raises before the run starts. This entry point replaces the runner's
git_state by one that falls back to the COMMIT file of the deployment (run_log.git_commit) and then calls the
runner's main() with the unchanged command line. With git present the runner's own function answers.

    python -m experiment.wall_crossing.cavity_runner_entry <cavity_runner arguments>
"""

from __future__ import annotations

import pathlib
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.validation import cavity_runner                 # noqa: E402
from experiment.wall_crossing.run_log import git_commit         # noqa: E402

_runner_git_state = cavity_runner.git_state


def git_state_or_commit_file(solver: str = "v6") -> dict:
    try:
        state = _runner_git_state(solver)
        if state.get("head"):
            return state
    except OSError:
        pass
    commit = git_commit(REPOSITORY_ROOT)
    return {"head": commit.replace("+dirty", ""), "dirty": commit.endswith("+dirty"), "source": "COMMIT file"}


cavity_runner.git_state = git_state_or_commit_file

if __name__ == "__main__":
    sys.exit(cavity_runner.main())
