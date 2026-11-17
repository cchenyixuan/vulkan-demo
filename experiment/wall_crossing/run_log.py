"""
run_log.py - one record per run in logs/wall_crossing/runs.jsonl (machine, GPU UUID, commit, case, command,
start and end time, status), and the run id the report cites.

    from experiment.wall_crossing.run_log import LoggedRun
    with LoggedRun(kind="single_step", case="...", gpu_uuids=[...]) as run:
        ...                         # run.run_id names the run; run.extra[...] = ... adds fields
    # the record is appended when the block ends: status "ok", or "failed: <exception>" (and re-raised)

A record is appended in one write of one line, so two processes on two GPUs can log at the same time.
Never put credentials into a record: the command line is stored as given.
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import secrets
import socket
import subprocess
import sys
from typing import Optional

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
RUN_LOG_PATH = REPOSITORY_ROOT / "logs" / "wall_crossing" / "runs.jsonl"


def git_commit(repository_root: pathlib.Path = REPOSITORY_ROOT) -> str:
    """Short commit of the tree the code runs from, with '+dirty' when tracked files differ from it; a COMMIT
    file (a deployed copy without git) takes its place when git is not available."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repository_root, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repository_root,
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        commit_file = repository_root / "COMMIT"
        if commit_file.exists():
            return commit_file.read_text(encoding="utf-8").strip()
        return "unknown"


def now_text() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def new_run_id() -> str:
    return "wc-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


def append_record(record: dict, path: pathlib.Path = RUN_LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n"
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)


class LoggedRun:
    """Context manager that appends one record to runs.jsonl when the block ends."""

    def __init__(self, kind: str, case: str, gpu_uuids: Optional[list] = None, command: Optional[str] = None,
                 machine: Optional[str] = None, commit: Optional[str] = None, run_id: Optional[str] = None,
                 path: pathlib.Path = RUN_LOG_PATH) -> None:
        self.run_id = run_id or new_run_id()
        self.path = path
        self.record = {
            "run_id": self.run_id,
            "kind": kind,
            "machine": machine or socket.gethostname(),
            "gpu_uuids": list(gpu_uuids or []),
            "commit": commit or git_commit(),
            "case": case,
            "command": command if command is not None else " ".join([os.path.basename(sys.executable)] + sys.argv),
            "started": None,
            "ended": None,
            "status": None,
        }
        self.extra: dict = {}

    def __enter__(self) -> "LoggedRun":
        self.record["started"] = now_text()
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        self.record["ended"] = now_text()
        if exception_type is None:
            self.record["status"] = self.extra.pop("status", "ok")
        else:
            self.record["status"] = f"failed: {exception_type.__name__}: {exception}"
        self.record.update(self.extra)
        append_record(self.record, self.path)
        return False


def vulkan_device_uuid(context) -> str:
    """deviceUUID (hex, no dashes) of a VulkanContextV7's physical device (cavity_runner.device_uuid's query)."""
    from experiment.validation.cavity_runner import device_uuid
    return device_uuid(context)
