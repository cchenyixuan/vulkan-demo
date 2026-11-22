"""
log_cleanup.py - the C: cleanup of this task series' outputs (claude/wall_store_report.md, section on the cleanup).

    .venv/Scripts/python.exe -m experiment.wall_crossing.log_cleanup plan     # writes the list, deletes nothing
    .venv/Scripts/python.exe -m experiment.wall_crossing.log_cleanup delete   # deletes exactly the planned files

Scope: logs/wall_crossing/ only (git-ignored outputs of the series). Every file gets a category and a decision:
  delete  snapshots/*.npz (frames), checkpoints/*.npz except the newest of each run directory and the ones the reports
          name, every other .npz (bitwise / canonical dumps, raw watch dumps, caches) except the ones the reports name
          and the particle watch's report/ and ring files and the cavity runs' sampling points and field caches;
  keep    everything else: summaries (json, jsonl, csv, md, txt), logs, figures, samples/ (the profile history the
          analyses read), final states (final.npz, stop_state.npz), the newest checkpoint of every run (the final state
          of the runs that write no final.npz, the long 2-D cavity runs among them), case files.
The plan goes to claude/wall_store/cleanup_plan.tsv (path, bytes, category, decision) and a per-directory summary to
claude/wall_store/tables/cleanup_summary.md; delete removes the files the plan marks and nothing else.
"""
from __future__ import annotations

import collections
import pathlib
import re
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
LOG_ROOT = REPOSITORY_ROOT / "logs" / "wall_crossing"
PLAN = REPOSITORY_ROOT / "claude" / "wall_store" / "cleanup_plan.tsv"
SUMMARY = REPOSITORY_ROOT / "claude" / "wall_store" / "tables" / "cleanup_summary.md"
REPORTS = (REPOSITORY_ROOT / "claude" / "wall_crossing_report.md", REPOSITORY_ROOT / "claude" / "wall_store_report.md")


def referenced_names() -> set:
    names = set()
    for report in REPORTS:
        if report.exists():
            names |= set(re.findall(r"([A-Za-z0-9_.-]+\.npz)", report.read_text(encoding="utf-8")))
    return names


def classify(path: pathlib.Path, newest_checkpoints: set, referenced: set) -> tuple:
    relative = path.relative_to(LOG_ROOT).as_posix()
    name = path.name
    if not name.endswith(".npz"):
        return "kept: not an array dump", "keep"
    if "/snapshots/" in f"/{relative}":
        return "snapshot (frame)", "delete"
    if "/checkpoints/" in f"/{relative}":
        if path in newest_checkpoints:
            return "checkpoint, newest of its run (final state)", "keep"
        if name in referenced and name not in ("final.npz",):
            return "checkpoint named in a report", "keep"
        return "checkpoint, intermediate", "delete"
    if "/samples/" in f"/{relative}":
        return "sample (profile history)", "keep"
    if name in ("final.npz", "stop_state.npz"):
        return "final state", "keep"
    if "/report/" in f"/{relative}" or name == "ring.npz":
        return "particle watch analysis named in a report", "keep"
    if name in referenced:
        return "array named in a report", "keep"
    if relative.split("/")[0] in ("cavity", "stream_function_check"):
        return "sampling points / field cache (analysis input; the snapshots are gone)", "keep"
    return "dump (bitwise / canonical / raw watch / cache)", "delete"


def plan() -> list:
    files = [path for path in LOG_ROOT.rglob("*") if path.is_file()]
    by_directory = collections.defaultdict(list)
    for path in files:
        if path.parent.name == "checkpoints" and path.suffix == ".npz":
            by_directory[path.parent].append(path)
    newest = {sorted(paths)[-1] for paths in by_directory.values()}
    referenced = referenced_names()
    rows = []
    for path in sorted(files):
        category, decision = classify(path, newest, referenced)
        rows.append((path.relative_to(REPOSITORY_ROOT).as_posix(), path.stat().st_size, category, decision))
    return rows


def write_plan(rows: list) -> None:
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLAN.write_text("path\tbytes\tcategory\tdecision\n"
                    + "".join(f"{path}\t{size}\t{category}\t{decision}\n" for path, size, category, decision in rows
                              if decision == "delete"),
                    encoding="utf-8", newline="\n")
    totals = collections.defaultdict(lambda: [0, 0, 0, 0])
    for path, size, category, decision in rows:
        top = "/".join(path.split("/")[2:3])
        entry = totals[(top, category if decision == "delete" else "（保留）")]
        entry[0] += size
        entry[1] += 1
    lines = ["| 目录（logs/wall_crossing/） | 类别 | 文件数 | 大小 |", "|---|---|---|---|"]
    delete_total = sum(size for _, size, _, decision in rows if decision == "delete")
    keep_total = sum(size for _, size, _, decision in rows if decision == "keep")
    for (top, category), (size, count, _, _) in sorted(totals.items(), key=lambda item: -item[1][0]):
        if category == "（保留）" or size < 5e6:
            continue
        lines.append(f"| `{top}/` | {category} | {count:,} | {size / 1e9:.2f} GB |")
    lines.append(f"| 合计删除 |  | {sum(1 for row in rows if row[3] == 'delete'):,} | {delete_total / 1e9:.2f} GB |")
    lines.append(f"| 合计保留 |  | {sum(1 for row in rows if row[3] == 'keep'):,} | {keep_total / 1e9:.2f} GB |")
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"[log_cleanup] plan: delete {delete_total / 1e9:.2f} GB, keep {keep_total / 1e9:.2f} GB -> {PLAN}")


def delete() -> None:
    lines = PLAN.read_text(encoding="utf-8").splitlines()[1:]
    removed = 0
    for line in lines:
        path_text, size, category, decision = line.split("\t")
        path = REPOSITORY_ROOT / path_text
        if decision != "delete" or not path.is_file() or not path.resolve().is_relative_to(LOG_ROOT.resolve()):
            continue
        path.unlink()
        removed += int(size)
    print(f"[log_cleanup] deleted {removed / 1e9:.2f} GB")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if action == "plan":
        write_plan(plan())
    elif action == "delete":
        delete()
    else:
        sys.exit("usage: log_cleanup.py plan|delete")
