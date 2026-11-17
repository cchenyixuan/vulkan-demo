"""
bitwise_check.py - are the existing wall modes bit-for-bit what they were at the base commit?

For each case: experiment/seam_audit/canonical_dump.py runs it from its initial state for N steps with the
solver of the base checkout and with this checkout's, twice each, and the dumps (every per-particle field of
every alive particle, sorted by a global particle id) are compared bit for bit:
    base  vs base'      the base build reproduces itself over this window (else the window proves nothing)
    this  vs this'      likewise for this build
    base  vs this       the answer
canonical_dump's own rules apply (its docstring): --canonical-lists sorts the bootstrap voxel lists and skips the
bootstrap defrag, so such a run must end before the first defrag (N < defrag cadence); without it the run goes
through defrags and its voxel lists are appended in thread arrival order, so only the self comparisons tell
whether a longer window is reproducible at all.

    .venv/Scripts/python.exe -m experiment.wall_crossing.bitwise_check --base C:/.../vulkan-demo-wall-crossing-base \
        --device-map 0 --uuid fb83f0dafd9e1d38dc05e7ae643e22a8 --out logs/wall_crossing/bitwise \
        --entry simple_n250=cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025/case.yaml:999:canonical \
        --entry adami_n250=cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025_adami/case.yaml:200:canonical

An entry is LABEL=CASE:STEPS:canonical or LABEL=CASE:STEPS:plain. K = the number of --device-map entries
(K > 1 adds --transport-extension, which canonical_dump needs to follow particles across a cut). Every dump
and every comparison is one runs.jsonl record; summary.json / summary.md hold the verdicts.
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

from experiment.wall_crossing.run_log import LoggedRun, git_commit     # noqa: E402


def parse_entry(text: str) -> tuple:
    label, _, target = text.partition("=")
    parts = target.rsplit(":", 2)
    if not label or len(parts) != 3 or parts[2] not in ("canonical", "plain"):
        raise argparse.ArgumentTypeError(f"entry {text!r}: expected LABEL=CASE:STEPS:canonical|plain")
    return label, parts[0], int(parts[1]), parts[2] == "canonical"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="checkout of the base commit")
    parser.add_argument("--device-map", required=True, help="one v7 device index per slab, e.g. 0 or 0,1")
    parser.add_argument("--uuid", action="append", required=True, help="deviceUUID of each device (repeat)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--entry", action="append", type=parse_entry, required=True)
    arguments = parser.parse_args()

    base = pathlib.Path(arguments.base).resolve()
    out = (REPOSITORY_ROOT / arguments.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    slabs = len(arguments.device_map.split(","))
    environment = dict(os.environ)
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    harness = str(REPOSITORY_ROOT / "experiment" / "seam_audit" / "canonical_dump.py")
    builds = {"base": base, "this": REPOSITORY_ROOT}
    commits = {name: git_commit(path) for name, path in builds.items()}

    summary = {"base": str(base).replace("\\", "/"), "commits": commits, "device_map": arguments.device_map,
               "uuids": arguments.uuid, "slabs": slabs, "entries": {}}
    lines = ["| 算例 | K | 步数 | 体素表 | base 自复现 | 本分支自复现 | base 对 本分支 | 记录 |", "|---|---|---|---|---|---|---|---|"]
    for label, case, steps, canonical in arguments.entry:
        dumps = {}
        run_ids = []
        for build, repeat in (("base", 1), ("this", 1), ("base", 2), ("this", 2)):
            dump_path = out / f"{label}_k{slabs}_{build}{repeat}.npz"
            log_path = out / f"{label}_k{slabs}_{build}{repeat}.log"
            command = [sys.executable, harness, "--solver", "v7", "--repo", str(builds[build]), "--case", case,
                       "--device-map", arguments.device_map, "--steps", str(steps), "--out", str(dump_path)]
            if canonical:
                command.append("--canonical-lists")
            if slabs > 1:
                command.append("--transport-extension")
            with LoggedRun(kind="bitwise_dump", case=case, gpu_uuids=arguments.uuid,
                           command=" ".join(command[1:]), commit=commits[build]) as run:
                run.extra["label"] = f"{label} K={slabs} {build}{repeat} ({steps} steps, "\
                                     f"{'canonical lists' if canonical else 'plain'})"
                with open(log_path, "w", encoding="utf-8", newline="\n") as log:
                    exit_code = subprocess.call(command, cwd=REPOSITORY_ROOT, env=environment, stdout=log,
                                                stderr=subprocess.STDOUT)
                run.extra["exit_code"] = exit_code
                run.extra["status"] = "ok" if exit_code == 0 else f"failed: exit code {exit_code}"
                run_ids.append(run.run_id)
            dumps[f"{build}{repeat}"] = (dump_path, exit_code)
            print(f"[bitwise_check] {label} K={slabs} {build}{repeat}: dump exit code {exit_code}", flush=True)

        verdicts = {}
        for name, first, second in (("base_self", "base1", "base2"), ("this_self", "this1", "this2"),
                                    ("base_vs_this", "base1", "this1")):
            if dumps[first][1] != 0 or dumps[second][1] != 0:
                verdicts[name] = "dump failed"
                continue
            command = [sys.executable, "-m", "experiment.seam_audit.canonical_dump", "--compare",
                       str(dumps[first][0]), str(dumps[second][0])]
            with LoggedRun(kind="bitwise_compare", case=case, gpu_uuids=[],
                           command=" ".join(command[1:])) as run:
                completed = subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, capture_output=True,
                                           text=True, encoding="utf-8", errors="replace")
                (out / f"{label}_k{slabs}_{name}.compare.log").write_text(
                    completed.stdout + completed.stderr, encoding="utf-8", newline="\n")
                text = completed.stdout + completed.stderr
                if "NOT COMPARABLE" in text:
                    verdict = "NOT COMPARABLE"
                elif "NOT bit-identical" in text:
                    verdict = "NOT bit-identical"
                elif "BIT-IDENTICAL" in text and completed.returncode == 0:
                    verdict = "BIT-IDENTICAL"
                else:
                    verdict = f"unclear (exit code {completed.returncode})"
                verdicts[name] = verdict
                run.extra["label"] = f"{label} K={slabs} {name}"
                run.extra["verdict"] = verdict
                run.extra["status"] = "ok" if verdict == "BIT-IDENTICAL" else f"failed: {verdict}"
                run_ids.append(run.run_id)
            print(f"[bitwise_check] {label} K={slabs} {name}: {verdicts[name]}", flush=True)
        summary["entries"][label] = {"case": case, "steps": steps, "canonical_lists": canonical,
                                     "verdicts": verdicts, "run_ids": run_ids}
        lines.append(f"| {label} | {slabs} | {steps} | {'canonical' if canonical else 'plain'} | "
                     f"{verdicts['base_self']} | {verdicts['this_self']} | {verdicts['base_vs_this']} | "
                     f"{run_ids[0]} … {run_ids[-1]} |")
    (out / f"summary_k{slabs}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    table = "\n".join(lines) + "\n"
    (out / f"summary_k{slabs}.md").write_text(table, encoding="utf-8", newline="\n")
    print("\n" + table)
    ok = all(entry["verdicts"]["base_vs_this"] == "BIT-IDENTICAL" for entry in summary["entries"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
