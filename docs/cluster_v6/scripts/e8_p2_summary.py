"""
e8_p2_summary.py — E8 phase 2, deliverable 11: the driver that runs every post-processing tool on the pulled job
directories (e8_A, e8_B) into one output directory, tolerant of missing pieces (a failed point, a missing trace or raw
directory, no v7_single pairs): each tool runs in its own process, its exit status and output are recorded in
OUT/summary_run.md, and the driver goes on with the next tool.

Tools (in order): e8_p2_tables (tables.json / md / csv), e8_scaling_figures (strong_scaling.pdf, weak_scaling.pdf),
e8_phases_table, e8_loss_sources, e8_hops, e8_bytes_check, e8_kernel_table, e8_correctness_tables, e8_single_ratio,
e8_migration_stats.

Usage:
    python docs/cluster_v6/scripts/e8_p2_summary.py --job DIR_A --job DIR_B --out-dir docs/cluster_v6/data/e8/phase2
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent


def run(script: str, arguments: list, out: pathlib.Path) -> dict:
    command = [sys.executable, str(HERE / script)] + arguments
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(HERE.parents[2]))
    seconds = time.perf_counter() - started
    log = out / "logs" / f"{pathlib.Path(script).stem}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout + "\n[stderr]\n" + completed.stderr, encoding="utf-8")
    return {"script": script, "rc": completed.returncode, "seconds": seconds,
            "last": (completed.stdout.strip().splitlines() or ["(no output)"])[-1][:200],
            "error": completed.stderr.strip().splitlines()[-1][:200] if completed.returncode and completed.stderr.strip() else ""}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--kernel-cases", default="cavity3d_n192,cavity3d_n392")
    arguments = parser.parse_args()
    out = pathlib.Path(arguments.out_dir).resolve()        # the tools run from the repository root: absolute paths
    out.mkdir(parents=True, exist_ok=True)
    jobs = [item for directory in arguments.job for item in ("--job", str(pathlib.Path(directory).resolve()))]
    plan = [("e8_p2_tables.py", jobs + ["--out-dir", str(out)]),
            ("e8_scaling_figures.py", ["--tables", str(out / "tables.json"), "--out", str(out / "figures")]),
            ("e8_phases_table.py", jobs + ["--out-dir", str(out)]),
            ("e8_loss_sources.py", jobs + ["--out-dir", str(out)]),
            ("e8_hops.py", jobs + ["--out-dir", str(out)]),
            ("e8_bytes_check.py", jobs + ["--out-dir", str(out)]),
            ("e8_kernel_table.py", jobs + ["--out-dir", str(out), "--cases", arguments.kernel_cases]),
            ("e8_correctness_tables.py", jobs + ["--out-dir", str(out)]),
            ("e8_single_ratio.py", jobs + ["--out-dir", str(out)]),
            ("e8_migration_stats.py", jobs + ["--out-dir", str(out)])]
    results = []
    for script, tool_arguments in plan:
        result = run(script, tool_arguments, out)
        results.append(result)
        print(f"[{'ok' if result['rc'] == 0 else 'FAIL rc ' + str(result['rc'])}] {script} ({result['seconds']:.1f} s): {result['last']}"
              + (f" | {result['error']}" if result["error"] else ""), flush=True)
    lines = ["| tool | rc | seconds | last line | error |", "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['script']} | {r['rc']} | {r['seconds']:.1f} | {r['last']} | {r['error']} |")
    (out / "summary_run.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    failures = [r["script"] for r in results if r["rc"]]
    print(f"[e8_p2_summary] {len(results) - len(failures)} of {len(results)} tools succeeded" + (f"; failed: {', '.join(failures)}" if failures else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
