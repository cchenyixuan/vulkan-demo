"""
kernel_timing.py - per-kernel GPU times of the v7 single-GPU step (experiment/v7/_run_v7_single_bench.py: one
command buffer per step, GPU timestamps after every kernel) for a list of cases, interleaved over several
rounds, on one GPU that runs nothing else.

    .venv/Scripts/python.exe -m experiment.wall_crossing.kernel_timing --device 1 \
        --uuid ae137c668a40f5acaab90a83f7cda175 --out logs/wall_crossing/timing/n1000 --rounds 3 \
        --entry branch_simple=cases/lid_driven_cavity_2d_n1000_xi0p001_eps0p0025/case.yaml \
        --entry base_simple=@C:/.../vulkan-demo-wall-crossing-base:cases/lid_driven_cavity_2d_n1000_xi0p001_eps0p0025/case.yaml

An entry is LABEL=CASE (this checkout) or LABEL=@CHECKOUT:CASE (the bench of another checkout, e.g. the base
commit's worktree, run from there). Every round runs every entry once, in the given order, so slow drifts of the
card (temperature, clocks) hit all entries alike. Each run: --max-steps steps, the first --warmup discarded
(they hold the first defrags, after which the particle order is the voxel order); the per-frame CSV is kept.

Output: <out>/<label>_r<round>.csv and .log, <out>/summary.json and summary.md with, per label and kernel, the
median over the rounds of the per-run mean time (and the spread over the rounds). One runs.jsonl record per run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import statistics
import subprocess
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.run_log import LoggedRun, git_commit     # noqa: E402

KERNEL_COLUMNS = ("predict_us", "update_voxel_us", "correction_density_us", "density_copy_us",
                  "wall_extrapolate_us", "force_us", "step_total_us", "frame_total_us")


def parse_entry(text: str) -> tuple:
    label, _, target = text.partition("=")
    if not label or not target:
        raise argparse.ArgumentTypeError(f"entry {text!r}: expected LABEL=CASE or LABEL=@CHECKOUT:CASE")
    if target.startswith("@"):
        checkout, separator, case = target[1:].rpartition(":")
        if not separator or not checkout:
            raise argparse.ArgumentTypeError(f"entry {text!r}: expected LABEL=@CHECKOUT:CASE")
        return label, pathlib.Path(checkout).resolve(), case
    return label, REPOSITORY_ROOT, target


def column_means(csv_path: pathlib.Path) -> dict:
    """Mean of every kernel column over the rows of one run's CSV (rows after the warmup only are written)."""
    sums: dict = {}
    counts: dict = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for name in KERNEL_COLUMNS:
                value = row.get(name, "")
                if value not in ("", None):
                    sums[name] = sums.get(name, 0.0) + float(value)
                    counts[name] = counts.get(name, 0) + 1
    return {name: sums[name] / counts[name] for name in sums}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--uuid", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--entry", action="append", type=parse_entry, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--warmup", type=int, default=3000)
    arguments = parser.parse_args()

    out_directory = (REPOSITORY_ROOT / arguments.out).resolve()
    out_directory.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["VK_LOADER_LAYERS_DISABLE"] = "VK_LAYER_KHRONOS_validation"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"

    runs: dict = {label: [] for label, _, _ in arguments.entry}
    for round_index in range(arguments.rounds):
        for label, checkout, case in arguments.entry:
            csv_path = out_directory / f"{label}_r{round_index}.csv"
            log_path = out_directory / f"{label}_r{round_index}.log"
            command = [sys.executable, "experiment/v7/_run_v7_single_bench.py", "--case", case,
                       "--device", str(arguments.device), "--max-steps", str(arguments.max_steps),
                       "--warmup", str(arguments.warmup), "--bench-window", str(arguments.max_steps),
                       "--bench-csv", str(csv_path)]
            with LoggedRun(kind="kernel_timing", case=case, gpu_uuids=[arguments.uuid],
                           command=" ".join(command[1:]), commit=git_commit(checkout)) as run:
                run.extra["label"] = f"{label} round {round_index}"
                if checkout != REPOSITORY_ROOT:
                    run.extra["checkout"] = str(checkout).replace("\\", "/")
                with open(log_path, "w", encoding="utf-8", newline="\n") as log:
                    exit_code = subprocess.call(command, cwd=checkout, env=environment, stdout=log,
                                                stderr=subprocess.STDOUT)
                run.extra["exit_code"] = exit_code
                if exit_code != 0:
                    run.extra["status"] = f"failed: exit code {exit_code}"
                    print(f"[kernel_timing] {label} round {round_index}: FAILED (exit code {exit_code}), see {log_path}")
                    continue
                means = column_means(csv_path)
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
                drift = "WARN: alive drift" in log_text
                run.extra["means_us"] = {name: round(value, 3) for name, value in means.items()}
                run.extra["alive_drift"] = drift
                runs[label].append({"round": round_index, "run_id": run.run_id, "means_us": means,
                                    "alive_drift": drift})
                print(f"[kernel_timing] {label} round {round_index} ({run.run_id}): step "
                      f"{means.get('step_total_us', float('nan')):.1f} us, fused "
                      f"{means.get('correction_density_us', float('nan')):.1f}, force "
                      f"{means.get('force_us', float('nan')):.1f}", flush=True)

    summary = {"device": arguments.device, "uuid": arguments.uuid, "max_steps": arguments.max_steps,
               "warmup": arguments.warmup, "rounds": arguments.rounds, "labels": {}}
    lines = ["| label | " + " | ".join(name[:-3] for name in KERNEL_COLUMNS) + " | runs |",
             "|---|" + "---|" * (len(KERNEL_COLUMNS) + 1)]
    for label, _, case in arguments.entry:
        entries = runs[label]
        row = {"case": case, "run_ids": [entry["run_id"] for entry in entries], "kernels": {}}
        cells = []
        for name in KERNEL_COLUMNS:
            values = [entry["means_us"][name] for entry in entries if name in entry["means_us"]]
            if values:
                row["kernels"][name] = {"median_us": statistics.median(values), "minimum_us": min(values),
                                        "maximum_us": max(values)}
                cells.append(f"{statistics.median(values):.1f} ({min(values):.1f}-{max(values):.1f})")
            else:
                cells.append("-")
        summary["labels"][label] = row
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {', '.join(row['run_ids'])} |")
    (out_directory / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    table = ("Per-kernel GPU time in microseconds per step: median over the rounds of the per-run mean "
             f"(minimum-maximum); {arguments.max_steps} steps per run, the first {arguments.warmup} discarded.\n\n"
             + "\n".join(lines) + "\n")
    (out_directory / "summary.md").write_text(table, encoding="utf-8", newline="\n")
    print("\n" + table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
