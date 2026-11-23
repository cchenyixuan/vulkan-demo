"""
lean_bits.py - the bit check of the EXPERIMENTAL lean fluid-segment builds and the near-wall lanes builds (claude/
wall_store_report.md section 9): every build evaluates the fused pass and force once on the same frozen state
(frozen_evaluate.py --sort-voxel-lists --split: bit-reproducible, the passes recorded as the step records them), and
its outputs are compared bit for bit with the default build's.

    .venv/Scripts/python.exe -m experiment.wall_crossing.lean_bits run --label n500_crossing \
        --case cases/..._crossing/case.yaml --state logs/wall_store/round3/frozen/state_n500_crossing.npz --device 1 \
        --builds base,base_again,zero_z,distance_squared,kernel_range,all,factored,all_factored,lanes2,lanes4,lanes8
    .venv/Scripts/python.exe -m experiment.wall_crossing.lean_bits table

A build is a set of switches (BUILDS). Output: logs/wall_store/round3/bits/<label>_<build>.npz (deleted by `table`
after it has read them unless --keep) and the table claude/wall_store/tables/round3_lean_bits.md with, per field, the
rows whose bits differ from base and the largest difference relative to the field's rms over all rows.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = REPOSITORY_ROOT / "logs" / "wall_store" / "round3" / "bits"
TABLES = REPOSITORY_ROOT / "claude" / "wall_store" / "tables"
FIELDS = ("density_rate", "acceleration", "shift", "correction_inverse", "density_gradient_kernel_sum")
BUILDS = {
    "base": {}, "base_again": {},
    "zero_z": {"V7_FLUID_SEGMENT_ZERO_Z": "1"},
    "distance_squared": {"V7_FLUID_SEGMENT_DISTANCE_SQUARED": "1"},
    "kernel_range": {"V7_FLUID_SEGMENT_KERNEL_RANGE": "1"},
    "all": {"V7_FLUID_SEGMENT_ZERO_Z": "1", "V7_FLUID_SEGMENT_DISTANCE_SQUARED": "1", "V7_FLUID_SEGMENT_KERNEL_RANGE": "1"},
    "zero_z_distance_squared": {"V7_FLUID_SEGMENT_ZERO_Z": "1", "V7_FLUID_SEGMENT_DISTANCE_SQUARED": "1"},
    "zero_z_kernel_range": {"V7_FLUID_SEGMENT_ZERO_Z": "1", "V7_FLUID_SEGMENT_KERNEL_RANGE": "1"},
    "distance_squared_kernel_range": {"V7_FLUID_SEGMENT_DISTANCE_SQUARED": "1", "V7_FLUID_SEGMENT_KERNEL_RANGE": "1"},
    "factored": {"V7_FLUID_SEGMENT_FORCE_FACTORED": "1"},
    "all_factored": {"V7_FLUID_SEGMENT_ZERO_Z": "1", "V7_FLUID_SEGMENT_DISTANCE_SQUARED": "1", "V7_FLUID_SEGMENT_KERNEL_RANGE": "1",
                     "V7_FLUID_SEGMENT_FORCE_FACTORED": "1"},
    "lanes2": {"V7_WALL_CROSSING_NEAR_WALL_LANES": "2"},
    "lanes4": {"V7_WALL_CROSSING_NEAR_WALL_LANES": "4"},
    "lanes8": {"V7_WALL_CROSSING_NEAR_WALL_LANES": "8"},
}


def run(arguments) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for build in arguments.builds.split(","):
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(("V7_FLUID_SEGMENT_", "V7_WALL_CROSSING_NEAR_WALL_LANES"))}
        environment.update(BUILDS[build])
        environment["OPENBLAS_NUM_THREADS"] = "1"
        out = OUT / f"{arguments.label}_{build}.npz"
        command = [sys.executable, str(REPOSITORY_ROOT / "experiment" / "wall_crossing" / "frozen_evaluate.py"),
                   "--repo", str(REPOSITORY_ROOT), "--case", arguments.case, "--state", arguments.state,
                   "--device", str(arguments.device), "--out", str(out), "--sort-voxel-lists", "--split"]
        result = subprocess.run(command, env=environment, capture_output=True, text=True)
        tail = (result.stdout + result.stderr).strip().splitlines()[-1:]
        print(f"[lean_bits] {arguments.label} {build}: exit {result.returncode} {tail}", flush=True)
        if result.returncode != 0:
            print(result.stdout[-3000:], result.stderr[-3000:])
            return result.returncode
    return 0


def compare(base: dict, other: dict) -> dict:
    result = {}
    for field in FIELDS:
        first, second = np.asarray(base[field]), np.asarray(other[field])
        if first.dtype == np.float32:
            differs = (first.view(np.uint32) != second.view(np.uint32)).reshape(first.shape[0], -1)
        else:
            differs = (first != second).reshape(first.shape[0], -1)
        difference = np.abs(first.astype(np.float64) - second.astype(np.float64))
        scale = float(np.sqrt(np.mean(first.astype(np.float64) ** 2))) or 1.0
        result[field] = {"rows": int(differs.any(axis=1).sum()), "of": int(first.shape[0]),
                         "largest_over_rms": float(difference.max() / scale)}
    return result


def table(arguments) -> int:
    """Compare the evaluations present with their label's base; merged into the summary of earlier calls (the
    evaluations of an earlier call are deleted after it), the table written from the merged summary."""
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    lines = []
    labels = sorted({path.name.rsplit("_", 1)[0] for path in OUT.glob("*_base.npz")})
    lines.append("| 算例 | 版本 | " + " | ".join(FIELDS) + " |")
    lines.append("|---|---|" + "---|" * len(FIELDS))
    for label in labels:
        base = dict(np.load(OUT / f"{label}_base.npz"))
        for build in BUILDS:
            path = OUT / f"{label}_{build}.npz"
            if build == "base" or not path.exists():
                continue
            summary.setdefault(label, {})[build] = compare(base, dict(np.load(path)))
    for label in sorted(summary):
        for build in BUILDS:
            result = summary[label].get(build)
            if result is None:
                continue
            cells = [("逐位相同" if entry["rows"] == 0
                      else f"{entry['rows']:,} / {entry['of']:,} 行；最大差 {entry['largest_over_rms']:.1e} 均方根")
                     for entry in result.values()]
            lines.append(f"| {label} | {build} | " + " | ".join(cells) + " |")
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "round3_lean_bits.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\n".join(lines))
    if not arguments.keep:
        for path in OUT.glob("*.npz"):
            path.unlink()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--label", required=True)
    run_parser.add_argument("--case", required=True)
    run_parser.add_argument("--state", required=True)
    run_parser.add_argument("--device", type=int, default=1)
    run_parser.add_argument("--builds", required=True)
    table_parser = sub.add_parser("table")
    table_parser.add_argument("--keep", action="store_true")
    arguments = parser.parse_args()
    return run(arguments) if arguments.action == "run" else table(arguments)


if __name__ == "__main__":
    sys.exit(main())
