"""
dual_timing.py - report section 22: phase times of a chain of slabs (experiment/v7/_run_v7_chain_bench.py
--anatomy: GPU timestamps of one whole frame per slab at every defrag, phase A / B / C, the kernels inside them,
the B -> C gap and the transfer queue's DMA segments) and the steady steps per second, for a list of entries
interleaved over several rounds.

    .venv/Scripts/python.exe -m experiment.wall_crossing.dual_timing --device-map 0,1 \
        --uuid fb83f0dafd9e1d38dc05e7ae643e22a8 --uuid ae137c668a40f5acaab90a83f7cda175 \
        --out logs/wall_crossing/timing/s22_3d_k2 --rounds 3 --max-steps 6000 --warmup 2000 \
        --entry "base_simple=@C:/.../vulkan-demo-wall-crossing-base:cases/cavity3d_1m/case.yaml" \
        --entry "split_mode1=cases/cavity3d_1m_crossing_split/case.yaml" \
        --entry "split_mode2=cases/cavity3d_1m_crossing_split/case.yaml;V7_WALL_CROSSING_SPLIT_BAND_MODE=2"

An entry is LABEL=CASE or LABEL=@CHECKOUT:CASE (that checkout's bench, run from there), optionally followed by
;NAME=VALUE environment settings for that entry only. Equal slab weights (--weights 1,...), depth 2. The anatomy
lines of frames after --warmup are averaged per slab and key (the bench prints one per defrag: a frame that the
pipeline drained before, so phase times are those of a frame running alone on its queue, without the overlap of
a pipelined frame; the steady rate comes from the pipelined run).

Output: <out>/<label>_r<round>.log, <out>/summary.json and summary.md (per label, slab and key: the median over the
rounds of the per-run mean, with the spread over the rounds); one runs.jsonl record per run.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.run_log import LoggedRun, git_commit     # noqa: E402

ANATOMY_PATTERN = re.compile(r"^\[anatomy\] f(\d+) s(\d+): (.*)$")
STEADY_PATTERN = re.compile(r"STEADY \(post-warmup \d+\): (\d+) steps in ([0-9.]+)s = ([0-9.]+) fps")
SUMMARY_KEYS = ("phase_a", "phase_b", "phase_c", "b_to_c_gap", "c_to_a_gap", "a_to_b_gap",
                "correction_density_interior", "correction_density_interior_split_interior",
                "correction_density_interior_split_near_wall", "force_deep_interior",
                "force_deep_interior_split_interior", "force_deep_interior_split_near_wall",
                "correction_density_boundary", "correction_density_boundary_split_interior",
                "correction_density_boundary_split_near_wall", "density_copy", "force",
                "force_boundary_split_interior", "force_boundary_split_near_wall", "install_leading",
                "install_trailing", "readback_leading_dma", "readback_trailing_dma", "upload_leading_dma",
                "upload_trailing_dma", "upload_leading_to_c_gap", "upload_trailing_to_c_gap")


def parse_entry(text: str) -> tuple:
    head, *settings = text.split(";")
    label, _, target = head.partition("=")
    if not label or not target:
        raise argparse.ArgumentTypeError(f"entry {text!r}: expected LABEL=CASE or LABEL=@CHECKOUT:CASE")
    environment = {}
    for setting in settings:
        name, separator, value = setting.partition("=")
        if not separator or not name:
            raise argparse.ArgumentTypeError(f"entry {text!r}: setting {setting!r} is not NAME=VALUE")
        environment[name] = value
    if target.startswith("@"):
        checkout, separator, case = target[1:].rpartition(":")
        if not separator or not checkout:
            raise argparse.ArgumentTypeError(f"entry {text!r}: expected LABEL=@CHECKOUT:CASE")
        return label, pathlib.Path(checkout).resolve(), case, environment
    return label, REPOSITORY_ROOT, target, environment


def parse_log(text: str, warmup: int) -> dict:
    """Mean of every anatomy key per slab over the frames after warmup, and the steady rate."""
    sums: dict = {}
    counts: dict = {}
    for line in text.splitlines():
        match = ANATOMY_PATTERN.match(line.strip())
        if not match or int(match.group(1)) <= warmup:
            continue
        slab = int(match.group(2))
        for item in match.group(3).split():
            key, _, value = item.partition("=")
            try:
                number = float(value)
            except ValueError:
                continue
            sums[(slab, key)] = sums.get((slab, key), 0.0) + number
            counts[(slab, key)] = counts.get((slab, key), 0) + 1
    steady = STEADY_PATTERN.search(text)
    means: dict = {}
    for (slab, key), total in sums.items():
        means.setdefault(str(slab), {})[key] = total / counts[(slab, key)]
    return {"anatomy_means_us": means, "anatomy_frames": max(counts.values()) if counts else 0,
            "steady_fps": float(steady.group(3)) if steady else None,
            "seam_or_conservation_failure": ("FAIL" in text) or ("DROPS=" in text)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device-map", required=True)
    parser.add_argument("--uuid", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--entry", action="append", type=parse_entry, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--no-anatomy", action="store_true",
                        help="report section 23: no bench at all (no timestamps in any recording): the steady rate "
                             "only")
    arguments = parser.parse_args()
    slab_count = len(arguments.device_map.split(","))

    out_directory = (REPOSITORY_ROOT / arguments.out).resolve()
    out_directory.mkdir(parents=True, exist_ok=True)
    base_environment = dict(os.environ)
    base_environment["VK_LOADER_LAYERS_DISABLE"] = "VK_LAYER_KHRONOS_validation"
    base_environment["OPENBLAS_NUM_THREADS"] = "1"
    base_environment["PYTHONIOENCODING"] = "utf-8"

    runs: dict = {label: [] for label, _, _, _ in arguments.entry}
    for round_index in range(arguments.rounds):
        for label, checkout, case, settings in arguments.entry:
            log_path = out_directory / f"{label}_r{round_index}.log"
            command = [sys.executable, "experiment/v7/_run_v7_chain_bench.py", "--case", case,
                       "--weights", ",".join(["1"] * slab_count), "--device-map", arguments.device_map,
                       "--max-steps", str(arguments.max_steps), "--warmup", str(arguments.warmup)]
            if not arguments.no_anatomy:
                command.append("--anatomy")
            environment = dict(base_environment, **settings)
            with LoggedRun(kind="dual_timing", case=case, gpu_uuids=arguments.uuid,
                           command=" ".join(command[1:]), commit=git_commit(checkout)) as run:
                run.extra["label"] = f"{label} round {round_index}"
                run.extra["environment"] = settings
                if checkout != REPOSITORY_ROOT:
                    run.extra["checkout"] = str(checkout).replace("\\", "/")
                with open(log_path, "w", encoding="utf-8", newline="\n") as log:
                    exit_code = subprocess.call(command, cwd=checkout, env=environment, stdout=log,
                                                stderr=subprocess.STDOUT)
                run.extra["exit_code"] = exit_code
                if exit_code != 0:
                    run.extra["status"] = f"failed: exit code {exit_code}"
                    print(f"[dual_timing] {label} round {round_index}: FAILED (exit code {exit_code}), see {log_path}")
                    continue
                parsed = parse_log(log_path.read_text(encoding="utf-8", errors="replace"), arguments.warmup)
                run.extra["steady_fps"] = parsed["steady_fps"]
                run.extra["anatomy_frames"] = parsed["anatomy_frames"]
                runs[label].append({"round": round_index, "run_id": run.run_id, **parsed})
                phase = {slab: {key: round(value) for key, value in keys.items()
                                if key in ("phase_b", "phase_c", "b_to_c_gap")}
                         for slab, keys in parsed["anatomy_means_us"].items()}
                print(f"[dual_timing] {label} round {round_index} ({run.run_id}): steady "
                      f"{parsed['steady_fps']} fps, {phase}", flush=True)

    summary = {"device_map": arguments.device_map, "uuids": arguments.uuid, "max_steps": arguments.max_steps,
               "warmup": arguments.warmup, "rounds": arguments.rounds, "labels": {}}
    lines = []
    for label, checkout, case, settings in arguments.entry:
        entries = runs[label]
        fps_values = [entry["steady_fps"] for entry in entries if entry["steady_fps"] is not None]
        row = {"case": case, "checkout": str(checkout).replace("\\", "/"), "environment": settings,
               "run_ids": [entry["run_id"] for entry in entries],
               "steady_fps": {"median": statistics.median(fps_values), "minimum": min(fps_values),
                              "maximum": max(fps_values)} if fps_values else None,
               "failures": sum(1 for entry in entries if entry["seam_or_conservation_failure"]), "slabs": {}}
        for slab in range(slab_count):
            keys = {}
            for key in SUMMARY_KEYS:
                values = [entry["anatomy_means_us"].get(str(slab), {}).get(key) for entry in entries]
                values = [value for value in values if value is not None]
                if values:
                    keys[key] = {"median_us": statistics.median(values), "minimum_us": min(values),
                                 "maximum_us": max(values)}
            row["slabs"][str(slab)] = keys
        summary["labels"][label] = row
        fps_text = (f"{row['steady_fps']['median']:.1f} ({row['steady_fps']['minimum']:.1f}-"
                    f"{row['steady_fps']['maximum']:.1f})") if row["steady_fps"] else "-"
        for slab in range(slab_count):
            keys = row["slabs"][str(slab)]
            cells = [f"{keys[key]['median_us']:.0f}" if key in keys else "-" for key in SUMMARY_KEYS]
            lines.append(f"| {label} | s{slab} | {fps_text if slab == 0 else ''} | " + " | ".join(cells) + " |")
    (out_directory / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    table = ("Phase and kernel GPU times in microseconds (median over the rounds of the per-run mean of the anatomy "
             f"frames after warmup {arguments.warmup}); steady steps per second of the pipelined run "
             "(median, minimum-maximum).\n\n"
             "| label | slab | steady fps | " + " | ".join(SUMMARY_KEYS) + " |\n"
             "|---|---|---|" + "---|" * len(SUMMARY_KEYS) + "\n" + "\n".join(lines) + "\n")
    (out_directory / "summary.md").write_text(table, encoding="utf-8", newline="\n")
    print("\n" + table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
